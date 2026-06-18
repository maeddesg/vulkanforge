//! Server-side memory subsystem (Stufe A) — SQLiteGraph + fastembed embedded
//! in the API process. The `remember → embed+store → recall → retrieve` loop,
//! project-scoped and persistent across server restarts.
//!
//! Design rationale + footguns are in `results/pre_memory_analyse.md` and
//! `results/recall_scoping_smoke.md`. Key decisions realized here:
//!
//! - **Per-project HNSW index** (Design B): one persistent index per
//!   `project_key`; `recall` only ever searches that index → physical
//!   project isolation, no id→project filter in the hot path.
//! - **node_id carried in the vector metadata** (the achievable form of the
//!   "vector-id == node-id" linchpin): the public `insert_vector` auto-assigns
//!   its own vector id, so we store the graph `node_id` in the vector's
//!   metadata and recover it on recall via `get_vector` → `get_entity`.
//! - **Prefix applied exactly once, here** (`search_document:` / `search_query:`
//!   — fastembed does NOT prepend it for nomic-v1.5).
//! - **Single-writer**: all SG ops serialized behind one `Mutex<SqliteGraph>`;
//!   the embedder (the expensive ~20 ms step) is a separate `Mutex` so it is
//!   never held together with the graph lock. Memory NEVER touches the GPU
//!   permit (handlers run it via `spawn_blocking`, off the inference path).
//! - **Cypher avoided** for find/list (its subset silently returns 0 on inline
//!   `{prop:val}`, `count()`, etc.): node lookups go through raw SQL on the
//!   public connection pool (`graph_entities(id,kind,name,data)`).
//!
//! Stufe A writes and reads only — no delete/archive/lifecycle (Stufe B+).

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::Serialize;
use serde_json::{json, Value};
use sqlitegraph::graph::{GraphEdge, GraphEntity, SqliteGraph};
use sqlitegraph::hnsw::{DistanceMetric, HnswConfig, HnswConfigBuilder};

/// Reserved project key used when a request omits `project_key`. Goes through
/// the exact same get-or-create path as a named project (no special case).
pub const GLOBAL_SCOPE: &str = "__global__";
const EMBED_DIM: usize = 768;
const PROJECT_KIND: &str = "Project";

/// Dedup-on-remember threshold (cosine **similarity** = `1 − distance`). A new
/// note whose nearest existing active note is at least this similar is treated
/// as a near-duplicate and NOT stored again. Deliberately conservative
/// (near-identical only) — better to let a borderline duplicate through than to
/// silently swallow a genuinely distinct note. Iterative; tune from real use
/// (cf. the B-2 trigger-frequency note).
const DEDUP_SIMILARITY: f32 = 0.92;

/// Extra candidates fetched beyond `k` on the **backfill** pass, when SUPERSEDES
/// suppression has emptied top-`k` slots and the result fell short. Generous so
/// a handful of superseded notes near the top still backfill to `k`; if more
/// than this are superseded the store is effectively exhausted of fresh notes
/// near the top, and recall honestly returns fewer (never more than available).
/// The whole pass is gated on suppression being active — no edges → never runs.
const SUPERSEDE_BACKFILL_EXTRA: usize = 20;

/// One recall result (vector hit enriched with its graph node).
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: i64,
    pub kind: String,
    pub name: String,
    pub text: String,
    pub status: String,
    /// Layer type (Schicht-Enabler). Always present; defaults to `"untyped"`
    /// for notes stored before typing existed (read-time default — no backfill).
    /// Serialised as `type`. Distinct from `kind` (the free-form remember
    /// category); `type` is the constrained layer enum ([`NoteType`]).
    #[serde(rename = "type")]
    pub note_type: String,
    /// If set, the id of a note that `SUPERSEDES` this one (this note is stale
    /// and is suppressed from default recall). `None`/omitted for current
    /// notes — so a non-superseded hit's JSON is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<i64>,
    /// Ids this note `DERIVES_FROM` (its justification). **Populated only on the
    /// `--explain` path** — empty/omitted in default recall, so default recall
    /// is byte-identical even with `DERIVES_FROM` edges (awareness ≠ injection).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub derives_from: Vec<i64>,
    /// Cosine **similarity** in `[0,1]` = `1 - cosine_distance` (the HNSW
    /// index returns distance; we present similarity, higher = closer).
    pub score: f32,
}

/// Layer type of a note (Schicht-Enabler). An explicit, non-embedding signal
/// set by the **user** (`/remember --type` / `/retype`) — never guessed by the
/// embedder or the agent. Old/untyped notes default to [`NoteType::Untyped`]
/// (no backfill — the default applies on read). Stored as a lowercase string
/// in the node's `data["type"]`. Extensible: per-layer *behaviour* (decay,
/// surfacing, pinning) is deliberately NOT here — this is just the label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoteType {
    Invariant,
    Working,
    Episodic,
    Decision,
    Failure,
    Untyped,
}

impl NoteType {
    pub fn as_str(self) -> &'static str {
        match self {
            NoteType::Invariant => "invariant",
            NoteType::Working => "working",
            NoteType::Episodic => "episodic",
            NoteType::Decision => "decision",
            NoteType::Failure => "failure",
            NoteType::Untyped => "untyped",
        }
    }

    /// Parse a user-supplied type. Case-insensitive. Unknown → `Err` (the
    /// caller surfaces a usage error — we never silently coerce a typo to a
    /// real layer).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "invariant" => Ok(NoteType::Invariant),
            "working" => Ok(NoteType::Working),
            "episodic" => Ok(NoteType::Episodic),
            "decision" => Ok(NoteType::Decision),
            "failure" => Ok(NoteType::Failure),
            "untyped" => Ok(NoteType::Untyped),
            other => Err(format!(
                "unknown note type {other:?} (expected one of: invariant, working, episodic, decision, failure, untyped)"
            )),
        }
    }
}

/// Why a candidate was cut from `returned`: wrong type for an active type
/// filter, scored below the adaptive relevance `threshold`, or survived both
/// but fell beyond the `top_k` cap. Surfaced per near-miss so `recall
/// --explain` can show the effect of each gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CutReason {
    /// Superseded — the note is the target of a `SUPERSEDES` edge (something
    /// replaced it). The superseder id is in the hit's `superseded_by`.
    #[serde(rename = "superseded")]
    Superseded,
    /// Note's type didn't match the active type filter.
    #[serde(rename = "type")]
    Type,
    /// Score below the adaptive threshold (`top_score − margin`).
    #[serde(rename = "threshold")]
    Threshold,
    /// Above the threshold but beyond the `top_k` cap.
    #[serde(rename = "top-k")]
    TopK,
}

/// Typed relationships between notes (the connection axis). `SUPERSEDES`
/// suppresses recall (stale → out); `DERIVES_FROM` is the Why-Graph (a note is
/// anchored in the notes that justify it) and **never** changes recall — it is
/// additive awareness only. Extensible (`CONTRADICTS` later). User-created;
/// the agent never makes edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Supersedes,
    DerivesFrom,
}

impl EdgeType {
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeType::Supersedes => "SUPERSEDES",
            EdgeType::DerivesFrom => "DERIVES_FROM",
        }
    }
}

/// One node of a `/why` Why-Graph trace: a note plus the notes it derives from
/// (its justification), recursively. `cycle` marks a node already on the path
/// (the user can build `A from B` + `B from A`); `truncated` marks the
/// depth-cap boundary. Both halt the traversal so it always terminates.
#[derive(Debug, Clone, Serialize)]
pub struct WhyNode {
    pub id: i64,
    pub kind: String,
    #[serde(rename = "type")]
    pub note_type: String,
    pub name: String,
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub derives_from: Vec<WhyNode>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cycle: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// A candidate that didn't make `returned`, with the gate that cut it.
#[derive(Debug, Clone, Serialize)]
pub struct NearMiss {
    #[serde(flatten)]
    pub hit: Hit,
    pub cut: CutReason,
}

/// Diagnostic view of a recall (built only for `recall --explain`; the default
/// recall path is unchanged). Carries the `returned` notes plus `near_miss` —
/// the candidates that fell outside, each labelled with the gate that cut it
/// (`threshold` vs `top-k`) — so a caller can see what recall *almost*
/// returned and exactly where the lines were drawn.
#[derive(Debug, Clone, Serialize)]
pub struct RecallExplain {
    /// The notes recall returns (the top-k that also clear the threshold).
    pub returned: Vec<Hit>,
    /// Candidates that were cut (highest score first), each with its reason.
    pub near_miss: Vec<NearMiss>,
    /// The hard `top_k` cap (the requested `k`, clamped).
    pub top_k: usize,
    /// The adaptive relevance cutoff (`top_score − margin`) when a margin is
    /// active, else `None` (pure top-k — no score threshold).
    pub threshold: Option<f32>,
    /// Embedded-query dimension (768 for Nomic-Embed-Text-v1.5).
    pub query_dim: usize,
    /// Score gap between the last `returned` hit and the first `near_miss`
    /// (similarity units). Larger = cleaner separation at the cut.
    /// `None` when either side is empty.
    pub separation: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectInfo {
    pub id: i64,
    pub project_key: String,
}

/// Result of [`MemoryStore::remember`]. `deduped` = the note was a
/// near-duplicate of an existing active note, so `id` is that **existing**
/// node and nothing new was stored (surfaced to the caller, never silent).
#[derive(Debug, Clone, Serialize)]
pub struct RememberOutcome {
    pub id: i64,
    pub deduped: bool,
}

/// Error from an id-targeting curation op (`archive`/`unarchive`/`delete`).
/// Lets the handler map a **client** error (the id doesn't exist → 404 Not
/// Found) apart from a real **server** fault (lock/DB/IO/embedder → 500)
/// WITHOUT fragile string-matching on the message. Other store methods keep a
/// flat `String` error (always a 500); only the curation ops need this split.
#[derive(Debug)]
pub enum CurateError {
    /// No note with this id in the (optional) scope — a client error (404).
    NotFound(i64),
    /// Any other failure (lock poisoned, DB/IO, embedder, index op) — 500.
    Internal(String),
}

impl std::fmt::Display for CurateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurateError::NotFound(id) => write!(f, "memory: note {id} not found"),
            CurateError::Internal(e) => write!(f, "{e}"),
        }
    }
}

/// Any `String` store error folds into `Internal`, so `?` keeps working inside
/// the curation methods (their helpers — `reindex_without`, `embed`,
/// `ensure_index`, `update_entity` — all return `Result<_, String>`). Only an
/// explicit `CurateError::NotFound(id)` becomes a 404.
impl From<String> for CurateError {
    fn from(e: String) -> Self {
        CurateError::Internal(e)
    }
}

/// The embedded memory store: one shared `sg.db` (nodes + edges + per-project
/// HNSW indexes) plus the Nomic embedder, both owned by the API process.
pub struct MemoryStore {
    embedder: Mutex<TextEmbedding>,
    graph: Mutex<SqliteGraph>,
    db_path: PathBuf,
}

fn hnsw_cfg() -> HnswConfig {
    // 768/cosine/m16/ef200 — the concept's parameters (results/recall_scoping_smoke.md).
    HnswConfigBuilder::new()
        .dimension(EMBED_DIM)
        .m_connections(16)
        .ef_construction(200)
        .distance_metric(DistanceMetric::Cosine)
        .build()
        .expect("static HNSW config is valid")
}

fn now_iso() -> String {
    // Cheap UTC-ish timestamp without a chrono dep: seconds since epoch.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

impl MemoryStore {
    /// Open (or create) the store at `db_path` and **eager-load** the embedder
    /// so the first request never pays the model download/init. The first ever
    /// start may fetch the ONNX model from HuggingFace into the cache dir
    /// (a sibling `embed-cache/` of the db); subsequent starts are offline.
    pub fn new(db_path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("memory: create db dir {}: {e}", parent.display()))?;
        }
        let cache_dir = db_path
            .parent()
            .map(|p| p.join("embed-cache"))
            .unwrap_or_else(|| PathBuf::from("embed-cache"));

        eprintln!("VulkanForge: memory — loading embedder (NomicEmbedTextV15Q, 768-dim, INT8/VNNI)…");
        let embedder = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15Q)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )
        .map_err(|e| format!("memory: embedder init: {e}"))?;

        let graph = SqliteGraph::open(&db_path)
            .map_err(|e| format!("memory: open {}: {e}", db_path.display()))?;

        eprintln!("VulkanForge: memory — store ready at {}", db_path.display());
        Ok(Self {
            embedder: Mutex::new(embedder),
            graph: Mutex::new(graph),
            db_path,
        })
    }

    fn scope(project_key: Option<&str>) -> String {
        match project_key {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => GLOBAL_SCOPE.to_string(),
        }
    }

    /// Embed with the model's required task prefix. fastembed does NOT prepend
    /// it for nomic-v1.5 — this is the single place we do.
    fn embed(&self, prefixed: String) -> Result<Vec<f32>, String> {
        let mut emb = self.embedder.lock().map_err(|_| "memory: embedder lock poisoned")?;
        let mut out = emb
            .embed(vec![prefixed], None)
            .map_err(|e| format!("memory: embed: {e}"))?;
        out.pop().ok_or_else(|| "memory: embed produced no vector".to_string())
    }

    /// Find the node id of a `kind`/`name` pair via raw SQL on the pool
    /// (`graph_entities`), or `None`. Cypher is avoided on purpose.
    fn find_node(graph: &SqliteGraph, kind: &str, name: &str) -> Option<i64> {
        let conn = graph.pool.get().ok()?;
        conn.query_row(
            "SELECT id FROM graph_entities WHERE kind = ?1 AND name = ?2 LIMIT 1",
            (kind, name),
            |r| r.get::<_, i64>(0),
        )
        .ok()
    }

    /// Get-or-create the Project structure node for `key`; returns its node id.
    fn ensure_project_node(graph: &SqliteGraph, key: &str) -> Result<i64, String> {
        if let Some(id) = Self::find_node(graph, PROJECT_KIND, key) {
            return Ok(id);
        }
        graph
            .insert_entity(&GraphEntity {
                id: 0,
                kind: PROJECT_KIND.to_string(),
                name: key.to_string(),
                file_path: None,
                data: json!({ "created_at": now_iso() }),
            })
            .map_err(|e| format!("memory: create project node: {e}"))
    }

    /// Get-or-create the per-project persistent HNSW index. `hnsw_index_persistent`
    /// errors on a duplicate name, so check existence first; the existence-check
    /// guard is dropped before re-locking (parking_lot is non-reentrant).
    fn ensure_index(graph: &SqliteGraph, key: &str) -> Result<(), String> {
        let exists = graph
            .get_hnsw_index(key)
            .map_err(|e| format!("memory: probe index {key}: {e}"))?
            .is_some(); // MutexGuard dropped at the end of this statement
        if !exists {
            // `let _ =` drops the returned MutexGuard immediately (releasing the
            // index-map lock) and acknowledges the `#[must_use]` guard.
            let _ = graph
                .hnsw_index_persistent(key, hnsw_cfg())
                .map_err(|e| format!("memory: create index {key}: {e}"))?;
        }
        Ok(())
    }

    /// Store a nugget: embed → (dedup check) → graph node → CONTAINS edge →
    /// vector (carrying `node_id` in its metadata). Returns the node id and
    /// whether it was deduped against an existing near-identical active note
    /// (then no new node was created).
    pub fn remember(
        &self,
        project_key: Option<&str>,
        kind: &str,
        text: &str,
        name: Option<&str>,
        metadata: Option<Value>,
    ) -> Result<RememberOutcome, String> {
        if kind.trim().is_empty() {
            return Err("memory: remember requires a non-empty `kind`".into());
        }
        if text.trim().is_empty() {
            return Err("memory: remember requires non-empty `text`".into());
        }
        let key = Self::scope(project_key);
        // Embed FIRST, outside the graph lock (the expensive ~20 ms step). The
        // SAME embedding is reused for the dedup search AND the insert below —
        // never embed twice.
        let emb = self.embed(format!("search_document: {text}"))?;
        if emb.len() != EMBED_DIM {
            return Err(format!("memory: embedder returned {} dims, expected {EMBED_DIM}", emb.len()));
        }

        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;

        // Dedup-on-remember: if the project index already holds a near-identical
        // note, return it instead of storing a duplicate. The index contains
        // only ACTIVE vectors (archive + delete both remove the vector), so a
        // hit is necessarily an active note. Conservative threshold; the
        // `deduped` flag surfaces this to the caller (never silently swallowed).
        if let Some((existing, sim)) = Self::nearest_similarity(&graph, &key, &emb)? {
            if sim >= DEDUP_SIMILARITY {
                // Light "reinforce" touch (bump updated_at); NOT the B-3b decay.
                Self::touch_updated_at(&graph, existing);
                return Ok(RememberOutcome { id: existing, deduped: true });
            }
        }

        // Content node. data carries the searchable text + lifecycle stub fields.
        let label = name.map(|s| s.to_string()).unwrap_or_else(|| {
            let t = text.trim();
            t.chars().take(48).collect::<String>()
        });
        let mut data = json!({
            "text": text,
            "status": "active",
            "pinned": false,
            "created_at": now_iso(),
        });
        if let Some(Value::Object(extra)) = metadata {
            if let Value::Object(base) = &mut data {
                for (k, v) in extra {
                    base.entry(k).or_insert(v);
                }
            }
        }
        let node_id = graph
            .insert_entity(&GraphEntity {
                id: 0,
                kind: kind.to_string(),
                name: label,
                file_path: None,
                data,
            })
            .map_err(|e| format!("memory: insert content node: {e}"))?;

        // Project structure node + CONTAINS edge (graph backbone for future enrich).
        let project_id = Self::ensure_project_node(&graph, &key)?;
        graph
            .insert_edge(&GraphEdge {
                id: 0,
                from_id: project_id,
                to_id: node_id,
                edge_type: "CONTAINS".to_string(),
                data: json!({}),
            })
            .map_err(|e| format!("memory: insert CONTAINS edge: {e}"))?;

        // Per-project vector index: add the embedding, metadata = node_id link.
        Self::ensure_index(&graph, &key)?;
        graph
            .get_hnsw_index_mut(&key, |idx| {
                let _vid = idx.insert_vector(&emb, Some(json!({ "node_id": node_id })))?;
                // Persist immediately so a restart restores it without re-embed
                // (3.2.4 incremental persist makes this cheap).
                idx.persist_topology()
            })
            .map_err(|e| format!("memory: index op {key}: {e}"))?
            .map_err(|e| format!("memory: vector add/persist {key}: {e}"))?;

        Ok(RememberOutcome { id: node_id, deduped: false })
    }

    /// Recall: embed query → search the project's index → enrich each hit's
    /// node. Missing index (nothing remembered yet) → empty, not an error.
    pub fn recall(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: usize,
    ) -> Result<Vec<Hit>, String> {
        self.recall_with_margin(project_key, query, k, None)
    }

    /// recall with an optional **adaptive relevance threshold**. `margin = None`
    /// is pure top-k — **byte-identical to [`recall`]** (today's behavior, and
    /// the disable path). `margin = Some(m)` keeps only notes scoring within `m`
    /// (cosine-similarity) of the top hit (`score ≥ top_score − m`); `top_k`
    /// stays the hard cap. The threshold is **relative-to-top** (adaptive: it
    /// scales with each query's best match — a fixed absolute cutoff is brittle
    /// because the cosine scale isn't calibrated across queries) and errs to
    /// **include** (a larger `m` keeps more — a dropped relevant note is worse
    /// than kept noise). This is a filter on the *existing* ranked set: no new
    /// scoring, no re-ranking.
    pub fn recall_with_margin(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: usize,
        margin: Option<f32>,
    ) -> Result<Vec<Hit>, String> {
        self.recall_filtered(project_key, query, k, margin, None, false)
    }

    /// recall with an optional adaptive relevance threshold **and** an optional
    /// **type filter** (Schicht-Enabler). `type_filter = Some(t)` keeps only
    /// notes of layer type `t` — an explicit, non-embedding signal that
    /// disambiguates *reliably* where similarity can't (adjacent-domain notes
    /// share a score band but differ in type). Both gates are filters on the
    /// existing ranked candidate set (`top_k` stays the hard cap); no new
    /// scoring, no re-ranking. `(None, None)` is byte-identical to plain
    /// [`recall`].
    pub fn recall_filtered(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: usize,
        margin: Option<f32>,
        type_filter: Option<NoteType>,
        include_superseded: bool,
    ) -> Result<Vec<Hit>, String> {
        // Gate chain (same order as `recall_explain`: superseded → type →
        // threshold), capped at `k`.
        let apply = |hits: Vec<Hit>, threshold: Option<f32>| -> Vec<Hit> {
            hits.into_iter()
                .filter(|h| include_superseded || h.superseded_by.is_none())
                .filter(|h| type_filter.is_none_or(|t| h.note_type == t.as_str()))
                .filter(|h| threshold.is_none_or(|t| h.score >= t))
                .take(k)
                .collect()
        };

        // First pass at exactly `k` — today's path. `recall_inner`'s own
        // fast-skip means **no extra read** when the store has no SUPERSEDES
        // edge, so this is byte-identical without edges.
        let (hits, _dim, threshold) = self.recall_inner(project_key, query, k, margin)?;
        let superseded_dropped = !include_superseded && hits.iter().any(|h| h.superseded_by.is_some());
        let result = apply(hits, threshold);

        // Backfill only when SUPERSEDES suppression actually removed top-k slots
        // AND that left us short of `k` — pull a deeper candidate pool and
        // re-run the gates (same order), capped at `k`. Type/threshold-only
        // narrowing (no superseded drop) keeps its intended under-return.
        if superseded_dropped && result.len() < k {
            let width = (k + SUPERSEDE_BACKFILL_EXTRA).min(100);
            let (wide, _dim, wthr) = self.recall_inner(project_key, query, width, margin)?;
            let backfilled = apply(wide, wthr);
            // Never promise more than the store holds; the deeper pass is a
            // superset of the first, so it's ≥ the short result.
            if backfilled.len() > result.len() {
                return Ok(backfilled);
            }
        }
        Ok(result)
    }

    /// `recall --explain`: search **wider** (`top_k + near_miss_extra`,
    /// clamped), then split the ranked candidates into `returned` (the top-k
    /// that also clear the threshold) and `near_miss` (everything cut), each
    /// near-miss labelled with the gate that cut it. Pure measurement: same
    /// embedding, same ranking, same score, same `margin` the live path would
    /// apply — it just exposes the candidates the default path discards.
    pub fn recall_explain(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: usize,
        near_miss_extra: usize,
        margin: Option<f32>,
        type_filter: Option<NoteType>,
        include_superseded: bool,
    ) -> Result<RecallExplain, String> {
        let k = k.clamp(1, 100);
        let width = (k + near_miss_extra).clamp(1, 100);
        let (all, query_dim, threshold) = self.recall_inner(project_key, query, width, margin)?;
        let mut returned = Vec::new();
        let mut near_miss = Vec::new();
        for (rank, hit) in all.into_iter().enumerate() {
            // Gate priority mirrors `recall_filtered`: superseded (stale) first,
            // then wrong type, then the threshold, then the top-k cap.
            let superseded = !include_superseded && hit.superseded_by.is_some();
            let wrong_type = type_filter.is_some_and(|t| hit.note_type != t.as_str());
            let below_threshold = threshold.is_some_and(|t| hit.score < t);
            if superseded {
                near_miss.push(NearMiss { hit, cut: CutReason::Superseded });
            } else if wrong_type {
                near_miss.push(NearMiss { hit, cut: CutReason::Type });
            } else if below_threshold {
                near_miss.push(NearMiss { hit, cut: CutReason::Threshold });
            } else if rank >= k {
                near_miss.push(NearMiss { hit, cut: CutReason::TopK });
            } else {
                returned.push(hit);
            }
        }
        let separation = match (returned.last(), near_miss.first()) {
            (Some(last), Some(first)) => Some(last.score - first.hit.score),
            _ => None,
        };
        // Why-Graph awareness (explain-only): annotate each shown note with the
        // ids it DERIVES_FROM. Deliberately NOT done in `recall_inner`/
        // `recall_filtered`, so default recall stays byte-identical even with
        // DERIVES_FROM edges (awareness ≠ injection; recall never changes).
        if let Ok(graph) = self.graph.lock() {
            if let Ok(conn) = graph.pool.get() {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT to_id FROM graph_edges WHERE from_id=?1 AND edge_type='DERIVES_FROM' ORDER BY to_id",
                ) {
                    let mut fill = |h: &mut Hit| {
                        if let Ok(rows) = stmt.query_map([h.id], |r| r.get::<_, i64>(0)) {
                            h.derives_from = rows.filter_map(|r| r.ok()).collect();
                        }
                    };
                    for h in &mut returned {
                        fill(h);
                    }
                    for nm in &mut near_miss {
                        fill(&mut nm.hit);
                    }
                }
            }
        }
        Ok(RecallExplain { returned, near_miss, top_k: k, threshold, query_dim, separation })
    }

    /// Shared recall core: embed the query, search the project index for the
    /// top `n`, enrich each hit's graph node, and compute the adaptive
    /// `threshold` (`top_score − margin`, or `None` when `margin` is `None` or
    /// the result is empty). Returns `(hits_unfiltered, query_dim, threshold)`;
    /// the caller decides whether to filter (`recall_with_margin`) or label
    /// (`recall_explain`), so both paths share one source of truth.
    fn recall_inner(
        &self,
        project_key: Option<&str>,
        query: &str,
        n: usize,
        margin: Option<f32>,
    ) -> Result<(Vec<Hit>, usize, Option<f32>), String> {
        if query.trim().is_empty() {
            return Err("memory: recall requires a non-empty `query`".into());
        }
        let key = Self::scope(project_key);
        let k = n.clamp(1, 100);
        let emb = self.embed(format!("search_query: {query}"))?;
        let query_dim = emb.len();

        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;

        // search inside the index lock; return (node_id, distance) pairs.
        // Closure returns Result<_, String> so we never name HnswError.
        let pairs: Vec<(i64, f32)> = match graph.get_hnsw_index_ref(&key, |idx| {
            match idx.search(&emb, k) {
                Err(e) => Err(format!("search: {e}")),
                Ok(hits) => Ok(hits
                    .into_iter()
                    .filter_map(|(vid, dist)| {
                        idx.get_vector(vid)
                            .ok()
                            .flatten()
                            .and_then(|(_v, meta)| meta.get("node_id").and_then(|x| x.as_i64()))
                            .map(|nid| (nid, dist))
                    })
                    .collect::<Vec<(i64, f32)>>()),
            }
        }) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(format!("memory: {e} ({key})")),
            Err(_index_missing) => return Ok((Vec::new(), query_dim, None)),
        };

        let mut hits = Vec::with_capacity(pairs.len());
        for (nid, dist) in pairs {
            if let Ok(node) = graph.get_entity(nid) {
                let text = node.data.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let status = node.data.get("status").and_then(|v| v.as_str()).unwrap_or("active").to_string();
                // Read-time default: notes stored before typing have no `type`
                // key → `untyped` (no backfill needed).
                let note_type = node.data.get("type").and_then(|v| v.as_str()).unwrap_or("untyped").to_string();
                hits.push(Hit {
                    id: node.id,
                    kind: node.kind,
                    name: node.name,
                    text,
                    status,
                    note_type,
                    superseded_by: None,
                    derives_from: Vec::new(),
                    score: 1.0 - dist,
                });
            }
        }
        // Supersession: a note that is the `to_id` of a SUPERSEDES edge is
        // stale (something replaced it). Index-backed (`idx_edges_to` +
        // `idx_edges_type`). Fast-skip the whole pass when the store holds no
        // SUPERSEDES edge at all → no edges = byte-identical recall.
        if let Ok(conn) = graph.pool.get() {
            let any: bool = conn
                .query_row("SELECT EXISTS(SELECT 1 FROM graph_edges WHERE edge_type='SUPERSEDES')", [], |r| r.get(0))
                .unwrap_or(false);
            if any {
                if let Ok(mut stmt) = conn.prepare_cached(
                    "SELECT from_id FROM graph_edges WHERE to_id=?1 AND edge_type='SUPERSEDES' LIMIT 1",
                ) {
                    for h in &mut hits {
                        h.superseded_by = stmt.query_row([h.id], |r| r.get::<_, i64>(0)).ok();
                    }
                }
            }
        }
        // Adaptive threshold: relative to the top hit's score (hits are
        // nearest-first, so `hits[0]` is the best match). `None` when no margin
        // is configured or the result is empty → no cutoff (pure top-k).
        let threshold = match (margin, hits.first()) {
            (Some(m), Some(top)) => Some(top.score - m),
            _ => None,
        };
        Ok((hits, query_dim, threshold))
    }

    // ---- Curation (Stufe B-3): dedup helpers + archive + delete ----

    /// Nearest existing note's `(node_id, cosine similarity)` for `emb` in the
    /// project index, or `None` if the index doesn't exist yet. Used for
    /// dedup-on-remember; the index holds only **active** vectors (archive and
    /// delete both remove the vector), so a hit is necessarily an active note.
    fn nearest_similarity(
        graph: &SqliteGraph,
        key: &str,
        emb: &[f32],
    ) -> Result<Option<(i64, f32)>, String> {
        let res = graph.get_hnsw_index_ref(key, |idx| match idx.search(emb, 1) {
            Err(e) => Err(format!("dedup search: {e}")),
            Ok(hits) => Ok(hits.into_iter().next().and_then(|(vid, dist)| {
                idx.get_vector(vid)
                    .ok()
                    .flatten()
                    .and_then(|(_v, meta)| meta.get("node_id").and_then(|x| x.as_i64()))
                    .map(|nid| (nid, 1.0 - dist))
            })),
        });
        match res {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(format!("memory: {e} ({key})")),
            Err(_index_missing) => Ok(None),
        }
    }

    /// Rebuild the project index from its surviving vectors, **dropping** the
    /// one belonging to `exclude_node_id`. The curation removal primitive
    /// (used by both archive and delete).
    ///
    /// **Why a rebuild, not `delete_vector`:** SQLiteGraph's `delete_vector`
    /// does **not** re-elect an HNSW entry point when it removes the current
    /// one, so deleting the entry-point vector leaves a non-empty index that
    /// `search` rejects with "Index not initialized" (verified against rev
    /// `d8219a8` — a gap the earlier scoping smoke didn't exercise; SG is used
    /// as-is, the workaround lives here). Reading the survivors' **stored**
    /// embeddings (`get_vector`, no re-embed), dropping the index
    /// (`delete_index` CASCADE-clears its `hnsw_vectors`), recreating it and
    /// re-inserting re-elects a valid entry point. O(N) per curation op — but
    /// curation is rare and user-driven.
    fn reindex_without(graph: &SqliteGraph, key: &str, exclude_node_id: i64) -> Result<(), String> {
        let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
        // The index id (also used for a belt-and-suspenders orphan sweep).
        // No index yet → nothing to remove.
        let index_id: i64 = match conn
            .query_row("SELECT id FROM hnsw_indexes WHERE name = ?1", (key,), |r| r.get::<_, i64>(0))
            .ok()
        {
            Some(id) => id,
            None => return Ok(()),
        };
        let vids: Vec<i64> = {
            let mut stmt = conn
                .prepare("SELECT id FROM hnsw_vectors WHERE index_id = ?1 ORDER BY id")
                .map_err(|e| format!("memory: reindex list prepare: {e}"))?;
            let rows = stmt
                .query_map((index_id,), |r| r.get::<_, i64>(0))
                .map_err(|e| format!("memory: reindex list query: {e}"))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| format!("memory: reindex list row: {e}"))?);
            }
            out
        };
        drop(conn); // release the pooled connection before re-locking the index map

        // Read survivors' stored embeddings. `get_vector` is a storage lookup
        // (no entry point needed), so it works even on a headless index.
        let mut survivors: Vec<(Vec<f32>, i64)> = Vec::with_capacity(vids.len());
        graph
            .get_hnsw_index_ref(key, |idx| {
                for vid in &vids {
                    if let Ok(Some((vec, meta))) = idx.get_vector(*vid as u64) {
                        if let Some(nid) = meta.get("node_id").and_then(|x| x.as_i64()) {
                            if nid != exclude_node_id {
                                survivors.push((vec, nid));
                            }
                        }
                    }
                }
            })
            .map_err(|e| format!("memory: reindex read {key}: {e}"))?;

        // Drop the index (CASCADE clears its vectors) + an explicit orphan sweep
        // (in case the FK cascade is off), then recreate it empty.
        graph
            .delete_hnsw_index(key)
            .map_err(|e| format!("memory: reindex drop {key}: {e}"))?;
        if let Ok(conn) = graph.pool.get() {
            let _ = conn.execute("DELETE FROM hnsw_vectors WHERE index_id = ?1", (index_id,));
        }
        let _ = graph
            .hnsw_index_persistent(key, hnsw_cfg())
            .map_err(|e| format!("memory: reindex recreate {key}: {e}"))?;

        // Re-insert survivors → re-elects a valid entry point.
        graph
            .get_hnsw_index_mut(key, |idx| -> Result<(), String> {
                for (vec, nid) in &survivors {
                    idx.insert_vector(vec, Some(json!({ "node_id": nid })))
                        .map_err(|e| format!("reinsert: {e}"))?;
                }
                idx.persist_topology().map_err(|e| format!("persist: {e}"))
            })
            .map_err(|e| format!("memory: reindex index op {key}: {e}"))?
            .map_err(|e| format!("memory: reindex reinsert {key}: {e}"))?;
        Ok(())
    }

    /// Raw `COUNT(*)` of vectors in the project index — the **honest** capacity
    /// signal. `statistics().vector_count` is stale-on-reopen
    /// (`results/recall_scoping_smoke.md`), so correctness/capacity must use
    /// this raw count (or recall behavior), **never** the cached counter.
    pub fn raw_vector_count(&self, project_key: Option<&str>) -> Result<i64, String> {
        let key = Self::scope(project_key);
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;
        let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
        conn.query_row(
            "SELECT COUNT(*) FROM hnsw_vectors v \
             JOIN hnsw_indexes i ON v.index_id = i.id WHERE i.name = ?1",
            (key.as_str(),),
            |r| r.get::<_, i64>(0),
        )
        .map_err(|e| format!("memory: raw vector count {key}: {e}"))
    }

    /// Light "reinforce" touch on a dedup hit: bump `updated_at`. NOT the full
    /// touch-on-read / staleness decay (that is B-3b). Best-effort — a failure
    /// here must never fail the remember.
    fn touch_updated_at(graph: &SqliteGraph, id: i64) {
        if let Ok(mut node) = graph.get_entity(id) {
            if let Value::Object(map) = &mut node.data {
                map.insert("updated_at".to_string(), json!(now_iso()));
            }
            let _ = graph.update_entity(&node);
        }
    }

    /// Archive a note: remove its vector from the project index (so it no longer
    /// surfaces in recall) but **keep the node** in the graph with
    /// `status="archived"` — a record. The node + text remain, so this is
    /// reversible: see [`unarchive`](Self::unarchive).
    pub fn archive(&self, project_key: Option<&str>, id: i64) -> Result<(), CurateError> {
        let key = Self::scope(project_key);
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        let mut node = graph
            .get_entity(id)
            .map_err(|_| CurateError::NotFound(id))?;
        // 1) out of recall: rebuild the index without this node's vector.
        Self::reindex_without(&graph, &key, id)?;
        // 2) mark the node archived (record stays in the graph).
        if let Value::Object(map) = &mut node.data {
            map.insert("status".to_string(), json!("archived"));
            map.insert("archived_at".to_string(), json!(now_iso()));
        }
        graph
            .update_entity(&node)
            .map_err(|e| format!("memory: archive update {id}: {e}"))?;
        Ok(())
    }

    /// Retype a note: set its layer `type` (Schicht-Enabler). **Pure metadata**
    /// — the embedding/vector and the recall ranking are untouched (no
    /// reindex), so it's reversible by retyping again. User curation, like
    /// `archive`/`unarchive` (no confirm needed — it only relabels). `NotFound`
    /// if the id doesn't exist. `project_key` is accepted for curation-API
    /// symmetry but unused: the note is addressed by its global node id.
    pub fn retype(
        &self,
        project_key: Option<&str>,
        id: i64,
        note_type: NoteType,
    ) -> Result<(), CurateError> {
        let _ = project_key;
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        let mut node = graph.get_entity(id).map_err(|_| CurateError::NotFound(id))?;
        if let Value::Object(map) = &mut node.data {
            map.insert("type".to_string(), json!(note_type.as_str()));
        }
        graph
            .update_entity(&node)
            .map_err(|e| format!("memory: retype update {id}: {e}"))?;
        Ok(())
    }

    /// Record that note `new_id` **supersedes** `old_id` — a typed `SUPERSEDES`
    /// edge `new → old`. `old_id` then becomes stale: default recall suppresses
    /// any note that is the *target* of a `SUPERSEDES` edge (so chains resolve
    /// naturally — every superseded link drops, only the un-superseded head
    /// survives). **Reversible** (the note is not deleted; see
    /// [`unsupersede`](Self::unsupersede)). User curation; the agent never
    /// creates edges. Idempotent (a duplicate link is a no-op). `NotFound` if
    /// either id is missing.
    pub fn supersede(
        &self,
        _project_key: Option<&str>,
        new_id: i64,
        old_id: i64,
    ) -> Result<(), CurateError> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        graph.get_entity(new_id).map_err(|_| CurateError::NotFound(new_id))?;
        graph.get_entity(old_id).map_err(|_| CurateError::NotFound(old_id))?;
        // Idempotent: skip if the link already exists.
        let exists = {
            let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
            conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM graph_edges WHERE from_id=?1 AND to_id=?2 AND edge_type='SUPERSEDES')",
                (new_id, old_id),
                |r| r.get::<_, bool>(0),
            )
            .map_err(|e| format!("memory: supersede exists check: {e}"))?
        };
        if !exists {
            graph
                .insert_edge(&GraphEdge {
                    id: 0,
                    from_id: new_id,
                    to_id: old_id,
                    edge_type: EdgeType::Supersedes.as_str().to_string(),
                    data: json!({ "created_at": now_iso() }),
                })
                .map_err(|e| format!("memory: supersede insert {new_id}->{old_id}: {e}"))?;
        }
        Ok(())
    }

    /// Release a supersession (`new_id SUPERSEDES old_id`) — `old_id` returns to
    /// recall. Inverse of [`supersede`](Self::supersede). Deletes the matching
    /// edge(s) via the SG API (keeps the adjacency caches consistent). Idempotent
    /// (no such edge → no-op). `NotFound` if either id is missing.
    pub fn unsupersede(
        &self,
        _project_key: Option<&str>,
        new_id: i64,
        old_id: i64,
    ) -> Result<(), CurateError> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        graph.get_entity(new_id).map_err(|_| CurateError::NotFound(new_id))?;
        graph.get_entity(old_id).map_err(|_| CurateError::NotFound(old_id))?;
        // Find the matching edge id(s) (raw read), then delete via the API so
        // the cache invalidation runs (a raw DELETE would desync the caches).
        let edge_ids: Vec<i64> = {
            let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
            let mut stmt = conn
                .prepare("SELECT id FROM graph_edges WHERE from_id=?1 AND to_id=?2 AND edge_type='SUPERSEDES'")
                .map_err(|e| format!("memory: unsupersede prepare: {e}"))?;
            let rows = stmt
                .query_map((new_id, old_id), |r| r.get::<_, i64>(0))
                .map_err(|e| format!("memory: unsupersede query: {e}"))?;
            rows.filter_map(|r| r.ok()).collect()
        };
        for eid in edge_ids {
            graph
                .delete_edge(eid)
                .map_err(|e| format!("memory: unsupersede delete edge {eid}: {e}"))?;
        }
        Ok(())
    }

    /// Record that `from_id` **derives from** each of `to_ids` — typed
    /// `DERIVES_FROM` edges `from → to` (the Why-Graph: a conclusion anchored in
    /// its evidence/premises). **Never changes recall** — it is additive
    /// awareness only (surfaced via `recall --explain` / [`why`](Self::why)).
    /// User curation; idempotent per link. `NotFound` if any id is missing.
    pub fn derive(
        &self,
        _project_key: Option<&str>,
        from_id: i64,
        to_ids: &[i64],
    ) -> Result<(), CurateError> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        graph.get_entity(from_id).map_err(|_| CurateError::NotFound(from_id))?;
        for &b in to_ids {
            graph.get_entity(b).map_err(|_| CurateError::NotFound(b))?;
        }
        // Collect the links that don't exist yet (idempotent), reading on a
        // pooled connection; then insert via the API (cache invalidation).
        let to_insert: Vec<i64> = {
            let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
            let mut stmt = conn
                .prepare("SELECT EXISTS(SELECT 1 FROM graph_edges WHERE from_id=?1 AND to_id=?2 AND edge_type='DERIVES_FROM')")
                .map_err(|e| format!("memory: derive prepare: {e}"))?;
            to_ids
                .iter()
                .copied()
                .filter(|&b| {
                    stmt.query_row((from_id, b), |r| r.get::<_, bool>(0)).map(|e| !e).unwrap_or(true)
                })
                .collect()
        };
        for b in to_insert {
            graph
                .insert_edge(&GraphEdge {
                    id: 0,
                    from_id,
                    to_id: b,
                    edge_type: EdgeType::DerivesFrom.as_str().to_string(),
                    data: json!({ "created_at": now_iso() }),
                })
                .map_err(|e| format!("memory: derive insert {from_id}->{b}: {e}"))?;
        }
        Ok(())
    }

    /// Release a derivation (`from_id DERIVES_FROM to_id`). Inverse of
    /// [`derive`](Self::derive); deletes via the SG API (cache-consistent).
    /// Idempotent. `NotFound` if either id is missing.
    pub fn underive(
        &self,
        _project_key: Option<&str>,
        from_id: i64,
        to_id: i64,
    ) -> Result<(), CurateError> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        graph.get_entity(from_id).map_err(|_| CurateError::NotFound(from_id))?;
        graph.get_entity(to_id).map_err(|_| CurateError::NotFound(to_id))?;
        let edge_ids: Vec<i64> = {
            let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
            let mut stmt = conn
                .prepare("SELECT id FROM graph_edges WHERE from_id=?1 AND to_id=?2 AND edge_type='DERIVES_FROM'")
                .map_err(|e| format!("memory: underive prepare: {e}"))?;
            let rows = stmt
                .query_map((from_id, to_id), |r| r.get::<_, i64>(0))
                .map_err(|e| format!("memory: underive query: {e}"))?;
            rows.filter_map(|r| r.ok()).collect()
        };
        for eid in edge_ids {
            graph
                .delete_edge(eid)
                .map_err(|e| format!("memory: underive delete edge {eid}: {e}"))?;
        }
        Ok(())
    }

    /// `/why <id>`: walk **outgoing `DERIVES_FROM` edges** from `id` and return
    /// the justification tree (`id → [premises] → …`). Read-only. Loads the
    /// `DERIVES_FROM` adjacency once (one `idx_edges_type`-backed query) and
    /// traverses in memory; a **path-based cycle guard** flags an ancestor
    /// re-entry (`A from B`, `B from A`) and a **depth cap** bounds the walk, so
    /// it always terminates. `NotFound` if `id` is missing.
    pub fn why(
        &self,
        _project_key: Option<&str>,
        id: i64,
        depth_cap: usize,
    ) -> Result<WhyNode, CurateError> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        graph.get_entity(id).map_err(|_| CurateError::NotFound(id))?;
        let adj: std::collections::HashMap<i64, Vec<i64>> = {
            let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
            let mut stmt = conn
                .prepare("SELECT from_id, to_id FROM graph_edges WHERE edge_type='DERIVES_FROM' ORDER BY from_id, to_id")
                .map_err(|e| format!("memory: why prepare: {e}"))?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                .map_err(|e| format!("memory: why query: {e}"))?;
            let mut m: std::collections::HashMap<i64, Vec<i64>> = std::collections::HashMap::new();
            for row in rows.flatten() {
                m.entry(row.0).or_default().push(row.1);
            }
            m
        };
        let cap = depth_cap.clamp(1, 32);
        let mut path = std::collections::HashSet::new();
        Ok(Self::build_why(&graph, &adj, id, 0, cap, &mut path))
    }

    /// Recursive Why-Graph builder. `path` holds the current ancestors → an id
    /// already on the path is a cycle (flagged, not expanded); `depth >= cap`
    /// truncates. Both halt recursion → guaranteed termination.
    fn build_why(
        graph: &SqliteGraph,
        adj: &std::collections::HashMap<i64, Vec<i64>>,
        id: i64,
        depth: usize,
        cap: usize,
        path: &mut std::collections::HashSet<i64>,
    ) -> WhyNode {
        let (kind, name, text, note_type) = match graph.get_entity(id) {
            Ok(n) => {
                let text = n.data.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let note_type = n.data.get("type").and_then(|v| v.as_str()).unwrap_or("untyped").to_string();
                (n.kind, n.name, text, note_type)
            }
            Err(_) => (String::new(), String::new(), String::new(), "untyped".to_string()),
        };
        let mut wn = WhyNode {
            id, kind, note_type, name, text,
            derives_from: Vec::new(), cycle: false, truncated: false,
        };
        if path.contains(&id) {
            wn.cycle = true;
            return wn;
        }
        if depth >= cap {
            wn.truncated = true;
            return wn;
        }
        path.insert(id);
        if let Some(children) = adj.get(&id) {
            for &c in children {
                wn.derives_from.push(Self::build_why(graph, adj, c, depth + 1, cap, path));
            }
        }
        path.remove(&id);
        wn
    }

    /// Hard delete: remove the vector from the index AND the node (its edges
    /// cascade via `delete_entity`) from the graph — gone from recall and the
    /// record. The delete is hard (`DELETE FROM`, no bloat); verify reductions
    /// with [`raw_vector_count`], never `statistics().vector_count`.
    pub fn delete(&self, project_key: Option<&str>, id: i64) -> Result<(), CurateError> {
        let key = Self::scope(project_key);
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        // Clear error if the note doesn't exist, before doing rebuild work.
        graph
            .get_entity(id)
            .map_err(|_| CurateError::NotFound(id))?;
        // Gone from recall: rebuild the index without this node's vector.
        Self::reindex_without(&graph, &key, id)?;
        // Gone from the graph too (its edges cascade via delete_entity).
        graph
            .delete_entity(id)
            .map_err(|e| format!("memory: delete node {id}: {e}"))?;
        Ok(())
    }

    /// Un-archive a note: the exact inverse of [`archive`](Self::archive) —
    /// flip `status` `archived → active` and re-insert its vector into the
    /// project index so it surfaces in recall again.
    ///
    /// **Why re-embed (not re-insert a kept vector):** `archive` drops the
    /// vector from the store (`reindex_without` excludes it and the recreated
    /// index no longer holds it — `raw_vector_count` → 0), so the embedding is
    /// **not** preserved. The node + its `text` are. The embedder is
    /// deterministic (greedy INT8 ONNX), so re-embedding `search_document:
    /// {text}` reproduces the original vector; we re-insert it through the same
    /// path as [`remember`](Self::remember), with the **node_id link intact**
    /// (the node was never deleted), so recall recovers the right note.
    ///
    /// **Idempotent:** an already-active note (or one never archived) is a
    /// no-op — it does NOT re-insert (that would leave a duplicate vector for
    /// one node). A missing note is a clean error, never a panic.
    pub fn unarchive(&self, project_key: Option<&str>, id: i64) -> Result<(), CurateError> {
        let key = Self::scope(project_key);

        // 1) Read the note's text + status under the graph lock, then release it
        //    BEFORE embedding (the embedder lock is never held with the graph
        //    lock — same discipline as remember). Idempotent no-op if not
        //    archived (so we never add a second vector for one node).
        let text = {
            let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
            let node = graph
                .get_entity(id)
                .map_err(|_| CurateError::NotFound(id))?;
            let status = node.data.get("status").and_then(|v| v.as_str()).unwrap_or("active");
            if status != "archived" {
                return Ok(()); // already active / never archived → no-op
            }
            node.data.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        if text.trim().is_empty() {
            return Err(CurateError::Internal(format!(
                "memory: unarchive: note {id} has no text to restore"
            )));
        }

        // 2) Re-embed the stored text (archive dropped the vector). Same prefix
        //    as remember → the deterministic embedder reproduces the vector.
        let emb = self.embed(format!("search_document: {text}"))?;
        if emb.len() != EMBED_DIM {
            return Err(CurateError::Internal(format!(
                "memory: embedder returned {} dims, expected {EMBED_DIM}",
                emb.len()
            )));
        }

        // 3) Re-insert the vector (node_id link intact) + flip status to active.
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned".to_string())?;
        // Re-check under the lock (guards a racing unarchive): still archived?
        let mut node = graph
            .get_entity(id)
            .map_err(|_| CurateError::NotFound(id))?;
        let status = node.data.get("status").and_then(|v| v.as_str()).unwrap_or("active");
        if status != "archived" {
            return Ok(()); // another writer restored it meanwhile → no-op
        }
        Self::ensure_index(&graph, &key)?;
        graph
            .get_hnsw_index_mut(&key, |idx| {
                let _vid = idx.insert_vector(&emb, Some(json!({ "node_id": id })))?;
                idx.persist_topology()
            })
            .map_err(|e| format!("memory: index op {key}: {e}"))?
            .map_err(|e| format!("memory: vector add/persist {key}: {e}"))?;
        if let Value::Object(map) = &mut node.data {
            map.insert("status".to_string(), json!("active"));
            map.insert("unarchived_at".to_string(), json!(now_iso()));
            map.remove("archived_at");
        }
        graph
            .update_entity(&node)
            .map_err(|e| format!("memory: unarchive update {id}: {e}"))?;
        Ok(())
    }

    /// Fetch a single note by id **regardless of status** — archived notes
    /// don't surface in recall, so this is how to inspect one (e.g. confirm an
    /// archive kept the record, or a future "show note" command). `score` is
    /// `1.0` (no query was run). `None` if the node is missing (e.g. deleted).
    pub fn get(&self, id: i64) -> Option<Hit> {
        let graph = self.graph.lock().ok()?;
        let node = graph.get_entity(id).ok()?;
        let text = node.data.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let status = node.data.get("status").and_then(|v| v.as_str()).unwrap_or("active").to_string();
        let note_type = node.data.get("type").and_then(|v| v.as_str()).unwrap_or("untyped").to_string();
        Some(Hit { id: node.id, kind: node.kind, name: node.name, text, status, note_type, superseded_by: None, derives_from: Vec::new(), score: 1.0 })
    }

    /// Idempotent: create the project structure node + its index (no error on
    /// re-create). Returns the project node id.
    pub fn create_project(&self, project_key: &str, _name: Option<&str>) -> Result<i64, String> {
        let key = Self::scope(Some(project_key));
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;
        let id = Self::ensure_project_node(&graph, &key)?;
        Self::ensure_index(&graph, &key)?;
        Ok(id)
    }

    /// List all project structure nodes (raw SQL — cypher edge-less MATCH is
    /// avoided).
    pub fn list_projects(&self) -> Result<Vec<ProjectInfo>, String> {
        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;
        let conn = graph.pool.get().map_err(|e| format!("memory: pool: {e}"))?;
        let mut stmt = conn
            .prepare("SELECT id, name FROM graph_entities WHERE kind = ?1 ORDER BY id")
            .map_err(|e| format!("memory: list prepare: {e}"))?;
        let rows = stmt
            .query_map((PROJECT_KIND,), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .map_err(|e| format!("memory: list query: {e}"))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, name) = row.map_err(|e| format!("memory: list row: {e}"))?;
            out.push(ProjectInfo { id, project_key: name });
        }
        Ok(out)
    }

    /// Flush HNSW topology + WAL checkpoint and close cleanly on shutdown.
    /// CPU/SQLite only — no `device_wait_idle` (that is the GPU teardown).
    pub fn shutdown(&self) {
        if let Ok(graph) = self.graph.lock() {
            // Persist every loaded index's topology, then WAL-checkpoint.
            if let Ok(names) = graph.list_hnsw_indexes() {
                for name in names {
                    let _ = graph.get_hnsw_index_mut(&name, |idx| idx.persist_topology());
                }
            }
            if let Ok(conn) = graph.pool.get() {
                let _ = conn.pragma_update(None, "wal_checkpoint", "TRUNCATE");
            }
        }
        eprintln!("VulkanForge: memory — store flushed ({}).", self.db_path.display());
    }
}
