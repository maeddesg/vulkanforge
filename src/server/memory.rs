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

/// One recall result (vector hit enriched with its graph node).
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: i64,
    pub kind: String,
    pub name: String,
    pub text: String,
    pub status: String,
    /// Cosine **similarity** in `[0,1]` = `1 - cosine_distance` (the HNSW
    /// index returns distance; we present similarity, higher = closer).
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectInfo {
    pub id: i64,
    pub project_key: String,
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

    /// Store a nugget: embed → graph node → CONTAINS edge → vector (carrying
    /// `node_id` in its metadata). Returns the new content node id.
    pub fn remember(
        &self,
        project_key: Option<&str>,
        kind: &str,
        text: &str,
        name: Option<&str>,
        metadata: Option<Value>,
    ) -> Result<i64, String> {
        if kind.trim().is_empty() {
            return Err("memory: remember requires a non-empty `kind`".into());
        }
        if text.trim().is_empty() {
            return Err("memory: remember requires non-empty `text`".into());
        }
        let key = Self::scope(project_key);
        // Embed FIRST, outside the graph lock (the expensive ~20 ms step).
        let emb = self.embed(format!("search_document: {text}"))?;
        if emb.len() != EMBED_DIM {
            return Err(format!("memory: embedder returned {} dims, expected {EMBED_DIM}", emb.len()));
        }

        let graph = self.graph.lock().map_err(|_| "memory: graph lock poisoned")?;

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

        Ok(node_id)
    }

    /// Recall: embed query → search the project's index → enrich each hit's
    /// node. Missing index (nothing remembered yet) → empty, not an error.
    pub fn recall(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: usize,
    ) -> Result<Vec<Hit>, String> {
        if query.trim().is_empty() {
            return Err("memory: recall requires a non-empty `query`".into());
        }
        let key = Self::scope(project_key);
        let k = k.clamp(1, 100);
        let emb = self.embed(format!("search_query: {query}"))?;

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
            Err(_index_missing) => return Ok(Vec::new()),
        };

        let mut hits = Vec::with_capacity(pairs.len());
        for (nid, dist) in pairs {
            if let Ok(node) = graph.get_entity(nid) {
                let text = node.data.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let status = node.data.get("status").and_then(|v| v.as_str()).unwrap_or("active").to_string();
                hits.push(Hit {
                    id: node.id,
                    kind: node.kind,
                    name: node.name,
                    text,
                    status,
                    score: 1.0 - dist,
                });
            }
        }
        Ok(hits)
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
