//! `/memory/*` handlers (Stufe A) — VF-native, separate from `/v1/*`.
//!
//! Each handler resolves the shared [`MemoryStore`] from `AppState` and runs
//! the (synchronous, CPU/SQLite) store op via `spawn_blocking` — so it never
//! blocks the async runtime and never touches the GPU permit. `project_key` is
//! optional everywhere → the global default scope.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::server::memory::{
    CurateError, Hit, MemoryStore, NearMiss, NoteType, ProjectInfo, RecallExplain, WhyNode,
};
use crate::server::state::AppState;

type ApiError = (StatusCode, String);

fn store(state: &Arc<AppState>) -> Result<Arc<MemoryStore>, ApiError> {
    state
        .memory
        .clone()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "memory subsystem unavailable".to_string()))
}

/// Map a store error to an HTTP status. The default store error is a flat
/// `String` → always a server fault (500). [`CurateError`] (the id-targeting
/// curation ops) distinguishes a missing id — a **client** error → **404 Not
/// Found** — from a real fault (→ 500). No fragile string-matching: the variant
/// carries the distinction. `ApiError` is a tuple alias, so this is a local
/// trait rather than a `From` impl (orphan rule).
trait IntoApiError {
    fn into_api_error(self) -> ApiError;
}

impl IntoApiError for String {
    fn into_api_error(self) -> ApiError {
        (StatusCode::INTERNAL_SERVER_ERROR, self)
    }
}

impl IntoApiError for CurateError {
    fn into_api_error(self) -> ApiError {
        let status = match &self {
            // A missing id is the caller's mistake, not a server fault.
            CurateError::NotFound(_) => StatusCode::NOT_FOUND,
            // Everything else (lock/DB/IO/embedder/index) stays a 500 — a real
            // server error must NOT be masked as a 404.
            CurateError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.to_string())
    }
}

async fn run_blocking<T, E, F>(f: F) -> Result<T, ApiError>
where
    F: FnOnce() -> Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: IntoApiError + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("memory: task join: {e}")))?
        .map_err(IntoApiError::into_api_error)
}

#[derive(Deserialize)]
pub struct RememberReq {
    pub project_key: Option<String>,
    pub kind: String,
    pub text: String,
    pub name: Option<String>,
    pub metadata: Option<Value>,
    /// Optional layer type (`invariant`/`working`/`episodic`/`decision`/
    /// `failure`). Absent → untyped (no type key written). Validated; an
    /// unknown value is a 400, never silently coerced.
    #[serde(default, rename = "type")]
    pub note_type: Option<String>,
}
#[derive(Serialize)]
pub struct RememberResp {
    pub id: i64,
    /// `true` when the note was a near-duplicate of an existing active note —
    /// then `id` is that existing node and nothing new was stored (Stufe B-3
    /// dedup). Surfaced so the caller can say "already known" instead of
    /// "remembered".
    pub deduped: bool,
}

pub async fn remember(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RememberReq>,
) -> Result<Json<RememberResp>, ApiError> {
    let mem = store(&state)?;
    // Validate + fold the explicit layer type into the note metadata (the store
    // merges it into `data["type"]`). Absent → untyped (no key → read-time
    // default). Unknown type → 400.
    let metadata = match req.note_type.as_deref() {
        Some(t) => {
            let nt = NoteType::parse(t).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
            let mut obj = match req.metadata {
                Some(Value::Object(m)) => m,
                _ => serde_json::Map::new(),
            };
            obj.insert("type".to_string(), Value::String(nt.as_str().to_string()));
            Some(Value::Object(obj))
        }
        None => req.metadata,
    };
    let outcome = run_blocking(move || {
        mem.remember(req.project_key.as_deref(), &req.kind, &req.text, req.name.as_deref(), metadata)
    })
    .await?;
    Ok(Json(RememberResp { id: outcome.id, deduped: outcome.deduped }))
}

/// Request for the curation endpoints (`/memory/archive`, `/memory/delete`):
/// a target note `id` in an optional `project_key` scope.
#[derive(Deserialize)]
pub struct IdReq {
    pub project_key: Option<String>,
    pub id: i64,
}
#[derive(Serialize)]
pub struct ArchiveResp {
    pub id: i64,
    pub status: String,
}
#[derive(Serialize)]
pub struct DeleteResp {
    pub id: i64,
    pub deleted: bool,
}
#[derive(Serialize)]
pub struct UnarchiveResp {
    pub id: i64,
    pub status: String,
}

/// `POST /memory/archive` — drop the note's vector from recall, keep the node
/// as an `archived` record (Stufe B-3).
pub async fn archive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IdReq>,
) -> Result<Json<ArchiveResp>, ApiError> {
    let mem = store(&state)?;
    let id = req.id;
    run_blocking(move || mem.archive(req.project_key.as_deref(), req.id)).await?;
    Ok(Json(ArchiveResp { id, status: "archived".to_string() }))
}

/// `POST /memory/delete` — hard-remove the note from recall AND the graph.
pub async fn delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IdReq>,
) -> Result<Json<DeleteResp>, ApiError> {
    let mem = store(&state)?;
    let id = req.id;
    run_blocking(move || mem.delete(req.project_key.as_deref(), req.id)).await?;
    Ok(Json(DeleteResp { id, deleted: true }))
}

/// `POST /memory/unarchive` — restore an archived note to active + recall (the
/// inverse of `/memory/archive`; user-driven recovery).
pub async fn unarchive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IdReq>,
) -> Result<Json<UnarchiveResp>, ApiError> {
    let mem = store(&state)?;
    let id = req.id;
    run_blocking(move || mem.unarchive(req.project_key.as_deref(), req.id)).await?;
    Ok(Json(UnarchiveResp { id, status: "active".to_string() }))
}

#[derive(Deserialize)]
pub struct RetypeReq {
    pub project_key: Option<String>,
    pub id: i64,
    #[serde(rename = "type")]
    pub note_type: String,
}
#[derive(Serialize)]
pub struct RetypeResp {
    pub id: i64,
    #[serde(rename = "type")]
    pub note_type: String,
}

/// `POST /memory/retype` — set a note's layer type (Schicht-Enabler). Pure
/// metadata, user-driven curation. Unknown type → 400; missing id → 404.
pub async fn retype(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RetypeReq>,
) -> Result<Json<RetypeResp>, ApiError> {
    let mem = store(&state)?;
    let id = req.id;
    let nt = NoteType::parse(&req.note_type).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    run_blocking(move || mem.retype(req.project_key.as_deref(), req.id, nt)).await?;
    Ok(Json(RetypeResp { id, note_type: nt.as_str().to_string() }))
}

#[derive(Deserialize)]
pub struct SupersedeReq {
    pub project_key: Option<String>,
    /// The note that replaces — `new_id SUPERSEDES old_id`.
    pub new_id: i64,
    /// The note being replaced (becomes stale / suppressed from recall).
    pub old_id: i64,
}
#[derive(Serialize)]
pub struct SupersedeResp {
    pub new_id: i64,
    pub old_id: i64,
    pub superseded: bool,
}

/// `POST /memory/supersede` — record `new_id SUPERSEDES old_id` (a typed edge).
/// `old_id` is suppressed from default recall (reversible). Missing id → 404;
/// self-supersede (`new_id == old_id`) → 400.
pub async fn supersede(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SupersedeReq>,
) -> Result<Json<SupersedeResp>, ApiError> {
    let mem = store(&state)?;
    let (new_id, old_id) = (req.new_id, req.old_id);
    if new_id == old_id {
        return Err((StatusCode::BAD_REQUEST, "a note cannot supersede itself".to_string()));
    }
    run_blocking(move || mem.supersede(req.project_key.as_deref(), req.new_id, req.old_id)).await?;
    Ok(Json(SupersedeResp { new_id, old_id, superseded: true }))
}

/// `POST /memory/unsupersede` — release `new_id SUPERSEDES old_id`; `old_id`
/// returns to recall. Inverse of `/memory/supersede`. Missing id → 404.
pub async fn unsupersede(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SupersedeReq>,
) -> Result<Json<SupersedeResp>, ApiError> {
    let mem = store(&state)?;
    let (new_id, old_id) = (req.new_id, req.old_id);
    run_blocking(move || mem.unsupersede(req.project_key.as_deref(), req.new_id, req.old_id)).await?;
    Ok(Json(SupersedeResp { new_id, old_id, superseded: false }))
}

#[derive(Deserialize)]
pub struct DeriveReq {
    pub project_key: Option<String>,
    /// The note that derives — `from_id DERIVES_FROM each of to_ids`.
    pub from_id: i64,
    /// The notes it is anchored in (evidence / premises).
    pub to_ids: Vec<i64>,
}
#[derive(Serialize)]
pub struct DeriveResp {
    pub from_id: i64,
    pub to_ids: Vec<i64>,
    pub derived: bool,
}

/// `POST /memory/derive` — record `from_id DERIVES_FROM to_ids` (the Why-Graph).
/// Never changes recall. Missing id → 404; empty `to_ids` → 400.
pub async fn derive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeriveReq>,
) -> Result<Json<DeriveResp>, ApiError> {
    let mem = store(&state)?;
    if req.to_ids.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "derive requires at least one source id".to_string()));
    }
    let (from_id, to_ids) = (req.from_id, req.to_ids.clone());
    run_blocking(move || mem.derive(req.project_key.as_deref(), req.from_id, &req.to_ids)).await?;
    Ok(Json(DeriveResp { from_id, to_ids, derived: true }))
}

#[derive(Deserialize)]
pub struct UnderiveReq {
    pub project_key: Option<String>,
    pub from_id: i64,
    pub to_id: i64,
}

/// `POST /memory/underive` — release `from_id DERIVES_FROM to_id`. Missing id → 404.
pub async fn underive(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnderiveReq>,
) -> Result<Json<DeriveResp>, ApiError> {
    let mem = store(&state)?;
    let (from_id, to_id) = (req.from_id, req.to_id);
    run_blocking(move || mem.underive(req.project_key.as_deref(), req.from_id, req.to_id)).await?;
    Ok(Json(DeriveResp { from_id, to_ids: vec![to_id], derived: false }))
}

/// Default `/why` traversal depth cap (kept small; the store clamps to [1,32]).
const WHY_DEFAULT_DEPTH: usize = 8;

#[derive(Deserialize)]
pub struct WhyReq {
    pub project_key: Option<String>,
    pub id: i64,
    #[serde(default)]
    pub depth: Option<usize>,
}

/// `POST /memory/why` — the Why-Graph justification tree for a note (read-only).
/// Missing id → 404.
pub async fn why(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WhyReq>,
) -> Result<Json<WhyNode>, ApiError> {
    let mem = store(&state)?;
    let depth = req.depth.unwrap_or(WHY_DEFAULT_DEPTH);
    let tree = run_blocking(move || mem.why(req.project_key.as_deref(), req.id, depth)).await?;
    Ok(Json(tree))
}

#[derive(Deserialize)]
pub struct RecallReq {
    pub project_key: Option<String>,
    pub query: String,
    pub k: Option<usize>,
    /// Opt-in retrieval-diagnostics (`recall --explain`). Default/absent →
    /// `false` → the response is byte-identical to the pre-explain shape.
    #[serde(default)]
    pub explain: Option<bool>,
    /// Opt-in type filter: keep only notes of this layer type. Absent → all
    /// types (behavior unchanged). Unknown value → 400.
    #[serde(default, rename = "type")]
    pub note_type: Option<String>,
    /// Opt-in: include notes that were superseded (default false → stale notes
    /// suppressed). They are never deleted, so this surfaces them again.
    #[serde(default)]
    pub include_superseded: Option<bool>,
}
/// Near-miss + cutoff metadata for `recall --explain`. Only serialised when
/// explain was requested (`skip_serializing_if`), so the default recall
/// response stays exactly `{"hits":[…]}`.
#[derive(Serialize)]
pub struct ExplainResp {
    pub top_k: usize,
    pub threshold: Option<f32>,
    pub query_dim: usize,
    pub near_miss: Vec<NearMiss>,
    pub separation: Option<f32>,
}
#[derive(Serialize)]
pub struct RecallResp {
    pub hits: Vec<Hit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<ExplainResp>,
}

/// Extra candidates fetched beyond `top_k` for the `--explain` near-miss view.
const EXPLAIN_NEAR_MISS_EXTRA: usize = 5;

/// Adaptive recall relevance margin from the environment (`VF_RECALL_MARGIN`,
/// cosine-similarity units). **Default OFF**: unset / unparseable / `≤ 0` →
/// `None` → pure top-k, so recall stays byte-identical to today. A positive
/// value (e.g. `0.13`) activates the relative-to-top threshold for every
/// recall (agent tool and user `/recall`). Read per-request, like the KV-reuse
/// switch — the default flip is mg's call, gated on the validation report.
fn recall_margin() -> Option<f32> {
    std::env::var("VF_RECALL_MARGIN")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|m| m.is_finite() && *m > 0.0)
}

pub async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallReq>,
) -> Result<Json<RecallResp>, ApiError> {
    let mem = store(&state)?;
    let k = req.k.unwrap_or(5);
    let margin = recall_margin();
    // Optional type filter (validated; unknown → 400). Absent → all types.
    let type_filter = match req.note_type.as_deref() {
        Some(t) => Some(NoteType::parse(t).map_err(|e| (StatusCode::BAD_REQUEST, e))?),
        None => None,
    };
    let include_superseded = req.include_superseded == Some(true);
    if req.explain == Some(true) {
        let RecallExplain { returned, near_miss, top_k, threshold, query_dim, separation } =
            run_blocking(move || {
                mem.recall_explain(
                    req.project_key.as_deref(), &req.query, k, EXPLAIN_NEAR_MISS_EXTRA, margin, type_filter, include_superseded,
                )
            })
            .await?;
        Ok(Json(RecallResp {
            hits: returned,
            explain: Some(ExplainResp { top_k, threshold, query_dim, near_miss, separation }),
        }))
    } else {
        let hits = run_blocking(move || {
            mem.recall_filtered(req.project_key.as_deref(), &req.query, k, margin, type_filter, include_superseded)
        })
        .await?;
        Ok(Json(RecallResp { hits, explain: None }))
    }
}

#[derive(Deserialize)]
pub struct ProjectReq {
    pub project_key: String,
    pub name: Option<String>,
}
#[derive(Serialize)]
pub struct ProjectResp {
    pub id: i64,
    pub project_key: String,
}

pub async fn create_project(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProjectReq>,
) -> Result<Json<ProjectResp>, ApiError> {
    let mem = store(&state)?;
    let key = req.project_key.clone();
    let id = run_blocking(move || mem.create_project(&req.project_key, req.name.as_deref())).await?;
    Ok(Json(ProjectResp { id, project_key: key }))
}

#[derive(Serialize)]
pub struct ProjectsResp {
    pub projects: Vec<ProjectInfo>,
}

pub async fn list_projects(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ProjectsResp>, ApiError> {
    let mem = store(&state)?;
    let projects = run_blocking(move || mem.list_projects()).await?;
    Ok(Json(ProjectsResp { projects }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curate_not_found_maps_to_404_other_errors_stay_500() {
        // The whole point of the sprint: a missing id is a client error (404),
        // every other failure stays a server error (500), and the message is
        // still informative. This is the mapping every curation endpoint shares
        // via `run_blocking` → `IntoApiError`.
        let nf = CurateError::NotFound(99_999).into_api_error();
        assert_eq!(nf.0, StatusCode::NOT_FOUND, "missing id → 404");
        assert!(nf.1.contains("not found"), "message stays informative: {}", nf.1);
        assert!(nf.1.contains("99999"), "names the id: {}", nf.1);

        // A real fault must NOT be masked as 404.
        let internal = CurateError::Internal("memory: graph lock poisoned".into()).into_api_error();
        assert_eq!(internal.0, StatusCode::INTERNAL_SERVER_ERROR, "real fault stays 500");
        assert!(internal.1.contains("lock poisoned"), "message preserved: {}", internal.1);

        // The default flat `String` store error (remember/recall/…) stays 500.
        let s = "memory: embed: boom".to_string().into_api_error();
        assert_eq!(s.0, StatusCode::INTERNAL_SERVER_ERROR, "String store error stays 500");
    }

    #[test]
    fn default_recall_response_omits_explain_field() {
        // `recall --explain` is additive: without it, the response JSON must be
        // byte-identical to the pre-explain shape — exactly `{"hits":[…]}`, no
        // `explain` key (guaranteed by `skip_serializing_if = "Option::is_none"`).
        let resp = RecallResp {
            hits: vec![Hit {
                id: 1,
                kind: "fact".into(),
                name: "n".into(),
                text: "t".into(),
                status: "active".into(),
                note_type: "untyped".into(),
                superseded_by: None,
                derives_from: Vec::new(),
                score: 0.9,
            }],
            explain: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("explain"), "default response leaked an explain key: {json}");
        assert!(json.starts_with("{\"hits\":"), "shape changed: {json}");

        // With explain set, the field appears.
        let ex = RecallResp {
            hits: vec![],
            explain: Some(ExplainResp {
                top_k: 5,
                threshold: None,
                query_dim: 768,
                near_miss: vec![],
                separation: None,
            }),
        };
        let json = serde_json::to_string(&ex).unwrap();
        assert!(json.contains("\"explain\""), "explain mode must include the field: {json}");
        assert!(json.contains("\"top_k\":5"), "explain carries the cutoff: {json}");
    }
}
