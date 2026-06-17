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

use crate::server::memory::{CurateError, Hit, MemoryStore, ProjectInfo};
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
    let outcome = run_blocking(move || {
        mem.remember(
            req.project_key.as_deref(),
            &req.kind,
            &req.text,
            req.name.as_deref(),
            req.metadata,
        )
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
pub struct RecallReq {
    pub project_key: Option<String>,
    pub query: String,
    pub k: Option<usize>,
}
#[derive(Serialize)]
pub struct RecallResp {
    pub hits: Vec<Hit>,
}

pub async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallReq>,
) -> Result<Json<RecallResp>, ApiError> {
    let mem = store(&state)?;
    let k = req.k.unwrap_or(5);
    let hits = run_blocking(move || mem.recall(req.project_key.as_deref(), &req.query, k)).await?;
    Ok(Json(RecallResp { hits }))
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
}
