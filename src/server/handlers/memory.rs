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

use crate::server::memory::{Hit, MemoryStore, ProjectInfo};
use crate::server::state::AppState;

type ApiError = (StatusCode, String);

fn store(state: &Arc<AppState>) -> Result<Arc<MemoryStore>, ApiError> {
    state
        .memory
        .clone()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "memory subsystem unavailable".to_string()))
}

async fn run_blocking<T, F>(f: F) -> Result<T, ApiError>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("memory: task join: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
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
}

pub async fn remember(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RememberReq>,
) -> Result<Json<RememberResp>, ApiError> {
    let mem = store(&state)?;
    let id = run_blocking(move || {
        mem.remember(
            req.project_key.as_deref(),
            &req.kind,
            &req.text,
            req.name.as_deref(),
            req.metadata,
        )
    })
    .await?;
    Ok(Json(RememberResp { id }))
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
