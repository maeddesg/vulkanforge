//! `GET /v1/models` (and `/models` alias) — enumerate the single
//! loaded model. Architecture §2.2.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Json;

use crate::server::state::AppState;
use crate::server::types::response::{ModelInfo, ModelListResponse};

pub async fn list(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let info = ModelInfo {
        id: state.model_id.clone(),
        object: "model",
        created: state.started_at,
        owned_by: "vulkanforge",
        permission: Vec::new(),
        root: state.model_id.clone(),
        parent: None,
    };
    Json(ModelListResponse::single(info))
}
