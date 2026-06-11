//! `GET /health` — liveness + model-load status.
//!
//! Architecture §2.3. Returns 200 with a `HealthResponse` once the
//! model is loaded. Sprint 2 ships only the always-ready variant;
//! the 503 `loading` shape stays in the type so async-load (Sprint 4
//! Open Decision #9) can fill it in later without a wire change.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Json;

use crate::server::error::ApiError;
use crate::server::state::AppState;
use crate::server::types::response::{HealthResponse, HealthStatus, KvCacheInfo};

pub async fn check(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HealthResponse>, ApiError> {
    // The session mutex is brief (just reading kv_cache config).
    // We hold it sync — no .await across the lock.
    // Hardening (F7): a poisoned mutex (a prior generation panicked while
    // holding it) now returns a clean 500 — matching the chat/completions
    // handlers — instead of panicking the health-check task on `.expect()`.
    let (max_seq, cur_pos) = {
        let s = state
            .session
            .lock()
            .map_err(|_| ApiError::internal("session mutex poisoned", "internal_error"))?;
        let cfg = &s.chat.forward.kv_cache.config;
        (cfg.max_seq_len, s.chat.current_pos)
    };

    Ok(Json(HealthResponse {
        status: HealthStatus::Ok,
        model_loaded: true,
        model_id: Some(state.model_id.clone()),
        version: env!("CARGO_PKG_VERSION"),
        kv_cache: Some(KvCacheInfo {
            max_seq_len: max_seq,
            current_pos: cur_pos,
        }),
    }))
}
