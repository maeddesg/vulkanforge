//! `GET /health` — liveness + model-load status.
//!
//! Architecture §2.3. Returns 200 with a `HealthResponse` once the
//! model is loaded. Sprint 2 ships only the always-ready variant;
//! the 503 `loading` shape stays in the type so async-load (Sprint 4
//! Open Decision #9) can fill it in later without a wire change.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Json;

use crate::server::state::AppState;
use crate::server::types::response::{HealthResponse, HealthStatus, KvCacheInfo};

pub async fn check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    // The session mutex is brief (just reading kv_cache config).
    // We hold it sync — no .await across the lock.
    let (max_seq, cur_pos) = {
        let s = state.session.lock().expect("session mutex poisoned");
        let cfg = &s.chat.forward.kv_cache.config;
        (cfg.max_seq_len, s.chat.current_pos)
    };

    Json(HealthResponse {
        status: HealthStatus::Ok,
        model_loaded: true,
        model_id: Some(state.model_id.clone()),
        version: env!("CARGO_PKG_VERSION"),
        kv_cache: Some(KvCacheInfo {
            max_seq_len: max_seq,
            current_pos: cur_pos,
        }),
    })
}
