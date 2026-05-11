//! axum Router for the v0.4 API server.
//!
//! Architecture §5.5. Primary OpenAI-standard paths under `/v1/`,
//! plus alias paths without the prefix (some clients omit it — and
//! llama.cpp / Ollama accept both). `/health` has no `/v1/` prefix
//! by convention.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use crate::server::handlers;
use crate::server::state::AppState;

pub fn build_router(state: Arc<AppState>, cors_enabled: bool) -> Router {
    let cors_layer = if cors_enabled {
        // `permissive` allows any origin/method/header. Required
        // for browser-based UIs (Open WebUI, SillyTavern) that run
        // on a different port. Opt-in via `--cors`.
        CorsLayer::permissive()
    } else {
        // No CORS layer = same-origin requests only. Curl, the
        // OpenAI Python SDK, and LangChain don't care; browser
        // pages on a foreign origin get blocked.
        CorsLayer::new()
    };

    Router::new()
        // Primary OpenAI-standard paths
        .route("/v1/chat/completions", post(handlers::chat::completions))
        .route("/v1/models", get(handlers::models::list))
        // Alias paths without the /v1/ prefix
        .route("/chat/completions", post(handlers::chat::completions))
        .route("/models", get(handlers::models::list))
        // Operations
        .route("/health", get(handlers::health::check))
        .layer(cors_layer)
        .with_state(state)
}
