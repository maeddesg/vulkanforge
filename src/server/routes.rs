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

    #[allow(unused_mut)] // `mut` is only used when the `memory` feature adds routes.
    let mut router = Router::new()
        // Primary OpenAI-standard paths
        .route("/v1/chat/completions", post(handlers::chat::completions))
        .route("/v1/completions", post(handlers::completions::completions))
        .route("/v1/models", get(handlers::models::list))
        // Alias paths without the /v1/ prefix
        .route("/chat/completions", post(handlers::chat::completions))
        .route("/completions", post(handlers::completions::completions))
        .route("/models", get(handlers::models::list))
        // Operations
        .route("/health", get(handlers::health::check));

    // VF-native memory subsystem (Stufe A) — NOT OpenAI; separate namespace.
    // Registered only when compiled with `--features memory`; on a lean build
    // these paths simply don't exist (404). Runtime activation is a second gate
    // (`serve --memory`): when off, the handlers themselves return 503.
    #[cfg(feature = "memory")]
    {
        router = router
            .route("/memory/remember", post(handlers::memory::remember))
            .route("/memory/recall", post(handlers::memory::recall))
            .route(
                "/memory/projects",
                post(handlers::memory::create_project).get(handlers::memory::list_projects),
            )
            // Curation (Stufe B-3): archive (out of recall, kept as record) +
            // delete (hard) + unarchive (restore an archived note to recall).
            .route("/memory/archive", post(handlers::memory::archive))
            .route("/memory/delete", post(handlers::memory::delete))
            .route("/memory/unarchive", post(handlers::memory::unarchive))
            // Schicht-Enabler: set a note's layer type (pure metadata).
            .route("/memory/retype", post(handlers::memory::retype))
            // Connections: SUPERSEDES edge (+ release) — stale-suppression.
            .route("/memory/supersede", post(handlers::memory::supersede))
            .route("/memory/unsupersede", post(handlers::memory::unsupersede))
            // Why-Graph: DERIVES_FROM edge (+ release) + the /why trace.
            .route("/memory/derive", post(handlers::memory::derive))
            .route("/memory/underive", post(handlers::memory::underive))
            .route("/memory/why", post(handlers::memory::why))
            // Conflict awareness: symmetric CONTRADICTS edge (+ release). Never
            // suppresses — flagged in --explain, reconciled via /supersede.
            .route("/memory/contradict", post(handlers::memory::contradict))
            .route("/memory/uncontradict", post(handlers::memory::uncontradict));
    }

    router.layer(cors_layer).with_state(state)
}
