//! axum handlers for the v0.4 API endpoints.
//!
//! One module per endpoint family:
//! - [`chat`] — `POST /v1/chat/completions` (+ alias)
//! - [`completions`] — `POST /v1/completions` (+ alias), legacy text-completion
//! - [`models`] — `GET /v1/models` (+ alias)
//! - [`health`] — `GET /health`

pub mod chat;
pub mod completions;
pub mod health;
pub mod memory;
pub mod models;
