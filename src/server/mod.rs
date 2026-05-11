//! VulkanForge v0.4 OpenAI-compatible API server.
//!
//! Sprint 1 (Foundation) landed the wire-format types, the error
//! representation, and the OpenAI-to-VF sampling-parameter mapping.
//! Sprint 2 adds the router, handlers, application state, and the
//! `vulkanforge serve` CLI subcommand for non-streaming requests.
//! Sprint 3 will lift the streaming-rejection in `chat::completions`
//! by wiring up an SSE adapter.

pub mod error;
pub mod handlers;
pub mod routes;
pub mod sampling;
pub mod serve;
pub mod state;
pub mod types;
