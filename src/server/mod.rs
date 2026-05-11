//! VulkanForge v0.4 OpenAI-compatible API server.
//!
//! Sprint 1 (Foundation) lands the wire-format types, the error
//! representation, and the OpenAI-to-VF sampling-parameter mapping.
//! Sprints 2-3 add the router, handlers, streaming adapter, and CLI
//! wiring per `docs/v0.4/api_server_architecture.md`.
//!
//! Nothing in this module touches the GPU. Everything is pure
//! request/response data and the math to map one to the other.

pub mod error;
pub mod sampling;
pub mod types;
