//! OpenAI-compatible request/response types. The wire-level shape
//! matches `docs/v0.4/api_server_architecture.md` §3.1-§3.6 exactly;
//! divergences (e.g. additional `stream_options` field on the
//! request) come from the Sprint-0 Merge-Sprint and are documented
//! at their declarations.

pub mod request;
pub mod response;
pub mod streaming;

/// Value of `system_fingerprint` returned in every chat completion
/// response and chunk. OpenAI uses this to identify the model+server
/// build; we encode the VF version. Sprint 2's serve-command will
/// override this at runtime if a `--system-fingerprint` flag is added.
pub const SYSTEM_FINGERPRINT: &str = concat!("vulkanforge-", env!("CARGO_PKG_VERSION"));
