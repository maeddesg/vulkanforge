//! VulkanForge library — re-exports the backend module tree so
//! integration tests under `tests/` can drive the Vulkan stack
//! through the same surface that `main.rs` uses.

// Crate-wide clippy allow-list. These lints fire heavily in the
// shader-dispatch / FFI code, are stylistic only, and cleaning them
// up would either churn working code (manual_div_ceil → div_ceil
// rewrites in 16 places, identical behaviour) or fight the codebase
// shape (too_many_arguments — every Vulkan dispatch helper takes
// device + registry + cmd + several buffers + dims by design).
//
// Re-evaluate post-v0.1.0 when there's time for a focused lint pass.
#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::ptr_arg,
    clippy::print_literal,
    clippy::op_ref,
    clippy::needless_question_mark,
    clippy::match_like_matches_macro,
    clippy::doc_lazy_continuation,
    clippy::collapsible_else_if,
)]

pub mod backend;
