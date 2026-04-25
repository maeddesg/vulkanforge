//! VulkanForge library — re-exports the backend module tree so
//! integration tests under `tests/` can drive the Vulkan stack
//! through the same surface that `main.rs` uses.

pub mod backend;
