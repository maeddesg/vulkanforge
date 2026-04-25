//! Vulkan backend (compute-only). Currently exposes the minimal
//! device-init wrapper used by the Phase-0 smoke test; future
//! Phase-1 work adds memory allocator, descriptor sets, pipelines,
//! and the Q4_K MMVQ dispatch.

pub mod device;
pub mod pipeline;
pub mod shaders;
