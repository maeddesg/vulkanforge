//! Pre-compiled SPIR-V blobs, embedded at build time.
//!
//! Phase 2A: every shader the decode pipeline needs is compiled by
//! `build.rs` and embedded here. `ShaderId` is the canonical handle
//! used by [`super::pipeline_registry::PipelineRegistry`] to look up
//! a compiled pipeline.

/// Canonical identifier for every compiled shader. Order maps to
/// the order in [`ALL_SHADERS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderId {
    MulMatVecQ4K,
    MulMatVecQ6K,
    RmsNorm,
    RopeNorm,
    Add,
    Mul,
    Silu,
    SoftMax,
    Copy,
}

impl ShaderId {
    pub fn name(self) -> &'static str {
        match self {
            ShaderId::MulMatVecQ4K => "mul_mat_vec_q4_k_f32_f32",
            ShaderId::MulMatVecQ6K => "mul_mat_vec_q6_k_f32_f32",
            ShaderId::RmsNorm => "rms_norm_f32",
            ShaderId::RopeNorm => "rope_norm_f32",
            ShaderId::Add => "add_f32",
            ShaderId::Mul => "mul_f32",
            ShaderId::Silu => "silu_f32",
            ShaderId::SoftMax => "soft_max_f32",
            ShaderId::Copy => "copy_f32_f32",
        }
    }

    pub fn spv_bytes(self) -> &'static [u8] {
        match self {
            ShaderId::MulMatVecQ4K => MUL_MAT_VEC_Q4_K_F32_F32,
            ShaderId::MulMatVecQ6K => MUL_MAT_VEC_Q6_K_F32_F32,
            ShaderId::RmsNorm => RMS_NORM_F32,
            ShaderId::RopeNorm => ROPE_NORM_F32,
            ShaderId::Add => ADD_F32,
            ShaderId::Mul => MUL_F32,
            ShaderId::Silu => SILU_F32,
            ShaderId::SoftMax => SOFT_MAX_F32,
            ShaderId::Copy => COPY_F32_F32,
        }
    }
}

pub const ALL_SHADERS: &[ShaderId] = &[
    ShaderId::MulMatVecQ4K,
    ShaderId::MulMatVecQ6K,
    ShaderId::RmsNorm,
    ShaderId::RopeNorm,
    ShaderId::Add,
    ShaderId::Mul,
    ShaderId::Silu,
    ShaderId::SoftMax,
    ShaderId::Copy,
];

pub const MUL_MAT_VEC_Q4_K_F32_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_k_f32_f32.spv"));
pub const MUL_MAT_VEC_Q6_K_F32_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q6_k_f32_f32.spv"));
pub const RMS_NORM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_f32.spv"));
pub const ROPE_NORM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_norm_f32.spv"));
pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));
pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_f32.spv"));
pub const SILU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_f32.spv"));
pub const SOFT_MAX_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/soft_max_f32.spv"));
pub const COPY_F32_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_f32_f32.spv"));

/// Decode a SPIR-V byte blob into u32 words. Vulkan consumes SPIR-V
/// as `&[u32]`; `include_bytes!` only gives us `&[u8]` whose alignment
/// is not guaranteed, so we copy into an owned, naturally-aligned
/// `Vec<u32>`.
pub fn spv_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V length not 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
