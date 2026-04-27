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
    RopeNeox,
    Add,
    Mul,
    Silu,
    SoftMax,
    Copy,
    ScalarAttn,
    // Phase 3C — integer-MMQ GEMM for Prefill (compiled and registered;
    // dispatch wiring lands in Phase 3D).
    MulMmqQ4K,
    MulMmqQ6K,
    QuantizeQ8_1,
    // Phase 4B — online-softmax decode attention; drop-in for ScalarAttn.
    FlashAttn,
    // Phase 4C — split-K (multi-WG-per-head) attention worker + reducer.
    // Dispatched together when n_tiles >= MULTI_WG_MIN_TILES; otherwise
    // forward.rs falls back to FlashAttn (single WG per head).
    FlashAttnSplit,
    FlashAttnReduce,
    // Phase 5B.1 — batched-Q flash attention for prefill. One dispatch
    // covers (n_heads, M, 1) with a causal mask, replacing the M-fold
    // FlashAttn dispatch loop the per-token prefill currently runs.
    FlashAttnBatch,
    // Phase 6 v0.1.2 — mul_mm.comp port. Q4_K / Q6_K weights × FP32
    // activations (no Q8_1 quantize step). Used by prefill_batch when
    // VULKANFORGE_USE_MUL_MM is set; mul_mmq stays as the gated
    // fallback.
    MulMmQ4K,
    MulMmQ6K,
    // Phase 7 (cont.) — aligned mul_mm. Same shader, built with
    // ALIGNED=1 / LOAD_VEC_B=4 / B_TYPE=vec4 so load_b_to_shmem takes
    // the vec4 path. Only valid when the shader's N (= seq_len) is
    // divisible by 4; runtime falls back to mul_mmq otherwise.
    MulMmQ4KAligned,
    MulMmQ6KAligned,
}

impl ShaderId {
    pub fn name(self) -> &'static str {
        match self {
            ShaderId::MulMatVecQ4K => "mul_mat_vec_q4_k_f32_f32",
            ShaderId::MulMatVecQ6K => "mul_mat_vec_q6_k_f32_f32",
            ShaderId::RmsNorm => "rms_norm_f32",
            ShaderId::RopeNorm => "rope_norm_f32",
            ShaderId::RopeNeox => "rope_neox_f32",
            ShaderId::Add => "add_f32",
            ShaderId::Mul => "mul_f32",
            ShaderId::Silu => "silu_f32",
            ShaderId::SoftMax => "soft_max_f32",
            ShaderId::Copy => "copy_f32_f32",
            ShaderId::ScalarAttn => "scalar_attn_f32",
            ShaderId::MulMmqQ4K => "mul_mmq_q4_k_f32",
            ShaderId::MulMmqQ6K => "mul_mmq_q6_k_f32",
            ShaderId::QuantizeQ8_1 => "quantize_q8_1_f32",
            ShaderId::FlashAttn => "flash_attn_f32",
            ShaderId::FlashAttnSplit => "flash_attn_split_f32",
            ShaderId::FlashAttnReduce => "flash_attn_reduce_f32",
            ShaderId::FlashAttnBatch => "flash_attn_batch_f32",
            ShaderId::MulMmQ4K => "mul_mm_q4_k_f32",
            ShaderId::MulMmQ6K => "mul_mm_q6_k_f32",
            ShaderId::MulMmQ4KAligned => "mul_mm_q4_k_f32_aligned",
            ShaderId::MulMmQ6KAligned => "mul_mm_q6_k_f32_aligned",
        }
    }

    pub fn spv_bytes(self) -> &'static [u8] {
        match self {
            ShaderId::MulMatVecQ4K => MUL_MAT_VEC_Q4_K_F32_F32,
            ShaderId::MulMatVecQ6K => MUL_MAT_VEC_Q6_K_F32_F32,
            ShaderId::RmsNorm => RMS_NORM_F32,
            ShaderId::RopeNorm => ROPE_NORM_F32,
            ShaderId::RopeNeox => ROPE_NEOX_F32,
            ShaderId::Add => ADD_F32,
            ShaderId::Mul => MUL_F32,
            ShaderId::Silu => SILU_F32,
            ShaderId::SoftMax => SOFT_MAX_F32,
            ShaderId::Copy => COPY_F32_F32,
            ShaderId::ScalarAttn => SCALAR_ATTN_F32,
            ShaderId::MulMmqQ4K => MUL_MMQ_Q4_K_F32,
            ShaderId::MulMmqQ6K => MUL_MMQ_Q6_K_F32,
            ShaderId::QuantizeQ8_1 => QUANTIZE_Q8_1_F32,
            ShaderId::FlashAttn => FLASH_ATTN_F32,
            ShaderId::FlashAttnSplit => FLASH_ATTN_SPLIT_F32,
            ShaderId::FlashAttnReduce => FLASH_ATTN_REDUCE_F32,
            ShaderId::FlashAttnBatch => FLASH_ATTN_BATCH_F32,
            ShaderId::MulMmQ4K => MUL_MM_Q4_K_F32,
            ShaderId::MulMmQ6K => MUL_MM_Q6_K_F32,
            ShaderId::MulMmQ4KAligned => MUL_MM_Q4_K_F32_ALIGNED,
            ShaderId::MulMmQ6KAligned => MUL_MM_Q6_K_F32_ALIGNED,
        }
    }
}

pub const ALL_SHADERS: &[ShaderId] = &[
    ShaderId::MulMatVecQ4K,
    ShaderId::MulMatVecQ6K,
    ShaderId::RmsNorm,
    ShaderId::RopeNorm,
    ShaderId::RopeNeox,
    ShaderId::Add,
    ShaderId::Mul,
    ShaderId::Silu,
    ShaderId::SoftMax,
    ShaderId::Copy,
    ShaderId::ScalarAttn,
    ShaderId::MulMmqQ4K,
    ShaderId::MulMmqQ6K,
    ShaderId::QuantizeQ8_1,
    ShaderId::FlashAttn,
    ShaderId::FlashAttnSplit,
    ShaderId::FlashAttnReduce,
    ShaderId::FlashAttnBatch,
    ShaderId::MulMmQ4K,
    ShaderId::MulMmQ6K,
    ShaderId::MulMmQ4KAligned,
    ShaderId::MulMmQ6KAligned,
];

pub const MUL_MAT_VEC_Q4_K_F32_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_k_f32_f32.spv"));
pub const MUL_MAT_VEC_Q6_K_F32_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q6_k_f32_f32.spv"));
pub const RMS_NORM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_f32.spv"));
pub const ROPE_NORM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_norm_f32.spv"));
pub const ROPE_NEOX_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_neox_f32.spv"));
pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));
pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_f32.spv"));
pub const SILU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_f32.spv"));
pub const SOFT_MAX_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/soft_max_f32.spv"));
pub const COPY_F32_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_f32_f32.spv"));
pub const SCALAR_ATTN_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/scalar_attn_f32.spv"));
pub const MUL_MMQ_Q4_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mmq_q4_k_f32.spv"));
pub const MUL_MMQ_Q6_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mmq_q6_k_f32.spv"));
pub const QUANTIZE_Q8_1_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1_f32.spv"));
pub const FLASH_ATTN_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_f32.spv"));
pub const FLASH_ATTN_SPLIT_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_split_f32.spv"));
pub const FLASH_ATTN_REDUCE_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_reduce_f32.spv"));
pub const FLASH_ATTN_BATCH_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_batch_f32.spv"));
pub const MUL_MM_Q4_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q4_k_f32.spv"));
pub const MUL_MM_Q6_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q6_k_f32.spv"));
pub const MUL_MM_Q4_K_F32_ALIGNED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q4_k_f32_aligned.spv"));
pub const MUL_MM_Q6_K_F32_ALIGNED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q6_k_f32_aligned.spv"));

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
