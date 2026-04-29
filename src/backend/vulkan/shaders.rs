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
    /// v0.2 Sprint 9a — fused SwiGLU: out[i] = silu(gate[i]) * up[i].
    /// Drop-in replacement for the separate Silu + Mul dispatch pair
    /// inside the FFN block. Saves one dispatch and one compute
    /// barrier per layer.
    SwiGLU,
    /// v0.2 Sprint 9b — fused residual-add + RMSNorm-mul. Combines
    /// `add_res1` (a + b → sum) with `rms_norm_ffn` (rms_norm(sum) *
    /// weight → norm_out) into a single dispatch. Saves one dispatch
    /// and one compute barrier per layer at the attn→ffn transition.
    MultiAddRms,
    /// v0.2 Sprint 9c.5 — fused rms_norm+mul+RoPE for Q/K-norm. Built
    /// from rms_norm.comp with RMS_NORM_ROPE_FUSION=1 so the
    /// normalized+gamma-multiplied intermediate is staged in LDS and
    /// then rotated by the rope_neox path in the same dispatch.
    /// Replaces the Q/K-norm + RoPE pair (4 dispatches + 2 barriers
    /// per layer) with 2 dispatches + 1 barrier.
    RmsNormMulRope,
    /// v0.2 Sprint 9d.2 — FP32 → packed-FP16 KV-cache write conversion
    /// shader. Replaces vkCmdCopyBuffer for prefill K/V → KV-cache
    /// writes when `KvCache::is_fp16()`.
    KvCopyFp16,
    /// v0.2 Sprint 9d.2 — FP16 KV-aware variant of
    /// FlashAttnTiledBr16Bc32. Same source SPV with `FP16_KV=1`
    /// build define; K/V bindings are `uint[]` packed FP16, decoded
    /// per-element via unpackHalf2x16.
    FlashAttnTiledBr16Bc32Fp16Kv,
    /// v0.2 Sprint 9d.2 — FP16 KV-aware variant of FlashAttnBatch
    /// (Br=1 fallback). Same source SPV, `FP16_KV=1` build define.
    FlashAttnBatchFp16Kv,
    /// v0.2 Sprint 9d.3 — FP16 KV-aware variant of FlashAttn
    /// (single-WG decode attention, seq_len ≤ 64). Same source
    /// SPV, `FP16_KV=1` build define.
    FlashAttnFp16Kv,
    /// v0.2 Sprint 9d.3 — FP16 KV-aware variant of FlashAttnSplit
    /// (multi-WG split-K decode worker, seq_len > 64). Same source
    /// SPV, `FP16_KV=1` build define.
    FlashAttnSplitFp16Kv,
    /// v0.2 Sprint 10B — scalar QK micro-benchmark. Computes Score =
    /// Q × K^T over Br=Bc=16, head_dim=128 in plain FP32 FMA. Used
    /// by examples/bench_qk to size up the coopmat win.
    BenchQkScalar,
    /// v0.2 Sprint 10B — coopmat QK micro-benchmark. Same shape,
    /// 16×16×16 FP16→FP32 WMMA fragments via VK_KHR_cooperative_matrix.
    BenchQkCoopmat,
    /// v0.2 Sprint 10C — coopmat flash-attention v1 (FP32 KV).
    /// Drop-in replacement for FlashAttnTiledBr16Bc32 with QK score
    /// computed via VK_KHR_cooperative_matrix WMMA; softmax + PV
    /// remain scalar (PV will move to coopmat in Sprint 10D).
    FlashAttnCoopmat,
    /// v0.2 Sprint 10C — coopmat flash-attention v1 (FP16 KV).
    /// Same source, FP16_KV=1 build define matching Sprint 9d.2's
    /// FP16 KV-cache packing.
    FlashAttnCoopmatFp16Kv,
    SoftMax,
    Copy,
    ScalarAttn,
    // Phase 3C — integer-MMQ GEMM for Prefill (compiled and registered;
    // dispatch wiring lands in Phase 3D).
    MulMmqQ4K,
    MulMmqQ6K,
    // Sprint 11C — large-tile (BM=128 BN=128) mul_mmq variants for
    // prefill at m>64 && n>64. Same SPV as MulMmqQ{4,6}K, only the
    // spec-constants differ (warptile values from llama.cpp's
    // l_warptile_mmq_int_k AMD-coopmat-override at gfx1201).
    MulMmqQ4KL,
    MulMmqQ6KL,
    // Sprint 11E — KHR coopmat mul_mm path (mul_mm.comp + COOPMAT=1).
    // Uses 16x16x16 FP16xFP16->FP32 fragments via coopMatMulAdd.
    // FP32 activations (skips Q8_1 quantize), FP16 LDS, FP32 accumulator.
    // Spec-constants from llama.cpp's warptile_mmq AMD-coopmat-override
    // (ggml-vulkan.cpp:3367) at gfx1201.
    MulMmQ4KCoopmat,
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
    // Sprint 7 / 7.5 — Br>1 tiled-Q flash attention. Identical
    // bindings/PC to FlashAttnBatch; dispatched as (n_heads,
    // ceil(M/BR), 1) with BR queries per WG sharing one K-tile load.
    // Three Br variants because GLSL shared arrays can't be sized by
    // spec constants; each is its own SPV. Selected at runtime via
    // VULKANFORGE_FA_TILED + VULKANFORGE_FA_BR=4|8|16. Default OFF.
    FlashAttnTiledBr4,
    FlashAttnTiledBr8,
    FlashAttnTiledBr16,
    // Sprint 7.6 — Br=16 with Bc=32 (smaller K-tile → less LDS,
    // more WG-occupancy headroom). Same dispatch shape as
    // FlashAttnTiledBr16; only the K-tile / scores LDS sizes differ.
    FlashAttnTiledBr16Bc32,
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
    // v0.2 Sprint 3A — Q4_K dequant-fusion coopmat GEMM with forward-
    // pass-compatible memory layout (B = [N, K] activations, C = [N, M]
    // output). Three BN variants for the per-shape selector. Default
    // OFF; gated behind `VULKANFORGE_COOPMAT=1` in `forward.rs`.
    MulCoopmatQ4KFwdBn64,
    MulCoopmatQ4KFwdBn32,
    MulCoopmatQ4KFwdBn16,
    // v0.2 Sprint 3B — naive Q4_K coopmat with BF16 narrowing.
    // Single subgroup per WG, single 16x16 output tile. Default
    // path for skinny-N prefill (seq_len ≤ 64) when
    // VULKANFORGE_COOPMAT=1.
    MulCoopmatQ4KNaiveBf16,
    // v0.2 Sprint 3C — N-padded variants. The runtime guarantees
    // pc.n is a multiple of 16 and the activation tail rows are
    // zeroed; the kernel can then use a direct ColumnMajor
    // coopMatStore (no LDS staging). FP8 variant retests Sprint 3A's
    // failed precision experiment now that the partial-tile-store
    // bug is eliminated.
    MulCoopmatQ4KNaivePaddedBf16,
    MulCoopmatQ4KNaivePaddedFp8,
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
            ShaderId::SwiGLU => "swiglu_f32",
            ShaderId::MultiAddRms => "multi_add_rms_f32",
            ShaderId::RmsNormMulRope => "rms_norm_mul_rope_f32",
            ShaderId::KvCopyFp16 => "kv_copy_fp16",
            ShaderId::FlashAttnTiledBr16Bc32Fp16Kv => "flash_attn_tiled_br16_bc32_fp16kv",
            ShaderId::FlashAttnBatchFp16Kv => "flash_attn_batch_fp16kv",
            ShaderId::FlashAttnFp16Kv => "flash_attn_fp16kv",
            ShaderId::FlashAttnSplitFp16Kv => "flash_attn_split_fp16kv",
            ShaderId::BenchQkScalar => "bench_qk_scalar",
            ShaderId::BenchQkCoopmat => "bench_qk_coopmat",
            ShaderId::FlashAttnCoopmat => "flash_attn_coopmat",
            ShaderId::FlashAttnCoopmatFp16Kv => "flash_attn_coopmat_fp16kv",
            ShaderId::SoftMax => "soft_max_f32",
            ShaderId::Copy => "copy_f32_f32",
            ShaderId::ScalarAttn => "scalar_attn_f32",
            ShaderId::MulMmqQ4K | ShaderId::MulMmqQ4KL => "mul_mmq_q4_k_f32",
            ShaderId::MulMmqQ6K | ShaderId::MulMmqQ6KL => "mul_mmq_q6_k_f32",
            ShaderId::MulMmQ4KCoopmat => "mul_mm_q4_k_f32_coopmat",
            ShaderId::QuantizeQ8_1 => "quantize_q8_1_f32",
            ShaderId::FlashAttn => "flash_attn_f32",
            ShaderId::FlashAttnSplit => "flash_attn_split_f32",
            ShaderId::FlashAttnReduce => "flash_attn_reduce_f32",
            ShaderId::FlashAttnBatch => "flash_attn_batch_f32",
            ShaderId::FlashAttnTiledBr4 => "flash_attn_tiled_br4",
            ShaderId::FlashAttnTiledBr8 => "flash_attn_tiled_br8",
            ShaderId::FlashAttnTiledBr16 => "flash_attn_tiled_br16",
            ShaderId::FlashAttnTiledBr16Bc32 => "flash_attn_tiled_br16_bc32",
            ShaderId::MulMmQ4K => "mul_mm_q4_k_f32",
            ShaderId::MulMmQ6K => "mul_mm_q6_k_f32",
            ShaderId::MulMmQ4KAligned => "mul_mm_q4_k_f32_aligned",
            ShaderId::MulMmQ6KAligned => "mul_mm_q6_k_f32_aligned",
            ShaderId::MulCoopmatQ4KFwdBn64 => "mul_coopmat_q4k_fwd_bn64",
            ShaderId::MulCoopmatQ4KFwdBn32 => "mul_coopmat_q4k_fwd_bn32",
            ShaderId::MulCoopmatQ4KFwdBn16 => "mul_coopmat_q4k_fwd_bn16",
            ShaderId::MulCoopmatQ4KNaiveBf16 => "mul_coopmat_q4k_naive_bf16",
            ShaderId::MulCoopmatQ4KNaivePaddedBf16 => "mul_coopmat_q4k_naive_padded_bf16",
            ShaderId::MulCoopmatQ4KNaivePaddedFp8 => "mul_coopmat_q4k_naive_padded_fp8",
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
            ShaderId::SwiGLU => SWIGLU_F32,
            ShaderId::MultiAddRms => MULTI_ADD_RMS_F32,
            ShaderId::RmsNormMulRope => RMS_NORM_MUL_ROPE_F32,
            ShaderId::KvCopyFp16 => KV_COPY_FP16,
            ShaderId::FlashAttnTiledBr16Bc32Fp16Kv => FLASH_ATTN_TILED_BR16_BC32_FP16KV,
            ShaderId::FlashAttnBatchFp16Kv => FLASH_ATTN_BATCH_FP16KV,
            ShaderId::FlashAttnFp16Kv => FLASH_ATTN_FP16KV,
            ShaderId::FlashAttnSplitFp16Kv => FLASH_ATTN_SPLIT_FP16KV,
            ShaderId::BenchQkScalar => BENCH_QK_SCALAR,
            ShaderId::BenchQkCoopmat => BENCH_QK_COOPMAT,
            ShaderId::FlashAttnCoopmat => FLASH_ATTN_COOPMAT,
            ShaderId::FlashAttnCoopmatFp16Kv => FLASH_ATTN_COOPMAT_FP16KV,
            ShaderId::SoftMax => SOFT_MAX_F32,
            ShaderId::Copy => COPY_F32_F32,
            ShaderId::ScalarAttn => SCALAR_ATTN_F32,
            ShaderId::MulMmqQ4K | ShaderId::MulMmqQ4KL => MUL_MMQ_Q4_K_F32,
            ShaderId::MulMmqQ6K | ShaderId::MulMmqQ6KL => MUL_MMQ_Q6_K_F32,
            ShaderId::QuantizeQ8_1 => QUANTIZE_Q8_1_F32,
            ShaderId::FlashAttn => FLASH_ATTN_F32,
            ShaderId::FlashAttnSplit => FLASH_ATTN_SPLIT_F32,
            ShaderId::FlashAttnReduce => FLASH_ATTN_REDUCE_F32,
            ShaderId::FlashAttnBatch => FLASH_ATTN_BATCH_F32,
            ShaderId::FlashAttnTiledBr4 => FLASH_ATTN_TILED_BR4,
            ShaderId::FlashAttnTiledBr8 => FLASH_ATTN_TILED_BR8,
            ShaderId::FlashAttnTiledBr16 => FLASH_ATTN_TILED_BR16,
            ShaderId::FlashAttnTiledBr16Bc32 => FLASH_ATTN_TILED_BR16_BC32,
            ShaderId::MulMmQ4K => MUL_MM_Q4_K_F32,
            ShaderId::MulMmQ6K => MUL_MM_Q6_K_F32,
            ShaderId::MulMmQ4KAligned => MUL_MM_Q4_K_F32_ALIGNED,
            ShaderId::MulMmQ6KAligned => MUL_MM_Q6_K_F32_ALIGNED,
            ShaderId::MulMmQ4KCoopmat => MUL_MM_Q4_K_F32_COOPMAT,
            ShaderId::MulCoopmatQ4KFwdBn64 => MUL_COOPMAT_Q4K_FWD_BN64,
            ShaderId::MulCoopmatQ4KFwdBn32 => MUL_COOPMAT_Q4K_FWD_BN32,
            ShaderId::MulCoopmatQ4KFwdBn16 => MUL_COOPMAT_Q4K_FWD_BN16,
            ShaderId::MulCoopmatQ4KNaiveBf16 => MUL_COOPMAT_Q4K_NAIVE_BF16,
            ShaderId::MulCoopmatQ4KNaivePaddedBf16 => MUL_COOPMAT_Q4K_NAIVE_PADDED_BF16,
            ShaderId::MulCoopmatQ4KNaivePaddedFp8 => MUL_COOPMAT_Q4K_NAIVE_PADDED_FP8,
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
    ShaderId::SwiGLU,
    ShaderId::MultiAddRms,
    ShaderId::RmsNormMulRope,
    ShaderId::KvCopyFp16,
    ShaderId::FlashAttnTiledBr16Bc32Fp16Kv,
    ShaderId::FlashAttnBatchFp16Kv,
    ShaderId::FlashAttnFp16Kv,
    ShaderId::FlashAttnSplitFp16Kv,
    ShaderId::BenchQkScalar,
    ShaderId::BenchQkCoopmat,
    ShaderId::FlashAttnCoopmat,
    ShaderId::FlashAttnCoopmatFp16Kv,
    ShaderId::SoftMax,
    ShaderId::Copy,
    ShaderId::ScalarAttn,
    ShaderId::MulMmqQ4K,
    ShaderId::MulMmqQ6K,
    ShaderId::MulMmqQ4KL,
    ShaderId::MulMmqQ6KL,
    ShaderId::MulMmQ4KCoopmat,
    ShaderId::QuantizeQ8_1,
    ShaderId::FlashAttn,
    ShaderId::FlashAttnSplit,
    ShaderId::FlashAttnReduce,
    ShaderId::FlashAttnBatch,
    ShaderId::FlashAttnTiledBr4,
    ShaderId::FlashAttnTiledBr8,
    ShaderId::FlashAttnTiledBr16,
    ShaderId::FlashAttnTiledBr16Bc32,
    ShaderId::MulMmQ4K,
    ShaderId::MulMmQ6K,
    ShaderId::MulMmQ4KAligned,
    ShaderId::MulMmQ6KAligned,
    ShaderId::MulCoopmatQ4KFwdBn64,
    ShaderId::MulCoopmatQ4KFwdBn32,
    ShaderId::MulCoopmatQ4KFwdBn16,
    ShaderId::MulCoopmatQ4KNaiveBf16,
    ShaderId::MulCoopmatQ4KNaivePaddedBf16,
    ShaderId::MulCoopmatQ4KNaivePaddedFp8,
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
pub const SWIGLU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu_f32.spv"));
pub const MULTI_ADD_RMS_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/multi_add_rms_f32.spv"));
pub const RMS_NORM_MUL_ROPE_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_mul_rope_f32.spv"));
pub const KV_COPY_FP16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kv_copy_fp16.spv"));
pub const FLASH_ATTN_TILED_BR16_BC32_FP16KV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_tiled_br16_bc32_fp16kv.spv"));
pub const FLASH_ATTN_BATCH_FP16KV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_batch_fp16kv.spv"));
pub const FLASH_ATTN_FP16KV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_fp16kv.spv"));
pub const FLASH_ATTN_SPLIT_FP16KV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_split_fp16kv.spv"));
pub const BENCH_QK_SCALAR: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bench_qk_scalar.spv"));
pub const BENCH_QK_COOPMAT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bench_qk_coopmat.spv"));
pub const FLASH_ATTN_COOPMAT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_coopmat.spv"));
pub const FLASH_ATTN_COOPMAT_FP16KV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_coopmat_fp16kv.spv"));
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
pub const FLASH_ATTN_TILED_BR4: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_tiled_br4.spv"));
pub const FLASH_ATTN_TILED_BR8: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_tiled_br8.spv"));
pub const FLASH_ATTN_TILED_BR16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_tiled_br16.spv"));
pub const FLASH_ATTN_TILED_BR16_BC32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_tiled_br16_bc32.spv"));
pub const MUL_MM_Q4_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q4_k_f32.spv"));
pub const MUL_MM_Q6_K_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q6_k_f32.spv"));
pub const MUL_MM_Q4_K_F32_ALIGNED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q4_k_f32_aligned.spv"));
pub const MUL_MM_Q6_K_F32_ALIGNED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q6_k_f32_aligned.spv"));
pub const MUL_MM_Q4_K_F32_COOPMAT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mm_q4_k_f32_coopmat.spv"));
pub const MUL_COOPMAT_Q4K_FWD_BN64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_fwd_bn64.spv"));
pub const MUL_COOPMAT_Q4K_FWD_BN32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_fwd_bn32.spv"));
pub const MUL_COOPMAT_Q4K_FWD_BN16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_fwd_bn16.spv"));
pub const MUL_COOPMAT_Q4K_NAIVE_BF16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_naive_bf16.spv"));
pub const MUL_COOPMAT_Q4K_NAIVE_PADDED_BF16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_naive_padded_bf16.spv"));
pub const MUL_COOPMAT_Q4K_NAIVE_PADDED_FP8: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_naive_padded_fp8.spv"));

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
