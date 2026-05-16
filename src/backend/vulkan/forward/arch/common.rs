//! Sprint 44B-1 — extracted from `forward/mod.rs` (pure code-move).
//!
//! Architecture-agnostic helpers shared by every dispatch path:
//! - `layer_weight*` family: tensor / scale-buffer / scale-block lookup.
//! - `is_fp8_layer_weight` / `layer_weight_shader` / `layer_weight_shader_gemm`
//!   / `layer_weight_shader_mmq`: per-quant pipeline selection.
//! - `layer_dims`: per-layer (head_dim, ffn_dim, rope_theta, rotary_dim)
//!   override that lets a single `Forward` instance carry a Gemma-4 stack
//!   with mixed `head_dim`/`ffn_dim`.
//! - `compute_barrier` / `transfer_to_compute_barrier`: pipeline barriers.

use ash::vk;

use super::super::super::device::VulkanDevice;
use super::super::super::gguf::{GgmlType, ModelConfig};
use super::super::super::loader::LoadedModel;
use super::super::super::shaders::ShaderId;

/// Phase 6/7 — pick the right GEMM shader for a given layer weight.
///
/// `gemm_kind` is the per-batch choice: `Mmq` (Q8_1 activations),
/// `MulMm` (FP32 activations, scalar B-loads), or `MulMmAligned`
/// (FP32 activations, vec4 B-loads — only safe when shader N is
/// divisible by 4). Mixed-quant in Qwen3 (`attn_v` + `ffn_down`
/// are Q6_K, the rest Q4_K) goes through both Q4_K and Q6_K paths.
///
/// Sprint 11C — when `gemm_kind == Mmq` and the dispatch shape would
/// fill the GPU at L-tile granularity, the L-tile pipeline (BM=BN=128)
/// is preferred over the default S-tile (BM=BN=64).
///
/// Empirical RDNA4 (RX 9070 XT, 64 CUs):
///   pp=128 (n=128, groups_y=1)  L-tile starved →  −27 % vs S
///   pp=256 (n=256, groups_y=2)  L-tile marginal →  −4 % vs S
///   pp=512 (n=512, groups_y=4)  L-tile fills    →  +4 %
///   pp≥1024                     L-tile dominates →  +4–5 %
///
/// Threshold pragmatically pinned at `m > 128 && n > 256`: L-tile
/// only when the dispatch produces ≥64 workgroups, matching the CU
/// count. At smaller shapes the S-tile keeps its dispatch density
/// advantage. (llama.cpp picks at `m<=64 || n<=64`; we're stricter
/// because we don't ship an M-tile fallback yet — Sprint 11D may add
/// one.)
///
/// `VULKANFORGE_DISABLE_L_TILE=1` forces every dispatch back to S.
pub(crate) fn layer_weight_shader_gemm(
    model: &LoadedModel,
    layer: u32,
    suffix: &str,
    gemm_kind: GemmKind,
    m: u32,
    n: u32,
    coopmat_q4k_mm: bool,
    coopmat_f16acc: bool,
) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    let ggml_type = model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type;
    let q6 = ggml_type == GgmlType::Q6K;
    let q3 = ggml_type == GgmlType::Q3K;
    let q5 = ggml_type == GgmlType::Q5K;
    let q40 = ggml_type == GgmlType::Q4_0;
    // Sprint 46C — F32 weight (Gemma-4 SafeTensors) routes through the
    // F32×F32 mul_mm SPVs added in Sprint 46B. There is no Mmq variant
    // (mul_mmq always quantises activations to Q8_1 — F32 weights have
    // no dequant path that matches that contract). Caller's gemm_kind
    // is forced to MulMm{,Aligned} in BatchExec when an F32 weight is
    // detected, so the Mmq arm here is unreachable.
    if ggml_type == GgmlType::F32 {
        return match gemm_kind {
            GemmKind::MulMmAligned => ShaderId::MulMmF32Aligned,
            GemmKind::MulMm => ShaderId::MulMmF32,
            GemmKind::Mmq => unreachable!(
                "F32 weights cannot use the Mmq path; BatchExec must \
                 force MulMm/MulMmAligned for F32 layers"
            ),
        };
    }
    let prefer_l = m > 128 && n > 256
        && std::env::var("VULKANFORGE_DISABLE_L_TILE")
            .map(|s| s != "1").unwrap_or(true);
    // Sprint 12L — for the coopmat path, prefer the LOAD_VEC_B=8
    // mat2x4 wide-load variant when seq_len (= n here) is divisible
    // by 8. Mirrors llama.cpp's f32_aligned_cm1 selection. Misaligned
    // shapes fall back to the unaligned coopmat (LOAD_VEC_B=2).
    let coopmat_aligned = coopmat_q4k_mm && n % 8 == 0;
    // Sprint 12M — pick M-tile (BM=64, BN=64) when seq_len is small.
    // Sprint 13A — pick S-tile (BM=32, BN=32) when seq_len is very small
    // (n ≤ 32). Three-way port of llama.cpp ggml_vk_guess_matmul_pipeline:7141:
    //   m ≤ 32 || n ≤ 32 → S-tile
    //   m ≤ 64 || n ≤ 64 → M-tile
    //   else             → L-tile
    // Reduced here to the dimension that varies with prompt size (m =
    // output rows is always >> 64 for our model: q_dim=4096, hidden=4096,
    // ffn=12288). WG-count vs N=12288:
    //   pp=64  L-tile = 1.5 WG/CU,  M-tile = 3 WG/CU
    //   pp=32  L-tile = 1.5 WG/CU,  M-tile = 3 WG/CU,  S-tile = 6 WG/CU
    let coopmat_s_tile = coopmat_q4k_mm && n <= 32;
    let coopmat_m_tile = coopmat_q4k_mm && n <= 64 && !coopmat_s_tile;
    // Sprint 19A — Q3_K + Q5_K MulMm coopmat coverage. Each quant has
    // its own SPV family (DATA_A is baked at SPIR-V build time), so we
    // branch by ggml_type first. The Q4_K / Q6_K / Q4_0 fallthrough
    // match below handles every other layout.
    if q3 {
        return match (gemm_kind, coopmat_q4k_mm, coopmat_aligned, coopmat_m_tile, coopmat_s_tile) {
            (GemmKind::MulMmAligned, _,     _,     _,     _    ) => ShaderId::MulMmQ3KAligned,
            (GemmKind::MulMm,        false, _,     _,     _    ) => ShaderId::MulMmQ3K,
            (GemmKind::MulMm,        true,  false, false, false) => ShaderId::MulMmQ3KCoopmat,
            (GemmKind::MulMm,        true,  true,  false, false) if coopmat_f16acc => ShaderId::MulMmQ3KAlignedCoopmatF16Acc,
            (GemmKind::MulMm,        true,  true,  false, false) => ShaderId::MulMmQ3KAlignedCoopmat,
            (GemmKind::MulMm,        true,  false, true,  false) => ShaderId::MulMmQ3KCoopmatM,
            (GemmKind::MulMm,        true,  true,  true,  false) => ShaderId::MulMmQ3KAlignedCoopmatM,
            (GemmKind::MulMm,        true,  false, false, true ) => ShaderId::MulMmQ3KCoopmatS,
            (GemmKind::MulMm,        true,  true,  false, true ) => ShaderId::MulMmQ3KAlignedCoopmatS,
            (GemmKind::MulMm,        true,  _,     true,  true ) => unreachable!("s_tile and m_tile are mutually exclusive"),
            (GemmKind::Mmq,          _,     _,     _,     _    ) => if prefer_l { ShaderId::MulMmqQ3KL } else { ShaderId::MulMmqQ3K },
        };
    }
    if q5 {
        return match (gemm_kind, coopmat_q4k_mm, coopmat_aligned, coopmat_m_tile, coopmat_s_tile) {
            (GemmKind::MulMmAligned, _,     _,     _,     _    ) => ShaderId::MulMmQ5KAligned,
            (GemmKind::MulMm,        false, _,     _,     _    ) => ShaderId::MulMmQ5K,
            (GemmKind::MulMm,        true,  false, false, false) => ShaderId::MulMmQ5KCoopmat,
            (GemmKind::MulMm,        true,  true,  false, false) if coopmat_f16acc => ShaderId::MulMmQ5KAlignedCoopmatF16Acc,
            (GemmKind::MulMm,        true,  true,  false, false) => ShaderId::MulMmQ5KAlignedCoopmat,
            (GemmKind::MulMm,        true,  false, true,  false) => ShaderId::MulMmQ5KCoopmatM,
            (GemmKind::MulMm,        true,  true,  true,  false) => ShaderId::MulMmQ5KAlignedCoopmatM,
            (GemmKind::MulMm,        true,  false, false, true ) => ShaderId::MulMmQ5KCoopmatS,
            (GemmKind::MulMm,        true,  true,  false, true ) => ShaderId::MulMmQ5KAlignedCoopmatS,
            (GemmKind::MulMm,        true,  _,     true,  true ) => unreachable!("s_tile and m_tile are mutually exclusive"),
            (GemmKind::Mmq,          _,     _,     _,     _    ) => if prefer_l { ShaderId::MulMmqQ5KL } else { ShaderId::MulMmqQ5K },
        };
    }
    // Sprint 52J — Q5_0 / Q5_1 / Q8_0 early-returns. Only Mmq variants
    // exist (no coopmat / MulMm / aligned SPVs). 26B-A4B Q3_K_M uses
    // Q8_0 for attn_k/v (prefill path) → Forward::new forces Mmq when
    // these quants are present. Same shape as the Q4_0 fallthrough at
    // line 183 below, but bound to the specific quant type up-front so
    // the dispatch doesn't fall into the Q4_K case silently.
    if ggml_type == GgmlType::Q5_0 {
        return match gemm_kind {
            GemmKind::Mmq => ShaderId::MulMmqQ5_0,
            other => panic!(
                "gemm: Q5_0 only supports Mmq (got {other:?}); \
                 Forward::new should force Mmq when Q5_0 weights are present",
            ),
        };
    }
    if ggml_type == GgmlType::Q5_1 {
        return match gemm_kind {
            GemmKind::Mmq => ShaderId::MulMmqQ5_1,
            other => panic!("gemm: Q5_1 only supports Mmq (got {other:?})"),
        };
    }
    if ggml_type == GgmlType::Q8_0 {
        return match gemm_kind {
            GemmKind::Mmq => ShaderId::MulMmqQ8_0,
            other => panic!("gemm: Q8_0 only supports Mmq (got {other:?})"),
        };
    }
    match (gemm_kind, q6) {
        (GemmKind::MulMmAligned, true)  => ShaderId::MulMmQ6KAligned,
        (GemmKind::MulMmAligned, false) => ShaderId::MulMmQ4KAligned,
        // Sprint 12K — when MM_COOPMAT is on, route Q6_K weights to the
        // dedicated Q6_K coopmat SPV instead of falling back to scalar
        // mul_mm FP32 (which was the slowest GEMM we have, gemm_down
        // 61ms→89ms regression in 12J).
        // Sprint 12L — pick the aligned coopmat variant when shape allows.
        // Sprint 12M — pick the M-tile coopmat variant when seq_len ≤ 64.
        // Sprint 13A — pick the S-tile coopmat variant when seq_len ≤ 32.
        // s_tile and m_tile are mutually exclusive by construction.
        // Sprint 13C — f16acc opt-in. Only redirects the aligned-L-tile
        // case (the bread-and-butter pp ≥ 128 dispatch), which is also
        // the only f16acc SPV we ship. Unaligned / M-tile / S-tile keep
        // the FP32-accumulator path even when the env var is set.
        (GemmKind::MulMm,        true)  => match (coopmat_q4k_mm, coopmat_aligned, coopmat_m_tile, coopmat_s_tile) {
            (false, _,     _,     _    ) => ShaderId::MulMmQ6K,
            (true,  false, false, false) => ShaderId::MulMmQ6KCoopmat,
            (true,  true,  false, false) if coopmat_f16acc => ShaderId::MulMmQ6KAlignedCoopmatF16Acc,
            (true,  true,  false, false) => ShaderId::MulMmQ6KAlignedCoopmat,
            (true,  false, true,  false) => ShaderId::MulMmQ6KCoopmatM,
            (true,  true,  true,  false) => ShaderId::MulMmQ6KAlignedCoopmatM,
            (true,  false, false, true ) => ShaderId::MulMmQ6KCoopmatS,
            (true,  true,  false, true ) => ShaderId::MulMmQ6KAlignedCoopmatS,
            (true,  _,     true,  true ) => unreachable!("s_tile and m_tile are mutually exclusive"),
        },
        (GemmKind::MulMm,        false) => match (coopmat_q4k_mm, coopmat_aligned, coopmat_m_tile, coopmat_s_tile) {
            (false, _,     _,     _    ) => ShaderId::MulMmQ4K,
            (true,  false, false, false) => ShaderId::MulMmQ4KCoopmat,
            (true,  true,  false, false) if coopmat_f16acc => ShaderId::MulMmQ4KAlignedCoopmatF16Acc,
            (true,  true,  false, false) => ShaderId::MulMmQ4KAlignedCoopmat,
            (true,  false, true,  false) => ShaderId::MulMmQ4KCoopmatM,
            (true,  true,  true,  false) => ShaderId::MulMmQ4KAlignedCoopmatM,
            (true,  false, false, true ) => ShaderId::MulMmQ4KCoopmatS,
            (true,  true,  false, true ) => ShaderId::MulMmQ4KAlignedCoopmatS,
            (true,  _,     true,  true ) => unreachable!("s_tile and m_tile are mutually exclusive"),
        },
        // Sprint 17B — Q3_K Mmq for Q3_K_M GGUFs. Production prefill
        // route for these models. Q3_K coopmat / MulMmAligned variants
        // intentionally not built; Forward::new detects Q3_K presence
        // and forces gemm_kind=Mmq for the whole batch.
        // Sprint 17C — Q5_K Mmq for Q3_K_M (attn_v + ffn_down) and
        // Q5_K_M (most layers). Routes by tensor type before the
        // generic Q4_K fall-through; without this Q5_K weights would
        // be read with the wrong block stride (176 B vs 144 B).
        (GemmKind::Mmq,          true)  => if prefer_l { ShaderId::MulMmqQ6KL } else { ShaderId::MulMmqQ6K },
        // Q3_K / Q5_K early-returned above (Sprint 19A); only Q4_0 + Q4_K
        // reach this fallthrough on the false-q6 side.
        (GemmKind::Mmq,          false) if q40 => if prefer_l { ShaderId::MulMmqQ4_0L } else { ShaderId::MulMmqQ4_0 },
        (GemmKind::Mmq,          false) => if prefer_l { ShaderId::MulMmqQ4KL } else { ShaderId::MulMmqQ4K },
    }
}

/// Per-batch GEMM dispatch kind. Decided once per `dispatch_layer_batch`
/// call from the configured `mul_mm_enabled` flag and the runtime
/// `seq_len % 4` alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmKind {
    /// `mul_mmq.comp`: Q8_1-quantized activations. Always valid.
    Mmq,
    /// `mul_mm.comp`: FP32 activations, scalar B-loads. Reachable
    /// only via diagnostic / parity tests; production routes either
    /// `Mmq` or `MulMmAligned`.
    #[allow(dead_code)]
    MulMm,
    /// `mul_mm.comp` with `ALIGNED=1 / LOAD_VEC_B=4 / B_TYPE=vec4`.
    /// Used when `mul_mm_enabled` and `seq_len % 4 == 0`. The
    /// vec4 B-load path skips the unaligned bounds check, so this
    /// is unsafe at unaligned `seq_len`.
    MulMmAligned,
}

#[allow(dead_code)] // Kept for parity with `layer_weight_shader_gemm`; still referenced
                    // by the `forward_layer_debug` helper paths in older diagnostic builds.
pub(crate) fn layer_weight_shader_mmq(model: &LoadedModel, layer: u32, suffix: &str) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    match model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type
    {
        GgmlType::Q6K => ShaderId::MulMmqQ6K,
        _ => ShaderId::MulMmqQ4K,
    }
}

/// Sprint 43C — per-layer (head_dim, ffn_dim, rope_theta) override for
/// Gemma-4. Returns the layer-specific values when `cfg.gemma4` is
/// `Some`; otherwise the uniform `cfg.head_dim` / `cfg.ffn_dim` /
/// `cfg.rope_freq_base`. The dispatch paths thread these through their
/// per-call push-constants so a single `Forward` instance covers a
/// stack with mixed `head_dim ∈ {256, 512}` and
/// `intermediate_size ∈ {6144, 12288}` correctly.
///
/// Note on the rotary-dim: `rope_partial_factor = Some(0.25)` for full-
/// attention layers in E2B → only the first 25 % of the head dim is
/// rotated (p-RoPE). When `None` the full head_dim is rotated (default
/// RoPE). The push-constant on the RoPE shader was extended to take
/// `rotary_dim` for this exact reason.
pub(crate) fn layer_dims(cfg: &ModelConfig, layer: u32) -> (u32, u32, f32, u32) {
    if let Some(g) = cfg.gemma4.as_ref() {
        let s = &g.layers[layer as usize];
        let rotary_dim = match s.rope_partial_factor {
            Some(f) => ((s.head_dim as f32) * f).round() as u32,
            None => s.head_dim,
        };
        (s.head_dim, s.intermediate_size, s.rope_theta, rotary_dim)
    } else {
        (cfg.head_dim, cfg.ffn_dim, cfg.rope_freq_base, cfg.head_dim)
    }
}

/// Sprint 51B-pre — per-layer KV-head count. Mirrors `layer_dims`
/// for `head_dim`: returns the Gemma-4 per-layer value when
/// `cfg.gemma4` is set, falls back to the uniform `cfg.n_kv_heads`
/// otherwise. Architectures without a per-layer KV-head override
/// (Llama, Qwen3, Mistral, …) hit the fallback and stay bit-identical
/// to pre-51B-pre behaviour. Used by every per-dispatch site that
/// currently reads `cfg.n_kv_heads` directly (executor.rs, runs.rs,
/// debug.rs); load-time / worst-case scratch sizing keeps the global
/// `cfg.n_kv_heads` (= the maximum across layer types).
pub(crate) fn n_kv_heads_for(cfg: &ModelConfig, layer: u32) -> u32 {
    if let Some(g) = cfg.gemma4.as_ref() {
        g.layers[layer as usize].n_kv_heads
    } else {
        cfg.n_kv_heads
    }
}

pub(crate) fn layer_weight(model: &LoadedModel, layer: u32, suffix: &str) -> vk::Buffer {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .buffer
        .handle
}

/// Sprint 20-M3 — per-tensor dequant scale lookup. Returns 1.0 for
/// GGUF tensors (which carry no per-tensor scale; their per-block
/// scales live in the weight bytes themselves) and the actual FP32
/// scale for SafeTensors FP8 tensors. `run_gemv` always writes this
/// value into push-constant `broadcast3`; only the FP8/F32 shaders
/// read the slot.
/// Sprint 24B — Optional layer weight (returns `None` if absent).
/// Used for the Qwen2 attention biases that don't appear on Llama /
/// Qwen3 / Mistral models.
pub(crate) fn layer_weight_opt(model: &LoadedModel, layer: u32, suffix: &str) -> Option<vk::Buffer> {
    let key = format!("blk.{layer}.{suffix}");
    model.tensor(&key).map(|t| t.buffer.handle)
}

/// Sprint 24A — Returns the per-tensor weight_scale scalar (1.0 for
/// per-channel models, GGUF tensors, and unquantized SafeTensors).
/// Consumed by the FP8 GEMV decode path via push-constant.
/// Per-channel decode coherence requires the GEMM path (binding 3
/// scale buffer) — Sprint 25 follow-up.
pub(crate) fn layer_weight_scale_scalar(model: &LoadedModel, layer: u32, suffix: &str) -> f32 {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .and_then(|t| t.weight_scale)
        .unwrap_or(1.0)
}

/// Sprint 24A — Returns the weight_scale buffer handle for an FP8
/// layer weight. `None` for unquantized tensors and for GGUF models;
/// the FP8 GEMM dispatch sites bind it to descriptor slot 3 of
/// the `MulCoopmatFp8*` kernels. Per-tensor models have the on-disk
/// scalar pre-broadcast to `[out_dim]` at upload time so the kernel
/// always indexes `scale[row]`.
pub(crate) fn layer_weight_scale_buf(model: &LoadedModel, layer: u32, suffix: &str) -> Option<vk::Buffer> {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .and_then(|t| t.scale_buffer.as_ref().map(|b| b.handle))
}

/// Sprint 35 — Returns the (block_n, block_k) tuple when a layer's
/// weight ships a 2D block-wise scale grid (Qwen3-FP8 / DeepSeek-V3-FP8).
/// `None` for per-tensor / per-channel scales — those callers stick
/// with the per-channel GEMV path.
pub(crate) fn layer_weight_scale_block(model: &LoadedModel, layer: u32, suffix: &str) -> Option<(u32, u32)> {
    let key = format!("blk.{layer}.{suffix}");
    model.tensor(&key).and_then(|t| t.scale_block)
}

/// Sprint 20-Wire — `true` iff a layer's weight tensor is FP8 E4M3.
/// Used at the 7 GEMM call sites in `dispatch_layer_batch` to route
/// SafeTensors FP8 weights through `run_gemm_fp8_naive` instead of
/// the K-quant `run_gemm`. GGUF models always return `false`.
pub(crate) fn is_fp8_layer_weight(model: &LoadedModel, layer: u32, suffix: &str) -> bool {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .map(|t| t.ggml_type == GgmlType::F8E4M3)
        .unwrap_or(false)
}

/// Sprint 46C — `true` iff a layer's weight tensor is FP32. Used by
/// `BatchExec::b_run_proj` to route Gemma-4 SafeTensors weights
/// through the dedicated `mul_mm_f32{,_aligned}.spv` SPVs (added in
/// Sprint 46B) instead of the Q4_K coopmat / Mmq path. GGUF models
/// always return `false`.
pub(crate) fn is_f32_layer_weight(model: &LoadedModel, layer: u32, suffix: &str) -> bool {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .map(|t| t.ggml_type == GgmlType::F32)
        .unwrap_or(false)
}

/// Q4_K_M mixes quant types — `attn_v.weight` and `ffn_down.weight`
/// are Q6_K, the rest are Q4_K. Pick the matching GEMV pipeline.
///
/// Sprint 14B — `subgroup` selects the subgroupAdd SPV variants
/// (Path A reduction) over the LDS tree-reduction stock variants
/// (Path B). Default-on at decode; opt-out via
/// `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`.
pub(crate) fn layer_weight_shader(model: &LoadedModel, layer: u32, suffix: &str, subgroup: bool) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    let ggml_type = model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type;
    // Sprint 17B + 17C + 17D — route by ggml_type. Each quant has
    // its own GEMV variant: Q3_K (110 B blocks), Q4_K (144 B), Q5_K
    // (176 B), Q6_K (210 B), Q4_0 (18 B / 32-weight blocks — note
    // smaller block size than the K-quants). Falling through to
    // Q4_K reads the wrong block stride and produces garbage —
    // the Sprint 17B-debug bug.
    // Sprint 20-M3 — F8E4M3 / F32 routing for SafeTensors models.
    // F8E4M3 → native FP8 GEMV; F32 → unquantized GEMV (used for
    // lm_head when the model excludes it from FP8 quantization).
    // F8/F32 paths are subgroup-agnostic (they hard-code one Wave64
    // workgroup per row).
    match (ggml_type, subgroup) {
        (GgmlType::F8E4M3, _) => ShaderId::MulMatVecFp8,
        (GgmlType::F32,    _) => ShaderId::MulMatVecF32,
        (GgmlType::F16,    _) => ShaderId::MulMatVecF16,
        (GgmlType::Q6K, true ) => ShaderId::MulMatVecQ6KSubgroup,
        (GgmlType::Q6K, false) => ShaderId::MulMatVecQ6K,
        (GgmlType::Q3K, true ) => ShaderId::MulMatVecQ3KSubgroup,
        (GgmlType::Q3K, false) => ShaderId::MulMatVecQ3K,
        (GgmlType::Q5K, true ) => ShaderId::MulMatVecQ5KSubgroup,
        (GgmlType::Q5K, false) => ShaderId::MulMatVecQ5K,
        (GgmlType::Q4K, true ) => ShaderId::MulMatVecQ4KSubgroup,
        (GgmlType::Q4K, false) => ShaderId::MulMatVecQ4K,
        (GgmlType::Q4_0, true ) => ShaderId::MulMatVecQ4_0Subgroup,
        (GgmlType::Q4_0, false) => ShaderId::MulMatVecQ4_0,
        // Sprint 52J — Q5_0 / Q5_1 / Q8_0 explicit arms. Previously
        // the catch-all `_ => Q4K` silently dispatched these to the
        // Q4_K shader → wrong dequant → NaN logits → CPU MoE router
        // `.partial_cmp().unwrap()` panic on 26B-A4B Q3_K_M GGUF.
        (GgmlType::Q5_0, true ) => ShaderId::MulMatVecQ5_0Subgroup,
        (GgmlType::Q5_0, false) => ShaderId::MulMatVecQ5_0,
        (GgmlType::Q5_1, true ) => ShaderId::MulMatVecQ5_1Subgroup,
        (GgmlType::Q5_1, false) => ShaderId::MulMatVecQ5_1,
        (GgmlType::Q8_0, true ) => ShaderId::MulMatVecQ8_0Subgroup,
        (GgmlType::Q8_0, false) => ShaderId::MulMatVecQ8_0,
        // Sprint 52J — hard error on truly unknown quants. Replaces
        // the prior silent Q4_K fallthrough so future unsupported
        // ggml types surface as an explicit panic at shader-pick time
        // rather than silently producing NaN math.
        (other, _) => panic!(
            "gemv_shader_for: no GEMV pipeline for ggml_type {other:?}; \
             add a `mul_mat_vec.comp` ShaderJob + ShaderId arm",
        ),
    }
}

/// Sprint 56C-2 — Per-layer wrapper around `gemv_indexed_shader_for`.
/// Looks up the tensor's `ggml_type` and dispatches to the matching
/// `MulMatVec*Id` pipeline.
pub(crate) fn layer_weight_indexed_shader(
    model: &LoadedModel,
    layer: u32,
    suffix: &str,
    subgroup: bool,
) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    let ggml_type = model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type;
    gemv_indexed_shader_for(ggml_type, subgroup)
}

/// Sprint 56C-2 — Indexed GEMV variant selector for GPU-direct MoE
/// expert FFN. Maps a quantization type + subgroup capability to the
/// `MulMatVec*Id` pipeline registered in Sprint 56C-1. Only the three
/// quants currently used by Gemma-4-26B-A4B's expert tensors are
/// supported (Q3_K gate_up on GGUF, Q4_K on SafeTensors, Q5_0 down on
/// GGUF). Other quants panic to surface dispatch-pick mistakes early.
pub(crate) fn gemv_indexed_shader_for(ggml_type: GgmlType, subgroup: bool) -> ShaderId {
    match (ggml_type, subgroup) {
        (GgmlType::Q3K, true ) => ShaderId::MulMatVecQ3KIdSubgroup,
        (GgmlType::Q3K, false) => ShaderId::MulMatVecQ3KId,
        (GgmlType::Q4K, true ) => ShaderId::MulMatVecQ4KIdSubgroup,
        (GgmlType::Q4K, false) => ShaderId::MulMatVecQ4KId,
        (GgmlType::Q5_0, true ) => ShaderId::MulMatVecQ5_0IdSubgroup,
        (GgmlType::Q5_0, false) => ShaderId::MulMatVecQ5_0Id,
        (GgmlType::Q4_0, true ) => ShaderId::MulMatVecQ4_0IdSubgroup,
        (GgmlType::Q4_0, false) => ShaderId::MulMatVecQ4_0Id,
        (other, _) => panic!(
            "gemv_indexed_shader_for: no indexed GEMV pipeline for \
             ggml_type {other:?}; add a `MulMatVec*Id` build.rs ShaderJob \
             + ShaderId arm",
        ),
    }
}

/// Sprint 61C — Phase 2' MMQ_ID variant selector. Mirror of
/// `gemv_indexed_shader_for` but returns the `mul_mmq.comp + MUL_MAT_ID`
/// pipelines registered in Sprint 61B. Used by
/// `b_step_moe_expert_ffn_grouped` to pick gate_up / down kernels.
///
/// Subgroup variant is preferred on RDNA4 (wave-ballot fast path), with
/// the stock variant available as a fallback under
/// `VF_MOE_GROUPED_SUBGROUP=0` (Sprint 61B build proved both compile).
pub(crate) fn mmq_id_shader_for(ggml_type: GgmlType, subgroup: bool) -> ShaderId {
    match (ggml_type, subgroup) {
        (GgmlType::Q3K, true ) => ShaderId::MulMmqQ3KMatIdSubgroup,
        (GgmlType::Q3K, false) => ShaderId::MulMmqQ3KMatId,
        (GgmlType::Q4K, true ) => ShaderId::MulMmqQ4KMatIdSubgroup,
        (GgmlType::Q4K, false) => ShaderId::MulMmqQ4KMatId,
        (GgmlType::Q5_0, true ) => ShaderId::MulMmqQ5_0MatIdSubgroup,
        (GgmlType::Q5_0, false) => ShaderId::MulMmqQ5_0MatId,
        (GgmlType::Q4_0, true ) => ShaderId::MulMmqQ4_0MatIdSubgroup,
        (GgmlType::Q4_0, false) => ShaderId::MulMmqQ4_0MatId,
        (other, _) => panic!(
            "mmq_id_shader_for: no MMQ_ID pipeline for ggml_type {other:?}; \
             add a `MulMmq*MatId` build.rs ShaderJob + ShaderId arm",
        ),
    }
}

/// Sprint 61C — model-driven MMQ_ID shader selector (mirror of
/// `layer_weight_indexed_shader`).
pub(crate) fn layer_weight_mmq_id_shader(
    model: &LoadedModel,
    layer: u32,
    suffix: &str,
    subgroup: bool,
) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    let ggml_type = model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type;
    mmq_id_shader_for(ggml_type, subgroup)
}

pub(crate) fn compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

/// Sprint 3C — sync a TRANSFER (e.g. `cmd_fill_buffer`) so its writes
/// are visible to the next compute shader read.
pub(crate) fn transfer_to_compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}
