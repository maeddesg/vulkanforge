//! Sprint 44C-1 — `LayerStep` enum + per-architecture plan builders.
//!
//! ## Why this file exists
//!
//! Sprint 43D-4 shipped a coherence fix that had to land in three
//! parallel forward paths (`dispatch_layer`, `dispatch_layer_batch`,
//! `dispatch_layer_partial`). The `feedback_layer_dispatch_paths`
//! memory was created because that fix initially missed the
//! prefill path, silently shipping broken Gemma-4 prefill output.
//!
//! 44C makes that bug class structurally impossible. The path
//! gets factored into:
//!
//! 1. **Plan** (this file): a `LayerPlan = Vec<LayerStep>` that
//!    enumerates every per-layer operation.  The builder is the
//!    *only* place a step's presence is decided.
//! 2. **Executors** (44C-2): one `impl LayerExecutor for {Decode,
//!    Batch, Debug}Exec` block per dispatch path.  Each `match
//!    LayerStep` is exhaustive — adding a variant requires every
//!    executor to handle it before the crate compiles.
//!
//! This file is 44C-1: the data definitions + builders + tests.
//! The executors arrive in 44C-2.  Until then nothing in the
//! existing dispatch code calls into this module — it's purely
//! additive.
//!
//! ## Plan vs dispatch decisions
//!
//! Each variant captures a *logical* operation, not a shader call.
//! The builder picks variants based on:
//! - the model architecture (Llama/Qwen/Mistral via
//!   `build_qwen3_layer`, Gemma-4 via `build_gemma4_layer`),
//! - per-layer config (per-layer head_dim / RoPE / KV-source for
//!   Gemma-4),
//! - the `LayerWeightFlags` lookup (which optional weights a layer
//!   actually has — Q/K/V biases for Qwen2; PLE tensors for
//!   Gemma-4).
//!
//! Executors translate variants to dispatches:
//! - `QProj` → `run_gemv` (decode) or `run_gemm` / `run_gemm_fp8_*`
//!   (batch).
//! - `AttnNorm` → `run_rms_norm` with `n_rows = 1` (decode) or
//!   `n_rows = seq_len` (batch).
//! - `Attention` → `run_scalar_attn` (decode, per-token) or
//!   `run_flash_attn_batch` / `run_flash_attn_tiled` (batch).
//!
//! Cross-layer optimisations (Sprint 9b.2 fused multi-add-rms with
//! the next layer's attn-norm in batch mode; the `batch_norm` seed
//! in `prefill_batch::record_prefill_seed`) are *executor-level* —
//! they don't change the plan.

// Sprint 44C-1 ships data + builders; the executors (44C-2) will be
// the first production callers. Suppress dead-code warnings until then.
#![allow(dead_code)]

use super::super::gguf::{Gemma4KvSource, Gemma4LayerKind, ModelConfig};
use super::super::loader::LoadedModel;

/// One operation inside a single layer's forward pass.
///
/// Each variant maps 1:1 to either a single `run_*` dispatch or a
/// fixed group of dispatches that always run together (e.g. `PleBlock`
/// is 5 dispatches that share scratch-buffer aliasing).
#[derive(Debug, Clone, PartialEq)]
pub enum LayerStep {
    /// Pre-attention RMSNorm (`input_layernorm` in HF).
    /// Decode: `run_rms_norm(input → hidden_norm)`.
    /// Batch: pre-seeded by `record_prefill_seed` for layer 0 or by
    ///        the previous layer's residual fusion (Sprint 9b.2);
    ///        the batch executor is allowed to detect this and emit
    ///        nothing.
    AttnNorm,

    /// Q linear projection. Always present.
    QProj,

    /// K linear projection. Skipped for Gemma-4 subscriber layers
    /// (they read K from a publisher's slab via `Attention.kv_layer`).
    KProj,

    /// V linear projection. Skipped alongside `KProj` for Gemma-4
    /// subscribers, and replaced by `VFromKRaw` on Gemma-4-26B-A4B
    /// full-attention layers (where `attention_k_eq_v: true`).
    VProj,

    /// Sprint 51B — Gemma-4-26B-A4B full-attention layers under
    /// `attention_k_eq_v: true` derive V from K's raw projection
    /// instead of running their own `v_proj`. Executor copies
    /// `k_buf` (raw, pre-norm, pre-RoPE) into `v_buf` so the
    /// downstream `VNorm` (parameterless RMSNorm) can run on V
    /// independently of K's own `KNormRope` chain.
    /// Emitted instead of `VProj` when the layer has
    /// `Gemma4LayerSpec.has_v_proj == false`.
    VFromKRaw,

    /// Q bias add (Qwen2 / DeepSeek-V2 attention biases). Builder
    /// only emits if `LayerWeightFlags::has_q_bias` is set.
    QBiasAdd,
    /// K bias add. Skipped for Gemma-4 subscribers (they have no
    /// own K to bias-add).
    KBiasAdd,
    /// V bias add. Skipped alongside `KBiasAdd`.
    VBiasAdd,

    /// Fused Q-RMSNorm + RoPE-NeoX. Emitted when `cfg.has_qk_norm`
    /// (Qwen3 has Q/K-norm; Llama / Mistral do not).
    QNormRope { rotary_dim: u32, freq_base: f32, theta_scale: f32 },
    /// Fused K-RMSNorm + RoPE-NeoX. Skipped for Gemma-4 subscribers.
    KNormRope { rotary_dim: u32, freq_base: f32, theta_scale: f32 },

    /// Plain Q-RoPE-NeoX. Emitted when `!cfg.has_qk_norm`.
    QRope { rotary_dim: u32, freq_base: f32, theta_scale: f32 },
    /// Plain K-RoPE-NeoX. Skipped for Gemma-4 subscribers.
    KRope { rotary_dim: u32, freq_base: f32, theta_scale: f32 },

    /// Parameterless V-RMSNorm (Gemma-4 attention V passes through
    /// `Gemma4RMSNorm(with_scale=False)`). Skipped for subscribers.
    VNorm,

    /// Write Q-rotated K + V to the per-layer KV-cache slab.
    /// Decode: per-token at `position`. Batch: bulk M rows starting
    /// at `base_pos`. Skipped for Gemma-4 subscribers.
    KvWrite,

    /// Causal scaled-dot-product attention.
    /// `kv_layer` = which layer's KV-slab to read (Gemma-4 KV-share:
    /// subscribers read a publisher's slab; everyone else reads
    /// their own).
    /// `kv_start` = sliding-window lower bound (Gemma-4 sliding
    /// layers attend to `[kv_start, position+1)` rather than
    /// `[0, position+1)`; non-sliding layers and non-Gemma-4 stacks
    /// always pass `0`).
    Attention { kv_layer: u32, kv_start: u32 },

    /// Output projection (`attn_output.weight`).
    OProj,

    /// Gemma-4 only: `o_normed = rms_norm(o) * post_attention_layernorm.weight`.
    /// Llama / Qwen3 plans don't emit this.
    PostAttnNorm,

    /// `res1 = input + (o or post_attn_normed_o)`. Always present.
    /// The Llama-path executor may fuse this with the following
    /// `PreFfnNorm` into a single `multi_add_rms` dispatch — that
    /// fusion is internal to the executor, not a plan variant.
    AttnResidualAdd,

    /// `hidden_norm = rms_norm(res1) * (ffn_norm.weight | pre_ffn_norm.weight)`.
    /// Llama / Qwen3 use `ffn_norm.weight`; Gemma-4 uses the new
    /// `pre_ffn_norm.weight` introduced in Sprint 43B-1.
    PreFfnNorm,

    /// FFN gate linear projection.
    GateProj,
    /// FFN up linear projection.
    UpProj,

    /// Activation-gated linear unit. `kind` selects the activation;
    /// the GLU multiply with `up` is part of the same shader.
    Activation { kind: ActivationKind },

    /// FFN down projection.
    DownProj,

    /// Gemma-4 only: `ffn_out_normed = rms_norm(ffn_out) * post_ffn_norm.weight`.
    PostFfnNorm,

    /// `output = res1 + (ffn_out or post_ffn_normed_ffn_out)`. Always
    /// present. The batch executor's Llama path may fuse this with
    /// the *next* layer's `AttnNorm` via `multi_add_rms`; the Gemma-4
    /// batch executor emits a separate seed RMSNorm after
    /// `LayerScalarMul`. Both are executor-level fusion choices.
    FfnResidualAdd,

    /// Gemma-4 Per-Layer Embedding integration block. Emitted only
    /// when `LayerWeightFlags::has_ple`. Internally 5 dispatches:
    /// gate-GEMV → GELU-GLU → projection-GEMV → RMSNorm → add.
    PleBlock,

    /// Final per-layer scalar multiply (`Gemma4TextDecoderLayer.layer_scalar`).
    /// The scalar buffer is `blk.{layer}.layer_scalar`.
    LayerScalarMul,

    // ============================================================
    // Sprint 51C — Gemma-4-26B-A4B MoE block (`enable_moe_block: true`).
    //
    // Each layer runs Dense-MLP (existing PreFfnNorm / GateProj /
    // UpProj / Activation / DownProj) AND a parallel MoE branch on
    // the same residual; their outputs sum and the existing
    // PostFfnNorm / FfnResidualAdd / LayerScalarMul finish the layer.
    //
    // The 6 variants below cover everything the Dense-MLP path
    // doesn't already provide. Implementations come in Sprint 51D
    // (`todo!()`-stubbed in both executors here).
    //
    // Buffer aliasing on the existing IntermediateSlot:
    //   ffn_out      — Dense-MLP DownProj output (lives until
    //                  PostDenseMlpNorm reads it; later reused for h2)
    //   scratch_a    — h1 (PostDenseMlpNorm output, lives until MoeBranchAdd)
    //   scratch_b    — MoE input (PreMoeNorm output of residual)
    //   ffn_hidden   — MoE expert weighted sum (Dense path is done with it)
    //   ffn_out      — h2 after PostMoeNorm (alias of Dense output slot)
    //   ffn_out      — h1+h2 after MoeBranchAdd (in-place add)
    // No new IntermediateSlot fields needed.
    // ============================================================

    /// Apply `post_feedforward_layernorm_1` to the Dense-MLP output.
    /// Reads `ffn_out`, writes `scratch_a` (= h1).
    PostDenseMlpNorm,

    /// Apply `pre_feedforward_layernorm_2` to the post-attention
    /// residual (the same one the Dense MLP read pre-`PreFfnNorm`).
    /// Reads `res1`, writes `scratch_b` (= MoE input).
    /// **Critical:** input is the residual, NOT the Dense MLP output.
    PreMoeNorm,

    /// MoE Router: `router.proj` GEMV [`hidden_size`→`n_experts`],
    /// followed by softmax + Top-K. CPU-side selection of the K
    /// active experts and their normalised weights.
    /// Reads `res1` / `batch_residual` (the RAW post-attention
    /// residual). The router internally does its own parameterless
    /// RMS-norm + per-channel `scale × inv_sqrt(hidden_size)`; passing
    /// the pre_ff_norm_2-multiplied residual would distort direction.
    /// Fills the router-scratch with `[token_idx → (top_k_indices[K],
    /// top_k_weights[K])]`. Sprint 51D-F: ordered BEFORE PreMoeNorm.
    MoeRoute { n_experts: u32, top_k: u32 },

    /// MoE Expert FFN: dispatch the K active experts per token and
    /// accumulate `weight_e × (down_proj(act(gate_e × up_proj(x))))`
    /// into `ffn_hidden`. Each expert is a `[moe_intermediate, hidden]`
    /// linear pair packed in `moe_experts.gate_up_proj` /
    /// `moe_experts.down_proj` (3D tensors over all 128 experts).
    MoeExpertFfn {
        n_experts: u32,
        top_k: u32,
        moe_intermediate: u32,
    },

    /// Apply `post_feedforward_layernorm_2` to the MoE branch output.
    /// Reads `ffn_hidden`, writes `ffn_out` (= h2, aliasing the
    /// freed Dense-MLP slot).
    PostMoeNorm,

    /// Sum the two parallel branches: `ffn_out = scratch_a + ffn_out`
    /// (h1 + h2). Distinct from `FfnResidualAdd` (which adds residual).
    MoeBranchAdd,
}

/// Activation flavour for the FFN gated-linear-unit shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// SiLU(gate) * up — Llama / Qwen.
    SwiGlu,
    /// PyTorch tanh-approx GELU(gate) * up — Gemma-4
    /// (`hidden_activation == "gelu_pytorch_tanh"`).
    GeluPytorchTanhGlu,
}

/// Ordered list of operations for a single layer's forward pass.
///
/// `LayerPlan[N]` is the plan for layer N. It contains exactly the
/// steps that layer should run — no implicit skips. If a step is in
/// the plan, every executor MUST dispatch it (their `match LayerStep`
/// is exhaustive).
pub type LayerPlan = Vec<LayerStep>;

/// Per-layer optional-weight presence flags. The builder cannot
/// query `LoadedModel` directly (we want it to be unit-testable
/// without a model), so the dispatcher wrapper extracts these and
/// passes them in.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerWeightFlags {
    /// `blk.{layer}.attn_q.bias` exists (Qwen2 / DeepSeek-V2).
    pub has_q_bias: bool,
    /// `blk.{layer}.attn_k.bias` exists.
    pub has_k_bias: bool,
    /// `blk.{layer}.attn_v.bias` exists.
    pub has_v_bias: bool,
    /// `model.ple_data.is_some()` — Gemma-4 with PLE weights loaded.
    /// Note: PLE is global to the model, not per-layer, but the flag
    /// still lives here for API symmetry.
    pub has_ple: bool,
}

/// Build the layer plan for a Llama / Qwen / Qwen3 / Mistral / DeepSeek
/// stack. Branches on `cfg.has_qk_norm` to pick fused vs plain RoPE.
pub fn build_qwen3_layer(
    cfg: &ModelConfig,
    layer: u32,
    flags: &LayerWeightFlags,
    rope_theta_scale_default: f32,
) -> LayerPlan {
    debug_assert!(cfg.gemma4.is_none(), "build_qwen3_layer called on Gemma-4 cfg");
    let mut plan: LayerPlan = Vec::with_capacity(20);

    // Standard transformer: every layer owns its KV.
    plan.push(LayerStep::AttnNorm);
    plan.push(LayerStep::QProj);
    plan.push(LayerStep::KProj);
    plan.push(LayerStep::VProj);

    if flags.has_q_bias { plan.push(LayerStep::QBiasAdd); }
    if flags.has_k_bias { plan.push(LayerStep::KBiasAdd); }
    if flags.has_v_bias { plan.push(LayerStep::VBiasAdd); }

    // Non-Gemma-4 RoPE: uniform head_dim and freq_base.
    let rotary_dim = cfg.head_dim;
    let freq_base = cfg.rope_freq_base;
    let theta_scale = rope_theta_scale_default;
    if cfg.has_qk_norm {
        plan.push(LayerStep::QNormRope { rotary_dim, freq_base, theta_scale });
        plan.push(LayerStep::KNormRope { rotary_dim, freq_base, theta_scale });
    } else {
        plan.push(LayerStep::QRope { rotary_dim, freq_base, theta_scale });
        plan.push(LayerStep::KRope { rotary_dim, freq_base, theta_scale });
    }

    plan.push(LayerStep::KvWrite);
    plan.push(LayerStep::Attention {
        kv_layer: layer,
        kv_start: 0,
    });
    plan.push(LayerStep::OProj);
    plan.push(LayerStep::AttnResidualAdd);
    plan.push(LayerStep::PreFfnNorm);
    plan.push(LayerStep::GateProj);
    plan.push(LayerStep::UpProj);
    plan.push(LayerStep::Activation { kind: ActivationKind::SwiGlu });
    plan.push(LayerStep::DownProj);
    plan.push(LayerStep::FfnResidualAdd);

    plan
}

/// Build the layer plan for a Gemma-4 stack. Encodes:
/// - per-layer head_dim + RoPE θ + p-RoPE rotary_dim,
/// - 4-norm structure (`PostAttnNorm`, `PreFfnNorm`, `PostFfnNorm` plus
///   the standard `AttnNorm`),
/// - parameterless `VNorm`,
/// - KV-share: subscribers omit `KProj` / `VProj` / `KNormRope` /
///   `KRope` / `VNorm` / `KvWrite`,
/// - sliding-window lower bound passed through `Attention.kv_start`,
/// - PLE 5-op block (when `has_ple`),
/// - per-layer scalar multiply.
pub fn build_gemma4_layer(
    cfg: &ModelConfig,
    layer: u32,
    flags: &LayerWeightFlags,
    _rope_theta_scale_default: f32,
) -> LayerPlan {
    let g = cfg
        .gemma4
        .as_ref()
        .expect("build_gemma4_layer called on non-Gemma-4 cfg");
    let s = &g.layers[layer as usize];

    let owns_kv = !matches!(
        s.kv_source,
        Gemma4KvSource::SubscribesSliding | Gemma4KvSource::SubscribesFull
    );

    // Per-layer RoPE params (mirrors `arch::gemma4::rope_params_for_layer`
    // without the `VF_DISABLE_PROPE` bisect override — the bisect lives
    // in the executor's lookup function so dev tooling stays orthogonal
    // to the plan).
    let rotary_dim = match s.rope_partial_factor {
        Some(f) => ((s.head_dim as f32) * f).round() as u32,
        None => s.head_dim,
    };
    let freq_base = s.rope_theta;
    let theta_scale = (1.0_f32 / freq_base).powf(2.0 / rotary_dim as f32);

    // KV-share: which layer's slab does this layer read from?
    let kv_layer = match s.kv_source {
        Gemma4KvSource::Own
        | Gemma4KvSource::OwnAndPublishesSliding
        | Gemma4KvSource::OwnAndPublishesFull => layer,
        Gemma4KvSource::SubscribesSliding => g
            .layers
            .iter()
            .enumerate()
            .find(|(_, ls)| matches!(ls.kv_source, Gemma4KvSource::OwnAndPublishesSliding))
            .map(|(i, _)| i as u32)
            .unwrap_or(layer),
        Gemma4KvSource::SubscribesFull => g
            .layers
            .iter()
            .enumerate()
            .find(|(_, ls)| matches!(ls.kv_source, Gemma4KvSource::OwnAndPublishesFull))
            .map(|(i, _)| i as u32)
            .unwrap_or(layer),
    };

    // Sliding-window lower bound. The plan stores the *layer-static*
    // contribution: the `position`-dependent `seq_len.saturating_sub(window)`
    // is computed at dispatch time. We encode the static decision
    // ("does this layer use a sliding window?") via a sentinel:
    //   - `kv_start = 0` for full-attention layers and non-Gemma-4
    //     (full causal history, no window).
    //   - `kv_start = sliding_window` for sliding layers (the
    //     executor reads `kv_start = max(0, position+1 - window)` at
    //     dispatch time using this as the window size).
    // This split keeps the plan position-independent.
    let kv_start = match s.kind {
        Gemma4LayerKind::Sliding => g.sliding_window,
        Gemma4LayerKind::Full => 0,
    };

    let mut plan: LayerPlan = Vec::with_capacity(28);
    plan.push(LayerStep::AttnNorm);
    plan.push(LayerStep::QProj);
    if owns_kv {
        plan.push(LayerStep::KProj);
        // Sprint 51B — Gemma-4-26B-A4B full-attention layers under
        // `attention_k_eq_v: true` skip the v_proj weight; V is
        // derived from K's raw projection. Emit `VFromKRaw` BEFORE
        // any norm/RoPE on K so v_buf gets the unrotated values.
        if s.has_v_proj {
            plan.push(LayerStep::VProj);
        } else {
            plan.push(LayerStep::VFromKRaw);
        }
    }
    if flags.has_q_bias { plan.push(LayerStep::QBiasAdd); }
    if owns_kv && flags.has_k_bias { plan.push(LayerStep::KBiasAdd); }
    if owns_kv && s.has_v_proj && flags.has_v_bias { plan.push(LayerStep::VBiasAdd); }

    // Gemma-4 has Q/K-norm in the reference HF model? — check
    // cfg.has_qk_norm. Currently Gemma-4-E2B has has_qk_norm=true
    // (Q-norm and K-norm weights live at attn_q_norm.weight / attn_k_norm.weight).
    if cfg.has_qk_norm {
        plan.push(LayerStep::QNormRope { rotary_dim, freq_base, theta_scale });
        if owns_kv {
            plan.push(LayerStep::KNormRope { rotary_dim, freq_base, theta_scale });
        }
    } else {
        plan.push(LayerStep::QRope { rotary_dim, freq_base, theta_scale });
        if owns_kv {
            plan.push(LayerStep::KRope { rotary_dim, freq_base, theta_scale });
        }
    }

    if owns_kv {
        plan.push(LayerStep::VNorm);
        plan.push(LayerStep::KvWrite);
    }

    plan.push(LayerStep::Attention { kv_layer, kv_start });
    plan.push(LayerStep::OProj);

    plan.push(LayerStep::PostAttnNorm);
    plan.push(LayerStep::AttnResidualAdd);

    // FFN block. E2B: Dense-MLP only.
    // 26B-A4B: Dense-MLP AND MoE in parallel, summed before
    // `PostFfnNorm`/`FfnResidualAdd`/`LayerScalarMul`.
    plan.push(LayerStep::PreFfnNorm);
    plan.push(LayerStep::GateProj);
    plan.push(LayerStep::UpProj);
    plan.push(LayerStep::Activation { kind: ActivationKind::GeluPytorchTanhGlu });
    plan.push(LayerStep::DownProj);

    if g.enable_moe_block {
        // Branch 1 close-out (h1): PostDenseMlpNorm reads ffn_out,
        // writes scratch_a.
        plan.push(LayerStep::PostDenseMlpNorm);
        // Branch 2 (parallel on residual). Sprint 51D-F — order matters:
        // HF `Gemma4TextDecoderLayer.forward` calls `self.router(residual)`
        // on the RAW post-attention residual (before pre_ff_norm_2),
        // because `Gemma4TextRouter` does its own parameterless
        // RMS-norm internally. Only the experts get
        // `pre_ff_norm_2(residual)` as input. Routing on the
        // pre_ff_norm_2-multiplied residual distorts the direction
        // (per-channel γ_2 ≠ uniform → wrong Top-K).
        //
        //   MoeRoute    reads res1 / batch_residual (raw)
        //   PreMoeNorm  res1 → scratch_b / batch_o (= pre_ff_norm_2(res1))
        //   MoeExpertFfn reads scratch_b / batch_o
        plan.push(LayerStep::MoeRoute {
            n_experts: g.n_experts,
            top_k: g.top_k_experts,
        });
        plan.push(LayerStep::PreMoeNorm);
        plan.push(LayerStep::MoeExpertFfn {
            n_experts: g.n_experts,
            top_k: g.top_k_experts,
            moe_intermediate: g.moe_intermediate_size,
        });
        plan.push(LayerStep::PostMoeNorm);
        // Combine: ffn_out = scratch_a + ffn_out (h1 + h2).
        plan.push(LayerStep::MoeBranchAdd);
    }

    plan.push(LayerStep::PostFfnNorm);
    plan.push(LayerStep::FfnResidualAdd);

    if flags.has_ple {
        plan.push(LayerStep::PleBlock);
    }
    plan.push(LayerStep::LayerScalarMul);

    plan
}

/// Production dispatcher: extracts `LayerWeightFlags` from
/// `LoadedModel` and routes to the matching architecture builder.
///
/// `rope_theta_scale_default` is `Forward::rope_theta_scale` (the
/// pre-computed `(1/cfg.rope_freq_base)^(2/cfg.head_dim)` for non-
/// Gemma-4 stacks; ignored on the Gemma-4 path because per-layer
/// builders compute their own).
pub fn build_layer_plan(
    cfg: &ModelConfig,
    model: &LoadedModel,
    layer: u32,
    rope_theta_scale_default: f32,
) -> LayerPlan {
    let flags = LayerWeightFlags {
        has_q_bias: model.tensor(&format!("blk.{layer}.attn_q.bias")).is_some(),
        has_k_bias: model.tensor(&format!("blk.{layer}.attn_k.bias")).is_some(),
        has_v_bias: model.tensor(&format!("blk.{layer}.attn_v.bias")).is_some(),
        has_ple: model.ple_data.is_some(),
    };

    if cfg.gemma4.is_some() {
        build_gemma4_layer(cfg, layer, &flags, rope_theta_scale_default)
    } else {
        build_qwen3_layer(cfg, layer, &flags, rope_theta_scale_default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::gguf::{Gemma4LayerSpec, RopeVariant};

    fn qwen3_cfg(has_qk_norm: bool) -> ModelConfig {
        ModelConfig {
            architecture: "qwen3".into(),
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            hidden_dim: 4096,
            ffn_dim: 12288,
            vocab_size: 151936,
            head_dim: 128,
            rope_freq_base: 1_000_000.0,
            rope_dim: 128,
            rope_variant: RopeVariant::Neox,
            context_length: 2048,
            has_qk_norm,
            rms_norm_eps: 1e-6,
            gemma4: None,
        }
    }

    fn gemma4_e2b_spec() -> Gemma4Spec {
        // 35-layer Gemma-4-E2B-it: layers [0, 15) own KV (with 13 +
        // 14 the publishers); layers [15, 35) subscribe.  Sliding /
        // full alternation: 0-2 sliding, 3 full, 4-6 sliding, 7
        // full, ..., layer 14 full (the full publisher).
        let mut layers = Vec::with_capacity(35);
        for i in 0..35u32 {
            // Pin the publisher layout: 13 publishes sliding, 14 publishes
            // full. Other layers alternate sliding/full at i % 4.
            let kind = match i {
                13 => Gemma4LayerKind::Sliding,
                14 => Gemma4LayerKind::Full,
                _ if i % 4 == 3 => Gemma4LayerKind::Full,
                _ => Gemma4LayerKind::Sliding,
            };
            let kv_source = match (i, kind) {
                (13, _) => Gemma4KvSource::OwnAndPublishesSliding,
                (14, _) => Gemma4KvSource::OwnAndPublishesFull,
                (i, _) if i < 13 => Gemma4KvSource::Own,
                (_, Gemma4LayerKind::Sliding) => Gemma4KvSource::SubscribesSliding,
                (_, Gemma4LayerKind::Full) => Gemma4KvSource::SubscribesFull,
            };
            let head_dim = match kind {
                Gemma4LayerKind::Sliding => 256,
                Gemma4LayerKind::Full => 512,
            };
            let rope_theta = match kind {
                Gemma4LayerKind::Sliding => 10_000.0,
                Gemma4LayerKind::Full => 1_000_000.0,
            };
            let rope_partial_factor = match kind {
                Gemma4LayerKind::Full => Some(0.25),
                _ => None,
            };
            layers.push(Gemma4LayerSpec {
                kind,
                head_dim,
                intermediate_size: 6144,
                has_kv_proj: matches!(
                    kv_source,
                    Gemma4KvSource::Own
                        | Gemma4KvSource::OwnAndPublishesSliding
                        | Gemma4KvSource::OwnAndPublishesFull
                ),
                kv_source,
                rope_theta,
                rope_partial_factor,
                // Sprint 51B-pre — E2B test fixture: uniform kv_heads=1.
                n_kv_heads: 1,
                // Sprint 51B — E2B test fixture: every layer has its
                // own v_proj (`attention_k_eq_v=false`).
                has_v_proj: true,
            });
        }
        Gemma4Spec {
            sliding_window: 512,
            final_logit_softcapping: Some(30.0),
            embed_scale: 39.18,
            hidden_activation: "gelu_pytorch_tanh".into(),
            tie_word_embeddings: true,
            first_kv_shared: 15,
            layer_scalars: vec![0.5; 35],
            layers,
            hidden_size_per_layer_input: 256,
            // Sprint 51C — E2B test fixture: no MoE block.
            enable_moe_block: false,
            n_experts: 0,
            top_k_experts: 0,
            moe_intermediate_size: 0,
        }
    }

    fn gemma4_cfg() -> ModelConfig {
        ModelConfig {
            architecture: "gemma4".into(),
            n_layers: 35,
            n_heads: 8,
            n_kv_heads: 2,
            hidden_dim: 1536,
            ffn_dim: 12288,
            vocab_size: 262144,
            head_dim: 512,
            rope_freq_base: 1_000_000.0,
            rope_dim: 512,
            rope_variant: RopeVariant::Neox,
            context_length: 8192,
            has_qk_norm: true,
            rms_norm_eps: 1e-6,
            gemma4: Some(gemma4_e2b_spec()),
        }
    }

    use super::super::super::gguf::Gemma4Spec;

    fn gemma4_only_steps() -> &'static [fn(&LayerStep) -> bool] {
        &[
            |s| matches!(s, LayerStep::VNorm),
            |s| matches!(s, LayerStep::PostAttnNorm),
            |s| matches!(s, LayerStep::PostFfnNorm),
            |s| matches!(s, LayerStep::PleBlock),
            |s| matches!(s, LayerStep::LayerScalarMul),
        ]
    }

    #[test]
    fn qwen3_plan_has_no_gemma4_steps() {
        let cfg = qwen3_cfg(true);
        let flags = LayerWeightFlags::default();
        let plan = build_qwen3_layer(&cfg, 0, &flags, 1.0);
        for step in &plan {
            for is_gemma in gemma4_only_steps() {
                assert!(
                    !is_gemma(step),
                    "Qwen3 plan should not contain Gemma-4 step: {step:?}",
                );
            }
        }
    }

    #[test]
    fn qwen3_plan_orders_attn_then_ffn() {
        let cfg = qwen3_cfg(true);
        let flags = LayerWeightFlags::default();
        let plan = build_qwen3_layer(&cfg, 0, &flags, 1.0);
        let attn_norm_idx = plan.iter().position(|s| matches!(s, LayerStep::AttnNorm)).unwrap();
        let attn_idx = plan.iter().position(|s| matches!(s, LayerStep::Attention { .. })).unwrap();
        let pre_ffn_idx = plan.iter().position(|s| matches!(s, LayerStep::PreFfnNorm)).unwrap();
        let ffn_residual_idx = plan.iter().position(|s| matches!(s, LayerStep::FfnResidualAdd)).unwrap();
        assert!(attn_norm_idx < attn_idx);
        assert!(attn_idx < pre_ffn_idx);
        assert!(pre_ffn_idx < ffn_residual_idx);
    }

    #[test]
    fn qwen3_no_qk_norm_uses_plain_rope() {
        let cfg = qwen3_cfg(false);
        let flags = LayerWeightFlags::default();
        let plan = build_qwen3_layer(&cfg, 0, &flags, 1.0);
        assert!(plan.iter().any(|s| matches!(s, LayerStep::QRope { .. })));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::KRope { .. })));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::QNormRope { .. })));
    }

    #[test]
    fn qwen3_with_qk_norm_uses_fused_rope() {
        let cfg = qwen3_cfg(true);
        let flags = LayerWeightFlags::default();
        let plan = build_qwen3_layer(&cfg, 0, &flags, 1.0);
        assert!(plan.iter().any(|s| matches!(s, LayerStep::QNormRope { .. })));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::KNormRope { .. })));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::QRope { .. })));
    }

    #[test]
    fn qwen3_biases_only_when_flag_set() {
        let cfg = qwen3_cfg(true);
        let no_bias = LayerWeightFlags::default();
        let plan = build_qwen3_layer(&cfg, 0, &no_bias, 1.0);
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::QBiasAdd)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::KBiasAdd)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::VBiasAdd)));

        let with_bias = LayerWeightFlags {
            has_q_bias: true,
            has_k_bias: true,
            has_v_bias: true,
            has_ple: false,
        };
        let plan = build_qwen3_layer(&cfg, 0, &with_bias, 1.0);
        assert!(plan.iter().any(|s| matches!(s, LayerStep::QBiasAdd)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::KBiasAdd)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::VBiasAdd)));
    }

    #[test]
    fn gemma4_owner_layer_has_kv_steps() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        // Layer 0 is Own.
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        assert!(plan.iter().any(|s| matches!(s, LayerStep::KProj)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::VProj)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::VNorm)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::KvWrite)));
    }

    #[test]
    fn gemma4_subscriber_layer_skips_kv_steps() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        // Layer 15 is the first Subscribes layer.
        let plan = build_gemma4_layer(&cfg, 15, &flags, 1.0);
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::KProj)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::VProj)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::VNorm)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::KvWrite)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::KNormRope { .. })));
        // But Q-side stays.
        assert!(plan.iter().any(|s| matches!(s, LayerStep::QProj)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::QNormRope { .. })));
    }

    #[test]
    fn gemma4_subscriber_routes_attention_to_publisher() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        // Layer 16 is sliding-subscriber in our fixture → reads from
        // layer 13's slab (the OwnAndPublishesSliding publisher).
        let plan = build_gemma4_layer(&cfg, 16, &flags, 1.0);
        let attn = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::Attention { kv_layer, .. } => Some(*kv_layer),
                _ => None,
            })
            .expect("Attention step missing");
        assert_eq!(attn, 13);

        // Layer 15 is full-subscriber → reads from layer 14.
        let plan = build_gemma4_layer(&cfg, 15, &flags, 1.0);
        let attn = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::Attention { kv_layer, .. } => Some(*kv_layer),
                _ => None,
            })
            .expect("Attention step missing");
        assert_eq!(attn, 14);
    }

    #[test]
    fn gemma4_sliding_layer_carries_window() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        // Layer 0 is sliding → `kv_start = sliding_window`.
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        let kv_start = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::Attention { kv_start, .. } => Some(*kv_start),
                _ => None,
            })
            .unwrap();
        assert_eq!(kv_start, 512);

        // Layer 3 is full → kv_start stays 0.
        let plan = build_gemma4_layer(&cfg, 3, &flags, 1.0);
        let kv_start = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::Attention { kv_start, .. } => Some(*kv_start),
                _ => None,
            })
            .unwrap();
        assert_eq!(kv_start, 0);
    }

    #[test]
    fn gemma4_uses_gelu_pytorch_tanh() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        let act = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::Activation { kind } => Some(*kind),
                _ => None,
            })
            .unwrap();
        assert_eq!(act, ActivationKind::GeluPytorchTanhGlu);
    }

    #[test]
    fn gemma4_full_layer_uses_partial_rotary() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        // Layer 3 is full → rotary_dim = 0.25 * 512 = 128.
        let plan = build_gemma4_layer(&cfg, 3, &flags, 1.0);
        let rotary = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::QNormRope { rotary_dim, .. } => Some(*rotary_dim),
                LayerStep::QRope { rotary_dim, .. } => Some(*rotary_dim),
                _ => None,
            })
            .unwrap();
        assert_eq!(rotary, 128);

        // Layer 0 is sliding → rotary_dim = 256 (full sliding head_dim).
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        let rotary = plan
            .iter()
            .find_map(|s| match s {
                LayerStep::QNormRope { rotary_dim, .. } => Some(*rotary_dim),
                LayerStep::QRope { rotary_dim, .. } => Some(*rotary_dim),
                _ => None,
            })
            .unwrap();
        assert_eq!(rotary, 256);
    }

    #[test]
    fn gemma4_has_four_norms_per_layer() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        // AttnNorm, PostAttnNorm, PreFfnNorm, PostFfnNorm — all four.
        assert!(plan.iter().any(|s| matches!(s, LayerStep::AttnNorm)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::PostAttnNorm)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::PreFfnNorm)));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::PostFfnNorm)));
    }

    #[test]
    fn gemma4_layer_scalar_is_last() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        assert!(matches!(plan.last(), Some(LayerStep::LayerScalarMul)));
    }

    #[test]
    fn gemma4_no_ple_when_flag_unset() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: false, ..Default::default() };
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::PleBlock)));
        // LayerScalarMul still fires.
        assert!(plan.iter().any(|s| matches!(s, LayerStep::LayerScalarMul)));
    }

    /// Sprint 51C — E2B's Gemma4Spec has `enable_moe_block: false`,
    /// so no MoE-related step ever appears in the layer plan and
    /// the existing Dense-MLP path runs end-to-end.
    #[test]
    fn gemma4_e2b_layer_plan_has_no_moe_steps() {
        let cfg = gemma4_cfg();
        let flags = LayerWeightFlags { has_ple: true, ..Default::default() };
        for layer in 0..cfg.n_layers {
            let plan = build_gemma4_layer(&cfg, layer, &flags, 1.0);
            assert!(
                !plan.iter().any(|s| matches!(
                    s,
                    LayerStep::PostDenseMlpNorm
                        | LayerStep::PreMoeNorm
                        | LayerStep::MoeRoute { .. }
                        | LayerStep::MoeExpertFfn { .. }
                        | LayerStep::PostMoeNorm
                        | LayerStep::MoeBranchAdd
                )),
                "E2B layer {layer} unexpectedly emits an MoE step"
            );
        }
    }

    /// Sprint 51C — when `enable_moe_block: true`, the layer plan
    /// emits all six new MoE steps in the documented order
    /// (PostDenseMlpNorm → MoeRoute → PreMoeNorm → MoeExpertFfn →
    /// PostMoeNorm → MoeBranchAdd) and surrounds them with the
    /// existing PreFfnNorm-block before and PostFfnNorm-block after.
    /// Sprint 51D-F: MoeRoute moved BEFORE PreMoeNorm so the router
    /// reads the raw post-attention residual, matching HF.
    #[test]
    fn gemma4_26b_moe_layer_plan_emits_six_steps() {
        let mut spec = gemma4_e2b_spec();
        spec.enable_moe_block = true;
        spec.n_experts = 128;
        spec.top_k_experts = 8;
        spec.moe_intermediate_size = 704;
        let cfg = ModelConfig {
            architecture: "gemma4".into(),
            n_layers: 30,
            n_heads: 16,
            n_kv_heads: 8,
            hidden_dim: 2816,
            ffn_dim: 2112,
            vocab_size: 262144,
            head_dim: 512,
            rope_freq_base: 1_000_000.0,
            rope_dim: 512,
            rope_variant: RopeVariant::Neox,
            context_length: 8192,
            has_qk_norm: true,
            rms_norm_eps: 1e-6,
            gemma4: Some(spec),
        };
        let flags = LayerWeightFlags::default();
        let plan = build_gemma4_layer(&cfg, 0, &flags, 1.0);

        let names: Vec<&'static str> = plan.iter().map(|s| match s {
            LayerStep::PostDenseMlpNorm => "PostDenseMlpNorm",
            LayerStep::PreMoeNorm => "PreMoeNorm",
            LayerStep::MoeRoute { .. } => "MoeRoute",
            LayerStep::MoeExpertFfn { .. } => "MoeExpertFfn",
            LayerStep::PostMoeNorm => "PostMoeNorm",
            LayerStep::MoeBranchAdd => "MoeBranchAdd",
            _ => "",
        }).filter(|s| !s.is_empty()).collect();
        assert_eq!(
            names,
            vec![
                "PostDenseMlpNorm", "MoeRoute", "PreMoeNorm",
                "MoeExpertFfn", "PostMoeNorm", "MoeBranchAdd",
            ],
            "MoE step order mismatch"
        );
    }

    #[test]
    fn all_plans_start_with_attn_norm() {
        let qcfg = qwen3_cfg(true);
        let qflags = LayerWeightFlags::default();
        for layer in 0..qcfg.n_layers {
            let plan = build_qwen3_layer(&qcfg, layer, &qflags, 1.0);
            assert!(matches!(plan.first(), Some(LayerStep::AttnNorm)),
                    "Qwen3 layer {layer} doesn't start with AttnNorm");
        }
        let gcfg = gemma4_cfg();
        let gflags = LayerWeightFlags { has_ple: true, ..Default::default() };
        for layer in 0..gcfg.n_layers {
            let plan = build_gemma4_layer(&gcfg, layer, &gflags, 1.0);
            assert!(matches!(plan.first(), Some(LayerStep::AttnNorm)),
                    "Gemma-4 layer {layer} doesn't start with AttnNorm");
        }
    }
}
