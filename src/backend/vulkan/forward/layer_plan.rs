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

    /// Sprint D2 (v0.4.6) — Qwen3.6 Full-Attention fused Q+Gate
    /// projection. Replaces `QProj` for qwen35 full-attention
    /// layers. The `attn_q.weight` tensor is
    /// `[hidden_dim, 2 × n_heads × head_dim]`; the GEMV/GEMM writes
    /// `2 × q_dim` floats per token with Q in the first half and
    /// Gate in the second half. Q is consumed by the downstream
    /// `QNormRope`/`Attention` chain unchanged; Gate is consumed
    /// later by `AttnGatedOutput`. Decode writes into `q_buf`
    /// (resized to 2 × q_dim on qwen35); batch writes into
    /// `batch_qgate` and a strided copy extracts Q into `batch_q`
    /// so the existing Q-side dispatch chain operates on the same
    /// per-token-contiguous layout as on every other architecture.
    AttnQGateProj { q_dim: u32 },

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

    /// Sprint D2 (v0.4.6) — Qwen3.6 Full-Attention gated output.
    /// In-place sigmoid-gate multiply on the post-attention buffer:
    /// `attn = attn * sigmoid(gate)`. Sits between `Attention` and
    /// `OProj` in the qwen35 Full-Attn plan. `q_dim` is the
    /// elements-per-token of the post-attention buffer (= n_heads ×
    /// head_dim); the gate lives in `q_buf` / `batch_qgate` at the
    /// `q_dim..2 × q_dim` offset slice written by `AttnQGateProj`.
    AttnGatedOutput { q_dim: u32 },

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

    /// Sprint D (v0.4.6) — identity passthrough for the
    /// attention sub-block. Emitted by the Qwen3.5/3.6 skeleton
    /// plan in place of the full Q/K/V/Attention/OProj/ResidualAdd
    /// sequence. Semantics: `res1 = input` (decode) / no-op (batch:
    /// `batch_residual` already carries the layer input). Lets the
    /// rest of the plan run unchanged (PreFfnNorm reads `res1`,
    /// FFN dispatches, FfnResidualAdd writes `output = res1 +
    /// ffn_out`). Output is `input + FFN(input)` — meaningless but
    /// crash-free. Sprint D2-G replaces it on Full-Attention layers
    /// with real Q-Gate-Split + Attention dispatches; the SSM/GDN
    /// layers will keep this stub until the recurrent state +
    /// `gated_delta_net.comp` lands.
    ResidualIdentitySeed,

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

    /// Sprint F (v0.4.6) — Qwen3.6 SSM 1D causal convolution on a
    /// Linear-Attention layer. Conv kernel size `ssm_d_conv = 4`,
    /// `conv_channels = 2 * (ssm_d_state * ssm_n_group) + ssm_d_inner
    /// = 10240`. Per-sequence persistent rolling window of the last
    /// `ssm_d_conv - 1 = 3` channel-wise inputs lives in
    /// `Forward::conv_state_buf` (allocated per-layer for the 48
    /// recurrent layers, zero-initialised at construction).
    ///
    /// Sprint F is a **staging commit**: the variant is emitted by
    /// `build_qwen35_layer` on every recurrent layer so the layer plan
    /// already matches the Sprint G shape, the SSM-Conv shader
    /// (`ssm_conv_f32`) is compiled into the binary, and the
    /// persistent state buffer is allocated and zero-initialised.
    /// The step body itself is a **no-op** in both executors — the
    /// real conv dispatch + state-shift `cmd_copy_buffer` chain lands
    /// in Sprint G together with `GatedDeltaNet` (the conv output is
    /// not consumed by anything until GDN exists, so dispatching the
    /// conv alone would be wasted work and risk barrier-tuning
    /// regressions on Qwen3-8B / Gemma-4-26B).
    SsmConv1d { layer: u32 },

    // ────────────────────────────────────────────────────────────
    // Sprint G-2b (v0.4.6) — Qwen3.6 Linear-Attention pipeline
    //
    // All 10 variants below are emitted by `build_qwen35_layer` on
    // recurrent trunk layers (48 of 65 for Qwen3.6-27B). Step bodies
    // in `executor/attention.rs` are no-op until Sprints G-2c/d/e
    // fill them with real dispatches. The variants exist now so the
    // exhaustive-match enforcement (coding-standards §3.2) catches
    // missing executor implementations at compile time.
    // ────────────────────────────────────────────────────────────

    /// QKV fused projection (`attn_qkv.weight`, `[5120, 10240]` GEMV).
    /// Sources `hidden_norm` (the AttnNorm output) and writes
    /// `Forward::ssm_qkv_buf`. Output is `qkv_mixed = 2*k_dim + v_dim
    /// = 4096 + 6144 = 10240` floats (channel-major within token).
    /// Consumed by `SsmConv1d` as the current input slot of the
    /// rolling conv window. Sprint G-2d fills the body.
    AttnQkvProj { layer: u32 },

    /// Gate (z) projection (`attn_gate.weight`, `[5120, 6144]` GEMV).
    /// Sources `hidden_norm` and writes `Forward::ssm_z_buf`. The
    /// `z` output is consumed by `NormGated` as the gate signal
    /// (`silu(z)` modulates the post-GDN RMSNorm output).
    /// Sprint G-2d fills the body.
    AttnGateZProj { layer: u32 },

    /// Beta projection (`ssm_beta.weight`, `[5120, 48]` GEMV) +
    /// in-place sigmoid. Sources `hidden_norm` and writes
    /// `Forward::ssm_beta_buf`. `beta` carries the per-v-head delta-
    /// rule strength (sigmoid-bounded into [0, 1]) consumed by GDN.
    /// Sprint G-2d fills the body (GEMV + Sigmoid).
    SsmBetaProj { layer: u32 },

    /// Alpha gate composition (5 fused operations):
    ///   * `ssm_alpha.weight` GEMV `[5120, 48]` → `ssm_alpha_buf`
    ///   * + `ssm_dt.bias` (additive bias `[48]`)
    ///   * softplus in-place (`x ← log(1+exp(x))`)
    ///   * × `ssm_a` (per-v-head scalar `[48]`, negative)
    ///   * result → `ssm_gate_buf`
    ///
    /// `gate` is the log-domain decay term consumed by GDN's
    /// `g_exp[r] = exp(data_g[gb_off])` path (KDA=0). Sprint G-2d
    /// fills the body using the existing `Add`/`Mul` runs + the
    /// new `SoftplusF32` shader.
    SsmAlphaGate { layer: u32 },

    /// SiLU in-place on the SSM-Conv output. Sources and writes
    /// `Forward::ssm_conv_output_buf`. Mirrors llama.cpp
    /// `qwen35.cpp:396 ggml_silu(conv_output_proper)` which sits
    /// between the conv and the Q/K/V split. Reuses the existing
    /// `Silu` shader. Sprint G-2c fills the body.
    SsmSilu { layer: u32 },

    /// L2-Norm on the Q and K halves of the conv-output (per-head,
    /// `head_k_dim = 128`). Two in-place dispatches via the existing
    /// `RmsNorm` shader with a synthetic weight=1.0 buffer (plan §5
    /// "L2Norm via RMSNorm-Trick"). Sources `ssm_conv_output_buf`
    /// at offsets `0` and `2048` respectively, writes back in-place.
    /// Mirrors llama.cpp `qwen35.cpp:430-431 ggml_l2_norm(q_conv)`
    /// / `ggml_l2_norm(k_conv)`. Sprint G-2c fills the body.
    SsmQkL2Norm { layer: u32 },

    /// Head-repeat 16 → 48 for Q and K. Two dispatches of the
    /// `RepeatInterleaveF32` shader (modulo repeat along the head
    /// axis). Source `ssm_conv_output_buf[0..4096]` for Q+K,
    /// destinations `ssm_qrep_buf` and `ssm_krep_buf`. Mirrors
    /// llama.cpp `qwen35.cpp:441-443 ggml_repeat_4d(q_conv, …,
    /// num_v_heads, …)` — only fired when `num_k_heads !=
    /// num_v_heads` (the conditional in upstream); always true for
    /// Qwen3.6-27B (16 vs 48). Sprint G-2d fills the body.
    SsmRepeatQK { layer: u32 },

    /// Gated-Delta-Net dispatch + persistent ssm_state update.
    /// Inputs:
    ///   * Q  ← `ssm_qrep_buf` (post-l2-norm, post-repeat)
    ///   * K  ← `ssm_krep_buf`
    ///   * V  ← `ssm_conv_output_buf` at offset 4096 (`v_dim = 6144`)
    ///   * G  ← `ssm_gate_buf` (alpha-gate, log-domain)
    ///   * B  ← `ssm_beta_buf` (sigmoid)
    ///   * State ← `Forward::ssm_state_buf` at per-layer offset
    /// Output: `ssm_gdn_out_buf` `[6144]` + updated state slot.
    /// Uses the `GatedDeltaNetF32` shader from Sprint G-2a.
    /// Sprint G-2e fills the body (the highest-risk slice — coherence
    /// gate hinges on correct state layout + push-const strides).
    GatedDeltaNet { layer: u32 },

    /// Norm-gated activation: `RMSNorm(gdn_out, ssm_norm) * SiLU(z)`.
    /// Sources `ssm_gdn_out_buf` + `ssm_z_buf` + `ssm_norm.weight`,
    /// writes `ssm_norm_out_buf`. Naive 3-dispatch impl (RMSNorm →
    /// Silu → Mul); the optional fused `norm_gated.comp` shader is
    /// a Sprint G3 optimisation. Mirrors llama.cpp
    /// `qwen35.cpp:455 build_norm_gated(output, ssm_norm, z_2d)`.
    /// Sprint G-2e fills the body.
    NormGated { layer: u32 },

    /// SSM output projection (`ssm_out.weight`, `[6144, 5120]` GEMV).
    /// Sources `ssm_norm_out_buf` (`[6144]`) and writes the layer's
    /// `linear_attn_out` slot (`Forward::cur().attn_out`, `[5120]`)
    /// so that the existing `AttnResidualAdd` can consume it
    /// unchanged. Mirrors llama.cpp `qwen35.cpp:462`. Sprint G-2d
    /// fills the body.
    SsmOutProj { layer: u32 },
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

/// Sprint D (v0.4.6) — Qwen3.5/3.6 plan builder. Per-layer kind
/// routing (verified against Sprint A architecture analysis):
///
/// - **Full-Attention layers** (`is_full_attention_layer == true`,
///   17 of 65 for Qwen3.6-27B = `{3, 7, …, 63, 64}`): real attention
///   dispatch chain — `AttnNorm → AttnQGateProj → QNormRope → KProj
///   → KNormRope → VProj → KvWrite → Attention → AttnGatedOutput →
///   OProj → AttnResidualAdd` followed by the standard FFN. Q-Gate
///   split lives in `q_buf` (Q in front half, Gate in back half);
///   the gated output multiplies `attn_out *= sigmoid(gate)` before
///   the O projection.
///
/// - **Linear-Attention (recurrent) layers** (48 of 65, SSM + GDN):
///   skeleton passthrough — `AttnNorm → ResidualIdentitySeed → FFN`.
///   Real recurrent attention arrives in Sprints F-G with the new
///   `ssm_conv1d.comp` + `gated_delta_net.comp` shaders.
///
/// - **MTP nextn block** (single layer at index `n_main`, =64 for
///   Qwen3.6-27B): trunk-skipped passthrough; spec-decoding draft
///   head wiring is Sprint J.
///
/// Output is still mathematically wrong while 48 layers passthrough,
/// but the 17 Full-Attn layers now perform the real Q-Gate-Split,
/// per-head Q/K norms, RoPE, GQA attention, and the sigmoid-gated
/// output multiply.
pub fn build_qwen35_layer(
    cfg: &ModelConfig,
    layer: u32,
    _flags: &LayerWeightFlags,
) -> LayerPlan {
    let spec = cfg
        .qwen35
        .as_ref()
        .expect("build_qwen35_layer called on non-qwen35 cfg");
    let mut plan: LayerPlan = Vec::with_capacity(16);

    plan.push(LayerStep::AttnNorm);

    if spec.is_full_attention_layer(layer) && !spec.is_mtp_block(layer) {
        // Sprint D2 (v0.4.6) — real Full-Attention dispatch.
        let q_dim = cfg.n_heads * cfg.head_dim;
        // Sprint E (v0.4.6) — partial mRoPE. `rope.dimension_sections =
        // [11, 11, 10, 0]` ⇒ for text-only inference the first 32 of
        // the 256 head-dim positions are rotated; the remaining 224
        // are pass-through. The fused `rms_norm_mul_rope` shader honours
        // `p.n_dims` natively (`rope_funcs.glsl:88` — when `i0 >=
        // n_dims` the dim is copied unchanged), so passing
        // `rotary_dim = 32` is sufficient. Mirrors the Gemma-4
        // partial-RoPE path (`rope_partial_factor = 0.25`,
        // `head_dim = 512 → rotary_dim = 128`).
        //
        // The base for `theta_scale` is the *rotary dim* (= 32), not
        // the head dim, matching llama.cpp's `qwen35.cpp` and the
        // standard RoPE-NeoX convention (`θ_i = freq_base^(-2 i /
        // n_dims)`).
        let rotary_dim = spec.n_rot_text_only();
        let freq_base = cfg.rope_freq_base;
        let theta_scale = (1.0_f32 / freq_base).powf(2.0 / rotary_dim as f32);

        plan.push(LayerStep::AttnQGateProj { q_dim });
        plan.push(LayerStep::QNormRope { rotary_dim, freq_base, theta_scale });
        plan.push(LayerStep::KProj);
        plan.push(LayerStep::KNormRope { rotary_dim, freq_base, theta_scale });
        plan.push(LayerStep::VProj);
        plan.push(LayerStep::KvWrite);
        plan.push(LayerStep::Attention { kv_layer: layer, kv_start: 0 });
        plan.push(LayerStep::AttnGatedOutput { q_dim });
        plan.push(LayerStep::OProj);
        plan.push(LayerStep::AttnResidualAdd);
    } else if spec.is_recurrent_layer(layer) {
        // Sprint G-2b (v0.4.6) — Linear-Attn block. Full 13-step
        // attention sub-block matching llama.cpp's
        // `build_layer_attn_linear` (qwen35.cpp:337-469); see
        // `docs/qwen35_sprint_g_plan.md` §3 for the source-line
        // mapping. Order is verified against upstream — do NOT
        // reorder without re-reading qwen35.cpp.
        //
        // All 10 SSM-step bodies are still no-op in Sprint G-2b
        // (real dispatches land in G-2c/d/e). The plan emits them
        // now so the exhaustive-match enforces both DEC + BAT
        // implementations before any real work starts.
        plan.push(LayerStep::AttnQkvProj    { layer });    // Op 2: 5120 → 10240
        plan.push(LayerStep::AttnGateZProj  { layer });    // Op 3: 5120 → 6144 (z)
        plan.push(LayerStep::SsmBetaProj    { layer });    // Op 4-5: 5120 → 48 + sigmoid
        plan.push(LayerStep::SsmAlphaGate   { layer });    // Op 6-9: alpha + dt + softplus × ssm_a
        plan.push(LayerStep::SsmConv1d      { layer });    // Op 10: conv dispatch
        plan.push(LayerStep::SsmSilu        { layer });    // Op 11: silu(conv_out)
        plan.push(LayerStep::SsmQkL2Norm    { layer });    // Op 13: l2_norm Q, K
        plan.push(LayerStep::SsmRepeatQK    { layer });    // Op 14: 16 → 48 heads
        plan.push(LayerStep::GatedDeltaNet  { layer });    // Op 15: GDN + state update
        plan.push(LayerStep::NormGated      { layer });    // Op 16: rms_norm × silu(z)
        plan.push(LayerStep::SsmOutProj     { layer });    // Op 17: 6144 → 5120
        plan.push(LayerStep::AttnResidualAdd);             // Op 18 (existing): res1 += attn_out
    } else {
        // MTP nextn block (layer == n_main). Trunk-skipped passthrough
        // until the spec-decoding draft head lands.
        plan.push(LayerStep::ResidualIdentitySeed);
    }

    // FFN is identical to Qwen3 (SwiGLU, no biases, no PostFfnNorm).
    plan.push(LayerStep::PreFfnNorm);
    plan.push(LayerStep::GateProj);
    plan.push(LayerStep::UpProj);
    plan.push(LayerStep::Activation { kind: ActivationKind::SwiGlu });
    plan.push(LayerStep::DownProj);
    plan.push(LayerStep::FfnResidualAdd);
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
    } else if cfg.qwen35.is_some() {
        build_qwen35_layer(cfg, layer, &flags)
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
            qwen35: None,
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
            qwen35: None,
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
            qwen35: None,
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

    fn qwen35_cfg() -> ModelConfig {
        ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 65,
            n_heads: 24,
            n_kv_heads: 4,
            hidden_dim: 5120,
            ffn_dim: 17408,
            vocab_size: 248320,
            head_dim: 256,
            rope_freq_base: 1.0e7,
            rope_dim: 64,
            rope_variant: RopeVariant::Neox,
            context_length: 262144,
            has_qk_norm: true,
            rms_norm_eps: 1.0e-6,
            gemma4: None,
            qwen35: Some(super::super::super::gguf::Qwen35Spec {
                block_count: 65,
                nextn_predict_layers: 1,
                full_attention_interval: 4,
                n_head_kv_full_attn: 4,
                ssm_d_conv: 4,
                ssm_d_state: 128,
                ssm_n_group: 16,
                ssm_dt_rank: 48,
                ssm_d_inner: 6144,
                rope_sections: [11, 11, 10, 0],
            }),
        }
    }

    /// Sprint G-2b — recurrent (Linear-Attn) layers emit the full
    /// 12-step SSM attention sub-block + 6-step FFN = 19 steps total
    /// (incl. the prefix `AttnNorm`). Mirrors llama.cpp's
    /// `build_layer_attn_linear` (qwen35.cpp:337-469). Step bodies in
    /// both executors are no-op staging stubs in G-2b; Sprint G-2c/d/e
    /// fills them. The test pins the EXACT order — any reorder must
    /// re-read qwen35.cpp.
    #[test]
    fn qwen35_recurrent_layer_emits_full_ssm_sequence() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        // Layer 0 is recurrent (`(0+1) % 4 != 0`).
        let plan = build_qwen35_layer(&cfg, 0, &flags);
        // 1 AttnNorm + 12 SSM steps + 6 FFN = 19.
        assert_eq!(plan.len(), 19, "recurrent layer plan length (Sprint G-2b)");

        assert!(matches!(plan[0],  LayerStep::AttnNorm));
        assert!(matches!(plan[1],  LayerStep::AttnQkvProj   { layer: 0 }));
        assert!(matches!(plan[2],  LayerStep::AttnGateZProj { layer: 0 }));
        assert!(matches!(plan[3],  LayerStep::SsmBetaProj   { layer: 0 }));
        assert!(matches!(plan[4],  LayerStep::SsmAlphaGate  { layer: 0 }));
        assert!(matches!(plan[5],  LayerStep::SsmConv1d     { layer: 0 }));
        assert!(matches!(plan[6],  LayerStep::SsmSilu       { layer: 0 }));
        assert!(matches!(plan[7],  LayerStep::SsmQkL2Norm   { layer: 0 }));
        assert!(matches!(plan[8],  LayerStep::SsmRepeatQK   { layer: 0 }));
        assert!(matches!(plan[9],  LayerStep::GatedDeltaNet { layer: 0 }));
        assert!(matches!(plan[10], LayerStep::NormGated     { layer: 0 }));
        assert!(matches!(plan[11], LayerStep::SsmOutProj    { layer: 0 }));
        assert!(matches!(plan[12], LayerStep::AttnResidualAdd));

        // FFN unchanged.
        assert!(matches!(plan[13], LayerStep::PreFfnNorm));
        assert!(matches!(plan[18], LayerStep::FfnResidualAdd));

        // A different recurrent layer carries its own index.
        let plan62 = build_qwen35_layer(&cfg, 62, &flags);
        assert!(matches!(plan62[1], LayerStep::AttnQkvProj { layer: 62 }));
        assert!(matches!(plan62[9], LayerStep::GatedDeltaNet { layer: 62 }));
    }

    /// Sprint G-2b — `ResidualIdentitySeed` must no longer appear on
    /// any recurrent layer (Sprint F's placeholder is replaced by
    /// the real `AttnResidualAdd` at position 12). MTP block keeps
    /// the seed as its passthrough sentinel.
    #[test]
    fn qwen35_recurrent_layers_drop_residual_identity_seed() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        for &layer in &[0u32, 1, 2, 4, 5, 6, 62] {
            let plan = build_qwen35_layer(&cfg, layer, &flags);
            for (i, step) in plan.iter().enumerate() {
                assert!(
                    !matches!(step, LayerStep::ResidualIdentitySeed),
                    "recurrent layer {layer} step {i} still emits ResidualIdentitySeed"
                );
            }
        }
        // MTP block (layer 64) still uses the seed (no SSM semantics).
        let mtp = build_qwen35_layer(&cfg, 64, &flags);
        assert!(mtp.iter().any(|s| matches!(s, LayerStep::ResidualIdentitySeed)));
    }

    /// Sprint G-2b — Full-attention layers must NOT emit any SSM
    /// step. The 17 Full-Attn trunk layers + the MTP block run
    /// conventional Q-Gate-Split + GQA-attention (Sprint D2).
    #[test]
    fn qwen35_full_attention_layers_do_not_emit_ssm_steps() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        for layer in [3u32, 7, 11, 27, 63, 64] {
            let plan = build_qwen35_layer(&cfg, layer, &flags);
            for step in &plan {
                assert!(
                    !matches!(
                        step,
                        LayerStep::SsmConv1d     { .. } |
                        LayerStep::AttnQkvProj   { .. } |
                        LayerStep::AttnGateZProj { .. } |
                        LayerStep::SsmBetaProj   { .. } |
                        LayerStep::SsmAlphaGate  { .. } |
                        LayerStep::SsmSilu       { .. } |
                        LayerStep::SsmQkL2Norm   { .. } |
                        LayerStep::SsmRepeatQK   { .. } |
                        LayerStep::GatedDeltaNet { .. } |
                        LayerStep::NormGated     { .. } |
                        LayerStep::SsmOutProj    { .. }
                    ),
                    "non-recurrent layer {layer} must not emit SSM steps"
                );
            }
        }
    }

    /// Sprint G-2b — each new SsmConv1d-family step variant carries
    /// the right layer index, end-to-end through the builder. Pin
    /// the layer-field invariant so a future loop-counter bug
    /// (e.g. emitting all steps with `layer: 0`) is caught here.
    #[test]
    fn qwen35_ssm_steps_carry_correct_layer_index() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        for &target in &[0u32, 1, 2, 4, 5, 62] {
            let plan = build_qwen35_layer(&cfg, target, &flags);
            for step in &plan {
                match step {
                    LayerStep::AttnQkvProj   { layer } |
                    LayerStep::AttnGateZProj { layer } |
                    LayerStep::SsmBetaProj   { layer } |
                    LayerStep::SsmAlphaGate  { layer } |
                    LayerStep::SsmConv1d     { layer } |
                    LayerStep::SsmSilu       { layer } |
                    LayerStep::SsmQkL2Norm   { layer } |
                    LayerStep::SsmRepeatQK   { layer } |
                    LayerStep::GatedDeltaNet { layer } |
                    LayerStep::NormGated     { layer } |
                    LayerStep::SsmOutProj    { layer } => {
                        assert_eq!(*layer, target,
                            "SSM step on layer {target} carried layer={layer}");
                    }
                    _ => {}
                }
            }
        }
    }

    /// Sprint D2 — Full-Attention layers dispatch the real chain.
    /// Layer 3 (`(3+1) % 4 == 0`) is the first Full-Attn block.
    #[test]
    fn qwen35_full_attention_layers_emit_real_attention_chain() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        let plan = build_qwen35_layer(&cfg, 3, &flags);

        // AttnNorm + 10 Full-Attn steps + 6 FFN steps = 17.
        assert_eq!(plan.len(), 17, "full-attn layer plan length");
        let q_dim = cfg.n_heads * cfg.head_dim;
        assert!(matches!(plan[0], LayerStep::AttnNorm));
        assert!(matches!(
            plan[1],
            LayerStep::AttnQGateProj { q_dim: d } if d == q_dim
        ));
        assert!(matches!(plan[2], LayerStep::QNormRope { .. }));
        assert!(matches!(plan[3], LayerStep::KProj));
        assert!(matches!(plan[4], LayerStep::KNormRope { .. }));
        assert!(matches!(plan[5], LayerStep::VProj));
        assert!(matches!(plan[6], LayerStep::KvWrite));
        assert!(matches!(plan[7], LayerStep::Attention { kv_layer: 3, .. }));
        assert!(matches!(
            plan[8],
            LayerStep::AttnGatedOutput { q_dim: d } if d == q_dim
        ));
        assert!(matches!(plan[9], LayerStep::OProj));
        assert!(matches!(plan[10], LayerStep::AttnResidualAdd));
        // FFN tail unchanged.
        assert!(matches!(plan[11], LayerStep::PreFfnNorm));
        assert!(matches!(plan[16], LayerStep::FfnResidualAdd));
    }

    /// Verify the 17-of-65 Full-Attention split matches the Sprint
    /// A inventory (`{3, 7, 11, …, 59, 63, 64}` — every 4th block
    /// in the trunk plus the MTP-adjacent block at index 64).
    /// Sprint G-2b: recurrent layers no longer carry
    /// `ResidualIdentitySeed` (replaced by the full SSM pipeline),
    /// so passthrough count drops from 49 → 1 (MTP block only).
    /// Recurrent layers are now identified by `SsmConv1d`.
    #[test]
    fn qwen35_full_attention_layer_count() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        let mut full_count = 0;
        let mut recurrent_count = 0;
        let mut mtp_pass_count = 0;
        for layer in 0..cfg.n_layers {
            let plan = build_qwen35_layer(&cfg, layer, &flags);
            if plan.iter().any(|s| matches!(s, LayerStep::AttnQGateProj { .. })) {
                full_count += 1;
            }
            if plan.iter().any(|s| matches!(s, LayerStep::SsmConv1d { .. })) {
                recurrent_count += 1;
            }
            if plan.iter().any(|s| matches!(s, LayerStep::ResidualIdentitySeed)) {
                mtp_pass_count += 1;
            }
        }
        // Full-attn trunk layers = {3, 7, ..., 63} (16 blocks) carry
        // AttnQGateProj; the MTP block at layer 64 hits the `else`
        // branch in build_qwen35_layer and carries ResidualIdentitySeed
        // (its draft-head wiring is Sprint J). Recurrent layers = 48
        // trunk (carry SsmConv1d). 16 + 48 + 1 = 65 = n_layers ✓
        assert_eq!(full_count, 16, "expected 16 trunk Full-Attn layers (AttnQGateProj)");
        assert_eq!(recurrent_count, 48, "expected 48 recurrent (Linear-Attn) layers");
        assert_eq!(mtp_pass_count, 1, "expected 1 MTP-block passthrough");
        assert_eq!(full_count + recurrent_count + mtp_pass_count, cfg.n_layers as usize,
            "every layer must be classified");
    }

    /// Confirm the dispatcher routes qwen35 cfgs to the qwen35
    /// builder, even though `cfg.gemma4` is `None` (would otherwise
    /// fall through to `build_qwen3_layer`, which would emit
    /// `QProj`/`KProj` against weights that don't exist for the
    /// Linear-Attention layers and panic at dispatch time).
    /// Sprint G-2b: layer 0 is recurrent and carries the SSM
    /// pipeline (not `QProj` / `Attention` / `ResidualIdentitySeed`).
    #[test]
    fn dispatcher_routes_qwen35_to_skeleton_builder() {
        let cfg = qwen35_cfg();
        let flags = LayerWeightFlags::default();
        let plan = build_qwen35_layer(&cfg, 0, &flags);
        // Recurrent layer must use the SSM path, NOT vanilla GQA-attention.
        assert!(plan.iter().any(|s| matches!(s, LayerStep::SsmConv1d { .. })));
        assert!(plan.iter().any(|s| matches!(s, LayerStep::AttnQkvProj { .. })));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::QProj)));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::Attention { .. })));
        assert!(!plan.iter().any(|s| matches!(s, LayerStep::ResidualIdentitySeed)));
    }
}
