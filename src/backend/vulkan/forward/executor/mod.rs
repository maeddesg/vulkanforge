//! Sprint 44C-2 — `LayerExecutor` trait + `DecodeExec` (Phase A).
//!
//! 44C-1 shipped the `LayerStep` enum and per-arch builders. This file
//! wires the executors that turn a `LayerPlan` into Vulkan dispatches.
//!
//! ## How the bug-class is prevented
//!
//! Each `impl LayerExecutor for {Decode,Batch}Exec` block contains a
//! single `match step { ... }` covering every `LayerStep` variant
//! exhaustively — *no* `_ => {}` wildcard. If a future sprint adds a
//! new variant (e.g. `AltUpProject`), every executor's `match` becomes
//! non-exhaustive and the crate fails to compile until each path
//! handles the variant. The "added a step in dispatch_layer but
//! forgot dispatch_layer_batch" class of bugs (memory
//! `feedback_layer_dispatch_paths`) becomes structurally
//! unrepresentable.
//!
//! ## Phase A scope
//!
//! - `DecodeExec` covers every variant. It mirrors the existing
//!   `dispatch_layer` (decode.rs) line-for-line; pure code-move into
//!   match arms.
//! - The Llama-only `AttnResidualAdd` → `PreFfnNorm` Sprint-9b fusion
//!   into `multi_add_rms` is preserved by a small lookahead in the
//!   `execute_layer` loop driver. Gemma-4's `PostAttnNorm` /
//!   `PreFfnNorm` 3-norm path keeps its separate dispatches.
//! - `BatchExec` lands in Phase B (44C-2 step 2) after owner review.
//!
//! Activation by `VF_USE_LAYER_PLAN=1` env var. Default OFF — the
//! existing inline body of `dispatch_layer` stays the production path
//! until Phase B + bit-identity validation pass.

#![allow(dead_code)]

use ash::vk;

use super::super::device::VulkanDevice;
use super::super::gguf::GgmlType;
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;

use super::arch::compute_barrier;
use super::layer_plan::{LayerPlan, LayerStep};
use super::state::Forward;
use super::super::loader::MoeRouterLayerData;

// Sprint 57B — submodule split. `dispatch` = keystone match. `attention`,
// `ffn` carry their respective step_* + b_step_* per-category.
mod attention;
mod control;
mod dispatch;
mod ffn;
mod moe;

/// Sprint 51D-D — one-shot log-on-first-call for the layer-0 router
/// decision (debug aid). Prevents flooding decode-token logs with
/// per-token prints; only the first decode token reports.
pub(super) static MOE_LAYER0_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Sprint 51D-D — COMPUTE_SHADER → TRANSFER barrier.
///
/// Equivalent of `transfer_to_compute_barrier` flipped: makes prior
/// shader writes visible to a `cmd_copy_buffer` reading those same
/// buffers. Used by the MoE route step to drain `scratch_b` (written
/// by `step_pre_moe_norm`'s rms_norm dispatch) before the host
/// readback copy.
pub(super) fn compute_to_transfer_barrier(dev: &super::super::device::VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

/// Sprint 51D-D — TRANSFER → HOST barrier.
///
/// Makes a just-issued `cmd_copy_buffer` write into a host-visible
/// buffer observable to host reads after the next queue-submit drains.
/// Required for the MoE router because the CPU reads
/// `moe_route_staging` directly through its mmap pointer; without
/// this barrier the HOST_READ access on the staging would race with
/// the TRANSFER_WRITE.
pub(super) fn transfer_to_host_barrier(dev: &super::super::device::VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

/// Sprint 51D-D — pure-CPU MoE router.
///
/// Implements the Gemma-4-26B-A4B routing formula
/// (transformers `Gemma4SparseMoeBlock.forward`):
///
/// 1. Parameterless RMS-norm of `hidden`.
/// 2. Multiply by `scale[hidden]` and `hidden_size**(-0.5)`
///    (the brief-mandated "scale × inv_sqrt" pre-MatMul correction).
/// 3. MatMul against `proj` shape `[n_experts, hidden]` row-major
///    (PyTorch convention `proj[e][h] = proj[e * hidden + h]`).
/// 4. Softmax → probs over experts.
/// 5. Top-K with sort-by-prob.
/// 6. Renormalize Top-K weights to sum=1.
/// 7. Multiply each by `per_expert_scale[idx]`.
///
/// All math runs in `f64` accumulators to avoid catastrophic
/// cancellation on the softmax denominator (n_experts=128).
pub(super) fn cpu_moe_route(
    hidden: &[f32],
    layer_data: &MoeRouterLayerData,
    hidden_size: usize,
    n_experts: usize,
    top_k: usize,
    eps: f32,
) -> Vec<(u32, f32)> {
    debug_assert_eq!(hidden.len(), hidden_size);
    debug_assert_eq!(layer_data.proj.len(), n_experts * hidden_size);
    debug_assert_eq!(layer_data.scale.len(), hidden_size);
    debug_assert_eq!(layer_data.per_expert_scale.len(), n_experts);

    // (1) Parameterless RMS norm.
    let mut sq_sum = 0.0_f64;
    for &v in hidden {
        sq_sum += (v as f64) * (v as f64);
    }
    let mean_sq = sq_sum / (hidden_size as f64);
    let rms_inv = 1.0 / (mean_sq + eps as f64).sqrt();

    // (2) Per-channel scale × hidden^(-0.5). Combine into one scalar
    //     per channel to avoid two passes.
    let inv_sqrt = (hidden_size as f64).powf(-0.5);
    let mut scaled = vec![0.0_f64; hidden_size];
    for i in 0..hidden_size {
        scaled[i] = (hidden[i] as f64) * rms_inv
                  * (layer_data.scale[i] as f64) * inv_sqrt;
    }

    // (3) MatMul: proj[e, :] · scaled  → scores[e].
    let mut scores = vec![0.0_f64; n_experts];
    for e in 0..n_experts {
        let row = &layer_data.proj[e * hidden_size..(e + 1) * hidden_size];
        let mut acc = 0.0_f64;
        for j in 0..hidden_size {
            acc += (row[j] as f64) * scaled[j];
        }
        scores[e] = acc;
    }

    // (4) Softmax (numerically stable: subtract max).
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    let mut probs = vec![0.0_f64; n_experts];
    for e in 0..n_experts {
        let p = (scores[e] - max).exp();
        probs[e] = p;
        sum += p;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // (5) Top-K (partial-sort by descending prob).
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(top_k);

    // (6) Renormalize selected probs to sum=1.
    let top_sum: f64 = idx.iter().map(|&i| probs[i]).sum();
    let renorm = if top_sum > 0.0 { 1.0 / top_sum } else { 0.0 };

    // (7) Apply per-expert scale.
    idx.into_iter()
        .map(|i| {
            let w = probs[i] * renorm * (layer_data.per_expert_scale[i] as f64);
            (i as u32, w as f32)
        })
        .collect()
}

/// Sprint 51D-D — sanity log for the layer-0 first-token router decision.
pub(super) fn log_router_decision(layer: u32, routing: &[(u32, f32)]) {
    let pre_renorm_sum: f32 = routing.iter().map(|&(_, w)| w).sum();
    let mut s = format!("Sprint 51D-D MoE Layer {layer} Top-{} = [", routing.len());
    for (i, &(e, w)) in routing.iter().enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("({e}, {w:.4})"));
    }
    s.push_str(&format!("], post-renorm×pes sum = {pre_renorm_sum:.4}"));
    eprintln!("{s}");
}

/// Per-call execution context shared between executors.
pub(super) struct ExecCtx<'a> {
    pub dev: &'a VulkanDevice,
    pub registry: &'a PipelineRegistry,
    pub cmd: vk::CommandBuffer,
    pub model: &'a LoadedModel,
    pub layer: u32,
    pub mode: ExecMode,
}

/// Execution mode — discriminates per-token decode from batched prefill.
/// Each executor type assumes a specific variant; mismatches panic.
#[derive(Clone, Copy)]
pub(super) enum ExecMode {
    Decode {
        position: u32,
        /// Layer's input buffer (caller-provided).
        input: vk::Buffer,
        /// Layer's output buffer (caller-provided).
        output: vk::Buffer,
    },
    Batch {
        seq_len: u32,
        base_pos: u32,
        /// Sprint 9b.2 cross-layer fusion: when `Some`, the layer's
        /// `FfnResidualAdd` may fuse with the next layer's
        /// `AttnNorm`. Set to `None` for the last layer.
        next_attn_norm_weight: Option<vk::Buffer>,
    },
}

/// Marker type implementing the per-token decode-path executor.
pub(super) struct DecodeExec;

impl DecodeExec {
    /// Drive a complete layer plan. The caller invokes this once per
    /// layer; `plan` is the output of `build_layer_plan(cfg, model,
    /// layer, ...)` for that specific layer.
    ///
    /// Implements the Llama `AttnResidualAdd` + `PreFfnNorm` →
    /// `multi_add_rms` fusion via a single-step lookahead. Gemma-4's
    /// `PostAttnNorm` precedes `AttnResidualAdd`, so the lookahead
    /// declines fusion when it sees the 3-norm path.
    pub(super) fn execute_layer(
        &self,
        fwd: &mut Forward,
        plan: &LayerPlan,
        ctx: &ExecCtx,
    ) {
        let cfg = fwd.config.clone();
        let mut i = 0;
        while i < plan.len() {
            // Llama path fusion: AttnResidualAdd + PreFfnNorm → one
            // multi_add_rms dispatch. Skipped on Gemma-4 (the plan
            // contains PostAttnNorm before AttnResidualAdd, so the
            // residual add reads a different scratch buffer and the
            // fusion can't apply).
            if cfg.gemma4.is_none()
                && matches!(plan.get(i), Some(LayerStep::AttnResidualAdd))
                && matches!(plan.get(i + 1), Some(LayerStep::PreFfnNorm))
            {
                self.fused_attn_residual_norm(fwd, &cfg, ctx);
                i += 2;
                continue;
            }
            self.execute_step(fwd, &plan[i], &cfg, ctx);
            i += 1;
        }
    }

    // ── per-step implementations ──────────────────────────────────────

    // DecodeExec attention steps (step_attn_norm through
    // step_attn_residual_add) live in executor/attention.rs.

    // DecodeExec dense-FFN steps (step_pre_ffn_norm through
    // step_ffn_residual_add) live in executor/ffn.rs.

    // DecodeExec layer-control steps (step_ple_block,
    // step_layer_scalar_mul) live in executor/control.rs.

    // DecodeExec MoE steps (step_post_dense_mlp_norm through
    // step_moe_expert_ffn) live in executor/moe.rs.
}

// ── BatchExec (Phase B of Sprint 44C-2) ───────────────────────────────
//
// State-less per-prompt prefill executor. Mirrors the existing
// `dispatch_layer_batch` (prefill.rs) line-for-line into a `match
// LayerStep` over all 26 variants.
//
// Differences vs DecodeExec:
// - GEMV → GEMM for every projection (run_gemm / run_gemm_fp8_* /
//   run_gemm_coopmat_q4k).
// - scalar_attn → flash_attn_batch / flash_attn_tiled (driven by
//   `fa_tiled_enabled`).
// - Per-token slot buffers → `batch_*` slabs.
// - Unconditional `compute_barrier(dev, cmd)` between dispatches —
//   the elision tracker is decode-only (the original prefill path
//   never calls `mark_written` / `maybe_compute_barrier`).
// - `AttnNorm` is a no-op: `batch_norm` is pre-seeded by
//   `record_prefill_seed` (layer 0) or by the previous layer's
//   Sprint-9b.2 multi_add_rms fusion (layers 1+).
// - `FfnResidualAdd`'s match arm reads `next_attn_norm_weight` from
//   `ExecMode::Batch`; on the Llama path it dispatches multi_add_rms
//   to fuse with the next layer's pre-seed (matching the legacy
//   call-site contract). Gemma-4 falls back to plain add and emits
//   the seed RMSNorm separately after `LayerScalarMul` (handled in
//   `execute_layer`'s loop tail).
// - **`batch_attn = false` legacy per-token loop is dropped**:
//   `BatchExec` always uses the batched-attention path. Owner
//   confirmed in 44C-2 brief.
// - **Quantize state-less**: each MMQ-routed projection re-quantizes
//   its input. Bit-identical (deterministic re-quantize); 3-extra
//   `quantize_q8_1` dispatches per layer when MMQ is selected.

pub(super) struct BatchExec;

impl BatchExec {
    /// Drive a complete batched-prefill layer plan. The caller invokes
    /// this once per layer; cross-layer fusion (Sprint 9b.2 +
    /// Gemma-4 next-attn-norm seed) is handled here using
    /// `next_attn_norm_weight` from `ExecMode::Batch`.
    pub(super) fn execute_layer(
        &self,
        fwd: &mut Forward,
        plan: &LayerPlan,
        ctx: &ExecCtx,
    ) {
        let cfg = fwd.config.clone();
        let mut i = 0;
        while i < plan.len() {
            // Llama-path lookahead: AttnResidualAdd + PreFfnNorm →
            // single multi_add_rms (Sprint 9b). Skipped on Gemma-4
            // (the plan has PostAttnNorm preceding AttnResidualAdd, so
            // the residual reads scratch instead of batch_o, and the
            // fusion shader can't model that).
            if cfg.gemma4.is_none()
                && matches!(plan.get(i), Some(LayerStep::AttnResidualAdd))
                && matches!(plan.get(i + 1), Some(LayerStep::PreFfnNorm))
            {
                self.b_fused_attn_residual_norm(fwd, &cfg, ctx);
                i += 2;
                continue;
            }
            self.execute_step(fwd, &plan[i], &cfg, ctx);
            i += 1;
        }
        // Gemma-4 cross-layer seed: emit a separate rms_norm to seed
        // the next layer's `batch_norm` from the (post-LayerScalarMul)
        // batch_residual. Llama fuses this into the FfnResidualAdd
        // match arm via multi_add_rms, so this only fires on Gemma-4.
        if cfg.gemma4.is_some() {
            let next_w = match ctx.mode {
                ExecMode::Batch { next_attn_norm_weight, .. } => next_attn_norm_weight,
                _ => unreachable!("BatchExec invoked with Decode mode"),
            };
            if let Some(w_next) = next_w {
                let seq_len = batch_seq_len(ctx);
                fwd.run_rms_norm(
                    ctx.dev, ctx.registry, ctx.cmd,
                    fwd.batch_residual.handle, w_next, fwd.batch_norm.handle,
                    cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "rms_norm_next_attn_b",
                );
                compute_barrier(ctx.dev, ctx.cmd);
            }
        }
    }

    // ── per-step impls ────────────────────────────────────────────────

    // BatchExec attention steps (b_step_attn_norm through
    // b_step_attn_residual_add) live in executor/attention.rs.

    // BatchExec dense-FFN steps (b_step_pre_ffn_norm through
    // b_step_ffn_residual_add) live in executor/ffn.rs.

    // BatchExec layer-control steps (b_step_ple_block,
    // b_step_layer_scalar_mul) live in executor/control.rs.

    // BatchExec MoE steps (b_step_post_dense_mlp_norm through
    // b_step_moe_expert_ffn) live in executor/moe.rs.

    // Shared GEMM-routing helper (b_run_proj) lives in
    // executor/control.rs.
}

/// Sprint 52ZA — see `feedback_batch_q8_reuse_mixed_quant`.
pub(super) fn quantize_input_after_q(model: &LoadedModel, layer: u32) -> bool {
    let q_type = model
        .tensor(&format!("blk.{layer}.attn_q.weight"))
        .map(|t| t.ggml_type);
    let q_took_mmq = matches!(
        q_type,
        Some(GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0),
    );
    !q_took_mmq
}

/// Helper: extract `seq_len` from a Batch ExecCtx.
pub(super) fn batch_seq_len(ctx: &ExecCtx) -> u32 {
    match ctx.mode {
        ExecMode::Batch { seq_len, .. } => seq_len,
        _ => unreachable!("BatchExec invoked with Decode mode"),
    }
}

/// Helper: extract `(seq_len, base_pos)` from a Batch ExecCtx.
pub(super) fn batch_seq_pos(ctx: &ExecCtx) -> (u32, u32) {
    match ctx.mode {
        ExecMode::Batch { seq_len, base_pos, .. } => (seq_len, base_pos),
        _ => unreachable!("BatchExec invoked with Decode mode"),
    }
}

// ── helpers ──────────────────────────────────────────────────────────

pub(super) fn decode_io(ctx: &ExecCtx) -> (vk::Buffer, vk::Buffer) {
    match ctx.mode {
        ExecMode::Decode { input, output, .. } => (input, output),
        ExecMode::Batch { .. } => unreachable!("DecodeExec invoked with Batch mode"),
    }
}

pub(super) fn decode_position(ctx: &ExecCtx) -> u32 {
    match ctx.mode {
        ExecMode::Decode { position, .. } => position,
        ExecMode::Batch { .. } => unreachable!("DecodeExec invoked with Batch mode"),
    }
}

/// Re-export of `arch::common::layer_dims` for the per-step helpers.
/// Returns `(head_dim, ffn_dim, rope_theta, rotary_dim)` per layer.
pub(super) use super::arch::layer_dims as layer_dims_local;

/// Sprint 51B-pre — per-layer KV-head count (8 / 2 split for the
/// Gemma-4-26B-A4B sliding / full pattern). Falls back to the
/// uniform `ModelConfig::n_kv_heads` for non-Gemma-4 architectures.
pub(super) use super::arch::n_kv_heads_for;
