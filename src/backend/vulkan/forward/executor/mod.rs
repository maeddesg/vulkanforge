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

// Sprint P0-1 — re-export the canonical MoE batched-decode gate so
// `setup.rs` can call it instead of duplicating the env check.
pub(crate) use moe::batched_decode_moe_enabled;
// Sprint P1-3 — re-export the fused-router gate so `run_moe_router_gpu`
// in `runs.rs` can pick the single-dispatch path.
pub(crate) use moe::moe_fused_router_enabled;

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
        // Sprint SG-1.3 — VF_USE_GRAPH=1 opt-in graph-dispatch path.
        // Default OFF; the imperative loop below is the production
        // default. When ON, build a per-layer VulkanGraph, sort it
        // topologically, and execute steps in that order.
        //
        // For Qwen3-8B the topological order matches plan-order (the
        // Builder's forward-only RAW edges preserve insertion order),
        // so output is bit-identical to the imperative path. SG-1.3's
        // value is *plumbing* — it proves Builder → resolve → sort →
        // execute_step round-trips correctly. The actual byte-range
        // barrier reduction (the analysis-Teil-4 lever) lands in SG-2
        // as a `Pass` over the same graph.
        if std::env::var("VF_USE_GRAPH").as_deref() == Ok("1") {
            self.execute_layer_via_graph(fwd, plan, ctx);
            return;
        }

        let cfg = fwd.config.clone();
        let mut i = 0;
        while i < plan.len() {
            // Llama path fusion: AttnResidualAdd + PreFfnNorm → one
            // multi_add_rms dispatch. Skipped on Gemma-4 (the plan
            // contains PostAttnNorm before AttnResidualAdd, so the
            // residual add reads a different scratch buffer and the
            // fusion can't apply).
            // Sprint D2 — Qwen3.6 carries the Pre-FFN norm as
            // `post_attention_norm.weight`, not `ffn_norm.weight`. The
            // fused multi_add_rms helper hard-codes the Llama/Qwen3
            // tensor name (Sprint 9b.2), so disable fusion on qwen35
            // and let the executor emit AttnResidualAdd + PreFfnNorm
            // as two separate dispatches. PreFfnNorm's tensor-name
            // branch (executor/ffn.rs) already picks the right
            // `post_attention_norm.weight` for qwen35.
            // Sprint P1-5 — Qwen3.6 now joins the fusion (per-arch
            // norm-weight name in fused_attn_residual_norm); gated by
            // VF_FUSE_QWEN36_RESNORM (default on). Gemma-4 stays out.
            if cfg.gemma4.is_none()
                && (cfg.qwen35.is_none() || dispatch::fuse_qwen36_resnorm_enabled())
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

    /// Sprint SG-1.3 — graph-dispatch path. Builds a per-layer
    /// `VulkanGraph` from `plan`, resolves byte-range dependencies,
    /// runs Kahn's topological sort, then calls `execute_step` on
    /// each node's source `LayerStep` in topological order.
    ///
    /// Output is bit-identical to the imperative `execute_layer` loop
    /// because the Builder emits strictly-forward RAW edges, so the
    /// topological order coincides with the build order. The plumbing
    /// is what proves out here — Sprint SG-2 will hang a barrier-
    /// optimization `Pass` off the same graph and emit byte-range-
    /// precise `vkCmdPipelineBarrier`s, replacing the imperative
    /// path's whole-buffer-dirty tracker.
    fn execute_layer_via_graph(
        &self,
        fwd: &mut Forward,
        plan: &LayerPlan,
        ctx: &ExecCtx,
    ) {
        use crate::backend::vulkan::graph::builder::{BufferMap, GraphBuilder};
        use crate::backend::vulkan::graph::node::GraphNode;
        use super::state::BarrierMode;
        use super::arch::compute_barrier;

        let cfg = fwd.config.clone();

        // Per-layer Builder. BufferMap carries real `vk::Buffer`
        // handles so byte-range dependency resolution distinguishes
        // distinct buffers correctly (not all-null poison).
        //
        // SG-1.4-f — input/output alternate per layer in `decode.rs:104`
        // (`std::mem::swap(&mut input, &mut output)`). L0 reads scratch_a +
        // writes scratch_b, L1 reads scratch_b + writes scratch_a, …
        // Earlier sprints used `fwd.cur().scratch_a.handle` for both
        // `scratch_a` and `layer_output`, which silently mis-modelled
        // the per-layer flow: the graph thought FfnResidualAdd wrote
        // scratch_a but the imperative `execute_step` actually wrote
        // scratch_b. Under `BarrierMode::GraphDriven` this meant
        // pending_writes was seeded with the wrong handle at the
        // layer boundary, so subsequent imperative reads of the real
        // output buffer fired NO barrier — silent race.
        //
        // Fix: thread the real `(input, output)` pair from
        // `decode_io(ctx)` into the BufferMap.
        let (layer_input, layer_output_handle) = decode_io(ctx);
        let bufs = BufferMap {
            scratch_a:    layer_input,
            // Sprint SG-1.6 — Gemma-4-26B MoE PreMoeNorm output / MoE
            // input. Always present on `Forward`.
            scratch_b:    fwd.cur().scratch_b.handle,
            hidden_norm:  fwd.cur().hidden_norm.handle,
            q_buf:        fwd.cur().q_buf.handle,
            k_buf:        fwd.cur().k_buf.handle,
            v_buf:        fwd.cur().v_buf.handle,
            attn_out:     fwd.cur().attn_out.handle,
            o_buf:        fwd.cur().o_buf.handle,
            res1:         fwd.cur().res1.handle,
            gate_buf:     fwd.cur().gate_buf.handle,
            up_buf:       fwd.cur().up_buf.handle,
            ffn_hidden:   fwd.cur().ffn_hidden.handle,
            ffn_out:      fwd.cur().ffn_out.handle,
            // SG-1.4-f — use the actual layer-output handle from
            // ExecCtx (alternates scratch_a/scratch_b per layer).
            layer_output: layer_output_handle,
            kv_cache:     fwd.kv_cache.k_buffer.handle,

            // Sprint SG-1.4 — Qwen3.6 SSM buffers (`Option<GpuBuffer>` on
            // Forward; None → null handle for non-qwen35 builds). Plans
            // for non-qwen35 architectures never emit the SSM/Q-Gate
            // variants, so null handles never appear in graph edges.
            ssm_qkv:         fwd.ssm_qkv_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_z:           fwd.ssm_z_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_beta:        fwd.ssm_beta_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_alpha:       fwd.ssm_alpha_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_gate:        fwd.ssm_gate_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_conv_input:  fwd.ssm_conv_input_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_conv_output: fwd.ssm_conv_output_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_qrep:        fwd.ssm_qrep_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_krep:        fwd.ssm_krep_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_gdn_out:     fwd.ssm_gdn_out_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_norm_out:    fwd.ssm_norm_out_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            ssm_state:       fwd.ssm_state_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            conv_state:      fwd.conv_state_buf.as_ref().map(|b| b.handle).unwrap_or(ash::vk::Buffer::null()),
            // Sprint SG-1.7 — Gemma-4 MoE router scratch buffers. None
            // on non-MoE builds; the dep-pass distinguishes by buffer
            // identity, so null-vs-null doesn't generate noise edges.
            moe_gate_up_out:     fwd.moe_router_gpu.as_ref().map(|r| r.grouped_gate_up_out.handle).unwrap_or(ash::vk::Buffer::null()),
            moe_glu_out:         fwd.moe_router_gpu.as_ref().map(|r| r.grouped_glu_out.handle).unwrap_or(ash::vk::Buffer::null()),
            moe_down_out:        fwd.moe_router_gpu.as_ref().map(|r| r.grouped_down_out.handle).unwrap_or(ash::vk::Buffer::null()),
            moe_router_logits:   fwd.moe_router_gpu.as_ref().map(|r| r.logits_scratch.handle).unwrap_or(ash::vk::Buffer::null()),
            moe_router_indices:  fwd.moe_router_gpu.as_ref().map(|r| r.indices_scratch.handle).unwrap_or(ash::vk::Buffer::null()),
            moe_router_weights:  fwd.moe_router_gpu.as_ref().map(|r| r.weights_scratch.handle).unwrap_or(ash::vk::Buffer::null()),
        };

        let graph = GraphBuilder::build_per_layer(&bufs, &cfg, ctx.layer, plan);

        // Mirror the imperative-path Llama AttnResidualAdd + PreFfnNorm
        // → multi_add_rms fusion (see the loop above). Identify the
        // index of the AttnResidualAdd that pairs with a following
        // PreFfnNorm; both indices then get skipped during the
        // topo-order walk and the fusion gets emitted at the
        // AttnResidualAdd's execution point.
        let mut fusion_start: Option<usize> = None;
        let mut skip_step: Vec<bool> = vec![false; plan.len()];
        // Sprint P1-5 — Qwen3.6 joins the fusion (default on, gated by
        // VF_FUSE_QWEN36_RESNORM); Gemma-4 stays excluded.
        if cfg.gemma4.is_none()
            && (cfg.qwen35.is_none() || dispatch::fuse_qwen36_resnorm_enabled())
        {
            for i in 0..plan.len().saturating_sub(1) {
                if matches!(plan[i], LayerStep::AttnResidualAdd)
                    && matches!(plan[i + 1], LayerStep::PreFfnNorm)
                {
                    fusion_start = Some(i);
                    skip_step[i + 1] = true;
                    break;
                }
            }
        }

        // ── Sprint SG-2 — Graph-driven barrier pass ──────────────
        //
        // Within this scope, `fwd.barrier_mode = GraphDriven` makes
        // `mark_written` + `maybe_compute_barrier` no-ops on the step-
        // body side. Barriers come exclusively from the graph's edge
        // set + a global high-water-mark check.
        //
        // The high-water mark records the *next position in
        // execution_order whose writes are NOT yet synced by a
        // `vkCmdPipelineBarrier`*. Initially 0 → at least one barrier
        // fires up-front to drain the previous layer's output (it
        // wrote `scratch_a` without imperative tracking, since the
        // prior layer's execute_layer_via_graph also ran in
        // GraphDriven mode).
        //
        // Algorithm per node at position P:
        //   1. Find any incoming edge (M → N) with pos(M) >= high_water.
        //   2. If at least one such edge exists → emit one
        //      cmd_pipeline_barrier and set high_water = P. The
        //      barrier is global, so a *single* emission covers every
        //      not-yet-synced incoming edge on N (regardless of
        //      source buffer).
        //   3. Execute the step (no-op `mark_written` /
        //      `maybe_compute_barrier` inside).
        //
        // After the layer: reset barrier_mode back to Imperative AND
        // emit a final global barrier so the *next* layer (which may
        // run in either mode) sees this layer's final writes
        // committed without needing to know what they were.
        // SG-1.4-b — qwen35 now uses graph-driven barriers thanks to
        // `Forward::force_internal_barrier`, which preserves SSM sub-
        // dispatch sync regardless of `BarrierMode` (see attention.rs
        // step_ssm_alpha_gate, step_ssm_beta_proj, step_ssm_conv1d,
        // step_norm_gated for the 6 forced sub-dispatch sites).
        // Gemma-4 still keeps imperative barriers — its MoE / KV-share
        // step bodies have additional barrier patterns that SG-1.5/1.6
        // will audit.
        // SG-1.4-c — kept at SG-1.4 stance. The o_buf-vs-attn_out
        // fix in `add_ssm_out_proj` (this sprint) closes one missing
        // RAW edge but the qwen35 flip still produces partial-
        // coherence (`" 0language- to any  5"`), indicating at least
        // one more missing edge or barrier site. Documented in
        // `results/sprint_sg1_4c_edge_audit.md`. Gemma-4 also stays
        // imperative.
        // SG-1.4-d / SG-1.4-f — diagnostic env gates for graph-driven
        // barrier scope. Used to bisect remaining qwen35 enable bug.
        //
        //  VF_GRAPH_BARRIERS_ALL=1 — force graph-driven on every arch
        //  VF_GRAPH_BARRIERS_LAYERS=N-M — only layers N..=M (incl)
        let force_all = std::env::var("VF_GRAPH_BARRIERS_ALL").as_deref() == Ok("1");
        let layer_range: Option<(u32, u32)> = std::env::var("VF_GRAPH_BARRIERS_LAYERS")
            .ok()
            .and_then(|v| {
                let parts: Vec<&str> = v.split('-').collect();
                if parts.len() == 2 {
                    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
                } else {
                    None
                }
            });
        // Sprint SG-1.7-bisect — graph-driven barriers ON for all
        // architectures now that the missing Gemma-4 reads (gate_buf
        // on Attn/FfnResidualAdd) and PleBlock's internal barriers
        // (force_internal_barrier) are in place.
        let use_graph_barriers = if force_all {
            true
        } else if let Some((lo, hi)) = layer_range {
            ctx.layer >= lo && ctx.layer <= hi
        } else {
            true
        };
        let trace = std::env::var("VF_BARRIER_TRACE").as_deref() == Ok("1");
        if use_graph_barriers {
            // SG-1.4-h — clear pending_writes BEFORE switching into
            // GraphDriven mode. Otherwise leftover state from the
            // imperative caller (prefill BatchExec, prior decode
            // layers in mixed VF_GRAPH_BARRIERS_LAYERS scenarios)
            // sits unflushed during the graph window. Hypothesis (3)
            // from `results/sprint_sg1_4g_l0_fix.md` §4 — the L0
            // deterministic break may stem from this state-leak.
            if trace && !fwd.pending_writes.is_empty() {
                eprintln!(
                    "[BTRACE] GRF L{:>2} entry: pending_writes had {} stale entries",
                    ctx.layer, fwd.pending_writes.len(),
                );
            }
            fwd.pending_writes.clear();
            fwd.barrier_mode = BarrierMode::GraphDriven;
        }

        // Inter-layer drain — the previous layer's last write (scratch_a
        // from its FfnResidualAdd, or attn_out etc.) needs to be visible
        // before AttnNorm reads it. Skipped when imperative barriers
        // are still active (they'd cover this themselves).
        let mut barriers_emitted: u32 = 0;
        if use_graph_barriers {
            if trace {
                eprintln!("[BTRACE] GRF L{:>2} inter-layer-drain", ctx.layer);
            }
            compute_barrier(ctx.dev, ctx.cmd);
            barriers_emitted = 1;
        }
        // Position 0 means "everything before is synced". Any incoming
        // edge with source-position 0 is already handled by the drain
        // barrier above; any source-position >= the next high-water is
        // still pending.
        let mut high_water: usize = 0;
        // execution_order position of each node, for quick edge-source
        // lookup. The Builder's topo sort is consistent with insertion
        // order for Qwen3-8B's forward-only edges, so this lookup is
        // identity in the common case — but we compute it anyway to
        // stay correct against future reorderings.
        let mut pos_of: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::with_capacity(graph.execution_order.len());
        for (i, &nid) in graph.execution_order.iter().enumerate() {
            pos_of.insert(nid, i);
        }

        for (position, &node_id) in graph.execution_order.iter().enumerate() {
            // Emit a barrier (covering this node's reads) if any
            // incoming edge sources sit at or above the high-water
            // mark — i.e. their writes were not yet covered by a
            // previous emission. Applies to both Dispatch and Transfer
            // nodes (SG-3 — TransferNode for GDN state copy is now
            // graph-tracked, not run inline inside the dispatch body).
            let mut need_barrier = false;
            for edge in &graph.edges {
                if edge.to != node_id {
                    continue;
                }
                let from_pos = match pos_of.get(&edge.from) {
                    Some(&p) => p,
                    None => continue,
                };
                if from_pos >= high_water {
                    need_barrier = true;
                    break;
                }
            }
            if need_barrier && use_graph_barriers {
                if trace {
                    let lbl = graph.nodes[node_id as usize].label().unwrap_or("?");
                    eprintln!(
                        "[BTRACE] GRF L{:>2} node={:>3} pos={:>3} label={}",
                        ctx.layer, node_id, position, lbl
                    );
                }
                compute_barrier(ctx.dev, ctx.cmd);
                barriers_emitted += 1;
                high_water = position;
            }

            match &graph.nodes[node_id as usize] {
                GraphNode::Dispatch(d) => {
                    use crate::backend::vulkan::graph::node::SubDispatch as SD;
                    match d.sub_dispatch {
                        SD::FullStep(step_idx_u32) => {
                            let step_idx = step_idx_u32 as usize;
                            if step_idx >= plan.len() {
                                panic!(
                                    "graph node L{} FullStep({}) out of bounds for plan.len()={}",
                                    ctx.layer, step_idx, plan.len(),
                                );
                            }
                            if skip_step[step_idx] {
                                continue;
                            }
                            if Some(step_idx) == fusion_start {
                                self.fused_attn_residual_norm(fwd, &cfg, ctx);
                            } else {
                                self.execute_step(fwd, &plan[step_idx], &cfg, ctx);
                            }
                        }
                        // ── SG-3 decomposed SSM sub-dispatches ─────────
                        SD::NormGatedRms => self.sub_norm_gated_rms(fwd, &cfg, ctx, d.layer),
                        SD::NormGatedSilu => self.sub_norm_gated_silu(fwd, &cfg, ctx),
                        SD::NormGatedMul => self.sub_norm_gated_mul(fwd, &cfg, ctx),
                        SD::AlphaGateGemv => self.sub_alpha_gate_gemv(fwd, &cfg, ctx, d.layer),
                        SD::AlphaGateAdd => self.sub_alpha_gate_add(fwd, &cfg, ctx, d.layer),
                        SD::AlphaGateSoftplus => self.sub_alpha_gate_softplus(fwd, &cfg, ctx),
                        SD::AlphaGateMulA => self.sub_alpha_gate_mul_a(fwd, &cfg, ctx, d.layer),
                        SD::ConvSetup => self.sub_conv_setup(fwd, &cfg, ctx, d.layer),
                        SD::ConvDispatch => self.sub_conv_dispatch(fwd, &cfg, ctx, d.layer),
                        SD::BetaGemv => self.sub_beta_gemv(fwd, &cfg, ctx, d.layer),
                        SD::BetaSigmoid => self.sub_beta_sigmoid(fwd, &cfg, ctx),
                        SD::GdnCompute => self.sub_gdn_compute(fwd, &cfg, ctx, d.layer),
                        SD::GdnStateCopy => unreachable!(
                            "GdnStateCopy is a TransferNode, not a DispatchNode"
                        ),
                        // ── Sprint SG-1.7 — Gemma-4 MoE + PLE sub-dispatches ──
                        SD::MoeRouterNormGemv =>
                            self.sub_moe_router_norm_gemv(fwd, &cfg, ctx, d.layer),
                        SD::MoeRouterSoftmaxTopk =>
                            self.sub_moe_router_softmax_topk(fwd, &cfg, ctx, d.layer),
                        SD::MoeFfnClear => unreachable!(
                            "MoeFfnClear is a TransferNode, not a DispatchNode"
                        ),
                        SD::MoeFfnGateUp =>
                            self.sub_moe_ffn_gate_up(fwd, &cfg, ctx, d.layer),
                        SD::MoeFfnGluSlot(slot) =>
                            self.sub_moe_ffn_glu(fwd, &cfg, ctx, d.layer, slot),
                        SD::MoeFfnDown =>
                            self.sub_moe_ffn_down(fwd, &cfg, ctx, d.layer),
                        SD::MoeFfnFmaSlot(slot) =>
                            self.sub_moe_ffn_fma(fwd, &cfg, ctx, d.layer, slot),
                        SD::PleGemvGate => self.sub_ple_gemv_gate(fwd, &cfg, ctx, d.layer),
                        SD::PleGeluGlu => self.sub_ple_gelu_glu(fwd, &cfg, ctx, d.layer),
                        SD::PleGemvProj => self.sub_ple_gemv_proj(fwd, &cfg, ctx, d.layer),
                        SD::PleRmsNorm => self.sub_ple_rms_norm(fwd, &cfg, ctx, d.layer),
                        SD::PleAddOutput => self.sub_ple_add_output(fwd, &cfg, ctx),
                    }
                }
                GraphNode::Transfer(t) => {
                    use crate::backend::vulkan::graph::node::SubDispatch as SD;
                    match t.sub_dispatch {
                        SD::GdnStateCopy => self.sub_gdn_state_copy(fwd, &cfg, ctx, t.layer),
                        SD::MoeFfnClear => self.sub_moe_ffn_clear(fwd, &cfg, ctx),
                        _ => unreachable!(
                            "TransferNode sub_dispatch must be a transfer variant, got {:?}",
                            t.sub_dispatch
                        ),
                    }
                }
            }
        }

        // Toggle back to imperative mode. Instead of an end-drain
        // (one extra barrier per layer), seed the imperative
        // pending_writes set with every buffer this layer's graph
        // wrote — so the next imperative consumer (next graph layer's
        // start-drain, or `dispatch_final`'s lm_head reads) will fire
        // a real barrier via the existing `maybe_compute_barrier`
        // mechanism. Net barrier count per layer: 1 inter-layer drain
        // + within-layer (typically 10) = ~11 — same ball-park as the
        // imperative path's ~10/layer.
        if use_graph_barriers {
            fwd.barrier_mode = BarrierMode::Imperative;
            fwd.pending_writes.clear();
            for node in &graph.nodes {
                for w in node.writes() {
                    use ash::vk::Handle;
                    fwd.pending_writes.insert(w.buffer.as_raw());
                }
            }
        }

        if std::env::var("VF_BARRIER_STATS").as_deref() == Ok("1") {
            eprintln!(
                "[GRAPH L{:>2}] {:>3} barriers emitted (graph nodes={}, edges={})",
                ctx.layer, barriers_emitted, graph.nodes.len(), graph.edges.len()
            );
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
            // Sprint D2 — same qwen35 carve-out as the DEC path above.
            // Sprint P1-5 — Qwen3.6 joins the fusion (default on, gated
            // by VF_FUSE_QWEN36_RESNORM); Gemma-4 stays excluded.
            if cfg.gemma4.is_none()
                && (cfg.qwen35.is_none() || dispatch::fuse_qwen36_resnorm_enabled())
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
