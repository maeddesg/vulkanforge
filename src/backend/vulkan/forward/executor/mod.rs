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
use super::super::gguf::{GgmlType, ModelConfig};
use super::super::loader::LoadedModel;
use super::super::pipeline::{GenericBinaryPushConstants, MatVecPushConstants, SwigluPushConstants};
use super::super::pipeline_registry::PipelineRegistry;
use super::super::shaders::ShaderId;

use super::arch::{
    GemmKind, compute_barrier, is_f32_layer_weight, is_fp8_layer_weight, layer_weight,
    layer_weight_opt, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_scale_scalar, layer_weight_shader, layer_weight_shader_gemm,
    transfer_to_compute_barrier,
};
use super::layer_plan::{ActivationKind, LayerPlan, LayerStep};
use super::state::Forward;
use super::super::loader::MoeRouterLayerData;

// Sprint 57B — submodule split. `dispatch` = keystone match. `attention`,
// `ffn` carry their respective step_* + b_step_* per-category.
mod attention;
mod dispatch;
mod ffn;

/// Sprint 51D-D — one-shot log-on-first-call for the layer-0 router
/// decision (debug aid). Prevents flooding decode-token logs with
/// per-token prints; only the first decode token reports.
static MOE_LAYER0_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Sprint 51D-D — COMPUTE_SHADER → TRANSFER barrier.
///
/// Equivalent of `transfer_to_compute_barrier` flipped: makes prior
/// shader writes visible to a `cmd_copy_buffer` reading those same
/// buffers. Used by the MoE route step to drain `scratch_b` (written
/// by `step_pre_moe_norm`'s rms_norm dispatch) before the host
/// readback copy.
fn compute_to_transfer_barrier(dev: &super::super::device::VulkanDevice, cmd: vk::CommandBuffer) {
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
fn transfer_to_host_barrier(dev: &super::super::device::VulkanDevice, cmd: vk::CommandBuffer) {
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
fn log_router_decision(layer: u32, routing: &[(u32, f32)]) {
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

    fn step_ple_block(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // 5-dispatch Gemma-4 PLE block. Mirrors decode.rs L1358–1453.
        let (_, output) = decode_io(ctx);
        let ple = fwd
            .config
            .gemma4
            .as_ref()
            .and_then(|_| ctx.model.ple_data.as_ref())
            .expect("PleBlock emitted but model has no PleData");
        let hps = ple.hps;
        let hps_bytes = (hps as u64) * 4;
        let ple_inputs_buf = fwd.cur().per_layer_inputs.handle;
        let ple_inputs_offset = (ctx.layer as u64) * hps_bytes;
        let gate_buf = fwd.cur().gate_buf.handle;
        let o_buf = fwd.cur().o_buf.handle;
        let attn_out = fwd.cur().attn_out.handle;

        // (1) gate = per_layer_input_gate @ output  — F32 GEMV (1536 → 256).
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[output]);
        let w_gate = layer_weight(ctx.model, ctx.layer, "per_layer_input_gate.weight");
        let s_gate = layer_weight_shader(
            ctx.model, ctx.layer, "per_layer_input_gate.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_gate = layer_weight_scale_scalar(
            ctx.model, ctx.layer, "per_layer_input_gate.weight",
        );
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s_gate, w_gate,
            output, gate_buf,
            cfg.hidden_dim, hps, scale_gate, "ple_gemv_gate",
        );
        fwd.mark_written(&[gate_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

        // (2) gate ← gelu_pytorch_tanh(gate) * per_layer_inputs[layer]
        let kernel = ctx.registry.get(ShaderId::GeluPytorchTanhGlu);
        let set = fwd.alloc_or_get_set(
            ctx.dev, kernel.descriptor_set_layout,
            &[
                (0, gate_buf, 0, hps_bytes),
                (1, ple_inputs_buf, ple_inputs_offset, hps_bytes),
                (2, gate_buf, 0, hps_bytes),
            ],
        );
        let pc = SwigluPushConstants { n: hps };
        let dispatch_x = (hps + 255) / 256;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        fwd.profile("ple_gelu_glu", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
        });
        fwd.mark_written(&[gate_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

        // (3) proj = per_layer_projection @ gate  — F32 GEMV (256 → 1536).
        let w_proj = layer_weight(ctx.model, ctx.layer, "per_layer_projection.weight");
        let s_proj = layer_weight_shader(
            ctx.model, ctx.layer, "per_layer_projection.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_proj = layer_weight_scale_scalar(
            ctx.model, ctx.layer, "per_layer_projection.weight",
        );
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s_proj, w_proj,
            gate_buf, o_buf,
            hps, cfg.hidden_dim, scale_proj, "ple_gemv_proj",
        );
        fwd.mark_written(&[o_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

        // (4) normed = rms_norm(proj, post_per_layer_input_norm.weight) → attn_out.
        let w_pln = layer_weight(ctx.model, ctx.layer, "post_per_layer_input_norm.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, o_buf, w_pln, attn_out,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "ple_rms_norm",
        );
        fwd.mark_written(&[attn_out]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[attn_out, output]);

        // (5) output += normed.
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            output, attn_out, output,
            cfg.hidden_dim, "add_ple",
        );
        fwd.mark_written(&[output]);
    }

    fn step_layer_scalar_mul(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (_, output) = decode_io(ctx);
        let scalar = layer_weight(ctx.model, ctx.layer, "layer_scalar");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[output]);
        fwd.run_mul_scalar_b(
            ctx.dev, ctx.registry, ctx.cmd, output, scalar, output,
            cfg.hidden_dim, "layer_scalar_mul",
        );
        fwd.mark_written(&[output]);
    }

    // === Sprint 51D-B Block 1 — Gemma-4-26B-A4B MoE FFN-Block norms + add ===

    /// `post_feedforward_layernorm_1` on Dense-MLP output.
    /// Reads `ffn_out` (DownProj output), writes `scratch_a` (= h1).
    fn step_post_dense_mlp_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let ffn_out = fwd.cur().ffn_out.handle;
        let scratch_a = fwd.cur().scratch_a.handle;
        let w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm_1.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_out]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, ffn_out, w, scratch_a,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_dense_mlp",
        );
        fwd.mark_written(&[scratch_a]);
    }

    /// `pre_feedforward_layernorm_2` on the post-attention residual
    /// (NOT on the Dense-MLP output — both branches read the same
    /// residual). Reads `res1`, writes `scratch_b` (MoE input).
    fn step_pre_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let res1 = fwd.cur().res1.handle;
        let scratch_b = fwd.cur().scratch_b.handle;
        let w = layer_weight(ctx.model, ctx.layer, "ffn_pre_norm_2.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[res1]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, res1, w, scratch_b,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_pre_moe",
        );
        fwd.mark_written(&[scratch_b]);
    }

    /// `post_feedforward_layernorm_2` on MoE output.
    /// Reads `ffn_hidden` (MoE expert weighted-sum), writes `ffn_out`
    /// (= h2, aliasing the freed Dense-output slot).
    fn step_post_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let ffn_hidden = fwd.cur().ffn_hidden.handle;
        let ffn_out = fwd.cur().ffn_out.handle;
        let w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm_2.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_hidden]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, ffn_hidden, w, ffn_out,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_moe",
        );
        fwd.mark_written(&[ffn_out]);
    }

    /// `ffn_out (h2) += scratch_a (h1)`. Distinct from
    /// `step_ffn_residual_add` which reads `res1` instead of `scratch_a`.
    fn step_moe_branch_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let ffn_out = fwd.cur().ffn_out.handle;
        let scratch_a = fwd.cur().scratch_a.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_out, scratch_a]);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            ffn_out, scratch_a, ffn_out,
            cfg.hidden_dim, "moe_branch_add",
        );
        fwd.mark_written(&[ffn_out]);
    }

    // === Sprint 51D-D — MoE routing + per-token expert FFN ===

    /// Copy the RAW post-attention residual (`res1`) to the
    /// host-readable staging buffer, mid-frame-submit so the GPU work
    /// drains, then run the router on the CPU and stash the Top-K
    /// `(expert_idx, weight)` tuples on `Forward` for the next step
    /// (`step_moe_expert_ffn`) to consume.
    ///
    /// Sprint 51D-F: the source is `res1`, NOT
    /// `scratch_b`/`pre_ff_norm_2(res1)`. HF
    /// `Gemma4TextDecoderLayer.forward` calls `self.router(residual)` on
    /// the raw post-attention residual; the router does its own
    /// parameterless RMS-norm + per-channel `scale` × `inv_sqrt(hidden)`
    /// internally. Routing on the pre_ff_norm_2-multiplied residual
    /// distorts the per-channel direction (γ_2 ≠ uniform) and picks the
    /// wrong Top-K experts.
    fn step_moe_route(
        &self, fwd: &mut Forward, _cfg: &ModelConfig, ctx: &ExecCtx,
        n_experts: u32, top_k: u32,
    ) {
        let router_data = ctx.model.moe_router_data
            .as_ref()
            .expect("MoeRoute step emitted but model has no MoeRouterData");
        debug_assert_eq!(router_data.n_experts, n_experts);
        debug_assert_eq!(router_data.top_k, top_k);
        let hidden = router_data.hidden_size;

        // ─── Sprint 56B: GPU-side router (if init_moe_router_gpu ran) ───
        if fwd.moe_router_gpu.is_some() {
            let res1 = fwd.cur().res1.handle;
            // Make the SHADER_WRITE on res1 (from step_attn_residual_add)
            // visible to the router's SHADER_READ. dispatch_layer's
            // maybe_compute_barrier doesn't fire here because res1 isn't
            // in the pending-writes set after the intervening dense FFN
            // pass — emit the barrier inline.
            let bar_pre = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar_pre), &[], &[],
                );
            }
            fwd.run_moe_router_gpu(
                ctx.dev, ctx.registry, ctx.cmd, ctx.layer, 1, res1, 0,
            );

            // Copy indices + weights into a single host-visible staging
            // buffer. The two GPU buffers are addressed independently;
            // we lay them out in `readback_staging` as
            // [indices | weights], each `top_k * 4` bytes.
            let router = fwd.moe_router_gpu.as_ref().unwrap();
            let topk_bytes = (top_k as u64) * 4;
            let indices_h = router.indices_scratch.handle;
            let weights_h = router.weights_scratch.handle;
            let staging_h = router.readback_staging.handle;
            // Compute → Transfer barrier on both indices + weights.
            let bar_post = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar_post), &[], &[],
                );
                let cp_idx = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: topk_bytes };
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, indices_h, staging_h, std::slice::from_ref(&cp_idx),
                );
                let cp_w = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: topk_bytes,
                    size: topk_bytes,
                };
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, weights_h, staging_h, std::slice::from_ref(&cp_w),
                );
            }
            transfer_to_host_barrier(ctx.dev, ctx.cmd);
            fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)
                .expect("mid_frame_submit_and_wait failed (GPU router readback)");

            // Read the staging blob and decode into (u32, f32) pairs.
            let raw = fwd.moe_router_gpu.as_ref().unwrap()
                .readback_staging.read_bytes()
                .expect("router readback staging not host-visible");
            let topk_usize = top_k as usize;
            let idx_slice: &[u32] = bytemuck::cast_slice(&raw[..(topk_usize * 4)]);
            let w_slice: &[f32] = bytemuck::cast_slice(
                &raw[(topk_usize * 4)..(topk_usize * 8)]
            );
            let routing: Vec<(u32, f32)> = idx_slice.iter()
                .zip(w_slice.iter())
                .map(|(&i, &w)| (i, w))
                .collect();

            if ctx.layer == 0
                && !MOE_LAYER0_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                log_router_decision(0, &routing);
            }
            fwd.moe_routing = Some(routing);
            return;
        }

        // ─── CPU fallback (legacy path, retained for non-Gemma-4 / unit tests) ───
        let bytes = (hidden as u64) * 4;
        let res1 = fwd.cur().res1.handle;
        compute_to_transfer_barrier(ctx.dev, ctx.cmd);
        let staging = fwd.moe_route_staging.handle;
        unsafe {
            let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes };
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, res1, staging, std::slice::from_ref(&region),
            );
        }
        transfer_to_host_barrier(ctx.dev, ctx.cmd);
        fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)
            .expect("mid_frame_submit_and_wait failed (decode MoE route)");
        let raw = fwd.moe_route_staging.read_bytes()
            .expect("moe_route_staging not host-visible");
        let hidden_state: Vec<f32> = raw[..(bytes as usize)]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let layer_data = &router_data.layers[ctx.layer as usize];
        let routing = cpu_moe_route(
            &hidden_state, layer_data,
            hidden as usize, n_experts as usize, top_k as usize,
            router_data.rms_norm_eps,
        );
        if ctx.layer == 0 && !MOE_LAYER0_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            log_router_decision(0, &routing);
        }
        fwd.moe_routing = Some(routing);
    }

    /// Per-token K-expert FFN. Reads the Top-K `(idx, weight)` tuples
    /// stashed by `step_moe_route`, dispatches `gate_up + GLU + down`
    /// once per expert against offset slices of the packed weight
    /// tensors, and accumulates each expert's `[hidden]` output into
    /// `ffn_hidden` weighted by the renormalized router weight.
    fn step_moe_expert_ffn(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
        n_experts: u32, _top_k: u32, moe_intermediate: u32,
    ) {
        let routing = fwd.moe_routing.take()
            .expect("MoeExpertFfn before MoeRoute populated routing");
        let scratch_b   = fwd.cur().scratch_b.handle;
        let gate_buf    = fwd.cur().gate_buf.handle;
        let up_buf      = fwd.cur().up_buf.handle;
        // Sprint 51D-D — per-expert down-GEMV output goes to o_buf
        // (hidden-sized, free during the FFN block) because gate_buf
        // is `ffn_bytes`-sized which is < hidden_bytes on 26B
        // (intermediate_size=2112 < hidden_size=2816).
        let o_buf       = fwd.cur().o_buf.handle;
        let ffn_hidden  = fwd.cur().ffn_hidden.handle;
        let h           = cfg.hidden_dim;
        let mi          = moe_intermediate;
        // Sprint 51D-D — padded-K alignment for `down_proj`. When
        // `moe_intermediate_size` (= K of the down GEMV) is not a
        // multiple of 256, the loader pads each row to the next 256
        // boundary in the Q4_K representation. The GEMV shader must
        // then push `ncols=mi_padded` (not `mi`) so it iterates over
        // the full 3 blocks per row instead of integer-truncating to
        // 2. `gate_up_proj` always has K=hidden, which is 256-aligned
        // for the shipped Gemma-4 configs, so no padding there.
        let mi_padded: u32 = ((mi + 255) / 256) * 256;
        // Sprint 52K — was hardcoded `* 144 / 256` (Q4_K block bytes /
        // K-quant block size) AND assumed Sprint-51D-D's `mi_padded` row
        // padding. That's correct for the SafeTensors path where Q4_K
        // pad-quantises down_proj rows at load (51D-D `padded_rows`
        // helper), but **wrong for GGUF**: llama.cpp stores 26B's
        // experts as Q3_K (gate_up) and Q5_0 (down) **without** row
        // padding, because Q3_K(K=2816) and Q5_0(K=704) are already
        // multiples of their respective block sizes (256 / 32).
        //
        // Robust fix: derive `bytes_per_expert` from the actual tensor's
        // `byte_size` / `n_experts` — that's the ground truth, padded
        // or not. Works for both the SafeTensors-quantised Q4_K and the
        // native GGUF Q3_K / Q5_0. Identical patch lands in
        // `step_moe_expert_ffn` (DEC path) AND `b_step_moe_expert_ffn`
        // (BAT path).
        let gate_up_t = ctx
            .model
            .tensor(&format!("blk.{}.moe_experts.gate_up_proj", ctx.layer))
            .expect("moe_experts.gate_up_proj missing");
        let down_t = ctx
            .model
            .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
            .expect("moe_experts.down_proj missing");
        let gate_up_bytes_per_expert: u64 = gate_up_t.byte_size / (n_experts as u64);
        let down_bytes_per_expert: u64 = down_t.byte_size / (n_experts as u64);
        let _ = mi_padded; // still computed for downstream barrier sizes
        // Sprint 52K — per-expert GEMV shader is now picked from the
        // actual weight ggml_type (was hardcoded Q4_K). 26B Q3_K_M has
        // gate_up=Q3_K + down=Q5_0 → without this fix, both dispatch
        // the Q4_K shader and read wrong block strides → garbage.
        let gate_up_shader = layer_weight_shader(
            ctx.model, ctx.layer,
            "moe_experts.gate_up_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let down_shader = layer_weight_shader(
            ctx.model, ctx.layer,
            "moe_experts.down_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let gate_up_w = layer_weight(ctx.model, ctx.layer, "moe_experts.gate_up_proj");
        let down_w    = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");

        // Zero the accumulator before the per-expert loop. Replaces a
        // dedicated "zero buffer" shader; the COMPUTE→TRANSFER barrier
        // around it is implicit because we just begin'd a fresh CB
        // (Sprint 51D-C primitive resets command-buffer state and the
        // barrier-elision tracker before this step runs).
        unsafe {
            ctx.dev.device.cmd_fill_buffer(
                ctx.cmd, ffn_hidden, 0, (h as u64) * 4, 0,
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        for &(expert_idx, weight) in &routing {
            // (a) gate_up GEMV: scratch_b × experts.gate_up_proj[e] → gate_buf [2*mi].
            let gate_up_off = (expert_idx as u64) * gate_up_bytes_per_expert;
            fwd.run_gemv_q4k_at_offset(
                ctx.dev, ctx.registry, ctx.cmd,
                gate_up_w, gate_up_off,
                scratch_b, gate_buf,
                h, 2 * mi,
                "moe_gate_up",
                gate_up_shader,
            );
            fwd.mark_written(&[gate_buf]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

            // (b) GeluPytorchTanhGlu: gate_buf[0..mi] (gate) × gelu(gate_buf[mi..2mi]) (up) → up_buf [mi].
            //     The shader expects 3 SSBOs (gate, up, output); we
            //     bind gate_buf at two different offsets to feed both
            //     halves of the packed gate_up output.
            let kernel = ctx.registry.get(ShaderId::GeluPytorchTanhGlu);
            let mi_bytes = (mi as u64) * 4;
            let set = fwd.alloc_or_get_set(
                ctx.dev, kernel.descriptor_set_layout,
                &[
                    (0, gate_buf, 0, mi_bytes),
                    (1, gate_buf, mi_bytes, mi_bytes),
                    (2, up_buf, 0, mi_bytes),
                ],
            );
            let pc = SwigluPushConstants { n: mi };
            let dispatch_x = (mi + 255) / 256;
            let layout = kernel.pipeline_layout;
            let pipeline = kernel.pipeline;
            fwd.profile("moe_glu", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                dev.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                );
                dev.device.cmd_push_constants(
                    cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                );
                dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
            });
            fwd.mark_written(&[up_buf]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[up_buf]);

            // (c) down GEMV: up_buf × experts.down_proj[e] → o_buf [hidden].
            //     Uses padded K (`mi_padded`); the input vector binding
            //     extends past `mi` valid floats but the corresponding
            //     padded weight columns are quantized zeros so the
            //     extra contributions are exactly 0.
            let down_off = (expert_idx as u64) * down_bytes_per_expert;
            // Sprint 52P — loader-aware K. GGUF and SafeTensors store
            // the 3-D MoE tensor with mirrored axis order:
            //   GGUF:        shape = [K, M, n_experts]
            //   SafeTensors: shape = [n_experts, M, K]
            // Sprint 52M used `shape[0]` unconditionally — correct for
            // GGUF (Q5_0 K=704) but read n_experts=128 on SafeTensors,
            // turning the down GEMV into a zero-output stride disaster.
            // Branch on whether `shape[0]` matches `n_experts`.
            let down_k = if down_t.shape[0] as u32 == n_experts {
                mi_padded
            } else {
                down_t.shape[0] as u32
            };
            fwd.run_gemv_q4k_at_offset(
                ctx.dev, ctx.registry, ctx.cmd,
                down_w, down_off,
                up_buf, o_buf,
                down_k, h,
                "moe_down",
                down_shader,
            );
            fwd.mark_written(&[o_buf]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

            // (d) ffn_hidden += weight * o_buf  (per-expert weighted accumulation).
            fwd.run_fma_add(
                ctx.dev, ctx.registry, ctx.cmd,
                o_buf, ffn_hidden, h, weight,
                "moe_fma_add",
            );
            fwd.mark_written(&[ffn_hidden]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_hidden]);
        }
    }
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

    fn b_step_ple_block(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Sprint 46D Step 4 — per-token PLE block for batch-prefill.
        //
        // Semantically identical to DecodeExec::step_ple_block but
        // looped over `seq_len`. Each iteration binds the relevant
        // per-token slice of `batch_residual` / scratch / per_layer_inputs
        // via descriptor (offset, range) so the small GEMV / Add / RMSNorm
        // shaders see a single 1×K vector. ~5 dispatches × seq_len ×
        // 35 layers per Gemma-4 forward — slow but correct. An
        // optimisation pass (single batched-GEMM PLE) is deferred until
        // after Sprint 46F lifts `force_per_token_prefill`.
        let seq_len = batch_seq_len(ctx);
        let g = cfg.gemma4.as_ref().expect("PleBlock without gemma4 config");
        let hps = g.hidden_size_per_layer_input;
        let hps_bytes = (hps as u64) * 4;
        let nl = cfg.n_layers as u64;
        let row_bytes_per_token = nl * hps_bytes;
        let layer_off_in_row = (ctx.layer as u64) * hps_bytes;
        let hidden = cfg.hidden_dim;
        let hidden_bytes = (hidden as u64) * 4;
        let (head_dim, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let q_dim_bytes = (q_dim as u64) * 4;
        let ffn_dim_bytes = (ffn_dim as u64) * 4;

        let ple_inputs = fwd.cur().per_layer_inputs.handle;
        let output = fwd.batch_residual.handle;
        let scratch_gate = fwd.batch_gate.handle;
        let scratch_proj = fwd.batch_o.handle;
        let scratch_normed = fwd.batch_attn_out.handle;

        let w_gate = layer_weight(ctx.model, ctx.layer, "per_layer_input_gate.weight");
        let w_proj = layer_weight(ctx.model, ctx.layer, "per_layer_projection.weight");
        let w_pln = layer_weight(ctx.model, ctx.layer, "post_per_layer_input_norm.weight");
        let s_gate = layer_weight_shader(
            ctx.model, ctx.layer, "per_layer_input_gate.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let s_proj = layer_weight_shader(
            ctx.model, ctx.layer, "per_layer_projection.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_gate = layer_weight_scale_scalar(
            ctx.model, ctx.layer, "per_layer_input_gate.weight",
        );
        let scale_proj = layer_weight_scale_scalar(
            ctx.model, ctx.layer, "per_layer_projection.weight",
        );

        // batch_residual was just written by FfnResidualAdd /
        // LayerScalarMul. Make those writes visible to step (1).
        compute_barrier(ctx.dev, ctx.cmd);

        for t in 0..seq_len {
            let t_u64 = t as u64;
            let out_off = t_u64 * hidden_bytes;
            let gate_off = t_u64 * ffn_dim_bytes;
            let proj_off = t_u64 * hidden_bytes;
            let normed_off = t_u64 * q_dim_bytes;
            let ple_off = t_u64 * row_bytes_per_token + layer_off_in_row;

            // (1) gate_t = per_layer_input_gate @ output[t]  (1536 → 256).
            {
                let kernel = ctx.registry.get(s_gate);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, w_gate, 0, 0),
                        (1, output, out_off, hidden_bytes),
                        (2, scratch_gate, gate_off, hps_bytes),
                    ],
                );
                let pc = MatVecPushConstants {
                    ncols: hidden, stride_a: hidden, stride_b: hidden, stride_d: hps,
                    batch_stride_a: hidden * hps, batch_stride_b: hidden, batch_stride_d: hps,
                    fusion_flags: 0, base_work_group_y: 0,
                    ne02: 1, ne12: 1, broadcast2: 1,
                    broadcast3: scale_gate.to_bits(),
                };
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("ple_gemv_gate_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, hps, 1, 1);
                });
                compute_barrier(ctx.dev, ctx.cmd);
            }

            // (2) gate_t ← gelu_pytorch_tanh(gate_t) * per_layer_inputs[t, layer].
            {
                let kernel = ctx.registry.get(ShaderId::GeluPytorchTanhGlu);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, scratch_gate, gate_off, hps_bytes),
                        (1, ple_inputs, ple_off, hps_bytes),
                        (2, scratch_gate, gate_off, hps_bytes),
                    ],
                );
                let pc = SwigluPushConstants { n: hps };
                let dispatch_x = (hps + 255) / 256;
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("ple_gelu_glu_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
                });
                compute_barrier(ctx.dev, ctx.cmd);
            }

            // (3) proj_t = per_layer_projection @ gate_t  (256 → 1536).
            {
                let kernel = ctx.registry.get(s_proj);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, w_proj, 0, 0),
                        (1, scratch_gate, gate_off, hps_bytes),
                        (2, scratch_proj, proj_off, hidden_bytes),
                    ],
                );
                let pc = MatVecPushConstants {
                    ncols: hps, stride_a: hps, stride_b: hps, stride_d: hidden,
                    batch_stride_a: hps * hidden, batch_stride_b: hps, batch_stride_d: hidden,
                    fusion_flags: 0, base_work_group_y: 0,
                    ne02: 1, ne12: 1, broadcast2: 1,
                    broadcast3: scale_proj.to_bits(),
                };
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("ple_gemv_proj_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, hidden, 1, 1);
                });
                compute_barrier(ctx.dev, ctx.cmd);
            }

            // (4) normed_t = rms_norm(proj_t, post_per_layer_input_norm.weight).
            {
                let kernel = ctx.registry.get(ShaderId::RmsNorm);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, scratch_proj, proj_off, hidden_bytes),
                        (1, w_pln, 0, 0),
                        (2, scratch_normed, normed_off, hidden_bytes),
                    ],
                );
                let pc = GenericBinaryPushConstants {
                    ne: hidden,
                    ne00: hidden, ne01: 1, ne02: 1, ne03: 1,
                    nb00: 1, nb01: hidden, nb02: hidden, nb03: hidden,
                    ne10: hidden, ne11: 1, ne12: 1, ne13: 1,
                    nb10: 1, nb11: hidden, nb12: hidden, nb13: hidden,
                    ne20: hidden, ne21: 1, ne22: 1, ne23: 1,
                    nb20: 1, nb21: hidden, nb22: hidden, nb23: hidden,
                    misalign_offsets: 0,
                    param1: cfg.rms_norm_eps, param2: 0.0, param3: 0,
                };
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("ple_rms_norm_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, 1, 1, 1);
                });
                compute_barrier(ctx.dev, ctx.cmd);
            }

            // (5) output[t] += normed_t.
            {
                let kernel = ctx.registry.get(ShaderId::Add);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, output, out_off, hidden_bytes),
                        (1, scratch_normed, normed_off, hidden_bytes),
                        (2, output, out_off, hidden_bytes),
                    ],
                );
                let pc = GenericBinaryPushConstants {
                    ne: hidden,
                    ne00: hidden, ne01: 1, ne02: 1, ne03: 1,
                    nb00: 1, nb01: hidden, nb02: hidden, nb03: hidden,
                    ne10: hidden, ne11: 1, ne12: 1, ne13: 1,
                    nb10: 1, nb11: hidden, nb12: hidden, nb13: hidden,
                    ne20: hidden, ne21: 1, ne22: 1, ne23: 1,
                    nb20: 1, nb21: hidden, nb22: hidden, nb23: hidden,
                    misalign_offsets: 0,
                    param1: 0.0, param2: 0.0, param3: 0,
                };
                let dispatch_y = (hidden + 511) / 512;
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("ple_add_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, 1, dispatch_y, 1);
                });
                // Cross-token barrier: token (t+1)'s step (1) reads
                // batch_residual at out_off + hidden_bytes — independent
                // range — but we still need WAW visibility for any
                // future consumer of `output` past the loop tail.
                compute_barrier(ctx.dev, ctx.cmd);
            }
        }
    }

    fn b_step_layer_scalar_mul(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 only.
        let seq_len = batch_seq_len(ctx);
        let scalar = layer_weight(ctx.model, ctx.layer, "layer_scalar");
        fwd.run_mul_scalar_b(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_residual.handle, scalar, fwd.batch_residual.handle,
            seq_len * cfg.hidden_dim, "layer_scalar_mul_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    // === Sprint 51D-B Block 1 — Gemma-4-26B-A4B MoE FFN-Block (batch) ===
    //
    // Buffer aliasing on the existing batch buffers:
    //   batch_ffn_out    Dense-MLP DownProj output (existing)
    //                    -> after PostMoeNorm: h2 (alias the freed slot)
    //                    -> after MoeBranchAdd: h1 + h2
    //   batch_norm       PreFfnNorm output (existing, free after DownProj)
    //                    -> h1 after PostDenseMlpNorm
    //   batch_o          OProj output (existing, free after AttnResidualAdd)
    //                    -> MoE input after PreMoeNorm
    //   batch_ffn_hidden FFN intermediate (existing) -> MoE expert sum

    fn b_step_post_dense_mlp_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm_1.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_ffn_out.handle, w, fwd.batch_norm.handle,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps,
            "rms_norm_post_dense_mlp_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_pre_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Reads batch_residual (NOT the Dense-MLP output) — both
        // branches operate on the same post-attention residual.
        let seq_len = batch_seq_len(ctx);
        let w = layer_weight(ctx.model, ctx.layer, "ffn_pre_norm_2.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_residual.handle, w, fwd.batch_o.handle,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps,
            "rms_norm_pre_moe_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_post_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm_2.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_ffn_hidden.handle, w, fwd.batch_ffn_out.handle,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps,
            "rms_norm_post_moe_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_moe_branch_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            fwd.batch_ffn_out.handle, fwd.batch_norm.handle, fwd.batch_ffn_out.handle,
            seq_len * cfg.hidden_dim, "moe_branch_add_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    // === Sprint 51D-D — Batch (prefill) MoE routing + expert FFN ===

    fn b_step_moe_route(
        &self, fwd: &mut Forward, _cfg: &ModelConfig, ctx: &ExecCtx,
        n_experts: u32, top_k: u32,
    ) {
        let router_data = ctx.model.moe_router_data
            .as_ref()
            .expect("MoeRoute step (batch) emitted but model has no MoeRouterData");
        let hidden = router_data.hidden_size;
        let seq_len = batch_seq_len(ctx);

        // ─── Sprint 56B: GPU-side router (if init_moe_router_gpu ran) ───
        if fwd.moe_router_gpu.is_some() {
            let batch_in = fwd.batch_residual.handle;
            // SHADER_WRITE → SHADER_READ on batch_residual: the
            // attn_residual_add step writes it earlier in the layer.
            let bar_pre = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar_pre), &[], &[],
                );
            }
            fwd.run_moe_router_gpu(
                ctx.dev, ctx.registry, ctx.cmd, ctx.layer, seq_len, batch_in, 0,
            );

            // Copy seq_len × top_k × (4+4) bytes into readback_staging.
            let router = fwd.moe_router_gpu.as_ref().unwrap();
            let topk_bytes = (top_k as u64) * 4;
            let all_topk_bytes = (seq_len as u64) * topk_bytes;
            let indices_h = router.indices_scratch.handle;
            let weights_h = router.weights_scratch.handle;
            let staging_h = router.readback_staging.handle;
            let bar_post = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar_post), &[], &[],
                );
                let cp_idx = vk::BufferCopy {
                    src_offset: 0, dst_offset: 0, size: all_topk_bytes,
                };
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, indices_h, staging_h, std::slice::from_ref(&cp_idx),
                );
                let cp_w = vk::BufferCopy {
                    src_offset: 0, dst_offset: all_topk_bytes, size: all_topk_bytes,
                };
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, weights_h, staging_h, std::slice::from_ref(&cp_w),
                );
            }
            transfer_to_host_barrier(ctx.dev, ctx.cmd);
            fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)
                .expect("mid_frame_submit_and_wait failed (batch GPU router readback)");

            let raw = fwd.moe_router_gpu.as_ref().unwrap()
                .readback_staging.read_bytes()
                .expect("router readback staging not host-visible");
            let topk_usize = top_k as usize;
            let total_topk = (seq_len as usize) * topk_usize;
            let idx_slice: &[u32] = bytemuck::cast_slice(&raw[..(total_topk * 4)]);
            let w_slice: &[f32] = bytemuck::cast_slice(
                &raw[(total_topk * 4)..(total_topk * 8)]
            );
            let mut all_routing: Vec<Vec<(u32, f32)>> = Vec::with_capacity(seq_len as usize);
            for t in 0..(seq_len as usize) {
                let off = t * topk_usize;
                let routing: Vec<(u32, f32)> = (0..topk_usize)
                    .map(|k| (idx_slice[off + k], w_slice[off + k]))
                    .collect();
                if ctx.layer == 0 && t == 0
                    && !MOE_LAYER0_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    log_router_decision(0, &routing);
                }
                all_routing.push(routing);
            }
            fwd.moe_routing_batch = Some(all_routing);
            return;
        }

        // ─── CPU fallback (legacy) ───
        let total_bytes = (seq_len as u64) * (hidden as u64) * 4;
        let batch_in = fwd.batch_residual.handle;
        compute_to_transfer_barrier(ctx.dev, ctx.cmd);
        let staging = fwd.moe_route_staging.handle;
        unsafe {
            let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: total_bytes };
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, batch_in, staging, std::slice::from_ref(&region),
            );
        }
        transfer_to_host_barrier(ctx.dev, ctx.cmd);
        fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)
            .expect("mid_frame_submit_and_wait failed (batch MoE route)");
        let raw = fwd.moe_route_staging.read_bytes()
            .expect("moe_route_staging not host-visible");
        let mut all_routing: Vec<Vec<(u32, f32)>> = Vec::with_capacity(seq_len as usize);
        let layer_data = &router_data.layers[ctx.layer as usize];
        let h_usize = hidden as usize;
        for t in 0..(seq_len as usize) {
            let off = t * h_usize * 4;
            let token_hidden: Vec<f32> = raw[off..off + h_usize * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let routing = cpu_moe_route(
                &token_hidden, layer_data,
                h_usize, n_experts as usize, top_k as usize,
                router_data.rms_norm_eps,
            );
            if ctx.layer == 0 && t == 0
                && !MOE_LAYER0_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                log_router_decision(0, &routing);
            }
            all_routing.push(routing);
        }
        fwd.moe_routing_batch = Some(all_routing);
    }

    fn b_step_moe_expert_ffn(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
        n_experts: u32, _top_k: u32, moe_intermediate: u32,
    ) {
        let routing_batch = fwd.moe_routing_batch.take()
            .expect("MoeExpertFfn (batch) before MoeRoute populated routing");
        let seq_len = batch_seq_len(ctx);
        let h = cfg.hidden_dim;
        let mi = moe_intermediate;
        let h_bytes = (h as u64) * 4;
        let mi_bytes = (mi as u64) * 4;
        // Sprint 51D-D — padded-K alignment (see decode-side comment).
        let mi_padded: u32 = ((mi + 255) / 256) * 256;
        // Sprint 52K — was hardcoded `* 144 / 256` (Q4_K block bytes /
        // K-quant block size) AND assumed Sprint-51D-D's `mi_padded` row
        // padding. That's correct for the SafeTensors path where Q4_K
        // pad-quantises down_proj rows at load (51D-D `padded_rows`
        // helper), but **wrong for GGUF**: llama.cpp stores 26B's
        // experts as Q3_K (gate_up) and Q5_0 (down) **without** row
        // padding, because Q3_K(K=2816) and Q5_0(K=704) are already
        // multiples of their respective block sizes (256 / 32).
        //
        // Robust fix: derive `bytes_per_expert` from the actual tensor's
        // `byte_size` / `n_experts` — that's the ground truth, padded
        // or not. Works for both the SafeTensors-quantised Q4_K and the
        // native GGUF Q3_K / Q5_0. Identical patch lands in
        // `step_moe_expert_ffn` (DEC path) AND `b_step_moe_expert_ffn`
        // (BAT path).
        let gate_up_t = ctx
            .model
            .tensor(&format!("blk.{}.moe_experts.gate_up_proj", ctx.layer))
            .expect("moe_experts.gate_up_proj missing");
        let down_t = ctx
            .model
            .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
            .expect("moe_experts.down_proj missing");
        let gate_up_bytes_per_expert: u64 = gate_up_t.byte_size / (n_experts as u64);
        let down_bytes_per_expert: u64 = down_t.byte_size / (n_experts as u64);
        let _ = mi_padded; // still computed for downstream barrier sizes
        // Sprint 52K — per-expert GEMV shader is now picked from the
        // actual weight ggml_type (was hardcoded Q4_K). 26B Q3_K_M has
        // gate_up=Q3_K + down=Q5_0 → without this fix, both dispatch
        // the Q4_K shader and read wrong block strides → garbage.
        let gate_up_shader = layer_weight_shader(
            ctx.model, ctx.layer,
            "moe_experts.gate_up_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let down_shader = layer_weight_shader(
            ctx.model, ctx.layer,
            "moe_experts.down_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let gate_up_w = layer_weight(ctx.model, ctx.layer, "moe_experts.gate_up_proj");
        let down_w    = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");
        let batch_in = fwd.batch_o.handle;
        let batch_out = fwd.batch_ffn_hidden.handle;

        // Zero the full batch_ffn_hidden output slab once.
        unsafe {
            ctx.dev.device.cmd_fill_buffer(
                ctx.cmd, batch_out, 0, (seq_len as u64) * h_bytes, 0,
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        // Per-token K-expert loop. Reuses the decode-slot per-token
        // gate_buf / up_buf for the gate_up + GLU intermediates (sized
        // ≥ 2*moe_int on 26B) and o_buf for the per-expert down output
        // (hidden-sized, free during the FFN block).
        let gate_buf = fwd.cur().gate_buf.handle;
        let up_buf   = fwd.cur().up_buf.handle;
        let o_buf    = fwd.cur().o_buf.handle;
        for t in 0..(seq_len as usize) {
            let in_off  = (t as u64) * h_bytes;
            let out_off = (t as u64) * h_bytes;
            let routing = &routing_batch[t];
            for &(expert_idx, weight) in routing {
                let gate_up_off = (expert_idx as u64) * gate_up_bytes_per_expert;
                let down_off    = (expert_idx as u64) * down_bytes_per_expert;

                // (a) gate_up GEMV: batch_in[t] × experts.gate_up_proj[e] → gate_buf [2*mi].
                fwd.run_gemv_q4k_at_offset_inout(
                    ctx.dev, ctx.registry, ctx.cmd,
                    gate_up_w, gate_up_off,
                    batch_in, in_off, (h as u64) * 4,
                    gate_buf, 0, (2 * mi as u64) * 4,
                    h, 2 * mi,
                    "moe_gate_up_b",
                    gate_up_shader,
                );
                fwd.mark_written(&[gate_buf]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

                // (b) GLU(gate, up) → up_buf [mi].
                let kernel = ctx.registry.get(ShaderId::GeluPytorchTanhGlu);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, gate_buf, 0, mi_bytes),
                        (1, gate_buf, mi_bytes, mi_bytes),
                        (2, up_buf, 0, mi_bytes),
                    ],
                );
                let pc = SwigluPushConstants { n: mi };
                let dispatch_x = (mi + 255) / 256;
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("moe_glu_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
                });
                fwd.mark_written(&[up_buf]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[up_buf]);

                // (c) down GEMV: up_buf × experts.down_proj[e] → o_buf [hidden].
                //     `mi_padded` matches decode path. up_buf range is
                //     `mi_bytes` (not the padded length) — the binding
                //     stops there and the shader will read past the
                //     range bound at most into the next valid buffer
                //     region; with WHOLE_SIZE-binding semantics the
                //     extra reads are in-bounds for the buffer (8448
                //     bytes ≥ mi_padded × 4 = 3072 bytes) and the
                //     padded weight columns are zero.
                // Sprint 52P — loader-aware K (mirror of the DEC fix in
                // `step_moe_expert_ffn`). GGUF stores [K, M, n_experts]
                // while SafeTensors stores [n_experts, M, K]; Sprint
                // 52M's `shape[0]` read K on GGUF but n_experts on
                // SafeTensors → zero MoE output → 26B regression from
                // the Sprint 51D-AN "Paris" baseline (b57e935). The
                // `up_buf` binding range tracks `down_k * 4` so the
                // shader's input stride matches the per-tensor K.
                let down_k = if down_t.shape[0] as u32 == n_experts {
                    mi_padded
                } else {
                    down_t.shape[0] as u32
                };
                fwd.run_gemv_q4k_at_offset_inout(
                    ctx.dev, ctx.registry, ctx.cmd,
                    down_w, down_off,
                    up_buf, 0, (down_k as u64) * 4,
                    o_buf, 0, h_bytes,
                    down_k, h,
                    "moe_down_b",
                    down_shader,
                );
                fwd.mark_written(&[o_buf]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

                // (d) batch_ffn_hidden[t] += weight * o_buf  (FmaAdd offset binding).
                fwd.run_fma_add_at_offset(
                    ctx.dev, ctx.registry, ctx.cmd,
                    o_buf, 0,
                    batch_out, out_off,
                    h, weight,
                    "moe_fma_add_b",
                );
                fwd.mark_written(&[batch_out]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[batch_out]);
            }
        }
        // Final compute_barrier so the next step (PostMoeNorm) sees a
        // clean dirty-tracker; mirrors the trailing barrier the other
        // batch steps emit.
        compute_barrier(ctx.dev, ctx.cmd);
    }

    // ── shared GEMM-routing helper ───────────────────────────────────

    /// Dispatch a single batched GEMM projection. Routes through one of:
    /// - `run_gemm_coopmat_q4k` (when `coopmat_q4k_enabled` for K-quant
    ///   weights),
    /// - `run_gemm_fp8_blockwise` / `run_gemm_fp8_naive` (FP8 weights),
    /// - `run_gemm` with the right `mul_mm` / `mul_mmq` shader.
    ///
    /// State-less: each MMQ call re-quantises its input. Bit-identical
    /// to the legacy gather-once pattern; pays 3 extra `quantize_q8_1`
    /// dispatches per layer in MMQ mode (negligible against the GEMM
    /// cost itself).
    /// `quantize_input = true` for the *first* projection of a stage
    /// (QProj for the attn stage, OProj alone, GateProj for the FFN
    /// stage, DownProj alone). The follow-up projections in the same
    /// stage (KProj/VProj after QProj; UpProj after GateProj) call
    /// with `false` and reuse the `batch_q8` buffer the first
    /// projection populated. This matches the legacy
    /// `dispatch_layer_batch` "quantize-once-per-stage" pattern and
    /// avoids the WAR hazard a per-projection re-quantize would
    /// introduce on `batch_q8`.
    #[allow(clippy::too_many_arguments)]
    fn b_run_proj(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        suffix: &str,
        input_fp32: vk::Buffer,
        output: vk::Buffer,
        m: u32,
        k: u32,
        n: u32,
        label: &'static str,
        quantize_input: bool,
    ) {
        let _ = cfg;
        let weight = layer_weight(ctx.model, ctx.layer, suffix);

        // Sprint 46C — F32 weights (Gemma-4 SafeTensors) take the
        // dedicated `mul_mm_f32{,_aligned}.spv` lane (Sprint 46B). No
        // Mmq variant exists for F32 (mul_mmq quantises activations to
        // Q8_1 and dequantises K-quant blocks per dispatch), so the
        // gemm_kind decision below is forced to MulMm/MulMmAligned and
        // the activation buffer stays FP32 (no `run_quantize_q8_1`
        // dispatch). Skips coopmat_q4k path because that shader assumes
        // packed K-quant blocks.
        let f32_weight = is_f32_layer_weight(ctx.model, ctx.layer, suffix);


        // Sprint 17B/17C/17D — Q4_0 forces MMQ (no MulMm SPV shipped).
        // Sprint 52J — Q5_0 / Q5_1 / Q8_0 also Mmq-only (mul_mm coopmat
        // variants would each need their own SPV family per quant —
        // out of scope; the integer-MMQ path covers 26B Q3_K_M's
        // attn_k/v (Q8_0), MoE down_exps (Q5_0), blk.0.ffn_down (Q5_1)).
        // Decision is **per-tensor** now (was only attn_q.weight) so
        // mixed-quant 26B GGUFs route each GEMM site correctly: attn_q
        // may be Q6_K (uses MulMm), attn_k Q8_0 (forces Mmq), etc.
        let weight_ggml_type = ctx
            .model
            .tensor(&format!("blk.{}.{suffix}", ctx.layer))
            .map(|t| t.ggml_type);
        let force_mmq = matches!(
            weight_ggml_type,
            Some(GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0),
        );
        let gemm_kind = if f32_weight {
            // F32 path: MulMmAligned when seq_len%4==0, MulMm otherwise.
            // Mmq is not a valid choice (no F32 Mmq shader), so we don't
            // even consider it.
            if n % 4 == 0 { GemmKind::MulMmAligned } else { GemmKind::MulMm }
        } else if force_mmq {
            GemmKind::Mmq
        } else if fwd.mul_mm_coopmat_enabled {
            GemmKind::MulMm
        } else if fwd.mul_mm_enabled {
            if n % 4 == 0 { GemmKind::MulMmAligned } else { GemmKind::Mmq }
        } else {
            GemmKind::Mmq
        };
        let use_mul_mm = matches!(gemm_kind, GemmKind::MulMm | GemmKind::MulMmAligned);

        // Coopmat path takes FP32 directly; pad seq_len up to tile.
        // Padding only fires on the first proj of a stage — the tail
        // zero-fill is idempotent (covers the same range to the same
        // value) but emitting it once keeps the legacy dispatch count.
        // Sprint 46C — skip coopmat_q4k for F32 weights (the kernel
        // expects packed Q4_K blocks, not raw float).
        if fwd.coopmat_q4k_enabled && !is_fp8_layer_weight(ctx.model, ctx.layer, suffix) && !f32_weight {
            let n_padded = Forward::pad_to_tile(n, 16);
            if quantize_input {
                fwd.zero_activation_tail(
                    ctx.dev, ctx.cmd, input_fp32,
                    n, n_padded, k,
                );
                transfer_to_compute_barrier(ctx.dev, ctx.cmd);
            }
            let (shader, bm, bn) = if n <= 64 {
                (fwd.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if n <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            fwd.run_gemm_coopmat_q4k(
                ctx.dev, ctx.registry, ctx.cmd, shader, weight,
                input_fp32, output,
                m, n_padded, k, bm, bn, label,
            );
            return;
        }

        // FP8 routing — always reads FP32 input directly, no quantize.
        if is_fp8_layer_weight(ctx.model, ctx.layer, suffix) {
            let scale = layer_weight_scale_buf(ctx.model, ctx.layer, suffix)
                .expect("FP8 GEMM requires a scale buffer");
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, suffix);
            if let Some((bn, bk)) = blk {
                fwd.run_gemm_fp8_blockwise(
                    ctx.dev, ctx.cmd, weight, scale,
                    input_fp32, output,
                    m, n, k, bn, bk, label,
                );
            } else {
                fwd.run_gemm_fp8_naive(
                    ctx.dev, ctx.registry, ctx.cmd, weight, scale,
                    input_fp32, output,
                    m, n, k, label,
                );
            }
            return;
        }

        // Standard GEMM path. Quantize once per stage; follow-up
        // projections reuse the populated `batch_q8`.
        let gemm_input = if use_mul_mm {
            input_fp32
        } else {
            if quantize_input {
                fwd.run_quantize_q8_1(
                    ctx.dev, ctx.registry, ctx.cmd,
                    input_fp32, fwd.batch_q8.handle,
                    n * k, "quantize_proj",
                );
                compute_barrier(ctx.dev, ctx.cmd);
            }
            fwd.batch_q8.handle
        };
        let shader = layer_weight_shader_gemm(
            ctx.model, ctx.layer, suffix, gemm_kind, m, n,
            fwd.mul_mm_coopmat_enabled, fwd.mul_mm_coopmat_f16acc_enabled,
        );
        fwd.run_gemm(
            ctx.dev, ctx.registry, ctx.cmd, shader, weight,
            gemm_input, output,
            m, n, k, label,
        );
    }
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
