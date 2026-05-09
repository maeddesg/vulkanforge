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

    /// Fused `multi_add_rms`: `res1 = input + o_buf; hidden_norm =
    /// rms_norm(res1) * ffn_norm.weight`. Llama path only.
    fn fused_attn_residual_norm(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
    ) {
        let (input, _) = decode_io(ctx);
        let o_buf = fwd.cur().o_buf.handle;
        let res1 = fwd.cur().res1.handle;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let w = layer_weight(ctx.model, ctx.layer, "ffn_norm.weight");
        // Read deps for this fused dispatch: input + o_buf.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input, o_buf]);
        fwd.run_multi_add_rms(
            ctx.dev, ctx.registry, ctx.cmd,
            input, o_buf, w,
            res1, hidden_norm,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "add_rms_ffn",
        );
        fwd.mark_written(&[res1, hidden_norm]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
    }

    /// Per-step dispatcher. Match must be exhaustive over `LayerStep`.
    /// Adding a variant breaks compilation here until handled.
    fn execute_step(
        &self,
        fwd: &mut Forward,
        step: &LayerStep,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
    ) {
        match step {
            LayerStep::AttnNorm => self.step_attn_norm(fwd, cfg, ctx),
            LayerStep::QProj => self.step_q_proj(fwd, cfg, ctx),
            LayerStep::KProj => self.step_k_proj(fwd, cfg, ctx),
            LayerStep::VProj => self.step_v_proj(fwd, cfg, ctx),
            LayerStep::QBiasAdd => self.step_q_bias(fwd, cfg, ctx),
            LayerStep::KBiasAdd => self.step_k_bias(fwd, cfg, ctx),
            LayerStep::VBiasAdd => self.step_v_bias(fwd, cfg, ctx),
            LayerStep::QNormRope { rotary_dim, freq_base, theta_scale } => {
                self.step_q_norm_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::KNormRope { rotary_dim, freq_base, theta_scale } => {
                self.step_k_norm_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::QRope { rotary_dim, freq_base, theta_scale } => {
                self.step_q_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::KRope { rotary_dim, freq_base, theta_scale } => {
                self.step_k_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::VNorm => self.step_v_norm(fwd, cfg, ctx),
            LayerStep::KvWrite => self.step_kv_write(fwd, cfg, ctx),
            LayerStep::Attention { kv_layer: _, kv_start: _ } => {
                self.step_attention(fwd, cfg, ctx)
            }
            LayerStep::OProj => self.step_o_proj(fwd, cfg, ctx),
            LayerStep::PostAttnNorm => self.step_post_attn_norm(fwd, cfg, ctx),
            LayerStep::AttnResidualAdd => self.step_attn_residual_add(fwd, cfg, ctx),
            LayerStep::PreFfnNorm => self.step_pre_ffn_norm(fwd, cfg, ctx),
            LayerStep::GateProj => self.step_gate_proj(fwd, cfg, ctx),
            LayerStep::UpProj => self.step_up_proj(fwd, cfg, ctx),
            LayerStep::Activation { kind } => self.step_activation(fwd, cfg, ctx, *kind),
            LayerStep::DownProj => self.step_down_proj(fwd, cfg, ctx),
            LayerStep::PostFfnNorm => self.step_post_ffn_norm(fwd, cfg, ctx),
            LayerStep::FfnResidualAdd => self.step_ffn_residual_add(fwd, cfg, ctx),
            LayerStep::PleBlock => self.step_ple_block(fwd, cfg, ctx),
            LayerStep::LayerScalarMul => self.step_layer_scalar_mul(fwd, cfg, ctx),
        }
    }

    // ── per-step implementations ──────────────────────────────────────

    fn step_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (input, _) = decode_io(ctx);
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let w = layer_weight(ctx.model, ctx.layer, "attn_norm.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, input, w, hidden_norm,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn",
        );
        fwd.mark_written(&[hidden_norm]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
    }

    fn step_q_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let q_buf = fwd.cur().q_buf.handle;
        let wq = layer_weight(ctx.model, ctx.layer, "attn_q.weight");
        let sq = layer_weight_shader(
            ctx.model, ctx.layer, "attn_q.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_q = layer_weight_scale_scalar(ctx.model, ctx.layer, "attn_q.weight");
        let sb_q = layer_weight_scale_buf(ctx.model, ctx.layer, "attn_q.weight");
        if let Some(s) = sb_q {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "attn_q.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wq, s, hidden_norm, q_buf,
                cfg.hidden_dim, q_dim, blk, "gemv_q",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sq, wq, hidden_norm, q_buf,
                cfg.hidden_dim, q_dim, scale_q, "gemv_q",
            );
        }
        fwd.mark_written(&[q_buf]);
    }

    fn step_k_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let k_buf = fwd.cur().k_buf.handle;
        let wk = layer_weight(ctx.model, ctx.layer, "attn_k.weight");
        let sk = layer_weight_shader(
            ctx.model, ctx.layer, "attn_k.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_k = layer_weight_scale_scalar(ctx.model, ctx.layer, "attn_k.weight");
        let sb_k = layer_weight_scale_buf(ctx.model, ctx.layer, "attn_k.weight");
        if let Some(s) = sb_k {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "attn_k.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wk, s, hidden_norm, k_buf,
                cfg.hidden_dim, kv_dim, blk, "gemv_k",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sk, wk, hidden_norm, k_buf,
                cfg.hidden_dim, kv_dim, scale_k, "gemv_k",
            );
        }
        fwd.mark_written(&[k_buf]);
    }

    fn step_v_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let v_buf = fwd.cur().v_buf.handle;
        let wv = layer_weight(ctx.model, ctx.layer, "attn_v.weight");
        let sv = layer_weight_shader(
            ctx.model, ctx.layer, "attn_v.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_v = layer_weight_scale_scalar(ctx.model, ctx.layer, "attn_v.weight");
        let sb_v = layer_weight_scale_buf(ctx.model, ctx.layer, "attn_v.weight");
        if let Some(s) = sb_v {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "attn_v.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wv, s, hidden_norm, v_buf,
                cfg.hidden_dim, kv_dim, blk, "gemv_v",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sv, wv, hidden_norm, v_buf,
                cfg.hidden_dim, kv_dim, scale_v, "gemv_v",
            );
        }
        fwd.mark_written(&[v_buf]);
    }

    fn step_q_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let q_buf = fwd.cur().q_buf.handle;
        // Source-of-truth check (mirrors the `if let Some(b)` path in
        // dispatch_layer): if the bias tensor is missing, the builder
        // shouldn't have emitted this step in the first place; treat
        // missing as a programming error.
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_q.bias")
            .expect("QBiasAdd emitted but attn_q.bias is missing");
        // Pre-bias: the GEMV that just wrote q_buf needs to be visible.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, q_buf, b, q_buf, q_dim, 1, "bias_q");
        fwd.mark_written(&[q_buf]);
    }

    fn step_k_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let k_buf = fwd.cur().k_buf.handle;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_k.bias")
            .expect("KBiasAdd emitted but attn_k.bias is missing");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k");
        fwd.mark_written(&[k_buf]);
    }

    fn step_v_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let v_buf = fwd.cur().v_buf.handle;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_v.bias")
            .expect("VBiasAdd emitted but attn_v.bias is missing");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[v_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, v_buf, b, v_buf, kv_dim, 1, "bias_v");
        fwd.mark_written(&[v_buf]);
    }

    fn step_q_norm_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_buf = fwd.cur().q_buf.handle;
        let wqn = layer_weight(ctx.model, ctx.layer, "attn_q_norm.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q_buf]);
        fwd.run_rms_norm_mul_rope(
            ctx.dev, ctx.registry, ctx.cmd,
            q_buf, wqn, q_buf,
            head_dim, rotary_dim, freq_base, theta_scale,
            cfg.n_heads, 1,
            cfg.rms_norm_eps, "rms_norm_mul_rope_q",
        );
        fwd.mark_written(&[q_buf]);
    }

    fn step_k_norm_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let k_buf = fwd.cur().k_buf.handle;
        let wkn = layer_weight(ctx.model, ctx.layer, "attn_k_norm.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        fwd.run_rms_norm_mul_rope(
            ctx.dev, ctx.registry, ctx.cmd,
            k_buf, wkn, k_buf,
            head_dim, rotary_dim, freq_base, theta_scale,
            n_kv_heads_for(cfg, ctx.layer), 1,
            cfg.rms_norm_eps, "rms_norm_mul_rope_k",
        );
        fwd.mark_written(&[k_buf]);
    }

    fn step_q_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let position = decode_position(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_buf = fwd.cur().q_buf.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q_buf]);
        fwd.run_rope_neox_with_pos_offset(
            ctx.dev, ctx.registry, ctx.cmd, q_buf, q_buf,
            head_dim, rotary_dim, freq_base, theta_scale,
            cfg.n_heads, position, 0, "rope_q",
        );
        fwd.mark_written(&[q_buf]);
    }

    fn step_k_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let position = decode_position(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let k_buf = fwd.cur().k_buf.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        fwd.run_rope_neox_with_pos_offset(
            ctx.dev, ctx.registry, ctx.cmd, k_buf, k_buf,
            head_dim, rotary_dim, freq_base, theta_scale,
            n_kv_heads_for(cfg, ctx.layer), position, 0, "rope_k",
        );
        fwd.mark_written(&[k_buf]);
    }

    fn step_v_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let v_buf = fwd.cur().v_buf.handle;
        let vnorm_ones = fwd.vnorm_ones.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[v_buf]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            v_buf, vnorm_ones, v_buf,
            head_dim, n_kv_heads_for(cfg, ctx.layer),
            cfg.rms_norm_eps, "rms_norm_v",
        );
        fwd.mark_written(&[v_buf]);
    }

    fn step_kv_write(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let position = decode_position(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let k_src = fwd.cur().k_buf.handle;
        let v_src = fwd.cur().v_buf.handle;
        let k_dst = fwd.kv_cache.k_buffer.handle;
        let v_dst = fwd.kv_cache.v_buffer.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_src, v_src]);
        let row_bytes = fwd.kv_cache.row_bytes(ctx.layer);
        let dst_off = fwd.kv_cache.pos_offset_bytes(ctx.layer, position);
        if fwd.kv_cache.is_fp8() {
            fwd.run_kv_store_fp8(
                ctx.dev, ctx.registry, ctx.cmd, k_src, k_dst, kv_dim, dst_off,
                "kv_store_fp8_k",
            );
            fwd.run_kv_store_fp8(
                ctx.dev, ctx.registry, ctx.cmd, v_src, v_dst, kv_dim, dst_off,
                "kv_store_fp8_v",
            );
        } else if fwd.kv_cache.is_fp16() {
            fwd.run_kv_copy_fp16(
                ctx.dev, ctx.registry, ctx.cmd, k_src, k_dst, kv_dim, dst_off,
                "kv_copy_fp16_k",
            );
            fwd.run_kv_copy_fp16(
                ctx.dev, ctx.registry, ctx.cmd, v_src, v_dst, kv_dim, dst_off,
                "kv_copy_fp16_v",
            );
        } else {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(row_bytes);
            fwd.profile("kv_write", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
            });
        }
        // Inline kv_bar — TRANSFER+COMPUTE → COMPUTE. Always emitted.
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&kv_bar), &[], &[],
            );
        }
        fwd.pending_writes.clear();
    }

    fn step_attention(&self, fwd: &mut Forward, _cfg: &ModelConfig, ctx: &ExecCtx) {
        let position = decode_position(ctx);
        let attn_out = fwd.cur().attn_out.handle;
        let q_buf = fwd.cur().q_buf.handle;
        // Subscribers don't run KvWrite (which would emit the kv_bar);
        // they need a plain compute→compute barrier on q_buf so post-
        // RoPE Q is committed before scalar_attn reads it.
        // Decoder: layer_owns_kv determines whether the upstream
        // emitted a KV-write or just q_buf. Either way maybe_compute
        // observes the dirty set correctly because subscribers never
        // issue mark_written on k/v.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q_buf]);
        fwd.run_scalar_attn(ctx.dev, ctx.registry, ctx.cmd, ctx.layer, position);
        fwd.mark_written(&[attn_out]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[attn_out]);
    }

    fn step_o_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let attn_out = fwd.cur().attn_out.handle;
        let o_buf = fwd.cur().o_buf.handle;
        let wo = layer_weight(ctx.model, ctx.layer, "attn_output.weight");
        let so = layer_weight_shader(
            ctx.model, ctx.layer, "attn_output.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_o = layer_weight_scale_scalar(ctx.model, ctx.layer, "attn_output.weight");
        let sb_o = layer_weight_scale_buf(ctx.model, ctx.layer, "attn_output.weight");
        if let Some(s) = sb_o {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "attn_output.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wo, s, attn_out, o_buf,
                q_dim, cfg.hidden_dim, blk, "gemv_o",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, so, wo, attn_out, o_buf,
                q_dim, cfg.hidden_dim, scale_o, "gemv_o",
            );
        }
        fwd.mark_written(&[o_buf]);
    }

    fn step_post_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 only: o_normed = rms_norm(o) * post_attention_layernorm.weight.
        // Reuses gate_buf as scratch (gate_buf isn't yet live this layer).
        let (input, _) = decode_io(ctx);
        let o_buf = fwd.cur().o_buf.handle;
        let gate_buf = fwd.cur().gate_buf.handle;
        let post_attn_w = layer_weight(ctx.model, ctx.layer, "ffn_norm.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input, o_buf]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, o_buf, post_attn_w, gate_buf,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_attn",
        );
        fwd.mark_written(&[gate_buf]);
    }

    fn step_attn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 path (PostAttnNorm just ran): res1 = input + gate_buf (the post-attn-normed o).
        // Llama path (handled by fused_attn_residual_norm) — should not reach here on Llama.
        let (input, _) = decode_io(ctx);
        let res1 = fwd.cur().res1.handle;
        // Pick the addend: gate_buf for Gemma-4 (post-attn-normed o);
        // o_buf for non-Gemma-4 paths that didn't fuse (defensive — Llama
        // takes the fused branch in the loop driver, so this only fires
        // when fusion is somehow disabled).
        let addend = if cfg.gemma4.is_some() {
            fwd.cur().gate_buf.handle
        } else {
            fwd.cur().o_buf.handle
        };
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input, addend]);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            input, addend, res1,
            cfg.hidden_dim, "add_res1",
        );
        fwd.mark_written(&[res1]);
    }

    fn step_pre_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 path: hidden_norm = rms_norm(res1) * pre_feedforward_layernorm.weight.
        // Llama path is fused via multi_add_rms with AttnResidualAdd in the loop driver.
        let res1 = fwd.cur().res1.handle;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let pre_ffn_w = layer_weight(ctx.model, ctx.layer, "ffn_pre_norm.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[res1]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, res1, pre_ffn_w, hidden_norm,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_pre_ffn",
        );
        fwd.mark_written(&[hidden_norm]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
    }

    fn step_gate_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let gate_buf = fwd.cur().gate_buf.handle;
        let wg = layer_weight(ctx.model, ctx.layer, "ffn_gate.weight");
        let sg = layer_weight_shader(
            ctx.model, ctx.layer, "ffn_gate.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_g = layer_weight_scale_scalar(ctx.model, ctx.layer, "ffn_gate.weight");
        let sb_g = layer_weight_scale_buf(ctx.model, ctx.layer, "ffn_gate.weight");
        if let Some(s) = sb_g {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "ffn_gate.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wg, s, hidden_norm, gate_buf,
                cfg.hidden_dim, ffn_dim, blk, "gemv_gate",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sg, wg, hidden_norm, gate_buf,
                cfg.hidden_dim, ffn_dim, scale_g, "gemv_gate",
            );
        }
        fwd.mark_written(&[gate_buf]);
    }

    fn step_up_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let up_buf = fwd.cur().up_buf.handle;
        let wu = layer_weight(ctx.model, ctx.layer, "ffn_up.weight");
        let su = layer_weight_shader(
            ctx.model, ctx.layer, "ffn_up.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_u = layer_weight_scale_scalar(ctx.model, ctx.layer, "ffn_up.weight");
        let sb_u = layer_weight_scale_buf(ctx.model, ctx.layer, "ffn_up.weight");
        if let Some(s) = sb_u {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "ffn_up.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wu, s, hidden_norm, up_buf,
                cfg.hidden_dim, ffn_dim, blk, "gemv_up",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, su, wu, hidden_norm, up_buf,
                cfg.hidden_dim, ffn_dim, scale_u, "gemv_up",
            );
        }
        fwd.mark_written(&[up_buf]);
    }

    fn step_activation(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        kind: ActivationKind,
    ) {
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let gate_buf = fwd.cur().gate_buf.handle;
        let up_buf = fwd.cur().up_buf.handle;
        let ffn_hidden = fwd.cur().ffn_hidden.handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf, up_buf]);
        match kind {
            ActivationKind::SwiGlu => fwd.run_swiglu(
                ctx.dev, ctx.registry, ctx.cmd,
                gate_buf, up_buf, ffn_hidden,
                ffn_dim, "swiglu",
            ),
            ActivationKind::GeluPytorchTanhGlu => fwd.run_gelu_pytorch_tanh_glu(
                ctx.dev, ctx.registry, ctx.cmd,
                gate_buf, up_buf, ffn_hidden,
                ffn_dim, "gelu_pt_glu",
            ),
        }
        fwd.mark_written(&[ffn_hidden]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_hidden]);
    }

    fn step_down_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let ffn_hidden = fwd.cur().ffn_hidden.handle;
        let ffn_out = fwd.cur().ffn_out.handle;
        let wd = layer_weight(ctx.model, ctx.layer, "ffn_down.weight");
        let sd = layer_weight_shader(
            ctx.model, ctx.layer, "ffn_down.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_d = layer_weight_scale_scalar(ctx.model, ctx.layer, "ffn_down.weight");
        let sb_d = layer_weight_scale_buf(ctx.model, ctx.layer, "ffn_down.weight");
        if let Some(s) = sb_d {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "ffn_down.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wd, s, ffn_hidden, ffn_out,
                ffn_dim, cfg.hidden_dim, blk, "gemv_down",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sd, wd, ffn_hidden, ffn_out,
                ffn_dim, cfg.hidden_dim, scale_d, "gemv_down",
            );
        }
        fwd.mark_written(&[ffn_out]);
    }

    fn step_post_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 only: ffn_out_normed = rms_norm(ffn_out) * post_ffn_norm.weight.
        // Reuses gate_buf as scratch (consumed by SwiGLU/GELU at this point).
        let ffn_out = fwd.cur().ffn_out.handle;
        let gate_buf = fwd.cur().gate_buf.handle;
        let post_ffn_w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[ffn_out]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd, ffn_out, post_ffn_w, gate_buf,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_ffn",
        );
        fwd.mark_written(&[gate_buf]);
    }

    fn step_ffn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Decode path: output = res1 + (ffn_out | post_ffn_normed_ffn_out).
        let (_, output) = decode_io(ctx);
        let res1 = fwd.cur().res1.handle;
        let addend = if cfg.gemma4.is_some() {
            fwd.cur().gate_buf.handle  // post-FFN-norm scratch
        } else {
            fwd.cur().ffn_out.handle
        };
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[res1, addend]);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            res1, addend, output,
            cfg.hidden_dim, "add_res2",
        );
        fwd.mark_written(&[output]);
    }

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

    fn execute_step(
        &self,
        fwd: &mut Forward,
        step: &LayerStep,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
    ) {
        match step {
            LayerStep::AttnNorm => self.b_step_attn_norm(fwd, cfg, ctx),
            LayerStep::QProj => self.b_step_q_proj(fwd, cfg, ctx),
            LayerStep::KProj => self.b_step_k_proj(fwd, cfg, ctx),
            LayerStep::VProj => self.b_step_v_proj(fwd, cfg, ctx),
            LayerStep::QBiasAdd => self.b_step_q_bias(fwd, cfg, ctx),
            LayerStep::KBiasAdd => self.b_step_k_bias(fwd, cfg, ctx),
            LayerStep::VBiasAdd => self.b_step_v_bias(fwd, cfg, ctx),
            LayerStep::QNormRope { rotary_dim, freq_base, theta_scale } => {
                self.b_step_q_norm_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::KNormRope { rotary_dim, freq_base, theta_scale } => {
                self.b_step_k_norm_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::QRope { rotary_dim, freq_base, theta_scale } => {
                self.b_step_q_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::KRope { rotary_dim, freq_base, theta_scale } => {
                self.b_step_k_rope(fwd, cfg, ctx, *rotary_dim, *freq_base, *theta_scale)
            }
            LayerStep::VNorm => self.b_step_v_norm(fwd, cfg, ctx),
            LayerStep::KvWrite => self.b_step_kv_write(fwd, cfg, ctx),
            LayerStep::Attention { kv_layer, kv_start } => {
                self.b_step_attention(fwd, cfg, ctx, *kv_layer, *kv_start)
            }
            LayerStep::OProj => self.b_step_o_proj(fwd, cfg, ctx),
            LayerStep::PostAttnNorm => self.b_step_post_attn_norm(fwd, cfg, ctx),
            LayerStep::AttnResidualAdd => self.b_step_attn_residual_add(fwd, cfg, ctx),
            LayerStep::PreFfnNorm => self.b_step_pre_ffn_norm(fwd, cfg, ctx),
            LayerStep::GateProj => self.b_step_gate_proj(fwd, cfg, ctx),
            LayerStep::UpProj => self.b_step_up_proj(fwd, cfg, ctx),
            LayerStep::Activation { kind } => self.b_step_activation(fwd, cfg, ctx, *kind),
            LayerStep::DownProj => self.b_step_down_proj(fwd, cfg, ctx),
            LayerStep::PostFfnNorm => self.b_step_post_ffn_norm(fwd, cfg, ctx),
            LayerStep::FfnResidualAdd => self.b_step_ffn_residual_add(fwd, cfg, ctx),
            LayerStep::PleBlock => self.b_step_ple_block(fwd, cfg, ctx),
            LayerStep::LayerScalarMul => self.b_step_layer_scalar_mul(fwd, cfg, ctx),
        }
    }

    // ── per-step impls ────────────────────────────────────────────────

    /// Llama-path Sprint-9b fusion: `multi_add_rms(batch_residual,
    /// batch_o, ffn_norm.weight → batch_residual, batch_norm)`.
    /// Replaces the AttnResidualAdd + PreFfnNorm pair when the
    /// loop driver detects the Llama pattern (no PostAttnNorm
    /// preceding).
    fn b_fused_attn_residual_norm(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
    ) {
        let seq_len = batch_seq_len(ctx);
        let w_ffn = layer_weight(ctx.model, ctx.layer, "ffn_norm.weight");
        fwd.run_multi_add_rms(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_residual.handle, fwd.batch_o.handle, w_ffn,
            fwd.batch_residual.handle, fwd.batch_norm.handle,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "add_rms_ffn_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_attn_norm(&self, _fwd: &mut Forward, _cfg: &ModelConfig, _ctx: &ExecCtx) {
        // No-op. `batch_norm` is pre-seeded:
        // - layer 0: by `prefill_batch::record_prefill_seed`,
        // - layers 1+: by the previous layer's FfnResidualAdd
        //   (Llama: multi_add_rms with this layer's attn_norm.weight;
        //    Gemma-4: separate rms_norm in `execute_layer` tail).
    }

    /// Sprint 47D — emit `compute_barrier` only on Gemma-4 subscriber
    /// layers (`layer >= first_kv_shared`), where Q's writes need to
    /// be flushed before the next stage reads `batch_q`. Owner layers
    /// already get a trailing barrier on the LAST step of each
    /// Q+K[+V] stage (V-proj / V-bias / K-norm-rope / K-rope) and the
    /// unconditional 46H emit cost ~7 % prefill on 36-layer Q4_K_M
    /// (Qwen3) — see Sprint 47C bisect. Models without `cfg.gemma4`
    /// (Qwen3, Llama, …) skip the barrier entirely.
    fn b_subscriber_q_barrier(&self, cfg: &ModelConfig, ctx: &ExecCtx) {
        if let Some(g) = cfg.gemma4.as_ref() {
            if ctx.layer >= g.first_kv_shared {
                compute_barrier(ctx.dev, ctx.cmd);
            }
        }
    }

    fn b_step_q_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        // First proj of attn stage — quantize batch_norm into batch_q8.
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_q.weight",
            fwd.batch_norm.handle,
            fwd.batch_q.handle,
            q_dim, cfg.hidden_dim, seq_len, "gemm_q",
            /* quantize_input = */ true,
        );
        // Sprint 46H + 47D — Gemma-4 subscribers skip KProj/VProj/biases
        // entirely; without flushing Q here, QNormRope races with
        // Q-proj's writes (garbage Q for q_idx>0). Owner layers already
        // get a trailing barrier in V-proj covering Q+K+V uniformly.
        self.b_subscriber_q_barrier(cfg, ctx);
    }

    fn b_step_k_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_k.weight",
            fwd.batch_norm.handle,
            fwd.batch_k.handle,
            kv_dim, cfg.hidden_dim, seq_len, "gemm_k",
            /* quantize_input = */ false,
        );
    }

    fn b_step_v_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_v.weight",
            fwd.batch_norm.handle,
            fwd.batch_v.handle,
            kv_dim, cfg.hidden_dim, seq_len, "gemm_v",
            /* quantize_input = */ false,
        );
        // After Q+K+V finished, emit a single barrier (matches the
        // legacy `compute_barrier` after the GEMM block).
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_q_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_q.bias")
            .expect("QBiasAdd emitted but attn_q.bias missing");
        fwd.run_bias_add(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_q.handle, b, fwd.batch_q.handle,
            q_dim, seq_len, "bias_q_b",
        );
    }

    fn b_step_k_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_k.bias")
            .expect("KBiasAdd emitted but attn_k.bias missing");
        fwd.run_bias_add(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_k.handle, b, fwd.batch_k.handle,
            kv_dim, seq_len, "bias_k_b",
        );
    }

    fn b_step_v_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_v.bias")
            .expect("VBiasAdd emitted but attn_v.bias missing");
        fwd.run_bias_add(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_v.handle, b, fwd.batch_v.handle,
            kv_dim, seq_len, "bias_v_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_q_norm_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let wqn = layer_weight(ctx.model, ctx.layer, "attn_q_norm.weight");
        fwd.run_rms_norm_mul_rope(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_q.handle, wqn, fwd.batch_q.handle,
            head_dim, rotary_dim, freq_base, theta_scale,
            cfg.n_heads, seq_len,
            cfg.rms_norm_eps, "rms_norm_mul_rope_q_b",
        );
        // Sprint 46H + 47D — subscribers skip the K side; without this
        // barrier Attention races with Q. Owner: K-norm-rope's trailing
        // barrier covers both.
        self.b_subscriber_q_barrier(cfg, ctx);
    }

    fn b_step_k_norm_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let wkn = layer_weight(ctx.model, ctx.layer, "attn_k_norm.weight");
        fwd.run_rms_norm_mul_rope(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_k.handle, wkn, fwd.batch_k.handle,
            head_dim, rotary_dim, freq_base, theta_scale,
            n_kv_heads_for(cfg, ctx.layer), seq_len,
            cfg.rms_norm_eps, "rms_norm_mul_rope_k_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_q_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        fwd.run_rope_batch(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_q.handle, fwd.batch_q.handle,
            head_dim, rotary_dim, freq_base, theta_scale,
            cfg.n_heads, seq_len, "rope_q_batch",
        );
        // Sprint 46H + 47D — see b_step_q_norm_rope.
        self.b_subscriber_q_barrier(cfg, ctx);
    }

    fn b_step_k_rope(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        fwd.run_rope_batch(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_k.handle, fwd.batch_k.handle,
            head_dim, rotary_dim, freq_base, theta_scale,
            n_kv_heads_for(cfg, ctx.layer), seq_len, "rope_k_batch",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_v_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4-only. The legacy `dispatch_layer_batch` doesn't emit
        // V-norm because Gemma-4 currently force-routes through
        // per-token prefill. This impl is included for future
        // batch-prefill activation; not exercised by Qwen3 paths.
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_v.handle, fwd.vnorm_ones.handle, fwd.batch_v.handle,
            head_dim, n_kv_heads_for(cfg, ctx.layer) * seq_len,
            cfg.rms_norm_eps, "rms_norm_v_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_kv_write(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (seq_len, base_pos) = batch_seq_pos(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let dst_off = fwd.kv_cache.pos_offset_bytes(ctx.layer, base_pos);
        let kv_elements = seq_len * kv_dim;
        if fwd.kv_cache.is_fp8() {
            fwd.run_kv_store_fp8(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_k.handle, fwd.kv_cache.k_buffer.handle,
                kv_elements, dst_off, "kv_store_fp8_k_b",
            );
            fwd.run_kv_store_fp8(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_v.handle, fwd.kv_cache.v_buffer.handle,
                kv_elements, dst_off, "kv_store_fp8_v_b",
            );
        } else if fwd.kv_cache.is_fp16() {
            fwd.run_kv_copy_fp16(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_k.handle, fwd.kv_cache.k_buffer.handle,
                kv_elements, dst_off, "kv_copy_fp16_k_b",
            );
            fwd.run_kv_copy_fp16(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_v.handle, fwd.kv_cache.v_buffer.handle,
                kv_elements, dst_off, "kv_copy_fp16_v_b",
            );
        } else {
            let kv_row_bytes = fwd.kv_cache.row_bytes(ctx.layer);
            let bulk_size = (seq_len as u64) * kv_row_bytes;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(bulk_size);
            unsafe {
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, fwd.batch_k.handle, fwd.kv_cache.k_buffer.handle,
                    std::slice::from_ref(&copy),
                );
                ctx.dev.device.cmd_copy_buffer(
                    ctx.cmd, fwd.batch_v.handle, fwd.kv_cache.v_buffer.handle,
                    std::slice::from_ref(&copy),
                );
            }
        }
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&kv_bar), &[], &[],
            );
        }
    }

    fn b_step_attention(
        &self,
        fwd: &mut Forward,
        _cfg: &ModelConfig,
        ctx: &ExecCtx,
        kv_layer: u32,
        kv_start_window: u32,
    ) {
        let (seq_len, base_pos) = batch_seq_pos(ctx);
        // Sprint 46E — `kv_start_window` carries the layer-static window
        // size (0 for full-attention / non-Gemma stacks, `sliding_window`
        // for Gemma-4 sliding layers). Convert to the global lower-bound
        // shape the shader consumes: `max(0, last_pos+1 - window_size)`.
        // For Qwen3 / Llama / Mistral / DSR1 (window=0) this resolves to
        // 0 → math is bit-identical to pre-46E.
        //
        // The "last_pos+1" target intentionally takes the LAST query's
        // window. Earlier queries in the batch attend to slightly fewer
        // positions than HF reference would — a known approximation
        // documented in `FlashAttnBatchPushConstants::kv_start`. Sprint
        // 46F's `force_per_token_prefill` lift will bisect against the
        // decode path and surface this if it matters in practice;
        // refining to per-query masking is a Sprint 46F follow-up.
        let kv_start = if kv_start_window == 0 {
            0
        } else {
            (base_pos + seq_len).saturating_sub(kv_start_window)
        };
        // Sprint 46F — flash_attn_tiled.comp / flash_attn_coopmat.comp
        // hardcode HEAD_DIM=128 (Qwen3 / Llama-3.1 / Mistral / DSR1). For
        // Gemma-4 (head_dim=256) those shaders would silently leave dims
        // 128..head_dim-1 zero in the attention output. Route Gemma-4
        // through `run_flash_attn_batch`, which uses per-thread striped
        // accumulators (Sprint 43D-2) and handles head_dim up to 512.
        let head_dim_layer = fwd.kv_cache.head_dim_for(ctx.layer);
        let force_fa_batch = head_dim_layer != 128;
        if fwd.fa_tiled_enabled && !force_fa_batch {
            fwd.run_flash_attn_tiled(
                ctx.dev, ctx.registry, ctx.cmd,
                ctx.layer,
                fwd.batch_q.handle,
                fwd.batch_attn_out.handle,
                seq_len, base_pos, base_pos + seq_len,
                kv_layer, kv_start,
            );
        } else {
            fwd.run_flash_attn_batch(
                ctx.dev, ctx.registry, ctx.cmd,
                ctx.layer,
                fwd.batch_q.handle,
                fwd.batch_attn_out.handle,
                seq_len, base_pos, base_pos + seq_len,
                kv_layer, kv_start,
            );
        }
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_o_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        // O-proj reads attn_out (seq_len × q_dim) and writes (seq_len × hidden).
        // Stage of one — always quantizes its own input.
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_output.weight",
            fwd.batch_attn_out.handle,
            fwd.batch_o.handle,
            cfg.hidden_dim, q_dim, seq_len, "gemm_o",
            /* quantize_input = */ true,
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_post_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let post_attn_w = layer_weight(ctx.model, ctx.layer, "ffn_norm.weight");
        // Reuse `batch_attn_out` as scratch (consumed by O-proj earlier).
        let scratch = fwd.batch_attn_out.handle;
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_o.handle, post_attn_w, scratch,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "rms_norm_post_attn_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_attn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 path: batch_residual += scratch (the post-attn-normed o,
        // written by PostAttnNorm). On Llama BatchExec doesn't actually
        // see this variant solo because the loop driver fuses with
        // PreFfnNorm.
        let seq_len = batch_seq_len(ctx);
        let scratch = if cfg.gemma4.is_some() {
            fwd.batch_attn_out.handle
        } else {
            fwd.batch_o.handle
        };
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            fwd.batch_residual.handle, scratch, fwd.batch_residual.handle,
            seq_len * cfg.hidden_dim, "add_res1_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_pre_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 path; Llama fuses via multi_add_rms in execute_layer's
        // lookahead — but BatchExec doesn't have that lookahead today,
        // so this also fires on Llama (matches the legacy 2-step path
        // when batch path runs unfused). Use ffn_norm.weight for Llama,
        // ffn_pre_norm.weight for Gemma-4.
        let seq_len = batch_seq_len(ctx);
        let suffix = if cfg.gemma4.is_some() {
            "ffn_pre_norm.weight"
        } else {
            "ffn_norm.weight"
        };
        let w = layer_weight(ctx.model, ctx.layer, suffix);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_residual.handle, w, fwd.batch_norm.handle,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "rms_norm_pre_ffn_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_gate_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        // First proj of FFN stage — quantize batch_norm.
        self.b_run_proj(
            fwd, cfg, ctx,
            "ffn_gate.weight",
            fwd.batch_norm.handle,
            fwd.batch_gate.handle,
            ffn_dim, cfg.hidden_dim, seq_len, "gemm_gate",
            /* quantize_input = */ true,
        );
    }

    fn b_step_up_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        self.b_run_proj(
            fwd, cfg, ctx,
            "ffn_up.weight",
            fwd.batch_norm.handle,
            fwd.batch_up.handle,
            ffn_dim, cfg.hidden_dim, seq_len, "gemm_up",
            /* quantize_input = */ false,
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_activation(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        kind: ActivationKind,
    ) {
        let seq_len = batch_seq_len(ctx);
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        let n = seq_len * ffn_dim;
        match kind {
            ActivationKind::SwiGlu => fwd.run_swiglu(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_gate.handle, fwd.batch_up.handle, fwd.batch_ffn_hidden.handle,
                n, "swiglu_b",
            ),
            ActivationKind::GeluPytorchTanhGlu => fwd.run_gelu_pytorch_tanh_glu(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_gate.handle, fwd.batch_up.handle, fwd.batch_ffn_hidden.handle,
                n, "gelu_pt_glu_b",
            ),
        }
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_down_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (_, ffn_dim, _, _) = layer_dims_local(cfg, ctx.layer);
        // Stage of one — always quantizes its own input.
        self.b_run_proj(
            fwd, cfg, ctx,
            "ffn_down.weight",
            fwd.batch_ffn_hidden.handle,
            fwd.batch_ffn_out.handle,
            cfg.hidden_dim, ffn_dim, seq_len, "gemm_down",
            /* quantize_input = */ true,
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_post_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        // Gemma-4 only.
        let seq_len = batch_seq_len(ctx);
        let post_ffn_w = layer_weight(ctx.model, ctx.layer, "ffn_post_norm.weight");
        let scratch = fwd.batch_attn_out.handle;
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            fwd.batch_ffn_out.handle, post_ffn_w, scratch,
            cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "rms_norm_post_ffn_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    fn b_step_ffn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (seq_len, _) = batch_seq_pos(ctx);
        let next_w = match ctx.mode {
            ExecMode::Batch { next_attn_norm_weight, .. } => next_attn_norm_weight,
            _ => unreachable!("BatchExec invoked with Decode mode"),
        };
        if cfg.gemma4.is_some() {
            // Gemma-4: plain add of (residual + post_ffn_normed_ffn_out).
            // The next-attn-norm seed is emitted by execute_layer's tail.
            let scratch = fwd.batch_attn_out.handle;
            fwd.run_binary(
                ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
                fwd.batch_residual.handle, scratch, fwd.batch_residual.handle,
                seq_len * cfg.hidden_dim, "add_res2_gemma4_b",
            );
            compute_barrier(ctx.dev, ctx.cmd);
        } else {
            // Llama: Sprint 9b.2 fused residual-add + next-layer attn-norm
            // when there's a next layer. Plain add otherwise.
            match next_w {
                Some(w_next) => {
                    fwd.run_multi_add_rms(
                        ctx.dev, ctx.registry, ctx.cmd,
                        fwd.batch_residual.handle, fwd.batch_ffn_out.handle, w_next,
                        fwd.batch_residual.handle,
                        fwd.batch_norm.handle,
                        cfg.hidden_dim, seq_len, cfg.rms_norm_eps, "add_rms_attn_next_b",
                    );
                }
                None => {
                    fwd.run_binary(
                        ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
                        fwd.batch_residual.handle, fwd.batch_ffn_out.handle,
                        fwd.batch_residual.handle,
                        seq_len * cfg.hidden_dim, "add_res2_b",
                    );
                }
            }
            compute_barrier(ctx.dev, ctx.cmd);
        }
    }

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
        let attn_q_type = ctx
            .model
            .tensor(&format!("blk.{}.attn_q.weight", ctx.layer))
            .map(|t| t.ggml_type);
        let force_mmq = matches!(attn_q_type, Some(GgmlType::Q4_0));
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

/// Helper: extract `seq_len` from a Batch ExecCtx.
fn batch_seq_len(ctx: &ExecCtx) -> u32 {
    match ctx.mode {
        ExecMode::Batch { seq_len, .. } => seq_len,
        _ => unreachable!("BatchExec invoked with Decode mode"),
    }
}

/// Helper: extract `(seq_len, base_pos)` from a Batch ExecCtx.
fn batch_seq_pos(ctx: &ExecCtx) -> (u32, u32) {
    match ctx.mode {
        ExecMode::Batch { seq_len, base_pos, .. } => (seq_len, base_pos),
        _ => unreachable!("BatchExec invoked with Decode mode"),
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn decode_io(ctx: &ExecCtx) -> (vk::Buffer, vk::Buffer) {
    match ctx.mode {
        ExecMode::Decode { input, output, .. } => (input, output),
        ExecMode::Batch { .. } => unreachable!("DecodeExec invoked with Batch mode"),
    }
}

fn decode_position(ctx: &ExecCtx) -> u32 {
    match ctx.mode {
        ExecMode::Decode { position, .. } => position,
        ExecMode::Batch { .. } => unreachable!("DecodeExec invoked with Batch mode"),
    }
}

/// Re-export of `arch::common::layer_dims` for the per-step helpers.
/// Returns `(head_dim, ffn_dim, rope_theta, rotary_dim)` per layer.
use super::arch::layer_dims as layer_dims_local;

/// Sprint 51B-pre — per-layer KV-head count (8 / 2 split for the
/// Gemma-4-26B-A4B sliding / full pattern). Falls back to the
/// uniform `ModelConfig::n_kv_heads` for non-Gemma-4 architectures.
use super::arch::n_kv_heads_for;
