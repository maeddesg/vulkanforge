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
use ash::vk::Handle;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::super::gguf::{GgmlType, ModelConfig};
use super::super::loader::LoadedModel;
use super::super::pipeline::SwigluPushConstants;
use super::super::pipeline_registry::PipelineRegistry;
use super::super::shaders::ShaderId;

use super::arch::{
    apply_final_logit_softcap, compute_barrier, layer_weight, layer_weight_opt,
    layer_weight_scale_block, layer_weight_scale_buf, layer_weight_scale_scalar,
    layer_weight_shader,
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
        let kv_dim = cfg.n_kv_heads * head_dim;
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
        let kv_dim = cfg.n_kv_heads * head_dim;
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
        let kv_dim = cfg.n_kv_heads * head_dim;
        let k_buf = fwd.cur().k_buf.handle;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_k.bias")
            .expect("KBiasAdd emitted but attn_k.bias is missing");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k");
        fwd.mark_written(&[k_buf]);
    }

    fn step_v_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = cfg.n_kv_heads * head_dim;
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
            cfg.n_kv_heads, 1,
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
            cfg.n_kv_heads, position, 0, "rope_k",
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
            head_dim, cfg.n_kv_heads,
            cfg.rms_norm_eps, "rms_norm_v",
        );
        fwd.mark_written(&[v_buf]);
    }

    fn step_kv_write(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let position = decode_position(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = cfg.n_kv_heads * head_dim;
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

/// Mirror of `arch::common::layer_dims` — duplicated locally so this
/// module doesn't need a `pub(super)` re-export. Returns
/// `(head_dim, ffn_dim, rope_theta, rotary_dim)` per layer.
fn layer_dims_local(cfg: &ModelConfig, layer: u32) -> (u32, u32, f32, u32) {
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

// Suppress unused-import warnings for items only used by the dispatch
// path (added to keep the file self-contained for 44C-2 review).
#[allow(unused_imports)]
use {
    apply_final_logit_softcap as _,
    compute_barrier as _,
    CommandContext as _,
    GgmlType as _,
    Handle as _,
};
