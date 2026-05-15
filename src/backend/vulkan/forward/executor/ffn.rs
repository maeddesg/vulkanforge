//! Sprint 57B — Dense FFN steps (DEC + BAT pendants).
//!
//! 7 step_* + 7 b_step_* covering the dense gate / up / activation /
//! down / pre-and-post-FFN-norm / residual-add chain. The cleanest
//! 1:1 category in the executor: every DEC step has a near-identical
//! BAT counterpart, differing only in `seq_len` handling and
//! per-token vs batched buffer layout.
//!
//! See `executor/mod.rs` for the `ExecCtx`, `DecodeExec`, `BatchExec`
//! types and helper functions.

use super::{
    batch_seq_len, batch_seq_pos, decode_io, layer_dims_local, BatchExec, DecodeExec, ExecCtx,
    ExecMode,
};
use super::super::arch::{
    compute_barrier, layer_weight, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_scale_scalar, layer_weight_shader,
};
use super::super::layer_plan::ActivationKind;
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;
use super::super::super::shaders::ShaderId;

impl DecodeExec {
    pub(super) fn step_pre_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_gate_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_up_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_activation(
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

    pub(super) fn step_down_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_post_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_ffn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
}

impl BatchExec {
    pub(super) fn b_step_pre_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_gate_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_up_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_activation(
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

    pub(super) fn b_step_down_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_post_ffn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_ffn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
}
