//! Sprint 57B — per-step dispatcher + Llama-fusion helpers.
//!
//! Both `DecodeExec::execute_step` and `BatchExec::execute_step` live
//! here as the keystone of the executor: the exhaustive `match step
//! { ... }` over every `LayerStep` variant. Adding a variant fails to
//! compile in both arms until handled, preserving the invariant
//! recorded in `memory/feedback_layer_dispatch_paths.md` (forgetting
//! to update one of the dispatchers was historically a recurring
//! source of silent-broken-path bugs).
//!
//! Also home to the Llama-only `fused_attn_residual_norm` /
//! `b_fused_attn_residual_norm` — invoked from `execute_layer`'s
//! one-step lookahead when it sees `AttnResidualAdd` followed by
//! `PreFfnNorm` without the Gemma-4 `PostAttnNorm` interleave.

use super::{
    batch_seq_len, decode_io, BatchExec, DecodeExec, ExecCtx,
};
use super::super::arch::{compute_barrier, layer_weight};
use super::super::layer_plan::LayerStep;
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;

impl DecodeExec {
    /// Fused `multi_add_rms`: `res1 = input + o_buf; hidden_norm =
    /// rms_norm(res1) * ffn_norm.weight`. Llama path only.
    pub(super) fn fused_attn_residual_norm(
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
    pub(super) fn execute_step(
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
            LayerStep::VFromKRaw => self.step_v_from_k_raw(fwd, cfg, ctx),
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
            // Sprint 51C — Gemma-4-26B-A4B MoE-block stubs. The
            // layer-plan builder gates these on `enable_moe_block`,
            // so E2B / Qwen3 / Llama never hit them. Loader rejects
            // 26B-A4B with a clear error until Sprint 51D wires the
            // tensor upload + dispatch.
            LayerStep::PostDenseMlpNorm => self.step_post_dense_mlp_norm(fwd, cfg, ctx),
            LayerStep::PreMoeNorm => self.step_pre_moe_norm(fwd, cfg, ctx),
            LayerStep::MoeRoute { n_experts, top_k } => {
                self.step_moe_route(fwd, cfg, ctx, *n_experts, *top_k)
            }
            LayerStep::MoeExpertFfn { n_experts, top_k, moe_intermediate } => {
                self.step_moe_expert_ffn(fwd, cfg, ctx, *n_experts, *top_k, *moe_intermediate)
            }
            LayerStep::PostMoeNorm => self.step_post_moe_norm(fwd, cfg, ctx),
            LayerStep::MoeBranchAdd => self.step_moe_branch_add(fwd, cfg, ctx),
        }
    }
}

impl BatchExec {
    /// Llama-path Sprint-9b fusion: `multi_add_rms(batch_residual,
    /// batch_o, ffn_norm.weight → batch_residual, batch_norm)`.
    /// Replaces the AttnResidualAdd + PreFfnNorm pair when the
    /// loop driver detects the Llama pattern (no PostAttnNorm
    /// preceding).
    pub(super) fn b_fused_attn_residual_norm(
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

    pub(super) fn execute_step(
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
            LayerStep::VFromKRaw => self.b_step_v_from_k_raw(fwd, cfg, ctx),
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
            // Sprint 51C — same MoE stubs as DecodeExec (see comment
            // on the DecodeExec match-arm). BatchExec is loaded
            // through the same `enable_moe_block` gate; never hit on
            // E2B / Qwen3 / Llama.
            LayerStep::PostDenseMlpNorm => self.b_step_post_dense_mlp_norm(fwd, cfg, ctx),
            LayerStep::PreMoeNorm => self.b_step_pre_moe_norm(fwd, cfg, ctx),
            LayerStep::MoeRoute { n_experts, top_k } => {
                self.b_step_moe_route(fwd, cfg, ctx, *n_experts, *top_k)
            }
            LayerStep::MoeExpertFfn { n_experts, top_k, moe_intermediate } => {
                self.b_step_moe_expert_ffn(fwd, cfg, ctx, *n_experts, *top_k, *moe_intermediate)
            }
            LayerStep::PostMoeNorm => self.b_step_post_moe_norm(fwd, cfg, ctx),
            LayerStep::MoeBranchAdd => self.b_step_moe_branch_add(fwd, cfg, ctx),
        }
    }
}

