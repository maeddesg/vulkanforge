//! Sprint 57B — Attention pipeline steps (DEC + BAT pendants).
//!
//! 18 step_* + 18 b_step_* covering norm/projection/RoPE/KV-write/
//! flash-attn/O-proj/post-attn-norm/residual-add. Plus 1 BAT-only
//! helper `b_subscriber_q_barrier` (Sprint 46H — Gemma-4 KV-share
//! synchronization).
//!
//! See `executor/mod.rs` for shared types + helpers.

use super::{
    batch_seq_len, batch_seq_pos, decode_io, decode_position, layer_dims_local,
    n_kv_heads_for, quantize_input_after_q, BatchExec, DecodeExec, ExecCtx, ExecMode,
};
use super::super::arch::{
    compute_barrier, gemma4_kv_read_layer, gemma4_kv_start, layer_weight,
    layer_weight_opt, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_scale_scalar, layer_weight_shader,
};
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;
use super::super::super::pipeline::RopePushConstants;
use super::super::super::shaders::ShaderId;

use ash::vk;

impl DecodeExec {
    pub(super) fn step_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_q_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_k_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    /// Sprint 51B — Gemma-4-26B-A4B `attention_k_eq_v` path. The
    /// layer has no `v_proj` weight; V is taken from K's raw
    /// projection (output of `step_k_proj`, which lives in
    /// `fwd.cur().k_buf` at this point — pre-norm, pre-RoPE).
    /// Copy that buffer into `v_buf` so the downstream `step_v_norm`
    /// (parameterless RMSNorm) operates on it independently of K's
    /// own `step_k_norm_rope` chain.
    pub(super) fn step_v_from_k_raw(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let bytes = (kv_dim as u64) * 4; // FP32 scratch
        let k_buf = fwd.cur().k_buf.handle;
        let v_buf = fwd.cur().v_buf.handle;
        let copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes);
        // Sprint 54D — SHADER_WRITE → TRANSFER_READ barrier on k_buf so
        // step_k_proj's GEMV output is visible to the cmd_copy_buffer
        // below. Without this, the transfer races against the compute
        // write and reads stale k_buf left over from the previous token's
        // CB (V[pos=0] = 0; V[pos=N] = K[pos=N-1] regression signature).
        let pre = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre), &[], &[],
            );
        }
        fwd.profile("v_from_k_raw", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
            dev.device.cmd_copy_buffer(cmd, k_buf, v_buf, std::slice::from_ref(&copy));
        });
        // TRANSFER_WRITE → SHADER_READ barrier on v_buf so downstream
        // step_v_norm's compute read sees the copied bytes — the
        // COMPUTE→COMPUTE barrier issued by maybe_compute_barrier does
        // not cover a prior TRANSFER_WRITE.
        let post = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&post), &[], &[],
            );
        }
        fwd.mark_written(&[v_buf]);
    }

    pub(super) fn step_v_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_q_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_k_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let k_buf = fwd.cur().k_buf.handle;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_k.bias")
            .expect("KBiasAdd emitted but attn_k.bias is missing");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k");
        fwd.mark_written(&[k_buf]);
    }

    pub(super) fn step_v_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let v_buf = fwd.cur().v_buf.handle;
        let b = layer_weight_opt(ctx.model, ctx.layer, "attn_v.bias")
            .expect("VBiasAdd emitted but attn_v.bias is missing");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[v_buf]);
        fwd.run_bias_add(ctx.dev, ctx.registry, ctx.cmd, v_buf, b, v_buf, kv_dim, 1, "bias_v");
        fwd.mark_written(&[v_buf]);
    }

    pub(super) fn step_q_norm_rope(
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

    pub(super) fn step_k_norm_rope(
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

    pub(super) fn step_q_rope(
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

    pub(super) fn step_k_rope(
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

    pub(super) fn step_v_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_kv_write(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_attention(&self, fwd: &mut Forward, _cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_o_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_post_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_attn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
}

impl BatchExec {
    pub(super) fn b_step_attn_norm(&self, _fwd: &mut Forward, _cfg: &ModelConfig, _ctx: &ExecCtx) {
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
    pub(super) fn b_subscriber_q_barrier(&self, cfg: &ModelConfig, ctx: &ExecCtx) {
        if let Some(g) = cfg.gemma4.as_ref() {
            if ctx.layer >= g.first_kv_shared {
                compute_barrier(ctx.dev, ctx.cmd);
            }
        }
    }

    pub(super) fn b_step_q_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_k_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_k.weight",
            fwd.batch_norm.handle,
            fwd.batch_k.handle,
            kv_dim, cfg.hidden_dim, seq_len, "gemm_k",
            quantize_input_after_q(ctx.model, ctx.layer),
        );
    }

    pub(super) fn b_step_v_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_v.weight",
            fwd.batch_norm.handle,
            fwd.batch_v.handle,
            kv_dim, cfg.hidden_dim, seq_len, "gemm_v",
            quantize_input_after_q(ctx.model, ctx.layer),
        );
        // After Q+K+V finished, emit a single barrier (matches the
        // legacy `compute_barrier` after the GEMM block).
        compute_barrier(ctx.dev, ctx.cmd);
    }

    /// Sprint 51B — batch counterpart of `step_v_from_k_raw`. Copies
    /// `batch_k` (raw output of `b_step_k_proj`) into `batch_v` so
    /// `b_step_v_norm` can run independently of K's normalization.
    /// Emits a `compute_barrier` to mirror the trailing barrier on
    /// `b_step_v_proj` (the GEMM block's universal Q+K+V→read sync).
    pub(super) fn b_step_v_from_k_raw(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let bytes = (seq_len as u64) * (kv_dim as u64) * 4; // FP32 batch row
        let src = fwd.batch_k.handle;
        let dst = fwd.batch_v.handle;
        let copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes);
        // Sprint 54D — SHADER_WRITE → TRANSFER_READ barrier on batch_k so
        // b_step_k_proj's GEMM output is visible to the transfer below
        // (same race that hit the decode path: cmd_copy_buffer was reading
        // stale K data from the previous chunk).
        let pre = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre), &[], &[],
            );
        }
        fwd.profile("v_from_k_raw_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
            dev.device.cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&copy));
        });
        // TRANSFER_WRITE → SHADER_READ barrier on batch_v for downstream
        // b_step_v_norm. The bare compute_barrier here previously only
        // covered SHADER→SHADER, leaving the transfer write un-flushed.
        let post = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&post), &[], &[],
            );
        }
    }

    pub(super) fn b_step_q_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_k_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_v_bias(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_q_norm_rope(
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

    pub(super) fn b_step_k_norm_rope(
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

    pub(super) fn b_step_q_rope(
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

    pub(super) fn b_step_k_rope(
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

    pub(super) fn b_step_v_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_kv_write(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_attention(
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

    pub(super) fn b_step_o_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_post_attn_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_attn_residual_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
}
