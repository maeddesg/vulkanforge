//! Sprint 57B — Attention pipeline steps (DEC + BAT pendants).
//!
//! 18 step_* + 18 b_step_* covering norm/projection/RoPE/KV-write/
//! flash-attn/O-proj/post-attn-norm/residual-add. Plus 1 BAT-only
//! helper `b_subscriber_q_barrier` (Sprint 46H — Gemma-4 KV-share
//! synchronization).
//!
//! See `executor/mod.rs` for shared types + helpers.

use super::{
    batch_seq_len, batch_seq_pos, compute_to_transfer_barrier,
    decode_io, decode_position, layer_dims_local,
    n_kv_heads_for, quantize_input_after_q, BatchExec, DecodeExec, ExecCtx,
};
use super::super::arch::{
    compute_barrier, layer_weight, layer_weight_with_offset,
    layer_weight_opt, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_scale_scalar, layer_weight_shader,
    transfer_to_compute_barrier,
};
use super::super::super::pipeline::GatedDeltaNetPushConstants;
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;
use super::super::super::shaders::ShaderId;

use ash::vk;

/// Sprint G-2c (v0.4.6) — Lazy zero-init for the persistent SSM
/// buffers (`conv_state_buf` + `ssm_state_buf`). Fires once per
/// `Forward` instance; subsequent calls observe the flag and bail out
/// without enqueuing transfers. Without this the very first decode /
/// prefill token reads whatever uninitialised GPU memory the
/// allocator handed us, which can produce NaN that propagates
/// forever via the recurrent state. Called from both the DEC and
/// BAT conv-step bodies so any entry path covers init. The
/// flag flips back to `false` on `/reset` in Sprint G-2e.
fn ensure_ssm_persistent_initialized(fwd: &mut Forward, ctx: &ExecCtx) {
    if fwd.ssm_persistent_initialized {
        return;
    }
    let conv_state = fwd.conv_state_buf.as_ref().expect(
        "ssm_persistent_initialized called without conv_state_buf",
    ).handle;
    let ssm_state = fwd.ssm_state_buf.as_ref().expect(
        "ssm_persistent_initialized called without ssm_state_buf",
    ).handle;
    unsafe {
        ctx.dev.device.cmd_fill_buffer(ctx.cmd, conv_state, 0, vk::WHOLE_SIZE, 0);
        ctx.dev.device.cmd_fill_buffer(ctx.cmd, ssm_state,  0, vk::WHOLE_SIZE, 0);
    }
    transfer_to_compute_barrier(ctx.dev, ctx.cmd);
    fwd.ssm_persistent_initialized = true;
}

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
        let (wq, wq_off, wq_sz) = layer_weight_with_offset(ctx.model, ctx.layer, "attn_q.weight");
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
                ctx.dev, ctx.registry, ctx.cmd, sq, wq, wq_off, wq_sz, hidden_norm, q_buf,
                cfg.hidden_dim, q_dim, scale_q, "gemv_q",
            );
        }
        fwd.mark_written(&[q_buf]);
    }

    /// Sprint D2 (v0.4.6) — Qwen3.6 Full-Attention fused Q+Gate
    /// projection. GEMV writes `2 × q_dim` floats into `q_buf`
    /// (resized in setup.rs on qwen35 stacks). Q occupies the
    /// first `q_dim` floats; Gate the second `q_dim`. Downstream
    /// QNormRope reads the first half unchanged.
    pub(super) fn step_attn_q_gate_proj(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        q_dim: u32,
    ) {
        let out_dim = q_dim * 2;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let q_buf = fwd.cur().q_buf.handle;
        let (wq, wq_off, wq_sz) = layer_weight_with_offset(ctx.model, ctx.layer, "attn_q.weight");
        let sq = layer_weight_shader(
            ctx.model, ctx.layer, "attn_q.weight", fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale_q = layer_weight_scale_scalar(ctx.model, ctx.layer, "attn_q.weight");
        let sb_q = layer_weight_scale_buf(ctx.model, ctx.layer, "attn_q.weight");
        if let Some(s) = sb_q {
            let blk = layer_weight_scale_block(ctx.model, ctx.layer, "attn_q.weight");
            fwd.run_gemv_fp8_dispatch(
                ctx.dev, ctx.cmd, wq, s, hidden_norm, q_buf,
                cfg.hidden_dim, out_dim, blk, "gemv_q_gate",
            );
        } else {
            fwd.run_gemv(
                ctx.dev, ctx.registry, ctx.cmd, sq, wq, wq_off, wq_sz, hidden_norm, q_buf,
                cfg.hidden_dim, out_dim, scale_q, "gemv_q_gate",
            );
        }
        fwd.mark_written(&[q_buf]);

        // Sprint G-2g — Qwen3.6 Q-Gate deinterleave (default-on for
        // qwen35 since G-2j coherence pass).
        //
        // llama.cpp `qwen35.cpp::build_qkvz` (268-296) views Qcur_full
        // as interleaved-per-head with `nb1 = 2 × head_dim`: each head's
        // 512 floats are `[Q_h (256), G_h (256)]`. VF's downstream
        // shaders (QNormRope, flash_attn, sigmoid_mul) read CONTIGUOUS
        // layout `[Q_0 ... Q_23, G_0 ... G_23]`. Deinterleave via two
        // `cmd_copy_buffer` passes using `up_buf` as scratch.
        if cfg.qwen35.is_some() {
            let head_dim = cfg.head_dim;
            let n_heads = cfg.n_heads;
            let head_bytes = (head_dim as u64) * 4;
            let stride_bytes = head_bytes * 2;
            let q_total_bytes = (q_dim as u64) * 4;
            let scratch = fwd.cur().up_buf.handle;

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

            let mut regions: Vec<vk::BufferCopy> =
                Vec::with_capacity(2 * n_heads as usize);
            for h in 0..n_heads as u64 {
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset(h * stride_bytes)
                        .dst_offset(h * head_bytes)
                        .size(head_bytes),
                );
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset(h * stride_bytes + head_bytes)
                        .dst_offset(q_total_bytes + h * head_bytes)
                        .size(head_bytes),
                );
            }
            fwd.profile("q_gate_deinterleave", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                dev.device.cmd_copy_buffer(cmd, q_buf, scratch, &regions);
            });

            let mid = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&mid), &[], &[],
                );
            }

            let total_bytes = (out_dim as u64) * 4;
            let copy_back = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(total_bytes);
            fwd.profile("q_gate_copy_back", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                dev.device.cmd_copy_buffer(cmd, scratch, q_buf, std::slice::from_ref(&copy_back));
            });

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
            fwd.mark_written(&[q_buf]);
        }
    }

    pub(super) fn step_k_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let kv_dim = n_kv_heads_for(cfg, ctx.layer) * head_dim;
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let k_buf = fwd.cur().k_buf.handle;
        let (wk, wk_off, wk_sz) = layer_weight_with_offset(ctx.model, ctx.layer, "attn_k.weight");
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
                ctx.dev, ctx.registry, ctx.cmd, sk, wk, wk_off, wk_sz, hidden_norm, k_buf,
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
        let (wv, wv_off, wv_sz) = layer_weight_with_offset(ctx.model, ctx.layer, "attn_v.weight");
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
                ctx.dev, ctx.registry, ctx.cmd, sv, wv, wv_off, wv_sz, hidden_norm, v_buf,
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
        // Sprint G-2h — Qwen3.6 needs a TRANSFER→COMPUTE barrier on
        // q_buf at QNormRope entry: the upstream Q-Gate-deinterleave
        // writes q_buf via cmd_copy_buffer (TRANSFER stage), and the
        // generic `maybe_compute_barrier` only issues SHADER→SHADER.
        // Other archs hit a plain COMPUTE write of q_buf (GEMV) so
        // SHADER→SHADER suffices.
        if cfg.qwen35.is_some() {
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                ctx.dev.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&mb), &[], &[],
                );
            }
        } else {
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q_buf]);
        }
        // Sprint G-2j — Qwen3.6 uses split RMSNorm + RoPE dispatches
        // (fused `rms_norm_mul_rope` is numerically wrong on
        // RADV/gfx1201 even after G-2i barrier fixes). Plus a
        // mid-frame-submit drain at L3 (first Full-Attention layer)
        // between the two dispatches to defuse a Heisenberg sync bug.
        // Other archs keep the fused path.
        if cfg.qwen35.is_some() {
            fwd.run_rms_norm(
                ctx.dev, ctx.registry, ctx.cmd,
                q_buf, wqn, q_buf,
                head_dim, cfg.n_heads,
                cfg.rms_norm_eps, "rms_norm_q_split",
            );
            fwd.mark_written(&[q_buf]);
            super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
            // Sprint G-3 — L3 mid_frame_submit_and_wait (added in G-2j)
            // was phantom: only required as a proxy for the async-decode
            // race. With sync-decode default-on for qwen35 (see
            // decode.rs:570) the L3 drain produces no behavioural
            // change. Removed.
            fwd.run_rope_neox_with_pos_offset(
                ctx.dev, ctx.registry, ctx.cmd, q_buf, q_buf,
                head_dim, rotary_dim, freq_base, theta_scale,
                cfg.n_heads,
                /* position = */ 0, // unused; pos is read from rope_pos_buf
                /* pos_buf_offset = */ 0,
                "rope_neox_q_split",
            );
            fwd.mark_written(&[q_buf]);
        } else {
            fwd.run_rms_norm_mul_rope(
                ctx.dev, ctx.registry, ctx.cmd,
                q_buf, wqn, q_buf,
                head_dim, rotary_dim, freq_base, theta_scale,
                cfg.n_heads, 1,
                cfg.rms_norm_eps, "rms_norm_mul_rope_q",
            );
            fwd.mark_written(&[q_buf]);
        }
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
        let n_kv = n_kv_heads_for(cfg, ctx.layer);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[k_buf]);
        // Sprint G-2j — Qwen3.6 uses split RMSNorm + RoPE dispatches
        // (see step_q_norm_rope for rationale).
        if cfg.qwen35.is_some() {
            // De-risk (gated, default-off): VF_DERISK_SKIP_KNORM skips the
            // k-norm in BOTH decode and batch to isolate proj-vs-norm.
            if std::env::var("VF_DERISK_SKIP_KNORM").as_deref() != Ok("1") {
                fwd.run_rms_norm(
                    ctx.dev, ctx.registry, ctx.cmd,
                    k_buf, wkn, k_buf,
                    head_dim, n_kv,
                    cfg.rms_norm_eps, "rms_norm_k_split",
                );
                fwd.mark_written(&[k_buf]);
                super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
            }
            // De-risk (gated, default-off): VF_DERISK_SKIP_KROPE skips the
            // k-rope in BOTH decode and batch k-norm-rope so the substep
            // harness can isolate norm-vs-rope as the head_dim=256 diff.
            if std::env::var("VF_DERISK_SKIP_KROPE").as_deref() != Ok("1") {
                fwd.run_rope_neox_with_pos_offset(
                    ctx.dev, ctx.registry, ctx.cmd, k_buf, k_buf,
                    head_dim, rotary_dim, freq_base, theta_scale,
                    n_kv,
                    /* position = */ 0,
                    /* pos_buf_offset = */ 0,
                    "rope_neox_k_split",
                );
                fwd.mark_written(&[k_buf]);
            }
        } else {
            fwd.run_rms_norm_mul_rope(
                ctx.dev, ctx.registry, ctx.cmd,
                k_buf, wkn, k_buf,
                head_dim, rotary_dim, freq_base, theta_scale,
                n_kv, 1,
                cfg.rms_norm_eps, "rms_norm_mul_rope_k",
            );
            fwd.mark_written(&[k_buf]);
        }
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
        // Sprint G-6 trailing-barrier strip restored Sprint G-7 — on the
        // Llama / Gemma-4 paths the immediate next step is `step_o_proj`,
        // which has NO leading `maybe_compute_barrier` of its own (relies
        // on the producer's trailing). Stripping this trailing made
        // async-decode'd Gemma-4-26B emit garbage tokens because the
        // O-proj GEMV started before scalar_attn's writes were visible.
        // Synchronisation-validation didn't catch the race in the
        // sync-decode path because the single command-buffer issue order
        // happened to drain attn_out before next dispatch. Keep this
        // trailing barrier — the Qwen3.6 Full-Attn path (which has
        // `step_attn_gated_output` between attention and o_proj) absorbs
        // it as a no-op via the elision tracker.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[attn_out]);
    }

    /// Sprint D2 (v0.4.6) — Qwen3.6 Full-Attention gated output.
    /// `attn_out[i] *= sigmoid(q_buf[q_dim + i])`. Decode path:
    /// single-token, gate sits contiguously at `q_buf[q_dim..2*q_dim]`
    /// so the shader's strided form collapses to `stride = q_dim`.
    pub(super) fn step_attn_gated_output(
        &self,
        fwd: &mut Forward,
        _cfg: &ModelConfig,
        ctx: &ExecCtx,
        q_dim: u32,
    ) {
        let attn_out = fwd.cur().attn_out.handle;
        let q_buf = fwd.cur().q_buf.handle;
        // Both buffers were written by earlier dispatches (attention
        // → attn_out, Q-Gate proj → q_buf). Flush both before the
        // sigmoid-mul reads the gate and writes the in-place result.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[attn_out, q_buf]);
        // De-risk (gated, default-off): VF_DERISK_SKIP_GATE skips the
        // output gate in BOTH decode and batch to test whether the
        // qwen35-only gated-output (interleaved-vs-concat gate layout) is
        // the residual batched-vs-decode divergence.
        if std::env::var("VF_DERISK_SKIP_GATE").as_deref() != Ok("1") {
            fwd.run_sigmoid_mul(
                ctx.dev, ctx.registry, ctx.cmd,
                q_buf, attn_out,
                q_dim,        // ne
                q_dim,        // chunk
                q_dim,        // stride (single token: 1 chunk per stride)
                q_dim,        // gate_offset (Gate starts after Q in q_buf)
                "sigmoid_mul_gated_out",
            );
            fwd.mark_written(&[attn_out]);
        }
        // Trailing barrier mirroring the batch path (b_step_attn_gated_output,
        // attention.rs:2281). step_o_proj reads attn_out via gemv but does NOT
        // maybe_compute_barrier on it, so the in-place sigmoid_mul write above
        // was unsynced before that read — a real RAW race in qwen35 decode
        // (Imperative mode + elision = the default), widened by head_dim=256.
        // This was the source of the "q36 pre-existing non-determinism": the
        // racy read returned a partially-gated attn_out, so greedy decode was
        // bit-non-deterministic (verified: OFF diverges across runs, ON is
        // bit-identical). The barrier makes o_proj read the fully-gated value —
        // the *intended* result — so this is a correctness fix, not a speed
        // trade-off (decode t/s neutral). DEFAULT-ON since v0.5.4;
        // VF_QWEN35_DEC_GATE_BARRIER=0 is a non-destructive escape-hatch.
        // qwen35-only (this step is only in the qwen35 full-attn plan) → zero
        // effect on Llama / Qwen3 / Gemma decode (verified byte-identical).
        if std::env::var("VF_QWEN35_DEC_GATE_BARRIER").as_deref() != Ok("0") {
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[attn_out]);
        }
    }

    pub(super) fn step_o_proj(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let (head_dim, _, _, _) = layer_dims_local(cfg, ctx.layer);
        let q_dim = cfg.n_heads * head_dim;
        let attn_out = fwd.cur().attn_out.handle;
        let o_buf = fwd.cur().o_buf.handle;
        let (wo, wo_off, wo_sz) = layer_weight_with_offset(ctx.model, ctx.layer, "attn_output.weight");
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
                ctx.dev, ctx.registry, ctx.cmd, so, wo, wo_off, wo_sz, attn_out, o_buf,
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

    /// Sprint D (v0.4.6) — Qwen3.5/3.6 skeleton passthrough.
    /// Decode path: `res1 = input` so the downstream PreFfnNorm
    /// (which reads `res1`) and FfnResidualAdd (`output = res1 +
    /// ffn_out`) collapse the attention sub-block to the identity.
    /// Replaces the QProj→OProj→AttnResidualAdd chain emitted by
    /// `build_qwen3_layer` / `build_gemma4_layer`. Sprint D2-G
    /// replaces this on Full-Attention layers with the real
    /// Q-Gate-Split + Attention dispatches.
    pub(super) fn step_residual_identity_seed(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
    ) {
        let (input, _) = decode_io(ctx);
        let res1 = fwd.cur().res1.handle;
        let bytes = (cfg.hidden_dim as u64) * 4;
        let copy = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(bytes);
        // input arrives from the previous layer's FfnResidualAdd
        // (compute-shader write). Flush before the transfer reads it.
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
        fwd.profile("residual_identity_seed", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
            dev.device.cmd_copy_buffer(cmd, input, res1, std::slice::from_ref(&copy));
        });
        // res1 is consumed by PreFfnNorm (compute read). TRANSFER_WRITE
        // → SHADER_READ barrier so the subsequent dispatch sees the
        // copied bytes.
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
        fwd.mark_written(&[res1]);
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

    /// Sprint G-2c (v0.4.6) — real SSM 1D convolution dispatch with
    /// persistent state-shift.
    ///
    /// Three dispatches per call:
    ///   1. **`ssm_conv_setup`** — one thread per channel builds the
    ///      4-slot `conv_input` window from `conv_state[..]` + the
    ///      current-token `ssm_qkv_buf`, AND slides the state window
    ///      one position left (new state = `[s1, s2, qkv]`). Fused so
    ///      we only read the per-channel state into registers once.
    ///   2. **`ssm_conv`** — Sprint F shader, channel-major-time-inner
    ///      layout, `n_t = n_s = 1` for decode.
    ///   3. Implicit barrier so downstream `SsmSilu` / `SsmQkL2Norm`
    ///      observe `conv_output`'s writes.
    ///
    /// First call lazily zero-fills the persistent `conv_state` +
    /// `ssm_state` buffers (so the first token doesn't read whatever
    /// uninitialised GPU memory the allocator handed us). The flag
    /// resets on `/reset` in Sprint G-2e.
    pub(super) fn step_ssm_conv1d(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_conv1d emitted for non-qwen35 config",
        );
        let conv_channels = spec.conv_channels();
        let d_conv = spec.ssm_d_conv;
        let slots = d_conv - 1;
        let recurrent_idx = spec.recurrent_index(layer);

        ensure_ssm_persistent_initialized(fwd, ctx);

        let conv_state_buf = fwd.conv_state_buf.as_ref().unwrap().handle;
        let qkv_buf       = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
        let conv_input    = fwd.ssm_conv_input_buf.as_ref().unwrap().handle;
        let conv_output   = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;

        let state_bytes_per_layer = slots as u64 * conv_channels as u64 * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;

        // Build conv_input + slide state window.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[qkv_buf]);
        fwd.run_ssm_conv_setup(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_state_buf, state_offset_bytes, state_bytes_per_layer,
            qkv_buf, conv_input,
            conv_channels, "ssm_conv_setup",
        );
        fwd.mark_written(&[conv_input, conv_state_buf]);
        // Sprint SG-1.4-b — sub-dispatch barrier MUST fire even under
        // BarrierMode::GraphDriven (the graph models step_ssm_conv1d
        // as one node; it can't place a barrier between the two
        // internal sub-dispatches).
        fwd.force_internal_barrier(ctx.dev, ctx.cmd, &[conv_input]);

        // Real conv dispatch (decode: n_t = 1, n_s = 1, ncs = d_conv).
        let conv_weight = layer_weight(ctx.model, layer, "ssm_conv1d.weight");
        fwd.run_ssm_conv(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_input, conv_weight, conv_output,
            d_conv, d_conv, conv_channels, 1, 1, "ssm_conv",
        );
        fwd.mark_written(&[conv_output]);
        // Sprint G-6 — trailing barrier elided. The next consumer
        // (step_ssm_silu's leading maybe_compute_barrier(&[conv_output]))
        // will emit a barrier once it sees conv_output in pending_writes;
        // emitting one here forces an unconditional drain of all in-flight
        // dispatches, which serializes the 4 RAW-independent SSM-side
        // GEMVs above it. Matches the lean FFN-step pattern in ffn.rs.
    }

    // ──────────────────────────────────────────────────────────
    // Sprint G-2b (v0.4.6) — Qwen3.6 Linear-Attn step stubs (DEC).
    //
    // All 10 are no-op staging bodies. Match-arms in
    // `executor/dispatch.rs` route to them so the exhaustive-match
    // gate stays satisfied; real bodies arrive in:
    //   G-2c → step_ssm_conv1d body, step_ssm_silu, step_ssm_qk_l2_norm
    //   G-2d → step_attn_qkv_proj, step_attn_gate_z_proj,
    //          step_ssm_beta_proj, step_ssm_alpha_gate,
    //          step_ssm_repeat_qk, step_ssm_out_proj
    //   G-2e → step_gated_delta_net, step_norm_gated
    // ──────────────────────────────────────────────────────────

    /// Sprint G-2d (v0.4.6) — `attn_qkv.weight` GEMV
    /// (`[hidden_dim → conv_channels]`). For Qwen3.6-27B: Q3_K
    /// `[5120, 10240]`. Writes the per-token mixed QKV slab into
    /// `ssm_qkv_buf` which `step_ssm_conv1d` then consumes as the
    /// current-token slot of the rolling conv window.
    pub(super) fn step_attn_qkv_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_attn_qkv_proj emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let qkv_buf = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "attn_qkv.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "attn_qkv.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "attn_qkv.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, hidden_norm, qkv_buf,
            cfg.hidden_dim, spec.conv_channels(), scale, "gemv_ssm_qkv",
        );
        fwd.mark_written(&[qkv_buf]);
        // Sprint G-6 — trailing barrier elided (see step_ssm_conv1d comment).
    }

    /// Sprint G-2d (v0.4.6) — `attn_gate.weight` GEMV
    /// (`[hidden_dim → ssm_d_inner]`). Q3_K `[5120, 6144]`. Output
    /// `ssm_z_buf` is the multiplicative gate `z` for the later
    /// NormGated step. Reads the same `hidden_norm` as AttnQkvProj
    /// (both readonly — no barrier needed between).
    pub(super) fn step_attn_gate_z_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_attn_gate_z_proj emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let z_buf = fwd.ssm_z_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "attn_gate.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "attn_gate.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "attn_gate.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, hidden_norm, z_buf,
            cfg.hidden_dim, spec.ssm_d_inner, scale, "gemv_ssm_z",
        );
        fwd.mark_written(&[z_buf]);
        // Sprint G-6 — trailing barrier elided. z_buf isn't read until
        // step_norm_gated, whose leading barrier will emit if needed.
    }

    /// Sprint G-2d (v0.4.6) — `ssm_beta.weight` GEMV `[5120, 48]` F32 +
    /// in-place sigmoid. Mirrors `qwen35.cpp:397
    /// ggml_sigmoid(beta_cur)`. Output feeds GDN's β binding directly.
    pub(super) fn step_ssm_beta_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_beta_proj emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let beta_buf = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "ssm_beta.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "ssm_beta.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_beta.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, hidden_norm, beta_buf,
            cfg.hidden_dim, spec.num_v_heads(), scale, "gemv_ssm_beta",
        );
        fwd.mark_written(&[beta_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[beta_buf]);
        fwd.run_sigmoid(
            ctx.dev, ctx.registry, ctx.cmd,
            beta_buf, spec.num_v_heads(), "ssm_beta_sigmoid",
        );
        fwd.mark_written(&[beta_buf]);
        // Sprint G-6 — trailing barrier elided. beta_buf is consumed
        // by step_gated_delta_net, whose leading barrier covers it.
    }

    /// Sprint G-2d (v0.4.6) — fused alpha-gate compose (Ops 6-9 of
    /// `build_layer_attn_linear`):
    ///   1. `alpha = ssm_alpha.weight @ hidden_norm`         (F32 GEMV)
    ///   2. `alpha += ssm_dt.bias`                            (elementwise)
    ///   3. `alpha = softplus(alpha)`                          (in-place)
    ///   4. `gate  = alpha * ssm_a`                            (elementwise)
    /// Output `ssm_gate_buf` is GDN's log-domain decay term (g
    /// binding).
    pub(super) fn step_ssm_alpha_gate(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_alpha_gate emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let alpha_buf = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        let gate_buf  = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let n_heads   = spec.num_v_heads();

        // 1. alpha = ssm_alpha.weight @ hidden_norm
        let (w_alpha, w_alpha_off, w_alpha_sz) =
            layer_weight_with_offset(ctx.model, layer, "ssm_alpha.weight");
        let s_alpha = layer_weight_shader(
            ctx.model, layer, "ssm_alpha.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_alpha.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[hidden_norm]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s_alpha, w_alpha,
            w_alpha_off, w_alpha_sz, hidden_norm, alpha_buf,
            cfg.hidden_dim, n_heads, scale, "gemv_ssm_alpha",
        );
        fwd.mark_written(&[alpha_buf]);
        // SG-1.4-b — sub-dispatch barrier (alpha just written by GEMV).
        fwd.force_internal_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        // 2. alpha += ssm_dt.bias
        let dt_bias = layer_weight(ctx.model, layer, "ssm_dt.bias");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            alpha_buf, dt_bias, alpha_buf,
            n_heads, "ssm_alpha_add_dt",
        );
        fwd.mark_written(&[alpha_buf]);
        // SG-1.4-b — sub-dispatch barrier (alpha just written by Add).
        fwd.force_internal_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        // 3. alpha = softplus(alpha) in-place
        fwd.run_softplus(
            ctx.dev, ctx.registry, ctx.cmd,
            alpha_buf, n_heads, "ssm_alpha_softplus",
        );
        fwd.mark_written(&[alpha_buf]);
        // SG-1.4-b — sub-dispatch barrier (alpha just written by Softplus).
        fwd.force_internal_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        // 4. gate = alpha * ssm_a (note: ssm_a has NO .weight suffix in GGUF).
        let ssm_a = layer_weight(ctx.model, layer, "ssm_a");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            alpha_buf, ssm_a, gate_buf,
            n_heads, "ssm_alpha_mul_a",
        );
        fwd.mark_written(&[gate_buf]);
        // Sprint G-6 — trailing barrier elided. gate_buf is consumed
        // by step_gated_delta_net (binding g), whose leading barrier
        // (&[q, k, v, g, beta]) covers it.
    }

    /// Sprint G-2c (v0.4.6) — in-place SiLU on `ssm_conv_output_buf`
    /// (Op 11 of `build_layer_attn_linear`). The silu pipeline reads
    /// binding 0 once into a register before writing binding 1, so the
    /// in-place aliasing (same buffer in both slots) is hazard-free.
    pub(super) fn step_ssm_silu(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_silu emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        fwd.run_silu(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            spec.conv_channels(), "ssm_silu",
        );
        fwd.mark_written(&[conv_output]);
        // Sprint G-6 — trailing barrier elided. step_ssm_qk_l2_norm's
        // leading barrier(&[conv_output]) covers the RAW dependency.
    }

    /// Sprint G-2c (v0.4.6) — in-place L2-norm on the Q and K slices
    /// of `ssm_conv_output_buf` (Op 13 of `build_layer_attn_linear`).
    ///
    /// Conv-output layout (channel-major, `conv_channels=10240` floats):
    ///   `[0..2048)`  → Q  (16 heads × 128 dim)
    ///   `[2048..4096)` → K  (16 heads × 128 dim)
    ///   `[4096..10240)` → V  (NOT normalised here — repeats unchanged)
    ///
    /// One dispatch per slice; the workgroup-per-row tree reduction
    /// inside `l2_norm.comp` handles the head-axis (16 rows × 128 cols).
    pub(super) fn step_ssm_qk_l2_norm(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_qk_l2_norm emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let head_dim = spec.head_k_dim();          // 128
        let n_heads  = spec.num_k_heads();         // 16
        let qk_floats = head_dim * n_heads;        // 2048 per slice
        let eps = cfg.rms_norm_eps;

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        // Q-slice: rows 0..n_heads at base_offset = 0.
        fwd.run_l2_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            head_dim, n_heads, /* base_offset_floats = */ 0,
            eps, "ssm_l2_norm_q",
        );
        // K-slice: rows 0..n_heads at base_offset = qk_floats (2048).
        fwd.run_l2_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            head_dim, n_heads, /* base_offset_floats = */ qk_floats,
            eps, "ssm_l2_norm_k",
        );
        fwd.mark_written(&[conv_output]);
        // Sprint G-6 — trailing barrier elided. step_ssm_repeat_qk's
        // leading barrier(&[conv_output]) covers the RAW dependency.
    }

    /// Sprint G-2d (v0.4.6) — Head-axis 16 → 48 repeat for Q + K
    /// post-L2-norm (Op 14 of `build_layer_attn_linear`). Reads two
    /// slices of `ssm_conv_output_buf`:
    ///   Q: bytes `[0, 2048×4)`  → `ssm_qrep_buf` `[48 × 128]`
    ///   K: bytes `[2048×4, 4096×4)` → `ssm_krep_buf` `[48 × 128]`
    /// Bound at offset via descriptor set, so no extra copy step.
    pub(super) fn step_ssm_repeat_qk(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_repeat_qk emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let qrep = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
        let krep = fwd.ssm_krep_buf.as_ref().unwrap().handle;
        let head_dim    = spec.head_k_dim();        // 128
        let n_src_heads = spec.num_k_heads();       // 16
        let n_dst_heads = spec.num_v_heads();       // 48
        let n_tokens    = 1u32;                     // decode
        let slice_bytes = head_dim as u64 * n_src_heads as u64 * 4; // 2048 floats × 4 = 8192 B

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        // Q at offset 0.
        fwd.run_repeat_interleave(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, /* src_offset_bytes = */ 0, slice_bytes, qrep,
            head_dim, n_src_heads, n_dst_heads, n_tokens, "ssm_repeat_q",
        );
        // K at offset 2048 floats = 8192 bytes.
        fwd.run_repeat_interleave(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, slice_bytes, slice_bytes, krep,
            head_dim, n_src_heads, n_dst_heads, n_tokens, "ssm_repeat_k",
        );
        fwd.mark_written(&[qrep, krep]);
        // Sprint G-6 — trailing barrier elided. step_gated_delta_net
        // reads q/k via its leading barrier(&[q, k, v, g, beta]).
    }

    /// Sprint G-2e (v0.4.6) — The recurrence at the heart of Qwen3.6
    /// Linear-Attention.
    ///
    /// Dispatches the 190-LOC `gated_delta_net.comp` (Sprint G-2a) with:
    ///   binding 0  Q      `ssm_qrep_buf`         [H × K]
    ///   binding 1  K      `ssm_krep_buf`         [H × K]
    ///   binding 2  V      `ssm_conv_output_buf`  V-slice at offset
    ///                                              `2 × head_k × num_k_heads`
    ///                                              floats (skip Q + K)
    ///   binding 3  G      `ssm_gate_buf`         [H]      log-decay
    ///   binding 4  Beta   `ssm_beta_buf`         [H]      post-sigmoid
    ///   binding 5  State  `ssm_state_buf`        per-layer slice (RO)
    ///   binding 6  Dst    `ssm_gdn_out_buf`      output [H × S_v]
    ///                                            + new state [H × S_v²]
    ///                                            co-located at `s_off`
    ///
    /// After the dispatch a `cmd_copy_buffer` copies the new state
    /// portion of dst back into `ssm_state_buf` at the per-layer
    /// offset for the next token to read. The GDN shader cannot
    /// overwrite State binding-5 in the same dispatch because it's
    /// declared `readonly` (Vulkan validation enforces this).
    ///
    /// Push consts mirror llama.cpp `vk_op_gated_delta_net_push_constants`
    /// 1:1 (17 scalars, 68 B). Strides are in **elements**.
    pub(super) fn step_gated_delta_net(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_gated_delta_net emitted for non-qwen35 config",
        );
        let h           = spec.num_v_heads();        // 48
        let s_v         = spec.ssm_d_state;          // 128
        let head_k      = spec.head_k_dim();         // 128
        let num_k_heads = spec.num_k_heads();        // 16
        let recurrent_idx = spec.recurrent_index(layer);

        let q     = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
        let k     = fwd.ssm_krep_buf.as_ref().unwrap().handle;
        let v     = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let g     = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let beta  = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let state = fwd.ssm_state_buf.as_ref().unwrap().handle;
        let dst   = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;

        // V-slice in conv_output: per qwen35.cpp:406-422 V sits at
        // offset `head_k_dim × num_k_heads × 2` floats (= Q + K halves).
        // For Qwen3.6: (128 × 16 × 2) = 4096 floats = 16 384 bytes.
        let v_offset_bytes = (head_k as u64) * (num_k_heads as u64) * 2 * 4;
        let v_bytes        = (s_v as u64) * (h as u64) * 4;     // 6144 × 4 = 24 576 B

        // Per-layer state slice (input + ultimate output destination).
        let state_bytes_per_layer = (h as u64) * (s_v as u64) * (s_v as u64) * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;

        // s_off (elements within dst) — where new state begins:
        //   output occupies dst[0 .. S_v × H × n_tokens × n_seqs)
        let s_off = s_v * h;

        let push = GatedDeltaNetPushConstants {
            h,
            n_tokens: 1,
            n_seqs:   1,
            s_off,
            // Q strides (elements). Layout [K, H, n_tokens=1, n_seqs=1].
            sq1: head_k, sq2: head_k * h, sq3: head_k * h,
            // V strides — same shape but inner = S_v (= K for square-state Qwen3.6).
            sv1: s_v,    sv2: s_v * h,    sv3: s_v * h,
            // Beta / g strides — [H, n_tokens, n_seqs], head-inner.
            sb1: 1, sb2: h, sb3: h,
            neq1: h,
            rq3:  1,
            scale: 1.0 / (s_v as f32).sqrt(),
            k: 1,
        };

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q, k, v, g, beta]);
        fwd.run_gated_delta_net(
            ctx.dev, ctx.registry, ctx.cmd,
            q, k,
            v, v_offset_bytes, v_bytes,
            g, beta,
            state, state_offset_bytes, state_bytes_per_layer,
            dst,
            s_v, /* cols_per_wg = SUBGROUP_SIZE/LANES_PER_COLUMN */ 2,
            &push, "gdn",
        );
        fwd.mark_written(&[dst]);

        // Copy the new state portion of `dst` back into `ssm_state_buf`
        // at the per-layer offset so the next token's GDN dispatch
        // reads it via binding 5.
        //
        // Sprint G-2i — full WAW/RAW barrier before the copy. The
        // synchronisation validation layer (`validate_sync = true`)
        // flagged WAW: cmd_copy_buffer writes `state` which was
        // previously written by cmd_fill_buffer (init) without a
        // chained transfer-write barrier. The intervening SHADER_READ
        // accesses don't satisfy the WAW chain. Need:
        //   src: SHADER_WRITE | SHADER_READ | TRANSFER_WRITE
        //   dst: TRANSFER_READ | TRANSFER_WRITE
        // at stages COMPUTE_SHADER|TRANSFER → TRANSFER. `compute_to_transfer_barrier`
        // alone (SHADER_WRITE → TRANSFER_READ) covered only the dst-buffer
        // RAW for reading, not the state-buffer WAW for writing.
        let pre_copy = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre_copy), &[], &[],
            );
        }
        let region = vk::BufferCopy {
            src_offset: (s_off as u64) * 4,
            dst_offset: state_offset_bytes,
            size:       state_bytes_per_layer,
        };
        unsafe {
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, dst, state, std::slice::from_ref(&region),
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);
        fwd.mark_written(&[state]);
    }

    /// Sprint G-2e (v0.4.6) — `RMSNorm(ssm_norm.weight) × SiLU(z)`,
    /// Op 16 of `build_layer_attn_linear`. Three dispatches (safe
    /// path per G-2d §11.2 recommendation):
    ///   1. RMSNorm: gdn_out [H × S_v] → norm_out, weight=`ssm_norm.weight`
    ///      (per-head, S_v cols × H rows).
    ///   2. SiLU on `z` in-place.
    ///   3. norm_out *= silu(z) elementwise.
    pub(super) fn step_norm_gated(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_norm_gated emitted for non-qwen35 config",
        );
        let h   = spec.num_v_heads();
        let s_v = spec.ssm_d_state;
        let ne  = h * s_v;

        let gdn_out  = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;
        let z        = fwd.ssm_z_buf.as_ref().unwrap().handle;
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        let norm_w   = layer_weight(ctx.model, layer, "ssm_norm.weight");

        // 1. RMSNorm per head.
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gdn_out]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            gdn_out, norm_w, norm_out,
            s_v, h, cfg.rms_norm_eps, "ssm_norm_gated_rms",
        );
        fwd.mark_written(&[norm_out]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[z]);

        // 2. SiLU on z in-place.
        fwd.run_silu(
            ctx.dev, ctx.registry, ctx.cmd,
            z, z, ne, "ssm_z_silu",
        );
        fwd.mark_written(&[z]);
        // SG-1.4-b — sub-dispatch barrier (norm_out + z both just
        // written by RMSNorm + SiLU above; Mul reads both).
        fwd.force_internal_barrier(ctx.dev, ctx.cmd, &[norm_out, z]);

        // 3. norm_out *= silu(z) (elementwise).
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            norm_out, z, norm_out,
            ne, "ssm_norm_gated_mul",
        );
        fwd.mark_written(&[norm_out]);
        // Sprint G-6 — trailing barrier elided. step_ssm_out_proj's
        // leading barrier(&[norm_out]) covers the RAW dependency.
    }

    /// Sprint G-2d (v0.4.6) — `ssm_out.weight` GEMV `[6144, 5120]` Q4_K
    /// (Op 17 of `build_layer_attn_linear`). Reads `ssm_norm_out_buf`
    /// (NormGated output, still no-op-zero until G-2e ships), writes
    /// the per-layer hidden-dim attention output into `o_buf` where
    /// `step_attn_residual_add` consumes it (`addend = o_buf` on the
    /// non-Gemma path).
    pub(super) fn step_ssm_out_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "step_ssm_out_proj emitted for non-qwen35 config",
        );
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        let o_buf    = fwd.cur().o_buf.handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "ssm_out.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "ssm_out.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_out.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[norm_out]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, norm_out, o_buf,
            spec.ssm_d_inner, cfg.hidden_dim, scale, "gemv_ssm_out",
        );
        fwd.mark_written(&[o_buf]);
        // Sprint G-6 — trailing barrier elided. step_attn_residual_add's
        // leading barrier(&[res1, addend]) (addend = o_buf) covers it.
    }

    // ──────────────────────────────────────────────────────────────────
    // Sprint SG-3 — decomposed SSM sub-dispatches.
    //
    // Each helper below issues exactly ONE compute dispatch (or one
    // transfer + barriers, for `sub_gdn_state_copy`). NO leading
    // `maybe_compute_barrier`, NO trailing `mark_written`, NO
    // `force_internal_barrier` — all synchronization comes from the
    // graph's edge set + barrier pass in `execute_layer_via_graph`.
    //
    // The corresponding monolithic `step_*` bodies above remain in
    // place: they are still called via `execute_step` from the
    // imperative path (`execute_layer` loop) and from the graph's
    // `SubDispatch::FullStep` arm. The decomposition splits ONE
    // FullStep into N SubDispatch nodes per step in the graph build.
    // ──────────────────────────────────────────────────────────────────

    // ── NormGated (3) ───────────────────────────────────────────────
    pub(super) fn sub_norm_gated_rms(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_norm_gated_rms emitted for non-qwen35 config",
        );
        let h   = spec.num_v_heads();
        let s_v = spec.ssm_d_state;
        let gdn_out  = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        let norm_w   = layer_weight(ctx.model, layer, "ssm_norm.weight");
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            gdn_out, norm_w, norm_out,
            s_v, h, cfg.rms_norm_eps, "ssm_norm_gated_rms",
        );
    }

    pub(super) fn sub_norm_gated_silu(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_norm_gated_silu emitted for non-qwen35 config",
        );
        let ne = spec.num_v_heads() * spec.ssm_d_state;
        let z  = fwd.ssm_z_buf.as_ref().unwrap().handle;
        fwd.run_silu(
            ctx.dev, ctx.registry, ctx.cmd,
            z, z, ne, "ssm_z_silu",
        );
    }

    pub(super) fn sub_norm_gated_mul(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_norm_gated_mul emitted for non-qwen35 config",
        );
        let ne = spec.num_v_heads() * spec.ssm_d_state;
        let z        = fwd.ssm_z_buf.as_ref().unwrap().handle;
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            norm_out, z, norm_out,
            ne, "ssm_norm_gated_mul",
        );
    }

    // ── SsmAlphaGate (4) ────────────────────────────────────────────
    pub(super) fn sub_alpha_gate_gemv(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_alpha_gate_gemv emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let alpha_buf   = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        let (w_alpha, w_alpha_off, w_alpha_sz) =
            layer_weight_with_offset(ctx.model, layer, "ssm_alpha.weight");
        let s_alpha = layer_weight_shader(
            ctx.model, layer, "ssm_alpha.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_alpha.weight");
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s_alpha, w_alpha,
            w_alpha_off, w_alpha_sz, hidden_norm, alpha_buf,
            cfg.hidden_dim, spec.num_v_heads(), scale, "gemv_ssm_alpha",
        );
    }

    pub(super) fn sub_alpha_gate_add(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_alpha_gate_add emitted for non-qwen35 config",
        );
        let alpha_buf = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        let dt_bias = layer_weight(ctx.model, layer, "ssm_dt.bias");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            alpha_buf, dt_bias, alpha_buf,
            spec.num_v_heads(), "ssm_alpha_add_dt",
        );
    }

    pub(super) fn sub_alpha_gate_softplus(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_alpha_gate_softplus emitted for non-qwen35 config",
        );
        let alpha_buf = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        fwd.run_softplus(
            ctx.dev, ctx.registry, ctx.cmd,
            alpha_buf, spec.num_v_heads(), "ssm_alpha_softplus",
        );
    }

    pub(super) fn sub_alpha_gate_mul_a(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_alpha_gate_mul_a emitted for non-qwen35 config",
        );
        let alpha_buf = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        let gate_buf  = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let ssm_a = layer_weight(ctx.model, layer, "ssm_a");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            alpha_buf, ssm_a, gate_buf,
            spec.num_v_heads(), "ssm_alpha_mul_a",
        );
    }

    // ── SsmConv1d (2) ───────────────────────────────────────────────
    pub(super) fn sub_conv_setup(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_conv_setup emitted for non-qwen35 config",
        );
        let conv_channels = spec.conv_channels();
        let d_conv = spec.ssm_d_conv;
        let slots = d_conv - 1;
        let recurrent_idx = spec.recurrent_index(layer);

        // First-token zero-init for the persistent SSM buffers. Flag-
        // gated; subsequent calls are a no-op. This is the only
        // remaining `transfer_to_compute_barrier` issued from inside
        // an SG-3 sub-dispatch — the cmd_fill_buffer happens before
        // any graph-tracked dispatch reads conv_state / ssm_state, so
        // the graph's edge pass doesn't model it.
        ensure_ssm_persistent_initialized(fwd, ctx);

        let conv_state_buf = fwd.conv_state_buf.as_ref().unwrap().handle;
        let qkv_buf       = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
        let conv_input    = fwd.ssm_conv_input_buf.as_ref().unwrap().handle;

        let state_bytes_per_layer = slots as u64 * conv_channels as u64 * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;

        fwd.run_ssm_conv_setup(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_state_buf, state_offset_bytes, state_bytes_per_layer,
            qkv_buf, conv_input,
            conv_channels, "ssm_conv_setup",
        );
    }

    pub(super) fn sub_conv_dispatch(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_conv_dispatch emitted for non-qwen35 config",
        );
        let conv_channels = spec.conv_channels();
        let d_conv = spec.ssm_d_conv;

        let conv_input  = fwd.ssm_conv_input_buf.as_ref().unwrap().handle;
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let conv_weight = layer_weight(ctx.model, layer, "ssm_conv1d.weight");

        fwd.run_ssm_conv(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_input, conv_weight, conv_output,
            d_conv, d_conv, conv_channels, 1, 1, "ssm_conv",
        );
    }

    // ── SsmBetaProj (2) ─────────────────────────────────────────────
    pub(super) fn sub_beta_gemv(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_beta_gemv emitted for non-qwen35 config",
        );
        let hidden_norm = fwd.cur().hidden_norm.handle;
        let beta_buf = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "ssm_beta.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "ssm_beta.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_beta.weight");
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, hidden_norm, beta_buf,
            cfg.hidden_dim, spec.num_v_heads(), scale, "gemv_ssm_beta",
        );
    }

    pub(super) fn sub_beta_sigmoid(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_beta_sigmoid emitted for non-qwen35 config",
        );
        let beta_buf = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        fwd.run_sigmoid(
            ctx.dev, ctx.registry, ctx.cmd,
            beta_buf, spec.num_v_heads(), "ssm_beta_sigmoid",
        );
    }

    // ── GatedDeltaNet (1 Dispatch + 1 Transfer) ─────────────────────
    pub(super) fn sub_gdn_compute(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_gdn_compute emitted for non-qwen35 config",
        );
        let h           = spec.num_v_heads();
        let s_v         = spec.ssm_d_state;
        let head_k      = spec.head_k_dim();
        let num_k_heads = spec.num_k_heads();
        let recurrent_idx = spec.recurrent_index(layer);

        let q     = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
        let k     = fwd.ssm_krep_buf.as_ref().unwrap().handle;
        let v     = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let g     = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let beta  = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let state = fwd.ssm_state_buf.as_ref().unwrap().handle;
        let dst   = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;

        let v_offset_bytes = (head_k as u64) * (num_k_heads as u64) * 2 * 4;
        let v_bytes        = (s_v as u64) * (h as u64) * 4;

        let state_bytes_per_layer = (h as u64) * (s_v as u64) * (s_v as u64) * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;

        let s_off = s_v * h;

        let push = GatedDeltaNetPushConstants {
            h,
            n_tokens: 1,
            n_seqs:   1,
            s_off,
            sq1: head_k, sq2: head_k * h, sq3: head_k * h,
            sv1: s_v,    sv2: s_v * h,    sv3: s_v * h,
            sb1: 1, sb2: h, sb3: h,
            neq1: h,
            rq3:  1,
            scale: 1.0 / (s_v as f32).sqrt(),
            k: 1,
        };

        fwd.run_gated_delta_net(
            ctx.dev, ctx.registry, ctx.cmd,
            q, k,
            v, v_offset_bytes, v_bytes,
            g, beta,
            state, state_offset_bytes, state_bytes_per_layer,
            dst,
            s_v, /* cols_per_wg */ 2,
            &push, "gdn",
        );
    }

    pub(super) fn sub_gdn_state_copy(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "sub_gdn_state_copy emitted for non-qwen35 config",
        );
        let h           = spec.num_v_heads();
        let s_v         = spec.ssm_d_state;
        let recurrent_idx = spec.recurrent_index(layer);

        let state = fwd.ssm_state_buf.as_ref().unwrap().handle;
        let dst   = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;

        let state_bytes_per_layer = (h as u64) * (s_v as u64) * (s_v as u64) * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;
        let s_off = s_v * h;

        // Sprint G-2i — WAW/RAW pre-copy barrier (compute writes
        // followed by transfer writes on `state`). This is a TRANSFER-
        // path stage transition that the graph's COMPUTE-only edge
        // pass doesn't model, so it stays inline here.
        let pre_copy = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre_copy), &[], &[],
            );
        }
        let region = vk::BufferCopy {
            src_offset: (s_off as u64) * 4,
            dst_offset: state_offset_bytes,
            size:       state_bytes_per_layer,
        };
        unsafe {
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, dst, state, std::slice::from_ref(&region),
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);
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

    /// Sprint D2 (v0.4.6) — Qwen3.6 Full-Attention fused Q+Gate
    /// batched projection. One GEMM writes `2 × q_dim` floats per
    /// token into `batch_qgate`; a strided `cmd_copy_buffer` then
    /// extracts the Q half into `batch_q` so downstream
    /// QNormRope/KvWrite/Attention dispatches see the same
    /// per-token-contiguous Q layout as on every other architecture.
    /// Gate stays in `batch_qgate` for `AttnGatedOutput` to consume.
    pub(super) fn b_step_attn_q_gate_proj(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        q_dim: u32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let total_dim = q_dim * 2;
        let qgate = fwd
            .batch_qgate
            .as_ref()
            .expect("AttnQGateProj emitted but batch_qgate not allocated (cfg.qwen35 is None)")
            .handle;
        // Single fused GEMM: 12288 outputs per token, Q in front
        // half, Gate in back half (per-token contiguous).
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_q.weight",
            fwd.batch_norm.handle,
            qgate,
            total_dim, cfg.hidden_dim, seq_len, "gemm_q_gate",
            /* quantize_input = */ true,
        );
        self.b_subscriber_q_barrier(cfg, ctx);

        // Sprint G-2i — BatchExec Q-Gate deinterleave fix.
        //
        // Qwen3.6's GEMM output for `attn_q.weight` lays out per token
        // 12288 floats as INTERLEAVED-per-head:
        //   [ Q_h0 (256), G_h0 (256), Q_h1 (256), G_h1 (256), …, Q_h23, G_h23 ]
        // (same layout discovered in G-2g for the DecodeExec path —
        // verified empirically by VF_FORCE_PER_TOKEN_PREFILL=1 giving
        // coherent " Paris." output where BatchExec gives gibberish).
        //
        // VF's downstream b_step_q_norm_rope / b_step_attention read
        // `batch_q` as CONTIGUOUS [ Q_h0, Q_h1, …, Q_h23 ] per token.
        // The previous extraction (one 6144-float copy per token from
        // offset 0) yielded [ Q_h0, G_h0, Q_h1, G_h1, … ] — every other
        // head was a Gate-as-Q. That was the BatchExec twin of the
        // G-2g DecodeExec bug.
        //
        // Fix: emit `seq_len × n_heads` 256-float copies that pick the
        // Q half of each head's 512-float slice and pack them
        // contiguously into batch_q. Gate stays interleaved in
        // batch_qgate; b_step_attn_gated_output handles the gate via
        // sigmoid_mul's strided form (chunk=head_dim, stride=2×head_dim,
        // gate_offset=head_dim).
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let head_bytes = (head_dim as u64) * 4;
        let stride_bytes_per_head = head_bytes * 2;
        let qgate_stride = (total_dim as u64) * 4;
        let q_bytes_per_token = (q_dim as u64) * 4;
        let mut regions =
            Vec::with_capacity((seq_len as usize) * (n_heads as usize));
        for t in 0..seq_len as u64 {
            for h in 0..n_heads as u64 {
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset(t * qgate_stride + h * stride_bytes_per_head)
                        .dst_offset(t * q_bytes_per_token + h * head_bytes)
                        .size(head_bytes),
                );
            }
        }
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
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, qgate, fwd.batch_q.handle, &regions,
            );
        }
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
        // Sprint G-2i — Qwen3.6 BatchExec also needs split RMSNorm + RoPE
        // and an unconditional `compute_barrier` between them. The fused
        // shader's intra-WG `barrier()` between the rms_norm phase and
        // the rope phase appears insufficient on RADV/gfx1201 for the
        // shape (m=seq_len) BatchExec uses — splitting gives `Paris.`
        // coherence on the standard chat template, fused gives garbage.
        let split = cfg.qwen35.is_some();
        if split {
            // Sprint (de-risk pin): the batched in-place q-norm raced the
            // upstream q-proj GEMM write to batch_q (RAW hazard) — the
            // decode step_q_norm_rope has this barrier (att.rs:448) but the
            // batch counterpart lacked it. head_dim=256 widens the write
            // window enough to manifest (confirmed: barrier → k substep
            // cos 0.99→1.0 vs the per-token decode oracle). Mirrors decode.
            super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
            fwd.run_rms_norm(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_q.handle, wqn, fwd.batch_q.handle,
                head_dim, cfg.n_heads * seq_len,
                cfg.rms_norm_eps, "rms_norm_q_b_split",
            );
            super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
            // run_rope_batch sets ne01=heads_per_token, ne02=seq_len,
            // so the shader reads `rope_data_pos[token_idx]` per (head,
            // token). run_rope_neox_with_pos_offset would set ne02=1
            // and feed every token rope_pos[0], which only works for
            // decode (single token).
            fwd.run_rope_batch(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_q.handle, fwd.batch_q.handle,
                head_dim, rotary_dim, freq_base, theta_scale,
                cfg.n_heads, seq_len,
                "rope_neox_q_b_split",
            );
            super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
        } else {
            fwd.run_rms_norm_mul_rope(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_q.handle, wqn, fwd.batch_q.handle,
                head_dim, rotary_dim, freq_base, theta_scale,
                cfg.n_heads, seq_len,
                cfg.rms_norm_eps, "rms_norm_mul_rope_q_b",
            );
        }
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
        let n_kv = n_kv_heads_for(cfg, ctx.layer);
        let split = cfg.qwen35.is_some();
        if split {
            // De-risk (gated, default-off): see step_k_norm_rope (SKIP_KNORM).
            if std::env::var("VF_DERISK_SKIP_KNORM").as_deref() != Ok("1") {
                // RAW-hazard fix: batched in-place k-norm raced the k-proj
                // GEMM write to batch_k. Mirrors decode (att.rs:448).
                // Confirmed via the substep harness (k cos 0.99→1.0).
                super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
                fwd.run_rms_norm(
                    ctx.dev, ctx.registry, ctx.cmd,
                    fwd.batch_k.handle, wkn, fwd.batch_k.handle,
                    head_dim, n_kv * seq_len,
                    cfg.rms_norm_eps, "rms_norm_k_b_split",
                );
                super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
            }
            // De-risk (gated, default-off): see step_k_norm_rope.
            if std::env::var("VF_DERISK_SKIP_KROPE").as_deref() != Ok("1") {
                fwd.run_rope_batch(
                    ctx.dev, ctx.registry, ctx.cmd,
                    fwd.batch_k.handle, fwd.batch_k.handle,
                    head_dim, rotary_dim, freq_base, theta_scale,
                    n_kv, seq_len,
                    "rope_neox_k_b_split",
                );
            }
        } else {
            fwd.run_rms_norm_mul_rope(
                ctx.dev, ctx.registry, ctx.cmd,
                fwd.batch_k.handle, wkn, fwd.batch_k.handle,
                head_dim, rotary_dim, freq_base, theta_scale,
                n_kv, seq_len,
                cfg.rms_norm_eps, "rms_norm_mul_rope_k_b",
            );
        }
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

    /// Sprint G-2i (v0.4.6, ex-D2) — Qwen3.6 Full-Attention gated output,
    /// batch path. Reads Gate from `batch_qgate` which is **interleaved
    /// per head** (not concat-halves as the pre-G-2i comment claimed):
    /// `[Q_h0(256), G_h0(256), Q_h1(256), G_h1(256), …]` per token.
    /// `sigmoid_mul`'s strided form handles this with chunk = head_dim,
    /// stride = 2 × head_dim, gate_offset = head_dim. The shader's
    /// `token` becomes a flat-head index across the whole batch; the
    /// per-token offset folds into stride-per-head arithmetic
    /// automatically (real_token × n_heads × stride + real_head ×
    /// stride yields the right offset into batch_qgate).
    pub(super) fn b_step_attn_gated_output(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        q_dim: u32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let qgate = fwd
            .batch_qgate
            .as_ref()
            .expect("AttnGatedOutput emitted but batch_qgate not allocated (cfg.qwen35 is None)")
            .handle;
        let ne = seq_len * q_dim;
        let head_dim = cfg.head_dim;
        // Sprint (de-risk pin): pre-read barrier mirroring decode
        // step_attn_gated_output (att.rs:644 maybe_compute_barrier on
        // [attn_out, q_buf]). The batch gated-output had only a TRAILING
        // barrier, so the in-place sigmoid_mul raced the upstream writes
        // to batch_qgate (gate) / batch_attn_out — a non-deterministic
        // RAW hazard (full-output cos varied 0.96–0.99 run-to-run). The
        // global compute_barrier flushes both gate + attn_out before read.
        super::super::arch::compute_barrier(ctx.dev, ctx.cmd);
        // Sprint G-2i — interleaved-per-head gate layout: per (token,
        // head) the gate sits at offset head_dim within a 2 × head_dim
        // slice.
        // De-risk (gated, default-off): see step_attn_gated_output.
        if std::env::var("VF_DERISK_SKIP_GATE").as_deref() != Ok("1") {
            fwd.run_sigmoid_mul(
                ctx.dev, ctx.registry, ctx.cmd,
                qgate, fwd.batch_attn_out.handle,
                ne,           // total elements in attn_out (seq × q_dim)
                head_dim,     // chunk: per-head elements in attn_out
                2 * head_dim, // stride: per-head elements in batch_qgate (Q+G)
                head_dim,     // gate_offset within stride (Gate after Q)
                "sigmoid_mul_gated_out_b",
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
        // Diagnostic toggle (gated, default-true=production): VF_BO_PROJ_NOQUANT=1
        // forces FP activation (quantize_input=false) to measure how much the
        // gemm input-quant contributes to the batch-vs-decode o-proj residual.
        let quant_in = std::env::var("VF_BO_PROJ_NOQUANT").as_deref() != Ok("1");
        self.b_run_proj(
            fwd, cfg, ctx,
            "attn_output.weight",
            fwd.batch_attn_out.handle,
            fwd.batch_o.handle,
            cfg.hidden_dim, q_dim, seq_len, "gemm_o",
            quant_in,
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

    /// Sprint D (v0.4.6) — Qwen3.5/3.6 skeleton passthrough.
    /// Batch path: no-op. `batch_residual` already carries the
    /// layer-input across layers (seeded by `record_prefill_seed`
    /// for layer 0, then in-place updated by each layer's
    /// AttnResidualAdd + FfnResidualAdd). Skipping the attention
    /// sub-block means leaving `batch_residual` untouched — which is
    /// exactly the identity passthrough we want. PreFfnNorm reads
    /// `batch_residual` next, so the residual stream stays
    /// well-defined without any transfer or compute work.
    pub(super) fn b_step_residual_identity_seed(
        &self,
        _fwd: &mut Forward,
        _cfg: &ModelConfig,
        _ctx: &ExecCtx,
    ) {
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

    /// Sprint G-2c (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_conv1d`].
    ///
    /// Recurrent layers MUST process tokens sequentially because
    /// `conv_state` and `ssm_state` update token-by-token (this is the
    /// `keep_rs()` branch in `delta-net-base.cpp:build_conv_state`).
    /// With the per-token scratch sized for `seq=1` (G-2b plan), the
    /// BAT body collapses to the same dispatch sequence as DEC — the
    /// upstream `b_step_attn_qkv_proj` (G-2d) will land per-token-aware
    /// writes into `ssm_qkv_buf` so this body just consumes whatever
    /// slot the executor has already produced. For G-2c the body
    /// matches DEC, exercising the conv pipeline at prefill smoke-test
    /// level without claiming multi-token correctness.
    pub(super) fn b_step_ssm_conv1d(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_conv1d emitted for non-qwen35 config",
        );
        let conv_channels = spec.conv_channels();
        let d_conv = spec.ssm_d_conv;
        let slots = d_conv - 1;
        let recurrent_idx = spec.recurrent_index(layer);

        ensure_ssm_persistent_initialized(fwd, ctx);

        let conv_state_buf = fwd.conv_state_buf.as_ref().unwrap().handle;
        let qkv_buf       = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
        let conv_input    = fwd.ssm_conv_input_buf.as_ref().unwrap().handle;
        let conv_output   = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;

        let state_bytes_per_layer = slots as u64 * conv_channels as u64 * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[qkv_buf]);
        fwd.run_ssm_conv_setup(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_state_buf, state_offset_bytes, state_bytes_per_layer,
            qkv_buf, conv_input,
            conv_channels, "b_ssm_conv_setup",
        );
        fwd.mark_written(&[conv_input, conv_state_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_input]);

        let conv_weight = layer_weight(ctx.model, layer, "ssm_conv1d.weight");
        fwd.run_ssm_conv(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_input, conv_weight, conv_output,
            d_conv, d_conv, conv_channels, 1, 1, "b_ssm_conv",
        );
        fwd.mark_written(&[conv_output]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
    }

    // ──────────────────────────────────────────────────────────
    // Sprint G-2b (v0.4.6) — Qwen3.6 Linear-Attn BAT step stubs.
    // Each one mirrors its DEC sibling above; bodies fill in tandem
    // (G-2c/d/e) so the two paths stay in sync per
    // `feedback_layer_dispatch_paths` (§4.2 same-file co-location).
    // Recurrent layers in qwen35.cpp run per-token in prefill via
    // `keep_rs()` too, so most of these will share the DEC body
    // verbatim once written. Sprint G-2c/d/e decides the exact
    // BAT delta on a per-step basis.
    // ──────────────────────────────────────────────────────────

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_attn_qkv_proj`]. Recurrent layers run
    /// per-token in prefill so the GEMV is dispatched in place of a
    /// batched mul_mm. With G-2b's decode-sized scratch the body
    /// processes token 0 only — multi-token correctness lands when
    /// scratch grows to seq_len.
    pub(super) fn b_step_attn_qkv_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_attn_qkv_proj emitted for non-qwen35 config",
        );
        let input = fwd.batch_norm.handle;
        let qkv_buf = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "attn_qkv.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "attn_qkv.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "attn_qkv.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, input, qkv_buf,
            cfg.hidden_dim, spec.conv_channels(), scale, "b_gemv_ssm_qkv",
        );
        fwd.mark_written(&[qkv_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[qkv_buf]);
    }

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_attn_gate_z_proj`].
    pub(super) fn b_step_attn_gate_z_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_attn_gate_z_proj emitted for non-qwen35 config",
        );
        let input = fwd.batch_norm.handle;
        let z_buf = fwd.ssm_z_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "attn_gate.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "attn_gate.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "attn_gate.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, input, z_buf,
            cfg.hidden_dim, spec.ssm_d_inner, scale, "b_gemv_ssm_z",
        );
        fwd.mark_written(&[z_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[z_buf]);
    }

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_beta_proj`].
    pub(super) fn b_step_ssm_beta_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_beta_proj emitted for non-qwen35 config",
        );
        let input = fwd.batch_norm.handle;
        let beta_buf = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "ssm_beta.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "ssm_beta.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_beta.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, input, beta_buf,
            cfg.hidden_dim, spec.num_v_heads(), scale, "b_gemv_ssm_beta",
        );
        fwd.mark_written(&[beta_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[beta_buf]);
        fwd.run_sigmoid(
            ctx.dev, ctx.registry, ctx.cmd,
            beta_buf, spec.num_v_heads(), "b_ssm_beta_sigmoid",
        );
        fwd.mark_written(&[beta_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[beta_buf]);
    }

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_alpha_gate`].
    pub(super) fn b_step_ssm_alpha_gate(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_alpha_gate emitted for non-qwen35 config",
        );
        let input = fwd.batch_norm.handle;
        let alpha_buf = fwd.ssm_alpha_buf.as_ref().unwrap().handle;
        let gate_buf  = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let n_heads   = spec.num_v_heads();

        let (w_alpha, w_alpha_off, w_alpha_sz) =
            layer_weight_with_offset(ctx.model, layer, "ssm_alpha.weight");
        let s_alpha = layer_weight_shader(
            ctx.model, layer, "ssm_alpha.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_alpha.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[input]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s_alpha, w_alpha,
            w_alpha_off, w_alpha_sz, input, alpha_buf,
            cfg.hidden_dim, n_heads, scale, "b_gemv_ssm_alpha",
        );
        fwd.mark_written(&[alpha_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        let dt_bias = layer_weight(ctx.model, layer, "ssm_dt.bias");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            alpha_buf, dt_bias, alpha_buf,
            n_heads, "b_ssm_alpha_add_dt",
        );
        fwd.mark_written(&[alpha_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        fwd.run_softplus(
            ctx.dev, ctx.registry, ctx.cmd,
            alpha_buf, n_heads, "b_ssm_alpha_softplus",
        );
        fwd.mark_written(&[alpha_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[alpha_buf]);

        let ssm_a = layer_weight(ctx.model, layer, "ssm_a");
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            alpha_buf, ssm_a, gate_buf,
            n_heads, "b_ssm_alpha_mul_a",
        );
        fwd.mark_written(&[gate_buf]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);
    }

    /// Sprint G-2c (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_silu`]. Pointwise, so identical to DEC.
    pub(super) fn b_step_ssm_silu(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_silu emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        fwd.run_silu(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            spec.conv_channels(), "b_ssm_silu",
        );
        fwd.mark_written(&[conv_output]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
    }

    /// Sprint G-2c (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_qk_l2_norm`]. Per-row reduction, so
    /// identical to DEC for `seq=1`-sized scratch (G-2b).
    pub(super) fn b_step_ssm_qk_l2_norm(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_qk_l2_norm emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let head_dim = spec.head_k_dim();
        let n_heads  = spec.num_k_heads();
        let qk_floats = head_dim * n_heads;
        let eps = cfg.rms_norm_eps;

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        fwd.run_l2_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            head_dim, n_heads, 0,
            eps, "b_ssm_l2_norm_q",
        );
        fwd.run_l2_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, conv_output,
            head_dim, n_heads, qk_floats,
            eps, "b_ssm_l2_norm_k",
        );
        fwd.mark_written(&[conv_output]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
    }

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_repeat_qk`].
    pub(super) fn b_step_ssm_repeat_qk(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, _layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_repeat_qk emitted for non-qwen35 config",
        );
        let conv_output = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let qrep = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
        let krep = fwd.ssm_krep_buf.as_ref().unwrap().handle;
        let head_dim    = spec.head_k_dim();
        let n_src_heads = spec.num_k_heads();
        let n_dst_heads = spec.num_v_heads();
        let slice_bytes = head_dim as u64 * n_src_heads as u64 * 4;

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[conv_output]);
        fwd.run_repeat_interleave(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, 0, slice_bytes, qrep,
            head_dim, n_src_heads, n_dst_heads, 1, "b_ssm_repeat_q",
        );
        fwd.run_repeat_interleave(
            ctx.dev, ctx.registry, ctx.cmd,
            conv_output, slice_bytes, slice_bytes, krep,
            head_dim, n_src_heads, n_dst_heads, 1, "b_ssm_repeat_k",
        );
        fwd.mark_written(&[qrep, krep]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[qrep, krep]);
    }

    /// Sprint G-2e (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_gated_delta_net`]. Recurrent state updates
    /// per-token; with decode-sized scratch the BAT body matches DEC
    /// for token 0 only (multi-token correctness lands when scratch
    /// grows to seq_len — out of scope for G-2e).
    pub(super) fn b_step_gated_delta_net(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_gated_delta_net emitted for non-qwen35 config",
        );
        let h           = spec.num_v_heads();
        let s_v         = spec.ssm_d_state;
        let head_k      = spec.head_k_dim();
        let num_k_heads = spec.num_k_heads();
        let recurrent_idx = spec.recurrent_index(layer);

        let q     = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
        let k     = fwd.ssm_krep_buf.as_ref().unwrap().handle;
        let v     = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
        let g     = fwd.ssm_gate_buf.as_ref().unwrap().handle;
        let beta  = fwd.ssm_beta_buf.as_ref().unwrap().handle;
        let state = fwd.ssm_state_buf.as_ref().unwrap().handle;
        let dst   = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;

        let v_offset_bytes = (head_k as u64) * (num_k_heads as u64) * 2 * 4;
        let v_bytes        = (s_v as u64) * (h as u64) * 4;
        let state_bytes_per_layer = (h as u64) * (s_v as u64) * (s_v as u64) * 4;
        let state_offset_bytes    = recurrent_idx as u64 * state_bytes_per_layer;
        let s_off = s_v * h;

        let push = GatedDeltaNetPushConstants {
            h, n_tokens: 1, n_seqs: 1, s_off,
            sq1: head_k, sq2: head_k * h, sq3: head_k * h,
            sv1: s_v,    sv2: s_v * h,    sv3: s_v * h,
            sb1: 1, sb2: h, sb3: h,
            neq1: h, rq3: 1,
            scale: 1.0 / (s_v as f32).sqrt(),
            k: 1,
        };

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[q, k, v, g, beta]);
        fwd.run_gated_delta_net(
            ctx.dev, ctx.registry, ctx.cmd,
            q, k,
            v, v_offset_bytes, v_bytes,
            g, beta,
            state, state_offset_bytes, state_bytes_per_layer,
            dst,
            s_v, 2,
            &push, "b_gdn",
        );
        fwd.mark_written(&[dst]);

        compute_to_transfer_barrier(ctx.dev, ctx.cmd);
        let region = vk::BufferCopy {
            src_offset: (s_off as u64) * 4,
            dst_offset: state_offset_bytes,
            size:       state_bytes_per_layer,
        };
        unsafe {
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, dst, state, std::slice::from_ref(&region),
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);
        fwd.mark_written(&[state]);
    }

    /// Sprint G-2e (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_norm_gated`].
    pub(super) fn b_step_norm_gated(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_norm_gated emitted for non-qwen35 config",
        );
        let h   = spec.num_v_heads();
        let s_v = spec.ssm_d_state;
        let ne  = h * s_v;

        let gdn_out  = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;
        let z        = fwd.ssm_z_buf.as_ref().unwrap().handle;
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        let norm_w   = layer_weight(ctx.model, layer, "ssm_norm.weight");

        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gdn_out]);
        fwd.run_rms_norm(
            ctx.dev, ctx.registry, ctx.cmd,
            gdn_out, norm_w, norm_out,
            s_v, h, cfg.rms_norm_eps, "b_ssm_norm_gated_rms",
        );
        fwd.mark_written(&[norm_out]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[z]);
        fwd.run_silu(
            ctx.dev, ctx.registry, ctx.cmd,
            z, z, ne, "b_ssm_z_silu",
        );
        fwd.mark_written(&[z]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[norm_out, z]);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
            norm_out, z, norm_out,
            ne, "b_ssm_norm_gated_mul",
        );
        fwd.mark_written(&[norm_out]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[norm_out]);
    }

    /// Sprint G-2d (v0.4.6) — batch counterpart of
    /// [`DecodeExec::step_ssm_out_proj`]. Writes only token 0's worth
    /// into `batch_o[0..hidden_dim]`; subsequent residual-add on
    /// tokens 1..N consumes stale/zero data (smoke-only correctness,
    /// matches the G-2b plan's decode-sized scratch).
    pub(super) fn b_step_ssm_out_proj(
        &self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx, layer: u32,
    ) {
        let spec = cfg.qwen35.as_ref().expect(
            "b_step_ssm_out_proj emitted for non-qwen35 config",
        );
        let norm_out = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
        let batch_o  = fwd.batch_o.handle;
        let (w, w_off, w_sz) = layer_weight_with_offset(ctx.model, layer, "ssm_out.weight");
        let s = layer_weight_shader(
            ctx.model, layer, "ssm_out.weight",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let scale = layer_weight_scale_scalar(ctx.model, layer, "ssm_out.weight");
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[norm_out]);
        fwd.run_gemv(
            ctx.dev, ctx.registry, ctx.cmd, s, w, w_off, w_sz, norm_out, batch_o,
            spec.ssm_d_inner, cfg.hidden_dim, scale, "b_gemv_ssm_out",
        );
        fwd.mark_written(&[batch_o]);
        fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[batch_o]);
    }
}
