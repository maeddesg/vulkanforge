//! Sprint 57B — Mixture-of-Experts pipeline steps (DEC + BAT pendants).
//!
//! 6 DEC step_* + 6 BAT b_step_* covering: PostDenseMlpNorm, PreMoeNorm,
//! PostMoeNorm, MoeBranchAdd, MoeRoute, MoeExpertFfn. The Gemma-4-26B-A4B
//! MoE block — never reached on E2B / Qwen3 / Llama / other dense models
//! (the layer-plan builder gates these on `enable_moe_block`).
//!
//! See `executor/mod.rs` for shared types + helpers and
//! `cpu_moe_route` (the CPU-router fallback used when GPU router is
//! disabled).

use super::{
    batch_seq_len, compute_to_transfer_barrier, cpu_moe_route,
    log_router_decision, transfer_to_host_barrier, BatchExec, DecodeExec, ExecCtx,
    MOE_LAYER0_LOGGED,
};
use super::super::arch::{
    compute_barrier, layer_weight, layer_weight_shader, transfer_to_compute_barrier,
};
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;
use super::super::super::pipeline::SwigluPushConstants;
use super::super::super::shaders::ShaderId;

use ash::vk;

impl DecodeExec {
    // === Sprint 51D-B Block 1 — Gemma-4-26B-A4B MoE FFN-Block norms + add ===

    /// `post_feedforward_layernorm_1` on Dense-MLP output.
    /// Reads `ffn_out` (DownProj output), writes `scratch_a` (= h1).
    pub(super) fn step_post_dense_mlp_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
    pub(super) fn step_pre_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
    pub(super) fn step_post_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
    pub(super) fn step_moe_branch_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
    pub(super) fn step_moe_route(
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
    pub(super) fn step_moe_expert_ffn(
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

impl BatchExec {
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

    pub(super) fn b_step_post_dense_mlp_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_pre_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_post_moe_norm(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_moe_branch_add(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
        let seq_len = batch_seq_len(ctx);
        fwd.run_binary(
            ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
            fwd.batch_ffn_out.handle, fwd.batch_norm.handle, fwd.batch_ffn_out.handle,
            seq_len * cfg.hidden_dim, "moe_branch_add_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);
    }

    // === Sprint 51D-D — Batch (prefill) MoE routing + expert FFN ===

    pub(super) fn b_step_moe_route(
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

    pub(super) fn b_step_moe_expert_ffn(
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
}
