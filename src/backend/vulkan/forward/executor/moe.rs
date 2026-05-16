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
    compute_barrier, layer_weight, layer_weight_indexed_shader, layer_weight_mmq_id_shader,
    layer_weight_shader, transfer_to_compute_barrier,
};
use super::super::state::Forward;
use super::super::super::gguf::ModelConfig;
use super::super::super::pipeline::SwigluPushConstants;
use super::super::super::shaders::ShaderId;

use ash::vk;

use super::super::gpu_direct_moe_enabled;

/// Sprint 61C — Phase 2' env-gate. `VF_MOE_GROUPED=1` activates the
/// MMQ_ID Expert-Grouped batched dispatch in `b_step_moe_expert_ffn`;
/// any other value (including unset) keeps the GPU-direct GEMV slot
/// loop from Sprint 56C-3 as the default. Cached after first read.
pub(crate) fn moe_grouped_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("VF_MOE_GROUPED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Sprint 61C — Subgroup-ballot vs scalar IDS-scan toggle for the
/// MMQ_ID kernels. `VF_MOE_GROUPED_SUBGROUP=0` falls back to the
/// stock variant; default ON since the subgroup build proved to load
/// cleanly on Mesa 26.1-rc3 / gfx1201 in Sprint 61B.
pub(crate) fn moe_grouped_subgroup_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("VF_MOE_GROUPED_SUBGROUP")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

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

            // Sprint 56C-2 — GPU-direct path: skip the readback. Emit a
            // COMPUTE→COMPUTE barrier so the indexed GEMV in
            // step_moe_expert_ffn sees the router's writes to
            // indices_scratch + weights_scratch. `fwd.moe_routing` is
            // intentionally left None — the expert FFN reads the values
            // from GPU buffers via `data_ids[]` / `weights[slot]` SSBOs.
            if gpu_direct_moe_enabled() {
                compute_barrier(ctx.dev, ctx.cmd);
                return;
            }

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
        n_experts: u32, top_k: u32, moe_intermediate: u32,
    ) {
        // Sprint 56C-2 — GPU-direct branch (env-gated, requires GPU router).
        // Replaces the CPU-readback slot loop with indexed GEMV + indexed
        // FMA dispatches that read `expert_id = data_ids[slot]` and
        // `weight = weights[slot]` from the router's GPU buffers.
        if gpu_direct_moe_enabled() && fwd.moe_router_gpu.is_some() {
            self.step_moe_expert_ffn_gpu_direct(fwd, cfg, ctx, n_experts, top_k, moe_intermediate);
            return;
        }
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

    /// Sprint 56C-2 — GPU-direct DEC expert FFN. Mirrors
    /// `step_moe_expert_ffn` but the per-slot `expert_id` and `weight`
    /// are read from the router's GPU buffers (`indices_scratch`,
    /// `weights_scratch`) via the `MUL_MAT_ID` indexed-GEMV +
    /// indexed-FMA-Add pipelines. No CPU readback, no
    /// `mid_frame_submit_and_wait`.
    fn step_moe_expert_ffn_gpu_direct(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        n_experts: u32,
        top_k: u32,
        moe_intermediate: u32,
    ) {
        let scratch_b  = fwd.cur().scratch_b.handle;
        let gate_buf   = fwd.cur().gate_buf.handle;
        let up_buf     = fwd.cur().up_buf.handle;
        let o_buf      = fwd.cur().o_buf.handle;
        let ffn_hidden = fwd.cur().ffn_hidden.handle;
        let h          = cfg.hidden_dim;
        let mi         = moe_intermediate;
        let mi_padded: u32 = ((mi + 255) / 256) * 256;

        let gate_up_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.gate_up_proj", ctx.layer))
            .expect("moe_experts.gate_up_proj missing");
        let down_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
            .expect("moe_experts.down_proj missing");

        // Elements per expert: bytes_per_expert × block_size / type_size.
        // The MUL_MAT_ID shader computes `a_offset = expert_id *
        // (batch_stride_a / QUANT_K)` so `batch_stride_a` is in
        // **elements** and the per-quant block size divides correctly.
        let gate_up_block_size = gate_up_t.ggml_type.block_size() as u32;
        let gate_up_type_size  = gate_up_t.ggml_type.type_size() as u32;
        let gate_up_bytes_per_expert = (gate_up_t.byte_size / (n_experts as u64)) as u32;
        let gate_up_elems_per_expert =
            gate_up_bytes_per_expert / gate_up_type_size * gate_up_block_size;

        let down_block_size = down_t.ggml_type.block_size() as u32;
        let down_type_size  = down_t.ggml_type.type_size() as u32;
        let down_bytes_per_expert = (down_t.byte_size / (n_experts as u64)) as u32;
        let down_elems_per_expert =
            down_bytes_per_expert / down_type_size * down_block_size;

        // Sprint 52P loader-aware K (mirror of CPU-readback path).
        let down_k = if down_t.shape[0] as u32 == n_experts {
            mi_padded
        } else {
            down_t.shape[0] as u32
        };

        let gate_up_shader = layer_weight_indexed_shader(
            ctx.model, ctx.layer,
            "moe_experts.gate_up_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let down_shader = layer_weight_indexed_shader(
            ctx.model, ctx.layer,
            "moe_experts.down_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let gate_up_w = layer_weight(ctx.model, ctx.layer, "moe_experts.gate_up_proj");
        let down_w    = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");

        let router = fwd.moe_router_gpu.as_ref()
            .expect("step_moe_expert_ffn_gpu_direct requires moe_router_gpu");
        let indices_buf = router.indices_scratch.handle;
        let weights_buf = router.weights_scratch.handle;

        // Zero the accumulator before the per-slot loop, same as the
        // CPU-readback path.
        unsafe {
            ctx.dev.device.cmd_fill_buffer(
                ctx.cmd, ffn_hidden, 0, (h as u64) * 4, 0,
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        let h_bytes  = (h as u64) * 4;
        let mi_bytes = (mi as u64) * 4;

        for slot in 0..top_k {
            // (a) gate_up GEMV (indexed): scratch_b × experts.gate_up_proj[indices[slot]] → gate_buf.
            fwd.run_gemv_indexed_at_offset(
                ctx.dev, ctx.registry, ctx.cmd,
                gate_up_w,
                scratch_b, 0, h_bytes,
                gate_buf, 0, 2 * mi_bytes,
                indices_buf,
                h, 2 * mi,
                gate_up_elems_per_expert,
                slot,
                "moe_gate_up_id",
                gate_up_shader,
            );
            fwd.mark_written(&[gate_buf]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

            // (b) GeluPytorchTanhGlu (unchanged from CPU-readback path).
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
            fwd.profile("moe_glu_id", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
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

            // (c) down GEMV (indexed): up_buf × experts.down_proj[indices[slot]] → o_buf.
            fwd.run_gemv_indexed_at_offset(
                ctx.dev, ctx.registry, ctx.cmd,
                down_w,
                up_buf, 0, (down_k as u64) * 4,
                o_buf, 0, h_bytes,
                indices_buf,
                down_k, h,
                down_elems_per_expert,
                slot,
                "moe_down_id",
                down_shader,
            );
            fwd.mark_written(&[o_buf]);
            fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

            // (d) Indexed FMA: ffn_hidden += weights[slot] * o_buf.
            fwd.run_fma_add_indexed(
                ctx.dev, ctx.registry, ctx.cmd,
                o_buf, 0,
                ffn_hidden, 0,
                weights_buf,
                h, slot,
                "moe_fma_add_id",
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

            // Sprint 56C-2 — GPU-direct path: skip the readback. Emit a
            // COMPUTE→COMPUTE barrier so b_step_moe_expert_ffn's indexed
            // GEMVs see the router's writes. `fwd.moe_routing_batch`
            // stays None — the BAT expert FFN reads from GPU buffers
            // via flat slot index `t * top_k + k`.
            if gpu_direct_moe_enabled() {
                compute_barrier(ctx.dev, ctx.cmd);
                return;
            }

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
        n_experts: u32, top_k: u32, moe_intermediate: u32,
    ) {
        // Sprint 61C — Phase 2' Expert-Grouped MMQ_ID branch.
        // Requires the GPU router so we can read indices_scratch /
        // weights_scratch for the CPU counting-sort. Skipped if the
        // env-gate is off (default).
        if moe_grouped_enabled() && fwd.moe_router_gpu.is_some() {
            self.b_step_moe_expert_ffn_grouped(
                fwd, cfg, ctx, n_experts, top_k, moe_intermediate,
            );
            return;
        }
        // Sprint 56C-2 — GPU-direct BAT branch.
        if gpu_direct_moe_enabled() && fwd.moe_router_gpu.is_some() {
            self.b_step_moe_expert_ffn_gpu_direct(
                fwd, cfg, ctx, n_experts, top_k, moe_intermediate,
            );
            return;
        }
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

    /// Sprint 56C-2 — GPU-direct BAT expert FFN. Per-(token, slot) loop
    /// where each dispatch reads its `expert_id` and `weight` from the
    /// router's GPU buffers via `data_ids[t*top_k + k]` /
    /// `weights[t*top_k + k]`. Mirrors the CPU-readback `b_step_moe_expert_ffn`
    /// structure but skips the host iteration of routing tuples.
    fn b_step_moe_expert_ffn_gpu_direct(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        n_experts: u32,
        top_k: u32,
        moe_intermediate: u32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let h  = cfg.hidden_dim;
        let mi = moe_intermediate;
        let h_bytes  = (h as u64) * 4;
        let mi_bytes = (mi as u64) * 4;
        let mi_padded: u32 = ((mi + 255) / 256) * 256;

        let gate_up_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.gate_up_proj", ctx.layer))
            .expect("moe_experts.gate_up_proj missing");
        let down_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
            .expect("moe_experts.down_proj missing");
        let gate_up_block_size = gate_up_t.ggml_type.block_size() as u32;
        let gate_up_type_size  = gate_up_t.ggml_type.type_size() as u32;
        let gate_up_bytes_per_expert = (gate_up_t.byte_size / (n_experts as u64)) as u32;
        let gate_up_elems_per_expert =
            gate_up_bytes_per_expert / gate_up_type_size * gate_up_block_size;

        let down_block_size = down_t.ggml_type.block_size() as u32;
        let down_type_size  = down_t.ggml_type.type_size() as u32;
        let down_bytes_per_expert = (down_t.byte_size / (n_experts as u64)) as u32;
        let down_elems_per_expert =
            down_bytes_per_expert / down_type_size * down_block_size;

        let down_k = if down_t.shape[0] as u32 == n_experts {
            mi_padded
        } else {
            down_t.shape[0] as u32
        };

        let gate_up_shader = layer_weight_indexed_shader(
            ctx.model, ctx.layer,
            "moe_experts.gate_up_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let down_shader = layer_weight_indexed_shader(
            ctx.model, ctx.layer,
            "moe_experts.down_proj",
            fwd.mul_mat_vec_subgroup_enabled,
        );
        let gate_up_w = layer_weight(ctx.model, ctx.layer, "moe_experts.gate_up_proj");
        let down_w    = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");

        let router = fwd.moe_router_gpu.as_ref()
            .expect("b_step_moe_expert_ffn_gpu_direct requires moe_router_gpu");
        let indices_buf = router.indices_scratch.handle;
        let weights_buf = router.weights_scratch.handle;

        let batch_in  = fwd.batch_o.handle;
        let batch_out = fwd.batch_ffn_hidden.handle;

        // Zero the full batch_ffn_hidden output slab once.
        unsafe {
            ctx.dev.device.cmd_fill_buffer(
                ctx.cmd, batch_out, 0, (seq_len as u64) * h_bytes, 0,
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        let gate_buf = fwd.cur().gate_buf.handle;
        let up_buf   = fwd.cur().up_buf.handle;
        let o_buf    = fwd.cur().o_buf.handle;
        for t in 0..(seq_len as usize) {
            let in_off  = (t as u64) * h_bytes;
            let out_off = (t as u64) * h_bytes;
            for k in 0..top_k {
                let flat_slot = (t as u32) * top_k + k;

                // (a) gate_up indexed GEMV.
                fwd.run_gemv_indexed_at_offset(
                    ctx.dev, ctx.registry, ctx.cmd,
                    gate_up_w,
                    batch_in, in_off, h_bytes,
                    gate_buf, 0, 2 * mi_bytes,
                    indices_buf,
                    h, 2 * mi,
                    gate_up_elems_per_expert,
                    flat_slot,
                    "moe_gate_up_id_b",
                    gate_up_shader,
                );
                fwd.mark_written(&[gate_buf]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[gate_buf]);

                // (b) GLU.
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
                fwd.profile("moe_glu_id_b", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
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

                // (c) down indexed GEMV.
                fwd.run_gemv_indexed_at_offset(
                    ctx.dev, ctx.registry, ctx.cmd,
                    down_w,
                    up_buf, 0, (down_k as u64) * 4,
                    o_buf, 0, h_bytes,
                    indices_buf,
                    down_k, h,
                    down_elems_per_expert,
                    flat_slot,
                    "moe_down_id_b",
                    down_shader,
                );
                fwd.mark_written(&[o_buf]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

                // (d) Indexed FMA into per-token slot of batch_out.
                fwd.run_fma_add_indexed(
                    ctx.dev, ctx.registry, ctx.cmd,
                    o_buf, 0,
                    batch_out, out_off,
                    weights_buf,
                    h, flat_slot,
                    "moe_fma_add_id_b",
                );
                fwd.mark_written(&[batch_out]);
                fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[batch_out]);
            }
        }
        // Final barrier mirroring the CPU-readback path.
        compute_barrier(ctx.dev, ctx.cmd);
    }

    /// Sprint 61C — Phase 2' Expert-Grouped Dispatch (env-gated).
    ///
    /// Replaces the per-(token × slot) indexed-GEMV slot loop from
    /// Sprint 56C with two batched `mul_mmq` MUL_MAT_ID GEMMs (gate_up
    /// + down). Tokens are grouped by expert via a tiny CPU counting-
    /// sort run from a readback of `indices_scratch` / `weights_scratch`.
    /// GLU and FMA-add stay per-slot for the pragmatic first land (no
    /// new shaders); the win is the 2× gate_up + 2× down GEMV-per-
    /// (token,slot) → 1× MMQ_ID per kernel collapse plus the 5×
    /// arithmetic intensity from M=`active_tokens_per_expert` row-batched
    /// per-expert weights fetch.
    ///
    /// Dispatch count per layer (seq=25, top_k=8, GGUF Q3_K_M):
    /// - Current (56C indexed-GEMV): 25 × 8 × 4 = 800
    /// - This path: 1 (q8_1) + 1 (gate_up MMQ_ID) + 25×8 (GLU)
    ///   + 1 (q8_1) + 1 (down MMQ_ID) + 25×8 (FMA add) = 404
    /// → ~2× CB-record overhead saving plus weight-fetch consolidation.
    ///
    /// The `mid_frame_submit_and_wait` to drain `indices_scratch` for
    /// CPU sort is the trade-off; Sprint 61A measured the previous
    /// per-layer drain at ~26 ms each (25 layers = 647 ms wall), but
    /// the readback here is *only* `seq_len × top_k × 8 B` (≈1.6 KB)
    /// vs the 51C-era `seq_len × hidden × 4 B` (~290 KB) so the drain
    /// is dominated by the submit-and-wait overhead, not the copy.
    fn b_step_moe_expert_ffn_grouped(
        &self,
        fwd: &mut Forward,
        cfg: &ModelConfig,
        ctx: &ExecCtx,
        n_experts: u32,
        top_k: u32,
        moe_intermediate: u32,
    ) {
        let seq_len = batch_seq_len(ctx);
        let h = cfg.hidden_dim;
        let mi = moe_intermediate;
        let h_bytes = (h as u64) * 4;
        let mi_bytes = (mi as u64) * 4;
        let two_mi_bytes = 2 * mi_bytes;

        // -------- 1. Snapshot all buffer handles --------
        // (Sprint 56B pattern: extract handles once so subsequent
        // &mut self calls don't fight the borrow checker.)
        let (
            router_indices,
            router_weights,
            router_readback_staging,
            grouped_data_ids,
            grouped_data_counts,
            grouped_input_q8,
            grouped_gate_up_out,
            grouped_glu_out,
            grouped_glu_q8,
            grouped_down_out,
        ) = {
            let router = fwd.moe_router_gpu.as_ref()
                .expect("Phase 2' grouped path requires moe_router_gpu");
            (
                router.indices_scratch.handle,
                router.weights_scratch.handle,
                router.readback_staging.handle,
                router.grouped_data_ids.handle,
                router.grouped_data_counts.handle,
                router.grouped_input_q8.handle,
                router.grouped_gate_up_out.handle,
                router.grouped_glu_out.handle,
                router.grouped_glu_q8.handle,
                router.grouped_down_out.handle,
            )
        };
        let batch_o_h = fwd.batch_o.handle;
        let batch_out = fwd.batch_ffn_hidden.handle;

        // -------- 2. Readback router indices+weights for grouping --------
        // The Sprint 56C GPU-direct b_step_moe_route emits a
        // compute_barrier and returns when gpu_direct_moe_enabled
        // — so indices/weights are GPU-resident here. Issue a small
        // (seq_len × top_k × 8 B) copy → submit → wait → CPU sort.
        let topk_bytes = (top_k as u64) * 4;
        let all_topk_bytes = (seq_len as u64) * topk_bytes;
        let bar_pre = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            ctx.dev.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&bar_pre), &[], &[],
            );
            let cp_idx = vk::BufferCopy {
                src_offset: 0, dst_offset: 0, size: all_topk_bytes,
            };
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, router_indices, router_readback_staging,
                std::slice::from_ref(&cp_idx),
            );
            let cp_w = vk::BufferCopy {
                src_offset: 0, dst_offset: all_topk_bytes, size: all_topk_bytes,
            };
            ctx.dev.device.cmd_copy_buffer(
                ctx.cmd, router_weights, router_readback_staging,
                std::slice::from_ref(&cp_w),
            );
        }
        transfer_to_host_barrier(ctx.dev, ctx.cmd);
        fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)
            .expect("mid_frame_submit_and_wait failed (grouped readback)");

        // -------- 3. CPU read of router indices/weights + count tally --------
        // CRITICAL: mul_mmq.comp's MUL_MAT_ID branch indexes IDS as
        // `data_ids[ii1 * nbi1 + ii0] = EXPERT_ID for token=ii1 slot=ii0`
        // (the *raw* router layout, not a sorted-by-expert order). The
        // shader scans IDS per workgroup-Z (expert) and gathers matching
        // (ii0, ii1) into shared `row_ids[]`. So we upload `idx_slice`
        // unchanged — NO counting-sort. Only `data_counts[expert]` is
        // computed CPU-side (used by the workgroup early-exit). FMA-add
        // also iterates (token, slot) in original order; output cell
        // `[t × top_k + s]` in `grouped_down_out` already lands at the
        // right place because the shader's D-write uses the *same*
        // (row_idx.y=token, row_idx.x=slot) it gathered from IDS.
        let (counts_vec, indices_host, weights_host) = {
            let raw = fwd.moe_router_gpu.as_ref().unwrap()
                .readback_staging.read_bytes()
                .expect("readback_staging not host-visible");
            let total_topk = (seq_len as usize) * (top_k as usize);
            let idx_slice: &[u32] = bytemuck::cast_slice(&raw[..(total_topk * 4)]);
            let w_slice: &[f32] = bytemuck::cast_slice(
                &raw[(total_topk * 4)..(total_topk * 8)],
            );
            if ctx.layer == 0
                && !MOE_LAYER0_LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                let routing0: Vec<(u32, f32)> = (0..(top_k as usize))
                    .map(|k| (idx_slice[k], w_slice[k])).collect();
                log_router_decision(0, &routing0);
            }
            let mut counts = vec![0u32; n_experts as usize];
            for &e in idx_slice {
                counts[e as usize] += 1;
            }
            (counts, idx_slice.to_vec(), w_slice.to_vec())
        };

        // -------- 4. Upload indices + counts to GPU SSBOs --------
        // cmd_update_buffer caps at 64 KB; for routine prefill batches
        // it's 25×8×4 = 800 B for data_ids, 128×4 = 512 B for counts.
        unsafe {
            ctx.dev.device.cmd_update_buffer(
                ctx.cmd, grouped_data_ids, 0,
                bytemuck::cast_slice(&indices_host),
            );
            ctx.dev.device.cmd_update_buffer(
                ctx.cmd, grouped_data_counts, 0,
                bytemuck::cast_slice(&counts_vec),
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        // -------- 5. Q8_1 quantize batch_o → grouped_input_q8 --------
        fwd.run_quantize_q8_1(
            ctx.dev, ctx.registry, ctx.cmd,
            batch_o_h, grouped_input_q8,
            seq_len * h,
            "moe_q8_input_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);

        // -------- 6. gate_up MMQ_ID (1 dispatch over all experts) --------
        let gate_up_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.gate_up_proj", ctx.layer))
            .expect("moe_experts.gate_up_proj missing");
        let gate_up_block_size = gate_up_t.ggml_type.block_size() as u32;
        let gate_up_type_size = gate_up_t.ggml_type.type_size() as u32;
        let gate_up_bytes_per_expert = (gate_up_t.byte_size / (n_experts as u64)) as u32;
        let gate_up_elems_per_expert =
            gate_up_bytes_per_expert / gate_up_type_size * gate_up_block_size;
        let subgroup = moe_grouped_subgroup_enabled();
        let gate_up_shader = layer_weight_mmq_id_shader(
            ctx.model, ctx.layer, "moe_experts.gate_up_proj", subgroup,
        );
        let gate_up_w = layer_weight(ctx.model, ctx.layer, "moe_experts.gate_up_proj");
        fwd.run_mmq_id_grouped(
            ctx.dev, ctx.registry, ctx.cmd,
            gate_up_shader,
            gate_up_w,
            grouped_input_q8,
            grouped_gate_up_out,
            grouped_data_ids,
            grouped_data_counts,
            2 * mi, // M
            h,      // K
            seq_len, top_k, n_experts,
            gate_up_elems_per_expert,
            h,      // stride_b (harmless with ne11=1)
            h,      // batch_stride_b (per-token K-stride in [seq_len × hidden] B)
            1,      // ne11=1: slot doesn't pick within-token B slice
            "moe_gate_up_grouped",
        );
        compute_barrier(ctx.dev, ctx.cmd);

        // -------- 7. Per-(token, slot) GLU --------
        // gate_up_out layout: [seq_len × top_k × 2*mi], so slot offset
        // is `(t * top_k + s) * 2*mi`. Gate occupies the first mi
        // elements of each slot block; up occupies the second mi.
        // GLU writes mi floats per slot into grouped_glu_out at
        // `(t * top_k + s) * mi`.
        for t in 0..(seq_len as u64) {
            for k in 0..(top_k as u64) {
                let slot_flat = t * (top_k as u64) + k;
                let gate_off = slot_flat * two_mi_bytes;
                let up_off = gate_off + mi_bytes;
                let out_off = slot_flat * mi_bytes;
                let kernel = ctx.registry.get(ShaderId::GeluPytorchTanhGlu);
                let set = fwd.alloc_or_get_set(
                    ctx.dev, kernel.descriptor_set_layout,
                    &[
                        (0, grouped_gate_up_out, gate_off, mi_bytes),
                        (1, grouped_gate_up_out, up_off,   mi_bytes),
                        (2, grouped_glu_out,     out_off,  mi_bytes),
                    ],
                );
                let pc = SwigluPushConstants { n: mi };
                let dispatch_x = (mi + 255) / 256;
                let layout = kernel.pipeline_layout;
                let pipeline = kernel.pipeline;
                fwd.profile("moe_glu_grouped", ctx.dev, ctx.cmd, |dev, cmd| unsafe {
                    dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                    );
                    dev.device.cmd_push_constants(
                        cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                    );
                    dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
                });
            }
        }
        compute_barrier(ctx.dev, ctx.cmd);

        // -------- Sprint 61C-2 bisect option: per-slot down GEMV --------
        // `VF_MOE_GROUPED_LEGACY_DOWN=1` swaps the grouped down MMQ_ID
        // (+ scatter-FMA loop) for a per-slot indexed-GEMV down + FMA,
        // reading the GLU output from `grouped_glu_out` at the per-slot
        // offset. If this branch produces "Paris.", the bug lies in the
        // grouped down MMQ_ID dispatch (Suspect #1 in 61C report); if
        // still wrong, the bug is upstream (gate_up MMQ_ID or GLU).
        if std::env::var("VF_MOE_GROUPED_LEGACY_DOWN")
            .map(|v| v == "1") .unwrap_or(false)
        {
            let down_t = ctx.model
                .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
                .expect("moe_experts.down_proj missing");
            let down_block_size = down_t.ggml_type.block_size() as u32;
            let down_type_size = down_t.ggml_type.type_size() as u32;
            let down_bytes_per_expert = (down_t.byte_size / (n_experts as u64)) as u32;
            let down_elems_per_expert =
                down_bytes_per_expert / down_type_size * down_block_size;
            let mi_padded: u32 = ((mi + 255) / 256) * 256;
            let down_k = if down_t.shape[0] as u32 == n_experts {
                mi_padded
            } else {
                down_t.shape[0] as u32
            };
            let down_shader_gemv = layer_weight_indexed_shader(
                ctx.model, ctx.layer, "moe_experts.down_proj",
                fwd.mul_mat_vec_subgroup_enabled,
            );
            let down_w = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");
            let router_idx = router_indices;
            let router_w = router_weights;
            let o_buf = fwd.cur().o_buf.handle;

            // Zero the output accumulator
            unsafe {
                ctx.dev.device.cmd_fill_buffer(
                    ctx.cmd, batch_out, 0, (seq_len as u64) * h_bytes, 0,
                );
            }
            transfer_to_compute_barrier(ctx.dev, ctx.cmd);

            for t in 0..(seq_len as u64) {
                let out_off = t * h_bytes;
                for k in 0..(top_k as u64) {
                    let slot_flat = (t * (top_k as u64) + k) as u32;
                    let glu_off = (slot_flat as u64) * mi_bytes;

                    // (a) Per-slot down GEMV, reading from grouped_glu_out
                    //     at the slot's mi-block offset.
                    fwd.run_gemv_indexed_at_offset(
                        ctx.dev, ctx.registry, ctx.cmd,
                        down_w,
                        grouped_glu_out, glu_off, (down_k as u64) * 4,
                        o_buf, 0, h_bytes,
                        router_idx,
                        down_k, h,
                        down_elems_per_expert,
                        slot_flat,
                        "moe_down_bisect",
                        down_shader_gemv,
                    );
                    fwd.mark_written(&[o_buf]);
                    fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[o_buf]);

                    // (b) Indexed FMA into per-token slot of batch_out.
                    fwd.run_fma_add_indexed(
                        ctx.dev, ctx.registry, ctx.cmd,
                        o_buf, 0,
                        batch_out, out_off,
                        router_w,
                        h, slot_flat,
                        "moe_fma_bisect",
                    );
                    fwd.mark_written(&[batch_out]);
                    fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[batch_out]);
                }
            }
            compute_barrier(ctx.dev, ctx.cmd);
            return;
        }

        // -------- 8. Q8_1 quantize GLU output → grouped_glu_q8 --------
        fwd.run_quantize_q8_1(
            ctx.dev, ctx.registry, ctx.cmd,
            grouped_glu_out, grouped_glu_q8,
            seq_len * top_k * mi,
            "moe_q8_glu_b",
        );
        compute_barrier(ctx.dev, ctx.cmd);

        // -------- 9. down MMQ_ID (1 dispatch over all experts) --------
        let down_t = ctx.model
            .tensor(&format!("blk.{}.moe_experts.down_proj", ctx.layer))
            .expect("moe_experts.down_proj missing");
        let down_block_size = down_t.ggml_type.block_size() as u32;
        let down_type_size = down_t.ggml_type.type_size() as u32;
        let down_bytes_per_expert = (down_t.byte_size / (n_experts as u64)) as u32;
        let down_elems_per_expert =
            down_bytes_per_expert / down_type_size * down_block_size;
        let down_shader = layer_weight_mmq_id_shader(
            ctx.model, ctx.layer, "moe_experts.down_proj", subgroup,
        );
        let down_w = layer_weight(ctx.model, ctx.layer, "moe_experts.down_proj");
        fwd.run_mmq_id_grouped(
            ctx.dev, ctx.registry, ctx.cmd,
            down_shader,
            down_w,
            grouped_glu_q8,
            grouped_down_out,
            grouped_data_ids,
            grouped_data_counts,
            h,       // M = hidden
            mi,      // K
            seq_len, top_k, n_experts,
            down_elems_per_expert,
            mi,            // stride_b — slot picks within-token block
            top_k * mi,    // batch_stride_b — per-token block size in B
            top_k,         // ne11 = top_k (slot ∈ [0, top_k) maps to mi-block)
            "moe_down_grouped",
        );
        compute_barrier(ctx.dev, ctx.cmd);

        // -------- 10. Zero output accumulator --------
        unsafe {
            ctx.dev.device.cmd_fill_buffer(
                ctx.cmd, batch_out, 0, (seq_len as u64) * h_bytes, 0,
            );
        }
        transfer_to_compute_barrier(ctx.dev, ctx.cmd);

        // -------- 11. Per-(token, slot) scalar-weight FMA add --------
        // grouped_down_out layout: [seq_len × top_k × hidden] indexed by
        // (token, slot) — the MMQ_ID kernel wrote each output to its
        // (row_idx.y=token, row_idx.x=slot) lane. Iterate in the
        // original router order so the weight at position
        // `t * top_k + s` matches the down_out lane at
        // `(t * top_k + s) * hidden`.
        //
        // Sprint 61C-2 diagnostic: `VF_MOE_GROUPED_INDEXED_FMA=1`
        // swaps run_fma_add_at_offset (scalar push-constant weight)
        // for run_fma_add_indexed (weight from router.weights_scratch).
        // If output becomes correct → bug was in the scalar FMA path.
        let use_indexed_fma = std::env::var("VF_MOE_GROUPED_INDEXED_FMA")
            .map(|v| v == "1").unwrap_or(false);
        for t in 0..(seq_len as u64) {
            for k in 0..(top_k as u64) {
                let flat = (t * (top_k as u64) + k) as usize;
                let in_off = (flat as u64) * h_bytes;
                let out_off = t * h_bytes;
                if use_indexed_fma {
                    fwd.run_fma_add_indexed(
                        ctx.dev, ctx.registry, ctx.cmd,
                        grouped_down_out, in_off,
                        batch_out, out_off,
                        router_weights,
                        h, flat as u32,
                        "moe_fma_add_grouped_idx",
                    );
                } else {
                    let weight = weights_host[flat];
                    fwd.run_fma_add_at_offset(
                        ctx.dev, ctx.registry, ctx.cmd,
                        grouped_down_out, in_off,
                        batch_out, out_off,
                        h, weight,
                        "moe_fma_add_grouped",
                    );
                }
            }
        }
        compute_barrier(ctx.dev, ctx.cmd);
    }
}
