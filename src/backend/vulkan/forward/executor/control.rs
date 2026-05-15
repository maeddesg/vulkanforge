//! Sprint 57B — Layer-control + shared helpers.
//!
//! Two Gemma-4 per-layer steps + one BAT-only GEMM-routing helper:
//!   - `step_ple_block` / `b_step_ple_block` — 5-dispatch per-layer
//!     embedding (PLE) block, only emitted by the Gemma-4 plan builder.
//!   - `step_layer_scalar_mul` / `b_step_layer_scalar_mul` — final
//!     `layer_scalar` multiply applied to the layer output.
//!   - `b_run_proj` — batched GEMM-routing helper invoked by every
//!     attention/FFN projection on the BAT path.

use super::{
    batch_seq_len, decode_io, layer_dims_local, BatchExec, DecodeExec, ExecCtx,
};
use super::super::arch::{
    GemmKind, compute_barrier, is_f32_layer_weight, is_fp8_layer_weight, layer_weight,
    layer_weight_scale_block, layer_weight_scale_buf, layer_weight_scale_scalar,
    layer_weight_shader, layer_weight_shader_gemm, transfer_to_compute_barrier,
};
use super::super::state::Forward;
use super::super::super::gguf::{GgmlType, ModelConfig};
use super::super::super::pipeline::{GenericBinaryPushConstants, MatVecPushConstants, SwigluPushConstants};
use super::super::super::shaders::ShaderId;

use ash::vk;

impl DecodeExec {
    pub(super) fn step_ple_block(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn step_layer_scalar_mul(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

impl BatchExec {
    pub(super) fn b_step_ple_block(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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

    pub(super) fn b_step_layer_scalar_mul(&self, fwd: &mut Forward, cfg: &ModelConfig, ctx: &ExecCtx) {
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
    pub(super) fn b_run_proj(
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
