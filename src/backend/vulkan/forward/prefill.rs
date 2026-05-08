//! Sprint 44B-4 — prefill orchestration extracted from `forward/mod.rs`.
//! Pure code-move.
//!
//! Owns the batched-prompt path:
//! - `prefill_batch` (pub): the entry point invoked by
//!   `crate::backend::vulkan::decode::generate_from_tokens` after the
//!   chat template is applied. Records all 36 layers' batched GEMMs +
//!   per-token elementwise / attention dispatches into one (or several
//!   `layers_per_submit`-many) command buffers and submits them in
//!   sequence.
//! - `record_prefill_seed` / `record_prefill_finalize` — phase-1 / phase-3
//!   of the multi-CB path: copy `batch_input` into `batch_residual`
//!   (seed) and pull the last token's per-channel state into the per-
//!   token slot for decode handoff (finalize).
//! - `dispatch_layer_batch` — the prefill twin of `dispatch_layer`.
//!   Same per-layer math; reads/writes batched buffers
//!   (`batch_residual` / `batch_q8` / `batch_q` / etc.) instead of
//!   per-token slots.

use ash::vk;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::super::gguf::{GgmlType, ModelConfig};
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;
use super::super::shaders::ShaderId;

use super::arch::{
    GemmKind, compute_barrier, is_fp8_layer_weight, layer_dims, layer_weight,
    layer_weight_opt, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_shader_gemm, rope_params_for_layer, transfer_to_compute_barrier,
};
use super::state::Forward;

impl Forward {
    /// Phase-3E prefill_batch — runs `token_ids` through all 36 layers
    /// in **one** command buffer using batched GEMMs for the 7 weight
    /// projections per layer. Per-token loops handle elementwise
    /// (RMSNorm / RoPE / SiLU / Add / Mul) and the causal attention.
    ///
    /// On exit:
    /// * `kv_cache.current_seq_len` advances by `token_ids.len()`.
    /// * The last token's logits are in `self.logits_buf`, ready for
    ///   the decode loop's argmax.
    ///
    /// Caller must supply pre-computed FP32 embeddings for every token
    /// (one row of `token_embd.weight` Q4_K-dequantised, see
    /// `decode::embedding_row`). The flattened `[seq_len × hidden_dim]`
    /// goes into `batch_input`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embeddings: &[f32],
        seq_len: u32,
        base_pos: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if seq_len == 0 {
            return Ok(());
        }
        if seq_len > self.max_prefill_tokens {
            return Err(format!(
                "prefill_batch: seq_len {seq_len} > max_prefill_tokens {}",
                self.max_prefill_tokens
            ).into());
        }
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let ffn = cfg.ffn_dim;
        let hidden_bytes = (hidden as u64) * 4;
        let kv_bytes = (kv_dim as u64) * 4;
        let q_bytes = (q_dim as u64) * 4;
        if (embeddings.len() as u32) != seq_len * hidden {
            return Err("prefill_batch: embeddings length mismatch".into());
        }

        // CPU → batch_input (host-visible).
        self.batch_input.write_bytes(bytemuck::cast_slice(embeddings))?;

        // Pre-stage RoPE positions for every token in the batch.
        // CRITICAL: all GPU dispatches in this submit run AFTER all
        // host writes complete, so we must write every per-token
        // position into a separate slot of rope_pos_buf BEFORE we
        // start recording — otherwise the per-token RoPE dispatches
        // would all read the last-written value (Phase 3E drift bug).
        let positions: Vec<u32> = (0..seq_len).map(|t| base_pos + t).collect();
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::cast_slice(&positions))?;

        // prefill_batch's per-token attention loop binds varying
        // pos_buf sub-ranges, so cached sets from a prior decode
        // can't be reused. Drop them up-front.
        self.reset_descriptor_pool_and_cache(dev)?;

        // Sprint 19B-A — branch on multi-submit pacing. When
        // `prefill_cbs` is non-empty, we record the same dispatches
        // into N separate command buffers (chunked at every
        // `layers_per_submit` layer) and submit them sequentially.
        // Queue-ordering on the same compute queue makes the
        // intermediate submits implicitly act as full memory
        // barriers — no explicit cross-CB synchronization is needed.
        // Only the final submit carries `prefill_fence`; we wait
        // once at the end. The legacy branch goes through
        // `cmd_ctx.one_shot` exactly as it did pre-19B-A.
        let multi = !self.prefill_cbs.is_empty();
        if !multi {
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                self.record_prefill_seed(dev, registry, cmd, model, &cfg, seq_len, hidden, hidden_bytes);
                for layer in 0..cfg.n_layers {
                    let next_w = if layer + 1 < cfg.n_layers {
                        Some(layer_weight(model, layer + 1, "attn_norm.weight"))
                    } else {
                        None
                    };
                    self.dispatch_layer_batch(
                        dev, registry, cmd, model, layer, seq_len, base_pos,
                        next_w,
                    );
                }
                self.record_prefill_finalize(dev, registry, cmd, model, seq_len, hidden_bytes);
                let _ = (kv_bytes, q_bytes, ffn);
            })?;
        } else {
            let interval = self.layers_per_submit;
            let n_chunks = self.prefill_cbs.len();
            debug_assert!(n_chunks > 0);

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin chunk 0 + record seed.
            let cb0 = self.prefill_cbs[0];
            unsafe {
                dev.device.reset_command_buffer(cb0, vk::CommandBufferResetFlags::empty())?;
                dev.device.begin_command_buffer(cb0, &begin_info)?;
            }
            self.record_prefill_seed(dev, registry, cb0, model, &cfg, seq_len, hidden, hidden_bytes);

            // Layer loop with chunk boundaries.
            let mut chunk: usize = 0;
            for layer in 0..cfg.n_layers {
                let cmd = self.prefill_cbs[chunk];
                let next_w = if layer + 1 < cfg.n_layers {
                    Some(layer_weight(model, layer + 1, "attn_norm.weight"))
                } else {
                    None
                };
                self.dispatch_layer_batch(
                    dev, registry, cmd, model, layer, seq_len, base_pos,
                    next_w,
                );
                // Submit boundary: end & submit current CB, begin next.
                // Only between chunks — never after the very last layer
                // (that's the last-chunk path with the finalize tail).
                let crossed_boundary = (layer + 1) % interval == 0;
                let last_layer = layer + 1 == cfg.n_layers;
                if crossed_boundary && !last_layer {
                    unsafe {
                        dev.device.end_command_buffer(cmd)?;
                        let cmds_arr = [cmd];
                        let submit = vk::SubmitInfo::default().command_buffers(&cmds_arr);
                        dev.device.queue_submit(
                            dev.compute_queue,
                            std::slice::from_ref(&submit),
                            vk::Fence::null(),
                        )?;
                    }
                    chunk += 1;
                    let next_cb = self.prefill_cbs[chunk];
                    unsafe {
                        dev.device.reset_command_buffer(next_cb, vk::CommandBufferResetFlags::empty())?;
                        dev.device.begin_command_buffer(next_cb, &begin_info)?;
                    }
                }
            }

            // Finalize tail in the last chunk's CB.
            let last_cb = self.prefill_cbs[chunk];
            self.record_prefill_finalize(dev, registry, last_cb, model, seq_len, hidden_bytes);
            unsafe {
                dev.device.end_command_buffer(last_cb)?;
                dev.device.reset_fences(std::slice::from_ref(&self.prefill_fence))?;
                let cmds_arr = [last_cb];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds_arr);
                dev.device.queue_submit(
                    dev.compute_queue,
                    std::slice::from_ref(&submit),
                    self.prefill_fence,
                )?;
                dev.device.wait_for_fences(&[self.prefill_fence], true, u64::MAX)?;
            }
            let _ = (kv_bytes, q_bytes, ffn);
        }

        self.kv_cache.current_seq_len = base_pos + seq_len;
        Ok(())
    }

    /// Sprint 19B-A — phase-1 of `prefill_batch`: copy batch_input into
    /// batch_residual, barrier, then seed `batch_norm` with layer 0's
    /// attn_norm output (Sprint 9b.2 cross-layer-fusion contract). Same
    /// dispatches as the legacy single-CB body — extracted so the
    /// multi-submit path can record this phase into chunk 0.
    fn record_prefill_seed(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        cfg: &ModelConfig,
        seq_len: u32,
        hidden: u32,
        hidden_bytes: u64,
    ) {
        let total_bytes = (seq_len as u64) * hidden_bytes;
        let copy = vk::BufferCopy::default()
            .src_offset(0).dst_offset(0).size(total_bytes);
        unsafe {
            dev.device.cmd_copy_buffer(
                cmd, self.batch_input.handle, self.batch_residual.handle,
                std::slice::from_ref(&copy),
            );
        }
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&bar), &[], &[],
            );
        }
        let w_attn_norm_0 = layer_weight(model, 0, "attn_norm.weight");
        self.run_rms_norm(
            dev, registry, cmd,
            self.batch_residual.handle, w_attn_norm_0, self.batch_norm.handle,
            hidden, seq_len, cfg.rms_norm_eps, "rms_norm_attn_b_seed",
        );
        compute_barrier(dev, cmd);
    }

    /// Sprint 19B-A — phase-3 of `prefill_batch`: copy the last token's
    /// row of `batch_residual` into `scratch_a`, then run final norm +
    /// LM head + the host-read barrier on `logits_buf`. Extracted from
    /// the legacy single-CB body for the multi-submit path's last
    /// chunk.
    fn record_prefill_finalize(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        seq_len: u32,
        hidden_bytes: u64,
    ) {
        let last_off = ((seq_len - 1) as u64) * hidden_bytes;
        self.copy_batch_row(
            dev, cmd, self.batch_residual.handle, last_off,
            self.cur().scratch_a.handle, hidden_bytes,
        );
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&bar), &[], &[],
            );
        }
        self.dispatch_final(dev, registry, cmd, model, self.cur().scratch_a.handle);
        // Sprint 27 — copy logits to host-mapped staging.
        self.record_logits_readback(dev, cmd);
    }

    /// One layer's worth of batched dispatches, recorded into `cmd`.
    /// Reads from `batch_residual`, writes back to `batch_residual`.
    ///
    /// Sprint 9b.2 — cross-layer fusion contract:
    /// * `batch_norm` MUST already contain `rms_norm(batch_residual) *
    ///   layer N's attn_norm.weight` on entry. The caller is responsible
    ///   for seeding it (separate `run_rms_norm` before the layer-0 call;
    ///   subsequent layers inherit it from the previous layer's
    ///   end-of-layer fusion).
    /// * `next_attn_norm_weight = Some(w)` activates the end-of-layer
    ///   `multi_add_rms(batch_residual, batch_ffn_out, w)` fusion that
    ///   simultaneously updates `batch_residual` AND populates
    ///   `batch_norm` with `rms_norm(...) * w` for the *next* layer.
    /// * `next_attn_norm_weight = None` (last layer) emits a plain
    ///   `add_res2_b` and leaves `batch_norm` untouched.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_layer_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        seq_len: u32,
        base_pos: u32,
        next_attn_norm_weight: Option<vk::Buffer>,
    ) {
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        // Sprint 43C — per-layer head_dim / ffn_dim for Gemma-4. Falls
        // back to cfg uniform on every other architecture.
        let (head_dim, ffn, _rope_theta, _rotary_dim) = layer_dims(&cfg, layer);
        // Sprint 43D-2 — per-layer RoPE bundle. Non-Gemma-4 stacks get
        // (head_dim, head_dim, cfg.rope_freq_base, self.rope_theta_scale)
        // — bit-identical to pre-43D-2.
        let (_rope_head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale) =
            rope_params_for_layer(&cfg, self.rope_theta_scale, layer);
        let kv_dim = cfg.n_kv_heads * head_dim;
        let q_dim = cfg.n_heads * head_dim;
        let kv_bytes = (kv_dim as u64) * 4;
        let q_bytes = (q_dim as u64) * 4;
        let ffn_bytes = (ffn as u64) * 4;

        // Sprint 43F — VF_BATCH_INPUT_DUMP=N (single-buffer at slot 0)
        // and VF_BATCH_STEP_DUMP=ALL (6-stage at slots 0..5) within
        // dispatch_layer_batch for `layer == N` / `layer == 0`. Both
        // share `hidden_staging` (sized 8× hidden_dim post-43F bump);
        // the post-submit read in wait_and_read_logits + this run's
        // sub-bisect script picks slots out by offset.
        let batch_step_dump = std::env::var("VF_BATCH_STEP_DUMP").ok();
        let do_step_dump = batch_step_dump.as_deref() == Some("ALL") && layer == 0;
        let stage_dump = |this: &Self, c: vk::CommandBuffer, src: vk::Buffer, stage: u32| {
            if !do_step_dump { return; }
            let bytes = (this.config.hidden_dim as u64) * 4;
            let dst_off = (stage as u64) * bytes;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(bytes);
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    c,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
                dev.device.cmd_copy_buffer(
                    c, src, this.hidden_staging.handle,
                    std::slice::from_ref(&copy),
                );
            }
        };
        // Stage 0 — batch_residual at function entry (= initial residual
        // for layer 0 = embedding). Already seeded by prefill_batch.
        stage_dump(self, cmd, self.batch_residual.handle, 0);
        // Stage 1 — batch_norm at function entry (= pre-seeded post-
        // attn-norm output for layer 0 by prefill_batch).
        stage_dump(self, cmd, self.batch_norm.handle, 1);

        let batch_input_dump_layer: i32 = std::env::var("VF_BATCH_INPUT_DUMP")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(-1);
        if batch_input_dump_layer == (layer as i32) {
            let bytes = (hidden as u64) * 4;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(bytes);
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
                dev.device.cmd_copy_buffer(
                    cmd, self.batch_residual.handle, self.hidden_staging.handle,
                    std::slice::from_ref(&copy),
                );
            }
        }

        // ---- (a) attn_norm: per-token RMSNorm into batch_norm. ----
        // Sprint 9b.2 — this used to dispatch run_rms_norm(batch_residual,
        // attn_norm.weight → batch_norm). Now `batch_norm` is pre-seeded
        // by either prefill_batch (for layer 0) or by the previous
        // layer's cross-layer fusion (Sprint 9b.2). Nothing to do here.

        // Phase 6/7 — mul_mm path takes FP32 activations directly, so
        // skip the Q8_1 quantize step. mul_mmq still needs it. The
        // aligned variant (vec4 B-loads) requires seq_len % 4 == 0;
        // if it isn't we fall back to mul_mmq because mul_mm with
        // scalar B-loads is ~45 % slower than mul_mmq at prefill.
        // Sprint 11E — when COOPMAT mul_mm is on, force the unaligned MulMm
        // path (we don't ship a COOPMAT-aligned SPV). Otherwise the existing
        // MulMmAligned fallback path stays.
        // Sprint 17B — Q3_K_M GGUFs ship most weights as Q3_K. We
        // built only the `mul_mmq_q3_k_f32.spv` prefill kernel for
        // Q3_K, not the MulMm / MulMmAligned coopmat variants Q4_K
        // has, so force Mmq for any batch that touches Q3_K weights.
        // Sprint 17C — same applies to Q5_K (used as attn_q in
        // Q5_K_M / Q5_K_S; used as attn_v + ffn_down in Q3_K_M, but
        // attn_q in Q3_K_M is Q3_K so the Q3_K check already catches
        // those models). Sprint 17D — same applies to Q4_0 (Qwen2.5
        // Q4_0 GGUFs).
        // Sprint 19A — Q3_K and Q5_K now ship the full MulMm
        // coopmat coverage (mul_mm_{q3_k,q5_k}_f32{,_aligned,_coopmat,
        // _aligned_coopmat,_aligned_coopmat_f16acc}.spv) and route
        // through layer_weight_shader_gemm just like Q4_K/Q6_K. Q4_0
        // is the only quant that still falls back to Mmq because
        // mul_mm_funcs.glsl has no Q4_0 dequant branch (only Q-K).
        let attn_q_type = model
            .tensor(&format!("blk.{}.attn_q.weight", layer))
            .map(|t| t.ggml_type);
        let force_mmq = matches!(
            attn_q_type,
            Some(GgmlType::Q4_0),
        );
        let gemm_kind = if force_mmq {
            GemmKind::Mmq
        } else if self.mul_mm_coopmat_enabled {
            GemmKind::MulMm
        } else if self.mul_mm_enabled {
            if seq_len % 4 == 0 { GemmKind::MulMmAligned } else { GemmKind::Mmq }
        } else {
            GemmKind::Mmq
        };
        let use_mul_mm = matches!(gemm_kind, GemmKind::MulMm | GemmKind::MulMmAligned);
        let gemm_input_attn = if use_mul_mm {
            self.batch_norm.handle
        } else {
            // ---- (b) Quantize attn_norm output → Q8_1 (mul_mmq path) ----
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_norm.handle, self.batch_q8.handle,
                seq_len * hidden, "quantize_attn",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };

        // ---- (c) Q/K/V GEMMs. Mixed-quant: V uses Q6_K. ----
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        let cm_mm = self.mul_mm_coopmat_enabled;
        let cm_f16 = self.mul_mm_coopmat_f16acc_enabled;
        let sq = layer_weight_shader_gemm(model, layer, "attn_q.weight", gemm_kind, q_dim, seq_len, cm_mm, cm_f16);
        let sk = layer_weight_shader_gemm(model, layer, "attn_k.weight", gemm_kind, kv_dim, seq_len, cm_mm, cm_f16);
        let sv = layer_weight_shader_gemm(model, layer, "attn_v.weight", gemm_kind, kv_dim, seq_len, cm_mm, cm_f16);
        // Sprint 3A: gemm_q can opt into the Q4_K coopmat fusion.
        // Coopmat reads activations from `batch_norm` (FP32) regardless
        // of the mul_mm/mul_mmq route the rest of the layer takes, so
        // the coopmat dispatch passes `self.batch_norm.handle` directly
        // — independent of `gemm_input_attn` (which is either FP32 or
        // Q8_1 depending on mul_mm_enabled). The other six GEMMs keep
        // the existing routing.
        if self.coopmat_q4k_enabled {
            // Sprint 3C — naive padded for skinny-N (covers all
            // typical Qwen3 prefill shapes). Pad seq_len up to a
            // multiple of 16 and zero the activation tail so every
            // output tile is full and the kernel can use a direct
            // ColumnMajor coopMatStore. The fused FP8/BF16 mode
            // toggles via `coopmat_fp8_enabled`.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_norm.handle,
                seq_len, n_padded, hidden,
            );
            // The fill-buffer is a TRANSFER op; gate the next
            // compute-shader read with a TRANSFER → COMPUTE barrier.
            transfer_to_compute_barrier(dev, cmd);

            let (qkw_shader, qkw_bm, qkw_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, qkw_shader, wq,
                self.batch_norm.handle, self.batch_q.handle,
                q_dim, n_padded, hidden, qkw_bm, qkw_bn, "gemm_q_coopmat",
            );
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, qkw_shader, wk,
                self.batch_norm.handle, self.batch_k.handle,
                kv_dim, n_padded, hidden, qkw_bm, qkw_bn, "gemm_k_coopmat",
            );
            // gemm_v stays on mul_mmq — Qwen3 uses Q6_K for attn_v
            // (mixed-quant) and we don't ship a Q6_K coopmat dequant
            // kernel yet. Activations for gemm_v are still in batch_q8
            // thanks to the earlier run_quantize_q8_1 in the mul_mmq
            // path.
            self.run_gemm(
                dev, registry, cmd, sv, wv,
                gemm_input_attn, self.batch_v.handle,
                kv_dim, seq_len, hidden, "gemm_v",
            );
        } else if is_fp8_layer_weight(model, layer, "attn_q.weight") {
            // Sprint 20-Wire — SafeTensors FP8 path. Routes Q/K/V
            // through `mul_coopmat_fp8_naive.comp`. Activation buffer
            // is `gemm_input_attn = batch_norm.handle` (FP32, set
            // above because gemm_kind = MulMm for FP8).
            let scale_q = layer_weight_scale_buf(model, layer, "attn_q.weight")
                .expect("FP8 GEMM Q requires a scale buffer");
            let blk_q = layer_weight_scale_block(model, layer, "attn_q.weight");
            let scale_k = layer_weight_scale_buf(model, layer, "attn_k.weight")
                .expect("FP8 GEMM K requires a scale buffer");
            let blk_k = layer_weight_scale_block(model, layer, "attn_k.weight");
            let scale_v = layer_weight_scale_buf(model, layer, "attn_v.weight")
                .expect("FP8 GEMM V requires a scale buffer");
            let blk_v = layer_weight_scale_block(model, layer, "attn_v.weight");
            if let Some((bn, bk)) = blk_q {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wq, scale_q,
                        gemm_input_attn, self.batch_q.handle,
                        q_dim, seq_len, hidden, bn, bk, "gemm_q_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wq, scale_q,
                        gemm_input_attn, self.batch_q.handle,
                        q_dim, seq_len, hidden, "gemm_q_fp8",
                    );
                }
            if let Some((bn, bk)) = blk_k {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wk, scale_k,
                        gemm_input_attn, self.batch_k.handle,
                        kv_dim, seq_len, hidden, bn, bk, "gemm_k_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wk, scale_k,
                        gemm_input_attn, self.batch_k.handle,
                        kv_dim, seq_len, hidden, "gemm_k_fp8",
                    );
                }
            if let Some((bn, bk)) = blk_v {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wv, scale_v,
                        gemm_input_attn, self.batch_v.handle,
                        kv_dim, seq_len, hidden, bn, bk, "gemm_v_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wv, scale_v,
                        gemm_input_attn, self.batch_v.handle,
                        kv_dim, seq_len, hidden, "gemm_v_fp8",
                    );
                }
        } else {
            self.run_gemm(
                dev, registry, cmd, sq, wq,
                gemm_input_attn, self.batch_q.handle,
                q_dim, seq_len, hidden, "gemm_q",
            );
            self.run_gemm(
                dev, registry, cmd, sk, wk,
                gemm_input_attn, self.batch_k.handle,
                kv_dim, seq_len, hidden, "gemm_k",
            );
            self.run_gemm(
                dev, registry, cmd, sv, wv,
                gemm_input_attn, self.batch_v.handle,
                kv_dim, seq_len, hidden, "gemm_v",
            );
        }
        compute_barrier(dev, cmd);

        // Sprint 24B — Q/K/V bias-add (Qwen2 attention biases). Skipped
        // for architectures without biases (Llama / Qwen3 / Mistral).
        // Broadcasts the [dim] bias over the seq_len rows in batch_q/k/v.
        let q_bias_b = layer_weight_opt(model, layer, "attn_q.bias");
        let k_bias_b = layer_weight_opt(model, layer, "attn_k.bias");
        let v_bias_b = layer_weight_opt(model, layer, "attn_v.bias");
        if q_bias_b.is_some() || k_bias_b.is_some() || v_bias_b.is_some() {
            let bq = self.batch_q.handle;
            let bk = self.batch_k.handle;
            let bv = self.batch_v.handle;
            if let Some(b) = q_bias_b { self.run_bias_add(dev, registry, cmd, bq, b, bq, q_dim, seq_len, "bias_q_b"); }
            if let Some(b) = k_bias_b { self.run_bias_add(dev, registry, cmd, bk, b, bk, kv_dim, seq_len, "bias_k_b"); }
            if let Some(b) = v_bias_b { self.run_bias_add(dev, registry, cmd, bv, b, bv, kv_dim, seq_len, "bias_v_b"); }
            compute_barrier(dev, cmd);
        }

        // Sprint 43F sub-bisect — Stage 2: post Q/K/V GEMM (pre-RoPE).
        stage_dump(self, cmd, self.batch_q.handle, 2);

        // ---- (d) Q/K-norm + RoPE + KV-cache write ----
        //
        // Two paths:
        //   * Phase 5B.3 fully-batched (default, when `batch_attn_enabled`):
        //     ONE dispatch each for Q-norm, K-norm, RoPE-Q, RoPE-K
        //     reading directly from / writing back to batch_q /
        //     batch_k, then ONE bulk `cmd_copy_buffer` per layer for
        //     K and V into the KV cache. Skips the per-token loop
        //     entirely. After this block batch_q holds post-RoPE Q
        //     for all M tokens; the attention call below consumes it.
        //
        //   * Per-token legacy (when `batch_attn_enabled = false`):
        //     One dispatch per (token, op) — same code path that
        //     shipped through Phase 5A. Gated below.
        let qk_norm_weights: Option<(vk::Buffer, vk::Buffer)> = if cfg.has_qk_norm {
            Some((
                layer_weight(model, layer, "attn_q_norm.weight"),
                layer_weight(model, layer, "attn_k_norm.weight"),
            ))
        } else {
            None
        };
        let batch_attn = self.batch_attn_enabled;

        if batch_attn {
            // ---- Phase 5B.3 fully-batched per-layer attention prep ----
            // Sprint 9c.5 — Q/K-norm + RoPE fused into a single
            // dispatch each (was: 4 dispatches + 2 barriers; now: 2
            // dispatches + 1 barrier per layer). Position-buffer is
            // pre-staged in prefill_batch with [base_pos, base_pos+1,
            // …, base_pos+M-1] at slots 0..M.
            //
            // The fused path requires a Q/K-norm weight to drive the
            // do_multiply branch. If the model has no qk_norm
            // (non-Qwen archs), fall back to the old separate
            // run_rope_batch dispatches with no rms_norm.
            if let Some((wqn, wkn)) = qk_norm_weights {
                self.run_rms_norm_mul_rope(
                    dev, registry, cmd,
                    self.batch_q.handle, wqn, self.batch_q.handle,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_heads, seq_len,
                    cfg.rms_norm_eps, "rms_norm_mul_rope_q_b",
                );
                self.run_rms_norm_mul_rope(
                    dev, registry, cmd,
                    self.batch_k.handle, wkn, self.batch_k.handle,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_kv_heads, seq_len,
                    cfg.rms_norm_eps, "rms_norm_mul_rope_k_b",
                );
            } else {
                // No qk_norm: keep the legacy stand-alone RoPE dispatches.
                self.run_rope_batch(
                    dev, registry, cmd,
                    self.batch_q.handle, self.batch_q.handle,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_heads, seq_len,
                    "rope_q_batch",
                );
                self.run_rope_batch(
                    dev, registry, cmd,
                    self.batch_k.handle, self.batch_k.handle,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_kv_heads, seq_len,
                    "rope_k_batch",
                );
            }
            compute_barrier(dev, cmd);
            // Sprint 43F sub-bisect — Stage 3: post-RoPE Q (pre-attn).
            stage_dump(self, cmd, self.batch_q.handle, 3);

            // 3. Bulk KV-cache write. K and V are M contiguous rows of
            //    `[n_kv_heads, head_dim]` in batch_k / batch_v; the
            //    cache slot for this layer at positions
            //    `base_pos..base_pos+M` is the same shape.
            //
            // Sprint 9d.2 — when the cache is FP16-allocated, the
            // raw byte copy can't be used (it would copy FP32 bytes
            // into a half-size FP16 slot). We dispatch the
            // `kv_copy_fp16` compute shader instead, which converts
            // FP32 → packed-FP16 element-wise. FP32 cache stays on
            // the cheap vkCmdCopyBuffer path.
            let dst_off = self.kv_cache.pos_offset_bytes(layer, base_pos);
            let kv_elements = (seq_len as u32) * kv_dim;
            if self.kv_cache.is_fp8() {
                self.run_kv_store_fp8(
                    dev, registry, cmd,
                    self.batch_k.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_store_fp8_k_b",
                );
                self.run_kv_store_fp8(
                    dev, registry, cmd,
                    self.batch_v.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_store_fp8_v_b",
                );
            } else if self.kv_cache.is_fp16() {
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.batch_k.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_k_b",
                );
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.batch_v.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_v_b",
                );
            } else {
                let kv_row_bytes = self.kv_cache.row_bytes(layer);
                let bulk_size = (seq_len as u64) * kv_row_bytes;
                let copy_k = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(dst_off).size(bulk_size);
                let copy_v = copy_k;
                unsafe {
                    dev.device.cmd_copy_buffer(
                        cmd, self.batch_k.handle, self.kv_cache.k_buffer.handle,
                        std::slice::from_ref(&copy_k),
                    );
                    dev.device.cmd_copy_buffer(
                        cmd, self.batch_v.handle, self.kv_cache.v_buffer.handle,
                        std::slice::from_ref(&copy_v),
                    );
                }
            }
            // Barrier: subsequent attention reads KV (compute). The
            // upstream write was either a transfer (FP32 path) or a
            // compute dispatch (FP16 path) — cover both stages.
            let kv_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&kv_bar), &[], &[],
                );
            }
        }

        // Per-token loop only runs when batch_attn is OFF (legacy
        // path). When batch_attn is ON we already handled Q/K-norm,
        // RoPE, and the KV-cache write above; the loop body below
        // would do that per-token, which is exactly what we replaced.
        if !batch_attn {
        for t in 0..seq_len {
            let pos = base_pos + t;
            // RoPE position lives at slot `t` of rope_pos_buf — written
            // upfront in prefill_batch (see drift-fix comment there).
            let rope_pos_offset = (t as u64) * 4;
            // Pull token-row Q/K/V into single-token scratch.
            let q_off = (t as u64) * q_bytes;
            let kv_off = (t as u64) * kv_bytes;
            self.copy_batch_row(dev, cmd, self.batch_q.handle, q_off, self.cur().q_buf.handle, q_bytes);
            self.copy_batch_row(dev, cmd, self.batch_k.handle, kv_off, self.cur().k_buf.handle, kv_bytes);
            self.copy_batch_row(dev, cmd, self.batch_v.handle, kv_off, self.cur().v_buf.handle, kv_bytes);
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
            }
            // Q/K-norm — Qwen-only.
            if let Some((wqn, wkn)) = qk_norm_weights {
                self.run_rms_norm(
                    dev, registry, cmd, self.cur().q_buf.handle, wqn, self.cur().q_buf.handle,
                    head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q_b",
                );
                self.run_rms_norm(
                    dev, registry, cmd, self.cur().k_buf.handle, wkn, self.cur().k_buf.handle,
                    head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k_b",
                );
                compute_barrier(dev, cmd);
            }
            // RoPE — each dispatch reads its OWN position slot.
            self.run_rope_neox_with_pos_offset(
                dev, registry, cmd, self.cur().q_buf.handle, self.cur().q_buf.handle,
                head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                cfg.n_heads, pos, rope_pos_offset, "rope_q_b",
            );
            self.run_rope_neox_with_pos_offset(
                dev, registry, cmd, self.cur().k_buf.handle, self.cur().k_buf.handle,
                head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                cfg.n_kv_heads, pos, rope_pos_offset, "rope_k_b",
            );
            compute_barrier(dev, cmd);
            // KV-cache write at this token's position. Sprint 9d.3 —
            // FP16 KV path. This per-token legacy branch fires when
            // batch_attn_enabled=false (the
            // `phase5b2_batch_attn_parity_qwen3_*` regression tests
            // exercise this exact code path).
            let row_bytes = self.kv_cache.row_bytes(layer);
            let dst_off = self.kv_cache.pos_offset_bytes(layer, pos);
            if self.kv_cache.is_fp8() {
                let kv_elements = kv_dim;
                self.run_kv_store_fp8(
                    dev, registry, cmd,
                    self.cur().k_buf.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_store_fp8_k_t",
                );
                self.run_kv_store_fp8(
                    dev, registry, cmd,
                    self.cur().v_buf.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_store_fp8_v_t",
                );
            } else if self.kv_cache.is_fp16() {
                let kv_elements = kv_dim;
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.cur().k_buf.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_k_t",
                );
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.cur().v_buf.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_v_t",
                );
            } else {
                let copy = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(dst_off).size(row_bytes);
                unsafe {
                    dev.device.cmd_copy_buffer(
                        cmd, self.cur().k_buf.handle, self.kv_cache.k_buffer.handle,
                        std::slice::from_ref(&copy),
                    );
                    dev.device.cmd_copy_buffer(
                        cmd, self.cur().v_buf.handle, self.kv_cache.v_buffer.handle,
                        std::slice::from_ref(&copy),
                    );
                }
            }
            let kv_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&kv_bar), &[], &[],
                );
            }
            // Per-token attention path (legacy). seq_len for the
            // attention dispatch is pos+1 (causal — only KV
            // positions 0..=pos visible).
            self.run_scalar_attn(dev, registry, cmd, layer, pos);
            compute_barrier(dev, cmd);
            // Store attn_out[t] back into batch_attn_out.
            let copy_back = vk::BufferCopy::default()
                .src_offset(0).dst_offset(q_off).size(q_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, self.cur().attn_out.handle, self.batch_attn_out.handle,
                    std::slice::from_ref(&copy_back),
                );
            }
            let pst = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&pst), &[], &[],
                );
            }
        }
        } // end if !batch_attn

        // ---- (d.5) Phase 5B.2 batched attention ----
        // Replaces M attention dispatches with one. Reads post-RoPE Q
        // from batch_q (staged in the loop above), K/V from the layer's
        // KV-cache slice (positions 0..=base_pos+seq_len-1), and
        // writes [seq_len, n_heads, head_dim] into batch_attn_out.
        //
        // Sprint 7 — VULKANFORGE_FA_TILED=1 routes through the Br>1
        // tiled-Q kernel (BR=4 queries per workgroup sharing a K-tile).
        // Default OFF; flash_attn_batch (Br=1) remains the proven path
        // until per-shape benches show tiled wins.
        if batch_attn {
            if self.fa_tiled_enabled {
                self.run_flash_attn_tiled(
                    dev, registry, cmd,
                    layer,
                    self.batch_q.handle,
                    self.batch_attn_out.handle,
                    seq_len,
                    base_pos,
                    base_pos + seq_len,
                );
            } else {
                self.run_flash_attn_batch(
                    dev, registry, cmd,
                    layer,
                    self.batch_q.handle,
                    self.batch_attn_out.handle,
                    seq_len,
                    base_pos,
                    base_pos + seq_len,
                );
            }
            compute_barrier(dev, cmd);
        }
        // Sprint 43F sub-bisect — Stage 4: post-attention.
        stage_dump(self, cmd, self.batch_attn_out.handle, 4);

        // ---- (e) Output projection: GEMM(attn_out → o_batch). ----
        let wo = layer_weight(model, layer, "attn_output.weight");
        if self.coopmat_q4k_enabled {
            // Coopmat path reads FP32 activations directly — skip the
            // q8_1 quantize for gemm_o. Pad N + zero tail.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_attn_out.handle,
                seq_len, n_padded, q_dim,
            );
            transfer_to_compute_barrier(dev, cmd);

            let (o_shader, o_bm, o_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, o_shader, wo,
                self.batch_attn_out.handle, self.batch_o.handle,
                hidden, n_padded, q_dim, o_bm, o_bn, "gemm_o_coopmat",
            );
        } else {
            let gemm_input_o = if use_mul_mm {
                self.batch_attn_out.handle
            } else {
                self.run_quantize_q8_1(
                    dev, registry, cmd,
                    self.batch_attn_out.handle, self.batch_q8.handle,
                    seq_len * q_dim, "quantize_attn_out",
                );
                compute_barrier(dev, cmd);
                self.batch_q8.handle
            };
            let so = layer_weight_shader_gemm(model, layer, "attn_output.weight", gemm_kind, hidden, seq_len, self.mul_mm_coopmat_enabled, self.mul_mm_coopmat_f16acc_enabled);
            if is_fp8_layer_weight(model, layer, "attn_output.weight") {
                let scale_o = layer_weight_scale_buf(model, layer, "attn_output.weight")
                    .expect("FP8 GEMM O requires a scale buffer");
            let blk_o = layer_weight_scale_block(model, layer, "attn_output.weight");
                if let Some((bn, bk)) = blk_o {
                        self.run_gemm_fp8_blockwise(
                            dev, cmd, wo, scale_o,
                            gemm_input_o, self.batch_o.handle,
                            hidden, seq_len, q_dim, bn, bk, "gemm_o_fp8_bw",
                        );
                    } else {
                        self.run_gemm_fp8_naive(
                            dev, registry, cmd, wo, scale_o,
                            gemm_input_o, self.batch_o.handle,
                            hidden, seq_len, q_dim, "gemm_o_fp8",
                        );
                    }
            } else {
                self.run_gemm(
                    dev, registry, cmd, so, wo,
                    gemm_input_o, self.batch_o.handle,
                    hidden, seq_len, q_dim, "gemm_o",
                );
            }
        }
        compute_barrier(dev, cmd);

        // ---- (f+g) Norms around the attention residual.
        //
        // Llama / Qwen layout (one norm):
        //   batch_residual += batch_o
        //   batch_norm     = rms_norm(batch_residual) * ffn_norm.weight
        // — fused into multi_add_rms via Sprint 9b.
        //
        // Sprint 43F (port of dispatch_layer's 43D-1 fork into the
        // batch path) — Gemma-4 layout (TWO norms; different residual
        // structure). Memory `feedback_layer_dispatch_paths`: the
        // fork must land in dispatch_layer + dispatch_layer_batch +
        // dispatch_layer_partial — only dispatch_layer was forked in
        // 43D-1, leaving the prefill path silently broken.
        //
        //   o_normed       = rms_norm(batch_o) * post_attention_layernorm.w
        //   batch_residual = batch_residual + o_normed     (in-place)
        //   batch_norm     = rms_norm(batch_residual) * pre_feedforward_layernorm.w
        //
        // (post_attention_layernorm is mapped to `ffn_norm.weight` in
        //  VF's hf_to_vf_name table; pre_feedforward_layernorm is the
        //  Sprint 43B-1 added `ffn_pre_norm.weight`.)
        if cfg.gemma4.is_some() {
            // batch_attn_out has been consumed by O-proj already and
            // is sized [max_pp × q_dim × 4]. For Gemma-4 worst-case
            // q_dim=4096 (full attention), so the [seq_len × hidden=
            // 1536] worth of bytes we need (≤ max_pp × 4096 × 4)
            // fits with plenty of headroom.
            let scratch = self.batch_attn_out.handle;
            let post_attn_w = layer_weight(model, layer, "ffn_norm.weight");
            self.run_rms_norm(
                dev, registry, cmd,
                self.batch_o.handle, post_attn_w, scratch,
                hidden, seq_len, cfg.rms_norm_eps, "rms_norm_post_attn_b",
            );
            compute_barrier(dev, cmd);
            self.run_binary(
                dev, registry, cmd, ShaderId::Add,
                self.batch_residual.handle, scratch, self.batch_residual.handle,
                seq_len * hidden, "add_res1_gemma4_b",
            );
            compute_barrier(dev, cmd);
            let pre_ffn_w = layer_weight(model, layer, "ffn_pre_norm.weight");
            self.run_rms_norm(
                dev, registry, cmd,
                self.batch_residual.handle, pre_ffn_w, self.batch_norm.handle,
                hidden, seq_len, cfg.rms_norm_eps, "rms_norm_pre_ffn_b",
            );
            compute_barrier(dev, cmd);
            // Sprint 43F sub-bisect — Stage 6: post (f+g) Gemma-4 fork
            // = batch_norm (input to gate/up GEMM) + batch_residual
            // (post attn-residual update, pre MLP).
            stage_dump(self, cmd, self.batch_norm.handle, 6);
            stage_dump(self, cmd, self.batch_residual.handle, 7);
        } else {
            let w_ffn_norm = layer_weight(model, layer, "ffn_norm.weight");
            self.run_multi_add_rms(
                dev, registry, cmd,
                self.batch_residual.handle, self.batch_o.handle, w_ffn_norm,
                /* sum_out  = */ self.batch_residual.handle,
                /* norm_out = */ self.batch_norm.handle,
                hidden, seq_len, cfg.rms_norm_eps, "add_rms_ffn_b",
            );
            compute_barrier(dev, cmd);
        }

        // ---- (h) Quantize FFN-norm output (mul_mmq path only). ----
        let gemm_input_ffn = if use_mul_mm {
            self.batch_norm.handle
        } else {
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_norm.handle, self.batch_q8.handle,
                seq_len * hidden, "quantize_ffn",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };

        // ---- (i) Gate + Up GEMMs. ----
        let wg = layer_weight(model, layer, "ffn_gate.weight");
        let wu = layer_weight(model, layer, "ffn_up.weight");
        if self.coopmat_q4k_enabled {
            // Both gemm_gate and gemm_up read batch_norm — already
            // padded for the attention-block coopmat dispatches at
            // the top of dispatch_layer_batch. The FFN-norm pass
            // *re*-writes batch_norm at this point so we have to
            // pad again.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_norm.handle,
                seq_len, n_padded, hidden,
            );
            transfer_to_compute_barrier(dev, cmd);

            let (gu_shader, gu_bm, gu_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, gu_shader, wg,
                self.batch_norm.handle, self.batch_gate.handle,
                ffn, n_padded, hidden, gu_bm, gu_bn, "gemm_gate_coopmat",
            );
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, gu_shader, wu,
                self.batch_norm.handle, self.batch_up.handle,
                ffn, n_padded, hidden, gu_bm, gu_bn, "gemm_up_coopmat",
            );
        } else if is_fp8_layer_weight(model, layer, "ffn_gate.weight") {
            // Sprint 20-Wire — SafeTensors FP8 path for gate + up.
            let scale_g = layer_weight_scale_buf(model, layer, "ffn_gate.weight")
                .expect("FP8 GEMM gate requires a scale buffer");
            let blk_g = layer_weight_scale_block(model, layer, "ffn_gate.weight");
            let scale_u = layer_weight_scale_buf(model, layer, "ffn_up.weight")
                .expect("FP8 GEMM up requires a scale buffer");
            let blk_u = layer_weight_scale_block(model, layer, "ffn_up.weight");
            if let Some((bn, bk)) = blk_g {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wg, scale_g,
                        gemm_input_ffn, self.batch_gate.handle,
                        ffn, seq_len, hidden, bn, bk, "gemm_gate_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wg, scale_g,
                        gemm_input_ffn, self.batch_gate.handle,
                        ffn, seq_len, hidden, "gemm_gate_fp8",
                    );
                }
            if let Some((bn, bk)) = blk_u {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wu, scale_u,
                        gemm_input_ffn, self.batch_up.handle,
                        ffn, seq_len, hidden, bn, bk, "gemm_up_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wu, scale_u,
                        gemm_input_ffn, self.batch_up.handle,
                        ffn, seq_len, hidden, "gemm_up_fp8",
                    );
                }
        } else {
            let sg = layer_weight_shader_gemm(model, layer, "ffn_gate.weight", gemm_kind, ffn, seq_len, self.mul_mm_coopmat_enabled, self.mul_mm_coopmat_f16acc_enabled);
            let su = layer_weight_shader_gemm(model, layer, "ffn_up.weight", gemm_kind, ffn, seq_len, self.mul_mm_coopmat_enabled, self.mul_mm_coopmat_f16acc_enabled);
            self.run_gemm(
                dev, registry, cmd, sg, wg,
                gemm_input_ffn, self.batch_gate.handle,
                ffn, seq_len, hidden, "gemm_gate",
            );
            self.run_gemm(
                dev, registry, cmd, su, wu,
                gemm_input_ffn, self.batch_up.handle,
                ffn, seq_len, hidden, "gemm_up",
            );
        }
        compute_barrier(dev, cmd);
        // Sprint 43F — Stages 10/11: post Gate/Up GEMMs.
        stage_dump(self, cmd, self.batch_gate.handle, 10);
        stage_dump(self, cmd, self.batch_up.handle, 11);

        // ---- (j) Fused activation-GLU: batch_ffn_hidden = act(gate) * up.
        // v0.2 Sprint 9a — replaces silu(gate→gate) + mul(gate, up→
        // ffn_hidden) with a single dispatch. Sprint 43D-2 — Gemma-4
        // substitutes pytorch-tanh GELU (HF `gelu_pytorch_tanh`).
        let use_gelu_pytorch_tanh = cfg
            .gemma4
            .as_ref()
            .map(|g| g.hidden_activation == "gelu_pytorch_tanh")
            .unwrap_or(false);
        if use_gelu_pytorch_tanh {
            self.run_gelu_pytorch_tanh_glu(
                dev, registry, cmd,
                self.batch_gate.handle, self.batch_up.handle, self.batch_ffn_hidden.handle,
                seq_len * ffn, "gelu_pt_glu_b",
            );
        } else {
            self.run_swiglu(
                dev, registry, cmd,
                self.batch_gate.handle, self.batch_up.handle, self.batch_ffn_hidden.handle,
                seq_len * ffn, "swiglu_b",
            );
        }
        compute_barrier(dev, cmd);
        // Sprint 43F — Stage 12: post-(Gemma-4-correct) GLU activation.
        stage_dump(self, cmd, self.batch_ffn_hidden.handle, 12);

        // ---- (k) Quantize ffn_hidden + Down-proj GEMM (Q4_K). ----
        // NOTE: gemm_down is left on mul_mmq even when coopmat is on.
        // The coopmat path produced NaN logits when all 6 Q4_K GEMMs
        // were swapped — bisect localised the divergence to gemm_down
        // (K = ffn = 11008, the longest K-chain in the model). Sprint
        // 3C will revisit this with header-caching for the 11008/256
        // = 43 blocks per row, and/or a per-row scaling pass.
        let gemm_input_down = if use_mul_mm {
            self.batch_ffn_hidden.handle
        } else {
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_ffn_hidden.handle, self.batch_q8.handle,
                seq_len * ffn, "quantize_ffn_h",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };
        let wd = layer_weight(model, layer, "ffn_down.weight");
        let sd = layer_weight_shader_gemm(model, layer, "ffn_down.weight", gemm_kind, hidden, seq_len, self.mul_mm_coopmat_enabled, self.mul_mm_coopmat_f16acc_enabled);
        if is_fp8_layer_weight(model, layer, "ffn_down.weight") {
            let scale_d = layer_weight_scale_buf(model, layer, "ffn_down.weight")
                .expect("FP8 GEMM down requires a scale buffer");
            let blk_d = layer_weight_scale_block(model, layer, "ffn_down.weight");
            if let Some((bn, bk)) = blk_d {
                    self.run_gemm_fp8_blockwise(
                        dev, cmd, wd, scale_d,
                        gemm_input_down, self.batch_ffn_out.handle,
                        hidden, seq_len, ffn, bn, bk, "gemm_down_fp8_bw",
                    );
                } else {
                    self.run_gemm_fp8_naive(
                        dev, registry, cmd, wd, scale_d,
                        gemm_input_down, self.batch_ffn_out.handle,
                        hidden, seq_len, ffn, "gemm_down_fp8",
                    );
                }
        } else {
            self.run_gemm(
                dev, registry, cmd, sd, wd,
                gemm_input_down, self.batch_ffn_out.handle,
                hidden, seq_len, ffn, "gemm_down",
            );
        }
        compute_barrier(dev, cmd);

        // ---- (l) Residual2 = residual + ffn_out + (cross-layer fuse). ----
        // Sprint 9b.2 — when there's a next layer, the residual update
        // is fused with that layer's `attn_norm` rms_norm-mul, putting
        // the next layer's pre-attn norm into batch_norm in the same
        // dispatch (and saving 1 dispatch + 1 barrier per layer
        // boundary). For the final layer, fall back to a plain add.
        // Sprint 43F — Gemma-4 has post_feedforward_layernorm
        // applied to ffn_out *before* the final residual add, plus
        // the (next-layer pre-attn) seed when non-last. Llama / Qwen
        // path is unchanged (single fused multi_add_rms or plain add).
        if cfg.gemma4.is_some() {
            // Sprint 43F sub-bisect — Stage 8: batch_ffn_out at entry to (l)
            // = MLP output before post-FFN-norm.
            stage_dump(self, cmd, self.batch_ffn_out.handle, 8);

            // 1) ffn_out_normed = rms_norm(batch_ffn_out) * ffn_post_norm.weight
            let post_ffn_w = layer_weight(model, layer, "ffn_post_norm.weight");
            // Reuse batch_attn_out as scratch (consumed by O-proj earlier).
            let scratch = self.batch_attn_out.handle;
            self.run_rms_norm(
                dev, registry, cmd,
                self.batch_ffn_out.handle, post_ffn_w, scratch,
                hidden, seq_len, cfg.rms_norm_eps, "rms_norm_post_ffn_b",
            );
            compute_barrier(dev, cmd);
            // Sprint 43F sub-bisect — Stage 9: post-(l)-step1 scratch
            // (= post post_feedforward_layernorm, pre add).
            stage_dump(self, cmd, scratch, 9);
            // 2) batch_residual = batch_residual + scratch
            self.run_binary(
                dev, registry, cmd, ShaderId::Add,
                self.batch_residual.handle, scratch, self.batch_residual.handle,
                seq_len * hidden, "add_res2_gemma4_b",
            );
            compute_barrier(dev, cmd);
            // Sprint 43D-4 — Gemma-4 per-layer scalar applied to
            // batch_residual BEFORE the next-layer attn_norm seed below.
            // (Gemma-4 currently routes through force_per_token_prefill
            // and never reaches this batch path, but the order here
            // matters for correctness if a future sprint re-enables
            // batch prefill: the next layer's input_layernorm must see
            // the scaled residual, not the un-scaled one.)
            // TODO: 44-batch — exercise this path once F32 mul_mm shader
            // family lands and force_per_token_prefill can be lifted.
            let scalar = layer_weight(model, layer, "layer_scalar");
            self.run_mul_scalar_b(
                dev, registry, cmd,
                self.batch_residual.handle, scalar, self.batch_residual.handle,
                seq_len * hidden, "layer_scalar_mul_b",
            );
            compute_barrier(dev, cmd);
            // 3) (non-last only) batch_norm = rms_norm(batch_residual)
            //    × next_attn_norm.weight   (= layer N+1's input_layernorm).
            if let Some(w_next) = next_attn_norm_weight {
                self.run_rms_norm(
                    dev, registry, cmd,
                    self.batch_residual.handle, w_next, self.batch_norm.handle,
                    hidden, seq_len, cfg.rms_norm_eps, "rms_norm_next_attn_b",
                );
                compute_barrier(dev, cmd);
            }
        } else {
            match next_attn_norm_weight {
                Some(w_next) => {
                    self.run_multi_add_rms(
                        dev, registry, cmd,
                        self.batch_residual.handle, self.batch_ffn_out.handle, w_next,
                        /* sum_out  = */ self.batch_residual.handle,
                        /* norm_out = */ self.batch_norm.handle,
                        hidden, seq_len, cfg.rms_norm_eps, "add_rms_attn_next_b",
                    );
                }
                None => {
                    self.run_binary(
                        dev, registry, cmd, ShaderId::Add,
                        self.batch_residual.handle, self.batch_ffn_out.handle,
                        self.batch_residual.handle,
                        seq_len * hidden, "add_res2_b",
                    );
                }
            }
            compute_barrier(dev, cmd);
        }
        // Sprint 43F sub-bisect — Stage 5: layer output (post-residual2).
        stage_dump(self, cmd, self.batch_residual.handle, 5);
        let _ = (kv_bytes, ffn_bytes); // some bytes locals only used by debug paths
    }
}
