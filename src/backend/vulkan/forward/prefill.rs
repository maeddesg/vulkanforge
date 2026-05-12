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
use super::super::gguf::ModelConfig;
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;

use super::arch::{compute_barrier, layer_weight, transfer_to_compute_barrier};
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
        // Sprint 46D — per-token IDs for Gemma-4 PLE pre-stage. Length
        // must equal `seq_len`. Other architectures pass an empty slice
        // (or any slice of correct length) and the PLE pre-stage is a
        // no-op (gated by `model.ple_data.is_some()`).
        token_ids: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !token_ids.is_empty() && (token_ids.len() as u32) != seq_len {
            return Err(format!(
                "prefill_batch: token_ids.len() {} != seq_len {seq_len}",
                token_ids.len()
            ).into());
        }
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

        // CPU → batch_input (host-visible). Sprint 46F — apply Gemma-4
        // `embed_scale = sqrt(hidden_size)` here, mirroring the decode
        // path (decode.rs:197-201). Without this the first decoder
        // layer of the Gemma-4 batch path would see un-scaled
        // embeddings — the v0.3.14 force_per_token workaround masked
        // this because forward_token applied the scale before its own
        // initial-embedding write.
        let scaled_embeddings: Option<Vec<f32>> = cfg.gemma4.as_ref().map(|g| {
            let s = g.embed_scale;
            embeddings.iter().map(|v| v * s).collect()
        });
        let batch_input_src: &[f32] = scaled_embeddings
            .as_deref()
            .unwrap_or(embeddings);
        self.batch_input.write_bytes(bytemuck::cast_slice(batch_input_src))?;

        // Pre-stage RoPE positions for every token in the batch.
        // CRITICAL: all GPU dispatches in this submit run AFTER all
        // host writes complete, so we must write every per-token
        // position into a separate slot of rope_pos_buf BEFORE we
        // start recording — otherwise the per-token RoPE dispatches
        // would all read the last-written value (Phase 3E drift bug).
        let positions: Vec<u32> = (0..seq_len).map(|t| base_pos + t).collect();
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::cast_slice(&positions))?;

        // Sprint 46D — Gemma-4 PLE pre-stage. `build_per_layer_inputs`
        // expects the *scaled* embedding (`embed_per_layer * sqrt(hidden)`)
        // — the same input `forward_token` feeds it on the decode path
        // (decode.rs:204). For each prefill token t we build the
        // `[num_layers × hps]` slice and pack them contiguously into
        // `per_layer_inputs` at offset `t * num_layers * hps * 4`.
        // The buffer was resized to `max_prefill_tokens × num_layers ×
        // hps × 4` in Sprint 46D step 1, so all M slots fit.
        //
        // Host-visible writes happen NOW (before CB record + submit).
        // Per the Sprint 46C blocker analysis, building the slot inside
        // a per-CB-step loop wouldn't work because CPU writes during
        // record are not serialised with GPU reads.
        //
        // No-op for non-Gemma-4 / Gemma-4-without-PLE (the buffer is a
        // 4-byte placeholder in those cases — see setup.rs).
        if let Some(_g) = cfg.gemma4.as_ref() {
            if let Some(ple) = model.ple_data.as_ref() {
                let h = hidden as usize;
                let nl = cfg.n_layers as usize;
                let hps = _g.hidden_size_per_layer_input as usize;
                let row_bytes = (nl * hps * 4) as u64;
                if (token_ids.len() as u32) != seq_len {
                    return Err(format!(
                        "prefill_batch: Gemma-4 PLE requires token_ids.len() == seq_len; \
                         got {} vs {seq_len}",
                        token_ids.len()
                    ).into());
                }
                // Sprint 46F — reuse the already-scaled embedding from
                // `scaled_embeddings` (built above for the batch_input
                // copy). Single source of truth means the decode-path
                // per-token PLE and the batch-path per-token PLE see
                // identical bits for the same token.
                let scaled_src = scaled_embeddings.as_deref()
                    .expect("Gemma-4 path missing scaled_embeddings");
                for t in 0..seq_len as usize {
                    let scaled = &scaled_src[t * h..(t + 1) * h];
                    let v = ple.build_per_layer_inputs(token_ids[t], scaled);
                    let off = (t as u64) * row_bytes;
                    self.cur_mut().per_layer_inputs
                        .write_bytes_at(off, bytemuck::cast_slice(&v))?;
                }
            }
        }

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
        // Sprint 51D-AN — Gemma-4 Full-attention layers route through
        // the proven DEC per-token path; Sliding layers keep the fast
        // batched dispatch. Workaround for the multiple parallel bugs
        // in BAT Full-layer dispatch (51D-AK flash_attn_batch attention
        // + 51D-AM FFN/PostAttnNorm). Cost: 5 of 30 layers serialise
        // per prefill (~18 mid-frame submits per Full layer).
        let is_full = self.config.gemma4.as_ref()
            .map(|g| matches!(
                g.layers[layer as usize].kind,
                super::super::gguf::Gemma4LayerKind::Full,
            ))
            .unwrap_or(false);
        if is_full {
            self.dispatch_full_layer_per_token(
                dev, registry, cmd, model, layer, seq_len, base_pos,
            );
            return;
        }
        let plan = super::layer_plan::build_layer_plan(
            &self.config, model, layer, self.rope_theta_scale,
        );
        let ctx = super::executor::ExecCtx {
            dev, registry, cmd, model, layer,
            mode: super::executor::ExecMode::Batch {
                seq_len, base_pos, next_attn_norm_weight,
            },
        };
        let exec = super::executor::BatchExec;
        exec.execute_layer(self, &plan, &ctx);
    }

    /// Sprint 51D-AN — per-token DEC dispatch for one Full layer of a
    /// BAT prefill batch. For each token in the batch:
    ///   1. mid-frame submit-and-wait drains prior GPU work and lets
    ///      us host-write rope_pos_buf[0] with this token's position;
    ///   2. cmd_copy_buffer brings `batch_residual[q_idx-row]` into
    ///      `cur().scratch_a` (the DEC layer's input slot);
    ///   3. `dispatch_layer` runs the proven DEC pipeline for this
    ///      single (layer, position) pair, writing `cur().scratch_b`;
    ///   4. cmd_copy_buffer writes back into the same row of
    ///      `batch_residual` so the next BAT layer sees the post-Full
    ///      hidden state in its expected layout.
    ///
    /// The mid-frame submit between iterations is the only mechanism
    /// that lets us update `rope_pos_buf[0]` host-side so each per-token
    /// DEC dispatch sees the right RoPE position (DEC RoPE binds
    /// `rope_pos_buf` at offset 0; the BAT-side pre-stage at prefill
    /// start writes positions 0..seq_len-1 contiguously, but DEC only
    /// reads the first slot).
    fn dispatch_full_layer_per_token(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        seq_len: u32,
        base_pos: u32,
    ) {
        let hidden_dim = self.config.hidden_dim;
        let hidden_bytes = (hidden_dim as u64) * 4;

        for q_idx in 0..seq_len {
            let position = base_pos + q_idx;
            let src_off = (q_idx as u64) * hidden_bytes;

            // 1. Drain prior commands so we can host-write rope_pos_buf.
            self.mid_frame_submit_and_wait(dev, cmd)
                .expect("51D-AN mid_frame_submit_and_wait failed");

            // 2. Host-write the per-token RoPE position. DEC's
            //    `run_rms_norm_mul_rope` binds rope_pos_buf at offset 0
            //    with size = m*4 = 4 (single token), so writing the first
            //    4 bytes is sufficient.
            self.cur_mut().rope_pos_buf
                .write_bytes(bytemuck::bytes_of(&position))
                .expect("51D-AN rope_pos_buf write failed");

            // 3. Copy batch_residual[q_idx row] → cur().scratch_a (DEC input).
            let scratch_a = self.cur().scratch_a.handle;
            let scratch_b = self.cur().scratch_b.handle;
            let batch_residual = self.batch_residual.handle;
            let copy_in = vk::BufferCopy::default()
                .src_offset(src_off).dst_offset(0).size(hidden_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, batch_residual, scratch_a,
                    std::slice::from_ref(&copy_in),
                );
            }
            transfer_to_compute_barrier(dev, cmd);

            // 4. Run the DEC layer (Q/K/V proj + norm/rope + attention +
            //    O-proj + FFN/MoE + residual + layer-scalar) for one
            //    token at `position`.
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position, scratch_a, scratch_b,
            );

            // 5. Copy scratch_b → batch_residual[q_idx row].
            //    Need a COMPUTE→TRANSFER barrier first: dispatch_layer's
            //    final step writes scratch_b (shader_write); the upcoming
            //    cmd_copy_buffer reads it (transfer_read).
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&mb), &[], &[],
                );
            }
            let copy_out = vk::BufferCopy::default()
                .src_offset(0).dst_offset(src_off).size(hidden_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, scratch_b, batch_residual,
                    std::slice::from_ref(&copy_out),
                );
            }
            // Sync TRANSFER write to batch_residual for any subsequent
            // compute reads (within this CB; the next iteration starts
            // fresh after mid-frame submit).
            transfer_to_compute_barrier(dev, cmd);
        }

        // Restore the BAT-shape pre-stage of rope_pos_buf so the next
        // (Sliding) BAT layer sees the full [base_pos, base_pos+1, …]
        // sequence in slot positions 0..seq_len-1. Without this restore,
        // rope_pos_buf[0] retains the last per-token DEC position
        // (= base_pos + seq_len - 1) and the next Sliding layer's
        // batch RoPE reads a corrupted token-0 position.
        self.mid_frame_submit_and_wait(dev, cmd)
            .expect("51D-AN final mid_frame_submit failed");
        let positions: Vec<u32> = (0..seq_len).map(|t| base_pos + t).collect();
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::cast_slice(&positions))
            .expect("51D-AN rope_pos_buf restore failed");
    }
}
