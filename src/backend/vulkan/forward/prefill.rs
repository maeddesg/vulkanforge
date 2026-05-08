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
}
