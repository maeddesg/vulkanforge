//! Sprint 44B-4 — decode-path orchestration extracted from
//! `forward/mod.rs`. Pure code-move.
//!
//! Owns the per-token forward path:
//! - Entry points: `forward_token`, `forward_token_profile`,
//!   `forward_token_profile_layers`, `logits` (re-read after a forward).
//! - Async-decode pipeline: `pre_record`, `fill_embed_and_submit`,
//!   `wait_and_read_logits`, plus the shared inner loop
//!   `record_decode_dispatches`.
//! - Per-layer dispatchers: `dispatch_layer` (decode), `dispatch_final`
//!   (final norm + lm_head, used by both decode and prefill).
//! - `record_logits_readback`: tiny end-of-CB copy from `logits_buf` to
//!   `logits_staging`, called from every dispatch_final variant.

use std::time::Duration;

use ash::vk;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::super::gguf::GgmlType;
use super::super::loader::LoadedModel;
use super::super::pipeline::SwigluPushConstants;
use super::super::pipeline_registry::PipelineRegistry;
use super::super::profiler::ShaderProfiler;
use super::super::shaders::ShaderId;

use super::arch::{
    apply_final_logit_softcap, gemma4_layer_owns_kv, layer_dims, layer_weight,
    layer_weight_opt, layer_weight_scale_block, layer_weight_scale_buf,
    layer_weight_scale_scalar, layer_weight_shader, rope_params_for_layer,
};
use super::debug::{maybe_dump_hidden_staging, maybe_dump_logits};
use super::state::{Forward, ForwardStats, ForwardTokenProfile};

impl Forward {
    /// Sprint 15E — record one decode forward (36 layers + lm_head)
    /// into `cmd`, reading buffers from `slots[slot]`. Used by both
    /// the existing serial path (called inside `cmd_ctx.one_shot`'s
    /// closure with `slot = self.current_slot`) and the async
    /// path (`pre_record` calls this on a slot-specific CB before
    /// the embedding has been written, since vkCmd* records buffer
    /// HANDLES not contents).
    ///
    /// Caller is responsible for: descriptor-pool reset (when not
    /// cache_enabled), barrier-elision state reset, profiler reset,
    /// and the surrounding begin/end CB calls.
    fn record_decode_dispatches(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        slot: usize,
        position: u32,
    ) {
        // Temporarily switch current_slot so dispatch_layer / dispatch_final
        // (which read self.cur() internally) hit the right slot.
        let saved = self.current_slot;
        self.current_slot = slot;

        let mut input = self.cur().scratch_a.handle;
        let mut output = self.cur().scratch_b.handle;
        // Sprint 43D Bisect — VF_LAYER_DUMP=N copies the hidden state
        // *after* layer N into hidden_staging so the caller can read +
        // dump it after the one_shot. N=999 means "after the final
        // layer, before dispatch_final" (= input to the final-norm).
        let dump_layer: i32 = std::env::var("VF_LAYER_DUMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1);
        // Sprint 43D-4 — VF_LAYER_DUMP_ALL=1 copies every layer's output
        // into hidden_staging at slot[layer], producing 35 (or whatever
        // n_layers is) consecutive `hidden_dim`-element FP32 chunks. The
        // host reads them after wait_and_read_logits via the existing
        // `hidden_staging.read_bytes()` path.
        let dump_all_layers = std::env::var("VF_LAYER_DUMP_ALL").is_ok();
        let bytes_per_slot = (self.config.hidden_dim as u64) * 4;
        for layer in 0..self.config.n_layers {
            self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
            let want_dump = dump_layer == (layer as i32) || dump_all_layers;
            if want_dump {
                // VF_LAYER_DUMP=N writes to slot 0; VF_LAYER_DUMP_ALL=1
                // writes to slot[layer]. The two modes don't intersect —
                // when ALL is set, slot 0 holds layer 0 anyway, so a
                // `VF_LAYER_DUMP=0` test still resolves to slot 0.
                let dst_slot = if dump_all_layers { layer as u64 } else { 0 };
                let copy = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(dst_slot * bytes_per_slot)
                    .size(bytes_per_slot);
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
                        cmd, output, self.hidden_staging.handle,
                        std::slice::from_ref(&copy),
                    );
                }
            }
            std::mem::swap(&mut input, &mut output);
        }
        if dump_layer == 999 {
            let bytes = (self.config.hidden_dim as u64) * 4;
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
                    cmd, input, self.hidden_staging.handle,
                    std::slice::from_ref(&copy),
                );
            }
        }
        self.dispatch_final(dev, registry, cmd, model, input);

        // Sprint 27 — copy logits from GpuOnly logits_buf to host-mapped
        // logits_staging, with surrounding compute→transfer→host barriers.
        self.record_logits_readback(dev, cmd);

        self.current_slot = saved;
    }

    /// Sprint 15E Stage 1 — pre-record CB[slot] for the given
    /// `position`. Buffer handles are referenced (slots[slot].*) but
    /// the embedding contents are NOT yet written to scratch_a; the
    /// caller writes them via `fill_embed_and_submit` after sampling
    /// determines the token. Designed to run on the CPU in parallel
    /// with the GPU executing the previous token's CB.
    pub fn pre_record(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        model: &LoadedModel,
        slot: usize,
        position: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cb = self.async_cbs[slot];
        let fence = self.async_fences[slot];
        // The fence may still be signaled from a previous token's GPU
        // completion — that's fine, we wait once more (it's a no-op
        // if the fence is already signaled at start of session).
        unsafe {
            dev.device.wait_for_fences(&[fence], true, u64::MAX)?;
            dev.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.device.begin_command_buffer(cb, &begin)?;
        }
        // Sprint 12D barrier-elision tracker — reset to fresh state
        // for this CB.
        self.reset_barrier_state();
        self.record_decode_dispatches(dev, registry, cb, model, slot, position);
        unsafe { dev.device.end_command_buffer(cb)?; }
        self.async_pending_record = Some(slot);
        Ok(())
    }

    /// Sprint 15E Stage 2/3 — write the embedding for this token into
    /// slot's scratch_a and rope_pos, then submit the pre-recorded CB.
    /// Caller must have called `pre_record(slot, position)` already.
    pub fn fill_embed_and_submit(
        &mut self,
        dev: &VulkanDevice,
        slot: usize,
        embedding: &[f32],
        position: u32,
        model: &LoadedModel,
        token_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(), self.config.hidden_dim
            ).into());
        }
        // Sprint 43D-3 + 43D-4 — Gemma-4 embed_scale (= sqrt(hidden_size))
        // applied to the initial token embedding, plus per-token PLE
        // build (token_identity + context_proj merge). Without this the
        // async decode path (Sprint 15E pipeline) bypassed `forward_token`'s
        // Gemma-4 prep and per_layer_inputs would hold stale values from
        // the last prefill token — causing the first decode token to
        // sample on a half-correct hidden state.
        if let Some(g) = self.config.gemma4.as_ref() {
            let s = g.embed_scale;
            let mut scaled = Vec::with_capacity(embedding.len());
            for &v in embedding {
                scaled.push(v * s);
            }
            self.slots[slot].scratch_a.write_bytes(bytemuck::cast_slice(&scaled))?;
            if let Some(ple) = model.ple_data.as_ref() {
                let v = ple.build_per_layer_inputs(token_id, &scaled);
                self.slots[slot]
                    .per_layer_inputs
                    .write_bytes(bytemuck::cast_slice(&v))?;
            }
        } else {
            self.slots[slot].scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        }
        self.slots[slot].rope_pos_buf.write_bytes(bytemuck::bytes_of(&(position as u32)))?;

        let cb = self.async_cbs[slot];
        let fence = self.async_fences[slot];
        unsafe {
            dev.device.reset_fences(&[fence])?;
            let cmds = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.device.queue_submit(dev.compute_queue, &[submit], fence)?;
        }
        self.async_pending_record = None;
        Ok(())
    }

    /// Sprint 15E Stage 4 — block on slot's fence and read logits from
    /// the (single) `logits_buf`. Must be called BEFORE the next
    /// submit hits the same logits_buf (i.e. before the next call to
    /// `fill_embed_and_submit`). In the 3-stage pipeline, the order is
    /// pre_record(N+1) → wait_and_read_logits(N) → sample → submit(N+1),
    /// so logits_buf is read while no CB is in flight on logits_buf.
    pub fn wait_and_read_logits(
        &mut self,
        dev: &VulkanDevice,
        slot: usize,
        model: &LoadedModel,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let fence = self.async_fences[slot];
        unsafe { dev.device.wait_for_fences(&[fence], true, u64::MAX)?; }
        // Sprint 40 Part 2 — when CPU lm_head is active, the GPU
        // submit copied the post-norm hidden state into
        // `hidden_staging` (via `dispatch_final`) and skipped the
        // GPU lm_head GEMV. Run the Q6_K GEMV on the CPU now and
        // return its logits directly, bypassing `logits_staging`.
        if let Some(ref cpu_lm) = model.cpu_lm_head {
            let hidden_n = self.config.hidden_dim as usize;
            let hidden: Vec<f32> = {
                let bytes = self.hidden_staging.read_bytes()?;
                bytemuck::cast_slice::<u8, f32>(&bytes[..hidden_n * 4]).to_vec()
            };
            maybe_dump_hidden_staging("wait_and_read_logits/cpu_lm_head", &hidden);
            let mut logits = vec![0.0_f32; cpu_lm.vocab_size];
            cpu_lm.forward(&hidden, &mut logits);
            maybe_dump_logits("cpu_lm_head pre-softcap", &logits);
            apply_final_logit_softcap(&self.config, &mut logits);
            maybe_dump_logits("cpu_lm_head post-softcap", &logits);
            return Ok(logits);
        }
        // Sprint 43F Block A§3 — dump hidden_staging contents in
        // wait_and_read_logits (the *actual* prod read path; the
        // earlier `forward_token` dump was unreachable for the async
        // pipeline). When VF_FINAL_NORM_DUMP=1 the recorded CB has
        // copied hidden_norm into hidden_staging at the position
        // BEFORE lm_head; reading it here shows what hidden_norm
        // ACTUALLY held at that point of the GPU execution timeline.
        if std::env::var("VF_FINAL_NORM_DUMP").is_ok()
            || std::env::var("VF_LAYER_DUMP").is_ok()
            || std::env::var("VF_BATCH_INPUT_DUMP").is_ok()
        {
            let h = self.config.hidden_dim as usize;
            let bytes = self.hidden_staging.read_bytes()?;
            let hidden: &[f32] = bytemuck::cast_slice::<u8, f32>(&bytes[..h * 4]);
            maybe_dump_hidden_staging("wait_and_read_logits/gpu_lm_head", hidden);
        }
        // Sprint 43D-4 — VF_LAYER_DUMP_ALL=1 dumps all-layer hidden states
        // staged earlier in record_decode_dispatches. Each layer N's
        // post-dispatch_layer output occupies slot N (= bytes
        // [N*h*4, (N+1)*h*4)) of the hidden_staging buffer (Sprint 43D-4
        // bumped capacity to 64 slots). When `VF_LAYER_DUMP_OUT=path`
        // is set, the full N_LAYERS×hidden FP32 blob is appended to
        // `path` (binary, little-endian); otherwise per-layer stats are
        // printed to stderr.
        if std::env::var("VF_LAYER_DUMP_ALL").is_ok() {
            let h = self.config.hidden_dim as usize;
            let nl = self.config.n_layers as usize;
            let bytes = self.hidden_staging.read_bytes()?;
            let need = nl * h * 4;
            if bytes.len() < need {
                eprintln!(
                    "[VF_LAYER_DUMP_ALL] WARNING: hidden_staging only {} bytes, need {} \
                     (n_layers={nl} × hidden={h} × 4) — dumping what fits",
                    bytes.len(), need,
                );
            }
            let out_path = std::env::var("VF_LAYER_DUMP_OUT").ok();
            if let Some(path) = out_path {
                let take = bytes.len().min(need);
                std::fs::write(&path, &bytes[..take])?;
                eprintln!(
                    "[VF_LAYER_DUMP_ALL] wrote {} bytes ({} layers × {} × 4) to {}",
                    take, take / (h * 4), h, path,
                );
            } else {
                for layer in 0..nl {
                    let off = layer * h * 4;
                    if off + h * 4 > bytes.len() { break; }
                    let slice: &[f32] =
                        bytemuck::cast_slice::<u8, f32>(&bytes[off..off + h * 4]);
                    maybe_dump_hidden_staging(&format!("layer{layer:02}"), slice);
                }
            }
        }
        if std::env::var("VF_BATCH_STEP_DUMP").as_deref() == Ok("ALL") {
            // Sprint 43F sub-bisect — read 6 hidden_staging slots, one per
            // recorded stage in dispatch_layer_batch for layer 0.
            let h = self.config.hidden_dim as usize;
            let bytes = self.hidden_staging.read_bytes()?;
            let labels = [
                "BATCH-L0-0 entry batch_residual",
                "BATCH-L0-1 entry batch_norm (post-attn-norm pre-seed)",
                "BATCH-L0-2 post Q/K/V GEMM (pre-RoPE batch_q)",
                "BATCH-L0-3 post-RoPE batch_q",
                "BATCH-L0-4 post-attention batch_attn_out",
                "BATCH-L0-5 post-residual2 batch_residual (layer output)",
                "BATCH-L0-6 post (f+g) batch_norm (= MLP input)",
                "BATCH-L0-7 post (f+g) batch_residual (post attn-residual)",
                "BATCH-L0-8 post-MLP batch_ffn_out (entry to (l) fork)",
                "BATCH-L0-9 post (l) step1 = ffn_normed scratch",
                "BATCH-L0-10 post-Gate GEMM (batch_gate)",
                "BATCH-L0-11 post-Up GEMM (batch_up)",
                "BATCH-L0-12 post-SwiGLU (batch_ffn_hidden)",
            ];
            for stage in 0..13 {
                let off = stage * h * 4;
                let slice: &[f32] =
                    bytemuck::cast_slice::<u8, f32>(&bytes[off..off + h * 4]);
                maybe_dump_hidden_staging(labels[stage], slice);
            }
        }
        // Sprint 27 — read from host-mapped staging copy.
        let bytes = self.logits_staging.read_bytes()?;
        let mut logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        ).to_vec();
        maybe_dump_logits("gpu_lm_head pre-softcap", &logits);
        apply_final_logit_softcap(&self.config, &mut logits);
        maybe_dump_logits("gpu_lm_head post-softcap", &logits);
        Ok(logits)
    }


    /// Phase 5A-2 drill-down: same wall-time semantics as
    /// `forward_token_profile` but additionally captures, INSIDE the
    /// command-buffer record block, per-layer wall time and a tally of
    /// `dispatch_final`'s wall time. Use this to decide whether
    /// command-buffer reuse should target the Rust-side per-layer
    /// setup (HashMap lookup, push-constants struct build) or the raw
    /// `vkCmd*` call cost.
    pub fn forward_token_profile_layers(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
    ) -> Result<(ForwardTokenProfile, Vec<Duration>, Duration), Box<dyn std::error::Error>> {
        use std::time::Instant;
        let pre_start = Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err("embedding length mismatch".into());
        }
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Phase 5A-2 Stage 2D: skip the per-forward pool reset when
        // CB-reuse is on — cached sets stay alive across tokens.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }
        // Sprint 25 — fp8pc never caches sets (run_gemv_fp8_perchannel
        // allocates fresh per call), so its pool must be reset every
        // forward regardless of cache_enabled. Without this, multi-prompt
        // bench (~336 sets/token × hundreds of tokens) overflows the
        // 8192-set pool from Sprint 24-Inline.
        // Sprint 30 — fp8pc_pool no longer reset per decode token.
        // The descriptor-set cache (`fp8pc_ds_cache`) keeps sets warm
        // across tokens; resetting the pool here would invalidate
        // them and force fresh allocation every forward. Reset
        // happens at prefill via `reset_descriptor_pool_and_cache`.
        unsafe {
            // Sprint 29 — lm_head pool stays per-forward-resetted (its
            // bindings change with the cur() slot, no caching applied).
            dev.device.reset_descriptor_pool(
                self.lmhead.pool, vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        let pre_setup = pre_start.elapsed();

        let n_layers = self.config.n_layers as usize;
        let mut per_layer: Vec<Duration> = Vec::with_capacity(n_layers);
        let mut final_dispatch = Duration::ZERO;

        // Sprint 12D — fresh barrier-elision state per forward.
        self.reset_barrier_state();
        let timings = cmd_ctx.one_shot_profiled(&dev.device, dev.compute_queue, |cmd| {
            let mut input = self.cur().scratch_a.handle;
            let mut output = self.cur().scratch_b.handle;
            for layer in 0..self.config.n_layers {
                let t = Instant::now();
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                per_layer.push(t.elapsed());
                std::mem::swap(&mut input, &mut output);
            }
            let t = Instant::now();
            self.dispatch_final(dev, registry, cmd, model, input);
            // Sprint 27 — copy logits to host-mapped staging.
            self.record_logits_readback(dev, cmd);
            final_dispatch = t.elapsed();
        })?;

        let read_start = Instant::now();
        let bytes = self.logits_staging.read_bytes()?;
        let _logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        let readback = read_start.elapsed();

        self.kv_cache.current_seq_len = position + 1;

        let profile = ForwardTokenProfile {
            pre_setup,
            reset: timings.reset,
            begin: timings.begin,
            record: timings.record,
            end: timings.end,
            submit: timings.submit,
            gpu_wait: timings.wait,
            readback,
        };
        Ok((profile, per_layer, final_dispatch))
    }

    /// Like [`forward_token`] but returns a CPU-time breakdown
    /// (host setup / record / submit / GPU-wait / readback). Phase-5A
    /// profiling: feeds the "where do the 3.3 ms go" question with
    /// real numbers so the optimisation target is data-driven.
    pub fn forward_token_profile(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
    ) -> Result<ForwardTokenProfile, Box<dyn std::error::Error>> {
        use std::time::Instant;
        let pre_start = Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(),
                self.config.hidden_dim
            )
            .into());
        }
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Phase 5A-2 Stage 2D: skip the per-forward pool reset when
        // CB-reuse is on — cached sets stay alive across tokens.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }
        // Sprint 25 — fp8pc never caches sets; reset its pool every forward.
        // Sprint 30 — fp8pc_pool no longer reset per decode token.
        // The descriptor-set cache (`fp8pc_ds_cache`) keeps sets warm
        // across tokens; resetting the pool here would invalidate
        // them and force fresh allocation every forward. Reset
        // happens at prefill via `reset_descriptor_pool_and_cache`.
        unsafe {
            // Sprint 29 — lm_head pool stays per-forward-resetted (its
            // bindings change with the cur() slot, no caching applied).
            dev.device.reset_descriptor_pool(
                self.lmhead.pool, vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        let pre_setup = pre_start.elapsed();

        // Sprint 12D — fresh barrier-elision state per forward.
        self.reset_barrier_state();
        let timings = cmd_ctx.one_shot_profiled(&dev.device, dev.compute_queue, |cmd| {
            let mut input = self.cur().scratch_a.handle;
            let mut output = self.cur().scratch_b.handle;
            for layer in 0..self.config.n_layers {
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                std::mem::swap(&mut input, &mut output);
            }
            self.dispatch_final(dev, registry, cmd, model, input);
            // Sprint 27 — copy logits to host-mapped staging.
            self.record_logits_readback(dev, cmd);
        })?;

        let read_start = Instant::now();
        let bytes = self.logits_staging.read_bytes()?;
        let _logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        let readback = read_start.elapsed();

        self.kv_cache.current_seq_len = position + 1;

        Ok(ForwardTokenProfile {
            pre_setup,
            reset: timings.reset,
            begin: timings.begin,
            record: timings.record,
            end: timings.end,
            submit: timings.submit,
            gpu_wait: timings.wait,
            readback,
        })
    }

    /// One decode step: writes `embedding` (length = hidden_dim) into
    /// the input slot, runs all 36 layers + final norm + LM head at
    /// the given `position`, and reads the logits back. Caller is
    /// responsible for the embedding lookup (CPU dequant of the GGUF
    /// `token_embd.weight` row).
    pub fn forward_token(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
        token_id: u32,
    ) -> Result<ForwardStats, Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        if std::env::var("VF_TRACE_FT").is_ok() {
            eprintln!("[VF_TRACE_FT] forward_token entry pos={position}");
        }
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(),
                self.config.hidden_dim
            )
            .into());
        }
        // Sprint 43D-3 — Gemma-4 embedding scale: HF
        // `Gemma4TextModel.forward` does
        // `hidden_states = inputs_embeds * sqrt(hidden_size)` before the
        // first decoder layer. `Gemma4Spec.embed_scale` was parsed in
        // 43B-2 but never applied — without this multiplier the initial
        // hidden state is ~40× too small (sqrt(1536) ≈ 39.2 on E2B),
        // and every downstream RMSNorm runs against the wrong magnitude.
        // Non-Gemma-4 architectures (Llama / Qwen / Mistral) write the
        // embedding through unmodified — same as before.
        //
        // Sprint 43D-4 — the scaled vector is also fed to the PLE build
        // as `inputs_embeds` for the context-aware projection
        // (per_layer_model_projection @ inputs_embeds → ctx_proj).
        if let Some(g) = self.config.gemma4.as_ref() {
            let s = g.embed_scale;
            let mut scaled = Vec::with_capacity(embedding.len());
            for &v in embedding {
                scaled.push(v * s);
            }
            self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(&scaled))?;
            // Pre-write the RoPE position buffer (Gemma-4 path).
            self.cur_mut().rope_pos_buf
                .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
            // Sprint 43D-4 — Gemma-4 PLE build:
            //   per_layer_inputs = (ctx_proj_normed + token_identity) × 1/√2
            // ctx_proj depends on the *scaled* embedding (HF's
            // `inputs_embeds`), so feed `scaled` directly. Skipped for
            // models without ple_data (= non-Gemma-4 or any Gemma-4
            // checkpoint that lacks the PLE tensors).
            if let Some(ple) = model.ple_data.as_ref() {
                let v = ple.build_per_layer_inputs(token_id, &scaled);
                self.cur_mut().per_layer_inputs.write_bytes(bytemuck::cast_slice(&v))?;
            }
        } else {
            self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
            // Pre-write the RoPE position buffer (non-Gemma-4 path).
            self.cur_mut().rope_pos_buf
                .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        }

        // Reset descriptor pool for fresh allocations this forward —
        // skipped when CB-reuse is on; cached sets stay valid.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }
        // Sprint 25 — fp8pc never caches sets; reset its pool every forward.
        // Sprint 30 — fp8pc_pool no longer reset per decode token.
        // The descriptor-set cache (`fp8pc_ds_cache`) keeps sets warm
        // across tokens; resetting the pool here would invalidate
        // them and force fresh allocation every forward. Reset
        // happens at prefill via `reset_descriptor_pool_and_cache`.
        unsafe {
            // Sprint 29 — lm_head pool stays per-forward-resetted (its
            // bindings change with the cur() slot, no caching applied).
            dev.device.reset_descriptor_pool(
                self.lmhead.pool, vk::DescriptorPoolResetFlags::empty(),
            )?;
        }

        // Pre-snapshot: we'll record per-layer profile boundaries.
        let mut per_layer_starts: Vec<usize> = Vec::with_capacity(self.config.n_layers as usize);

        // Sprint 12D — fresh barrier-elision state per forward.
        self.reset_barrier_state();
        let cur_slot = self.current_slot;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            if let Some(p) = self.profiler.as_mut() {
                p.reset(&dev.device, cmd);
            }
            // (Sprint 15E — share the same recording body with the async
            // path. The serial path skips per-layer profile-boundary
            // capture; the async path doesn't use the profiler.)
            for _ in 0..self.config.n_layers {
                if let Some(p) = self.profiler.as_ref() {
                    per_layer_starts.push(p.entries_len());
                }
            }
            self.record_decode_dispatches(dev, registry, cmd, model, cur_slot, position);
        })?;

        // Sprint 43D-4 — VF_LAYER_DUMP_ALL=1 in the forward_token path
        // (force_per_token prefill). Mirrors the wait_and_read_logits
        // version so SafeTensors models that route through forward_token
        // get the same all-layer dump on the LAST prefill token. The
        // VF_LAYER_DUMP_OUT path is overwritten on every call — drivers
        // running multi-token prompts will get the LAST token's layer
        // outputs (which is exactly what HF's
        // output_hidden_states[-1, last_pos, :] compares against).
        //
        // VF_DUMP_LAYER34_STAGES=1 additionally fills slots 36..42 with
        // 7 intra-layer-34 stages — extend the read window so they land
        // in the output blob too. Layout becomes:
        //   slot 0..n_layers-1 = post-layer-N hidden state
        //   slot 35            = (unused; we write to 36..42)
        //   slot 36..42        = 7 stages of layer 34 (1..=7)
        if std::env::var("VF_LAYER_DUMP_ALL").is_ok() {
            let h = self.config.hidden_dim as usize;
            let nl = self.config.n_layers as usize;
            let bytes = self.hidden_staging.read_bytes()?;
            let need_layers = nl * h * 4;
            let stage_slot_max: usize =
                if std::env::var("VF_DUMP_LAYER34_STAGES").is_ok() { 43 } else { nl };
            let need = stage_slot_max * h * 4;
            let take = bytes.len().min(need.max(need_layers));
            if let Some(path) = std::env::var("VF_LAYER_DUMP_OUT").ok() {
                std::fs::write(&path, &bytes[..take])?;
                eprintln!(
                    "[VF_LAYER_DUMP_ALL] forward_token wrote {} bytes ({} slots × {} × 4) to {}",
                    take, take / (h * 4), h, path,
                );
            } else {
                for layer in 0..nl {
                    let off = layer * h * 4;
                    if off + h * 4 > take { break; }
                    let slice: &[f32] =
                        bytemuck::cast_slice::<u8, f32>(&bytes[off..off + h * 4]);
                    maybe_dump_hidden_staging(&format!("forward_token_layer{layer:02}"), slice);
                }
            }
        }

        // Sprint 43D Bisect — read + dump hidden_staging if VF_LAYER_DUMP
        // selected a layer or VF_FINAL_NORM_DUMP is on. Logged once per
        // forward_token call.
        if std::env::var("VF_LAYER_DUMP").is_ok()
            || std::env::var("VF_FINAL_NORM_DUMP").is_ok()
        {
            let h = self.config.hidden_dim as usize;
            let bytes = self.hidden_staging.read_bytes()?;
            let hidden: &[f32] = bytemuck::cast_slice::<u8, f32>(&bytes[..h * 4]);
            let mut nan = 0u32; let mut inf = 0u32;
            let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
            let mut sum: f64 = 0.0;
            for &v in hidden {
                if v.is_nan() { nan += 1; continue; }
                if v.is_infinite() { inf += 1; continue; }
                if v < mn { mn = v; }
                if v > mx { mx = v; }
                sum += v as f64;
            }
            let valid = (h as u32) - nan - inf;
            let mean = if valid > 0 { sum / (valid as f64) } else { 0.0 };
            eprintln!(
                "[VF_LAYER_DUMP] hidden post-dispatch: \
                 nan={nan} inf={inf} min={mn:.4} max={mx:.4} mean={mean:.4}"
            );
            eprintln!("[VF_LAYER_DUMP] first16 = {:?}", &hidden[..16]);
        }

        // Sprint 40 Part 2 — CPU lm_head post-pass. `dispatch_final`
        // already copied the post-norm hidden state into
        // `hidden_staging` and skipped the GPU lm_head GEMV. We
        // now run the Q6_K GEMV on the CPU and overwrite
        // `logits_staging` with the result so the existing readback
        // path (next few lines) picks up the right values.
        if let Some(ref cpu_lm) = model.cpu_lm_head {
            let hidden_n = self.config.hidden_dim as usize;
            // Borrow ends at the closure's `}` — we then mutably
            // borrow `self.logits_staging` via `write_bytes`.
            let hidden: Vec<f32> = {
                let bytes = self.hidden_staging.read_bytes()?;
                bytemuck::cast_slice::<u8, f32>(&bytes[..hidden_n * 4]).to_vec()
            };
            let vocab = self.config.vocab_size as usize;
            let mut logits = vec![0.0_f32; vocab];
            cpu_lm.forward(&hidden, &mut logits);
            self.logits_staging
                .write_bytes(bytemuck::cast_slice(&logits))?;
        }

        // Logits readback (Sprint 27 — from host-mapped staging).
        let bytes = self.logits_staging.read_bytes()?;
        let _logits: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        // Move to a stable result via clone — bytes is borrowed.
        let logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();

        // Profiling.
        let samples = if let Some(p) = self.profiler.as_ref() {
            p.collect(&dev.device).unwrap_or_default()
        } else {
            Vec::new()
        };
        let per_shader = ShaderProfiler::aggregate(&samples);
        let mut per_layer: Vec<Duration> = Vec::new();
        if !per_layer_starts.is_empty() && !samples.is_empty() {
            for w in per_layer_starts.windows(2) {
                let (s, e) = (w[0], w[1]);
                if e > s {
                    per_layer.push(samples[s..e].iter().map(|x| x.elapsed).sum());
                }
            }
            // The last layer's slice runs up to (samples.len() - 2 extras).
            let last_start = *per_layer_starts.last().unwrap();
            let last_end = samples.len().saturating_sub(2);
            if last_end > last_start {
                per_layer.push(samples[last_start..last_end].iter().map(|x| x.elapsed).sum());
            }
        }

        // Bump KV state.
        self.kv_cache.current_seq_len = position + 1;

        // Stash logits in a side-channel for the test (avoiding a return-by-value
        // borrowing dance).
        let _ = logits;

        Ok(ForwardStats {
            total: started.elapsed(),
            per_shader,
            per_layer,
            samples,
        })
    }


    /// Read the most recently written logits — call after
    /// `forward_token` returns successfully.
    pub fn logits(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Sprint 27 — read from host-mapped staging copy.
        let bytes = self.logits_staging.read_bytes()?;
        let mut logits: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        // Sprint 43D-4 — match `wait_and_read_logits` semantics: apply
        // the final-logit softcap (Gemma-4) before returning. Without
        // this the prefill-final logits (read via `forward.logits()`
        // after force_per_token loops) skip the softcap that the async
        // decode path applies, so the first sampled token comes off the
        // un-capped distribution while every subsequent decode token
        // comes off the capped one. Matches HF's ordering (softcap is
        // a single output transform, not per-token-conditional).
        maybe_dump_logits("logits()/forward_token-prefill pre-softcap", &logits);
        apply_final_logit_softcap(&self.config, &mut logits);
        maybe_dump_logits("logits()/forward_token-prefill post-softcap", &logits);
        Ok(logits)
    }


    // -------------------------------------------------------------
    // Per-layer + final + helpers.
    // -------------------------------------------------------------

    pub(super) fn dispatch_layer(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input: vk::Buffer,
        output: vk::Buffer,
    ) {
        let cfg = self.config.clone();
        // Sprint 43F Block A — log hidden_norm identity at first /
        // last layer to detect slot-pointer drift.
        if std::env::var("VF_BUF_ID").is_ok()
            && (layer == 0 || layer == cfg.n_layers - 1)
        {
            let hn = self.cur().hidden_norm.handle;
            let hn_size = self.cur().hidden_norm.size;
            eprintln!(
                "[BUF-ID] dispatch_layer entry layer={layer}: slot={} \
                 input={:?} output={:?} hidden_norm={:?} hidden_norm_size={hn_size}",
                self.current_slot, input, output, hn,
            );
        }
        // Sprint 43D-4 — VF_DUMP_LAYER34_STAGES=1 captures 7 intra-layer
        // stages of layer 34 into hidden_staging slots 35..41 (since
        // slots 0..34 hold the all-layer dump). Read out via the
        // existing VF_LAYER_DUMP_ALL = 1 path; this gate just adds the
        // extra cmd_copy_buffer dispatches inside dispatch_layer.
        let dump_l34_stages =
            std::env::var("VF_DUMP_LAYER34_STAGES").is_ok() && layer == cfg.n_layers - 1;
        let hs_handle = self.hidden_staging.handle;
        let hs_size_bytes = (cfg.hidden_dim as u64) * 4;
        let mut stage_dump = |this_dev: &VulkanDevice,
                              cmd_buf: vk::CommandBuffer,
                              stage: u32,
                              src: vk::Buffer| {
            if !dump_l34_stages {
                return;
            }
            // Slots 35..41 (7 stages); offset = (35 + stage) * hidden × 4.
            let dst_off = (35 + stage as u64) * hs_size_bytes;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(hs_size_bytes);
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                this_dev.device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
                this_dev.device.cmd_copy_buffer(
                    cmd_buf, src, hs_handle, std::slice::from_ref(&copy),
                );
            }
        };
        // Sprint 43C — per-layer head_dim / ffn_dim / rope_theta /
        // rotary_dim. Falls back to cfg uniform for non-Gemma-4
        // architectures (Llama / Qwen / etc.) — same numbers as
        // before, no behaviour change on the existing path.
        let (head_dim, ffn_dim, _rope_theta, _rotary_dim) = layer_dims(&cfg, layer);
        // Sprint 43D-2 — per-layer RoPE (head_dim, rotary_dim, freq_base,
        // theta_scale). Gemma-4 sliding: rotary_dim=256, θ=10 000.
        // Gemma-4 full: rotary_dim=128 (= 0.25 × 512), θ=1 000 000. Other
        // archs: uniform cfg values + the precomputed self.rope_theta_scale.
        let (rope_head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale) =
            rope_params_for_layer(&cfg, self.rope_theta_scale, layer);
        let _ = rope_head_dim; // Phys head_dim already captured as `head_dim`.
        let q_dim = cfg.n_heads * head_dim;
        let kv_dim = cfg.n_kv_heads * head_dim;
        // Sprint 43D-2 — KV-share: layers in the Gemma-4 tail subscribe
        // to a publisher's slab instead of computing their own K/V. Skip
        // the K/V GEMV + KV-write for them (saves compute) and route
        // attention reads to the publisher's slab.
        let layer_owns_kv = gemma4_layer_owns_kv(&cfg, layer);

        let q_buf = self.cur().q_buf.handle;
        let k_buf = self.cur().k_buf.handle;
        let v_buf = self.cur().v_buf.handle;
        let hidden_norm = self.cur().hidden_norm.handle;
        let attn_out = self.cur().attn_out.handle;
        let o_buf = self.cur().o_buf.handle;
        let res1 = self.cur().res1.handle;
        let gate_buf = self.cur().gate_buf.handle;
        let up_buf = self.cur().up_buf.handle;
        let ffn_hidden = self.cur().ffn_hidden.handle;
        let ffn_out = self.cur().ffn_out.handle;

        // (a) attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, hidden_norm,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        self.mark_written(&[hidden_norm]);
        // Sprint 43D-4 STAGE 1 — post-attn-norm (= input_layernorm output).
        stage_dump(dev, cmd, 1, hidden_norm);
        // Next: (b) reads hidden_norm.
        self.maybe_compute_barrier(dev, cmd, &[hidden_norm]);

        // (b) Q/K/V projections. Sprint 43D-2 — Gemma-4 SubscribesXxx
        // layers skip the K/V projections (and the KV-cache write below)
        // because attention reads K/V from the publisher's slab. Q is
        // always projected, since Q rotates in the *current* layer's
        // RoPE space — never shared.
        let wq = layer_weight(model, layer, "attn_q.weight");
        let sq = layer_weight_shader(model, layer, "attn_q.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_q = layer_weight_scale_scalar(model, layer, "attn_q.weight");
        let sb_q = layer_weight_scale_buf(model, layer, "attn_q.weight");
        if let Some(s) = sb_q {
            self.run_gemv_fp8_dispatch(dev, cmd, wq, s, hidden_norm, q_buf,
                cfg.hidden_dim, q_dim,
                layer_weight_scale_block(model, layer, "attn_q.weight"), "gemv_q");
        } else {
            self.run_gemv(dev, registry, cmd, sq, wq, hidden_norm, q_buf,
                cfg.hidden_dim, q_dim, scale_q, "gemv_q");
        }
        if layer_owns_kv {
            // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
            // matching GEMV pipeline per tensor's actual ggml_type.
            let wk = layer_weight(model, layer, "attn_k.weight");
            let wv = layer_weight(model, layer, "attn_v.weight");
            let sk = layer_weight_shader(model, layer, "attn_k.weight", self.mul_mat_vec_subgroup_enabled);
            let sv = layer_weight_shader(model, layer, "attn_v.weight", self.mul_mat_vec_subgroup_enabled);
            let scale_k = layer_weight_scale_scalar(model, layer, "attn_k.weight");
            let scale_v = layer_weight_scale_scalar(model, layer, "attn_v.weight");
            let sb_k = layer_weight_scale_buf(model, layer, "attn_k.weight");
            let sb_v = layer_weight_scale_buf(model, layer, "attn_v.weight");
            if let Some(s) = sb_k {
                self.run_gemv_fp8_dispatch(dev, cmd, wk, s, hidden_norm, k_buf,
                    cfg.hidden_dim, kv_dim,
                    layer_weight_scale_block(model, layer, "attn_k.weight"), "gemv_k");
            } else {
                self.run_gemv(dev, registry, cmd, sk, wk, hidden_norm, k_buf,
                    cfg.hidden_dim, kv_dim, scale_k, "gemv_k");
            }
            if let Some(s) = sb_v {
                self.run_gemv_fp8_dispatch(dev, cmd, wv, s, hidden_norm, v_buf,
                    cfg.hidden_dim, kv_dim,
                    layer_weight_scale_block(model, layer, "attn_v.weight"), "gemv_v");
            } else {
                self.run_gemv(dev, registry, cmd, sv, wv, hidden_norm, v_buf,
                    cfg.hidden_dim, kv_dim, scale_v, "gemv_v");
            }
            self.mark_written(&[q_buf, k_buf, v_buf]);
        } else {
            self.mark_written(&[q_buf]);
        }

        // Sprint 25 — Q/K/V bias-add (Qwen2 attention biases). MUST run
        // before RoPE / QK-norm: q_proj.bias is conceptually part of
        // the q_proj linear, so the bias must be folded into q before
        // any positional rotation. Mirrors dispatch_layer_partial
        // (line ~1547) and dispatch_layer_batch (line ~4307); without
        // this block the production decode path drops the biases and
        // produces structured-but-incoherent output on Qwen2.5.
        // Skipped for arches without biases (Llama / Qwen3 / Mistral).
        let q_bias = layer_weight_opt(model, layer, "attn_q.bias");
        let k_bias = layer_weight_opt(model, layer, "attn_k.bias");
        let v_bias = layer_weight_opt(model, layer, "attn_v.bias");
        if q_bias.is_some() || (layer_owns_kv && (k_bias.is_some() || v_bias.is_some())) {
            // GEMV → bias-add hazard: bias reads what GEMV just wrote.
            self.maybe_compute_barrier(dev, cmd, &[q_buf, k_buf, v_buf]);
            if let Some(b) = q_bias {
                self.run_bias_add(dev, registry, cmd, q_buf, b, q_buf, q_dim, 1, "bias_q");
            }
            if layer_owns_kv {
                if let Some(b) = k_bias {
                    self.run_bias_add(dev, registry, cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k");
                }
                if let Some(b) = v_bias {
                    self.run_bias_add(dev, registry, cmd, v_buf, b, v_buf, kv_dim, 1, "bias_v");
                }
            }
            if layer_owns_kv {
                self.mark_written(&[q_buf, k_buf, v_buf]);
            } else {
                self.mark_written(&[q_buf]);
            }
        }

        // Next: (c) reads q_buf, k_buf (or (d) RoPE if no qk_norm).
        self.maybe_compute_barrier(dev, cmd, &[q_buf, k_buf]);

        // (c+d) Q/K norm + RoPE NeoX, fused via rms_norm_mul_rope when
        // the model has Q/K-norm weights (Sprint 12E — same SPV used by
        // dispatch_layer_batch since Sprint 9c.5). Saves 2 dispatches +
        // 1 barrier per layer in decode by collapsing
        // rms_norm_q + rope_q + rms_norm_k + rope_k = 4 dispatches into
        // 2 fused dispatches.
        //
        // Models without `has_qk_norm` keep the original
        // separate-rope-only path (no norm step to fuse with).
        //
        // Sprint 43D-2 — pass per-layer (rotary_dim, freq_base,
        // theta_scale) so Gemma-4 sliding (rotary=256, θ=10 000) and
        // Gemma-4 full (rotary=128, θ=1 000 000, p-RoPE) use the
        // correct rotation. SubscribesXxx layers skip the K-rotate
        // (their k_buf is stale anyway — they read from the publisher's
        // already-rotated KV slab).
        if cfg.has_qk_norm {
            let wqn = layer_weight(model, layer, "attn_q_norm.weight");
            self.run_rms_norm_mul_rope(
                dev, registry, cmd,
                q_buf, wqn, q_buf,
                head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                cfg.n_heads, /* m = */ 1,
                cfg.rms_norm_eps, "rms_norm_mul_rope_q",
            );
            if layer_owns_kv {
                let wkn = layer_weight(model, layer, "attn_k_norm.weight");
                self.run_rms_norm_mul_rope(
                    dev, registry, cmd,
                    k_buf, wkn, k_buf,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_kv_heads, /* m = */ 1,
                    cfg.rms_norm_eps, "rms_norm_mul_rope_k",
                );
            }
        } else {
            self.run_rope_neox_with_pos_offset(
                dev, registry, cmd, q_buf, q_buf,
                head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                cfg.n_heads, position, /* pos_buf_offset = */ 0, "rope_q",
            );
            if layer_owns_kv {
                self.run_rope_neox_with_pos_offset(
                    dev, registry, cmd, k_buf, k_buf,
                    head_dim, rope_rotary_dim, rope_freq_base, rope_theta_scale,
                    cfg.n_kv_heads, position, /* pos_buf_offset = */ 0, "rope_k",
                );
            }
        }
        if layer_owns_kv {
            self.mark_written(&[q_buf, k_buf]);
        } else {
            self.mark_written(&[q_buf]);
        }
        // Sprint 43D-4 — Gemma-4 V-norm. HF Gemma4TextAttention applies a
        // parameterless RMSNorm to V (head_dim axis, with_scale=False)
        // BEFORE the KV-cache write. Without this the V projection
        // direction drifts steeply across early layers (cosine-similarity
        // analysis showed 0.95 → 0.57 between layer 1 → 2 on Gemma-4-E2B
        // for prompt "Hi"). The shader is the standard `run_rms_norm`
        // with a synthetic `vnorm_ones` gamma vector — equivalent to
        // `v * rsqrt(mean(v²) + eps)` per HF.
        // Subscribers don't compute their own V (they read publisher's
        // already-normed cache slab) — skip.
        if cfg.gemma4.is_some() && layer_owns_kv {
            self.maybe_compute_barrier(dev, cmd, &[v_buf]);
            self.run_rms_norm(
                dev, registry, cmd,
                v_buf, self.vnorm_ones.handle, v_buf,
                head_dim, cfg.n_kv_heads,
                cfg.rms_norm_eps, "rms_norm_v",
            );
            self.mark_written(&[v_buf]);
        }
        // Next: (e) KV-write reads k_buf, v_buf.
        if layer_owns_kv {
            self.maybe_compute_barrier(dev, cmd, &[k_buf, v_buf]);
        }

        // (e) KV-cache write — pos-major.
        // Sprint 9d.3: when the cache is FP16-allocated, dispatch the
        // FP32 → packed-FP16 conversion compute shader. Otherwise the
        // cheap vkCmdCopyBuffer transfer (FP32 → FP32) wins.
        // Sprint 43C — for Gemma-4 sliding layers (head_dim=256), only
        // `kv_dim` valid bytes are written into a slot allocated for
        // `cfg.n_kv_heads * cfg.head_dim` (uniform max=512). The upper
        // half is unused; subsequent attention reads honour the per-
        // layer head_dim push-constant and never touch the unused tail.
        // Sprint 43D-2 — Gemma-4 SubscribesXxx layers skip the cache
        // write entirely (they have no own K/V to publish; the
        // attention dispatch reads the publisher's slab via
        // `gemma4_kv_read_layer`).
        if layer_owns_kv {
            let row_bytes = self.kv_cache.row_bytes(layer);
            let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
            let k_src = self.cur().k_buf.handle;
            let v_src = self.cur().v_buf.handle;
            let k_dst = self.kv_cache.k_buffer.handle;
            let v_dst = self.kv_cache.v_buffer.handle;
            if self.kv_cache.is_fp8() {
                let kv_elements = kv_dim;
                self.run_kv_store_fp8(
                    dev, registry, cmd, k_src, k_dst, kv_elements, dst_off, "kv_store_fp8_k",
                );
                self.run_kv_store_fp8(
                    dev, registry, cmd, v_src, v_dst, kv_elements, dst_off, "kv_store_fp8_v",
                );
            } else if self.kv_cache.is_fp16() {
                let kv_elements = kv_dim;
                self.run_kv_copy_fp16(
                    dev, registry, cmd, k_src, k_dst, kv_elements, dst_off, "kv_copy_fp16_k",
                );
                self.run_kv_copy_fp16(
                    dev, registry, cmd, v_src, v_dst, kv_elements, dst_off, "kv_copy_fp16_v",
                );
            } else {
                let copy = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(dst_off).size(row_bytes);
                self.profile("kv_write", dev, cmd, |dev, cmd| unsafe {
                    dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                    dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
                });
            }
            // (e) inline kv_bar — TRANSFER+COMPUTE → COMPUTE. Always
            // emitted (not elided): it's the only barrier with a TRANSFER
            // stage in this layer, the elision tracker only governs pure
            // compute_barrier sites. After this barrier the dirty set
            // also clears (a global VkMemoryBarrier covers everything).
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
            self.pending_writes.clear();
        } else {
            // Subscribes layer: emit a pure compute→compute barrier so
            // that the upcoming attention dispatch sees Q-RoPE writes
            // committed (the publisher's KV slab was written in an
            // earlier layer's KV-write block — that path emitted its
            // own kv_bar at the time). The publisher's slab is stable
            // by the time we reach this layer, so no fresh KV barrier
            // is needed; we only need a barrier for q_buf.
            self.maybe_compute_barrier(dev, cmd, &[q_buf]);
        }

        // (f) Attention.
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        self.mark_written(&[attn_out]);
        // Next: (g) O-proj reads attn_out.
        self.maybe_compute_barrier(dev, cmd, &[attn_out]);

        // (g) Output projection.
        let wo = layer_weight(model, layer, "attn_output.weight");
        let so = layer_weight_shader(model, layer, "attn_output.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_o = layer_weight_scale_scalar(model, layer, "attn_output.weight");
        let sb_o = layer_weight_scale_buf(model, layer, "attn_output.weight");
        if let Some(s) = sb_o {
            self.run_gemv_fp8_dispatch(dev, cmd, wo, s, attn_out, o_buf,
                q_dim, cfg.hidden_dim,
                layer_weight_scale_block(model, layer, "attn_output.weight"), "gemv_o");
        } else {
            self.run_gemv(dev, registry, cmd, so, wo, attn_out, o_buf,
                q_dim, cfg.hidden_dim, scale_o, "gemv_o");
        }
        self.mark_written(&[o_buf]);
        // Sprint 43D-4 STAGE 2 — post-attention (o_buf, after O-projection).
        stage_dump(dev, cmd, 2, o_buf);
        // Next: (h+i) reads input + o_buf.
        self.maybe_compute_barrier(dev, cmd, &[input, o_buf]);

        // (h+i) Norms around the attention residual.
        //
        // Llama / Qwen layout (one norm):
        //   res1 = input + o_buf
        //   hidden_norm = rms_norm(res1) * ffn_norm.weight
        // — fused into multi_add_rms via Sprint 9b.
        //
        // Sprint 43D Diagnose — Gemma-4 layout (TWO norms; a different
        // residual structure):
        //   o_normed = rms_norm(o_buf) * post_attention_layernorm.weight
        //   res1 = input + o_normed
        //   hidden_norm = rms_norm(res1) * pre_feedforward_layernorm.weight
        // (post_attention_layernorm is mapped to VF's `ffn_norm.weight`;
        //  pre_feedforward_layernorm is the new `ffn_pre_norm.weight`
        //  added in Sprint 43B-1.)
        if cfg.gemma4.is_some() {
            // Step 1: o_normed = rms_norm(o_buf) * post_attention_layernorm
            let post_attn_w = layer_weight(model, layer, "ffn_norm.weight");
            // Reuse `gate_buf` as scratch for the normed o (gate_buf is
            // not yet live — it's filled by the FFN gate GEMV below).
            self.run_rms_norm(dev, registry, cmd, o_buf, post_attn_w, gate_buf,
                              cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_attn");
            self.mark_written(&[gate_buf]);
            self.maybe_compute_barrier(dev, cmd, &[input, gate_buf]);

            // Step 2: res1 = input + gate_buf
            self.run_binary(dev, registry, cmd, ShaderId::Add,
                            input, gate_buf, res1,
                            cfg.hidden_dim, "add_res1_gemma4");
            self.mark_written(&[res1]);
            // Sprint 43D-4 STAGE 3 — post-residual1 (= input + post_attn_norm(attn)).
            stage_dump(dev, cmd, 3, res1);
            self.maybe_compute_barrier(dev, cmd, &[res1]);

            // Step 3: hidden_norm = rms_norm(res1) * pre_feedforward_layernorm
            let pre_ffn_w = layer_weight(model, layer, "ffn_pre_norm.weight");
            self.run_rms_norm(dev, registry, cmd, res1, pre_ffn_w, hidden_norm,
                              cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_pre_ffn");
            self.mark_written(&[hidden_norm]);
            self.maybe_compute_barrier(dev, cmd, &[hidden_norm]);
        } else {
            let w = layer_weight(model, layer, "ffn_norm.weight");
            self.run_multi_add_rms(
                dev, registry, cmd,
                input, o_buf, w,
                /* sum_out  = */ res1,
                /* norm_out = */ hidden_norm,
                cfg.hidden_dim, 1, cfg.rms_norm_eps, "add_rms_ffn",
            );
            self.mark_written(&[res1, hidden_norm]);
            // Next: (j) gate/up read hidden_norm.
            self.maybe_compute_barrier(dev, cmd, &[hidden_norm]);
        }

        // (j) gate / up
        let wg = layer_weight(model, layer, "ffn_gate.weight");
        let wu = layer_weight(model, layer, "ffn_up.weight");
        let sg = layer_weight_shader(model, layer, "ffn_gate.weight", self.mul_mat_vec_subgroup_enabled);
        let su = layer_weight_shader(model, layer, "ffn_up.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_g = layer_weight_scale_scalar(model, layer, "ffn_gate.weight");
        let scale_u = layer_weight_scale_scalar(model, layer, "ffn_up.weight");
        let sb_g = layer_weight_scale_buf(model, layer, "ffn_gate.weight");
        let sb_u = layer_weight_scale_buf(model, layer, "ffn_up.weight");
        if let Some(s) = sb_g {
            self.run_gemv_fp8_dispatch(dev, cmd, wg, s, hidden_norm, gate_buf,
                cfg.hidden_dim, ffn_dim,
                layer_weight_scale_block(model, layer, "ffn_gate.weight"), "gemv_gate");
        } else {
            self.run_gemv(dev, registry, cmd, sg, wg, hidden_norm, gate_buf,
                cfg.hidden_dim, ffn_dim, scale_g, "gemv_gate");
        }
        if let Some(s) = sb_u {
            self.run_gemv_fp8_dispatch(dev, cmd, wu, s, hidden_norm, up_buf,
                cfg.hidden_dim, ffn_dim,
                layer_weight_scale_block(model, layer, "ffn_up.weight"), "gemv_up");
        } else {
            self.run_gemv(dev, registry, cmd, su, wu, hidden_norm, up_buf,
                cfg.hidden_dim, ffn_dim, scale_u, "gemv_up");
        }
        self.mark_written(&[gate_buf, up_buf]);
        // Next: (k+l) swiglu reads gate_buf + up_buf.
        self.maybe_compute_barrier(dev, cmd, &[gate_buf, up_buf]);

        // (k+l) Fused activation-GLU: ffn_hidden = act(gate) * up.
        // v0.2 Sprint 9a folds silu(gate→gate) + mul(gate, up→ffn_hidden)
        // into one dispatch. Sprint 43D-2 — Gemma-4 substitutes the
        // pytorch-tanh GELU for SiLU (HF `gelu_pytorch_tanh`); same
        // dispatch shape + bindings, only the activation differs.
        let use_gelu_pytorch_tanh = cfg
            .gemma4
            .as_ref()
            .map(|g| g.hidden_activation == "gelu_pytorch_tanh")
            .unwrap_or(false);
        if use_gelu_pytorch_tanh {
            self.run_gelu_pytorch_tanh_glu(
                dev, registry, cmd,
                gate_buf, up_buf, ffn_hidden,
                ffn_dim, "gelu_pt_glu",
            );
        } else {
            self.run_swiglu(
                dev, registry, cmd,
                gate_buf, up_buf, ffn_hidden,
                ffn_dim, "swiglu",
            );
        }
        self.mark_written(&[ffn_hidden]);
        // Next: (m) FFN down reads ffn_hidden.
        self.maybe_compute_barrier(dev, cmd, &[ffn_hidden]);

        // (m) FFN down — Q6_K in Q4_K_M, Q4_K otherwise.
        let wd = layer_weight(model, layer, "ffn_down.weight");
        let sd = layer_weight_shader(model, layer, "ffn_down.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_d = layer_weight_scale_scalar(model, layer, "ffn_down.weight");
        let sb_d = layer_weight_scale_buf(model, layer, "ffn_down.weight");
        if let Some(s) = sb_d {
            self.run_gemv_fp8_dispatch(dev, cmd, wd, s, ffn_hidden, ffn_out,
                ffn_dim, cfg.hidden_dim,
                layer_weight_scale_block(model, layer, "ffn_down.weight"), "gemv_down");
        } else {
            self.run_gemv(dev, registry, cmd, sd, wd, ffn_hidden, ffn_out,
                ffn_dim, cfg.hidden_dim, scale_d, "gemv_down");
        }
        self.mark_written(&[ffn_out]);
        // Sprint 43D-4 STAGE 4 — post-MLP (ffn_out, after ffn_down GEMV).
        stage_dump(dev, cmd, 4, ffn_out);
        // Sprint 43D Diagnose — Gemma-4 has a post_feedforward_layernorm
        // applied to ffn_out *before* the final residual add.
        if cfg.gemma4.is_some() {
            self.maybe_compute_barrier(dev, cmd, &[ffn_out]);
            let post_ffn_w = layer_weight(model, layer, "ffn_post_norm.weight");
            // Reuse gate_buf again as scratch (it's been consumed by
            // SwiGLU at this point — its hidden_dim/ffn_dim sized
            // allocation comfortably holds the hidden-dim-sized result).
            self.run_rms_norm(dev, registry, cmd, ffn_out, post_ffn_w, gate_buf,
                              cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_post_ffn");
            self.mark_written(&[gate_buf]);
            self.maybe_compute_barrier(dev, cmd, &[res1, gate_buf]);
            self.run_binary(dev, registry, cmd, ShaderId::Add,
                            res1, gate_buf, output,
                            cfg.hidden_dim, "add_res2_gemma4");
            // Sprint 43D-4 STAGE 5 — post-residual2 (output = res1 + post_ffn_norm(ffn_out)).
            stage_dump(dev, cmd, 5, output);
        } else {
            self.maybe_compute_barrier(dev, cmd, &[res1, ffn_out]);
            // (n) Residual2 = res1 + ffn_out → output
            self.run_binary(dev, registry, cmd, ShaderId::Add,
                            res1, ffn_out, output,
                            cfg.hidden_dim, "add_res2");
        }
        self.mark_written(&[output]);

        // Sprint 43D-3 — PLE (Per-Layer Embeddings) integration block.
        // After the standard layer output (`output` = residual + attn +
        // ffn) we add a per-layer modulation contribution gated by the
        // pre-staged per_layer_inputs[layer] slice (built CPU-side in
        // `forward_token` from `embed_tokens_per_layer[token_id]`).
        //
        // Formula (per HF transformers `Gemma3nTextDecoderLayer.forward`,
        // sans AltUp/laurel which VF doesn't yet implement):
        //   gate    = per_layer_input_gate @ output           # [hps]
        //   gate    = gelu_pytorch_tanh(gate) * per_layer_inputs[layer]
        //   proj    = per_layer_projection @ gate             # [hidden]
        //   normed  = post_per_layer_input_norm(proj)         # [hidden]
        //   output += normed
        //
        // Buffer reuse — by this point in the layer everything except
        // `output` and `res1` is conceptually consumed:
        //   gate_buf   — 256-dim PLE gate scratch (alloc'd ffn_dim, fits)
        //   o_buf      — 1536-dim PLE projection output
        //   attn_out   — 1536-dim post-rmsnorm output (distinct from o_buf
        //                so rms_norm doesn't run in-place)
        //   res1       — left untouched (would race with future inputs?
        //                no — caller is next layer or dispatch_final).
        if let Some(ple) = model.ple_data.as_ref() {
            // Gate-stage scratch is bound from the ffn-sized `gate_buf`;
            // we only touch the first `hps × 4` bytes.
            let hps = ple.hps;
            let hps_bytes = (hps as u64) * 4;
            let hidden_bytes = (cfg.hidden_dim as u64) * 4;
            let ple_inputs_buf = self.cur().per_layer_inputs.handle;
            let ple_inputs_offset = (layer as u64) * hps_bytes;

            // (1) gate = per_layer_input_gate @ output  — F32 GEMV (1536 → 256).
            //     Run after a barrier on `output` (just written by add_res2).
            self.maybe_compute_barrier(dev, cmd, &[output]);
            let w_gate = layer_weight(model, layer, "per_layer_input_gate.weight");
            let s_gate = layer_weight_shader(
                model, layer, "per_layer_input_gate.weight",
                self.mul_mat_vec_subgroup_enabled,
            );
            let scale_gate = layer_weight_scale_scalar(model, layer, "per_layer_input_gate.weight");
            self.run_gemv(
                dev, registry, cmd, s_gate, w_gate,
                output, gate_buf,
                cfg.hidden_dim, hps, scale_gate,
                "ple_gemv_gate",
            );
            self.mark_written(&[gate_buf]);
            self.maybe_compute_barrier(dev, cmd, &[gate_buf]);

            // (2) gate ← gelu_pytorch_tanh(gate) * per_layer_inputs[layer]
            //     Use the GLU shader: out[i] = gelu(a[i]) * b[i]. Bind
            //     gate_buf as both `a` (read) and `out` (write), with the
            //     PLE-input slice as `b`. The shader reads a[i] before
            //     writing out[i], so the alias is safe (one element per
            //     thread, no dependency across lanes).
            let kernel = registry.get(ShaderId::GeluPytorchTanhGlu);
            let set = self.alloc_or_get_set(
                dev, kernel.descriptor_set_layout,
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
            self.profile("ple_gelu_glu", dev, cmd, |dev, cmd| unsafe {
                dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                dev.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
                );
                dev.device.cmd_push_constants(
                    cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
                );
                dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
            });
            self.mark_written(&[gate_buf]);
            self.maybe_compute_barrier(dev, cmd, &[gate_buf]);

            // (3) proj = per_layer_projection @ gate  — F32 GEMV (256 → 1536).
            let w_proj = layer_weight(model, layer, "per_layer_projection.weight");
            let s_proj = layer_weight_shader(
                model, layer, "per_layer_projection.weight",
                self.mul_mat_vec_subgroup_enabled,
            );
            let scale_proj = layer_weight_scale_scalar(model, layer, "per_layer_projection.weight");
            self.run_gemv(
                dev, registry, cmd, s_proj, w_proj,
                gate_buf, o_buf,
                hps, cfg.hidden_dim, scale_proj,
                "ple_gemv_proj",
            );
            self.mark_written(&[o_buf]);
            self.maybe_compute_barrier(dev, cmd, &[o_buf]);
            let _ = hidden_bytes;

            // (4) normed = rms_norm(proj, post_per_layer_input_norm.weight).
            //     o_buf → attn_out (distinct buffer; avoids in-place edge case).
            let w_pln = layer_weight(model, layer, "post_per_layer_input_norm.weight");
            self.run_rms_norm(
                dev, registry, cmd, o_buf, w_pln, attn_out,
                cfg.hidden_dim, 1, cfg.rms_norm_eps, "ple_rms_norm",
            );
            self.mark_written(&[attn_out]);
            self.maybe_compute_barrier(dev, cmd, &[attn_out, output]);

            // (5) output += normed.
            self.run_binary(
                dev, registry, cmd, ShaderId::Add,
                output, attn_out, output,
                cfg.hidden_dim, "add_ple",
            );
            self.mark_written(&[output]);
            // Sprint 43D-4 STAGE 6 — post-PLE (output, after ple-add).
            stage_dump(dev, cmd, 6, output);
        }

        // Sprint 43D-4 — Gemma-4 per-layer scalar (HF
        // `Gemma4TextDecoderLayer.forward` final line:
        // `hidden_states *= self.layer_scalar`). The scalar is a learned
        // [1] tensor per layer with values 0.018..0.871 (mean 0.527 on
        // E2B). Without this multiply, layer-0 output is 56× too large,
        // layer-14 35× — the cascade through residual additions in
        // subsequent layers overwhelms the actual signal and forces
        // the logit distribution to collapse to a few high-freq tokens
        // (Sprint 43D-3 mode-collapse symptom). Applied as the very
        // last op before the layer returns. Non-Gemma-4 stacks skip
        // this entirely.
        if cfg.gemma4.is_some() {
            self.maybe_compute_barrier(dev, cmd, &[output]);
            let scalar = layer_weight(model, layer, "layer_scalar");
            self.run_mul_scalar_b(
                dev, registry, cmd, output, scalar, output,
                cfg.hidden_dim, "layer_scalar_mul",
            );
            self.mark_written(&[output]);
            // Sprint 43D-4 STAGE 7 — post-layer_scalar (= final layer output).
            stage_dump(dev, cmd, 7, output);
        }

        // Next: next layer's attn_norm reads `output` (= its `input`),
        // or dispatch_final reads `output`. Either way: caller's first
        // op reads `output`.
        self.maybe_compute_barrier(dev, cmd, &[output]);
    }

    pub(super) fn dispatch_final(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        input: vk::Buffer,
    ) {
        let w_norm = model
            .tensor("output_norm.weight")
            .expect("output_norm.weight")
            .buffer
            .handle;
        // Sprint 40 Part 2 — CPU lm_head branch. When the loader has
        // built a `CpuLmHead` (only happens when `VF_CPU_LM_HEAD=1`
        // was set), we still run the final RMS norm on the GPU but
        // skip the GPU lm_head GEMV. Instead we copy the post-norm
        // hidden state into the host-mapped `hidden_staging` buffer.
        // After the CB submits + completes, `forward_token` reads
        // the hidden state, runs `cpu_lm_head.forward`, and writes
        // the logits into `logits_staging` directly. The `cmd_copy`
        // here is tiny (~16 KB) and folds into the same submit, so
        // the only added serialization is the CPU GEMV itself —
        // hence "Phase A blocking".
        if model.cpu_lm_head.is_some() {
            let hidden_norm = self.cur().hidden_norm.handle;
            self.run_rms_norm(
                dev, registry, cmd,
                input, w_norm, hidden_norm,
                self.config.hidden_dim, 1, self.config.rms_norm_eps, "rms_norm_final",
            );
            self.mark_written(&[hidden_norm]);
            // Make the RMS-norm SHADER_WRITE visible to the
            // upcoming TRANSFER_READ.
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(hidden_norm)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            let copy = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: (self.config.hidden_dim as u64) * 4,
            };
            // Then make the staging buffer's TRANSFER_WRITE visible
            // to the host read that follows submit.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.hidden_staging.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&pre),
                    &[],
                );
                dev.device.cmd_copy_buffer(
                    cmd,
                    hidden_norm,
                    self.hidden_staging.handle,
                    std::slice::from_ref(&copy),
                );
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&post),
                    &[],
                );
            }
            // Skip the GPU `lm_head` GEMV entirely. `record_logits_readback`
            // still runs after this on the existing `logits_buf` /
            // `logits_staging` pair — it copies a stale logits buffer
            // that `forward_token` overwrites with the CPU result
            // immediately afterwards. The wasted ~600 KB GPU→CPU
            // copy is negligible.
            return;
        }
        // LM head: prefer dedicated `output.weight`; fall back to tied
        // `token_embd.weight` (Phase 2 doesn't tie weights, but be safe).
        let lm = model
            .tensor("output.weight")
            .or_else(|| model.tensor("token_embd.weight"))
            .expect("LM head present");
        let w_lm = lm.buffer.handle;
        // Sprint 20-M3 — lm_head can be F32 (SafeTensors FP8 models
        // exclude lm_head from quantization) or F8E4M3 (some FP8
        // builds also quantize lm_head). Both route through the
        // dedicated 1-WG-per-row shaders introduced in Sprint 20.
        let lm_shader = match (lm.ggml_type, self.mul_mat_vec_subgroup_enabled) {
            (GgmlType::F8E4M3, _) => ShaderId::MulMatVecFp8,
            (GgmlType::F32,    _) => ShaderId::MulMatVecF32,
            (GgmlType::F16,    _) => ShaderId::MulMatVecF16,
            (GgmlType::Q6K, true ) => ShaderId::MulMatVecQ6KSubgroup,
            (GgmlType::Q6K, false) => ShaderId::MulMatVecQ6K,
            (_,             true ) => ShaderId::MulMatVecQ4KSubgroup,
            (_,             false) => ShaderId::MulMatVecQ4K,
        };
        let lm_scale = lm.weight_scale.unwrap_or(1.0);

        let hidden_norm = self.cur().hidden_norm.handle;
        let hidden_norm_size = self.cur().hidden_norm.size;
        if std::env::var("VF_BUF_ID").is_ok() {
            eprintln!(
                "[BUF-ID] dispatch_final entry: slot={} input={:?} hidden_norm={:?} \
                 hidden_norm_size={} w_norm={:?}",
                self.current_slot, input, hidden_norm, hidden_norm_size, w_norm,
            );
        }
        self.run_rms_norm(
            dev, registry, cmd,
            input, w_norm, hidden_norm,
            self.config.hidden_dim, 1, self.config.rms_norm_eps, "rms_norm_final",
        );
        if std::env::var("VF_BUF_ID").is_ok() {
            eprintln!(
                "[BUF-ID] dispatch_final after rms_norm_final: hidden_norm={:?} \
                 mark_written queued for hidden_norm",
                hidden_norm,
            );
        }
        self.mark_written(&[hidden_norm]);
        // Sprint 43E-2 — force-barrier hidden_norm before lm_head /
        // diagnose ops below. The default `maybe_compute_barrier` path
        // elides this barrier (cache_enabled tracker thinks hidden_norm
        // was already flushed by a previous layer's mark_written).
        // Without this, lm_head reads stale hidden_norm — including
        // NaN bits from undefined GpuOnly memory init — and produces
        // all-NaN logits. VF_NO_FINAL_BARRIER=1 skips this opt-in to
        // verify the hypothesis.
        if std::env::var("VF_NO_FINAL_BARRIER").is_err() {
            let bar = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(hidden_norm)
                .offset(0).size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&bar),
                    &[],
                );
            }
        }
        // Sprint 43E-2 — VF_LOGITS_BUF_ZERO=1 fills logits_buf with zero
        // *before* the lm_head dispatch. If the post-lm_head logits stay
        // NaN, the GEMV is actually writing NaN. If they read as zero,
        // the GEMV isn't writing at all (synchronization / dispatch
        // issue). Diagnose-only; default off.
        if std::env::var("VF_LOGITS_BUF_ZERO").is_ok() {
            let logits_bytes = (self.config.vocab_size as u64) * 4;
            unsafe {
                dev.device.cmd_fill_buffer(
                    cmd, self.logits_buf.handle, 0, logits_bytes, 0,
                );
            }
            // Make TRANSFER_WRITE visible to upcoming SHADER_WRITE
            // (lm_head GEMV).
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
            }
        }

        // Sprint 43D Bisect — VF_FINAL_NORM_DUMP=1 copies hidden_norm
        // (= post-final-RMSNorm, pre-lm_head) into hidden_staging so
        // forward_token can dump it. Used to bisect whether NaN enters
        // in the final RMSNorm or in the lm_head GEMV.
        if std::env::var("VF_FINAL_NORM_DUMP").is_ok() {
            if std::env::var("VF_BUF_ID").is_ok() {
                eprintln!(
                    "[BUF-ID] VF_FINAL_NORM_DUMP read: src(hidden_norm)={:?} \
                     dst(hidden_staging)={:?} hidden_staging_size={}",
                    hidden_norm, self.hidden_staging.handle, self.hidden_staging.size,
                );
            }
            let bytes = (self.config.hidden_dim as u64) * 4;
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
                    cmd, hidden_norm, self.hidden_staging.handle,
                    std::slice::from_ref(&copy),
                );
            }
        }
        // Next: lm_head GEMV reads hidden_norm.
        self.maybe_compute_barrier(dev, cmd, &[hidden_norm]);
        // Sprint 29 — F16 lm_head goes through the dedicated harness
        // pipeline (PipelineCache::null + dedicated pool). Other
        // dtypes (F8E4M3 / F32 / Q-quants) keep the production
        // registry path. Toggle via VF_LMHEAD_HARNESS=0 to compare.
        let use_harness = matches!(lm_shader, ShaderId::MulMatVecF16)
            && std::env::var("VF_LMHEAD_HARNESS").map(|v| v != "0").unwrap_or(true);
        // Sprint 43E-2 — VF_SKIP_LM_HEAD=1 skips the lm_head GEMV
        // entirely. Combined with VF_LOGITS_BUF_ZERO=1, logits should
        // read back as 0.0 (the fill value). If they read NaN
        // instead, the corruption is downstream of lm_head — i.e.
        // either record_logits_readback or wait_and_read_logits.
        if std::env::var("VF_SKIP_LM_HEAD").is_ok() {
            // No-op; let logits_buf retain whatever zero-fill /
            // previous content it had.
        } else if std::env::var("VF_LM_COPY_HIDDEN").is_ok() {
            // Sprint 43E-2 — replace lm_head GEMV with a literal
            // cmd_copy_buffer of hidden_norm into the *first*
            // hidden_dim slots of logits_buf.
            if std::env::var("VF_BUF_ID").is_ok() {
                eprintln!(
                    "[BUF-ID] VF_LM_COPY_HIDDEN: src(hidden_norm)={:?} \
                     dst(logits_buf)={:?} logits_buf_size={}",
                    hidden_norm, self.logits_buf.handle, self.logits_buf.size,
                );
            }
            let bytes = (self.config.hidden_dim as u64) * 4;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, hidden_norm, self.logits_buf.handle,
                    std::slice::from_ref(&copy),
                );
            }
        } else if use_harness {
            if std::env::var("VF_BUF_ID").is_ok() {
                eprintln!(
                    "[BUF-ID] lm_head harness: w_lm={:?} hidden_norm={:?} logits_buf={:?}",
                    w_lm, hidden_norm, self.logits_buf.handle,
                );
            }
            self.run_gemv_lmhead_dedicated(
                dev, cmd, w_lm, hidden_norm, self.logits_buf.handle,
                self.config.hidden_dim, self.config.vocab_size, lm_scale,
            );
        } else {
            if std::env::var("VF_BUF_ID").is_ok() {
                eprintln!(
                    "[BUF-ID] lm_head GEMV: shader={:?} w_lm={:?} hidden_norm={:?} \
                     logits_buf={:?} k={} m={} scale={:.4}",
                    lm_shader, w_lm, hidden_norm, self.logits_buf.handle,
                    self.config.hidden_dim, self.config.vocab_size, lm_scale,
                );
            }
            self.run_gemv(
                dev, registry, cmd, lm_shader,
                w_lm, hidden_norm, self.logits_buf.handle,
                self.config.hidden_dim, self.config.vocab_size, lm_scale, "lm_head",
            );
        }
        self.mark_written(&[self.logits_buf.handle]);
    }

    /// Sprint 27 — copy `logits_buf` (GpuOnly) to `logits_staging`
    /// (GpuToCpu) inside `cmd`, with the surrounding barriers. Replaces
    /// the previous `SHADER_WRITE → HOST_READ` barrier on the
    /// host-mapped `logits_buf`. Tiny GPU work (vocab × 4 bytes), but
    /// removes the scattered-host-write stall that inflated lm_head's
    /// timestamp by ~27 ms on 14B-FP8 (Sprint 27 finding).
    pub(super) fn record_logits_readback(&self, dev: &VulkanDevice, cmd: vk::CommandBuffer) {
        let logits_bytes = (self.config.vocab_size as u64) * 4;
        // (1) make lm_head's SHADER_WRITE visible to the upcoming
        // TRANSFER_READ — execution + memory dependency.
        let pre = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.logits_buf.handle)
            .offset(0).size(vk::WHOLE_SIZE);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[pre], &[],
            );
        }
        // (2) cmd_copy_buffer logits_buf → logits_staging.
        let copy = vk::BufferCopy::default().size(logits_bytes);
        unsafe {
            dev.device.cmd_copy_buffer(
                cmd,
                self.logits_buf.handle,
                self.logits_staging.handle,
                std::slice::from_ref(&copy),
            );
        }
        // (3) make the TRANSFER_WRITE visible to HOST_READ on the
        // staging buffer.
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.logits_staging.handle)
            .offset(0).size(vk::WHOLE_SIZE);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[], &[post], &[],
            );
        }
    }
}
