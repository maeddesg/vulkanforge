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
use super::super::pipeline_registry::PipelineRegistry;
use super::super::profiler::ShaderProfiler;
use super::super::shaders::ShaderId;

use super::arch::apply_final_logit_softcap;
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

    /// Per-layer decode dispatch. Builds the layer's plan and runs it
    /// through `DecodeExec`. Sprint 44C-3 — the inline body is gone;
    /// the executor is the only path.
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
        let plan = super::layer_plan::build_layer_plan(
            &self.config, model, layer, self.rope_theta_scale,
        );
        let ctx = super::executor::ExecCtx {
            dev, registry, cmd, model, layer,
            mode: super::executor::ExecMode::Decode { position, input, output },
        };
        let exec = super::executor::DecodeExec;
        exec.execute_layer(self, &plan, &ctx);
        // Trailing barrier: the next layer (or `dispatch_final`) reads
        // `output`. The executor's per-step barriers cover intra-layer
        // ordering; this caps the layer with the cross-layer
        // hand-off.
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
