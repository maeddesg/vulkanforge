//! Sprint 44B-4 — debug / introspection helpers extracted from
//! `forward/mod.rs`. Pure code-move.
//!
//! Three groups of helpers live here:
//! - `forward_layer_debug` + `forward_layer_debug_intermediate` —
//!   single-layer drivers used by the regression-test `tests/` and the
//!   Sprint 24/26 bisect harnesses; they run a fresh seed through ONE
//!   layer up to a configurable halt point and read scratch_b back to
//!   the host.
//! - `dispatch_layer_partial` — the halt-point-aware variant of
//!   `dispatch_layer`, only ever called from `forward_layer_debug*`.
//! - `read_scratch_b` — host stage-and-readback for any of the GpuOnly
//!   per-token scratch buffers; used by the layer-debug helpers and (in
//!   diagnostic builds) by `dispatch_final` via `VF_FINAL_NORM_DUMP`.
//!
//! Two free `maybe_dump_*` env-gated logging helpers also live here.
//! They're called from the decode hot path (`wait_and_read_logits`,
//! `forward_token`, `logits`, `dispatch_final`) so they're `pub(super)`
//! — see `forward/decode.rs` for the call sites.

use ash::vk;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;

use super::arch::{
    compute_barrier, layer_weight, layer_weight_opt, layer_weight_scale_block,
    layer_weight_scale_buf, layer_weight_scale_scalar, layer_weight_shader,
    n_kv_heads_for,
};
use super::executor::ExecCtx;
use super::layer_plan::LayerStep;
use super::state::{DebugTarget, Forward};

impl Forward {
    /// Debug helper — run ONE layer up to a chosen halt point and
    /// return the relevant intermediate as a `Vec<f32>`. The buffer
    /// returned has *exactly* the size of the relevant intermediate
    /// (e.g. n_heads*head_dim for AfterQkvProj-Q, not hidden_dim).
    pub fn forward_layer_debug_intermediate(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
        target: DebugTarget,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Debug path uses a partial layer dispatch with non-stable
        // bindings — invalidate any cached sets from prior decode runs.
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer_partial(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, target,
            );
        })?;

        // Pick the relevant buffer + size for readback.
        let cfg = self.config.clone();
        let (src_buf, count) = match target {
            DebugTarget::AttnNorm => (self.cur().hidden_norm.handle, cfg.hidden_dim as u64),
            DebugTarget::QProj | DebugTarget::QNormRope => (
                self.cur().q_buf.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::KProj | DebugTarget::KNormRope => (
                self.cur().k_buf.handle,
                (n_kv_heads_for(&cfg, layer) * cfg.head_dim) as u64,
            ),
            DebugTarget::VProj => (
                self.cur().v_buf.handle,
                (n_kv_heads_for(&cfg, layer) * cfg.head_dim) as u64,
            ),
            DebugTarget::AttnOut => (
                self.cur().attn_out.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
        };
        // Stage into scratch_a (host-visible).
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(src_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(count * 4);
            dev.device.cmd_copy_buffer(cmd, src_buf, self.cur().scratch_a.handle,
                std::slice::from_ref(&copy));
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes = self.cur().scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes[..(count as usize) * 4]).to_vec())
    }

    fn dispatch_layer_partial(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input: vk::Buffer,
        halt: DebugTarget,
    ) {
        let cfg = self.config.clone();

        // attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, self.cur().hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        if halt == DebugTarget::AttnNorm { return; }
        compute_barrier(dev, cmd);

        // Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight", self.mul_mat_vec_subgroup_enabled);
        let sk = layer_weight_shader(model, layer, "attn_k.weight", self.mul_mat_vec_subgroup_enabled);
        let sv = layer_weight_shader(model, layer, "attn_v.weight", self.mul_mat_vec_subgroup_enabled);
        let scale_q = layer_weight_scale_scalar(model, layer, "attn_q.weight");
        let scale_k = layer_weight_scale_scalar(model, layer, "attn_k.weight");
        let scale_v = layer_weight_scale_scalar(model, layer, "attn_v.weight");
        // Sprint 24-Inline Step 0 — route FP8 weights through the
        // harness-style per-channel path when a scale buffer exists.
        let sb_q = layer_weight_scale_buf(model, layer, "attn_q.weight");
        let sb_k = layer_weight_scale_buf(model, layer, "attn_k.weight");
        let sb_v = layer_weight_scale_buf(model, layer, "attn_v.weight");
        let q_h = self.cur().q_buf.handle;
        let k_h = self.cur().k_buf.handle;
        let v_h = self.cur().v_buf.handle;
        let in_h = self.cur().hidden_norm.handle;
        if let Some(s) = sb_q {
            self.run_gemv_fp8_dispatch(dev, cmd, wq, s, in_h, q_h,
                cfg.hidden_dim, cfg.n_heads * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_q.weight"), "gemv_q");
        } else {
            self.run_gemv(dev, registry, cmd, sq, wq, in_h, q_h,
                cfg.hidden_dim, cfg.n_heads * cfg.head_dim, scale_q, "gemv_q");
        }
        if let Some(s) = sb_k {
            self.run_gemv_fp8_dispatch(dev, cmd, wk, s, in_h, k_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_k.weight"), "gemv_k");
        } else {
            self.run_gemv(dev, registry, cmd, sk, wk, in_h, k_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim, scale_k, "gemv_k");
        }
        if let Some(s) = sb_v {
            self.run_gemv_fp8_dispatch(dev, cmd, wv, s, in_h, v_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_v.weight"), "gemv_v");
        } else {
            self.run_gemv(dev, registry, cmd, sv, wv, in_h, v_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim, scale_v, "gemv_v");
        }
        if matches!(halt, DebugTarget::QProj | DebugTarget::KProj | DebugTarget::VProj) {
            return;
        }
        compute_barrier(dev, cmd);

        // Sprint 24B — Q/K/V bias-add (Qwen2 attention biases). Skipped
        // for architectures without biases (Llama, Qwen3, Mistral).
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = n_kv_heads_for(&cfg, layer) * cfg.head_dim;
        let q_bias = layer_weight_opt(model, layer, "attn_q.bias");
        let k_bias = layer_weight_opt(model, layer, "attn_k.bias");
        let v_bias = layer_weight_opt(model, layer, "attn_v.bias");
        if q_bias.is_some() || k_bias.is_some() || v_bias.is_some() {
            let q_buf = self.cur().q_buf.handle;
            let k_buf = self.cur().k_buf.handle;
            let v_buf = self.cur().v_buf.handle;
            if let Some(b) = q_bias { self.run_bias_add(dev, registry, cmd, q_buf, b, q_buf, q_dim, 1, "bias_q"); }
            if let Some(b) = k_bias { self.run_bias_add(dev, registry, cmd, k_buf, b, k_buf, kv_dim, 1, "bias_k"); }
            if let Some(b) = v_bias { self.run_bias_add(dev, registry, cmd, v_buf, b, v_buf, kv_dim, 1, "bias_v"); }
            compute_barrier(dev, cmd);
        }

        // Q/K norm — Qwen-only (Phase 4D: gated on cfg.has_qk_norm).
        if cfg.has_qk_norm {
            let wqn = layer_weight(model, layer, "attn_q_norm.weight");
            let wkn = layer_weight(model, layer, "attn_k_norm.weight");
            self.run_rms_norm(dev, registry, cmd,
                             self.cur().q_buf.handle, wqn, self.cur().q_buf.handle,
                             cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
            self.run_rms_norm(dev, registry, cmd,
                             self.cur().k_buf.handle, wkn, self.cur().k_buf.handle,
                             cfg.head_dim, n_kv_heads_for(&cfg, layer), cfg.rms_norm_eps, "rms_norm_k");
            compute_barrier(dev, cmd);
        }

        // RoPE
        self.run_rope_neox(dev, registry, cmd, self.cur().q_buf.handle, self.cur().q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.cur().k_buf.handle, self.cur().k_buf.handle,
                           cfg.head_dim, n_kv_heads_for(&cfg, layer), position, "rope_k");
        if matches!(halt, DebugTarget::QNormRope | DebugTarget::KNormRope) {
            return;
        }
        compute_barrier(dev, cmd);

        // KV write — Sprint 9d.3: dispatch the FP32→FP16 conversion
        // compute shader when the cache is FP16-allocated, otherwise
        // keep the cheap vkCmdCopyBuffer transfer.
        let row_bytes = self.kv_cache.row_bytes(layer);
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let k_src = self.cur().k_buf.handle;
        let v_src = self.cur().v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        if self.kv_cache.is_fp8() {
            let kv_elements = n_kv_heads_for(&cfg, layer) * cfg.head_dim;
            self.run_kv_store_fp8(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off,
                "kv_store_fp8_k_d",
            );
            self.run_kv_store_fp8(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off,
                "kv_store_fp8_v_d",
            );
        } else if self.kv_cache.is_fp16() {
            let kv_elements = n_kv_heads_for(&cfg, layer) * cfg.head_dim;
            self.run_kv_copy_fp16(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off,
                "kv_copy_fp16_k_d",
            );
            self.run_kv_copy_fp16(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off,
                "kv_copy_fp16_v_d",
            );
        } else {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(row_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
            }
        }
        // Barrier covers either upstream (transfer or compute write).
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

        // Attention
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        // Stops here regardless — caller wanted attn_out.
        let _ = halt; // attention is the only further halt point in this partial dispatcher
    }

    /// Debug helper — run ONE layer, returning the post-layer
    /// activations as `Vec<f32>`. Each call submits its own command
    /// buffer; layers must be invoked in order, with `position`
    /// constant (we're tracing a single token through all layers).
    pub fn forward_layer_debug(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Phase-2C debug: stage input via scratch_a, single layer
        // dispatch, read scratch_b. We rebuild the descriptor pool
        // (small) from a single big reset.
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
            // Readback barrier so the next host map sees scratch_b.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[post],
                    &[],
                );
            }
        })?;
        // scratch_b is GpuOnly — we can't read it. Need to stage.
        // Hack: alloc a host-visible read buf on the fly and copy.
        Ok(self.read_scratch_b(dev, cmd_ctx)?)
    }

    /// Stage scratch_b (GpuOnly) into a host-visible buffer and read
    /// it. Per-call alloc — for debug only.
    fn read_scratch_b(
        &self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Use the host-visible scratch_a as the readback target —
        // it's the same size as scratch_b. We don't need scratch_a's
        // content after this call (caller writes a fresh input next
        // round).
        let bytes = (self.config.hidden_dim as u64) * 4;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            // Pre-barrier: previous SHADER_WRITE → TRANSFER_READ
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes);
            dev.device.cmd_copy_buffer(
                cmd,
                self.cur().scratch_b.handle,
                self.cur().scratch_a.handle,
                std::slice::from_ref(&copy),
            );
            // Post-barrier: TRANSFER_WRITE → HOST_READ
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cur().scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes_slice = self.cur().scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes_slice[..(self.config.hidden_dim as usize) * 4],
        )
        .to_vec())
    }
}

// ── Sprint G-2f throwaway: per-step dump for Qwen3.6 coherence bisect ──
//
// `VF_DUMP_LAYER_0=1` activates `dump_after_step` after every LayerStep on
// layer 0 of the DECODE path. Each dump stages a slice of the relevant SSM
// scratch buffer into `cur().scratch_a` (host-visible, 5120 floats wide),
// then mid-frame-submits + waits + reads + prints first-10 floats and
// min/max/mean stats.
//
// REQUIREMENTS for callers:
// - Set `VULKANFORGE_DISABLE_ASYNC_DECODE=1` to avoid the async-decode
//   hazard (see `memory/feedback_async_decode_dump_hazard.md`).
// - Caps dump width at 5120 floats (scratch_a size at hidden_dim=5120);
//   stats are computed over the sampled subset, not the whole buffer.
// - Throwaway: REMOVE THIS BLOCK + the `dump_after_step` call in
//   `executor/mod.rs` once the bisect lands a coherent Qwen3.6 fix.

/// Sprint G-2f — copy `[src_buf @ off_floats .. off_floats + n_floats)` to
/// host, print first 10 floats + min/max/mean over the read region.
///
/// `n_floats` is clamped to `scratch_a` capacity (`hidden_dim`, currently
/// 5120 on Qwen3.6). Mid-frame-submits + waits — expensive, do not call
/// outside `VF_DUMP_LAYER_0` gate.
pub(super) fn dump_buffer_layer0(
    fwd: &mut Forward,
    dev: &VulkanDevice,
    cmd: vk::CommandBuffer,
    src_buf: vk::Buffer,
    off_floats: u64,
    n_floats: usize,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Cap to first hidden_dim floats — Sprint G-2f bisect prints first 10
    // + small magnitude stats; full-buffer stats aren't needed and we'd
    // need a much larger staging anyway. `hidden_staging` is 64 ×
    // hidden_dim × 4 (= 1.28 MB on Qwen3.6) so we have plenty of room.
    let cap = fwd.config.hidden_dim as usize;
    let n = n_floats.min(cap);
    let bytes = (n as u64) * 4;
    let off_bytes = off_floats * 4;

    // CRITICAL: do NOT use `cur().scratch_a` as the staging here — it's
    // the layer's input buffer at layer 0 and overwriting it corrupts
    // AttnResidualAdd's `input` operand (Sprint G-2f false-positive trap).
    // Use `hidden_staging` (separate, GpuToCpu, only written by
    // dispatch_final which hasn't run yet for layer 0).
    let stage = fwd.hidden_staging.handle;
    unsafe {
        let pre = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(src_buf)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[], std::slice::from_ref(&pre), &[],
        );
        let copy = vk::BufferCopy::default()
            .src_offset(off_bytes).dst_offset(0).size(bytes);
        dev.device.cmd_copy_buffer(cmd, src_buf, stage, std::slice::from_ref(&copy));
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(stage)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[], std::slice::from_ref(&post), &[],
        );
    }

    fwd.mid_frame_submit_and_wait(dev, cmd)?;

    let raw = fwd.hidden_staging.read_bytes()?;
    let floats: &[f32] = bytemuck::cast_slice(&raw[..n * 4]);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut nan = 0u32;
    for &v in floats {
        if v.is_nan() { nan += 1; continue; }
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    let valid = (n as u32).saturating_sub(nan).max(1) as f64;
    let mean = sum / valid;
    let head: Vec<f32> = floats.iter().take(10).copied().collect();
    eprintln!(
        "[DUMP] {:<28} n={:>7} min={:>11.5} max={:>11.5} mean={:>11.5} nan={:>3}  first10={:?}",
        label, n, min, max, mean, nan, head,
    );
    Ok(())
}

/// Sprint G-2f — per-step dump dispatcher. Reads the relevant scratch
/// buffer for `step`, writes it to stderr. No-op when:
/// - `ctx.layer != 0` (caller already gates, but defensive)
/// - `step` is not in the Qwen3.6 Linear-Attn sequence (only SSM steps + AttnNorm)
/// - `cfg.qwen35.is_none()` (other architectures don't have the SSM buffers)
pub(super) fn dump_after_step(fwd: &mut Forward, ctx: &ExecCtx, step: &LayerStep) {
    // Caller (executor/mod.rs) gates layer + position; here we just dump.
    let pos_tag = match ctx.mode {
        super::executor::ExecMode::Decode { position, .. } => position,
        _ => 0,
    };
    let layer_tag = format!("L{}p{}", ctx.layer, pos_tag);
    let cfg = fwd.config.clone();
    let Some(spec) = cfg.qwen35.as_ref() else { return; };
    let head_k = spec.head_k_dim() as u64;          // 128
    let n_k    = spec.num_k_heads() as u64;          // 16
    let head_v = spec.head_v_dim() as u64;          // 128
    let n_v    = spec.num_v_heads() as u64;          // 48
    let inner  = spec.ssm_d_inner as u64;            // 6144 = head_v × n_v
    let conv_c = spec.conv_channels() as u64;        // 10240 = 2·head_k·n_k + head_v·n_v
    let qk_slice = (head_k * n_k) as usize;          // 2048

    // Helper to print after running an attempt; ignore Err to avoid
    // killing the run on a single dump failure.
    let dev = ctx.dev;
    let cmd = ctx.cmd;

    match step {
        LayerStep::AttnNorm => {
            let h = fwd.cur().hidden_norm.handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, cfg.hidden_dim as usize, &format!("{layer_tag} attn_norm"));
        }
        LayerStep::AttnQkvProj { .. } => {
            let h = fwd.ssm_qkv_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, conv_c as usize, &format!("{layer_tag} ssm_qkv_proj"));
        }
        LayerStep::AttnGateZProj { .. } => {
            let h = fwd.ssm_z_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, inner as usize, &format!("{layer_tag} ssm_gate_z_proj"));
        }
        LayerStep::SsmBetaProj { .. } => {
            let h = fwd.ssm_beta_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, n_v as usize, &format!("{layer_tag} ssm_beta_proj"));
        }
        LayerStep::SsmAlphaGate { .. } => {
            let h = fwd.ssm_gate_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, n_v as usize, &format!("{layer_tag} ssm_alpha_gate"));
        }
        LayerStep::SsmConv1d { .. } => {
            let h = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, conv_c as usize, &format!("{layer_tag} ssm_conv1d"));
        }
        LayerStep::SsmSilu { .. } => {
            let h = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, conv_c as usize, &format!("{layer_tag} ssm_silu"));
        }
        LayerStep::SsmQkL2Norm { .. } => {
            let h = fwd.ssm_conv_output_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, qk_slice, &format!("{layer_tag} ssm_qk_l2_norm_Q"));
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, qk_slice as u64, qk_slice, &format!("{layer_tag} ssm_qk_l2_norm_K"));
        }
        LayerStep::SsmRepeatQK { .. } => {
            let q = fwd.ssm_qrep_buf.as_ref().unwrap().handle;
            let k = fwd.ssm_krep_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 0, inner as usize, &format!("{layer_tag} ssm_repeat_Q"));
            let _ = dump_buffer_layer0(fwd, dev, cmd, k, 0, inner as usize, &format!("{layer_tag} ssm_repeat_K"));
        }
        LayerStep::GatedDeltaNet { .. } => {
            let dst = fwd.ssm_gdn_out_buf.as_ref().unwrap().handle;
            // GDN output (first H × S_v = 6144 floats) + sample of new state.
            let _ = dump_buffer_layer0(fwd, dev, cmd, dst, 0, inner as usize, &format!("{layer_tag} gdn_output"));
            // The next slot is the new state — sample first 512 floats only.
            let s_off = inner; // floats
            let _ = dump_buffer_layer0(fwd, dev, cmd, dst, s_off, 512, &format!("{layer_tag} gdn_new_state"));
        }
        LayerStep::NormGated { .. } => {
            let h = fwd.ssm_norm_out_buf.as_ref().unwrap().handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, inner as usize, &format!("{layer_tag} norm_gated"));
        }
        LayerStep::SsmOutProj { .. } => {
            let h = fwd.cur().o_buf.handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, cfg.hidden_dim as usize, &format!("{layer_tag} ssm_out_proj"));
        }
        LayerStep::AttnResidualAdd => {
            let r = fwd.cur().res1.handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, r, 0, cfg.hidden_dim as usize, &format!("{layer_tag} attn_residual"));
        }
        LayerStep::PreFfnNorm => {
            let h = fwd.cur().hidden_norm.handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, h, 0, cfg.hidden_dim as usize, &format!("{layer_tag} post_attn_norm"));
        }
        LayerStep::FfnResidualAdd => {
            // Output buffer for this layer = decode_io's `output`. Read it
            // via the executor's ExecMode::Decode { output, .. }.
            if let super::executor::ExecMode::Decode { output, .. } = ctx.mode {
                let _ = dump_buffer_layer0(fwd, dev, cmd, output, 0, cfg.hidden_dim as usize, &format!("{layer_tag} ffn_residual"));
            }
        }
        // ── Sprint G-2h Full-Attn dump points (L3 / L7 / … bisect) ──
        LayerStep::AttnQGateProj { q_dim } => {
            let q = fwd.cur().q_buf.handle;
            let q_dim_u = *q_dim as usize;
            // First Q-only slice (post-deinterleave when env-gate on; raw
            // interleaved otherwise).
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 0, q_dim_u, &format!("{layer_tag} qgate_proj_Q"));
            // Gate slice (post-deinterleave: contiguous at offset q_dim;
            // pre-deinterleave: actually 2nd dim of head 0).
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, q_dim_u as u64, q_dim_u, &format!("{layer_tag} qgate_proj_G"));
            // Probe positions 0, 256, 512 for layout verification:
            let head_dim = cfg.head_dim as u64;
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 0, 16, &format!("{layer_tag} qgate_first16"));
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, head_dim, 16, &format!("{layer_tag} qgate_at_head_dim"));
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 2 * head_dim, 16, &format!("{layer_tag} qgate_at_2x_head_dim"));
        }
        LayerStep::QNormRope { .. } | LayerStep::QRope { .. } => {
            let q = fwd.cur().q_buf.handle;
            let q_dim = (cfg.n_heads * cfg.head_dim) as usize;
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 0, q_dim, &format!("{layer_tag} q_post_normrope"));
            // Dump head 0 first 16 floats specifically.
            let _ = dump_buffer_layer0(fwd, dev, cmd, q, 0, 16, &format!("{layer_tag} q_h0_first16"));
        }
        LayerStep::KProj => {
            let k = fwd.cur().k_buf.handle;
            // Determine kv_dim via metadata (n_kv_heads for FA layer).
            let kv_dim = if let Some(spec) = cfg.qwen35.as_ref() {
                if spec.is_full_attention_layer(ctx.layer) {
                    (spec.n_head_kv_full_attn * cfg.head_dim) as usize
                } else { 0 }
            } else { (cfg.n_kv_heads * cfg.head_dim) as usize };
            if kv_dim > 0 {
                let _ = dump_buffer_layer0(fwd, dev, cmd, k, 0, kv_dim, &format!("{layer_tag} k_proj"));
            }
        }
        LayerStep::KNormRope { .. } | LayerStep::KRope { .. } => {
            let k = fwd.cur().k_buf.handle;
            let kv_dim = if let Some(spec) = cfg.qwen35.as_ref() {
                if spec.is_full_attention_layer(ctx.layer) {
                    (spec.n_head_kv_full_attn * cfg.head_dim) as usize
                } else { 0 }
            } else { (cfg.n_kv_heads * cfg.head_dim) as usize };
            if kv_dim > 0 {
                let _ = dump_buffer_layer0(fwd, dev, cmd, k, 0, kv_dim, &format!("{layer_tag} k_post_normrope"));
                let _ = dump_buffer_layer0(fwd, dev, cmd, k, 0, 16, &format!("{layer_tag} k_h0_first16"));
            }
        }
        LayerStep::VProj => {
            let v = fwd.cur().v_buf.handle;
            let kv_dim = if let Some(spec) = cfg.qwen35.as_ref() {
                if spec.is_full_attention_layer(ctx.layer) {
                    (spec.n_head_kv_full_attn * cfg.head_dim) as usize
                } else { 0 }
            } else { (cfg.n_kv_heads * cfg.head_dim) as usize };
            if kv_dim > 0 {
                let _ = dump_buffer_layer0(fwd, dev, cmd, v, 0, kv_dim, &format!("{layer_tag} v_proj"));
            }
        }
        LayerStep::Attention { .. } => {
            let a = fwd.cur().attn_out.handle;
            let q_dim = (cfg.n_heads * cfg.head_dim) as usize;
            let _ = dump_buffer_layer0(fwd, dev, cmd, a, 0, q_dim, &format!("{layer_tag} attn_pregate"));
        }
        LayerStep::AttnGatedOutput { .. } => {
            let a = fwd.cur().attn_out.handle;
            let q_dim = (cfg.n_heads * cfg.head_dim) as usize;
            let _ = dump_buffer_layer0(fwd, dev, cmd, a, 0, q_dim, &format!("{layer_tag} attn_gated"));
        }
        LayerStep::OProj => {
            let o = fwd.cur().o_buf.handle;
            let _ = dump_buffer_layer0(fwd, dev, cmd, o, 0, cfg.hidden_dim as usize, &format!("{layer_tag} o_proj"));
        }
        _ => {}
    }
}

/// Sprint 43D-1 diagnose helper — env-gated logit-distribution dump.
/// Set `VF_LOGIT_DUMP=1` (or `=2` for verbose) to print per-step
/// argmax / top5 / mean / min / max / NaN-count to stderr during
/// `wait_and_read_logits`. No-op otherwise. Used to bisect whether
/// the Gemma-4 `<pad>`-argmax-collapse is a coherent distribution
/// gone wrong, an all-NaN forward, or a particular logit slot
/// dominating from a soft-cap clipping artifact.
pub(super) fn maybe_dump_logits(tag: &str, logits: &[f32]) {
    let level: u32 = std::env::var("VF_LOGIT_DUMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if level == 0 {
        return;
    }
    let n = logits.len();
    let mut max_v = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut sum: f64 = 0.0;
    let mut nan_count: u32 = 0;
    let mut inf_count: u32 = 0;
    let mut max_idx: usize = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        sum += v as f64;
        if v > max_v {
            max_v = v;
            max_idx = i;
        }
        if v < min_v {
            min_v = v;
        }
    }
    let valid = (n as u32) - nan_count - inf_count;
    let mean = if valid > 0 { sum / (valid as f64) } else { 0.0 };

    // Top-5 by descending logit value (NaN/Inf-skipped).
    let mut sorted: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(usize, f32)> = sorted.into_iter().take(5).collect();

    eprintln!(
        "[VF_LOGIT_DUMP] {tag}: n={n} argmax_idx={max_idx} max={max_v:.4} \
         min={min_v:.4} mean={mean:.4} nan={nan_count} inf={inf_count}"
    );
    eprintln!("[VF_LOGIT_DUMP] {tag}: top5={top5:?}");
    if level >= 2 {
        // Verbose: dump first 16 + last 16 logits to spot patterns
        // (e.g. id 0..15 mostly zero suggests pad-collapse artifact).
        let head: Vec<(usize, f32)> = logits.iter().enumerate().take(16).map(|(i, &v)| (i, v)).collect();
        let tail_start = n.saturating_sub(16);
        let tail: Vec<(usize, f32)> = logits[tail_start..].iter().enumerate()
            .map(|(i, &v)| (tail_start + i, v)).collect();
        eprintln!("[VF_LOGIT_DUMP] {tag}: head16={head:?}");
        eprintln!("[VF_LOGIT_DUMP] {tag}: tail16={tail:?}");
    }
}

/// Sprint 43F Block A — dump stats of `hidden_staging` contents
/// (= hidden_norm post-final-RMSNorm, copied via the in-CB
/// VF_FINAL_NORM_DUMP block). Always-on print when called; gating
/// by env happens in the call sites.
pub(super) fn maybe_dump_hidden_staging(tag: &str, hidden: &[f32]) {
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
    let valid = (hidden.len() as u32) - nan - inf;
    let mean = if valid > 0 { sum / (valid as f64) } else { 0.0 };
    eprintln!(
        "[MAIN-CB-DUMP] {tag}: n={} nan={} inf={} min={:.4} max={:.4} mean={:.4}",
        hidden.len(), nan, inf, mn, mx, mean,
    );
    let head: Vec<f32> = hidden.iter().take(8).copied().collect();
    eprintln!("[MAIN-CB-DUMP] {tag}: first8 = {:?}", head);
}
