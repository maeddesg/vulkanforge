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
    compute_barrier, layer_weight, layer_weight_with_offset, layer_weight_opt,
    layer_weight_scale_block, layer_weight_scale_buf, layer_weight_scale_scalar,
    layer_weight_shader, n_kv_heads_for,
};
use super::state::{DebugTarget, Forward};
use super::super::buffers::GpuBuffer;
use super::super::pipeline::GatedDeltaNetPushConstants;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

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
        let (wq, wq_off, wq_sz) = layer_weight_with_offset(model, layer, "attn_q.weight");
        let (wk, wk_off, wk_sz) = layer_weight_with_offset(model, layer, "attn_k.weight");
        let (wv, wv_off, wv_sz) = layer_weight_with_offset(model, layer, "attn_v.weight");
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
            self.run_gemv(dev, registry, cmd, sq, wq, wq_off, wq_sz, in_h, q_h,
                cfg.hidden_dim, cfg.n_heads * cfg.head_dim, scale_q, "gemv_q");
        }
        if let Some(s) = sb_k {
            self.run_gemv_fp8_dispatch(dev, cmd, wk, s, in_h, k_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_k.weight"), "gemv_k");
        } else {
            self.run_gemv(dev, registry, cmd, sk, wk, wk_off, wk_sz, in_h, k_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim, scale_k, "gemv_k");
        }
        if let Some(s) = sb_v {
            self.run_gemv_fp8_dispatch(dev, cmd, wv, s, in_h, v_h,
                cfg.hidden_dim, n_kv_heads_for(&cfg, layer) * cfg.head_dim,
                layer_weight_scale_block(model, layer, "attn_v.weight"), "gemv_v");
        } else {
            self.run_gemv(dev, registry, cmd, sv, wv, wv_off, wv_sz, in_h, v_h,
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

    /// De-risk substep helper — stage the first `count` floats of a
    /// GpuOnly buffer into the host-visible `scratch_a` and read them.
    /// `count` must be <= scratch_a capacity (hidden_dim). Used to read
    /// k/v rows (kv_dim <= hidden) for the qwen35 Full-Attn substep diff.
    fn stage_buf_prefix(
        &self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
        src: vk::Buffer,
        count: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = (count as u64) * 4;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(src)
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
                cmd, src, self.cur().scratch_a.handle, std::slice::from_ref(&copy),
            );
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
        let raw = self.cur().scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(&raw[..(count as usize) * 4]).to_vec())
    }

    /// De-risk substep helper — stage `count` floats from `src` starting at
    /// float offset `src_off`, chunked through `scratch_a` (≤ hidden per
    /// pass) so buffers larger than scratch_a (q_dim=6144, 2·q_dim=12288)
    /// can be read for the gated-output input bisect.
    fn stage_buf_region(
        &self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
        src: vk::Buffer,
        src_off: u32,
        count: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let hidden = self.config.hidden_dim;
        let mut out = vec![0f32; count as usize];
        let mut done = 0u32;
        while done < count {
            let chunk = (count - done).min(hidden);
            let src_byte_off = ((src_off + done) as u64) * 4;
            let bytes = (chunk as u64) * 4;
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                let pre = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(src).offset(0).size(vk::WHOLE_SIZE);
                dev.device.cmd_pipeline_barrier(
                    cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(), &[], std::slice::from_ref(&pre), &[],
                );
                let copy = vk::BufferCopy::default()
                    .src_offset(src_byte_off).dst_offset(0).size(bytes);
                dev.device.cmd_copy_buffer(
                    cmd, src, self.cur().scratch_a.handle, std::slice::from_ref(&copy),
                );
                let post = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.cur().scratch_a.handle).offset(0).size(vk::WHOLE_SIZE);
                dev.device.cmd_pipeline_barrier(
                    cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(), &[], std::slice::from_ref(&post), &[],
                );
            })?;
            let raw = self.cur().scratch_a.read_bytes()?;
            out[done as usize..(done + chunk) as usize]
                .copy_from_slice(bytemuck::cast_slice::<u8, f32>(&raw[..(chunk as usize) * 4]));
            done += chunk;
        }
        Ok(out)
    }

    /// De-risk gated-output bisect — run ONE real DECODE layer and return
    /// `q_buf` (2·q_dim: the concat [Q(q_dim), G(q_dim)] after the Q-Gate
    /// deinterleave; the gate G half is the sigmoid_mul gate source).
    pub fn forward_layer_debug_qbuf(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let q_total = cfg.n_heads * cfg.head_dim * 2;
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf.write_bytes(bytemuck::bytes_of(&position))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
        })?;
        self.stage_buf_region(dev, cmd_ctx, self.cur().q_buf.handle, 0, q_total)
    }

    /// De-risk gated-output bisect — return `batch_qgate` (2·q_dim) after a
    /// batched single layer (the interleaved [Q_h0,G_h0,…] gate source).
    pub fn forward_layer_batch_qgate(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _ = self.forward_layer_batch_debug(dev, registry, cmd_ctx, model, layer, 1, input)?;
        let q_total = self.config.n_heads * self.config.head_dim * 2;
        let qgate = self.batch_qgate.as_ref()
            .ok_or("batch_qgate not allocated (not qwen35)")?
            .handle;
        self.stage_buf_region(dev, cmd_ctx, qgate, 0, q_total)
    }

    /// De-risk gated-output localization — run ONE real DECODE layer and
    /// return `attn_out` (q_dim, head-major [head0(hd)..head_{nh-1}(hd)]).
    /// For a qwen35 full-attn layer this is the buffer the gated-output
    /// `sigmoid_mul` writes IN PLACE; OProj reads it (doesn't write) and
    /// AttnResidualAdd reads `o_buf` (not attn_out) and qwen35 emits no
    /// PostAttnNorm, so the gated attention output survives to the layer
    /// boundary. Compared against `forward_layer_batch_attn_out` to localize
    /// the deterministic gate-application divergence per head. KV reset so
    /// pos 0 attends only to itself (matches the batched single-token path).
    pub fn forward_layer_debug_attn_out(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let q_dim = cfg.n_heads * cfg.head_dim;
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf.write_bytes(bytemuck::bytes_of(&position))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        self.kv_cache.reset();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
        })?;
        self.stage_buf_region(dev, cmd_ctx, self.cur().attn_out.handle, 0, q_dim)
    }

    /// De-risk gated-output localization — run ONE BatchExec layer @N=1 and
    /// return `batch_attn_out` (q_dim). qwen35 emits no PostAttnNorm and
    /// AttnResidualAdd reads `batch_o`, so `batch_attn_out` holds the
    /// post-gate attention output at the layer boundary (its in-place
    /// `sigmoid_mul` result). Element-wise/per-head diff vs the decode
    /// sibling pins the gate-application misalignment.
    pub fn forward_layer_batch_attn_out(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _ = self.forward_layer_batch_debug(dev, registry, cmd_ctx, model, layer, 1, input)?;
        let q_dim = self.config.n_heads * self.config.head_dim;
        self.stage_buf_region(dev, cmd_ctx, self.batch_attn_out.handle, 0, q_dim)
    }

    /// De-risk gated-output localization — run ONE DECODE layer and return
    /// the o-proj output `o_buf` (hidden_dim). qwen35 emits no PostAttnNorm
    /// and AttnResidualAdd only reads o_buf, so it survives to the boundary.
    /// Decode o-proj is a `gemv` (reads the FP32 attn_out activation
    /// directly); the batch sibling is a `gemm` that QUANTIZES its input —
    /// comparing the two outputs tests whether o-proj input-quantization is
    /// the amplifier of the gate-range-compressed attn_out difference.
    pub fn forward_layer_debug_o_buf(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf.write_bytes(bytemuck::bytes_of(&position))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        self.kv_cache.reset();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
        })?;
        self.stage_buf_prefix(dev, cmd_ctx, self.cur().o_buf.handle, hidden)
    }

    /// De-risk gated-output localization — run ONE BatchExec layer @N=1 and
    /// return the o-proj output `batch_o` (hidden_dim). The FFN does not
    /// reuse batch_o, so it survives to the boundary.
    pub fn forward_layer_batch_o(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _ = self.forward_layer_batch_debug(dev, registry, cmd_ctx, model, layer, 1, input)?;
        let hidden = self.config.hidden_dim;
        self.stage_buf_prefix(dev, cmd_ctx, self.batch_o.handle, hidden)
    }

    /// GDN-Completion #1 verification (gated `VF_QWEN35_GDN_VERIFY`): verify
    /// the `gated_delta_net.comp` recurrence over `n_tokens=N` (register-
    /// resident state carried across the internal token loop) against the
    /// per-token decode recurrence (`n_tokens=1` ×N with buffer state-carry,
    /// mirroring `step_gated_delta_net`'s dst→state copy-back). Identical
    /// synthetic inputs from a zero initial state → per-position cos for
    /// N = 2,4,8,16. RECURRENCE-ONLY (the shader does NOT do the causal
    /// conv; that is a separate wiring-sprint check). Self-contained
    /// host-visible buffers — the production ssm_* buffers are decode-sized
    /// (n_tokens=1) and too small for N. No production behavior (BatchExec
    /// GDN stays stubbed). Pre-check gate before building the GDN wiring.
    pub fn gdn_recurrence_verify(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        allocator: &mut Allocator,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let spec = self.config.qwen35.clone().expect("gdn_recurrence_verify: not qwen35");
        let h = spec.num_v_heads();           // 48
        let s_v = spec.ssm_d_state;           // 128
        let head_k = spec.head_k_dim();       // 128
        let scale = 1.0f32 / (s_v as f32).sqrt();
        let (hu, svu, hku) = (h as usize, s_v as usize, head_k as usize);

        let q_per = hku * hu;                 // 6144  (head_k × H)
        let v_per = svu * hu;                 // 6144  (S_v × H)
        let g_per = hu;                       // 48    (scalar per head)
        let state_floats = hu * svu * svu;    // 786432 (H × S_v × S_v)
        let n_max = 16usize;

        // Synthetic generators — identical for path-A token-t and path-B
        // token-t (indexed by GLOBAL token t). g ≤ 0 so exp(g) ∈ (0,1] (decay).
        let qf = |t: usize, hh: usize, i: usize| 0.15 * (0.30 * t as f32 + 0.020 * i as f32 + 0.05 * hh as f32).sin() + 0.05 * (0.011 * i as f32).cos();
        let kf = |t: usize, hh: usize, i: usize| 0.12 * (0.25 * t as f32 + 0.017 * i as f32 + 0.03 * hh as f32).cos() + 0.04 * (0.009 * (i * (hh + 1)) as f32).sin();
        let vf = |t: usize, hh: usize, c: usize| 0.20 * (0.20 * t as f32 + 0.013 * c as f32 + 0.04 * hh as f32).sin();
        let gf = |t: usize, hh: usize| -0.10 - 0.02 * (((t + hh) % 5) as f32);
        let bf = |t: usize, hh: usize| 0.5 + 0.1 * (0.4 * t as f32 + 0.2 * hh as f32).sin();

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut mk = |alloc: &mut Allocator, floats: usize, name: &str| {
            GpuBuffer::new(&dev.device, alloc, (floats as u64) * 4, usage, MemoryLocation::CpuToGpu, name)
        };
        // Path A (N_max tokens), shared state, path B (1 token).
        let mut q_a = mk(allocator, n_max * q_per, "gdnv_qA")?;
        let mut k_a = mk(allocator, n_max * q_per, "gdnv_kA")?;
        let mut v_a = mk(allocator, n_max * v_per, "gdnv_vA")?;
        let mut g_a = mk(allocator, n_max * g_per, "gdnv_gA")?;
        let mut b_a = mk(allocator, n_max * g_per, "gdnv_bA")?;
        let mut state = mk(allocator, state_floats, "gdnv_state")?;
        let dst_a = mk(allocator, n_max * v_per + state_floats, "gdnv_dstA")?;
        let mut q_1 = mk(allocator, q_per, "gdnv_q1")?;
        let mut k_1 = mk(allocator, q_per, "gdnv_k1")?;
        let mut v_1 = mk(allocator, v_per, "gdnv_v1")?;
        let mut g_1 = mk(allocator, g_per, "gdnv_g1")?;
        let mut b_1 = mk(allocator, g_per, "gdnv_b1")?;
        let dst_1 = mk(allocator, v_per + state_floats, "gdnv_dst1")?;

        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx, mut omax) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..a.len().min(b.len()) {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y; na += x * x; nb += y * y;
                let d = (x - y).abs(); if d > mx { mx = d; } if x.abs() > omax { omax = x.abs(); }
            }
            let c = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 };
            (c, if omax > 0.0 { mx / omax } else { mx })
        };
        let host_barrier = |dev: &VulkanDevice, cmd: vk::CommandBuffer| {
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe { dev.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), std::slice::from_ref(&mb), &[], &[]); }
        };

        let zero = vec![0f32; state_floats];
        let st_bytes = (state_floats as u64) * 4;
        println!("\n=== GDN recurrence verify (shader n_tokens=N vs decode n_tokens=1×N), h={h} s_v={s_v} head_k={head_k} ===");
        let mut all_ok = true;
        for &n in &[2usize, 4, 8, 16] {
            // ---- fill path-A inputs (tokens 0..n) ----
            let mut qv = vec![0f32; n * q_per]; let mut kv = vec![0f32; n * q_per];
            let mut vv = vec![0f32; n * v_per]; let mut gv = vec![0f32; n * g_per]; let mut bv = vec![0f32; n * g_per];
            for t in 0..n {
                for hh in 0..hu {
                    for i in 0..hku { qv[t * q_per + hh * hku + i] = qf(t, hh, i); kv[t * q_per + hh * hku + i] = kf(t, hh, i); }
                    for c in 0..svu { vv[t * v_per + hh * svu + c] = vf(t, hh, c); }
                    gv[t * g_per + hh] = gf(t, hh); bv[t * g_per + hh] = bf(t, hh);
                }
            }
            q_a.write_bytes(bytemuck::cast_slice(&qv))?; k_a.write_bytes(bytemuck::cast_slice(&kv))?;
            v_a.write_bytes(bytemuck::cast_slice(&vv))?; g_a.write_bytes(bytemuck::cast_slice(&gv))?;
            b_a.write_bytes(bytemuck::cast_slice(&bv))?; state.write_bytes(bytemuck::cast_slice(&zero))?;

            // ---- Path A: ONE dispatch n_tokens=N ----
            let push_a = GatedDeltaNetPushConstants {
                h, n_tokens: n as u32, n_seqs: 1, s_off: (s_v * h) * (n as u32),
                sq1: head_k, sq2: head_k * h, sq3: head_k * h,
                sv1: s_v, sv2: s_v * h, sv3: s_v * h, sb1: 1, sb2: h, sb3: h,
                neq1: h, rq3: 1, scale, k: 1,
            };
            self.reset_descriptor_pool_and_cache(dev)?;
            let (qh, kh, vh, gh, bh, sh, dh) = (q_a.handle, k_a.handle, v_a.handle, g_a.handle, b_a.handle, state.handle, dst_a.handle);
            let v_bytes_a = (n * v_per) as u64 * 4;
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                self.run_gated_delta_net(dev, registry, cmd, qh, 0, kh, 0, vh, 0, v_bytes_a, gh, 0, bh, 0, sh, 0, st_bytes, dh, 0, s_v, 2, &push_a, "gdnv_a");
                host_barrier(dev, cmd);
            })?;
            let out_a = bytemuck::cast_slice::<u8, f32>(&dst_a.read_bytes()?[..(n * v_per) * 4]).to_vec();

            // ---- Path B: n_tokens=1 ×N with buffer state-carry ----
            state.write_bytes(bytemuck::cast_slice(&zero))?;
            let push_b = GatedDeltaNetPushConstants {
                h, n_tokens: 1, n_seqs: 1, s_off: s_v * h,
                sq1: head_k, sq2: head_k * h, sq3: head_k * h,
                sv1: s_v, sv2: s_v * h, sv3: s_v * h, sb1: 1, sb2: h, sb3: h,
                neq1: h, rq3: 1, scale, k: 1,
            };
            let mut out_b = vec![0f32; n * v_per];
            let (q1h, k1h, v1h, g1h, b1h, d1h) = (q_1.handle, k_1.handle, v_1.handle, g_1.handle, b_1.handle, dst_1.handle);
            let s_off_b = (s_v * h) as usize;
            for t in 0..n {
                let mut q1 = vec![0f32; q_per]; let mut k1 = vec![0f32; q_per];
                let mut v1 = vec![0f32; v_per]; let mut g1 = vec![0f32; g_per]; let mut b1 = vec![0f32; g_per];
                for hh in 0..hu {
                    for i in 0..hku { q1[hh * hku + i] = qf(t, hh, i); k1[hh * hku + i] = kf(t, hh, i); }
                    for c in 0..svu { v1[hh * svu + c] = vf(t, hh, c); }
                    g1[hh] = gf(t, hh); b1[hh] = bf(t, hh);
                }
                q_1.write_bytes(bytemuck::cast_slice(&q1))?; k_1.write_bytes(bytemuck::cast_slice(&k1))?;
                v_1.write_bytes(bytemuck::cast_slice(&v1))?; g_1.write_bytes(bytemuck::cast_slice(&g1))?;
                b_1.write_bytes(bytemuck::cast_slice(&b1))?;
                let sh2 = state.handle;
                self.reset_descriptor_pool_and_cache(dev)?;
                cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                    self.run_gated_delta_net(dev, registry, cmd, q1h, 0, k1h, 0, v1h, 0, (v_per as u64) * 4, g1h, 0, b1h, 0, sh2, 0, st_bytes, d1h, 0, s_v, 2, &push_b, "gdnv_b");
                    host_barrier(dev, cmd);
                })?;
                let db = dst_1.read_bytes()?;
                out_b[t * v_per..(t + 1) * v_per].copy_from_slice(bytemuck::cast_slice::<u8, f32>(&db[..v_per * 4]));
                // carry: dst1[s_off .. s_off+state] → state buffer (mirrors decode copy-back)
                let new_state = bytemuck::cast_slice::<u8, f32>(&db[s_off_b * 4..(s_off_b + state_floats) * 4]).to_vec();
                state.write_bytes(bytemuck::cast_slice(&new_state))?;
            }

            // ---- per-position cos ----
            let (mut worst_c, mut worst_r, mut worst_t) = (1f64, 0f64, 0usize);
            for t in 0..n {
                let (c, r) = cosf(&out_a[t * v_per..(t + 1) * v_per], &out_b[t * v_per..(t + 1) * v_per]);
                if c < worst_c { worst_c = c; worst_t = t; }
                if r > worst_r { worst_r = r; }
                if n <= 4 || t == 0 || t == 1 || t == n - 1 { println!("  N={n} pos {t:2}: cos={c:.6} rel={r:.3e}"); }
            }
            let ok = worst_c > 0.999 && worst_r < 0.02;
            all_ok &= ok;
            println!("  => N={n}: worst_cos={worst_c:.6} (pos {worst_t}) worst_rel={worst_r:.3e} -> {}",
                if ok { "MATCH" } else { "MISMATCH" });
        }
        println!("=== GDN VERDICT: {} ===",
            if all_ok { "MATCH — recurrence over n_tokens=N verified → GO wiring" }
            else { "MISMATCH — shader n_tokens=N loop drift (see position pattern)" });

        for buf in [q_a, k_a, v_a, g_a, b_a, state, dst_a, q_1, k_1, v_1, g_1, b_1, dst_1] {
            buf.destroy(&dev.device, allocator);
        }
        Ok(all_ok)
    }

    /// GDN-Completion #2 Step 0 (gated `VF_QWEN35_GDN_VERIFY`): verify the
    /// causal conv (`ssm_conv.comp`) over `n_t=N` against the per-token
    /// decode conv (`n_t=1` ×N with conv-state window carry). The shader
    /// itself windows over output tokens (grid `i2`, reads `[i2..i2+nc]`);
    /// correctness over N hinges on the conv_input layout
    /// `[conv_state(nc-1) ++ N current]` + zero-init prefill padding. We
    /// build conv_input on the host directly (the decode `ssm_conv_setup`
    /// is single-token): Path A = `[zero-pad(nc-1) ++ N]` one dispatch;
    /// Path B = `[state(nc-1) ++ 1]` per token with host state-slide
    /// (mirrors `step_ssm_conv1d`). Per-position bit-identity, N=2,4,8,16.
    /// Verification-only; no production behavior.
    pub fn gdn_conv_verify(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        allocator: &mut Allocator,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let spec = self.config.qwen35.clone().expect("gdn_conv_verify: not qwen35");
        let nr = spec.conv_channels() as usize;   // 10240
        let nc = spec.ssm_d_conv as usize;        // 4
        let pad = nc - 1;                          // 3
        let n_max = 16usize;
        // Synthetic per-(channel,token) input and per-(channel,tap) kernel.
        let inf = |ch: usize, t: usize| 0.10 * (0.020 * ch as f32 + 0.50 * t as f32).sin() + 0.05 * (0.30 * t as f32).cos();
        let wf = |ch: usize, j: usize| 0.20 * (0.70 * j as f32 + 0.001 * ch as f32).cos();

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut mk = |alloc: &mut Allocator, floats: usize, name: &str| {
            GpuBuffer::new(&dev.device, alloc, (floats as u64) * 4, usage, MemoryLocation::CpuToGpu, name)
        };
        let mut w_b = mk(allocator, nr * nc, "convv_w")?;
        let mut wv = vec![0f32; nr * nc];
        for ch in 0..nr { for j in 0..nc { wv[ch * nc + j] = wf(ch, j); } }
        w_b.write_bytes(bytemuck::cast_slice(&wv))?;
        let mut in_a = mk(allocator, nr * (pad + n_max), "convv_inA")?;
        let dst_a = mk(allocator, nr * n_max, "convv_dstA")?;
        let mut in_b = mk(allocator, nr * nc, "convv_inB")?;
        let dst_b = mk(allocator, nr, "convv_dstB")?;

        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx, mut omax) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..a.len().min(b.len()) {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y; na += x * x; nb += y * y;
                let d = (x - y).abs(); if d > mx { mx = d; } if x.abs() > omax { omax = x.abs(); }
            }
            (if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 }, if omax > 0.0 { mx / omax } else { mx })
        };
        let host_barrier = |dev: &VulkanDevice, cmd: vk::CommandBuffer| {
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe { dev.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), std::slice::from_ref(&mb), &[], &[]); }
        };
        let (wh, iah, dah, ibh, dbh) = (w_b.handle, in_a.handle, dst_a.handle, in_b.handle, dst_b.handle);
        println!("\n=== GDN conv verify (ssm_conv n_t=N vs decode n_t=1×N), nr={nr} nc={nc} ===");
        let mut all_ok = true;
        for &n in &[2usize, 4, 8, 16] {
            let ncs_a = pad + n;
            // Path A: conv_input = [zero(pad) ++ input[ch][0..n]] per channel.
            let mut iva = vec![0f32; nr * ncs_a];
            for ch in 0..nr { for t in 0..n { iva[ch * ncs_a + pad + t] = inf(ch, t); } }
            in_a.write_bytes(bytemuck::cast_slice(&iva))?;
            self.reset_descriptor_pool_and_cache(dev)?;
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                self.run_ssm_conv(dev, registry, cmd, iah, wh, dah, 0, nc as u32, ncs_a as u32, nr as u32, n as u32, 1, "convv_a");
                host_barrier(dev, cmd);
            })?;
            let out_a = bytemuck::cast_slice::<u8, f32>(&dst_a.read_bytes()?[..(nr * n) * 4]).to_vec();

            // Path B: zero conv-state, n_t=1 ×N with host window slide.
            let mut state = vec![0f32; nr * pad];
            let mut out_b = vec![0f32; nr * n];
            for t in 0..n {
                let mut ivb = vec![0f32; nr * nc];
                for ch in 0..nr {
                    for s in 0..pad { ivb[ch * nc + s] = state[ch * pad + s]; }
                    ivb[ch * nc + pad] = inf(ch, t);
                }
                in_b.write_bytes(bytemuck::cast_slice(&ivb))?;
                self.reset_descriptor_pool_and_cache(dev)?;
                cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                    self.run_ssm_conv(dev, registry, cmd, ibh, wh, dbh, 0, nc as u32, nc as u32, nr as u32, 1, 1, "convv_b");
                    host_barrier(dev, cmd);
                })?;
                let db = dst_b.read_bytes()?;
                out_b[t * nr..(t + 1) * nr].copy_from_slice(bytemuck::cast_slice::<u8, f32>(&db[..nr * 4]));
                // slide window: new state = [state[1..], input_t]
                for ch in 0..nr {
                    for s in 0..pad - 1 { state[ch * pad + s] = state[ch * pad + s + 1]; }
                    state[ch * pad + pad - 1] = inf(ch, t);
                }
            }

            let (mut worst_c, mut worst_r, mut worst_t) = (1f64, 0f64, 0usize);
            for t in 0..n {
                let (c, r) = cosf(&out_a[t * nr..(t + 1) * nr], &out_b[t * nr..(t + 1) * nr]);
                if c < worst_c { worst_c = c; worst_t = t; }
                if r > worst_r { worst_r = r; }
                if n <= 4 || t == 0 || t == 1 || t == n - 1 { println!("  conv N={n} pos {t:2}: cos={c:.6} rel={r:.3e}"); }
            }
            let ok = worst_c > 0.999 && worst_r < 1e-4;
            all_ok &= ok;
            println!("  => conv N={n}: worst_cos={worst_c:.6} (pos {worst_t}) worst_rel={worst_r:.3e} -> {}",
                if ok { "MATCH" } else { "MISMATCH" });
        }
        println!("=== GDN CONV VERDICT: {} ===",
            if all_ok { "MATCH — conv over n_t=N verified" } else { "MISMATCH — conv n_t=N layout/window/state drift" });
        for buf in [w_b, in_a, dst_a, in_b, dst_b] { buf.destroy(&dev.device, allocator); }
        Ok(all_ok)
    }

    /// GDN-Completion #3 Step 1 (gated `VF_QWEN35_GDN_VERIFY`): verify the
    /// batched input PROJECTION (`b_step_attn_qkv_proj`, GEMM M=N) against
    /// the per-token decode projection (GEMV). qkv = `attn_qkv.weight @
    /// rms_norm(input)` depends only on the normed layer input (NOT on
    /// recurrent state), so we compare `ssm_qkv_buf` from one batched
    /// recurrent-layer run vs N single-token decode runs. GEMM reorders the
    /// FP accumulation vs GEMV → expect cos→1.0 at the ~1e-5 baseline, NOT
    /// bit-identical. A divergence ≫1e-5 = projection wiring/sizing bug.
    pub fn gdn_proj_verify(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let spec = cfg.qwen35.as_ref().expect("gdn_proj_verify: not qwen35");
        let hidden = cfg.hidden_dim as usize;
        let cc = spec.conv_channels() as usize;
        // First recurrent (Linear-Attn) layer.
        let layer = (0..spec.block_count).find(|&l| spec.is_recurrent_layer(l)).expect("no recurrent layer");
        let n_max = 8usize;
        let mut x = vec![0f32; n_max * hidden];
        for p in 0..n_max { for i in 0..hidden {
            x[p * hidden + i] = 0.18 * (0.29 * p as f32 + 0.015 * i as f32).sin() + 0.06 * (0.004 * i as f32).cos() - 0.10;
        }}
        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx, mut omax) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..a.len().min(b.len()) {
                let (u, v) = (a[i] as f64, b[i] as f64);
                dot += u * v; na += u * u; nb += v * v;
                let d = (u - v).abs(); if d > mx { mx = d; } if u.abs() > omax { omax = u.abs(); }
            }
            (if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 }, if omax > 0.0 { mx / omax } else { mx })
        };
        println!("\n=== GDN proj verify (qkv GEMM M=N vs decode GEMV ×N), recurrent layer {layer}, cc={cc} ===");
        let mut all_ok = true;
        for &n in &[2usize, 4, 8] {
            let qkv_h = self.ssm_qkv_buf.as_ref().unwrap().handle;
            // Batch: one recurrent-layer batched run → ssm_qkv_buf [N × cc].
            let _ = self.forward_layer_batch_debug(dev, registry, cmd_ctx, model, layer, n as u32, &x[..n * hidden])?;
            let qkv_n = self.stage_buf_region(dev, cmd_ctx, qkv_h, 0, (n * cc) as u32)?;
            // Decode: per-token GEMV → ssm_qkv_buf [cc] (qkv is state-independent).
            let mut worst_c = 1f64; let mut worst_r = 0f64; let mut worst_t = 0;
            for t in 0..n {
                self.kv_cache.reset();
                let _ = self.forward_layer_debug(dev, registry, cmd_ctx, model, layer, 0, &x[t * hidden..(t + 1) * hidden])?;
                let qkv_t = self.stage_buf_region(dev, cmd_ctx, qkv_h, 0, cc as u32)?;
                let (c, r) = cosf(&qkv_n[t * cc..(t + 1) * cc], &qkv_t);
                if c < worst_c { worst_c = c; worst_t = t; }
                if r > worst_r { worst_r = r; }
            }
            let ok = worst_c > 0.999 && worst_r < 0.02;
            all_ok &= ok;
            println!("  => proj N={n}: worst_cos={worst_c:.6} (pos {worst_t}) worst_rel={worst_r:.3e} -> {}",
                if ok { "MATCH (@~1e-5 GEMM baseline)" } else { "MISMATCH (>baseline = wiring/sizing)" });
        }
        println!("=== GDN PROJ VERDICT: {} ===",
            if all_ok { "MATCH — batched qkv projection (GEMM M=N) correct @baseline" } else { "MISMATCH — projection wiring/sizing bug" });
        Ok(all_ok)
    }

    /// GDN-Completion #4b Step 1 FLOOR (gated `VF_QWEN35_GDN_VERIFY`): verify
    /// the SERIAL state-carry cores' per-token loop (offsets + state-carry)
    /// BIT-IDENTICALLY against the genuine decode single-token path. The
    /// batched loop reuses the identical single-token body (no GEMM reorder)
    /// → rel must be 0.000e0, the sharpest carry/offset check. Drives the
    /// same run_ssm_conv_setup/run_ssm_conv/run_gated_delta_net primitives the
    /// b_step bodies use: batched reads token-t slots via offsets, decode
    /// copies token t into a 1-token slot (offset 0). Mismatch pattern:
    /// pos0-ok/from-pos1 = state-carry bug; wrong-position = offset bug.
    pub fn gdn_serial_verify(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        allocator: &mut Allocator,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let spec = self.config.qwen35.clone().expect("gdn_serial_verify: not qwen35");
        let cc = spec.conv_channels() as usize;     // 10240
        let nc = spec.ssm_d_conv as usize;          // 4
        let pad = nc - 1;
        let h = spec.num_v_heads() as usize;         // 48
        let s_v = spec.ssm_d_state as usize;         // 128
        let head_k = spec.head_k_dim() as usize;     // 128
        let n_max = 16usize;
        let scale = 1.0f32 / (s_v as f32).sqrt();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut mk = |a: &mut Allocator, floats: usize, name: &str| {
            GpuBuffer::new(&dev.device, a, (floats as u64) * 4, usage, MemoryLocation::CpuToGpu, name)
        };
        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx, mut omax) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..a.len().min(b.len()) {
                let (u, v) = (a[i] as f64, b[i] as f64);
                dot += u * v; na += u * u; nb += v * v;
                let d = (u - v).abs(); if d > mx { mx = d; } if u.abs() > omax { omax = u.abs(); }
            }
            (if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 }, if omax > 0.0 { mx / omax } else { mx })
        };
        let hbar = |dev: &VulkanDevice, cmd: vk::CommandBuffer| {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe { dev.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), std::slice::from_ref(&mb), &[], &[]); }
        };
        let compbar = |dev: &VulkanDevice, cmd: vk::CommandBuffer| {
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            unsafe { dev.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), std::slice::from_ref(&mb), &[], &[]); }
        };

        // ── shared synthetic buffers (host-visible) ──
        let qpt = head_k * h;                         // qrep/krep per token = 6144
        let opt = s_v * h;                            // gdn output per token = 6144
        let state_f = h * s_v * s_v;                  // 786432
        let mut qkv_n  = mk(allocator, n_max * cc, "sv_qkvN")?;
        let mut conv_w = mk(allocator, cc * nc, "sv_convw")?;
        let mut convst = mk(allocator, cc * pad, "sv_convst")?;
        let conv_in = mk(allocator, cc * nc, "sv_convin")?;
        let conv_oN = mk(allocator, n_max * cc, "sv_convoN")?;
        let conv_o1 = mk(allocator, cc, "sv_convo1")?;
        let mut qN = mk(allocator, n_max * qpt, "sv_qN")?;
        let mut kN = mk(allocator, n_max * qpt, "sv_kN")?;
        let mut vN = mk(allocator, n_max * cc, "sv_vN")?;     // V-slice at +4096 per token
        let mut gN = mk(allocator, n_max * h, "sv_gN")?;
        let mut bN = mk(allocator, n_max * h, "sv_bN")?;
        let mut ssmst = mk(allocator, state_f, "sv_ssmst")?;
        let gdnN = mk(allocator, n_max * opt + state_f, "sv_gdnN")?;
        let gdn1 = mk(allocator, opt + state_f, "sv_gdn1")?;
        let mut q1 = mk(allocator, qpt, "sv_q1")?; let mut k1 = mk(allocator, qpt, "sv_k1")?;
        let mut v1 = mk(allocator, cc, "sv_v1")?; let mut g1 = mk(allocator, h, "sv_g1")?; let mut b1 = mk(allocator, h, "sv_b1")?;
        let v_slice_off = (head_k * spec.num_k_heads() as usize * 2) as u64 * 4; // 4096*4

        // synthetic fills (bounded; g≤0 → exp∈(0,1])
        let f1 = |x: usize| 0.12 * (0.013 * x as f32).sin() + 0.04 * (0.0007 * x as f32).cos();
        let mut convw = vec![0f32; cc * nc]; for i in 0..cc*nc { convw[i] = 0.2 * (0.7 * (i % nc) as f32 + 0.001 * i as f32).cos(); }
        conv_w.write_bytes(bytemuck::cast_slice(&convw))?;
        let zeros_cs = vec![0f32; cc * pad]; let zeros_ss = vec![0f32; state_f];

        println!("\n=== GDN serial-core verify (per-token loop vs decode single-token, BIT-IDENT) ===");
        let mut all_ok = true;
        for &n in &[2usize, 4, 8, 16] {
            // fill qkv_n + GDN inputs (global token t)
            let mut qkvv = vec![0f32; n*cc]; for t in 0..n { for c in 0..cc { qkvv[t*cc+c] = f1(t*131 + c); } }
            qkv_n.write_bytes(bytemuck::cast_slice(&qkvv))?;
            let mut qv=vec![0f32;n*qpt]; let mut kv=vec![0f32;n*qpt]; let mut vv=vec![0f32;n*cc]; let mut gv=vec![0f32;n*h]; let mut bv=vec![0f32;n*h];
            for t in 0..n {
                for i in 0..qpt { qv[t*qpt+i]=f1(t*17+i); kv[t*qpt+i]=f1(t*19+i+3); }
                for c in 0..(s_v*h) { vv[t*cc + (v_slice_off as usize/4) + c] = f1(t*23+c+1); }
                for j in 0..h { gv[t*h+j]= -0.10 - 0.02*(((t+j)%5) as f32); bv[t*h+j]=0.5+0.1*((0.4*t as f32+0.2*j as f32).sin()); }
            }
            qN.write_bytes(bytemuck::cast_slice(&qv))?; kN.write_bytes(bytemuck::cast_slice(&kv))?;
            vN.write_bytes(bytemuck::cast_slice(&vv))?; gN.write_bytes(bytemuck::cast_slice(&gv))?; bN.write_bytes(bytemuck::cast_slice(&bv))?;

            // ── CONV: batched per-token loop (mirrors b_step_ssm_conv1d) ──
            convst.write_bytes(bytemuck::cast_slice(&zeros_cs))?;
            let (qkvh, cwh, csh, cih, coNh) = (qkv_n.handle, conv_w.handle, convst.handle, conv_in.handle, conv_oN.handle);
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                for t in 0..n as u64 {
                    let off = t * (cc as u64) * 4;
                    self.run_ssm_conv_setup(dev, registry, cmd, csh, 0, (cc*pad) as u64*4, qkvh, off, cih, cc as u32, "sv_setup");
                    compbar(dev, cmd);
                    self.run_ssm_conv(dev, registry, cmd, cih, cwh, coNh, off, nc as u32, nc as u32, cc as u32, 1, 1, "sv_conv");
                    compbar(dev, cmd);
                }
                hbar(dev, cmd);
            })?;
            let conv_a = bytemuck::cast_slice::<u8,f32>(&conv_oN.read_bytes()?[..(n*cc)*4]).to_vec();
            // ── CONV: decode single-token (offset 0, token copied to slot 0) ──
            convst.write_bytes(bytemuck::cast_slice(&zeros_cs))?;
            let mut conv_b = vec![0f32; n*cc];
            let (q1kvh, co1h) = (qkv_n.handle, conv_o1.handle); // reuse qkv_n offset via setup qkv_off
            for t in 0..n as u64 {
                cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                    self.run_ssm_conv_setup(dev, registry, cmd, csh, 0, (cc*pad) as u64*4, q1kvh, t*(cc as u64)*4, cih, cc as u32, "sv_setupd");
                    compbar(dev, cmd);
                    self.run_ssm_conv(dev, registry, cmd, cih, cwh, co1h, 0, nc as u32, nc as u32, cc as u32, 1, 1, "sv_convd");
                    hbar(dev, cmd);
                })?;
                conv_b[(t as usize)*cc..(t as usize+1)*cc].copy_from_slice(bytemuck::cast_slice::<u8,f32>(&conv_o1.read_bytes()?[..cc*4]));
            }
            let (mut wc, mut wr, mut wt) = (1f64, 0f64, 0usize);
            for t in 0..n { let (c,r)=cosf(&conv_a[t*cc..(t+1)*cc], &conv_b[t*cc..(t+1)*cc]); if c<wc {wc=c; wt=t;} if r>wr {wr=r;} }
            let conv_ok = wr == 0.0;
            println!("  conv N={n}: worst_cos={wc:.6} worst_rel={wr:.3e} (pos {wt}) -> {}", if conv_ok {"BIT-IDENT"} else {"MISMATCH"});

            // ── GDN: batched per-token loop (mirrors b_step_gated_delta_net) ──
            ssmst.write_bytes(bytemuck::cast_slice(&zeros_ss))?;
            let (qh,kh,vh,gh,bh,sh,dNh) = (qN.handle,kN.handle,vN.handle,gN.handle,bN.handle,ssmst.handle,gdnN.handle);
            let st_region = (n as u64) * (opt as u64) * 4;
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                for t in 0..n as u64 {
                    let push = GatedDeltaNetPushConstants {
                        h: h as u32, n_tokens: 1, n_seqs: 1, s_off: (n as u32 - t as u32)*(s_v as u32)*(h as u32),
                        sq1: head_k as u32, sq2:(head_k*h) as u32, sq3:(head_k*h) as u32,
                        sv1: s_v as u32, sv2:(s_v*h) as u32, sv3:(s_v*h) as u32, sb1:1, sb2:h as u32, sb3:h as u32,
                        neq1: h as u32, rq3:1, scale, k:1,
                    };
                    self.run_gated_delta_net(dev, registry, cmd,
                        qh, t*(qpt as u64)*4, kh, t*(qpt as u64)*4,
                        vh, t*(cc as u64)*4 + v_slice_off, (opt as u64)*4,
                        gh, t*(h as u64)*4, bh, t*(h as u64)*4,
                        sh, 0, (state_f as u64)*4, dNh, t*(opt as u64)*4, s_v as u32, 2, &push, "sv_gdn");
                    compbar(dev, cmd);
                    let region = vk::BufferCopy { src_offset: st_region, dst_offset: 0, size: (state_f as u64)*4 };
                    unsafe { dev.device.cmd_copy_buffer(cmd, dNh, sh, std::slice::from_ref(&region)); }
                    compbar(dev, cmd);
                }
                hbar(dev, cmd);
            })?;
            let gdn_a = bytemuck::cast_slice::<u8,f32>(&gdnN.read_bytes()?[..(n*opt)*4]).to_vec();
            // ── GDN: decode single-token (offset 0, ssm_state-carry) ──
            ssmst.write_bytes(bytemuck::cast_slice(&zeros_ss))?;
            let mut gdn_b = vec![0f32; n*opt];
            let d1h = gdn1.handle;
            for t in 0..n as u64 {
                // copy token t inputs to 1-tok slots
                q1.write_bytes(bytemuck::cast_slice(&qv[(t as usize)*qpt..(t as usize+1)*qpt]))?;
                k1.write_bytes(bytemuck::cast_slice(&kv[(t as usize)*qpt..(t as usize+1)*qpt]))?;
                v1.write_bytes(bytemuck::cast_slice(&vv[(t as usize)*cc..(t as usize+1)*cc]))?;
                g1.write_bytes(bytemuck::cast_slice(&gv[(t as usize)*h..(t as usize+1)*h]))?;
                b1.write_bytes(bytemuck::cast_slice(&bv[(t as usize)*h..(t as usize+1)*h]))?;
                let push = GatedDeltaNetPushConstants {
                    h: h as u32, n_tokens: 1, n_seqs: 1, s_off:(s_v*h) as u32,
                    sq1: head_k as u32, sq2:(head_k*h) as u32, sq3:(head_k*h) as u32,
                    sv1: s_v as u32, sv2:(s_v*h) as u32, sv3:(s_v*h) as u32, sb1:1, sb2:h as u32, sb3:h as u32,
                    neq1: h as u32, rq3:1, scale, k:1,
                };
                let (q1h,k1h,v1h,g1h,b1h)=(q1.handle,k1.handle,v1.handle,g1.handle,b1.handle);
                cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                    self.run_gated_delta_net(dev, registry, cmd,
                        q1h, 0, k1h, 0, v1h, v_slice_off, (opt as u64)*4,
                        g1h, 0, b1h, 0, sh, 0, (state_f as u64)*4, d1h, 0, s_v as u32, 2, &push, "sv_gdnd");
                    compbar(dev, cmd);
                    let region = vk::BufferCopy { src_offset: (opt as u64)*4, dst_offset: 0, size: (state_f as u64)*4 };
                    unsafe { dev.device.cmd_copy_buffer(cmd, d1h, sh, std::slice::from_ref(&region)); }
                    hbar(dev, cmd);
                })?;
                gdn_b[(t as usize)*opt..(t as usize+1)*opt].copy_from_slice(bytemuck::cast_slice::<u8,f32>(&gdn1.read_bytes()?[..opt*4]));
            }
            let (mut wc2, mut wr2, mut wt2) = (1f64, 0f64, 0usize);
            for t in 0..n { let (c,r)=cosf(&gdn_a[t*opt..(t+1)*opt], &gdn_b[t*opt..(t+1)*opt]); if c<wc2 {wc2=c; wt2=t;} if r>wr2 {wr2=r;} }
            let gdn_ok = wr2 == 0.0;
            println!("  gdn  N={n}: worst_cos={wc2:.6} worst_rel={wr2:.3e} (pos {wt2}) -> {}", if gdn_ok {"BIT-IDENT"} else {"MISMATCH"});
            all_ok &= conv_ok && gdn_ok;
        }
        println!("=== GDN SERIAL-CORE VERDICT: {} ===",
            if all_ok { "BIT-IDENT — serial conv + GDN per-token loops (offset+state-carry) correct" }
            else { "MISMATCH — carry (pos0-ok/from-pos1) or offset (wrong-pos) bug" });
        for buf in [qkv_n, conv_w, convst, conv_in, conv_oN, conv_o1, qN, kN, vN, gN, bN, ssmst, gdnN, gdn1, q1, k1, v1, g1, b1] {
            buf.destroy(&dev.device, allocator);
        }
        Ok(all_ok)
    }

    /// GDN-Completion #5 (gated `VF_QWEN35_GDN_VERIFY`) — FULL-LAYER
    /// integration: the whole batched GDN (Linear-Attn) layer vs the
    /// per-token decode path, against the deterministic oracle. Resets the
    /// recurrent state (conv_state/ssm_state via the `ssm_persistent_
    /// initialized` flag) before each run so both start from scratch; the
    /// decode oracle carries state across its N single-token calls. The
    /// pipeline now contains the projection/out-proj GEMMs → ~1e-5 FP-reorder
    /// baseline (cos→1.0, not bit-identical). Mismatch ≫1e-5 localizes to a
    /// mechanical op or an inter-mechanical-stage handoff (cores + projections
    /// + serial loops already verified).
    pub fn gdn_layer_verify(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let spec = cfg.qwen35.as_ref().expect("gdn_layer_verify: not qwen35");
        let hidden = cfg.hidden_dim as usize;
        let layer = (0..spec.block_count).find(|&l| spec.is_recurrent_layer(l)).expect("no recurrent layer");
        let n_max = 32usize;
        let mut x = vec![0f32; n_max * hidden];
        for p in 0..n_max { for i in 0..hidden {
            x[p * hidden + i] = 0.18 * (0.29 * p as f32 + 0.015 * i as f32).sin() + 0.06 * (0.004 * i as f32).cos() - 0.10;
        }}
        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx, mut omax) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..a.len().min(b.len()) {
                let (u, v) = (a[i] as f64, b[i] as f64);
                dot += u * v; na += u * u; nb += v * v;
                let d = (u - v).abs(); if d > mx { mx = d; } if u.abs() > omax { omax = u.abs(); }
            }
            (if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 0.0 }, if omax > 0.0 { mx / omax } else { mx })
        };
        println!("\n=== GDN full-layer integration (batched recurrent layer {layer} vs decode per-token) ===");
        let mut all_ok = true;
        for &n in &[2usize, 4, 8, 16, 32] {
            // Batched: reset recurrent state, run the whole batched layer.
            self.ssm_persistent_initialized = false;
            let (batched, _k, _v) = self.forward_layer_batch_debug(dev, registry, cmd_ctx, model, layer, n as u32, &x[..n * hidden])?;
            // Decode oracle: reset state, per-token with state-carry.
            self.ssm_persistent_initialized = false;
            self.kv_cache.reset();
            let mut oracle = vec![0f32; n * hidden];
            for t in 0..n {
                let y = self.forward_layer_debug(dev, registry, cmd_ctx, model, layer, t as u32, &x[t * hidden..(t + 1) * hidden])?;
                oracle[t * hidden..(t + 1) * hidden].copy_from_slice(&y);
            }
            let (mut wc, mut wr, mut wt) = (1f64, 0f64, 0usize);
            for t in 0..n {
                let (c, r) = cosf(&oracle[t * hidden..(t + 1) * hidden], &batched[t * hidden..(t + 1) * hidden]);
                if c < wc { wc = c; wt = t; }
                if r > wr { wr = r; }
                if n <= 4 || t == 0 || t == 1 || t == n - 1 { println!("  layer N={n} pos {t:2}: cos={c:.6} rel={r:.3e}"); }
            }
            let ok = wc > 0.999 && wr < 0.02;
            all_ok &= ok;
            println!("  => layer N={n}: worst_cos={wc:.6} (pos {wt}) worst_rel={wr:.3e} -> {}",
                if ok { "MATCH @~1e-5" } else { "MISMATCH" });
        }
        // Restore clean state for any subsequent use.
        self.ssm_persistent_initialized = false;
        self.kv_cache.reset();
        println!("=== GDN FULL-LAYER VERDICT: {} ===",
            if all_ok { "MATCH @~1e-5 — batched GDN layer correct → COMPLETE" } else { "MISMATCH — localize op/handoff" });
        Ok(all_ok)
    }

    /// De-risk substep oracle — run ONE real DECODE layer (`dispatch_layer`,
    /// the same path `forward_token` uses) at `position` and return its
    /// post-layer `(k_buf, v_buf)` rows (each `kv_dim`). At N=1/pos 0 these
    /// are the post-k-norm-post-rope k and the v-proj v. KV-cache state is
    /// the caller's responsibility (reset + in-order for multi-token).
    pub fn forward_layer_debug_kv(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let kv_dim = n_kv_heads_for(&cfg, layer) * cfg.head_dim;
        self.cur_mut().scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.cur_mut().rope_pos_buf.write_bytes(bytemuck::bytes_of(&position))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.cur().scratch_a.handle, self.cur().scratch_b.handle,
            );
        })?;
        let k = self.stage_buf_prefix(dev, cmd_ctx, self.cur().k_buf.handle, kv_dim)?;
        let v = self.stage_buf_prefix(dev, cmd_ctx, self.cur().v_buf.handle, kv_dim)?;
        Ok((k, v))
    }

    /// De-risk helper (gated diagnostic) — run ONE BatchExec layer on a
    /// synthetic `[seq_len × hidden]` input and return the post-layer
    /// residual stream `[seq_len × hidden]`. Used to validate the batched
    /// NON-GDN Full-Attention path in isolation against the per-token
    /// `forward_layer_debug` oracle (qwen35 G-2i residual de-risk).
    ///
    /// Seeds `batch_norm` with THIS layer's `attn_norm.weight` (the
    /// cross-layer-fusion contract `dispatch_layer_batch` expects on
    /// entry — `b_step_attn_norm` is a no-op), resets the KV cache, runs
    /// at `base_pos=0` with `next_attn_norm_weight=None` (last-layer plain
    /// residual add, no fusion to a next layer). Reads `batch_residual`
    /// back one position at a time through the host-visible `scratch_a`.
    /// Returns `(layer_out [seq_len×hidden], k_row0 [kv_dim], v_row0 [kv_dim])`.
    /// `k_row0`/`v_row0` are the first kv_dim floats of `batch_k`/`batch_v`
    /// after the layer; at N=1 that is unambiguously position 0's
    /// post-k-norm-rope k and v-proj v (no batch-layout stride to reason
    /// about), used for the substep diff vs `forward_layer_debug_kv`.
    pub fn forward_layer_batch_debug(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        seq_len: u32,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        let hidden_bytes = (hidden as u64) * 4;
        assert_eq!(
            input.len() as u32,
            seq_len * hidden,
            "forward_layer_batch_debug: input must be seq_len*hidden",
        );
        let positions: Vec<u32> = (0..seq_len).collect();
        self.cur_mut()
            .rope_pos_buf
            .write_bytes(bytemuck::cast_slice(&positions))?;
        self.batch_input.write_bytes(bytemuck::cast_slice(input))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        self.kv_cache.reset();

        let total_bytes = (seq_len as u64) * hidden_bytes;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            // Seed: batch_residual = batch_input.
            let copy = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(total_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd,
                    self.batch_input.handle,
                    self.batch_residual.handle,
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
                    std::slice::from_ref(&bar),
                    &[],
                    &[],
                );
            }
            // Seed batch_norm = rms_norm(batch_residual) * layer.attn_norm.weight
            // (the contract dispatch_layer_batch expects on entry).
            let w = layer_weight(model, layer, "attn_norm.weight");
            self.run_rms_norm(
                dev, registry, cmd,
                self.batch_residual.handle, w, self.batch_norm.handle,
                hidden, seq_len, cfg.rms_norm_eps, "rms_norm_batch_debug_seed",
            );
            compute_barrier(dev, cmd);
            // Run the batched layer (no fusion into a next layer).
            self.dispatch_layer_batch(
                dev, registry, cmd, model, layer, seq_len, 0, None,
            );
        })?;

        // Substep capture: batch_k / batch_v rows (kv_dim, fit scratch_a).
        // Captured BEFORE the layer-output loop (both reuse scratch_a).
        let kv_dim = n_kv_heads_for(&cfg, layer) * cfg.head_dim;
        let k_row0 = self.stage_buf_prefix(dev, cmd_ctx, self.batch_k.handle, kv_dim)?;
        let v_row0 = self.stage_buf_prefix(dev, cmd_ctx, self.batch_v.handle, kv_dim)?;

        // Read batch_residual position-by-position through scratch_a.
        let mut out = vec![0f32; (seq_len * hidden) as usize];
        for p in 0..seq_len {
            let off = (p as u64) * hidden_bytes;
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
                let pre = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.batch_residual.handle)
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
                }
                self.copy_batch_row(
                    dev, cmd, self.batch_residual.handle, off,
                    self.cur().scratch_a.handle, hidden_bytes,
                );
                let post = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.cur().scratch_a.handle)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                unsafe {
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
            })?;
            let bytes = self.cur().scratch_a.read_bytes()?;
            let row = bytemuck::cast_slice::<u8, f32>(&bytes[..(hidden as usize) * 4]);
            out[(p * hidden) as usize..((p + 1) * hidden) as usize]
                .copy_from_slice(row);
        }
        Ok((out, k_row0, v_row0))
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

/// Sprint D.0 — `VF_DISPATCH_LOG=1` prints the dispatch grid for each
/// distinct (shader-label, grid) pair exactly once. Joined with the
/// `VF_GPU_TIMER` per-shader breakdown (label → ms/tok) this gives the
/// shader → grid → duration mapping used to diagnose under-occupied
/// dispatches (grid ≪ 64 CUs). Pure diagnostic, default OFF, no effect
/// on execution. Called from the dispatch helpers in `runs.rs`.
pub(super) fn log_dispatch(label: &str, gx: u32, gy: u32, gz: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        std::env::var("VF_DISPATCH_LOG").map(|v| v == "1").unwrap_or(false)
    }) {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let key = format!("{label}|{gx}x{gy}x{gz}");
    if seen.lock().unwrap().insert(key) {
        let total_wg = gx as u64 * gy as u64 * gz as u64;
        eprintln!("VF_DISPATCH  {label:<28} grid=({gx},{gy},{gz})  total_wg={total_wg}");
    }
}
