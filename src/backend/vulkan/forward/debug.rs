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
                self.run_gated_delta_net(dev, registry, cmd, qh, kh, vh, 0, v_bytes_a, gh, bh, sh, 0, st_bytes, dh, s_v, 2, &push_a, "gdnv_a");
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
                    self.run_gated_delta_net(dev, registry, cmd, q1h, k1h, v1h, 0, (v_per as u64) * 4, g1h, b1h, sh2, 0, st_bytes, d1h, s_v, 2, &push_b, "gdnv_b");
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
