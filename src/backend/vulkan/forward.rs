//! Forward-pass orchestration for Qwen3 decode.
//!
//! Phase 2C. One [`Forward`] instance owns:
//! - per-token scratch buffers (ping-pong + per-projection slots),
//! - the K/V cache,
//! - one long-lived descriptor pool, reset between forwards,
//! - tiny RoPE auxiliary buffers (pos / ff / indices),
//! - an optional [`ShaderProfiler`].
//!
//! [`Forward::forward_token`] dispatches the embedding lookup → 36
//! transformer layers → final RMSNorm → LM head and reads the logits
//! back. Each shader path gets a method on `Forward` that allocates
//! a descriptor set from the pool, writes it, and dispatches.
//!
//! Layer ordering (Qwen3 with QK-norm):
//! ```text
//! input ─→ attn_norm ─→ Wq/Wk/Wv (3× GEMV)
//!         q ─→ q_norm ─→ RoPE-NeoX
//!         k ─→ k_norm ─→ RoPE-NeoX  ─→ KV cache (pos-major copy)
//!         v ────────────────────────→ KV cache
//!         attention (scalar_attn) ──→ Wo ─→ residual1
//!         ffn_norm ─→ gate, up (2× GEMV) ─→ silu(gate)·up ─→ Wdown
//!         residual1 + Wdown_out ──→ next-layer input
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::buffers::GpuBuffer;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::gguf::{GgmlType, ModelConfig};
use super::kv_cache::KvCache;
use super::loader::LoadedModel;
use super::pipeline::{
    ComputeKernel, GenericBinaryPushConstants, GenericHeadPushConstants,
    MatVecPushConstants, RopePushConstants, ScalarAttnPushConstants,
};
use super::pipeline_registry::PipelineRegistry;
use super::profiler::{ShaderProfiler, TimingSample};
use super::shaders::ShaderId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugTarget {
    AttnNorm,
    QProj,
    KProj,
    VProj,
    QNormRope,
    KNormRope,
    AttnOut,
}

pub struct ForwardStats {
    pub total: Duration,
    pub per_shader: BTreeMap<String, (Duration, u32)>,
    pub per_layer: Vec<Duration>,
    pub samples: Vec<TimingSample>,
}

pub struct Forward {
    // Scratch (per-token reuse).
    scratch_a: GpuBuffer,
    scratch_b: GpuBuffer,
    hidden_norm: GpuBuffer,
    q_buf: GpuBuffer,
    k_buf: GpuBuffer,
    v_buf: GpuBuffer,
    attn_out: GpuBuffer,
    o_buf: GpuBuffer,
    res1: GpuBuffer,
    gate_buf: GpuBuffer,
    up_buf: GpuBuffer,
    ffn_hidden: GpuBuffer,
    ffn_out: GpuBuffer,
    logits_buf: GpuBuffer,
    // Always-bound dummies for unused descriptor slots.
    fuse0: GpuBuffer,
    fuse1: GpuBuffer,
    rope_pos_buf: GpuBuffer,    // 4 B host-visible: writes the current position
    rope_ff_buf: GpuBuffer,     // 16 B unused (has_ff=0)
    rope_idx_buf: GpuBuffer,    // 16 B unused (set_rows_stride=0)

    pub kv_cache: KvCache,
    pub config: ModelConfig,

    descriptor_pool: vk::DescriptorPool,
    pub profiler: Option<ShaderProfiler>,

    rope_theta_scale: f32,
    attn_scale: f32,
}

impl Forward {
    pub fn new(
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        kv_cache: KvCache,
        config: ModelConfig,
        profiler: Option<ShaderProfiler>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = &dev.device;
        let mut mk_storage = |size: u64, location: MemoryLocation, name: &str| {
            GpuBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                location,
                name,
            )
        };

        let hidden_bytes = (config.hidden_dim as u64) * 4;
        let q_bytes = (config.n_heads as u64) * (config.head_dim as u64) * 4;
        let kv_bytes = (config.n_kv_heads as u64) * (config.head_dim as u64) * 4;
        let ffn_bytes = (config.ffn_dim as u64) * 4;
        let logits_bytes = (config.vocab_size as u64) * 4;

        let scratch_a = mk_storage(hidden_bytes, MemoryLocation::CpuToGpu, "scratch_a")?;
        let scratch_b = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "scratch_b")?;
        let hidden_norm = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "hidden_norm")?;
        let q_buf = mk_storage(q_bytes, MemoryLocation::GpuOnly, "q_buf")?;
        let k_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, "k_buf")?;
        let v_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, "v_buf")?;
        let attn_out = mk_storage(q_bytes, MemoryLocation::GpuOnly, "attn_out")?;
        let o_buf = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "o_buf")?;
        let res1 = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "res1")?;
        let gate_buf = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "gate_buf")?;
        let up_buf = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "up_buf")?;
        let ffn_hidden = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "ffn_hidden")?;
        let ffn_out = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "ffn_out")?;
        let logits_buf = mk_storage(logits_bytes, MemoryLocation::GpuToCpu, "logits_buf")?;
        let fuse0 = mk_storage(16, MemoryLocation::GpuOnly, "fuse0_dummy")?;
        let fuse1 = mk_storage(16, MemoryLocation::GpuOnly, "fuse1_dummy")?;
        let rope_pos_buf = mk_storage(4, MemoryLocation::CpuToGpu, "rope_pos")?;
        let rope_ff_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_ff_dummy")?;
        let rope_idx_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_idx_dummy")?;

        // Descriptor pool sized for one full forward: 18 dispatches/layer
        // × n_layers + a handful of extras (final_norm + lm_head + a
        // safety margin so we never spuriously hit OUT_OF_POOL_MEMORY).
        // Up to 5 storage descriptors per dispatch.
        let dispatches = 25 * config.n_layers + 32;
        let max_descriptors = dispatches * 5;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_descriptors,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(dispatches)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let attn_scale = 1.0_f32 / (config.head_dim as f32).sqrt();
        let rope_theta_scale =
            (1.0_f32 / config.rope_freq_base).powf(2.0 / config.head_dim as f32);

        Ok(Self {
            scratch_a, scratch_b, hidden_norm,
            q_buf, k_buf, v_buf, attn_out, o_buf, res1,
            gate_buf, up_buf, ffn_hidden, ffn_out,
            logits_buf,
            fuse0, fuse1,
            rope_pos_buf, rope_ff_buf, rope_idx_buf,
            kv_cache, config,
            descriptor_pool,
            profiler,
            rope_theta_scale, attn_scale,
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
    ) -> Result<ForwardStats, Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(),
                self.config.hidden_dim
            )
            .into());
        }
        self.scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        // Pre-write the RoPE position buffer.
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;

        // Reset descriptor pool for fresh allocations this forward.
        unsafe {
            dev.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;
        }

        // Pre-snapshot: we'll record per-layer profile boundaries.
        let mut per_layer_starts: Vec<usize> = Vec::with_capacity(self.config.n_layers as usize);

        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            if let Some(p) = self.profiler.as_mut() {
                p.reset(&dev.device, cmd);
            }

            let mut input = self.scratch_a.handle;
            let mut output = self.scratch_b.handle;

            for layer in 0..self.config.n_layers {
                if let Some(p) = self.profiler.as_ref() {
                    per_layer_starts.push(p.entries_len());
                }
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                std::mem::swap(&mut input, &mut output);
            }

            // After last layer, `input` holds the activation we feed
            // into final-norm + LM head.
            self.dispatch_final(dev, registry, cmd, model, input);

            // Make logits visible to the host.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.logits_buf.handle)
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

        // Logits readback.
        let bytes = self.logits_buf.read_bytes()?;
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
        self.scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        unsafe {
            dev.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer_partial(
                dev, registry, cmd, model, layer, position,
                self.scratch_a.handle, target,
            );
        })?;

        // Pick the relevant buffer + size for readback.
        let cfg = self.config.clone();
        let (src_buf, count) = match target {
            DebugTarget::AttnNorm => (self.hidden_norm.handle, cfg.hidden_dim as u64),
            DebugTarget::QProj | DebugTarget::QNormRope => (
                self.q_buf.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::KProj | DebugTarget::KNormRope => (
                self.k_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::VProj => (
                self.v_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::AttnOut => (
                self.attn_out.handle,
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
            dev.device.cmd_copy_buffer(cmd, src_buf, self.scratch_a.handle,
                std::slice::from_ref(&copy));
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_a.handle)
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
        let bytes = self.scratch_a.read_bytes()?;
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
        self.run_rms_norm(dev, registry, cmd, input, w, self.hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        if halt == DebugTarget::AttnNorm { return; }
        compute_barrier(dev, cmd);

        // Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight");
        let sk = layer_weight_shader(model, layer, "attn_k.weight");
        let sv = layer_weight_shader(model, layer, "attn_v.weight");
        self.run_gemv(dev, registry, cmd, sq,
                      wq, self.hidden_norm.handle, self.q_buf.handle,
                      cfg.hidden_dim, cfg.n_heads * cfg.head_dim, "gemv_q");
        self.run_gemv(dev, registry, cmd, sk,
                      wk, self.hidden_norm.handle, self.k_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_k");
        self.run_gemv(dev, registry, cmd, sv,
                      wv, self.hidden_norm.handle, self.v_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_v");
        if matches!(halt, DebugTarget::QProj | DebugTarget::KProj | DebugTarget::VProj) {
            return;
        }
        compute_barrier(dev, cmd);

        // Q/K norm
        let wqn = layer_weight(model, layer, "attn_q_norm.weight");
        let wkn = layer_weight(model, layer, "attn_k_norm.weight");
        self.run_rms_norm(dev, registry, cmd,
                         self.q_buf.handle, wqn, self.q_buf.handle,
                         cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
        self.run_rms_norm(dev, registry, cmd,
                         self.k_buf.handle, wkn, self.k_buf.handle,
                         cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k");
        compute_barrier(dev, cmd);

        // RoPE
        self.run_rope_neox(dev, registry, cmd, self.q_buf.handle, self.q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.k_buf.handle, self.k_buf.handle,
                           cfg.head_dim, cfg.n_kv_heads, position, "rope_k");
        if matches!(halt, DebugTarget::QNormRope | DebugTarget::KNormRope) {
            return;
        }
        compute_barrier(dev, cmd);

        // KV write
        let row_bytes = self.kv_cache.row_bytes();
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let copy = vk::BufferCopy::default()
            .src_offset(0).dst_offset(dst_off).size(row_bytes);
        let k_src = self.k_buf.handle;
        let v_src = self.v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        unsafe {
            dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
            dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
        }
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
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
        self.scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        unsafe {
            dev.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.scratch_a.handle, self.scratch_b.handle,
            );
            // Readback barrier so the next host map sees scratch_b.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_b.handle)
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
                .buffer(self.scratch_b.handle)
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
                self.scratch_b.handle,
                self.scratch_a.handle,
                std::slice::from_ref(&copy),
            );
            // Post-barrier: TRANSFER_WRITE → HOST_READ
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_a.handle)
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
        let bytes_slice = self.scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes_slice[..(self.config.hidden_dim as usize) * 4],
        )
        .to_vec())
    }

    /// Read the most recently written logits — call after
    /// `forward_token` returns successfully.
    pub fn logits(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = self.logits_buf.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        self.scratch_a.destroy(device, allocator);
        self.scratch_b.destroy(device, allocator);
        self.hidden_norm.destroy(device, allocator);
        self.q_buf.destroy(device, allocator);
        self.k_buf.destroy(device, allocator);
        self.v_buf.destroy(device, allocator);
        self.attn_out.destroy(device, allocator);
        self.o_buf.destroy(device, allocator);
        self.res1.destroy(device, allocator);
        self.gate_buf.destroy(device, allocator);
        self.up_buf.destroy(device, allocator);
        self.ffn_hidden.destroy(device, allocator);
        self.ffn_out.destroy(device, allocator);
        self.logits_buf.destroy(device, allocator);
        self.fuse0.destroy(device, allocator);
        self.fuse1.destroy(device, allocator);
        self.rope_pos_buf.destroy(device, allocator);
        self.rope_ff_buf.destroy(device, allocator);
        self.rope_idx_buf.destroy(device, allocator);
        self.kv_cache.destroy(device, allocator);
        if let Some(p) = self.profiler {
            p.destroy(device);
        }
    }

    // -------------------------------------------------------------
    // Per-layer + final + helpers.
    // -------------------------------------------------------------

    fn dispatch_layer(
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

        // (a) attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, self.hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        compute_barrier(dev, cmd);

        // (b) Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight");
        let sk = layer_weight_shader(model, layer, "attn_k.weight");
        let sv = layer_weight_shader(model, layer, "attn_v.weight");
        self.run_gemv(dev, registry, cmd, sq,
                      wq, self.hidden_norm.handle, self.q_buf.handle,
                      cfg.hidden_dim, cfg.n_heads * cfg.head_dim, "gemv_q");
        self.run_gemv(dev, registry, cmd, sk,
                      wk, self.hidden_norm.handle, self.k_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_k");
        self.run_gemv(dev, registry, cmd, sv,
                      wv, self.hidden_norm.handle, self.v_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_v");
        compute_barrier(dev, cmd);

        // (c) Q/K norm (per head)
        let wqn = layer_weight(model, layer, "attn_q_norm.weight");
        let wkn = layer_weight(model, layer, "attn_k_norm.weight");
        self.run_rms_norm(dev, registry, cmd,
                         self.q_buf.handle, wqn, self.q_buf.handle,
                         cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
        self.run_rms_norm(dev, registry, cmd,
                         self.k_buf.handle, wkn, self.k_buf.handle,
                         cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k");
        compute_barrier(dev, cmd);

        // (d) RoPE NeoX on Q and K
        self.run_rope_neox(dev, registry, cmd, self.q_buf.handle, self.q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.k_buf.handle, self.k_buf.handle,
                           cfg.head_dim, cfg.n_kv_heads, position, "rope_k");
        compute_barrier(dev, cmd);

        // (e) KV-cache write — pos-major.
        let row_bytes = self.kv_cache.row_bytes();
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let copy = vk::BufferCopy::default()
            .src_offset(0).dst_offset(dst_off).size(row_bytes);
        let k_src = self.k_buf.handle;
        let v_src = self.v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        self.profile("kv_write", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
            dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
        });
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

        // (f) Attention.
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        compute_barrier(dev, cmd);

        // (g) Output projection.
        let wo = layer_weight(model, layer, "attn_output.weight");
        let so = layer_weight_shader(model, layer, "attn_output.weight");
        self.run_gemv(dev, registry, cmd, so,
                      wo, self.attn_out.handle, self.o_buf.handle,
                      cfg.n_heads * cfg.head_dim, cfg.hidden_dim, "gemv_o");
        compute_barrier(dev, cmd);

        // (h) Residual1 = input + o_buf
        self.run_binary(dev, registry, cmd, ShaderId::Add,
                        input, self.o_buf.handle, self.res1.handle,
                        cfg.hidden_dim, "add_res1");
        compute_barrier(dev, cmd);

        // (i) ffn_norm
        let w = layer_weight(model, layer, "ffn_norm.weight");
        self.run_rms_norm(dev, registry, cmd,
                         self.res1.handle, w, self.hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_ffn");
        compute_barrier(dev, cmd);

        // (j) gate / up
        let wg = layer_weight(model, layer, "ffn_gate.weight");
        let wu = layer_weight(model, layer, "ffn_up.weight");
        let sg = layer_weight_shader(model, layer, "ffn_gate.weight");
        let su = layer_weight_shader(model, layer, "ffn_up.weight");
        self.run_gemv(dev, registry, cmd, sg,
                      wg, self.hidden_norm.handle, self.gate_buf.handle,
                      cfg.hidden_dim, cfg.ffn_dim, "gemv_gate");
        self.run_gemv(dev, registry, cmd, su,
                      wu, self.hidden_norm.handle, self.up_buf.handle,
                      cfg.hidden_dim, cfg.ffn_dim, "gemv_up");
        compute_barrier(dev, cmd);

        // (k) silu(gate) → gate_buf in place
        self.run_silu(dev, registry, cmd, self.gate_buf.handle, self.gate_buf.handle,
                      cfg.ffn_dim, "silu_gate");
        compute_barrier(dev, cmd);

        // (l) ffn_hidden = gate × up
        self.run_binary(dev, registry, cmd, ShaderId::Mul,
                        self.gate_buf.handle, self.up_buf.handle, self.ffn_hidden.handle,
                        cfg.ffn_dim, "mul_gate_up");
        compute_barrier(dev, cmd);

        // (m) FFN down — Q6_K in Q4_K_M, Q4_K otherwise.
        let wd = layer_weight(model, layer, "ffn_down.weight");
        let sd = layer_weight_shader(model, layer, "ffn_down.weight");
        self.run_gemv(dev, registry, cmd, sd,
                      wd, self.ffn_hidden.handle, self.ffn_out.handle,
                      cfg.ffn_dim, cfg.hidden_dim, "gemv_down");
        compute_barrier(dev, cmd);

        // (n) Residual2 = res1 + ffn_out → output
        self.run_binary(dev, registry, cmd, ShaderId::Add,
                        self.res1.handle, self.ffn_out.handle, output,
                        cfg.hidden_dim, "add_res2");
        compute_barrier(dev, cmd);
    }

    fn dispatch_final(
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
        // LM head: prefer dedicated `output.weight`; fall back to tied
        // `token_embd.weight` (Phase 2 doesn't tie weights, but be safe).
        let lm = model
            .tensor("output.weight")
            .or_else(|| model.tensor("token_embd.weight"))
            .expect("LM head present");
        let w_lm = lm.buffer.handle;
        let lm_shader = match lm.ggml_type {
            GgmlType::Q6K => ShaderId::MulMatVecQ6K,
            _ => ShaderId::MulMatVecQ4K,
        };

        self.run_rms_norm(
            dev, registry, cmd,
            input, w_norm, self.hidden_norm.handle,
            self.config.hidden_dim, 1, self.config.rms_norm_eps, "rms_norm_final",
        );
        compute_barrier(dev, cmd);
        self.run_gemv(
            dev, registry, cmd, lm_shader,
            w_lm, self.hidden_norm.handle, self.logits_buf.handle,
            self.config.hidden_dim, self.config.vocab_size, "lm_head",
        );
    }

    fn cpu_embedding_lookup(
        &self,
        _model: &LoadedModel,
        _token_id: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Phase-2C shortcut: small-magnitude deterministic input.
        // Real embedding read from token_embd.weight ships in
        // Phase 2D once we keep the GgufFile mmap alongside the
        // LoadedModel. For Phase 2C we only check "logits aren't
        // all-zero / NaN", which is dominated by weights.
        //
        // Magnitude tuned small (~0.02 RMS) so the chain
        //     embd → 36 layers w/ residuals → final norm → LM head
        // doesn't blow up to overflow / underflow on any single
        // RMSNorm step.
        let n = self.config.hidden_dim as usize;
        // DEBUG: temporarily zero input to isolate NaN source.
        let v = vec![0.0f32; n];
        Ok(v)
    }

    // -------------------------------------------------------------
    // Per-shader dispatch methods.
    // -------------------------------------------------------------

    fn alloc_set(&self, dev: &VulkanDevice, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        unsafe { dev.device.allocate_descriptor_sets(&info) }
            .expect("descriptor_set alloc")[0]
    }

    fn write_bindings(
        &self,
        dev: &VulkanDevice,
        set: vk::DescriptorSet,
        bindings: &[(u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
    ) {
        let infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|&(_, buf, off, range)| vk::DescriptorBufferInfo {
                buffer: buf,
                offset: off,
                range: if range == 0 { vk::WHOLE_SIZE } else { range },
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .enumerate()
            .map(|(i, &(b, _, _, _))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Wraps `f` in optional begin/end timestamp queries.
    fn profile<F>(&mut self, name: &str, dev: &VulkanDevice, cmd: vk::CommandBuffer, f: F)
    where
        F: FnOnce(&VulkanDevice, vk::CommandBuffer),
    {
        let token = self
            .profiler
            .as_mut()
            .map(|p| p.begin(&dev.device, cmd, name.to_string()));
        f(dev, cmd);
        if let (Some(p), Some(t)) = (self.profiler.as_mut(), token) {
            p.end(&dev.device, cmd, t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gemv(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader: ShaderId,
        weights: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        label: &str,
    ) {
        let kernel = registry.get(shader);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        self.write_bindings(
            dev, set,
            &[
                (0, weights, 0, 0),
                (1, input, 0, 0),
                (2, output, 0, 0),
                (3, self.fuse0.handle, 0, 0),
                (4, self.fuse1.handle, 0, 0),
            ],
        );
        let pc = MatVecPushConstants {
            ncols: k, stride_a: k, stride_b: k, stride_d: m,
            batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
            fusion_flags: 0, base_work_group_y: 0,
            ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            // Phase-3C: GEMV pipeline is now built with NUM_ROWS = MMV_NUM_ROWS
            // (= 2). Each workgroup writes NUM_ROWS output rows, so the
            // dispatch count divides — ceiling-div to handle a tail
            // workgroup when m isn't a multiple of NUM_ROWS (the shader
            // bounds-checks via `first_row + NUM_ROWS <= stride_d`).
            let n_rows = super::pipeline_registry::MMV_NUM_ROWS;
            let groups = (m + n_rows - 1) / n_rows;
            dev.device.cmd_dispatch(cmd, groups, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rms_norm(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        weight: vk::Buffer,
        output: vk::Buffer,
        cols: u32,
        rows: u32,
        eps: f32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::RmsNorm);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        self.write_bindings(
            dev, set,
            &[(0, input, 0, 0), (1, weight, 0, 0), (2, output, 0, 0)],
        );
        let pc = GenericBinaryPushConstants {
            ne: cols * rows,
            ne00: cols, ne01: rows, ne02: 1, ne03: 1,
            nb00: 1, nb01: cols, nb02: cols * rows, nb03: cols * rows,
            ne10: cols, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: cols, nb12: cols, nb13: cols,
            ne20: cols, ne21: rows, ne22: 1, ne23: 1,
            nb20: 1, nb21: cols, nb22: cols * rows, nb23: cols * rows,
            misalign_offsets: 0,
            param1: eps, param2: 0.0, param3: 0,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, rows, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rope_neox(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        n_rows: u32,
        position: u32,
        label: &str,
    ) {
        let _ = position; // already written into rope_pos_buf
        let kernel = registry.get(ShaderId::RopeNeox);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        self.write_bindings(
            dev, set,
            &[
                (0, input, 0, 0),
                (1, self.rope_pos_buf.handle, 0, 0),
                (2, self.rope_ff_buf.handle, 0, 0),
                (3, output, 0, 0),
                (4, self.rope_idx_buf.handle, 0, 0),
            ],
        );
        let pc = RopePushConstants {
            rope_mode: 2, // GGML_ROPE_TYPE_NEOX
            nrows: n_rows,
            n_dims: head_dim,
            freq_scale: 1.0,
            freq_base: self.config.rope_freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale: self.rope_theta_scale,
            has_ff: 0,
            sections: [0; 4],
            is_imrope: 0,
            is_back: 0,
            set_rows_stride: 0,
            ne00: head_dim,
            ne01: n_rows,
            ne02: 1,
            nb01: head_dim,
            nb02: head_dim * n_rows,
            nb03: head_dim * n_rows,
            nb11: head_dim,
            nb12: head_dim,
            nb13: head_dim,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, n_rows, 1, 1);
        });
    }

    fn run_scalar_attn(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        position: u32,
    ) {
        let cfg = self.config.clone();
        let kernel = registry.get(ShaderId::ScalarAttn);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        let layer_off = self.kv_cache.layer_offset_bytes(layer);
        let layer_size = (self.kv_cache.config.max_seq_len as u64)
            * (cfg.n_kv_heads as u64)
            * (cfg.head_dim as u64)
            * 4;
        self.write_bindings(
            dev, set,
            &[
                (0, self.q_buf.handle, 0, 0),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, self.attn_out.handle, 0, 0),
            ],
        );
        let pc = ScalarAttnPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            seq_len: position + 1,
            max_seq: self.kv_cache.config.max_seq_len,
            scale: self.attn_scale,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile("scalar_attn", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, cfg.n_heads, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_binary(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader: ShaderId,
        a: vk::Buffer,
        b: vk::Buffer,
        d: vk::Buffer,
        n: u32,
        label: &str,
    ) {
        let kernel = registry.get(shader);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        self.write_bindings(dev, set, &[(0, a, 0, 0), (1, b, 0, 0), (2, d, 0, 0)]);
        let pc = GenericBinaryPushConstants {
            ne: n,
            ne00: n, ne01: 1, ne02: 1, ne03: 1,
            nb00: 1, nb01: n, nb02: n, nb03: n,
            ne10: n, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: n, nb12: n, nb13: n,
            ne20: n, ne21: 1, ne22: 1, ne23: 1,
            nb20: 1, nb21: n, nb22: n, nb23: n,
            misalign_offsets: 0,
            param1: 0.0, param2: 0.0, param3: 0,
        };
        let dispatch_y = (n + 511) / 512;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, 1, dispatch_y, 1);
        });
    }

    fn run_silu(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        n: u32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::Silu);
        let set = self.alloc_set(dev, kernel.descriptor_set_layout);
        self.write_bindings(dev, set, &[(0, input, 0, 0), (1, output, 0, 0)]);
        let pc = GenericHeadPushConstants {
            kx: n, ky: 1,
            param1: 0.0, param2: 0.0, param3: 0.0, param4: 0.0,
        };
        let dispatch_x = (n + 511) / 512;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
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

fn layer_weight(model: &LoadedModel, layer: u32, suffix: &str) -> vk::Buffer {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .buffer
        .handle
}

/// Q4_K_M mixes quant types — `attn_v.weight` and `ffn_down.weight`
/// are Q6_K, the rest are Q4_K. Pick the matching GEMV pipeline.
fn layer_weight_shader(model: &LoadedModel, layer: u32, suffix: &str) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    match model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type
    {
        GgmlType::Q6K => ShaderId::MulMatVecQ6K,
        _ => ShaderId::MulMatVecQ4K,
    }
}

fn compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

