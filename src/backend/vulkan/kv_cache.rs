//! Per-layer rolling K/V cache backed by two `vk::Buffer`s.
//!
//! Phase 2C. **Pos-major** storage layout, in elements:
//!   `K[layer, pos, kv_head, dim]`  → flat index
//!     `layer * max_seq * n_kv_heads * head_dim
//!      + pos * n_kv_heads * head_dim
//!      + kv_head * head_dim
//!      + dim`
//! and the same shape for V. Pos-major (vs head-major) means one
//! token's K/V across all kv_heads is contiguous — a single
//! `vkCmdCopyBuffer` per K and V at every token, instead of
//! n_kv_heads separate regions per token.
//!
//! A single `current_seq_len` is shared across layers — every layer
//! sees the same context window.
//!
//! Phase-2 budget for Qwen3-8B at max_seq=2048:
//!   per element type-size   = 4 B (f32)
//!   per layer per buffer    = 8 (kv_heads) × 2048 × 128 = 2 097 152 elems
//!   per layer per buffer    = 8 MiB
//!   for 36 layers × 2 buffers (K + V) = 576 MiB
//! comfortably fits in the ~11 GiB free after Qwen3-8B weight upload.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::buffers::GpuBuffer;

pub struct KvCacheConfig {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
}

pub struct KvCache {
    pub k_buffer: GpuBuffer,
    pub v_buffer: GpuBuffer,
    pub config: KvCacheConfig,
    pub current_seq_len: u32,
}

impl KvCache {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = (config.n_layers as u64)
            * (config.n_kv_heads as u64)
            * (config.max_seq_len as u64)
            * (config.head_dim as u64)
            * (std::mem::size_of::<f32>() as u64);

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let k_buffer = GpuBuffer::new(
            device,
            allocator,
            bytes,
            usage,
            MemoryLocation::GpuOnly,
            "kv_cache_K",
        )?;
        let v_buffer = GpuBuffer::new(
            device,
            allocator,
            bytes,
            usage,
            MemoryLocation::GpuOnly,
            "kv_cache_V",
        )?;
        Ok(Self {
            k_buffer,
            v_buffer,
            config,
            current_seq_len: 0,
        })
    }

    /// Bytes per (layer × kv_head × max_seq × head_dim × f32) buffer.
    pub fn bytes_per_buffer(&self) -> u64 {
        self.k_buffer.size
    }

    /// Flat element offset for one layer's slice in K/V (stride per
    /// layer = `n_kv_heads * max_seq * head_dim`).
    pub fn layer_offset_elems(&self, layer: u32) -> u64 {
        (layer as u64)
            * (self.config.n_kv_heads as u64)
            * (self.config.max_seq_len as u64)
            * (self.config.head_dim as u64)
    }

    /// Byte offset for K[layer, pos, kv_head=0, dim=0] — start of one
    /// token's row across all kv_heads. Pos-major layout: heads at
    /// the same pos are contiguous, so a single `vkCmdCopyBuffer`
    /// region of [`row_bytes()`] writes all kv_heads in one go.
    pub fn pos_offset_bytes(&self, layer: u32, pos: u32) -> u64 {
        let elems = self.layer_offset_elems(layer)
            + (pos as u64) * (self.config.n_kv_heads as u64) * (self.config.head_dim as u64);
        elems * (std::mem::size_of::<f32>() as u64)
    }

    /// Bytes per (kv_head × head_dim) — one token's full K-row /
    /// V-row of the cache.
    pub fn row_bytes(&self) -> u64 {
        (self.config.n_kv_heads as u64)
            * (self.config.head_dim as u64)
            * (std::mem::size_of::<f32>() as u64)
    }

    /// Byte offset for K[layer, pos=0, kv_head=0, dim=0] — pass to
    /// the attention shader so its `t * pos_stride` indexing is
    /// rooted at the right layer.
    pub fn layer_offset_bytes(&self, layer: u32) -> u64 {
        self.layer_offset_elems(layer) * (std::mem::size_of::<f32>() as u64)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.k_buffer.destroy(device, allocator);
        self.v_buffer.destroy(device, allocator);
    }
}
