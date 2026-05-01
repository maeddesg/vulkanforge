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

/// v0.2 Sprint 9d.1 — KV-cache element type. Selected at allocation
/// time and immutable afterwards.
///
/// `F32` is the default and matches the current shader pipeline
/// (every attention shader and every KV-write `vkCmdCopyBuffer`
/// reads/writes plain `float`). `F16` halves the cache footprint
/// (4 B → 2 B per element) but is **not yet functional**: Sprint
/// 9d.1 only wires up the buffer-sizing math; the FP32 → FP16
/// conversion compute shader and FP16-aware attention SPVs land in
/// Sprint 9d.2 / 9d.3. Allocating `F16` today + reading it with
/// the existing FP32 shaders produces garbage, so the env-var that
/// selects it is opt-in and `KvCache::new` emits a loud warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    F32,
    F16,
}

impl KvDtype {
    pub fn element_size(self) -> u64 {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            KvDtype::F32 => "FP32",
            KvDtype::F16 => "FP16",
        }
    }
}

/// Read `VULKANFORGE_FP16_KV` and return the resulting dtype.
///
/// Sprint 9d.3 — flipped to **default ON** after the full FP16 KV
/// path (prefill + decode) was verified to keep all 167 regression
/// tests green, including the strict argmax-parity end-to-end tests
/// (`phase3e_prefill_batch_matches_token_by_token_top5`,
/// `phase5b2_decode_after_batched_prefill_qwen3`,
/// `phase_prompt16_alice_context_retention_qwen3`, etc.).
///
/// FP16 KV is now the default to match llama.cpp's Qwen3 behavior
/// and to give every run the 50% VRAM saving that enables longer
/// contexts. Opt-out: `VULKANFORGE_FP16_KV=0` for bit-exact FP32 KV.
fn kv_dtype_from_env() -> KvDtype {
    match std::env::var("VULKANFORGE_FP16_KV") {
        Ok(s) if s == "0" => KvDtype::F32, // explicit opt-out
        _ => KvDtype::F16,                 // default ON
    }
}

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
    /// v0.2 Sprint 9d.1 — element type chosen at allocation time.
    /// Defaults to `F32`; `VULKANFORGE_FP16_KV=1` flips to `F16`.
    pub kv_dtype: KvDtype,
}

impl KvCache {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Sprint 9d.1 — runtime-selectable dtype (default F32).
        let kv_dtype = kv_dtype_from_env();
        let elem_size = kv_dtype.element_size();
        let bytes = (config.n_layers as u64)
            * (config.n_kv_heads as u64)
            * (config.max_seq_len as u64)
            * (config.head_dim as u64)
            * elem_size;

        // Sprint 9d.3 — the WARN log from 9d.1 is gone; FP16 KV is
        // now end-to-end functional and the regression suite passes
        // with default ON. The startup config log below is the only
        // place we mention the dtype.

        let mb_total = bytes * 2 / (1024 * 1024);
        eprintln!(
            "VulkanForge: KV cache {} ({}B/elem) × 2 buffers = {} MB \
             ({} layers × {} kv_heads × {} max_seq × {} head_dim)",
            kv_dtype.label(),
            elem_size,
            mb_total,
            config.n_layers,
            config.n_kv_heads,
            config.max_seq_len,
            config.head_dim,
        );

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
            kv_dtype,
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
        elems * self.kv_dtype.element_size()
    }

    /// Bytes per (kv_head × head_dim) — one token's full K-row /
    /// V-row of the cache.
    pub fn row_bytes(&self) -> u64 {
        (self.config.n_kv_heads as u64)
            * (self.config.head_dim as u64)
            * self.kv_dtype.element_size()
    }

    /// Byte offset for K[layer, pos=0, kv_head=0, dim=0] — pass to
    /// the attention shader so its `t * pos_stride` indexing is
    /// rooted at the right layer.
    pub fn layer_offset_bytes(&self, layer: u32) -> u64 {
        self.layer_offset_elems(layer) * self.kv_dtype.element_size()
    }

    pub fn is_fp16(&self) -> bool {
        self.kv_dtype == KvDtype::F16
    }

    /// Total bytes for one layer's K (or V) slice in the cache —
    /// `max_seq_len × n_kv_heads × head_dim × element_size`. Used as
    /// the `range` argument when binding the KV cache to attention
    /// descriptors. Sprint 9d.1: replaces hardcoded `* 4`
    /// computations at the call sites.
    pub fn layer_size_bytes(&self) -> u64 {
        (self.config.max_seq_len as u64)
            * (self.config.n_kv_heads as u64)
            * (self.config.head_dim as u64)
            * self.kv_dtype.element_size()
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.k_buffer.destroy(device, allocator);
        self.v_buffer.destroy(device, allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_dtype_element_size() {
        assert_eq!(KvDtype::F32.element_size(), 4);
        assert_eq!(KvDtype::F16.element_size(), 2);
    }

    #[test]
    fn kv_dtype_label() {
        assert_eq!(KvDtype::F32.label(), "FP32");
        assert_eq!(KvDtype::F16.label(), "FP16");
    }

    /// Sprint 9d.1 — verify the element-size scaling cascades through
    /// the byte-offset accessors. This is a pure-arithmetic test (no
    /// Vulkan device, no allocation) — it directly constructs a
    /// `KvCache` value with synthetic GpuBuffer placeholders... well,
    /// actually GpuBuffer carries vk handles so we can't fabricate
    /// one. We test the offset math on `KvDtype` and the
    /// configuration arithmetic instead.
    #[test]
    fn layer_size_scales_with_dtype() {
        // Qwen3-8B realistic config.
        let n_layers: u64 = 36;
        let n_kv_heads: u64 = 8;
        let head_dim: u64 = 128;
        let max_seq: u64 = 2048;

        let f32_layer_bytes =
            max_seq * n_kv_heads * head_dim * KvDtype::F32.element_size();
        let f16_layer_bytes =
            max_seq * n_kv_heads * head_dim * KvDtype::F16.element_size();

        // FP16 must be exactly half of FP32 for the same shape.
        assert_eq!(f32_layer_bytes, 2 * f16_layer_bytes);

        // FP32 total (K + V across all layers): 36 × 2 × 8 × 128 ×
        // 2048 × 4 = 603 979 776 B = 576 MB.
        let f32_total = 2 * f32_layer_bytes * n_layers;
        assert_eq!(f32_total, 576 * 1024 * 1024);

        // FP16: 288 MB.
        let f16_total = 2 * f16_layer_bytes * n_layers;
        assert_eq!(f16_total, 288 * 1024 * 1024);
    }
}
