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
    /// Sprint 18A — IEEE-754 FP8 E4M3 (1 byte/element). Halves
    /// KV-cache VRAM vs F16; opt-in via `VULKANFORGE_KV_FP8=1`.
    /// Requires the FP8 device feature: setting `KV_FP8=1` also
    /// implies `VULKANFORGE_ENABLE_FP8=1` so device.rs wires up
    /// `VK_EXT_shader_float8` automatically. Production-ready
    /// from chat on Qwen3-8B forward — all five attention paths
    /// (`flash_attn`, `flash_attn_split`, `flash_attn_batch`,
    /// `flash_attn_tiled_br16_bc32`, `flash_attn_coopmat`) have
    /// FP8_KV variants.
    F8,
}

impl KvDtype {
    pub fn element_size(self) -> u64 {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
            KvDtype::F8 => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            KvDtype::F32 => "FP32",
            KvDtype::F16 => "FP16",
            KvDtype::F8 => "FP8",
        }
    }
}

/// Read the KV-cache dtype env vars and return the resulting choice.
///
/// Precedence: `VULKANFORGE_KV_FP8=1` wins (and implies
/// `VULKANFORGE_ENABLE_FP8=1`). Otherwise `VULKANFORGE_FP16_KV=0`
/// requests FP32. Default is FP16, matching v0.3.x behaviour.
fn kv_dtype_from_env() -> KvDtype {
    if std::env::var("VULKANFORGE_KV_FP8")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        // Imply the device-feature opt-in so the user doesn't have
        // to set both. SAFETY: smoke-test / pre-Forward init only.
        if std::env::var("VULKANFORGE_ENABLE_FP8").is_err() {
            unsafe { std::env::set_var("VULKANFORGE_ENABLE_FP8", "1") };
        }
        return KvDtype::F8;
    }
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
    /// Sprint 43D-1 — per-layer head_dim (heterogeneous KV cache for
    /// Gemma-4 sliding=256 / full=512). When `None`, every layer uses
    /// the uniform `head_dim` above (Llama / Qwen / Gemma-1 path).
    /// When `Some(v)`, `v.len() == n_layers` and each entry overrides
    /// the layer's per-position stride; the total cache size is the
    /// sum of `max_seq × kv_heads_for(i) × per_layer_head_dim[i] × elem`
    /// across layers, and `pos_offset_bytes(layer, pos)` walks the
    /// cumulative offset table for the heterogeneous layout.
    pub per_layer_head_dim: Option<Vec<u32>>,
    /// Sprint 51B-pre — per-layer KV-head count. `None` → all layers
    /// use the uniform `n_kv_heads` above (E2B / Qwen3 / Llama).
    /// `Some(v)` → 26B-A4B-style mixed sliding (`num_key_value_heads`)
    /// and full (`num_global_key_value_heads`) layers; the cumulative
    /// offset table sums `max_seq × per_layer_n_kv_heads[i] ×
    /// head_dim_for(i) × elem` per layer.
    pub per_layer_n_kv_heads: Option<Vec<u32>>,
}

pub struct KvCache {
    pub k_buffer: GpuBuffer,
    pub v_buffer: GpuBuffer,
    pub config: KvCacheConfig,
    pub current_seq_len: u32,
    /// v0.2 Sprint 9d.1 — element type chosen at allocation time.
    /// Defaults to `F32`; `VULKANFORGE_FP16_KV=1` flips to `F16`.
    pub kv_dtype: KvDtype,
    /// Sprint 43D-1 — cumulative byte-offset table for heterogeneous
    /// per-layer head_dim. `layer_byte_offsets[i]` is the start of
    /// layer `i`'s slab in K (and in V). `[n_layers + 1]` entries —
    /// the last entry equals `bytes_per_buffer`. `None` for uniform-
    /// stride caches; consumer falls back to the closed-form
    /// `layer * (max_seq × n_kv_heads × head_dim × elem)`.
    pub layer_byte_offsets: Option<Vec<u64>>,
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
        // Sprint 43D-1 — heterogeneous layout: build the cumulative
        // offset table when `per_layer_head_dim` is set. `bytes` then
        // sums per-layer slab sizes; offsets table feeds the
        // pos_offset_bytes / layer_offset_bytes / layer_size_bytes
        // helpers downstream.
        // Sprint 51B-pre — heterogeneous offset table is taken whenever
        // either head_dim OR kv_heads vary per layer. Both Vecs have
        // the same length convention (`n_layers`); when only one is
        // set, the other contributes its uniform value to the product.
        let any_per_layer = config.per_layer_head_dim.is_some()
            || config.per_layer_n_kv_heads.is_some();
        let (bytes, layer_byte_offsets) = if any_per_layer {
            if let Some(plhd) = config.per_layer_head_dim.as_ref() {
                assert_eq!(
                    plhd.len() as u32, config.n_layers,
                    "per_layer_head_dim length must equal n_layers"
                );
            }
            if let Some(plkv) = config.per_layer_n_kv_heads.as_ref() {
                assert_eq!(
                    plkv.len() as u32, config.n_layers,
                    "per_layer_n_kv_heads length must equal n_layers"
                );
            }
            let mut offsets = Vec::with_capacity((config.n_layers as usize) + 1);
            let mut acc: u64 = 0;
            offsets.push(acc);
            for i in 0..(config.n_layers as usize) {
                let hd = config
                    .per_layer_head_dim
                    .as_ref()
                    .map(|v| v[i])
                    .unwrap_or(config.head_dim);
                let kvh = config
                    .per_layer_n_kv_heads
                    .as_ref()
                    .map(|v| v[i])
                    .unwrap_or(config.n_kv_heads);
                let layer_bytes = (config.max_seq_len as u64)
                    * (kvh as u64)
                    * (hd as u64)
                    * elem_size;
                acc += layer_bytes;
                offsets.push(acc);
            }
            (acc, Some(offsets))
        } else {
            let total = (config.n_layers as u64)
                * (config.n_kv_heads as u64)
                * (config.max_seq_len as u64)
                * (config.head_dim as u64)
                * elem_size;
            (total, None)
        };
        let _ = elem_size; // also referenced via kv_dtype below; silence warning if drift

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
            layer_byte_offsets,
        })
    }

    /// Sprint 43C — one-shot device-local zero-fill of both KV
    /// buffers. Required for Gemma-4: the cache layout is uniform
    /// (`cfg.head_dim = max(256, 512) = 512`), but sliding layers
    /// only write the first 256 elements of each per-position slot.
    /// Without zero-fill, the upper 256 elements of each pos would
    /// contain undefined memory; attention reads them with a
    /// uniform-stride push-const and pollutes the dot product. With
    /// the upper half initialised to zero, `Q[h, 256..512] × 0 = 0`
    /// — the attention math reduces to the head_dim=256 form
    /// arithmetically, even though the shader runs over the full
    /// uniform stride.
    ///
    /// This costs one vkCmdFillBuffer per buffer at startup. Llama /
    /// Qwen models call it too (uniform head_dim → zero-fill is a
    /// no-op semantically; the upper region is fully written by every
    /// layer), but the explicit zero gives all paths the same defined
    /// initial state.
    pub fn zero_fill(
        &self,
        dev: &super::device::VulkanDevice,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_ctx = super::commands::CommandContext::new(
            &dev.device,
            dev.queue_family_index,
        )?;
        let bytes = self.bytes_per_buffer();
        let result = cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_fill_buffer(cmd, self.k_buffer.handle, 0, bytes, 0);
            dev.device.cmd_fill_buffer(cmd, self.v_buffer.handle, 0, bytes, 0);
        });
        // Sprint 55A — CommandContext has no Drop impl; destroy
        // explicitly to avoid the 3-leaked-objects (CB + Fence + Pool)
        // warning at vkDestroyDevice time. Pattern mirrors every other
        // CommandContext::new call site that follows it with destroy().
        cmd_ctx.destroy(&dev.device);
        result?;
        Ok(())
    }

    /// Bytes per (layer × kv_head × max_seq × head_dim × f32) buffer.
    pub fn bytes_per_buffer(&self) -> u64 {
        self.k_buffer.size
    }

    /// Sprint 43D-1 — head_dim of layer `layer`. Returns the per-layer
    /// override when set (Gemma-4 heterogeneous), else the uniform
    /// `config.head_dim`.
    pub fn head_dim_for(&self, layer: u32) -> u32 {
        match self.config.per_layer_head_dim.as_ref() {
            Some(v) => v[layer as usize],
            None => self.config.head_dim,
        }
    }

    /// Sprint 51B-pre — per-layer KV-head count. Mirrors
    /// `head_dim_for(layer)`; falls back to the uniform
    /// `config.n_kv_heads` when no per-layer override is set.
    pub fn kv_heads_for(&self, layer: u32) -> u32 {
        match self.config.per_layer_n_kv_heads.as_ref() {
            Some(v) => v[layer as usize],
            None => self.config.n_kv_heads,
        }
    }

    /// Flat element offset for one layer's slice in K/V. Sprint
    /// 43D-1: heterogeneous-layout-aware.
    pub fn layer_offset_elems(&self, layer: u32) -> u64 {
        if let Some(off_table) = self.layer_byte_offsets.as_ref() {
            off_table[layer as usize] / self.kv_dtype.element_size()
        } else {
            (layer as u64)
                * (self.config.n_kv_heads as u64)
                * (self.config.max_seq_len as u64)
                * (self.config.head_dim as u64)
        }
    }

    /// Byte offset for K[layer, pos, kv_head=0, dim=0] — start of one
    /// token's row across all kv_heads. Pos-major layout: heads at
    /// the same pos are contiguous, so a single `vkCmdCopyBuffer`
    /// region of [`row_bytes()`] writes all kv_heads in one go.
    pub fn pos_offset_bytes(&self, layer: u32, pos: u32) -> u64 {
        let layer_byte = self.layer_offset_bytes(layer);
        let row = self.row_bytes(layer);
        layer_byte + (pos as u64) * row
    }

    /// Bytes per (kv_head × head_dim) — one token's full K-row /
    /// V-row of the cache. Sprint 43D-1: parameterised on layer for
    /// heterogeneous Gemma-4 layouts; old uniform layouts return the
    /// same value for every layer.
    pub fn row_bytes(&self, layer: u32) -> u64 {
        (self.kv_heads_for(layer) as u64)
            * (self.head_dim_for(layer) as u64)
            * self.kv_dtype.element_size()
    }

    /// Byte offset for K[layer, pos=0, kv_head=0, dim=0] — pass to
    /// the attention shader so its `t * pos_stride` indexing is
    /// rooted at the right layer. Sprint 43D-1: heterogeneous-layout-
    /// aware (cumulative table for Gemma-4, closed-form otherwise).
    pub fn layer_offset_bytes(&self, layer: u32) -> u64 {
        if let Some(off_table) = self.layer_byte_offsets.as_ref() {
            off_table[layer as usize]
        } else {
            self.layer_offset_elems(layer) * self.kv_dtype.element_size()
        }
    }

    pub fn is_fp8(&self) -> bool {
        self.kv_dtype == KvDtype::F8
    }

    pub fn is_fp16(&self) -> bool {
        self.kv_dtype == KvDtype::F16
    }

    /// Total bytes for one layer's K (or V) slice in the cache —
    /// `max_seq_len × n_kv_heads × per_layer_head_dim × element_size`.
    /// Sprint 43D-1: parameterised on layer.
    pub fn layer_size_bytes(&self, layer: u32) -> u64 {
        (self.config.max_seq_len as u64)
            * (self.kv_heads_for(layer) as u64)
            * (self.head_dim_for(layer) as u64)
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
