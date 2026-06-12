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
///
/// `pub(crate)` so the server's auto-ctx-size sizing can read the
/// *active* KV dtype (its byte width drives the KV-per-token estimate)
/// from the same source `KvCache::new` uses — no divergence.
pub(crate) fn kv_dtype_from_env() -> KvDtype {
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

/// Fail-loud guard for the known-broken **non-FP8 KV-cache path of the
/// gemma-4-MoE (26B-A4B)** attention. With F32/F16 KV this model produces a
/// Layer-0 attention NaN → garbage `<pad>` output (latent VF bug, version-
/// invariant; root cause in `results/gemma_nan_release_bisect.md`). Only FP8
/// (E4M3) KV is correct for it — the canonical mandatory KV mode at 26B.
///
/// Scope is deliberately narrow: **only gemma-4-MoE**. Dense models *and*
/// qwen35 run F32/F16 KV correctly (verified blast-radius) and are NOT
/// guarded. The caller derives `is_gemma4_moe` from the parsed config as
/// `config.gemma4.as_ref().map_or(false, |g| g.enable_moe_block)` — gemma-4
/// arch AND `expert_count > 0` (excludes E2B, which has no experts).
///
/// Default: hard `Err` (no silent NaN). Escape hatch
/// `VULKANFORGE_ALLOW_BROKEN_KV=1` → loud warning + proceed (for the eventual
/// numerical fix work on the F32/F16-KV path).
pub fn guard_kv_precision(is_gemma4_moe: bool, kv_dtype: KvDtype) -> Result<(), String> {
    let force = std::env::var("VULKANFORGE_ALLOW_BROKEN_KV")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    guard_kv_precision_inner(is_gemma4_moe, kv_dtype, force)
}

/// Pure core of [`guard_kv_precision`] — `force` injected so the override
/// path is unit-testable without mutating process env (avoids test races).
fn guard_kv_precision_inner(
    is_gemma4_moe: bool,
    kv_dtype: KvDtype,
    force: bool,
) -> Result<(), String> {
    if !is_gemma4_moe || kv_dtype == KvDtype::F8 {
        return Ok(());
    }
    let msg = format!(
        "gemma-4-MoE (26B) requires VULKANFORGE_KV_FP8=1 — non-FP8 KV (here: {}; both F16 and F32 \
         are affected) is known-broken for this model (Layer-0 attention NaN → garbage output); \
         only FP8 (E4M3) KV is correct. Set VULKANFORGE_KV_FP8=1, or VULKANFORGE_ALLOW_BROKEN_KV=1 \
         to force.",
        kv_dtype.label()
    );
    if force {
        eprintln!("VulkanForge: ⚠️  {msg}");
        eprintln!(
            "VulkanForge: ⚠️  proceeding anyway (VULKANFORGE_ALLOW_BROKEN_KV=1) — output will be invalid."
        );
        Ok(())
    } else {
        Err(msg)
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

/// KV-cache bytes consumed **per context token** (both K and V buffers),
/// mirroring the allocation math in [`KvCache::new`]: `2 × elem_size ×
/// Σ_layer (n_kv_heads(layer) × head_dim(layer))`. Honors the per-layer
/// overrides (heterogeneous Gemma-4 sliding/full, Qwen3.6 recurrent
/// layers that contribute 0 KV heads). `max_ctx × this` is therefore the
/// exact total KV footprint at `max_seq_len = max_ctx`.
///
/// Used by the server's auto-ctx-size sizing to invert that relation:
/// `max_ctx = avail_vram / kv_bytes_per_token`. Pure / file-derived
/// inputs only → unit-testable without a device.
pub(crate) fn kv_bytes_per_token(
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    per_layer_head_dim: Option<&[u32]>,
    per_layer_n_kv_heads: Option<&[u32]>,
    elem_size: u64,
) -> u64 {
    let mut per_token_elems: u64 = 0;
    for i in 0..(n_layers as usize) {
        let hd = per_layer_head_dim.map(|v| v[i]).unwrap_or(head_dim) as u64;
        let kvh = per_layer_n_kv_heads.map(|v| v[i]).unwrap_or(n_kv_heads) as u64;
        per_token_elems = per_token_elems.saturating_add(kvh.saturating_mul(hd));
    }
    // ×2 for the separate K and V buffers (KvCache::new allocates both).
    per_token_elems.saturating_mul(elem_size).saturating_mul(2)
}

pub struct KvCache {
    pub k_buffer: GpuBuffer,
    pub v_buffer: GpuBuffer,
    pub config: KvCacheConfig,
    pub current_seq_len: u32,
    /// v0.2 Sprint 9d.1 — element type chosen at allocation time.
    /// Effective default is **F16** (`kv_dtype_from_env`):
    /// `VULKANFORGE_FP16_KV=0` opts out to F32; `VULKANFORGE_KV_FP8=1`
    /// selects F8 (E4M3).
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
                // Hardening (F5): checked products — file-derived dims feed
                // these. checked_* gives a clean Err instead of a wrap
                // (release) / panic (overflow-checks) on an absurd config.
                let layer_bytes = (config.max_seq_len as u64)
                    .checked_mul(kvh as u64)
                    .and_then(|x| x.checked_mul(hd as u64))
                    .and_then(|x| x.checked_mul(elem_size))
                    .ok_or("kv-cache per-layer size overflow")?;
                acc = acc
                    .checked_add(layer_bytes)
                    .ok_or("kv-cache total size overflow")?;
                offsets.push(acc);
            }
            (acc, Some(offsets))
        } else {
            let total = (config.n_layers as u64)
                .checked_mul(config.n_kv_heads as u64)
                .and_then(|x| x.checked_mul(config.max_seq_len as u64))
                .and_then(|x| x.checked_mul(config.head_dim as u64))
                .and_then(|x| x.checked_mul(elem_size))
                .ok_or("kv-cache size overflow")?;
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

    /// Sprint F.2a — zero ONLY one layer's K/V region (NOT the whole
    /// cache — that would erase the trunk's prompt/decode KV mid-run).
    /// Used to make the qwen35 MTP block's cold KV slot (block 64, never
    /// written by the trunk after F.1-fix) read zeros instead of garbage
    /// before the first draft attention. One `vkCmdFillBuffer` per buffer.
    pub fn zero_layer(
        &self,
        dev: &super::device::VulkanDevice,
        layer: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let off = self.layer_offset_bytes(layer);
        let size = self.layer_size_bytes(layer);
        let cmd_ctx = super::commands::CommandContext::new(
            &dev.device,
            dev.queue_family_index,
        )?;
        let result = cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_fill_buffer(cmd, self.k_buffer.handle, off, size, 0);
            dev.device.cmd_fill_buffer(cmd, self.v_buffer.handle, off, size, 0);
        });
        cmd_ctx.destroy(&dev.device);
        result?;
        Ok(())
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

    #[test]
    fn kv_bytes_per_token_uniform_matches_alloc() {
        // Qwen3-14B-ish: 40 layers × 8 kv_heads × 128 head_dim.
        let (nl, kvh, hd) = (40u32, 8u32, 128u32);
        // F16 (2B): 2 × 2 × Σ(8×128) = 2 × 2 × 40×1024 = 163_840 B/tok.
        let f16 = kv_bytes_per_token(nl, kvh, hd, None, None, 2);
        assert_eq!(f16, 2 * 2 * (nl as u64) * (kvh as u64) * (hd as u64));
        // The full cache at max_seq equals KvCache::new's `2 × bytes`:
        // 2 buffers × (nl × kvh × max_seq × hd × elem).
        let max_seq: u64 = 12288;
        assert_eq!(
            f16 * max_seq,
            2 * (nl as u64) * (kvh as u64) * max_seq * (hd as u64) * 2,
        );
    }

    #[test]
    fn kv_bytes_per_token_fp8_is_half_of_f16() {
        let (nl, kvh, hd) = (40u32, 8u32, 128u32);
        let f16 = kv_bytes_per_token(nl, kvh, hd, None, None, 2);
        let fp8 = kv_bytes_per_token(nl, kvh, hd, None, None, 1);
        assert_eq!(fp8 * 2, f16, "FP8 (1B) KV must be exactly half of F16 (2B)");
    }

    #[test]
    fn kv_bytes_per_token_heterogeneous_sums_per_layer() {
        // 4-layer mixed cache: 2 sliding (kvh=2, hd=256) + 2 full
        // (kvh=4, hd=512), FP8 (1B). Σ(kvh×hd) = 2·(2·256) + 2·(4·512)
        // = 1024 + 4096 = 5120 elems/tok → ×1 ×2 buffers = 10_240 B/tok.
        let plhd = [256u32, 256, 512, 512];
        let plkv = [2u32, 2, 4, 4];
        let got = kv_bytes_per_token(4, 999, 999, Some(&plhd), Some(&plkv), 1);
        assert_eq!(got, 2 * (2 * 256 + 2 * 256 + 4 * 512 + 4 * 512));
        // The uniform fallback args (999) must be ignored when per-layer
        // tables are present.
        assert_eq!(got, 10_240);
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

    // ---- Sprint [kv-guard]: fail-loud guard for non-FP8 KV @ gemma-4-MoE ----
    // Pure tests on `guard_kv_precision_inner` (force injected; no env mutation).

    /// gemma-4-MoE + non-FP8 KV (F32 and F16) → hard Err with an actionable
    /// message naming the required env var.
    #[test]
    fn guard_gemma4_moe_nonfp8_kv_is_err() {
        for kv in [KvDtype::F32, KvDtype::F16] {
            let r = guard_kv_precision_inner(true, kv, false);
            assert!(r.is_err(), "{kv:?} KV on gemma-4-MoE must Err");
            assert!(
                r.unwrap_err().contains("VULKANFORGE_KV_FP8=1"),
                "message must name the fix env var"
            );
        }
    }

    /// gemma-4-MoE + FP8 KV → no false positive (loads normally).
    #[test]
    fn guard_gemma4_moe_fp8_kv_is_ok() {
        assert!(guard_kv_precision_inner(true, KvDtype::F8, false).is_ok());
    }

    /// Non-gemma-4-MoE (dense / qwen35) + non-FP8 KV → guard must NOT fire
    /// (these run F32/F16 KV correctly per the blast-radius).
    #[test]
    fn guard_non_gemma4_moe_nonfp8_kv_is_ok() {
        for kv in [KvDtype::F32, KvDtype::F16, KvDtype::F8] {
            assert!(
                guard_kv_precision_inner(false, kv, false).is_ok(),
                "dense/qwen35 ({kv:?}) must not be guarded"
            );
        }
    }

    /// Override (`VULKANFORGE_ALLOW_BROKEN_KV=1` → force=true) → warning, not
    /// Err, even for gemma-4-MoE + non-FP8.
    #[test]
    fn guard_override_forces_ok() {
        assert!(guard_kv_precision_inner(true, KvDtype::F32, true).is_ok());
        assert!(guard_kv_precision_inner(true, KvDtype::F16, true).is_ok());
    }
}
