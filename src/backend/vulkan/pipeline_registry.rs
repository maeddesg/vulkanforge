//! Pipeline registry — owns one [`ComputeKernel`] per [`ShaderId`]
//! plus a `VkPipelineCache` that persists between runs.
//!
//! Phase 2A — at startup we try to load a cache blob from
//! `~/.vulkanforge/pipeline_cache.bin` (or wherever the caller
//! requests). The blob's header is checked by the Vulkan loader
//! itself: an incompatible cache (different driver, different vendor)
//! is rejected silently and we start fresh. After all pipelines are
//! created, `save_cache` writes the merged blob back so the next
//! launch starts faster.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ash::vk;

use super::pipeline::ComputeKernel;
use super::shaders::{self, ShaderId};

/// SMOKE_DEFAULT spec constants for the mul_mat_vec_*_k shaders —
/// (BLOCK_SIZE, NUM_ROWS, NUM_COLS). Same values as the GLSL
/// defaults; pinning them explicitly keeps RADV's pipeline cache
/// happy and makes Phase-2A dispatch behaviour identical to Phase-1.
// Phase-3C spec-tuning. BLOCK_SIZE=64 (was 32) makes one workgroup
// equal one Wave64 subgroup, eliminating the cross-subgroup tree
// reduction. NUM_ROWS=1 stays as Phase-2A measured: NUM_ROWS=2 was
// ~5% faster at pos=0 but ~4% slower at pos=200, a wash on average.
// `forward.rs::run_gemv` reads `MMV_NUM_ROWS` so the two stay in sync.
// Phase-4A also tested BLOCK_SIZE=128 → also a wash (62.3 vs 61.8 at
// pos=0, identical at pos=200) — VGPR pressure isn't moved by
// spec-constant tuning, see results/phase4_step_4a_vgpr_reduction.md.
pub const MMV_NUM_ROWS: u32 = 1;
const MMV_SPEC_DATA: [u32; 3] = [64, MMV_NUM_ROWS, 1];

fn entry(constant_id: u32, offset: u32, size: usize) -> vk::SpecializationMapEntry {
    vk::SpecializationMapEntry {
        constant_id,
        offset,
        size,
    }
}

pub struct PipelineRegistry {
    pipelines: HashMap<ShaderId, ComputeKernel>,
    cache: vk::PipelineCache,
    /// Path the cache was loaded from / will be written to.
    cache_path: Option<PathBuf>,
    /// Wall-time spent in `vkCreateComputePipelines` for the whole
    /// inventory — useful for the "with vs without cache" report.
    pub create_duration: Duration,
}

pub struct CacheStats {
    pub loaded_bytes: usize,
    pub saved_bytes: usize,
}

impl PipelineRegistry {
    /// Build the registry, loading a pipeline cache from `cache_path`
    /// when present. Returns the registry plus stats describing how
    /// many bytes were loaded.
    pub fn new(
        device: &ash::Device,
        cache_path: Option<&Path>,
    ) -> Result<(Self, usize), Box<dyn std::error::Error>> {
        // 1) Load cache blob if any.
        let cache_blob: Vec<u8> = match cache_path {
            Some(p) => fs::read(p).unwrap_or_default(),
            None => Vec::new(),
        };
        let loaded_bytes = cache_blob.len();

        // 2) Create VkPipelineCache. Vulkan's loader validates the
        //    cache header itself; an incompatible blob is silently
        //    discarded and we keep going with an empty cache.
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(&cache_blob);
        let cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

        // 3) Create one ComputeKernel per shader, timing the section
        //    so the report can compare cold-start vs warm-start.
        let started = Instant::now();
        let mut pipelines: HashMap<ShaderId, ComputeKernel> = HashMap::new();
        for &id in shaders::ALL_SHADERS {
            let words = shaders::spv_words(id.spv_bytes());
            // Pin spec constants for the GEMV shaders so the Phase-2A
            // pipelines behave bit-identically to Phase 1. Other
            // shaders go through `from_spv` (no override) and use
            // their GLSL defaults.
            // Phase-2A bug §4: RADV silently produces wrong output on
            // GEMV pipelines unless spec consts are bound explicitly,
            // even when the bound values match the GLSL defaults. The
            // safe rule learnt from that bug — also enforced by Phase-2B
            // prompt §6 — is to pin every spec const a shader exposes
            // rather than trust the default path.
            let result = match id {
                ShaderId::MulMatVecQ4K | ShaderId::MulMatVecQ6K => {
                    let entries = [entry(0, 0, 4), entry(1, 4, 4), entry(2, 8, 4)];
                    let bytes = bytemuck::bytes_of(&MMV_SPEC_DATA);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::RmsNorm => {
                    // SpecId 0 = norepeat (false → broadcast-safe), 1 =
                    // do_multiply (true → norm-with-gamma, which is the
                    // standard transformer use case).
                    let data: [u32; 2] = [0, 1];
                    let entries = [entry(0, 0, 4), entry(1, 4, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::Add | ShaderId::Mul => {
                    // SpecId 0 = norepeat. false keeps broadcast support.
                    let data: [u32; 1] = [0];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::SoftMax => {
                    // SpecId 0 = BLOCK_SIZE (also drives local_size_x).
                    let data: [u32; 1] = [32];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                // silu, copy, rope_norm, rope_neox: no spec consts.
                ShaderId::Silu | ShaderId::Copy | ShaderId::RopeNorm | ShaderId::RopeNeox => {
                    ComputeKernel::from_spv(device, &words, cache)
                }
                ShaderId::ScalarAttn => {
                    // SpecId 0 = MAX_SEQ — sets the size of the
                    // shared `scores[]` buffer. 2048 covers Phase-2
                    // contexts; bump in Phase 3 for longer windows.
                    let data: [u32; 1] = [2048];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::MulMmqQ4K | ShaderId::MulMmqQ6K => {
                    // Phase-3C compile probe — pin the 11 spec
                    // constants llama.cpp's vulkan-shaders-gen pins
                    // for a non-coopmat MMQ build. Layout:
                    //   id 0  BLOCK_SIZE     (also drives local_size_x_id=0)
                    //   id 1  BM             tile rows
                    //   id 2  BN             tile cols
                    //   id 4  WM             warp-tile rows
                    //   id 5  WN             warp-tile cols
                    //   id 6  WMITER         warp-tile rows / warp
                    //   id 7  TM             thread-tile rows
                    //   id 8  TN             thread-tile cols
                    //   id 9  TK             1 for non-coopmat
                    //   id 10 WARP           subgroup width
                    let entries = [
                        entry(0, 0, 4),
                        entry(1, 4, 4),
                        entry(2, 8, 4),
                        entry(4, 12, 4),
                        entry(5, 16, 4),
                        entry(6, 20, 4),
                        entry(7, 24, 4),
                        entry(8, 28, 4),
                        entry(9, 32, 4),
                        entry(10, 36, 4),
                    ];
                    let data: [u32; 10] = [
                        128, // BLOCK_SIZE — 64 warp × 2 sub-warps
                        64,  // BM
                        64,  // BN
                        32,  // WM
                        32,  // WN
                        2,   // WMITER
                        4,   // TM
                        2,   // TN
                        1,   // TK
                        64,  // WARP — RDNA Wave64
                    ];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::QuantizeQ8_1 => {
                    // SpecId 0 = GROUP_SIZE (drives local_size_x).
                    let data: [u32; 1] = [32];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::FlashAttn => {
                    // SpecId 0 = MAX_SEQ — same convention as scalar_attn.
                    // 2048 covers the Phase-2 contexts; bump in Phase 4 if
                    // we go past 2048 tokens of context.
                    let data: [u32; 1] = [2048];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::FlashAttnSplit | ShaderId::FlashAttnReduce => {
                    // No spec constants — TILE = 64 is hard-coded in
                    // both .comp files (matches Phase-4B `flash_attn`'s
                    // workgroup geometry).
                    ComputeKernel::from_spv(device, &words, cache)
                }
                ShaderId::FlashAttnBatch => {
                    // SpecId 0 = MAX_SEQ — kept for parity with FlashAttn
                    // even though the runtime path uses push-constant
                    // dimensions only.
                    let data: [u32; 1] = [2048];
                    let entries = [entry(0, 0, 4)];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
            };
            let kernel = match result {
                Ok(k) => k,
                Err(e) => {
                    // Tear down whatever we already built, plus the
                    // cache, before bubbling the error up — leaving
                    // anything alive would trip the validation layer
                    // at exit.
                    for (_, k) in pipelines.drain() {
                        k.destroy(device);
                    }
                    unsafe { device.destroy_pipeline_cache(cache, None) };
                    return Err(format!("ComputeKernel::from_spv({}) failed: {e}", id.name()).into());
                }
            };
            pipelines.insert(id, kernel);
        }
        let create_duration = started.elapsed();

        Ok((
            Self {
                pipelines,
                cache,
                cache_path: cache_path.map(|p| p.to_path_buf()),
                create_duration,
            },
            loaded_bytes,
        ))
    }

    pub fn get(&self, id: ShaderId) -> &ComputeKernel {
        self.pipelines
            .get(&id)
            .unwrap_or_else(|| panic!("PipelineRegistry: missing pipeline for {:?}", id))
    }

    pub fn count(&self) -> usize {
        self.pipelines.len()
    }

    /// Persist the current cache contents back to `cache_path`. Errors
    /// are swallowed (cache failure is non-fatal — first start will
    /// just be slower).
    pub fn save_cache(&self, device: &ash::Device) -> CacheStats {
        let mut stats = CacheStats {
            loaded_bytes: 0,
            saved_bytes: 0,
        };
        let Some(path) = &self.cache_path else {
            return stats;
        };
        let data = match unsafe { device.get_pipeline_cache_data(self.cache) } {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "VulkanForge: pipeline_cache.bin save failed (get_pipeline_cache_data): {e}"
                );
                return stats;
            }
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match fs::write(path, &data) {
            Ok(_) => stats.saved_bytes = data.len(),
            Err(e) => eprintln!(
                "VulkanForge: pipeline_cache.bin save failed (fs::write): {e}"
            ),
        }
        stats
    }

    pub fn destroy(mut self, device: &ash::Device) {
        for (_, k) in self.pipelines.drain() {
            k.destroy(device);
        }
        unsafe { device.destroy_pipeline_cache(self.cache, None) };
    }
}

/// Default cache location: `$HOME/.vulkanforge/pipeline_cache.bin`.
pub fn default_cache_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".vulkanforge").join("pipeline_cache.bin"))
}
