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
                ShaderId::RmsNorm | ShaderId::RmsNormMulRope => {
                    // SpecId 0 = norepeat (false → broadcast-safe), 1 =
                    // do_multiply (true → norm-with-gamma, which is the
                    // standard transformer use case). Sprint 9c.5 reuses
                    // the same spec for the RMS_NORM_ROPE_FUSION SPV.
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
                // silu, swiglu, multi_add_rms, copy, rope_norm,
                // rope_neox: no spec consts.
                ShaderId::Silu
                | ShaderId::SwiGLU
                | ShaderId::MultiAddRms
                | ShaderId::Copy
                | ShaderId::RopeNorm
                | ShaderId::RopeNeox => {
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
                ShaderId::MulMmqQ4K | ShaderId::MulMmqQ6K
                | ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL => {
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
                    // Phase 6 v0.1.2 cont. — GEMM tile-tuning.
                    //
                    // Phase 3C originally pinned TM=4 / TN=2 (the
                    // default in llama.cpp's vulkan-shaders-gen for
                    // a non-coopmat MMQ build). Bench sweep over the
                    // 5-prompt suite on Qwen3-8B at v0.1.2:
                    //   TM=4 TN=2  → 716 tok/s  (Phase 3C baseline)
                    //   TM=2 TN=2  → 776 tok/s  (+8 %)
                    //   TM=2 TN=4  → 789 tok/s  (+10 %, new default)
                    //   TM=8 TN=1  → 669 tok/s  (-7 %)
                    // The N-dim benefits more from per-thread tile
                    // coverage on RDNA4 than the M-dim does — the
                    // GEMM is K-major-loaded (Q4_K weights along
                    // rows, Q8_1 activations along cols), so giving
                    // each thread 4 N-cols × 2 M-rows reuses cached
                    // weight rows better than 2 N-cols × 4 M-rows.
                    //
                    // Constraint: WNITER = WM*WN/(WARP*TM*TN*WMITER)
                    // must be a positive integer. With WM=WN=32,
                    // WARP=64, WMITER=2: TM*TN ≤ 8.
                    //
                    // Override via VULKANFORGE_GEMM_BLOCK_SIZE / _TM
                    // / _TN env vars for A/B testing without rebuild.
                    //
                    // Phase 7 — BLOCK_SIZE = NUM_WARPS * WARP must cover
                    // (BM/WM)*(BN/WN) warp tiles per workgroup.
                    // For BM=BN=64, WM=WN=32, WARP=64 (Wave64): we need
                    // Sprint 4 — Grand-Audit-driven pivot was attempted
                    // (BLOCK_SIZE 256→128, WMITER 2→1, TN 4→2 to match
                    // llama.cpp's M variant for K-quants) and **regressed
                    // by -19%** on the 5-prompt bench (740 → 598
                    // tok/s). Other sweeps tested:
                    //   BS=256 WMITER=2 TM=4 TN=2 (Audit's TM/TN swap)  → 691 tok/s (-7%)
                    //   BS=128 WMITER=1 WM=WN=32 TM=2 TN=4              → 596 tok/s (-19%)
                    //   BS=128 WMITER=1 WM=64 WN=32 TM=2 TN=2 (M-var)   → 618 tok/s (-16%)
                    //   BS=64  WMITER=1 (S-var-ish)                     → 532 tok/s (-28%)
                    //   BS=256 WMITER=2 TM=2 TN=4 (Phase-7 default)    → 740 tok/s (baseline)
                    //
                    // The 2.17× gap to llama.cpp is NOT in the kernel
                    // spec constants — they're a local optimum already.
                    // See results/v02_sprint4_spec_constants.md for the
                    // bisection log.
                    //
                    // Each parameter is overridable via env var so future
                    // sprints can sweep without rebuild. Defaults match
                    // the Phase-7 v0.1.3 baseline.
                    //
                    // Constraint reminder (Phase-7 silent-corruption bug):
                    //   NUM_WARPS = BLOCK_SIZE / WARP = 256/64 = 4
                    //   warp tiles per WG = (BM/WM)·(BN/WN)·WMITER
                    //                     = (64/32)·(64/32)·2 = 8
                    //   → 4 warps cover 8 tiles via WNITER=2. ✓
                    // Sprint 11C — L-tile pinned from llama.cpp's
                    // l_warptile_mmq_int_k AMD-coopmat-override
                    // (ggml-vulkan.cpp:3368, gfx1201 path):
                    //   { 256, 128, 128, 32, 64, 64, 1, 4, 2, 1, 64 }
                    // Constraints satisfied:
                    //   WNITER    = WM·WN/(WARP·TM·TN·WMITER)
                    //             = 64·64/(64·4·2·1) = 8 ✓ ganzzahlig
                    //   NUM_WARPS = BLOCK_SIZE/WARP = 256/64 = 4
                    //             = (BM/WM)·(BN/WN) = 2·2 = 4 ✓ Phase-7-coverage
                    // The L-tile is hard-pinned (no env override) — its
                    // values come from llama.cpp upstream production
                    // tuning and the constraint chain breaks if any of
                    // BM/BN/WM/WN/WMITER/TM/TN move independently. The
                    // S-tile keeps the env-var override surface for the
                    // existing v0.1.x debugging workflow.
                    let is_l = matches!(id, ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL);
                    let (block_size, bm, bn, wm, wn, wmiter, tm, tn) = if is_l {
                        (256, 128, 128, 64, 64, 1, 4, 2)
                    } else {
                        let block_size: u32 = std::env::var("VULKANFORGE_GEMM_BLOCK_SIZE")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
                        let bm: u32 = std::env::var("VULKANFORGE_GEMM_BM")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
                        let bn: u32 = std::env::var("VULKANFORGE_GEMM_BN")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
                        let wm: u32 = std::env::var("VULKANFORGE_GEMM_WM")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
                        let wn: u32 = std::env::var("VULKANFORGE_GEMM_WN")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
                        let wmiter: u32 = std::env::var("VULKANFORGE_GEMM_WMITER")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
                        let tm: u32 = std::env::var("VULKANFORGE_GEMM_TM")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
                        let tn: u32 = std::env::var("VULKANFORGE_GEMM_TN")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
                        (block_size, bm, bn, wm, wn, wmiter, tm, tn)
                    };
                    let data: [u32; 10] = [
                        block_size,
                        bm,
                        bn,
                        wm,
                        wn,
                        wmiter,
                        tm,
                        tn,
                        1,   // TK
                        64,  // WARP — RDNA Wave64 (subgroup_size_8 = max(64,8) on gfx1201)
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
                ShaderId::FlashAttn | ShaderId::FlashAttnFp16Kv => {
                    // SpecId 0 = MAX_SEQ — same convention as scalar_attn.
                    // 2048 covers the Phase-2 contexts; bump in Phase 4 if
                    // we go past 2048 tokens of context.
                    // Sprint 9d.3 — FlashAttnFp16Kv shares the same
                    // spec layout (FP16_KV is a build-time #define,
                    // not a spec const).
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
                ShaderId::MulMmQ4K | ShaderId::MulMmQ6K
                | ShaderId::MulMmQ4KAligned | ShaderId::MulMmQ6KAligned => {
                    // Phase 6 v0.1.2 — mul_mm.comp port. Same spec-
                    // constant layout as MulMmqQ4K/Q6K but no
                    // ACC_TYPEV2 (the build defines that as `vec2` —
                    // we still pin via spec-constant for runtime
                    // tunability).
                    //
                    // Phase 6 v0.1.2 (cont.) tile-tuning win was
                    // TM=2 / TN=4 on mul_mmq; the same shape
                    // applies here because the inner-loop math is
                    // identical (FP32 accumulator, vec2-packed
                    // shared memory).
                    // Phase 7 — same coverage rule as MulMmq (see comment
                    // in that branch above): BLOCK_SIZE = 256 → 4 warps.
                    let block_size: u32 = std::env::var("VULKANFORGE_GEMM_BLOCK_SIZE")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
                    let tm: u32 = std::env::var("VULKANFORGE_GEMM_TM")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
                    let tn: u32 = std::env::var("VULKANFORGE_GEMM_TN")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
                    let entries = [
                        entry(0, 0, 4),
                        entry(1, 4, 4),
                        entry(2, 8, 4),
                        entry(3, 12, 4),
                        entry(4, 16, 4),
                        entry(5, 20, 4),
                        entry(6, 24, 4),
                        entry(7, 28, 4),
                        entry(8, 32, 4),
                        entry(9, 36, 4),
                        entry(10, 40, 4),
                    ];
                    let data: [u32; 11] = [
                        block_size, // BLOCK_SIZE
                        64,  // BM
                        64,  // BN
                        32,  // BK
                        32,  // WM
                        32,  // WN
                        2,   // WMITER
                        tm,  // TM
                        tn,  // TN
                        1,   // TK (no coopmat)
                        64,  // WARP — RDNA Wave64
                    ];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
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
                ShaderId::ProbeInt8Coopmat => {
                    // Sprint 11F — pure runtime probe, no spec-constants.
                    // local_size_x=64 hardcoded; layout = (3 SSBOs).
                    ComputeKernel::from_spv(device, &words, cache)
                }
                ShaderId::MulMmQ4KCoopmat => {
                    // Sprint 11E — mul_mm.comp + COOPMAT, KHR coopmat
                    // 16x16x16 FP16xFP16->FP32 fragments. Spec-constants
                    // pinned from llama.cpp's warptile_mmq AMD-coopmat-
                    // override (ggml-vulkan.cpp:3367) at gfx1201:
                    //   { 256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64 }
                    //
                    // WNITER = WM·WN/(WARP·TM·TN·WMITER) is non-integer
                    // here (64·64/(64·16·16·2) = 0.125) — but the COOPMAT
                    // path of mul_mm.comp uses cms_per_row = WM/TM = 4
                    // and cms_per_col = WN/TN = 4 in the inner loop
                    // (line 178-179) instead of WNITER, so the 0
                    // truncation in the unused scalar fallback is
                    // benign.
                    //
                    // NUM_WARPS = BLOCK_SIZE/WARP = 4 = (BM/WM)·(BN/WN)
                    //                                = 2·2 ✓
                    let entries = [
                        entry(0, 0, 4),
                        entry(1, 4, 4),
                        entry(2, 8, 4),
                        entry(3, 12, 4),
                        entry(4, 16, 4),
                        entry(5, 20, 4),
                        entry(6, 24, 4),
                        entry(7, 28, 4),
                        entry(8, 32, 4),
                        entry(9, 36, 4),
                        entry(10, 40, 4),
                    ];
                    let data: [u32; 11] = [
                        256, // BLOCK_SIZE
                        128, // BM
                        128, // BN
                        32,  // BK (Q4_K → 32, per shader comment)
                        64,  // WM
                        64,  // WN
                        2,   // WMITER
                        16,  // TM (coopmat_m)
                        16,  // TN (coopmat_n)
                        16,  // TK (coopmat_k)
                        64,  // WARP — RDNA Wave64
                    ];
                    let bytes = bytemuck::bytes_of(&data);
                    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
                }
                ShaderId::FlashAttnTiledBr4
                | ShaderId::FlashAttnTiledBr8
                | ShaderId::FlashAttnTiledBr16
                | ShaderId::FlashAttnTiledBr16Bc32
                | ShaderId::FlashAttnTiledBr16Bc32Fp16Kv
                | ShaderId::FlashAttnBatchFp16Kv
                | ShaderId::FlashAttnSplitFp16Kv
                | ShaderId::KvCopyFp16
                | ShaderId::BenchQkScalar
                | ShaderId::BenchQkCoopmat
                | ShaderId::FlashAttnCoopmat
                | ShaderId::FlashAttnCoopmatFp16Kv => {
                    // No spec constants — BR/BC/HEAD_DIM/FP16_KV are
                    // baked in via -DBR=N -DBC=N (-DFP16_KV=1) at
                    // SPIR-V build time. KvCopyFp16 has no spec
                    // constants either.
                    ComputeKernel::from_spv(device, &words, cache)
                }
                // Sprint 3A — Q4_K coopmat with forward-pass layout.
                // No spec constants: the BN tile size is already baked
                // in at SPIR-V build time via -DBN; the rest of the
                // kernel geometry (BLOCK_SIZE=256, BM=64, BK=16) is
                // hard-coded `const uint` in the shader source.
                ShaderId::MulCoopmatQ4KFwdBn64
                | ShaderId::MulCoopmatQ4KFwdBn32
                | ShaderId::MulCoopmatQ4KFwdBn16
                | ShaderId::MulCoopmatQ4KNaiveBf16
                | ShaderId::MulCoopmatQ4KNaivePaddedBf16
                | ShaderId::MulCoopmatQ4KNaivePaddedFp8 => {
                    ComputeKernel::from_spv(device, &words, cache)
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
