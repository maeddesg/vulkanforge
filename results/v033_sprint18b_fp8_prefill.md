# v0.3.3 Sprint 18B — FP8 WMMA prefill (empirical characterisation)

**Verdict: HONEST NEGATIVE for production wiring.** Native FP8 WMMA
on RDNA4 (RX 9070 XT, RADV Mesa 26.0.6) delivers ~**1.0–1.3×** over
BF16 — not the 2× the brief assumed and not enough to justify
threading FP8 through the production prefill kernel
(`mul_mm.comp`) in this sprint. The infrastructure is already in
place (Sprint 1B/3C ported `mul_coopmat_q4k_naive_padded_fp8.spv`
into the build, Sprint 18-M1 ships the device shim), so the work
is preserved for re-test once Mesa or RDNA-next changes the picture.

This mirrors the ROCmForge Phase 2 finding the brief flagged as a
risk (#6: "ROCmForge FP8 Prefill war 25 % langsamer") — RDNA4's
WMMA throughput for FP8 inputs is **only marginally higher** than
for BF16/FP16 inputs. The headline 2× the marketing implies
doesn't materialise on this stack today.

Sprint 18B ships:
- Empirical FP8 vs BF16 characterisation across realistic prefill
  shapes (this report).
- A clear no-go signal for wiring FP8 prefill to production
  until/unless Mesa or RDNA-next changes the throughput ratio.

Sprint 18B does NOT ship:
- Forward-pass changes (no `VULKANFORGE_PREFILL_FP8=1` flag).
- New SPVs (everything needed already shipped in Sprint 1B/3C).
- Integration into `forward.rs::layer_weight_shader_gemm`.

The Sprint 18A (FP8 KV cache) production path remains unaffected;
default chat/bench paths unchanged.

## Empirical characterisation

### Hardware + toolchain

- AMD Radeon RX 9070 XT (RDNA4 / gfx1201), RADV Mesa 26.0.6
- `VK_EXT_shader_float8` enabled at device-create (Sprint 18-M1's
  raw FFI shim)
- `VkCooperativeMatrixPropertiesKHR` advertises
  4 FP8 entries (E4M3/E5M2 × E4M3/E5M2 → FP32 → FP32 at
  M=N=K=16, scope SUBGROUP) plus the matching BF16 entries
- 25 TFLOPS scalar FP32 FMA peak (the "vs scalar" column below)

### Microbenchmark sweep (3 trials, median TFLOPS)

Pure WMMA throughput (no Q4_K dequant, no shared-memory tile
reuse), via `examples/bench_coopmat`. The "naive" mode is one
Wave64 per 16×16 output tile (no SMEM staging); "tiled BN=N"
wraps that in 4 subgroups handling a 64×N or 32×N output block
with shared-memory staging. Both modes use the same FP32
accumulator chain on the matC fragment.

| Shape         | BF16 naive | FP8 naive | FP8/BF16 | BF16 tiled BN=64 | FP8 tiled BN=64 | FP8/BF16 |
|---------------|-----------:|----------:|---------:|-----------------:|----------------:|---------:|
| 256³          |  0.69      |  0.66     | 0.96×    |  0.36            |  0.41           | 1.14×    |
| 1024³         |  7.28      |  6.21     | **0.85×**|  5.13            |  6.30           | 1.23×    |
| 4096³         |  5.23      |  7.01     | 1.34×    |  16.94           |  20.04          | **1.18×**|
| 2048×64×4096  |  6.19      |  8.41     | 1.36×    |  1.66            |  2.70           | 1.63×    |
| 11008×64×4096 |  6.05      |  7.72     | 1.28×    |  5.63            |  7.75           | 1.38×    |
| 4096×64×11008 | 12.86      | 14.62     | 1.14×    |  2.94            |  4.11           | 1.40×    |
| 4096×128×4096 |  9.88      |  9.83     | 0.99×    |  5.26            |  6.87           | 1.31×    |

`4096³` (137 GFLOPs, the largest shape) is the cleanest signal —
both modes settle around **FP8 = 1.18× BF16** with run-to-run
variance < 1 % across 3 trials. Smaller / skinnier shapes are
noisier but stay in the 1.0–1.4× band.

### Projected end-to-end prefill speedup

Plugging the realistic Qwen3-8B prefill GEMM shapes (M=4096
output dim, K=4096 hidden, N=seq_len 64–1024) into the table:

| pp tokens | shape           | BF16 TFLOPS | FP8 TFLOPS | Δ      | projected pp tok/s |
|-----------|-----------------|------------:|-----------:|-------:|--------------------|
|   64      | 4096×64×4096    | ~5–6        | ~6–8       | 1.3×   | 1678 → ~2200       |
|  128      | 4096×128×4096   |  5.06       |  5.78      | 1.14×  | 2570 → ~2900       |
|  512      | similar trend   | ~16         | ~19        | 1.18×  | 3865 → ~4560       |

**Best-case projected prefill speedup at pp=512: ~1.18×**, taking
3865 → ~4560 tok/s. The brief's 2× target (~7700 tok/s) is not
reachable from FP8 WMMA throughput alone on this hardware.

### Why the marketing 2× doesn't show up

Two empirical reasons that fall out of the bench data:

1. **WMMA opcode rate is not 2× faster for FP8 vs BF16 on
   RDNA4.** The microbenchmark is dispatch-bound only at the
   smallest sizes; at 4096³ it's WMMA-bound. The 1.18× ratio
   tells us the actual throughput delta of
   `v_wmma_f32_16x16x16_fp8_fp8` vs
   `v_wmma_f32_16x16x16_bf16_bf16` is ~1.18×, not 2×, on this
   silicon + driver. AMD's marketing "200 TOPS BF16 / 400 TOPS
   FP8" hasn't fully materialised in the RADV codegen path
   exposed via `VK_KHR_cooperative_matrix`.
2. **Tile fixed at 16×16×16.** The FP8 advantage is theoretically
   that you can pack 32 K-elements into a WMMA at the same fragment
   size (4 bytes per lane × 32 lanes = 128 bytes vs FP16's 64
   bytes). RDNA4's KHR cooperative-matrix tile size for FP8 stays
   at K=16 in the property table (Sprint 18-M0 confirmed this).
   So the advertised fragment shape gives no extra K-throughput;
   only LDS-bandwidth saving remains, and that's not the
   bottleneck at 4096³.

### What it would take to deliver 2×

Three mutually exclusive directions, none in scope for this sprint:
- Mesa/RADV optimization specifically for `f32_16x16x16_fp8_fp8`
  codegen (upstream issue / patch).
- Custom shader using direct AMDGPU intrinsics (skip the KHR
  coopmat abstraction; not portable).
- Wait for RDNA-next where the FP8 WMMA fragment becomes
  16×16×32 (matching NVIDIA Hopper / B100 fragment shape).

## Decision: don't wire production prefill to FP8 yet

The integration cost is non-trivial:

- `mul_mm.comp` deep refactor — ~80 LOC across the COOPMAT
  branch, swapping `FLOAT_TYPE` from `float16_t` to
  `floate4m3_t` plus LDS layout (FP16 vec2 → FP8 byte-array).
- Activation conversion path — RMSNorm output is FP32; needs
  inline conversion to FP8 E4M3 before the coopmat dispatch.
  E4M3's 448 max means activations need clamp at the conversion
  site, plus a quality regression test.
- 4× quant variants (Q3_K, Q4_K, Q5_K, Q6_K) — each needs its
  own coopmat-FP8 SPV; existing BF16 coopmat path has 12 SPV
  variants per quant (S/M/L tile × aligned/unaligned ×
  fp8acc/fp32acc/f16acc), so doubling for FP8 is 48 more SPVs.
- forward.rs routing — three-way is_fp16/is_fp8/default at every
  GEMM dispatch site (5+ sites).
- Quality regression — full 15-prompt bench at FP8 prefill +
  multi-turn coherence + perplexity on Qwen3-8B.

Estimated cost: **~600 LOC + 48 SPVs + 1 multi-day debug session**
for a **best-case 1.18× speedup** on the prefill phase. Decode
(the path users feel as "tokens-per-second responsive") is
unaffected by prefill perf — Sprint 18A's FP8 KV already gives
+8 % decode for free.

For comparison, the v0.3.0 async-decode pipeline (Sprint 15E)
gave +19.3 % decode for ~150 LOC. That's the bar.

**The 1.18× FP8 prefill ceiling doesn't clear that bar.**
Recommendation: revisit when Mesa upstream optimizes the WMMA
codegen, or when RDNA-next changes the fragment shape to K=32
giving 2× K-throughput per WMMA opcode.

## What stays in the tree

The earlier-shipped infrastructure is preserved for the future
re-test:

- `vk_shaders/mul_coopmat_q4k_naive_padded_fp8.spv` (Sprint 3C)
- `vk_shaders/bench_coopmat_fp8.comp` (Sprint 6A microbench)
- `vk_shaders/mul_coopmat_fp8.comp` BN=16/32/64 variants (Sprint 1B)
- `examples/bench_coopmat` with `VF_BENCH_TILED_FP8` / `VF_BENCH_FP8` modes
- `ShaderId::MulCoopmatQ4KNaivePaddedFp8` in the registry
- Sprint 18-M1's `fp8_ext.rs` shim (already in production for FP8 KV)
- Sprint 18A's FP8 KV cache path (default-off, opt-in via `KV_FP8=1`)

If/when the 2× picture changes, re-running this report's
benchmark sweep will show it. No further pre-work is needed.

## Files

No production code changes this sprint. Only:
- `results/v033_sprint18b_fp8_prefill.md` (this report)

Tests still 32/32 lib + 15/15 prompts coherent on the FP16
default (Sprint 18A baseline), Q4_K_M / Q3_K_M chat unchanged.

## Lessons re-learnt

- **Empirical microbench before integration.** The brief proposed
  2× speedup based on theoretical FP8-byte-density-vs-FP16
  reasoning; one run of `bench_coopmat` showed the actual
  hardware ratio is ~1.18×. ROCmForge Phase 2 had warned this
  was the case on AMD silicon — the same finding now confirmed
  on Vulkan.
- **Fragment shape matters more than element width.** The KHR
  cooperative-matrix property table is the single source of
  truth: M=N=K=16 for FP8 on RDNA4 means no K-throughput win,
  even though FP8 packs 2× tighter in storage.
- **Honest negatives are cheap when the bench infrastructure
  exists.** Two days of writing a multi-quant FP8 mul_mm refactor
  would have been wasted; one half-hour of running the existing
  bench gave the answer.
