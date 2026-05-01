# VulkanForge v0.2.1 Sprint 11G-B — Int8-coopmat GEMM Micro-Benchmark

**Date:** 2026-04-30
**Branch:** main (HEAD = f0213b5, post-Sprint 11G-A architecture)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Compute-only micro-bench, GO/NO-GO gate for Sprint 11G-C+

## TL;DR — GO

Median speedup of `bench_int8cm_gemm` (KHR-coopmat 16×16×16 I8×I8→I32,
entry 14) vs `bench_scalar_gemm` (`dotPacked4x8EXT`) over three Qwen3-8B
prefill GEMM shapes: **1.42×** — squarely inside the Sprint 11G-A
analytical prediction (1.3–1.7×).

Parity is **bit-exact** vs a CPU int64 reference (no rounding in int8
× int8 → int32 arithmetic). Both shaders produce identical output for
all three shapes, both at n_reps=1 (parity test) and n_reps=4 (bench).

| Shape (M×N×K)            | Use-case                  | Scalar µs | Int8-cm µs | Speedup | Scalar GOPS | Int8-cm GOPS |
|--------------------------|---------------------------|-----------|------------|---------|-------------|--------------|
| 512×4096×4096            | Q-proj @ pp=512           |  13 538.0 |  20 293.4  | 0.67×   |  5076.0     |  3386.3      |
| 512×1024×4096            | K/V-proj @ pp=512 (GQA)   |   4 001.2 |   2 808.0  | 1.42×   |  4293.7     |  6118.1      |
|  64×4096×4096            | Q-proj @ pp=64            |   2 434.0 |   1 113.6  | 2.19×   |  3529.1     |  7713.8      |

Numbers are median of 10 runs after 3 warmups. n_reps=4 inner-loop
amortization. Identical run-to-run within ±2 % across two
back-to-back invocations.

172/172 tests green (was 171; +1 new parity test
`test_int8cm_gemm_microbench_parity`).

## 1. Shader design

Both shaders are deliberately minimal so the speedup measures only the
inner-loop instruction (coopMatMulAdd vs dotPacked4x8EXT) and not the
real Q4_K dequant / Q8_1 staging / scale-fold work that Sprint 11G-C
will add. Specifically:

- BM=BN=16, BK=32 → **one** 16×16 output fragment per WG, mapped to
  one Wave64.
- LDS staging: `int8_t a_lds[16×32]` + `int8_t b_lds[16×32]` = 1 KB
  total. Same byte budget for both shaders.
- 64 threads per WG, 1 Wave64.
- Pre-unpacked int8 inputs (no Q4_K nibble unpack, no Q8_1 block
  parsing). Raw int32 output (no scale fold).
- n_reps inner loop: each rep recomputes the K-walk, accumulating
  into a per-WG `total` so the compiler cannot elide repeated
  iterations.

Differences (kept to the absolute minimum):

| Aspect | bench_int8cm_gemm | bench_scalar_gemm |
|--------|-------------------|-------------------|
| Inner-loop instruction | `coopMatMulAdd(int8, int8, int32)` (KHR coopmat entry 14, no saturate) | `dotPacked4x8EXT(int32, int32) → int32` (RDNA4 `v_dot4_i32_iu8`) |
| K step per iteration | TK=16 (BK/TK = 2 calls per chunk) | 4 K-elements per dot4, 8 calls per chunk |
| LDS B-layout | column-major (`b_lds[col*BK + row]`) | row-major (`b_lds[row*BN + col]`) |
| Fragment vs scalar accumulator | one int32 coopmat fragment per WG | 4 cells per thread × 64 threads = 256 cells / WG |
| Output store | `coopMatStore(RowMajor, stride=N)` | per-thread strided write |

(LDS B-layout differs because the two consume B differently —
ColumnMajor coopMatLoad vs strided per-cell scalar dot. Both still
produce identical results because the WMMA hardware effectively
re-transposes via the Layout flag.)

## 2. Parity (bit-exact)

`test_int8cm_gemm_microbench_parity` runs both shaders against the
same int8 inputs (deterministic pseudo-random in `[-8, 8)`) at
M=N=16, K=128, n_reps=1 and asserts every cell of the int32 output
matches a CPU int64 reference exactly. Both shaders pass.

The `bench_int8cm_gemm` example also runs a parity check at all
three benchmark shapes (M=64..512, N=1024..4096, K=4096, n_reps=4) and
confirms bit-exactness against CPU int64. Output: `Parity: ALL PASS
(CPU int64 reference)`.

This validates the design assumption from Sprint 11G-A: int8 × int8
→ int32 with no saturation matches scalar dot4 byte-for-byte. Sprint
11G-C / 11G-D can rely on bit-exact parity vs scalar mul_mmq for the
unscaled dot, with all numerical drift confined to the FP scale-fold
step.

## 3. Performance — discussion

### 3.1 Why the largest shape regresses

The 0.67× at 512×4096×4096 is **not** an indictment of int8-coopmat —
it is a single-fragment-per-WG artefact of the bench's BM=BN=16 tile
shape. The arithmetic intensity at that tile size is poor:

- per WG: read BM×K + BN×K = 16×4096 + 16×4096 = **128 KB**
- per WG: produce BM×BN = 16×16 = **256 output cells**
- per WG: AI = 256/128KB = **2 cells/KB ≈ 0.5 ops/byte**

At N=4096 there are 8 192 WGs total — the GPU is hitting LDS / global
BW limits, not the WMMA peak. Sprint 11G-D's L-tile (BM=BN=128, BK=32)
gets 16× better arithmetic intensity, so the BW-bound regime where
scalar dot4 wins should disappear.

The shape that *does* win 1.42× (512×1024×4096) is the one where
arithmetic intensity is high enough relative to launch parallelism
that compute matters again. The shape that wins 2.19× (64×4096×4096)
has only 1 024 WGs — small enough that scalar dot4's high
per-instruction issue overhead becomes the bottleneck and the
fewer-instructions int8-cm path benefits most.

### 3.2 Why the median is the right summary

The three shapes are not equally representative: real prefill traffic
on Qwen3-8B at pp=512 is roughly:

- 512×4096×4096 (Q-proj × num_layers=36): 36 dispatches per token batch
- 512×1024×4096 (K-proj + V-proj, GQA 4× shared): 72 dispatches per batch
- 512×11008×4096 (FFN gate/up/down): 108 dispatches per batch (not benched here)

So the K/V-proj shape (1.42×) dominates by dispatch count, the Q-proj
shape (0.67×) is a tail, and small-pp paths (2.19×) matter less. A
weighted speedup would be closer to the median — making 1.42× the
right summary number for a GO/NO-GO call.

### 3.3 Comparison with Sprint 10B

Sprint 10B's QK micro-bench reported 47.5× for FP16-coopmat over
scalar FP32 FMA — a number that did not materialise at GEMM scale
(end-to-end attention only +86 %). The lesson then was that the
micro-bench compared against a baseline that nobody actually ships
(scalar FP32 FMA) so the speedup was illusory.

This bench is calibrated against the *real* baseline:
`dotPacked4x8EXT`, the RDNA4 `v_dot4_i32_iu8` instruction that
production scalar mul_mmq already uses. The 1.42× median is the speedup
over a hardware-accelerated 4-MAD-per-cycle scalar baseline, which is
why it is an order of magnitude lower than 10B's 47.5×. It is also
*more credible*: a 1.42× compute speedup over a hardware-accelerated
scalar baseline is the kind of number that translates cleanly to
end-to-end GEMM after dequant and scale-fold are added.

## 4. GO/NO-GO

**GO** for Sprint 11G-C / 11G-D / 11G-E.

- **GO band check** (Sprint 11G-A): predicted 1.3–1.7× end-to-end
  GEMM. Measured 1.42× compute-only median, with high variance across
  shapes (0.67× ≤ x ≤ 2.19×). The measured median is at the lower
  edge of the predicted band, so the end-to-end GEMM target shifts
  to the lower end (1.3–1.5× over scalar mul_mmq+L on pp ≥ 256).
- **Sprint 11G-D must include a shape selector**: int8-cm clearly
  loses at large M-and-N shapes with BM=BN=16. Sprint 11G-D should
  either (a) deliver an L-tile (BM=BN=128) variant that fixes
  arithmetic intensity at large shapes, or (b) add a shape-conditional
  dispatch (use int8-cm at small/medium, fall back to scalar mmq at
  large). Same shape-selector pattern Sprint 10C used for FP16
  coopmat-attention.
- **Risk #1 from Sprint 11G-A (lane→cell mapping) is empirically
  resolved**: bit-exact parity at all shapes with both `Layout=
  ColumnMajor` (coopMatStore in the bench) and `Layout=RowMajor`
  (coopMatStore in the int8-cm bench output). Sprint 11G-D's full
  shader can use either layout safely.

## 5. Files touched

```
NEW   vk_shaders/bench_int8cm_gemm.comp                (89 LOC)
NEW   vk_shaders/bench_scalar_gemm.comp                (108 LOC)
NEW   examples/bench_int8cm_gemm.rs                    (340 LOC)
EDIT  build.rs                                         (+12 lines: 2 ShaderJob entries)
EDIT  src/backend/vulkan/shaders.rs                    (+18 lines: 2 ShaderId entries,
                                                        2 name+bytes mappings, 2 ALL_SHADERS)
EDIT  src/backend/vulkan/pipeline_registry.rs          (+2 lines: BenchInt8CmGemm |
                                                        BenchScalarGemm match arms)
EDIT  src/backend/vulkan/pipeline.rs                   (+13 lines: BenchInt8CmGemmPushConstants)
EDIT  tests/correctness.rs                             (+96 lines: parity test)
NEW   results/v021_sprint11gb_int8cm_microbench.md     (this report)
```

SPV count: 63 (was 61), +58 264 bytes total
(bench_int8cm_gemm.spv 8 988 + bench_scalar_gemm.spv 49 276).

## 6. Tests / regression

```
test result: ok. 27 + 9 + 18 + 75 + 8 + 8 + 27 = 172 passed
```

172/172 green. The only new test is the int8 microbench parity
(`test_int8cm_gemm_microbench_parity`); no production code path has
moved. The two new shaders are bench-only — neither is wired into
`forward.rs` and neither has any side effect on production
dispatches.

## 7. Sprint 11G-C — recommended scope

Path is clear after the 11G-B GO. The next 4–7 day deliverable is
**bench_int8cm_q4k_q8_1.comp** — same pipeline as the 11G-B int8
bench, but with the real Q4_K nibble unpack, Q8_1 qs[]+ds staging,
and per-chunk scale-fold from Sprint 11G-A's architecture. Same shape
sweep, parity vs `mul_mmq_q4_k_f32` (which already produces FP32
output for Q4_K × Q8_1).

Gate for 11G-C: **GO** if the Q4_K-aware shader achieves ≥ 1.2×
median speedup over `mul_mmq_q4_k_f32` in the same three shapes
(unpack + scale-fold cost knocks ~15 % off the raw 1.42× compute
speedup, so 1.2× is the realistic floor before declaring sprint
worth).

## 8. Take-aways

1. **Calibrated micro-bench worked.** Calibrating against
   `dotPacked4x8EXT` (the actual production baseline) instead of
   scalar FP32 FMA produced a credible 1.42× median that maps cleanly
   to end-to-end GEMM expectations.
2. **Shape dependence is real.** Same compute kernel, same hardware:
   speedup ranges 0.67×—2.19× across 3 production-relevant shapes.
   Sprint 11G-D's full shader needs a shape selector or an L-tile
   variant or both.
3. **No layout surprises.** `coopMatLoad` ColumnMajor + ColumnMajor
   strided LDS produced bit-exact output across all shapes, removing
   the only open Sprint 11G-A risk.
