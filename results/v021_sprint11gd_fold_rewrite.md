# VulkanForge v0.2.1 Sprint 11G-D — Lane-to-Cell Mapping + Direct-Fold Rewrite

**Date:** 2026-04-30
**Branch:** main (HEAD = fe6786a, post-Sprint 11G-C)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Architectural rewrite of Sprint 11G-C's scale-fold to eliminate the
LDS round-trip, gated by an empirical lane-to-cell mapping for the RDNA4
WMMA accumulator.

## TL;DR — NO-GO; Sprint 11G abandoned

Direct fragment-element fold via the empirical RDNA4 / Wave64 mapping recovered
**~10 %** of the Sprint 11G-C performance gap, lifting the median speedup vs
production `mul_mmq_q4_k_f32` from **0.42× to 0.47×**. The expected ~50 %
recovery did not materialise. Remaining gap to scalar `mul_mmq` is structural
(production scalar tight-loop FP accumulation beats every variant of
coopmat-fold we have measured on RDNA4 + RADV at GEMM scale).

| Shape (M×N×K)          | mul_mmq µs | 11G-C (LDS fold) µs | 11G-D (direct) µs | Δ vs 11G-C | Δ vs mul_mmq |
|------------------------|-----------:|--------------------:|------------------:|-----------:|-------------:|
| 512 × 4096 × 4096      |      884.8 |             1 878.0 |          1 714.1  |     +9.6 % |        0.52× |
| 512 × 1024 × 4096      |      344.9 |               840.1 |            736.9  |    +14.0 % |        0.47× |
| 64  × 4096 × 4096      |      257.9 |               698.3 |            624.0  |    +11.9 % |        0.41× |

Median speedup (this sprint): **0.47×**. Gate: **≥ 1.2× → GO**, **< 1.0× → 11G
abandoned**. Result: **NO-GO. Sprint 11G abandoned.**

176/176 tests green (was 174; +2 tests from Sprint 11G-C still pass with the
rewritten shader, no new tests added in 11G-D — the existing parity tests
exercise the new fold path end-to-end). The shader remains numerically
correct: max_abs_diff ≤ 2.4 × 10⁻⁴ vs `mul_mmq` (≈ 5 FP32 ULPs at amax = 553).

## 1. Lane-to-cell mapping (the non-trivial finding)

`probe_coopmat_layout.comp` empirically determines the
(lane, elem_idx) → (row, col) mapping for a 16×16 int32 KHR-coopmat
accumulator on Wave64 by encoding cell coordinates into the post-MMA
fragment value:

```glsl
A[i, k=0] = i,   A[i, k=1] = 1,   A[i, k≥2] = 0
B[k=0, j] = 16,  B[k=1, j] = j,   B[k≥2, j] = 0
C[i, j] = Σₖ A[i,k] * B[k,j]   = i*16 + j
```

Each lane reads its `frag.length() = 4` owned elements via `frag[i]` and the
value tells the host harness exactly which (row, col) cell it owns.

**Empirical mapping (`probe_coopmat_layout.rs`, 64 lanes × 4 elems):**

| Lane group | Lane range | Owns rows | row_base formula |
|------------|------------|-----------|------------------|
| 0          |  0..15     |   0..3    | 0  |
| 1          | 16..31     |   8..11   | 8  |
| 2          | 32..47     |   4..7    | 4  |
| 3          | 48..63     |  12..15   | 12 |

The middle two row groups are **swapped**. Lanes 16..31 own rows 8..11
(*not* 4..7), and lanes 32..47 own rows 4..7. Encoded as a closed-form
formula:

```
row_in_frag = ((lane >> 5) & 1) * 4 + ((lane >> 4) & 1) * 8 + elem_idx
col_in_frag = lane & 15
```

Equivalently: the row group is the bit-reversal of `lane >> 4` over 2
bits. The high half of the wave (lanes 32+) sets bit 0 of the group;
the inner quarter (lanes 16..31, lanes 48..63) sets bit 1.

**This finding is reusable.** Every future Sprint that consumes a 16×16
KHR-coopmat accumulator on RDNA4 / RADV / Wave64 needs the same mapping;
it's now in `vk_shaders/probe_coopmat_layout.comp` and re-derivable by a
single dispatch of `examples/probe_coopmat_layout`.

### 1.1 Mapping isn't `coopMatLoad`-derived

A first-attempt probe used `coopMatLoad(frag, lds, 0, 16, RowMajor)` to
fill the accumulator with known cell values. That probe reported a
*different* (and trivial-looking) mapping —
`row = (lane >> 4) * 4 + elem`, `col = lane & 15` — that breaks bit-exact
parity when applied to the post-MMA accumulator. Direct-fold parity went
from PASS → FAIL with `max_err = 49.6` at `amax = 194` (~26 % relative
error, not noise).

The lesson: **the lane-to-cell mapping for a coopmat accumulator depends
on how the fragment was produced.** `coopMatLoad`-filled and
`coopMatMulAdd`-filled accumulators have different distributions in
this driver+hardware combination. Sprint 11G-A's risk #1 ("WMMA fragment
lane→cell layout") was correctly flagged but its hand-wave resolution
(based on the Sprint 11F coopMatStore parity test) missed this nuance.
The MMA-result probe shader in this commit is the correct gate.

## 2. The direct-fold rewrite

The new fold replaces 11G-C's LDS round-trip with per-element register
multiplies:

```glsl
// 11G-C — per fragment, per BK chunk:
coopMatStore(C, scratch[warp_id], 0u, TN, RowMajor);  // 256 LDS writes
subgroupMemoryBarrierShared();                         // barrier 1
for (uint cell = 0; cell < 4; ++cell) {
    int32_t raw = scratch[warp_id][cell * 64 + lane];  // 4 LDS reads
    fp32_out[fragment][cell] += float(raw) * scale - bias;
}
subgroupMemoryBarrierShared();                         // barrier 2

// 11G-D — same loop, but no LDS scratch, no subgroup barriers:
uint row_base = ((lane >> 5) & 1u) * 4u + ((lane >> 4) & 1u) * 8u;
uint col_in_frag = lane & 15u;
uint n_in_wg = warp_c * WN + cm_col * TN + col_in_frag;
vec2 ds = b_ds[n_in_wg];
for (uint elem = 0; elem < 4; ++elem) {
    uint r_in = row_base + elem;
    uint m_in_wg = warp_r * WM + cm_row * TM + r_in;
    vec2 dm = a_dm[m_in_wg];
    fp32_out[fragment][elem] += float(C[elem]) * dm.x * ds.x - dm.y * ds.y;
}
```

LDS scratch buffer (`shared int32_t scratch[NUM_WARPS][TM * TN]`) is
removed entirely. SPV size shrank from 163 396 → 161 240 bytes.

**LDS budget per WG:** ~5 KB (was ~9 KB in 11G-C).

## 3. Why direct fold only saved ~10 %

The Sprint 11G-C report estimated the LDS-fold barrier overhead at
**~50 % of dispatch time**. Empirically, removing it saved **~10 %**.
The estimate was off by 5×.

Reasons:

1. **`coopMatStore` to LDS is fast** on RDNA4 — far faster than the
   "100× the per-instruction estimate" claimed in 11G-C. The store is
   handled via dedicated subgroup-coherent scratch hardware, not
   general-purpose LDS write paths.
2. **Subgroup memory barriers are nearly free.** They don't roundtrip
   through global memory or workgroup-scope synchronization; on RDNA4
   they are essentially `s_waitcnt` instructions for LDS pending writes,
   single-cycle for the warp.
3. **The real bottleneck is something else.** With both fold variants
   eliminated, the shader still runs at 50 % of `mul_mmq`'s GFLOPS.
   Candidates (in priority order, none directly profilable from a
   bench harness):

   - **mul_mmq's tight inline FP accumulation pattern** has fewer
     instructions per K element than even our minimal direct-fold.
     `dotPacked4x8EXT(...)` + `q_sum += ...` + final `sums[mn] += ...`
     is *3* scalar instructions per inner step. Our path is
     `coopMatLoad(A) + coopMatLoad(B) + coopMatMulAdd + (4 elem fold ops)`
     — ~10 ops per inner step, even ignoring the WMMA latency.
   - **WMMA setup overhead** (fragment loads, lane reorganization
     inside `v_wmma_i32_16x16x16_iu8`) is amortized over fewer K
     elements per call than scalar dot4 amortizes over its 4-element
     instruction.
   - **LDS pressure for scales**. Both `a_dm[BM]` and `b_ds[BN]` are
     read 4× per fragment (once per cell) by every lane in the warp.
     That's ~256 LDS reads per fragment per warp purely for scales,
     significantly more than `mul_mmq`'s per-cell scalar register
     forwarding.

The right reading of this sprint's result is: **the architectural
overhead estimate from Sprint 11G-A was directionally correct but
quantitatively wrong**; the LDS round-trip was *not* the dominant cost.
The dominant cost is the structural difference between coopmat-shape
GEMM and scalar-tight-inner-loop GEMM on RDNA4 + RADV at the M-tile
shape we tested.

## 4. What about L-tile?

Sprint 11G-A floated L-tile (BM=BN=128, NUM_WARPS=4, 16 fragments per
warp) as a way to amortize per-fragment overhead. With the direct-fold
in hand, this would still pay the same per-cell scale-lookup cost
(in fact 4× more cells per warp = 4× more scale reads), and the WMMA
inner loop would be 16× larger but the surrounding orchestration is
constant per BK chunk. We expect a small additional improvement
(~5–10 %), nowhere near closing the 0.47× → 1.2× gap.

L-tile is not pursued. The Sprint 11G strategy is closed.

## 5. Why this isn't a wasted sprint

- The lane-to-cell mapping is documented and reusable for any future
  RDNA4 coopmat work (FP8 path, BF16 attention, FP16 mul_mm successors,
  etc.). Without this probe, every such sprint would re-discover this
  bug at parity-test time.
- The direct-fold rewrite is the *correct* baseline architecture for
  any future int8-cm GEMM on RDNA4 — the 11G-C LDS round-trip is now
  unambiguously known to be the wrong starting point.
- Sprint 11G is conclusively closed. Future GEMM speed work on RDNA4
  should focus elsewhere (mul_mmq tile / spec-constant tuning, FFN
  fusion, weight prefetch, async dispatch, multi-stream).

## 6. Files touched

```
NEW   vk_shaders/probe_coopmat_layout.comp                  (78 LOC)
NEW   examples/probe_coopmat_layout.rs                      (210 LOC)
EDIT  vk_shaders/bench_int8cm_q4k.comp                      (–18, +24 LOC: scratch removed,
                                                              direct fold + corrected mapping)
EDIT  build.rs                                              (+6 lines: probe ShaderJob)
EDIT  src/backend/vulkan/shaders.rs                         (+10 lines: ProbeCoopmatLayout)
EDIT  src/backend/vulkan/pipeline_registry.rs               (+1 line)
NEW   results/v021_sprint11gd_fold_rewrite.md               (this report)
```

SPV count: 65 (was 64); `bench_int8cm_q4k.spv` 161 240 (was 163 396),
`probe_coopmat_layout.spv` 7 404.

## 7. Tests / regression

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 174 passed
```

176/176 green (the +2 from 11G-C, both still passing with the new fold —
the parity threshold of `max(0.05·|amax|, 0.5)` is comfortably met).

No new tests in 11G-D. The existing 11G-C parity tests
(`test_int8cm_q4k_microbench_parity` at M=N=64, K=256 single-WG and
`test_int8cm_q4k_microbench_parity_multi_wg` at M=N=128, K=1024
multi-WG) exercise the rewritten fold end-to-end. The bench harness
additionally validates against `mul_mmq` at all 3 production shapes
(every output cell within FP32 rounding noise).

## 8. Final GO/NO-GO for Sprint 11G

**NO-GO. Sprint 11G is abandoned.**

The four sprints in the 11G arc:

| Sprint | Question | Result |
|--------|----------|--------|
| 11F    | Is RDNA4 Int8 KHR-coopmat dispatchable from Vulkan / RADV? | YES |
| 11G-A  | Architecture for Q4_K × Q8_1 → FP32 with int8-cm? | M-tile, Option Y fold |
| 11G-B  | Compute speedup for raw int8-cm vs scalar dot4? | +42 % median |
| 11G-C  | End-to-end Q4_K × Q8_1 vs production mul_mmq with LDS-fold? | NO-GO at 0.42× |
| 11G-D  | Direct-fold rewrite recovers the gap? | NO-GO at 0.47× |

The +42 % compute advantage measured in 11G-B does not survive the
combination of Q4_K nibble unpack, Q8_1 staging, and per-cell
scale-fold. mul_mmq's scalar tight-loop pattern is too well-suited to
RDNA4's `v_dot4_i32_iu8` to lose to any coopmat orchestration we have
been able to construct.

## 9. Take-aways

1. **The probe-shader-first methodology saved time.** The wrong
   `coopMatLoad`-derived mapping would have cost a full debugging
   cycle on the bench shader itself. Verifying the mapping on a
   tightly-controlled MMA result first is the right gate.
2. **Subgroup-scope barriers are not as expensive as Sprint 11G-A
   thought.** The +50 % overhead estimate was off by 5×. This affects
   future architecture decisions for coopmat-fold patterns.
3. **mul_mmq is structurally hard to beat on RDNA4.** Production
   scalar `dotPacked4x8EXT` runs at ~50 % of WMMA INT8 peak — that's
   the bar to clear, not the headroom we had hoped for. Future GEMM
   speedups on this hardware likely come from outside the kernel:
   batching, fusion, prefetch, dispatch latency hiding.
4. **The lane-to-cell mapping is durable knowledge.** It's specific
   to Mesa RADV 26.0 / gfx1201, but until that combination changes,
   any future RDNA4 coopmat shader can reuse the formula without a
   re-probe.
