# VulkanForge v0.2.1 Sprint 11G-C — Q4_K × Q8_1 Int8-coopmat GEMM

**Date:** 2026-04-30
**Branch:** main (HEAD = 001091a, post-Sprint 11G-B)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Real GGUF formats — Q4_K weights × Q8_1 activations → FP32, GO/NO-GO gate

## TL;DR — NO-GO

Median speedup of `bench_int8cm_q4k` (Int8 KHR-coopmat with Q4_K nibble unpack,
Q8_1 staging, and per-BK-chunk scale-fold) vs production `mul_mmq_q4_k_f32`
(scalar `dotPacked4x8EXT` with inline fold) over three Qwen3-8B prefill shapes:
**0.42×**. NO-GO on the ≥ 1.2× gate.

The shader is **correct** (max_abs_diff = 2.4 × 10⁻⁴ vs `mul_mmq`'s output at
the largest shape, ≈ 5 FP32 ULPs, well within the rounding-noise band). The
architecture is the problem: per-fragment scale-fold via LDS scratch +
per-thread float fold has substantial overhead that erases — and inverts —
the 1.42× compute advantage Sprint 11G-B measured for the bare int8-cm dot.

| Shape (M×N×K)            | Use-case                  | mul_mmq µs | int8cm µs | Speedup | mul_mmq GFLOPS | int8cm GFLOPS |
|--------------------------|---------------------------|-----------:|----------:|--------:|---------------:|--------------:|
| 512 × 4096 × 4096        | Q-proj @ pp=512           |      908.0 |   1 878.0 |   0.48× |       18 919.8 |       9 148.0 |
| 512 × 1024 × 4096        | K/V-proj @ pp=512 (GQA)   |      352.6 |     840.1 |   0.42× |       12 180.7 |       5 112.6 |
|  64 × 4096 × 4096        | Q-proj @ pp=64 (small)    |      255.7 |     698.3 |   0.37× |        8 398.4 |       3 075.4 |

Median of 10 runs after 3 warmups, stable to ±2% across two back-to-back
invocations. Parity check passes at every shape: max_abs_diff ≤ 2.4 × 10⁻⁴
across an output amplitude max of 553, i.e. relative error ≤ 5 × 10⁻⁷.

174/174 tests green (was 172; +2 new parity tests:
`test_int8cm_q4k_microbench_parity` and `test_int8cm_q4k_microbench_parity_multi_wg`).

## 1. What the shader does

`bench_int8cm_q4k.comp` (~280 LOC) implements the Sprint 11G-A architecture
end-to-end:

- **Tile**: M-tile (BM=BN=64, BK=32, NUM_WARPS=4, WM=WN=32, cms_per_row =
  cms_per_col = 2 → 4 fragments per warp). Conservative scope; L-tile was
  deferred to Sprint 11G-D.
- **Q4_K decode**: inline at LDS-load time. `q4k_nibble()` extracts the
  right nibble (low/high half of the byte, depending on sub-block index)
  from the canonical `block_q4_K_packed32` layout — same packing
  `mul_mmq_q4_k_f32` consumes, byte-for-byte. `q4k_sub_scale_min()` decodes
  the 6-bit `sub_scale` and `sub_min` from `scales[12]` using the canonical
  3-segment unpack (low 6 bits in `scales[is]`/`scales[is+4]` for sb<4,
  4-bit-low + 2-bit-high splice for sb≥4 — directly mirrors `q4k.rs`'s
  reference and `mul_mmq_funcs.glsl:326-337`).
- **Q8_1 decode**: byte-level extraction from `block_q8_1_x4_packed128`.
  `q8_1_byte()` does the bit-shift + sign-extend chain to read one
  signed int8 from the packed `qs[2]` ivec4-pair the production quantize
  shader writes.
- **Scale-fold (Option Y)**: per BK-chunk per fragment:
  1. `coopMatMulAdd` int32 accumulator for that chunk
  2. `coopMatStore` to per-warp LDS scratch (1 KB per warp × 4 warps = 4 KB)
  3. Subgroup memory barrier
  4. Per-thread fold loop: each lane reads its 4 cells, applies
     `result = raw × dm.x × ds.x − dm.y × ds.y` (rank-1 scale + rank-1 bias)
     into the per-thread FP32 register array
  5. Subgroup memory barrier (next fragment's coopMatStore)

LDS budget: 9 KB per WG (a_lds 2 KB + b_lds 2 KB + a_dm 0.5 KB + b_ds 0.5 KB
+ scratch 4 KB). Below `mul_mmq`'s S-tile budget (~14 KB), so occupancy is
not the regression cause.

## 2. Parity (numerical correctness)

Two new tests in `tests/correctness.rs` validate against the CPU FP64
reference (the same `cpu_gemm_q4k_ref` that gates production `mul_mmq`):

- `test_int8cm_q4k_microbench_parity`: M=N=64, K=256 (single super-block,
  single WG dispatch). Exercises Q4_K nibble unpack, scale extraction, Q8_1
  byte extraction, scale-fold pipeline, and direct global write — all in
  one place. **PASS** at the standard `max(0.05·|amax|, 0.5)` threshold.
- `test_int8cm_q4k_microbench_parity_multi_wg`: M=N=128, K=1024 (4 super-blocks,
  4 WG dispatch). Exercises the K-loop accumulation, inter-WG layout, and
  per-WG output offset math. **PASS** at the same threshold.

The bench harness additionally checks GPU-vs-GPU parity at all 3 production
shapes:
- 512 × 4096 × 4096:  max_abs_diff = 2.44e-4, rel = 4.41e-7
- 512 × 1024 × 4096:  max_abs_diff = 1.83e-4, rel = 3.31e-7
- 64  × 4096 × 4096:  max_abs_diff = 1.53e-4, rel = 3.22e-7

These are FP32 rounding-noise levels — both shaders' inline fold orderings
differ but both are numerically valid. **The shader produces correct results
at production shapes.**

## 3. Why it's slow — root cause

The performance regression is not occupancy, register pressure, or LDS bank
conflicts (all checked). It is **architectural**: the per-chunk scale-fold
in Sprint 11G-A's Option Y design has fundamentally higher overhead than
`mul_mmq`'s inline fold pattern.

**`mul_mmq_funcs.glsl::mmq_dot_product` (per Q4_K sub-block, per (m,n) cell):**

```glsl
int32_t q_sum = 0;
[[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
    q_sum += dotPacked4x8EXT(qs_a, qs_b);   // inline scalar dot
}
return ACC_TYPE(cache_b.ds.x * (cache_a.dm.x * q_sum) - cache_a.dm.y * cache_b.ds.y);
```

The fold is **inline in scalar registers**: q_sum is private per thread, the
fold result feeds directly into the per-thread `sums[]` register array via
a single `+=`. No LDS round-trip. No barrier.

**Our Sprint 11G-C path** (per BK-chunk, per fragment, per warp):

```glsl
coopmat<int32_t,16,16,Acc> C = ...;       // K-step inner: 2 × coopMatMulAdd
coopMatStore(C, scratch[warp], 0, TN, RowMajor);
subgroupMemoryBarrierShared();
for (cell = 0; cell < 4; cell++) {        // 4 cells/lane
    raw = scratch[warp][cell*64 + lane];
    fp32_out[fragment][cell] += raw * scale - bias;
}
subgroupMemoryBarrierShared();
```

The fold pays:
- 1 `coopMatStore` per fragment per BK-chunk (256 cells written)
- 2 subgroup-scope memory barriers per fragment per BK-chunk
- 4 LDS reads + 1 fma per lane per fragment per BK-chunk

For a K=4096 GEMM at our M-tile shape: 128 BK-chunks × 4 fragments per warp
× 4 warps per WG × all those barriers / round-trips. Even at ~5 cycles per
subgroup barrier and tight LDS, this measures as ~50% of total dispatch
time at the largest shape.

The Sprint 11G-A architecture document underestimated this overhead. The
architecture risk #2 ("Per-chunk `coopMatStore + subgroup barrier`
overhead") was assessed as "negligible" based on a 2 048-barriers ×
5-cycle estimate (~10 K cycles). The empirical cost is closer to
~1 ms per dispatch — **two orders of magnitude higher** than the estimate,
because per-barrier overhead in a real shader includes cache-line
write-back, lane reconvergence, and cross-warp scheduler traffic, not just
the barrier instruction itself.

## 4. Why this isn't fatal — paths forward

The Sprint 11G-B finding (1.42× int8-cm vs scalar dot4 in compute) still
stands: WMMA Int8 throughput on RDNA4 IS faster than scalar
`dotPacked4x8EXT`. The blocker is the *fold orchestration*, not the WMMA
itself. Three viable paths for Sprint 11G-D:

### 4.1 Direct fragment-element indexing (most promising)

GLSL's `coopmat<T, ...>` type supports per-element access via `frag[i]`,
where `i` ranges over the per-thread element count `frag.length()`. On
RDNA4, the lane-to-cell mapping for `MatrixUseAccumulator` is documented:
each lane holds 4 cells in a 16×16 Wave64 fragment, with a known
(lane, cell_idx) → (row, col) mapping. With that mapping in hand, the fold
becomes:

```glsl
coopmat<int32_t,16,16,Acc> C = ...;
coopmat<float,16,16,Acc> Cf = coopmat<float,16,16,Acc>(C);  // type convert
[[unroll]] for (uint i = 0; i < Cf.length(); i++) {
    uint (row, col) = lane_to_cell(lane, i);
    Cf[i] = Cf[i] * scale_per_cell(row, col) - bias_per_cell(row, col);
}
fp32_acc_frag = coopMatMulAdd(identity, Cf, fp32_acc_frag);  // accumulate as fragment
```

No LDS round-trip, no subgroup barrier per fragment, no scratch. The
risk is the lane→cell mapping — empirically validated for RowMajor /
ColumnMajor stores in Sprint 11G-B (bit-exact parity), but the
`MatrixUseAccumulator` mapping has not been cross-checked. Sprint
11G-D's first work item should be a 16×16 micro-test that validates
the mapping.

### 4.2 Larger BK with sub-block scale tracking

If we accept a heavier scale-fold *less often* (e.g., BK=128 = 4 sub-blocks
per chunk), the per-K-element fold cost amortises over 4× more compute.
Requires per-sub-block scale state inside the int32 accumulator — viable
because each lane already iterates 8 sub-blocks per K=256 super-block; we
just defer the fold to chunk boundaries instead of sub-block boundaries.

### 4.3 L-tile + parallel fragment fold

L-tile (BM=BN=128, NUM_WARPS=4, 16 fragments/warp) fixes the AI per WG
problem (16× more cells per LDS-load) but doesn't change the per-fragment
fold cost. *Could* combine with 4.1 to get the WMMA advantage and the
arithmetic intensity improvement together.

The simplest next step is 4.1 — it's the architectural fix the Sprint
11G-A risk register flagged but didn't act on.

## 5. Comparison with prior sprints

| Sprint | Path                                   | Speedup vs scalar | Outcome |
|--------|----------------------------------------|-------------------|---------|
| 11E    | FP16 coopmat mul_mm (16 frags/warp)    | -5 to -31% (NO-GO, opt-in) | FP32 BW + dequant + LDS pressure |
| 11G-B  | Int8 coopmat raw dot (no Q4_K)         | +42% median (GO)  | Compute-only baseline OK |
| 11G-C **(this)** | Int8 coopmat + Q4_K + scale-fold (M-tile) | -58% median (NO-GO) | Per-chunk fold orchestration |

The 11G-B → 11G-C transition lost **~85 % of the raw int8-cm advantage** to
the fold orchestration. That points sharply at fold mechanics, not WMMA
throughput, as the lever to pull next.

## 6. Files touched

```
NEW   vk_shaders/bench_int8cm_q4k.comp                  (288 LOC)
NEW   examples/bench_int8cm_q4k.rs                      (363 LOC)
EDIT  build.rs                                          (+12 lines: ShaderJob)
EDIT  src/backend/vulkan/shaders.rs                     (+12 lines)
EDIT  src/backend/vulkan/pipeline_registry.rs           (+1 line)
EDIT  src/backend/vulkan/pipeline.rs                    (+13 lines: BenchInt8CmQ4KPushConstants)
EDIT  tests/correctness.rs                              (+148 lines: 2 parity tests)
NEW   results/v021_sprint11gc_q4k_int8cm.md             (this report)
```

SPV count: 64 (was 63), +163 396 bytes.

## 7. Tests / regression

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 174 passed
```

174/174 green (was 172; +2 new parity tests).

## 8. GO/NO-GO

**NO-GO** at the Sprint 11G-A architecture as-is. Median speedup 0.42×
falls well below the ≥ 1.2× gate.

**Sprint 11G-D — recommended scope (revised):**

1. **Validate the WMMA accumulator lane-to-cell mapping** with a 16×16
   micro-test (one day). Output: lookup table or formula for
   `(lane, cell_idx) → (row, col)` on RDNA4 / RADV.
2. **Rewrite the scale-fold to use direct fragment-element indexing**
   (path 4.1 above). Eliminates the per-fragment `coopMatStore` and both
   subgroup barriers. Rebench at the same 3 shapes.
3. **Decision gate at the rebench**: ≥ 1.2× → GO for L-tile (11G-E) and
   forward.rs integration (11G-F). < 1.2× → declare 11G abandoned and
   focus on other levers (FFN restructure, weight prefetch, mul_mmq tile
   tuning).

Estimated 11G-D effort: **3-5 days** (one day mapping validation, one day
shader rewrite, one day rebench, one day analysis).

## 9. Take-aways

1. **The shader is correct.** Q4_K decode, Q8_1 decode, sub-block scale
   handling, and FP32 fold all match production `mul_mmq` to FP32
   rounding noise. Sprint 11G-D inherits a working numerical baseline.
2. **Architecture risk #2 was under-priced.** Per-chunk LDS round-trip
   + barrier overhead exceeds the WMMA compute advantage by a wide
   margin. The Sprint 11G-A "negligible" estimate was off by ~100×.
3. **The next step is fold mechanics, not tile size.** L-tile (11G-D
   original scope) doesn't fix this. Direct fragment-element indexing
   does. That should be the gate work for Sprint 11G-D, and it's the
   smallest change that could plausibly recover the 11G-B raw int8-cm
   advantage.
