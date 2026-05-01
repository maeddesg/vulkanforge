# VulkanForge v0.2.1 Sprint 11G-A — Int8-coopmat GEMM Architektur

**Date:** 2026-04-30
**Branch:** main (HEAD = fa024c1, post-Sprint 11F push)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Pure architecture / design — no shader code, no benchmark

## TL;DR

Designable. The Int8-coopmat KHR-coopmat GEMM is a clean fit for Q4_K×Q8_1 prefill on
RDNA4: `block_q4_K` super-blocks of 256 elements decompose into 8 sub-blocks of 32,
which exactly matches `BK=32` from the existing scalar `mul_mmq` pipeline AND maps
to two coopmat 16×16×16 fragment loads per K-tile. Q4_K nibbles unpack to int8
[0..15] in LDS; Q8_1 `qs[32]` already is int8 and stages directly to LDS; per-cell
scales (`d_a[i] × d_b[j]`) and bias (`dmin_a[i] × s_b[j]`) fold against an int32
coopmat accumulator that is reset per BK chunk and added to a per-thread FP32
register accumulator that lives across the whole K-loop.

LDS footprint estimate for the L tile (BM=BN=128, BK=32) is **~13–14 KB**, matching
scalar `mul_mmq`'s L tile (4 WG/CU). That is the key win: same occupancy as the
scalar-mmq baseline, but the inner-loop arithmetic moves from `dotPacked4x8EXT` on
the VALU to RDNA4's WMMA AI Accelerators. Analytical compute speedup: **~2× over
scalar mul_mmq** for the inner-loop dot work, end-to-end **1.3–1.7×** GEMM speedup
plausible after BW and dequant overhead are accounted for.

This document is the blueprint for Sprint 11G-B (correctness micro-bench against
scalar mmq) and 11G-D (full shader + parity).

## 1. Q4_K block layout and nibble unpack

### 1.1 Canonical struct (ggml-common.h, llama.cpp upstream)

```c
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ggml_half d;               // FP16 super-block scale
    ggml_half dmin;            // FP16 super-block min
    uint8_t scales[12];        // 8 sub-block (scale,min) packed as 6-bit each
    uint8_t qs[QK_K/2];        // 256 4-bit nibbles → 128 bytes
} block_q4_K;
// sizeof = 2 + 2 + 12 + 128 = 144 bytes for 256 weights = 4.5 bpw
```

256 weights split into **8 sub-blocks of 32 elements**. Each sub-block has its own
`(sub_scale, sub_min)` pair stored as 6-bit values inside `scales[12]`. `dequant`
formula:

```
w[s][i] = d × sub_scale[s] × q[s][i]  -  dmin × sub_min[s]      (i = 0..31)
```

Per sub-block this collapses to a vec2 `dm = (d × sub_scale, dmin × sub_min)`,
which is exactly what `mul_mmq_funcs.glsl` already pre-computes at LDS load
(line 326–337):

```glsl
buf_a[buf_ib].dm = FLOAT_TYPEV2(vec2(data_a_packed32[ib_k].dm) * vec2(scale_dm));
```

### 1.2 How `mul_mmq` consumes Q4_K nibbles (existing scalar path)

`mul_mmq_shmem_types.glsl` line 60–65, Q4_K cache entry covers exactly **one
sub-block (32 elements)**:

```glsl
struct block_a_cache {
    uint32_t qs[4];           // 32 nibbles, packed 8 nibbles per uint32
    FLOAT_TYPEV2 dm;          // per sub-block (d×sub_scale, dmin×sub_min)
};
```

Pack pattern (`mul_mmq_funcs.glsl` line 313–317): `vals0 = lower nibbles of 4
bytes`, `vals1 = upper nibbles`, then `qs = vals0 | (vals1 << 4)` — i.e., 4 low
nibbles in low 4 bits of each byte and 4 high nibbles in upper 4 bits of each
byte. That packing is optimised for `dotPacked4x8EXT(qs_a, qs_b)` consumption at
mmq_dot_product time, which is **not** what coopmat wants — coopmat needs raw
contiguous int8 values in row-major (or col-major) LDS layout.

### 1.3 Nibble-unpack for Int8 coopmat

The unpack runs **at LDS-load time**, exactly where `block_a_to_shmem` runs in
the scalar path. New helper (Sprint 11G-D):

```
// Per Q4_K sub-block (32 elements):
for (i = 0..15):
    byte = data_a[ib].qs[sub_offset + i];
    a_lds[row × BK + col + 2i + 0] = int8_t(byte & 0x0F);   // nibble 2i
    a_lds[row × BK + col + 2i + 1] = int8_t(byte >> 4);     // nibble 2i+1

// Plus the dm vec2 (already computed identically to scalar path):
a_dm[row][k_chunk] = FLOAT_TYPEV2(d × sub_scale, dmin × sub_min);
```

Q4_K nibbles are unsigned `[0..15]` — they fit cleanly in signed int8 `[-128..127]`
with no sign trickery. RDNA4 KHR-coopmat entry 14 (`I8 × I8 → I32, no saturating`)
treats both operands as signed. That is fine: `signed_dot([0..15], [-128..127])`
gives the same numeric result as the unsigned-on-A interpretation used by the
existing scalar mmq.

The total scratch in LDS is exactly `BM × BK × 1B`. No `qs[4] uint32` packing,
no `dotPacked4x8EXT` shuffle layout. The trade-off: we lose the existing
`block_a_to_shmem` four-bytes-at-a-time vectorised load. Mitigation: the unpack
is a single `bitfieldExtract`-class instruction per nibble, dispatched
cooperatively across the WG.

## 2. Q8_1 block layout and B-fragment staging

### 2.1 Canonical struct

```c
#define QK8_1 32
typedef struct {
    ggml_half d;       // FP16 scale
    ggml_half s;       // FP16 sum: d × Σ(qs[i])
    int8_t qs[QK8_1];  // 32 int8 quantised activations
} block_q8_1;
// sizeof = 2 + 2 + 32 = 36 bytes per 32 elements
```

`qs[]` is **already int8** — no unpack required. `s` is the pre-computed sum
`d × Σ(qs[i])` which folds the Q4_K min-correction term `dmin × sub_min × s_b`
without re-summing at consume time.

### 2.2 LDS staging for B (Q8_1 → int8)

Direct `coopMatLoad` from a Q8_1 buffer is **not viable** because Q8_1 stride
is 36 bytes per 32 elements (interleaved `d`, `s`, `qs[]`), while `coopMatLoad`
expects a contiguous int8 array with stride matching the matrix dimension. The
clean approach is the same LDS staging the scalar mmq path already uses:

```
shared int8_t b_lds[BN × BK];            // raw int8 activations
shared FLOAT_TYPEV2 b_ds[BN × (BK/32)];  // (d, s) per Q8_1 block per col
```

Cooperative load pattern: each thread reads `qs[k]` for its assigned `(col, k)`
slot, copies the byte into `b_lds`. Block scales `(d, s)` are loaded once per
column per K-chunk by a designated thread.

Same shape (`buf_b[BN × BK_STEP]` with `block_b_cache { int32_t qs[8];
FLOAT_TYPEV2 ds; }`) already exists in `mul_mmq_shmem_types.glsl` — the int8
variant just stores `qs[]` as a raw byte array instead of `int32_t[8]`.

## 3. Scale-folding (int32 accumulator → FP32 result)

### 3.1 The actual contribution per BK chunk

For one BK=32 chunk (= one Q4_K sub-block on the A side, one Q8_1 block on the
B side), the contribution to `result[i][j]` is:

```
contrib[i][j] = d_a[i] × d_b[j] × Σ_{k=0..31}(nibble[i][k] × qs8[k][j])
              - dmin_a[i] × s_b[j]
```

The **inner sum** is exactly what `coopMatMulAdd` over `BK/TK = 32/16 = 2`
fragment loads computes as int32. The outer scale `d_a[i] × d_b[j]` is a
**rank-1 product** that varies per cell `(i,j)`, and the bias
`dmin_a[i] × s_b[j]` is also rank-1.

### 3.2 Why option X (fold once at end) is wrong

Q4_K **sub-block** scales differ between the 8 sub-blocks within one
super-block, AND between rows of A (different super-blocks per row). If the
int32 accumulator is held across multiple BK chunks, contributions from
different sub-blocks get summed **before** their differing `d_a[i] × sub_scale`
factors are applied — silent corruption.

`BK = 32 = 1 sub-block`, which is the **largest** value of BK that keeps scales
constant within an int32 accumulator. Choosing `BK > 32` (e.g., BK=64) does NOT
help and actively breaks correctness without per-thread per-sub-block fold.

### 3.3 Why option Z (pre-bake scales into int8) is impossible

`int8_t` range is `[-128, 127]`. Pre-multiplying nibble × sub_scale (up to
`15 × 63 = 945`) overflows int8. Pre-baking scales is unviable.

### 3.4 Decision: option Y (fold per BK chunk into per-thread FP32 register)

Per BK=32 chunk:

1. Inner coopmat loop runs `BK / TK = 2` `coopMatMulAdd` steps that accumulate
   into one int32 fragment `cm_int32[cm_row, cm_col]`, initialised to zero
   at the start of the chunk.
2. After both fragment-loads complete:
   ```
   coopMatStore(cm_int32, scratch_lds_per_warp, ...)
   subgroup_memory_barrier()
   ```
3. Per-thread fold: each lane reads its assigned cells from
   `scratch_lds_per_warp`, multiplies by `d_a[i] × d_b[j]`, subtracts
   `dmin_a[i] × s_b[j]`, and adds the result to the per-thread FP32
   accumulator `fp32_sums[m, n]` (already per-cell-private since each lane
   owns a fixed slice of each fragment via the WMMA layout).
4. Reset `cm_int32 = 0` for the next chunk.

This is a direct adaptation of `mul_mmq.comp`'s pattern (line 207–272): the
FP32 register accumulator persists across the K-loop, and the per-chunk
scale-fold produces the FP contribution exactly as `mmq_dot_product` does
today.

The per-chunk `coopMatStore + subgroup barrier` cost is the price of correct
sub-block scale handling. It uses only **subgroup-scope** barriers, which on
RDNA4 are single-cycle relative to `OpControlBarrier`-class WG-wide barriers.

## 4. LDS footprint and occupancy

### 4.1 L tile (BM=128, BN=128, BK=32, NUM_WARPS=4)

| Buffer | Size |
|---|---|
| `a_lds` (int8 weights) | 128 × 32 × 1B = **4 KB** |
| `a_dm` (FP16 vec2 per row, 1 chunk) | 128 × 4B = 0.5 KB |
| `b_lds` (int8 activations) | 128 × 32 × 1B = **4 KB** |
| `b_ds` (FP16 vec2 per col, 1 chunk) | 128 × 4B = 0.5 KB |
| Coopmat scratch: 1 fragment × NUM_WARPS | 16 × 16 × 4B × 4 = **4 KB** |
| **Total** | **~13 KB** |

(BK_STEP doubling, if added later for load/compute overlap, doubles A+B → +8 KB →
not worth it without measured win.)

64 KB LDS / 13 KB = 4.92 → **4 WG/CU**.

This **matches the scalar mul_mmq L tile** (~14 KB → 4 WG/CU) measured in
Sprint 11C — i.e., we do **not** lose occupancy. Compare to FP16 coopmat mul_mm
(Sprint 11E, ~24 KB → 2 WG/CU) which lost the race largely *because* of
occupancy regression.

### 4.2 M tile (BM=64, BN=64, BK=32, NUM_WARPS=4)

| Buffer | Size |
|---|---|
| `a_lds` | 64 × 32 = 2 KB |
| `a_dm` | 64 × 4B = 0.25 KB |
| `b_lds` | 64 × 32 = 2 KB |
| `b_ds` | 64 × 4B = 0.25 KB |
| Scratch (4 warps × 1 fragment) | 4 KB |
| **Total** | **~8.5 KB** |

64 / 8.5 ≈ **7 WG/CU**. Useful for `m=64..128` shapes (e.g., FFN gate/up of
small batches).

### 4.3 S tile (BM=32, BN=32, BK=32, NUM_WARPS=1)

| Buffer | Size |
|---|---|
| `a_lds` | 32 × 32 = 1 KB |
| `a_dm` | 32 × 4B = 0.13 KB |
| `b_lds` | 32 × 32 = 1 KB |
| `b_ds` | 32 × 4B = 0.13 KB |
| Scratch (1 warp × 1 fragment) | 1 KB |
| **Total** | **~3.3 KB** |

64 / 3.3 ≈ **19 WG/CU**. Targets `m ≤ 32` shapes (decode-stage GEMV-like).
NUM_WARPS=1 gives minimum WG size and maximum dispatch parallelism.

## 5. Spec constants and BLOCK_SIZE

The Int8-coopmat shader will reuse the same spec-constant ID layout as
`mul_mm.comp` COOPMAT path so that the registry plumbing (Sprint 11C/11E)
extends naturally. Constants:

| ID | Name | L | M | S | Notes |
|----|------|---|---|---|-------|
| 0 | `BLOCK_SIZE` | 256 | 256 | 64 | NUM_WARPS × WARP |
| 1 | `BM` | 128 | 64 | 32 | A tile rows |
| 2 | `BN` | 128 | 64 | 32 | B tile cols |
| 3 | `BK` | 32 | 32 | 32 | **Must = 32** (Q4_K sub-block) |
| 4 | `WM` | 64 | 32 | 32 | Per-warp M sub-tile |
| 5 | `WN` | 64 | 32 | 32 | Per-warp N sub-tile |
| 6 | `WMITER` | 1 | 1 | 1 | Coopmat path: WSUBM = WM/WMITER |
| 7 | `TM` | 16 | 16 | 16 | Coopmat fragment M (RDNA4 fixed) |
| 8 | `TN` | 16 | 16 | 16 | Coopmat fragment N (RDNA4 fixed) |
| 9 | `TK` | 16 | 16 | 16 | Coopmat fragment K (RDNA4 fixed) |
| 10 | `WARP` | 64 | 64 | 64 | Wave64 on RDNA4 |

Derived per warp:

```
NUM_WARPS    = BLOCK_SIZE / WARP                  # 4 / 4 / 1
cms_per_row  = WM / TM                            # 4 / 2 / 2
cms_per_col  = WN / TN                            # 4 / 2 / 2
fragments_per_warp = cms_per_row × cms_per_col    # 16 / 4 / 4
output_cells_per_wg = BM × BN                     # 16384 / 4096 / 1024
```

Constraint check (Sprint 7 silent-corruption rule):
`(BM/WM) × (BN/WN) = NUM_WARPS` →
- L: (128/64) × (128/64) = 2 × 2 = 4 ✓
- M: (64/32) × (64/32) = 2 × 2 = 4 ✓
- S: (32/32) × (32/32) = 1 × 1 = 1 ✓

## 6. Inner-loop pseudocode

```text
shared int8_t       a_lds [BM][BK];
shared int8_t       b_lds [BN][BK];
shared FLOAT_TYPEV2 a_dm  [BM];                          // (d×sub_scale, dmin×sub_min) per row, current chunk
shared FLOAT_TYPEV2 b_ds  [BN];                          // (d, s) per col, current chunk
shared int32_t      scratch[NUM_WARPS][TM × TN];         // per-warp coopmat scratch

// Per-thread FP32 accumulator (one element per cell this lane owns):
float fp32_acc[fragments_per_warp][cells_per_lane] = 0;

for (uint k_chunk = 0; k_chunk < K; k_chunk += BK) {
    // ---- 1. Cooperative LDS load ----
    cooperative_unpack_q4k_to_int8(a_lds, a_dm, weights, k_chunk);   // 4-bit nibble → int8
    cooperative_load_q8_1_qs(b_lds, b_ds, activations, k_chunk);     // direct int8 copy
    barrier();                                                         // WG-wide

    // ---- 2. Reset int32 fragments for this chunk ----
    coopmat<int32_t, 16, 16, MatrixUseAccumulator>
        cm_int32[cms_per_row × cms_per_col] = { 0 };

    // ---- 3. Coopmat MMA inner loop (BK/TK steps) ----
    [[unroll]] for (uint i = 0; i < BK; i += TK) {
        [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
            coopmat<int8_t, 16, 16, MatrixUseA> A_frag;
            coopMatLoad(A_frag, a_lds,
                        (warp_r × WM + cm_row × TM) × BK + i,
                        BK, gl_CooperativeMatrixLayoutRowMajor);

            [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col++) {
                coopmat<int8_t, 16, 16, MatrixUseB> B_frag;
                coopMatLoad(B_frag, b_lds,
                            (warp_c × WN + cm_col × TN) × BK + i,
                            BK, gl_CooperativeMatrixLayoutColumnMajor);

                cm_int32[cm_col × cms_per_row + cm_row] =
                    coopMatMulAdd(A_frag, B_frag,
                                  cm_int32[cm_col × cms_per_row + cm_row]);
            }
        }
    }

    // ---- 4. Per-fragment scale-fold into FP32 register ----
    [[unroll]] for (uint f = 0; f < fragments_per_warp; f++) {
        coopMatStore(cm_int32[f], scratch[warp_i], 0,
                     TM, gl_CooperativeMatrixLayoutColumnMajor);
        subgroupMemoryBarrierShared();

        [[unroll]] for (uint cell = 0; cell < cells_per_lane; cell++) {
            uvec2 (i, j)   = lane_to_cell(lane, cell, f);   // WMMA layout
            int32_t  raw   = scratch[warp_i][cell_index(lane, cell)];
            float    scale = float(a_dm[row(i)].x) * float(b_ds[col(j)].x);
            float    bias  = float(a_dm[row(i)].y) * float(b_ds[col(j)].y);
            fp32_acc[f][cell] += scale * float(raw) - bias;
        }
        subgroupMemoryBarrierShared();
    }

    barrier();                                                         // before next chunk
}

// ---- 5. Tail: per-thread FP32 sums → data_d (same as mul_mmq tail) ----
[[unroll]] for (uint f = 0; f < fragments_per_warp; f++) {
    [[unroll]] for (uint cell = 0; cell < cells_per_lane; cell++) {
        uvec2 (i, j) = lane_to_cell(lane, cell, f);
        if (i < M && j < N) {
            data_d[(dc + j) × stride_d + dr + i] = D_TYPE(fp32_acc[f][cell]);
        }
    }
}
```

## 7. Risks and mitigations

1. **WMMA fragment lane→cell layout**
   The per-thread fold (step 4) needs the explicit RDNA4 WMMA mapping from
   `(lane, fragment_index)` → `(row, col)` within the 16×16 fragment, to look up
   the right `a_dm[i]` and `b_ds[j]`.
   *Mitigation:* The Sprint 11F probe already validated that
   `coopMatStore(..., gl_CooperativeMatrixLayoutColumnMajor)` lays cells out
   contiguously in column-major order in the destination buffer (256/256 cells
   correct in the smoke). With column-major store, lane `t` and `cell` index
   map to `(row, col) = (t, cell)` for `MatrixUseAccumulator` — drop into the
   simple flat scratch indexing in pseudocode line 4. Validate this empirically
   in Sprint 11G-B with a parity micro-test.

2. **Per-chunk `coopMatStore + subgroup barrier` overhead**
   16 fragment-stores × 128 K-chunks = 2048 subgroup-scope barriers per WG over a
   K=4096 GEMM.
   *Mitigation:* Subgroup-memory barriers on RDNA4 are <5 cycles each (vs >50
   for WG-wide). 2048 × 5 = 10K cycles ≈ 0.005 ms at 2.4 GHz — negligible
   relative to the >100 ms WG runtime. If it ever shows up as a hot-spot, the
   fold can be batched (store 4 fragments, 4 subgroup-barriers, 4 folds → one
   amortised barrier).

3. **int8 dot-product overflow in int32 accumulator**
   Worst case per fragment: `Σ_{k=0..15} nibble[k] × qs8[k]` with
   `nibble ∈ [0,15]`, `qs8 ∈ [-128,127]` → max
   `|Σ| ≤ 16 × 15 × 128 = 30 720`. Well within `int32` range
   (`±2.1×10^9`). No saturation flag required; entry 14 (no-saturation) is
   safe.

4. **dmin × s bias term**
   The Q4_K min-correction `dmin_a[i] × s_b[j]` is **not** part of the coopmat
   inner sum — it's a separate per-cell rank-1 subtraction in the scale-fold
   step. `s_b[j]` is the pre-computed `d_b × Σ(qs8_b)` from the Q8_1 block;
   the per-thread fold reads it from `b_ds[j].y`. This matches scalar
   `mmq_dot_product` line 362 exactly.

5. **BK = 32 is non-negotiable**
   BK is the K-extent over which the int32 accumulator must hold constant
   scales. Q4_K sub-blocks are 32 elements. Q8_1 blocks are 32 elements. BK=32
   is the only value that respects both. Do not try BK=64 / BK=16 without a
   redesigned per-element scale fold inside the coopmat inner loop.

6. **Fragment-mixing gotchas (Sprint 10D lesson)**
   Mixing scalar nibble unpack (1 byte per nibble pair, scattered loads) with
   coopmat fragment loads (cooperative `coopMatLoad` requiring contiguous
   strided LDS) was the failure mode in Sprint 10D's first attempt at
   coopmat-Q4K. Mitigation: keep the unpack 100% in LDS-staging time (step 1),
   never mix `dotPacked4x8EXT` packing with `coopMatLoad` consumption in the
   same buffer.

## 8. Compute speedup analysis (vs scalar mul_mmq L)

Per BK=32 chunk, per warp, **scalar mul_mmq L tile**
(BM=128 BN=128 WM=64 WN=64 WMITER=1 TM=4 TN=2):

```
sums cells per warp     = WM × WN              = 4096
sums cells per lane     = 4096 / 64            = 64
dotPacked4x8EXT per lane= 8 × cells_per_lane   = 512
                          ↑
                          8 = BK/4 (one iqs handles 4 K elements)
total dot4 per warp     = 512 × 64 lanes       = 32768
MADs per dot4           = 4
MADs per warp per chunk = 32768 × 4            = 131 072
```

Throughput on RDNA3/4: `v_dot4_i32_iu8` is 4 MADs / 2 cycles per Wave64 lane →
**128 MADs/cycle per SIMD**.

Per BK=32 chunk, per warp, **Int8 coopmat L tile** (BM=128 BN=128 WM=64 WN=64
TM=TN=TK=16):

```
fragments per warp      = (WM/TM) × (WN/TN)    = 16
coopMatMulAdd per chunk = fragments × (BK/TK)  = 16 × 2 = 32
MACs per coopMatMulAdd  = 16 × 16 × 16         = 4096
MACs per warp per chunk = 32 × 4096            = 131 072    ← same as scalar!
```

Throughput on RDNA4 AI Accelerators: `v_wmma_i32_16x16x16_iu8` issues at twice
the int8 throughput of RDNA3 base WMMA → **~256 MACs/cycle per SIMD**.

**Inner-loop compute speedup ≈ 2×** vs scalar `dotPacked4x8EXT` for the same
total MAC count, because RDNA4's WMMA path consumes those MACs in half the
cycles.

End-to-end GEMM speedup is bounded by:
- LDS BW (same in both paths, BK=32, both load int8 staging),
- HBM BW (same — Q4_K weights, Q8_1 activations both read once),
- Scale-fold overhead (small, 2K subgroup barriers),
- Tail / dispatch overhead (constant).

**Realistic end-to-end target: 1.3–1.7×** speedup over current scalar mul_mmq+L
on `pp ≥ 256` shapes. Below that the M / S tile dispatch parallelism dominates
and scalar mmq wins on dispatch overhead.

## 9. Comparison summary (this sprint vs prior sprints)

| Sprint | Path | LDS | WG/CU | Result |
|--------|------|-----|-------|--------|
| 11C (current) | scalar mul_mmq L | ~14 KB | 4 | +4-5% pp≥512 (baseline for 11G) |
| 11E | FP16 coopmat mul_mm L | ~24 KB | 2 | NO-GO, -5 to -31% |
| 11G (this design) | Int8 coopmat L | ~13 KB | 4 | analytical: +30-70% |

The crucial observation: Int8 coopmat ties scalar mmq on LDS / occupancy
(unlike FP16-coopmat, which lost a lane of occupancy and bled performance) and
*then* doubles the inner-loop compute via WMMA AI accelerators.

## 10. Forward plan

| Sprint | Scope | Effort |
|--------|-------|--------|
| 11G-A (this) | Architecture document, no code | 1 day ✓ |
| 11G-B | Micro-bench shader: pure GEMM, A=int8, B=int8 → int32, no scale-fold, against scalar dotPacked4x8 baseline at L tile shape — validate the 2× compute claim | 2-3 days |
| 11G-C | Add Q4_K nibble unpack and Q8_1 staging in LDS; parity vs `mul_mmq_q4_k` for fixed shapes (qproj, FFN gate at pp=256, 512) | 3-4 days |
| 11G-D | Full pipeline: register, build.rs job, S/M/L threshold (mirror Sprint 11C), 4 new tests, integrate into forward.rs behind `VULKANFORGE_USE_INT8_COOPMAT` | 3-4 days |
| 11G-E | Default-flip if it wins ≥ +20% on pp-sweep; otherwise opt-in like Sprint 11E COOPMAT | 1 day |

Total: **2–3 weeks** if 11G-B confirms the speedup. Stop at 11G-B if not.

## 11. Gate for Sprint 11G-B

**GO** for Sprint 11G-B (micro-bench).

The architecture is internally consistent: BK=32 satisfies both Q4_K
sub-block alignment and Q8_1 block alignment, scale-fold is well-defined per
chunk, LDS budget matches scalar mmq's L-tile occupancy, and the analytical
compute speedup is 2× from RDNA4's WMMA AI accelerators.

The single biggest open assumption is the `coopMatStore` lane→cell mapping
(risk #1). Sprint 11G-B's micro-bench validates it directly: if a parity test
across a 16×16 int32 fragment with column-major store reads the right
per-cell value through `scratch[warp_i][cell_index(lane, cell)]`, the design
is locked.

## Files touched

```
NEW   results/v021_sprint11ga_int8cm_architecture.md   (this report)
```

No source code, no shader code, no build.rs change. 171/171 tests unchanged.
