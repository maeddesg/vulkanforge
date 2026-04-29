# VulkanForge v0.2.1 Sprint 11B-C — L-Tile mul_mmq Pipeline

**Date:** 2026-04-29
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV, kernel 7.0.1-cachyos
**Model:** Qwen3-8B-Q4_K_M
**Basis:** Sprint 11A (GEMM gap = pipeline coverage, not shader code)

## TL;DR

Added an `L`-tile `mul_mmq` pipeline (BM=BN=128) alongside the existing
`S`-tile (BM=BN=64), with a runtime selector that picks `L` when
`m > 128 && n > 256`. **Same SPV — only spec constants differ.**

| pp   | v0.2.0 (S only) | v0.2.1 (S + L) | Δ        |
|------|----------------:|---------------:|---------:|
|  64  |          1511  |          1515  |  +0.3 %  |
| 128  |          2001  |          2009  |  +0.4 %  |
| 256  |          2200  |          2197  |  −0.1 %  |
| **512** |       2255  |     **2346**  | **+4.0 %** |
| **1024**|       2204  |     **2306**  | **+4.6 %** |
| **2048**|       1997  |     **2091**  | **+4.7 %** |
| **4096**|       1659  |     **1732**  | **+4.4 %** |

Tests: **169/169 green** (167 baseline + 2 new L-tile parity tests).

The win is consistent +4–5 % at pp ≥ 512 and zero (within noise) below
the threshold. Smaller than the +118 % we'd want to close the gap to
llama.cpp Vulkan (now **0.54×** instead of **0.52×** at pp=512), but
this is the *first* of several sprint-11 deliverables. M-tile and
aligned variants are sprint 11D candidates.

## Method

### Spec-constant values from llama.cpp upstream

The L-tile values are pinned to llama.cpp's
`l_warptile_mmq_int_k` AMD-coopmat-override
(`ggml-vulkan.cpp:3368`, the path that fires for RDNA4 + KHR coopmat):

```cpp
l_warptile_mmq_int_k = { 256, 128, 128, 32, subgroup_size_16, 64, 1, 4, 2, 1, subgroup_size_16 };
//                       ^    ^    ^    ^   ^                   ^   ^  ^  ^  ^  ^
//                       BS   BM   BN   BK  WM                  WN  WMITER TM TN TK WARP
```

On gfx1201 with `subgroup_size = 64` (Wave64): `subgroup_size_16 =
max(64, 16) = 64`. Resolved values:

```
BLOCK_SIZE=256  BM=128  BN=128  BK=32  WM=64  WN=64  WMITER=1  TM=4  TN=2  WARP=64
```

### Constraint verification

The mul_mmq.comp shader computes `WNITER` from the spec constants
(line 138):

```glsl
const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
```

L tile: `WNITER = 64·64 / (64·4·2·1) = 8` ✓ integer.

Phase-7 silent-corruption rule (warp-tiles per WG = NUM_WARPS):
- `NUM_WARPS = BLOCK_SIZE / WARP = 256 / 64 = 4`
- `(BM/WM) · (BN/WN) = 2 · 2 = 4`  ✓

Both constraints hold. Bit-identical SPV (Sprint 11A confirmed)
plus integer-valid spec constants from upstream production = no
crash risk at the shader level.

### Threshold tuning

Initial implementation used llama.cpp's threshold
(`m > 64 && n > 64`). Result on RDNA4:

| pp  | initial L threshold | reason                                    |
|-----|--------------------:|-------------------------------------------|
| 128 |             −27.2 % | n=128 → groups_y=1 → 32 WGs total → CU-starved |
| 256 |              −4.4 % | n=256 → groups_y=2 → 64 WGs total → marginal    |
| 512 |              +4.1 % | n=512 → groups_y=4 → 128 WGs → fills CUs        |

RDNA4 has **64 CUs**. With BM=BN=128 dispatch, you need
`groups_x · groups_y ≥ 64` to fill. For the typical Qwen3 prefill
GEMMs `m ∈ {1024, 4096, 12288}` (so `groups_x ∈ {8, 32, 96}`), the
binding constraint is on `n`:

- At m=4096 (Q-proj) we need `groups_y ≥ 2` → `n ≥ 129` works
- At m=1024 (K/V-proj GQA) we need `groups_y ≥ 8` → `n ≥ 897`

Pragmatic threshold: **`m > 128 && n > 256`**. At `n=257`, even the
smallest prefill GEMM (m=1024) gets `groups_x=8 · groups_y=2 = 16`
WGs — still below ideal but no longer regressing.

Tightening would help m=1024 cases but tank pp=256 across-the-board.
Since pp=256 is a real workload (the 15-prompt suite hits it
regularly) and pp ≥ 512 already wins, the conservative threshold
is the right tradeoff.

`VULKANFORGE_DISABLE_L_TILE=1` opts out and was used to verify the
selector logic — opt-out gives 2250 / 2197 / 1991 at pp=512 / 1024 /
2048, matching v0.2.0's 2255 / 2204 / 1997 within noise.

## Implementation

### `shaders.rs`

Added two ShaderId variants. Both reuse the existing SPVs:

```rust
ShaderId::MulMmqQ4KL,   // → mul_mmq_q4_k_f32.spv
ShaderId::MulMmqQ6KL,   // → mul_mmq_q6_k_f32.spv
```

### `pipeline_registry.rs`

Extended the existing `MulMmqQ{4,6}K` arm to also handle the `L`
variants. The L-tile values are **hard-pinned** (no env-var override)
because the constraint chain (`WNITER` integer, NUM_WARPS coverage)
breaks if any of BM/BN/WM/WN/WMITER/TM/TN move independently. The
S-tile keeps the existing `VULKANFORGE_GEMM_*` env-var override
surface for the v0.1.x debugging workflow.

### `forward.rs`

`run_gemm` now derives `(BM, BN)` from the dispatched `ShaderId`:

```rust
let (bm, bn): (u32, u32) = match shader_id {
    ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL => (128, 128),
    _ => (64, 64),
};
let groups_x = (m + bm - 1) / bm;
let groups_y = (n + bn - 1) / bn;
```

Previously dispatch geometry was hard-coded to `(64, 64)`.

`layer_weight_shader_gemm` now takes `(m, n)` and picks the L-tile
when the threshold is met:

```rust
let prefer_l = m > 128 && n > 256
    && std::env::var("VULKANFORGE_DISABLE_L_TILE")
        .map(|s| s != "1").unwrap_or(true);
match (gemm_kind, q6) {
    (GemmKind::Mmq, true)  => if prefer_l { MulMmqQ6KL } else { MulMmqQ6K },
    (GemmKind::Mmq, false) => if prefer_l { MulMmqQ4KL } else { MulMmqQ4K },
    // mul_mm path unchanged
}
```

All 7 prefill GEMM call sites updated to pass `(m, n)` to the
selector.

## Tests

Two new parity tests in `tests/correctness.rs`:

```
test_gemm_q4k_full_tile_128x128_mul_mmq_l
    M=N=128 K=256, exactly one L-tile WG (groups=1×1×1).
    Phase-7-style coverage check: every warp tile must be
    initialised. Threshold: 5 % * |amax|.

test_gemm_q4k_l_tile_qwen3_qproj_parity
    M=N=512 K=1024, multi-WG L-tile dispatch (groups=4×4×1).
    Production-shape parity for the Qwen3 Q-projection at pp=512.
    Same threshold.
```

Both pass with `max_err << threshold`.

Full regression: **169 / 169 passing** (was 167, +2 new).

## Performance

### pp-sweep (RUNS=5 median, no Citrix)

| pp   | v0.2.0  | v0.2.1  | Δ        | llama.cpp ratio |
|------|--------:|--------:|---------:|----------------:|
|   64 |  1511   |  1515   |  +0.3 %  |    0.66×        |
|  128 |  2001   |  2009   |  +0.4 %  |    0.56×        |
|  256 |  2200   |  2197   |  −0.1 %  |    0.55×        |
|  512 |  2255   |  2346   |  +4.0 %  |  **0.54×**      |
| 1024 |  2204   |  2306   |  +4.6 %  |  **0.55×**      |
| 2048 |  1997   |  2091   |  +4.7 %  |  **0.55×**      |
| 4096 |  1659   |  1732   |  +4.4 %  |  **0.53×**      |

### 15-prompt benchmark (Qwen3-8B)

| Metric                    | v0.2.0 | v0.2.1 | Δ        |
|---------------------------|-------:|-------:|---------:|
| Median prefill tok/s      |  1068  |  1068  |    0.0 % |
| Median decode tok/s       |  90.5  |  90.7  |  +0.2 %  |
| Coherent prompts          | 15/15  | 15/15  |    —     |

Prefill is unchanged because every 15-prompt prompt has pp ≤ 198 < 256,
i.e. all fall under the L-tile threshold and route to S. This is
expected — the L-tile only helps long-context prefill, which is what
production large-prompt usage looks like (chat with attached docs,
RAG, long-system-prompt agents).

### Disabling L-tile (`VULKANFORGE_DISABLE_L_TILE=1`)

Verified that opt-out reproduces v0.2.0 numbers exactly:

| pp   | DISABLE_L_TILE=1 | v0.2.0 baseline |
|------|-----------------:|----------------:|
|  512 |             2250 |            2255 |
| 1024 |             2197 |            2204 |
| 2048 |             1991 |            1997 |

Differences are run-to-run noise (≤0.3 %). The selector is wired
correctly.

## Why not larger gains?

A few analytical notes for sprint 11D planning:

1. **Activation buffer Q8_1 quantize is a fixed cost.** At pp=512 we
   pay ~7 dispatches of `quantize_q8_1` per layer (one per GEMM).
   Each touches `seq_len × hidden = 512 · 4096 = 2 M` floats ≈ 8 MB
   read + 2.25 MB write. That's ~5 % of the total prefill bandwidth
   that the L-tile can't speed up.
2. **K/V projections are tiny GEMMs.** Qwen3-8B GQA: kv_dim = 1024,
   so K/V GEMMs at pp=512 are M=1024, N=512, K=4096. With L-tile:
   `groups = (8, 4, 1) = 32` WGs — half-empties the GPU. These
   GEMMs would prefer an `M-tile` (BM=BN=64 with longer BK) that
   keeps WG count high while still beating the S-tile's
   per-WG-K-prologue-cost.
3. **Down-projection has the same shape problem inverted.** At
   M=4096 N=512 K=12288 the K-dim is huge → BK=32 means 384 K-loop
   iterations per WG, dwarfing the per-WG setup cost. Increasing
   BK could help here but mul_mmq.comp pins BK=32 via `#define`.
4. **mul_mmq.comp is scalar.** llama.cpp's
   `matmul_q4_k_q8_1_int_dp.comp` (which we don't ship) uses
   `dot4_i8packed` (KHR_shader_integer_dot_product) — 4 INT8
   multiply-adds per scalar instruction. That's a separate SPV
   port, sprint 11E-class.

Closing the 0.55× gap fully needs all four of: M-tile + aligned
variants + integer-DP shader + maybe coopmat GEMM (sprint 11G's
`Eigenbau` candidate). Sprint 11C is the cheap +4 % from upstream
production tuning we hadn't shipped.

## Files touched

```
EDIT  src/backend/vulkan/shaders.rs
        + 4 lines: ShaderId::MulMmqQ4KL, MulMmqQ6KL + name + bytes + ALL_SHADERS

EDIT  src/backend/vulkan/pipeline_registry.rs
        +20 / -8 lines: extended MulMmqQ{4,6}K branch to handle L variants
        Hard-pinned values, comment block with constraint check.

EDIT  src/backend/vulkan/forward.rs
        + 4 lines: BM/BN derive from ShaderId in run_gemm
        + 18 lines: layer_weight_shader_gemm signature + selector + comment
        + 7 sites updated to pass (m, n)

EDIT  tests/correctness.rs
        +160 lines: two new parity tests
                    test_gemm_q4k_full_tile_128x128_mul_mmq_l
                    test_gemm_q4k_l_tile_qwen3_qproj_parity

NEW   results/v021_sprint11bc_tile_pipelines.md  (this file)
```

No SPV rebuild — `mul_mmq_q{4,6}_k_f32.spv` is unchanged, and the
build.rs-checked total stays at 4 047 972 bytes across 59 shaders.

## Forward look

Sprint 11D candidates, in order of expected ROI:

1. **M-tile (BM=BN=64, BK=32, BLOCK_SIZE=128, WM=64, WN=32, WMITER=2,
   TM=2, TN=2)** — for 128 < n ≤ 256 and the K/V GQA shapes. Catches
   the gap our threshold left open. Same `mul_mmq.spv`.
2. **Aligned variants** — `LOAD_VEC_B=4` skips the K-tail bounds
   check when seq_len % 4 == 0. We already ship aligned for `mul_mm`;
   adding it for mul_mmq is one new build.rs entry per tile.
3. **Down-projection BK tuning** — investigate whether widening BK
   to 64 or 128 for the K=12288 down GEMM is feasible (BK is `#define`
   not spec-const, would need a separate SPV).
4. **Integer dot-product shader** (`mul_mmq_int_dp.comp` port) —
   adds `VK_KHR_shader_integer_dot_product` requirement. ~2 days of
   shader port + extension wiring. The 4-element-per-instruction
   integer mul is what lets llama.cpp hit 4317 tok/s at pp=512.

Sprint 11G `Eigenbau` KHR coopmat GEMM remains the long-term
ceiling for the prefill GEMM gap — parked until 11D-E ROI is
measured.

## Take-aways

1. **Pre-check applies *recursively*.** Sprint 11A pre-checked the
   shader port and found it shipped. Sprint 11C pre-checked the
   pipeline-config gap and found a +4 % win sitting in upstream's
   AMD-override branch we hadn't picked up.
2. **Threshold tuning matters as much as tile size.** llama.cpp's
   threshold is wrong for our shipped shape (no M-tile fallback);
   blindly copying it cost 27 % at pp=128 in the first iteration.
3. **Production tuning > theoretical analysis.** llama.cpp's
   `l_warptile_mmq_int_k` AMD-coopmat-override has been hand-tuned
   on RDNA over multiple llama.cpp releases. The values came in,
   constraints held, parity tests passed, +4 % showed up. Same SPV
   we already had.
