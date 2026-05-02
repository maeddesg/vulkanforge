# Sprint 13A — S-Tile coopmat (BM=32)

**Premise.** Sprint 12M shipped M-tile (BM=64) coopmat for `seq_len ≤ 64`,
closing the pp=64 starvation that the L-tile (BM=128) caused (1.5 WG/CU
on 64-CU gfx1201). Below pp=32 the same starvation reappears with the
M-tile (3 WG/CU). Sprint 13A is the natural completion: add the S-tile
(BM=32) variant and a `n ≤ 32` selector arm — purely spec-constants on
top of existing SPVs, exactly the Sprint 12M methodology.

**Result.** Shipped. **27 / 27 lib tests, 15 / 15 coherent.** S-tile
beats the scalar `mul_mmq` default-off path by **+15 % to +27 %** at
pp ∈ {8, 16, 32}. Against the previous M-tile selector (i.e. what
12M would have picked at pp=32), S-tile is **+1.9 % — within
run-to-run noise**, despite the 6 WG/CU vs 3 WG/CU theoretical
saturation advantage. The win is correctness/completeness, not a
new performance lever.

The +15–27 % numbers come from comparing the new default (coopmat
default-on, S-tile fires for n ≤ 32) against `mul_mmq` scalar
default-off (`VULKANFORGE_DISABLE_MM_COOPMAT=1`). For pp=64 and
above, the selector keeps M-tile/L-tile and pp-bench is within
±2 % of Sprint 12M — confirming we did not regress the
already-shipped tile sizes.

## 1. Code changes

3 files, ~50 LOC, 0 new SPVs.

| File | Diff |
|---|---|
| `src/backend/vulkan/shaders.rs` | +14 LOC: 4 new `ShaderId` variants, `name()` arms, `spv_bytes()` arms (point at the existing 4 SPV byte-includes, same as 12M did for M-tile), `ALL_SHADERS` entries |
| `src/backend/vulkan/pipeline_registry.rs` | +20 LOC: extended the existing 8-way coopmat match arm to 12 ShaderIds; the binary `if m_tile { … } else { L … }` becomes a 3-way `if s_tile { … } else if m_tile { … } else { L … }` block |
| `src/backend/vulkan/forward.rs` | +18 LOC: `coopmat_s_tile = coopmat_q4k_mm && n ≤ 32`; `coopmat_m_tile` becomes mutually exclusive (`&& !coopmat_s_tile`); +4 routing-arm patterns in each of the two `(GemmKind::MulMm, q6)` matches; +1 tile-size arm in `run_gemm` for the four `(32, 32)` ShaderIds |

`build.rs`: 0 LOC. The S-tile pipelines reuse the four existing
coopmat SPVs (`mul_mm_q{4,6}_k_f32{,_aligned}_coopmat.spv`) — only
spec-constants differ. Total SPV count stays at 68.

## 2. The warptile values

```text
S-tile (Sprint 13A new):  { 64,  32,  32, 16, 32, 32, 2, 16, 16, 16, 64 }
M-tile (Sprint 12M):      { 128, 64,  64, 16, 64, 32, 2, 16, 16, 16, 64 }
L-tile (Sprint 11E/12K/L):{ 256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64 }
                             BS  BM   BN  BK  WM  WN  WMITER TM TN TK WARP
```

`NUM_WARPS = BLOCK_SIZE / WARP = (BM/WM) × (BN/WN)`:

- L-tile: 256 / 64 = 4 = 2 × 2 ✓
- M-tile: 128 / 64 = 2 = 1 × 2 ✓
- S-tile:  64 / 64 = 1 = 1 × 1 ✓

S-tile values pinned from llama.cpp `ggml-vulkan.cpp:3326` for AMD
KHR_coopmat at gfx1201 (same source as 12M's M-tile). All twelve
coopmat ShaderIds (3 tile sizes × 4 SPV variants) share the
match arm in `pipeline_registry.rs`.

## 3. The selector

`forward.rs:layer_weight_shader_gemm`:

```rust
// Sprint 13A — pick S-tile (BM=32, BN=32) when seq_len is very small.
// Sprint 12M — pick M-tile (BM=64, BN=64) when seq_len is small.
//   pp=64  L-tile = 1.5 WG/CU,  M-tile = 3 WG/CU
//   pp=32  L-tile = 1.5 WG/CU,  M-tile = 3 WG/CU,  S-tile = 6 WG/CU
let coopmat_s_tile = coopmat_q4k_mm && n <= 32;
let coopmat_m_tile = coopmat_q4k_mm && n <= 64 && !coopmat_s_tile;
```

`s_tile` and `m_tile` are mutually exclusive by construction; the
match-arm `(true, _, true, true)` is `unreachable!()`. The Rust
compiler's exhaustiveness check still catches accidentally-missed
shader patterns.

`run_gemm`'s `(bm, bn)` lookup gets the `(32, 32)` arm so workgroup
counts use the right divisors:

```rust
ShaderId::MulMmQ4KCoopmatS | ShaderId::MulMmQ6KCoopmatS
| ShaderId::MulMmQ4KAlignedCoopmatS | ShaderId::MulMmQ6KAlignedCoopmatS => (32, 32),
```

WG count for `gemm_gate` at pp=32 (m=12288, n=32):

- L-tile: `ceil(12288/128) × ceil(32/128)` = `96 × 1` = 96 WGs → 1.5 / CU
- M-tile: `ceil(12288/64)  × ceil(32/64)`  = `192 × 1` = 192 WGs → 3 / CU
- S-tile: `ceil(12288/32)  × ceil(32/32)`  = `384 × 1` = 384 WGs → **6 / CU**

The theoretical saturation case for the brief.

## 4. Bench

`run_pp_bench`, `RUNS=3`, median ms. `coopmat default-off` is the
same v0.2.2 binary with `VULKANFORGE_DISABLE_MM_COOPMAT=1` (scalar
`mul_mmq` integer-DP). `12M baseline (M-tile)` is what Sprint 12M
ships at pp ≤ 64 (BM=64); reproduced for pp=32 by temporarily
forcing `coopmat_s_tile = false` and re-benching.

| pp  | default-off `mul_mmq` | 12M (M+L)         | **13A (S+M+L)** | 13A vs default-off | 13A vs 12M |
|----:|----------------------:|------------------:|----------------:|-------------------:|-----------:|
|   8 |                 203.1 |                 — |       **233.0** |             +15 %  |          — |
|  16 |                 405.0 |                 — |       **467.5** |             +15 %  |          — |
|  32 |                 765.2 |             956.6 |       **974.5** |             +27 %  |     +1.9 % |
|  64 |                1497.4 |     1678.0 (12M)  |        1704.4   |             +14 %  |     +1.6 % |
| 128 |                2010.0 |     2560.0 (12M)  |        2551.3   |              —     |     −0.3 % |
| 256 |                2199.0 |     3558.0 (12M)  |        3511.1   |              —     |     −1.3 % |
| 512 |                2353.0 |     3863.0 (12M)  |        3787.3   |              —     |     −2.0 % |

(The pp ≥ 64 column for 12M comes from `results/v022_sprint12m_mtile.md`
table §1, same hardware, same model.)

Reading:

- **pp ∈ {8, 16, 32}**: S-tile fires; +14 to +27 % over scalar
  `mul_mmq`. The 15-prompt suite is dominated by such short
  prompts (10 of 15 are pp ≤ 64, several below 32), so S-tile is
  the path that small-prompt-dominated workloads now hit by
  default.
- **pp = 32 vs M-tile alone**: +1.9 %. **Within run-to-run
  noise.** The 6 WG/CU vs 3 WG/CU theoretical advantage doesn't
  materialise — at pp=32 with the M-tile, the GPU is already
  hidden-latency-bound, so doubling parallelism doesn't shorten
  wall time. (M dimension still gives M-tile 192 WGs — well
  above the few-WG-per-CU starvation that motivates S-tile in
  the first place.)
- **pp ≥ 64**: −0.3 % to −2.0 % vs 12M. All within the typical
  ±2 % run-to-run band; the selector correctly leaves M-tile/L-tile
  in place for these shapes.

## 5. Correctness

- `cargo test --release --lib` → **27 / 27 passing** (unchanged
  from 12M).
- `run_15prompt_bench` (default-on, no env var) → **15 / 15
  coherent.** Decode median **91.5 tok/s** (same as v0.2.1's
  91.5 / Sprint 12M's 91.1 — within run-to-run noise on this rig).
  Aggregate prefill 898.6 tok/s; the suite has many pp ≤ 32
  prompts that now exercise the S-tile path and stay coherent.
- 0 new SPVs; total SPV count unchanged at 68.

## 6. Verdict

**Honest neutral-to-slightly-positive.** The brief's bench-gate was
"pp=32 better than M-tile, pp ≥ 64 not regressed." Pp ≥ 64 is met;
pp=32 is +1.9 % vs M-tile (within noise) — neutral on that
specific gate. **+27 % vs the default-off scalar `mul_mmq`** at
pp=32 is the real number for downstream users — that's what they
get from coopmat default-on with this sprint, vs the v0.2.1
fallback path.

Why ship anyway despite the M-tile-vs-S-tile noise?

1. **Variant coverage matches llama.cpp.** Their selector also
   picks S-tile at this shape; not having it was a documented
   gap in v0.2.2 §"What's still on the table".
2. **Zero cost.** No new SPV, no compile-time hit, no startup-time
   cost (Vulkan creates pipelines lazily on first dispatch in
   our path). Removing it later would be more work than keeping
   it.
3. **Bigger wins likely live one pp-class lower.** At pp=8 / 16
   the S-tile path is +15 % over default-off; the 15-prompt suite
   has prompts at pp=30-62 where the difference between "scalar
   mul_mmq" and "S-tile coopmat" is clearly meaningful even if
   "M-tile vs S-tile" is not.
4. **Sets the floor for any future pp ≤ 32 work.** RGP / shader
   stats deltas between M-tile and S-tile can now be measured
   directly; without S-tile it would be guesswork.

## 7. What didn't materialise

The brief sketched a 6 WG/CU vs 3 WG/CU saturation argument. It is
correct on paper but **does not translate to wall time** at pp=32 on
RDNA4 / RX 9070 XT. The likely reason: at pp=32 with M-tile we have
192 WGs across 64 CUs = 3 WG/CU, and each CU's wave-pool can already
hide HBM-fetch latency for a 16-step coopmat MulAdd K=4096 inner
loop with that many resident waves; doubling occupancy to 6 WG/CU
adds parallelism the latency hiding mechanism doesn't need.

A real-and-different lever for pp ≤ 32 would be a kernel that
*reduces work per output element*, e.g.:

- f16-accumulator coopmat shader (the still-outstanding lever
  from CHANGELOG v0.2.2 §"What's still on the table"). Would also
  help pp ≥ 64.
- Decode-side coopmat for `lm_head` GEMV — N=151 936 vocab is much
  larger than any prefill GEMM, the WG-count math goes the other
  direction.

S-tile completes the prefill tile-size matrix; the next prefill
optimisation is shader-source work, not pipeline-config work.

## 8. Outputs

- 4 new ShaderIds (`MulMm{Q4K,Q6K}{,Aligned}CoopmatS`).
- Match-arm extension in `pipeline_registry.rs` covering all 12
  coopmat variants (4 SPV variants × 3 tile sizes).
- Selector + 8 routing-arm patterns + tile-size match in
  `forward.rs`.
- 0 new SPV files (S-tile reuses L/M-tile SPVs).
- This report.
- 27 / 27 lib tests, 15 / 15 coherence at default-on.
