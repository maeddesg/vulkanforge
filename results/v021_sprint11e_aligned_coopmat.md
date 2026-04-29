# VulkanForge v0.2.1 Sprint 11E — Aligned mul_mmq + COOPMAT mul_mm

**Date:** 2026-04-29
**Branch:** main (HEAD = 5a1214b, post-Sprint 11D)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV
**Mode:** Mixed — Part A pre-check hit (no work); Part B real implementation + bench

## TL;DR

**Part A — Aligned mul_mmq:** Pre-check (#5 in five sprints) hit.
`mul_mmq.comp` has `LOAD_VEC_B = 16` hardcoded (line 102) and there is
**no aligned variant in upstream llama.cpp** — `vulkan-shaders-gen.cpp:594`
builds mul_mmq with no `ALIGNED` define. Q8_1 block loads are inherently
16-byte aligned. Nothing to port. **No code change for Part A.**

**Part B — COOPMAT mul_mm:** Real work — built, parity-tested, benched.
**NO-GO as default.** Faster than scalar `mul_mm` but 5–31 % slower than
the existing `mul_mmq + L tile` baseline across all pp values. Kept as
opt-in (`VULKANFORGE_USE_MM_COOPMAT=1`) for diagnostic / parity
purposes.

The remaining 0.46× gap to llama.cpp Vulkan is **not closable on
RDNA4 by switching to KHR coopmat for GEMM** at the configurations we
tested. Integer-DP `mul_mmq` (already shipped since v0.1.x, see
Sprint 11D) wins because the BW saving from Q8_1 activations (vs FP32)
outweighs the FP16-coopmat compute speedup.

| pp   | mul_mmq + L (default) | mul_mm + COOPMAT | Δ COOPMAT vs mmq+L |
|------|----------------------:|-----------------:|-------------------:|
|  128 |                 2009  |            1394  |          −30.6 %   |
|  256 |                 2197  |            1921  |          −12.6 %   |
|  512 |                 2346  |            2222  |           −5.3 %   |
| 1024 |                 2306  |            2150  |           −6.8 %   |
| 2048 |                 2091  |            1959  |           −6.3 %   |

170/170 tests green (+1 new COOPMAT parity test). Commit only, no push.

## Part A — Aligned mul_mmq (pre-check hit)

### What the brief proposed

> mul_mm has aligned variants (MulMmQ4KAligned, MulMmQ6KAligned).
> mul_mmq has KEINE aligned Varianten. … LOAD_VEC_B=4: vektorisierter
> Load der B-Matrix (Q8_1 Aktivierungen).

### What's actually in the codebase

`vk_shaders/mul_mmq.comp:101-102`:
```glsl
#define LOAD_VEC_A (4 * QUANT_R_MMQ)
#define LOAD_VEC_B 16
```

**`LOAD_VEC_B = 16` is hardcoded.** Q8_1 activations are read in
16-byte chunks regardless of `seq_len % anything`. The "alignment"
that `mul_mm.comp`'s aligned variant exploits — vec4-of-floats vs
scalar-float B-loads — has no analogue in `mul_mmq.comp` because the
block-quant input format already enforces aligned 16-byte reads.

### What llama.cpp upstream does

`vulkan-shaders-gen.cpp:594`:
```cpp
string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp",
    merge_maps(merge_maps(base_dict, float_type_dict),
               {{data_a_key, "1"}, {"D_TYPE", "float"}}),
    fp16, coopmat, coopmat2, f16acc);
```

**No `ALIGNED`, no `LOAD_VEC_B`** in the merged dict. mul_mmq is
built once per quant type, not twice (`_aligned` vs unaligned). The
aligned-variant build pattern (lines 524-552 and 583-588) applies
only to `mul_mm.comp`.

### Decision

No code change. Documented as the 5th hypothesis-pre-check hit
(8b conditional barriers, 9c rms_norm_mul, 11A Q8_1+mul_mm port,
11D integer-DP, **11E-A aligned mul_mmq**).

## Part B — COOPMAT mul_mm

### Compile probe

Added a build.rs entry for `mul_mm.comp` with the COOPMAT define set:

```rust
ShaderJob {
    out_name: "mul_mm_q4_k_f32_coopmat.spv",
    entry_source: "mul_mm.comp",
    defines: &[
        ("DATA_A_Q4_K", "1"),
        ("A_TYPE", "block_q4_K"),
        ("A_TYPE_PACKED32", "block_q4_K_packed32"),
        ("B_TYPE", "float"),
        ("D_TYPE", "float"),
        ("FLOAT16", "1"),                  // gates GL_EXT_shader_explicit_arithmetic_types_float16
        ("FLOAT_TYPE", "float16_t"),
        ("FLOAT_TYPEV2", "f16vec2"),
        ("FLOAT_TYPEV4", "f16vec4"),
        ("ACC_TYPE", "float"),
        ("ACC_TYPEV2", "vec2"),
        ("LOAD_VEC_A", "4"),
        ("COOPMAT", "1"),
    ],
},
```

First attempt failed:
```
mul_mm.comp:128: error: 'qualifier: float16 types can only be in
  uniform block or buffer storage' : required extension not requested
```

Root cause: `mul_mm.comp:6-8` gates `GL_EXT_shader_explicit_arithmetic_types_float16`
on `#ifdef FLOAT16`. Without that define, FP16 *arithmetic* on
`float16_t` types isn't allowed (only FP16 *storage* is via
`GL_EXT_shader_16bit_storage`). Adding `("FLOAT16", "1")` fixed it.

SPIR-V verified:
```
$ spirv-dis mul_mm_q4_k_f32_coopmat.spv | head -10
   OpCapability Float16
   OpCapability VulkanMemoryModel
   OpCapability CooperativeMatrixKHR
   OpExtension "SPV_KHR_cooperative_matrix"
```

SPV size: 195 332 bytes (vs scalar mul_mm 194 992) — coopmat path is
inlined, total stays ~comparable.

### Pipeline configuration

Spec-constants pinned from llama.cpp's `warptile_mmq` AMD-coopmat-override
(`ggml-vulkan.cpp:3367`) at gfx1201:

```
{ BLOCK_SIZE=256, BM=128, BN=128, BK=32, WM=64, WN=64, WMITER=2,
  TM=16, TN=16, TK=16, WARP=64 }
```

`TM=TN=TK=16` because `coopmat_m = coopmat_n = coopmat_k = 16` on
RDNA4 (KHR coopmat 16x16x16 FP16 fragment).

`WNITER = WM·WN/(WARP·TM·TN·WMITER) = 64·64/(64·16·16·2) = 0.125`
— **non-integer**, but the COOPMAT path uses `cms_per_row = WM/TM
= 4` and `cms_per_col = WN/TN = 4` in the inner loop instead
(`mul_mm.comp:178-179`), so the 0-truncation in the unused scalar
fallback branch is benign.

`NUM_WARPS = BLOCK_SIZE/WARP = 4 = (BM/WM)·(BN/WN) = 2·2 ✓`
(Phase-7 silent-corruption coverage rule).

### Wiring

- `ShaderId::MulMmQ4KCoopmat` (Q4_K only — no Q6_K COOPMAT SPV
  shipped this sprint; Q6_K stays on scalar `mul_mm`).
- `Forward::mul_mm_coopmat_enabled` field, init from
  `VULKANFORGE_USE_MM_COOPMAT=1` env var.
- Setting `mul_mm_coopmat_enabled` forces `mul_mm_enabled=true` and
  `gemm_kind = GemmKind::MulMm` (skips the seq_len%4 aligned switch
  since we don't ship a COOPMAT-aligned SPV).
- `layer_weight_shader_gemm` extended with `coopmat_q4k_mm: bool`
  parameter; routes `(MulMm, Q4K)` to `MulMmQ4KCoopmat` when set.
- `run_gemm` BM/BN dispatch derivation extended:
  `MulMmQ4KCoopmat → (128, 128)`.

### Parity test

```rust
test_gemm_q4k_coopmat_mul_mm_parity:
    M=N=512, K=1024, dispatch (4, 4, 1) → 16 WGs of 128x128.
    Compares against cpu_gemm_q4k_ref.
    Threshold: 5% × |amax| (matches mul_mmq L-tile band — Q4_K weight
    round-off dominates either way).
```

Passed. No NaN/Inf. FP16 LDS introduces ~0.5–1 % drift vs scalar
FP32 GEMM, which sits well under the threshold.

### Bench (GO/NO-GO gate)

Direct A/B with the existing `mul_mmq + L` default at the typical
Qwen3 prefill shapes:

| pp   | mul_mmq + L (default) | mul_mm + COOPMAT | Δ        |
|------|----------------------:|-----------------:|---------:|
|  128 |                 2009  |            1394  | −30.6 %  |
|  256 |                 2197  |            1921  | −12.6 %  |
|  512 |                 2346  |            2222  |  −5.3 %  |
| 1024 |                 2306  |            2150  |  −6.8 %  |
| 2048 |                 2091  |            1959  |  −6.3 %  |

Numbers are RUNS=5 medians for COOPMAT and RUNS=5 medians from
Sprint 11C for the baseline.

**COOPMAT mul_mm is faster than scalar mul_mm** (the v0.1.3 baseline
that was ~45 % slower than mul_mmq) but **slower than mul_mmq + L**
across all pp values. The gap narrows at large pp (where compute
dominates) but never closes.

### Decision: NO-GO for default

`VULKANFORGE_USE_MM_COOPMAT=1` stays as an opt-in env var for
diagnostic / parity work, but the default routing remains
`mul_mmq + L` for prefill GEMMs.

## Why COOPMAT mul_mm loses to mul_mmq+L on RDNA4

This is the analytically interesting question. Three contributing
factors:

### 1. Activation bandwidth (3.5× more for mul_mm)

For Qwen3-8B Q-projection at pp=512: `M=4096, K=4096, N=512`.
- mul_mmq B-buffer: 4096·512·1.125 B (Q8_1) = 2.4 MB
- mul_mm B-buffer: 4096·512·4 B (FP32) = 8.4 MB

The COOPMAT path reads B as FP32 and converts to FP16 in
`load_b_to_shmem`. The DRAM read is still 4 B/element, so the
LLC/L1 traffic is 3.5× more than mul_mmq's Q8_1 path. At BW-bound
shapes (small M·N, large K) this dominates.

### 2. Q4_K weight dequant

mul_mm dequantizes Q4_K weights to FP16 in shared memory before
feeding the coopmat A fragment. mul_mmq operates directly on the
Q4_K bit-packed weights with `dotPacked4x8EXT` (4×INT8 dot product
per instruction). The "save the dequant pass" advantage of integer
GEMM is real on RDNA4.

### 3. FP16 LDS for both A and B

mul_mm + COOPMAT stages both A and B in FP16 LDS. Per Sprint 10D's
finding, LDS occupancy is sensitive: with 16 KB of FP16 LDS plus
the mul_mm `coopmat_stage` (TM·TN·NUM_WARPS·4 B = 16·16·4·4 = 4 KB)
plus padding, total LDS ≈ 24 KB per WG. RDNA4 LDS budget is 64 KB
per CU, so 2-3 WGs/CU. mul_mmq with cached Q-blocks (~16 KB total)
fits 4 WGs/CU.

Higher occupancy on mul_mmq means the K-loop's memory latency is
hidden better.

### What would close the gap?

Hypotheses for a sprint 11F+:

1. **Integer-DP coopmat (`coopMatMulAdd` with int8 fragments)** —
   does RDNA4 advertise the `CooperativeMatrixKHR` `Component0=Sint8`
   variant? If yes, we could keep mul_mmq's int8 path AND get the
   coopmat WMMA win. Compile probe needed.
2. **Per-shape variant selection** — at small pp where we regress
   the most, mul_mmq+L dispatches fewer WGs; mul_mmq+M (BM=BN=64
   M-tile) might be the right intermediate.
3. **Skip the mul_mm path entirely** and focus on tile-pipeline
   coverage for mul_mmq (the M-tile sprint 11D suggested).

## Files touched

```
EDIT  build.rs
        +25 lines: mul_mm_q4_k_f32_coopmat SPV build job

EDIT  src/backend/vulkan/shaders.rs
        + 8 lines: ShaderId::MulMmQ4KCoopmat + name + bytes + ALL_SHADERS

EDIT  src/backend/vulkan/pipeline_registry.rs
        +50 lines: COOPMAT pipeline branch (spec-constants from
        llama.cpp warptile_mmq AMD-override, hard-pinned)

EDIT  src/backend/vulkan/forward.rs
        +13 lines: mul_mm_coopmat_enabled field + env-var init
        + layer_weight_shader_gemm signature extended with coopmat_q4k_mm bool
        + 7 call sites updated to pass self.mul_mm_coopmat_enabled
        + run_gemm BM/BN dispatch handles MulMmQ4KCoopmat → (128, 128)
        + GemmKind selection forced to MulMm when coopmat enabled

EDIT  tests/correctness.rs
        +75 lines: test_gemm_q4k_coopmat_mul_mm_parity
                    M=N=512 K=1024, parity vs CPU reference

NEW   results/v021_sprint11e_aligned_coopmat.md (this file)
```

SPV bytes: +1 SPV (60 total, was 59), total bytes 4 243 304 (was
4 047 972, +195 332 ≈ size of new COOPMAT SPV).

## Tests / regression

```
test result: ok. 27 + 9 + 18 + 73 + 8 + 8 + 27 = 170 passed
```

170/170 green (was 169, +1 new test). All defaults unchanged — opt-in
`VULKANFORGE_USE_MM_COOPMAT=1` only enables the new path.

## Take-aways

1. **5th pre-check hit (Part A).** The "aligned mul_mmq" lever the
   brief proposed doesn't exist in upstream — Q8_1 block loads are
   inherently aligned. `grep` saved a half-day port.
2. **COOPMAT mul_mm works but loses on RDNA4.** Built clean, parity
   passes, but mul_mmq's integer-DP path with L-tile is faster. The
   activation-BW saving from Q8_1 dominates the FP16-coopmat compute
   speedup on this workload.
3. **The 0.46× gap to llama.cpp Vulkan is structural and small-prefill
   dominated.** At pp ≥ 512 we're at 0.54× and the mul_mm-COOPMAT
   experiment pushed that *down* to 0.51×. The honest finding is that
   on RDNA4, switching the prefill GEMM kernel architecture isn't
   the path forward — integer-DP mul_mmq is already near-optimal.
4. **Future direction:** sprint 11F should try **integer-DP coopmat**
   (if RDNA4 advertises Sint8 fragment support) before declaring
   the prefill GEMM gap unclosable.

## Forward look

- Sprint 11F: Probe `VkCooperativeMatrixPropertiesKHR` for Sint8
  component types on gfx1201. If supported: try the coopmat-int-dp
  path (combines mul_mmq's BW advantage with coopmat WMMA throughput).
- Sprint 11G: M-tile mul_mmq (BM=BN=64 with M-warptile spec-consts) for
  the 64 < n ≤ 256 zone our L-tile threshold currently leaves on S.
- Sprint 11I: Eigenbau KHR coopmat GEMM (parked from Sprint 11A) —
  defer until 11F's probe rules in or out the integer-DP coopmat
  path.
