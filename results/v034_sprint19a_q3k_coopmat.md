# Sprint 19A — Q3_K + Q5_K CoopMat Prefill (mul_mm.comp)

**Date:** 2026-05-03
**Branch:** main (post-v0.3.3)
**Goal:** Close the Q3_K_M prefill gap to llama.cpp by porting the Q4_K
cooperative-matrix path (Sprint 12K) to Q3_K and Q5_K.

## Hypothesis

Q3_K_M GGUFs prefill at 0.59× llama.cpp (2258 vs 3844 tok/s @ pp=512)
because Sprint 17B shipped Q3_K with **only** the integer-MMQ path
(`mul_mmq.comp`), while Q4_K has had the FP16 WMMA `mul_mm.comp` path
since Sprint 12K. Q3_K_M also routes its attn_v + ffn_down weights
through Q5_K, which Sprint 17C wired Mmq-only as well.

If we add the FP16 WMMA SPV family for Q3_K and Q5_K (cloning the
Q4_K/Q6_K build matrix) and route through it, prefill should match Q4_K_M.

## Pre-check

Before writing code:
1. `mul_mm_funcs.glsl` line 166 — `#elif defined(DATA_A_Q3_K)` branch
   exists, "2 values per idx" comment, reads only `data_a_packed16`.
2. `mul_mm_funcs.glsl` line 225 — `#elif defined(DATA_A_Q5_K)` branch
   exists, "4 values per idx" comment, reads only `data_a_packed32`.
3. `dequant_funcs.glsl` and `types.glsl` carry the Q3_K + Q5_K block
   layouts already (Sprint 17B/17C).

So the GLSL is complete. This sprint is purely build-defines + ShaderId
+ pipeline-registry + routing — no shader source changes. Matches the
"shader source vs config" memory rule.

## Implementation

### `build.rs` (+10 SPV jobs, 87 → 97 SPVs)

Cloned the Q4_K/Q6_K mul_mm matrix:

| SPV | Q4_K analogue | LOAD_VEC_A |
|---|---|---|
| `mul_mm_q3_k_f32.spv` | `mul_mm_q6_k_f32.spv` | 2 (Q6_K-shape) |
| `mul_mm_q3_k_f32_aligned.spv` | `mul_mm_q6_k_f32_aligned.spv` | 2 |
| `mul_mm_q3_k_f32_coopmat.spv` | `mul_mm_q6_k_f32_coopmat.spv` | 2 |
| `mul_mm_q3_k_f32_aligned_coopmat.spv` | … | 2 |
| `mul_mm_q3_k_f32_aligned_coopmat_f16acc.spv` | … | 2 |
| `mul_mm_q5_k_f32.spv` | `mul_mm_q4_k_f32.spv` | 4 (Q4_K-shape) |
| `mul_mm_q5_k_f32_aligned.spv` | … | 4 |
| `mul_mm_q5_k_f32_coopmat.spv` | … | 4 |
| `mul_mm_q5_k_f32_aligned_coopmat.spv` | … | 4 |
| `mul_mm_q5_k_f32_aligned_coopmat_f16acc.spv` | … | 4 |

Q3_K uses LOAD_VEC_A=2 (mirror of Q6_K's "2 values per idx" Mmq dequant
that only writes one FLOAT_TYPEV2 per invocation). Q5_K uses LOAD_VEC_A=4
(mirror of Q4_K).

### `shaders.rs` (+18 ShaderIds, +10 SPV consts)

Each quant gets nine `MulMm{Q3K,Q5K}{,Aligned,Coopmat,AlignedCoopmat,
CoopmatM,AlignedCoopmatM,CoopmatS,AlignedCoopmatS,AlignedCoopmatF16Acc}`
ShaderIds. The M/S tile variants reuse the L-tile SPV bytes (BM/BN/BK
are spec-constants, see Sprint 12M/13A).

### `pipeline_registry.rs`

Added the new ShaderIds to the existing match arms — all 18 new
variants go through the same spec-constant blocks as Q4_K/Q6_K (the
warptile shape is identical across quants for a given tile size).
Extended the `s_tile` / `m_tile` matchers to recognise the Q3_K + Q5_K
variants.

### `forward.rs`

1. **`run_gemm` bm/bn match arms**: added the 12 new coopmat ShaderIds
   to the (128,128), (64,64), (32,32) cases (Sprint 17B-style latent
   bug avoidance — every L/M/S tile variant must be present).
2. **`force_mmq`**: `Q3_K | Q5_K | Q4_0 → Q4_0` only. Q3_K and Q5_K
   now flow into the regular MulMm path.
3. **`layer_weight_shader_gemm`**: added two early-return branches
   (Q3_K and Q5_K) before the existing Q4_K/Q6_K match. Each picks
   the right MulMm variant from the same five-way (gemm_kind,
   coopmat_q4k_mm, coopmat_aligned, coopmat_m_tile, coopmat_s_tile)
   tuple as Q4_K. Removed the now-dead `if q3` / `if q5` Mmq guards
   in the Q4_K fall-through match.

## Pitfall caught

First pp_bench run after the change reported **2274 tok/s** at pp=512
— identical to the Mmq baseline. A trace eprintln in `dispatch_layer_batch`
revealed the routing was correct (`MulMmQ3KAlignedCoopmat`,
`MulMmQ5KAlignedCoopmat`), but `cargo build --release` did **not**
rebuild example binaries. `target/release/examples/run_pp_bench` was
still the v0.3.3 binary linking the old library snapshot. After
`cargo build --release --examples` the same trace fired the new ShaderIds
**and** pp=512 jumped to **3450 tok/s**. Memory-saving lesson: examples
need explicit `--examples` after a library-only change.

## Results

### Q3_K_M (Qwen3-8B), median over 3 runs

| pp | Mmq baseline (v0.3.3) | CoopMat (Sprint 19A) | Speedup | vs llama.cpp |
|---:|----------------------:|---------------------:|--------:|-------------:|
|  64 | (~1230, est.) | 1650 tok/s | 1.34× | — |
| 128 | (~1580, est.) | 2531 tok/s | 1.60× | — |
| 256 | (~1730, est.) | 3384 tok/s | 1.96× | — |
| **512** | **2258 tok/s** | **3536 tok/s** | **1.57×** | **0.92×** (3844) |
| 1024 | (~2240, est.) | 3362 tok/s | 1.50× | — |

(Mmq baseline numbers from Sprint 17B/17C reports + a fresh stale-binary
run done before rebuilding examples.)

### Q4_K_M (Qwen3-8B) — regression check

| pp | v0.3.3 | Sprint 19A |
|---:|-------:|-----------:|
|  64 | 1678 tok/s | 1696 tok/s |
| 128 | 2576 | 2578 |
| 256 | 3558 | 3549 |
| **512** | **3865** | **3828** (within noise) |
| 1024 | 3738 | 3718 |

### Decode (1-tok prompt + 32 gen, median over 3 runs)

| Model | v0.3.3 | Sprint 19A | Δ |
|---|---:|---:|---:|
| Q3_K_M | 133.7 tok/s (FP8 KV) | 131.7 tok/s | within noise |
| Q4_K_M | 118.5 tok/s (FP8 KV) | 116.5 tok/s | within noise |

Decode uses the GEMV path (`mul_mat_vec_*`) — coopmat changes only
touch GEMM, so decode is unaffected.

### 15-prompt suite

* Q3_K_M: **15/15 coherent**, median decode 121.6 tok/s, median prefill
  812.7 tok/s (small-prompt average; the dedicated pp=512 run is the
  apples-to-apples llama.cpp comparison).
* Q4_K_M: **15/15 coherent**, median decode 109.3 tok/s, median prefill
  850.7 tok/s.

## Routing trace (verification)

For Q3_K_M at L0:

| seq_len | sq | sk | sv |
|---:|---|---|---|
|  64 | MulMmQ3KAlignedCoopmat**M** | … | MulMmQ5KAlignedCoopmat**M** |
| 128 | MulMmQ3KAlignedCoopmat | … | MulMmQ5KAlignedCoopmat |
| 512 | MulMmQ3KAlignedCoopmat | … | MulMmQ5KAlignedCoopmat |
| 1024| MulMmQ3KAlignedCoopmat | … | MulMmQ5KAlignedCoopmat |

So both new SPV families are exercised. attn_v in Q3_K_M is Q5_K
(mixed-quant) — the Q5_K branch of `layer_weight_shader_gemm` returns
MulMmQ5KAlignedCoopmat, confirming the new Q5_K coopmat path is live.

## Outcome

Bench gate **3500 tok/s @ pp=512 for Q3_K_M**: **PASSED** (3536 tok/s,
0.92× llama.cpp).

* Single largest prefill gain in any v0.3.x sprint: **+57% at pp=512**
  on Q3_K_M, no decode regression, no Q4_K_M regression, all suites
  coherent.
* Q3_K_M now sits at **0.92× llama.cpp** prefill — same regime as
  Q4_K_M (0.89×). The remaining gap is the same ~10% multi-submit /
  command-buffer-reuse plumbing that bottlenecks every quant family
  (open work item from v0.3.3).
* Q5_K_M is now wired but untested at runtime (no Q5_K_M GGUF in the
  test set). The Q5_K mul_mm path is exercised through Q3_K_M's
  attn_v / ffn_down weights at every pp — so the SPVs are at least
  numerically sound.

## Files touched

* `build.rs` — +10 ShaderJobs (Q3_K + Q5_K mul_mm matrix).
* `src/backend/vulkan/shaders.rs` — +18 ShaderIds, +10 SPV consts.
* `src/backend/vulkan/pipeline_registry.rs` — Q3_K + Q5_K added to
  3 existing match arms (standard MulMm, coopmat, s_tile / m_tile
  matchers).
* `src/backend/vulkan/forward.rs` — `run_gemm` tile shape match;
  `force_mmq` narrowed to Q4_0; `layer_weight_shader_gemm` Q3_K + Q5_K
  early-returns.
