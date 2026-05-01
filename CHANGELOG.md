# Changelog

## v0.2.2 — coopmat WMMA prefill default-on (2026-05-01)

### Headline

**Prefill peak +71 %** (2255 → 3863 tok/s @ pp=512), reaching **0.89 ×
llama.cpp Vulkan** prefill at pp ≥ 256 (up from 0.52 × in v0.2.0).
KHR cooperative-matrix WMMA prefill is now **default-on** for Q4_K and
Q6_K GEMMs. Decode unchanged at 91.1 tok/s median (0.80 × llama.cpp).
27 / 27 lib tests, 15 / 15 coherent on the bench suite.

### Performance (Qwen3-8B-Q4_K_M, RX 9070 XT, RUNS=5 median)

| pp   | v0.2.0  | v0.2.2  | Δ        | vs llama.cpp Vulkan |
|------|--------:|--------:|---------:|--------------------:|
|   64 |   1511  |   1678  |  +11 %   | 0.73 × |
|  128 |   2001  |   2560  |  +28 %   | 0.70 × |
|  256 |   2200  |   3558  |  +62 %   | 0.89 × |
|  512 |   2255  | **3863** | **+71 %** | **0.89 ×** |
| 1024 |   2204  |   3748  |  +70 %   | 0.90 × |
| 2048 |   1997  |   3172  |  +59 %   | 0.84 × |

llama.cpp reference: build 23b8cc4 with `-fa 1` on the same hardware.

### What changed

- **KHR cooperative-matrix WMMA prefill** — Q4_K and Q6_K GEMM
  dispatch now flows through RDNA4's 128 AI Accelerators via
  `VK_KHR_cooperative_matrix`, mirroring llama.cpp's `mul_mm.comp`
  pipeline. All shader sources remain **byte-identical** to llama.cpp
  HEAD (`md5sum` confirmed across the entire arc) — every gain came
  from build-defines, spec-constants, SPV variants and runtime
  routing.
- **Aligned coopmat variant** — `LOAD_VEC_B=8` with `B_TYPE=mat2x4`,
  4 × wider B-matrix loads. Selected when `seq_len % 8 == 0`. Single
  biggest sprint gain (+64 % at pp=512).
- **L-tile + M-tile pipelines** — L `{256,128,128,32,64,64,2,16,16,16,64}`
  and M `{128,64,64,16,64,32,2,16,16,16,64}` warptiles share SPV
  binaries; only spec-constants differ. The runtime selector is a
  port of llama.cpp's `ggml_vk_guess_matmul_pipeline`.
- **Q6_K coopmat shader** — `mul_mm_q6_k_f32_coopmat.spv` built with
  `LOAD_VEC_A=2` (Q6_K is 2 weights / idx, not 4), removing the
  scalar-FP32 fallback that previously routed `ffn_down` /
  `attn_v` GEMMs through the slowest path.
- **Default-on toggle** — `VULKANFORGE_DISABLE_MM_COOPMAT=1` (or the
  legacy `VULKANFORGE_USE_MM_COOPMAT=0`) opts out. Default with no
  env var: **on**.

### New shaders / SPVs

```
NEW   vk_shaders/spirv/mul_mm_q6_k_f32_coopmat.spv          (Sprint 12K)
NEW   vk_shaders/spirv/mul_mm_q4_k_f32_aligned_coopmat.spv  (Sprint 12L)
NEW   vk_shaders/spirv/mul_mm_q6_k_f32_aligned_coopmat.spv  (Sprint 12L)
```

M-tile pipelines are 0 new SPVs — they reuse the L-tile binaries via
spec-constants.

### Sprint highlights (v0.2.2 series, 2026-05-01)

- **Sprint 12I — prefill RGP profiling.** Confirmed the v0.2.0/v0.2.1
  prefill gap is entirely the missing KHR coopmat WMMA pipeline.
  GEMV-side shaders are already at 77–91 % peak HBM bandwidth
  (verified in 12G-D / 12H).
- **Sprint 12J — coopmat WMMA prefill (diagnosis).** First end-to-end
  coopmat run was 6 % slower than `mul_mmq`. Root cause: Q6_K GEMMs
  fell to the scalar `mul_mm` FP32 path because no Q6_K coopmat SPV
  existed.
- **Sprint 12K — Q6_K coopmat shader + routing.** Built the Q6_K
  coopmat SPV, fixed `(GemmKind::MulMm, q6) => MulMmQ6K` routing arm.
  `gemm_down` 89 568 → 43 074 µs (−51 %), pp=512 2 348 → 2 697.
- **Sprint 12L — aligned LOAD_VEC_B=8 mat2x4.** Largest single-sprint
  win. Closes pp=128 from 1 427 to 2 576, pp=512 to 3 858 (0.89 ×
  llama.cpp). Regressed pp=64 by 18 % (L-tile starvation).
- **Sprint 12M — M-tile + default-on.** Added `BM=64` warptile for
  small `seq_len`, fixed pp=64 regression (1 234 → 1 678), flipped
  coopmat default-on.

### Sprint 12 analysis arc (v0.2.1 work feeding into v0.2.2)

- **12A / 12B / 12C** — llama.cpp + VulkanForge Vulkan-backend audits
  + gap analysis. Identified KHR coopmat as the remaining lever.
- **12D** — barrier elision: 0 % wall-time impact. Honest negative.
- **12E** — decode norm+rope fusion: +1 %. Dispatch overhead is not
  the decode bottleneck. Honest negative.
- **12G-A / G-B** — ggml shared-layer audit + VulkanForge
  shared-layer audit (we have none of it).
- **12G-C / G-D** — per-dispatch GPU timestamp profiling + RGP GUI
  capture analysis. Discovered the `vkCmdWriteTimestamp` artifact
  for back-to-back RAW-independent dispatches (inflates the
  second's `TOP_OF_PIPE` reading).
- **12H — Q6_K BW recovery.** Honest negative: GEMV/GEMM shaders are
  byte-identical to llama.cpp HEAD; the "50 % peak BW" reading was
  an RGP `INSTRUCTION_TIMING` perturbation artifact (real BW
  77 %).

### Key methodology findings

- Per-dispatch CPU overhead is **not** the bottleneck on RDNA4 at
  steady-state decode (verified at 0.1 % CPU residency).
- `vkCmdWriteTimestamp` is unreliable for barrier-less dispatch
  pairs — a 1-line dispatch-order swap detects the artifact.
- `RADV_THREAD_TRACE_INSTRUCTION_TIMING` inflates kernel durations
  by 50–60 % vs no-instruction tracing — needed for source mapping
  but not for absolute wall-time numbers.
- All compute-shader sources are upstream-identical to llama.cpp;
  the prefill gap was 100 % build-define / spec-constant / SPV
  / routing, never GLSL.
- Pre-check methodology (md5sum vs upstream + variant-table diff)
  saved weeks across Sprints 11A / 11D / 11E / 12A / 12H — six
  hits, all honest negatives.

### Negative results (kept as documentation)

- **12D** barrier elision — every barrier is at a RAW boundary;
  0 % can be elided.
- **12E** norm + rope fusion — +1 %, below noise; dispatch
  overhead is not the lever.
- **12H** Q6_K shader optimisation — md5-identical to upstream;
  nothing to port.
- **12J** coopmat WMMA first-pass — Q6_K regression masked the
  Q4_K wins; resolved in 12K.

### Files added / changed in v0.2.2

```
EDIT  Cargo.toml                              0.2.0 → 0.2.2
EDIT  src/backend/vulkan/forward.rs           coopmat selector + routing
                                              + default-on env-var parsing
EDIT  src/backend/vulkan/pipeline_registry.rs L/M-tile spec-constants
EDIT  src/backend/vulkan/shaders.rs           +7 ShaderId variants
EDIT  build.rs                                +3 SPV compile jobs
NEW   vk_shaders/.../mul_mm_q6_k_f32_coopmat.spv
NEW   vk_shaders/.../mul_mm_q4_k_f32_aligned_coopmat.spv
NEW   vk_shaders/.../mul_mm_q6_k_f32_aligned_coopmat.spv
NEW   results/v021_sprint12{a,b,c,d,e,f}_*.md
NEW   results/v021_sprint12g{a,b,c}_*.md
NEW   results/v021_sprint12gd_*.md (3 files: analysis, retry, GUI)
NEW   results/v021_sprint12h_q6k_bw_recovery.md
NEW   results/v022_sprint12i_prefill_rgp.md
NEW   results/v022_sprint12j_coopmat_prefill.md
NEW   results/v022_sprint12k_q6k_coopmat.md
NEW   results/v022_sprint12l_sml_tiles.md
NEW   results/v022_sprint12m_mtile.md
EDIT  README.md                               v0.2.2 perf table + features
EDIT  CHANGELOG.md                            this entry
```

### What's still on the table

- **S-tile (BM=32)** for pp ≤ 32 / short-prompt 15-prompt suite.
  ~30 LOC, ~1 hr.
- **Decode coopmat for `lm_head`** (vocab-major GEMV with N=151 936) —
  RGP showed 6 % of decode; ~3 % decode improvement potential.
- **f16-accumulator coopmat variant** (`f16acc`) llama.cpp ships —
  closes the remaining ~0.10–0.15 × peak-WMMA gap. Bigger lift.

---

## v0.2.1 — sprint 11/12 prefill instrumentation (internal, 2026-04-30)

Sprints 11G-A through 12H landed on `main` between v0.2.0 and v0.2.2
without a tagged release. Highlights:

- **L-tile `mul_mmq`** prefill spec-constants tuned (Sprint 11E).
  +4–5 % across pp range.
- **`SyncTracker`** barrier infrastructure with elision audit
  (Sprint 12D).
- **`rms_norm_mul_rope`** decode-side fusion shader experiments
  (Sprint 12E, negative result).
- **Per-dispatch GPU timestamp profiler** (`ShaderProfiler`,
  Sprint 12G-C) and **RGP capture infrastructure** (Sprint 12G-C
  / G-D).
- **End-state perf snapshot** (Sprint 12F): decode 91.5 tok/s,
  pp=512 2 352 tok/s.

These are individually committed but not released; the v0.2.2 tag
covers the full range.

---

## v0.2.0 — coopmat attention + FP16 KV + kernel fusion (2026-04-29)

### Headline

**Prefill peak +118 %** (1037 → 2255 tok/s @ pp=512), **pp=4096 unblocked**
(was DEVICE_LOST), **decode +2 %** (88.6 → 90.5 tok/s). Reaches **0.52×
llama.cpp Vulkan prefill peak** and **0.79× decode** on Qwen3-8B-Q4_K_M.
167/167 tests green across 30+ sprints in 2 days.

### Performance (Qwen3-8B-Q4_K_M, RX 9070 XT, RUNS=5 median)

| pp   | v0.2.0  | v0.1.3 (15-prompt med) | Δ        |
|------|--------:|-----------------------:|---------:|
|  64  |  1511   |                  ~600  | +152 %   |
| 128  |  2001   |                  ~900  | +122 %   |
| 256  |  2200   |                 ~1037  | +112 %   |
| 512  |  2255   |                 ~1037  | +118 %   |
| 1024 |  2204   |                  ~900  | +145 %   |
| 2048 |  1997   |                  ~700  | +185 %   |
| 4096 |  1659   |                 CRASH  |  unblock |

(v0.1.3 didn't ship a pp-sweep; column shows the 15-prompt-bench
medians which mix prompt lengths — directionally fair, not apples-to-
apples. The pp-sweep numbers are what `examples/run_pp_bench` produces.)

### Sprint highlights (v0.2.0 series, 2026-04-28 → 2026-04-29)

- **Sprint 5–7** — tiled flash-attention `flash_attn_tiled.comp`
  (Br=16, Bc=32) + Br/Bc sweep. Default Br=16 / Bc=32. +164 % at pp=1024
  vs the v0.1.3 `flash_attn_batch` shader.
- **Sprint 8a** — flash-attention default ON.
- **Sprint 8b / 8b.1** — conditional barriers honest-negative; llama.cpp
  barrier analysis preserved as documentation.
- **Sprint 9d.1–9d.3** — FP16 KV-cache infrastructure → prefill hot-path
  (+21 % @ pp=2048) → default ON. Half the cache VRAM at no parity cost.
- **Sprint 10A** — `flash_attn_cm2.comp` deep-dive; pivoted to
  `flash_attn_cm1.comp` (cm2 is `GL_NV_cooperative_matrix2`-only,
  RDNA4 only advertises `VK_KHR_cooperative_matrix`).
- **Sprint 10B** — isolated coopmat-QK microbench. **47.5× scalar FMA**
  on Br=Bc=16 — STRONG GO for end-to-end integration.
- **Sprint 10C** — `flash_attn_coopmat.comp` v1: KHR coopmat for QK,
  scalar softmax + scalar PV. Drop-in for `flash_attn_tiled` with the
  same bindings, dispatch geometry, and online-softmax state.
  **+85.8 % at pp=2048** vs scalar tiled.
- **Sprint 10D** — PV-coopmat with LDS-scratch hybrid. Passed 167/167
  parity but regressed pp-sweep 1–24 %. Reverted per the brief's
  fallback rule. Honest-negative in `results/v02_sprint10d_pv_coopmat.md`.
- **Sprint 10E** — coopmat attention default ON (env opt-out via
  `VULKANFORGE_COOPMAT_ATTN=0`).
- **Sprint 10E.5** — pp=4096 TDR-crash investigation. Bisection showed
  `COOPMAT_ATTN` is the determining factor; default-ON fixes it. No
  code change committed — 10E was already the fix.
- **Sprint 10F** — final bench + docs + push (this release).

### Fused kernels added across the v0.2 series

| Kernel             | Replaces                                  | Site                |
|--------------------|-------------------------------------------|---------------------|
| `swiglu`           | `silu` + `mul`                            | FFN                 |
| `multi_add_rms`    | `add` + `add` + `rms_norm` (×2 sites)     | block in/out        |
| `rms_norm_mul_rope`| `rms_norm` + `mul` + `rope`               | Q-norm + RoPE       |

Net: **−5 dispatches per layer** (Qwen3-8B has 36 layers).

### Coopmat attention details

`flash_attn_coopmat.comp` is a drop-in replacement for the scalar
`flash_attn_tiled.comp`. The QK score matrix is computed by a single
16×16×16 coopmat MulAdd chain over `head_dim=128` (8 steps), with
`q_lds` and `k_lds` staged in FP16 LDS (4 KB each) and `scores_lds`
in FP32 (1 KB) — total 9 KB LDS vs 26 KB for the scalar shader. K^T
is obtained via a `ColumnMajor` `coopMatLoad` with stride=head_dim.

Softmax + PV remain scalar (per-thread `my_out0/my_out1[BR]`
accumulators) — Sprint 10D's PV-coopmat regressed end-to-end and was
reverted. KHR rev2 only (no NV cm2 dependencies).

FP16-KV variant present and selected automatically when the cache is
allocated FP16 (default).

### TDR resolution

pp=4096 used to return `DEVICE_LOST` because scalar
`flash_attn_tiled`'s last-chunk attention (kv_len=4096) crossed RADV's
~5 s TDR window. Coopmat brings the per-tile compute under the
watchdog. Bisection (Sprint 10E.5) confirmed `COOPMAT_ATTN` is the
single determining variable — `FP16_KV` is irrelevant to the crash.

### Test suite

```
test result: ok. 27 + 9 + 18 + 70 + 8 + 8 + 27 = 167 passed; 0 failed
```

All green. Doc-tests: 0/0.

### Files added in v0.2 series (selected)

```
NEW   vk_shaders/flash_attn_tiled.comp           (Sprints 5–7.6)
NEW   vk_shaders/flash_attn_coopmat.comp         (Sprint 10C)
NEW   vk_shaders/flash_attn_coopmat_fp16kv.comp  (Sprint 10C, build var)
NEW   vk_shaders/swiglu.comp                     (kernel fusion)
NEW   vk_shaders/multi_add_rms.comp              (kernel fusion)
NEW   vk_shaders/rms_norm_mul_rope.comp          (kernel fusion)
NEW   vk_shaders/bench_qk_scalar.comp            (Sprint 10B microbench)
NEW   vk_shaders/bench_qk_coopmat.comp           (Sprint 10B microbench)
NEW   examples/bench_qk.rs                       (Sprint 10B microbench)
NEW   examples/run_pp_bench.rs                   (Sprint 9d.2 pp-sweep)
NEW   results/v02_sprint{5,6,7,7.5,7.6,8a,8b,8b.1}_*.md
NEW   results/v02_sprint9d{,.1,.2,.3}_*.md
NEW   results/v02_sprint10{a,b,c,d,e,e5,f}_*.md
EDIT  src/backend/vulkan/forward.rs               (coopmat selector, FP16 KV path)
EDIT  src/backend/vulkan/kv_cache.rs              (FP16 layout)
EDIT  src/backend/vulkan/shaders.rs               (53 → 59 ShaderId entries)
EDIT  build.rs                                    (new SPV compile jobs)
EDIT  Cargo.toml                                  (0.1.3 → 0.2.0)
EDIT  README.md                                   (v0.2.0 perf table)
EDIT  CHANGELOG.md                                (this entry)
```

---

## v0.1.3 — Phase 7 mul_mm.comp debug + silent mul_mmq fix (2026-04-27)

### Performance addendum — first corrected 16-prompt benchmark (added later same day)

All v0.1.0 – v0.1.2 prefill numbers were inflated by the `BLOCK_SIZE = 128`
bug below — half the GEMM work was silently skipped. v0.1.3 ships the
**first accurate prefill measurements**:

| Model | Decode med | Prefill med | Coh | Alice |
|---|---:|---:|---:|---:|
| Qwen3-8B | 88.6 | 1037.4 | 15/15 | 3/3 |
| Meta-Llama-3.1-8B | 94.8 | 1092.7 | 12/15 | 3/3 |
| DeepSeek-R1-Distill-Llama | 94.3 | 904.1 | 15/15 | 3/3 |
| Mistral-7B-Instruct-v0.3 | 100.1 | 939.3 | 15/15 | 3/3 |

vs the (now-invalidated) v0.1.2 numbers:

| Model | Δ Decode | Δ Prefill | Note |
|---|---:|---:|---|
| Qwen3-8B | +0.1 | **−7.0 %** | full GEMM tile now |
| Meta-Llama-3.1-8B | +0.2 | **−9.5 %** |  |
| DeepSeek-R1-Distill-Llama | −1.2 | **−6.1 %** |  |
| Mistral-7B-Instruct-v0.3 | −0.1 | **−6.6 %** |  |

Decode is unchanged because the GEMV path (`mul_mat_vec_*.comp`)
doesn't tile its output and was unaffected. Prefill is consistently
lower because the corrected GEMM does ~2× the work per output tile.
Llama-3.1's coherence drops 13/15 → 12/15 on short numeric prompts
(`Simple Sequence`, `Arithmetic`) — the bench's "repeating garbage"
heuristic trips on legitimate digit-only replies; the multi-turn
Alice test still passes 3/3 and the regression suite's top-1 /
top-5 parity gates are identical to v0.1.2. Per-model logs in
`results/v013_logs/`. Full report in `results/phase7_v013_benchmark.md`.

### Headline

Two bugs uncovered while bringing up the `mul_mm.comp` port from
Phase 6 v0.1.2 — one of them was silently corrupting `mul_mmq` output
in production for any prompt longer than 32 tokens. Both fixed; full
test suite up from 82 → 93 tests, all green.

### Bug 1 — `BLOCK_SIZE / NUM_WARPS` undercoverage (affected mul_mmq + mul_mm)

Every workgroup must have enough warps to cover all `(BM/WM) × (BN/WN)`
warp tiles. With `BM = BN = 64`, `WM = WN = 32`, four warp tiles per
workgroup are needed. `BLOCK_SIZE = 128` on RDNA Wave64 produces only
`128 / 64 = 2` warps; `warp_c = warp_i / (BM/WM)` was therefore always
`0`, so cols `[WN, BN) = [32, 64)` of every output tile were never
written. The bug went undetected because:

- The pre-existing `test_gemm_q4k_vs_gemv_seq1_parity` test ran
  `M = 2, N = 1` — both far inside the bounds-check, so the missing
  warp was clipped anyway.
- `phase3e_prefill_batch_matches_token_by_token_top5` runs the
  "Explain what a mutex is in one sentence." chat-templated prompt,
  which tokenises to ~29 tokens — below the 32-col threshold.

Fix: bump default `BLOCK_SIZE` from 128 → 256 in
`pipeline_registry.rs` for both `MulMmqQ4K/Q6K` and `MulMmQ4K/Q6K`
spec-constants.

A new dedicated test `test_gemm_q4k_full_tile_64x64_mul_mmq` runs
`M = N = 64, K = 256` against a CPU reference and would have caught
this immediately. Added.

### Bug 2 — Q6_K `LOAD_VEC_A` mismatch (affected mul_mm only)

The Q6_K `load_a_to_shmem` branch in `mul_mm_funcs.glsl` is
hard-coded for **2 weights per idx**:

```glsl
const uint ib = idx / 128;          // 2 values per idx
...
buf_a[buf_idx] = FLOAT_TYPEV2(q.x, q.y);     // 1 vec2 = 2 weights
```

The Q4_K branch above it is **4 weights per idx** and writes two
`vec2`s. We had compiled both with `LOAD_VEC_A = 4` (matching
llama.cpp's `vulkan-shaders-gen.cpp:560`). On the Q6_K path that
left `buf_a[buf_idx + 1]` uninitialised, surfacing as `NaN` logits
once the GEMM hit a layer whose weights were Q6_K (`ffn_down`,
`token_embd` on Qwen3-8B-Q4_K_M).

Fix: pin `LOAD_VEC_A = 2` for the `mul_mm_q6_k_f32` build job.

### Status of mul_mm

* Bit-exact across all 11 new GEMM-parity unit tests (covering
  `K = 256/512/2048/11008`, aligned + unaligned `N`, single + multi
  `BM/BN` tiles, real-prefill `M=2048 N=62 K=2048`, and `ffn_down`
  dimensions).
* Phase-3E top-5 vs per-token GEMV: **5/5 overlap, top-1 = 151667**
  with `VULKANFORGE_USE_MUL_MM=1`.
* Default stays **OFF** — `mul_mmq` is ~45 % faster at prefill on
  `Qwen3-8B-Q4_K_M` (FP32 activations into LDS take 4× the bandwidth
  of `Q8_1`-packed activations). Opt in with
  `VULKANFORGE_USE_MUL_MM=1` when you specifically want to validate
  drift attributable to `Q8_1` quantisation of activations.

| Prompt | mul_mmq | mul_mm | Δ |
|---|---:|---:|---:|
| 29 tok mutex | 545 tok/s | 309 tok/s | −43 % |
| 55 tok essay | 980 tok/s | 538 tok/s | −45 % |

(Same hardware, BLOCK_SIZE = 256 in both cases. Decode is unchanged
because decode uses GEMV, not GEMM.)

### Test suite

`cargo test --release` — **93 / 93 green** (was 82). The 11 new tests
sit under `test_mul_mm_q4k_*` and `test_gemm_q4k_full_tile_64x64_*`
in `tests/correctness.rs`.

Full investigation: `results/phase7_mul_mm_debug.md`.

## v0.1.2 — Phase 6 fallback work (2026-04-27)

### Performance addendum — GEMM tile-tuning (added later same day)

Sweep over `mul_mmq.comp`'s spec-constants found a single new
default — `TM=2 TN=4` (was `TM=4 TN=2`) — that lifts prefill
median by **+3 to +6 % across all four supported models**:

| Model | v0.1.1 | v0.1.2 (TM=2 TN=4) | Δ |
|---|---:|---:|---:|
| Qwen3-8B | 1082.3 | 1115.6 | +3.1 % |
| Meta-Llama-3.1-8B | 1140.4 | 1207.6 | +5.9 % |
| DeepSeek-R1-Distill | 919.0 | 963.0 | +4.8 % |
| Mistral-7B-v0.3 | 949.0 | 1005.7 | +6.0 % |

Single-line pipeline-registration change. No shader edits, no SPV
rebuilds (the values are spec-constants, the shader's SPIR-V is
unchanged). `VULKANFORGE_GEMM_{BLOCK_SIZE,TM,TN}` env vars added
for future A/B testing without rebuilding.

Sweep details in `results/phase6_v012_tile_tuning.md`.

### Headline

Coopmat / WMMA path was found non-viable for v0.1.x — `mul_mm_cm2.comp`
depends end-to-end on `GL_NV_cooperative_matrix2` and RADV gfx1201
advertises only `VK_KHR_cooperative_matrix`. A from-scratch KHR-only
GEMM kernel was descoped to v0.2 (3-4 weeks). v0.1.2 ships the
fallback work-list from Phase 6A's §4.3 that doesn't require a new
kernel:

- **Pipeline-cache wired through** — `save_cache()` was implemented
  in v0.1.0 but never called. v0.1.2 calls it at REPL shutdown. Cold
  start writes 158 KB of compiled pipelines to
  `$HOME/.vulkanforge/pipeline_cache.bin`; the next start loads them
  back and skips the ACO compile pass. (Steady-state perf unchanged —
  this is purely a startup-latency win.)
- **Sampling: temperature / top-k / top-p / repetition-penalty.** The
  legacy greedy path is preserved as the `temperature == 0.0` short-
  circuit, so every benchmark and regression test stays byte-
  deterministic. Configurable via `VF_TEMPERATURE`, `VF_TOP_K`,
  `VF_TOP_P`, `VF_REPETITION_PENALTY`, `VF_SEED` in the REPL. RNG is
  a small xorshift64* that takes a per-run seed.
- **Phase 6A coopmat probe + naive WMMA bench** retained as artefacts
  (`examples/probe_coopmat.rs`, `examples/bench_coopmat.rs`).

### Performance

15-prompt suite, Qwen3-8B-Q4_K_M:

| Metric | v0.1.1 | v0.1.2 | Δ |
|---|---:|---:|---|
| Decode median | 88.8 | 88.3 | run-to-run noise |
| Prefill median | 1082.3 | 1050.2 | run-to-run noise |
| Coherent | 14/15 | 14/15 | unchanged |
| **Cold-start pipeline compile** | every run | every run + persisted | ~150 ms saved on warm starts |

The throughput numbers are unchanged because sampling at `T=0`
short-circuits to argmax — same code path as v0.1.1 — and the
pipeline cache only changes ACO-compile time at process start.

### Deferred

- **FP16 KV-cache** (Phase 6 §3) — touches 4 attention shaders
  (`flash_attn{_split,_reduce,_batch}.comp` + KV-cache buffer
  layout). Marginal expected gain (~+2-3 % decode at long context,
  ~50 MB VRAM headroom) vs non-trivial regression risk on the
  multi-turn correctness gate. Deferred to v0.1.3.
- **Barrier-coalescing** (Phase 6 §2) — every `compute_barrier` in
  `dispatch_layer_batch` is RAW-required. The remaining wins would
  need shader fusion (silu+mul, attn_norm+quantize), which is
  v0.2-class work.
- **Coopmat KHR-only GEMM from scratch** (Phase 6A §4.2) — v0.2 /
  Phase 7 milestone, ~3-4 weeks. Patterns from
  `flash_attn_cm1.comp` + `mul_mm.comp` in llama.cpp.

### Tests

```
unit (lib)         24   (+5: sampling unit tests)
correctness        33   (no change)
regression         25   (no change)
TOTAL              82   ALL GREEN
cargo clippy --release --tests --examples  →  clean
```

### Files added / changed in v0.1.2

```
EDIT  Cargo.toml                              0.1.1 → 0.1.2
EDIT  src/main.rs                             save_cache() call + sampling env vars
EDIT  src/backend/vulkan/decode.rs            Sampling struct + sample_next_token
                                              + 5 unit tests
EDIT  examples/run_alice_test.rs              GenerateConfig::sampling field
EDIT  examples/run_15prompt_bench.rs          ditto
EDIT  examples/run_validation.rs              ditto
EDIT  examples/sample_decode.rs               ditto
EDIT  tests/regression.rs                     ditto (4 sites)
EDIT  README.md                               sampling env-var doc
EDIT  CHANGELOG.md                            this entry
NEW   results/phase6_v012_optimizations.md    full report
```

---

## v0.1.1 — Phase 5B + 5C combined (2026-04-27)

### Headline performance (RX 9070 XT, 15-prompt suite)

| Model | Decode tok/s (median) | Prefill tok/s (median) | Δ vs Phase 5A |
|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | **88.8** | **1082.3** | prefill +167 % (was 404.9) |
| Meta-Llama-3.1-8B-Instruct | **94.8** | **1140.4** | prefill +133 % (was 489.9) |
| DeepSeek-R1-Distill-Llama-8B | **95.2** | **919.0** | prefill +112 % (was 433.9) |
| Mistral-7B-Instruct-v0.3 | **100.4** | **949.0** | (new model) |

VulkanForge prefill is now above the **ROCmForge HIP backend** ceiling
(~768 tok/s) for the first time and reaches ~48 % of llama.cpp Vulkan
(2274 tok/s, build 23b8cc4 `-fa 1`). Decode unchanged from Phase 5A
at ~76 % of llama.cpp Vulkan. Alice 6-turn multi-turn context-
retention test: **3 / 3 critical turns on all four models**.

### Phase 5B — fully-batched prefill (5B.1 + 5B.2 + 5B.3)

- **`flash_attn_batch.comp`** (Phase 5B.1): batched-Q flash attention
  shader. One dispatch covers (n_heads, M, 1) with a per-query causal
  mask `causal_len = q_start + q_idx + 1`. 145 LOC, 12 816 B SPIR-V.
  Eight isolated parity tests vs an f64 CPU reference.
- **`Forward::prefill_batch` integration** (Phase 5B.2): replaces
  the M-fold per-token attention dispatch loop with a single
  `flash_attn_batch` call. `+26 %` median prefill on Qwen3.
- **Per-token loop eliminated** (Phase 5B.3): batched RoPE (one
  dispatch per Q/K with `ne02 = M` and `rope_pos_buf[i2]`), batched
  Q/K-norm (`rms_norm` with `nrows = M × heads_per_token`), bulk
  KV-cache write (one `cmd_copy_buffer` per K/V per layer). Per-
  token sub-dispatch count `~22 860 → ~756` for `pp=62` (`~30 ×`).
  `+69 %` median prefill on top of 5B.2.
- Gated on `VULKANFORGE_BATCH_ATTN` (default ON; `=0` falls back to
  the per-token attention loop, useful for parity testing).
- No new shaders for 5B.2 / 5B.3 — all integration was host-side
  re-binding of existing `rope_neox.comp` / `rope_norm.comp` /
  `rms_norm.comp`.

### Phase 5C — SPM Tokenizer + Mistral Support

- SentencePiece Unigram tokenizer (greedy bigram-merge, mirrors
  llama.cpp's `llm_tokenizer_spm`). 422 LOC.
- Mistral-7B-Instruct-v0.3 support (`[INST] {body} [/INST]` template
  with the brackets emitted as their dedicated vocab ids 3 / 4).
- `Tokenizer` is now a dispatch struct over an internal
  `TokenizerInner::{Bpe, Spm}` enum.
- 4 new regression tests + 5 new lib unit tests for SPM + Mistral.

### Prompt 16 — Alice multi-turn context retention

- Six-turn `ChatSession` exchange with NO `reset()` between turns.
- Three critical turns ask the model to recall name / city / both.
- All four supported models 3 / 3 PASS — multi-turn KV-cache
  + chat-template-continuation is correct end-to-end.

### Test infrastructure

- `RUST_TEST_THREADS = 4` in `.cargo/config.toml` (the regression
  suite now has 25 tests each loading ~5 GiB of weights into
  the 16 GiB VRAM budget; default `num_cpus`-many threads
  manifest as `VK_ERROR_DEVICE_LOST`).
- 77 tests total (19 lib unit + 33 correctness + 25 regression).
- Regression-suite wall-clock dropped 86 s → 36 s after Phase 5B.3
  (every prefill-using test now goes through the batched path).

### Files added / changed in v0.1.1

```
NEW   vk_shaders/flash_attn_batch.comp
NEW   src/backend/vulkan/spm.rs
NEW   examples/run_alice_test.rs
NEW   examples/probe_batch_attn.rs
NEW   examples/spm_dump.rs
NEW   inference_test_prompts_16.json
NEW   inference_test_prompts_mistral_5.json
NEW   .cargo/config.toml
NEW   results/phase5b_step_1_batch_attn.md
NEW   results/phase5b_step_2_integration.md
NEW   results/phase5b_step_3_batch_ops.md
NEW   results/phase5b_step_4_benchmark.md
NEW   results/phase5c_spm_tokenizer.md
NEW   results/prompt16_alice_test.md

EDIT  src/backend/vulkan/forward.rs
EDIT  src/backend/vulkan/tokenizer.rs           (refactored to dispatch over BPE/SPM)
EDIT  src/backend/vulkan/chat_template.rs       (+ ChatTemplate::Mistral)
EDIT  src/backend/vulkan/pipeline.rs            (+ FlashAttnBatchPushConstants)
EDIT  src/backend/vulkan/pipeline_registry.rs
EDIT  src/backend/vulkan/shaders.rs             (+ ShaderId::FlashAttnBatch)
EDIT  src/backend/vulkan/mod.rs                 (pub mod spm)
EDIT  build.rs                                  (+ flash_attn_batch compile job)
EDIT  src/lib.rs                                (+ clippy::large_enum_variant allow)
EDIT  tests/regression.rs                       (+ 8 new tests)
EDIT  tests/correctness.rs                      (+ 8 new batch-attn parity tests)
EDIT  Cargo.toml                                (0.1.0 → 0.1.1)
EDIT  README.md                                 (perf table refresh)
EDIT  CHANGELOG.md                              (this entry)
```

---

## Phase 5A — CB-Reuse via Persistent Descriptor Sets (2026-04-26)

### Headline numbers (RX 9070 XT, gfx1201, 15-prompt suite)

| Model | Decode median tok/s | Δ vs 4D |
|---|---:|---:|
| Qwen3-8B-Q4_K_M | **88.5** | +22 % (was 72.4) |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | **94.6** | +16 % (was 81.5) |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | **94.8** | +17 % (was 81.3) |

Forward-pass per-token CPU breakdown (Qwen3, pos=100):

| Phase | Phase 4D | Phase 5A | Δ |
|---|---:|---:|---:|
| RECORD wall | 3.57 ms | **1.96 ms** | -45 % |
| per-layer | 96 µs | **51 µs** | -47 % |
| TOTAL | 13.7 ms | **11.2 ms** | -18 % |

### Added
- `Forward::alloc_or_get_set` — descriptor-set cache keyed on
  `(layout, bindings)` signature (8-binding fixed-size key, no heap
  alloc per call). On the decode hot path, every dispatch now does a
  `HashMap::get` instead of `vkAllocateDescriptorSets +
  vkUpdateDescriptorSets`.
- `BindingSignature` / `BindingEntry` types in `forward.rs`.
- `Forward::reset_descriptor_pool_and_cache` — used by paths whose
  bindings vary across calls (`prefill_batch`, `forward_layer_debug{,
  _intermediate}`).
- `CommandContext::one_shot_profiled` + `OneShotTimings` — wall-time
  breakdown for reset / begin / record / end / submit / wait. Used by
  the new `forward_token_profile` / `forward_token_profile_layers`
  paths and the `examples/profile_forward` driver.
- `examples/profile_forward.rs` — Phase-5A profiling harness:
  per-position phase breakdown plus drill-down into per-layer
  dispatch time inside the record block.
- New regression test `phase5a_cb_reuse_parity_qwen3` — runs Qwen3-8B
  for 16 tokens with `cache_enabled=false` and `cache_enabled=true`,
  asserts max abs logit diff `< 1e-6` and identical argmax at every
  step. Bit-exact (max abs err = 0) in practice.
- `Forward::set_cache_enabled` / `cache_enabled` — overrides the env
  var pick for tests.
- `results/phase5a_step_1_dgc_poc.md` — VK_EXT_device_generated_commands
  spec + RADV implementation study. NO-GO: the spec disallows
  intra-sequence barriers, capping host-call reduction at ~37 %, and
  ash 0.38 lacks EXT bindings. Documented as-is.
- `results/phase5a_step_2_cb_reuse.md` — CPU profile + Stage 2D
  implementation report.
- `results/phase5a_step_3_ship.md` — full 15-prompt benchmark on all
  three supported models with cache default-on.

### Changed
- **CB-reuse is now the DEFAULT.** `VULKANFORGE_CB_REUSE=0` (or
  `false`) opts back into the Phase-4D direct path for debugging /
  A/B comparisons. Any other value (or unset) keeps the cache on.
- `forward_token` skips `reset_descriptor_pool` when the cache is on
  — sets accumulate for the lifetime of the `Forward` instance.
- Descriptor pool sized 4× larger (`max_sets *= 4`) so a prefill_batch
  invalidation followed by a long decode can rebuild the cache without
  hitting the limit.
- All 19 `alloc_set + write_bindings` call-pairs in `forward.rs` now
  go through `alloc_or_get_set`. `dispatch_layer` and `dispatch_final`
  unchanged structurally.
- `forward.rs` removed dead `cpu_embedding_lookup` and unused
  `hidden_bytes` local; `examples/run_15prompt_bench.rs` gated dead
  fields with `#[allow(dead_code)]`.

### Verified
- 17/17 regression + 25/25 correctness tests pass with cache **on**.
- 17/17 regression + 25/25 correctness tests pass with cache **off**
  (`VULKANFORGE_CB_REUSE=0`).
- Bit-exact parity (`max_abs_err = 0e0`) at all 16 tested positions on
  Qwen3-8B.
- Coherent decode on all three supported models in the full
  15-prompt suite (some bench-heuristic false-negatives on
  digits-only / emoji-only outputs — outputs themselves are correct).

### Deferred (still on Phase 5+ backlog)
- Stage 2A — pipeline-handle cache + push-constant templates. After
  Stage 2D the per-layer time is already 51 µs, so additional savings
  from 2A are projected at ~5-7 µs/layer → ~+1-2 % decode. Not worth
  the additional code surface right now.
- Stage 2B — full CB reuse via UBO-driven dynamic parameters. Would
  require shader changes for ~17 shaders for at most ~+10 % decode
  beyond Stage 2D. Off the table since 2D alone landed > 80 tok/s.
- VK_EXT_device_generated_commands. NO-GO documented.

## Phase 4D — Multi-Model + Polish (2026-04-26)

### Added
- `RopeVariant::{Norm, Neox}` in `ModelConfig`, auto-detected from
  `general.architecture` (Qwen* → Neox, llama / mistral / deepseek → Norm).
  `forward.rs::run_rope_neox_with_pos_offset` dispatches the matching shader.
- `ChatTemplate` enum in new `backend::vulkan::chat_template` module, with
  `detect(gguf, tokenizer)` that prefers the embedded Jinja `chat_template`
  string over the architecture name. Variants: `ChatML`, `Llama3`,
  `DeepSeekR1`, `Raw`.
- `ChatSession::new_with_template` constructor — `ChatSession::new` keeps
  ChatML as the back-compat default for existing callers.
- `Tokenizer::flavour()` and `Tokenizer::special_id(name)` for generic
  special-token lookup. Llama-3 family (`pre="llama-bpe"`) is now a
  recognised flavour, with its own pre-split regex (`\p{N}{1,3}` rather
  than Qwen2's `\p{N}+`) and EOS namespace (`<|eot_id|>`,
  `<|end_of_text|>`, `<|eom_id|>` all terminate).
- `ModelConfig` now records `rope_variant` and continues to auto-detect
  `has_qk_norm` from `blk.0.attn_q_norm` tensor presence.
- `forward.rs` gates Q/K-norm dispatches on `cfg.has_qk_norm` — Llama family
  (no Q/K-norm tensors) skips them entirely.
- `examples/probe_model.rs` — dumps architecture + tokenizer + Q/K-norm
  tensor presence for any GGUF.
- `examples/sample_decode.rs` — runs one prompt through any model, prints
  the full decoded text. Useful for eyeballing coherence beyond the bench
  excerpt heuristic.
- README.md and this CHANGELOG.md.

### Changed
- `Tokenizer::im_start_id` / `im_end_id` / `endoftext_id` are now
  `Option<u32>` — populated for Qwen2/3, `None` for Llama-3. Callers
  that need the Qwen-specific ChatML ids must `.expect()` or check.
- `apply_chat_template` (the Phase-2D ChatML helper in `tokenizer.rs`)
  now panics when invoked on a non-Qwen tokenizer; new code should use
  `ChatTemplate::render_first_turn` instead.
- `decode::generate` auto-detects the chat template via `ChatTemplate::detect`
  rather than hard-coding ChatML.
- `examples/run_15prompt_bench.rs` honours `VF_NUM_PROMPTS` (truncates the
  prompt list) and prints the detected `arch / rope / template / qk_norm`
  before the run.

### Verified
- Qwen3-8B-Q4_K_M — 5/5 coherent, 72.4 tok/s decode (median, 5-prompt subset).
- Meta-Llama-3.1-8B-Instruct-Q4_K_M — 5/5 coherent, 81.5 tok/s decode.
- DeepSeek-R1-Distill-Llama-8B-Q4_K_M — 5/5 coherent, 81.3 tok/s decode
  (reasoning format with `<think>` priming).
- 16/16 regression + 25/25 correctness tests pass (no Phase-3/4A/4B/4C
  parity tests regressed).

### Deferred
- Mistral-7B-Instruct-v0.3 — `tokenizer.ggml.model = "llama"` (SPM
  unigram). Fails at tokenizer load with `BadModel("llama")`. SPM decoder
  is Phase 5 work.
- Gemma-4 — out of scope (different tensor layout).
- DeepSeek-R1-Distill-Qwen-7B — not present in `~/models`; the available
  DeepSeek file is the Llama-distill variant. Documented as the tested
  one.

### Notes for ROCmForge users
- The "Llama-3.1 instruction-blind" failure mode reported in
  `~/projects/ROCmForge/results/inference_test_20260425.md` does **not**
  reproduce on VulkanForge: "What is 2+2?" → "The answer is 4."
  Llama-3.1 generates correct, on-topic Python code, prose, and chain-of-
  thought reasoning across the 5-prompt suite. The seven hypotheses ruled
  out in ROCmForge's bug were therefore mooted here — likely something in
  the HIP backend's attention or RoPE path, not the chat-template / RoPE
  variant / EOS detection code that VulkanForge implemented for Phase 4D.

## Phase 4C — Multi-WG Attention (2026-04-25)
- Split-K attention worker + reducer with online-softmax merge.
- +41% aggregate decode on the 15-prompt suite (47.8 → 67.2 tok/s).
- 3 new parity tests at seq=64/200/2048.

## Phase 4B — Flash Attention (drop-in) (2026-04-25)
- Online-softmax flash-attention shader. ~tied perf with scalar_attn but
  served as the foundation for 4C.

## Phase 4A — GEMV VGPR Reduction (negative result) (2026-04-24)
- Documented that shaderc optimisation flags don't move ACO's register
  allocator; RGA offline mode can't see our spec constants.

## Earlier phases
See `results/phase{1,2,3}_*.md` for prior write-ups.
