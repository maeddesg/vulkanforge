# Changelog

## v0.3.8 — Block-wise FP8 (Qwen3-FP8) (2026-05-08)

### Headline

**Block-wise FP8 support — Qwen3-8B-FP8 runs end-to-end at 64.5 tok/s
decode, 770 tok/s prefill at pp=512.** This is the format used by
every official Qwen3 / Qwen3.5 / DeepSeek-V3 FP8 release on HuggingFace
(`weight_block_size: [128, 128]`). All three FP8 scaling strategies —
per-tensor, per-channel, block-wise — are now auto-detected from the
SafeTensors metadata. Block-wise prefill matches the per-tensor 8B
Llama path (770 vs 757 tok/s); the scale fold has no measurable
overhead in the GEMM steady state.

### Block-wise FP8 Performance (Qwen3-8B-FP8, RX 9070 XT, Mesa 26.1-rc3)

| Sprint                | Decode    | pp=64    | pp=128   | pp=512    | Avg W   | tok/s/W |
|-----------------------|----------:|---------:|---------:|----------:|--------:|--------:|
| 35 (decode-only)      |  64.5     |  153.8   |  166.4   |  157.2    |  305 W  |  0.21   |
| **36 (BN=32 GEMM)**   | **64.5**  | **669.7**| **721.9**|**770.1**  |**156 W**|**0.41** |
| Δ vs Sprint 35        | unchanged | +335 %   | +334 %   | **+390 %**| −49 %   | +95 %   |

Decode unchanged (Sprint 36 only touched the prefill GEMM path). Avg
power drops because the BN=32 GEMM finishes the prefill batch faster
than the GEMV-loop fallback could.

### What changed

**Sprint 35 — Block-wise FP8 Decode** (commit `c41767c`, 711 ins / 97 del)

* `vk_shaders/mul_mat_vec_fp8_blockwise.comp` (77 LOC) — new GEMV with
  per-K-block scale lookup (`scale[(row/block_n) * num_kblocks + b]`)
  and wave-level `subgroupAdd` reduction. Same 4-binding scheme as
  `mul_mat_vec_fp8_perchannel.comp`; differs in the 8-u32 push
  constant block carrying `block_size_n`, `block_size_k`,
  `num_kblocks`, and `input/output_off_floats` slots for prefill
  fallback dispatching.
* `loader.rs`: 2D scale shape parsing, both `.weight_scale` and
  `.weight_scale_inv` suffixes recognised, BF16 / F16 / F32 scale
  dtypes, `qwen3` joins `llama` / `qwen2` in the SafeTensors arch
  matcher. Trivially-2D `[N, 1]` shapes (Qwen2.5 per-channel) collapse
  to per-channel automatically.
* `forward.rs`: `fp8bw_*` pipeline triple + descriptor cache,
  `run_gemv_fp8_dispatch` routing wrapper, all 10 GEMV call sites in
  `dispatch_layer` / `dispatch_layer_partial` updated.
* `GpuTensor.scale_block: Option<(u32, u32)>` field.
* Initial prefill via `run_gemm_fp8_blockwise_via_gemv_loop`
  (M sequential GEMVs — slow but correct, replaced by Sprint 36).

**Sprint 36 — Block-wise FP8 GEMM** (commit `1aa6521`, 436 ins / 13 del)

* `vk_shaders/mul_coopmat_fp8_bn32_blockwise.comp` (161 LOC) — BN=32
  cooperative matrix GEMM. Strategy: fold the per-block scale into
  the A-tile load (`bf16(float(fp8) * block_scale)`) so the WMMA
  chain sees pre-scaled values and the output write is a plain copy.
  One extra VALU per FP8 element, no coopmat scalar-multiply needed.
* Architectural pre-condition: BM=64 divides block_n=128 AND wg_m is
  BM-aligned, so `m_block_base = wg_m / block_n` is constant per
  workgroup. The entire BM×BK A-tile lies within one (n_block,
  k_block) cell, so a single scalar covers all 1024 FP8 elements per
  K-iter.
* `Fp8BlockwiseGemmPushConstants` (36 B = 9 × u32) in `pipeline.rs`.
* `fp8bwgemm_*` pipeline triple (mirror of fp8bw GEMV with the new
  push range size), `run_gemm_fp8_blockwise` method.
* Replaces the GEMV-loop fallback at all 7 GEMM call sites in
  `dispatch_layer_batch`. The loop helper stays in tree
  (`#[allow(dead_code)]`) for shapes where BM∤block_n / BK∤block_k.

### Loader auto-detection

The SafeTensors loader now recognises all three FP8 scaling strategies
and routes each to the right GEMV/GEMM pipeline without any
configuration:

| On-disk suffix          | Scale shape                | Strategy      | Routed kernel                              |
|-------------------------|----------------------------|---------------|--------------------------------------------|
| `.weight_scale`         | `[]` / `[1]`              | per-tensor    | `run_gemv_fp8_perchannel` (broadcast)      |
| `.weight_scale`         | `[N]` or `[N, 1]`          | per-channel   | `run_gemv_fp8_perchannel`                  |
| `.weight_scale_inv`     | `[N/bn, K/bk]`             | block-wise    | `run_gemv_fp8_blockwise` (decode) + `run_gemm_fp8_blockwise` (prefill) |

Scale dtype: BF16 / F16 / F32 — all three handled. The `_inv` suffix
is just an upstream naming convention (DeepSeek/Qwen3 ship the
precomputed inverse); the dequant math is `weight × scale` either way.

### Reference models

| Model                                 | Format            | VRAM     | Decode    | pp=512    |
|---------------------------------------|-------------------|---------:|----------:|----------:|
| `neuralmagic/Meta-Llama-3.1-8B-FP8`   | per-tensor FP8    |  7.5 GB  |  69 t/s   |  757 t/s  |
| `larryvrh/Qwen2.5-14B-Instruct-FP8`   | per-channel FP8   | 13.8 GB  |  14 t/s   |  338 t/s  |
| **`Qwen/Qwen3-8B-FP8`**               | **block-wise [128,128] FP8** | **~8.5 GB** | **64 t/s** | **770 t/s** |

### Mesa timeout fix

Intermittent `ring gfx_0.0.0 timeout` on Mesa 26.1-rc3 with large
compute command buffers (~1200 dispatches) was caused by the 2-second
default amdgpu compute timeout. Our 14B prefill at pp=1024 takes
~2.9 seconds. Fix: `amdgpu.lockup_timeout=10000,10000` kernel
parameter. With the timeout fix Mesa 26.1-rc3 lands within ±5 % of
26.0.6 on FP8 prefill — not an in-driver regression. Credit:
Pierre-Eric Pelloux-Prayer (Mesa).

### Benchmark comparison (carried forward from v0.3.7 sprints)

Full VulkanForge vs llama.cpp (Vulkan + ROCm) benchmark with power
measurement at `results/v038_bench_comparison.md`. Headline: VF wins
decode `tok/s/W` 1.6–1.9 × across every directly comparable 8B
configuration, both vs llama.cpp ROCm and vs llama.cpp Vulkan.

* vLLM 0.20.1 ROCm comparison added: VF 2.1× faster on single-user
  decode (64.5 vs ~30 tok/s) and 5.1× on single-prompt prefill
  (770 vs ~150 tok/s) on Qwen3-8B-FP8. vLLM lacks tuned kernel
  configs for gfx1201 ("Using default W8A8 Block FP8 kernel
  config") and runs `--enforce-eager` (no CUDAGraphs); numbers
  reflect out-of-the-box experience on RDNA4 consumer hardware,
  not vLLM's best case.

### RDNA4 ISA Analysis + Mesa RFEs

* Sprint 32B: RDNA4 ISA deep-read identified native FP8 WMMA and
  `GLOBAL_LOAD_TR_B64` as blocked by RADV (not application-fixable).
* Sprint 34: confirmed ACO does not emit `GLOBAL_LOAD_TR` for
  `coopMatLoad` from SSBO. Test artefacts at
  `tests/sprint34_load_tr/` — public-facing for upstream RFE.

### SPV count

104 → 106 SPIR-V shaders (`mul_mat_vec_fp8_blockwise` +
`mul_coopmat_fp8_bn32_blockwise`).

### Reports

* `results/v038_sprint35_blockwise_fp8.md` — Sprint 35 decode-only.
* `results/v038_sprint36_blockwise_gemm.md` — Sprint 36 BN=32 GEMM.
* `results/v038_bench_comparison.md` — VF vs llama.cpp v0.3.7 bench
  with power.

### Known limitations

* **14B FP8 decode: 14 tok/s.** Runtime-state interaction confirmed
  by RGP — not fixable at application layer.
* **Block-wise FP8 GEMM constraints:** `block_n % BM == 0` and
  `block_k % BK == 0` (with BM=64, BK=16). Qwen3 and DeepSeek-V3
  `[128, 128]` satisfy both. Models with smaller blocks would fall
  back to the GEMV-loop helper.
* **FP8 SafeTensors models require `--tokenizer-from`** pointing at
  a GGUF from the same model family — SafeTensors models don't ship
  the GGUF tokenizer that VF's BPE consumes.
* **VF bench accepts only Q4_K_M GGUF** — `vulkanforge bench` rejects
  Q8_0 / other GGUF quants. FP8 SafeTensors are accepted via
  `--tokenizer-from`.
* **`VULKANFORGE_ENABLE_FP8=1` still required** to enable the
  `VK_EXT_shader_float8` device extension at startup.

### Commit trail

```
Sprint 34B  c979e7e  Organize Mesa LOAD_TR test artefacts
Sprint 34C  results/v038_bench_comparison.md  VF vs llama.cpp ROCm + Vulkan
Sprint 34D  results/v038_bench_comparison.md  Vulkan-vs-Vulkan, 15-prompt
Sprint 35   c41767c  Block-wise FP8 decode (Qwen3-8B-FP8, 64.5 tok/s)
Sprint 36   1aa6521  Block-wise FP8 GEMM (pp=512: 157 → 770 tok/s, +390 %)
```

---

## v0.3.7 — FP8 prefill +113% (2026-05-07)

### Headline

**14B Qwen2.5-FP8 prefill at pp=512 goes from 159 to 338 tok/s.**
First real performance sprint since Sprint 25B. The cooperative-
matrix FP8 GEMM kernel was running at 7-18% TFLOPS efficiency;
BN=32 tiling doubles the WMMA work per activation read and brings
efficiency to ~30%. Decode and GGUF unchanged.

### FP8 Prefill Performance (RX 9070 XT, Mesa 26.0.6)

| Model           | pp=64  | pp=128 | pp=512 | pp=1024 |
|-----------------|-------:|-------:|-------:|--------:|
| **14B v0.3.7**  | **270**| **313**| **338**| **331** |
| 14B v0.3.5      | 184    | 186    | 159    | crash*  |
| **Δ**           | +47%   | +68%   |**+113%**| fixed  |
| **8B v0.3.7**   | **661**| **702**| **757**| **777** |
| 8B v0.3.5       | 567    | 620    | 699    | 712     |
| **Δ**           | +17%   | +13%   | +8%    | +9%     |

\* The pp=1024 crash on the BN=16 baseline was an intermittent Mesa
26.0.6 ring timeout. BN=32 halves `groups_y` and stays below the
queue threshold that triggers the timeout.

Decode unchanged (14.6 / 73 tok/s on 14B / 8B FP8). GGUF Qwen3-8B
Q4_K_M unchanged (118 tok/s decode, 3871 tok/s prefill at pp=512).

### What changed

* `vk_shaders/mul_coopmat_fp8_bn32.comp` (new, ~200 LOC) — BN=32 /
  BLOCK_SIZE=512 / 4×2 M×N subgroup grid / 64×32 output tile per
  WG. Same SPV-time FP8 → BF16 narrow as `mul_coopmat_fp8_multi_wg`,
  same WMMA 16×16×16, same descriptor-set + push-constant layout —
  just larger tiles and four times more WMMA per loaded activation
  byte. Default ON for all FP8 prefill GEMMs at `m ≥ 64 && n ≥ 64`.
  Opt-out via `VF_FP8_GEMM_BN32=0`.
* `vk_shaders/mul_coopmat_fp8_bn64.comp` (new, Sprint 33) — BN=64 /
  BLOCK_SIZE=1024 (RDNA4 max workgroup) / 4×4 M×N subgroup grid /
  64×64 output tile. Honest-negative experiment: ~3% slower than
  BN=32 at pp=512 due to LDS-occupancy drop (5→3 WGs/CU). Stays in
  tree as opt-in via `VF_FP8_GEMM_BN=64` for future tuning.
* Three-way GEMM routing in `forward.rs::run_gemm_fp8_naive`:
  BN=64 (opt-in via env) → BN=32 (default) → BN=16 multi_wg
  (legacy fallback) → 16×16 single-tile naive (small-M fallback).
  Override via `VF_FP8_GEMM_BN={16,32,64}`.
* `examples/fp8_prefill_shape_bench.rs` (new, Sprint 32 Phase 0) —
  per-shape per-pp prefill profiler. Loads SafeTensors FP8 model,
  attaches `ShaderProfiler`, runs `prefill_batch` at varying
  `seq_len ∈ {64,128,256,512}`, prints per-call wall-time and
  effective TFLOPS efficiency. Permanent diagnostic tool.
* +2 SPIR-V shaders (`mul_coopmat_fp8_bn32_v2.spv`,
  `mul_coopmat_fp8_bn64_v2.spv`).

### RDNA4 ISA Deep-Read (Sprint 32B, no code change)

Read of the AMD RDNA4 ISA reference produced a ranked list of
hardware features for future FP8 sprints:

* **P0:** Native FP8 WMMA (`V_WMMA_F32_16X16X16_FP8_FP8`,
  `V_WMMA_F32_16X16X16_BF8_BF8`) — gates on Mesa/RADV exposing the
  cooperative-matrix FP8 types. Currently FP8 is software-narrowed
  to BF16 before WMMA, costing throughput.
* **P1:** `GLOBAL_LOAD_TR_B64` — hardware transpose load. Could
  remove the LDS staging step for the activation tile.
* **P2:** WMMA hazard restructuring (V_NOP elimination, reorder of
  back-to-back WMMA-with-shared-accumulator).
* Questions filed on AMD Developer Forum and Phoronix Vulkan Forum.

### Why this won where Sprints 27-31 didn't

Three previous optimisation sprints (29B harness, 30 DS-cache,
31 SubgroupAdd) all landed 0% wall-clock change despite shipping
real architectural improvements. They each targeted **per-call CPU
overhead** (descriptor allocation, LDS occupancy ceiling). Sprint 32
targeted **per-call GPU work** — the amount of compute done per byte
of activation read. That is the right abstraction layer for a kernel
that is BW-saturated on weight reads but underutilising the WMMA
tensor cores: arithmetic intensity is too low. BN=16 → BN=32
doubles the WMMA work per loaded activation tile; at the previous
7-18% TFLOPS efficiency there was ample headroom to absorb that.

### Architecture (carries forward from v0.3.6)

v0.3.6 was a cleanup release; all of it is in v0.3.7:

* **SubgroupAdd reduction** in all 4 GEMV shaders (Sprint 31) —
  per-WG LDS 4096 → 0 B, theoretical occupancy 6/16 → 16/16
  wavefronts/SIMD, 6 `barrier()` calls eliminated per dispatch.
* **fp8pc descriptor-set cache** (Sprint 30) — pool 524 288 → 1024
  sets, −33 MiB descriptor metadata.
* **lm_head harness pipeline** with `VF_LMHEAD_HARNESS=0` A/B toggle.

### Known limitations

* 14B FP8 decode at 14 tok/s (~30% bandwidth efficiency) — runtime-
  state interaction confirmed by RGP at the wavefront level. Mesa
  bug report filed; not addressable at the application layer.
* 14B FP8 prefill at 338 tok/s — ~30% TFLOPS efficiency. Further
  gains need native FP8 WMMA + GLOBAL_LOAD_TR support in Mesa/RADV.
* BN=64 is ~3% slower than BN=32 (occupancy-limited at
  BLOCK_SIZE=1024 = RDNA4 max). Kept as opt-in.
* Block-wise FP8 (Qwen3-FP8 format, block_size=128) not yet
  supported. Per-tensor and per-channel are.
* FP8 SafeTensors models still require `--tokenizer-from <gguf>`.

### Commit trail

```
sprint32 phase 0: per-shape FP8 prefill diagnostic
sprint32 phase 1: BN=32 FP8 GEMM (+113% on 14B pp=512)
sprint33:         BN=64 FP8 GEMM opt-in (honest-negative, ~3% slower)
v0.3.7:           release roll-up
```

## v0.3.6 — architecture cleanup (2026-05-07)

### Headline

Cleanup release on top of v0.3.5. Three optimization sprints
(Sprints 29B / 30 / 31) confirmed that the 14B Qwen2.5-FP8 decode
gap (14 tok/s, ~30% of theoretical bandwidth) is **not at the
application layer** — every plausible per-call / per-dispatch
optimization moves resource usage in the right direction without
moving the wall clock. RGP analysis (Sprint 29B) showed the kernel
runs at peak BW at the wavefront level. A Mesa bug report has been
filed upstream; the runtime gap is monitored, not chased further at
the application level.

What did ship:

* **SubgroupAdd reduction** replaces LDS-tree in all four GEMV
  shaders (Sprint 31):
  - `mul_mat_vec_f16.comp` (lm_head)
  - `mul_mat_vec_f32.comp` (GGUF lm_head)
  - `mul_mat_vec_fp8.comp` (8B FP8 per-tensor)
  - `mul_mat_vec_fp8_perchannel.comp` (14B FP8 per-channel)

  Per-WG LDS: 4096 B → **0 B**.
  Theoretical occupancy: 6/16 → **16/16** wavefronts/SIMD
  (now VGPR-limited at maximum).
  Eliminates 6 `barrier()` calls per dispatch.
  Wave64 + REQUIRE_FULL_SUBGROUPS + BLOCK_SIZE=64 means one
  workgroup == one subgroup, so `subgroupAdd` is the workgroup-wide
  sum; `subgroupElect()` replaces `if (tid == 0)` for the
  single-writer predicate.

* **fp8pc descriptor-set cache** (Sprint 30): keyed on the
  `(weight, input, output, scale)` buffer tuple, mirrors the
  production `alloc_or_get_set` pattern. After a 2-token warmup
  (one per async slot), every per-channel FP8 GEMV is a 100%
  cache hit. Sprint 25B's 524 288-set pool was a workaround for
  the no-cache, every-call-fresh-alloc pattern; with caching the
  pool only needs to hold unique keys (~672 worst case for
  Qwen2.5-14B), so it shrinks to **1024 sets**.

  VRAM saved: ~33 MiB → ~50 KiB descriptor metadata.

* **lm_head harness pipeline** stays in the codebase as permanent
  A/B infrastructure: the dedicated `lmhead_*` resources
  (PipelineCache::null + dedicated DSL/pool) from Sprint 29 are
  toggleable via `VF_LMHEAD_HARNESS=0`, useful for any future
  sprint that needs to attribute a perf delta between the runtime
  registry path and a fresh harness path.

### Diagnostic tools added

* `examples/cb_backpressure_test.rs` — N dummy GEMVs preceding
  a timed lm_head dispatch in one CB (Sprint 28).
* `examples/vram_pressure_test.rs` — lm_head shape GEMV with
  configurable VRAM ballast, supports `VF_BALLAST_FIRST=1` to test
  high-VRAM-offset placement (Sprints 28B + 29).

### Performance

No wall-clock change from v0.3.5. Decode rates remain:

| Config | v0.3.5 | v0.3.6 |
|---|---:|---:|
| 14B FP8 (Qwen2.5) | 14.1 tok/s | 14.3 tok/s |
| 8B FP8 (Llama-3.1) | 68.1 tok/s | 70.1 tok/s |
| 8B GGUF Q4_K_M (Llama-3.1) | ~118 tok/s | 113 tok/s |
| 8B GGUF Q4_K_M (Qwen3) | ~118 tok/s | 111 tok/s |

All deltas are within run-to-run noise (±2 tok/s). 15/15 coherence
on every config.

### Known limitations (updated)

* 14B FP8 decode: 14.3 tok/s (~30% BW efficiency). **Bottleneck
  confirmed below the application layer** by 13 ruled-out
  hypotheses across Sprints 26-31, including the RGP wavefront
  analysis. A Mesa bug report covers the most plausible upstream
  trigger (`results/mesa_bugreport_info.md`); revisit when a
  fix lands.
* 14B FP8 prefill at `pp=1024` is intermittent on **Mesa 26.1.0-rc3**
  (one observed `VK_ERROR_DEVICE_LOST`, recoverable via ring
  reset). **Mesa 26.0.6 stays the recommended driver.**
* Block-wise FP8 (Qwen3-FP8 format, block_size=128) — not supported.
* RoPE scaling — detected but not applied; positions >8K may drift.
* FP8 SafeTensors models require `--tokenizer-from <gguf>` (no
  native `tokenizer.json` parser yet).

**Removed from v0.3.5's known-limitations list:**
* `fp8pc_pool` 524 288-set workaround — replaced by the DS cache
  (Sprint 30).

### Commit trail

```
30   fp8pc descriptor-set cache (cleanup, no perf gain)
31   subgroupAdd replaces LDS-tree reduction (cleanup)
```

---

## v0.3.5 — first 14B FP8 on a 16 GiB consumer GPU (2026-05-06)

### Headline

**Qwen2.5-14B-Instruct-FP8 runs coherently on a 16 GiB consumer GPU.**
13.77 GiB VRAM, 14.1 tok/s decode, 169 tok/s prefill @ pp=512,
15/15 on the canonical coherence suite (smoke / code / prose /
reasoning / context-stress / numerics / emoji-tokenizer). Including
exact arithmetic (`17 × 23 = 391`), C++ thread-safe LRU cache
generation, Go REST-API skeleton, German narrative prose, and emoji
identification. **No other open-source inference engine ships this
configuration today.**

Adds per-channel FP8 quantization (SSBO scale-vector kernels for
both GEMV decode and GEMM prefill paths), Qwen2 architecture support
(Q/K/V bias-add in all three dispatch paths, ChatML template
detection via tokenizer.chat_template, rope_theta plumbed from HF
`config.json`), and a logits-buffer architecture refactor (GpuOnly
+ host-mapped staging copy via `record_logits_readback`).

### Native FP8 performance

| Model + quant                      | VRAM      | Decode    | Prefill @ pp=512 | Coherent |
|------------------------------------|----------:|----------:|-----------------:|---------:|
| Meta-Llama-3.1-8B-Instruct-FP8     |  7.48 GiB | 68.1 t/s  |   424 t/s        |  15/15   |
| Qwen2.5-14B-Instruct-FP8           | 13.77 GiB | 14.1 t/s  |   169 t/s        |  15/15   |

8B FP8 sits at ~79% of the theoretical bandwidth limit. 14B FP8
sits at ~30% of the BW limit — the per-channel GEMV path
(`run_gemv_fp8_perchannel`) is correctness-first, isolated from
the shared `PipelineRegistry` and descriptor-set cache by design
(Sprint 24-Inline). Re-merging into the registry with
`BindingSignature` set-caching is the v0.3.6 target; Sprint 26's
profile estimates +8 tok/s from that change alone.

A second perf wart on 14B: the `lm_head` GEMV measures ~30 ms in
the runtime decode CB but ~2.7 ms in standalone benchmarks for the
exact same dispatch (Sprints 27–29B). Twelve hypotheses ruled out
across five sprints; RGP analysis (Sprint 29B) confirms the kernel
runs identically at the wavefront level on 14B vs 8B. The cause is
a runtime-state interaction we haven't yet isolated. Deferred to
v0.3.6+ once the per-channel GEMV optimization lands and reshapes
the per-token cost.

### Multi-architecture additions

* `model_type='qwen2'` accepted by the SafeTensors loader (Sprint 24C).
* HF tensor-name remap: `self_attn.{q,k,v}_proj.bias → attn_{q,k,v}.bias`.
* `dispatch_layer` now runs Q/K/V bias-add between the GEMV writes
  and RoPE / QK-norm (Sprint 25 — fixed the post-Sprint-24B regression
  where bias was missing on the production decode path).
* ChatML template detected via the GGUF `tokenizer.chat_template`
  string match on `<|im_start|>` (Sprint 25).
* `rope_theta` from HF `config.json` flows through to
  `cfg.rope_freq_base` (no hardcoded 5e5; Qwen2.5's 1e6 reaches the
  rope shader correctly).

### GGUF performance (unchanged from v0.3.4)

All 5 GGUF Q4_K_M / Q3_K_M models still beat llama.cpp Vulkan
decode by 1.04–1.06×. No GGUF-path changes in v0.3.5.

### Infrastructure changes

* Per-channel FP8 weight scale: SSBO scale-vector binding (binding 3)
  on the dedicated `mul_mat_vec_fp8_perchannel.spv` variant; loader
  broadcasts per-tensor scalars to per-row vectors for shape-uniform
  GEMV dispatch (Sprint 24A).
* Per-channel FP8 GEMV: dedicated pipeline + descriptor pool +
  pipeline layout + DSL on `Forward`, built with `PipelineCache::null`
  in `Forward::new` (Sprint 24-Inline harness pattern). Pool sized
  524 288 sets (~25 MiB) to absorb async-decode set accumulation.
* Q/K/V bias-add: broadcast-aware `run_bias_add` helper, dispatched
  in `dispatch_layer` (production decode), `dispatch_layer_partial`
  (debug), and `dispatch_layer_batch` (prefill) — guarded by
  `attn_*.bias` presence so Llama / Qwen3 / Mistral arches no-op.
* `logits_buf` refactored: `MemoryLocation::GpuToCpu` →
  `MemoryLocation::GpuOnly` + new `logits_staging` (GpuToCpu) and
  `record_logits_readback` helper. Cleaner separation of GPU compute
  vs host-readable transfer (Sprint 27).
* Sprint 29 lm_head harness: dedicated `lmhead_*` resources on
  `Forward` (mirrors Sprint 24-Inline's fp8pc_* pattern). Toggle
  via `VF_LMHEAD_HARNESS=0` to fall back to the production registry
  for A/B comparison. Pool sized 524 288 sets to match fp8pc_pool.
* 103 SPIR-V shaders (was 102 in v0.3.4) — adds
  `mul_mat_vec_fp8_perchannel.spv` for the dedicated harness path.
* 37 lib tests + FP8 GEMV/GEMM correctness tests, all green.

### Diagnostic tools added (permanent, kept in `examples/`)

* `examples/profile_decode_safetensors.rs` — per-dispatch GPU
  TIMESTAMP profile for SafeTensors FP8 models. Activates the
  fully-instrumented `ShaderProfiler` that's already plumbed
  through every dispatch site.
* `examples/gemv_f16_shape_bench.rs` — F16 GEMV shape sweep
  (M, K matrix) with TIMESTAMP, fresh pipeline, fresh allocator.
* `examples/cb_backpressure_test.rs` — N dummy GEMVs preceding
  the timed lm_head dispatch in one CB; rules out command-processor
  backpressure as a cause of the runtime gap.
* `examples/vram_pressure_test.rs` — lm_head shape GEMV with
  configurable VRAM ballast (allocate-first or allocate-after);
  rules out VRAM occupancy and weight-buffer placement effects.
* `examples/fp8_gemv_standalone.rs` — isolated per-channel FP8 GEMV
  reproducer against real Llama-3.1-FP8 weights (Sprint 24-Harness).

### Known limitations

* 14B FP8 decode: 14.1 tok/s (~30% BW efficiency). Per-channel GEMV
  path is correctness-first, not yet optimized. v0.3.6 target:
  ~30 tok/s after registry re-integration.
* 14B FP8 prefill: 169 tok/s (vs 8B's 424 tok/s). Same kernel-maturity
  gap. Optimization deferred.
* `lm_head` runtime gap: 30 ms vs 2.7 ms standalone — root cause not
  isolated despite five sprints of diagnostic. Detailed
  hypothesis-tracker in `results/v035_sprint29b_rgp_analysis.md`.
* Block-wise FP8 (Qwen3-FP8 format, block_size=128) — not supported.
* RoPE scaling — detected but not applied; positions >8K may drift.
* FP8 SafeTensors models require `--tokenizer-from <gguf>` (no
  native `tokenizer.json` parser yet).
* `fp8pc_pool` and `lmhead_pool` sized at 524 288 sets each
  (~25 MiB combined) — workaround for async-decode set
  accumulation, not yet replaced by per-slot sub-pools or
  `vkFreeDescriptorSets`.

### Commit trail

```
24A         per-channel FP8 GEMM scale buffer
24B         Q/K/V bias-add (Qwen2 attention biases)
24C         accept model_type='qwen2'
24-Fix      / Bisect — DSL collision bisect (negative)
24-Harness  standalone per-channel FP8 GEMV reproducer (PASS @ 0.11%)
24-Inline   per-channel FP8 GEMV via dedicated harness path (FIX shipped)
25          decode bias-add fix → 14B FIRST COHERENT TOKEN
25B         15-prompt benchmark suite — 15/15 on both FP8 models
26          per-dispatch GPU TIMESTAMP profile (data-only)
27          lm_head harness diagnostic (honest negative, arch cleanup)
28          CB-backpressure test (negative)
28B         VRAM-pressure + disk pipeline-cache test (negative)
29          lm_head harness pattern + ballast-first (negative)
29B         RGP capture + analysis — kernel identical at wavefront level
```

---

## v0.3.4 — native FP8 LLM, multi-submit prefill, Q3_K coopmat (2026-05-03)

### Headline

**First Vulkan engine to run a complete FP8 LLM end-to-end.**
`Meta-Llama-3.1-8B-Instruct-FP8` (HuggingFace SafeTensors,
`compressed-tensors` per-tensor) loads natively without
unpacking, runs greedy-coherent chat, and fits a 16 GiB
consumer GPU at:

| Metric                | Value     |
|-----------------------|----------:|
| GPU footprint         | 7.48 GiB  |
| Decode                | 68.5 t/s  |
| Prefill @ pp=512      | 695 t/s   |
| Coherence (greedy)    | "The answer to 2+2 is 4." (bit-identical across sprints) |

Carries forward all v0.3.3 GGUF Q4_K_M / Q3_K_M decode wins —
VulkanForge still beats llama.cpp Vulkan decode on 4 / 5 configs
(FP16 KV) and 5 / 5 configs (FP8 KV).

### Sprint 19A — Q3_K + Q5_K coopmat prefill

Ported the Q4_K coopmat shader pattern (`mul_coopmat_q4k_naive` +
the dequant scaffolding in `dequant_head.glsl`) to Q3_K and Q5_K.
Q3_K_M GGUFs now hit the WMMA prefill path instead of falling
back to scalar mmq, and Q5_K_M is wired for any future model
that ships in that quant.

### Sprint 19A correctness fix — example binary rebuild

`cargo build --release` does not relink `target/release/examples/*`
after a `src/` edit. Bench numbers were 2 274 → 3 536 tok/s after
adding `--examples` to the build command. Captured as a memory
note for future sprints.

### Sprint 19B — compute-graph plan analysis (research)

Read llama.cpp's `ggml-vulkan.cpp` (~17 000 LOC) to identify the
remaining levers behind its prefill lead. Conclusion: buffer
aliasing is **not** the gap; the levers are **multi-submit
pacing** (split the prefill graph into sub-submits) and **fusion**
(KV-write fusion, post-attn-add+norm fusion). Multi-submit became
Sprint 19B-A; KV-write fusion was tested in 19B-B.

### Sprint 19B-A — multi-submit prefill pacing

Mirrored llama.cpp's "submit every 100 graph nodes" pattern. The
prefill `Forward::prefill_batch` path now allocates a small
`prefill_pool` of command buffers + a `prefill_fence`, builds
`layers_per_submit` chunks of layers per CB, submits, lets the
GPU start, builds the next chunk in parallel. **Neutral on
Q4_K_M prefill** (already CPU-record-hidden for that quant), but
the infrastructure is the prerequisite for any future graph-level
fusion work.

### Sprint 19B-B — KV-write fusion (honest negative)

Tried to fuse `kv_copy_fp16` (or `kv_store_fp8`) into the QKV-proj
GEMM dispatch. Negative result on RDNA4: the KV-write is already
~0.1 % of prefill wall-time; fusion adds shader-source
complexity for unmeasurable gain. Branch reverted, report
preserved (`results/v034_sprint19b_b_kv_fusion.md`).

### Sprint 20 — Native FP8 LLM Support (3 milestones)

End-to-end FP8 SafeTensors → native FP8 GEMV → end-to-end FP8
chat:

- **20-M1** — HuggingFace SafeTensors loader. Single-file +
  sharded (`*.safetensors.index.json`); `compressed-tensors`
  quantization config support; per-tensor scale extraction;
  `hf_to_vf_name()` mapping. Supports F8E4M3, F8E5M2, F16, BF16,
  F32 dtypes. New: `src/safetensors.rs` (~310 LOC), new:
  `src/hf_config.rs` (~150 LOC).
- **20-M2** — `mul_mat_vec_fp8.comp` GEMV decode kernel. Reads
  weights as `floate4m3_t` via `uintBitsToFloate4m3EXT`, accumulates
  into FP32, post-multiplies by `weight_scale` push-constant.
  Same dispatch shape as the F32 GEMV; binding count matches the
  FP32 / FP16 GEMVs after spirv-opt strips unused fuse dummies.
- **20-M3** — End-to-end FP8 chat for
  `Meta-Llama-3.1-8B-Instruct-FP8`. `vulkanforge chat
  --tokenizer-from <gguf-or-hf-repo>` borrows the tokenizer from
  a sibling GGUF or HF repo (the FP8 SafeTensors release ships
  weights without `tokenizer.json`). Greedy output: "The answer
  to 2+2 is 4."

### Sprint 20-GEMM + 20-Wire — FP8 GEMM prefill

`mul_coopmat_fp8_naive.comp` does the FP8 GEMM prefill via
the BF16-narrow-fragment WMMA path
(`VK_KHR_cooperative_matrix` 16×16×16 BF16, FP32 accumulator).
Layout: A = `[M×K]` row-major FP8, B = `[N×K]` row-major FP32,
C = `[N×M]` row-major FP32. Sprint 20-Wire wired this kernel
into `dispatch_layer_batch` with an `is_fp8_layer_weight`
branch; FP8 prefill went from 8.7 s (FP8→FP32 dequant +
F32 GEMM fallback) to 695 tok/s @ pp=512.

### Sprint 21A — Aligned-load FP8 GEMM

Reshaped the inner loop to read 4 contiguous FP8 bytes per A-side
thread (1 uint32 / thread) and 4 contiguous K-positions per B-side
thread (vec4 / thread). Cuts memory-instruction count, raising
single-tile FP8 GEMM throughput at typical prefill shapes.

### Sprint 21B — Multi-WG FP8 GEMM

`mul_coopmat_fp8_multi_wg.comp` — 4 Wave64 subgroups per WG, 64×16
output tile, adapted from the BF16 `mul_coopmat_bf16` BN=16 mode.
Gated on `m ≥ 64 && n ≥ 64` to prevent regression at small pp
(first run regressed −10 % at pp=28 when ungated).

### Sprint 21C — bench --tokenizer-from

`vulkanforge bench --tokenizer-from <hf-or-gguf>` borrows a
tokenizer for a SafeTensors model so the bench harness can run
pp-sweeps on FP8 builds with no on-disk `tokenizer.json`.

### Sprint 22 — --max-context CLI flag

`--max-context <N>` on `chat` and `bench` overrides the model's
default KV-cache capacity, so long-context prefill (pp > 4096) is
selectable from the CLI without source edits.

### Sprint 22B — Skip embed_tokens GPU upload (−1.96 GiB)

When `output.weight` (`lm_head`) is present in the SafeTensors
map and the config does not tie embeddings, the `token_embd.weight`
GPU upload is skipped — the embedding lookup runs from the
host-side Vec<f32> already loaded for the prefill warm-up. Saves
1.96 GiB on Llama-3.1-FP8 (10.42 → 8.46 GiB).

### Sprint 22C — FP16 lm_head (−0.98 GiB, +9 % decode)

`lm_head.weight` is now narrowed BF16 → FP16 at load time (was
BF16 → FP32 expansion). New shader `mul_mat_vec_f16.comp` reads
weights as `uint[]`, decodes via `unpackHalf2x16`. Halves
lm_head GEMV's VRAM bandwidth and yields a free **+9 % decode**
on top of the −0.98 GiB VRAM win. lm_head bit-identical greedy
output before / after.

Cumulative VRAM saved by Sprints 22B + 22C: **−2.94 GiB
(−28.2 %)** on Llama-3.1-FP8.

### Sprint 23 — Qwen2.5-14B FP8 (honest negative)

Pre-check stopped this sprint. `larryvrh/Qwen2.5-14B-Instruct-FP8`
uses **per-channel** weight scaling (`strategy: "channel"`),
incompatible with VF's per-tensor FP8 GEMV/GEMM kernels. Bias-add
without per-channel support would land an unusable loader path.
No code committed; model preserved at
`~/models/Qwen2.5-14B-Instruct-FP8/` for a future per-channel
sprint. Full report in
`results/v034_sprint23_qwen25_14b_fp8.md`.

VRAM viability for 14B FP8 confirmed: ~14.5 GiB total fits 16 GiB
once per-channel scale buffers + bias-add land.

### Stats

- **102 SPIR-V pipelines** (was 87 in v0.3.3; +15 in v0.3.4:
  FP8 GEMV + 3 FP8 GEMM variants + Q3_K/Q5_K coopmat S/M/L tiles
  + FP16 lm_head GEMV + supporting variants)
- **37 lib tests** + 40+ GPU correctness tests (was 32 lib in
  v0.3.3; +5 SafeTensors / hf_config / FP8 GEMM helper tests)
- ~1 800 LOC net delta over v0.3.3 across the 19A / 19B-A / 20
  (M1+M2+M3+GEMM+Wire) / 21A / 21B / 21C / 22 / 22B / 22C
  sprints (Sprint 23 added 0 LOC; only an honest-negative report)
- Llama-3.1-8B-FP8: **7.48 GiB GPU, 68.5 tok/s decode,
  695 tok/s prefill @ pp=512**

## v0.3.3 — native FP8 KV cache + benchmark dominance (2026-05-03)

### Headline

**Faster than llama.cpp Vulkan on every tested model+quant
configuration.** With `VULKANFORGE_KV_FP8=1`, all 5 / 5 configs
beat llama.cpp's tg128; even on the default FP16-KV path,
4 / 5 still win.

| Model + quant                       | VF FP16-KV | VF FP8-KV  | llama.cpp Vulkan | best / lc.cpp |
|-------------------------------------|-----------:|-----------:|-----------------:|--------------:|
| Qwen3-8B Q3_K_M                     |    131.7   |  **133.7** |     128.7        |   **1.04 ×**  |
| Mistral-7B-Instruct-v0.3 Q4_K_M     |    130.0   |  **131.8** |     124.2        |   **1.06 ×**  |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M |    121.2   |  **122.9** |     117.7        |   **1.04 ×**  |
| Meta-Llama-3.1-8B-Instruct Q4_K_M   |    121.4   |  **122.8** |     117.6        |   **1.04 ×**  |
| Qwen3-8B Q4_K_M                     |    116.9   |  **118.5** |     113.1        |   **1.05 ×**  |

Bench: RX 9070 XT, RADV Mesa 26.0.6; llama.cpp build 23b8cc4.
Full report: `results/v033_comprehensive_benchmark.md`.

### Native FP8 KV cache (Sprint 18A)

VulkanForge is the first Vulkan LLM engine with **native FP8 E4M3
KV cache** via `VK_EXT_shader_float8`. Sibling shader of
`kv_copy_fp16.comp` packs 4 FP8 lanes per `uint`; all five
attention shaders (`flash_attn`, `flash_attn_split`,
`flash_attn_batch`, `flash_attn_tiled`, `flash_attn_coopmat`)
gained `#if FP8_KV` branches that decode bytes via
`uintBitsToFloate4m3EXT`.

| Model         | FP16 KV  | FP8 KV   | VRAM saved | Decode bonus |
|---------------|---------:|---------:|-----------:|-------------:|
| Qwen3-8B      |   288 MB |  144 MB  |    −50 %   |    +1.4 %    |
| Llama-3.1-8B  |   256 MB |  128 MB  |    −50 %   |    +1.2 %    |
| Mistral-7B    |   256 MB |  128 MB  |    −50 %   |    +1.4 %    |

Quality: 15 / 15 coherent on `run_15prompt_bench` (matching FP16
baseline), multi-turn KV recall verified. Enable:
`VULKANFORGE_KV_FP8=1`.

### FP8 device infrastructure (Sprint 18-M0 / 18-M1)

- **Sprint 18-M0** — Go/No-Go smoke test. Verified RADV exposes
  4 FP8 cooperative-matrix entries (E4M3/E5M2 × E4M3/E5M2 → FP32,
  M=N=K=16) and glslang 16.2 compiles `GL_EXT_float_e4m3` with
  the `Float8EXT` SPIR-V capability declared.
- **Sprint 18-M1** — 70 LOC raw FFI shim (`fp8_ext.rs`) for
  `VK_EXT_shader_float8` since ash 0.38 doesn't ship the bindings
  yet. FP8 E4M3 round-trip and FP8 cooperative-matrix matmul both
  bit-exact vs CPU reference (max_err = 0.0). Opt-in via
  `VULKANFORGE_ENABLE_FP8=1`. Replace with ash bindings once
  ash 0.39+ ships.

### FP8 prefill characterisation (Sprint 18B — honest negative)

FP8 WMMA throughput on RDNA4 + RADV is **~1.18 × BF16, not 2 ×**
(measured at 4096³, 137 GFLOPs, 3-trial median). The KHR
cooperative-matrix fragment shape stays at K=16 for FP8 (per
property table — Sprint 18-M0 enumeration), so there's no
K-throughput advantage; only LDS-bandwidth saving remains, and
that's not the bottleneck. Integration cost (~600 LOC + 48 SPV
variants for `mul_mm.comp` deep refactor + 4-quant matrix +
activation conversion) doesn't clear the bar for a 1.18 × ceiling.

Decision: don't wire FP8 prefill to production. Infrastructure
preserved (Sprint 1B/3C/6A FP8 shaders + bench harness) for
re-test on Mesa update or RDNA-next.

### Stats

- **87 SPIR-V pipelines** (was 81 in v0.3.2; +6 FP8 variants
  for kv_store + 5 attention shaders)
- **32 lib tests** + 40+ GPU correctness tests (was 27 lib in
  v0.3.2; +5 FP8 helper unit tests)
- 15 / 15 coherent on Qwen3-8B Q4_K_M for both FP16-KV (109
  tok/s) and FP8-KV (114 tok/s) baselines
- ~860 LOC net delta over v0.3.2 across the 18-M0 / 18-M1 / 18A /
  18B sprints + comprehensive benchmark

## v0.3.2 — multi-arch + K-quant family (2026-05-03)

### Headline

**Qwen3-8B-Q3_K_M decode 131.1 tok/s = 1.15 × llama.cpp Vulkan.**
Three of four tested configurations now lead llama.cpp:

| Model + quant                     | Decode (tok/s) | vs llama.cpp |
|-----------------------------------|---------------:|-------------:|
| Qwen3-8B-Q3_K_M                   |      **131.1** |   **1.15 ×** |
| Mistral-7B-Instruct-v0.3 Q4_K_M   |          130.0 |       1.14 × |
| Meta-Llama-3.1-8B-Instruct Q4_K_M |          121.1 |       1.06 × |
| Qwen3-8B-Q4_K_M                   |          109.0 |       0.95 × |

Prefill peak 3 865 tok/s @ pp=512 (Qwen3-8B-Q4_K_M) unchanged from
v0.3.0. Async-decode pipeline + coopmat prefill from v0.3.0 carry
forward.

### Sprint 17A — multi-architecture support

Qwen3 / Qwen2.5 / Llama-3.1 / Mistral-7B / DeepSeek-R1-Distill-Llama
all run end-to-end on the same forward pass. Pre-check found that
the loader was already generic over tensor names, RoPE variant was
already arch-selected, Q/K-norm was already gated by tensor presence,
and `ChatTemplate::Llama3 / Mistral / DeepSeekR1` had shipped at
v0.2.x without runtime wiring — one line in `inference_support()`
to add `"llama"` to the arch whitelist + one call to
`ChatTemplate::detect()` from the GGUF metadata unblocked the whole
family. Net source delta: 12 LOC.

### Sprint 17B + 17B-debug — Q3_K shader

Decode GEMV + Mmq prefill shaders for Q3_K, byte-identical to
llama.cpp upstream. Initial 17B ship was broken — Q3_K_M chat
emitted `!!!!!!!!!` despite hitting the bandwidth target. The
debug session ruled out CPU dequant (test added), ruled out the
GEMV decode shader (test added, bit-exact vs CPU), and pinned the
root cause in 17C. Also fixed a latent `MulMmqQ3KL` L-tile dispatch
bug — pipeline_registry pinned BM=128 spec-constants but
`run_gemm` dispatched with bm=64 groups, causing workgroups to
race on output tiles.

### Sprint 17C — Q5_K shader (Q3_K_M unblock + Q5_K_M unlock)

`dump_q3k_m_layer0_quant_types` walked every weight in
`Qwen3-8B-Q3_K_M.gguf` and revealed that `attn_v.weight` and
`ffn_down.weight` are **Q5_K, not Q4_K** as the Sprint 17B brief
assumed. Without a Q5_K shader, those tensors fell through to
`MulMmqQ4K` and read 144-byte blocks out of a 176-byte stride →
garbage compute → uniform-logit collapse. Sprint 17C copied the
upstream `mul_mat_vec_q5_k.comp`, added Q5_K to the Mmq build
defines, ported `dequantize_row_q5_K` to Rust, and wired the
`GgmlType::Q5K` match arm. Result: Q3_K_M coherent at 131 tok/s and
Q5_K_S / Q5_K_M file_types unlocked for free (same shader covers
both bulk-Q5_K models).

### Sprint 17D — Q4_0 shader infrastructure (gated)

Q4_0 the shader ships and is bit-exact vs CPU at every shape
tested. **No Qwen2.5 Q4_0 GGUF runs end-to-end yet** because the
brief's "all weights Q4_0" assumption didn't survive contact with
real GGUFs:

- 7B-Pure: pure Q4_0 — needs Q/K/V bias-add (architectural)
- 7B / 14B: Q4_1 ffn_down — needs a Q4_1 shader sprint
- 0.5B: Q8_0 output.weight — needs a Q8_0 shader sprint

`file_type=2` stays gated out of preflight; the infrastructure is
ready for the future arch-Qwen2.5 sprint. Generic
`mul_mat_vec.comp` + `dequant_funcs.glsl` from upstream copied as
the first non-K-quant GEMV path.

### What's tested

- 27 / 27 lib tests
- 15 / 15 prompts coherent on Qwen3-8B Q4_K_M @ 109 tok/s
  (`run_15prompt_bench`, unchanged from v0.3.1)
- Q3_K_M, Q4_K_M, Llama-3.1, Mistral, DeepSeek-R1-Distill all
  produce coherent reasoning output
- 9 new diagnostic + GPU-correctness tests across Q3_K, Q5_K, Q4_0
- 6 new Mmq parity tests in `tests/correctness.rs` (Q3_K + Q5_K)
- 81 SPIR-V pipelines (was 75 in v0.3.1)

### Limitations

- Qwen2.5 Q4_0 GGUFs gated out of preflight. Cleanest blocker is
  Q/K/V bias-add (every Qwen2.5 variant); Q4_1 and Q8_0 shaders are
  smaller follow-on tasks once biases land.
- Q3_K / Q5_K prefill is Mmq-only (no MulMm coopmat path); Q4_K_M's
  faster prefill comes from the coopmat path. `MulMmQ3K/Q5KCoopmat`
  would close that gap if it matters for these quants.

## v0.3.1 — CLI + auto-detection + sampling polish (2026-05-02)

### Headline

**Packaging release.** v0.3.0's decode breakthrough (109.0 tok/s,
0.95 × llama.cpp Vulkan) and prefill (3865 tok/s pp=512, 0.89 ×)
carry forward unchanged. v0.3.1 ships the binary surface on top of
that engine: a real `vulkanforge` CLI, GGUF auto-detection on every
file in `~/models/`, and a tightened sampling UX. 27 / 27 lib tests,
15 / 15 coherent at decode 109.0 tok/s.

### Sprint 16A — clap CLI port

Refactored `src/main.rs` from an env-var-driven REPL into a
clap-dispatched binary with three subcommands. The chat REPL,
sampling pipeline, and decode hot path are unchanged — `chat`
forwards every flag into the existing `ChatSession`.

```
vulkanforge chat  --model <gguf>   # interactive multi-turn REPL
vulkanforge bench --model <gguf>   # short decode + pp sweep (greedy)
vulkanforge info  --model <gguf>   # GGUF metadata + GPU info
```

Each flag has a `VF_*` env-var fallback (`VF_TEMPERATURE`, `VF_SEED`,
…). `bench` always uses greedy regardless of env state — the
15-prompt and pp-sweep examples remain the canonical performance
harness.

### Sprint 16B — GGUF auto-detection + preflight

`ModelConfig::from_gguf()` was already arch-generic
(`general.architecture` used as a metadata-key prefix), so the gap
was the surface:

- `info` now displays **Quantization** (`general.file_type` →
  Q4_K_M / Q4_0 / Q6_K / …), **Tokenizer** (`gpt2` vs `llama`),
  **Context length**, and a real `Status` line.
- `preflight_supported()` opens the GGUF header (no Vulkan init) and
  short-circuits `chat` / `bench` with a clean error when the
  architecture (only `qwen2` / `qwen3` run end-to-end) or
  quantization (only Q4_K_M) isn't wired through the forward pass.

Sweep across 10 GGUFs in `~/models/`: 9 / 10 produce clean info
output. `gemma-4-E4B` hits an unrelated tensor-type-30 parser limit
in `gguf.rs` (separate concern).

### Sprint 16C — sampling UX polish

Pre-check showed the sampling pipeline was already shipped
(`sample_next_token` does rep-penalty → temp → softmax → top-K →
top-P → weighted draw, with all CLI flags wired through). The gap
was UX:

- **`--seed` defaults to `SystemTime::now().nanos`** when unspecified
  so `--temperature 0.7` gives a fresh sequence each run. Explicit
  `--seed N` / `VF_SEED=N` still pin reproducibility.
- **Banner shows sampling config** — `Sampling: greedy (temperature=0)`
  on the default path, `Sampling: temp=0.70 top_k=40 top_p=0.90 rep_pen=1.00 seed=auto`
  otherwise.
- **README** rewritten around the new CLI; CLI flag table + matching
  `VF_*` env-var table.

Defaults remain temperature=0 / greedy — preserves the documented
byte-deterministic invariant.

### Bugfixes

- **Vulkan device features** — `Vulkan12Features.shaderFloat16`,
  `vulkan_memory_model`, plus `VK_KHR_cooperative_matrix` extension
  and feature struct in the device pNext chain. RADV ran the binary
  without them (it's permissive); the validation layer correctly
  flagged them as missing-feature requests.
- **UTF-8 streaming** — added `Tokenizer::decode_token_bytes()` plus
  a per-decode UTF-8 stream buffer so single tokens carrying a
  fragment of a multi-byte codepoint (e.g. one of 4 bytes in 😊)
  no longer get lossy-converted to U+FFFD per token. Emoji render
  cleanly mid-stream.
- **Sprint-3 FP8/BF16 probe shaders** — moved
  `MulCoopmatQ4KFwd*` / `MulCoopmatQ4KNaive*` (which declare
  `Float8EXT` / `BFloat16TypeKHR` SPIR-V capabilities not exposable
  through ash 0.38) out of `ALL_SHADERS` into a new
  `COOPMAT_Q4K_OPTIONAL_SHADERS` list. The registry only loads them
  when `VULKANFORGE_COOPMAT=1` is set (same gate as the Sprint-3
  forward path). 16 validation-layer false-negatives → 0.
- **Chat REPL line editing** — replaced `stdin().lock().lines()` with
  `rustyline::DefaultEditor`. ←/→ cursor movement, ↑/↓ history,
  Ctrl+A/E line nav, Ctrl+C/D clean exit; non-tty stdin still works.

## v0.3.0 — async pipelined decode (2026-05-02)

### Headline

**Decode breakthrough: 91.1 → 109.0 tok/s (+19.3 %, 0.95 × llama.cpp
Vulkan).** The first decode performance gain since v0.2.0. The
3-stage async pipelined decode loop hides CPU command-recording
inside the GPU compute window of the previous token, matching the
+20 % theoretical ceiling Sprint 15A measured. Prefill unchanged
at 0.89 × llama.cpp Vulkan @ pp=512 (3 865 tok/s — async only
touches the decode GEMV path). 27 / 27 lib tests, 15 / 15 coherent
under both async and serial paths, bit-identical generated output.

| Metric  | v0.2.0 | v0.2.4 | **v0.3.0** | llama.cpp | Ratio (v0.3.0) |
|---------|-------:|-------:|-----------:|----------:|---------------:|
| Decode  |   90.5 |   91.1 |  **109.0** |     114.2 |       **0.95×** |
| pp=512  |   2255 |   3863 |       3865 |      4326 |          0.89× |
| pp=1024 |   2204 |   3748 |       3742 |      4173 |          0.90× |

Benchmark: Qwen3-8B-Q4_K_M, RX 9070 XT (gfx1201), RADV Mesa 26.0.6.
llama.cpp: build 23b8cc4 with `-fa 1` on the same hardware.

### Shipped — Sprint 15D + 15E

#### Sprint 15D — Double-buffered intermediate buffers

Extracted 17 per-forward scratch buffers into a new
`IntermediateSlot` struct (`scratch_a`, `scratch_b`, `hidden_norm`,
`q_buf`, `k_buf`, `v_buf`, `attn_out`, `o_buf`, `res1`,
`gate_buf`, `up_buf`, `ffn_hidden`, `ffn_out`, `rope_pos_buf`,
`fa_scratch_out`, `fa_scratch_max`, `fa_scratch_sum`). `Forward`
now holds `slots: [IntermediateSlot; 2]` and a `current_slot`
index. 96+ buffer references across `forward.rs` were rewritten
via `cur()` / `cur_mut()` accessors. **Performance-neutral** in
isolation — Sprint 15E uses the slot pair to allow the next
token's CB to be recorded while the current token's GPU work is
in flight. VRAM impact: ~1-2 MB.

#### Sprint 15E — 3-stage async pipelined decode loop

Three new public methods on `Forward`:

- `pre_record(slot, position)` — records the standard 36-layer +
  lm_head dispatch sequence into `async_cbs[slot]`, referencing
  `slots[slot]` buffer handles. Does **not** submit; runs in
  parallel with GPU work on the other slot.
- `fill_embed_and_submit(slot, embedding, position)` — host-writes
  the embedding into `slots[slot].scratch_a` and the position
  into `slots[slot].rope_pos_buf`, then submits `async_cbs[slot]`.
- `wait_and_read_logits(slot)` — blocks on `async_fences[slot]`
  and reads `logits_buf`.

`Forward` got 4 new fields: `async_pool: vk::CommandPool`,
`async_cbs: [vk::CommandBuffer; 2]`,
`async_fences: [vk::Fence; 2]`, `async_pending_record:
Option<usize>`. Allocated in `Forward::new`, freed in
`Forward::destroy`. The `record_decode_dispatches` helper was
extracted from `forward_token`'s `cmd_ctx.one_shot` closure body
so both serial and async paths share the recording sequence
(DRY).

`decode.rs::generate_from_tokens` got an
`if async_decode { … } else { serial fallback }` branch. The
async branch implements the 3-stage rolling pipeline:

```
Token N:                                              GPU runs CB[N] (~9 ms)
  Stage 1: pre_record(CB[N+1])  ──────┐
  Stage 2: wait(CB[N]) → readback     │  CPU recording of CB[N+1]
           → sample → embed lookup    │  hidden inside GPU(N) window
  Stage 3: fill_embed → submit ───────┘
           CB[N+1] starts on GPU
```

Per-token wall: max(record 1 836 µs, GPU 9 034 µs) + ~80 µs
sequential = 9 114 µs → **109.7 tok/s theoretical, 109.0
measured** (99.4 % of ceiling). The CPU recording is fully
hidden inside the GPU's compute window.

This works because Vulkan records buffer **handles**, not buffer
*contents* — the embedding is written into `slots[(N+1)%2].scratch_a`
*after* `pre_record` but *before* submit. Vulkan's queue
ordering on a single queue guarantees CB[N+1] starts after CB[N]
without needing timeline semaphores.

Default-on; opt-out: `VULKANFORGE_DISABLE_ASYNC_DECODE=1`.
Output is **bit-identical** between async and serial paths.
First-token cold-start (no prev to wait for), EOS-mid-pipe
(pre-recorded CB left unsubmitted, recycled next session), and
drain-final are all handled.

Shipped LOC: +227 / −47 in `forward.rs`, +102 in `decode.rs`.

### Investigation — Sprints 12-15 decode-gap analysis arc

Across 4 weeks of work, **13 decode-gap hypotheses** were
systematically tested and falsified before the 14th — the correct
3-stage pipeline shape — delivered the +19.3 % win. Each prior
sprint either shipped infrastructure that 15E ultimately needed
(Sprint 14A's `requiredSubgroupSize`, 14B's subgroup GEMV, 15D's
slot refactor) or eliminated a candidate cleanly:

| #  | Hypothesis                                | Sprint | Result on RDNA4                |
|---:|--------------------------------------------|--------|--------------------------------|
|  1 | Barrier elision (dirty-flag tracker)       | 12D    | 0 % wall-time impact           |
|  2 | Norm + RoPE fusion                         | 12E    | +1 % (run-to-run noise)        |
|  3 | Q6_K shader optimisation                   | 12H    | upstream-identical             |
|  4 | Mesa 26.0.6 → 26.1-rc3 driver upgrade      | 13B    | ±2 % noise (llama.cpp also flat) |
|  5 | f16-accumulator coopmat shader             | 13C    | −2 % (FP16 emulated on RDNA4)  |
|  6 | Wave32 / VOPD dual-issue codegen           | 13D    | 0 % decode (memory-bound)      |
|  7 | `MMV_NUM_ROWS=2` with LDS Path B           | 13E    | −2.9 % (LDS-doubling penalty)  |
|  8 | `subgroupAdd` GEMV reduction (Path A)      | 14B    | +0.16 % (within noise)         |
|  9 | `MMV_NUM_ROWS=2` with Path A               | 14C    | −1.5 % (per-WG VGPR)           |
| 10 | Template CB-reuse via UBO                  | 15A    | falsified at source-reading    |
| 11 | `lm_head` NUM_ROWS / coopmat               | 15B    | already at 94 % HBM ceiling    |
| 12 | 1-day async prototype                      | 15C    | infra-blocked (no double-buffering) |
| 13 | Naive 2-stage pipeline (record after wait) | 15E §2 | wouldn't pipeline (traced)     |
| **14** | **3-stage pipeline (correct shape)**   | **15E** | **+19.3 % (shipped)**         |

The methodology — measure first, falsify cheap, build only what
survives the math — produced 13 honest negatives and 1 real
positive. The analytical prediction (109.7 tok/s from CPU/GPU
timing measurements) matched the final implementation
(109.0 tok/s) to within 1 %.

### Configuration

| Variable                              | Default | Effect |
|---------------------------------------|---------|--------|
| `VULKANFORGE_DISABLE_ASYNC_DECODE=1`  | off     | **Disable async pipeline; use serial decode loop. New in v0.3.0.** |
| `VULKANFORGE_DISABLE_MM_COOPMAT=1`    | off     | Disable coopmat prefill (use scalar `mul_mmq`). |
| `VULKANFORGE_COOPMAT_F16ACC=1`        | off     | Opt-in FP16 accumulator (RDNA4-neutral; may help non-RDNA hardware). |
| `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` | off     | Disable subgroupAdd GEMV reduction. |

### What's still on the table

The remaining ~5 % decode gap to llama.cpp lives in:

- **Dedicated `lm_head` coopmat dispatch** (Sprint 15B analysis):
  ~1 % decode lift estimated. lm_head is Q6_K and already at 94 %
  HBM ceiling — coopmat can't reduce HBM bytes, so the lift is
  small.
- **Buffer-aliasing / live-set reduction**: 0–5 %, unmeasured.
  ~20 SSBOs live per layer vs llama.cpp's 3-4. May matter for
  L2 thrashing.
- **`quantize_q8_1` fusion into the GEMM dispatch** (prefill,
  smaller win): few % at pp=512.

These are v0.4-class architectural changes if they materialise.

### Files added / changed in v0.3.0

```
EDIT  Cargo.toml                              0.2.4 → 0.3.0
EDIT  src/backend/vulkan/forward.rs           +466 / −228 LOC across 15D + 15E:
                                                Sprint 15D (already shipped):
                                                  IntermediateSlot struct, slots[2],
                                                  cur()/cur_mut() helpers, 96+ refs
                                                Sprint 15E (this release):
                                                  4 new fields (async_pool, async_cbs[2],
                                                  async_fences[2], async_pending_record),
                                                  3 new public methods, record_decode_dispatches
                                                  helper extraction
EDIT  src/backend/vulkan/decode.rs            +102 LOC (async path branch in
                                                generate_from_tokens with first-token,
                                                EOS, and drain handling)
NEW   results/v030_sprint15a_cb_reuse.md
NEW   results/v030_sprint15b_lm_head.md
NEW   results/v030_sprint15c_async_prototype.md
NEW   results/v030_sprint15de_async_decode.md
NEW   results/v030_sprint15e_async_pipeline.md
EDIT  README.md                               v0.3.0 perf table + features
EDIT  CHANGELOG.md                            this entry
```

### Stats

- 72 SPVs (unchanged from v0.2.4)
- 27 lib tests, 15 / 15 coherent on both async and serial paths
- 12 coopmat GEMM pipeline variants (S/M/L × Q4_K/Q6_K ×
  aligned/unaligned) + 2 f16acc opt-in + 4 GEMV (subgroup +
  stock × Q4_K/Q6_K)
- All shader sources byte-identical to llama.cpp upstream

---

## v0.2.4 — subgroup-arithmetic GEMV + decode-gap analysis (2026-05-02)

### Headline

Default-config performance is **unchanged** from v0.2.2 / v0.2.3:
prefill 0.89 × llama.cpp Vulkan at pp=512 (3863 tok/s), decode
91.1 tok/s median (0.80 ×). v0.2.4 is an **infrastructure +
analysis** release. Two shipped infrastructure changes
(`requiredSubgroupSize=64` pipeline plumbing; `subgroupAdd`
GEMV reduction default-on) and one investigation that closed
the GEMV-pipeline branch of the decode-optimisation tree
(`MMV_NUM_ROWS=2` re-tested with subgroupAdd, still regresses
on RDNA4, reverted). The release is anchored by a definitive
9-hypothesis decode-gap analysis spanning Sprints 12-14.
27 / 27 lib tests, 15 / 15 coherent on the bench suite,
72 SPVs (+2 vs v0.2.3).

### Shipped changes

- **`requiredSubgroupSize=64` pipeline plumbing** (Sprint 14A).
  Device-creation now enables `Vulkan13Features::subgroupSizeControl`
  + `computeFullSubgroups`. `ComputeKernel::from_spv_with_spec`
  accepts an optional `required_subgroup_size: Option<u32>`
  parameter; when `Some(N)` it chains
  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` and sets
  the `REQUIRE_FULL_SUBGROUPS` stage flag. GEMV pipelines (the
  four `MulMatVec*` ShaderIds, including the new `_subgroup`
  variants) pin `Some(64)`; every other pipeline keeps `None`.
  No measurable performance delta (verified ±3 % at every pp);
  this is purely infrastructure for Sprint 14B and any future
  work that depends on a fixed subgroup size.
  Sprint 14A: `results/v024_sprint14a_subgroup_size.md`.

- **Subgroup-arithmetic GEMV reduction** (Sprint 14B,
  default-on). New SPVs
  `mul_mat_vec_q{4,6}_k_f32_f32_subgroup.spv` built with
  `USE_SUBGROUP_ADD=1` flip `mul_mat_vec_base.glsl`'s
  `reduce_result()` from the LDS tree-reduction (Path B,
  6 barrier levels at BLOCK_SIZE=64) to a single wave-wide
  `subgroupAdd` (Path A, 0 barriers, 0 LDS for the actual
  reduction step). Matches llama.cpp's RDNA4 GEMV recipe
  (`ggml-vulkan.cpp:4180`) exactly. **Wall-time impact ≤ 0.16 %
  at pos=200** — the LDS reduction was already a < 0.2 % slice
  of per-dispatch GEMV time. The change is shipped because (a) it
  matches upstream, (b) it is the prerequisite for any future
  GEMV port that depends on subgroupSize being pinned, and
  (c) it is cleanly correctness-verified (15 / 15 coherent under
  both paths).
  Opt out: `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` (falls back to
  the LDS Path B SPVs).
  Sprint 14B: `results/v024_sprint14b_subgroup_gemv.md`.

### Investigation: `MMV_NUM_ROWS=2` redux (Sprint 14C, reverted)

Sprint 13E showed `NUM_ROWS=2` was a regression with Path B
(LDS-doubling at NUM_ROWS=2 → gemv_q +21 %). Hypothesis: with
Path A active, NUM_ROWS=2 should finally be net-positive
(`subgroupAdd` cost is constant per row).

Verdict: **falsified again.** One-line spec-constant flip,
27 / 27 lib tests, 15 / 15 coherent. But:

- Per-dispatch GEMV total at pos=200: 10 618 µs (NR=2 Path A)
  vs 10 188 µs (NR=1 Path A baseline) = **+4.2 % slower**.
- 15-prompt decode median: 90.1 tok/s vs 91.5 (NR=1) = **−1.5 %**.

The Sprint 13E +21 % `gemv_q` disaster *is* gone (same-session
sanity test confirmed Path A and Path B are now within 0.08 % at
NUM_ROWS=2 → Path A did fix the LDS-doubling lever). The
remaining +4 % regression is structural: per-WG VGPR pressure
grows with NUM_ROWS, ACO's superblock-loop unroll heuristics are
tuned for NR=1, and GEMVs are already memory-bandwidth-bound on
RDNA4 (77-91 % peak HBM per Sprint 12G-D / 12H), so halving the
WG count from 1024 → 512 just reorders bytes.

Reverted to `MMV_NUM_ROWS = 1`. The pipeline-registry comment
documents the full 13E → 14B → 14C arc.
Sprint 14C: `results/v024_sprint14c_numrows2_redux.md`.

### Decode-gap analysis: 9 hypotheses, all falsified

The 91 → 114 tok/s decode gap to llama.cpp on RDNA4 + this
codebase has now been exhaustively investigated at the
shader-config / pipeline-config / driver-flag level. Every
"port llama.cpp's recipe" hypothesis tested in Sprints 12-14
has been measured and falsified:

| # | Hypothesis | Sprint | Result on RDNA4 |
|---|---|---|---|
| 1 | Barrier elision (dirty-flag tracker) | 12D | 0 % wall-time impact |
| 2 | Norm + RoPE fusion | 12E | +1 % (run-to-run noise) |
| 3 | Q6_K shader optimisation | 12H | upstream-identical, nothing to port |
| 4 | Mesa 26.0.6 → 26.1-rc3 driver | 13B | ±2 % noise (llama.cpp also flat) |
| 5 | f16-accumulator coopmat shader | 13C | −2 % (FP16 fragment emulated, not native) |
| 6 | Wave32 / VOPD dual-issue codegen | 13D | 0 % decode (memory-bound, not VALU-bound) |
| 7 | `MMV_NUM_ROWS=2` with LDS Path B | 13E | −2.9 % (LDS-doubling penalty) |
| 8 | `subgroupAdd` GEMV reduction (Path A) | 14B | +0.16 % (within noise) |
| 9 | `MMV_NUM_ROWS=2` with Path A | 14C | −1.5 % (per-WG VGPR + saturated CUs) |

**The remaining gap is structural at the graph level.** The
candidates that have *not* been falsified:

- **Multi-submit / command-buffer reuse decode loop.** llama.cpp
  re-submits a single prebuilt CB per token; we re-record per
  token. Sprint 5A's CB-reuse infrastructure exists for parity
  testing only — extending it to the production decode path
  needs template-based push-constants and touches every shader's
  pipeline-layout. Estimated lift: 5-10 % decode.
- **Dedicated `lm_head` coopmat dispatch.** N=151 936; ~6 % of
  decode forward today. A coopmat path could shave ~3 %.
- **Buffer-aliasing / live-set reduction.** 20+ live SSBOs per
  layer vs llama.cpp's 3-4. Possibly L2-relevant; not yet
  measured.
- **`quantize_q8_1` fusion into GEMM dispatch.** Few % at
  pp=512 estimated.

These are v0.3-class architectural changes, not single-line
config flips. The Sprint 12-14 arc closes the
shader-and-pipeline-config branch of the optimisation tree.

### Configuration

- `VULKANFORGE_DISABLE_MM_COOPMAT=1` — disable coopmat for prefill
  (use `mul_mmq` integer-DP fallback).
- `VULKANFORGE_COOPMAT_F16ACC=1` — opt-in FP16 accumulator
  (RDNA4-neutral).
- **`VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` (new)** — disable
  subgroupAdd GEMV reduction; falls back to LDS tree-reduction
  Path B.

### New SPVs

```
NEW   mul_mat_vec_q4_k_f32_f32_subgroup.spv  (Sprint 14B, Path A)
NEW   mul_mat_vec_q6_k_f32_f32_subgroup.spv  (Sprint 14B, Path A)
```

Total SPV count went 70 → 72.

### Files added / changed in v0.2.4

```
EDIT  Cargo.toml                              0.2.3 → 0.2.4
EDIT  build.rs                                +30 LOC (2 _subgroup ShaderJobs)
EDIT  src/backend/vulkan/device.rs            +9 LOC (Vulkan13Features
                                              subgroupSizeControl + computeFullSubgroups)
EDIT  src/backend/vulkan/pipeline.rs          +18 LOC (required_subgroup_size param,
                                              VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
                                              chained on pNext, REQUIRE_FULL_SUBGROUPS flag)
EDIT  src/backend/vulkan/pipeline_registry.rs +7 LOC (13 from_spv_with_spec
                                              call sites updated; GEMV match arm
                                              extended to 4 ShaderIds with Some(64) pin;
                                              MMV_NUM_ROWS comment refresh for 14B/14C)
EDIT  src/backend/vulkan/shaders.rs           +12 LOC (2 ShaderId variants
                                              MulMatVec{Q4K,Q6K}Subgroup + plumbing)
EDIT  src/backend/vulkan/forward.rs           +30 LOC (mul_mat_vec_subgroup_enabled
                                              field + env-var; layer_weight_shader
                                              extended with subgroup boolean; 9 call
                                              sites + lm_head selection updated)
NEW   results/v024_sprint14a_subgroup_size.md
NEW   results/v024_sprint14b_subgroup_gemv.md
NEW   results/v024_sprint14c_numrows2_redux.md
EDIT  README.md                               v0.2.4 features + env var + 9-hyp table
EDIT  CHANGELOG.md                            this entry
```

### What's still on the table for v0.3

- **Multi-submit / CB-reuse decode loop** (5–10 % decode lift estimated).
- **Dedicated `lm_head` coopmat dispatch** (~3 % decode).
- **Buffer-aliasing / live-set reduction**.
- **`quantize_q8_1` fusion into the GEMM dispatch** (small prefill lift).

---

## v0.2.3 — tile coverage complete + RDNA4 root-cause analysis (2026-05-02)

### Headline

Default-config performance is **unchanged** from v0.2.2: prefill
0.89 × llama.cpp Vulkan at pp=512 (3863 tok/s), decode 91.1 tok/s
(0.80 ×). v0.2.3 is an **analysis + completeness** release. Two
shipped changes (S-tile coopmat for pp ≤ 32; opt-in f16-accumulator
coopmat) and four investigation sprints (Mesa 26.1, f16acc analysis,
Wave32 / VOPD probe, `MMV_NUM_ROWS=2`) that systematically located
where the remaining gap to llama.cpp lives. 27 / 27 lib tests,
15 / 15 coherent on the bench suite, 70 SPVs (+2 vs v0.2.2).

### Shipped changes

- **S-tile coopmat (BM=32)** for `seq_len ≤ 32`. Completes the S/M/L
  tile matrix; matches llama.cpp's `ggml_vk_guess_matmul_pipeline`
  variant coverage. 4 new pipeline variants via spec-constants
  (`{64, 32, 32, 16, 32, 32, 2, 16, 16, 16, 64}`); zero new SPVs.
  Selector: `n ≤ 32 → S, n ≤ 64 → M, else → L`. At pp=32, S-tile is
  +27 % over the scalar `mul_mmq` default-off path (765 → 975 tok/s);
  vs M-tile alone +1.9 % (within run-to-run noise — the
  saturation-by-WG-count theory does not translate at this shape on
  RDNA4 because the kernel is already hiding HBM latency).
  Sprint 13A: `results/v023_sprint13a_stile.md`.

- **f16-accumulator coopmat path** (opt-in, default OFF).
  `VULKANFORGE_COOPMAT_F16ACC=1` redirects the aligned-L-tile
  coopmat dispatch to a new SPV with `ACC_TYPE = float16_t`,
  `ACC_TYPE_MAX = float16_t(65504.0)`, and `D_TYPE = float`
  (writeback stays FP32). 2 new SPVs:
  `mul_mm_q{4,6}_k_f32_aligned_coopmat_f16acc.spv`. 27 / 27 lib tests
  + 15 / 15 coherent under f16acc — precision-safe on Qwen3-8B-Q4_K_M
  including the K=12288 reduction in `gemm_down`. **Performance on
  RDNA4: −2 %** at pp=512 because RDNA4 has only
  `v_wmma_f32_16x16x16_fp16` (FP32 result fragment) — the
  cooperative-matrix API exposes a FP16→FP16 fragment shape, but
  ACO lowers it onto the same f32 hardware path with
  FP32↔FP16 conversions, costing wall time. Retained for users on
  hardware with native f16 accumulator (NVIDIA Ampere+, Intel XMX).
  Sprint 13C: `results/v023_sprint13c_f16acc.md`.

### Investigation sprints (no perf change, documented)

- **Sprint 13B — Mesa 26.1-rc3 driver test.** Built Mesa 26.1.0-rc3
  locally (`~/tmp/mesa-26.1/`, RADV-only, opt-in via
  `VK_ICD_FILENAMES` + `LD_LIBRARY_PATH`; system Mesa 26.0.6
  untouched). Both VulkanForge and llama.cpp are flat between Mesa
  26.0.6 and 26.1-rc3 (≤ ±2.3 % at every pp). The Mesa 26.1 f16acc
  driver patches do not move our coopmat path on RDNA4 — neither do
  they move llama.cpp's. **Production stays on Mesa 26.0.6.**

- **Sprint 13D — Wave32 / VOPD probe.** On Mesa 26.1-rc3 with
  `RADV_PERFTEST=cswave32` ACO emits **3 546** `v_dual_*` instructions
  (vs 65 under Wave64) — the codegen feature works as designed.
  But wall-time is flat: decode neutral (90.7 → 90.8), prefill mildly
  negative (−1.8 to −3.2 % across pp). Sprint 12G-D's "27.5 % VALU
  utilisation, 72.5 % idle" was memory-wait, not unfilled VALU slots;
  doubled VOPD issuance fills cycles that were already wait-states.
  **Wave32 not recommended for v0.3.** 27 / 27 lib + 15 / 15 coherent.

- **Sprint 13E — `MMV_NUM_ROWS=2` GEMV pipelines.** llama.cpp
  unconditionally uses `NUM_ROWS = 2` for K-quant GEMV pipelines on
  non-GCN AMD (`ggml-vulkan.cpp:4128 rm_kq = 2`). The Vulkan shader
  source is byte-identical and `NUM_ROWS` is a spec-constant
  (`mul_mat_vec_base.glsl:90`), so the change is a one-line edit.
  Same-session A/B with `profile_positions`: NUM_ROWS=2 is
  **measurably slower** per-dispatch — gemv_q +21 %, gemv_k +7.7 %,
  gemv_v +2.6 %; forward wall +2.9 %. Reason: llama.cpp pairs
  NUM_ROWS=2 with the subgroup-arithmetic reduction (`subgroupAdd`)
  gated on `USE_SUBGROUP_ADD` + `requireFullSubgroups` +
  `requiredSubgroupSize=64` at pipeline creation. We ship only the
  LDS-tree-reduction fallback path; doubling NUM_ROWS doubles LDS
  traffic without halving reduction depth. **Reverted.** Real port
  is a Sprint 14 / v0.3 infrastructure item.

### Where the remaining gap lives — root-cause analysis

The remaining ~10 % prefill gap (0.89 ×) and ~20 % decode gap (0.80 ×)
to llama.cpp on RDNA4 is **NOT** in:

- Shader source (md5-identical to upstream `23b8cc4`, Sprint 12H/12I)
- Coopmat tile spec-constants (S/M/L parity, Sprint 12M / 13A)
- Driver version (Mesa 26.0.6 ≡ 26.1-rc3 within noise, Sprint 13B)
- f16-accumulator coopmat (emulated on RDNA4, Sprint 13C)
- Wave32 / VOPD dual-issue (memory-bound workload, Sprint 13D)
- `MMV_NUM_ROWS=2` GEMV (needs subgroup arithmetic first, Sprint 13E)

It **IS** in pipeline-creation infrastructure:

1. `requireFullSubgroups` + `requiredSubgroupSize=64` pinning at
   pipeline creation (we don't pass these).
2. Subgroup-arithmetic reduction in `mul_mat_vec_base.glsl` Path A
   (`subgroupAdd` over 64 lanes) vs our Path B (LDS tree reduction
   in 6 levels with barriers).
3. Multi-submit pipeline parallelism in prefill (Sprint 12B audit).
4. `quantize_q8_1` fusion into the GEMM dispatch (Sprint 12I §6).

These are v0.3-class infrastructure work items touching every shader's
pipeline path. Outside the scope of single-line config flips.

### Configuration

- `VULKANFORGE_DISABLE_MM_COOPMAT=1` — disable coopmat for prefill
  (use the `mul_mmq` integer-DP fallback).
- **`VULKANFORGE_COOPMAT_F16ACC=1` (new)** — opt-in FP16 accumulator
  for the aligned-L-tile coopmat path. RDNA4-neutral-to-slightly-negative.
- On Mesa 26.1+: `RADV_PERFTEST=cswave32` to test Wave32 (neutral).

### New shaders / SPVs

```
NEW   mul_mm_q4_k_f32_aligned_coopmat_f16acc.spv  (Sprint 13C)
NEW   mul_mm_q6_k_f32_aligned_coopmat_f16acc.spv  (Sprint 13C)
```

Total SPV count went 68 → 70. S-tile pipelines are 0 new SPVs
(reuse L/M-tile binaries via spec-constants).

### Files added / changed in v0.2.3

```
EDIT  Cargo.toml                              0.2.2 → 0.2.3
EDIT  build.rs                                +44 LOC (2 f16acc ShaderJobs)
EDIT  src/backend/vulkan/shaders.rs           +30 LOC (4 S-tile + 2 f16acc ShaderIds)
EDIT  src/backend/vulkan/pipeline_registry.rs +20 LOC (S-tile spec-constants
                                              + f16acc folded into match arm
                                              + NUM_ROWS=1 rationale comment)
EDIT  src/backend/vulkan/forward.rs           +35 LOC (S-tile selector,
                                              f16acc env var + struct field +
                                              routing guard, six call-site
                                              updates for the new arg)
NEW   results/v023_sprint13a_stile.md
NEW   results/v023_sprint13b_mesa26.1_test.md
NEW   results/v023_sprint13c_f16acc.md
NEW   results/v023_sprint13d_wave32_probe.md
NEW   results/v023_sprint13e_mmv_numrows.md
EDIT  README.md                               v0.2.3 perf table + features
EDIT  CHANGELOG.md                            this entry
```

### What's still on the table for v0.3

- Subgroup-arithmetic GEMV pipeline + `requiredSubgroupSize` plumbing
  (unblocks NUM_ROWS=2 — Sprint 13E §7).
- Decode-side coopmat for `lm_head` (vocab-major GEMV with
  N=151 936; ~3 % decode improvement potential).
- Multi-submit prefill graph (Sprint 12B audit; ~5–10 % at pp=512).
- `quantize_q8_1` fusion into the GEMM dispatch (Sprint 12I §6).

---

## v0.2.2 — coopmat WMMA prefill default-on (2026-05-01)

### Headline

**Prefill peak +64 %** at pp=512 (2353 → 3863 tok/s) over the
v0.2.2 default-off `mul_mmq` path; release-to-release v0.2.0 → v0.2.2
is **+71 %** (2255 → 3863). Reaches **0.89 × llama.cpp Vulkan**
prefill at pp ≥ 256 (up from 0.52 × in v0.2.0). KHR cooperative-matrix
WMMA prefill is now **default-on** for Q4_K and Q6_K GEMMs. Decode
unchanged at 91.1 tok/s median (0.80 × llama.cpp). 176 / 176 tests
green (27 lib + 149 integration), 15 / 15 coherent on the bench suite.

### Performance (Qwen3-8B-Q4_K_M, RX 9070 XT, RUNS=5 median)

| pp   | v0.2.0 | v0.2.2 default-off | v0.2.2 default-on | vs default-off | v0.2.0 → v0.2.2 | vs llama.cpp Vulkan |
|------|-------:|-------------------:|------------------:|---------------:|----------------:|--------------------:|
|   64 |   1511 |               1513 |              1678 |          +11 % |          +11 %  | 0.73 × |
|  128 |   2001 |               2010 |              2560 |          +27 % |          +28 %  | 0.70 × |
|  256 |   2200 |               2199 |              3558 |          +62 % |          +62 %  | 0.89 × |
|  512 |   2255 |               2353 |          **3863** |       **+64 %** |       **+71 %** | **0.89 ×** |
| 1024 |   2204 |               2306 |              3748 |          +63 % |          +70 %  | 0.90 × |
| 2048 |   1997 |               2088 |              3172 |          +52 % |          +59 %  | 0.84 × |

`default-off` = the same v0.2.2 binary with
`VULKANFORGE_DISABLE_MM_COOPMAT=1`, i.e. the integer-DP `mul_mmq`
fallback path that v0.2.1 shipped as default. Source for these
numbers: Sprint 12M `results/v022_sprint12m_mtile.md` table §1.

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
