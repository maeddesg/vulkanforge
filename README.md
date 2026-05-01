# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

**v0.2.3** — analysis + completeness release on top of v0.2.2's
coopmat default-on prefill. The S/M/L coopmat tile matrix is now
complete (S-tile BM=32 added for `seq_len ≤ 32`, matching llama.cpp's
variant coverage), an opt-in f16-accumulator coopmat path is shipped,
and five Sprint 13 investigation reports document the remaining gap
to llama.cpp on RDNA4. Default-config performance is **unchanged**
from v0.2.2: prefill 0.89× llama.cpp Vulkan at pp=512 (3863 tok/s),
decode 91.1 tok/s median (0.80×). Supports the Qwen, Llama-3,
DeepSeek-R1 and Mistral GGUF families out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | llama (SPM) | Mistral | ✅ |

### Key features (v0.2.3)

- **KHR cooperative matrix WMMA prefill** (default ON) — Q4_K and Q6_K
  GEMM dispatched through RDNA4's 128 AI Accelerators via
  `VK_KHR_cooperative_matrix`. **S-tile (BM=32) + M-tile (BM=64) +
  L-tile (BM=128)** pipelines with a runtime selector that mirrors
  llama.cpp's `ggml_vk_guess_matmul_pipeline` (`n ≤ 32 → S`,
  `n ≤ 64 → M`, else L). Aligned variant uses `LOAD_VEC_B=8` with
  `B_TYPE=mat2x4` for 4× wider B-matrix loads. Opt out with
  `VULKANFORGE_DISABLE_MM_COOPMAT=1`.
- **f16-accumulator coopmat path** (opt-in via
  `VULKANFORGE_COOPMAT_F16ACC=1`) — FP16 accumulator instead of FP32.
  Default OFF. RDNA4-neutral-to-slightly-negative because the FP16
  fragment is emulated on top of `v_wmma_f32_16x16x16_fp16`. Retained
  for hardware with native f16 accumulator support (NVIDIA Ampere+,
  Intel XMX).
- **coopmat QK attention** — KHR cooperative matrix WMMA replaces the
  scalar inner loop in `flash_attn_coopmat.comp`. ~85 % faster prefill
  at pp=2048 vs scalar; resolves the pp=4096 TDR crash.
- **FP16 KV-cache** (default ON) — half the cache VRAM, +21 % prefill
  at pp=2048. Opt out with `VULKANFORGE_FP16_KV=0`.
- **5 fused kernels** — `swiglu`, `multi_add_rms` (×2 sites),
  `rms_norm_mul_rope` — −5 dispatches per layer.
- **Tiled flash-attention** — Br=16 / Bc=32 with online softmax.
- **pp=4096 supported** — previously crashed with TDR.

Gemma-4 is out of scope (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, gfx1201, RDNA 4)

### Prefill throughput sweep (Qwen3-8B-Q4_K_M, RUNS=5 median)

| pp   | VulkanForge v0.2.3 | VulkanForge v0.2.0 | llama.cpp Vulkan | Ratio (v0.2.3) |
|------|-------------------:|-------------------:|-----------------:|---------------:|
|   32 |          **975**   |              —     |             —    | — |
|   64 |              1678  |              1511  |             2285 | 0.73× |
|  128 |              2560  |              2001  |             3637 | 0.70× |
|  256 |              3558  |              2200  |             3995 | 0.89× |
|  512 |          **3863**  |              2255  |             4326 | **0.89×** |
| 1024 |              3748  |              2204  |             4173 | 0.90× |
| 2048 |              3172  |              1997  |             3765 | 0.84× |

llama.cpp reference: build 23b8cc4 with `-fa 1` on the same hardware.
Peak prefill throughput is **3863 tok/s @ pp=512** (unchanged from
v0.2.2). pp=32 is the new measurement enabled by Sprint 13A's S-tile
coopmat — 975 tok/s vs 765 tok/s on the scalar `mul_mmq` default-off
path (+27 %). The pp ≥ 256 ratio is within run-to-run noise of
llama.cpp's `mul_mm` Vulkan path; pp ≤ 128 carries a 0.70–0.73 × gap
that lives in pipeline-creation infrastructure (subgroup-arithmetic
reduction), not shader source. See "Limitations" and Sprint 13E.

### 4-system comparison (Qwen3-8B, same hardware)

| System                     | Decode tok/s | Prefill peak tok/s | Decode ratio | Prefill ratio |
|----------------------------|-------------:|-------------------:|-------------:|--------------:|
| llama.cpp Vulkan           |      114.2   |              4326  |       1.00×  |        1.00×  |
| **VulkanForge v0.2.3**     |   **91.1**   |          **3863**  |    **0.80×** |     **0.89×** |
| VulkanForge v0.2.0         |       90.5   |              2255  |       0.79×  |        0.52×  |
| llama.cpp ROCm             |       87.5   |              3684  |       0.77×  |        0.85×  |
| ROCmForge (HIP)            |       95.4   |               769  |       0.84×  |        0.18×  |

vs v0.2.0: decode +0.7 % (90.5 → 91.1, run-to-run noise), prefill peak
**+71 %** (2255 → 3863). Decode unchanged because coopmat is
prefill-only — GEMV continues through the existing `mul_mat_vec_*`
shaders. ROCm / ROCmForge HIP rows are carried forward from v0.2.0's
4-system run; not re-measured for v0.2.3. v0.2.3 default-config
performance is identical to v0.2.2 — Sprint 13A added pp ≤ 32 coverage
without changing pp ≥ 64 numbers, and Sprint 13B-13E investigations
were measurement-only.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 176 tests across 7 binaries (27 lib, 149 integration)
```

The build compiles 70 SPIR-V binaries (53 in v0.2.0, 65 in v0.2.1,
68 in v0.2.2, +2 in v0.2.3: Q4_K and Q6_K aligned coopmat with f16
accumulator).

MSRV is **Rust 1.85** (edition 2024). Build dependencies require a working
`shaderc` install (the `shaderc-sys` crate); on Arch / CachyOS this is
`shaderc` from the official repos. `VK_KHR_cooperative_matrix` must be
advertised by the driver — RADV gfx1201 with Mesa 26.0.5+ qualifies.
Mesa 26.1-rc3 is functionally fine (Sprint 13B) but does not improve
performance vs 26.0.6; recommended driver remains **Mesa 26.0.6**.

## Run

The default 15-prompt benchmark on the bundled prompts JSON:
```bash
VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
  cargo run --release --example run_15prompt_bench
```

Run any single prompt against any supported model:
```bash
VF_MODEL_PATH=$HOME/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
VF_PROMPT="What is 2+2?" \
VF_MAX_TOKENS=80 \
  cargo run --release --example sample_decode
```

Prefill throughput sweep (the source for the perf table above):
```bash
VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
  cargo run --release --example run_pp_bench
```

## Configuration (environment variables)

### Default-on toggles (set to `0` / `false` / `true` to override)

| Variable | Default | Effect |
|---|---|---|
| `VULKANFORGE_DISABLE_MM_COOPMAT=1` | off (coopmat ON) | Falls back to scalar `mul_mmq` GEMM (v0.2.1 behaviour). |
| `VULKANFORGE_USE_MM_COOPMAT=0` | (legacy alias) | Same effect as `DISABLE_MM_COOPMAT=1`. |
| `VULKANFORGE_COOPMAT_F16ACC=1` | off (FP32 acc) | Opt-in FP16 accumulator for the aligned-L-tile coopmat path. **RDNA4-neutral-to-slightly-negative** (emulated, not native). May benefit NVIDIA Ampere+ / Intel XMX hardware. New in v0.2.3. |
| `VULKANFORGE_FP16_KV=0` | on | Use FP32 KV cache (2× VRAM, parity with pre-v0.2.0). |
| `VULKANFORGE_COOPMAT_ATTN=0` | on | Disable coopmat QK attention; falls back to scalar tiled. **DEVICE_LOSTs at pp ≥ 4096** — debugging only. |
| `VULKANFORGE_BATCH_ATTN=0` | on | Per-token attention loop instead of batched. Parity testing only. |
| `VULKANFORGE_CB_REUSE=0` | on | Disable descriptor-set cache; pre-v0.1.0 codepath. |

### Driver-side flags (Mesa 26.1+)

| Variable | Effect |
|---|---|
| `RADV_PERFTEST=cswave32` | Compile compute shaders to Wave32 (enables RDNA4 VOPD dual-issue). Tested in Sprint 13D: ACO emits 3 546 dual-issue instructions, but wall-time is neutral on this workload (memory-bandwidth-bound, not VALU-bound). |

### Sampling (per-run)

`VF_TEMPERATURE`, `VF_TOP_K`, `VF_TOP_P`, `VF_REPETITION_PENALTY`,
`VF_SEED`. `VF_TEMPERATURE=0` (default) short-circuits to deterministic
argmax.

### GEMM tile-tuning (advanced)

`VULKANFORGE_GEMM_{BLOCK_SIZE,BM,BN,WM,WN,WMITER,TM,TN}` override the
spec-constants used to instantiate `mul_mmq` pipelines. Useful for A/B
tile sweeps without rebuilding SPV.

## Architecture

* `src/backend/vulkan/device.rs` — physical-device pick + queue family.
* `src/backend/vulkan/gguf.rs` — GGUF v3 parser + `ModelConfig` (auto-detects
  rope variant, qk-norm presence, vocab size, etc).
* `src/backend/vulkan/tokenizer.rs` — byte-level BPE for the `gpt2` tokenizer
  model. Picks the correct pre-split regex per `tokenizer.ggml.pre` (`qwen2` or
  `llama-bpe`).
* `src/backend/vulkan/spm.rs` — SentencePiece Unigram tokenizer (Mistral).
* `src/backend/vulkan/chat_template.rs` — `ChatTemplate` enum (ChatML / Llama3
  / DeepSeekR1 / Mistral / Raw) with auto-detection from the GGUF metadata.
* `src/backend/vulkan/forward.rs` — single-token + batched prefill graph.
  `layer_weight_shader_gemm` routes coopmat dispatches across S/M/L
  tiles, aligned/unaligned, and the f16acc opt-in path.
* `src/backend/vulkan/pipeline_registry.rs` — pipeline-layout +
  spec-constants, including the `mul_mm` S/M/L tile warptile blocks
  and the GEMV `MMV_NUM_ROWS` (= 1 — the v0.2.3 rationale for *not*
  going to 2 is in `results/v023_sprint13e_mmv_numrows.md`).

## Conventions

* Keep `unsafe` blocks scoped to single FFI calls.
* No swapchain, no graphics-queue paths.
* Spec-constants for the GEMV / GEMM / coopmat shaders are pinned in
  `pipeline_registry.rs` — RADV silently produces wrong results when a
  pipeline relies on GLSL defaults.
* Vulkan compute shaders ported from llama.cpp (`mul_mm.comp`,
  `mul_mmq.comp`, `mul_mat_vec_q*_k.comp`) are kept md5-identical to
  upstream HEAD. Performance differences are resolved through
  build-defines, spec-constants, SPV variants, and runtime routing
  rather than shader-source forks.

## Reports

Phase write-ups live in `results/`. Notable v0.2 series:

* `v02_sprint10c_coopmat_qk.md` — coopmat QK attention bring-up.
* `v021_sprint12c_gap_analysis.md` — pre-coopmat-GEMM gap analysis.
* `v021_sprint12gc_rgp_profiling.md` — per-dispatch GPU profiling.
* `v022_sprint12i_prefill_rgp.md` — prefill bottleneck root cause.
* `v022_sprint12k_q6k_coopmat.md` — Q6_K coopmat shader port.
* `v022_sprint12l_sml_tiles.md` — aligned LOAD_VEC_B=8 mat2x4.
* `v022_sprint12m_mtile.md` — M-tile + coopmat default-on.
* `v023_sprint13a_stile.md` — S-tile coopmat (BM=32) for pp ≤ 32.
* `v023_sprint13b_mesa26.1_test.md` — Mesa 26.1-rc3 driver test (neutral).
* `v023_sprint13c_f16acc.md` — f16-accumulator coopmat (opt-in, RDNA4-neutral).
* `v023_sprint13d_wave32_probe.md` — Wave32 / VOPD probe (neutral).
* `v023_sprint13e_mmv_numrows.md` — `MMV_NUM_ROWS=2` GEMV (slight regression).

## Limitations

* Single batch — concurrent sessions need separate `Forward` instances.
* **Decode at 0.80× llama.cpp Vulkan** — coopmat is prefill-only.
  Decode-side coopmat (e.g. `lm_head` GEMV) remains a v0.3 candidate.
* **Remaining ~0.10–0.15× prefill / ~0.20× decode gap** to llama.cpp
  is in pipeline-creation infrastructure, not shader source. Sprint 13
  systematically falsified five "port llama.cpp's config" hypotheses:
  Mesa 26.1-rc3 (neutral, 13B), f16 accumulator (emulated on RDNA4,
  13C), Wave32 / VOPD (memory-bound workload, 13D), and
  `MMV_NUM_ROWS=2` (needs subgroup-arithmetic reduction, 13E). The
  real next levers — `requiredSubgroupSize` pinning, subgroup-add
  GEMV reductions, multi-submit prefill, `quantize_q8_1` GEMM fusion
  — are v0.3-class infrastructure work.
* All compute shaders ported from llama.cpp (`mul_mm.comp`,
  `mul_mmq.comp`, `mul_mat_vec_q*_k.comp`) are **byte-identical to
  upstream HEAD `23b8cc4`**. Performance differences are configuration,
  not source.
* `VULKANFORGE_COOPMAT_ATTN=0` (explicit opt-out) still DEVICE_LOSTs at
  pp ≥ 4096 — scalar attention exceeds the RADV TDR window at long
  contexts. Default-ON works; opt-out is debugging-only.
