# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

**v0.2.2** — KHR cooperative-matrix WMMA prefill is now **default-on**, closing
the prefill gap to llama.cpp Vulkan from 0.52× (v0.2.0) to **0.89×** at
pp=512. Single-batch greedy decode (with optional temperature / top-k /
top-p sampling) + multi-turn chat sessions with persistent KV cache.
Supports the Qwen, Llama-3, DeepSeek-R1 and Mistral GGUF families
out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | llama (SPM) | Mistral | ✅ |

### Key features (v0.2.2)

- **KHR cooperative matrix WMMA prefill** (default ON) — Q4_K and Q6_K
  GEMM dispatched through RDNA4's 128 AI Accelerators via
  `VK_KHR_cooperative_matrix`. L-tile (BM=128) + M-tile (BM=64)
  pipelines with a runtime selector that mirrors llama.cpp's
  `ggml_vk_guess_matmul_pipeline`. Aligned variant uses `LOAD_VEC_B=8`
  with `B_TYPE=mat2x4` for 4× wider B-matrix loads. Opt out with
  `VULKANFORGE_DISABLE_MM_COOPMAT=1`.
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

| pp   | VulkanForge v0.2.2 | VulkanForge v0.2.0 | llama.cpp Vulkan | Ratio (v0.2.2) |
|------|-------------------:|-------------------:|-----------------:|---------------:|
|   64 |              1678  |              1511  |             2285 | 0.73× |
|  128 |              2560  |              2001  |             3637 | 0.70× |
|  256 |              3558  |              2200  |             3995 | 0.89× |
|  512 |          **3863**  |              2255  |             4326 | **0.89×** |
| 1024 |              3748  |              2204  |             4173 | 0.90× |
| 2048 |              3172  |              1997  |             3765 | 0.84× |

llama.cpp reference: build 23b8cc4 with `-fa 1` on the same hardware.
Peak prefill throughput is **3863 tok/s @ pp=512**, up from 2255 in
v0.2.0 (+71 %). The pp ≥ 256 ratio is now within run-to-run noise of
llama.cpp's `mul_mm` Vulkan path.

### 4-system comparison (Qwen3-8B, same hardware)

| System                     | Decode tok/s | Prefill peak tok/s | Decode ratio | Prefill ratio |
|----------------------------|-------------:|-------------------:|-------------:|--------------:|
| llama.cpp Vulkan           |      114.2   |              4326  |       1.00×  |        1.00×  |
| **VulkanForge v0.2.2**     |   **91.1**   |          **3863**  |    **0.80×** |     **0.89×** |
| VulkanForge v0.2.0         |       90.5   |              2255  |       0.79×  |        0.52×  |
| llama.cpp ROCm             |       87.5   |              3684  |       0.77×  |        0.85×  |
| ROCmForge (HIP)            |       95.4   |               769  |       0.84×  |        0.18×  |

vs v0.2.0: decode +0.7 % (90.5 → 91.1, run-to-run noise), prefill peak
**+71 %** (2255 → 3863). Decode unchanged because coopmat is
prefill-only — GEMV continues through the existing `mul_mat_vec_*`
shaders.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 27/27 lib tests
```

MSRV is **Rust 1.85** (edition 2024). Build dependencies require a working
`shaderc` install (the `shaderc-sys` crate); on Arch / CachyOS this is
`shaderc` from the official repos. `VK_KHR_cooperative_matrix` must be
advertised by the driver — RADV gfx1201 with Mesa 26.0.5+ qualifies.

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

| Variable | Default | Effect when disabled |
|---|---|---|
| `VULKANFORGE_DISABLE_MM_COOPMAT=1` | off (coopmat ON) | Falls back to scalar `mul_mmq` GEMM (v0.2.1 behaviour). |
| `VULKANFORGE_USE_MM_COOPMAT=0` | (legacy alias) | Same effect as `DISABLE_MM_COOPMAT=1`. |
| `VULKANFORGE_FP16_KV=0` | on | Use FP32 KV cache (2× VRAM, parity with pre-v0.2.0). |
| `VULKANFORGE_COOPMAT_ATTN=0` | on | Disable coopmat QK attention; falls back to scalar tiled. **DEVICE_LOSTs at pp ≥ 4096** — debugging only. |
| `VULKANFORGE_BATCH_ATTN=0` | on | Per-token attention loop instead of batched. Parity testing only. |
| `VULKANFORGE_CB_REUSE=0` | on | Disable descriptor-set cache; pre-v0.1.0 codepath. |

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
  v0.2.2 adds `layer_weight_shader_gemm` for coopmat tile/aligned/M-tile
  routing.
* `src/backend/vulkan/pipeline_registry.rs` — pipeline-layout +
  spec-constants, including the `mul_mm` L/M-tile warptile blocks.

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

## Limitations

* Single batch — concurrent sessions need separate `Forward` instances.
* Decode at 0.80× llama.cpp Vulkan — coopmat is prefill-only.
  Decode-side coopmat (e.g. `lm_head` GEMV) remains a v0.3 candidate.
* The remaining ~0.10–0.15× peak-WMMA gap to llama.cpp likely requires
  the `f16acc` coopmat variant llama.cpp ships; deferred.
* Short-prompt regression in the 15-prompt suite (pp ≤ 50) — M-tile
  is partially undersaturated below pp=64. Workaround:
  `VULKANFORGE_DISABLE_MM_COOPMAT=1` for short-prompt-dominated
  workloads. An S-tile (BM=32) variant is the natural follow-up.
* `VULKANFORGE_COOPMAT_ATTN=0` (explicit opt-out) still DEVICE_LOSTs at
  pp ≥ 4096 — scalar attention exceeds the RADV TDR window at long
  contexts. Default-ON works; opt-out is debugging-only.
