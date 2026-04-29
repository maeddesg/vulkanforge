# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

**v0.2.0** — coopmat QK attention + FP16 KV-cache + 5 fused kernels.
Single-batch greedy decode (with optional temperature / top-k / top-p
sampling) + multi-turn chat sessions with persistent KV cache.
Supports the Qwen, Llama-3, DeepSeek-R1 and Mistral GGUF families out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | llama (SPM) | Mistral | ✅ |

### Key features (v0.2.0)

- **coopmat QK attention** — KHR_cooperative_matrix WMMA replaces the scalar
  inner loop in `flash_attn_coopmat.comp`. ~85 % faster prefill at
  pp=2048 vs scalar; resolves the pp=4096 TDR crash entirely.
- **FP16 KV-cache** (default ON) — half the cache VRAM, +21 % prefill at
  pp=2048. Opt out with `VULKANFORGE_FP16_KV=0`.
- **5 fused kernels** — `swiglu`, `multi_add_rms` (×2 sites),
  `rms_norm_mul_rope` — −5 dispatches per layer.
- **Tiled flash-attention** — Br=16 / Bc=32 with online softmax,
  Br/Bc swept across the prefill range.
- **pp=4096 supported** — previously crashed with TDR; now 1659 tok/s.

Gemma-4 is out of scope (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, gfx1201, RDNA 4)

### Prefill throughput sweep (Qwen3-8B-Q4_K_M, RUNS=5 median)

| pp   | VulkanForge v0.2.0 | llama.cpp Vulkan | Ratio |
|------|-------------------:|-----------------:|------:|
|   64 |              1511  |             2286 | 0.66× |
|  128 |              2001  |             3603 | 0.56× |
|  256 |              2200  |             3999 | 0.55× |
|  512 |          **2255**  |             4317 | 0.52× |
| 1024 |              2204  |             4189 | 0.53× |
| 2048 |              1997  |             3771 | 0.53× |
| 4096 |              1659  |             3272 | 0.51× |

llama.cpp reference: build 23b8cc4 with `-fa 1` on the same hardware.
Peak prefill throughput is 2255 tok/s @ pp=512. **pp=4096 used to
DEVICE_LOST (TDR);** Sprint 10E's coopmat-default-ON brings the
last-chunk attention dispatch back under the watchdog.

### 15-prompt benchmark suite (Qwen3-8B-Q4_K_M)

| Metric                     | Value          |
|----------------------------|----------------|
| **Median prefill tok/s**   | **1068**       |
| **Median decode tok/s**    | **90.5**       |
| Aggregate prefill tok/s    | 1089           |
| Aggregate decode tok/s     | 85.5           |
| Coherent prompts           | 15/15          |
| Total prompt tokens        | 802            |
| Total decode tokens        | 6080           |

The 15-prompt median is dominated by short prompts (pp=20–198) and so
sits well below the pp-sweep peak.

### 4-system comparison (Qwen3-8B, same hardware)

| System                     | Decode tok/s | Prefill peak tok/s | Decode ratio | Prefill ratio |
|----------------------------|-------------:|-------------------:|-------------:|--------------:|
| llama.cpp Vulkan           |      114.2   |              4317  |       1.00×  |        1.00×  |
| **VulkanForge v0.2.0**     |   **90.5**   |          **2255**  |    **0.79×** |     **0.52×** |
| llama.cpp ROCm             |       87.5   |              3684  |       0.77×  |        0.85×  |
| ROCmForge (HIP)            |       95.4   |               769  |       0.84×  |        0.18×  |

vs v0.1.3: decode +2.1 % (88.6 → 90.5), prefill peak **+118 %** (1037
→ 2255) — coopmat QK + FP16 KV + the fused kernels combined. `Alice`
multi-turn context-retention test still passes 3/3 on every model.

Decode is at 76 % of llama.cpp Vulkan and **above** llama.cpp ROCm /
ROCmForge HIP across all four models. The v0.1.3 prefill numbers are
~6–10 % below the (incorrect) v0.1.2 figures because the GEMM now
covers the full BM × BN tile instead of half of it.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 167 tests across 7 binaries
```

MSRV is **Rust 1.85** (edition 2024). Build dependencies require a working
`shaderc` install (the `shaderc-sys` crate); on Arch / CachyOS this is
`shaderc` from the official repos.

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

Drop a tokenizer / metadata sanity probe on any GGUF:
```bash
VF_MODEL_PATH=$HOME/models/<file>.gguf \
  cargo run --release --example probe_model
```

## Architecture

* `src/backend/vulkan/device.rs` — physical-device pick + queue family.
* `src/backend/vulkan/gguf.rs` — GGUF v3 parser + `ModelConfig` (auto-detects
  rope variant, qk-norm presence, vocab size, etc).
* `src/backend/vulkan/tokenizer.rs` — byte-level BPE for the `gpt2` tokenizer
  model. Picks the correct pre-split regex per `tokenizer.ggml.pre` (`qwen2` or
  `llama-bpe`).
* `src/backend/vulkan/chat_template.rs` — `ChatTemplate` enum (ChatML / Llama3
  / DeepSeekR1 / Raw) with auto-detection from the GGUF metadata.
* `src/backend/vulkan/forward.rs` — single-token + batched prefill graph.
* `src/backend/vulkan/forward.rs::run_flash_attn_split_reduce` — Phase-4C
  multi-WG attention (worker + reducer with online softmax merge).
* `src/backend/vulkan/forward.rs::alloc_or_get_set` — Phase-5A descriptor-
  set cache (eliminates the per-token `vkAllocateDescriptorSets` /
  `vkUpdateDescriptorSets` overhead on the decode hot path; on by default,
  set `VULKANFORGE_CB_REUSE=0` to disable).

## Conventions

* Keep `unsafe` blocks scoped to single FFI calls.
* No swapchain, no graphics-queue paths.
* Spec-constants for the GEMV / GEMM shaders are pinned in
  `pipeline_registry.rs` — RADV silently produces wrong results when the
  pipeline relies on GLSL defaults (Phase-2A bug §4).

## Reports

Phase write-ups live in `results/`:
* `phase4_step_4a_vgpr_reduction.md` — negative result on shader-side VGPR cuts
* `phase4_step_4b_flash_attention.md` — online-softmax flash-attention drop-in
* `phase4_step_4c_multi_wg_attention.md` — split-K multi-WG attention (+41%)
* `phase4_step_4d_multi_model_release.md` — multi-model + chat templates
* `phase5a_step_1_dgc_poc.md` — VK_EXT_device_generated_commands study (NO-GO)
* `phase5a_step_2_cb_reuse.md` — CPU-profile + descriptor-set-cache (Stage 2D)
* `phase5a_step_3_ship.md` — CB-reuse default-on + 15-prompt all models

## Limitations

* Single batch — concurrent sessions need separate `Forward` instances.
* GEMM stays scalar (`mul_mmq.comp`) — the coopmat work in v0.2 covers QK
  attention only. A coopmat GEMM would close most of the remaining
  ~48 % prefill gap to llama.cpp Vulkan; that's v0.3 territory.
* PV-coopmat attempted in Sprint 10D (LDS-scratch hybrid) regressed end-to-end
  by 1–24 % and was reverted; documented as honest negative in
  `results/v02_sprint10d_pv_coopmat.md`.
* `VULKANFORGE_COOPMAT_ATTN=0` (explicit opt-out) still DEVICE_LOSTs at
  pp ≥ 4096 — scalar attention exceeds the RADV TDR window at long
  contexts. Default-ON works; opt-out is debugging-only.
