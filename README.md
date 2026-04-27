# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

v0.1.3 — Phase 7 mul_mm.comp debug + silent mul_mmq fix.
Single-batch greedy decode (with optional temperature / top-k / top-p
sampling) + multi-turn chat sessions with persistent KV cache.
Supports the Qwen, Llama-3 and Mistral GGUF families out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | llama (SPM) | Mistral | ✅ |

Gemma-4 is out of scope (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, gfx1201, RDNA 4)

> **v0.1.3 numbers below are the first correct prefill measurements.**
> All v0.1.0 – v0.1.2 prefill numbers were inflated by a `BLOCK_SIZE = 128`
> bug that left columns 32–63 of every output tile unwritten — half the
> GEMM work was silently skipped, which made prefill *appear* ~7–10 %
> faster than it actually was. See `results/phase7_mul_mm_debug.md` for
> the full investigation.

Full 15-prompt benchmark suite + 6-turn Alice multi-turn test
(prompt 16) for all four supported models, on `BLOCK_SIZE = 256`:

| Model | Decode tok/s (median) | Prefill tok/s (median) | Coherent | Alice |
|---|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M | 88.6 | 1037.4 | 15/15 | 3/3 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 94.8 | 1092.7 | 12/15 | 3/3 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 94.3 | 904.1 | 15/15 | 3/3 |
| Mistral-7B-Instruct-v0.3.Q4_K_M | 100.1 | 939.3 | 15/15 | 3/3 |

`Coherent` is the bench's automatic ✓/✗ heuristic; the Llama-3.1
false-negatives are digits-only / very short numeric replies that the
heuristic's "repeating garbage" check flags but the underlying output
is correct (the multi-turn `Alice` test passes 3/3 on every model,
and the regression suite's `phase3e` top-1 / top-5 parity gates pass
identically to v0.1.2). `Alice` is the multi-turn context-retention
test asking the model to recall "Alice" / "Berlin" across 6 turns.

Reference 4-system comparison on the same hardware (Qwen3-8B,
llama.cpp Vulkan build 23b8cc4 with `-fa 1`, tg128 / pp62):

| System | Decode tok/s | Prefill tok/s | Decode ratio | Prefill ratio |
|---|---:|---:|---:|---:|
| llama.cpp Vulkan | 116.2 | 2274 | 1.00× | 1.00× |
| **VulkanForge v0.1.3** | **88.6** | **1037 (med, 15-prompt)** | **0.76×** | **~0.46×*** |
| llama.cpp ROCm | 87.5 | 3684 | 0.75× | 1.62× |
| ROCmForge (HIP) | 95.4 | 768.6 | 0.82× | 0.34× |

*Prefill ratio is mixed-prompt-length on our side (20–200 tokens) vs
the fixed pp62 batch on llama.cpp's; at pp=62 specifically the
REST-API prompt hits 1418 tok/s → 62 % of llama.cpp's 2274 tok/s;
at pp=200 the gap widens (longer-prompt GEMM utilisation is the
next bottleneck).

Decode is at 76 % of llama.cpp Vulkan and **above** llama.cpp ROCm /
ROCmForge HIP across all four models. The v0.1.3 prefill numbers are
~6–10 % below the (incorrect) v0.1.2 figures because the GEMM now
covers the full BM × BN tile instead of half of it.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 24 + 44 + 25 = 93 tests
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

* Greedy decode only (no temperature / top-k / top-p sampling).
* No quantized cache (KV is f32, ~2 GiB at 8k context).
* Single batch — concurrent sessions need separate `Forward` instances.
* SPM tokenizer not implemented — Mistral / Llama-2 are blocked on this.
* No coopmat / WMMA path — Phase 4 attention + Phase 5A CB-reuse brought
  decode to ~83 % of llama.cpp Vulkan; closing the remaining ~17 % is
  Phase 5+ work (likely a coopmat-style GEMV / improved attention shader).
