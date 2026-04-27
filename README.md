# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

v0.1.1 — Phase 5C (SPM Tokenizer + Mistral). Single-batch greedy decode +
multi-turn chat sessions with persistent KV cache. Supports the Qwen, Llama-3
and Mistral GGUF families out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | llama (SPM) | Mistral | ✅ new in v0.1.1 |

Gemma-4 is out of scope for v0.1.1 (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, gfx1201, RDNA 4)

15-prompt suite for the Phase 5A models:

| Model | Decode tok/s (median) | Prefill tok/s (median) | Coherent |
|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | 88.5 | 404.9 | 14/15 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 94.6 | 489.9 | 13/15 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 94.8 | 433.9 | 15/15 |

5-prompt smoke for Mistral (added in v0.1.1):

| Model | Decode tok/s (median) | Prefill tok/s (median) | Coherent |
|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3.Q4_K_M | 102.7 | 333.6 | 5/5 |

`Coherent` is the bench's automatic ✓/✗ heuristic; the false-negatives on
Qwen3 (1) and Llama-3.1 (2) are digits-only / emoji-only outputs that the
heuristic flags but are actually correct.

Reference 4-system comparison on the same hardware:

| System | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| llama.cpp Vulkan | 114.2 | 4314 |
| **VulkanForge (this repo, Phase 5A)** | **88.5–94.8** | **404–489** |
| llama.cpp ROCm | 87.5 | 3684 |
| ROCmForge (HIP backend) | 95.4 | 768.6 |

Decode performance is now within ~17 % of llama.cpp Vulkan and ahead of
llama.cpp ROCm. Prefill remains a Phase 5B target.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 19 + 25 + 21 = 65 tests
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
