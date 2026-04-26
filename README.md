# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

Phase 4D (Multi-Model + Polish). Single-batch greedy decode + multi-turn chat
sessions with persistent KV cache. Supports the Qwen and Llama-3 GGUF families
out-of-the-box.

| Model | Arch | Tokenizer | Chat template | Status |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | qwen3 | gpt2 / qwen2 | ChatML | ✅ reference |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | llama | gpt2 / llama-bpe | Llama3 | ✅ |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | llama | gpt2 / llama-bpe | DeepSeek-R1 | ✅ |
| Mistral-7B-Instruct-v0.3.Q4_K_M | llama | **llama (SPM)** | Mistral | ❌ deferred |

Mistral fails at tokenizer load with `BadModel("llama")` — the `llama` tokenizer
model is SentencePiece (vocab=32768, scores array). VulkanForge only ships a
GPT-2 byte-level BPE today; an SPM unigram decoder is planned for Phase 5.

Gemma-4 is out of scope for Phase 4D (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, 5-prompt subset of the 15-prompt suite)

| Model | Decode tok/s (median) | Prefill tok/s (median) | Coherent |
|---|---:|---:|---:|
| Qwen3-8B (reference) | 72.4 | 288.1 | 5/5 |
| Llama-3.1-8B-Instruct | 81.5 | 358.2 | 5/5 |
| DeepSeek-R1-Distill-Llama-8B | 81.3 | 306.7 | 5/5 |

Llama-3.1 decode is ~12% faster than Qwen3 because Llama has 32 layers (Qwen3
has 36) and no Q/K-norm dispatches per layer. Reference numbers from Phase 4C:
llama.cpp Vulkan 114 tok/s, llama.cpp ROCm 88 tok/s on the same hardware.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release --tests      # 16 + 25 = 41 tests
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
* `phase4_step_4d_multi_model_release.md` — this phase

## Limitations

* Greedy decode only (no temperature / top-k / top-p sampling).
* No quantized cache (KV is f32, ~2 GiB at 8k context).
* Single batch — concurrent sessions need separate `Forward` instances.
* SPM tokenizer not implemented — Mistral / Llama-2 are blocked on this.
* No coopmat / WMMA path — Phase 4 attention work brought decode to ~70%
  of llama.cpp Vulkan; closing the rest is Phase 5+ work.
