# Changelog

## v0.1.1 — Phase 5C: SPM Tokenizer + Mistral Support (2026-04-27)

### Headline

Adds SentencePiece tokenizer support, unblocking the Mistral / Llama-2
GGUF family. `Mistral-7B-Instruct-v0.3-Q4_K_M` joins Qwen3, Llama-3.1
and DeepSeek-R1 in the supported set — four models on a shared backend.

### Mistral-7B-Instruct-v0.3 — 5-prompt smoke

| Metric | Value |
|---|---:|
| Decode median | **102.7 tok/s** |
| Prefill median | 333.6 tok/s |
| Coherence | 5 / 5 |

Prompts: Mutex Explanation · Haiku · 2+2 · Translate-to-German · Prime
Check (Python). All five returned coherent, on-topic outputs.

### Added

- `src/backend/vulkan/spm.rs` — SentencePiece tokenizer using the
  greedy bigram-merge algorithm (mirrors llama.cpp's
  `llm_tokenizer_spm`). Score-priority merge over a doubly-linked-list
  of UTF-8-char symbols, byte-fallback (`<0xHH>`) and `<unk>` for
  unmappable chars, optional `▁` (U+2581) leading-space normalisation.
- `Tokenizer::is_spm` / `Tokenizer::encode_no_prefix` — SPM-aware
  helpers used by the Mistral chat template.
- `ChatTemplate::Mistral` — `<s>[INST] {body} [/INST]` rendering with
  the `[INST]` / `[/INST]` brackets emitted as their dedicated
  vocab ids (3 / 4) instead of being re-tokenised as ASCII. Auto-
  detected from the GGUF Jinja template.
- 4 new regression tests (`phase5c_*`) for the SPM tokenizer + Mistral
  chat template on `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`.
- 4 new lib unit tests covering byte-token parsing, UTF-8 char-length
  arithmetic, SPM normalisation, and the bigram priority queue.
- `examples/spm_dump.rs` — tokenizer / vocab / template diagnostic
  driver.
- `inference_test_prompts_mistral_5.json` — 5-prompt suite used by
  `run_15prompt_bench` for Mistral coherence checks.

### Changed

- `Tokenizer` is now a dispatch struct over an internal
  `TokenizerInner::{Bpe, Spm}` enum. Public field surface (`bos_id`,
  `eos_id`, `im_start_id`, …) is unchanged; the `flavour()` method
  now returns `Option<TokenizerFlavour>` and is `None` for SPM-backed
  tokenizers (Mistral, Llama-2).
- `ChatTemplate::detect` recognises Mistral templates by their `[INST]`
  marker before falling back to the tokenizer flavour. SPM-flavoured
  models with no Jinja signature default to `ChatTemplate::Mistral`.
- `TokenizerError::Malformed(String)` added for SPM array-length
  validation.

### Notes / non-goals

- No prefill optimisation in this release (Phase 5B target).
- No Llama-2 chat support (different template even though the
  tokenizer is the same SPM family — out of scope here).

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
