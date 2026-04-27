# Changelog

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
