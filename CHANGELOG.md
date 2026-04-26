# Changelog

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
