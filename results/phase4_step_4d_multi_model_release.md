# Phase 4D — Multi-Model + Polish

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RADV GFX1201, RDNA 4)
**Goal:** Extend the engine beyond Qwen3 to Meta-Llama-3.1, DeepSeek-R1, and
Mistral. Add automatic chat-template + RoPE-variant + Q/K-norm detection.
Write up the release notes (README + CHANGELOG) so the project is
presentable.

## TL;DR

* **Llama-3 family lit up cleanly.** Llama-3.1-8B-Instruct and
  DeepSeek-R1-Distill-Llama-8B both render coherent output across the
  5-prompt smoke. The "Llama-3.1 instruction-blind" failure mode that
  ROCmForge documented (7 hypotheses ruled out, never resolved) does
  **not** reproduce on VulkanForge.
* **Mistral is blocked, as predicted.** Tokenizer model is SentencePiece
  (`tokenizer.ggml.model = "llama"`); we only have a GPT-2 byte-level BPE.
  Fails at load with a clean `BadModel("llama")` error. Punted to Phase 5.
* **DeepSeek file in `~/models` is the Llama-distill, not the Qwen-distill.**
  The Phase-4D prompt mentioned `DeepSeek-R1-Distill-Qwen-7B`; the actual
  file present is `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`. We tested
  what was on disk.
* **Shipped infrastructure**: `RopeVariant`, `ChatTemplate` enum +
  auto-detection, generic special-token lookup, `cfg.has_qk_norm` gating
  on the Q/K-norm dispatches in `forward.rs`, `tokenizer.ggml.pre`-driven
  pre-split regex selection.
* **Tests**: 16/16 regression + 25/25 correctness pass; no Phase-3/4A/4B/4C
  parity test regressed.

## Probe results

`examples/probe_model.rs` dumped the relevant metadata up-front:

| Model | arch | tokenizer | chat_template marker | qk_norm | rope_base | rope_dim_count | bos/eos |
|---|---|---|---|---|---:|---:|---:|
| Qwen3-8B | qwen3 | gpt2/qwen2 | `<|im_start|>` (ChatML) | ✓ | 1 000 000 | 128 (key_length) | 151643/151645 |
| Llama-3.1-8B-Instruct | llama | gpt2/llama-bpe | `<|start_header_id|>` (Llama-3) | ✗ | 500 000 | 128 | 128000/128009 |
| DeepSeek-R1-Distill-Llama-8B | llama | gpt2/llama-bpe | `<｜User｜>` (DeepSeek-R1) | ✗ | 500 000 | 128 | 128000/**128001** |
| Mistral-7B-Instruct-v0.3 | llama | **llama (SPM)** | `[INST]` (Mistral) | ✗ | 1 000 000 | 128 | 1/2 |

DeepSeek and Llama-3.1 share the GGUF arch (`llama`), tokenizer model
(`gpt2`), pre-split (`llama-bpe`), and most architecture metadata. They
diverge on:
* the EOS id (128009 `<|eot_id|>` vs 128001 `<|end_of_text|>`),
* the embedded chat template (Llama-3 standard vs DeepSeek-R1 reasoning
  format with mandatory `<think>` priming).

The detector handles this: `ChatTemplate::detect` matches `<｜User｜>` in
the Jinja string and returns `DeepSeekR1` even though `general.architecture
= "llama"`.

## Implementation

### `ModelConfig` extensions (`gguf.rs`)

```rust
pub enum RopeVariant { Norm, Neox }

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // ...
        let rope_variant = match arch {
            "qwen2" | "qwen2moe" | "qwen2vl" | "qwen3" | "qwen3moe"
            | "phi2" | "phi3" | "gpt-neox" | "stablelm" => RopeVariant::Neox,
            _ => RopeVariant::Norm,
        };
        // ...
    }
}
```

`has_qk_norm` was already detected from tensor presence
(`blk.0.attn_q_norm.weight`); Phase 4D wires it through the forward graph.

### Tokenizer flavour (`tokenizer.rs`)

```rust
pub enum TokenizerFlavour { Qwen2, Llama3 }

let flavour = match pre {
    "qwen2" => TokenizerFlavour::Qwen2,
    "llama-bpe" => TokenizerFlavour::Llama3,
    other => return Err(TokenizerError::BadModel(format!("pre={other}"))),
};
```

Llama-3 picks a slightly different pre-split regex (`\p{N}{1,3}` instead
of `\p{N}+`) and a wider EOS set (`<|eot_id|>`, `<|end_of_text|>`,
`<|eom_id|>` all terminate decode). The Qwen-specific
`im_start_id`/`im_end_id`/`endoftext_id` fields became `Option<u32>`.

### `ChatTemplate` (`chat_template.rs`)

A new module owns four templates:
* `ChatML` — `<|im_start|>role\n…<|im_end|>\n` (Qwen)
* `Llama3` — `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`
* `DeepSeekR1` — `<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜><think>\n`
  (the trailing `<think>` is part of the priming so reasoning starts
  automatically — this matches the HuggingFace template exactly)
* `Raw` — no special tokens, `BOS + system + \n + user`.

`ChatTemplate::detect(gguf, tokenizer)` first scans the embedded Jinja
template for unique markers (`<｜User｜>` → DeepSeek; `<|start_header_id|>`
→ Llama-3; `<|im_start|>` → ChatML), and only falls back to the tokenizer
flavour if no marker matches. The DeepSeek-on-Llama-arch case forced this
order: arch alone would have shipped the wrong template.

### Forward graph (`forward.rs`)

Three call sites guarded:

```rust
if cfg.has_qk_norm {
    let wqn = layer_weight(model, layer, "attn_q_norm.weight");
    let wkn = layer_weight(model, layer, "attn_k_norm.weight");
    self.run_rms_norm(/* …Q… */);
    self.run_rms_norm(/* …K… */);
    compute_barrier(dev, cmd);
}
```

`run_rope_neox_with_pos_offset` now dispatches the right shader:
```rust
let (shader_id, rope_mode) = match self.config.rope_variant {
    RopeVariant::Neox => (ShaderId::RopeNeox, 2u32),
    RopeVariant::Norm => (ShaderId::RopeNorm, 0u32),
};
```
The `RopeNorm` SPIR-V was already in the registry (Phase 2A built it as
part of the inventory); only the Phase-4D dispatch path is new.

## 5-prompt benchmark (5/15 from `inference_test_prompts_15.json`)

All runs: `VF_NUM_PROMPTS=5 cargo run --release --example run_15prompt_bench`
on RX 9070 XT, validation layer ON. Aggregate is over the 5 prompts only.

| Model | Decode (median tok/s) | Prefill (median tok/s) | Decode aggregate | Coherent |
|---|---:|---:|---:|---:|
| Qwen3-8B | 72.4 | 288.1 | 68.7 tok/s | 5/5 |
| Llama-3.1-8B | 81.5 | 358.2 | 75.9 tok/s | 5/5 *(see note)* |
| DeepSeek-R1-Distill-Llama-8B | 81.3 | 306.7 | 77.8 tok/s | 5/5 |

*Note on Llama-3.1*: the bench's coherence heuristic flagged prompt #2
(`Zähle von 1 bis 10`) as `✗` because the response is digits-only and
its threshold is "≥4 alphabetic chars". The actual output:

```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
```

…is correct and EOS-terminated at 29 tokens. All five Llama-3.1 outputs
are genuine, on-topic answers; the `4/5` printed by the bench is a false
positive in the heuristic, not a model failure.

### Decode-quality samples

**Llama-3.1 — "What is 2+2?"** (eos=true, 6 generated tokens):
```
The answer is 4.
```

**Llama-3.1 — "Write a Python function that checks if a number is prime."**
```python
def is_prime(n):
    """Checks if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```
…followed by a correct three-paragraph explanation. Stopped on EOS.

**DeepSeek-R1-Distill-Llama — "What is 2+2?"** (in `<think>` reasoning
phase as expected for an R1 model):
```
Okay, so I need to figure out what 2 plus 2 is. Hmm, let me think about
this step by step. First, I know that addition is a basic math
operation, and it's one of the first things you learn. So, adding two
numbers together should be straightforward. […]
```

The `<think>` block continues past the 120-token cap; that's normal for
R1 models and matches their HuggingFace behaviour. With
`think_filter=true`, only the post-`</think>` answer would be streamed
to the user.

## Why Llama-3.1 works here but not on ROCmForge

The ROCmForge incident report
(`~/projects/ROCmForge/results/inference_test_20260425.md`) ruled out
seven hypotheses for the "instruction-blind" failure (model was generating
completions but not *answering* the user). VulkanForge wired up the same
chat-template, RoPE variant, and EOS detection from scratch in Phase 4D
and "What is 2+2?" answers cleanly with "The answer is 4." The Phase-4D
prompt was correct that we shouldn't sink time into debugging the ROCm
issue — the failure was almost certainly localised to the HIP backend's
attention or RoPE path on ROCmForge, not the cross-cutting concerns
VulkanForge had to implement anyway. We carried over none of that
debugging effort and got working output on the first try after the
infrastructure was in place.

## Mistral SPM blocker

The Mistral GGUF declares `tokenizer.ggml.model = "llama"`, vocab=32768,
plus a `tokenizer.ggml.scores` array — the tell-tale signature of a
SentencePiece unigram tokenizer (Llama-2 / Mistral / Falcon-3 use it).
VulkanForge's `Tokenizer::from_gguf` rejects this up-front:

```
Error: BadModel("llama")
```

Implementing SPM means adding:
1. A unigram vocabulary table (token + log-prob).
2. A Viterbi-style best-segmentation pass replacing the current BPE
   merge loop.
3. A new pre-tokenizer (typically just whitespace + `▁` byte-fallback).
4. A different decode path (the `▁` U+2581 character is treated as space).

That's a 200-300 LoC addition, not blocking on Phase 4D's other work, so
it's deferred to Phase 5. The decision is documented in CHANGELOG.md.

## DeepSeek model name discrepancy

The Phase-4D prompt mentioned `DeepSeek-R1-Distill-Qwen-7B`. What's on
disk in `~/models` is `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`. The
two distills differ in:
* base architecture (Llama vs Qwen) and consequently RoPE variant + Q/K-norm,
* tokenizer (gpt2/llama-bpe vs gpt2/qwen2),
* chat template (DeepSeek-R1 stays the same Jinja, but the surrounding
  EOS / BOS namespaces change).

We tested the Llama-distill since it was the available file. The
infrastructure (`ChatTemplate::DeepSeekR1`) is identical for both — the
only difference at runtime is which arch metadata flows into
`ModelConfig`, which the Phase-4D code path handles.

## Files touched

```
src/backend/vulkan/gguf.rs          + RopeVariant, rope_variant field
src/backend/vulkan/tokenizer.rs     + Llama3 flavour, eos_id, special_id, llama-bpe regex
src/backend/vulkan/chat_template.rs (new file) ChatTemplate enum + detect + render
src/backend/vulkan/chat.rs          + new_with_template, dispatch via ChatTemplate
src/backend/vulkan/decode.rs        ChatTemplate::detect in generate()
src/backend/vulkan/forward.rs       has_qk_norm gating, RoPE variant dispatch
src/backend/vulkan/mod.rs           pub mod chat_template
examples/probe_model.rs             (new) per-model metadata dump
examples/sample_decode.rs           (new) one-prompt full-text sampler
examples/run_15prompt_bench.rs      VF_NUM_PROMPTS, ChatTemplate auto-detect
examples/dump_tokenizer_test.rs     Option<u32> field updates
tests/regression.rs                 Option<u32> field updates
README.md                           Phase-4D status table + run instructions
CHANGELOG.md                        new — release notes since Phase 4C
```

## Open follow-ups

* **SPM tokenizer** for Mistral / Llama-2 / Falcon-3 — Phase 5.
* **DeepSeek-R1-Distill-Qwen-7B** — would test that DeepSeek template
  works on Qwen arch too. Need to download the GGUF.
* **Bench coherence heuristic** — currently flags digits-only outputs
  as `✗`. Adjusting it would only matter if we keep using it as a
  release gate.
* **Sampling** — temperature / top-k / top-p. Useful for chat UX but not
  required for the deterministic bench.
* **Quantised KV cache** — would buy back ~50% of the KV memory budget
  and help fit a 32k-context Llama-3.1 onto the 16 GiB card.

## Status

Phase 4D done. Three model families verified end-to-end on real GPU
hardware, with the infrastructure (`RopeVariant`, `ChatTemplate`,
`has_qk_norm` gating) already in place to drop in any new arch that
uses the GPT-2 byte-level BPE.
