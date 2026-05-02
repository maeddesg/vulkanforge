# v0.3.2 Sprint 17A — Llama-3.1 / Mistral / DeepSeek-R1-Distill

**Result:** Three new models running end-to-end on the v0.3 forward
pass. Total code change: **3 lines + a `use` statement.** No new
shaders, no new templates, no per-arch forward-pass branches.

| Model                                       | Arch  | Layers | Decode (greedy) | Status |
|---------------------------------------------|-------|--------|----------------:|--------|
| Qwen3-8B Q4_K_M                             | qwen3 | 36     | 108.7 tok/s     | reference (15 / 15 coherent) |
| Meta-Llama-3.1-8B-Instruct Q4_K_M           | llama | 32     | **121.7 tok/s** | new |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M         | llama | 32     | **113.5 tok/s** | new |
| Mistral-7B-Instruct-v0.3 Q4_K_M             | llama | 32     | **124.0 tok/s** | new |

Bench column is the in-binary `vulkanforge bench --runs 3` decode
median; the 15-prompt regression harness is Qwen3-only and stays at
108.7 tok/s — no regression on the reference model. Llama-3.1 prefill
also lifts modestly (3991 tok/s pp=512 vs Qwen3's 3865) since it has
4 fewer transformer layers per token.

## Pre-check (per memory `feedback_sprint_hypothesis_check.md`)

The brief proposed implementing:
- a **bias toggle** for Qwen `attn_q.bias` vs Llama (no bias)
- a **Llama-3 chat template** with `<|start_header_id|>` / `<|eot_id|>`
- **architecture-based template selection**
- **dim-from-config** plumbing in case dims were hardcoded

Pre-check disproved every premise:

| Brief claim | Actual state |
|---|---|
| "ModelConfig::from_gguf hardcoded to Qwen3" | Already arch-generic via `{arch}.X` keys (since Sprint 16B). |
| "Tensor names need mapping" | Forward pass uses standard llama.cpp names (`attn_q.weight`, `ffn_gate.weight`, …) — same on Llama and Qwen. |
| "Bias toggle needed" | No bias-add path exists in the forward pass for either family. Qwen3 8B Q4_K_M GGUF doesn't actually ship `attn_q.bias`; Qwen runs without it at 109 tok/s. |
| "Q/K-norm is Qwen-only" | Already gated by `cfg.has_qk_norm`, detected by tensor presence (`gguf.rs:408`). |
| "RoPE differs (Llama Norm vs Qwen Neox)" | Already arch-selected at `gguf.rs:415` — Qwen* → `Neox`, others → `Norm`. |
| "Llama-3 chat template needed" | `ChatTemplate::Llama3` plus `render_llama3_first` / `_continuation` already shipped (`chat_template.rs:101,114`). DeepSeek-R1 and Mistral templates also in place. |
| "Architecture-based template selection needed" | `ChatTemplate::detect()` already inspects the embedded Jinja `tokenizer.chat_template` and falls back to `Tokenizer::flavour()` (`chat_template.rs:49`). |
| "FFN dim 14336 vs 12288 needs handling" | All buffer sizes derived from `cfg.ffn_dim`. |
| "Vocab 128256 vs 151936 needs handling" | All buffer sizes derived from `cfg.vocab_size`. |

## What was actually broken

**One bug, one missing wire-up.**

### `src/main.rs:284` — `ChatSession::new()` hardcoded ChatML

Before:
```rust
let mut session = ChatSession::new(forward, system_prompt.clone());
```

`ChatSession::new()` constructs with `ChatTemplate::ChatML` regardless
of model. `ChatTemplate::detect()` was wired and worked, but no caller
in `main.rs` ever invoked it. Result: any non-Qwen model that lacked
`<|im_start|>` / `<|im_end|>` in its vocab panicked at first prefill
(`render_chatml_first` calls `.expect("ChatML render needs <|im_start|>")`).

After:
```rust
let template = ChatTemplate::detect(&gguf, &tokenizer);
let mut session = ChatSession::new_with_template(
    forward, system_prompt.clone(), template,
);
```

This was a Sprint 14-or-earlier bug that hid behind
"`vulkanforge chat` only runs on Qwen3" — uncovered the moment
preflight stopped blocking Llama.

### `src/main.rs:645` — `inference_support()` whitelist

Before:
```rust
let arch_ok = matches!(arch, "qwen2" | "qwen3");
```

After:
```rust
let arch_ok = matches!(arch, "qwen2" | "qwen3" | "llama");
```

That's it.

## Live-test transcripts

```
$ vulkanforge chat --model ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
Model:   .../Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
  4.58 GiB · 32 layers · hidden=4096 · heads=32 · kv_heads=8 · head_dim=128
  vocab=128256 · ctx_max=2048 · rope_freq_base=500000

> What is 2+2? Answer in one short sentence.
The answer is 4.
[34 prompt, 6 gen, prefill 536 tok/s, decode 109.9 tok/s]

$ vulkanforge chat --model ~/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
> What is 2+2? Answer in one short sentence.
Okay, so I need to figure out what 2 plus 2 is. Hmm, let me think.
I remember from school that addition is combining quantities …
[25 prompt, 80 gen, prefill 389 tok/s, decode 113.5 tok/s]

$ vulkanforge chat --model ~/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
> What is 2+2? Answer in one short sentence.
The sum of 2 and 2 is 4.
[24 prompt, 12 gen, prefill 583 tok/s, decode 124.0 tok/s]
```

DeepSeek-R1-Distill correctly emits its thinking-mode preamble; the
existing `<think>…</think>` filter handles the wrap when enabled.

## Notes on the three models

- **Meta-Llama-3.1-8B-Instruct** — embedded chat template uses
  `<|start_header_id|>` / `<|eot_id|>`, picked up by
  `ChatTemplate::detect()`'s first pattern check. Tokenizer flavour
  fallback also resolves to `Llama3`.
- **DeepSeek-R1-Distill-Llama-8B** — GGUF self-reports
  `architecture=llama` (it's a Llama-3 distill). Embedded chat
  template ships `<｜User｜>` / `<｜Assistant｜>` glyphs, picked up
  as `ChatTemplate::DeepSeekR1`.
- **Mistral-7B-Instruct-v0.3** — also self-reports
  `architecture=llama` in its GGUF; embedded chat template uses
  `[INST]` / `[/INST]`, picked up as `ChatTemplate::Mistral`. SPM
  tokenizer (`tokenizer.ggml.model="llama"`).

## Regressions

- **Qwen3-8B Q4_K_M** — `cargo run --release --example
  run_15prompt_bench`: **15 / 15 coherent**, median decode 108.7
  tok/s, median prefill 825.7 tok/s. Within run-to-run noise of v0.3.1.
- **`cargo test --release --lib`** — **27 / 27**.

## Known issue, *not* introduced by this sprint

`Qwen2.5-0.5B-Instruct-Q4_K_M` (the only Qwen2.5 in Q4_K_M; the 7B and
14B variants are Q4_0 and gated by quant) crashes during prefill with
`embeddings length mismatch`. Stack:

```
❌ generation error: prefill_batch: embeddings length mismatch
```

Likely a pre-existing assumption mismatch with the model's unusual
dims (`hidden_dim=896`, `head_dim=64`, vs the bread-and-butter 4096 /
128 the forward path was tuned for). Predates Sprint 17A — Sprint 16B
opened qwen2 in `inference_support()` but no Qwen2.5 ever got
end-to-end-tested at runtime. Out of scope here; deserves its own
sprint with the small-model dim sweep.

## Files

- `src/main.rs` — `+11 / -1` (`inference_support()` widened, `ChatSession`
  construction routed through `ChatTemplate::detect()`)
- `README.md` — `+8 / -7` (status section reflects multi-arch)
- `results/v032_sprint17a_llama.md` — this file

The brief estimated this sprint at ~200 LOC (new chat template, bias
toggle, dim plumbing). Real cost: 12 LOC. Two earlier sprints (14
shipped templates that were never auto-selected; 16B opened the
preflight gate) had already done the hard part.
