# Prompt 16 — Alice Multi-Turn Context-Retention Test

**Date:** 2026-04-27
**Version:** v0.1.1 (no version bump — this is a test, not a feature)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5

---

## Headline

| Engine | Qwen3-8B | Llama-3.1-8B | DeepSeek-R1-Distill | Mistral-7B-v0.3 |
|---|---|---|---|---|
| **VulkanForge** | **3 / 3 PASS** | **3 / 3 PASS** | **3 / 3 PASS** | **3 / 3 PASS** |
| **llama.cpp Vulkan** | 3 / 3 PASS | 3 / 3 PASS | 3 / 3 PASS | 3 / 3 PASS |

Multi-turn KV-cache persistence works correctly on all four supported
models. No bug in the multi-turn path.

---

## 1 — Test design

Six-turn `ChatSession` exchange with **NO `reset()` between turns**.
Three of the six turns are "critical" — they ask the model to recall
a fact the user gave earlier. A pass means the response contains the
expected keyword(s).

```
turn 1  "My name is Alice."                          → soft (Alice)
turn 2  "What is 2 + 2?"                             → soft (4)
turn 3  "What is my name?"                           → CRITICAL (Alice)
turn 4  "I live in Berlin."                          → soft (Berlin)
turn 5  "Where do I live?"                           → CRITICAL (Berlin)
turn 6  "What is my name and where do I live?"       → CRITICAL (Alice + Berlin)
```

A failure on turn 3, 5, or 6 means the KV-cache is not persisting
across turns — either it's being silently reset, the position offset
is wrong, or the chat-template continuation is rendered with a
boundary mismatch. Bug class, not a model-quality issue.

Implementation:
- `examples/run_alice_test.rs` (250 LOC) — standalone driver per model.
- `tests/regression.rs::phase_prompt16_alice_context_retention_qwen3` —
  asserts 3 / 3 critical for Qwen3 in CI.

`think_filter` is **off** for this test — DeepSeek-R1's `<think>` block
can legitimately contain the recall keyword and we want to score the
full generated output.

---

## 2 — VulkanForge results

### 2.1 Per-model summary

| Model | Critical | Avg decode | Avg prefill | Final KV pos |
|---|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | **3 / 3** | 88.0 tok/s | 214 tok/s | 467 / 2048 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | **3 / 3** | 94.0 tok/s | 248 tok/s | 179 / 2048 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | **3 / 3** | 94.2 tok/s | 202 tok/s | 437 / 2048 |
| Mistral-7B-Instruct-v0.3.Q4_K_M | **3 / 3** | 100.2 tok/s | 185 tok/s | 200 / 2048 |

Final KV pos differs because each model's verbosity and
chat-template overhead are different (DeepSeek emits long `<think>`
blocks; Mistral is terse). All within the 2048 ctx budget.

### 2.2 Verbatim transcripts (critical turns only)

#### Qwen3-8B-Q4_K_M

```
T3 USER:  What is my name?
T3 ASST:  …Alice said, "My name is Alice." Then I responded with…  ✅
T5 USER:  Where do I live?
T5 ASST:  …Earlier, they said, "I live in Berlin." So the answer…  ✅
T6 USER:  What is my name and where do I live?
T6 ASST:  …They introduced themselves as Alice and mentioned living in Berlin…  ✅
```

#### Llama-3.1-8B (cleanest output, no `<think>`)

```
T3 USER:  What is my name?
T3 ASST:  Your name is Alice!  ✅
T5 USER:  Where do I live?
T5 ASST:  You live in Berlin, Germany!  ✅
T6 USER:  What is my name and where do I live?
T6 ASST:  Your name is Alice, and you live in Berlin, Germany!  ✅
```

#### DeepSeek-R1-Distill-Llama-8B

```
T3 USER:  What is my name?
T3 ASST:  …In our previous interaction, they introduced themselves as Alice.
          Now they're asking for their name…  ✅
T5 USER:  Where do I live?
T5 ASST:  …In the previous conversation, they mentioned living in Berlin,
          so I should refer back to that…  ✅
T6 USER:  What is my name and where do I live?
T6 ASST:  …From the conversation history, the user introduced themselves as
          Alice and mentioned living in Berlin…  ✅
```

#### Mistral-7B-Instruct-v0.3 (terse, no thinking block)

```
T3 USER:  What is my name?
T3 ASST:  Your name is Alice, as you mentioned earlier.  ✅
T5 USER:  Where do I live?
T5 ASST:  You live in Berlin, as you mentioned earlier.  ✅
T6 USER:  What is my name and where do I live?
T6 ASST:  Your name is Alice and you live in Berlin.  ✅
```

### 2.3 Soft-turn note

`Turn 2: 2+2` failed soft on Qwen3 and DeepSeek because their `<think>`
block didn't reach the literal "4" within the 40-token budget — the
reasoning was truncated mid-stream. Not a KV-cache bug; bumping
`max_tokens` would close it. Llama-3.1 and Mistral pass turn 2 at the
same budget because they don't have a thinking phase.

---

## 3 — llama.cpp Vulkan reference

### 3.1 Method note

`llama-cli -cnv` (interactive conversation mode) crashes when stdin is
piped (`std::thread::join` on shutdown after the input pipe closes).
Worked around by using `llama-completion` and rebuilding the full
ChatML / Llama-3 / DeepSeek / Mistral conversation history per turn —
each invocation is `previous_history + new_user_turn`. KV-cache
correctness for llama.cpp is the reference; the rebuild approach
just replays the same prompts and shows what the answer + tok/s look
like. Script: `/tmp/llamacpp_alice.sh`.

### 3.2 Per-model summary (llama.cpp Vulkan)

Per-turn perf reported by `common_perf_print`; values are medians
across the 6 turns.

| Model | Critical | Decode median | Prefill median |
|---|---|---:|---:|
| Qwen3-8B-Q4_K_M | **3 / 3** | 113.8 tok/s | 3541 tok/s |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | **3 / 3** | 126.2 tok/s | 3212 tok/s |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | **3 / 3** | 118.0 tok/s | 3465 tok/s |
| Mistral-7B-Instruct-v0.3.Q4_K_M | **3 / 3** | 126.2 tok/s | 3212 tok/s |

### 3.3 llama.cpp transcript spot-check (Mistral)

```
T3 USER:  What is my name?
T3 ASST:  Your name is Alice. (You mentioned it in our previous interaction!)  ✅
T5 USER:  Where do I live?
T5 ASST:   You mentioned earlier that you live in Berlin. Is there
          anything else you would like to know…  ✅
T6 USER:  What is my name and where do I live?
T6 ASST:   Your name is Alice and you mentioned earlier that you
          live in Berlin…  ✅
```

---

## 4 — Comparison: VulkanForge vs llama.cpp Vulkan

### 4.1 Correctness

Identical: 3 / 3 critical PASS on every model on both engines.

### 4.2 Throughput

| Model | VF decode | llama.cpp decode | Ratio | VF prefill | llama.cpp prefill | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 88.0 | 113.8 | 0.77× | 214 | 3541 | 0.06× |
| Llama-3.1-8B | 94.0 | 126.2 | 0.74× | 248 | 3212 | 0.08× |
| DeepSeek-R1 | 94.2 | 118.0 | 0.80× | 202 | 3465 | 0.06× |
| Mistral-7B-v0.3 | 100.2 | 126.2 | 0.79× | 185 | 3212 | 0.06× |

Decode is within ~75–80 % of llama.cpp Vulkan, consistent with the
single-turn 15-prompt suite. Prefill is the known Phase 5B target —
~6–8 % of llama.cpp.

Note: the two prefill numbers are not directly comparable because the
methodology differs (VulkanForge runs a real per-turn prefill with
KV-cache reuse; the llama.cpp script rebuilds the full prompt each
turn, so its prefill count and timing include all prior tokens). The
single-turn 15-prompt comparison from Phase 5A remains the
apples-to-apples prefill reference: VF 405 tok/s vs llama.cpp 4314
tok/s on Qwen3.

---

## 5 — KV-cache health check

Across all 6 turns the position counter monotonically increases as
expected (no resets, no reuse-of-same-slot). Sample (Qwen3):

```
turn 1: pos 0   → 84   (system+user 24 prompt + 60 gen)
turn 2: pos 84  → 142  (continuation +18 prompt + 40 gen)
turn 3: pos 142 → 207  (continuation +15 prompt + 50 gen)
turn 4: pos 207 → 282  (continuation +15 prompt + 60 gen)
turn 5: pos 282 → 347  (continuation +15 prompt + 50 gen)
turn 6: pos 347 → 467  (continuation +20 prompt + 100 gen)
```

Each subsequent prefill takes only the bytes of the new user turn
(15–20 tokens — `<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`)
plus the new user content, NOT the entire history. That's the
multi-turn invariant: prior turns sit in the KV cache and don't have
to be re-prefilled.

For attention, position 64+ crosses into the split-K decode path
(Phase-4C `flash_attn_split` + `flash_attn_reduce`); that path is
exercised on every turn from turn 2 onward and produces correct
recall, so split-K + KV-cache co-existence is also covered by this
test.

---

## 6 — Tests

```
unit (lib)         19   (no change)
correctness        25   (no change)
regression         22   (+1: phase_prompt16_alice_context_retention_qwen3)
total              66   ALL GREEN
```

`cargo test --release`  →  66 / 66 PASS.
`cargo clippy --release --tests --examples`  →  clean.

---

## 7 — Files touched

```
NEW   examples/run_alice_test.rs                Alice 6-turn driver
NEW   inference_test_prompts_16.json            15 single-turn + 1 multi-turn
NEW   results/prompt16_alice_test.md            this report

EDIT  tests/regression.rs                       +1 phase_prompt16_* test
```

No changes to library code; the multi-turn path was already in place
since Phase 3B and verified end-to-end here.

---

## 8 — Diagnostics path (if a future regression triggers FAIL)

If a critical turn ever fails:

1. Verify KV-cache position monotonically grows turn-to-turn. A jump
   back to 0 mid-test means `session.reset()` is being called
   accidentally; a non-monotonic offset means `current_pos` accounting
   is off.
2. Check `ChatTemplate::render_continuation` output for the model's
   template — the continuation must replay the *unconfirmed* EOS the
   model emitted at the end of the previous turn (e.g. `<|im_end|>\n`
   for ChatML, `<|eot_id|>` for Llama-3, `</s>` for Mistral).
3. Run with `print_stream=true` and inspect the model's actual answer
   text — sometimes the model recalls correctly but the keyword
   matcher is too strict (e.g. wrong case, German alias for the
   place name).
4. Compare against the same prompt sequence on llama.cpp via
   `/tmp/llamacpp_alice.sh` to see whether the recall question is
   even answerable for that specific model + prompt phrasing.

---

## 9 — Out of scope

- No performance optimisation in this prompt (Phase 5B work).
- No new chat templates beyond the four already supported.
- No external memory / RAG.

---

## 10 — Commit

Single squashed commit on `main`:

```
prompt-16: Alice context-retention test (multi-turn)
```

No push.
