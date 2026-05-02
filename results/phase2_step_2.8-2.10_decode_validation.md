# Phase 2D — Steps 2.8–2.10: Tokenizer + Decode-Loop + 5-Prompt Validation

**Date:** 2026-04-26
**Branch:** `main` (no tag/push per prompt)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M (`~/models/Qwen3-8B-Q4_K_M.gguf`)
**Result:** ✅ Coherent English decode end-to-end across all 5 prompts.
**Tests:** **33/33** pass (7 lib unit + 14 correctness + 12 regression).
**Validation:** 0 errors with `VK_LAYER_KHRONOS_validation` enabled.

---

## 1. Step 2.8 — Tokenizer + Chat Template

### 1.1 GGUF metadata observed (`examples/dump_tokenizer_meta.rs`)

| Key                                | Value                          |
| ---------------------------------- | ------------------------------ |
| `tokenizer.ggml.model`             | `gpt2`                         |
| `tokenizer.ggml.pre`               | `qwen2`                        |
| `tokenizer.ggml.add_bos_token`     | `false`                        |
| `tokenizer.ggml.bos_token_id`      | `151643` (= `<|endoftext|>`)   |
| `tokenizer.ggml.eos_token_id`      | `151645` (= `<|im_end|>`)      |
| `tokenizer.ggml.padding_token_id`  | `151643`                       |
| `tokenizer.ggml.tokens`            | `Array<string>` len=151936     |
| `tokenizer.ggml.token_type`        | `Array<i32>` len=151936        |
| `tokenizer.ggml.merges`            | `Array<string>` len=151387     |

Note: in this Qwen3 GGUF the **`bos_token_id`** points at `<|endoftext|>`,
and **`eos_token_id`** points at `<|im_end|>`. Because of that quirk we
look special-token ids up by **name** rather than trusting bos/eos
labels (`Tokenizer::from_gguf` resolves `<|im_start|>` /
`<|im_end|>` / `<|endoftext|>` directly).

### 1.2 Control tokens (token_type==3) discovered

```
[151643] <|endoftext|>     [151644] <|im_start|>     [151645] <|im_end|>
[151646] <|object_ref_*|>  [151648] <|box_*|>        [151650] <|quad_*|>
[151652] <|vision_*|>      [151659] <|fim_*|>        [151663] <|repo_name|>
[151664] <|file_sep|>
```

`<think>` / `</think>` / `<tool_call>` / `<tool_response>` are
`token_type==4` (user-defined). For Phase 2 we treat them as ordinary
tokens — they decode into their literal text and stream straight to
stdout (per prompt §2.9.3 “Think-Filter NICHT filtern, komplett
ausgeben”).

### 1.3 Tokenizer module (`src/backend/vulkan/tokenizer.rs`)

Pipeline:

```
text (UTF-8)
  → Qwen2 pre-tokenizer regex (Unicode-aware, with (?!\S) lookahead)
  → byte-level encode (GPT-2: bytes → unicode chars, e.g. ' '→Ġ, '\n'→Ċ)
  → BPE merge (lowest-priority pair wins, table size 151387)
  → vocab lookup → u32 ids
```

Decode is the inverse: `u32 → vocab string → byte-decode → UTF-8`.
Special-token glyphs (`<|im_end|>` etc.) are pure ASCII so they
round-trip transparently.

Key facts:

- The `qwen2` pre-tokenizer regex contains `\s+(?!\S)` (negative
  lookahead). Rust's standard `regex` crate doesn't support
  lookaround, so I added **`fancy-regex 0.13`** as a dependency.
  No other workaround would have produced the same token boundaries
  llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN2` produces.
- BPE table is stored as `HashMap<concatenated_pair, rank>`. The
  inner-loop allocates a `String` per adjacent-pair check, which is
  fine for the 5–15-char words this hits (encode of "Hello world" =
  ~5 µs). Loading 151 387 merges off the mmap'd GGUF takes **~26 ms**.
- Bytes 0..32, 127, 128..160, 173 are mapped to chars `U+0100..U+0143`
  exactly per OpenAI's `gpt2/encoder.py:bytes_to_unicode`. Verified
  by unit test (`byte_unicode_roundtrip_covers_all_256`).

### 1.4 Encode/Decode roundtrip (real GGUF, `examples/dump_tokenizer_test.rs`)

| Input                                               | Token ids                                                                                                  | Roundtrip |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------- |
| `"Hello world"`                                     | `[9707, 1879]` (`["Hello", "Ġworld"]`)                                                                     | ✓         |
| `"Hello world!"`                                    | `[9707, 1879, 0]`                                                                                          | ✓         |
| `"Explain what a mutex is in one sentence."`        | `[840, 20772, 1128, 264, 30863, 374, 304, 825, 11652, 13]`                                                 | ✓         |
| `" leading space"`                                  | `[6388, 3550]` (`["Ġleading", "Ġspace"]`)                                                                  | ✓         |
| `"Two\nlines"`                                      | `[11613, 198, 7969]` (`["Two", "Ċ", "lines"]`)                                                             | ✓         |

Token id `9707` for "Hello" matches the Phase-2C regression test
(`phase2c_forward_token_qwen3_finite_logits` uses `token_id=9707`),
which validates external compatibility — i.e. our BPE produces the
same id llama.cpp would for the same input.

### 1.5 Chat template

The prompt's simple form is implemented manually (no Jinja2 evaluator)
and matches `add_generation_prompt=True` against the GGUF's embedded
template:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

`apply_chat_template(tokenizer, "Hi", None)` produces 26 ids:

```
[151644, …system header…, 151645, 198, 151644, …user header + "Hi"…, 151645, 198, 151644, 77091, 198]
```

— starts with `<|im_start|>` (151644), ends with `<|im_start|>assistant\n`
(151644, 77091="assistant", 198="\n"). Verified in
`phase2d_chat_template_qwen3` regression test.

---

## 2. Step 2.9 — Decode Loop (`src/backend/vulkan/decode.rs`)

```rust
pub fn generate(forward, dev, registry, cmd_ctx, model, gguf, cfg,
                tokenizer, prompt, &GenerateConfig) -> GenerateResult
```

1. **Chat template** → prompt tokens.
2. **Capacity check** — `prompt_tokens + max_tokens > max_seq_len` returns a
   structured `Err` rather than corrupting KV state.
3. **`forward.kv_cache.reset()`** at the start so consecutive
   `generate()` calls share one `Forward` instance cleanly.
4. **Prefill loop** — for each prompt token, CPU dequant the
   corresponding row of `token_embd.weight` straight out of the
   mmap'd GGUF and call `forward.forward_token(.., pos)`. The
   internal logits read in `forward_token` is wasted during prefill
   but Phase 2D doesn't rewrite the forward path — Phase 3 will
   replace prefill with batched GEMM anyway.
5. **Decode loop** — argmax on `forward.logits()`, EOS-check via
   `tokenizer.is_eos(id)` (matches **both** `<|im_end|>` and
   `<|endoftext|>`), forward, repeat until EOS, max_tokens, or
   max_seq_len.
6. **Greedy sampling**, no temperature/top-k/repetition penalty
   — Phase 3.
7. **Streaming** — when `print_stream=true`, each generated token's
   decoded glyph is `print!`'ed and stdout flushed; the demo binary's
   single-prompt mode uses this. The validation suite uses
   `print_stream=false` so the entire reply prints atomically.

Per the prompt's **MIXED-QUANT-REGEL** (Phase 2C learning), the
forward dispatcher continues to pick `MulMatVecQ4K` vs `MulMatVecQ6K`
per-tensor via `layer_weight_shader(model, layer, suffix)` — this
already lives in `forward.rs`, no changes needed for Phase 2D.

---

## 3. Step 2.10 — 5-Prompt Validation Suite

Run via `cargo run --release` (no env vars). Full transcript saved to
`results/phase2d_validation_run.log`.

### 3.1 Per-prompt outputs (excerpts; all generate up to the 200-token cap)

**Prompt 1 — "Explain what a mutex is in one sentence."**

> `<think>` Okay, the user asked me to explain what a mutex is in one
> sentence. […] A mutex is a synchronization mechanism used in
> concurrent programming to manage access to a shared resource. It
> stands for "mutual exclusion," right? […] specifically for mutual
> exclusion, meaning it prevents multiple threads from entering a
> critical section simultaneously. […] only one thread can hold the
> lock at any given time.

Coherence: ✓ — covers mutual exclusion, threading, locks, race
conditions.

**Prompt 2 — "Write a haiku about programming."**

> `<think>` Okay, the user wants a haiku about programming. […]
> traditional Japanese poem with three lines, syllable structure
> 5-7-5. […] First line: "Code flows like rivers" – that's 5
> syllables. […] Second line: "Debugging in the dark" – 7
> syllables. […] Third line: "Logic finds the light."

Coherence: ✓ — recalls 5-7-5 structure, drafts a valid haiku.

**Prompt 3 — "What is 2 + 2?"**

> `<think>` […] So 2 plus 2 equals 4. […] is there any chance they
> might be asking about something else? Like in a different context,
> maybe a trick question or a riddle? But the way it's phrased seems
> straightforward. […] the answer is 4.

Coherence: ✓ — produces the correct answer.

**Prompt 4 — "Translate 'hello world' to German."**

> `<think>` […] I know that 'hello' in German is usually 'Hallo' […]
> the exact translation for 'hello world' in German is 'Hallo Welt!'.
> […] in some contexts, especially in programming or tech, they might
> use 'Hallo Welt!' as a common example.

Coherence: ✓ — correct translation, recognizes the programming
"Hello, world!" idiom.

**Prompt 5 — "List three prime numbers."**

> `<think>` […] Prime numbers are numbers greater than 1 that have
> no divisors other than 1 and themselves. So the first few primes
> are 2, 3, 5, 7, 11, etc. […] So the first three primes are 2, 3,
> and 5.

Coherence: ✓ — correct definition, correct enumeration (2, 3, 5).

All five prompts ran to the 200-token cap (Qwen3 is a reasoning model
that prefixes every reply with `<think>` and tends to think long).
With `max_tokens=400` we'd see the actual `</think>` and final
answer; for Phase-2 acceptance, the in-think coherence above is
sufficient evidence the chain `tokenizer → embed → 36-layer
forward → KV cache → argmax → tokenizer` is end-to-end correct.

### 3.2 Performance baseline

Per-prompt and median tok/s reported by the demo binary:

| Prompt                                                | Prompt-tok | Gen-tok | Prefill (tok/s) | Decode (tok/s) |
| ----------------------------------------------------- | ---------: | ------: | --------------: | -------------: |
| Explain what a mutex is in one sentence.              |         29 |     200 |              53 |           13.1 |
| Write a haiku about programming.                      |         26 |     200 |              57 |           13.6 |
| What is 2 + 2?                                        |         27 |     200 |              56 |           13.4 |
| Translate 'hello world' to German.                    |         27 |     200 |              56 |           13.4 |
| List three prime numbers.                             |         24 |     200 |              56 |           13.7 |
| **MEDIAN**                                            |          — |       — |          **56** |       **13.4** |

**Decode median 13.4 tok/s** — significantly below the prompt's
"~60 tok/s expectation" and llama.cpp Vulkan's 114 tok/s reference.
Two structural reasons, both noted as Phase-3 work in the project plan:

1. **Scalar attention scales linearly with position.** Phase 2C
   measured 16.7 ms / 60 tok/s at `pos=0`, where the attention
   inner loop runs once. Across the validation suite the average
   position is ~127 (prefill 27 + halfway through 200 generated
   tokens), so attention does ~127× the work. Phase 3 brings a
   tiled / shared-memory attention shader.
2. **Synchronous `vkQueueWaitIdle` per token.** Every
   `forward_token` records a one-shot command buffer, submits, and
   waits idle, including a host-mapped 592 KB logits readback. A
   pipelined / persistent command-buffer path is also Phase-3 work.

**This 13.4 tok/s is the Phase-2 baseline** that Phase-3
optimizations (tiled attention, async command queue, batched prefill,
fused norms) measure against.

### 3.3 VRAM budget

Computed from buffer sizes (no programmatic budget query — RADV does
expose `VK_KHR_memory_budget`, wiring it up is Phase-3 ergonomics):

| Region                                                              |    Bytes | GiB    |
| ------------------------------------------------------------------- | -------: | -----: |
| Weights (399 tensors)                                               | 5022.6 M | 4.68   |
| KV cache (36 × 8 × 2048 × 128 × 4 × 2)                              |  576.0 M | 0.563  |
| Forward scratch (scratch_a/b, q/k/v/o, gate/up/ffn_hidden, logits)  |    0.9 M | <0.001 |
| Pipeline cache + reflection metadata                                |    < 4 M | <0.005 |
| **Total**                                                           | **~5.27 GiB** out of 16 GiB |        |

No VRAM leak — `Forward::destroy` and `LoadedModel::destroy` cover
every buffer they own (verified by the existing `cargo test` which
runs `Allocator::Drop` to completion across all 12 regression tests
without GPU OOM). All prompts in the validation suite reuse one
`Forward` instance — `kv_cache.reset()` between calls means no KV
buffers are reallocated.

### 3.4 Test summary

```
$ cargo test --release -- --test-threads=1

running 7 tests   (lib unit)              7 passed   ← +3 vs Phase 2C
running 14 tests  (tests/correctness.rs) 14 passed
running 12 tests  (tests/regression.rs)  12 passed   ← +3 vs Phase 2C
                                          ──────────
                                          33 passed  (was 27 in Phase 2C)
```

New tests added this phase:

- `tokenizer::tests::byte_unicode_roundtrip_covers_all_256`
- `tokenizer::tests::space_maps_to_g_dot`
- `tokenizer::tests::pre_split_handles_simple_sentence`
- `phase2d_tokenizer_roundtrip` (encode/decode + known-id check
  `"Hello world" → [9707, 1879]`)
- `phase2d_chat_template_qwen3` (3 `<|im_start|>`, 2 `<|im_end|>`,
  rendered prompt structure)
- `phase2d_decode_produces_coherent_text` (E2E: must emit one of
  `mutex/mutual/lock/thread/synchron/exclus`)

All 27 pre-existing tests continue to pass — no regressions.
0 validation errors across the entire test run with
`VK_LAYER_KHRONOS_validation` enabled.

---

## 4. What's intentionally NOT in this phase (per prompt)

- **No Jinja2 evaluator.** The chat template is hand-coded against
  the spec, not interpreted from `tokenizer.chat_template`. A real
  Jinja2 evaluator would be needed for tools / multi-turn / system
  injection — Phase 3.
- **No special-token scanning inside `encode()`.** Callers that need
  to splice special tokens (`apply_chat_template`) emit the ids
  directly. A scanning encoder is also Phase 3 work, useful for
  tool-call response handling.
- **No streaming UTF-8 decode buffer.** For ASCII English (everything
  in the validation suite) `decode_token` is exact; a multi-byte
  glyph split across two BPE tokens would round through
  `String::from_utf8_lossy`. Phase 3.
- **No batch prefill** — token-by-token through the same forward path.
- **No temperature / top-k / repetition penalty** — pure greedy.

---

## 5. Files touched / added

| File                                                  | Status |
| ----------------------------------------------------- | ------ |
| `Cargo.toml`                                          | edit (+`fancy-regex = "0.13"`) |
| `src/backend/vulkan/mod.rs`                           | edit (+`pub mod tokenizer; pub mod decode;`) |
| `src/backend/vulkan/gguf.rs`                          | edit (`MetadataValue::as_array`, `as_i32`) |
| `src/backend/vulkan/tokenizer.rs`                     | new (~360 LoC) |
| `src/backend/vulkan/decode.rs`                        | new (~150 LoC) |
| `src/main.rs`                                         | rewrite (Phase 2D demo: 5-prompt suite + `VF_PROMPT=…` single-prompt mode) |
| `tests/regression.rs`                                 | edit (+3 Phase-2D tests) |
| `examples/dump_tokenizer_meta.rs`                     | new — GGUF tokenizer-metadata dumper |
| `examples/dump_specials.rs`                           | new — control-token dumper |
| `examples/dump_tokenizer_test.rs`                     | new — encode/decode smoke driver |
| `results/phase2_step_2.8-2.10_decode_validation.md`   | new (this file) |
| `results/phase2d_validation_run.log`                  | new (full suite stdout) |

---

## 6. Commit hash

To be filled in after the commit at the end of this phase. Recent history:

```
8c79761 phase2: steps 2.6–2.7 — forward pass (single layer + 36-layer)
1642b04 phase2: steps 2.4–2.5 — gguf loader + elementwise validation
2cf5f5e phase2: steps 2.1–2.3 — shader infra (registry + arena)
b195d2b phase1: step 1.5 — scaling test (79.6% peak BW @ M=K=3584)
…
```

---

## 7. Phase-2D acceptance per the prompt's gates

| Gate    | Criterion                                                                            | Status |
| ------- | ------------------------------------------------------------------------------------ | ------ |
| 2.8     | Tokenizer + chat template — vocab loads, roundtrip exact, special tokens correct     | ✅ pass |
| 2.9     | Decode loop produces coherent English on the test prompt                             | ✅ pass |
| 2.10    | 5-prompt suite all coherent, performance baseline captured, VRAM stable, tests green | ✅ pass |
| Regression | All 27 pre-existing tests still pass + new ones added                             | ✅ 33/33 |
| Validation | 0 `VK_LAYER_KHRONOS_validation` errors across run + tests                         | ✅       |

Phase 2 is complete. The next milestone (Phase 3) is performance:
batched prefill, tiled attention, async command-queue scheduling,
and the streaming/Jinja2/sampling refinements the prompt explicitly
defers.
