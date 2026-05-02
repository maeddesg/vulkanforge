# Phase 3B — Multi-Turn + Streaming + Think-Filter + CLI

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M
**Status:**
  - ✅ Step 2 (Multi-Turn / ChatSession) — done
  - ✅ Step 3 (Streaming + Think-Filter) — done
  - ✅ Step 4 (Interactive CLI) — done
  - ⏸ **Step 1 (Prefill GEMM) — STOPPED at the prompt's explicit
       complexity gate.** Analysis captured below as Phase-3C input.
**Tests:** **45/45** pass (was 35 in Phase 3A; +10 in Phase 3B).
**Validation:** 5-prompt suite still coherent; decode median **67.9 tok/s**
(vs Phase 3A 66.8 — within noise, no regression).
**0 validation errors.**

---

## 1. Step 1 — Prefill GEMM: STOPPED, Phase-3C work

### 1.1 Why it stopped

The prompt's explicit gate (§Bekannte Fallstricke / §1):

> Falls der GEMM-Shader > 1 Stunde zum Integrieren braucht: STOP und
> berichten. Möglicherweise braucht er einen eigenen
> DescriptorSetLayout der nicht in die bestehende Registry passt.

The actual llama.cpp shaders at
`~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`:

| File                              | Lines |
| --------------------------------- | ----: |
| `mul_mm.comp`                     | 464   |
| `mul_mmq.comp`                    | 311   |
| `mul_mm_funcs.glsl`               | 598   |
| `mul_mmq_funcs.glsl`              | 454   |
| `mul_mmq_shmem_types.glsl`        | 84    |
| `mul_mat_split_k_reduce.comp`     | (1-pass split-K reducer) |
| **Total integration surface**     | **~1900 lines of GLSL** |

`mul_mmq.comp` is the production path llama.cpp uses on AMD; the FP
variant (`mul_mm.comp`) would require dequantising weights to FP and
inflating 4.68 GiB → ~21 GiB, exceeding the 16 GiB VRAM budget. So
`mul_mmq` is the only viable option, and it brings hard prerequisites:

| Requirement                                    | Status on this machine |
| ---------------------------------------------- | ---------------------- |
| `GL_EXT_integer_dot_product` shader extension  | ✅ supported           |
| `integerDotProduct8BitSignedAccelerated`       | ✅ true                |
| `integerDotProduct4x8BitPackedSignedAccelerated` | ✅ true              |
| **A `quantize_q8_1` shader for activations**   | ❌ does not exist yet — `mul_mmq` reads B as `block_q8_1_x4_packed128`, so input activations must be quantised to Q8_1 on the fly |
| 11 specialisation constants                    | new — current registry takes 1 |
| 13 push-constant fields (M, N, K, strides, batch_strides, k_split, ne0/2, broadcast2/3, num_batches, base_work_group_z) | requires a new push-constant struct + reflection round |
| Custom shared-memory block types (`block_a_cache`, `block_b_cache`) | new include path through `mul_mmq_shmem_types.glsl` |
| Optional `MUL_MAT_ID` / `MUL_MAT_ID_USE_SUBGROUPS` paths | not needed for Phase 3, but the `#ifdef` ladder must be navigated to compile a clean MoE-disabled variant |

### 1.2 What's actually missing — concrete

Net new code that Phase 3C will need to produce, listed so the next
prompt can be sized realistically:

1. New shader: `quantize_q8_1.comp` (FP32 input → `block_q8_1_x4_packed128`
   output). ~80 lines, novel layout, must be unit-tested vs an FP32×Q8_1×FP32
   round-trip.
2. New shader: `mul_mmq_q4_k_f32_f32.comp` and `mul_mmq_q6_k_f32_f32.comp`
   wrappers — each is `#include "mul_mmq.comp"` with the right `A_TYPE`,
   `A_TYPE_PACKED32`, and `QUANT_R_MMQ` defines.
3. Optional split-K reducer (`mul_mat_split_k_reduce.comp`) for large-K
   GEMMs; for Qwen3-8B the projections at K=4096 likely don't need it.
4. New push-constant struct with 13 fields, ~52 B. Must clear
   `spirv_reflect.rs` (the reflector currently handles nested structs —
   verified during Phase 2A — but the LM-head-style array+struct chains
   in `mul_mm_funcs.glsl` haven't been exercised yet).
5. New `ShaderId` entries: `MulMatQ4K`, `MulMatQ6K`, `QuantizeQ8_1`.
6. A new `Forward::run_gemm` path that:
   - allocates a transient Q8_1-quantised activations buffer (~K bytes per
     token block),
   - dispatches `quantize_q8_1` on the FP32 activations,
   - dispatches the per-projection GEMM,
   - threads the seq_len axis through as the M dimension.
7. `Forward::prefill_batch` — bundles the 7 per-layer GEMM dispatches
   per layer × 36 layers, with token-by-token attention in between
   (the prompt's "pragmatischer Ansatz" — full batch attention with
   causal mask is a Phase-4 item).
8. CPU reference for `Q4_K × Q8_1 → FP32` GEMM correctness tests
   (`< 1e-2` per the prompt's gate).

Realistic effort: **2-4 hours of focused integration + 1-2 hours of
debug/correctness tuning**, well over the prompt's 1-hour STOP
threshold. The hardware path is fully supported; only integration
remains. Re-opening this in Phase 3C is the natural next step.

### 1.3 What did *not* change in this phase

Prefill remains token-by-token through the same GEMV forward pass:

```
Phase 3A baseline:  79 tok/s  (mutex-prompt prefill)
Phase 3B same:      82 tok/s  (within run-to-run variance)
```

No "batched-command-buffer" workaround was introduced — that would
have been "selbst entscheiden / Workaround bauen" (REGEL 2). The
prompt explicitly opens the STOP path, and that's what this phase
takes.

---

## 2. Step 2 — ChatSession (Multi-Turn + KV Persistence)

`src/backend/vulkan/chat.rs` (new, ~250 LoC). Built on top of the
new lower-level `decode::generate_from_tokens` (which takes a
prefill-token slice + start position and **does not reset the KV
cache**). The Phase-2D `decode::generate(prompt)` is preserved as a
back-compat wrapper.

### 2.1 Public API

```rust
pub struct ChatSession {
    pub forward: Forward,
    pub history: Vec<ChatTurn>,
    pub system_prompt: String,
    pub current_pos: u32,
    pub turn_count: u32,
}

impl ChatSession {
    pub fn new(forward: Forward, system: impl Into<String>) -> Self;
    pub fn send_streaming<F: FnMut(&str)>(...) -> Result<TurnResult, ChatError>;
    pub fn send(...) -> Result<TurnResult, ChatError>;          // non-streaming convenience
    pub fn reset(&mut self);
    pub fn remaining_tokens(&self) -> u32;
    pub fn max_seq_len(&self) -> u32;
}
```

### 2.2 Turn-boundary mechanics

| Turn type     | Prefill prefix                                                    |
| ------------- | ----------------------------------------------------------------- |
| **First**     | `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n` |
| **Subsequent** | `<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n` (the leading `<|im_end|>` is the closer the model emitted as EOS in the previous turn but didn't commit to the KV cache — re-injecting it here gives the next turn the same conditioning the chat template implies) |

`reset()` clears the KV cache, `history`, `current_pos`, and
`turn_count`, returning the session to first-turn state.

### 2.3 Context overflow

A structured `ChatError::ContextOverflow { current_pos, needed,
max_seq_len }` rather than a panic:

```text
ChatError: context overflow: current_pos=200 + needed=110 > max_seq_len=64. Use /reset to clear history.
```

The `phase3b_chat_session_context_overflow_clean_error` regression
test asserts exactly this: tiny `max_seq_len=64`, normal-length
prompt → `Err(ContextOverflow)` not panic.

### 2.4 Regression test

`phase3b_chat_session_multi_turn_carries_and_resets` is the
"strongest signal" test:

1. Turn 1: `"My name is Alice. Please remember it."`
2. Turn 2: `"What is my name?"` → must contain `"alice"` (case-insensitive)
3. `reset()`
4. Turn 3: same question → must NOT contain `"alice"`

All three assertions hold under greedy decode.

---

## 3. Step 3 — Streaming Callback + Think-Filter

### 3.1 Streaming

`decode::generate_from_tokens` takes `on_token: &mut dyn FnMut(u32, &str)`
and calls it once per generated token (after the per-token forward,
before EOS check). Existing call sites that just want stdout streaming
keep using `GenerateConfig::print_stream = true`; programmatic
callers (CLI, future TUI / web frontend) wire their own callback.

### 3.2 Think-Filter — text-state-machine, not token-id match

The probe in `examples/probe_think_tokens.rs` confirmed:

```
encode("<think>")    → [13708, 766, 29]   ← ["<th","ink",">"]
encode("</think>")   → [522, 26865, 29]   ← ["</","think",">"]
```

**Both** the BPE form and the special-token form (ids 151667 / 151668,
discovered in Phase 2D's `dump_specials.rs`) decode to the same
literal text. So the safest filter operates on **decoded text**, with
a tiny state machine that buffers across token boundaries:

```rust
pub struct ThinkFilter {
    in_think: bool,
    pending: String,
}
impl ThinkFilter {
    pub fn push(&mut self, chunk: &str) -> String;  // visible portion
    pub fn flush(&mut self) -> String;              // end-of-stream
    pub fn strip_all(text: &str) -> String;         // whole-string convenience
}
```

State transitions (in `pending`):

```
Normal:
  if pending.find("<think>")    → emit prefix + drain through ">", → InThink
  elif suffix-could-be("<think>") → emit safe-prefix, hold tail
  else                            → emit pending, clear

InThink:
  if pending.find("</think>")   → drain through ">", → Normal
  elif suffix-could-be("</think>") → hold tail (drop prefix)
  else                              → drop pending entirely
```

The "suffix could be a prefix of the tag" check (`partial_tag_split`)
is the part that handles the
`["<th","ink",">"]` / `["</","think",">"]` BPE chunking correctly: when
a chunk ends with `"<th"`, the filter holds those 3 bytes back until
the next chunk arrives.

### 3.3 Test coverage (ThinkFilter)

7 lib unit tests cover the full state machine:

| Test                                              | Scenario                              |
| ------------------------------------------------- | ------------------------------------- |
| `think_filter_passthrough_when_no_tags`           | identity for tag-free text            |
| `think_filter_strips_full_block_in_one_chunk`     | `"Hello <think>x</think>World"` → `"Hello World"` |
| `think_filter_handles_token_split_boundaries`     | feeds chunks `["Hello ","<th","ink",">",...]` exactly |
| `think_filter_empty_block`                        | `<think></think>foo` → `foo`          |
| `think_filter_only_open_no_close_drops_tail`      | unclosed `<think>` consumes the rest  |
| `think_filter_partial_open_at_end_held_back`      | `"hi <th"` then `"ello"` → `"hi <thello"` (no false positive) |
| `partial_tag_split_basic`                         | the suffix-prefix helper              |

Plus one regression test (`phase3b_think_filter_strips_real_text`)
that runs `strip_all` against a Qwen3-shaped reasoning output and
asserts the visible answer survives.

---

## 4. Step 4 — Interactive CLI

`src/main.rs` rewritten as the chat REPL. The Phase-2D 5-prompt
validation suite moved verbatim to `examples/run_validation.rs`
(still runnable with `cargo run --release --example run_validation`).

### 4.1 Banner

```text
VulkanForge v0.1.0 — Phase 3B chat REPL
  Model:   /home/maeddes/models/Qwen3-8B-Q4_K_M.gguf
    4.68 GiB · 36 layers · hidden=4096 · heads=32 · kv_heads=8 · head_dim=128
    vocab=151936 · ctx_max=2048 · rope_freq_base=1000000
  Loaded in 0.6 s
  Pipeline cache: 180492 bytes loaded · 11 shaders ready
  think-filter: on · max_tokens/turn: 80
  Type /help for commands, /quit to exit.
```

### 4.2 Commands

| Command            | Effect                                                     |
| ------------------ | ---------------------------------------------------------- |
| `/quit` / `/q`     | exit cleanly (full teardown, no VRAM leak)                 |
| `/exit`            | alias for `/quit`                                           |
| `/reset`           | `ChatSession::reset()`                                      |
| `/stats`           | context usage, turn count, last-turn timing                 |
| `/think`           | toggle the think-filter                                     |
| `/help` / `/h`     | list the above                                              |
| any other text     | treated as a user message → `ChatSession::send_streaming`   |

Empty lines are skipped. End-of-input (Ctrl-D) exits as if `/quit`.

### 4.3 Env vars (kept for non-interactive / CI use)

| Var                     | Effect                                                |
| ----------------------- | ----------------------------------------------------- |
| `VF_MODEL_PATH`         | override GGUF path                                    |
| `VF_MAX_TOKENS`         | per-turn cap (default 400)                            |
| `VF_SYSTEM`             | system prompt                                         |
| `VF_NO_THINK_FILTER=1`  | start with the filter off                             |
| `VF_PROMPT="..."`       | run a single non-interactive turn (used by smoke runs) |

### 4.4 Streaming wiring

```text
ChatSession::send_streaming
  ↓ generate_from_tokens (per generated token)
    ↓ on_token(id, raw_text)
      ↓ ThinkFilter::push (if filter enabled)
        ↓ on_visible(visible_chunk)   ← writes to stdout, flush
```

Output appears interactively as the model generates, with the
think-filter stripping `<think>...</think>` blocks across token
boundaries.

---

## 5. Performance — Phase 3A baseline vs Phase 3B

5-prompt validation (`cargo run --release --example run_validation`,
median of 5 prompts × 200 tokens each, all coherent English):

| Metric                       | Phase 3A | Phase 3B | Δ        |
| ---------------------------- | -------: | -------: | -------: |
| Median **decode tok/s**      | 66.8     | **67.9** | +1.6 %   |
| Median **prefill tok/s**     | 79       | **82**   | +3.8 %   |
| Multi-Turn carries context   | n/a      | **✅**   | new       |
| `/reset` clears KV state     | n/a      | **✅**   | new       |
| Context-overflow clean error | n/a      | **✅**   | new       |
| Streaming (callback-based)   | rudimentary | **✅** | promoted from `print!`  |
| Think-Filter                 | n/a      | **✅**   | new (text state machine) |
| Interactive REPL             | n/a      | **✅**   | new       |

The decode/prefill numbers are within run-to-run noise of Phase 3A —
exactly as expected, since Steps 2-4 are orthogonal to the GEMV
hot path. Step 1 (Prefill GEMM) was the only piece that would have
moved prefill numbers, and that's deferred.

### 5.1 5-prompt suite (regression)

All 5 prompts produced the same on-topic, coherent answers as Phase
3A. Full transcript: `results/phase3b_validation_run.log`.

---

## 6. Test Summary

```
$ cargo test --release -- --test-threads=1

running 14 tests   (lib unit)              14 passed   ← +7 ThinkFilter
running 16 tests   (tests/correctness.rs)  16 passed   (unchanged)
running 15 tests   (tests/regression.rs)   15 passed   ← +3 Phase 3B
                                          ───────────
                                          45 passed   (was 35 in Phase 3A)
```

New tests added this phase:

- 6 × `ThinkFilter::*` lib unit tests (7th covers `partial_tag_split`)
- `phase3b_think_filter_strips_real_text` (text regression)
- `phase3b_chat_session_multi_turn_carries_and_resets` (Alice / reset)
- `phase3b_chat_session_context_overflow_clean_error` (structured error)

All 35 pre-existing tests still pass. 0 validation errors with
`VK_LAYER_KHRONOS_validation` enabled across the full run.

---

## 7. Files changed

| File                                          | Status |
| --------------------------------------------- | ------ |
| `src/backend/vulkan/decode.rs`                | extended — `GenerateConfig::think_filter`, new `generate_from_tokens` low-level driver, `ThinkFilter` + 7 unit tests, `GenerateResult::visible_text` |
| `src/backend/vulkan/chat.rs`                  | new — `ChatSession`, `ChatTurn`, `ChatError`, `TurnResult` |
| `src/backend/vulkan/mod.rs`                   | edit — `pub mod chat;` |
| `src/main.rs`                                 | rewrite — interactive chat REPL |
| `examples/run_validation.rs`                  | new — Phase-2D 5-prompt suite (lifted out of `main.rs`) |
| `examples/probe_think_tokens.rs`              | new — diagnostic for `<think>` tokenisation |
| `tests/regression.rs`                         | edit — +3 Phase-3B tests, fixed pre-existing GenerateConfig literal |
| `results/phase3_step_3b_features.md`          | new — this report |
| `results/phase3b_validation_run.log`          | new — 5-prompt suite stdout |

**Untouched:**
- `forward.rs`, `pipeline.rs`, `pipeline_registry.rs`, `kv_cache.rs`,
  `tokenizer.rs`, `loader.rs`, `gguf.rs`, `device.rs`, `q4k.rs` —
  Phase 3B is a Rust API + REPL feature delta; no shader, descriptor,
  or KV-layout changes.

---

## 8. Acceptance gates

| Gate                                                      | Status |
| --------------------------------------------------------- | :----: |
| 45/45 tests green (35 pre-existing + 10 new)              |   ✅   |
| 5/5 validation prompts coherent, no decode regression     |   ✅   |
| Multi-Turn carries context across turns                   |   ✅   |
| `/reset` clears KV state                                  |   ✅   |
| Context-overflow → structured error, not panic            |   ✅   |
| Streaming: per-token callback works                       |   ✅   |
| Think-Filter handles BPE-token-split boundaries           |   ✅   |
| Interactive CLI with `/reset /quit /stats /think /help`   |   ✅   |
| 0 validation errors                                       |   ✅   |
| **Prefill ≥ 1000 tok/s at pp=29**                        |   ⏸ deferred to Phase 3C (GEMM analysis above) |

---

## 9. Commit hash

To be filled in by the commit at the end of this run.
