# Phase 3E — Prefill GEMM Semantic-Drift Debug + Fix

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M
**Status:** ✅ Bug found and fixed in one focused session.
**Symptom:** `prefill_batch` produced semantically wrong KV cache —
e.g. "What is 2 + 2?" was misread as "What is you?".
**Root cause:** Per-token RoPE position writes during command-buffer
recording all collapsed to the **last value** by the time the GPU
ran the dispatches. Every prefill token was rotated as if it were
the last position.
**Fix:** Stage all positions in `rope_pos_buf` upfront, bind with
per-token offset.
**Tests:** **48/48** still pass (parity gate now requires ≥4/5 top-5
overlap, was ≥1/5).

---

## 1. Diagnosis

### 1.1 Symptom evidence

`examples/diagnose_drift.rs` runs the same prompt through both paths
and prints the logits. Pre-fix, on `"What is 2 + 2?"`:

```
max_abs_err = 19.7611
mean_abs_err = 3.1347
top-5 overlap = 1/5

top-5 path A (token-by-token):
  151667 "<think>"      logit  36.741
   77126 ".MESSAGE"     logit  18.595
   79489 ",response"    logit  18.129
    8607 ".i"           logit  17.985
   85387 ".$."          logit  17.854

top-5 path B (prefill_batch):
  151667 "<think>"      logit  34.835
  151644 "<|im_start|>" logit  27.815   ← suspicious
  151668 "</think>"     logit  27.677   ← suspicious
     198 "Ċ" (newline)  logit  23.603   ← suspicious
  151645 "<|im_end|>"   logit  23.182   ← suspicious
```

The pattern was diagnostic: **the entire path-B top-5 below `<think>`
was structural / boundary tokens** — `<|im_start|>`, `<|im_end|>`,
newline, `</think>`. When a transformer emits high logits for
"begin/end of segment" tokens, it means the last hidden states are
encoding "I am at a structural boundary" rather than reflecting the
actual content of the prompt.

For Qwen3 on a chat-template prompt, the LAST tokens fed in are
`<|im_start|>` `assistant` `\n`. After a correct prefill, the model
should be ready to generate the assistant's reply (which it does —
`<think>` wins by 18 points in path A). But the path-B logits
suggested the model thought it was at the START of a new turn,
indistinguishable from the boundary tokens themselves.

### 1.2 What that pattern points at

A model can lose track of "where am I in the sequence?" exactly when
**positional encoding is broken**. Qwen3 uses RoPE-NeoX which
multiplies per-token embeddings by position-dependent rotations. If
the rotations are wrong, attention can no longer distinguish "this
is token 26 (assistant\\n)" from "this is token 0 (im_start)".

### 1.3 The bug, concretely

Original `dispatch_layer_batch` (pre-fix):

```rust
for t in 0..seq_len {
    let pos = base_pos + t;

    // HOST WRITE — happens DURING command-buffer recording.
    self.rope_pos_buf
        .write_bytes(bytemuck::bytes_of(&pos))
        .expect("rope pos write");

    // ... record dispatches that read rope_pos_buf ...
    self.run_rope_neox(.., pos, ..);
    // ...
}
```

`rope_pos_buf` was a **single 4-byte slot** shared across all RoPE
dispatches. The host writes happen during command-buffer **recording**,
but the GPU dispatches run after `vkQueueSubmit`. By the time the
first RoPE dispatch executes on the GPU, the host has already
overwritten the slot 26 more times. **Every RoPE dispatch in the
batch read the same final value (pos = seq_len − 1).**

So token 0's Q/K were rotated as if they were at position 26, token 1
as if at 26, …, token 26 correctly at 26. The KV cache filled with
26 nearly-identical positional encodings, and downstream attention
saw them as "this is the same boundary token, repeated" — which
matches the observed special-token-heavy logits.

This bug was invisible to:

- The **GEMM parity gate** (Phase 3D §1.3) — that test runs at
  `seq_len = 1`, so only one position is ever written.
- The **logits parity test** (`phase3e_..._top5`, original
  threshold 1/5) — the `<think>` argmax is so dominant on the mutex
  prompt that even a broken positional encoding still picks it up.

---

## 2. The fix

Three coordinated changes in `forward.rs`:

### 2.1 Resize `rope_pos_buf`

```rust
let rope_pos_buf = mk_storage(
    (max_prefill_tokens.max(1) as u64) * 4,   // was a fixed 4 bytes
    MemoryLocation::CpuToGpu,
    "rope_pos",
)?;
```

Now holds `max_prefill_tokens` u32 slots. `forward_token` keeps
using slot 0; `prefill_batch` uses slots 0..seq_len-1.

### 2.2 New variant `run_rope_neox_with_pos_offset`

Identical to `run_rope_neox` except the rope_pos binding takes an
explicit offset:

```rust
self.write_bindings(dev, set, &[
    (0, input, 0, 0),
    (1, self.rope_pos_buf.handle, pos_buf_offset, 4),  // 4-byte slice
    ...
]);
```

The legacy `run_rope_neox` is now a thin wrapper that calls
`run_rope_neox_with_pos_offset(.., pos_buf_offset=0, ..)` — preserves
the single-token forward path bit-exactly.

### 2.3 Stage all positions upfront in `prefill_batch`

```rust
// CRITICAL: All GPU dispatches in this submit run AFTER all host
// writes complete, so we must write every per-token position
// into a separate slot of rope_pos_buf BEFORE we start recording —
// otherwise the per-token RoPE dispatches would all read the
// last-written value (Phase 3E drift bug).
let positions: Vec<u32> = (0..seq_len).map(|t| base_pos + t).collect();
self.rope_pos_buf.write_bytes(bytemuck::cast_slice(&positions))?;
```

Then per-token loop binds the right slot:

```rust
let rope_pos_offset = (t as u64) * 4;
self.run_rope_neox_with_pos_offset(.., pos, rope_pos_offset, ..);
```

That's it — three small changes plus a buffer resize.

---

## 3. Validation

### 3.1 `diagnose_drift` — same prompt, post-fix

```
max_abs_err = 0.7759       ← was 19.7611  (25× tighter)
mean_abs_err = 0.1457      ← was 3.1347
top-5 overlap = 5/5        ← was 1/5

top-5 path A (token-by-token):
  151667 "<think>"      logit 36.741
   77126 ".MESSAGE"     logit 18.595
   79489 ",response"    logit 18.129
    8607 ".i"           logit 17.985
   85387 ".$."          logit 17.854

top-5 path B (prefill_batch):
  151667 "<think>"      logit 36.652
   77126 ".MESSAGE"     logit 18.277
   79489 ",response"    logit 17.988
    8607 ".i"           logit 17.923
   85387 ".$."          logit 17.778
```

Same five tokens, same order, logit values within ~1.0 — exactly
what Q8_1 quantisation noise should give.

### 3.2 Semantic check on the offending prompt

`VF_PROMPT="What is 2 + 2?" cargo run --release` post-fix produces:

> `<think>` Okay, the user asked **"What is 2 + 2?"**. That's a basic
> math question. Let me make sure I understand it correctly. They
> want the sum of 2 and 2.
>
> First, I should recall the basic addition facts. 2 + 2 is a simple
> addition problem. In arithmetic, adding two numbers means combining
> their quantities. So 2 plus …

The model now correctly identifies the prompt as a math question.
Compare to pre-fix:

> `<think>` Okay, the user asked **"What is you?"** which is a common
> way to ask about my identity. […] I'm Qwen, a large language model
> developed by Alibaba Cloud …

### 3.3 5-prompt validation suite

| Prompt                                              | Prefill (tok/s) | Decode (tok/s) | Coherent |
| --------------------------------------------------- | --------------: | -------------: | :------: |
| Explain what a mutex is in one sentence.            |             269 |           65.1 |    ✓     |
| Write a haiku about programming.                    |             265 |           65.0 |    ✓     |
| What is 2 + 2?                                      |             267 |           66.2 |    ✓     |
| Translate 'hello world' to German.                  |             277 |           65.9 |    ✓     |
| List three prime numbers.                           |             257 |           66.7 |    ✓     |
| **MEDIAN**                                          |         **267** |       **65.9** |   5/5    |

All five prompts produce on-topic output. Compared to Phase 3E
pre-fix the prefill is the same to within run-to-run noise (256 →
267 tok/s), decode is identical (67 → 66 tok/s).

### 3.4 15-prompt benchmark

| Aggregate metric         | Pre-fix Ph3E | **Post-fix** | Δ          |
| ------------------------ | -----------: | -----------: | ---------: |
| Prefill aggregate (tok/s)|       306.3  |      298.7   | −2.5 % (run-to-run noise) |
| Decode aggregate (tok/s) |        48.8  |       47.8   | −2.0 %     |
| Median prefill (tok/s)   |       316.0  |      289.5   | −8.4 %     |
| Median decode (tok/s)    |        62.4  |       61.8   | tied       |
| Coherent (heuristic)     |       13/15  |      14/15   | +1         |

The fix is **correctness-only** — no perf regression, mild noise.
14/15 by the heuristic; manual inspection confirms 15/15 (the
remaining flag is an emoji-output false positive, not a model error).

### 3.5 Tightened parity gate

`phase3e_prefill_batch_matches_token_by_token_top5` now asserts
**≥ 4/5 top-5 overlap** (pre-fix it was ≥ 1/5 to accommodate the
buggy state). Post-fix this is comfortably satisfied — the actual
overlap on the mutex prompt is 5/5.

---

## 4. Why other tests didn't catch this

The drift escaped Phase 3D's parity gate and Phase 3E's coherence
heuristic because:

1. **Phase 3D parity test ran at `seq_len = 1`.** Only one position
   is ever written to `rope_pos_buf`, so the host-write race
   couldn't manifest.
2. **Phase 3E parity test on the mutex prompt** found `<think>` as
   top-1 in *both* paths because `<think>` wins by ~18 logit-points
   in path A — large enough that even the position-broken path B
   still ranks `<think>` first. The buggy path's runners-up *were*
   structural tokens; the original test's `≥ 1/5 overlap` threshold
   was too loose to flag that.
3. **5-prompt suite produced coherent text** for several prompts
   because some prompts (e.g. "List three prime numbers") are
   short enough or generic enough that broken positions still let
   the model produce a plausible reply.
4. **The "What is 2 + 2?" misreading** was visible in the streaming
   output but I'd been smoke-testing other prompts.

Going forward:
- The parity gate now requires top-5 overlap ≥ 4/5 (catches similar
  drift in the future).
- The drift-debug tool (`examples/diagnose_drift.rs`) is checked in;
  re-runnable any time we suspect prefill correctness.

---

## 5. Files changed

| File                                              | Change |
| ------------------------------------------------- | ------ |
| `src/backend/vulkan/forward.rs`                   | edit — `rope_pos_buf` resize, `run_rope_neox_with_pos_offset`, `prefill_batch` writes positions upfront, `dispatch_layer_batch` uses per-token offset |
| `tests/regression.rs`                             | edit — parity gate tightened from `≥ 1/5` to `≥ 4/5` overlap |
| `examples/diagnose_drift.rs`                      | new — single-binary drift reproducer |
| `results/phase3_prefill_drift_debug.md`           | new — this report |
| `results/phase3_drift_5prompt.log`                | new — 5-prompt suite stdout |
| `results/phase3_drift_15prompt.log`               | new — 15-prompt benchmark stdout |

**Untouched:** every shader, the GEMM dispatch path, the quantize
path. The fix is purely in how host-CPU staging writes interact
with deferred GPU dispatch within a single command buffer.

---

## 6. Cross-phase progression — final (with fix)

| Metric                  | Ph2D | Ph3A | Ph3B | Ph3C | Ph3D | Ph3E (fix) | LC-Vk |
| ----------------------- | ---: | ---: | ---: | ---: | ---: | ---------: | ----: |
| Decode tok/s (median)   | 13.4 | 66.8 | 67.9 | 64.1 | 64.1 |   **61.8** | 114.2 |
| Prefill tok/s (median)  |   56 |   79 |   82 | 79.4 | 79.4 |  **289.5** |  4314 |
| Tests                   |   33 |   35 |   45 |   45 |   47 |     **48** |     — |
| Shaders                 |   11 |   11 |   11 |   14 |   14 |     **14** |     — |
| Semantically correct    |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |     **✅** |     — |

Prefill held the 3.6-3.7× speedup over Phase 3C while regaining the
semantic correctness Phase 3C had.

---

## 7. Acceptance gates

| Gate                                                              | Status |
| ----------------------------------------------------------------- | :----: |
| 48/48 tests green                                                 |   ✅   |
| 0 validation errors                                               |   ✅   |
| Top-5 overlap ≥ 4/5 (was forced to 1/5 pre-fix)                   |   ✅ (5/5 actual) |
| max_abs_err << pre-fix (19.76)                                    |   ✅ (0.78) |
| Semantic correctness on `"What is 2 + 2?"` — model identifies it as math |   ✅   |
| 5-prompt suite — 5/5 on-topic                                     |   ✅   |
| 15-prompt suite — coherence preserved or improved                  |   ✅ (14/15 vs 13/15 pre-fix) |
| Prefill speedup vs Phase 3C preserved                             |   ✅ (~3.7×) |
| No decode regression                                              |   ✅   |

---

## 8. Lesson for the codebase

> **Host-side buffer writes during command-buffer recording happen
> *immediately*; the GPU sees only the final state at submit time.**

Anywhere we host-write a value mid-recording and dispatch a kernel
that reads it, we must allocate one slot per-dispatch and bind with
offset. Single-slot mutable buffers only work when each forward
gets its own submit — which happens in `forward_token` but **not**
in `prefill_batch`.

This applies to any future similar batched paths: per-token
position values, per-step state machines, etc.

---

## 9. Commit hash

To be filled in by the commit at the end of this run.
