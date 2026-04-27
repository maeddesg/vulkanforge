# Phase 5B.2 — Batched-Q Attention Integration in `prefill_batch`

**Date:** 2026-04-27
**Version:** v0.1.1 (no version bump — integration step)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Builds on:** Phase 5B.1 — `flash_attn_batch.comp` (8 / 8 isolated parity)

---

## 1 — Bestandsaufnahme: `prefill_batch` structure

Read first, before changing anything. Findings:

```
fn prefill_batch:
  for layer in 0..n_layers:
    dispatch_layer_batch(layer, seq_len, base_pos):
      // Batched (per-layer, all M tokens at once):
      attn_norm_batch    [batch_residual → batch_norm]
      quantize_q8_1_batch
      GEMM_q  [M × (q_dim)]   = Q for all tokens
      GEMM_k  [M × kv_dim]
      GEMM_v  [M × kv_dim]

      // Per-token sub-loop  ← THE BOTTLENECK
      for t in 0..seq_len:
        copy batch_q[t]  → q_buf
        copy batch_k[t]  → k_buf
        copy batch_v[t]  → v_buf
        Q-norm / K-norm  (Qwen3 only)
        RoPE on q_buf, k_buf
        write k_buf, v_buf into KV-cache slot (layer, base_pos+t)
        run_scalar_attn(layer, pos)        ← M attention dispatches
        copy attn_out → batch_attn_out[t]

      // Batched (per-layer, all M tokens at once):
      GEMM_o, residual1, ffn_norm, GEMM_gate, GEMM_up, silu, mul,
      GEMM_down, residual2
```

Buffer-layout findings (matched against Phase 5B.1's
`flash_attn_batch` requirements):

| Buffer | Layout | Match? |
|---|---|---|
| `batch_q` (Q-GEMM out) | `[seq_len, n_heads, head_dim]` row-major | ✅ exact |
| KV cache layer slice | `[max_seq, n_kv_heads, head_dim]` | ✅ exact |
| `batch_attn_out` | `[seq_len, n_heads, head_dim]` row-major | ✅ exact |

**No reshape needed.** The shader can read `batch_q` directly and
write into `batch_attn_out` directly. The KV cache is bound exactly
the way `run_scalar_attn` already does it (layer offset + size).

---

## 2 — Integration plan

Two separate paths kept side-by-side, switched on `batch_attn_enabled`
(env var `VULKANFORGE_BATCH_ATTN`, default ON):

```
for t in 0..seq_len:
    copy batch_{q,k,v}[t] → {q,k,v}_buf       ← keep
    Q/K-norm                                   ← keep
    RoPE on q_buf, k_buf                       ← keep
    KV-cache write at (layer, base_pos+t)      ← keep

    if batch_attn:
        copy q_buf → batch_q[t]                ← NEW: stash post-RoPE Q
    else:
        run_scalar_attn(layer, pos)            ← legacy
        copy attn_out → batch_attn_out[t]      ← legacy

# AFTER the loop, when batch_attn:
if batch_attn:
    flash_attn_batch(
        q_buf=batch_q,
        k_buf=KV-cache[layer], v_buf=KV-cache[layer],
        o_buf=batch_attn_out,
        m=seq_len, q_start=base_pos, n_kv=base_pos+seq_len,
    )
```

The per-token loop still does the per-token RoPE / KV-cache write
(reusing the existing position-offset rope dispatch and the
single-token K/V copy). The only addition inside the loop is one
buffer copy `q_buf → batch_q[t]` (~1 KB, negligible). The single
batched attention call replaces M scalar/flash-attn dispatches.

---

## 3 — Buffer management

Q stash: the post-RoPE Q for token `t` is in `q_buf` after the
existing `run_rope_neox_with_pos_offset` call. We `cmd_copy_buffer`
it back to `batch_q[t]` (offset `t * q_bytes`, length `q_bytes`).
That overwrites the original GEMM-output Q for that row, which was
already consumed at the start of the iteration — safe.

KV cache: written by the existing per-token `cmd_copy_buffer` from
`{k,v}_buf` to `kv_cache.{k,v}_buffer` at `pos_offset_bytes(layer,
base_pos+t)`. Unchanged.

Output: `flash_attn_batch` writes its `[m, n_heads, head_dim]`
output directly into `batch_attn_out` at offset 0 — same buffer the
old loop's per-token `cmd_copy_buffer` was filling. The downstream
O-GEMM reads `batch_attn_out` unchanged.

No new buffer allocations. No VRAM increase.

---

## 4 — Files touched

```
EDIT  src/backend/vulkan/forward.rs                  — see §4.1, §4.2
EDIT  tests/regression.rs                            — +3 phase5b2_* tests
NEW   .cargo/config.toml                             — RUST_TEST_THREADS=4
NEW   results/phase5b_step_2_integration.md          — this report
```

No shader changes (5B.1 shipped that). No new push-constant types.
No KV-cache layout changes.

### 4.1 `forward.rs` — additions

```
+ struct field: batch_attn_enabled: bool
+ env var: VULKANFORGE_BATCH_ATTN  (default ON, "0"/"false" → OFF)
+ getter / setter: batch_attn_enabled() / set_batch_attn_enabled(bool)
+ method:    fn run_flash_attn_batch(&mut self, dev, registry, cmd,
              layer, q_buf, o_buf, m, q_start, n_kv)
+ FlashAttnBatchPushConstants imported from pipeline.rs
```

### 4.2 `forward.rs` — `dispatch_layer_batch` modification

Per-token loop body restructured into two blocks:

- Block A (always): copy + Q/K-norm + RoPE + KV-cache write +
  optional `q_buf → batch_q[t]` stash when batch_attn is on.
- Block B (only when `!batch_attn`): the legacy `run_scalar_attn` +
  `attn_out → batch_attn_out[t]` copy.

After the loop: one `flash_attn_batch` dispatch + one barrier when
batch_attn is on.

`.cargo/config.toml` sets `RUST_TEST_THREADS = "4"` — `tests/
regression.rs` now has 25 tests, every one of which loads ~5 GiB of
Qwen3 weights, and the default `num_cpus`-many parallel test threads
overcommit the 16 GiB VRAM budget on the RX 9070 XT (manifests as
`VK_ERROR_DEVICE_LOST` mid-run). 4 threads keep peak VRAM at ~20 GiB
worst-case and the regression suite still completes in <60 s.

---

## 5 — Correctness tests

Three new `phase5b2_*` regression tests, all on Qwen3-8B (the model
with the strictest GQA / Q-K-norm / NeoX-RoPE combination — if any
backend bug breaks the integration, this is the model that surfaces
it).

### 5.1 Parity gate (per-token vs batched, same prompt)

`phase5b2_batch_attn_parity_qwen3_short`
- Prompt: `"Explain what a mutex is in one sentence."` (≤64 tokens)
- Build two `Forward` instances, one with `batch_attn_enabled=false`,
  one with `=true`. Run `prefill_batch`. Compare logits.
- **Result:** argmax identical, top-5 overlap **5/5**.

`phase5b2_batch_attn_parity_qwen3_two_tiles`
- Prompt: 64-token mutex/semaphore explanation (capped to 64,
  exercises **two TILE iterations** for the last query in the
  batched shader's per-query causal-triangle loop).
- Same parity gate as above.
- **Result:** argmax identical, top-5 overlap **5/5**.

### 5.2 Multi-turn KV-cache survival

`phase5b2_decode_after_batched_prefill_qwen3`
- Two-turn `ChatSession` with batch_attn=ON throughout.
- Turn 1: `"I live in Berlin."`  (batched prefill writes KV positions 0..N-1)
- Turn 2: `"Where do I live?"`  (continuation prefill at q_start=N,
  followed by greedy decode that reads the full prior cache)
- Asserts the decode response contains `"Berlin"` (lower-case).
- **Result:** ✅ — model recalled "Berlin", same outcome as the
  Phase-5C Alice test except the Qwen3 prefill went through the
  new path on both turns.

### 5.3 Existing parity tests carried forward

The Phase-3E `phase3e_prefill_batch_matches_token_by_token_top5`
already compares `prefill_batch` against `forward_token` token-by-
token. Since Phase 5B.2 is now the default for `prefill_batch`,
this test transitively covers the new path against the per-token
fallback as well. **Result: still passing** (top-5 overlap ≥ 4/5,
argmax identical).

`phase_prompt16_alice_context_retention_qwen3` (3 / 3 critical)
also runs end-to-end through the new prefill path and passes.

### 5.4 Test count

```
unit (lib)         19   (no change)
correctness        33   (no change — Phase 5B.1's 8 tests already in place)
regression         25   (+3 phase5b2_*)
doctests            0
TOTAL              77   ALL GREEN

cargo clippy --release --tests --examples  →  clean
```

---

## 6 — Quick performance check (5 prompts, Qwen3-8B)

`run_15prompt_bench` truncated to the first 5 prompts. Greedy decode
with `think_filter=false`. Two runs, only the env var differs.

| # | Prompt | pp tok | gen tok | Prefill OFF | Prefill ON | Speedup |
|---|---|---:|---:|---:|---:|---:|
| 1 | Greeting (Hallo) | 20 | 64 | 261.6 | 297.6 | +14 % |
| 2 | Simple Sequence (count to 10) | 31 | 64 | 337.5 | 425.4 | +26 % |
| 3 | Prime Check (Python) | 31 | 256 | 337.6 | 418.5 | +24 % |
| 4 | LRU Cache (C++) | 47 | 512 | 408.8 | 547.1 | +34 % |
| 5 | REST API (Go) | 62 | 1024 | 454.7 | 652.0 | **+43 %** |
| | **MEDIAN** | | | **337.6** | **425.4** | **+26 %** |
| | **AGGREGATE** | 191 | 1920 | **373.4** | **483.3** | **+29 %** |

Coherence: 5 / 5 in both runs.
Decode: identical (~84 tok/s) in both runs — Phase 5B.2 only
changes the prefill attention path, not decode.

The speedup scales with prompt length (saved attention dispatches
× per-dispatch overhead grows with M):

```
pp=20  → +14 %
pp=31  → +25 %  (median of two)
pp=47  → +34 %
pp=62  → +43 %
```

Phase 5B.1's standalone shader-only timing (M=100 → 0.52 ms vs
~300 ms in the per-token loop) overstated the end-to-end win
because the GEMM / RoPE / RMSNorm dispatches are still per-layer
(not per-token, but they each cost ~0.5 ms × 36 layers). Net
prefill speedup at pp=62 is +43 %, with most of the residual gap
to llama.cpp Vulkan (4314 tok/s) sitting in the still-per-token
RoPE / KV-cache write loop and the GEMM pipeline. Phase 5B.3 is
where the full 15-prompt suite + 4-system comparison lands.

---

## 7 — Console summary

```
═══ Phase 5B.2 — Batch-Attention Integration ═══
Parity:       argmax identical, top-5 overlap 5/5 (short + two_tiles)
Multi-turn:   decode after batched prefill recalls "Berlin" on Qwen3
Prefill:      ALT 337.6 tok/s → NEU 425.4 tok/s (+26 % median, +43 % at pp=62)
Coherence:    5/5
Tests:        77/77 green  (+3 phase5b2_*)
Clippy:       clean
Commit:       (appended after `git commit`)
```

---

## 8 — What's still per-token (next-step targets)

The integration touches only the attention dispatch. The remaining
per-token work inside `dispatch_layer_batch`'s loop is:

```
copy batch_{q,k,v}[t] → {q,k,v}_buf      (3× cmd_copy_buffer per t)
Q/K-norm (Qwen-only)                      (2× rms_norm dispatches per t)
RoPE on q_buf, k_buf                      (2× rope dispatches per t)
KV-cache write                            (2× cmd_copy_buffer per t)
copy q_buf → batch_q[t]                   (1× cmd_copy_buffer per t, NEW)
```

For Qwen3 at pp=62 across 36 layers, that's 62 × (3 + 2 + 2 + 2 + 1)
× 36 = ~22 000 sub-dispatches per layer. The next phase (5B.3 or
5B-follow-up) can fold these into per-layer batched calls:

- RoPE: a batched variant that processes `[seq_len, n_heads,
  head_dim]` at once, reading positions from `rope_pos_buf` slot t.
- Q/K-norm: rms_norm already supports `nrows > 1`, so a single call
  per Qwen Q-norm and one per K-norm should work.
- KV-cache write: a single `cmd_copy_buffer` per layer (regions
  array indexed by token).

These are mechanical batchings of already-existing shaders — no new
.comp files. They'd remove the remaining per-token loop entirely
and likely close most of the prefill gap to llama.cpp Vulkan.

---

## 9 — Out of scope (this phase)

- Full 15-prompt benchmark + 4-system comparison → Phase 5B.3.
- Llama.cpp Vulkan reference run on the same 5 prompts → Phase 5B.3.
- Batching the RoPE / Q-K-norm / KV-cache-write dispatches → backlog.
- Shader-side optimisation of `flash_attn_batch.comp` (split-K, FP16 KV) → backlog.
