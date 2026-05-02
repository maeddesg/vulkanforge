# Phase 5B.3 — Per-Token Loop Eliminated

**Date:** 2026-04-27
**Version:** v0.1.1 (no version bump — integration step)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Builds on:** Phase 5B.2 — `flash_attn_batch` integration (425 tok/s prefill)

---

## 1 — Bestandsaufnahme: shaders are already batch-ready

Read first, before changing anything. Findings:

### `rope_neox.comp` / `rope_norm.comp`

```glsl
const uint i0  = 2*gl_GlobalInvocationID.y;
const uint row = gl_GlobalInvocationID.x + 32768 * gl_GlobalInvocationID.z;
if (row >= pc.nrows) return;
const uint i3 = row / (pc.ne01*pc.ne02);
const uint i2 = (row - i3 * pc.ne01*pc.ne02) / pc.ne01;   // ← token index
const uint i1 = (row - i3 * pc.ne01*pc.ne02 - i2 * pc.ne01); // ← head index
rope_neox(i0, i1, i2, i3, pc);
```

The position is read from the storage buffer **`rope_data_pos[i2]`**
inside `rope_funcs.glsl` — i.e. the shader is already
**multi-position-aware**. With `ne02 = M` and a `rope_pos_buf` filled
with `[base_pos+0, base_pos+1, …, base_pos+M-1]`, one dispatch covers
every (token, head) pair.

### `rms_norm.comp`

```glsl
const uint nrows = gl_NumWorkGroups.x;
const uint row   = gl_WorkGroupID.x;
```

`rms_norm` already iterates rows from `gl_WorkGroupID.x`. Dispatching
`(M × heads, 1, 1)` instead of `(heads, 1, 1)` runs Q-norm or K-norm
across every (token, head) row, with the single-row weight broadcast
across all rows by the existing `ne11=1` convention.

**Conclusion:** zero shader changes needed. Phase 5B.3 is purely a
host-side restructuring of `dispatch_layer_batch`.

---

## 2 — Integration

### 2.1 Old (Phase 5B.2)

```
for layer in 0..n_layers:
    attn_norm_batch + quantize + GEMM_q/k/v          (already batched)

    for t in 0..seq_len:                              ← M-fold loop
        copy batch_q[t] → q_buf                        (3 cmd_copy)
        copy batch_k[t] → k_buf
        copy batch_v[t] → v_buf
        Q-norm / K-norm (Qwen)                         (2 dispatches)
        RoPE Q + RoPE K                                (2 dispatches)
        cmd_copy_buffer K, V into KV-cache slot        (2 cmd_copy)
        copy q_buf → batch_q[t]                        (1 cmd_copy, 5B.2)

    flash_attn_batch                                  (1 dispatch, 5B.2)
    GEMM_o + residual + FFN-norm + GEMM ffn / silu / mul / GEMM down + residual2
```

Per-layer per-token sub-dispatch count: **10 × M**.

### 2.2 New (Phase 5B.3)

```
for layer in 0..n_layers:
    attn_norm_batch + quantize + GEMM_q/k/v

    if batch_attn_enabled:                           ← Phase 5B.3 default
        rms_norm Q-batch (Qwen-only)                  (1 dispatch, nrows=M*n_heads)
        rms_norm K-batch (Qwen-only)                  (1 dispatch, nrows=M*n_kv_heads)
        rope_batch Q                                  (1 dispatch)
        rope_batch K                                  (1 dispatch)
        cmd_copy_buffer batch_k → kv_cache (M rows)   (1 cmd_copy)
        cmd_copy_buffer batch_v → kv_cache (M rows)   (1 cmd_copy)
    else:                                             ← legacy fallback
        for t in 0..seq_len:
            (the old per-token loop, unchanged)

    flash_attn_batch                                  (1 dispatch, 5B.2)
    GEMM_o + residual + FFN…
```

Per-layer dispatch count when `batch_attn_enabled = true`:
- Qwen3 (has Q/K-norm): **6 ops** (Q-norm, K-norm, RoPE Q, RoPE K,
  K-bulk-copy, V-bulk-copy) **+ flash_attn_batch + GEMMs** ≈ ~13–15
  total dispatches per layer.
- Llama-3 / Mistral / DeepSeek (no Q/K-norm): 4 ops + attention.

### 2.3 Dispatch-count reduction

For Qwen3 with `pp=62`:

| Path | Pre-attn ops | Total / layer | × 36 layers |
|---|---:|---:|---:|
| 5B.2 per-token loop | 10 × 62 = 620 | ~635 | **~22 860** |
| 5B.3 fully batched | 6 | ~21 | **~756** |

Net reduction: **~30 ×**.

---

## 3 — Buffer + barrier discipline

The new path operates **in-place** on `batch_q` and `batch_k`:

```
batch_q  (Q-GEMM out, [M, n_heads, head_dim])
   ↓ rms_norm Q-batch (Qwen)               (in-place: input == output)
   ↓ rope_batch Q                           (in-place)
   → flash_attn_batch reads it as Q

batch_k  (K-GEMM out, [M, n_kv_heads, head_dim])
   ↓ rms_norm K-batch (Qwen)
   ↓ rope_batch K
   ↓ cmd_copy_buffer batch_k → kv_cache    (M contiguous rows starting at base_pos)

batch_v  (V-GEMM out — no norm, no RoPE)
   ↓ cmd_copy_buffer batch_v → kv_cache    (same shape as K)
```

Barriers:

1. After Q/K-norm + RoPE: COMPUTE_SHADER → COMPUTE_SHADER barrier
   (subsequent dispatch reads the post-RoPE values).
2. After bulk KV-write: TRANSFER+COMPUTE → COMPUTE_SHADER barrier
   (`flash_attn_batch` reads the cache; the cmd_copy_buffer wrote
   the K side, the V side is also TRANSFER, and the existing
   pre-attention barrier from `compute_barrier` afterwards covers
   the post-attention reads).

The `batch_v` buffer takes the bulk-copy directly out of the
V-GEMM output — no synchronisation problem because batch_v is not
modified between the GEMM and the copy.

KV-cache layout: `[max_seq, n_kv_heads, head_dim]` per layer. Source
`batch_k` / `batch_v` are `[M, n_kv_heads, head_dim]`. The row-stride
matches (`n_kv_heads × head_dim × 4` bytes), so a single
`cmd_copy_buffer` of `M × row_bytes` lands the whole batch at the
correct cache slot.

---

## 4 — Files touched

```
EDIT  src/backend/vulkan/forward.rs            +run_rope_batch
                                               + dispatch_layer_batch:
                                                  Phase 5B.3 batched-prep block
                                                  per-token loop now gated on
                                                  !batch_attn_enabled (legacy path)
NEW   results/phase5b_step_3_batch_ops.md      this report
```

No shader changes. No new push-constant types. No KV-cache layout
changes. No new buffers.

---

## 5 — Correctness

### 5.1 Existing tests carry forward

All Phase-3 / 5A / 5B.2 / 5C / Prompt-16 tests keep passing through
the new fully-batched default path:

| Test | Coverage | Result |
|---|---|:---:|
| `phase3e_prefill_batch_matches_token_by_token_top5` | full prefill vs forward_token argmax + top-5 | ✅ |
| `phase5b2_batch_attn_parity_qwen3_short` | `batch_attn=false` (legacy per-token) vs `=true` (5B.3 batched) — argmax + top-5 | ✅ |
| `phase5b2_batch_attn_parity_qwen3_two_tiles` | same, 64-token prompt across two attention TILEs | ✅ |
| `phase5b2_decode_after_batched_prefill_qwen3` | multi-turn KV-cache survival after batched prefill | ✅ |
| `phase_prompt16_alice_context_retention_qwen3` | 6-turn Alice multi-turn (3/3 critical) | ✅ |

`phase5b2_batch_attn_parity_qwen3_short` and `_two_tiles` are
particularly important here: they explicitly compare the **legacy
per-token loop** (`batch_attn_enabled=false`) against the **fully-
batched 5B.3 path** (`batch_attn_enabled=true`) on the same prompt
and assert argmax + top-5 ≥ 4 / 5. Both pass with the new code,
which means batched RoPE + batched RMS-norm + bulk KV-write
produces the same logits as the per-token loop.

### 5.2 Cross-model Alice (multi-turn smoke)

All four supported models pass the 3-critical-turn Alice test with
the 5B.3 path active by default:

| Model | Critical | First-turn prefill | First-turn decode |
|---|:---:|---:|---:|
| Qwen3-8B | 3 / 3 ✅ | 463 tok/s | 90.0 tok/s |
| Meta-Llama-3.1-8B | 3 / 3 ✅ | 516 tok/s | 98.8 tok/s |
| DeepSeek-R1-Distill | 3 / 3 ✅ | 337 tok/s | 97.9 tok/s |
| Mistral-7B-v0.3 | 3 / 3 ✅ | 354 tok/s | 105.6 tok/s |

Soft turn 2 ("What is 2+2?") still fails for Qwen3 / DeepSeek due
to the `<think>` block exceeding the 40-token budget — that's a
budget detail, not a Phase 5B.3 issue (same outcome as Phase 5C).

### 5.3 Test count

```
unit (lib)         19   (no change)
correctness        33   (no change)
regression         25   (no change — 5B.2 tests carry forward)
doctests            0
TOTAL              77   ALL GREEN

cargo clippy --release --tests --examples  →  clean
cargo test --release  →  77 / 77 in ~36 s (vs ~86 s in Phase 5B.2)
```

The regression-suite wall-clock time itself dropped >2 × — every
prefill in every test now uses the batched path.

---

## 6 — Performance (5-prompt smoke, Qwen3-8B)

`run_15prompt_bench` truncated to the first 5 prompts. Greedy
decode, `think_filter=false`. Two runs, only env var differs.

### 6.1 Per-prompt prefill throughput

| # | Prompt | pp tok | gen tok | OFF (legacy) | ON (5B.3) | vs 5B.2 ON | Speedup vs OFF |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Greeting | 20 | 64 | — | 374.8† | (n/a) | n/a |
| 2 | Simple Sequence | 31 | 64 | 338.8 | 711.5 | 425.4 → 711.5  (+67 %) | **+110 %** |
| 3 | Prime Check (Python) | 31 | 256 | 339.5 | 719.7 | 418.5 → 719.7  (+72 %) | **+112 %** |
| 4 | LRU Cache (C++) | 47 | 512 | 410.8 | 1122.1 | 547.1 → 1122.1 (+105 %) | **+173 %** |
| 5 | REST API (Go) | 62 | 1024 | 455.1 | 1458.8 | 652.0 → 1458.8 (+124 %) | **+221 %** |
| | **MEDIAN** | | | **339.5** | **719.7** | 425.4 → 719.7 (+69 %) | **+112 %** |
| | **AGGREGATE** | 191 | 1920 | **374.4** | **851.2** | 483.3 → 851.2 (+76 %) | **+127 %** |

†Greeting run with default (5B.3 ON), no separate OFF measurement.

### 6.2 Decode throughput (control)

| Path | Median decode tok/s |
|---|---:|
| OFF (legacy per-token) | 88.7 |
| ON (5B.3 fully batched) | 88.6 |

**Delta 0.1 tok/s** — Phase 5B.3 only changes the prefill attention
prep; decode is unaffected. This confirms no regression on the
hot path.

### 6.3 4-system reference, prefill comparison

```
                          Decode tok/s   Prefill tok/s
llama.cpp Vulkan:         114.2          4314           reference
ROCmForge HIP:            95.4           768.6          previous prefill ceiling
llama.cpp ROCm:           87.5           3684
VulkanForge Phase 5B.3:   88.6           719.7 (med) / 851.2 (agg)
```

VulkanForge prefill went from **~9 % of llama.cpp Vulkan** at the
Phase-5A baseline (405 tok/s) to **~20 % at Phase 5B.3** (851 tok/s
aggregate) — and is now **above the ROCmForge HIP ceiling**.
Decode (88.6 tok/s) sits at ~78 % of llama.cpp Vulkan, unchanged
from earlier phases.

The remaining gap to llama.cpp Vulkan's prefill is mostly in the
GEMM pipeline (mul_mmq utilisation) and pipeline-cache warm-up. The
attention path no longer dominates.

---

## 7 — Console summary

```
═══ Phase 5B.3 — Fully Batched Prefill ═══
RoPE:         batched (one dispatch per Q/K, ne02=M, rope_pos_buf[i2])
Q/K-norm:     batched (rms_norm with nrows = M * heads_per_token)
KV-Write:     1× bulk cmd_copy_buffer per K and per V per layer
Per-Token-Loop: ELIMINATED (legacy path retained under VULKANFORGE_BATCH_ATTN=0)
Dispatches:   ~22 860 → ~756 (pp=62, ~30× reduction)
Prefill:      425.4 → 719.7 tok/s median (+69 %, +127 % vs OFF)
Decode:       88.6 tok/s (unchanged ✅)
Parity:       argmax identical, top-5 ≥ 4/5 (vs legacy per-token)
Cross-model:  Alice 3/3 critical on all 4 models (Qwen3, Llama-3.1, DeepSeek, Mistral)
Tests:        77/77 green
Regression:   86 s → 36 s wall-clock
Clippy:       clean
Commit:       (appended after `git commit`)
```

---

## 8 — What's left for Phase 5B.4 / future work

- **GEMM pipeline-utilisation work.** With prefill attention at
  ~roundtrip + ~ε, the next bottleneck moves to the per-layer GEMM
  Q/K/V/O/gate/up/down. llama.cpp Vulkan reaches >4000 tok/s prefill
  through `coopmat`-fused matrix-vector and aggressive
  pipeline-cache pre-warming — both candidates for follow-up.
- **15-prompt + llama.cpp 4-system comparison.** Out of scope here;
  the next phase will run the full suite and produce the 4-row
  comparison table from §6.3 with all 5 prompts and across all
  4 models.
- **FP16 KV-cache.** Still half-precision elsewhere; cuts KV memory
  in half and unblocks larger contexts.

---

## 9 — Out of scope (this phase)

- 15-prompt benchmark + 4-system comparison (next phase).
- Llama.cpp Vulkan reference run on the same prompts (next phase).
- GEMM pipeline-cache work, batch-size sweep, FP16 KV-cache.
