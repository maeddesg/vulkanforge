# Phase 5B.1 — Batched-Q Flash Attention Shader

**Date:** 2026-04-27
**Version:** v0.1.1 (no version bump — shader development)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Scope:** **shader + isolated parity tests only.** No forward-pass
integration (that's Phase 5B.2).

---

## 1 — Why

`Forward::prefill_batch` currently runs Phase-4B `flash_attn` once per
prompt token, leading to O(M) attention dispatches at prefill time. On
a 500-token prompt that's 500 dispatches at ~3 ms each = ~1500 ms in
the attention step alone — most of the gap to llama.cpp Vulkan
(VulkanForge prefill 405 tok/s vs llama.cpp 4314 tok/s).

This phase delivers `flash_attn_batch.comp`, a single-dispatch shader
that handles **all M queries × N KV positions in one go** with a
per-query causal mask. Same online-softmax inner loop as the decode
shader, but parameterised over a query index. Phase 5B.2 will wire
this into `prefill_batch`.

---

## 2 — Shader design

### 2.1 Algorithm

Identical online-softmax recurrence to Phase-4B `flash_attn.comp`:

```
for each tile of TILE=64 KV positions:
    score      = Q · K[t] · scale          (per thread, masked above causal_len)
    tile_max   = subgroupMax(score)
    new_max    = max(running_max, tile_max)
    k          = exp(running_max - new_max)
    running_out *= k
    p          = exp(score - new_max)        // 0 for masked or out-of-causal lanes
    tile_sum   = subgroupAdd(p)
    running_sum = running_sum * k + tile_sum
    running_max = new_max
    running_out += p * V[t]                  // staged via 64-float LDS
finally: O = running_out / running_sum
```

The single change vs the decode shader is the loop bound:

```
causal_len = q_start + q_idx + 1     // per-query causal triangle
            (clamped against n_kv as defence-in-depth)
```

For tile-base ≥ causal_len every lane masks to `score = -inf`, which
yields `p = 0` and contributes nothing to either `running_sum` or
`running_out`. The first valid lane is always reachable because every
batch query has at least one visible key (`q_idx ≥ 0` ⇒ `causal_len ≥ 1`).

### 2.2 Dispatch shape

```
(n_heads, M, 1)   — one workgroup per (head, query)
```

For Qwen3 / Llama-3.1 / Mistral GQA (n_heads=32, head_dim=128) the
WG count is 32 × M:

```
M=30   →    960 WGs   (already 7.5× the decode shader's 128)
M=100  →   3200 WGs
M=500  →  16000 WGs   (RX 9070 XT happily soaks this)
```

Workgroup geometry (Wave64, TILE=64, two output dims per thread, 256 B
LDS scores tile) is **bit-identical to `flash_attn.comp`** so the same
GPU-side codegen + occupancy story carries over.

### 2.3 Buffer layout

| Buffer | Layout | Note |
|---|---|---|
| `Q` | `[m, n_heads, head_dim]` (row-major) | per-query stride `n_heads × head_dim` |
| `K` | `[n_kv, n_kv_heads, head_dim]` | matches existing KV cache exactly |
| `V` | `[n_kv, n_kv_heads, head_dim]` | matches existing KV cache exactly |
| `O` | `[m, n_heads, head_dim]` | same row-major as `Q` |

K/V layout is identical to what `flash_attn.comp` already consumes,
so Phase 5B.2's `prefill_batch` integration can feed the existing KV
buffers without copying.

### 2.4 Push constants (`FlashAttnBatchPushConstants`, 28 B)

```
n_heads     u32
n_kv_heads  u32
head_dim    u32
m           u32   number of queries in this batch
n_kv        u32   total populated KV positions (= q_start + m at prefill)
q_start     u32   KV position of the first query in this batch
scale       f32   1 / sqrt(head_dim)
```

`q_start > 0` is supported up-front so the integration step can run
this shader across multiple turns of a chat (queries appended after
prior cached context).

### 2.5 LOC

| File | New | Total |
|---|---:|---:|
| `vk_shaders/flash_attn_batch.comp` | 145 | 145 |
| `src/backend/vulkan/pipeline.rs` (FlashAttnBatchPushConstants) | 24 | — |
| `src/backend/vulkan/shaders.rs` (ShaderId + blob) | 6 | — |
| `src/backend/vulkan/pipeline_registry.rs` (registration) | 9 | — |
| `build.rs` (compile job) | 8 | — |
| `tests/correctness.rs` (8 new tests) | 350 | — |
| `examples/probe_batch_attn.rs` | 295 | 295 |

SPIR-V output: **12 816 B** (vs 11 768 B for the decode shader — +9 %
size, mostly the per-query offset arithmetic).

---

## 3 — Parity tests

All 8 parity tests are isolated dispatches in `tests/correctness.rs`,
graded against an **f64** CPU reference (`cpu_batch_attn_reference`).
GQA dimensions: `n_heads=32, n_kv_heads=8, head_dim=128`.

### 3.1 Test results

| Test | M | q_start | Tolerance | max_abs_err | Result |
|---|---:|---:|---:|---:|:---:|
| `m1_vs_cpu` | 1 | 0 | 1e-4 | **0.000e0** | ✅ |
| `m4_vs_cpu` | 4 | 0 | 1e-3 | **1.79e-7** | ✅ |
| `m16_vs_cpu` | 16 | 0 | 1e-3 | **2.98e-7** | ✅ |
| `m64_vs_cpu` | 64 | 0 | 1e-3 | **7.15e-7** | ✅ |
| `m200_vs_cpu` | 200 | 0 | 5e-3 | **1.19e-6** | ✅ |
| `q_start_offset` | 4 | 60 | 1e-3 | **4.77e-7** | ✅ |
| `m1_matches_flash_attn` | 1 | 0 | 1e-5 | **< 1e-6** | ✅ |
| `causal_mask_isolates_queries` | 4 | 0 | exact (≤ 1e-5) | **< 1e-5** | ✅ |

Every error is at f32 round-off level (~1e-7 to 1e-6). The wider
tolerance bands in the test suite are deliberate margin — actual
errors come in 3-6 orders of magnitude under threshold.

### 3.2 What each test exercises

- **`m1_vs_cpu`** — softmax over a single key, `running_sum = p` and
  `O = V[0]`. Exact zero error because there's no accumulation
  reordering.
- **`m4_vs_cpu`** — full causal triangle: q=0 sees K[0], q=1 sees
  K[0..2], …, q=3 sees K[0..4]. The LDS `scores_lds` is partially
  populated (causal tail), so the V-loop needs `tile_size = min(TILE,
  causal_len - tile_base)` to stop short.
- **`m16_vs_cpu`** — 16 queries inside one TILE, exercises the
  per-query `causal_len` bookkeeping inside a single tile.
- **`m64_vs_cpu`** — exactly one full TILE for q_idx=63; tile
  boundary transition (causal_len 64 → 65) hits the tail-tile path
  for all later queries.
- **`m200_vs_cpu`** — 3.125 × TILE; deepest accumulation path,
  q_idx=199 walks 4 tiles. Numerical envelope still 1.2e-6.
- **`q_start_offset`** — `q_start=60, m=4` → causal_len 61..=64.
  Verifies the `q_start` arithmetic and the `min(causal_len, n_kv)`
  clamp.
- **`m1_matches_flash_attn`** — same Q/K/V fed to both shaders;
  outputs must agree to 1e-5. Confirms the batch shader is a strict
  superset of the decode shader at M=1.
- **`causal_mask_isolates_queries`** — runs the SAME Q[0] with
  M=1 and M=4. Query 0's output must be identical between the two
  runs (a non-causal shader would let q=0 see q=3's K/V); also
  asserts that q_3's output differs from q_0's, showing the extra
  KV positions actually changed the answer.

---

## 4 — Performance smoke

**Indicator only** — single shader dispatched in isolation, NOT a full
prefill measurement (Phase 5B.2 will produce the real numbers).

```
case        warmup_ms        median_ms     (median of 5 dispatches)
─────────────────────────────────────────
M=100            0.58            0.523
M=500           14.79            9.654
```

For comparison, the existing **per-token attention loop** at the same
M (using Phase-4C split-K decode):

```
M=100  →  100 dispatches × ~3 ms ≈ 300 ms
M=500  →  500 dispatches × ~3 ms ≈ 1500 ms
```

So the batched shader is ~150-600× faster for the attention step
alone at these batch sizes. (Phase 5B.2 will show how much of that
translates to end-to-end prefill — the projection / FFN GEMMs and the
KV-cache writes are unaffected by this change.)

`warmup_ms > median_ms` for M=500 mostly captures the first
descriptor-set / pool allocation; subsequent dispatches don't pay
that cost.

---

## 5 — Test count

```
unit (lib)         19   (no change)
correctness        33   (+8: phase5b.1 batch attn parity tests)
regression         22   (no change)
doctests            0
TOTAL              74   ALL GREEN
```

`cargo test --release` → 74 / 74 PASS.
`cargo clippy --release --tests --examples` → clean.

---

## 6 — Console summary

```
═══ Phase 5B.1 — Batch Attention Shader ═══
Tests:        8/8 parity PASS
Shader:       flash_attn_batch.comp (145 LOC, 12 816 B SPIR-V)
max_abs_err:  ≤ 1.19e-6 across all M ∈ {1, 4, 16, 64, 200}
M=100:        0.52 ms median  (vs ~300 ms in 100× per-token loop)
M=500:        9.65 ms median  (vs ~1500 ms in 500× per-token loop)
Cargo tests:  74/74 green  (+8 vs Phase 5C)
Clippy:       clean
Commit:       (to be appended after `git commit`)
```

---

## 7 — Files touched

```
NEW   vk_shaders/flash_attn_batch.comp                 batched-Q shader
NEW   examples/probe_batch_attn.rs                     parity + perf probe
NEW   results/phase5b_step_1_batch_attn.md             this report

EDIT  build.rs                                         compile job
EDIT  src/backend/vulkan/shaders.rs                    ShaderId::FlashAttnBatch
EDIT  src/backend/vulkan/pipeline.rs                   FlashAttnBatchPushConstants
EDIT  src/backend/vulkan/pipeline_registry.rs          pipeline registration
EDIT  tests/correctness.rs                             +8 phase5b.1 tests
```

No changes to `forward.rs`, `decode.rs`, `chat.rs` — the shader is
strictly additive.

---

## 8 — Out of scope

- **Forward-pass wiring** — Phase 5B.2 (`Forward::prefill_batch` will
  call `flash_attn_batch` instead of looping over `flash_attn` per
  token).
- **End-to-end prefill benchmark** — Phase 5B.3 (15-prompt suite with
  the integrated path).
- **Split-K extension** — fallback shape for very long prompts. Only
  worth pursuing if Phase 5B.3 shows the per-(head, query) WG count
  isn't enough to keep the GPU busy. With M ≥ 30 we already have
  ≥ 960 WGs, which is ample.
- **FP16 KV cache** — orthogonal change; touching weight storage,
  not attention math.
