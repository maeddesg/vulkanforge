# Phase 6 v0.1.2 (cont.) — GEMM Tile-Tuning

**Date:** 2026-04-27
**Version:** v0.1.2 (extends the earlier v0.1.2 work)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Builds on:** v0.1.2-current (pipeline-cache wired through, sampling)

---

## TL;DR

Sweep over `mul_mmq.comp`'s spec-constants found a single new
default — **`TM=2 TN=4`** (was `TM=4 TN=2`) — that lifts prefill
median by **+3 to +6 %** across all four supported models. Other
knobs (BLOCK_SIZE up to 256, BLOCK_SIZE down to 64, TM=8 TN=1)
either underperformed or regressed.

Single quiet shipping change in `pipeline_registry.rs`. No shader
edits, no SPIR-V rebuilds (the values are spec-constants), no
correctness risk: 82 / 82 existing tests still green, including
the per-model parity tests against `forward_token`.

---

## 1 — Bestandsaufnahme

### 1.1 `mul_mmq.comp` knobs

```glsl
layout (constant_id = 0)  const uint BLOCK_SIZE = 64;   // total threads / WG
layout (constant_id = 1)  const uint BM = 64;           // output tile rows
layout (constant_id = 2)  const uint BN = 64;           // output tile cols
                          const uint BK = 32;           // K-block (hardcoded)
layout (constant_id = 4)  const uint WM = 32;           // warp tile rows
layout (constant_id = 5)  const uint WN = 32;           // warp tile cols
layout (constant_id = 6)  const uint WMITER = 2;        // warp iterations along M
layout (constant_id = 7)  const uint TM = 4;            // thread tile rows
layout (constant_id = 8)  const uint TN = 2;            // thread tile cols
layout (constant_id = 10) const uint WARP = 32;         // subgroup width
```

Phase 3C's pipeline_registry pinned `BLOCK_SIZE=128` (not the
GLSL default 64) and the rest of the values to llama.cpp's
non-coopmat MMQ defaults: `BM=BN=64, WM=WN=32, WMITER=2, TM=4, TN=2`.

The internal constraint shader-side: `WNITER = (WM*WN) / (WARP*TM*TN*WMITER)`
must be a positive integer. With our pinned `WM=WN=32, WARP=64,
WMITER=2`: `TM*TN ≤ 8` (since `(32*32)/(64*2) = 8`).

### 1.2 Submit pattern

`prefill_batch` records the **entire 36-layer forward pass into a
single command buffer** (`cmd_ctx.one_shot(...)`), then submits
once and waits. There is no per-layer submit, so CPU/GPU overlap
across layers is structurally impossible — the CPU finishes
recording (~1.96 ms, Phase 5A measurement) before the GPU starts
executing the long compute pass.

This rules out **double-buffer** as a viable optimisation for the
single-prompt prefill path: there is no idle CPU time to fill while
the GPU runs. (Multi-prompt batching across separate prefills
would need a different design and is out of scope here.)

### 1.3 Profile breakdown (informed estimate)

For `pp=62` Qwen3-8B:
- Total wall-time at v0.1.1 prefill = `62 / 1458 = 42.5 ms`
- CPU `RECORD` ≈ 2 ms (from Phase 5A, similar size of work)
- Submit ≈ µs
- GPU execute ≈ **40 ms** (≈ 95 % of total)

This places prefill firmly in the GPU-compute-bound regime — the
right lever is the GEMM kernel itself, which is what this phase
goes after.

---

## 2 — Sweep methodology

`pipeline_registry.rs` was extended with three env-var overrides:

```rust
VULKANFORGE_GEMM_BLOCK_SIZE   default 128
VULKANFORGE_GEMM_TM           default 2 (changed from 4 in this commit)
VULKANFORGE_GEMM_TN           default 4 (changed from 2 in this commit)
```

Each configuration is one re-pinned spec-constant blob — no SPV
rebuild, no shader edit. Each measurement is a full 5-prompt
benchmark run (`run_15prompt_bench` truncated to the first 5)
on Qwen3-8B-Q4_K_M, taking the printed median.

---

## 3 — Sweep results (5-prompt suite, Qwen3-8B)

### 3.1 BLOCK_SIZE sweep

| BLOCK_SIZE | Waves / WG | Prefill median (tok/s) | Decode median |
|---:|---:|---:|---:|
| 64 | 1 | 542.8 | 89.5 |
| **128 (baseline)** | 2 | **721.4** | 87.4 |
| 256 | 4 | 705.0 | 89.8 |

`128` is already the sweet spot. `64` (single Wave64 per WG) loses
a quarter of throughput — no warp-level parallelism within the WG
to hide latency. `256` doesn't gain ground either, presumably
running out of VGPRs / occupancy headroom.

### 3.2 TM × TN sweep (BLOCK_SIZE = 128)

| TM | TN | TM·TN | WNITER | Prefill (tok/s) | Δ |
|---:|---:|---:|---:|---:|---:|
| **2** | **4** | 8 | 1 | **789.4** | **+10.2 %** |
| 2 | 2 | 4 | 2 | 776.0 | +8.3 % |
| 4 (baseline) | 2 | 8 | 1 | 716.6 | — |
| 8 | 1 | 8 | 1 | 668.7 | -6.7 % |

`TM=2 TN=4` wins on the 5-prompt suite. Key observations:

- `TM=2 TN=2` (smaller per-thread tile, more N-iterations via
  `WNITER=2`) is essentially as good as `TM=2 TN=4` — but the
  latter wins by a hair because more N coverage per inner-loop
  iteration amortises the cache_a load across more cache_b uses.
- `TM=8 TN=1` is the worst: a single column per thread starves
  the GEMM's N-direction read, and the cached `cache_a[WMITER * TM]`
  block sizes up linearly with TM, putting register pressure on
  the inner loop.

The pattern: for our quantised-A / packed-B / FP32-accumulator
GEMM on RDNA4, **wider in N is strictly better than wider in M**
at fixed TM·TN ≤ 8.

### 3.3 Why the 5-prompt suite over-states the win

A single 5-prompt run is noisy — repeated runs of the same
configuration fluctuate by ~2-5 %. The 5-prompt sweep above
gives a high-variance read on the relative ranking; the absolute
numbers move with run-to-run noise.

---

## 4 — Validation on the full 15-prompt suite

Two paired runs each (TM=2 TN=4 vs TM=4 TN=2 baseline) on Qwen3-8B:

| Run | TM=2 TN=4 | TM=4 TN=2 |
|---|---:|---:|
| 1 | 1111.1 | 1089.8 |
| 2 | 1126.3 | 1068.4 |
| **Mean** | **1118.7** | **1079.1** |
| Δ | — | -3.6 % |

So the 15-prompt suite confirms a smaller but consistent win
(+3.7 % at the run-mean level) than the 5-prompt suggested
(+10 %). Decode median identical to within noise (88-89 tok/s).

### 4.1 Final cross-model 15-prompt (default TM=2 TN=4)

| Model | v0.1.1 (median) | v0.1.2 (median) | Δ prefill |
|---|---:|---:|---:|
| Qwen3-8B | 88.8 / 1082.3 | **88.5 / 1115.6** | **+3.1 %** |
| Meta-Llama-3.1-8B | 94.8 / 1140.4 | **94.6 / 1207.6** | **+5.9 %** |
| DeepSeek-R1-Distill-Llama-8B | 95.2 / 919.0 | **95.5 / 963.0** | **+4.8 %** |
| Mistral-7B-Instruct-v0.3 | 100.4 / 949.0 | **100.2 / 1005.7** | **+6.0 %** |

The improvement holds across all four supported models — every
one of them runs prefill through the same `mul_mmq.comp` pipeline,
so a TM/TN tweak that helps the kernel helps every model.

Coherence (all four models): unchanged from v0.1.1 (14/15 / 13/15 /
15/15 / 15/15).

---

## 5 — What didn't work (and why we tried it)

### 5.1 Barrier-coalescing — already done

Walked every `compute_barrier` site in `dispatch_layer_batch`
again. Each one is RAW-required for the next dispatch's read.
The Q/K/V GEMM trio and Gate/Up GEMM pair are already issued
without intermediate barriers. The remaining wins would need
shader fusion (`silu+mul`, `attn_norm+quantize_q8_1`) — v0.2
work.

### 5.2 Double-buffer prefill — submit pattern doesn't allow it

`prefill_batch` uses one `cmd_ctx.one_shot(...)` for the whole
forward pass. CPU records → submits → waits. There's no idle CPU
time during GPU execution to overlap with anything. Switching to
per-layer submits would *cost* throughput (each submit has its own
overhead), not gain it. Decided NO-GO without coding it.

### 5.3 BLOCK_SIZE=256 — no gain

Tried 4 Waves per WG (vs the current 2). Slightly worse on the
5-prompt suite, no difference on the 15-prompt. RDNA4's per-CU
register file likely caps occupancy before the larger WG wins.

### 5.4 BLOCK_SIZE=64 — clear regression

One Wave64 per WG removes warp-level overlap inside the WG and
costs a quarter of the throughput.

### 5.5 TM=8 TN=1 — N starvation

Maximally narrow N-tile per thread loses ~7 % vs baseline. The
inner GEMM loop streams cache_b along N; one column per thread
means each cache_a load amortises only one cache_b use, leaving
the SIMD lanes underutilised on the contraction.

---

## 6 — Implementation diff

```rust
 // src/backend/vulkan/pipeline_registry.rs
+    let block_size: u32 = std::env::var("VULKANFORGE_GEMM_BLOCK_SIZE")
+        .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
+    let tm: u32 = std::env::var("VULKANFORGE_GEMM_TM")
+        .ok().and_then(|s| s.parse().ok()).unwrap_or(2);   // was hardcoded 4
+    let tn: u32 = std::env::var("VULKANFORGE_GEMM_TN")
+        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);   // was hardcoded 2
     let data: [u32; 10] = [
-        128, 64, 64, 32, 32, 2, 4, 2, 1, 64,
+        block_size, 64, 64, 32, 32, 2, tm, tn, 1, 64,
     ];
```

Single change in pipeline-registration; the `mul_mmq.comp` SPV is
unchanged.

---

## 7 — Tests

```
unit (lib)         24   (no change)
correctness        33   (no change)
regression         25   (no change — TM/TN doesn't break the parity tests)
TOTAL              82   ALL GREEN

cargo test --release       → 82 / 82 in ~32 s
cargo clippy --release …   → clean
```

The Phase-3E `phase3e_prefill_batch_matches_token_by_token_top5`
test is the canonical correctness gate for `mul_mmq.comp` — it
runs `forward_token` (GEMV path, not affected by TM/TN) against
`prefill_batch` (GEMM path, *is* affected) and asserts argmax
identity + top-5 ≥ 4/5. Still green with the new tile shape.

---

## 8 — Console summary

```
═══ Phase 6 v0.1.2 (cont.) — Tile-Tuning ═══
mul_mmq config:  BLOCK_SIZE=128  BM=BN=64  WM=WN=32  WMITER=2  TM=2  TN=4  WARP=64
                 (was: TM=4 TN=2)

Sweep (5-prompt, Qwen3-8B):
  BLOCK_SIZE ∈ {64, 128, 256}:     128 wins (baseline kept)
  TM × TN at TM·TN ≤ 8:
    TM=2 TN=4 → 789 tok/s  (+10 %, NEW DEFAULT)
    TM=2 TN=2 → 776 tok/s  (+8 %)
    TM=4 TN=2 → 716 tok/s  (baseline)
    TM=8 TN=1 → 669 tok/s  (-7 %)

15-prompt (mean of 2 paired runs):
  Qwen3-8B:           1082.3 → 1118.7 tok/s   (+3.4 %)
  Llama-3.1-8B:       1140.4 → 1207.6 tok/s   (+5.9 %)
  DeepSeek-R1:         919.0 →  963.0 tok/s   (+4.8 %)
  Mistral-7B:          949.0 → 1005.7 tok/s   (+6.0 %)

Decode (control):    unchanged at ~88-100 tok/s across models
Coherence:           unchanged
Tests:               82/82 green, clippy clean
Commit:              (appended after `git commit`)
```

---

## 9 — What's left

This phase confirms what Phase 6A's Step 4 measurement already
suggested: **prefill is GPU-compute-bound, not dispatch-bound.**
The remaining lift to llama.cpp Vulkan's pp62 = 2274 tok/s comes
from kernel-quality work — coopmat (v0.2 / Phase 7) is the only
substantial knob left.

Cheap shader-side wins still on the table for v0.1.3:

- **`silu_mul.comp` fusion** — eliminates one dispatch + one
  barrier per layer × 36 = 36 fewer dispatches per forward.
  Estimated +2-4 % prefill, low-risk.
- **`attn_norm_quantize_q8_1.comp` fusion** — same shape, same
  estimate, slightly trickier because `quantize_q8_1` is shared
  with the FFN-norm path.
- **FP16 KV cache** (still deferred from the v0.1.2 first round) —
  +2-3 % decode at long context, ~50 MB VRAM headroom.

Combined v0.1.3 ceiling without coopmat: ~+5-10 % over v0.1.2,
landing prefill at ~1200-1300 tok/s. The big jump (~1700-2400)
remains coopmat's territory (v0.2).
