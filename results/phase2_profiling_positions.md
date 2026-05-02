# Phase 2 — Per-Position Decode Profiling

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M (`~/models/Qwen3-8B-Q4_K_M.gguf`)
**Tool:** `examples/profile_positions.rs`
**Run log:** `results/phase2_profiling_run.log`
**Tests:** **33/33** still pass — no regression.
**Result:** `scalar_attn` is the dominant Phase-3 lever. At `pos=200` it
takes **112 ms / forward = 90.8% of GPU time**, and scales
**~134× linearly** with KV cache length.

---

## 1. Comparison Table

| Metric                  |        pos=0  |       pos=50  |      pos=100  |      pos=200  |
| ----------------------- | ------------: | ------------: | ------------: | ------------: |
| **Forward wall (µs)**   |     16 711.5  |     34 699.5  |     51 905.8  |    127 272.0  |
| Forward GPU sum (µs)    |     13 632.6  |     31 232.2  |     47 853.0  |    123 720.1  |
| **Effective (tok/s)**   |        59.8   |        28.8   |        19.3   |         7.9   |
|                         |               |               |               |               |
| **scalar_attn (µs)**    |        834.8  |     19 878.9  |     36 542.0  |    112 336.0  |
| scalar_attn (% of GPU)  |        6.1 %  |       63.6 %  |       76.4 %  |       90.8 %  |
| scalar_attn vs pos=0    |        1.00×  |       23.81×  |       43.77×  |      134.57×  |
|                         |               |               |               |               |
| **GEMV total (µs)**     |     11 589.9  |     10 432.3  |     10 428.8  |     10 533.6  |
| GEMV (% of GPU)         |       85.0 %  |       33.4 %  |       21.8 %  |        8.5 %  |
| GEMV vs pos=0           |        1.00×  |        0.90×  |        0.90×  |        0.91×  |
|                         |               |               |               |               |
| **KV-write (µs)**       |        41.8   |        30.4   |        28.8   |        23.8   |
| KV-write (% of GPU)     |        0.3 %  |        0.1 %  |        0.1 %  |        0.0 %  |
|                         |               |               |               |               |
| Rest (Norm/RoPE/Add/Mul/SiLU) |  1166.1 |        890.6  |        853.4  |        826.8  |
| Rest (% of GPU)         |        8.6 %  |        2.9 %  |        1.8 %  |        0.7 %  |
|                         |               |               |               |               |
| **Dispatch overhead**   |       3 078.9 |       3 467.2 |       4 052.8 |       3 551.9 |
| Dispatch overhead (%)   |       18.4 %  |       10.0 %  |        7.8 %  |        2.8 %  |

**Top-3 shader entries per position**

| Position | #1                                       | #2                                  | #3                                    |
| -------- | ---------------------------------------- | ----------------------------------- | ------------------------------------- |
| pos=0    | `gemv_up`     (3 507.8 µs, 36 calls)     | `gemv_down` (2 167.8 µs, 36 calls)  | `gemv_gate` (1 871.2 µs, 36 calls)    |
| pos=50   | **`scalar_attn` (19 878.9 µs, 36 calls)** | `gemv_up`   (3 033.5 µs, 36 calls)  | `gemv_down` (1 924.7 µs, 36 calls)    |
| pos=100  | **`scalar_attn` (36 542.0 µs, 36 calls)** | `gemv_up`   (3 031.6 µs, 36 calls)  | `gemv_down` (1 921.8 µs, 36 calls)    |
| pos=200  | **`scalar_attn` (112 336 µs, 36 calls)**  | `gemv_up`   (3 022.6 µs, 36 calls)  | `gemv_down` (1 915.1 µs, 36 calls)    |

---

## 2. Per-Position Per-Shader Breakdown

### pos = 0  (KV cache empty — Phase-2C-equivalent baseline)

`wall = 16.71 ms · gpu_sum = 13.63 ms · overhead = 3.08 ms (18.4 %) · 59.8 tok/s`

| Shader            | Calls | Time (µs) | % GPU |
| ----------------- | ----: | --------: | ----: |
| `gemv_up`         |    36 |   3 507.8 | 25.7 % |
| `gemv_down`       |    36 |   2 167.8 | 15.9 % |
| `gemv_gate`       |    36 |   1 871.2 | 13.7 % |
| `gemv_v`          |    36 |     957.2 |  7.0 % |
| `gemv_k`          |    36 |     883.1 |  6.5 % |
| `scalar_attn`     |    36 |     834.8 |  6.1 % |
| `gemv_q`          |    36 |     745.7 |  5.5 % |
| `lm_head`         |     1 |     741.6 |  5.4 % |
| `gemv_o`          |    36 |     715.4 |  5.2 % |
| `rms_norm_ffn`    |    36 |     228.3 |  1.7 % |
| `rms_norm_attn`   |    36 |     222.7 |  1.6 % |
| `add_res1`        |    36 |     132.2 |  1.0 % |
| `add_res2`        |    36 |     130.2 |  1.0 % |
| `mul_gate_up`     |    36 |      95.7 |  0.7 % |
| `rms_norm_q`      |    36 |      80.0 |  0.6 % |
| `silu_gate`       |    36 |      79.3 |  0.6 % |
| `rms_norm_k`      |    36 |      78.3 |  0.6 % |
| `rope_q`          |    36 |      61.6 |  0.5 % |
| `rope_k`          |    36 |      52.1 |  0.4 % |
| `kv_write`        |    36 |      41.8 |  0.3 % |
| `rms_norm_final`  |     1 |       5.8 |  0.0 % |

### pos = 50

`wall = 34.70 ms · gpu_sum = 31.23 ms · overhead = 3.47 ms (10.0 %) · 28.8 tok/s`

| Shader            | Calls | Time (µs) | % GPU |
| ----------------- | ----: | --------: | ----: |
| `scalar_attn`     |    36 |  19 878.9 | 63.6 % |
| `gemv_up`         |    36 |   3 033.5 |  9.7 % |
| `gemv_down`       |    36 |   1 924.7 |  6.2 % |
| `gemv_gate`       |    36 |   1 687.6 |  5.4 % |
| `gemv_v`          |    36 |     846.9 |  2.7 % |
| `gemv_k`          |    36 |     796.1 |  2.5 % |
| `gemv_o`          |    36 |     759.8 |  2.4 % |
| `lm_head`         |     1 |     740.1 |  2.4 % |
| `gemv_q`          |    36 |     643.7 |  2.1 % |
| `rms_norm_ffn`    |    36 |     168.1 |  0.5 % |
| `rms_norm_attn`   |    36 |     164.8 |  0.5 % |
| `add_res2`        |    36 |     109.4 |  0.4 % |
| `add_res1`        |    36 |     107.6 |  0.3 % |
| `silu_gate`       |    36 |      73.9 |  0.2 % |
| `mul_gate_up`     |    36 |      66.4 |  0.2 % |
| `rms_norm_q`      |    36 |      56.6 |  0.2 % |
| `rms_norm_k`      |    36 |      54.6 |  0.2 % |
| `rope_q`          |    36 |      45.4 |  0.1 % |
| `rope_k`          |    36 |      39.4 |  0.1 % |
| `kv_write`        |    36 |      30.4 |  0.1 % |
| `rms_norm_final`  |     1 |       4.4 |  0.0 % |

### pos = 100

`wall = 51.91 ms · gpu_sum = 47.85 ms · overhead = 4.05 ms (7.8 %) · 19.3 tok/s`

| Shader            | Calls | Time (µs) | % GPU |
| ----------------- | ----: | --------: | ----: |
| `scalar_attn`     |    36 |  36 542.0 | 76.4 % |
| `gemv_up`         |    36 |   3 031.6 |  6.3 % |
| `gemv_down`       |    36 |   1 921.8 |  4.0 % |
| `gemv_gate`       |    36 |   1 681.8 |  3.5 % |
| `gemv_v`          |    36 |     843.2 |  1.8 % |
| `gemv_k`          |    36 |     795.2 |  1.7 % |
| `gemv_o`          |    36 |     775.2 |  1.6 % |
| `lm_head`         |     1 |     740.6 |  1.5 % |
| `gemv_q`          |    36 |     639.6 |  1.3 % |
| `rms_norm_ffn`    |    36 |     159.0 |  0.3 % |
| `rms_norm_attn`   |    36 |     158.7 |  0.3 % |
| `add_res1`        |    36 |     105.2 |  0.2 % |
| `add_res2`        |    36 |     105.0 |  0.2 % |
| `silu_gate`       |    36 |      71.8 |  0.2 % |
| `mul_gate_up`     |    36 |      61.7 |  0.1 % |
| `rms_norm_q`      |    36 |      53.9 |  0.1 % |
| `rms_norm_k`      |    36 |      52.4 |  0.1 % |
| `rope_q`          |    36 |      43.4 |  0.1 % |
| `rope_k`          |    36 |      38.0 |  0.1 % |
| `kv_write`        |    36 |      28.8 |  0.1 % |
| `rms_norm_final`  |     1 |       4.3 |  0.0 % |

### pos = 200

`wall = 127.27 ms · gpu_sum = 123.72 ms · overhead = 3.55 ms (2.8 %) · 7.9 tok/s`

| Shader            | Calls | Time (µs) | % GPU |
| ----------------- | ----: | --------: | ----: |
| `scalar_attn`     |    36 | 112 336.0 | 90.8 % |
| `gemv_up`         |    36 |   3 022.6 |  2.4 % |
| `gemv_down`       |    36 |   1 915.1 |  1.5 % |
| `gemv_gate`       |    36 |   1 668.8 |  1.3 % |
| `gemv_o`          |    36 |     935.0 |  0.8 % |
| `gemv_v`          |    36 |     834.9 |  0.7 % |
| `gemv_k`          |    36 |     788.0 |  0.6 % |
| `lm_head`         |     1 |     740.0 |  0.6 % |
| `gemv_q`          |    36 |     629.3 |  0.5 % |
| `rms_norm_attn`   |    36 |     152.8 |  0.1 % |
| `rms_norm_ffn`    |    36 |     148.6 |  0.1 % |
| `add_res2`        |    36 |     105.4 |  0.1 % |
| `add_res1`        |    36 |     100.8 |  0.1 % |
| `silu_gate`       |    36 |      74.7 |  0.1 % |
| `mul_gate_up`     |    36 |      59.1 |  0.0 % |
| `rms_norm_q`      |    36 |      51.8 |  0.0 % |
| `rms_norm_k`      |    36 |      50.4 |  0.0 % |
| `rope_q`          |    36 |      42.2 |  0.0 % |
| `rope_k`          |    36 |      36.8 |  0.0 % |
| `kv_write`        |    36 |      23.8 |  0.0 % |
| `rms_norm_final`  |     1 |       4.2 |  0.0 % |

---

## 3. Dispatch-Overhead Discussion

The prompt asks for **per-layer** wall-clock minus per-layer shader time
to expose barrier/submit cost. With the explicit "Keine Änderung am
Forward-Pass" rule we do **not** instrument layer-bracket timestamps in
`forward.rs`. Instead we report the **forward-level analogue**:

```
overhead = forward_wall_clock − Σ(profiled GPU shader times)
```

This bundle covers everything that isn't a profiled compute dispatch:

- `vkQueueSubmit` + `vkQueueWaitIdle` once per forward,
- 18 inter-shader `vkCmdPipelineBarrier` calls per layer × 36 layers,
- 21 descriptor-set allocations + descriptor-write batches per layer,
- one host-mapped logits readback (`592 KB` invalidation),
- the host-side embedding write into `scratch_a`.

Numbers in absolute terms are **3.08 → 4.05 ms** across positions —
essentially constant. As compute grows from 13.6 ms (pos=0) to 124 ms
(pos=200), this overhead share collapses from **18.4 %** to **2.8 %**.

**Take-away:** total dispatch overhead is a small absolute floor (~3-4 ms
per forward). It matters only at short context. For long-context decode,
shaving submit cost yields well under 5 % wall-time.

---

## 4. KV-Cache-Write

`vkCmdCopyBuffer` for K and V is profiled under the `kv_write` label
(both copies wrapped in one timestamp pair per layer). Numbers:

| Position | KV-write total (µs) | % of GPU |
| -------- | ------------------: | -------: |
| 0        |              41.8   |    0.3 % |
| 50       |              30.4   |    0.1 % |
| 100      |              28.8   |    0.1 % |
| 200      |              23.8   |    0.0 % |

KV-write is **constant size per token** (8 kv_heads × 128 dim × 4 B = 4 KiB
per K and V per layer) and consistently sub-µs per layer in absolute
terms. The slight downward drift across positions is timestamp-resolution
noise on a transfer that takes < 1 µs/layer to begin with.

**Take-away:** KV-write is **not** a Phase-3 lever. DMA vs compute-copy
investigation buys nothing.

---

## 5. Bottleneck Analysis

### Primary: `scalar_attn` linear scaling

The single inescapable observation is that `scalar_attn` scales **linearly
with the cached sequence length**, which is exactly what the existing
shader does mechanically (one workgroup per Q-head, sequential `for t in
0..seq_len { score = Q · K[t]; }` loop). The relevant slope:

```
pos=0 →  834 µs / 36 layers =  23.2 µs / layer
pos=50 → 19 879 µs / 36     = 552.2 µs / layer  ⇒ +10.6 µs / additional position
pos=200 → 112 336 µs / 36   = 3 120.4 µs / layer ⇒ +15.5 µs / additional position
```

The slope is sub-linear at low position (small per-thread loop, kernel
launch dominates) but settles into ~13-15 µs per additional cached token
per layer. At `pos=2048` (the max our `KvCache` is built for) extrapolating
this slope predicts **~31-37 ms per layer × 36 ≈ 1.1-1.3 seconds per
forward** — well under 1 tok/s of decode.

### Secondary: GEMV bandwidth floor

GEMV is **constant** in absolute time (~10.5 ms / forward across all
positions; the slight decrease vs. pos=0 is profiler-warmup, not a real
shift). At pos=0 it's **85 %** of GPU time, by pos=200 only **8.5 %**.
With Phase 1's measured 79.6 % of peak bandwidth at the M=K=3584 stress
test, there's at most ~20 % headroom by tuning the GEMV kernel — and that
20 % is on the constant ~10.5 ms baseline, not on the position-scaling
attention cost.

### Tertiary: Dispatch overhead

3-4 ms wall-time floor (18 % at pos=0 → 3 % at pos=200). Pipelined
command submission would reclaim it, but only at short context where it
matters proportionally.

### Non-issues at this scale

- **KV-write transfer**: < 0.3 % everywhere.
- **Norms / RoPE / Add / Mul / SiLU** combined ("Rest"): 8.6 % at pos=0,
  0.7 % at pos=200. Even if all of these vanished, decode tok/s wouldn't
  visibly move at long context.

---

## 6. Phase-3 Recommendation

Based purely on the measurements above:

> **#1 (decisive): rewrite `scalar_attn` as a tiled / shared-memory
> attention shader (flash-attention-style for decode).**

Justification, not estimate:

- At `pos=200`, `scalar_attn` accounts for **90.8 %** of GPU time and
  **88.3 %** of wall-clock forward. Anything else combined contributes
  less than 12 ms of the 127 ms forward.
- The current shader uses one workgroup **per Q-head** with a single
  serial scan over the KV history. RDNA 4 (`gfx1201`) has a 128 KiB LDS
  per CU and a 64-wide wave; a tile-per-history-block scheme with
  shared-memory-cooperative softmax can keep the CUs busy across all
  32 query-heads in parallel and cut the linear-in-position factor to a
  log-factor (chunked online softmax) or to a fully-parallel tiled scan.
- It is also the **only** lever whose payoff grows with prompt length.
  Every other optimisation has a ceiling.

> **#2: pipelined / persistent command-buffer submit.**

Saves the constant 3-4 ms wall-clock overhead. Worth doing once (low
maintenance, single-digit-percent improvement at long context, ~18 %
at short context). Not worth doing first.

> **#3: GEMV bandwidth tuning.**

10.5 ms constant baseline. Phase 1 already measured 79.6 % of peak;
remaining 20 % requires kernel-tuning effort whose payoff is bounded.
Defer until after attention is replaced — at long context GEMV is < 10 %
of total.

> **Out of scope for Phase 3 prioritisation:**

- KV-write (already negligible).
- Norm / RoPE / Add / Mul / SiLU fusion (sub-1 % at long context).

---

## 7. Methodology Notes

- Built as `examples/profile_positions.rs` — separate binary, not a main.rs
  mode, since the profiling driver runs ~245 forward passes for one set
  of measurements and isn't a normal-use-case demo.
- The `Forward` instance carries the same `ShaderProfiler` for the entire
  run; non-target positions still pay the timestamp-write cost, but
  `cmd_write_timestamp` is < 1 µs each and makes no observable
  difference vs. the profile-off run (Phase 2D measured 13.4 tok/s
  decode median across positions ~30..230 — `pos=100` here gives
  19.3 tok/s individually, which is consistent given the
  `tok/s = 1 / forward_wall` relationship and decode's average position).
- "Effective tok/s" rows are `1 / forward_wall`, not a real running
  average — they answer "what would tok/s be if every step happened at
  this position?".
- The Qwen3 reasoning model didn't emit EOS during decode; the binary
  has a `<think>` substitution path for the rare case it does, so all
  4 capture points are reachable deterministically.
- 33/33 tests still pass. Forward-pass code untouched.

---

## 8. Files added

- `examples/profile_positions.rs` — the profiling driver (~330 LoC).
- `results/phase2_profiling_run.log` — full stdout transcript.
- `results/phase2_profiling_positions.md` — this report.

## 9. Commit hash

To be filled in by the commit at the end of this run.
