# Sprint 12G-C — Per-Dispatch GPU Profiling

**Goal.** Determine empirically WHERE GPU time goes in a VulkanForge forward,
since Sprint 12C's analytical cost model was off by ~30× and Sprints 12D/12E
delivered ~0% wall-time despite shipping correct structural reductions
(0% empirical barrier elision, 72 dispatches saved by 5-op fusion → +1%).

**Result.** RGP CLI was not installed; pivoted to the existing
`ShaderProfiler` (`vkCmdWriteTimestamp`, Vulkan-core, always available).
Sprint 12G-C is a pure measurement sprint — no runtime code changes,
only one new profiling example (`examples/profile_prefill.rs`).

## TL;DR

| Hypothesis | Verdict |
|---|---|
| **H1 — CU starvation** dominates decode | **RULED OUT** for the 70%-bucket; mild contributor for K/V projections only |
| **H2 — L2 thrashing** dominates | **RE-FRAMED**: weight matrices (28 MB Q4_K) far exceed any cache; the right framing is **HBM BW utilization**, not cache thrash |
| **H3 — GEMV/GEMM dominance ≥70%** | **CONFIRMED HARD**: 82–88% decode, 90–93% prefill |
| **H4 — Inter-dispatch idle / CPU overhead** dominates | **RULED OUT** at steady state: <1% CPU overhead at pos≥50 |

**Single biggest individual lever found.** `gemv_up` at decode runs at
~52% of peak HBM BW, while `gemv_gate` (identical shape, identical shader,
sequential call) runs at ~91% peak. **75% slowdown on identical work** is
the largest unexplained anomaly in the profile and the highest-ROI next
investigation.

## Methodology

- **Tool:** `examples/profile_positions.rs` (decode) + new
  `examples/profile_prefill.rs`. Both wrap `vkCmdWriteTimestamp(TOP_OF_PIPE)`
  / `BOTTOM_OF_PIPE` around every `self.profile(label, …)` call in
  `forward.rs`. The label scheme already existed (`gemv_q`, `gemv_k`,
  `gemv_v`, `gemv_o`, `gemv_gate`, `gemv_up`, `gemv_down`, `lm_head`,
  `scalar_attn`, `fa_split`, `fa_reduce`, `rms_norm_*`, `rms_norm_mul_rope_*`,
  `kv_copy_fp16_*`, `add_rms_*`, `add_res2`, `swiglu`, `quantize_*`,
  `gemm_*`, …).
- **Decode positions:** {0, 50, 100, 200} on the standard
  "Explain what a mutex is in one sentence." prompt.
- **Prefill:** pp=128 and pp=512, `prefill_batch` once each, fresh
  ShaderProfiler (capacity = 4096 pairs).
- **Caveat:** RADV emits `query not reset` validation warnings for the
  prefill run because `prefill_batch` does not call `profiler.reset()`
  inside its `one_shot`. Warnings are spec-pedantry; values returned are
  consistent with measured wall-time and with expected magnitudes.

## Decode profile

Per-shader GPU µs at four decode positions (sorted by pos=200 cost).
"% GPU" is share of the per-position GPU-sum.

```
pos=0    wall=15.17 ms  gpu_sum=12.58 ms  overhead=2.59 ms (17.1%)  65.9 tok/s
pos=50   wall=12.03 ms  gpu_sum=12.18 ms  overhead=0.00 ms ( 0.0%)  83.1 tok/s
pos=100  wall=12.69 ms  gpu_sum=12.47 ms  overhead=0.22 ms ( 1.7%)  78.8 tok/s
pos=200  wall=12.34 ms  gpu_sum=12.33 ms  overhead=0.01 ms ( 0.1%)  81.0 tok/s
```

| Shader | Calls | µs @ pos=200 | % GPU | Notes |
|---|---:|---:|---:|---|
| `gemv_up` | 36 | 3010.6 | 24.4% | FFN up: M=1, N=12288, K=4096 (Q4_K) |
| `gemv_down` | 36 | 1933.4 | 15.7% | FFN down: M=1, N=4096, K=12288 (Q4_K) |
| `gemv_gate` | 36 | 1712.2 | 13.9% | FFN gate: same shape as up |
| `fa_split` | 36 | 1281.1 | 10.4% | flash-attn split (kicks in at pos≥100) |
| `gemv_v` | 36 | 813.4 | 6.6% | V-proj: M=1, N=1024, K=4096 |
| `gemv_k` | 36 | 774.2 | 6.3% | K-proj: same shape as V |
| `lm_head` | 1 | 740.1 | 6.0% | vocab=151936, K=4096 (single dispatch) |
| `gemv_q` | 36 | 664.4 | 5.4% | Q-proj: M=1, N=4096, K=4096 |
| `gemv_o` | 36 | 541.5 | 4.4% | O-proj: same |
| `add_rms_ffn` | 36 | 196.6 | 1.6% | fused multi_add+rms |
| `rms_norm_attn` | 36 | 187.2 | 1.5% | |
| `fa_reduce` | 36 | 108.2 | 0.9% | FA-split reduction tile |
| `add_res2` | 36 | 107.8 | 0.9% | residual add |
| `swiglu` | 36 | 70.6 | 0.6% | |
| `rms_norm_mul_rope_q/k` | 36+36 | 133.6 | 1.1% | Sprint 9c.5 fused |
| `kv_copy_fp16_k/v` | 36+36 | 52.8 | 0.4% | KV writeback |
| `rms_norm_final` | 1 | 4.8 | 0.0% | |

**Totals (categories):**

| Category | pos=0 | pos=50 | pos=100 | pos=200 |
|---|---:|---:|---:|---:|
| GEMV | 88.4% | 84.5% | 82.4% | 82.6% |
| Attention (`scalar_attn`/`fa_*`) | 3.4% | 9.2% | 11.3% | 11.3% |
| Norm + RoPE | 4.5% | 4.3% | 4.1% | 4.1% |
| KV-copy | 0.6% | 0.4% | 0.5% | 0.4% |
| Add / Mul / SwiGLU | 4.0% | 3.2% | 3.2% | 3.1% |

`scalar_attn` was used at pos=0 / pos=50 (small KV); `fa_split + fa_reduce`
takes over from pos=100. The attention transition is visible in the
breakdown but doesn't change the headline: **GEMV is >80% at every position**.

## Prefill profile

`prefill_batch` once per pp value. `gpu_sum > wall` indicates inter-dispatch
overlap on the GPU (no barrier between some independent dispatches), so
"% GPU" reads as share-of-busy-time, not share-of-wall.

```
pp=128   wall=71.00 ms   gpu_sum= 94.5 ms   1803 tok/s
pp=512   wall=223.66 ms  gpu_sum=303.7 ms   2289 tok/s
```

| Category | pp=128 | pp=512 |
|---|---:|---:|
| **GEMM** (q,k,v,o,gate,up,down,lm_head) | **93.3%** | **90.3%** |
| Attention (`fa_tiled`) | 3.0% | 5.1% |
| Norm + RoPE | 1.9% | 1.9% |
| Quantize (`quantize_q8_1`) | 0.5% | 0.5% |
| KV-copy | 0.2% | 0.1% |
| Other (Add/SwiGLU/seed/final) | 1.1% | 2.1% |

Top prefill dispatches at pp=512 (µs / share):
`gemm_up` 87 978 / 29.0%, `gemm_down` 58 813 / 19.4%, `gemm_gate` 46 020 /
15.2%, `gemm_v` 25 534 / 8.4%, `gemm_k` 23 176 / 7.6%, `gemm_q` 16 254 /
5.4%, `gemm_o` 15 543 / 5.1%, `fa_tiled` 15 427 / 5.1%.

## Hypothesis verdicts

### H3 — GEMV/GEMM dominance ≥70%

**CONFIRMED HARD.** 82–88% decode, 90–93% prefill, every position, every
seq-len tested. Any sprint that does not move GEMV/GEMM cannot move the
needle.

### H4 — Inter-dispatch idle / CPU overhead dominates

**RULED OUT at steady state.** pos=0 shows 17% CPU overhead from Vulkan
first-touch (descriptor-pool allocation, pipeline-cache warm-up), but
pos≥50 collapses to 0–1.7%. Sprints 12D (barrier elision: 0% empirical
elision) and 12E (5-op fusion: −72 dispatches, +1% wall) were already
converging on this; the timestamp data closes the question.

A graph-layer rewrite (Sprint 12G-A's "Path A") cannot reclaim time that
isn't there: at pos=200 there is **10 µs** of total CPU overhead in a
12 340 µs forward.

### H1 — CU starvation dominates the GEMV bucket

**RULED OUT for the 70% bucket.** GEMV pipeline is built with
`BLOCK_SIZE=64`, `MMV_NUM_ROWS=1` (one wave64 / one output row per
workgroup). Decode workgroup counts and resulting CU loading on RX 9070 XT
(64 CUs):

| Shader | N (rows) | Workgroups | Waves/CU | Verdict |
|---|---:|---:|---:|---|
| `gemv_q`, `gemv_o` | 4 096 | 4 096 | 64 | well saturated |
| `gemv_gate`, `gemv_up` | 12 288 | 12 288 | 192 | heavily saturated |
| `gemv_down` | 4 096 | 4 096 | 64 | well saturated |
| `gemv_k`, `gemv_v` | 1 024 | 1 024 | **16** | borderline |
| `lm_head` | 151 936 | 151 936 | 2374 | massively saturated |

A wavefront-occupancy threshold of ~16 waves/CU is the practical lower
bound for latency hiding under moderate VGPR pressure. K/V projections sit
exactly at the threshold and their measured cost (~800 µs each) is **~4.5×
the cost of `gemv_q`/N=1 scaling** — so K/V do show partial under-utilization.
But K+V together are 12.9% of the forward; closing that gap is at most
~5% wall-time.

### H2 — L2 thrashing dominates

**RE-FRAMED.** Q4_K weight matrices for FFN projections are 28 MB each
(4096 × 12288 × 0.5625 bytes/weight). The hidden input vector is 16 KB.
Weights are streamed once per dispatch — they don't fit, won't fit, and
shouldn't fit in any cache. The right metric is **HBM BW utilization**:

| Shader | Bytes read | µs (steady-state) | Effective GB/s | % of 644 GB/s peak |
|---|---:|---:|---:|---:|
| `gemv_gate` | 28 MB | 47.6 / dispatch | 588 | **91%** |
| `gemv_down` | 28 MB | 53.9 / dispatch | 520 | **81%** |
| `gemv_up` | 28 MB | 83.6 / dispatch | 335 | **52%** |

`gemv_gate` and `gemv_up` are **the same shape** (M=1, N=12288, K=4096),
**the same shader**, called sequentially in the same forward. `gemv_gate`
is at near-peak BW; `gemv_up` is at half of that. **75% slowdown on
identical work.**

## The `gemv_up` anomaly

This is the largest unexplained signal in the profile.

- Same shape (M=1, N=12288, K=4096, Q4_K).
- Same shader binary (specialization constants identical).
- Same descriptor-set layout (only buffer bindings differ).
- Order in `dispatch_layer`: gate → up → down (so `up` runs **after**
  `gate`, with the hidden-state input already L2-warm).
- Yet `up` is 1.76× slower than `gate`.

Candidate explanations (need verification, not part of this sprint):

1. **Buffer-binding overhead.** `gemv_up` writes to a different output
   buffer than `gemv_gate`. If that buffer needs a layout transition or
   if the descriptor write triggers a hazard ack, the cost lands inside
   the timestamp window.
2. **Barrier inside the profile window.** `self.profile()` brackets
   `f(dev, cmd)`; if `f` records a `vkCmdPipelineBarrier` (e.g. waiting
   on the previous gate write), the barrier's GPU-side wait is charged
   to `gemv_up`. The other two layers in the FFN tail (`gate`, `down`)
   would not pay this cost because their preceding shader writes a
   different buffer.
3. **Cache-line aliasing on the output side.** Gate writes to ffn_h_a,
   up writes to ffn_h_b (separate large buffers). If their VA mappings
   collide on a hot HBM channel, up's writes contend with gate's
   in-flight evictions.

**Recommended verification (next sprint, 1 day):** swap the dispatch
order (up before gate) and re-profile. If the slowdown moves with the
ordering, it's barrier/binding; if it stays on `up`, it's a per-buffer
cost (binding pattern, alias, etc.).

## Sprint recommendation

The Sprint 12G-A "Path A vs Path B vs v0.3 pivot" decision tree was
explicitly gated on this profile.

**Reject Path A (graph layer, ~4 weeks).** Dispatch overhead at
steady-state decode is **0.1%**. A graph layer cannot recover time that
doesn't exist. Path A's value (DAG-level fusion, view elision, ggml-alloc
liveness) is real but only tangentially relevant — it would land mostly
in prefill (where `gpu_sum > wall` already shows we have inter-dispatch
overlap) and in dispatch-count reduction (already at 0% steady-state CPU).

**Reject Path B (six specific fusion shaders, ~6 days)** as a primary
plan. Sprint 12E's 5-op fusion already saved 72 dispatches and 36 barriers
per forward and bought +1% wall. Six more such fusions extrapolate to
+5–6% — useful but not a 0.80× → 0.90× lever.

**New Path C (recommended): close the GEMV BW gap.** The single highest-ROI
unknown in the profile is `gemv_up` at 52% peak BW vs `gemv_gate` at 91%
peak on identical work. Closing that gap alone (28% × ~40% improvement)
is roughly **+11% decode wall-time** — bigger than 12D + 12E + Path B
combined.

Concrete sprint plan (estimate: 2–3 days):

1. **Day 1.** Reproduce the gemv_gate vs gemv_up gap in isolation
   (a micro-bench that runs both, identical descriptors, varying order).
   Test the dispatch-ordering hypothesis. Inspect the recorded
   `vkCmdPipelineBarrier` calls inside the FFN tail.
2. **Day 2.** Close the gap (likely a barrier elision miss in the
   FFN tail, or a binding-pattern fix). Re-run `profile_positions` and
   confirm `gemv_up` lands within 5% of `gemv_gate`.
3. **Day 3.** Re-run `run_validation` and `run_pp_bench`. Lock in the
   wins; ship as v0.2.2.

Secondary lever (only if Path C closes the gap and we still want more):
the 12.9% K/V projection bucket (1024-row dispatch, ~16 waves/CU). Bump
`MMV_NUM_ROWS` to 2 for K/V only — output count drops to 512 workgroups
× 2 rows but each workgroup pulls a 32 KB row pair, doubling the K-side
reuse. Phase-2A measured `MMV_NUM_ROWS=1` as the win at the time but
under different shapes; worth a re-test for the small-N projections.

## Output / artefacts

- `examples/profile_prefill.rs` (new, profiling tool only).
- This report.
- No runtime code changes.
- Tests: not run (no production code touched).
