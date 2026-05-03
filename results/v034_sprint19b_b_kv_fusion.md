# Sprint 19B-B — KV-write fusion (honest negative)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 19B-A, commit `670afbd`)
**Goal:** Decide whether to fuse `kv_copy_*` / `kv_store_*` dispatches
into the existing pipeline (Option D from the brief: combine K + V
into a single dispatch, ~120 LOC). Per Sprint 19B's plan, gate is
pp=512 ≥ +2%.

## Outcome

**Honest negative.** No code shipped. Sprint 12D's lesson reaffirmed:
KV-write dispatch overhead is below the measurement noise floor on
RDNA4 / RADV.

## Pre-check (mandatory before writing any code)

The brief itself contains the relevant arithmetic:

> Bei pp=512: total ~260ms → 216µs = 0.08% → NOISE!
> Bei Decode: total ~9ms → 216µs = 2.4% → MÖGLICH aber Sprint 12D = 0%!

Sprint 12D had already demonstrated that *barrier* elimination on
the decode path = 0%. KV-write fusion is the same physics — eliminating
dispatch boundaries on a path already saturated by GPU work. The
brief's own pessimistic estimate is closer to ground truth than the
optimistic +2% target.

## Empirical upper-bound measurement

Rather than spend a day building the K + V combined dispatch shader,
I added a diagnostic env-var (`VULKANFORGE_SKIP_KV_WRITE_DISPATCH=1`)
that **completely skips** the K/V cache writes (output goes garbage
but timing is preserved). This measures the *upper bound* of what any
KV-write fusion could possibly save.

### Prefill (Q4_K_M, multi-submit on, 5 runs each)

| pp | KV writes ENABLED | KV writes SKIPPED | Δ |
|---:|---:|---:|---:|
|  64 | 1796 tok/s | 1780 tok/s | **−0.9%** |
| 128 | 2707 tok/s | 2687 tok/s | **−0.7%** |
| 256 | 3650 tok/s | 3589 tok/s | **−1.7%** |
| 512 | 3886 tok/s | 3901 tok/s | **+0.4%** |
|1024 | 3749 tok/s | 3788 tok/s | **+1.0%** |

Skipping KV writes is *slower* than keeping them at pp=64-256 — pure
run-to-run noise. The pp=512-1024 "wins" of +0.4-1.0% are within typical
±0.5-1% bench noise we routinely observe across runs. **The actual
upper bound is indistinguishable from zero.**

### Decode (Q4_K_M, 5 runs each)

| Mode | tok/s |
|---|---:|
| KV writes enabled (baseline) | 117.4 |
| KV writes skipped (upper bound) | 117.5 |

**Δ = +0.1%** = noise.

## Why it's noise

The KV-write dispatch reads K (or V) from `batch_k` / `k_buf`
(seq_len × n_kv_heads × head_dim FP32 = 512 × 1024 × 4 B = 2 MiB at
pp=512) and writes the converted FP16 output to the cache. At pp=512
that's a real workload — ~4 µs of actual compute, not just dispatch
overhead. Combining it with the V write into one dispatch wouldn't
eliminate the work — it would just overlap two ~4 µs operations into
one ~6-8 µs operation. The wave scheduler already overlaps adjacent
RAW-independent dispatches on RDNA4 (per Sprint 12D + the per-dispatch
timestamp memory rule), so the savings are theoretical, not measurable.

At decode (n_elements = 1024 = 4 KiB), the dispatches are pure
overhead — ~2-3 µs each, ~144 µs total per token across 36 layers.
But that's still 1.6% of a 9 ms decode step on paper, and **zero**
in practice because the next dispatch (flash_attn) takes ~50 µs and
masks the entire KV-write tail through queue-pipelining inside the
GPU.

## Decision matrix

| Option | LOC | Expected gain | Real gain | Verdict |
|---|---:|---:|---:|---|
| D — fuse K+V into one dispatch shader | ~120 | brief: +2% pp=512 | upper bound ≤ 1% pp=512 | **skip** |
| A — fuse K-write into RoPE shader | ~80 | half of D (K only, not V) | ≤ 0.5% | **skip** |
| B — fuse into rms_norm_mul_rope | ~80 | same as A | ≤ 0.5% | **skip** |
| E — vkCmdCopyBuffer (transfer instead of compute) | ~50 | doesn't apply: FP16/FP8 need conversion | n/a | **skip** |

None of the four options meet the +2% pp=512 gate. Per the
"VulkanForge coding-agent workflow" memory rule, when a sub-sprint
is below its bench gate, the right move is an honest-negative report
and skip — not "ship it anyway because we wrote the code."

## What this confirms

1. **Sprint 12D's lesson generalises.** Whenever a proposed optimization
   is "fewer dispatches" or "fewer barriers" on the decode/prefill hot
   path, the prior should be 0% measured impact unless we have a
   specific reason to expect otherwise (e.g. Sprint 19B-A: the lever
   was *recording-overlap*, not dispatch-count).

2. **Sprint 19B's plan was right to demote 19B-B to "maybe."** The
   plan ranked 19B-A (multi-submit) as the largest measurable lever,
   19B-B (KV-write fusion) as targeted but smaller, and 19B-C
   (graph-aware barriers) as a likely re-test of 12D. 19B-A delivered
   +5-8% at small pp; 19B-B is below noise. Order matters.

3. **The remaining 8-10% prefill gap to llama.cpp is not in
   dispatch-count optimizations** — at least not in the K/V-write
   region. Future investigation should focus on:
   * The kernel-side gap (small-K coopmat variants, tile shapes —
     where Sprint 19A's +57% lived).
   * Pipeline cache hits / specialization-constant invalidation.
   * The actual GPU-time profile (rgp / radv perfdoc) to find which
     individual dispatches still sit on the critical path at pp=512.

## What was changed

The diagnostic env-var (`VULKANFORGE_SKIP_KV_WRITE_DISPATCH`) used to
gather these numbers was added temporarily, then **reverted** because
shipping a "produces garbage output" debug knob is a foot-gun. The
working tree is clean — no code changes from this sprint, only this
report.

## Re-testing hooks for the future

If RADV or Mesa later changes how dispatch boundaries are scheduled
(e.g. higher overhead per `vkCmdDispatch` call, or a queue-submit
batching change), the same upper-bound experiment can be re-run by
re-applying the same skip-env-var to the two `dispatch_layer_batch`
KV-write blocks. If the gap reopens, then 19B-B becomes worth doing.

## Files touched

* `results/v034_sprint19b_b_kv_fusion.md` — this report.

(No code changes shipped.)
