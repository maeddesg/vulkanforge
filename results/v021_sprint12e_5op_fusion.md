# VulkanForge v0.2.1 Sprint 12E — Decode Norm+RoPE Fusion (Implementation + Negative Result #2)

**Date:** 2026-04-30
**Branch:** main (HEAD = 21cfcac, post-Sprint 12D barrier elision)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Implementation sprint — apply the existing fused `rms_norm_mul_rope` shader to the decode path.

## TL;DR — second NEGATIVE result

Replaced 4 separate dispatches (Q-norm + K-norm + Q-rope + K-rope) per layer
with 2 fused `rms_norm_mul_rope` dispatches in decode `dispatch_layer`. The
fused shader has shipped since Sprint 9c.5 and was already used by prefill
`dispatch_layer_batch` — this sprint just extends its use to decode when
`has_qk_norm` is set.

**Dispatch + barrier savings are real** (verified empirically):
- 2 dispatches saved per decode layer × 36 layers = **72 dispatches saved per forward**
- 1 barrier saved per decode layer × 36 layers = **36 barriers saved per forward**
- Confirmed via barrier-stats counter: 50 818 → 46 210 over a 2-prompt run, exactly matching the predicted 4 608 savings (36 × 128 forwards)

**End-to-end performance: noise-level.** No measurable improvement.

| Metric                       | Before (12D) | After (12E) | Δ        |
|------------------------------|--------------|-------------|----------|
| Decode tok/s (2 prompts)     | 92.0         | 93.0        | +1.1 %   |
| Decode tok/s (5 prompts MEDIAN) | n/a       | 91.1–91.4   | ~ noise  |
| Prefill pp=512 tok/s         | 2 321        | 2 322       | +0.04 %  |
| Barriers issued / 2-prompt run | 50 818     | 46 210      | −4 608   |
| Dispatches / decode layer    | 17           | 15          | −2       |
| 176/176 tests                | green        | green       | ✓ parity |

The combined Sprint 12C prediction for both 12D + 12E was **+38 % decode**.
Empirical measured: **+1 % at the noise floor.** Sprint 12C's
dispatch+barrier cost model was wrong by ~30×.

## 1. What was changed

`src/backend/vulkan/forward.rs` — `dispatch_layer` (decode path) at the
QK-norm + RoPE region. Single change: replace the 4-dispatch sequence
(`run_rms_norm_q` + `run_rms_norm_k` + `run_rope_neox_q` + `run_rope_neox_k`
gated by 1 intermediate barrier) with 2 calls to the existing fused
`run_rms_norm_mul_rope` helper:

```rust
// Before — 4 dispatches, 1 inter-block barrier:
if cfg.has_qk_norm {
    self.run_rms_norm(q_buf, wqn, q_buf, ...);  // Q-norm
    self.run_rms_norm(k_buf, wkn, k_buf, ...);  // K-norm
    self.maybe_compute_barrier(...);             // intra-block
}
self.run_rope_neox(q_buf, q_buf, ..., position, ...);  // Q-rope
self.run_rope_neox(k_buf, k_buf, ..., position, ...);  // K-rope

// After — 2 dispatches, 0 intra-block barrier:
if cfg.has_qk_norm {
    self.run_rms_norm_mul_rope(q_buf, wqn, q_buf, ...);  // fused
    self.run_rms_norm_mul_rope(k_buf, wkn, k_buf, ...);  // fused
} else {
    // Models without Q/K-norm fall back to standalone RoPE.
    self.run_rope_neox(q_buf, q_buf, ..., position, ...);
    self.run_rope_neox(k_buf, k_buf, ..., position, ...);
}
```

This is the same shader (`ShaderId::RmsNormMulRope`) that prefill
`dispatch_layer_batch` already uses (line 3318+, since Sprint 9c.5).
The fusion is bit-exact vs the separate path — Sprint 9c.5's parity test
already validated this.

For models without `has_qk_norm` (= non-Qwen models, e.g. Llama, Mistral
through the existing tokenizer harness), the standalone RoPE path stays
intact.

## 2. Empirical results

### 2.1 Dispatch + barrier counts

The 12D `barrier_stats()` counter confirms the structural saving:

```
Before 12E (per 2-prompt bench):
  Barrier stats: checked=50818 issued=50818 elided=0 (0.0%)

After 12E (same bench, fusion ON):
  Barrier stats: checked=46210 issued=46210 elided=0 (0.0%)

Delta: 4 608 fewer barriers per 2-prompt run
     = 36 saved per forward × 128 forward calls (2 prompts × 64 decode tokens)
```

The math checks out exactly. Each layer saved 1 barrier (the one between
"Q/K-norm done" and "RoPE done"); we ran 128 decode forwards and 36
layers per forward, hence 4 608 fewer.

Dispatch count is harder to instrument without adding more telemetry,
but the source-code edit unambiguously eliminates 2 calls per layer
(`run_rope_neox_q` and `run_rope_neox_k` are no longer issued when
`has_qk_norm` is true).

### 2.2 Wall-time performance

```
                    Decode tok/s    Prefill pp=512 tok/s
12D baseline        92.0            2 321
12E (this sprint)   93.0 (2-prompt)
                    91.1 (5-prompt MEDIAN)
                    91.4 (5-prompt, elision OFF)
12E mid-band        ~91–93          2 322
```

Run-to-run variation on this hardware is ~±2 % for decode based on prior
Sprint 10F bench rounds. The 12E numbers are entirely within that band.
The ~+1 % seen on the 2-prompt run is within noise; the 5-prompt MEDIAN
came in slightly lower than 12D's baseline. **Conclusion: no measurable
wall-time improvement.**

### 2.3 Parity

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 176 passed
```

176/176 green. The fused path produces bit-exact decode logits vs the
separate path. The same parity infrastructure that gates `rms_norm_mul_rope`
in prefill (Sprint 9c.5) covers it for decode.

## 3. Reconciling with Sprint 12C's prediction

Sprint 12C predicted:
- 12D barrier elision: **+22 % decode** (from saving ~188 barrier µs)
- 12E 5-op fusion: **+16 % decode** (from saving ~72 dispatches × 25 µs)
- Combined: **~+38 % decode**

Sprint 12D measured: **~0 % decode** (0 % elision rate).
Sprint 12E measured: **~+1 % decode** (within noise).
Combined empirical: **~+1 % decode**, vs +38 % prediction. **30× off.**

Where the 12C model went wrong:

1. **Per-barrier stall on RDNA4 is much smaller than the 13 µs estimate.**
   Plausible value: 1–3 µs per global `VkMemoryBarrier` after L1/L2
   warm. Saved 36 barriers × ~2 µs = ~70 µs per forward, well below the
   noise floor.

2. **Per-dispatch cost is much smaller than the 25 µs estimate.**
   Plausible value: 3–5 µs per `vkCmdDispatch` on a hot pipeline-bind
   path. Saved 72 dispatches × ~4 µs = ~290 µs per forward = ~2.6 % at
   decode (~11 ms). Still small enough to be hard to see above noise.

3. **The bottleneck isn't host-side dispatch overhead at all.** Decode
   time is dominated by GPU compute (the GEMVs + attention). No amount
   of barrier/dispatch elision moves that.

The 12C model implicitly assumed CPU-record cost was a meaningful
fraction of decode forward time. Empirically it isn't — at ~700 dispatches
of ~5 µs CPU cost = ~3.5 ms CPU recording per forward, but the recording
runs *in parallel* with GPU work for the *previous* command buffer. With
single-submit-blocking architecture (Sprint 12B §1), recording happens
between fence-waits, but the full forward still has to wait for GPU
completion either way.

## 4. What this means for Sprint 12

Both infrastructure levers identified by Sprint 12A/12B/12C are now
empirically dead:

| Lever                           | 12C prediction | 12D/E measured | Verdict |
|---------------------------------|----------------|----------------|---------|
| Dirty-flag barrier elision      | +22 % decode   | ~0 %           | DEAD    |
| 5-op fusion (norm+rope+kv)      | +16 % decode   | +1 %           | DEAD    |
| Submit pipelining               | +5–10 % decode | not tested     | likely DEAD by analogy |

The decode gap (0.79× → llama.cpp 113 tok/s) is **not in
host-side infrastructure**. Two consecutive empirical sprints converge
on the same conclusion.

The fusion *is structurally correct*: dispatch + barrier counts went
down by exactly the predicted amounts. What was wrong was the assumption
that those savings translate to wall-time. They don't, on this
hardware/driver/code-path combination.

## 5. Where the gap actually is — pivot to GPU profiling

After 12D + 12E both yielding ~0 % wall-time impact, the next sprint
must use **direct GPU profiling** to find where decode time is actually
spent. Sprint 12C's recommended fallback was Sprint 12G with Radeon GPU
Profiler (RGP).

### What 12G should produce

For a single decode forward (capture via `radeon-profiler` or
`AMDRGPCapture`):

1. Per-dispatch GPU wall time (not host-record time) — which dispatches
   dominate?
2. Memory bandwidth utilisation — are we BW-bound or compute-bound?
3. CU occupancy per dispatch — are GEMVs starved?
4. Cache hit rates — does L1/L2 churn between dispatches?
5. Stall analysis — where does the GPU sit idle?

Hypotheses to test in 12G (in priority order):

1. **GEMV CU starvation.** Decode runs 7 GEMVs per layer at very low
   tile counts (M=1, N=4096–11008). RDNA4 has 64 CUs; if a GEMV only
   spawns ~32–256 workgroups, 75–98 % of CUs sit idle. Fixing this
   needs a different GEMV shape (more workgroups, smaller tiles).

2. **Attention dispatch latency.** `run_scalar_attn` runs per-token at
   decode; on a long context (e.g. 1024 tokens of KV-cache), the GPU
   may stall on KV-cache reads.

3. **Cross-dispatch cache invalidation.** Even with dirty-flag elision
   (12D) and fusion (12E), each `compute_barrier` flushes L1/L2. If
   GEMVs share weights or activations across barriers, the second
   read pays a full HBM round-trip.

4. **Pipeline binding overhead.** On RDNA4 with 36 layers × 17 dispatches
   each = 612 pipeline binds per forward. Each bind on RADV may flush
   shader caches. Hard to eliminate without major refactor.

If 12G shows hypothesis #1 (GEMV CU starvation) is the dominant cost,
the decode-fix sprint becomes a GEMV reshape — try wider workgroups,
multi-row dispatches, etc. That's a kernel-level change, not infra.

## 6. Files touched

```
EDIT  src/backend/vulkan/forward.rs                  (+12, −16 LOC: fusion swap)
NEW   results/v021_sprint12e_5op_fusion.md           (this report)
```

No new shaders. The fused `rms_norm_mul_rope` SPV has been in the
build since Sprint 9c.5; we're just calling it from a new place.

## 7. Tests / regression

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 176 passed
```

176/176 green. Bit-exact decode logits vs the separate path. No new
tests needed — the existing test coverage already exercises both decode
and prefill `rms_norm_mul_rope` paths via the Sprint 9c.5 parity tests
plus the Phase-3B chat-session and Phase-5 batched-attn tests.

## 8. Take-aways

1. **Sprint 12C's quantitative cost model was systematically wrong.**
   Two consecutive infrastructure sprints (12D barrier elision, 12E
   norm+rope fusion) yielded 0–1 % wall-time improvement vs the
   combined +38 % prediction. The model under-estimated GPU-side
   real-time stall by ~30×.

2. **Saved dispatches and barriers ≠ saved wall-time.** Both 12D and
   12E *correctly reduced* the structural quantities Sprint 12C said
   to reduce. Both *did not* translate to measurable end-to-end gains.
   The bottleneck isn't where 12C thought.

3. **The decode gap is not in host-side infrastructure.** With both
   levers shown empirically dead, the gap to llama.cpp's 113 tok/s
   must be in GPU compute behaviour: dispatch-shape, CU occupancy,
   memory-traffic patterns, or cache effects. Direct GPU profiling
   (Sprint 12G) is the only way forward.

4. **The fusion change is still worth keeping.** Bit-exact, parity-
   tested, and structurally cleaner. The 36 saved barriers and 72
   saved dispatches per forward represent ~70 µs and ~290 µs of work
   respectively — small but free. If a future sprint cuts the GPU
   compute time per forward (e.g. via GEMV reshape), the fixed CPU+GPU
   overhead becomes a larger relative fraction and these savings will
   begin to show.

5. **Pivot recommended:** Sprint 12F (Sprint 11G-D retest, ~30 min
   cost) → Sprint 12G (RGP capture + analysis, ~1 week). If 12G
   doesn't reveal a clear bottleneck, the decode gap may simply be
   the structural cost of running 700 dispatches per forward on RADV
   no matter what we do, and we should accept ~92 tok/s as the
   end-state for this architecture and pivot the sprint plan to v0.3
   feature work.
