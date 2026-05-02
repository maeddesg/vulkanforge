# Sprint 15C — Async Multi-Submit Prototype: STOP at feasibility check

**Premise.** Sprint 15A's analysis identified async multi-submit /
CPU-GPU pipelining as the only remaining unfalsified candidate with
material lift potential (estimated +5–15 % decode). Sprint 15B
falsified the smaller-scope `lm_head` candidate. Sprint 15C was
scoped as a 1-day prototype to MEASURE whether
`record(CB[N+1])` can actually overlap with `GPU(CB[N])` on this
codebase. Bench-gate: decode ≥ 100 tok/s.

**Verdict.** **STOP at the Schritt 1 feasibility check.** The
prototype as scoped is infeasible in 1 day because **every
intermediate buffer in the decode forward path is shared across
tokens and read+written within the recorded CB**. Without
double-buffering each of those (~15 buffers), recording CB[N+1]
while CB[N] is still in flight on the GPU is a race condition,
not an optimisation.

The "1-day prototype" framing in the brief assumed the test
infrastructure existed; it doesn't. A genuine measurement
requires the multi-day double-buffering refactor as a
prerequisite. Per the brief's "BEI UNKLARHEITEN SOFORT STOP" rule,
no code changes shipped. This is the **12th honest stop** in the
Sprint 12-15 arc.

The good news: the **theoretical ceiling for the lever still
holds at ~110 tok/s** (Sprint 15A measured RECORD = 1 836 µs,
GPU_WAIT = 9 034 µs, sequential overhead ~80 µs; perfect overlap →
max(RECORD, GPU) + overhead = ~9 114 µs → 109.7 tok/s, +20 % over
91 baseline). Comfortably above the 100 tok/s gate. **The lever is
real; what's not real is the 1-day prototype.**

## 1. Feasibility matrix — per-resource sharing analysis

Walked through `forward_token` (lines 874-953) and the buffer
declarations (lines 379-405). Every buffer that the recorded CB
touches:

| Buffer (line) | Memory location | Used in CB | CPU writes between tokens? | Pipeline-overlap blocker? |
|---|---|---|---|---|
| **scratch_a** (379) | `CpuToGpu` (host-visible) | Read by layer 0, also written by layer's first dispatch chain | **YES** (`write_bytes(embedding)` at line 892) | **❌ HARD BLOCKER** — CPU can't write while GPU reads |
| **scratch_b** (380) | `GpuOnly` | Read+written across layers (ping-pong with scratch_a) | No | **❌ BLOCKER** — CB[N+1]'s first layer overwrites scratch_b before CB[N]'s last layer is done reading it |
| **hidden_norm** (381) | `GpuOnly` | Per-layer scratch | No | **❌ BLOCKER** — same per-CB write/read |
| **q_buf, k_buf, v_buf** (382-384) | `GpuOnly` | Per-layer scratch | No | **❌ BLOCKER** |
| **attn_out** (385) | `GpuOnly` | Per-layer scratch | No | **❌ BLOCKER** |
| **o_buf, res1** (386-387) | `GpuOnly` | Per-layer scratch | No | **❌ BLOCKER** |
| **gate_buf, up_buf** (388-389) | `GpuOnly` | Per-layer FFN scratch | No | **❌ BLOCKER** |
| **ffn_hidden, ffn_out** (390-391) | `GpuOnly` | Per-layer FFN scratch | No | **❌ BLOCKER** |
| **logits_buf** (392) | `GpuToCpu` | Final output, read on host post-fence | No | ✅ Disjoint use — readback happens after wait |
| **rope_pos_buf** (402) | `CpuToGpu` | Read in RoPE | **YES** (`write_bytes(position)` at line 894) | **❌ HARD BLOCKER** — same as scratch_a |
| **kv_cache_k, kv_cache_v** | `GpuOnly` | Layer-N writes slot N; layer-N+1 reads slots 0..N then writes N+1 | No | ⚠️ GPU-side serial (timeline semaphore expressible) — recording can overlap |
| **fuse0, fuse1** (393-394) | `GpuOnly` | Fusion-flag dummies | No | ✅ Read-only by GEMV, never the source of a race |
| **Weight buffers** (model.tensors) | `GpuOnly` | Read-only across all forwards | No | ✅ Fully shareable |

**Total per-forward intermediate buffers that need double-buffering
to permit `record(N+1)` overlap with `GPU(N)`: ~14** (everything in
the rows marked BLOCKER).

The KV-cache row is interesting: `record(CB[N+1])` references the
KV-cache buffer handle; the *content* dependency (CB[N+1] reads
positions written by CB[N]) is a GPU-side ordering constraint that
**timeline semaphores** can express cleanly without blocking CPU
recording. So the KV cache is not a record-time blocker, only a
GPU-execution-order constraint.

The CpuToGpu blockers (scratch_a, rope_pos_buf) are different in
kind: they're the buffers we **CPU-write into** between tokens. If
the GPU is still reading them for the previous token, the CPU
write is a true memory race, not just a scheduling issue. These
specifically need double-buffering or a shadow staging buffer.

## 2. What the brief's "1-day prototype" actually requires

To validly measure overlap of `record(N+1)` with `GPU(N)`, the
prototype needs:

1. **Double-buffered scratch infrastructure**: at minimum
   `scratch_a × 2`, `rope_pos_buf × 2`, plus all `GpuOnly`
   intermediates listed above (since CB[N+1]'s layer-0 first
   dispatch writes scratch_b, but CB[N]'s layer-35 last dispatch
   may still be reading it). ~14 buffers × 2 = ~28 buffer
   handles.
2. **Two parallel command-buffer / fence pairs**, alternating per
   token.
3. **Two parallel descriptor-set caches** (or a unified cache that
   guarantees the descriptor sets used by CB[N+1] don't overlap
   the ones bound in CB[N]'s in-flight execution — Vulkan spec
   `VUID-vkCmdBindDescriptorSets-pDescriptorSets-00357` requires
   that bound sets stay valid until the CB completes).
4. **Per-buffer-set selection logic** in every dispatch helper in
   `forward.rs` (every `dispatch_layer*`, `run_gemv`, `run_rms_norm`,
   `run_flash_attn_*`, etc.) — they need to take a "ping/pong" index
   and use the matching scratch slot. ~20+ call sites.
5. **Timeline semaphore plumbing** for the KV-cache cross-CB
   dependency.
6. **Two-token rolling readback / sample / embedding-write loop**
   that interleaves CPU work against alternating fences without
   serialising them.

Estimated scope: **3-5 days of careful work**, plus a correctness
push (cb-reuse parity tests need to extend across multiple
in-flight tokens).

## 3. Why I am not "just trying it" with hacks

A measurement-only second-thread record (kicking off CB[N+1]'s
recording on a worker thread while the main thread waits for
CB[N]'s fence, but never actually submitting the second CB) was
considered. It would reveal whether the CPU work *can* be hidden
in principle. But:

- Vulkan command pools are single-thread per spec. We'd need a
  second pool, a second descriptor pool, second pipeline-cache
  references — the recording infrastructure shadow-replicated.
- `Forward` is `&mut self` for `dispatch_layer*`. The second
  thread can't share state. Reorganising for two-thread access is
  itself a multi-day refactor.
- Even if measurement showed perfect overlap with the second
  thread, that doesn't validate that the same overlap is
  achievable with a correctly-pipelined production change. The
  measurement would be evidence-of-no-block, not evidence-of-win.

Per the brief's principle (Sprint 12 Lektion: "IMMER messen, nie
schätzen") — the measurement is genuinely infeasible without
real infrastructure. The right move is to acknowledge that and
stop.

## 4. The theoretical ceiling still holds

Without running the prototype, we already have the upper-bound
arithmetic from Sprint 15A:

```
Sprint 15A measurement (pos=150-199 median, repeatable):
  reset:     22 µs   (per-token CB reset, can't overlap)
  begin:      1 µs   (per-token CB begin)
  RECORD: 1836 µs   (the lever — currently serial with GPU)
  end:        0 µs
  submit:    14 µs   (host-only)
  GPU_WAIT: 9034 µs  (the wall — actually serial today)
  readback:  22 µs   (post-fence)
  TOTAL:  10934 µs → 91.5 tok/s
```

Perfect overlap of RECORD with GPU_WAIT (best case):

```
  Sequential phases per token: reset + begin + end + submit + readback +
                               sample + embed_write ≈ 80 µs
  Overlapping phases:          max(RECORD, GPU_WAIT) = max(1836, 9034) = 9034 µs
  TOTAL_async ≈                80 + 9034 = 9114 µs → 109.7 tok/s
```

That is **+20 % vs the 91 tok/s baseline**, comfortably above the
100 tok/s bench-gate. The lever is real; the prototype to measure
it just requires more infrastructure than 1 day allows.

If the overlap is partial — say 80 % efficient because of
semaphore synchronisation costs or warmer-cache-only-when-gpu-idle
effects:

```
  Hidden RECORD: 0.8 × 1836 = 1469 µs hidden
  Wall reduction: 1469 µs
  TOTAL_async ≈ 10934 - 1469 = 9465 µs → 105.7 tok/s
```

Even at 80 % efficiency, we'd clear the 100 tok/s gate. The
**downside risk** is that overlap fails completely (some hidden
descriptor-pool serialisation or RADV scheduler artifact we can't
predict from the spec) and we get the 91-tok/s plateau back. But
that's a risk the multi-day infrastructure refactor would
discover, not the 1-day prototype.

## 5. Recommendation: scope Sprint 15D properly

The brief's framing ("1-day prototype") presumed we could test
the hypothesis cheaply. We can't, because the buffer layout was
designed for a single in-flight CB at a time. To unlock async
multi-submit:

- **Sprint 15D — Double-buffered intermediate buffers**: ~3-5 days
  to refactor `forward.rs` so every `GpuOnly` per-forward
  intermediate becomes a `[GpuBuffer; 2]` array, dispatch helpers
  take a slot index, descriptor caches are per-slot. This is
  pure infrastructure with no perf impact on its own — the test
  is "27/27 tests + 15/15 coherent + decode unchanged at 91
  tok/s" since we'd still be running serial.
- **Sprint 15E — Async pipelined decode loop**: ~2-3 days.
  Two-CB / two-fence rolling submit, CPU bookkeeping interleaved
  against alternating fences, KV-cache timeline-semaphore
  plumbing. Bench-gate: decode ≥ 100 tok/s.

Total scope: ~1-2 weeks of careful work. **NOT a single sprint.**

The user (next session) should decide whether to commit that
capacity. If yes: split as 15D/15E above. If no: v0.2.4 stays
the production release; v0.3 holds at "infrastructure pending"
until budgeted.

## 6. The honest pattern

Sprints 15A, 15B, and 15C have all stopped early at pre-check
or feasibility:

- **15A**: hypothesis (template CB-reuse via UBO) falsified at
  source-reading — llama.cpp doesn't do that.
- **15B**: hypothesis (lm_head NUM_ROWS=2 / coopmat) falsified
  at quant-type lookup — output.weight is Q6_K, already at
  94 % HBM ceiling.
- **15C** (this report): hypothesis can't be tested in 1 day
  because the prototype itself requires infrastructure that
  takes ~1 week to land.

**Each stop has been progressively expensive in elapsed time but
progressively *cheaper* in risk-adjusted terms** — better to find
out at minute 5 (15B's quant lookup) than minute 5 hours into a
push_constant→UBO refactor that wouldn't have helped.

The decode-gap analysis ledger now has **12 entries**, eleven of
which are "tested and falsified" and the twelfth (this) is
"infrastructure-blocked, ceiling theoretically calculable at
+20 %". This is what comprehensive optimisation work looks like
at the v0.2 → v0.3 boundary on this hardware.

## 7. What this sprint contributes

- **Per-buffer feasibility analysis** for async multi-submit
  (Section 1): exactly which buffers block the overlap and
  exactly what the fix shape is.
- **Refactor scope estimate** for Sprint 15D / 15E (Section 2),
  with concrete LOC areas (every `dispatch_*` helper in
  `forward.rs`).
- **Theoretical ceiling computation** (Section 4): +20 %
  decode is the real ceiling on this hardware with this
  infrastructure stack; we now know what success looks like
  before committing to the build.
- **A clean stop signal** for the v0.2.x release line. v0.2.4 is
  the production release; v0.3 isn't another shader sprint
  away — it's a deliberate, scoped, multi-day infrastructure
  reset.

## 8. Outputs

- This report (`results/v030_sprint15c_async_prototype.md`).
- **No code changes.** v0.2.4 binary unchanged. 27 / 27 lib
  tests, 15 / 15 coherent.
- The Sprint 15D / 15E follow-on plan is captured in Section 5;
  next session can use that directly as a brief.
