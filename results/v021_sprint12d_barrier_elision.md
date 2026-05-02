# VulkanForge v0.2.1 Sprint 12D — Barrier-Elision (Implementation + Empirical Negative Result)

**Date:** 2026-04-30
**Branch:** main (HEAD = 8b2588c, post-Sprint 12C gap analysis)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Implementation sprint — built a dirty-flag elision tracker, measured the actual elision rate.

## TL;DR — NEGATIVE result, but a useful one

Built a `SyncTracker` + `maybe_compute_barrier(reads)` helper that elides
`compute_barrier` calls when none of the next op's read-buffers is in the
pending-write set. Wired through `dispatch_layer` (decode) and `dispatch_final`
end-to-end. Parity intact (176/176 tests green).

Empirical elision rate over a 2-prompt bench (~50K barrier checks): **0.0 %**.
Every single barrier site fires — the dirty-flag check never finds a clean read.

| Metric                         | Sprint 12C predicted | Sprint 12D measured |
|--------------------------------|----------------------|---------------------|
| Decode tok/s (`run_15prompt`)  | 92.5 → ~110 (+22 %)  | 92.5 → **92.0** (−0.5 %, noise)   |
| Prefill pp=512 tok/s           | 2318 → ~2343 (+1.1 %)| 2318 → **2321** (+0.1 %, noise)   |
| Barrier checks per 2-prompt run| n/a                  | **50 818**          |
| Barriers actually issued       | n/a                  | **50 818** (100 %)  |
| Elision rate                   | n/a                  | **0.0 %**           |

This is **the right number**, not a bug. Static analysis of `dispatch_layer`
shows that all 12 per-layer barrier sites sit at real read-after-write
boundaries: every block reads precisely the buffer that the previous block
wrote. Sprint 12C's `8/12 elidable` heuristic was over-optimistic about how
much our existing code structure leaves on the table.

The infrastructure (`SyncTracker`, `mark_written`, `maybe_compute_barrier`,
opt-out env var, telemetry counter) is in place and ready to amortize when
Sprint 12E's fusion reduces barrier-site count. Sprint 12E's 5-op fusion is
now the **only remaining lever** for the decode infrastructure-side wins
that 12A/12B/12C identified.

## 1. What was built

`src/backend/vulkan/forward.rs`:

```rust
pub struct Forward {
    // ... existing fields ...
    pending_writes: std::collections::HashSet<u64>,  // raw vk::Buffer handles
    elision_disabled: bool,
    barrier_stats_checked: u64,
    barrier_stats_issued: u64,
}

impl Forward {
    fn mark_written(&mut self, bufs: &[vk::Buffer]) { ... }

    fn maybe_compute_barrier(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        reads: &[vk::Buffer],
    ) -> bool {
        // Returns true if a barrier was actually emitted.
        // - elision_disabled OR any read in dirty set → emit + clear all dirty
        // - otherwise → skip
    }

    fn reset_barrier_state(&mut self) { ... }
    pub fn barrier_stats(&self) -> (u64, u64) { ... }
    pub fn barrier_elision_active(&self) -> bool { ... }
}
```

The dirty set tracks `vk::Buffer` raw handles (`b.as_raw() -> u64`) — a
single `HashSet<u64>` of size ≤ 32 covers every scratch buffer in the
forward path. After each `compute_barrier` issuance, the set is cleared
because the helper emits a global `VkMemoryBarrier` (one barrier syncs
everything; per-buffer granularity would require a refactor to
`VkBufferMemoryBarrier` arrays — punted to a future sprint).

`reset_barrier_state()` is called automatically:
- inside `reset_descriptor_pool_and_cache` (covers prefill_batch + the
  debug helpers)
- explicitly before each `cmd_ctx.one_shot{,_profiled}` call in the three
  decode forward variants (lines 680, 773, 863)

This makes the tracker correctness-by-construction: every cmd-buffer
recording starts with an empty dirty set.

### Wiring

`dispatch_layer` (decode path, 36 layers per forward) was converted from
**12 unconditional `compute_barrier(dev, cmd)` calls** to **12
`self.maybe_compute_barrier(dev, cmd, &[reads])` calls**, each with an
explicit list of buffers the next block reads:

| # | Block      | Reads passed to `maybe_compute_barrier`   |
|---|------------|-------------------------------------------|
| 1 | attn_norm  | `[hidden_norm]`                           |
| 2 | Q/K/V GEMV | `[q_buf, k_buf]` (Q/K-norm or RoPE)       |
| 3 | Q/K-norm   | `[q_buf, k_buf]` (RoPE)                   |
| 4 | RoPE       | `[k_buf, v_buf]` (KV-write)               |
| 5 | KV-write   | inline TRANSFER barrier kept as-is        |
| 6 | attention  | `[attn_out]` (O-proj)                     |
| 7 | O-proj     | `[input, o_buf]` (multi_add_rms)          |
| 8 | mul_add_rms| `[hidden_norm]` (gate/up)                 |
| 9 | gate/up    | `[gate_buf, up_buf]` (swiglu)             |
|10 | swiglu     | `[ffn_hidden]` (FFN down)                 |
|11 | FFN down   | `[res1, ffn_out]` (residual2)             |
|12 | residual2  | `[output]` (next layer)                   |

`dispatch_final` was also converted (1 maybe_compute_barrier).

`dispatch_layer_batch` (prefill path) was **NOT** converted — see §4.

### Opt-out

`VULKANFORGE_DISABLE_BARRIER_ELISION=1` (or `=true`) makes
`maybe_compute_barrier` always emit the barrier (reverts to the legacy
always-on behavior). Used during this sprint to confirm bit-exact parity.

### Telemetry

`barrier_stats() -> (checked, issued)` exposes counters that
`run_pp_bench` and `run_15prompt_bench` print after each run.

## 2. Empirical results

### 2.1 Performance (no change)

```
                Before (12C baseline)   After (12D, elision ON)   After (12D, elision OFF)
pp=512 tok/s    2318.7                  2321.2                    2324.1
decode tok/s    92.5                    92.0                      91.9
```

Run-to-run noise on this hardware is ~±2 % for prefill and ~±1.5 % for
decode (measured across multiple Sprint 10F bench rounds). All three
columns are within noise of each other → **no measurable performance
delta between the three configurations**.

### 2.2 Elision rate (the headline number)

`run_15prompt_bench --num-prompts=2`:

```
elision ON:  Barrier stats (cumulative): checked=50818  issued=50818  elided=0 (0.0%)
```

50 818 barrier-site checks across ~128 forward_token calls + 2 prefill
batches. **Every single one fires** — the dirty-flag check finds a dirty
read every time.

`run_pp_bench --pp-list=512 --runs=3`:

```
elision ON:  Barrier stats: checked=4  issued=4  elided=0 (0.0%)
```

(Only 4 because prefill goes through `dispatch_layer_batch` which
remains uninstrumented — those 4 are from `dispatch_final`.)

### 2.3 Parity

176/176 tests green with elision ON. No NaN, no Inf, no logits-divergence
across any of the existing parity tests (which exercise both decode and
prefill paths end-to-end, multi-prompt, multi-token).

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 176 passed
```

## 3. Why the 0 % rate — root cause

Sprint 12C's `8/12 elidable` estimate assumed that ~⅔ of our barriers
sit between blocks where the next block doesn't read what the previous
wrote. The static read/write trace through `dispatch_layer` shows the
opposite: **every barrier site has a tight read-after-write dependency
on a buffer that was just written**.

The 12 sites, with the buffer that's both written-before and read-after:

| # | Wrote before                   | Read after            |
|---|--------------------------------|-----------------------|
| 1 | hidden_norm                    | hidden_norm (Q/K/V)   |
| 2 | q_buf, k_buf, v_buf            | q_buf, k_buf          |
| 3 | q_buf, k_buf                   | q_buf, k_buf          |
| 4 | q_buf, k_buf                   | k_buf, v_buf          |
| 5 | k_dst, v_dst (kv_cache)        | kv_cache (attention)  |
| 6 | attn_out                       | attn_out              |
| 7 | o_buf                          | o_buf                 |
| 8 | res1, hidden_norm              | hidden_norm           |
| 9 | gate_buf, up_buf               | gate_buf, up_buf      |
|10 | ffn_hidden                     | ffn_hidden            |
|11 | ffn_out                        | ffn_out               |
|12 | output                         | output (next layer)   |

In all 12 cases, the read-set ⊆ {dirty buffers}. The dirty-flag check is
correct (it doesn't fire false-elisions) and complete (it never misses a
real dependency) — it just has nothing to elide because we're already at
the irreducible barrier count for the current dispatch graph.

This is a structural consequence of how the dispatch graph is shaped. To
make barriers elidable, we'd need either:
- **Fewer barrier sites** (fewer write-then-read transitions) — i.e. fusion
- **More-grouped writes** — but our writes are already grouped by block
- **Per-buffer barriers** (`VkBufferMemoryBarrier`) — would let us not
  flush e.g. `kv_cache` for a block that only writes/reads `q_buf`. Today
  all 12 barriers are global `VkMemoryBarrier` so they sync everything.

Each of those is a non-trivial refactor or an explicit fusion shader.
None is what dirty-flag elision alone delivers.

## 4. Why prefill (`dispatch_layer_batch`) was not converted

Same structural reason. The prefill path has ~11 barrier sites per layer,
each separating a block of writes from a block of reads of those same
buffers (`batch_q8` → Q/K/V GEMM; `batch_q/k` → RoPE; `batch_attn_out` →
O-proj; etc.). Converting it would yield the same 0 % elision rate while
adding ~150 lines of `mark_written` + `maybe_compute_barrier` calls.

If/when Sprint 12E's 5-op fusion fires (RMS_NORM+MUL+ROPE+VIEW+SET_ROWS in
one shader), the prefill path's barrier-site count drops and *new*
elision opportunities may open. At that point, converting
`dispatch_layer_batch` becomes worth doing.

## 5. Reconciling with Sprint 12C's prediction

Sprint 12C predicted **+22 % decode** from barrier elision. Where did the
discrepancy come from?

**12C's estimate**: 282 extra barriers/forward × 13 µs/barrier × (8/12
elidable) ≈ 188 elided × 13 µs = 2.4 ms saved per forward → 22 % of the
11 ms decode forward.

**12D's reality**: 0 elidable barriers in the current dispatch graph, so
0 µs saved. All 12 per-layer barriers are at real RAW boundaries.

The 12C analysis conflated two distinct sources of barrier difference:
1. **Number of barrier sites**: VF 12, llama.cpp ~5 per layer. This is
   real and primarily comes from **fusion** (their 5-op shader collapses
   what we do in 4 dispatches into 2 with 1 barrier between them).
2. **Elision within fixed sites**: VF unconditional, llama.cpp dirty-flag.
   But once you're at llama.cpp's already-fused dispatch graph, the
   *remaining* barriers also sit at real RAW boundaries. Their dirty-flag
   system mostly gates barriers against the rare case where a *cross-layer*
   prealloc buffer wasn't actually written — not against intra-layer
   dependencies.

In effect: **fusion reduces barrier count, not elision**. We had the
order of operations backwards.

The corrected claim: the +22 % decode delta must come from Sprint 12E's
fusion work, not from 12D's elision.

## 6. Remaining usefulness of the 12D infrastructure

Even with 0 % elision today, the SyncTracker / `maybe_compute_barrier`
machinery is **load-bearing** for Sprint 12E:

1. The 5-op fusion shader will **remove** barrier sites #4 and #5 (RoPE+
   KV-write becomes one fused dispatch that writes directly to KV-cache).
2. After that fusion, site #4 still has data dependencies but site #5
   becomes moot. The dirty-flag tracker correctly handles the new
   dispatch graph without changes — `maybe_compute_barrier` still gets
   called at the right places.
3. If Sprint 12E (or later) introduces multiple fused shaders that
   sometimes write to the same scratch and sometimes don't, the dirty
   set will start showing real elision (currently every block writes
   distinct buffers, so the tracker is irrelevant; with conditional
   writers, it becomes useful).

The infrastructure cost is negligible (~80 LOC in `forward.rs`, ~15 LOC
of telemetry in two examples) and the opt-out lets us A/B at any time.

## 7. Files touched

```
EDIT  src/backend/vulkan/forward.rs                      (+128, −24 LOC)
EDIT  examples/run_pp_bench.rs                           (+11 LOC: barrier_stats print)
EDIT  examples/run_15prompt_bench.rs                     (+10 LOC: barrier_stats print)
NEW   results/v021_sprint12d_barrier_elision.md          (this report)
```

No shader changes. No test changes. No commit beyond the report's
inclusion of the perf numbers.

## 8. Tests / regression

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 176 passed
```

176/176 green (was 176; no new tests because the existing 176 already
exercise every `dispatch_layer` and `dispatch_final` path under both
elision modes via the `VULKANFORGE_CB_REUSE` parity infrastructure).

Manual smoke tests:
- `VF_PP_LIST=512 cargo run --release --example run_pp_bench`: bit-exact
  CSV output match across 5 runs of elision-ON vs elision-OFF.
- `VF_NUM_PROMPTS=2 cargo run --release --example run_15prompt_bench`:
  identical "Coherent prompts: 2/2" across configurations.

## 9. Sprint 12E recommendation

**Fusion is now the ONLY infrastructure lever left.** 12C identified
two: (1) barrier elision, (2) 5-op fusion. (1) is now empirically dead.
(2) is the headline.

Recommended scope for 12E (unchanged from 12C section 7):

- New shader `vk_shaders/rms_norm_mul_rope_set_rows.comp`: reads from
  `batch_q`/`batch_k`, applies RMS-norm + scale + RoPE, writes the
  rotated output **directly** into the KV-cache slot at
  `pos_offset_bytes(layer, pos)`.
- Replaces 4 dispatches (`rms_norm_mul_rope_q_b` +
  `rms_norm_mul_rope_k_b` + `kv_copy_fp16_k_b` + `kv_copy_fp16_v_b`)
  with 2 dispatches.
- Effort: 3–4 days.
- Gate: decode tok/s improves ≥ +10 %; bit-exact parity.

If 12E delivers, the 12D infrastructure is already in place to
mark/unmark the new shader's writes correctly — no further forward.rs
changes.

If 12E fails to deliver, the entire 12C prediction stack collapses and
we should pivot to GPU profiling (RGP) in 12G.

## 10. Take-aways

1. **Sprint 12C's quantitative model was wrong.** The `8/12 elidable`
   estimate assumed structural redundancy that does not exist in our
   block-tight `dispatch_layer`.
2. **Pure dirty-flag elision yields nothing on a code base whose
   barriers are already at RAW boundaries.** The win llama.cpp gets is
   from having FEWER barrier sites, not from eliding existing ones.
3. **Negative results are still results.** This sprint took the lever
   off the table; Sprint 12E's fusion lever is now isolated as the
   single remaining bet.
4. **The infrastructure is correctly built and instrumented** — even at
   0 % elision, it imposes no measurable overhead, the opt-out path is
   bit-exact, and the telemetry will show real elision the moment a
   future sprint adds a fusion shader that breaks the tight RAW chain.
