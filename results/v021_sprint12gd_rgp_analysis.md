# Sprint 12G-D — `gemv_up` anomaly analysis: it's a measurement artifact

**Goal.** Sprint 12G-C concluded that `gemv_up` runs at 52% peak HBM BW
while `gemv_gate` (identical shape, identical shader, sequential call)
runs at 91% peak — a **75% slowdown on identical work** — and named
this the highest-ROI lever in the forward (~+10% decode if closed).

**Result.** **The gap is a `vkCmdWriteTimestamp` measurement artifact,
not a real GPU inefficiency.** A single 1-line dispatch-order swap
makes the slowdown migrate to whichever GEMV runs second. Both
projections actually run at ~91% peak HBM BW. The 12G-C "Path C"
recommendation is invalid; v0.2.1 has no remaining FFN BW lever.

## Methodology

Sprint 12G-D was authorised as **profiling + analysis only, no committed
runtime code**. Steps performed (in order, cheapest first):

1. **Source-code analysis** — read `dispatch_layer` to verify whether
   any barrier or descriptor work lands inside the `self.profile()`
   window or between `gemv_gate` and `gemv_up`.
2. **RGP capture** — install/configure RADV thread-trace via
   `MESA_VK_TRACE=rgp`, produce `.rgp` artefacts for the GUI.
3. **Dispatch-order swap experiment** — temporarily reorder
   `gate → up` to `up → gate` in `forward.rs`, re-run
   `profile_positions`, observe which dispatch reports the high cost.
   **Reverted before commit.**

Step 1 already pointed at the artifact. Step 3 confirmed it decisively;
the RGP captures from step 2 are kept on disk as backup but not needed
to call the verdict.

## Step 1 — source-code analysis

`forward.rs:1797`:

```rust
fn profile<F>(&mut self, name: &str, dev: &VulkanDevice, cmd: vk::CommandBuffer, f: F)
where F: FnOnce(&VulkanDevice, vk::CommandBuffer)
{
    let token = self.profiler.as_mut()
        .map(|p| p.begin(&dev.device, cmd, name.to_string()));   // TOP_OF_PIPE
    f(dev, cmd);
    if let (Some(p), Some(t)) = (self.profiler.as_mut(), token) {
        p.end(&dev.device, cmd, t);                              // BOTTOM_OF_PIPE
    }
}
```

`f()` for a GEMV records exactly four commands:
`cmd_bind_pipeline` + `cmd_bind_descriptor_sets` + `cmd_push_constants`
+ `cmd_dispatch`. **No barrier inside the profile window.**

`forward.rs:1503-1518` — the FFN gate/up pair:

```rust
self.maybe_compute_barrier(dev, cmd, &[hidden_norm]);           // before gate

self.run_gemv(.., wg, hidden_norm, gate_buf, .., "gemv_gate");  // gate
self.run_gemv(.., wu, hidden_norm, up_buf,   .., "gemv_up");    // up
self.mark_written(&[gate_buf, up_buf]);
self.maybe_compute_barrier(dev, cmd, &[gate_buf, up_buf]);      // after up
```

**There is no barrier between `gemv_gate` and `gemv_up`.** Both read the
same `hidden_norm` input, write to disjoint output buffers (`gate_buf`,
`up_buf`) that are not consumed until SwiGLU after the second barrier.
This is correct (RAW-independent dispatches don't need a barrier between
them).

But it's also the source of the artifact: with no intervening barrier,
`vkCmdWriteTimestamp(TOP_OF_PIPE)` for the second dispatch fires at
**command-processor queue time** (very early on the AMD GFX CP), while
its `BOTTOM_OF_PIPE` fires **after both dispatches' shader work
completes** (since they execute in submission order on the same
compute queue). The reported time for the second dispatch becomes:

```
second_measured = (CP queue time of second TOP)
                  to (shader engine completion of second BOTTOM)
                ≈ first_exec_time + second_exec_time
```

For the FFN pair specifically:

```
gate_measured = gate_exec_time                       ≈ 47.6 µs
up_measured   = gate_exec_time + up_exec_time        ≈ 47.6 + 36 µs ≈ 83.6 µs
```

This matches the Sprint 12G-C numbers exactly. Time to falsify or
confirm by experiment.

## Step 2 — RGP captures

RGP CLI is installed (`/usr/bin/RadeonGPUProfiler`, package
`radeon-gpu-profiler 2.6.1-2`) but is a Qt GUI; no batch CLI for
disassembly / event-timing extraction. RADV thread-trace works via
Mesa env vars:

```fish
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=1 \
RADV_THREAD_TRACE_BUFFER_SIZE=536870912 \
  cargo run --release --example rgp_capture
```

Two captures kept on disk for later GUI inspection:

| File | Size | Submit |
|---|---:|---|
| `/tmp/rgp_capture_2026.04.30_14.22.25.rgp` | 11 MB | first prefill submit (small buffer; later submits failed) |
| `/tmp/rgp_capture_2026.04.30_14.22.38.rgp` | 14 MB | second prefill submit |

The 161-MB-per-submit captures from the 512-MB-buffer run were deleted
to save disk space — they are reproducible from the env-var invocation
above. The verdict came from step 3, so detailed RGP-GUI timeline /
occupancy / cache-stats screenshots were not needed.

## Step 3 — dispatch-order swap experiment (decisive)

Temporary edit (reverted before commit):

```rust
// before:
self.run_gemv(.., wg, hidden_norm, gate_buf, .., "gemv_gate");
self.run_gemv(.., wu, hidden_norm, up_buf,   .., "gemv_up");

// experiment:
self.run_gemv(.., wu, hidden_norm, up_buf,   .., "gemv_up");
self.run_gemv(.., wg, hidden_norm, gate_buf, .., "gemv_gate");
```

Re-ran `cargo run --release --example profile_positions`.

### Result — gate ↔ up timings flip

| Decode pos | Order | `gemv_gate` (µs sum / dispatch avg) | `gemv_up` (µs sum / dispatch avg) |
|---|---|---:|---:|
| 50 | gate→up (orig) | 1714 / **47.6** | 3013 / **83.6** |
| 50 | up→gate (swap) | **3018** / **83.8** | **1689** / **46.9** |
| 100 | gate→up (orig) | 1723 / 47.9 | 3017 / 83.8 |
| 100 | up→gate (swap) | **3030** / **84.2** | **1703** / **47.3** |
| 200 | gate→up (orig) | 1712 / 47.6 | 3010 / 83.6 |
| 200 | up→gate (swap) | **3013** / **83.7** | **1690** / **46.9** |

**The 75% slowdown follows the position, not the buffer.** It is not
a property of `gemv_up`; it is a property of "the second GEMV in a
barrier-less pair".

### Wall-time also moved (slightly)

| pos | gate→up wall | up→gate wall | Δ |
|---|---:|---:|---:|
| 50 | 12.03 ms | **11.74 ms** | −2.4% |
| 100 | 12.69 ms | **12.23 ms** | −3.6% |
| 200 | 12.34 ms | **12.03 ms** | −2.5% |

The swap is ~3% faster wall-time. This is real (consistent across three
positions) but small. Plausible explanations: HBM row-buffer state
preferring `up`'s weight matrix layout going first, or descriptor-pool
ordering. Worth keeping in mind, **but is not a 10% lever and would
need its own investigation before shipping** — and crucially, it is
*not* what Sprint 12G-C identified.

## Verdict

**The 75% `gemv_up` slowdown reported in Sprint 12G-C is a
measurement artifact of `vkCmdWriteTimestamp` semantics on RADV
when two RAW-independent dispatches run back-to-back without a
barrier.** It is not a real performance gap. Both `gemv_gate` and
`gemv_up` actually execute in ~47 µs/dispatch, both at ~91% peak
HBM BW — i.e. **already near the theoretical ceiling**.

### What this invalidates from Sprint 12G-C

- **Path C (close the `gemv_up` gap, ~+10% decode wall): INVALID.**
  There is no gap to close. The "biggest single lever" doesn't exist.
- The Sprint 12G-C BW-utilization table is misleading: `gemv_up` at
  "52%" in 12G-C is actually `gate_exec + up_exec` measured against
  `up`'s data volume alone. Treating it as `up_exec / 28 MB` was the
  error.

### What this doesn't change

- **GEMV/GEMM dominance ≥ 70% of forward time** (12G-C H3): still
  true. GEMV is the bucket; we just can't extract more from `gemv_up`.
- **CPU/dispatch overhead at steady-state decode is ~0%** (12G-C H4):
  still true. Path A (graph layer) still rejected.
- **K/V projections at ~16 waves/CU border on under-utilization**
  (12G-C, secondary lever): still true, still small (~5% of forward).

### Sprint 12 cumulative reality check

| | Decode tok/s | Wall (forward) | Δ from baseline |
|---|---:|---:|---:|
| v0.2.0 baseline | ~90 | 11.1 ms | — |
| v0.2.1 final (12F) | 91.5 | 10.9 ms | +1.7% |
| v0.2.1 + Path B (extrapolated) | ~95 | ~10.5 ms | ~+5% |
| **Theoretical ceiling** (forward = sum of real exec times at near-peak BW) | ~100 | ~10 ms | ~+11% |

The remaining 0.20× gap to llama.cpp at decode is **not** going to
come from the FFN GEMVs. They're already at the HBM ceiling. The gap
must live elsewhere — most plausibly in attention (`scalar_attn` and
`fa_split`/`fa_reduce` together are ~11% of decode at pos≥100) or in
the `lm_head` GEMV (5–6% of decode). Or it's structural and won't
close in v0.2.x.

## Lessons / process improvements

1. **`vkCmdWriteTimestamp` per-dispatch profiling is not a valid
   side-by-side comparison between RAW-independent dispatches when
   no barrier separates them.** The second dispatch's timestamp is
   inflated. To get clean per-dispatch GPU timings on RADV, either:
   - insert a `vkCmdPipelineBarrier` between every profiled pair
     (artificial; perturbs perf), OR
   - subtract the predecessor's time before reporting "second
     dispatch" cost (post-hoc correction; only works for known pairs),
     OR
   - profile each dispatch in its own `vkQueueSubmit` (most
     accurate; large overhead).
2. **Always run a swap-or-shuffle experiment before naming a single
   dispatch as "the lever".** A 1-line edit invalidated Sprint 12G-C's
   headline finding in 5 minutes.
3. Sprint 12G-C's "$\text{BW} = \text{bytes}/\text{measured µs}$"
   formula only works when the µs represents that dispatch alone.
   The sub-50% BW for `gemv_up` was an arithmetic consequence of the
   inflated denominator, nothing physical.

## Output / artefacts

- Two `.rgp` captures kept at `/tmp/rgp_capture_2026.04.30_14.22.{25,38}.rgp`
  for optional GUI inspection.
- This report.
- **No runtime code changes** (the dispatch-order swap was reverted;
  `git diff src/` is empty before commit).
