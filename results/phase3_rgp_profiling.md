# Phase 3 — RGP Deep-Profiling (Pre-Phase-4 Hardware Analysis)

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Mesa:** `26.0.5-arch2.4`
**RGP:** `/opt/radeon-gpu-profiler/RadeonGPUProfiler` v2.6.1
**Capture:** `captures/decode_forward.rgp` (144 MB, one full decode forward)

> **Status: Capture infrastructure landed, GUI analysis pending.**
> The .rgp file is produced and ready to open. The numerical values
> in §3-§9 below need to be filled in by opening the capture in
> Radeon GPU Profiler — that part of the work cannot be done from
> the CLI. §10 has the step-by-step instructions for what to look
> at and which boxes to fill.

---

## 1. Capture setup

### 1.1 Toolchain verified

- ✅ `mesa 26.0.5-arch2.4` (≥ 25.1, RGP support present on Linux)
- ✅ `RadeonGPUProfiler` 2.6.1 installed at `/opt/radeon-gpu-profiler/`
- ✅ Mesa env vars discovered via `strings libvulkan_radeon.so | grep TRACE`:
  - `MESA_VK_TRACE=rgp`
  - `MESA_VK_TRACE_PER_SUBMIT=1` (compute-only equivalent of frame trigger)
  - `RADV_THREAD_TRACE_BUFFER_SIZE` (in **bytes**, not MiB — `512*1024*1024 = 536870912`)

### 1.2 Capture binary

`examples/rgp_capture.rs` — minimal forward-pass driver. Workload:

1. Load Qwen3-8B Q4_K_M
2. Tokenise the chat-template form of `"Explain what a mutex is in one sentence."` (29 tokens)
3. Token-by-token GEMV prefill (NOT `prefill_batch` — Phase 4 targets the decode hot path)
4. `DECODE_TOKENS = 1` (single forward pass; RADV writes ~144 MB per submit at the 512 MiB SQTT buffer size, so a single decode keeps the file count manageable)
5. Clean shutdown

### 1.3 Capture command (used)

```sh
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=1 \
RADV_THREAD_TRACE_BUFFER_SIZE=536870912 \
cargo run --release --example rgp_capture
```

Output (RADV stderr, abridged):

```
radv: Thread trace support is enabled
  (initial buffer size: 32 MiB → grown to 512 MiB,
   instruction timing: enabled,
   cache counters: enabled,
   queue events: enabled).
…
RGP capture saved to '/tmp/rgp_capture_2026.04.26_14.23.36.rgp'
[rgp_capture] prefill done: 29 tokens in 2692.1 ms
[rgp_capture] decode done: 1 steps in 92.6 ms
[rgp_capture] clean exit
```

The 2.7 s prefill / 92 ms decode timings are **inflated by SQTT
overhead** (RADV docs warn ~5-20 % perf cost when tracing is
on). Use these numbers only for relative analysis, not for
performance regression checks.

### 1.4 Canonical capture file

```
captures/decode_forward.rgp        144 MB
```

This is one specific forward pass (1 prefill submit at the timestamp
`14.23.36`). All decode forwards in our pipeline take the same shape
(36 layer dispatches × ~18 sub-shaders + final norm + LM head), so
any of the captured submits is representative.

### 1.5 Validation layer

Validation layer was **enabled** during the capture (the existing
`device.rs` always enables it when available). Phase-3-stable
correctness (48/48 tests) means we don't need it for this run, but
the SQTT-on-validation-on overhead doesn't change the *relative*
shape of the capture. If the absolute numbers feel off later, a
non-validation re-capture can be done by gating
`device.rs::VALIDATION_LAYER` behind an env var.

---

## 2. What the capture answers (questions from the prompt)

| #  | Question                                                  | Answer source in RGP                            | Status |
| -- | --------------------------------------------------------- | ----------------------------------------------- | ------ |
| Q1 | Why is GEMV at 79.6 % peak BW, not 95 %?                 | Per-shader Memory Counter pane (L2 hit, VRAM BW)| 🔲 GUI |
| Q2 | How much GPU time is barrier-idle?                        | Wavefront Occupancy timeline + Events pane     | 🔲 GUI |
| Q3 | What's `scalar_attn` wavefront occupancy?                 | Per-event Occupancy column                      | 🔲 GUI |
| Q4 | Is GEMV memory- or compute-bound?                         | Instruction Timing — VMEM vs VALU ratio         | 🔲 GUI |
| Q5 | Pipeline bubbles between 650 dispatches?                  | System Activity timeline — idle gaps            | 🔲 GUI |

---

## 3. Capture-level overview (TODO from RGP GUI)

Open `captures/decode_forward.rgp` in `RadeonGPUProfiler`, **Frame Summary** view:

```
GPU Active time:         _____ ms
Number of dispatches:    ___ (expected ~650 = 36 layers × 18 + finals)
Number of barriers:      ___ (expected ~660 — barrier per dispatch + extras)
Total idle time:         _____ ms
Idle time fraction:      __ %
```

---

## 4. Per-shader breakdown — top 5 (TODO)

Open the **Events** pane, sort by Duration, descending:

| Shader        | Calls | Duration (µs) | Wavefronts | Occupancy | VGPRs/wave | LDS bytes |
| ------------- | ----: | ------------: | ---------: | --------: | ---------: | --------: |
| `gemv_up`     |    36 |               |            |           |            |           |
| `gemv_down`   |    36 |               |            |           |            |           |
| `gemv_gate`   |    36 |               |            |           |            |           |
| `scalar_attn` |    36 |               |            |           |            |           |
| `lm_head`     |     1 |               |            |           |            |           |

Cross-check against `results/phase3a_profile_run.log`:

- ShaderProfiler reported `gemv_up = 3.5 ms / 36 calls = 97 µs each`. RGP per-call duration should be close (within SQTT overhead).
- `scalar_attn` ShaderProfiler-time scales with position; the captured forward is at pos=29, so expect ~1.5 ms total / 36 calls = ~42 µs per layer.

---

## 5. Instruction Timing — `gemv_q4k` (TODO)

In the **Pipeline State** or **Instruction Timing** view for one
`gemv_up` instance:

| Instruction class | Cycles | % of shader time |
| ----------------- | -----: | ---------------: |
| VMEM (memory read) |        |                  |
| VALU (compute)     |        |                  |
| SALU (scalar)      |        |                  |
| LDS (shared)       |        |                  |
| Wait / stall       |        |                  |

Phase-4 implication:

- **`VMEM/VALU > 2:1`** → memory-bound, Phase-1's 79.6 % BW number is the cap. Spec-tuning won't help further; **flash-attention** gives the next 2× by amortising the same memory bandwidth across more compute.
- **`VMEM/VALU < 1:1`** → compute-bound, Phase-1's BW measurement was a coincidence; targeted ALU optimisation (different tiling, vec4 loads) becomes the priority.

---

## 6. Memory counters (TODO)

In the **Memory Performance** pane:

| Metric                      | Measured | Phase-1 baseline / expected |
| --------------------------- | -------: | --------------------------: |
| L2 cache hit rate           |       %  | < 50 % (4.68 GB weights ≫ L2) |
| VRAM read BW (peak GB/s)    |          | 608 GB/s peak               |
| VRAM read BW (actual GB/s)  |          | ≈ 484 (Phase-1 = 79.6 % of peak) |
| L2→shader BW                |          | (driver-dependent reference) |

If L2 hit-rate < 5 %: every weight read goes to VRAM, our 79.6 %
BW number is the achievable cap. If 30-50 %: some reuse is
happening, we could chase it further.

---

## 7. Wavefront occupancy timeline (TODO)

In the **Wavefront Occupancy** pane:

- Average occupancy across the whole submit: ___ %
- Lowest sustained occupancy: ___ % during which shader: ___
- Visible idle gaps (timeline at 0 occupancy):
  - Number of gaps: ___
  - Total gap time: ___ µs
  - Gap pattern: regular (between every dispatch?) or clustered?

Expected:

- `scalar_attn` should be ~50 % occupancy (32 workgroups across 64 CUs).
- GEMV should be near-100 % during compute, dipping at barriers.
- Idle gaps between dispatches should be < 1 µs each. If they're
  > 5 µs, our `compute_barrier()` is over-conservative.

---

## 8. Barrier analysis (TODO)

In the **Events** pane filtered for `vkCmdPipelineBarrier`:

| Metric                       | Measured |
| ---------------------------- | -------: |
| Barrier count this submit    |          |
| Total barrier-wait time      |          |
| Barrier wait as % of submit  |          |
| Worst single barrier         |          |

Cross-check vs ShaderProfiler "Dispatch overhead" (Phase 3C: 2.5-3.0 ms typical at pos=200). If RGP shows similar barrier-time, the overhead is real driver work, not Rust-side cost.

---

## 9. Phase-4 priority recommendation (TODO)

Fill in from §3-§8 findings, then pick a path:

```
[ ] Memory-bound + minimal barrier idle
    → Phase 4 path A: Flash-attention first (kills the only
      variable-cost factor at long context). Async-submit second.

[ ] Memory-bound + significant barrier idle (> 5 % submit time)
    → Phase 4 path B: Barrier minimisation in parallel with
      flash-attention.

[ ] Compute-bound (surprise)
    → Phase 4 path C: Re-evaluate spec-tuning. Spec-tuning got
      4-5 % but maybe wasn't aggressive enough on TM/TN/WMITER.

[ ] Occupancy-limited (any major shader < 30 %)
    → Phase 4 path D: VGPR-budget audit. Try shader variants
      with less register pressure.
```

---

## 10. How to actually fill in the GUI placeholders

This part requires opening the `.rgp` in the Radeon GPU Profiler GUI.
Step-by-step:

1. **Open the capture**

   ```sh
   /opt/radeon-gpu-profiler/RadeonGPUProfiler captures/decode_forward.rgp
   ```

   First load takes ~10-30 s for a 144 MB capture. RGP indexes the
   trace into its in-memory model.

2. **Verify the capture is intact**

   - Welcome / Summary screen should show: GPU model
     (gfx1201 / Radeon RX 9070 XT), driver (RADV 26.0.5), API (Vulkan),
     and a non-zero "GPU active" duration.
   - If the screen says "incomplete trace": SQTT buffer overflowed
     during capture. Re-run §1.3 with a larger
     `RADV_THREAD_TRACE_BUFFER_SIZE` (e.g. `1073741824` = 1 GiB).

3. **Top-5 shader breakdown** (fills §4)

   Navigate: **Events** view (left sidebar). Sort the `Duration`
   column descending. Read the top 5 rows; for each, the
   columns include `Wavefronts`, `VGPRs`, `LDS`, `Occupancy`.

4. **Instruction Timing** (fills §5)

   Click on one `gemv_up` event in the Events pane → right-click →
   "Show in Instruction Timing" (or use the toolbar icon). RGP
   shows a stacked column with VMEM/VALU/SALU/LDS/wait cycles.

5. **Memory Counters** (fills §6)

   Pane: **Memory Performance** (left sidebar). Cards titled
   "L2 Cache Hit", "VRAM Read", "L2→Shader Read". Read the
   averages from the card's value field.

6. **Occupancy timeline** (fills §7)

   Pane: **Wavefront Occupancy** (left sidebar) — large
   timeline along the top. The Y-axis is occupancy 0-100 %, X
   axis is time. Hover individual valleys to see what shader
   is running.

7. **Barrier analysis** (fills §8)

   Events pane → filter type = `vkCmdPipelineBarrier`. Sum the
   `Duration` column for the totals. The "barrier wait time"
   is what RGP labels — it's the GPU stall, not the
   `vkCmdPipelineBarrier` host-side call.

8. **Save findings to this file**

   The TODO blocks above are tagged with `🔲 GUI`. As you fill them
   in, replace the empty cells with the measured values. The
   recommendation in §9 should follow naturally from §4-§8 once
   the numbers are in.

---

## 11. What stays the same regardless of GUI findings

- The capture works (~144 MB, 0 "Failed to capture" warnings).
- All 48 regression tests still pass — no code change in this
  phase.
- The capture is reproducible with the env-var combination in §1.3.

---

## 12. Files added

| File                                               | Status |
| -------------------------------------------------- | ------ |
| `examples/rgp_capture.rs`                          | new — minimal forward-pass capture driver |
| `captures/decode_forward.rgp`                      | new — 144 MB SQTT trace |
| `results/phase3_rgp_profiling.md`                  | new — this report (skeleton) |

**Untouched:** the entire `src/` tree, every shader. RGP profiling
is observe-only.

---

## 13. Commit hash

To be filled in by the commit at the end of this run.
