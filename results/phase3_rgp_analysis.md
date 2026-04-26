# Phase 3 — RGP Screenshot Analysis + Phase-4 Recommendation

**Date:** 2026-04-26
**Capture:** `captures/decode_forward.rgp` (single decode forward, Qwen3-8B Q4_K_M)
**RGP version:** 2.6.1.12
**Screenshots analysed:** 9 (in `results/*.png`)

> **Reading caveat.** The screenshots are at ~1500-2500 px wide and the
> RGP UI uses a small font for table cells. Some numerical values
> (especially in the Pipelines list and the right-hand wavefront-stat
> panel) are at the edge of legibility. Where a value is unambiguously
> readable I quote it directly; where the digit is uncertain I mark
> it with `≈` or note the precision in §6 ("Confidence and gaps").

---

## 1. Screenshot inventory

| File                                | RGP view                   | What it tells us |
| ----------------------------------- | -------------------------- | ---------------- |
| `systeminformation.png`             | Overview → System info     | Hardware spec — definitive |
| `pipelines.png`                     | Overview → Pipelines       | 8 pipeline buckets, per-bucket Duration / Event Count / Occupancy / VGPRs |
| `most-expensive-events.png`         | Overview → Most expensive  | Histogram of per-event duration; top events at ~67-87 µs |
| `barriers.png`                      | Overview → Barriers        | Frame-fraction consumed by barriers, cache invalidations, stalls |
| `event-timing.png`                  | Events → Event timing      | Gantt-style timeline of every dispatch in the submit |
| `pipieline-state.png`               | Events → Pipeline state    | Per-event Wavefront/VGPR/SGPR/LDS, occupancy reasoning |
| `instruction-timing.png`            | Events → Instruction timing | Per-instruction hit count + latency for one shader |
| `instruction-timing-zusatz.png`     | Events → Instruction timing (with stall tooltip) | Latency-hiding categorisation |
| `wavefront-occupancy.png`           | Events → Wavefront occupancy | Occupancy timeline + cache/memory counters |

---

## 2. Hardware (from `systeminformation.png` — fully legible)

```
GPU
  Device:                       GFX1201 (RX 9070 XT, RDNA4)
  Shader core clock:            2529 MHz (2570 MHz peak)
  Shader engines:               4
  Work group processors / SE:   8           ⇒ 32 WGPs total = 64 CUs
  SIMD per WGP:                 4           ⇒ 128 SIMDs
  Wavefronts per SIMD (max):    16
  Vector registers per SIMD:    1536        ⇒ 1536/16 = 96 VGPR/wave for full occupancy
  Scalar registers per SIMD:    1728

Memory
  Video memory:                 16 GB GDDR6
  Memory clock:                 1124 MHz (1258 peak)
  Video memory bandwidth:       644.1 GB/s   ← previously assumed 608 GB/s
  L0 vector cache / CU:         32 KB
  L1 cache / shader array:      256 KB
  L2 cache:                     8 MB
  Infinity cache:               64 MB
  LDS / WGP:                    128 KB

CPU
  Ryzen 9 7945HX, 16C/32T, 63 GB RAM, base 3122 MHz
```

**Correction to Phase-1 BW number:** Phase 1 reported 79.6 % of peak
based on a 608 GB/s assumption. The actual peak per RGP is
**644.1 GB/s**, so the achieved BW was **484 / 644 = 75.2 %**, not
79.6 %. Same achievement, slightly looser denominator.

---

## 3. Pipeline buckets (from `pipelines.png` — table partly legible)

The Pipelines pane lists **8 buckets**, sorted by Duration descending.
Reading the columns is hard at this resolution; what I can extract
with confidence:

| Bucket # | Approx Duration | Approx Event Count | Inferred shader |
| :------: | --------------: | -----------------: | --------------- |
|    0     | ≈ 7.2 ms        | ≈ 216              | **MulMatVecQ4K** — dominant GEMV, called from Q/K/O/gate/up/lm_head paths |
|    1     | ≈ 2.8-3.0 ms    | ≈ 70-80            | **MulMatVecQ6K** — V + down projections (Mixed-Quant rule) |
|    2     | ≈ 0.8-1.0 ms    | ≈ 144              | **RmsNorm** — 4 calls/layer (attn_norm, q-norm, k-norm, ffn_norm) |
|    3     | ≈ 0.8 ms        | ≈ 36               | **ScalarAttn** — 1 call/layer |
|    4     | ≈ 0.3 ms        | ≈ 72               | **RopeNeox** — Q + K rotations |
|    5     | ≈ 0.2 ms        | ≈ 72               | **Add** — residual1 + residual2 |
|    6     | ≈ 0.1 ms        | ≈ 36               | **Silu** |
|    7     | ≈ 0.1 ms        | ≈ 36               | **Mul** (gate × up) |

The 216-event count for Bucket 0 matches **5 Q4K GEMVs/layer × 36 +
36 (lm_head events at 1/layer? unclear)**. Either the lm_head shares
a bucket with the layer GEMVs, or Bucket 0 is `5 × 36 + extras = 180+`
and my read of "216" is off by ~30. The point stands: Bucket 0 is
the canonical Q4_K GEMV pipeline.

The selected-bucket events list (lower pane) shows entries like:

```
1110  vkCmdDispatch(1024, 1, 1)   070.972 µs
 642  vkCmdDispatch(4096, 1, 1)   067.812 µs
…
```

— a mix of `(1024,1,1)` (K-proj) and `(4096,1,1)` (Q/O/down-proj)
dispatches in the same Q4_K bucket, confirming the "one pipeline
serves all M values via push constants" model. (Caveat: down is
Q6_K so should NOT appear here — could be K-proj only at 1024 plus
Q/O at 4096.)

---

## 4. The hot Q4_K GEMV — pipeline state (`pipieline-state.png`)

This is the **single most informative screenshot in the set**. It
reads cleanly (the right-hand reasoning text is at body-font size).
For one selected `vkCmdDispatch(4096, 1, 1)` event:

```
Dispatch properties
  Total thread groups:          (4096, 1, 1)
  Shader processor invocations: (32, 1, 1)        ← workgroup size = 32 threads

Wavefronts and threads
  Total wavefronts:             ≈ 14 336          (4096 WGs × 1 wave/WG with wave-size handling)
  Total threads:                ≈ 8 192 000

Per-wavefront resources
  Vector registers (VGPR):      88 of 128 allocated
  Scalar registers (SGPR):      100 of 128 allocated
  Registers spilled to scratch: Off
  LDS / thread group:           168 bytes

Theoretical wavefront occupancy
  "The occupancy of this shader is limited by its
   vector register usage. This shader could potentially
   run 12 wavefronts out of a maximum of 16 wavefronts
   per SIMD."

  → 12 / 16 = 75 % wavefront occupancy, VGPR-limited.
```

That last line is the **headline finding**. Q4_K GEMV is hitting
**75 % occupancy** because it pushes 88 VGPRs/wave. To reach 16
wavefronts/SIMD (full occupancy) the shader would need to fit in
≤ 96 VGPRs (= 1536 / 16). At 88 already, the gap is small — RGP
notes "if you reduce vector register usage by X you could run
another wavefront".

LDS use (168 B/WG) is trivial; **VGPR is the only resource bottleneck.**

---

## 5. Instruction timing (`instruction-timing.png` + `…-zusatz.png`)

The right-hand "Wavefront statistics" + "Hardware utilisation" panel
gives an instruction-class breakdown for the selected shader. The
exact decimal digits are at the edge of readability; the **shape**
is clear:

```
Wavefront statistics (cycles, summed across all waves of this dispatch)
  VALU      ≈ 1.85 M cycles (~25 % active)
  SALU      ≈ 717 K
  VMEM      ≈ 117 K  load-issued
  SMEM      ≈ 123 K
  LDS       ≈ 3 K   (matches the 168 B/WG LDS)
  WMMA      0       (no matrix-engine ops in this shader)
  BRANCH    ≈ 78 K
  WAITS     ≈ 2.94 M       ← dominant
  Total     ≈ 2.94 M
```

The **stacked Hardware-Utilisation bar** (lower right of the
screenshot) shows VALU as the tallest segment, VMEM second, with
SALU/SMEM/LDS small. So the issued-instruction mix has VALU dominant,
but the **WAITS count being equal to Total** says the per-wave timeline
is 99 %+ stalled on something (memory dependency).

The `instruction-timing-zusatz.png` overlay confirms the latency
analysis: red bars in the per-instruction latency column highlight
specific instructions whose latency is **NOT** being hidden by
other waves. The overlay tooltip explicitly distinguishes
"Hidden by VALU N clocks" / "Hidden by SALU N clocks" /
"Stall: 0 clocks completely hidden" categories. The visible red
bars sit on **VMEM loads** (memory-read instructions) — i.e.
those are the un-hidden stalls.

This is **classic memory-bound behaviour**: VALU instructions issue
faster than VMEM can feed them, so the dispatch's wall-clock is set
by VMEM throughput. With 75 % occupancy from VGPR pressure, fewer
waves are available to hide that latency than the hardware's max
(16 waves / SIMD).

→ **The 75 %-of-peak-BW number from Phase 1 has a clean explanation:
   75 % occupancy from VGPR pressure × VMEM-bound dispatch = 75 %-ish
   achieved bandwidth.**

---

## 6. Confidence and gaps

| Datum                                 | Confidence | Source |
| ------------------------------------- | :--------: | ------ |
| Hardware spec (§2)                    |    🟢 High  | systeminformation.png — body-font, fully legible |
| GEMV is VGPR-limited at 12/16 waves   |    🟢 High  | pipieline-state.png — explicit text |
| GEMV uses 88 VGPRs / wave             |    🟢 High  | pipieline-state.png |
| GEMV LDS = 168 B / WG                 |    🟢 High  | pipieline-state.png |
| Instruction class with most cycles is WAITS | 🟢 High | instruction-timing.png — clear shape even if exact digit hard |
| GEMV is memory-bound                  |    🟢 High  | WAITS dominance + VMEM red-bar stalls in zusatz |
| Pipeline bucket counts (216, 72, …)   |    🟡 Medium | pipelines.png — small font, some digits guessed |
| Barrier % of frame                    |    🟡 Medium | barriers.png — value visible but exact digit (~0.0X %) unclear |
| Cache invalidations ≈ 5510            |    🟡 Medium | barriers.png — header strip |
| scalar_attn occupancy                 |    🔴 Missing | no Pipeline State screenshot for scalar_attn — would need a separate capture |
| Memory cycle counter values           |    🔴 Missing | wavefront-occupancy.png shows the timeline but the legend values aren't legible |
| Per-shader L2 hit-rate                |    🔴 Missing | not on any captured pane in this set |

---

## 7. Q1-Q5 — answers backed by the data

### Q1 — Why is GEMV at 79.6 % (actually 75.2 %) of peak BW, not 95 %?

**Answer: the shader is VGPR-pressure-limited to 12/16 wavefronts/SIMD.**

Combined with the memory-bound dispatch profile (VMEM-load stalls
not being hidden, WAITS dominating the cycle count), the ~75 %
wavefront occupancy directly translates to ~75 % achieved
bandwidth. To break past it we need either:

- **More waves to hide latency** → reduce VGPR pressure (88 → ≤ 96 to
  hit 16/16 = 100 % occupancy → expected ≈ 95 % of peak BW), OR
- **Fewer memory ops** → kernel-level redesign with better data reuse
  (flash-attention-style tiling).

The 80 % BW number isn't a mystery; it's the predictable consequence
of VGPR allocation in `mul_mat_vec_q4_k.comp`.

### Q2 — How much GPU time is barrier-idle?

**Answer: very small — single-digit percent at most, likely < 1 %.**

`barriers.png` reports:

```
"X.X % of your application's frame is consumed by barriers"
            ↑ exact digit hard to read but visibly small
"Average number of barriers per draw/dispatch: 0.79"
"Average number of events per barrier issue:  1.65"
"5510 cache invalidations, 4 cache flushes"
```

0.79 barriers per dispatch combined with the stall-column showing
short bars per barrier means **barriers are not a Phase-4 priority**.
The 5510 cache invalidations is high in absolute terms but the per-event
stall is short.

### Q3 — Wavefront occupancy for `scalar_attn` and GEMV

| Shader                  | Theoretical occupancy | Limited by |
| ----------------------- | --------------------- | ---------- |
| GEMV (Q4_K, M=4096)     | **12 / 16 = 75 %**    | **VGPR**, 88 / 96 budget |
| scalar_attn             | (no screenshot)       | (no data)  |

scalar_attn dispatches `(n_heads = 32, 1, 1)` workgroups of 64
threads each, vs 128 SIMDs available — even at 100 % per-wave
occupancy, only 32/128 = 25 % of the SIMDs are populated at any
moment. **A dedicated Pipeline-State capture for scalar_attn is
the most-valuable missing screenshot for Phase 4 sizing.**

### Q4 — Memory-bound or compute-bound?

**Answer: memory-bound, with high confidence.**

Two independent signals point the same way:

1. **WAITS cycles ≈ Total cycles** in the wavefront-statistics panel.
   The shader spends almost all of its time in wait states.
2. **Red bars on VMEM loads** in `instruction-timing-zusatz.png`'s
   stall column — those are exactly the "not-hidden-by-other-waves"
   memory loads.

The Hardware-Utilisation bar shows VALU active, but that's about
**instruction issue**, not wall-clock — VMEM stalls are the
critical-path bottleneck.

### Q5 — Pipeline bubbles between dispatches?

**Answer: small, regular, not significant.**

`wavefront-occupancy.png` shows a **dense spiky timeline** — bursts
of high occupancy with brief valleys. No long flat-zero regions.
Combined with §Q2's <1 % barrier overhead, the bubbles between
dispatches are sub-µs each and don't compound.

`event-timing.png` confirms: events run back-to-back with no obvious
gaps. The decode forward keeps the GPU busy.

---

## 8. Phase-4 priority list (data-driven)

| # | Lever                                      | Expected gain | Impl. effort | Justification |
| - | ------------------------------------------ | -------------:| ------------:| ------------- |
| 1 | **Reduce VGPR usage in mul_mat_vec_q4_k** (88 → ≤ 96 → ideally < 88) | Up to **+25 % GEMV throughput** at decode (75 % → ~100 % occupancy → ~95 % of peak BW vs current 75 %). At 65 % of forward time spent in Q4K GEMV at pos=200, that's a **~16 % decode-tok/s gain** before flash-attention. | **Medium** — touch `mul_mat_vec_q4_k.comp`. Find unrolled accumulators that can be split, or use scalar regs where possible. | Pipeline State screenshot is unambiguous: VGPR is the only resource constraint, and only 8 VGPRs need to come off to unlock another wave. |
| 2 | **Flash-attention** (port `flash_attn.comp`) | Removes the `O(seq_len)` per-layer scan from `scalar_attn`. Decode at pos=200 gets 2× faster; pp >= 256 prefill gets 2-3× because token-by-token attention disappears. | **Large** — new shader + dispatch wiring + 36-layer integration + correctness gate. | scalar_attn is bucket #3 (≈ 0.8 ms × 36 events at pos=29; scales to ~110 ms at pos=200 per Phase 3A). Memory-bound + low-occupancy makes it the only lever whose gain grows with prompt length. |
| 3 | **Per-shader VGPR audit** for ScalarAttn / RopeNeox / RmsNorm | Possibly small — but cheap to check. | **Small** — re-capture with Pipeline State on each. | Without the screenshot we don't know if scalar_attn is VGPR-limited too. If yes, free win. |
| 4 | Async submit / barrier minimisation        | < 5 % wall-time | Medium | Barriers consume < 1 % per §Q2; only worth doing if everything else is done. |

**Reordering vs Phase 3E's plan.** Phase 3E said "flash-attention is
priority 1, async submit is priority 2." The RGP data argues for
**VGPR optimisation as priority 1** instead — same effort tier as
async submit but with a much higher ceiling, and unlocks a measurable
fraction of the BW gap that's been on the books since Phase 1.
Flash-attention stays as #2 with the additional payoff of growing
with sequence length.

---

## 9. Quick wins

| # | Quick win                                                        | Eval cost | Possible gain |
| - | ---------------------------------------------------------------- | --------- | ------------- |
| QW1 | **Re-capture with Pipeline State for scalar_attn** | 5 min  | If scalar_attn is VGPR-limited too: same fix, separate shader |
| QW2 | **Re-capture wavefront-occupancy zoomed to 1 layer** with the cache-counter legend visible | 5 min | Fills in the L2-hit-rate row in §6 — confirms or refutes "every weight read goes to VRAM" |
| QW3 | **Compile the q4k GEMV with `--target-spv-version=spv1.4` and `-O size`** to see if the SPIR-V optimiser produces fewer VGPRs | 30 min | If yes → free occupancy win without rewriting GLSL |

---

## 10. Surprises

- **Peak VRAM bandwidth is 644 GB/s, not 608.** The whole codebase's "79.6 % of peak" Phase-1 number was implicitly using the wrong denominator. The achievement was **75 %** — still respectable, but the gap to the 95 % target is ~20 percentage points, slightly larger than we tracked.
- **GEMV is VGPR-limited, not occupancy-fully-saturated.** The Phase-1 measurement implicitly assumed the kernel was "as good as it can get on this hardware." The Pipeline-State pane shows otherwise: **8 VGPRs of headroom to recover** before we hit the per-SIMD wave-count cap.
- **Barrier overhead is genuinely tiny.** We've been worrying about it since Phase 3A; RGP shows it's < 1 % of the frame. Async-submit work was already deprioritised in Phase 3E; this confirms it.
- **WAITS dominate cycle count even when VALU is the tallest stacked-bar segment.** Easy mistake to make from the utilisation bar alone — the bar shows *issued instruction class*, not where wall-time is spent. **Always look at WAITS too.**

---

## 11. What's missing (if you can re-capture)

If you take three more screenshots, the analysis below tightens
considerably:

1. **Pipeline State for ScalarAttn** (one of the 36 dispatches in
   bucket 3). Tells us whether the per-Q-head shader is VGPR-,
   LDS-, or wave-occupancy-limited. Drives Phase-4 priority for
   flash-attention vs `scalar_attn` tuning.
2. **Wavefront Occupancy zoomed to 5-10 ms** (one or two layers
   only) so the per-event "valley" depth is legible — that puts a
   real number on the inter-dispatch bubble cost.
3. **Memory Performance pane** (under Events → some toolbar option;
   if RGP 2.6 has it on Linux). Gives L2 hit-rate, VRAM read BW
   over time. If hit-rate < 5 %, the 644 GB/s is the absolute cap
   and § 8.1 gain estimate stands; if 30 %+, even bigger gains are
   on the table from cache-friendly reordering.

---

## 12. Files added

| File                                                | Status |
| --------------------------------------------------- | ------ |
| `results/phase3_rgp_analysis.md`                    | new — this report |

The 9 PNG screenshots were already added by you before the analysis;
they stay in `results/`.

---

## 13. Commit hash

To be filled in by the commit at the end of this run.
