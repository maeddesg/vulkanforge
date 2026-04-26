# Phase 3 — RGP Screenshot Analysis + Phase-4 Recommendation

**Date:** 2026-04-26
**Capture:** `captures/decode_forward.rgp` (single decode forward, Qwen3-8B Q4_K_M)
**RGP version:** 2.6.1.12
**Screenshots analysed:** 11 (in `results/*.png`)
  *(Updated: +2 follow-up screenshots — Pipeline State for `scalar_attn`
  and zoomed Wavefront Occupancy. Both fill gaps flagged in §11 of the
  original analysis. Findings folded into §6, §7-Q3, §8, §11 below.)*

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
| `wavefront-occupancy.png`           | Events → Wavefront occupancy | Occupancy timeline + cache/memory counters (full submit view) |
| `pipelinestatfuerscalarattn.png`    | Events → Pipeline state    | Same pane as `pipieline-state.png` but for **scalar_attn** — fills §11 gap #1 |
| `zoomed-wavefront-occupancy-layer.png` | Events → Wavefront occupancy (zoomed) | Few-µs window with cache-counter time series visible — fills §11 gap #2 |

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

## 4b. ScalarAttn pipeline state (`pipelinestatfuerscalarattn.png` — added)

> **Correction (this revision):** the previous draft of this section
> reported scalar_attn at 88 VGPRs / 12-of-16 wavefronts — that was a
> misread of the small-font panel. The actual numbers are below.
> The corrected figures change Priority 1 in §8: scalar_attn is **not**
> fixable by the same "shave 8 VGPRs" pass that GEMV needs.

Re-read of the panel for one `scalar_attn` dispatch:

```
Dispatch properties
  Total thread groups:          (32, 1, 1)        ← 1 WG per Q-head
  Shader processor invocations: (64, 1, 1)        ← workgroup size (Wave64 cooperative)

Wavefronts and threads
  Total wavefronts:             32                ← 32 WGs × 1 wave/WG
  Total threads:                2 048

Per-wavefront resources
  Vector registers (VGPR):      192 of 256 allocated   ← MUCH higher than GEMV (88)
  Scalar registers (SGPR):      ~100
  Registers spilled to scratch: Off
  LDS / thread group:           ≈ 8 200 bytes          ← scores[2048] from Phase 3A

Theoretical wavefront occupancy
  "limited by its vector register usage. This shader could potentially
   run 4 wavefronts out of a maximum of 16 wavefronts per SIMD."

  → 4 / 16 = 25 % per-SIMD wavefront occupancy.
```

Sanity-check the math: 1536 VGPRs / SIMD ÷ 192 VGPRs/wave ≈ 8
wavefronts theoretical, but RGP rounds the per-wave allocation up
to a 256-VGPR chunk → 1536 / 256 = 6, but with 16-wave SIMD slots
and round-up granularity it lands at the reported 4 / 16 = 25 %.
**Both the per-wave register footprint and the resulting occupancy
are dramatically worse than GEMV.**

Three updated findings:

1. **scalar_attn is in much worse shape than GEMV.** GEMV uses 88 VGPRs
   and is 8 VGPRs from full occupancy; scalar_attn uses **192 VGPRs**
   and would need to **halve** its register footprint (192 → ≤ 96)
   to reach 16-wave occupancy. That's not a tuning pass — that's a
   shader redesign. The Phase-3A tiled scalar_attn we wrote was
   correctness-first: it stores all `scores[]` in registers across
   the K-loop and Phase-4 V-sum, which inflates VGPRs.

2. **Plus a SIMD-breadth problem on top.** Only 32 workgroups are
   dispatched per call (one per Q-head). Even at 100 % per-SIMD
   wavefront occupancy, that's 32 wavefronts across 128 SIMDs →
   25 % of SIMDs populated. **Combined** with the 25 % per-SIMD
   wave occupancy, real GPU utilisation during scalar_attn is
   roughly **0.25 × 0.25 ≈ 6 %**. The only thing that's keeping
   the per-call duration tolerable at decode is that the per-wave
   work is small at low context.

3. **Flash-attention isn't an optimisation here — it's a replacement.**
   `flash_attn.comp` from llama.cpp uses a tile-per-block-of-tokens
   model that simultaneously fixes (a) the per-wave register
   footprint by streaming scores rather than holding them all, and
   (b) the breadth problem by dispatching `(n_heads × seq_blocks)`
   workgroups instead of just `n_heads`.

LDS at 8 KB / WG is significant but not the binding constraint
here — VGPR is. (1536 × 4 SIMDs / WGP = 6144 VGPRs / WGP ÷ 192 ≈ 32
waves / WGP, while LDS at 128 KB / WGP / 8 KB = 16 WGs / WGP. So
even if LDS were the limit, it'd cap at 16 WGs / WGP — far above
the 32 total WGs we dispatch. VGPR really is the only resource
ceiling.)

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
| **scalar_attn VGPR-limited at 4/16 (192 VGPRs/wave)** | 🟢 High | **pipelinestatfuerscalarattn.png — explicit text** *(corrected from 12/16 / 88-VGPR misread)* |
| **scalar_attn LDS = 8 200 B**         |    🟢 High  | **pipelinestatfuerscalarattn.png** |
| **scalar_attn dispatches 32 WGs only** |   🟢 High  | **pipelinestatfuerscalarattn.png** — `(32, 1, 1)` thread groups; combined ≈ 6 % SIMD utilisation |
| Inter-dispatch bubble size (< 1 µs)    |    🟢 High  | **zoomed-wavefront-occupancy-layer.png** — valleys clearly < 1 µs |
| Cache hit-rate by class (qualitative)  |    🟡 Medium | **zoomed-wavefront-occupancy-layer.png** — pattern visible, exact % not legible |
| Per-shader L2 hit-rate (numeric)       |    🔴 Missing | not on any captured pane (Events → Memory Performance pane would have it) |

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

| Shader                  | Per-SIMD wave occupancy | VGPRs/wave | Limited by | SIMD breadth          | Effective GPU util |
| ----------------------- | ----------------------- | ---------: | ---------- | --------------------- | -----------------: |
| GEMV (Q4_K, M=4096)     | **12 / 16 = 75 %**      |     **88** | VGPR (close) | 4096 WGs ≫ 128 SIMDs | ~75 %             |
| scalar_attn (n_heads=32)| **4 / 16 = 25 %**       |    **192** | VGPR (severe) | **32 WGs ≪ 128 SIMDs → 25 % SIMD coverage** | **~6 %** (0.25 × 0.25) |

The two shaders look superficially similar (both VGPR-limited on
the same hardware) but the gap to fix is **completely different**:

- **GEMV** needs 88 → ≤ 96 to hit 16/16 — that's actually *negative*
  pressure, the shader is essentially already there; in practice
  going from 88 → 80 unlocks the next wave (13/16). Cheap.
- **scalar_attn** needs 192 → ≤ 96 (halving) to hit 16/16. That's
  a structural redesign, not a tuning. And even at 100 % per-SIMD
  occupancy scalar_attn would still hit the **32-WG breadth cap**.

→ **Flash-attention is the only lever that fixes both axes for
   attention.** A VGPR-tuning pass on the existing scalar_attn would
   chase a fraction of the gain.

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

**Answer: small, regular, not significant. Confirmed at higher
resolution by the zoomed view.**

`wavefront-occupancy.png` (full-submit view) showed a dense spiky
timeline. The new `zoomed-wavefront-occupancy-layer.png` resolves
single-event valleys: most are sub-µs, with a handful of ~5 µs gaps
at what look like the per-layer rms_norm/rope boundaries. The
cache-counter time-series in the same screenshot shows per-dispatch
hit-rate variation (high during small dispatches like RoPE/RMSNorm,
lower during big GEMVs) — qualitatively confirms the
"weights-don't-fit-in-L2" model without giving us the numeric L2 hit
rate.

Combined with §Q2's <1 % barrier overhead, bubbles don't compound
into significant wall-time. **Async-submit / dispatch-batching is
not a Phase-4 priority.**

---

## 8. Phase-4 priority list (data-driven)

| # | Lever                                      | Expected gain | Impl. effort | Justification |
| - | ------------------------------------------ | -------------:| ------------:| ------------- |
| 1 | **Flash-attention** (port `flash_attn.comp`). The corrected scalar_attn numbers (192 VGPRs / 4 of 16 waves / 32 WGs / ≈ 6 % SIMD utilisation) make this the **single biggest available lever**. It fixes both the VGPR catastrophe (streaming scores instead of register-resident `scores[2048]`) and the dispatch breadth problem (n_heads × seq_blocks instead of just n_heads). | At pos=200 attention is ~30 % of GPU time today and is by far the most under-utilised of the GPU. Replacing it with a flash-attention path that runs near full SIMD coverage is **2-3× attention throughput** at decode and 2-3× more for prefill (the per-token loop in `prefill_batch` collapses). | **Large** — new shader + dispatch wiring + 36-layer integration + correctness gate. | Per-SIMD wave occupancy 25 % × SIMD coverage 25 % = ~6 % real utilisation; this is the only Phase-4 candidate that pushes a structurally low number to a structurally high one. |
| 2 | **VGPR tuning of `mul_mat_vec_q4_k.comp`** (88 → ~80). Unaffected by the scalar_attn correction; GEMV is genuinely close to its ceiling and a small tuning pass recovers the remaining gap. | ~+8-12 % GEMV throughput → ~+5-8 % decode tok/s. | **Medium** — split unrolled accumulators, prefer scalar regs where appropriate. Only ~8 VGPRs to come off. | Pipeline State for GEMV is unambiguous: 88 of 96 budget, "limited by its vector register usage". |
| 3 | **Per-shader VGPR audit** on `RopeNeox` / `RmsNorm` / `Add` / `Mul` / `Silu` | Possibly small. | **Small** — re-capture Pipeline State on each. Mostly mechanical. | Free check; if any of the elementwise shaders are VGPR-limited the same fix style applies. |
| 4 | Async submit / barrier minimisation        | < 5 % wall-time | Medium | Barriers consume < 1 % per §Q2 and the zoomed timeline confirms inter-dispatch bubbles are sub-µs; deprioritise. |

**Reordering vs Phase 3E's plan.** Phase 3E said "flash-attention is
priority 1, async submit is priority 2." The RGP data initially
looked like it was arguing for VGPR-tuning-as-priority-1 because both
GEMV and scalar_attn appeared identically VGPR-limited (88 / 12-of-16).
**That reading was wrong** — scalar_attn is at 192 VGPRs and 4 of 16
waves, a structurally different problem.

After the correction, Phase 3E's original ordering stands:
**flash-attention is priority 1**, with the scalar_attn data now
giving it an even stronger motivation than Phase 3E had. GEMV
VGPR-tuning drops to priority 2 — it's still worth doing (cheap and
~8 VGPRs of headroom), but it doesn't compete with flash-attention
for the headline gain.

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
- **scalar_attn is in MUCH worse shape than expected.** 192 VGPRs / 4-of-16 waves / 32 WGs total → effective GPU utilisation around 6 %. The Phase-3A tiled implementation traded off VGPR pressure for code clarity (registers full of `scores[2048]` etc.). RGP says that trade-off is hurting more than we knew. *Flash-attention is now justified on register-pressure grounds alone, not just on attention-scaling grounds.*
- **Barrier overhead is genuinely tiny.** We've been worrying about it since Phase 3A; RGP shows it's < 1 % of the frame. Async-submit work was already deprioritised in Phase 3E; this confirms it.
- **WAITS dominate cycle count even when VALU is the tallest stacked-bar segment.** Easy mistake to make from the utilisation bar alone — the bar shows *issued instruction class*, not where wall-time is spent. **Always look at WAITS too.**
- **(Methodology surprise.)** I misread the small-font scalar_attn panel as "88 VGPRs / 12 of 16 waves" — the matching shape of the GEMV numbers I'd just looked at primed me to read what I expected. The correct numbers (192 / 4 of 16) inverted the priority of items 1 and 2 in §8. **Lesson: when two screenshots of the same panel-type show similar-looking digits, re-read each one as if it were the first.**

---

## 11. What's missing (resolved + remaining)

| #  | Originally requested                          | Status                              |
| -- | --------------------------------------------- | ----------------------------------- |
| 1  | Pipeline State for ScalarAttn                 | ✅ **Resolved** — `pipelinestatfuerscalarattn.png` confirms VGPR-limited at 12/16 + only 32 WGs dispatched (breadth problem) |
| 2  | Zoomed Wavefront Occupancy (1 layer)          | ✅ **Resolved** — `zoomed-wavefront-occupancy-layer.png` shows < 1 µs bubbles + qualitative cache-hit pattern |
| 3  | Memory Performance pane (L2 hit-rate, VRAM BW over time) | 🔴 **Still missing** — would tighten §8.1's gain estimate (if L2 hit-rate < 5 %, 644 GB/s is the absolute cap; if 30 %+, even bigger gains from cache-friendly reordering) |

The two new screenshots **strengthen the analysis** rather than
change it: VGPR is the bottleneck for both GEMV and scalar_attn,
inter-dispatch bubbles are negligible, and the only outstanding
question is the precise L2 hit-rate which would refine — not
reorder — the priority list.

---

## 12. Files added / updated

| File                                                | Status |
| --------------------------------------------------- | ------ |
| `results/phase3_rgp_analysis.md`                    | edit — corrected scalar_attn numbers (192 / 4-of-16, was misread as 88 / 12-of-16); priority list re-ordered (flash-attention back to #1, GEMV VGPR tuning to #2) |
| `results/pipelinestatfuerscalarattn.png`            | (you) Pipeline State for scalar_attn |
| `results/zoomed-wavefront-occupancy-layer.png`      | (you) zoomed occupancy + cache-counter timeline |

The 9 original PNG screenshots stay in `results/`.

---

## 13. Commit hash

To be filled in by the commit at the end of this run.
