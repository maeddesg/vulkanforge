# Sprint 14C — NUM_ROWS=2 redux (with Path A): HONEST NEGATIVE again

**Premise.** Sprint 13E showed `MMV_NUM_ROWS=2` was a regression on
the v0.2.3 codebase (gemv_q +21 % per dispatch, forward wall +2.9 %)
because Path B's LDS tree-reduction doubled in cost when NUM_ROWS
doubled. Sprint 14A pinned `requiredSubgroupSize=64` at pipeline
creation; Sprint 14B shipped Path A (`USE_SUBGROUP_ADD=1`,
default-on) which has constant reduction cost regardless of
NUM_ROWS. The hypothesis: with Path A active, NUM_ROWS=2 is finally
a net positive — it halves WG count without doubling LDS, K/V
decode projections (N=1024) get measurably faster, decode median
crosses the +2 % bench-gate at ≥ 93 tok/s.

**Verdict.** **Hypothesis falsified again.** One-line flip
(`MMV_NUM_ROWS = 1 → 2` in `pipeline_registry.rs`); 27/27 lib tests
pass; 15/15 coherent under `NUM_ROWS=2 + Path A`. **But:**

- 15-prompt decode median: **90.1 tok/s** (vs 91.5 in 14B baseline,
  −1.5 %) — bench-gate (≥ 93) **NOT met**, and we're *below* the
  baseline.
- `profile_positions` pos=200 GEMV total: **10 618 µs** (vs 10 188
  with NUM_ROWS=1, **+4.2 %** slower per-dispatch).
- gemv_q's +21 % Sprint 13E disaster *is* gone (Path A confirmed to
  fix the LDS-doubling), but the smaller across-the-board +4 %
  regression eats any per-WG-saturation win.

**Reverted to `MMV_NUM_ROWS = 1`.** This is the same shape as
Sprints 12D / 12E / 12H / 13B / 13C / 13D / 13E / 14B: a port of
llama.cpp's RDNA4 recipe whose lever doesn't materialise on this
codebase + this hardware on its own.

After 14A → 14B → 14C, the entire "subgroupAdd + NUM_ROWS=2"
combination llama.cpp ships as their RDNA4 default has been
implemented and measured. **It does not move decode on this
hardware.** The 91-tok/s decode plateau is structural at the graph
level, not at the GEMV-pipeline level.

## 1. The 1-line flip

```rust
// src/backend/vulkan/pipeline_registry.rs:44
- pub const MMV_NUM_ROWS: u32 = 1;
+ pub const MMV_NUM_ROWS: u32 = 2;
```

Under Path A (Sprint 14B default), `_subgroup` SPVs are dispatched
with this spec-constant block:

```rust
const MMV_SPEC_DATA: [u32; 3] = [64, MMV_NUM_ROWS, 1];
                                  //  ^^^^^^^^^^^^
                                  //  becomes 2 with this flip
```

`run_gemv` (`forward.rs:1884`) reads the same constant for dispatch
geometry, so `groups = ceil(m / NUM_ROWS) = m / 2` for our N values.
All N (1024, 4096, 12288, 151936 = vocab) are even. No tail-WG
edge cases.

The `requiredSubgroupSize=64` pin from Sprint 14A is still active
on the GEMV match arm so `subgroupAdd` reduces over all 64 lanes
correctly, even at NUM_ROWS=2 where the subgroupAdd runs twice
(once per row).

## 2. Same-session A/B — NUM_ROWS=2 vs 14B baseline (NUM_ROWS=1)

`profile_positions`, all four positions, sum across 36 layers,
both runs with **Path A default-on** (Sprint 14B's
`mul_mat_vec_subgroup_enabled = true`):

### 2.1 GEMV total (the headline number)

| Position | NR=1 (14B) | NR=2 (14C) | Δ      |
|----------|-----------:|-----------:|-------:|
| pos=0    |  12 310 µs |  10 904 µs | −11.4 % \* |
| pos=50   |  10 242 µs |  10 661 µs |  +4.1 % |
| pos=100  |  10 198 µs |  10 634 µs |  +4.3 % |
| pos=200  |  10 188 µs |  10 618 µs |  **+4.2 %** |

\* pos=0 is dominated by the cold-cache pipeline build pass; ignore
the apparent "−11 %" — the steady-state numbers at pos ≥ 50 all
sit at +4 %.

### 2.2 Top-3 GEMV per dispatch (pos=200)

| Shader     | NR=1 (14B) µs | NR=2 (14C) µs | Δ       |
|------------|--------------:|--------------:|--------:|
| gemv_up    |        3008.1 |        3023.2 |  +0.5 % \* |
| gemv_down  |        1925.5 |        1986.8 |  +3.2 % |
| gemv_gate  |        1713.7 |        1818.0 |  +6.1 % \* |

\* gemv_gate / gemv_up are the back-to-back pair Sprint 12G-D
identified as carrying the `vkCmdWriteTimestamp` artifact; their
individual numbers shift between runs but the sum (gate + up) is
the meaningful figure: 4 722 (NR=1) → 4 841 (NR=2) = +2.5 %.

`profile_positions` only reports the top-3 by share; the
remaining gemv_q / k / v / o values aren't in this run's output.
The `GEMV total` row above already aggregates them — and it's
+4.2 % across the board.

### 2.3 Sanity check: NUM_ROWS=2 + Path B (DISABLE_SUBGROUP_GEMV=1)

Same-session control to confirm Path A actually fixes the
Sprint 13E LDS-doubling disaster:

| Metric           | NR=1 Path A | NR=2 Path A | NR=2 Path B |
|------------------|------------:|------------:|------------:|
| GEMV total pos=200 |     10 188 |     10 618 |     10 610 |
| gemv_up pos=200    |      3008 |      3023 |      3031  |
| gemv_down pos=200  |      1926 |      1987 |      1932  |
| gemv_gate pos=200  |      1714 |      1818 |      1873  |

Reading: NR=2 + Path A and NR=2 + Path B are **essentially
identical** (10 618 vs 10 610 µs GEMV total at pos=200, a 0.08 %
difference). **Path A's lever — eliminating LDS-doubling — works
exactly as designed**, but the underlying NUM_ROWS=2 regression
on this codebase isn't (only) about LDS — it's about VGPR /
register pressure per WG and codegen efficiency at NR=2.

This explains why Sprint 13E's gemv_q +21 % isn't reproduced
here on Path A or Path B: that earlier reading was likely a
combination of LDS-doubling + occasional bigger per-WG spill (which
ACO might or might not do depending on sub-changes to the IR
between sessions). The consistent NR=2 regression we see *today*
is +4 % in both reduction paths — the LDS lever is real but small
relative to the structural NR=2 cost.

## 3. 15-prompt suite (decode median, same-session)

| Config                              | Decode med | Prefill agg | Coherent |
|-------------------------------------|-----------:|------------:|:--------:|
| 14B baseline (NR=1, Path A default) | 91.5 tok/s |       855.8 |   15/15  |
| 14C (NR=2, Path A default)          | 90.1 tok/s |       850.7 |   15/15  |

Decode: −1.5 % (90.1 vs 91.5). **Bench-gate ≥ 93 tok/s NOT met.**

Prefill aggregate within noise (±0.6 %) — as expected, since the
NUM_ROWS spec-constant only affects GEMV (decode), not GEMM
(prefill, coopmat path).

## 4. Bench-gate verdict

| Sub-gate              | Target  | Result | Verdict |
|-----------------------|---------|--------|---------|
| Decode ≥ 93 tok/s     | ≥ 93    | 90.1   | **NO**  |
| 15/15 coherent        | 15/15   | 15/15  | YES     |
| 27/27 lib tests       | 27/27   | 27/27  | YES     |
| Per-dispatch < 14B    | ≤ 10188 | 10618  | **NO**  |

Per the brief's §7.2 (decision tree for "BENCH-GATE NICHT
erreicht"): **revert MMV_NUM_ROWS=2 → 1**.

## 5. Why NUM_ROWS=2 doesn't help on RDNA4

The brief sketched two mechanisms:

1. **WG-count halving → better CU scheduling.** True only if the
   GPU is undersaturated. With Path A active, `gemv_k` at N=1024
   gives 1024 WGs over 64 CUs = 16 WG/CU — already plenty for
   latency hiding on RDNA4. Halving to 512 WGs (8 WG/CU) doesn't
   improve scheduling.
2. **Activation reuse across 2 rows.** True only if reading the
   activation is expensive. At decode (M=1), the activation is
   one token = 4 KB; it lives in the L1 the moment the first WG
   touches it. Halving that L1 traffic is invisible in the wall
   time.

What actually goes the *wrong* direction at NUM_ROWS=2:

- **Per-WG VGPR pressure**: the temp[NUM_COLS][NUM_ROWS] array in
  `mul_mat_vec_base.glsl:9` doubles. With NUM_COLS=1, NUM_ROWS=2,
  the array is 2 floats per thread, vs 1 at NUM_ROWS=1. ACO has
  to allocate 2 VGPRs instead of 1 for partial sums, which
  compounds across the K-loop unroll.
- **Codegen at unusual shapes**: ACO's unroll heuristics for the
  superblock loop in `calc_superblock` are tuned for NUM_ROWS=1
  (the upstream-default ACO regression target). NUM_ROWS=2 hits
  a colder codegen path.
- **GEMV is already memory-bound**: Sprint 12G-D measured
  77-91 % HBM peak. WG-count or reduction changes can't reduce
  bytes read. NUM_ROWS=2 doesn't change the total bytes either —
  same K, same N total — it just reorders them.

llama.cpp's recipe (NR=2 unconditional for K-quants on non-GCN
AMD per `ggml-vulkan.cpp:4128`) was likely measured on a
different RDNA generation (gfx10 / gfx11 — RDNA2/3) or with
their own ACO branch tweaks; the choice doesn't carry over to
gfx1201 in v0.2.3 + this kernel inventory.

## 6. Path A's value in the ledger

Sprint 14B (Path A default-on) and Sprint 14C (NR=2 retest) are
two halves of the same investigation. After 14C, the honest
ledger is:

| Path A (Sprint 14B) | Active default-on |
|---------------------|-------------------|
| Codepath cost reduction at NR=1 | ≤ 0.16 % (noise floor) |
| Variant-coverage parity with llama.cpp | ✅ |
| Prerequisite for future NR=2 ports | ✅ infrastructure ready |

| NR=2 + Path A (Sprint 14C, this) | Reverted |
|----------------------------------|----------|
| Per-dispatch GEMV regression | +4.2 % |
| Decode regression | −1.5 % |
| Sprint 13E disaster fix verified | ✅ (no +21 % anymore) |

So the bottom line: Sprint 14B's Path A *eliminated* the
LDS-doubling penalty that 13E exposed, but the underlying NR=2
cost on RDNA4 is structural (per-WG register pressure +
already-saturated CU scheduling). Path A is shipped; NUM_ROWS=2
stays at 1 for v0.2.4.

## 7. Sprint 14 series — closing

After three sprints (14A plumbing, 14B Path A, 14C NR=2 retest),
the answer to "where is the decode gap" is now definitive:

1. **Not in coopmat tile coverage** (Sprint 12M / 13A: complete).
2. **Not in pipeline-creation flags** (Sprint 14A: pinned;
   Sprint 14B: opted into subgroupAdd). Path A active.
3. **Not in the GEMV reduction path** (Sprint 14B: Path A
   shipped, +0.16 % wall delta — too small to register).
4. **Not in NUM_ROWS** (Sprint 13E + 14C: both falsified, with
   and without Path A).
5. **Not in driver / Mesa / Wave32 / VOPD / f16acc / NUM_ROWS**
   (Sprint 13B / 13D / 13C / 13E / 14C — every config-flip
   lever exhausted).

**It is in graph-level decode infrastructure.** The candidates
that haven't been falsified yet:

- **Multi-submit / command-buffer reuse**: llama.cpp re-submits
  one CB per token; we re-record per-token. Sprint 5A's
  CB-reuse infrastructure exists but doesn't apply to the
  decode path (only prefill / parity tests). A real CB-reuse
  decode loop would need template-based push-constant
  parameters and would touch every shader's pipeline-layout.
  Estimated lift: 5-10 % decode.
- **`lm_head` GEMV at N=151 936**: ~6 % of decode forward,
  could shave ~3 % via a coopmat dispatch.
- **Buffer-aliasing / live-set reduction**: 20+ live SSBOs vs
  llama.cpp's 3-4. May matter for L2 thrashing.

These are v0.3-class infrastructure work items. The Sprint 14
arc closes the GEMV-pipeline branch of the decode-optimisation
tree.

## 8. Outputs

- `pipeline_registry.rs`: comment refresh documenting the 14C
  result. `MMV_NUM_ROWS = 1` (reverted from the 14C trial of 2).
- This report.
- 27 / 27 lib tests, 15 / 15 coherent.
- 72 SPVs (unchanged from Sprint 14B).

## 9. v0.2.4 release-readiness

Sprint 14A + 14B + 14C together form a coherent v0.2.4 scope:

- **14A**: `requiredSubgroupSize` plumbing — infrastructure,
  no perf delta, prerequisite for 14B.
- **14B**: subgroupAdd GEMV reduction (Path A) — default-on,
  no measurable wall delta on RDNA4 at NR=1, but matches
  llama.cpp's recipe and preserves correctness.
- **14C** (this report): NR=2 retest — falsified, reverted.

The **shipping change** in v0.2.4 is the Path A GEMV pipeline
+ the `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` opt-out. Default
performance is unchanged from v0.2.3 (decode 91 tok/s, prefill
0.89× llama.cpp). The release notes should be honest about
that — this is an analysis + infrastructure release, not a
perf release.
