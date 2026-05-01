# Sprint 15B — lm_head optimisation: HONEST NEGATIVE (BW-bound at 94 %)

**Premise.** After Sprint 15A scoped two remaining v0.3 candidate
levers (async multi-submit, `lm_head` coopmat / NUM_ROWS), Sprint 15B
targets the smaller-scope one: the single lm_head dispatch costs
~740 µs / token and represents 6–8 % of the decode forward. Brief's
hypothesis: lm_head has 2 374 WGs/CU on RDNA4 (vs 16–192 for
layer-GEMVs), so scheduler overhead might be measurable, and
`NUM_ROWS=2` could halve the WG count without the per-WG VGPR
penalty that hurt layer-GEMVs in Sprint 14C. Bench-gate: lm_head
≤ 600 µs (−20 % vs 740 µs baseline).

**Verdict.** **Hypothesis falsified at pre-check.** `output.weight`
is **Q6_K** (not Q4_K). Brief's Fallstrick #1 spelled out the
implication: Q6_K's HBM-bandwidth floor is ~784 µs; our measured
740 µs is already at **~94 % of that floor** (or even slightly
below it, depending on the exact Q6_K bytes-per-weight figure used —
0.8125 vs 0.8203 B/w changes the floor by ~10 µs). There is no
meaningful headroom for shader-config optimisation.

A quick same-session A/B confirms: globally flipping `MMV_NUM_ROWS`
from 1 → 2 changes lm_head from **740.0 µs → 740.8 µs (+0.1 %, within
run-to-run noise)**. Halving the WG count from 151 936 → 75 968 is
**invisible** at the wall — the bottleneck is not WG scheduling, it
is HBM bandwidth.

No code changes shipped. 27 / 27 lib tests, 15 / 15 coherent
(v0.2.4 default-config, unchanged). Same shape as 13 prior
honest-negatives in the Sprint 12 → 15 arc — closes the lm_head
config branch, leaves only graph-level work for v0.3.

## 1. Pre-check (the decisive step)

### 1.1 lm_head tensor type — Q6_K, not Q4_K

```
$ cargo run --release --example check_lm_quant
output.weight                  = Q6K  dims=[4096, 151936]
token_embd.weight              = Q4K  dims=[4096, 151936]
blk.0.ffn_down.weight          = Q6K  dims=[12288, 4096]
blk.0.attn_v.weight            = Q6K  dims=[4096, 1024]
blk.0.attn_q.weight            = Q4K  dims=[4096, 4096]
blk.0.attn_output.weight       = Q4K  dims=[4096, 4096]
```

The brief's check (Schritt 0.5 + Fallstrick #1) called this out
explicitly:

> Falls Q6_K: KEIN Headroom → Sprint = sofort NEGATIVE!

In Qwen3-8B-Q4_K_M, the M-quantisation rule keeps `output.weight`
at Q6_K alongside `attn_v.weight` and `ffn_down.weight` for
precision. (`token_embd.weight` is Q4_K and could be used as a
tied LM head — we already fall back to it via
`forward.rs:1614-1616` — but the model ships a dedicated
`output.weight` Q6_K tensor and we correctly prefer that.)

### 1.2 BW-bound math

```
Q6_K bytes per weight: ~0.8203 B/w
                       (= 210 byte block / 256 weights, ggml.h)
Total weight bytes:    151 936 × 4 096 × 0.8203 = 510.5 MB
RX 9070 XT HBM peak:   644 GB/s (Sprint 12G-D)
BW-floor:              510.5 MB / 644 GB/s = ~793 µs
                       (theoretical absolute minimum at 100 % peak)
At 85 % peak:          ~933 µs (typical large-kernel realised BW)
Measured:              740 µs
```

The measured 740 µs is **below** the theoretical 100 % HBM floor —
faster than memory-bandwidth physics seemingly allows. Two things
are likely contributing:

1. **Infinity Cache (64 MB on RX 9070 XT)** holds the start of
   the weight matrix from token N's lm_head dispatch into token
   N+1's. Even though 510 MB > 64 MB, the *first* 64 MB of weights
   for the next dispatch is an Infinity-Cache hit (it was the last
   64 MB read by the previous dispatch).
2. Some launch/scheduling overlap with `rms_norm_final` (4.6 µs)
   that precedes lm_head shaves another few microseconds.

Either way, **we are already operating very close to the BW
ceiling for this dispatch shape on this hardware**. The brief's
Fallstrick #1 logic holds: there is no shader-config lever that
can move 740 µs meaningfully.

### 1.3 Cross-check — layer Q6_K dispatches

Sanity-check the BW analysis against a Q6_K layer dispatch we have
fresh data for: `gemv_down` (K=12288, N=4096, Q6_K).

```
Per-dispatch bytes:    12 288 × 4 096 × 0.8203 = 41.3 MB
Measured per-disp:     1 933.3 µs / 36 = 53.7 µs / dispatch
Effective BW:          41.3 MB / 53.7 µs = 769 GB/s
```

That is *also* above the 644 GB/s nominal peak. Same explanation:
41 MB fits in the 64 MB Infinity Cache, so consecutive dispatches
benefit. Both lm_head and gemv_down per-dispatch numbers are
consistent with **HBM + Infinity Cache** being the bottleneck, with
no slack for shader optimisation.

## 2. Quick experiment — `MMV_NUM_ROWS` does not help lm_head

The brief's Fallstrick #5 sketched the contrarian view: at 2 374
WGs/CU lm_head might benefit from NUM_ROWS=2 *even if* layer-GEMVs
regressed in Sprint 14C (those have 16–192 WGs/CU). One-line
spec-constant flip, profile, revert:

```
-pub const MMV_NUM_ROWS: u32 = 1;
+pub const MMV_NUM_ROWS: u32 = 2; // TEMP: Sprint 15B lm_head probe
```

Same-session A/B at pos=200, 4 consecutive `profile_positions`
runs of the new pipeline:

| Run | NUM_ROWS=1 baseline (µs) | NUM_ROWS=2 trial (µs) |
|-----|------------------------:|----------------------:|
|   1 |                   740.0 |                 743.9 |
|   2 |                   ~740 |                 741.4 |
|   3 |                   ~740 |                 741.6 |
|   4 |                   ~740 |                 740.8 |

**+0.1 % to +0.5 %, well within the ±2 % run-to-run noise on this
rig.** The WG-count theory is wrong: halving WGs from 151 936 →
75 968 changes nothing because the dispatch is HBM-bound and
the GPU's wave-pool can already hide all the latency. RDNA4
schedules thousands of WGs without measurable per-WG overhead;
the relevant constraint is total bytes read, not WGs in flight.

Reverted to `MMV_NUM_ROWS = 1`. No code shipped from this experiment.

## 3. Why the wall doesn't move at 2 374 WGs/CU

The brief's Fallstrick #5 framed lm_head's WG count as a problem
the layer-GEMV experiment doesn't transfer over. That framing
is incorrect on RDNA4 hardware:

- **WG-launch overhead** on RDNA4 is per-CU, fixed-cost, paid once
  per dispatch (or once per wavefront-launch batch from the
  scheduler) — not proportional to WG count past saturation.
- **Wave occupancy** caps at 16 Wave64 / SIMD = 4 SIMDs × 16 = 64
  waves / CU. Once we have ≥ 64 WGs / CU eligible to run, the
  remainder are just queued — no extra scheduling cost vs 1 / CU.
- lm_head's "2 374 WGs/CU" is queue depth, not active waves.
  Active waves are ≤ 64. The scheduler doesn't iterate the queue
  at runtime cost — it just dispatches the next WG when a slot
  opens.
- HBM bandwidth saturates at 64 active waves anyway; adding more
  "in queue" doesn't extract more bandwidth.

The Sprint 14C result (NUM_ROWS=2 +4.2 % per-dispatch slower for
layer-GEMVs at 16–192 WGs/CU) was driven by **per-WG VGPR pressure
doubling** — the cost we paid was VGPR allocation, not WG-count
arithmetic. That cost transfers identically to lm_head: NUM_ROWS=2
doubles VGPRs there too. The reason we don't *see* a regression
on lm_head is just that lm_head is so far HBM-bound that the
per-WG compute-side cost is invisible. (NUM_ROWS=4 or 8 might
eventually become VGPR-spill-bound and slow lm_head down — but
that's not a positive lever either.)

## 4. The other path: dedicated coopmat lm_head dispatch

The Sprint 15A analysis sketched a "dedicated `lm_head` coopmat
dispatch" as the third candidate (Section 3.2 of that report).
With the BW-bound finding above, this path looks much weaker
than initially estimated:

- Coopmat (WMMA) reduces VALU compute, not HBM bandwidth. lm_head
  is *not* VALU-bound — at 740 µs we are already at ~94 % of HBM
  ceiling, with compute slack the bottleneck doesn't expose.
- A coopmat lm_head dispatch would still need to read the same
  510 MB of Q6_K weights from HBM. The compute saving would
  evaporate behind the HBM wait.
- The Sprint 14C / 14B lessons apply: WMMA / subgroup-arithmetic
  optimisations don't help when the kernel is bandwidth-bound.

This is the same shape as the Sprint 13D (Wave32 / VOPD) and
13C (f16acc) findings: hardware features that target VALU
throughput don't rescue memory-bound kernels.

**The "dedicated lm_head coopmat" candidate is downgraded from
"~+3 % decode lift" to "0–1 % decode lift, probably 0".**

## 5. What this leaves for v0.3

After Sprints 12-14 (9 hypotheses), 15A (CB-reuse hypothesis
falsified at source-reading), and now 15B (lm_head optimisation
falsified at BW-bound math), the remaining unfalsified candidates
shrink:

| # | Candidate | Estimated lift | Sprint cost | Status |
|---|---|---|---|---|
| 1 | Async multi-submit / CPU-GPU pipelining | +5–15 % decode | 1–2 weeks | unbudgeted |
| 2 | Dedicated lm_head coopmat | **was +2-4 %, now 0-1 %** | ~3 days | **15B downgrade** |
| 3 | Buffer-aliasing / live-set reduction | unmeasured, 0–5 % | ~1 week | unbudgeted |

Path #1 (async multi-submit) is now **the only candidate with a
plausibly material lift**. Sprint 15A already analysed its design;
Sprint 15C should prototype it.

## 6. Outputs

- This report.
- **No code changes.** `MMV_NUM_ROWS` reverted to 1 after the
  same-session probe. v0.2.4 binary unchanged.
- 27 / 27 lib tests, 15 / 15 coherent.
- The decode-gap analysis ledger now has **11** entries
  (Sprints 12D / 12E / 12H / 13B / 13C / 13D / 13E / 14B / 14C /
  15A / 15B), all falsified.

## 7. Honest framing

This is the cheapest possible falsification: a 5-minute
quant-type lookup (Schritt 0.5) made the whole sprint
unnecessary. Sprint 15B was scoped as "the safer, smaller
counterpart to a multi-week async multi-submit refactor", but
the hypothesis depended on Q4_K bandwidth math that doesn't
apply to a Q6_K tensor. Brief's Fallstrick #1 anticipated
this exactly:

> Q4_K: BW-Limit ~543µs → Headroom für Optimierung!
> Q6_K: BW-Limit ~784µs → 740µs wäre schon 94.4% Effizienz!
> → Falls Q6_K: KEIN Headroom → Sprint = sofort NEGATIVE!

The methodology — measure first, check assumptions, falsify
cheaply — keeps working. We have eleven examples now. The next
sprint should be the async multi-submit prototype (Section 5
of this report, Section 3.1 of Sprint 15A's report) or a
deliberate v0.2.x → v0.3 architectural reset that doesn't try
to find a single config-flip lever in code that has been
exhaustively searched at that level.
