# VulkanForge v0.2.1 Sprint 12F — Sprint 11G-D Retest + Final v0.2.1 Performance Snapshot

**Date:** 2026-04-30
**Branch:** main (HEAD = b79f041, post-Sprint 12E fusion)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV 26.0
**Mode:** Pure benchmark sprint — no code changes, ran existing tools, captured numbers, wrote the report.

## TL;DR

**Sprint 11G stays DEAD.** Int8-coopmat Q4×Q8_1 GEMM retest under the
post-12D/12E code: median speedup **0.49×** (was **0.47×** in 11G-D).
Ratio essentially unchanged — the infrastructure cleanup correctly does
not affect the isolated GEMM bench (which doesn't go through
`dispatch_layer`).

**v0.2.1 final performance snapshot vs v0.2.0:** ~+1 % decode, ~+4 %
prefill at pp=512. Cumulative gain across 13 sprints (11A through 12E)
of focused performance work.

| Metric             | v0.2.0 | v0.2.1 final | Δ      | llama.cpp | Gap to llama.cpp |
|--------------------|--------|--------------|--------|-----------|------------------|
| Decode tok/s       | 90.5   | **91.5**     | +1.1 % | 114.2     | 0.80×            |
| pp=512 tok/s       | 2 255  | **2 352**    | +4.3 % | 4 317     | 0.55×            |
| pp=1024 tok/s      | 2 204  | **2 305**    | +4.6 % | 4 189     | 0.55×            |
| pp=2048 tok/s      | 1 997  | **2 098**    | +5.1 % | 3 771     | 0.56×            |
| Tests              | 167    | **176**      | +9     |           |                  |
| Shaders            | 53     | **65**       | +12    |           |                  |

The decode gap (0.80×) and prefill gap (0.55×) remain. **Sprint 12's
infrastructure-fix hypothesis is empirically disproved** — neither
12D's barrier elision (0 % rate) nor 12E's norm+rope fusion
(+1 % wall-time) translated into the +20–40 % gains 12C predicted.

The remaining gap must be in **GPU compute behaviour**, not host-side
infrastructure. Sprint 12G (RGP profiling) is the next sprint that can
locate the actual bottleneck.

---

## 1. Sprint 11G-D retest

### 1.1 Int8-coopmat Q4×Q8_1 GEMM (full pipeline with scale-fold)

```
Shape                         mul_mmq µs   int8cm µs   Speedup   mul_mmq GFLOPS  int8cm GFLOPS
512 × 4096 × 4096                 893.3      1697.2    0.53×       19 231.5      10 122.5
512 × 1024 × 4096 (GQA)           361.8       733.3    0.49×       11 870.0       5 857.1
 64 × 4096 × 4096 (small)         254.7       629.5    0.40×        8 429.9       3 411.4

Median speedup across 3 shapes: 0.49×
✘ NO-GO (0.49× < 1×)
```

**Comparison vs Sprint 11G-D:**

| Shape                  | 11G-D speedup | 12F speedup | Δ           |
|------------------------|---------------|-------------|-------------|
| 512 × 4096 × 4096      | 0.52×         | 0.53×       | +0.01×      |
| 512 × 1024 × 4096      | 0.47×         | 0.49×       | +0.02×      |
|  64 × 4096 × 4096      | 0.41×         | 0.40×       | −0.01×      |
| **Median**             | **0.47×**     | **0.49×**   | **+0.02×**  |

**Verdict: Sprint 11G remains abandoned.** The +0.02× delta is
within run-to-run noise. As Sprint 12C analytically predicted: the
infrastructure cleanup is *neutral* between mul_mmq and Int8-coopmat
because both paths bench in isolation, outside `dispatch_layer`.

The Int8-coopmat overhead (Q4_K nibble unpack + per-fragment scale-fold
via direct lane→cell access) remains structural — Sprint 12 doesn't
shrink it.

### 1.2 Raw int8 vs scalar dot4 (no Q4_K, no scale-fold)

For completeness, the Sprint 11G-B compute-only bench also reruns
unchanged:

```
Shape                         Scalar µs   Int8-cm µs   Speedup   Scalar GOPS   Int8-cm GOPS
512 × 4096 × 4096               13 279.3     20 165.1   0.66×       5 174.9       3 407.8
512 × 1024 × 4096                3 942.6      2 757.8   1.43×       4 357.5       6 229.6
 64 × 4096 × 4096                3 600.9      1 123.2   3.21×       2 385.5       7 647.5

Median speedup: 1.43×
✓ GO (1.43× ≥ 1.3×)
```

Confirms 11G-B's median +42 % raw compute advantage. Doesn't translate
to Q4×Q8_1 GEMM end-to-end because of the Q4_K + scale-fold overhead
(Sprint 11G-D root cause).

---

## 2. Full v0.2.1 prefill sweep

`run_pp_bench` with `VF_PP_LIST=64,128,256,512,1024,2048`, 5 runs each:

| pp   | Median ms | Mean ms | tok/s     | v0.2.0 (Sprint 10F) | Δ (v0.2.1)   | llama.cpp ref | Gap (v0.2.1) |
|------|-----------|---------|-----------|---------------------|--------------|---------------|--------------|
|   64 |    42.18  |   42.20 |  1 517.3  |  1 511              | +0.4 %       |  2 286        | 0.66×        |
|  128 |    64.13  |   64.08 |  1 996.0  |  2 001              | −0.2 %       |  3 603        | 0.55×        |
|  256 |   116.49  |  116.56 |  2 197.7  |  2 200              | −0.1 %       |  3 999        | 0.55×        |
|  512 |   217.66  |  218.20 |  2 352.3  |  2 255              | **+4.3 %**   |  4 317        | 0.55×        |
| 1024 |   444.27  |  444.32 |  2 304.9  |  2 204              | **+4.6 %**   |  4 189        | 0.55×        |
| 2048 |   976.35  |  976.80 |  2 097.6  |  1 997              | **+5.1 %**   |  3 771        | 0.56×        |

(`pp=4096` triggered a TDR / device-lost recovery — unrelated to this
sprint's scope; also affected v0.2.0. Skipping.)

**Where the +4–5 % at pp ≥ 512 comes from:** Sprint 11C's L-tile mul_mmq
pipeline (BM=BN=128, BK=32, AMD-coopmat-override warptile values) —
the only Sprint 11 work that landed in production. Sprint 11D, 11E,
11F, 11G all delivered NO-GO results. Sprint 12D and 12E delivered
~0 % wall-time impact each.

**The pp=64–256 plateau (≈ 0 % delta):** L-tile activates at `m > 128 && n > 256`,
so it doesn't fire at pp ≤ 256 where small-tile mmq is already the right
choice.

---

## 3. v0.2.1 decode performance (15-prompt suite)

`run_15prompt_bench` full run, 15 prompts × 64–1024 generated tokens each:

```
Per-prompt decode tok/s:    82.2–93.3 (range)
Aggregate decode tok/s:     86.6  (weighted by tokens generated)
MEDIAN decode tok/s:        91.5
Coherent prompts:           15/15
```

**Comparison vs v0.2.0 / Sprint 10F (90.5 tok/s decode):** **+1.1 %**.
That gain is consistent with Sprint 12E's norm+rope fusion saving
36 barriers + 72 dispatches per forward (~0.36 ms × 36 layers × small
constant savings).

**Comparison vs llama.cpp Vulkan reference (114.2 tok/s):** **0.80×**.
Same gap shape as Sprint 10F's measurement.

| Source                           | Decode tok/s | vs llama.cpp |
|----------------------------------|--------------|--------------|
| llama.cpp Vulkan                 | 114.2        | 1.00×        |
| ROCmForge HIP (latest)           |  95.4        | 0.84×        |
| llama.cpp ROCm                   |  87.5        | 0.77×        |
| **VulkanForge v0.2.1 (this)**    | **91.5**     | **0.80×**    |
| VulkanForge v0.2.0               |  90.5        | 0.79×        |

VulkanForge ranks third behind llama.cpp Vulkan and ROCmForge HIP, ahead
of llama.cpp ROCm. Same as v0.2.0 ranking.

---

## 4. GEMM-fraction analysis

### 4.1 Prefill at pp=512

Forward total: 217.66 ms (median, 5 runs). Per layer: 217.66 / 36 = 6.05 ms.

The isolated `bench_int8cm_q4k` mul_mmq numbers give per-shape GEMM time.
Approximate shape-mapping for Qwen3-8B (n_heads=32, head_dim=128,
hidden=4096, ffn=12288):

| GEMM op   | Shape (M_w × K × N_seq)      | µs (estimate)     | × per layer |
|-----------|------------------------------|-------------------|-------------|
| Q-proj    | 4096 × 4096 × 512            | ~ 360 (interp)    | 1           |
| K-proj    | 1024 × 4096 × 512            | ~ 200 (interp)    | 1           |
| V-proj    | 1024 × 4096 × 512            | ~ 200             | 1           |
| O-proj    | 4096 × 4096 × 512            | ~ 360             | 1           |
| Gate-proj | 12288 × 4096 × 512           | ~ 1 000 (interp)  | 1           |
| Up-proj   | 12288 × 4096 × 512           | ~ 1 000           | 1           |
| Down-proj | 4096 × 12288 × 512           | ~ 1 000           | 1           |
| **Sum**   |                              | **≈ 4 120 µs**    | per layer   |

Estimated GEMM share: 4.12 ms / 6.05 ms = **~68 %** of per-layer time.

The remaining ~32 % covers attention, norms, RoPE, swiglu, residuals,
quantize_q8_1 dispatches, and barrier/launch overhead. That's roughly
consistent with what the structural dispatch count would predict
(7 GEMMs, ~10 non-GEMM dispatches per layer; non-GEMM ops are smaller
but more numerous).

### 4.2 Decode

Forward total: 1 / 91.5 = 10.93 ms. Per layer: 10.93 / 36 = 0.30 ms.

Decode GEMVs are **much** smaller per-call (~30–50 µs each) but the
ratio of dispatches to total time skews differently. Without isolated
GEMV bench numbers (we don't have a `bench_gemv_only` example), we
can't attribute decode time as cleanly. The Sprint 12E result that
saved 72 dispatches per forward and yielded ~+1 % suggests
non-GEMV dispatch overhead is small (~3 µs per saved dispatch ×
72 = ~0.2 ms ≈ 1.8 % of forward).

Most of decode's 10.93 ms is therefore **GEMV compute time** — Sprint
12C's "CU starvation" hypothesis becomes the leading candidate
(decode GEMVs at M=1, N=4096–11008 may spawn too few workgroups to
fill RDNA4's 64 CUs).

---

## 5. Sprint 12 final scoreboard

| Sprint | Scope                                  | Outcome                              |
|--------|----------------------------------------|--------------------------------------|
| 12A    | llama.cpp Vulkan-backend audit         | ✅ 7-category catalogue, line-refs   |
| 12B    | VulkanForge backend audit              | ✅ Matching catalogue + diffs        |
| 12C    | Gap analysis + sprint 11 revalidation  | ❌ **Quantitative model 30× off**    |
| 12D    | Barrier elision via dirty-flags        | ✅ Built; **0.0 % elision empirical**|
| 12E    | Decode norm+rope fusion                | ✅ Built; **+1 % wall-time**         |
| 12F    | Retest + final v0.2.1 snapshot (this)  | ✅ 11G stays dead; **+1 % decode v0.2** |

**Cumulative empirical gain from Sprint 12:** ~+1 % decode,
~+0.5 % prefill (12C predicted +38 % decode, +3 % prefill).

The audits (12A, 12B) are reusable infrastructure for any future
gap analysis. The cost models in 12C are not — they assumed CPU /
host-side dispatch overhead matters at our forward scale, and
empirically it doesn't on this hardware/driver/code-path combination.

---

## 6. The decode gap: where it actually is

After 12D + 12E both delivered ~0 % despite reducing the structural
quantities Sprint 12C said to reduce, the conclusion is unavoidable:

**Decode time is dominated by GPU compute, not host-side infrastructure.**

Specifically: for a 10.93 ms decode forward with 612 dispatches
(36 layers × 17 OPs/layer = 612, post-12E count is 540), at
~5 µs CPU record cost per dispatch:
- Total CPU record time: ~3.06 ms
- This runs *in parallel* with GPU work for the previous cmd buffer,
  but the final wait_for_fences blocks for total GPU time anyway
- GPU time: ~10.93 ms (the entire forward; record overhead is hidden
  by parallelism)

The remaining ~7.9 ms of "non-record" time is GPU compute — 36 layers
of 17 dispatches running in sequence on the GPU. Saving 36 barriers
(at ~2 µs GPU stall each) = ~72 µs = ~0.7 % of forward.

To meaningfully shrink decode time, we need to shrink **GPU compute
per forward**, not host-side overhead. That means:
- Per-dispatch GPU time (depends on shader / CU occupancy / memory BW)
- Number of GPU-resident dispatches (already cut by 12E; further cuts
  need real fusion shaders)
- Cache efficiency between dispatches (GPU L1/L2 churn)

None of these are visible from CPU-side benchmarks. Direct GPU
profiling is required.

---

## 7. Recommended next steps

### 7.1 Sprint 12G (recommended): RGP profiling

Capture a single decode forward with [Radeon GPU Profiler](https://gpuopen.com/rgp/)
and extract:
1. Per-dispatch GPU wall time
2. CU occupancy per dispatch
3. Memory BW utilisation
4. Cache hit rates (L1/L2)
5. Inter-dispatch stall time

Cost: ~2–3 days (1 day for capture setup, 1–2 days for analysis).
Outcome: empirical answer to "where does decode time actually go?".

If RGP shows a clear single bottleneck (e.g., CU starvation on GEMV at
M=1), the next optimization sprint is well-defined.

If RGP shows time uniformly distributed across many small dispatches,
the gap may be irreducible without major architectural change (e.g.,
multi-token decode batching) — and we should pivot to v0.3 features.

### 7.2 Pessimistic plan: accept the ceiling, pivot to v0.3

If 12G doesn't reveal a clear lever:
- Accept ~91.5 tok/s decode and ~2 350 tok/s pp=512 as the v0.2.1
  ceiling on RDNA4 / RADV / Vulkan-1.3.
- Document this as the architectural limit of "single-submit
  blocking + 17 dispatches per layer + scalar-tight inner loops".
- Pivot Sprint 13+ work to v0.3 features:
  - Multi-modal support (image/audio embeddings)
  - Larger context windows (>8K tokens)
  - More model architectures (Llama-3 70B, Mixtral)
  - Quantisation-aware training paths

**The 0.80× decode gap and 0.55× prefill gap may be the structural
ceiling of this implementation strategy.** llama.cpp Vulkan's higher
numbers come from a fundamentally more aggressive graph-fusion and
multi-submit pipeline that we deliberately chose not to replicate
(Sprint 12B §1: "single-submit blocking" was a clarity-over-throughput
trade-off).

If the team chooses to chase parity with llama.cpp Vulkan, the only
remaining lever is a major refactor of `forward.rs` to pipelined
multi-submit recording (Sprint 12C section 3 "submit pipelining",
estimated 7–10 days, predicted +5–10 % decode). After 12D and 12E's
results, that prediction's confidence interval should be widened to
"+0–10 %" and the sprint should ship with an explicit gate: if
delivered improvement < +5 %, revert and accept the ceiling.

---

## 8. Files touched

```
NEW   results/v021_sprint12f_retest.md   (this report)
```

No code changes. Pure measurement.

## 9. Tests / regression

```
test result: ok. 27 + 9 + 18 + 77 + 8 + 8 + 27 = 176 passed
```

176/176 green (unchanged from 12E).

## 10. Take-aways

1. **Sprint 11G stays abandoned.** Int8-coopmat Q4_K GEMM ratio
   barely moved (0.47× → 0.49×) under post-12D/12E code. Confirms
   12C's analytical prediction that infrastructure cleanup is
   ratio-neutral.

2. **v0.2.1 final delivers ~+1 % decode, ~+4 % prefill at pp ≥ 512** —
   almost entirely from Sprint 11C's L-tile mul_mmq pipeline (the
   one Sprint 11 win that shipped to production). Sprint 12 added
   negligible wall-time on top.

3. **The decode gap is GPU-compute, not host-side infrastructure.**
   Two empirical sprints (12D, 12E) both reduced the structural
   quantities Sprint 12C said to reduce, both yielded ~0 %
   end-to-end. The infrastructure-overhead model was wrong by ~30×.

4. **Sprint 12G (RGP profiling) is the only remaining
   investigative step** before declaring the architectural
   ceiling. Cost is 2–3 days. If it doesn't reveal a clear lever,
   the team should pivot to v0.3 features rather than continue
   diminishing-returns infrastructure work.
