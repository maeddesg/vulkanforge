# VulkanForge v0.2 Sprint 10F — Final Bench + Documentation + Official Release Push

**Date:** 2026-04-29
**Tag:** v0.2.0
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV, kernel 7.0.1-cachyos
**Model:** Qwen3-8B-Q4_K_M
**Environment:** Citrix terminated → first measurements without ~7 % background GPU load.

## TL;DR

This is the closing sprint for the v0.2 series — a clean re-bench
without Citrix in the background, full README + CHANGELOG refresh,
167/167 regression tests, and the official `git push` for v0.2.0.

**Key finding:** the Citrix-Δ that we'd been hedging against the entire
v0.2 series turned out to be **near zero** (~−1.6 % to +0.9 %, indistinguishable
from run-to-run noise). The Sprint 10E.5 numbers were already honest;
the "5–10 % conservative" expectation in earlier reports was wrong. The
Citrix terminal was visible but not GPU-active enough to skew numbers.

## Headline performance

### Prefill pp-sweep (RUNS=5 median, no Citrix)

| pp   | tok/s   | vs Sprint 10E.5 (RUNS=3) | llama.cpp Vulkan | Ratio  |
|------|--------:|-------------------------:|-----------------:|-------:|
|   64 |  1510.8 |                   −1.6 % |             2286 | 0.66×  |
|  128 |  2000.8 |                   −0.4 % |             3603 | 0.56×  |
|  256 |  2200.3 |                   +0.9 % |             3999 | 0.55×  |
|  512 |**2254.6** |                 +0.5 % |             4317 | 0.52×  |
| 1024 |  2204.3 |                   +0.7 % |             4189 | 0.53×  |
| 2048 |  1997.4 |                   +0.5 % |             3771 | 0.53×  |
| 4096 |  1659.4 |                   +0.4 % |             3272 | 0.51×  |

Peak prefill: **2254.6 tok/s @ pp=512** = **0.52× llama.cpp Vulkan**.

The "Citrix-Δ" column is run-to-run noise, not a Citrix savings — see
the methodology note below.

### 15-prompt benchmark (Qwen3-8B-Q4_K_M, all 15 prompts)

| Metric                     | Value          |
|----------------------------|----------------|
| Median prefill tok/s       | **1068**       |
| Median decode tok/s        | **90.5**       |
| Aggregate prefill tok/s    | 1089           |
| Aggregate decode tok/s     | 85.5           |
| Coherent prompts           | 15/15          |
| Total prompt tokens        | 802            |
| Total decode tokens        | 6080           |

15-prompt details:

| #  | Prompt                          | pp  | gen  | prefill tok/s | decode tok/s | Coh |
|----|---------------------------------|----:|-----:|--------------:|-------------:|:---:|
|  1 | Greeting                        |  20 |   64 |        378.6  |        91.7  | ✓   |
|  2 | Simple Sequence                 |  31 |   64 |        734.8  |        91.2  | ✓   |
|  3 | Prime Check (Python)            |  31 |  256 |        748.0  |        90.7  | ✓   |
|  4 | LRU Cache (C++)                 |  47 |  512 |       1111.0  |        90.4  | ✓   |
|  5 | REST API (Go)                   |  62 | 1024 |       1459.5  |        80.9  | ✓   |
|  6 | Mutex Explanation               |  29 |  128 |        701.2  |        91.4  | ✓   |
|  7 | TCP vs UDP                      |  39 |  512 |        926.3  |        90.5  | ✓   |
|  8 | GPU Architecture Blog Post      |  58 | 1024 |       1375.9  |        81.2  | ✓   |
|  9 | Binary Search Complexity        |  30 |  256 |        732.8  |        90.8  | ✓   |
| 10 | Debug Code                      |  45 |  256 |       1069.6  |        90.8  | ✓   |
| 11 | Distributed Message Queue       |  62 | 1024 |       1465.9  |        81.1  | ✓   |
| 12 | Long System Prompt + Question   | 198 |  256 |       1667.5  |        90.5  | ✓   |
| 13 | Long Output Story               |  67 |  512 |       1068.1  |        90.2  | ✓   |
| 14 | Arithmetic (Q4_K Precision)     |  31 |   64 |        753.1  |        91.8  | ✓   |
| 15 | Emoji/Special Characters        |  52 |  128 |       1244.9  |        90.5  | ✓   |

Decode dips to ~81 tok/s on the 1024-token-output prompts (5/8/11)
because at gen=1024 the kv_len reaches ~1100 and the decode-time
attention cost grows with kv_len. The pp-sweep peak (pp=512,
gen=0) is unaffected.

### 4-system comparison (Qwen3-8B, RX 9070 XT)

| System                  | Decode tok/s | Prefill peak tok/s | Decode | Prefill |
|-------------------------|-------------:|-------------------:|-------:|--------:|
| llama.cpp Vulkan (`-fa 1`) |    114.2  |              4317  | 1.00×  |  1.00×  |
| **VulkanForge v0.2.0**   |   **90.5** |          **2255**  | **0.79×** | **0.52×** |
| llama.cpp ROCm           |      87.5  |              3684  | 0.77×  |  0.85×  |
| ROCmForge HIP            |      95.4  |               769  | 0.84×  |  0.18×  |

vs **VulkanForge v0.1.3** (last shipped: 88.6 / 1037 med 15-prompt):
**+2.1 % decode, +118 % prefill peak.**

## Methodology

### Citrix-Δ — empirical zero

Sprint 10E.5 ran with `RUNS=3` while a Citrix terminal was visible (~7 %
background GPU load reported by `radeontop`). This sprint ran the same
benches with Citrix terminated and `RUNS=5`. Direct comparison:

| pp   | 10E.5 (Citrix, RUNS=3) | 10F (no Citrix, RUNS=5) | Δ      |
|------|-----------------------:|------------------------:|-------:|
|   64 |                 1535.3 |                  1510.8 | −1.6 % |
|  128 |                 2009.4 |                  2000.8 | −0.4 % |
|  256 |                 2181.1 |                  2200.3 | +0.9 % |
|  512 |                 2244.2 |                  2254.6 | +0.5 % |
| 1024 |                 2189.3 |                  2204.3 | +0.7 % |
| 2048 |                 1988.0 |                  1997.4 | +0.5 % |
| 4096 |                 1652.0 |                  1659.4 | +0.4 % |

The Δ is **inside run-to-run noise** at every pp — sometimes positive,
sometimes negative. **Conclusion: Citrix wasn't actually GPU-stealing
under the workload we were measuring.** The visible-but-idle terminal
window doesn't share the compute queue. Earlier sprints' "we'd be
~5–10 % faster without Citrix" claims were a hedge; the truth is the
numbers were already correct.

This is worth recording: the v0.2 series' performance claims are
defensible *as published*. Anybody re-running the benches on cleaner
hardware should expect the same numbers, not 5 % higher.

### What changed sprint-to-sprint

The Citrix conjecture entered the project in earlier sprints (Sprint
9-series notes mentioned "absolute Zahlen sind konservativ"). It was a
reasonable hedge, but it was also a hedge — never measured. This sprint
measures it. Outcome: zero.

For future v0.3 work, drop the hedging language. Report the numbers
that came out of `run_pp_bench`; they're what the GPU actually delivered.

## Decode-only verification

The 15-prompt bench's median decode of **90.5 tok/s** matches Sprint
10E.5's earlier observation. Decode hot-path is GEMV-bound and was not
substantially touched in v0.2 (the v0.2 wins are all in prefill). The
+2.1 % vs v0.1.3 (88.6 → 90.5) comes from coopmat attention helping the
*long-context* decode steps (gen=1024 cases see kv_len ≥ 1024 by the
last token), plus FP16 KV reducing memory pressure in the decode
attention loop.

## Documentation updates

### `README.md` — rewrote three sections

1. **Status block** — bumped to v0.2.0 with a "Key features (v0.2.0)"
   subsection covering coopmat QK, FP16 KV, fused kernels, tiled FA,
   and the pp=4096 unblock.
2. **Performance section** — replaced the v0.1.3 mixed-prompt-length
   table with the dedicated pp-sweep table + 15-prompt summary table +
   refreshed 4-system comparison.
3. **Limitations section** — removed "no coopmat / WMMA path" (we have
   one now), removed "no quantized cache (KV is f32)" (FP16 default),
   removed "SPM tokenizer not implemented" (was added in v0.1.1).
   Added forward-looking notes: GEMM is still scalar (v0.3 territory),
   PV-coopmat documented as honest negative, scalar-attention TDR risk
   only on explicit opt-out.

### `CHANGELOG.md` — added v0.2.0 entry

Sprint-by-sprint summary of the v0.2 series:
- Sprints 5–7 (tiled FA + Br/Bc sweep)
- Sprint 8a / 8b / 8b.1 (FA default ON, conditional barriers honest neg.)
- Sprints 9d.1–9d.3 (FP16 KV infrastructure → hot-path → default)
- Sprint 10A (cm2 deep-dive, pivot to cm1)
- Sprint 10B (microbench, +47.5×, GO)
- Sprint 10C (coopmat QK v1, +85.8 % @ pp=2048)
- Sprint 10D (PV-coopmat honest negative)
- Sprint 10E (default ON)
- Sprint 10E.5 (TDR investigation, no code change)
- Sprint 10F (this release)

Plus the fused-kernels table, coopmat-attention details, TDR resolution
notes, and the 167/167 test status.

## Test suite

```
test result: ok. 27 passed; 0 failed; 0 ignored  (lib unit tests)
test result: ok.  0 passed; 0 failed; 0 ignored  (doc-tests)
test result: ok.  9 passed; 0 failed; 0 ignored  (correctness, set 1)
test result: ok. 18 passed; 0 failed; 0 ignored  (correctness, set 2)
test result: ok. 70 passed; 0 failed; 0 ignored  (correctness, set 3)
test result: ok.  8 passed; 0 failed; 0 ignored  (sprint-isolated)
test result: ok.  8 passed; 0 failed; 0 ignored  (sprint-isolated 2)
test result: ok. 27 passed; 0 failed; 0 ignored  (regression)
test result: ok.  0 passed; 0 failed; 0 ignored  (doc-tests)
                                                  ────
                                            TOTAL  167 / 167 ✓
```

Wall time: 71 s (well under the 5-minute budget).

## Decision log

- **No new code changes.** This sprint is bench + docs + push. The
  underlying performance is what Sprint 10E shipped; we're just
  measuring it cleanly and writing it down.
- **Citrix-Δ documented at zero.** Drop the hedging language going
  forward.
- **README's "v0.1.3" disclaimer block removed.** That paragraph
  warned that v0.1.0–v0.1.2 prefill numbers were inflated by the
  `BLOCK_SIZE = 128` bug. v0.1.3 is now also history; the relevant
  caveat for v0.2 readers is in the CHANGELOG.

## Files touched

- `README.md` — Status block, Performance section, Limitations.
- `CHANGELOG.md` — new v0.2.0 entry.
- `results/v02_sprint10f_final_bench.md` — this report.

No source / shader changes.

## Push

Authorised by the user as part of this sprint brief ("DAS IST DER
OFFIZIELLER v0.2 PUSH"). 7 commits on top of `origin/main` will be
pushed:

```
6f7b859 v0.2 sprint10e.5: pp=4096 TDR-crash investigation
2a24a6d v0.2 sprint10e: coopmat attention default ON
c619294 v0.2 sprint10d: PV-coopmat — honest negative result
bf256cf v0.2 sprint10c: coopmat flash-attention v1 (QK only)
a83b6d1 v0.2 sprint10b: isolated coopmat QK micro-benchmark — STRONG GO
7a77393 v0.2 sprint10a: flash_attn_cm2 deep-dive — pivot to cm1
…  + this sprint's commit (10F: bench + docs + push)
```

## Forward look (v0.3 candidates)

- **coopmat GEMM** for prefill — the real prize. Current prefill peak
  is GEMM-bound (`mul_mmq.comp`). Closing the 0.52× → ~0.85× gap
  to llama.cpp Vulkan needs a KHR-coopmat GEMM kernel. Patterns from
  llama.cpp's `mul_mm.comp` + the Sprint 10B microbench experience.
- **PV-coopmat retry** — Sprint 10D's LDS-scratch approach failed.
  Alternative: A-fragment pull from a `pscore` coopmat without the
  LDS round-trip, or feed `pscore` through subgroup ops. Risk
  category: same FP16/FP32 mixing gotcha that 10D ran into.
- **Decode-path optimisation** — Sprint 10E's coopmat already applies
  to decode (same `flash_attn_tiled` cascade). The ceiling is GEMV
  bandwidth. Possible: weight-prefetch into LDS for the GEMV inner
  loop, or a coopmat-style reduction across heads.
