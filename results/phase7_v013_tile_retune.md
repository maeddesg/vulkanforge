# Phase 7 — TM / TN re-tune at `BLOCK_SIZE = 256`

**Date:** 2026-04-27
**Build:** v0.1.3 (`BLOCK_SIZE = 256`, `mul_mm` opt-in)
**Hardware:** AMD Radeon RX 9070 XT, gfx1201, RDNA 4
**Outcome:** `TM = 2, TN = 4` confirmed optimal — **default unchanged**.

---

## Why re-tune

The Phase 6 v0.1.2 sweep that picked `TM = 2, TN = 4` ran at the buggy
`BLOCK_SIZE = 128` (only 2 warps per workgroup, half of every output
tile unwritten). With v0.1.3's correct `BLOCK_SIZE = 256` (4 warps),
register pressure, LDS usage, and occupancy all change — the v0.1.2
ranking might no longer apply.

The constraint set is fixed by the GEMM kernel itself:

* `WNITER = (WM × WN) / (WARP × TM × TN × WMITER)` must be a positive
  integer. With `WM = WN = 32`, `WARP = 64`, `WMITER = 2` that
  collapses to **`TM × TN ≤ 8`**.
* The inner-loop body uses `TM / 2` (paired-row FMA), so **`TM` must
  be even** (≥ 2). That eliminates the prompt's tentative
  `TM = 1, TN = 4` and `TM = 1, TN = 8` candidates outright — they
  would compile but the inner loop would do zero iterations.

Valid grid: `TM ∈ {2, 4, 8}` × `TN ∈ {1, 2, 4}` with `TM × TN ≤ 8`,
giving six configs.

---

## Sweep — 5-prompt suite, Qwen3-8B-Q4_K_M

`VF_NUM_PROMPTS=5` truncates `inference_test_prompts_15.json` to the
first five (Greeting, Simple Sequence, Prime Check, LRU Cache,
REST API). Prompt sizes 20–62 tokens, generated 64–1024 tokens each.

| TM | TN | TM·TN | WNITER | Decode med | Prefill med | Δ vs current |
|---:|---:|---:|---:|---:|---:|---:|
| **2** | **4** | **8** | **1** | **89.3** | **737.8** | **0.0 % (current)** |
|  2 | 2 | 4 | 2 | 88.9 | 743.0 | **+0.7 %** |
|  2 | 1 | 2 | 4 | 88.9 | 740.5 | +0.4 % |
|  4 | 1 | 4 | 2 | 88.7 | 681.6 | −7.6 % |
|  4 | 2 | 8 | 1 | 88.7 | 684.1 | −7.3 % |
|  8 | 1 | 8 | 1 | 89.0 | 568.5 | **−23 %** |

Two clean clusters:

* **`TM = 2`** (any `TN`) lands in 738–743 tok/s. `TN = 2` edges out
  `TN = 4` by 0.7 %.
* **`TM = 4`** drops to ~683 tok/s (−7 %).
* **`TM = 8`** collapses to 569 tok/s (−23 %) — same direction as the
  v0.1.2 result (v0.1.2: `TM = 8 TN = 1` was −7 %; bigger swing now
  because the 256-thread workgroup wants more independent rows to
  fill latency, and `TM = 8` over-serialises one row).

**Decode is unchanged** (88.7–89.3 tok/s) — `TM`/`TN` only affect the
GEMM tile shape, and decode runs on `mul_mat_vec_*.comp`, which is
GEMV-shaped.

---

## 15-prompt confirmation (current vs apparent 5-prompt winner)

The 5-prompt sweep flagged `TM = 2 TN = 2` as 0.7 % ahead. To rule out
sampling noise we re-ran each on the full 15-prompt suite:

| Config | 15-prompt median prefill | Total prefill | Coherent |
|---|---:|---:|---:|
| `TM = 2 TN = 4` (current) | 1047.1 tok/s | 1024.2 tok/s | 15 / 15 |
| `TM = 2 TN = 2` (candidate) | 1055.2 tok/s | 1027.9 tok/s | 15 / 15 |

Run-to-run noise on the 15-prompt suite is **~1 %** — the earlier
`TM = 2 TN = 4` run from `phase7_v013_benchmark.md` measured 1037.4
tok/s for the same config, putting today's two `TM = 2 TN = 4`
samples at 1037 / 1047 — already a 1 % spread. The
`TM = 2 TN = 2` median (1055.2) lies inside that noise band.

Verdict: the 0.4–0.8 % observed difference is run-to-run sampling
noise; **neither config is statistically faster**. Keep
`TM = 2 TN = 4` as the default — it carries forward from v0.1.2 with
no churn.

---

## What changed from the v0.1.2 ranking

| Config | v0.1.2 (BS = 128, **bug**) | v0.1.3 (BS = 256, correct) | Change |
|---|---:|---:|---|
| `TM = 2 TN = 4` | 789 (+10 %, "winner") | 738 (baseline) | still in winning cluster |
| `TM = 2 TN = 2` | 776 (+8 %) | 743 (+0.7 %) | now joint-winner |
| `TM = 4 TN = 2` | 716 (baseline) | 684 (−7.3 %) | dropped |
| `TM = 8 TN = 1` | 669 (−7 %) | 569 (−23 %) | dropped further |

The v0.1.2 numbers above are not directly comparable because they're
scoring a half-GEMM (cols `[32, 64)` of every output tile were
unwritten and counted as zero work in the elapsed time). What does
hold up is the **shape of the ranking**:

* `TM = 2` is still the right family.
* `TM = 8` is still the worst — the inner FMA loop is too serial.
* `TM = 4` was second-best at BS = 128 but is decisively beaten by
  `TM = 2` at BS = 256. Fewer rows per thread = more independent
  ALU work to fill the doubled workgroup with useful instructions.

---

## Pitfalls noted by the prompt — checked

1. **TM = 1 candidates `(1, 4)` / `(1, 8)`** — invalid; `TM / 2`
   integer division yields zero loop iterations. Kernel would compile
   silently and produce zero output. Skipped.
2. **Decode-side regressions** — none across all six configs.
3. **Tile-tuning robustness across `BLOCK_SIZE`** — confirmed for
   `TM = 2 TN = 4`. The v0.1.2 sweep result was directionally right
   even on the buggy GEMM.

---

## Tests

`cargo test --release` — **93 / 93 green** (24 lib + 44 correctness +
25 regression). No code change in this sub-phase.

## Files modified

| File | Change |
|---|---|
| `results/phase7_v013_tile_retune.md` | This file |
| `results/v013_retune_logs/*.log` | Raw bench logs (8 files) |

No source changes. `pipeline_registry.rs` already has
`unwrap_or(2)` / `unwrap_or(4)` for `VULKANFORGE_GEMM_TM` /
`VULKANFORGE_GEMM_TN` — that stays.

## Reproduce

```fish
for tm tn in 2 4 / 4 2 / 2 2 / 4 1 / 8 1 / 2 1
    set name "tm{$tm}_tn{$tn}"
    VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
    VF_NUM_PROMPTS=5 \
    VULKANFORGE_GEMM_TM=$tm VULKANFORGE_GEMM_TN=$tn \
        ./target/release/examples/run_15prompt_bench \
        | tee results/v013_retune_logs/$name.log
end
```
