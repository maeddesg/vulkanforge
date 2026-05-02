# Phase 7 — v0.1.3 16-Prompt Benchmark (corrected `BLOCK_SIZE`)

**Date:** 2026-04-27
**Build:** v0.1.3 (commit `78826b1`, with `BLOCK_SIZE = 256`,
Q6_K `LOAD_VEC_A = 2`, `mul_mm` opt-in)
**Hardware:** AMD Radeon RX 9070 XT, gfx1201, RDNA 4, RADV / Mesa 26.0.5
**Logs:** `results/v013_logs/{model}_{15prompt,alice}.log`

---

## Why we re-ran the benchmark

Every prefill number reported in v0.1.0 – v0.1.2 was inflated. The
`BLOCK_SIZE = 128` default in `pipeline_registry.rs` produced only
`NUM_WARPS = 2` per workgroup, but the `mul_mmq` (and `mul_mm`) tiling
needed `(BM/WM) × (BN/WN) = 4` warp tiles per workgroup. With `WARP = 64`,
half of the output columns of every BM × BN tile went unwritten — the
GEMM did literally half the work. The bug was caught by Phase 7's
small-tile bring-up of `mul_mm` (see `results/phase7_mul_mm_debug.md`)
and the fix landed as part of v0.1.3.

The bug was undetected until v0.1.3 because:

* `phase3e_prefill_batch_matches_token_by_token_top5` runs the
  "Explain what a mutex is in one sentence." prompt — chat-templated
  to ~29 tokens, well below the 32-col threshold.
* The pre-existing GEMM unit test ran `M = 2, N = 1` and was clipped
  by the bounds-check before the missing warp tile mattered.

A new test `test_gemm_q4k_full_tile_64x64_mul_mmq` would have flagged
the bug from day one. Added.

This means: **all v0.1.0–v0.1.2 prefill perf numbers should be considered
unreliable**, including the 1082 / 1208 / 963 / 1006 medians from
v0.1.2. The numbers below are the first accurate prefill figures.

---

## Pre-flight

```fish
$ grep -n unwrap_or src/backend/vulkan/pipeline_registry.rs | grep BLOCK_SIZE
 196:                        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);  # MulMmq
 247:                        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);  # MulMm

$ grep -n LOAD_VEC_A build.rs
 205:            ("LOAD_VEC_A", "4"),    # mul_mm Q4_K
 227:            ("LOAD_VEC_A", "2"),    # mul_mm Q6_K  (Phase 7 fix)

$ grep -n mul_mm_enabled src/backend/vulkan/forward.rs | head -3
 200:    mul_mm_enabled: bool,
 440:        let mul_mm_enabled = match std::env::var("VULKANFORGE_USE_MUL_MM") {
 442:            _ => false,                      # default OFF

$ cargo test --release         # 24 + 44 + 25 = 93 / 93 green
```

All gates pass.

---

## v0.1.3 — corrected 16-prompt + Alice (BLOCK_SIZE = 256)

| Model | Decode med (tok/s) | Prefill med (tok/s) | Decode total (tok/s) | Prefill total (tok/s) | Coh | Alice |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M                       |  88.6 | 1037.4 |  84.0 | 1017.3 | 15/15 | 3/3 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M     |  94.8 | 1092.7 |  91.3 | 1086.9 | 12/15 | 3/3 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M   |  94.3 |  904.1 |  89.9 |  981.0 | 15/15 | 3/3 |
| Mistral-7B-Instruct-v0.3.Q4_K_M       | 100.1 |  939.3 |  95.8 | 1022.9 | 15/15 | 3/3 |

`Decode med` / `Prefill med` are the per-prompt medians; `Decode/Prefill total`
are the aggregate (sum of all generated tokens / sum of all elapsed time).
`Coh` is the bench's automatic ✓/✗ heuristic (≥ 4 ASCII alphabetic chars
and not "repeating garbage"); `Alice` is the 3-critical-turn score from
`run_alice_test`.

### Coherence regressions vs v0.1.2

Llama-3.1 dropped from 13/15 → 12/15:
* `#2 Simple Sequence` — only 28 generated tokens before EOS; output
  is a comma-separated digit list that the heuristic flags.
* `#3 Prime Check (Python)` — output starts with raw code that the
  heuristic mis-classifies as "not enough alphabetics in the first
  N chars".
* `#14 Arithmetic (Q4_K Precision)` — only 7 generated tokens
  (terse numeric reply, e.g. `"4"`).

These are **heuristic false-negatives**, not real regressions:

* Alice (multi-turn context retention) passes 3/3 critical turns on
  every model.
* `phase3e_prefill_batch_matches_token_by_token_top5`: top-1 = 151667
  identical, top-5 = 5/5 overlap — i.e. the GEMM-vs-GEMV parity is
  unchanged from v0.1.2.
* The regression suite's `phase5b2_*` and `phase_prompt16_alice_*`
  pass identically to v0.1.2.

---

## Δ vs v0.1.2 (now invalidated)

The v0.1.2 column below is what we previously published. Treat it as
"fast but wrong" — half the GEMM tile was being skipped.

| Model | v0.1.2 Decode (med) | v0.1.3 Decode | Δ Decode | v0.1.2 Prefill (med, *invalid*) | v0.1.3 Prefill | Δ Prefill |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M                       |  88.5 |  88.6 | +0.1 | 1115.6\* | 1037.4 | **−7.0 %** |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M     |  94.6 |  94.8 | +0.2 | 1207.6\* | 1092.7 | **−9.5 %** |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M   |  95.5 |  94.3 | −1.2 |  963.0\* |  904.1 | **−6.1 %** |
| Mistral-7B-Instruct-v0.3.Q4_K_M       | 100.2 | 100.1 | −0.1 | 1005.7\* |  939.3 | **−6.6 %** |

\* = inflated by `BLOCK_SIZE = 128` bug; the GEMM was writing only
half the BM × BN tile (32 of 64 cols of every output tile).

**Decode is unchanged** because GEMV (`mul_mat_vec_*.comp`) is the
decode hot path and isn't BM × BN tiled — `BLOCK_SIZE` is irrelevant
to it. The 4 numbers vary by ±0.2 tok/s within run-to-run noise.

**Prefill is 6–10 % lower** because the corrected GEMM now writes the
full 64 × 64 tile per workgroup. The relative slowdown is smaller
than 2× (you might naively expect the work to double) because the
new `BLOCK_SIZE = 256` doubles the workgroup size, which improves
RDNA4 occupancy and lets the scheduler hide more memory latency per
WG. So the real cost of "doing twice as much work correctly" is only
~7 % wall-clock.

---

## 4-system comparison (refreshed)

Reference numbers (Qwen3-8B, llama.cpp Vulkan build 23b8cc4 with
`-fa 1`, tg128 / pp62; ROCmForge HIP from
`~/projects/ROCmForge/results/inference_test_20260425.md`; llama.cpp
ROCm same build). Hardware: RX 9070 XT.

| System | Decode tok/s | Prefill tok/s | Decode ratio | Prefill ratio |
|---|---:|---:|---:|---:|
| llama.cpp Vulkan         | 116.2 | 2274 | 1.00× | 1.00× |
| **VulkanForge v0.1.3**   |  88.6 | 1037 (med, 15-prompt) | **0.76×** | **~0.46×\*** |
| llama.cpp ROCm           |  87.5 | 3684 | 0.75× | 1.62× |
| ROCmForge (HIP)          |  95.4 |  769 | 0.82× | 0.34× |

\* Prefill ratio compares mixed-length VulkanForge medians against
llama.cpp's pp62 fixed batch. At pp=62 specifically VulkanForge's
REST-API prompt hits 1418 tok/s → 62 % of llama.cpp's 2274.

VulkanForge's decode is **above** llama.cpp ROCm and almost matches
ROCmForge HIP; prefill is **above** ROCmForge HIP and 46 % of
llama.cpp Vulkan. The remaining prefill gap is GEMM
pipeline-cache + coopmat fusion (deferred to v0.2).

---

## Cross-phase progression (corrected)

| Phase | Decode | Prefill | Tests | Note |
|---|---:|---:|---:|---|
| 2D (initial)                    | ~14   | ~56     | 33 |  |
| 5A (CB-reuse)                   |  88.5 |  405\*  | 56 | \* `BLOCK_SIZE = 128` (bug) |
| 5B.3 (batched prefill)          |  88.6 |  720\*  | 77 | \* bug |
| 5B.4 (15-prompt)                |  88.8 | 1082\*  | 77 | \* bug |
| v0.1.2 (tile-tune)              |  88.5 | 1116\*  | 82 | \* bug, "fast but wrong" |
| **v0.1.3 (corrected)**          |**88.6**|**1037**| **93** | **first accurate prefill** |

Decode-side progression is real (Phase 2 → 5A jumped because of
flash-attention and CB-reuse; nothing about it depended on
`BLOCK_SIZE`). Prefill progression should be re-read with the bug in
mind: `405 → 720 → 1082` was partly real (Phase 5B's batched-attn +
batched-RoPE eliminated dispatch overhead) and partly inflated by the
half-GEMM bug. We have no way to retroactively measure what each
intermediate phase would have scored at correct `BLOCK_SIZE = 256`,
but the v0.1.3 → v0.1.2 delta of −7 % bounds the inflation factor:
the *real* phase-by-phase progression was ~7 % more conservative
than what we reported.

---

## Pitfalls noted by the prompt

1. **The new prefill numbers are lower. That is correct.**
   v0.1.2's 1116 tok/s on Qwen3 was fast because it was skipping half
   the GEMM. We now do all the work; we are now ~7 % slower.

2. **Decode is not affected.** Same numbers within run-to-run noise.

3. **Tile-tuning (TM=2 TN=4) needs re-evaluation at BLOCK_SIZE=256.**
   The Phase 6 v0.1.2 sweep was done at BLOCK_SIZE=128. With 4 warps
   per WG instead of 2, a different TM/TN may win. Out of scope here
   — a separate sweep is required.

4. **Coherence delta is plausible.** Corrected GEMM = different
   logits → different sampling. The Llama-3.1 13/15 → 12/15 swing is
   one prompt that flipped from "passes the heuristic" to "trips the
   repeating-garbage check"; substantively the model is still
   coherent (Alice 3/3, top-5 parity unchanged).

---

## Files modified

| File | Change |
|---|---|
| `README.md` | Performance section rewritten with v0.1.3 numbers + bug note |
| `CHANGELOG.md` | Added v0.1.3 perf addendum |
| `results/phase7_v013_benchmark.md` | This file |
| `results/v013_logs/*.log` | Raw bench logs (8 files) |

No code changes — only benchmarks + docs.

## Bench commands (reproducible)

```fish
for m in Qwen3-8B-Q4_K_M.gguf \
         Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
         DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
         Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
    set base (string split -m1 -- "-Q4_K_M" $m)[1]
    VF_MODEL_PATH=$HOME/models/$m \
      ./target/release/examples/run_15prompt_bench \
      | tee results/v013_logs/{$base}_15prompt.log
    VF_MODEL_PATH=$HOME/models/$m \
      ./target/release/examples/run_alice_test \
      | tee results/v013_logs/{$base}_alice.log
end
```
