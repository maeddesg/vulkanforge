# Phase 3A — Tiled Attention Shader

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`,
`subgroupSize = 64`, `SUBGROUP_FEATURE_ARITHMETIC_BIT`)
**Model:** Qwen3-8B Q4_K_M
**Result:** ✅ All Phase-2-baseline targets hit. Async-submit (Schritt 2)
not required per the prompt's gate (achieved > 50 tok/s at pos=200).
**Tests:** **35/35** pass (was 33; +2 tiled-attention regression tests).
**Validation:** 0 `VK_LAYER_KHRONOS_validation` errors.

---

## 1. Tiled Attention Shader

### 1.1 Approach

`vk_shaders/scalar_attn.comp` was rewritten **in-place** (same file
name, same `ShaderId::ScalarAttn`, same push-constant struct, same
4-binding layout). The entire forward dispatch path in `forward.rs`
is unchanged — Phase 3A is a single-file shader replacement.

The new shader is **94 lines** (vs the prompt's "< 200 lines" budget,
the old scalar one was 75). Algorithm:

```
local_size_x = 64        — exactly one Wave64 subgroup per workgroup
shared float scores[2048]  — 8 KiB LDS per workgroup (max_seq=2048)

Phase 1: scoring         (parallel over t, stride 64)
         → my_max collected per thread
Phase 2: subgroupMax     (single instruction, workgroup-wide)
Phase 3: exp(s − max)    (parallel over t, stride 64)
         + my_sum     →  subgroupAdd for ∑exp
Phase 4: weighted V-sum  (parallel over output dim d, stride 64)
         head_dim=128 / WGSIZE=64 → 2 d-values per thread
```

Two `barrier()` calls — between Phase 1 / Phase 3 (so Phase 1's
`scores[]` writes are visible) and between Phase 3 / Phase 4 (so
Phase 3's exp-overwrites are visible). Subgroup ops handle the
reductions in a single instruction each, no shared-memory tree
reduction needed.

### 1.2 Why workgroup_size = subgroup_size

Querying `VkPhysicalDeviceSubgroupProperties` on RADV/`gfx1201`
returns `subgroupSize=64`, with `minSubgroupSize=32` /
`maxSubgroupSize=64`. The default Wave64 means `local_size_x = 64`
gives exactly **one subgroup per workgroup**, so `subgroupAdd` and
`subgroupMax` ARE workgroup-wide reductions. No shared-memory tree
reduction or cross-subgroup merge is needed — that simplification
keeps the shader at 94 lines.

The pipeline still pins the spec constant explicitly via
`PipelineRegistry` (Phase-2A's "SPEC-DEFAULT-REGEL"); only `MAX_SEQ=2048`
is specialised, the workgroup size is hardcoded.

### 1.3 Push constants — unchanged

```rust
pub struct ScalarAttnPushConstants {
    pub n_heads: u32, pub n_kv_heads: u32, pub head_dim: u32,
    pub seq_len: u32, pub max_seq: u32, pub scale: f32,
}  // 24 B  — identical to Phase 2C
```

The prompt sketched an extended struct with q/k/v/o byte offsets;
those weren't necessary because `forward.rs` already slices per-layer
K/V via descriptor binding offsets (Phase 2C's offset-binding test
covers it). Keeping the struct identical means **zero changes** to
`pipeline.rs` and `forward.rs::run_scalar_attn`.

---

## 2. Correctness Tests

| Test                                              | Status | Note                                                    |
| ------------------------------------------------- | :----: | ------------------------------------------------------- |
| `test_scalar_attn_single_token`                   |   ✓    | Phase-2C drop-in regression                             |
| `test_scalar_attn_two_tokens`                     |   ✓    | Phase-2C drop-in regression                             |
| `test_scalar_attn_qwen3_dims_seq1`                |   ✓    | Phase-2C drop-in regression                             |
| `test_scalar_attn_qwen3_dims_with_binding_offset` |   ✓    | Phase-2C drop-in regression — slicing K/V per-layer     |
| **`test_tiled_attn_seq64_vs_cpu`**                |   ✓    | new — full wavefront; vs `f64`-reference CPU attention  |
| **`test_tiled_attn_seq200_vs_cpu`**               |   ✓    | new — uneven `seq_len % WGSIZE = 8`; ≈ Phase 2 pos=200  |
| `phase2c_forward_token_qwen3_finite_logits`       |   ✓    | full 36-layer forward still produces finite logits      |
| `phase2d_decode_produces_coherent_text`           |   ✓    | greedy decode still emits mutex-related keywords        |

**Passing the 4 pre-existing scalar_attn tests is the strongest
signal**: they exercise the same fixtures the Phase-2C debug walk
used to localise the original NaN bug, on the exact Qwen3 dimensions
(n_heads=32, n_kv_heads=8, head_dim=128) the forward dispatcher hits
in production. The new tiled shader is a true drop-in.

The two new tests use `f64`-internal CPU reference attention with the
GQA layout (`kvh = h / 4` for Qwen3) and check `max_abs_err < 1e-3`.
Threshold reflects the f32-vs-f64 rounding chain through softmax +
weighted-sum, not numerical instability.

Full suite:

```
running 7  tests   (lib unit)              7 passed
running 16 tests   (tests/correctness.rs) 16 passed   ← +2 vs Phase 2D
running 12 tests   (tests/regression.rs)  12 passed
                                          ──────────
                                          35 passed   (was 33 in Phase 2D)
```

---

## 3. Performance — Phase 2 baseline vs Phase 3A

### 3.1 Comparison table (decode profile, Qwen3-8B, RX 9070 XT)

| Metric                | Phase 2     |  Phase 3A    | Improvement |
| --------------------- | ----------: | -----------: | ----------: |
| **pos=0**             |             |              |             |
| Forward (ms)          | 16.7        | 16.9         | ≈1.0×       |
| Decode (tok/s)        | 59.8        | 59.0         | ≈1.0×       |
| `scalar_attn` (µs)    |   834       |   468        | **1.78× faster** |
| `scalar_attn` (% GPU) |  6.1 %      |  3.5 %       |             |
|                       |             |              |             |
| **pos=50**            |             |              |             |
| Forward (ms)          | 34.7        | 15.6         | **2.23× faster** |
| Decode (tok/s)        | 28.8        | 64.2         | **2.23× faster** |
| `scalar_attn` (µs)    | 19 879      |  1 390       | **14.30× faster** |
| `scalar_attn` (% GPU) | 63.6 %      | 10.7 %       |             |
|                       |             |              |             |
| **pos=100**           |             |              |             |
| Forward (ms)          | 51.9        | 16.2         | **3.20× faster** |
| Decode (tok/s)        | 19.3        | 61.6         | **3.19× faster** |
| `scalar_attn` (µs)    | 36 542      |  2 608       | **14.01× faster** |
| `scalar_attn` (% GPU) | 76.4 %      | 18.5 %       |             |
|                       |             |              |             |
| **pos=200**  ← target |             |              |             |
| Forward (ms)          | 127.3       | **18.2**     | **6.99× faster** |
| Decode (tok/s)        | 7.9         | **55.0**     | **6.96× faster** |
| `scalar_attn` (µs)    | 112 336     |  **4 576**   | **24.55× faster** |
| `scalar_attn` (% GPU) | 90.8 %      | 29.1 %       |             |
| `scalar_attn` vs pos=0 | 134.57×    | **9.77×**    | growth flattened ~14× |

**Targets vs achieved (per prompt §1.7):**

| Target                                  | Required        | Achieved             | Status |
| --------------------------------------- | --------------- | -------------------- | :----: |
| `scalar_attn` < 5 ms at pos=200         | < 5 ms          | **4.58 ms**          |   ✅   |
| Forward 127 ms → ~20 ms at pos=200       | ~20 ms          | **18.2 ms**          |   ✅   |
| ~50 tok/s at pos=200                    | ≥ 50            | **55.0**             |   ✅   |
| Speedup ≥ 10× (lower abort bound)       | ≥ 10×           | **24.5×** at pos=200 |   ✅   |

The prompt's *expected* range was 40–100×; we achieved 24.5×. The
remaining linear-in-`seq_len` factor in the new shader is **Phase 4's
inner t-loop** — V-sum is parallelised over `d` but still sequential
over `t` (each thread reads `scores[t]` × `seq_len` times across the
inner loop). Eliminating that residual would require a different
tile shape (e.g. flash-attention-style chunked V-accumulation) and is
out of scope for Phase 3A (prompt §1.4 explicitly defers
flash-attention to later phases).

### 3.2 What's now the bottleneck (post-fix profile, pos=200)

```
Forward 18.2 ms · GPU sum 15.7 ms · dispatch overhead 2.45 ms (13.5 %)
  scalar_attn   29.1 %    ← was 90.8 %
  GEMV total    65.0 %    ← was 8.5 %  (constant ~10.3 ms, now visible)
  Rest           5.7 %
  KV-write       0.2 %
```

**GEMV is the new dominant cost** — exactly as the Phase-2 profiling
report predicted: GEMV is constant ~10.3 ms / forward, so once
attention stops drowning it out, it surfaces as 65 % of GPU time at
pos=200. Phase 1 measured it at 79.6 % of peak BW already, so further
GEMV tuning has a ~20 % ceiling. **Async-submit / pipelined dispatch
becomes the next attainable lever** — at 2.45 ms, the dispatch
overhead is now 13.5 % of forward wall, and shrinking it pushes
toward 60-65 tok/s at pos=200.

### 3.3 5-Prompt Validation Suite (vs Phase 2D)

Same 5 prompts as Phase 2D's validation; all decoded coherently.
Run logged in `results/phase3a_validation_run.log`.

| Prompt                                                | Phase 2D  | Phase 3A | Speedup |
| ----------------------------------------------------- | --------: | -------: | ------: |
| Explain what a mutex is in one sentence.              |  13.1     | **66.8** |  5.10×  |
| Write a haiku about programming.                      |  13.6     | **66.8** |  4.91×  |
| What is 2 + 2?                                        |  13.4     | **67.3** |  5.02×  |
| Translate 'hello world' to German.                    |  13.4     | **67.0** |  5.00×  |
| List three prime numbers.                             |  13.7     | **66.8** |  4.88×  |
| **MEDIAN decode tok/s**                               | **13.4**  | **66.8** | **4.99×** |
| MEDIAN prefill tok/s                                  |    56     |    79    |  1.41×  |

End-to-end user-visible decode is now **5.0× faster**, and we hit
**58.6 % of llama.cpp Vulkan's 114 tok/s reference** (was 11.8 %).
All 5 prompts produced the same on-topic English content as in Phase
2D — no regression in coherence or chat-template handling.

---

## 4. Async Submit — Skipped per Prompt's Gate

> §2 from the prompt:
> "Schritt 2 — Async Submit (**Optional, nur falls Schritt 1 < 50 tok/s
> bei pos=200**)"

Achieved decode at pos=200 = **55.0 tok/s ≥ 50 tok/s** → gate not
triggered, async submit skipped. The current dispatch overhead
(2.45 ms / 13.5 % at pos=200) is the natural starting point for
Phase 3B.

---

## 5. Files changed

| File                                              | Change |
| ------------------------------------------------- | ------ |
| `vk_shaders/scalar_attn.comp`                     | rewrite — 94-line tiled implementation |
| `tests/correctness.rs`                            | +2 tests (`test_tiled_attn_seq{64,200}_vs_cpu`) and a `cpu_attention_reference` helper |
| `results/phase3_step_3a_attention_fix.md`         | new — this report |
| `results/phase3a_profile_run.log`                 | new — full per-position breakdown stdout |
| `results/phase3a_validation_run.log`              | new — 5-prompt suite stdout |

**Untouched:**
- `forward.rs` (no dispatch changes — same `run_scalar_attn` calls)
- `pipeline.rs` (`ScalarAttnPushConstants` unchanged at 24 B)
- `pipeline_registry.rs`, `shaders.rs`, `device.rs`
- All other shaders

---

## 6. Bekannte Fallstricke from the prompt — actual outcomes

| Pitfall                                       | Outcome |
| --------------------------------------------- | ------- |
| `subgroupSize` ≠ 64                          | Verified `subgroupSize = 64` via `vulkaninfo` before writing the shader; matches `local_size_x = 64`. |
| LDS budget for `scores[2048]` (8 KB)          | Fits — RDNA 4 has 128 KB LDS / CU. Even with 32 workgroups co-resident: 256 KB total, well within budget. |
| Bank conflicts                                | None — each thread reads/writes its own stride-64 slot of `scores[]` in Phase 1 / Phase 3; in Phase 4 all 64 threads broadcast-read the same `scores[t]`, which is the no-conflict broadcast path. |
| Numerical stability (max-trick softmax)       | Implemented via `subgroupMax` before `exp(s − max)`. CPU-vs-GPU `max_abs_err < 1e-3` for both seq=64 and seq=200. |
| `seq_len < 64`                                | Threads with `t >= seq_len` simply skip the loop (Phase 1 / Phase 3) and contribute `my_max=-1e30` / `my_sum=0` to the reductions — neutral, no UB. The four pre-existing Phase-2C scalar_attn tests with `seq_len = 1` and `seq_len = 2` pass unchanged. |
| Output reduction over 128 dims                | Avoided — Phase 4 parallelises **over** `d` instead of over `t`, so no per-element subgroupAdd. Each thread writes its own `o[d]` directly. |
| Pipeline-Registry update                      | Not needed — same `ShaderId`, same SPIR-V slot, same descriptor layout. |

---

## 7. Acceptance gates

| Gate                                                     | Status |
| -------------------------------------------------------- | :----: |
| 35/35 tests green (33 pre-existing + 2 new tiled-attn)   |   ✅   |
| 5/5 prompts coherent in validation suite                 |   ✅   |
| Decode tok/s ≥ 50 at pos=200                             |   ✅ (55.0) |
| `scalar_attn` < 5 ms at pos=200                          |   ✅ (4.58 ms) |
| Forward at pos=200 close to ~20 ms                        |   ✅ (18.2 ms) |
| 0 validation errors                                      |   ✅   |
| No change to `forward.rs` dispatch path                   |   ✅   |
| Phase-2 regression tests still pass                      |   ✅   |

---

## 8. Commit hash

To be filled in by the commit at the end of this run.
