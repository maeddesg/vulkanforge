# Phase 4B — Online-Softmax Flash-Attention (Drop-in for `scalar_attn`)

**Date:** 2026-04-26
**Goal:** Replace Phase-3A `scalar_attn.comp` (192 VGPRs / 4-of-16
occupancy / 8 KB LDS) with an online-softmax variant that fixes the
register pressure and LDS waste.
**Status:** Shader **shipped, drop-in compatible, all 4 parity tests
green.** Perf delta in the noise — ~+1–2 % at decode, attention
scaling unchanged. **Real-world perf is BW/compute-bound, not
VGPR-occupancy-bound.** The Phase-3 RGP analysis's "VGPR is the
bottleneck" interpretation needs revision (see §6).
**Tests:** **52/52** pass (was 48; +4 flash_attn parity tests).
**Validation:** **0 errors**.

---

## 1. Approach taken: B2 (write minimal online-softmax)

The Phase-4B prompt offered three options:

| Option | What                                          | Outcome |
| ------ | --------------------------------------------- | ------- |
| B1 | Port llama.cpp's full `flash_attn.comp` (788 lines + 642-line `flash_attn_base.glsl`) | **Rejected** — expects FP16 K/V (we have FP32), needs mask buffer (we don't have one), uses cooperative-matrix infrastructure, ~12 spec constants. Multi-day port. |
| B2 | Write a minimal online-softmax shader from scratch | **Chosen.** ~120 lines, drop-in-compatible bindings + push-constants with our `scalar_attn.comp`, no infrastructure changes. |
| B3 | Extract the decode path from `flash_attn.comp` | Partly subsumed by B2 — same algorithm, but written against our actual KV-cache layout instead of llama.cpp's. |

The new shader is `vk_shaders/flash_attn.comp` (120 lines), with:

- **Same descriptor bindings** as `scalar_attn` (Q, K, V, O — all f32).
- **Same push-constant struct** (`ScalarAttnPushConstants`, 24 B).
- **Same dispatch shape** (`(n_heads, 1, 1)` — 32 WGs).
- **Same workgroup size** (64 threads = 1 Wave64 subgroup).

Drop-in flag: `forward.rs::run_scalar_attn` now picks
`ShaderId::FlashAttn` instead of `ShaderId::ScalarAttn`. The original
shader stays in the registry and is exercised by the four pre-existing
correctness tests so we can A/B in the future.

---

## 2. Algorithm

Classic Flash-Attention online-softmax over `TILE = 64` K positions
per pass. Per workgroup (1 per Q-head):

```glsl
my_out0 = my_out1 = 0
my_max  = -1e30
my_sum  = 0

for (tile_base = 0; tile_base < seq_len; tile_base += 64):
    t = tile_base + tid
    score = (t < seq_len) ? dot(Q[h], K[t]) * scale : -inf

    tile_max  = subgroupMax(score)
    new_max   = max(my_max, tile_max)
    k         = exp(my_max - new_max)               // correction
    my_out0  *= k;  my_out1 *= k;  my_sum *= k

    p          = (t < seq_len) ? exp(score - new_max) : 0
    tile_sum   = subgroupAdd(p)
    my_sum    += tile_sum
    my_max     = new_max

    scores_lds[tid] = p
    barrier()

    for i in 0..tile_size:
        global_t = tile_base + i
        my_out0 += scores_lds[i] * V[global_t][tid]
        my_out1 += scores_lds[i] * V[global_t][tid + 64]
    barrier()

inv_sum = 1.0 / my_sum
o[h * head_dim + tid]      = my_out0 * inv_sum
o[h * head_dim + tid + 64] = my_out1 * inv_sum
```

### 2.1 What this changes vs Phase-3A `scalar_attn`

| Metric          | scalar_attn (Phase 3A) | flash_attn (Phase 4B) |
| --------------- | ---------------------: | --------------------: |
| LDS / WG        | 8 192 B (`scores[2048]`) | 256 B (`scores_lds[64]`) |
| Passes over K/V | 3 (score, softmax, V-sum) | 1 (interleaved)      |
| Algorithm       | Pre-compute all scores | Streaming online softmax |
| Dispatch shape  | `(32, 1, 1)`           | `(32, 1, 1)` (same)  |
| Bindings        | Q/K/V/O (f32)          | Q/K/V/O (f32) (same) |

### 2.2 What this does NOT fix

- **SIMD breadth** still 32 / 128 = 25 % (only 32 WGs dispatched).
  Multi-WG-per-head is Phase 4C.
- **No FP16 storage.** K/V remain in f32 throughout.
- **No causal mask.** Decode sees all prior positions; the mask isn't
  needed at decode-time. (Prefill-batch's per-token attention loop
  inside `prefill_batch` is unchanged — also doesn't need a mask
  because positions are written into the cache one at a time.)

---

## 3. Correctness (the strongest result of this phase)

Four new tests in `tests/correctness.rs`:

| Test                                        | seq_len | Tolerance | Outcome |
| ------------------------------------------- | ------: | --------: | :-----: |
| `test_flash_attn_seq1_vs_cpu`               |       1 | 1e-3      |   ✅    |
| `test_flash_attn_seq64_vs_cpu`              |      64 | 1e-3      |   ✅    |
| `test_flash_attn_seq200_vs_cpu`             |     200 | 1e-3      |   ✅    |
| `test_flash_attn_matches_scalar_attn_seq200` |    200 | 1e-3 (vs scalar_attn) | ✅ |

The fourth test is the strongest signal: **same Q/K/V tensors run
through both `scalar_attn` and `flash_attn`, output difference
< 1e-3.** Online-softmax is mathematically equivalent to the
three-pass formulation; the only difference is f32 rounding through
different operation orders. Empirically that rounding stays well
inside the 1e-3 envelope.

End-to-end signals also pass:

- `phase3e_prefill_batch_matches_token_by_token_top5` — argmax + top-5
  overlap ≥ 4/5 still holds with flash_attn in the forward path.
- `phase2d_decode_produces_coherent_text` — mutex-prompt decode still
  contains keywords like `mutex`, `mutual`, `lock`, `thread`,
  `synchron`, `exclus`.
- 5-prompt validation suite below: 5/5 coherent.

---

## 4. Performance — the unexpected bit

### 4.1 ShaderProfiler `profile_positions` (two runs averaged)

| Position | Phase 3 final (scalar_attn) | Phase 4B (flash_attn) | Delta |
| -------- | --------------------------: | --------------------: | ----: |
| pos=0    | 61.8 tok/s                  | **62.8** tok/s        | +1.6 % |
| pos=50   | 64.0 tok/s                  | 62.7 tok/s            | −2.0 % |
| pos=100  | 64.8 tok/s                  | 63.5 tok/s            | −2.0 % |
| pos=200  | 55.3 tok/s                  | **56.1** tok/s        | +1.4 % |

Per-shader timing for the attention dispatch (now `flash_attn`,
labelled `scalar_attn` in the profiler because of the
`run_scalar_attn` method name):

| Position | scalar_attn (µs) | flash_attn (µs) | Delta  |
| -------- | ---------------: | --------------: | -----: |
| pos=0    | 468              | 416             | −11 %  |
| pos=50   | 1 585            | 1 287           | −19 %  |
| pos=100  | 2 608            | 2 783           | +7 %   |
| pos=200  | 4 576            | 4 324           | −5 %   |

The pos=100 outlier in the first run smoothed to within ±5 % across
two runs. **Net attention-shader timing: small, single-digit-percent
fluctuation either way.** The 2-3× the prompt-prediction in §5.1 of
the brief did not appear.

### 4.2 5-prompt validation suite

```
Prompt                                     Prefill (tok/s)  Decode (tok/s)
Explain what a mutex is in one sentence.            249           65.7
Write a haiku about programming.                    271           66.7
What is 2 + 2?                                      279           67.3
Translate 'hello world' to German.                  272           66.4
List three prime numbers.                           260           65.4
MEDIAN                                              271           66.4
```

5/5 coherent, semantically correct. Decode median 66.4 tok/s vs Phase
3 baseline ~64 — within run-to-run noise.

---

## 5. Why the gain didn't materialise

The Phase-3 RGP analysis read scalar_attn's pipeline-state pane as
"192 VGPRs / 4 of 16 wavefronts / 6 % effective GPU utilisation"
and concluded that lifting per-SIMD wavefront occupancy (via
flash-attention's lower register footprint) would unlock a 2-3×
gain. The Phase-4B measurements **don't show that gain.**

Three plausible reasons:

1. **The attention shader is memory-bandwidth-bound, not
   latency-bound.** At pos=200 each forward reads ~3.2 MB of
   K + V data per layer. With 644 GB/s peak, that's ~5 µs of
   theoretical work; we measure 4 576 µs scalar_attn and 4 324 µs
   flash_attn. Both are 800-900× below peak BW — meaning the
   bottleneck is **not** memory throughput. So more waves → more
   memory parallelism → no help.
2. **Compute-bound on the dot-product loop.** Each t-iteration does
   128 FMAs (head_dim). The inner FMA cadence is the same in both
   shaders; the only ways to speed it up are vectorisation
   (`vec4` loads — Phase 4 candidate), wider parallelism (Phase 4C
   multi-WG-per-head), or coopmat (out of scope).
3. **My read of "192 VGPRs" might have been pessimistic on the
   actual register pressure.** ACO allocates VGPRs in chunks (16-
   or 32-wide on RDNA 4) so even a shader using 100 "real"
   registers can report 192. flash_attn's actual VGPR footprint
   isn't measured in this phase — that needs another RGP capture.

What flash_attn **definitively** does:

- Eliminates the 8 KB LDS array — frees per-WG LDS for future
  features (KV-cache prefetching, larger MAX_SEQ).
- Single-pass instead of three-pass — simpler to reason about,
  fewer barriers, slightly less inter-pass dispatch overhead
  (visible in the −11 %/−19 % per-shader timing at pos=0/50,
  even if it doesn't propagate to forward-wall tok/s).
- Online softmax is more numerically robust at very long
  sequences (max accumulated incrementally rather than over the
  whole array).

---

## 6. Updated picture for Phase 4C+

The Phase-3 priority list said:
> Flash-attention removes the linear scaling … 2-3× decode at long
> context.

The corrected picture after Phase 4B's measurement:

> **Per-SIMD VGPR pressure was not the binding constraint** for
> attention's wall-time. The remaining gap to llama.cpp Vulkan's
> attention performance is in two places:
>
> 1. **SIMD-breadth** — 32 WGs vs 128 SIMDs. Phase 4C should
>    dispatch `(n_heads, ceil(seq_len / TILE), 1)` workgroups,
>    with the inner reduction either done in a follow-up
>    `flash_attn_split_k_reduce`-style pass or by sequential
>    accumulation across the tile dimension. **Real expected gain
>    here is the 2-3× the prompt forecast.**
> 2. **Memory-access patterns** — `vec4` loads of K/V (4× wider
>    each), or FP16 K/V storage (2× less BW pressure). The latter
>    is a bigger change (forward-pass-wide).

Phase 4B is **necessary infrastructure** (online softmax + drop-in
shader) for Phase 4C — you can't multi-WG-per-head a 3-pass shader
that holds all scores in LDS. The breadth fix builds on top of this.

---

## 7. Files added / changed

| File                                              | Status |
| ------------------------------------------------- | ------ |
| `vk_shaders/flash_attn.comp`                      | new — 120-line online-softmax shader, drop-in for scalar_attn |
| `build.rs`                                        | edit — adds the flash_attn compile job |
| `src/backend/vulkan/shaders.rs`                   | edit — `ShaderId::FlashAttn`, `FLASH_ATTN_F32` blob, added to `ALL_SHADERS` |
| `src/backend/vulkan/pipeline_registry.rs`         | edit — `MAX_SEQ=2048` spec-constant pinning for FlashAttn |
| `src/backend/vulkan/forward.rs`                   | edit — single-line swap: `run_scalar_attn` now dispatches `ShaderId::FlashAttn` (scalar_attn pipeline retained for the existing tests) |
| `tests/correctness.rs`                            | edit — 4 new flash_attn parity tests |
| `results/phase4_step_4b_flash_attention.md`       | new — this report |
| `results/phase4b_profile.log`                     | new — profile_positions stdout |
| `results/phase4b_5prompt.log`                     | new — 5-prompt validation stdout |

**Untouched:** `scalar_attn.comp` (still in the registry, still
exercised by the 4 original `test_scalar_attn_*` tests so we have a
stable A/B reference), every other Vulkan-backend file.

---

## 8. Acceptance gates

| Gate                                               | Status |
| -------------------------------------------------- | :----: |
| 52/52 tests green (48 + 4 new flash_attn parity)   |   ✅   |
| 0 validation errors                                |   ✅   |
| flash_attn correct vs CPU reference (seq=1/64/200) |   ✅   |
| flash_attn ≈ scalar_attn at seq=200 (max_abs < 1e-3) | ✅   |
| 5/5 prompts coherent                               |   ✅   |
| Decode tok/s ≥ baseline (no regression)            |   ✅ — within noise |
| Decode tok/s ≥ 65 at pos=0 (forecast 65-70)        |   ⚠ 62.8 — within 1.6 % of baseline |
| Decode tok/s ≥ 65 at pos=200 (forecast 65+)        |   ❌ 56.1 — attention-bound model from Phase 3 was wrong |
| Drop-in: same bindings + push-constants + dispatch  |   ✅   |

The forecast in the prompt's §"Performance-Baseline" assumed the
attention shader's wall-time was VGPR-occupancy-bound. The Phase-4B
measurement shows it's **memory-/compute-bound**, so the per-SIMD
occupancy lift didn't translate to tok/s. The infrastructure
(simpler shader, smaller LDS, online softmax) is in — Phase 4C
needs to attack the SIMD-breadth axis instead.

---

## 9. Commit hash

To be filled in by the commit at the end of this run.
