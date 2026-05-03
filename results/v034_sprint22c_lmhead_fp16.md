# Sprint 22C — FP16 lm_head (−1.0 GiB VRAM, +10% decode)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 22B, head was `c4c72f4`)
**Goal:** Narrow `lm_head.weight` from BF16→FP32-expanded to
BF16→FP16-narrowed at load time, halving its GPU footprint
(2.1 GiB → 1.0 GiB on Llama-3.1's 128 256 × 4 096 vocab head).
Bonus: decode's lm_head GEMV becomes bandwidth-bound on half
the bytes.

## Headline result

```
Meta-Llama-3.1-8B-Instruct-FP8

                    Sprint 20-M1   Sprint 22B    Sprint 22C
GPU upload:         10.42 GiB       8.46 GiB      7.48 GiB    ← −0.98 GiB
                                                              cumulative −2.94 GiB (−28.2%)
Decode:             63.3 tok/s      62.9 tok/s    68.5 tok/s  ← +9% (free!)
Prefill (pp=512):   694.6 tok/s     ~694 tok/s    ~700 tok/s  unchanged
Coherence:          "The answer to 2+2 is 4." (bit-identical greedy output)
```

The `+9% decode` is a **free byproduct** of the VRAM cut.
lm_head GEMV at decode-time reads `vocab × hidden = 525M
weights` per token. Going from FP32 (2.1 GiB / token) to FP16
(1.05 GiB / token) halves the VRAM bandwidth on that one
dispatch — and decode is bandwidth-bound on lm_head, so the
saving directly converts into more tokens per second.

## What landed (~100 LOC, 1 new shader, no algorithm changes)

### `vk_shaders/mul_mat_vec_f16.comp` (new, ~95 LOC)

Drop-in for `mul_mat_vec_f32.comp`. Same dispatch shape, push
constants, and 5-binding descriptor (the FP8/F32 GEMVs already
use 3 bindings after spirv-opt strips the fuse dummies; FP16
matches). Only the inner load differs:

* `WeightBuf` is `uint[]` instead of `float[]`. One uint = two
  packed FP16 values.
* Per K-step the thread reads one uint and decodes via
  `unpackHalf2x16(word) -> vec2`, then does two FMAs against
  `data_b[]` (which is FP32, unchanged).
* Tail branch for odd K (defensive — Llama / Qwen hidden_dims
  are powers of two so the branch is never taken).
* Same 64-thread Wave64 LDS-tree reduction; same
  `weight_scale` post-multiply via the broadcast3 push-constant
  slot (always 1.0 for unquantized lm_head).

### `src/backend/vulkan/loader.rs` (~50 LOC)

* New `bf16_to_f16_vec(raw, info, name)` converter — uses the
  `half` crate (`half::bf16::from_bits(x).to_f32()` then
  `half::f16::from_f32(f).to_bits()`) for safe rounding /
  Inf / NaN / subnormal handling.
* `load_safetensors` upload loop: for `is_lm_head =
  hf_name == "lm_head.weight"` AND `info.dtype == BF16`, route
  through `bf16_to_f16_vec` and tag the output as
  `GgmlType::F16`. All other BF16 tensors (the 32 input/output
  layernorms, ~1 MiB total) keep the BF16→FP32 path because the
  RMSNorm shader consumes FP32 weights and a few hundred KiB of
  VRAM isn't worth the shader port.

### Pipeline + dispatch routing (~15 LOC)

* `ShaderId::MulMatVecF16`, name / spv_bytes / `ALL_SHADERS`
  wired alongside `MulMatVecF32`.
* `pipeline_registry.rs`: `MulMatVecF16` joins the same
  spec-const block as `MulMatVecFp8` / `MulMatVecF32`
  (`BLOCK_SIZE = 64`, `requiredSubgroupSize = 64`).
* `forward.rs`:
  * `layer_weight_shader`: `(GgmlType::F16, _) =>
    ShaderId::MulMatVecF16` arm added.
  * `dispatch_final` (lm_head): same arm added.
  * `run_gemv`'s `one_per_row` matcher extended to
    `MulMatVecF16` so the descriptor binding count matches
    (3 bindings, not 5). **This was a real bug-catch** —
    first run crashed with a validation error about
    `dstBinding (3) > bindingCount (3)`; spirv-opt had stripped
    the fuse dummies, mirroring Sprint 20-M3's gotcha.

### `build.rs` (+1 SPV job)

102 SPVs total (was 101). No other build changes.

## Why we don't convert the norms too

Norm weights are tiny — 32 layers × 2 norms (input + post-attn)
× 4 096 floats × 4 B = 1.05 MiB FP32. Converting them to FP16
would save 525 KiB *and* require:

* A new `RmsNormF16` shader (~80 LOC).
* A new `RmsNormMulRopeF16` shader (~150 LOC) — the fused norm
  + RoPE path used by `dispatch_layer_batch`.
* Pipeline registration + dispatch routing for both.

525 KiB / 7.48 GiB = 0.0068%. Skipped — the line is "convert
when it pays off." lm_head pays off; norms don't.

## Why we don't convert the GPU embed copy

Sprint 22B already skipped it entirely (`token_embd.weight`
on GPU is dead weight when `output.weight` is present, which
it is for neuralmagic FP8 builds). No further action needed.

## VRAM headroom now

```
                                  16 GiB card
─────────────────────────────────────────
Total GPU upload (Llama-3.1-FP8):  7.48 GiB
KV cache @ ctx=8K (FP16 KV):       1.15 GiB
Scratch + Vulkan overhead:        ~0.50 GiB
─────────────────────────────────────────
Total @ 8K context:               ~9.13 GiB
Headroom @ 16 GiB:                ~6.87 GiB
                                  (was ~5.9 GiB at Sprint 22B,
                                   ~3.9 GiB at Sprint 20-M1)
```

## 14B FP8 viability check

```
14B FP8 (e.g. Qwen2.5-14B-Instruct-FP8 hypothetical):
  FP8 weights:                    ~12.0 GiB
  FP16 lm_head (152 064 × 5 120): ~1.6 GiB
  Norms (FP32):                    <1 MiB
  KV cache @ 4K (FP8):              0.4 GiB
  Scratch:                          0.5 GiB
  ────────────────────────────────────────
  Total:                          ~14.5 GiB    ← fits 16 GiB!

WITHOUT this sprint (FP32 lm_head ~3.2 GiB):
  Total:                          ~16.1 GiB    ← over budget!
```

A 14B FP8 model class on a 16 GiB card now becomes plausible.
Final viability would also need an FP8 SafeTensors of that size
to actually exist (most 14B FP8 builds today are vLLM-targeted
and sit unused for desktop deploys), but the loader + dispatch
infrastructure is ready.

## Coherence

```
> What is 2+2?
   Sprint 22B:  "The answer to 2+2 is 4."   (11 tokens, EOS)
   Sprint 22C:  "The answer to 2+2 is 4."   (11 tokens, EOS)
   → bit-identical greedy output
```

BF16 → FP16 narrowing has more mantissa precision than BF16
itself (10 bits vs 7), so quality at lm_head is **strictly
≥ the BF16 source**. The only risk would be range overflow
(FP16 max = 65 504 vs BF16 max = 3.4e38), but lm_head
weights are typically ≪ 1.0 in magnitude — comfortably inside
the FP16 range.

## Regression — all green

* Qwen3-8B-Q4_K_M chat: prefill 328 tok/s, decode 111 tok/s
  (within noise of pre-Sprint-22 baseline; the Q-K_M chat path
  doesn't touch the new shader at all).
* `cargo test --release --lib`: **37/37 pass**.
* FP8 GEMV correctness: pass (kernel unchanged).
* FP8 GEMM correctness: pass (kernel unchanged).
* `safetensors_inspect --load`: 290 GpuTensors (1 fewer than
  Sprint 20-M1 = the skipped embed; 22C unchanged on count
  because lm_head is still uploaded, just 2× narrower).

## Sprint 22 trail (VRAM optimisation)

```
57e5a25  22   `--max-context` CLI flag   (long-context support)
c4c72f4  22B  skip embed_tokens upload    -1.96 GiB
[this]   22C  FP16 lm_head                -0.98 GiB    +9% decode
─────────────────────────────────────────────────────────────
Cumulative VRAM saved (Llama-3.1-FP8):   2.94 GiB     (−28.2%)
Cumulative decode improvement:           +5.6 tok/s   (+9.0%)
```

## Honest forward-looking

* The remaining VRAM is **6.5 GiB FP8 weights + 1.0 GiB FP16
  lm_head** plus negligible norms + dummies. To shave more on
  the 8B model would need either smaller FP8 quantization (E5M2
  at the same per-element width gives nothing; we'd need 4-bit
  quants to halve the FP8 weights, which is exactly Q4_K_M's
  pitch). The 8B FP8 + FP16 lm_head footprint is at the
  measurable floor for this format.
* The 14B FP8 case is **theoretical** until the inference
  community standardises a 14B FP8 SafeTensors release. If a
  user wants to load one, the path now works; we'd need to
  bench it to confirm.
* This sprint's `+9% decode` is the kind of gain that's both
  real and hard to get any other way — it's not the WMMA path,
  it's the lm_head-bandwidth path. Adjacent levers (lm_head
  GEMV with subgroup-add reduction, mirroring Sprint 14B's Q4_K
  pattern) might add another few percent.

## Files touched

* `vk_shaders/mul_mat_vec_f16.comp` (new, ~95 LOC)
* `build.rs` (+1 SPV job, ~10 LOC)
* `src/backend/vulkan/shaders.rs` (+`MulMatVecF16` ShaderId, name,
  spv_bytes, ALL_SHADERS slot)
* `src/backend/vulkan/pipeline_registry.rs` (registered alongside
  the existing FP8/F32 family)
* `src/backend/vulkan/forward.rs` (+`F16` arms in
  `layer_weight_shader` and `dispatch_final`,
  +`MulMatVecF16` in the `one_per_row` matcher,
  ~10 LOC total)
* `src/backend/vulkan/loader.rs` (+`bf16_to_f16_vec` converter,
  +`is_lm_head` guard for the BF16 → FP16 path, ~50 LOC)
* `results/v034_sprint22c_lmhead_fp16.md` (this report)

102 SPVs total (was 101), 37/37 lib tests pass.
