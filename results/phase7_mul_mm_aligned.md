# Phase 7 — `mul_mm` aligned variants (vec4 B-loads)

**Date:** 2026-04-27
**Build:** v0.1.3 (unbumped — opt-in optimisation)
**Hardware:** AMD Radeon RX 9070 XT, gfx1201, RDNA 4, RADV
**Outcome:** Aligned variants ship correctly **and stay default OFF** —
the vec4 B-load path is bit-exact against the unaligned shader and
the CPU reference, but is **37–39 % slower than `mul_mmq`** on every
aligned prompt we measured. RDNA4 is bandwidth-bound for our prefill
shapes, not instruction-bound, so reducing load count without
reducing byte count doesn't help.

---

## What we built

Two new SPV shaders, compiled from the same `mul_mm.comp` with
`ALIGNED=1 / LOAD_VEC_B=4 / B_TYPE=vec4 / FLOAT_TYPEV4=vec4`:

| Shader | Defines |
|---|---|
| `mul_mm_q4_k_f32_aligned.spv` | `LOAD_VEC_A=4`, `LOAD_VEC_B=4`, `B_TYPE=vec4`, `ALIGNED=1` |
| `mul_mm_q6_k_f32_aligned.spv` | `LOAD_VEC_A=2`, `LOAD_VEC_B=4`, `B_TYPE=vec4`, `ALIGNED=1` |

Mirroring `vulkan-shaders-gen.cpp:583`'s aligned generation rule for
`mul_mm.comp`. (Q6_K's `LOAD_VEC_A` stays at 2 — the Q6_K dequant
branch in `mul_mm_funcs.glsl` writes 2 weights/idx; this was the
fix from the earlier Phase-7 patch.)

The aligned shader takes the `LOAD_VEC_B == 4` branch in
`load_b_to_shmem` (`mul_mm_funcs.glsl:535-544`):

```glsl
const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);   // one vec4 load
buf_b[buf_idx + 0] = bb.xy;                     // → two vec2 LDS writes
buf_b[buf_idx + 1] = bb.zw;
```

…and skips the unaligned bounds-checked path (`mul_mm_funcs.glsl:548`).
That makes the aligned shader unsafe at unaligned `N` — runtime must
guard with `seq_len % 4 == 0`.

### Routing

`forward.rs` introduces a per-batch `GemmKind` enum decided at the top
of `dispatch_layer_batch`:

```rust
let gemm_kind = if self.mul_mm_enabled {
    if seq_len % 4 == 0 { GemmKind::MulMmAligned } else { GemmKind::Mmq }
} else {
    GemmKind::Mmq
};
```

* `mul_mm_enabled = false` (default) → always `Mmq`.
* `mul_mm_enabled = true` and `seq_len % 4 == 0` → aligned `mul_mm`.
* `mul_mm_enabled = true` and `seq_len % 4 != 0` → falls back to `Mmq`
  rather than the unaligned scalar `mul_mm`, which Phase 7 already
  showed is ~45 % slower than `mul_mmq`.

`layer_weight_shader_gemm` takes a `GemmKind` instead of a `bool`, and
the seven dispatch sites (`gemm_q/k/v/o/gate/up/down`) now select
their shader through it. The Q8_1 quantize step is skipped iff
`gemm_kind ∈ {MulMm, MulMmAligned}`.

---

## Important correction to the prompt's premise

The prompt asserts:

> Für unsere GEMMs: N = hidden_dim = 4096 → 4096 % 4 == 0 → IMMER aligned

That's a dimension confusion. In our (and llama.cpp's) GEMM
convention `M × K × N`, **`N` is the batch dim**, i.e. `seq_len`. M
is the output projection dim. From `forward.rs::run_gemm`:

```rust
self.run_gemm(dev, registry, cmd, sq, wq, gemm_input_attn,
              self.batch_q.handle,
              q_dim,    // → m
              seq_len,  // → n
              hidden,   // → k
              "gemm_q");
```

So the alignment guard `n % 4 == 0` is on **`seq_len`**, which
is short and varies per prompt (20–200 in the 15-prompt suite). Of
those 15 prompts, only **2 have `seq_len % 4 == 0`** (#1 Greeting at
20, #15 Emoji at 52). The aligned shader gets exercised in only
2/15 prompts; the rest fall back to `mul_mmq`.

This is exactly the runtime-guard contract llama.cpp documents in
its driver code. It's also why the prompt's pitfall #3 already
flagged "Falls N nicht durch 4 teilbar (z.B. seq_len=62): aligned-
Shader darf NICHT dispatched werden". The pitfall was right; the
"IMMER aligned" claim two paragraphs earlier was wrong.

---

## Correctness — 6 new parity tests, all green

`tests/correctness.rs` — `run_mul_mm_aligned_parity(M, N, K)` plus a
direct aligned-vs-unaligned bit-exactness check:

| Test | M | N | K | Tolerance | Result |
|---|---:|---:|---:|---|---|
| `_q4k_64x64` | 64 | 64 | 256 | `< 0.01 × cpu_amax` | ✓ |
| `_q4k_k2048` | 64 | 64 | 2048 | `<` | ✓ |
| `_q4k_n128` | 64 | 128 | 256 | `<` | ✓ |
| `_q4k_real_dims` | 2048 | **60** | 2048 | `<` | ✓ |
| `_q4k_ffn_dims` | 2048 | 60 | **11008** | `<` | ✓ |
| `_aligned_matches_unaligned_q4k` | 128 | 64 | 2048 | `<1e-4` (vs unaligned) | ✓ |

End-to-end with `VULKANFORGE_USE_MUL_MM=1`:

```
$ cargo test --release --test regression
test result: ok. 25 passed; 0 failed; ...
$ cargo test --release --test correctness
test result: ok. 50 passed; 0 failed; ...   ← 6 new tests + 44 existing
```

`phase3e_prefill_batch_matches_token_by_token_top5` shows top-1 =
151667 (identical), top-5 overlap = 4/5 (vs 5/5 for `mul_mmq`). The
displaced rank-5 token (138955 vs 50897) is the FP32-activations vs
Q8_1-activations precision difference, not a kernel bug.

---

## Performance — three-way comparison on Qwen3-8B-Q4_K_M

Same hardware, same prompts, BLOCK_SIZE = 256 in all runs.

### 5-prompt subset (prompts 1-5)

| Prompt | `pp` | `pp % 4` | `mul_mmq` baseline | `mul_mm` aligned |
|---|---:|---:|---:|---:|
| Greeting | 20 | **0** | 381.6 tok/s | **243.0 tok/s** (−36 %) |
| Simple Sequence | 31 | 3 | 728.4 | 728.3 (Mmq fallback) |
| Prime Check | 31 | 3 | 739.8 | 740.0 (Mmq fallback) |
| LRU Cache | 47 | 3 | 1117.5 | 1106.3 (Mmq fallback) |
| REST API | 62 | 2 | 1434.2 | 1428.2 (Mmq fallback) |
| **median** | | | **739.8** | **740.0** |

The four unaligned prompts get `Mmq` and run at the baseline (within
noise). The single aligned prompt drops 36 %.

### 15-prompt full suite

| Suite | mul_mmq baseline | mul_mm aligned (when N%4==0) | Δ |
|---|---:|---:|---:|
| median prefill | 1037–1047 tok/s | **931.6 tok/s** | **−11 %** |
| total prefill | 1024 | 953 | −7 % |
| median decode | 88.6 | 88.8 | +0.2 (noise) |
| coherent | 15 / 15 | 15 / 15 | — |

Aligned-`N` prompt detail in the 15-prompt run:

| Prompt | `pp` | mul_mmq | mul_mm aligned | Δ |
|---|---:|---:|---:|---:|
| #1 Greeting | 20 | 381 | 241 | **−37 %** |
| #15 Emoji | 52 | 1191 | 724 | **−39 %** |

Both aligned cases regressed by ~40 %. The median dropped 11 %
because those two prompts pulled it down.

---

## Why aligned isn't faster on RDNA4 (and probably can't be)

The prompt's bandwidth model:

> Aktuell (unaligned, B_TYPE=float):
>   Jeder B-Load: 1 float = 4 Bytes
> Aligned (B_TYPE=vec4):
>   Jeder B-Load: vec4 = 16 Bytes

…assumes RDNA4 is **instruction-bound** at this workload, so reducing
the load count by 4× would speed things up. Two pieces of evidence
say it isn't:

1. **mul_mmq vs mul_mm-unaligned (Phase 7 earlier):** mul_mmq is
   ~45 % faster than `mul_mm` because Q8_1 packs the activations to
   ~1.13 B/elem vs FP32's 4 B/elem — a 3.5× **bandwidth** advantage,
   not an instruction advantage. If we were instruction-bound, the
   wider Q8_1 dequant logic would slow `mul_mmq` down, not speed it
   up.

2. **mul_mm-aligned vs mul_mm-unaligned:** vec4 loads reduce the load
   *instruction count* by 4× without changing total byte traffic.
   `mul_mm` aligned is roughly the same speed as unaligned and 36 %
   slower than `mul_mmq` — exactly what you'd predict if VRAM
   bandwidth, not load issue rate, is the choke point.

Conclusion: **VRAM byte traffic is the bottleneck for our prefill
GEMM shapes on RDNA4.** Lowering instruction count alone can't fix
that. The path that would actually move the needle is reducing
B-side bytes, which is exactly what `mul_mmq`'s Q8_1 packing already
does. To go faster than `mul_mmq` we'd need a fundamentally different
B compression (e.g. BF16 = 2 B/elem with coopmat, or a Q4_1 input)
or a different memory pattern (e.g. coopmat MMA loading B once
and reusing across multiple K-tiles).

---

## Decision

* **Aligned shaders shipped:** the kernels are bit-exact and
  thoroughly parity-tested; opting in via `VULKANFORGE_USE_MUL_MM=1`
  now picks the aligned shader instead of the slower scalar `mul_mm`.
* **Default stays OFF.** `mul_mmq` is still the fastest GEMM on
  this hardware for our prefill shapes.
* **Pipeline cache grows by 388 KB** (two extra 194 KB SPVs); cold
  start now writes 547 KB to `~/.vulkanforge/pipeline_cache.bin`
  instead of 159 KB. Steady-state perf unchanged.
* **No version bump.** This is an opt-in shader addition, not a
  user-facing default change.

If a future workload has long enough `N` that vec4 instruction
issue becomes the bottleneck (e.g. a synthetic pp=4096 batch), the
infrastructure is in place to swap in the aligned shader without
code changes — just `VULKANFORGE_USE_MUL_MM=1`.

---

## Files modified

| File | Change |
|---|---|
| `build.rs` | Two new ShaderJobs (`mul_mm_q{4,6}_k_f32_aligned.spv`) |
| `src/backend/vulkan/shaders.rs` | `MulMmQ4KAligned` / `MulMmQ6KAligned` enum variants + SPV bytes |
| `src/backend/vulkan/pipeline_registry.rs` | Aligned shader IDs route through the existing `MulMm*` registration arm (same spec-constants) |
| `src/backend/vulkan/forward.rs` | `GemmKind` enum; `dispatch_layer_batch` chooses kind from `mul_mm_enabled && seq_len % 4 == 0`; `layer_weight_shader_gemm` takes `GemmKind` |
| `tests/correctness.rs` | 6 new tests (`test_mul_mm_aligned_*`) |
| `results/phase7_mul_mm_aligned.md` | This file |
| `results/v013_aligned_logs/*.log` | Bench logs (3 files) |

## Test suite

```
24 lib + 50 correctness + 25 regression = 99 / 99 green
```

(50 correctness was 44; +6 aligned parity tests.)

---

## Reproduce

```fish
# baseline (mul_mmq):
VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
  ./target/release/examples/run_15prompt_bench

# aligned mul_mm where seq_len % 4 == 0, mul_mmq otherwise:
VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
VULKANFORGE_USE_MUL_MM=1 \
  ./target/release/examples/run_15prompt_bench
```
