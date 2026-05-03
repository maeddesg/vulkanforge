# Sprint 21A — Aligned-load FP8 GEMM (+69% prefill)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 20-Wire, head was `9a40b3e`)
**Goal:** Apply Sprint 12L's "wider-load" win (which lifted Q4_K
prefill +71%) to the FP8 GEMM kernel — replace the per-byte /
per-element load patterns with explicit 4-element grouped reads
that the compiler can coalesce on RDNA4.

## Headline result

```
> What is 2+2?  →  "The answer to 2+2 is 4."   (bit-identical vs Sprint 20-Wire)

| pp  | Sprint 20-Wire (naive) | Sprint 21A (aligned) | Δ      |
|-----|-----------------------:|---------------------:|-------:|
|  28 |  216.8 tok/s           |  377.4 tok/s         | +74 %  |
| 406 |  347.8 tok/s           |  589.4 tok/s         | +69 %  |
```

The bench gate of pp=406 ≥ 450 tok/s is cleared at 589 tok/s —
more than +30% past the minimum, halfway to the brief's 700-tok/s
"EXZELLENT" target.

## What changed (~50 LOC, single shader)

`vk_shaders/mul_coopmat_fp8_naive.comp` — kernel signature, push
constants, descriptor layout, ShaderId, build job, pipeline
registration, and `Forward::run_gemm_fp8_naive` are **unchanged**.
Only the inner per-K-step load pattern was rewritten:

### A-side: 1 uint32 per thread per K-step (was 4 redundant reads)

Each of the 64 threads' 4 K-positions live in the same uint32 word
(4 contiguous bytes). The naive kernel called `fp8_load(global_row,
gk)` four times — each call loaded `data_a[idx >> 2]` independently,
relying on the compiler to CSE the four identical word indices.
The new code does **one explicit `data_a[idx >> 2]` read per
thread per K-step**, then unpacks all 4 bytes from that word into
LDS:

```glsl
const uint a_gk_base   = kk + a_k_base;
const uint a_elem_idx  = global_row * pc.stride_a + a_gk_base;
const bool a_word_full = row_in_bounds && (a_gk_base + 3u < pc.k);
uint word = 0u;
if (a_word_full) {
    word = data_a[a_elem_idx >> 2u];
}
[[unroll]] for (uint w = 0u; w < 4u; w++) {
    const uint8_t byte = uint8_t((word >> (w * 8u)) & 0xFFu);
    const float val    = float(uintBitsToFloate4m3EXT(byte));
    buf_a[a_row * A_STRIDE + a_k_base + w] = ELEM_TYPE(val);
}
```

(Plus a tail branch for `pc.k % 4 != 0`, never hit by Llama-3
shapes — K=4096 / 14336 are both multiples of 16 already.)

### B-side: 4 contiguous K-positions per thread (coalesced `vec4`)

The naive kernel's thread mapping was
`(k_idx = fidx/16, n_idx = fidx%16)`, where each thread's 4 reads
were strided **by 4 in K** for the same N — un-coalescable.

The new mapping mirrors the A-side: `(b_n = tid/4, b_k_base =
(tid&3)*4)`. Each thread reads 4 **contiguous** K-positions of
one N-row of B as a single `vec4`:

```glsl
vec4 act4 = vec4(b[base + 0u], b[base + 1u], b[base + 2u], b[base + 3u]);
buf_b[(b_k_base + 0u) * B_STRIDE + b_n] = ELEM_TYPE(act4.x);
buf_b[(b_k_base + 1u) * B_STRIDE + b_n] = ELEM_TYPE(act4.y);
buf_b[(b_k_base + 2u) * B_STRIDE + b_n] = ELEM_TYPE(act4.z);
buf_b[(b_k_base + 3u) * B_STRIDE + b_n] = ELEM_TYPE(act4.w);
```

The 4-scalar pattern is what RADV/ACO turns into a single
16-byte coalesced burst when the base offset is 16-byte aligned
(`gn * stride_b + b_k_base` with `b_k_base ∈ {0,4,8,12}` → always
4-aligned in elements = 16-aligned in bytes since float = 4 B).

The output store + `weight_scale` post-multiply are unchanged; the
LDS layout `buf_b[k * B_STRIDE + n]` is the same so `coopMatLoad`
stays correct.

## Why this works

Sprint 12L's Q4_K analogue (LOAD_VEC_B=8 / `mat2x4` aligned coopmat)
hit +71% by the same mechanism: turning many small loads into
fewer wide loads reduces memory-pipeline pressure and lets the
WMMA stay fed. The FP8 kernel was bandwidth-bound on the same
axis. The same lever, two days apart.

The change is purely a load-pattern refactor; numerics are
**bit-identical** — same FP32 LDS staging, same BF16 coopMatLoad,
same FP32 accumulator, same per-tensor `weight_scale` post-multiply:

```
$ VULKANFORGE_ENABLE_FP8=1 cargo test --release --test fp8_gemm_correctness -- --nocapture
FP8 GEMM: max_abs=0.043406, rms_err=0.008280,
          rms_err/max_out=0.0006, max_out=14.694953
test fp8_gemm_matches_cpu_reference ... ok
```

(Same numbers as Sprint 20-GEMM; the optimization rearranges
*when* loads happen, not *what* values they read.)

## Coherence

```
> What is 2+2?
   Sprint 20-Wire:  "The answer to 2+2 is 4."   (11 tokens, EOS)
   Sprint 21A:      "The answer to 2+2 is 4."   (11 tokens, EOS)
   → bit-identical greedy output

> [406-token padded prompt about ML]
   Sprint 21A:  "It seems like you're..."  (5 tokens, capped)
   → coherent continuation
```

## GGUF regression

* Qwen3-8B-Q4_K_M chat: prefill 411 tok/s, decode 111.6 tok/s
  (within noise of pre-Sprint-21 baseline).
* `cargo test --release --lib`: **37/37 pass**.
* FP8 GEMV correctness test: pass (max_abs 6e-6 unchanged).
* FP8 GEMM correctness test: pass (rms / max_out 0.06% unchanged).

The 7 K-quant GEMM call sites in `dispatch_layer_batch` were
untouched. Only the `mul_coopmat_fp8_naive.comp` shader changed,
and it's only reached via `is_fp8_layer_weight(...)` for SafeTensors
F8E4M3 tensors.

## Files touched

* `vk_shaders/mul_coopmat_fp8_naive.comp` — load-pattern rewrite,
  ~50 LOC net change.
* `results/v034_sprint21a_fp8_aligned.md` — this file.

100 SPVs total (unchanged), kernel SPV size 23,068 → 23,072 B
(essentially unchanged — the optimization is in instruction
selection, not in adding new code paths).

## Sprint 20+21 trail (perf-of-prefill story)

```
9d3db6b — M3:    FP8 chat at 59 tok/s prefill (per-token GEMV)
9a40b3e — Wire:  FP8 GEMM wired in,  348 tok/s prefill (pp=406)
[this]  — 21A:   aligned loads,      589 tok/s prefill (pp=406)
```

* M3 → Wire: 59 → 348 = **5.9×** (changing prefill *kind* —
  per-token GEMV → batched GEMM).
* Wire → 21A: 348 → 589 = **1.7×** (50 LOC of load-pattern
  tightening on the existing GEMM).
* Composite M3 → 21A: 59 → 589 = **9.98×** prefill speedup
  on a 406-token prompt.

For a real chat prompt (≈400 tokens), prefill went from
"~7 second wait" to "~0.7 second wait." The product story moves
from "demo only" to "usable" with this commit.

## What's still on the table (next sprints)

1. **Multi-WG-per-tile coopmat variant** (Sprint 12M analogue).
   The current kernel uses 1 Wave64 per 16×16 tile. A 4-Wave64,
   64×64-tile variant amortizes LDS staging across more output
   work. Likely +30-50% at pp ≥ 256.
2. **TILE_K = 32 K-step** with the same 64-thread budget. Halves
   the K-loop iterations + barriers. Trade: 2× LDS, occupancy
   risk.
3. **`bench --tokenizer-from` plumbing** so pp=64/128/256/512/1024
   sweeps run on SafeTensors. ~30 LOC, mechanical.
4. **FP8-FP8-FP32 native WMMA path** (`coopmat<floate4m3_t, ...>`)
   instead of the current FP8 → BF16 narrow. Sprint 18B's measured
   1.18× WMMA ceiling means the gain is small even if it works,
   and the activation-clipping risk (FP8 max=448) is real. Not
   prioritized.

## Outcome

**Bench gate cleared at +69%.** Numerics bit-identical to
Sprint 20-Wire. GGUF path untouched. Sprint 12L's wider-load
pattern transfers cleanly to FP8.
