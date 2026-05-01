# Phase 7 — Aligned vec4 B-loads investigation (STOP)

**Date:** 2026-04-27
**Build:** v0.1.3 (commit `aab1bba`)
**Outcome:** **STOP per Rule 2 ("BEI UNKLARHEITEN SOFORT STOP")** —
the prompt's premise is incorrect. mul_mmq does not have, and cannot
trivially get, an aligned vec4-B-load variant. The aligned path
exists only for `mul_mm.comp` (FP32 input), not `mul_mmq.comp`
(Q8_1 input). No code changes; this report only.

---

## What the prompt asked for

> llama.cpp baut ZWEI Varianten jedes GEMM-Shaders:
> mul_mmq_q4_k_f32_aligned.spv: LOAD_VEC_B=4, B_TYPE=vec4, ALIGNED=1

Expected gain: +30–50 % prefill via 4× wider B-loads into LDS.

The prompt itself flagged the most probable failure mode in pitfall
#2: *"Unsere Activations sind Q8_1 (nach quantize_q8_1). B_TYPE=vec4
interpretiert Q8_1-Daten als vec4 — stimmt die Alignment? Q8_1 Blöcke
sind 36 Bytes, nicht 16-byte-aligned. Falls Q8_1: vec4-Load auf
unaligned Q8_1 = Crash oder Garbage."*

That pitfall is the actual answer. Below is the evidence.

---

## Evidence — mul_mmq has no ALIGNED path

### 1. Direct grep on both shaders

```
$ grep -n 'ALIGNED\|aligned' \
    ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp \
    vk_shaders/mul_mmq.comp
(no output — zero matches in either file)
```

Neither our `mul_mmq.comp` nor llama.cpp's source has any
`#ifdef ALIGNED` block, any `#if LOAD_VEC_B == 4` branch, or any
mention of `aligned` in comments.

### 2. The shader is byte-identical to llama.cpp's

```
$ diff -q ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp \
          vk_shaders/mul_mmq.comp
(exit 0 — files identical)
```

We're not running a stale port; we have the upstream shader exactly.

### 3. llama.cpp generates only ONE mul_mmq variant

```
$ grep 'string_to_spv.*mmq\|mul_mmq' \
    ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
594: string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp",
       merge_maps(merge_maps(base_dict, float_type_dict),
       {{data_a_key, "1"}, {"D_TYPE", "float"},}),
       fp16, coopmat, coopmat2, f16acc);
```

That's the complete list. **No `mul_mmq..._aligned`** generation
exists. The aligned variants the prompt cites at lines 582-583 use
`source_name = "mul_mm.comp"` (FP32-input GEMM), not mul_mmq.

### 4. mul_mmq.comp's B binding is hard-coded to a Q8_1 packed struct

`vk_shaders/mul_mmq.comp:33`:

```glsl
layout (binding = 1) readonly buffer B { block_q8_1_x4_packed128 data_b[]; };
```

There is **no** `#ifdef ALIGNED` switch on the binding. `B_TYPE` is
not referenced at all in mul_mmq.comp. Setting `B_TYPE=vec4` as a
build define would have no effect on the binding — the shader
ignores it.

### 5. `LOAD_VEC_B = 16` is hard-coded, not a define

`vk_shaders/mul_mmq.comp:102`:

```glsl
#define LOAD_VEC_B 16
```

This is set unconditionally (not `#ifndef LOAD_VEC_B / #define ...`).
It's the *count of int8 quants* loaded per Q8_1 sub-block via the
packed struct, not a vec4-of-floats count. Forcing `LOAD_VEC_B = 4`
via build defines would either be ignored (because of the
unconditional `#define`) or, if the order changed, would break the
load arithmetic (`loadr_b = thread.x % (BK / LOAD_VEC_B)` — at
`LOAD_VEC_B = 4` we'd compute thread%8 instead of thread%2, and
`loadstride_b` would change correspondingly, walking off the Q8_1
block grid).

### 6. Why the aligned vec4 path can't apply to Q8_1 B

A `vec4` is 16 bytes (4 × `float`). The Q8_1 sub-block layout is:

```
struct block_q8_1 {
    f16 d;              // scale
    f16 sum;            // bias / sum
    int8 qs[32];        // 32 quants
};                       // total 36 bytes
```

The `block_q8_1_x4_packed128` variant interleaves 4 sub-blocks into
a 128-bit-aligned shape (used by mul_mmq for coalesced ivec4 loads
of the integer quants). Reinterpreting that buffer as `vec4 data_b[]`
would do the following over the first 16 bytes:

| byte offset | actual content | read as `vec4.x` (float) |
|---:|---|---|
| 0–1 | `d` (fp16 scale, e.g. 0x3C00 → 1.0) | bits would render as ~0.0 (denormal) |
| 2–3 | `sum` (fp16 bias) | combined into the same float |
| 4–7 | `qs[0..3]` (4× int8) | int8 byte pattern read as float bits |

The first `vec4.x` would be a meaningless float built from `d`/`sum`
fp16 patterns; `vec4.y..vec4.w` would be fictional floats made from
int8 quant bytes. The integer-dot-product pipeline mul_mmq expects
gets nothing it can use. This is the "Crash oder Garbage" outcome
the prompt's pitfall #2 already predicted.

---

## Where the aligned path *does* exist — and why we shouldn't ship it now

The aligned vec4 path is for `mul_mm.comp`:

```
582-583  string_to_spv(shader_name + "_" + tname + "_f32",         "mul_mm.comp", ..., {"B_TYPE","float"}, ...);
         string_to_spv(shader_name + "_" + tname + "_f32_aligned", "mul_mm.comp", ...,
                       {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32 /* "vec4" */}, {"ALIGNED","1"});
```

`mul_mm.comp` takes **raw FP32 B** (no quantize_q8_1 step), so
`B { float data_b[]; }` is naturally vec4-loadable when the start
offset and N stride are 4-aligned.

But — from `results/phase7_v013_benchmark.md` — `mul_mm` (currently
opt-in) is **45 % slower than mul_mmq at prefill** because raw FP32
B has 4× the bandwidth of Q8_1-packed B. An aligned vec4 variant
might recoup some of that (vec4 load reduces instruction count, not
bandwidth), but it would have to close a 45 % gap before becoming
the default. Even at the +30–50 % the prompt hopes for, the aligned
mul_mm could roughly match — not beat — mul_mmq.

That's a worthwhile experiment, but it is not what this prompt
asked for, and it is not a drop-in build-define change. It needs:

* a new `mul_mm_q4_k_f32_aligned.spv` build job (LOAD_VEC_B=4,
  B_TYPE=vec4, ALIGNED=1),
* runtime selection (`N % 4 == 0` → aligned, else unaligned),
* parity tests at every shape we run (M=2048 N=62 K=2048,
  M=2048 N=62 K=11008, both Q4_K and Q6_K),
* a perf comparison against the *current* mul_mmq baseline (1037
  tok/s on Qwen3-8B), not against mul_mm's slower variant.

That's a Phase-8 work item, not a Phase-7 patch.

---

## Why the +30–50 % expectation doesn't carry over

The prompt's bandwidth math:

> Aktuell (unaligned, B_TYPE=float):
>   Jeder B-Load: 1 float = 4 Bytes
> Aligned (B_TYPE=vec4):
>   Jeder B-Load: vec4 = 16 Bytes

…is the model for **mul_mm** (FP32 B). For **mul_mmq**, the actual
B-load is *already* 16 bytes wide:

```glsl
// mul_mmq_funcs.glsl:432
const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
```

That `ivec4` is a 4×int32 = 16-byte coalesced load, fetching 16
int8 quants in one go. There's no "narrower load" hidden in
mul_mmq's B path that a vec4 reinterpretation would widen.

---

## What would actually move mul_mmq prefill

For reference, candidate optimisations that *would* apply to mul_mmq
(none of them done in this prompt):

1. **Pipeline cache / lookaside cache for descriptor sets at GEMM
   dispatch boundaries** — already done in v0.1.0/v0.1.2; the
   marginal gains here are small.
2. **Coopmat (KHR cooperative_matrix) integer-MMA path on RDNA4**
   — Phase 6A confirmed BF16 × BF16 → FP32 WMMA exists. Integer
   coopmat (`v_wmma_i32_16x16x16_iu8`) is the right primitive but
   needs a from-scratch shader. Tracked for v0.2.
3. **Bigger BN / BM tiles to amortise per-WG dispatch overhead** —
   the v0.1.3 sweep showed `TM=2 TN=4` is at the local optimum for
   our spec-constants. Going to BN=128 or BM=128 changes occupancy
   pressure and needs a separate sweep.
4. **Split-K for very long K** — already used by FlashAttn; not
   applied to GEMM.

---

## Decision

**No code changes.** Per the explicit STOP gate in Schritt 1.2 and
Rule 2 of this prompt's preamble, I'm not adding build defines or
shader-IDs that would compile against a kernel without an ALIGNED
path. Doing so would either (a) silently leave the unaligned code
path live and waste a pipeline slot, or (b) trip the existing
`#define LOAD_VEC_B 16` and produce a kernel that mis-indexes the
Q8_1 block grid.

The 93/93 test suite is unchanged; tag `v0.1.3` (already pushed at
the previous turn) covers the v0.1.3 release surface as documented.

If the user wants to pursue aligned vec4 B-loads, the right
follow-up is a `mul_mm` aligned variant (Phase 8), not a mul_mmq
build-define change.
