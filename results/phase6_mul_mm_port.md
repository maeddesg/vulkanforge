# Phase 6 — `mul_mm.comp` port (STOP report)

**Date:** 2026-04-27
**Version:** v0.1.2 (no version bump — port shipped behind opt-in flag)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Status:** **STOP** per prompt rule 2 ("BEI UNKLARHEITEN SOFORT STOP").
The shader compiles and dispatches without driver errors but
produces garbage / NaN logits. Default OFF; opt-in via
`VULKANFORGE_USE_MUL_MM=1`.

---

## TL;DR

Ported llama.cpp's `mul_mm.comp` (MIT, 464 LOC) into the Phase-6
prefill GEMM path with all the host-side wiring (build entry, push-
constant struct, descriptor layout, `dispatch_layer_batch` switch
gated on `VULKANFORGE_USE_MUL_MM`, helper to pick the right
shader-id per layer).

Two configurations tried, both produce wrong output:

| Config | `LOAD_VEC_A` | Result |
|---|---|---|
| Initial: shader default | 1 | Finite logits, but argmax=64918 vs 151667 — **0/5 top-5 overlap** |
| llama.cpp's value (vulkan-shaders-gen.cpp:560) | 4 | **NaN/Inf logits** |

mul_mmq (the existing path) keeps producing the right output, so
the bug is isolated to the new mul_mm path.

The infrastructure (shader, build job, pipeline registration, helper
function, integration into `dispatch_layer_batch`) is committed for
future debugging — flip `VULKANFORGE_USE_MUL_MM=1` to reactivate it.

---

## 1 — Bestandsaufnahme (validated)

### 1.1 Helper files already in repo

We already have `vk_shaders/types.glsl`, `mul_mm_funcs.glsl`,
`mul_mm_id_funcs.glsl` from the `mul_mmq` port (Phase 3C). These
are byte-identical to llama.cpp's, so the shader's `#include`s
resolve as-is.

### 1.2 mul_mm vs mul_mmq

```
                    mul_mm.comp                mul_mmq.comp (current)
A (weights)         Q4_K / Q6_K               Q4_K / Q6_K
B (activations)     FP32 directly             Q8_1 (block_q8_1_x4)
                    (no quantize step)        (needs quantize_q8_1 dispatch)
D (output)          FP32                      FP32
Inner loop          scalar FMA                integer dot-product (DP4A-style)
LDS layout          buf_a/b: vec2-packed,     buf_a/b: per-quant struct
                    SHMEM_STRIDE = BK/2 + 1   (block_a_cache / block_b_cache)
LOAD_VEC_A (Q4_K)   4                         8 (= 4 × QUANT_R_MMQ)
```

### 1.3 Why the port is desirable in principle

- llama.cpp picks `mul_mm` over `mul_mmq` on RADV for `M > 1`
  prefill — it's their faster path when integer-dot isn't a win.
- Removes one dispatch per GEMM (the `quantize_q8_1` step) — that
  was 5 quantize dispatches per layer in our prefill path
  (`quantize_attn`, `quantize_attn_out`, `quantize_ffn`,
  `quantize_ffn_h`).
- Same descriptor layout (3 SSBOs: A / B / D) as `mul_mmq`, same
  push-constant struct (16 × u32 = 64 B), so the host-side wiring
  is essentially a one-line shader-id swap plus the buffer-binding
  change (FP32 batch_norm vs Q8_1 batch_q8).

---

## 2 — Implementation

```
NEW   vk_shaders/mul_mm.comp                   464 LOC, copy of llama.cpp's
NEW   vk_shaders/mul_mm_funcs.glsl             598 LOC (already in repo from Phase 3C)
NEW   vk_shaders/mul_mm_id_funcs.glsl          74 LOC (already in repo)

EDIT  build.rs                                 +2 ShaderJobs (Q4_K, Q6_K)
EDIT  src/backend/vulkan/shaders.rs            +ShaderId::MulMmQ4K, MulMmQ6K
EDIT  src/backend/vulkan/pipeline.rs           pub type MulMmPushConstants
                                                = MmqPushConstants
EDIT  src/backend/vulkan/pipeline_registry.rs  pipeline registration with
                                                full spec-constant set
EDIT  src/backend/vulkan/forward.rs            +mul_mm_enabled field +
                                                set/get helpers +
                                                conditional skip of
                                                quantize_q8_1 dispatches +
                                                layer_weight_shader_gemm
                                                (picks mul_mm vs mul_mmq id)
```

Build defines for Q4_K (matched against llama.cpp's
`vulkan-shaders-gen.cpp` line 582 + 569 + 560):

```rust
[
    ("DATA_A_Q4_K", "1"),
    ("A_TYPE", "block_q4_K"),
    ("A_TYPE_PACKED32", "block_q4_K_packed32"),
    ("B_TYPE", "float"),
    ("D_TYPE", "float"),
    ("FLOAT_TYPE", "float"),
    ("FLOAT_TYPEV2", "vec2"),
    ("ACC_TYPE", "float"),
    ("ACC_TYPEV2", "vec2"),
    ("LOAD_VEC_A", "4"),
]
```

(Q6_K mirrors this with `DATA_A_Q6_K`, `A_TYPE_PACKED16`, etc.)

Spec-constants pinned in `pipeline_registry.rs`:

```rust
[
    BLOCK_SIZE = 128, BM = 64, BN = 64, BK = 32,
    WM = 32, WN = 32, WMITER = 2,
    TM = 2, TN = 4,            // Phase 6 v0.1.2 cont. winning shape
    TK = 1,
    WARP = 64,                 // RDNA Wave64
]
```

`shaderc 0.8` produces a 195 KB SPV (Q4_K) / 194 KB SPV (Q6_K)
without compile errors or warnings.

---

## 3 — Symptoms

### 3.1 Initial attempt (shader-default `LOAD_VEC_A=1`)

```
[parity] top1_a=151667 top1_b=64918
         top5_a=[151667, 85387, 151668, 34894, 50897]
         top5_b=[64918, 114045, 20679, 85112, 52955]
         overlap=0
```

argmax landed on a token (`64918`) far from the correct
`151667 = <think>` — total mismatch. Logits were finite (no
NaN/Inf), so the GEMM produced a self-consistent but wrong matrix.

### 3.2 Fixed attempt (`LOAD_VEC_A=4`, matching llama.cpp)

After tracing through `vulkan-shaders-gen.cpp`:

```cpp
// gen.cpp:560-561
else if (tname == "q4_k" || tname == "q6_k" || …)
    load_vec_quant = "4";
// gen.cpp:569
load_vec_a_unaligned = (… || tname == "f32" || …) ? "1" : load_vec_quant;
```

…llama.cpp passes **`LOAD_VEC_A = 4`** for the Q4_K/Q6_K unaligned
mul_mm builds. The Q4_K dequant block in `mul_mm_funcs.glsl` is
hard-coded to that assumption (line 194: `const uint ib = idx / 64`,
where 64 idx-units = 256 weights / 4 weights-per-idx = block-coverage).

Re-built with `LOAD_VEC_A = 4`:

```
prefill_batch produced NaN/Inf logits
```

So the indexing is now in the right *block coverage* but somewhere
along the K-loop, the shader reads / writes invalid data and
NaN propagates.

### 3.3 What we tried not to overlook

| Hypothesis | Verified? |
|---|---|
| Strides / push-constant field order | ✅ matches llama.cpp's `vk_mat_mat_push_constants` (we have 16 fields, llama.cpp has 17 with trailing `padded_n` — shader ignores it; reflection-driven push-constant size in our pipeline registry is 64 bytes, matching) |
| Stride-A divisor (`stride_a / LOAD_VEC_A`) | math checks out for ir ∈ [0, blocks_m) |
| Stride-B (FP32 layout `[N][K]`, stride = K) | matches mul_mmq's working setup |
| Bounds-checks on B for unaligned N=62 vs BN=64 | shader's `idx_n < p.N` guard fills out-of-bounds with zero — should be safe |
| Spec-constants pinned correctly (BK=32, etc.) | yes, all 11 ids pinned |
| Build defines (FLOAT_TYPE / ACC_TYPE / ACC_TYPEV2) | added all of them; shaderc compiles cleanly |
| LDS bank-conflict padding (`SHMEM_STRIDE = BK/2 + 1`) | matches llama.cpp |

### 3.4 What probably is the issue (not yet pinpointed)

Best guess after the trace: there's a load-pattern detail in
`mul_mm_funcs.glsl`'s Q4_K branch (lines 190–225) that the shader
expects to *be paired* with a particular outer-loop count or with
the `ALIGNED` define being set. The unaligned variant might
require N to be a multiple of `BN` or some additional `padded_n`
handling in the shader that our build doesn't compile in.

Other plausible culprits:

- **Some `loadr_a` / `loadc_a` arithmetic mismatch** when
  `LOAD_VEC_A=4` and `BLOCK_SIZE=128`. Our 128-thread WG layout
  is `loadr_a ∈ [0, 8)`, `loadc_a ∈ [0, 16)` per pass; llama.cpp
  may dispatch with a different `BLOCK_SIZE` for unaligned Q4_K.
- **`BK_STEP` mismatch**: `mul_mm.comp` `#define BK_STEP 2` for
  quants. Cache load loop runs `BK / BK_STEP = 16` iterations,
  reading `buf_a[(...) + i]` for `i ∈ [0, 16)`. With `LOAD_VEC_A=4`
  the dequant writes 2 vec2 entries per loadr_a slot — interaction
  with `BK_STEP` may require recomputing `SHMEM_STRIDE`.

Pinpointing this needs GPU-side instrumentation (e.g. RGP capture,
or a tiny Q4_K dequant-only debug shader to validate the LDS
contents), which is multi-day work.

---

## 4 — Decision

Per prompt rule 2: STOP. Default `mul_mm_enabled = false`.

The ported shader + integration code stays in the repo so future
debugging can flip the flag and iterate. The 82-test regression
suite stays green with `mul_mm` OFF (= the existing v0.1.2-cont.
mul_mmq path with TM=2 TN=4).

```
unit (lib)         24
correctness        33
regression         25
TOTAL              82   ALL GREEN with VULKANFORGE_USE_MUL_MM=0 (default)
```

---

## 5 — Findings worth keeping

Even though the port doesn't ship live, the work produced:

- **Confirmed `LOAD_VEC_A` semantics** for Q4_K mul_mm (`/ LOAD_VEC_A`
  is a divisor on the A-side stride, idx counts in groups of
  `LOAD_VEC_A` weight elements). This is a useful reference for
  any future kernel-level work.
- **Verified `mul_mm.comp` push-constant struct** matches our
  existing `MmqPushConstants` exactly (16 × u32 = 64 B). The
  `padded_n` field in llama.cpp's C++ wrapper is shader-ignored
  trailing data, and is the reason their PC struct is 17 fields
  while the shader sees 16.
- **Located the dispatch helper convention**:
  `cmd_dispatch(CEIL_DIV(M, BM), CEIL_DIV(N, BN), groups_z)` — same
  shape we already use for `mul_mmq`.
- **`vulkan-shaders-gen.cpp` documents the build matrix**: every
  quant type × {f32, f16, aligned, unaligned} combination has its
  own `LOAD_VEC_A` / `LOAD_VEC_B` / `B_TYPE` recipe (lines 540–600).
  Q4_K unaligned is `LOAD_VEC_A = 4, LOAD_VEC_B undefined,
  B_TYPE = float`. (Q4_K aligned would be `LOAD_VEC_B = load_vec`,
  `B_TYPE = aligned_b_type_f32` = `vec4`, `ALIGNED = 1` — different
  shader path.)

---

## 6 — What would unblock this in v0.2

1. **GPU-side instrumentation** to dump intermediate `buf_a` /
   `buf_b` LDS contents after the load step and compare against an
   FP32 dequant of the same Q4_K block.
2. **Smaller-input bring-up**: dispatch with M=64, N=16, K=32 (one
   tile in each direction) and compare the resulting 64×16 matrix
   element-by-element against an FP32 reference. The mismatch
   pattern should reveal whether it's a stride bug, an LDS-write
   bug, or a cache-read bug.
3. **Try the aligned variant** (`-DALIGNED=1` + `LOAD_VEC_B=4` +
   `B_TYPE=vec4`) on a padded-N batch (`N` rounded up to BN=64
   with zero-fill). If that works, the unaligned path's load
   bounds check is the likely culprit.

Effort estimate to debug to green: 2-4 days, matching the v0.1.3
budget mentioned earlier. Best done as a dedicated phase rather
than a side-task.

---

## 7 — Console summary

```
═══ Phase 6 — mul_mm port (STOP) ═══
Shader:      vk_shaders/mul_mm.comp (464 LOC, 195 KB SPV Q4_K)
Helpers:     mul_mm_funcs.glsl + mul_mm_id_funcs.glsl already in repo
Status:      compiles + dispatches, but logits are wrong
             (NaN with LOAD_VEC_A=4, garbage with LOAD_VEC_A=1)
Default:     OFF (`VULKANFORGE_USE_MUL_MM=1` to opt in)
Tests:       82/82 green with default OFF
             phase3e_prefill_batch_matches_token_by_token_top5
             FAILS with USE_MUL_MM=1 (overlap=0, NaN/garbage logits)
Diagnostic:  needs GPU-side LDS dump + small-tile bring-up
             (~2-4 days, deferred to v0.2)
```

---

## 8 — Files

```
NEW   vk_shaders/mul_mm.comp                       (copy of llama.cpp 464 LOC)
NEW   results/phase6_mul_mm_port.md                this report

EDIT  build.rs                                     +2 ShaderJobs
EDIT  src/backend/vulkan/shaders.rs                +2 ShaderIds + blob refs
EDIT  src/backend/vulkan/pipeline.rs               +MulMmPushConstants alias
EDIT  src/backend/vulkan/pipeline_registry.rs      +pipeline registration
EDIT  src/backend/vulkan/forward.rs                +mul_mm_enabled field
                                                   +setter / getter
                                                   +conditional quantize
                                                    skip in dispatch_layer_batch
                                                   +layer_weight_shader_gemm
```

The infrastructure is purely additive — no existing code path
behaves differently when `VULKANFORGE_USE_MUL_MM` is unset (default).
