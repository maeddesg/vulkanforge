# VulkanForge v0.2.1 Sprint 11F — Sint8 coopmat Probe

**Date:** 2026-04-29
**Branch:** main (HEAD = 588d937, post-Sprint 11E)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV
**Mode:** Probe-only — 1-hour GO/NO-GO gate for an Int8-coopmat path

## TL;DR — STRONG GO

RDNA4 (gfx1201) advertises **20 cooperative-matrix configurations**
through `VK_KHR_cooperative_matrix`, including:

- **Entry 14: I8 × I8 → I32 → I32, 16×16×16, Subgroup, no saturating**
- Entry 15: same with saturating arithmetic
- Entries 4–13: U8/I8 mixed and U8 input variants
- Entries 0–3: FP8 (E4M3, E5M2) × FP8 → FP32

GLSL `coopmat<int8_t, ...>` syntax compiles cleanly via shaderc. The
emitted SPIR-V declares `OpCapability CooperativeMatrixKHR` and
`OpCapability Int8`. A runtime smoke test (16×16 INT8 ones × ones →
INT32 sixteens) **passes**: pipeline create OK, dispatch OK, all 256
output cells contain the expected value 16.

**Outcome C from the brief: STRONG GO for Sprint 11G** — Eigenbau
KHR-coopmat-int-DP GEMM, combining mul_mmq's bandwidth advantage
(Q8_1 activations + bit-packed Q4_K weights) with WMMA throughput.
This is the only candidate left that could materially close the
0.46× prefill GEMM gap to llama.cpp Vulkan on RDNA4.

171/171 tests green (170 + new `test_int8_coopmat_runtime_smoke`).

## Detail

### Step 1 — Device-side enumeration

The repo already shipped `examples/probe_coopmat.rs` (Phase 6A
diagnostic). Running it against gfx1201:

```
Device: AMD Radeon RX 9070 XT (RADV GFX1201)
API:    1.4.335
VK_KHR_cooperative_matrix:  PRESENT
VK_KHR_shader_bfloat16:     PRESENT

VK_KHR_cooperative_matrix properties: 20 entry(ies)
#      M   N   K   AType    BType    CType    ResultType  scope     saturating
─────────────────────────────────────────────────────────────────────────────────
0     16  16  16   E4M3     E4M3     FP32     FP32        Subgroup  no
1     16  16  16   E4M3     E5M2     FP32     FP32        Subgroup  no
2     16  16  16   E5M2     E4M3     FP32     FP32        Subgroup  no
3     16  16  16   E5M2     E5M2     FP32     FP32        Subgroup  no
4     16  16  16   U8       U8       U32      U32         Subgroup  no
5     16  16  16   U8       U8       I32      I32         Subgroup  no
6     16  16  16   U8       U8       I32      I32         Subgroup  yes
7     16  16  16   U8       I8       U32      U32         Subgroup  no
8     16  16  16   U8       I8       I32      I32         Subgroup  no
9     16  16  16   U8       I8       I32      I32         Subgroup  yes
10    16  16  16   I8       U8       U32      U32         Subgroup  no
11    16  16  16   I8       U8       I32      I32         Subgroup  no
12    16  16  16   I8       U8       I32      I32         Subgroup  yes
13    16  16  16   I8       I8       U32      U32         Subgroup  no
14    16  16  16   I8       I8       I32      I32         Subgroup  no   ← target
15    16  16  16   I8       I8       I32      I32         Subgroup  yes
16    16  16  16   FP16     FP16     FP16     FP16        Subgroup  no
17    16  16  16   BF16     BF16     BF16     BF16        Subgroup  no
18    16  16  16   FP16     FP16     FP32     FP32        Subgroup  no
19    16  16  16   BF16     BF16     FP32     FP32        Subgroup  no
```

(`vulkaninfo` only prints `cooperativeMatrixSupportedStages = COMPUTE`
in its main output and doesn't expand the per-config list — the
existing Rust probe is the source of truth.)

The brief's "FP16/BF16 only" hypothesis was wrong. RADV on Mesa 26.0
exposes the full RDNA4 WMMA matrix, **including the integer paths**
that ROCmForge reaches through HIP. There is no Vulkan-vs-HIP feature
hole for this op family.

### Step 2 — GLSL compile probe

Smoke shader `vk_shaders/probe_int8_coopmat.comp` (35 LOC):

```glsl
#version 450
#extension GL_KHR_cooperative_matrix             : require
#extension GL_KHR_memory_scope_semantics         : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(local_size_x = 64) in;
layout(set=0, binding=0) readonly  buffer A_Buf { int8_t  a_data[]; };
layout(set=0, binding=1) readonly  buffer B_Buf { int8_t  b_data[]; };
layout(set=0, binding=2) writeonly buffer C_Buf { int32_t c_data[]; };

void main() {
    coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> A;
    coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> B;
    coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> C =
        coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);

    coopMatLoad(A, a_data, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(B, b_data, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    C = coopMatMulAdd(A, B, C);
    coopMatStore(C, c_data, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
}
```

Compiles via `shaderc` with no warnings. SPV size: 3 540 bytes.

`spirv-dis probe_int8_coopmat.spv | head -10`:
```
OpCapability Shader
OpCapability Int8
OpCapability StorageBuffer8BitAccess
OpCapability VulkanMemoryModel
OpCapability CooperativeMatrixKHR
OpExtension "SPV_KHR_cooperative_matrix"
```

Both `Int8` and `CooperativeMatrixKHR` capabilities are emitted. The
SPIR-V is well-formed for the Int8-coopmat path.

### Step 3 — Runtime smoke

Wired the probe shader into `shaders.rs::ShaderId::ProbeInt8Coopmat`,
`pipeline_registry.rs` (no spec-constants), and a one-shot test:

```
test_int8_coopmat_runtime_smoke:
    A = i8[16][16] = all 1
    B = i8[16][16] = all 1
    Dispatch (1, 1, 1) of probe_int8_coopmat
    C = i32[16][16] expected all 16   (sum_k=0..15 of 1·1)
```

Result:
```
test test_int8_coopmat_runtime_smoke ... ok
```

256/256 cells equal 16. Pipeline create succeeded (no driver crash).
Dispatch succeeded. The compute output is correct.

The `coopmat<int8_t,...>` GLSL path **works end-to-end** on RDNA4 +
RADV + shaderc. No spec compliance asterisks.

## Why this is the right next step

Re-derived from Sprint 11E's analysis of *why* COOPMAT mul_mm lost
to mul_mmq+L:

1. **3.5× more activation BW** when COOPMAT path uses FP32-in-FP16-LDS.
   Int8 coopmat eliminates this — Q8_1 activations stay int8 all the
   way into the B fragment.
2. **Q4_K weight dequant overhead** before feeding FP16 fragments. Int8
   coopmat skips it — Q4_K can be unpacked directly to int8 (each 4-bit
   nibble extends to a signed byte) into the A fragment, with the
   per-block `d`/`m` scales applied to the int32 accumulator at the end.
3. **Higher LDS pressure** with FP16-LDS for both A and B. Int8 LDS is
   half the footprint → potentially restores the 3-4 WG/CU occupancy
   that mul_mmq enjoys.

So Int8-coopmat is the only candidate that might *combine* mul_mmq's
BW/occupancy advantage with coopmat's WMMA throughput. Sprint 11E
demonstrated FP-coopmat alone doesn't get there. Sprint 11G on
Int8-coopmat is the next experiment worth running.

## Files touched

```
NEW   vk_shaders/probe_int8_coopmat.comp   (35 LOC)
EDIT  build.rs                              (+8 lines: ShaderJob entry)
EDIT  src/backend/vulkan/shaders.rs         (+5 lines: ShaderId, name+bytes,
                                              ALL_SHADERS list)
EDIT  src/backend/vulkan/pipeline_registry.rs
                                            (+5 lines: ProbeInt8Coopmat branch
                                              with no spec-constants)
EDIT  tests/correctness.rs                  (+57 lines:
                                              test_int8_coopmat_runtime_smoke)
NEW   results/v021_sprint11f_sint8_coopmat_probe.md  (this report)
```

SPV count: 61 (was 60), +3 540 bytes total.

## Tests / regression

```
test result: ok. 27 + 9 + 18 + 74 + 8 + 8 + 27 = 171 passed
```

171/171 green. No production code paths touched — probe is wired into
the registry but not dispatched outside its dedicated smoke test.

## Sprint 11G — recommended scope

Eigenbau KHR-coopmat-int-DP GEMM. Patterns to merge:

1. **mul_mmq.comp shape** for Q4_K weight unpack into LDS / int8
   fragments (existing 4-bit-to-int8 dequant helper in
   `mul_mmq_funcs.glsl`).
2. **mul_mm.comp + COOPMAT shape** for the coopmat A/B/Accumulator
   fragment chain, with the extension list from Sprint 11E and Sprint
   10C plus `GL_EXT_shader_explicit_arithmetic_types_int8/int32`.
3. **Q8_1 B-buffer staging** — read int8 quant bytes directly, fold
   the per-block `d`, `s` scales into the int32 accumulator at the
   end of each K-tile (saving the FP16 conversion).
4. **Spec constants** mirroring `warptile_mmq_int_k` AMD-coopmat-
   override (Sprint 11C's L tile, with TM=TN=TK=16 from the coopmat
   fragment dim).

Estimated effort: **2–3 weeks** (Sprint 10C class — new shader, new
parity tests, A/B sweep, then default-flip if it wins). Risk: medium
— same fragment-mixing gotchas Sprint 10D hit, plus the new shape of
mixing scalar-int8 weight dequant with coopmat fragment loads.

Highest-payoff candidate left for prefill GEMM. Recommended next.

## Take-aways

1. **The 4-instruction sprint paid off again.** Probe → compile →
   smoke → report. 1 hour wall time, GO/NO-GO answered with high
   confidence.
2. **Vulkan/RADV is NOT the ceiling on RDNA4 Int8 WMMA.** All the
   Int8 fragment configs HIP/ROCm exposes via the underlying ISA are
   advertised through `VK_KHR_cooperative_matrix` and dispatchable.
3. **Sprint 11G is the right next sprint.** No more pre-check hits to
   look for in this neighbourhood — the Int8-coopmat path is real,
   un-shipped by anyone (llama.cpp upstream included), and could
   genuinely close meaningful share of the prefill GEMM gap.
