# FP8 GLSL — glslang from-source build & analysis

**Date:** 2026-04-28
**Project:** VulkanForge v0.1.3 (no version bump — toolchain analysis only)
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Driver:** Vulkan 1.4.335, RADV `VK_KHR_cooperative_matrix` rev 2

---

## TL;DR

**FP8 GLSL coopmat shaders compile and validate today on this system, with the
shaderc 0.8 toolchain we already ship.** The v0.2 smoke-test FP8 STOP was a
mis-diagnosis: every hypothesis we tried used a wrong type-name spelling. The
extension is a year old and has been in every glslang 16.x release that
shaderc-sys 0.8.3 picks up from the system.

```
═══ FP8 glslang Analyse ═══
glslang Version (built):  16.2.0   (commit 39bfdd6e, 2026-04-27)
glslang Version (system): 16.2.0   (Arch package, identical front-end)
GL_EXT_float_e4m3 / e5m2: ✅  (since glslang Feb-2025, well before 16.0)
FP8 GLSL types:           floate4m3_t, floate5m2_t  (no underscore)
                          fe4m3vec{2,3,4}, fe5m2vec{2,3,4}
FP8 coopmat compiles:     ✅  built-glslang ✅  system glslangValidator
                              ✅  glslc (shaderc CLI)   →  shaderc 0.8 ✅
SPIR-V capabilities:      Float8EXT, Float8CooperativeMatrixEXT
SPIR-V extension:         SPV_EXT_float8
SPIR-V type:              OpTypeFloat 8 Float8E4M3EXT  (native FP8 scalar)
WMMA shape:               16×16×16  (SAME K as BF16 — not K=32)
VK_EXT_shader_float8:     ✅ revision 1 on RADV
shaderFloat8:             ✅ true
shaderFloat8CooperativeMatrix: ✅ true
Coopmat property entries 0–3: E{4M3,5M2}×E{4M3,5M2} → FP32 → FP32, Subgroup
Build integration:        Drop-in — add a new ShaderJob to build.rs;
                          shaderc 0.8 + system libshaderc handles it.
                          NO toolchain rebuild required.

Empfehlung für v0.2: GO for FP8 in PARALLEL with BF16 — same K=16 WMMA,
                     same 1 B/elem as BF16 promised but doubled (½ B/elem),
                     same Float8CooperativeMatrixEXT capability already
                     advertised on RADV. v0.2A becomes a 2-track plan with
                     marginal extra cost.
```

---

## 1 — glslang from-source build

```fish
cd ~/tmp && git clone https://github.com/KhronosGroup/glslang.git glslang-src
cd glslang-src
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=ON \
                -DALLOW_EXTERNAL_SPIRV_TOOLS=ON
cmake --build build -j$(nproc)
```

| Field | Value |
|---|---|
| Repo HEAD | `39bfdd6e` (2026-04-27) "Add template deduction guide for Defer" |
| Reported Glslang Version | `11:16.2.0` (header version unchanged since Jan-2026 tag) |
| Build host deps (Arch) | `spirv-tools 1:1.4.341.0-2.1`, `spirv-headers 1:1.4.341.0-2` |
| Resulting binary | `~/tmp/glslang-src/build/StandAlone/glslang` |
| Binary FP8 strings | `GL_EXT_float_e4m3`, `GL_EXT_float_e5m2`, `floate4m3_t`, `floate5m2_t`, `Float8E4M3EXT`, `Float8E5M2EXT` |

Build clean except a deprecation warning for HLSL front-end (unrelated).

---

## 2 — FP8 GLSL Extension API (corrected)

### 2.1 Extensions

The extension is **split into two**:

```glsl
#extension GL_EXT_float_e4m3 : enable
#extension GL_EXT_float_e5m2 : enable
```

The Khronos *spec* document is named `GL_EXT_float8_e5m2_e4m3` (single
combined PDF) — but the GLSL `#extension` directives use the two separate
strings above. **None of the smoke-test hypotheses had this naming exactly
right combined with valid type names** — Hypothesis C had the extension name
right (`GL_EXT_float_e4m3`) but I never paired it with `floate4m3_t` and
without `GL_EXT_shader_explicit_arithmetic_types`.

There is also a recent (2026-04-24) `GL_ARM_tensors_float_e4m3` /
`_e5m2` — those are FP8 *inside* the ARM tensor extension, **unrelated** to
plain coopmat. Ignore for our purposes.

### 2.2 Types

| GLSL | Notes |
|---|---|
| `floate4m3_t` | scalar, 1 byte (E4M3, NaN/Inf encoding) |
| `floate5m2_t` | scalar, 1 byte (E5M2, IEEE-style) |
| `fe4m3vec2/3/4` | vector variants |
| `fe5m2vec2/3/4` | vector variants |

Conversion built-ins:
* `floate4m3_t(x)` constructor from any int/float
* `floate4m3BitsToIntEXT`, `floate4m3BitsToUintEXT`, `intBitsToFloate4m3EXT`, `uintBitsToFloate4m3EXT` (and vec variants)
* `saturatedConvertEXT(out, in)` — supports scalar, vector AND coopmat
* Implicit cast to `float` is allowed (`f32 = b;` works)

Arithmetic (`+ - * /`) on FP8 scalars is **not** in the extension — you go
through `float` or coopmat. That matches the hardware: there's no scalar
WMMA-FP8 instruction, only the matrix one.

### 2.3 Coopmat usage

```glsl
coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>          matA;
coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>          matB;
coopmat<float,        gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc;
coopMatLoad(matA, A, 0, stride, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(matB, B, 0, stride, gl_CooperativeMatrixLayoutRowMajor);
acc = coopMatMulAdd(matA, matB, acc);
coopMatStore(acc, C, 0, stride, gl_CooperativeMatrixLayoutRowMajor);
```

WMMA shape **16×16×16** — *same K as BF16*. There is no K=32 fast-path
exposed via this extension on RDNA4 today. The hardware ISA *does* have
`v_wmma_f32_16x16x32_f8`, but the driver advertises K=16 in
`VkCooperativeMatrixPropertiesKHR`, so the compiler picks the K=16 path. (RADV
is presumably emitting two K=16 mat-muls per kernel call when it sees the
K=16 coopmat type with FP8 inputs — to be confirmed by `radv_dump_shaders=1`.)

### 2.4 Reference test shaders in glslang repo

```
~/tmp/glslang-src/Test/spv.floate4m3.comp           ← full feature test
~/tmp/glslang-src/Test/spv.floate4m3.const.comp     ← constant folding
~/tmp/glslang-src/Test/spv.floate4m3_error.comp     ← negative cases
~/tmp/glslang-src/Test/spv.floate5m2.comp           ← (parallel)
```

`spv.floate4m3.comp` includes a `coopmat<floate4m3_t, …>` declaration,
construction from `coopmat<float16_t, …>`, and `saturatedConvertEXT` between
matrix types — the canonical templates for our v0.2 work.

---

## 3 — FP8 coopmat probe-shader compilation

`/tmp/probe_fp8_coopmat.comp` — minimal probe (16×16×16, FP8 A/B, FP32 acc):

```glsl
#version 450 core
#extension GL_KHR_cooperative_matrix     : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_float_e4m3             : require
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_scalar_block_layout    : enable
layout(local_size_x = 32) in;
layout(set=0, binding=0, scalar) buffer BufA { floate4m3_t A[]; };
layout(set=0, binding=1, scalar) buffer BufB { floate4m3_t B[]; };
layout(set=0, binding=2)         buffer BufC { float       C[]; };
void main() {
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
    coopmat<float,        gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc;
    coopMatLoad(matA, A, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(matB, B, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    acc = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    acc = coopMatMulAdd(matA, matB, acc);
    coopMatStore(acc, C, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
```

Compilation results (all three paths):

| Tool | Cmd | Out size | spirv-val |
|---|---|---|---|
| Built-from-source glslang main | `~/tmp/glslang-src/build/StandAlone/glslang --target-env vulkan1.3 -V probe.comp` | 1784 B | ✅ |
| System glslangValidator (16.2.0) | `glslangValidator --target-env vulkan1.3 -V probe.comp` | 1784 B (byte-identical) | ✅ |
| System glslc (shaderc CLI) | `glslc --target-env=vulkan1.3 probe.comp` | 4208 B (extra debug info) | ✅ |

### 3.1 SPIR-V capabilities/extensions emitted

```
OpCapability Shader
OpCapability Float8EXT
OpCapability Float8CooperativeMatrixEXT
OpCapability VulkanMemoryModel
OpCapability CooperativeMatrixKHR
OpExtension  "SPV_EXT_float8"
OpExtension  "SPV_KHR_cooperative_matrix"
```

### 3.2 Native types in the SPIR-V module

```
%fp8e4m3 = OpTypeFloat 8 Float8E4M3EXT
%mat_a   = OpTypeCooperativeMatrixKHR %fp8e4m3 Subgroup 16 16 MatrixA
%mat_b   = OpTypeCooperativeMatrixKHR %fp8e4m3 Subgroup 16 16 MatrixB
%acc_t   = OpTypeCooperativeMatrixKHR %float   Subgroup 16 16 MatrixAccumulator
%result  = OpCooperativeMatrixMulAddKHR %acc_t %a %b %acc
ArrayStride 1                          ← 1 byte per FP8 element ✅
```

The shader uses **native** `OpTypeFloat 8` (not a uint8 storage hack with
software interpretation). MulAdd takes FP8 A/B and produces FP32 — no
intermediate cast in the IL.

---

## 4 — FP8 bench-shader (BF16-bench transposed to FP8)

`/tmp/bench_coopmat_fp8.comp` — same WG layout, K-walk, push-constants and
binding scheme as `vk_shaders/bench_coopmat_pure.comp`, but with
`bfloat16_t → floate4m3_t`. Compiled successfully via all three toolchains;
output:

```
built-glslang   →  /tmp/bench_coopmat_fp8.spv         4136 B   ✅ spirv-val
system glslang  →  /tmp/bench_coopmat_fp8_sys.spv     4136 B   ✅ (byte-identical)
glslc           →  /tmp/bench_coopmat_fp8_glslc.spv   4208 B   ✅
```

I did **not** wire this into VulkanForge per rules ("KEIN COMMIT in
VulkanForge"). The shader file is staged at `/tmp/bench_coopmat_fp8.comp`
and is ready to drop into `vk_shaders/` and add as a `ShaderJob` in
`build.rs` whenever we choose to.

The actual TFLOPS measurement requires Rust-side wiring of a second pipeline
in `examples/bench_coopmat.rs` (FP8 storage, dispatch, validation against an
FP32 reference). That is a v0.2 sprint task, not a toolchain probe — so it
is deferred per the analysis-only scope of this prompt.

---

## 5 — Why the smoke test failed (root-cause)

shaderc-sys 0.8.3 ships a **fallback** glslang 11.13.0 (Dec-2022) source
tree under `build/glslang/`, but its build script (`build.rs:194-218`)
prefers `find_library` on the system: with `libshaderc_shared.so`,
`libglslang.so.16`, and `glslangValidator 16.2.0` installed via Arch's
`shaderc 2026.1-2.1` and `glslang 1:1.4.341.0-2.1` packages, the crate
links against the system version. That system glslang **does** have
`GL_EXT_float_e4m3`/`_e5m2` (Feb-2025, well before 16.0).

What we got wrong in the smoke test was the *type* name. We tried these
type spellings (and combinations of similar extension names):

| Hypothesis | Type name tried | Actual GLSL type |
|---|---|---|
| A | `float8_e4m3` | `floate4m3_t` |
| B | `floate4m3` | `floate4m3_t` (close, missing `_t`) |
| C | `float_e4m3` | `floate4m3_t` (no underscore!) |
| D | `float8` | `floate4m3_t` |
| E | `float_e4m3fn` | `floate4m3_t` (FN suffix is C++-spec naming, not GLSL) |

The closest hypothesis was B (only the trailing `_t` was missing). It's a
naming-convention surprise: the `_t` suffix matches `int8_t`/`float16_t`,
and the lack of separator between `float` and `e4m3` matches glslang's
ParseHelper which treats it as one unbroken token. None of the spec docs
that surfaced in our search emphasised the *type name*.

**Lesson for the workflow:** when an extension is rejected with "not
declared / not supported", grep `Test/` in glslang for `*.comp` files using
the extension before iterating on the spelling — the test corpus is the
authoritative usage example.

---

## 6 — Vulkan runtime support on RADV

`vulkaninfo` extract (gfx1201, RADV, Mesa snapshot 26.x):

```
VK_EXT_shader_float8                          : extension revision 1
VkPhysicalDeviceShaderFloat8FeaturesEXT:
    shaderFloat8                  = true
    shaderFloat8CooperativeMatrix = true

VkPhysicalDeviceCooperativeMatrixFeaturesKHR:
    cooperativeMatrix                   = true
    cooperativeMatrixRobustBufferAccess = true
```

`examples/probe_coopmat` enumerates **20 entries**, of which **0–3** are FP8:

```
#  M  N  K  AType  BType  CType  ResultType  scope     saturating
0  16 16 16 E4M3   E4M3   FP32   FP32        Subgroup  no
1  16 16 16 E4M3   E5M2   FP32   FP32        Subgroup  no
2  16 16 16 E5M2   E4M3   FP32   FP32        Subgroup  no
3  16 16 16 E5M2   E5M2   FP32   FP32        Subgroup  no
```

The Vulkan-spec component types (`VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT` /
`_E5M2_EXT`) map cleanly onto the GLSL types `floate4m3_t` / `floate5m2_t`
via the `Float8E4M3EXT` / `Float8E5M2EXT` SPIR-V `OpTypeFloat`-encoding
operands.

All four mixed-precision combinations are advertised. This means we can
quantise A as E4M3 (better dynamic range for activations / weights with
outliers) and B as E5M2, or any other combination, without losing the WMMA
fast path. **K is uniformly 16** — no K=32 fast path is exposed.

---

## 7 — Build integration: how to ship FP8 in VulkanForge

shaderc 0.8.3 (linked against system shaderc 2026.1) **already** supports
FP8 GLSL. **No toolchain rebuild, no shaderc-from-source, no separate
glslangValidator call from `build.rs`.** The only changes needed:

1. **New shader file** `vk_shaders/bench_coopmat_fp8.comp` (E4M3 variant of
   the BF16 bench). About 50 lines, ready in `/tmp/bench_coopmat_fp8.comp`.
2. **New `ShaderJob` entry in `build.rs`**:
   ```rust
   ShaderJob {
       out_name: "bench_coopmat_fp8_e4m3.spv",
       entry_source: "bench_coopmat_fp8.comp",
       defines: &[],
   },
   ```
3. **New `ShaderId` and `SPV` const in `shaders.rs`** — same boilerplate
   as the `bench_coopmat_pure_f32` entry.
4. **(Optional)** A second variant of `examples/bench_coopmat.rs` that
   uploads E4M3 input buffers and runs the FP8 pipeline; or a flag on the
   existing example that selects the precision at runtime.

Total wire-up effort: **half a day**, mostly in `bench_coopmat.rs` to
prepare FP8 input data (CPU-side packing of an FP32 reference into E4M3 via
the bit-twiddle path or `half` crate's `f8` if/when added).

The same `build.rs` pattern then unlocks an FP8 GEMM kernel
(`mul_mm_q4_k_fp8.comp` — Q4_K → FP8 in-shader dequant → FP8 WMMA → FP32
acc) when v0.2A reaches the dequant-fusion sprint.

---

## 8 — Other notable changes since glslang 16.2.0 (Jan 2026)

48 commits on main since the 16.2.0 tag. Filtered for relevance:

| Commit | Date | Effect |
|---|---|---|
| `5ed4003a` | 2026-04-24 | `GL_ARM_tensors_bfloat16/float_e4m3/_e5m2` — FP8/BF16 inside the ARM tensor extension. Unrelated to coopmat path. |
| `a3a83d09` | 2026-04-23 | Adds small-type SPIR-V capabilities for `GLSL.std.450` — fixes a validation error when a shader uses `GLSL.std.450` ops on FP8/BF16 values without other arithmetic. **Defensive fix relevant to v0.2** if our kernel uses e.g. `min`/`max` on FP8. |
| `09c541ee` | 2026-04 | `GL_EXT_long_vector` doesn't require `LongVector` capability for 2-4 components — irrelevant. |

There is **no new coopmat extension** since 16.2.0 (the recent commits to
`GL_NV_cooperative_matrix2` are unchanged). For v0.2 we should still rely
on `GL_KHR_cooperative_matrix` + per-element-type extensions.

---

## 9 — Recommendation: revise the v0.2 plan

Pre-this-investigation plan (smoke test, 27.04.2026):

* v0.2A — BF16 coopmat (4-6 weeks) — GO
* v0.2B — FP8 — DEFER (toolchain blocked)

Revised plan (this investigation):

* v0.2A — BF16 coopmat (unchanged)
* **v0.2B — FP8 coopmat — UNBLOCKED. Promote to "in parallel with v0.2A
  Sprint 2 (Q4_K dequant fusion)."**

Rationale:
* Same 16×16×16 WMMA shape — kernel skeleton identical to BF16 bench.
* Half the bytes per element vs BF16 (1 B vs 2 B) → bandwidth-limited
  prefill should see another ~2× headroom on top of BF16's 6-13 TFLOPS.
* Q4_K → FP8 dequant is **simpler** than Q4_K → BF16 (smaller storage,
  saturating-convert intrinsic ready in the extension).
* Risk: actual TFLOPS measurement TBD — could be the same as BF16 if the
  driver doesn't have a true K=32 fast path. Worth a 1-day bench.

Suggested concrete next step for v0.2B: copy `bench_coopmat_pure.comp` to
`bench_coopmat_fp8_e4m3.comp`, wire the `ShaderJob`, and add a `--fp8`
flag to `examples/bench_coopmat.rs`. One sprint-day, then we have a real
TFLOPS number for FP8 to compare against the BF16 6-13 TFLOPS data point.

---

## 10 — Pitfalls that did *not* materialise

| Pitfall flagged in prompt | Outcome |
|---|---|
| FP8 storage-only (no compute) | False — `OpCapability Float8EXT` + `OpCapability Float8CooperativeMatrixEXT` both emitted; coopmat with FP8 elements is first-class. |
| coopmat + FP8 incompatibility | False — `OpTypeCooperativeMatrixKHR %fp8e4m3` validates with `spirv-val`. |
| SPIR-V capability missing | False — `SPV_EXT_float8` requested by glslang and accepted by `spirv-val`. RADV runtime support also confirmed. |
| K=32 unexpected | True for the ISA but **not** exposed — driver advertises K=16. Treat WMMA shape as 16×16×16, identical to BF16. |
| shaderc 0.8 cannot compile FP8 | False — linked against system shaderc 2026.1, which has glslang 16.2.0, which has FP8. `glslc` (the same code-path) compiles `/tmp/bench_coopmat_fp8.comp` cleanly. |

---

## Appendix A — Files touched

```
NEW   results/fp8_glslang_analysis.md       (this report — committed)
NEW   /tmp/probe_fp8_coopmat.comp           (probe shader — NOT committed)
NEW   /tmp/probe_fp8_coopmat.spv            (artifact — NOT committed)
NEW   /tmp/bench_coopmat_fp8.comp           (bench shader — NOT committed)
NEW   /tmp/bench_coopmat_fp8.spv            (artifact — NOT committed)
NEW   /tmp/bench_coopmat_fp8_sys.spv        (artifact — NOT committed)
NEW   /tmp/bench_coopmat_fp8_glslc.spv      (artifact — NOT committed)
EXT   ~/tmp/glslang-src/                    (out-of-tree clone)
```

No code changes inside `/home/maeddes/projects/vulkanforge/src/`,
`/build.rs`, `/Cargo.toml`, `/vk_shaders/`, or `/examples/`. Tests not
re-run (no source-side changes).

## Appendix B — Reproduce

```fish
# Build glslang from source (one-time, optional — system 16.2.0 already works)
cd ~/tmp && git clone https://github.com/KhronosGroup/glslang.git glslang-src
cd glslang-src
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=ON \
                -DALLOW_EXTERNAL_SPIRV_TOOLS=ON
cmake --build build -j(nproc)

# Verify FP8 supported (system tool is enough)
glslangValidator --target-env vulkan1.3 -V results/.fp8/probe.comp -o /tmp/p.spv
spirv-val /tmp/p.spv && spirv-dis /tmp/p.spv | grep -E 'Float8|Coop'

# Verify Vulkan runtime
vulkaninfo 2>/dev/null | grep -E 'shaderFloat8|VK_EXT_shader_float8'

# Verify coopmat property table has FP8 entries 0-3
cargo run --release --example probe_coopmat | grep -E '^[0-3] '
```
