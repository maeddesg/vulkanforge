# v0.3.3 Sprint 18-M0 — Native FP8 smoke test (Go/No-Go)

**Verdict:** hardware + toolchain ready. **One concrete blocker
identified**: ash 0.38 doesn't yet ship `VK_EXT_shader_float8`
bindings, so Tests 4–5 (round-trip + WMMA matmul, both of which
need the FP8 feature enabled at device create) are deferred until
either (a) ash bumps to a version with FP8, or (b) we add ~30 LOC
of raw FFI for the feature struct.

The brief's premises also got two factual things wrong on the way
in; both are now corrected and documented in this report.

## Hardware + driver story (TL;DR)

| Layer                  | Status   | Notes |
|------------------------|----------|-------|
| RX 9070 XT (gfx1201)   | ✅ READY | `v_wmma_f32_16x16x16_fp8_fp8` confirmed in RDNA4 ISA |
| RADV (Mesa 26.0.6)     | ✅ READY | `vulkaninfo`: `shaderFloat8 = true`, `shaderFloat8CooperativeMatrix = true` |
| Vulkan headers (1.4.341)| ✅ READY | `VK_EXT_shader_float8` defined; FP8 component types in the coopmat enum |
| glslang 16.2.0 (1.4.341)| ✅ READY | `GL_EXT_float_e4m3` / `GL_EXT_float_e5m2`; types `floate4m3_t` / `floate5m2_t`; SPIR-V codegen emits the FP8 capability |
| ash 0.38.0+1.3.281     | ❌ GAP   | Latest crates.io release; tracks Vulkan SDK 1.3.281, predates FP8 |

## Test results

```
=== TEST 1: ash + Vulkan SDK versions ===
ash crate         : 0.38.0+1.3.281 (Cargo.toml pin)
System headers    : VK_HEADER_VERSION 341 (1.4.341, has VK_EXT_shader_float8)
ash FP8 structs   : ABSENT (PhysicalDeviceShaderFloat8FeaturesEXT,
                    FLOAT8_E4M3_EXT, FLOAT8_E5M2_EXT not in ash 0.38)
Driver advertises : shaderFloat8 + shaderFloat8CooperativeMatrix

=== TEST 2: cooperative-matrix property table — 20 entries ===
  [ 0]  A=FP8_E4M3   B=FP8_E4M3   C=FP32    R=FP32    M=16 N=16 K=16
  [ 1]  A=FP8_E4M3   B=FP8_E5M2   C=FP32    R=FP32    M=16 N=16 K=16
  [ 2]  A=FP8_E5M2   B=FP8_E4M3   C=FP32    R=FP32    M=16 N=16 K=16
  [ 3]  A=FP8_E5M2   B=FP8_E5M2   C=FP32    R=FP32    M=16 N=16 K=16
  [ 4-15] INT8/UINT8 mixed, INT32/UINT32 acc, M=N=K=16 (Phase 11 INT8 path)
  [16, 18] FP16 → FP16 + FP16 → FP32, M=N=K=16 (current coopmat prefill path)
  [17, 19] BF16 → BF16 + BF16 → FP32, M=N=K=16

Counts: FP8=4, BF16=2, INT8/UINT8=12, FP16=2
PASS — driver exposes 4 FP8 coopmat entries (all 4 sign combinations)

=== TEST 3: glslangValidator FP8 shader compile ===
#version 450
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_float_e5m2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
... uintBitsToFloate4m3EXT / uintBitsToFloate5m2EXT ...

exit status: 0
spirv-dis: SPIR-V DECLARES the FP8 capability
PASS — FP8 shader compiles end-to-end

=== TEST 4: round-trip CPU ↔ GPU — DEFERRED ===
Reason: would need to enable PhysicalDeviceShaderFloat8FeaturesEXT
at vkCreateDevice time. ash 0.38 doesn't ship that struct.

=== TEST 5: FP8 cooperative-matrix matmul — DEFERRED ===
Reason: same as Test 4 plus needs the FP8 GLSL coopmat extension
(GL_EXT_float_cooperative_matrix? — separate from the type
extensions; spec name TBD). Need both the device feature AND the
GLSL extension chain to compile a FP8-input coopmat shader.
```

## Two brief errors caught on the way

**Error 1 — wrong numeric constants for the FP8 component types.**
The brief stated `VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT = 1000491000`
and `_E5M2_EXT = 1000491001`. Per the actual `vulkan_core.h`
shipped with `vulkan-headers 1.4.341`:

```c
VK_COMPONENT_TYPE_SINT8_PACKED_NV = 1000491000,
VK_COMPONENT_TYPE_UINT8_PACKED_NV = 1000491001,
VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT = 1000491002,
VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT = 1000491003,
```

The `_NV` codes are an unrelated NVIDIA cooperative-vector
extension. With the wrong constants Test 2 first reported four
`UNKNOWN` entries and "FP8=0" — the smoke test is now updated
with the correct codes and reports `FP8=4`.

**Error 2 — wrong GLSL extension name.**
The brief proposed `#extension GL_EXT_shader_float8 : require`
and types `float8_e4m3fn_t`. glslang doesn't recognize either.
Per `strings libglslang.so.16`, the actual upstream extensions
(Khronos glslang PR #3969) are:

```glsl
#extension GL_EXT_float_e4m3 : require   // floate4m3_t
#extension GL_EXT_float_e5m2 : require   // floate5m2_t
```

with bit-cast helpers `uintBitsToFloate4m3EXT(uint8_t)`,
`floate4m3BitsToUintEXT(floate4m3_t)`, vec types
`fe4m3vec2/3/4` and `fe5m2vec2/3/4`. The smoke test compiles
cleanly with these.

## What's actually blocking Sprint 18A / 18B

ash 0.38.0+1.3.281 is the latest published version on crates.io
(verified via `cargo info ash`); ash HEAD on GitHub also has no
`extensions/ext/shader_float8.rs` and no `Float8` symbols. Until
ash regenerates against Vulkan SDK 1.4.x, we have three options:

1. **Wait for ash 0.39 / 0.40** — straightforward but
   schedule-dependent.
2. **Patched ash via `[patch.crates-io] ash = { git = ... }`** —
   if anyone has an open PR that bumps the SDK target.
3. **Raw FFI shim** — declare
   `#[repr(C)] PhysicalDeviceShaderFloat8FeaturesEXT { ... }`
   ourselves (~30 LOC), unsafe-push it into the `pNext` chain at
   `vkCreateDevice` time, and add the `1000491002 / 003` raw u32
   match arms to our coopmat helpers. Doable in a day but it's
   maintenance debt every time ash bumps and the names overlap.

Recommended: **option 3** for one-sprint smoke test; **option 1
or 2** for production.

## Path forward

| Sprint | Goal                                  | Blocker        |
|--------|---------------------------------------|----------------|
| 18-M0  | This smoke test                       | ✅ Done         |
| 18-M1  | FP8 device-feature enablement (Test 4) | ash gap (option 3 or wait) |
| 18-M2  | FP8 cooperative-matrix matmul (Test 5) | + FP8 coopmat GLSL ext name |
| 18A    | FP8 KV cache                          | needs 18-M1     |
| 18B    | FP8 WMMA prefill (the headline)       | needs 18-M2     |

## Files

- `examples/fp8_smoke.rs` — new (235 LOC, all 3 + 2 deferred tests)
- `results/v033_sprint18_m0_fp8_smoke.md` — this report

No production code touched. No `Cargo.toml` change.

## Run

```bash
cargo run --release --example fp8_smoke
```
