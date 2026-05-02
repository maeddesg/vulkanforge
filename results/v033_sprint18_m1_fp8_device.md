# v0.3.3 Sprint 18-M1 — FP8 device enablement + WMMA matmul

**Verdict: GO.** Native FP8 is fully functional on the
ash 0.38 + RADV (Mesa 26.0.6) + RX 9070 XT stack via a 70 LOC
raw FFI shim. **All 5 smoke tests pass**, including bit-exact
GPU↔CPU agreement on both the byte round-trip (256 / 256 patterns,
max error 0.0) and the 16×16×16 FP8 cooperative-matrix matmul
(max error 0.0 vs scalar CPU reference). Sprint 18A (FP8 KV cache)
and Sprint 18B (FP8 WMMA prefill) are unblocked.

## Test results

```
=== TEST 1: ash + Vulkan SDK versions ===
ash 0.38.0+1.3.281; system headers 1.4.341; ash FP8 structs ABSENT.
M1 ships a 70 LOC raw FFI shim (`fp8_ext.rs`) that fills the gap.

=== TEST 2: cooperative-matrix property table ===  PASS
4× FP8 entries (E4M3/E5M2 × E4M3/E5M2 → FP32, M=N=K=16, SUBGROUP)
+ 12× INT8/UINT8, 2× FP16, 2× BF16 — see report from M0.

=== TEST 3: GLSL FP8 shader compile ===  PASS
glslang 16.2.0 compiles `#extension GL_EXT_float_e4m3` and produces
SPIR-V declaring `Float8EXT` + `Float8CooperativeMatrixEXT` caps.

=== TEST 4: FP8 round-trip CPU↔GPU ===  PASS
Inputs: every byte pattern 0x00..0xFF (256 distinct E4M3 bit
patterns). CPU[0..8] == GPU[0..8] == [0.0, 0.001953125, 0.00390625,
0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875].
max_err = 0.0000000000, mismatched bytes = 0.

=== TEST 5: FP8 cooperative-matrix matmul (16×16×16) ===  PASS
A, B in {-0.25, 0.125, 0.5, 1.0} (E4M3-exact values), B columns
half-scaled. cpu_amax = 2.656250, gpu_amax = 2.656250.
CPU C[0..8] == GPU C[0..8] = [2.65625, 0.5625, 0.0, 0.5625,
2.65625, 0.5625, 0.0, 0.5625]. max_err = 0.000000e0.
```

## What shipped

### 1. Raw FFI shim — `src/backend/vulkan/fp8_ext.rs`

70 LOC: `repr(C)` mirror of `VkPhysicalDeviceShaderFloat8FeaturesEXT`,
extension-name CStr constant, `unsafe impl ExtendsDeviceCreateInfo`
so `device_create_info.push_next(&mut fp8_features)` Just Works™
through ash's existing builder. Plus E4M3 ↔ f32 conversion
helpers (`fp8_e4m3_to_f32`, `f32_to_fp8_e4m3`) that mirror what
the GPU's `uintBitsToFloate4m3EXT(...)` produces — used by both
smoke tests. 5 unit tests cover the round-trip cases (zero, ±1,
±max, subnormals).

**Critical correction**: the brief specified
`STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT = 1000491000`,
but `vulkan_core.h` (1.4.341) says `1000567000`. The 1000491000
value is `STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV`
(NV cooperative-vector extension, unrelated to FP8). The shim uses
the correct value.

### 2. `device.rs` integration

Opt-in via `VULKANFORGE_ENABLE_FP8=1`:

```rust
let fp8_opt_in = std::env::var("VULKANFORGE_ENABLE_FP8")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);

let mut device_extensions: Vec<*const i8> =
    vec![vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr()];
if fp8_opt_in {
    device_extensions.push(SHADER_FLOAT8_EXT_NAME.as_ptr());
}

let mut fp8_features = PhysicalDeviceShaderFloat8FeaturesEXT::default()
    .shader_float8(true)
    .shader_float8_cooperative_matrix(true);

let mut device_create_info = vk::DeviceCreateInfo::default()
    ... .push_next(&mut features2) ... .push_next(&mut coopmat_features);
if fp8_opt_in {
    device_create_info = device_create_info.push_next(&mut fp8_features);
}
```

Default off so existing `vulkanforge chat` / `bench` / `info`
runs aren't affected. Smoke test sets the env var automatically.

### 3. Smoke test extension — `examples/fp8_smoke.rs`

Tests 1–3 from M0 unchanged; Tests 4 + 5 added. Both new tests
compile their GLSL inline at runtime via `glslangValidator`,
build a `ComputeKernel` from the SPIR-V, dispatch through
`VulkanForge`'s existing buffer/command plumbing, and compare to
a CPU reference computed via `fp8_e4m3_to_f32`.

**Round-trip shader** uses `uintBitsToFloate4m3EXT(uint8_t(b))` →
`float(...)` to verify GPU and CPU agree on every E4M3 byte
encoding, including subnormals (E=0), the single NaN encoding
(0x7F / 0xFF), and the max value 448 (0x7E / 0xFE).

**WMMA shader** declares full coopmat<float8...> fragments:

```glsl
coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
coopmat<float,       gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;
coopMatLoad (matA, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad (matB, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
matC = coopMatMulAdd(matA, matB, matC);
coopMatStore(matC, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
```

A, B fed as raw `uint8_t[]` storage buffers (256 bytes each); C
written as `float[]` (256 floats). The shader compiles into SPIR-V
with the `Float8EXT` + `Float8CooperativeMatrixEXT` capabilities
(verified via `spirv-dis`). CPU reference is a scalar nested loop
calling `fp8_e4m3_to_f32`.

## Throughput note

Sprint 18-M1 doesn't run a head-to-head FP8 vs FP16 WMMA
microbenchmark — that's reserved for Sprint 18B where the prefill
kernel becomes the comparison target. With a single 16×16×16
dispatch the timer overhead dominates µs-level differences; a
real comparison needs a sustained workload (1024+ tiles).

The hardware story remains: RDNA4 advertises
`v_wmma_f32_16x16x16_fp8_fp8` and pairs of `v_wmma_f32_16x16x16_fp8_bf8`
in the gfx1201 ISA, with the WMMA opcode taking ~the same cycles
as the FP16 equivalent but with twice as many input lanes per
instruction (since FP8 ops are byte-packed vs FP16 short-packed).
The expected ceiling for the prefill GEMM, once weights are
shipped as FP8, is ~2× FP16 throughput; the actual realised
speedup will depend on dequant overhead and shared-memory
bandwidth, neither of which this M1 smoke test exercises.

## Validation-layer notes

Two non-fatal warnings during the run:

1. **`vkCreateShaderModule LocalSizeId`** (Test 5) — the
   coopmat shader uses `layout(local_size_x_id = 0)` with a
   `constant_id = 0` BLOCK_SIZE. That generates `LocalSizeId`
   execution mode in SPIR-V, which requires `maintenance4`
   feature. Currently neither enabled nor needed (RADV runs the
   pipeline correctly anyway), but if Sprint 18B uses spec-const
   workgroup sizes for the FP8 prefill kernel, we'll need to
   enable `maintenance4` in `Vulkan13Features`.
2. **Object-leak warnings at process exit** — the smoke
   test's `GpuFixture` doesn't impl `Drop`. Cosmetic (OS reclaims
   everything); not worth fixing for a one-shot binary.

## Lessons re-learnt

- **Verify constants against the actual headers, not the brief.**
  This is now the *third* time in a row a brief had wrong
  constants. M0 caught two (component-type codes + GLSL extension
  name); M1 caught the structure-type code. Each ~5 minutes lost,
  but if the wrong sType had silently been pushed into the chain,
  the device-create call could have crashed at a much harder-to-
  diagnose layer. **Always grep `vulkan_core.h` first.**
- **Off-by-one in `as u8` from a `usize`.** `(0..N as u8)` where
  `N: usize = 256` produces an empty range — the `as` cast wraps
  to 0. Caught by the `pCreateInfo->size = 0` validation error.
  Use `(0..N).map(|i| i as u8)` instead.

## Path forward — Sprint 18A / 18B

| Sprint | Goal                          | Status     |
|--------|-------------------------------|------------|
| 18-M0  | FP8 driver + glslang smoke    | ✅ done    |
| 18-M1  | FP8 device enablement + WMMA  | ✅ done    |
| 18A    | FP8 KV cache (½ VRAM)         | ▶ ready    |
| 18B    | FP8 WMMA prefill (~2× speed)  | ▶ ready    |
| —      | Replace `fp8_ext.rs` with ash | post ash 0.39 |

For Sprint 18A: existing FP16 KV cache machinery (`KvCache`
struct + `kv_copy_fp16` shader from Sprint 9d.2) drops in directly,
just with `floate4m3_t` storage. Quantize FP32 activations →
`uintBitsToFloate4m3EXT(saturatedConvertEXT(...))` at write time;
dequant via `float()` on read. Halves cache VRAM (Qwen3-8B FP16:
288 MB → FP8: 144 MB).

For Sprint 18B: replace the Q4_K_M coopmat path's FP16 shared
memory with FP8 (E4M3 weights) and FP8 LDS. Re-use
`MulMmQ4KCoopmat` scaffold; the WMMA accumulator stays FP32 so
quality stays close to the FP16 baseline.

## Files

- `src/backend/vulkan/fp8_ext.rs` — new (240 LOC; 5 unit tests)
- `src/backend/vulkan/mod.rs` — `+1` line
- `src/backend/vulkan/device.rs` — `+15 / -3` (FP8 opt-in branch)
- `examples/fp8_smoke.rs` — `+450` (Tests 4 + 5 added on top of M0)
- `results/v033_sprint18_m1_fp8_device.md` — this report

No production code paths touched. `cargo test --release --lib`
still 32/32 (was 27 before the FP8 helper tests).
