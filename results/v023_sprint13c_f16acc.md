# Sprint 13C — f16-Accumulator coopmat shader: HONEST NEGATIVE on RDNA4

**Premise.** After Sprint 13A, our coopmat pipeline-config is at parity
with llama.cpp (Q4_K + Q6_K, aligned + unaligned, S/M/L tiles, identical
warptiles). The remaining pp=512 prefill gap to llama.cpp (3863 vs 4326,
0.89 ×) was hypothesised to be the f16-accumulator coopmat variant
llama.cpp ships and we don't (`ACC_TYPE = float16_t`, half the
accumulator VGPR footprint). Sprint 13B already confirmed the lever is
shader-side, not driver-side (Mesa 26.0.6 vs 26.1-rc3 was neutral).

**Verdict.** Pre-check passed (RDNA4 advertises FP16×FP16→FP16→FP16 at
16×16×16; llama.cpp's `coopmat_support_16x16x16_f16acc` is true on this
GPU; their dequant Q*_K GEMM at default precision routes through the
f16acc path). Built the f16acc SPVs (Q4_K + Q6_K, aligned). Wired an
opt-in env var. **Result: pp=512 with f16acc = 3766.5 tok/s, vs 3846.1
tok/s with f32acc — a 2 % regression, well below the 4000 tok/s gate.**
15 / 15 coherent (precision is fine), 27 / 27 lib tests, no crash.
**Bench-gate not met.** f16acc is **shipped as opt-in** for future
re-evaluation but **does not default-on**. The structural cause is the
RDNA4 WMMA hardware itself — `v_wmma_f32_16x16x16_fp16` is the only
matrix instruction; f16-accumulator coopmat is API-supported but
emulated, not native, so it doesn't reduce VGPRs or improve throughput
on this GPU.

This is the same shape as Sprints 12D / 12E / 12H / 13B — a falsified
performance hypothesis caught with small-effort empirical work, kept as
opt-in code for future hardware where it might actually help.

## 1. Pre-check (passed) — does llama.cpp use f16acc on RDNA4?

The brief explicitly required this gate before any shader work, per
Fallstrick #1 + #4. Walking the chain:

### 1.1 RDNA4 cooperative-matrix capability table

`probe_coopmat` (`examples/probe_coopmat.rs`) on this GPU:

```
VK_KHR_cooperative_matrix properties: 20 entry(ies)
#      M   N   K  AType  BType  CType  ResultType  scope     saturating
…
16    16  16  16  FP16   FP16   FP16   FP16        Subgroup  no
18    16  16  16  FP16   FP16   FP32   FP32        Subgroup  no
```

Entry **16** is exactly what llama.cpp (`ggml-vulkan.cpp:5453-5462`)
checks for to set `coopmat_support_16x16x16_f16acc = true`. So the
runtime feature flag is asserted on this hardware.

### 1.2 llama.cpp's runtime selector

`ggml-vulkan.cpp:6211`:

```cpp
return (ctx->device->fp16
        && ctx->device->coopmat_acc_f16_support
        && prec == GGML_PREC_DEFAULT)
       ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc
       : …                                            .f32acc;
```

For dequant Q4_K / Q6_K GEMMs at default precision (the regular prefill
path), llama.cpp picks the f16acc variant when the device supports it.
On this GPU it does. So the working hypothesis "we lose ~10–15 % at
pp=512 because llama.cpp uses f16acc and we don't" was a
load-bearing claim about real codepaths, not a paper-only difference.

### 1.3 llama.cpp's build recipe

`vulkan-shaders/vulkan-shaders-gen.cpp:451-454`:

```cpp
base_dict["ACC_TYPE"  ] = f16acc ? "float16_t" : "float";
base_dict["ACC_TYPEV2"] = f16acc ? "f16vec2"   : "vec2";
if (f16acc) {
    base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
}
```

Three define deltas vs the f32acc variant. `FLOAT_TYPE` /
`FLOAT_TYPEV{2,4,8}` already differ from the non-coopmat path (always
FP16 in our coopmat SPVs since 12L); `D_TYPE` stays `float` so writeback
is FP32 even with f16 accumulator. **Pre-check ✓.**

## 2. Defines — exact diff vs Sprint 12L's aligned coopmat

| Define          | f32acc (12L)               | f16acc (13C, new)           |
|-----------------|----------------------------|-----------------------------|
| `DATA_A_Q*_K`   | `1`                        | `1` (same)                  |
| `A_TYPE`        | `block_q*_K`               | same                        |
| `A_TYPE_PACKED*`| `block_q*_K_packed{32,16}` | same                        |
| `B_TYPE`        | `mat2x4`                   | same                        |
| `D_TYPE`        | `float`                    | **same** (writeback stays FP32) |
| `FLOAT_TYPE`    | `float16_t`                | same (already FP16)         |
| `FLOAT_TYPEV{2,4,8}` | `f16vec2 / f16vec4 / f16mat2x4` | same         |
| `ACC_TYPE`      | `float`                    | **`float16_t`** ←           |
| `ACC_TYPEV2`    | `vec2`                     | **`f16vec2`** ←             |
| `ACC_TYPE_MAX`  | (unset)                    | **`float16_t(65504.0)`** ←  |
| `LOAD_VEC_A`    | `4` / `2`                  | same                        |
| `LOAD_VEC_B`    | `8`                        | same                        |
| `ALIGNED`       | `1`                        | same                        |
| `COOPMAT`       | `1`                        | same                        |

In `mul_mm.comp`:

- Line 134: `shared ACC_TYPE coopmat_stage[…];` — staging buffer is now
  half the LDS footprint per warp (16 B vs 32 B per fragment cell).
- Line 255: `coopmat<ACC_TYPE,…,gl_MatrixUseAccumulator> sums[…];` —
  the cooperative-matrix accumulator fragment becomes `coopmat<float16_t,…>`.
- Lines 333–339 (scalar fallback path, unused on coopmat builds): `fma`
  on `ACC_TYPE` would similarly be FP16-typed — not exercised here.
- Lines 352–362: `clamp(sums[j][i], -ACC_TYPE_MAX, ACC_TYPE_MAX)` —
  `ACC_TYPE_MAX` is `float16_t(65504.0)`, which protects against FP16
  saturation just before the `D_TYPE(...)` writeback to `data_d`.

Two new SPVs:

```
mul_mm_q4_k_f32_aligned_coopmat_f16acc.spv  195 588 B
mul_mm_q6_k_f32_aligned_coopmat_f16acc.spv  194 600 B
```

Total SPV count went 68 → 70.

## 3. Code wiring

| File | Change |
|---|---|
| `build.rs` | +44 LOC: 2 new `ShaderJob`s mirroring 12L's aligned coopmat with the three f16acc deltas |
| `src/backend/vulkan/shaders.rs` | +18 LOC: 2 new `ShaderId` variants + `name()` arms + `spv_bytes()` arms + 2 byte-include `pub const`s + `ALL_SHADERS` entries |
| `src/backend/vulkan/pipeline_registry.rs` | +2 LOC: extended the existing 12-way coopmat match arm to 14 ShaderIds. f16acc shaders fall through to the `else { L-tile }` branch (we ship f16acc only at L-tile in this sprint, matching the brief's "MINIMAL: nur L-Tile + aligned" recommendation) |
| `src/backend/vulkan/forward.rs` | +18 LOC: `mul_mm_coopmat_f16acc_enabled` field + env-var read; `layer_weight_shader_gemm` gets a 9th param `coopmat_f16acc`; two `match` arms each get a guard `if coopmat_f16acc` that redirects the aligned-L-tile case to the `*F16Acc` ShaderId; `run_gemm`'s tile-size lookup adds the new ShaderIds to the `(128, 128)` arm; six call sites updated |

Routing: f16acc only redirects the **aligned + L-tile** dispatch
(coopmat_q4k_mm = true, aligned = true, m_tile = false, s_tile = false).
Unaligned shapes, M-tile, and S-tile keep their FP32-accumulator
path even with the env var on. This matches the SPV inventory and
keeps the perturbation surface narrow.

Default OFF: opt-in via `VULKANFORGE_COOPMAT_F16ACC=1`.

## 4. Correctness

- `cargo test --release --lib` (default) → **27 / 27 passing**.
- `VULKANFORGE_COOPMAT_F16ACC=1 cargo test --release --lib` → **27 / 27**.
- `VULKANFORGE_COOPMAT_F16ACC=1 run_15prompt_bench` → **15 / 15
  coherent**, decode median **91.7 tok/s**, prefill median 850.4 tok/s
  (within run-to-run noise of v0.2.2 / 13A).

The K=12288 reduction in `gemm_down` was the primary precision-risk
worry — FP16 accumulation over 768 `coopMatMulAdd` iterations could in
principle accumulate enough relative error to break Q4_K_M coherence.
**It does not on this model**, with the `ACC_TYPE_MAX` clamp present.
That matches llama.cpp's experience: f16acc has been their default for
RDNA-class hardware where supported, and Qwen3-8B-Q4_K_M coherence is
preserved.

## 5. Performance — RUNS=5, median ms

```
$ cargo run --release --example run_pp_bench           # f32acc (default)
$ VULKANFORGE_COOPMAT_F16ACC=1 cargo run --release …   # f16acc
```

| pp   | f32acc (tok/s) | f16acc (tok/s) | Δ tok/s | Δ %    |
|------|---------------:|---------------:|--------:|-------:|
|   64 |        1691.5  |        1686.9  |    −4.6 | −0.3 % |
|  128 |        2562.8  |        2527.0  |   −35.8 | −1.4 % |
|  256 |        3552.2  |        3492.1  |   −60.1 | −1.7 % |
|  512 |    **3846.1**  |    **3766.5**  |   −79.6 | **−2.1 %** |
| 1024 |        3734.5  |        3660.1  |   −74.4 | −2.0 % |
| 2048 |        3165.6  |        3112.4  |   −53.2 | −1.7 % |

**f16acc is consistently a hair slower across the entire pp range, not
faster.** All deltas sit in the −0.3 % to −2.1 % band, just outside the
usual ±2 % run-to-run noise — directionally a small regression, not
a meaningful win.

## 6. Bench-gate verdict

| Sub-gate               | Target           | Result      | Verdict |
|------------------------|------------------|-------------|---------|
| pp=512 f16acc ≥ 4000   | 4000 tok/s       | **3766.5**  | **NO**  |
| 15/15 coherent         | 15/15            | 15/15       | YES     |
| 27/27 lib tests        | 27/27            | 27/27       | YES     |
| Decode ≥ 90 tok/s      | 90               | 91.7        | YES     |

Bench-gate not met. f16acc does not default-on and does not move the
gap to llama.cpp — they are getting their f16acc gain from somewhere
else, or it was never a meaningful gain on RDNA4 to begin with.

## 7. Why didn't f16acc help on RDNA4?

The brief's Fallstrick #4 spelled the answer out in advance:

> RDNA4 HW: `v_wmma_f32_16x16x16_fp16` → Accumulator IST f32!
> Es gibt KEIN `v_wmma_f16_16x16x16_fp16` auf RDNA4!

What the cooperative-matrix capability table reports (entry 16:
FP16×FP16→FP16→FP16) is **API-level support**, not a different
underlying instruction. RDNA4 has exactly one wave-matrix instruction
family, `v_wmma_f32_16x16x16_fp16` and its bf16/i8 siblings; the
result fragment is always 32-bit. When a Vulkan compute shader asks
for `coopmat<float16_t, …, gl_MatrixUseAccumulator>`, ACO emits the
same f32 WMMA + an FP16 store / FP32-to-FP16 conversion. So:

- **Accumulator VGPRs**: not actually halved — the fragment lives in
  the same number of VGPRs at instruction time; it's the f16
  *destination buffer* that is half-sized, not the wave register file.
- **WMMA throughput**: identical to f32acc — same hardware, same
  cycle count.
- **Conversion overhead**: FP32→FP16 narrowing on every accumulator
  store, FP16→FP32 widening on every accumulator load (yes, those
  exist inside the K-loop in some lowerings) — **net cost**, not a
  saving.

The 2 % slowdown we see is consistent with that conversion cost
showing up across 36 layers × 7 GEMMs × hundreds of K-loop iterations.
Coherence is fine because `D_TYPE = float` and `ACC_TYPE_MAX` clamping
keep the writeback numerically clean.

This is **vendor-specific behaviour**. The same shader on:

- **NVIDIA** (Ampere+, Hopper): `wmma::fragment<…, half>` is a real
  hardware fragment with half VGPR footprint and higher TFLOPs/W —
  f16acc is a clear win there.
- **Intel Battlemage / Lunar Lake** (Xe-cores with XMX): similar
  story, dedicated f16 accumulator paths.
- **RDNA3** (gfx11): same WMMA instruction family as RDNA4 — same
  result expected, no f16acc gain.

llama.cpp's f16acc default-on for "supported" devices is therefore
**slightly counterproductive on RDNA4**. The benefit they cite in
their own benchmarks is on NVIDIA / Intel hardware, generalised to a
device-cap check that doesn't filter out emulated cases.

Sprint 13B already located the gap: not in the driver. Sprint 13C
locates it: not in this shader-source variant either. **The remaining
~10 % to llama.cpp at pp=512 must come from elsewhere** — most
plausibly:

1. **Pipeline parallelism / multi-submit.** Sprint 12B's audit found
   we do single-submit blocking; llama.cpp uses several command
   buffers in flight. Could be 5–10 % at pp=512.
2. **Dispatch packing.** Their `quantize_q8_1` is fused into the GEMM;
   we dispatch it separately. Few %, possibly observable at pp=512.
3. **Buffer aliasing / cache warm-up.** They reuse fewer live buffers
   per layer; our 20+ live SSBOs may cost L2 thrashing. Unmeasured.
4. **Per-dispatch CPU overhead at the very-large-pp end.** We measured
   ~0.1 % at decode, but prefill_batch's 600 dispatches/submit could
   accumulate a small CPU-side cost.

None of these are addressable by editing `mul_mm.comp` — they're
graph-shape / scheduling work. v0.3 territory.

## 8. What we keep, what we throw away

**Keep (committed):**

- 2 new SPVs (Q4_K + Q6_K aligned f16acc) — cheap, correct, opt-in.
- `MulMm{Q4K,Q6K}AlignedCoopmatF16Acc` ShaderIds + plumbing.
- `VULKANFORGE_COOPMAT_F16ACC=1` opt-in env var.
- This report.

**Don't ship default-on:** the perf delta is −2 %; coherence is fine
but there is no positive case for the user.

**Don't expand variant coverage** (no S-tile / M-tile / unaligned
f16acc SPVs): would just multiply build time without changing the
RDNA4-specific result. The path is wired so a future sprint targeting
NVIDIA / Intel hardware can grow it cheaply.

## 9. Outputs

- `build.rs` +44 LOC.
- `src/backend/vulkan/shaders.rs` +18 LOC.
- `src/backend/vulkan/pipeline_registry.rs` +2 LOC.
- `src/backend/vulkan/forward.rs` +18 LOC, 1 new struct field, 6 call
  sites updated, 1 new env var.
- 2 new SPVs (total: 70).
- This report.
- 27 / 27 lib tests, 15 / 15 coherence with `VULKANFORGE_COOPMAT_F16ACC=1`.

## 10. Recommendation for v0.2.3

- Ship the f16acc opt-in path as part of v0.2.3. **Default OFF.**
- Document `VULKANFORGE_COOPMAT_F16ACC=1` in README's env-var table
  with the RDNA4-neutral caveat (so users on other GPUs may benefit).
- Pivot the next prefill sprint away from shader-source variants
  toward graph-shape / multi-submit work. Sprint 13D (Wave32 / VOPD
  on Mesa 26.1) is independent and can run in parallel.
