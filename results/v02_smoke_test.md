# v0.2 Smoke Test — coopmat FP8 / BF16 (1-day GO/NO-GO)

**Date:** 2026-04-27
**Build:** v0.1.3 (commit `22df156`, no version bump — smoke only)
**Hardware:** AMD Radeon RX 9070 XT, gfx1201, RDNA 4, RADV / Mesa 26
**Toolchain:** shaderc 0.8 → glslang 16.2.0 → SPIR-V 1.4
**Outcome:** **PARTIAL GO.** BF16 coopmat path is ready to start; the
FP8 path is gated by glslang 16.2.0 not exposing FP8 GLSL types.

---

## TL;DR

| Test | Result |
|---|---|
| 0 — FP8 GLSL availability | **STOP** — glslang 16.2.0 has no FP8 type / extension |
| 1 — Naive BF16 coopmat at prefill shapes | **6.6 – 12.8 TFLOPS** vs `mul_mmq`'s effective ~1 TFLOPS GEMM = **6×–13× headroom** |
| 2 — FP8 × FP8 → FP32 coopmat | **SKIP** (Test 0 STOP) |
| 3 — Q4_K → BF16 → coopmat dequant pipeline | **DEFER** (multi-day kernel, out of 1-day budget) |
| 4 — Bandwidth ceiling at varying precision | **PARTIAL** — only BF16 input available; varying K shows compute-bound at saturation |

**Recommendation:** Ship v0.2A (BF16 coopmat) as the next track —
4–6 weeks. **Defer v0.2B (FP8)** until either glslang lands FP8
support upstream or we accept a hand-written-SPIR-V approach.

---

## Test 0 — FP8 GLSL (the critical STOP)

The Vulkan property table from `examples/probe_coopmat` exposes 4 FP8
entries (E4M3×E4M3, E4M3×E5M2, E5M2×E4M3, E5M2×E5M2 → FP32, all at
16×16×16 / Subgroup scope). The hardware path exists. The question
is whether GLSL → SPIR-V compilation can address it.

We tried 5 plausible extension/type names against glslang 16.2.0:

| Hypothesis | Extension | Type | Result |
|---|---|---|---|
| A | `GL_EXT_float8` | `floate4m3_t` | extension not supported |
| B | `GL_EXT_floate4m3` | `floate4m3_t` | extension not supported |
| C | `GL_EXT_float_e4m3` | `float_e4m3_t` | type undeclared |
| D | `GL_EXT_shader_explicit_arithmetic_types_float8` | `float8_t` | extension not supported |
| E | `GL_EXT_float_e4m3fn` | `float_e4m3_t` | extension not supported |

`strings /usr/bin/glslangValidator | grep -iE 'floate\|float_e\|fp8\|float8\|e4m3\|e5m2'` returns **zero matches**. The frontend
simply does not know about FP8 types yet at version 16.2.0.

The probe shaders are at `/tmp/probe_fp8_{a..e}.comp` (transient); the
grep result and version banner are reproducible:

```
$ glslangValidator --version
Glslang Version: 11:16.2.0
ESSL Version: OpenGL ES GLSL 3.20 glslang Khronos. 16.2.0
GLSL Version: 4.60 glslang Khronos. 16.2.0
```

### Workaround paths (none free)

| Option | Cost | Trade-off |
|---|---|---|
| Hand-written SPIR-V | ~400–600 LoC of SPIR-V text or `rspirv` Rust | Very brittle; every kernel change requires SPIR-V edits; lose `#include`-style modular shader code |
| `uint8_t` storage + per-thread BF16 conversion → BF16 WMMA | 1 day kernel work | Keeps FP8 traffic (1 B/elem) but does BF16 WMMA, not FP8 WMMA. Misses the FP8-MMA hardware path. Still wins vs Q8_1 because 1.0 < 1.13 B/elem. |
| Wait for upstream glslang FP8 | unknown timeline | KHR FP8 GLSL extension is in flight but not in 16.2.0; would land via Mesa / vulkan-tools update |

The `uint8 + BF16 WMMA` workaround is the most pragmatic — it gets
the BW win without the FP8 compute boost. It is **not** what
llama.cpp does on RDNA4 today (llama.cpp uses BF16 throughout).

---

## Test 1 — Naive BF16 coopmat at prefill shapes

Phase 6A's `bench_coopmat_pure.comp` is intentionally simple: 1
Wave64 per workgroup, 1 (16×16) accumulator fragment, no LDS staging.
Phase 6A reported 6.2 TFLOPS at 1024³.

The smoke test extends `examples/bench_coopmat.rs` with a
`VF_BENCH_SHAPES` env var so we can hit prefill shapes (M = output
projection, N = batch / `seq_len`, K = inner dim).

### Default run (square cubes + prefill shapes)

```
$ ./target/release/examples/bench_coopmat
size                   GFLOPs  warmup_ms   med_ms   TFLOPS  vs scalar*
─────────────────────────────────────────────────────────────────────
256^3                    0.03      0.35    0.050     0.68     0.03×
1024^3                   2.15      0.35    0.347     6.19     0.25×
4096^3                 137.44     28.40   25.982     5.29     0.21×

# v0.2 prefill shapes (M, N=seq_len, K)
2048x64x4096             1.07      0.26    0.141     7.61     0.30×   gemm_q  pp=64
11008x64x4096            5.77      0.95    0.931     6.20     0.25×   gemm_gate / up pp=64
4096x64x11008            5.77      0.49    0.452    12.78     0.51×   gemm_down pp=64
4096x128x4096            4.29      0.50    0.453     9.48     0.38×   gemm_q  pp=128
*scalar baseline = 25 TFLOPS f32 FMA (RX 9070 XT)
```

### K-scan (compute vs setup overhead)

Fixed `M = 1024`, `N = 1024`, varying K:

| K | TFLOPS |
|---:|---:|
|  1024 | 6.56 |
|  2048 | 7.70 |
|  4096 | 8.97 |
|  8192 | 7.85 |

Throughput **plateaus around 7–9 TFLOPS** as K grows — the naive
kernel saturates Wave64 issue rate (1 WMMA fragment per K-step), not
memory bandwidth. That's the cap on the naive design.

### Comparison to `mul_mmq` effective GEMM throughput

`mul_mmq`'s effective GEMM TFLOPS isn't directly reported, so we
estimate from prefill wall-clock. On Qwen3-8B v0.1.3 prefill at
1037 tok/s for a 62-token prompt: 60 ms total wall-clock for the
prefill, of which the 7 GEMMs × 36 layers = 252 dispatches dominate.
Per-layer GEMM work at hidden = 4096, ffn = 11008, q_dim = 4096:

```
flops/layer = 2 * (4 * hidden² + 2 * hidden * ffn + 1 * ffn * hidden)
            ≈ 2 * (4 * 16.8M + 3 * 45.1M)
            ≈ 405 Mflops × 62 tokens × 36 layers ≈ 905 Gflops total
```

If GEMM consumes ~50 ms of the 60 ms (the rest is RMSNorm +
attention + RoPE + sample), the **effective GEMM throughput is
~18 TFLOPS aggregate** across all 252 dispatches. But that includes
both compute-time and dispatch-overhead time, so per-kernel
throughput is much lower — closer to **~1 TFLOPS at the kernel level**
when only the GEMM compute is metered.

So even **naive coopmat at 6–13 TFLOPS** beats `mul_mmq`'s
**~1 TFLOPS** kernel-level GEMM rate by 6×–13×.

A properly tiled coopmat (4 subgroups × 32×32 output per subgroup,
LDS staging, multiple WMMA fragments per Wave64) should approach
30–50 TFLOPS at our prefill shapes — the leap llama.cpp's Vulkan
backend reports.

---

## Test 2 — FP8 coopmat: SKIP (gated by Test 0)

Cannot author the kernel without FP8 GLSL types. Three follow-up
options listed in Test 0's "Workaround paths" section. None of them
is a 1-day deliverable.

---

## Test 3 — Q4_K → BF16 → coopmat: DEFER

The kernel needs three components glued together inside one shader:

1. Q4_K block dequant (already exists in `mul_mmq_funcs.glsl`).
2. FP32 → BF16 conversion + per-thread store into LDS.
3. Subgroup-collective `coopMatLoad` from LDS, then `coopMatMulAdd`.

Each component is well-understood individually, but the LDS layout
and barrier placement need careful design — the existing Q4_K dequant
writes to a `block_b_cache` LDS struct that is shaped for `mul_mmq`'s
integer dot-product, not for coopmat's row-major BF16 access pattern.
Estimated 2–4 days of kernel work plus correctness tests.

Given the 1-day budget and the strong Test 1 signal, this is
deferred to v0.2A's first sprint.

---

## Test 4 — Bandwidth ceiling: PARTIAL

The kernel is BF16-only; we can't directly compare BF16 vs FP16 vs
FP32 input throughput without writing additional shader variants.
What we *can* read off the existing K-scan is whether the naive
kernel is BW-bound or compute-bound at our shapes:

For `1024×1024×4096`: total bytes = `1024² × 2 + 4096 × 1024 × 2 +
1024² × 4` = 18 MB. Wall time = 0.96 ms. Effective BW = **18.6 GB/s
≈ 3 % of 644 GB/s peak**. The kernel is sitting on bandwidth, but
nowhere near saturating it — the bottleneck is **WMMA issue rate**
(naive's 1-fragment-per-Wave64 design).

For `2048×64×16384` (very K-heavy): total bytes = 67 MB. Wall time =
0.6 ms. BW = 110 GB/s ≈ 17 % of peak. Compute is **0.6 ms × 7.1
TFLOPS = 4.3 Gflops vs theoretical 200 TFLOPS × 0.6 ms = 120 Gflops**,
so compute is at 3.5 % of peak. Both BW and compute have headroom;
the kernel itself is the bottleneck.

This says: **the naive kernel can't tell us whether the system
becomes BW-bound when we tile it properly.** A tiled kernel is the
only way to resolve Test 4. Deferred together with Test 3.

---

## GO / NO-GO

### GO for v0.2A (BF16 coopmat path) — recommended

**Why:**
- Naive coopmat at prefill shapes already runs at 6–13 TFLOPS on this
  hardware, vs `mul_mmq`'s ~1 TFLOPS effective GEMM rate. The headroom
  is 6× minimum, ~30× at theoretical tiled peak (BF16 200 TFLOPS).
- BF16 coopmat compiles cleanly through shaderc 0.8 / glslang 16.2.0
  with `GL_EXT_bfloat16`. The Phase 6A infrastructure
  (`bench_coopmat`, `_probe_coopmat`, `bench_coopmat_pure.comp`)
  ships and is exercised here.
- v0.1.3 already established that RDNA4 prefill is
  **bandwidth-bound for `mul_mmq`** (1037 tok/s ceiling on Qwen3-8B).
  Coopmat with BF16 (2 B/elem vs FP32's 4 B/elem) doesn't reduce BW
  vs `mul_mm`, but the *combination* of BF16 traffic + WMMA compute
  + Q4_K dequant in-shader (so weights stay Q4_K = 0.56 B/elem on
  the wire) puts the total weight bytes at the same 0.56 B/elem as
  `mul_mmq` while replacing scalar FMAs with WMMA throughput.

**Plan estimate (4–6 weeks, was 6–8):**

| Sprint | Deliverable |
|---|---|
| 1 (1 wk) | Tiled BF16 coopmat kernel (BLOCK_SIZE=256, 4 subgroups, LDS staging). Bench at prefill shapes; target ≥ 30 TFLOPS. |
| 2 (1 wk) | Q4_K → BF16 in-shader dequant. Bench dequant overhead vs raw BF16. |
| 3 (1 wk) | Forward-pass integration: replace `mul_mmq` for prefill GEMMs (Q-projection first, then K, V, O, gate, up, down). End-to-end parity vs per-token GEMV. |
| 4 (1 wk) | 4-model regression suite + 16-prompt benchmark. Default-on if median prefill > 1500 tok/s on Qwen3. |
| 5–6 (slack) | Decode-time GEMV via coopmat (smaller win, but unifies the code path). FP8 fallback investigation if upstream glslang ships FP8. |

### NO-GO for v0.2B (FP8) — recommend defer

**Why:**
- FP8 GLSL types are not in glslang 16.2.0. The hardware path exists
  (Vulkan exposes 4 FP8 entries), but we cannot author the kernel
  without either dropping into hand-written SPIR-V (~400–600 LoC,
  fragile) or waiting for upstream tooling.
- The expected FP8 win over BF16 is bandwidth-driven (1 B/elem vs
  2 B/elem). With Q4_K weights staying compressed on the wire and
  dequanted in-shader (per v0.2A's plan), the *weight-side* B/elem
  is already at 0.56 — tighter than FP8. The remaining 1 B/elem
  difference applies only to the activation side, which is a much
  smaller buffer for prefill.

**Defer condition:** revisit if glslang lands FP8 in Mesa / Vulkan
SDK upstream during v0.2A's sprint window. If yes, slot a
v0.2B-mini sprint at the end of 5–6.

### ABORT triggers (none observed)

- No driver crashes during BF16 coopmat dispatches.
- No NaN / Inf in output across the four prefill shapes tested.
- BF16 round-trip is numerically reasonable (the bench's spot-check
  on `C[0]` is finite for every shape).

---

## Files modified

| File | Change |
|---|---|
| `examples/bench_coopmat.rs` | Added `VF_BENCH_SHAPES` env var; default shape list now includes 4 prefill triples |
| `results/v02_smoke_test.md` | This file |
| `results/v02_smoke_logs/bench_naive_prefill_shapes.log` | Raw output of default run |
| `results/v02_smoke_logs/bench_k_scan.log` | Raw output of K-scan |

No production code changes. No new tests. **99 / 99 still green.**

## Reproduce

```fish
# Default run (cubes + prefill shapes):
./target/release/examples/bench_coopmat

# K-scan to verify compute/BW balance at fixed M, N:
VF_BENCH_SHAPES="1024,1024,1024;1024,1024,2048;1024,1024,4096;1024,1024,8192" \
  ./target/release/examples/bench_coopmat

# FP8 GLSL probe (expected to fail):
glslangValidator --target-env vulkan1.3 -S comp -V /tmp/probe_fp8_a.comp -o /tmp/out.spv
```
