# Phase 6B — STOP report (port plan non-viable on RADV gfx1201)

**Date:** 2026-04-27
**Version:** v0.1.1 (no version bump — no code shipped this phase)
**Status:** STOP per prompt rule 2 ("BEI UNKLARHEITEN SOFORT STOP")
**Action taken:** none beyond analysis (no shader written, no code changed)

---

## TL;DR

The Phase 6A GO recommendation §4.1 — *"port `mul_mm_cm2.comp` from
llama.cpp, ~1–2 weeks, MIT-licensed, proven kernel"* — **does not work
on RADV gfx1201**. `mul_mm_cm2.comp` is built end-to-end on
`GL_NV_cooperative_matrix2` (workgroup-scope coopmat, tensor-layout
API, decode-functions). RADV exposes `VK_KHR_cooperative_matrix`
only — the NV extension is unavailable. The shader will fail to
compile on the device's driver.

**Phase 6A Phase 6A's recommendation was a mis-read of the source.** I saw
`#extension GL_KHR_cooperative_matrix : enable` at the top of
`mul_mm_cm2.comp` and assumed it was the KHR path with NV as an
extension on top. It is the opposite: KHR is the base spec the NV
cm2 builds on, and the KHR-only fallback is **`mul_mm.comp` (no
coopmat at all)**.

This phase reports the finding, leaves the production code
untouched, and lays out the two viable next steps.

---

## 1 — Evidence

### 1.1 mul_mm_cm2.comp depends on NV cm2 *unconditionally*

```
$ grep -nE "ScopeWorkgroup|tensorLayoutNV|coopMat.*NV" mul_mm_cm2.comp | head -8
278:    tensorLayoutNV<2> tensorLayoutA = createTensorLayoutNV(2);
279:    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutAClamp ...
334:    coopmat<ACC_TYPE, gl_ScopeWorkgroup, BM, BNover4, gl_MatrixUseAccumulator> sum ...
344:    coopmat<MAT_TYPE, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> mat_a;
345:    coopmat<MAT_TYPE, gl_ScopeWorkgroup, BK, BNover4, gl_MatrixUseB> mat_b;
500:    coopMatLoadTensorNV(mat_b, data_b, pos_b, sliceTensorLayoutNV(...) ...);
```

Every `coopmat<…>` declaration uses `gl_ScopeWorkgroup`. KHR
cooperative_matrix supports **subgroup scope only** — `gl_ScopeWorkgroup`
is an NV cm2 extension. There is no `#ifdef` to fall back to subgroup
scope.

`coopMatLoadTensorNV(...)` is the NV cm2 tensor-load primitive that
takes a `tensorLayoutNV<…>` plus a `decodeFunc*` callback for inline
quantised-data decoding. KHR has `coopMatLoad(matrix, buffer, offset,
stride, layout)` which takes nothing of the sort.

`#extension GL_KHR_cooperative_matrix : enable` (line 12) is the
*base* extension every NV cm2 shader must enable — KHR provides the
`coopmat<…>` type system; NV cm2 layers on top of it. Listing it
does not imply a KHR-only path.

### 1.2 RADV gfx1201 advertises only KHR

```
$ vulkaninfo 2>/dev/null | grep -E "VK_NV_cooperative|VK_KHR_cooperative"
        VK_KHR_cooperative_matrix                     : extension revision 2
                                                        ↑ only this
```

No `VK_NV_cooperative_matrix2`. Compiling `mul_mm_cm2.comp` against
RADV with `glslangValidator -V --target-env vulkan1.3` would either
fail at SPV-validation (NV ops with no extension declared) or produce
an SPV the driver can't load.

### 1.3 llama.cpp's KHR-only fallback is mul_mm.comp (no coopmat)

```
$ ls ggml/src/ggml-vulkan/vulkan-shaders/mul_mm*
mul_mm_cm2.comp     ← NV cm2 (verified above)
mul_mm.comp         ← shared-memory tiled GEMM, NO coopmat (464 LOC)
```

`mul_mm.comp` is what llama.cpp falls back to on drivers without NV
cm2. It uses manual subgroup loads + LDS tiling, no WMMA. Porting
this gives us no AI-accelerator usage — it's the path the existing
`mul_mmq.comp` already covers (and our Phase 5B.3 batched-prefill
prep already runs through).

There is **no** `mul_mm_cm1.comp` (a hypothetical KHR-only coopmat
GEMM). llama.cpp does not ship one.

### 1.4 flash_attn_cm1.comp is KHR coopmat, but it's an attention kernel

```
flash_attn_cm1.comp:
  #extension GL_KHR_cooperative_matrix : enable
  coopmat<float16_t, gl_ScopeSubgroup, MatBc, 16, gl_MatrixUseA> KMat;
  coopmat<float16_t, gl_ScopeSubgroup, 16,    MatBr, gl_MatrixUseB> QMat;
  coopMatLoad(KMat, kvsh, ...);
```

This is a real KHR-only coopmat shader: subgroup-scope, tile sizes
fixed at 16×16, multiple subgroups per workgroup with per-subgroup
matrices. It's a useful **pattern source** for a from-scratch KHR
GEMM, but it is *not* a GEMM and it's tightly coupled to flash-
attention's softmax loop. There's no port-and-adapt route.

---

## 2 — What this changes vs Phase 6A's GO recommendation

| Phase 6A item | Status after Phase 6B analysis |
|---|---|
| `BF16 × BF16 → FP32` advertised on device | ✅ still true — entry 19 is real |
| `shaderc 0.8` compiles coopmat + bfloat16 | ✅ still true |
| Naive WMMA bench: 6 TFLOPS at 1024³ | ✅ still true (still useful as a baseline) |
| **Port `mul_mm_cm2` (~1-2 weeks, MIT, proven)** | ❌ **NOT VIABLE** — depends on NV cm2 |
| llama.cpp's KHR fallback as a port target | ❌ that fallback is `mul_mm.comp`, no coopmat |
| KHR coopmat GEMM exists upstream anywhere | ❌ none found in llama.cpp |

The hardware gates remain green. The kernel-availability gate
collapsed.

---

## 3 — Revised options

### 3.1 Option A — write a KHR-only coopmat GEMM from scratch

What it looks like:
- `gl_ScopeSubgroup` only — every `coopmat<…>` lives inside a single
  Wave64.
- Multiple subgroups per workgroup carry multiple 16×16 fragments;
  shared-memory tiling moves data between global and LDS.
- Pattern source: llama.cpp's `flash_attn_cm1.comp` (KHR coopmat
  patterns) + `mul_mm.comp` (LDS-tiled GEMM patterns).
- Q4_K dequant happens in the LDS-staging pass — the same
  `dequant_q4k` we already have in `mul_mmq.comp`, repointed to
  write BF16 into shared memory instead of unpacking into Q8_1.

Effort estimate (revised, more realistic than Phase 6A's §4.2):
- Skeleton + KHR coopmat GEMM (FP16×FP16→FP32, dense): **~1 week**
- LDS tiling + bank-conflict-free layout: **~3-5 days**
- Q4_K-dequant-into-LDS-as-BF16: **~3-5 days**
- Parity tests + tuning to beat `mul_mmq` at M=64: **~1 week**
- **Total: ~3-4 weeks** with moderate risk (correctness + perf
  tuning are both non-trivial).

Expected ceiling: **~25-40 TFLOPS** effective on the realistic
prefill workload (informed by what a similarly-shaped KHR-only
kernel would achieve when the LDS-tile shape and WMMA-per-WG
count are tuned for the 64-CU / 128-AI-Accel topology). That's
in the same envelope as Phase 6A's §4.1 estimate (the hardware
ceiling didn't change), just with a longer engineering path.

### 3.2 Option B — NO-GO on coopmat, ride the fallback work-list

Phase 6A's §4.3 in summary:
- pipeline-cache pre-warming (~+5-10 % prefill, 1 day)
- GEMM tile-tuning via spec-constants (~+10-20 %, 2-3 days)
- FP16 KV cache (~+2-3 % decode, ~50 % VRAM headroom, 2-3 days)
- async dual-buffer prefill (~+15-25 %, 3-5 days)
- **Combined**: ~1500–1800 tok/s prefill (vs 1082 today, vs
  Option A's projected 1700–2400). Less peak, no kernel-correctness
  risk, ~2 weeks of clearly bounded work.

### 3.3 Option C — pivot the same kernel work onto FP16

`FP16 × FP16 → FP32` is also entry 18 of the property table. FP16
has **3-VALU cast cost from FP32**, vs BF16's 5 VALU. The "Q4_K
weight → FP16 vs BF16" question favours FP16 slightly on the cast
side; precision-wise FP16 is worse for accumulation (5-bit exponent
range) but the FP32 accumulator already handles that.

This is an Option-A variant — same engineering effort, swap
`bfloat16_t` for `float16_t`. Decision can be deferred to inside
Option A and explored cheaply.

---

## 4 — Recommendation

**My read: Option B (NO-GO and fallback list) is the higher-ROI
move right now.** Reasons:

- Option A doubles Phase 6A's effort estimate (3-4 weeks vs 1-2)
  while landing in roughly the same TFLOPS ceiling.
- The fallback list (Option B) is mostly mechanical work with
  predictable wins, and removes the "kernel correctness risk that
  blocks the release" failure mode.
- **The Phase 5B series already closed most of the prefill gap
  cheaply** (405 → 1082 tok/s). The next 30–50 % comes from
  general-purpose pipeline / cache work; the next 100 %+ is the
  coopmat ceiling. Going for the ceiling requires the kernel work
  no matter which path we pick — but doing the cheap wins first is
  strictly the right ordering.
- coopmat work doesn't go away if we pick B; it just becomes
  Phase 7 with the fallback wins as the new baseline (1500–1800
  tok/s) and a clearer hardware-ceiling target.

If the user prefers to stay aggressive on the prefill curve,
Option A is fine — but the right way to scope it is "v0.2 milestone
work, 4 weeks", not "v0.1.2, 1-2 weeks".

---

## 5 — Files

```
NEW   results/phase6b_coopmat_gemm.md     — this STOP report
```

No production code changed. No new shaders. No new tests. Phase
6A's `examples/probe_coopmat.rs`, `examples/bench_coopmat.rs`,
`vk_shaders/_probe_coopmat.comp`, `vk_shaders/bench_coopmat_pure.comp`,
and `build.rs` jobs all remain — they're still useful as the
"hardware works" baseline.

---

## 6 — Tests

Unchanged from Phase 6A:

```
unit (lib)         19   ALL GREEN
correctness        33
regression         25
TOTAL              77
cargo test --release       → 77/77 in ~36 s
cargo clippy --release …   → clean
```

---

## 7 — Console summary

```
═══ Phase 6B — STOP report ═══
Finding:    mul_mm_cm2.comp depends on GL_NV_cooperative_matrix2
            (gl_ScopeWorkgroup + tensorLayoutNV + decodeFuncNV).
            RADV gfx1201 advertises only VK_KHR_cooperative_matrix.
            → No port path for mul_mm_cm2 on this hardware.

Phase 6A
recommendation:
  §4.1 "port mul_mm_cm2"   ❌ NOT VIABLE
  §4.2 "from scratch KHR"  ✅ viable (revised: 3-4 weeks)
  §4.3 "NO-GO + fallback"  ✅ viable (~2 weeks, +30-50% prefill)

Recommendation: Option B (fallback work-list) for v0.1.2;
                Option A as v0.2 / Phase 7 milestone.

No code changes. 77/77 tests still green.
Commit: report-only.
```
