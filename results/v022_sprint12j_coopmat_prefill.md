# Sprint 12J — coopmat WMMA prefill: diagnosis (bench-gate not reached)

**Premise (from Sprint 12I).** The 4 × prefill gap to llama.cpp is the
KHR_coopmat WMMA pipeline. llama.cpp ships it default-on; our coopmat
path is *slower* than the default `mul_mmq` integer-DP path. Goal:
fix the configuration / build / routing so
`VULKANFORGE_USE_MM_COOPMAT=1` reaches pp=512 ≥ 3500 tok/s, then ship
it default-on for prefill.

**Verdict.** **Bench-gate not reached.** Required changes are
structural, not configuration: our coopmat path **regresses Q6_K
GEMMs** (gemm_v + gemm_down) by routing them through the slower
scalar `mul_mm` path while only Q4_K GEMMs benefit from the WMMA
fragments. Closing this gap requires either a Q6_K coopmat SPV
(doesn't exist in our build) or a routing change that mixes
`mul_mm` (Q4_K) and `mul_mmq` (Q6_K) within a single layer — both
non-trivial. Sprint 12J commits the diagnosis only; the engineering
sprint that follows is scoped in §7.

The Sprint 11E warptile values **are correct** (md5/diff against
llama.cpp HEAD = identical for `mul_mmq.comp`, `mul_mm.comp`,
`mul_mat_vec_q4_k.comp`, `mul_mat_vec_q6_k.comp`; warptile constants
in `pipeline_registry.rs:396-408` match `ggml-vulkan.cpp:3367` for
`l_warptile_mmq` on AMD KHR_coopmat). The shader port from Sprint
11E is fine. **What's missing is everything around the shader.**

## 1. Pre-check: shader-source identity (passes)

```
$ md5sum vk_shaders/mul_mm.comp ~/tmp/llama.cpp/.../mul_mm.comp
0bdf44a455deb59b29fc8621a559058b  vk_shaders/mul_mm.comp
0bdf44a455deb59b29fc8621a559058b  …/llama.cpp/…/mul_mm.comp
```

Identical, like in Sprint 12H/12I. The 4 × gap is **not** in the
GLSL source.

## 2. Sprint 11E warptile values: still correct

`pipeline_registry.rs:396-408` for `MulMmQ4KCoopmat`:

```
[ 256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64 ]
  BS  BM   BN   BK  WM  WN  WMITER  TM  TN  TK  WARP
```

llama.cpp `ggml-vulkan.cpp:3367` (the AMD-coopmat override that
fires on RADV gfx1201, KHR_coopmat available, non-AMDPRO driver):

```cpp
l_warptile_mmq = l_warptile_mmq_int = {
    256, 128, 128, 32, subgroup_size_8, 64, 2,
    tm_m, tn_m, tk_m, subgroup_size_8
};
```

With `subgroup_size = 64` on RDNA4 → `subgroup_size_8 = max(64, 8)
= 64`, and `tm_m = tn_m = tk_m = device->coopmat_m/n/k = 16` (KHR
coopmat 16×16×16 fragments) →

```
{ 256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64 }   ← identical
```

So Sprint 11E correctly pinned the spec-constants. **They are not
the problem.**

## 3. Per-shader prefill profile — the actual bottleneck

`profile_prefill VF_PP=512`, default vs coopmat-on, sum of µs over
36 layers:

| Shader (per-dispatch / per-forward sum) | Default `mul_mmq` | `USE_MM_COOPMAT=1` | Δ |
|---|---:|---:|---:|
| **`gemm_q`** (Q4_K, M=512 N=4096 K=4096) | 17 041 | **14 629** | −2 412 ✅ |
| **`gemm_o`** (Q4_K, M=512 N=4096 K=4096) | 16 181 | **14 488** | −1 693 ✅ |
| **`gemm_gate`** (Q4_K, M=512 N=12288 K=4096) | 47 722 | **40 790** | −6 932 ✅ |
| **`gemm_up`** (Q4_K, same shape, artifact) | 91 157 | 81 326 | −9 831 ✅ |
| `gemm_k` (Q4_K, M=512 N=1024 K=4096) | 24 288 | 24 243 | −45 (≈) |
| **`gemm_v`** (**Q6_K**, M=512 N=1024 K=4096) | 26 979 | **27 774** | +795 ❌ |
| **`gemm_down`** (**Q6_K**, M=512 N=4096 K=12288) | 61 054 | **89 568** | +28 514 ❌❌❌ |
| Wall total | 232.05 ms | 246.76 ms | +14.7 ms (slower!) |
| Effective | 2206 tok/s | 2075 tok/s | −5.9 % |

**The Q4_K GEMMs win nicely with coopmat (−21 ms saved across q/o/gate/up).
The Q6_K GEMMs lose more than that (+29 ms regression on v/down).**
Net: the coopmat path is ~6 % SLOWER than the integer-DP default.

### 3.1 Why Q6_K regresses

`forward.rs:3805-3814 layer_weight_shader_gemm`:

```rust
match (gemm_kind, q6) {
    (GemmKind::MulMmAligned, true)  => ShaderId::MulMmQ6KAligned,
    (GemmKind::MulMmAligned, false) => ShaderId::MulMmQ4KAligned,
    (GemmKind::MulMm,        true)  => ShaderId::MulMmQ6K,        // ← scalar mul_mm
    (GemmKind::MulMm,        false) => if coopmat_q4k_mm { MulMmQ4KCoopmat } else { MulMmQ4K },
    (GemmKind::Mmq,          true)  => if prefer_l { MulMmqQ6KL } else { MulMmqQ6K },
    (GemmKind::Mmq,          false) => if prefer_l { MulMmqQ4KL } else { MulMmqQ4K },
}
```

When `VULKANFORGE_USE_MM_COOPMAT=1`, `gemm_kind = MulMm`. Q4_K
weights take the `MulMmQ4KCoopmat` branch (the Sprint 11E coopmat
SPV) — fast. Q6_K weights take the `MulMmQ6K` branch — that's
**plain scalar mul_mm with FP32 activations**, which is the
**slowest GEMM path we have** (Sprint 12I 4-way bench showed
`mul_mm` alone gives 1151 tok/s @ pp=512, vs default 2348).

So flipping `USE_MM_COOPMAT=1` *gains* on Q4_K GEMMs but
*regresses* on Q6_K GEMMs, and the Q6_K regression is bigger
because `gemm_down` (K=12288, the largest reduction) is on Q6_K and
takes the worst hit (61 ms → 89 ms).

### 3.2 Why we don't have a `MulMmQ6KCoopmat`

`build.rs` defines `mul_mm_q4_k_f32_coopmat.spv` with
`("DATA_A_Q4_K", "1")` etc., but no analogous Q6_K entry. There is
no `mul_mm_q6_k_f32_coopmat.spv` in the SPV output directory.

llama.cpp builds **both**: `matmul_q4_k_f32_cm1` and
`matmul_q6_k_f32_cm1` (cm1 = KHR coopmat 1, see lines 3704-3751
of `ggml-vulkan.cpp`). They're built from the same `mul_mm.comp`
source via the `DATA_A_*` macros plus `COOPMAT=1` and the right
`A_TYPE`, `A_TYPE_PACKED32`, `FLOAT_TYPE` etc.

## 4. Pipeline variant coverage — second structural gap

llama.cpp creates **6 variants per quantization type**:
`{s, m, l} × {unaligned, aligned}`. The runtime selects via
`ggml_vk_guess_matmul_pipeline` (`ggml-vulkan.cpp:7141`) based on
m/n shape:

```cpp
if (m <= 32 || n <= 32)         → s tile (BM=32,  BN=32)
else if (m <= 64 || n <= 64)    → m tile (BM=64,  BN=64)
else                             → l tile (BM=128, BN=128)
```

We ship **one variant per coopmat shader (L tile only)**. At
pp=512 with N=12288, all dispatches still hit the L tile, so this
isn't the dominant gap at pp=512. But at smaller pp:

| pp | VF default | VF coopmat | llama.cpp | VF/llama (default) |
|---:|---:|---:|---:|---:|
| 128 | 2004 | 1393 | 3631 | 0.55 × |
| 256 | 2197 | 1907 | 3975 | 0.55 × |
| 512 | 2348 | 2207 | 4324 | 0.54 × |
| 1024 | 2303 | 2155 | 4177 | 0.55 × |
| 2048 | 2082 | 1963 | 3756 | 0.55 × |

The 0.55 × ratio holds across **all** pp values measured. So even
at pp=512 (where everyone uses L tile), llama.cpp is 1.84 × ahead.
That ratio cannot come from tile-variant coverage alone (we both
use L). It comes from the WMMA matrix-cores path being engineered
end-to-end (Q6_K coopmat shader, aligned variant, FP16 activations,
…) on llama.cpp's side and not on ours.

## 5. What a "real" fix looks like

To close the prefill gap to llama.cpp parity (or near-parity), the
work is structural:

1. **Build a Q6_K coopmat SPV.** Add a `mul_mm_q6_k_f32_coopmat.spv`
   ShaderJob in `build.rs` mirroring the Q4_K one with
   `DATA_A_Q6_K=1`, `A_TYPE=block_q6_K`, `A_TYPE_PACKED32=block_q6_K_packed32`,
   plus `COOPMAT=1` and the FLOAT16 family of defines. Add
   `MulMmQ6KCoopmat` in the `ShaderId` enum and pipeline_registry.
2. **Route Q6_K to coopmat when enabled.** Patch
   `layer_weight_shader_gemm:3811`:
   `(GemmKind::MulMm, true) → if coopmat_q4k_mm { MulMmQ6KCoopmat } else { MulMmQ6K }`.
3. **Build & route an aligned Q4_K coopmat SPV** (and Q6_K) so
   pp values divisible by `LOAD_VEC_B` = 4 take the faster
   load-vec-4 path. Mirrors the `*_aligned` SPVs we already have
   for the non-coopmat mul_mm path.
4. **Build & route S- and M-tile coopmat variants.** Three
   spec-constant blocks per quant (s/m/l) and a runtime selector
   like llama.cpp's `ggml_vk_guess_matmul_pipeline`. This is the
   biggest piece: matters mostly at small pp (≤256) but makes the
   variant table complete.
5. **Verify `gemm_down` actually hits the coopmat path.** With the
   Sprint 11E artifact in mind (gate→up sequence inflates the
   second dispatch's timestamp), confirm via RGP per-event timing,
   not just timestamp profiling.

Items 1+2 together should buy back the +29 ms `gemm_down`
regression and turn the ~6 % coopmat slowdown into the originally-
expected 8-15 % win. Items 3+4 chase the remaining gap to
llama.cpp.

I scoped this on a bug-budget basis: items 1+2 alone are ~150
LOC of careful changes (Q6_K coopmat shader build + ShaderId +
pipeline_registry spec block + routing) and need correctness
testing against the 15-prompt suite. Items 3+4 add another ~300
LOC. Doing all of this in one sprint without good incremental
tests risks shipping a broken default-on coopmat that regresses
the 0.80 × decode ratio we already have.

The brief's bench-gate of pp=512 ≥ 3500 tok/s also won't be
reached with items 1+2 alone:

- Today: pp=512 = 2207 tok/s (coopmat) / 2348 (default).
- With Q6_K fix, expected: ~2350 + ~400 (Q4_K wins kept; Q6_K
  brought back to default-or-better) ≈ 2750 tok/s. Still 0.64 ×
  llama.cpp.
- 3500 tok/s requires the full S/M/L tile coverage + aligned
  variants + FP16 activations on the matrix-core path — i.e.
  items 3+4 too.

## 6. Tests / regression status

No code changed. `cargo test --release --lib` reports
**27 / 27 lib tests pass** — identical to before this sprint.

## 7. Recommendation

**Sprint 12K — Q6_K coopmat shader + routing (items 1+2 above).**
Smallest viable step. Targets ~+15 % prefill (pp=512: 2207 →
~2750 tok/s with coopmat-on; default `mul_mmq` stays the fallback).
**Estimated effort: 2-3 days**, including:

- `build.rs` ShaderJob + `shaders.rs` enum entries + bytes-include.
- `pipeline_registry.rs` spec-constant block (reuse the Sprint 11E
  values).
- `layer_weight_shader_gemm` routing patch.
- Parity test: gemm_v / gemm_down output bit-comparable (or within
  FP16 noise) vs the `MulMmqQ6K` baseline.
- Bench-gate-2: pp=512 ≥ 2700 tok/s (≥ +15 % vs default).

Sprint 12L (sized after 12K confirms the recipe works) adds
items 3+4: aligned variant, S/M tiles, runtime selector. Targets
the bench-gate of 3500 tok/s.

If neither 12K nor 12L lands quickly: VulkanForge's prefill ceiling
is ~2350 tok/s on this hardware until someone writes a real
matrix-core GEMM stack. That's the honest project-level statement.

## 8. Outputs

- This report.
- **No code changes.** No new tests.
- 27 / 27 lib tests still green.
- All measurements reproducible via:
  - default: `VF_PP_LIST=128,256,512,1024,2048 VF_PP_RUNS=3 cargo run --release --example run_pp_bench`
  - coopmat-on: same with `VULKANFORGE_USE_MM_COOPMAT=1`
  - per-shader: `VF_PP=512 cargo run --release --example profile_prefill`
  - llama.cpp: `~/tmp/llama.cpp/build-vulkan/bin/llama-bench -m … -p N -n 0 -ngl 99 -r 3`
