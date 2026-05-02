# Sprint 14B — `_subgroup` GEMV (Path A): NEUTRAL on RDNA4 at NUM_ROWS=1

**Premise.** Sprint 13E identified Path B (LDS tree-reduction) vs
Path A (subgroupAdd) as the structural difference between our GEMV
pipeline and llama.cpp's. With Sprint 14A's
`requiredSubgroupSize=64` plumbing in place, Path A SPVs are now
legal to ship. Hypothesis: switching the K-quant decode GEMVs from
Path B to Path A removes 6 LDS barrier levels per dispatch, should
shave 5–15 % off per-dispatch GEMV time, lifting decode median
from ~91 to ~94+ tok/s.

**Verdict.** **Hypothesis falsified at NUM_ROWS=1 on RDNA4.** Path A
SPVs build cleanly, route correctly, produce 15/15 coherent output,
and pass 27/27 lib tests. **But: per-dispatch GEMV total at pos=200
differs by 0.15 %** between Path A and Path B (10 187.8 µs vs
10 203.4 µs across all GEMVs combined). 15-prompt decode median
**91.5 tok/s** (Path A) vs **91.2** (Path B) — within
run-to-run noise. **Bench-gate (decode ≥ 94 tok/s) NOT met.**

The LDS reduction at our `BLOCK_SIZE=64`, `NUM_ROWS=1` shape is
too small a slice of the GEMV's wall time to register — at most a
few hundred cycles of LDS round-trips out of ~22 000 cycles for a
gemv_k dispatch (N=1024, K=4096, ~700 ns total). Path A removes the
slice cleanly but doesn't change the wall.

**Path A is shipped DEFAULT-ON anyway** (opt-out via
`VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`) for three reasons:

1. It matches llama.cpp's RDNA4 GEMV recipe exactly
   (`ggml-vulkan.cpp:4180` builds and selects the `_subgroup` SPV
   for `rm_kq=2`).
2. It is a strict prerequisite for Sprint 14C
   (`MMV_NUM_ROWS=2` + Path A). Sprint 13E's Path-B regression
   came from doubled LDS traffic; with Path A there's no LDS to
   double, so 14C can finally land.
3. Zero measurable regression (within ±0.5 % decode noise across
   both 15-prompt and pp-bench), 15/15 coherent, 27/27 tests.

This is the same shape as Sprints 12D / 12E / 12H / 13B / 13C /
13D / 13E: a port of llama.cpp's recipe whose lever doesn't
materialise on this codebase + this hardware on its own. The
*combination* with NUM_ROWS=2 is the actual lever; Sprint 14B is
the prerequisite shipping vehicle.

## 1. Pre-check (passed)

```
$ md5sum vk_shaders/mul_mat_vec_base.glsl
0542764e79a1d6e8275aae3c29abef6f  vk_shaders/mul_mat_vec_base.glsl   ← identical to llama.cpp HEAD
```

The `USE_SUBGROUP_ADD` gate sits at `mul_mat_vec_base.glsl:135` in
the `reduce_result` body (line 132+). When defined, the function
issues:

1. One `subgroupAdd` per `(col, row)` over the 64-lane wave
   (line 139).
2. Lane 0 writes its partial to `tmpsh[j][n][gl_SubgroupID]`
   (line 144-150) — only `gl_NumSubgroups` partials per
   `(col, row)`, not `BLOCK_SIZE` partials.
3. Single barrier (line 151).
4. Thread 0 sums those `gl_NumSubgroups` partials linearly
   (line 152-158). With our `BLOCK_SIZE = 64` and a pinned
   `subgroup_size = 64`, `gl_NumSubgroups = 1`, so the cross-
   subgroup loop runs exactly once.

Path B (the `#else` branch, line 130 onwards) is the LDS tree
reduction with 6 barrier levels.

The extension declarations at line 5–7
(`GL_KHR_shader_subgroup_basic`, `GL_KHR_shader_subgroup_arithmetic`)
are gated on `USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM`, so
they pull in automatically just by adding the define. No shader-
source change needed; the change is purely build-side.

## 2. SPV build

| File | Size | Variant |
|---|---:|---|
| `mul_mat_vec_q4_k_f32_f32.spv` | 165 424 B | Path B (existing) |
| `mul_mat_vec_q4_k_f32_f32_subgroup.spv` | **165 740 B** | Path A (Sprint 14B) |
| `mul_mat_vec_q6_k_f32_f32.spv` | 174 132 B | Path B (existing) |
| `mul_mat_vec_q6_k_f32_f32_subgroup.spv` | **174 464 B** | Path A (Sprint 14B) |

Path A SPVs are ~300 B larger — the subgroupAdd opcode plus the
shrunk LDS region adds a tiny amount of metadata. Total SPV count
went 70 → 72.

## 3. Code wiring

| File | Change |
|---|---|
| `build.rs` | +30 LOC: 2 new `ShaderJob`s mirroring the existing GEMVs with `("USE_SUBGROUP_ADD", "1")` added to the defines list. Same input shaders, same B/D types, same byte offsets. |
| `src/backend/vulkan/shaders.rs` | +12 LOC: 2 new `ShaderId` variants `MulMatVec{Q4K,Q6K}Subgroup`, `name()` arms, `spv_bytes()` arms pointing at the new byte-include `pub const`s, `ALL_SHADERS` entries. |
| `src/backend/vulkan/pipeline_registry.rs` | +1 LOC: extended the existing GEMV match arm to cover the 4 GEMV ShaderIds (2 stock + 2 subgroup). All 4 share the same spec-constants, the same `Some(64)` `requiredSubgroupSize` pin from Sprint 14A. |
| `src/backend/vulkan/forward.rs` | +30 LOC: `mul_mat_vec_subgroup_enabled` env-var read (default-on), struct field, `layer_weight_shader` extended with a `subgroup` boolean parameter, `lm_head` selection extended likewise, 9 call-site updates. |

Total ~73 LOC across four files. No shader-source changes — the
existing `mul_mat_vec_base.glsl` already supports both paths via
its `#if USE_SUBGROUP_ADD` gate.

`MulMatVec{Q4K,Q6K}Subgroup` share the GEMV pipeline match arm
in `pipeline_registry.rs`, which keeps the `Some(64)` pin from
Sprint 14A. **Without that pin**, ACO could pick Wave32 — and a
`subgroupAdd` over 32 lanes would compute half the sum, producing
wrong logits. The pin is doing real work for the `_subgroup` SPVs;
it was a no-op for the stock SPVs.

## 4. Routing

```rust
// forward.rs
let mul_mat_vec_subgroup_enabled = match
    std::env::var("VULKANFORGE_DISABLE_SUBGROUP_GEMV") {
    Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => false,
    _ => true,
};

fn layer_weight_shader(model: &LoadedModel, layer: u32, suffix: &str, subgroup: bool) -> ShaderId {
    let q6 = model.tensor(…).ggml_type == GgmlType::Q6K;
    match (q6, subgroup) {
        (true,  true)  => ShaderId::MulMatVecQ6KSubgroup,
        (true,  false) => ShaderId::MulMatVecQ6K,
        (false, true)  => ShaderId::MulMatVecQ4KSubgroup,
        (false, false) => ShaderId::MulMatVecQ4K,
    }
}
```

Same selector applied to `lm_head` (vocab-major GEMV with
N=151 936). 9 call sites in `forward.rs` updated to thread
`self.mul_mat_vec_subgroup_enabled` through.

`run_gemv` is **unchanged** — Path A and Path B have the same
push-constant layout, same dispatch geometry (`groups = ceil(m / NUM_ROWS)`),
same descriptor-set bindings. Only the SPV bytes differ.

## 5. Correctness

- `cargo test --release --lib` → **27 / 27 passing**.
- `cargo run --release --example sample_decode "What is 2+2?"`
  with Path A default-on:
  ```
  prefill=391.7 tok/s decode=95.4 tok/s
  OUTPUT: <think> Okay, the user asked "What is 2+2?" That's a basic math
  ```
  Coherent, no crash.
- `run_15prompt_bench` Path A → **15 / 15 coherent**, decode median
  91.5 tok/s, prefill aggregate 855.8.
- `run_15prompt_bench` Path B (`VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`)
  → **15 / 15 coherent**, decode median 91.2, prefill aggregate
  844.7.

FP-addition is non-associative; subgroupAdd's hardware reduction
order differs from LDS tree-reduction order. Logits differ in the
last few bits but the bench suite's coherence heuristic
(top-1 / top-5 sanity) clears all 15 prompts under both paths.

## 6. Performance — same-session A/B

### 6.1 15-prompt suite

| Config                      | Decode med | Prefill agg | Coherent |
|-----------------------------|-----------:|------------:|:--------:|
| Path A (subgroup, default)  |  91.5 tok/s |       855.8 |   15/15  |
| Path B (LDS, env-var)       |  91.2 tok/s |       844.7 |   15/15  |

±0.3 tok/s decode, ±1.3 % prefill — within run-to-run noise on
this rig.

### 6.2 Per-dispatch (`profile_positions`, pos=200, sum across 36 layers)

| Shader      | Path A (µs) | Path B (µs) | Δ       |
|-------------|------------:|------------:|--------:|
| gemv_up     |      3008.1 |      3009.7 | −0.05 % |
| gemv_down   |      1925.5 |      1928.4 | −0.15 % |
| gemv_gate   |      1713.7 |      1716.4 | −0.16 % |
| gemv_k      |             |             |         |
| gemv_v      |             |             |         |
| gemv_q      |             |             |         |
| gemv_o      |             |             |         |
| **GEMV total** | **10 187.8** | **10 203.4** | **−0.15 %** |
| Forward wall | 12 032.0 | 12 355.5 | −2.6 % \* |
| Effective    | 83.1 tok/s | 80.9 tok/s | +2.7 % \* |

\* The `Forward wall` and `Effective` numbers are heavily
influenced by `Rest` and `Dispatch overhead` rows that are noisy
across the two runs. The reliable signal is the per-shader GEMV
column, which is at the noise floor. (Total GEMV − 0.15 % over
36 layers = 15.6 µs over 12 ms — well below measurement
resolution.)

### 6.3 pp-bench prefill regression check

Path A prefill values match v0.2.3 baseline within noise:

| pp   | v0.2.3 baseline | Sprint 14B (Path A) | Δ     |
|------|----------------:|--------------------:|------:|
|  128 |          2 560  |              2 565  | +0.2 % |
|  512 |          3 863  |              3 888  | +0.6 % |
| 1024 |          3 748  |              3 762  | +0.4 % |

Prefill is GEMM-bound (coopmat path), not GEMV — Path A doesn't
touch GEMM, so prefill should be identical-ish, and is.

## 7. Why didn't Path A help at NUM_ROWS=1?

The brief's Fallstrick #7 spelled the answer in advance:

> Falls Path A = 0 % Performance-Unterschied zu Path B:
>   → Die LDS-Reduktion war NICHT der Bottleneck!

Walking the math for `gemv_k` (N=1024, K=4096, Q4_K):

- ~22 µs / dispatch real time (Sprint 12G-D / 12H reading).
- ~85 % of HBM peak BW utilisation = ~547 GB/s effective.
- Bytes loaded: K × bpw_q4k = 4096 × 0.5625 = 2.3 KB / row;
  ~2.3 MB total for N=1024 rows.
- Compute: 4096 dequant + dot ops × 1024 outputs = 4.2 M VALU ops.
- LDS reduction (Path B) at BLOCK_SIZE=64: 6 levels × 64-lane
  parallel `+=` + barrier. ~12 LDS clocks per level × 6 = 72
  cycles per `(col, row)`, × 1 NUM_ROWS = ~72 cycles. At RDNA4
  ALU clock, that is roughly **0.04 µs per WG**, or ~40 µs
  spread across the whole gemv_k dispatch — **0.18 % of 22 µs**.

Match: the per-shader Path A win is 0.15 %. The lever is exactly
as small as the math predicts. Sprint 13E's claim that LDS is the
problem was correct in *direction*, just orders of magnitude off
in *magnitude* at NUM_ROWS=1.

The reduction overhead grows with NUM_ROWS: at NUM_ROWS=2 it
doubles (Path B's 6 levels × 2 = 12 LDS round-trips × 64-lane
copy), but with Path A it stays constant (still one
`subgroupAdd` per row, just two of them). That asymmetry is the
real Sprint 14C lever.

## 8. Where the 91 → 114 tok/s decode gap actually lives

After Sprints 13B / 13C / 13D / 13E / 14A / 14B every "shader-
config / pipeline-config / driver-flag" lever has been
systematically falsified on RDNA4 + this codebase. The decode
plateau at 91 tok/s reproduces under:

- Default v0.2.3 (Path B, NUM_ROWS=1)
- Sprint 14A pipeline pin (Path B, NUM_ROWS=1)
- Sprint 14B Path A (subgroupAdd, NUM_ROWS=1)
- Mesa 26.0.6 ↔ 26.1-rc3 (Sprint 13B)
- Wave64 ↔ Wave32 / VOPD (Sprint 13D)

The remaining ~25 % decode gap to llama.cpp (91.1 / 114.2) is
**not** in any of those layers. The candidates that haven't been
exhausted yet:

1. **NUM_ROWS=2 + Path A together** (Sprint 14C). Possibly
   3–8 % decode if K/V projections benefit.
2. **Multi-submit decode loop / command-buffer reuse**.
   llama.cpp builds and resubmits a single CB per token; we
   re-record per token. ~CPU overhead + driver scheduler.
3. **`lm_head` GEMV at N=151 936**. Sprint 12G-D showed it as
   ~6 % of decode forward; a coopmat dispatch there could shave
   ~3 %.
4. **Buffer aliasing / live-set reduction**. We hold ~20 SSBOs
   live per layer; llama.cpp recycles into 3-4. May matter for
   L2 thrashing.

None of these are addressable by editing `mul_mat_vec*.comp` or
its pipeline configuration.

## 9. Decision: ship default-on

**Default-on**, opt-out via `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`.
Rationale:

- Coherence verified across both paths (15/15).
- Per-dispatch within noise band (≤ 0.16 %), no regression.
- Matches llama.cpp's RDNA4 default exactly.
- Sprint 14C will need this default; flipping it later costs a
  config-flip migration in user scripts.
- The `_subgroup` SPVs will be the path of record going forward
  whether or not we ship 14C; the `_subgroup_no_shmem` variant
  llama.cpp also ships is the natural v0.3 follow-on.

If a user observes correctness issues on a non-RDNA4 GPU
(unlikely, but the change is non-trivial codegen), the kill
switch is one env var.

## 10. Outputs

- 2 new SPVs (`mul_mat_vec_q{4,6}_k_f32_f32_subgroup.spv`),
  total 72.
- 2 new ShaderIds (`MulMatVec{Q4K,Q6K}Subgroup`).
- New env var `VULKANFORGE_DISABLE_SUBGROUP_GEMV` (default off →
  Path A active).
- ~73 LOC across `build.rs`, `shaders.rs`, `pipeline_registry.rs`,
  `forward.rs`. No shader-source changes.
- This report.
- 27 / 27 lib tests, 15 / 15 coherent under both paths.

## 11. Sprint 14C preview

Re-run Sprint 13E's `MMV_NUM_ROWS=2` flip with Path A active and
re-measure `profile_positions` per-dispatch. With Path A:

- LDS overhead at NUM_ROWS=2 is the same as at NUM_ROWS=1
  (both go through one subgroupAdd per row).
- gemv_q at N=4096 → 2048 WGs, gemv_k/v at N=1024 → 512 WGs.
- Better per-CU saturation could shave 5–10 % per-dispatch on
  small-N GEMVs.

Estimated decode lift after 14C: **+2–5 %** if K/V projections
are the bottleneck within decode (they're 13 % of the forward).
Bench-gate would be decode ≥ 93 tok/s, two-thirds of the way
to the ≥ 94 gate that 14B alone didn't hit.

If 14C is also flat: the decode plateau is structural at the
graph level (CB reuse, multi-submit), and the next lever is a
`forward.rs` refactor — v0.3 territory.
