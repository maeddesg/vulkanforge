# Sprint 13E — MMV_NUM_ROWS=2: HONEST NEGATIVE on this codebase

**Premise.** Sprint 12G-D RGP profiling located gemv_k / gemv_v at ~22-23
µs / dispatch, ~13 % of decode forward; the K/V projections at N=1024
also have the lowest WG-count of any decode dispatch (1024 / 64 CUs =
16 WG/CU). llama.cpp's `ggml-vulkan.cpp:4128` sets `rm_kq = 2` for
non-GCN AMD (which includes RDNA4) — every K-quant GEMV pipeline gets
`NUM_ROWS=2` unconditionally, halving WG count and giving each WG two
output rows to share an activation read. Hypothesis: porting that
recipe should pick up a small but real decode win, especially at K/V.

**Verdict.** **Hypothesis falsified.** Pre-check passed (shader source
byte-identical to llama.cpp HEAD, `NUM_ROWS` is `constant_id = 1` in
`mul_mat_vec_base.glsl:90` so the change is a one-line
spec-constant flip with zero new SPVs). Made the change, ran same-
session A/B with `profile_positions` and 15-prompt bench:

- 15-prompt decode median: 91.6 / 90.9 (NUM_ROWS=1) vs 91.6 / 91.0 / 90.9
  (NUM_ROWS=2). Within run-to-run noise.
- `profile_positions` pos=200 same-session: NUM_ROWS=2 is **measurably
  slower** per dispatch on small-N GEMVs:
    - gemv_q (N=4096): 669 → 809 µs (+21 %)
    - gemv_k (N=1024): 777 → 837 µs (+7.7 %)
    - gemv_v (N=1024): 818 → 839 µs (+2.6 %)
    - gemv_o (N=4096): 543 → 548 µs (~flat)
    - gemv_down (N=4096): 1941 → 1926 µs (~flat)
    - Forward wall: 11 990 → 12 337 µs (+2.9 %)

Reverted to `NUM_ROWS = 1`. 27 / 27 lib tests + 15 / 15 coherent on
both configurations. **Bench-gate (decode ≥ 94 tok/s) not met under
either configuration** (decode plateau is 91 tok/s on this codebase
regardless of NUM_ROWS) — but the gate failure is not what makes this
sprint negative; it's that NUM_ROWS=2 is *worse*, not just flat.

This is the same shape as Sprints 12D / 12E / 12H / 13B / 13C: a port
of llama.cpp's recipe that does not translate on our infrastructure
even when shader source is identical, identified with a small-effort
empirical test. The brief's Fallstrick #5 spelled the answer in
advance:

> Sprint 12G-C zeigte: gemv_k/v bei ~85-91% Peak BW! Die BW ist FAST
> am Ceiling! NUM_ROWS=2 verbessert Weight-REUSE (L2 Cache!) — ABER:
> bei M=1 Decode gibt es fast KEINEN Activation-Reuse!

The deeper reason isn't bandwidth though — it's that llama.cpp's
NUM_ROWS=2 path uses a different reduction strategy than ours, see §6.

## 1. Pre-check (passed) — shader supports NUM_ROWS as spec-constant

```
$ grep -n "NUM_ROWS" vk_shaders/mul_mat_vec_base.glsl
89:layout (constant_id = 0) const uint BLOCK_SIZE = 32;
90:layout (constant_id = 1) const uint NUM_ROWS = 1;
91:layout (constant_id = 2) const uint NUM_COLS = 1;
130:shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
132:void reduce_result(...)

$ md5sum vk_shaders/mul_mat_vec_q4_k.comp …llama.cpp/.../mul_mat_vec_q4_k.comp
9112134d…  vk_shaders/mul_mat_vec_q4_k.comp
9112134d…  …/llama.cpp/…/mul_mat_vec_q4_k.comp   ← identical

$ md5sum vk_shaders/mul_mat_vec_q6_k.comp …llama.cpp/.../mul_mat_vec_q6_k.comp
2fe68d21…  vk_shaders/mul_mat_vec_q6_k.comp
2fe68d21…  …/llama.cpp/…/mul_mat_vec_q6_k.comp   ← identical
```

`NUM_ROWS` is a Vulkan spec-constant (`constant_id = 1`), not a
compile-time `#define`. Changing the value is a one-line edit to
`MMV_NUM_ROWS` in `pipeline_registry.rs:34`; the dispatch path in
`forward.rs::run_gemv:1884` already does
`groups = (m + n_rows - 1) / n_rows`. **Zero new SPVs.**

The shader's existing `compute_outputs` writes `NUM_ROWS` output rows
per workgroup, with `first_row = NUM_ROWS * gl_WorkGroupID.x` and a
bounds check `if (first_row + NUM_ROWS <= p.stride_d)` (line 126).
All our N values (1024, 4096, 12288, 151936) are even, so no tail-WG
remainder.

## 2. llama.cpp's recipe (`ggml-vulkan.cpp:4127-4142`)

```cpp
uint32_t rm_stdq = 1;
uint32_t rm_kq = 2;          // ← K-quants (Q4_K, Q6_K, etc.)
…
if (device->vendor_id == VK_VENDOR_ID_AMD) {
    if (device->architecture == AMD_GCN) {
        rm_stdq = 2;
        rm_kq = 4;            // GCN gets NUM_ROWS=4
    }
    // RDNA*: rm_kq stays at default 2
}
```

So on RDNA4 (gfx1201, non-GCN), every K-quant GEMV pipeline is built
with `NUM_ROWS = 2`. Pipeline create call (`ggml-vulkan.cpp:4180`):

```cpp
ggml_vk_create_pipeline(...,
    "mul_mat_vec_q4_k_f32_f32", …,
    {rm_kq, 1, 1},                 // wg_denoms (output dim divides by rm_kq)
    {wg_size_subgroup16, rm_kq, i+1},  // spec_constants {BLOCK_SIZE, NUM_ROWS, NUM_COLS}
    1, true, use_subgroups16, force_subgroup_size16);   // ← see §6
```

Two important pipeline-creation parameters we **do not** match:

- `use_subgroups16 = use_subgroups && subgroup_min_size_16` — gates
  the subgroup-arithmetic reduction in `reduce_result`.
- `force_subgroup_size16 = …subgroup_size16 (= 64 on RDNA4)` —
  pinned via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`.

Our `ComputeKernel::from_spv_with_spec` (in `pipeline_registry.rs`)
does **not** pass `requireFullSubgroups` or pin a subgroup size on
the GEMV pipelines. That choice has been fine for NUM_ROWS=1; it
becomes the limiting factor at NUM_ROWS=2, see §6.

## 3. Code change (made and reverted)

One-line edit:

```rust
// pipeline_registry.rs:34
- pub const MMV_NUM_ROWS: u32 = 1;
+ pub const MMV_NUM_ROWS: u32 = 2;
```

`run_gemv` already reads `MMV_NUM_ROWS` to set
`groups = ceil(m / n_rows)`. The constant is consumed twice (pipeline
spec-constant block + dispatch geometry) so they cannot drift.

Build clean, 27 / 27 lib tests pass, 15 / 15 coherent. The math is
correct end-to-end — this is a pure perf negative, not a correctness
issue.

## 4. Same-session A/B — 15-prompt bench

`run_15prompt_bench`, decode median across the 15-prompt suite:

| Config             | Run 1 | Run 2 | Run 3 |
|--------------------|------:|------:|------:|
| `MMV_NUM_ROWS = 1` |  91.6 |  90.9 |   —   |
| `MMV_NUM_ROWS = 2` |  91.6 |  91.0 |  90.9 |

Within run-to-run noise (±0.5 tok/s). The 15-prompt suite is
dominated by a few longer decodes (1024-tok generation) and is not
sensitive enough to a 2-3 % per-dispatch shift on a single dispatch
class — but `profile_positions` is.

## 5. Same-session A/B — `profile_positions` pos=200

Steady-state per-shader profile, 36 layers, total µs across all
calls in one forward:

| Shader      | N      | NUM_ROWS=1 (µs) | NUM_ROWS=2 (µs) | Δ      |
|-------------|-------:|----------------:|----------------:|-------:|
| gemv_q      |   4096 |             669 |             809 | **+21 %** |
| gemv_k      |   1024 |             777 |             837 |  +7.7 % |
| gemv_v      |   1024 |             818 |             839 |  +2.6 % |
| gemv_o      |   4096 |             543 |             548 |  +0.9 % |
| gemv_down   |   4096 |            1941 |            1926 |  −0.8 % |
| gemv_gate   |  12288 |            1715 |            1870 |  +9.0 % * |
| gemv_up     |  12288 |            3010 |            3030 |  +0.7 % * |
| fa_split    |    —   |            1283 |            1283 |   0.0 % |
| Forward wall|        |          11 990 |          12 337 | **+2.9 %** |
| Effective   |        |    83.4 tok/s   |    81.1 tok/s   | −2.8 %  |

\* The gemv_gate / gemv_up pair fires back-to-back without a barrier
   between them; per Sprint 12G-D's documented timestamp artifact,
   the second dispatch's `vkCmdWriteTimestamp` reading inflates with
   the first's GPU time. Don't read those two rows literally — the
   sum of the two is the meaningful number, and it is +175 µs (3.7 %
   slower in aggregate, before pp=64 outliers wash out).

The honest reading: NUM_ROWS=2 is **slower per dispatch on small-N
GEMVs** by single-digit-to-21 %, with the largest hit on the
N=4096 Q4_K case (gemv_q). The per-shader regression is real and
reproducible; the 15-prompt aggregate hides it because attention +
larger GEMVs dominate the wall.

## 6. Why NUM_ROWS=2 is slower on our pipeline

`mul_mat_vec_base.glsl` ships **two** `reduce_result` paths:

```glsl
// Path A — used when MUL_MAT_VEC_BASE_GLSL_USE_SUBGROUP_ADD is defined
// (line 89-126). One subgroupAdd per (col, row). Cost: O(NUM_COLS·NUM_ROWS)
// subgroup ops, no LDS.

// Path B — fallback (line 128-167). LDS-based parallel tree reduction
// across BLOCK_SIZE threads. Cost: O(NUM_COLS·NUM_ROWS·log(BLOCK_SIZE))
// LDS reads/writes, plus a barrier per level.
```

llama.cpp's pipeline creation (`ggml-vulkan.cpp:4180`) passes
`use_subgroups16 = true` AND `force_subgroup_size16 = 64`. That:

1. Defines `MUL_MAT_VEC_BASE_GLSL_USE_SUBGROUP_ADD` at SPV-build time
   so the shader takes Path A.
2. Pins `requiredSubgroupSize = 64` at pipeline creation so the
   subgroupAdd actually runs across all 64 wave lanes, getting full
   benefit.

Our pipeline path takes Path B (LDS reduction) regardless of NUM_ROWS.
At NUM_ROWS=1 that's BLOCK_SIZE=64 LDS ops → tree reduction in 6
levels, each with a barrier. At NUM_ROWS=2 it's 2× the LDS traffic
plus the same 6 barrier levels, which more than eats the WG-count
halving benefit. llama.cpp's Path A scales linearly in NUM_ROWS so
their NUM_ROWS=2 lands cleanly.

Confirmation: `mul_mat_vec_base.glsl:128-167` shows the LDS path is
the version we ship (no extension include for `subgroup_arithmetic`,
no `requireFullSubgroups` flag in our pipeline creation). The shader
source is byte-identical to llama.cpp; the **pipeline-creation infra**
isn't.

## 7. What it would take to actually port

If this turned into a real sprint:

1. **Add a subgroup-arithmetic GEMV variant.** llama.cpp builds three
   variants per quant via `vulkan-shaders-gen.cpp:711-713`:
   `mul_mat_vec_q4_k_f32_f32`, `_subgroup`, `_subgroup_no_shmem`. We
   ship only the first.
2. **Pin required subgroup size 64 at pipeline creation.** Needs a
   `VkPhysicalDeviceVulkan13Features::subgroupSizeControl` feature
   check + `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`
   plumbing in `ComputeKernel::from_spv_with_spec`. Not a drop-in
   change — touches the pipeline-creation path used by every other
   shader too.
3. **Then** flip `MMV_NUM_ROWS = 2` and pick up llama.cpp's small-N
   GEMV win. Estimated lift: maybe 3-5 % decode if K/V are the bound,
   1-2 % otherwise. Not the +3 tok/s the brief sketched, but
   directionally positive once the reduction infra is in place.

That is a Sprint 14-class effort touching a fundamental piece of our
pipeline-creation code. Not in scope for Sprint 13E. Recommend
deferring to v0.3 or pairing with whatever sprint introduces
`requireFullSubgroups` for other shaders.

## 8. What didn't materialise

The brief's optimistic estimate (+0.4-0.7 % wall, possibly +1.9 % if
extended to all N≤4096 GEMVs) assumed each GEMV's per-dispatch time
would shrink by ~5 % from doubled activation reuse. Two reasons it
didn't:

- **Activation is one token at decode (M=1)**, so the
  activation-reuse benefit is approximately zero — the brief's
  Fallstrick #5 flagged this. The hypothesised gain comes from
  fewer-but-bigger workgroups giving better CU-level scheduling, not
  from cache reuse.
- **GEMV is bandwidth-bound near ceiling already** (Sprint 12G-D /
  12H: 77-91 % peak HBM). Halving the WG count doesn't reduce the
  total bytes read; it just moves them across fewer waves. With
  ACO's current scheduling and our LDS-reduction path, fewer waves
  per CU at NUM_ROWS=2 is *worse* for HBM-latency hiding, not better.

## 9. Outputs

- `src/backend/vulkan/pipeline_registry.rs` — comment updated to
  document Sprint 13E re-verification of Phase-2A's wash. `MMV_NUM_ROWS`
  stays at `1`.
- This report.
- **No code changes shipped** beyond the comment refresh. v0.2.2
  binary unchanged. 27 / 27 lib tests, 15 / 15 coherent.

## 10. Sprint 13 takeaway

After 13A (S-tile, neutral on RDNA4), 13B (Mesa 26.1, neutral),
13C (f16acc, slight regression), and 13E (NUM_ROWS=2, slight
regression), every prefill / decode lever that was a "port llama.cpp's
config" hypothesis has now been measured and falsified on this
codebase + this hardware. The remaining gap to llama.cpp lives at the
**pipeline-infrastructure** layer:

- Required subgroup size pinning
- Subgroup-arithmetic GEMV / FA reductions
- Multi-submit overlap
- `quantize_q8_1` fusion into the GEMM dispatch

These are larger work items than a single config flip. v0.3 territory
or a dedicated infrastructure sprint. The prefill arc that started in
Sprint 12I (0.54 × → 0.89 × llama.cpp at pp=512) cannot be pushed
materially further with shader-source / spec-constant work alone.
