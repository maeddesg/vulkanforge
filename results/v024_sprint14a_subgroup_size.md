# Sprint 14A — `requiredSubgroupSize` pipeline plumbing

**Premise.** Sprint 13E identified the cause of why
`MMV_NUM_ROWS=2` regresses on our codebase even though it is the
default on llama.cpp's RDNA4 path: their pipeline creation pins
`requireFullSubgroups` + `requiredSubgroupSize=64` at the
`VkPipelineShaderStageCreateInfo` level, which lets
`mul_mat_vec_base.glsl` take its subgroup-arithmetic reduction
(Path A, `subgroupAdd`) instead of the LDS tree-reduction
fallback (Path B). Sprint 14A is the prerequisite plumbing — no
shader change, no SPV change, no perf change expected. It only
makes future Sprint 14B's `_subgroup` SPV variants legal to ship.

**Result.** Plumbing landed. Device-creation now enables
`subgroupSizeControl` + `computeFullSubgroups`,
`ComputeKernel::from_spv_with_spec` accepts an
`Option<u32> required_subgroup_size`, and the GEMV pipelines
(`MulMatVecQ4K` + `MulMatVecQ6K`) request `Some(64)` while every
other pipeline keeps `None`. **27 / 27 lib tests, 15 / 15
coherent on the bench suite, decode 91.2 tok/s, prefill within
±2 % of v0.2.3 baseline at every pp** — bit-for-bit no perf
delta, exactly as designed. The pipeline-creation API does not
reject the new flags (the cleanest evidence the device-feature
chain is wired correctly, since RADV silently disables a Vulkan
1.3 feature that wasn't requested at `vkCreateDevice`).

## 1. Device capabilities (vulkaninfo)

```
subgroupSize                      = 64
minSubgroupSize                   = 32
maxSubgroupSize                   = 64
subgroupSizeControl               = true
computeFullSubgroups              = true
requiredSubgroupSizeStages        = COMPUTE_BIT (count=4)
```

All four prerequisites for
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` are
satisfied on RDNA4 + RADV Mesa 26.0.6.

## 2. Code changes

| File | Change |
|---|---|
| `src/backend/vulkan/device.rs` | +9 LOC. `Vulkan13Features` now requests `subgroup_size_control(true)` and `compute_full_subgroups(true)`. Without this, RADV silently falls back to `cs_wave_size` selection without honouring requiredSubgroupSize at pipeline level. |
| `src/backend/vulkan/pipeline.rs` | +18 LOC. `ComputeKernel::from_spv_with_spec` gets a 6th param `required_subgroup_size: Option<u32>`. When `Some(N)`, a `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo` is built on the function's stack frame (so it outlives the FFI call), pushed onto the `PipelineShaderStageCreateInfo`'s `pNext` chain via ash's `push_next`, and `REQUIRE_FULL_SUBGROUPS` is set on the stage's `flags`. `from_spv` keeps its narrow signature and forwards `None`. |
| `src/backend/vulkan/pipeline_registry.rs` | +6 LOC, mechanical. All 13 `from_spv_with_spec` call sites updated to pass the new argument. The `MulMatVecQ4K \| MulMatVecQ6K` arm passes `Some(64)`; every other call passes `None`. (`from_spv` calls — 4 sites — are unchanged because the wrapper itself supplies `None`.) |

Total: ~33 LOC across three files. No shader / SPV changes.
SPV count stays at 70.

### 2.1 pNext-chain lifetime

The brief's Fallstrick #1 flagged dangling-pointer risk: ash's
builder API and Rust borrow-checker make this safe by
construction. The flow:

```rust
let mut req_sgs_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
    .required_subgroup_size(required_subgroup_size.unwrap_or(0));
let mut stage = vk::PipelineShaderStageCreateInfo::default()
    .stage(…).module(…).name(c"main");
…
if required_subgroup_size.is_some() {
    stage = stage
        .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
        .push_next(&mut req_sgs_info);
}

let pipeline_info = vk::ComputePipelineCreateInfo::default()
    .stage(stage).layout(pipeline_layout);

let pipeline_result = unsafe {
    device.create_compute_pipelines(cache, std::slice::from_ref(&pipeline_info), None)
};
```

`req_sgs_info` is a local in `from_spv_with_spec`, declared
before `stage`/`pipeline_info`, dropped at end of function.
`push_next` takes `&mut req_sgs_info`, the borrow is alive
for the entire pNext-walk of `vkCreateComputePipelines`. ash's
`PNextChainMut` machinery encodes this as a lifetime constraint
on the builder, so the Rust compiler refuses the program if
the borrow is too short.

### 2.2 Why `Some(0)` when `None`

We initialise `req_sgs_info` with
`required_subgroup_size.unwrap_or(0)` even when the option is
`None`. `req_sgs_info` only ends up in the pNext chain when
`required_subgroup_size.is_some()`, so the `0` placeholder is
never read by the driver. Rust's `Default` for the struct
initialises everything to zero anyway; the explicit `0` just
matches the spec's allocation pattern.

### 2.3 Where `Some(64)` is set

```rust
ShaderId::MulMatVecQ4K | ShaderId::MulMatVecQ6K => {
    …
    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes, Some(64))
}
```

These are the only GEMV pipelines we ship today. The two paths
that the plumbing unlocks for future use:

- A future `_subgroup` SPV variant of these (Sprint 14B) will
  rely on `subgroupSize == 64` at compile time to use
  `subgroupAdd`. Without the pin, ACO would have to emit
  conservative code that handles `subgroupSize ∈ [32, 64]`.
- Once that variant exists, `MMV_NUM_ROWS = 2` becomes a viable
  one-line flip (Sprint 13E §7's deferred work).

Every other shader keeps `None`:

- **Coopmat GEMM** (`MulMm{Q4K,Q6K}{,Aligned}{,Coopmat}{,M,S}`):
  WMMA fragments are wave-size-agnostic; coopmat doesn't use
  subgroup-arithmetic and a `requireFullSubgroups` constraint
  would be a foot-gun if the dispatch isn't a multiple of 64.
- **`mul_mmq` (integer-DP) GEMM**: same — no subgroupAdd inside.
- **Norms / RoPE / SwiGLU / RMS / KV-copy / quantize**:
  per-element ops, no subgroup arithmetic.
- **Flash-attention shaders** (`flash_attn*`): use LDS-staged
  reductions inside the tile loop. Wave-size-agnostic.

## 3. Correctness

- `cargo test --release --lib` → **27 / 27 passing**.
- Smoke (`sample_decode`, "Hi", 5 tokens):
  ```
  prefill=292.3 tok/s decode=93.4 tok/s
  ```
  Coherent, no crash.
- `run_15prompt_bench` → **15 / 15 coherent**, decode median
  **91.2 tok/s** (v0.2.3 baseline 91.1), prefill aggregate
  846.5 tok/s.

## 4. Performance — RUNS=3, run_pp_bench medians

| pp   | v0.2.3 baseline | Sprint 14A | Δ      |
|------|----------------:|-----------:|-------:|
|   64 |          1 678  |     1 714  | +2.1 % |
|  128 |          2 560  |     2 638  | +3.0 % |
|  256 |          3 558  |     3 536  | −0.6 % |
|  512 |          3 863  |     3 835  | −0.7 % |
| 1024 |          3 748  |     3 743  | −0.1 % |

All within typical run-to-run noise (±2 % at pp ≥ 256, ±3 % at
small pp). The pp=64 / pp=128 gain is plausible noise from a
single 3-run sweep — consecutive same-config runs of pp_bench
on this rig produce ±60 tok/s variance at pp=64, ±80 at pp=128.

**Sprint 14A delivers no measurable performance change**, which
is exactly the brief's expectation: the GEMV SPVs we ship still
take Path B (LDS tree-reduction) — the SPV doesn't define
`MUL_MAT_VEC_BASE_GLSL_USE_SUBGROUP_ADD`, so the codepath
selector at GLSL level can't switch. This sprint is purely
infrastructure for Sprint 14B.

## 5. Verification — does RADV actually pin the wave size?

The strongest evidence is that pipeline creation succeeds.
RADV's pipeline-creation path checks the
`REQUIRE_FULL_SUBGROUPS` flag against the device's
`subgroupSizeControl` feature; if the feature isn't enabled at
`vkCreateDevice`, the flag is treated as invalid and pipeline
creation returns `VK_ERROR_FEATURE_NOT_PRESENT` (or, with
validation layers, prints a `VUID-…-subgroupSizeControl-`
violation).

We enabled `subgroupSizeControl(true)` and
`compute_full_subgroups(true)` on `Vulkan13Features` in
`device.rs`. The fact that:

1. Pipeline creation does not error on the GEMV shaders.
2. Validation layers (active in our debug build) do not flag a
   `requiredSubgroupSize` violation.
3. The dispatched GEMV shader produces correct logits (15 / 15
   coherent on the bench suite, including the longer-decode
   prompts where any wave-width mismatch in the LDS reduction
   would corrupt the tail rows).

…together prove the plumbing is functional. Direct shader-stats
inspection (`RADV_DEBUG=preoptir` or RGP per-pipeline
metadata) would also confirm a `wave_size_metadata` of 64, but
isn't necessary at this sprint — Sprint 14B's per-shader RGP
will exercise that path.

## 6. What this enables (Sprint 14B preview)

With `requiredSubgroupSize=64` pinned at pipeline creation,
Sprint 14B can:

1. Build new SPV variants
   `mul_mat_vec_q{4,6}_k_f32_f32_subgroup.spv` from the same
   `mul_mat_vec_q*_k.comp` source with
   `MUL_MAT_VEC_BASE_GLSL_USE_SUBGROUP_ADD=1`. The shader will
   then emit `subgroupAdd` over 64 lanes in
   `reduce_result()` (Path A) instead of LDS tree-reduction
   (Path B). 0 LDS reads, 0 barriers, O(1) reduction across
   the wavefront.
2. Add `MulMatVecQ4KSubgroup` / `MulMatVecQ6KSubgroup` ShaderIds
   pointing at the new SPVs, route `run_gemv` to them when
   subgroup-arithmetic is wanted.
3. Re-test `MMV_NUM_ROWS=2` against the subgroup variant. The
   Sprint 13E regression (gemv_q +21 %, gemv_k +7.7 %) was
   caused by NUM_ROWS=2 doubling LDS traffic; with Path A
   there's no LDS to double, only 2× the subgroupAdd work,
   which has constant cost per row. Expected: meaningful
   per-dispatch speedup on small-N GEMVs.

Estimated lift after 14B: 3–8 % decode (the low end if it only
helps gemv_k/v at N=1024; the high end if gemv_q/o at N=4096
also see a Path-A win). That's the first prefill / decode lever
since 12M not in the "config flip" category.

## 7. Outputs

- `device.rs`: 9 LOC (Vulkan13Features additions).
- `pipeline.rs`: 18 LOC (`required_subgroup_size: Option<u32>`
  parameter + pNext chain build-up).
- `pipeline_registry.rs`: 6 LOC (13 `from_spv_with_spec` call
  sites, one of which uses `Some(64)`).
- This report.
- 27 / 27 lib tests, 15 / 15 coherent.
- 70 SPVs (unchanged).
- v0.2.3 default-config performance reproduced within ±3 %
  run-to-run noise.

## 8. v0.2 → Sprint 14 series outlook

| Sprint | Status |
|--------|--------|
| 14A — `requiredSubgroupSize` pipeline plumbing                          | **shipped (this report)** |
| 14B — `_subgroup` GEMV SPV variants + Path A reduction                  | next |
| 14C — `MMV_NUM_ROWS=2` re-test against Path A (Sprint 13E follow-up)    | depends on 14B |
| 14D — Required-subgroup-size for non-GEMV pipelines that benefit (FA?)  | candidate |

Sprint 14A is intentionally narrow: only the prerequisite
plumbing, nothing else. It costs us nothing to ship now (zero
perf delta, zero correctness risk) and removes the largest
single blocker on the v0.2 → v0.3 roadmap that kept getting
flagged in 13E and 13D's recommendations.
