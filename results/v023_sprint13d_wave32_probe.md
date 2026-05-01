# Sprint 13D — Wave32 / VOPD probe (EXPERIMENTAL): NEUTRAL on this workload

**Premise.** RDNA4 (gfx1201) gates three features behind Wave32:
VOPD dual-issue VALU, dynamic VGPR allocation, and 32-Waves-per-SIMD
occupancy. Sprint 12G-D RGP showed VALU at 27.5 % utilisation (busy
27.5 %, idle 72.5 %) and 13/16 SGPR-limited occupancy on our coopmat
GEMM dispatches. If the 72.5 % idle time is hiding compute capacity
that VOPD could fill, Wave32 should show a clear win. Sprint 13B
already verified Mesa 26.1-rc3 is functionally stable on gfx1201, so
the prerequisite is met.

**Verdict.** **EXPERIMENTAL — production stays Wave64 + Mesa 26.0.6.**
ACO does emit VOPD massively under Wave32 (**3 546 `v_dual_*`
instructions** vs **65** under Wave64 across our entire shader set
under Mesa 26.1, RADV_PERFTEST=cswave32). 27 / 27 lib tests + 15 / 15
coherent under Wave32, no crashes, no precision regressions.
**But: decode is flat (90.8 vs 90.7 tok/s), prefill is mildly worse
(−1.8 % to −3.2 % across pp ∈ {64, 128, 256, 512, 1024})** —
within run-to-run noise on the decode side, just outside the noise
band on prefill.

The brief's outcome bins:

| Bin                          | Decode | Prefill |
|------------------------------|:------:|:-------:|
| **+10 to +20 %** (game-changer) | —      | —      |
| **+3 to +10 %** (v0.3 evaluation) | —      | —      |
| **±2 %** (deferred)          | **YES** | (close, decode side)  |
| **negative** (don't use)     | —      | mild on prefill |

So this lands **decode-NEUTRAL, prefill-MILDLY-NEGATIVE**, mapping to
the brief's "deferred" bin overall. **No runtime code change shipped.**
Wave32 / VOPD is not a v0.3 priority on RDNA4 + this workload.

This is the **fifth honest negative** in the Sprint 13 series after
12D / 12E / 12H / 13B / 13C / 13E. The prefill arc that took us from
0.54 × → 0.89 × llama.cpp at pp=512 (Sprints 12I → 12M) was the last
lever attainable through shader-config or driver work alone.

## 1. Setup

| Component | Value |
|---|---|
| Mesa | 26.1.0-rc3, local at `~/tmp/mesa-26.1/` (Sprint 13B) |
| System Mesa | 26.0.6-arch2.2, **untouched** |
| Activation | `VK_ICD_FILENAMES=…` + `LD_LIBRARY_PATH=…` + `RADV_PERFTEST=cswave32` |
| Hardware | AMD RX 9070 XT (RDNA4, gfx1201) |
| Workload | Qwen3-8B-Q4_K_M, v0.2.2 binary post-Sprint 13E |

## 2. Finding the Wave32 flag

The brief sketched `RADV_PERFTEST=w32cs` as the most likely name; the
actual flag in Mesa 26.1-rc3 is **`RADV_PERFTEST=cswave32`**:

```
$ grep -rn "cswave32" ~/tmp/mesa-26.1.0-rc3/src/amd/vulkan/
…/radv_instance.c:110:   {"cswave32", RADV_PERFTEST_CS_WAVE_32},
…/radv_shader.c:398:         default_wave_size = pdev->cs_wave_size;
…/radv_shader.c:400:      /* Games don't always request full subgroups when they should,
                              which can cause bugs if cswave32 […] */
```

`vulkaninfo` doesn't reflect the flag in `subgroupSize` (always 64) —
that field is the device-default subgroup size, not the per-shader
wave size ACO chooses. The flag changes the compute-shader codegen
path inside RADV, visible only by inspecting compiled shader assembly.

## 3. Smoke + correctness

```
$ … RADV_PERFTEST=cswave32 cargo run --release --example sample_decode \
    VF_PROMPT="Say hi in one short sentence." VF_MAX_TOKENS=12

OUTPUT: <think> Okay, the user asked me to say hi in
prefill=400.2 tok/s decode=94.5 tok/s
```

No crash, coherent text. Lib tests:

```
$ … RADV_PERFTEST=cswave32 cargo test --release --lib
test result: ok. 27 passed; 0 failed; 0 ignored
```

Same as Wave64. **No subgroup-size-related correctness break** —
none of our shaders hard-codes Wave64. Coopmat 16×16×16 fragments
work correctly under Wave32 too (the lane→cell mapping inside the
WMMA hardware is the same regardless of wave size).

## 4. VOPD instruction count

Pinned the `RADV_DEBUG=asm` shader-disassembly dump and counted
`v_dual_*` instructions across the whole inventory loaded for one
forward pass:

| Wave size | `v_dual_*` instruction count |
|-----------|-----------------------------:|
| **Wave64** | **65** |
| **Wave32** | **3 546** |

ACO does emit VOPD aggressively under Wave32 — 54× more dual-issue
instructions than Wave64. The infrastructure works as designed.
The 65 v_dual under Wave64 are stragglers from auxiliary code paths
(probably some graphics / loader bits that bypass the `cs_wave_size`
gate); the 3 546 figure is the substantive signal that compute
shaders are getting dual-issued under Wave32.

**This is the cleanest evidence in this sprint series that the lever
is real at the codegen level.** The question is whether it translates
to wall-time. Answer below.

## 5. Performance — same Mesa, Wave64 vs Wave32

15-prompt suite (decode + prefill medians, single run each):

| Metric              | Wave64 (Mesa 26.1) | Wave32 (Mesa 26.1, cswave32) | Δ %     |
|---------------------|------------------:|------------------------------:|--------:|
| Decode median tok/s |              90.7 |                          90.8 |  +0.1 % |
| Prefill median tok/s|             840.7 |                         850.2 |  +1.1 % |
| Coherent prompts    |             15/15 |                         15/15 |    —    |

`run_pp_bench`, RUNS=3, median ms / tok/s:

| pp   | Wave64 (tok/s) | Wave32 (tok/s) | Δ %    |
|------|---------------:|---------------:|-------:|
|   64 |        1 709.9 |        1 665.0 | −2.6 % |
|  128 |        2 589.6 |        2 506.5 | −3.2 % |
|  256 |        3 500.4 |        3 424.8 | −2.2 % |
|  512 |        3 800.5 |        3 733.9 | −1.8 % |
| 1024 |        3 726.3 |        3 646.7 | −2.1 % |

Reading:

- **Decode is dead-flat.** The 27.5 % VALU utilisation in 12G-D is not
  hiding latent throughput VOPD can fill. The decode bottleneck is
  HBM bandwidth on the large GEMVs (gemv_gate/up at 91 % peak BW per
  12G-D), and VOPD doesn't reduce bytes read.
- **Prefill regresses 2-3 % at every pp.** Wave32 doubles the wave
  count per workgroup (one Wave64 = two Wave32) but each wave is
  half-width, so register-file footprint per wave halves while
  scheduler load doubles. For our coopmat GEMM dispatches, the
  scheduling overhead apparently outweighs the per-wave occupancy
  benefit. coopmat / WMMA itself is a matrix-core operation, not a
  VALU operation — VOPD isn't applicable to the WMMA work, only to
  the dequantise / activation-load surrounding code. That code is
  small relative to the K-loop.
- **15-prompt aggregate prefill +1.1 %** is interesting — the suite
  has many sub-64-token prompts where the coopmat M/S-tile
  selection kicks in and the GEMM is already undersaturated. Wave32
  giving more independent waves can hide latency at those tiny
  shapes. But the gain is below the noise band (the same suite
  shows ±0.5 tok/s variance from one run to the next).

## 6. Why VOPD didn't translate to wall time

Sprint 12G-D's "27.5 % VALU utilisation, 72.5 % idle" was a clue
that *looked* like VALU headroom, but the 72.5 % idle is dominated
by memory-load wait states, not unfilled VALU slots. RDNA4's
`s_waitcnt vmcnt` is what is actually idling. VOPD turns 1 VALU
instruction into 1 VOPD instruction that issues 2 ops in 1 cycle —
but if the VALU is waiting on memory anyway, you've replaced a 1-cycle
instruction with a 1-cycle instruction. Same wall time.

The places VOPD *would* help on this workload:

1. **The dequant inner loop** of `mul_mat_vec_q4_k.comp` —
   `v_dual_add_f32`, `v_dual_mul_f32` patterns there should fuse
   nicely. They probably are: the 3 546 v_dual count is real, not
   ornamental.
2. **The Q6_K bit-blend in `mul_mat_vec_q6_k.comp`** — `ql ^ qh`
   patterns, similar story.

But both of those run inside K-loops that are bandwidth-bound. The
inner-loop ALU cost was already small. Halving it doesn't shorten
the K-loop; the next memory load arrives at the same time.

This is the **same shape** as Sprint 13C's f16acc finding: an
RDNA4-supported, ACO-emitted feature that doesn't move the needle
because the underlying hardware path is already at its memory-bound
ceiling. Sprint 13B's Mesa-update finding was the precursor: when
both the implementation and llama.cpp are flat between driver
versions, the lever isn't in the driver. Now we know the lever isn't
in VOPD either.

## 7. Per-dispatch profile (skipped per brief — bench-gate not met)

Brief §3 made per-dispatch profiling conditional on Wave32 ≥ +5 %.
None of our metrics met that bar, so detailed profiling per shader
under Wave32 wasn't extracted. If a future sprint wants to revisit:

- Run `profile_positions` under both wave sizes (same as Sprint 13E
  did for NUM_ROWS).
- The shaders that *would* show a Wave32 win are the ones with the
  most VALU-per-byte: **`scalar_attn`** at decode pos=50 (1 113 µs)
  and **lm_head** (740 µs). They're not on the steady-state hot
  path post-Sprint 12, which is why the headline numbers don't move.
- coopmat-using shaders won't move regardless because the WMMA
  fragment is wave-size-agnostic.

## 8. Verdict map

| Brief bin | Decode | Prefill | Choice |
|---|:---:|:---:|---|
| +10 to +20 % | — | — | not reached |
| +3 to +10 % | — | — | not reached |
| ±2 % | **YES** (+0.1 %) | borderline (−1.8 to −3.2 %) | **deferred** |
| negative | — | mild | — |

Mapping to the brief's recommendation tiers:

- **Not v0.3 priority.** The Sprint 14-class "subgroup-arithmetic
  GEMV / required-subgroup-size pinning" infra refactor (queued
  out of Sprint 13E) is the real lever. Wave32 alone doesn't justify
  the migration cost.
- **Not negative enough to "don't use".** It works correctly,
  doesn't crash, doesn't break coherence. If a user's workload were
  unusually VALU-bound (say, a very small model with big batch sizes
  dominated by elementwise ops), Wave32 might help — but Qwen3-8B
  prefill / decode isn't that workload.
- **Keep the env-var path open.** A user who wants to test
  Wave32 on their own model + driver can do so:
  ```
  VK_ICD_FILENAMES=/path/to/mesa-26.1+/share/vulkan/icd.d/radeon_icd.x86_64.json \
  LD_LIBRARY_PATH=/path/to/mesa-26.1+/lib \
  RADV_PERFTEST=cswave32 \
    cargo run --release --example run_15prompt_bench
  ```
  No code changes required.

## 9. Sprint 13 series — closing summary

| Sprint | Lever                                  | Outcome on RDNA4 |
|--------|----------------------------------------|------------------|
| 13A    | S-tile coopmat (BM=32)                 | NEUTRAL (vs M-tile, +1.9 % within noise; +27 % vs scalar-default-off at pp=32) |
| 13B    | Mesa 26.1-rc3 driver upgrade           | NEUTRAL (within ±2.3 % at every pp, llama.cpp flat too) |
| 13C    | f16-accumulator coopmat shader         | SLIGHT NEGATIVE (−2 % at pp=512; opt-in retained) |
| 13E    | MMV_NUM_ROWS=2 GEMV pipelines          | SMALL NEGATIVE (gemv_q +21 % per dispatch; reverted) |
| **13D** | **Wave32 / VOPD**                     | **NEUTRAL on decode, mild prefill regression** |

**Pattern.** Every "port llama.cpp's config" / "use the latest
driver feature" hypothesis tested in Sprint 13 has been falsified.
The shader source and pipeline configs are now in parity with
llama.cpp where it matters. The remaining 0.10–0.15 × peak-WMMA gap
to llama.cpp at pp=512 lives in their **pipeline-creation
infrastructure**: required-subgroup-size pinning, subgroup-arithmetic
reduction paths in GEMV, multi-submit overlap, `quantize_q8_1`
fusion into the GEMM dispatch. Those are Sprint 14 / v0.3-class
infrastructure work, not single-line config flips.

## 10. Outputs

- **No runtime code changes** committed in this sprint (per brief §6).
- This report.
- `~/tmp/mesa-26.1/` already in place from Sprint 13B; rollback
  remains `rm -rf ~/tmp/mesa-26.1*`. System Mesa 26.0.6 untouched.
- Bench logs retained at `/tmp/13d_bench.txt`.

## 11. Recommendation for v0.3

**Do not migrate to Wave32.** The 3 546 VOPD instructions emitted
by ACO confirm the codegen feature is real, but the workload's
memory-bandwidth ceiling means dual-issued VALU lands in idle
cycles that were already wait-states. Any v0.3 prefill /
decode performance work should target the pipeline-creation
infrastructure instead:

1. `requireFullSubgroups` + `requiredSubgroupSize=64` plumbing in
   `ComputeKernel::from_spv_with_spec` (Sprint 13E §7).
2. The `_subgroup` and `_subgroup_no_shmem` GEMV variants
   llama.cpp ships (Sprint 13E §7).
3. Multi-submit prefill (Sprint 12B audit; ~5–10 % at pp=512
   estimated, no source-shader work).
4. `quantize_q8_1` fusion into the GEMM dispatch (Sprint 12I §6).

Wave32 / VOPD is a follow-on candidate *after* the GEMV reduction
path has been moved to subgroup arithmetic — at that point VOPD
might find more uncovered VALU cycles in a smaller-footprint
reduction. Re-test then.
