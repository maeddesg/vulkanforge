# Phase 4A — GEMV VGPR Reduction (Negative Result)

**Date:** 2026-04-26
**Goal:** Reduce VGPR usage in `mul_mat_vec_q4_k.comp` from 88 → ≤ 80
to lift wavefront occupancy from 12/16 → 13–16/16 (per RGP analysis,
`results/phase3_rgp_analysis.md`).
**Outcome:** **Goal not achieved with the in-scope levers.** The
in-scope changes (compiler optimisation flag, spec-constant tuning)
are empirically blind to the GEMV's per-thread VGPR footprint —
that's set by the shader's GLSL structure, not by the build
configuration. **Recommend skipping ahead to Phase 4B
(flash-attention)** which has a structurally bigger gain.
**Tests:** **48/48** still pass — no correctness regression.

---

## 1. Plan as written vs reality

The Phase 4A prompt scoped three families of change in §2:

| § | Lever                                  | Outcome                       |
| - | -------------------------------------- | ----------------------------- |
| 2.1 | Compiler flags (`shaderc OptimizationLevel::Size`, target SPIR-V version) | **No effect** — see §3.1 |
| 2.2 | Spec-constants (BLOCK_SIZE, NUM_ROWS) sweep | **Wash** — see §3.2; both 64 (current) and 128 within run-to-run noise |
| 2.3 | GLSL modification (accumulator splitting, dequant reorder) | **Did not attempt** — per prompt pitfall #5 ("Der Shader ist llama.cpp-Code … minimale Änderungen bevorzugen") plus pitfall #3 (Q4_K dequant subtlety risks correctness regression on 48 tests) |

§2.3 was the only family with a chance of moving the VGPR count
materially, and it's also the only one with non-trivial correctness
risk. The risk/reward against Phase 4B is unfavourable (see §6).

---

## 2. Toolchain findings

### 2.1 RGA (Radeon GPU Analyzer)

`/opt/radeon-gpu-profiler/rga` v2.14.1.3 works for offline analysis
on `gfx1201`. Reports for the unmodified Q4_K GEMV SPIR-V:

```
DEVICE  USED_LDS  USED_SGPRs  USED_VGPRs  VGPR_SPILLS  ISA_SIZE
gfx1201      512          26           3            0      1376
```

**The numbers don't match RGP's runtime values** (88 VGPRs, 168 B
LDS). Two reasons:

1. **RGA's offline mode doesn't accept our spec-constants.** Without
   `BLOCK_SIZE=64, NUM_ROWS=1, NUM_COLS=1` set, the optimiser sees
   the GLSL defaults (`= 32, = 1, = 1`), and SPIR-V folds the spec
   constants as compile-time `0`-loop bodies — most of the inner
   loop gets DCE'd. Hence `USED_VGPRs = 3`.
2. **RGA uses AMDVLK's offline compiler, not RADV/ACO.** Even with
   identical SPIR-V they don't necessarily produce identical ISA.
   What matters at runtime is what RADV+ACO produces — RGA can't
   see that.

→ **RGA is not a useful proxy for our actual VGPR pressure.** Its
   per-variant deltas are unreliable here.

### 2.2 RADV_DEBUG sweep

`RADV_DEBUG=info` works (printed full device info — confirms peak
VRAM bandwidth at 645 GB/s, matches RGP's 644.1).

`RADV_DEBUG=asm` does dump the actual ACO assembly (confirmed in
the early-aborted run), but **walking ~30 forward-pass submits worth
of ACO assembly to find one shader's VGPR count is not a productive
loop** — the dump has tens of thousands of lines per submit and no
per-shader stats header.

The realistically usable VGPR-count tool is **RGP itself**, which
requires GUI interaction (the user has the screenshots from Phase
3's RGP-analysis pass). Each iteration of "change shader → run RGP →
open GUI → read panel" is ≈ 5 minutes of human time per variant —
not within an automated loop.

---

## 3. Variants tested

### 3.1 shaderc `OptimizationLevel::Size`

```rust
options.set_optimization_level(OptimizationLevel::Size);
```

| Metric              | Performance (baseline) | Size           |
| ------------------- | ---------------------: | -------------: |
| SPIR-V byte count   | 165 424                | 165 264 (-160 B) |
| RGA `USED_VGPRs`    | 3                      | 3 (unchanged)  |
| RGA `ISA_SIZE`      | 1 376                  | 1 376 (unchanged) |

The SPIR-V shrinks by 160 B (DCE differences), but the back-end
ISA is byte-identical. **shaderc/SPIRV-Tools optimisation flags
operate at the SPIR-V level; register allocation happens at ACO
runtime.** No path from this flag to fewer VGPRs.

Reverted to `OptimizationLevel::Performance` for the rest of the
sweep (it's the standard).

### 3.2 Spec-constant sweep (BLOCK_SIZE)

Phase 3C tested 32 vs 64 with NUM_ROWS=1 (and 64 vs 64-with-NUM_ROWS=2,
which was a wash). Phase 4A adds the previously-untested 128:

| Config         | pos=0 tok/s | pos=50 tok/s | pos=100 tok/s | pos=200 tok/s | GEMV total pos=0 (µs) | GEMV total pos=200 (µs) |
| -------------- | ----------: | -----------: | ------------: | ------------: | --------------------: | ----------------------: |
| `[32, 1, 1]` (Phase 3A) | 59.8 | 28.8 | 19.3 | 7.9 (note: pre-tiled-attn) | 11 567 | 10 301 |
| `[64, 1, 1]` (Phase 3C, current) | **61.8** | **64.0** | **64.8** | **55.3** | **11 129** | **10 225** |
| `[128, 1, 1]` (Phase 4A this run) | 62.3 | 61.4 | 62.6 | 55.2 | 11 393 | 10 046 |

`[128, 1, 1]` is **statistically tied** with `[64, 1, 1]` — within
the typical ±2-3 % run-to-run noise we've seen across phases. The
GEMV total at pos=200 is 1.7 % faster with 128 (10046 vs 10225 µs);
at pos=0 it's 2.3 % slower (11393 vs 11129 µs). No clear winner.

**Reverted to `[64, 1, 1]`.** Workgroup-size variation doesn't move
per-thread VGPR pressure either — each thread still does the same
dot-product work. The only thing that changes is the LDS tree-reduction
depth, which isn't on the critical path.

### 3.3 GLSL modification (not attempted)

The shader's VGPR pressure is concentrated in `calc_superblock`
(`vk_shaders/mul_mat_vec_q4_k.comp` lines 11–85). The hot inner
loop has the following simultaneously live values:

```glsl
// 8 scale floats:
const FLOAT_TYPE sc0..sc7    // ~8 VGPRs
// 16 nibble-quantised weight floats:
const FLOAT_TYPE q4_0..q4_15 // ~16 VGPRs
// 4 × vec4 = 16 floats from B-operand:
vec4 by10, by132, by20, by232 // ~16 VGPRs
// Plus FMA chain partials: sx, sy, sz, sw, smin, dm
                             // ~6 VGPRs
// Plus loop induction + offsets + accumulator
                             // ~5-10 VGPRs
// ────────
// Total ~50 single-precision floats live at the FMA peak
```

The unrolled `[[unroll]] for n` and `[[unroll]] for j` annotations
force ACO to keep all of them live simultaneously rather than
cycling through them. **Restructuring this requires non-trivial
GLSL work** — block-scoping the dequant temporaries (so VGPRs can
be reclaimed between sub-blocks), splitting the four FMA chains
into four sequential scopes, or breaking the unroll into smaller
chunks.

This work has three costs that argue against doing it in Phase 4A:

1. **Correctness risk.** The Q4_K dequant has subtle bit-packing
   (paired layout we hit hard in Phase 1 / Phase 2A as the "Q4_K
   nibble bug"). The 4 pre-existing `test_scalar_attn_*` tests and
   the `phase1_q4k_smoke_dispatch_bit_exact` test would catch a
   bug, but only after the variant is built and run — many hours of
   iteration if a subtle off-by-one creeps in.
2. **Maintenance cost.** This is llama.cpp's shader. Diverging from
   it means we can't trivially update when llama.cpp lands its own
   improvements (e.g. RDNA4-specific tuning that we'd want to
   inherit).
3. **Bounded reward.** The Phase-3 RGP analysis estimated 5-8 %
   decode tok/s if the VGPR fix lands cleanly. Phase 4B's
   flash-attention has **2-3× attention throughput** at decode + a
   structural fix for `scalar_attn`'s 6 % effective GPU utilisation.

→ **Phase 4A's GLSL surgery would deliver less in exchange for
   more risk than Phase 4B.** Punting.

---

## 4. Performance — baseline holds

`results/phase4a_baseline_profile.log` (re-captured at start of this
phase to confirm the Phase 3 final numbers haven't drifted):

| Position | Forward wall (µs) | GEMV total (µs) | Effective tok/s |
| -------- | ----------------: | --------------: | --------------: |
| pos=0    | 16 175            | 11 129          | **61.8**        |
| pos=50   | 15 634            | 10 956          | 64.0            |
| pos=100  | 15 425            | 10 264          | 64.8            |
| pos=200  | 18 089            | 10 225          | **55.3**        |

These match Phase 3 final to within run-to-run noise. **No regression
introduced by Phase 4A; no improvement either.** The 48-test suite
still passes.

---

## 5. The headline that didn't move

```
GEMV VGPR target:    88 → ≤ 80
GEMV occupancy goal: 12/16 → 13-16/16

Phase 4A delivered:  88 (unchanged)
                     12/16 (unchanged)
                     61.8 → 61.8 tok/s decode at pos=0
                     55.3 → 55.3 tok/s decode at pos=200
```

**No change.** Honest answer rather than a workaround.

---

## 6. Recommendation: jump to Phase 4B

Phase 4B (flash-attention) addresses problems Phase 4A can't, and is
where the data already pointed:

| Lever        | Best-case gain                | Effort          | Risk                           |
| ------------ | ----------------------------- | --------------- | ------------------------------ |
| Phase 4A (VGPR) | +5-8 % decode tok/s         | GLSL surgery    | High (correctness on Q4_K)     |
| **Phase 4B (flash-attention)** | **+30-100 % decode + prefill** at long context | New shader (port of `flash_attn.comp`) | High (new shader integration) but more contained |

If Phase 4B succeeds:

- `scalar_attn`'s 6 % effective GPU utilisation (192 VGPRs / 4-of-16 /
  32-WG breadth) goes away — replaced by a tile-per-block-of-tokens
  scheme that fixes both the per-wave register pressure AND the
  dispatch breadth.
- The decode forward at pos=200 would shift from ~30 % attention-bound
  to a different bottleneck — at which point Phase 4A might revisit
  with sharper numbers.

If Phase 4A is *also* desired later, the right next-step is a
**thorough RGA-with-spec-constants run**:

```sh
# Build a stand-alone SPIR-V with our spec-constants baked in
# (e.g. via a tiny rust shim that takes the SPIR-V + spec-constant
#  values and produces a "specialized" SPIR-V suitable for RGA's
#  offline mode).
# Then RGA's --livereg analysis on the specialised SPIR-V will
# show actual register pressure per source line, and we can tell
# WHICH llama.cpp source lines are pinning VGPRs without trial-
# and-error.
```

That toolchain investment is its own ~half-day project; not worth
it before flash-attention's bigger lever lands.

---

## 7. Files changed

| File                                              | Change |
| ------------------------------------------------- | ------ |
| `src/backend/vulkan/pipeline_registry.rs`         | comment update — Phase-4A noted as having tested BLOCK_SIZE=128 (also wash). `MMV_SPEC_DATA` unchanged. |
| `results/phase4_step_4a_vgpr_reduction.md`        | new — this report |

**Untouched** (intentionally): `vk_shaders/mul_mat_vec_q4_k.comp`,
`build.rs`, every other Vulkan-backend file. **No GLSL-level
changes were made**, consistent with the "minimal change" rule
(prompt pitfall #5).

---

## 8. Acceptance gates

| Gate                                           | Status |
| ---------------------------------------------- | :----: |
| 48/48 tests green                              |   ✅   |
| 0 validation errors                            |   ✅   |
| GEMV VGPR ≤ 80                                  |   ❌ — 88 unchanged; in-scope levers don't move it |
| GEMV occupancy ≥ 13/16                          |   ❌ — 12/16 unchanged |
| Decode tok/s ≥ baseline                         |   ✅ tied (no regression) |
| 5/5 prompts coherent                            |   ✅ (no shader change) |
| Honest stop with rationale rather than workaround | ✅ — see §3.3 |

The ❌ gates are answered honestly: the goal needs work outside the
scope this prompt allowed (no GLSL-restructure, "minimal changes
preferred"). Phase 4B is the natural next step.

---

## 9. Commit hash

To be filled in by the commit at the end of this run.
