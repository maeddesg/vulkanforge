# VulkanForge v0.2.1 Sprint 11A — Q8_1 + mul_mm Deep-Dive

**Date:** 2026-04-29
**Branch:** main (HEAD = 44f329c v0.2.0)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV
**Mode:** Analysis-only sprint (no shader / runtime changes)

## TL;DR — sprint hypothesis was wrong

The brief's framing — "we use a GEMV-port (mul_mat_vec_q) for prefill;
porting `mul_mm.comp` + `quantize_q8_1.comp` would close the 0.52× gap"
— **does not match the codebase.**

VulkanForge **already** ships:
- `vk_shaders/mul_mmq.comp` — **bit-identical** to llama.cpp's `mul_mmq.comp` (311 LOC, `diff` empty).
- `vk_shaders/mul_mm.comp` — **bit-identical** to llama.cpp's `mul_mm.comp` (FP32 GEMM, fallback).
- `vk_shaders/quantize_q8_1.comp` — **bit-identical** to llama.cpp's, dispatched before every `mul_mmq` GEMM.
- `vk_shaders/mul_mmq_funcs.glsl`, `mul_mm_funcs.glsl`, `mul_mmq_shmem_types.glsl` — also bit-identical.

`forward.rs:3030-3540` already calls `run_quantize_q8_1` for all 7 prefill
GEMMs (Q/K/V/O + Gate/Up/Down). `mul_mmq` is the default for prefill;
`mul_mm` is the gated `VULKANFORGE_USE_MUL_MM=1` fallback (and is in
fact ~45 % slower at prefill than `mul_mmq`, per CHANGELOG v0.1.3).

So Q8_1 + mul_mmq is **post-fusion state for us** — the work the brief
proposes was shipped in v0.1.x and refined in Phase 7 (v0.1.3 BLOCK_SIZE
fix). What's actually missing is **two different things** that the
brief glances past:

1. **Multi-tile pipeline (S/M/L variants)** — llama.cpp ships three
   `mul_mmq` pipelines per quant type with different `(BM, BN, BK,
   BLOCK_SIZE)` tile sizes and selects at runtime based on `(M, N)`.
   We ship **one** pipeline (`BM=64, BN=64, BK=32, BLOCK_SIZE=256`),
   close to llama.cpp's *Small* tile, used **for everything**.

2. **No KHR coopmat GEMM** — the brief speculates about
   `mul_mm_cm1.comp` as the "endgame." There **is no `mul_mm_cm1.comp`
   in upstream llama.cpp.** The only coopmat GEMM upstream is
   `mul_mm_cm2.comp`, which uses both `GL_KHR_cooperative_matrix` *and*
   `GL_NV_cooperative_matrix2` (line 13 of `mul_mm_cm2.comp`) — same
   NV-only blocker that Sprint 10A hit on `flash_attn_cm2.comp`.

This sprint pivots from "port what we don't have" to "tune the pipelines
we already have like llama.cpp does." The recommended Sprint 11 work
is **multi-tile pipeline + L/M/S runtime selection** (à la
`ggml_vk_guess_matmul_pipeline`), not Q8_1 / mul_mm portage.

## Pre-check (per `feedback_sprint_hypothesis_check.md`)

The memory rule says: for "fuse X+Y" sprints, verify the codebase
isn't already in the post-fusion state before writing code. This sprint
is exactly that case — and the brief's premise was the fusion. Pre-check
saved us from a 2–4 week port of work already done.

Concrete diffs (all empty):
```
$ diff vk_shaders/mul_mmq.comp        ~/tmp/llama.cpp/.../mul_mmq.comp
$ diff vk_shaders/mul_mm.comp         ~/tmp/llama.cpp/.../mul_mm.comp
$ diff vk_shaders/mul_mm_funcs.glsl   ~/tmp/llama.cpp/.../mul_mm_funcs.glsl
$ diff vk_shaders/mul_mmq_funcs.glsl  ~/tmp/llama.cpp/.../mul_mmq_funcs.glsl
$ diff vk_shaders/mul_mmq_shmem_types.glsl ~/tmp/llama.cpp/.../mul_mmq_shmem_types.glsl
$ diff vk_shaders/quantize_q8_1.comp  ~/tmp/llama.cpp/.../quantize_q8_1.comp
$ diff vk_shaders/mul_mat_vec_q4_k.comp ~/tmp/llama.cpp/.../mul_mat_vec_q4_k.comp
```

## Q8_1 format reference (for future readers)

```c
typedef struct {
    ggml_half d;       // 2 B — scale (delta)
    ggml_half s;       // 2 B — sum: Σ(qs[i]) × d, precomputed
    int8_t qs[QK8_1];  // 32 B — 32 quantised values
} block_q8_1;          // 36 B total = 1.125 B/element
```

Confirmed in `~/tmp/llama.cpp/ggml/src/ggml-common.h`. The precomputed
sum `s` is consumed by `mul_mmq_funcs.glsl` for Q4_K × Q8_1 dot-products
and lets the inner loop skip a separate `Σ qa[i]` pass. We use this
already (the bit-identical `mul_mmq_funcs.glsl`).

## Bandwidth note (correcting the brief)

The brief's BW calculation says Q8_1 saves 1.51× over FP32 activations
and that "alone explains +50 % speedup." But this comparison applies
between *us with FP32 activations* and *llama.cpp with Q8_1 activations*.
**We're already on Q8_1.** The activation BW for both us and llama.cpp
at pp=512 / Q-proj is the same 11.8 MB/GEMM — no speedup is sitting on
the table from switching activation format.

## Where the gap actually is — single-tile vs L/M/S

### Our pipeline (one tile size for all M, N)

`src/backend/vulkan/pipeline_registry.rs:222-279` (MulMmqQ4K/Q6K branch):

```
BLOCK_SIZE = 256
BM         = 64
BN         = 64
BK         = 32
WM = WN    = 32
TM         = 2
TN         = 4
WMITER     = 2
WARP       = 64 (Wave64)
NUM_WARPS  = 4 (= 256/64)
```

This is the only `mul_mmq` pipeline we ship. Used for every prefill
GEMM regardless of `(M, N)`. The same shape (`BM=64 BN=64`) is used for
the FP32 fallback (`MulMmQ4K/Q6K`) at lines 280-323.

### llama.cpp's pipelines (three tiles + aligned variants per quant)

From `~/tmp/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` line 3294-3296:

```
l_warptile_mmq_k = { BLOCK_SIZE=256, BM=128, BN=256, BK=64, ?=1 };
m_warptile_mmq_k = { BLOCK_SIZE=256, BM=128, BN=128, BK=64, ?=1 };
s_warptile_mmq_k = { BLOCK_SIZE=256, BM=32,  BN=64,  BK=128, ?=0 };
```

(The `_k` suffix is the K-quants variant — Q4_K, Q5_K, Q6_K. There are
parallel `_mmq` and `_mmq_int` tables for non-K-quants and integer
acc. RDNA4 with KHR coopmat falls onto the `_k` table here because it
doesn't have coopmat2.)

Plus `aligned` and `unaligned` variants of each → **six pipelines per
quant type**, vs our one.

### Selection logic (no-coopmat2 branch at lines 7171-7177)

```cpp
if (mul_mat_s[type] && (m <= 32 || n <= 32))      return mmp->s;  // small
if (mul_mat_m[type] && (m <= 64 || n <= 64))      return mmp->m;  // medium
return mmp->l;                                                     // large
```

Plus an `aligned` flag passed alongside that picks the aligned variant
when the K dimension is a multiple of `align`.

### What llama.cpp would pick for our prefill GEMMs

Qwen3-8B-Q4_K_M, pp=512, Q-projection: `M=512, N=4096, K=4096`.
Both M and N are > 64 → llama.cpp picks **L** = `BM=128 BN=256 BK=64`.

We use `BM=64 BN=64 BK=32`. Per-WG work: `BM × BN × BK = 64·64·32 =
131 072 ops` vs llama.cpp's L `128·256·64 = 2 097 152 ops` — **16×
more work per workgroup**. Output tile area: 64² = 4 K cells vs
128·256 = 32 K cells (8×). LDS reuse and register reuse scale roughly
linearly with tile area; a tile that is 8× larger amortises the K-loop
prologue/epilogue 8× better.

For pp ≤ 32 (or N ≤ 32), llama.cpp falls to **S** = `BM=32 BN=64
BK=128` — more cells along K, fewer along M/N. We use the same
`BM=64 BN=64 BK=32` even at decode-adjacent shapes; we'd actually be
*overshooting* the M dimension there. (Decode itself is GEMV via
`mul_mat_vec_q4_k.comp`, not GEMM; this is about prefill at small M.)

### Why this matters for the 0.52× gap

The gap is dominated by **GEMM utilisation, not GEMM correctness**:

- Both implementations execute the same SPV instructions per cell (the
  shaders are bit-identical).
- We compute `M·N / (64·64)` workgroups; llama.cpp computes `M·N /
  (128·256)` workgroups for L. At M=512 N=4096 that's 512 vs 64 —
  we're dispatching **8× more workgroups**, each doing 8× less work.
- The inner-loop count along K is the same (`K / BK` × inner pattern),
  but llama.cpp's BK=64 vs our BK=32 means our per-WG K-prologue runs
  twice as often per output cell.
- Smaller WGs hit Wave64 occupancy headroom less efficiently — RDNA4
  schedules well at 4 active waves/CU, and our `BLOCK_SIZE=256` already
  hits that, but the per-WG memory traffic is now the binding cost,
  not occupancy.

I have not measured this — this is an analytical case, not a profiled
one. Sprint 11B should profile per-shader timing first to confirm GEMM
is actually the dominant share of prefill before chasing the L/M/S
work.

## What about coopmat GEMM?

`vk_shaders/mul_mm_cm2.comp` lines 12-13:

```glsl
#extension GL_KHR_cooperative_matrix    : enable
#extension GL_NV_cooperative_matrix2    : enable
```

Plus uses `coopMatLoadTensorNV`, `coopMatPerElementNV`,
`coopMatReduceNV`, `tensorLayoutNV` — all NV-only intrinsics, same set
that blocked Sprint 10A's flash_attn_cm2.comp port.

**There is no `mul_mm_cm1.comp` in upstream llama.cpp.** The brief is
wrong on this point (it explicitly references `mul_mm_cm1.comp`).
KHR-portable coopmat GEMM is something *we'd have to write ourselves*
— the same `Eigenbau` path Sprint 10C took for attention.

For RDNA4 (KHR coopmat rev2 only) the realistic options are:

1. **No coopmat GEMM, focus on tile-size tuning** (recommended for
   Sprint 11B). Closes the 0.52× gap by chasing llama.cpp's scalar
   optimisations we haven't shipped yet.
2. **Eigenbau KHR coopmat GEMM** (Sprint 10C-style, but for GEMM).
   Higher ceiling, much higher engineering risk. Q4_K weight dequant
   into a coopmat A-fragment is non-trivial; Q8_1 → `coopMatLoad`
   needs a dequant-to-FP16 staging step (no native int8 fragment in
   KHR rev2).

## Inventory: what we ship vs llama.cpp upstream

| File                            | LOC ours | LOC ll.cpp | Diff | Used? |
|---------------------------------|----------|-----------|------|-------|
| `quantize_q8_1.comp`            | 127      | 127       | empty | ✅ all 7 prefill GEMMs (`forward.rs:3030+`) |
| `mul_mmq.comp` (Q8_1 GEMM)      | 311      | 311       | empty | ✅ default prefill |
| `mul_mmq_funcs.glsl`            | 454      | 454       | empty | (incl) |
| `mul_mmq_shmem_types.glsl`      | 30+      | 30+       | empty | (incl) |
| `mul_mm.comp` (FP32 GEMM)       | 464      | 464       | empty | ⚠ gated `VULKANFORGE_USE_MUL_MM=1` (slower) |
| `mul_mm_funcs.glsl`             | 598      | 598       | empty | (incl) |
| `mul_mat_vec_q4_k.comp` (GEMV)  | 134      | 134       | empty | ✅ decode hot path |
| `mul_mm_cm2.comp` (NV-coopmat)  | —        | 624       | (n/a) | ❌ NV-only |
| `mul_mm_cm1.comp` (KHR-coopmat) | —        | **does not exist** | (n/a) | ❌ |

Pipeline configurations:

| Aspect           | VulkanForge                         | llama.cpp upstream                              |
|------------------|-------------------------------------|-------------------------------------------------|
| Q8_1 quantize    | yes, before each `mul_mmq`           | yes, before each `mul_mmq`                      |
| `mul_mmq` shader | bit-identical                        | (origin)                                        |
| Tile pipelines   | **1** (`BM=64, BN=64, BK=32`)        | **6 per quant**: {S,M,L} × {aligned,unaligned}  |
| Aligned variants | yes for `mul_mm` (`MulMmQ4KAligned`); not for `mul_mmq` | yes (separate aligned/unaligned per S/M/L)     |
| Runtime select   | none (always the one pipeline)       | `ggml_vk_guess_matmul_pipeline` (M/N thresholds)|
| coopmat GEMM     | none                                 | cm2 (NV-only)                                   |

## Sprint 11 phase plan — revised

The brief's plan (Phases B–F as Q8_1 / mul_mm port + integration) is
**moot** — that work is done. Replacement plan, ordered:

### Phase 11B — Profile prefill before tuning

Before chasing tile sizes, confirm where the time actually goes. Run
`forward_token_profile_layers` (already exists, `forward.rs:614`) on
a `pp=512` and `pp=2048` warmed-up prefill batch and report per-shader
share. Hypothesis: GEMM ≥ 60 % of forward-pass wall time on prefill;
if it's lower, the tile-size sprint pays back less than expected.

Cost: ~2 hours scripting + bench. Output: `results/v021_sprint11b_profile.md`.

Gate: Phase 11C only proceeds if GEMM ≥ 50 % of pp=512 wall time.

### Phase 11C — Add `MulMmqQ{4,6}K_M` and `_L` pipelines

Use llama.cpp's `m_warptile_mmq_k` and `l_warptile_mmq_k` values:

```
M: BLOCK_SIZE=256, BM=128, BN=128, BK=64, WMITER=1
L: BLOCK_SIZE=256, BM=128, BN=256, BK=64, WMITER=1
```

(`_k` variants because Q4_K and Q6_K are K-quants. Sprint 11C should
also re-derive the `WM/WN/TM/TN/WMITER` values from the
`s_warptile_mmq_k` constants in `ggml-vulkan.cpp:3294-3296` rather than
copying the Phase-7 sweep blindly.)

These are spec-constant-only changes — the bit-identical
`mul_mmq.comp` already supports them. **No SPV rebuild needed.** Just
new entries in `pipeline_registry.rs` for the new ShaderId variants.

Cost: ~1 day. Risk: low (pure spec-constant change, parity should hold
trivially because the shader logic is unchanged).

### Phase 11D — Runtime tile selector

Port `ggml_vk_guess_matmul_pipeline` (lines 7141-7177). For each
prefill `mul_mmq` site, pick the pipeline based on `(M, N)`:

```rust
fn pick_mmq_tile(m: u32, n: u32) -> ShaderId {
    if m <= 32 || n <= 32 { ShaderId::MulMmqQ4K }       // S = today's default
    if m <= 64 || n <= 64 { ShaderId::MulMmqQ4K_M }
    return                  ShaderId::MulMmqQ4K_L;
}
```

Wire into `forward.rs::run_mul_mmq` (or wherever the dispatch lives).

Cost: ~half a day.

### Phase 11E — Aligned variants for `mul_mmq`

llama.cpp ships aligned variants per tile. We ship aligned only for
`mul_mm`. Add aligned variants of the new `mul_mmq` tiles. Aligned
shader is the same `mul_mmq.comp`, just with a build-time `ALIGNED`
define toggling the K-loop prologue (the upstream shader already
supports this).

Cost: ~half a day.

### Phase 11F — Sweep + default

Re-run the pp-sweep (`examples/run_pp_bench`). Expected: at pp ≥ 64
the L tile should win; at pp ≤ 32 the S tile (today's default).
Confirm parity (167/167 + Alice 3/3) with the new selector.

Cost: ~1 day.

### Phase 11G (optional, far horizon) — Eigenbau KHR coopmat GEMM

If 11B-F doesn't close enough of the gap: KHR coopmat GEMM from
scratch. Patterns from `flash_attn_coopmat.comp` (Sprint 10C) +
llama.cpp's `mul_mm_cm2.comp` minus the NV intrinsics. **Multi-week
work**, similar shape to Sprint 10C but more LDS pressure and a
trickier weight-dequant step.

Defer until 11B-F numbers are in.

## Estimated end-state

- 11C-F is **two engineering days**, not "2-4 weeks" — it's
  spec-constants + dispatch logic, not new shaders.
- Expected speedup: hard to put a number on without 11B's profile.
  llama.cpp's pp=512 is **1.91× ours** — closing even half the gap
  with a no-shader-change L/M/S setup would be a strong outcome.
- 11G coopmat GEMM remains the long-term ceiling and is parked.

## Files touched

- `results/v021_sprint11a_gemm_deepdive.md` — this report.

No source / shader changes.

## Take-aways

1. **Pre-check matters.** The codebase was already past the brief's
   starting point. The hypothesis-check feedback memory paid for itself
   on this sprint.
2. **Bit-identical shader ≠ bit-identical performance.** Same shader,
   six pipelines vs one pipeline → 8× WG-volume difference at L tile.
   The optimisation gap is configuration, not code.
3. **No KHR coopmat GEMM upstream.** Anyone aiming there does it
   themselves, like Sprint 10C did for attention.
4. **Sprint 11 is small, not big.** L/M/S + selector is a couple of
   days of work, not a multi-week port — *if* 11B's profile confirms
   GEMM is the bottleneck.
