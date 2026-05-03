# Sprint 19B — Compute-Graph Plan (analysis only, no code)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 19A, commit `1aebccf`)
**Goal:** Decide whether to refactor VulkanForge onto a compute-graph
architecture and what the smallest profitable slice looks like. Pure
analysis — no code in this sprint.

## TL;DR

The sprint brief assumed three things that turn out to be **wrong** when
you actually read llama.cpp's `ggml-vulkan.cpp`:

1. **"Buffer aliasing is the largest lever."** False. llama.cpp does
   *not* have a general aliasing pool. It has 5 watermark-resized named
   scratch buffers (`prealloc_x`, `prealloc_y`, `prealloc_split_k`,
   `prealloc_add_rms_partials`, `sync_staging`). No lifetime analysis,
   no alias graph. VulkanForge's ~30 named buffers cost VRAM, not perf.
2. **"VulkanForge needs a graph for fusion."** Partially false. VF
   already ships fused kernels (`SwiGLU`, `MultiAddRms`, `RmsNormMulRope`,
   cross-layer `multi_add_rms` from Sprint 9b.2). The fusion delta on
   *Qwen3-8B* (no MoE, no bias) is narrow — see §3.2.
3. **"Big-bang graph refactor delivers ~10%."** Risky to guarantee.
   llama.cpp's main *measurable* prefill levers are **multi-submit
   pacing** (`~3-8 vkQueueSubmit` per prefill, every 100 nodes / 100 MB
   matmul) and **graph-window reordering** (`NUM_TO_CHECK = 20` sliding
   window). Both can be added to VF's existing dispatcher without a
   full graph rewrite.

**Recommendation:** Option **D** (new) — *extract the two empirically
profitable patterns from llama.cpp into VF's existing dispatcher; do
NOT build a generic graph IR.* See §6.

## 1. llama.cpp's actual Vulkan architecture

Reference file: `~/tmp/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
(16 944 LOC). Key entry points:

| Concern | Function | File:line |
|---|---|---|
| Per-forward dispatch loop | `ggml_backend_vk_graph_compute` | `:14436` |
| Per-node dispatch emit | `ggml_vk_build_graph` | `:12901` |
| Pre-traversal node reorder | `ggml_vk_graph_optimize` | `:14816` |
| Barrier emit (graph-aware) | `ggml_vk_sync_buffers` | `:2852` |
| Scratch buffer fields | `prealloc_*` struct | `:1890-1912` |
| Fusion dispatch table | inline match in `graph_compute` | `:14537-14640` |

### 1.1 Graph traversal

llama.cpp's `ggml_cgraph` is an array of `ggml_tensor*` nodes (each
node carries op type + src refs). `ggml_backend_vk_graph_compute`
walks `nodes[]` linearly with `i` advanced by `1 + num_additional_fused_ops`
per step. A separate **graph-rewriter** (`ggml_vk_graph_optimize`)
runs *before* this loop and reorders nodes within sliding 20-node
windows to interleave independent work — preserving fusion patterns
that the dispatcher's matcher will later recognize.

### 1.2 Buffer pool — five named, monotonically-grown scratches

There is no aliasing analysis. The struct at `:1890`:

```cpp
size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k,
       prealloc_size_add_rms_partials, prealloc_size_add_rms_partials_offset;
vk_buffer prealloc_x, prealloc_y, prealloc_split_k,
          prealloc_add_rms_partials, sync_staging;
```

Allocation strategy (`:12862-12894`): if `requested > current_size`
then destroy + reallocate. There's a one-entry "last-tensor /
last-pipeline" cache for `prealloc_y` (`:1903-1904`) so a second
dispatch consuming the same dequantized activations skips the
re-dequant — the only "lifetime-aware" optimization. Persistent
buffers (model weights, KV cache, activation tensors) are owned by
ggml's `ggml_backend_buffer` mechanism and are not scratch.

For an 8B Qwen3-like model: **~5 distinct scratch buffers + per-tensor
persistent buffers** — directly comparable to VF's split between the
"forward scratch" pile and the per-tensor weights.

### 1.3 Multi-submit (the largest measurable prefill lever)

`:14507-14755` — each forward-pass is split into multiple submits to
overlap CPU command-buffer recording with GPU execution:

```cpp
int nodes_per_submit = 100;
uint64_t mul_mat_bytes_per_submit =
    std::min(uint64_t(100*1000*1000), ctx->last_total_mul_mat_bytes / 40u);
```

Submit fires when ≥100 nodes accumulated, OR ≥100 MB of matmul weights
touched, OR last node, OR <20% of nodes remaining. First three submits
of a large prefill use 2× the byte threshold (`:14751-14753`) for
warmup.

Empirical sizing for an 8B prefill: **~3-8 submits, one CB each.**
Decode (one token) typically fits in **1 submit**. This is one of two
places where llama.cpp's "graph" actually *changes wall-clock time* on
prefill vs a hand-recorded single-CB approach.

### 1.4 Barrier strategy — graph-aware decision, coarse barrier

The dispatcher tracks `unsynced_nodes_written` and `unsynced_nodes_read`
lists (`:12974-13020`) and emits a barrier only when the upcoming
node's src/dst memory-overlaps an unsynced earlier write
(`overlaps_unsynced` at `:12948-12969`). When emitted, the barrier
itself is a single coarse global memory barrier
(`ShaderRead | ShaderWrite | TransferRead | TransferWrite` on both
sides). So the **lever is the elision decision**, not finer-grained
barriers.

VulkanForge already does call-site barrier elision (Sprint 12D
report — measured 0% on decode). A graph-aware version on prefill is
not obviously different from what we already do.

### 1.5 Fusion patterns

llama.cpp matches and dispatches these patterns directly in
`ggml_backend_vk_graph_compute` (`:14537-14640`):

| Pattern | Applies to Qwen3-8B prefill? | VF status |
|---|---|---|
| `MULTI_ADD` (variable-arity) | yes (residual chain) | **already** (`MultiAddRms`, Sprint 9b) |
| `MULTI_ADD_RMS` | yes | **already** (cross-layer fusion, Sprint 9b.2) |
| `MUL_MAT_ADD` (matmul + bias) | **no** (Qwen3 has no bias) | n/a |
| `MUL_MAT_ADD_ADD` | no | n/a |
| `MUL_MAT_ID_*` (MoE) | no (Qwen3-8B is dense) | n/a |
| `RMS_NORM_MUL` | yes | **already** (`RmsNormMulRope` covers this) |
| `RMS_NORM_MUL_ROPE` | yes | **already** (Sprint 9c.5) |
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` | yes (KV-cache writeback) | **partial** — VF separates the SET_ROWS step |
| `ROPE_VIEW_SET_ROWS` | yes (KV-cache writeback) | partial — VF has `kv_copy_fp16` / `kv_store_fp8` as a separate dispatch |
| `TOPK_MOE_*` | no | n/a |

Fusion delta on Qwen3-8B is **2 patterns**: `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS`
(merge KV-cache writeback into the RoPE pass) and `ROPE_VIEW_SET_ROWS`.
Both save 1 dispatch + 1 barrier per layer per K/V (so up to 4 dispatch
+ 4 barriers per layer for prefill). Worth porting — but it is **two
fused shaders + two routing branches**, not a graph IR.

### 1.6 Graph reordering

`ggml_vk_graph_optimize` (`:14816-15035+`) greedily pulls forward
independent nodes within a 20-node window, while preserving fusion
patterns. View nodes get a second pass. The goal is to interleave
work that would otherwise stall.

VulkanForge's hand-rolled prefill dispatch already executes ops in a
*near-optimal* fixed order (the same llama.cpp converges to). The
window reorderer is mostly load-bearing for arbitrary user graphs;
for a fixed transformer block it's a 0-1% lever.

### 1.7 Decode vs prefill

llama.cpp uses **one unified path**. The branch falls out of pipeline
selection inside the matmul op (`mul_mat_vec_*` for N=1 vs
cooperative-matrix MMQ for large N at `:7563`, `:701-708`) and the
submit heuristic firing fewer times for short graphs. There is no
`if (decode) { … } else { … }` at the dispatch level.

## 2. VulkanForge's current dispatch surface

Quick measurements over `src/backend/vulkan/forward.rs` (4 398 LOC):

| Metric | Count | Note |
|---|---|---|
| `cmd_dispatch` call sites | 18 | each wrapped in a `run_*` helper called per layer |
| `compute_barrier` call sites | 42 | conservatively per-`run_*`-pair |
| Persistent decode buffers | 17 | `scratch_a`, `q_buf`, `fa_scratch_*`, etc. |
| Prefill-batch buffers | 12 | `batch_*` family |
| Total persistent SSBOs | ~30 | none aliased |
| Dispatch entry points | 3 | `dispatch_layer`, `dispatch_layer_batch`, `dispatch_final` |
| `Forward::new` buffer allocations | 13 | each grows a `GpuBuffer` |

Per-layer dispatches in `dispatch_layer_batch` (rough count from the
function body): RMSNorm seed (once-per-prefill), Q/K/V GEMMs (3),
optional Q8_1 quantize (1, only on Mmq), Q/K-norm + RoPE (1 fused),
KV-store (1), batched flash-attention (1), attn_output GEMM (1),
multi_add_rms (1, fused with next-layer norm seed), gate/up/down GEMMs
(3), SwiGLU (1, fused). **≈ 13-14 dispatches per layer × 36 layers
≈ 470-500 dispatches per prefill.** Same order as llama.cpp's ~250-350
post-fusion count — VF is already fusion-dense for Qwen3-8B.

## 3. Gap analysis (corrected)

### 3.1 Lever ranking, vs original brief

| Lever | Brief said | Reality | Worth in v0.3.4? |
|---|---|---|---|
| Buffer aliasing | "biggest lever" | **NON-LEVER** — llama.cpp doesn't do it; VF's 30 buffers are a VRAM line item, not a perf line item | **Skip** |
| Op fusion (broad graph IR) | "+2-3% prefill" | VF already covers ≥80% of llama.cpp's fusions for Qwen3-8B. Delta is two specific patterns (KV-write fusion). | Targeted — see §4.B |
| Barrier elimination | "+0-2% prefill" | Sprint 12D measured 0% on decode; graph-aware variant is same idea, marginal | **Skip / merge into §4.D** |
| Multi-submit pacing | "+? % pp ≤ 128" | **biggest under-tapped lever**: llama.cpp does this and ships 3-8 submits for an 8B prefill | **Yes — see §4.A** |
| Dispatch scheduling (reorder) | "+0-1%" | llama.cpp does it but for fixed transformer prefill it's marginal | **Skip** |
| Fusion (RMS+MUL etc.) | "+0-1%" | already done | **N/A** |

### 3.2 Where the residual ~10% gap to llama.cpp lives

After Sprint 19A:

* Q3_K_M pp=512: 3536 vs 3844 → **0.92×** (gap = 308 tok/s = 8%)
* Q4_K_M pp=512: 3865 vs 4314 → **0.90×** (gap = 449 tok/s = 10%)
* Llama-3.1 pp=512: 3990 vs ~4350 (est.) → ~0.92×

For an 8B prefill at pp=512 = ~143 ms, an 8% gap is **~11 ms = ~32 µs/layer**
across 36 layers. That budget is consistent with **2 saved dispatches +
2 saved barriers per layer** at RDNA4 dispatch overhead (~8-12 µs each
on RADV) — or the multi-submit overlap savings. It's *not* consistent
with a "graph framework would shave 10% by inspection" thesis.

### 3.3 Where we're worst

The pp ≤ 128 regime is at **0.71-0.74× llama.cpp** (Sprint 18 comprehensive
benchmark). At those sizes the dispatch is **CPU-recording-bound** (every
matmul finishes faster than it can be re-recorded), exactly the regime
multi-submit overlap is designed for.

## 4. Sub-sprint plan (revised)

The brief's 7-step plan is rejected; the real critical-path is much
shorter. Each sub-sprint below is **1-3 days, single-commit, has a
hard bench gate, and can be skipped without breaking the next.**

### 19B-A. Multi-submit pacing for prefill

**Hypothesis (testable):** splitting `prefill_batch` into multiple
`vkQueueSubmit` calls overlaps CPU recording of submit N+1 with GPU
execution of submit N, closing the pp ≤ 128 gap (currently ~0.71-0.74×
llama.cpp).

**Scope:**
* Add a submit-boundary heuristic to `prefill_batch` mirroring
  llama.cpp's: emit a submit every M layers OR every B bytes of matmul,
  whichever first, with M and B env-tunable.
* Use the existing `CommandContext` to allocate N command buffers from
  a pool; chain them with timeline-style fences.
* Default: 4 submits for a 36-layer model (every 9 layers), tuned via env.

**Files:** `src/backend/vulkan/commands.rs` (multi-CB plumbing),
`src/backend/vulkan/forward.rs::prefill_batch` (submit boundary
emission). Estimate: ~150 LOC.

**Bench gate:** pp=128 ≥ +5% over Sprint 19A (currently 2575 tok/s →
target ≥2700). Larger pp may benefit too. Decode unchanged.

**Risk:** low. Failure mode is a noisy +0%; doesn't break anything.

### 19B-B. Fuse `kv_copy_*` into `rms_norm_mul_rope`

**Hypothesis (testable):** llama.cpp's `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS`
fusion saves 2 dispatches + 2 barriers per layer (one each for K and V
KV-cache write). At 36 layers that's 72 dispatches + 72 barriers per
prefill, worth ~300-400 µs at pp ≥ 256.

**Scope:**
* Extend `vk_shaders/rms_norm.comp`'s `RMS_NORM_ROPE_FUSION` path
  with an additional output-binding that writes directly to the
  KV-cache slot, skipping the separate `kv_copy_fp16` /
  `kv_store_fp8` dispatch. Two new SPV variants: FP16-KV and FP8-KV.
* Routing change in `dispatch_layer_batch` to call the fused variant
  when KV target is FP16 or FP8 and there's no MQA stride mismatch.
* Skip when `VULKANFORGE_DISABLE_RMSROPE_KV_FUSE=1` for parity tests.

**Files:** `vk_shaders/rms_norm.comp` (+ new bindings),
`src/backend/vulkan/forward.rs` (routing), `build.rs` (+2 SPVs),
`src/backend/vulkan/shaders.rs` (+2 ShaderIds). Estimate: ~120 LOC.

**Bench gate:** pp=512 ≥ +2% Q4_K_M (currently 3865 → target ≥3940).
Bit-exact check against the 15-prompt suite.

**Risk:** medium. KV-cache layout assumptions (page-major vs
token-major) need careful preservation. Bit-exactness has to hold
across FP16 and FP8 KV.

### 19B-C. (skip-or-do) graph-aware barrier emission for prefill

**Hypothesis (testable):** of VF's 42 `compute_barrier` call sites, a
fraction is RAW-redundant in `dispatch_layer_batch`'s actual data flow.
A static analysis pass (the buffer doesn't have to live across
arbitrary IR — just the linear dispatch sequence) can elide N of them.

**Scope:**
* Add a tiny per-layer-batch tracker: `(buffer_id → last_writer_op_idx)`,
  emit a barrier only when the next reader's `idx > last_writer_idx
  - elidable_window`.
* No graph IR. Just a 50-LOC tracker passed through `dispatch_layer_batch`.

**Files:** `src/backend/vulkan/forward.rs`. Estimate: ~80 LOC.

**Bench gate:** ≥+1% pp=512 OR honest negative (file the report and
revert). Sprint 12D's 0% on decode is the precedent — this is a
re-test in the prefill regime.

**Risk:** low. Pure additive; failure is observable as no perf change
and we revert.

### 19B-D. (deferred) generic graph IR

**Decision:** *Do not build.* llama.cpp itself doesn't have a generic
"compute graph IR over a buffer-aliased pool" — the analogous pieces
(graph reorder, fusion match) are 200-300 LOC of pattern-matching over
a node array, plus 5 named scratch buffers. VulkanForge's existing
hand-rolled `dispatch_layer_batch` is at the same density and is
already fusion-dense for Qwen3-8B (§3.2).

If a future sprint *does* want a graph IR, the trigger should be
**onboarding a non-Qwen architecture with a different op DAG (e.g.
MoE, gated attention)** — not chasing prefill perf.

## 5. Dependency graph (revised)

```
19B-A (multi-submit) ─┐
                      ├── independent, parallel-able after this plan
19B-B (rmsrope+kv)  ──┤
                      │
19B-C (barriers)    ──┘  (can run last; if 19B-A + 19B-B already
                          erased the gap, may not be needed)
```

No sub-sprint depends on another. Each is ≤ 3 days, ≤ 200 LOC,
hard bench gate.

## 6. Recommendation

**Option D: Targeted ports, not graph framework.**

* **Do 19B-A first** (multi-submit). Highest expected payoff (close
  the pp ≤ 128 gap), lowest risk, smallest LOC. If this lands +5% on
  pp=128 and +2% on pp=512, we're at parity-ish without touching
  shaders.
* **Then 19B-B** (KV-write fusion). Targeted, two new SPVs, bit-exact
  testable, addresses one of the two real fusion gaps vs llama.cpp.
* **Then maybe 19B-C** (barrier elision in prefill). Cheap to try; if
  it's noise (Sprint 12D's outcome on decode), file an honest negative.

**Reject Option A (full graph):** the assumed 5-15% upside isn't
visible in llama.cpp's actual code — *llama.cpp's lever is the two
optimizations I extracted into 19B-A and 19B-B*, not a graph IR. A
3-4 week refactor for what is achievable in 2-3 focused 1-2 day
sprints would violate every memory rule we have ("13 of 14 isolated
optimizations were negative").

**Reject Option B (buffer-aliasing only):** non-lever (§3.1).
VRAM cleanup is a different goal and should not be sold as a perf
sprint.

**Reject Option C (port llama.cpp's full pool):** there is no full
pool to port. 5 named watermark scratches is what llama.cpp has;
that's not different in kind from what VF already has — just
named differently.

## 7. Honest caveats

1. **The pp ≤ 128 gap (~0.71-0.74×) might not be CPU-bound.** It might
   also be RADV's queue-submission latency or the small-K coopmat
   variant just being slower. 19B-A will *reveal* which by measuring
   pp=64/128/256 with submit boundaries every 4/9/18 layers — if the
   curve doesn't shift, the gap is in the kernel, not the recorder.
2. **Sprint 19A landed +57% on Q3_K_M pp=512.** The "easy" lever was
   shader-routing config, not architecture. We should keep checking
   for similar shader-side wins (Q4_0 mul_mm port? coopmat-aligned at
   pp%4≠0 fallbacks?) before committing weeks to a refactor.
3. **The 8-10% residual gap may simply be irreducible** on RADV at
   gfx1201 vs llama.cpp's mature dispatcher. VF being ahead of llama.cpp
   on decode (v0.3.3) and 0.92× on prefill is already an honest result.

## 8. Decision matrix for the user

| Question | Answer |
|---|---|
| Is a generic compute-graph IR worth building? | **No.** No evidence in llama.cpp it would close the gap; existing fusion coverage is 80%+. |
| Is multi-submit pacing worth porting? | **Yes** — it is the largest *measurable* prefill lever in `ggml-vulkan.cpp`. |
| Is buffer aliasing worth doing? | **No** as a perf sprint. Maybe later as a VRAM sprint. |
| Is the residual gap closeable? | **Probably partially** (5% via 19B-A + 19B-B), maybe not fully. |
| Suggested next sprint | **19B-A (multi-submit)** — 1-2 days, hard gate at pp=128 ≥ +5%. |
