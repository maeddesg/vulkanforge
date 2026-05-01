# VulkanForge v0.2.1 Sprint 12G-B — VulkanForge Shared Layer Audit

**Date:** 2026-04-30
**Branch:** main (HEAD = 691d59b, post-Sprint 12G-A ggml audit)
**Scope:** `src/backend/vulkan/forward.rs` (3 906 LOC), `kv_cache.rs`, `loader.rs`, `gguf.rs`, `pipeline.rs`, `pipeline_registry.rs`, `commands.rs`, `buffers.rs`
**Mode:** Pure analysis — mirrors 12G-A's structure for our own backend.

## TL;DR — we have basically none of llama.cpp's graph-layer machinery

Sprint 12G-A documented five structural advantages in llama.cpp's
shared ggml layer:

1. View/Reshape elision via `ggml_vk_is_empty()`
2. `ggml_set_rows` as the KV-cache write op (graph node, fusable)
3. `ggml_vk_graph_optimize` reordering for fusion patterns
4. `ggml-alloc` register-allocator-grade buffer planning
5. `ggml_backend_sched` multi-copy pipeline parallelism

**We have zero of these.** Our `forward.rs` is a 3 906-LOC imperative
function that simultaneously declares the Qwen3 forward topology AND
issues the per-OP Vulkan dispatches. There is no graph layer, no
liveness tracker, no node reordering, no view abstraction, no
multi-copy scheduling.

This is by design (simpler code, faster iteration in Phase 1–11) but
it explains why Sprint 12D (barrier elision) and 12E (norm+rope
fusion) yielded ~0 % wall-time despite saving ~280 barriers and 72
dispatches per forward: those savings cannot benefit operations that
don't exist in llama.cpp in the first place. **We're optimising the
wrong layer.**

The actionable conclusion is **path C (hybrid)**: build specific fusion
shaders and view-as-offset patterns now (1–3 day projects matching
what 12G-A identified per category), and defer the architectural graph
rewrite to v0.3+. The cumulative impact estimate from path B alone is
**+10–20 % decode** if RGP profiling (12G-C) confirms dispatch
*count* rather than per-dispatch *cost* is the lever. Path A
(graph-layer rewrite) could plausibly close the rest, but it's a
4-week effort and should be gated on path B's actual measured wins.

---

## 1. "Graph"-construction — do we have one?

| Property                          | Value                                              |
|-----------------------------------|----------------------------------------------------|
| Compute-graph / DAG structure     | **NO** — neither `Graph`, `Node`, nor `Op` exist   |
| Op-enum / tagged-union            | **NO** — direct method calls only                  |
| Intermediate representation       | **NO** — `forward.rs` is pure imperative           |
| Forward-pass architecture         | imperative `for layer in 0..n_layers { dispatch_layer(...) }` |
| `dispatch_layer` LOC              | ~210 (`forward.rs:1339-1551`)                      |
| `dispatch_layer_batch` LOC        | ~700 (`forward.rs:3023-3700`)                      |
| Number of `run_*` helper methods  | **17** (run_gemv, run_rms_norm, run_rope_neox, run_scalar_attn, run_flash_attn_*, run_binary, run_multi_add_rms, run_kv_copy_fp16, run_rms_norm_mul_rope, run_swiglu, run_quantize_q8_1, run_gemm, run_gemm_coopmat_q4k, …) |

Per-decode-layer dispatch enumeration (post-Sprint 12E, Qwen3-8B,
`has_qk_norm = true`, FP16 KV-cache):

| # | Helper                | Computes                  | Reads             | Writes      |
|---|-----------------------|---------------------------|-------------------|-------------|
| 1 | run_rms_norm          | attn_norm                 | input             | hidden_norm |
| 2 | run_gemv              | Q-projection              | hidden_norm       | q_buf       |
| 3 | run_gemv              | K-projection              | hidden_norm       | k_buf       |
| 4 | run_gemv              | V-projection              | hidden_norm       | v_buf       |
| 5 | run_rms_norm_mul_rope | Q-norm + Q-rope (fused)   | q_buf             | q_buf       |
| 6 | run_rms_norm_mul_rope | K-norm + K-rope (fused)   | k_buf             | k_buf       |
| 7 | run_kv_copy_fp16      | K → KV-cache slot (FP32→FP16) | k_buf, kv_cache_K | kv_cache_K |
| 8 | run_kv_copy_fp16      | V → KV-cache slot         | v_buf, kv_cache_V | kv_cache_V |
| 9 | run_scalar_attn       | full attention            | q_buf, kv_cache   | attn_out    |
|10 | run_gemv              | O-projection              | attn_out          | o_buf       |
|11 | run_multi_add_rms     | residual1 + ffn_norm (fused) | input, o_buf  | res1, hidden_norm |
|12 | run_gemv              | Gate-projection           | hidden_norm       | gate_buf    |
|13 | run_gemv              | Up-projection             | hidden_norm       | up_buf      |
|14 | run_swiglu            | silu(gate) * up (fused)   | gate_buf, up_buf  | ffn_hidden  |
|15 | run_gemv              | Down-projection           | ffn_hidden        | ffn_out     |
|16 | run_binary (Add)      | residual2                 | res1, ffn_out     | output      |

**= 16 dispatches per decode layer + 1 inline `kv_bar` barrier (TRANSFER+COMPUTE → COMPUTE).**
Plus 2 final dispatches (rms_norm_final + lm_head). Per 36-layer forward:
**578 dispatches** total, **396 compute_barriers** + 36 kv_bar transfer-stage barriers.

Per-prefill-layer enumeration (`dispatch_layer_batch`, default
`mul_mm_enabled` + `batch_attn_enabled`, has_qk_norm): **18 dispatches/layer**
in the mul_mmq path, **14 dispatches/layer** in the mul_mm path
(the 4 quantize_q8_1 calls drop). See Sprint 12B §6.2 for the full list.

**vs llama.cpp (12G-A):**
- llama.cpp Qwen3 builder: **110 LOC** (`src/models/qwen3.cpp`) — declares the WHAT only
- llama.cpp dispatches per layer: **10–13** (post-fusion, post-graph-optimize, post-view-elision)
- VulkanForge dispatches per layer: **16** (decode), **14–18** (prefill)
- **3–6 extra dispatches per layer** = ~108–216 extra dispatches per forward

---

## 2. Graph optimization / VIEW handling

| Property                                 | Value                                              |
|------------------------------------------|----------------------------------------------------|
| Graph optimizer (reorder/fuse pass)      | **NO** — we have no graph to optimize              |
| VIEW operations                          | not represented; we don't have the concept         |
| RESHAPE operations                       | not represented                                    |
| Explicit `cmd_copy_buffer` calls         | **2** in `dispatch_layer` (FP32 KV-write fallback for non-FP16 cache), **3** in `dispatch_layer_batch` (batch_input → batch_residual seed + KV bulk-copy K + V) |
| Sub-buffer offsets (view-equivalent)     | yes via `alloc_or_get_set(binding, buf, offset, range)` (e.g. `rope_pos_buf` with per-token offset, KV-cache slot offset for `kv_copy_fp16`) |
| Manual in-place ops                      | yes — `run_rope_neox(q_buf, q_buf, …)` reads + writes same buffer (line 1418), `run_rms_norm_mul_rope(q_buf, wqn, q_buf, …)` (line 1409–1417) |

### 2.1 Dispatch-by-dispatch comparison vs llama.cpp

For each VulkanForge decode dispatch, what would the equivalent llama.cpp graph node look like, and would it be a real GPU dispatch?

| # | Our helper            | llama.cpp graph op              | llama.cpp Vulkan dispatch?         |
|---|-----------------------|---------------------------------|------------------------------------|
| 1 | run_rms_norm          | `GGML_OP_RMS_NORM`              | yes (often fused with next MUL)    |
| 2 | run_gemv (Q)          | `GGML_OP_MUL_MAT`               | yes                                |
| 3 | run_gemv (K)          | `GGML_OP_MUL_MAT`               | yes                                |
| 4 | run_gemv (V)          | `GGML_OP_MUL_MAT`               | yes                                |
| 5 | run_rms_norm_mul_rope (Q) | `RMS_NORM + MUL + ROPE` (fused) | yes — single fused dispatch     |
| 6 | run_rms_norm_mul_rope (K) | `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS` (fused 5-op, writes K cache directly) | yes — **single dispatch** that writes the cache |
| 7 | run_kv_copy_fp16 (K)  | `VIEW + SET_ROWS` (fused into #6 above) | **NO** — already done by #6 |
| 8 | run_kv_copy_fp16 (V)  | `VIEW + SET_ROWS` for V         | yes — single dispatch              |
| 9 | run_scalar_attn       | `GGML_OP_FLASH_ATTN_EXT`        | yes                                |
|10 | run_gemv (O)          | `GGML_OP_MUL_MAT`               | yes                                |
|11 | run_multi_add_rms     | `ADD + RMS_NORM + MUL` (fused)  | yes (we already match)             |
|12 | run_gemv (Gate)       | `GGML_OP_MUL_MAT`               | yes                                |
|13 | run_gemv (Up)         | `GGML_OP_MUL_MAT`               | yes                                |
|14 | run_swiglu            | `SILU + MUL` or fused SwiGLU    | yes                                |
|15 | run_gemv (Down)       | `GGML_OP_MUL_MAT`               | yes                                |
|16 | run_binary (Add)      | `GGML_OP_ADD` (often cross-layer fused with next layer's RMS_NORM+MUL) | yes (last layer only) — typically fused away |

### 2.2 Which of OUR dispatches would be NO-OPS or FOLDED at llama.cpp?

Specifically:
- **#7 (run_kv_copy_fp16 for K)** — folded into #6 by their 5-op fusion.
  Saves **1 dispatch per layer × 36 = 36 dispatches per forward**.
- **#16 (run_binary Add residual2)** for layers 0..n-2 — folded
  into the next layer's `run_multi_add_rms` (#11). Saves **1
  dispatch per intermediate layer × 35 = 35 dispatches per forward**.
  (We already do this in prefill `dispatch_layer_batch` via Sprint
  9b.2 cross-layer fusion. We do NOT do it in decode `dispatch_layer`.)

**Per decode forward: 71 dispatches that llama.cpp eliminates and we don't.**

**vs llama.cpp (12G-A):**
- `ggml_vk_graph_optimize`: 200 LOC, 8 fusion patterns, reorders adjacent nodes
- `ggml_vk_is_empty()` recognises NONE/RESHAPE/TRANSPOSE/VIEW/PERMUTE
- We have neither, but `alloc_or_get_set(buffer, offset, range)` is the low-level plumbing that *could* support a view-as-offset representation if we built the graph layer to drive it.

---

## 3. Memory planning / buffer-liveness

| Property                              | Value                                              |
|---------------------------------------|----------------------------------------------------|
| Allocator                             | `gpu_allocator::vulkan::Allocator` (generic suballocator) |
| Liveness analysis                     | **NO**                                             |
| In-place reuse driven by liveness     | **NO** — manual same-buffer-for-input-and-output   |
| Persistent named scratch buffers (decode) | **20** (`forward.rs:161-184`)                  |
| Persistent named scratch buffers (prefill) | **+12 batch_*** (`forward.rs:280-296`)        |
| All scratch buffers always live       | yes — pre-allocated in `Forward::new`, never freed during forward |
| Buffer aliasing across ops            | **NO** — each named buffer gets its own VkBuffer + own allocation |
| `vkAllocateMemory` per token          | **0** ✓ (Sprint 12B confirmed)                     |

Decode scratch inventory (sizes for Qwen3-8B: hidden=4096, n_heads=32,
n_kv_heads=8, head_dim=128, ffn=12288):

| Buffer        | Size      | Memory location |
|---------------|-----------|-----------------|
| scratch_a     | 16 KB     | CpuToGpu        |
| scratch_b     | 16 KB     | GpuOnly         |
| hidden_norm   | 16 KB     | GpuOnly         |
| q_buf         | 16 KB     | GpuOnly         |
| k_buf         | 4 KB      | GpuOnly         |
| v_buf         | 4 KB      | GpuOnly         |
| attn_out      | 16 KB     | GpuOnly         |
| o_buf         | 16 KB     | GpuOnly         |
| res1          | 16 KB     | GpuOnly         |
| gate_buf      | 48 KB     | GpuOnly         |
| up_buf        | 48 KB     | GpuOnly         |
| ffn_hidden    | 48 KB     | GpuOnly         |
| ffn_out       | 16 KB     | GpuOnly         |
| logits_buf    | 600 KB    | GpuToCpu        |
| fuse0/1       | 16 B each | GpuOnly (dummies) |
| rope_*        | 1 KB total| mixed           |
| fa_scratch_*  | varies    | GpuOnly         |
| **total**     | **~1.0 MB** | (decode-only)  |

This is a tiny absolute footprint. The issue is not VRAM consumption
but **cache locality**: 20+ distinct buffers means Q-proj's q_buf and
Gate-proj's gate_buf (which are never live at the same time) get
*different* L2 cache lines. With ggml-alloc's liveness reuse,
llama.cpp would pack both into the same physical allocation, giving
better L1/L2 hit rates on consecutive GEMVs that share working sets.

### 3.1 In-place ops we already do

- `run_rms_norm_mul_rope(q_buf, wqn, q_buf, …)` — Q-norm reads and
  writes the same buffer.
- Same for K (`run_rms_norm_mul_rope(k_buf, wkn, k_buf, …)`).
- Without QK-norm, `run_rope_neox(q_buf, q_buf, …)` is in-place.
- `run_multi_add_rms(res1=input, o_buf, w, sum_out=res1, norm_out=hidden_norm)` writes `res1` in-place.

But these are all **manual** — they happen because the dispatch_layer
author chose them. There is no abstraction that would let an unrelated
`run_*` helper say "I can be in-place" and have a planner fold accordingly.

### 3.2 Buffers that *could* alias but don't

The following are never live simultaneously in dispatch_layer:

- `q_buf` (used at #2–#9) vs `gate_buf` or `up_buf` (used at #12–#14)
- `attn_out` (#9–#10) vs `ffn_hidden` (#14–#15)
- `o_buf` (#10–#11) vs `ffn_out` (#15–#16)

A liveness-aware planner could share buffers across these pairs. The
gain would be a smaller working set (~3–4 distinct buffers instead of
20) — better L2 cache reuse. Hard to quantify without profiling.

**vs llama.cpp (12G-A):**
- `ggml-alloc` (`ggml_gallocr`, 1 248 LOC): liveness via
  `n_children`/`n_views`, in-place when `ops_can_inplace`, free-list
  allocator, multi-chunk
- Result: ~3–4 live buffers per layer in steady state
- Ours: ~20 live buffers always (4–5× working set)

---

## 4. OP-dispatch mapping

For each VulkanForge decode dispatch, what's eliminable / fusable:

| # | Helper                | Total / forward (36 layers) | Eliminable?               | Fusable?                          |
|---|-----------------------|----------------------------:|---------------------------|-----------------------------------|
| 1 | run_rms_norm          | 36                          | no                        | already fused with #11 in prefill |
| 2–4 | run_gemv (Q/K/V)    | 108                         | no — distinct outputs     | no                                |
| 5 | run_rms_norm_mul_rope (Q) | 36                      | no                        | no                                |
| 6 | run_rms_norm_mul_rope (K) | 36                      | no                        | **yes — fuse with #7** (5-op)     |
| 7 | run_kv_copy_fp16 (K)  | 36                          | **yes if we fuse with #6**| —                                 |
| 8 | run_kv_copy_fp16 (V)  | 36                          | no                        | optional fuse with #6+#7          |
| 9 | run_scalar_attn       | 36                          | no                        | no                                |
|10 | run_gemv (O)          | 36                          | no                        | no (Qwen3 has no bias)            |
|11 | run_multi_add_rms     | 36                          | no                        | already fused                     |
|12–13 | run_gemv (Gate/Up) | 72                          | no                        | maybe parallel-issue (already are)|
|14 | run_swiglu            | 36                          | no                        | already fused                     |
|15 | run_gemv (Down)       | 36                          | no                        | no                                |
|16 | run_binary (Add)      | 36                          | **yes if we cross-layer-fuse** with #11 of next layer | — |
|   | **eliminable subtotal** |                           | **#7 (36) + #16 (35) = 71 dispatches/forward** | |

So out of **578 decode dispatches per forward**, ~71 are
structurally eliminable matching llama.cpp's known fusions. That's
**~12 % reduction** if we matched llama.cpp's existing patterns.

---

## 5. Compute scheduling

| Property                              | Value                                              |
|---------------------------------------|----------------------------------------------------|
| Dispatch ordering                     | strictly sequential within each `dispatch_layer`   |
| Q/K/V issued back-to-back (no inter-barrier) | yes (#2, #3, #4 share `hidden_norm` input, write distinct outputs) ✓ |
| Gate/Up issued back-to-back           | yes (#12, #13 share `hidden_norm`) ✓               |
| CPU/GPU overlap                       | **NO** — `commands.rs::one_shot` blocks on `wait_for_fences` |
| Pipeline parallelism (`n_copies`)     | **NO**                                             |
| Multi-token batching for decode       | **NO** — 1 token per `forward_token` call          |
| Speculative decode                    | not implemented                                    |
| Submit count per forward              | **1** (Sprint 12B confirmed)                       |
| `vkWaitForFences` per forward         | **1** (blocking, after submit)                     |

Sprint 12B already showed our model is "single submit + blocking
wait". We trade per-submit overhead (winning vs llama.cpp's 5–10
submits) for zero CPU/GPU overlap (losing vs llama.cpp's `n_copies`
event-pipelined recording).

For decode at 92 tok/s = 10.9 ms / token, the CPU cost of recording
578 dispatches is roughly 578 × 5 µs = **2.9 ms** that could
*potentially* overlap with GPU work for the previous token.

**vs llama.cpp (12G-A):**
- `ggml_backend_sched` allocates events per backend per copy (`n_copies`)
- The next forward's recording can begin while the previous forward's
  GPU work runs — saves ~2–3 ms of decode latency (estimate)
- Our single-`one_shot` blocking pattern can't replicate this without
  restructuring `forward_token` into a producer/consumer pair

---

## 6. Backend abstraction

| Property                              | Value                                              |
|---------------------------------------|----------------------------------------------------|
| `forward.rs` LOC                      | **3 906**                                          |
| WHAT/HOW separation                   | **NONE** — `forward.rs` does both                  |
| Model architecture definition         | hard-coded for Qwen3 in `dispatch_layer` (line 1 says "Forward-pass orchestration for Qwen3 decode") |
| Per-architecture branches             | only `cfg.has_qk_norm` (Qwen-only path) and `cfg.rope_variant` (Neox vs norm) |
| Could a graph layer be inserted between `Forward::forward_token` and the `run_*` helpers? | yes, but it requires touching `dispatch_layer` (~210 LOC) and `dispatch_layer_batch` (~700 LOC) |
| Number of supported architectures     | **1** (Qwen3); other archs would need new dispatch_layer variants |

`forward.rs` simultaneously holds:
- The Qwen3-specific OP sequence (the WHAT)
- The Vulkan-specific descriptor management, push-constants packing,
  barrier emission, scratch-buffer routing (the HOW)
- The forward-pass orchestration (one_shot, fence wait, logits
  readback)

To support a new model architecture (Llama-3, Mistral, Mixtral) under
the current design, we'd duplicate ~210 LOC of `dispatch_layer` per
architecture, copying all the dispatch+barrier+scratch routing logic.
With an abstraction layer (Op-enum + interpreter), the Qwen3 file
would shrink to ~80 LOC of pure topology, and adding Llama-3 would be
~80 LOC more.

**vs llama.cpp (12G-A):**

| Layer       | llama.cpp                    | VulkanForge          |
|-------------|------------------------------|----------------------|
| WHAT        | `src/models/qwen3.cpp` (110 LOC, pure ggml ops) | embedded in `dispatch_layer` |
| MEMORY      | `ggml-alloc.c` (1 248 LOC)   | manual scratch buffers |
| SCHED       | `ggml-backend.cpp` (2 371 LOC) | absent              |
| OPTIMIZER   | `ggml_vk_graph_optimize` (200 LOC) | absent          |
| HOW         | `ggml-vulkan.cpp` (16 944 LOC) | the rest of `forward.rs` (~3 700 LOC) |

llama.cpp has ~21 000 LOC across 5 files for what we cram into 3 906
LOC of one file. The trade-off is real: their structure enables
N model architectures with O(N×80) LOC; ours scales as O(N×210) LOC
plus the orchestration cost duplicated each time. We chose simplicity
over generality, and that choice prevents the optimisations 12G-A
catalogued.

---

## 7. The core question — what's missing, and what does it cost to add?

### 7.1 Missing-component table

| # | Component                              | llama.cpp has | We have | Estimated wall-time impact (post-RGP) | Fix effort | Path |
|---|----------------------------------------|:-------------:|:-------:|:-------------------------------------:|:----------:|:----:|
| 1 | Compute graph (DAG / Op enum)          | yes           | NO      | 0 % directly; enabler for 2/3/4/5     | 2–3 weeks  | A    |
| 2 | Graph optimizer (reorder for fusion)   | yes (200 LOC) | NO      | 0 % alone; enables fusion patterns    | 1 week     | A (after 1) |
| 3 | VIEW/RESHAPE elision                   | yes           | NO      | 0–2 % (we don't currently emit views) | 1 week     | A (after 1) |
| 4 | Liveness-based buffer allocator        | yes (1 248 LOC) | NO    | 1–3 % (cache locality)               | 2 weeks    | A (after 1) |
| 5 | In-place op tagging                    | yes           | manual  | 0 % (we already do it manually)       | 0          | (existing)  |
| 6 | Pipeline parallelism (`n_copies`)      | yes           | NO      | 5–10 % decode (CPU/GPU overlap)      | 1 week     | A (independent) |
| 7 | SET_ROWS-style KV-write fused with ROPE| yes           | NO      | 1–3 % decode (saves 36 dispatches)   | 3 days     | B           |
| 8 | Cross-layer fusion in DECODE (#16 + next #11) | yes (in prefill only for us) | NO | 1–3 % decode (saves 35 dispatches) | 3 days     | B           |
| 9 | MULTI_ADD ≤ N-arity                    | yes (≤ 9)     | binary only | 0 % (Qwen3 dense doesn't trigger) | 1 week     | (skip)      |
|10 | MAT_MAT + ADD bias fusion              | yes           | NO      | 0 % (Qwen3 has no bias)              | (skip)     | (skip)      |

### 7.2 Path B (specific fixes, no graph layer)

Two concrete dispatches we could eliminate without building a graph
layer:

#### B-1: Fused `K-norm + K-rope + KV-write` shader (`fused_kv_set_rows.comp`)

Replace `run_rms_norm_mul_rope (K)` + `run_kv_copy_fp16 (K)` with a
single dispatch that reads K, applies norm + rope, and writes the
result directly into the FP16 KV-cache slot at `pos_offset_bytes(layer, pos)`.

- Effort: ~3 days (new shader, 1 new ShaderId, 1 new run_* helper, parity test)
- Saves: 36 dispatches per decode forward
- Estimated impact: 1–3 % decode (assuming dispatch overhead is the lever — RGP must confirm)

#### B-2: Cross-layer residual fusion in DECODE

Replicate the prefill Sprint 9b.2 pattern in `dispatch_layer`: replace
the trailing `run_binary(Add)` with the next layer's `run_multi_add_rms`,
combining residual2 + next-layer attn_norm into a single dispatch.

- Effort: ~3 days (refactor dispatch_layer to know about cross-layer state, test)
- Saves: 35 dispatches per decode forward
- Estimated impact: 1–3 % decode

#### Path B total: ~6 days, +2–6 % decode (estimate)

### 7.3 Path A (graph layer rewrite)

A 4-week architectural project that introduces:
- `enum Op { RmsNorm, MulMat, RopeNeox, KvSetRows, ... }`
- `struct ForwardGraph { nodes: Vec<Node>, allocator: GraphAlloc }`
- A graph builder per supported architecture (Qwen3 → ~80 LOC)
- A liveness-based buffer planner (mirroring `ggml-alloc`)
- A fusion-aware reorder pass (mirroring `ggml_vk_graph_optimize`)
- A graph executor that drives the existing `run_*` helpers via the
  optimised plan

Effort: 3–4 weeks, high test load, risk of new bugs.
Estimated impact: enables paths 2/3/4 above and unlocks future
multi-architecture support. Direct decode wall-time: probably +5–10 %
beyond path B.

### 7.4 Path C (hybrid — recommended)

1. **Sprint 12G-C (RGP profiling, 2–3 days):** Capture per-dispatch
   GPU time on a decode forward. Find the actual bottleneck (CU
   starvation? memory BW? cache churn?).
2. **Sprint 12H/I (path B fixes, ~6 days total):** Implement B-1 and
   B-2 *only if* RGP confirms dispatch count is the lever. If RGP
   shows GEMV CU starvation, skip B and address that.
3. **Sprint 13+ (path A, 3–4 weeks):** If path B delivers > 5 % and
   RGP suggests more is recoverable via fusion/reorder, undertake
   the graph rewrite. Otherwise accept the ceiling and pivot to v0.3.

---

## Key findings

1. **VulkanForge has no graph layer at all.** Both the Qwen3
   architecture (the WHAT) and the Vulkan dispatch orchestration (the
   HOW) live in `forward.rs`'s `dispatch_layer` (210 LOC) and
   `dispatch_layer_batch` (700 LOC). To add a model architecture or
   any of llama.cpp's graph-level optimisations, we have to either
   duplicate that code or introduce the missing abstraction.

2. **Two specific fusions match known llama.cpp patterns and are
   worth targeting without a full graph rewrite.** A `fused
   K-norm+K-rope+KV-write` shader (saves 36 dispatches/forward) and a
   cross-layer residual fusion in decode (saves 35
   dispatches/forward) match what 12G-A documented. Each is ~3 days.

3. **Buffer working-set is ~5× larger than llama.cpp's.** We hold 20
   simultaneously-live scratch buffers; a liveness planner would pack
   them into ~3–4. The wall-time impact is hard to quantify without
   GPU cache profiling but is plausibly 1–3 % decode via better L2
   reuse on consecutive GEMVs.

4. **Pipeline parallelism is the largest single architectural gap.**
   `n_copies` event-driven recording could overlap ~3 ms of CPU
   record with the previous forward's GPU work. That alone is a 5–10
   % decode lever, but it requires restructuring `commands.rs::one_shot`
   into a producer/consumer pair — a 1-week project.

5. **The path forward depends on RGP data.** Sprint 12D and 12E both
   delivered ~0 % despite making correct structural reductions; the
   reason is dispatch-overhead is *not* the dominant cost on this
   hardware/code-path. Until RGP tells us where decode time
   *actually* goes (GEMV compute? attention? memory BW?
   per-dispatch barrier stall?), any further infrastructure work is
   uncalibrated. **Sprint 12G-C is the gate.**

---

## Files audited

```
src/backend/vulkan/forward.rs          (3 906 LOC, full)
src/backend/vulkan/buffers.rs          (115 LOC, full)
src/backend/vulkan/commands.rs         (142 LOC, already from 12B)
src/backend/vulkan/kv_cache.rs         (272 LOC, partial)
src/backend/vulkan/loader.rs           (255 LOC, partial — LoadedModel struct)
src/backend/vulkan/gguf.rs             (540 LOC, partial — ModelConfig)
src/backend/vulkan/pipeline.rs         (579 LOC, already from 12B)
src/backend/vulkan/pipeline_registry.rs (531 LOC, already from 12B)
```

Total: ~6 300 LOC of which ~2 500 LOC actually read in detail
(forward.rs in full; others as needed).
