# VulkanForge v0.2.1 Sprint 12G-A — llama.cpp Shared ggml Layer Audit

**Date:** 2026-04-30
**Branch:** main (HEAD = 36b1200, post-Sprint 12F retest)
**llama.cpp commit:** 23b8cc4 (`~/tmp/llama.cpp/`)
**Mode:** Pure analysis — fills the blind spot Sprint 12A left (everything BELOW the Vulkan backend).

## TL;DR — the structural advantages we missed

Sprint 12A audited only `ggml-vulkan.cpp`. Sprints 12D and 12E built and
measured the two infrastructure levers that audit identified, both
yielded ~0 % wall-time. **The reason** is that llama.cpp's host-side
infrastructure advantages live one level *below* the Vulkan backend, in
`ggml-alloc.c`, `ggml-backend.cpp`, `llama-graph.cpp`, the per-model
builders, and a graph-optimize pass *inside* `ggml-vulkan.cpp` we missed.

Five structural advantages, none of which we have:

1. **VIEW / RESHAPE / TRANSPOSE / PERMUTE never become Vulkan dispatches.**
   `ggml_vk_is_empty()` (line 14024) skips all of them. They are pure
   metadata operations on the host. Our `dispatch_layer` doesn't even
   distinguish: every operation we issue is a real shader dispatch.
2. **`ggml_set_rows` is a single graph op that writes to the KV-cache.**
   `cpy_k` returns `ggml_set_rows(K_cache, K_cur, k_idxs)` —
   one tensor node, one dispatch. We use a separate `kv_copy_fp16` shader
   that does an explicit buffer copy.
3. **Graph optimizer reorders nodes to bring fusion patterns adjacent.**
   `ggml_vk_graph_optimize` (line 14816, ~200 LOC) recognizes
   `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS`, `MUL_MAT+ADD(+ADD)`,
   `MUL_MAT_ID+ADD_ID+MUL`, and four `TOPK_MOE_*` patterns, then orders
   the graph so the fusion code in `ggml_vk_build_graph` can recognize them.
4. **`ggml-alloc` does proper liveness-based buffer reuse.** `ggml_gallocr`
   (1 248 LOC) tracks `n_children` and `n_views` per tensor; when both
   reach 0 the buffer goes back to a per-chunk free-list. `ggml_op_can_inplace`
   ops skip allocation entirely and reuse a parent's buffer. Our
   `forward.rs` has 32 hand-named persistent scratch buffers — no
   liveness, no reuse beyond the manual scratch_a/scratch_b ping-pong.
5. **`ggml_backend_sched` runs `n_copies` for pipeline parallelism.**
   It allocates events per backend and per copy, then can record/wait
   across copies to overlap CPU recording with GPU work for the
   previous copy. Our single-`one_shot` blocking submit pattern can't
   pipeline at all.

These are not micro-optimizations. **(1) and (2) directly explain why
llama.cpp Vulkan dispatches 10–13 ops/layer while we dispatch 17–18.**
Sprint 12B's "missing fusion" analysis identified the right symptom but
attributed it to a single missing 5-op fused shader. The actual cause is
broader: a chain of host-side machinery — graph build helpers, sched
split, graph_optimize, fuse-aware dispatch loop — that produces fewer
Vulkan dispatches *for the same logical work*.

---

## 1. Graph construction (`llama-graph.cpp`, `src/models/qwen3.cpp`)

`src/models/qwen3.cpp` (110 LOC total!) is the entire per-layer
forward-pass description for Qwen3. Per layer, the logical OPs are:

```cpp
cur     = build_norm(inpL, attn_norm, ..., LLM_NORM_RMS, il);   // 1× norm
auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, ...); // 1× qkv (3 mul_mat)
Qcur    = build_norm(Qcur, attn_q_norm, ...);                    // q-norm
Qcur    = ggml_rope_ext(ctx, Qcur, inp_pos, ...);                // q-rope
Kcur    = build_norm(Kcur, attn_k_norm, ...);                    // k-norm
Kcur    = ggml_rope_ext(ctx, Kcur, inp_pos, ...);                // k-rope
cur     = build_attn(inp_attn, wo, ..., Qcur, Kcur, Vcur, ...);  // attention (incl. KV write + O proj)
ffn_inp = ggml_add(ctx, cur, inpSA);                              // residual1
cur     = build_norm(ffn_inp, ffn_norm, ...);                    // ffn-norm
cur     = build_ffn(cur, ffn_up, ffn_gate, ffn_down, ...);       // ffn (gate, up, swiglu, down)
cur     = ggml_add(ctx, cur, ffn_inp);                           // residual2
```

Each of these high-level builders expands to multiple low-level
`ggml_tensor` graph nodes:

- `build_attn` issues the Q/K/V matmuls, then `mctx_cur->cpy_k(...)` and
  `cpy_v(...)` (both `ggml_set_rows`), then attention compute, then output
  projection `wo`.
- `build_qkv` does either one fused QKV matmul (with `ggml_view_3d` to
  slice into Q/K/V) or three separate matmuls — depending on whether
  `wqkv` exists.
- `build_ffn` issues the gate, up, swiglu (mul + silu), and down matmuls.

A critical comment in `build_attn` (line 2200 of `llama-graph.cpp`)
states:

> // expand k later to enable rope fusion which directly writes into
> // k-v cache

i.e. the order in which `build_forward_expand` is called controls
fusability. The graph-build code is *engineered* to enable fusion at
later stages.

| Property                                 | Value                                             |
|------------------------------------------|---------------------------------------------------|
| Per-Qwen3-layer build LOC                | **110** (`src/models/qwen3.cpp`)                  |
| Logical operations per layer             | ~12–13 (norm, qkv, q-norm, q-rope, k-norm, k-rope, attn-incl-KV-write+O-proj, residual1, ffn-norm, gate/up/swiglu/down, residual2) |
| Where the graph is built                 | `ggml_build_forward_expand(gf, cur)` per output  |
| Total ggml ops in the enum               | **100** (`GGML_OP_NONE` = 0 to `GGML_OP_COUNT`)  |
| Helper `build_attn` actions              | Q/K/V matmuls + cpy_k (set_rows) + cpy_v (set_rows) + attn_mha + wo matmul |
| `cpy_k` returns                          | `ggml_set_rows(ctx, k, k_cur, k_idxs)` (one node) |

---

## 2. Graph optimization (`ggml_vk_graph_optimize`, `ggml-vulkan.cpp:14816`)

**A graph-pass we missed in Sprint 12A.** Lives inside the Vulkan backend
but operates on the cgraph BEFORE dispatch. ~200 LOC. Triggered via the
backend `iface.graph_optimize` hook (line 559 of `ggml-backend.cpp`).

What it does, concretely:

1. **Identifies "empty" nodes** that don't generate dispatches:
   `is_empty(node)` = `op == NONE || RESHAPE || TRANSPOSE || VIEW || PERMUTE`
   (line 14826). These are skipped during reordering.
2. **Recognizes fusion patterns** that should stay adjacent:
   - `RMS_NORM + MUL` (with optional ROPE follow-on)
   - `MUL_MAT + ADD` (and `+ADD`)
   - `MUL_MAT_ID + ADD_ID` (`+MUL`)
   - `ROPE + VIEW + SET_ROWS` (the K-cache write fusion)
   - 4× `TOPK_MOE_*` patterns
3. **Reorders the graph** so that subsequent dispatch loop in
   `ggml_vk_build_graph` will see fusable patterns as consecutive nodes.

The reorder logic walks the original graph and, for each not-yet-emitted
node, scans up to 20 successors looking for nodes that don't depend on
any unemitted predecessor (within a small set of allowed exceptions for
the fusion pairs above). When it finds the start of a known pattern
(e.g. RMS_NORM followed by MUL), it greedily pulls the rest of the
pattern (e.g. ROPE → VIEW → SET_ROWS) directly after. The result is a
graph where the dispatch loop's `ggml_can_fuse_subgraph(...)` check
fires reliably.

This is the *missing piece* between "the graph has 17 ops worth of work"
and "the backend issues 11 dispatches": the graph-optimize pass plus
the fuse-aware dispatch loop.

| Property                              | Value                                              |
|---------------------------------------|----------------------------------------------------|
| Function                              | `ggml_vk_graph_optimize` (line 14816)              |
| Line count                            | ~200                                               |
| Disable env var                       | `GGML_VK_DISABLE_GRAPH_OPTIMIZE`                   |
| `is_empty` ops                        | NONE, RESHAPE, TRANSPOSE, VIEW, PERMUTE            |
| Recognized fusion patterns            | 8 (RMS_NORM+MUL, MAT+ADD(+ADD), MAT_ID+ADD_ID(+MUL), ROPE+VIEW+SET_ROWS, RMS+MUL+ROPE+VIEW+SET_ROWS, 4× TOPK_MOE) |

---

## 3. Memory planning (`ggml-alloc.c`, 1 248 LOC)

`ggml_gallocr_alloc_graph_impl` (line 717) is the planner. Algorithm:

1. **Pass 1 — count children & views:** for each node, increment
   `n_children` of its sources and `n_views` of its `view_src`.
2. **Pass 2 — allocate in node order:**
   - For each node, attempt **in-place reuse** of a parent's buffer if
     `ggml_op_can_inplace(node->op)` is true and the parent has exactly
     1 child and 0 views. Reused → don't allocate new memory, free the
     parent's slot.
   - Otherwise allocate via `ggml_dyn_tallocr_alloc` (free-list
     allocator).
   - After processing the node, decrement `n_children` of each source.
     When a source reaches `n_children == 0 && n_views == 0`, mark it
     freeable — its buffer goes back to the free-list.
3. **Free-list allocator** (`ggml_dyn_tallocr`): per-chunk free-list with
   `MAX_FREE_BLOCKS` entries; on free, blocks are coalesced; on alloc,
   a "best-fit-with-reuse-bonus" scoring picks where to place the
   tensor.
4. **Multi-chunk:** new chunks allocated when current chunk fills.
   `max_size` per chunk tracked.

This is **register-allocation-style** liveness analysis applied to GPU
buffers. The result is dramatically lower peak memory and much higher
buffer reuse than our 32-named-persistent-scratch model.

| Property                              | Value                                              |
|---------------------------------------|----------------------------------------------------|
| Planner function                      | `ggml_gallocr_alloc_graph_impl` (line 717)        |
| Algorithm                             | Liveness + free-list + in-place when allowed     |
| Reference counts tracked              | `n_children`, `n_views` per tensor                |
| Free-list size                        | `MAX_FREE_BLOCKS` per chunk                       |
| In-place handling                     | yes, via `ggml_op_can_inplace`                    |
| View handling                         | views share the parent's buffer — never allocate own |
| Allocation strategy                   | First-fit with reuse-bonus (line 233)             |
| Persistent scratch buffers (VF)       | **32** (forward.rs:161-184, manually named)       |

**Net effect:** the same logical graph that needs (say) 12 distinct
intermediate tensors can compile to ~3-4 distinct device buffers in
llama.cpp via in-place + lifetime reuse. We always have all 32 live
simultaneously (because we pre-allocate and never free).

---

## 4. OP → dispatch mapping (Vulkan backend's `ggml_vk_compute_forward`)

The Vulkan backend's per-node switch (`ggml-vulkan.cpp:13143-...`) handles
each `GGML_OP_*` value. Critically:

- **`GGML_OP_VIEW`, `GGML_OP_RESHAPE`, `GGML_OP_TRANSPOSE`,
  `GGML_OP_PERMUTE`, `GGML_OP_NONE`: NO dispatch.** They are pure host-side
  metadata operations. The op switch case for these doesn't appear in the
  dispatch path; `ggml_vk_is_empty()` handles them upstream.
- **`GGML_OP_SET_ROWS`: ONE dispatch** via the existing
  `pipeline_set_rows_i32[GGML_TYPE_*]` (line 752). This is what writes K
  and V into the cache. Fused with preceding ROPE when the pattern is
  recognized.
- **`GGML_OP_CONT`: ONE dispatch** when the source is non-contiguous,
  otherwise a no-op (line 9434).
- **`GGML_OP_GET_ROWS`: ONE dispatch** for embedding-table lookup and
  for the prefill-only "select rows" optimization (`inp_out_ids`).

| GGML op category                     | Generates Vulkan dispatch? |
|--------------------------------------|----------------------------|
| Compute ops (MUL_MAT, NORM, ROPE, …) | Yes                        |
| Fused patterns                       | One dispatch per pattern   |
| VIEW / RESHAPE / TRANSPOSE / PERMUTE | **No** — host-only metadata|
| NONE                                 | **No**                     |
| CONT (already contiguous)            | No                         |
| CONT (non-contiguous)                | Yes                        |
| SET_ROWS                             | Yes (single)               |
| GET_ROWS                             | Yes (single)               |

**`quantize_q8_1`** is **backend-internal**, NOT a graph op. The function
`ggml_vk_quantize_q8_1` (line 7449) is invoked by the Vulkan backend's
`ggml_vk_mul_mat` setup whenever `quantize_y` is true (line 7538). The
graph never sees a "quantize" node — it sees a `MUL_MAT` and the backend
inserts the quantize internally. This matches what we do.

---

## 5. Compute scheduling (`ggml_backend_sched`, `ggml-backend.cpp`)

The Sched orchestrates execution **across multiple backends** (typical
case: Vulkan + CPU). Per call to `ggml_backend_sched_graph_compute`:

1. **Split graph** (`ggml_backend_sched_split_graph`, line 1014) — assigns
   each node to a backend (mostly Vulkan for our workload), then
   chops the linear node list into runs of same-backend nodes. Each
   run becomes one "split".
2. **Allocate buffers** per split.
3. **Compute splits in order** (`ggml_backend_sched_compute_splits`, line
   1541):
   - For each split's input tensors, copy from source backend to this
     backend (no-op when both are GPU).
   - Issue `ggml_backend_graph_compute_async(split_backend, split_graph)`.
   - Record an event after compute so subsequent splits can wait.
4. **Pipeline parallelism** via `n_copies` (line 1700-area + struct
   field): allocates events per backend per copy. The next iteration's
   recording can begin while the current iteration's GPU work runs —
   this is the "pipelining" Sprint 12C identified as a potential lever.

For Qwen3-8B fully on Vulkan, there is typically **1 split per forward**
that goes to `ggml_backend_vk_graph_compute`. The Vulkan backend then
does its own internal batching (Sprint 12A: 5–10 submits per forward via
the `nodes_per_submit + mul_mat_bytes_per_submit` heuristic).

| Property                              | Value                                          |
|---------------------------------------|------------------------------------------------|
| Splitter function                     | `ggml_backend_sched_split_graph` (line 1014)   |
| Split criterion                       | Per-tensor backend assignment + view-op skipping |
| `n_splits` for Qwen3-8B all-Vulkan    | **1**                                          |
| Pipeline parallelism                  | yes, via `n_copies` events                     |
| Per-split graph_compute call          | `iface.graph_compute(backend, split_graph)`    |
| Cross-backend tensor copies           | yes, automatic via `ggml_backend_tensor_copy` |

---

## 6. Backend abstraction (`ggml_backend_i`, `ggml-backend-impl.h`)

The vtable a backend implements:

```c
struct ggml_backend_i {
    const char * (*get_name)(...);
    void         (*free)(...);
    void         (*set_tensor_async)(...);
    void         (*get_tensor_async)(...);
    void         (*set_tensor_2d_async)(...);
    void         (*get_tensor_2d_async)(...);
    bool         (*cpy_tensor_async)(...);
    void         (*synchronize)(...);
    // graph plans (unused currently)
    ggml_backend_graph_plan_t (*graph_plan_create)(...);
    void                      (*graph_plan_free)(...);
    void                      (*graph_plan_update)(...);
    enum ggml_status          (*graph_plan_compute)(...);
    // hot path:
    enum ggml_status          (*graph_compute)(backend, cgraph);
    void (*event_record)(backend, event);
    void (*event_wait)  (backend, event);
    void (*graph_optimize)(backend, cgraph);  // ← we missed this in 12A
};
```

The Vulkan backend's vtable assignment (~line 14000 of ggml-vulkan.cpp):

```c
.graph_compute  = ggml_backend_vk_graph_compute,
.event_record   = ggml_backend_vk_event_record,
.event_wait     = ggml_backend_vk_event_wait,
.graph_optimize = ggml_vk_graph_optimize,        // ← the optimizer pass
```

The Vulkan backend gets the **entire sub-graph** for its split via
`graph_compute`, AFTER the sched has already let the backend reorder it
via `graph_optimize`. The backend then:

1. Walks node-by-node
2. At each "fuse-eligible" node, calls `ggml_can_fuse_subgraph(...)` to
   check for a matching pattern (e.g. RMS+MUL+ROPE+VIEW+SET_ROWS at
   line 14570) — succeeds because the optimizer placed the pattern
   adjacent
3. Sets `ctx->num_additional_fused_ops` and emits ONE dispatch covering
   the fused pattern
4. Skips ahead by `num_additional_fused_ops` nodes

So fusion is a **two-stage** mechanism: graph-optimize reorders, then the
dispatch loop merges adjacent runs.

---

## 7. The complete pipeline — what reaches the Vulkan backend

```
                                            ───────────────
   User-supplied tokens ──> embed lookup ──> token tensor
                                            ───────────────
                                                   │
                                                   ▼
                                       per-layer build helpers in
                                       src/models/qwen3.cpp  (110 LOC)
                                       │  build_norm
                                       │  build_qkv         (mat_mul + view_3d slicing)
                                       │  build_norm        (q-norm)
                                       │  ggml_rope_ext     (q-rope)
                                       │  build_norm        (k-norm)
                                       │  ggml_rope_ext     (k-rope)
                                       │  build_attn        (cpy_k=set_rows + cpy_v=set_rows + attn_mha + wo)
                                       │  ggml_add
                                       │  build_norm        (ffn-norm)
                                       │  build_ffn         (gate + up + silu*mul + down)
                                       │  ggml_add
                                       ▼
                                       ggml_cgraph (linearised list of ggml_tensors)
                                                   │
                                                   ▼
                                       ggml_backend_sched_split_graph
                                       └─> 1 split (all-Vulkan for our workload)
                                                   │
                                                   ▼
                                       ggml-alloc: liveness-based buffer assignment
                                       (in-place reuse + free-list, ~3–4 distinct buffers
                                        for ~12 logical intermediates per layer)
                                                   │
                                                   ▼
                                       backend->iface.graph_optimize(cgraph)
                                       │
                                       │  ggml_vk_graph_optimize
                                       │  └─> reorder nodes so RMS+MUL+ROPE+VIEW+SET_ROWS,
                                       │       MUL_MAT+ADD, etc. are adjacent
                                                   │
                                                   ▼
                                       backend->iface.graph_compute(cgraph)
                                       │
                                       │  ggml_vk_build_graph (per-node dispatch loop)
                                       │  └─> at each fuse-eligible node:
                                       │       check ggml_can_fuse_subgraph(...)
                                       │       → emit ONE dispatch covering N nodes
                                       │       → skip N-1 nodes
                                       │
                                       ▼
                                       Vulkan dispatches (10–13 per layer, post-fusion)
```

By the time we reach actual Vulkan dispatches, llama.cpp has already
performed:

- **VIEW/RESHAPE/PERMUTE elision**: zero dispatches for any of these.
- **Buffer compaction**: dramatically fewer device buffers via liveness.
- **Topological reordering**: fusion patterns guaranteed adjacent.
- **Pattern-based fusion**: 5-op `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` and
  ~7 others collapsed into single dispatches.
- **Cross-backend overlap**: `n_copies` lets CPU record next forward
  while GPU runs current.

---

## Key findings — the structural advantages we missed

1. **The graph layer eliminates entire op categories before dispatch.**
   VIEW, RESHAPE, TRANSPOSE, PERMUTE produce *zero* Vulkan dispatches in
   llama.cpp because `ggml_vk_is_empty()` recognises them as pure
   metadata. In our `forward.rs`, every operation we issue maps 1:1 to
   a real shader dispatch — we have no concept of "view" as a metadata
   op. This alone accounts for several of the 4–7 dispatches per layer
   we issue beyond llama.cpp's count.

2. **`ggml_set_rows` is the K/V-cache write — one graph node, one
   dispatch.** The "5-op fusion `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS`" Sprint
   12B identified as our biggest missing fusion is actually the
   *combined* effect of: (a) cpy_k being already a single graph op
   (`ggml_set_rows`), (b) graph_optimize pulling ROPE→VIEW→SET_ROWS
   adjacent, and (c) the dispatch loop's pattern matcher merging them.
   Adding "the fused shader" alone (Sprint 12E's plan) didn't deliver
   because we don't have the host-side machinery (a) and (b) — and our
   current `kv_copy_fp16` shader is structurally different from
   `set_rows`.

3. **`ggml-alloc` is register-allocator-grade GPU memory planning we
   don't have.** Liveness via reference counts, in-place reuse,
   free-list allocator. The same logical graph that compiles to ~3–4
   live device buffers in llama.cpp expands to all 32 named scratch
   buffers held simultaneously in our `Forward` struct. The cache
   working set difference is plausibly large (working-set fits in L2
   for llama.cpp, doesn't for us → more HBM round-trips on every
   GEMV/GEMM).

4. **The "5-10 submits" Sprint 12A measured comes from sched-level
   pipelining via `n_copies`, not just from chunking inside Vulkan.**
   Sprint 12A focused only on `ggml_vk_submit`'s 100-node heuristic,
   missing that `ggml_backend_sched` *also* has multi-copy scheduling
   that lets recording overlap with execution. Our single-`one_shot`
   blocking pattern can't replicate this without an overhaul of how
   `forward_token` records and submits work.

5. **`ggml_vk_graph_optimize` is the missing reorder pass, and it lives
   inside the Vulkan backend.** Sprint 12A's `grep -n "optimize"` should
   have found this. The function is 200 LOC of dependency-aware node
   reordering whose sole purpose is to bring fusion patterns adjacent
   for the dispatch loop's pattern matcher. Without an equivalent in
   our code, even if we shipped a 5-op fused shader (Sprint 12E), the
   *graph nodes* would not be in the right order for it to fire — but
   this is moot for us since we don't have a graph at all.

## What this means for VulkanForge

The audits in 12A, 12B and the gap analysis in 12C concluded "the gap
is host-side infrastructure". 12D and 12E built the two infrastructure
levers from that conclusion and measured ~0 % improvement. The fix to
that contradiction is **not** that infrastructure doesn't matter — it's
that **infrastructure means something fundamentally different** in
llama.cpp than what 12C imagined:

- **What 12C meant by "infrastructure":** dispatch overhead, barrier
  cost, descriptor-set caching — host-side per-dispatch cost.
- **What llama.cpp actually does:** a complete graph-based execution
  pipeline with view elision, liveness allocation, fusion-aware
  reordering, and multi-copy scheduling — host-side per-graph cost,
  which then *avoids whole categories of GPU work*.

The structural fix would be to introduce a graph layer in VulkanForge:

1. `forward.rs` would build a list of `Op` enum values (with src/dst
   tensors as references) instead of issuing dispatches directly
2. A graph-build pass would establish reference counts (children/views)
3. A liveness-based allocator would assign device buffers
4. A fusion-aware reorder pass would identify and merge patterns
5. The dispatch loop would consume the optimized graph

That is an **architectural rewrite** of the Forward path, not an
incremental optimization. Effort estimate: 2–4 weeks for a competent
implementation, with risk of new bugs in scratch-buffer allocation and
fusion correctness.

The cheaper alternative: keep the imperative `forward.rs` style but
introduce **explicit fusion shaders** for the patterns that matter most:
- A `kv_cache_set_rows` shader that takes Q/K post-rope and writes
  directly to the KV cache slot, eliminating the separate `kv_copy_fp16`
  step.
- A `mul_mat_with_q8_1_quantize` super-shader that does activation
  quantize + GEMM in one dispatch.

Each of these is a 1–3 day project with measurable wall-time impact
(now that we know dispatch *count* matters more than barrier *count*).
But: until we have RGP profiling data (Sprint 12G-B), we don't know
which lever delivers GPU-time savings vs just CPU-record savings.

---

## Files audited

```
~/tmp/llama.cpp/ggml/src/ggml.c                 (7 760 LOC, partial: graph build primitives)
~/tmp/llama.cpp/ggml/src/ggml-alloc.c           (1 248 LOC, full)
~/tmp/llama.cpp/ggml/src/ggml-backend.cpp       (2 371 LOC, partial: sched + sched_compute_splits)
~/tmp/llama.cpp/ggml/src/ggml-backend-impl.h    (interface struct)
~/tmp/llama.cpp/src/llama-graph.cpp             (2 924 LOC, partial: build_attn, build_qkv, build_ffn, cpy_k)
~/tmp/llama.cpp/src/models/qwen3.cpp            (110 LOC, full)
~/tmp/llama.cpp/src/llama-kv-cache.cpp          (partial: cpy_k, cpy_v)
~/tmp/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp  (16 944 LOC, lines 14816+ for graph_optimize, 14570+ for fusion gating, 7449+ for quantize_q8_1)
```

Total: ~32 000 LOC of which ~5 000 LOC actually read in detail.

## Take-aways

1. **Sprint 12A had a blind spot.** It audited `ggml-vulkan.cpp`'s
   submit / barrier / pipeline machinery but missed the graph-optimize
   pass living inside the same file (line 14816), and entirely missed
   the upstream graph build / sched / alloc layers (~22 000 LOC of
   infrastructure beneath the Vulkan backend).

2. **Sprint 12C's mental model was wrong about what "infrastructure"
   means.** It imagined per-dispatch CPU overhead. The actual
   advantage is per-graph machinery that produces fewer dispatches
   total — VIEW/RESHAPE elision, liveness alloc, fusion-aware reorder.

3. **Sprint 12D and 12E delivered ~0 % because they optimised the
   wrong layer.** Eliding barriers and fusing one pair of dispatches
   in a 612-dispatch forward doesn't move the needle when the
   structural advantage is "we don't have those 612 dispatches in the
   first place".

4. **Two paths forward:** (a) introduce a graph layer (4-week rewrite),
   or (b) write specific fusion shaders (1-3 day projects each). Both
   require RGP profiling first to know which dispatches dominate
   wall-time.

5. **The graph_optimize hook is recent.** The `iface.graph_optimize`
   slot was added so backends can do their own node reordering before
   compute. We could add an equivalent path in our codebase even
   without a full graph rewrite — though the leverage is much smaller
   without an upstream graph that has the full op-list to reorder.
