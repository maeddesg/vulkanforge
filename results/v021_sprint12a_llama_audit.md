# VulkanForge v0.2.1 Sprint 12A — llama.cpp Vulkan-Backend Infrastructure Audit

**Date:** 2026-04-30
**Branch:** main (HEAD = 8dcdbda, post-Sprint 11G-D)
**llama.cpp commit:** 23b8cc4 (`~/tmp/llama.cpp/ggml/src/ggml-vulkan/`)
**Mode:** Pure analysis — no VulkanForge code changes, no llama.cpp changes.

## Why this audit

Sprint 11 ran 5 GPU-kernel optimizations against llama.cpp's prefill GEMM and
all five came back NO-GO (Int8-coopmat, FP16-coopmat, aligned mul_mmq,
integer-DP, Q8_1 — all already shipped, all not faster). Our shaders are
bit-identical to llama.cpp upstream, the features overlap, but the 0.54×
gap is stable.

That means the gap is **not in the kernels** — it is in the host-side
infrastructure: command-buffer batching, descriptor management, sync
strategy, fusion. This audit catalogues llama.cpp's choices in 7 categories
so Sprint 12B can produce the matching catalogue for VulkanForge and a
diff worth optimising.

`ggml-vulkan.cpp` total LOC: **16 944**.

---

## 1. Command-Buffer + Submit

| Property | Value |
|---|---|
| Submits per prefill (pp=512, ~720 nodes) | **5–10** (not per-layer) |
| Submits per decode (1 token, ~25 nodes/layer × 36 layers ≈ 900 nodes) | **5–10** (same heuristic) |
| `vkCmdBeginCommandBuffer` / `vkCmdEndCommandBuffer` | **per submit boundary** (not per OP) |
| CommandBuffer reuse | **yes** — pool flagged `TRANSIENT_BIT \| RESET_COMMAND_BUFFER_BIT` (line 906); pool reset every 10 buffers (`cleanup_frequency = 10`, line 2608) |
| CommandPool count | 2 per device: `compute_cmd_pool`, `transfer_cmd_pool` (line 1927–1928) |
| Context-Objekt | `vk_context` (weak-ref `compute_ctx` + `transfer_ctx` in `ggml_backend_vk_context`, line 1914–1916). Lazy-created via `ggml_vk_get_compute_ctx`; many OPs accumulate into one ctx before submit |
| Batching: OPs per CommandBuffer | up to `nodes_per_submit = 100` (line 14511), or until `mul_mat_bytes_per_submit` threshold (line 14516, default `min(100 MB, last_total_mul_mat_bytes / 40)`), or `almost_ready` (when `< 20 %` nodes remain) |
| Submit ramp-up | First 3 submits: `mul_mat_bytes_per_submit *= 2` per submit (line 14751–14752). Start fast (avoid GPU idle), ramp slower (better batching) |
| Separate Compute/Transfer Queue | **yes** when distinct queue family available (line 5136); cross-queue sync via timeline semaphore. Falls back to single queue otherwise (line 5139, `single_queue`) |

**Submit body (`ggml_vk_submit`, line 2409):** builds a `std::vector<vk::SubmitInfo>` of all queued submissions, takes `queue_mutex`, calls `queue.submit(submit_infos, fence)` *once* per ctx flush. So even within one ctx, multiple cmd-buffer sequences can share a single `vkQueueSubmit` call.

The submit heuristic at line 14716 — `submit = (submitted_nodes >= 100) || (mul_mat_bytes >= threshold) || (almost_ready) || last_node` — is the central dispatch-batching policy. Submit-while-GPU-busy is the goal.

---

## 2. Synchronisation + Barriers

| Property | Value |
|---|---|
| Barriers per Layer (typical Qwen3 prefill) | **0–4** with elision (see below) |
| `vkCmdPipelineBarrier` callsites | **49** of `ggml_vk_sync_buffers` (line 2852); only **1** real `pipelineBarrier` call (line 2861) — the rest are guarded callsites that *may* skip via dirty-flag elision |
| Barrier Type | Single global **MemoryBarrier** (NOT BufferMemoryBarrier) covering all buffers |
| `srcStage` / `dstStage` | both = `q->stage_flags` (= `COMPUTE_SHADER` for compute queue, `TRANSFER` for transfer queue) |
| `srcAccessMask` / `dstAccessMask` | `eShaderRead \| eShaderWrite \| eTransferRead \| eTransferWrite` (full set) |
| Barrier-Granularity | **batched / elided**, not per dispatch |
| `vkQueueWaitIdle` / `vkDeviceWaitIdle` | **0** in hot path |
| `vkWaitForFences` | **11** total in source — all in test/init/transfer fallback paths, **none** between dispatches in steady-state forward |
| Fences | `ctx->fence`, `ctx->almost_ready_fence` — 1+1 per backend instance, not pooled per-submit |
| Semaphores | **Timeline Semaphores** (Vulkan 1.2+) for cross-queue sync. `transfer_semaphore` (line 1917) signals transfer→compute completion |
| Events | helpers exist (`ggml_vk_set_event` line 2884, `ggml_vk_wait_events` line 2892) but only used in pre-allocated buffer reuse paths, not in the standard graph_compute |

**Barrier elision via dirty flags** (line 1912 + line 7680–7740):
3 flags `prealloc_x_need_sync`, `prealloc_y_need_sync`, `prealloc_split_k_need_sync` track whether the prealloc scratch buffer has pending writes. A consumer (`if (need_sync) { sync_buffers(); }`) only inserts a barrier when actually needed. Dirty flags are cleared by the barrier and re-set by the next writer.

**Pipeline-and-tensor cache for prealloc_y** (line 1903–1904):
`prealloc_y_last_pipeline_used` + `prealloc_y_last_tensor_used` — if the *same* pipeline writes the *same* source tensor consecutively (e.g. two GEMMs with same activation), the second dequant/quantize_q8_1 dispatch is **skipped entirely** (line 7693–7710). This is a content-aware deduplication, not just a barrier saver.

---

## 3. Buffer + Memory Management

| Property | Value |
|---|---|
| `vkAllocateMemory` callsites | **2** (line 2722, 2740, both inside `ggml_vk_create_buffer`) |
| `vkAllocateMemory` per init | one per buffer; total ≈ 10–15 for an 8B model (weight tensors + prealloc + staging) |
| `vkAllocateMemory` per token | **0** in steady state (prealloc buffers resized only when graph shape grows) |
| Buffer Pool / Bucket allocator | **NO** general buffer pool. Replaced by 4 named prealloc scratch buffers |
| Arena/Suballocator | `vk_subbuffer { buffer, offset, size }` (offset into a larger buffer); GGML tensors carry `vk_tensor_offset` into a per-buffer-context dev_buffer |
| Pre-allocated scratch | `prealloc_x`, `prealloc_y`, `prealloc_split_k`, `prealloc_add_rms_partials`, `sync_staging` (line 1891) |
| Prealloc resize | only when `current.size < required` — destroy + re-create (line `ggml_vk_preallocate_buffers`); stable across forwards |
| Memory Mapping | **persistent** — `mapMemory(VK_WHOLE_SIZE)` once at buffer create (line 2770), pointer cached in `buf->ptr` |
| Staging Buffer | `sync_staging` only used for fallback (when `_async` returns false). Hot path is direct-mapped DEVICE_LOCAL+HOST_VISIBLE (ReBAR) |
| Pinned host memory pool | `device->pinned_memory` (line 862) — `vector<tuple<ptr, size, vk_buffer>>`. `ggml_vk_host_get` does a pointer-range scan to find pre-pinned host buffers (zero-copy) |
| Buffer Device Address | yes when `device->buffer_device_address` (line 2782–2785) — caches `bda_addr` per buffer |
| Memory selection chain | DEVICE_LOCAL+HOST_VISIBLE (ReBAR) → DEVICE_LOCAL → HOST_VISIBLE+HOST_COHERENT (line 2799–2826). Per-vendor branches: `prefer_host_memory` (env var), `uma`, `disable_host_visible_vidmem` |

**Key insight:** llama.cpp's "buffer pool" is not a size-class allocator — it's *named scratch buffers + grow-on-demand*. The prealloc_* set covers all dispatch-time scratch needs; per-OP tensor buffers come from a single big device buffer per `vk_buffer_context` carved up by offsets.

---

## 4. Descriptor Set Management

| Property | Value |
|---|---|
| Descriptor Pool model | **pre-alloc, multi-pool** — `VK_DEVICE_DESCRIPTOR_POOL_SIZE = 256` sets per pool (line 105). New pool added when sets exhausted |
| Pool growth strategy | grow by **50 %** of current size (line 2369: `needed = max(3 * current / 2, requirements)`) to avoid frequent allocations |
| Descriptor Set allocation | **pre-allocated + reused** via `descriptor_set_idx` index (line 1924). Reset to 0 each forward (line 13482) |
| Descriptor Set update | `updateDescriptorSets` — one Write per dispatch with all bindings batched (line 6618–6620) |
| Push Descriptors | **NOT used** (0 occurrences of `pushDescriptor` / `cmdPushDescriptorSet`) |
| Descriptor Templates | **NOT used** (0 occurrences of `DescriptorUpdateTemplate`) |
| Bindings per Set | **fixed at 12** (`MAX_PARAMETER_COUNT = 12`, line 127). Unused bindings ignored per pipeline |
| DescriptorSetLayout | **single global `dsl`** for all pipelines (line 5589, line 677). One layout handles every kernel via the 12-binding superset |
| Sets per Pipeline | 1 |
| Dynamic Offsets | **not used** (subbuffer offsets baked into `DescriptorBufferInfo`) |

Per-dispatch host overhead (line 6603–6628 in `ggml_vk_dispatch_pipeline`):
1. `updateDescriptorSets` (1 write, 12 buffers max)
2. `pushConstants`
3. `bindPipeline`
4. `bindDescriptorSets`
5. `dispatch`

= **5 commands per OP**, no inline barrier (barrier is separately driven by dirty-flag).

---

## 5. Pipeline + Shader Management

| Property | Value |
|---|---|
| Pipeline Cache on disk | **NO** (0 occurrences of `PipelineCache` / `createPipelineCache`). All pipelines compiled fresh per process start |
| `ggml_vk_create_pipeline` callsites | **373** in source — many guarded by `if (device->pipeline_X)` checks |
| Pipelines per Q4_K | **6** primary: l/m/s × {aligned, unaligned}; plus `_id` MoE variants, `_int_k` integer-key variants. ~12–18 total per quant in the int_k path |
| Pipeline creation timing | **lazy on first dispatch** (`pipeline->needed = true` then `ggml_vk_load_shaders(device)`, line 6604) |
| Parallel compile | yes — `std::async` + `compile_count` mutex/condvar (line 2125–2128, 3469–3473), capped at `hardware_concurrency()`. Comment notes "no longer benefitting" — kept for backwards compatibility |
| Spec-Constants | **11 ints** for matmul: `{BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP}` (line 3367 AMD-coopmat-override). Plus per-pipeline custom sets |
| Per-vendor / per-architecture warptile overrides | **yes** (line 3360–3380): AMD-GCN gets `{256,64,64,32,16,16,2,2,2,1,16}`; AMD coopmat gets `{256,128,128,32,subgroup_size_8,64,2,tm_m,tn_m,tk_m,subgroup_size_8}`; Intel XE2 gets `{512,128,128,32,...}` |
| Pipeline selection | `ggml_vk_guess_matmul_pipeline` (line ~7300+). Non-coopmat2 thresholds: `m≤32 \|\| n≤32` → S, `m≤64 \|\| n≤64` → M, else L. coopmat2 path adds `tiles_l/m vs shader_core_count` + split_k considerations |
| Aligned variant gate | `n % align == 0` where `align = {128, 64, 32}` for L/M/S |
| RDNA-specific subgroup-size tables | `rdna1_pipelines`, `rdna2_pipelines` static maps (line 3191, 3198) — per-name subgroup-size override |

`get_subgroup_size(name, architecture)` (referenced line 3437) selects the "preferred" wavefront size per pipeline based on name pattern. RDNA1/2 distinct from RDNA3+/CDNA, AMD distinct from Intel.

---

## 6. Dispatch-Pattern

| Property | Value |
|---|---|
| `vkCmdDispatch` (vulkan-hpp `.dispatch()`) | always last call in `ggml_vk_dispatch_pipeline` (line 6628) |
| `ggml_vk_dispatch_pipeline` callsites | **49** |
| Dispatches per Transformer-Layer (Qwen3-8B dense, no fusion) | ~19–25 OPs/layer (matches our 19) |
| **Dispatches per layer with llama.cpp fusion** | **~10–13 OPs/layer** (5–10 OPs saved per layer) |
| Per-OP host commands | 5 (update + push + bindPipeline + bindDS + dispatch) |
| Per-OP barrier | conditional — only when dirty flag set |
| Dispatch-Geometry | `wg = ceil(elements / wg_denoms)` per axis (line 6588–6590) |

**Fusion catalogue** (line 11078 area, `ctx->num_additional_fused_ops` set per node):

| Pattern | `num_additional_fused_ops` | Saves |
|---|---:|---:|
| `MUL_MAT + ADD` (bias) | 1 | 1 dispatch |
| `MUL_MAT + ADD + ADD` (bias + residual) | 2 | 2 dispatches |
| `MUL_MAT_ID + ADD_ID` (MoE bias) | 1 | 1 |
| `MUL_MAT_ID + ADD_ID + MUL` (MoE bias + scale) | 2 | 2 |
| `MUL_MAT_ID + MUL` (MoE scale) | 1 | 1 |
| `MULTI_ADD` (variable arity) | up to 8 | up to 8 |
| `RMS_NORM + MUL` | 1 | 1 |
| `RMS_NORM + MUL + ROPE` | 2 | 2 |
| `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS` (5-op KV-cache write fusion) | 4 | 4 |
| `ROPE + VIEW + SET_ROWS` | 2 | 2 |
| `TOPK_MOE_*` (4 modes for routing) | up to 4 | up to 4 |

Fusion gating (line 11078–11150): each pattern has a `ggml_vk_can_fuse_*` predicate (correctness check on tensor edges, shapes, pipeline availability) — fusion is opportunistic, not always taken.

**No fused gate+up GEMM.** The two GEMMs stay separate; only `silu(gate) * up` is collapsed via a single SwiGLU dispatch (`pipeline_swiglu`, line 798).

**Quantize_q8_1**: scheduled via `ggml_vk_quantize_q8_1` (called from `ggml_vk_mul_mat`, line 7708) when `quantize_y` is true. It writes to `prealloc_y` and is cached: if the *same* tensor was already quantized to the same prealloc by the same pipeline, the dispatch is **skipped entirely** (line 7693–7710 — see §2 above). Practical effect: quantize_q8_1 cost ≈ 1× per unique activation tensor per forward, not per consuming GEMM.

---

## 7. Host-Device Transfer

| Property | Value |
|---|---|
| Weight Upload | once per model load via `ggml_backend_vk_set_tensor_async`. With ReBAR available, weights land in DEVICE_LOCAL+HOST_VISIBLE — no staging copy |
| Weight Memory | **DEVICE_LOCAL+HOST_VISIBLE** (mappable VRAM, ReBAR) preferred; falls back to DEVICE_LOCAL with staging |
| Activation Transfer (CPU → GPU) | embedding tokens go via `set_tensor_async`. With separate transfer queue: parallel with compute. With pinned host buffer (registered through `ggml_backend_vk_host_buffer_type`): zero-copy via `ggml_vk_host_get` lookup |
| Output Transfer (GPU → CPU) | logits via `get_tensor_async` — uses `compute_ctx` (no transfer queue for reads) and falls back to staging copy if the device buffer isn't HOST_VISIBLE |
| Staging Buffer | `device->sync_staging` + `ctx->sync_staging` — fallback only, resized on demand (`ggml_vk_ensure_sync_staging_buffer`, line 6722) |
| Transfer Queue | **separate async** when distinct queue family exists (line 5136 + `async_use_transfer_queue` device flag) |
| Transfers per token (steady-state decode) | **2** (token-id in, logits out) — both async via timeline semaphore handshake |
| `deferred_memcpy` | host-side copies into staging are **deferred until just before submit** (line 6706) so they happen after CPU has set up all OPs but before GPU starts — minimises wall-time overlap of CPU and GPU work |

**Cross-queue handshake**: transfer queue's timeline semaphore (`ctx->transfer_semaphore`) is incremented on each transfer submit. The next compute_ctx waits on it via `result->s->wait_semaphores.push_back(ctx->transfer_semaphore)` (line 6671–6674). This makes weight uploads / token uploads cleanly overlap with previous compute.

---

## 8. Architecture overview

| Item | Value |
|---|---|
| `ggml-vulkan.cpp` LOC | **16 944** |
| Top-level structs | `vk_pipeline_struct` (133), `vk_matmul_pipeline_struct` (163), `vk_device_struct` (184), `vk_buffer_struct` (188), `vk_queue` (229), `vk_command_pool` (207), `vk_command_buffer` (199), `vk_context_struct`, `ggml_backend_vk_context` (1880-area) |
| Forward-pass entry | `ggml_backend_vk_graph_compute` (line 14439) |
| Inner per-node dispatch | `ggml_vk_build_graph` (line 12901) — switch on `node->op`, calls per-OP helpers (`ggml_vk_mul_mat`, `ggml_vk_flash_attn`, etc.) |
| Forward-pass submit logic | `graph_compute` main loop (line 14695–14760): per-node fusion detection → `build_graph` enqueues into compute_ctx → submit-decision (line 14716) every 100 nodes / 100 MB / almost_ready → `ggml_vk_compute_forward` flushes via `ggml_vk_submit` |
| Init | `ggml_backend_vk_init` (line 15130) → `ggml_vk_instance_init` → `ggml_vk_get_device` (line 4872) |
| Cleanup | per-pool reset every 10 buffers (`cleanup_frequency`, line 2608); descriptor sets zero'd per forward (line 13482) |
| Backend interface | `ggml_backend_vk_interface` (around line 14000): `set_tensor_async`, `get_tensor_async`, `cpy_tensor_async`, `graph_compute`, `event_record`, `event_wait` |

---

## Key Findings

1. **The submit boundary is THE central optimization.** llama.cpp does *not* submit per-OP. The heuristic
   `submit = (submitted_nodes ≥ 100) || (mul_mat_bytes ≥ threshold) || almost_ready || last_node`
   produces **5–10 submits per forward pass**, not 720. With ramp-up (`mul_mat_bytes_per_submit *= 2` for first 3 submits), the early CPU-bound work submits fast and the later compute-bound work submits in larger chunks. This is the single biggest infrastructure lever.

2. **Barriers are dirty-flag-driven, not per-dispatch.** `ggml_vk_sync_buffers` is called 49 times in the source but most are guarded by `if (prealloc_X_need_sync)`. Plus per-pipeline + per-tensor caching skips redundant quantize/dequant entirely. The result: a typical Qwen3 layer issues **0–4 barriers**, not one per OP.

3. **Aggressive multi-OP fusion at the graph layer.** Fusion catalogue covers MAT+ADD(+ADD), RMS_NORM+MUL(+ROPE(+VIEW+SET_ROWS)), MULTI_ADD (up to 9-way), TOPK_MOE (4 modes). On Qwen3-style dense models the saving is ~5–10 dispatches per layer (∼25 % reduction). The 5-op `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` fusion writes the rotated KV directly into the cache slot — no intermediate buffer, no barrier between norm/rope/cache write.

4. **Persistent descriptor sets, persistent map, persistent prealloc.** `vkAllocateMemory` is called **0 times per token** in steady state. Descriptor sets are pre-allocated and indexed (`descriptor_set_idx`); pool grows by 50 %. Memory is `mapMemory(VK_WHOLE_SIZE)`'d once. The hot path has zero Vulkan-object creation.

5. **Two queues, timeline-semaphore handshake.** Compute and transfer run on distinct queue families when available (RDNA4: yes — separate transfer queue advertised). Transfer→compute sync uses a timeline semaphore `transfer_semaphore` that the next compute_ctx waits on. Weight uploads / token uploads overlap with compute essentially for free.

6. **Per-vendor warptile tuning is the second-biggest infrastructure lever.** AMD-GCN gets one set of `{BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP}`, AMD-coopmat-capable gets another, Intel XE2 a third. RDNA1/2 also get per-pipeline-name subgroup-size overrides. We ported the AMD-coopmat-override values in Sprint 11C — that's why our scalar mmq L-tile is competitive.

7. **No pipeline cache on disk.** Despite the surface area (hundreds of pipeline variants), llama.cpp does not persist `VkPipelineCache` to disk. Compile-on-first-use is fast enough on RADV that this isn't worth the disk-I/O complexity. Init is bounded by SPIR-V module creation, not pipeline compilation.

The structural conclusion: **llama.cpp's Vulkan win comes from minimising host-side overhead, not from faster shaders.** Submit batching saves ~140 `vkQueueSubmit` round-trips per forward; barrier elision saves ~60+ `vkCmdPipelineBarrier` calls; fusion saves ~5–10 dispatches per layer (≈ 200 per forward at 36 layers); persistent allocation saves zero-cost-target per-token alloc. Sprint 12B should produce the matching catalogue for VulkanForge to identify which of these levers we have not pulled.
