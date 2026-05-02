# VulkanForge v0.2.1 Sprint 12B — VulkanForge Vulkan-Backend Audit

**Date:** 2026-04-30
**Branch:** main (HEAD = 0a47e8e, post-Sprint 12A llama.cpp audit)
**Files audited:** `src/backend/vulkan/*.rs` — **10 460 LOC** total, of which `forward.rs` = 3 757 LOC, `pipeline.rs` = 579, `pipeline_registry.rs` = 531, `commands.rs` = 142, `buffers.rs` = 115, `device.rs` = 240, `loader.rs` = 255, `vram_arena.rs` = 255, `kv_cache.rs` = 272, `q4k.rs` = 345, `shaders.rs` = 442, plus tokenizer/spm/chat/decode/etc.

Same 7 categories as Sprint 12A (llama.cpp audit). Each section ends with the explicit `vs llama.cpp` diff.

---

## 1. Command-Buffer + Submit

| Property | Value (line ref) |
|---|---|
| Submits per Forward (decode 1 token) | **1** (`commands.rs::one_shot_profiled`, line 102: `device.queue_submit(...)`) |
| Submits per Prefill batch | **1** (one `one_shot` call wraps the whole batch — `forward.rs:2894`) |
| Wait per Submit | **1** `vkWaitForFences` immediately after, **BLOCKING** (`commands.rs:110`) |
| `vkBeginCommandBuffer` / `vkEndCommandBuffer` | **per Forward call** (= per submit). All ~700 dispatches recorded in ONE primary cmdbuf (`commands.rs:89`, `commands.rs:97`) |
| CommandBuffer Reuse | **yes** — `RESET_COMMAND_BUFFER` flag on pool (`commands.rs:28`); `device.reset_command_buffer(cmd, ::empty())` per `one_shot` (`commands.rs:81`) |
| CommandBuffer Pool | **single** pool, **single** cmdbuf, **single** fence in `CommandContext` (`commands.rs:18-20`). Persistent across Forward instance lifetime |
| `one_shot` callsites in `forward.rs` | **8** — one per public forward variant (`forward_token`, `forward_token_profile`, `prefill_batch`, debug helpers) |
| Batching (OPs per cmdbuf) | **whole forward** in one cmdbuf — typically **~700 dispatches at 36 layers × ~19 OPs/layer** for prefill, or **~36 × 19 = 684** for decode |
| Separate Compute / Transfer Queue | **NO** — single `compute_queue` (`device.rs:24`, `device.rs:161`); no transfer queue created |
| Timeline Semaphores | **NO** — pure fence-based sync |

**vs llama.cpp:**
- Submits: VF **1** vs llama.cpp **5–10** — **opposite trade-off**
- VF batches *everything* into one cmdbuf, llama.cpp splits into ~5-10 chunks of ~100 nodes each
- VF blocks (wait_for_fences) immediately after submit. llama.cpp's multiple submits are pipelined via fences without hot-path waits.
- VF: 1 cmd, 1 fence, 1 pool. llama.cpp: command-pool with reuse cycle (cleanup_frequency=10 buffers).
- **VF wins on submit-overhead** (~140 fewer vkQueueSubmit per equivalent forward). **llama.cpp wins on CPU/GPU overlap** (CPU records next chunk while GPU runs current chunk).

---

## 2. Synchronisation + Barriers

| Property | Value (line ref) |
|---|---|
| `cmd_pipeline_barrier` callsites in `forward.rs` | **20** |
| `compute_barrier(dev, cmd)` helper callsites in `dispatch_layer` (decode) | **12** per layer (`forward.rs:1313+`) |
| `compute_barrier` callsites in `dispatch_layer_batch` (prefill) | **~11** per layer (`forward.rs:3010+`) |
| Barriers per Forward (36 layers, decode) | **~432** (12 × 36) |
| Barriers per Prefill batch | **~396** (11 × 36) plus per-batch transfer barriers |
| Barrier-Typ | `vk::MemoryBarrier` (global, NOT BufferMemoryBarrier) (`forward.rs:3729-3742` `compute_barrier`) |
| `srcStage` / `dstStage` | both `COMPUTE_SHADER` for compute steps; `TRANSFER`+`COMPUTE_SHADER` → `COMPUTE_SHADER` for KV-cache writes (`forward.rs:1404`) |
| `srcAccessMask` / `dstAccessMask` | `SHADER_WRITE` → `SHADER_READ \| SHADER_WRITE` |
| Barrier after EVERY logical block | **YES** — unconditional barrier between (a) attn_norm, (b) Q/K/V, (c) Q/K-norm, (d) RoPE, (e) KV-write, (f) attention, (g) O-proj, (h+i) add+ffn_norm, (j) gate/up, (k+l) swiglu, (m) down, (n) residual2 |
| Barrier-Elision / Dirty-Flags | **NO** — every block emits a `compute_barrier` regardless of whether the next read needs it |
| `vkQueueWaitIdle` / `vkDeviceWaitIdle` | **0** in `src/backend/vulkan/` (verified by `grep -rn` ) |
| `vkWaitForFences` callsites | **1** in hot path (`commands.rs:110`, fence per `one_shot`) |
| Fences | **1** persistent fence per `CommandContext`, reset at start of each `one_shot` (`commands.rs:36`, `commands.rs:82`) |
| Semaphores | **0** (no timeline, no binary) |
| Events | **0** |

**vs llama.cpp:**
- Barriers per Layer: VF **~12** vs llama.cpp **0–4** with dirty-flag elision — **VF emits ~3× more barriers**
- Barriers per Forward: VF **~432** vs llama.cpp **~150** estimated
- Elision strategy: VF has **none**; llama.cpp has 3 dirty flags + per-pipeline-and-tensor cache for q8_1 quantize that elides redundant dispatches entirely
- WaitIdle: both 0. ✓
- Fence count: both 1 per submit. ✓
- Semaphores: VF none; llama.cpp uses timeline semaphores for cross-queue.
- **The barrier-density gap is the single biggest sync delta.** Each `compute_barrier` on RDNA4 + RADV is ~1–10 µs; ~280 extra barriers per forward × ~5 µs ≈ **1.4 ms** of pure barrier overhead. At pp=512 our forward is ~50–100 ms — barriers are 1.5–3 % of forward time, modest in absolute terms but a clear, measurable opt target.

---

## 3. Buffer + Memory Management

| Property | Value (line ref) |
|---|---|
| `GpuBuffer::new` callsites in `forward.rs` | **1** (the `mk_storage` helper alias) |
| `GpuBuffer::new` callsites in hot path (forward_token / prefill_batch) | **0** ✓ |
| `vkAllocateMemory` per Token | **0** ✓ — gpu-allocator suballocates from pre-allocated heaps |
| Allocator | `gpu_allocator::vulkan::Allocator` (single instance per device) — handles bucket/free-list internally |
| Buffer Pool / Arena | **`vram_arena.rs`** exists (255 LOC) but is unused by the forward path; `gpu_allocator` provides the suballocation in production |
| Pre-alloc Scratch Buffers (decode) | **20** named buffers in `Forward` struct (`forward.rs:161-184`): `scratch_a/b`, `hidden_norm`, `q_buf`, `k_buf`, `v_buf`, `attn_out`, `o_buf`, `res1`, `gate_buf`, `up_buf`, `ffn_hidden`, `ffn_out`, `logits_buf`, `fuse0/1` (dummies), `rope_*` (3) |
| Pre-alloc Scratch Buffers (prefill) | **+12** more `batch_*` buffers (line 280-296), sized for `max_prefill_tokens × hidden`/`q_dim`/etc. |
| Memory-Mapping | **persistent** via gpu-allocator's `Allocation::mapped_slice_mut()`. Buffer creation (`buffers.rs:31-60`) calls `bind_buffer_memory` once; mapped pointer cached |
| Memory Locations used | `GpuOnly` (DEVICE_LOCAL, ~28 buffers), `CpuToGpu` (HOST_VISIBLE, 3 buffers: `scratch_a`, `rope_pos_buf`, `batch_input`), `GpuToCpu` (HOST_READABLE, 1 buffer: `logits_buf`) |
| Staging-Buffer | **for weights only** — 1 GiB CpuToGpu staging in `loader.rs:25` (`STAGING_BYTES`), destroyed after model load. Not used per-token. |
| Weight Memory | **DEVICE_LOCAL** (`MemoryLocation::GpuOnly` in `loader.rs`). NOT mapped from host. |
| Buffer-Pool growth | not applicable — gpu-allocator handles growth opaquely |

**vs llama.cpp:**
- vkAllocateMemory per Token: VF **0** ✓ vs llama.cpp **0** ✓ — **tied**
- Pre-alloc model: VF **20–32 named buffers** vs llama.cpp **4 named scratch + per-tensor `vk_subbuffer`** (offsets into one big device buffer per buffer-context)
- Memory mapping: both persistent. ✓
- Weight memory: **VF DEVICE_LOCAL only** vs **llama.cpp DEVICE_LOCAL+HOST_VISIBLE (ReBAR) preferred**. Performance impact is the *load-time* staging copy we always pay; per-token impact is zero.
- Buffer-pool: **gpu-allocator** vs **named scratch + offset-arithmetic**. Both achieve zero-alloc-per-token. Functional parity.

---

## 4. Descriptor Set Management

| Property | Value (line ref) |
|---|---|
| Descriptor Pool | **1 per `Forward`**, sized at construction (`forward.rs:464`) |
| Pool `max_sets` | `(per_layer_sets * n_layers + 64) * 4` ≈ for max_pp=256: `(14 + 5*256 + 2) * 36 + 64` × 4 ≈ **186 624** descriptor sets allocatable |
| Pool `descriptor_count` | `max_sets * 8` STORAGE_BUFFER descriptors (line 461) |
| DescriptorSetLayout | **per-pipeline, reflected from SPIR-V** (`pipeline.rs:489`); each shader has its own DSL with bindings count = number of buffers it uses. NOT a single global DSL. |
| Bindings per Set | varies per pipeline — reflected via `spirv_reflect.rs` (`pipeline.rs:475-491`) |
| Sets per Pipeline | 1 (only `set = 0` supported; `pipeline.rs:475-477`) |
| Push Descriptors | **NO** (0 occurrences of `push_descriptor`) |
| Descriptor Templates | **NO** (0 occurrences of `DescriptorUpdateTemplate`) |
| Dynamic Offsets | **NO** (offsets baked into `DescriptorBufferInfo`) |
| Descriptor Update | `vkUpdateDescriptorSets` in `write_bindings` — one Write per binding (NOT a batched single Write); allocate first, write second |
| Descriptor Set per Dispatch | `alloc_or_get_set` (`forward.rs:1586-1606`): with cache enabled (default ON via `VULKANFORGE_CB_REUSE`), **HashMap lookup** keyed by `BindingSignature(layout, bindings)` returns the cached `vk::DescriptorSet` (zero Vulkan calls). On miss: allocate + write + insert |
| Cache reset | only `prefill_batch` (`forward.rs:2892`) and debug helpers reset the pool + cache; decode forwards keep cache populated |

**vs llama.cpp:**
- Pool model: VF **HashMap-keyed cache** vs llama.cpp **pre-allocated indexed array** (`descriptor_set_idx`). Functionally similar (both reuse sets); VF pays a hash per dispatch, llama.cpp pays an index increment.
- DSL: VF **per-pipeline (reflected)** vs llama.cpp **single global DSL with MAX_PARAMETER_COUNT=12** for *all* pipelines. llama.cpp uses fewer DSLs (cheaper to switch); VF uses tighter DSLs (fewer wasted bindings per dispatch). Net wash.
- Update: both `vkUpdateDescriptorSets`. Neither uses push descriptors or templates.
- Pool growth: VF **fixed at construction** with 4× headroom; llama.cpp **grows by 50 %** on demand.
- **No major delta here** — both approaches achieve descriptor-set-reuse across forwards. Hash overhead of `BindingSignature::new` per dispatch is the marginal cost VF pays that llama.cpp doesn't.

---

## 5. Pipeline + Shader Management

| Property | Value (line ref) |
|---|---|
| `VkPipelineCache` on disk | **YES** (`pipeline_registry.rs:75-79`) — load from `cache_path` (default `~/.vulkanforge/pipeline_cache.bin`), save back via `save_cache` after pipeline creation |
| Pipeline cache load behaviour | Vulkan loader validates header; incompatible blob (different driver/vendor) silently discarded |
| ShaderId enum entries | **104** (`shaders.rs`) |
| SPV blobs | **65** unique `.spv` files (Sprint 11G-D state) |
| Pipeline-Erstellung | **eager at PipelineRegistry::new** — for-loop over `ALL_SHADERS` (`pipeline_registry.rs:85`) creates every pipeline up-front |
| Parallel compile | **NO** — sequential loop, no `std::thread::spawn` or `rayon` |
| Spec-Constants | per-shader explicit; matmul GEMM gets the 11-int warptile array (`pipeline_registry.rs:240-275`) — `BLOCK_SIZE=256, BM=BN=64, WM=WN=32, WMITER=2, TM=2, TN=4, TK=1, WARP=64` for S-tile, `BM=BN=128, WM=WN=64, WMITER=1, TM=4, TN=2` for L-tile |
| L-tile Spec-Constants source | Sprint 11C ports llama.cpp's `l_warptile_mmq_int_k` AMD-coopmat-override values (`pipeline_registry.rs:242` matches `ggml-vulkan.cpp:3367` byte-for-byte) |
| Pipeline Selection | `layer_weight_shader_gemm` (`forward.rs:3640-3686`) picks shader by `(GemmKind, q4/q6, prefer_l)`. `prefer_l = m > 128 && n > 256` |
| Pipeline Selection thresholds | S/L only — NO M-tile path. Sprint 11G abandoned the M-tile int8-coopmat shader. |
| Per-GPU / Per-Vendor overrides | **NO** — hard-coded for RDNA4 / gfx1201 throughout. All warptiles tuned for AMD-coopmat. |

**vs llama.cpp:**
- Disk cache: **VF YES** vs **llama.cpp NO**. VF wins cold-start time on repeat launches.
- Compile timing: VF **eager** vs llama.cpp **lazy on-first-use**. VF pays full compile cost up front; llama.cpp amortises compile across the first few forwards. Steady-state identical.
- Parallel compile: VF **none** vs llama.cpp `std::async` + condvar capped at `hardware_concurrency()`. Init-only difference.
- Per-vendor overrides: **VF none** vs llama.cpp **AMD-GCN ≠ AMD-coopmat ≠ Intel-XE2 ≠ AMD-Windows-old**. VF is single-target (RDNA4 only); not a gap for our use case.
- Spec-Constants: identical *content* for the AMD-coopmat path (Sprint 11C verified bit-for-bit). ✓

---

## 6. Dispatch-Pattern

| Property | Value (line ref) |
|---|---|
| `cmd_dispatch` callsites in `forward.rs` | **17** |
| `dispatch_layer` (decode) OPs per layer | **~14–15 dispatches** (Qwen3-8B, mul_mmq path, has_qk_norm = true) |
| `dispatch_layer_batch` (prefill) OPs per layer | **~17–19 dispatches** (mul_mmq path, batch_attn ON) — see itemised list below |

### 6.1 Decode `dispatch_layer` — itemised (Qwen3-8B, default mul_mmq):
1. `run_rms_norm` (attn_norm) — `forward.rs:1325`
2. `run_gemv` Q-proj
3. `run_gemv` K-proj
4. `run_gemv` V-proj
5. `run_rms_norm` Q-norm (only Qwen has_qk_norm)
6. `run_rms_norm` K-norm
7. `run_rope_neox` Q
8. `run_rope_neox` K
9. KV-write — `cmd_copy_buffer` (FP32) **OR** `run_kv_copy_fp16` (FP16)
10. `run_scalar_attn` (or `run_flash_attn` / `run_flash_attn_split` / `run_flash_attn_tiled` / `run_flash_attn_coopmat`) — 1 dispatch
11. `run_gemv` O-proj
12. `run_multi_add_rms` (residual1 + ffn_norm fused) — Sprint 9b
13. `run_gemv` Gate-proj
14. `run_gemv` Up-proj
15. `run_swiglu` (silu(gate)*up fused) — Sprint 9a
16. `run_gemv` Down-proj
17. `run_binary` Add (residual2)

= **17 dispatches** per layer in the decode path. (Counting compute dispatches only — the FP32 KV-write is a transfer, not a dispatch.)

### 6.2 Prefill `dispatch_layer_batch` — itemised (default mul_mmq + batch_attn ON):
1. `run_quantize_q8_1` (attn_norm output → batch_q8) — only when `mul_mmq` (skipped on `mul_mm` path)
2. `run_gemm` Q-proj
3. `run_gemm` K-proj
4. `run_gemm` V-proj
5. `run_rms_norm_mul_rope` Q (Qwen has_qk_norm) — Sprint 9c.5 fused (norm+rope)
6. `run_rms_norm_mul_rope` K
7. KV-write — `cmd_copy_buffer` (FP32) or `run_kv_copy_fp16_k_b` (FP16) + V → 0 or 2 dispatches
8. `run_flash_attn_batch` / `run_flash_attn_tiled_br` / `run_flash_attn_coopmat` — 1 dispatch
9. `run_quantize_q8_1` (attn_out → batch_q8) — `mul_mmq` only
10. `run_gemm` O-proj
11. `run_multi_add_rms` (residual1 += O, fused with ffn_norm) — Sprint 9b
12. `run_quantize_q8_1` (ffn_norm output → batch_q8) — `mul_mmq` only
13. `run_gemm` Gate-proj
14. `run_gemm` Up-proj
15. `run_swiglu` (silu(gate)*up fused) — Sprint 9a
16. `run_quantize_q8_1` (ffn_hidden → batch_q8) — `mul_mmq` only
17. `run_gemm` Down-proj
18. `run_multi_add_rms` (residual2 += ffn_out, fused with **next** layer's attn_norm) — Sprint 9b.2 cross-layer

= **18 dispatches** per layer in `mul_mmq` prefill path (FP16 KV adds 2 → 20). On `mul_mm` (FP32 activation) path: **drop the 4 `quantize_q8_1` calls → 14 dispatches**.

### 6.3 Fused OPs we already have

- `multi_add_rms` (Sprint 9b): `residual_add + rms_norm * weight` in 1 dispatch — covers up to 2 inputs
- `swiglu` (Sprint 9a): `silu(gate) * up` in 1 dispatch
- `rms_norm_mul_rope` (Sprint 9c.5): `rms_norm + mul + rope_neox` in 1 dispatch — used in batch path Q/K
- Sprint 9b.2 cross-layer fusion: residual2 + next-layer-attn_norm fused

### 6.4 Fused OPs llama.cpp HAS but VF does NOT

- `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS` (5-op): rope output written **directly into the KV-cache slot** — 1 dispatch instead of 2 (rope + KV-copy)
- `MUL_MAT + ADD` / `MUL_MAT + ADD + ADD`: bias add fused into the GEMM kernel (irrelevant for Qwen3-8B which has no bias, but blocks generic-model support)
- `MUL_MAT_ID + ADD_ID + MUL`: MoE bias + scale fusion (irrelevant for dense Qwen3, blocks MoE)
- `MULTI_ADD` (up to 9-arity): variable-arity adds in 1 dispatch — VF's `multi_add_rms` only covers binary
- `TOPK_MOE` (4 modes): MoE routing fusion

### 6.5 quantize_q8_1 redundancy check

| Source tensor | Consumers | VF quantizes how often per layer |
|---|---|---|
| post-attn_norm (`batch_norm`) | Q, K, V projections (3) | **1** (shared via `batch_q8`) ✓ |
| post-attention (`batch_attn_out`) | O projection (1) | **1** ✓ |
| post-ffn_norm (`batch_norm`) | Gate, Up projections (2) | **1** (shared via `batch_q8`) ✓ |
| post-swiglu (`batch_ffn_hidden`) | Down projection (1) | **1** ✓ |
| **Total redundancy** | — | **0** — every quantize feeds ≥ 1 consumer with no double-dispatch |

**Note**: VF's `batch_q8` is reused — it is overwritten between Q/K/V batch and FFN batch. Each rewrite is a single dispatch. **There is no redundant q8_1 dispatch in VF prefill.**

llama.cpp's per-pipeline + per-tensor q8_1 cache (line 7693–7710 of `ggml-vulkan.cpp`) handles a *different* case: when the **same tensor** is consumed by **the same pipeline** in a later op (e.g., across layer boundaries), the second quantize is skipped entirely. For our prefill layer-by-layer pattern this rarely fires — each layer writes a fresh `batch_norm`/`batch_attn_out`. Caching would matter for shared-input multi-output paths (less common in dense models).

### 6.6 Per-Dispatch helper

`run_gemm` (`forward.rs:2627`) does:
1. `alloc_or_get_set` — HashMap lookup (cached) or allocate + write (uncached)
2. `cmd_bind_pipeline`
3. `cmd_bind_descriptor_sets`
4. `cmd_push_constants`
5. `cmd_dispatch`
6. (NO inline barrier — barrier is emitted by the *caller* via `compute_barrier`)

= **5 commands per OP**, identical to llama.cpp's `ggml_vk_dispatch_pipeline`. ✓

**vs llama.cpp:**
- Decode dispatches per layer: VF **17** vs llama.cpp **~10–13** with full fusion. Gap = **4–7 extra dispatches per layer × 36 = 144–252 extra dispatches per forward**
- Prefill dispatches per layer: VF **18 (mmq)** or **14 (mul_mm)** vs llama.cpp **~10–13** with full fusion
- The biggest single missing fusion: **RMS_NORM+MUL+ROPE+VIEW+SET_ROWS** (5-op). VF currently issues `rms_norm_mul_rope_q_b` + `rms_norm_mul_rope_k_b` + `kv_copy_fp16_k_b` + `kv_copy_fp16_v_b` (4 dispatches) where llama.cpp issues 2 fused dispatches.
- q8_1 redundancy: **VF 0** vs llama.cpp **0 (after caching)** — tied, but VF gets there by buffer-aliasing rather than caching; functional parity in the dense case.
- Per-dispatch helper command count: both 5. ✓

---

## 7. Host-Device Transfer

| Property | Value (line ref) |
|---|---|
| Weight Upload | once at model load via `loader.rs::new`. Pattern: 1 GiB `CpuToGpu` staging buffer (`loader.rs:25` `STAGING_BYTES = 1024 * 1024 * 1024`) → batched `cmd_copy_buffer` → DEVICE_LOCAL tensor buffer. ≤ 5 `one_shot` submits per the comment. |
| Weight Memory | **DEVICE_LOCAL** (`MemoryLocation::GpuOnly` in `loader.rs:tensor_create`). Not host-mapped. |
| Activation Transfer (CPU → GPU): decode embedding | direct write to host-visible `scratch_a` (`forward.rs:642` `scratch_a.write_bytes(embedding)`). The shader reads from `scratch_a` directly — no `cmd_copy_buffer` for the embedding |
| Activation Transfer (CPU → GPU): prefill embeddings | direct write to host-visible `batch_input` (`forward.rs:2877`), then 1 `cmd_copy_buffer batch_input → batch_residual` once at the start of the prefill (`forward.rs:2900-2906`) |
| Output Transfer (GPU → CPU): logits | `logits_buf` is `GpuToCpu` (HOST_READABLE) — `cmd_pipeline_barrier` with `HOST_READ` access mask before fence wait (`forward.rs:871` for forward_token), then `read_bytes()` returns persistent-mapped slice |
| Staging-Buffer (per-token) | **none** — `scratch_a` doubles as host-visible storage; no separate staging buffer |
| Transfer-Queue | **NONE** — single `compute_queue` does both compute and transfers |
| Transfers per Token (decode): host writes | **2** — embedding write to `scratch_a` (`bytemuck::cast_slice` + `write_bytes`), position write to `rope_pos_buf` (4 bytes) |
| Transfers per Token (decode): host reads | **1** — logits readback |
| Total transfers per Token (decode) | **2 writes + 1 read = 3 host memcpys, 0 vkCmdCopyBuffer** |
| `deferred_memcpy` equivalent | **NO** — host writes happen *before* the `one_shot` recording starts |

**vs llama.cpp:**
- Weight Memory: **VF DEVICE_LOCAL** vs **llama.cpp DEVICE_LOCAL+HOST_VISIBLE (ReBAR)**. VF requires staging copy at load time; llama.cpp can `memcpy` directly into mapped VRAM. Not a per-token cost.
- Transfer queue: **VF none** vs **llama.cpp separate transfer queue + timeline semaphore**. llama.cpp can overlap weight upload / KV streaming with compute on a separate queue. Per-token cost: minimal in steady state (Qwen3-8B is fully loaded), but matters for incremental KV writes if we ever wanted async.
- Transfers per Token: VF **3** vs llama.cpp **2** (= 1 token-id write + 1 logits read). The extra write in VF is `rope_pos_buf` — llama.cpp encodes positions inline in push constants.
- `deferred_memcpy`: VF emits host writes *before* `one_shot` (so they're synchronous-relative-to-recording but happen before submit). llama.cpp defers them until immediately before submit so CPU has more cycles to record dispatches. Negligible per-forward impact.
- **No major delta here** — both pre-allocate, both persistent-map, both use single-pass weight upload. Per-token transfer cost is small in both.

---

## 8. Quick-Win Zusammenfassung

Sofort sichtbare Gaps (Sprint 12C wird priorisieren):

### Tier 1 — high impact, scoped
1. **Barrier elision (Cat 2): VF emits ~432 barriers/forward, llama.cpp ~150.** VF could replicate the 3-flag dirty-bit elision (or per-buffer `last_writer` tracking) to drop ~280 barriers/forward → ~1.4 ms saved at ~5 µs/barrier. Effort: ~1–2 weeks (touch every `compute_barrier` callsite in forward.rs, plus a `BufferState` tracker on `Forward`).
2. **5-op fusion `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` (Cat 6):** VF currently does 4 dispatches (rope_q, rope_k, kv_copy_k, kv_copy_v) where llama.cpp does 2. New SPV `rms_norm_mul_rope_set_rows.comp` would write the post-rope output directly into the KV-cache slot. Saves 2 dispatches per layer × 36 = **72 dispatches per forward**. Effort: ~3–4 days.
3. **Submit splitting + pipelining (Cat 1):** VF blocks on a single big submit; llama.cpp pipelines 5–10 submits. The win here is CPU/GPU overlap on the *decode* path (where ~30 % of forward is CPU recording). Effort: medium — needs a streaming `forward_layer` API + a fence-pool. Likely 1–2 weeks. **Quantitative gain unclear without measurement** (it's the lever Sprint 12A flagged as central; depends on whether CPU-record time exceeds GPU-compute time for any submit chunk).

### Tier 2 — medium impact, scoped
4. **`MULTI_ADD` arity > 2 (Cat 6):** llama.cpp's `multi_add_rms` covers up to 9 inputs in one dispatch. VF's covers 2. Useful for MoE / gated paths but Qwen3-8B doesn't exercise it. Effort: 1 week. Skip for now.
5. **Per-vendor warptile overrides (Cat 5):** VF tunes for RDNA4 only. Not a gap for our target hardware. **Skip — explicit non-goal.**

### Tier 3 — low impact / questionable
6. **HashMap → indexed cache for descriptor sets (Cat 4):** marginal hash-overhead saving (~50 ns/dispatch × ~700 dispatches/forward = ~35 µs/forward). Not worth the rewrite.
7. **ReBAR weight memory (Cat 3+7):** load-time-only saving; doesn't affect steady-state. Skip.
8. **Transfer queue (Cat 7):** in steady state Qwen3-8B is fully loaded — separate transfer queue mostly useful for KV-cache streaming or incremental weight updates. Not currently a bottleneck. Skip.

### Sprint 12C should priority-rank Tier 1 by ROI

Best initial bet: **#1 (barrier elision)** — pure infrastructure change, no shader work, clear measurement target (~1 ms saved per forward), broadly orthogonal to the other levers. After that, **#2 (5-op fusion)** is the single largest dispatch-count reduction in the catalogue.
