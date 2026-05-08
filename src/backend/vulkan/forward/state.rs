//! Sprint 44B-1 — extracted from `forward/mod.rs` (pure code-move).
//!
//! Hosts the data definitions for the forward-pass orchestrator:
//! - SPV byte slices for harness-style FP8 / lm_head pipelines,
//! - the descriptor-set cache key (`BindingSignature`),
//! - public-facing stats / profile types (`ForwardStats`, `ForwardTokenProfile`,
//!   `DebugTarget`),
//! - per-token scratch grouping (`IntermediateSlot`),
//! - the long-lived `Forward` struct itself.
//!
//! The `impl Forward` block (and the per-shader dispatch helpers) stays in
//! `mod.rs`. To make that work across modules, fields that used to be private
//! are now `pub(super)`; their semantics are unchanged.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use ash::vk;
use ash::vk::Handle;

use super::super::buffers::GpuBuffer;
use super::super::gguf::ModelConfig;
use super::super::kv_cache::KvCache;
use super::super::profiler::{ShaderProfiler, TimingSample};
use super::harness::HarnessPipeline;

// Sprint 24-Inline DEBUG — per-channel FP8 GEMV variant SPV. Compiled
// by build.rs from `vk_shaders/mul_mat_vec_fp8_perchannel.comp`.
pub(super) const MUL_MAT_VEC_FP8_PERCHANNEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_fp8_perchannel.spv"));

// Sprint 35 — block-wise FP8 GEMV SPV for Qwen3-FP8 / DeepSeek-V3-FP8.
// Same 4-binding scheme as the per-channel variant; the difference is
// a 6 × u32 push-constant block carrying `block_size_n`, `block_size_k`,
// `num_kblocks`, and the indexing pattern `scale[(row/block_n)*num_kblocks + b]`.
pub(super) const MUL_MAT_VEC_FP8_BLOCKWISE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_fp8_blockwise.spv"));

// Sprint 36 — block-wise FP8 GEMM (BN=32) SPV. Replaces the Sprint 35
// GEMV-loop fallback. Push-constants carry `block_n`, `block_k`,
// `num_kblocks`; the kernel folds the per-block scale into the A-tile
// load before the WMMA chain.
pub(super) const MUL_COOPMAT_FP8_BN32_BLOCKWISE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_fp8_bn32_blockwise.spv"));

// Sprint 38 Part 2 — block-wise FP8 GEMM with NATIVE FP8 WMMA. Same
// descriptor set + push-constant layout as the BF16 cousin, but the
// shader uses `coopmat<floate4m3_t>` and applies the per-block scale
// via a partial accumulator multiplied by the scalar block scale,
// then summed into the total accumulator. Routed when
// `VF_FP8_NATIVE_WMMA=1` for block-wise FP8 models.
pub(super) const MUL_COOPMAT_FP8_NATIVE_BN32_BLOCKWISE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_fp8_native_bn32_blockwise.spv"));

// Sprint 29 — F16 GEMV SPV for the dedicated lm_head pipeline. Same
// SPV as the production `MulMatVecF16` registry entry; the harness
// pattern (PipelineCache::null + dedicated DSL/pool) bypasses the
// shared `vk::PipelineCache` to test whether cross-pipeline cache
// state is responsible for lm_head's runtime slowdown vs standalone.
pub(super) const MUL_MAT_VEC_F16_LMHEAD: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_f16.spv"));

// ---- Phase 5A-2 Stage 2D: descriptor-set cache key ----
//
// We cache a `vk::DescriptorSet` per unique binding signature to avoid
// the per-dispatch `vkAllocateDescriptorSets` + `vkUpdateDescriptorSets`
// pair on the decode hot path. Buffer handles, offsets and ranges are
// stable across tokens (only buffer *contents* change), so the same
// pre-written set can be re-bound for every token.
//
// The signature key is fixed-size (no allocation) so HashMap insert
// + lookup stay cheap. `MAX_BINDINGS_PER_SET = 8` covers our largest
// shader (`flash_attn_split.comp` uses 6 bindings).

pub(super) const MAX_BINDINGS_PER_SET: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct BindingEntry {
    binding: u32,
    buffer: u64,
    offset: u64,
    range: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BindingSignature {
    layout: u64,
    n: u8,
    entries: [BindingEntry; MAX_BINDINGS_PER_SET],
}

impl BindingSignature {
    pub(super) fn new(
        layout: vk::DescriptorSetLayout,
        bindings: &[(u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
    ) -> Self {
        assert!(
            bindings.len() <= MAX_BINDINGS_PER_SET,
            "BindingSignature: {} > MAX_BINDINGS_PER_SET={}",
            bindings.len(), MAX_BINDINGS_PER_SET,
        );
        let mut entries = [BindingEntry::default(); MAX_BINDINGS_PER_SET];
        for (i, &(b, buf, off, range)) in bindings.iter().enumerate() {
            entries[i] = BindingEntry {
                binding: b,
                buffer: buf.as_raw(),
                offset: off,
                range,
            };
        }
        Self {
            layout: layout.as_raw(),
            n: bindings.len() as u8,
            entries,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugTarget {
    AttnNorm,
    QProj,
    KProj,
    VProj,
    QNormRope,
    KNormRope,
    AttnOut,
}

pub struct ForwardStats {
    pub total: Duration,
    pub per_shader: BTreeMap<String, (Duration, u32)>,
    pub per_layer: Vec<Duration>,
    pub samples: Vec<TimingSample>,
}

/// CPU-side time breakdown of one `forward_token` call. Phase 5A-2
/// uses this to localise the per-token overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardTokenProfile {
    /// `embedding` upload + RoPE-pos write + descriptor-pool reset.
    pub pre_setup: Duration,
    /// `vkResetCommandBuffer` + fence reset.
    pub reset: Duration,
    /// `vkBeginCommandBuffer`.
    pub begin: Duration,
    /// The whole record block — every per-layer Rust setup (HashMap
    /// pipeline lookup, push-constants struct build, dispatch-dim
    /// math) plus every `vkCmdBindPipeline` / `vkCmdPushConstants` /
    /// `vkCmdDispatch` / `vkCmdPipelineBarrier` call.
    pub record: Duration,
    /// `vkEndCommandBuffer`.
    pub end: Duration,
    /// `vkQueueSubmit` (host-side; does not include GPU work).
    pub submit: Duration,
    /// `vkWaitForFences` — pure GPU wall-clock until the queue drains.
    pub gpu_wait: Duration,
    /// Logits readback (`read_bytes` from host-visible buffer + cast).
    pub readback: Duration,
}

impl ForwardTokenProfile {
    /// Sum of all CPU phases (everything except `gpu_wait`).
    pub fn cpu_total(&self) -> Duration {
        self.pre_setup + self.reset + self.begin + self.record + self.end
            + self.submit + self.readback
    }
    pub fn total(&self) -> Duration {
        self.cpu_total() + self.gpu_wait
    }
}

/// Sprint 15D — per-forward intermediate buffers grouped into a single
/// struct so the whole set can be double-buffered for async pipelined
/// decode. Two slots alternate per token; while the GPU runs CB[N]
/// reading from `slots[N % 2]`, the CPU records CB[N+1] referencing
/// `slots[(N+1) % 2]`. Without this double-buffering, the CPU host-write
/// to `scratch_a` / `rope_pos_buf` for token N+1 would race with the GPU's
/// read for token N, and the per-layer ping-pong target `scratch_b`
/// would be overwritten by CB[N+1]'s first dispatch before CB[N]'s last
/// dispatch finishes reading it.
///
/// Prefill (`dispatch_layer_batch`) is single-shot per prompt — async
/// pipelining adds nothing, so it always uses `slots[0]` for any decode
/// scratch it shares with the per-token path (currently just
/// `scratch_a` for the first layer's input upload).
pub struct IntermediateSlot {
    pub scratch_a: GpuBuffer,
    pub scratch_b: GpuBuffer,
    pub hidden_norm: GpuBuffer,
    pub q_buf: GpuBuffer,
    pub k_buf: GpuBuffer,
    pub v_buf: GpuBuffer,
    pub attn_out: GpuBuffer,
    pub o_buf: GpuBuffer,
    pub res1: GpuBuffer,
    pub gate_buf: GpuBuffer,
    pub up_buf: GpuBuffer,
    pub ffn_hidden: GpuBuffer,
    pub ffn_out: GpuBuffer,
    pub rope_pos_buf: GpuBuffer,
    pub fa_scratch_out: GpuBuffer,
    pub fa_scratch_max: GpuBuffer,
    pub fa_scratch_sum: GpuBuffer,
    /// Sprint 43D-3 — Per-Layer-Inputs staging buffer for Gemma-4 PLE.
    /// Filled per token from CPU-side `PleData::build_per_layer_inputs`
    /// (lookup + scale + per-layer rms_norm) and read by
    /// `dispatch_layer`'s PLE block at offset
    /// `layer * hps * 4`. Sized `num_layers × hps × 4` for Gemma-4
    /// (= 35 × 256 × 4 = 35.84 KB on E2B); 4 bytes (placeholder) for
    /// every other arch — never read on those paths.
    pub per_layer_inputs: GpuBuffer,
}

pub struct Forward {
    /// Sprint 15D — double-buffered per-forward scratch. `current_slot`
    /// alternates 0/1 per decode token. Prefill always uses slots[0].
    pub(super) slots: [IntermediateSlot; 2],
    pub(super) current_slot: usize,
    /// Sprint 15E — async decode infrastructure. Two CB+fence pairs
    /// alternating per token so `pre_record(CB[N+1])` can run on the
    /// CPU during `GPU(CB[N])`. Allocated alongside the existing
    /// CommandContext-driven serial path (which decode.rs picks
    /// between via `VULKANFORGE_DISABLE_ASYNC_DECODE=1`).
    pub(super) async_pool: vk::CommandPool,
    pub(super) async_cbs: [vk::CommandBuffer; 2],
    pub(super) async_fences: [vk::Fence; 2],
    /// Tracks which async slot has a CB pre-recorded but not yet
    /// submitted (for the rare `start_async` then EOS-cancel path).
    /// `None` means no CB is currently pre-recorded.
    pub(super) async_pending_record: Option<usize>,
    /// Sprint 19B-A — multi-submit prefill pacing. When
    /// `layers_per_submit < n_layers`, `prefill_batch` records the
    /// per-layer dispatches into N separate command buffers (drawn
    /// from `prefill_pool`) and submits them in sequence — only the
    /// last carrying `prefill_fence`. Queue-ordering on the same
    /// compute queue makes the in-between submits implicitly
    /// barrier-equivalent at queue boundaries, so no explicit
    /// inter-CB synchronization is required. Empty `prefill_cbs`
    /// (the default when no env override is set) means single-submit
    /// legacy path through `cmd_ctx.one_shot`.
    pub(super) prefill_pool: vk::CommandPool,
    pub(super) prefill_cbs: Vec<vk::CommandBuffer>,
    pub(super) prefill_fence: vk::Fence,
    pub(super) layers_per_submit: u32,
    pub(super) logits_buf: GpuBuffer,
    /// Sprint 27 — host-readable staging copy of `logits_buf`.
    /// `logits_buf` is now `GpuOnly` (fast GPU-local writes from
    /// `lm_head`); after each forward the CB ends with a tiny
    /// `cmd_copy_buffer` into this staging buffer, which is the one
    /// the host actually reads.
    pub(super) logits_staging: GpuBuffer,
    /// Sprint 40 Part 2 — host-readable staging copy of the
    /// post-norm hidden state. Used by the CPU `lm_head` offload
    /// (`VF_CPU_LM_HEAD=1`): when `model.cpu_lm_head` is `Some`,
    /// `dispatch_final` copies `hidden_norm` here instead of
    /// running the GPU lm_head GEMV. Always allocated (small,
    /// `hidden_dim * 4` bytes — ~16 KB on 8B, ~20 KB on 14B), so
    /// the same `Forward` instance handles both paths cleanly.
    pub(super) hidden_staging: GpuBuffer,
    // Always-bound dummies for unused descriptor slots.
    pub(super) fuse0: GpuBuffer,
    pub(super) fuse1: GpuBuffer,
    pub(super) rope_ff_buf: GpuBuffer,     // 16 B unused (has_ff=0)
    pub(super) rope_idx_buf: GpuBuffer,    // 16 B unused (set_rows_stride=0)
    /// Sprint 43D-4 — Gemma-4 V-norm gamma buffer. HF Gemma4 attention
    /// applies a parameterless `Gemma4RMSNorm(head_dim, with_scale=False)`
    /// to V after the V-projection: `v / sqrt(mean(v²) + eps)`. No
    /// weight tensor on disk (with_scale=False has no `weight`), so
    /// VF synthesises an all-ones gamma buffer here and feeds it to
    /// the existing run_rms_norm shader. Sized for the maximum
    /// per-layer head_dim (512 on Gemma-4-E2B full layers; 128 on
    /// Llama / Qwen). Filled with 1.0 floats once at construction.
    pub(super) vnorm_ones: GpuBuffer,

    pub kv_cache: KvCache,
    pub config: ModelConfig,

    pub(super) descriptor_pool: vk::DescriptorPool,
    /// Phase 5A-2 Stage 2D: descriptor-set cache for the
    /// Phase 5B.2 toggle for the batched-Q prefill attention path.
    /// `true` (default; `VULKANFORGE_BATCH_ATTN=0` opts out) replaces
    /// the per-token attention dispatch loop in `dispatch_layer_batch`
    /// with a single `flash_attn_batch` dispatch reading post-RoPE Q
    /// from `batch_q` and writing to `batch_attn_out`.
    pub(super) batch_attn_enabled: bool,

    /// Phase 6 v0.1.2 toggle for the mul_mm.comp port. `true`
    /// (default; `VULKANFORGE_USE_MUL_MM=0` opts out) routes the
    /// 7 prefill GEMMs (Q/K/V/O/gate/up/down) through mul_mm with
    /// FP32 activations, skipping the `quantize_q8_1` dispatch
    /// before each. mul_mmq stays as the gated fallback.
    pub(super) mul_mm_enabled: bool,

    /// Sprint 3A — opt-in coopmat (Q4_K dequant-fusion → FP8 WMMA)
    /// for `gemm_q` only. Default OFF; `VULKANFORGE_COOPMAT=1` flips
    /// to the coopmat path. The other 6 prefill GEMMs (K/V/O/gate/
    /// up/down) keep their mul_mmq routing in 3A — Sprint 3B widens
    /// the switch.
    pub(super) coopmat_q4k_enabled: bool,
    /// Sprint 11E — when ON, Q4_K mul_mm dispatches use the
    /// `mul_mm.comp + COOPMAT` SPV (KHR coopmat 16x16x16). Forces
    /// `mul_mm_enabled=true` because COOPMAT mul_mm reads FP32
    /// activations. Q6_K stays on scalar mul_mm (no COOPMAT SPV
    /// for Q6_K shipped). Opt-in via `VULKANFORGE_USE_MM_COOPMAT=1`.
    pub(super) mul_mm_coopmat_enabled: bool,

    /// Sprint 13C — when coopmat is on AND this flag is set, route the
    /// L-tile aligned coopmat dispatches through the f16-accumulator
    /// SPVs (ACC_TYPE = float16_t) instead of the FP32-accumulator SPVs.
    /// Halves accumulator VGPR pressure; precision-risk on long K
    /// reductions (K=12288 in gemm_down) so default OFF.
    /// `VULKANFORGE_COOPMAT_F16ACC=1` opts in.
    pub(super) mul_mm_coopmat_f16acc_enabled: bool,

    /// Sprint 14B — route decode K-quant GEMV through the `_subgroup`
    /// SPV variants which use `subgroupAdd` (Path A) instead of the
    /// LDS tree-reduction (Path B). Default ON; opt-out via
    /// `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`. Requires the
    /// requiredSubgroupSize=64 pipeline pin shipped in Sprint 14A.
    pub(super) mul_mat_vec_subgroup_enabled: bool,

    /// Sprint 3C — when coopmat is on, route through the FP8-narrowed
    /// padded variant instead of BF16. Default OFF;
    /// `VULKANFORGE_COOPMAT_FP8=1` flips it on. The two paths emit
    /// the same logits up to FP8 vs BF16 quantisation grid difference;
    /// FP8 saves ~50% LDS and a 4× VALU on the convert.
    pub(super) coopmat_fp8_enabled: bool,

    /// Sprint 7 / 7.5 — flips prefill_batch attention from
    /// `flash_attn_batch` (Br=1) to `flash_attn_tiled_br{N}` (BR
    /// queries per workgroup sharing K-tile loads). Set via
    /// `VULKANFORGE_FA_TILED=1`. Default OFF.
    pub(super) fa_tiled_enabled: bool,
    /// Sprint 7.5 — selects the BR variant when `fa_tiled_enabled` is
    /// on. Read from `VULKANFORGE_FA_BR` env var; valid values are
    /// 4, 8, 16. Default 16 (Sprint 7.5 sweep winner).
    pub(super) fa_tiled_br: u32,
    /// Sprint 7.6 — selects the BC (K-tile width) variant. Only
    /// applies when `fa_tiled_br = 16` (the only Br for which a
    /// Bc=32 SPV exists). Read from `VULKANFORGE_FA_BC`; valid
    /// values are 32, 64. Default 64 (matches Sprint 7.5 baseline
    /// until the Bc sweep proves Bc=32 is at least as good).
    pub(super) fa_tiled_bc: u32,
    /// v0.2 Sprint 10C — coopmat flash-attention v1. When set,
    /// `run_flash_attn_tiled`'s shader selector replaces the scalar
    /// FlashAttnTiledBr16Bc32 SPV with FlashAttnCoopmat (QK via
    /// VK_KHR_cooperative_matrix WMMA, softmax + PV stay scalar).
    /// Forces Br=16/Bc=16 (the only shape the coopmat SPV ships).
    ///
    /// v0.2 Sprint 10E — flipped to default ON. Opt-out:
    /// `VULKANFORGE_COOPMAT_ATTN=0` for the original scalar
    /// flash_attn_tiled path (Sprint 7.6/8a default). Pairs with the
    /// FP16 KV default (Sprint 9d.3): when both are ON, the selector
    /// picks `FlashAttnCoopmatFp16Kv`.
    pub(super) coopmat_attn_enabled: bool,

    /// `forward_token` hot path. When `cache_enabled` is true (env
    /// `VULKANFORGE_CB_REUSE=1` at construction), `alloc_or_get_set`
    /// reuses sets across tokens instead of resetting the pool.
    pub(super) set_cache: HashMap<BindingSignature, vk::DescriptorSet>,
    pub(super) cache_enabled: bool,
    pub profiler: Option<ShaderProfiler>,

    // Sprint 24-Inline — Step 0: harness-style FP8 GEMV per-channel
    // resources, freshly created at Forward construction with the
    // perchannel SPV variant + null pipeline cache + dedicated
    // descriptor pool. Used for FP8 GEMV when scale_buffer is Some.
    //
    // Sprint 44B-2 — the {module, dsl, layout, pipeline, pool} quintuple
    // is bundled into a `HarnessPipeline` (see `harness.rs`); the
    // descriptor-set cache stays here so dispatch wrappers can `clear()`
    // it on prefill without touching the harness.
    pub(super) fp8pc: HarnessPipeline,
    /// Sprint 30 — descriptor-set cache for `run_gemv_fp8_perchannel`.
    /// Key: (weight, input, output, scale) buffer handles as `u64`.
    /// Hit  → reuse the cached set (no `vkAllocate` / `vkUpdate` calls).
    /// Miss → allocate + write + insert.
    /// Pool is no longer reset per-decode-token, so sets stay valid
    /// across the whole decode run; cleared on prefill via
    /// `reset_descriptor_pool_and_cache`. With ~336 unique keys per
    /// forward × 2 async slots, a pool of 1024 sets covers the
    /// worst case (vs 524288 in v0.3.5 — that was for the no-cache,
    /// every-call-fresh-alloc pattern).
    pub(super) fp8pc_ds_cache: HashMap<(u64, u64, u64, u64), vk::DescriptorSet>,

    // Sprint 35 — block-wise FP8 GEMV resources. Mirror of fp8pc with
    // a different SPV (`mul_mat_vec_fp8_blockwise`) and a 6-u32 push
    // constant block (vs perchannel's 13-u32 MatVecPushConstants).
    // 4 storage-buffer bindings identical to perchannel so the loader
    // and the routing wrapper can share the descriptor-set cache key
    // shape. Activated when a layer weight has `scale_block.is_some()`.
    pub(super) fp8bw: HarnessPipeline,
    pub(super) fp8bw_ds_cache: HashMap<(u64, u64, u64, u64), vk::DescriptorSet>,

    // Sprint 36 — block-wise FP8 GEMM resources (BN=32 prefill kernel).
    // Same 4-binding descriptor scheme as the GEMV variant; differs in
    // the SPV (`mul_coopmat_fp8_bn32_blockwise`) and a 9-u32 push
    // constant block. Replaces the Sprint 35 GEMV-loop fallback in
    // `dispatch_layer_batch`.
    pub(super) fp8bwgemm: HarnessPipeline,
    pub(super) fp8bwgemm_ds_cache: HashMap<(u64, u64, u64, u64), vk::DescriptorSet>,

    // Sprint 38 Part 2 — block-wise FP8 GEMM with native FP8 WMMA.
    // Shares dsl/pipeline_layout/pool with the BF16 cousin above; only
    // the shader module + pipeline differ. Routed when
    // `VF_FP8_NATIVE_WMMA=1` for block-wise FP8 models.
    pub(super) fp8bwgemm_native_shader_module: vk::ShaderModule,
    pub(super) fp8bwgemm_native_pipeline: vk::Pipeline,

    // Sprint 29 — harness-style dedicated F16 GEMV resources for the
    // lm_head dispatch (M=152064, K=5120 on Qwen2.5-14B-FP8). Mirrors
    // the production registry entry's spec / required-subgroup / SPV,
    // but uses `PipelineCache::null` and a dedicated descriptor pool
    // so the lm_head pipeline never shares ACO codegen state with the
    // 102-pipeline shared `vk::PipelineCache`. Runtime measurements
    // showed lm_head at ~30 ms via the shared registry vs ~3 ms in a
    // standalone bench (Sprints 26-28B). This path tests whether the
    // shared cache is the cause; if it is, the harness pattern (same
    // approach that fixed Sprint 24-Inline's per-channel FP8 GEMV)
    // closes the gap.
    pub(super) lmhead: HarnessPipeline,

    pub(super) rope_theta_scale: f32,
    pub(super) attn_scale: f32,

    // ---- Phase-3E batch-prefill scratch ----
    // Allocated once in `new` based on `max_prefill_tokens`. Memory
    // budget at the default (256 tokens) is ~60 MB — well within the
    // 10 GiB free after Qwen3-8B weight upload.
    pub max_prefill_tokens: u32,
    /// `[max_pp × hidden_dim]` host-visible — caller writes the
    /// per-token f32 embeddings here before dispatching `prefill_batch`.
    pub(super) batch_input: GpuBuffer,
    /// `[max_pp × hidden_dim]` ping-pong target for the per-layer
    /// residual chain.
    pub(super) batch_residual: GpuBuffer,
    /// `[max_pp × hidden_dim]` post-RMSNorm activations.
    pub(super) batch_norm: GpuBuffer,
    /// `[max_pp × hidden_dim]` Q8_1-quantised activations (input to
    /// every GEMM in `prefill_batch`).
    pub(super) batch_q8: GpuBuffer,
    /// Q-projection batch output: `[max_pp × n_heads × head_dim]` f32,
    /// laid out as N×M row-major (see Phase 3D §4.1).
    pub(super) batch_q: GpuBuffer,
    pub(super) batch_k: GpuBuffer,
    pub(super) batch_v: GpuBuffer,
    pub(super) batch_attn_out: GpuBuffer,
    pub(super) batch_o: GpuBuffer,
    pub(super) batch_gate: GpuBuffer,
    pub(super) batch_up: GpuBuffer,
    pub(super) batch_ffn_hidden: GpuBuffer,
    pub(super) batch_ffn_out: GpuBuffer,

    /// Sprint 12D — barrier elision via dirty-flag tracking. The set
    /// holds `vk::Buffer` raw handles that have been written since the
    /// last `compute_barrier`. `maybe_compute_barrier(reads)` skips the
    /// barrier if none of the read buffers is in the dirty set.
    /// Cleared by every barrier issuance (since `compute_barrier` is a
    /// global `VkMemoryBarrier`, one barrier syncs everything).
    /// `VULKANFORGE_DISABLE_BARRIER_ELISION=1` falls back to the legacy
    /// always-barrier path for debugging / parity comparison.
    pub(super) pending_writes: std::collections::HashSet<u64>,
    pub(super) elision_disabled: bool,
    pub(super) barrier_stats_checked: u64,
    pub(super) barrier_stats_issued: u64,
}
