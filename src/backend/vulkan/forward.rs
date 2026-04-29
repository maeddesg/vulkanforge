//! Forward-pass orchestration for Qwen3 decode.
//!
//! Phase 2C. One [`Forward`] instance owns:
//! - per-token scratch buffers (ping-pong + per-projection slots),
//! - the K/V cache,
//! - one long-lived descriptor pool, reset between forwards,
//! - tiny RoPE auxiliary buffers (pos / ff / indices),
//! - an optional [`ShaderProfiler`].
//!
//! [`Forward::forward_token`] dispatches the embedding lookup → 36
//! transformer layers → final RMSNorm → LM head and reads the logits
//! back. Each shader path gets a method on `Forward` that allocates
//! a descriptor set from the pool, writes it, and dispatches.
//!
//! Layer ordering (Qwen3 with QK-norm):
//! ```text
//! input ─→ attn_norm ─→ Wq/Wk/Wv (3× GEMV)
//!         q ─→ q_norm ─→ RoPE-NeoX
//!         k ─→ k_norm ─→ RoPE-NeoX  ─→ KV cache (pos-major copy)
//!         v ────────────────────────→ KV cache
//!         attention (scalar_attn) ──→ Wo ─→ residual1
//!         ffn_norm ─→ gate, up (2× GEMV) ─→ silu(gate)·up ─→ Wdown
//!         residual1 + Wdown_out ──→ next-layer input
//! ```

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use ash::vk;
use ash::vk::Handle;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::buffers::GpuBuffer;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::gguf::{GgmlType, ModelConfig};
use super::kv_cache::KvCache;
use super::loader::LoadedModel;
use super::pipeline::{
    FlashAttnBatchPushConstants, FlashAttnReducePushConstants, FlashAttnSplitPushConstants,
    GenericBinaryPushConstants, KvCopyFp16PushConstants, MatVecPushConstants,
    CoopmatPushConstants, MmqPushConstants, MultiAddRmsPushConstants,
    Q8_1QuantizePushConstants, RmsNormMulRopePushConstants, RopePushConstants,
    ScalarAttnPushConstants, SwigluPushConstants,
};
use super::pipeline_registry::PipelineRegistry;
use super::profiler::{ShaderProfiler, TimingSample};
use super::shaders::ShaderId;

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

const MAX_BINDINGS_PER_SET: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct BindingEntry {
    binding: u32,
    buffer: u64,
    offset: u64,
    range: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BindingSignature {
    layout: u64,
    n: u8,
    entries: [BindingEntry; MAX_BINDINGS_PER_SET],
}

impl BindingSignature {
    fn new(
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

pub struct Forward {
    // Scratch (per-token reuse).
    scratch_a: GpuBuffer,
    scratch_b: GpuBuffer,
    hidden_norm: GpuBuffer,
    q_buf: GpuBuffer,
    k_buf: GpuBuffer,
    v_buf: GpuBuffer,
    attn_out: GpuBuffer,
    o_buf: GpuBuffer,
    res1: GpuBuffer,
    gate_buf: GpuBuffer,
    up_buf: GpuBuffer,
    ffn_hidden: GpuBuffer,
    ffn_out: GpuBuffer,
    logits_buf: GpuBuffer,
    // Always-bound dummies for unused descriptor slots.
    fuse0: GpuBuffer,
    fuse1: GpuBuffer,
    rope_pos_buf: GpuBuffer,    // 4 B host-visible: writes the current position
    rope_ff_buf: GpuBuffer,     // 16 B unused (has_ff=0)
    rope_idx_buf: GpuBuffer,    // 16 B unused (set_rows_stride=0)

    pub kv_cache: KvCache,
    pub config: ModelConfig,

    descriptor_pool: vk::DescriptorPool,
    /// Phase 5A-2 Stage 2D: descriptor-set cache for the
    /// Phase 5B.2 toggle for the batched-Q prefill attention path.
    /// `true` (default; `VULKANFORGE_BATCH_ATTN=0` opts out) replaces
    /// the per-token attention dispatch loop in `dispatch_layer_batch`
    /// with a single `flash_attn_batch` dispatch reading post-RoPE Q
    /// from `batch_q` and writing to `batch_attn_out`.
    batch_attn_enabled: bool,

    /// Phase 6 v0.1.2 toggle for the mul_mm.comp port. `true`
    /// (default; `VULKANFORGE_USE_MUL_MM=0` opts out) routes the
    /// 7 prefill GEMMs (Q/K/V/O/gate/up/down) through mul_mm with
    /// FP32 activations, skipping the `quantize_q8_1` dispatch
    /// before each. mul_mmq stays as the gated fallback.
    mul_mm_enabled: bool,

    /// Sprint 3A — opt-in coopmat (Q4_K dequant-fusion → FP8 WMMA)
    /// for `gemm_q` only. Default OFF; `VULKANFORGE_COOPMAT=1` flips
    /// to the coopmat path. The other 6 prefill GEMMs (K/V/O/gate/
    /// up/down) keep their mul_mmq routing in 3A — Sprint 3B widens
    /// the switch.
    coopmat_q4k_enabled: bool,
    /// Sprint 11E — when ON, Q4_K mul_mm dispatches use the
    /// `mul_mm.comp + COOPMAT` SPV (KHR coopmat 16x16x16). Forces
    /// `mul_mm_enabled=true` because COOPMAT mul_mm reads FP32
    /// activations. Q6_K stays on scalar mul_mm (no COOPMAT SPV
    /// for Q6_K shipped). Opt-in via `VULKANFORGE_USE_MM_COOPMAT=1`.
    mul_mm_coopmat_enabled: bool,

    /// Sprint 3C — when coopmat is on, route through the FP8-narrowed
    /// padded variant instead of BF16. Default OFF;
    /// `VULKANFORGE_COOPMAT_FP8=1` flips it on. The two paths emit
    /// the same logits up to FP8 vs BF16 quantisation grid difference;
    /// FP8 saves ~50% LDS and a 4× VALU on the convert.
    coopmat_fp8_enabled: bool,

    /// Sprint 7 / 7.5 — flips prefill_batch attention from
    /// `flash_attn_batch` (Br=1) to `flash_attn_tiled_br{N}` (BR
    /// queries per workgroup sharing K-tile loads). Set via
    /// `VULKANFORGE_FA_TILED=1`. Default OFF.
    fa_tiled_enabled: bool,
    /// Sprint 7.5 — selects the BR variant when `fa_tiled_enabled` is
    /// on. Read from `VULKANFORGE_FA_BR` env var; valid values are
    /// 4, 8, 16. Default 16 (Sprint 7.5 sweep winner).
    fa_tiled_br: u32,
    /// Sprint 7.6 — selects the BC (K-tile width) variant. Only
    /// applies when `fa_tiled_br = 16` (the only Br for which a
    /// Bc=32 SPV exists). Read from `VULKANFORGE_FA_BC`; valid
    /// values are 32, 64. Default 64 (matches Sprint 7.5 baseline
    /// until the Bc sweep proves Bc=32 is at least as good).
    fa_tiled_bc: u32,
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
    coopmat_attn_enabled: bool,

    /// `forward_token` hot path. When `cache_enabled` is true (env
    /// `VULKANFORGE_CB_REUSE=1` at construction), `alloc_or_get_set`
    /// reuses sets across tokens instead of resetting the pool.
    set_cache: HashMap<BindingSignature, vk::DescriptorSet>,
    cache_enabled: bool,
    pub profiler: Option<ShaderProfiler>,

    rope_theta_scale: f32,
    attn_scale: f32,

    // ---- Phase-4C split-K attention scratch ----
    // Sized once at construction for the worst-case max_seq_len.
    // Written by `flash_attn_split.comp` and read by
    // `flash_attn_reduce.comp`. Single-buffer per kind — only one
    // attention dispatch is in flight at a time.
    fa_scratch_out: GpuBuffer,
    fa_scratch_max: GpuBuffer,
    fa_scratch_sum: GpuBuffer,

    // ---- Phase-3E batch-prefill scratch ----
    // Allocated once in `new` based on `max_prefill_tokens`. Memory
    // budget at the default (256 tokens) is ~60 MB — well within the
    // 10 GiB free after Qwen3-8B weight upload.
    pub max_prefill_tokens: u32,
    /// `[max_pp × hidden_dim]` host-visible — caller writes the
    /// per-token f32 embeddings here before dispatching `prefill_batch`.
    batch_input: GpuBuffer,
    /// `[max_pp × hidden_dim]` ping-pong target for the per-layer
    /// residual chain.
    batch_residual: GpuBuffer,
    /// `[max_pp × hidden_dim]` post-RMSNorm activations.
    batch_norm: GpuBuffer,
    /// `[max_pp × hidden_dim]` Q8_1-quantised activations (input to
    /// every GEMM in `prefill_batch`).
    batch_q8: GpuBuffer,
    /// Q-projection batch output: `[max_pp × n_heads × head_dim]` f32,
    /// laid out as N×M row-major (see Phase 3D §4.1).
    batch_q: GpuBuffer,
    batch_k: GpuBuffer,
    batch_v: GpuBuffer,
    batch_attn_out: GpuBuffer,
    batch_o: GpuBuffer,
    batch_gate: GpuBuffer,
    batch_up: GpuBuffer,
    batch_ffn_hidden: GpuBuffer,
    batch_ffn_out: GpuBuffer,
}

impl Forward {
    pub fn new(
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        kv_cache: KvCache,
        config: ModelConfig,
        profiler: Option<ShaderProfiler>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Sprint 5 — bump default from 256 to 1024 to lift the
        // pp>256 cliff (decode.rs:429 routes prefill_len >
        // max_prefill_tokens through forward_token, collapsing
        // throughput to ~90 tok/s = decode rate). VRAM cost at
        // Qwen3-8B dims: ~240 MB vs ~60 MB. Override via env var
        // VULKANFORGE_MAX_PREFILL when callers need a smaller cap
        // (low-VRAM hosts, CI VMs).
        let max_pp: u32 = std::env::var("VULKANFORGE_MAX_PREFILL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        Self::new_with_prefill(dev, allocator, kv_cache, config, profiler, max_pp)
    }

    /// Like `new` but accepts an explicit `max_prefill_tokens` cap so
    /// tests can keep VRAM small.
    pub fn new_with_prefill(
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        kv_cache: KvCache,
        config: ModelConfig,
        profiler: Option<ShaderProfiler>,
        max_prefill_tokens: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = &dev.device;
        let mut mk_storage = |size: u64, location: MemoryLocation, name: &str| {
            GpuBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                location,
                name,
            )
        };

        let hidden_bytes = (config.hidden_dim as u64) * 4;
        let q_bytes = (config.n_heads as u64) * (config.head_dim as u64) * 4;
        let kv_bytes = (config.n_kv_heads as u64) * (config.head_dim as u64) * 4;
        let ffn_bytes = (config.ffn_dim as u64) * 4;
        let logits_bytes = (config.vocab_size as u64) * 4;

        let scratch_a = mk_storage(hidden_bytes, MemoryLocation::CpuToGpu, "scratch_a")?;
        let scratch_b = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "scratch_b")?;
        let hidden_norm = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "hidden_norm")?;
        let q_buf = mk_storage(q_bytes, MemoryLocation::GpuOnly, "q_buf")?;
        let k_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, "k_buf")?;
        let v_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, "v_buf")?;
        let attn_out = mk_storage(q_bytes, MemoryLocation::GpuOnly, "attn_out")?;
        let o_buf = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "o_buf")?;
        let res1 = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "res1")?;
        let gate_buf = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "gate_buf")?;
        let up_buf = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "up_buf")?;
        let ffn_hidden = mk_storage(ffn_bytes, MemoryLocation::GpuOnly, "ffn_hidden")?;
        let ffn_out = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, "ffn_out")?;
        let logits_buf = mk_storage(logits_bytes, MemoryLocation::GpuToCpu, "logits_buf")?;
        let fuse0 = mk_storage(16, MemoryLocation::GpuOnly, "fuse0_dummy")?;
        let fuse1 = mk_storage(16, MemoryLocation::GpuOnly, "fuse1_dummy")?;
        // Phase 3E drift-fix: rope_pos_buf must hold one slot per
        // prefill token (otherwise the per-token host writes during
        // command-buffer recording all collapse to the last value
        // by the time the GPU executes — every token gets the same
        // RoPE position). Forward_token uses slot 0 of this buffer;
        // prefill_batch writes positions 0..seq_len into slots
        // 0..seq_len-1 before submit and binds with per-token offset.
        let rope_pos_buf = mk_storage(
            (max_prefill_tokens.max(1) as u64) * 4,
            MemoryLocation::CpuToGpu, "rope_pos",
        )?;

        // Phase-4C scratch for split-K attention. Sized for the
        // worst case at max_seq_len; per-call dispatches use only the
        // prefix that matches the current `n_tiles`.
        const FA_TILE: u64 = 64;
        let fa_max_tiles =
            (kv_cache.config.max_seq_len as u64 + FA_TILE - 1) / FA_TILE;
        let fa_scratch_out_bytes = (config.n_heads as u64)
            * fa_max_tiles
            * (config.head_dim as u64)
            * 4;
        let fa_scratch_red_bytes = (config.n_heads as u64) * fa_max_tiles * 4;
        let fa_scratch_out = mk_storage(
            fa_scratch_out_bytes,
            MemoryLocation::GpuOnly,
            "fa_scratch_out",
        )?;
        let fa_scratch_max = mk_storage(
            fa_scratch_red_bytes,
            MemoryLocation::GpuOnly,
            "fa_scratch_max",
        )?;
        let fa_scratch_sum = mk_storage(
            fa_scratch_red_bytes,
            MemoryLocation::GpuOnly,
            "fa_scratch_sum",
        )?;
        let rope_ff_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_ff_dummy")?;
        let rope_idx_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_idx_dummy")?;

        // ---- Phase-3E batch-prefill scratch buffers ----
        // Allocations sized for the largest prompt we'll batch in one
        // submit. ~60 MB at the 256-token default — comfortable within
        // the ~10 GiB free after weights.
        //
        // Sprint 3C — round max_pp up to a multiple of 16 so the
        // coopmat path's N-padding fill stays in-bounds. Without this
        // padding, `cmd_fill_buffer` for rows seq_len..pad_to_tile(seq_len, 16)
        // can write past the buffer end (causing subtle memory
        // corruption that drifts the long-tail logit ranking).
        let max_pp_raw = max_prefill_tokens as u64;
        let max_pp = (max_pp_raw + 15) / 16 * 16;
        let pp_hidden = max_pp * (config.hidden_dim as u64) * 4;
        let pp_kv     = max_pp * (config.n_kv_heads as u64) * (config.head_dim as u64) * 4;
        let pp_q      = max_pp * (config.n_heads as u64)   * (config.head_dim as u64) * 4;
        let pp_ffn    = max_pp * (config.ffn_dim as u64) * 4;
        // Q8_1_x4 packs 128 elements / 144 bytes.
        let pp_q8     = max_pp * (config.hidden_dim as u64) / 128 * 144;
        let pp_q8_ffn = max_pp * (config.ffn_dim as u64)    / 128 * 144;
        let pp_q8_max = pp_q8.max(pp_q8_ffn);

        let batch_input      = mk_storage(pp_hidden, MemoryLocation::CpuToGpu, "batch_input")?;
        let batch_residual   = mk_storage(pp_hidden, MemoryLocation::GpuOnly,  "batch_residual")?;
        let batch_norm       = mk_storage(pp_hidden, MemoryLocation::GpuOnly,  "batch_norm")?;
        let batch_q8         = mk_storage(pp_q8_max, MemoryLocation::GpuOnly,  "batch_q8")?;
        let batch_q          = mk_storage(pp_q,      MemoryLocation::GpuOnly,  "batch_q")?;
        let batch_k          = mk_storage(pp_kv,     MemoryLocation::GpuOnly,  "batch_k")?;
        let batch_v          = mk_storage(pp_kv,     MemoryLocation::GpuOnly,  "batch_v")?;
        let batch_attn_out   = mk_storage(pp_q,      MemoryLocation::GpuOnly,  "batch_attn_out")?;
        let batch_o          = mk_storage(pp_hidden, MemoryLocation::GpuOnly,  "batch_o")?;
        let batch_gate       = mk_storage(pp_ffn,    MemoryLocation::GpuOnly,  "batch_gate")?;
        let batch_up         = mk_storage(pp_ffn,    MemoryLocation::GpuOnly,  "batch_up")?;
        let batch_ffn_hidden = mk_storage(pp_ffn,    MemoryLocation::GpuOnly,  "batch_ffn_hidden")?;
        let batch_ffn_out    = mk_storage(pp_hidden, MemoryLocation::GpuOnly,  "batch_ffn_out")?;

        // Descriptor pool sized for one prefill_batch submit at
        // max_prefill_tokens, plus the per-token forward fallback.
        // Per-layer set count: 14 GEMM/quantize/norm + 5 × seq_len
        // attention-loop sets + 2 residuals.
        // For max_pp = 256: (14 + 5*256 + 2) × 36 = 46656; round up.
        let per_layer_sets = 16 + 5 * max_prefill_tokens.max(1);
        // Phase 5A-2 Stage 2D: when CB-reuse is on, the pool is no
        // longer reset between forwards — sets accumulate in the cache
        // for the lifetime of the Forward instance. Headroom of 4×
        // covers the worst case (a prefill_batch run that fills the
        // attention-loop bucket, followed by many decode tokens that
        // populate the per-(layer,slot) bucket).
        let dispatches = (per_layer_sets * config.n_layers + 64) * 4;
        let max_descriptors = dispatches * 8;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_descriptors,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(dispatches)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Phase 5A-3: descriptor-set cache is now ON by default.
        // `VULKANFORGE_CB_REUSE=0` (or `false`) opts back out for
        // debugging or A/B comparisons; any other value (or unset)
        // keeps the cache enabled. The Stage 2D parity test
        // (`phase5a_cb_reuse_parity_qwen3`) confirmed bit-exact output
        // against the direct path, so the cache is safe as default.
        let cache_enabled = match std::env::var("VULKANFORGE_CB_REUSE") {
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
            _ => true,
        };

        // Phase 5B.2: batched-Q prefill attention is ON by default; set
        // `VULKANFORGE_BATCH_ATTN=0` to fall back to the per-token
        // attention loop. The phase5b_2 parity test confirms argmax
        // identity against the per-token path on all 4 supported models.
        let batch_attn_enabled = match std::env::var("VULKANFORGE_BATCH_ATTN") {
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
            _ => true,
        };

        // Phase 6 v0.1.2: mul_mm.comp port is OFF by default — the
        // Phase 7 — mul_mm.comp is now bit-exact across all 11 unit
        // tests + the phase3e/5b2 regressions (top-5 = 5/5 vs the
        // per-token GEMV path). Two bugs were fixed:
        //   (1) BLOCK_SIZE 128 → 256: NUM_WARPS = BLOCK_SIZE/WARP must
        //       cover (BM/WM)*(BN/WN) warp tiles. With WARP=64, BM=BN=64,
        //       WM=WN=32 we need 4 warps. The previous 128 silently
        //       dropped cols [WN, BN) of every output tile because
        //       warp_c (= warp_i / (BM/WM)) was always 0. Also affected
        //       mul_mmq, which was producing wrong results for prompts
        //       > 32 tokens — undetected because regression tests use
        //       short prompts.
        //   (2) Q6_K LOAD_VEC_A 4 → 2: the Q6_K branch in
        //       mul_mm_funcs.glsl emits one vec2 (2 weights) per idx
        //       but Q4_K's branch emits two (4 weights). Compiling
        //       Q6_K with LOAD_VEC_A=4 left half of buf_a uninitialised
        //       → NaN logits at scale.
        // Default stays OFF because mul_mmq is ~45 % faster at prefill
        // (Q8_1 activations vs FP32: 4× less B-bandwidth into LDS).
        // Opt in via VULKANFORGE_USE_MUL_MM=1 when bit-exact FP32 input
        // matters (e.g. validating quant-induced drift).
        // Sprint 11E — VULKANFORGE_USE_MM_COOPMAT implies VULKANFORGE_USE_MUL_MM
        // because the COOPMAT path lives in mul_mm.comp (FP32 activations),
        // not mul_mmq.comp.
        let mul_mm_coopmat_enabled = match std::env::var("VULKANFORGE_USE_MM_COOPMAT") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };
        let mul_mm_enabled = mul_mm_coopmat_enabled || match std::env::var("VULKANFORGE_USE_MUL_MM") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };

        // Sprint 3A — Q4_K coopmat for gemm_q only. Default OFF until
        // logits-parity is established at scale.
        let coopmat_q4k_enabled = match std::env::var("VULKANFORGE_COOPMAT") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };

        // Sprint 3C — FP8 narrow inside the coopmat path. Default
        // OFF; only meaningful when coopmat itself is enabled.
        // Sprint 7 / 7.5 / 7.6 / 8a — Br>1 tiled-Q flash-attention is
        // now the default path. Empirically wins on every pp ≥ 128
        // (+11% to +164% vs Br=1) and is within mess-rauschen at
        // pp ≤ 64 (-2 to -4%). Opt-out via VULKANFORGE_FA_TILED=0
        // for the (rare) ultra-short-prompt latency cases.
        let fa_tiled_enabled = match std::env::var("VULKANFORGE_FA_TILED") {
            Ok(s) => s != "0",
            Err(_) => true,
        };
        // Sprint 7.5 — pick which Br variant to dispatch when fa_tiled
        // is on. Default 16 (Sprint-7.5 sweep winner: +138% at pp=1024
        // vs Br=1, beats Br=4/Br=8 across every measured pp). Falls
        // back to 16 for unknown / missing values.
        let fa_tiled_br: u32 = match std::env::var("VULKANFORGE_FA_BR")
            .ok().and_then(|s| s.parse::<u32>().ok())
        {
            Some(4)  => 4,
            Some(8)  => 8,
            Some(16) => 16,
            _        => 16,
        };
        // Sprint 7.6 — pick which Bc variant for Br=16. Default 32
        // (Sprint 7.6 sweep winner: +1% to +11% single-shot, +15%
        // to +19% chunked vs Bc=64). Only honoured when fa_tiled_br
        // == 16 because that's the only Br with a Bc=32 SPV.
        let fa_tiled_bc: u32 = match std::env::var("VULKANFORGE_FA_BC")
            .ok().and_then(|s| s.parse::<u32>().ok())
        {
            Some(32) => 32,
            Some(64) => 64,
            _        => 32,
        };

        let coopmat_fp8_enabled = match std::env::var("VULKANFORGE_COOPMAT_FP8") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };

        // v0.2 Sprint 10C — coopmat flash-attention v1 (QK coopmat,
        // softmax + PV scalar). Forces fa_tiled_br=16 + fa_tiled_bc=16
        // because that's the only shape the coopmat SPV ships.
        //
        // v0.2 Sprint 10E — flipped to **default ON** after the
        // QK-only coopmat path showed a +7-86% pp-sweep win across
        // pp ∈ [128, 2048] under COOPMAT_ATTN=1 (Sprint 10C bench).
        // Sprint 10D's PV-coopmat extension was reverted as a
        // negative result; the production path stays at QK-only
        // coopmat which we now flip default. Opt-out: set
        // VULKANFORGE_COOPMAT_ATTN=0 for the original scalar
        // flash_attn_tiled path.
        let coopmat_attn_enabled = match std::env::var("VULKANFORGE_COOPMAT_ATTN") {
            Ok(s) if s == "0" => false, // explicit opt-out
            _ => true,                  // default ON
        };

        // Note for tests: callers that need to override the env-var
        // pick can use `set_cache_enabled` / `set_batch_attn_enabled`
        // after construction.

        let attn_scale = 1.0_f32 / (config.head_dim as f32).sqrt();
        let rope_theta_scale =
            (1.0_f32 / config.rope_freq_base).powf(2.0 / config.head_dim as f32);

        Ok(Self {
            scratch_a, scratch_b, hidden_norm,
            q_buf, k_buf, v_buf, attn_out, o_buf, res1,
            gate_buf, up_buf, ffn_hidden, ffn_out,
            logits_buf,
            fuse0, fuse1,
            rope_pos_buf, rope_ff_buf, rope_idx_buf,
            kv_cache, config,
            descriptor_pool,
            set_cache: HashMap::new(),
            cache_enabled,
            batch_attn_enabled,
            mul_mm_enabled,
            coopmat_q4k_enabled,
            mul_mm_coopmat_enabled,
            coopmat_fp8_enabled,
            fa_tiled_enabled,
            fa_tiled_br,
            fa_tiled_bc,
            coopmat_attn_enabled,
            profiler,
            rope_theta_scale, attn_scale,
            fa_scratch_out, fa_scratch_max, fa_scratch_sum,
            max_prefill_tokens,
            batch_input, batch_residual, batch_norm, batch_q8,
            batch_q, batch_k, batch_v, batch_attn_out, batch_o,
            batch_gate, batch_up, batch_ffn_hidden, batch_ffn_out,
        })
    }

    /// Phase 5A-2 drill-down: same wall-time semantics as
    /// `forward_token_profile` but additionally captures, INSIDE the
    /// command-buffer record block, per-layer wall time and a tally of
    /// `dispatch_final`'s wall time. Use this to decide whether
    /// command-buffer reuse should target the Rust-side per-layer
    /// setup (HashMap lookup, push-constants struct build) or the raw
    /// `vkCmd*` call cost.
    pub fn forward_token_profile_layers(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
    ) -> Result<(ForwardTokenProfile, Vec<Duration>, Duration), Box<dyn std::error::Error>> {
        use std::time::Instant;
        let pre_start = Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err("embedding length mismatch".into());
        }
        self.scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Phase 5A-2 Stage 2D: skip the per-forward pool reset when
        // CB-reuse is on — cached sets stay alive across tokens.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }
        let pre_setup = pre_start.elapsed();

        let n_layers = self.config.n_layers as usize;
        let mut per_layer: Vec<Duration> = Vec::with_capacity(n_layers);
        let mut final_dispatch = Duration::ZERO;

        let timings = cmd_ctx.one_shot_profiled(&dev.device, dev.compute_queue, |cmd| {
            let mut input = self.scratch_a.handle;
            let mut output = self.scratch_b.handle;
            for layer in 0..self.config.n_layers {
                let t = Instant::now();
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                per_layer.push(t.elapsed());
                std::mem::swap(&mut input, &mut output);
            }
            let t = Instant::now();
            self.dispatch_final(dev, registry, cmd, model, input);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.logits_buf.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[], &[post], &[],
                );
            }
            final_dispatch = t.elapsed();
        })?;

        let read_start = Instant::now();
        let bytes = self.logits_buf.read_bytes()?;
        let _logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        let readback = read_start.elapsed();

        self.kv_cache.current_seq_len = position + 1;

        let profile = ForwardTokenProfile {
            pre_setup,
            reset: timings.reset,
            begin: timings.begin,
            record: timings.record,
            end: timings.end,
            submit: timings.submit,
            gpu_wait: timings.wait,
            readback,
        };
        Ok((profile, per_layer, final_dispatch))
    }

    /// Like [`forward_token`] but returns a CPU-time breakdown
    /// (host setup / record / submit / GPU-wait / readback). Phase-5A
    /// profiling: feeds the "where do the 3.3 ms go" question with
    /// real numbers so the optimisation target is data-driven.
    pub fn forward_token_profile(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
    ) -> Result<ForwardTokenProfile, Box<dyn std::error::Error>> {
        use std::time::Instant;
        let pre_start = Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(),
                self.config.hidden_dim
            )
            .into());
        }
        self.scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Phase 5A-2 Stage 2D: skip the per-forward pool reset when
        // CB-reuse is on — cached sets stay alive across tokens.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }
        let pre_setup = pre_start.elapsed();

        let timings = cmd_ctx.one_shot_profiled(&dev.device, dev.compute_queue, |cmd| {
            let mut input = self.scratch_a.handle;
            let mut output = self.scratch_b.handle;
            for layer in 0..self.config.n_layers {
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                std::mem::swap(&mut input, &mut output);
            }
            self.dispatch_final(dev, registry, cmd, model, input);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.logits_buf.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[], &[post], &[],
                );
            }
        })?;

        let read_start = Instant::now();
        let bytes = self.logits_buf.read_bytes()?;
        let _logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        let readback = read_start.elapsed();

        self.kv_cache.current_seq_len = position + 1;

        Ok(ForwardTokenProfile {
            pre_setup,
            reset: timings.reset,
            begin: timings.begin,
            record: timings.record,
            end: timings.end,
            submit: timings.submit,
            gpu_wait: timings.wait,
            readback,
        })
    }

    /// One decode step: writes `embedding` (length = hidden_dim) into
    /// the input slot, runs all 36 layers + final norm + LM head at
    /// the given `position`, and reads the logits back. Caller is
    /// responsible for the embedding lookup (CPU dequant of the GGUF
    /// `token_embd.weight` row).
    pub fn forward_token(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embedding: &[f32],
        position: u32,
    ) -> Result<ForwardStats, Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        if embedding.len() != self.config.hidden_dim as usize {
            return Err(format!(
                "embedding length {} != hidden_dim {}",
                embedding.len(),
                self.config.hidden_dim
            )
            .into());
        }
        self.scratch_a.write_bytes(bytemuck::cast_slice(embedding))?;
        // Pre-write the RoPE position buffer.
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;

        // Reset descriptor pool for fresh allocations this forward —
        // skipped when CB-reuse is on; cached sets stay valid.
        if !self.cache_enabled {
            unsafe {
                dev.device.reset_descriptor_pool(
                    self.descriptor_pool, vk::DescriptorPoolResetFlags::empty(),
                )?;
            }
        }

        // Pre-snapshot: we'll record per-layer profile boundaries.
        let mut per_layer_starts: Vec<usize> = Vec::with_capacity(self.config.n_layers as usize);

        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            if let Some(p) = self.profiler.as_mut() {
                p.reset(&dev.device, cmd);
            }

            let mut input = self.scratch_a.handle;
            let mut output = self.scratch_b.handle;

            for layer in 0..self.config.n_layers {
                if let Some(p) = self.profiler.as_ref() {
                    per_layer_starts.push(p.entries_len());
                }
                self.dispatch_layer(dev, registry, cmd, model, layer, position, input, output);
                std::mem::swap(&mut input, &mut output);
            }

            // After last layer, `input` holds the activation we feed
            // into final-norm + LM head.
            self.dispatch_final(dev, registry, cmd, model, input);

            // Make logits visible to the host.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.logits_buf.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[post],
                    &[],
                );
            }
        })?;

        // Logits readback.
        let bytes = self.logits_buf.read_bytes()?;
        let _logits: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();
        // Move to a stable result via clone — bytes is borrowed.
        let logits = bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec();

        // Profiling.
        let samples = if let Some(p) = self.profiler.as_ref() {
            p.collect(&dev.device).unwrap_or_default()
        } else {
            Vec::new()
        };
        let per_shader = ShaderProfiler::aggregate(&samples);
        let mut per_layer: Vec<Duration> = Vec::new();
        if !per_layer_starts.is_empty() && !samples.is_empty() {
            for w in per_layer_starts.windows(2) {
                let (s, e) = (w[0], w[1]);
                if e > s {
                    per_layer.push(samples[s..e].iter().map(|x| x.elapsed).sum());
                }
            }
            // The last layer's slice runs up to (samples.len() - 2 extras).
            let last_start = *per_layer_starts.last().unwrap();
            let last_end = samples.len().saturating_sub(2);
            if last_end > last_start {
                per_layer.push(samples[last_start..last_end].iter().map(|x| x.elapsed).sum());
            }
        }

        // Bump KV state.
        self.kv_cache.current_seq_len = position + 1;

        // Stash logits in a side-channel for the test (avoiding a return-by-value
        // borrowing dance).
        let _ = logits;

        Ok(ForwardStats {
            total: started.elapsed(),
            per_shader,
            per_layer,
            samples,
        })
    }

    /// Debug helper — run ONE layer up to a chosen halt point and
    /// return the relevant intermediate as a `Vec<f32>`. The buffer
    /// returned has *exactly* the size of the relevant intermediate
    /// (e.g. n_heads*head_dim for AfterQkvProj-Q, not hidden_dim).
    pub fn forward_layer_debug_intermediate(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
        target: DebugTarget,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        // Debug path uses a partial layer dispatch with non-stable
        // bindings — invalidate any cached sets from prior decode runs.
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer_partial(
                dev, registry, cmd, model, layer, position,
                self.scratch_a.handle, target,
            );
        })?;

        // Pick the relevant buffer + size for readback.
        let cfg = self.config.clone();
        let (src_buf, count) = match target {
            DebugTarget::AttnNorm => (self.hidden_norm.handle, cfg.hidden_dim as u64),
            DebugTarget::QProj | DebugTarget::QNormRope => (
                self.q_buf.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::KProj | DebugTarget::KNormRope => (
                self.k_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::VProj => (
                self.v_buf.handle,
                (cfg.n_kv_heads * cfg.head_dim) as u64,
            ),
            DebugTarget::AttnOut => (
                self.attn_out.handle,
                (cfg.n_heads * cfg.head_dim) as u64,
            ),
        };
        // Stage into scratch_a (host-visible).
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(src_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(count * 4);
            dev.device.cmd_copy_buffer(cmd, src_buf, self.scratch_a.handle,
                std::slice::from_ref(&copy));
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes = self.scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes[..(count as usize) * 4]).to_vec())
    }

    fn dispatch_layer_partial(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input: vk::Buffer,
        halt: DebugTarget,
    ) {
        let cfg = self.config.clone();

        // attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, self.hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        if halt == DebugTarget::AttnNorm { return; }
        compute_barrier(dev, cmd);

        // Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight");
        let sk = layer_weight_shader(model, layer, "attn_k.weight");
        let sv = layer_weight_shader(model, layer, "attn_v.weight");
        self.run_gemv(dev, registry, cmd, sq,
                      wq, self.hidden_norm.handle, self.q_buf.handle,
                      cfg.hidden_dim, cfg.n_heads * cfg.head_dim, "gemv_q");
        self.run_gemv(dev, registry, cmd, sk,
                      wk, self.hidden_norm.handle, self.k_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_k");
        self.run_gemv(dev, registry, cmd, sv,
                      wv, self.hidden_norm.handle, self.v_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_v");
        if matches!(halt, DebugTarget::QProj | DebugTarget::KProj | DebugTarget::VProj) {
            return;
        }
        compute_barrier(dev, cmd);

        // Q/K norm — Qwen-only (Phase 4D: gated on cfg.has_qk_norm).
        if cfg.has_qk_norm {
            let wqn = layer_weight(model, layer, "attn_q_norm.weight");
            let wkn = layer_weight(model, layer, "attn_k_norm.weight");
            self.run_rms_norm(dev, registry, cmd,
                             self.q_buf.handle, wqn, self.q_buf.handle,
                             cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
            self.run_rms_norm(dev, registry, cmd,
                             self.k_buf.handle, wkn, self.k_buf.handle,
                             cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k");
            compute_barrier(dev, cmd);
        }

        // RoPE
        self.run_rope_neox(dev, registry, cmd, self.q_buf.handle, self.q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.k_buf.handle, self.k_buf.handle,
                           cfg.head_dim, cfg.n_kv_heads, position, "rope_k");
        if matches!(halt, DebugTarget::QNormRope | DebugTarget::KNormRope) {
            return;
        }
        compute_barrier(dev, cmd);

        // KV write — Sprint 9d.3: dispatch the FP32→FP16 conversion
        // compute shader when the cache is FP16-allocated, otherwise
        // keep the cheap vkCmdCopyBuffer transfer.
        let row_bytes = self.kv_cache.row_bytes();
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let k_src = self.k_buf.handle;
        let v_src = self.v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        if self.kv_cache.is_fp16() {
            let kv_elements = cfg.n_kv_heads * cfg.head_dim;
            self.run_kv_copy_fp16(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off,
                "kv_copy_fp16_k_d",
            );
            self.run_kv_copy_fp16(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off,
                "kv_copy_fp16_v_d",
            );
        } else {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(row_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
            }
        }
        // Barrier covers either upstream (transfer or compute write).
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&kv_bar), &[], &[],
            );
        }

        // Attention
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        // Stops here regardless — caller wanted attn_out.
        let _ = halt; // attention is the only further halt point in this partial dispatcher
    }

    /// Debug helper — run ONE layer, returning the post-layer
    /// activations as `Vec<f32>`. Each call submits its own command
    /// buffer; layers must be invoked in order, with `position`
    /// constant (we're tracing a single token through all layers).
    pub fn forward_layer_debug(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input_data: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Phase-2C debug: stage input via scratch_a, single layer
        // dispatch, read scratch_b. We rebuild the descriptor pool
        // (small) from a single big reset.
        self.scratch_a.write_bytes(bytemuck::cast_slice(input_data))?;
        self.rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&(position as u32)))?;
        self.reset_descriptor_pool_and_cache(dev)?;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            self.dispatch_layer(
                dev, registry, cmd, model, layer, position,
                self.scratch_a.handle, self.scratch_b.handle,
            );
            // Readback barrier so the next host map sees scratch_b.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[post],
                    &[],
                );
            }
        })?;
        // scratch_b is GpuOnly — we can't read it. Need to stage.
        // Hack: alloc a host-visible read buf on the fly and copy.
        Ok(self.read_scratch_b(dev, cmd_ctx)?)
    }

    /// Stage scratch_b (GpuOnly) into a host-visible buffer and read
    /// it. Per-call alloc — for debug only.
    fn read_scratch_b(
        &self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Use the host-visible scratch_a as the readback target —
        // it's the same size as scratch_b. We don't need scratch_a's
        // content after this call (caller writes a fresh input next
        // round).
        let bytes = (self.config.hidden_dim as u64) * 4;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            // Pre-barrier: previous SHADER_WRITE → TRANSFER_READ
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_b.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&pre),
                &[],
            );
            let copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes);
            dev.device.cmd_copy_buffer(
                cmd,
                self.scratch_b.handle,
                self.scratch_a.handle,
                std::slice::from_ref(&copy),
            );
            // Post-barrier: TRANSFER_WRITE → HOST_READ
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.scratch_a.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })?;
        let bytes_slice = self.scratch_a.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes_slice[..(self.config.hidden_dim as usize) * 4],
        )
        .to_vec())
    }

    /// Read the most recently written logits — call after
    /// `forward_token` returns successfully.
    pub fn logits(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = self.logits_buf.read_bytes()?;
        Ok(bytemuck::cast_slice::<u8, f32>(
            &bytes[..(self.config.vocab_size as usize) * 4],
        )
        .to_vec())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        self.scratch_a.destroy(device, allocator);
        self.scratch_b.destroy(device, allocator);
        self.hidden_norm.destroy(device, allocator);
        self.q_buf.destroy(device, allocator);
        self.k_buf.destroy(device, allocator);
        self.v_buf.destroy(device, allocator);
        self.attn_out.destroy(device, allocator);
        self.o_buf.destroy(device, allocator);
        self.res1.destroy(device, allocator);
        self.gate_buf.destroy(device, allocator);
        self.up_buf.destroy(device, allocator);
        self.ffn_hidden.destroy(device, allocator);
        self.ffn_out.destroy(device, allocator);
        self.logits_buf.destroy(device, allocator);
        self.fuse0.destroy(device, allocator);
        self.fuse1.destroy(device, allocator);
        self.rope_pos_buf.destroy(device, allocator);
        self.rope_ff_buf.destroy(device, allocator);
        self.rope_idx_buf.destroy(device, allocator);
        self.fa_scratch_out.destroy(device, allocator);
        self.fa_scratch_max.destroy(device, allocator);
        self.fa_scratch_sum.destroy(device, allocator);
        self.batch_input.destroy(device, allocator);
        self.batch_residual.destroy(device, allocator);
        self.batch_norm.destroy(device, allocator);
        self.batch_q8.destroy(device, allocator);
        self.batch_q.destroy(device, allocator);
        self.batch_k.destroy(device, allocator);
        self.batch_v.destroy(device, allocator);
        self.batch_attn_out.destroy(device, allocator);
        self.batch_o.destroy(device, allocator);
        self.batch_gate.destroy(device, allocator);
        self.batch_up.destroy(device, allocator);
        self.batch_ffn_hidden.destroy(device, allocator);
        self.batch_ffn_out.destroy(device, allocator);
        self.kv_cache.destroy(device, allocator);
        if let Some(p) = self.profiler {
            p.destroy(device);
        }
    }

    // -------------------------------------------------------------
    // Per-layer + final + helpers.
    // -------------------------------------------------------------

    fn dispatch_layer(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        position: u32,
        input: vk::Buffer,
        output: vk::Buffer,
    ) {
        let cfg = self.config.clone();

        // (a) attn_norm
        let w = layer_weight(model, layer, "attn_norm.weight");
        self.run_rms_norm(dev, registry, cmd, input, w, self.hidden_norm.handle,
                         cfg.hidden_dim, 1, cfg.rms_norm_eps, "rms_norm_attn");
        compute_barrier(dev, cmd);

        // (b) Q/K/V projections
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        // attn_v.weight is Q6_K in Q4_K_M (mixed-quant) — pick the
        // matching GEMV pipeline per tensor's actual ggml_type.
        let sq = layer_weight_shader(model, layer, "attn_q.weight");
        let sk = layer_weight_shader(model, layer, "attn_k.weight");
        let sv = layer_weight_shader(model, layer, "attn_v.weight");
        self.run_gemv(dev, registry, cmd, sq,
                      wq, self.hidden_norm.handle, self.q_buf.handle,
                      cfg.hidden_dim, cfg.n_heads * cfg.head_dim, "gemv_q");
        self.run_gemv(dev, registry, cmd, sk,
                      wk, self.hidden_norm.handle, self.k_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_k");
        self.run_gemv(dev, registry, cmd, sv,
                      wv, self.hidden_norm.handle, self.v_buf.handle,
                      cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, "gemv_v");
        compute_barrier(dev, cmd);

        // (c) Q/K norm (per head) — only Qwen* sets this (Phase 4D).
        if cfg.has_qk_norm {
            let wqn = layer_weight(model, layer, "attn_q_norm.weight");
            let wkn = layer_weight(model, layer, "attn_k_norm.weight");
            self.run_rms_norm(dev, registry, cmd,
                             self.q_buf.handle, wqn, self.q_buf.handle,
                             cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q");
            self.run_rms_norm(dev, registry, cmd,
                             self.k_buf.handle, wkn, self.k_buf.handle,
                             cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k");
            compute_barrier(dev, cmd);
        }

        // (d) RoPE NeoX on Q and K
        self.run_rope_neox(dev, registry, cmd, self.q_buf.handle, self.q_buf.handle,
                           cfg.head_dim, cfg.n_heads, position, "rope_q");
        self.run_rope_neox(dev, registry, cmd, self.k_buf.handle, self.k_buf.handle,
                           cfg.head_dim, cfg.n_kv_heads, position, "rope_k");
        compute_barrier(dev, cmd);

        // (e) KV-cache write — pos-major.
        // Sprint 9d.3: when the cache is FP16-allocated, dispatch the
        // FP32 → packed-FP16 conversion compute shader. Otherwise the
        // cheap vkCmdCopyBuffer transfer (FP32 → FP32) wins.
        let row_bytes = self.kv_cache.row_bytes();
        let dst_off = self.kv_cache.pos_offset_bytes(layer, position);
        let k_src = self.k_buf.handle;
        let v_src = self.v_buf.handle;
        let k_dst = self.kv_cache.k_buffer.handle;
        let v_dst = self.kv_cache.v_buffer.handle;
        if self.kv_cache.is_fp16() {
            let kv_elements = cfg.n_kv_heads * cfg.head_dim;
            self.run_kv_copy_fp16(
                dev, registry, cmd, k_src, k_dst, kv_elements, dst_off, "kv_copy_fp16_k",
            );
            self.run_kv_copy_fp16(
                dev, registry, cmd, v_src, v_dst, kv_elements, dst_off, "kv_copy_fp16_v",
            );
        } else {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(dst_off).size(row_bytes);
            self.profile("kv_write", dev, cmd, |dev, cmd| unsafe {
                dev.device.cmd_copy_buffer(cmd, k_src, k_dst, std::slice::from_ref(&copy));
                dev.device.cmd_copy_buffer(cmd, v_src, v_dst, std::slice::from_ref(&copy));
            });
        }
        let kv_bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&kv_bar), &[], &[],
            );
        }

        // (f) Attention.
        self.run_scalar_attn(dev, registry, cmd, layer, position);
        compute_barrier(dev, cmd);

        // (g) Output projection.
        let wo = layer_weight(model, layer, "attn_output.weight");
        let so = layer_weight_shader(model, layer, "attn_output.weight");
        self.run_gemv(dev, registry, cmd, so,
                      wo, self.attn_out.handle, self.o_buf.handle,
                      cfg.n_heads * cfg.head_dim, cfg.hidden_dim, "gemv_o");
        compute_barrier(dev, cmd);

        // (h+i) Fused add_res1 + ffn_norm. v0.2 Sprint 9b folds
        //   res1 = input + o_buf
        //   hidden_norm = rms_norm(res1) * ffn_norm.weight
        // into one dispatch.
        let w = layer_weight(model, layer, "ffn_norm.weight");
        self.run_multi_add_rms(
            dev, registry, cmd,
            input, self.o_buf.handle, w,
            /* sum_out  = */ self.res1.handle,
            /* norm_out = */ self.hidden_norm.handle,
            cfg.hidden_dim, 1, cfg.rms_norm_eps, "add_rms_ffn",
        );
        compute_barrier(dev, cmd);

        // (j) gate / up
        let wg = layer_weight(model, layer, "ffn_gate.weight");
        let wu = layer_weight(model, layer, "ffn_up.weight");
        let sg = layer_weight_shader(model, layer, "ffn_gate.weight");
        let su = layer_weight_shader(model, layer, "ffn_up.weight");
        self.run_gemv(dev, registry, cmd, sg,
                      wg, self.hidden_norm.handle, self.gate_buf.handle,
                      cfg.hidden_dim, cfg.ffn_dim, "gemv_gate");
        self.run_gemv(dev, registry, cmd, su,
                      wu, self.hidden_norm.handle, self.up_buf.handle,
                      cfg.hidden_dim, cfg.ffn_dim, "gemv_up");
        compute_barrier(dev, cmd);

        // (k+l) Fused SwiGLU: ffn_hidden = silu(gate) * up. v0.2
        // Sprint 9a folds the previous silu(gate→gate) + barrier +
        // mul(gate, up→ffn_hidden) into one dispatch.
        self.run_swiglu(
            dev, registry, cmd,
            self.gate_buf.handle, self.up_buf.handle, self.ffn_hidden.handle,
            cfg.ffn_dim, "swiglu",
        );
        compute_barrier(dev, cmd);

        // (m) FFN down — Q6_K in Q4_K_M, Q4_K otherwise.
        let wd = layer_weight(model, layer, "ffn_down.weight");
        let sd = layer_weight_shader(model, layer, "ffn_down.weight");
        self.run_gemv(dev, registry, cmd, sd,
                      wd, self.ffn_hidden.handle, self.ffn_out.handle,
                      cfg.ffn_dim, cfg.hidden_dim, "gemv_down");
        compute_barrier(dev, cmd);

        // (n) Residual2 = res1 + ffn_out → output
        self.run_binary(dev, registry, cmd, ShaderId::Add,
                        self.res1.handle, self.ffn_out.handle, output,
                        cfg.hidden_dim, "add_res2");
        compute_barrier(dev, cmd);
    }

    fn dispatch_final(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        input: vk::Buffer,
    ) {
        let w_norm = model
            .tensor("output_norm.weight")
            .expect("output_norm.weight")
            .buffer
            .handle;
        // LM head: prefer dedicated `output.weight`; fall back to tied
        // `token_embd.weight` (Phase 2 doesn't tie weights, but be safe).
        let lm = model
            .tensor("output.weight")
            .or_else(|| model.tensor("token_embd.weight"))
            .expect("LM head present");
        let w_lm = lm.buffer.handle;
        let lm_shader = match lm.ggml_type {
            GgmlType::Q6K => ShaderId::MulMatVecQ6K,
            _ => ShaderId::MulMatVecQ4K,
        };

        self.run_rms_norm(
            dev, registry, cmd,
            input, w_norm, self.hidden_norm.handle,
            self.config.hidden_dim, 1, self.config.rms_norm_eps, "rms_norm_final",
        );
        compute_barrier(dev, cmd);
        self.run_gemv(
            dev, registry, cmd, lm_shader,
            w_lm, self.hidden_norm.handle, self.logits_buf.handle,
            self.config.hidden_dim, self.config.vocab_size, "lm_head",
        );
    }

    // -------------------------------------------------------------
    // Per-shader dispatch methods.
    // -------------------------------------------------------------

    fn alloc_set(&self, dev: &VulkanDevice, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        unsafe { dev.device.allocate_descriptor_sets(&info) }
            .expect("descriptor_set alloc")[0]
    }

    /// Override the cache-enabled flag set by `Forward::new` from
    /// `VULKANFORGE_CB_REUSE`. Used by the parity regression test to
    /// build two Forward instances with explicit cache settings; not
    /// part of the normal lifecycle.
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache_enabled = enabled;
    }

    pub fn cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Phase 5B.2 test escape hatch — toggle the batched-Q prefill
    /// attention path independently of `VULKANFORGE_BATCH_ATTN`. The
    /// `phase5b2_*` parity tests build two Forward instances with
    /// explicit batch-attn settings; not part of the normal lifecycle.
    pub fn set_batch_attn_enabled(&mut self, enabled: bool) {
        self.batch_attn_enabled = enabled;
    }

    pub fn batch_attn_enabled(&self) -> bool {
        self.batch_attn_enabled
    }

    /// Phase 6 v0.1.2 test escape hatch — toggle the mul_mm.comp
    /// path independently of `VULKANFORGE_USE_MUL_MM`.
    pub fn set_mul_mm_enabled(&mut self, enabled: bool) {
        self.mul_mm_enabled = enabled;
    }

    pub fn mul_mm_enabled(&self) -> bool {
        self.mul_mm_enabled
    }

    /// Sprint 3A escape hatch — toggle the Q4_K coopmat fusion path
    /// for `gemm_q` independently of `VULKANFORGE_COOPMAT`. Used by
    /// the parity test in `tests/regression.rs` to construct two
    /// Forward instances in the same process: one with mul_mmq and
    /// one with coopmat, then compare logits.
    pub fn set_coopmat_q4k_enabled(&mut self, enabled: bool) {
        self.coopmat_q4k_enabled = enabled;
    }

    pub fn coopmat_q4k_enabled(&self) -> bool {
        self.coopmat_q4k_enabled
    }

    pub fn set_coopmat_fp8_enabled(&mut self, enabled: bool) {
        self.coopmat_fp8_enabled = enabled;
    }

    /// Phase 5A-2 Stage 2D: cache-aware descriptor-set fetch. When
    /// `cache_enabled` is true and the (layout, bindings) key matches
    /// a previously-built set, the cached handle is returned without
    /// any further Vulkan calls. Otherwise the set is allocated +
    /// written + cached. When `cache_enabled` is false, behaves
    /// exactly like `alloc_set + write_bindings` did before.
    fn alloc_or_get_set(
        &mut self,
        dev: &VulkanDevice,
        layout: vk::DescriptorSetLayout,
        bindings: &[(u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
    ) -> vk::DescriptorSet {
        if !self.cache_enabled {
            let set = self.alloc_set(dev, layout);
            self.write_bindings(dev, set, bindings);
            return set;
        }
        let key = BindingSignature::new(layout, bindings);
        if let Some(&set) = self.set_cache.get(&key) {
            return set;
        }
        let set = self.alloc_set(dev, layout);
        self.write_bindings(dev, set, bindings);
        self.set_cache.insert(key, set);
        set
    }

    /// Reset the descriptor pool *and* clear the cache. Used by
    /// `prefill_batch` and the debug helpers, which need fresh sets
    /// because their bindings vary across calls (per-token offsets,
    /// pos-buf sub-ranges).
    fn reset_descriptor_pool_and_cache(&mut self, dev: &VulkanDevice) -> Result<(), vk::Result> {
        unsafe {
            dev.device.reset_descriptor_pool(
                self.descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        self.set_cache.clear();
        Ok(())
    }

    fn write_bindings(
        &self,
        dev: &VulkanDevice,
        set: vk::DescriptorSet,
        bindings: &[(u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
    ) {
        let infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|&(_, buf, off, range)| vk::DescriptorBufferInfo {
                buffer: buf,
                offset: off,
                range: if range == 0 { vk::WHOLE_SIZE } else { range },
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .enumerate()
            .map(|(i, &(b, _, _, _))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Wraps `f` in optional begin/end timestamp queries.
    fn profile<F>(&mut self, name: &str, dev: &VulkanDevice, cmd: vk::CommandBuffer, f: F)
    where
        F: FnOnce(&VulkanDevice, vk::CommandBuffer),
    {
        let token = self
            .profiler
            .as_mut()
            .map(|p| p.begin(&dev.device, cmd, name.to_string()));
        f(dev, cmd);
        if let (Some(p), Some(t)) = (self.profiler.as_mut(), token) {
            p.end(&dev.device, cmd, t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gemv(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader: ShaderId,
        weights: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        label: &str,
    ) {
        let kernel = registry.get(shader);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, weights, 0, 0),
                (1, input, 0, 0),
                (2, output, 0, 0),
                (3, self.fuse0.handle, 0, 0),
                (4, self.fuse1.handle, 0, 0),
            ],
        );
        let pc = MatVecPushConstants {
            ncols: k, stride_a: k, stride_b: k, stride_d: m,
            batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
            fusion_flags: 0, base_work_group_y: 0,
            ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            // Phase-3C: GEMV pipeline is now built with NUM_ROWS = MMV_NUM_ROWS
            // (= 2). Each workgroup writes NUM_ROWS output rows, so the
            // dispatch count divides — ceiling-div to handle a tail
            // workgroup when m isn't a multiple of NUM_ROWS (the shader
            // bounds-checks via `first_row + NUM_ROWS <= stride_d`).
            let n_rows = super::pipeline_registry::MMV_NUM_ROWS;
            let groups = (m + n_rows - 1) / n_rows;
            dev.device.cmd_dispatch(cmd, groups, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rms_norm(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        weight: vk::Buffer,
        output: vk::Buffer,
        cols: u32,
        rows: u32,
        eps: f32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::RmsNorm);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, input, 0, 0), (1, weight, 0, 0), (2, output, 0, 0)],
        );
        let pc = GenericBinaryPushConstants {
            ne: cols * rows,
            ne00: cols, ne01: rows, ne02: 1, ne03: 1,
            nb00: 1, nb01: cols, nb02: cols * rows, nb03: cols * rows,
            ne10: cols, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: cols, nb12: cols, nb13: cols,
            ne20: cols, ne21: rows, ne22: 1, ne23: 1,
            nb20: 1, nb21: cols, nb22: cols * rows, nb23: cols * rows,
            misalign_offsets: 0,
            param1: eps, param2: 0.0, param3: 0,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, rows, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rope_neox(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        n_rows: u32,
        position: u32,
        label: &str,
    ) {
        // Bind rope_pos_buf at slot 0 (the legacy single-position
        // path used by forward_token). prefill_batch uses
        // run_rope_neox_with_pos_offset to read its own slot.
        self.run_rope_neox_with_pos_offset(
            dev, registry, cmd, input, output, head_dim, n_rows,
            position, /* pos_buf_offset = */ 0, label,
        );
    }

    /// Variant of `run_rope_neox` that binds `rope_pos_buf` starting
    /// at `pos_buf_offset` bytes — required by `prefill_batch` to give
    /// each per-token RoPE dispatch its own pre-staged position slot.
    #[allow(clippy::too_many_arguments)]
    fn run_rope_neox_with_pos_offset(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        n_rows: u32,
        position: u32,
        pos_buf_offset: u64,
        label: &str,
    ) {
        let _ = position; // The pos value is in rope_pos_buf at offset; not in PC.
        // Phase-4D: pick the variant-correct shader. Qwen* uses NeoX
        // (rotates [i, i+n_dims/2] pairs); Llama / Mistral / DeepSeek
        // use the standard adjacent-pair form (rope_norm.comp).
        let (shader_id, rope_mode) = match self.config.rope_variant {
            crate::backend::vulkan::gguf::RopeVariant::Neox => (ShaderId::RopeNeox, 2u32),
            crate::backend::vulkan::gguf::RopeVariant::Norm => (ShaderId::RopeNorm, 0u32),
        };
        let kernel = registry.get(shader_id);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, input, 0, 0),
                // 4-byte slot starting at pos_buf_offset.
                (1, self.rope_pos_buf.handle, pos_buf_offset, 4),
                (2, self.rope_ff_buf.handle, 0, 0),
                (3, output, 0, 0),
                (4, self.rope_idx_buf.handle, 0, 0),
            ],
        );
        let pc = RopePushConstants {
            rope_mode, // 0 = NORM, 2 = NEOX
            nrows: n_rows,
            n_dims: head_dim,
            freq_scale: 1.0,
            freq_base: self.config.rope_freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale: self.rope_theta_scale,
            has_ff: 0,
            sections: [0; 4],
            is_imrope: 0,
            is_back: 0,
            set_rows_stride: 0,
            ne00: head_dim,
            ne01: n_rows,
            ne02: 1,
            nb01: head_dim,
            nb02: head_dim * n_rows,
            nb03: head_dim * n_rows,
            nb11: head_dim,
            nb12: head_dim,
            nb13: head_dim,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, n_rows, 1, 1);
        });
    }

    fn run_scalar_attn(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        position: u32,
    ) {
        let cfg = self.config.clone();
        // Phase 4C: pick the multi-WG split-K path when there are
        // enough tiles to amortise the second dispatch + barrier.
        // Threshold of 2 means seq_len > TILE (= 64) goes through
        // split+reduce; everything shorter takes the Phase-4B
        // single-WG flash_attn path.
        const FA_TILE: u32 = 64;
        const MULTI_WG_MIN_TILES: u32 = 2;
        let seq_len = position + 1;
        let n_tiles = (seq_len + FA_TILE - 1) / FA_TILE;
        if n_tiles >= MULTI_WG_MIN_TILES {
            self.run_flash_attn_split_reduce(dev, registry, cmd, layer, position, n_tiles);
            return;
        }
        // Phase 4B: forward path now dispatches the online-softmax
        // flash_attn shader instead of the Phase-3A tiled scalar_attn.
        // Sprint 9d.3 — FP16 KV-aware variant when the cache is
        // FP16-allocated.
        let kernel = registry.get(if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnFp16Kv
        } else {
            ShaderId::FlashAttn
        });
        let layer_off = self.kv_cache.layer_offset_bytes(layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes();
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, self.q_buf.handle, 0, 0),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, self.attn_out.handle, 0, 0),
            ],
        );
        let pc = ScalarAttnPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            seq_len: position + 1,
            max_seq: self.kv_cache.config.max_seq_len,
            scale: self.attn_scale,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile("scalar_attn", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, cfg.n_heads, 1, 1);
        });
    }

    /// Phase-5B.3 batched RoPE. The per-token prefill loop ran one
    /// `run_rope_neox_with_pos_offset` dispatch per (layer, token,
    /// Q-or-K), each carrying the position via `pos_buf_offset = t*4`.
    /// This helper folds all M tokens into one dispatch by setting
    /// `ne02 = m` (the shader's "samp" axis) and binding the full
    /// `rope_pos_buf[0..m]` so the inner kernel reads
    /// `rope_data_pos[i2]` where `i2` is the token index decoded from
    /// the work-group id.
    ///
    /// The shader itself is unchanged from Phase 4D — `rope_neox.comp`
    /// / `rope_norm.comp` already do the per-row arithmetic via
    /// `i3 / i2 / i1` decomposition; we just feed it the right
    /// strides.
    #[allow(clippy::too_many_arguments)]
    fn run_rope_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        heads_per_token: u32,
        m: u32,
        label: &str,
    ) {
        let (shader_id, rope_mode) = match self.config.rope_variant {
            crate::backend::vulkan::gguf::RopeVariant::Neox => (ShaderId::RopeNeox, 2u32),
            crate::backend::vulkan::gguf::RopeVariant::Norm => (ShaderId::RopeNorm, 0u32),
        };
        let kernel = registry.get(shader_id);
        // Bind the whole `rope_pos_buf[0..m]` once — the shader reads
        // `rope_data_pos[i2]` per row, where i2 is the token index.
        let pos_size = (m as u64) * 4;
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, input, 0, 0),
                (1, self.rope_pos_buf.handle, 0, pos_size),
                (2, self.rope_ff_buf.handle, 0, 0),
                (3, output, 0, 0),
                (4, self.rope_idx_buf.handle, 0, 0),
            ],
        );
        let nrows = heads_per_token * m;
        let pc = RopePushConstants {
            rope_mode,
            nrows,
            n_dims: head_dim,
            freq_scale: 1.0,
            freq_base: self.config.rope_freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale: self.rope_theta_scale,
            has_ff: 0,
            sections: [0; 4],
            is_imrope: 0,
            is_back: 0,
            set_rows_stride: 0,
            ne00: head_dim,
            ne01: heads_per_token,
            ne02: m,
            nb01: head_dim,
            nb02: head_dim * heads_per_token,
            nb03: head_dim * heads_per_token * m,
            nb11: head_dim,
            nb12: head_dim * heads_per_token,
            nb13: head_dim * heads_per_token * m,
        };
        // The shader recovers row from `gl_GlobalInvocationID.x +
        // 32768 * gl_GlobalInvocationID.z`, so dispatch with z
        // multiplexing once nrows clears 32 768. For Qwen3 / Llama
        // worst case this is m=2048 × n_heads=32 = 65 536, fits with
        // dispatch_z = 2.
        let dispatch_x = if nrows > 32768 { 32768 } else { nrows };
        let dispatch_z = (nrows + 32767) / 32768;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, dispatch_x, 1, dispatch_z);
        });
    }

    /// Phase-5B.2 batched-Q flash attention. One dispatch over
    /// `(n_heads, m, 1)` covers all M queries against the current
    /// layer's KV cache with a per-query causal mask
    /// `causal_len = q_start + q_idx + 1`. Replaces the M-fold
    /// per-token attention loop in `dispatch_layer_batch` when
    /// `batch_attn_enabled` is set.
    ///
    /// `q_buf`: storage buffer holding `[m, n_heads, head_dim]` post-
    /// RoPE Q values (the layer-batch path stages those into
    /// `batch_q` after the per-token RoPE pass).
    ///
    /// `o_buf`: storage buffer that receives `[m, n_heads, head_dim]`
    /// attention output. The layer-batch path passes `batch_attn_out`.
    #[allow(clippy::too_many_arguments)]
    fn run_flash_attn_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        q_buf: vk::Buffer,
        o_buf: vk::Buffer,
        m: u32,
        q_start: u32,
        n_kv: u32,
    ) {
        let cfg = self.config.clone();
        // v0.2 Sprint 9d.2 — pick the FP16-KV-aware variant when the
        // cache is allocated as FP16; otherwise the original FP32 SPV.
        let kernel = registry.get(if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnBatchFp16Kv
        } else {
            ShaderId::FlashAttnBatch
        });
        let layer_off = self.kv_cache.layer_offset_bytes(layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes();
        let q_bytes_total = (m as u64) * (cfg.n_heads as u64) * (cfg.head_dim as u64) * 4;
        let set = self.alloc_or_get_set(
            dev,
            kernel.descriptor_set_layout,
            &[
                (0, q_buf, 0, q_bytes_total),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, o_buf, 0, q_bytes_total),
            ],
        );
        let pc = FlashAttnBatchPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            m,
            n_kv,
            q_start,
            scale: self.attn_scale,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        let n_heads = cfg.n_heads;
        self.profile("fa_batch", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, n_heads, m, 1);
        });
    }

    /// Sprint 7 / 7.5 — Br>1 tiled-Q flash attention dispatch.
    /// Identical bind / push layout to `run_flash_attn_batch`; the
    /// shader ID is selected per `self.fa_tiled_br` and the dispatch
    /// shape is `(n_heads, ceil(m/BR), 1)` where BR is baked into
    /// each SPV via `-DBR=4|8|16`.
    #[allow(clippy::too_many_arguments)]
    fn run_flash_attn_tiled(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        q_buf: vk::Buffer,
        o_buf: vk::Buffer,
        m: u32,
        q_start: u32,
        n_kv: u32,
    ) {
        // v0.2 Sprint 10C — coopmat shader takes priority when
        // VULKANFORGE_COOPMAT_ATTN=1 is set. Forces Br=16 (the only
        // shape the coopmat SPV ships); Bc=16 is implicit in the
        // coopmat SPV's own #defines.
        let (shader_id, br) = if self.coopmat_attn_enabled {
            if self.kv_cache.is_fp16() {
                (ShaderId::FlashAttnCoopmatFp16Kv, 16u32)
            } else {
                (ShaderId::FlashAttnCoopmat, 16u32)
            }
        } else if self.kv_cache.is_fp16() {
            // v0.2 Sprint 9d.2 — FP16 KV variant lives only for the
            // (16, 32) shape today. The other Br/Bc combos always read
            // FP32 KV; if the cache is FP16-allocated and the user
            // selected a non-default Br/Bc, panic loudly — the SPV would
            // misinterpret packed-FP16 data as FP32.
            match (self.fa_tiled_br, self.fa_tiled_bc) {
                (16, 32) => (ShaderId::FlashAttnTiledBr16Bc32Fp16Kv, 16u32),
                _ => panic!(
                    "VULKANFORGE_FP16_KV=1 requires the default Br=16/Bc=32 \
                     tiled flash-attn variant; got Br={}/Bc={}. \
                     Sprint 9d.2 only ships FP16 SPVs for the default shape.",
                    self.fa_tiled_br, self.fa_tiled_bc,
                ),
            }
        } else {
            match (self.fa_tiled_br, self.fa_tiled_bc) {
                (16, 32) => (ShaderId::FlashAttnTiledBr16Bc32, 16u32),
                (16, _)  => (ShaderId::FlashAttnTiledBr16,     16u32),
                (8,  _)  => (ShaderId::FlashAttnTiledBr8,       8u32),
                _        => (ShaderId::FlashAttnTiledBr4,       4u32),
            }
        };
        let cfg = self.config.clone();
        let kernel = registry.get(shader_id);
        let layer_off = self.kv_cache.layer_offset_bytes(layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes();
        let q_bytes_total = (m as u64) * (cfg.n_heads as u64) * (cfg.head_dim as u64) * 4;
        let set = self.alloc_or_get_set(
            dev,
            kernel.descriptor_set_layout,
            &[
                (0, q_buf, 0, q_bytes_total),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, o_buf, 0, q_bytes_total),
            ],
        );
        let pc = FlashAttnBatchPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            m,
            n_kv,
            q_start,
            scale: self.attn_scale,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        let n_heads = cfg.n_heads;
        let q_tiles = m.div_ceil(br);
        self.profile("fa_tiled", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, n_heads, q_tiles, 1);
        });
    }

    /// Phase-4C split-K attention: dispatches the per-tile worker
    /// across `(n_heads, n_tiles, 1)` workgroups, then a reducer over
    /// `(n_heads, 1, 1)` that combines the partials with online-softmax
    /// correction.  Inserts the required compute→compute barrier
    /// between the two passes (the reducer reads the worker's writes
    /// out of `fa_scratch_*`).
    fn run_flash_attn_split_reduce(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        position: u32,
        n_tiles: u32,
    ) {
        let cfg = self.config.clone();
        let layer_off = self.kv_cache.layer_offset_bytes(layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes();

        // ---- Split-K worker ----
        // Sprint 9d.3 — FP16 KV-aware variant of the split-K worker.
        // The reducer (FlashAttnReduce) doesn't read KV (only partials),
        // so it stays on the FP32 SPV.
        let split_kernel = registry.get(if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnSplitFp16Kv
        } else {
            ShaderId::FlashAttnSplit
        });
        let split_set = self.alloc_or_get_set(
            dev, split_kernel.descriptor_set_layout,
            &[
                (0, self.q_buf.handle, 0, 0),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, self.fa_scratch_out.handle, 0, 0),
                (4, self.fa_scratch_max.handle, 0, 0),
                (5, self.fa_scratch_sum.handle, 0, 0),
            ],
        );
        let split_pc = FlashAttnSplitPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            seq_len: position + 1,
            max_seq: self.kv_cache.config.max_seq_len,
            scale: self.attn_scale,
            n_tiles,
        };
        let split_layout = split_kernel.pipeline_layout;
        let split_pipeline = split_kernel.pipeline;
        let n_heads = cfg.n_heads;
        self.profile("fa_split", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, split_pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, split_layout, 0, &[split_set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, split_layout, vk::ShaderStageFlags::COMPUTE, 0,
                bytemuck::bytes_of(&split_pc),
            );
            dev.device.cmd_dispatch(cmd, n_heads, n_tiles, 1);
        });

        // Compute → compute barrier: the reducer reads what the
        // worker just wrote into the three fa_scratch_* buffers.
        compute_barrier(dev, cmd);

        // ---- Reduce ----
        let red_kernel = registry.get(ShaderId::FlashAttnReduce);
        let red_set = self.alloc_or_get_set(
            dev, red_kernel.descriptor_set_layout,
            &[
                (0, self.fa_scratch_out.handle, 0, 0),
                (1, self.fa_scratch_max.handle, 0, 0),
                (2, self.fa_scratch_sum.handle, 0, 0),
                (3, self.attn_out.handle, 0, 0),
            ],
        );
        let red_pc = FlashAttnReducePushConstants {
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            n_tiles,
        };
        let red_layout = red_kernel.pipeline_layout;
        let red_pipeline = red_kernel.pipeline;
        self.profile("fa_reduce", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, red_pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, red_layout, 0, &[red_set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, red_layout, vk::ShaderStageFlags::COMPUTE, 0,
                bytemuck::bytes_of(&red_pc),
            );
            dev.device.cmd_dispatch(cmd, n_heads, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run_binary(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader: ShaderId,
        a: vk::Buffer,
        b: vk::Buffer,
        d: vk::Buffer,
        n: u32,
        label: &str,
    ) {
        let kernel = registry.get(shader);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, a, 0, 0), (1, b, 0, 0), (2, d, 0, 0)],
        );
        let pc = GenericBinaryPushConstants {
            ne: n,
            ne00: n, ne01: 1, ne02: 1, ne03: 1,
            nb00: 1, nb01: n, nb02: n, nb03: n,
            ne10: n, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: n, nb12: n, nb13: n,
            ne20: n, ne21: 1, ne22: 1, ne23: 1,
            nb20: 1, nb21: n, nb22: n, nb23: n,
            misalign_offsets: 0,
            param1: 0.0, param2: 0.0, param3: 0,
        };
        let dispatch_y = (n + 511) / 512;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, 1, dispatch_y, 1);
        });
    }

    /// v0.2 Sprint 9b — fused residual-add + RMSNorm-mul dispatch.
    /// Computes `sum = a + b`, `norm_out = rms_norm(sum) * weight`
    /// in one pass. `sum` may alias `a` for in-place residual updates
    /// (the batched dispatch passes `batch_residual` for both).
    /// Replaces a separate `add` + barrier + `rms_norm` pair, saving
    /// one dispatch and one compute barrier per layer.
    #[allow(clippy::too_many_arguments)]
    fn run_multi_add_rms(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        a: vk::Buffer,
        b: vk::Buffer,
        weight: vk::Buffer,
        sum_out: vk::Buffer,
        norm_out: vk::Buffer,
        cols: u32,
        rows: u32,
        eps: f32,
        label: &'static str,
    ) {
        let kernel = registry.get(ShaderId::MultiAddRms);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, a, 0, 0),
                (1, b, 0, 0),
                (2, weight, 0, 0),
                (3, sum_out, 0, 0),
                (4, norm_out, 0, 0),
            ],
        );
        let pc = MultiAddRmsPushConstants { ne00: cols, n_rows: rows, eps };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, rows, 1, 1);
        });
    }

    /// v0.2 Sprint 9d.2 — FP32 → packed-FP16 KV-cache write.
    /// Replaces `vkCmdCopyBuffer` for prefill K/V uploads when
    /// `KvCache::is_fp16()`. Each thread converts one (a, b) pair to
    /// one packed `uint`, so dispatch_x = ceil(n_elements / 2 / 256).
    /// `dst_byte_offset` is the destination offset in **bytes** (as
    /// returned by `KvCache::pos_offset_bytes`); the helper converts
    /// to uint units (= bytes / 4) for the shader.
    #[allow(clippy::too_many_arguments)]
    fn run_kv_copy_fp16(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Buffer,
        n_elements: u32,
        dst_byte_offset: u64,
        label: &'static str,
    ) {
        debug_assert_eq!(
            dst_byte_offset % 4,
            0,
            "kv_copy_fp16: dst_byte_offset must be uint-aligned (got {dst_byte_offset})"
        );
        let dst_uint_offset = (dst_byte_offset / 4) as u32;
        let kernel = registry.get(ShaderId::KvCopyFp16);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, src, 0, 0), (1, dst, 0, 0)],
        );
        let pc = KvCopyFp16PushConstants {
            n_elements,
            dst_uint_offset,
            src_float_offset: 0,
        };
        // 256 threads/WG, 2 elements/thread → 512 elements/WG.
        let dispatch_x = (n_elements + 511) / 512;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
        });
    }

    /// v0.2 Sprint 9c.5 — fused rms_norm+mul+RoPE-NeoX dispatch for
    /// Q/K-norm. One dispatch covers what previously took two
    /// (`run_rms_norm` + `run_rope_batch`) per Q or K projection.
    ///
    /// Buffer layout in `qk` is `[m, heads_per_token, head_dim]`
    /// (token-major, head_dim contiguous). Each WG normalizes one row
    /// of `head_dim` elements (one (token, head) pair), multiplies by
    /// the per-dim `weight[head_dim]`, then applies RoPE-NeoX in-place
    /// using the position from `rope_pos_buf[i2]` (i2 = token index).
    /// Output `qk_out` may alias `qk` for in-place rotation (the
    /// production callers do this).
    #[allow(clippy::too_many_arguments)]
    fn run_rms_norm_mul_rope(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        qk: vk::Buffer,
        weight: vk::Buffer,
        qk_out: vk::Buffer,
        head_dim: u32,
        heads_per_token: u32,
        m: u32,
        eps: f32,
        label: &'static str,
    ) {
        let kernel = registry.get(ShaderId::RmsNormMulRope);
        // Binding map matches rms_norm.comp's RMS_NORM_ROPE_FUSION
        // path: 0=A, 1=B(weight), 3=pos, 4=ff, 5=output, 6=set_rows_idx.
        // Binding 2 is intentionally unused by the shader; we omit it.
        let pos_size = (m as u64) * 4;
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, qk, 0, 0),
                (1, weight, 0, 0),
                (3, self.rope_pos_buf.handle, 0, pos_size),
                (4, self.rope_ff_buf.handle, 0, 0),
                (5, qk_out, 0, 0),
                (6, self.rope_idx_buf.handle, 0, 0),
            ],
        );

        let rope_mode: u32 = match self.config.rope_variant {
            crate::backend::vulkan::gguf::RopeVariant::Neox => 2,
            crate::backend::vulkan::gguf::RopeVariant::Norm => 0,
        };

        // CRITICAL dispatch geometry: the fused shader maps
        // gl_WorkGroupID.x → row (head_idx), gl_WorkGroupID.y → channel
        // (token_idx). The rope step then reads `rope_data_pos[channel]`,
        // so the y-dim *must* be the token dimension; otherwise every
        // token would rotate using pos=0. Use (heads_per_token, m, 1).
        let pc = RmsNormMulRopePushConstants {
            // GenericBinary header — describes the rms_norm input/output.
            // ne00 = ncols (head_dim), ne01 = head_count along X workgroups,
            // ne02 = token_count along Y workgroups. nb01/nb02 give the
            // strides in elements that the shader applies to row/channel.
            ne: head_dim * heads_per_token * m,
            ne00: head_dim, ne01: heads_per_token, ne02: m, ne03: 1,
            nb00: 1, nb01: head_dim, nb02: head_dim * heads_per_token,
            nb03: head_dim * heads_per_token * m,
            // Weight (data_b) is the single per-dim gamma vector; broadcast
            // identical across all rows/channels.
            ne10: head_dim, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: head_dim, nb12: head_dim, nb13: head_dim,
            // Output stride matches input (in-place rotation).
            ne20: head_dim, ne21: heads_per_token, ne22: m, ne23: 1,
            nb20: 1, nb21: head_dim, nb22: head_dim * heads_per_token,
            nb23: head_dim * heads_per_token * m,
            misalign_offsets: 0,
            param1: eps,
            param2: 0.0,
            param3: 0,
            // rope_params — mirror what `run_rope_batch` writes for the
            // stand-alone RoPE pass so the rotation is bit-equivalent.
            // ne01/ne02 here are *element-shape* (heads × tokens), not
            // workgroup counts (which the shader derives from
            // gl_NumWorkGroups). Strides are in elements.
            rope: RopePushConstants {
                rope_mode,
                nrows: heads_per_token * m,
                n_dims: head_dim,
                freq_scale: 1.0,
                freq_base: self.config.rope_freq_base,
                ext_factor: 0.0,
                attn_factor: 1.0,
                corr_dims: [0.0, 0.0],
                theta_scale: self.rope_theta_scale,
                has_ff: 0,
                sections: [0; 4],
                is_imrope: 0,
                is_back: 0,
                set_rows_stride: 0,
                ne00: head_dim,
                ne01: heads_per_token,
                ne02: m,
                nb01: head_dim,
                nb02: head_dim * heads_per_token,
                nb03: head_dim * heads_per_token * m,
                nb11: head_dim,
                nb12: head_dim * heads_per_token,
                nb13: head_dim * heads_per_token * m,
            },
        };

        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, heads_per_token, m, 1);
        });
    }

    /// v0.2 Sprint 9a — fused SwiGLU dispatch.
    /// `out[i] = silu(gate[i]) * up[i]` over `n` FP32 elements.
    /// Replaces `run_silu(g→g) + barrier + run_binary(Mul, g, u, o)`
    /// with a single dispatch that keeps the SiLU intermediate in
    /// registers (no global-memory round-trip).
    #[allow(clippy::too_many_arguments)]
    fn run_swiglu(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        gate: vk::Buffer,
        up: vk::Buffer,
        out: vk::Buffer,
        n: u32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::SwiGLU);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, gate, 0, 0), (1, up, 0, 0), (2, out, 0, 0)],
        );
        let pc = SwigluPushConstants { n };
        // local_size_x = 256 in swiglu.comp, 1 element per thread.
        let dispatch_x = (n + 255) / 256;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, dispatch_x, 1, 1);
        });
    }

    // -----------------------------------------------------------------
    // Phase-3E batch GEMM dispatch helpers + prefill_batch orchestration.
    // -----------------------------------------------------------------

    /// Dispatch `quantize_q8_1` over `n_elements` floats, packing them
    /// into `block_q8_1_x4` blocks (128 elements / 144 bytes each).
    fn run_quantize_q8_1(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        n_elements: u32,
        label: &'static str,
    ) {
        let kernel = registry.get(ShaderId::QuantizeQ8_1);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, input, 0, 0), (1, output, 0, 0)],
        );
        let num_blocks = (n_elements + 127) / 128;
        let pc = Q8_1QuantizePushConstants {
            ne: n_elements,
            num_blocks,
        };
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, num_blocks, 1, 1);
        });
    }

    /// Dispatch `mul_mmq` GEMM. Layout per Phase 3D §4.1:
    ///   A = weights, M×K row-major, `stride_a = K`
    ///   B = activations, Q8_1 packed, virtual stride_b = K (in elements)
    ///   D = output, N×M row-major, `stride_d = M`
    /// Each output row holds one token's `M`-dim projection.
    #[allow(clippy::too_many_arguments)]
    fn run_gemm(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader_id: ShaderId,
        weights: vk::Buffer,
        activations_q8: vk::Buffer,
        output: vk::Buffer,
        m: u32,
        n: u32,
        k: u32,
        label: &'static str,
    ) {
        let kernel = registry.get(shader_id);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, weights, 0, 0),
                (1, activations_q8, 0, 0),
                (2, output, 0, 0),
            ],
        );
        let pc = MmqPushConstants {
            m, n, k,
            stride_a: k,
            stride_b: k,
            stride_d: m,
            batch_stride_a: m * k,
            batch_stride_b: n * k,
            batch_stride_d: m * n,
            base_work_group_z: 0,
            num_batches: 1,
            k_split: k,
            ne02: 1,
            ne12: 1,
            broadcast2: 1,
            broadcast3: 1,
        };
        // BM/BN come from the pipeline's spec-constants. S-tile
        // (MulMm*Q{4,6}K, MulMmqQ{4,6}K) → 64×64. L-tile
        // (MulMmqQ{4,6}KL, MulMmQ4KCoopmat) → 128×128. Matches the
        // pipeline_registry.rs spec-constant block.
        let (bm, bn): (u32, u32) = match shader_id {
            ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL | ShaderId::MulMmQ4KCoopmat => (128, 128),
            _ => (64, 64),
        };
        let groups_x = (m + bm - 1) / bm;
        let groups_y = (n + bn - 1) / bn;
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        });
    }

    /// Sprint 3C — round `n` up to the next multiple of `tile`. Lets
    /// the coopmat dispatch sees a full-tile N regardless of the
    /// real seq_len — paired with `zero_activation_tail` below this
    /// eliminates the partial-tile-store bug that ate Sprint 3A.
    fn pad_to_tile(n: u32, tile: u32) -> u32 {
        (n + tile - 1) / tile * tile
    }

    /// Sprint 3C — fill the `[n, n_padded)` rows of an [N, K]
    /// row-major activation buffer with zeros. The coopmat dispatch
    /// sees N = n_padded; the extra rows multiply with weights to
    /// produce zeros in the padded output rows, which downstream code
    /// (RoPE / attention / sampler) ignores because it walks only
    /// `seq_len = n` rows.
    fn zero_activation_tail(
        &self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        buf: vk::Buffer,
        n: u32,
        n_padded: u32,
        k: u32,
    ) {
        if n_padded <= n {
            return;
        }
        // Diagnostic: VULKANFORGE_COOPMAT_NO_FILL skips the fill so we
        // can isolate whether the fill or the n_padded dispatch is the
        // cause of any logits drift.
        if std::env::var("VULKANFORGE_COOPMAT_NO_FILL").is_ok() {
            return;
        }
        let offset_bytes = (n as u64) * (k as u64) * 4;
        let size_bytes = ((n_padded - n) as u64) * (k as u64) * 4;
        unsafe {
            dev.device
                .cmd_fill_buffer(cmd, buf, offset_bytes, size_bytes, 0);
        }
    }

    /// Sprint 3C — pick the naive padded shader (BF16 vs FP8) per
    /// `coopmat_fp8_enabled`.
    fn coopmat_naive_padded_shader(&self) -> ShaderId {
        if self.coopmat_fp8_enabled {
            ShaderId::MulCoopmatQ4KNaivePaddedFp8
        } else if std::env::var("VULKANFORGE_COOPMAT_LEGACY_STORE").is_ok() {
            // Diagnostic — fall back to Sprint 3B's LDS-staged store.
            // Used by the test that bisects the top-5 drop.
            ShaderId::MulCoopmatQ4KNaiveBf16
        } else {
            ShaderId::MulCoopmatQ4KNaivePaddedBf16
        }
    }

    /// Sprint 3A — dispatch the Q4_K dequant-fusion coopmat GEMM with
    /// the forward-pass-compatible memory layout. Inputs:
    ///
    /// * `weights` : Q4_K block buffer, M × K rows × cols of weights
    ///               packed as 144 B / 256-weight blocks.
    /// * `acts_f32`: FP32 activations, [N, K] = [seq_len, hidden]
    ///               row-major (i.e. the runtime `batch_norm` buffer
    ///               after RMSNorm — *not* the Q8_1 quantised one).
    /// * `output`  : FP32 output, [N, M] = [seq_len, output_dim]
    ///               row-major (matches the mul_mmq output convention
    ///               so downstream RoPE / attention paths stay
    ///               unchanged).
    ///
    /// `m`, `n`, `k` follow the same convention `run_gemm` uses:
    /// `m` = output_dim, `n` = seq_len, `k` = hidden. The shader's
    /// `stride_b = K` and `stride_c = M` reflect the [N, K] / [N, M]
    /// row-major layout under `-DFORWARD_LAYOUT`.
    #[allow(clippy::too_many_arguments)]
    fn run_gemm_coopmat_q4k(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        shader_id: ShaderId,
        weights: vk::Buffer,
        acts_f32: vk::Buffer,
        output: vk::Buffer,
        m: u32,
        n: u32,
        k: u32,
        bm_tile: u32,
        bn_tile: u32,
        label: &'static str,
    ) {
        let kernel = registry.get(shader_id);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, weights, 0, 0),
                (1, acts_f32, 0, 0),
                (2, output, 0, 0),
            ],
        );
        let pc = CoopmatPushConstants {
            m, n, k,
            stride_a: k,   // weights stride in elements
            stride_b: k,   // FORWARD_LAYOUT: B is [N, K], stride = K
            stride_c: m,   // FORWARD_LAYOUT: C is [N, M], stride = M
        };
        let groups_x = m.div_ceil(bm_tile);
        let groups_y = n.div_ceil(bn_tile);
        let layout = kernel.pipeline_layout;
        let pipeline = kernel.pipeline;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        });
    }

    /// Copy one row out of an `[seq_len, dim]` batched GEMM output
    /// into a single-token buffer so the existing per-token RMSNorm
    /// / RoPE / attention helpers can run unchanged.
    fn copy_batch_row(
        &self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        src_offset: u64,
        dst: vk::Buffer,
        bytes: u64,
    ) {
        let copy = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(0)
            .size(bytes);
        unsafe {
            dev.device.cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&copy));
        }
    }

    /// Phase-3E prefill_batch — runs `token_ids` through all 36 layers
    /// in **one** command buffer using batched GEMMs for the 7 weight
    /// projections per layer. Per-token loops handle elementwise
    /// (RMSNorm / RoPE / SiLU / Add / Mul) and the causal attention.
    ///
    /// On exit:
    /// * `kv_cache.current_seq_len` advances by `token_ids.len()`.
    /// * The last token's logits are in `self.logits_buf`, ready for
    ///   the decode loop's argmax.
    ///
    /// Caller must supply pre-computed FP32 embeddings for every token
    /// (one row of `token_embd.weight` Q4_K-dequantised, see
    /// `decode::embedding_row`). The flattened `[seq_len × hidden_dim]`
    /// goes into `batch_input`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        embeddings: &[f32],
        seq_len: u32,
        base_pos: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if seq_len == 0 {
            return Ok(());
        }
        if seq_len > self.max_prefill_tokens {
            return Err(format!(
                "prefill_batch: seq_len {seq_len} > max_prefill_tokens {}",
                self.max_prefill_tokens
            ).into());
        }
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let ffn = cfg.ffn_dim;
        let hidden_bytes = (hidden as u64) * 4;
        let kv_bytes = (kv_dim as u64) * 4;
        let q_bytes = (q_dim as u64) * 4;
        if (embeddings.len() as u32) != seq_len * hidden {
            return Err("prefill_batch: embeddings length mismatch".into());
        }

        // CPU → batch_input (host-visible).
        self.batch_input.write_bytes(bytemuck::cast_slice(embeddings))?;

        // Pre-stage RoPE positions for every token in the batch.
        // CRITICAL: all GPU dispatches in this submit run AFTER all
        // host writes complete, so we must write every per-token
        // position into a separate slot of rope_pos_buf BEFORE we
        // start recording — otherwise the per-token RoPE dispatches
        // would all read the last-written value (Phase 3E drift bug).
        let positions: Vec<u32> = (0..seq_len).map(|t| base_pos + t).collect();
        self.rope_pos_buf
            .write_bytes(bytemuck::cast_slice(&positions))?;

        // prefill_batch's per-token attention loop binds varying
        // pos_buf sub-ranges, so cached sets from a prior decode
        // can't be reused. Drop them up-front.
        self.reset_descriptor_pool_and_cache(dev)?;

        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            // Seed residual chain from the embedded inputs. A single
            // device-side copy is cheaper than a host re-write.
            let total_bytes = (seq_len as u64) * hidden_bytes;
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(total_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, self.batch_input.handle, self.batch_residual.handle,
                    std::slice::from_ref(&copy),
                );
            }
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
            }

            // Sprint 9b.2 — seed `batch_norm` for layer 0. Subsequent
            // layers inherit `batch_norm` from the previous layer's
            // end-of-layer cross-layer fusion (multi_add_rms with
            // `next_attn_norm_weight = Some(...)`).
            let w_attn_norm_0 = layer_weight(model, 0, "attn_norm.weight");
            self.run_rms_norm(
                dev, registry, cmd,
                self.batch_residual.handle, w_attn_norm_0, self.batch_norm.handle,
                hidden, seq_len, cfg.rms_norm_eps, "rms_norm_attn_b_seed",
            );
            compute_barrier(dev, cmd);

            for layer in 0..cfg.n_layers {
                // Pass the *next* layer's attn_norm weight so the
                // end-of-layer fusion can pre-populate batch_norm for
                // the next iteration. None on the final layer falls
                // back to a plain add_res2.
                let next_w = if layer + 1 < cfg.n_layers {
                    Some(layer_weight(model, layer + 1, "attn_norm.weight"))
                } else {
                    None
                };
                self.dispatch_layer_batch(
                    dev, registry, cmd, model, layer, seq_len, base_pos,
                    next_w,
                );
            }

            // Final norm + LM head — only the LAST token needs logits.
            // Copy the last row of batch_residual into scratch_a and
            // run the existing per-token final path.
            let last_off = ((seq_len - 1) as u64) * hidden_bytes;
            self.copy_batch_row(
                dev, cmd, self.batch_residual.handle, last_off,
                self.scratch_a.handle, hidden_bytes,
            );
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
            }
            self.dispatch_final(dev, registry, cmd, model, self.scratch_a.handle);

            // Logits → host.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.logits_buf.handle)
                .offset(0).size(vk::WHOLE_SIZE);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[], &[post], &[],
                );
            }
            // Silence unused locals.
            let _ = (kv_bytes, q_bytes, ffn);
        })?;

        self.kv_cache.current_seq_len = base_pos + seq_len;
        Ok(())
    }

    /// One layer's worth of batched dispatches, recorded into `cmd`.
    /// Reads from `batch_residual`, writes back to `batch_residual`.
    ///
    /// Sprint 9b.2 — cross-layer fusion contract:
    /// * `batch_norm` MUST already contain `rms_norm(batch_residual) *
    ///   layer N's attn_norm.weight` on entry. The caller is responsible
    ///   for seeding it (separate `run_rms_norm` before the layer-0 call;
    ///   subsequent layers inherit it from the previous layer's
    ///   end-of-layer fusion).
    /// * `next_attn_norm_weight = Some(w)` activates the end-of-layer
    ///   `multi_add_rms(batch_residual, batch_ffn_out, w)` fusion that
    ///   simultaneously updates `batch_residual` AND populates
    ///   `batch_norm` with `rms_norm(...) * w` for the *next* layer.
    /// * `next_attn_norm_weight = None` (last layer) emits a plain
    ///   `add_res2_b` and leaves `batch_norm` untouched.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_layer_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        model: &LoadedModel,
        layer: u32,
        seq_len: u32,
        base_pos: u32,
        next_attn_norm_weight: Option<vk::Buffer>,
    ) {
        let cfg = self.config.clone();
        let hidden = cfg.hidden_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let ffn = cfg.ffn_dim;
        let kv_bytes = (kv_dim as u64) * 4;
        let q_bytes = (q_dim as u64) * 4;
        let ffn_bytes = (ffn as u64) * 4;

        // ---- (a) attn_norm: per-token RMSNorm into batch_norm. ----
        // Sprint 9b.2 — this used to dispatch run_rms_norm(batch_residual,
        // attn_norm.weight → batch_norm). Now `batch_norm` is pre-seeded
        // by either prefill_batch (for layer 0) or by the previous
        // layer's cross-layer fusion (Sprint 9b.2). Nothing to do here.

        // Phase 6/7 — mul_mm path takes FP32 activations directly, so
        // skip the Q8_1 quantize step. mul_mmq still needs it. The
        // aligned variant (vec4 B-loads) requires seq_len % 4 == 0;
        // if it isn't we fall back to mul_mmq because mul_mm with
        // scalar B-loads is ~45 % slower than mul_mmq at prefill.
        // Sprint 11E — when COOPMAT mul_mm is on, force the unaligned MulMm
        // path (we don't ship a COOPMAT-aligned SPV). Otherwise the existing
        // MulMmAligned fallback path stays.
        let gemm_kind = if self.mul_mm_coopmat_enabled {
            GemmKind::MulMm
        } else if self.mul_mm_enabled {
            if seq_len % 4 == 0 { GemmKind::MulMmAligned } else { GemmKind::Mmq }
        } else {
            GemmKind::Mmq
        };
        let use_mul_mm = matches!(gemm_kind, GemmKind::MulMm | GemmKind::MulMmAligned);
        let gemm_input_attn = if use_mul_mm {
            self.batch_norm.handle
        } else {
            // ---- (b) Quantize attn_norm output → Q8_1 (mul_mmq path) ----
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_norm.handle, self.batch_q8.handle,
                seq_len * hidden, "quantize_attn",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };

        // ---- (c) Q/K/V GEMMs. Mixed-quant: V uses Q6_K. ----
        let wq = layer_weight(model, layer, "attn_q.weight");
        let wk = layer_weight(model, layer, "attn_k.weight");
        let wv = layer_weight(model, layer, "attn_v.weight");
        let cm_mm = self.mul_mm_coopmat_enabled;
        let sq = layer_weight_shader_gemm(model, layer, "attn_q.weight", gemm_kind, q_dim, seq_len, cm_mm);
        let sk = layer_weight_shader_gemm(model, layer, "attn_k.weight", gemm_kind, kv_dim, seq_len, cm_mm);
        let sv = layer_weight_shader_gemm(model, layer, "attn_v.weight", gemm_kind, kv_dim, seq_len, cm_mm);
        // Sprint 3A: gemm_q can opt into the Q4_K coopmat fusion.
        // Coopmat reads activations from `batch_norm` (FP32) regardless
        // of the mul_mm/mul_mmq route the rest of the layer takes, so
        // the coopmat dispatch passes `self.batch_norm.handle` directly
        // — independent of `gemm_input_attn` (which is either FP32 or
        // Q8_1 depending on mul_mm_enabled). The other six GEMMs keep
        // the existing routing.
        if self.coopmat_q4k_enabled {
            // Sprint 3C — naive padded for skinny-N (covers all
            // typical Qwen3 prefill shapes). Pad seq_len up to a
            // multiple of 16 and zero the activation tail so every
            // output tile is full and the kernel can use a direct
            // ColumnMajor coopMatStore. The fused FP8/BF16 mode
            // toggles via `coopmat_fp8_enabled`.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_norm.handle,
                seq_len, n_padded, hidden,
            );
            // The fill-buffer is a TRANSFER op; gate the next
            // compute-shader read with a TRANSFER → COMPUTE barrier.
            transfer_to_compute_barrier(dev, cmd);

            let (qkw_shader, qkw_bm, qkw_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, qkw_shader, wq,
                self.batch_norm.handle, self.batch_q.handle,
                q_dim, n_padded, hidden, qkw_bm, qkw_bn, "gemm_q_coopmat",
            );
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, qkw_shader, wk,
                self.batch_norm.handle, self.batch_k.handle,
                kv_dim, n_padded, hidden, qkw_bm, qkw_bn, "gemm_k_coopmat",
            );
            // gemm_v stays on mul_mmq — Qwen3 uses Q6_K for attn_v
            // (mixed-quant) and we don't ship a Q6_K coopmat dequant
            // kernel yet. Activations for gemm_v are still in batch_q8
            // thanks to the earlier run_quantize_q8_1 in the mul_mmq
            // path.
            self.run_gemm(
                dev, registry, cmd, sv, wv,
                gemm_input_attn, self.batch_v.handle,
                kv_dim, seq_len, hidden, "gemm_v",
            );
        } else {
            self.run_gemm(
                dev, registry, cmd, sq, wq,
                gemm_input_attn, self.batch_q.handle,
                q_dim, seq_len, hidden, "gemm_q",
            );
            self.run_gemm(
                dev, registry, cmd, sk, wk,
                gemm_input_attn, self.batch_k.handle,
                kv_dim, seq_len, hidden, "gemm_k",
            );
            self.run_gemm(
                dev, registry, cmd, sv, wv,
                gemm_input_attn, self.batch_v.handle,
                kv_dim, seq_len, hidden, "gemm_v",
            );
        }
        compute_barrier(dev, cmd);

        // ---- (d) Q/K-norm + RoPE + KV-cache write ----
        //
        // Two paths:
        //   * Phase 5B.3 fully-batched (default, when `batch_attn_enabled`):
        //     ONE dispatch each for Q-norm, K-norm, RoPE-Q, RoPE-K
        //     reading directly from / writing back to batch_q /
        //     batch_k, then ONE bulk `cmd_copy_buffer` per layer for
        //     K and V into the KV cache. Skips the per-token loop
        //     entirely. After this block batch_q holds post-RoPE Q
        //     for all M tokens; the attention call below consumes it.
        //
        //   * Per-token legacy (when `batch_attn_enabled = false`):
        //     One dispatch per (token, op) — same code path that
        //     shipped through Phase 5A. Gated below.
        let qk_norm_weights: Option<(vk::Buffer, vk::Buffer)> = if cfg.has_qk_norm {
            Some((
                layer_weight(model, layer, "attn_q_norm.weight"),
                layer_weight(model, layer, "attn_k_norm.weight"),
            ))
        } else {
            None
        };
        let batch_attn = self.batch_attn_enabled;

        if batch_attn {
            // ---- Phase 5B.3 fully-batched per-layer attention prep ----
            // Sprint 9c.5 — Q/K-norm + RoPE fused into a single
            // dispatch each (was: 4 dispatches + 2 barriers; now: 2
            // dispatches + 1 barrier per layer). Position-buffer is
            // pre-staged in prefill_batch with [base_pos, base_pos+1,
            // …, base_pos+M-1] at slots 0..M.
            //
            // The fused path requires a Q/K-norm weight to drive the
            // do_multiply branch. If the model has no qk_norm
            // (non-Qwen archs), fall back to the old separate
            // run_rope_batch dispatches with no rms_norm.
            if let Some((wqn, wkn)) = qk_norm_weights {
                self.run_rms_norm_mul_rope(
                    dev, registry, cmd,
                    self.batch_q.handle, wqn, self.batch_q.handle,
                    cfg.head_dim, cfg.n_heads, seq_len,
                    cfg.rms_norm_eps, "rms_norm_mul_rope_q_b",
                );
                self.run_rms_norm_mul_rope(
                    dev, registry, cmd,
                    self.batch_k.handle, wkn, self.batch_k.handle,
                    cfg.head_dim, cfg.n_kv_heads, seq_len,
                    cfg.rms_norm_eps, "rms_norm_mul_rope_k_b",
                );
            } else {
                // No qk_norm: keep the legacy stand-alone RoPE dispatches.
                self.run_rope_batch(
                    dev, registry, cmd,
                    self.batch_q.handle, self.batch_q.handle,
                    cfg.head_dim, cfg.n_heads, seq_len,
                    "rope_q_batch",
                );
                self.run_rope_batch(
                    dev, registry, cmd,
                    self.batch_k.handle, self.batch_k.handle,
                    cfg.head_dim, cfg.n_kv_heads, seq_len,
                    "rope_k_batch",
                );
            }
            compute_barrier(dev, cmd);

            // 3. Bulk KV-cache write. K and V are M contiguous rows of
            //    `[n_kv_heads, head_dim]` in batch_k / batch_v; the
            //    cache slot for this layer at positions
            //    `base_pos..base_pos+M` is the same shape.
            //
            // Sprint 9d.2 — when the cache is FP16-allocated, the
            // raw byte copy can't be used (it would copy FP32 bytes
            // into a half-size FP16 slot). We dispatch the
            // `kv_copy_fp16` compute shader instead, which converts
            // FP32 → packed-FP16 element-wise. FP32 cache stays on
            // the cheap vkCmdCopyBuffer path.
            let dst_off = self.kv_cache.pos_offset_bytes(layer, base_pos);
            let kv_elements = (seq_len as u32) * cfg.n_kv_heads * cfg.head_dim;
            if self.kv_cache.is_fp16() {
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.batch_k.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_k_b",
                );
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.batch_v.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_v_b",
                );
            } else {
                let kv_row_bytes = self.kv_cache.row_bytes();
                let bulk_size = (seq_len as u64) * kv_row_bytes;
                let copy_k = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(dst_off).size(bulk_size);
                let copy_v = copy_k;
                unsafe {
                    dev.device.cmd_copy_buffer(
                        cmd, self.batch_k.handle, self.kv_cache.k_buffer.handle,
                        std::slice::from_ref(&copy_k),
                    );
                    dev.device.cmd_copy_buffer(
                        cmd, self.batch_v.handle, self.kv_cache.v_buffer.handle,
                        std::slice::from_ref(&copy_v),
                    );
                }
            }
            // Barrier: subsequent attention reads KV (compute). The
            // upstream write was either a transfer (FP32 path) or a
            // compute dispatch (FP16 path) — cover both stages.
            let kv_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&kv_bar), &[], &[],
                );
            }
        }

        // Per-token loop only runs when batch_attn is OFF (legacy
        // path). When batch_attn is ON we already handled Q/K-norm,
        // RoPE, and the KV-cache write above; the loop body below
        // would do that per-token, which is exactly what we replaced.
        if !batch_attn {
        for t in 0..seq_len {
            let pos = base_pos + t;
            // RoPE position lives at slot `t` of rope_pos_buf — written
            // upfront in prefill_batch (see drift-fix comment there).
            let rope_pos_offset = (t as u64) * 4;
            // Pull token-row Q/K/V into single-token scratch.
            let q_off = (t as u64) * q_bytes;
            let kv_off = (t as u64) * kv_bytes;
            self.copy_batch_row(dev, cmd, self.batch_q.handle, q_off, self.q_buf.handle, q_bytes);
            self.copy_batch_row(dev, cmd, self.batch_k.handle, kv_off, self.k_buf.handle, kv_bytes);
            self.copy_batch_row(dev, cmd, self.batch_v.handle, kv_off, self.v_buf.handle, kv_bytes);
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
            }
            // Q/K-norm — Qwen-only.
            if let Some((wqn, wkn)) = qk_norm_weights {
                self.run_rms_norm(
                    dev, registry, cmd, self.q_buf.handle, wqn, self.q_buf.handle,
                    cfg.head_dim, cfg.n_heads, cfg.rms_norm_eps, "rms_norm_q_b",
                );
                self.run_rms_norm(
                    dev, registry, cmd, self.k_buf.handle, wkn, self.k_buf.handle,
                    cfg.head_dim, cfg.n_kv_heads, cfg.rms_norm_eps, "rms_norm_k_b",
                );
                compute_barrier(dev, cmd);
            }
            // RoPE — each dispatch reads its OWN position slot.
            self.run_rope_neox_with_pos_offset(
                dev, registry, cmd, self.q_buf.handle, self.q_buf.handle,
                cfg.head_dim, cfg.n_heads, pos, rope_pos_offset, "rope_q_b",
            );
            self.run_rope_neox_with_pos_offset(
                dev, registry, cmd, self.k_buf.handle, self.k_buf.handle,
                cfg.head_dim, cfg.n_kv_heads, pos, rope_pos_offset, "rope_k_b",
            );
            compute_barrier(dev, cmd);
            // KV-cache write at this token's position. Sprint 9d.3 —
            // FP16 KV path. This per-token legacy branch fires when
            // batch_attn_enabled=false (the
            // `phase5b2_batch_attn_parity_qwen3_*` regression tests
            // exercise this exact code path).
            let row_bytes = self.kv_cache.row_bytes();
            let dst_off = self.kv_cache.pos_offset_bytes(layer, pos);
            if self.kv_cache.is_fp16() {
                let kv_elements = cfg.n_kv_heads * cfg.head_dim;
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.k_buf.handle, self.kv_cache.k_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_k_t",
                );
                self.run_kv_copy_fp16(
                    dev, registry, cmd,
                    self.v_buf.handle, self.kv_cache.v_buffer.handle,
                    kv_elements, dst_off, "kv_copy_fp16_v_t",
                );
            } else {
                let copy = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(dst_off).size(row_bytes);
                unsafe {
                    dev.device.cmd_copy_buffer(
                        cmd, self.k_buf.handle, self.kv_cache.k_buffer.handle,
                        std::slice::from_ref(&copy),
                    );
                    dev.device.cmd_copy_buffer(
                        cmd, self.v_buf.handle, self.kv_cache.v_buffer.handle,
                        std::slice::from_ref(&copy),
                    );
                }
            }
            let kv_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&kv_bar), &[], &[],
                );
            }
            // Per-token attention path (legacy). seq_len for the
            // attention dispatch is pos+1 (causal — only KV
            // positions 0..=pos visible).
            self.run_scalar_attn(dev, registry, cmd, layer, pos);
            compute_barrier(dev, cmd);
            // Store attn_out[t] back into batch_attn_out.
            let copy_back = vk::BufferCopy::default()
                .src_offset(0).dst_offset(q_off).size(q_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, self.attn_out.handle, self.batch_attn_out.handle,
                    std::slice::from_ref(&copy_back),
                );
            }
            let pst = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&pst), &[], &[],
                );
            }
        }
        } // end if !batch_attn

        // ---- (d.5) Phase 5B.2 batched attention ----
        // Replaces M attention dispatches with one. Reads post-RoPE Q
        // from batch_q (staged in the loop above), K/V from the layer's
        // KV-cache slice (positions 0..=base_pos+seq_len-1), and
        // writes [seq_len, n_heads, head_dim] into batch_attn_out.
        //
        // Sprint 7 — VULKANFORGE_FA_TILED=1 routes through the Br>1
        // tiled-Q kernel (BR=4 queries per workgroup sharing a K-tile).
        // Default OFF; flash_attn_batch (Br=1) remains the proven path
        // until per-shape benches show tiled wins.
        if batch_attn {
            if self.fa_tiled_enabled {
                self.run_flash_attn_tiled(
                    dev, registry, cmd,
                    layer,
                    self.batch_q.handle,
                    self.batch_attn_out.handle,
                    seq_len,
                    base_pos,
                    base_pos + seq_len,
                );
            } else {
                self.run_flash_attn_batch(
                    dev, registry, cmd,
                    layer,
                    self.batch_q.handle,
                    self.batch_attn_out.handle,
                    seq_len,
                    base_pos,
                    base_pos + seq_len,
                );
            }
            compute_barrier(dev, cmd);
        }

        // ---- (e) Output projection: GEMM(attn_out → o_batch). ----
        let wo = layer_weight(model, layer, "attn_output.weight");
        if self.coopmat_q4k_enabled {
            // Coopmat path reads FP32 activations directly — skip the
            // q8_1 quantize for gemm_o. Pad N + zero tail.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_attn_out.handle,
                seq_len, n_padded, q_dim,
            );
            transfer_to_compute_barrier(dev, cmd);

            let (o_shader, o_bm, o_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, o_shader, wo,
                self.batch_attn_out.handle, self.batch_o.handle,
                hidden, n_padded, q_dim, o_bm, o_bn, "gemm_o_coopmat",
            );
        } else {
            let gemm_input_o = if use_mul_mm {
                self.batch_attn_out.handle
            } else {
                self.run_quantize_q8_1(
                    dev, registry, cmd,
                    self.batch_attn_out.handle, self.batch_q8.handle,
                    seq_len * q_dim, "quantize_attn_out",
                );
                compute_barrier(dev, cmd);
                self.batch_q8.handle
            };
            let so = layer_weight_shader_gemm(model, layer, "attn_output.weight", gemm_kind, hidden, seq_len, self.mul_mm_coopmat_enabled);
            self.run_gemm(
                dev, registry, cmd, so, wo,
                gemm_input_o, self.batch_o.handle,
                hidden, seq_len, q_dim, "gemm_o",
            );
        }
        compute_barrier(dev, cmd);

        // ---- (f+g) Fused residual-add + ffn_norm. v0.2 Sprint 9b
        // folds add_res1_b (batch_residual += batch_o, in-place) with
        // rms_norm_ffn_b (batch_norm = rms_norm(batch_residual) *
        // ffn_norm.weight) into one dispatch. `sum_out` aliases `a`
        // (both = batch_residual) for the in-place residual update.
        let w_ffn_norm = layer_weight(model, layer, "ffn_norm.weight");
        self.run_multi_add_rms(
            dev, registry, cmd,
            self.batch_residual.handle, self.batch_o.handle, w_ffn_norm,
            /* sum_out  = */ self.batch_residual.handle,
            /* norm_out = */ self.batch_norm.handle,
            hidden, seq_len, cfg.rms_norm_eps, "add_rms_ffn_b",
        );
        compute_barrier(dev, cmd);

        // ---- (h) Quantize FFN-norm output (mul_mmq path only). ----
        let gemm_input_ffn = if use_mul_mm {
            self.batch_norm.handle
        } else {
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_norm.handle, self.batch_q8.handle,
                seq_len * hidden, "quantize_ffn",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };

        // ---- (i) Gate + Up GEMMs. ----
        let wg = layer_weight(model, layer, "ffn_gate.weight");
        let wu = layer_weight(model, layer, "ffn_up.weight");
        if self.coopmat_q4k_enabled {
            // Both gemm_gate and gemm_up read batch_norm — already
            // padded for the attention-block coopmat dispatches at
            // the top of dispatch_layer_batch. The FFN-norm pass
            // *re*-writes batch_norm at this point so we have to
            // pad again.
            let n_padded = Self::pad_to_tile(seq_len, 16);
            self.zero_activation_tail(
                dev, cmd, self.batch_norm.handle,
                seq_len, n_padded, hidden,
            );
            transfer_to_compute_barrier(dev, cmd);

            let (gu_shader, gu_bm, gu_bn) = if seq_len <= 64 {
                (self.coopmat_naive_padded_shader(), 16u32, 16u32)
            } else if seq_len <= 128 {
                (ShaderId::MulCoopmatQ4KFwdBn32, 64u32, 32u32)
            } else {
                (ShaderId::MulCoopmatQ4KFwdBn64, 64u32, 64u32)
            };
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, gu_shader, wg,
                self.batch_norm.handle, self.batch_gate.handle,
                ffn, n_padded, hidden, gu_bm, gu_bn, "gemm_gate_coopmat",
            );
            self.run_gemm_coopmat_q4k(
                dev, registry, cmd, gu_shader, wu,
                self.batch_norm.handle, self.batch_up.handle,
                ffn, n_padded, hidden, gu_bm, gu_bn, "gemm_up_coopmat",
            );
        } else {
            let sg = layer_weight_shader_gemm(model, layer, "ffn_gate.weight", gemm_kind, ffn, seq_len, self.mul_mm_coopmat_enabled);
            let su = layer_weight_shader_gemm(model, layer, "ffn_up.weight", gemm_kind, ffn, seq_len, self.mul_mm_coopmat_enabled);
            self.run_gemm(
                dev, registry, cmd, sg, wg,
                gemm_input_ffn, self.batch_gate.handle,
                ffn, seq_len, hidden, "gemm_gate",
            );
            self.run_gemm(
                dev, registry, cmd, su, wu,
                gemm_input_ffn, self.batch_up.handle,
                ffn, seq_len, hidden, "gemm_up",
            );
        }
        compute_barrier(dev, cmd);

        // ---- (j) Fused SwiGLU: batch_ffn_hidden = silu(gate) * up. ----
        // v0.2 Sprint 9a — replaces the silu(gate→gate) + barrier +
        // mul(gate, up→ffn_hidden) pair with a single dispatch.
        self.run_swiglu(
            dev, registry, cmd,
            self.batch_gate.handle, self.batch_up.handle, self.batch_ffn_hidden.handle,
            seq_len * ffn, "swiglu_b",
        );
        compute_barrier(dev, cmd);

        // ---- (k) Quantize ffn_hidden + Down-proj GEMM (Q4_K). ----
        // NOTE: gemm_down is left on mul_mmq even when coopmat is on.
        // The coopmat path produced NaN logits when all 6 Q4_K GEMMs
        // were swapped — bisect localised the divergence to gemm_down
        // (K = ffn = 11008, the longest K-chain in the model). Sprint
        // 3C will revisit this with header-caching for the 11008/256
        // = 43 blocks per row, and/or a per-row scaling pass.
        let gemm_input_down = if use_mul_mm {
            self.batch_ffn_hidden.handle
        } else {
            self.run_quantize_q8_1(
                dev, registry, cmd,
                self.batch_ffn_hidden.handle, self.batch_q8.handle,
                seq_len * ffn, "quantize_ffn_h",
            );
            compute_barrier(dev, cmd);
            self.batch_q8.handle
        };
        let wd = layer_weight(model, layer, "ffn_down.weight");
        let sd = layer_weight_shader_gemm(model, layer, "ffn_down.weight", gemm_kind, hidden, seq_len, self.mul_mm_coopmat_enabled);
        self.run_gemm(
            dev, registry, cmd, sd, wd,
            gemm_input_down, self.batch_ffn_out.handle,
            hidden, seq_len, ffn, "gemm_down",
        );
        compute_barrier(dev, cmd);

        // ---- (l) Residual2 = residual + ffn_out + (cross-layer fuse). ----
        // Sprint 9b.2 — when there's a next layer, the residual update
        // is fused with that layer's `attn_norm` rms_norm-mul, putting
        // the next layer's pre-attn norm into batch_norm in the same
        // dispatch (and saving 1 dispatch + 1 barrier per layer
        // boundary). For the final layer, fall back to a plain add.
        match next_attn_norm_weight {
            Some(w_next) => {
                self.run_multi_add_rms(
                    dev, registry, cmd,
                    self.batch_residual.handle, self.batch_ffn_out.handle, w_next,
                    /* sum_out  = */ self.batch_residual.handle,
                    /* norm_out = */ self.batch_norm.handle,
                    hidden, seq_len, cfg.rms_norm_eps, "add_rms_attn_next_b",
                );
            }
            None => {
                self.run_binary(
                    dev, registry, cmd, ShaderId::Add,
                    self.batch_residual.handle, self.batch_ffn_out.handle,
                    self.batch_residual.handle,
                    seq_len * hidden, "add_res2_b",
                );
            }
        }
        compute_barrier(dev, cmd);
        let _ = (kv_bytes, ffn_bytes); // some bytes locals only used by debug paths
    }
}

/// Phase 6/7 — pick the right GEMM shader for a given layer weight.
///
/// `gemm_kind` is the per-batch choice: `Mmq` (Q8_1 activations),
/// `MulMm` (FP32 activations, scalar B-loads), or `MulMmAligned`
/// (FP32 activations, vec4 B-loads — only safe when shader N is
/// divisible by 4). Mixed-quant in Qwen3 (`attn_v` + `ffn_down`
/// are Q6_K, the rest Q4_K) goes through both Q4_K and Q6_K paths.
///
/// Sprint 11C — when `gemm_kind == Mmq` and the dispatch shape would
/// fill the GPU at L-tile granularity, the L-tile pipeline (BM=BN=128)
/// is preferred over the default S-tile (BM=BN=64).
///
/// Empirical RDNA4 (RX 9070 XT, 64 CUs):
///   pp=128 (n=128, groups_y=1)  L-tile starved →  −27 % vs S
///   pp=256 (n=256, groups_y=2)  L-tile marginal →  −4 % vs S
///   pp=512 (n=512, groups_y=4)  L-tile fills    →  +4 %
///   pp≥1024                     L-tile dominates →  +4–5 %
///
/// Threshold pragmatically pinned at `m > 128 && n > 256`: L-tile
/// only when the dispatch produces ≥64 workgroups, matching the CU
/// count. At smaller shapes the S-tile keeps its dispatch density
/// advantage. (llama.cpp picks at `m<=64 || n<=64`; we're stricter
/// because we don't ship an M-tile fallback yet — Sprint 11D may add
/// one.)
///
/// `VULKANFORGE_DISABLE_L_TILE=1` forces every dispatch back to S.
fn layer_weight_shader_gemm(
    model: &LoadedModel,
    layer: u32,
    suffix: &str,
    gemm_kind: GemmKind,
    m: u32,
    n: u32,
    coopmat_q4k_mm: bool,
) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    let q6 = model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type
        == GgmlType::Q6K;
    let prefer_l = m > 128 && n > 256
        && std::env::var("VULKANFORGE_DISABLE_L_TILE")
            .map(|s| s != "1").unwrap_or(true);
    match (gemm_kind, q6) {
        (GemmKind::MulMmAligned, true)  => ShaderId::MulMmQ6KAligned,
        (GemmKind::MulMmAligned, false) => ShaderId::MulMmQ4KAligned,
        (GemmKind::MulMm,        true)  => ShaderId::MulMmQ6K,
        (GemmKind::MulMm,        false) => if coopmat_q4k_mm { ShaderId::MulMmQ4KCoopmat } else { ShaderId::MulMmQ4K },
        (GemmKind::Mmq,          true)  => if prefer_l { ShaderId::MulMmqQ6KL } else { ShaderId::MulMmqQ6K },
        (GemmKind::Mmq,          false) => if prefer_l { ShaderId::MulMmqQ4KL } else { ShaderId::MulMmqQ4K },
    }
}

/// Per-batch GEMM dispatch kind. Decided once per `dispatch_layer_batch`
/// call from the configured `mul_mm_enabled` flag and the runtime
/// `seq_len % 4` alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmKind {
    /// `mul_mmq.comp`: Q8_1-quantized activations. Always valid.
    Mmq,
    /// `mul_mm.comp`: FP32 activations, scalar B-loads. Reachable
    /// only via diagnostic / parity tests; production routes either
    /// `Mmq` or `MulMmAligned`.
    #[allow(dead_code)]
    MulMm,
    /// `mul_mm.comp` with `ALIGNED=1 / LOAD_VEC_B=4 / B_TYPE=vec4`.
    /// Used when `mul_mm_enabled` and `seq_len % 4 == 0`. The
    /// vec4 B-load path skips the unaligned bounds check, so this
    /// is unsafe at unaligned `seq_len`.
    MulMmAligned,
}

#[allow(dead_code)] // Kept for parity with `layer_weight_shader_gemm`; still referenced
                    // by the `forward_layer_debug` helper paths in older diagnostic builds.
fn layer_weight_shader_mmq(model: &LoadedModel, layer: u32, suffix: &str) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    match model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type
    {
        GgmlType::Q6K => ShaderId::MulMmqQ6K,
        _ => ShaderId::MulMmqQ4K,
    }
}

fn layer_weight(model: &LoadedModel, layer: u32, suffix: &str) -> vk::Buffer {
    let key = format!("blk.{layer}.{suffix}");
    model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .buffer
        .handle
}

/// Q4_K_M mixes quant types — `attn_v.weight` and `ffn_down.weight`
/// are Q6_K, the rest are Q4_K. Pick the matching GEMV pipeline.
fn layer_weight_shader(model: &LoadedModel, layer: u32, suffix: &str) -> ShaderId {
    let key = format!("blk.{layer}.{suffix}");
    match model
        .tensor(&key)
        .unwrap_or_else(|| panic!("missing tensor '{key}'"))
        .ggml_type
    {
        GgmlType::Q6K => ShaderId::MulMatVecQ6K,
        _ => ShaderId::MulMatVecQ4K,
    }
}

fn compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

/// Sprint 3C — sync a TRANSFER (e.g. `cmd_fill_buffer`) so its writes
/// are visible to the next compute shader read.
fn transfer_to_compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb),
            &[], &[],
        );
    }
}

