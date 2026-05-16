//! Sprint 44B-2 — Forward lifecycle (new / new_with_prefill / destroy /
//! feature-flag setters) extracted from `forward/mod.rs` (pure code-move).
//!
//! Lives in a sibling `impl Forward { ... }` block — Rust permits a type
//! to have multiple `impl` blocks in the same crate. Methods that touch
//! `pub(super)` fields (set via state.rs in 44B-1) keep working because
//! `setup` is a child of `forward`, and `super` of `state.rs` resolves
//! to `forward` — the same module that contains `setup`.

use std::collections::HashMap;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::super::buffers::GpuBuffer;
use super::super::device::VulkanDevice;
use super::super::gguf::ModelConfig;
use super::super::kv_cache::KvCache;
use super::super::pipeline::{
    Fp8BlockwiseGemmPushConstants, Fp8BlockwiseGemvPushConstants, MatVecPushConstants,
};
use super::super::profiler::ShaderProfiler;

use super::harness;
use super::state::{
    Forward, IntermediateSlot,
    MUL_COOPMAT_FP8_BN32_BLOCKWISE, MUL_COOPMAT_FP8_NATIVE_BN32_BLOCKWISE,
    MUL_MAT_VEC_F16_LMHEAD, MUL_MAT_VEC_FP8_BLOCKWISE, MUL_MAT_VEC_FP8_PERCHANNEL,
};

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

        // Sprint 15D — allocate per-slot buffer suite. Each
        // IntermediateSlot holds the 17 per-forward scratch buffers;
        // we build two slots so async pipelined decode (Sprint 15E)
        // can record CB[N+1] while CB[N] runs on the GPU without
        // racing on shared scratch.
        //
        // Decode alternates `current_slot` 0/1 per token. Prefill
        // (`dispatch_layer_batch`) is single-shot per prompt and always
        // uses slots[0] — async pipelining wouldn't help a one-shot
        // submit anyway.
        let mut alloc_slot = |slot_idx: usize| -> Result<IntermediateSlot, Box<dyn std::error::Error>> {
            let suf = if slot_idx == 0 { "" } else { "_s1" };
            let scratch_a = mk_storage(hidden_bytes, MemoryLocation::CpuToGpu, &format!("scratch_a{suf}"))?;
            let scratch_b = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, &format!("scratch_b{suf}"))?;
            let hidden_norm = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, &format!("hidden_norm{suf}"))?;
            let q_buf = mk_storage(q_bytes, MemoryLocation::GpuOnly, &format!("q_buf{suf}"))?;
            let k_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, &format!("k_buf{suf}"))?;
            let v_buf = mk_storage(kv_bytes, MemoryLocation::GpuOnly, &format!("v_buf{suf}"))?;
            let attn_out = mk_storage(q_bytes, MemoryLocation::GpuOnly, &format!("attn_out{suf}"))?;
            let o_buf = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, &format!("o_buf{suf}"))?;
            let res1 = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, &format!("res1{suf}"))?;
            // Sprint 51D-R — `gate_buf` is reused as the post-attn-norm
            // scratch (`step_post_attn_norm` writes `cfg.hidden_dim`
            // floats into it before `step_gate_proj` overwrites with
            // `ffn_dim` floats). On every architecture except
            // Gemma-4-26B-A4B, `ffn_dim ≥ hidden_dim`, so `ffn_bytes`
            // already covers the worst case. 26B has `ffn_dim=2112 <
            // hidden_dim=2816`, dropping the last 704 PostAttnNorm
            // writes silently to OOB on the GPU and zeroing the tail
            // of `gate_buf`. The zeroed tail then propagates into
            // `step_attn_residual_add` (`res1 = input + gate_buf`)
            // and from there through every downstream RMSNorm. Mirror
            // the `ffn_hidden_bytes` `max(...)` pattern below. `up_buf`
            // gets the same treatment defensively (no current
            // hidden-sized write but cheap insurance against a future
            // analogous reuse).
            let gate_up_bytes = ffn_bytes.max(hidden_bytes);
            let gate_buf = mk_storage(gate_up_bytes, MemoryLocation::GpuOnly, &format!("gate_buf{suf}"))?;
            let up_buf = mk_storage(gate_up_bytes, MemoryLocation::GpuOnly, &format!("up_buf{suf}"))?;
            // Sprint 51D-D — Gemma-4-26B-A4B has `intermediate_size=2112 <
            // hidden_size=2816`, so the standard `ffn_bytes` allocation
            // is too small for: (a) the MoE per-token accumulator that
            // collects K weighted expert outputs (each of size
            // `[hidden]`), and (b) the legacy 51D-B PostMoeNorm path
            // which reads `cols=hidden_dim` floats from `ffn_hidden`
            // (latent OOB on 26B, harmless on every other arch where
            // `ffn_dim ≥ hidden_dim`). `max(ffn_bytes, hidden_bytes)`
            // is unconditional: on Qwen3 / Llama / E2B `ffn_dim` is
            // already several × `hidden_dim`, so the `.max(…)` is a
            // no-op there.
            let ffn_hidden_bytes = ffn_bytes.max(hidden_bytes);
            let ffn_hidden = mk_storage(ffn_hidden_bytes, MemoryLocation::GpuOnly, &format!("ffn_hidden{suf}"))?;
            let ffn_out = mk_storage(hidden_bytes, MemoryLocation::GpuOnly, &format!("ffn_out{suf}"))?;
            // Phase 3E drift-fix: rope_pos_buf must hold one slot per
            // prefill token (otherwise the per-token host writes during
            // command-buffer recording all collapse to the last value
            // by the time the GPU executes — every token gets the same
            // RoPE position). Forward_token uses slot 0 of this buffer;
            // prefill_batch writes positions 0..seq_len into slots
            // 0..seq_len-1 before submit and binds with per-token offset.
            let rope_pos_buf = mk_storage(
                (max_prefill_tokens.max(1) as u64) * 4,
                MemoryLocation::CpuToGpu, &format!("rope_pos{suf}"),
            )?;
            // Phase-4C scratch for split-K attention. Sized for the
            // worst case at max_seq_len; per-call dispatches use only the
            // prefix that matches the current `n_tiles`. Sprint 15D —
            // double-buffered along with the rest of the per-forward
            // scratch.
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
                &format!("fa_scratch_out{suf}"),
            )?;
            let fa_scratch_max = mk_storage(
                fa_scratch_red_bytes,
                MemoryLocation::GpuOnly,
                &format!("fa_scratch_max{suf}"),
            )?;
            let fa_scratch_sum = mk_storage(
                fa_scratch_red_bytes,
                MemoryLocation::GpuOnly,
                &format!("fa_scratch_sum{suf}"),
            )?;
            // Sprint 43D-3 — per-slot per-layer-inputs staging
            // (Gemma-4 PLE). CpuToGpu so the host-side build runs
            // directly into the mapped page; layer dispatches read it
            // via descriptor binding (per-token offset added in 46D).
            //
            // Sprint 46D — sized `max_prefill_tokens × num_layers × hps × 4`
            // for Gemma-4 so the batch-prefill pre-stage can write all
            // M tokens' PLE inputs before CB record (host writes during
            // record are NOT serialised with GPU reads — see Sprint
            // 46C blocker analysis). Decode (`forward_token`) keeps
            // writing into the token-0 slot. 4-byte placeholder for
            // every other architecture.
            let ple_bytes: u64 = match config.gemma4.as_ref() {
                Some(g) if g.hidden_size_per_layer_input > 0 => {
                    (max_prefill_tokens.max(1) as u64)
                        * (config.n_layers as u64)
                        * (g.hidden_size_per_layer_input as u64)
                        * 4
                }
                // Sprint 51D-A — Gemma-4-26B-A4B has no PLE
                // (`hidden_size_per_layer_input == 0`). The legacy code
                // computed `0 * n_layers * 0 * 4 = 0` and tried to
                // allocate a zero-byte GpuBuffer, which `gpu-allocator`
                // rejects with `Invalid AllocationCreateDesc` (Vulkan
                // VUID-VkBufferCreateInfo-size-00912 also forbids
                // `size == 0`). Mirror the non-Gemma-4 branch and use a
                // 4-byte placeholder; the buffer is never read on the
                // 26B path because the layer plan has no `PleBlock`
                // step and the descriptor binding is always-on but
                // disabled by `ple_pc.has_ple = 0`.
                _ => 4,
            };
            let per_layer_inputs = mk_storage(
                ple_bytes,
                MemoryLocation::CpuToGpu,
                &format!("per_layer_inputs{suf}"),
            )?;
            Ok(IntermediateSlot {
                scratch_a, scratch_b, hidden_norm,
                q_buf, k_buf, v_buf, attn_out, o_buf, res1,
                gate_buf, up_buf, ffn_hidden, ffn_out,
                rope_pos_buf,
                fa_scratch_out, fa_scratch_max, fa_scratch_sum,
                per_layer_inputs,
            })
        };
        let slot0 = alloc_slot(0)?;
        let slot1 = alloc_slot(1)?;
        // Sprint 27 — `logits_buf` was `GpuToCpu` (host-mapped) for
        // direct CPU readback. On 14B-FP8 (vocab=152064) the lm_head
        // GEMV dispatched 152k scattered 4-byte writes through PCIe
        // and the BOTTOM_OF_PIPE fence stalled until they drained,
        // inflating per-token decode time by ~30 ms (vs 2.7 ms
        // BW-limited in the standalone shape bench).
        // Fix: keep lm_head's writes in fast GPU-local memory, then
        // copy the (small) logits buffer to a host-visible staging
        // buffer at end-of-forward.
        let logits_buf = mk_storage(logits_bytes, MemoryLocation::GpuOnly, "logits_buf")?;
        let logits_staging =
            mk_storage(logits_bytes, MemoryLocation::GpuToCpu, "logits_staging")?;
        // Sprint 40 Part 2 — host-readable staging for the CPU lm_head
        // offload. Sized for the post-norm hidden state (FP32, length
        // = hidden_dim). Cheap regardless of whether the offload is
        // active; keeps the buffer-management code path uniform.
        let hidden_bytes_size: u64 =
            (config.hidden_dim as u64) * std::mem::size_of::<f32>() as u64;
        // Sprint 43F — bump hidden_staging to 16× hidden_dim slots so
        // VF_BATCH_STEP_DUMP can capture 6+ intra-layer stages of
        // Layer 0's dispatch_layer_batch in a single run.
        // Sprint 43D-4 — bump again to 64 slots so VF_LAYER_DUMP_ALL
        // can capture every per-layer output (≤64 layers — covers all
        // shipped configs incl. Gemma-4-E2B at 35 + final-norm slot).
        // The CPU lm_head path only reads the first hidden_dim slot —
        // extra capacity is harmless when unused.
        let hidden_staging_size = hidden_bytes_size * 64;
        let hidden_staging = mk_storage(
            hidden_staging_size,
            MemoryLocation::GpuToCpu,
            "hidden_staging",
        )?;
        let fuse0 = mk_storage(16, MemoryLocation::GpuOnly, "fuse0_dummy")?;
        let fuse1 = mk_storage(16, MemoryLocation::GpuOnly, "fuse1_dummy")?;

        // (fa_scratch_out / max / sum are now allocated per slot above
        //  inside `alloc_slot`; rope_pos is also per-slot.)
        let rope_ff_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_ff_dummy")?;
        let rope_idx_buf = mk_storage(16, MemoryLocation::GpuOnly, "rope_idx_dummy")?;
        // Sprint 43D-4 — Gemma-4 V-norm gamma. Single buffer of 1.0
        // floats, length = max per-layer head_dim. Used as the gamma
        // operand to `run_rms_norm` so the shader effectively reduces
        // to the parameterless `v / rms(v)` formula HF applies.
        let vnorm_max_dim = config.head_dim.max(
            config.gemma4.as_ref()
                .and_then(|g| g.layers.iter().map(|l| l.head_dim).max())
                .unwrap_or(0),
        );
        let vnorm_bytes = (vnorm_max_dim as u64) * 4;
        let mut vnorm_ones = mk_storage(
            vnorm_bytes.max(4), MemoryLocation::CpuToGpu, "vnorm_ones",
        )?;
        {
            let ones: Vec<f32> = vec![1.0_f32; vnorm_max_dim as usize];
            vnorm_ones.write_bytes(bytemuck::cast_slice(&ones))?;
        }

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
        // Sprint 51D-D — same `max(ffn, hidden)` rationale as the
        // per-slot `ffn_hidden` above, applied per-token across the
        // full prefill batch.
        let batch_ffn_hidden_bytes = pp_ffn.max(pp_hidden);
        let batch_ffn_hidden = mk_storage(batch_ffn_hidden_bytes, MemoryLocation::GpuOnly,  "batch_ffn_hidden")?;
        let batch_ffn_out    = mk_storage(pp_hidden, MemoryLocation::GpuOnly,  "batch_ffn_out")?;

        // Sprint 51D-D — host-readable staging for MoE router input.
        // Sized for the worst case: full prefill batch × hidden × 4 B.
        // Cheap (~1.4 MB on 26B at pp=128); always allocated so the
        // buffer-management code stays uniform across MoE / non-MoE
        // models.
        let moe_route_staging_bytes = pp_hidden.max(
            (config.hidden_dim as u64) * 4,
        );
        let moe_route_staging = mk_storage(
            moe_route_staging_bytes,
            MemoryLocation::GpuToCpu,
            "moe_route_staging",
        )?;

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

        // Sprint 15E — async-mode CB pool, two CBs + two fences for the
        // 3-stage pipeline. Same compute queue family as the main path.
        let async_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(dev.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let async_pool = unsafe { device.create_command_pool(&async_pool_info, None)? };
        let async_cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(async_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        let async_cb_vec = unsafe { device.allocate_command_buffers(&async_cb_alloc)? };
        let async_cbs = [async_cb_vec[0], async_cb_vec[1]];
        // Fences start signaled so the first wait on a "fresh" slot
        // doesn't block (no CB has been submitted yet).
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let async_fences = [
            unsafe { device.create_fence(&fence_info, None)? },
            unsafe { device.create_fence(&fence_info, None)? },
        ];

        // Sprint 19B-A — multi-submit prefill pacing. Default-on at
        // interval=4 (i.e. submit every 4 layers, ~9 submits for a
        // 36-layer Qwen3). Sweep on Q4_K_M and Q3_K_M:
        //   pp=32   +7-8%   pp=64   +7-9%   pp=128  +5-5.4%
        //   pp=256  +3-3.5% pp=512  +1-1.4% pp=1024 +0.6%
        // Bit-exact output, decode untouched (forward_token doesn't go
        // through prefill_batch). Override:
        //   VULKANFORGE_PREFILL_SUBMIT_INTERVAL=0  → legacy single submit
        //   VULKANFORGE_PREFILL_SUBMIT_INTERVAL=N  → custom interval
        // If N >= n_layers we also fall back to single-submit because
        // the chunk count would be 1.
        let layers_per_submit = std::env::var("VULKANFORGE_PREFILL_SUBMIT_INTERVAL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4);
        let prefill_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(dev.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let prefill_pool = unsafe { device.create_command_pool(&prefill_pool_info, None)? };
        let prefill_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        // Sprint 51D-C — dedicated mid-frame fence for Gemma-4 MoE router
        // GPU→CPU readback. Lifetime is fully contained inside one
        // `mid_frame_submit_and_wait` call, so a single fence reused
        // across MoE layers is safe.
        let mid_frame_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let prefill_cbs = if layers_per_submit > 0 && layers_per_submit < config.n_layers {
            // ceil(n_layers / layers_per_submit) chunks for the layer
            // loop; the last chunk also carries the final-norm + lm_head
            // tail, so no extra CB is needed.
            let n_chunks = config.n_layers.div_ceil(layers_per_submit);
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(prefill_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(n_chunks);
            unsafe { device.allocate_command_buffers(&alloc)? }
        } else {
            Vec::new()
        };

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

        // Sprint 44C-3 — `batch_attn_enabled` was the toggle for the
        // per-token attention fallback in `dispatch_layer_batch`. The
        // executor (BatchExec) always uses flash-attn; the fallback is
        // gone. The `VULKANFORGE_BATCH_ATTN` env var is now a no-op.

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
        // Sprint 12M — coopmat is now DEFAULT-ON for prefill. The path
        // wins +11-64 % over mul_mmq across pp=64..2048 after Sprint 12K
        // (Q6_K coopmat) + 12L (LOAD_VEC_B=8 mat2x4 aligned) + 12M
        // (M-tile selector). Opt-out: VULKANFORGE_DISABLE_MM_COOPMAT=1
        // (or the legacy VULKANFORGE_USE_MM_COOPMAT=0 still disables).
        let mul_mm_coopmat_enabled = match (
            std::env::var("VULKANFORGE_DISABLE_MM_COOPMAT"),
            std::env::var("VULKANFORGE_USE_MM_COOPMAT"),
        ) {
            (Ok(v), _) if v == "1" || v.eq_ignore_ascii_case("true") => false,
            (_, Ok(v)) if v == "0" || v.eq_ignore_ascii_case("false") => false,
            _ => true, // default-on
        };
        let mul_mm_enabled = mul_mm_coopmat_enabled || match std::env::var("VULKANFORGE_USE_MUL_MM") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };

        // Sprint 13C — f16-accumulator coopmat opt-in. Selects the
        // ACC_TYPE=float16_t aligned coopmat SPVs for L-tile dispatches.
        // Default OFF until precision is validated end-to-end (FP16 acc
        // over K=12288 reductions in gemm_down is the worst case).
        let mul_mm_coopmat_f16acc_enabled = match std::env::var("VULKANFORGE_COOPMAT_F16ACC") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            _ => false,
        };

        // Sprint 14B — subgroupAdd ("Path A") GEMV reduction. Default
        // ON. The _subgroup SPVs replace the LDS tree-reduction
        // (6 barrier levels) with one wave-wide subgroupAdd, requiring
        // the Sprint 14A requiredSubgroupSize=64 pipeline pin to be
        // legally consumable.
        let mul_mat_vec_subgroup_enabled = match std::env::var("VULKANFORGE_DISABLE_SUBGROUP_GEMV") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => false,
            _ => true,
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

        // Sprint 24-Inline Step 0 — harness-style FP8 GEMV resources,
        // built fresh here with PipelineCache::null and a dedicated
        // descriptor pool. EXACTLY mirrors the layout in
        // examples/fp8_gemv_standalone.rs that's known to PASS.
        // Sprint 44B-2 — the {module, dsl, layout, pipeline, pool}
        // tuple is now built by `harness::HarnessPipeline::new`.
        let fp8pc = harness::HarnessPipeline::new(
            device,
            MUL_MAT_VEC_FP8_PERCHANNEL,
            /* n_bindings */ 4,
            /* push_size  */ std::mem::size_of::<MatVecPushConstants>() as u32,
            /* spec const */ Some(64),
            // Sprint 30 — pool sized for the *cache*, not for
            // every-call-fresh-allocation. With ~336 unique
            // (weight, input, output, scale) keys per forward × 2
            // async slots, 1024 sets covers worst case with comfortable
            // headroom. Memory cost ~50 KiB descriptor-set metadata
            // (down from ~33 MiB at 524288 in v0.3.5).
            /* max_sets   */ 1024,
        )?;

        // Sprint 35 — block-wise FP8 GEMV pipeline. Identical 4-binding
        // descriptor layout to fp8pc, different SPV + 6 × u32 push
        // constants (`Fp8BlockwiseGemvPushConstants`).
        // Pool sized like fp8pc: ~1024 sets cover the cache for
        // (weight, input, output, scale) keys × async slots.
        let fp8bw = harness::HarnessPipeline::new(
            device,
            MUL_MAT_VEC_FP8_BLOCKWISE,
            /* n_bindings */ 4,
            /* push_size  */ std::mem::size_of::<Fp8BlockwiseGemvPushConstants>() as u32,
            /* spec const */ Some(64),
            /* max_sets   */ 1024,
        )?;

        // Sprint 36 — block-wise FP8 GEMM (BN=32) pipeline. Same
        // 4-binding scheme as the GEMV variants; differs in the SPV
        // and the 9-u32 push constant block. No spec constant — the
        // BN=32 kernel hard-codes its tile shape. requiredSubgroupSize=64
        // (the BN=32 kernel is Wave64). Pool sized like fp8bw GEMV.
        let fp8bwgemm = harness::HarnessPipeline::new(
            device,
            MUL_COOPMAT_FP8_BN32_BLOCKWISE,
            /* n_bindings */ 4,
            /* push_size  */ std::mem::size_of::<Fp8BlockwiseGemmPushConstants>() as u32,
            /* spec const */ None,
            /* max_sets   */ 1024,
        )?;

        // Sprint 38 Part 2 — block-wise FP8 GEMM with native FP8 WMMA.
        // Parallel pipeline that reuses fp8bwgemm.dsl / fp8bwgemm.layout
        // (identical descriptor + push-constant layout) but binds the
        // native FP8 SPV. Selected at dispatch time by env flag, so we
        // pay the build cost of one extra pipeline (cheap) and skip
        // duplicating DSL/layout/pool. Stays inline (not a HarnessPipeline)
        // because it's a 2-handle aux pair, not a full quintuple.
        let (fp8bwgemm_native_shader_module, fp8bwgemm_native_pipeline) = unsafe {
            let words: Vec<u32> = MUL_COOPMAT_FP8_NATIVE_BN32_BLOCKWISE
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )?;
            let mut subgroup_info =
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                    .required_subgroup_size(64);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .push_next(&mut subgroup_info);
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(fp8bwgemm.layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0];
            (module, pipeline)
        };

        // Sprint 29 — dedicated F16 GEMV pipeline for lm_head. Mirrors
        // the production registry's MulMatVecF16 entry (BLOCK_SIZE=64
        // spec, requiredSubgroupSize=64, REQUIRE_FULL_SUBGROUPS, 5
        // bindings: weight + input + output + 2 fuse dummies) but with
        // `PipelineCache::null` and a small dedicated pool.
        // Pool: lm_head dispatched once per forward, but the
        // async-decode pipeline (pre_record / fill_embed_and_submit)
        // doesn't reset the pool between tokens — same issue as
        // Sprint 25B's fp8pc_pool. Match that pool's sizing
        // (524288 sets, ~25 MB) so a long generation can't exhaust
        // it. Reset happens at all 3 sync forward sites + inside
        // reset_descriptor_pool_and_cache (prefill).
        let lmhead = harness::HarnessPipeline::new(
            device,
            MUL_MAT_VEC_F16_LMHEAD,
            /* n_bindings */ 5,
            /* push_size  */ std::mem::size_of::<MatVecPushConstants>() as u32,
            /* spec const */ Some(64),
            /* max_sets   */ 524288,
        )?;

        Ok(Self {
            slots: [slot0, slot1],
            current_slot: 0,
            async_pool,
            async_cbs,
            async_fences,
            async_pending_record: None,
            prefill_pool,
            prefill_cbs,
            prefill_fence,
            layers_per_submit,
            mid_frame_fence,
            moe_route_staging,
            moe_routing: None,
            moe_routing_batch: None,
            // Sprint 56B — populated post-construction by
            // `Forward::init_moe_router_gpu(model)` (see decode.rs setup
            // path). `None` for non-MoE models and until the loader's
            // `moe_router_data` is uploaded.
            moe_router_gpu: None,
            logits_buf,
            logits_staging,
            hidden_staging,
            fuse0, fuse1,
            rope_ff_buf, rope_idx_buf, vnorm_ones,
            kv_cache, config,
            descriptor_pool,
            set_cache: HashMap::new(),
            cache_enabled,
            mul_mm_enabled,
            coopmat_q4k_enabled,
            mul_mm_coopmat_enabled,
            mul_mm_coopmat_f16acc_enabled,
            mul_mat_vec_subgroup_enabled,
            coopmat_fp8_enabled,
            fa_tiled_enabled,
            fa_tiled_br,
            fa_tiled_bc,
            coopmat_attn_enabled,
            profiler,
            rope_theta_scale, attn_scale,
            max_prefill_tokens,
            batch_input, batch_residual, batch_norm, batch_q8,
            batch_q, batch_k, batch_v, batch_attn_out, batch_o,
            batch_gate, batch_up, batch_ffn_hidden, batch_ffn_out,
            // Sprint 12D — barrier elision tracker.
            pending_writes: std::collections::HashSet::with_capacity(32),
            elision_disabled: std::env::var("VULKANFORGE_DISABLE_BARRIER_ELISION")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            barrier_stats_checked: 0,
            barrier_stats_issued: 0,
            fp8pc,
            fp8pc_ds_cache: HashMap::new(),
            fp8bw,
            fp8bw_ds_cache: HashMap::new(),
            fp8bwgemm,
            fp8bwgemm_ds_cache: HashMap::new(),
            fp8bwgemm_native_shader_module,
            fp8bwgemm_native_pipeline,
            lmhead,
            native_fp8_wmma: dev.native_fp8_wmma,
        })
    }

    /// Sprint 56B — upload `LoadedModel::moe_router_data` to GPU buffers
    /// and allocate per-prefill scratch space. Call once, after the
    /// model is loaded and before the first forward. No-op when the
    /// model has no router data (non-Gemma-4 or Gemma-4 without
    /// `enable_moe_block`).
    ///
    /// Buffer sizing (26B-A4B, hidden=2816, n_experts=128, top_k=8):
    /// - proj: 128 × 2816 × 4 = 1.37 MB per layer (25 MoE layers on 26B)
    /// - scale: 2816 × 4 = 11 KB per layer
    /// - pes: 128 × 4 = 512 B per layer
    /// - Total per-layer: ~1.38 MB × 25 layers ≈ 34.5 MB VRAM
    /// - logits_scratch: max_seq × 128 × 4 (= 2 MB at max_seq=4096)
    /// - indices/weights_scratch: max_seq × 8 × 4 each (= 128 KB each)
    /// - readback_staging: max_seq × 64 (host-visible)
    pub fn init_moe_router_gpu(
        &mut self,
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        model: &super::super::loader::LoadedModel,
        max_seq: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let router_data = match model.moe_router_data.as_ref() {
            Some(d) => d,
            None => return Ok(()),
        };
        let device = &dev.device;
        let n_layers = self.config.n_layers as usize;
        let n_experts = router_data.n_experts;
        let top_k = router_data.top_k;
        let hidden_size = router_data.hidden_size;

        let mk_storage = |allocator: &mut Allocator, size: u64, loc: MemoryLocation, name: &str|
            -> Result<GpuBuffer, Box<dyn std::error::Error>> {
            Ok(GpuBuffer::new(
                device, allocator, size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                loc, name,
            )?)
        };

        let mut layers: Vec<super::state::MoeRouterGpuLayer> = Vec::new();
        let mut layer_to_gpu_idx: Vec<u32> = vec![u32::MAX; n_layers];

        for (layer_idx, ld) in router_data.layers.iter().enumerate() {
            // Skip layers that have no real router data (proj empty =
            // non-MoE layer on a per-layer-gated config).
            if ld.proj.is_empty() {
                continue;
            }
            let proj_bytes = (ld.proj.len() as u64) * 4;
            let scale_bytes = (ld.scale.len() as u64) * 4;
            let pes_bytes = (ld.per_expert_scale.len() as u64) * 4;

            // Sprint 56B — these buffers are populated once at init and
            // never written again on the GPU; CpuToGpu (host-mapped) is
            // the right memory type to avoid a staging round-trip.
            let mut proj_buf = mk_storage(
                allocator, proj_bytes, MemoryLocation::CpuToGpu,
                &format!("moe_router_proj_L{layer_idx}"))?;
            let mut scale_buf = mk_storage(
                allocator, scale_bytes, MemoryLocation::CpuToGpu,
                &format!("moe_router_scale_L{layer_idx}"))?;
            let mut pes_buf = mk_storage(
                allocator, pes_bytes, MemoryLocation::CpuToGpu,
                &format!("moe_router_pes_L{layer_idx}"))?;
            proj_buf.write_bytes(bytemuck::cast_slice(&ld.proj))?;
            scale_buf.write_bytes(bytemuck::cast_slice(&ld.scale))?;
            pes_buf.write_bytes(bytemuck::cast_slice(&ld.per_expert_scale))?;

            layer_to_gpu_idx[layer_idx] = layers.len() as u32;
            layers.push(super::state::MoeRouterGpuLayer {
                proj: proj_buf,
                scale: scale_buf,
                pes: pes_buf,
            });
        }

        let logits_bytes = (max_seq as u64) * (n_experts as u64) * 4;
        let indices_bytes = (max_seq as u64) * (top_k as u64) * 4;
        let weights_bytes = indices_bytes;
        let readback_bytes = (max_seq as u64) * (top_k as u64) * 8; // u32 idx + f32 weight

        let logits_scratch = mk_storage(
            allocator, logits_bytes, MemoryLocation::GpuOnly, "moe_router_logits_scratch")?;
        let indices_scratch = mk_storage(
            allocator, indices_bytes, MemoryLocation::GpuOnly, "moe_router_indices_scratch")?;
        let weights_scratch = mk_storage(
            allocator, weights_bytes, MemoryLocation::GpuOnly, "moe_router_weights_scratch")?;
        let readback_staging = mk_storage(
            allocator, readback_bytes, MemoryLocation::GpuToCpu, "moe_router_readback_staging")?;

        // Sprint 61C — Phase 2' Expert-Grouped Dispatch scratch.
        // Sized for the worst-case prefill batch. All buffers are
        // STORAGE_BUFFER + TRANSFER_DST so cmd_update_buffer can
        // populate data_ids / data_counts directly from CB.
        //
        // Q8_1 packs 128 elements into 144 bytes; rounded up to whole
        // blocks. FP32 outputs are sized for (max_seq × top_k × M)
        // where M is 2*moe_int (gate_up), moe_int (glu_out), or
        // hidden_size (down_out).
        let mi = self.config.gemma4.as_ref()
            .map(|g| g.moe_intermediate_size)
            .unwrap_or(0);
        let q8_1_bytes = |n_elems: u64| -> u64 {
            // (n + 127) / 128 * 144
            ((n_elems + 127) / 128) * 144
        };
        let input_q8_bytes = q8_1_bytes((max_seq as u64) * (hidden_size as u64));
        let gate_up_out_bytes = (max_seq as u64) * (top_k as u64) * 2 * (mi as u64) * 4;
        let glu_out_bytes = (max_seq as u64) * (top_k as u64) * (mi as u64) * 4;
        let glu_q8_bytes = q8_1_bytes((max_seq as u64) * (top_k as u64) * (mi as u64));
        let down_out_bytes = (max_seq as u64) * (top_k as u64) * (hidden_size as u64) * 4;
        let data_ids_bytes = (max_seq as u64) * (top_k as u64) * 4;
        let data_counts_bytes = (n_experts as u64) * 4;

        let grouped_input_q8 = mk_storage(
            allocator, input_q8_bytes, MemoryLocation::GpuOnly, "moe_grouped_input_q8")?;
        let grouped_gate_up_out = mk_storage(
            allocator, gate_up_out_bytes, MemoryLocation::GpuOnly, "moe_grouped_gate_up_out")?;
        let grouped_glu_out = mk_storage(
            allocator, glu_out_bytes, MemoryLocation::GpuOnly, "moe_grouped_glu_out")?;
        let grouped_glu_q8 = mk_storage(
            allocator, glu_q8_bytes, MemoryLocation::GpuOnly, "moe_grouped_glu_q8")?;
        let grouped_down_out = mk_storage(
            allocator, down_out_bytes, MemoryLocation::GpuOnly, "moe_grouped_down_out")?;
        let grouped_data_ids = mk_storage(
            allocator, data_ids_bytes, MemoryLocation::GpuOnly, "moe_grouped_data_ids")?;
        let grouped_data_counts = mk_storage(
            allocator, data_counts_bytes, MemoryLocation::GpuOnly, "moe_grouped_data_counts")?;

        self.moe_router_gpu = Some(super::state::MoeRouterGpu {
            layers,
            layer_to_gpu_idx,
            logits_scratch,
            indices_scratch,
            weights_scratch,
            readback_staging,
            n_experts,
            top_k,
            hidden_size,
            rms_norm_eps: router_data.rms_norm_eps,
            grouped_input_q8,
            grouped_gate_up_out,
            grouped_glu_out,
            grouped_glu_q8,
            grouped_down_out,
            grouped_data_ids,
            grouped_data_counts,
            grouped_weights_host: std::sync::Mutex::new(Vec::with_capacity(
                (max_seq as usize) * (top_k as usize),
            )),
            grouped_data_ids_host: std::sync::Mutex::new(Vec::with_capacity(
                (max_seq as usize) * (top_k as usize),
            )),
            moe_intermediate: mi,
            max_seq,
        });
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // Sprint 44B-2 — harness-pipeline teardown collapsed into one
        // call per pipeline (Sprint 24-Inline / 35 / 36 / 29).
        self.fp8pc.destroy(device);
        self.fp8bw.destroy(device);
        // Sprint 38 Part 2 — native FP8 block-wise pipeline shares
        // dsl/layout/pool with fp8bwgemm; tear it down BEFORE the
        // shared harness so the pipeline-layout destroy below sees no
        // dangling pipeline.
        unsafe {
            device.destroy_pipeline(self.fp8bwgemm_native_pipeline, None);
            device.destroy_shader_module(self.fp8bwgemm_native_shader_module, None);
        }
        self.fp8bwgemm.destroy(device);
        self.lmhead.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            // Sprint 15E — async CB pool + fences.
            for f in self.async_fences {
                device.destroy_fence(f, None);
            }
            // CBs are freed implicitly with the pool.
            device.destroy_command_pool(self.async_pool, None);
            // Sprint 19B-A — multi-submit prefill pool + fence.
            device.destroy_fence(self.prefill_fence, None);
            device.destroy_command_pool(self.prefill_pool, None);
            // Sprint 51D-C — dedicated mid-frame submit fence.
            device.destroy_fence(self.mid_frame_fence, None);
        }
        // Sprint 15D — destroy both intermediate slots in turn.
        let [slot0, slot1] = self.slots;
        for s in [slot0, slot1] {
            s.scratch_a.destroy(device, allocator);
            s.scratch_b.destroy(device, allocator);
            s.hidden_norm.destroy(device, allocator);
            s.q_buf.destroy(device, allocator);
            s.k_buf.destroy(device, allocator);
            s.v_buf.destroy(device, allocator);
            s.attn_out.destroy(device, allocator);
            s.o_buf.destroy(device, allocator);
            s.res1.destroy(device, allocator);
            s.gate_buf.destroy(device, allocator);
            s.up_buf.destroy(device, allocator);
            s.ffn_hidden.destroy(device, allocator);
            s.ffn_out.destroy(device, allocator);
            s.rope_pos_buf.destroy(device, allocator);
            s.fa_scratch_out.destroy(device, allocator);
            s.fa_scratch_max.destroy(device, allocator);
            s.fa_scratch_sum.destroy(device, allocator);
            s.per_layer_inputs.destroy(device, allocator);
        }
        self.logits_buf.destroy(device, allocator);
        self.logits_staging.destroy(device, allocator);
        self.hidden_staging.destroy(device, allocator);
        self.fuse0.destroy(device, allocator);
        self.fuse1.destroy(device, allocator);
        self.rope_ff_buf.destroy(device, allocator);
        self.rope_idx_buf.destroy(device, allocator);
        self.vnorm_ones.destroy(device, allocator);
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
        // Sprint 51D-D — MoE router staging.
        self.moe_route_staging.destroy(device, allocator);
        // Sprint 56B — GPU-side MoE router buffers.
        if let Some(gpu) = self.moe_router_gpu {
            for layer in gpu.layers {
                layer.proj.destroy(device, allocator);
                layer.scale.destroy(device, allocator);
                layer.pes.destroy(device, allocator);
            }
            gpu.logits_scratch.destroy(device, allocator);
            gpu.indices_scratch.destroy(device, allocator);
            gpu.weights_scratch.destroy(device, allocator);
            gpu.readback_staging.destroy(device, allocator);
            // Sprint 61C — Phase 2' Expert-Grouped Dispatch scratch.
            gpu.grouped_input_q8.destroy(device, allocator);
            gpu.grouped_gate_up_out.destroy(device, allocator);
            gpu.grouped_glu_out.destroy(device, allocator);
            gpu.grouped_glu_q8.destroy(device, allocator);
            gpu.grouped_down_out.destroy(device, allocator);
            gpu.grouped_data_ids.destroy(device, allocator);
            gpu.grouped_data_counts.destroy(device, allocator);
        }
        self.kv_cache.destroy(device, allocator);
        if let Some(p) = self.profiler {
            p.destroy(device);
        }
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

    // Sprint 44C-3 — set_batch_attn_enabled / batch_attn_enabled
    // setter+getter removed: the per-token attention fallback the
    // toggle gated is gone. Tests that previously called the setter
    // either pass through the LayerExecutor unconditionally or were
    // removed alongside the legacy dispatch path.

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

}
