//! Forward-pass orchestration for Qwen3 / Llama / Gemma-4 decode.
//!
//! One [`Forward`] instance owns:
//! - per-token scratch buffers (ping-pong + per-projection slots),
//! - the K/V cache,
//! - one long-lived descriptor pool, reset between forwards,
//! - tiny RoPE auxiliary buffers (pos / ff / indices),
//! - an optional [`crate::backend::vulkan::profiler::ShaderProfiler`].
//!
//! [`Forward::forward_token`] dispatches the embedding lookup → N
//! transformer layers → final RMSNorm → LM head and reads the logits
//! back. Each shader path gets a method on `Forward` that allocates
//! a descriptor set from the pool, writes it, and dispatches.
//!
//! Layer ordering (Qwen3 with QK-norm — Llama / Gemma-4 differ in the
//! per-layer pre/post-norm + V-norm + sliding-window mask):
//! ```text
//! input ─→ attn_norm ─→ Wq/Wk/Wv (3× GEMV)
//!         q ─→ q_norm ─→ RoPE-NeoX
//!         k ─→ k_norm ─→ RoPE-NeoX  ─→ KV cache (pos-major copy)
//!         v ────────────────────────→ KV cache
//!         attention (scalar_attn) ──→ Wo ─→ residual1
//!         ffn_norm ─→ gate, up (2× GEMV) ─→ silu(gate)·up ─→ Wdown
//!         residual1 + Wdown_out ──→ next-layer input
//! ```
//!
//! ## Module layout (Sprint 44B)
//!
//! The implementation is split across sibling files under this directory.
//! Each file holds an `impl Forward { ... }` block; Rust merges them at
//! typecheck time.
//!
//! - [`state`]    — `Forward`, `IntermediateSlot`, `DebugTarget`,
//!   profile / stats structs, and the descriptor-set cache key.
//! - [`harness`]  — `HarnessPipeline` (the four FP8 / lm_head dedicated
//!   pipeline quintuples).
//! - [`setup`]    — `Forward::new` / `new_with_prefill` / `destroy` /
//!   feature-flag setters.
//! - [`runs`]     — 33 per-shader GPU-dispatch helpers (`run_gemv`,
//!   `run_rms_norm`, `run_flash_attn_*`, …).
//! - [`decode`]   — per-token forward path (`forward_token`,
//!   `dispatch_layer`, `dispatch_final`, async-decode pipeline).
//! - [`prefill`]  — batched-prompt path (`prefill_batch`,
//!   `dispatch_layer_batch`).
//! - [`mid_frame`] — mid-frame submit-and-wait primitive used by
//!   Gemma-4 MoE routing for GPU→CPU Top-K readback (Sprint 51D-C).
//! - [`debug`]    — single-layer drivers + env-gated dump helpers.
//! - [`arch`]     — architecture-specific helpers (Gemma-4 KV-share /
//!   p-RoPE / sliding-window, layer_weight family, GEMM picker,
//!   barriers).
//!
//! This file (`mod.rs`) keeps only the bits every path needs: the
//! active-slot accessors (`cur` / `cur_mut`), descriptor-set allocation
//! + caching (`alloc_or_get_set`, `write_bindings`), barrier-elision
//! (`maybe_compute_barrier`, `mark_written`, `reset_barrier_state`),
//! the `profile` timestamp wrapper, and `reset_descriptor_pool_and_cache`.

use ash::vk;
use ash::vk::Handle;

use super::device::VulkanDevice;

mod arch;
mod debug;
mod decode;
mod executor;
mod harness;
mod layer_plan;
mod mid_frame;
mod prefill;
mod runs;
mod setup;
mod state;
pub use state::{
    DebugTarget, Forward, ForwardStats, ForwardTokenProfile, IntermediateSlot,
};

/// Sprint 56C-3 — global env-var switch for the GPU-direct MoE expert
/// FFN path (default ON; `VF_GPU_DIRECT_MOE=0` reverts to the legacy
/// CPU-readback). Cached after first read.
///
/// Lives in `forward/mod.rs` (not `executor/moe.rs`) because both
/// `executor::moe::step_moe_route` and `super::decode.rs::run_decode`
/// need to query it — the latter pairs the flag with async-decode
/// safety (mid_frame_submit_and_wait is only inserted when this
/// returns false, so async stays safe for MoE when the flag is on).
pub(crate) fn gpu_direct_moe_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("VF_GPU_DIRECT_MOE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}
use arch::compute_barrier;
use state::BindingSignature;

impl Forward {
    /// Sprint 15D — accessor for the active intermediate-buffer slot.
    /// `forward_token` (decode) toggles `current_slot` 0/1 per token.
    /// `prefill_batch` (single-shot per prompt) always uses `slots[0]`.
    #[inline]
    pub fn cur(&self) -> &IntermediateSlot {
        &self.slots[self.current_slot]
    }

    /// Mutable view of the active slot — used by host-write helpers
    /// (`scratch_a.write_bytes`, `rope_pos_buf.write_bytes`) where the
    /// backing memory-mapped buffer needs &mut access.
    #[inline]
    pub fn cur_mut(&mut self) -> &mut IntermediateSlot {
        &mut self.slots[self.current_slot]
    }


    // -------------------------------------------------------------
    // Descriptor-set allocation + caching, barrier elision, profiling.
    // -------------------------------------------------------------

    fn alloc_set(&self, dev: &VulkanDevice, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        unsafe { dev.device.allocate_descriptor_sets(&info) }
            .expect("descriptor_set alloc")[0]
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
            // Sprint 24-Inline — reset the dedicated FP8 per-channel pool
            // alongside the main one. Sets allocated last forward become
            // free again.
            dev.device.reset_descriptor_pool(
                self.fp8pc.pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
            // Sprint 35 — same for the block-wise FP8 pool.
            dev.device.reset_descriptor_pool(
                self.fp8bw.pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
            // Sprint 36 — same for the block-wise FP8 GEMM pool.
            dev.device.reset_descriptor_pool(
                self.fp8bwgemm.pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
            // Sprint 29 — same for the lm_head dedicated pool.
            dev.device.reset_descriptor_pool(
                self.lmhead.pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        self.set_cache.clear();
        // Sprint 30 — pool reset just invalidated every cached fp8pc
        // descriptor set; drop the keys.
        self.fp8pc_ds_cache.clear();
        // Sprint 35 — same for the block-wise cache.
        self.fp8bw_ds_cache.clear();
        // Sprint 36 — same for the block-wise GEMM cache.
        self.fp8bwgemm_ds_cache.clear();
        // Sprint 12D — also reset the barrier-elision tracker so the
        // first dispatch in the next forward can't see stale dirty
        // flags from a previous (already-fence-waited) submit.
        self.reset_barrier_state();
        Ok(())
    }

    /// Sprint 12D — mark `bufs` as pending-write so the next
    /// `maybe_compute_barrier` that reads any of them will fire a
    /// barrier. No-op when elision is disabled (legacy unconditional
    /// path doesn't need the tracker).
    #[inline]
    fn mark_written(&mut self, bufs: &[vk::Buffer]) {
        if self.elision_disabled {
            return;
        }
        for b in bufs {
            self.pending_writes.insert(b.as_raw());
        }
    }

    /// Sprint 12D — issue a `compute_barrier` only when at least one of
    /// `reads` is in the pending-write set. After issuance, every dirty
    /// flag clears (the barrier is a global `VkMemoryBarrier`).
    /// Returns `true` if a barrier was actually emitted (telemetry).
    fn maybe_compute_barrier(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        reads: &[vk::Buffer],
    ) -> bool {
        self.barrier_stats_checked = self.barrier_stats_checked.saturating_add(1);
        if self.elision_disabled {
            compute_barrier(dev, cmd);
            self.barrier_stats_issued = self.barrier_stats_issued.saturating_add(1);
            return true;
        }
        let any_dirty = reads.iter().any(|b| self.pending_writes.contains(&b.as_raw()));
        if any_dirty {
            compute_barrier(dev, cmd);
            self.pending_writes.clear();
            self.barrier_stats_issued = self.barrier_stats_issued.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Sprint 12D — flush dirty state at the end of every forward so
    /// the next forward starts with a clean slate (the
    /// `vkWaitForFences` between submits guarantees ordering anyway,
    /// so leaving stale entries would only over-issue barriers, not
    /// cause a race).
    #[inline]
    fn reset_barrier_state(&mut self) {
        self.pending_writes.clear();
    }

    /// Public accessor for the barrier-elision counters (used by
    /// `examples/run_pp_bench` and the parity tests; not part of any
    /// hot path).
    pub fn barrier_stats(&self) -> (u64, u64) {
        (self.barrier_stats_checked, self.barrier_stats_issued)
    }

    /// Public accessor for the elision flag.
    pub fn barrier_elision_active(&self) -> bool {
        !self.elision_disabled
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


}


