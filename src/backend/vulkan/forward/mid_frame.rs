//! Sprint 51D-C — mid-frame submit-and-wait pattern.
//!
//! Gemma-4 MoE decode requires a GPU→CPU sync in the middle of
//! `forward_token`'s otherwise-monolithic command buffer: the router
//! output (Top-K expert indices) must reach the host so the per-token
//! per-expert dispatches can be recorded with the right buffer offsets.
//!
//! ## What this primitive does
//!
//! Given the currently-recording command buffer `cmd`:
//! 1. `vkEndCommandBuffer(cmd)`
//! 2. `vkQueueSubmit(cmd, mid_frame_fence)`
//! 3. `vkWaitForFences(mid_frame_fence)`  ← GPU work observable on host
//! 4. `vkResetCommandBuffer(cmd, 0)`
//! 5. `vkBeginCommandBuffer(cmd, ONE_TIME_SUBMIT)`  ← caller continues
//!
//! `vkResetCommandBuffer` rewinds state but keeps the same handle, so
//! `ExecCtx.cmd` (a `vk::CommandBuffer`, which is `Copy` and `u64`-sized)
//! stays valid for the rest of the recording without a return-value
//! dance through the executor's match-loop.
//!
//! ## Why a dedicated fence
//!
//! The outer `commands::CommandContext::one_shot` does `reset_fences` +
//! `queue_submit(.., cmd_ctx.fence)` + `wait_for_fences(.., cmd_ctx.fence)`
//! at frame end. If the mid-frame submit reused `cmd_ctx.fence` we'd have
//! two distinct submits signalling the same fence object across one
//! closure call, with `reset_fences` interleaved in the middle — fragile.
//! `mid_frame_fence` is allocated alongside `prefill_fence` in
//! `Forward::new`; its lifetime is fully contained inside this helper,
//! so a single fence reused across N MoE layers is correct.
//!
//! ## Pattern precedent in the codebase
//!
//! VFs prefill (`prefill.rs:175-260`, Sprint 19B-A) already does manual
//! `end_command_buffer` + `queue_submit` + reset+begin between chunk
//! boundaries — this helper is the same shape with `wait_for_fences`
//! added between submit and reset (because we actually need GPU→CPU
//! visibility, not just queue-ordering).

use ash::vk;

use super::super::device::VulkanDevice;
use super::state::Forward;

impl Forward {
    /// End the currently-recording CB, submit it on the compute queue,
    /// block on `mid_frame_fence`, then reset+begin the SAME CB so the
    /// caller can keep recording into `cmd` afterwards.
    ///
    /// The CB handle does not change (Vulkan reset is in-place), so
    /// callers do not need to thread a new handle through. After this
    /// call returns, GPU work submitted before the call is observable
    /// to the host (e.g., a host-mapped staging buffer holding router
    /// output is safe to read).
    ///
    /// **Caller responsibilities:**
    /// - Issue any required `cmd_pipeline_barrier` on the readback
    ///   buffer (e.g., COMPUTE → HOST) BEFORE invoking this helper.
    /// - Perform the host-side memcpy from the staging buffer AFTER
    ///   this returns.
    /// - Do not call from inside an `impl LayerExecutor` step that the
    ///   bit-id-sensitive non-MoE paths (Llama / Qwen3 / Gemma-4-E2B
    ///   without `enable_moe_block`) traverse.
    pub(super) fn mid_frame_submit_and_wait(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let device = &dev.device;
        let fence = self.mid_frame_fence;

        unsafe {
            // 1. End the current recording.
            device.end_command_buffer(cmd)?;

            // 2. Reset + submit. Reset BEFORE submit so a stale signalled
            //    state from a prior MoE layer in the same forward pass
            //    can't make `wait_for_fences` return immediately.
            device.reset_fences(std::slice::from_ref(&fence))?;
            let cmds_arr = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds_arr);
            device.queue_submit(
                dev.compute_queue,
                std::slice::from_ref(&submit),
                fence,
            )?;

            // 3. Block until the GPU work above is observable to the host.
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            // 4. Reset the CB in place. The handle stays valid; only
            //    recorded state is rewound. Pool flag
            //    RESET_COMMAND_BUFFER (set on `cmd_ctx.pool`) makes this
            //    legal; same flag is used by `commands::one_shot`.
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            // 5. Re-open the CB so the caller can keep recording.
            //    ONE_TIME_SUBMIT mirrors `one_shot`'s begin-info and is
            //    correct because this CB will be submitted exactly once
            //    more (at frame end via the outer `one_shot`).
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd, &begin)?;
        }

        // Sprint 12D barrier-elision tracker reset: every dispatch we
        // recorded before the mid-frame submit has now drained on the
        // GPU, so the "pending writes" set is no longer relevant — the
        // first dispatch in the freshly-begun CB has no RAW hazard
        // against them. Reset prevents the next call to
        // `maybe_compute_barrier` from skipping a needed barrier on the
        // (false-)assumption that a since-completed write is still
        // pending.
        self.reset_barrier_state();

        Ok(())
    }
}
