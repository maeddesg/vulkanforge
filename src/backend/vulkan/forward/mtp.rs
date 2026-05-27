//! Sprint F.1 — Multi-Token-Prediction (MTP) self-speculative decode
//! support for Qwen3.6 (`qwen35`, Gated-Delta-Net hybrid).
//!
//! **Phase-4 spike (this file's current scope): recurrent-state
//! snapshot / restore.** Self-speculative decode drafts N tokens with
//! the `nextn` head, verifies them in a batch=N forward, and — when a
//! draft is rejected — must roll the model back to the last accepted
//! position. The make-or-break (F.1 R3) is that the Gated-Delta-Net
//! recurrent state is updated **destructively in-place**
//! (`step_gated_delta_net` copies the new state back over the old at a
//! fixed per-layer offset), so a rejected draft cannot be undone unless
//! the pre-draft state was snapshotted.
//!
//! The persistent recurrent surface is exactly two GpuOnly buffers:
//! `ssm_state_buf` (~144 MB on 27B) and `conv_state_buf` (~5.6 MB).
//! Every other `ssm_*` buffer is per-token scratch (recomputed each
//! forward) and needs no snapshot. The KV cache of the 17 full-attn
//! layers needs **no** GPU truncation: a rejected token's KV slot at
//! `pos` is overwritten by the real forward at the same `pos`; only the
//! CPU `current_seq_len` counter must be rewound (see
//! `set_kv_seq_len`).
//!
//! Snapshot buffers are allocated (mirroring the two live buffers) only
//! when `cfg.qwen35.is_some()` **and** `VF_MTP` / `VF_MTP_ROLLBACK_TEST`
//! is set (`setup.rs`), so non-MTP runs pay nothing.

use ash::vk;

use super::super::commands::CommandContext;
use super::super::device::VulkanDevice;
use super::state::Forward;

impl Forward {
    /// True when the MTP rollback snapshot buffers were allocated
    /// (qwen35 + `VF_MTP`/`VF_MTP_ROLLBACK_TEST`).
    pub(crate) fn mtp_snapshot_ready(&self) -> bool {
        self.ssm_state_snapshot_buf.is_some() || self.conv_state_snapshot_buf.is_some()
    }

    /// Current KV write position (`= number of tokens committed`).
    pub(crate) fn kv_seq_len(&self) -> u32 {
        self.kv_cache.current_seq_len
    }

    /// Rewind/advance the KV write position. Used by the rollback path
    /// to undo a throwaway forward's `current_seq_len += 1`; the stale
    /// KV slot itself is overwritten by the next real forward.
    pub(crate) fn set_kv_seq_len(&mut self, n: u32) {
        self.kv_cache.current_seq_len = n;
    }

    /// Copy the live recurrent state (`ssm_state_buf`, `conv_state_buf`)
    /// into the snapshot buffers. Standalone submit + wait.
    pub(crate) fn snapshot_recurrent_state(
        &mut self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.copy_recurrent(dev, cmd_ctx, true)
    }

    /// Restore the recurrent state from the snapshot buffers (inverse of
    /// `snapshot_recurrent_state`). Standalone submit + wait.
    pub(crate) fn restore_recurrent_state(
        &mut self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.copy_recurrent(dev, cmd_ctx, false)
    }

    /// Debug (Phase-4 spike): FNV-1a hash of a leading sample of the
    /// live `ssm_state_buf`, read back via a GpuOnly→GpuToCpu copy into
    /// the (dense-qwen35-unused) `moe_route_staging` buffer. Lets the
    /// rollback self-test prove snapshot/restore **byte fidelity**
    /// independent of decode's run-to-run nondeterminism: after a
    /// throwaway forward the hash must change, and after restore it must
    /// return to the snapshot-time value.
    pub(crate) fn mtp_debug_ssm_hash(
        &mut self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let n = match self.ssm_state_buf.as_ref() {
            Some(b) => b.size.min(self.moe_route_staging.size),
            None => return Ok(0),
        };
        let src = self.ssm_state_buf.as_ref().unwrap().handle;
        let dst = self.moe_route_staging.handle;
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            let region = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(n);
            unsafe {
                dev.device
                    .cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&region));
            }
        })?;
        let bytes = self.moe_route_staging.read_bytes()?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &bytes[..n as usize] {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(h)
    }

    /// Whole-buffer GPU→GPU copy of the two persistent recurrent
    /// buffers. `to_snapshot = true` saves (live → snapshot), `false`
    /// restores (snapshot → live). No-op when the snapshot buffers were
    /// not allocated.
    fn copy_recurrent(
        &mut self,
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
        to_snapshot: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // (src, dst, size) for each buffer that exists on both sides.
        let mut copies: Vec<(vk::Buffer, vk::Buffer, u64)> = Vec::with_capacity(2);
        if let (Some(live), Some(snap)) =
            (self.ssm_state_buf.as_ref(), self.ssm_state_snapshot_buf.as_ref())
        {
            let (src, dst) = if to_snapshot {
                (live.handle, snap.handle)
            } else {
                (snap.handle, live.handle)
            };
            copies.push((src, dst, live.size));
        }
        if let (Some(live), Some(snap)) =
            (self.conv_state_buf.as_ref(), self.conv_state_snapshot_buf.as_ref())
        {
            let (src, dst) = if to_snapshot {
                (live.handle, snap.handle)
            } else {
                (snap.handle, live.handle)
            };
            copies.push((src, dst, live.size));
        }
        // Diagnostic VF_MTP_RB_EMPTY: still submit a one_shot but with
        // NO copy commands — isolates "extra submit perturbs the
        // pipeline" from "copy content corrupts state".
        if std::env::var("VF_MTP_RB_EMPTY").as_deref() == Ok("1") {
            copies.clear();
        }
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            for (src, dst, size) in &copies {
                let region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(*size);
                unsafe {
                    dev.device
                        .cmd_copy_buffer(cmd, *src, *dst, std::slice::from_ref(&region));
                }
            }
        })?;
        Ok(())
    }
}
