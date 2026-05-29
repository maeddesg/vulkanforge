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
use super::super::gguf::GgmlType;
use super::super::loader::LoadedModel;
use super::super::pipeline_registry::PipelineRegistry;
use super::super::shaders::ShaderId;
use super::arch::common::{compute_barrier, transfer_to_compute_barrier};
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

    /// Sprint F.2a — run the qwen35 MTP **nextn draft head** for one
    /// token (n=1) and return its logits. Given the trunk's pre-final-
    /// norm hidden `h_t` (already captured in `mtp_h_buf`) and the
    /// embedding `e` of the just-produced token, predicts the
    /// distribution for the token AFTER it:
    ///
    /// `concat(RMSNorm(e,enorm) ∥ RMSNorm(h_t,hnorm))` → `eh_proj`
    /// (q8_0, 2·n_embd→n_embd) → block-64 full-attn transformer →
    /// `shared_head_norm` → `lm_head`. Block-64 attends over its OWN
    /// (layer-`n_main`) KV slot — zero-init'd lazily on first call so
    /// the cold prompt positions read zeros, not garbage (F.2a §1b).
    /// One standalone submit; reuses `cur()` scratch + `logits_buf`
    /// (safe — runs between trunk tokens). Default OFF (`VF_MTP_DRAFT`).
    pub(crate) fn mtp_draft_logits(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        e_embedding: &[f32],
        position: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n_main = self
            .config
            .qwen35
            .as_ref()
            .expect("mtp_draft_logits on non-qwen35")
            .n_main();

        // --- weights ---
        let tw = |name: String| model.tensor(&name).expect("MTP tensor missing");
        let enorm_w = tw(format!("blk.{n_main}.nextn.enorm.weight")).buffer.handle;
        let hnorm_w = tw(format!("blk.{n_main}.nextn.hnorm.weight")).buffer.handle;
        let ehp = tw(format!("blk.{n_main}.nextn.eh_proj.weight"));
        let ehp_w = ehp.buffer.handle;
        let ehp_scale = ehp.weight_scale.unwrap_or(1.0);
        // Sprint B Phase 2 — bucket sub-range for MTP eh_proj weight.
        let (ehp_off, ehp_sz) = if ehp.buffer.is_shared() {
            (ehp.byte_offset, ehp.byte_size)
        } else {
            (0u64, 0u64)
        };
        // shared_head_norm if present, else output_norm.
        let shn_w = model
            .tensor(&format!("blk.{n_main}.nextn.shared_head_norm.weight"))
            .or_else(|| model.tensor("output_norm.weight"))
            .expect("MTP head norm missing")
            .buffer
            .handle;
        let lm = model
            .tensor("output.weight")
            .or_else(|| model.tensor("token_embd.weight"))
            .expect("LM head present");
        let lm_scale = lm.weight_scale.unwrap_or(1.0);
        // Sprint B Phase 2 — bucket-aware lm_head lookup (same pattern
        // as decode.rs::dispatch_final): `named_weight_with_offset`
        // returns `(handle, 0, 0)` on the legacy path or the per-tensor
        // sub-range when the lm_head is packed in a T0 bucket.
        let (w_lm, w_lm_off, w_lm_sz) = if model.tensor("output.weight").is_some() {
            super::arch::named_weight_with_offset(model, "output.weight")
        } else {
            super::arch::named_weight_with_offset(model, "token_embd.weight")
        };
        let sub = self.mul_mat_vec_subgroup_enabled;
        let lm_shader = match (lm.ggml_type, sub) {
            (GgmlType::F8E4M3, _) => ShaderId::MulMatVecFp8,
            (GgmlType::F32, _) => ShaderId::MulMatVecF32,
            (GgmlType::F16, _) => ShaderId::MulMatVecF16,
            (GgmlType::Q6K, true) => ShaderId::MulMatVecQ6KSubgroup,
            (GgmlType::Q6K, false) => ShaderId::MulMatVecQ6K,
            (_, true) => ShaderId::MulMatVecQ4KSubgroup,
            (_, false) => ShaderId::MulMatVecQ4K,
        };
        let ehp_shader = if sub {
            ShaderId::MulMatVecQ8_0Subgroup
        } else {
            ShaderId::MulMatVecQ8_0
        };

        // --- lazy zero-init of block-64's cold KV slot (§1b) ---
        if !self.mtp_block64_kv_zeroed {
            self.kv_cache.zero_layer(dev, n_main)?;
            self.mtp_block64_kv_zeroed = true;
        }

        // --- host-write the token embedding ---
        self.mtp_e_buf
            .as_mut()
            .expect("mtp_e_buf")
            .write_bytes(bytemuck::cast_slice(e_embedding))?;

        // --- host-write the draft RoPE position ---
        // The qwen35 q/k-norm-rope step reads the position from
        // `rope_pos_buf[0]` (attention.rs:418 "pos is read from
        // rope_pos_buf"), NOT from `ctx.position`. The draft block must
        // RoPE Q/K at the draft position (pos+1); without this it reuses
        // the trunk's stale position → mis-rotated Q/K → degraded attn.
        self.cur_mut()
            .rope_pos_buf
            .write_bytes(bytemuck::bytes_of(&position))?;

        // --- gather Copy handles + the plan before the closure ---
        let e_buf = self.mtp_e_buf.as_ref().unwrap().handle;
        let h_buf = self.mtp_h_buf.as_ref().unwrap().handle;
        let enorm_buf = self.mtp_enorm_buf.as_ref().unwrap().handle;
        let hnorm_buf = self.mtp_hnorm_buf.as_ref().unwrap().handle;
        let cat_buf = self.mtp_cat_buf.as_ref().unwrap().handle;
        let scratch_a = self.cur().scratch_a.handle;
        let scratch_b = self.cur().scratch_b.handle;
        let hidden_norm = self.cur().hidden_norm.handle;
        let logits_buf = self.logits_buf.handle;
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;
        let hidden_bytes = (hidden as u64) * 4;
        let draft_plan = super::layer_plan::build_qwen35_draft_block_plan(&self.config)
            .expect("draft block plan");

        self.reset_barrier_state();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            // 1. e_norm = RMSNorm(e, enorm) ; h_norm = RMSNorm(h_t, hnorm)
            self.run_rms_norm(dev, registry, cmd, e_buf, enorm_w, enorm_buf,
                hidden, 1, eps, "mtp_enorm");
            self.run_rms_norm(dev, registry, cmd, h_buf, hnorm_w, hnorm_buf,
                hidden, 1, eps, "mtp_hnorm");
            // compute→transfer: norms (SHADER_WRITE) → copies (TRANSFER_READ)
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                dev.device.cmd_pipeline_barrier(cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(), std::slice::from_ref(&bar), &[], &[]);
            }
            // 2. concat e_norm ∥ h_norm (e first) into cat_buf
            let c0 = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(hidden_bytes);
            let c1 = vk::BufferCopy::default().src_offset(0).dst_offset(hidden_bytes).size(hidden_bytes);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, enorm_buf, cat_buf, std::slice::from_ref(&c0));
                dev.device.cmd_copy_buffer(cmd, hnorm_buf, cat_buf, std::slice::from_ref(&c1));
            }
            transfer_to_compute_barrier(dev, cmd);
            // 3. cur = eh_proj @ concat  (2·hidden → hidden) → scratch_a
            self.run_gemv(dev, registry, cmd, ehp_shader, ehp_w, ehp_off, ehp_sz,
                cat_buf, scratch_a,
                hidden * 2, hidden, ehp_scale, "mtp_eh_proj");
            compute_barrier(dev, cmd);
            // 4. block-64 transformer: scratch_a (inpSA) → scratch_b
            self.reset_barrier_state();
            let ctx = super::executor::ExecCtx {
                dev, registry, cmd, model, layer: n_main,
                mode: super::executor::ExecMode::Decode {
                    position, input: scratch_a, output: scratch_b,
                },
            };
            super::executor::DecodeExec.execute_layer(self, &draft_plan, &ctx);
            compute_barrier(dev, cmd);
            // 5. shared_head_norm → hidden_norm
            self.run_rms_norm(dev, registry, cmd, scratch_b, shn_w, hidden_norm,
                hidden, 1, eps, "mtp_shared_head_norm");
            compute_barrier(dev, cmd);
            // 6. lm_head GEMV → logits_buf ; then readback to logits_staging
            self.run_gemv(dev, registry, cmd, lm_shader, w_lm, w_lm_off, w_lm_sz,
                hidden_norm, logits_buf,
                hidden, vocab, lm_scale, "mtp_lm_head");
            self.record_logits_readback(dev, cmd);
        })?;

        self.logits()
    }
}
