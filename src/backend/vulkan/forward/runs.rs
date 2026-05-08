//! Sprint 44B-3 — per-shader GPU-dispatch helpers extracted from
//! `forward/mod.rs`. Pure code-move.
//!
//! These are the ~30 `run_*` wrappers that each take buffer handles +
//! push constants + a `ShaderId`, allocate (or cache-fetch) a descriptor
//! set, and dispatch a single compute shader. They are called from the
//! orchestration paths in `forward/mod.rs` (`dispatch_layer`,
//! `dispatch_layer_batch`, `dispatch_layer_partial`, `dispatch_final`,
//! `prefill_batch`).
//!
//! Plus four utility helpers used only by the dispatch paths:
//! `pad_to_tile` (round seq_len up to the coopmat tile boundary),
//! `zero_activation_tail` (zero the padding rows so the kernel sees a
//! clean L-tile), `coopmat_naive_padded_shader` (route the padded
//! coopmat dispatch to the right SPV variant), and `copy_batch_row`
//! (extract a single row from `batch_*` into the per-token slot for
//! prefill_batch's last-token decode handoff).
//!
//! Visibility: every helper is `pub(super)` so the parent `forward/mod.rs`
//! can call them; nothing here is exposed beyond `forward`. The `impl
//! Forward { ... }` block in this file is merged with the one in
//! `forward/mod.rs` at typecheck time — Rust permits multiple `impl`
//! blocks for the same type within the same crate.

use ash::vk;
use ash::vk::Handle;

use super::super::device::VulkanDevice;
use super::super::pipeline::{
    CoopmatPushConstants, FlashAttnBatchPushConstants, FlashAttnReducePushConstants,
    FlashAttnSplitPushConstants, Fp8BlockwiseGemmPushConstants,
    Fp8BlockwiseGemvPushConstants, Fp8GemmPushConstants, GenericBinaryPushConstants,
    KvCopyFp16PushConstants, MatVecPushConstants, MmqPushConstants,
    MultiAddRmsPushConstants, Q8_1QuantizePushConstants, RmsNormMulRopePushConstants,
    RopePushConstants, ScalarAttnPushConstants, SwigluPushConstants,
};
use super::super::pipeline_registry::PipelineRegistry;
use super::super::shaders::ShaderId;

use super::arch::{compute_barrier, gemma4_kv_read_layer, gemma4_kv_start};
use super::state::Forward;

impl Forward {
    /// Sprint 24-Inline Step 0 — harness-style FP8 per-channel GEMV
    /// dispatch, using the dedicated `fp8pc_*` resources created in
    /// `Forward::new` with `PipelineCache::null` and a 4-binding DSL.
    /// Sprint 30 — descriptor sets are now cached by binding tuple,
    /// so the per-dispatch `vkAllocateDescriptorSets` /
    /// `vkUpdateDescriptorSets` cost is paid only on the first
    /// occurrence of a given (weight, input, output, scale) combo.
    /// Across decode tokens with the same model loaded, every call
    /// after warm-up is a pure cache hit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv_fp8_perchannel(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        scale: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        label: &str,
    ) {
        // Sprint 30 — cache lookup. Key = the four bound buffers.
        // Layout, range, and offset are constant for this dispatch
        // type (whole-buffer reads from binding 0..3) so they don't
        // need to factor into the key.
        let key = (weights.as_raw(), input.as_raw(), output.as_raw(), scale.as_raw());
        let set = if let Some(&cached) = self.fp8pc_ds_cache.get(&key) {
            cached
        } else {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.fp8pc.pool)
                .set_layouts(std::slice::from_ref(&self.fp8pc.dsl));
            let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info) }
                .expect("fp8pc descriptor set alloc")[0];

            let infos = [
                vk::DescriptorBufferInfo::default().buffer(weights).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(input).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(output).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(scale).offset(0).range(vk::WHOLE_SIZE),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..4)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
            self.fp8pc_ds_cache.insert(key, set);
            set
        };

        let pc = MatVecPushConstants {
            ncols: k, stride_a: k, stride_b: k, stride_d: m,
            batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
            fusion_flags: 0, base_work_group_y: 0,
            ne02: 1, ne12: 1, broadcast2: 1,
            broadcast3: 1u32,
        };
        let pipeline = self.fp8pc.pipeline;
        let layout = self.fp8pc.layout;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, m, 1, 1);
        });
    }

    /// Sprint 35 — block-wise FP8 GEMV. Same descriptor-set scheme as
    /// the per-channel variant (4 storage buffers: weight, input,
    /// output, scale) so the cache-key shape is identical. The scale
    /// SSBO is interpreted as a row-major `[N/block_n, K/block_k]`
    /// FP32 grid; the shader recomputes
    /// `scale[(row/block_n) * num_kblocks + b]` per K-block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv_fp8_blockwise(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        scale: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        block_n: u32,
        block_k: u32,
        label: &str,
    ) {
        self.run_gemv_fp8_blockwise_at(
            dev, cmd, weights, scale, input, output, k, m, block_n, block_k, 0, 0, label,
        );
    }

    /// Sprint 35 — block-wise GEMV with explicit input/output token
    /// offsets. The looped prefill fallback (`run_gemm_fp8_blockwise_via_gemv_loop`)
    /// dispatches one of these per prompt token, stepping through the
    /// stacked activation buffer without rebinding descriptor sets.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv_fp8_blockwise_at(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        scale: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        block_n: u32,
        block_k: u32,
        input_off_floats: u32,
        output_off_floats: u32,
        label: &str,
    ) {
        debug_assert!(k % block_k == 0, "k {k} not divisible by block_k {block_k}");
        let key = (weights.as_raw(), input.as_raw(), output.as_raw(), scale.as_raw());
        let set = if let Some(&cached) = self.fp8bw_ds_cache.get(&key) {
            cached
        } else {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.fp8bw.pool)
                .set_layouts(std::slice::from_ref(&self.fp8bw.dsl));
            let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info) }
                .expect("fp8bw descriptor set alloc")[0];
            let infos = [
                vk::DescriptorBufferInfo::default().buffer(weights).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(input).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(output).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(scale).offset(0).range(vk::WHOLE_SIZE),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..4)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
            self.fp8bw_ds_cache.insert(key, set);
            set
        };

        let pc = Fp8BlockwiseGemvPushConstants {
            ncols: k,
            stride_a: k,
            stride_d: m,
            block_size_n: block_n,
            block_size_k: block_k,
            num_kblocks: k / block_k,
            input_off_floats,
            output_off_floats,
        };
        let pipeline = self.fp8bw.pipeline;
        let layout = self.fp8bw.layout;
        self.profile(label, dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, m, 1, 1);
        });
    }

    /// Sprint 36 — block-wise FP8 GEMM (BN=32 cooperative matrix kernel).
    /// Replaces the Sprint 35 GEMV-loop fallback. Same descriptor scheme
    /// as the per-channel `MulCoopmatFp8Bn32` registry entry; differs in
    /// the SPV (`mul_coopmat_fp8_bn32_blockwise.comp` folds the per-block
    /// scale into the A-tile load) and the 9-u32 push constant layout.
    ///
    /// Constraints (asserted): BM=64 must divide block_n, BK=16 must
    /// divide block_k. Qwen3-FP8 and DeepSeek-V3-FP8 satisfy both with
    /// `weight_block_size=[128, 128]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemm_fp8_blockwise(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weight_buf: vk::Buffer,
        scale_buf: vk::Buffer,
        activations_fp32: vk::Buffer,
        output: vk::Buffer,
        m: u32,
        n: u32,
        k: u32,
        block_n: u32,
        block_k: u32,
        label: &'static str,
    ) {
        const BM: u32 = 64;
        const BN: u32 = 32;
        const BK: u32 = 16;
        debug_assert!(block_n % BM == 0,
            "block_n {block_n} must be a multiple of BM={BM} for blockwise GEMM");
        debug_assert!(block_k % BK == 0,
            "block_k {block_k} must be a multiple of BK={BK} for blockwise GEMM");
        debug_assert!(k % block_k == 0,
            "k {k} must be a multiple of block_k {block_k}");

        let key = (
            weight_buf.as_raw(),
            activations_fp32.as_raw(),
            output.as_raw(),
            scale_buf.as_raw(),
        );
        let set = if let Some(&cached) = self.fp8bwgemm_ds_cache.get(&key) {
            cached
        } else {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.fp8bwgemm.pool)
                .set_layouts(std::slice::from_ref(&self.fp8bwgemm.dsl));
            let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info) }
                .expect("fp8bwgemm descriptor set alloc")[0];
            let infos = [
                vk::DescriptorBufferInfo::default().buffer(weight_buf).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(activations_fp32).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(output).offset(0).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(scale_buf).offset(0).range(vk::WHOLE_SIZE),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..4)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
            self.fp8bwgemm_ds_cache.insert(key, set);
            set
        };

        let pc = Fp8BlockwiseGemmPushConstants {
            m, n, k,
            stride_a: k,
            stride_b: k,
            stride_c: m,
            block_n,
            block_k,
            num_kblocks: k / block_k,
        };
        let groups_x = (m + BM - 1) / BM;
        let groups_y = (n + BN - 1) / BN;
        // Sprint 38 Part 2 + Sprint 39 — `VF_FP8_NATIVE_WMMA=1` selects
        // the native FP8 cooperative-matrix pipeline. Sprint 38 Part 2
        // shipped with this routing disabled because the naive
        // FP32→FP8 cast on the B-tile destroyed too much dynamic range
        // for block-wise weights; Sprint 39 adds an in-shader
        // per-k_block dynamic activation absmax + rescale, so the
        // native path now produces coherent output and the routing is
        // re-enabled.
        let native_fp8_wmma = std::env::var("VF_FP8_NATIVE_WMMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let pipeline = if native_fp8_wmma {
            self.fp8bwgemm_native_pipeline
        } else {
            self.fp8bwgemm.pipeline
        };
        let layout = self.fp8bwgemm.layout;
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

    /// Sprint 35 — prefill fallback for block-wise FP8 weights.
    /// Superseded by `run_gemm_fp8_blockwise` (Sprint 36) for the
    /// common `[128, 128]` shape. Kept for shapes where BM∤block_n
    /// or BK∤block_k.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(super) fn run_gemm_fp8_blockwise_via_gemv_loop(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        scale: vk::Buffer,
        activations: vk::Buffer,
        output: vk::Buffer,
        n: u32,        // output rows per token
        seq_len: u32,  // M (token count)
        k: u32,        // input dim per token
        block_n: u32,
        block_k: u32,
        label: &'static str,
    ) {
        for t in 0..seq_len {
            self.run_gemv_fp8_blockwise_at(
                dev, cmd, weights, scale, activations, output,
                k, n, block_n, block_k,
                t * k, t * n,
                label,
            );
        }
    }

    /// Sprint 35 — routing wrapper: per-channel vs block-wise.
    /// Picks the right pipeline based on whether the caller passed
    /// a `block` tuple. Keeps the call sites in `dispatch_layer*`
    /// to one extra `layer_weight_scale_block` lookup + one method
    /// call.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv_fp8_dispatch(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        scale: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        block: Option<(u32, u32)>,
        label: &'static str,
    ) {
        match block {
            Some((bn, bk)) => self.run_gemv_fp8_blockwise(
                dev, cmd, weights, scale, input, output, k, m, bn, bk, label,
            ),
            None => self.run_gemv_fp8_perchannel(
                dev, cmd, weights, scale, input, output, k, m, label,
            ),
        }
    }

    /// Sprint 29 — dedicated F16 GEMV dispatch for lm_head. Mirrors
    /// the production `run_gemv` for `MulMatVecF16` (same SPV, same
    /// spec / requiredSubgroupSize, same push-constant struct, same
    /// 5-binding scheme with fuse0/fuse1 dummies, same one-WG-per-row
    /// dispatch geometry) but uses the `lmhead_*` resources built
    /// with `PipelineCache::null` and a dedicated descriptor pool.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv_lmhead_dedicated(
        &mut self,
        dev: &VulkanDevice,
        cmd: vk::CommandBuffer,
        weights: vk::Buffer,
        input: vk::Buffer,
        output: vk::Buffer,
        k: u32,
        m: u32,
        weight_scale: f32,
    ) {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.lmhead.pool)
            .set_layouts(std::slice::from_ref(&self.lmhead.dsl));
        let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info) }
            .expect("lmhead descriptor set alloc")[0];
        let infos = [
            vk::DescriptorBufferInfo::default().buffer(weights).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(input).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(output).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(self.fuse0.handle).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(self.fuse1.handle).offset(0).range(vk::WHOLE_SIZE),
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..5)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

        let pc = MatVecPushConstants {
            ncols: k, stride_a: k, stride_b: k, stride_d: m,
            batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
            fusion_flags: 0, base_work_group_y: 0,
            ne02: 1, ne12: 1, broadcast2: 1,
            broadcast3: weight_scale.to_bits(),
        };
        let pipeline = self.lmhead.pipeline;
        let layout = self.lmhead.layout;
        self.profile("lm_head", dev, cmd, |dev, cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, m, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemv(
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
        weight_scale: f32,
        label: &str,
    ) {
        let kernel = registry.get(shader);
        let one_per_row = matches!(
            shader,
            ShaderId::MulMatVecFp8 | ShaderId::MulMatVecF32 | ShaderId::MulMatVecF16,
        );
        let set = if one_per_row {
            self.alloc_or_get_set(
                dev, kernel.descriptor_set_layout,
                &[
                    (0, weights, 0, 0),
                    (1, input, 0, 0),
                    (2, output, 0, 0),
                ],
            )
        } else {
            self.alloc_or_get_set(
                dev, kernel.descriptor_set_layout,
                &[
                    (0, weights, 0, 0),
                    (1, input, 0, 0),
                    (2, output, 0, 0),
                    (3, self.fuse0.handle, 0, 0),
                    (4, self.fuse1.handle, 0, 0),
                ],
            )
        };
        // FP8 GEMV reads `weight_scale` from `broadcast3` (the last
        // u32 slot is `float weight_scale` in the GLSL push block).
        let pc = MatVecPushConstants {
            ncols: k, stride_a: k, stride_b: k, stride_d: m,
            batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
            fusion_flags: 0, base_work_group_y: 0,
            ne02: 1, ne12: 1, broadcast2: 1,
            broadcast3: weight_scale.to_bits(),
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
            let groups = if one_per_row {
                m
            } else {
                let n_rows = crate::backend::vulkan::pipeline_registry::MMV_NUM_ROWS;
                (m + n_rows - 1) / n_rows
            };
            dev.device.cmd_dispatch(cmd, groups, 1, 1);
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_rms_norm(
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
    pub(super) fn run_rope_neox(
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
        let theta_scale = self.rope_theta_scale;
        let freq_base = self.config.rope_freq_base;
        // Bind rope_pos_buf at slot 0 (the legacy single-position
        // path used by forward_token). prefill_batch uses
        // run_rope_neox_with_pos_offset to read its own slot.
        self.run_rope_neox_with_pos_offset(
            dev, registry, cmd, input, output, head_dim, head_dim,
            freq_base, theta_scale, n_rows,
            position, /* pos_buf_offset = */ 0, label,
        );
    }

    /// Variant of `run_rope_neox` that binds `rope_pos_buf` starting
    /// at `pos_buf_offset` bytes — required by `prefill_batch` to give
    /// each per-token RoPE dispatch its own pre-staged position slot.
    ///
    /// Sprint 43D-2 — `rotary_dim`, `freq_base`, `theta_scale` lifted to
    /// per-call so Gemma-4's per-layer p-RoPE (rotary_dim = 0.25×head_dim
    /// for full layers, full head_dim for sliding layers) and per-layer
    /// θ (10 000 sliding / 1 000 000 full) can be threaded through. For
    /// non-Gemma-4 callers (`rotary_dim == head_dim`, `freq_base ==
    /// cfg.rope_freq_base`, `theta_scale == self.rope_theta_scale`)
    /// the math is bit-identical to pre-43D-2.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_rope_neox_with_pos_offset(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
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
                (1, self.cur().rope_pos_buf.handle, pos_buf_offset, 4),
                (2, self.rope_ff_buf.handle, 0, 0),
                (3, output, 0, 0),
                (4, self.rope_idx_buf.handle, 0, 0),
            ],
        );
        let pc = RopePushConstants {
            rope_mode, // 0 = NORM, 2 = NEOX
            nrows: n_rows,
            n_dims: rotary_dim,
            freq_scale: 1.0,
            freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale,
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

    pub(super) fn run_scalar_attn(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        position: u32,
    ) {
        let cfg = self.config.clone();
        // Sprint 43D-2 — KV-share for Gemma-4: read K/V from the
        // publisher's slab when this layer subscribes; for non-shared
        // layers `kv_layer == layer` so the slab math is unchanged.
        let kv_layer = gemma4_kv_read_layer(&cfg, layer);
        // Sprint 43D-2 — sliding-window-attention lower bound. 0 for
        // every non-Gemma-4 / non-sliding layer.
        let kv_start = gemma4_kv_start(&cfg, layer, position);
        // Sprint 43D-1 — heterogeneous KV-cache layout. The shader's
        // `head_dim` push-const drives both the per-head loop bound and
        // the per-position stride; with the cumulative offset table in
        // KvCache, layer N's slab now starts at a slab whose stride
        // matches per_layer_head_dim[N], so the push-const can carry the
        // correct per-layer value. Default path (uniform layout)
        // resolves to cfg.head_dim and self.attn_scale unchanged.
        let head_dim_layer = self.kv_cache.head_dim_for(layer);
        let attn_scale_layer = if cfg.gemma4.is_some() {
            // Sprint 43D-4 — HF Gemma4TextAttention sets `self.scaling
            // = 1.0` (line 972 of modular_gemma4.py). The standard
            // `1/√head_dim` is absorbed into the post-Q-norm magnitude
            // (Q-norm normalizes Q to unit-rms-per-head, so Q·Kᵀ stays
            // bounded without an explicit scale). VF pre-43D-4 was
            // applying `1/√head_dim` on top, double-scaling the scores
            // and damaging attention math — visible as cosine cliff at
            // Layer 2 in the layer-by-layer divergence analysis.
            1.0_f32
        } else if head_dim_layer != cfg.head_dim {
            1.0_f32 / (head_dim_layer as f32).sqrt()
        } else {
            self.attn_scale
        };
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
            self.run_flash_attn_split_reduce(
                dev, registry, cmd, layer, position, n_tiles, kv_layer, kv_start,
            );
            return;
        }
        // Phase 4B: forward path now dispatches the online-softmax
        // flash_attn shader instead of the Phase-3A tiled scalar_attn.
        // Sprint 9d.3 — FP16 KV-aware variant when the cache is
        // FP16-allocated. Sprint 18A — FP8 KV variant.
        let kernel = registry.get(if self.kv_cache.is_fp8() {
            ShaderId::FlashAttnFp8Kv
        } else if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnFp16Kv
        } else {
            ShaderId::FlashAttn
        });
        let layer_off = self.kv_cache.layer_offset_bytes(kv_layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes(kv_layer);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, self.cur().q_buf.handle, 0, 0),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, self.cur().attn_out.handle, 0, 0),
            ],
        );
        let pc = ScalarAttnPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: head_dim_layer,
            seq_len: position + 1,
            max_seq: self.kv_cache.config.max_seq_len,
            scale: attn_scale_layer,
            kv_start,
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_rope_batch(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        input: vk::Buffer,
        output: vk::Buffer,
        head_dim: u32,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
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
                (1, self.cur().rope_pos_buf.handle, 0, pos_size),
                (2, self.rope_ff_buf.handle, 0, 0),
                (3, output, 0, 0),
                (4, self.rope_idx_buf.handle, 0, 0),
            ],
        );
        let nrows = heads_per_token * m;
        let pc = RopePushConstants {
            rope_mode,
            nrows,
            n_dims: rotary_dim,
            freq_scale: 1.0,
            freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale,
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
    pub(super) fn run_flash_attn_batch(
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
        kv_layer: u32,
        kv_start: u32,
    ) {
        let cfg = self.config.clone();
        // v0.2 Sprint 9d.2 — pick the FP16-KV-aware variant when the
        // cache is allocated as FP16; otherwise the original FP32 SPV.
        // Sprint 18A — FP8 KV variant.
        let kernel = registry.get(if self.kv_cache.is_fp8() {
            ShaderId::FlashAttnBatchFp8Kv
        } else if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnBatchFp16Kv
        } else {
            ShaderId::FlashAttnBatch
        });
        // Sprint 46E — KV-share for Gemma-4 batch prefill: read K/V
        // from the publisher's slab when this layer subscribes. For
        // non-shared layers `kv_layer == layer` so the slab math is
        // unchanged. Mirrors the decode-path `run_flash_attn_split_reduce`
        // pattern (which already uses kv_layer for the binding offset).
        let layer_off = self.kv_cache.layer_offset_bytes(kv_layer);
        let layer_size = self.kv_cache.layer_size_bytes(kv_layer);
        // Sprint 43D-1 — per-layer head_dim for Gemma-4 heterogeneous.
        let head_dim_layer = self.kv_cache.head_dim_for(layer);
        // Sprint 43D-4 — see decode-path comment: Gemma-4 attention
        // uses scaling=1.0 (Q-norm absorbs the 1/√d).
        let attn_scale_layer = if cfg.gemma4.is_some() {
            1.0_f32
        } else if head_dim_layer != cfg.head_dim {
            1.0_f32 / (head_dim_layer as f32).sqrt()
        } else {
            self.attn_scale
        };
        let q_bytes_total = (m as u64) * (cfg.n_heads as u64) * (head_dim_layer as u64) * 4;
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
            head_dim: head_dim_layer,
            m,
            n_kv,
            q_start,
            scale: attn_scale_layer,
            kv_start,
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
    pub(super) fn run_flash_attn_tiled(
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
        kv_layer: u32,
        kv_start: u32,
    ) {
        // v0.2 Sprint 10C — coopmat shader takes priority when
        // VULKANFORGE_COOPMAT_ATTN=1 is set. Forces Br=16 (the only
        // shape the coopmat SPV ships); Bc=16 is implicit in the
        // coopmat SPV's own #defines.
        // Sprint 18A — FP8 KV variants for both coopmat and tiled
        // (Br=16/Bc=32) paths.
        let (shader_id, br) = if self.coopmat_attn_enabled {
            if self.kv_cache.is_fp8() {
                (ShaderId::FlashAttnCoopmatFp8Kv, 16u32)
            } else if self.kv_cache.is_fp16() {
                (ShaderId::FlashAttnCoopmatFp16Kv, 16u32)
            } else {
                (ShaderId::FlashAttnCoopmat, 16u32)
            }
        } else if self.kv_cache.is_fp8() {
            match (self.fa_tiled_br, self.fa_tiled_bc) {
                (16, 32) => (ShaderId::FlashAttnTiledBr16Bc32Fp8Kv, 16u32),
                _ => panic!(
                    "VULKANFORGE_KV_FP8=1 requires the default Br=16/Bc=32 \
                     tiled flash-attn variant; got Br={}/Bc={}.",
                    self.fa_tiled_br, self.fa_tiled_bc,
                ),
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
        // Sprint 46E — KV-share: bind publisher's slab via kv_layer
        // (mirrors the decode split-reduce path).
        let layer_off = self.kv_cache.layer_offset_bytes(kv_layer);
        let layer_size = self.kv_cache.layer_size_bytes(kv_layer);
        // Sprint 43D-1 — per-layer head_dim for Gemma-4 heterogeneous.
        let head_dim_layer = self.kv_cache.head_dim_for(layer);
        // Sprint 43D-4 — see decode-path comment: Gemma-4 attention
        // uses scaling=1.0 (Q-norm absorbs the 1/√d).
        let attn_scale_layer = if cfg.gemma4.is_some() {
            1.0_f32
        } else if head_dim_layer != cfg.head_dim {
            1.0_f32 / (head_dim_layer as f32).sqrt()
        } else {
            self.attn_scale
        };
        let q_bytes_total = (m as u64) * (cfg.n_heads as u64) * (head_dim_layer as u64) * 4;
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
            head_dim: head_dim_layer,
            m,
            n_kv,
            q_start,
            scale: attn_scale_layer,
            kv_start,
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_flash_attn_split_reduce(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        layer: u32,
        position: u32,
        n_tiles: u32,
        kv_layer: u32,
        kv_start: u32,
    ) {
        let cfg = self.config.clone();
        let layer_off = self.kv_cache.layer_offset_bytes(kv_layer);
        // v0.2 Sprint 9d.1 — KvCache::layer_size_bytes scales by
        // the configured KV element size (FP32 = 4 B by default).
        let layer_size = self.kv_cache.layer_size_bytes(kv_layer);
        // Sprint 43D-1 — per-layer head_dim for Gemma-4 heterogeneous.
        let head_dim_layer = self.kv_cache.head_dim_for(layer);
        // Sprint 43D-4 — see decode-path comment: Gemma-4 attention
        // uses scaling=1.0 (Q-norm absorbs the 1/√d).
        let attn_scale_layer = if cfg.gemma4.is_some() {
            1.0_f32
        } else if head_dim_layer != cfg.head_dim {
            1.0_f32 / (head_dim_layer as f32).sqrt()
        } else {
            self.attn_scale
        };

        // ---- Split-K worker ----
        // Sprint 9d.3 — FP16 KV-aware variant of the split-K worker.
        // The reducer (FlashAttnReduce) doesn't read KV (only partials),
        // so it stays on the FP32 SPV.
        // Sprint 18A — FP8 KV variant.
        let split_kernel = registry.get(if self.kv_cache.is_fp8() {
            ShaderId::FlashAttnSplitFp8Kv
        } else if self.kv_cache.is_fp16() {
            ShaderId::FlashAttnSplitFp16Kv
        } else {
            ShaderId::FlashAttnSplit
        });
        let split_set = self.alloc_or_get_set(
            dev, split_kernel.descriptor_set_layout,
            &[
                (0, self.cur().q_buf.handle, 0, 0),
                (1, self.kv_cache.k_buffer.handle, layer_off, layer_size),
                (2, self.kv_cache.v_buffer.handle, layer_off, layer_size),
                (3, self.cur().fa_scratch_out.handle, 0, 0),
                (4, self.cur().fa_scratch_max.handle, 0, 0),
                (5, self.cur().fa_scratch_sum.handle, 0, 0),
            ],
        );
        let split_pc = FlashAttnSplitPushConstants {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: head_dim_layer,
            seq_len: position + 1,
            max_seq: self.kv_cache.config.max_seq_len,
            scale: attn_scale_layer,
            n_tiles,
            kv_start,
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
                (0, self.cur().fa_scratch_out.handle, 0, 0),
                (1, self.cur().fa_scratch_max.handle, 0, 0),
                (2, self.cur().fa_scratch_sum.handle, 0, 0),
                (3, self.cur().attn_out.handle, 0, 0),
            ],
        );
        let red_pc = FlashAttnReducePushConstants {
            n_heads: cfg.n_heads,
            head_dim: head_dim_layer,
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
    /// Sprint 24B — Broadcast-aware bias-add. Computes `d = a + b`
    /// where `a` is `[rows × dim]` and `b` is `[dim]`. Used for the
    /// Q/K/V projection biases on Qwen2-style architectures. Decode
    /// passes `rows = 1` (single-token), prefill passes `rows = seq_len`.
    /// Llama-style models without biases skip this dispatch entirely.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_bias_add(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        a: vk::Buffer,
        b: vk::Buffer,
        d: vk::Buffer,
        dim: u32,
        rows: u32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::Add);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, a, 0, 0), (1, b, 0, 0), (2, d, 0, 0)],
        );
        let total = dim * rows;
        // Broadcast pattern: a is [dim × rows] (4D shape: [dim, rows, 1, 1]);
        // b is [dim] (4D shape: [dim, 1, 1, 1]). The shader's `fastmod`
        // path on `src1_idx` clamps i01 to 0 when ne11 == 1, giving
        // bias[i00] for every row.
        let pc = GenericBinaryPushConstants {
            ne: total,
            ne00: dim, ne01: rows, ne02: 1, ne03: 1,
            nb00: 1, nb01: dim, nb02: total, nb03: total,
            ne10: dim, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: dim, nb12: dim, nb13: dim,
            ne20: dim, ne21: rows, ne22: 1, ne23: 1,
            nb20: 1, nb21: dim, nb22: total, nb23: total,
            misalign_offsets: 0,
            param1: 0.0, param2: 0.0, param3: 0,
        };
        let dispatch_y = (total + 511) / 512;
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

    pub(super) fn run_binary(
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

    /// Sprint 43D-4 — broadcast-Mul where `b` is a single-scalar buffer.
    /// Computes `d[i] = a[i] * b[0]`. Implemented by routing through the
    /// existing `Mul` shader with `ne1*=1, nb1*=0` so the broadcast
    /// `src1_idx(...)` resolves to 0 for every output index. Used for
    /// Gemma-4's per-layer `layer_scalar` final multiply: each decoder
    /// layer's output gets `output *= layer_scalar[layer]` with a
    /// learned 0.018..0.871 scalar (HF
    /// `Gemma4TextDecoderLayer.forward` end of layer).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_mul_scalar_b(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        a: vk::Buffer,
        b_scalar: vk::Buffer,
        d: vk::Buffer,
        n: u32,
        label: &str,
    ) {
        let kernel = registry.get(ShaderId::Mul);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, a, 0, 0), (1, b_scalar, 0, 0), (2, d, 0, 0)],
        );
        let pc = GenericBinaryPushConstants {
            ne: n,
            ne00: n, ne01: 1, ne02: 1, ne03: 1,
            nb00: 1, nb01: n, nb02: n, nb03: n,
            // Broadcast: b is a single scalar. ne1*=1, nb1*=0 makes
            // src1_idx(...) = fastmod(i, 1)*0 + ... = 0 for every i.
            ne10: 1, ne11: 1, ne12: 1, ne13: 1,
            nb10: 0, nb11: 0, nb12: 0, nb13: 0,
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
    pub(super) fn run_multi_add_rms(
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
    pub(super) fn run_kv_copy_fp16(
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

    /// Sprint 18A — FP32 → FP8 E4M3 KV-cache write. Sibling of
    /// `run_kv_copy_fp16`; same push-constant struct but the shader
    /// packs **4** FP8 values per uint instead of 2 FP16 values, so
    /// dispatch_x = ceil(n_elements / 4 / 256) = ceil(n / 1024).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_kv_store_fp8(
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
            "kv_store_fp8: dst_byte_offset must be uint-aligned (got {dst_byte_offset})"
        );
        let dst_uint_offset = (dst_byte_offset / 4) as u32;
        let kernel = registry.get(ShaderId::KvCopyFp8);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, src, 0, 0), (1, dst, 0, 0)],
        );
        let pc = KvCopyFp16PushConstants {
            n_elements,
            dst_uint_offset,
            src_float_offset: 0,
        };
        // 256 threads/WG, 4 elements/thread → 1024 elements/WG.
        let dispatch_x = (n_elements + 1023) / 1024;
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
    pub(super) fn run_rms_norm_mul_rope(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        qk: vk::Buffer,
        weight: vk::Buffer,
        qk_out: vk::Buffer,
        head_dim: u32,
        rotary_dim: u32,
        freq_base: f32,
        theta_scale: f32,
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
                (3, self.cur().rope_pos_buf.handle, 0, pos_size),
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
                n_dims: rotary_dim,
                freq_scale: 1.0,
                freq_base,
                ext_factor: 0.0,
                attn_factor: 1.0,
                corr_dims: [0.0, 0.0],
                theta_scale,
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
    pub(super) fn run_swiglu(
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

    /// Sprint 43D-2 — GELU(pytorch_tanh)-GLU dispatch (Gemma-4 FFN).
    /// Same shape / bindings / push-block as SwiGLU; the only difference
    /// is the activation on `gate` (pytorch-tanh GELU vs SiLU).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gelu_pytorch_tanh_glu(
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
        let kernel = registry.get(ShaderId::GeluPytorchTanhGlu);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[(0, gate, 0, 0), (1, up, 0, 0), (2, out, 0, 0)],
        );
        let pc = SwigluPushConstants { n };
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
    pub(super) fn run_quantize_q8_1(
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
    /// Sprint 20-Wire — dispatch the native FP8 prefill GEMM
    /// (`MulCoopmatFp8Naive`). Same output layout as `run_gemm`
    /// (`[N × M]` row-major, `stride_d = M`) so the rest of the
    /// prefill pipeline plugs in unchanged. Activation buffer must
    /// be FP32 (the kernel does FP32→BF16 narrow in LDS, mirroring
    /// the Q4_K naive coopmat path); SafeTensors prefill always
    /// reaches this with `gemm_input_* = batch_norm.handle` because
    /// `force_mmq` only fires for Q4_0 — FP8 routes through `MulMm`.
    /// Per-tensor `weight_scale` rides in the last push-constant
    /// slot (`f32::to_bits()`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemm_fp8_naive(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd: vk::CommandBuffer,
        weight_buf: vk::Buffer,
        scale_buf: vk::Buffer,
        activations_fp32: vk::Buffer,
        output: vk::Buffer,
        m: u32,
        n: u32,
        k: u32,
        label: &'static str,
    ) {
        // Sprint 21B — multi-WG kernel (BM=64, 4 subgroups share
        // the activation tile in LDS). Wins on big-prefill shapes
        // (~+6% at pp=406) but the larger workgroup hurts at very
        // small N where the dispatch is already small enough that
        // 4× fewer WGs starves the GPU. Empirical crossover lands
        // around N=64; gating on `m >= 64 && n >= 64` keeps the
        // single-tile kernel for short prompts and decode-style
        // batches that ever route through prefill.
        // Sprint 32 Phase 1 — BN=32 variant (DEFAULT, +113% on 14B pp=512).
        // Sprint 33 — BN=64 variant available behind `VF_FP8_GEMM_BN=64`
        // opt-in only. Hypothesis was that another N-tile doubling would
        // continue the +113% trend, but a clean A/B on Qwen2.5-14B FP8
        // pp=512 measured BN=64 = 308 t/s mean vs BN=32 = 319 t/s mean
        // (-3% for BN=64). The larger tile drops occupancy from ~5 WGs/CU
        // to ~3 WGs/CU and that loses more wall time than the activation
        // reuse saves at this shape. Honest negative result; the BN=64
        // shader stays in the tree as opt-in for future tuning (e.g. on
        // larger models where the FFN GEMM is bigger).
        //
        // Override via `VF_FP8_GEMM_BN={16,32,64}`. Legacy
        // `VF_FP8_GEMM_BN32=0` still respected as opt-out to BN=16.
        // Sprint 38 Part 1 — `VF_FP8_NATIVE_WMMA=1` opt-in selects the
        // FP8×FP8 cooperative-matrix variant of BN=32 (Mesa 26.1+).
        let bn_override = std::env::var("VF_FP8_GEMM_BN")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let bn32_disabled = std::env::var("VF_FP8_GEMM_BN32")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let bn_target = bn_override.unwrap_or(if bn32_disabled { 16 } else { 32 });
        let native_fp8_wmma = std::env::var("VF_FP8_NATIVE_WMMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let use_bn64 = bn_target >= 64 && m >= 64 && n >= 64;
        let use_bn32 = !use_bn64 && bn_target >= 32 && m >= 64 && n >= 64;
        let use_native = native_fp8_wmma && use_bn32;
        let multi_wg = m >= 64 && n >= 64;
        let shader = if use_native {
            ShaderId::MulCoopmatFp8NativeBn32
        } else if use_bn64 {
            ShaderId::MulCoopmatFp8Bn64
        } else if use_bn32 {
            ShaderId::MulCoopmatFp8Bn32
        } else if multi_wg {
            ShaderId::MulCoopmatFp8MultiWg
        } else {
            ShaderId::MulCoopmatFp8Naive
        };
        let kernel = registry.get(shader);
        let set = self.alloc_or_get_set(
            dev, kernel.descriptor_set_layout,
            &[
                (0, weight_buf, 0, 0),
                (1, activations_fp32, 0, 0),
                (2, output, 0, 0),
                (3, scale_buf, 0, 0),
            ],
        );
        let pc = Fp8GemmPushConstants {
            m, n, k,
            stride_a: k,
            stride_b: k,
            stride_c: m,
            weight_scale_bits: 0,
        };
        let bm = if use_bn64 || use_bn32 || multi_wg { 64u32 } else { 16u32 };
        let bn = if use_bn64 { 64u32 } else if use_bn32 { 32u32 } else { 16u32 };
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

    /// Each output row holds one token's `M`-dim projection.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_gemm(
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
            ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL
            | ShaderId::MulMmqQ3KL | ShaderId::MulMmqQ5KL
            | ShaderId::MulMmqQ4_0L
            | ShaderId::MulMmQ4KCoopmat | ShaderId::MulMmQ6KCoopmat
            | ShaderId::MulMmQ4KAlignedCoopmat | ShaderId::MulMmQ6KAlignedCoopmat
            | ShaderId::MulMmQ4KAlignedCoopmatF16Acc
            | ShaderId::MulMmQ6KAlignedCoopmatF16Acc
            // Sprint 19A — Q3_K + Q5_K coopmat L-tile share BM=BN=128.
            | ShaderId::MulMmQ3KCoopmat | ShaderId::MulMmQ5KCoopmat
            | ShaderId::MulMmQ3KAlignedCoopmat | ShaderId::MulMmQ5KAlignedCoopmat
            | ShaderId::MulMmQ3KAlignedCoopmatF16Acc
            | ShaderId::MulMmQ5KAlignedCoopmatF16Acc => (128, 128),
            // Sprint 12M — M-tile coopmat: BM=BN=64, same shape as the
            // S-tile mul_mmq variants. Used when seq_len ≤ 64.
            ShaderId::MulMmQ4KCoopmatM | ShaderId::MulMmQ6KCoopmatM
            | ShaderId::MulMmQ4KAlignedCoopmatM | ShaderId::MulMmQ6KAlignedCoopmatM
            | ShaderId::MulMmQ3KCoopmatM | ShaderId::MulMmQ5KCoopmatM
            | ShaderId::MulMmQ3KAlignedCoopmatM | ShaderId::MulMmQ5KAlignedCoopmatM => (64, 64),
            // Sprint 13A — S-tile coopmat: BM=BN=32. Used when seq_len ≤ 32.
            ShaderId::MulMmQ4KCoopmatS | ShaderId::MulMmQ6KCoopmatS
            | ShaderId::MulMmQ4KAlignedCoopmatS | ShaderId::MulMmQ6KAlignedCoopmatS
            | ShaderId::MulMmQ3KCoopmatS | ShaderId::MulMmQ5KCoopmatS
            | ShaderId::MulMmQ3KAlignedCoopmatS | ShaderId::MulMmQ5KAlignedCoopmatS => (32, 32),
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
    pub(super) fn pad_to_tile(n: u32, tile: u32) -> u32 {
        (n + tile - 1) / tile * tile
    }

    /// Sprint 3C — fill the `[n, n_padded)` rows of an [N, K]
    /// row-major activation buffer with zeros. The coopmat dispatch
    /// sees N = n_padded; the extra rows multiply with weights to
    /// produce zeros in the padded output rows, which downstream code
    /// (RoPE / attention / sampler) ignores because it walks only
    /// `seq_len = n` rows.
    pub(super) fn zero_activation_tail(
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
    pub(super) fn coopmat_naive_padded_shader(&self) -> ShaderId {
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
    pub(super) fn run_gemm_coopmat_q4k(
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
    pub(super) fn copy_batch_row(
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
}
