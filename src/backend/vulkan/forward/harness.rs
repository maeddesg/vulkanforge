//! Sprint 44B-2 — DRY-collapse of the four "harness-style" pipeline init
//! blocks that used to live inline in `Forward::new_with_prefill`.
//!
//! ## What's a "harness pipeline"?
//!
//! Sprint 24-Inline / 25 / 29 / 35 / 36 each shipped a dedicated compute
//! pipeline that bypasses the shared `PipelineRegistry` (with its shared
//! `vk::PipelineCache`) so its ACO codegen never collides with the other
//! ~100 pipelines. The pattern: `PipelineCache::null` + dedicated DSL
//! + dedicated descriptor pool. Same idea fixed Sprint 24-Inline's
//! per-channel FP8 GEMV (which was 10× slower in the shared registry vs
//! standalone) and Sprint 29's lm_head (~30 ms vs ~3 ms standalone).
//!
//! Today four such pipelines exist: `fp8pc` (per-channel FP8 GEMV),
//! `fp8bw` (block-wise FP8 GEMV), `fp8bwgemm` (block-wise FP8 GEMM
//! BN=32), and `lmhead` (F16 GEMV for `lm_head`). Their init code was
//! near-copy-paste with three real differences:
//!
//! - **`n_bindings`**: 4 (every FP8 variant) or 5 (lm_head adds two
//!   fuse-dummy slots).
//! - **`push_size`**: each pipeline has its own push-constant struct.
//! - **`block_size_spec`**: every variant except `fp8bwgemm` ships a
//!   `BLOCK_SIZE=64` specialization constant.
//! - **`max_sets`**: 1024 for the FP8 variants (descriptor-set cache
//!   covers ~336 unique keys × 2 async slots); 524288 for `lmhead`
//!   (legacy from Sprint 25B's Sprint-25-pre cache pattern; the pool
//!   never fills, but historical reset semantics rely on the size).
//!
//! `HarnessPipeline::new` parameterises those four knobs and emits a
//! byte-identical pipeline to what the four inline blocks produced.
//! `destroy` is its symmetric teardown.
//!
//! Resources NOT bundled into `HarnessPipeline`:
//! - the per-pipeline `descriptor-set cache` HashMaps (`fp8pc_ds_cache`,
//!   `fp8bw_ds_cache`, `fp8bwgemm_ds_cache`) — they live on `Forward` so
//!   the dispatch wrappers can `clear()` them on prefill;
//! - the *native FP8 WMMA* pipeline pair (`fp8bwgemm_native_shader_module`
//!   / `fp8bwgemm_native_pipeline`) — those reuse `fp8bwgemm`'s DSL +
//!   pipeline layout + pool, so they're constructed inline and stay as
//!   loose fields on `Forward`.

use ash::vk;

pub(super) struct HarnessPipeline {
    pub(super) module: vk::ShaderModule,
    pub(super) dsl: vk::DescriptorSetLayout,
    pub(super) layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
    pub(super) pool: vk::DescriptorPool,
}

impl HarnessPipeline {
    /// Build a harness-style compute pipeline with `PipelineCache::null`
    /// + a dedicated descriptor pool. See module docs for parameter
    /// semantics.
    pub(super) fn new(
        device: &ash::Device,
        spv_bytes: &[u8],
        n_bindings: u32,
        push_size: u32,
        block_size_spec: Option<u32>,
        max_sets: u32,
    ) -> Result<Self, vk::Result> {
        let words: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )?
        };

        let dsl_bindings: Vec<_> = (0..n_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings),
                None,
            )?
        };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_size);
        let layouts_arr = [dsl];
        let push_ranges = [push_range];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts_arr)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };

        // requiredSubgroupSize=64 + REQUIRE_FULL_SUBGROUPS pin every
        // harness pipeline to a Wave64 dispatch on RDNA — the Sprint 14A
        // pin all four call sites used historically. Lifetime: kept
        // alive until `create_compute_pipelines` returns below.
        let mut subgroup_info =
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(64);

        // Two pipeline-build paths because only the spec-info one needs
        // SpecializationInfo on the borrow chain. Both branches end in
        // `create_compute_pipelines`; the spec_* locals only have to
        // live until that call returns inside the same arm.
        let pipeline = if let Some(block_size) = block_size_spec {
            let spec_entries = [vk::SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            }];
            let spec_data = bytemuck::bytes_of(&block_size);
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(spec_data);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .specialization_info(&spec_info)
                .push_next(&mut subgroup_info);
            unsafe {
                device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(layout)],
                        None,
                    )
                    .map_err(|(_, e)| e)?[0]
            }
        } else {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .push_next(&mut subgroup_info);
            unsafe {
                device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(layout)],
                        None,
                    )
                    .map_err(|(_, e)| e)?[0]
            }
        };

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: max_sets * n_bindings,
                    }])
                    .flags(vk::DescriptorPoolCreateFlags::empty()),
                None,
            )?
        };

        Ok(Self {
            module,
            dsl,
            layout,
            pipeline,
            pool,
        })
    }

    /// Symmetric teardown — order mirrors the inverse of `new`'s
    /// `create_*` calls so dependency ordering is preserved.
    pub(super) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}
