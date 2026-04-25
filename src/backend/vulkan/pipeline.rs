//! Compute-pipeline wrapper for a single SPIR-V kernel.
//!
//! Phase 1 / Step 1.2: builds the descriptor-set layout, pipeline
//! layout (with the shader's 52-byte push-constant block), shader
//! module, and compute pipeline for the Q4_K GEMV kernel.
//!
//! The interface is hard-wired to the `mul_mat_vec_q4_k` shader
//! family: 5 `STORAGE_BUFFER` bindings at descriptor set 0 (binding
//! 0 = weights A, 1 = input B, 2 = output D, 3 = fuse0, 4 = fuse1)
//! and three specialization constants (`BLOCK_SIZE`, `NUM_ROWS`,
//! `NUM_COLS`). Variants for other quants/dtypes will reuse this
//! shape — see results/phase1_step_1.0_shader_analysis.md §2.

use ash::vk;

/// Number of `STORAGE_BUFFER` bindings expected at descriptor set 0.
/// Confirmed from SPIR-V disassembly in step 1.1.
pub const NUM_BINDINGS: u32 = 5;

/// Push-constant block size as declared in `mul_mat_vec_base.glsl` —
/// 13 × `uint32_t`. Vulkan requires push-constant ranges to be a
/// multiple of 4 (already true) and ≤ `maxPushConstantsSize` (≥ 128
/// guaranteed by spec).
pub const PUSH_CONSTANT_BYTES: u32 = 52;

/// Matches `vk_mat_vec_push_constants` in llama.cpp's
/// `ggml-vulkan.cpp:992` and the GLSL `parameter` block in
/// `mul_mat_vec_base.glsl:16-41` (no MUL_MAT_ID branch). Kept
/// `#[repr(C)]` so the byte layout fed to `vkCmdPushConstants` is
/// the same the shader expects.
///
/// All strides are in *elements* (not bytes); the shader divides
/// `batch_stride_a` by `QUANT_K` itself when indexing Q4_K blocks.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatVecPushConstants {
    pub ncols: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_d: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub fusion_flags: u32,
    pub base_work_group_y: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub broadcast2: u32,
    pub broadcast3: u32,
}

const _: () =
    assert!(std::mem::size_of::<MatVecPushConstants>() == PUSH_CONSTANT_BYTES as usize);

/// Specialization-constant values supplied at pipeline-create time.
/// Keep this `#[repr(C)]` so the byte offsets fed to
/// `VkSpecializationMapEntry` line up with the field order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpecConstants {
    /// SpecId 0 — also drives `local_size_x`.
    pub block_size: u32,
    /// SpecId 1 — rows computed per workgroup.
    pub num_rows: u32,
    /// SpecId 2 — columns computed per workgroup.
    pub num_cols: u32,
}

impl SpecConstants {
    /// Defaults that match the GLSL declarations in
    /// `mul_mat_vec_base.glsl:89-91`. Hardware-agnostic; the
    /// RDNA4-tuned (64, 2, 1) variant comes in step 1.5.
    pub const SMOKE_DEFAULT: Self = Self {
        block_size: 32,
        num_rows: 1,
        num_cols: 1,
    };
}

pub struct ComputeKernel {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputeKernel {
    /// Build the kernel from a SPIR-V blob and a set of specialization
    /// constants. Caller owns the resulting handles and must call
    /// [`ComputeKernel::destroy`] before the [`ash::Device`] is torn
    /// down.
    pub fn new(
        device: &ash::Device,
        spv_words: &[u32],
        spec: SpecConstants,
    ) -> Result<Self, vk::Result> {
        // Descriptor set layout: 5 SSBO slots, all visible to the
        // compute stage. SPIR-V variables that alias the same binding
        // (binding 0 has 3 such aliases for block_q4_K /_packed16
        // /_packed32 — see step 1.1 §4) collapse to a single Vulkan
        // binding slot; descriptor_count stays at 1.
        let bindings: [vk::DescriptorSetLayoutBinding; NUM_BINDINGS as usize] =
            std::array::from_fn(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            });

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(e);
            }
        };

        let module_info = vk::ShaderModuleCreateInfo::default().code(spv_words);
        let shader_module = match unsafe { device.create_shader_module(&module_info, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return Err(e);
            }
        };

        // Specialization constants: one entry per field in
        // `SpecConstants`. constantID matches the GLSL SpecId
        // decorations (0, 1, 2).
        let spec_entries = [
            vk::SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            },
            vk::SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: 4,
            },
            vk::SpecializationMapEntry {
                constant_id: 2,
                offset: 8,
                size: 4,
            },
        ];
        let spec_bytes: [u8; 12] = bytemuck::cast([spec.block_size, spec.num_rows, spec.num_cols]);
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(&spec_bytes);

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main")
            .specialization_info(&spec_info);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline_result = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        };
        // Shader module is no longer needed after pipeline creation.
        unsafe { device.destroy_shader_module(shader_module, None) };

        let pipeline = match pipeline_result {
            Ok(mut v) => v.remove(0),
            Err((mut v, e)) => {
                for p in v.drain(..) {
                    if p != vk::Pipeline::null() {
                        unsafe { device.destroy_pipeline(p, None) };
                    }
                }
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return Err(e);
            }
        };

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    /// Tear the kernel down. Must be called before the owning
    /// `ash::Device` is dropped — there is no Drop impl on purpose,
    /// because `ComputeKernel` does not hold a reference to the
    /// device handle.
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
