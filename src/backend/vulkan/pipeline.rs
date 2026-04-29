//! Compute-pipeline wrapper for a single SPIR-V kernel.
//!
//! Phase 2A — `ComputeKernel::from_spv` reflects the SPIR-V at
//! pipeline-create time and builds a matching descriptor-set layout
//! and pipeline layout, no per-shader hard-coding. The
//! [`PipelineRegistry`](super::pipeline_registry::PipelineRegistry)
//! uses this to instantiate every shader in the inventory uniformly.
//!
//! The interface still consists of:
//!   - `pub pipeline: vk::Pipeline`
//!   - `pub pipeline_layout: vk::PipelineLayout`
//!   - `pub descriptor_set_layout: vk::DescriptorSetLayout`
//!
//! and an explicit `destroy(self, &Device)` — there is no `Drop` impl
//! because the kernel does not own a `Device` handle.

use ash::vk;

use super::spirv_reflect::{self, ReflectedShader};

pub const PUSH_CONSTANT_BYTES: u32 = 52;

/// llama.cpp's `vk_mat_vec_push_constants` layout (13 × u32 = 52 B).
/// Used by callers that dispatch the Q4_K / Q6_K GEMV pipelines.
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

// ---- Phase-2B: per-shader push-constant structs ----------------------
//
// One `#[repr(C)]` struct per GLSL push-constant block, with the field
// order taken straight from the shader source (see vk_shaders/*.glsl
// or generic_*_head.glsl). Compile-time `assert!` pins the size against
// the SPIR-V reflection result so a future shader-source edit can't
// silently desynchronise the layout.

/// `generic_binary_head.glsl` push block — used by `rms_norm`, `add`,
/// `mul`. 29 × 4 = 116 B. All `nb*` are *element* strides
/// (`ggml_nb / type_size`), not byte strides — matches llama.cpp's
/// `vk_op_binary_push_constants`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenericBinaryPushConstants {
    pub ne: u32,
    pub ne00: u32, pub ne01: u32, pub ne02: u32, pub ne03: u32,
    pub nb00: u32, pub nb01: u32, pub nb02: u32, pub nb03: u32,
    pub ne10: u32, pub ne11: u32, pub ne12: u32, pub ne13: u32,
    pub nb10: u32, pub nb11: u32, pub nb12: u32, pub nb13: u32,
    pub ne20: u32, pub ne21: u32, pub ne22: u32, pub ne23: u32,
    pub nb20: u32, pub nb21: u32, pub nb22: u32, pub nb23: u32,
    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,
    pub param3: i32,
}
const _: () = assert!(std::mem::size_of::<GenericBinaryPushConstants>() == 116);

/// `generic_head.glsl` push block — used by `silu`. 6 × 4 = 24 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenericHeadPushConstants {
    pub kx: u32,
    pub ky: u32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
    pub param4: f32,
}
const _: () = assert!(std::mem::size_of::<GenericHeadPushConstants>() == 24);

/// v0.2 Sprint 9a — fused SwiGLU push block. Single u32 element-count.
/// SPIR-V reflection rounds the layout up to 4 B already, so no
/// trailing padding is needed.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SwigluPushConstants {
    pub n: u32,
}
const _: () = assert!(std::mem::size_of::<SwigluPushConstants>() == 4);

/// v0.2 Sprint 9b — fused residual-add + RMSNorm-mul push block.
/// 2 × u32 + 1 × f32 = 12 B. Field order matches `multi_add_rms.comp`'s
/// `parameter` block exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiAddRmsPushConstants {
    pub ne00: u32,
    pub n_rows: u32,
    pub eps: f32,
}
const _: () = assert!(std::mem::size_of::<MultiAddRmsPushConstants>() == 12);

/// `generic_unary_head.glsl` push block — used by `copy`. 32 × 4 = 128 B.
/// The trailing six `ne0_*mp/L` and `ne1_*mp/L` fields are fastdiv
/// constants (see [`init_fastdiv_values`]) — set them, do not leave
/// them as 0 unless every relevant `ne*` is exactly 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenericUnaryPushConstants {
    pub ne: u32,
    pub ne00: u32, pub ne01: u32, pub ne02: u32, pub ne03: u32,
    pub nb00: u32, pub nb01: u32, pub nb02: u32, pub nb03: u32,
    pub ne10: u32, pub ne11: u32, pub ne12: u32, pub ne13: u32,
    pub nb10: u32, pub nb11: u32, pub nb12: u32, pub nb13: u32,
    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,
    pub ne0_012mp: u32, pub ne0_012l: u32,
    pub ne0_01mp: u32,  pub ne0_01l: u32,
    pub ne0_0mp: u32,   pub ne0_0l: u32,
    pub ne1_012mp: u32, pub ne1_012l: u32,
    pub ne1_01mp: u32,  pub ne1_01l: u32,
    pub ne1_0mp: u32,   pub ne1_0l: u32,
}
const _: () = assert!(std::mem::size_of::<GenericUnaryPushConstants>() == 128);

/// `soft_max.comp` push block. 17 × 4 = 68 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftMaxPushConstants {
    pub kx: u32,
    pub ky: u32,
    pub ne00: u32,
    pub ne01: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub ne13: u32,
    pub nb11: u32,
    pub nb12: u32,
    pub nb13: u32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: u32,
    pub nrows_x: u32,
    pub has_sinks: u32,
}
const _: () = assert!(std::mem::size_of::<SoftMaxPushConstants>() == 68);

/// `rope_params.glsl` struct — push-constant block for both `rope_norm`
/// and `rope_neox`. 23 fields, 108 B (matches reflection).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopePushConstants {
    pub rope_mode: u32,
    pub nrows: u32,
    pub n_dims: u32,
    pub freq_scale: f32,
    pub freq_base: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_dims: [f32; 2],
    pub theta_scale: f32,
    pub has_ff: u32,
    pub sections: [i32; 4],
    pub is_imrope: u32,
    pub is_back: u32,
    pub set_rows_stride: u32,
    pub ne00: u32,
    pub ne01: u32,
    pub ne02: u32,
    pub nb01: u32,
    pub nb02: u32,
    pub nb03: u32,
    pub nb11: u32,
    pub nb12: u32,
    pub nb13: u32,
}
const _: () = assert!(std::mem::size_of::<RopePushConstants>() == 108);

/// `scalar_attn.comp` push block. 5 × u32 + 1 × f32 = 24 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarAttnPushConstants {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub scale: f32,
}
const _: () = assert!(std::mem::size_of::<ScalarAttnPushConstants>() == 24);

/// `mul_mmq.comp` push block (non-MUL_MAT_ID variant). 16 × u32 = 64 B.
/// Field order matches the GLSL `parameter` block exactly — see
/// `vk_shaders/mul_mmq.comp` lines 41-68.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MmqPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_d: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub base_work_group_z: u32,
    pub num_batches: u32,
    pub k_split: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub broadcast2: u32,
    pub broadcast3: u32,
}
const _: () = assert!(std::mem::size_of::<MmqPushConstants>() == 64);

/// `mul_mm.comp` push block — Phase 6 v0.1.2 port from llama.cpp.
/// Same field order as `MmqPushConstants` (the two shaders share the
/// llama.cpp non-MUL_MAT_ID parameter struct), so we re-use it
/// directly. The two struct types stay distinct in Rust just for
/// call-site readability — `Forward::run_mul_mm` takes
/// `MulMmPushConstants` and `Forward::run_gemm` (mul_mmq) takes
/// `MmqPushConstants`.
pub type MulMmPushConstants = MmqPushConstants;

/// Sprint 3A — `mul_coopmat_q4k.comp` push block (24 B).
/// Field order matches the GLSL `PC` block exactly — 6 × u32. The
/// runtime fills in `stride_b = K` and `stride_c = M` because the
/// Sprint 3A variant is built with `-DFORWARD_LAYOUT` (B is [N, K],
/// C is [N, M] row-major).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoopmatPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_c: u32,
}
const _: () = assert!(std::mem::size_of::<CoopmatPushConstants>() == 24);

/// `quantize_q8_1.comp` push block. 2 × u32 = 8 B.
/// `ne` is the total f32 element count of the input. `num_blocks` is
/// the workgroup-loop upper bound (with `QBLOCK_X4` defined that's the
/// number of 128-element x4 blocks in the output).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Q8_1QuantizePushConstants {
    pub ne: u32,
    pub num_blocks: u32,
}
const _: () = assert!(std::mem::size_of::<Q8_1QuantizePushConstants>() == 8);

/// Phase-4C split-K worker (`flash_attn_split.comp`) push block.
/// 6 × u32 + 1 × f32 + 1 × u32 = 32 B. Drop-in additions over
/// `ScalarAttnPushConstants`: one extra `n_tiles` field tells the
/// shader the dispatch's Y dimension so it can compute its
/// `partial_idx = h * n_tiles + tile_idx` without round-tripping to
/// the host.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlashAttnSplitPushConstants {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub scale: f32,
    pub n_tiles: u32,
}
const _: () = assert!(std::mem::size_of::<FlashAttnSplitPushConstants>() == 28);

/// Phase-4C split-K reducer (`flash_attn_reduce.comp`) push block.
/// 3 × u32 = 12 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlashAttnReducePushConstants {
    pub n_heads: u32,
    pub head_dim: u32,
    pub n_tiles: u32,
}
const _: () = assert!(std::mem::size_of::<FlashAttnReducePushConstants>() == 12);

/// Phase-5B.1 batched-Q flash attention (`flash_attn_batch.comp`) push
/// block. 6 × u32 + 1 × f32 = 28 B. Layout matches the GLSL `Params`
/// block in `vk_shaders/flash_attn_batch.comp`.
///
/// `m` is the number of queries in this batch, `n_kv` the total
/// populated KV positions, `q_start` the KV position of the first
/// query (equals `n_kv - m` whenever the batch is the most recent
/// chunk added to the cache, but kept explicit so the shader does
/// not have to derive it). The causal mask uses
/// `causal_len = q_start + q_idx + 1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlashAttnBatchPushConstants {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub m: u32,
    pub n_kv: u32,
    pub q_start: u32,
    pub scale: f32,
}
const _: () = assert!(std::mem::size_of::<FlashAttnBatchPushConstants>() == 28);

/// llama.cpp's `init_fastdiv_values`. Used by [`GenericUnaryPushConstants`]
/// to populate the `ne*_*mp/L` fields — without these, `copy`'s SPIR-V
/// fastdiv path divides by a magic-of-zero and produces garbage indices.
pub fn init_fastdiv_values(d: u32) -> (u32, u32) {
    if d <= 1 {
        return (1, 0);
    }
    let mut l = 0u32;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;
    (mp, l)
}

/// GLSL-default specialization constants for the Q4_K / Q6_K GEMV
/// shaders. Provided as a value object for callers that dispatch
/// these kernels and want to record the values used at create time.
#[derive(Clone, Copy, Debug)]
pub struct SpecConstants {
    pub block_size: u32,
    pub num_rows: u32,
    pub num_cols: u32,
}

impl SpecConstants {
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
    pub reflection: ReflectedShader,
}

impl ComputeKernel {
    /// Build a kernel by reflecting the SPIR-V. No spec-constant
    /// override — the GLSL defaults (or runtime spec lookup at
    /// dispatch-time) decide the workgroup shape.
    pub fn from_spv(
        device: &ash::Device,
        spv_words: &[u32],
        cache: vk::PipelineCache,
    ) -> Result<Self, vk::Result> {
        Self::from_spv_with_spec(device, spv_words, cache, &[], &[])
    }

    /// Like [`Self::from_spv`] but lets the caller pin specific
    /// specialization constants. `spec_entries` lists `(constant_id,
    /// offset, size)` triples and `spec_data` is the contiguous byte
    /// buffer they reference.
    pub fn from_spv_with_spec(
        device: &ash::Device,
        spv_words: &[u32],
        cache: vk::PipelineCache,
        spec_entries: &[vk::SpecializationMapEntry],
        spec_data: &[u8],
    ) -> Result<Self, vk::Result> {
        let reflection = spirv_reflect::reflect(spv_words);

        // Descriptor-set layout — single set 0, one binding per
        // reflected resource. STORAGE_BUFFER for `buffer` blocks,
        // UNIFORM_BUFFER for `uniform` blocks (none expected here, but
        // the reflector handles both).
        let mut bindings = Vec::with_capacity(reflection.bindings.len());
        for b in &reflection.bindings {
            // We only support one descriptor set today (set 0). If a
            // shader declares set > 0 we still create a single layout
            // but caller / dispatch-side will need to bind matching
            // sets. Phase-2A inventory is set-0 only.
            assert_eq!(
                b.set, 0,
                "shader declares descriptor set {} (only set 0 supported in Phase 2A)",
                b.set
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.descriptor_type)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

        // Pipeline layout: one descriptor-set layout, one push-constant
        // range covering the reflected size. No range if the shader
        // has no push constants — Vulkan rejects zero-size ranges.
        let push_ranges_buf = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(reflection.push_constant_size)];
        let push_ranges: &[vk::PushConstantRange] = if reflection.push_constant_size > 0 {
            &push_ranges_buf
        } else {
            &[]
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(push_ranges);
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

        let spec_info = vk::SpecializationInfo::default()
            .map_entries(spec_entries)
            .data(spec_data);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        if !spec_entries.is_empty() {
            stage = stage.specialization_info(&spec_info);
        }

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline_result = unsafe {
            device.create_compute_pipelines(cache, std::slice::from_ref(&pipeline_info), None)
        };
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
            reflection,
        })
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
