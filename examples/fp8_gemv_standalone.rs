//! Sprint 24-Harness — standalone per-channel FP8 GEMV reproducer.
//!
//! Loads `blk.0.attn_q.weight` from a HuggingFace SafeTensors model,
//! creates a *fresh* compute pipeline (no PipelineRegistry, no Forward,
//! no descriptor cache), dispatches one FP8 GEMV with a per-output-row
//! scale buffer, and compares the GPU output against a CPU reference.
//!
//! - PASS  → shader + 4-binding pipeline are correct in isolation, and
//!           the per-channel runtime breakage lives in Forward / cache /
//!           pipeline_registry / dispatch ordering.
//! - FAIL  → shader + pipeline are themselves broken.

use std::os::raw::c_void;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::fp8_ext::fp8_e4m3_to_f32;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
use vulkanforge::safetensors::SafeTensorsFile;

/// Sprint 24-Harness — per-channel SPV variant compiled by build.rs
/// from `vk_shaders/mul_mat_vec_fp8_perchannel.comp`. Distinct from
/// the production `MUL_MAT_VEC_FP8` (per-tensor) so the harness's
/// 4-binding scheme stays self-contained.
const MUL_MAT_VEC_FP8_PERCHANNEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_fp8_perchannel.spv"));

fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

fn bf16_bytes_to_f32_vec(raw: &[u8]) -> Vec<f32> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        out.push(bf16_to_f32(bf));
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: fp8_gemv_standalone <safetensors-dir>");
            std::process::exit(1);
        });

    eprintln!("[1] Vulkan device init");
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;

    eprintln!("[2] SafeTensors load");
    let st = SafeTensorsFile::open(std::path::Path::new(&model_dir))?;

    let weight_name = "model.layers.0.self_attn.q_proj.weight";
    let scale_name = "model.layers.0.self_attn.q_proj.weight_scale";

    let weight_info = st.tensors.get(weight_name)
        .ok_or_else(|| format!("missing tensor: {weight_name}"))?;
    let scale_info = st.tensors.get(scale_name)
        .ok_or_else(|| format!("missing tensor: {scale_name}"))?;

    let weight_bytes = st.tensor_bytes(weight_info);
    let m = weight_info.shape[0]; // out_dim
    let k = weight_info.shape[1]; // in_dim
    eprintln!("    weight: [{m} × {k}] FP8 ({} bytes)", weight_bytes.len());

    let scale_raw = st.tensor_bytes(scale_info);
    let scale_f32 = bf16_bytes_to_f32_vec(scale_raw);
    eprintln!("    scale shape: {:?}, first={:.6}, len={}",
        scale_info.shape, scale_f32[0], scale_f32.len());

    // Per-tensor → broadcast to [m]; per-channel → as-is.
    let scale_vec: Vec<f32> = if scale_f32.len() == 1 {
        eprintln!("    → broadcasting scalar to [{}] vector", m);
        vec![scale_f32[0]; m]
    } else if scale_f32.len() == m {
        scale_f32.clone()
    } else {
        return Err(format!(
            "scale length {} doesn't match scalar(1) or out_dim({})",
            scale_f32.len(), m
        ).into());
    };

    eprintln!("[3] Generate input vector ({} floats)", k);
    let input: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.001) - 0.5).sin()).collect();
    eprintln!("    input first5: {:.6?}", &input[..5]);

    eprintln!("[4] Upload buffers");
    let mut weight_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        weight_bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "harness_weight",
    )?;
    weight_buf.write_bytes(weight_bytes)?;

    let mut scale_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (m * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "harness_scale",
    )?;
    scale_buf.write_bytes(bytemuck::cast_slice(&scale_vec))?;

    let mut input_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (k * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "harness_input",
    )?;
    input_buf.write_bytes(bytemuck::cast_slice(&input))?;

    let output_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (m * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuToCpu,
        "harness_output",
    )?;

    eprintln!("[5] Build fresh compute pipeline");

    // Shader module — per-channel variant SPV (binding 3 = ScaleBuf).
    let words: Vec<u32> = MUL_MAT_VEC_FP8_PERCHANNEL
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shader_module = unsafe {
        dev.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&words),
            None,
        )?
    };

    // Descriptor-set layout — 4 STORAGE_BUFFER bindings.
    let dsl_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl = unsafe {
        dev.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings),
            None,
        )?
    };

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<MatVecPushConstants>() as u32);
    let layouts = [dsl];
    let push_ranges = [push_range];
    let pipeline_layout = unsafe {
        dev.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )?
    };

    // Spec-constant 0 = BLOCK_SIZE, matches the production registry.
    let block_size: u32 = 64;
    let spec_entries = [vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: 4,
    }];
    let spec_data = bytemuck::bytes_of(&block_size);
    let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_entries)
        .data(spec_data);

    // requiredSubgroupSize=64 — matches Sprint 14A's pin so ACO emits Wave64.
    let mut subgroup_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
        .required_subgroup_size(64);

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main")
        .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
        .specialization_info(&spec_info)
        .push_next(&mut subgroup_info);

    let pipeline = unsafe {
        dev.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        ).map_err(|(_, e)| e)?[0]
    };

    eprintln!("[6] Allocate descriptor set + write 4 bindings");
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 4,
    }];
    let pool = unsafe {
        dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    };
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info)?[0] };

    let infos = [
        vk::DescriptorBufferInfo::default().buffer(weight_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(input_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(output_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(scale_buf.handle).offset(0).range(vk::WHOLE_SIZE),
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

    eprintln!("[7] Dispatch");
    // MatVecPushConstants laid out exactly as the production GEMV uses it.
    // The FP8 shader currently ignores `broadcast3` — scale[row] comes
    // from binding 3 — but we still pass a sensible value (1) so the
    // memory pattern is identical to runtime.
    let pc = MatVecPushConstants {
        ncols: k as u32,
        stride_a: k as u32,
        stride_b: k as u32,
        stride_d: m as u32,
        batch_stride_a: (k * m) as u32,
        batch_stride_b: k as u32,
        batch_stride_d: m as u32,
        fusion_flags: 0,
        base_work_group_y: 0,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };

    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;
    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout,
            0, &[set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE,
            0, bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, m as u32, 1, 1);
    })?;

    eprintln!("[8] Readback + compare");
    let raw = output_buf.read_bytes()?;
    let gpu_output: &[f32] = bytemuck::cast_slice(raw);

    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f64;
        for j in 0..k {
            let fp8_byte = weight_bytes[row * k + j];
            let w = fp8_e4m3_to_f32(fp8_byte) as f64;
            sum += w * (input[j] as f64);
        }
        cpu_output[row] = (sum as f32) * scale_vec[row];
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_zeros_gpu = 0usize;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_abs { max_abs = diff; }
        let denom = cpu_output[i].abs().max(1e-6);
        let rel = diff / denom;
        if rel > max_rel { max_rel = rel; }
        if gpu_output[i] == 0.0 { n_zeros_gpu += 1; }
    }

    let gpu_max = gpu_output.iter().cloned().fold(0.0f32, f32::max);
    let cpu_max = cpu_output.iter().cloned().fold(0.0f32, f32::max);

    eprintln!();
    eprintln!("=== RESULTS ===");
    eprintln!("GPU first5:  {:.6?}", &gpu_output[..5]);
    eprintln!("CPU first5:  {:.6?}", &cpu_output[..5]);
    eprintln!("max_abs:     {:.6}", max_abs);
    eprintln!("max_rel:     {:.6}", max_rel);
    eprintln!("GPU zeros:   {} / {} ({:.1}%)",
        n_zeros_gpu, m, n_zeros_gpu as f64 / m as f64 * 100.0);
    eprintln!("GPU max:     {:.6}", gpu_max);
    eprintln!("CPU max:     {:.6}", cpu_max);
    let _ = std::ptr::null::<c_void>(); // keep c_void import alive

    if n_zeros_gpu == m {
        eprintln!("\n🚨 GPU output is ALL ZEROS!");
        eprintln!("   → Shader produced no output");
        eprintln!("   → Pipeline / Binding / Dispatch is broken in isolation");
    } else if max_rel < 0.01 {
        eprintln!("\n✅ PASS — max_rel {:.4}% < 1%", max_rel * 100.0);
        eprintln!("   → Shader + pipeline + 4-binding DSL are CORRECT in isolation");
        eprintln!("   → Bug is in runtime integration (Forward / cache / dispatch ordering)");
    } else {
        eprintln!("\n❌ FAIL — max_rel {:.4}% > 1%", max_rel * 100.0);
        eprintln!("   → Shader runs but produces wrong values");
        eprintln!("   → Investigate weight layout / scale indexing / push constants");
    }

    // Cleanup.
    unsafe {
        dev.device.destroy_descriptor_pool(pool, None);
        dev.device.destroy_pipeline(pipeline, None);
        dev.device.destroy_pipeline_layout(pipeline_layout, None);
        dev.device.destroy_descriptor_set_layout(dsl, None);
        dev.device.destroy_shader_module(shader_module, None);
    }
    output_buf.destroy(&dev.device, &mut allocator);
    input_buf.destroy(&dev.device, &mut allocator);
    scale_buf.destroy(&dev.device, &mut allocator);
    weight_buf.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);

    Ok(())
}
