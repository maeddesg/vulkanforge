//! Sprint 27 — F16 GEMV shape bisect.
//!
//! Builds a fresh `mul_mat_vec_f16` pipeline (PipelineCache::null,
//! BLOCK_SIZE=64, REQUIRE_FULL_SUBGROUPS, requiredSubgroupSize=64 —
//! same as the production registry) and dispatches GEMV at varying
//! (M, K) shapes against random buffers, measuring GPU wall-time
//! per-dispatch via VkQueryPool TIMESTAMP queries.
//!
//! Goal: localise why the 14B Qwen2.5-FP8 lm_head (M=152064, K=5120)
//! takes 29.7 ms vs the 8B Llama-3.1-FP8 lm_head (M=128256, K=4096)
//! at 1.5 ms — same shader, different shapes, 19.5× difference.
//!
//! Data values are random; correctness is not validated. Timing only.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
use vulkanforge::backend::vulkan::shaders::MUL_MAT_VEC_F16;

const MAX_M: u32 = 160000;
const MAX_K: u32 = 8192;
const RUNS_PER_SHAPE: u32 = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    // Big enough to cover all test shapes' weight matrix.
    let weight_bytes = (MAX_M as u64) * (MAX_K as u64) * 2;
    eprintln!("Allocating weight buffer: {:.2} GiB", weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    let weight_buf = GpuBuffer::new(
        &dev.device, &mut allocator, weight_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly, "shape_bench_weight",
    )?;
    // Sprint 27 — Fill the *entire* weight buffer with non-zero data.
    // Touching just one row leaves the rest as undefined VRAM, which
    // RDNA4 may serve from compressed/zero pages without going to
    // memory — making the bench artificially fast and missing the
    // real BW cost.
    {
        let chunk_elems: u32 = 8 * 1024 * 1024; // 16 MiB per chunk
        let mut tmp = GpuBuffer::new(
            &dev.device, &mut allocator, (chunk_elems * 2) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "seed_tmp",
        )?;
        let seed: Vec<u16> = (0..chunk_elems).map(|i| ((i & 0xff) as u16) | 0x3c00).collect();
        tmp.write_bytes(bytemuck::cast_slice(&seed))?;
        let total_u16 = (MAX_M as u64) * (MAX_K as u64);
        let mut filled: u64 = 0;
        while filled < total_u16 {
            let remaining = total_u16 - filled;
            let chunk = remaining.min(chunk_elems as u64);
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                let copy = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(filled * 2)
                    .size(chunk * 2);
                dev.device.cmd_copy_buffer(cmd, tmp.handle, weight_buf.handle, std::slice::from_ref(&copy));
            })?;
            filled += chunk;
        }
        eprintln!("Fully filled weight buffer ({:.2} GiB)", (total_u16 * 2) as f64 / (1024.0 * 1024.0 * 1024.0));
        tmp.destroy(&dev.device, &mut allocator);
    }

    let mut input_buf = GpuBuffer::new(
        &dev.device, &mut allocator, (MAX_K * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "shape_bench_input",
    )?;
    let input: Vec<f32> = (0..MAX_K).map(|i| ((i as f32) * 0.001).sin()).collect();
    input_buf.write_bytes(bytemuck::cast_slice(&input))?;

    // Sprint 27 — Match the runtime's logits_buf placement
    // (GpuToCpu = host-mapped, PCIe-attached) so we measure scattered
    // PCIe-write latency, not VRAM-local writes.
    let output_loc = if std::env::var("VF_OUTPUT_GPU_ONLY").is_ok() {
        MemoryLocation::GpuOnly
    } else {
        MemoryLocation::GpuToCpu
    };
    eprintln!("Output buffer location: {:?}", output_loc);
    let output_buf = GpuBuffer::new(
        &dev.device, &mut allocator, (MAX_M * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, output_loc, "shape_bench_output",
    )?;

    let fuse0 = GpuBuffer::new(
        &dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse0_dummy",
    )?;
    let fuse1 = GpuBuffer::new(
        &dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse1_dummy",
    )?;

    // -- Pipeline --
    let words: Vec<u32> = MUL_MAT_VEC_F16.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let shader_module = unsafe {
        dev.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };

    let dsl_bindings: Vec<_> = (0..5).map(|i| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
    }).collect();
    let dsl = unsafe {
        dev.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings), None,
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
                .set_layouts(&layouts).push_constant_ranges(&push_ranges),
            None,
        )?
    };

    let block_size: u32 = 64;
    let spec_entries = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
    let spec_data = bytemuck::bytes_of(&block_size);
    let spec_info = vk::SpecializationInfo::default().map_entries(&spec_entries).data(spec_data);
    let mut subgroup_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
        .required_subgroup_size(64);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module).name(c"main")
        .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
        .specialization_info(&spec_info)
        .push_next(&mut subgroup_info);

    let pipeline = unsafe {
        dev.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
            None,
        ).map_err(|(_, e)| e)?[0]
    };

    // -- Descriptor set --
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8,
    }];
    let pool = unsafe {
        dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&pool_sizes), None,
        )?
    };
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info)?[0] };

    let infos = [
        vk::DescriptorBufferInfo::default().buffer(weight_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(input_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(output_buf.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(fuse0.handle).offset(0).range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default().buffer(fuse1.handle).offset(0).range(vk::WHOLE_SIZE),
    ];
    let writes: Vec<_> = (0..5u32).map(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set).dst_binding(i).descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[i as usize]))
    }).collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    // -- TIMESTAMP query pool --
    let qpool = unsafe {
        dev.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP).query_count(2),
            None,
        )?
    };
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let ts_period_ns = props.limits.timestamp_period as f64;

    // -- Shape sweep --
    let shapes: &[(u32, u32, &str)] = &[
        (128256, 4096, "8B real        (128256 × 4096)"),
        (152064, 5120, "14B real       (152064 × 5120)"),
        (152064, 4096, "M=14B  K=8B    (152064 × 4096)"),
        (128256, 5120, "M=8B   K=14B   (128256 × 5120)"),
        (131072, 5120, "M=2^17 K=14B   (131072 × 5120)"),
        (152064, 2048, "M=14B  K=2048  (152064 × 2048)"),
        (152064, 4608, "M=14B  K=4608  (152064 × 4608)"),
        (152064, 6144, "M=14B  K=6144  (152064 × 6144)"),
        (152064, 8192, "M=14B  K=8192  (152064 × 8192)"),
        ( 32768, 5120, "M=2^15 K=14B   ( 32768 × 5120)"),
        ( 65536, 5120, "M=2^16 K=14B   ( 65536 × 5120)"),
        (128000, 5120, "M=128K K=14B   (128000 × 5120)"),
        (160000, 5120, "M=160K K=14B   (160000 × 5120)"),
    ];

    println!();
    println!("{:<32} {:>10} {:>10} {:>10} {:>10}", "shape", "min ms", "med ms", "max ms", "GB/s");
    println!("{}", "-".repeat(76));

    for &(m, k, label) in shapes {
        if m > MAX_M || k > MAX_K {
            println!("{:<32} (out of range)", label);
            continue;
        }
        let mut times_ms: Vec<f64> = Vec::with_capacity(RUNS_PER_SHAPE as usize);
        for _ in 0..RUNS_PER_SHAPE {
            let pc = MatVecPushConstants {
                ncols: k, stride_a: k, stride_b: k, stride_d: m,
                batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
                fusion_flags: 0, base_work_group_y: 0,
                ne02: 1, ne12: 1, broadcast2: 1,
                broadcast3: 1.0_f32.to_bits(),
            };
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                dev.device.cmd_reset_query_pool(cmd, qpool, 0, 2);
                dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                dev.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[],
                );
                dev.device.cmd_push_constants(
                    cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&pc),
                );
                dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, 0);
                dev.device.cmd_dispatch(cmd, m, 1, 1);
                dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, 1);
            })?;

            let mut data = [0u64; 2];
            unsafe {
                dev.device.get_query_pool_results(
                    qpool, 0, &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            let ticks = data[1].wrapping_sub(data[0]);
            let ns = (ticks as f64) * ts_period_ns;
            times_ms.push(ns / 1e6);
        }
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_ms = times_ms[0];
        let med_ms = times_ms[times_ms.len() / 2];
        let max_ms = times_ms[times_ms.len() - 1];
        let weight_gb = (m as f64) * (k as f64) * 2.0 / 1e9;
        let bw_gbps = weight_gb / (med_ms / 1000.0);
        println!("{:<32} {:>10.3} {:>10.3} {:>10.3} {:>10.1}",
                 label, min_ms, med_ms, max_ms, bw_gbps);
    }

    // Cleanup
    unsafe {
        dev.device.destroy_query_pool(qpool, None);
        dev.device.destroy_pipeline(pipeline, None);
        dev.device.destroy_pipeline_layout(pipeline_layout, None);
        dev.device.destroy_descriptor_set_layout(dsl, None);
        dev.device.destroy_descriptor_pool(pool, None);
        dev.device.destroy_shader_module(shader_module, None);
    }
    drop(weight_buf);
    drop(output_buf);
    drop(input_buf);
    drop(fuse0);
    drop(fuse1);
    drop(allocator);
    Ok(())
}
