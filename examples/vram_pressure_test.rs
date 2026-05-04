//! Sprint 28B — VRAM-pressure test for the lm_head perf gap.
//!
//! Sprint 27 + 28 ruled out: shader perf, PCIe scattered writes,
//! memory compression, CB-backpressure, shape-specific issues.
//!
//! Remaining suspects (Sprint 28's open list):
//!   1. VRAM pressure / page-table hot-spots — 14B model is 13.77 GiB
//!      vs ~2.5 GiB in the standalone bench (this is the test).
//!   2. Weight-buffer placement (vram_arena vs fresh allocator chunk).
//!   3. Pipeline cache poisoning.
//!
//! This bench reproduces the lm_head shape (M=152064, K=5120) but
//! allocates ~13 GiB of "ballast" buffers BEFORE the timed dispatch
//! to put the GPU at the same VRAM occupancy as a 14B model load.
//! If lm_head time inflates from ~2.7 ms to ~30 ms with ballast,
//! VRAM pressure is the cause.
//!
//! Usage:
//!   cargo run --release --example vram_pressure_test -- [ballast_gib]
//!
//! `ballast_gib` defaults to 13. Use 0 to disable ballast (control).

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
use vulkanforge::backend::vulkan::shaders::MUL_MAT_VEC_F16;

const LM_M: u32 = 152064;
const LM_K: u32 = 5120;
const RUNS: u32 = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ballast_gib: u64 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok()).unwrap_or(13);

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

    // Sprint 29 — `VF_BALLAST_FIRST=1` swaps the order: ballast
    // gets allocated FIRST so that the lm_head weight buffer ends
    // up at a high VRAM offset (mirroring the runtime, where the
    // SafeTensors loader allocates ~50 tensors before output.weight).
    let ballast_first = std::env::var("VF_BALLAST_FIRST").is_ok();
    if ballast_first {
        eprintln!("VF_BALLAST_FIRST=1 — allocating ballast BEFORE lm_head buffers");
    }

    // Step 1 — allocate the lm_head working set FIRST (1.56 GiB
    // weight + 5KB input + 608KB output + 16B*2 fuse), so it sits
    // near the bottom of VRAM. Then allocate ballast on top to
    // mirror a 14B model's full VRAM occupancy.
    // If VF_BALLAST_FIRST is set, allocate ballast FIRST so the
    // lm_head weight buffer ends up at a high VRAM offset.
    let lm_weight_bytes = (LM_M as u64) * (LM_K as u64) * 2;
    let mut early_ballast: Vec<GpuBuffer> = Vec::new();
    if ballast_first && ballast_gib > 0 {
        let chunk_size_gib: u64 = 512 * 1024 * 1024;
        let total = ballast_gib * 1024 * 1024 * 1024;
        let mut allocated: u64 = 0;
        while allocated < total {
            let remaining = total - allocated;
            let size = remaining.min(chunk_size_gib);
            early_ballast.push(GpuBuffer::new(
                &dev.device, &mut allocator, size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly, "early_ballast",
            )?);
            allocated += size;
        }
        eprintln!("  early ballast: {} chunks before lm_weight", early_ballast.len());
    }

    eprintln!("Allocating lm_head weight: {:.2} GiB", lm_weight_bytes as f64 / (1024.0_f64.powi(3)));
    let lm_weight = GpuBuffer::new(
        &dev.device, &mut allocator, lm_weight_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly, "lm_weight",
    )?;
    let mut input_buf = GpuBuffer::new(
        &dev.device, &mut allocator, (LM_K as u64) * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "input_buf",
    )?;
    let input: Vec<f32> = (0..LM_K).map(|i| ((i as f32) * 0.001).sin()).collect();
    input_buf.write_bytes(bytemuck::cast_slice(&input))?;
    let output_buf = GpuBuffer::new(
        &dev.device, &mut allocator, (LM_M as u64) * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "output_buf",
    )?;
    let fuse0 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse0")?;
    let fuse1 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse1")?;

    // Fill lm_weight with non-zero data (avoid memory-compression fast path).
    let chunk_elems: u32 = 8 * 1024 * 1024; // 16 MiB
    let mut tmp = GpuBuffer::new(
        &dev.device, &mut allocator, (chunk_elems * 2) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "seed_tmp",
    )?;
    let seed: Vec<u16> = (0..chunk_elems).map(|i| ((i & 0xff) as u16) | 0x3c00).collect();
    tmp.write_bytes(bytemuck::cast_slice(&seed))?;
    let mut filled: u64 = 0;
    let total_u16 = lm_weight_bytes / 2;
    while filled < total_u16 {
        let remaining = total_u16 - filled;
        let chunk = remaining.min(chunk_elems as u64);
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let copy = vk::BufferCopy::default()
                .src_offset(0).dst_offset(filled * 2).size(chunk * 2);
            dev.device.cmd_copy_buffer(cmd, tmp.handle, lm_weight.handle, std::slice::from_ref(&copy));
        })?;
        filled += chunk;
    }
    eprintln!("  lm_head weight filled");

    // Step 2 — allocate ballast (skipped if ballast was allocated early).
    let mut ballast_bufs: Vec<GpuBuffer> = Vec::new();
    if ballast_gib > 0 && !ballast_first {
        eprintln!("Allocating {} GiB of ballast (chunked)...", ballast_gib);
        // gpu_allocator can be fussy about big single allocations.
        // Allocate as multiple 512 MiB chunks.
        let chunk_size_gib: u64 = 512 * 1024 * 1024; // 512 MiB
        let total = ballast_gib * 1024 * 1024 * 1024;
        let mut allocated: u64 = 0;
        while allocated < total {
            let remaining = total - allocated;
            let size = remaining.min(chunk_size_gib);
            let buf = GpuBuffer::new(
                &dev.device, &mut allocator, size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                &format!("ballast_{}", ballast_bufs.len()),
            )?;
            // Touch the chunk so it's actually backed by memory.
            let touch_bytes = size.min(64 * 1024 * 1024); // first 64 MiB per chunk
            let mut touch_filled: u64 = 0;
            while touch_filled < touch_bytes {
                let n = (touch_bytes - touch_filled).min((chunk_elems * 2) as u64);
                cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                    let copy = vk::BufferCopy::default()
                        .src_offset(0).dst_offset(touch_filled).size(n);
                    dev.device.cmd_copy_buffer(cmd, tmp.handle, buf.handle, std::slice::from_ref(&copy));
                })?;
                touch_filled += n;
            }
            ballast_bufs.push(buf);
            allocated += size;
            if (allocated / chunk_size_gib) % 4 == 0 {
                eprintln!("  ballast: {:.1} / {} GiB", allocated as f64 / (1024.0_f64.powi(3)), ballast_gib);
            }
        }
        eprintln!("  ballast filled: {} chunks, {} GiB", ballast_bufs.len(), ballast_gib);
    }
    tmp.destroy(&dev.device, &mut allocator);

    // Step 3 — descriptor set + qpool, then run the bench.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8,
    }];
    let pool = unsafe {
        dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None,
        )?
    };
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc_info)?[0] };
    let infos = [
        vk::DescriptorBufferInfo::default().buffer(lm_weight.handle).offset(0).range(vk::WHOLE_SIZE),
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
    unsafe { dev.device.update_descriptor_sets(&writes, &[]); }

    let qpool = unsafe {
        dev.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP).query_count(2),
            None,
        )?
    };
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let ts_period_ns = props.limits.timestamp_period as f64;

    let pc = MatVecPushConstants {
        ncols: LM_K, stride_a: LM_K, stride_b: LM_K, stride_d: LM_M,
        batch_stride_a: LM_K * LM_M, batch_stride_b: LM_K, batch_stride_d: LM_M,
        fusion_flags: 0, base_work_group_y: 0,
        ne02: 1, ne12: 1, broadcast2: 1,
        broadcast3: 1.0_f32.to_bits(),
    };

    let mut times: Vec<f64> = Vec::with_capacity(RUNS as usize);
    for _ in 0..RUNS {
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
            dev.device.cmd_dispatch(cmd, LM_M, 1, 1);
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
        times.push(ns / 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let weight_gb = lm_weight_bytes as f64 / 1e9;
    let med = times[times.len() / 2];
    let bw = weight_gb / (med / 1000.0);

    println!();
    println!("=== VRAM-pressure test ===");
    println!("Ballast: {} GiB ({} chunks)", ballast_gib, ballast_bufs.len());
    println!("lm_head shape: {} × {} (FP16, {:.2} GiB weights)", LM_M, LM_K, weight_gb);
    println!("Runs: {}", RUNS);
    println!();
    println!("  min ms  | med ms  | max ms  | median GB/s");
    println!(" ---------|---------|---------|------------");
    println!("  {:>6.3}  | {:>6.3}  | {:>6.3}  | {:>10.1}",
             times[0], med, times[times.len() - 1], bw);
    println!();
    println!("Reference (no ballast, Sprint 27 standalone): ~2.27 ms / ~685 GB/s");
    println!("Runtime (14B model loaded, Sprint 26 profiler): ~29.7 ms");

    // Cleanup
    unsafe {
        dev.device.destroy_query_pool(qpool, None);
        dev.device.destroy_pipeline(pipeline, None);
        dev.device.destroy_pipeline_layout(pipeline_layout, None);
        dev.device.destroy_descriptor_set_layout(dsl, None);
        dev.device.destroy_descriptor_pool(pool, None);
        dev.device.destroy_shader_module(shader_module, None);
    }
    drop(lm_weight);
    drop(output_buf);
    drop(input_buf);
    drop(fuse0);
    drop(fuse1);
    for b in ballast_bufs {
        b.destroy(&dev.device, &mut allocator);
    }
    for b in early_ballast {
        b.destroy(&dev.device, &mut allocator);
    }
    drop(allocator);
    Ok(())
}
