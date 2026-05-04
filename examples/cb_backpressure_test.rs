//! Sprint 28 Phase 1 — CB-backpressure test.
//!
//! Hypothesis: Sprint 27 found that the F16 GEMV at the 14B lm_head
//! shape (M=152064, K=5120) runs at peak BW (~685 GB/s, ~2.7 ms) in a
//! standalone CB, but the runtime profile measures ~30 ms in a CB
//! that already holds ~1200 dispatches before it. If true, dummy
//! dispatches inserted before the timed lm_head should reproduce the
//! slowdown — the lm_head TIMESTAMP should scale with the number of
//! prior dispatches.
//!
//! This bench dispatches `N` "dummy" small F16 GEMVs (M=4096 K=4096,
//! ≈32 MB weights, real GPU work) into a single CB, then measures
//! the lm_head dispatch (M=152064 K=5120) at the END of the same CB
//! via TIMESTAMP. Sweeps N over a range that brackets the runtime's
//! 1200 dispatch count.
//!
//! Expected outcomes:
//!   * lm_head time scales with N → CB-backpressure confirmed → ship
//!     multi-submit decode in Phase 2.
//!   * lm_head time is roughly constant → CB-backpressure rejected →
//!     stop, look elsewhere (VRAM placement, pipeline cache).

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
use vulkanforge::backend::vulkan::shaders::MUL_MAT_VEC_F16;

const LM_M: u32 = 152064;     // 14B vocab
const LM_K: u32 = 5120;       // 14B hidden
const DUMMY_M: u32 = 4096;    // realistic layer GEMV
const DUMMY_K: u32 = 4096;
const RUNS_PER_SWEEP: u32 = 5;

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

    // -- Pipeline (single F16 GEMV, reused for both lm_head and dummies) --
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

    // -- Buffers --
    // lm_head shape: 152064 × 5120 × 2B = 1.56 GiB
    let lm_weight_bytes = (LM_M as u64) * (LM_K as u64) * 2;
    let lm_weight = GpuBuffer::new(
        &dev.device, &mut allocator, lm_weight_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly, "lm_weight",
    )?;

    // dummy GEMV shape: 4096 × 4096 × 2B = 32 MiB
    let dummy_weight_bytes = (DUMMY_M as u64) * (DUMMY_K as u64) * 2;
    let dummy_weight = GpuBuffer::new(
        &dev.device, &mut allocator, dummy_weight_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly, "dummy_weight",
    )?;

    // Fill both with non-zero data (touched VRAM).
    let chunk_elems: u32 = 8 * 1024 * 1024; // 16 MiB per chunk
    let mut tmp = GpuBuffer::new(
        &dev.device, &mut allocator, (chunk_elems * 2) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "seed_tmp",
    )?;
    let seed: Vec<u16> = (0..chunk_elems).map(|i| ((i & 0xff) as u16) | 0x3c00).collect();
    tmp.write_bytes(bytemuck::cast_slice(&seed))?;
    for (target, total_u16) in [(&lm_weight, lm_weight_bytes / 2), (&dummy_weight, dummy_weight_bytes / 2)] {
        let mut filled: u64 = 0;
        while filled < total_u16 {
            let remaining = total_u16 - filled;
            let chunk = remaining.min(chunk_elems as u64);
            cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                let copy = vk::BufferCopy::default()
                    .src_offset(0).dst_offset(filled * 2).size(chunk * 2);
                dev.device.cmd_copy_buffer(cmd, tmp.handle, target.handle, std::slice::from_ref(&copy));
            })?;
            filled += chunk;
        }
    }
    tmp.destroy(&dev.device, &mut allocator);

    // Inputs (FP32) and outputs (FP32).
    let input_bytes = (LM_K.max(DUMMY_K) as u64) * 4;
    let mut input_buf = GpuBuffer::new(
        &dev.device, &mut allocator, input_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "input_buf",
    )?;
    let input: Vec<f32> = (0..(input_bytes / 4) as u32)
        .map(|i| ((i as f32) * 0.001).sin()).collect();
    input_buf.write_bytes(bytemuck::cast_slice(&input))?;

    let output_bytes = (LM_M as u64) * 4;
    let output_buf = GpuBuffer::new(
        &dev.device, &mut allocator, output_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "output_buf",
    )?;

    let fuse0 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse0")?;
    let fuse1 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse1")?;

    // -- Two descriptor sets: one for lm_head (binds lm_weight), one for dummy --
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 16,
    }];
    let pool = unsafe {
        dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&pool_sizes), None,
        )?
    };
    let set_layouts = [dsl, dsl];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&set_layouts);
    let sets = unsafe { dev.device.allocate_descriptor_sets(&alloc_info)? };
    let lm_set = sets[0];
    let dummy_set = sets[1];

    let write_set = |set: vk::DescriptorSet, weight_buf: &GpuBuffer| {
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
        unsafe { dev.device.update_descriptor_sets(&writes, &[]); }
    };
    write_set(lm_set, &lm_weight);
    write_set(dummy_set, &dummy_weight);

    // -- TIMESTAMP query pool (1 pair) --
    let qpool = unsafe {
        dev.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP).query_count(2),
            None,
        )?
    };
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let ts_period_ns = props.limits.timestamp_period as f64;

    let dummy_pc = MatVecPushConstants {
        ncols: DUMMY_K, stride_a: DUMMY_K, stride_b: DUMMY_K, stride_d: DUMMY_M,
        batch_stride_a: DUMMY_K * DUMMY_M, batch_stride_b: DUMMY_K, batch_stride_d: DUMMY_M,
        fusion_flags: 0, base_work_group_y: 0,
        ne02: 1, ne12: 1, broadcast2: 1,
        broadcast3: 1.0_f32.to_bits(),
    };
    let lm_pc = MatVecPushConstants {
        ncols: LM_K, stride_a: LM_K, stride_b: LM_K, stride_d: LM_M,
        batch_stride_a: LM_K * LM_M, batch_stride_b: LM_K, batch_stride_d: LM_M,
        fusion_flags: 0, base_work_group_y: 0,
        ne02: 1, ne12: 1, broadcast2: 1,
        broadcast3: 1.0_f32.to_bits(),
    };

    println!();
    println!("CB-backpressure sweep — lm_head ({} × {}) after N dummy GEMVs ({} × {})",
             LM_M, LM_K, DUMMY_M, DUMMY_K);
    println!("{:>8} {:>10} {:>10} {:>10}", "n_dummy", "min ms", "med ms", "max ms");
    println!("{}", "-".repeat(40));

    let n_dummies = [0u32, 50, 100, 200, 400, 800, 1200, 1600, 2400];
    for &n in &n_dummies {
        // Warmup once (not measured).
        run_one(&dev, &cmd_ctx, qpool, pipeline, pipeline_layout,
                lm_set, dummy_set, &lm_pc, &dummy_pc, n)?;
        let mut times: Vec<f64> = Vec::with_capacity(RUNS_PER_SWEEP as usize);
        for _ in 0..RUNS_PER_SWEEP {
            let ms = run_one(&dev, &cmd_ctx, qpool, pipeline, pipeline_layout,
                             lm_set, dummy_set, &lm_pc, &dummy_pc, n)?;
            times.push(ms);
            // Read the timestamp value computed inside run_one.
            let mut data = [0u64; 2];
            unsafe {
                dev.device.get_query_pool_results(
                    qpool, 0, &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            let _ = data;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{:>8} {:>10.3} {:>10.3} {:>10.3}",
                 n, times[0], times[times.len() / 2], times[times.len() - 1]);
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
    drop(lm_weight);
    drop(dummy_weight);
    drop(output_buf);
    drop(input_buf);
    drop(fuse0);
    drop(fuse1);
    drop(allocator);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    dev: &VulkanDevice,
    cmd_ctx: &CommandContext,
    qpool: vk::QueryPool,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    lm_set: vk::DescriptorSet,
    dummy_set: vk::DescriptorSet,
    lm_pc: &MatVecPushConstants,
    dummy_pc: &MatVecPushConstants,
    n_dummies: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device.cmd_reset_query_pool(cmd, qpool, 0, 2);

        // (1) N dummy dispatches — all bound to dummy_set, so they
        // are RAW-independent (no barriers between them, like the
        // runtime's layer GEMVs which are also RAW-independent within
        // the same WG-row group).
        if n_dummies > 0 {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[dummy_set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(dummy_pc),
            );
            for _ in 0..n_dummies {
                dev.device.cmd_dispatch(cmd, DUMMY_M, 1, 1);
            }
        }

        // (2) compute→compute barrier (mimics the runtime's
        // `maybe_compute_barrier` before lm_head).
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&mb), &[], &[],
        );

        // (3) lm_head dispatch — bracketed by TIMESTAMPs.
        dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, 0);
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[lm_set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE,
            0, bytemuck::bytes_of(lm_pc),
        );
        dev.device.cmd_dispatch(cmd, LM_M, 1, 1);
        dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, 1);
    })?;

    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let ts_period_ns = props.limits.timestamp_period as f64;
    let mut data = [0u64; 2];
    unsafe {
        dev.device.get_query_pool_results(
            qpool, 0, &mut data,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
    }
    let ticks = data[1].wrapping_sub(data[0]);
    let ns = (ticks as f64) * ts_period_ns;
    Ok(ns / 1e6)
}
