//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 1 / Step 1.4: device + pipeline + GPU buffers + staging
//! upload + dispatch + readback + parity check against the CPU
//! reference. Step 1.5 will scale to realistic Qwen-3-class
//! dimensions and add 100-run timing statistics.

mod backend;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use backend::vulkan::buffers::GpuBuffer;
use backend::vulkan::commands::CommandContext;
use backend::vulkan::device::VulkanDevice;
use backend::vulkan::pipeline::{
    ComputeKernel, MatVecPushConstants, PUSH_CONSTANT_BYTES, SpecConstants,
};
use backend::vulkan::q4k;
use backend::vulkan::shaders;

const M: usize = 2; // output rows
const K: usize = q4k::QUANT_K; // input length / cols of A — 256 (one Q4_K block per row)

/// Smoke-test thresholds. Q4_K can quantize 1.0/2.0 exactly, so we
/// expect bit-equal output here; the loose bounds catch rounding
/// drift if the shader path computes in fp16.
const ABS_ERR_THRESHOLD: f32 = 1e-2;
const REL_ERR_THRESHOLD: f32 = 0.05;

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge v0.1.0");

    let dev = VulkanDevice::new()?;
    println!("✅ Vulkan device initialized");

    let device_props = unsafe {
        dev.instance
            .get_physical_device_properties(dev.physical_device)
    };
    let queue_family_props = unsafe {
        dev.instance
            .get_physical_device_queue_family_properties(dev.physical_device)
    };
    let timestamp_period = device_props.limits.timestamp_period;
    let timestamp_valid_bits =
        queue_family_props[dev.queue_family_index as usize].timestamp_valid_bits;
    if timestamp_valid_bits == 0 {
        return Err("Compute queue does not support timestamp queries".into());
    }
    println!(
        "  Timestamp: {} valid bits, {:.3} ns/tick",
        timestamp_valid_bits, timestamp_period
    );

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    println!("✅ gpu_allocator initialized");

    // ---- Step 1.2 — pipeline.
    let spv = shaders::spv_words(shaders::MUL_MAT_VEC_Q4_K_F32_F32);
    let kernel = ComputeKernel::new(&dev.device, &spv, SpecConstants::SMOKE_DEFAULT)?;
    println!(
        "✅ Q4_K GEMV pipeline (BLOCK_SIZE={}, NUM_ROWS={}, NUM_COLS={})",
        SpecConstants::SMOKE_DEFAULT.block_size,
        SpecConstants::SMOKE_DEFAULT.num_rows,
        SpecConstants::SMOKE_DEFAULT.num_cols,
    );

    // ---- Step 1.3 — buffers + test data.
    let weights_bytes = q4k::build_smoke_weights();
    let input: Vec<f32> = vec![1.0; K];
    let input_bytes: &[u8] = bytemuck::cast_slice(&input);

    let weights_size = weights_bytes.len() as u64; // 288
    let input_size = input_bytes.len() as u64; // 1024
    let output_size = (M * std::mem::size_of::<f32>()) as u64; // 8
    let dummy_size: u64 = 16;

    let storage_dst = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

    let weights_buf = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        weights_size,
        storage_dst,
        MemoryLocation::GpuOnly,
        "weights",
    )?;
    let input_buf = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        input_size,
        storage_dst,
        MemoryLocation::GpuOnly,
        "input",
    )?;
    let output_buf = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        output_size,
        storage_only,
        MemoryLocation::GpuToCpu,
        "output",
    )?;
    let fuse0_buf = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        dummy_size,
        storage_only,
        MemoryLocation::GpuOnly,
        "fuse0_dummy",
    )?;
    let fuse1_buf = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        dummy_size,
        storage_only,
        MemoryLocation::GpuOnly,
        "fuse1_dummy",
    )?;
    println!(
        "✅ Storage buffers: weights={} B, input={} B, output={} B, fuse0={} B, fuse1={} B",
        weights_size, input_size, output_size, dummy_size, dummy_size,
    );

    let mut staging_weights = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        weights_size,
        staging_src,
        MemoryLocation::CpuToGpu,
        "staging_weights",
    )?;
    let mut staging_input = GpuBuffer::new(
        &dev.device,
        &mut allocator,
        input_size,
        staging_src,
        MemoryLocation::CpuToGpu,
        "staging_input",
    )?;
    staging_weights.write_bytes(&weights_bytes)?;
    staging_input.write_bytes(input_bytes)?;

    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;
    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
        let copy_w = vk::BufferCopy::default().size(weights_size);
        let copy_i = vk::BufferCopy::default().size(input_size);
        unsafe {
            dev.device.cmd_copy_buffer(
                cmd,
                staging_weights.handle,
                weights_buf.handle,
                std::slice::from_ref(&copy_w),
            );
            dev.device.cmd_copy_buffer(
                cmd,
                staging_input.handle,
                input_buf.handle,
                std::slice::from_ref(&copy_i),
            );
        }
    })?;
    println!("✅ Staging upload complete");

    let cpu_ref = q4k::cpu_gemv(&weights_bytes, M, K, &input);
    println!(
        "  CPU reference: [{:.6}, {:.6}]",
        cpu_ref[0], cpu_ref[1]
    );

    // ---- Step 1.4 — descriptor pool, descriptor set, dispatch, readback.

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None)? };

    let layouts = [kernel.descriptor_set_layout];
    let set_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_set = unsafe { dev.device.allocate_descriptor_sets(&set_alloc)? }[0];

    let buffer_infos = [
        vk::DescriptorBufferInfo {
            buffer: weights_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: input_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: output_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: fuse0_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: fuse1_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[2..3]),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[3..4]),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[4..5]),
    ];
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
    println!("✅ Descriptor set updated (5 storage-buffer bindings)");

    let qp_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query_pool = unsafe { dev.device.create_query_pool(&qp_info, None)? };

    // Push-constant values per step 1.0 §2.1 with M=2 K=256 batch=1.
    let pc = MatVecPushConstants {
        ncols: K as u32,
        stride_a: K as u32,
        stride_b: K as u32,
        stride_d: M as u32,
        batch_stride_a: (K * M) as u32,
        batch_stride_b: K as u32,
        batch_stride_d: M as u32,
        fusion_flags: 0,
        base_work_group_y: 0,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };
    let pc_bytes: &[u8] = bytemuck::bytes_of(&pc);
    assert_eq!(pc_bytes.len(), PUSH_CONSTANT_BYTES as usize);

    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device.cmd_reset_query_pool(cmd, query_pool, 0, 2);

        // TRANSFER_WRITE (prior staging upload) → SHADER_READ. Submit
        // boundaries serialize execution but not memory visibility on
        // their own; this barrier is what makes the upload observable
        // to the dispatch, even when validation's sync-validation is
        // off.
        let pre = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(weights_buf.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(input_buf.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE),
        ];
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &pre,
            &[],
        );

        dev.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0);

        dev.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            kernel.pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        dev.device.cmd_push_constants(
            cmd,
            kernel.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        // Dispatch dimensions: ceil(M / NUM_ROWS) × batch × 1.
        // SMOKE_DEFAULT.num_rows = 1, M = 2, batch = 1 → (2, 1, 1).
        dev.device.cmd_dispatch(cmd, 2, 1, 1);

        dev.device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_pool,
            1,
        );

        // SHADER_WRITE → HOST_READ on output, so the host map below
        // is guaranteed to see the dispatch's writes.
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(output_buf.handle)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[post],
            &[],
        );
    })?;
    println!("✅ Dispatch executed (2, 1, 1)");

    let mut timestamps = [0u64; 2];
    unsafe {
        dev.device.get_query_pool_results(
            query_pool,
            0,
            &mut timestamps,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
    }
    let ticks = timestamps[1].wrapping_sub(timestamps[0]);
    let kernel_ns = ticks as f64 * timestamp_period as f64;
    let kernel_us = kernel_ns / 1000.0;
    println!(
        "  Kernel time: {:.3} µs ({} ticks × {:.3} ns)",
        kernel_us, ticks, timestamp_period
    );

    let output_bytes = output_buf.read_bytes()?;
    let gpu_out = [
        f32::from_le_bytes(output_bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(output_bytes[4..8].try_into().unwrap()),
    ];
    println!("  GPU output: [{:.6}, {:.6}]", gpu_out[0], gpu_out[1]);
    println!("  CPU ref   : [{:.6}, {:.6}]", cpu_ref[0], cpu_ref[1]);

    let all_zeros = gpu_out.iter().all(|&v| v == 0.0);
    let any_nonfinite = gpu_out.iter().any(|&v| !v.is_finite());
    if all_zeros {
        return Err("smoke check: GPU output is all-zeros".into());
    }
    if any_nonfinite {
        return Err("smoke check: GPU output contains NaN/Inf".into());
    }

    let max_abs_err = gpu_out
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    let max_rel_err = gpu_out
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| if *c == 0.0 { 0.0 } else { (g - c).abs() / c.abs() })
        .fold(0.0f32, f32::max);
    println!(
        "  max_abs_err = {:.6e}, max_rel_err = {:.6e}",
        max_abs_err, max_rel_err
    );
    if max_abs_err > ABS_ERR_THRESHOLD {
        return Err(format!(
            "smoke check: max_abs_err {:.6e} exceeds threshold {:.0e}",
            max_abs_err, ABS_ERR_THRESHOLD
        )
        .into());
    }
    if max_rel_err > REL_ERR_THRESHOLD {
        return Err(format!(
            "smoke check: max_rel_err {:.6e} exceeds threshold {:.0e}",
            max_rel_err, REL_ERR_THRESHOLD
        )
        .into());
    }
    println!("✅ Smoke test PASSED — GPU output matches CPU reference");

    // ---- Teardown — descriptor pool first (frees the set), then
    // query pool, then user-managed objects in reverse order.
    unsafe {
        dev.device.destroy_query_pool(query_pool, None);
        dev.device.destroy_descriptor_pool(descriptor_pool, None);
    }
    staging_input.destroy(&dev.device, &mut allocator);
    staging_weights.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    fuse1_buf.destroy(&dev.device, &mut allocator);
    fuse0_buf.destroy(&dev.device, &mut allocator);
    output_buf.destroy(&dev.device, &mut allocator);
    input_buf.destroy(&dev.device, &mut allocator);
    weights_buf.destroy(&dev.device, &mut allocator);
    kernel.destroy(&dev.device);
    drop(allocator);
    println!("✅ Teardown clean");
    Ok(())
}
