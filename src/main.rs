//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 1 / Step 1.3: device + pipeline + GPU-buffer allocation +
//! staging upload + CPU reference. The dispatch and readback land
//! in step 1.4; this run only validates that the buffer plumbing
//! works end-to-end and produces no validation noise.

mod backend;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use backend::vulkan::buffers::GpuBuffer;
use backend::vulkan::commands::CommandContext;
use backend::vulkan::device::VulkanDevice;
use backend::vulkan::pipeline::{ComputeKernel, SpecConstants};
use backend::vulkan::q4k;
use backend::vulkan::shaders;

const M: usize = 2; // output rows
const K: usize = q4k::QUANT_K; // input length / cols of A — 256 (one Q4_K block per row)

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

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    println!("✅ gpu_allocator initialized");

    // Step 1.2 — pipeline.
    let spv = shaders::spv_words(shaders::MUL_MAT_VEC_Q4_K_F32_F32);
    let kernel = ComputeKernel::new(&dev.device, &spv, SpecConstants::SMOKE_DEFAULT)?;
    println!(
        "✅ Q4_K GEMV pipeline (BLOCK_SIZE={}, NUM_ROWS={}, NUM_COLS={})",
        SpecConstants::SMOKE_DEFAULT.block_size,
        SpecConstants::SMOKE_DEFAULT.num_rows,
        SpecConstants::SMOKE_DEFAULT.num_cols,
    );

    // Step 1.3 — buffers + test data.
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

    // Staging-upload buffers (host-visible, transient, drop after copy).
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
    println!("✅ Staging buffers populated (host-visible)");

    // Run the copies on a one-shot command buffer on the compute queue.
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
    println!("✅ Staging upload complete (weights + input → DEVICE_LOCAL)");

    // Step 1.3 closes with the CPU reference. Step 1.4 will compare
    // the GPU output against this.
    let cpu_ref = q4k::cpu_gemv(&weights_bytes, M, K, &input);
    println!(
        "✅ CPU reference: output[0]={:.3}, output[1]={:.3}",
        cpu_ref[0], cpu_ref[1]
    );
    let expected = [256.0f32, 512.0];
    for i in 0..M {
        let err = (cpu_ref[i] - expected[i]).abs();
        if err > 1e-3 {
            return Err(format!(
                "CPU reference output[{i}]={} does not match analytical {} (err={})",
                cpu_ref[i], expected[i], err
            )
            .into());
        }
    }
    println!(
        "✅ CPU reference matches analytical expectation [256.0, 512.0] (max abs err < 1e-3)"
    );

    // ---- Teardown — buffers → command pool → kernel → allocator → device.
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
