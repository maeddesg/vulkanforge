//! v0.2.1 Sprint 11G-C — Q4_K x Q8_1 -> FP32 GEMM bench.
//!
//! Drops both `bench_int8cm_q4k` (Int8 KHR-coopmat, M-tile) and
//! `mul_mmq_q4_k_f32` (production scalar `dotPacked4x8EXT`) on the same
//! packed buffers (block_q4_K_packed32 weights + block_q8_1_x4_packed128
//! activations) and reports per-shape median µs + GFLOPS + speedup.
//!
//! Parity is checked GPU-vs-GPU (max_abs between the two shaders' FP32
//! outputs, normalised by the result amplitude). Both kernels were
//! already validated against a CPU FP64 reference at small shapes in
//! `tests/correctness.rs::test_int8cm_q4k_microbench_parity*` and
//! `test_gemm_q4k_l_tile_qwen3_qproj_parity` respectively — we don't
//! repeat that here.
//!
//! Shapes mirror Qwen3-8B prefill GEMM:
//!   - 512 × 4096 × 4096   Q-projection at pp=512
//!   - 512 × 1024 × 4096   K/V-projection at pp=512 (GQA, kv_dim=1024)
//!   -  64 × 4096 × 4096   Q-projection at pp=64 (small dispatch)
//!
//! Env vars:
//!   VF_GEMM_REPS    bench iterations per shader per shape (default 10)

use std::time::Instant;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::{
    BenchInt8CmQ4KPushConstants, MmqPushConstants, Q8_1QuantizePushConstants,
};
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::q4k;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const SHAPES: &[(u32, u32, u32, &str)] = &[
    (512, 4096, 4096, "Q-proj @ pp=512"),
    (512, 1024, 4096, "K/V-proj @ pp=512 (GQA kv_dim=1024)"),
    (64, 4096, 4096, "Q-proj @ pp=64 (small dispatch)"),
];

fn parse_env_u32(key: &str, default_val: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default_val)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\u{2718} {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let reps = parse_env_u32("VF_GEMM_REPS", 10);

    println!(
        "VulkanForge bench_int8cm_q4k — Q4_K weights x Q8_1 activations -> FP32, reps={reps}"
    );
    println!();

    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let (registry, _) = PipelineRegistry::new(&dev.device, None)?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    println!(
        "{:<48} {:>14} {:>14} {:>10} {:>14} {:>14}",
        "Shape", "mul_mmq µs", "int8cm µs", "Speedup", "mul_mmq GFLOPS", "int8cm GFLOPS"
    );
    println!("{:-<116}", "");

    let mut speedups: Vec<f64> = Vec::new();

    for &(m, n, k, label) in SHAPES {
        assert!(k % 256 == 0, "K must be a multiple of 256");

        // --- Random Q4_K weights ---
        let weights = q4k::build_random_weights(m as usize, k as usize, 0xCAFEBABE_DEADC0DE);
        let mut w_buf = GpuBuffer::new(
            &dev.device, &mut allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w",
        )?;
        w_buf.write_bytes(&weights)?;

        // --- Random FP32 activations [N, K] row-major ---
        let acts: Vec<f32> = (0..(n * k))
            .map(|i| {
                let t = i as f32 * 0.0173;
                t.sin() * 0.4 + (t * 1.7).cos() * 0.2
            })
            .collect();
        let mut act_buf = GpuBuffer::new(
            &dev.device, &mut allocator, (acts.len() * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "act",
        )?;
        act_buf.write_bytes(bytemuck::cast_slice(&acts))?;

        // --- Q8_1 buffer (filled by quantize_q8_1 dispatch) ---
        let q8_blocks_x4 = ((n * k + 127) / 128) as usize;
        let q8_buf = GpuBuffer::new(
            &dev.device, &mut allocator, (q8_blocks_x4 * 144) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "q8",
        )?;

        // Two output buffers, one per shader, host-readable for parity check.
        let out_mul_mmq = GpuBuffer::new(
            &dev.device, &mut allocator, ((m * n) as u64) * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "out_mul_mmq",
        )?;
        let out_int8cm = GpuBuffer::new(
            &dev.device, &mut allocator, ((m * n) as u64) * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "out_int8cm",
        )?;

        // Quantize FP32 acts -> Q8_1 once; both shaders read the same buffer.
        run_quantize_q8_1(
            &dev, &registry, &cmd_ctx,
            act_buf.handle, q8_buf.handle, n * k, q8_blocks_x4 as u32,
        )?;

        // --- Time mul_mmq (production scalar dotPacked4x8EXT) ---
        let mul_mmq_us = run_mul_mmq(
            &dev, &registry, &cmd_ctx,
            w_buf.handle, q8_buf.handle, out_mul_mmq.handle,
            m, n, k, reps,
        )?;
        let mul_mmq_out_bytes = out_mul_mmq.read_bytes()?;
        let mul_mmq_out: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&mul_mmq_out_bytes[..(m * n) as usize * 4]).to_vec();

        // --- Time int8-cm Q4K ---
        let int8cm_us = run_int8cm_q4k(
            &dev, &registry, &cmd_ctx,
            w_buf.handle, q8_buf.handle, out_int8cm.handle,
            m, n, k, reps,
        )?;
        let int8cm_out_bytes = out_int8cm.read_bytes()?;
        let int8cm_out: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&int8cm_out_bytes[..(m * n) as usize * 4]).to_vec();

        // GPU-vs-GPU parity (both already validated against CPU FP64 in tests).
        let max_amax = mul_mmq_out
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let max_abs_diff = mul_mmq_out
            .iter()
            .zip(int8cm_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let rel = if max_amax > 0.0 { max_abs_diff / max_amax } else { 0.0 };

        let speedup = mul_mmq_us / int8cm_us;
        // 2 * M * N * K ops per dispatch (one MAC = 2 flops conventionally).
        let total_ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let mul_mmq_gops = total_ops / mul_mmq_us / 1e3;
        let int8cm_gops = total_ops / int8cm_us / 1e3;

        let label_full = format!("{m}×{n}×{k} ({label})");
        println!(
            "{:<48} {:>14.1} {:>14.1} {:>9.2}× {:>14.1} {:>14.1}",
            label_full, mul_mmq_us, int8cm_us, speedup, mul_mmq_gops, int8cm_gops
        );
        println!(
            "    parity: max_abs_diff = {max_abs_diff:.4e}, rel = {rel:.3e} (amax = {max_amax:.3})"
        );

        speedups.push(speedup);

        // Bench-loop drops buffers naturally on next iteration.
        drop(w_buf);
        drop(act_buf);
        drop(q8_buf);
        drop(out_mul_mmq);
        drop(out_int8cm);
    }

    println!();
    let median_speedup = {
        let mut s = speedups.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    println!("Median speedup across {} shapes: {:.2}×", speedups.len(), median_speedup);
    println!();
    if median_speedup >= 1.5 {
        println!(
            "\u{2714} STRONG GO ({median_speedup:.2}× ≥ 1.5×) — Sprint 11G-D L-tile is high-confidence."
        );
    } else if median_speedup >= 1.2 {
        println!(
            "\u{2714} GO ({median_speedup:.2}× ≥ 1.2×) — Sprint 11G-D worth pursuing."
        );
    } else if median_speedup >= 1.0 {
        println!(
            "\u{26a0} CONDITIONAL ({median_speedup:.2}×) — marginal win, review L-tile potential."
        );
    } else {
        println!(
            "\u{2718} NO-GO ({median_speedup:.2}× < 1×) — Q4_K + scale-fold overhead kills the int8-cm gain."
        );
    }

    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}

fn run_quantize_q8_1(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    input: vk::Buffer,
    output: vk::Buffer,
    n_elements: u32,
    num_blocks: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = registry.get(ShaderId::QuantizeQ8_1);
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 2,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None)? };
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc)? }[0];
    let infos = [
        vk::DescriptorBufferInfo { buffer: input, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: output, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes: Vec<vk::WriteDescriptorSet> = (0..2)
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        })
        .collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let pc = Q8_1QuantizePushConstants { ne: n_elements, num_blocks };
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;
    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, num_blocks, 1, 1);
    })?;

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_mul_mmq(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    weights: vk::Buffer,
    q8: vk::Buffer,
    out: vk::Buffer,
    m: u32,
    n: u32,
    k: u32,
    reps: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let kernel = registry.get(ShaderId::MulMmqQ4K);
    let (pool, set) = make_descriptor(dev, kernel.descriptor_set_layout, &[weights, q8, out])?;

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
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    // mul_mmq_q4_k S-tile: BM=BN=64. Match production small-tile dispatch.
    let groups_x = m.div_ceil(64);
    let groups_y = n.div_ceil(64);
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;

    let dispatch = |cmd: vk::CommandBuffer| unsafe {
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(out).offset(0).size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[], std::slice::from_ref(&post), &[],
        );
    };

    for _ in 0..3 {
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, dispatch)?;
    }
    let mut times: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let t0 = Instant::now();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, dispatch)?;
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    Ok(median)
}

#[allow(clippy::too_many_arguments)]
fn run_int8cm_q4k(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    weights: vk::Buffer,
    q8: vk::Buffer,
    out: vk::Buffer,
    m: u32,
    n: u32,
    k: u32,
    reps: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let kernel = registry.get(ShaderId::BenchInt8CmQ4K);
    let (pool, set) = make_descriptor(dev, kernel.descriptor_set_layout, &[weights, q8, out])?;

    let pc = BenchInt8CmQ4KPushConstants { m, n, k };
    // BM=BN=64 in the shader.
    let groups_x = m.div_ceil(64);
    let groups_y = n.div_ceil(64);
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;

    let dispatch = |cmd: vk::CommandBuffer| unsafe {
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(out).offset(0).size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[], std::slice::from_ref(&post), &[],
        );
    };

    for _ in 0..3 {
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, dispatch)?;
    }
    let mut times: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let t0 = Instant::now();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, dispatch)?;
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    Ok(median)
}

fn make_descriptor(
    dev: &VulkanDevice,
    set_layout: vk::DescriptorSetLayout,
    buffers: &[vk::Buffer],
) -> Result<(vk::DescriptorPool, vk::DescriptorSet), Box<dyn std::error::Error>> {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: buffers.len() as u32,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None)? };
    let layouts = [set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc)? }[0];

    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|&b| vk::DescriptorBufferInfo {
            buffer: b,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        })
        .collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
    Ok((pool, set))
}
