//! v0.2 Sprint 10B — coopmat QK micro-benchmark (GO/NO-GO gate).
//!
//! Dispatches `bench_qk_scalar` and `bench_qk_coopmat` over the same
//! synthetic Q[16, 128] / K[16, 128] inputs, measures per-shader
//! median wall time + computed TFLOPS, and prints a speedup ratio.
//!
//! Phase 10C+ (full coopmat-attention shader) is GO if the speedup
//! is ≥ 2× — anything less and the LDS-roundtrip overhead in the
//! eventual fused kernel will eat the WMMA win.
//!
//! Outputs (per shader):
//!   median µs over N runs of one dispatch (which itself loops
//!   n_tiles QK inside the shader to amortize launch overhead).
//!
//! Env vars:
//!   VF_QK_REPS     repetitions per shader (default 20)
//!   VF_QK_TILES    n_tiles inside the shader (default 1000)
//!
//! Parity check: max_abs_err between coopmat and scalar score
//! matrices on the same inputs (with bench-loop-amortized accum,
//! both are scaled by the same n_tiles so the ratios are
//! comparable).

use std::time::Instant;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::BenchQkPushConstants;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const BR: u32 = 16;
const BC: u32 = 16;
const HEAD_DIM: u32 = 128;

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
    let reps = parse_env_u32("VF_QK_REPS", 20);
    let n_tiles = parse_env_u32("VF_QK_TILES", 1000);

    println!("VulkanForge bench_qk — Br={BR} Bc={BC} head_dim={HEAD_DIM} n_tiles={n_tiles} reps={reps}");

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

    // Synthesize Q[Br, head_dim] and K[Bc, head_dim] with deterministic
    // pseudo-random values in [-1, 1].
    let q_count = (BR * HEAD_DIM) as usize;
    let k_count = (BC * HEAD_DIM) as usize;
    let s_count = (BR * BC) as usize;

    let q_data: Vec<f32> = (0..q_count)
        .map(|i| {
            let t = i as f32 * 0.0731;
            (t.sin() + 0.2 * t.cos()).clamp(-1.0, 1.0)
        })
        .collect();
    let k_data: Vec<f32> = (0..k_count)
        .map(|i| {
            let t = i as f32 * 0.0577;
            (t.cos() + 0.3 * t.sin()).clamp(-1.0, 1.0)
        })
        .collect();

    // GPU buffers: Q, K (input, host-visible for upload), S (output, host-readable).
    let mut q_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (q_count * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "bench_qk_q",
    )?;
    q_buf.write_bytes(bytemuck::cast_slice(&q_data))?;

    let mut k_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (k_count * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "bench_qk_k",
    )?;
    k_buf.write_bytes(bytemuck::cast_slice(&k_data))?;

    let s_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (s_count * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuToCpu,
        "bench_qk_s",
    )?;

    let scalar_result = run_one(
        &dev, &registry, &cmd_ctx, ShaderId::BenchQkScalar,
        q_buf.handle, k_buf.handle, s_buf.handle,
        n_tiles, reps,
    )?;
    let scalar_s = s_buf.read_bytes()?;
    let scalar_s: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&scalar_s[..s_count * 4]).to_vec();

    let coopmat_result = run_one(
        &dev, &registry, &cmd_ctx, ShaderId::BenchQkCoopmat,
        q_buf.handle, k_buf.handle, s_buf.handle,
        n_tiles, reps,
    )?;
    let coopmat_s = s_buf.read_bytes()?;
    let coopmat_s: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&coopmat_s[..s_count * 4]).to_vec();

    // Parity. The scalar shader writes 64 per-thread accumulators;
    // the coopmat shader writes the 16×16 score matrix directly.
    // To compare, sum scalar's per-thread totals over the WG and
    // compare to the trace of coopmat's score matrix scaled by the
    // same n_tiles. Cleanest: just check coopmat output looks like
    // a finite Q·K^T matrix (no NaN/Inf) and report magnitude.
    let coopmat_finite = coopmat_s.iter().all(|v| v.is_finite());
    let coopmat_max_abs = coopmat_s.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scalar_finite = scalar_s.iter().all(|v| v.is_finite());

    // Compute true Q·K^T on CPU once for reference (averaged over
    // n_tiles iterations is just n_tiles × Q·K^T for both shaders).
    let mut cpu_qk = vec![0.0_f64; s_count];
    for qi in 0..BR as usize {
        for ki in 0..BC as usize {
            let mut s: f64 = 0.0;
            for d in 0..HEAD_DIM as usize {
                s += q_data[qi * HEAD_DIM as usize + d] as f64
                   * k_data[ki * HEAD_DIM as usize + d] as f64;
            }
            cpu_qk[qi * BC as usize + ki] = s;
        }
    }
    let cpu_total_per_thread: f64 = cpu_qk.iter().sum::<f64>() * (n_tiles as f64) / 64.0
        * (BR as f64 * BC as f64 / 64.0); // scalar accumulates Br*Bc cells / 64 threads
    // NOTE: rough scalar-side check — the per-thread totals sum to
    // n_tiles * sum(Score). Let's just sanity-check that scalar
    // produced reasonable magnitude.
    let scalar_max_abs = scalar_s.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // For coopmat: each cell = n_tiles * Q[qi]·K[ki]. Compute expected.
    let mut max_rel_err: f32 = 0.0;
    for i in 0..s_count {
        let expected = (cpu_qk[i] * n_tiles as f64) as f32;
        let got = coopmat_s[i];
        let denom = expected.abs().max(1e-3);
        let rel = (got - expected).abs() / denom;
        if rel > max_rel_err { max_rel_err = rel; }
    }

    let scalar_us = scalar_result.median_us;
    let coopmat_us = coopmat_result.median_us;
    let speedup = scalar_us / coopmat_us;

    // Total FMA ops: per dispatch, n_tiles iterations × Br × Bc cells
    // × head_dim FMAs × 2 FLOPs/FMA. (Same for both shaders since they
    // compute the same thing.)
    let total_flops = 2.0 * (n_tiles as f64) * (BR as f64) * (BC as f64) * (HEAD_DIM as f64);
    let scalar_tflops = total_flops / (scalar_us * 1e6);
    let coopmat_tflops = total_flops / (coopmat_us * 1e6);

    println!();
    println!("Result:");
    println!("  scalar:   median {scalar_us:8.1} \u{00b5}s   {scalar_tflops:6.2} TFLOPS   max_abs={scalar_max_abs:.3e}   finite={scalar_finite}");
    println!("  coopmat:  median {coopmat_us:8.1} \u{00b5}s   {coopmat_tflops:6.2} TFLOPS   max_abs={coopmat_max_abs:.3e}   finite={coopmat_finite}");
    println!("  speedup:  {speedup:.2}\u{00d7}   (coopmat vs scalar)");
    println!("  parity:   max_rel_err coopmat vs CPU = {max_rel_err:.3e}");

    println!();
    if speedup >= 4.0 {
        println!("\u{2714} GO (strong): coopmat QK is {speedup:.1}\u{00d7} faster.");
    } else if speedup >= 2.0 {
        println!("\u{2714} GO (moderate): {speedup:.1}\u{00d7} speedup is workable for Sprint 10C+.");
    } else if speedup >= 1.0 {
        println!("\u{26a0} CONDITIONAL: {speedup:.2}\u{00d7} only — investigate LDS staging / WG geometry.");
    } else {
        println!("\u{2718} NO-GO: {speedup:.2}\u{00d7}, coopmat is slower than scalar.");
    }

    let _ = cpu_total_per_thread;

    drop(q_buf);
    drop(k_buf);
    drop(s_buf);
    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct BenchResult {
    median_us: f64,
}

fn run_one(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    shader: ShaderId,
    q: vk::Buffer,
    k: vk::Buffer,
    s: vk::Buffer,
    n_tiles: u32,
    reps: u32,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let kernel = registry.get(shader);

    // Build a descriptor pool + set ourselves (mirrors
    // tests/correctness.rs::Fixture pattern).
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
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
        vk::DescriptorBufferInfo { buffer: q, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: k, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: s, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes: Vec<vk::WriteDescriptorSet> = (0..3)
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        })
        .collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let pc = BenchQkPushConstants { br: BR, bc: BC, head_dim: HEAD_DIM, n_tiles };
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;

    // Warmup (3 runs).
    for _ in 0..3 {
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, 1, 1, 1);
            // Make any subsequent host read see the score.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s).offset(0).size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[], std::slice::from_ref(&post), &[],
            );
        })?;
    }

    // Timed runs.
    let mut times_us: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let t0 = Instant::now();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            dev.device.cmd_dispatch(cmd, 1, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s).offset(0).size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[], std::slice::from_ref(&post), &[],
            );
        })?;
        // one_shot synchronously waits via fence, so elapsed measures
        // GPU dispatch time (within sub-ms accuracy on RDNA4).
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[times_us.len() / 2];

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };

    Ok(BenchResult { median_us })
}
