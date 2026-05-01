//! v0.2.1 Sprint 11G-B — Int8 coopmat GEMM micro-benchmark (GO/NO-GO gate).
//!
//! Dispatches `bench_int8cm_gemm` and `bench_scalar_gemm` over the same
//! synthetic int8 inputs (A[M, K] and B[K, N]) at several shapes that
//! mirror Qwen3-8B's prefill GEMM:
//!   - 512 × 4096 × 4096   (Q-projection at pp=512)
//!   - 512 × 1024 × 4096   (K/V-projection at pp=512, GQA: kv_dim=1024)
//!   -  64 × 4096 × 4096   (small-pp shape, dispatch-bound)
//!
//! Both shaders share the same tile shape (BM=BN=16, BK=32), the same
//! 64-thread Wave64 WG layout, and the same n_reps amortization loop.
//! The only difference is the inner K instruction: coopMatMulAdd vs
//! dotPacked4x8EXT. Speedup is the GO/NO-GO answer for Sprint 11G-C.
//!
//! Parity: int8 × int8 → int32 has no rounding, so both shaders
//! produce bit-identical output if the kernel logic is correct. We
//! verify against a CPU int64 reference to be sure no overflow occurs.
//!
//! Env vars:
//!   VF_GEMM_REPS    bench iterations per shape per shader (default 10)
//!   VF_GEMM_NREPS   inner-loop n_reps inside the shader (default 4)

use std::time::Instant;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::BenchInt8CmGemmPushConstants;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const SHAPES: &[(u32, u32, u32, &str)] = &[
    (512, 4096, 4096, "Q-proj @ pp=512"),
    (512, 1024, 4096, "K/V-proj @ pp=512 (GQA kv_dim=1024)"),
    (64, 4096, 4096, "Q-proj @ pp=64 (dispatch-bound)"),
];

fn parse_env_u32(key: &str, default_val: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default_val)
}

fn pseudo_i8(seed: u32) -> Vec<i8> {
    // Deterministic pseudo-random ints in [-8, 8). Using a small range
    // keeps every product in i8 range (max |·|=64) and a length-K dot
    // bounded well inside i32 even for K=4096 (max ≈ 4096·64·8 ≈ 2.1M).
    let mut s = seed.wrapping_mul(2_654_435_761);
    let mut out = Vec::with_capacity(seed as usize);
    for _ in 0..seed {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let v = ((s >> 16) & 0xff) as i32;
        out.push(((v % 16) - 8) as i8);
    }
    out
}

fn cpu_reference_gemm(a: &[i8], b: &[i8], m: u32, n: u32, k: u32, n_reps: u32) -> Vec<i32> {
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let mut c = vec![0_i64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s: i64 = 0;
            for ki in 0..k {
                s += a[mi * k + ki] as i64 * b[ki * n + ni] as i64;
            }
            c[mi * n + ni] = s;
        }
    }
    // Both shaders accumulate `total += C` over n_reps; final value =
    // n_reps * single-pass result.
    c.into_iter().map(|v| (v * n_reps as i64) as i32).collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\u{2718} {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let reps = parse_env_u32("VF_GEMM_REPS", 10);
    let n_reps = parse_env_u32("VF_GEMM_NREPS", 4);

    println!(
        "VulkanForge bench_int8cm_gemm — reps={reps} n_reps={n_reps} (inner-loop amortization)"
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
        "Shape", "Scalar (µs)", "Int8-cm (µs)", "Speedup", "Scalar GOPS", "Int8-cm GOPS"
    );
    println!("{:-<116}", "");

    let mut all_pass = true;
    let mut speedups: Vec<f64> = Vec::new();

    for &(m, n, k, label) in SHAPES {
        let a = pseudo_i8(m * k);
        let b = pseudo_i8(k * n);

        let a_bytes: Vec<u8> = a.iter().map(|&v| v as u8).collect();
        let b_bytes: Vec<u8> = b.iter().map(|&v| v as u8).collect();

        let mut a_buf = GpuBuffer::new(
            &dev.device,
            &mut allocator,
            a_bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "bench_a",
        )?;
        a_buf.write_bytes(&a_bytes)?;
        let mut b_buf = GpuBuffer::new(
            &dev.device,
            &mut allocator,
            b_bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "bench_b",
        )?;
        b_buf.write_bytes(&b_bytes)?;

        let c_size = (m * n) as u64 * 4;
        let c_buf = GpuBuffer::new(
            &dev.device,
            &mut allocator,
            c_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu,
            "bench_c",
        )?;

        // CPU reference for parity.
        let cpu_ref = cpu_reference_gemm(&a, &b, m, n, k, n_reps);

        // Scalar shader.
        let scalar_us = run_one(
            &dev,
            &registry,
            &cmd_ctx,
            ShaderId::BenchScalarGemm,
            a_buf.handle,
            b_buf.handle,
            c_buf.handle,
            m,
            n,
            k,
            n_reps,
            reps,
        )?;
        let scalar_out_bytes = c_buf.read_bytes()?;
        let scalar_out: Vec<i32> =
            bytemuck::cast_slice::<u8, i32>(&scalar_out_bytes[..(m * n) as usize * 4]).to_vec();
        let scalar_pass = scalar_out
            .iter()
            .zip(cpu_ref.iter())
            .all(|(&got, &exp)| got == exp);

        // Int8-cm shader.
        let int8cm_us = run_one(
            &dev,
            &registry,
            &cmd_ctx,
            ShaderId::BenchInt8CmGemm,
            a_buf.handle,
            b_buf.handle,
            c_buf.handle,
            m,
            n,
            k,
            n_reps,
            reps,
        )?;
        let int8cm_out_bytes = c_buf.read_bytes()?;
        let int8cm_out: Vec<i32> =
            bytemuck::cast_slice::<u8, i32>(&int8cm_out_bytes[..(m * n) as usize * 4]).to_vec();
        let int8cm_pass = int8cm_out
            .iter()
            .zip(cpu_ref.iter())
            .all(|(&got, &exp)| got == exp);

        let bit_identical = scalar_out == int8cm_out;

        let speedup = scalar_us / int8cm_us;
        // Total GEMM ops per dispatch: 2 * M * N * K * n_reps.
        let total_ops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (n_reps as f64);
        let scalar_gops = total_ops / scalar_us / 1e3;
        let int8cm_gops = total_ops / int8cm_us / 1e3;

        let label_full = format!("{m}×{n}×{k} ({label})");
        println!(
            "{:<48} {:>14.1} {:>14.1} {:>9.2}× {:>14.1} {:>14.1}",
            label_full, scalar_us, int8cm_us, speedup, scalar_gops, int8cm_gops
        );

        speedups.push(speedup);

        if !scalar_pass {
            println!("    \u{2718} scalar parity FAILED vs CPU int64 reference");
            // Find first mismatch to help debugging.
            let first = scalar_out
                .iter()
                .zip(cpu_ref.iter())
                .enumerate()
                .find(|(_, (g, e))| g != e);
            if let Some((idx, (g, e))) = first {
                let mi = idx as u32 / n;
                let ni = idx as u32 % n;
                println!("      first mismatch at C[{mi}][{ni}]: got={g}, expected={e}");
            }
            all_pass = false;
        }
        if !int8cm_pass {
            println!("    \u{2718} int8-cm parity FAILED vs CPU int64 reference");
            let first = int8cm_out
                .iter()
                .zip(cpu_ref.iter())
                .enumerate()
                .find(|(_, (g, e))| g != e);
            if let Some((idx, (g, e))) = first {
                let mi = idx as u32 / n;
                let ni = idx as u32 % n;
                println!("      first mismatch at C[{mi}][{ni}]: got={g}, expected={e}");
            }
            all_pass = false;
        }
        if scalar_pass && int8cm_pass && !bit_identical {
            println!(
                "    \u{26a0} both pass vs CPU but scalar != int8-cm bytewise (should be impossible)"
            );
        }

        drop(a_buf);
        drop(b_buf);
        drop(c_buf);
    }

    println!();
    println!("Parity: {}", if all_pass { "ALL PASS (CPU int64 reference)" } else { "FAILURES — see above" });
    let median_speedup = {
        let mut s = speedups.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    println!("Median speedup across {} shapes: {:.2}×", speedups.len(), median_speedup);
    println!();
    if median_speedup >= 1.7 {
        println!("\u{2714} STRONG GO ({median_speedup:.2}× ≥ 1.7×) — Sprint 11G-C is high-confidence.");
    } else if median_speedup >= 1.3 {
        println!("\u{2714} GO ({median_speedup:.2}× ≥ 1.3×) — within Sprint 11G-A analytical band.");
    } else if median_speedup >= 1.1 {
        println!(
            "\u{26a0} CONDITIONAL ({median_speedup:.2}×) — only marginal win, review effort/value."
        );
    } else if median_speedup >= 1.0 {
        println!(
            "\u{2718} NO-GO ({median_speedup:.2}×) — coopmat ties or barely beats scalar dot4."
        );
    } else {
        println!(
            "\u{2718} REGRESSION ({median_speedup:.2}× < 1×) — coopmat is slower than scalar dot4."
        );
    }

    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    shader: ShaderId,
    a: vk::Buffer,
    b: vk::Buffer,
    c: vk::Buffer,
    m: u32,
    n: u32,
    k: u32,
    n_reps: u32,
    reps: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let kernel = registry.get(shader);

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
        vk::DescriptorBufferInfo { buffer: a, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: c, offset: 0, range: vk::WHOLE_SIZE },
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

    let pc = BenchInt8CmGemmPushConstants { m, n, k, n_reps };
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;
    let groups_x = m / 16; // BM=16
    let groups_y = n / 16; // BN=16
    assert!(m % 16 == 0 && n % 16 == 0, "shape must be 16-aligned for the bench");

    let dispatch = |cmd: vk::CommandBuffer| unsafe {
        dev.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        dev.device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(c)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&post),
            &[],
        );
    };

    // Warmup.
    for _ in 0..3 {
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| dispatch(cmd))?;
    }

    let mut times_us: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let t0 = Instant::now();
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| dispatch(cmd))?;
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[times_us.len() / 2];

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };

    Ok(median_us)
}
