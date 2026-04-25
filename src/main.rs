//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 1 / Step 1.5: scaling test. Runs three configurations of
//! the Q4_K GEMV pipeline:
//!
//!   - smoke   (M=2,    K=256)   — deterministic baseline; bit-exact
//!                                  CPU parity (regression check vs 1.4)
//!   - decode  (M=1,    K=3584)  — realistic single-token decode
//!   - stress  (M=3584, K=3584)  — quadratic GEMV, RDNA4 stress test
//!
//! For each config: random Q4_K weights + random input → 100×
//! dispatch in a single submit, with memory barriers between
//! dispatches; per-dispatch kernel time captured via timestamp
//! query pool; statistics + effective bandwidth + smoke checks
//! against the CPU reference.

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

/// Peak HBM/GDDR bandwidth for the RX 9070 XT, used for percent-of-peak
/// reporting. The vision doc cites 608 GB/s — ROCm-side benchmarks
/// have measured slightly higher (~640 GB/s) but 608 is the conservative
/// number we align with.
const PEAK_BW_GB_S: f64 = 608.0;

const SMOKE_ABS_THRESHOLD: f32 = 1e-2;
const SMOKE_REL_THRESHOLD: f32 = 0.05; // 5%

/// What test data to feed a config.
#[derive(Clone, Copy)]
enum DataKind {
    /// Bit-exact analytical case (1.4 baseline). Expects max_abs_err = 0.
    Smoke,
    /// Random Q4_K weights + random input. Expects rel/abs within
    /// the smoke thresholds (Q4_K + fp32 accumulation noise).
    Random {
        weights_seed: u64,
        input_seed: u64,
    },
}

struct BenchConfig {
    name: &'static str,
    m: usize,
    k: usize,
    iterations: usize,
    data: DataKind,
}

const CONFIGS: &[BenchConfig] = &[
    BenchConfig {
        name: "smoke (M=2, K=256)",
        m: 2,
        k: 256,
        iterations: 1,
        data: DataKind::Smoke,
    },
    BenchConfig {
        name: "decode (M=1, K=3584) — realistic single-token GEMV",
        m: 1,
        k: 3584,
        iterations: 100,
        data: DataKind::Random {
            weights_seed: 0xD110_DECD,
            input_seed: 0x42_42_42_42,
        },
    },
    BenchConfig {
        name: "stress (M=3584, K=3584) — quadratic GEMV",
        m: 3584,
        k: 3584,
        iterations: 100,
        data: DataKind::Random {
            weights_seed: 0x57_5E_55_57,
            input_seed: 0xBE_EF_CA_FE,
        },
    },
];

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
    println!("  Peak BW reference: {:.1} GB/s", PEAK_BW_GB_S);

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    println!("✅ gpu_allocator initialized");

    let spv = shaders::spv_words(shaders::MUL_MAT_VEC_Q4_K_F32_F32);
    let kernel = ComputeKernel::new(&dev.device, &spv, SpecConstants::SMOKE_DEFAULT)?;
    println!(
        "✅ Q4_K GEMV pipeline (BLOCK_SIZE={}, NUM_ROWS={}, NUM_COLS={})",
        SpecConstants::SMOKE_DEFAULT.block_size,
        SpecConstants::SMOKE_DEFAULT.num_rows,
        SpecConstants::SMOKE_DEFAULT.num_cols,
    );
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    for cfg in CONFIGS {
        run_config(
            &dev,
            &mut allocator,
            &kernel,
            &cmd_ctx,
            timestamp_period,
            cfg,
        )?;
    }

    cmd_ctx.destroy(&dev.device);
    kernel.destroy(&dev.device);
    drop(allocator);
    println!("\n✅ Phase-1 PoC complete — teardown clean");
    Ok(())
}

fn run_config(
    dev: &VulkanDevice,
    allocator: &mut Allocator,
    kernel: &ComputeKernel,
    cmd_ctx: &CommandContext,
    timestamp_period: f32,
    cfg: &BenchConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", cfg.name);
    println!(
        "  iterations: {}, blocks/row: {}, weights MB: {:.3}",
        cfg.iterations,
        cfg.k / q4k::QUANT_K,
        (cfg.m * (cfg.k / q4k::QUANT_K) * q4k::BLOCK_BYTES) as f64 / (1024.0 * 1024.0),
    );

    let (weights_bytes, input_vec) = match cfg.data {
        DataKind::Smoke => (q4k::build_smoke_weights(), vec![1.0f32; cfg.k]),
        DataKind::Random {
            weights_seed,
            input_seed,
        } => (
            q4k::build_random_weights(cfg.m, cfg.k, weights_seed),
            q4k::build_random_input(cfg.k, input_seed, 0.1),
        ),
    };
    let input_bytes_slice: &[u8] = bytemuck::cast_slice(&input_vec);

    let weights_size = weights_bytes.len() as u64;
    let input_size = input_bytes_slice.len() as u64;
    let output_size = (cfg.m * std::mem::size_of::<f32>()) as u64;
    let dummy_size: u64 = 16;

    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

    let weights_buf = GpuBuffer::new(
        &dev.device,
        allocator,
        weights_size,
        storage_dst,
        MemoryLocation::GpuOnly,
        "weights",
    )?;
    let input_buf = GpuBuffer::new(
        &dev.device,
        allocator,
        input_size,
        storage_dst,
        MemoryLocation::GpuOnly,
        "input",
    )?;
    let output_buf = GpuBuffer::new(
        &dev.device,
        allocator,
        output_size,
        storage_only,
        MemoryLocation::GpuToCpu,
        "output",
    )?;
    let fuse0_buf = GpuBuffer::new(
        &dev.device,
        allocator,
        dummy_size,
        storage_only,
        MemoryLocation::GpuOnly,
        "fuse0_dummy",
    )?;
    let fuse1_buf = GpuBuffer::new(
        &dev.device,
        allocator,
        dummy_size,
        storage_only,
        MemoryLocation::GpuOnly,
        "fuse1_dummy",
    )?;
    let mut staging_weights = GpuBuffer::new(
        &dev.device,
        allocator,
        weights_size,
        staging_src,
        MemoryLocation::CpuToGpu,
        "staging_weights",
    )?;
    let mut staging_input = GpuBuffer::new(
        &dev.device,
        allocator,
        input_size,
        staging_src,
        MemoryLocation::CpuToGpu,
        "staging_input",
    )?;
    staging_weights.write_bytes(&weights_bytes)?;
    staging_input.write_bytes(input_bytes_slice)?;

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
    let writes: [vk::WriteDescriptorSet; 5] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos[i..i + 1])
    });
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let qp_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2 * cfg.iterations as u32);
    let query_pool = unsafe { dev.device.create_query_pool(&qp_info, None)? };

    let pc = MatVecPushConstants {
        ncols: cfg.k as u32,
        stride_a: cfg.k as u32,
        stride_b: cfg.k as u32,
        stride_d: cfg.m as u32,
        batch_stride_a: (cfg.k * cfg.m) as u32,
        batch_stride_b: cfg.k as u32,
        batch_stride_d: cfg.m as u32,
        fusion_flags: 0,
        base_work_group_y: 0,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };
    let pc_bytes: &[u8] = bytemuck::bytes_of(&pc);
    assert_eq!(pc_bytes.len(), PUSH_CONSTANT_BYTES as usize);

    // Dispatch X = ceil(M / NUM_ROWS) with NUM_ROWS=1 from
    // SMOKE_DEFAULT spec constants. Y = batch (1 here), Z = 1 unless
    // M exceeds maxComputeWorkGroupCount[0] (the AMD limit is 65535,
    // M=3584 fits).
    let group_x = cfg.m as u32; // NUM_ROWS=1

    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device
            .cmd_reset_query_pool(cmd, query_pool, 0, 2 * cfg.iterations as u32);

        let pre_barriers = [
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
            &pre_barriers,
            &[],
        );

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

        // Per-dispatch barrier: every iteration reads weights+input
        // and writes output. The repeat reads of weights/input are
        // safe (RAR has no hazard); the WAW on output requires a
        // SHADER_WRITE -> SHADER_WRITE/READ barrier so the driver
        // serialises the dispatches and we measure them sequentially.
        let inter_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ);

        for i in 0..cfg.iterations as u32 {
            if i > 0 {
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[inter_barrier],
                    &[],
                    &[],
                );
            }
            dev.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                query_pool,
                2 * i,
            );
            dev.device.cmd_dispatch(cmd, group_x, 1, 1);
            dev.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                2 * i + 1,
            );
        }

        let post_barrier = vk::BufferMemoryBarrier::default()
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
            &[post_barrier],
            &[],
        );
    })?;

    // From here on, errors must not short-circuit the function — we
    // need to free the descriptor/query pools and buffers regardless,
    // or VkDestroyDevice trips the validation layer at exit. Wrap the
    // verification + reporting in a closure, run it, then unconditionally
    // tear everything down.
    let outcome: Result<(), Box<dyn std::error::Error>> = (|| {
        let mut timestamps = vec![0u64; 2 * cfg.iterations];
        unsafe {
            dev.device.get_query_pool_results(
                query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }

            let kernel_us: Vec<f64> = (0..cfg.iterations)
            .map(|i| {
                let ticks = timestamps[2 * i + 1].wrapping_sub(timestamps[2 * i]);
                ticks as f64 * timestamp_period as f64 / 1000.0
            })
            .collect();

        let stats = Stats::compute(&kernel_us);

        let bytes_per_dispatch = weights_size + input_size;
        let bw_gbps = if stats.median > 0.0 {
            (bytes_per_dispatch as f64 / 1e9) / (stats.median / 1e6)
        } else {
            0.0
        };
        let peak_pct = bw_gbps / PEAK_BW_GB_S * 100.0;

        println!(
            "  Kernel time (µs):  min={:.3}  median={:.3}  p95={:.3}  max={:.3}  stddev={:.3}",
            stats.min, stats.median, stats.p95, stats.max, stats.stddev,
        );
        println!(
            "  Bandwidth (median): {:.1} GB/s = {:.1}% of peak {:.0} GB/s  (per-dispatch reads {} B)",
            bw_gbps, peak_pct, PEAK_BW_GB_S, bytes_per_dispatch
        );

        let cpu_ref = q4k::cpu_gemv(&weights_bytes, cfg.m, cfg.k, &input_vec);
        let output_bytes = output_buf.read_bytes()?;
        let gpu_out: Vec<f32> = output_bytes[..output_size as usize]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        if gpu_out.iter().all(|&v| v == 0.0) {
            return Err(format!("[{}] smoke check: GPU output is all-zeros", cfg.name).into());
        }
        if gpu_out.iter().any(|&v| !v.is_finite()) {
            return Err(format!("[{}] smoke check: GPU output contains NaN/Inf", cfg.name).into());
        }

        let max_abs_err = gpu_out
            .iter()
            .zip(cpu_ref.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        // Adaptive relative error: skip near-zero CPU values (where rel
        // error blows up arbitrarily) and use the absolute error there.
        let max_rel_err = gpu_out
            .iter()
            .zip(cpu_ref.iter())
            .map(|(g, c)| {
                if c.abs() < 1e-3 {
                    (g - c).abs()
                } else {
                    (g - c).abs() / c.abs()
                }
            })
            .fold(0.0f32, f32::max);

        let preview_n = gpu_out.len().min(8);
        println!(
            "  GPU first {}: {:?}",
            preview_n,
            &gpu_out[..preview_n]
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
        );
        println!(
            "  CPU first {}: {:?}",
            preview_n,
            &cpu_ref[..preview_n]
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
        );
        println!(
            "  max_abs_err = {:.6e}, max_rel_err = {:.6e}",
            max_abs_err, max_rel_err
        );

        let (abs_thresh, rel_thresh) = match cfg.data {
            DataKind::Smoke => (0.0f32, 0.0f32),
            DataKind::Random { .. } => (SMOKE_ABS_THRESHOLD, SMOKE_REL_THRESHOLD),
        };
        if max_abs_err > abs_thresh {
            return Err(format!(
                "[{}] max_abs_err {:.6e} exceeds threshold {:.0e}",
                cfg.name, max_abs_err, abs_thresh
            )
            .into());
        }
        if max_rel_err > rel_thresh {
            return Err(format!(
                "[{}] max_rel_err {:.6e} exceeds threshold {:.0e}",
                cfg.name, max_rel_err, rel_thresh
            )
            .into());
        }
        println!("  ✅ smoke checks passed");

        if matches!(cfg.data, DataKind::Random { .. }) && peak_pct < 30.0 {
            eprintln!(
                "  ⚠️  bandwidth utilisation {:.1}% < 30% — likely fundamental issue (spec-constants? dispatch shape?)",
                peak_pct
            );
        }
        Ok(())
    })();

    // Cleanup runs unconditionally so VkDestroyDevice has no orphans
    // even when verification fails partway through.
    unsafe {
        dev.device.destroy_query_pool(query_pool, None);
        dev.device.destroy_descriptor_pool(descriptor_pool, None);
    }
    staging_input.destroy(&dev.device, allocator);
    staging_weights.destroy(&dev.device, allocator);
    fuse1_buf.destroy(&dev.device, allocator);
    fuse0_buf.destroy(&dev.device, allocator);
    output_buf.destroy(&dev.device, allocator);
    input_buf.destroy(&dev.device, allocator);
    weights_buf.destroy(&dev.device, allocator);

    outcome
}

struct Stats {
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
    stddev: f64,
}

impl Stats {
    fn compute(samples: &[f64]) -> Self {
        let n = samples.len();
        assert!(n > 0);
        let mut sorted: Vec<f64> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let p95_idx = ((n as f64) * 0.95).floor() as usize;
        let p95_idx = p95_idx.min(n - 1);
        Self {
            min: sorted[0],
            median: sorted[n / 2],
            p95: sorted[p95_idx],
            max: sorted[n - 1],
            stddev,
        }
    }
}
