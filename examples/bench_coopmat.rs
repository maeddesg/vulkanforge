//! Phase 6A — pure-WMMA throughput benchmark.
//!
//! Builds a dedicated `vk::Device` with `VK_KHR_cooperative_matrix`,
//! `VK_KHR_shader_bfloat16`, and the matching feature chain enabled
//! (independent of the runtime `VulkanDevice` — this is a benchmark,
//! not a production code path), loads `bench_coopmat_pure_f32.spv`
//! built by `build.rs`, and runs `C = A · B` with
//!
//!   A : [M, K] BF16 row-major
//!   B : [K, N] BF16 row-major
//!   C : [M, N] FP32 row-major
//!
//! at three representative sizes and reports TFLOPS for each.
//!
//! TFLOPS = 2 · M · N · K / time, since each MAC is 2 floating-point
//! ops. Theoretical peak for RDNA4 / RX 9070 XT at BF16 is roughly
//! 200 TOPS; the AI-accelerator headline number from the spec sheet.
//!
//! Usage:  cargo run --release --example bench_coopmat

#![allow(clippy::too_many_arguments)]

use std::ffi::CStr;
use std::time::Instant;

use ash::{khr, vk};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;

const SHADER_SPV_BF16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bench_coopmat_pure_f32.spv"));
const SHADER_SPV_FP8: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bench_coopmat_fp8_e4m3.spv"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMode {
    Bf16,
    Fp8E4m3,
}

impl BenchMode {
    fn from_env() -> Self {
        match std::env::var("VF_BENCH_FP8").ok().as_deref() {
            Some("1") | Some("true") | Some("e4m3") => BenchMode::Fp8E4m3,
            _ => BenchMode::Bf16,
        }
    }

    fn label(self) -> &'static str {
        match self {
            BenchMode::Bf16 => "BF16",
            BenchMode::Fp8E4m3 => "FP8-E4M3",
        }
    }

    fn spv(self) -> &'static [u8] {
        match self {
            BenchMode::Bf16 => SHADER_SPV_BF16,
            BenchMode::Fp8E4m3 => SHADER_SPV_FP8,
        }
    }

    fn input_bytes_per_elem(self) -> u64 {
        match self {
            BenchMode::Bf16 => 2,
            BenchMode::Fp8E4m3 => 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConsts {
    m: u32,
    n: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = BenchMode::from_env();
    let entry = unsafe { ash::Entry::load()? };

    // ---- Instance ----
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"VulkanForge bench_coopmat")
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    // ---- Physical device ----
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = physical_devices
        .iter()
        .copied()
        .find(|&pd| {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap_or(physical_devices[0]);

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    println!(
        "bench_coopmat — {} ({})",
        name.to_string_lossy(),
        mode.label()
    );

    // ---- Queue family with COMPUTE ----
    let qfp = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let queue_family_index = qfp
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or("no compute queue")? as u32;

    // ---- Logical device with coopmat + BF16 enabled ----
    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // Always enable both BF16 and FP8 extensions — driver supports both
    // and the per-feature toggles below pick which capability the
    // pipeline actually uses.
    let extensions: [*const i8; 4] = [
        khr::cooperative_matrix::NAME.as_ptr(),
        c"VK_KHR_shader_bfloat16".as_ptr(),
        c"VK_EXT_shader_float8".as_ptr(),
        c"VK_KHR_shader_subgroup_uniform_control_flow".as_ptr(),
    ];

    // Feature chain: PhysicalDeviceFeatures2 → 1.1 (16-bit storage),
    // 1.2 (8-bit shader int, etc.), 1.3, plus coopmat + bfloat16 + fp8.
    // ash 0.38 doesn't ship builders for VK_KHR_shader_bfloat16 or
    // VK_EXT_shader_float8 — both extensions were ratified after the
    // bundled spec rev — so we splice their feature structs into the
    // chain manually before `create_device`.
    let mut feat_coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
        .cooperative_matrix(true);
    let mut feat_bf16 = PhysicalDeviceShaderBfloat16FeaturesKHR {
        s_type: VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        p_next: std::ptr::null_mut(),
        shader_bfloat16_type: vk::TRUE,
        shader_bfloat16_dot_product: vk::FALSE,
        shader_bfloat16_cooperative_matrix: vk::TRUE,
    };
    let mut feat_fp8 = PhysicalDeviceShaderFloat8FeaturesEXT {
        s_type: VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
        p_next: std::ptr::null_mut(),
        shader_float8: vk::TRUE,
        shader_float8_cooperative_matrix: vk::TRUE,
    };
    let mut feat13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true)
        .shader_int8(true);
    let mut feat11 = vk::PhysicalDeviceVulkan11Features::default()
        .storage_buffer16_bit_access(true)
        .uniform_and_storage_buffer16_bit_access(true);
    let core = vk::PhysicalDeviceFeatures::default().shader_int16(true);
    let mut feat2 = vk::PhysicalDeviceFeatures2::default().features(core);

    // Splice the BF16 + FP8 feature structs into feat_coopmat's p_next
    // chain: feat_coopmat → feat_bf16 → feat_fp8 → (rest). Done here
    // before push_next swallows feat_coopmat's p_next.
    feat_fp8.p_next = feat_coopmat.p_next;
    feat_bf16.p_next = &mut feat_fp8 as *mut _ as *mut std::ffi::c_void;
    feat_coopmat.p_next = &mut feat_bf16 as *mut _ as *mut std::ffi::c_void;

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&extensions)
        .push_next(&mut feat2)
        .push_next(&mut feat11)
        .push_next(&mut feat12)
        .push_next(&mut feat13)
        .push_next(&mut feat_coopmat);

    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    let physical_device_for_alloc = physical_device;

    // ---- Allocator + command context ----
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: physical_device_for_alloc,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let cmd_ctx = CommandContext::new(&device, queue_family_index)?;

    // ---- Pipeline / descriptor set layout from SPV ----
    let words: Vec<u32> = mode
        .spv()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let module = unsafe { device.create_shader_module(&module_info, None)? };

    // 3 storage buffers (A, B, C).
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let dsl = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<PushConsts>() as u32)];
    let pl_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&dsl))
        .push_constant_ranges(&push_range);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_info, None)? };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let cp_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[cp_info], None)
            .map_err(|(_, e)| e)?
    };
    let pipeline = pipelines[0];

    println!("\n{:<22} {:<14} {:>11} {:>10} {:>10} {:>11}",
        "size", "GFLOPs", "warmup_ms", "med_ms", "TFLOPS", "vs scalar*");
    println!("{}", "─".repeat(86));

    // Phase 6A original baseline (square cubes), plus v0.2 smoke-test
    // prefill-realistic shapes. Override via VF_BENCH_SHAPES =
    // "m1,n1,k1;m2,n2,k2;..." to bench arbitrary triples.
    let shapes: Vec<(u32, u32, u32)> = std::env::var("VF_BENCH_SHAPES")
        .ok()
        .and_then(|s| {
            s.split(';')
                .filter_map(|tri| {
                    let mut p = tri.split(',');
                    Some((
                        p.next()?.trim().parse().ok()?,
                        p.next()?.trim().parse().ok()?,
                        p.next()?.trim().parse().ok()?,
                    ))
                })
                .collect::<Vec<_>>()
                .into()
        })
        .unwrap_or_else(|| vec![
            // Phase 6A square cubes (kept for back-compat)
            (256, 256, 256),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            // v0.2 smoke-test prefill shapes (M = output, N = seq_len,
            // K = input). Width 64 mirrors a 64-token prefill batch.
            (2048,  64, 4096),    // gemm_q  @ pp=64
            (11008, 64, 4096),    // gemm_gate / gemm_up @ pp=64
            (4096,  64, 11008),   // gemm_down @ pp=64
            (4096, 128, 4096),    // gemm_q  @ pp=128 (more N parallelism)
        ]);

    for &(m, n, k) in &shapes {
        let res = run_size(
            mode,
            &device, &mut allocator, queue, &cmd_ctx,
            pipeline, pipeline_layout, dsl,
            m, n, k,
        )?;
        let label = if m == n && n == k {
            format!("{}^3", m)
        } else {
            format!("{}x{}x{}", m, n, k)
        };
        println!(
            "{:<22} {:<14.2} {:>11.2} {:>10.3} {:>10.2} {:>10.2}×",
            label,
            res.gflops,
            res.warmup_ms,
            res.median_ms,
            res.tflops,
            res.tflops / SCALAR_FMA_TFLOPS_BASELINE,
        );
    }

    println!(
        "\n*scalar baseline = {} TFLOPS f32 FMA (RX 9070 XT theoretical peak)",
        SCALAR_FMA_TFLOPS_BASELINE
    );

    // ---- Cleanup ----
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(module, None);
    }
    cmd_ctx.destroy(&device);
    drop(allocator);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

const SCALAR_FMA_TFLOPS_BASELINE: f64 = 25.0;

struct RunResult {
    gflops: f64,
    warmup_ms: f64,
    median_ms: f64,
    tflops: f64,
}

fn run_size(
    mode: BenchMode,
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    cmd_ctx: &CommandContext,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    m: u32, n: u32, k: u32,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    assert!(m % 16 == 0 && n % 16 == 0 && k % 16 == 0, "WMMA tile is 16×16×16");

    let elem = mode.input_bytes_per_elem();
    let a_bytes = (m as u64) * (k as u64) * elem;
    let b_bytes = (k as u64) * (n as u64) * elem;
    let c_bytes = (m as u64) * (n as u64) * 4;

    // Deterministic input values via a small CPU pattern. Throughput
    // bench, not precision — values just need to be numerically sane
    // (no NaN/Inf in inputs, no Inf in accumulator).
    let a_raw: Vec<u8> = match mode {
        BenchMode::Bf16 => bytemuck::cast_slice::<u16, u8>(
            &(0..(m * k))
                .map(|i| f32_to_bf16(0.001 * ((i % 64) as f32 - 32.0)))
                .collect::<Vec<u16>>()
        ).to_vec(),
        BenchMode::Fp8E4m3 => (0..(m * k))
            .map(|i| f32_to_fp8_e4m3(0.001 * ((i % 64) as f32 - 32.0)))
            .collect(),
    };
    let b_raw: Vec<u8> = match mode {
        BenchMode::Bf16 => bytemuck::cast_slice::<u16, u8>(
            &(0..(k * n))
                .map(|i| f32_to_bf16(0.001 * (((i / 17) % 32) as f32)))
                .collect::<Vec<u16>>()
        ).to_vec(),
        BenchMode::Fp8E4m3 => (0..(k * n))
            .map(|i| f32_to_fp8_e4m3(0.001 * (((i / 17) % 32) as f32)))
            .collect(),
    };

    let mut buf_a = GpuBuffer::new(
        device, allocator, a_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu, "bench_a",
    )?;
    let mut buf_b = GpuBuffer::new(
        device, allocator, b_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu, "bench_b",
    )?;
    let buf_c = GpuBuffer::new(
        device, allocator, c_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu, "bench_c",
    )?;
    buf_a.write_bytes(&a_raw)?;
    buf_b.write_bytes(&b_raw)?;

    // Descriptor pool + set.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    };
    let layouts = [dsl];
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )?[0]
    };

    let infos = [
        vk::DescriptorBufferInfo {
            buffer: buf_a.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_b.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_c.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
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
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let pc = PushConsts {
        m, n, k,
        stride_a: k,
        stride_b: n,
        stride_c: n,
    };
    let groups_x = m / 16;
    let groups_y = n / 16;

    let dispatch_once = |label: &str| -> f64 {
        let t0 = Instant::now();
        cmd_ctx.one_shot(device, queue, |cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[],
            );
            device.cmd_push_constants(
                cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(cmd, groups_x, groups_y, 1);
            // Make output visible to host for read-back of the last call.
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buf_c.handle)
                .offset(0).size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[], std::slice::from_ref(&post), &[],
            );
        }).expect(label);
        t0.elapsed().as_secs_f64() * 1000.0
    };

    let warmup_ms = dispatch_once("warmup");
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        samples.push(dispatch_once("timed"));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = samples[samples.len() / 2];

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = flops / (median_ms / 1000.0) / 1.0e12;
    let gflops = flops / 1.0e9;

    // Sanity: read back C[0] (first f32). NaN/Inf would tank the result.
    let bytes = buf_c.read_bytes()?;
    let first_val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if !first_val.is_finite() {
        eprintln!("⚠ {label}: C[0] = {first_val:?} (non-finite — coopmat unhappy?)",
                  label = format!("{}x{}x{}", m, n, k));
    }

    // Optional correctness probe — controlled by VF_BENCH_CHECK=1. Computes
    // a CPU f64 reference for C[0] using the same FP8/BF16 dequant as the
    // shader, prints rel-err side-by-side. Only runs once per shape, so
    // negligible cost for the bench.
    let abs_err = if std::env::var("VF_BENCH_CHECK").ok().as_deref() == Some("1") {
        let mut acc: f64 = 0.0;
        let stride_a = k as usize;
        let stride_b = n as usize;
        for kk in 0..(k as usize) {
            let a_idx = kk;
            let b_idx = kk * stride_b;
            let a = match mode {
                BenchMode::Bf16 => bf16_to_f32(u16::from_le_bytes([
                    a_raw[2 * a_idx], a_raw[2 * a_idx + 1],
                ])) as f64,
                BenchMode::Fp8E4m3 => fp8_e4m3_to_f32(a_raw[a_idx]) as f64,
            };
            let b = match mode {
                BenchMode::Bf16 => bf16_to_f32(u16::from_le_bytes([
                    b_raw[2 * b_idx], b_raw[2 * b_idx + 1],
                ])) as f64,
                BenchMode::Fp8E4m3 => fp8_e4m3_to_f32(b_raw[b_idx]) as f64,
            };
            acc += a * b;
        }
        let _ = stride_a;
        let cpu_ref = acc as f32;
        let err = (first_val - cpu_ref).abs();
        eprintln!(
            "  C[0]: gpu={first_val:>12.6} cpu_ref={cpu_ref:>12.6}  abs_err={err:.3e}"
        );
        Some(err)
    } else {
        None
    };
    let _ = abs_err;

    unsafe { device.destroy_descriptor_pool(pool, None) };
    buf_a.destroy(device, allocator);
    buf_b.destroy(device, allocator);
    buf_c.destroy(device, allocator);

    Ok(RunResult { gflops, warmup_ms, median_ms, tflops })
}

fn f32_to_bf16(x: f32) -> u16 {
    // Round-to-nearest-even bf16 packing.
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let bias = 0x7fff + lsb;
    ((bits.wrapping_add(bias)) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as u32;
    if exp == 15 && mant == 7 {
        // E4M3 NaN.
        return f32::NAN;
    }
    let s = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        if mant == 0 {
            return 0.0 * s;
        }
        // Subnormal: 2^-6 * (mant / 8)
        return s * (mant as f32) / 8.0 / 64.0;
    }
    // Normal: 2^(exp-7) * (1 + mant/8)
    let m = 1.0 + (mant as f32) / 8.0;
    s * m * (2.0f32).powi(exp - 7)
}

// FP8 E4M3 packing — 1 sign + 4 exponent + 3 mantissa, bias=7.
// E4M3 has only NaN (0x7F / 0xFF), no Inf. Range ±448. We saturate
// out-of-range to ±max_finite (0x7E / 0xFE) and return ±0 for
// underflow rather than emitting subnormals — bench data does not need
// subnormal precision and avoiding them keeps the conversion simple.
fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let f32_exp = ((bits >> 23) & 0xff) as i32;
    let f32_mant = bits & 0x7f_ffff;

    if f32_exp == 0 {
        return sign << 7;
    }
    let unbiased = f32_exp - 127;
    let new_exp = unbiased + 7;
    if new_exp >= 16 {
        // Saturate to max finite ±448.0.
        return (sign << 7) | 0x7E;
    }
    if new_exp <= 0 {
        return sign << 7;
    }
    // Round-to-nearest-even: keep 3 mantissa bits, look at the 4th for
    // rounding direction.
    let round_bit = (f32_mant >> 19) & 1;
    let sticky = f32_mant & 0x7_ffff;
    let mut mant3 = ((f32_mant >> 20) & 0x07) as u8;
    let mut exp4 = new_exp as u8;
    if round_bit == 1 && (sticky != 0 || (mant3 & 1) == 1) {
        mant3 = mant3.wrapping_add(1);
        if mant3 == 8 {
            mant3 = 0;
            exp4 = exp4.wrapping_add(1);
            if exp4 >= 16 {
                return (sign << 7) | 0x7E;
            }
        }
    }
    // Avoid emitting NaN bit pattern (e=15, m=7) — the saturate-to-7E
    // above means we should never reach (15, 7) here, but assert
    // defensively in debug.
    debug_assert!(!(exp4 == 15 && mant3 == 7), "would emit NaN");
    (sign << 7) | ((exp4 & 0x0f) << 3) | (mant3 & 0x07)
}

// ---- Manual structs for VK_KHR_shader_bfloat16 / VK_EXT_shader_float8
// (ash 0.38 doesn't ship builders for either) ----

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
struct PhysicalDeviceShaderBfloat16FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

const VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: vk::StructureType =
    vk::StructureType::from_raw(1000141000);

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
struct PhysicalDeviceShaderFloat8FeaturesEXT {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_float8: vk::Bool32,
    shader_float8_cooperative_matrix: vk::Bool32,
}

// VkStructureType per VK_EXT_shader_float8 spec (registry entry 567).
const VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: vk::StructureType =
    vk::StructureType::from_raw(1000567000);
