//! Sprint 2A correctness + microbench for the isolated Q4_K dequant
//! debug shader. Three SPVs compile from the same source with different
//! output element types: FP32 (no convert), BF16 (5-VALU-op narrowing
//! sequence on RDNA4), FP8 E4M3 (one v_cvt_pk_fp8_f32 per pair, in the
//! native ISA — that's the smoking-gun this file is meant to confirm).
//!
//! Each test allocates a small Q4_K weight blob, runs the GPU shader,
//! reads back the output as FP32, and compares element-by-element to
//! the CPU `dequant_block` reference. Tolerances widen with output type
//! quantisation: FP32 is bit-exact, BF16 within ~1e-2, FP8 within ~0.15.

#![allow(clippy::too_many_arguments)]

use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ash::{khr, vk};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use half::f16;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::q4k::{BLOCK_BYTES, QUANT_K, build_random_weights, dequant_block};

const SHADER_SPV_FP32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4k_fp32.spv"));
const SHADER_SPV_BF16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4k_bf16.spv"));
const SHADER_SPV_FP8: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4k_fp8.spv"));

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConsts {
    n_blocks: u32,
}

#[derive(Clone, Copy, Debug)]
enum OutKind {
    Fp32,
    Bf16,
    Fp8,
}

impl OutKind {
    fn spv(self) -> &'static [u8] {
        match self {
            OutKind::Fp32 => SHADER_SPV_FP32,
            OutKind::Bf16 => SHADER_SPV_BF16,
            OutKind::Fp8 => SHADER_SPV_FP8,
        }
    }
    fn out_bytes_per_elem(self) -> u64 {
        match self {
            OutKind::Fp32 => 4,
            OutKind::Bf16 => 2,
            OutKind::Fp8 => 1,
        }
    }
    fn label(self) -> &'static str {
        match self {
            OutKind::Fp32 => "FP32",
            OutKind::Bf16 => "BF16",
            OutKind::Fp8 => "FP8 E4M3",
        }
    }
}

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
const VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: vk::StructureType =
    vk::StructureType::from_raw(1000567000);

struct Harness {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    physical_device: vk::PhysicalDevice,
    /// Vulkan requires external synchronisation of vkQueueSubmit on a
    /// given queue. Parallel `cargo test` runs would otherwise race
    /// inside libdrm — guard the dispatch path with a global mutex.
    queue_lock: Mutex<()>,
}

unsafe impl Send for Harness {}
unsafe impl Sync for Harness {}

fn harness() -> &'static Harness {
    static H: OnceLock<Harness> = OnceLock::new();
    H.get_or_init(|| build_harness().expect("Vulkan harness setup"))
}

fn build_harness() -> Result<Harness, Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"VulkanForge dequant_q4k tests")
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )?
    };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = physical_devices
        .iter()
        .copied()
        .find(|&pd| {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap_or(physical_devices[0]);

    let qfp = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let queue_family_index = qfp
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or("no compute queue")? as u32;

    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let extensions: [*const i8; 3] = [
        khr::cooperative_matrix::NAME.as_ptr(),
        c"VK_KHR_shader_bfloat16".as_ptr(),
        c"VK_EXT_shader_float8".as_ptr(),
    ];

    let mut feat_coopmat =
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
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
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let _name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };

    Ok(Harness {
        _entry: entry,
        instance,
        device,
        queue,
        queue_family_index,
        physical_device,
        queue_lock: Mutex::new(()),
    })
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Software FP8-E4M3 → FP32 dequant. NaN bit pattern (0x7F / 0xFF) is
/// the only special case that doesn't have a corresponding finite value.
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as u32;
    if exp == 15 && mant == 7 {
        return f32::NAN;
    }
    let s = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        if mant == 0 {
            return 0.0 * s;
        }
        return s * (mant as f32) / 8.0 / 64.0;
    }
    let m = 1.0 + (mant as f32) / 8.0;
    s * m * (2.0f32).powi(exp - 7)
}

/// Run the dequant shader on `weights` and read the GPU output back as f32.
/// Returns the dispatch median in milliseconds (for microbench callers) and
/// the converted-to-f32 output buffer.
fn run_dequant(
    h: &Harness,
    kind: OutKind,
    weights: &[u8],
    samples: usize,
) -> Result<(Vec<f32>, f64), Box<dyn std::error::Error>> {
    assert_eq!(weights.len() % BLOCK_BYTES, 0);
    let n_blocks = weights.len() / BLOCK_BYTES;
    let n_elems = n_blocks * QUANT_K;

    let device = &h.device;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: h.instance.clone(),
        device: device.clone(),
        physical_device: h.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let cmd_ctx = CommandContext::new(device, h.queue_family_index)?;

    // ---- Pipeline ----
    let words: Vec<u32> = kind
        .spv()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?
    };
    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<PushConsts>() as u32)];
    let pl = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&dsl))
                .push_constant_ranges(&push_range),
            None,
        )?
    };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let pipelines = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pl)],
                None,
            )
            .map_err(|(_, e)| e)?
    };
    let pipeline = pipelines[0];

    // ---- Buffers ----
    let in_bytes = weights.len() as u64;
    let out_bytes = (n_elems as u64) * kind.out_bytes_per_elem();
    let mut buf_in = GpuBuffer::new(
        device,
        &mut allocator,
        in_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "dq_in",
    )?;
    let buf_out = GpuBuffer::new(
        device,
        &mut allocator,
        out_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
        "dq_out",
    )?;
    buf_in.write_bytes(weights)?;

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 2,
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
            buffer: buf_in.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_out.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
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
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let pc = PushConsts { n_blocks: n_blocks as u32 };

    let dispatch_once = |label: &str| -> f64 {
        let _guard = h.queue_lock.lock().unwrap();
        let t0 = Instant::now();
        cmd_ctx.one_shot(device, h.queue, |cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pl,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(cmd, n_blocks as u32, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buf_out.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&post),
                &[],
            );
        })
        .expect(label);
        t0.elapsed().as_secs_f64() * 1000.0
    };

    let _ = dispatch_once("warmup");
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        times.push(dispatch_once("timed"));
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = times[times.len() / 2];

    // ---- Read back ----
    let raw = buf_out.read_bytes()?;
    let out_f32: Vec<f32> = match kind {
        OutKind::Fp32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        OutKind::Bf16 => raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        OutKind::Fp8 => raw.iter().map(|&b| fp8_e4m3_to_f32(b)).collect(),
    };

    unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pl, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(module, None);
    }
    buf_in.destroy(device, &mut allocator);
    buf_out.destroy(device, &mut allocator);
    cmd_ctx.destroy(device);
    drop(allocator);

    Ok((out_f32, median_ms))
}

fn cpu_dequant_all(weights: &[u8]) -> Vec<f32> {
    let n_blocks = weights.len() / BLOCK_BYTES;
    let mut out = vec![0.0f32; n_blocks * QUANT_K];
    for b in 0..n_blocks {
        let block: &[u8; BLOCK_BYTES] = (&weights[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES])
            .try_into()
            .unwrap();
        let dq = dequant_block(block);
        out[b * QUANT_K..(b + 1) * QUANT_K].copy_from_slice(&dq);
    }
    out
}

fn max_abs_err(gpu: &[f32], cpu: &[f32]) -> f32 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max)
}

fn check(kind: OutKind, n_blocks: usize, seed: u64, tol: f32) {
    let h = harness();
    let weights = build_random_weights(n_blocks, QUANT_K, seed);
    let (gpu, _ms) = run_dequant(h, kind, &weights, 1).expect("dequant run");
    let cpu = cpu_dequant_all(&weights);
    let err = max_abs_err(&gpu, &cpu);
    eprintln!(
        "dequant {:?}: blocks={n_blocks} seed={seed} max_abs_err={err:.4e} (tol {tol:.4e})",
        kind
    );
    assert!(
        err < tol,
        "dequant {:?} blocks={n_blocks}: max_abs_err {err} > tol {tol}",
        kind
    );
}

#[test]
fn dequant_q4k_fp32_single_block() {
    check(OutKind::Fp32, 1, 0xA5A5_A5A5, 1e-6);
}

#[test]
fn dequant_q4k_fp32_multi_block() {
    check(OutKind::Fp32, 100, 0x1234_5678, 1e-6);
}

#[test]
fn dequant_q4k_bf16_single_block() {
    // BF16 has 7 mantissa bits (~0.4% relative precision). Q4_K dequant
    // for our random fixture produces values up to ~10 magnitude, so
    // an absolute tolerance of 5e-2 covers the worst-case per-element
    // rounding plus the f16 dm conversion roundoff.
    check(OutKind::Bf16, 1, 0xA5A5_A5A5, 1e-1);
}

#[test]
fn dequant_q4k_bf16_multi_block() {
    check(OutKind::Bf16, 100, 0x1234_5678, 1e-1);
}

#[test]
fn dequant_q4k_fp8_single_block() {
    // FP8 E4M3 has 3 mantissa bits (~12.5% relative precision). Same
    // ~10-magnitude fixture → ~1.3 absolute worst-case. Set tol to 1.5
    // for headroom against the rare nibble that lands exactly on a
    // grid boundary.
    check(OutKind::Fp8, 1, 0xA5A5_A5A5, 1.5);
}

#[test]
fn dequant_q4k_fp8_multi_block() {
    check(OutKind::Fp8, 100, 0x1234_5678, 1.5);
}

/// Sanity: GPU FP32 output bit-equals the CPU `dequant_block` reference.
/// The Sprint 1A bf16 / fp8 GEMMs derive their precision floor from
/// this — if FP32-out is broken everything downstream is wrong.
#[test]
fn dequant_q4k_fp32_bit_exact() {
    let h = harness();
    let weights = build_random_weights(64, QUANT_K, 0xCAFEBABE);
    let (gpu, _) = run_dequant(h, OutKind::Fp32, &weights, 1).expect("fp32 run");
    let cpu = cpu_dequant_all(&weights);
    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            c.to_bits(),
            "element {i} bit mismatch: gpu={g} ({:#x}) cpu={c} ({:#x})",
            g.to_bits(),
            c.to_bits()
        );
    }
}

// ---- Microbench (reads the env var VF_BENCH_DEQUANT=1 to opt in;
// otherwise the test ignores its body to keep `cargo test` fast and
// silent on dispatch numbers) -----------------------------------------------

#[test]
fn dequant_q4k_microbench() {
    if std::env::var("VF_BENCH_DEQUANT").ok().as_deref() != Some("1") {
        eprintln!("dequant_q4k_microbench: set VF_BENCH_DEQUANT=1 to enable");
        return;
    }
    let h = harness();
    // ~1M weights = 4096 blocks — large enough to swamp launch overhead,
    // small enough that the buffers fit in a few MB.
    let n_blocks = 4096;
    let weights = build_random_weights(n_blocks, QUANT_K, 0xDEADBEEF);
    let n_elems = n_blocks * QUANT_K;

    println!(
        "\n{:<14} {:>11} {:>14} {:>14}",
        "variant", "median_ms", "GW/s", "out_GiB/s"
    );
    println!("{}", "-".repeat(58));
    for kind in [OutKind::Fp32, OutKind::Bf16, OutKind::Fp8] {
        let (_, ms) = run_dequant(h, kind, &weights, 11).expect("bench run");
        let secs = ms / 1000.0;
        let gw_per_s = (n_elems as f64) / secs / 1e9;
        let out_gib_per_s =
            (n_elems as f64 * kind.out_bytes_per_elem() as f64) / secs / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{:<14} {:>11.3} {:>14.2} {:>14.2}",
            kind.label(),
            ms,
            gw_per_s,
            out_gib_per_s
        );
    }
}

/// Statistics: keep the import alive for `f16` even if no test uses it
/// directly — readers searching for "f16" in this file should find a
/// single canonical line.
#[allow(dead_code)]
fn _keep_f16_import() {
    let _ = f16::from_f32(0.0);
}
