//! Sprint 2B correctness + microbench for the Q4_K dequant-fusion
//! coopmat GEMM kernel (`vk_shaders/mul_coopmat_q4k.comp`). Three SPVs
//! cover BN ∈ {16, 32, 64} and stay structurally identical to the
//! sprint-1B `mul_coopmat_fp8_*.comp` pipelines — only the A load
//! phase changes.
//!
//! Inputs: A as Q4_K weights (`block_q4_K[]`, 144 B/256 weights), B as
//! FP32 activations row-major. Output: FP32 row-major. The CPU
//! reference dequantises the same Q4_K via `q4k::dequant_block` and
//! GEMMs in f64 — the GPU output is allowed to drift by FP8's ~12.5 %
//! quantisation grid plus K-dependent FP32 accumulation noise.

#![allow(clippy::too_many_arguments)]

use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};

use ash::{khr, vk};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::q4k::{
    BLOCK_BYTES, QUANT_K, build_random_input, build_random_weights, dequant_block,
};

const SPV_BN64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_bn64.spv"));
const SPV_BN32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_bn32.spv"));
const SPV_BN16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_coopmat_q4k_bn16.spv"));

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

#[derive(Clone, Copy, Debug)]
enum Bn {
    Bn64,
    Bn32,
    Bn16,
}

impl Bn {
    fn spv(self) -> &'static [u8] {
        match self {
            Bn::Bn64 => SPV_BN64,
            Bn::Bn32 => SPV_BN32,
            Bn::Bn16 => SPV_BN16,
        }
    }
    fn tile_n(self) -> u32 {
        match self {
            Bn::Bn64 => 64,
            Bn::Bn32 => 32,
            Bn::Bn16 => 16,
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
        .application_name(c"VulkanForge coopmat_q4k tests")
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

fn run_q4k_gemm(
    h: &Harness,
    bn: Bn,
    weights_q4k: &[u8],
    acts_f32: &[f32],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    assert!(
        m % 16 == 0 && n % 16 == 0,
        "WMMA requires M and N multiples of 16"
    );
    assert_eq!((k as usize) % QUANT_K, 0, "K must be a multiple of QUANT_K");
    assert_eq!(
        weights_q4k.len(),
        (m as usize) * ((k as usize) / QUANT_K) * BLOCK_BYTES
    );
    assert_eq!(acts_f32.len(), (k as usize) * (n as usize));

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
    let words: Vec<u32> = bn
        .spv()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3u32)
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
    let a_bytes = weights_q4k.len() as u64;
    let b_bytes = (acts_f32.len() as u64) * 4;
    let c_bytes = (m as u64) * (n as u64) * 4;
    let mut buf_a = GpuBuffer::new(
        device,
        &mut allocator,
        a_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "q4k_a",
    )?;
    let mut buf_b = GpuBuffer::new(
        device,
        &mut allocator,
        b_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "q4k_b",
    )?;
    let buf_c = GpuBuffer::new(
        device,
        &mut allocator,
        c_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
        "q4k_c",
    )?;
    buf_a.write_bytes(weights_q4k)?;
    buf_b.write_bytes(bytemuck::cast_slice(acts_f32))?;

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
            buffer: buf_a.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_b.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_c.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
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
        m,
        n,
        k,
        stride_a: k,
        stride_b: n,
        stride_c: n,
    };
    let groups_x = m.div_ceil(64);
    let groups_y = n.div_ceil(bn.tile_n());

    {
        let _g = h.queue_lock.lock().unwrap();
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
            device.cmd_dispatch(cmd, groups_x, groups_y, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buf_c.handle)
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
        })?;
    }

    let raw = buf_c.read_bytes()?;
    let mut c = vec![0f32; (m * n) as usize];
    for (i, dst) in c.iter_mut().enumerate() {
        *dst = f32::from_le_bytes([
            raw[4 * i],
            raw[4 * i + 1],
            raw[4 * i + 2],
            raw[4 * i + 3],
        ]);
    }

    unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pl, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(module, None);
    }
    buf_a.destroy(device, &mut allocator);
    buf_b.destroy(device, &mut allocator);
    buf_c.destroy(device, &mut allocator);
    cmd_ctx.destroy(device);
    drop(allocator);

    Ok(c)
}

/// CPU reference: dequantise A row-by-row, then GEMM in f64 against B.
/// B is row-major K×N like the GPU expects; output is row-major M×N.
fn cpu_q4k_gemm_ref(weights_q4k: &[u8], acts: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let bpr = k / QUANT_K;
    let mut dq_rows = vec![0f32; m * k];
    for r in 0..m {
        for b in 0..bpr {
            let off = (r * bpr + b) * BLOCK_BYTES;
            let block: &[u8; BLOCK_BYTES] = (&weights_q4k[off..off + BLOCK_BYTES])
                .try_into()
                .unwrap();
            let dq = dequant_block(block);
            dq_rows[r * k + b * QUANT_K..r * k + (b + 1) * QUANT_K].copy_from_slice(&dq);
        }
    }
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for e in 0..k {
                acc += (dq_rows[r * k + e] as f64) * (acts[e * n + j] as f64);
            }
            out[r * n + j] = acc as f32;
        }
    }
    out
}

fn max_abs_err(gpu: &[f32], cpu: &[f32]) -> f32 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max)
}

fn check_q4k(bn: Bn, m: u32, n: u32, k: u32, w_seed: u64, a_seed: u64, tol: f32) {
    let h = harness();
    let weights = build_random_weights(m as usize, k as usize, w_seed);
    let acts = build_random_input((k * n) as usize, a_seed, 1.0);
    let gpu = run_q4k_gemm(h, bn, &weights, &acts, m, n, k).expect("q4k gemm");
    let cpu = cpu_q4k_gemm_ref(&weights, &acts, m as usize, n as usize, k as usize);
    let err = max_abs_err(&gpu, &cpu);
    eprintln!(
        "q4k_fused {:?}: M={m} N={n} K={k} max_abs_err={err:.4e} (tol {tol:.4e})",
        bn
    );
    assert!(
        err < tol,
        "q4k_fused {:?} M={m} N={n} K={k}: max_abs_err {err} > tol {tol}",
        bn
    );
}

// Tolerances reflect *double* FP8 quantisation: A is Q4_K → FP32 →
// FP8, B is FP32 → FP8. With weights up to ~10 magnitude and acts in
// [-1,1], FP8's ~12% relative precision plus K-step accumulation
// drift puts realistic max_abs_err around `0.6 * sqrt(K) * scale`,
// where scale ≈ typical product magnitude. The numbers below were
// observed on this fixture and have ~3x headroom over the worst
// per-element drift seen across seeds.
#[test]
fn q4k_coopmat_bn64_m64_n64_k256() {
    check_q4k(Bn::Bn64, 64, 64, 256, 0xA5A5_A5A5, 0x5A5A_5A5A, 5.0);
}

#[test]
fn q4k_coopmat_bn64_m64_n64_k1024() {
    check_q4k(Bn::Bn64, 64, 64, 1024, 0xA5A5_A5A5, 0x5A5A_5A5A, 10.0);
}

#[test]
fn q4k_coopmat_bn64_m64_n64_k4096() {
    check_q4k(Bn::Bn64, 64, 64, 4096, 0xA5A5_A5A5, 0x5A5A_5A5A, 20.0);
}

#[test]
fn q4k_coopmat_bn64_m128_n128_k4096() {
    check_q4k(Bn::Bn64, 128, 128, 4096, 0x1234_5678, 0x9876_5432, 30.0);
}

#[test]
fn q4k_coopmat_bn16_m64_n16_k1024() {
    check_q4k(Bn::Bn16, 64, 16, 1024, 0xCAFEBABE, 0xDEADBEEF, 10.0);
}

#[test]
fn q4k_coopmat_bn32_m64_n32_k1024() {
    check_q4k(Bn::Bn32, 64, 32, 1024, 0xCAFEBABE, 0xDEADBEEF, 10.0);
}

#[test]
fn q4k_coopmat_prefill_2048_64_4096() {
    // Real prefill Q-projection shape, BN=16 picks max N-parallelism.
    check_q4k(Bn::Bn16, 2048, 64, 4096, 0xC001_D00D, 0xBEEF_FACE, 30.0);
}

#[test]
fn q4k_coopmat_bn16_matches_bn64() {
    // BN=16 vs BN=64 over the same Q4_K weights and FP32 activations
    // — both kernels accumulate in FP32 with BK=16 K-walks; per-WMMA
    // tile partitioning is the only thing that differs, so the
    // outputs should agree within FP32 noise.
    let h = harness();
    let (m, n, k) = (256u32, 64u32, 1024u32);
    let weights = build_random_weights(m as usize, k as usize, 7);
    let acts = build_random_input((k * n) as usize, 13, 1.0);
    let g16 = run_q4k_gemm(h, Bn::Bn16, &weights, &acts, m, n, k).expect("bn16");
    let g64 = run_q4k_gemm(h, Bn::Bn64, &weights, &acts, m, n, k).expect("bn64");
    let err = max_abs_err(&g16, &g64);
    eprintln!("q4k bn16-vs-bn64 max_abs_err = {err:.4e}");
    // Identical operands + identical accumulation order → numerically
    // identical down to FP32 ULPs.
    assert!(err < 1e-3, "bn16 vs bn64 divergence too large: {err}");
}

// ---- argmax-parity vs the existing scalar mul_mmq path (Phase 7).
// Skipped by default because invoking the runtime Vulkan stack here
// would pull in the full pipeline registry — that wiring is Sprint 3.
// See `tests/correctness.rs::test_gemm_q4k_*` for argmax-style checks
// against the coopmat output once Sprint 3 lands the registry slot.

// ---- Throughput microbench. Opt in with VF_BENCH_Q4K=1 to print
// median ms / TFLOPS for the 7 default shapes plus a K-scan. Picks a
// per-shape BN (BN=16 for N≤32, BN=32 for N≤64, BN=64 otherwise) and
// times Q4_K-fused dispatches.

fn pick_bn(n: u32) -> Bn {
    if n <= 32 {
        Bn::Bn16
    } else if n <= 64 {
        Bn::Bn32
    } else {
        Bn::Bn64
    }
}

/// Setup-once / dispatch-many bench harness. `run_q4k_gemm` rebuilds
/// the pipeline and buffers every call which would dominate the timing
/// for the larger square shapes; this version pays setup once and
/// times `samples` dispatches.
fn time_q4k_dispatch(
    h: &Harness,
    bn: Bn,
    weights: &[u8],
    acts: &[f32],
    m: u32,
    n: u32,
    k: u32,
    samples: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
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

    let words: Vec<u32> = bn
        .spv()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3u32)
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

    let a_bytes = weights.len() as u64;
    let b_bytes = (acts.len() as u64) * 4;
    let c_bytes = (m as u64) * (n as u64) * 4;
    let mut buf_a = GpuBuffer::new(
        device,
        &mut allocator,
        a_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "q4k_bench_a",
    )?;
    let mut buf_b = GpuBuffer::new(
        device,
        &mut allocator,
        b_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "q4k_bench_b",
    )?;
    let buf_c = GpuBuffer::new(
        device,
        &mut allocator,
        c_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
        "q4k_bench_c",
    )?;
    buf_a.write_bytes(weights)?;
    buf_b.write_bytes(bytemuck::cast_slice(acts))?;

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
            buffer: buf_a.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_b.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: buf_c.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
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
        m,
        n,
        k,
        stride_a: k,
        stride_b: n,
        stride_c: n,
    };
    let groups_x = m.div_ceil(64);
    let groups_y = n.div_ceil(bn.tile_n());

    let dispatch_once = |label: &str| -> f64 {
        let _g = h.queue_lock.lock().unwrap();
        let t0 = std::time::Instant::now();
        cmd_ctx
            .one_shot(device, h.queue, |cmd| unsafe {
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
                device.cmd_dispatch(cmd, groups_x, groups_y, 1);
                let post = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buf_c.handle)
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

    let _warm = dispatch_once("warmup");
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        times.push(dispatch_once("timed"));
    }

    unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pl, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(module, None);
    }
    buf_a.destroy(device, &mut allocator);
    buf_b.destroy(device, &mut allocator);
    buf_c.destroy(device, &mut allocator);
    cmd_ctx.destroy(device);
    drop(allocator);

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

#[test]
fn q4k_coopmat_microbench() {
    if std::env::var("VF_BENCH_Q4K").ok().as_deref() != Some("1") {
        eprintln!("q4k_coopmat_microbench: set VF_BENCH_Q4K=1 to enable");
        return;
    }
    let h = harness();
    let shapes = [
        (256u32, 256u32, 256u32),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (2048, 64, 4096),
        (11008, 64, 4096),
        (4096, 64, 11008),
        (4096, 128, 4096),
    ];

    println!(
        "\n{:<22} {:>11} {:>10} {:>11} {:>11}",
        "shape", "BN", "med_ms", "TFLOPS", "ratio_mmq"
    );
    println!("{}", "─".repeat(70));
    for &(m, n, k) in &shapes {
        let bn = pick_bn(n);
        let weights = build_random_weights(m as usize, k as usize, 42);
        let acts = build_random_input((k * n) as usize, 137, 1.0);
        let ms = time_q4k_dispatch(h, bn, &weights, &acts, m, n, k, 5).expect("bench");
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let tflops = flops / (ms / 1000.0) / 1.0e12;
        // Scalar mul_mmq baseline taken at ~1 TFLOPS effective from
        // Phase-7 v0.1.3 numbers — coarse but consistent.
        let mmq_ratio = tflops / 1.0;
        let label = if m == n && n == k {
            format!("{}^3", m)
        } else {
            format!("{}x{}x{}", m, n, k)
        };
        println!(
            "{:<22} {:>11} {:>10.3} {:>11.2} {:>10.2}×",
            label,
            format!("{:?}", bn),
            ms,
            tflops,
            mmq_ratio
        );
    }
}
