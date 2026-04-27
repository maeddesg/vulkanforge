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

const SHADER_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bench_coopmat_pure_f32.spv"));

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
    println!("Phase 6A bench_coopmat — {}", name.to_string_lossy());

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

    let extensions: [*const i8; 3] = [
        khr::cooperative_matrix::NAME.as_ptr(),
        c"VK_KHR_shader_bfloat16".as_ptr(),
        c"VK_KHR_shader_subgroup_uniform_control_flow".as_ptr(),
    ];

    // Feature chain: PhysicalDeviceFeatures2 → 1.1 (16-bit storage),
    // 1.2 (8-bit shader int, etc.), 1.3, plus coopmat + bfloat16.
    // ash 0.38 doesn't ship a builder for VK_KHR_shader_bfloat16 — the
    // extension was ratified after the bundled spec rev — so we splice
    // its feature struct into the chain manually before
    // `create_device`.
    let mut feat_coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
        .cooperative_matrix(true);
    let mut feat_bf16 = PhysicalDeviceShaderBfloat16FeaturesKHR {
        s_type: VK_STRUCT_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        p_next: std::ptr::null_mut(),
        shader_bfloat16_type: vk::TRUE,
        shader_bfloat16_dot_product: vk::FALSE,
        shader_bfloat16_cooperative_matrix: vk::TRUE,
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

    // Splice the BF16 feature struct between feat_coopmat and whatever
    // ash chained next: feat_coopmat → feat_bf16 → (rest). Done here
    // before push_next swallows feat_coopmat's p_next.
    feat_bf16.p_next = feat_coopmat.p_next;
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
    let words: Vec<u32> = SHADER_SPV
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

    // Inputs are BF16 — 2 bytes each.
    let a_bytes = (m as u64) * (k as u64) * 2;
    let b_bytes = (k as u64) * (n as u64) * 2;
    let c_bytes = (m as u64) * (n as u64) * 4;

    // Deterministic BF16 values via a small CPU pattern. We just need
    // *some* numerically valid input — the benchmark grades throughput,
    // not precision.
    let a_data: Vec<u16> = (0..(m * k))
        .map(|i| f32_to_bf16(0.001 * ((i % 64) as f32 - 32.0)))
        .collect();
    let b_data: Vec<u16> = (0..(k * n))
        .map(|i| f32_to_bf16(0.001 * (((i / 17) % 32) as f32)))
        .collect();

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
    buf_a.write_bytes(bytemuck::cast_slice(&a_data))?;
    buf_b.write_bytes(bytemuck::cast_slice(&b_data))?;

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
        eprintln!("⚠ {dim}^3: C[0] = {first_val:?} (non-finite — coopmat unhappy?)",
                  dim = m);
    }

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

// ---- Manual struct for VK_KHR_shader_bfloat16 (ash 0.38 doesn't ship it) ----

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
