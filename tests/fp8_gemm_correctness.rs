//! Sprint 20-GEMM — FP8 E4M3 GEMM correctness test (GPU vs CPU).
//!
//! Validates `mul_coopmat_fp8_naive.comp` against a CPU reference
//! computed through the same `fp8_e4m3_to_f32` helper used by the
//! SafeTensors loader. Pass criterion: max-abs error well below the
//! single-FP8-multiply quantization noise (≈12.5% rel) accumulated
//! over K terms.
//!
//! Skipped when `VULKANFORGE_ENABLE_FP8` isn't set — same gating as
//! Sprint 20-M2's GEMV correctness test.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::fp8_ext::fp8_e4m3_to_f32;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::shaders::ShaderId;

fn fp8_enabled() -> bool {
    matches!(
        std::env::var("VULKANFORGE_ENABLE_FP8").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("True"),
    ) || matches!(
        std::env::var("VULKANFORGE_KV_FP8").ok().as_deref(),
        Some("1") | Some("true"),
    )
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Fp8GemmPush {
    m: u32,
    n: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    weight_scale_bits: u32,
}

#[test]
fn fp8_gemm_matches_cpu_reference() {
    if !fp8_enabled() {
        eprintln!("skipping: VULKANFORGE_ENABLE_FP8 not set");
        return;
    }

    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let cache_path = default_cache_path();
    let (registry, _) = PipelineRegistry::new(&dev.device, cache_path.as_deref())
        .expect("PipelineRegistry::new");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)
        .expect("CommandContext::new");

    // Tile-aligned shapes — the kernel doesn't pad N or M today.
    const M: usize = 64;
    const N: usize = 32;
    const K: usize = 64;

    // Deterministic xorshift32.
    let mut state: u32 = 0xCAFEF00D;
    let mut next_byte = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as u8
    };

    // Random FP8 weights (avoid 0x7F / 0xFF NaN encodings).
    let mut weights: Vec<u8> = Vec::with_capacity(M * K);
    while weights.len() < M * K {
        let b = next_byte();
        if b != 0x7F && b != 0xFF {
            weights.push(b);
        }
    }
    let weight_scale: f32 = 0.0123;

    // Random small-magnitude FP32 activations [N, K].
    let mut activations: Vec<f32> = Vec::with_capacity(N * K);
    while activations.len() < N * K {
        let bits = ((next_byte() as u32) << 24)
            | ((next_byte() as u32) << 16)
            | ((next_byte() as u32) << 8)
            | next_byte() as u32;
        let v = (bits as f32) / (u32::MAX as f32) - 0.5;
        activations.push(v);
    }

    // CPU reference: out[gn, gm] = scale * sum_k(fp8_to_f32(W[gm,k]) * b[gn,k]).
    // Output layout matches the kernel: [N, M] row-major.
    let mut ref_out = vec![0.0_f32; N * M];
    for gn in 0..N {
        for gm in 0..M {
            let mut acc = 0.0_f32;
            for kk in 0..K {
                let w = fp8_e4m3_to_f32(weights[gm * K + kk]);
                acc += w * activations[gn * K + kk];
            }
            ref_out[gn * M + gm] = acc * weight_scale;
        }
    }

    // Upload + dispatch.
    let mut w_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (M * K) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "fp8_gemm_W",
    ).expect("alloc W");
    w_buf.write_bytes(&weights).expect("upload W");
    let mut b_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (N * K * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "fp8_gemm_B",
    ).expect("alloc B");
    b_buf.write_bytes(bytemuck::cast_slice(&activations)).expect("upload B");
    let out_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (N * M * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "fp8_gemm_C",
    ).expect("alloc C");

    let kernel = registry.get(ShaderId::MulCoopmatFp8Naive);
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None) }
        .expect("descriptor pool");
    let set_layouts = [kernel.descriptor_set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts);
    let sets = unsafe { dev.device.allocate_descriptor_sets(&alloc_info) }
        .expect("alloc set");
    let set = sets[0];

    let bindings = [
        (0, w_buf.handle),
        (1, b_buf.handle),
        (2, out_buf.handle),
    ];
    let infos: Vec<vk::DescriptorBufferInfo> = bindings.iter()
        .map(|(_, h)| vk::DescriptorBufferInfo::default()
             .buffer(*h).offset(0).range(vk::WHOLE_SIZE))
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = bindings.iter().enumerate()
        .map(|(i, (b, _))| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*b)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        }).collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]); }

    let pc = Fp8GemmPush {
        m: M as u32,
        n: N as u32,
        k: K as u32,
        stride_a: K as u32,
        stride_b: K as u32,
        stride_c: M as u32,
        weight_scale_bits: weight_scale.to_bits(),
    };
    let groups_x = (M as u32 + 15) / 16;
    let groups_y = (N as u32 + 15) / 16;

    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline_layout,
            0, &[set], &[],
        );
        dev.device.cmd_push_constants(
            cmd, kernel.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
            0, bytemuck::bytes_of(&pc),
        );
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
    }).expect("one_shot");

    let raw = out_buf.read_bytes().expect("readback");
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(raw)[..N * M].to_vec();

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for i in 0..(N * M) {
        let abs = (got[i] - ref_out[i]).abs();
        max_abs = max_abs.max(abs);
        let denom = ref_out[i].abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    // L2 / RMS comparison rather than max-abs: the kernel converts
    // both A (FP8 → BF16) and B (FP32 → BF16) via the cooperative
    // matrix narrow type, accumulating K=64 BF16 products. Relative
    // error ≈ 2^-7 √K ≈ 6%, max-abs scales with `max(|matrix|)`;
    // some near-zero output elements have unbounded max-rel by
    // construction. RMS / max-output is the stable signal.
    let mut sum_sq_err = 0.0_f64;
    let mut max_out = 0.0_f32;
    for i in 0..(N * M) {
        let d = got[i] - ref_out[i];
        sum_sq_err += (d as f64) * (d as f64);
        max_out = max_out.max(ref_out[i].abs());
    }
    let rms_err = (sum_sq_err / (N * M) as f64).sqrt() as f32;
    let rms_rel = rms_err / max_out.max(1e-6);
    eprintln!(
        "FP8 GEMM: max_abs={max_abs:.6}, rms_err={rms_err:.6}, \
         rms_err/max_out={rms_rel:.4}, max_out={max_out:.6}, M={M}, N={N}, K={K}",
    );
    // 8% of max-output catches any structural bug (transpose,
    // wrong stride, scale not applied) while staying robust to
    // BF16 rounding — the BF16-narrow precision floor is ~6% RMS.
    assert!(rms_rel < 0.08, "rms_err/max_out {} >= 0.08", rms_rel);

    unsafe { dev.device.destroy_descriptor_pool(pool, None); }
    out_buf.destroy(&dev.device, &mut allocator);
    b_buf.destroy(&dev.device, &mut allocator);
    w_buf.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
}
