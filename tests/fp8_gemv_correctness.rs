//! Sprint 20-M2 — FP8 E4M3 GEMV correctness test (GPU vs CPU).
//!
//! Synthesises a random FP8 weight matrix + FP32 input vector + an
//! arbitrary per-tensor weight scale, dispatches the new
//! `mul_mat_vec_fp8.comp` shader, and compares the result against
//! a CPU FP32 reference computed with the same FP8→FP32 helper used
//! at load time (`fp8_ext::fp8_e4m3_to_f32`).
//!
//! Pass criterion: max-abs and max-relative error fall within a
//! tolerance large enough for FP8's intrinsic ~12.5% per-multiply
//! quantization error to wash out across `K` accumulations but
//! tight enough to catch a routing / endianness / scale-not-applied
//! regression.
//!
//! Skipped at runtime when FP8 isn't actually advertised by the
//! device (per `VULKANFORGE_ENABLE_FP8=1` opt-in or RDNA4 + Mesa
//! 26.0.6+).

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::fp8_ext::fp8_e4m3_to_f32;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
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

#[test]
fn fp8_gemv_matches_cpu_reference() {
    // Sprint 20-M2 lands the FP8 GEMV but routing into chat/decode is
    // sprint-20-M3. Until the user opts FP8 on at the device level the
    // test must skip — the shader's `GL_EXT_float_e4m3` extension would
    // otherwise fail to instantiate. CI without the env var → green
    // skip. Real bring-up: `VULKANFORGE_ENABLE_FP8=1 cargo test`.
    if !fp8_enabled() {
        eprintln!(
            "skipping: VULKANFORGE_ENABLE_FP8 / VULKANFORGE_KV_FP8 not set; \
             FP8 device feature not opted in."
        );
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

    // Synthesize: M = 64 output rows, K = 256 input cols (multiple of
    // 4 so the shader's K4 main loop handles it cleanly; M = 64 keeps
    // dispatches manageable in the test).
    const M: usize = 64;
    const K: usize = 256;

    // Deterministic PRNG (xorshift32) — keeps the test reproducible
    // without pulling in a `rand` dep.
    let mut state: u32 = 0xCAFEF00D;
    let mut next_byte = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as u8
    };

    // FP8 weight matrix as raw bytes. Avoid 0x7F / 0xFF (NaN encodings
    // for E4M3) so the CPU and GPU see well-defined values.
    let mut w_fp8: Vec<u8> = Vec::with_capacity(M * K);
    while w_fp8.len() < M * K {
        let b = next_byte();
        if b != 0x7F && b != 0xFF {
            w_fp8.push(b);
        }
    }
    let mut input: Vec<f32> = Vec::with_capacity(K);
    while input.len() < K {
        let bits = ((next_byte() as u32) << 24)
            | ((next_byte() as u32) << 16)
            | ((next_byte() as u32) << 8)
            | next_byte() as u32;
        // Map to a small range to avoid catastrophic accumulation.
        let v = (bits as f32) / (u32::MAX as f32) - 0.5;
        input.push(v);
    }
    let weight_scale: f32 = 0.0123_f32;

    // CPU reference: out[r] = scale * sum_k(fp8_to_f32(W[r,k]) * input[k]).
    let mut ref_out = vec![0.0_f32; M];
    for r in 0..M {
        let mut acc = 0.0_f32;
        for k in 0..K {
            let w = fp8_e4m3_to_f32(w_fp8[r * K + k]);
            acc += w * input[k];
        }
        ref_out[r] = acc * weight_scale;
    }

    // Upload to GPU.
    let mut w_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (M * K) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "fp8_gemv_W",
    ).expect("alloc W");
    w_buf.write_bytes(&w_fp8).expect("upload W");
    let mut in_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (K * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
        "fp8_gemv_in",
    ).expect("alloc in");
    in_buf.write_bytes(bytemuck::cast_slice(&input)).expect("upload in");
    let out_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (M * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "fp8_gemv_out",
    ).expect("alloc out");
    // Dummy fuse buffers (binding 3, 4) — never read.
    let fuse0 = GpuBuffer::new(
        &dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse0",
    ).expect("fuse0");
    let fuse1 = GpuBuffer::new(
        &dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse1",
    ).expect("fuse1");

    // Bind a fresh descriptor set for the dispatch. Tests don't
    // share Forward's set cache, so allocate a tiny pool here.
    let kernel = registry.get(ShaderId::MulMatVecFp8);
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(5)];
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
        (1, in_buf.handle),
        (2, out_buf.handle),
        (3, fuse0.handle),
        (4, fuse1.handle),
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

    // Dispatch.
    let pc = MatVecPushConstants {
        ncols: K as u32,
        stride_a: K as u32,
        stride_b: K as u32,
        stride_d: M as u32,
        batch_stride_a: (K * M) as u32,
        batch_stride_b: K as u32,
        batch_stride_d: M as u32,
        fusion_flags: 0,
        base_work_group_y: 0,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: weight_scale.to_bits(),
    };
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
        dev.device.cmd_dispatch(cmd, M as u32, 1, 1);
    }).expect("one_shot");

    // Read back. `read_bytes` returns a borrow into the GpuToCpu mapping;
    // copy out into our owned `got` so we can drop the buffer at end-of-test.
    let raw = out_buf.read_bytes().expect("readback");
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(raw)[..M].to_vec();

    // Compare. With K=256 random products of FP8 weights × FP32 inputs,
    // the per-element accumulation drift between CPU sequential vs
    // GPU tree-reduce is ~ K * eps_fp32 * |max-product| — well below
    // 1e-4 relative for our value range. A 1e-3 abs/rel cap catches
    // any actual routing bug while staying robust to ordering.
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for r in 0..M {
        let abs = (got[r] - ref_out[r]).abs();
        max_abs = max_abs.max(abs);
        let denom = ref_out[r].abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    eprintln!("FP8 GEMV: max_abs={max_abs:.6}, max_rel={max_rel:.6}, M={M}, K={K}");
    assert!(max_abs < 1e-3, "max_abs {} >= 1e-3", max_abs);
    assert!(max_rel < 1e-3, "max_rel {} >= 1e-3", max_rel);

    // Cleanup.
    unsafe { dev.device.destroy_descriptor_pool(pool, None); }
    out_buf.destroy(&dev.device, &mut allocator);
    in_buf.destroy(&dev.device, &mut allocator);
    w_buf.destroy(&dev.device, &mut allocator);
    fuse0.destroy(&dev.device, &mut allocator);
    fuse1.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
}
