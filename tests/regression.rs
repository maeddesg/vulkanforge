//! Phase-1 + Phase-2A regression tests. Run with `cargo test`.
//!
//! These talk to a real Vulkan device; each test creates its own
//! `VulkanDevice` so they don't share state and can be run in any
//! order (or in parallel).

use std::path::PathBuf;
use std::time::Duration;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::{MatVecPushConstants, PUSH_CONSTANT_BYTES};
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::q4k;
use vulkanforge::backend::vulkan::shaders::{self, ShaderId, ALL_SHADERS};
use vulkanforge::backend::vulkan::spirv_reflect;
use vulkanforge::backend::vulkan::vram_arena::{
    ArenaConfig, ArenaError, BufferViewError, VramArena,
};

const SPIRV_MAGIC: [u8; 4] = [0x03, 0x02, 0x23, 0x07];

#[test]
fn phase2a_all_shaders_compile_to_spirv() {
    // Cheap sanity: every embedded blob is a valid SPIR-V binary in
    // structure (magic number + 4-byte aligned). Catches a build.rs
    // regression that would otherwise only surface as a runtime
    // pipeline-create error.
    for &id in ALL_SHADERS {
        let bytes = id.spv_bytes();
        assert!(
            bytes.len() >= 5 * 4,
            "{:?}: SPIR-V too short ({})",
            id,
            bytes.len()
        );
        assert_eq!(
            &bytes[0..4],
            &SPIRV_MAGIC,
            "{:?}: SPIR-V magic missing",
            id
        );
        assert_eq!(bytes.len() % 4, 0, "{:?}: SPIR-V not 4-byte aligned", id);

        // Reflection must succeed and report at least one binding for
        // every shader in the inventory (none of them are "empty").
        let reflection = spirv_reflect::reflect(&shaders::spv_words(bytes));
        assert!(
            !reflection.bindings.is_empty(),
            "{:?}: reflection found 0 bindings",
            id
        );
    }
}

#[test]
fn phase2a_q4k_unit_tests_referenced() {
    // Phase-1 unit tests live in src/backend/vulkan/q4k.rs and run
    // automatically with `cargo test`. This integration test just
    // verifies the data-generation entry points still work and the
    // CPU GEMV produces the analytical answer — protecting the
    // pair-layout bug (Q4_K nibble-bug) from regressing.
    let weights = q4k::build_smoke_weights();
    let input: Vec<f32> = vec![1.0; q4k::QUANT_K];
    let out = q4k::cpu_gemv(&weights, 2, q4k::QUANT_K, &input);
    assert!((out[0] - 256.0).abs() < 1e-3);
    assert!((out[1] - 512.0).abs() < 1e-3);
}

#[test]
fn phase2a_pipeline_registry_creates_all() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let cache_path: Option<PathBuf> = None; // never touch user cache from tests
    let (registry, loaded) = PipelineRegistry::new(&dev.device, cache_path.as_deref())
        .expect("PipelineRegistry::new");
    assert_eq!(loaded, 0, "no cache path → no bytes loaded");
    assert_eq!(
        registry.count(),
        ALL_SHADERS.len(),
        "registry must hold one pipeline per ShaderId"
    );
    // Every ShaderId must be retrievable.
    for &id in ALL_SHADERS {
        let _ = registry.get(id);
    }
    // Sanity: cold-start pipeline creation should be fast (< 5 s).
    // This catches a hang in vkCreateComputePipelines.
    assert!(
        registry.create_duration < Duration::from_secs(5),
        "pipeline creation took {:?}",
        registry.create_duration
    );
    registry.destroy(&dev.device);
}

#[test]
fn phase2a_vram_arena_zones_and_pingpong() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let config = ArenaConfig {
        weights_bytes: 4 * 1024 * 1024,
        kv_cache_bytes: 2 * 1024 * 1024,
        scratch_bytes: 256 * 1024,
    };
    let arena = VramArena::new(&dev.instance, dev.physical_device, &dev.device, config)
        .expect("VramArena::new");
    let l = arena.layout;
    // No overlap, no gaps beyond the 4 KiB zone alignment.
    assert_eq!(l.weights.offset, 0);
    assert_eq!(l.kv_cache.offset, l.weights.size);
    assert_eq!(l.scratch.offset, l.weights.size + l.kv_cache.size);
    assert!(arena.total_bytes >= l.scratch.offset + l.scratch.size);
    assert!(l.weights.size >= config.weights_bytes);
    assert!(l.kv_cache.size >= config.kv_cache_bytes);
    assert!(l.scratch.size >= config.scratch_bytes);

    // Ping-pong: even/odd layers map to the two halves of scratch.
    let (off0, size0) = arena.scratch_for_layer(0);
    let (off1, size1) = arena.scratch_for_layer(1);
    let (off2, _) = arena.scratch_for_layer(2);
    assert_eq!(size0, size1);
    assert_eq!(size0, l.scratch.size / 2);
    assert_ne!(off0, off1, "ping-pong halves must differ");
    assert_eq!(off0, off2, "even layers reuse the same half");
    assert!(off0 + size0 <= l.scratch.offset + l.scratch.size);
    assert!(off1 + size1 <= l.scratch.offset + l.scratch.size);

    // Buffer views in each zone should bind cleanly.
    let weights_view = arena
        .create_buffer(
            &dev.device,
            l.weights.offset,
            128 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in weights zone");
    let kv_view = arena
        .create_buffer(
            &dev.device,
            l.kv_cache.offset,
            64 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in kv zone");
    let scratch_view = arena
        .create_buffer(
            &dev.device,
            off0,
            size0,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in scratch zone");

    // Out-of-bounds is rejected with a structured error.
    let oob = arena.create_buffer(
        &dev.device,
        arena.total_bytes - 16,
        128,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    );
    assert!(matches!(oob, Err(BufferViewError::OutOfBounds { .. })));

    unsafe {
        dev.device.destroy_buffer(weights_view, None);
        dev.device.destroy_buffer(kv_view, None);
        dev.device.destroy_buffer(scratch_view, None);
    }
    arena.destroy(&dev.device);
}

#[test]
fn phase2a_vram_arena_oom_clean_error() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    // Request more than maxMemoryAllocationSize ever could be.
    let huge = ArenaConfig {
        weights_bytes: u64::MAX / 2,
        kv_cache_bytes: u64::MAX / 4,
        scratch_bytes: 0,
    };
    let result = VramArena::new(&dev.instance, dev.physical_device, &dev.device, huge);
    assert!(
        matches!(result, Err(ArenaError::AllocationTooLarge { .. })),
        "expected AllocationTooLarge, got {:?}",
        result.as_ref().err()
    );
}

/// Phase-1 regression: the original 1.4 smoke test. Bit-exact output
/// from the Q4_K GEMV pipeline → [256.0, 512.0]. This is the test the
/// prompt's regression strategy explicitly calls out as `BEHALTEN`.
#[test]
fn phase1_q4k_smoke_dispatch_bit_exact() {
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

    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("PipelineRegistry::new");
    let kernel = registry.get(ShaderId::MulMatVecQ4K);

    const M: usize = 2;
    const K: usize = q4k::QUANT_K;

    let weights_bytes = q4k::build_smoke_weights();
    let input: Vec<f32> = vec![1.0; K];
    let input_bytes_slice: &[u8] = bytemuck::cast_slice(&input);

    let weights_size = weights_bytes.len() as u64;
    let input_size = input_bytes_slice.len() as u64;
    let output_size = (M * std::mem::size_of::<f32>()) as u64;
    let dummy_size: u64 = 16;

    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

    let weights_buf = GpuBuffer::new(&dev.device, &mut allocator, weights_size, storage_dst, MemoryLocation::GpuOnly, "weights").unwrap();
    let input_buf = GpuBuffer::new(&dev.device, &mut allocator, input_size, storage_dst, MemoryLocation::GpuOnly, "input").unwrap();
    let output_buf = GpuBuffer::new(&dev.device, &mut allocator, output_size, storage_only, MemoryLocation::GpuToCpu, "output").unwrap();
    let fuse0 = GpuBuffer::new(&dev.device, &mut allocator, dummy_size, storage_only, MemoryLocation::GpuOnly, "fuse0").unwrap();
    let fuse1 = GpuBuffer::new(&dev.device, &mut allocator, dummy_size, storage_only, MemoryLocation::GpuOnly, "fuse1").unwrap();

    let mut staging_w = GpuBuffer::new(&dev.device, &mut allocator, weights_size, staging_src, MemoryLocation::CpuToGpu, "staging_w").unwrap();
    let mut staging_i = GpuBuffer::new(&dev.device, &mut allocator, input_size, staging_src, MemoryLocation::CpuToGpu, "staging_i").unwrap();
    staging_w.write_bytes(&weights_bytes).unwrap();
    staging_i.write_bytes(input_bytes_slice).unwrap();

    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).unwrap();
    cmd_ctx
        .one_shot(&dev.device, dev.compute_queue, |cmd| {
            let copy_w = vk::BufferCopy::default().size(weights_size);
            let copy_i = vk::BufferCopy::default().size(input_size);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, staging_w.handle, weights_buf.handle, std::slice::from_ref(&copy_w));
                dev.device.cmd_copy_buffer(cmd, staging_i.handle, input_buf.handle, std::slice::from_ref(&copy_i));
            }
        })
        .unwrap();

    // Descriptor pool + set + writes.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];

    let infos = [
        vk::DescriptorBufferInfo { buffer: weights_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: input_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: output_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: fuse0.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: fuse1.handle, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes: [vk::WriteDescriptorSet; 5] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos[i..i + 1])
    });
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let pc = MatVecPushConstants {
        ncols: K as u32, stride_a: K as u32, stride_b: K as u32, stride_d: M as u32,
        batch_stride_a: (K * M) as u32, batch_stride_b: K as u32, batch_stride_d: M as u32,
        fusion_flags: 0, base_work_group_y: 0, ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1,
    };
    let pc_bytes: &[u8] = bytemuck::bytes_of(&pc);
    assert_eq!(pc_bytes.len(), PUSH_CONSTANT_BYTES as usize);

    cmd_ctx
        .one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(weights_buf.handle).offset(0).size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(input_buf.handle).offset(0).size(vk::WHOLE_SIZE),
            ];
            dev.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[], &pre, &[],
            );
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline_layout, 0, &[set], &[]);
            dev.device.cmd_push_constants(cmd, kernel.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            dev.device.cmd_dispatch(cmd, 2, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output_buf.handle).offset(0).size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[], &[post], &[],
            );
        })
        .unwrap();

    let output_bytes = output_buf.read_bytes().unwrap();
    let g0 = f32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
    let g1 = f32::from_le_bytes(output_bytes[4..8].try_into().unwrap());
    assert_eq!(g0, 256.0, "Phase-1 smoke regression: output[0]");
    assert_eq!(g1, 512.0, "Phase-1 smoke regression: output[1]");

    // Cleanup
    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    staging_i.destroy(&dev.device, &mut allocator);
    staging_w.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    fuse1.destroy(&dev.device, &mut allocator);
    fuse0.destroy(&dev.device, &mut allocator);
    output_buf.destroy(&dev.device, &mut allocator);
    input_buf.destroy(&dev.device, &mut allocator);
    weights_buf.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}
