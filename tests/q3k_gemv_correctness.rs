//! Sprint 17B follow-up — Q3_K GPU GEMV correctness check.
//!
//! Hypothesis #1 (CPU dequant) was ruled out by
//! q3k_dequant_sanity.rs: q3k::dequant_block matches Q4_K-requant
//! values to within expected RMS noise.
//!
//! This test isolates the *GPU* GEMV shader (mul_mat_vec_q3_k.comp).
//! Strategy: take real Q3_K blocks from the Q3_K_M GGUF, run
//! MulMatVecQ3K on the GPU, and compare to a CPU GEMV computed via
//! q3k::dequant_block + scalar dot product. If they disagree, the
//! GPU shader is the bug.

use std::path::PathBuf;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::pipeline::{MatVecPushConstants, PUSH_CONSTANT_BYTES};
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::q3k;
use vulkanforge::backend::vulkan::shaders::ShaderId;

fn home_models() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME unset");
    PathBuf::from(home).join("models")
}

/// Run MulMatVecQ3K against MulMatVecQ3KSubgroup against CPU. Each row
/// is one Q3_K block (256 weights). Input is all 1.0 → expected
/// output[r] = Σ dequant_block(row_r). Mismatch on either GPU variant
/// vs CPU pinpoints the GPU shader as the bug.
fn run_gpu_vs_cpu(shader: ShaderId, label: &str) {
    let p = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    if !p.exists() {
        eprintln!("skip {label} — {} not found", p.display());
        return;
    }
    let gguf = GgufFile::open(&p).expect("open gguf");
    let info = gguf
        .tensor("token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(info.ggml_type, GgmlType::Q3K);
    let bytes = gguf.tensor_bytes(info);

    // M=4 rows × K=4096 cols (= 16 blocks per row). Mirrors a real
    // Qwen3-8B layer GEMV (hidden=4096), exercising the
    // `all_threads=true` main-loop path (16 blocks > it_size=4).
    const M: usize = 4;
    const K: usize = 4096;
    let blocks_per_row = K / q3k::QUANT_K;
    let bytes_per_row = blocks_per_row * q3k::BLOCK_BYTES;

    // CPU reference — dequant + scalar dot with all-ones input.
    let mut cpu_out = [0.0f32; M];
    let weights_bytes: Vec<u8> = bytes[..M * bytes_per_row].to_vec();
    let input: Vec<f32> = vec![1.0; K];
    for r in 0..M {
        for b in 0..blocks_per_row {
            let off = r * bytes_per_row + b * q3k::BLOCK_BYTES;
            let block: &[u8; q3k::BLOCK_BYTES] = weights_bytes
                [off..off + q3k::BLOCK_BYTES]
                .try_into()
                .unwrap();
            let dq = q3k::dequant_block(block);
            for k in 0..q3k::QUANT_K {
                cpu_out[r] += dq[k] * input[b * q3k::QUANT_K + k];
            }
        }
    }
    eprintln!("[{label}] CPU expected: {:?}", cpu_out);

    // GPU dispatch.
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

    let (registry, _) =
        PipelineRegistry::new(&dev.device, None).expect("PipelineRegistry::new");
    let kernel = registry.get(shader);

    let input_bytes_slice: &[u8] = bytemuck::cast_slice(&input);
    let weights_size = weights_bytes.len() as u64;
    let input_size = input_bytes_slice.len() as u64;
    let output_size = (M * std::mem::size_of::<f32>()) as u64;
    let dummy_size: u64 = 16;

    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

    let weights_buf = GpuBuffer::new(
        &dev.device, &mut allocator, weights_size, storage_dst,
        MemoryLocation::GpuOnly, "weights",
    ).unwrap();
    let input_buf = GpuBuffer::new(
        &dev.device, &mut allocator, input_size, storage_dst,
        MemoryLocation::GpuOnly, "input",
    ).unwrap();
    let output_buf = GpuBuffer::new(
        &dev.device, &mut allocator, output_size, storage_only,
        MemoryLocation::GpuToCpu, "output",
    ).unwrap();
    let fuse0 = GpuBuffer::new(
        &dev.device, &mut allocator, dummy_size, storage_only,
        MemoryLocation::GpuOnly, "fuse0",
    ).unwrap();
    let fuse1 = GpuBuffer::new(
        &dev.device, &mut allocator, dummy_size, storage_only,
        MemoryLocation::GpuOnly, "fuse1",
    ).unwrap();

    let mut staging_w = GpuBuffer::new(
        &dev.device, &mut allocator, weights_size, staging_src,
        MemoryLocation::CpuToGpu, "staging_w",
    ).unwrap();
    let mut staging_i = GpuBuffer::new(
        &dev.device, &mut allocator, input_size, staging_src,
        MemoryLocation::CpuToGpu, "staging_i",
    ).unwrap();
    staging_w.write_bytes(&weights_bytes).unwrap();
    staging_i.write_bytes(input_bytes_slice).unwrap();

    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).unwrap();
    cmd_ctx
        .one_shot(&dev.device, dev.compute_queue, |cmd| {
            let copy_w = vk::BufferCopy::default().size(weights_size);
            let copy_i = vk::BufferCopy::default().size(input_size);
            unsafe {
                dev.device.cmd_copy_buffer(
                    cmd, staging_w.handle, weights_buf.handle,
                    std::slice::from_ref(&copy_w),
                );
                dev.device.cmd_copy_buffer(
                    cmd, staging_i.handle, input_buf.handle,
                    std::slice::from_ref(&copy_i),
                );
            }
        })
        .unwrap();

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1).pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];

    let infos = [
        vk::DescriptorBufferInfo {
            buffer: weights_buf.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: input_buf.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: output_buf.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: fuse0.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: fuse1.handle, offset: 0, range: vk::WHOLE_SIZE,
        },
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
        broadcast3: 1,
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
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &pre, &[],
            );
            dev.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline,
            );
            dev.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout, 0, &[set], &[],
            );
            dev.device.cmd_push_constants(
                cmd, kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
            );
            // M workgroups, one row each.
            dev.device.cmd_dispatch(cmd, M as u32, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output_buf.handle).offset(0).size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[], &[post], &[],
            );
        })
        .unwrap();

    let output_bytes = output_buf.read_bytes().unwrap();
    let mut gpu_out = [0.0f32; M];
    for r in 0..M {
        gpu_out[r] = f32::from_le_bytes(
            output_bytes[r * 4..r * 4 + 4].try_into().unwrap(),
        );
    }
    eprintln!("[{label}] GPU got     : {:?}", gpu_out);

    unsafe { dev.device.destroy_descriptor_pool(pool, None) };

    // Tolerance: dequant produces values up to ~0.2; row sum up to ~50.
    // 1e-2 relative is generous but should catch anything wrong by >1%.
    for r in 0..M {
        let diff = (gpu_out[r] - cpu_out[r]).abs();
        let scale = cpu_out[r].abs().max(1e-3);
        let rel = diff / scale;
        eprintln!("[{label}] row {r}: cpu={:.6} gpu={:.6} rel={:.4}",
            cpu_out[r], gpu_out[r], rel);
        assert!(
            rel < 1e-2,
            "[{label}] row {r}: GPU={} CPU={} rel-error={}",
            gpu_out[r], cpu_out[r], rel,
        );
    }
}

#[test]
fn q3k_gemv_stock_matches_cpu() {
    run_gpu_vs_cpu(ShaderId::MulMatVecQ3K, "stock");
}

#[test]
fn q3k_gemv_subgroup_matches_cpu() {
    run_gpu_vs_cpu(ShaderId::MulMatVecQ3KSubgroup, "subgroup");
}
