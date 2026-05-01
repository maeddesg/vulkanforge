//! Phase 5B.1 diagnostic — measures the f32 max-abs-error of
//! `flash_attn_batch.comp` against an f64 CPU reference at the same
//! sweep of batch sizes the parity tests use, then times the GPU
//! dispatch at M=100 and M=500 as a "is this shader plausibly worth
//! integrating" smoke test.
//!
//! Not a benchmark in the proper sense — single-shader dispatch only,
//! no Q/K/V projection, no FFN. Phase 5B.2 will land the real
//! end-to-end prefill numbers.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::needless_question_mark)]

use std::time::Instant;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::FlashAttnBatchPushConstants;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::shaders::ShaderId;

const N_HEADS: u32 = 32;
const N_KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let cache_path = default_cache_path();
    let (registry, _) = PipelineRegistry::new(&dev.device, cache_path.as_deref())?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    println!("Phase 5B.1 — flash_attn_batch.comp parity sweep");
    println!(
        "  n_heads={} n_kv_heads={} head_dim={} (Qwen3 / Llama-3.1 / Mistral GQA 4:1)",
        N_HEADS, N_KV_HEADS, HEAD_DIM,
    );
    println!();
    println!("{:<28} {:>4} {:>10} {:>14} {:>10}", "case", "m", "q_start", "max_abs_err", "result");
    println!("{}", "─".repeat(72));

    let cases: &[(&str, u32, u32, f32)] = &[
        ("m1_vs_cpu",                       1,    0, 1e-4),
        ("m4_vs_cpu",                       4,    0, 1e-3),
        ("m16_vs_cpu",                     16,    0, 1e-3),
        ("m64_vs_cpu",                     64,    0, 1e-3),
        ("m200_vs_cpu",                   200,    0, 5e-3),
        ("q_start_offset (m=4, q_start=60)", 4,  60, 1e-3),
    ];

    let mut all_pass = true;
    for &(name, m, q_start, threshold) in cases {
        let err = run_parity(&dev, &mut allocator, &registry, &cmd_ctx, m, q_start)?;
        let ok = err < threshold && err.is_finite();
        if !ok {
            all_pass = false;
        }
        println!(
            "{:<28} {:>4} {:>10} {:>14.3e} {:>10}",
            name, m, q_start, err, if ok { "PASS" } else { "FAIL" }
        );
    }

    println!();
    println!("Phase 5B.1 — dispatch-time smoke (single shader, GPU-only)");
    println!("{:<10} {:>10} {:>16}", "case", "warmup_ms", "median_ms");
    println!("{}", "─".repeat(40));
    for &m in &[100u32, 500u32] {
        let (warm, med) = time_dispatch(&dev, &mut allocator, &registry, &cmd_ctx, m, 5)?;
        println!("M={:<8} {:>10.2} {:>16.3}", m, warm, med);
    }

    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    println!();
    println!("{}", if all_pass { "ALL PARITY OK" } else { "PARITY FAIL — see table above" });
    Ok(())
}

fn run_parity(
    dev: &VulkanDevice,
    allocator: &mut Allocator,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    m: u32,
    q_start: u32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let n_kv = q_start + m;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    let (q, k, v) = build_inputs(m, n_kv);
    let cpu = cpu_reference(&q, &k, &v, m, n_kv, q_start, scale);

    let q_buf = host_buf(dev, allocator, &q)?;
    let k_buf = host_buf(dev, allocator, &k)?;
    let v_buf = host_buf(dev, allocator, &v)?;
    let o_buf = output_buf(dev, allocator, q.len())?;

    dispatch_once(dev, registry, cmd_ctx, &[&q_buf, &k_buf, &v_buf, &o_buf], m, n_kv, q_start, scale);

    let bytes = o_buf.read_bytes()?;
    let gpu: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes[..q.len() * 4]).to_vec();

    q_buf.destroy(&dev.device, allocator);
    k_buf.destroy(&dev.device, allocator);
    v_buf.destroy(&dev.device, allocator);
    o_buf.destroy(&dev.device, allocator);

    Ok(gpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max))
}

fn time_dispatch(
    dev: &VulkanDevice,
    allocator: &mut Allocator,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    m: u32,
    repeats: u32,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let n_kv = m;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = build_inputs(m, n_kv);

    let q_buf = host_buf(dev, allocator, &q)?;
    let k_buf = host_buf(dev, allocator, &k)?;
    let v_buf = host_buf(dev, allocator, &v)?;
    let o_buf = output_buf(dev, allocator, q.len())?;
    let buffers = [&q_buf, &k_buf, &v_buf, &o_buf];

    // Warmup — first dispatch carries pipeline-create + cmd-pool init costs.
    let t0 = Instant::now();
    dispatch_once(dev, registry, cmd_ctx, &buffers, m, n_kv, 0, scale);
    let warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Timed: median of `repeats` invocations.
    let mut samples: Vec<f64> = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        dispatch_once(dev, registry, cmd_ctx, &buffers, m, n_kv, 0, scale);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[samples.len() / 2];

    q_buf.destroy(&dev.device, allocator);
    k_buf.destroy(&dev.device, allocator);
    v_buf.destroy(&dev.device, allocator);
    o_buf.destroy(&dev.device, allocator);
    Ok((warmup_ms, med))
}

fn dispatch_once(
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    buffers: &[&GpuBuffer],
    m: u32,
    n_kv: u32,
    q_start: u32,
    scale: f32,
) {
    let kernel = registry.get(ShaderId::FlashAttnBatch);
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: buffers.len() as u32,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None).unwrap() };
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc).unwrap() }[0];

    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|b| vk::DescriptorBufferInfo {
            buffer: b.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        })
        .collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let pc = FlashAttnBatchPushConstants {
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        m,
        n_kv,
        q_start,
        scale,
    };

    let device = dev.device.clone();
    let queue = dev.compute_queue;
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;
    let last_buf = buffers.last().unwrap().handle;
    cmd_ctx
        .one_shot(&device, queue, |cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(cmd, N_HEADS, m, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(last_buf)
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
        .expect("one_shot");
    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
}

fn build_inputs(m: u32, n_kv: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pos_stride = (N_KV_HEADS * HEAD_DIM) as usize;
    let kv_size = (n_kv as usize) * pos_stride;
    let q_count = (m * N_HEADS * HEAD_DIM) as usize;

    let mut q = vec![0.0f32; q_count];
    for q_idx in 0..(m as usize) {
        for h in 0..(N_HEADS as usize) {
            for d in 0..(HEAD_DIM as usize) {
                let off = (q_idx * N_HEADS as usize + h) * HEAD_DIM as usize + d;
                q[off] = ((q_idx as f32) * 0.017 + (h as f32) * 0.011 + (d as f32) * 0.003).sin();
            }
        }
    }
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];
    for t in 0..(n_kv as usize) {
        for kvh in 0..(N_KV_HEADS as usize) {
            for d in 0..(HEAD_DIM as usize) {
                let off = t * pos_stride + kvh * (HEAD_DIM as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + (d as f32) * 0.001 + (kvh as f32) * 0.7).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.5 + (d as f32) * 0.0007).sin();
            }
        }
    }
    (q, k, v)
}

fn cpu_reference(
    q: &[f32], k: &[f32], v: &[f32],
    m: u32, n_kv: u32, q_start: u32, scale: f32,
) -> Vec<f32> {
    let head_dim_us = HEAD_DIM as usize;
    let pos_stride = (N_KV_HEADS * HEAD_DIM) as usize;
    let group_size = N_HEADS / N_KV_HEADS;
    let scale_d = scale as f64;
    let mut out = vec![0.0f32; (m * N_HEADS * HEAD_DIM) as usize];
    for q_idx in 0..m {
        let causal_len = ((q_start + q_idx + 1).min(n_kv)) as usize;
        for h in 0..N_HEADS {
            let kvh = (h / group_size) as usize;
            let q_off = ((q_idx * N_HEADS + h) * HEAD_DIM) as usize;
            let mut scores = Vec::with_capacity(causal_len);
            let mut max_score = f64::NEG_INFINITY;
            for t in 0..causal_len {
                let k_off = t * pos_stride + kvh * head_dim_us;
                let mut dot = 0.0f64;
                for d in 0..head_dim_us {
                    dot += (q[q_off + d] as f64) * (k[k_off + d] as f64);
                }
                let s = dot * scale_d;
                if s > max_score { max_score = s; }
                scores.push(s);
            }
            let mut weights: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() { *w /= sum; }
            let o_off = ((q_idx * N_HEADS + h) * HEAD_DIM) as usize;
            for d in 0..head_dim_us {
                let mut acc = 0.0f64;
                for t in 0..causal_len {
                    let v_off = t * pos_stride + kvh * head_dim_us;
                    acc += weights[t] * (v[v_off + d] as f64);
                }
                out[o_off + d] = acc as f32;
            }
        }
    }
    out
}

fn host_buf(dev: &VulkanDevice, allocator: &mut Allocator, data: &[f32]) -> Result<GpuBuffer, Box<dyn std::error::Error>> {
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    let mut buf = GpuBuffer::new(
        &dev.device, allocator,
        bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "probe_in",
    )?;
    buf.write_bytes(bytes)?;
    Ok(buf)
}

fn output_buf(dev: &VulkanDevice, allocator: &mut Allocator, count: usize) -> Result<GpuBuffer, Box<dyn std::error::Error>> {
    Ok(GpuBuffer::new(
        &dev.device, allocator,
        (count * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuToCpu,
        "probe_out",
    )?)
}
