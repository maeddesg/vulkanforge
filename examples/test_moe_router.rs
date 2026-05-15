//! Sprint 56A — standalone validation of the GPU-side MoE router.
//!
//! Replicates `cpu_moe_route` (executor.rs:124) on the GPU via two
//! compute shaders:
//!   - `moe_router_norm_gemv.spv`     : RMSNorm + per-channel scale +
//!                                      [n_experts, hidden] GEMV
//!   - `moe_router_softmax_topk.spv`  : softmax + top-K + renorm + pes
//!
//! Generates a deterministic random router config (token, weights),
//! computes the reference top-K (idx, weight) tuples on the CPU, runs
//! the same inputs through the two GPU shaders, and compares.
//!
//! Pass criteria:
//!   - Top-K indices match exactly between CPU and GPU paths
//!   - Top-K weights match within 1e-3 relative (CPU uses FP64
//!     accumulators, GPU is FP32 — small drift is expected and the
//!     router decisions are stable far inside that tolerance)
//!
//! Run:
//!   cargo run --release --example test_moe_router

use std::os::raw::c_void;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;

const NORM_GEMV_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/moe_router_norm_gemv.spv"));
const SOFTMAX_TOPK_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/moe_router_softmax_topk.spv"));

const HIDDEN_SIZE: u32 = 2816;
const N_EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const RMS_EPS: f32 = 1e-6;
const SEQ_LEN: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormGemvPC {
    seq_len: u32,
    hidden_size: u32,
    n_experts: u32,
    rms_norm_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxTopkPC {
    seq_len: u32,
    n_experts: u32,
    top_k: u32,
}

/// Deterministic xorshift — same RNG style as VF's sampler.
fn next_rand(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Scale to [-1, 1) for hidden/proj.
    ((s as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
}

/// Reference implementation: mirrors `cpu_moe_route` in executor.rs.
fn cpu_route(
    hidden: &[f32],
    proj: &[f32],
    scale: &[f32],
    pes: &[f32],
    hidden_size: usize,
    n_experts: usize,
    top_k: usize,
    eps: f32,
) -> Vec<(u32, f32)> {
    let mut sq_sum = 0.0_f64;
    for &v in hidden {
        sq_sum += (v as f64) * (v as f64);
    }
    let rms_inv = 1.0 / (sq_sum / (hidden_size as f64) + eps as f64).sqrt();
    let inv_sqrt = (hidden_size as f64).powf(-0.5);
    let mut scaled = vec![0.0_f64; hidden_size];
    for i in 0..hidden_size {
        scaled[i] = (hidden[i] as f64) * rms_inv * (scale[i] as f64) * inv_sqrt;
    }
    let mut scores = vec![0.0_f64; n_experts];
    for e in 0..n_experts {
        let row = &proj[e * hidden_size..(e + 1) * hidden_size];
        let mut acc = 0.0_f64;
        for j in 0..hidden_size {
            acc += (row[j] as f64) * scaled[j];
        }
        scores[e] = acc;
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs = vec![0.0_f64; n_experts];
    let mut sum = 0.0_f64;
    for e in 0..n_experts {
        probs[e] = (scores[e] - max).exp();
        sum += probs[e];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(top_k);
    let top_sum: f64 = idx.iter().map(|&i| probs[i]).sum();
    let renorm = if top_sum > 0.0 { 1.0 / top_sum } else { 0.0 };
    idx.into_iter()
        .map(|i| {
            let w = probs[i] * renorm * (pes[i] as f64);
            (i as u32, w as f32)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[1] Vulkan init");
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    eprintln!("[2] Generate deterministic test data");
    let mut rng = 0xDEADBEEFCAFE_u64;
    let h = HIDDEN_SIZE as usize;
    let ne = N_EXPERTS as usize;
    let sl = SEQ_LEN as usize;
    let k = TOP_K as usize;
    // Hidden: seq_len × hidden_size, scaled to typical residual magnitudes (~1.0 rms).
    let mut hidden: Vec<f32> = (0..sl * h).map(|_| next_rand(&mut rng) * 1.5).collect();
    // Proj: [n_experts, hidden_size]
    let mut proj: Vec<f32> = (0..ne * h).map(|_| next_rand(&mut rng) * 0.02).collect();
    // Channel scale: ~1.0 with some spread
    let scale: Vec<f32> = (0..h).map(|_| 1.0 + next_rand(&mut rng) * 0.1).collect();
    // Per-expert scale: ~1.0
    let pes: Vec<f32> = (0..ne).map(|_| 1.0 + next_rand(&mut rng) * 0.05).collect();

    eprintln!("[3] CPU reference per token");
    let mut cpu_results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(sl);
    for t in 0..sl {
        let token_hidden = &hidden[t * h..(t + 1) * h];
        let r = cpu_route(token_hidden, &proj, &scale, &pes, h, ne, k, RMS_EPS);
        cpu_results.push(r);
    }

    eprintln!("[4] Allocate GPU buffers + upload inputs");
    // Inputs.
    let mut hidden_buf = mk_buffer(
        &dev, &mut allocator, (sl * h * 4) as u64, "hidden", MemoryLocation::CpuToGpu,
    )?;
    let mut proj_buf = mk_buffer(
        &dev, &mut allocator, (ne * h * 4) as u64, "proj", MemoryLocation::CpuToGpu,
    )?;
    let mut scale_buf = mk_buffer(
        &dev, &mut allocator, (h * 4) as u64, "scale", MemoryLocation::CpuToGpu,
    )?;
    let mut pes_buf = mk_buffer(
        &dev, &mut allocator, (ne * 4) as u64, "pes", MemoryLocation::CpuToGpu,
    )?;
    hidden_buf.write_bytes(bytemuck::cast_slice(&hidden))?;
    proj_buf.write_bytes(bytemuck::cast_slice(&proj))?;
    scale_buf.write_bytes(bytemuck::cast_slice(&scale))?;
    pes_buf.write_bytes(bytemuck::cast_slice(&pes))?;

    // Intermediate + outputs.
    let logits_buf = mk_buffer(
        &dev, &mut allocator, (sl * ne * 4) as u64, "logits", MemoryLocation::GpuOnly,
    )?;
    let indices_buf = mk_buffer(
        &dev, &mut allocator, (sl * k * 4) as u64, "indices", MemoryLocation::GpuToCpu,
    )?;
    let weights_buf = mk_buffer(
        &dev, &mut allocator, (sl * k * 4) as u64, "weights", MemoryLocation::GpuToCpu,
    )?;

    eprintln!("[5] Build pipelines");
    let p1 = build_pipeline(&dev, NORM_GEMV_SPV, 4)?;
    let p2 = build_pipeline(&dev, SOFTMAX_TOPK_SPV, 4)?;

    eprintln!("[6] Bind descriptors + dispatch");
    // Pipeline 1: hidden, proj, scale → logits
    let pool = unsafe {
        dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 8,
                },
            ]),
            None,
        )?
    };
    let s1_layouts = [p1.dsl];
    let s1_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&s1_layouts);
    let s1 = unsafe { dev.device.allocate_descriptor_sets(&s1_alloc)?[0] };
    write_set(&dev, s1, &[&hidden_buf, &proj_buf, &scale_buf, &logits_buf]);

    let s2_layouts = [p2.dsl];
    let s2_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&s2_layouts);
    let s2 = unsafe { dev.device.allocate_descriptor_sets(&s2_alloc)?[0] };
    write_set(&dev, s2, &[&logits_buf, &pes_buf, &indices_buf, &weights_buf]);

    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        // Shader 1
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p1.pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, p1.layout, 0, &[s1], &[],
        );
        let pc1 = NormGemvPC {
            seq_len: SEQ_LEN,
            hidden_size: HIDDEN_SIZE,
            n_experts: N_EXPERTS,
            rms_norm_eps: RMS_EPS,
        };
        dev.device.cmd_push_constants(
            cmd, p1.layout, vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::bytes_of(&pc1),
        );
        dev.device.cmd_dispatch(cmd, SEQ_LEN, 1, 1);

        // Barrier
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&bar), &[], &[],
        );

        // Shader 2
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p2.pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, p2.layout, 0, &[s2], &[],
        );
        let pc2 = SoftmaxTopkPC {
            seq_len: SEQ_LEN,
            n_experts: N_EXPERTS,
            top_k: TOP_K,
        };
        dev.device.cmd_push_constants(
            cmd, p2.layout, vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::bytes_of(&pc2),
        );
        dev.device.cmd_dispatch(cmd, SEQ_LEN, 1, 1);
    })?;

    eprintln!("[7] Readback + compare");
    let idx_bytes = indices_buf.read_bytes()?;
    let w_bytes = weights_buf.read_bytes()?;
    let gpu_idx: &[u32] = bytemuck::cast_slice(&idx_bytes[..(sl * k) * 4]);
    let gpu_w: &[f32] = bytemuck::cast_slice(&w_bytes[..(sl * k) * 4]);

    let mut all_pass = true;
    let mut max_rel_err = 0.0_f64;
    let mut bad_tokens = 0u32;

    for t in 0..sl {
        let cpu = &cpu_results[t];
        let gpu_t_idx = &gpu_idx[t * k..(t + 1) * k];
        let gpu_t_w = &gpu_w[t * k..(t + 1) * k];
        let mut tok_ok = true;
        for (k_pos, &(c_e, c_w)) in cpu.iter().enumerate() {
            let g_e = gpu_t_idx[k_pos];
            let g_w = gpu_t_w[k_pos];
            if g_e != c_e {
                tok_ok = false;
                eprintln!(
                    "  token {t} k={k_pos}: idx mismatch CPU={c_e} GPU={g_e}"
                );
            }
            let rel = if c_w.abs() > 1e-9 {
                ((g_w - c_w) / c_w).abs() as f64
            } else {
                (g_w - c_w).abs() as f64
            };
            if rel > max_rel_err {
                max_rel_err = rel;
            }
            if rel > 1e-3 {
                tok_ok = false;
                eprintln!(
                    "  token {t} k={k_pos}: weight drift {rel:.2e} CPU={c_w:.6} GPU={g_w:.6}"
                );
            }
        }
        if !tok_ok {
            bad_tokens += 1;
            all_pass = false;
        }
    }

    eprintln!(
        "  → {} tokens, {} bad, max rel-err on weights = {:.2e}",
        sl, bad_tokens, max_rel_err
    );
    if all_pass {
        eprintln!("[8] PASS — GPU router output matches cpu_moe_route within 1e-3 relative");
    } else {
        eprintln!("[8] FAIL");
        std::process::exit(1);
    }

    // Cleanup
    hidden_buf.destroy(&dev.device, &mut allocator);
    proj_buf.destroy(&dev.device, &mut allocator);
    scale_buf.destroy(&dev.device, &mut allocator);
    pes_buf.destroy(&dev.device, &mut allocator);
    logits_buf.destroy(&dev.device, &mut allocator);
    indices_buf.destroy(&dev.device, &mut allocator);
    weights_buf.destroy(&dev.device, &mut allocator);
    unsafe {
        dev.device.destroy_descriptor_pool(pool, None);
        dev.device.destroy_pipeline(p1.pipeline, None);
        dev.device.destroy_pipeline_layout(p1.layout, None);
        dev.device.destroy_descriptor_set_layout(p1.dsl, None);
        dev.device.destroy_shader_module(p1.module, None);
        dev.device.destroy_pipeline(p2.pipeline, None);
        dev.device.destroy_pipeline_layout(p2.layout, None);
        dev.device.destroy_descriptor_set_layout(p2.dsl, None);
        dev.device.destroy_shader_module(p2.module, None);
    }
    cmd_ctx.destroy(&dev.device);
    let _ = hidden; let _ = proj;
    Ok(())
}

fn mk_buffer(
    dev: &VulkanDevice,
    allocator: &mut Allocator,
    size: u64,
    label: &str,
    loc: MemoryLocation,
) -> Result<GpuBuffer, Box<dyn std::error::Error>> {
    Ok(GpuBuffer::new(
        &dev.device,
        allocator,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        loc,
        label,
    )?)
}

struct Pipe {
    module: vk::ShaderModule,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

fn build_pipeline(
    dev: &VulkanDevice,
    spv: &[u8],
    n_bindings: u32,
) -> Result<Pipe, Box<dyn std::error::Error>> {
    let code_u32: Vec<u32> = spv.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let module = unsafe {
        dev.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&code_u32),
            None,
        )?
    };
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe {
        dev.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?
    };
    // Both shaders use the same push constant size (4 × u32/f32 = 16 B
    // for the larger of the two).
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<NormGemvPC>() as u32);
    let layouts = [dsl];
    let push_ranges = [push_range];
    let layout = unsafe {
        dev.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )?
    };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let pipeline = unsafe {
        dev.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
            None,
        ).map_err(|(_, e)| e)?[0]
    };
    Ok(Pipe { module, dsl, layout, pipeline })
}

fn write_set(dev: &VulkanDevice, set: vk::DescriptorSet, bufs: &[&GpuBuffer]) {
    let infos: Vec<vk::DescriptorBufferInfo> = bufs.iter().map(|b| {
        vk::DescriptorBufferInfo::default()
            .buffer(b.handle).offset(0).range(vk::WHOLE_SIZE)
    }).collect();
    let writes: Vec<vk::WriteDescriptorSet> = (0..bufs.len())
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        })
        .collect();
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
}

// Silence unused-import warning on `c_void` (kept to mirror the
// fp8_gemv_standalone example's import set; not technically needed).
#[allow(dead_code)]
fn _unused() -> *mut c_void { std::ptr::null_mut() }
