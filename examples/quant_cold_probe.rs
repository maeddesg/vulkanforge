//! Sprint D.3 — Quant cold-probe (kernel-vs-environment diagnosis).
//!
//! D.2 showed an isolated COLD *F16* GEMV hits VRAM peak (664 GB/s) while
//! in-context GEMVs run at 8–17 %, and concluded "it's the environment, not
//! the kernel". That conclusion confounded three variables: quant/kernel, byte
//! size, and orchestration (it compared in-context Q8_0 against isolated-cold
//! F16). This probe removes the confounds: it runs the REAL quant kernels
//! (Q6_K, Q8_0) ISOLATED + COLD at the exact in-context byte sizes, plus F16
//! controls at MATCHED bytes — leaving only the environment as a variable.
//!
//! Method (D.2 cold-MALL protocol): read W → read a disjoint >64 MB region
//! (MALL flush, 64 MB Infinity Cache) → read W cold → time it. Each timed read
//! is its own submit (clean timing, no surrounding dispatches/barriers).
//! Pipelines come from the real `PipelineRegistry` (correct spec constants +
//! Wave64 pin). Weight buffers are sized to the exact GGUF block byte counts;
//! values are random (timing only).
//!
//! Standalone diagnostic binary — does not touch the inference path.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::MatVecPushConstants;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const RUNS: u32 = 16;

struct Probe {
    label: &'static str,
    shader: ShaderId,
    m: u32,            // output rows (= dispatch grid, NUM_ROWS=1)
    k: u32,            // input cols
    weight_bytes: u64, // exact block-format byte count
    bindings: u32,     // 3 (one-per-row F16) or 5 (K-quant / generic)
}

fn fill_buffer(
    dev: &VulkanDevice, allocator: &mut Allocator, cmd_ctx: &CommandContext,
    buf: &GpuBuffer, bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // 16 MiB host seed (non-zero, non-uniform → no zero/compressed pages),
    // copied to fill the GpuOnly buffer so it is genuinely VRAM-resident.
    let seed_bytes: u64 = 16 * 1024 * 1024;
    let mut seed = GpuBuffer::new(&dev.device, allocator, seed_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "seed")?;
    let sv: Vec<u8> = (0..seed_bytes).map(|i| ((i * 1103515245 + 12345) >> 7) as u8).collect();
    seed.write_bytes(&sv)?;
    let mut off = 0u64;
    while off < bytes {
        let n = (bytes - off).min(seed_bytes);
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let cp = vk::BufferCopy::default().src_offset(0).dst_offset(off).size(n);
            dev.device.cmd_copy_buffer(cmd, seed.handle, buf.handle, std::slice::from_ref(&cp));
        })?;
        off += n;
    }
    seed.destroy(&dev.device, allocator);
    Ok(())
}

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
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;
    let (registry, _) = PipelineRegistry::new(&dev.device, None)?;

    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let ts_period_ns = props.limits.timestamp_period as f64;
    let qpool = unsafe {
        dev.device.create_query_pool(&vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?
    };

    // ── Probe matrix — real 26B byte sizes + F16 controls at matched bytes ──
    // Q6_K [2816→4096]: 11 blocks/row × 210 B × 4096 rows = 9.46 MB
    // Q8_0 [2816→2048]: 88 blocks/row ×  34 B × 2048 rows = 6.13 MB
    let q6k_bytes: u64 = 4096 * (2816 / 256) * 210;   // 9,461,760
    let q80_bytes: u64 = 2048 * (2816 / 32)  * 34;    // 6,127,616
    let probes = [
        Probe { label: "Q6_K  gemv_q  (in-ctx 54)", shader: ShaderId::MulMatVecQ6KSubgroup,    m: 4096, k: 2816, weight_bytes: q6k_bytes, bindings: 5 },
        Probe { label: "Q6_K_MLP gemv_q (D.2 var)",  shader: ShaderId::MulMatVecQ6KSubgroupMlp, m: 4096, k: 2816, weight_bytes: q6k_bytes, bindings: 5 },
        Probe { label: "Q8_0  gemv_k  (in-ctx 108)", shader: ShaderId::MulMatVecQ8_0Subgroup,   m: 2048, k: 2816, weight_bytes: q80_bytes, bindings: 5 },
        // F16 controls @ matched bytes (same grid m, K shrunk to match bytes).
        Probe { label: "F16   @9.46MB (ctrl gemv_q)", shader: ShaderId::MulMatVecF16, m: 4096, k: 1155, weight_bytes: 4096 * 1155 * 2, bindings: 3 },
        Probe { label: "F16   @6.13MB (ctrl gemv_k)", shader: ShaderId::MulMatVecF16, m: 2048, k: 1497, weight_bytes: 2048 * 1497 * 2, bindings: 3 },
    ];

    // Shared input / output / fuse buffers (K is the max over probes).
    let max_k = probes.iter().map(|p| p.k).max().unwrap();
    let max_m = probes.iter().map(|p| p.m).max().unwrap();
    let mut input_buf = GpuBuffer::new(&dev.device, &mut allocator, (max_k * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "input")?;
    let inv: Vec<f32> = (0..max_k).map(|i| ((i as f32) * 0.001).sin()).collect();
    input_buf.write_bytes(bytemuck::cast_slice(&inv))?;
    let output_buf = GpuBuffer::new(&dev.device, &mut allocator, (max_m * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "output")?;
    let fuse0 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse0")?;
    let fuse1 = GpuBuffer::new(&dev.device, &mut allocator, 16,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "fuse1")?;

    // Flush buffer: 256 MB ≫ 64 MB MALL. Read it (F16 GEMV) to evict W.
    let flush_bytes: u64 = 256 * 1024 * 1024;
    let flush_buf = GpuBuffer::new(&dev.device, &mut allocator, flush_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly, "flush")?;
    fill_buffer(&dev, &mut allocator, &cmd_ctx, &flush_buf, flush_bytes)?;
    let flush_k: u32 = 2816;
    let flush_m: u32 = (flush_bytes / (flush_k as u64 * 2)) as u32; // F16 rows over 256 MB

    // Descriptor pool: 6 sets (5 probes + 1 flush) × up to 5 bindings.
    let pool = unsafe {
        dev.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default()
            .max_sets(8)
            .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 40 }]), None)?
    };

    let make_set = |weight: vk::Buffer, dsl: vk::DescriptorSetLayout, bindings: u32|
        -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layouts = [dsl];
        let set = unsafe { dev.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?[0] };
        let bufs = [weight, input_buf.handle, output_buf.handle, fuse0.handle, fuse1.handle];
        let infos: Vec<_> = (0..bindings as usize).map(|i|
            vk::DescriptorBufferInfo::default().buffer(bufs[i]).offset(0).range(vk::WHOLE_SIZE)).collect();
        let writes: Vec<_> = (0..bindings).map(|i|
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i).descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i as usize]))).collect();
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    };

    let pc_for = |m: u32, k: u32| MatVecPushConstants {
        ncols: k, stride_a: k, stride_b: k, stride_d: m,
        batch_stride_a: k * m, batch_stride_b: k, batch_stride_d: m,
        fusion_flags: 0, base_work_group_y: 0, ne02: 1, ne12: 1, broadcast2: 1,
        broadcast3: 1.0_f32.to_bits(),
    };

    let dispatch = |pipeline: vk::Pipeline, layout: vk::PipelineLayout, set: vk::DescriptorSet,
                    m: u32, k: u32, timed: bool| -> Result<f64, Box<dyn std::error::Error>> {
        let pc = pc_for(m, k);
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            if timed { dev.device.cmd_reset_query_pool(cmd, qpool, 0, 2); }
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            dev.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
            if timed { dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, 0); }
            dev.device.cmd_dispatch(cmd, m, 1, 1);
            if timed { dev.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, 1); }
        })?;
        if !timed { return Ok(0.0); }
        let mut data = [0u64; 2];
        unsafe { dev.device.get_query_pool_results(qpool, 0, &mut data,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
        Ok((data[1].wrapping_sub(data[0]) as f64) * ts_period_ns / 1e6)
    };

    // Flush pipeline + set (F16 over the 256 MB flush buffer).
    let fk = registry.get(ShaderId::MulMatVecF16);
    let flush_set = make_set(flush_buf.handle, fk.descriptor_set_layout, 3)?;

    println!();
    println!("=== Sprint D.3 — quant cold-probe (isolated dispatch, MALL-flushed) ===");
    println!("{:<30} {:>10} {:>10} {:>12} {:>12}", "kernel", "MB", "warm GB/s", "COLD GB/s", "cold ms");
    println!("{}", "-".repeat(78));

    for p in &probes {
        let k_ = registry.get(p.shader);
        // Each probe gets its OWN weight buffer (so the flush, reading the
        // shared 256 MB flush_buf, evicts this W from the MALL).
        let wbuf = GpuBuffer::new(&dev.device, &mut allocator, p.weight_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly, "wprobe")?;
        fill_buffer(&dev, &mut allocator, &cmd_ctx, &wbuf, p.weight_bytes)?;
        let set = make_set(wbuf.handle, k_.descriptor_set_layout, p.bindings)?;

        let mut warm = Vec::new();
        for _ in 0..RUNS {
            dispatch(k_.pipeline, k_.pipeline_layout, set, p.m, p.k, false)?; // prime → MALL
            warm.push(dispatch(k_.pipeline, k_.pipeline_layout, set, p.m, p.k, true)?);
        }
        let mut cold = Vec::new();
        for _ in 0..RUNS {
            dispatch(fk.pipeline, fk.pipeline_layout, flush_set, flush_m, flush_k, false)?; // evict W
            cold.push(dispatch(k_.pipeline, k_.pipeline_layout, set, p.m, p.k, true)?);
        }
        warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cold.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let wm = warm[warm.len() / 2];
        let cm = cold[cold.len() / 2];
        let gb = p.weight_bytes as f64 / 1e9;
        println!("{:<30} {:>10.2} {:>10.1} {:>12.1} {:>12.4}",
            p.label, p.weight_bytes as f64 / 1e6, gb / (wm / 1000.0), gb / (cm / 1000.0), cm);

        wbuf.destroy(&dev.device, &mut allocator);
    }

    unsafe { dev.device.destroy_query_pool(qpool, None); dev.device.destroy_descriptor_pool(pool, None); }
    flush_buf.destroy(&dev.device, &mut allocator);
    input_buf.destroy(&dev.device, &mut allocator);
    output_buf.destroy(&dev.device, &mut allocator);
    fuse0.destroy(&dev.device, &mut allocator);
    fuse1.destroy(&dev.device, &mut allocator);
    Ok(())
}
