//! v0.2.1 Sprint 11G-D — empirical RDNA4 / RADV WMMA accumulator
//! lane->cell mapping probe.
//!
//! Stages a 16x16 int32 fragment with cell (r, c) = r*16 + c, dispatches
//! `probe_coopmat_layout`, and decodes the (lane, elem_idx, value) triples
//! the shader writes per lane back into a (lane, elem) -> (row, col)
//! mapping. Prints the full table plus a regularity check (looking for
//! row(lane, i) = lane / 4 + i * something patterns) so the Sprint 11G-D
//! direct-fold rewrite can substitute a closed-form mapping function for
//! the LDS round-trip + per-thread scratch read.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

fn main() {
    if let Err(e) = run() {
        eprintln!("\u{2718} {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    let (registry, _) = PipelineRegistry::new(&dev.device, None)?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    // Output budget: 64 lanes * up to 8 elements * 3 ints (lane, elem, val) = 1536 ints.
    let out_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        (64 * 8 * 3 * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu,
        "probe_out",
    )?;

    let len_buf = GpuBuffer::new(
        &dev.device, &mut allocator,
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu,
        "probe_len",
    )?;

    let kernel = registry.get(ShaderId::ProbeCoopmatLayout);
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 2,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None)? };
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { dev.device.allocate_descriptor_sets(&alloc)? }[0];
    let infos = [
        vk::DescriptorBufferInfo { buffer: out_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: len_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
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
    unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;
    let out_h = out_buf.handle;
    let len_h = len_buf.handle;

    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
        dev.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[],
        );
        dev.device.cmd_dispatch(cmd, 1, 1, 1);
        let posts = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(out_h).offset(0).size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(len_h).offset(0).size(vk::WHOLE_SIZE),
        ];
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[], &posts, &[],
        );
    })?;

    let len_bytes = len_buf.read_bytes()?;
    let n_elems = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]);
    println!("frag.length() = {n_elems}");
    println!();

    let out_bytes = out_buf.read_bytes()?;
    let total_triples = (64 * n_elems) as usize;
    let triples: Vec<i32> = bytemuck::cast_slice::<u8, i32>(&out_bytes[..total_triples * 12])
        .to_vec();

    println!("(lane, elem_idx) -> (row, col):");
    println!();
    println!("{:>4} | {:>4} | {:>4} | {:>4} | {:>4}",
             "lane", "elem", "val", "row", "col");
    println!("{:->4} | {:->4} | {:->4} | {:->4} | {:->4}", "", "", "", "", "");

    let mut mapping: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(total_triples);
    for chunk in triples.chunks(3) {
        let lane = chunk[0] as u32;
        let elem = chunk[1] as u32;
        let val = chunk[2];
        let row = (val / 16) as u32;
        let col = (val % 16) as u32;
        mapping.push((lane, elem, row, col));
    }
    mapping.sort_by_key(|&(l, e, _, _)| (l, e));

    // Full mapping: one line per lane, all elements on the same line.
    for lane in 0..64u32 {
        let elems: Vec<String> = mapping.iter()
            .filter(|&&(l, _, _, _)| l == lane)
            .map(|&(_, e, r, c)| format!("e{e}=({r:2},{c:2})"))
            .collect();
        println!("lane {lane:2}: {}", elems.join("  "));
    }
    println!();

    // Try the most common candidate formula for Wave64 16x16 accumulator
    // and see if it matches:
    //   row = (lane / 16) * 4 + elem,  col = lane % 16        ("row in bands, lane is column")
    //   row = lane / 4 + (elem % 4) * something,  col = (lane % 4) * 4 + elem
    //   row = lane,                col = elem * 4 + ?         (only for short fragments)
    // The probe data tells us which (if any) is right.
    let mut formulas_passed: Vec<&str> = Vec::new();

    // Formula A: row = (lane >> 4) * (n_elems) + elem, col = lane & 15
    if mapping
        .iter()
        .all(|&(l, e, r, c)| r == (l >> 4) * n_elems + e && c == (l & 15))
    {
        formulas_passed.push("A: row = (lane / 16) * len + elem, col = lane % 16");
    }
    // Formula B: row = (lane >> 4) + (elem * 4),  col = lane & 15
    if mapping
        .iter()
        .all(|&(l, e, r, c)| r == (l >> 4) + e * 4 && c == (l & 15))
    {
        formulas_passed.push("B: row = lane/16 + elem*4, col = lane % 16");
    }
    // Formula C: row = lane >> 2,  col = (lane & 3) * 4 + elem
    if mapping
        .iter()
        .all(|&(l, e, r, c)| r == l >> 2 && c == (l & 3) * 4 + e)
    {
        formulas_passed.push("C: row = lane / 4, col = (lane % 4) * 4 + elem");
    }
    // Formula D: half-Wave32 split. row depends on (lane >= 32), elem and lane%32
    if mapping
        .iter()
        .all(|&(l, e, r, c)| {
            let half = l >> 5; // 0 or 1
            let lo = l & 31u32;
            r == half * 8 + lo / 4 + e * 0 && c == (lo & 3) * 4 + e
        })
    {
        formulas_passed.push("D: row = (lane / 32) * 8 + (lane % 32) / 4, col = (lane % 4) * 4 + elem");
    }
    // Formula E: half rows. row depends on lane and not elem; col depends on elem
    if mapping
        .iter()
        .all(|&(l, e, r, c)| {
            let half = l >> 5;
            let lo = l & 31u32;
            r == half * 8 + lo / 4 && c == (lo & 3) * 4 + e
        })
    {
        formulas_passed.push("E: row = (lane / 32) * 8 + (lane % 32) / 4, col = (lane % 4) * 4 + elem");
    }
    // Formula F: row = lane, col = elem (only valid if length() == 16, lane in [0..15])
    if n_elems == 16
        && mapping
            .iter()
            .all(|&(l, e, r, c)| r == l && c == e)
    {
        formulas_passed.push("F: row = lane, col = elem (lane in 0..15, only for length 16)");
    }

    if formulas_passed.is_empty() {
        println!("No simple formula matched. The mapping must be encoded as a table.");
        // Print the full mapping for documentation.
        println!();
        println!("FULL MAPPING (lane, elem) -> (row, col):");
        for &(lane, elem, row, col) in &mapping {
            println!("  ({lane:2}, {elem}) -> ({row:2}, {col:2})");
        }
    } else {
        for f in &formulas_passed {
            println!("\u{2714} formula matched: {f}");
        }
    }

    drop(out_buf);
    drop(len_buf);
    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    cmd_ctx.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}
