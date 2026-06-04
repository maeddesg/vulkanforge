//! Coopmat layout-semantics correctness probe (see
//! tests/sprint34_load_tr/test_coopmat_layout_check.comp).
//!
//! Computes acc = A * I with A[r][c] = r*16+c, A loaded ColumnMajor
//! (the LOAD_TR-firing layout for USE_A), and reports whether the result
//! is RowMajor-correct or transposed — i.e. whether a direct ColumnMajor
//! coopMatLoad of a [M×K] row-major weight (what triggers LOAD_TR) yields
//! the correct A tile or its transpose.
//!
//!   source ~/tmp/mesa-main-source.env.sh
//!   cargo run --release --example coopmat_check -- tests/sprint34_load_tr/test_coopmat_layout_check.spv

use std::io::Cursor;

use ash::vk;
use half::f16;
use vulkanforge::backend::vulkan::device::VulkanDevice;

unsafe fn make_buffer(
    dev: &VulkanDevice, size: u64, host: bool,
) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
    let buf = dev.device.create_buffer(
        &vk::BufferCreateInfo::default().size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE), None).unwrap();
    let req = dev.device.get_buffer_memory_requirements(buf);
    let props = dev.instance.get_physical_device_memory_properties(dev.physical_device);
    let want = if host {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else { vk::MemoryPropertyFlags::DEVICE_LOCAL };
    let idx = (0..props.memory_type_count).find(|&i| {
        (req.memory_type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(want)
    }).unwrap();
    let mem = dev.device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx),
        None).unwrap();
    dev.device.bind_buffer_memory(buf, mem, 0).unwrap();
    let ptr = if host {
        dev.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8
    } else { std::ptr::null_mut() };
    (buf, mem, ptr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spv = std::env::args().nth(1).ok_or("usage: coopmat_check <spv>")?;
    let dev = VulkanDevice::new()?;
    let d = &dev.device;
    unsafe {
        // A[i] = i (row-major [16x16]); B = identity; C = 256 f32 out.
        let (a_buf, a_mem, a_ptr) = make_buffer(&dev, 256 * 2, true);
        let (b_buf, b_mem, b_ptr) = make_buffer(&dev, 256 * 2, true);
        let (c_buf, c_mem, c_ptr) = make_buffer(&dev, 256 * 4, true);
        let a: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
        let mut b = vec![f16::from_f32(0.0); 256];
        for i in 0..16 { b[i * 16 + i] = f16::from_f32(1.0); }
        std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, a_ptr, 256 * 2);
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const u8, b_ptr, 256 * 2);
        std::ptr::write_bytes(c_ptr, 0, 256 * 4);

        let mut code = Cursor::new(std::fs::read(&spv)?);
        let module = d.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut code)?), None)?;
        let binds: Vec<_> = (0..3).map(|i| vk::DescriptorSetLayoutBinding::default()
            .binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let dsl = d.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)?;
        let set_layouts = [dsl];
        let pl = d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)?;
        let mut sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(64);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").push_next(&mut sg);
        let pipe = d.create_compute_pipelines(vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl)], None)
            .map_err(|(_, e)| e)?[0];

        let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default()
            .max_sets(1).pool_sizes(&[vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3)]), None)?;
        let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(&set_layouts))?[0];
        let infos = [
            vk::DescriptorBufferInfo::default().buffer(a_buf).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(b_buf).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(c_buf).range(vk::WHOLE_SIZE),
        ];
        let writes: Vec<_> = (0..3).map(|i| vk::WriteDescriptorSet::default()
            .dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[i]))).collect();
        d.update_descriptor_sets(&writes, &[]);

        let cpool = d.create_command_pool(&vk::CommandPoolCreateInfo::default()
            .queue_family_index(dev.queue_family_index), None)?;
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
            .command_pool(cpool).command_buffer_count(1))?[0];
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[]);
        d.cmd_dispatch(cmd, 1, 1, 1);
        d.end_command_buffer(cmd)?;
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cmds = [cmd];
        d.queue_submit(dev.compute_queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)?;
        d.wait_for_fences(&[fence], true, u64::MAX)?;

        let c = std::slice::from_raw_parts(c_ptr as *const f32, 256);
        // Probe a few off-diagonal cells.
        let probe = |i: usize, j: usize| c[i * 16 + j];
        println!("C[0][1] = {}  (rowmajor-correct: 1, transposed: 16)", probe(0, 1));
        println!("C[1][0] = {}  (rowmajor-correct: 16, transposed: 1)", probe(1, 0));
        println!("C[2][5] = {}  (rowmajor-correct: 37, transposed: 82)", probe(2, 5));
        let correct = (probe(0,1) - 1.0).abs() < 0.5 && (probe(1,0) - 16.0).abs() < 0.5;
        let transposed = (probe(0,1) - 16.0).abs() < 0.5 && (probe(1,0) - 1.0).abs() < 0.5;
        println!("VERDICT: {}", if correct { "ColumnMajor-A load is ROWMAJOR-CORRECT (flip compensates → usable)" }
                 else if transposed { "ColumnMajor-A load is TRANSPOSED (wrong A tile for [M×K]-rowmajor weights)" }
                 else { "neither — unexpected (check WMMA/identity)" });

        for (m, mem) in [(a_mem,()),(b_mem,()),(c_mem,())].iter().map(|x| (x.0, x.1)) { let _ = mem; d.free_memory(m, None); }
        d.destroy_buffer(a_buf, None); d.destroy_buffer(b_buf, None); d.destroy_buffer(c_buf, None);
        d.destroy_pipeline(pipe, None); d.destroy_pipeline_layout(pl, None);
        d.destroy_descriptor_pool(pool, None); d.destroy_descriptor_set_layout(dsl, None);
        d.destroy_shader_module(module, None); d.destroy_command_pool(cpool, None);
        d.destroy_fence(fence, None);
    }
    Ok(())
}
