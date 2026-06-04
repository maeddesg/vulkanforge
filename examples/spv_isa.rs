//! Minimal SPIR-V → compute pipeline harness for ISA inspection.
//!
//! Creates a VkComputePipeline from an arbitrary `.spv` (arg 1) under the
//! active RADV. With `RADV_DEBUG=shaders` set, ACO dumps the compiled ISA
//! to stderr at pipeline-creation time — so this lets us inspect what the
//! *current* RADV/ACO emits for a standalone coopmat shader (e.g. the
//! Sprint-34 LOAD_TR reproducer) under mesa-main, which RGA's bundled
//! compiler cannot do.
//!
//!   source ~/tmp/mesa-main-source.env.sh
//!   RADV_DEBUG=shaders cargo run --release --example spv_isa -- <file.spv> [subgroup_size]
//!
//! Descriptor layout is fixed to the reproducer's: 2 storage buffers
//! (binding 0,1) + an 8-byte push-constant range. `subgroup_size` (default
//! 64) pins VkPipelineShaderStageRequiredSubgroupSize for Wave64 coopmat.

use std::io::Cursor;

use ash::vk;
use vulkanforge::backend::vulkan::device::VulkanDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spv_path = std::env::args().nth(1).ok_or("usage: spv_isa <file.spv> [subgroup_size]")?;
    let sg: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let dev = VulkanDevice::new()?;
    let d = &dev.device;

    let mut code_cursor = Cursor::new(std::fs::read(&spv_path)?);
    let code = ash::util::read_spv(&mut code_cursor)?;
    let module =
        unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? };

    // 2 storage buffers (binding 0,1), compute stage.
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl = unsafe {
        d.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?
    };
    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)];
    let set_layouts = [dsl];
    let pl_layout = unsafe {
        d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts).push_constant_ranges(&pc_range), None)?
    };

    let entry = c"main";
    let mut sg_info =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(sg);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry)
        .push_next(&mut sg_info);
    let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl_layout);

    // ACO compiles + (with RADV_DEBUG=shaders) dumps ISA right here.
    let pipeline = unsafe {
        d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)?
    };
    println!("pipeline created for {spv_path} (subgroup_size={sg}) — ISA dumped above if RADV_DEBUG=shaders");

    unsafe {
        d.destroy_pipeline(pipeline[0], None);
        d.destroy_pipeline_layout(pl_layout, None);
        d.destroy_descriptor_set_layout(dsl, None);
        d.destroy_shader_module(module, None);
    }
    Ok(())
}
