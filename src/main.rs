//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 0 smoke test: enumerate the GPU, init a logical device + a
//! compute queue, and dump heap info. Compute pipelines, dispatch,
//! and the Q4_K MMVQ port arrive in Phase 1.

mod backend;

use ash::vk;

fn main() {
    println!("VulkanForge v0.1.0");

    let dev = match backend::vulkan::device::VulkanDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ Vulkan init failed: {e}");
            std::process::exit(1);
        }
    };

    println!("✅ Vulkan device initialized");

    let props = unsafe {
        dev.instance
            .get_physical_device_properties(dev.physical_device)
    };
    println!(
        "  API Version: {}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version)
    );

    let mem_props = unsafe {
        dev.instance
            .get_physical_device_memory_properties(dev.physical_device)
    };
    for i in 0..mem_props.memory_heap_count {
        let heap = mem_props.memory_heaps[i as usize];
        println!(
            "  Heap {}: {} MB {:?}",
            i,
            heap.size / 1024 / 1024,
            heap.flags
        );
    }
}
