//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 1 / Step 1.2: device init + Q4_K GEMV compute pipeline
//! creation. The pipeline is built and immediately torn down — actual
//! buffer allocation, descriptor binding, and dispatch arrive in
//! steps 1.3 / 1.4.

mod backend;

use ash::vk;

use backend::vulkan::device::VulkanDevice;
use backend::vulkan::pipeline::{ComputeKernel, SpecConstants};
use backend::vulkan::shaders;

fn main() {
    println!("VulkanForge v0.1.0");

    let dev = match VulkanDevice::new() {
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

    // Step 1.2 — build the Q4_K GEMV pipeline. Using SMOKE_DEFAULT
    // spec constants (BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1) keeps
    // the dispatch hardware-agnostic; RDNA4-tuned values come in 1.5.
    let spv = shaders::spv_words(shaders::MUL_MAT_VEC_Q4_K_F32_F32);
    println!(
        "  SPIR-V: mul_mat_vec_q4_k_f32_f32 — {} bytes / {} words",
        shaders::MUL_MAT_VEC_Q4_K_F32_F32.len(),
        spv.len()
    );

    let kernel = match ComputeKernel::new(&dev.device, &spv, SpecConstants::SMOKE_DEFAULT) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("❌ ComputeKernel::new failed: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "✅ Q4_K GEMV pipeline created (spec: BLOCK_SIZE={}, NUM_ROWS={}, NUM_COLS={})",
        SpecConstants::SMOKE_DEFAULT.block_size,
        SpecConstants::SMOKE_DEFAULT.num_rows,
        SpecConstants::SMOKE_DEFAULT.num_cols,
    );

    kernel.destroy(&dev.device);
    println!("✅ Pipeline destroyed cleanly");
}
