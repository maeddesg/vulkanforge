//! VulkanForge — Vulkan-based LLM inference engine for AMD RDNA 4.
//!
//! Phase 2A demo: device init → PipelineRegistry (with on-disk
//! pipeline cache) → VRAM arena → reflection summary table → clean
//! teardown. The actual decode loop lives in Phase 2B/C.

use std::time::Instant;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline_registry::{PipelineRegistry, default_cache_path};
use vulkanforge::backend::vulkan::shaders::ALL_SHADERS;
use vulkanforge::backend::vulkan::vram_arena::{ArenaConfig, VramArena};

/// Phase-2A arena demo budget. Real Phase-2B will use ~13 GB
/// (Qwen3-8B Q4_K_M weights + KV cache); here a 256-MiB toy budget
/// keeps the arena demo fast and CI-friendly.
const ARENA_DEMO_CONFIG: ArenaConfig = ArenaConfig {
    weights_bytes: 200 * 1024 * 1024,
    kv_cache_bytes: 50 * 1024 * 1024,
    scratch_bytes: 4 * 1024 * 1024,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge v0.1.0 — Phase 2A demo");

    let dev = VulkanDevice::new()?;
    println!("✅ Vulkan device initialized");

    let max_alloc = backend::vulkan::vram_arena::query_max_memory_allocation_size(
        &dev.instance,
        dev.physical_device,
    );
    println!(
        "  maxMemoryAllocationSize: {:.1} GiB",
        (max_alloc as f64) / (1024.0 * 1024.0 * 1024.0)
    );

    // gpu-allocator stays around for buffer-level helpers used by the
    // Phase-1 regression dispatch — the new VramArena handles zones
    // for actual model weights, but we keep gpu-allocator for the
    // small ad-hoc storage / staging buffers in the dispatch test.
    let allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    println!("✅ gpu_allocator initialized");

    // ---- Step 2.1 / 2.2: Pipeline registry + pipeline cache ----
    let cache_path = default_cache_path();
    let cache_path_display = cache_path
        .as_deref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<no $HOME>".into());
    println!("  Pipeline cache: {cache_path_display}");

    let cold_started = Instant::now();
    let (registry, loaded_bytes) =
        PipelineRegistry::new(&dev.device, cache_path.as_deref())?;
    let total_setup = cold_started.elapsed();

    println!(
        "✅ PipelineRegistry: {} shaders, vkCreateComputePipelines = {:.3} ms (incl. shader-module setup), \
         total registry setup = {:.3} ms",
        registry.count(),
        registry.create_duration.as_secs_f64() * 1000.0,
        total_setup.as_secs_f64() * 1000.0,
    );
    if loaded_bytes > 0 {
        println!("  Loaded {} B from existing pipeline cache (warm start)", loaded_bytes);
    } else {
        println!("  No existing cache — cold start");
    }

    // Reflection summary table per shader.
    println!("\n  ┌─────────────────────────────────┬──────┬────────┬─────────────┬────────────────┐");
    println!("  │ Shader                          │ Bind │ PC (B) │ SpecIds     │ LocalSize      │");
    println!("  ├─────────────────────────────────┼──────┼────────┼─────────────┼────────────────┤");
    for &id in ALL_SHADERS {
        let k = registry.get(id);
        let r = &k.reflection;
        let spec_str = if r.spec_constants.is_empty() {
            "—".to_string()
        } else {
            r.spec_constants
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "  │ {:<31} │ {:>4} │ {:>6} │ {:<11} │ {:>3}×{:>3}×{:>3}    │",
            id.name(),
            r.bindings.len(),
            r.push_constant_size,
            spec_str,
            r.local_size[0],
            r.local_size[1],
            r.local_size[2],
        );
    }
    println!("  └─────────────────────────────────┴──────┴────────┴─────────────┴────────────────┘");

    // ---- Step 2.3: VRAM arena ----
    let arena_total = ARENA_DEMO_CONFIG.weights_bytes
        + ARENA_DEMO_CONFIG.kv_cache_bytes
        + ARENA_DEMO_CONFIG.scratch_bytes;
    println!(
        "\n  Arena demo budget: {:.1} MiB total \
         (weights {:.1} MiB, KV {:.1} MiB, scratch {:.1} MiB)",
        (arena_total as f64) / (1024.0 * 1024.0),
        (ARENA_DEMO_CONFIG.weights_bytes as f64) / (1024.0 * 1024.0),
        (ARENA_DEMO_CONFIG.kv_cache_bytes as f64) / (1024.0 * 1024.0),
        (ARENA_DEMO_CONFIG.scratch_bytes as f64) / (1024.0 * 1024.0),
    );

    let arena = VramArena::new(
        &dev.instance,
        dev.physical_device,
        &dev.device,
        ARENA_DEMO_CONFIG,
    )?;
    println!(
        "✅ VramArena: {} B allocated on memory_type_index {} (DEVICE_LOCAL), zone alignment {} B",
        arena.total_bytes, arena.memory_type_index, arena.zone_alignment
    );
    println!(
        "  Zones — weights: {}..{} (size {}), kv_cache: {}..{} (size {}), scratch: {}..{} (size {})",
        arena.layout.weights.offset,
        arena.layout.weights.offset + arena.layout.weights.size,
        arena.layout.weights.size,
        arena.layout.kv_cache.offset,
        arena.layout.kv_cache.offset + arena.layout.kv_cache.size,
        arena.layout.kv_cache.size,
        arena.layout.scratch.offset,
        arena.layout.scratch.offset + arena.layout.scratch.size,
        arena.layout.scratch.size,
    );

    // Demo: create a buffer in each zone to verify bind paths work.
    let weights_view = arena.create_buffer(
        &dev.device,
        arena.layout.weights.offset,
        144 * 256, // 256 Q4_K blocks
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let kv_view = arena.create_buffer(
        &dev.device,
        arena.layout.kv_cache.offset,
        4096,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let (scratch0_off, scratch_half) = arena.scratch_for_layer(0);
    let (scratch1_off, _) = arena.scratch_for_layer(1);
    assert_ne!(scratch0_off, scratch1_off);
    let scratch0_view = arena.create_buffer(
        &dev.device,
        scratch0_off,
        scratch_half,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    println!(
        "  ✅ Buffer views: weights@{:#x}, kv@{:#x}, scratch_layer_0@{:#x}, scratch_layer_1_offset={:#x} (ping-pong)",
        arena.layout.weights.offset,
        arena.layout.kv_cache.offset,
        scratch0_off,
        scratch1_off,
    );

    // ---- Save pipeline cache for next start ----
    let cache_stats = registry.save_cache(&dev.device);
    if cache_stats.saved_bytes > 0 {
        println!(
            "  Pipeline cache saved: {} B → {}",
            cache_stats.saved_bytes, cache_path_display
        );
    }

    // ---- Teardown — buffers → arena → registry → allocator → device.
    unsafe {
        dev.device.destroy_buffer(weights_view, None);
        dev.device.destroy_buffer(kv_view, None);
        dev.device.destroy_buffer(scratch0_view, None);
    }
    arena.destroy(&dev.device);
    registry.destroy(&dev.device);
    drop(allocator);
    println!("✅ Phase 2A teardown clean");
    Ok(())
}
