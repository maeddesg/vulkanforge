//! VulkanForge — Phase 2B demo.
//!
//! Loads `~/models/Qwen3-8B-Q4_K_M.gguf`, parses the metadata,
//! reports the model config, uploads every tensor to VRAM, and
//! dumps a small summary. The actual decode loop ships in Phase 2C.

use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::shaders::ALL_SHADERS;

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge v0.1.0 — Phase 2B demo");

    let dev = VulkanDevice::new()?;
    println!("✅ Vulkan device initialized");

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;
    println!("✅ gpu_allocator initialized");

    let cache_path = default_cache_path();
    let (registry, loaded_cache) = PipelineRegistry::new(&dev.device, cache_path.as_deref())?;
    println!(
        "✅ PipelineRegistry: {} shaders ({:.1} ms create, {} B cache loaded)",
        registry.count(),
        registry.create_duration.as_secs_f64() * 1000.0,
        loaded_cache,
    );
    let _ = ALL_SHADERS;

    // ---- GGUF: parse + introspect.
    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });
    println!("\n  GGUF: {}", model_path.display());

    let parse_started = Instant::now();
    let gguf = GgufFile::open(&model_path)?;
    let parse_dt = parse_started.elapsed();
    println!(
        "  parsed: {} tensors, {} metadata KVs, alignment {} B (parse {:.1} ms)",
        gguf.tensor_count,
        gguf.metadata.len(),
        gguf.alignment,
        parse_dt.as_secs_f64() * 1000.0,
    );

    let cfg = ModelConfig::from_gguf(&gguf)?;
    print_config(&cfg);

    // Tensor inventory by quant type.
    let (mut by_type, mut total_bytes): (std::collections::BTreeMap<&str, (u32, u64)>, u64) =
        (std::collections::BTreeMap::new(), 0);
    for info in gguf.tensors.values() {
        let key = ggml_type_name(info.ggml_type);
        let entry = by_type.entry(key).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += info.byte_size();
        total_bytes += info.byte_size();
    }
    println!(
        "\n  Tensor inventory ({} tensors, total {:.2} GiB):",
        gguf.tensors.len(),
        (total_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
    );
    for (ty, (count, bytes)) in &by_type {
        println!(
            "    {:<6}: {:>4} tensors, {:>10.2} MiB",
            ty,
            count,
            (*bytes as f64) / (1024.0 * 1024.0)
        );
    }

    // ---- Load every tensor to VRAM.
    println!("\n  Uploading weights to VRAM (1-GiB staging buffer, batched)…");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "✅ LoadedModel: {} tensors, {:.2} GiB uploaded in {:.1} s ({:.1} GB/s)",
        model.tensors.len(),
        (model.bytes_uploaded as f64) / (1024.0 * 1024.0 * 1024.0),
        model.upload_duration.as_secs_f64(),
        (model.bytes_uploaded as f64 / 1e9) / model.upload_duration.as_secs_f64(),
    );

    // Spot-check a couple tensors.
    for name in [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q_norm.weight",
        "blk.35.ffn_down.weight",
        "output_norm.weight",
    ] {
        if let Some(t) = model.tensor(name) {
            println!(
                "  {:<35} shape={:?} type={:<5} {:>9} B",
                name,
                t.shape,
                ggml_type_name(t.ggml_type),
                t.byte_size,
            );
        } else {
            println!("  {:<35} (missing)", name);
        }
    }

    // ---- Cleanup.
    let cache_stats = registry.save_cache(&dev.device);
    if cache_stats.saved_bytes > 0 {
        println!(
            "  Pipeline cache saved: {} B → {}",
            cache_stats.saved_bytes,
            cache_path.as_deref().map(|p| p.display().to_string()).unwrap_or_default()
        );
    }

    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    println!("✅ Phase 2B teardown clean");
    Ok(())
}

fn print_config(c: &ModelConfig) {
    println!("\n  ModelConfig:");
    println!("    architecture        : {}", c.architecture);
    println!("    n_layers            : {}", c.n_layers);
    println!("    n_heads             : {}", c.n_heads);
    println!("    n_kv_heads          : {} (GQA group size {})", c.n_kv_heads, c.n_heads / c.n_kv_heads);
    println!("    hidden_dim          : {}", c.hidden_dim);
    println!("    ffn_dim             : {}", c.ffn_dim);
    println!("    head_dim            : {}", c.head_dim);
    println!("    rope_freq_base      : {}", c.rope_freq_base);
    println!("    context_length      : {}", c.context_length);
    println!("    vocab_size          : {}", c.vocab_size);
    println!("    rms_norm_eps        : {:e}", c.rms_norm_eps);
    println!("    has_qk_norm         : {}", c.has_qk_norm);
}

fn ggml_type_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q8_1 => "Q8_1",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::Q8K => "Q8_K",
    }
}
