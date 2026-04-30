//! Sprint 12G-C — per-shader prefill GPU profiling.
//!
//! Mirrors `profile_positions.rs` but for the batched prefill path.
//! Creates a fresh `ShaderProfiler` (no inter-call reset needed since
//! prefill_batch is the first dispatch into the profiler), runs one
//! `prefill_batch(pp)` call, and prints the per-shader breakdown.
//!
//! Run:
//!   VF_PP=128 cargo run --release --example profile_prefill
//!   VF_PP=512 cargo run --release --example profile_prefill

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::profiler::ShaderProfiler;

const MAX_SEQ_LEN: u32 = 1024;

fn main() {
    if let Err(e) = run() {
        eprintln!("ERR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let pp: u32 = std::env::var("VF_PP").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    println!("VulkanForge — prefill profiling at pp={pp}");

    if std::env::var("VULKANFORGE_MAX_PREFILL").is_err() {
        unsafe { std::env::set_var("VULKANFORGE_MAX_PREFILL", pp.to_string()); }
    }

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

    let model_path = std::env::var("VF_MODEL_PATH").ok().map(PathBuf::from)
        .unwrap_or_else(|| std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
            .expect("$HOME unset"));
    println!("  GGUF: {}", model_path.display());

    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let load_start = Instant::now();
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "  loaded {} tensors in {:.1} s",
        model.tensors.len(), load_start.elapsed().as_secs_f64(),
    );

    let kv_cache = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        max_seq_len: MAX_SEQ_LEN,
    })?;

    // 4096 pairs is plenty for pp <= 512 (≤ ~16 dispatches/layer × 36 + bracket).
    let profiler = ShaderProfiler::new(
        &dev.instance, dev.physical_device, dev.queue_family_index,
        &dev.device, 4096,
    )?;
    let mut forward = Forward::new_with_prefill(
        &dev, &mut allocator, kv_cache, cfg.clone(), Some(profiler), pp,
    )?;

    // Build a pp×hidden embedding (token id 0; GEMMs do same work).
    let one_embd = embedding_row(&gguf, &cfg, 0)?;
    let mut embeds = Vec::with_capacity((pp as usize) * one_embd.len());
    for _ in 0..pp { embeds.extend_from_slice(&one_embd); }

    // Single measured prefill — profiler has not been reset, but
    // next_query starts at 0 because nothing else has used it.
    forward.kv_cache.reset();
    let t0 = Instant::now();
    forward.prefill_batch(&dev, &registry, &cmd_ctx, &model, &embeds, pp, 0)?;
    let wall = t0.elapsed();

    // Collect.
    let samples = forward.profiler.as_ref()
        .ok_or("profiler missing")?
        .collect(&dev.device)?;
    let agg = ShaderProfiler::aggregate(&samples);
    let total_gpu: Duration = samples.iter().map(|s| s.elapsed).sum();

    println!(
        "\n--- pp={pp} ---  wall={:.2} ms  gpu_sum={:.2} ms  effective={:.0} tok/s",
        wall.as_secs_f64() * 1000.0,
        total_gpu.as_secs_f64() * 1000.0,
        (pp as f64) / wall.as_secs_f64(),
    );

    let mut rows: Vec<(&String, Duration, u32)> =
        agg.iter().map(|(n, (d, c))| (n, *d, *c)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let total_us = total_gpu.as_secs_f64() * 1e6;
    println!("  {:<24}{:>6}{:>11}{:>9}", "Shader", "Calls", "Time (µs)", "% GPU");
    let mut by_cat: BTreeMap<&'static str, (Duration, u32)> = BTreeMap::new();
    for (name, d, calls) in &rows {
        let us = d.as_secs_f64() * 1e6;
        let pct = if total_us > 0.0 { us / total_us * 100.0 } else { 0.0 };
        println!("  {:<24}{:>6}{:>11.1}{:>8.1}%", name, calls, us, pct);
        let c = category(name);
        let e = by_cat.entry(c).or_insert((Duration::ZERO, 0));
        e.0 += *d;
        e.1 += *calls;
    }
    println!("\n  Categories:");
    let mut cats: Vec<(&&'static str, &(Duration, u32))> = by_cat.iter().collect();
    cats.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (cat, (d, n)) in cats {
        let us = d.as_secs_f64() * 1e6;
        let pct = if total_us > 0.0 { us / total_us * 100.0 } else { 0.0 };
        println!("  {:<24}{:>6}{:>11.1}{:>8.1}%", cat, n, us, pct);
    }

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

fn category(label: &str) -> &'static str {
    if label.starts_with("gemm_") || label == "lm_head" { "GEMM" }
    else if label.starts_with("gemv_") { "GEMV" }
    else if label == "fa_batch" || label == "fa_tiled" || label.starts_with("fa_") || label == "scalar_attn" { "Attention" }
    else if label == "kv_write" || label.starts_with("kv_") { "KV-write" }
    else if label.starts_with("rms_norm_") { "Norm" }
    else if label.starts_with("rope_") { "RoPE" }
    else if label.starts_with("add_") { "Add" }
    else if label.starts_with("mul_") { "Mul" }
    else if label.starts_with("silu_") || label == "swiglu" { "SiLU/SwiGLU" }
    else if label.starts_with("quantize_q8_1") { "Quantize" }
    else if label.starts_with("copy") { "Copy" }
    else { "Other" }
}
