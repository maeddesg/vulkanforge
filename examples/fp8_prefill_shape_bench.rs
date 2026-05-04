//! Sprint 32 Phase 0 — per-shape FP8 prefill profiler.
//!
//! Loads a SafeTensors FP8 model, attaches a `ShaderProfiler`, and
//! runs `prefill_batch` at varying `seq_len` (pp ∈ {64, 128, 256, 512}).
//! For each pp: reset profiler → prefill → collect → aggregate per
//! label → print per-call wall-time and effective TFLOPS.
//!
//! Goal: explain why 14B FP8 prefill shows non-monotonic
//! tok/s with pp:
//!   pp=64  → 184 tok/s
//!   pp=128 → 183 tok/s
//!   pp=512 → 158 tok/s   ← drops!
//!
//! If the per-call GEMM time scales linearly with M and the TFLOPS
//! efficiency stays low (< 20%), tile-size is the lever (Sprint 32
//! Phase 1: BN=32 experiment). If GEMM time goes super-linear,
//! cache-thrashing is the lever (different fix). If attention
//! dominates at high pp, that's a separate sprint entirely.
//!
//! Usage:
//!   cargo run --release --example fp8_prefill_shape_bench -- \
//!     ~/models/Qwen2.5-14B-Instruct-FP8/ \
//!     ~/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::profiler::ShaderProfiler;

const PROFILE_CAPACITY: u32 = 8192;
const PP_VALUES: &[u32] = &[64, 128, 256, 512];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("usage: <model_dir> <tokenizer_gguf>")?);
    let _tokenizer_path = PathBuf::from(args.next().ok_or("usage: <model_dir> <tokenizer_gguf>")?);

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

    println!("Loading SafeTensors FP8 model...");
    let (model, _host_embed, _hf) =
        LoadedModel::load_safetensors(&dev, &mut allocator, &model_dir)?;
    let cfg = model.config.clone();
    println!(
        "  arch={} layers={} heads={}/{} hidden={} ffn={} vocab={}",
        cfg.architecture, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.hidden_dim, cfg.ffn_dim, cfg.vocab_size,
    );

    let kv_cache = KvCache::new(
        &dev.device, &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: 1024,
        },
    )?;
    let profiler = ShaderProfiler::new(
        &dev.instance, dev.physical_device, dev.queue_family_index,
        &dev.device, PROFILE_CAPACITY,
    )?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), Some(profiler))?;

    // Dummy embeddings: use a fixed input vector replicated across all
    // positions. We don't care about correctness — only timing.
    let max_pp = *PP_VALUES.iter().max().unwrap();
    let hidden = cfg.hidden_dim as usize;
    let mut single: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.001).sin()).collect();
    // Normalise so the embedding has reasonable magnitude.
    let norm: f32 = single.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for v in &mut single {
        *v /= norm;
    }
    let mut prefill_embeds: Vec<f32> = Vec::with_capacity(max_pp as usize * hidden);
    for _ in 0..max_pp {
        prefill_embeds.extend_from_slice(&single);
    }

    println!("\n=== Per-pp prefill profile (single forward, no warmup) ===\n");

    for &pp in PP_VALUES {
        forward.kv_cache.reset();

        // Reset profiler in its own one-shot CB before prefill.
        cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| {
            if let Some(p) = forward.profiler.as_mut() {
                p.reset(&dev.device, cmd);
            }
        })?;

        let slice = &prefill_embeds[..(pp as usize) * hidden];
        let t = Instant::now();
        forward.prefill_batch(&dev, &registry, &cmd_ctx, &model, slice, pp, 0)?;
        // prefill_batch's last submit completes before the function
        // returns (cmd_ctx.one_shot blocks on the fence), so all
        // timestamps are visible by here.
        let wall = t.elapsed();

        let samples = forward.profiler.as_ref().unwrap().collect(&dev.device).unwrap_or_default();
        let agg = ShaderProfiler::aggregate(&samples);

        let mut total_label_us: u128 = 0;
        let mut sorted: Vec<(String, Duration, u32)> = agg.into_iter()
            .map(|(k, (d, c))| (k, d, c)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (_, d, _) in &sorted { total_label_us += d.as_micros(); }

        // Theoretical FLOPS for the model's prefill GEMMs at this pp.
        // Each layer has 7 GEMMs across attention + FFN:
        //   Q : pp × hidden × hidden        (FLOPs = 2 × pp × n_heads*head_dim × hidden)
        //   K : pp × kv_dim × hidden
        //   V : pp × kv_dim × hidden
        //   O : pp × hidden × hidden
        //   Gate : pp × ffn_dim × hidden
        //   Up   : pp × ffn_dim × hidden
        //   Down : pp × hidden × ffn_dim
        let q_dim = (cfg.n_heads * cfg.head_dim) as u64;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let h = cfg.hidden_dim as u64;
        let f = cfg.ffn_dim as u64;
        let layers = cfg.n_layers as u64;
        let pp64 = pp as u64;
        // FLOPs for each GEMM = 2*M*N*K. M=pp.
        let gemm_flops_per_layer: u64 =
              2 * pp64 * q_dim * h     // Q
            + 2 * pp64 * kv_dim * h    // K
            + 2 * pp64 * kv_dim * h    // V
            + 2 * pp64 * h * h         // O (input from attn = q_dim, output = hidden)
            + 2 * pp64 * f * h         // Gate
            + 2 * pp64 * f * h         // Up
            + 2 * pp64 * h * f;        // Down
        let total_gemm_flops = gemm_flops_per_layer * layers;

        println!("==== pp = {} ====", pp);
        println!("  wall:           {:>9.3} ms", wall.as_secs_f64() * 1000.0);
        println!("  Σ profile:      {:>9.3} ms ({} unique labels)",
                 total_label_us as f64 / 1000.0, sorted.len());
        println!("  observed tok/s: {:>9.1} (= {} / {:.3} s)",
                 (pp as f64) / wall.as_secs_f64(), pp, wall.as_secs_f64());
        println!("  total GEMM FLOPs (across layers): {:.2} GFLOPs",
                 total_gemm_flops as f64 / 1e9);

        // Print the top labels by total time. For per-call ms, divide
        // by count (in dispatch_layer_batch each label fires once per
        // layer, so count = n_layers).
        println!("  {:<28} {:>10} {:>9} {:>10} {:>10}",
                 "label", "total ms", "calls", "avg ms", "% of Σ");
        println!("  {}", "-".repeat(73));
        for (name, dur, count) in &sorted {
            let total_ms = dur.as_secs_f64() * 1000.0;
            let avg_ms = total_ms / (*count as f64);
            let pct = 100.0 * (total_ms * 1000.0) as f64 / total_label_us.max(1) as f64;
            println!("  {:<28} {:>10.3} {:>9} {:>10.4} {:>9.1}%",
                     name, total_ms, count, avg_ms, pct);
        }

        // Compute approximate TFLOPS efficiency for the most-cited
        // labels by guessing their GEMM shape from the label name.
        println!();
    }

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}
