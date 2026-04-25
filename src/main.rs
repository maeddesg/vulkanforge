//! VulkanForge — Phase 2C debug walk.
//!
//! Runs the forward pass one layer at a time, reading back the
//! activations between layers to find the first NaN-producing layer.

use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::q4k;

const MAX_SEQ_LEN: u32 = 2048;

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge v0.1.0 — Phase 2C debug walk");

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

    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });
    println!("  GGUF: {}", model_path.display());

    let parse_start = Instant::now();
    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    println!("  parsed in {:.1} ms", parse_start.elapsed().as_secs_f64() * 1000.0);

    let load_start = Instant::now();
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "✅ {} tensors, {:.2} GiB in {:.1} s",
        model.tensors.len(),
        (model.bytes_uploaded as f64) / (1024.0 * 1024.0 * 1024.0),
        load_start.elapsed().as_secs_f64()
    );

    let kv_cache = KvCache::new(
        &dev.device, &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: MAX_SEQ_LEN,
        },
    )?;

    let profiler = if std::env::var("VF_PROFILE").is_ok() {
        Some(vulkanforge::backend::vulkan::profiler::ShaderProfiler::new(
            &dev.instance, dev.physical_device, dev.queue_family_index, &dev.device, 1024,
        )?)
    } else {
        None
    };
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), profiler)?;

    // Initial input — choose by env var:
    //   VF_INPUT=zero      → all zeros
    //   VF_INPUT=linspace  → 0.02 * linspace(-0.5, 0.5)
    //   VF_INPUT=embd      → real CPU-dequant of token_embd row (default)
    //   VF_TOKEN=N         → token id for VF_INPUT=embd (default 9707)
    let kind = std::env::var("VF_INPUT").unwrap_or_else(|_| "embd".into());
    let token_id: u32 = std::env::var("VF_TOKEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9707);
    let mut act: Vec<f32> = match kind.as_str() {
        "zero" => vec![0.0; cfg.hidden_dim as usize],
        "linspace" => (0..cfg.hidden_dim as usize)
            .map(|i| 0.02 * ((i as f32) / (cfg.hidden_dim as f32) - 0.5))
            .collect(),
        _ => embedding_row(&gguf, &cfg, token_id)?,
    };
    println!(
        "\n  layer-walk debug — initial input stats: {}",
        stats(&act)
    );

    let position: u32 = 0;
    let max_layers = std::env::var("VF_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(cfg.n_layers);

    // Optional intra-layer step debug for layer 0.
    if std::env::var("VF_TRACE_L0").is_ok() {
        use vulkanforge::backend::vulkan::forward::DebugTarget;
        for tgt in [
            DebugTarget::AttnNorm, DebugTarget::QProj, DebugTarget::KProj, DebugTarget::VProj,
            DebugTarget::QNormRope, DebugTarget::KNormRope, DebugTarget::AttnOut,
        ] {
            let v = forward.forward_layer_debug_intermediate(
                &dev, &registry, &cmd_ctx, &model,
                0, position, &act, tgt,
            )?;
            println!("    layer 0, {:?}: {}", tgt, stats(&v));
        }
        return Ok(());
    }

    if std::env::var("VF_FULL").is_ok() {
        // Full forward + LM head, like the Phase-2C demo proper.
        let stats_obj = forward.forward_token(
            &dev, &registry, &cmd_ctx, &model, &act, position,
        )?;
        let logits = forward.logits()?;
        println!(
            "  Forward total: {:.2} ms",
            stats_obj.total.as_secs_f64() * 1000.0
        );
        if !stats_obj.per_shader.is_empty() {
            let total_us: f64 = stats_obj.per_shader.values().map(|(d, _)| d.as_secs_f64() * 1e6).sum();
            println!("  Per-shader breakdown ({} shader entries):", stats_obj.per_shader.len());
            let mut bd: Vec<_> = stats_obj.per_shader.iter().collect();
            bd.sort_by(|a, b| b.1.0.cmp(&a.1.0));
            for (name, (d, n)) in bd.iter().take(12) {
                println!(
                    "    {:<22} {:>4} calls  {:>9.3} µs  {:>5.1}%",
                    name, n,
                    d.as_secs_f64() * 1e6,
                    d.as_secs_f64() * 1e6 / total_us * 100.0,
                );
            }
            if !stats_obj.per_layer.is_empty() {
                let lus: Vec<f64> = stats_obj.per_layer.iter().map(|d| d.as_secs_f64() * 1e6).collect();
                let min = lus.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = lus.iter().cloned().fold(0.0_f64, f64::max);
                let mean = lus.iter().sum::<f64>() / lus.len() as f64;
                println!(
                    "  Per-layer time (µs): min={:.0}  mean={:.0}  max={:.0}  ({} layers)",
                    min, mean, max, lus.len()
                );
            }
        }
        let nan = logits.iter().any(|v| !v.is_finite());
        println!(
            "  Logits: {} elements  any NaN/Inf: {}",
            logits.len(), nan
        );
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (idx, val)) in indexed[..10].iter().enumerate() {
            println!("    #{:>2}  id={:>6}  logit={:.4}", rank + 1, idx, val);
        }
        forward.destroy(&dev.device, &mut allocator);
        cmd_ctx.destroy(&dev.device);
        model.destroy(&dev.device, &mut allocator);
        registry.destroy(&dev.device);
        drop(allocator);
        let _ = vk::Buffer::null();
        return Ok(());
    }

    for layer in 0..max_layers {
        let out = forward.forward_layer_debug(
            &dev, &registry, &cmd_ctx, &model,
            layer, position, &act,
        )?;
        let st = stats(&out);
        let nan = st.contains("NaN") || st.contains("Inf");
        let marker = if nan { "❌" } else { "✓ " };
        println!("    layer {:>2}: {} {}", layer, marker, st);
        if nan {
            println!("\n  → first NaN-producing layer: {layer}");
            // Dump first 16 values of the input and output to help
            // narrow down WHICH dispatch in this layer caused it.
            print!("    input[..8]:  ");
            for v in &act[..8] { print!("{:>10.5} ", v); }
            println!();
            print!("    output[..8]: ");
            for v in &out[..8] { print!("{:>10.5} ", v); }
            println!();
            break;
        }
        act = out;
    }

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

/// CPU-side embedding lookup: read a Q4_K row from token_embd.weight
/// straight out of the mmap'd GGUF and dequantise to f32.
fn embedding_row(
    gguf: &GgufFile,
    cfg: &ModelConfig,
    token_id: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let info = gguf
        .tensor("token_embd.weight")
        .ok_or("token_embd.weight not in GGUF")?;
    let blocks_per_row = (cfg.hidden_dim as usize) / q4k::QUANT_K;
    let row_bytes = blocks_per_row * q4k::BLOCK_BYTES;
    let bytes = gguf.tensor_bytes(info);
    let row_off = (token_id as usize) * row_bytes;
    if row_off + row_bytes > bytes.len() {
        return Err(format!("token_id {} out of range", token_id).into());
    }
    let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
    for b in 0..blocks_per_row {
        let blk_off = row_off + b * q4k::BLOCK_BYTES;
        let block: &[u8; q4k::BLOCK_BYTES] =
            (&bytes[blk_off..blk_off + q4k::BLOCK_BYTES]).try_into().unwrap();
        let dq = q4k::dequant_block(block);
        out.extend_from_slice(&dq);
    }
    Ok(out)
}

fn stats(v: &[f32]) -> String {
    let n = v.len();
    let nan = v.iter().any(|x| x.is_nan());
    let inf = v.iter().any(|x| x.is_infinite());
    let finite_n = v.iter().filter(|x| x.is_finite()).count();
    if nan || inf {
        return format!("len={n} NaN={nan} Inf={inf} finite={finite_n}/{n}");
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &x in v {
        if x < min { min = x; }
        if x > max { max = x; }
        sum += x as f64;
        sum_sq += (x as f64) * (x as f64);
    }
    let mean = sum / n as f64;
    let std = (sum_sq / n as f64 - mean * mean).max(0.0).sqrt();
    format!("len={n}  min={:.4}  mean={:.4}  std={:.4}  max={:.4}", min, mean, std, max)
}
