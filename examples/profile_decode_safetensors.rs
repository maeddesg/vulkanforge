//! Sprint 26 — Per-dispatch GPU TIMESTAMP profiling for SafeTensors
//! FP8 models.
//!
//! VF already has a fully-instrumented `ShaderProfiler` (TIMESTAMP
//! query pool, every dispatch wrapped in `self.profile(label, ...)`).
//! It just isn't wired through the chat / bench paths today. This
//! example loads a SafeTensors FP8 model, runs prefill + a small
//! number of decode tokens with the profiler attached, and prints
//! per-label aggregate breakdown sorted by total time.
//!
//! Goal: localise the 70% BW-efficiency gap on Qwen2.5-14B-FP8
//! (13.9 t/s observed vs ~46 t/s BW-limited).
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example profile_decode_safetensors -- \
//!     ~/models/Qwen2.5-14B-Instruct-FP8/ \
//!     ~/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
//! ```
//!
//! Reads `VF_PROFILE_TOKENS` (default 8) for how many decode tokens
//! to aggregate over.

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::time::Duration;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::GgufFile;
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::profiler::ShaderProfiler;
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const USER_PROMPT: &str = "Hello.";
const PROFILE_CAPACITY: u32 = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("usage: profile_decode_safetensors <model_dir> <tokenizer_gguf>")?);
    let tokenizer_path = PathBuf::from(args.next().ok_or("usage: profile_decode_safetensors <model_dir> <tokenizer_gguf>")?);
    let n_profile_tokens: u32 = env::var("VF_PROFILE_TOKENS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8);

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

    let (model, host_embed, _hf) =
        LoadedModel::load_safetensors(&dev, &mut allocator, &model_dir)?;
    let cfg = model.config.clone();

    let tok_gguf = GgufFile::open(&tokenizer_path)?;
    let tokenizer = Tokenizer::from_gguf(&tok_gguf)?;
    let template = ChatTemplate::detect(&tok_gguf, &tokenizer);
    let prompt_tokens = template.render_first_turn(&tokenizer, SYSTEM_PROMPT, USER_PROMPT);

    println!(
        "model: {}\n  arch={} layers={} heads={}/{} hidden={} ffn={} vocab={}\n  prompt: {} tokens, profile_tokens: {}\n",
        model_dir.display(),
        cfg.architecture, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.hidden_dim, cfg.ffn_dim, cfg.vocab_size,
        prompt_tokens.len(), n_profile_tokens,
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

    // ---- Prefill (chunked, batched) ----
    let chunk_size = forward.max_prefill_tokens.max(1) as usize;
    let mut pos = 0u32;
    for chunk in prompt_tokens.chunks(chunk_size) {
        let mut chunk_embeds: Vec<f32> = Vec::with_capacity(chunk.len() * cfg.hidden_dim as usize);
        for &tid in chunk {
            let start = (tid as usize) * cfg.hidden_dim as usize;
            chunk_embeds.extend_from_slice(&host_embed[start..start + cfg.hidden_dim as usize]);
        }
        forward.prefill_batch(&dev, &registry, &cmd_ctx, &model, &chunk_embeds, chunk.len() as u32, pos, &[])?;
        pos += chunk.len() as u32;
    }
    println!("prefilled to pos={}\n", pos);

    // ---- Decode 1 warmup token (no profile capture, KV state advances) ----
    let mut last_logits = forward.logits()?;
    let next_id = argmax(&last_logits) as u32;
    let embd = embedding_row_host(&host_embed, &cfg, next_id);
    let _stats = forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos)?;
    last_logits = forward.logits()?;
    pos += 1;

    // ---- Profile N decode tokens, accumulate per-shader totals ----
    let mut totals: BTreeMap<String, (Duration, u32)> = BTreeMap::new();
    let mut token_totals: Vec<Duration> = Vec::new();
    let mut wall_totals: Vec<Duration> = Vec::new();
    for _ in 0..n_profile_tokens {
        let next_id = argmax(&last_logits) as u32;
        let embd = embedding_row_host(&host_embed, &cfg, next_id);
        let stats = forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos)?;
        last_logits = forward.logits()?;
        pos += 1;

        // Sum across this token's per-shader entries into the global totals.
        let mut tok_total = Duration::ZERO;
        for (name, (dur, count)) in &stats.per_shader {
            let entry = totals.entry(name.clone()).or_insert((Duration::ZERO, 0));
            entry.0 += *dur;
            entry.1 += *count;
            tok_total += *dur;
        }
        token_totals.push(tok_total);
        wall_totals.push(stats.total);
    }
    let avg_wall = wall_totals.iter().sum::<Duration>().as_secs_f64() / wall_totals.len() as f64;
    println!("\nWall time per forward_token (avg): {:.2} ms", avg_wall * 1000.0);

    let n = n_profile_tokens.max(1) as f64;
    let avg_token = token_totals.iter().sum::<Duration>().as_secs_f64() / n;
    let projected_dec = 1.0 / avg_token;

    println!("\n=== Per-shader breakdown (averaged over {} decode tokens at pos~{}) ===", n_profile_tokens, pos - 1);
    println!("{:<32} {:>10} {:>8} {:>10} {:>8}", "label", "avg ms", "count", "total ms", "% tok");
    println!("{}", "-".repeat(72));

    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0));  // sort by total time desc
    let total_token_ms = avg_token * 1000.0;
    for (name, (dur, count)) in &rows {
        let total_ms = dur.as_secs_f64() * 1000.0;
        let avg_ms = total_ms / n;
        let avg_call_us = total_ms * 1000.0 / *count as f64;
        let pct = if total_token_ms > 0.0 { 100.0 * (avg_ms / total_token_ms) } else { 0.0 };
        let _ = avg_call_us;
        println!("{:<32} {:>10.3} {:>8} {:>10.3} {:>7.1}%", name, avg_ms, count / n_profile_tokens.max(1), total_ms, pct);
    }
    println!("{}", "-".repeat(72));
    println!("{:<32} {:>10.3}                       100.0%", "Σ per-shader (token avg)", total_token_ms);
    println!("\nProjected decode rate: {:.1} tok/s (avg per-shader sum {:.2} ms/token)",
             projected_dec, total_token_ms);

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}

fn argmax(xs: &[f32]) -> usize {
    let mut idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best { best = x; idx = i; }
    }
    idx
}

fn embedding_row_host(host_embed: &[f32], cfg: &vulkanforge::backend::vulkan::gguf::ModelConfig, tid: u32) -> Vec<f32> {
    let h = cfg.hidden_dim as usize;
    let start = (tid as usize) * h;
    host_embed[start..start + h].to_vec()
}
