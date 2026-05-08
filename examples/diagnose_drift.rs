//! Phase-3E drift debugger.
//!
//! For a given prompt:
//! 1. Run token-by-token GEMV prefill → logits_a
//! 2. Run prefill_batch GEMM → logits_b
//! 3. Compare:
//!     - argmax (greedy first-token equality)
//!     - top-5 IDs + their decoded strings
//!     - max_abs_err and L2 norm of (logits_a - logits_b)
//!     - per-position Q-projection deviation (layer 0 only)
//!
//! Output is enough to localise whether the bug is in the GEMM dispatch
//! itself, the per-layer orchestration, or just expected Q8_1 noise.

use std::path::PathBuf;

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
use vulkanforge::backend::vulkan::tokenizer::{apply_chat_template, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::var("VF_PROMPT")
        .unwrap_or_else(|_| "What is 2 + 2?".to_string());
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
    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let prompt_tokens = apply_chat_template(&tokenizer, &prompt, None);
    let seq_len = prompt_tokens.len() as u32;
    println!("Prompt: {prompt:?}");
    println!("Tokens: {} ids, first 8 = {:?}", seq_len, &prompt_tokens[..seq_len.min(8) as usize]);

    // Path A: token-by-token GEMV.
    let kv_a = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 512,
    })?;
    let mut fwd_a = Forward::new_with_prefill(
        &dev, &mut allocator, kv_a, cfg.clone(), None, /* max_pp = */ 1,
    )?;
    fwd_a.kv_cache.reset();
    for (pos, &tid) in prompt_tokens.iter().enumerate() {
        let embd = embedding_row(&gguf, &cfg, tid)?;
        fwd_a.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos as u32)?;
    }
    let logits_a = fwd_a.logits()?;

    // Path B: prefill_batch GEMM.
    let kv_b = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 512,
    })?;
    let mut fwd_b = Forward::new_with_prefill(
        &dev, &mut allocator, kv_b, cfg.clone(), None, /* max_pp = */ seq_len,
    )?;
    fwd_b.kv_cache.reset();
    let mut all_embeds: Vec<f32> = Vec::with_capacity((seq_len * cfg.hidden_dim) as usize);
    for &tid in &prompt_tokens {
        all_embeds.extend(embedding_row(&gguf, &cfg, tid)?);
    }
    fwd_b.prefill_batch(&dev, &registry, &cmd_ctx, &model, &all_embeds, seq_len, 0, &[])?;
    let logits_b = fwd_b.logits()?;

    println!("\n=== Logits comparison ===");
    let max_abs: f32 = logits_a.iter().zip(&logits_b)
        .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let mean_abs: f64 = logits_a.iter().zip(&logits_b)
        .map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / logits_a.len() as f64;
    let l2: f64 = logits_a.iter().zip(&logits_b)
        .map(|(a, b)| { let d = (a - b) as f64; d * d }).sum::<f64>().sqrt();
    println!("  max_abs_err = {max_abs:.4}");
    println!("  mean_abs_err = {mean_abs:.4}");
    println!("  l2(diff)    = {l2:.4}");

    let argmax = |v: &[f32]| -> u32 {
        v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
    };
    let top1_a = argmax(&logits_a);
    let top1_b = argmax(&logits_b);
    println!(
        "  top1 a = {top1_a} {:?}",
        tokenizer.token_str(top1_a).unwrap_or("?"),
    );
    println!(
        "  top1 b = {top1_b} {:?}",
        tokenizer.token_str(top1_b).unwrap_or("?"),
    );
    println!("  argmax_match = {}", top1_a == top1_b);

    let top_k = |v: &[f32], k: usize| -> Vec<(u32, f32)> {
        let mut idx: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        idx.into_iter().take(k).map(|(i, x)| (i as u32, x)).collect()
    };
    let top5_a = top_k(&logits_a, 5);
    let top5_b = top_k(&logits_b, 5);
    println!("\n  top-5 path A (token-by-token):");
    for (id, l) in &top5_a {
        println!(
            "    id={:>6}  logit={:7.3}  {:?}",
            id, l, tokenizer.token_str(*id).unwrap_or("?")
        );
    }
    println!("  top-5 path B (prefill_batch):");
    for (id, l) in &top5_b {
        println!(
            "    id={:>6}  logit={:7.3}  {:?}",
            id, l, tokenizer.token_str(*id).unwrap_or("?")
        );
    }
    let overlap = top5_a.iter()
        .filter(|(id, _)| top5_b.iter().any(|(b, _)| b == id))
        .count();
    println!("  top-5 overlap = {overlap}/5");

    fwd_a.destroy(&dev.device, &mut allocator);
    fwd_b.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}
