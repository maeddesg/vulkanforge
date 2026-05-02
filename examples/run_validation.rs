//! Phase-2D 5-prompt regression suite — moved out of `main.rs` in
//! Phase 3B when the binary became an interactive chat REPL.
//!
//! Drives `decode::generate` (single-turn, KV reset between prompts)
//! against a fixed list of prompts and prints per-prompt + median
//! tok/s. Used in CI / regression checks; behaviour-equivalent to the
//! Phase-2D `cargo run` default.

use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::{generate, GenerateConfig, GenerateResult};
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

const MAX_SEQ_LEN: u32 = 2048;

const VALIDATION_PROMPTS: &[&str] = &[
    "Explain what a mutex is in one sentence.",
    "Write a haiku about programming.",
    "What is 2 + 2?",
    "Translate 'hello world' to German.",
    "List three prime numbers.",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge — 5-prompt validation suite");
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

    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let load_start = Instant::now();
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "  loaded {} tensors, {:.2} GiB in {:.1} s",
        model.tensors.len(),
        (model.bytes_uploaded as f64) / (1024.0 * 1024.0 * 1024.0),
        load_start.elapsed().as_secs_f64(),
    );
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: MAX_SEQ_LEN,
        },
    )?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;

    let max_tokens: u32 = std::env::var("VF_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let mut results: Vec<(String, GenerateResult)> = Vec::new();
    println!("\n=== 5-Prompt Validation Suite ===");
    for (i, &prompt) in VALIDATION_PROMPTS.iter().enumerate() {
        println!("\n--- Prompt {} ---", i + 1);
        println!("> {prompt}");
        let r = generate(
            &mut forward, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            prompt,
            &GenerateConfig { max_tokens, print_stream: false, think_filter: false, sampling: Default::default() },
        )?;
        println!("\n{}", r.generated_text);
        println!(
            "  [prompt={} gen={} stopped_on_eos={} prefill={:.0} tok/s decode={:.1} tok/s]",
            r.prompt_tokens,
            r.generated_tokens,
            r.stopped_on_eos,
            r.prefill_tok_s(),
            r.decode_tok_s(),
        );
        results.push((prompt.to_string(), r));
    }

    println!("\n=== Summary ===");
    println!(
        "{:<54}  {:>6}  {:>5}  {:>9}  {:>10}",
        "Prompt", "Prompt", "Gen", "Prefill", "Decode"
    );
    println!(
        "{:<54}  {:>6}  {:>5}  {:>9}  {:>10}",
        "", "(tok)", "(tok)", "(tok/s)", "(tok/s)"
    );
    for (p, r) in &results {
        let short = if p.len() > 50 { format!("{}…", &p[..49]) } else { p.clone() };
        println!(
            "{:<54}  {:>6}  {:>5}  {:>9.0}  {:>10.1}",
            short, r.prompt_tokens, r.generated_tokens,
            r.prefill_tok_s(), r.decode_tok_s(),
        );
    }
    let mut decodes: Vec<f64> = results.iter().map(|(_, r)| r.decode_tok_s()).collect();
    let mut prefills: Vec<f64> = results.iter().map(|(_, r)| r.prefill_tok_s()).collect();
    decodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    prefills.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_d = decodes[decodes.len() / 2];
    let med_p = prefills[prefills.len() / 2];
    println!(
        "{:<54}  {:>6}  {:>5}  {:>9.0}  {:>10.1}",
        "MEDIAN", "—", "—", med_p, med_d,
    );

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}
