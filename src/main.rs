//! VulkanForge — Phase 2D demo driver: tokenizer + decode loop +
//! 5-prompt validation suite.
//!
//! Modes (env-driven, mutually exclusive):
//!   default              → 5-prompt validation suite, prints
//!                          per-prompt summary + median tok/s.
//!   VF_PROMPT="..."     → run a single user prompt, stream output
//!                          to stdout.
//!   VF_TRACE_L0=1        → Phase-2C intra-layer-0 debug trace
//!                          (kept for regression-debugging).
//!   VF_LAYER_WALK=1      → Phase-2C per-layer NaN walk on the raw
//!                          token-9707 embedding (kept for the same
//!                          reason).
//!   VF_FULL=1            → one full forward + LM-head dump (no
//!                          tokenizer, no decode loop).
//!
//! Other env vars:
//!   VF_MODEL_PATH=...    → override model file (default
//!                          $HOME/models/Qwen3-8B-Q4_K_M.gguf).
//!   VF_MAX_TOKENS=N      → cap decode length (default 200).
//!   VF_PROFILE=1         → enable shader timestamp profiler in
//!                          forward passes (slower, prints
//!                          per-shader breakdown for VF_FULL).

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
    println!("VulkanForge v0.1.0 — Phase 2D decode driver");

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
    println!(
        "  parsed in {:.1} ms",
        parse_start.elapsed().as_secs_f64() * 1000.0
    );

    let load_start = Instant::now();
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "✅ {} tensors, {:.2} GiB in {:.1} s",
        model.tensors.len(),
        (model.bytes_uploaded as f64) / (1024.0 * 1024.0 * 1024.0),
        load_start.elapsed().as_secs_f64()
    );

    let tok_start = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    println!(
        "✅ tokenizer: {} tokens in {:.1} ms",
        tokenizer.vocab_size(),
        tok_start.elapsed().as_secs_f64() * 1000.0
    );

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

    let profiler = if std::env::var("VF_PROFILE").is_ok() {
        Some(vulkanforge::backend::vulkan::profiler::ShaderProfiler::new(
            &dev.instance,
            dev.physical_device,
            dev.queue_family_index,
            &dev.device,
            1024,
        )?)
    } else {
        None
    };
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), profiler)?;

    let max_tokens: u32 = std::env::var("VF_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    if let Ok(prompt) = std::env::var("VF_PROMPT") {
        run_single(
            &mut forward, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            &prompt, max_tokens,
        )?;
    } else {
        run_validation_suite(
            &mut forward, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            max_tokens,
        )?;
    }

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    gguf: &GgufFile,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Single prompt ===");
    println!("> {prompt}\n");
    let cfg_g = GenerateConfig {
        max_tokens,
        print_stream: true,
    };
    let r = generate(
        forward, dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
        prompt, &cfg_g,
    )?;
    print_result_summary(&r);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_validation_suite(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    gguf: &GgufFile,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    max_tokens: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 5-Prompt Validation Suite ===");
    let mut results: Vec<(String, GenerateResult)> = Vec::new();

    for (i, &prompt) in VALIDATION_PROMPTS.iter().enumerate() {
        println!("\n--- Prompt {} ---", i + 1);
        println!("> {prompt}");
        let r = generate(
            forward, dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
            prompt,
            &GenerateConfig {
                max_tokens,
                print_stream: false,
            },
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
            short,
            r.prompt_tokens,
            r.generated_tokens,
            r.prefill_tok_s(),
            r.decode_tok_s(),
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
    Ok(())
}

fn print_result_summary(r: &GenerateResult) {
    println!("\n--- summary ---");
    println!(
        "prompt_tokens   = {}\ngenerated_tokens= {}\nstopped_on_eos  = {}\nprefill_time    = {:.2} ms ({:.0} tok/s)\ndecode_time     = {:.2} ms ({:.1} tok/s)",
        r.prompt_tokens,
        r.generated_tokens,
        r.stopped_on_eos,
        r.prefill_time.as_secs_f64() * 1000.0,
        r.prefill_tok_s(),
        r.decode_time.as_secs_f64() * 1000.0,
        r.decode_tok_s(),
    );
}
