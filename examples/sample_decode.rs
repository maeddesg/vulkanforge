//! Phase-4D quick decode sampler — runs one prompt through any model
//! and prints the full generated text. Used to eyeball coherence
//! beyond the 80-char excerpt the bench shows.

use std::path::PathBuf;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat::ChatSession;
use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::GenerateConfig;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

const MAX_SEQ_LEN: u32 = 2048;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .expect("set VF_MODEL_PATH");
    let prompt = std::env::var("VF_PROMPT")
        .unwrap_or_else(|_| "Write a Python function that checks if a number is prime.".to_string());
    let max_tokens: u32 = std::env::var("VF_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

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

    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let kv_cache = KvCache::new(
        &dev.device, &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: MAX_SEQ_LEN,
        },
    )?;
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;
    let template = ChatTemplate::detect(&gguf, &tokenizer);
    println!(
        "model={}\narch={} rope={:?} qk_norm={} template={:?} eos_id={}",
        model_path.display(), cfg.architecture, cfg.rope_variant,
        cfg.has_qk_norm, template, tokenizer.eos_id,
    );
    println!("---\nPROMPT: {prompt}\n---");
    let mut session = ChatSession::new_with_template(
        forward, "You are a helpful assistant.", template,
    );
    let cfg_g = GenerateConfig {
        max_tokens, print_stream: false, think_filter: false,
    };
    let turn = session.send(
        &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
        &prompt, &cfg_g,
    )?;
    println!("OUTPUT ({} gen tokens, eos={}):\n{}",
        turn.generated_tokens, turn.stopped_on_eos, turn.generated_text);
    println!("---\nprefill={:.1} tok/s decode={:.1} tok/s",
        turn.prefill_tok_s(), turn.decode_tok_s());

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}
