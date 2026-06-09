//! Minimal RGP-capture driver — designed to be wrapped in
//! `MESA_VK_TRACE=rgp` (or `RADV_THREAD_TRACE=...`) so the resulting
//! `.rgp` is small but representative of the decode hot path.
//!
//! Workload:
//!   1. Load Qwen3-8B Q4_K_M.
//!   2. Tokenise the chat-template form of "Explain what a mutex is."
//!   3. Token-by-token GEMV prefill (NOT prefill_batch — we want to
//!      profile the decode-equivalent dispatch path, not the
//!      Phase-3E batched GEMM, since Phase 4 targets decode).
//!   4. Decode 20 tokens.
//!   5. Clean shutdown.
//!
//! Validation layers are intentionally **off** (correctness already
//! verified by the 48-test suite; the capture should reflect realistic
//! perf, not validation-instrumented perf).
//!
//! Run with one of these env-var combinations:
//!
//!   MESA_VK_TRACE=rgp \
//!       cargo run --release --example rgp_capture
//!
//!   RADV_THREAD_TRACE=0 RADV_THREAD_TRACE_TRIGGER=/tmp/trigger \
//!       cargo run --release --example rgp_capture
//!
//! The `.rgp` file lands in CWD (or wherever Mesa decides to put it).

use std::path::PathBuf;
use std::time::Instant;

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

const PROMPT: &str = "Explain what a mutex is in one sentence.";
// Only one decode forward — RADV's per-submit RGP capture writes
// ~144 MB per submit, so a single representative forward keeps the
// disk usage at one file.
const DECODE_TOKENS: u32 = 1;
const MAX_SEQ_LEN: u32 = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[rgp_capture] start");
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
    let model = LoadedModel::load(&dev, &mut allocator, &gguf, None)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let kv_cache = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        max_seq_len: MAX_SEQ_LEN,
        per_layer_head_dim: cfg
            .gemma4
            .as_ref()
            .map(|g| g.layers.iter().map(|s| s.head_dim).collect()),
        per_layer_n_kv_heads: cfg
            .gemma4
            .as_ref()
            .map(|g| g.layers.iter().map(|s| s.n_kv_heads).collect::<Vec<_>>())
            .or_else(|| cfg.qwen35.as_ref().map(|q| {
                (0..q.block_count)
                    .map(|l| if q.is_full_attention_layer(l) {
                        q.n_head_kv_full_attn
                    } else { 0 })
                    .collect::<Vec<_>>()
            })),
    })?;
    // Capture binary uses the token-by-token GEMV path, so a small
    // max_prefill_tokens keeps the descriptor pool from blowing up.
    let mut forward = Forward::new_with_prefill(
        &dev, &mut allocator, kv_cache, cfg.clone(), None, /*max_pp=*/ 1,
    )?;

    let prompt_tokens = apply_chat_template(&tokenizer, PROMPT, None);
    eprintln!(
        "[rgp_capture] prompt = {:?}, {} tokens, will decode {} more",
        PROMPT, prompt_tokens.len(), DECODE_TOKENS,
    );

    // ---- Prefill (token-by-token GEMV) ----
    let t0 = Instant::now();
    let mut pos: u32 = 0;
    for &tid in &prompt_tokens {
        let embd = embedding_row(&gguf, &cfg, tid)?;
        forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos, tid)?;
        pos += 1;
    }
    eprintln!(
        "[rgp_capture] prefill done: {} tokens in {:.1} ms",
        prompt_tokens.len(),
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    // ---- Decode ----
    let t1 = Instant::now();
    for _step in 0..DECODE_TOKENS {
        let logits = forward.logits()?;
        let next_id = argmax(&logits) as u32;
        if tokenizer.is_eos(next_id) {
            break;
        }
        let embd = embedding_row(&gguf, &cfg, next_id)?;
        forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos, next_id)?;
        pos += 1;
    }
    eprintln!(
        "[rgp_capture] decode done: {} steps in {:.1} ms",
        DECODE_TOKENS,
        t1.elapsed().as_secs_f64() * 1000.0,
    );

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    eprintln!("[rgp_capture] clean exit");
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}
