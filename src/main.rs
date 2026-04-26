// See lib.rs for rationale.
#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::ptr_arg,
    clippy::print_literal,
    clippy::op_ref,
    clippy::needless_question_mark,
    clippy::match_like_matches_macro,
    clippy::doc_lazy_continuation,
    clippy::collapsible_else_if,
)]

//! VulkanForge — Phase 3B interactive chat REPL.
//!
//! Loads Qwen3-8B once at startup, then drops into a `>` prompt
//! that runs each user turn through [`ChatSession::send_streaming`].
//! KV cache is shared across turns; slash-commands let the user
//! reset, query, or quit.
//!
//! Slash-commands:
//!   /reset    — clear KV cache + history, start fresh
//!   /quit     — exit cleanly
//!   /stats    — context usage, turn count, last decode tok/s
//!   /think    — toggle the think-filter (default ON)
//!   /help     — list commands
//!
//! Env vars:
//!   VF_MODEL_PATH        path to GGUF (default ~/models/Qwen3-8B-Q4_K_M.gguf)
//!   VF_MAX_TOKENS        per-turn cap (default 400)
//!   VF_SYSTEM            system prompt (default "You are a helpful assistant.")
//!   VF_NO_THINK_FILTER   set to disable the think-filter
//!   VF_PROMPT="..."     run a single turn non-interactively (CI / regression)

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat::{ChatError, ChatSession, TurnResult};
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
const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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
    let (registry, pipelines_loaded) =
        PipelineRegistry::new(&dev.device, cache_path.as_deref())?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });
    let max_tokens: u32 = std::env::var("VF_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let system_prompt = std::env::var("VF_SYSTEM").unwrap_or_else(|_| DEFAULT_SYSTEM.to_string());
    let mut think_filter = std::env::var("VF_NO_THINK_FILTER").is_err();

    let load_start = Instant::now();
    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
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
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;

    print_banner(
        &model_path,
        &cfg,
        model.bytes_uploaded,
        load_start.elapsed().as_secs_f64(),
        pipelines_loaded,
        think_filter,
        max_tokens,
    );

    let mut session = ChatSession::new(forward, system_prompt.clone());
    let mut last_turn: Option<TurnResult> = None;

    if let Ok(prompt) = std::env::var("VF_PROMPT") {
        // Non-interactive: run one turn, print stats, exit.
        match send_turn(&mut session, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg,
                        &tokenizer, &prompt, max_tokens, think_filter)
        {
            Ok(r) => last_turn = Some(r),
            Err(ChatError::ContextOverflow { .. }) => {
                eprintln!("\n[context overflow]");
            }
            Err(e) => return Err(Box::new(e)),
        }
        if let Some(r) = &last_turn {
            print_inline_stats(r);
        }
    } else {
        // Interactive REPL.
        let stdin = std::io::stdin();
        let mut lines = stdin.lock().lines();
        loop {
            print!("\n> ");
            std::io::stdout().flush().ok();
            let line = match lines.next() {
                Some(Ok(s)) => s,
                Some(Err(_)) | None => break, // EOF (Ctrl-D)
            };
            let trimmed = line.trim();
            match trimmed {
                "" => continue,
                "/quit" | "/q" | "/exit" => break,
                "/help" | "/h" => print_help(),
                "/reset" => {
                    session.reset();
                    last_turn = None;
                    println!("  (context cleared)");
                }
                "/stats" => print_stats(&session, last_turn.as_ref()),
                "/think" => {
                    think_filter = !think_filter;
                    println!("  think-filter: {}", if think_filter { "on" } else { "off" });
                }
                _ => {
                    println!();
                    match send_turn(
                        &mut session, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg,
                        &tokenizer, trimmed, max_tokens, think_filter,
                    ) {
                        Ok(r) => {
                            print_inline_stats(&r);
                            last_turn = Some(r);
                        }
                        Err(ChatError::ContextOverflow { current_pos, needed, max_seq_len }) => {
                            eprintln!(
                                "\n  [context overflow: {current_pos} + {needed} > {max_seq_len}]"
                            );
                            eprintln!("  Use /reset to start a new conversation.");
                        }
                        Err(e) => {
                            eprintln!("\n  [error: {e}]");
                        }
                    }
                }
            }
        }
    }

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn send_turn(
    session: &mut ChatSession,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    gguf: &GgufFile,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    user_message: &str,
    max_tokens: u32,
    think_filter: bool,
) -> Result<TurnResult, ChatError> {
    let cfg_g = GenerateConfig {
        max_tokens,
        print_stream: false,
        think_filter,
    };
    session.send_streaming(
        dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
        user_message, &cfg_g,
        |visible| {
            print!("{visible}");
            std::io::stdout().flush().ok();
        },
    )
}

fn print_banner(
    model_path: &PathBuf,
    cfg: &ModelConfig,
    bytes_uploaded: u64,
    load_secs: f64,
    pipelines_loaded: usize,
    think_filter: bool,
    max_tokens: u32,
) {
    println!("\nVulkanForge v0.1.0 — Phase 3B chat REPL");
    println!("  Model:   {}", model_path.display());
    println!(
        "    {:.2} GiB · {} layers · hidden={} · heads={} · kv_heads={} · head_dim={}",
        (bytes_uploaded as f64) / (1024.0 * 1024.0 * 1024.0),
        cfg.n_layers,
        cfg.hidden_dim,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    );
    println!(
        "    vocab={} · ctx_max={} · rope_freq_base={:.0}",
        cfg.vocab_size, MAX_SEQ_LEN, cfg.rope_freq_base,
    );
    println!("  Loaded in {:.1} s", load_secs);
    println!(
        "  Pipeline cache: {} bytes loaded · {} shaders ready",
        pipelines_loaded,
        vulkanforge::backend::vulkan::shaders::ALL_SHADERS.len(),
    );
    println!(
        "  think-filter: {} · max_tokens/turn: {}",
        if think_filter { "on" } else { "off" },
        max_tokens,
    );
    println!("  Type /help for commands, /quit to exit.");
}

fn print_help() {
    println!("  /reset     clear KV cache + history");
    println!("  /quit      exit");
    println!("  /stats     show context usage and last-turn timing");
    println!("  /think     toggle <think>…</think> filter");
    println!("  /help      this list");
}

fn print_stats(session: &ChatSession, last: Option<&TurnResult>) {
    println!(
        "  Context: {}/{} tokens used  ({} free)",
        session.current_pos,
        session.max_seq_len(),
        session.remaining_tokens(),
    );
    println!("  Turns:   {}", session.turn_count);
    if let Some(r) = last {
        println!(
            "  Last turn: {} prompt → {} gen, prefill {:.0} tok/s, decode {:.1} tok/s, stopped_on_eos={}",
            r.prompt_tokens,
            r.generated_tokens,
            r.prefill_tok_s(),
            r.decode_tok_s(),
            r.stopped_on_eos,
        );
    }
}

fn print_inline_stats(r: &TurnResult) {
    println!(
        "\n  [{} prompt, {} gen, prefill {:.0} tok/s, decode {:.1} tok/s{}]",
        r.prompt_tokens,
        r.generated_tokens,
        r.prefill_tok_s(),
        r.decode_tok_s(),
        if r.stopped_on_eos { "" } else { ", capped" },
    );
}
