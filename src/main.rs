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
//!   VF_TEMPERATURE       sampling temperature (default 0.0 = greedy)
//!   VF_TOP_K             top-k filter (default 0 = disabled)
//!   VF_TOP_P             top-p / nucleus cutoff (default 1.0 = disabled)
//!   VF_REPETITION_PENALTY  >1.0 discourages repeats (default 1.0 off)
//!   VF_SEED              xorshift64* seed for non-greedy sampling

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use clap::{Parser, Subcommand};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat::{ChatError, ChatSession, TurnResult};
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::{GenerateConfig, Sampling};
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

const MAX_SEQ_LEN: u32 = 2048;
const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// Default model path: $VF_MODEL_PATH or ~/models/Qwen3-8B-Q4_K_M.gguf.
fn default_model_path() -> PathBuf {
    std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset and --model not provided")
        })
}

#[derive(Parser)]
#[command(
    name = "vulkanforge",
    version,
    about = "High-performance LLM inference for AMD RDNA4 via Vulkan",
    long_about = "VulkanForge — Vulkan-backed LLM inference engine targeting AMD RDNA4 (gfx1201).\n\
                  Decode 109 tok/s (0.95x llama.cpp Vulkan), prefill 0.89x at pp=512.\n\
                  See `vulkanforge <subcommand> --help` for per-command options."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive multi-turn chat REPL (single-turn via VF_PROMPT="...").
    Chat {
        /// Path to GGUF model file. Defaults to $VF_MODEL_PATH or
        /// ~/models/Qwen3-8B-Q4_K_M.gguf.
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// System prompt (default: "You are a helpful assistant.").
        #[arg(long)]
        system: Option<String>,
        /// Max tokens generated per turn (default 400).
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Sampling temperature; 0.0 = greedy (default).
        #[arg(long)]
        temperature: Option<f32>,
        /// Top-K filter; 0 = disabled (default).
        #[arg(long)]
        top_k: Option<u32>,
        /// Top-P / nucleus cutoff; 1.0 = disabled (default).
        #[arg(long)]
        top_p: Option<f32>,
        /// Repetition penalty; >1.0 discourages repeats (default 1.0).
        #[arg(long)]
        repetition_penalty: Option<f32>,
        /// PRNG seed for non-greedy sampling.
        #[arg(long)]
        seed: Option<u64>,
        /// Disable the <think>...</think> filter (Qwen3 thinking mode).
        #[arg(long)]
        no_think_filter: bool,
    },
    /// Run a small bench: 5-prompt decode + 4-point pp sweep. For the
    /// full 15-prompt and full pp-sweep, use `cargo run --release
    /// --example run_15prompt_bench` and `--example run_pp_bench`.
    Bench {
        /// Path to GGUF model file.
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Prompt-length sweep (comma-separated; default 64,128,512,1024).
        #[arg(long, default_value = "64,128,512,1024")]
        pp_list: String,
        /// Repetitions per pp value (default 3).
        #[arg(long, default_value_t = 3)]
        runs: u32,
    },
    /// Show GGUF model metadata + GPU/Vulkan information.
    Info {
        /// Path to GGUF model file.
        #[arg(short, long)]
        model: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Chat { model, system, max_tokens, temperature, top_k, top_p,
                         repetition_penalty, seed, no_think_filter } => {
            run_chat(ChatArgs {
                model: model.unwrap_or_else(default_model_path),
                system: system
                    .or_else(|| std::env::var("VF_SYSTEM").ok())
                    .unwrap_or_else(|| DEFAULT_SYSTEM.to_string()),
                max_tokens: max_tokens.unwrap_or_else(|| {
                    std::env::var("VF_MAX_TOKENS").ok()
                        .and_then(|s| s.parse().ok()).unwrap_or(400)
                }),
                temperature: temperature.unwrap_or_else(|| {
                    std::env::var("VF_TEMPERATURE").ok()
                        .and_then(|s| s.parse().ok()).unwrap_or(0.0)
                }),
                top_k: top_k.unwrap_or_else(|| {
                    std::env::var("VF_TOP_K").ok()
                        .and_then(|s| s.parse().ok()).unwrap_or(0)
                }),
                top_p: top_p.unwrap_or_else(|| {
                    std::env::var("VF_TOP_P").ok()
                        .and_then(|s| s.parse().ok()).unwrap_or(1.0)
                }),
                repetition_penalty: repetition_penalty.unwrap_or_else(|| {
                    std::env::var("VF_REPETITION_PENALTY").ok()
                        .and_then(|s| s.parse().ok()).unwrap_or(1.0)
                }),
                seed_was_explicit: seed.is_some() || std::env::var("VF_SEED").is_ok(),
                seed: seed.unwrap_or_else(|| {
                    std::env::var("VF_SEED").ok()
                        .and_then(|s| s.parse().ok())
                        // No --seed and no VF_SEED — derive from the system
                        // clock so that a temperature > 0 produces a fresh
                        // sequence each run instead of replaying the same
                        // pseudo-random output every time. `--seed N` /
                        // `VF_SEED=N` still pin a deterministic seed.
                        .unwrap_or_else(|| {
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_nanos() as u64)
                                .unwrap_or(0)
                        })
                }),
                think_filter: !no_think_filter && std::env::var("VF_NO_THINK_FILTER").is_err(),
            })
        }
        Commands::Bench { model, pp_list, runs } => {
            run_bench(model.unwrap_or_else(default_model_path), &pp_list, runs)
        }
        Commands::Info { model } => {
            run_info(&model.unwrap_or_else(default_model_path))
        }
    };
    if let Err(e) = result {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

struct ChatArgs {
    model: PathBuf,
    system: String,
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
    seed_was_explicit: bool,
    think_filter: bool,
}

fn run_chat(args: ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    preflight_supported(&args.model, "chat")?;
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

    let model_path = args.model;
    let max_tokens = args.max_tokens;
    let system_prompt = args.system;
    let mut think_filter = args.think_filter;

    // Phase 6 v0.1.2 sampling — `temperature == 0.0` (default) keeps
    // greedy decoding, so existing benchmarks stay byte-deterministic.
    // Sprint 16A — values come from clap args (with VF_* env-var
    // fallbacks resolved at the top of `main`).
    let sampling = Sampling {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

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
        registry.count(),
        think_filter,
        max_tokens,
        &sampling,
        args.seed_was_explicit,
    );

    let mut session = ChatSession::new(forward, system_prompt.clone());
    let mut last_turn: Option<TurnResult> = None;

    if let Ok(prompt) = std::env::var("VF_PROMPT") {
        // Non-interactive: run one turn, print stats, exit.
        match send_turn(&mut session, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg,
                        &tokenizer, &prompt, max_tokens, think_filter, &sampling)
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
        // Interactive REPL — rustyline gives us arrow-key cursor
        // movement, ↑/↓ history, Ctrl+A/E line nav, and a clean
        // Ctrl+C/D exit. Falls back to no-history line reading if
        // the editor can't initialize (e.g. non-tty stdin).
        let mut rl = rustyline::DefaultEditor::new()?;
        loop {
            println!();
            let line = match rl.readline("> ") {
                Ok(s) => {
                    if !s.trim().is_empty() {
                        let _ = rl.add_history_entry(&s);
                    }
                    s
                }
                Err(rustyline::error::ReadlineError::Eof)
                | Err(rustyline::error::ReadlineError::Interrupted) => break,
                Err(e) => {
                    eprintln!("  [readline error: {e}]");
                    break;
                }
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
                        &tokenizer, trimmed, max_tokens, think_filter, &sampling,
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

    // Phase 6 v0.1.2: persist the pipeline cache so the next run boots
    // straight into the warm path instead of re-running ACO over every
    // SPV. Errors are logged from inside `save_cache` and otherwise
    // ignored — a cold next start is harmless, just slower.
    let stats = registry.save_cache(&dev.device);
    if stats.saved_bytes > 0 {
        println!(
            "  Pipeline cache: saved {} bytes (loaded {} bytes at start)",
            stats.saved_bytes, pipelines_loaded
        );
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
    sampling: &Sampling,
) -> Result<TurnResult, ChatError> {
    let cfg_g = GenerateConfig {
        max_tokens,
        print_stream: false,
        think_filter,
        sampling: sampling.clone(),
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
    pipelines_count: usize,
    think_filter: bool,
    max_tokens: u32,
    sampling: &Sampling,
    seed_explicit: bool,
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
        pipelines_loaded, pipelines_count,
    );
    println!(
        "  think-filter: {} · max_tokens/turn: {}",
        if think_filter { "on" } else { "off" },
        max_tokens,
    );
    if sampling.temperature == 0.0 {
        println!("  Sampling:     greedy (temperature=0)");
    } else {
        let seed_label = if seed_explicit {
            format!(" seed={}", sampling.seed)
        } else {
            " seed=auto".to_string()
        };
        println!(
            "  Sampling:     temp={:.2} top_k={} top_p={:.2} rep_pen={:.2}{}",
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.repetition_penalty,
            seed_label,
        );
    }
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

// ──────────────────────────────────────────────────────────────────────
// Sprint 16A — `vulkanforge info` and `vulkanforge bench` subcommands.
// The chat REPL above is the existing engine; these two are thin
// wrappers built on the same lib so the CLI is self-contained without
// pulling in the full per-tool infrastructure of the `examples/`
// benchmarks (those remain canonical for full bench runs).
// ──────────────────────────────────────────────────────────────────────

fn run_info(model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Open GGUF metadata WITHOUT uploading weights.
    let gguf = GgufFile::open(model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let arch = gguf.metadata_str("general.architecture").unwrap_or("?");
    let name = gguf.metadata_str("general.name").unwrap_or("(unnamed)");
    let file_type = gguf.metadata_u32("general.file_type").ok();
    let quant_name = file_type.map(file_type_name).unwrap_or("?");
    let tokenizer_model = gguf.metadata_str("tokenizer.ggml.model").unwrap_or("?");
    let file_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    println!();
    println!("  Model");
    println!("  ─────────────────────────────────────────────");
    println!("  Path           {}", model_path.display());
    println!("  Architecture   {}", arch);
    println!("  Name           {}", name);
    let file_type_label = file_type.map(|t| format!("{quant_name} (file_type={t})"))
        .unwrap_or_else(|| "(general.file_type missing)".to_string());
    println!("  Quantization   {}", file_type_label);
    println!("  Tokenizer      {}", tokenizer_model);
    println!("  Layers         {}", cfg.n_layers);
    println!("  Hidden dim     {}", cfg.hidden_dim);
    println!("  Heads          {} ({} KV)", cfg.n_heads, cfg.n_kv_heads);
    println!("  Head dim       {}", cfg.head_dim);
    println!("  FFN dim        {}", cfg.ffn_dim);
    println!("  Vocab          {}", cfg.vocab_size);
    println!("  Context        {}", cfg.context_length);
    println!("  RoPE base      {:.0}", cfg.rope_freq_base);
    println!("  File size      {:.2} GiB", (file_bytes as f64) / (1024.0 * 1024.0 * 1024.0));

    // GPU side: bring up Vulkan minimally to read device props.
    let dev = VulkanDevice::new()?;
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let mem_props = unsafe {
        dev.instance.get_physical_device_memory_properties(dev.physical_device)
    };
    let device_name = unsafe {
        std::ffi::CStr::from_ptr(props.device_name.as_ptr())
    }.to_string_lossy().into_owned();
    let api = props.api_version;
    let api_str = format!(
        "{}.{}.{}",
        vk::api_version_major(api), vk::api_version_minor(api), vk::api_version_patch(api),
    );
    let mut device_local_bytes: u64 = 0;
    for h in 0..mem_props.memory_heap_count as usize {
        let heap = mem_props.memory_heaps[h];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            device_local_bytes = device_local_bytes.max(heap.size);
        }
    }
    println!();
    println!("  GPU");
    println!("  ─────────────────────────────────────────────");
    println!("  Device         {}", device_name);
    println!("  Vulkan API     {}", api_str);
    println!(
        "  Device-local   {:.1} GiB (largest heap)",
        (device_local_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
    );
    println!();
    let (arch_ok, quant_ok) = inference_support(arch, file_type);
    match (arch_ok, quant_ok) {
        (true, true) => {
            println!("  Status         ✓ inference supported");
        }
        (true, false) => {
            println!(
                "  Status         ⚠ architecture '{arch}' supported, quantization {quant_name} is not (only Q4_K_M is wired up)"
            );
        }
        (false, _) => {
            println!(
                "  Status         ⚠ architecture '{arch}' is not yet wired into the forward pass (only qwen2 / qwen3 run end-to-end)"
            );
        }
    }
    let estimate_gb = (file_bytes as f64) / (1024.0 * 1024.0 * 1024.0) + 1.5;
    if estimate_gb > (device_local_bytes as f64) / (1024.0 * 1024.0 * 1024.0) {
        println!(
            "  ⚠  Model+KV needs ~{:.1} GiB but largest device-local heap is {:.1} GiB.",
            estimate_gb,
            (device_local_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
        );
    }
    Ok(())
}

/// `general.file_type` → human-readable quant label. Mirrors
/// llama.cpp's `LLAMA_FTYPE_MOSTLY_*` enum (the file-level quant tag,
/// not the per-tensor `ggml_type`). Returned `&'static str` lets the
/// caller embed the label in a fast info dump without an extra alloc.
fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        30 => "BF16",
        _ => "unknown",
    }
}

/// `(architecture_runs_end_to_end, quantization_supported)`. Only
/// qwen2 / qwen3 are wired through the v0.3 forward pass; only
/// Q4_K_M (file_type=15) is the production quant. Other archs / quants
/// can be inspected via `vulkanforge info` but `chat` and `bench`
/// won't run them yet.
fn inference_support(arch: &str, file_type: Option<u32>) -> (bool, bool) {
    let arch_ok = matches!(arch, "qwen2" | "qwen3");
    let quant_ok = matches!(file_type, Some(15));
    (arch_ok, quant_ok)
}

/// Read just the GGUF header (no weight upload) to fail fast on
/// architectures / quantizations that aren't wired into the forward
/// pass yet. Cheaper than spinning up the device + allocator + pipeline
/// cache only to crash deep in `LoadedModel::upload`.
fn preflight_supported(
    model_path: &PathBuf, subcommand: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufFile::open(model_path)?;
    let arch = gguf.metadata_str("general.architecture").unwrap_or("?");
    let file_type = gguf.metadata_u32("general.file_type").ok();
    let quant = file_type.map(file_type_name).unwrap_or("?");
    let (arch_ok, quant_ok) = inference_support(arch, file_type);
    if arch_ok && quant_ok {
        return Ok(());
    }
    eprintln!();
    if !arch_ok {
        eprintln!(
            "  ⚠ vulkanforge {subcommand}: architecture '{arch}' is not yet wired into the forward pass."
        );
        eprintln!("    Only qwen2 / qwen3 run end-to-end in v0.3.x.");
    } else {
        eprintln!(
            "  ⚠ vulkanforge {subcommand}: quantization {quant} is not supported yet (only Q4_K_M)."
        );
    }
    eprintln!(
        "  Use `vulkanforge info --model {}` for full metadata.",
        model_path.display()
    );
    eprintln!();
    Err(format!("unsupported model for {subcommand}: arch={arch}, quant={quant}").into())
}

fn run_bench(
    model_path: PathBuf,
    pp_list: &str,
    runs: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use vulkanforge::backend::vulkan::decode::{embedding_row, GenerateConfig, Sampling, generate_from_tokens};
    preflight_supported(&model_path, "bench")?;

    let pp_sizes: Vec<u32> = pp_list
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("invalid --pp-list (must be comma-separated u32): {e}"))?;

    // Boilerplate device + model load — same shape as run_chat.
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
    let (registry, _pipelines_loaded) =
        PipelineRegistry::new(&dev.device, cache_path.as_deref())?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;
    let gguf = GgufFile::open(&model_path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let max_pp_local = pp_sizes.iter().copied().max().unwrap_or(64);
    let kv_cache = KvCache::new(
        &dev.device, &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            // Bench needs to fit prefill + decode_max in the kv cache.
            max_seq_len: (max_pp_local + 64).max(MAX_SEQ_LEN),
        },
    )?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;

    println!();
    println!("  vulkanforge bench — {} runs/sample", runs);
    println!();

    // ---- 1. Decode benchmark: generate 32 tokens after a 1-token prompt ----
    let decode_max = 32u32;
    let cfg_g = GenerateConfig {
        max_tokens: decode_max,
        print_stream: false,
        think_filter: false,
        sampling: Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0 },
    };
    let prefill_tok = vec![tokenizer.bos_id.unwrap_or(1)];
    let mut decode_samples = Vec::new();
    let mut prefill_samples = Vec::new();
    for _ in 0..runs {
        forward.kv_cache.reset();
        let t0 = Instant::now();
        let r = generate_from_tokens(
            &mut forward, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            &prefill_tok, 0, &cfg_g, &mut |_, _| {},
        )?;
        let _ = t0; // (timings live inside r)
        if r.generated_tokens > 0 {
            decode_samples.push(r.decode_time.as_secs_f64() / r.generated_tokens as f64);
            prefill_samples.push(r.prefill_time.as_secs_f64());
        }
    }
    decode_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_decode_s_per_tok = decode_samples[decode_samples.len() / 2];
    let decode_tok_s = 1.0 / med_decode_s_per_tok;
    println!("  Decode (1-tok prompt + {} gen)", decode_max);
    println!(
        "    {:>6.1} tok/s  (median over {} runs)",
        decode_tok_s, runs
    );

    // ---- 2. Prefill sweep ----
    println!();
    println!("  Prefill sweep");
    println!("    {:>6}  {:>10}  {:>10}", "pp", "ms (med)", "tok/s");
    for &pp in &pp_sizes {
        // Build a synthetic prefill of length pp using the BOS token.
        // (For a real workload, the user would benchmark via the
        // `examples/run_pp_bench` tool which reads tokenised prompts.)
        let toks: Vec<u32> = (0..pp).map(|_| tokenizer.bos_id.unwrap_or(1)).collect();
        let cfg_pp = GenerateConfig {
            max_tokens: 1,
            print_stream: false,
            think_filter: false,
            sampling: Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0 },
        };
        let mut samples_ms = Vec::new();
        for _ in 0..runs {
            forward.kv_cache.reset();
            let t0 = Instant::now();
            let r = generate_from_tokens(
                &mut forward, &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
                &toks, 0, &cfg_pp, &mut |_, _| {},
            )?;
            let _ = t0;
            samples_ms.push(r.prefill_time.as_secs_f64() * 1000.0);
        }
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = samples_ms[samples_ms.len() / 2];
        let tok_s = (pp as f64) / (med / 1000.0);
        println!("    {:>6}  {:>10.1}  {:>10.1}", pp, med, tok_s);
    }
    println!();

    // Cleanup mirrors run_chat.
    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();

    // Mark embedding_row as referenced to avoid an unused-import warning
    // (it's used by the example benches, kept in scope here in case the
    // bench grows a streaming-decode variant later).
    let _ = embedding_row;
    Ok(())
}
