//! Prompt-16 Alice context-retention test.
//!
//! Six-turn `ChatSession` exchange with NO `reset()` between turns —
//! exercises the multi-turn KV-cache path (Phase 3B) on a real model.
//! Three "critical" turns ask the model to recall facts the user gave
//! earlier (name, city, both); they pass iff the response contains the
//! expected keywords. Soft turns (1, 2, 4) just have to produce a
//! coherent response.
//!
//! On an 8B-class model with a working KV-cache the score should be
//! 3 / 3. Anything less points to a bug in the multi-turn code path
//! (KV-cache being reset, position offset wrong, chat-template
//! continuation rendered with the wrong boundary, …).
//!
//! Run:
//!   VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
//!     cargo run --release --example run_alice_test

use std::path::PathBuf;
use std::time::Duration;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat::{ChatSession, TurnResult};
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

struct Turn {
    user: &'static str,
    expect_contains: &'static [&'static str],
    critical: bool,
    max_tokens: u32,
}

const ALICE_TURNS: &[Turn] = &[
    Turn {
        user: "My name is Alice.",
        expect_contains: &["Alice"],
        critical: false,
        max_tokens: 60,
    },
    Turn {
        user: "What is 2 + 2?",
        expect_contains: &["4"],
        critical: false,
        max_tokens: 40,
    },
    Turn {
        user: "What is my name?",
        expect_contains: &["Alice"],
        critical: true,
        max_tokens: 50,
    },
    Turn {
        user: "I live in Berlin.",
        expect_contains: &["Berlin"],
        critical: false,
        max_tokens: 60,
    },
    Turn {
        user: "Where do I live?",
        expect_contains: &["Berlin"],
        critical: true,
        max_tokens: 50,
    },
    Turn {
        user: "What is my name and where do I live?",
        expect_contains: &["Alice", "Berlin"],
        critical: true,
        max_tokens: 100,
    },
];

#[derive(Debug)]
#[allow(dead_code)] // Debug-printed; some fields kept for future structured output.
struct TurnRecord {
    idx: usize,
    user: String,
    response: String,
    expected: Vec<String>,
    passed: bool,
    critical: bool,
    decode_tok_s: f64,
    prompt_tokens: u32,
    generated_tokens: u32,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });

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
    let template = ChatTemplate::detect(&gguf, &tokenizer);

    println!("VulkanForge — Alice multi-turn context-retention test");
    println!("  model    = {}", model_path.display());
    println!("  arch     = {}", cfg.architecture);
    println!("  template = {:?}", template);
    println!("  rope     = {:?}", cfg.rope_variant);
    println!("  qk_norm  = {}", cfg.has_qk_norm);
    println!();

    let mut session = ChatSession::new_with_template(
        forward,
        "You are a helpful assistant.",
        template,
    );

    let mut records: Vec<TurnRecord> = Vec::new();
    let mut total_prefill = Duration::ZERO;
    let mut total_decode = Duration::ZERO;
    let mut total_prefill_tok = 0u32;
    let mut total_decode_tok = 0u32;

    // think_filter=false: DeepSeek-R1's <think>…</think> may legitimately
    // contain the keywords. We score against the full generated_text.
    let cfg_g = GenerateConfig {
        max_tokens: 0,
        print_stream: false,
        think_filter: false,
    };

    for (i, t) in ALICE_TURNS.iter().enumerate() {
        let local_cfg = GenerateConfig {
            max_tokens: t.max_tokens,
            ..cfg_g.clone()
        };
        let turn: TurnResult = session.send(
            &dev,
            &registry,
            &cmd_ctx,
            &model,
            &gguf,
            &cfg,
            &tokenizer,
            t.user,
            &local_cfg,
        )?;

        let resp_lower = turn.generated_text.to_lowercase();
        let passed = t
            .expect_contains
            .iter()
            .all(|kw| resp_lower.contains(&kw.to_lowercase()));

        let badge = if passed { "✅" } else { "❌" };
        let crit = if t.critical { " [CRITICAL]" } else { "" };
        let preview = first_n_chars(&turn.generated_text, 240).replace('\n', " ⏎ ");
        println!("─── Turn {}/{}{}", i + 1, ALICE_TURNS.len(), crit);
        println!("  USER: {}", t.user);
        println!("  ASST: {}", preview);
        println!(
            "  KW {:?} → {} · pp={} gen={} · {:.0} tok/s prefill, {:.1} tok/s decode · pos={}",
            t.expect_contains,
            badge,
            turn.prompt_tokens,
            turn.generated_tokens,
            turn.prefill_tok_s(),
            turn.decode_tok_s(),
            session.current_pos,
        );
        println!();

        records.push(TurnRecord {
            idx: i + 1,
            user: t.user.to_string(),
            response: turn.generated_text.clone(),
            expected: t.expect_contains.iter().map(|s| s.to_string()).collect(),
            passed,
            critical: t.critical,
            decode_tok_s: turn.decode_tok_s(),
            prompt_tokens: turn.prompt_tokens,
            generated_tokens: turn.generated_tokens,
        });
        total_prefill += turn.prefill_time;
        total_decode += turn.decode_time;
        total_prefill_tok += turn.prompt_tokens;
        total_decode_tok += turn.generated_tokens;
    }

    let critical_total = records.iter().filter(|r| r.critical).count();
    let critical_passed = records.iter().filter(|r| r.critical && r.passed).count();
    let agg_prefill_tok_s = total_prefill_tok as f64 / total_prefill.as_secs_f64().max(1e-9);
    let agg_decode_tok_s = total_decode_tok as f64 / total_decode.as_secs_f64().max(1e-9);

    println!("=== Alice summary ===");
    for r in &records {
        let crit = if r.critical { "*" } else { " " };
        println!(
            "  {} Turn {}: {}  {:?} → {}",
            crit,
            r.idx,
            if r.passed { "✅" } else { "❌" },
            r.expected,
            first_n_chars(&r.response, 80).replace('\n', " ⏎ "),
        );
    }
    println!();
    println!(
        "  Critical score: {}/{}  ({})",
        critical_passed,
        critical_total,
        if critical_passed == critical_total {
            "PASS"
        } else if critical_passed * 3 >= critical_total * 2 {
            "PARTIAL"
        } else {
            "FAIL"
        },
    );
    println!(
        "  Aggregate: {} prompt + {} gen tokens, prefill {:.0} tok/s, decode {:.1} tok/s",
        total_prefill_tok, total_decode_tok, agg_prefill_tok_s, agg_decode_tok_s,
    );
    println!("  Final KV pos: {} / {}", session.current_pos, MAX_SEQ_LEN);

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();

    if critical_passed != critical_total {
        std::process::exit(2);
    }
    Ok(())
}

fn first_n_chars(s: &str, n: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= n {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out
}
