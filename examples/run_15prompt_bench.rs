//! Phase-3C 15-prompt benchmark — runs `inference_test_prompts_15.json`
//! through VulkanForge with KV-cache reset between prompts and prints
//! a per-prompt + aggregate tok/s table for the 4-system comparison.
//!
//! Each prompt is a fresh `ChatSession::send` with `think_filter=false`
//! (matches the comparison runs in
//! `~/projects/ROCmForge/results/inference_test_20260425.md` which
//! also use greedy + no filtering).
//!
//! Run: `cargo run --release --example run_15prompt_bench`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::chat::{ChatSession, TurnResult};
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

#[derive(Debug)]
struct PromptSpec {
    id: u32,
    name: String,
    category: String,
    max_tokens: u32,
    prompt: String,
}

fn parse_prompts(path: &str) -> Result<Vec<PromptSpec>, Box<dyn std::error::Error>> {
    // Tiny hand-rolled JSON parsing — the file is well-formed and we
    // only care about a handful of keys per prompt entry. Avoids
    // pulling in serde just for this benchmark.
    let raw = std::fs::read_to_string(path)?;
    let mut prompts: Vec<PromptSpec> = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = raw[cursor..].find("\"id\":") {
        let s = cursor + start;
        let id: u32 = json_int(&raw[s + 5..])?;
        let name = json_str_after(&raw, s, "\"name\":")?;
        let category = json_str_after(&raw, s, "\"category\":")?;
        let max_tokens: u32 = json_int_after(&raw, s, "\"max_tokens\":")?;
        let prompt = json_str_after(&raw, s, "\"prompt\":")?;
        prompts.push(PromptSpec { id, name, category, max_tokens, prompt });
        cursor = s + 5;
    }
    if prompts.is_empty() {
        return Err("no prompts parsed".into());
    }
    Ok(prompts)
}

fn json_int(s: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let s = s.trim_start();
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    Ok(s[..end].parse()?)
}
fn json_int_after(haystack: &str, after: usize, key: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let idx = haystack[after..].find(key).ok_or_else(|| format!("missing {key}"))?;
    json_int(&haystack[after + idx + key.len()..])
}
fn json_str_after(haystack: &str, after: usize, key: &str) -> Result<String, Box<dyn std::error::Error>> {
    let idx = haystack[after..]
        .find(key)
        .ok_or_else(|| format!("missing {key}"))?;
    let s = &haystack[after + idx + key.len()..];
    let q1 = s.find('"').ok_or("no opening quote")?;
    // Walk past the opener, accept escaped quotes inside the body.
    let mut end = q1 + 1;
    while end < s.len() {
        let b = s.as_bytes()[end];
        if b == b'\\' { end += 2; continue; }
        if b == b'"' { break; }
        end += 1;
    }
    Ok(unescape_json(&s[q1 + 1..end]))
}
fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'n' => out.push('\n'),
                b't' => out.push('\t'),
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                _ => { out.push(bytes[i] as char); out.push(bytes[i+1] as char); }
            }
            i += 2;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

#[derive(Debug)]
struct Row {
    id: u32,
    name: String,
    category: String,
    prefill_tok: u32,
    decode_tok: u32,
    prefill_ms: f64,
    decode_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    coherent: bool,
    excerpt: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let prompts_path = std::env::var("VF_PROMPTS")
        .unwrap_or_else(|_| "inference_test_prompts_15.json".to_string());
    let prompts = parse_prompts(&prompts_path)?;
    println!("VulkanForge — 15-prompt benchmark ({} prompts)", prompts.len());

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
    let load_start = Instant::now();
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
    println!(
        "  {} loaded in {:.1} s",
        model_path.display(),
        load_start.elapsed().as_secs_f64()
    );
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
    let mut session = ChatSession::new(forward, "You are a helpful assistant.");

    let mut rows: Vec<Row> = Vec::new();
    let mut total_prefill_tok = 0u32;
    let mut total_decode_tok = 0u32;
    let mut total_prefill = Duration::ZERO;
    let mut total_decode = Duration::ZERO;

    for p in &prompts {
        // Cap max_tokens so each prompt fits in the 2048-context KV.
        // ROCmForge's reference run ran at higher caps, but Phase 3B's
        // chat session is bounded by max_seq_len; we mirror what fits.
        let max_tok = p
            .max_tokens
            .min(MAX_SEQ_LEN.saturating_sub(/* generous buffer */ 200));
        let cfg_g = GenerateConfig {
            max_tokens: max_tok,
            print_stream: false,
            think_filter: false,
        };
        // KV reset between every prompt — apples-to-apples with ROCmForge.
        session.reset();

        let result = session.send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            &p.prompt, &cfg_g,
        );
        let turn: TurnResult = match result {
            Ok(t) => t,
            Err(e) => {
                eprintln!("⚠ prompt #{} '{}' failed: {e}", p.id, p.name);
                continue;
            }
        };

        let lower = turn.generated_text.to_lowercase();
        // Coherence heuristic: at least 4 ASCII letters in the output
        // and not 90%+ a single char (catches NaN-runs and total
        // gibberish; not bulletproof but caught the Phase-2C regression
        // we'd otherwise have shipped).
        let alpha = lower.chars().filter(|c| c.is_ascii_alphabetic()).count();
        let coherent = alpha >= 4 && !is_repeating_garbage(&lower);

        let excerpt = first_n_chars(&turn.generated_text, 80).replace('\n', " ");
        rows.push(Row {
            id: p.id,
            name: p.name.clone(),
            category: p.category.clone(),
            prefill_tok: turn.prompt_tokens,
            decode_tok: turn.generated_tokens,
            prefill_ms: turn.prefill_time.as_secs_f64() * 1000.0,
            decode_ms: turn.decode_time.as_secs_f64() * 1000.0,
            prefill_tok_s: turn.prefill_tok_s(),
            decode_tok_s: turn.decode_tok_s(),
            coherent,
            excerpt,
        });
        total_prefill_tok += turn.prompt_tokens;
        total_decode_tok += turn.generated_tokens;
        total_prefill += turn.prefill_time;
        total_decode += turn.decode_time;

        println!(
            "  #{:>2} {:<32} pp={:>3} gen={:>4}  prefill={:>6.1} tok/s  decode={:>5.1} tok/s  {}",
            p.id, truncate(&p.name, 32),
            turn.prompt_tokens, turn.generated_tokens,
            turn.prefill_tok_s(), turn.decode_tok_s(),
            if coherent { "✓" } else { "✗" }
        );
    }

    println!("\n=== Per-prompt results ===");
    println!(
        "{:<3} {:<26} {:<14} {:>5} {:>5} {:>10} {:>10}  {}",
        "#", "Name", "Category", "PromTk", "GenTk", "Prefill", "Decode", "Coh"
    );
    for r in &rows {
        println!(
            "{:<3} {:<26} {:<14} {:>5} {:>5} {:>9.1}  {:>9.1}  {}",
            r.id, truncate(&r.name, 26), truncate(&r.category, 14),
            r.prefill_tok, r.decode_tok,
            r.prefill_tok_s, r.decode_tok_s,
            if r.coherent { "✓" } else { "✗" }
        );
    }

    let agg_prefill_s = total_prefill.as_secs_f64();
    let agg_decode_s = total_decode.as_secs_f64();
    let agg_pre = total_prefill_tok as f64 / agg_prefill_s.max(1e-9);
    let agg_dec = total_decode_tok as f64 / agg_decode_s.max(1e-9);

    println!("\n=== Aggregate ===");
    println!("  Prompt tokens:  {}", total_prefill_tok);
    println!("  Decode tokens:  {}", total_decode_tok);
    println!("  Prefill total:  {:.1} ms  →  {:.1} tok/s", agg_prefill_s * 1000.0, agg_pre);
    println!("  Decode total:   {:.1} ms  →  {:.1} tok/s", agg_decode_s * 1000.0, agg_dec);

    let mut decs: Vec<f64> = rows.iter().map(|r| r.decode_tok_s).collect();
    let mut pres: Vec<f64> = rows.iter().map(|r| r.prefill_tok_s).collect();
    decs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pres.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  MEDIAN decode:  {:.1} tok/s", decs[decs.len() / 2]);
    println!("  MEDIAN prefill: {:.1} tok/s", pres[pres.len() / 2]);
    let coherent_n = rows.iter().filter(|r| r.coherent).count();
    println!("  Coherent prompts: {}/{}", coherent_n, rows.len());

    println!("\n=== 4-System Comparison ===");
    println!("                          Decode tok/s  Prefill tok/s");
    println!("  llama.cpp Vulkan:           114.2          4314    (reference)");
    println!("  ROCmForge HIP (latest):      95.4           768.6  (~/projects/ROCmForge/results/inference_test_20260425.md)");
    println!("  llama.cpp ROCm:              87.5          3684    (reference)");
    println!("  VulkanForge Phase 3C:       {:>5.1}         {:>6.1}  ← this run", agg_dec, agg_pre);

    forward_destroy(session, &dev, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

fn forward_destroy(s: ChatSession, dev: &VulkanDevice, allocator: &mut Allocator) {
    s.forward.destroy(&dev.device, allocator);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

fn first_n_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn is_repeating_garbage(s: &str) -> bool {
    if s.len() < 8 {
        return false;
    }
    // Any run of 16+ identical *non-whitespace* bytes is almost
    // certainly degenerate (catches NaN-runs and total gibberish).
    // Whitespace runs are normal in code (deep indentation) and in
    // prose (paragraph breaks), so they're excluded.
    let bytes = s.as_bytes();
    let mut run = 1usize;
    for w in bytes.windows(2) {
        let same = w[0] == w[1];
        let is_ws = w[0] == b' ' || w[0] == b'\n' || w[0] == b'\t';
        if same && !is_ws {
            run += 1;
            if run >= 16 {
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}
