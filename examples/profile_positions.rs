//! Per-position decode profiling — runs the real "Explain what a
//! mutex is in one sentence." prefill + decode loop, captures the
//! [`ForwardStats`] from the profiler at positions {0, 50, 100, 200},
//! and prints a categorised breakdown plus a comparison table.
//!
//! The Forward instance always carries a profiler; at non-target
//! positions we just discard the resulting samples. The actual
//! `Forward` dispatch path is unmodified — Phase 2D's "Keine Änderung
//! am Forward-Pass" rule.
//!
//! Categorisation comes from the labels passed to `self.profile(...)`
//! inside `forward.rs`:
//!   GEMV         : gemv_q/k/v/o/gate/up/down + lm_head
//!   Attention    : scalar_attn
//!   KV-write     : kv_write (transfer, not compute)
//!   Norm         : rms_norm_*
//!   RoPE         : rope_*
//!   Add/Mul/SiLU : add_*, mul_*, silu_*
//!
//! Run:
//!   cargo run --release --example profile_positions

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::{Forward, ForwardStats};
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use vulkanforge::backend::vulkan::profiler::ShaderProfiler;
use vulkanforge::backend::vulkan::tokenizer::{apply_chat_template, Tokenizer};

const MAX_SEQ_LEN: u32 = 512;
const PROFILE_POSITIONS: &[u32] = &[0, 50, 100, 200];
const PROMPT: &str = "Explain what a mutex is in one sentence.";
const MAX_DECODE_TOKENS: u32 = 220; // enough to reach pos=200

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("VulkanForge — Phase-2 profiling at decode positions {PROFILE_POSITIONS:?}");

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

    // 25 dispatches/layer × 36 layers + LM-head bracket = 904 entries
    // worst case, well under capacity_pairs = 1024.
    let profiler = ShaderProfiler::new(
        &dev.instance,
        dev.physical_device,
        dev.queue_family_index,
        &dev.device,
        1024,
    )?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), Some(profiler))?;

    let prompt_ids = apply_chat_template(&tokenizer, PROMPT, None);
    println!(
        "\n  Prompt:           {:?}\n  Prompt-tokens:    {}",
        PROMPT,
        prompt_ids.len()
    );
    let max_pos = *PROFILE_POSITIONS.iter().max().unwrap();
    if (prompt_ids.len() as u32 + MAX_DECODE_TOKENS) <= max_pos {
        return Err(format!(
            "max position {max_pos} unreachable: prompt={} + decode={} only reaches {}",
            prompt_ids.len(),
            MAX_DECODE_TOKENS,
            prompt_ids.len() as u32 + MAX_DECODE_TOKENS,
        )
        .into());
    }

    // -----------------------------------------------------------
    // Drive prefill + decode, capture ForwardStats at target positions.
    // -----------------------------------------------------------
    forward.kv_cache.reset();
    let mut samples_by_pos: BTreeMap<u32, ForwardStats> = BTreeMap::new();
    let mut last_logits: Vec<f32> = Vec::new();

    let total_positions = prompt_ids.len() as u32 + MAX_DECODE_TOKENS;
    let mut pos: u32 = 0;
    while pos < total_positions {
        // Pick the next embedding.
        let embd = if (pos as usize) < prompt_ids.len() {
            embedding_row(&gguf, &cfg, prompt_ids[pos as usize])?
        } else {
            // Greedy decode from the last logits we read back.
            let next_id = argmax(&last_logits) as u32;
            if tokenizer.is_eos(next_id) {
                // Don't emit EOS into the KV cache — but we still need
                // to keep going if we haven't hit our highest profile
                // position yet. Substitute a known-non-EOS token so
                // the cache keeps growing for measurement purposes.
                eprintln!(
                    "  (note: EOS hit at pos={pos}; substituting <think> id to keep advancing)"
                );
                embedding_row(&gguf, &cfg, 151667 /* <think> */)?
            } else {
                embedding_row(&gguf, &cfg, next_id)?
            }
        };

        let stats = forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos)?;
        last_logits = forward.logits()?;

        if PROFILE_POSITIONS.contains(&pos) {
            println!("  → captured profile at pos={pos}");
            samples_by_pos.insert(pos, stats);
        }
        pos += 1;
        if samples_by_pos.len() == PROFILE_POSITIONS.len() {
            break;
        }
    }

    // -----------------------------------------------------------
    // Report.
    // -----------------------------------------------------------
    println!("\n=========== Per-Position Breakdown ===========");
    let mut summaries: BTreeMap<u32, PositionSummary> = BTreeMap::new();
    for &p in PROFILE_POSITIONS {
        let stats = match samples_by_pos.get(&p) {
            Some(s) => s,
            None => {
                println!("\nPosition {p}: NOT CAPTURED");
                continue;
            }
        };
        let summary = summarise(stats);
        print_per_position(p, &summary, stats);
        summaries.insert(p, summary);
    }

    println!("\n=========== Comparison Table ===========\n");
    print_comparison_table(&summaries);

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    let _ = vk::Buffer::null();
    Ok(())
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PositionSummary {
    forward_total: Duration,
    forward_total_gpu: Duration, // sum of all profiled GPU samples
    gemv: Duration,
    gemv_calls: u32,
    attention: Duration,
    attention_calls: u32,
    kv_write: Duration,
    kv_write_calls: u32,
    rest: Duration,
    rest_calls: u32,
    /// Top-3 single-shader categories by total time.
    top3: Vec<(String, Duration, u32)>,
}

fn category(label: &str) -> &'static str {
    if label.starts_with("gemv_") || label == "lm_head" {
        "GEMV"
    } else if label == "scalar_attn" {
        "Attention"
    } else if label == "kv_write" {
        "KV-write"
    } else if label.starts_with("rms_norm_") {
        "Norm"
    } else if label.starts_with("rope_") {
        "RoPE"
    } else if label.starts_with("add_") {
        "Add"
    } else if label.starts_with("mul_") {
        "Mul"
    } else if label.starts_with("silu_") {
        "SiLU"
    } else if label.starts_with("copy") {
        "Copy"
    } else {
        "Other"
    }
}

fn summarise(stats: &ForwardStats) -> PositionSummary {
    let mut gemv = Duration::ZERO;
    let mut gemv_calls = 0u32;
    let mut attention = Duration::ZERO;
    let mut attention_calls = 0u32;
    let mut kv_write = Duration::ZERO;
    let mut kv_write_calls = 0u32;
    let mut rest = Duration::ZERO;
    let mut rest_calls = 0u32;
    let mut total_gpu = Duration::ZERO;

    let mut by_cat: BTreeMap<String, (Duration, u32)> = BTreeMap::new();
    for (name, (d, n)) in &stats.per_shader {
        total_gpu += *d;
        let cat = category(name).to_string();
        let entry = by_cat.entry(cat.clone()).or_insert((Duration::ZERO, 0));
        entry.0 += *d;
        entry.1 += *n;
        match cat.as_str() {
            "GEMV" => {
                gemv += *d;
                gemv_calls += *n;
            }
            "Attention" => {
                attention += *d;
                attention_calls += *n;
            }
            "KV-write" => {
                kv_write += *d;
                kv_write_calls += *n;
            }
            _ => {
                rest += *d;
                rest_calls += *n;
            }
        }
    }

    // Top-3 by total time among individual shader names (not categories).
    let mut shaders: Vec<(String, Duration, u32)> = stats
        .per_shader
        .iter()
        .map(|(n, (d, c))| (n.clone(), *d, *c))
        .collect();
    shaders.sort_by(|a, b| b.1.cmp(&a.1));
    let top3 = shaders.into_iter().take(3).collect();

    PositionSummary {
        forward_total: stats.total,
        forward_total_gpu: total_gpu,
        gemv,
        gemv_calls,
        attention,
        attention_calls,
        kv_write,
        kv_write_calls,
        rest,
        rest_calls,
        top3,
    }
}

fn print_per_position(pos: u32, s: &PositionSummary, stats: &ForwardStats) {
    let total_ms = s.forward_total.as_secs_f64() * 1000.0;
    let gpu_ms = s.forward_total_gpu.as_secs_f64() * 1000.0;
    let overhead = s.forward_total.saturating_sub(s.forward_total_gpu);
    let overhead_pct = if !s.forward_total.is_zero() {
        (overhead.as_secs_f64() / s.forward_total.as_secs_f64()) * 100.0
    } else {
        0.0
    };
    println!(
        "\n--- pos={pos} ---  wall={:>6.2} ms  gpu_sum={:>6.2} ms  overhead={:>5.2} ms ({:.1}%)  effective={:>5.1} tok/s",
        total_ms,
        gpu_ms,
        overhead.as_secs_f64() * 1000.0,
        overhead_pct,
        1000.0 / total_ms,
    );

    // Full per-shader table sorted by time.
    let mut rows: Vec<(&String, Duration, u32)> = stats
        .per_shader
        .iter()
        .map(|(n, (d, c))| (n, *d, *c))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let cat_total = stats.per_shader.values().map(|(d, _)| *d).sum::<Duration>();
    let cat_total_us = cat_total.as_secs_f64() * 1e6;
    println!(
        "  {:<22}{:>6}{:>11}{:>9}",
        "Shader", "Calls", "Time (µs)", "% GPU"
    );
    for (name, d, calls) in rows {
        let us = d.as_secs_f64() * 1e6;
        let pct = if cat_total_us > 0.0 { us / cat_total_us * 100.0 } else { 0.0 };
        println!(
            "  {:<22}{:>6}{:>11.1}{:>8.1}%",
            name, calls, us, pct,
        );
    }
}

fn print_comparison_table(summaries: &BTreeMap<u32, PositionSummary>) {
    let positions: Vec<u32> = summaries.keys().copied().collect();
    let header = format!(
        "{:<24}{}",
        "Metric",
        positions
            .iter()
            .map(|p| format!("{:>14}", format!("pos={p}")))
            .collect::<String>()
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let row_us = |label: &str, f: &dyn Fn(&PositionSummary) -> f64| {
        let mut line = format!("{label:<24}");
        for p in &positions {
            let v = f(&summaries[p]);
            line.push_str(&format!("{:>14.1}", v));
        }
        println!("{line}");
    };
    let row_pct = |label: &str, f: &dyn Fn(&PositionSummary) -> f64| {
        let mut line = format!("{label:<24}");
        for p in &positions {
            line.push_str(&format!("{:>13.1}%", f(&summaries[p])));
        }
        println!("{line}");
    };
    let row_str = |label: &str, f: &dyn Fn(&PositionSummary) -> String| {
        let mut line = format!("{label:<24}");
        for p in &positions {
            line.push_str(&format!("{:>14}", f(&summaries[p])));
        }
        println!("{line}");
    };

    row_us("Forward wall (µs)", &|s| s.forward_total.as_secs_f64() * 1e6);
    row_us("Forward GPU sum (µs)", &|s| s.forward_total_gpu.as_secs_f64() * 1e6);
    row_str("Effective (tok/s)", &|s| {
        format!("{:.1}", 1.0 / s.forward_total.as_secs_f64())
    });
    println!();

    let pos0_attn = summaries.get(&0).map(|s| s.attention.as_secs_f64()).unwrap_or(0.0);
    let pos0_attn_calls = summaries.get(&0).map(|s| s.attention_calls).unwrap_or(0);
    let pos0_per_call = if pos0_attn_calls > 0 { pos0_attn / pos0_attn_calls as f64 } else { 0.0 };
    let pos0_gemv = summaries.get(&0).map(|s| s.gemv.as_secs_f64()).unwrap_or(0.0);

    row_us("scalar_attn (µs)", &|s| s.attention.as_secs_f64() * 1e6);
    row_pct("scalar_attn (%)", &|s| {
        let total_us = s.forward_total_gpu.as_secs_f64();
        if total_us > 0.0 { s.attention.as_secs_f64() / total_us * 100.0 } else { 0.0 }
    });
    row_str("scalar_attn vs pos=0", &|s| {
        if pos0_attn > 0.0 {
            format!("{:.2}×", s.attention.as_secs_f64() / pos0_attn)
        } else {
            "—".to_string()
        }
    });
    let _ = pos0_per_call; // expose later if useful
    println!();

    row_us("GEMV total (µs)", &|s| s.gemv.as_secs_f64() * 1e6);
    row_pct("GEMV (%)", &|s| {
        let total_us = s.forward_total_gpu.as_secs_f64();
        if total_us > 0.0 { s.gemv.as_secs_f64() / total_us * 100.0 } else { 0.0 }
    });
    row_str("GEMV vs pos=0", &|s| {
        if pos0_gemv > 0.0 {
            format!("{:.2}×", s.gemv.as_secs_f64() / pos0_gemv)
        } else {
            "—".to_string()
        }
    });
    println!();

    row_us("KV-write (µs)", &|s| s.kv_write.as_secs_f64() * 1e6);
    row_pct("KV-write (%)", &|s| {
        let total_us = s.forward_total_gpu.as_secs_f64();
        if total_us > 0.0 { s.kv_write.as_secs_f64() / total_us * 100.0 } else { 0.0 }
    });
    println!();

    row_us("Rest (µs)", &|s| s.rest.as_secs_f64() * 1e6);
    row_pct("Rest (%)", &|s| {
        let total_us = s.forward_total_gpu.as_secs_f64();
        if total_us > 0.0 { s.rest.as_secs_f64() / total_us * 100.0 } else { 0.0 }
    });
    println!();

    row_us("Dispatch overhead (µs)", &|s| {
        s.forward_total.saturating_sub(s.forward_total_gpu).as_secs_f64() * 1e6
    });
    row_pct("Dispatch overhead (%)", &|s| {
        let wall = s.forward_total.as_secs_f64();
        if wall > 0.0 {
            s.forward_total.saturating_sub(s.forward_total_gpu).as_secs_f64() / wall * 100.0
        } else {
            0.0
        }
    });
    println!();

    println!("Top-3 per position:");
    for p in &positions {
        let s = &summaries[p];
        let mut bits: Vec<String> = Vec::new();
        for (n, d, c) in &s.top3 {
            bits.push(format!(
                "{} ({:.1} µs, {} calls)",
                n,
                d.as_secs_f64() * 1e6,
                c,
            ));
        }
        println!("  pos={p}: {}", bits.join(", "));
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}
