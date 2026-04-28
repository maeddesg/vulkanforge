//! Sprint 5 — direct prefill_batch micro-bench at exact pp values.
//!
//! Synthesizes a `pp × hidden_dim` embedding tensor by repeating the
//! BOS token's embedding row, then calls `prefill_batch` directly and
//! measures wall-clock GPU dispatch time. This mirrors what
//! `llama-bench -p N -tg 0` does — no tokenizer / chat template /
//! decode loop, just the prefill GEMM stack at a controlled N.
//!
//! Env vars:
//!   VF_PP_LIST       comma-separated pp values (default 64)
//!   VF_PP_RUNS       repetitions per pp (default 3)
//!   VF_PP_WARMUP     warmup runs per pp (default 1)
//!   VF_MODEL_PATH    GGUF model (default $HOME/models/Qwen3-8B-Q4_K_M.gguf)
//!   VULKANFORGE_COOPMAT=1   route gemm_q through the coopmat fusion path
//!   VULKANFORGE_MAX_PREFILL N  cap (default Sprint-5 1024; must be ≥ max pp)
//!
//! Output: one line per pp with median ms and tok/s.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};

const MAX_SEQ_LEN: u32 = 2048;

fn parse_env_u32(key: &str, default_val: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_val)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let pp_list: Vec<u32> = std::env::var("VF_PP_LIST")
        .unwrap_or_else(|_| "64".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if pp_list.is_empty() {
        return Err("VF_PP_LIST empty after parse".into());
    }
    let runs = parse_env_u32("VF_PP_RUNS", 3);
    let warmup = parse_env_u32("VF_PP_WARMUP", 1);

    let max_pp_needed = *pp_list.iter().max().unwrap();
    // Make sure the max_prefill_tokens cap covers our biggest pp.
    if std::env::var("VULKANFORGE_MAX_PREFILL").is_err() {
        unsafe { std::env::set_var("VULKANFORGE_MAX_PREFILL", max_pp_needed.to_string()); }
    }

    let coopmat = std::env::var("VULKANFORGE_COOPMAT").map(|v| v == "1").unwrap_or(false);

    println!(
        "VulkanForge pp-bench — {} pp values, {} runs each (warmup {}), coopmat={}",
        pp_list.len(),
        runs,
        warmup,
        coopmat,
    );
    println!("  max_pp cap = {}", max_pp_needed);

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
    println!(
        "  arch={} hidden={} ffn={} layers={}",
        cfg.architecture, cfg.hidden_dim, cfg.ffn_dim, cfg.n_layers,
    );

    // Build a single-token embedding (token id 0 is fine — the GEMM
    // ops do the same work regardless of input values).
    let one_embd = embedding_row(&gguf, &cfg, 0)?;
    if one_embd.len() != cfg.hidden_dim as usize {
        return Err(format!(
            "embedding length mismatch: {} != {}",
            one_embd.len(),
            cfg.hidden_dim
        ).into());
    }

    println!();
    println!("  pp     median_ms     mean_ms      tok/s");
    println!("  ---  -----------  -----------  --------");

    let mut rows: Vec<(u32, f64, f64, f64)> = Vec::new();
    for &pp in &pp_list {
        if pp > forward.max_prefill_tokens {
            eprintln!(
                "  skip pp={} (exceeds max_prefill_tokens={})",
                pp, forward.max_prefill_tokens
            );
            continue;
        }
        // Synthesize embeddings: pp copies of the same row.
        let mut embeds = Vec::with_capacity((pp as usize) * one_embd.len());
        for _ in 0..pp {
            embeds.extend_from_slice(&one_embd);
        }

        // Warmup
        for _ in 0..warmup {
            forward.kv_cache.reset();
            forward.prefill_batch(&dev, &registry, &cmd_ctx, &model, &embeds, pp, 0)?;
        }

        // Measured runs
        let mut times: Vec<Duration> = Vec::with_capacity(runs as usize);
        for _ in 0..runs {
            forward.kv_cache.reset();
            let t0 = Instant::now();
            forward.prefill_batch(&dev, &registry, &cmd_ctx, &model, &embeds, pp, 0)?;
            times.push(t0.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];
        let mean: Duration = times.iter().sum::<Duration>() / (times.len() as u32);
        let median_ms = median.as_secs_f64() * 1000.0;
        let mean_ms = mean.as_secs_f64() * 1000.0;
        let toks_per_sec = (pp as f64) / median.as_secs_f64();

        println!(
            "  {:>3}  {:>9.3}    {:>9.3}    {:>7.1}",
            pp, median_ms, mean_ms, toks_per_sec,
        );
        rows.push((pp, median_ms, mean_ms, toks_per_sec));
    }

    println!();
    println!("CSV (pp,median_ms,mean_ms,toks_per_sec):");
    for (pp, med, mean, tps) in &rows {
        println!("{},{:.4},{:.4},{:.2}", pp, med, mean, tps);
    }
    Ok(())
}
