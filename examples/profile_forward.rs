//! Phase 5A-2 step 1 — CPU profile of `forward_token`.
//!
//! Runs Qwen3-8B forward token-by-token from pos=0 to pos=210 and
//! prints the per-phase wall-time breakdown at three checkpoints
//! (pos=0 / 100 / 200) plus the median over each 50-token window.
//!
//! The goal is to localise the 3.3 ms per-token "dispatch overhead"
//! seen in earlier phases — is it Rust-side per-layer setup, raw
//! `vkCmd*` calls, GPU-wait, or something else?

use std::path::PathBuf;
use std::time::Duration;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::{Forward, ForwardTokenProfile};
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};

const MAX_SEQ_LEN: u32 = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });
    let n_tokens: u32 = std::env::var("VF_NUM_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(220);

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
    let kv_cache = KvCache::new(
        &dev.device, &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: MAX_SEQ_LEN,
        },
    )?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;

    println!(
        "model: {}\narch={} layers={} heads={}/{} hidden={} n_tokens={}\n",
        model_path.display(), cfg.architecture, cfg.n_layers,
        cfg.n_heads, cfg.n_kv_heads, cfg.hidden_dim, n_tokens,
    );

    let dummy_token: u32 = 1;
    let embd = embedding_row(&gguf, &cfg, dummy_token)?;

    let mut samples: Vec<(u32, ForwardTokenProfile)> = Vec::with_capacity(n_tokens as usize);
    // Warm-up: first forward includes pipeline-cache load, allocator
    // first-touch, page faults — exclude from the breakdown averages.
    let _warm = forward.forward_token_profile(&dev, &registry, &cmd_ctx, &model, &embd, 0)?;
    forward.kv_cache.reset();

    for pos in 0..n_tokens {
        let p = forward.forward_token_profile(&dev, &registry, &cmd_ctx, &model, &embd, pos)?;
        samples.push((pos, p));
    }

    println!("Per-phase wall time (microseconds), single-token forward:\n");
    println!("{:>6}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "pos", "pre_setup", "reset", "begin", "RECORD", "end", "submit",
        "GPU_WAIT", "readback", "TOTAL");
    println!("{}", "-".repeat(99));
    for &checkpoint in &[0u32, 1, 50, 100, 150, 200, 210] {
        let cp = checkpoint.min(n_tokens.saturating_sub(1));
        let p = samples[cp as usize].1;
        print_row(cp, &p);
    }

    println!("\nMedian over each 50-token window:");
    println!("{}", "-".repeat(99));
    let windows = [(0u32, 50u32), (50, 100), (100, 150), (150, 200), (200, n_tokens.min(220))];
    for &(lo, hi) in &windows {
        if hi > samples.len() as u32 || lo >= hi {
            continue;
        }
        let med = median_phases(&samples[lo as usize..hi as usize]);
        println!("{:>3}-{:<3}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            lo, hi - 1,
            us(med.pre_setup), us(med.reset), us(med.begin), us(med.record),
            us(med.end), us(med.submit), us(med.gpu_wait),
            us(med.readback), us(med.total()),
        );
    }

    // ---------- Drill-down: per-layer + dispatch_final breakdown ----------
    println!("\nDrill-down at pos=100 — per-layer time inside RECORD block:");
    forward.kv_cache.reset();
    // Re-prime KV up to pos=100 with cheap no-op-ish forwards using
    // forward_token (no profile) so we land in a comparable state.
    for pos in 0..100u32 {
        let _ = forward.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos)?;
    }
    let (drill, per_layer, dfin) = forward.forward_token_profile_layers(
        &dev, &registry, &cmd_ctx, &model, &embd, 100,
    )?;
    let layer_sum: Duration = per_layer.iter().sum();
    let layer_min = per_layer.iter().min().copied().unwrap_or_default();
    let layer_max = per_layer.iter().max().copied().unwrap_or_default();
    let mut sorted = per_layer.clone();
    sorted.sort();
    let layer_med = sorted[sorted.len() / 2];

    println!("  RECORD wall              {:>6} µs  (one-shot timer)", us(drill.record));
    println!("  Σ per-layer dispatches   {:>6} µs  ({:.1}% of RECORD)",
        us(layer_sum),
        100.0 * layer_sum.as_secs_f64() / drill.record.as_secs_f64());
    println!("  per-layer min/med/max    {:>6} / {:>6} / {:>6} µs",
        us(layer_min), us(layer_med), us(layer_max));
    println!("  dispatch_final + barrier {:>6} µs", us(dfin));
    let unaccounted = drill.record.as_micros() as i64
        - layer_sum.as_micros() as i64
        - dfin.as_micros() as i64;
    println!("  unaccounted              {:>6} µs", unaccounted);

    // Estimate "vkCmd-only" share by recording an empty CB.
    println!("\nFloor: empty record block (no dispatches at all):");
    let empty_t = std::time::Instant::now();
    cmd_ctx.one_shot(&dev.device, dev.compute_queue, |_| {})?;
    println!("  Empty one_shot (reset+begin+empty-record+end+submit+wait)  {:>6} µs",
        us(empty_t.elapsed()));

    println!("\nLegend:");
    println!("  pre_setup = embedding upload + RoPE-pos write + descriptor-pool reset");
    println!("  reset     = vkResetCommandBuffer + reset_fences");
    println!("  begin     = vkBeginCommandBuffer");
    println!("  RECORD    = ALL per-layer dispatch_layer + dispatch_final + final barrier");
    println!("                (HashMap pipeline lookup + push-const struct build +");
    println!("                 vkCmdBindPipeline + vkCmdPushConstants + vkCmdDispatch + barriers)");
    println!("  end       = vkEndCommandBuffer");
    println!("  submit    = vkQueueSubmit (host-only; does not block on GPU)");
    println!("  GPU_WAIT  = vkWaitForFences (true GPU wall-clock)");
    println!("  readback  = read logits buffer back to host");

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
    Ok(())
}

fn print_row(pos: u32, p: &ForwardTokenProfile) {
    println!("{:>6}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        pos,
        us(p.pre_setup), us(p.reset), us(p.begin), us(p.record),
        us(p.end), us(p.submit), us(p.gpu_wait),
        us(p.readback), us(p.total()),
    );
}

fn us(d: Duration) -> u64 {
    d.as_micros() as u64
}

fn median_phases(samples: &[(u32, ForwardTokenProfile)]) -> ForwardTokenProfile {
    let mut pre: Vec<Duration> = samples.iter().map(|(_, p)| p.pre_setup).collect();
    let mut reset: Vec<Duration> = samples.iter().map(|(_, p)| p.reset).collect();
    let mut begin: Vec<Duration> = samples.iter().map(|(_, p)| p.begin).collect();
    let mut rec: Vec<Duration> = samples.iter().map(|(_, p)| p.record).collect();
    let mut end: Vec<Duration> = samples.iter().map(|(_, p)| p.end).collect();
    let mut sub: Vec<Duration> = samples.iter().map(|(_, p)| p.submit).collect();
    let mut wait: Vec<Duration> = samples.iter().map(|(_, p)| p.gpu_wait).collect();
    let mut rb: Vec<Duration> = samples.iter().map(|(_, p)| p.readback).collect();
    for v in [&mut pre, &mut reset, &mut begin, &mut rec, &mut end, &mut sub, &mut wait, &mut rb] {
        v.sort();
    }
    let mid = samples.len() / 2;
    ForwardTokenProfile {
        pre_setup: pre[mid],
        reset: reset[mid],
        begin: begin[mid],
        record: rec[mid],
        end: end[mid],
        submit: sub[mid],
        gpu_wait: wait[mid],
        readback: rb[mid],
    }
}
