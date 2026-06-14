//! `vulkanforge serve` entry point.
//!
//! Architecture §5.4 (CLI integration), §5.5 (router build), §5.6
//! (graceful shutdown). Sprint 2 covers the GGUF-load path; the
//! SafeTensors-directory variant is deferred to Sprint 4 polish.
//!
//! Concurrency model (§5.3): single Semaphore permit guards
//! request entry. The handlers acquire the permit in
//! `try_acquire()` mode → 429 on contention rather than queueing.

use std::path::PathBuf;
use std::sync::Arc;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::backend::vulkan::chat::ChatSession;
use crate::backend::vulkan::chat_template::ChatTemplate;
use crate::backend::vulkan::commands::CommandContext;
use crate::backend::vulkan::device::VulkanDevice;
use crate::backend::vulkan::forward::Forward;
use crate::backend::vulkan::gguf::{GgufFile, ModelConfig};
use crate::backend::vulkan::kv_cache::{
    kv_bytes_per_token, kv_dtype_from_env, KvCache, KvCacheConfig,
};
use crate::backend::vulkan::loader::LoadedModel;
use crate::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use crate::backend::vulkan::tokenizer::Tokenizer;

use super::auto_ctx::{self, CtxBound};
use super::routes::build_router;
use super::state::{AppState, ServerSession};

/// Default KV-cache capacity if `--ctx-size` isn't passed. Matches
/// `vulkanforge chat`'s `MAX_SEQ_LEN` constant.
const DEFAULT_CTX_SIZE: u32 = 2048;

pub struct ServeArgs {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub cors: bool,
    pub tokenizer_from: Option<PathBuf>,
    pub ctx_size: Option<u32>,
    /// Override the lowercased-basename default with a custom id
    /// reported by `/v1/models` and the chat completion `model` field.
    pub served_model_name: Option<String>,
    /// When `true`, the server-wide ThinkFilter default flips from
    /// ON to OFF. Useful for Qwen3-style models behind clients that
    /// can't pass `chat_template_kwargs.enable_thinking: false`
    /// (e.g. Open WebUI): with the filter off by default, the
    /// `<think>...</think>` block reaches the client verbatim
    /// instead of being stripped to an empty content string.
    /// Per-request override via `chat_template_kwargs` still wins.
    pub no_think_filter: bool,
}

pub fn run(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    // SafeTensors directories are valid models but need a different
    // load path (see `main.rs::run_chat_safetensors`): host-resident
    // embeddings via `EmbeddingSource::Host`, `apply_pre_device` for
    // FP8/quantize-on-load detection, and `Tokenizer::from_hf_dir`
    // for the JSON tokenizer. Reusing the CLI's `ChatSession` would
    // need a refactor (the helper hard-codes `EmbeddingSource::Gguf`).
    // Scope-deferred to v0.4.1 — the FP8 SafeTensors smoke target
    // ships through `vulkanforge chat` today, so the gap doesn't
    // block the v0.4.0 server release.
    if args.model.is_dir() {
        return Err(
            "SafeTensors directory models aren't supported by `vulkanforge serve` in v0.4.0. \
             Use `vulkanforge chat --model <dir>` for FP8/HF SafeTensors models; \
             SafeTensors serve support is planned for v0.4.1 (host-embed pipeline refactor)."
                .into(),
        );
    }

    let model_id = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| derive_model_id(&args.model));

    let session = load_gguf_session(&args)?;

    // Stufe A — bring up the server-side memory store (SQLiteGraph + embedder,
    // embedded). Eager-loads the embedder so the first request doesn't pay it.
    // A failure here (e.g. the model can't be fetched on a first, offline
    // start) is logged and the server runs WITHOUT memory (`/memory/*` → 503)
    // rather than refusing to serve inference.
    let memory = match crate::server::memory::MemoryStore::new(memory_db_path()) {
        Ok(m) => Some(std::sync::Arc::new(m)),
        Err(e) => {
            eprintln!("VulkanForge: memory subsystem DISABLED — init failed: {e}");
            None
        }
    };

    let state = Arc::new(AppState::new(
        model_id,
        args.model.clone(),
        session,
        !args.no_think_filter,
        memory,
    ));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(serve_inner(state, &args))
}

async fn serve_inner(state: Arc<AppState>, args: &ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Keep a clone so we can recover sole ownership for an ordered GPU
    // teardown once axum's graceful shutdown returns. `build_router` moves
    // the other clone into the router's `with_state`.
    let teardown_state = Arc::clone(&state);
    let app = build_router(state, args.cors);
    let addr = format!("{}:{}", args.host, args.port);
    eprintln!("VulkanForge API server listening on http://{addr}");
    eprintln!("  endpoints: POST /v1/chat/completions  POST /v1/completions  GET /v1/models  GET /health");
    eprintln!("  CORS:      {}", if args.cors { "enabled (any origin)" } else { "off (same-origin)" });
    eprintln!("  press Ctrl+C to shut down");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Graceful shutdown returned. First flush the memory store (CPU/SQLite —
    // no `device_wait_idle`; persist HNSW topology + WAL-checkpoint), then the
    // GPU teardown. Order is independent (disjoint resources) but memory-first
    // is tidy.
    if let Some(mem) = &teardown_state.memory {
        mem.shutdown();
    }

    // axum stopped accepting new connections and awaited every in-flight
    // request (each fence-waits inside `one_shot`), then the serve future
    // dropped the router — the only other `AppState` ref. Recover sole
    // ownership and tear the GPU state down explicitly and in order
    // (`device_wait_idle` → `.destroy()` chain → allocator → device). Without
    // this the `Arc`/`AppState` drop runs `ServerSession`'s field drop, which
    // never calls the `.destroy()` chain (every child object leaks →
    // `vkDestroyDevice` flags them) and used to free memory against a
    // destroyed device → SIGSEGV.
    teardown_gpu_state(teardown_state);
    Ok(())
}

/// Resolve the memory-store db path: `$VF_MEMORY_DB` if set, else
/// `~/.vulkanforge/memory.db` (a sibling `embed-cache/` holds the model).
fn memory_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("VF_MEMORY_DB") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".vulkanforge").join("memory.db")
}

/// Recover sole ownership of the GPU state and run [`ServerSession::teardown`].
///
/// Falls back to the implicit `Drop` (no UAF — `ServerSession`'s
/// allocator-before-device field order is the safety net) and a loud log if
/// some `AppState` reference unexpectedly outlived graceful shutdown, rather
/// than panicking during shutdown.
fn teardown_gpu_state(state: Arc<AppState>) {
    match Arc::try_unwrap(state) {
        Ok(app_state) => {
            let session = app_state
                .session
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            session.teardown();
            eprintln!("VulkanForge: GPU teardown complete.");
        }
        Err(state) => {
            eprintln!(
                "VulkanForge: shutdown — {} extra AppState reference(s) still live; \
                 skipping explicit GPU teardown (relying on Drop; allocator-before-device \
                 field order avoids a UAF, but child objects may leak).",
                Arc::strong_count(&state) - 1
            );
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => eprintln!("install SIGTERM handler failed: {e}"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => eprintln!("\n[Ctrl+C] shutting down…"),
        _ = terminate => eprintln!("\n[SIGTERM] shutting down…"),
    }
}

// =========================================================================
// Model load (GGUF only in Sprint 2)
// =========================================================================

fn load_gguf_session(args: &ServeArgs) -> Result<ServerSession, Box<dyn std::error::Error>> {
    let dev = VulkanDevice::new()?;
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })?;

    // Parse model metadata BEFORE the pipeline registry: auto-ctx-size
    // needs n_layers/n_kv_heads/head_dim/context_length, and the registry
    // bakes `max_seq` into a spec-constant — so the context must be decided
    // first. (Reordered from the old "registry then gguf" sequence; the
    // registry only depends on the device + the chosen max_seq.)
    let gguf = GgufFile::open(&args.model)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;

    // Per-layer KV tables (heterogeneous Gemma-4 sliding/full; Qwen3.6
    // recurrent layers contribute 0 KV heads). Derived once here, then
    // reused for BOTH the KV-per-token estimate (auto-ctx) and the
    // KvCacheConfig below — single source of truth, no divergence.
    let per_layer_head_dim: Option<Vec<u32>> = cfg
        .gemma4
        .as_ref()
        .map(|g| g.layers.iter().map(|s| s.head_dim).collect());
    // Sprint D2 — Qwen3.6 per-layer KV-heads (0 for recurrent Linear-Attn
    // layers, n_head_kv_full_attn for Full-Attn); drops the FP8 KV cache
    // from 520 MB to ~136 MB.
    let per_layer_n_kv_heads: Option<Vec<u32>> = cfg
        .gemma4
        .as_ref()
        .map(|g| g.layers.iter().map(|s| s.n_kv_heads).collect::<Vec<_>>())
        .or_else(|| {
            cfg.qwen35.as_ref().map(|q| {
                (0..q.block_count)
                    .map(|l| if q.is_full_attention_layer(l) { q.n_head_kv_full_attn } else { 0 })
                    .collect::<Vec<_>>()
            })
        });

    let max_context = resolve_ctx_size(
        args,
        &dev,
        &cfg,
        &args.model,
        per_layer_head_dim.as_deref(),
        per_layer_n_kv_heads.as_deref(),
    );

    let cache_path = default_cache_path();
    let (registry, _pipelines_loaded) =
        PipelineRegistry::new_with_max_seq(&dev.device, cache_path.as_deref(), max_context)?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    // serve path: no --gamma-from CLI option yet (Sprint 52F scope was
    // chat only; serve is a follow-up).
    let model = LoadedModel::load(&dev, &mut allocator, &gguf, None)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    if max_context > cfg.context_length {
        eprintln!(
            "VulkanForge: --ctx-size {} exceeds model context_length {} — \
             generation past the training horizon may produce nonsense.",
            max_context, cfg.context_length,
        );
    }
    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: max_context,
            per_layer_head_dim,
            per_layer_n_kv_heads,
        },
    )?;
    kv_cache.zero_fill(&dev)?;
    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;
    // Mirror the CLI setup (main.rs:819/821, :1275/1277): register the
    // Sprint-B bucket handles, then init the GPU-side MoE router. Both are
    // no-ops for non-bucketed / non-MoE models, but Gemma-4 (MoE) needs
    // `init_moe_router_gpu` or `moe_router_gpu` stays `None` and the
    // executor falls back to the legacy CPU slot-loop path that is
    // "retained for non-Gemma-4 / unit tests" (executor/moe.rs:382) →
    // garbage decode. `init_moe_router_gpu` also allocates the grouped /
    // batched-decode scratch (VF_MOE_GROUPED default-ON).
    forward.register_buckets(&model);
    forward.init_moe_router_gpu(&dev, &mut allocator, &model, max_context)?;

    let template = ChatTemplate::detect(&gguf, &tokenizer);

    // Empty system prompt at construction time — request handlers
    // overwrite it with the request's system message (if any)
    // before each generation (Decision §3 / handler reset path).
    let chat = ChatSession::new_with_template(forward, String::new(), template);

    Ok(ServerSession {
        dev,
        allocator,
        registry,
        cmd_ctx,
        model,
        gguf,
        cfg,
        tokenizer,
        template,
        chat,
        cached_tokens: Vec::new(),
    })
}

// =========================================================================
// Context-size resolution (explicit override or VRAM-aware auto)
// =========================================================================

/// Resolve the KV-cache context size.
///
/// An explicit `--ctx-size N` is used verbatim (unchanged behavior). When
/// omitted, auto-size from live VRAM (`VK_EXT_memory_budget`) + model
/// metadata and print a one-line, fully-itemized rationale — no silent
/// magic, so the user can reproduce the decision and override it. If the
/// VRAM budget can't be read (extension absent / CI), fall back to the
/// fixed default rather than guessing.
fn resolve_ctx_size(
    args: &ServeArgs,
    dev: &VulkanDevice,
    cfg: &ModelConfig,
    model_path: &std::path::Path,
    per_layer_head_dim: Option<&[u32]>,
    per_layer_n_kv_heads: Option<&[u32]>,
) -> u32 {
    if let Some(n) = args.ctx_size {
        eprintln!("VulkanForge: ctx-size = {n} (explicit --ctx-size override)");
        return n;
    }

    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let Some(free) = dev.free_vram_bytes() else {
        eprintln!(
            "VulkanForge: auto ctx-size unavailable (VK_EXT_memory_budget not reported) — \
             using default {DEFAULT_CTX_SIZE}. Pass --ctx-size N to override."
        );
        return DEFAULT_CTX_SIZE;
    };

    // Weights footprint = GGUF on-disk size (the quantized tensors are
    // mmap'd and uploaded as-is, so for GGUF this closely tracks VRAM
    // residency). `free` is measured here, before weights + registry +
    // scratch are allocated, so the reserve must cover all of those.
    let weights = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    let elem = kv_dtype_from_env().element_size();
    let kv_bpt = kv_bytes_per_token(
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim,
        per_layer_head_dim,
        per_layer_n_kv_heads,
        elem,
    );
    let reserve = auto_ctx::reserve_bytes();
    // Hardware LDS ceiling: scalar_attn.comp's `scores[MAX_SEQ]` is built
    // unconditionally, so MAX_SEQ × 4 B must fit maxComputeSharedMemorySize
    // (else pipeline creation aborts the load).
    let lds_cap = auto_ctx::lds_ctx_cap(dev.max_compute_shared_memory_bytes());
    let a = auto_ctx::compute_auto_ctx(
        free,
        weights,
        reserve,
        kv_bpt,
        cfg.context_length,
        lds_cap,
        auto_ctx::SANE_CAP,
    );

    let bound = match a.bound {
        CtxBound::Vram => "VRAM".to_string(),
        CtxBound::ModelMax => format!("model max context {}", cfg.context_length),
        CtxBound::HwLds => format!("hardware LDS limit {lds_cap}"),
        CtxBound::SaneCap => format!("sane cap {}", auto_ctx::SANE_CAP),
    };
    let kv_mib_per_tok = kv_bpt as f64 / (1024.0 * 1024.0);
    eprintln!(
        "VulkanForge: auto ctx-size = {} (free {:.2}G − weights {:.2}G − reserve {:.2}G \
         = {:.2}G for KV / {:.3} MiB/tok; bound: {bound}; override with --ctx-size N)",
        a.ctx,
        gib(free),
        gib(weights),
        gib(reserve),
        gib(a.avail_for_kv),
        kv_mib_per_tok,
    );
    a.ctx
}

// =========================================================================
// Model-ID helper
// =========================================================================

/// Derive the served-model-name from the path: lowercased basename
/// without extension.
///
///   /a/b/Qwen3-8B-Q4_K_M.gguf   → "qwen3-8b-q4_k_m"
///   /a/b/SomeDir                → "somedir"
///   ./model                     → "model"
pub fn derive_model_id(path: &std::path::Path) -> String {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    name.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::derive_model_id;
    use std::path::PathBuf;

    #[test]
    fn id_lowercases_basename_and_strips_extension() {
        assert_eq!(derive_model_id(&PathBuf::from("/m/Qwen3-8B-Q4_K_M.gguf")), "qwen3-8b-q4_k_m");
    }

    #[test]
    fn id_for_directory_path_uses_dir_name() {
        assert_eq!(derive_model_id(&PathBuf::from("/m/SomeDir")), "somedir");
    }

    #[test]
    fn id_falls_back_to_model_when_path_empty() {
        // file_stem on an empty path returns None.
        assert_eq!(derive_model_id(&PathBuf::from("")), "model");
    }
}
