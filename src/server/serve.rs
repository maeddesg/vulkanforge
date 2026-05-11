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
use crate::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use crate::backend::vulkan::loader::LoadedModel;
use crate::backend::vulkan::pipeline_registry::{default_cache_path, PipelineRegistry};
use crate::backend::vulkan::tokenizer::Tokenizer;

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
    let state = Arc::new(AppState::new(
        model_id,
        args.model.clone(),
        session,
        !args.no_think_filter,
    ));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(serve_inner(state, &args))
}

async fn serve_inner(state: Arc<AppState>, args: &ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let app = build_router(state, args.cors);
    let addr = format!("{}:{}", args.host, args.port);
    eprintln!("VulkanForge API server listening on http://{addr}");
    eprintln!("  endpoints: POST /v1/chat/completions  GET /v1/models  GET /health");
    eprintln!("  CORS:      {}", if args.cors { "enabled (any origin)" } else { "off (same-origin)" });
    eprintln!("  press Ctrl+C to shut down");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
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

    let max_context = args.ctx_size.unwrap_or(DEFAULT_CTX_SIZE);
    let cache_path = default_cache_path();
    let (registry, _pipelines_loaded) =
        PipelineRegistry::new_with_max_seq(&dev.device, cache_path.as_deref(), max_context)?;
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

    let gguf = GgufFile::open(&args.model)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let model = LoadedModel::load(&dev, &mut allocator, &gguf)?;
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
            per_layer_head_dim: cfg
                .gemma4
                .as_ref()
                .map(|g| g.layers.iter().map(|s| s.head_dim).collect()),
            per_layer_n_kv_heads: cfg
                .gemma4
                .as_ref()
                .map(|g| g.layers.iter().map(|s| s.n_kv_heads).collect()),
        },
    )?;
    kv_cache.zero_fill(&dev)?;
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)?;

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
    })
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
