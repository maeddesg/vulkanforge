//! Shared server state: the loaded model, tokenizer, GPU pipeline,
//! and per-request concurrency controls.
//!
//! Architecture references:
//! - §5.2 — `AppState` shape (model_id, semaphore, session mutex).
//! - §5.3 — single-request concurrency (Semaphore(1) + 429 reject).
//! - §6.x — `ServerSession` owns everything the chat handler needs.
//!
//! All the heavy Vulkan/GPU bits live behind a synchronous
//! [`std::sync::Mutex`] inside [`AppState`]. Handlers acquire a
//! [`tokio::sync::Semaphore`] permit FIRST (concurrency gate), then
//! `tokio::task::spawn_blocking` over the locked session. The
//! Mutex is held only inside the blocking task, so the runtime
//! never needs `Send` across `.await` points.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use gpu_allocator::vulkan::Allocator;
use tokio::sync::Semaphore;

use crate::backend::vulkan::chat::ChatSession;
use crate::backend::vulkan::chat_template::ChatTemplate;
use crate::backend::vulkan::commands::CommandContext;
use crate::backend::vulkan::device::VulkanDevice;
use crate::backend::vulkan::gguf::{GgufFile, ModelConfig};
use crate::backend::vulkan::loader::LoadedModel;
use crate::backend::vulkan::pipeline_registry::PipelineRegistry;
use crate::backend::vulkan::tokenizer::Tokenizer;

/// Top-level state shared across all axum handlers.
///
/// Always wrapped in [`std::sync::Arc`] before being handed to
/// [`axum::extract::State`].
pub struct AppState {
    /// `id` the server reports in `/v1/models` and the `model`
    /// field of every chat completion response. Lowercased basename
    /// of the model path unless overridden via `--served-model-name`
    /// (Sprint 4).
    pub model_id: String,

    /// Path the model was loaded from (kept for `/health` debug).
    pub model_path: PathBuf,

    /// Unix epoch seconds at server startup. Reported in
    /// `/v1/models` and used for uptime in `/health`.
    pub started_at: u64,

    /// Single-slot semaphore enforcing concurrency = 1 per
    /// Decision §5.3. `try_acquire_owned()` returns `Err` immediately
    /// when another request is in flight → handler returns 429.
    /// `Arc` so the owned-permit type is `'static`, allowing the
    /// permit to move into `tokio::task::spawn_blocking`.
    pub permit: Arc<Semaphore>,

    /// Holds the loaded model, tokenizer, and the
    /// long-lived Vulkan / pipeline objects. Locked only inside
    /// [`tokio::task::spawn_blocking`] tasks (the Mutex is sync).
    pub session: Mutex<ServerSession>,

    /// Default ThinkFilter state when a request doesn't set
    /// `chat_template_kwargs.enable_thinking` explicitly. `true`
    /// (= filter ON) preserves Sprint-3+ CLI behaviour. Set to
    /// `false` via `vulkanforge serve --no-think-filter` for
    /// clients like Open WebUI that can't pass the kwargs (those
    /// clients otherwise see empty content for Qwen3-style models
    /// when the `<think>` block consumes the whole reply).
    pub default_think_filter: bool,
}

impl AppState {
    pub fn new(
        model_id: String,
        model_path: PathBuf,
        session: ServerSession,
        default_think_filter: bool,
    ) -> Self {
        Self {
            model_id,
            model_path,
            started_at: unix_now_secs(),
            permit: Arc::new(Semaphore::new(1)),
            session: Mutex::new(session),
            default_think_filter,
        }
    }
}

/// All the per-instance GPU + model state needed to serve a chat
/// completion. Re-used across requests; the chat session inside
/// gets `reset()` between requests so v0.4 stays stateless
/// (Decision §3 — no prefix-cache, fresh KV per call).
pub struct ServerSession {
    pub dev: VulkanDevice,
    pub allocator: Allocator,
    pub registry: PipelineRegistry,
    pub cmd_ctx: CommandContext,
    pub model: LoadedModel,
    pub gguf: GgufFile,
    pub cfg: ModelConfig,
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    /// Wraps `Forward` (which owns the KV cache). The chat session's
    /// `system_prompt` is overwritten per-request because OpenAI
    /// puts system content into the `messages[]` array, not into
    /// some out-of-band session config.
    pub chat: ChatSession,
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
