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
use crate::backend::vulkan::decode::{
    generate_from_tokens, EmbeddingSource, GenerateConfig, GenerateResult,
};
use crate::backend::vulkan::device::VulkanDevice;
use crate::backend::vulkan::gguf::{GgufFile, ModelConfig};
use crate::backend::vulkan::loader::LoadedModel;
use crate::backend::vulkan::pipeline_registry::PipelineRegistry;
use crate::backend::vulkan::tokenizer::Tokenizer;

/// Gated cross-request KV prefix-reuse switch (`VF_KV_PREFIX_REUSE`,
/// default OFF). Checked per-request so the OFF path stays zero-overhead
/// and bit-identical to the v0.4 stateless behavior.
pub fn kv_prefix_reuse_enabled() -> bool {
    std::env::var("VF_KV_PREFIX_REUSE").as_deref() == Ok("1")
}

/// Longest common token prefix length — exact id match, position by
/// position, up to the first divergence.
fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

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
    /// Sprint KV-prefix-reuse — the full token sequence
    /// `[prompt + generated]` of the LAST cleanly-completed request,
    /// whose KV is still resident in `chat.forward.kv_cache`. Empty
    /// when there's nothing to reuse (fresh server, or after an
    /// invalidation). Only consulted when `VF_KV_PREFIX_REUSE` is on.
    pub cached_tokens: Vec<u32>,
}

impl ServerSession {
    /// Run a generation for the full token sequence `full_tokens` with
    /// gated cross-request KV prefix reuse.
    ///
    /// `reuse == false` (the v0.4 default): reset the KV and prefill the
    /// whole sequence from position 0 — **bit-identical** to the
    /// pre-reuse stateless behavior.
    ///
    /// `reuse == true`: keep the prior request's resident KV and prefill
    /// only the suffix after the longest common token prefix `k`
    /// (`k = min(lcp, len-1)` so ≥1 token is always prefilled and the
    /// decode seeds from fresh logits). The reused KV `[0..k)` is, by
    /// construction, byte-identical to a fresh prefill of the same `k`
    /// tokens at the same positions (same ids, same RoPE positions, same
    /// deterministic FP8 quantization). On clean completion the
    /// `[prompt + generated]` sequence is retained for the next request;
    /// on error the KV + cache are invalidated (a partial/inconsistent
    /// KV must never be reused).
    ///
    /// `on_token` is the caller's per-token sink (ThinkFilter / SSE); the
    /// generated ids are collected internally for the retained sequence.
    pub fn generate_reuse(
        &mut self,
        full_tokens: &[u32],
        cfg_g: &GenerateConfig,
        reuse: bool,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerateResult, Box<dyn std::error::Error>> {
        // Sprint stream-engagement — gated per-request instrumentation
        // (`VF_KV_REUSE_DEBUG`). Logs identically from both the streaming
        // and non-streaming handlers (shared fn), so the prefill-token
        // count / prefill-time comparison is apples-to-apples. Never on
        // in production.
        let debug = std::env::var("VF_KV_REUSE_DEBUG").as_deref() == Ok("1");
        let cached_len_before = self.cached_tokens.len();

        // --- KV prepare: choose start_pos `k`, set the seq-len counter ---
        let k: u32 = if reuse {
            let lcp = longest_common_prefix(&self.cached_tokens, full_tokens);
            let mut k = lcp.min(full_tokens.len().saturating_sub(1));
            // TEETH-test hook (gated, never set in production): intentionally
            // over-reuse past the true common prefix to prove the byte-ident
            // gate catches a wrong reuse. Bounded by the resident KV length
            // and the new sequence length.
            if let Ok(n) = std::env::var("VF_KV_REUSE_OVERSHOOT") {
                if let Ok(extra) = n.parse::<usize>() {
                    k = (k + extra)
                        .min(self.cached_tokens.len())
                        .min(full_tokens.len().saturating_sub(1));
                }
            }
            if k == 0 {
                self.cached_tokens.clear();
                self.chat.forward.kv_cache.reset();
                0
            } else {
                // Keep the resident KV [0..k); the suffix prefill at
                // base_pos = k overwrites [k..) and resets the counter to
                // k + suffix. `reset()` only zeroes the counter (the KV
                // bytes persist), so this is the whole reuse mechanism.
                self.chat.forward.kv_cache.current_seq_len = k as u32;
                k as u32
            }
        } else {
            self.cached_tokens.clear();
            self.chat.forward.kv_cache.reset();
            0
        };

        let mut gen_ids: Vec<u32> = Vec::new();
        let result = {
            let mut collect = |id: u32, raw: &str| {
                gen_ids.push(id);
                on_token(id, raw);
            };
            let ServerSession {
                dev, registry, cmd_ctx, model, gguf, cfg, tokenizer, chat, ..
            } = self;
            generate_from_tokens(
                &mut chat.forward,
                dev, registry, cmd_ctx, model,
                EmbeddingSource::Gguf(gguf),
                cfg, tokenizer,
                &full_tokens[k as usize..], k, cfg_g, false, &mut collect,
            )
        };

        match &result {
            Ok(_) => {
                if reuse {
                    let mut seq = Vec::with_capacity(full_tokens.len() + gen_ids.len());
                    seq.extend_from_slice(full_tokens);
                    seq.extend_from_slice(&gen_ids);
                    self.cached_tokens = seq;
                } else {
                    self.cached_tokens.clear();
                }
            }
            Err(_) => {
                // Mid-decode failure → KV state is inconsistent.
                self.cached_tokens.clear();
                self.chat.forward.kv_cache.reset();
            }
        }

        if debug {
            let prefill_tokens = full_tokens.len() as u32 - k;
            let prefill_ms = result
                .as_ref()
                .map(|r| r.prefill_time.as_secs_f64() * 1e3)
                .unwrap_or(-1.0);
            eprintln!(
                "[VF_KV_REUSE] reuse_requested={reuse} cached_len_before={cached_len_before} \
                 new_len={} k={k} prefill_tokens={prefill_tokens} prefill_ms={prefill_ms:.1} \
                 cached_len_after={}",
                full_tokens.len(),
                self.cached_tokens.len(),
            );
        }
        result
    }

    /// Drop the retained KV-reuse cache + reset the KV. Used when a
    /// stream is cancelled mid-decode (the partial KV must never be
    /// reused) and on session teardown paths.
    pub fn kv_invalidate(&mut self) {
        self.cached_tokens.clear();
        self.chat.forward.kv_cache.reset();
    }
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_empty_cache_is_zero() {
        assert_eq!(longest_common_prefix(&[], &[1, 2, 3]), 0);
    }

    #[test]
    fn lcp_no_match_is_zero() {
        assert_eq!(longest_common_prefix(&[9, 8, 7], &[1, 2, 3]), 0);
    }

    #[test]
    fn lcp_partial_stops_at_first_divergence() {
        // shared [10,11,12] then diverge (13 vs 99)
        assert_eq!(longest_common_prefix(&[10, 11, 12, 13, 14], &[10, 11, 12, 99]), 3);
    }

    #[test]
    fn lcp_full_when_new_is_prefix_of_cache() {
        assert_eq!(longest_common_prefix(&[1, 2, 3, 4, 5], &[1, 2, 3]), 3);
    }

    #[test]
    fn lcp_identical() {
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn k_caps_at_len_minus_one_so_one_token_always_prefilled() {
        // Exact-match case: lcp == new.len(); k must be capped to len-1
        // so the decode seeds from a fresh prefill of >=1 token.
        let cache = vec![1u32, 2, 3, 4];
        let new = vec![1u32, 2, 3, 4];
        let lcp = longest_common_prefix(&cache, &new);
        let k = lcp.min(new.len().saturating_sub(1));
        assert_eq!(lcp, 4);
        assert_eq!(k, 3, "k must be len-1, not the full length");
    }

    #[test]
    fn reuse_env_flag_default_off() {
        // Without the env set, reuse must be OFF (bit-identical v0.4 path).
        // (Can't safely mutate process env in parallel tests; assert the
        // parse contract: only "1" enables.)
        assert!(!kv_prefix_reuse_enabled() || std::env::var("VF_KV_PREFIX_REUSE").as_deref() == Ok("1"));
    }
}
