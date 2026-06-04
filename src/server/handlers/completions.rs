//! `POST /v1/completions` (and `/completions` alias) — legacy
//! text-completion.
//!
//! Maximal reuse of the chat path. The ONLY semantic difference is the
//! prompt: a **raw string** tokenized directly (with special-token
//! parsing so a pre-rendered chat-template string round-trips) and fed
//! to the SAME [`generate_from_tokens`] core — **no chat template is
//! applied**. Sampling mapping, the concurrency permit, KV reset,
//! `max_tokens` clamping, the ThinkFilter, the SSE transport, and the
//! usage computation are all shared with `chat.rs` / `sampling.rs` /
//! `stream.rs`.
//!
//! Response shape is the OpenAI `text_completion` form (`cmpl-…` id,
//! `choices[].text`).

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;

use crate::backend::vulkan::decode::{GenerateConfig, Sampling, ThinkFilter};
use crate::server::cancel::CancelToken;
use crate::server::error::ApiError;
use crate::server::handlers::chat::{clamp_max_tokens, seed_from_clock};
use crate::server::sampling::SamplingParams;
use crate::server::state::{AppState, ServerSession};
use crate::server::stream::{sse_response, ChunkMeta, StreamEvent, StreamKind};
use crate::server::types::request::CompletionRequest;
use crate::server::types::response::{
    CompletionChoice, CompletionResponse, FinishReason, Usage,
};

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let normalised = validate_and_normalise(&req, state.default_think_filter)?;

    // Same 5 s-wait concurrency gate as chat (§5.3). Owned permit moves
    // into the blocking task.
    let permit = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.permit.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(_)) => {
            return Err(ApiError::internal(
                "request-permit semaphore closed",
                "internal_error",
            ))
        }
        Err(_) => return Err(ApiError::ServerBusy),
    };

    if normalised.stream {
        Ok(start_streaming(state, normalised, permit))
    } else {
        let state_clone = state.clone();
        let join = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut session_guard = state_clone
                .session
                .lock()
                .map_err(|_| ApiError::internal("session mutex poisoned", "internal_error"))?;
            process_request(&mut session_guard, &state_clone.model_id, normalised)
        });
        let body = join
            .await
            .map_err(|e| ApiError::internal(format!("task join: {e}"), "internal_error"))??;
        Ok((axum::http::StatusCode::OK, Json(body)).into_response())
    }
}

// =========================================================================
// Validation
// =========================================================================

#[derive(Debug)]
struct NormalisedRequest {
    prompt: String,
    sampling: SamplingParams,
    stream: bool,
    include_usage: bool,
    enable_thinking: bool,
}

fn validate_and_normalise(
    req: &CompletionRequest,
    default_think_filter: bool,
) -> Result<NormalisedRequest, ApiError> {
    // §8.1 unsupported_n — mirror chat.
    if let Some(n) = req.n {
        if n != 1 {
            return Err(ApiError::invalid_with_param(
                format!("only n=1 supported, got n={n}"),
                "unsupported_n",
                "n",
            ));
        }
    }

    // Empty prompt is allowed by OpenAI (completes from BOS-less raw),
    // but a zero-length prompt has no anchor — reject for clarity.
    if req.prompt.is_empty() {
        return Err(ApiError::invalid_with_param(
            "prompt must not be empty",
            "empty_prompt",
            "prompt",
        ));
    }

    // Accepted-but-ignored fields → one warn-log line each (bounded by
    // client cadence; the handler runs once per request).
    macro_rules! warn_ignored {
        ($cond:expr, $name:literal) => {
            if $cond {
                eprintln!(
                    "[vulkanforge serve] /v1/completions: `{}` accepted but ignored (not supported in v0.4)",
                    $name
                );
            }
        };
    }
    warn_ignored!(req.suffix.is_some(), "suffix");
    warn_ignored!(req.best_of.map(|b| b > 1).unwrap_or(false), "best_of");
    warn_ignored!(req.logprobs.is_some(), "logprobs");
    warn_ignored!(req.echo.unwrap_or(false), "echo");
    warn_ignored!(req.logit_bias.is_some(), "logit_bias");
    warn_ignored!(req.presence_penalty.map(|p| p != 0.0).unwrap_or(false), "presence_penalty");

    let sampling = SamplingParams::from_completion_request(req);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    Ok(NormalisedRequest {
        prompt: req.prompt.clone(),
        sampling,
        stream: req.stream,
        include_usage,
        // Completions has no `chat_template_kwargs`; apply the server
        // default (the `--no-think-filter` switch), same as chat does
        // when the client omits the kwargs.
        enable_thinking: default_think_filter,
    })
}

// =========================================================================
// Processing (runs inside spawn_blocking) — shared core, no template
// =========================================================================

/// Completions counterpart of chat's `prepare_generate`: reset KV,
/// tokenize the RAW prompt (with special-token parsing — no chat
/// template), clamp `max_tokens`, build the `GenerateConfig`. Every
/// step except the tokenization source is identical to chat.
fn prepare_generate(
    session: &mut ServerSession,
    req: &NormalisedRequest,
    cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(Vec<u32>, GenerateConfig), ApiError> {
    // KV reset / prefix-reuse is decided in `generate_reuse`.
    let max_tokens = req.sampling.max_tokens.unwrap_or(200).max(1);

    // THE one difference from chat: raw prompt → tokens, no template.
    let prefill_tokens = session.tokenizer.encode_with_special(&req.prompt);
    if prefill_tokens.is_empty() {
        return Err(ApiError::invalid_with_param(
            "prompt tokenized to zero tokens",
            "empty_prompt",
            "prompt",
        ));
    }

    let max_seq = session.chat.forward.kv_cache.config.max_seq_len;
    let max_tokens = clamp_max_tokens(prefill_tokens.len() as u32, max_tokens, max_seq)?;

    let vf_sampling = Sampling {
        temperature: req.sampling.temperature,
        top_k: req.sampling.top_k.unwrap_or(0),
        top_p: req.sampling.top_p,
        repetition_penalty: req.sampling.repetition_penalty,
        seed: req.sampling.seed.unwrap_or_else(seed_from_clock),
    };
    let cfg_g = GenerateConfig {
        max_tokens,
        print_stream: false,
        think_filter: req.enable_thinking,
        sampling: vf_sampling,
        cancel_token,
    };
    Ok((prefill_tokens, cfg_g))
}

fn process_request(
    session: &mut ServerSession,
    model_id: &str,
    req: NormalisedRequest,
) -> Result<CompletionResponse, ApiError> {
    let reuse = crate::server::state::kv_prefix_reuse_enabled();
    let (prefill_tokens, cfg_g) = prepare_generate(session, &req, None)?;
    let max_seq = session.chat.forward.kv_cache.config.max_seq_len;

    let mut filter = if cfg_g.think_filter { Some(ThinkFilter::new()) } else { None };
    let mut on_token = move |_id: u32, raw: &str| {
        if let Some(f) = filter.as_mut() {
            let _ = f.push(raw);
        }
    };

    let result = session
        .generate_reuse(&prefill_tokens, &cfg_g, reuse, &mut on_token)
        .map_err(|e| {
            crate::server::handlers::chat::map_gen_err(
                e, prefill_tokens.len() as u32, cfg_g.max_tokens, max_seq,
            )
        })?;

    let finish_reason = if result.stopped_on_eos {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    let choice = CompletionChoice {
        text: result.visible_text,
        index: 0,
        logprobs: None,
        finish_reason,
    };
    let usage = Usage::new(result.prompt_tokens as u32, result.generated_tokens as u32);
    Ok(CompletionResponse::new(new_cmpl_id(), model_id.to_string(), choice, usage))
}

fn new_cmpl_id() -> String {
    let u = uuid::Uuid::new_v4();
    let mut s = String::with_capacity(5 + 32);
    s.push_str("cmpl-");
    s.push_str(&u.simple().to_string());
    s
}

// =========================================================================
// Streaming path — reuses the SSE transport with StreamKind::Completion
// =========================================================================

fn start_streaming(
    state: Arc<AppState>,
    req: NormalisedRequest,
    permit: OwnedSemaphorePermit,
) -> Response {
    let cmpl_id = new_cmpl_id();
    let model_id = state.model_id.clone();
    let cancel = CancelToken::new();
    let include_usage = req.include_usage;

    let (tx, rx) = mpsc::channel::<StreamEvent>(64);

    let state_for_task = state.clone();
    let cancel_for_task = cancel.clone();
    let cancel_for_sse = cancel.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut session_guard = match state_for_task.session.lock() {
            Ok(g) => g,
            Err(_) => {
                let _ = tx.blocking_send(StreamEvent::Error(ApiError::internal(
                    "session mutex poisoned",
                    "internal_error",
                )));
                return;
            }
        };
        run_streaming_generation(&mut session_guard, req, tx, cancel_for_task, include_usage);
    });

    let meta = ChunkMeta { id: cmpl_id, model: model_id, kind: StreamKind::Completion };
    sse_response(rx, meta, cancel_for_sse).into_response()
}

fn run_streaming_generation(
    session: &mut ServerSession,
    req: NormalisedRequest,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancelToken,
    include_usage: bool,
) {
    let (prefill_tokens, cfg_g) = match prepare_generate(session, &req, Some(cancel.as_arc())) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.blocking_send(StreamEvent::Error(e));
            return;
        }
    };
    let prefill_len = prefill_tokens.len() as u32;
    let reuse = crate::server::state::kv_prefix_reuse_enabled();
    let max_seq = session.chat.forward.kv_cache.config.max_seq_len;

    // text_completion has no header chunk; the SSE adapter maps a
    // Header event to zero output for StreamKind::Completion, so we
    // simply don't send one. (Sending one would also be a no-op.)

    let mut filter = if cfg_g.think_filter { Some(ThinkFilter::new()) } else { None };
    let tx_for_cb = tx.clone();
    let cancel_for_cb = cancel.clone();
    let mut on_token = move |_id: u32, raw: &str| {
        if cancel_for_cb.is_cancelled() {
            return;
        }
        if tx_for_cb.is_closed() {
            cancel_for_cb.cancel();
            return;
        }
        let visible: String = match filter.as_mut() {
            Some(f) => f.push(raw),
            None => raw.to_string(),
        };
        if visible.is_empty() {
            return;
        }
        if tx_for_cb.blocking_send(StreamEvent::Delta(visible)).is_err() {
            cancel_for_cb.cancel();
        }
    };

    let result = session.generate_reuse(&prefill_tokens, &cfg_g, reuse, &mut on_token);

    // Cancelled mid-decode → drop the partial KV (never reuse it).
    if cancel.is_cancelled() {
        session.kv_invalidate();
    }

    match result {
        Ok(g) => {
            let finish = if g.stopped_on_eos {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            let usage = Usage::new(g.prompt_tokens as u32, g.generated_tokens as u32);
            let _ = tx.blocking_send(StreamEvent::Final { finish, usage, include_usage });
        }
        Err(e) => {
            let api_err = crate::server::handlers::chat::map_gen_err(
                e, prefill_len, cfg_g.max_tokens, max_seq,
            );
            let _ = tx.blocking_send(StreamEvent::Error(api_err));
        }
    }
}

// =========================================================================
// Tests (no GPU — validation + id shape only)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req_from(j: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(j).unwrap()
    }

    #[test]
    fn minimal_prompt_validates() {
        let req = req_from(json!({"prompt": "The capital of France is"}));
        let n = validate_and_normalise(&req, true).unwrap();
        assert_eq!(n.prompt, "The capital of France is");
        assert!(!n.stream);
        assert!(n.enable_thinking);
    }

    #[test]
    fn empty_prompt_rejected() {
        let req = req_from(json!({"prompt": ""}));
        let e = validate_and_normalise(&req, true).unwrap_err();
        assert_eq!(e.code(), "empty_prompt");
    }

    #[test]
    fn n_greater_than_one_rejected() {
        let req = req_from(json!({"prompt": "x", "n": 2}));
        let e = validate_and_normalise(&req, true).unwrap_err();
        assert_eq!(e.code(), "unsupported_n");
    }

    #[test]
    fn unsupported_fields_warn_but_do_not_fail() {
        let req = req_from(json!({
            "prompt": "x",
            "suffix": "END", "best_of": 4, "logprobs": 5,
            "echo": true, "logit_bias": {"1": -100},
        }));
        // Must succeed despite the unsupported fields.
        let n = validate_and_normalise(&req, false).unwrap();
        assert_eq!(n.prompt, "x");
        assert!(!n.enable_thinking);
    }

    #[test]
    fn default_think_filter_propagates() {
        let on = validate_and_normalise(&req_from(json!({"prompt": "x"})), true).unwrap();
        let off = validate_and_normalise(&req_from(json!({"prompt": "x"})), false).unwrap();
        assert!(on.enable_thinking);
        assert!(!off.enable_thinking);
    }

    #[test]
    fn stream_options_include_usage_parsed() {
        let req = req_from(json!({
            "prompt": "x", "stream": true,
            "stream_options": {"include_usage": true},
        }));
        let n = validate_and_normalise(&req, true).unwrap();
        assert!(n.stream);
        assert!(n.include_usage);
    }

    #[test]
    fn cmpl_id_has_prefix_and_32_hex_suffix() {
        let id = new_cmpl_id();
        assert!(id.starts_with("cmpl-"));
        let hex = &id[5..];
        assert_eq!(hex.len(), 32);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn model_field_is_optional() {
        // Unlike chat, the legacy body may omit `model` entirely.
        let req = req_from(json!({"prompt": "x"}));
        assert!(req.model.is_none());
        let _ = validate_and_normalise(&req, true).unwrap();
    }
}
