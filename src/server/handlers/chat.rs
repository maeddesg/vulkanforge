//! `POST /v1/chat/completions` (and `/chat/completions` alias).
//!
//! Architecture §2.1 (endpoint), §6 (chat-template integration),
//! §7 (sampling mapping), §8 (error mapping).
//!
//! Sprint 5 scope (latest):
//! - **Non-streaming** (`stream: false`): blocking task → JSON body.
//! - **Streaming** (`stream: true`): mpsc-bridged SSE response.
//! - **Multi-turn history** — `messages` can contain assistant
//!   replies from prior turns; the entire history is re-rendered
//!   from scratch on each request (stateless per Decision §3,
//!   no prefix-cache).
//!
//! Validation order (§8.4) — `model` field is NOT validated
//! (Decision §2 / Merge-Sprint #1):
//! 1. `n ≤ 1`
//! 2. `messages` non-empty
//! 3. each role ≠ `tool`, each content ≠ `image_url`
//! 4. at most one `system` message, and only as `messages[0]`
//! 5. ≥1 `user` message, last entry must be `user`
//! 6. Acquire permit → else 429
//! 7. `spawn_blocking` → reset KV → `template.render_full_history`
//!    → `generate_from_tokens`
//! 8. Either: serialise the result into `ChatCompletionResponse`
//!    (non-streaming) OR pipe each visible token into the SSE
//!    adapter (streaming).
//!
//! Context-overflow propagation: `ChatSession::send_streaming`
//! returns `ChatError::ContextOverflow` which we map to a 400 with
//! `code = "context_length_exceeded"` (Decision §2).

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;

use crate::backend::vulkan::chat_template::{HistoryRole, RenderMessage};
use crate::backend::vulkan::decode::{
    generate_from_tokens, EmbeddingSource, GenerateConfig, Sampling, ThinkFilter,
};
use crate::server::cancel::CancelToken;
use crate::server::error::ApiError;
use crate::server::sampling::SamplingParams;
use crate::server::state::{AppState, ServerSession};
use crate::server::stream::{sse_response, ChunkMeta, StreamEvent};
use crate::server::types::request::{
    ChatCompletionRequest, ContentPart, MessageContent, Role,
};
use crate::server::types::response::{
    AssistantMessage, ChatCompletionResponse, Choice, FinishReason, Usage,
};

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // 1-4. Validate. Cheap, no GPU touch yet.
    let normalised = validate_and_normalise(&req)?;

    // 5. Concurrency gate. §5.3 — try_acquire_owned, 429 on miss.
    // Owned variant lets us move the permit into the blocking
    // thread; the borrowed variant (try_acquire) would tie the
    // permit's lifetime to the borrow of `state.permit`.
    let permit = state
        .permit
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::ServerBusy)?;

    if normalised.stream {
        // 6a. Streaming path — return SSE immediately; the GPU work
        // happens in a detached spawn_blocking task that feeds the
        // mpsc channel the SSE adapter is polling.
        Ok(start_streaming(state, normalised, permit))
    } else {
        // 6b. Non-streaming path — block until the response body
        // is ready, then return one JSON document.
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

/// Result of request validation: a normalised single-turn (system +
/// user) plus the resolved sampling params.
#[derive(Debug)]
struct NormalisedRequest {
    /// Full conversation history (possibly system + alternating
    /// user/assistant), validated to end with a user turn. Sprint 5
    /// renders this verbatim per-request — the GPU sees a fresh
    /// prefill at position 0 each time.
    history: Vec<(HistoryRole, String)>,
    sampling: SamplingParams,
    /// `true` when the request asked for `stream: true`. Decides
    /// between SSE adapter (Sprint 3) and a single JSON body.
    stream: bool,
    /// Populated from `stream_options.include_usage`. Only honoured
    /// when `stream` is true.
    include_usage: bool,
    /// `chat_template_kwargs.enable_thinking`. Default `true`
    /// preserves CLI/Sprint-3 behaviour. `false` disables the
    /// ThinkFilter so `<think>...</think>` blocks reach the client.
    enable_thinking: bool,
}

fn validate_and_normalise(req: &ChatCompletionRequest) -> Result<NormalisedRequest, ApiError> {
    // §8.1 unsupported_n
    if let Some(n) = req.n {
        if n != 1 {
            return Err(ApiError::invalid_with_param(
                format!("only n=1 supported, got n={n}"),
                "unsupported_n",
                "n",
            ));
        }
    }

    // §8.1 no_user_message
    if req.messages.is_empty() {
        return Err(ApiError::invalid_with_param(
            "messages array must not be empty",
            "no_user_message",
            "messages",
        ));
    }

    // §8.1 — collect history, reject unsupported roles/content,
    // enforce single optional leading system + non-empty user list +
    // user is the last message.
    let mut history: Vec<(HistoryRole, String)> = Vec::with_capacity(req.messages.len());
    let mut saw_user = false;
    let mut saw_system = false;
    for (i, msg) in req.messages.iter().enumerate() {
        match msg.role {
            Role::Tool => {
                return Err(ApiError::invalid_with_param(
                    "tool role not supported in v0.4",
                    "unsupported_role",
                    format!("messages[{i}].role"),
                ));
            }
            Role::System => {
                if saw_system {
                    return Err(ApiError::invalid_with_param(
                        "more than one system message",
                        "duplicate_system_message",
                        format!("messages[{i}].role"),
                    ));
                }
                if i != 0 {
                    return Err(ApiError::invalid_with_param(
                        "system message must be the first entry",
                        "invalid_message_order",
                        format!("messages[{i}].role"),
                    ));
                }
                saw_system = true;
                history.push((HistoryRole::System, content_to_string(&msg.content, i)?));
            }
            Role::User => {
                saw_user = true;
                history.push((HistoryRole::User, content_to_string(&msg.content, i)?));
            }
            Role::Assistant => {
                history.push((HistoryRole::Assistant, content_to_string(&msg.content, i)?));
            }
        }
    }
    if !saw_user {
        return Err(ApiError::invalid_with_param(
            "no user message present",
            "no_user_message",
            "messages",
        ));
    }
    // Last non-system entry must be a user turn — otherwise the
    // model doesn't know what to answer.
    match history.last() {
        Some((HistoryRole::User, _)) => {}
        _ => {
            return Err(ApiError::invalid_with_param(
                "last message must be role: \"user\"",
                "last_message_not_user",
                "messages",
            ));
        }
    }

    let sampling = SamplingParams::from_request(req);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let enable_thinking = req
        .chat_template_kwargs
        .as_ref()
        .map(|o| o.enable_thinking)
        .unwrap_or(true);

    // §7.1 — presence_penalty is accepted-but-ignored. Surface it
    // here so operators can spot clients that depend on it; the
    // handler runs once per request so the log volume is bounded
    // by client cadence.
    if let Some(pp) = req.presence_penalty {
        if pp != 0.0 {
            eprintln!(
                "[vulkanforge serve] presence_penalty={pp} ignored (not supported in v0.4)"
            );
        }
    }

    Ok(NormalisedRequest {
        history,
        sampling,
        stream: req.stream,
        include_usage,
        enable_thinking,
    })
}

fn content_to_string(content: &MessageContent, msg_idx: usize) -> Result<String, ApiError> {
    match content {
        MessageContent::Text(s) => Ok(s.clone()),
        MessageContent::Parts(parts) => {
            let mut acc = String::new();
            for (j, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Text { text } => acc.push_str(text),
                    ContentPart::ImageUrl { .. } => {
                        return Err(ApiError::invalid_with_param(
                            "image_url content parts are not supported in v0.4",
                            "unsupported_content_type",
                            format!("messages[{msg_idx}].content[{j}]"),
                        ));
                    }
                }
            }
            Ok(acc)
        }
    }
}

// =========================================================================
// Processing (runs inside spawn_blocking)
// =========================================================================

/// Common Sprint 5 prep: render the multi-turn prefill against the
/// session's chat template, reset the KV cache, validate the prompt
/// fits the context window, and build the `GenerateConfig`.
fn prepare_generate(
    session: &mut ServerSession,
    req: &NormalisedRequest,
    cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(Vec<u32>, GenerateConfig, u32), ApiError> {
    // Stateless per Decision §3: every call wipes the KV cache.
    session.chat.forward.kv_cache.reset();

    let max_tokens = req.sampling.max_tokens.unwrap_or(200).max(1);

    // Render the full history as a single prefill token sequence.
    let messages: Vec<RenderMessage<'_>> = req
        .history
        .iter()
        .map(|(role, content)| RenderMessage { role: *role, content: content.as_str() })
        .collect();
    let prefill_tokens = session.template.render_full_history(&session.tokenizer, &messages);

    // §7.5 — pre-flight context check. The decode loop itself also
    // gates on `max_seq_len`, but raising the structured error here
    // gives clients the OpenAI-spec'd 400 instead of a 500.
    let max_seq = session.chat.forward.kv_cache.config.max_seq_len;
    if (prefill_tokens.len() as u32).saturating_add(max_tokens) > max_seq {
        return Err(ApiError::ContextLengthExceeded {
            prompt_tokens: prefill_tokens.len() as u32,
            max_tokens,
            context_window: max_seq,
        });
    }

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
        // §4.5 — ThinkFilter ON by default; OFF when the request
        // explicitly sets `chat_template_kwargs.enable_thinking: false`.
        think_filter: req.enable_thinking,
        sampling: vf_sampling,
        cancel_token,
    };

    Ok((prefill_tokens, cfg_g, max_tokens))
}

fn process_request(
    session: &mut ServerSession,
    model_id: &str,
    req: NormalisedRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    let (prefill_tokens, cfg_g, _max_tokens) = prepare_generate(session, &req, None)?;

    // Set up an incremental ThinkFilter so the visible/raw split
    // matches what the streaming path produces. For non-streaming
    // we discard the incremental chunks and read the aggregated
    // `visible_text` from the result.
    let mut filter = if cfg_g.think_filter {
        Some(ThinkFilter::new())
    } else {
        None
    };
    let mut on_token = move |_id: u32, raw: &str| {
        if let Some(f) = filter.as_mut() {
            let _visible = f.push(raw);
        }
    };

    let ServerSession {
        dev,
        registry,
        cmd_ctx,
        model,
        gguf,
        cfg,
        tokenizer,
        chat,
        ..
    } = session;

    let result = generate_from_tokens(
        &mut chat.forward,
        dev, registry, cmd_ctx, model,
        EmbeddingSource::Gguf(gguf),
        cfg, tokenizer,
        &prefill_tokens, 0, &cfg_g, false, &mut on_token,
    )
    .map_err(|e| {
        // The decode-loop's pre-flight check produces a string error
        // for the context-overflow case; everything else is a real
        // 500. Both rare in practice given the pre-check above.
        let s = e.to_string();
        if s.contains("max_seq_len") {
            ApiError::ContextLengthExceeded {
                prompt_tokens: prefill_tokens.len() as u32,
                max_tokens: cfg_g.max_tokens,
                context_window: chat.forward.kv_cache.config.max_seq_len,
            }
        } else {
            ApiError::internal(format!("generation: {s}"), "gpu_error")
        }
    })?;

    let finish_reason = if result.stopped_on_eos {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };

    let id = new_chatcmpl_id();
    let choice = Choice {
        index: 0,
        message: AssistantMessage::new(result.visible_text),
        logprobs: None,
        finish_reason,
    };
    let usage = Usage::new(
        result.prompt_tokens as u32,
        result.generated_tokens as u32,
    );
    Ok(ChatCompletionResponse::new(id, model_id.to_string(), choice, usage))
}

fn new_chatcmpl_id() -> String {
    // `chatcmpl-` prefix matches OpenAI's wire convention. We use
    // the uuid v4 hex (no dashes) so the suffix is 32 chars.
    let u = uuid::Uuid::new_v4();
    let mut s = String::with_capacity(9 + 32);
    s.push_str("chatcmpl-");
    s.push_str(&u.simple().to_string());
    s
}

fn seed_from_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// =========================================================================
// Streaming path
// =========================================================================

/// Returns the axum response (an `Sse<...>::into_response()`) and
/// detaches a `spawn_blocking` task that drives the GPU forward and
/// pushes `StreamEvent`s into the mpsc channel that the SSE adapter
/// is polling. The blocking task takes ownership of the
/// concurrency permit so it lives exactly as long as the work.
fn start_streaming(
    state: Arc<AppState>,
    req: NormalisedRequest,
    permit: OwnedSemaphorePermit,
) -> Response {
    let chatcmpl_id = new_chatcmpl_id();
    let model_id = state.model_id.clone();
    let cancel = CancelToken::new();
    let include_usage = req.include_usage;

    // §4.2 — bounded channel. Capacity 64 gives a generous burst
    // buffer while still applying backpressure: once 64 tokens are
    // queued unread, `tx.blocking_send` parks the GPU thread until
    // the client catches up. Unbounded would risk OOM if a slow
    // client sat there long enough.
    let (tx, rx) = mpsc::channel::<StreamEvent>(64);

    let state_for_task = state.clone();
    let cancel_for_task = cancel.clone();
    let cancel_for_sse = cancel.clone();
    let model_id_for_task = model_id.clone();
    tokio::task::spawn_blocking(move || {
        // Permit held for the whole generation; dropped on return.
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
        run_streaming_generation(
            &mut session_guard,
            &model_id_for_task,
            req,
            tx,
            cancel_for_task,
            include_usage,
        );
    });

    let meta = ChunkMeta {
        id: chatcmpl_id,
        model: model_id,
    };
    // The SSE adapter takes ownership of a CancelToken clone; its
    // Drop impl flips the flag if the client disconnects mid-stream,
    // catching cases where mpsc-Sender's error pathway alone is
    // slow to fire (Sprint 4 cancel-latency fix).
    sse_response(rx, meta, cancel_for_sse).into_response()
}

/// Run the chat generation, pushing `StreamEvent`s into `tx`. Designed
/// to be called *inside* `spawn_blocking`: it uses blocking-mpsc
/// sends and `generate_from_tokens` directly (no ChatSession — that
/// helper is tuned for the CLI's KV-cache-carrying REPL, which is
/// the opposite of what stateless multi-turn needs).
fn run_streaming_generation(
    session: &mut ServerSession,
    _model_id: &str,
    req: NormalisedRequest,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancelToken,
    include_usage: bool,
) {
    let (prefill_tokens, cfg_g, _max_tokens) =
        match prepare_generate(session, &req, Some(cancel.as_arc())) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(e));
                return;
            }
        };
    let prefill_len = prefill_tokens.len() as u32;

    // Emit the header chunk first. If the client is already gone
    // (rare but possible), bail out before doing any GPU work.
    if tx.blocking_send(StreamEvent::Header).is_err() {
        cancel.cancel();
        return;
    }

    // Per-token ThinkFilter, mirroring the chat.rs pattern. The
    // `visible` chunks (post-filter) are what we stream as
    // `StreamEvent::Delta`; the raw chunks stay buffered inside the
    // filter so a partial `</think>` tag near the boundary is held
    // back until it can be resolved.
    let mut filter = if cfg_g.think_filter {
        Some(ThinkFilter::new())
    } else {
        None
    };

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
        let visible_owned: String = match filter.as_mut() {
            Some(f) => f.push(raw),
            None => raw.to_string(),
        };
        if visible_owned.is_empty() {
            return;
        }
        if tx_for_cb
            .blocking_send(StreamEvent::Delta(visible_owned))
            .is_err()
        {
            // Client disconnected mid-stream — flip the flag so the
            // decode loop exits within the next token.
            cancel_for_cb.cancel();
        }
    };

    let ServerSession {
        dev,
        registry,
        cmd_ctx,
        model,
        gguf,
        cfg,
        tokenizer,
        chat,
        ..
    } = session;

    let result = generate_from_tokens(
        &mut chat.forward,
        dev, registry, cmd_ctx, model,
        EmbeddingSource::Gguf(gguf),
        cfg, tokenizer,
        &prefill_tokens, 0, &cfg_g, false, &mut on_token,
    );

    match result {
        Ok(g) => {
            let finish = if g.stopped_on_eos {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            let usage = Usage::new(g.prompt_tokens as u32, g.generated_tokens as u32);
            // Best-effort: if the client already disconnected the
            // send returns Err; we drop the event and exit.
            let _ = tx.blocking_send(StreamEvent::Final {
                finish,
                usage,
                include_usage,
            });
        }
        Err(e) => {
            let s = e.to_string();
            let api_err = if s.contains("max_seq_len") {
                ApiError::ContextLengthExceeded {
                    prompt_tokens: prefill_len,
                    max_tokens: cfg_g.max_tokens,
                    context_window: chat.forward.kv_cache.config.max_seq_len,
                }
            } else {
                ApiError::internal(format!("generation: {s}"), "gpu_error")
            };
            let _ = tx.blocking_send(StreamEvent::Error(api_err));
        }
    }
    // Dropping tx here lets the SSE adapter's stream complete; it
    // then appends the synthetic `data: [DONE]` marker.
}

// =========================================================================
// Tests (no GPU — only validation logic)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req_from(j: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(j).unwrap()
    }

    #[test]
    fn single_user_message_validates() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history.len(), 1);
        assert_eq!(n.history[0].0, HistoryRole::User);
        assert_eq!(n.history[0].1, "Hi");
    }

    #[test]
    fn system_plus_user_validates() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history.len(), 2);
        assert_eq!(n.history[0].0, HistoryRole::System);
        assert_eq!(n.history[0].1, "Be brief.");
        assert_eq!(n.history[1].0, HistoryRole::User);
    }

    #[test]
    fn developer_role_is_treated_as_system() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history[0].0, HistoryRole::System);
        assert_eq!(n.history[0].1, "Be concise.");
    }

    #[test]
    fn empty_messages_rejected() {
        let req = req_from(json!({"model": "x", "messages": []}));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "no_user_message");
    }

    #[test]
    fn multi_turn_history_accepted() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "I am Tom."},
                {"role": "assistant", "content": "Hi Tom!"},
                {"role": "user", "content": "Who am I?"},
            ],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history.len(), 4);
        let roles: Vec<HistoryRole> = n.history.iter().map(|(r, _)| *r).collect();
        assert_eq!(
            roles,
            vec![
                HistoryRole::System,
                HistoryRole::User,
                HistoryRole::Assistant,
                HistoryRole::User,
            ]
        );
    }

    #[test]
    fn multi_turn_without_system_accepted() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
            ],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history.len(), 5);
        assert!(n.history.iter().all(|(r, _)| !matches!(r, HistoryRole::System)));
    }

    #[test]
    fn last_message_must_be_user() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "last_message_not_user");
    }

    #[test]
    fn only_assistant_rejected_as_no_user() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "assistant", "content": "stale"}],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "no_user_message");
    }

    #[test]
    fn only_system_rejected_as_no_user() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "system", "content": "Be brief."}],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "no_user_message");
    }

    #[test]
    fn tool_role_rejected() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "tool", "content": "result"},
                {"role": "user", "content": "ok"},
            ],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "unsupported_role");
    }

    #[test]
    fn image_url_part_rejected() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look:"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                ],
            }],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "unsupported_content_type");
    }

    #[test]
    fn n_greater_than_one_rejected() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 3,
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "unsupported_n");
    }

    #[test]
    fn streaming_is_accepted_in_sprint_3() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert!(n.stream);
        assert!(n.include_usage);
    }

    #[test]
    fn non_streaming_default_is_false() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert!(!n.stream);
        assert!(!n.include_usage);
    }

    #[test]
    fn enable_thinking_defaults_to_true() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert!(n.enable_thinking, "ThinkFilter should default ON");
    }

    #[test]
    fn enable_thinking_false_propagates() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {"enable_thinking": false},
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert!(!n.enable_thinking);
    }

    #[test]
    fn empty_chat_template_kwargs_keeps_default_thinking_on() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {},
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert!(n.enable_thinking);
    }

    #[test]
    fn duplicate_system_rejected() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "system", "content": "a"},
                {"role": "system", "content": "b"},
                {"role": "user", "content": "Hi"},
            ],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "duplicate_system_message");
    }

    #[test]
    fn system_after_user_rejected() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Be concise."},
            ],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "invalid_message_order");
    }

    #[test]
    fn text_content_parts_concat() {
        let req = req_from(json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }],
        }));
        let n = validate_and_normalise(&req).unwrap();
        assert_eq!(n.history.last().unwrap().1, "Hello world");
    }

    #[test]
    fn chatcmpl_id_has_prefix_and_32_hex_suffix() {
        let id = new_chatcmpl_id();
        assert!(id.starts_with("chatcmpl-"));
        let hex = &id[9..];
        assert_eq!(hex.len(), 32);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
