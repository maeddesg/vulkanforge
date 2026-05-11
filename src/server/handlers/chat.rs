//! `POST /v1/chat/completions` (and `/chat/completions` alias).
//!
//! Architecture §2.1 (endpoint), §6 (chat-template integration),
//! §7 (sampling mapping), §8 (error mapping).
//!
//! Sprint 3 scope:
//! - **Non-streaming** (`stream: false`): blocking task → JSON body.
//! - **Streaming** (`stream: true`): mpsc-bridged SSE response
//!   with header + delta chunks + final + optional usage-only +
//!   `[DONE]` marker. Client TCP-disconnect → `tx.blocking_send`
//!   returns `Err` → `CancelToken` flips → decode loop exits on
//!   the next iteration (Sprint 3 §4.4).
//! - **Single-turn requests only** — `messages = [system?, user]`.
//!   Multi-turn (assistant messages in history) is rejected with
//!   400 `unsupported_multi_turn` because the full-history renderer
//!   lives in Sprint 6 (architecture §6.1).
//!
//! Validation order (§8.4) — model field is NOT validated
//! (Decision §2 / Merge-Sprint #1):
//! 1. n ≤ 1
//! 2. messages non-empty
//! 3. each role ≠ tool, each content ≠ image_url
//! 4. exactly one user message, last message must be that user
//!    (system before it is optional)
//! 5. Acquire permit → else 429
//! 6. spawn_blocking → reset session → render → send_streaming
//! 7. Either: map TurnResult → ChatCompletionResponse JSON,
//!    or: stream Header/Delta/Final/Usage events through SSE adapter
//!
//! Context-overflow propagation: `ChatSession::send_streaming`
//! returns `ChatError::ContextOverflow` which we map to a 400 with
//! `code = "context_length_exceeded"` (Decision §2).

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;

use crate::backend::vulkan::chat::ChatError;
use crate::backend::vulkan::decode::{GenerateConfig, Sampling};
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
    system: String,
    user: String,
    sampling: SamplingParams,
    /// `true` when the request asked for `stream: true`. Decides
    /// between SSE adapter (Sprint 3) and a single JSON body.
    stream: bool,
    /// Populated from `stream_options.include_usage`. Only honoured
    /// when `stream` is true.
    include_usage: bool,
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

    // §8.1 unsupported_role / unsupported_content_type — and check
    // shape (single-turn only in Sprint 2).
    let mut system: Option<String> = None;
    let mut user: Option<String> = None;
    for (i, msg) in req.messages.iter().enumerate() {
        match msg.role {
            Role::Tool => {
                return Err(ApiError::invalid_with_param(
                    "tool role not supported in v0.4",
                    "unsupported_role",
                    format!("messages[{i}].role"),
                ));
            }
            Role::Assistant => {
                // v0.4 Sprint 6 will render multi-turn history.
                return Err(ApiError::invalid_with_param(
                    "multi-turn history (assistant messages) is planned for v0.4 Sprint 6; \
                     send only [system?, user] in Sprint 2",
                    "unsupported_multi_turn",
                    format!("messages[{i}].role"),
                ));
            }
            Role::System => {
                if system.is_some() {
                    return Err(ApiError::invalid_with_param(
                        "more than one system message",
                        "duplicate_system_message",
                        format!("messages[{i}].role"),
                    ));
                }
                if user.is_some() {
                    return Err(ApiError::invalid_with_param(
                        "system message must precede user message",
                        "invalid_message_order",
                        format!("messages[{i}].role"),
                    ));
                }
                system = Some(content_to_string(&msg.content, i)?);
            }
            Role::User => {
                if user.is_some() {
                    return Err(ApiError::invalid_with_param(
                        "multiple user messages not supported in v0.4 Sprint 2 \
                         (multi-turn history is Sprint 6)",
                        "unsupported_multi_turn",
                        format!("messages[{i}].role"),
                    ));
                }
                user = Some(content_to_string(&msg.content, i)?);
            }
        }
    }
    let Some(user) = user else {
        return Err(ApiError::invalid_with_param(
            "no user message present",
            "no_user_message",
            "messages",
        ));
    };

    let sampling = SamplingParams::from_request(req);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    Ok(NormalisedRequest {
        system: system.unwrap_or_default(),
        user,
        sampling,
        stream: req.stream,
        include_usage,
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

fn process_request(
    session: &mut crate::server::state::ServerSession,
    model_id: &str,
    req: NormalisedRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    // Stateless per Decision §3: every call wipes the KV cache.
    session.chat.reset();
    session.chat.system_prompt = req.system;

    let max_tokens = req.sampling.max_tokens.unwrap_or(200).max(1);

    // VF's existing Sampling struct. Greedy if temperature == 0.
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
        // Apply ThinkFilter by default for templates that produce
        // <think> blocks. The existing CLI defaults to filter-on
        // unless `--no-think-filter`; we follow suit. A future
        // request-level override (`enable_thinking` extension) can
        // wire through this flag.
        think_filter: true,
        sampling: vf_sampling,
        cancel_token: None,
    };

    // Re-bind to a separate local to keep the borrow checker happy
    // (ChatSession::send_streaming takes &mut self plus several
    // &-borrows of session fields, but they're all distinct fields).
    let user_msg = req.user;
    let crate::server::state::ServerSession {
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

    // No-op visible callback for non-streaming; the entire output is
    // returned via TurnResult.visible_text at the end.
    let on_visible = |_text: &str| {};

    let turn = chat
        .send_streaming(
            dev, registry, cmd_ctx, model, gguf, cfg, tokenizer, &user_msg, &cfg_g, on_visible,
        )
        .map_err(map_chat_error)?;

    // §2.1 finish_reason mapping:
    //   stopped_on_eos       → "stop"
    //   else (max_tokens hit) → "length"
    let finish_reason = if turn.stopped_on_eos {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };

    let id = new_chatcmpl_id();
    let choice = Choice {
        index: 0,
        message: AssistantMessage::new(turn.visible_text),
        logprobs: None,
        finish_reason,
    };
    let usage = Usage::new(turn.prompt_tokens, turn.generated_tokens);
    Ok(ChatCompletionResponse::new(id, model_id.to_string(), choice, usage))
}

fn map_chat_error(e: ChatError) -> ApiError {
    match e {
        ChatError::ContextOverflow {
            current_pos,
            needed,
            max_seq_len,
        } => ApiError::ContextLengthExceeded {
            prompt_tokens: current_pos.saturating_add(needed).saturating_sub(max_seq_len),
            max_tokens: needed,
            context_window: max_seq_len,
        },
        ChatError::Generation(err) => ApiError::internal(format!("generation: {err}"), "gpu_error"),
    }
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
    sse_response(rx, meta).into_response()
}

/// Run the chat generation, pushing `StreamEvent`s into `tx`. Designed
/// to be called *inside* `spawn_blocking`: it uses blocking-mpsc
/// sends and the synchronous chat-session API.
fn run_streaming_generation(
    session: &mut ServerSession,
    model_id: &str,
    req: NormalisedRequest,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancelToken,
    include_usage: bool,
) {
    let _ = model_id; // model id lives on the chunk meta; not needed
                       // inside per-chunk construction here.

    // Stateless per Decision §3.
    session.chat.reset();
    session.chat.system_prompt = req.system;

    let max_tokens = req.sampling.max_tokens.unwrap_or(200).max(1);
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
        think_filter: true,
        sampling: vf_sampling,
        cancel_token: Some(cancel.as_arc()),
    };

    // Emit the header chunk first. If the client is already gone
    // (rare but possible), bail out before doing any GPU work.
    if tx.blocking_send(StreamEvent::Header).is_err() {
        cancel.cancel();
        return;
    }

    let user_msg = req.user;
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

    // The on_visible callback runs once per visible token chunk
    // (post-ThinkFilter). It both sends the delta to the SSE channel
    // and watches for client-disconnect.
    let tx_for_cb = tx.clone();
    let cancel_for_cb = cancel.clone();
    let on_visible = move |text: &str| {
        if text.is_empty() {
            return;
        }
        if tx_for_cb
            .blocking_send(StreamEvent::Delta(text.to_string()))
            .is_err()
        {
            // Client disconnected — flip the cancel flag so the
            // decode loop exits on its next iteration.
            cancel_for_cb.cancel();
        }
    };

    let result =
        chat.send_streaming(dev, registry, cmd_ctx, model, gguf, cfg, tokenizer, &user_msg, &cfg_g, on_visible);

    match result {
        Ok(turn) => {
            let finish = if turn.stopped_on_eos {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            let usage = Usage::new(turn.prompt_tokens, turn.generated_tokens);
            // Best-effort: if the client already disconnected the
            // send returns Err; we drop the event and exit.
            let _ = tx.blocking_send(StreamEvent::Final {
                finish,
                usage,
                include_usage,
            });
        }
        Err(e) => {
            let _ = tx.blocking_send(StreamEvent::Error(map_chat_error(e)));
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
        assert_eq!(n.user, "Hi");
        assert!(n.system.is_empty());
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
        assert_eq!(n.system, "Be brief.");
        assert_eq!(n.user, "Hi");
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
        assert_eq!(n.system, "Be concise.");
    }

    #[test]
    fn empty_messages_rejected() {
        let req = req_from(json!({"model": "x", "messages": []}));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "no_user_message");
    }

    #[test]
    fn assistant_message_rejected_with_multi_turn_code() {
        let req = req_from(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        }));
        let e = validate_and_normalise(&req).unwrap_err();
        assert_eq!(e.code(), "unsupported_multi_turn");
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
        assert_eq!(n.user, "Hello world");
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
