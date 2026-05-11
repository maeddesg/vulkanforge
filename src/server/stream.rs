//! SSE stream adapter for `POST /v1/chat/completions` with
//! `stream: true`.
//!
//! Architecture §4.2 (channel topology), §4.3 (SSE format).
//!
//! Layout:
//! - The chat handler creates an [`mpsc::Receiver<StreamEvent>`] and
//!   hands it to [`sse_response`], which converts each event into
//!   an [`axum::response::sse::Event`] carrying a single JSON chunk.
//! - When the channel closes (GPU thread done, or cancelled), the
//!   stream appends a synthetic `data: [DONE]\n\n` marker per
//!   OpenAI spec.
//! - Keep-alive comments fire every 15 s so intermediate proxies
//!   don't drop a long-prefilling connection.

use std::convert::Infallible;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::error::ApiError;
use crate::server::types::response::{FinishReason, Usage};
use crate::server::types::streaming::ChatCompletionChunk;

/// One unit of SSE output the GPU-side spawn_blocking task sends to
/// the SSE adapter. The adapter is responsible for shaping each into
/// the correct JSON chunk and for appending the trailing `[DONE]`.
#[derive(Debug)]
pub enum StreamEvent {
    /// First chunk — `delta.role = "assistant"`.
    Header,
    /// Mid-stream token text (post-think-filter).
    Delta(String),
    /// Final chunk — empty delta, populated `finish_reason`.
    Final {
        finish: FinishReason,
        /// Always present so the optional usage-chunk can pull
        /// from the same `Final` event.
        usage: Usage,
        /// When `true`, emit an additional `usage_only` chunk
        /// immediately after the final chunk (Merge-Sprint #5,
        /// `stream_options.include_usage`).
        include_usage: bool,
    },
    /// Mid-stream error (GPU crash, tokenizer failure, etc.).
    /// Per §8.3 we surface this as a JSON event with `event: error`.
    Error(ApiError),
}

/// Metadata that is constant across every chunk of one streamed
/// response. Filled in by the chat handler once before spawning
/// the blocking task.
#[derive(Clone, Debug)]
pub struct ChunkMeta {
    pub id: String,
    pub model: String,
}

/// Build the axum [`Sse`] response from a receiver of [`StreamEvent`]s.
///
/// The returned stream:
/// 1. Yields a JSON chunk per [`StreamEvent`] (`Header` → header,
///    `Delta` → delta, `Final` → final + optional usage-only).
/// 2. Appends a synthetic `data: [DONE]` marker when the channel
///    closes.
///
/// Keep-alive is set to 15 s of idle (per architecture §4.3).
pub fn sse_response(
    rx: mpsc::Receiver<StreamEvent>,
    meta: ChunkMeta,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx)
        .flat_map(move |ev| futures_util::stream::iter(events_for(&meta, ev)))
        .chain(futures_util::stream::iter(std::iter::once(Ok::<_, Infallible>(
            Event::default().data("[DONE]"),
        ))));

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// Translate one [`StreamEvent`] into one or more SSE events.
/// `Final` events with `include_usage = true` expand into two events
/// (final-chunk + usage-only-chunk).
fn events_for(meta: &ChunkMeta, ev: StreamEvent) -> Vec<Result<Event, Infallible>> {
    match ev {
        StreamEvent::Header => {
            let chunk = ChatCompletionChunk::header(meta.id.clone(), meta.model.clone());
            vec![Ok(json_event(&chunk))]
        }
        StreamEvent::Delta(text) => {
            let chunk = ChatCompletionChunk::delta(meta.id.clone(), meta.model.clone(), text);
            vec![Ok(json_event(&chunk))]
        }
        StreamEvent::Final { finish, usage, include_usage } => {
            let mut events = Vec::with_capacity(2);
            let final_chunk = ChatCompletionChunk::final_chunk(meta.id.clone(), meta.model.clone(), finish);
            events.push(Ok(json_event(&final_chunk)));
            if include_usage {
                let usage_chunk =
                    ChatCompletionChunk::usage_only(meta.id.clone(), meta.model.clone(), usage);
                events.push(Ok(json_event(&usage_chunk)));
            }
            events
        }
        StreamEvent::Error(e) => {
            // SSE convention: an event named `error` with the
            // OpenAI error-body as the JSON data. Some clients
            // ignore the event-type and parse data as a chunk;
            // they'll see a JSON they don't recognise and stop,
            // which is fine — the connection is going away.
            let body = e.to_response_body();
            let json = serde_json::to_string(&body).unwrap_or_else(|_| String::from("{}"));
            vec![Ok(Event::default().event("error").data(json))]
        }
    }
}

fn json_event(chunk: &ChatCompletionChunk) -> Event {
    let s = serde_json::to_string(chunk).unwrap_or_else(|_| String::from("{}"));
    Event::default().data(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMeta {
        ChunkMeta {
            id: "chatcmpl-test".into(),
            model: "qwen3-8b".into(),
        }
    }

    // The axum `Event` type intentionally hides its rendered form,
    // so instead of round-tripping through SSE we test the inputs
    // to the SSE adapter directly. The Sprint 1 streaming-chunk
    // tests already cover the JSON shape produced by
    // `ChatCompletionChunk::{header, delta, final_chunk, usage_only}`.
    // Here we verify the *event-count* invariants enforced by
    // `events_for` (one event per non-Final variant, two for Final
    // with include_usage = true, one for Final without).

    #[test]
    fn header_yields_one_event() {
        let evs = events_for(&meta(), StreamEvent::Header);
        assert_eq!(evs.len(), 1);
        assert!(evs[0].is_ok());
    }

    #[test]
    fn delta_yields_one_event() {
        let evs = events_for(&meta(), StreamEvent::Delta("Paris".into()));
        assert_eq!(evs.len(), 1);
        assert!(evs[0].is_ok());
    }

    #[test]
    fn final_without_usage_yields_one_event() {
        let evs = events_for(
            &meta(),
            StreamEvent::Final {
                finish: FinishReason::Stop,
                usage: Usage::new(10, 5),
                include_usage: false,
            },
        );
        assert_eq!(evs.len(), 1);
    }

    #[test]
    fn final_with_include_usage_yields_two_events() {
        let evs = events_for(
            &meta(),
            StreamEvent::Final {
                finish: FinishReason::Length,
                usage: Usage::new(42, 17),
                include_usage: true,
            },
        );
        assert_eq!(evs.len(), 2);
    }

    #[test]
    fn error_yields_one_event() {
        let evs = events_for(
            &meta(),
            StreamEvent::Error(ApiError::internal("boom", "gpu_error")),
        );
        assert_eq!(evs.len(), 1);
    }
}
