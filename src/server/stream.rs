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
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::cancel::CancelToken;
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
/// 3. Triggers [`CancelToken::cancel`] when the stream is dropped,
///    which guarantees the GPU thread sees the signal even if the
///    `mpsc::Sender::blocking_send` error pathway is slow to fire
///    (Sprint 4 cancel-latency fix — see [`CancelOnDrop`]).
///
/// Keep-alive is set to 1 s of idle so axum forces a write into the
/// TCP socket every second; on a closed socket that write errors,
/// which causes hyper to drop the response body and (transitively)
/// the inner stream — triggering the drop guard above.
pub fn sse_response(
    rx: mpsc::Receiver<StreamEvent>,
    meta: ChunkMeta,
    cancel: CancelToken,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let inner = ReceiverStream::new(rx)
        .flat_map(move |ev| futures_util::stream::iter(events_for(&meta, ev)))
        .chain(futures_util::stream::iter(std::iter::once(Ok::<_, Infallible>(
            Event::default().data("[DONE]"),
        ))));

    // Drop-guard the entire chain. axum's `Sse` retains the wrapped
    // stream until either it polls to completion (server finished
    // generation) OR hyper drops the response body (client closed
    // the TCP socket). In the second case the `Drop` impl on
    // `CancelOnDrop` flips the cancel token, the GPU thread's
    // decode loop sees the flag on its next iteration, and aborts
    // cleanly — bounding wasted compute at ~1 token plus the
    // 1 s keep-alive interval needed to surface the closed-socket
    // write error.
    let guarded = CancelOnDrop { stream: inner, cancel };

    Sse::new(guarded).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(1))
            .text("keep-alive"),
    )
}

/// Stream adapter that flips a [`CancelToken`] when dropped.
///
/// Used to translate "client TCP-close → axum drops the response
/// body" into "GPU thread sees cancel flag" without relying on
/// the mpsc-Sender error-pathway alone (which Sprint 3 found to
/// not propagate fast enough on its own).
struct CancelOnDrop<S> {
    stream: S,
    cancel: CancelToken,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        // Idempotent: cancel() is a single atomic store, safe to
        // call even on the normal completion path (the GPU has
        // already exited the decode loop by then, so the flag has
        // no effect).
        self.cancel.cancel();
    }
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
