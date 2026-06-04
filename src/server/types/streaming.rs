//! Streaming response types for `POST /v1/chat/completions` with
//! `stream: true`.
//!
//! Mirrors §3.3 of the architecture document. The streaming sequence
//! (per §4.3) is:
//!
//! 1. Header chunk — `delta.role = "assistant"`, no content.
//! 2. N delta chunks — `delta.content = Some(...)`.
//! 3. Final chunk — `delta = {}`, `finish_reason = Some(...)`.
//! 4. (optional) Usage chunk — `choices = []`, `usage = Some(...)`.
//!    Only when `stream_options.include_usage: true`. Merge-Sprint #5.
//! 5. SSE marker `data: [DONE]\n\n` (emitted outside this module,
//!    by the SSE adapter in Sprint 3).

use serde::Serialize;

use super::response::{FinishReason, Usage};

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// `chatcmpl-<id>` — identical across every chunk of a single
    /// streamed response.
    pub id: String,
    pub object: &'static str, // always "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    /// Usage is `None` for header/delta/final chunks. Only the
    /// optional trailing usage chunk (when `stream_options.include_usage`
    /// is set) populates this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    /// Header chunk (chunk #1): announces the role, no content yet.
    pub fn header(id: String, model: String) -> Self {
        Self::base(id, model, vec![ChunkChoice::header()], None)
    }

    /// Delta chunk: incremental text since the previous chunk.
    pub fn delta(id: String, model: String, text: String) -> Self {
        Self::base(id, model, vec![ChunkChoice::delta(text)], None)
    }

    /// Final chunk: empty delta + finish_reason.
    pub fn final_chunk(id: String, model: String, finish: FinishReason) -> Self {
        Self::base(id, model, vec![ChunkChoice::final_chunk(finish)], None)
    }

    /// Tool-calls delta chunk (v1: the whole block emitted at once, no
    /// char-incremental argument streaming).
    pub fn tool_calls(id: String, model: String, calls: Vec<DeltaToolCall>) -> Self {
        Self::base(id, model, vec![ChunkChoice::tool_calls(calls)], None)
    }

    /// Usage-only chunk: `choices = []`, `usage` populated. Emitted
    /// after the final chunk when `stream_options.include_usage: true`.
    pub fn usage_only(id: String, model: String, usage: Usage) -> Self {
        Self::base(id, model, Vec::new(), Some(usage))
    }

    fn base(id: String, model: String, choices: Vec<ChunkChoice>, usage: Option<Usage>) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created: unix_now_secs(),
            model,
            system_fingerprint: Some(super::SYSTEM_FINGERPRINT.into()),
            choices,
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl ChunkChoice {
    fn header() -> Self {
        Self {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }
    }

    fn delta(text: String) -> Self {
        Self {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(text),
                tool_calls: None,
            },
            finish_reason: None,
        }
    }

    fn final_chunk(finish: FinishReason) -> Self {
        Self {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(finish),
        }
    }

    fn tool_calls(calls: Vec<DeltaToolCall>) -> Self {
        Self {
            index: 0,
            delta: Delta { role: None, content: None, tool_calls: Some(calls) },
            finish_reason: None,
        }
    }
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    /// Only populated in the header chunk; `Some("assistant")`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Populated in mid-stream delta chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Populated in the tool-calls delta (v1: complete block at once).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// One tool call in a streaming `delta.tool_calls` (v1: `index`, full
/// `id`/`name`/`arguments` in a single delta — no char-incremental args).
#[derive(Debug, Serialize, Clone)]
pub struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub function: DeltaToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
pub struct DeltaToolCallFunction {
    pub name: String,
    pub arguments: String,
}

// =========================================================================
// Legacy text-completion streaming chunk (POST /v1/completions, stream)
// =========================================================================

/// Streaming chunk for `/v1/completions`. Mirrors `ChatCompletionChunk`
/// but with the `text_completion` object and a flat `choices[].text`
/// field (no `delta`/`role`). `cmpl-` id.
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str, // always "text_completion"
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl CompletionChunk {
    /// Mid-stream text chunk.
    pub fn delta(id: String, model: String, text: String) -> Self {
        Self::base(
            id, model,
            vec![CompletionChunkChoice { index: 0, text, finish_reason: None }],
            None,
        )
    }

    /// Final chunk: empty text + finish_reason.
    pub fn final_chunk(id: String, model: String, finish: FinishReason) -> Self {
        Self::base(
            id, model,
            vec![CompletionChunkChoice { index: 0, text: String::new(), finish_reason: Some(finish) }],
            None,
        )
    }

    /// Usage-only trailing chunk (`stream_options.include_usage`).
    pub fn usage_only(id: String, model: String, usage: Usage) -> Self {
        Self::base(id, model, Vec::new(), Some(usage))
    }

    fn base(id: String, model: String, choices: Vec<CompletionChunkChoice>, usage: Option<Usage>) -> Self {
        Self {
            id,
            object: "text_completion",
            created: unix_now_secs(),
            model,
            system_fingerprint: Some(super::SYSTEM_FINGERPRINT.into()),
            choices,
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
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

    fn make_id() -> String {
        "chatcmpl-test".into()
    }
    fn make_model() -> String {
        "qwen3-8b".into()
    }

    #[test]
    fn header_chunk_has_role_only_no_content_no_finish() {
        let c = ChatCompletionChunk::header(make_id(), make_model());
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert!(v["choices"][0]["delta"].get("content").is_none());
        assert!(v["choices"][0].get("finish_reason").is_none());
        assert!(v.get("usage").is_none());
    }

    #[test]
    fn delta_chunk_has_content_no_role_no_finish() {
        let c = ChatCompletionChunk::delta(make_id(), make_model(), "Paris".into());
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "Paris");
        assert!(v["choices"][0]["delta"].get("role").is_none());
        assert!(v["choices"][0].get("finish_reason").is_none());
    }

    #[test]
    fn final_chunk_has_empty_delta_and_finish() {
        let c = ChatCompletionChunk::final_chunk(make_id(), make_model(), FinishReason::Length);
        let v = serde_json::to_value(&c).unwrap();
        // delta should serialize as empty object {}
        assert_eq!(v["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn usage_only_chunk_has_empty_choices_and_usage() {
        let c = ChatCompletionChunk::usage_only(make_id(), make_model(), Usage::new(42, 17));
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["prompt_tokens"], 42);
        assert_eq!(v["usage"]["completion_tokens"], 17);
        assert_eq!(v["usage"]["total_tokens"], 59);
    }

    #[test]
    fn delta_empty_serializes_as_empty_object() {
        let d = Delta::default();
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v, serde_json::json!({}));
    }

    #[test]
    fn id_and_object_are_stable_across_chunks() {
        let id = "chatcmpl-XYZ".to_string();
        let header = ChatCompletionChunk::header(id.clone(), make_model());
        let delta  = ChatCompletionChunk::delta(id.clone(), make_model(), "t".into());
        let fin    = ChatCompletionChunk::final_chunk(id.clone(), make_model(), FinishReason::Stop);
        for c in &[&header, &delta, &fin] {
            assert_eq!(c.id, id);
            assert_eq!(c.object, "chat.completion.chunk");
        }
    }
}
