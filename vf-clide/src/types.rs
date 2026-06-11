// SPDX-License-Identifier: GPL-3.0-only
//! Minimal OpenAI chat-completions wire types (client side).
//!
//! Only the fields vf-clide actually sends/reads are modelled; the server
//! tolerates extra fields and we ignore the ones we don't need.

use serde::{Deserialize, Serialize};

/// One chat message (`role` + `content`). Phase 1 uses string content
/// only (no content-part arrays, no tool roles).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub stream: bool,
}

// ---- Non-streaming response (only the fields we read) ----

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: RespMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RespMessage {
    #[serde(default)]
    pub content: Option<String>,
}

// ---- Streaming chunk (`stream: true`) ----

#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
}

/// Strip `<think>…</think>` reasoning blocks (Qwen3 / DeepSeek-R1 /
/// Gemma-4) from a string. Used when re-sending prior assistant turns
/// into the history so old chain-of-thought doesn't bloat the context.
/// Tolerates an unclosed `<think>` (drops the remainder) and multiple
/// blocks. If the server already strips reasoning (its ThinkFilter is on
/// by default), this is a harmless no-op.
pub fn strip_think(s: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len()..];
        match after.find(CLOSE) {
            Some(close) => rest = &after[close + CLOSE.len()..],
            None => {
                rest = ""; // unclosed block → drop the remainder
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_minimal_fields() {
        let req = ChatRequest {
            model: "Qwen3-14B-Q4_K_M".into(),
            messages: vec![ChatMessage::user("Hi")],
            temperature: Some(0.0),
            max_tokens: None,
            stream: true,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "Qwen3-14B-Q4_K_M");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "Hi");
        assert_eq!(v["stream"], true);
        assert_eq!(v["temperature"], 0.0);
        // max_tokens omitted when None.
        assert!(v.get("max_tokens").is_none());
    }

    #[test]
    fn non_streaming_response_parses() {
        let json = r#"{"id":"x","object":"chat.completion","choices":[
            {"index":0,"message":{"role":"assistant","content":"Paris."},"finish_reason":"stop"}
        ],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let r: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.choices[0].message.content.as_deref(), Some("Paris."));
        assert_eq!(r.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn streaming_chunk_parses_delta() {
        let json = r#"{"id":"x","object":"chat.completion.chunk","choices":[
            {"index":0,"delta":{"content":"Par"},"finish_reason":null}
        ]}"#;
        let c: ChatChunk = serde_json::from_str(json).unwrap();
        assert_eq!(c.choices[0].delta.content.as_deref(), Some("Par"));
    }

    #[test]
    fn streaming_final_chunk_has_empty_delta() {
        let json = r#"{"id":"x","object":"chat.completion.chunk","choices":[
            {"index":0,"delta":{},"finish_reason":"stop"}
        ]}"#;
        let c: ChatChunk = serde_json::from_str(json).unwrap();
        assert!(c.choices[0].delta.content.is_none());
        assert_eq!(c.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn strip_think_removes_block() {
        assert_eq!(strip_think("<think>reasoning</think>Paris."), "Paris.");
        assert_eq!(strip_think("a<think>x</think>b<think>y</think>c"), "abc");
        assert_eq!(strip_think("no tags here"), "no tags here");
    }

    #[test]
    fn strip_think_tolerates_unclosed() {
        assert_eq!(strip_think("answer<think>unfinished reasoning"), "answer");
    }
}
