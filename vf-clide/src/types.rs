// SPDX-License-Identifier: GPL-3.0-only
//! Minimal OpenAI chat-completions wire types (client side).
//!
//! Only the fields vf-clide actually sends/reads are modelled; the server
//! tolerates extra fields and we ignore the ones we don't need.

use serde::{Deserialize, Serialize};

/// One chat message (`role` + `content`). The tool fields are
/// Phase-2 (agent loop) additions: `tool_call_id` on `role:"tool"`
/// results, `tool_calls` on a replayed `role:"assistant"` turn. Both
/// are omitted from the wire when `None`, so the plain chat path is
/// unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), tool_call_id: None, tool_calls: None }
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
    /// A `role:"tool"` result message answering a specific call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
    /// A replayed assistant turn that made `tool_calls` (content is
    /// whatever text preceded the calls — often empty for Qwen3).
    pub fn assistant_with_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }
}

/// One tool call (OpenAI shape) — both directions: parsed out of a
/// response, and replayed into the next request's assistant turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments as a JSON **string** (OpenAI convention).
    #[serde(default)]
    pub arguments: String,
}

fn function_kind() -> String {
    "function".into()
}

/// A tool's risk class — the single source of truth for permission
/// gating, declared **at the tool definition** (in each `*_tool()` fn),
/// never in a separate map that could drift from the dispatch.
///
/// Ordered tiers (`ReadOnly` < `Mutating` < `Exec`); each headless
/// opt-in flag raises an auto-approve *ceiling* that implies the lower
/// tiers (see `agent::Gate`):
/// - `ReadOnly` — `read_file`, `search`. Auto-approved by `--yes`.
/// - `Mutating` — `write_file` (writes, but **confined** to the workspace).
///   Auto-approved by `--allow-mutating`.
/// - `Exec` — `shell`. The **only** non-confinable tool, so the gate is
///   its *sole* guard → its own top tier, auto-approved only by the
///   loudly-named `--allow-shell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRisk {
    ReadOnly,
    Mutating,
    Exec,
}

impl ToolRisk {
    /// Numeric tier for ceiling comparison (higher = riskier).
    pub fn rank(self) -> u8 {
        match self {
            ToolRisk::ReadOnly => 1,
            ToolRisk::Mutating => 2,
            ToolRisk::Exec => 3,
        }
    }
}

/// A tool definition sent in the request `tools` array. `risk` is local
/// metadata (the permission class) and is **not** part of the OpenAI wire
/// format, so it is skipped during serialization.
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: &'static str, // always "function"
    pub function: ToolDef,
    #[serde(skip)]
    pub risk: ToolRisk,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl Tool {
    /// Define a function tool with its risk class (the gating source of truth).
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        risk: ToolRisk,
    ) -> Self {
        Self {
            kind: "function",
            function: ToolDef { name: name.into(), description: description.into(), parameters },
            risk,
        }
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
    /// Tool definitions (agent loop). Omitted from the wire for the
    /// plain chat path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
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
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
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
            tools: None,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "Qwen3-14B-Q4_K_M");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "Hi");
        assert_eq!(v["stream"], true);
        assert_eq!(v["temperature"], 0.0);
        // max_tokens + tools omitted when None.
        assert!(v.get("max_tokens").is_none());
        assert!(v.get("tools").is_none());
        // a plain user message has no tool fields on the wire.
        assert!(v["messages"][0].get("tool_call_id").is_none());
        assert!(v["messages"][0].get("tool_calls").is_none());
    }

    #[test]
    fn tool_call_parses_from_response_message() {
        let json = r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}]},
            "finish_reason":"tool_calls"}]}"#;
        let r: ChatResponse = serde_json::from_str(json).unwrap();
        let m = &r.choices[0].message;
        assert!(m.content.is_none());
        let calls = m.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
        assert_eq!(r.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn tool_and_assistant_messages_serialize_with_tool_fields() {
        let call = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "read_file".into(), arguments: r#"{"path":"/tmp/x"}"#.into() },
        };
        let assistant = ChatMessage::assistant_with_tool_calls("", vec![call.clone()]);
        let av = serde_json::to_value(&assistant).unwrap();
        assert_eq!(av["role"], "assistant");
        assert_eq!(av["tool_calls"][0]["function"]["name"], "read_file");
        assert!(av.get("tool_call_id").is_none());

        let result = ChatMessage::tool("call_1", "file body");
        let rv = serde_json::to_value(&result).unwrap();
        assert_eq!(rv["role"], "tool");
        assert_eq!(rv["tool_call_id"], "call_1");
        assert_eq!(rv["content"], "file body");
        assert!(rv.get("tool_calls").is_none());
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
