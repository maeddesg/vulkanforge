//! Non-streaming response types for `POST /v1/chat/completions` and
//! the simpler `GET /v1/models`, `GET /health` payloads.
//!
//! Mirrors §3.2, §3.4, §3.5 of the architecture document.

use serde::Serialize;

// =========================================================================
// §3.2 — Chat Completion Response (non-streaming)
// =========================================================================

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// `chatcmpl-<uuid_v4_hex_no_dashes>`.
    pub id: String,
    pub object: &'static str, // always "chat.completion"
    /// Unix epoch seconds.
    pub created: u64,
    /// The loaded model's id, NOT the request's `model` field
    /// (Merge-Sprint #1 — `model` in the request is ignored).
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, choice: Choice, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion",
            created: unix_now_secs(),
            model,
            system_fingerprint: Some(crate::server::types::SYSTEM_FINGERPRINT.into()),
            choices: vec![choice],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str, // always "assistant"
    pub content: String,
}

impl AssistantMessage {
    pub fn new(content: String) -> Self {
        Self { role: "assistant", content }
    }
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// EOS token, stop-string match, or template-boundary token.
    Stop,
    /// `max_tokens` cap reached.
    Length,
    /// Reserved for v0.5+ (content-filter integration).
    ContentFilter,
    // ToolCalls reserved for v0.5+ once tool-calling lands.
}

#[derive(Debug, Serialize, Clone, Copy, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

// =========================================================================
// §3.4 — Models list
// =========================================================================

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str, // always "list"
    pub data: Vec<ModelInfo>,
}

impl ModelListResponse {
    pub fn single(model: ModelInfo) -> Self {
        Self { object: "list", data: vec![model] }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str, // always "model"
    pub created: u64,
    pub owned_by: &'static str, // always "vulkanforge"
    pub permission: Vec<serde_json::Value>, // always empty in v0.4
    pub root: String, // mirrors id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>, // always None in v0.4
}

impl ModelInfo {
    pub fn from_id(id: String) -> Self {
        let root = id.clone();
        Self {
            id,
            object: "model",
            created: unix_now_secs(),
            owned_by: "vulkanforge",
            permission: Vec::new(),
            root,
            parent: None,
        }
    }
}

// =========================================================================
// §3.5 — Health
// =========================================================================

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub model_loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache: Option<KvCacheInfo>,
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    Loading,
    Error,
}

#[derive(Debug, Serialize)]
pub struct KvCacheInfo {
    pub max_seq_len: u32,
    pub current_pos: u32,
}

// =========================================================================
// Helpers
// =========================================================================

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_completion_response_serializes_to_openai_shape() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc123".into(),
            object: "chat.completion",
            created: 1715420000,
            model: "qwen3-8b".into(),
            system_fingerprint: Some("vulkanforge-0.4.0".into()),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage::new("Paris".into()),
                logprobs: None,
                finish_reason: FinishReason::Stop,
            }],
            usage: Usage::new(28, 1),
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["id"], "chatcmpl-abc123");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Paris");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 28);
        assert_eq!(v["usage"]["completion_tokens"], 1);
        assert_eq!(v["usage"]["total_tokens"], 29);
        // logprobs is Option<None> → skipped
        assert!(v["choices"][0].get("logprobs").is_none());
    }

    #[test]
    fn usage_total_is_sum_of_prompt_and_completion() {
        let u = Usage::new(42, 17);
        assert_eq!(u.total_tokens, 59);
    }

    #[test]
    fn finish_reason_serializes_snake_case() {
        assert_eq!(serde_json::to_value(FinishReason::Stop).unwrap(), json!("stop"));
        assert_eq!(serde_json::to_value(FinishReason::Length).unwrap(), json!("length"));
        assert_eq!(serde_json::to_value(FinishReason::ContentFilter).unwrap(), json!("content_filter"));
    }

    #[test]
    fn model_info_root_mirrors_id() {
        let info = ModelInfo::from_id("qwen3-8b".into());
        assert_eq!(info.id, info.root);
        assert_eq!(info.owned_by, "vulkanforge");
        assert!(info.permission.is_empty());
        assert!(info.parent.is_none());
        assert_eq!(info.object, "model");
    }

    #[test]
    fn model_list_single_wraps_into_data_array() {
        let resp = ModelListResponse::single(ModelInfo::from_id("qwen3-8b".into()));
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert_eq!(v["data"][0]["id"], "qwen3-8b");
    }

    #[test]
    fn health_status_serializes_snake_case() {
        assert_eq!(serde_json::to_value(HealthStatus::Ok).unwrap(), json!("ok"));
        assert_eq!(serde_json::to_value(HealthStatus::Loading).unwrap(), json!("loading"));
        assert_eq!(serde_json::to_value(HealthStatus::Error).unwrap(), json!("error"));
    }

    #[test]
    fn health_skips_optional_fields_when_none() {
        let h = HealthResponse {
            status: HealthStatus::Loading,
            model_loaded: false,
            model_id: None,
            version: "0.4.0",
            kv_cache: None,
        };
        let v = serde_json::to_value(&h).unwrap();
        assert!(v.get("model_id").is_none());
        assert!(v.get("kv_cache").is_none());
        assert_eq!(v["status"], "loading");
    }
}
