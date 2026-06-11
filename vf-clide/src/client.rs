// SPDX-License-Identifier: GPL-3.0-only
//! Minimal OpenAI chat-completions client over HTTP.
//!
//! Talks to a running `vulkanforge serve` instance. Supports both
//! non-streaming (`chat_once`) and SSE streaming (`chat_stream`). Phase 1
//! sends chat only — no `tools` parameter (that is Phase 2).

use futures_util::StreamExt;

use crate::types::{ChatChunk, ChatMessage, ChatRequest, ChatResponse};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Result of a completion: the visible text plus the server's
/// `finish_reason` (`"stop"` | `"length"` | …), needed so the caller can
/// flag truncation / empty answers.
#[derive(Debug, Clone, Default)]
pub struct ChatOutcome {
    pub text: String,
    pub finish_reason: Option<String>,
}

/// If generation stopped at the token cap (`finish_reason == "length"`),
/// return an actionable truncation notice; otherwise `None`.
pub fn truncation_notice(finish_reason: Option<&str>, max_tokens: Option<u32>) -> Option<String> {
    if finish_reason == Some("length") {
        let n = max_tokens.map(|m| m.to_string()).unwrap_or_else(|| "the server default".into());
        Some(format!("[truncated at the token limit ({n}) — raise it with --max-tokens / /max-tokens]"))
    } else {
        None
    }
}

/// If the visible answer is empty/whitespace (the server stripped a
/// `<think>` block that consumed the whole budget), return an actionable
/// empty-answer notice; otherwise `None`.
pub fn empty_notice(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        Some(
            "[empty answer — the budget was likely consumed by the <think> block; \
             give more tokens (--max-tokens) or disable thinking (--no-think, or /no-think in the REPL)]"
                .to_string(),
        )
    } else {
        None
    }
}

/// A chat client bound to one server + model. `model`/`temperature`/
/// `max_tokens` are public so the REPL can flip them at runtime.
pub struct Client {
    http: reqwest::Client,
    base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Client {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self {
            http: reqwest::Client::new(),
            base_url,
            model: model.into(),
            temperature: Some(0.0),
            max_tokens: None,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    /// Build the request body (public for testability).
    pub fn build_request(&self, messages: Vec<ChatMessage>, stream: bool) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream,
        }
    }

    /// Non-streaming completion → text + finish_reason.
    pub async fn chat_once(&self, messages: Vec<ChatMessage>) -> Result<ChatOutcome> {
        let req = self.build_request(messages, false);
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }
        let parsed: ChatResponse = resp.json().await?;
        let choice = parsed.choices.into_iter().next();
        let finish_reason = choice.as_ref().and_then(|c| c.finish_reason.clone());
        let text = choice.and_then(|c| c.message.content).unwrap_or_default();
        Ok(ChatOutcome { text, finish_reason })
    }

    /// Streaming completion. `on_token` is called for each content delta
    /// as it arrives; the full text + finish_reason are also returned.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        mut on_token: impl FnMut(&str),
    ) -> Result<ChatOutcome> {
        let req = self.build_request(messages, true);
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }

        let mut full = String::new();
        let mut finish: Option<String> = None;
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            // SSE events are separated by a blank line ("\n\n").
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf[..pos].to_string();
                buf.drain(..pos + 2);
                if handle_event(&event, &mut on_token, &mut full, &mut finish) {
                    break 'outer; // `[DONE]`
                }
            }
        }
        Ok(ChatOutcome { text: full, finish_reason: finish })
    }
}

/// Process one SSE event block. Returns `true` on the `[DONE]` marker.
/// Appends content deltas (via `on_token` + `full`) and records the last
/// non-null `finish_reason` seen.
fn handle_event(
    event: &str,
    on_token: &mut impl FnMut(&str),
    full: &mut String,
    finish: &mut Option<String>,
) -> bool {
    for line in event.lines() {
        let line = line.trim_start();
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            return true;
        }
        if let Ok(parsed) = serde_json::from_str::<ChatChunk>(data) {
            if let Some(choice) = parsed.choices.into_iter().next() {
                if let Some(fr) = choice.finish_reason {
                    *finish = Some(fr);
                }
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        on_token(&text);
                        full.push_str(&text);
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_trailing_slash_trimmed() {
        let c = Client::new("http://localhost:8080/", "m");
        assert_eq!(c.endpoint(), "http://localhost:8080/v1/chat/completions");
    }

    #[test]
    fn build_request_carries_model_and_stream() {
        let c = Client::new("http://localhost:8080", "Qwen3-14B-Q4_K_M");
        let r = c.build_request(vec![ChatMessage::user("hi")], true);
        assert_eq!(r.model, "Qwen3-14B-Q4_K_M");
        assert!(r.stream);
        assert_eq!(r.temperature, Some(0.0));
        assert_eq!(r.messages.len(), 1);
    }

    #[test]
    fn handle_event_extracts_delta_done_and_finish() {
        let mut got = String::new();
        let mut full = String::new();
        let mut finish = None;
        let done = handle_event(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
            &mut |t| got.push_str(t),
            &mut full,
            &mut finish,
        );
        assert!(!done);
        assert_eq!(got, "Hel");
        assert_eq!(full, "Hel");
        assert!(finish.is_none());

        // Final chunk carries finish_reason.
        let _ = handle_event(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}",
            &mut |_t| {},
            &mut full,
            &mut finish,
        );
        assert_eq!(finish.as_deref(), Some("length"));

        let done = handle_event("data: [DONE]", &mut |_t| {}, &mut full, &mut finish);
        assert!(done);
    }

    #[test]
    fn handle_event_ignores_non_data_lines() {
        let mut full = String::new();
        let mut finish = None;
        let done = handle_event(": keep-alive comment", &mut |_t| {}, &mut full, &mut finish);
        assert!(!done);
        assert!(full.is_empty());
    }

    #[test]
    fn truncation_notice_only_on_length() {
        assert!(truncation_notice(Some("length"), Some(2048)).unwrap().contains("2048"));
        assert!(truncation_notice(Some("stop"), Some(2048)).is_none());
        assert!(truncation_notice(None, None).is_none());
    }

    #[test]
    fn empty_notice_only_on_blank() {
        assert!(empty_notice("").is_some());
        assert!(empty_notice("   \n  ").is_some());
        assert!(empty_notice("Hello").is_none());
        // a short but present answer must NOT trip the empty marker
        assert!(empty_notice("4").is_none());
    }
}
