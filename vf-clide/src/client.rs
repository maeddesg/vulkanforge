// SPDX-License-Identifier: GPL-3.0-only
//! Minimal OpenAI chat-completions client over HTTP.
//!
//! Talks to a running `vulkanforge serve` instance. Supports both
//! non-streaming (`chat_once`) and SSE streaming (`chat_stream`). Phase 1
//! sends chat only — no `tools` parameter (that is Phase 2).

use futures_util::StreamExt;

use crate::types::{ChatChunk, ChatMessage, ChatRequest, ChatResponse};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

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

    /// Non-streaming completion → the full assistant content.
    pub async fn chat_once(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let req = self.build_request(messages, false);
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }
        let parsed: ChatResponse = resp.json().await?;
        Ok(parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default())
    }

    /// Streaming completion. `on_token` is called for each content delta
    /// as it arrives; the full concatenated text is also returned.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let req = self.build_request(messages, true);
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }

        let mut full = String::new();
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            // SSE events are separated by a blank line ("\n\n").
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf[..pos].to_string();
                buf.drain(..pos + 2);
                if let Some(done) = handle_event(&event, &mut on_token, &mut full) {
                    if done {
                        return Ok(full);
                    }
                }
            }
        }
        Ok(full)
    }
}

/// Process one SSE event block. Returns `Some(true)` on the `[DONE]`
/// marker, `Some(false)` for a handled data line, `None` for ignorable
/// lines (comments / blanks).
fn handle_event(event: &str, on_token: &mut impl FnMut(&str), full: &mut String) -> Option<bool> {
    let mut handled = None;
    for line in event.lines() {
        let line = line.trim_start();
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            return Some(true);
        }
        if let Ok(parsed) = serde_json::from_str::<ChatChunk>(data) {
            if let Some(choice) = parsed.choices.into_iter().next() {
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        on_token(&text);
                        full.push_str(&text);
                    }
                }
            }
            handled = Some(false);
        }
    }
    handled
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
    fn handle_event_extracts_delta_and_done() {
        let mut got = String::new();
        let mut full = String::new();
        let r = handle_event(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
            &mut |t| got.push_str(t),
            &mut full,
        );
        assert_eq!(r, Some(false));
        assert_eq!(got, "Hel");
        assert_eq!(full, "Hel");

        let done = handle_event("data: [DONE]", &mut |_t| {}, &mut full);
        assert_eq!(done, Some(true));
    }

    #[test]
    fn handle_event_ignores_non_data_lines() {
        let mut full = String::new();
        let r = handle_event(": keep-alive comment", &mut |_t| {}, &mut full);
        assert_eq!(r, None);
        assert!(full.is_empty());
    }
}
