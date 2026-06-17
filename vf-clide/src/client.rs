// SPDX-License-Identifier: GPL-3.0-only
//! Minimal OpenAI chat-completions client over HTTP.
//!
//! Talks to a running `vulkanforge serve` instance. Supports both
//! non-streaming (`chat_once`) and SSE streaming (`chat_stream`). Phase 1
//! sends chat only — no `tools` parameter (that is Phase 2).

use futures_util::StreamExt;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::types::{
    ArchiveResponse, ChatChunk, ChatMessage, ChatRequest, ChatResponse, CurateRequest,
    DeleteResponse, ProjectsResponse, RecallRequest, RecallResponse, RememberRequest,
    RememberResponse, StreamOptions, Tool, ToolCall, UnarchiveResponse, Usage,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Outcome of a `/memory/*` call. A thin client must treat the three states
/// distinctly: transport failure (`Err` — the server is unreachable), memory
/// **off** (`Disabled` — HTTP 503, a normal opt-in state since memory is
/// off by default, **never** an error), and success (`Ok`).
#[derive(Debug)]
pub enum MemCall<T> {
    Ok(T),
    Disabled,
}

/// Whether an HTTP status means "memory subsystem not enabled" — the server
/// returns **503** for `/memory/*` when started without `--memory`. Pure, so
/// the 503-vs-transport branch is unit-testable without a live server.
pub fn is_memory_disabled(status: u16) -> bool {
    status == 503
}

/// Result of a completion: the visible text plus the server's
/// `finish_reason` (`"stop"` | `"length"` | …), needed so the caller can
/// flag truncation / empty answers.
#[derive(Debug, Clone, Default)]
pub struct ChatOutcome {
    pub text: String,
    pub finish_reason: Option<String>,
    /// Server-reported token usage (non-stream always; stream when
    /// `include_usage` was set). `None` if the server omitted it.
    pub usage: Option<Usage>,
}

/// Result of one agent turn (a `tools`-enabled completion): the
/// assistant's text (often `None` when it only called tools), the tool
/// calls it requested, and the finish_reason.
#[derive(Debug, Clone, Default)]
pub struct AgentTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    /// Server-reported token usage for this turn (`None` if omitted).
    pub usage: Option<Usage>,
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
            // Match the CLI default so library users (e.g. the agent loop)
            // don't silently inherit the server's small floor and get a
            // thinking model's answer eaten by its `<think>` block.
            max_tokens: Some(6144),
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
            tools: None,
            stream_options: None,
        }
    }

    /// Non-streaming completion → text + finish_reason + usage.
    pub async fn chat_once(&self, messages: Vec<ChatMessage>) -> Result<ChatOutcome> {
        let req = self.build_request(messages, false);
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }
        let parsed: ChatResponse = resp.json().await?;
        let usage = parsed.usage;
        let choice = parsed.choices.into_iter().next();
        let finish_reason = choice.as_ref().and_then(|c| c.finish_reason.clone());
        let text = choice.and_then(|c| c.message.content).unwrap_or_default();
        Ok(ChatOutcome { text, finish_reason, usage })
    }

    /// One agent turn: non-streaming completion **with `tools`**, returning
    /// the assistant's text (if any), any `tool_calls` it made, and the
    /// finish_reason. Tool calls require the full message, so this path is
    /// non-streaming.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: &[Tool],
    ) -> Result<AgentTurn> {
        let mut req = self.build_request(messages, false);
        req.tools = Some(tools.to_vec());
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }
        let parsed: ChatResponse = resp.json().await?;
        let usage = parsed.usage;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or("server returned no choices")?;
        let finish_reason = choice.finish_reason;
        let tool_calls = choice.message.tool_calls.unwrap_or_default();
        Ok(AgentTurn { content: choice.message.content, tool_calls, finish_reason, usage })
    }

    /// Streaming completion. `on_token` is called for each content delta
    /// as it arrives; the full text + finish_reason are also returned.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        mut on_token: impl FnMut(&str),
    ) -> Result<ChatOutcome> {
        let mut req = self.build_request(messages, true);
        // Ask the server to append a final `usage` chunk (verified: VF
        // sends `choices:[] + usage` before `[DONE]`).
        req.stream_options = Some(StreamOptions { include_usage: true });
        let resp = self.http.post(self.endpoint()).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("server returned {status}: {body}").into());
        }

        let mut full = String::new();
        let mut finish: Option<String> = None;
        let mut usage: Option<Usage> = None;
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            // SSE events are separated by a blank line ("\n\n").
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf[..pos].to_string();
                buf.drain(..pos + 2);
                if handle_event(&event, &mut on_token, &mut full, &mut finish, &mut usage) {
                    break 'outer; // `[DONE]`
                }
            }
        }
        Ok(ChatOutcome { text: full, finish_reason: finish, usage })
    }

    // ---- Memory subsystem (VF-native `/memory/*`, opt-in on the server) ----

    fn memory_url(&self, path: &str) -> String {
        format!("{}/memory/{}", self.base_url, path)
    }

    /// `POST /memory/recall`. `Ok(MemCall::Disabled)` on 503 (memory off);
    /// `Err` only on transport / other HTTP errors. The server applies the
    /// `search_query:` prefix, so `query` is the raw user text.
    pub async fn memory_recall(
        &self,
        project_key: Option<&str>,
        query: &str,
        k: u32,
    ) -> Result<MemCall<RecallResponse>> {
        let body = RecallRequest {
            project_key: project_key.map(str::to_string),
            query: query.to_string(),
            k,
        };
        self.memory_post("recall", &body).await
    }

    /// `POST /memory/remember`. Stufe B-1 stores manual notes (`kind:"Note"`).
    pub async fn memory_remember(
        &self,
        project_key: Option<&str>,
        kind: &str,
        text: &str,
    ) -> Result<MemCall<RememberResponse>> {
        let body = RememberRequest {
            project_key: project_key.map(str::to_string),
            kind: kind.to_string(),
            text: text.to_string(),
        };
        self.memory_post("remember", &body).await
    }

    /// `GET /memory/projects` — the scopes the server already knows.
    pub async fn memory_projects(&self) -> Result<MemCall<ProjectsResponse>> {
        let resp = self.http.get(self.memory_url("projects")).send().await?;
        self.memory_parse(resp).await
    }

    /// `POST /memory/archive` — drop a note from recall, keep it as a record.
    pub async fn memory_archive(
        &self,
        project_key: Option<&str>,
        id: i64,
    ) -> Result<MemCall<ArchiveResponse>> {
        let body = CurateRequest { project_key: project_key.map(str::to_string), id };
        self.memory_post("archive", &body).await
    }

    /// `POST /memory/delete` — hard-remove a note from recall and the graph.
    pub async fn memory_delete(
        &self,
        project_key: Option<&str>,
        id: i64,
    ) -> Result<MemCall<DeleteResponse>> {
        let body = CurateRequest { project_key: project_key.map(str::to_string), id };
        self.memory_post("delete", &body).await
    }

    /// `POST /memory/unarchive` — restore an archived note to active + recall
    /// (the inverse of `memory_archive`; user-driven recovery).
    pub async fn memory_unarchive(
        &self,
        project_key: Option<&str>,
        id: i64,
    ) -> Result<MemCall<UnarchiveResponse>> {
        let body = CurateRequest { project_key: project_key.map(str::to_string), id };
        self.memory_post("unarchive", &body).await
    }

    async fn memory_post<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<MemCall<T>> {
        let resp = self.http.post(self.memory_url(path)).json(body).send().await?;
        self.memory_parse(resp).await
    }

    /// Map a `/memory/*` HTTP response to a [`MemCall`]: 503 → `Disabled`
    /// (memory off), other non-success → `Err`, success → parse the body.
    async fn memory_parse<T: DeserializeOwned>(
        &self,
        resp: reqwest::Response,
    ) -> Result<MemCall<T>> {
        let status = resp.status();
        if is_memory_disabled(status.as_u16()) {
            return Ok(MemCall::Disabled);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("memory: server returned {status}: {body}").into());
        }
        Ok(MemCall::Ok(resp.json().await?))
    }
}

/// Process one SSE event block. Returns `true` on the `[DONE]` marker.
/// Appends content deltas (via `on_token` + `full`), records the last
/// non-null `finish_reason`, and captures the final `usage` chunk (which
/// carries empty `choices` + `usage`).
fn handle_event(
    event: &str,
    on_token: &mut impl FnMut(&str),
    full: &mut String,
    finish: &mut Option<String>,
    usage: &mut Option<Usage>,
) -> bool {
    for line in event.lines() {
        let line = line.trim_start();
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            return true;
        }
        if let Ok(parsed) = serde_json::from_str::<ChatChunk>(data) {
            // Final usage chunk: empty choices + usage — capture, don't error.
            if let Some(u) = parsed.usage {
                *usage = Some(u);
            }
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
        assert!(r.tools.is_none());
        // build_request itself doesn't set stream_options — chat_stream does.
        assert!(r.stream_options.is_none());
    }

    #[test]
    fn default_max_tokens_is_6144() {
        // Phase 1: the library Client defaults to the CLI budget (6144),
        // not None → server floor (200), so the agent loop isn't truncated.
        let c = Client::new("http://localhost:8080", "m");
        assert_eq!(c.max_tokens, Some(6144));
    }

    #[test]
    fn handle_event_extracts_delta_done_and_finish() {
        let mut got = String::new();
        let mut full = String::new();
        let mut finish = None;
        let mut usage = None;
        let done = handle_event(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
            &mut |t| got.push_str(t),
            &mut full,
            &mut finish,
            &mut usage,
        );
        assert!(!done);
        assert_eq!(got, "Hel");
        assert_eq!(full, "Hel");
        assert!(finish.is_none());
        assert!(usage.is_none());

        // Final chunk carries finish_reason.
        let _ = handle_event(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}",
            &mut |_t| {},
            &mut full,
            &mut finish,
            &mut usage,
        );
        assert_eq!(finish.as_deref(), Some("length"));

        // The verified final usage chunk (empty choices + usage) is captured.
        let _ = handle_event(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":14,\"completion_tokens\":16,\"total_tokens\":30}}",
            &mut |_t| {},
            &mut full,
            &mut finish,
            &mut usage,
        );
        assert_eq!(usage.unwrap().completion_tokens, Some(16));

        let done = handle_event("data: [DONE]", &mut |_t| {}, &mut full, &mut finish, &mut usage);
        assert!(done);
    }

    #[test]
    fn handle_event_ignores_non_data_lines() {
        let mut full = String::new();
        let mut finish = None;
        let mut usage = None;
        let done = handle_event(": keep-alive comment", &mut |_t| {}, &mut full, &mut finish, &mut usage);
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

    #[test]
    fn memory_disabled_only_on_503() {
        // 503 = memory off (a normal opt-in state); everything else is not.
        assert!(is_memory_disabled(503));
        assert!(!is_memory_disabled(200));
        assert!(!is_memory_disabled(404));
        assert!(!is_memory_disabled(500));
        assert!(!is_memory_disabled(502));
    }

    #[test]
    fn memory_url_joins_base_and_path() {
        let c = Client::new("http://localhost:8080/", "m");
        assert_eq!(c.memory_url("recall"), "http://localhost:8080/memory/recall");
        assert_eq!(c.memory_url("projects"), "http://localhost:8080/memory/projects");
    }
}
