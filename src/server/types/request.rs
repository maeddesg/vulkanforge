//! Request types for `POST /v1/chat/completions`.
//!
//! Mirrors `docs/v0.4/api_server_architecture.md` §3.1 with the
//! merge-sprint addition of `StreamOptions` (§2.1 row introduced
//! when `stream_options.include_usage` support was approved).

use serde::Deserialize;

/// Top-level JSON body of a chat completion request.
///
/// `deny_unknown_fields` is deliberately NOT applied here — OpenAI
/// clients regularly forward fields we don't model (`service_tier`,
/// future stream options, custom telemetry keys). Strict validation
/// happens on Tier-1 fields (`messages`, `role`); Tier-2 unknown
/// fields are silently ignored.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Single-Model server: accepted but ignored. The response will
    /// carry the *loaded* model id regardless. Decision: v0.4
    /// architecture §2 / Merge-Sprint change #1.
    pub model: String,

    pub messages: Vec<Message>,

    #[serde(default)]
    pub stream: bool,

    /// Only `include_usage: true` is honoured; other sub-fields are
    /// accepted-but-ignored. Merge-Sprint change #5.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    /// VF/llama.cpp extension. Currently honours one key:
    /// `enable_thinking: false` disables the ThinkFilter, surfacing
    /// `<think>...</think>` blocks verbatim. Default: filter ON for
    /// templates that emit `<think>` (Qwen3, DeepSeek-R1, Gemma-4-26B).
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,

    // ---- OpenAI sampling fields (all optional, default per §7.1) ----
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Accepted, logged, but ignored — VF's sampler has no
    /// presence-specific path. Spec §7.1.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// OpenAI telemetry field, no effect on generation.
    #[serde(default)]
    pub user: Option<String>,
    /// Number of choices. v0.4 supports only `n = 1`; anything else
    /// is rejected at request-validation time.
    #[serde(default)]
    pub n: Option<u32>,

    // ---- Accepted-but-ignored OpenAI fields (Tier-2) ----
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Function/tool definitions (OpenAI `tools`). When present, the
    /// handler renders them into the prompt (Qwen3/Hermes `<tools>`
    /// section) and parses `<tool_call>` blocks out of the output. v1
    /// = Qwen3/Hermes format only.
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    /// `"auto"` (default) or `"none"`. `"required"` / named-function
    /// forcing are documented follow-ups (parsed leniently — unknown
    /// values fall back to auto).
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Parsed for forward-compat; v1 emits at most the calls the model
    /// produces (no server-side parallelism control).
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    // ---- VF extensions (also accepted from clients that know us) ----
    /// VF-extension. When both `top_k` and OpenAI sampling fields
    /// are present, the VF-ext wins (§7.1).
    #[serde(default)]
    pub top_k: Option<u32>,
    /// VF-extension. Overrides the `frequency_penalty` → rep-penalty
    /// mapping when both are set (§7.2).
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// VF-extension, parsed for forward-compat. v0.4 doesn't honour
    /// it (no min-p sampler path yet); v0.5+ will (§7.1).
    #[serde(default)]
    pub min_p: Option<f32>,
}

/// Top-level JSON body of a legacy text-completion request
/// (`POST /v1/completions`). Identical sampling surface to
/// [`ChatCompletionRequest`]; the ONLY semantic difference is `prompt`
/// (a raw string) in place of `messages` + the skipped chat-template
/// step. `deny_unknown_fields` is deliberately NOT applied (same
/// rationale as chat).
///
/// `prompt` is a single string in this sprint; array-of-strings and
/// token-id-array prompts are documented follow-ups.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Single-Model server: accepted but ignored (mirrors chat).
    #[serde(default)]
    pub model: Option<String>,

    /// The raw prompt. Tokenized with special-token parsing (so a
    /// pre-rendered chat-template string round-trips), but NOT wrapped
    /// in any chat template.
    pub prompt: String,

    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    // ---- OpenAI sampling fields (shared with chat, §7.1) ----
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Accepted, logged, ignored (no presence path).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,
    /// Only `n = 1` supported (rejected otherwise, like chat).
    #[serde(default)]
    pub n: Option<u32>,

    // ---- Accepted-but-ignored legacy-completion fields (warn-logged) ----
    /// Text inserted after the completion — no fill-in-middle path.
    #[serde(default)]
    pub suffix: Option<String>,
    /// Server-side candidate generation; we only ever produce one.
    #[serde(default)]
    pub best_of: Option<u32>,
    /// Per-token logprobs — not emitted.
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Prepend the prompt to the completion — not implemented.
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,

    // ---- VF extensions (shared with chat) ----
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
}

/// `stream` is true and `stream_options.include_usage = true` → emit
/// an extra usage chunk before `data: [DONE]`. Merge-Sprint #5.
#[derive(Debug, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// llama.cpp-compatible knob for templates that emit thinking blocks.
/// `enable_thinking: false` lets the model generate the `<think>`
/// block but disables the server-side ThinkFilter, so the raw text
/// (including the tags) reaches the client. Useful for debugging
/// the chain-of-thought of thinking-mode models.
#[derive(Debug, Deserialize, Default)]
pub struct ChatTemplateKwargs {
    /// Default `true` — keep ThinkFilter ON (existing CLI default).
    #[serde(default = "default_true")]
    pub enable_thinking: bool,
}

fn default_true() -> bool {
    true
}

/// `stop` can be a single string or an array of strings (up to 4
/// per OpenAI spec). Length-clamping happens during request
/// validation, not at parse time.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Flatten into a `Vec<String>` for downstream processing.
    /// Returns at most `max_count` entries (excess silently dropped
    /// after validation has already rejected over-long requests).
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: Role,
    /// Optional: an `assistant` message that ONLY makes tool calls has
    /// `content: null`; a `tool` message carries the result here.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Present on `role:"tool"` messages — echoes the id of the
    /// `assistant.tool_calls[*].id` this result answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Present on `assistant` history messages that called tools — the
    /// prior turn's calls, rendered back into the prompt as
    /// `<tool_call>…</tool_call>` so multi-turn round-trips work.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallSpec>>,
}

/// One entry of the OpenAI `tools` array (v1: only `type:"function"`).
#[derive(Debug, Deserialize, Clone)]
pub struct ChatTool {
    #[serde(rename = "type", default = "default_function_type")]
    pub kind: String,
    pub function: ToolFunctionDef,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON-Schema for the arguments. Passed through verbatim into the
    /// rendered `<tools>` section (the model reads it as documentation).
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

fn default_function_type() -> String {
    "function".to_string()
}

/// OpenAI `tool_choice`. v1 honours the string forms `"auto"` / `"none"`;
/// an object (named-function forcing) parses but is treated as `auto`
/// (documented follow-up).
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(serde_json::Value),
}

impl ToolChoice {
    /// `true` when the client explicitly disabled tools (`"none"`).
    pub fn is_none_mode(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "none")
    }
}

/// A tool call as it appears in `assistant.tool_calls` history (same
/// wire shape as the response `ToolCall`).
#[derive(Debug, Deserialize, Clone)]
pub struct ToolCallSpec {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: ToolCallFunctionSpec,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolCallFunctionSpec {
    pub name: String,
    /// Arguments as a JSON **string** (OpenAI convention).
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// `developer` is an OpenAI alias added in late 2024 that
    /// behaves identically to `system` — we collapse the two.
    /// Merge-Sprint change #3.
    #[serde(alias = "developer")]
    System,
    User,
    Assistant,
    /// Parsed for forward compatibility, rejected by the handler in
    /// v0.4. Spec §8.1 (`unsupported_role`).
    Tool,
}

/// OpenAI permits `content` to be either a plain string or an array
/// of typed content parts (e.g. interleaved text + images). v0.4
/// accepts the array shape at parse time but rejects any non-text
/// parts in the handler.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    /// Parsed for forward-compat; rejected by handler in v0.4 with
    /// `unsupported_content_type` (§8.1).
    ImageUrl { image_url: serde_json::Value },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_request_parses_with_defaults() {
        let json = r#"{
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(req.model, "qwen3-8b");
        assert_eq!(req.messages.len(), 1);
        assert!(matches!(req.messages[0].role, Role::User));
        assert!(matches!(&req.messages[0].content, Some(MessageContent::Text(t)) if t == "Hi"));
        assert!(!req.stream);
        assert!(req.stream_options.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn full_request_parses_all_fields() {
        // Build the JSON with escape sequences so the literal "###"
        // (used as a stop-string in the test data) doesn't collide
        // with Rust 2024's raw-string-hash rules.
        let json = "{\
            \"model\": \"x\", \"messages\": [{\"role\":\"user\",\"content\":\"hi\"}],\
            \"stream\": true,\
            \"stream_options\": {\"include_usage\": true},\
            \"max_tokens\": 256, \"temperature\": 0.7, \"top_p\": 0.9,\
            \"frequency_penalty\": 1.2, \"presence_penalty\": 0.3,\
            \"stop\": [\"END\", \"###\"], \"seed\": 42, \"user\": \"user-1\", \"n\": 1,\
            \"top_k\": 40, \"repetition_penalty\": 1.1, \"min_p\": 0.05\
        }";
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert!(req.stream);
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.frequency_penalty, Some(1.2));
        assert_eq!(req.presence_penalty, Some(0.3));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.repetition_penalty, Some(1.1));
        assert_eq!(req.min_p, Some(0.05));

        match req.stop.unwrap() {
            StopSequence::Multiple(v) => assert_eq!(v, vec!["END", "###"]),
            _ => panic!("expected Multiple"),
        }
    }

    #[test]
    fn developer_role_aliases_to_system() {
        let json = r#"{"model":"x","messages":[
            {"role": "developer", "content": "You are concise."},
            {"role": "user", "content": "Hi"}
        ]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.messages[0].role, Role::System));
    }

    #[test]
    fn tool_role_parses_but_is_distinguishable() {
        // The handler will reject Tool later; parser must accept it.
        let json = r#"{"model":"x","messages":[
            {"role": "tool", "content": "tool output"}
        ]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.messages[0].role, Role::Tool));
    }

    #[test]
    fn missing_messages_field_fails() {
        let json = r#"{"model": "x"}"#;
        let err = serde_json::from_str::<ChatCompletionRequest>(json).unwrap_err();
        assert!(err.to_string().contains("messages"), "got: {err}");
    }

    #[test]
    fn invalid_role_fails() {
        let json = r#"{"model":"x","messages":[{"role":"bogus","content":"x"}]}"#;
        let err = serde_json::from_str::<ChatCompletionRequest>(json).unwrap_err();
        assert!(err.to_string().contains("role") || err.to_string().contains("variant"), "got: {err}");
    }

    #[test]
    fn stop_as_single_string() {
        let json = r#"{"model":"x","messages":[{"role":"user","content":"x"}],"stop":"<END>"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match req.stop.unwrap() {
            StopSequence::Single(s) => assert_eq!(s, "<END>"),
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn stop_as_array() {
        let json = r#"{"model":"x","messages":[{"role":"user","content":"x"}],"stop":["a","b"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match req.stop.unwrap() {
            StopSequence::Multiple(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Multiple"),
        }
    }

    #[test]
    fn stop_sequence_into_vec_flattens() {
        assert_eq!(StopSequence::Single("X".into()).into_vec(), vec!["X"]);
        assert_eq!(
            StopSequence::Multiple(vec!["a".into(), "b".into()]).into_vec(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn content_array_with_text_part_parses() {
        let json = r#"{"model":"x","messages":[
            {"role":"user","content":[{"type":"text","text":"Hello"}]}
        ]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 1);
                assert!(matches!(&parts[0], ContentPart::Text { text } if text == "Hello"));
            }
            _ => panic!("expected Parts"),
        }
    }

    #[test]
    fn content_array_with_image_url_parses_but_is_distinguishable() {
        // Handler will 400 with `unsupported_content_type`; parser
        // must round-trip the JSON so the validator can see it.
        let json = r#"{"model":"x","messages":[
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}
        ]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                assert!(matches!(&parts[0], ContentPart::ImageUrl { .. }));
            }
            _ => panic!("expected Parts"),
        }
    }

    #[test]
    fn unknown_top_level_field_silently_accepted() {
        // OpenAI clients send service_tier / store / etc.
        let json = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"x"}],
            "service_tier": "auto",
            "store": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "x");
    }
}
