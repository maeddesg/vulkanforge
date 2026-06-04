//! Function/tool-calling glue (OpenAI-compatible, Qwen3/Hermes format).
//!
//! Two model-specific halves, both confined here so the hand-rolled
//! chat template stays untouched:
//!  - **Render** tool definitions + tool-call/tool-result history into
//!    the prompt as plain ChatML text (the Qwen3 `<tools>` system
//!    section, `<tool_call>` for assistant calls, `<tool_response>` for
//!    results). The existing ChatML renderer then tokenizes it verbatim.
//!  - **Parse** `<tool_call>{json}</tool_call>` blocks out of the model
//!    output back into the OpenAI `tool_calls` shape.
//!
//! v1 scope: Qwen3/Hermes only, `tool_choice` auto/none. Verified
//! against Qwen3-8B (the model emits
//! `<tool_call>\n{"name":…,"arguments":{…}}\n</tool_call>`).

use crate::server::types::request::{ChatTool, ToolCallSpec};
use crate::server::types::response::ToolCall;

/// Build the Qwen3/Hermes tool system-prompt section. Appended to (or
/// becomes) the `system` message content. Mirrors the upstream Qwen3
/// `chat_template.jinja` tools block 1:1.
pub fn render_tools_section(tools: &[ChatTool]) -> String {
    let mut s = String::new();
    s.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
    s.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for t in tools {
        // One compact JSON object per line, exactly as the Qwen3 template
        // emits (`{"type": "function", "function": {...}}`).
        let obj = serde_json::json!({
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            }
        });
        s.push('\n');
        s.push_str(&serde_json::to_string(&obj).unwrap_or_default());
    }
    s.push_str("\n</tools>\n\n");
    s.push_str(
        "For each function call, return a json object with function name and arguments within \
         <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

/// Render an `assistant` history message that made tool calls back into
/// the prompt: optional leading text, then one `<tool_call>` block per
/// call (Qwen3 format). Used for the multi-turn round-trip.
pub fn render_assistant_tool_calls(text: Option<&str>, calls: &[ToolCallSpec]) -> String {
    let mut s = String::new();
    if let Some(t) = text {
        if !t.is_empty() {
            s.push_str(t);
            s.push('\n');
        }
    }
    for (i, c) in calls.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        let args: serde_json::Value = c
            .function
            .arguments
            .as_deref()
            .and_then(|a| serde_json::from_str(a).ok())
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let obj = serde_json::json!({ "name": c.function.name, "arguments": args });
        s.push_str("<tool_call>\n");
        s.push_str(&serde_json::to_string(&obj).unwrap_or_default());
        s.push_str("\n</tool_call>");
    }
    s
}

/// Render a `role:"tool"` result message into the prompt. Qwen3 wraps
/// tool results in a `user` turn containing `<tool_response>`.
pub fn render_tool_result(content: &str) -> String {
    format!("<tool_response>\n{content}\n</tool_response>")
}

/// Parse `<tool_call>…</tool_call>` blocks out of generated text.
///
/// Returns `(leading_text, tool_calls)`:
///  - `leading_text` = the non-tool-call text (trimmed); `None` if empty.
///  - one [`ToolCall`] per well-formed block. Robust to: multiple calls,
///    text+call mix, malformed JSON (best-effort name + raw args, never
///    panics). No `<tool_call>` → `(Some(text), [])`.
pub fn parse_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut text_parts = String::new();
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        text_parts.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len()..];
        let (inner, next) = match after.find(CLOSE) {
            Some(close) => (&after[..close], &after[close + CLOSE.len()..]),
            // Unterminated block (e.g. max_tokens hit mid-call): take the
            // remainder as the inner, stop.
            None => (after, ""),
        };
        if let Some(tc) = parse_one_call(inner.trim(), calls.len()) {
            calls.push(tc);
        }
        rest = next;
    }
    text_parts.push_str(rest);
    let trimmed = text_parts.trim();
    let content = if trimmed.is_empty() { None } else { Some(trimmed.to_string()) };
    (content, calls)
}

/// Parse one block's inner JSON (`{"name":…,"arguments":{…}}`) into a
/// [`ToolCall`]. Best-effort on malformed JSON.
fn parse_one_call(inner: &str, idx: usize) -> Option<ToolCall> {
    let id = format!("call_{}", short_id(idx));
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(inner) {
        let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
        if name.is_empty() {
            return None;
        }
        // Arguments → JSON string (OpenAI convention). Re-serialize the
        // object; default to "{}" when absent.
        let arguments = v
            .get("arguments")
            .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());
        return Some(ToolCall::new(id, name, arguments));
    }
    // Malformed JSON — salvage a name + return the raw inner as args so
    // the call surfaces instead of being silently dropped.
    if let Some(name) = salvage_name(inner) {
        return Some(ToolCall::new(id, name, inner.to_string()));
    }
    None
}

fn salvage_name(inner: &str) -> Option<String> {
    // Look for `"name"` : `"value"` without a full JSON parse.
    let key = inner.find("\"name\"")?;
    let after = &inner[key + 6..];
    let colon = after.find(':')?;
    let after = after[colon + 1..].trim_start();
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

/// Short deterministic-ish id suffix. OpenAI uses `call_<random>`; we
/// use a uuid simple form, falling back to the index if needed.
fn short_id(idx: usize) -> String {
    let u = uuid::Uuid::new_v4().simple().to_string();
    format!("{}{idx}", &u[..u.len().min(20)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_call_parses() {
        let (content, calls) = parse_tool_calls(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>",
        );
        assert!(content.is_none());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"Paris\"}");
        assert_eq!(calls[0].kind, "function");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn text_plus_call_splits() {
        let (content, calls) =
            parse_tool_calls("Let me check.\n<tool_call>\n{\"name\":\"f\",\"arguments\":{}}\n</tool_call>");
        assert_eq!(content.as_deref(), Some("Let me check."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn multiple_calls() {
        let (_c, calls) = parse_tool_calls(
            "<tool_call>\n{\"name\":\"a\",\"arguments\":{\"x\":1}}\n</tool_call>\n\
             <tool_call>\n{\"name\":\"b\",\"arguments\":{}}\n</tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn no_tool_call_is_plain_text() {
        let (content, calls) = parse_tool_calls("Just a normal answer.");
        assert_eq!(content.as_deref(), Some("Just a normal answer."));
        assert!(calls.is_empty());
    }

    #[test]
    fn missing_arguments_defaults_empty_object() {
        let (_c, calls) = parse_tool_calls("<tool_call>\n{\"name\":\"ping\"}\n</tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn malformed_json_salvages_name_not_crash() {
        // Truncated/!valid JSON but a recognizable name → surface the call.
        let (_c, calls) =
            parse_tool_calls("<tool_call>\n{\"name\": \"do_thing\", \"arguments\": {bad}\n</tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "do_thing");
    }

    #[test]
    fn unterminated_block_does_not_panic() {
        let (_c, calls) = parse_tool_calls("<tool_call>\n{\"name\":\"x\",\"arguments\":{\"a\":1}");
        // Best-effort: salvages name.
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "x");
    }

    #[test]
    fn render_tools_section_has_tags_and_schema() {
        use crate::server::types::request::{ToolFunctionDef};
        let t = ChatTool {
            kind: "function".into(),
            function: ToolFunctionDef {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                parameters: Some(serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}})),
            },
        };
        let s = render_tools_section(&[t]);
        assert!(s.contains("<tools>"));
        assert!(s.contains("</tools>"));
        assert!(s.contains("get_weather"));
        assert!(s.contains("<tool_call>"));
    }
}
