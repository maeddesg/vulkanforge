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

// =====================================================================
// Gemma-4 tool format (Sprint gemma-tool-layer)
// =====================================================================
//
// Authoritative format, extracted from the gemma-4-26B GGUF
// `tokenizer.chat_template` (NOT guessed):
//   call:   <|tool_call>call:NAME{key:value,…}<tool_call|>
//   args:   non-JSON `key:value` pairs; strings are `"…"` (the `<|"|>`
//           quote special-token detokenises to `"`).
// The `<tool_call|>` terminator is also the runaway-stop marker the
// server watches for (the gemma tool tokens are not in the EOG set).

/// The gemma-4 tool-call terminator. The server adds this as a
/// string-level stop (via the cancel flag) when a gemma model has tools
/// active, killing the `<|tool_response>`-repetition runaway.
pub const GEMMA_TOOL_CALL_END: &str = "<tool_call|>";

/// **Permissive** gemma tool-call parser. Recognises BOTH the gemma-4
/// native form (`<|tool_call>call:NAME{k:v}<tool_call|>`, emitted by
/// gemma-QAT) AND the Qwen/Hermes `<tool_call>{json}</tool_call>` form
/// (emitted by gemma-Q3, which follows the in-prompt instruction). Both
/// normalise to the same OpenAI `tool_calls` output. Robust: multiple
/// calls, text+call mix, malformed args, unterminated block — never panics.
pub fn parse_tool_calls_gemma(text: &str) -> (Option<String>, Vec<ToolCall>) {
    // 1) Extract gemma-native `<|tool_call>…<tool_call|>` blocks first
    //    (their delimiters don't overlap the Qwen `<tool_call>` literal).
    let (residual, mut calls) = parse_gemma_native(text);
    // 2) Run the Qwen parser on what's left (gemma-Q3 fallback path).
    let (content, qwen_calls) = parse_tool_calls(&residual);
    calls.extend(qwen_calls);
    (content, calls)
}

fn parse_gemma_native(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<|tool_call>";
    const CLOSE: &str = "<tool_call|>";
    let mut residual = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        residual.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len()..];
        let (inner, next) = match after.find(CLOSE) {
            Some(c) => (&after[..c], &after[c + CLOSE.len()..]),
            None => (after, ""), // unterminated (max_tokens / cancel mid-call)
        };
        if let Some(tc) = parse_gemma_one(inner.trim(), calls.len()) {
            calls.push(tc);
        }
        rest = next;
    }
    residual.push_str(rest);
    (residual, calls)
}

/// Parse one gemma block inner (`call:NAME{key:value,…}`) → [`ToolCall`].
fn parse_gemma_one(inner: &str, idx: usize) -> Option<ToolCall> {
    let inner = inner.strip_prefix("call:").unwrap_or(inner).trim();
    let brace = inner.find('{')?;
    let name = inner[..brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args_part = &inner[brace + 1..];
    // Take everything up to the last `}` as the argument body.
    let args = args_part.rfind('}').map(|e| &args_part[..e]).unwrap_or(args_part);
    let arguments = gemma_args_to_json(args);
    Some(ToolCall::new(format!("call_{}", short_id(idx)), name, arguments))
}

/// Convert gemma's non-JSON `key:value,key:value` argument body into a
/// JSON object string (OpenAI `arguments` convention).
fn gemma_args_to_json(args: &str) -> String {
    let mut obj = serde_json::Map::new();
    for pair in split_top_level_commas(args) {
        if let Some((k, v)) = pair.split_once(':') {
            let key = k.trim().trim_matches('"').to_string();
            if key.is_empty() {
                continue;
            }
            obj.insert(key, parse_gemma_value(v.trim()));
        }
    }
    serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_else(|_| "{}".to_string())
}

fn parse_gemma_value(v: &str) -> serde_json::Value {
    let v = v.trim();
    if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') {
        return serde_json::Value::String(v[1..v.len() - 1].to_string());
    }
    match v {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" => return serde_json::Value::Null,
        _ => {}
    }
    if let Ok(n) = v.parse::<i64>() {
        return serde_json::Value::from(n);
    }
    if let Ok(f) = v.parse::<f64>() {
        return serde_json::json!(f);
    }
    // Try as embedded JSON (nested object/array); else raw string.
    if let Ok(j) = serde_json::from_str::<serde_json::Value>(v) {
        return j;
    }
    serde_json::Value::String(v.to_string())
}

/// Split on top-level commas, ignoring commas inside quotes or nested
/// `{}` / `[]`.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_q = false;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '"' => {
                in_q = !in_q;
                cur.push(c);
            }
            '{' | '[' if !in_q => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' if !in_q => {
                depth -= 1;
                cur.push(c);
            }
            ',' if !in_q && depth == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
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

    // ---- Gemma permissive parser (gemma-native + Qwen fallback) ----

    #[test]
    fn gemma_native_single_call() {
        let (content, calls) = parse_tool_calls_gemma(
            "<|tool_call>call:get_weather{location: \"Tokyo\"}<tool_call|>",
        );
        assert!(content.is_none());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"Tokyo\"}");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn gemma_native_two_args_and_types() {
        let (_c, calls) = parse_tool_calls_gemma(
            "<|tool_call>call:search_files{directory: \"src\", pattern: \"*.rs\"}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_files");
        let v: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(v["directory"], "src");
        assert_eq!(v["pattern"], "*.rs");
    }

    #[test]
    fn gemma_numeric_and_bool_args() {
        let (_c, calls) = parse_tool_calls_gemma("<|tool_call>call:f{n: 42, flag: true}<tool_call|>");
        let v: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(v["n"], 42);
        assert_eq!(v["flag"], true);
    }

    #[test]
    fn gemma_qwen_fallback_still_parses() {
        // gemma-Q3 follows the in-prompt Qwen instruction → must still parse.
        let (content, calls) = parse_tool_calls_gemma(
            "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call>",
        );
        assert!(content.is_none());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"Paris\"}");
    }

    #[test]
    fn gemma_text_plus_call_splits() {
        let (content, calls) =
            parse_tool_calls_gemma("Let me check.\n<|tool_call>call:f{}<tool_call|>");
        assert_eq!(content.as_deref(), Some("Let me check."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn gemma_unterminated_does_not_panic() {
        // Runaway cut mid-call (cancel/max_tokens) — salvage name + args.
        let (_c, calls) = parse_tool_calls_gemma("<|tool_call>call:do_thing{x: 1}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "do_thing");
    }

    #[test]
    fn gemma_no_call_is_plain_text() {
        let (content, calls) = parse_tool_calls_gemma("The capital of France is Paris.");
        assert_eq!(content.as_deref(), Some("The capital of France is Paris."));
        assert!(calls.is_empty());
    }

    #[test]
    fn gemma_runaway_tail_ignored_after_first_call() {
        // QAT runaway: a valid call followed by repeated <|tool_response>.
        let (_c, calls) = parse_tool_calls_gemma(
            "<|tool_call>call:get_weather{location: \"Tokyo\"}<tool_call|><|tool_response><|tool_response>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }
}
