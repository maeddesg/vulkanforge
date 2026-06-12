// SPDX-License-Identifier: GPL-3.0-only
//! Slice-1 agent loop (opt-in via `--agent`).
//!
//! The thinnest runnable loop: one tool (`read_file`), a permission gate,
//! a full `tool_call` roundtrip (model → tool → result back → continue),
//! and a loop cap. Default coder = Qwen3-14B-Q4 (JSON tool args). gemma's
//! `{k:v}` format is out of scope for this slice.

use std::io::Write;

use crate::client::{AgentTurn, Client, Result};
use crate::tools::{execute_read_file, read_file_tool, READ_FILE};
use crate::types::{ChatMessage, ToolCall};

/// Maximum tool-call iterations before stopping with a visible marker.
pub const LOOP_CAP: usize = 8;

/// How a tool call gets approved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// `--yes`: auto-approve (acceptable for the read-only `read_file`).
    AutoApprove,
    /// TTY / REPL: interactive `y/N` per call.
    Interactive,
    /// Headless without `--yes`: deny, report, end the loop gracefully.
    DenyHeadless,
}

/// How the loop ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopEnd {
    /// The model produced a final answer (no more tool calls).
    Final { content: Option<String>, finish_reason: Option<String> },
    /// The loop cap was hit (the task may be incomplete).
    CapReached,
}

/// Decide a single tool call. `Interactive` prompts on the TTY; the other
/// two are pure. `true` = run the tool.
pub fn approve(mode: Permission, name: &str, args: &str) -> bool {
    match mode {
        Permission::AutoApprove => {
            eprintln!("[agent] auto-approved (--yes): {name}({args})");
            true
        }
        Permission::DenyHeadless => {
            eprintln!("[agent] tool call denied (headless without --yes): {name}({args})");
            false
        }
        Permission::Interactive => prompt_yn(name, args),
    }
}

fn prompt_yn(name: &str, args: &str) -> bool {
    eprint!("[agent] allow {name}({args})? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => false, // EOF / no input → deny (default)
        Ok(_) => matches!(line.trim(), "y" | "Y" | "yes" | "Yes"),
    }
}

/// Permission + dispatch for one tool call → the tool-result string.
/// A denied call returns a denial result (so the loop continues and the
/// model can react), without executing the tool.
pub(crate) fn handle_call(call: &ToolCall, mode: Permission) -> String {
    let name = call.function.name.as_str();
    let args = call.function.arguments.as_str();
    if !approve(mode, name, args) {
        return format!("Tool call \"{name}\" was denied by the user; it was not executed.");
    }
    match name {
        READ_FILE => execute_read_file(args),
        other => format!("Tool error: unknown tool \"{other}\" (only read_file is available in this build)."),
    }
}

/// The core loop, generic over the turn source so it is unit-testable
/// without a network. `send(messages)` yields the model's next turn (it
/// must include the tool definitions). Appends the OpenAI roundtrip
/// messages — one assistant turn carrying all calls, then one `tool`
/// result per call — and re-sends, until the model stops calling tools or
/// the cap is hit.
pub async fn run_loop<F, Fut>(
    mut messages: Vec<ChatMessage>,
    mode: Permission,
    cap: usize,
    mut send: F,
) -> Result<LoopEnd>
where
    F: FnMut(Vec<ChatMessage>) -> Fut,
    Fut: std::future::Future<Output = Result<AgentTurn>>,
{
    for _ in 0..cap {
        let turn = send(messages.clone()).await?;
        if turn.tool_calls.is_empty() {
            return Ok(LoopEnd::Final {
                content: turn.content,
                finish_reason: turn.finish_reason,
            });
        }
        // One assistant turn carrying all the calls...
        let assistant_text = turn.content.clone().unwrap_or_default();
        messages.push(ChatMessage::assistant_with_tool_calls(assistant_text, turn.tool_calls.clone()));
        // ...then one `tool` result per call, in order.
        for call in &turn.tool_calls {
            let result = handle_call(call, mode);
            messages.push(ChatMessage::tool(call.id.clone(), result));
        }
    }
    Ok(LoopEnd::CapReached)
}

/// Run the agent loop against a live server with the Slice-1 tool set
/// (`read_file`). Returns how it ended; the caller renders the result.
pub async fn run(client: &Client, messages: Vec<ChatMessage>, mode: Permission) -> Result<LoopEnd> {
    let tools = [read_file_tool()];
    run_loop(messages, mode, LOOP_CAP, |msgs| client.chat_with_tools(msgs, &tools)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCallFunction;
    use std::io::Write;

    fn read_call(path: &str) -> ToolCall {
        ToolCall {
            id: "call_x".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "read_file".into(),
                arguments: format!("{{\"path\":\"{path}\"}}"),
            },
        }
    }

    fn tmp(name: &str, body: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vf_clide_agent_{name}"));
        std::fs::File::create(&p).unwrap().write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn approve_modes_are_pure_for_auto_and_deny() {
        assert!(approve(Permission::AutoApprove, "read_file", "{}"));
        assert!(!approve(Permission::DenyHeadless, "read_file", "{}"));
    }

    #[test]
    fn denied_call_is_not_executed() {
        // The file exists with a sentinel; a denied call must NOT read it.
        let p = tmp("deny", "SENTINEL-SECRET");
        let call = read_call(&p.display().to_string());
        let out = handle_call(&call, Permission::DenyHeadless);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!out.contains("SENTINEL-SECRET"), "tool ran despite denial: {out}");
        // ...and an approved call DOES read it.
        let out2 = handle_call(&call, Permission::AutoApprove);
        assert!(out2.contains("SENTINEL-SECRET"), "approved call should read: {out2}");
    }

    #[test]
    fn unknown_tool_is_a_structured_error() {
        let mut call = read_call("/tmp/x");
        call.function.name = "delete_everything".into();
        let out = handle_call(&call, Permission::AutoApprove);
        assert!(out.contains("unknown tool"), "got: {out}");
    }

    #[tokio::test]
    async fn loop_finishes_when_no_tool_calls() {
        // send returns a plain text answer → Final immediately, one send.
        let mut count = 0;
        let end = run_loop(vec![ChatMessage::user("hi")], Permission::AutoApprove, LOOP_CAP, |_m| {
            count += 1;
            std::future::ready(Ok(AgentTurn {
                content: Some("done".into()),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
            }))
        })
        .await
        .unwrap();
        assert_eq!(end, LoopEnd::Final { content: Some("done".into()), finish_reason: Some("stop".into()) });
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn loop_stops_at_cap_when_model_never_stops() {
        // send ALWAYS asks for a tool → the loop must stop at the cap.
        let p = tmp("cap", "x");
        let call = read_call(&p.display().to_string());
        let mut count = 0;
        let end = run_loop(vec![ChatMessage::user("go")], Permission::AutoApprove, LOOP_CAP, |_m| {
            count += 1;
            let call = call.clone();
            std::future::ready(Ok(AgentTurn {
                content: None,
                tool_calls: vec![call],
                finish_reason: Some("tool_calls".into()),
            }))
        })
        .await
        .unwrap();
        assert_eq!(end, LoopEnd::CapReached);
        assert_eq!(count, LOOP_CAP, "loop must send exactly cap times then stop");
    }

    #[tokio::test]
    async fn history_assembly_keeps_openai_order() {
        // After a tool call, the next send must see: …user, assistant(with
        // tool_calls), tool(result). Capture the messages of the 2nd send.
        let p = tmp("order", "FILE-BODY");
        let call = read_call(&p.display().to_string());
        let mut seen_second: Option<Vec<ChatMessage>> = None;
        let mut iter = 0;
        let _ = run_loop(vec![ChatMessage::user("read it")], Permission::AutoApprove, LOOP_CAP, |msgs| {
            iter += 1;
            if iter == 2 {
                seen_second = Some(msgs);
                std::future::ready(Ok(AgentTurn {
                    content: Some("final".into()),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                }))
            } else {
                std::future::ready(Ok(AgentTurn {
                    content: None,
                    tool_calls: vec![call.clone()],
                    finish_reason: Some("tool_calls".into()),
                }))
            }
        })
        .await
        .unwrap();
        let m = seen_second.expect("a second send must happen after the tool call");
        assert_eq!(m[0].role, "user");
        assert_eq!(m[1].role, "assistant");
        assert!(m[1].tool_calls.is_some(), "assistant turn must replay tool_calls");
        assert_eq!(m[2].role, "tool");
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_x"));
        assert_eq!(m[2].content, "FILE-BODY");
    }
}
