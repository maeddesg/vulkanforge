// SPDX-License-Identifier: GPL-3.0-only
//! Agent loop (opt-in via `--agent`).
//!
//! A runnable loop with a **risk-aware permission gate** and a
//! **workspace-confined** tool set: `read_file`/`search` (read-only) and
//! `write_file`/`shell` (mutating). The full `tool_call` roundtrip (model
//! → gate → tool → result back → continue) plus a loop cap. Default coder
//! = Qwen3-14B-Q4 (JSON tool args). gemma's `{k:v}` format is out of scope.
//!
//! Safety ordering (Slice 2): the gate (risk classes + `--allow-mutating`)
//! and the workspace confinement exist **before** the mutating tools, so
//! `write`/`shell` are never reachable ungated.

use std::io::Write;
use std::path::Path;

use crate::client::{AgentTurn, Client, Result};
use crate::tools::{
    all_tools, execute_read_file, execute_search, execute_shell, execute_write_file, READ_FILE,
    SEARCH, SHELL, WRITE_FILE,
};
use crate::types::{ChatMessage, Tool, ToolCall, ToolRisk};

/// Maximum tool-call iterations before stopping with a visible marker.
pub const LOOP_CAP: usize = 8;

/// How a tool call gets approved (interaction style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// `--yes`: auto-approve **read-only** tools. Mutating tools still
    /// require `--allow-mutating` (see [`Gate`]).
    AutoApprove,
    /// TTY / REPL: interactive `y/N` per call (mutating tools get a
    /// visible warning).
    Interactive,
    /// Headless without `--yes`: read-only tools denied; mutating tools
    /// allowed only with `--allow-mutating`.
    DenyHeadless,
}

/// The full permission gate: interaction [`Permission`] mode **plus** the
/// explicit `--allow-mutating` headless opt-in. The core rule: `--yes`
/// approves safe reads, **never** waves a mutating tool (`write`/`shell`)
/// through — that needs an interactive yes or `--allow-mutating`.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub mode: Permission,
    pub allow_mutating: bool,
}

impl Gate {
    pub fn new(mode: Permission, allow_mutating: bool) -> Self {
        Self { mode, allow_mutating }
    }
}

/// How the loop ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopEnd {
    /// The model produced a final answer (no more tool calls).
    Final { content: Option<String>, finish_reason: Option<String> },
    /// The loop cap was hit (the task may be incomplete).
    CapReached,
}

/// Decide a single tool call given its **risk class** (from the tool
/// definition). `true` = run the tool.
///
/// - `AutoApprove` (`--yes`): auto-approves `ReadOnly`; `Mutating` only if
///   `--allow-mutating`, else denied.
/// - `DenyHeadless`: `ReadOnly` denied (no `--yes`); `Mutating` only if
///   `--allow-mutating`.
/// - `Interactive`: prompts `y/N`; `Mutating` shows a visible warning.
pub fn approve(gate: Gate, name: &str, risk: ToolRisk, args: &str) -> bool {
    if gate.mode == Permission::Interactive {
        return prompt_yn(name, risk, args);
    }
    let yes = gate.mode == Permission::AutoApprove;
    match risk {
        ToolRisk::ReadOnly => {
            if yes {
                eprintln!("[agent] auto-approved (--yes, read-only): {name}({args})");
                true
            } else {
                eprintln!("[agent] denied (headless without --yes): {name}({args})");
                false
            }
        }
        ToolRisk::Mutating => {
            if gate.allow_mutating {
                eprintln!("[agent] auto-approved (--allow-mutating, MUTATING): {name}({args})");
                true
            } else {
                eprintln!(
                    "[agent] DENIED — \"{name}\" is a MUTATING tool; --yes does NOT auto-approve it. \
                     Pass --allow-mutating to permit mutating tools headless: {name}({args})"
                );
                false
            }
        }
    }
}

fn prompt_yn(name: &str, risk: ToolRisk, args: &str) -> bool {
    if risk == ToolRisk::Mutating {
        eprintln!("[agent] ⚠  MUTATING tool — this writes to disk / executes a command.");
    }
    eprint!("[agent] allow {name}({args})? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => false, // EOF / no input → deny (default)
        Ok(_) => matches!(line.trim(), "y" | "Y" | "yes" | "Yes"),
    }
}

/// Permission + dispatch for one tool call → the tool-result string.
/// The risk class is read from the tool **definition** (`tools`), so the
/// gate can't drift from the schema. A denied call returns a denial result
/// (loop continues, model can react) without executing the tool. All file
/// tools receive `workspace` for confinement.
pub(crate) fn handle_call(call: &ToolCall, gate: Gate, workspace: &Path, tools: &[Tool]) -> String {
    let name = call.function.name.as_str();
    let args = call.function.arguments.as_str();
    // Risk from the definition; an unknown tool has no executor → error
    // before gating (nothing to run, nothing to prompt for).
    let Some(def) = tools.iter().find(|t| t.function.name == name) else {
        return format!(
            "Tool error: unknown tool \"{name}\" (available: read_file, write_file, search, shell)."
        );
    };
    if !approve(gate, name, def.risk, args) {
        return format!("Tool call \"{name}\" was denied; it was not executed.");
    }
    match name {
        READ_FILE => execute_read_file(args, workspace),
        WRITE_FILE => execute_write_file(args, workspace),
        SEARCH => execute_search(args, workspace),
        SHELL => execute_shell(args, workspace),
        other => format!("Tool error: unknown tool \"{other}\"."),
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
    gate: Gate,
    workspace: &Path,
    tools: &[Tool],
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
            let result = handle_call(call, gate, workspace, tools);
            messages.push(ChatMessage::tool(call.id.clone(), result));
        }
    }
    Ok(LoopEnd::CapReached)
}

/// Run the agent loop against a live server with the full Slice-2 tool set
/// (`read_file`, `write_file`, `search`, `shell`). Returns how it ended;
/// the caller renders the result.
pub async fn run(
    client: &Client,
    messages: Vec<ChatMessage>,
    gate: Gate,
    workspace: &Path,
) -> Result<LoopEnd> {
    let tools = all_tools();
    run_loop(messages, gate, workspace, &tools, LOOP_CAP, |msgs| {
        client.chat_with_tools(msgs, &tools)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::read_file_tool;
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

    fn ws() -> std::path::PathBuf {
        std::env::temp_dir().canonicalize().unwrap()
    }

    #[test]
    fn yes_approves_readonly_not_mutating() {
        let auto = Gate::new(Permission::AutoApprove, false);
        // --yes auto-approves a read-only tool...
        assert!(approve(auto, "read_file", ToolRisk::ReadOnly, "{}"));
        // ...but NOT a mutating one (the core rule).
        assert!(!approve(auto, "shell", ToolRisk::Mutating, "{}"));
        // --allow-mutating flips the mutating case on.
        let allow = Gate::new(Permission::AutoApprove, true);
        assert!(approve(allow, "shell", ToolRisk::Mutating, "{}"));
        assert!(approve(allow, "read_file", ToolRisk::ReadOnly, "{}"));
    }

    #[test]
    fn deny_headless_denies_readonly_too() {
        let deny = Gate::new(Permission::DenyHeadless, false);
        assert!(!approve(deny, "read_file", ToolRisk::ReadOnly, "{}"));
        assert!(!approve(deny, "write_file", ToolRisk::Mutating, "{}"));
        // ...unless --allow-mutating is set (gates mutating independently).
        let deny_allow = Gate::new(Permission::DenyHeadless, true);
        assert!(approve(deny_allow, "write_file", ToolRisk::Mutating, "{}"));
    }

    #[test]
    fn mutating_denied_call_is_not_executed() {
        // A write that would clobber a sentinel must NOT run when denied.
        let dir = ws();
        let target = dir.join("vf_clide_should_not_exist.txt");
        let _ = std::fs::remove_file(&target);
        let call = ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "write_file".into(),
                arguments: format!("{{\"path\":\"{}\",\"content\":\"X\"}}", target.display()),
            },
        };
        let tools = all_tools();
        // --yes alone must NOT execute the mutating write.
        let out = handle_call(&call, Gate::new(Permission::AutoApprove, false), &dir, &tools);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!target.exists(), "write ran despite --yes-without-allow-mutating: {out}");
    }

    #[test]
    fn readonly_denied_call_is_not_executed() {
        let p = tmp("deny", "SENTINEL-SECRET");
        let dir = ws();
        let call = read_call(&p.display().to_string());
        let tools = all_tools();
        let out = handle_call(&call, Gate::new(Permission::DenyHeadless, false), &dir, &tools);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!out.contains("SENTINEL-SECRET"), "tool ran despite denial: {out}");
        // ...and an approved (--yes, read-only) call DOES read it.
        let out2 = handle_call(&call, Gate::new(Permission::AutoApprove, false), &dir, &tools);
        assert!(out2.contains("SENTINEL-SECRET"), "approved read should run: {out2}");
    }

    #[test]
    fn unknown_tool_is_a_structured_error() {
        let mut call = read_call("/tmp/x");
        call.function.name = "delete_everything".into();
        let tools = all_tools();
        let out = handle_call(&call, Gate::new(Permission::AutoApprove, true), &ws(), &tools);
        assert!(out.contains("unknown tool"), "got: {out}");
    }

    #[tokio::test]
    async fn loop_finishes_when_no_tool_calls() {
        let mut count = 0;
        let tools = all_tools();
        let end = run_loop(
            vec![ChatMessage::user("hi")],
            Gate::new(Permission::AutoApprove, false),
            &ws(),
            &tools,
            LOOP_CAP,
            |_m| {
                count += 1;
                std::future::ready(Ok(AgentTurn {
                    content: Some("done".into()),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                }))
            },
        )
        .await
        .unwrap();
        assert_eq!(end, LoopEnd::Final { content: Some("done".into()), finish_reason: Some("stop".into()) });
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn loop_stops_at_cap_when_model_never_stops() {
        let p = tmp("cap", "x");
        let call = read_call(&p.display().to_string());
        let tools = all_tools();
        let mut count = 0;
        let end = run_loop(
            vec![ChatMessage::user("go")],
            Gate::new(Permission::AutoApprove, false),
            &ws(),
            &tools,
            LOOP_CAP,
            |_m| {
                count += 1;
                let call = call.clone();
                std::future::ready(Ok(AgentTurn {
                    content: None,
                    tool_calls: vec![call],
                    finish_reason: Some("tool_calls".into()),
                }))
            },
        )
        .await
        .unwrap();
        assert_eq!(end, LoopEnd::CapReached);
        assert_eq!(count, LOOP_CAP, "loop must send exactly cap times then stop");
    }

    #[tokio::test]
    async fn history_assembly_keeps_openai_order() {
        let p = tmp("order", "FILE-BODY");
        let call = read_call(&p.display().to_string());
        let tools = all_tools();
        let mut seen_second: Option<Vec<ChatMessage>> = None;
        let mut iter = 0;
        let _ = run_loop(
            vec![ChatMessage::user("read it")],
            Gate::new(Permission::AutoApprove, false),
            &ws(),
            &tools,
            LOOP_CAP,
            |msgs| {
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
            },
        )
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
