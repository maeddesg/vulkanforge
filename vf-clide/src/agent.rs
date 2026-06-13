// SPDX-License-Identifier: GPL-3.0-only
//! Agent loop (opt-in via `--agent`).
//!
//! A runnable loop with a **tiered, risk-aware permission gate** and a
//! **workspace-confined** tool set: `read_file`/`search` (read-only),
//! `write_file` (mutating, confined) and `shell` (exec, NOT confinable).
//! The full `tool_call` roundtrip (model → gate → tool → result back →
//! continue) plus a loop cap. Default coder = Qwen3-14B-Q4 (JSON tool
//! args). gemma's `{k:v}` format is out of scope.
//!
//! Safety ordering: the gate (risk tiers + cumulative opt-in flags) and
//! the workspace confinement exist **before** the mutating/exec tools, so
//! `write`/`shell` are never reachable ungated. `shell` is the only
//! non-confinable tool, so it gets its own top tier (`Exec`) — its sole
//! guard is the gate (`--allow-shell`), never the workspace root.

use std::io::Write;
use std::path::Path;

use crate::client::{AgentTurn, Client, Result};
use crate::tools::{
    all_tools, execute_read_file, execute_search, execute_shell, execute_write_file, read_agents_md,
    READ_FILE, SEARCH, SHELL, WRITE_FILE,
};
use crate::types::{ChatMessage, Tool, ToolCall, ToolRisk};

/// Maximum tool-call iterations before stopping with a visible marker.
pub const LOOP_CAP: usize = 8;

/// Built-in operating instructions — the agent's "constitution". Kept
/// short on purpose; appended with the workspace `AGENTS.md` when present.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a coding assistant working inside a project workspace through tools. Use \
the tools (read_file, search, write_file, shell) to inspect and change the \
workspace rather than guessing; prefer reading or searching before writing, and \
make minimal, targeted changes. Keep your responses concise.\n\n\
Respect the permission model: some tools need explicit user permission and a call \
may be denied. If a tool call is denied, do NOT claim you performed the action — \
briefly state what you intended to do, that it was denied, and how to proceed \
(e.g. re-run granting the needed permission). Never invent file contents or \
command output you did not actually receive from a tool.";

/// How a tool call gets approved (interaction style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// REPL / TTY: prompt `y/N` per call (riskier tiers get louder warnings).
    Interactive,
    /// Headless: auto-approve a call iff its risk tier ≤ the gate ceiling.
    Headless,
}

/// The permission gate: interaction [`Permission`] mode + (headless only)
/// the auto-approve **ceiling** — the highest [`ToolRisk`] tier approved
/// without a prompt. Opt-in flags raise the ceiling **cumulatively**, each
/// implying the lower tiers:
/// `--yes` → `ReadOnly`, `--allow-mutating` → `Mutating`,
/// `--allow-shell` → `Exec`. So `--allow-shell` also auto-approves
/// reads+writes, and `--allow-mutating` also auto-approves reads (this
/// removes the Slice-2 "mutating-yes / reads-no" inconsistency). `None`
/// ceiling = deny everything headless.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub mode: Permission,
    pub auto_ceiling: Option<ToolRisk>,
}

impl Gate {
    /// Interactive (REPL) gate — always prompts; ceiling unused.
    pub fn interactive() -> Self {
        Self { mode: Permission::Interactive, auto_ceiling: None }
    }
    /// Headless gate auto-approving up to `ceiling` (None = deny all).
    pub fn headless(ceiling: Option<ToolRisk>) -> Self {
        Self { mode: Permission::Headless, auto_ceiling: ceiling }
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

/// Decide a single tool call given its **risk tier** (from the tool
/// definition). `true` = run the tool. Headless approves iff
/// `risk ≤ ceiling`; Interactive prompts (louder warning for higher tiers).
pub fn approve(gate: Gate, name: &str, risk: ToolRisk, args: &str) -> bool {
    match gate.mode {
        Permission::Interactive => prompt_yn(name, risk, args),
        Permission::Headless => {
            let approved = gate.auto_ceiling.is_some_and(|c| risk.rank() <= c.rank());
            if approved {
                eprintln!("[agent] auto-approved ({}): {name}({args})", tier_label(risk));
                true
            } else {
                eprintln!(
                    "[agent] DENIED — \"{name}\" is {} and needs {} (headless); not executed: {name}({args})",
                    tier_label(risk),
                    needed_flag(risk)
                );
                false
            }
        }
    }
}

fn tier_label(risk: ToolRisk) -> &'static str {
    match risk {
        ToolRisk::ReadOnly => "read-only",
        ToolRisk::Mutating => "MUTATING",
        ToolRisk::Exec => "EXEC/shell",
    }
}

fn needed_flag(risk: ToolRisk) -> &'static str {
    match risk {
        ToolRisk::ReadOnly => "--yes",
        ToolRisk::Mutating => "--allow-mutating",
        ToolRisk::Exec => "--allow-shell",
    }
}

fn prompt_yn(name: &str, risk: ToolRisk, args: &str) -> bool {
    match risk {
        ToolRisk::Exec => {
            eprintln!("[agent] ⚠⚠ EXEC — runs a shell command (NOT confined to the workspace).")
        }
        ToolRisk::Mutating => {
            eprintln!("[agent] ⚠  MUTATING — writes to disk (within the workspace).")
        }
        ToolRisk::ReadOnly => {}
    }
    eprint!("[agent] allow {name}({args})? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => false, // EOF / no input → deny (default)
        Ok(_) => matches!(line.trim(), "y" | "Y" | "yes" | "Yes"),
    }
}

/// Build the agent's system prompt (constitution).
/// - `no_system` → `None` (no system message).
/// - `system_file` → that file's contents verbatim (a user-chosen CLI
///   path, read directly), replacing the default.
/// - otherwise → the built-in default, with the workspace `AGENTS.md`
///   (read **confined**, consistent with the file tools) appended when
///   present. A missing / escaping / unreadable `AGENTS.md` is simply
///   skipped (no crash).
pub fn system_prompt(
    workspace: &Path,
    system_file: Option<&str>,
    no_system: bool,
) -> Result<Option<String>> {
    if no_system {
        return Ok(None);
    }
    if let Some(path) = system_file {
        let content = std::fs::read_to_string(path).map_err(|e| format!("--system {path}: {e}"))?;
        return Ok(Some(content));
    }
    let mut prompt = DEFAULT_SYSTEM_PROMPT.to_string();
    if let Some(agents) = read_agents_md(workspace) {
        let trimmed = agents.trim();
        if !trimmed.is_empty() {
            prompt.push_str("\n\n# Project instructions (from AGENTS.md)\n");
            prompt.push_str(trimmed);
        }
    }
    Ok(Some(prompt))
}

/// Permission + dispatch for one tool call → the tool-result string.
/// The risk tier is read from the tool **definition** (`tools`), so the
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

/// Run the agent loop against a live server with the full tool set
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

    fn named_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: name.into(), arguments: args.into() },
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

    // ---- tiered ceiling gate ----

    #[test]
    fn ceiling_gates_by_tier_with_implication() {
        // --yes → ReadOnly ceiling: reads only.
        let yes = Gate::headless(Some(ToolRisk::ReadOnly));
        assert!(approve(yes, "read_file", ToolRisk::ReadOnly, "{}"));
        assert!(!approve(yes, "write_file", ToolRisk::Mutating, "{}"));
        assert!(!approve(yes, "shell", ToolRisk::Exec, "{}"));
        // --allow-mutating → Mutating ceiling: reads + writes, NOT shell.
        let mutating = Gate::headless(Some(ToolRisk::Mutating));
        assert!(approve(mutating, "read_file", ToolRisk::ReadOnly, "{}")); // implication
        assert!(approve(mutating, "write_file", ToolRisk::Mutating, "{}"));
        assert!(!approve(mutating, "shell", ToolRisk::Exec, "{}"));
        // --allow-shell → Exec ceiling: everything (implies lower tiers).
        let shell = Gate::headless(Some(ToolRisk::Exec));
        assert!(approve(shell, "read_file", ToolRisk::ReadOnly, "{}"));
        assert!(approve(shell, "write_file", ToolRisk::Mutating, "{}"));
        assert!(approve(shell, "shell", ToolRisk::Exec, "{}"));
        // no flag → deny all.
        let none = Gate::headless(None);
        assert!(!approve(none, "read_file", ToolRisk::ReadOnly, "{}"));
        assert!(!approve(none, "shell", ToolRisk::Exec, "{}"));
    }

    #[test]
    fn shell_not_executed_under_mutating_ceiling() {
        // --allow-mutating must NOT run shell (Exec tier). Prove with a
        // sentinel side effect that must NOT happen.
        let dir = ws();
        let marker = dir.join("vf_clide_shell_sentinel.txt");
        let _ = std::fs::remove_file(&marker);
        let call = named_call("shell", &format!("{{\"command\":\"touch {}\"}}", marker.display()));
        let tools = all_tools();
        let out = handle_call(&call, Gate::headless(Some(ToolRisk::Mutating)), &dir, &tools);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!marker.exists(), "shell ran under --allow-mutating (should need --allow-shell)");
        // ...and --allow-shell DOES run it.
        let out2 = handle_call(&call, Gate::headless(Some(ToolRisk::Exec)), &dir, &tools);
        assert!(out2.contains("exit_code=0"), "shell should run with --allow-shell: {out2}");
        assert!(marker.exists(), "shell did not run under --allow-shell");
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn write_not_executed_under_readonly_ceiling() {
        let dir = ws();
        let target = dir.join("vf_clide_write_sentinel.txt");
        let _ = std::fs::remove_file(&target);
        let call = named_call(
            "write_file",
            &format!("{{\"path\":\"{}\",\"content\":\"X\"}}", target.display()),
        );
        let tools = all_tools();
        let out = handle_call(&call, Gate::headless(Some(ToolRisk::ReadOnly)), &dir, &tools);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!target.exists(), "write ran under --yes (should need --allow-mutating)");
    }

    #[test]
    fn readonly_denied_when_no_ceiling() {
        let p = tmp("deny", "SENTINEL-SECRET");
        let dir = ws();
        let call = read_call(&p.display().to_string());
        let tools = all_tools();
        let out = handle_call(&call, Gate::headless(None), &dir, &tools);
        assert!(out.contains("denied"), "got: {out}");
        assert!(!out.contains("SENTINEL-SECRET"), "tool ran despite denial: {out}");
        let out2 = handle_call(&call, Gate::headless(Some(ToolRisk::ReadOnly)), &dir, &tools);
        assert!(out2.contains("SENTINEL-SECRET"), "approved read should run: {out2}");
    }

    #[test]
    fn unknown_tool_is_a_structured_error() {
        let call = named_call("delete_everything", "{}");
        let tools = all_tools();
        let out = handle_call(&call, Gate::headless(Some(ToolRisk::Exec)), &ws(), &tools);
        assert!(out.contains("unknown tool"), "got: {out}");
    }

    // ---- constitution / system prompt ----

    fn fresh_dir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vf_clide_sys_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p.canonicalize().unwrap()
    }

    #[test]
    fn default_system_prompt_present() {
        let dir = fresh_dir("default");
        let sp = system_prompt(&dir, None, false).unwrap().unwrap();
        assert!(sp.contains("coding assistant"));
        assert!(!sp.contains("AGENTS.md"), "no AGENTS.md present → no append");
    }

    #[test]
    fn agents_md_is_appended_when_present() {
        let dir = fresh_dir("agents");
        std::fs::write(dir.join("AGENTS.md"), b"Always use tabs. Run `cargo test`.").unwrap();
        let sp = system_prompt(&dir, None, false).unwrap().unwrap();
        assert!(sp.contains("coding assistant"), "default kept");
        assert!(sp.contains("Always use tabs"), "AGENTS.md appended");
        assert!(sp.contains("from AGENTS.md"), "append header present");
    }

    #[test]
    fn system_override_replaces_default() {
        let dir = fresh_dir("override");
        let f = dir.join("custom.txt");
        std::fs::write(&f, b"You are a terse bot.").unwrap();
        let sp = system_prompt(&dir, Some(&f.display().to_string()), false).unwrap().unwrap();
        assert_eq!(sp, "You are a terse bot.");
        assert!(!sp.contains("coding assistant"), "default replaced, not appended");
    }

    #[test]
    fn no_system_yields_none() {
        let dir = fresh_dir("none");
        assert!(system_prompt(&dir, None, true).unwrap().is_none());
    }

    #[test]
    fn missing_agents_md_does_not_crash() {
        let dir = fresh_dir("missing");
        // No AGENTS.md → just the default, no error.
        let sp = system_prompt(&dir, None, false).unwrap().unwrap();
        assert!(sp.contains("coding assistant"));
    }

    // ---- loop (unregressed; ReadOnly ceiling) ----

    #[tokio::test]
    async fn loop_finishes_when_no_tool_calls() {
        let mut count = 0;
        let tools = all_tools();
        let end = run_loop(
            vec![ChatMessage::user("hi")],
            Gate::headless(Some(ToolRisk::ReadOnly)),
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
            Gate::headless(Some(ToolRisk::ReadOnly)),
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
            Gate::headless(Some(ToolRisk::ReadOnly)),
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
