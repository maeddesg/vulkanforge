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

use serde::Deserialize;

use crate::client::{AgentTurn, Client, MemCall, Result};
use crate::tools::{
    all_tools, execute_read_file, execute_search, execute_shell, execute_write_file, memory_tools,
    read_agents_md, READ_FILE, RECALL, REMEMBER, SEARCH, SHELL, WRITE_FILE,
};
use crate::status::StatusBar;
use crate::types::{ChatMessage, MemoryHit, Tool, ToolCall, ToolRisk, Usage};

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
briefly state what you intended to do, that it was denied, and how to proceed. \
There are two distinct kinds of denial, resolved differently:\n\
- Permission denial (a tool above the current approval level, e.g. write_file or \
shell): lifted only by the user re-running vf-clide with the matching flag \
(--allow-mutating for write_file, --allow-shell for shell). Suggest that — never \
ask for operating-system or filesystem permissions.\n\
- Workspace-confinement denial (a path outside the project workspace): absolute. \
No flag overrides it; do not request elevated permissions or call the target \
system-critical — work within the workspace instead.\n\n\
If memory tools (recall, remember) are available, use them judiciously: recall \
relevant prior context before significant work, and remember durable decisions, \
learnings, or bugs worth keeping across sessions — not routine actions.\n\n\
Never invent file contents or command output you did not actually receive from a tool.";

/// How a tool call gets approved (interaction style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// REPL / TTY: auto-approve calls at or below the gate ceiling, prompt
    /// `y/N` for calls *above* it (riskier tiers get louder warnings).
    Interactive,
    /// Headless: auto-approve a call iff its risk tier ≤ the gate ceiling,
    /// otherwise deny (no prompt — the flags are the only opt-in).
    Headless,
}

/// The permission gate: interaction [`Permission`] mode + the auto-approve
/// **ceiling** — the highest [`ToolRisk`] tier approved without a prompt.
/// The ceiling applies in **both** modes: a call at or below it is
/// auto-approved (and still printed, so the human sees every tool); a call
/// above it is **prompted** in the REPL and **denied** headless. Opt-in
/// flags raise the ceiling **cumulatively**, each implying the lower tiers:
/// `--yes` → `ReadOnly`, `--allow-mutating` → `Mutating`,
/// `--allow-shell` → `Exec`. So `--allow-shell` also auto-approves
/// reads+writes, and `--allow-mutating` also auto-approves reads (this
/// removes the Slice-2 "mutating-yes / reads-no" inconsistency). `None`
/// ceiling = prompt-everything (REPL) / deny-everything (headless).
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub mode: Permission,
    pub auto_ceiling: Option<ToolRisk>,
}

impl Gate {
    /// Interactive (REPL) gate: auto-approve up to `ceiling`, prompt above
    /// it. The flag ceiling is honored here too — consistent with headless,
    /// not laxer (confinement still bounds read/write independently).
    pub fn interactive(ceiling: Option<ToolRisk>) -> Self {
        Self { mode: Permission::Interactive, auto_ceiling: ceiling }
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

/// What to do with a tool call, independent of any I/O: run it without
/// asking (`Auto`), ask the user (`Prompt`), or refuse it (`Deny`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Auto,
    Prompt,
    Deny,
}

/// Pure decision from mode + ceiling + the call's risk tier — no stdin, no
/// I/O, so the whole truth table is unit-testable. A call at or below the
/// ceiling is `Auto` in **both** modes (the human still sees it via the
/// printed auto-approve line, but only confirms calls *above* the ceiling).
/// Above the ceiling: the REPL prompts, headless denies (the cumulative
/// `--yes`/`--allow-*` flags are the only headless opt-in). Headless never
/// returns `Prompt`; Interactive never returns `Deny`.
pub fn decide(mode: Permission, ceiling: Option<ToolRisk>, risk: ToolRisk) -> Decision {
    let at_or_below_ceiling = ceiling.is_some_and(|c| risk.rank() <= c.rank());
    if at_or_below_ceiling {
        Decision::Auto
    } else {
        match mode {
            Permission::Interactive => Decision::Prompt,
            Permission::Headless => Decision::Deny,
        }
    }
}

/// Resolve a single tool call to run/skip and perform the side effects
/// (print the auto / deny line, or prompt y/N). `true` = run the tool. The
/// decision itself is the pure [`decide`]; this only wraps it with I/O. The
/// risk tier comes from the tool definition. Auto-approved calls print in
/// **both** modes so the human always sees every tool that ran.
pub fn approve(gate: Gate, name: &str, risk: ToolRisk, args: &str) -> bool {
    match decide(gate.mode, gate.auto_ceiling, risk) {
        Decision::Auto => {
            eprintln!("[agent] auto-approved ({}): {name}({args})", tier_label(risk));
            true
        }
        Decision::Prompt => prompt_yn(name, risk, args),
        Decision::Deny => {
            eprintln!(
                "[agent] DENIED — \"{name}\" is {} and needs {} (headless); not executed: {name}({args})",
                tier_label(risk),
                needed_flag(risk)
            );
            false
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
pub async fn run_loop<F, Fut, M, MFut>(
    mut messages: Vec<ChatMessage>,
    gate: Gate,
    workspace: &Path,
    tools: &[Tool],
    status: Option<&StatusBar>,
    cap: usize,
    mut send: F,
    mut mem_call: M,
) -> Result<(LoopEnd, Usage)>
where
    F: FnMut(Vec<ChatMessage>) -> Fut,
    Fut: std::future::Future<Output = Result<AgentTurn>>,
    M: FnMut(ToolCall) -> MFut,
    MFut: std::future::Future<Output = Option<String>>,
{
    // Accumulate token usage across every model call this turn made (the
    // loop may issue several requests; each is billed, so summing them is
    // the honest total work for the user turn).
    let mut acc = Usage::default();
    for _ in 0..cap {
        if let Some(s) = status {
            s.set_action("thinking…");
        }
        let turn = send(messages.clone()).await?;
        fold_usage(&mut acc, &turn.usage);
        if turn.tool_calls.is_empty() {
            return Ok((
                LoopEnd::Final { content: turn.content, finish_reason: turn.finish_reason },
                acc,
            ));
        }
        // One assistant turn carrying all the calls...
        let assistant_text = turn.content.clone().unwrap_or_default();
        messages.push(ChatMessage::assistant_with_tool_calls(assistant_text, turn.tool_calls.clone()));
        // ...then one `tool` result per call, in order.
        for call in &turn.tool_calls {
            if let Some(s) = status {
                s.set_action(format!("running {}(…)", call.function.name));
            }
            // Memory tools (recall/remember) dispatch on their own axis —
            // async, via the HTTP client, BEFORE the file/shell gate (they
            // touch neither files nor the shell). The closure returns
            // Some(result) iff it handled the call; None falls through to the
            // gated file/shell dispatch.
            let result = match mem_call(call.clone()).await {
                Some(r) => r,
                None => handle_call(call, gate, workspace, tools),
            };
            messages.push(ChatMessage::tool(call.id.clone(), result));
        }
    }
    Ok((LoopEnd::CapReached, acc))
}

/// Sum one model call's usage into the running accumulator (Option fields
/// become `Some(running sum)`; `total` falls back to prompt+completion).
fn fold_usage(acc: &mut Usage, u: &Option<Usage>) {
    let Some(u) = u else { return };
    let p = u.prompt_tokens.unwrap_or(0);
    let c = u.completion_tokens.unwrap_or(0);
    let t = u.total_tokens.unwrap_or(p + c);
    acc.prompt_tokens = Some(acc.prompt_tokens.unwrap_or(0) + p);
    acc.completion_tokens = Some(acc.completion_tokens.unwrap_or(0) + c);
    acc.total_tokens = Some(acc.total_tokens.unwrap_or(0) + t);
}

/// Run the agent loop against a live server with the file/shell tool set
/// (`read_file`, `write_file`, `search`, `shell`) plus — when the server
/// reports memory enabled — the memory tools (`recall`, `remember`).
/// `project` is the memory scope (the workspace-derived `project_key`).
/// Returns how it ended; the caller renders the result.
pub async fn run(
    client: &Client,
    messages: Vec<ChatMessage>,
    gate: Gate,
    workspace: &Path,
    project: Option<&str>,
    status: Option<&StatusBar>,
) -> Result<(LoopEnd, Usage)> {
    // Startup probe: offer the memory tools only if the server has memory on.
    // `GET /memory/projects` → 200 = enabled; 503 / transport error = not
    // offered, so the model never makes a doomed memory call.
    let memory_enabled = matches!(client.memory_projects().await, Ok(MemCall::Ok(_)));
    let tools = agent_tools(memory_enabled);
    run_loop(
        messages,
        gate,
        workspace,
        &tools,
        status,
        LOOP_CAP,
        |msgs| client.chat_with_tools(msgs, &tools),
        |call| dispatch_memory(client, project, call),
    )
    .await
}

/// The agent's tool set: the file/shell tools, plus the memory tools when the
/// server reports memory enabled (the startup probe in [`run`]).
pub fn agent_tools(memory_enabled: bool) -> Vec<Tool> {
    let mut tools = all_tools();
    if memory_enabled {
        tools.extend(memory_tools());
    }
    tools
}

/// Whether `name` is a memory tool (handled on the separate async axis, not
/// the file/shell permission gate).
fn is_memory_tool(name: &str) -> bool {
    matches!(name, RECALL | REMEMBER)
}

/// Dispatch a memory tool call via the HTTP client. `Some(result)` for
/// `recall`/`remember` (handled here, **before** the file/shell gate);
/// `None` for any other tool (falls through to the gated dispatch). The names
/// are recognized even if the server turns out memory-off (race): the call
/// then returns a clean "memory not enabled" result, never a crash. Every
/// memory action prints a visible marker (philosophy: results are visible).
async fn dispatch_memory(client: &Client, project: Option<&str>, call: ToolCall) -> Option<String> {
    let name = call.function.name.as_str();
    if !is_memory_tool(name) {
        return None;
    }
    let args = call.function.arguments.as_str();
    Some(match name {
        RECALL => execute_recall(client, project, args).await,
        REMEMBER => execute_remember(client, project, args).await,
        _ => unreachable!("guarded by is_memory_tool"),
    })
}

async fn execute_recall(client: &Client, project: Option<&str>, args: &str) -> String {
    #[derive(Deserialize)]
    struct RecallArgs {
        query: String,
        #[serde(default)]
        k: Option<u32>,
    }
    let parsed: RecallArgs = match serde_json::from_str(args) {
        Ok(a) => a,
        Err(e) => return format!("Tool error: invalid recall arguments: {e}"),
    };
    if parsed.query.trim().is_empty() {
        return "Tool error: recall requires a non-empty query.".to_string();
    }
    let k = parsed.k.unwrap_or(5).clamp(1, 10);
    match client.memory_recall(project, &parsed.query, k).await {
        Ok(MemCall::Ok(resp)) => {
            eprintln!("[agent] recalled {} note(s) for \"{}\"", resp.hits.len(), parsed.query);
            format_recall(&resp.hits)
        }
        Ok(MemCall::Disabled) => {
            eprintln!("[agent] recall: memory is not enabled on this server");
            "Memory is not enabled on this server; no notes are available.".to_string()
        }
        Err(e) => {
            eprintln!("[agent] recall failed: {e}");
            format!("Tool error: recall failed: {e}")
        }
    }
}

async fn execute_remember(client: &Client, project: Option<&str>, args: &str) -> String {
    #[derive(Deserialize)]
    struct RememberArgs {
        #[serde(default)]
        kind: String,
        text: String,
    }
    let parsed: RememberArgs = match serde_json::from_str(args) {
        Ok(a) => a,
        Err(e) => return format!("Tool error: invalid remember arguments: {e}"),
    };
    if parsed.text.trim().is_empty() {
        return "Tool error: remember requires non-empty text.".to_string();
    }
    let kind = if parsed.kind.trim().is_empty() { "Note" } else { parsed.kind.trim() };
    match client.memory_remember(project, kind, &parsed.text).await {
        Ok(MemCall::Ok(resp)) => {
            eprintln!("[agent] remembered [{kind}]: {}", short(&parsed.text));
            format!("Stored as [{kind}] (id {}).", resp.id)
        }
        Ok(MemCall::Disabled) => {
            eprintln!("[agent] remember: memory is not enabled on this server");
            "Memory is not enabled on this server; nothing was stored.".to_string()
        }
        Err(e) => {
            eprintln!("[agent] remember failed: {e}");
            format!("Tool error: remember failed: {e}")
        }
    }
}

/// Format recall hits as a machine-readable tool result — **full text** (the
/// model needs the content, not a snippet); empty → an explicit "no notes" line.
fn format_recall(hits: &[MemoryHit]) -> String {
    if hits.is_empty() {
        return "No relevant notes found in project memory.".to_string();
    }
    let mut out = format!("Found {} note(s) in project memory:\n", hits.len());
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!("[{}] ({}, score {:.2}) {}\n", i + 1, h.kind, h.score, h.text.trim()));
    }
    out
}

/// One-line preview of remembered text for the visible marker (the full text
/// still goes to the store).
fn short(s: &str) -> String {
    let t = s.trim().replace('\n', " ");
    if t.chars().count() > 80 {
        let head: String = t.chars().take(79).collect();
        format!("{head}…")
    } else {
        t
    }
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

    // ---- pure decision truth table (no stdin) ----

    #[test]
    fn decide_repl_honors_ceiling_auto_below_prompt_above() {
        use Decision::*;
        use Permission::Interactive as I;
        use ToolRisk::*;
        // no flags (ceiling None): prompt for every tier.
        assert_eq!(decide(I, None, ReadOnly), Prompt);
        assert_eq!(decide(I, None, Mutating), Prompt);
        assert_eq!(decide(I, None, Exec), Prompt);
        // --yes (ReadOnly): read auto, write/shell prompt.
        assert_eq!(decide(I, Some(ReadOnly), ReadOnly), Auto);
        assert_eq!(decide(I, Some(ReadOnly), Mutating), Prompt);
        assert_eq!(decide(I, Some(ReadOnly), Exec), Prompt);
        // --allow-mutating: read+write auto, shell prompt.
        assert_eq!(decide(I, Some(Mutating), ReadOnly), Auto);
        assert_eq!(decide(I, Some(Mutating), Mutating), Auto);
        assert_eq!(decide(I, Some(Mutating), Exec), Prompt);
        // --allow-shell: everything auto.
        assert_eq!(decide(I, Some(Exec), ReadOnly), Auto);
        assert_eq!(decide(I, Some(Exec), Mutating), Auto);
        assert_eq!(decide(I, Some(Exec), Exec), Auto);
        // REPL never denies.
        for ceil in [None, Some(ReadOnly), Some(Mutating), Some(Exec)] {
            for risk in [ReadOnly, Mutating, Exec] {
                assert_ne!(decide(I, ceil, risk), Deny, "REPL must never deny");
            }
        }
    }

    #[test]
    fn decide_headless_denies_above_ceiling_unchanged() {
        use Decision::*;
        use Permission::Headless as H;
        use ToolRisk::*;
        // auto at/below ceiling, DENY above — same as before this change.
        assert_eq!(decide(H, Some(ReadOnly), ReadOnly), Auto);
        assert_eq!(decide(H, Some(ReadOnly), Mutating), Deny);
        assert_eq!(decide(H, Some(ReadOnly), Exec), Deny);
        assert_eq!(decide(H, Some(Mutating), Mutating), Auto);
        assert_eq!(decide(H, Some(Mutating), Exec), Deny);
        assert_eq!(decide(H, Some(Exec), Exec), Auto);
        assert_eq!(decide(H, None, ReadOnly), Deny); // no flag → deny all
        // headless NEVER prompts.
        for ceil in [None, Some(ReadOnly), Some(Mutating), Some(Exec)] {
            for risk in [ReadOnly, Mutating, Exec] {
                assert_ne!(decide(H, ceil, risk), Prompt, "headless must never prompt");
            }
        }
    }

    #[test]
    fn constitution_distinguishes_denial_kinds() {
        // The wording must name both flags (permission denial) AND tell the
        // model NOT to ask for OS/filesystem rights (confinement denial).
        assert!(DEFAULT_SYSTEM_PROMPT.contains("--allow-mutating"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("--allow-shell"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("Workspace-confinement denial"));
        assert!(DEFAULT_SYSTEM_PROMPT.contains("operating-system or filesystem permissions"));
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
        let (end, _usage) = run_loop(
            vec![ChatMessage::user("hi")],
            Gate::headless(Some(ToolRisk::ReadOnly)),
            &ws(),
            &tools,
            None,
            LOOP_CAP,
            |_m| {
                count += 1;
                std::future::ready(Ok(AgentTurn {
                    content: Some("done".into()),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: None,
                }))
            },
            |_call| std::future::ready(None::<String>),
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
        let (end, _usage) = run_loop(
            vec![ChatMessage::user("go")],
            Gate::headless(Some(ToolRisk::ReadOnly)),
            &ws(),
            &tools,
            None,
            LOOP_CAP,
            |_m| {
                count += 1;
                let call = call.clone();
                std::future::ready(Ok(AgentTurn {
                    content: None,
                    tool_calls: vec![call],
                    finish_reason: Some("tool_calls".into()),
                    usage: None,
                }))
            },
            |_call| std::future::ready(None::<String>),
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
            None,
            LOOP_CAP,
            |msgs| {
                iter += 1;
                if iter == 2 {
                    seen_second = Some(msgs);
                    std::future::ready(Ok(AgentTurn {
                        content: Some("final".into()),
                        tool_calls: vec![],
                        finish_reason: Some("stop".into()),
                        usage: None,
                    }))
                } else {
                    std::future::ready(Ok(AgentTurn {
                        content: None,
                        tool_calls: vec![call.clone()],
                        finish_reason: Some("tool_calls".into()),
                        usage: None,
                    }))
                }
            },
            |_call| std::future::ready(None::<String>),
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

    #[tokio::test]
    async fn loop_accumulates_usage_across_calls() {
        // One tool roundtrip then a final answer; each model call reports
        // usage → the returned turn usage is the sum of all calls.
        let p = tmp("usage", "BODY");
        let call = read_call(&p.display().to_string());
        let tools = all_tools();
        let mut iter = 0;
        let (_end, usage) = run_loop(
            vec![ChatMessage::user("go")],
            Gate::headless(Some(ToolRisk::ReadOnly)),
            &ws(),
            &tools,
            None,
            LOOP_CAP,
            |_m| {
                iter += 1;
                let turn = if iter == 1 {
                    AgentTurn {
                        content: None,
                        tool_calls: vec![call.clone()],
                        finish_reason: Some("tool_calls".into()),
                        usage: Some(Usage { prompt_tokens: Some(10), completion_tokens: Some(2), total_tokens: Some(12) }),
                    }
                } else {
                    AgentTurn {
                        content: Some("ok".into()),
                        tool_calls: vec![],
                        finish_reason: Some("stop".into()),
                        usage: Some(Usage { prompt_tokens: Some(20), completion_tokens: Some(3), total_tokens: Some(23) }),
                    }
                };
                std::future::ready(Ok(turn))
            },
            |_call| std::future::ready(None::<String>),
        )
        .await
        .unwrap();
        assert_eq!(usage.prompt_tokens, Some(30));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(35));
    }

    // ---- memory tools (Stufe B-2) ----

    #[test]
    fn agent_tools_includes_memory_only_when_enabled() {
        let names = |ts: &[Tool]| ts.iter().map(|t| t.function.name.clone()).collect::<Vec<_>>();
        let off = names(&agent_tools(false));
        assert!(!off.iter().any(|n| n == RECALL || n == REMEMBER), "memory off → not offered: {off:?}");
        assert!(off.iter().any(|n| n == READ_FILE), "file tools always present");
        let on = names(&agent_tools(true));
        assert!(on.iter().any(|n| n == RECALL), "memory on → recall offered");
        assert!(on.iter().any(|n| n == REMEMBER), "memory on → remember offered");
        assert_eq!(on.len(), off.len() + 2, "exactly the two memory tools added");
    }

    #[test]
    fn is_memory_tool_recognizes_recall_and_remember() {
        assert!(is_memory_tool(RECALL));
        assert!(is_memory_tool(REMEMBER));
        assert!(!is_memory_tool(READ_FILE));
        assert!(!is_memory_tool(SHELL));
        assert!(!is_memory_tool("delete_everything"));
    }

    #[test]
    fn format_recall_includes_full_text_and_count() {
        assert_eq!(format_recall(&[]), "No relevant notes found in project memory.");
        let hits = vec![
            MemoryHit { id: 1, kind: "Decision".into(), name: "n".into(),
                text: "MTP parked because MoE was net-negative".into(), status: "active".into(), score: 0.83 },
            MemoryHit { id: 2, kind: "Bug".into(), name: "n".into(),
                text: "gfx1201 barrier reduction did not help".into(), status: "active".into(), score: 0.71 },
        ];
        let out = format_recall(&hits);
        assert!(out.contains("Found 2 note(s)"));
        assert!(out.contains("(Decision, score 0.83)"));
        assert!(out.contains("MTP parked because MoE was net-negative"), "full text, not a snippet");
        assert!(out.contains("gfx1201 barrier reduction did not help"));
    }

    #[tokio::test]
    async fn memory_dispatch_intercepts_before_the_gate() {
        // A recall call under a DENY-ALL file/shell gate must still run — memory
        // is a separate axis, intercepted by the mem closure before handle_call.
        let tools = agent_tools(true);
        let call = named_call("recall", "{\"query\":\"x\"}");
        let mut iter = 0;
        let (end, _u) = run_loop(
            vec![ChatMessage::user("go")],
            Gate::headless(None), // deny everything on the file/shell axis
            &ws(),
            &tools,
            None,
            LOOP_CAP,
            |_m| {
                iter += 1;
                let c = call.clone();
                let turn = if iter == 1 {
                    AgentTurn { content: None, tool_calls: vec![c], finish_reason: Some("tool_calls".into()), usage: None }
                } else {
                    AgentTurn { content: Some("done".into()), tool_calls: vec![], finish_reason: Some("stop".into()), usage: None }
                };
                std::future::ready(Ok(turn))
            },
            // Stand-in for dispatch_memory: handle recall/remember, else None.
            |c: ToolCall| std::future::ready(
                is_memory_tool(&c.function.name).then(|| "Found 1 note(s) in project memory:\n[1] (Note, score 0.9) hi\n".to_string())
            ),
        ).await.unwrap();
        // The recall was handled (not denied) → the loop did the roundtrip and
        // then finished normally.
        assert_eq!(iter, 2, "recall handled on the memory axis → roundtrip + final");
        assert_eq!(end, LoopEnd::Final { content: Some("done".into()), finish_reason: Some("stop".into()) });
    }
}
