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

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use serde::Deserialize;

use crate::client::{AgentTurn, Client, MemCall, Result};
use crate::tools::{
    all_tools, execute_read_file, execute_search, execute_shell, execute_write_file, memory_tools,
    read_agents_md, ARCHIVE, READ_FILE, RECALL, REMEMBER, SEARCH, SHELL, WRITE_FILE,
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
    read_yes_no()
}

/// Read a `y/N` answer from stdin. EOF / read error / anything other than an
/// explicit yes → `false` (deny by default). The prompt itself is printed by
/// the caller (so it can render a tool- or archive-specific warning first).
fn read_yes_no() -> bool {
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

/// Whether project memory is available this session and, when on, whether the
/// **active scope** already holds notes. Drives the wording in
/// [`self_state_prompt`]: the proactive-recall nudge fires only for a scope
/// that already has notes, so a fresh/empty project never triggers Over-Recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAwareness {
    /// Server reports memory off (no recall/remember tools offered).
    Off,
    /// Memory on. `scope_has_notes` = the active `project_key` already appears
    /// in `GET /memory/projects` (i.e. it has ≥1 stored note).
    On { scope_has_notes: bool },
}

/// One permission line for the self-state block, mode-aware via [`decide`]:
/// `Auto` → allowed, `Prompt` (REPL) → allowed-with-confirmation, `Deny`
/// (headless above ceiling) → denied + the flag that lifts it.
fn perm_line(gate: Gate, risk: ToolRisk, label: &str) -> String {
    let status = match decide(gate.mode, gate.auto_ceiling, risk) {
        Decision::Auto => "allowed".to_string(),
        // Above the ceiling in the REPL: NOT pre-authorized — the user must
        // approve each call. (Sharper than a bare "allowed": it is gated.)
        Decision::Prompt => "confirm-gated (you approve each call; not pre-authorized)".to_string(),
        Decision::Deny => format!("denied — needs {} at startup", needed_flag(risk)),
    };
    format!("- {label}: {status}\n")
}

/// Build the dynamic **self-state** block appended to the constitution so the
/// agent has an accurate self-image (the B-2 real-use finding: it otherwise
/// guesses its tools/permissions/memory limits):
/// - its CURRENT tool permissions (so it never claims it lacks a permission it
///   has — finding #3 — nor offers one it doesn't);
/// - the active memory scope + that `recall` reads MEMORY while `search` reads
///   FILES (finding #1), id-citation discipline (finding #5), and the curation
///   boundary (it may `archive` a note it recalled this session — the user
///   confirms — while `forget`/delete stays user-only, finding #4);
/// - a **scoped** proactive-recall nudge, emitted only when the scope already
///   has notes (findings #1/#2), so empty scopes don't cause Over-Recall.
/// Pure → unit-testable.
pub fn self_state_prompt(gate: Gate, project: Option<&str>, mem: MemoryAwareness) -> String {
    let mut s = String::from("\n\n# Current session state\n");
    s.push_str(
        "Your tool permissions right now (do NOT claim a tool needs a permission it already has, \
         and do NOT offer one it lacks):\n",
    );
    s.push_str(&perm_line(gate, ToolRisk::ReadOnly, "read_file / search (read-only, workspace-confined)"));
    s.push_str(&perm_line(gate, ToolRisk::Mutating, "write_file (workspace-confined)"));
    s.push_str(&perm_line(
        gate,
        ToolRisk::Exec,
        "shell (un-confined: a command can touch paths outside the workspace)",
    ));
    match mem {
        MemoryAwareness::Off => {
            s.push_str("\nProject memory is not available this session (no recall/remember).\n");
        }
        MemoryAwareness::On { scope_has_notes } => {
            let scope = project.unwrap_or("(global)");
            s.push_str(&format!("\nProject memory (scope `{scope}`):\n"));
            s.push_str(
                "- Use `recall` to read this project's stored notes (past decisions, conventions, \
                 context) and `remember` to store durable ones. `recall` reads MEMORY — it is NOT \
                 `search`, which reads workspace FILES.\n",
            );
            s.push_str(
                "- When you reference a recalled note, cite its exact id from the recall result \
                 (e.g. `id 5`); never invent an id.\n",
            );
            s.push_str(
                "- You can `archive` a note you recalled THIS session: call `archive` with its \
                 exact id and a short reason. The user confirms it first (seeing the note's real \
                 content), and you can only archive a note you actually recalled this session — \
                 so use it for a clearly stale or superseded note. Archiving is reversible: the \
                 user can restore a note with `/unarchive <id>`.\n",
            );
            s.push_str(
                "- `forget` (permanent delete) is user-only: you have no delete tool and must not \
                 invent one. If asked to forget or delete a note, tell the user to run \
                 `/forget <id>`.\n",
            );
            if scope_has_notes {
                s.push_str(
                    "- This project already has stored memory. For questions about prior decisions, \
                     conventions, or \"what did we…/why did we…/what was our rule for…\", call \
                     `recall` BEFORE searching files or answering from general knowledge.\n",
                );
            }
        }
    }
    s
}

/// Append [`self_state_prompt`] to the leading `system` message in place. Only
/// an existing system message is augmented — so `--no-system` (no constitution)
/// stays honored: with no system message, nothing is injected.
fn inject_self_state(
    messages: &mut [ChatMessage],
    gate: Gate,
    project: Option<&str>,
    mem: MemoryAwareness,
) {
    if let Some(first) = messages.first_mut() {
        if first.role == "system" {
            first.content.push_str(&self_state_prompt(gate, project, mem));
        }
    }
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
    mut messages: Vec<ChatMessage>,
    gate: Gate,
    workspace: &Path,
    project: Option<&str>,
    status: Option<&StatusBar>,
) -> Result<(LoopEnd, Usage)> {
    // Startup probe: `GET /memory/projects` → 200 = memory on (offer the
    // tools); 503 / transport error = not offered, so the model never makes a
    // doomed memory call. The returned scope list also tells us whether THIS
    // project already has notes (→ a worthwhile proactive recall; awareness is
    // count/presence-only, never note content — the philosophy boundary).
    let probe = client.memory_projects().await;
    let memory_enabled = matches!(probe, Ok(MemCall::Ok(_)));
    let awareness = match &probe {
        Ok(MemCall::Ok(resp)) => MemoryAwareness::On {
            scope_has_notes: project
                .is_some_and(|p| resp.projects.iter().any(|pi| pi.project_key == p)),
        },
        _ => MemoryAwareness::Off,
    };
    // Give the agent an accurate self-image: append the live permissions +
    // memory scope/boundaries to the constitution before the loop starts.
    inject_self_state(&mut messages, gate, project, awareness);
    let tools = agent_tools(memory_enabled);
    // Session recall cache: `recall` results land here keyed by note id, so
    // `archive` can be bound to notes the agent actually recalled THIS session
    // (anti-hallucination — see [`plan_archive`]) and the confirm can render
    // the note's REAL content, never the model's free-text claim. Lives for
    // the whole turn; interior-mutable so the recall (writer) and archive
    // (reader) sides share it across the loop without a borrow conflict.
    let cache: RecallCache = RefCell::new(HashMap::new());
    run_loop(
        messages,
        gate,
        workspace,
        &tools,
        status,
        LOOP_CAP,
        |msgs| client.chat_with_tools(msgs, &tools),
        |call| dispatch_memory(client, project, gate, &cache, call),
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
/// the file/shell permission gate). `recall`/`remember` run unconfirmed;
/// `archive` is on the same axis but gated **confirm-always**.
fn is_memory_tool(name: &str) -> bool {
    matches!(name, RECALL | REMEMBER | ARCHIVE)
}

/// One recalled note kept in the session [`RecallCache`]: the fields needed to
/// render an honest archive confirmation (the note's REAL content) and to
/// prove the agent actually recalled it this session.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedNote {
    kind: String,
    name: String,
    text: String,
}

/// Per-turn `id → note` cache of everything `recall` returned this session.
/// Interior-mutable so the recall (writer) and archive (reader) sides share it
/// across the loop. A `RefCell` (not a `Mutex`) is fine: the agent loop is a
/// single task and the cache is only touched between `.await`s, never across one.
type RecallCache = RefCell<HashMap<i64, CachedNote>>;

/// Dispatch a memory tool call via the HTTP client. `Some(result)` for
/// `recall`/`remember`/`archive` (handled here, **before** the file/shell
/// gate); `None` for any other tool (falls through to the gated dispatch). The
/// names are recognized even if the server turns out memory-off (race): the
/// call then returns a clean "memory not enabled" result, never a crash. Every
/// memory action prints a visible marker (philosophy: results are visible).
/// `gate` is used only by `archive` (confirm-always — the ceiling never lifts it).
async fn dispatch_memory(
    client: &Client,
    project: Option<&str>,
    gate: Gate,
    cache: &RecallCache,
    call: ToolCall,
) -> Option<String> {
    let name = call.function.name.as_str();
    if !is_memory_tool(name) {
        return None;
    }
    let args = call.function.arguments.as_str();
    Some(match name {
        RECALL => execute_recall(client, project, cache, args).await,
        REMEMBER => execute_remember(client, project, args).await,
        // The production confirm: render the warning to stderr (done inside
        // execute_archive) then read a real y/N from stdin.
        ARCHIVE => execute_archive(client, project, gate, cache, args, read_yes_no).await,
        _ => unreachable!("guarded by is_memory_tool"),
    })
}

async fn execute_recall(
    client: &Client,
    project: Option<&str>,
    cache: &RecallCache,
    args: &str,
) -> String {
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
            // Ground future `archive` calls: cache each hit's id → real content
            // (scoped borrow_mut, dropped before returning — no await held).
            {
                let mut c = cache.borrow_mut();
                for h in &resp.hits {
                    c.insert(
                        h.id,
                        CachedNote { kind: h.kind.clone(), name: h.name.clone(), text: h.text.clone() },
                    );
                }
            }
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
        // Dedup hit (Stufe B-3): honest "already known", not a second store.
        Ok(MemCall::Ok(resp)) if resp.deduped => {
            eprintln!("[agent] already known [{kind}] (id {}): {}", resp.id, short(&parsed.text));
            format!("Already in memory as [{kind}] (id {}); not stored again.", resp.id)
        }
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

/// The pre-server outcome of an `archive` request — what to do *before* any
/// `/memory/archive` call. Keeps the cache/gate logic pure (no I/O), so the
/// confirm-always and grounding rules are unit-testable without stdin or a
/// live server.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ArchivePlan {
    /// The id wasn't recalled this session (or an empty reason) — refuse
    /// outright; never reaches the server (anti-hallucination guard).
    Refused(String),
    /// Headless (no TTY to confirm) — deny, consistent with the above-ceiling
    /// headless rule; never reaches the server.
    Denied(String),
    /// REPL — ask the user to confirm (showing this note's REAL content) before
    /// the server call. Carries the cached note so the confirm can't be spoofed
    /// by the model's arguments.
    Confirm(CachedNote),
}

/// Decide what to do with an `archive(id)` request from cache + gate alone —
/// **no I/O**. The permission ceiling is deliberately ignored: `archive` is
/// **confirm-always** in the REPL (even under `--allow-shell`) and denied
/// headless. The id must be in the session recall cache, binding `archive` to
/// notes the agent actually recalled (so a hallucinated id is refused, and no
/// per-id server lookup is needed).
fn plan_archive(gate: Gate, cache: &HashMap<i64, CachedNote>, id: i64) -> ArchivePlan {
    let Some(note) = cache.get(&id) else {
        return ArchivePlan::Refused(format!(
            "id {id} was not recalled this session — recall it first."
        ));
    };
    match gate.mode {
        // No TTY headless → can't confirm → deny (never auto-archive a note).
        Permission::Headless => ArchivePlan::Denied(format!(
            "archive needs an interactive confirmation (none headless); id {id} not archived."
        )),
        // REPL → confirm-always, regardless of the ceiling.
        Permission::Interactive => ArchivePlan::Confirm(note.clone()),
    }
}

/// Render the archive confirmation shown to the user. Uses the cached note's
/// **real** `name`/`text` (never the model's free-text claim) so a wrong id
/// paired with a plausible invented description is caught by eye. Ends with the
/// `[y/N]` prompt (no trailing newline) so the answer lands on the same line.
fn render_archive_confirm(id: i64, note: &CachedNote, reason: &str) -> String {
    let label = if !note.name.trim().is_empty() {
        note.name.trim()
    } else if !note.kind.trim().is_empty() {
        note.kind.trim()
    } else {
        "note"
    };
    format!(
        "[agent] ⚠ ARCHIVE note {id} — \"{label}\": {}\n        reason: {}\n        Drops it from \
         future recall but keeps it as an archived record (no hard delete) — reversible: the user \
         can restore it with `/unarchive {id}`.\n[agent] archive note {id}? [y/N] ",
        short(&note.text),
        reason.trim(),
    )
}

/// `archive` agent tool (Stufe B-3 curation) — drop a recalled note from future
/// recall, kept as an archived record. On the memory axis (before the file/shell
/// gate) but **confirm-always**: REPL asks y/N (even under `--allow-shell`),
/// headless denies. Bound to ids recalled this session via `cache`. `confirm`
/// reads the y/N answer (injected for tests); the warning itself is rendered
/// here to stderr so stdout stays byte-clean.
async fn execute_archive(
    client: &Client,
    project: Option<&str>,
    gate: Gate,
    cache: &RecallCache,
    args: &str,
    mut confirm: impl FnMut() -> bool,
) -> String {
    #[derive(Deserialize)]
    struct ArchiveArgs {
        id: i64,
        #[serde(default)]
        reason: String,
    }
    let parsed: ArchiveArgs = match serde_json::from_str(args) {
        Ok(a) => a,
        Err(e) => return format!("Tool error: invalid archive arguments: {e}"),
    };
    if parsed.reason.trim().is_empty() {
        return "Tool error: archive requires a non-empty reason.".to_string();
    }
    // Pure decision from cache + gate (scoped borrow, dropped before any await).
    let plan = plan_archive(gate, &cache.borrow(), parsed.id);
    let note = match plan {
        ArchivePlan::Refused(why) => {
            eprintln!("[agent] archive refused: {why}");
            return format!("Cannot archive: {why}");
        }
        ArchivePlan::Denied(why) => {
            eprintln!("[agent] archive denied: {why}");
            return format!("Archive not done: {why}");
        }
        ArchivePlan::Confirm(note) => note,
    };
    // Show the REAL note + the agent's reason, then read y/N.
    eprint!("{}", render_archive_confirm(parsed.id, &note, &parsed.reason));
    let _ = std::io::stderr().flush();
    if !confirm() {
        eprintln!("[agent] archive declined — id {} not archived", parsed.id);
        return format!(
            "Archive of note {} was declined by the user; nothing was changed.",
            parsed.id
        );
    }
    match client.memory_archive(project, parsed.id).await {
        Ok(MemCall::Ok(_)) => {
            eprintln!("[agent] archived note {}", parsed.id);
            format!(
                "Archived note {} (dropped from recall; kept as an archived record).",
                parsed.id
            )
        }
        Ok(MemCall::Disabled) => {
            eprintln!("[agent] archive: memory is not enabled on this server");
            "Memory is not enabled on this server; nothing was archived.".to_string()
        }
        Err(e) => {
            eprintln!("[agent] archive failed: {e}");
            format!("Tool error: archive failed: {e}")
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
    for h in hits {
        // The REAL note id (not a positional index) so the model cites it
        // correctly — B-2 finding #5: it reported the position ("id 1") for a
        // note whose true id was 5, because the position was all this result
        // exposed.
        out.push_str(&format!("[id {}] ({}, score {:.2}) {}\n", h.id, h.kind, h.score, h.text.trim()));
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
        assert!(on.iter().any(|n| n == ARCHIVE), "memory on → archive offered (B-3)");
        // forget/delete is NOT an agent tool — hard delete stays user-only.
        assert!(!on.iter().any(|n| n == "forget" || n == "delete"), "no delete tool: {on:?}");
        assert_eq!(on.len(), off.len() + 3, "exactly the three memory tools added");
    }

    #[test]
    fn is_memory_tool_recognizes_recall_remember_archive() {
        assert!(is_memory_tool(RECALL));
        assert!(is_memory_tool(REMEMBER));
        assert!(is_memory_tool(ARCHIVE));
        assert!(!is_memory_tool(READ_FILE));
        assert!(!is_memory_tool(SHELL));
        assert!(!is_memory_tool("forget"));
        assert!(!is_memory_tool("delete_everything"));
    }

    #[test]
    fn format_recall_shows_real_id_not_position() {
        assert_eq!(format_recall(&[]), "No relevant notes found in project memory.");
        let hits = vec![
            MemoryHit { id: 5, kind: "Decision".into(), name: "n".into(),
                text: "MTP parked because MoE was net-negative".into(), status: "active".into(),
                note_type: "untyped".into(), superseded_by: None, derives_from: Vec::new(), frontier_via: None, conflicts_with: Vec::new(), contested_by: None, score: 0.83 },
            MemoryHit { id: 9, kind: "Bug".into(), name: "n".into(),
                text: "gfx1201 barrier reduction did not help".into(), status: "active".into(),
                note_type: "untyped".into(), superseded_by: None, derives_from: Vec::new(), frontier_via: None, conflicts_with: Vec::new(), contested_by: None, score: 0.71 },
        ];
        let out = format_recall(&hits);
        assert!(out.contains("Found 2 note(s)"));
        assert!(out.contains("(Decision, score 0.83)"));
        // The REAL ids, not the positional index (B-2 #5): note 1 has id 5.
        assert!(out.contains("[id 5]"), "must show the real id: {out}");
        assert!(out.contains("[id 9]"), "must show the real id: {out}");
        assert!(!out.contains("[1]") && !out.contains("[2]"), "must NOT use the positional index: {out}");
        assert!(out.contains("MTP parked because MoE was net-negative"), "full text, not a snippet");
        assert!(out.contains("gfx1201 barrier reduction did not help"));
    }

    // ---- self-state (B-2 tuning): permissions + memory awareness ----

    #[test]
    fn self_state_reports_real_permissions() {
        use ToolRisk::*;
        // --allow-shell (Exec ceiling), headless: every tool allowed; never
        // claims it needs the flag it already has (finding #3). shell carries
        // the un-confined caveat; confined tools say so.
        let shell = self_state_prompt(Gate::headless(Some(Exec)), Some("proj"), MemoryAwareness::Off);
        assert!(shell.contains("shell (un-confined"), "shell flagged un-confined: {shell}");
        assert!(
            shell.contains("shell (un-confined: a command can touch paths outside the workspace): allowed"),
            "{shell}"
        );
        assert!(!shell.contains("needs --allow-shell"), "must not claim a permission it has: {shell}");
        assert!(shell.contains("write_file (workspace-confined): allowed"), "{shell}");
        // --yes (ReadOnly ceiling), headless: read allowed, write+shell denied
        // and the lifting flag named.
        let yes = self_state_prompt(Gate::headless(Some(ReadOnly)), Some("proj"), MemoryAwareness::Off);
        assert!(yes.contains("read_file / search (read-only, workspace-confined): allowed"), "{yes}");
        assert!(yes.contains("write_file (workspace-confined): denied — needs --allow-mutating"), "{yes}");
        assert!(
            yes.contains("shell (un-confined: a command can touch paths outside the workspace): denied — needs --allow-shell"),
            "{yes}"
        );
    }

    #[test]
    fn self_state_write_is_confirm_gated_in_repl_below_ceiling() {
        use ToolRisk::*;
        // REPL with --yes (ReadOnly ceiling): write_file/shell sit ABOVE the
        // ceiling → confirm-gated, NOT a bare "allowed" (Schritt-0 sharpening).
        let repl = self_state_prompt(Gate::interactive(Some(ReadOnly)), Some("p"), MemoryAwareness::Off);
        assert!(repl.contains("write_file (workspace-confined): confirm-gated"), "{repl}");
        assert!(
            repl.contains("shell (un-confined: a command can touch paths outside the workspace): confirm-gated"),
            "{repl}"
        );
        assert!(repl.contains("you approve each call"), "{repl}");
        // read-only is at the ceiling → genuinely allowed.
        assert!(repl.contains("read_file / search (read-only, workspace-confined): allowed"), "{repl}");
    }

    #[test]
    fn self_state_memory_off_says_unavailable() {
        let s = self_state_prompt(Gate::headless(Some(ToolRisk::Exec)), Some("proj"), MemoryAwareness::Off);
        assert!(s.contains("memory is not available"), "{s}");
        // no curation / nudge guidance when memory is off.
        assert!(!s.contains("/forget"), "{s}");
        assert!(!s.contains("BEFORE searching"), "{s}");
    }

    #[test]
    fn self_state_memory_on_has_scope_boundaries_and_id_rule() {
        let s = self_state_prompt(
            Gate::interactive(Some(ToolRisk::Exec)),
            Some("vulkanforge-1a2b3c4d"),
            MemoryAwareness::On { scope_has_notes: true },
        );
        assert!(s.contains("scope `vulkanforge-1a2b3c4d`"), "{s}");
        assert!(s.contains("recall") && s.contains("remember"));
        assert!(s.contains("reads MEMORY") && s.contains("reads workspace FILES"), "recall-vs-search (#1): {s}");
        assert!(s.contains("cite its exact id"), "id rule (#5): {s}");
        // Curation boundary (#4): agent MAY archive a recalled note (confirmed);
        // forget/delete stays user-only and it must not invent a delete.
        assert!(
            s.contains("`archive`") && s.contains("recalled THIS session"),
            "archive is offered: {s}"
        );
        assert!(
            s.contains("`forget`") && s.contains("user-only") && s.contains("not invent"),
            "forget user-only / no invented delete (#4): {s}"
        );
        assert!(s.contains("/forget"), "points at the user command: {s}");
    }

    #[test]
    fn self_state_distinguishes_archive_from_forget() {
        // The agent's two curation truths must be unambiguous: it CAN archive
        // (with confirmation) but CANNOT delete (user-only).
        let s = self_state_prompt(
            Gate::interactive(Some(ToolRisk::ReadOnly)),
            Some("proj"),
            MemoryAwareness::On { scope_has_notes: false },
        );
        // archive: presented as an available, confirmed action.
        assert!(s.contains("You can `archive`"), "archive presented as allowed: {s}");
        // forget: explicitly user-only, no delete tool.
        assert!(
            s.contains("`forget` (permanent delete) is user-only"),
            "forget marked user-only: {s}"
        );
        assert!(s.contains("no delete tool"), "states it has no delete tool: {s}");
    }

    #[test]
    fn self_state_scoped_nudge_only_when_scope_has_notes() {
        let g = Gate::headless(Some(ToolRisk::Exec));
        let with = self_state_prompt(g, Some("p"), MemoryAwareness::On { scope_has_notes: true });
        let without = self_state_prompt(g, Some("p"), MemoryAwareness::On { scope_has_notes: false });
        assert!(with.contains("call `recall` BEFORE"), "notes present → proactive nudge: {with}");
        assert!(!without.contains("call `recall` BEFORE"), "empty scope → no Over-Recall nudge: {without}");
        // both still expose recall/remember + the user-only curation boundary.
        assert!(without.contains("recall") && without.contains("/forget"), "{without}");
    }

    #[test]
    fn inject_self_state_augments_system_only() {
        let g = Gate::headless(Some(ToolRisk::Exec));
        // System present → appended in place (constitution kept, not replaced).
        let mut with_sys = vec![ChatMessage::system("CONSTITUTION"), ChatMessage::user("hi")];
        inject_self_state(&mut with_sys, g, Some("p"), MemoryAwareness::On { scope_has_notes: false });
        assert!(with_sys[0].content.starts_with("CONSTITUTION"), "appended, not replaced");
        assert!(with_sys[0].content.contains("Current session state"), "self-state appended");
        assert_eq!(with_sys[1].content, "hi", "user message untouched");
        // No system message (--no-system) → nothing injected.
        let mut no_sys = vec![ChatMessage::user("hi")];
        inject_self_state(&mut no_sys, g, Some("p"), MemoryAwareness::Off);
        assert_eq!(no_sys.len(), 1);
        assert_eq!(no_sys[0].content, "hi", "no system message → untouched (honors --no-system)");
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

    // ---- archive (Stufe B-3 curation): confirm-always, grounded, honest ----

    fn note(kind: &str, name: &str, text: &str) -> CachedNote {
        CachedNote { kind: kind.into(), name: name.into(), text: text.into() }
    }

    fn cache_with(id: i64, n: CachedNote) -> RecallCache {
        let c = RefCell::new(HashMap::new());
        c.borrow_mut().insert(id, n);
        c
    }

    #[test]
    fn plan_archive_is_confirm_always_in_repl_even_under_allow_shell() {
        let c = HashMap::from([(7, note("Decision", "MTP", "parked"))]);
        // REPL, regardless of ceiling (None / --yes / --allow-shell) → Confirm.
        for ceil in [None, Some(ToolRisk::ReadOnly), Some(ToolRisk::Exec)] {
            assert_eq!(
                plan_archive(Gate::interactive(ceil), &c, 7),
                ArchivePlan::Confirm(note("Decision", "MTP", "parked")),
                "REPL archive must confirm-always (ceiling {ceil:?})"
            );
        }
    }

    #[test]
    fn plan_archive_headless_denies_even_under_allow_shell() {
        let c = HashMap::from([(7, note("Decision", "MTP", "parked"))]);
        for ceil in [None, Some(ToolRisk::Exec)] {
            assert!(
                matches!(plan_archive(Gate::headless(ceil), &c, 7), ArchivePlan::Denied(_)),
                "headless archive must deny (ceiling {ceil:?})"
            );
        }
    }

    #[test]
    fn plan_archive_refuses_id_not_recalled_this_session() {
        let c = HashMap::new(); // nothing recalled
        let plan = plan_archive(Gate::interactive(Some(ToolRisk::Exec)), &c, 7);
        match plan {
            ArchivePlan::Refused(why) => assert!(why.contains("recall it first"), "{why}"),
            other => panic!("ungrounded id must be Refused, got {other:?}"),
        }
    }

    #[test]
    fn render_archive_confirm_shows_real_content_not_agent_claim() {
        let n = note("Decision", "MTP", "MTP parked because MoE was net-negative");
        // The model's reason DIVERGES from the note's real text on purpose.
        let out = render_archive_confirm(7, &n, "the user said this note is about coffee");
        assert!(out.contains("ARCHIVE note 7"), "{out}");
        assert!(out.contains("\"MTP\""), "shows the real label: {out}");
        assert!(out.contains("MTP parked because MoE was net-negative"), "shows the REAL text: {out}");
        // the agent's claim appears only as a labeled 'reason:', never as the
        // note's content — so a wrong id + invented description is caught by eye.
        assert!(out.contains("reason: the user said this note is about coffee"), "{out}");
        // Honest, now-reversible wording: kept as a record + restorable via /unarchive.
        assert!(out.contains("no hard delete") && out.contains("/unarchive 7"), "reversible wording: {out}");
        assert!(out.trim_end().ends_with("[y/N]"), "ends with the prompt: {out}");
    }

    fn dead_client() -> Client {
        // port 0 never connects — any actual server call returns Err, so a
        // result that is NOT an error proves the server was never called.
        Client::new("http://localhost:0", "m")
    }

    #[tokio::test]
    async fn execute_archive_confirm_always_declines_under_exec_ceiling_no_server_call() {
        // REPL with --allow-shell (Exec ceiling): archive must STILL ask (not
        // auto-run). The injected confirm says No → declined, server untouched.
        let c = cache_with(7, note("Decision", "MTP", "parked"));
        let out = execute_archive(
            &dead_client(),
            Some("p"),
            Gate::interactive(Some(ToolRisk::Exec)),
            &c,
            r#"{"id":7,"reason":"stale"}"#,
            || false,
        )
        .await;
        assert!(out.contains("declined"), "confirm-always then No → declined: {out}");
        assert!(!out.contains("failed"), "server must not have been called: {out}");
    }

    #[tokio::test]
    async fn execute_archive_declined_does_not_archive() {
        let c = cache_with(7, note("Decision", "MTP", "parked"));
        let out = execute_archive(
            &dead_client(),
            Some("p"),
            Gate::interactive(None),
            &c,
            r#"{"id":7,"reason":"stale"}"#,
            || false,
        )
        .await;
        assert!(out.contains("declined") && out.contains("nothing was changed"), "{out}");
        assert!(!out.contains("failed"), "no server call on decline: {out}");
    }

    #[tokio::test]
    async fn execute_archive_refuses_ungrounded_id_without_server_call() {
        let c: RecallCache = RefCell::new(HashMap::new()); // nothing recalled
        // confirm says yes, but it must never get there — id isn't grounded.
        let out = execute_archive(
            &dead_client(),
            Some("p"),
            Gate::interactive(Some(ToolRisk::Exec)),
            &c,
            r#"{"id":7,"reason":"stale"}"#,
            || true,
        )
        .await;
        assert!(out.contains("recall it first"), "ungrounded id refused: {out}");
        assert!(!out.contains("failed"), "no server call on refusal: {out}");
    }

    #[tokio::test]
    async fn execute_archive_headless_denied_no_hang() {
        let c = cache_with(7, note("Decision", "MTP", "parked"));
        let out = execute_archive(
            &dead_client(),
            Some("p"),
            Gate::headless(Some(ToolRisk::Exec)), // even --allow-shell headless
            &c,
            r#"{"id":7,"reason":"stale"}"#,
            || true,
        )
        .await;
        assert!(out.contains("not archived") && out.contains("interactive"), "{out}");
        assert!(!out.contains("failed"), "no server call when denied: {out}");
    }

    #[tokio::test]
    async fn execute_archive_requires_a_reason() {
        let c = cache_with(7, note("Decision", "MTP", "parked"));
        let out = execute_archive(
            &dead_client(),
            Some("p"),
            Gate::interactive(Some(ToolRisk::Exec)),
            &c,
            r#"{"id":7,"reason":"  "}"#,
            || true,
        )
        .await;
        assert!(out.contains("non-empty reason"), "empty reason rejected: {out}");
    }

    #[tokio::test]
    async fn recall_caches_ids_for_later_archive_grounding() {
        // Stand-in proof that the recall→cache→archive binding is wired: after a
        // recall populates the cache, plan_archive accepts that id (and only it).
        let c: RecallCache = RefCell::new(HashMap::new());
        c.borrow_mut().insert(5, note("Bug", "x", "y")); // emulate a recall write
        assert!(matches!(
            plan_archive(Gate::interactive(None), &c.borrow(), 5),
            ArchivePlan::Confirm(_)
        ));
        assert!(matches!(
            plan_archive(Gate::interactive(None), &c.borrow(), 6),
            ArchivePlan::Refused(_)
        ));
    }
}
