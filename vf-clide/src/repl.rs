// SPDX-License-Identifier: GPL-3.0-only
//! Interactive REPL: streamed chat with in-session history.
//!
//! Commands: `/quit` (`/q`, `/exit`), `/clear` (drop history),
//! `/model <name>` (switch model), and the memory commands (Stufe B-1):
//! `/project [key]` (show + list / switch scope), `/recall <query>`
//! (manual semantic recall — displays hits, no auto-inject), `/remember
//! <text>` (store a note). Anything else is sent to the model.

use std::io::Write;

use rustyline::DefaultEditor;

use crate::client::{empty_notice, truncation_notice, Client, MemCall, Result};
use crate::memory::Memory;
use crate::status::StatusBar;
use crate::types::{strip_think, ChatMessage, ToolRisk};

/// Parsed REPL command. `None` from [`parse_command`] means "not a
/// command — treat the line as a chat prompt".
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Quit,
    Clear,
    Model(String),
    /// `/think` (true → thinking on) / `/no-think` (false → off). Note:
    /// `/no-think` is the REPL toggle; the model directive `/no_think`
    /// (underscore) is appended to message *content*, never parsed here.
    Think(bool),
    MaxTokens(u32),
    /// `/project` (no arg → show current scope + list known projects) or
    /// `/project <key>` (switch the session's memory scope).
    Project(Option<String>),
    /// `/recall <query>` — manual semantic recall against the current
    /// project's memory; displays hits (no auto-inject — that is B-2).
    Recall(String),
    /// `/remember <text>` — store a manual note (`kind:"Note"`).
    Remember(String),
    /// `/archive <id>` — drop a note from recall (kept as an archived record).
    Archive(i64),
    /// `/unarchive <id>` — restore an archived note to recall (inverse of
    /// `/archive`). User-driven recovery; the agent has no such tool.
    Unarchive(i64),
    /// `/forget <id>` — hard-delete a note from recall AND the graph.
    Forget(i64),
    /// A `/`-line that isn't a known command (or a malformed one). The
    /// string is a human-readable hint; the REPL prints it and does not
    /// send the line to the model.
    Unknown(String),
}

/// Parse a REPL input line. Returns `None` for normal chat input,
/// `Some(Command)` for a `/`-prefixed control line.
pub fn parse_command(line: &str) -> Option<Command> {
    let line = line.trim();
    if !line.starts_with('/') {
        return None;
    }
    let mut parts = line.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    Some(match cmd {
        "/quit" | "/q" | "/exit" => Command::Quit,
        "/clear" => Command::Clear,
        "/model" => match arg {
            Some(m) => Command::Model(m),
            None => Command::Unknown("usage: /model <name>".into()),
        },
        "/think" => Command::Think(true),
        "/no-think" => Command::Think(false),
        "/max-tokens" => match arg.and_then(|a| a.parse::<u32>().ok()) {
            Some(n) if n > 0 => Command::MaxTokens(n),
            _ => Command::Unknown("usage: /max-tokens <N>".into()),
        },
        "/project" => Command::Project(arg),
        "/recall" => match arg {
            Some(q) => Command::Recall(q),
            None => Command::Unknown("usage: /recall <query>".into()),
        },
        "/remember" => match arg {
            Some(t) => Command::Remember(t),
            None => Command::Unknown("usage: /remember <text>".into()),
        },
        "/archive" => match arg.and_then(|a| a.parse::<i64>().ok()) {
            Some(id) => Command::Archive(id),
            None => Command::Unknown("usage: /archive <id>  (id from /recall)".into()),
        },
        "/unarchive" => match arg.and_then(|a| a.parse::<i64>().ok()) {
            Some(id) => Command::Unarchive(id),
            None => Command::Unknown("usage: /unarchive <id>  (id from /recall)".into()),
        },
        "/forget" => match arg.and_then(|a| a.parse::<i64>().ok()) {
            Some(id) => Command::Forget(id),
            None => Command::Unknown("usage: /forget <id>  (id from /recall)".into()),
        },
        other => Command::Unknown(format!("unknown command: {other}")),
    })
}

/// Shown when a `/memory/*` call returns 503 — memory is off by default
/// (opt-in since v1.0.1), which is a normal state, not an error.
const MEMORY_OFF_HINT: &str = "memory is not enabled on this server (start it with `serve --memory`)";

/// One-line preview of a recalled note: newlines flattened, capped at ~80
/// chars with an ellipsis so long notes don't wrap the terminal.
fn snippet(text: &str) -> String {
    let t = text.trim().replace('\n', " ");
    if t.chars().count() > 80 {
        let head: String = t.chars().take(79).collect();
        format!("{head}…")
    } else {
        t
    }
}

/// REPL state: a client, the memory seam, optional project name, and the
/// in-memory conversation history.
pub struct Repl {
    client: Client,
    memory: Box<dyn Memory>,
    project: Option<String>,
    history: Vec<ChatMessage>,
    /// When true, append the `/no_think` directive to each new user turn
    /// so thinking models skip the `<think>` block (Qwen3 convention).
    no_think: bool,
    /// When true (`--agent`), each turn runs the agent loop with
    /// interactive tool-call permission instead of plain chat.
    agent: bool,
    /// Workspace root for the file tools (confinement). Set via
    /// [`Repl::with_workspace`]; defaults to `.` (agent unused otherwise).
    workspace: std::path::PathBuf,
    /// Agent system prompt (constitution). Prepended as `messages[0]` in
    /// agent mode only — the plain chat path is unaffected.
    system: Option<String>,
    /// Auto-approve ceiling from the cumulative `--yes`/`--allow-*` flags.
    /// In the REPL, calls at or below it are auto-approved (still printed);
    /// calls above it prompt `y/N`. `None` = prompt for everything.
    ceiling: Option<ToolRisk>,
}

impl Repl {
    pub fn new(client: Client, memory: Box<dyn Memory>, project: Option<String>) -> Self {
        Self {
            client,
            memory,
            project,
            history: Vec::new(),
            no_think: false,
            agent: false,
            workspace: std::path::PathBuf::from("."),
            system: None,
            ceiling: None,
        }
    }

    /// Start with thinking disabled (`--no-think`).
    pub fn with_no_think(mut self, no_think: bool) -> Self {
        self.no_think = no_think;
        self
    }

    /// Start in agent mode (`--agent`): tool-calling loop with interactive
    /// permission per call.
    pub fn with_agent(mut self, agent: bool) -> Self {
        self.agent = agent;
        self
    }

    /// Set the (canonicalized) workspace root for the file tools.
    pub fn with_workspace(mut self, workspace: std::path::PathBuf) -> Self {
        self.workspace = workspace;
        self
    }

    /// Set the agent constitution (system prompt); used in agent mode only.
    pub fn with_system(mut self, system: Option<String>) -> Self {
        self.system = system;
        self
    }

    /// Set the cumulative auto-approve ceiling (`--yes`/`--allow-*`). Honored
    /// in the REPL: at/below it = auto-approve (printed), above it = prompt.
    pub fn with_ceiling(mut self, ceiling: Option<ToolRisk>) -> Self {
        self.ceiling = ceiling;
        self
    }

    /// Build the message list for one turn: optional memory context
    /// (no-op in Phase 1) + prior history + the new user input. This is
    /// the single point where the memory seam is consulted. When
    /// `no_think` is set, the `/no_think` directive is appended to the
    /// new user content (NOT a slash-command — it belongs in the message).
    pub fn build_messages(&self, user_input: &str) -> Vec<ChatMessage> {
        let mut msgs = Vec::with_capacity(self.history.len() + 3);
        // Agent constitution at the very front — agent mode only, so the
        // plain chat path is byte-for-byte unchanged.
        if self.agent {
            if let Some(sys) = &self.system {
                msgs.push(ChatMessage::system(sys.clone()));
            }
        }
        if let Some(ctx) = self.memory.context_for(self.project.as_deref(), &self.history) {
            msgs.push(ChatMessage::system(ctx));
        }
        msgs.extend(self.history.iter().cloned());
        let content = if self.no_think {
            format!("{user_input} /no_think")
        } else {
            user_input.to_string()
        };
        msgs.push(ChatMessage::user(content));
        msgs
    }

    /// `/project` (no arg): show the current scope + list the projects the
    /// server already knows.
    async fn show_project(&self) {
        println!("project: {}", self.project.as_deref().unwrap_or("(none)"));
        match self.client.memory_projects().await {
            Ok(MemCall::Ok(resp)) => {
                if resp.projects.is_empty() {
                    println!("  (no projects stored yet)");
                } else {
                    for p in resp.projects {
                        let here = Some(p.project_key.as_str()) == self.project.as_deref();
                        println!("  {} {}", if here { "*" } else { " " }, p.project_key);
                    }
                }
            }
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/recall <query>`: display the current project's matches (no auto-inject).
    /// Shows the note `id` per hit so it can be targeted by `/archive`/`/forget`.
    async fn recall(&self, query: &str) {
        match self.client.memory_recall(self.project.as_deref(), query, 5).await {
            Ok(MemCall::Ok(resp)) if resp.hits.is_empty() => println!("(no matches)"),
            Ok(MemCall::Ok(resp)) => {
                for h in resp.hits {
                    println!("  #{} [{} · {:.2}] {}", h.id, h.kind, h.score, snippet(&h.text));
                }
            }
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/remember <text>`: store a manual note (`kind:"Note"`). Honest about a
    /// dedup hit (says "already known" instead of "stored").
    async fn remember(&self, text: &str) {
        match self.client.memory_remember(self.project.as_deref(), "Note", text).await {
            Ok(MemCall::Ok(resp)) if resp.deduped => println!("(already known, id {})", resp.id),
            Ok(MemCall::Ok(resp)) => println!("(stored, id {})", resp.id),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/archive <id>`: drop a note from recall, keep it as an archived record.
    async fn archive(&self, id: i64) {
        match self.client.memory_archive(self.project.as_deref(), id).await {
            Ok(MemCall::Ok(_)) => println!("(archived id {id} — out of recall, kept as a record)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/unarchive <id>`: restore an archived note to active + recall (inverse
    /// of `/archive`).
    async fn unarchive(&self, id: i64) {
        match self.client.memory_unarchive(self.project.as_deref(), id).await {
            Ok(MemCall::Ok(_)) => println!("(unarchived id {id} — restored to recall)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/forget <id>`: hard-delete a note from recall and the graph.
    async fn forget(&self, id: i64) {
        match self.client.memory_delete(self.project.as_deref(), id).await {
            Ok(MemCall::Ok(_)) => println!("(forgotten id {id} — deleted from memory)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        use std::io::IsTerminal;

        let mut rl = DefaultEditor::new().map_err(|e| format!("readline init: {e}"))?;
        println!(
            "vf-clide — model: {} · project: {}",
            self.client.model,
            self.project.as_deref().unwrap_or("(none)"),
        );
        println!(
            "commands: /quit /clear /model <name> /max-tokens <N> /think /no-think · \
             memory: /project [key] /recall <query> /remember <text> /archive <id> \
             /unarchive <id> /forget <id>",
        );
        // Pinned status line — only when stdout is a TTY (no-op otherwise).
        let bar = StatusBar::new(std::io::stdout().is_terminal());
        bar.enter();
        loop {
            bar.set_action("idle");
            let line = match rl.readline("> ") {
                Ok(l) => l,
                Err(_) => break, // Ctrl-D / Ctrl-C at the prompt → exit
            };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let _ = rl.add_history_entry(line.as_str());

            if let Some(cmd) = parse_command(&line) {
                match cmd {
                    Command::Quit => break,
                    Command::Clear => {
                        self.history.clear();
                        println!("(history cleared)");
                    }
                    Command::Model(m) => {
                        self.client.model = m.clone();
                        println!("(model → {m})");
                    }
                    Command::Think(on) => {
                        self.no_think = !on;
                        println!("(thinking {})", if on { "on" } else { "off (/no_think appended)" });
                    }
                    Command::MaxTokens(n) => {
                        self.client.max_tokens = Some(n);
                        println!("(max-tokens → {n})");
                    }
                    // Memory commands (Stufe B-1) — direct user actions, so no
                    // permission ceiling (the ceiling guards autonomous tool/
                    // shell calls, not what the user types). They call the
                    // server's `/memory/*` endpoints and display the result.
                    Command::Project(None) => self.show_project().await,
                    Command::Project(Some(key)) => {
                        self.project = Some(key.clone());
                        println!("(project → {key})");
                    }
                    Command::Recall(query) => self.recall(&query).await,
                    Command::Remember(text) => self.remember(&text).await,
                    Command::Archive(id) => self.archive(id).await,
                    Command::Unarchive(id) => self.unarchive(id).await,
                    Command::Forget(id) => self.forget(id).await,
                    Command::Unknown(hint) => println!("({hint})"),
                }
                continue;
            }

            // Agent turn — tool-calling loop with interactive permission.
            // Interactive mode prompts y/N per call (mutating tools get a
            // visible warning), so no `--allow-mutating` is needed here.
            // The loop pushes `thinking…` / `running <tool>(…)` to the bar.
            if self.agent {
                let msgs = self.build_messages(&line);
                let gate = crate::agent::Gate::interactive(self.ceiling);
                let fut = crate::agent::run(
                    &self.client,
                    msgs,
                    gate,
                    &self.workspace,
                    self.project.as_deref(),
                    Some(&bar),
                );
                let res = tokio::select! {
                    r = fut => r,
                    _ = tokio::signal::ctrl_c() => {
                        println!("\n(interrupted)");
                        bar.leave();
                        std::process::exit(130);
                    }
                };
                match res {
                    Ok((crate::agent::LoopEnd::Final { content, finish_reason }, usage)) => {
                        let text = content.unwrap_or_default();
                        println!("{text}");
                        if let Some(m) = empty_notice(&text) {
                            eprintln!("{m}");
                        } else if let Some(m) =
                            truncation_notice(finish_reason.as_deref(), self.client.max_tokens)
                        {
                            eprintln!("{m}");
                        }
                        self.history.push(ChatMessage::user(line));
                        self.history.push(ChatMessage::assistant(strip_think(&text)));
                        bar.record_turn(usage);
                    }
                    Ok((crate::agent::LoopEnd::CapReached, usage)) => {
                        eprintln!(
                            "[agent] stopped: reached the tool-call loop cap ({}). \
                             The task may be incomplete.",
                            crate::agent::LOOP_CAP
                        );
                        // Keep the user turn in history so context isn't lost.
                        self.history.push(ChatMessage::user(line));
                        bar.record_turn(usage);
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }

            // Normal turn — stream the answer live.
            bar.set_action("generating…");
            let msgs = self.build_messages(&line);
            let mut stdout = std::io::stdout();
            let fut = self.client.chat_stream(msgs, |t| {
                print!("{t}");
                let _ = stdout.flush();
            });
            let result = tokio::select! {
                r = fut => r,
                _ = tokio::signal::ctrl_c() => {
                    println!();
                    bar.leave();
                    std::process::exit(130);
                }
            };
            println!();
            match result {
                Ok(o) => {
                    // Surface empty / truncated answers (instead of a
                    // silent blank line or a cut-off sentence).
                    if let Some(m) = empty_notice(&o.text) {
                        eprintln!("{m}");
                    } else if let Some(m) =
                        truncation_notice(o.finish_reason.as_deref(), self.client.max_tokens)
                    {
                        eprintln!("{m}");
                    }
                    // Store the turn (raw user line; /no_think is re-applied
                    // per-turn by build_messages). Strip prior-turn <think>.
                    self.history.push(ChatMessage::user(line));
                    self.history.push(ChatMessage::assistant(strip_think(&o.text)));
                    bar.record_turn(o.usage.unwrap_or_default());
                }
                Err(e) => eprintln!("error: {e}"),
            }
        }
        bar.leave();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NoopMemory;

    #[test]
    fn constitution_is_messages0_in_agent_mode_only() {
        // Agent mode + a system prompt → it leads as messages[0].
        let client = Client::new("http://localhost:0", "m");
        let repl = Repl::new(client, Box::new(NoopMemory), None)
            .with_agent(true)
            .with_system(Some("CONSTITUTION".into()));
        let msgs = repl.build_messages("hi");
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "CONSTITUTION");
        assert_eq!(msgs.last().unwrap().role, "user");

        // Plain chat mode → NO system prepend (chat path unchanged).
        let client2 = Client::new("http://localhost:0", "m");
        let chat = Repl::new(client2, Box::new(NoopMemory), None)
            .with_system(Some("CONSTITUTION".into())); // agent=false
        let m2 = chat.build_messages("hi");
        assert_eq!(m2.len(), 1);
        assert_eq!(m2[0].role, "user");
    }

    #[test]
    fn build_messages_accumulates_history() {
        // Deterministic test of the REPL's multi-turn mechanism (no server).
        let client = Client::new("http://localhost:0", "m");
        let mut repl = Repl::new(client, Box::new(NoopMemory), None);
        repl.history.push(ChatMessage::user("turn1"));
        repl.history.push(ChatMessage::assistant("ans1"));

        let msgs = repl.build_messages("turn2");
        // NoopMemory injects nothing → just history(2) + new user(1).
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], ChatMessage::user("turn1"));
        assert_eq!(msgs[1], ChatMessage::assistant("ans1"));
        assert_eq!(msgs[2], ChatMessage::user("turn2"));
    }

    #[test]
    fn no_think_appends_directive_to_content() {
        let client = Client::new("http://localhost:0", "m");
        let repl = Repl::new(client, Box::new(NoopMemory), None).with_no_think(true);
        let msgs = repl.build_messages("explain mutex");
        assert_eq!(msgs.last().unwrap().content, "explain mutex /no_think");
        // ...and NOT when thinking is on.
        let client2 = Client::new("http://localhost:0", "m");
        let repl2 = Repl::new(client2, Box::new(NoopMemory), None);
        assert_eq!(repl2.build_messages("explain mutex").last().unwrap().content, "explain mutex");
    }

    #[test]
    fn think_and_max_tokens_commands() {
        assert_eq!(parse_command("/think"), Some(Command::Think(true)));
        assert_eq!(parse_command("/no-think"), Some(Command::Think(false)));
        assert_eq!(parse_command("/max-tokens 4096"), Some(Command::MaxTokens(4096)));
        assert!(matches!(parse_command("/max-tokens"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/max-tokens abc"), Some(Command::Unknown(_))));
    }

    #[test]
    fn normal_input_is_not_a_command() {
        assert_eq!(parse_command("hello world"), None);
        assert_eq!(parse_command("what is 2+2?"), None);
    }

    #[test]
    fn quit_aliases() {
        assert_eq!(parse_command("/quit"), Some(Command::Quit));
        assert_eq!(parse_command("/q"), Some(Command::Quit));
        assert_eq!(parse_command("/exit"), Some(Command::Quit));
    }

    #[test]
    fn clear_command() {
        assert_eq!(parse_command("/clear"), Some(Command::Clear));
    }

    #[test]
    fn model_command_with_and_without_arg() {
        assert_eq!(parse_command("/model qwen3-14b"), Some(Command::Model("qwen3-14b".into())));
        assert_eq!(parse_command("/model   spaced-name "), Some(Command::Model("spaced-name".into())));
        assert!(matches!(parse_command("/model"), Some(Command::Unknown(_))));
    }

    #[test]
    fn unknown_slash_command() {
        assert!(matches!(parse_command("/bogus"), Some(Command::Unknown(_))));
    }

    #[test]
    fn project_command_show_and_switch() {
        assert_eq!(parse_command("/project"), Some(Command::Project(None)));
        assert_eq!(parse_command("/project foo"), Some(Command::Project(Some("foo".into()))));
        assert_eq!(parse_command("/project   spaced "), Some(Command::Project(Some("spaced".into()))));
    }

    #[test]
    fn recall_command_needs_a_query() {
        assert_eq!(parse_command("/recall do fewer barriers help?"),
            Some(Command::Recall("do fewer barriers help?".into())));
        assert!(matches!(parse_command("/recall"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/recall   "), Some(Command::Unknown(_))));
    }

    #[test]
    fn remember_command_needs_text() {
        assert_eq!(parse_command("/remember dispatch reduction did not help"),
            Some(Command::Remember("dispatch reduction did not help".into())));
        assert!(matches!(parse_command("/remember"), Some(Command::Unknown(_))));
    }

    #[test]
    fn archive_unarchive_and_forget_parse_numeric_id() {
        assert_eq!(parse_command("/archive 7"), Some(Command::Archive(7)));
        assert_eq!(parse_command("/unarchive 7"), Some(Command::Unarchive(7)));
        assert_eq!(parse_command("/forget 42"), Some(Command::Forget(42)));
        // missing or non-numeric id → usage hint, not a crash.
        assert!(matches!(parse_command("/archive"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/unarchive"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/unarchive xy"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/forget abc"), Some(Command::Unknown(_))));
    }

    #[test]
    fn snippet_caps_long_text_and_flattens_newlines() {
        assert_eq!(snippet("short  line"), "short  line");
        assert_eq!(snippet("a\nb"), "a b");
        let long = "x".repeat(200);
        let s = snippet(&long);
        assert!(s.chars().count() <= 80);
        assert!(s.ends_with('…'));
    }
}
