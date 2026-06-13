// SPDX-License-Identifier: GPL-3.0-only
//! Interactive REPL: streamed chat with in-session history.
//!
//! Commands: `/quit` (`/q`, `/exit`), `/clear` (drop history),
//! `/model <name>` (switch model). Anything else is sent to the model.

use std::io::Write;

use rustyline::DefaultEditor;

use crate::client::{empty_notice, truncation_notice, Client, Result};
use crate::memory::Memory;
use crate::types::{strip_think, ChatMessage};

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
        other => Command::Unknown(format!("unknown command: {other}")),
    })
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

    pub async fn run(&mut self) -> Result<()> {
        let mut rl = DefaultEditor::new().map_err(|e| format!("readline init: {e}"))?;
        println!(
            "vf-clide — model: {} · commands: /quit /clear /model <name> /max-tokens <N> /think /no-think",
            self.client.model
        );
        loop {
            let line = match rl.readline("> ") {
                Ok(l) => l,
                Err(_) => break, // Ctrl-D / Ctrl-C → exit
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
                    Command::Unknown(hint) => println!("({hint})"),
                }
                continue;
            }

            // Agent turn — tool-calling loop with interactive permission.
            // Interactive mode prompts y/N per call (mutating tools get a
            // visible warning), so no `--allow-mutating` is needed here.
            if self.agent {
                let msgs = self.build_messages(&line);
                let gate = crate::agent::Gate::interactive();
                match crate::agent::run(&self.client, msgs, gate, &self.workspace).await {
                    Ok(crate::agent::LoopEnd::Final { content, finish_reason }) => {
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
                    }
                    Ok(crate::agent::LoopEnd::CapReached) => {
                        eprintln!(
                            "[agent] stopped: reached the tool-call loop cap ({}). \
                             The task may be incomplete.",
                            crate::agent::LOOP_CAP
                        );
                        // Keep the user turn in history so context isn't lost.
                        self.history.push(ChatMessage::user(line));
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }

            // Normal turn — stream the answer live.
            let msgs = self.build_messages(&line);
            let mut stdout = std::io::stdout();
            let result = self
                .client
                .chat_stream(msgs, |t| {
                    print!("{t}");
                    let _ = stdout.flush();
                })
                .await;
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
                }
                Err(e) => eprintln!("error: {e}"),
            }
        }
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
}
