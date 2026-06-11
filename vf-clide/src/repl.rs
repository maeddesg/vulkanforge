// SPDX-License-Identifier: GPL-3.0-only
//! Interactive REPL: streamed chat with in-session history.
//!
//! Commands: `/quit` (`/q`, `/exit`), `/clear` (drop history),
//! `/model <name>` (switch model). Anything else is sent to the model.

use std::io::Write;

use rustyline::DefaultEditor;

use crate::client::{Client, Result};
use crate::memory::Memory;
use crate::types::{strip_think, ChatMessage};

/// Parsed REPL command. `None` from [`parse_command`] means "not a
/// command — treat the line as a chat prompt".
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Quit,
    Clear,
    Model(String),
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
}

impl Repl {
    pub fn new(client: Client, memory: Box<dyn Memory>, project: Option<String>) -> Self {
        Self { client, memory, project, history: Vec::new() }
    }

    /// Build the message list for one turn: optional memory context
    /// (no-op in Phase 1) + prior history + the new user input. This is
    /// the single point where the memory seam is consulted.
    pub fn build_messages(&self, user_input: &str) -> Vec<ChatMessage> {
        let mut msgs = Vec::with_capacity(self.history.len() + 2);
        if let Some(ctx) = self.memory.context_for(self.project.as_deref(), &self.history) {
            msgs.push(ChatMessage::system(ctx));
        }
        msgs.extend(self.history.iter().cloned());
        msgs.push(ChatMessage::user(user_input));
        msgs
    }

    pub async fn run(&mut self) -> Result<()> {
        let mut rl = DefaultEditor::new().map_err(|e| format!("readline init: {e}"))?;
        println!(
            "vf-clide — model: {} · commands: /quit /clear /model <name>",
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
                    Command::Unknown(hint) => println!("({hint})"),
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
                Ok(text) => {
                    // Store the turn. Strip prior-turn <think> from the
                    // assistant reply so it doesn't bloat later context.
                    self.history.push(ChatMessage::user(line));
                    self.history.push(ChatMessage::assistant(strip_think(&text)));
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
