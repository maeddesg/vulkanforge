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
use crate::types::{strip_think, ChatMessage, RecallResponse, ToolRisk};

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
    /// `/recall <query> [--explain] [--type <T>] [--include-superseded]
    /// [--frontier]` — manual semantic recall against the current project's
    /// memory; displays hits (no auto-inject — that is B-2). `--explain` adds
    /// the diagnostics view; `--type <T>` filters to one layer type;
    /// `--include-superseded` surfaces stale (superseded) notes; `--frontier`
    /// reserves slots for `DERIVES_FROM`-linked evidence below the similarity
    /// cut (opt-in; default recall is unchanged).
    Recall { query: String, explain: bool, note_type: Option<String>, include_superseded: bool, frontier: bool },
    /// `/remember [--type <T>] <text>` — store a manual note (`kind:"Note"`).
    /// `--type` sets the layer type (else untyped).
    Remember { text: String, note_type: Option<String> },
    /// `/retype <id> <T>` — set an existing note's layer type (user curation).
    Retype { id: i64, note_type: String },
    /// `/supersede <new_id> <old_id>` — record that `new_id` replaces `old_id`
    /// (old becomes stale, suppressed from recall). User curation.
    Supersede { new_id: i64, old_id: i64 },
    /// `/unsupersede <new_id> <old_id>` — release a supersession (old returns
    /// to recall). Inverse of `/supersede`.
    Unsupersede { new_id: i64, old_id: i64 },
    /// `/derive <A> from <B> [<C> …]` — record that A is anchored in B[,C…]
    /// (Why-Graph). Never changes recall.
    Derive { from_id: i64, to_ids: Vec<i64> },
    /// `/underive <A> from <B>` — release a derivation. Inverse of `/derive`.
    Underive { from_id: i64, to_id: i64 },
    /// `/why <id>` — print the Why-Graph justification tree for a note.
    Why { id: i64 },
    /// `/contradict <a> <b>` — record that two notes conflict (`CONTRADICTS`,
    /// symmetric). Flagged in `--explain`, never suppressed; resolve via
    /// `/supersede`. User curation.
    Contradict { a: i64, b: i64 },
    /// `/uncontradict <a> <b>` — release a contradiction. Inverse of
    /// `/contradict` (direction-independent).
    Uncontradict { a: i64, b: i64 },
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
        "/recall" => {
            let (query, explain, note_type, include_superseded, frontier) =
                parse_recall_flags(arg.as_deref().unwrap_or(""));
            if query.is_empty() {
                Command::Unknown(
                    "usage: /recall <query> [--explain] [--type <T>] [--include-superseded] [--frontier]".into(),
                )
            } else {
                Command::Recall { query, explain, note_type, include_superseded, frontier }
            }
        }
        "/remember" => match arg {
            // Optional leading `--type <T>`; the rest is the note text verbatim
            // (text whitespace preserved — only a leading type flag is stripped).
            Some(a) => match a.strip_prefix("--type ") {
                Some(rest) => {
                    let mut it = rest.splitn(2, char::is_whitespace);
                    let t = it.next().unwrap_or("").trim().to_string();
                    let text = it.next().map(|s| s.trim().to_string()).unwrap_or_default();
                    if t.is_empty() || text.is_empty() {
                        Command::Unknown("usage: /remember [--type <T>] <text>".into())
                    } else {
                        Command::Remember { text, note_type: Some(t) }
                    }
                }
                None => Command::Remember { text: a, note_type: None },
            },
            None => Command::Unknown("usage: /remember [--type <T>] <text>".into()),
        },
        "/retype" => {
            let parts: Vec<&str> = arg.as_deref().unwrap_or("").split_whitespace().collect();
            match (parts.first().and_then(|s| s.parse::<i64>().ok()), parts.get(1)) {
                (Some(id), Some(t)) => Command::Retype { id, note_type: (*t).to_string() },
                _ => Command::Unknown("usage: /retype <id> <type>  (id from /recall)".into()),
            }
        }
        "/supersede" => match two_ids(arg.as_deref()) {
            Some((new_id, old_id)) => Command::Supersede { new_id, old_id },
            None => Command::Unknown("usage: /supersede <new_id> <old_id>  (new replaces old)".into()),
        },
        "/unsupersede" => match two_ids(arg.as_deref()) {
            Some((new_id, old_id)) => Command::Unsupersede { new_id, old_id },
            None => Command::Unknown("usage: /unsupersede <new_id> <old_id>".into()),
        },
        "/derive" => match parse_derive(arg.as_deref()) {
            Some((from_id, to_ids)) => Command::Derive { from_id, to_ids },
            None => Command::Unknown("usage: /derive <id> from <id> [<id> …]  (A anchored in B…)".into()),
        },
        "/underive" => {
            // `/underive <A> from <B>` (single link).
            let toks: Vec<&str> = arg.as_deref().unwrap_or("").split_whitespace().collect();
            match (toks.first().and_then(|s| s.parse::<i64>().ok()), toks.get(1), toks.get(2).and_then(|s| s.parse::<i64>().ok())) {
                (Some(from_id), Some(&"from"), Some(to_id)) => Command::Underive { from_id, to_id },
                _ => Command::Unknown("usage: /underive <id> from <id>".into()),
            }
        }
        "/why" => match arg.as_deref().and_then(|a| a.trim().parse::<i64>().ok()) {
            Some(id) => Command::Why { id },
            None => Command::Unknown("usage: /why <id>  (id from /recall)".into()),
        },
        "/contradict" => match two_ids(arg.as_deref()) {
            Some((a, b)) => Command::Contradict { a, b },
            None => Command::Unknown("usage: /contradict <id> <id>  (the two notes conflict)".into()),
        },
        "/uncontradict" => match two_ids(arg.as_deref()) {
            Some((a, b)) => Command::Uncontradict { a, b },
            None => Command::Unknown("usage: /uncontradict <id> <id>".into()),
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

/// Parse `/recall` flags out of the arg: `--explain` (bool), `--type <T>`
/// (value), `--include-superseded` (bool), `--frontier` (bool). Returns
/// `(query, explain, note_type, include_superseded, frontier)` with all flags
/// removed from the query text. Query whitespace is collapsed (it's embedded —
/// insensitive).
fn parse_recall_flags(arg: &str) -> (String, bool, Option<String>, bool, bool) {
    let toks: Vec<&str> = arg.split_whitespace().collect();
    let mut explain = false;
    let mut note_type = None;
    let mut include_superseded = false;
    let mut frontier = false;
    let mut rest: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            "--explain" => explain = true,
            "--include-superseded" => include_superseded = true,
            "--frontier" => frontier = true,
            "--type" => {
                note_type = toks.get(i + 1).map(|s| s.to_string());
                i += 1; // also consume the value token
            }
            other => rest.push(other),
        }
        i += 1;
    }
    (rest.join(" "), explain, note_type, include_superseded, frontier)
}

/// Parse two whitespace-separated note ids (`<new_id> <old_id>`) for the
/// supersede commands. `None` if either is missing or not an integer.
fn two_ids(arg: Option<&str>) -> Option<(i64, i64)> {
    let p: Vec<&str> = arg.unwrap_or("").split_whitespace().collect();
    Some((p.first()?.parse().ok()?, p.get(1)?.parse().ok()?))
}

/// Parse `<A> from <B> [<C> …]` for the derive commands → `(A, [B, C, …])`.
/// `None` if `A` isn't an int, the `from` keyword is missing, or no source ids.
fn parse_derive(arg: Option<&str>) -> Option<(i64, Vec<i64>)> {
    let toks: Vec<&str> = arg.unwrap_or("").split_whitespace().collect();
    let from = toks.first()?.parse::<i64>().ok()?;
    if *toks.get(1)? != "from" {
        return None;
    }
    let tos: Vec<i64> = toks[2..].iter().filter_map(|s| s.parse().ok()).collect();
    if tos.is_empty() {
        return None;
    }
    Some((from, tos))
}

/// `"  ↑ frontier via #4"` for a hit the `--frontier` mode pulled into a
/// reserved slot, or `""` for a plain similarity (seed) hit. Marks *what* the
/// frontier rescued so the user can judge whether it earned its slot.
fn frontier_suffix(frontier_via: Option<i64>) -> String {
    match frontier_via {
        Some(seed) => format!("  \u{2191} frontier via #{seed}"),
        None => String::new(),
    }
}

/// `"  ⚠ conflicts with #3, #7"` for a hit that is party to a `CONTRADICTS`
/// edge, or `""` when it conflicts with nothing. **Awareness only** — symmetric,
/// no side is suppressed; the user reconciles with `/supersede`.
fn conflicts_suffix(conflicts_with: &[i64]) -> String {
    if conflicts_with.is_empty() {
        String::new()
    } else {
        let ids: Vec<String> = conflicts_with.iter().map(|id| format!("#{id}")).collect();
        format!("  \u{26a0} conflicts with {}", ids.join(", "))
    }
}

/// `" · derives from #3, #7"` for the `--explain` line, or `""` when the note
/// has no `DERIVES_FROM` edges. Awareness only — recall is unchanged.
fn derives_suffix(derives_from: &[i64]) -> String {
    if derives_from.is_empty() {
        String::new()
    } else {
        let ids: Vec<String> = derives_from.iter().map(|id| format!("#{id}")).collect();
        format!("  · derives from {}", ids.join(", "))
    }
}

/// Render a `/why` justification tree as an indented outline. Each line is a
/// note (`#id [type] snippet`) with `(cycle)` / `(…depth cap)` markers.
fn format_why(tree: &crate::types::WhyNode) -> String {
    fn walk(out: &mut String, n: &crate::types::WhyNode, depth: usize) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        let mark = if n.cycle {
            "  (cycle — already shown)"
        } else if n.truncated {
            "  (… depth cap)"
        } else {
            ""
        };
        let _ = writeln!(out, "{indent}#{} [{}] {}{mark}", n.id, n.note_type, snippet(&n.text));
        for child in &n.derives_from {
            walk(out, child, depth + 1);
        }
    }
    let mut out = String::new();
    walk(&mut out, tree, 0);
    if tree.derives_from.is_empty() && !tree.cycle && !tree.truncated {
        out.push_str("  (no derivation recorded — nothing this note is anchored in)\n");
    }
    out
}

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

/// Render the `recall --explain` retrieval-diagnostics view: the embedded
/// query + explicit cutoff, the RETURNED hits (rank · score · id), the
/// NEAR-MISS candidates that fell just outside the cutoff, and the score
/// separation between them. Surfaces existing signal only — no new scoring.
/// Pure (returns the block) so the format is unit-testable.
fn format_explain(query: &str, resp: &RecallResponse) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let Some(ex) = resp.explain.as_ref() else {
        // Server returned no explain block (e.g. a pre-explain server). Be
        // honest rather than rendering an empty diagnostic.
        let _ = writeln!(out, "recall \"{query}\": server did not return explain data (needs a newer engine)");
        return out;
    };
    let threshold = match ex.threshold {
        Some(t) => format!("{t:.3}"),
        None => "none (pure top-k ranking)".to_string(),
    };
    let _ = writeln!(out, "recall \"{}\"  (query embedded · {}-dim)", query, ex.query_dim);
    let _ = writeln!(out, "cutoff: top-k = {}   threshold: {}", ex.top_k, threshold);

    let _ = writeln!(out, "RETURNED ({}):", resp.hits.len());
    if resp.hits.is_empty() {
        let _ = writeln!(out, "  (nothing — the index is empty for this scope)");
    } else {
        for (i, h) in resp.hits.iter().enumerate() {
            let _ = writeln!(
                out, "  {:>2}. [{} · {:.3}]  {}   (id {}){}{}{}",
                i + 1, h.note_type, h.score, snippet(&h.text), h.id,
                frontier_suffix(h.frontier_via), derives_suffix(&h.derives_from),
                conflicts_suffix(&h.conflicts_with),
            );
        }
    }

    if ex.near_miss.is_empty() {
        let _ = writeln!(out, "NEAR-MISS: none (nothing fell outside the cut for this scope)");
    } else {
        let _ = writeln!(out, "NEAR-MISS ({}):", ex.near_miss.len());
        for (i, nm) in ex.near_miss.iter().enumerate() {
            let h = &nm.hit;
            let rank = resp.hits.len() + i + 1;
            // Label the gate that cut it: below the relevance threshold, or
            // beyond the top-k cap. (Server sends "threshold" / "top-k".)
            // Label the gate that cut it. For `superseded`, name the superseder;
            // for `frontier-withheld` (Edge-Type-Priors negative signal), name
            // the contesting seed (`⚠ frontier withheld — contested by #seed`).
            let cut = match (nm.cut.as_str(), h.superseded_by, h.contested_by) {
                ("superseded", Some(by), _) => format!("superseded by #{by}"),
                ("frontier-withheld", _, Some(seed)) => format!("\u{26a0} frontier withheld — contested by #{seed}"),
                ("frontier-withheld", _, None) => "\u{26a0} frontier withheld".to_string(),
                ("", _, _) => "top-k".to_string(),
                (other, _, _) => other.to_string(),
            };
            let _ = writeln!(
                out, "  {:>2}. [{} · {:.3}]  {}   (id {}, cut: {}){}{}",
                rank, h.note_type, h.score, snippet(&h.text), h.id, cut,
                derives_suffix(&h.derives_from), conflicts_suffix(&h.conflicts_with),
            );
        }
    }

    // Conflict pairs with BOTH parties in the returned set — the high-value
    // case: the contradiction is visible side by side, ready to reconcile.
    let returned_ids: std::collections::HashSet<i64> = resp.hits.iter().map(|h| h.id).collect();
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for h in &resp.hits {
        for &other in &h.conflicts_with {
            if h.id < other && returned_ids.contains(&other) {
                pairs.push((h.id, other));
            }
        }
    }
    pairs.sort();
    pairs.dedup();
    if !pairs.is_empty() {
        let _ = writeln!(out, "CONFLICT PAIRS ({}):", pairs.len());
        for (a, b) in &pairs {
            let _ = writeln!(out, "  \u{26a0} conflict pair: #{a} \u{2194} #{b}  (resolve with /supersede)");
        }
    }

    match ex.separation {
        Some(gap) => {
            let verdict = if gap >= 0.05 {
                "clean separation"
            } else {
                "tight — cutoff sits in a dense region"
            };
            let _ = writeln!(out, "separation: {gap:.3}  (last hit \u{2212} first near-miss) \u{2014} {verdict}");
        }
        None => {
            let _ = writeln!(out, "separation: n/a (need both a returned hit and a near-miss)");
        }
    }
    out
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

    /// `/recall <query> [--explain] [--type <T>] [--frontier]`: display the
    /// current project's matches (no auto-inject). Shows `id` + layer `type` per
    /// hit so it can be targeted by `/archive`/`/forget`/`/retype`. `--explain`
    /// renders the retrieval-diagnostics view; `--type <T>` filters to one layer
    /// type; `--frontier` reserves slots for `DERIVES_FROM`-linked evidence
    /// below the similarity cut (a frontier hit is marked `↑ via #seed`).
    async fn recall(&self, query: &str, explain: bool, note_type: Option<&str>, include_superseded: bool, frontier: bool) {
        let call = self
            .client
            .memory_recall_opts(self.project.as_deref(), query, 5, explain, note_type, include_superseded, frontier)
            .await;
        match call {
            Ok(MemCall::Ok(resp)) => {
                if explain {
                    print!("{}", format_explain(query, &resp));
                } else if resp.hits.is_empty() {
                    println!("(no matches)");
                } else {
                    for h in &resp.hits {
                        println!("  #{} [{} · {:.2}] {}{}", h.id, h.note_type, h.score, snippet(&h.text), frontier_suffix(h.frontier_via));
                    }
                }
            }
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/remember [--type <T>] <text>`: store a manual note (`kind:"Note"`),
    /// optionally typed. Honest about a dedup hit ("already known").
    async fn remember(&self, text: &str, note_type: Option<&str>) {
        match self.client.memory_remember_typed(self.project.as_deref(), "Note", text, note_type).await {
            Ok(MemCall::Ok(resp)) if resp.deduped => println!("(already known, id {})", resp.id),
            Ok(MemCall::Ok(resp)) => match note_type {
                Some(t) => println!("(stored, id {}, type {t})", resp.id),
                None => println!("(stored, id {})", resp.id),
            },
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/retype <id> <T>`: set a note's layer type (user curation, pure
    /// metadata). Surfaces the server's typed errors (unknown type → 400,
    /// missing id → 404) as the printed error.
    async fn retype(&self, id: i64, note_type: &str) {
        match self.client.memory_retype(self.project.as_deref(), id, note_type).await {
            Ok(MemCall::Ok(resp)) => println!("(retyped id {} → {})", resp.id, resp.note_type),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/supersede <new> <old>`: record that `new` replaces `old` — `old`
    /// becomes stale and drops out of recall (reversible via `/unsupersede`).
    async fn supersede(&self, new_id: i64, old_id: i64) {
        match self.client.memory_supersede(self.project.as_deref(), new_id, old_id).await {
            Ok(MemCall::Ok(_)) => println!("(superseded: #{old_id} replaced by #{new_id} — out of recall, recoverable)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/unsupersede <new> <old>`: release the supersession — `old` returns to
    /// recall (inverse of `/supersede`).
    async fn unsupersede(&self, new_id: i64, old_id: i64) {
        match self.client.memory_unsupersede(self.project.as_deref(), new_id, old_id).await {
            Ok(MemCall::Ok(_)) => println!("(unsuperseded: #{old_id} restored to recall)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/derive <A> from <B> [<C> …]`: record that A is anchored in B[,C…]
    /// (Why-Graph). Never changes recall — additive awareness only.
    async fn derive(&self, from_id: i64, to_ids: Vec<i64>) {
        let label: Vec<String> = to_ids.iter().map(|id| format!("#{id}")).collect();
        match self.client.memory_derive(self.project.as_deref(), from_id, to_ids).await {
            Ok(MemCall::Ok(_)) => println!("(#{from_id} derives from {} — recorded, recall unchanged)", label.join(", ")),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/underive <A> from <B>`: release a derivation (inverse of `/derive`).
    async fn underive(&self, from_id: i64, to_id: i64) {
        match self.client.memory_underive(self.project.as_deref(), from_id, to_id).await {
            Ok(MemCall::Ok(_)) => println!("(released: #{from_id} no longer derives from #{to_id})"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/contradict <a> <b>`: record that two notes conflict (symmetric). Never
    /// suppresses — surfaced in `--explain`; resolve with `/supersede`.
    async fn contradict(&self, a: i64, b: i64) {
        match self.client.memory_contradict(self.project.as_deref(), a, b).await {
            Ok(MemCall::Ok(_)) => println!("(conflict recorded: #{a} ↔ #{b} — flagged in --explain, recall unchanged; resolve with /supersede)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/uncontradict <a> <b>`: release a contradiction (inverse of
    /// `/contradict`, direction-independent).
    async fn uncontradict(&self, a: i64, b: i64) {
        match self.client.memory_uncontradict(self.project.as_deref(), a, b).await {
            Ok(MemCall::Ok(_)) => println!("(released: #{a} and #{b} no longer flagged as conflicting)"),
            Ok(MemCall::Disabled) => println!("({MEMORY_OFF_HINT})"),
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `/why <id>`: print the Why-Graph justification tree (read-only).
    async fn why(&self, id: i64) {
        match self.client.memory_why(self.project.as_deref(), id).await {
            Ok(MemCall::Ok(tree)) => print!("{}", format_why(&tree)),
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
             memory: /project [key] /recall <query> [--explain] [--type <T>] \
             [--include-superseded] [--frontier] /remember [--type <T>] <text> /retype <id> <T> \
             /supersede <new> <old> /unsupersede <new> <old> \
             /derive <id> from <id…> /underive <id> from <id> /why <id> \
             /contradict <id> <id> /uncontradict <id> <id> \
             /archive <id> /unarchive <id> /forget <id>",
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
                    Command::Recall { query, explain, note_type, include_superseded, frontier } => {
                        self.recall(&query, explain, note_type.as_deref(), include_superseded, frontier).await
                    }
                    Command::Remember { text, note_type } => {
                        self.remember(&text, note_type.as_deref()).await
                    }
                    Command::Retype { id, note_type } => self.retype(id, &note_type).await,
                    Command::Supersede { new_id, old_id } => self.supersede(new_id, old_id).await,
                    Command::Unsupersede { new_id, old_id } => self.unsupersede(new_id, old_id).await,
                    Command::Derive { from_id, to_ids } => self.derive(from_id, to_ids).await,
                    Command::Underive { from_id, to_id } => self.underive(from_id, to_id).await,
                    Command::Why { id } => self.why(id).await,
                    Command::Contradict { a, b } => self.contradict(a, b).await,
                    Command::Uncontradict { a, b } => self.uncontradict(a, b).await,
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
            Some(Command::Recall { query: "do fewer barriers help?".into(), explain: false, note_type: None, include_superseded: false, frontier: false }));
        assert!(matches!(parse_command("/recall"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/recall   "), Some(Command::Unknown(_))));
    }

    #[test]
    fn recall_flags_parse_and_strip_from_query() {
        // `--explain` anywhere flips the flag and is removed from the query.
        assert_eq!(
            parse_command("/recall magic number --explain"),
            Some(Command::Recall { query: "magic number".into(), explain: true, note_type: None, include_superseded: false, frontier: false }),
        );
        // `--type <T>` consumes its value and is stripped from the query.
        assert_eq!(
            parse_command("/recall design --type decision --explain"),
            Some(Command::Recall { query: "design".into(), explain: true, note_type: Some("decision".into()), include_superseded: false, frontier: false }),
        );
        assert_eq!(
            parse_command("/recall --type working kernel speed"),
            Some(Command::Recall { query: "kernel speed".into(), explain: false, note_type: Some("working".into()), include_superseded: false, frontier: false }),
        );
        // `--include-superseded` is a bool flag, stripped from the query.
        assert_eq!(
            parse_command("/recall old design --include-superseded --type decision"),
            Some(Command::Recall { query: "old design".into(), explain: false, note_type: Some("decision".into()), include_superseded: true, frontier: false }),
        );
        // Without flags they stay default.
        assert_eq!(
            parse_command("/recall magic number"),
            Some(Command::Recall { query: "magic number".into(), explain: false, note_type: None, include_superseded: false, frontier: false }),
        );
        // `--frontier` is a bool flag, stripped from the query, default false.
        assert_eq!(
            parse_command("/recall kv reuse rationale --frontier"),
            Some(Command::Recall { query: "kv reuse rationale".into(), explain: false, note_type: None, include_superseded: false, frontier: true }),
        );
        // `--frontier` composes with `--explain` and `--type`, order-insensitive.
        assert_eq!(
            parse_command("/recall design --frontier --explain --type decision"),
            Some(Command::Recall { query: "design".into(), explain: true, note_type: Some("decision".into()), include_superseded: false, frontier: true }),
        );
        // Flags with no query is a usage error, not an empty recall.
        assert!(matches!(parse_command("/recall --explain"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/recall --type decision"), Some(Command::Unknown(_))));
    }

    #[test]
    fn supersede_commands_parse_two_ids() {
        assert_eq!(parse_command("/supersede 9 5"), Some(Command::Supersede { new_id: 9, old_id: 5 }));
        assert_eq!(parse_command("/unsupersede 9 5"), Some(Command::Unsupersede { new_id: 9, old_id: 5 }));
        assert!(matches!(parse_command("/supersede 9"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/supersede a b"), Some(Command::Unknown(_))));
    }

    #[test]
    fn derive_underive_why_parse() {
        // /derive <A> from <B> [<C> …]
        assert_eq!(parse_command("/derive 5 from 3 7"), Some(Command::Derive { from_id: 5, to_ids: vec![3, 7] }));
        assert_eq!(parse_command("/derive 5 from 3"), Some(Command::Derive { from_id: 5, to_ids: vec![3] }));
        assert!(matches!(parse_command("/derive 5 3 7"), Some(Command::Unknown(_))), "missing 'from'");
        assert!(matches!(parse_command("/derive 5 from"), Some(Command::Unknown(_))), "no sources");
        // /underive <A> from <B>
        assert_eq!(parse_command("/underive 5 from 3"), Some(Command::Underive { from_id: 5, to_id: 3 }));
        assert!(matches!(parse_command("/underive 5 3"), Some(Command::Unknown(_))));
        // /why <id>
        assert_eq!(parse_command("/why 5"), Some(Command::Why { id: 5 }));
        assert!(matches!(parse_command("/why"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/why abc"), Some(Command::Unknown(_))));
    }

    #[test]
    fn contradict_commands_parse_two_ids() {
        assert_eq!(parse_command("/contradict 9 5"), Some(Command::Contradict { a: 9, b: 5 }));
        assert_eq!(parse_command("/uncontradict 9 5"), Some(Command::Uncontradict { a: 9, b: 5 }));
        // missing / non-numeric id → usage hint, not a crash.
        assert!(matches!(parse_command("/contradict 9"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/contradict a b"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/uncontradict"), Some(Command::Unknown(_))));
    }

    #[test]
    fn format_explain_flags_conflicts_and_conflict_pairs() {
        use crate::types::{ExplainInfo, MemoryHit, RecallResponse};
        let mk = |id: i64, score: f32, text: &str, conflicts: Vec<i64>| MemoryHit {
            id, kind: "fact".into(), name: String::new(), text: text.into(),
            status: "active".into(), note_type: "decision".into(), superseded_by: None,
            derives_from: Vec::new(), frontier_via: None, conflicts_with: conflicts, contested_by: None, score,
        };
        // #1 and #2 conflict and are BOTH returned (the pair case); #3 conflicts
        // with #9 which is NOT in the result (flag only, no pair).
        let resp = RecallResponse {
            hits: vec![
                mk(1, 0.95, "We default KV reuse to on", vec![2]),
                mk(2, 0.92, "We keep KV reuse off for safety", vec![1]),
                mk(3, 0.70, "Sliding window masks tokens", vec![9]),
            ],
            explain: Some(ExplainInfo {
                top_k: 3, threshold: None, query_dim: 768, near_miss: vec![], separation: None,
            }),
        };
        let out = format_explain("kv reuse default", &resp);
        assert!(out.contains("\u{26a0} conflicts with #2"), "hit flags its conflict partner: {out}");
        assert!(out.contains("\u{26a0} conflicts with #1"), "symmetric flag on the other side: {out}");
        assert!(out.contains("\u{26a0} conflicts with #9"), "flags a partner outside the result too: {out}");
        // The pair section appears ONCE for #1↔#2 (both present), not for #3↔#9.
        assert!(out.contains("CONFLICT PAIRS (1)"), "exactly one in-result pair: {out}");
        assert!(out.contains("conflict pair: #1 \u{2194} #2"), "names the pair both ways: {out}");
        assert!(!out.contains("#3 \u{2194}") && !out.contains("\u{2194} #9"), "no pair when one party is absent: {out}");
        assert!(out.contains("resolve with /supersede"), "points to the resolution path: {out}");
    }

    #[test]
    fn format_explain_renders_frontier_withheld_with_contesting_seed() {
        use crate::types::{ExplainInfo, MemoryHit, NearMiss, RecallResponse};
        let base = |id: i64, score: f32, text: &str| MemoryHit {
            id, kind: "fact".into(), name: String::new(), text: text.into(),
            status: "active".into(), note_type: "working".into(), superseded_by: None,
            derives_from: Vec::new(), frontier_via: None, conflicts_with: Vec::new(),
            contested_by: None, score,
        };
        // A frontier candidate withheld because it contests seed #7.
        let mut withheld = base(4, 0.61, "Contested premise the frontier did not amplify");
        withheld.contested_by = Some(7);
        let resp = RecallResponse {
            hits: vec![base(7, 0.95, "The seed that the candidate contradicts")],
            explain: Some(ExplainInfo {
                top_k: 3, threshold: None, query_dim: 768,
                near_miss: vec![NearMiss { hit: withheld, cut: "frontier-withheld".into() }],
                separation: None,
            }),
        };
        let out = format_explain("kv reuse rationale", &resp);
        assert!(out.contains("frontier withheld \u{2014} contested by #7"),
            "withheld candidate names the contesting seed: {out}");
        // A withheld candidate with no contested_by still renders honestly.
        let mut bare = base(5, 0.60, "Withheld but seed unknown");
        bare.contested_by = None;
        let resp2 = RecallResponse {
            hits: vec![],
            explain: Some(ExplainInfo {
                top_k: 3, threshold: None, query_dim: 768,
                near_miss: vec![NearMiss { hit: bare, cut: "frontier-withheld".into() }],
                separation: None,
            }),
        };
        assert!(format_explain("q", &resp2).contains("\u{26a0} frontier withheld"),
            "withheld renders even without a named seed");
    }

    #[test]
    fn format_why_renders_indented_tree_with_markers() {
        use crate::types::WhyNode;
        let leaf = |id: i64, text: &str| WhyNode {
            id, kind: "fact".into(), note_type: "working".into(), name: String::new(),
            text: text.into(), derives_from: vec![], cycle: false, truncated: false,
        };
        let tree = WhyNode {
            id: 1, kind: "fact".into(), note_type: "decision".into(), name: String::new(),
            text: "the decision".into(),
            derives_from: vec![leaf(2, "premise two"), leaf(3, "premise three")],
            cycle: false, truncated: false,
        };
        let out = format_why(&tree);
        assert!(out.contains("#1 [decision] the decision"), "root: {out}");
        assert!(out.contains("  #2 [working] premise two"), "indented child: {out}");
        assert!(out.contains("  #3 [working] premise three"), "second child: {out}");

        // Cycle + truncated markers + the empty case.
        let cyc = WhyNode { cycle: true, ..leaf(9, "loop node") };
        assert!(format_why(&cyc).contains("(cycle"), "cycle marker");
        let trunc = WhyNode { truncated: true, ..leaf(9, "deep node") };
        assert!(format_why(&trunc).contains("depth cap"), "truncated marker");
        assert!(format_why(&leaf(9, "lonely")).contains("no derivation recorded"), "empty case");
    }

    #[test]
    fn remember_type_and_retype_parse() {
        // /remember --type <T> <text> sets the type; text is preserved verbatim.
        assert_eq!(
            parse_command("/remember --type decision we chose margin 0.15"),
            Some(Command::Remember { text: "we chose margin 0.15".into(), note_type: Some("decision".into()) }),
        );
        // Without --type → untyped.
        assert_eq!(
            parse_command("/remember plain note"),
            Some(Command::Remember { text: "plain note".into(), note_type: None }),
        );
        // --type without text → usage error.
        assert!(matches!(parse_command("/remember --type decision"), Some(Command::Unknown(_))));
        // /retype <id> <T>.
        assert_eq!(
            parse_command("/retype 7 invariant"),
            Some(Command::Retype { id: 7, note_type: "invariant".into() }),
        );
        assert!(matches!(parse_command("/retype 7"), Some(Command::Unknown(_))));
        assert!(matches!(parse_command("/retype abc decision"), Some(Command::Unknown(_))));
    }

    #[test]
    fn format_explain_renders_hits_near_miss_cutoff_and_separation() {
        use crate::types::{ExplainInfo, MemoryHit, NearMiss, RecallResponse};
        let hit = |id: i64, score: f32, text: &str| MemoryHit {
            id, kind: "fact".into(), name: String::new(), text: text.into(),
            status: "active".into(), note_type: "decision".into(), superseded_by: None,
            derives_from: Vec::new(), frontier_via: None, conflicts_with: Vec::new(), contested_by: None, score,
        };
        let nm = |id: i64, score: f32, text: &str, cut: &str| NearMiss {
            hit: hit(id, score, text), cut: cut.into(),
        };
        // With an active threshold: shows the value + per-near-miss cut label.
        let superseded = NearMiss {
            hit: MemoryHit { superseded_by: Some(7), ..hit(8, 0.90, "An old superseded decision") },
            cut: "superseded".into(),
        };
        let resp = RecallResponse {
            hits: vec![hit(1, 0.96, "The magic number is 391"), hit(2, 0.81, "Another relevant note")],
            explain: Some(ExplainInfo {
                top_k: 2,
                threshold: Some(0.80),
                query_dim: 768,
                near_miss: vec![
                    superseded,
                    nm(3, 0.69, "A noisy note below threshold", "threshold"),
                    nm(4, 0.85, "A relevant note beyond the cap", "top-k"),
                ],
                separation: Some(0.12),
            }),
        };
        let out = format_explain("magic number", &resp);
        assert!(out.contains("768-dim"), "shows embedded query dim: {out}");
        assert!(out.contains("top-k = 2"), "shows the cutoff: {out}");
        assert!(out.contains("threshold: 0.800"), "shows the real threshold value: {out}");
        assert!(out.contains("RETURNED (2)"), "lists returned hits: {out}");
        assert!(out.contains("cut: superseded by #7"), "names the superseder: {out}");
        assert!(out.contains("cut: threshold"), "labels the threshold cut: {out}");
        assert!(out.contains("cut: top-k"), "labels the top-k cut: {out}");
        assert!(out.contains("(id 3"), "near-miss carries its id: {out}");
        assert!(out.contains("separation: 0.120"), "shows the gap: {out}");
        assert!(out.contains("clean separation"), "0.12 gap → clean verdict: {out}");

        // No threshold (margin off) → honest "none" + no near-misses line.
        let resp2 = RecallResponse {
            hits: vec![hit(1, 0.96, "only note")],
            explain: Some(ExplainInfo {
                top_k: 5, threshold: None, query_dim: 768, near_miss: vec![], separation: None,
            }),
        };
        let out2 = format_explain("q", &resp2);
        assert!(out2.contains("none (pure top-k ranking)"), "honest about no threshold: {out2}");
        assert!(out2.contains("NEAR-MISS: none"), "explicit no-near-miss line: {out2}");
        assert!(out2.contains("separation: n/a"), "n/a separation when no near-miss: {out2}");
    }

    #[test]
    fn format_explain_labels_frontier_picks_and_reserved_seeds() {
        use crate::types::{ExplainInfo, MemoryHit, NearMiss, RecallResponse};
        let base = |id: i64, score: f32, text: &str| MemoryHit {
            id, kind: "fact".into(), name: String::new(), text: text.into(),
            status: "active".into(), note_type: "working".into(), superseded_by: None,
            derives_from: Vec::new(), frontier_via: None, conflicts_with: Vec::new(), contested_by: None, score,
        };
        // A returned set with one seed and one frontier-rescued pick (via #1).
        let mut pick = base(4, 0.62, "Linked evidence pulled up by the frontier");
        pick.frontier_via = Some(1);
        let resp = RecallResponse {
            hits: vec![base(1, 0.95, "The seed that anchored the frontier"), pick],
            explain: Some(ExplainInfo {
                top_k: 2,
                threshold: None,
                query_dim: 768,
                // A seed displaced to make room for the frontier pick.
                near_miss: vec![NearMiss {
                    hit: base(2, 0.80, "A seed displaced by the reservation"),
                    cut: "frontier-reserved".into(),
                }],
                separation: Some(-0.18),
            }),
        };
        let out = format_explain("kv reuse rationale", &resp);
        assert!(out.contains("\u{2191} frontier via #1"), "frontier pick labelled with its seed: {out}");
        assert!(out.contains("cut: frontier-reserved"), "displaced seed labelled frontier-reserved: {out}");
        // The plain seed line must NOT carry a frontier marker.
        assert!(out.contains("The seed that anchored the frontier   (id 1)"),
            "seed hit has no frontier marker: {out}");
    }

    #[test]
    fn remember_command_needs_text() {
        assert_eq!(parse_command("/remember dispatch reduction did not help"),
            Some(Command::Remember { text: "dispatch reduction did not help".into(), note_type: None }));
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
