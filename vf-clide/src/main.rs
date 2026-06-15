// SPDX-License-Identifier: GPL-3.0-only
//! vf-clide binary: interactive REPL by default, or a headless one-shot
//! with `-p/--prompt`. Talks to a running `vulkanforge serve` over HTTP.

use clap::Parser;

use vf_clide::client::Client;
use vf_clide::memory::{Memory, NoopMemory};
use vf_clide::repl::Repl;
use vf_clide::types::ChatMessage;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Parser, Debug)]
#[command(name = "vf-clide", version, about = "Lean CLI client for VulkanForge's OpenAI-compatible API")]
struct Args {
    /// One-shot prompt (headless): print one answer and exit. Omit for
    /// the interactive REPL.
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Base URL of the running VulkanForge server.
    #[arg(long, default_value = "http://localhost:8080")]
    url: String,

    /// Model id to request. Default is the reliable tool-capable coder
    /// (no special env needed).
    #[arg(long, default_value = "Qwen3-14B-Q4_K_M")]
    model: String,

    /// Disable streaming (headless one-shot only).
    #[arg(long)]
    no_stream: bool,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Max generated tokens. Default is generous so a thinking model has
    /// room for the full `<think>` block AND a complete answer. It is a
    /// *cap*, not a target: short answers stop early at EOS and cost
    /// nothing. 6144 is the empirically-measured minimum that lets a
    /// thinking model (Qwen3-14B) finish a real coding task — its think
    /// block alone runs ~5000 tokens, so 4096 yields a think-only/empty
    /// answer (see results/vf_clide_default4096.md).
    #[arg(long, default_value_t = 6144)]
    max_tokens: u32,

    /// Disable thinking: append the `/no_think` directive to the prompt
    /// (Qwen3 convention) so the budget goes to the answer, not reasoning.
    #[arg(long)]
    no_think: bool,

    /// Memory scope (`project_key`) for the REPL `/recall`/`/remember`
    /// commands. Default: derived deterministically from `--workspace`
    /// (so a directory's memory persists across sessions). This flag
    /// overrides that derived key with an explicit one.
    #[arg(long)]
    project: Option<String>,

    /// Opt into the agent loop: the model may call tools (Slice 1:
    /// `read_file`, read-only) and the client runs the tool-call roundtrip.
    /// Without this flag vf-clide is a plain chat client.
    #[arg(long)]
    agent: bool,

    /// Auto-approve **read-only** tools (`read_file`, `search`) in `--agent`
    /// mode. Does NOT approve `write_file` (needs `--allow-mutating`) or
    /// `shell` (needs `--allow-shell`). In the REPL, reads are then
    /// auto-approved (still printed) and write/shell still prompt `y/N`;
    /// headless, calls above the tier are denied.
    #[arg(long)]
    yes: bool,

    /// Permit **`write_file`** (mutating, workspace-confined) without a
    /// prompt. Implies `--yes` (also auto-approves reads). Does NOT permit
    /// `shell`. In the REPL `shell` still prompts; headless it is denied.
    #[arg(long)]
    allow_mutating: bool,

    /// Permit **`shell`** (exec — NOT workspace-confinable, highest risk)
    /// without a prompt. Implies `--allow-mutating` + `--yes`. The loud,
    /// explicit "I accept arbitrary command execution" opt-in.
    #[arg(long)]
    allow_shell: bool,

    /// Workspace root for the file tools (`read_file`/`write_file`/`search`).
    /// Paths are confined inside it (`..`/symlinks that escape are rejected).
    /// Default: the current directory. `shell` runs here but is NOT confined.
    #[arg(long)]
    workspace: Option<String>,

    /// Override the built-in agent system prompt with the contents of this
    /// file (replaces it entirely). Without it, the default is used and any
    /// `AGENTS.md` in the workspace root is appended.
    #[arg(long)]
    system: Option<String>,

    /// Disable the agent system prompt entirely (no system message).
    #[arg(long)]
    no_system: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut client = Client::new(&args.url, &args.model);
    client.temperature = Some(args.temperature);
    client.max_tokens = Some(args.max_tokens);

    // Workspace root for the file tools — canonicalized ONCE so the
    // confinement check compares canonical-to-canonical (resolves `..`
    // and symlinks in the root itself).
    let workspace = resolve_workspace(args.workspace.as_deref())?;
    // Cumulative auto-approve ceiling from the opt-in flags (each implies
    // the lower tiers). Applies to BOTH modes: headless auto-approve, and
    // REPL auto-approve-below / prompt-above.
    let ceiling = auto_ceiling(args.yes, args.allow_mutating, args.allow_shell);
    // The agent constitution (system prompt). Computed up front so a bad
    // `--system` path errors before we connect.
    let system = vf_clide::agent::system_prompt(&workspace, args.system.as_deref(), args.no_system)?;

    match args.prompt {
        Some(prompt) if args.agent => {
            let gate = vf_clide::agent::Gate::headless(ceiling);
            run_agent_headless(client, prompt, args.no_think, gate, workspace, system).await
        }
        Some(prompt) => run_headless(client, prompt, args.no_stream, args.no_think, args.project).await,
        None => {
            // Session memory scope (Stufe B-1): an explicit `--project` wins;
            // otherwise derive a deterministic key from the canonical
            // workspace so a directory's memory persists across sessions.
            let project = args
                .project
                .clone()
                .unwrap_or_else(|| vf_clide::memory::derive_project_key(&workspace));
            let mut repl = Repl::new(client, Box::new(NoopMemory), Some(project))
                .with_no_think(args.no_think)
                .with_agent(args.agent)
                .with_workspace(workspace)
                .with_system(system)
                .with_ceiling(ceiling);
            repl.run().await
        }
    }
}

/// Map the cumulative opt-in flags to the auto-approve ceiling (used by
/// both the headless gate and the REPL gate).
/// `--allow-shell` ⊃ `--allow-mutating` ⊃ `--yes`; absent = `None`
/// (headless: deny all; REPL: prompt for everything).
fn auto_ceiling(yes: bool, allow_mutating: bool, allow_shell: bool) -> Option<vf_clide::types::ToolRisk> {
    use vf_clide::types::ToolRisk;
    if allow_shell {
        Some(ToolRisk::Exec)
    } else if allow_mutating {
        Some(ToolRisk::Mutating)
    } else if yes {
        Some(ToolRisk::ReadOnly)
    } else {
        None
    }
}

/// Resolve + canonicalize the workspace root (default = cwd).
fn resolve_workspace(arg: Option<&str>) -> Result<std::path::PathBuf> {
    let raw = match arg {
        Some(p) => std::path::PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    raw.canonicalize()
        .map_err(|e| format!("--workspace {}: {e}", raw.display()).into())
}

/// Headless agent loop. The `gate` carries the cumulative auto-approve
/// ceiling from `--yes`/`--allow-mutating`/`--allow-shell`; `system` is the
/// constitution (prepended as the first message). A call above the ceiling
/// is denied and the loop ends gracefully.
async fn run_agent_headless(
    client: Client,
    prompt: String,
    no_think: bool,
    gate: vf_clide::agent::Gate,
    workspace: std::path::PathBuf,
    system: Option<String>,
) -> Result<()> {
    use vf_clide::agent::{self, LoopEnd, LOOP_CAP};
    use vf_clide::client::{empty_notice, truncation_notice};

    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = system {
        msgs.push(ChatMessage::system(sys));
    }
    let content = if no_think { format!("{prompt} /no_think") } else { prompt };
    msgs.push(ChatMessage::user(content));

    let max_tokens = client.max_tokens;
    // Headless stays byte-clean → no status bar, discard the turn usage.
    let (end, _usage) = agent::run(&client, msgs, gate, &workspace, None).await?;
    match end {
        LoopEnd::Final { content, finish_reason } => {
            let text = content.unwrap_or_default();
            println!("{text}");
            if let Some(m) = empty_notice(&text) {
                eprintln!("{m}");
            } else if let Some(m) = truncation_notice(finish_reason.as_deref(), max_tokens) {
                eprintln!("{m}");
            }
        }
        LoopEnd::CapReached => {
            eprintln!(
                "[agent] stopped: reached the tool-call loop cap ({LOOP_CAP}). \
                 The task may be incomplete — simplify the request or raise the cap."
            );
        }
    }
    Ok(())
}

async fn run_headless(
    client: Client,
    prompt: String,
    no_stream: bool,
    no_think: bool,
    project: Option<String>,
) -> Result<()> {
    use vf_clide::client::{empty_notice, truncation_notice};

    // Single point where the (no-op) memory seam is consulted, mirroring
    // the REPL path.
    let memory = NoopMemory;
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ctx) = memory.context_for(project.as_deref(), &[]) {
        msgs.push(ChatMessage::system(ctx));
    }
    // `/no_think` is a model directive in the message content (Qwen3), not
    // a CLI/slash command.
    let content = if no_think { format!("{prompt} /no_think") } else { prompt };
    msgs.push(ChatMessage::user(content));

    let max_tokens = client.max_tokens;
    let outcome = if no_stream {
        let o = client.chat_once(msgs).await?;
        println!("{}", o.text);
        o
    } else {
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let o = client
            .chat_stream(msgs, |t| {
                print!("{t}");
                let _ = stdout.flush();
            })
            .await?;
        println!();
        o
    };

    // Make truncation / empty answers visible instead of silent.
    if let Some(m) = empty_notice(&outcome.text) {
        eprintln!("{m}");
    } else if let Some(m) = truncation_notice(outcome.finish_reason.as_deref(), max_tokens) {
        eprintln!("{m}");
    }
    Ok(())
}
