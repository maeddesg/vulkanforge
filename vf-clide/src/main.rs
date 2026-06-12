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

    /// Project name — memory seam. Phase 1 parses and threads it through
    /// but it is a no-op (see `memory::NoopMemory`).
    #[arg(long)]
    project: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut client = Client::new(&args.url, &args.model);
    client.temperature = Some(args.temperature);
    client.max_tokens = Some(args.max_tokens);

    match args.prompt {
        Some(prompt) => run_headless(client, prompt, args.no_stream, args.no_think, args.project).await,
        None => {
            let mut repl =
                Repl::new(client, Box::new(NoopMemory), args.project).with_no_think(args.no_think);
            repl.run().await
        }
    }
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
