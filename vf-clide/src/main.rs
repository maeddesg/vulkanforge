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

    match args.prompt {
        Some(prompt) => run_headless(client, prompt, args.no_stream, args.project).await,
        None => {
            let mut repl = Repl::new(client, Box::new(NoopMemory), args.project);
            repl.run().await
        }
    }
}

async fn run_headless(
    client: Client,
    prompt: String,
    no_stream: bool,
    project: Option<String>,
) -> Result<()> {
    // Single point where the (no-op) memory seam is consulted, mirroring
    // the REPL path.
    let memory = NoopMemory;
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ctx) = memory.context_for(project.as_deref(), &[]) {
        msgs.push(ChatMessage::system(ctx));
    }
    msgs.push(ChatMessage::user(prompt));

    if no_stream {
        let text = client.chat_once(msgs).await?;
        println!("{text}");
    } else {
        use std::io::Write;
        let mut stdout = std::io::stdout();
        client
            .chat_stream(msgs, |t| {
                print!("{t}");
                let _ = stdout.flush();
            })
            .await?;
        println!();
    }
    Ok(())
}
