//! Sanity-test the Qwen3 BPE tokenizer end-to-end on the real GGUF.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::GgufFile;
use vulkanforge::backend::vulkan::tokenizer::{apply_chat_template, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("VF_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("models").join("Qwen3-8B-Q4_K_M.gguf"))
                .expect("$HOME unset")
        });
    let gguf = GgufFile::open(&path)?;
    let t0 = std::time::Instant::now();
    let tok = Tokenizer::from_gguf(&gguf)?;
    println!("Tokenizer loaded in {:.2} s", t0.elapsed().as_secs_f64());
    println!("  vocab_size = {}", tok.vocab_size());
    println!("  bos_id     = {:?}", tok.bos_id);
    println!("  eos_id     = {} ({:?})", tok.eos_id, tok.token_str(tok.eos_id));
    if let Some(et) = tok.endoftext_id {
        println!("  endoftext  = {} ({:?})", et, tok.token_str(et));
    }
    if let Some(s) = tok.im_start_id {
        println!("  im_start_id= {} ({:?})", s, tok.token_str(s));
    }
    if let Some(e) = tok.im_end_id {
        println!("  im_end_id  = {} ({:?})", e, tok.token_str(e));
    }

    for sample in [
        "Hello world",
        "Hello world!",
        "Explain what a mutex is in one sentence.",
        " leading space",
        "Two\nlines",
    ] {
        let ids = tok.encode(sample);
        let back = tok.decode(&ids);
        let id_strs: Vec<&str> = ids.iter().map(|&i| tok.token_str(i).unwrap_or("?")).collect();
        println!("\n{:?}", sample);
        println!("  ids = {:?}", ids);
        println!("  toks= {:?}", id_strs);
        println!("  back= {:?}  match={}", back, back == sample);
    }

    let prompt = apply_chat_template(&tok, "Explain what a mutex is.", None);
    println!(
        "\nChat-template tokens: len={} first6={:?} last6={:?}",
        prompt.len(),
        &prompt[..6.min(prompt.len())],
        &prompt[prompt.len().saturating_sub(6)..],
    );
    let chat_back = tok.decode(&prompt);
    println!("  rendered:\n---\n{chat_back}\n---");
    Ok(())
}
