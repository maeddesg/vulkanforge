//! Tiny inspector: are `<think>` / `</think>` single special tokens or
//! sequences? Determines whether the think-filter needs a state machine
//! or a single-id check.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::GgufFile;
use vulkanforge::backend::vulkan::tokenizer::Tokenizer;

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
    let tok = Tokenizer::from_gguf(&gguf)?;

    for s in ["<think>", "</think>", "Hello <think>x</think>World"] {
        let ids = tok.encode(s);
        let toks: Vec<&str> = ids.iter().map(|&i| tok.token_str(i).unwrap_or("?")).collect();
        println!("{:<40} → {:?}\n  toks: {:?}", s, ids, toks);
    }

    // Also try the "natural" stream form Qwen3 emits — typically a
    // newline before/after the <think> block.
    for s in ["\n<think>\n", "\n</think>\n\n", "<think>\nfoo\n</think>"] {
        let ids = tok.encode(s);
        let toks: Vec<&str> = ids.iter().map(|&i| tok.token_str(i).unwrap_or("?")).collect();
        println!("{:?} → {:?}\n  toks: {:?}", s, ids, toks);
    }
    Ok(())
}
