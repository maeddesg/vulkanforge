//! Print every Control-type (token_type=3) token plus the immediate
//! neighbourhood of bos/eos/pad ids — used to verify special tokens
//! are well-defined before we wire them into the tokenizer.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::{GgufFile, MetadataValue};

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

    let tokens = gguf.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| if let MetadataValue::Array(a) = v { Some(a) } else { None })
        .ok_or("tokens missing")?;
    let types = gguf.metadata.get("tokenizer.ggml.token_type")
        .and_then(|v| if let MetadataValue::Array(a) = v { Some(a) } else { None })
        .ok_or("token_type missing")?;

    println!("Total tokens: {}", tokens.len());
    println!("\n== Control tokens (token_type == 3) ==");
    for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
        if let (MetadataValue::String(s), MetadataValue::I32(t)) = (tok, ty) {
            if *t == 3 {
                println!("  [{i}] type={t} {:?}", s);
            }
        }
    }

    println!("\n== bos=151643, eos=151645, pad=151643 — neighbourhood ==");
    for i in 151640..151700.min(tokens.len()) {
        if let (MetadataValue::String(s), MetadataValue::I32(t)) = (&tokens[i], &types[i]) {
            println!("  [{i}] type={t} {:?}", s);
        }
    }
    Ok(())
}
