//! Phase-4D probe: dump architecture + tokenizer + Q/K-norm tensor presence
//! for any GGUF. Used to plan multi-model support.
//!
//! Usage: VF_MODEL_PATH=~/models/...gguf cargo run --release --example probe_model

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

    println!("# probe_model: {}", path.display());
    let gguf = GgufFile::open(&path)?;

    let arch = gguf.metadata_str("general.architecture").unwrap_or("?");
    println!("\n## general");
    println!("architecture       = {arch:?}");
    if let Ok(name) = gguf.metadata_str("general.name") {
        println!("name               = {name:?}");
    }

    println!("\n## arch keys ({arch}.*)");
    let arch_prefix = format!("{arch}.");
    let mut arch_keys: Vec<&String> = gguf
        .metadata
        .keys()
        .filter(|k| k.starts_with(&arch_prefix))
        .collect();
    arch_keys.sort();
    for k in arch_keys {
        println!("{k:40} = {}", short(&gguf.metadata[k]));
    }

    println!("\n## tokenizer keys");
    let mut tok_keys: Vec<&String> = gguf
        .metadata
        .keys()
        .filter(|k| k.starts_with("tokenizer."))
        .collect();
    tok_keys.sort();
    for k in tok_keys {
        let v = &gguf.metadata[k];
        match v {
            MetadataValue::Array(a) => {
                let kind = a.first().map(describe_kind).unwrap_or("?");
                println!("{k:40} = Array<{kind}> len={}", a.len());
            }
            MetadataValue::String(s) => {
                let trimmed: String = s.chars().take(120).collect();
                let suffix = if s.len() > 120 { "…" } else { "" };
                println!("{k:40} = String({trimmed:?}{suffix})");
            }
            other => println!("{k:40} = {}", short(other)),
        }
    }

    println!("\n## tensor presence (Q/K-norm probe)");
    for name in [
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
    ] {
        let present = gguf.tensors.contains_key(name);
        println!("{name:40} {}", if present { "PRESENT" } else { "absent" });
    }

    println!("\n## tensor count = {}", gguf.tensors.len());

    Ok(())
}

fn describe_kind(v: &MetadataValue) -> &'static str {
    match v {
        MetadataValue::U8(_) => "u8",
        MetadataValue::I8(_) => "i8",
        MetadataValue::U16(_) => "u16",
        MetadataValue::I16(_) => "i16",
        MetadataValue::U32(_) => "u32",
        MetadataValue::I32(_) => "i32",
        MetadataValue::F32(_) => "f32",
        MetadataValue::Bool(_) => "bool",
        MetadataValue::String(_) => "string",
        MetadataValue::Array(_) => "array",
        MetadataValue::U64(_) => "u64",
        MetadataValue::I64(_) => "i64",
        MetadataValue::F64(_) => "f64",
    }
}

fn short(v: &MetadataValue) -> String {
    match v {
        MetadataValue::String(s) => format!("{:?}", s),
        MetadataValue::U32(x) => x.to_string(),
        MetadataValue::I32(x) => x.to_string(),
        MetadataValue::U64(x) => x.to_string(),
        MetadataValue::I64(x) => x.to_string(),
        MetadataValue::F32(x) => x.to_string(),
        MetadataValue::F64(x) => x.to_string(),
        MetadataValue::Bool(x) => x.to_string(),
        MetadataValue::Array(a) => format!("Array(len={})", a.len()),
        other => format!("{:?}", other),
    }
}
