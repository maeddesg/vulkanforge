//! Phase-2D inspection helper: prints the full set of `tokenizer.*`
//! metadata in the GGUF, sample tokens, and the bos/eos ids.

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
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for k in keys {
        if !k.starts_with("tokenizer.") {
            continue;
        }
        let v = &gguf.metadata[k];
        match v {
            MetadataValue::Array(arr) => {
                let inner_kind = arr.first().map(describe_kind).unwrap_or("?");
                println!("{k}: Array<{inner_kind}> len={}", arr.len());
                for (i, x) in arr.iter().enumerate().take(8) {
                    println!("  [{i}] {}", short(x));
                }
                if arr.len() > 8 {
                    println!("  …");
                }
            }
            other => println!("{k}: {}", short(other)),
        }
    }
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
