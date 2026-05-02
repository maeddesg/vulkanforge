//! Sprint 17D — Q4_0 CPU dequant sanity check.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::q4_0;

fn home_models() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME unset");
    PathBuf::from(home).join("models")
}

fn open_or_skip(path: &PathBuf) -> Option<GgufFile> {
    if !path.exists() {
        eprintln!("skip — {} not found", path.display());
        return None;
    }
    Some(GgufFile::open(path).expect("open gguf"))
}

#[test]
fn q4_0_dequant_block_is_finite_and_nonzero() {
    let p = home_models().join("Qwen2.5-7B-Instruct-Q4_0.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    let info = gguf
        .tensor("token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(info.ggml_type, GgmlType::Q4_0);
    let bytes = gguf.tensor_bytes(info);
    assert!(bytes.len() >= q4_0::BLOCK_BYTES, "too few bytes");

    for blk in 0..8 {
        let off = blk * q4_0::BLOCK_BYTES;
        let block: &[u8; q4_0::BLOCK_BYTES] = bytes[off..off + q4_0::BLOCK_BYTES]
            .try_into()
            .unwrap();
        let dq = q4_0::dequant_block(block);

        let nonzero = dq.iter().filter(|&&x| x != 0.0).count();
        let nonfinite = dq.iter().filter(|&&x| !x.is_finite()).count();
        let max_abs = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "block {blk}: nonzero={nonzero}/32 nonfinite={nonfinite} max|x|={max_abs:.6e} all={:?}",
            dq
        );

        assert_eq!(nonfinite, 0, "block {blk} has NaN/Inf");
        assert!(nonzero > 0, "block {blk} all zeros");
        assert!(max_abs < 5.0, "block {blk} suspiciously large values");
    }
}

fn dump_one(model: &str, gguf: &GgufFile) {
    eprintln!("=== {model} layer 0 weight types ===");
    for suffix in [
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ] {
        let key = format!("blk.0.{suffix}");
        if let Some(t) = gguf.tensor(&key) {
            eprintln!("  blk.0.{suffix:<22} → {:?} dims={:?}", t.ggml_type, t.dimensions);
        }
    }
    for key in ["token_embd.weight", "output.weight"] {
        if let Some(t) = gguf.tensor(key) {
            eprintln!("  {key:<22} → {:?} dims={:?}", t.ggml_type, t.dimensions);
        }
    }
    // Walk all tensors and list distinct quant types
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for (_, info) in &gguf.tensors {
        *counts.entry(format!("{:?}", info.ggml_type)).or_default() += 1;
    }
    eprintln!("  All-tensor counts: {:?}", counts);
}

#[test]
fn dump_qwen25_layer0_quant_types() {
    for (label, fname) in [
        ("0.5B", "qwen2.5-0.5b-instruct-q4_0.gguf"),
        ("7B",   "Qwen2.5-7B-Instruct-Q4_0.gguf"),
        ("7B-Pure", "Qwen2.5-7B-Instruct-Q4_0-Pure.gguf"),
        ("14B",  "Qwen2.5-14B-Instruct-Q4_0.gguf"),
    ] {
        let p = home_models().join(fname);
        let Some(gguf) = open_or_skip(&p) else { continue };
        dump_one(label, &gguf);
    }
}

#[test]
fn _legacy_dump_qwen25_7b_layer0_quant_types() {
    let p = home_models().join("Qwen2.5-7B-Instruct-Q4_0.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    eprintln!("=== Qwen2.5-7B Q4_0 layer 0 weight types ===");
    for suffix in [
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q.bias", "attn_k.bias", "attn_v.bias",
        "attn_norm.weight",
        "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "ffn_norm.weight",
    ] {
        let key = format!("blk.0.{suffix}");
        if let Some(t) = gguf.tensor(&key) {
            eprintln!("  blk.0.{suffix:<22} → {:?} dims={:?}", t.ggml_type, t.dimensions);
        } else {
            eprintln!("  blk.0.{suffix:<22} → MISSING");
        }
    }
    eprintln!("=== Top-level tensors ===");
    for key in ["token_embd.weight", "output.weight", "output_norm.weight"] {
        if let Some(t) = gguf.tensor(key) {
            eprintln!("  {key:<22} → {:?} dims={:?}", t.ggml_type, t.dimensions);
        } else {
            eprintln!("  {key:<22} → MISSING");
        }
    }
}
