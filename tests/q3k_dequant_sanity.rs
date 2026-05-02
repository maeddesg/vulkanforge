//! Sprint 17B follow-up — Q3_K CPU dequant sanity check.
//!
//! Q3_K_M Qwen3 produces "!!!!!!" garbage despite hitting bandwidth
//! target. Cheapest hypothesis: `q3k::dequant_block` is wrong.
//!
//! Strategy: Q3_K_M is a lossy requant of Q4_K_M (same source weights),
//! so token_embd.weight rows should match within ~5% RMS. If RMS is
//! huge or values are NaN/all-zero, the bug is in q3k.rs.
//!
//! These tests are gated on the GGUFs being present in ~/models/ —
//! they skip cleanly otherwise so CI without the models doesn't fail.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::{q3k, q4k};

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
fn q3k_dequant_block_is_finite_and_nonzero() {
    let p = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    let info = gguf
        .tensor("token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(info.ggml_type, GgmlType::Q3K);
    let bytes = gguf.tensor_bytes(info);
    assert!(bytes.len() >= q3k::BLOCK_BYTES, "too few bytes");

    // First 8 blocks (= 8 × 256 = 2048 weights of row 0 of token_embd)
    for blk in 0..8 {
        let off = blk * q3k::BLOCK_BYTES;
        let block: &[u8; q3k::BLOCK_BYTES] = bytes[off..off + q3k::BLOCK_BYTES]
            .try_into()
            .unwrap();
        let dq = q3k::dequant_block(block);

        let nonzero = dq.iter().filter(|&&x| x != 0.0).count();
        let nonfinite = dq.iter().filter(|&&x| !x.is_finite()).count();
        let max_abs = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "block {blk}: nonzero={nonzero}/256 nonfinite={nonfinite} max|x|={max_abs:.6e} first8={:?}",
            &dq[..8]
        );

        assert_eq!(nonfinite, 0, "block {blk} has NaN/Inf");
        assert!(nonzero > 0, "block {blk} all zeros");
        assert!(max_abs < 10.0, "block {blk} suspiciously large values");
    }
}

#[test]
fn dump_q3k_m_layer0_quant_types() {
    let p = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    eprintln!("=== Q3_K_M Qwen3-8B layer 0 weight types ===");
    for suffix in [
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "attn_norm.weight",
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

    // Also dump layer 1 just in case Q3_K_M has irregular types
    eprintln!("=== Q3_K_M Qwen3-8B layer 1 weight types ===");
    for suffix in [
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ] {
        let key = format!("blk.1.{suffix}");
        if let Some(t) = gguf.tensor(&key) {
            eprintln!("  blk.1.{suffix:<22} → {:?}", t.ggml_type);
        }
    }
}

#[test]
fn q3k_vs_q4k_embedding_rms_within_bound() {
    let q3 = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    let q4 = home_models().join("Qwen3-8B-Q4_K_M.gguf");
    let Some(g3) = open_or_skip(&q3) else { return };
    let Some(g4) = open_or_skip(&q4) else { return };

    let i3 = g3.tensor("token_embd.weight").unwrap();
    let i4 = g4.tensor("token_embd.weight").unwrap();
    assert_eq!(i3.ggml_type, GgmlType::Q3K);
    assert_eq!(i4.ggml_type, GgmlType::Q4K);
    assert_eq!(i3.dimensions, i4.dimensions, "embd shape mismatch");

    let b3 = g3.tensor_bytes(i3);
    let b4 = g4.tensor_bytes(i4);

    // Compare row 0 (one full embedding row = hidden_dim weights).
    // Qwen3-8B hidden_dim = 4096 → 16 blocks of 256.
    let hidden = i3.dimensions[0] as usize;
    let blocks = hidden / 256;
    let mut sse = 0.0f64;
    let mut sum_sq3 = 0.0f64;
    let mut sum_sq4 = 0.0f64;
    let mut max_abs3 = 0.0f32;
    let mut max_abs4 = 0.0f32;
    for blk in 0..blocks {
        let off3 = blk * q3k::BLOCK_BYTES;
        let off4 = blk * q4k::BLOCK_BYTES;
        let blk3: &[u8; q3k::BLOCK_BYTES] = b3[off3..off3 + q3k::BLOCK_BYTES]
            .try_into()
            .unwrap();
        let blk4: &[u8; q4k::BLOCK_BYTES] = b4[off4..off4 + q4k::BLOCK_BYTES]
            .try_into()
            .unwrap();
        let dq3 = q3k::dequant_block(blk3);
        let dq4 = q4k::dequant_block(blk4);
        for k in 0..256 {
            let d = (dq3[k] - dq4[k]) as f64;
            sse += d * d;
            sum_sq3 += (dq3[k] as f64) * (dq3[k] as f64);
            sum_sq4 += (dq4[k] as f64) * (dq4[k] as f64);
            max_abs3 = max_abs3.max(dq3[k].abs());
            max_abs4 = max_abs4.max(dq4[k].abs());
        }
    }
    let n = (blocks * 256) as f64;
    let rms = (sse / n).sqrt();
    let rms3 = (sum_sq3 / n).sqrt();
    let rms4 = (sum_sq4 / n).sqrt();
    let rel = rms / rms4;
    eprintln!(
        "row 0: hidden={hidden} blocks={blocks} | RMS(diff)={rms:.6e} RMS(Q3)={rms3:.6e} RMS(Q4)={rms4:.6e} rel={rel:.4}"
    );
    eprintln!("max|Q3|={max_abs3:.6e}, max|Q4|={max_abs4:.6e}");

    // Q3_K is a lossier requant of the same FP weights. Empirically
    // the per-row L2 should differ from Q4_K by ~5–15%; we guard at
    // 30% to leave headroom but flag a clearly broken dequant.
    assert!(
        rel < 0.30,
        "Q3 vs Q4 embedding row 0 rel-RMS {rel:.4} > 0.30 — Q3 dequant likely wrong"
    );
    assert!(
        rms3 > 0.5 * rms4,
        "Q3 RMS {rms3:.4e} << Q4 RMS {rms4:.4e} — Q3 is producing tiny values"
    );
}
