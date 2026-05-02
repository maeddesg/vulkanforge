//! Sprint 17C — Q5_K CPU dequant sanity check.
//!
//! Q3_K_M GGUFs ship `attn_v.weight` and `ffn_down.weight` as Q5_K
//! (the Sprint 17B blocker). These tests confirm `q5k::dequant_block`
//! produces sane values and is consistent with the Q4_K_M counterpart
//! of the same model (Q5_K is a higher-fidelity requant of the same
//! source weights, so the per-block RMS should be in the same
//! ballpark).
//!
//! Tests are gated on the GGUFs being present in ~/models/ — they
//! skip cleanly otherwise so CI without the models doesn't fail.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::{q4k, q5k};

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
fn q5k_dequant_block_is_finite_and_nonzero() {
    // Load Q3_K_M, find the first Q5_K tensor (attn_v.weight blk.0).
    let p = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    let info = gguf
        .tensor("blk.0.attn_v.weight")
        .expect("blk.0.attn_v.weight present");
    assert_eq!(info.ggml_type, GgmlType::Q5K, "expected Q5_K for attn_v");
    let bytes = gguf.tensor_bytes(info);
    assert!(bytes.len() >= q5k::BLOCK_BYTES, "too few bytes");

    // First 8 blocks (= 8 × 256 = 2048 weights worth)
    for blk in 0..8 {
        let off = blk * q5k::BLOCK_BYTES;
        let block: &[u8; q5k::BLOCK_BYTES] = bytes[off..off + q5k::BLOCK_BYTES]
            .try_into()
            .unwrap();
        let dq = q5k::dequant_block(block);

        let nonzero = dq.iter().filter(|&&x| x != 0.0).count();
        let nonfinite = dq.iter().filter(|&&x| !x.is_finite()).count();
        let max_abs = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "block {blk}: nonzero={nonzero}/256 nonfinite={nonfinite} max|x|={max_abs:.6e} first8={:?}",
            &dq[..8]
        );

        assert_eq!(nonfinite, 0, "block {blk} has NaN/Inf");
        assert!(nonzero > 0, "block {blk} all zeros");
        // attn_v weights are typically ≤ 0.1 in magnitude
        assert!(max_abs < 5.0, "block {blk} suspiciously large values");
    }
}

#[test]
fn q5k_vs_q4k_attn_v_rms_within_bound() {
    // Q3_K_M attn_v is Q5_K; Q4_K_M attn_v is Q6_K (one tier higher).
    // We compare against the Q4_K_M attn_v expressed as Q4_K, which
    // we'd find in a Q4_K_S file_type — but that's not what we have.
    // Instead, we just sanity-check that Q5_K row 0 has a sensible
    // L2 norm: not zero, not insane, similar order to a Q4_K row.
    let p_q3k = home_models().join("Qwen3-8B-Q3_K_M.gguf");
    let p_q4k = home_models().join("Qwen3-8B-Q4_K_M.gguf");
    let Some(g3) = open_or_skip(&p_q3k) else { return };
    let Some(g4) = open_or_skip(&p_q4k) else { return };

    let i_q5 = g3.tensor("blk.0.attn_v.weight").expect("Q3_K_M attn_v");
    assert_eq!(i_q5.ggml_type, GgmlType::Q5K);

    // The Q4_K_M counterpart is typically Q6_K (mixed-quant pattern).
    // We pick the Q4_K_M attn_output (which IS Q4_K) for an
    // order-of-magnitude RMS comparison instead — different tensor,
    // same model class, useful sanity check for "values plausible".
    let i_q4 = g4
        .tensor("blk.0.attn_output.weight")
        .expect("Q4_K_M attn_output");
    assert_eq!(i_q4.ggml_type, GgmlType::Q4K);

    let b5 = g3.tensor_bytes(i_q5);
    let b4 = g4.tensor_bytes(i_q4);
    // Compare row-0 norms (same hidden_dim).
    // Q3_K_M attn_v dims: [hidden, kv_dim]. We just look at one row.
    let hidden = i_q5.dimensions[0] as usize;
    let blocks = hidden / 256;

    let mut sum_sq5 = 0.0f64;
    let mut sum_sq4 = 0.0f64;
    let mut max_abs5 = 0.0f32;
    let mut max_abs4 = 0.0f32;
    for blk in 0..blocks {
        let off5 = blk * q5k::BLOCK_BYTES;
        let off4 = blk * q4k::BLOCK_BYTES;
        let blk5: &[u8; q5k::BLOCK_BYTES] = b5[off5..off5 + q5k::BLOCK_BYTES]
            .try_into().unwrap();
        let blk4: &[u8; q4k::BLOCK_BYTES] = b4[off4..off4 + q4k::BLOCK_BYTES]
            .try_into().unwrap();
        let dq5 = q5k::dequant_block(blk5);
        let dq4 = q4k::dequant_block(blk4);
        for k in 0..256 {
            sum_sq5 += (dq5[k] as f64) * (dq5[k] as f64);
            sum_sq4 += (dq4[k] as f64) * (dq4[k] as f64);
            max_abs5 = max_abs5.max(dq5[k].abs());
            max_abs4 = max_abs4.max(dq4[k].abs());
        }
    }
    let n = (blocks * 256) as f64;
    let rms5 = (sum_sq5 / n).sqrt();
    let rms4 = (sum_sq4 / n).sqrt();
    eprintln!(
        "Q3_K_M attn_v[0] RMS={rms5:.6e} max|x|={max_abs5:.6e}; \
         Q4_K_M attn_output[0] RMS={rms4:.6e} max|x|={max_abs4:.6e}"
    );
    assert!(rms5 > 1e-4, "Q5_K attn_v looks empty (RMS too small)");
    assert!(rms5 < 1.0, "Q5_K attn_v looks insane (RMS too large)");
    // Both are weight matrices; they should be within an order of
    // magnitude of each other.
    let ratio = rms5 / rms4;
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "RMS ratio Q5/Q4 = {ratio} out of expected range [0.1, 10]"
    );
}
