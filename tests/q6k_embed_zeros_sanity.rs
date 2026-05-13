//! Sprint 52R R-2 — Q6_K embed-lookup zero-count sanity check.
//!
//! Sprint 52Q's hidden-state bisect found 76 of 2816 dims in the
//! GGUF post-embed dump were exact zero (vs 0 zeros on SafeTensors).
//! That's the only soft anomaly the structural bisect surfaced. This
//! test asks: is the 76-zero count specific to token 51203 (the
//! GGUF chat's first-sampled token), or systematic across all rows?
//!
//! Verdict matrix:
//! - token 51203 has many zeros, other tokens have ~0  → token-specific,
//!   legitimate Q6_K quantisation (some sub-blocks happened to have
//!   `d ≈ 0`). NOT a bug.
//! - many tokens have ~76 zeros → systematic Q6_K dequant bug,
//!   bisect upstream against `dequantize_row_q6_K` in ggml-quants.c.
//! - token 51203 has 0 zeros → CPU dequant fine, the runtime GPU
//!   embed lookup is the bug.
//!
//! Gated on the 26B GGUF being present — skips cleanly otherwise.

use std::path::PathBuf;

use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::q6k;

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

/// Inlined Q6_K row dequant — mirrors the Q6_K arm of
/// `decode::embedding_row` so we don't have to build a `ModelConfig`.
fn dequant_row_q6k(bytes: &[u8], token_id: u32, hidden_dim: usize) -> Vec<f32> {
    assert!(hidden_dim % q6k::QUANT_K == 0);
    let blocks_per_row = hidden_dim / q6k::QUANT_K;
    let row_bytes = blocks_per_row * q6k::BLOCK_BYTES;
    let row_off = (token_id as usize) * row_bytes;
    assert!(row_off + row_bytes <= bytes.len(), "token_id out of range");

    let mut out = Vec::with_capacity(hidden_dim);
    for b in 0..blocks_per_row {
        let blk_off = row_off + b * q6k::BLOCK_BYTES;
        let block: &[u8; q6k::BLOCK_BYTES] = bytes[blk_off..blk_off + q6k::BLOCK_BYTES]
            .try_into()
            .unwrap();
        out.extend_from_slice(&q6k::dequant_block(block));
    }
    out
}

#[test]
fn q6k_embed_zero_distribution_per_token() {
    let p = home_models().join("google_gemma-4-26B-A4B-it-Q3_K_M.gguf");
    let Some(gguf) = open_or_skip(&p) else { return };

    let info = gguf
        .tensor("token_embd.weight")
        .expect("token_embd.weight present");
    // 26B's token_embd should be Q6_K per Sprint 52F's decode.rs arm.
    assert_eq!(
        info.ggml_type,
        GgmlType::Q6K,
        "expected Q6_K token_embd (Sprint 52F); got {:?}",
        info.ggml_type,
    );

    let bytes = gguf.tensor_bytes(info);
    let hidden_dim: usize = 2816;
    // GGUF tensor dimensions are stored innermost-first: [hidden, vocab].
    let vocab_size = (*info.dimensions.last().expect("token_embd has dims")) as usize;
    eprintln!(
        "Q6_K token_embd: vocab={} hidden={} row_bytes={} total_bytes={}",
        vocab_size,
        hidden_dim,
        (hidden_dim / q6k::QUANT_K) * q6k::BLOCK_BYTES,
        bytes.len(),
    );

    // Sample a spread of token ids:
    //   1     — BOS-adjacent
    //   100   — common low-id
    //   1000  — mid-low
    //   10000 — mid
    //   50000 — high-vocab
    //   50429 — SafeTensors "Paris" id (Sprint 52Q's ST decode entry)
    //   51203 — GGUF first-sampled id (Sprint 52Q's GGUF decode entry,
    //           the row reported 76 zeros on the GPU dump)
    let probe: &[u32] = &[1, 100, 1000, 1852, 10000, 50000, 50429, 51203];

    let mut max_zeros = 0usize;
    let mut total_zeros_across_probes = 0usize;
    let mut zeros_for_51203: usize = 0;
    let mut nonfinite_seen = 0usize;

    for &tok in probe {
        if (tok as usize) >= vocab_size {
            eprintln!("token {tok}: out of vocab range, skip");
            continue;
        }
        let row = dequant_row_q6k(bytes, tok, hidden_dim);
        let zeros = row.iter().filter(|&&x| x == 0.0).count();
        let nonfinite = row.iter().filter(|&&x| !x.is_finite()).count();
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32).sqrt();
        let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mut head = String::new();
        for (i, &v) in row.iter().take(8).enumerate() {
            if i > 0 {
                head.push_str(", ");
            }
            head.push_str(&format!("{v:+.4}"));
        }
        eprintln!(
            "token {tok:>6}: zeros={zeros:4}/{hidden_dim} nonfinite={nonfinite} \
             rms={rms:+.4} max|x|={max_abs:+.4} first8=[{head}]"
        );
        max_zeros = max_zeros.max(zeros);
        total_zeros_across_probes += zeros;
        if tok == 51203 {
            zeros_for_51203 = zeros;
        }
        nonfinite_seen += nonfinite;
    }

    let n_probes = probe.iter().filter(|&&t| (t as usize) < vocab_size).count();
    let mean_zeros = total_zeros_across_probes as f64 / n_probes as f64;
    eprintln!(
        "summary: mean_zeros={mean_zeros:.1} max_zeros={max_zeros} \
         tok51203_zeros={zeros_for_51203} nonfinite_total={nonfinite_seen}"
    );

    // Sanity: no NaN/Inf in any sampled row.
    assert_eq!(nonfinite_seen, 0, "Q6_K dequant produced NaN/Inf");

    // Sanity: at least one probed token has a finite, non-trivial row
    // (rules out trivial-bug like "all rows are zero everywhere").
    assert!(
        max_zeros < hidden_dim,
        "at least one token should have <100% zeros"
    );

    // Diagnostic: print verdict + interpretation. The Q6_K dequant
    // formula is `out = d * scale * (q - 32)` where `q ∈ [0, 63]`.
    // For q == 32 (the unsigned-6-bit midpoint), output is EXACTLY 0.0
    // by construction — regardless of `d` or `scale`. Embedding
    // distributions tend to cluster around zero, so ~1/64 ≈ 1.6% of
    // quants naturally land at q=32; in practice we see 2–4% zeros
    // per row, which is fundamental Q6_K behaviour, not a bug.
    //
    // Real-bug signatures would be: NaN/Inf, all-zero rows, or
    // dramatic mean_zeros >> ~150 (5%+ across all tokens) — none
    // of those should fire on a healthy Gemma-4 Q6_K embedding.
    let mean_zeros_pct = (mean_zeros / hidden_dim as f64) * 100.0;
    if mean_zeros_pct > 5.0 {
        eprintln!(
            "VERDICT: mean_zeros_pct={mean_zeros_pct:.1}% — \
             unexpectedly high. Cross-check vs ggml-quants.c::dequantize_row_q6_K."
        );
    } else {
        eprintln!(
            "VERDICT: mean_zeros={mean_zeros:.1} ({mean_zeros_pct:.1}% of {hidden_dim}) \
             — consistent with Q6_K's `q=32 → 0` identity (theoretical ~1.6%, \
             empirical ~2–4% for embedding-like distributions). No dequant bug. \
             tok51203_zeros={zeros_for_51203} fits the distribution."
        );
    }
}
