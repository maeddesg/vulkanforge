//! Sprint 7 — Rust reference implementation of tiled-Q flash-attention.
//!
//! Standalone CPU implementation that mirrors the algorithm we plan to
//! ship as a GLSL shader. Validating the algorithm in pure Rust first
//! lets us:
//!   * iterate on the tiling math without GPU compile cycles
//!   * compare against a naive O(N²)-memory implementation that any
//!     reader can verify by inspection
//!   * generate "golden" outputs that the GPU shader's parity tests
//!     can compare against
//!
//! The reference is **identical in result** to the naive QK^T → softmax
//! → V path; the only difference is memory access pattern (tiles)
//! plus per-query online softmax bookkeeping. Bit-for-bit equality
//! is not expected (rounding order changes), but max_abs should be
//! < 1e-5 for the FP32 inputs we test with.

use std::f32;

/// Naive attention: materializes the full N×kv_len score matrix.
/// Causal masking is OPTIONAL and applies position `pos[qi]` (in
/// global KV indexing) versus key index `ki` (also global).
fn naive_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    q_start: usize,
    causal: bool,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), kv_len * head_dim);
    assert_eq!(v.len(), kv_len * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * head_dim];
    for qi in 0..seq_len {
        let q_pos = q_start + qi;
        // 1) Compute scaled scores S[ki] = Q[qi] · K[ki] / sqrt(d)
        let mut s = vec![0.0f32; kv_len];
        for (ki, sk) in s.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            *sk = if causal && ki > q_pos {
                f32::NEG_INFINITY
            } else {
                dot * scale
            };
        }
        // 2) Softmax over ki. Subtract max for stability.
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for sk in s.iter_mut() {
            *sk = (*sk - m).exp();
            sum += *sk;
        }
        let inv_sum = 1.0 / sum;
        // 3) Output: out[qi] = sum_ki softmax(s)[ki] * V[ki]
        for ki in 0..kv_len {
            let p = s[ki] * inv_sum;
            for d in 0..head_dim {
                out[qi * head_dim + d] += p * v[ki * head_dim + d];
            }
        }
    }
    out
}

/// Tiled-Q flash-attention reference. Mirrors the planned GLSL shader:
/// outer loop over Q-tiles (Br queries each), inner loop over K/V-tiles
/// (Bc keys each), online softmax bookkeeping per query. Same final
/// result as `naive_attn` modulo float rounding order.
///
/// Causal mask: position of query `qi` in global indexing is
/// `q_start + qi`. A key at global index `ki` is visible iff `!causal
/// || ki <= q_start + qi`.
fn flash_attn_tiled_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    q_start: usize,
    br: usize,
    bc: usize,
    causal: bool,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), kv_len * head_dim);
    assert_eq!(v.len(), kv_len * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * head_dim];

    let mut q_tile_start = 0usize;
    while q_tile_start < seq_len {
        let actual_br = (q_tile_start + br).min(seq_len) - q_tile_start;
        // Per-query online-softmax state. Br entries each.
        let mut m = vec![f32::NEG_INFINITY; actual_br];
        let mut l = vec![0.0f32; actual_br];
        let mut o = vec![0.0f32; actual_br * head_dim];

        let mut kv_tile_start = 0usize;
        while kv_tile_start < kv_len {
            let actual_bc = (kv_tile_start + bc).min(kv_len) - kv_tile_start;

            // Compute scores for this Q-tile × K-tile sub-block.
            // S[qi][ki] for qi in 0..actual_br, ki in 0..actual_bc.
            let mut s = vec![0.0f32; actual_br * actual_bc];
            for qi in 0..actual_br {
                let q_pos = q_start + q_tile_start + qi;
                for ki in 0..actual_bc {
                    let global_k = kv_tile_start + ki;
                    if causal && global_k > q_pos {
                        s[qi * actual_bc + ki] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(q_tile_start + qi) * head_dim + d]
                            * k[global_k * head_dim + d];
                    }
                    s[qi * actual_bc + ki] = dot * scale;
                }
            }

            // Online softmax update: per query, fold this tile's
            // contribution into running max + sum + output.
            for qi in 0..actual_br {
                // Row-max of this tile for query qi.
                let mut row_max = f32::NEG_INFINITY;
                for ki in 0..actual_bc {
                    let v_s = s[qi * actual_bc + ki];
                    if v_s > row_max {
                        row_max = v_s;
                    }
                }
                let new_m = m[qi].max(row_max);
                // If new_m is still -inf (entire tile masked out for
                // this query), skip the rescale — there's nothing to
                // contribute and exp(0) = 1 would corrupt l/o.
                if new_m == f32::NEG_INFINITY {
                    continue;
                }
                // Rescale running output and sum to the new pivot.
                let scale_factor = if m[qi] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m[qi] - new_m).exp()
                };
                l[qi] *= scale_factor;
                for d in 0..head_dim {
                    o[qi * head_dim + d] *= scale_factor;
                }
                // Accumulate this tile's P × V into o, and P into l.
                for ki in 0..actual_bc {
                    let p = (s[qi * actual_bc + ki] - new_m).exp();
                    if p == 0.0 { continue; }
                    l[qi] += p;
                    let v_row = (kv_tile_start + ki) * head_dim;
                    for d in 0..head_dim {
                        o[qi * head_dim + d] += p * v[v_row + d];
                    }
                }
                m[qi] = new_m;
            }
            kv_tile_start += bc;
        }

        // Normalize and write back.
        for qi in 0..actual_br {
            // l[qi] can only be 0 if every key for this query was
            // masked out — that shouldn't happen for valid prefill
            // (queries always see at least their own position).
            assert!(l[qi] > 0.0, "tiled FA: query {} saw zero unmasked keys", qi);
            let inv_l = 1.0 / l[qi];
            for d in 0..head_dim {
                out[(q_tile_start + qi) * head_dim + d] = o[qi * head_dim + d] * inv_l;
            }
        }
        q_tile_start += br;
    }
    out
}

fn rand_floats(n: usize, seed: u64) -> Vec<f32> {
    // Simple xorshift PRNG, deterministic by seed.
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to [-1, 1].
        let bits = (state & 0x00FFFFFF) as u32;
        let v = (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        out.push(v);
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_one(
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    q_start: usize,
    br: usize,
    bc: usize,
    causal: bool,
    seed: u64,
) -> f32 {
    let q = rand_floats(seq_len * head_dim, seed);
    let k = rand_floats(kv_len * head_dim, seed.wrapping_add(1));
    let v = rand_floats(kv_len * head_dim, seed.wrapping_add(2));
    let naive = naive_attn(&q, &k, &v, head_dim, seq_len, kv_len, q_start, causal);
    let tiled = flash_attn_tiled_ref(
        &q, &k, &v, head_dim, seq_len, kv_len, q_start, br, bc, causal,
    );
    let diff = max_abs_diff(&naive, &tiled);
    eprintln!(
        "[fa_ref] hd={head_dim} seq={seq_len} kv={kv_len} q_start={q_start} \
         br={br} bc={bc} causal={causal} max_abs={diff:.2e}"
    );
    diff
}

#[test]
fn fa_ref_small_no_causal() {
    let diff = run_one(128, 16, 16, 0, 4, 4, false, 42);
    assert!(diff < 1e-4, "small no-causal mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_medium_no_causal() {
    let diff = run_one(128, 64, 64, 0, 16, 32, false, 43);
    assert!(diff < 1e-4, "medium no-causal mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_causal_short() {
    let diff = run_one(128, 32, 32, 0, 16, 16, true, 44);
    assert!(diff < 1e-4, "causal short mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_causal_chunked() {
    // Chunk-2 simulation: queries 0..32 are at global positions
    // 64..96, KV cache holds positions 0..96 from the previous chunk
    // plus this one. Causal mask active across the whole range.
    let diff = run_one(128, 32, 96, 64, 8, 16, true, 45);
    assert!(diff < 1e-4, "causal chunked mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_partial_q_tile() {
    // seq_len=30 is not a multiple of br=16 — last Q-tile is partial.
    let diff = run_one(128, 30, 64, 0, 16, 32, true, 46);
    assert!(diff < 1e-4, "partial Q-tile mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_partial_kv_tile() {
    // kv_len=50 is not a multiple of bc=32 — last K-tile is partial.
    let diff = run_one(128, 32, 50, 0, 16, 32, true, 47);
    assert!(diff < 1e-4, "partial KV-tile mismatch: {diff:.2e}");
}

#[test]
fn fa_ref_br_invariance() {
    // Same input, different Br/Bc — output must agree (modulo
    // rounding). Tightly bounds floating-point order divergence.
    let q = rand_floats(64 * 128, 100);
    let k = rand_floats(128 * 128, 101);
    let v = rand_floats(128 * 128, 102);
    let ref_naive = naive_attn(&q, &k, &v, 128, 64, 128, 0, true);
    for &(br, bc) in &[(1usize, 8usize), (4, 16), (8, 32), (16, 32), (16, 64)] {
        let tiled = flash_attn_tiled_ref(
            &q, &k, &v, 128, 64, 128, 0, br, bc, true,
        );
        let diff = max_abs_diff(&ref_naive, &tiled);
        eprintln!("[fa_ref] br={br} bc={bc} max_abs={diff:.2e}");
        assert!(diff < 1e-4, "br={br} bc={bc} drifted: {diff:.2e}");
    }
}

#[test]
fn fa_ref_long_kv() {
    // pp=128 in chunk-1 + 1024 prior KV is a realistic Sprint-5B
    // chunked-prefill scenario.
    let diff = run_one(128, 128, 1152, 1024, 16, 32, true, 200);
    assert!(diff < 1e-4, "long KV mismatch: {diff:.2e}");
}
