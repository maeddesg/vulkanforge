//! Q4_K_M (Q4_K) quantization — pure Rust port of llama.cpp's
//! `quantize_row_q4_K_ref` (`ggml-quants.c:1395`).
//!
//! Block layout (144 B, super-block = 256 elements = 8 sub-blocks × 32):
//!
//! ```text
//! offset  size  field
//!   0      2    d           FP16 super-block scale
//!   2      2    dmin        FP16 super-block min
//!   4     12    scales[12]  6-bit packed: 8 sub-block scales + 8 sub-block mins
//!  16    128    qs[128]     4-bit quantized values, 2 per byte
//! ```
//!
//! `scales[12]` packs 16 × 6-bit values (8 sub-block scales `ls[0..8]` and
//! 8 sub-block mins `lm[0..8]`) into 12 bytes per the inverse of
//! `get_scale_min_k4` (`ggml-quants.c:818`):
//!
//! - `j < 4`: `scales[j] = ls[j]`,        `scales[j+4] = lm[j]`     (lower 6 bits)
//! - `j ≥ 4`: `scales[j+4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)`,
//!            `scales[j-4] |= (ls[j] >> 4) << 6`,
//!            `scales[j  ] |= (lm[j] >> 4) << 6`
//!
//! Quantized values per sub-block: `q = clamp(round((x + dmin*lm) / (d*ls)), 0, 15)`.
//! Two adjacent sub-blocks (j, j+1) interleave into 32 `qs` bytes:
//! `qs[l] = L[low_j*32 + l] | (L[high_j*32 + l] << 4)` for l in 0..32.
//!
//! The reference implementation uses `make_qkx2_quants` for per-sub-block
//! scale/min selection — initialise from min/max, then iterate `nstep=20`
//! candidate rescalings and keep the one with lowest weighted MSE. This
//! Rust port preserves that algorithm verbatim.
//!
//! Sprint 50A. Performance is not optimised — pure scalar Rust, intended
//! for one-shot model-load quantization. AVX-512 / SIMD is a follow-up
//! if load time becomes a bottleneck.

use half::f16;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const Q4K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144

const N_SUB: usize = 8;            // sub-blocks per super-block
const SUB_LEN: usize = QK_K / N_SUB; // = 32 elements per sub-block

/// Banker's rounding via FP magic — bit-identical to llama.cpp's
/// `nearest_int` (`ggml-quants.c:559`).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let bits = val.to_bits() as i32;
    (bits & 0x007F_FFFF) - 0x0040_0000
}

/// Per-sub-block scale + min selector, port of `make_qkx2_quants`
/// (`ggml-quants.c:737`). Returns `(scale, abs_min, L_quants)` where
/// `L_quants[i] ∈ [0, nmax]` and `dequant_i ≈ scale * L_quants[i] - abs_min`.
///
/// Initial linear fit, then `nstep` MSE-minimised rescalings. Per
/// `quantize_row_q4_K_ref` call site: `nmax=15`, `rmin=-1.0`,
/// `rdelta=0.1`, `nstep=20`, `use_mad=false`.
fn make_qkx2(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32, [u8; SUB_LEN]) {
    let n = x.len();
    debug_assert_eq!(n, weights.len());

    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    let mut l_out = [0u8; SUB_LEN];
    if (max - min).abs() < f32::EPSILON {
        return (0.0, -min, l_out);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax) as u8;
        l_out[i] = l;
        let diff = scale * l as f32 + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    if nstep < 1 {
        return (scale, -min, l_out);
    }

    let mut laux = [0u8; SUB_LEN];
    for is in 0..=nstep {
        let iscale_try = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_try * (x[i] - min)).clamp(0, nmax) as u8;
            laux[i] = l;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l as f32) * (l as f32);
            sum_xl += w * (l as f32) * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if cur_error < best_error {
                l_out.copy_from_slice(&laux);
                best_error = cur_error;
                scale = this_scale;
                iscale = 1.0 / this_scale;
                min = this_min;
                let _ = iscale;
            }
        }
    }
    (scale, -min, l_out)
}

/// Pack one 256-element super-block into 144 bytes. Mirrors
/// `quantize_row_q4_K_ref` (`ggml-quants.c:1395-1465`).
fn quantize_block_q4k(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), QK_K);
    debug_assert_eq!(output.len(), Q4K_BLOCK_BYTES);

    // Step 1 — per-sub-block make_qkx2.
    let mut scales = [0f32; N_SUB];
    let mut mins = [0f32; N_SUB];
    let mut l_all = [0u8; QK_K];
    let mut weights = [0f32; SUB_LEN];

    for j in 0..N_SUB {
        let xj = &input[j * SUB_LEN..(j + 1) * SUB_LEN];
        let sum_x2: f32 = xj.iter().map(|v| v * v).sum();
        let av_x = (sum_x2 / SUB_LEN as f32).sqrt();
        for l in 0..SUB_LEN {
            weights[l] = av_x + xj[l].abs();
        }
        let (sc, abs_min, lq) = make_qkx2(xj, &weights, 15, -1.0, 0.1, 20);
        scales[j] = sc;
        mins[j] = abs_min;
        l_all[j * SUB_LEN..(j + 1) * SUB_LEN].copy_from_slice(&lq);
    }

    // Step 2 — global d/dmin from max sub-block scale/min.
    let max_scale = scales.iter().cloned().fold(0f32, f32::max);
    let max_min = mins.iter().cloned().fold(0f32, f32::max);
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    // Step 3 — pack ls[8] + lm[8] (each 6-bit) into a local 12-byte buf.
    // Done in a local array so output[..] can be written for d/dmin first
    // and then the scales block copied in without overlapping borrows.
    let mut scales_buf = [0u8; K_SCALE_SIZE];
    for j in 0..N_SUB {
        let ls = (nearest_int(inv_scale * scales[j]) as i32).clamp(0, 63) as u8;
        let lm = (nearest_int(inv_min * mins[j]) as i32).clamp(0, 63) as u8;
        if j < 4 {
            scales_buf[j] = ls;
            scales_buf[j + 4] = lm;
        } else {
            scales_buf[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
            scales_buf[j - 4] |= (ls >> 4) << 6;
            scales_buf[j] |= (lm >> 4) << 6;
        }
    }

    // Step 4 — write d/dmin (FP16) into output[0..4] and copy scales_buf.
    let d_fp32 = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
    let dmin_fp32 = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
    let d_h = f16::from_f32(d_fp32);
    let dmin_h = f16::from_f32(dmin_fp32);
    output[0..2].copy_from_slice(&d_h.to_bits().to_le_bytes());
    output[2..4].copy_from_slice(&dmin_h.to_bits().to_le_bytes());
    output[4..4 + K_SCALE_SIZE].copy_from_slice(&scales_buf);

    // Step 5 — re-quantize with the now-globally-quantized scales / mins,
    // matching dequantize_row_q4_K's read-back. This is the "L" recompute
    // that quantize_row_q4_K_ref does after writing scales[].
    let d_decoded = d_h.to_f32();
    let dmin_decoded = dmin_h.to_f32();
    for j in 0..N_SUB {
        let (sc, m) = unpack_scale_min_k4(j, &scales_buf);
        let d_eff = d_decoded * sc as f32;
        if d_eff == 0.0 {
            for ii in 0..SUB_LEN {
                l_all[j * SUB_LEN + ii] = 0;
            }
            continue;
        }
        let dm_eff = dmin_decoded * m as f32;
        for ii in 0..SUB_LEN {
            let v = nearest_int((input[j * SUB_LEN + ii] + dm_eff) / d_eff);
            l_all[j * SUB_LEN + ii] = v.clamp(0, 15) as u8;
        }
    }

    // Step 6 — pack 4-bit values into qs[128]. Pairs of sub-blocks
    // (0+1, 2+3, 4+5, 6+7) interleave: qs[l] = L_low[l] | (L_high[l] << 4).
    let qs = &mut output[16..16 + QK_K / 2];
    for pair in 0..4 {
        let dst = &mut qs[pair * 32..(pair + 1) * 32];
        let lo_off = pair * 64;
        let hi_off = pair * 64 + 32;
        for l in 0..32 {
            dst[l] = l_all[lo_off + l] | (l_all[hi_off + l] << 4);
        }
    }
}

/// Inverse of the 6-bit pack — port of `get_scale_min_k4`
/// (`ggml-quants.c:818`).
#[inline]
fn unpack_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Quantize a flat FP32 row to Q4_K_M blocks. Length must be a multiple
/// of `QK_K` (256). Sprint 50C — block-level rayon parallelism over the
/// (independent) super-blocks; each block is bit-identical to the serial
/// path.
pub fn quantize_f32_to_q4k(input: &[f32]) -> Vec<u8> {
    assert!(
        input.len() % QK_K == 0,
        "Q4_K requires input length to be a multiple of {QK_K} (got {})",
        input.len()
    );
    let n_blocks = input.len() / QK_K;
    let mut output = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
    output
        .par_chunks_mut(Q4K_BLOCK_BYTES)
        .enumerate()
        .for_each(|(b, yi)| {
            let xi = &input[b * QK_K..(b + 1) * QK_K];
            quantize_block_q4k(xi, yi);
        });
    output
}

/// Sprint 51D-D — Q4_K quantizer that pads each "row" to the next
/// multiple of 256 with zeros before quantization. Required for 3D
/// expert weights whose innermost dim isn't 256-aligned (Gemma-4-26B-A4B
/// `experts.down_proj` has K=704 = `moe_intermediate_size`; Q4_K's
/// 256-element block layout would otherwise drop the last 192
/// elements per row when the GEMV shader runs with `ncols/QUANT_K`
/// integer-truncated to 2 instead of 2.75).
///
/// Inputs:
/// - `input` length = `n_rows × k_orig`. For 3D expert tensors, `n_rows`
///   collapses the leading two dims (e.g., `[128 × 2816, 704]` for
///   26B's down_proj).
/// - `k_orig`: original innermost dim length.
/// - `k_padded`: target innermost dim length, must be a multiple of 256
///   and ≥ `k_orig`. Padded positions are zero-filled before
///   quantization so the dequantized result is exactly 0 there
///   (verified by `all_zeros_round_trip`); the padded zeros contribute
///   nothing to the GEMV sum, so the math is unchanged.
///
/// Output bytes layout matches the standard Q4_K row format:
/// `n_rows × (k_padded / 256) × 144` bytes. The shader dispatch pushes
/// `ncols = k_padded` (not `k_orig`); the input vector binding may
/// have garbage past `k_orig` but is multiplied by zero.
pub fn quantize_f32_to_q4k_padded_rows(
    input: &[f32],
    n_rows: usize,
    k_orig: usize,
    k_padded: usize,
) -> Vec<u8> {
    assert!(
        k_padded % QK_K == 0,
        "k_padded {k_padded} must be a multiple of {QK_K}"
    );
    assert!(
        k_padded >= k_orig,
        "k_padded {k_padded} must be ≥ k_orig {k_orig}"
    );
    assert_eq!(
        input.len(),
        n_rows * k_orig,
        "input len mismatch: got {}, expected {} × {} = {}",
        input.len(),
        n_rows,
        k_orig,
        n_rows * k_orig,
    );
    let n_blocks_per_row = k_padded / QK_K;
    let total_blocks = n_rows * n_blocks_per_row;
    let mut output = vec![0u8; total_blocks * Q4K_BLOCK_BYTES];
    output
        .par_chunks_mut(n_blocks_per_row * Q4K_BLOCK_BYTES)
        .enumerate()
        .for_each(|(r, row_bytes)| {
            // Build a padded row in a thread-local buffer (cheap;
            // k_padded is at most a few thousand floats).
            let mut padded = vec![0.0_f32; k_padded];
            padded[..k_orig].copy_from_slice(&input[r * k_orig..(r + 1) * k_orig]);
            // padded[k_orig..] already zero-initialized.
            for b in 0..n_blocks_per_row {
                let xi = &padded[b * QK_K..(b + 1) * QK_K];
                let yi = &mut row_bytes[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES];
                quantize_block_q4k(xi, yi);
            }
        });
    output
}

/// Serial reference impl of `quantize_f32_to_q4k`. Public only via
/// `#[cfg(test)]` — used by the bit-identity test that proves the
/// rayon path is functionally equivalent.
#[cfg(test)]
pub(crate) fn quantize_f32_to_q4k_serial(input: &[f32]) -> Vec<u8> {
    assert!(input.len() % QK_K == 0);
    let n_blocks = input.len() / QK_K;
    let mut output = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let xi = &input[b * QK_K..(b + 1) * QK_K];
        let yi = &mut output[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES];
        quantize_block_q4k(xi, yi);
    }
    output
}

/// Inverse of `quantize_f32_to_q4k`. Mirrors `dequantize_row_q4_K`
/// (`ggml-quants.c:1467`). Validation only — not perf-tuned.
pub fn dequantize_q4k_to_f32(input: &[u8]) -> Vec<f32> {
    assert!(
        input.len() % Q4K_BLOCK_BYTES == 0,
        "Q4_K dequant input must be a multiple of {Q4K_BLOCK_BYTES} bytes"
    );
    let n_blocks = input.len() / Q4K_BLOCK_BYTES;
    let mut output = vec![0f32; n_blocks * QK_K];
    for b in 0..n_blocks {
        let blk = &input[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES];
        let out = &mut output[b * QK_K..(b + 1) * QK_K];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qs = &blk[16..16 + QK_K / 2];
        for pair in 0..4 {
            let q = &qs[pair * 32..(pair + 1) * 32];
            let (sc1, m1) = unpack_scale_min_k4(pair * 2, scales);
            let (sc2, m2) = unpack_scale_min_k4(pair * 2 + 1, scales);
            let d1 = d * sc1 as f32;
            let dm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let dm2 = dmin * m2 as f32;
            for l in 0..32 {
                out[pair * 64 + l] = d1 * (q[l] & 0x0F) as f32 - dm1;
                out[pair * 64 + 32 + l] = d2 * (q[l] >> 4) as f32 - dm2;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
    }

    #[test]
    fn block_size_matches_llama_cpp() {
        assert_eq!(Q4K_BLOCK_BYTES, 144);
    }

    #[test]
    fn all_zeros_round_trip() {
        let input = vec![0.0f32; QK_K];
        let q = quantize_f32_to_q4k(&input);
        let dq = dequantize_q4k_to_f32(&q);
        assert!(dq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn uniform_ramp_round_trip() {
        // 4-bit grid over [0, 1] — expected max_err ≈ 1/30 for the
        // optimal scale, looser bound for make_qkx2's MSE-minimisation.
        let input: Vec<f32> = (0..QK_K).map(|i| i as f32 / (QK_K - 1) as f32).collect();
        let q = quantize_f32_to_q4k(&input);
        let dq = dequantize_q4k_to_f32(&q);
        let err = max_abs_err(&input, &dq);
        assert!(err < 0.05, "uniform ramp max_abs_err = {err}");
    }

    #[test]
    fn signed_uniform_round_trip() {
        // [-1, 1] — exercises the dmin path (negative min).
        let input: Vec<f32> = (0..QK_K)
            .map(|i| 2.0 * (i as f32 / (QK_K - 1) as f32) - 1.0)
            .collect();
        let q = quantize_f32_to_q4k(&input);
        let dq = dequantize_q4k_to_f32(&q);
        let err = max_abs_err(&input, &dq);
        assert!(err < 0.10, "signed ramp max_abs_err = {err}");
    }

    #[test]
    fn pseudo_random_round_trip() {
        // Lehmer LCG, deterministic, std-dev ≈ 0.5.
        let mut s: u32 = 0x1234_5678;
        let mut next = || {
            s = s.wrapping_mul(48271);
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..QK_K).map(|_| next()).collect();
        let q = quantize_f32_to_q4k(&input);
        let dq = dequantize_q4k_to_f32(&q);
        let err = max_abs_err(&input, &dq);
        // 4-bit on a [-1,1] uniform distribution: expected ~1/30 = 0.033;
        // make_qkx2's MSE optimum may exceed that on a few outliers.
        assert!(err < 0.10, "pseudo-random max_abs_err = {err}");
    }

    #[test]
    fn multi_block_aligned() {
        let input: Vec<f32> = (0..QK_K * 5).map(|i| (i as f32 * 0.013).sin()).collect();
        let q = quantize_f32_to_q4k(&input);
        assert_eq!(q.len(), 5 * Q4K_BLOCK_BYTES);
        let dq = dequantize_q4k_to_f32(&q);
        let err = max_abs_err(&input, &dq);
        assert!(err < 0.05, "multi-block sin max_abs_err = {err}");
    }

    #[test]
    fn parallel_matches_serial() {
        // 100 super-blocks of varied content — exercise enough work that
        // rayon actually splits across threads, not just the auto
        // fall-back to a single thread.
        let mut input: Vec<f32> = Vec::with_capacity(QK_K * 100);
        let mut s: u32 = 0xCAFE_BABE;
        for i in 0..QK_K * 100 {
            // Mix a smooth sin component with deterministic noise so
            // both make_qkx2 branches (smooth + outlier-y) get hit.
            s = s.wrapping_mul(48271);
            let noise = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
            input.push((i as f32 * 0.013).sin() * 0.5 + noise * 0.05);
        }
        let serial = quantize_f32_to_q4k_serial(&input);
        let parallel = quantize_f32_to_q4k(&input);
        assert_eq!(
            serial.len(),
            parallel.len(),
            "serial / parallel byte-count differ"
        );
        // Bit-identity: per-block is independent, no shared state — the
        // two outputs MUST match byte-for-byte. Mismatch implies a
        // race-condition bug.
        assert_eq!(serial, parallel, "rayon output diverges from serial");
    }

    #[test]
    fn scale_min_pack_round_trip() {
        // Pack 8 random ls + 8 random lm into 12 bytes via the same
        // encode used in quantize_block_q4k, then unpack and compare.
        let ls: [u8; 8] = [0, 1, 30, 63, 5, 17, 42, 50];
        let lm: [u8; 8] = [60, 7, 0, 33, 18, 63, 1, 27];
        let mut packed = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                packed[j] = ls[j];
                packed[j + 4] = lm[j];
            } else {
                packed[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                packed[j - 4] |= (ls[j] >> 4) << 6;
                packed[j] |= (lm[j] >> 4) << 6;
            }
        }
        for j in 0..8 {
            let (sc, m) = unpack_scale_min_k4(j, &packed);
            assert_eq!(sc, ls[j], "ls[{j}]");
            assert_eq!(m, lm[j], "lm[{j}]");
        }
    }
}
