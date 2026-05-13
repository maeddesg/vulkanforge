//! AVX-512 accelerated Q4_K quantization — gated behind runtime
//! feature detection. Bit-identity target: each 144-byte block must
//! match the scalar path in `quantize.rs` byte-for-byte. The fallback
//! is the scalar path; this file is unused on non-x86_64 builds and on
//! x86_64 CPUs without AVX-512F at runtime.
//!
//! Sprint v0.4.x — AVX-512 Quantizer Phase 2. References:
//! `results/avx512_quantizer_analysis.md` (Phase 1 plan).
//!
//! **Bit-identity strategy:** the hot loops compute per-element values
//! (`w*x`, `w*l`, `w*l*l`, `iscale * (x - min)`, the nearest-int magic
//! and clamp) in 16-wide AVX-512. Horizontal reductions, however, are
//! done in scalar **sequential** order via a stack store — `_mm512_reduce_*_ps`
//! is a butterfly-tree sum that differs from scalar `for i in 0..n { s += v[i] }`
//! at ULP level, which propagates through `cur_error < best_error`
//! comparisons and breaks bit-identity on adversarial inputs.
//! FMA (`_mm512_fmadd_ps`) is also avoided in the accumulators — the
//! scalar path is built without `target_feature=+fma` so it uses
//! separate mul+add (two roundings), which we mirror.
//!
//! min / max reductions use SIMD-fast `_mm512_reduce_min_ps` /
//! `_max_ps` — for non-NaN floats these are order-independent.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

const QK_K: usize = 256;
const SUB_LEN: usize = 32;
const N_SUB: usize = 8;
const K_SCALE_SIZE: usize = 12;
const Q4K_BLOCK_BYTES: usize = 144;

/// Runtime-detect entry point. Cheap to call once per `par_chunks_mut`
/// closure but should be lifted out of any inner loop.
#[inline]
pub fn avx512_available() -> bool {
    if std::env::var_os("VF_NO_AVX512_QUANT").is_some() {
        return false;
    }
    is_x86_feature_detected!("avx512f")
}

/// 16-wide AVX-512 implementation of `quantize.rs::nearest_int`.
/// Bit-identical: `(v + 12_582_912.0).bits() & 0x007F_FFFF - 0x0040_0000`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn nearest_int_avx512(v: __m512) -> __m512i {
    unsafe {
        let magic = _mm512_set1_ps(12_582_912.0);
        let mantissa_mask = _mm512_set1_epi32(0x007F_FFFF);
        let mantissa_offset = _mm512_set1_epi32(0x0040_0000);
        let added = _mm512_add_ps(v, magic);
        let bits = _mm512_castps_si512(added);
        _mm512_sub_epi32(_mm512_and_si512(bits, mantissa_mask), mantissa_offset)
    }
}

/// Clamp each i32 lane to `[0, nmax_v]`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn clamp_i32_avx512(v: __m512i, nmax_v: __m512i, zero_v: __m512i) -> __m512i {
    unsafe { _mm512_max_epi32(_mm512_min_epi32(v, nmax_v), zero_v) }
}

/// Sequential-order reduction over 32 lanes packed into two ZMMs.
/// Matches scalar `s = a[0]; for i in 1..32 { s += a[i] }` exactly.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn seq_reduce32(lo: __m512, hi: __m512) -> f32 {
    let mut buf = [0f32; SUB_LEN];
    unsafe {
        _mm512_storeu_ps(buf.as_mut_ptr(), lo);
        _mm512_storeu_ps(buf.as_mut_ptr().add(16), hi);
    }
    let mut s = buf[0];
    for v in buf.iter().take(SUB_LEN).skip(1) {
        s += *v;
    }
    s
}

/// 32-element `make_qkx2`. SAFETY: caller must ensure `x.len() == SUB_LEN`,
/// `weights.len() == SUB_LEN`, AVX-512F available, and `nmax >= 0`.
#[target_feature(enable = "avx512f")]
unsafe fn make_qkx2_avx512(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32, [u8; SUB_LEN]) {
    debug_assert_eq!(x.len(), SUB_LEN);
    debug_assert_eq!(weights.len(), SUB_LEN);

    let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr()) };
    let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(16)) };
    let w_lo = unsafe { _mm512_loadu_ps(weights.as_ptr()) };
    let w_hi = unsafe { _mm512_loadu_ps(weights.as_ptr().add(16)) };

    // PART A — min / max are order-invariant for non-NaN floats, so SIMD
    // reduce matches scalar.
    let mut min_f = unsafe {
        _mm512_reduce_min_ps(_mm512_min_ps(x_lo, x_hi))
    };
    let max_f = unsafe {
        _mm512_reduce_max_ps(_mm512_max_ps(x_lo, x_hi))
    };
    // sum_w, sum_x — sequential-order reduction for bit-identity.
    let sum_w_f = unsafe { seq_reduce32(w_lo, w_hi) };
    let wx_lo = unsafe { _mm512_mul_ps(w_lo, x_lo) };
    let wx_hi = unsafe { _mm512_mul_ps(w_hi, x_hi) };
    let sum_x_f = unsafe { seq_reduce32(wx_lo, wx_hi) };

    if min_f > 0.0 {
        min_f = 0.0;
    }
    let mut l_out = [0u8; SUB_LEN];
    if (max_f - min_f).abs() < f32::EPSILON {
        return (0.0, -min_f, l_out);
    }

    let nmax_v = unsafe { _mm512_set1_epi32(nmax) };
    let zero_v = unsafe { _mm512_setzero_si512() };

    // PART B — initial L + best_error using iscale = nmax / (max - min).
    let mut min_var = min_f;
    let mut min_vec = unsafe { _mm512_set1_ps(min_var) };
    let iscale = nmax as f32 / (max_f - min_var);
    let mut scale = 1.0 / iscale;
    let iscale_v = unsafe { _mm512_set1_ps(iscale) };

    let scaled_lo = unsafe { _mm512_mul_ps(_mm512_sub_ps(x_lo, min_vec), iscale_v) };
    let scaled_hi = unsafe { _mm512_mul_ps(_mm512_sub_ps(x_hi, min_vec), iscale_v) };
    let li_lo = unsafe { nearest_int_avx512(scaled_lo) };
    let li_hi = unsafe { nearest_int_avx512(scaled_hi) };
    let l_lo = unsafe { clamp_i32_avx512(li_lo, nmax_v, zero_v) };
    let l_hi = unsafe { clamp_i32_avx512(li_hi, nmax_v, zero_v) };
    let lf_lo = unsafe { _mm512_cvtepi32_ps(l_lo) };
    let lf_hi = unsafe { _mm512_cvtepi32_ps(l_hi) };

    // diff = scale*l + min - x  (two ops: scale*l then add (min-x))
    // Matches scalar `scale * l + min - x` evaluation order.
    let scale_v_b = unsafe { _mm512_set1_ps(scale) };
    let sl_lo = unsafe { _mm512_mul_ps(scale_v_b, lf_lo) };
    let sl_hi = unsafe { _mm512_mul_ps(scale_v_b, lf_hi) };
    let diff_lo = unsafe { _mm512_sub_ps(_mm512_add_ps(sl_lo, min_vec), x_lo) };
    let diff_hi = unsafe { _mm512_sub_ps(_mm512_add_ps(sl_hi, min_vec), x_hi) };
    // err = w * diff * diff (two muls, not FMA)
    let d2_lo = unsafe { _mm512_mul_ps(diff_lo, diff_lo) };
    let d2_hi = unsafe { _mm512_mul_ps(diff_hi, diff_hi) };
    let wd2_lo = unsafe { _mm512_mul_ps(w_lo, d2_lo) };
    let wd2_hi = unsafe { _mm512_mul_ps(w_hi, d2_hi) };
    let mut best_error = unsafe { seq_reduce32(wd2_lo, wd2_hi) };

    unsafe { store_l(&mut l_out, l_lo, l_hi) };

    if nstep < 1 {
        return (scale, -min_var, l_out);
    }

    // PART C — nstep rescaling.
    for is in 0..=nstep {
        let denom = max_f - min_var;
        if denom <= 0.0 {
            // Defensive: matches scalar (initial early-exit guards the
            // `==` case; updates only decrease min_var so denom grows).
            continue;
        }
        let iscale_try = (rmin + rdelta * is as f32 + nmax as f32) / denom;
        let iscale_try_v = unsafe { _mm512_set1_ps(iscale_try) };

        let scaled_lo = unsafe { _mm512_mul_ps(_mm512_sub_ps(x_lo, min_vec), iscale_try_v) };
        let scaled_hi = unsafe { _mm512_mul_ps(_mm512_sub_ps(x_hi, min_vec), iscale_try_v) };
        let li_lo = unsafe { nearest_int_avx512(scaled_lo) };
        let li_hi = unsafe { nearest_int_avx512(scaled_hi) };
        let l_lo = unsafe { clamp_i32_avx512(li_lo, nmax_v, zero_v) };
        let l_hi = unsafe { clamp_i32_avx512(li_hi, nmax_v, zero_v) };
        let lf_lo = unsafe { _mm512_cvtepi32_ps(l_lo) };
        let lf_hi = unsafe { _mm512_cvtepi32_ps(l_hi) };

        // sum_l  = Σ w*l
        let wl_lo = unsafe { _mm512_mul_ps(w_lo, lf_lo) };
        let wl_hi = unsafe { _mm512_mul_ps(w_hi, lf_hi) };
        let sum_l = unsafe { seq_reduce32(wl_lo, wl_hi) };
        // sum_l2 = Σ w * l * l. Scalar: `w * (l as f32) * (l as f32)` = (w*l)*l, no FMA.
        let wll_lo = unsafe { _mm512_mul_ps(wl_lo, lf_lo) };
        let wll_hi = unsafe { _mm512_mul_ps(wl_hi, lf_hi) };
        let sum_l2 = unsafe { seq_reduce32(wll_lo, wll_hi) };
        // sum_xl = Σ w * l * x. Scalar: `w * (l as f32) * x[i]` = (w*l)*x, no FMA.
        let wlx_lo = unsafe { _mm512_mul_ps(wl_lo, x_lo) };
        let wlx_hi = unsafe { _mm512_mul_ps(wl_hi, x_hi) };
        let sum_xl = unsafe { seq_reduce32(wlx_lo, wlx_hi) };

        let d = sum_w_f * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w_f * sum_xl - sum_x_f * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x_f - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let ts_v = unsafe { _mm512_set1_ps(this_scale) };
            let tm_v = unsafe { _mm512_set1_ps(this_min) };
            // diff = this_scale * l + this_min - x  (two ops, no FMA)
            let sl_lo = unsafe { _mm512_mul_ps(ts_v, lf_lo) };
            let sl_hi = unsafe { _mm512_mul_ps(ts_v, lf_hi) };
            let diff_lo = unsafe { _mm512_sub_ps(_mm512_add_ps(sl_lo, tm_v), x_lo) };
            let diff_hi = unsafe { _mm512_sub_ps(_mm512_add_ps(sl_hi, tm_v), x_hi) };
            let d2_lo = unsafe { _mm512_mul_ps(diff_lo, diff_lo) };
            let d2_hi = unsafe { _mm512_mul_ps(diff_hi, diff_hi) };
            let wd2_lo = unsafe { _mm512_mul_ps(w_lo, d2_lo) };
            let wd2_hi = unsafe { _mm512_mul_ps(w_hi, d2_hi) };
            let cur_error = unsafe { seq_reduce32(wd2_lo, wd2_hi) };

            if cur_error < best_error {
                unsafe { store_l(&mut l_out, l_lo, l_hi) };
                best_error = cur_error;
                scale = this_scale;
                min_var = this_min;
                min_vec = tm_v;
            }
        }
    }
    (scale, -min_var, l_out)
}

/// Store 32 i32 lanes (∈ [0, 15]) into a `[u8; 32]` slot.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_l(dst: &mut [u8; SUB_LEN], l_lo: __m512i, l_hi: __m512i) {
    let mut buf = [0i32; SUB_LEN];
    unsafe {
        _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, l_lo);
        _mm512_storeu_si512(buf.as_mut_ptr().add(16) as *mut __m512i, l_hi);
    }
    for i in 0..SUB_LEN {
        dst[i] = buf[i] as u8;
    }
}

/// Inverse of the 6-bit pack — duplicate of `quantize::unpack_scale_min_k4`.
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

/// AVX-512 version of `quantize_block_q4k`. Step 1 calls `make_qkx2_avx512`
/// 8×. Step 5 uses SIMD for the 32-element re-quant per sub-block. Steps
/// 2/3/4/6 stay scalar (each is <1% of total time).
///
/// SAFETY: `input.len() == QK_K`, `output.len() == Q4K_BLOCK_BYTES`,
/// AVX-512F available.
#[target_feature(enable = "avx512f")]
pub unsafe fn quantize_block_q4k_avx512(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), QK_K);
    debug_assert_eq!(output.len(), Q4K_BLOCK_BYTES);

    // Step 1 — per-sub-block make_qkx2.
    let mut scales = [0f32; N_SUB];
    let mut mins = [0f32; N_SUB];
    let mut l_all = [0u8; QK_K];
    let mut weights = [0f32; SUB_LEN];

    for j in 0..N_SUB {
        let xj = &input[j * SUB_LEN..(j + 1) * SUB_LEN];

        // sum_x2 = Σ x*x — sequential order to match scalar
        // `iter().map(|v| v*v).sum()` which uses scalar fold.
        let x_lo = unsafe { _mm512_loadu_ps(xj.as_ptr()) };
        let x_hi = unsafe { _mm512_loadu_ps(xj.as_ptr().add(16)) };
        let sq_lo = unsafe { _mm512_mul_ps(x_lo, x_lo) };
        let sq_hi = unsafe { _mm512_mul_ps(x_hi, x_hi) };
        let sum_x2 = unsafe { seq_reduce32(sq_lo, sq_hi) };
        let av_x = (sum_x2 / SUB_LEN as f32).sqrt();

        // weights[l] = av_x + |x[l]|
        let avx_v = unsafe { _mm512_set1_ps(av_x) };
        let abs_mask = unsafe { _mm512_set1_epi32(0x7FFF_FFFFu32 as i32) };
        let abs_lo = unsafe {
            _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x_lo), abs_mask))
        };
        let abs_hi = unsafe {
            _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x_hi), abs_mask))
        };
        unsafe {
            _mm512_storeu_ps(weights.as_mut_ptr(),         _mm512_add_ps(avx_v, abs_lo));
            _mm512_storeu_ps(weights.as_mut_ptr().add(16), _mm512_add_ps(avx_v, abs_hi));
        }

        let (sc, abs_min, lq) =
            unsafe { make_qkx2_avx512(xj, &weights, 15, -1.0, 0.1, 20) };
        scales[j] = sc;
        mins[j] = abs_min;
        l_all[j * SUB_LEN..(j + 1) * SUB_LEN].copy_from_slice(&lq);
    }

    // Step 2 — global d/dmin.
    let max_scale = scales.iter().cloned().fold(0f32, f32::max);
    let max_min = mins.iter().cloned().fold(0f32, f32::max);
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    // Step 3 — pack ls[8] + lm[8] (each 6-bit) into 12-byte buf.
    let mut scales_buf = [0u8; K_SCALE_SIZE];
    for j in 0..N_SUB {
        let ls = (nearest_int_scalar(inv_scale * scales[j]) as i32).clamp(0, 63) as u8;
        let lm = (nearest_int_scalar(inv_min * mins[j]) as i32).clamp(0, 63) as u8;
        if j < 4 {
            scales_buf[j] = ls;
            scales_buf[j + 4] = lm;
        } else {
            scales_buf[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
            scales_buf[j - 4] |= (ls >> 4) << 6;
            scales_buf[j] |= (lm >> 4) << 6;
        }
    }

    // Step 4 — write d/dmin (FP16) and scales_buf.
    let d_fp32 = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
    let dmin_fp32 = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
    let d_h = half::f16::from_f32(d_fp32);
    let dmin_h = half::f16::from_f32(dmin_fp32);
    output[0..2].copy_from_slice(&d_h.to_bits().to_le_bytes());
    output[2..4].copy_from_slice(&dmin_h.to_bits().to_le_bytes());
    output[4..4 + K_SCALE_SIZE].copy_from_slice(&scales_buf);

    // Step 5 — re-quantize with globally-quantized scales/mins.
    let d_decoded = d_h.to_f32();
    let dmin_decoded = dmin_h.to_f32();
    let nmax_q4 = unsafe { _mm512_set1_epi32(15) };
    let zero_v = unsafe { _mm512_setzero_si512() };
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
        let dm_v = unsafe { _mm512_set1_ps(dm_eff) };
        let d_eff_v = unsafe { _mm512_set1_ps(d_eff) };
        let xj = &input[j * SUB_LEN..(j + 1) * SUB_LEN];
        let x_lo = unsafe { _mm512_loadu_ps(xj.as_ptr()) };
        let x_hi = unsafe { _mm512_loadu_ps(xj.as_ptr().add(16)) };
        // v = nearest_int((x + dm_eff) / d_eff). Division (not rcp+mul)
        // for bit-parity with scalar.
        let v_lo = unsafe { _mm512_div_ps(_mm512_add_ps(x_lo, dm_v), d_eff_v) };
        let v_hi = unsafe { _mm512_div_ps(_mm512_add_ps(x_hi, dm_v), d_eff_v) };
        let li_lo = unsafe { nearest_int_avx512(v_lo) };
        let li_hi = unsafe { nearest_int_avx512(v_hi) };
        let l_lo = unsafe { clamp_i32_avx512(li_lo, nmax_q4, zero_v) };
        let l_hi = unsafe { clamp_i32_avx512(li_hi, nmax_q4, zero_v) };
        let mut tmp = [0u8; SUB_LEN];
        unsafe { store_l(&mut tmp, l_lo, l_hi) };
        l_all[j * SUB_LEN..(j + 1) * SUB_LEN].copy_from_slice(&tmp);
    }

    // Step 6 — pack 4-bit values into qs[128]. Scalar.
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

#[inline]
fn nearest_int_scalar(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let bits = val.to_bits() as i32;
    (bits & 0x007F_FFFF) - 0x0040_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_avx512() -> bool {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f not available — skipping");
            return true;
        }
        false
    }

    fn scalar_block(input: &[f32]) -> [u8; Q4K_BLOCK_BYTES] {
        let bytes = crate::quantize::quantize_f32_to_q4k_serial(input);
        assert_eq!(bytes.len(), Q4K_BLOCK_BYTES);
        let mut out = [0u8; Q4K_BLOCK_BYTES];
        out.copy_from_slice(&bytes);
        out
    }

    fn avx_block(input: &[f32]) -> [u8; Q4K_BLOCK_BYTES] {
        let mut out = [0u8; Q4K_BLOCK_BYTES];
        unsafe { quantize_block_q4k_avx512(input, &mut out) };
        out
    }

    #[test]
    fn avx512_bit_identical_all_zeros() {
        if skip_if_no_avx512() { return; }
        let input = vec![0.0f32; QK_K];
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    #[test]
    fn avx512_bit_identical_uniform_ramp() {
        if skip_if_no_avx512() { return; }
        let input: Vec<f32> = (0..QK_K).map(|i| i as f32 / (QK_K - 1) as f32).collect();
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    #[test]
    fn avx512_bit_identical_signed_uniform() {
        if skip_if_no_avx512() { return; }
        let input: Vec<f32> = (0..QK_K)
            .map(|i| 2.0 * (i as f32 / (QK_K - 1) as f32) - 1.0)
            .collect();
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    #[test]
    fn avx512_bit_identical_sin_block() {
        if skip_if_no_avx512() { return; }
        let input: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.013).sin()).collect();
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    #[test]
    fn avx512_bit_identical_lcg_noise() {
        if skip_if_no_avx512() { return; }
        let mut s: u32 = 0xCAFE_BABE;
        let input: Vec<f32> = (0..QK_K)
            .map(|_| {
                s = s.wrapping_mul(48271);
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    #[test]
    fn avx512_bit_identical_mixed_smooth_plus_noise() {
        if skip_if_no_avx512() { return; }
        let mut s: u32 = 0x1234_5678;
        let input: Vec<f32> = (0..QK_K)
            .map(|i| {
                s = s.wrapping_mul(48271);
                let noise = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
                (i as f32 * 0.013).sin() * 0.5 + noise * 0.05
            })
            .collect();
        assert_eq!(scalar_block(&input), avx_block(&input));
    }

    /// Stress test: 100 blocks (same input the canonical
    /// `parallel_matches_serial` test in `quantize.rs` uses) — bit-identity
    /// must hold per-block AND across the whole multi-block buffer.
    #[test]
    fn avx512_bit_identical_100_block_stress() {
        if skip_if_no_avx512() { return; }
        let mut input: Vec<f32> = Vec::with_capacity(QK_K * 100);
        let mut s: u32 = 0xCAFE_BABE;
        for i in 0..QK_K * 100 {
            s = s.wrapping_mul(48271);
            let noise = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
            input.push((i as f32 * 0.013).sin() * 0.5 + noise * 0.05);
        }
        for b in 0..100 {
            let block = &input[b * QK_K..(b + 1) * QK_K];
            let scalar = scalar_block(block);
            let avx = avx_block(block);
            assert_eq!(
                scalar, avx,
                "block {b} of 100 diverges (first 32 scalar={:?} avx={:?})",
                &scalar[..32], &avx[..32]
            );
        }
    }

    #[test]
    fn avx512_dispatcher_drives_avx_when_available() {
        let want = is_x86_feature_detected!("avx512f")
            && std::env::var_os("VF_NO_AVX512_QUANT").is_none();
        assert_eq!(avx512_available(), want);
    }
}
