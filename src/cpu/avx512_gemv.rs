//! AVX-512 Q6_K GEMV inner loops for the CPU `lm_head` offload.
//!
//! Sprint 40 Part 2's scalar `dot_q6k_block` does `Q6_K dequant +
//! FMA` element-by-element. LLVM does not auto-vectorize the
//! per-element nibble unpack into AVX-512, so the scalar path
//! lands at ~17 tok/s decode on Llama-8B-FP8 (4× slower than the
//! GPU baseline).
//!
//! Sprint 41A: hybrid kernel. The Q6_K dequantization stays
//! scalar — picking 16 nibbles + 16 high-2-bit values from the
//! packed `ql[128]` / `qh[64]` arrays involves variable-position
//! shifts that AVX-512 can do but require non-trivial
//! `vpshufb` / `vpsrlvd` choreography. We produce the 16
//! dequantized FP32 values into a small stack scratch and feed
//! them to an `_mm512_fmadd_ps` against the matching 16-element
//! slice of the hidden state. The FMA chain replaces 16 scalar
//! mul-add pairs with one AVX-512 instruction; on Zen 4
//! double-pumped that's 16 FMAs in two cycles per port instead
//! of 16 separate scalar pair-issues.
//!
//! Sprint 41B: full vectorized dequant. The nibble unpack now
//! happens inside AVX-512 too — `_mm_unpacklo_epi8` /
//! `_mm_unpackhi_epi8` interleave the low/high QL nibbles into
//! the right element order, `_mm_shuffle_epi8` repeats QH bytes
//! 4× so each 4-element group lands in adjacent lanes,
//! `_mm512_srlv_epi32` does the per-lane variable shift for the
//! 2-bit high payload, and the `_mm512_cvtepi32_ps` +
//! `_mm512_fmadd_ps` chain finishes off two 16-element FMAs per
//! 32-element iteration. Eight such iterations cover one Q6_K
//! block (256 elements). See [`dot_q6k_block_avx512_full`].
//!
//! Calling convention: every public function here is `unsafe`
//! and gated on `#[target_feature(enable = "avx512f")]`. The
//! safe wrapper in [`crate::cpu::lm_head`] runtime-detects
//! AVX-512 via `is_x86_feature_detected!` and falls back to the
//! scalar implementation when absent.

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use crate::cpu::q6k::{Q6KBlock, QK_K};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// AVX-512 dot product: one Q6_K block · 256-element FP32 vector.
///
/// # Safety
/// Caller must ensure the running CPU supports AVX-512F (and BW
/// for the future vectorized-dequant version, but this hybrid
/// only needs F). Use `is_x86_feature_detected!("avx512f")` at the
/// safe-API boundary.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_q6k_block_avx512(block: &Q6KBlock, hidden: &[f32]) -> f32 {
    debug_assert_eq!(hidden.len(), QK_K);
    let d = block.d.to_f32();

    // 16-lane FP32 accumulator. The horizontal sum at the end
    // collapses it to a scalar.
    let mut acc = unsafe { _mm512_setzero_ps() };

    // Scratch buffer for the 16 scalar-dequantized weights of one
    // chunk. Stack-resident, no allocation. Writing through this
    // buffer lets the FMA see all 16 values as a single AVX-512
    // load.
    let mut vals = [0.0f32; 16];

    // 256 / 16 = 16 chunks per block. Each chunk lives entirely
    // inside one Q6_K sub-block (sub-block size = 16 elements),
    // so `sub_scale` is constant across the chunk and we hoist
    // its computation out of the inner loop.
    for chunk in 0..(QK_K / 16) {
        let base = chunk * 16;
        let sub_scale = d * block.scales[chunk] as f32;

        // Scalar Q6_K dequant for the 16 elements of this chunk.
        // Same algorithm as `Q6KBlock::dequantize`, just inlined
        // and pre-multiplied by the constant `sub_scale`.
        for i in 0..16 {
            let idx = base + i;
            let lo = (block.ql[idx >> 1] >> (4 * (idx & 1))) & 0x0F;
            let hi = (block.qh[idx >> 2] >> (2 * (idx & 3))) & 0x03;
            let qu = lo | (hi << 4);
            let q = qu as i32 - 32;
            vals[i] = sub_scale * q as f32;
        }

        // AVX-512 FMA: acc += dequant ⊙ hidden_chunk.
        unsafe {
            let dequant_vec = _mm512_loadu_ps(vals.as_ptr());
            let hidden_vec = _mm512_loadu_ps(hidden.as_ptr().add(base));
            acc = _mm512_fmadd_ps(dequant_vec, hidden_vec, acc);
        }
    }

    // Horizontal sum: collapse the 16 lanes to a single FP32.
    // `_mm512_reduce_add_ps` is the platform-provided reduction;
    // some toolchains lower it to a series of `vextracti64x4` +
    // `vpadd` + `vhaddps` but the result is the same, and the
    // call site is once per output row so the cost is dwarfed
    // by the per-block FMAs.
    unsafe { _mm512_reduce_add_ps(acc) }
}

/// Sprint 41B — full vectorized AVX-512 dequant + FMA. Replaces
/// the Sprint 41A hybrid's per-element scalar nibble unpack with
/// SIMD bit-manipulation, lifting the dequant onto the same vector
/// pipe as the FMA chain so the inner loop runs at one Q6_K block
/// per ~10 AVX-512 instructions instead of ~256 scalar ops + 16
/// FMAs.
///
/// Layout per 32-element iteration (8 iters cover one block):
///
/// ```text
///   load 16 ql bytes (xmm)         → 32 nibbles
///     _mm_and_si128 + _mm_srli_epi16 → low / high nibble bytes
///     _mm_unpacklo/hi_epi8           → elements 0..15 / 16..31
///     _mm512_cvtepu8_epi32           → 16 × i32 each
///   load 8 qh bytes (u64 → xmm)    → 32 × 2-bit
///     _mm_shuffle_epi8 [0,0,0,0,…]   → byte-replicated 4×
///     _mm512_cvtepu8_epi32           → 16 × i32 each
///     _mm512_srlv_epi32 [0,2,4,6,…]  → per-lane variable shift
///     _mm512_and_epi32 0x03          → keep 2-bit payload
///   q6 = lo | (hi << 4); q_signed = q6 - 32
///   dq = (i32→f32 cvt) × sub_scale
///   acc = vfmadd231ps(dq, hidden_chunk, acc)
/// ```
///
/// Two 16-element FMAs per 32-element iter (sub-block scale changes
/// at +16 inside each iter — block.scales has one entry per 16
/// elements). Total: 16 FMAs per Q6_K block, dwarfed by the 16
/// loads + ~30 ops of dequant.
///
/// # Safety
/// Requires AVX-512F + BW + VL. Caller must runtime-detect (or
/// build with `-C target-cpu=znver4` / `-C target-feature=...`).
/// The lm_head dispatcher does the runtime check.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn dot_q6k_block_avx512_full(block: &Q6KBlock, hidden: &[f32]) -> f32 {
    debug_assert_eq!(hidden.len(), QK_K);
    let d = block.d.to_f32();
    let mut acc = unsafe { _mm512_setzero_ps() };

    let ql_ptr = block.ql.as_ptr();
    let qh_ptr = block.qh.as_ptr();
    let hidden_ptr = hidden.as_ptr();

    // Constants hoisted out of the loop. The compiler does this
    // automatically too, but writing it explicitly makes the ASM
    // easier to read with `objdump`.
    let mask_0f_8 = unsafe { _mm_set1_epi8(0x0F) };
    // Repeat each input byte 4× across a 16-byte register: bytes
    // 0..3 of the input fill lanes 0..15 of the output.
    let shuf_lo = unsafe {
        _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
    };
    let shuf_hi = unsafe {
        _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7)
    };
    // Per-lane right-shift counts to extract the four 2-bit
    // payloads from each replicated qh byte. Set in lane order
    // (lane 0 first); set_epi32 takes args in reverse.
    let shift_pattern = unsafe {
        _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0)
    };
    let mask_03_32 = unsafe { _mm512_set1_epi32(0x03) };
    let offset_32 = unsafe { _mm512_set1_epi32(32) };

    // 8 iters × 32 elements = 256 (= QK_K).
    for iter in 0..(QK_K / 32) {
        let base = iter * 32;

        // ── ql: 16 bytes → 32 × i32 (low 4 bits each) ──
        let ql_128 = unsafe {
            _mm_loadu_si128(ql_ptr.add(base / 2) as *const __m128i)
        };
        let lo_even = unsafe { _mm_and_si128(ql_128, mask_0f_8) };
        let lo_odd = unsafe {
            _mm_and_si128(_mm_srli_epi16(ql_128, 4), mask_0f_8)
        };
        let lo_lo = unsafe { _mm_unpacklo_epi8(lo_even, lo_odd) }; // elems  0..15
        let lo_hi = unsafe { _mm_unpackhi_epi8(lo_even, lo_odd) }; // elems 16..31
        let lo_32_a = unsafe { _mm512_cvtepu8_epi32(lo_lo) };
        let lo_32_b = unsafe { _mm512_cvtepu8_epi32(lo_hi) };

        // ── qh: 8 bytes → 32 × i32 (high 2 bits each) ──
        // Read 8 unaligned bytes as one u64; safe because
        // block.qh is a [u8; 64] field of a #[repr(C)] struct,
        // 64-byte aligned to its parent block boundary. Even
        // when called on the last block of an array, the
        // scales[16] + d: f16 fields after qh keep the trailing
        // 8 bytes inside the struct.
        let qh_64 = unsafe {
            std::ptr::read_unaligned(qh_ptr.add(base / 4) as *const u64)
        };
        let qh_128 = unsafe { _mm_set1_epi64x(qh_64 as i64) };
        let qh_lo_bytes = unsafe { _mm_shuffle_epi8(qh_128, shuf_lo) };
        let qh_hi_bytes = unsafe { _mm_shuffle_epi8(qh_128, shuf_hi) };
        let qh_32_a = unsafe { _mm512_cvtepu8_epi32(qh_lo_bytes) };
        let qh_32_b = unsafe { _mm512_cvtepu8_epi32(qh_hi_bytes) };
        let hi_shifted_a = unsafe { _mm512_srlv_epi32(qh_32_a, shift_pattern) };
        let hi_shifted_b = unsafe { _mm512_srlv_epi32(qh_32_b, shift_pattern) };
        let hi_2bit_a = unsafe { _mm512_and_epi32(hi_shifted_a, mask_03_32) };
        let hi_2bit_b = unsafe { _mm512_and_epi32(hi_shifted_b, mask_03_32) };

        // ── q6 = lo | (hi << 4); q_signed = q6 - 32 ──
        let q6_a = unsafe {
            _mm512_or_epi32(lo_32_a, _mm512_slli_epi32(hi_2bit_a, 4))
        };
        let q6_b = unsafe {
            _mm512_or_epi32(lo_32_b, _mm512_slli_epi32(hi_2bit_b, 4))
        };
        let qs_a = unsafe { _mm512_sub_epi32(q6_a, offset_32) };
        let qs_b = unsafe { _mm512_sub_epi32(q6_b, offset_32) };

        // ── i32 → fp32, multiply by per-sub-block scale ──
        let qf_a = unsafe { _mm512_cvtepi32_ps(qs_a) };
        let qf_b = unsafe { _mm512_cvtepi32_ps(qs_b) };
        // 32-element iter spans two 16-element sub-blocks; pick
        // the matching scales[] entries.
        let scale_a = unsafe {
            _mm512_set1_ps(d * block.scales[base / 16] as f32)
        };
        let scale_b = unsafe {
            _mm512_set1_ps(d * block.scales[(base + 16) / 16] as f32)
        };
        let dq_a = unsafe { _mm512_mul_ps(qf_a, scale_a) };
        let dq_b = unsafe { _mm512_mul_ps(qf_b, scale_b) };

        // ── FMA: acc += dq ⊙ hidden_chunk ──
        let h_a = unsafe { _mm512_loadu_ps(hidden_ptr.add(base)) };
        let h_b = unsafe { _mm512_loadu_ps(hidden_ptr.add(base + 16)) };
        acc = unsafe { _mm512_fmadd_ps(dq_a, h_a, acc) };
        acc = unsafe { _mm512_fmadd_ps(dq_b, h_b, acc) };
    }

    unsafe { _mm512_reduce_add_ps(acc) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::q6k::quantize_to_q6k;

    /// AVX-512 vs scalar dot product round-trip. Generates a
    /// known random Q6_K block and a random hidden vector, then
    /// checks that the AVX-512 result matches the scalar
    /// `Q6KBlock::dequantize` + manual FP32 dot to within FP32
    /// rounding.
    #[test]
    fn avx512_dot_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f not available — skipping AVX-512 test");
            return;
        }

        // Deterministic LCG (no RNG dep).
        let mut state: u64 = 0xBADD_CAFE_C0DE_F00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let src: Vec<f32> = (0..QK_K).map(|_| next() * 0.5).collect();
        let blocks = quantize_to_q6k(&src);
        assert_eq!(blocks.len(), 1);
        let hidden: Vec<f32> = (0..QK_K).map(|_| next() * 0.3).collect();

        // Scalar reference: dequantize + manual dot.
        let mut dequant = [0.0f32; QK_K];
        blocks[0].dequantize(&mut dequant);
        let scalar_dot: f32 = dequant
            .iter()
            .zip(hidden.iter())
            .map(|(a, b)| a * b)
            .sum();

        // AVX-512.
        let avx_dot = unsafe { dot_q6k_block_avx512(&blocks[0], &hidden) };

        // Allow a tiny tolerance for FP32 reduction-order
        // differences. Both paths see identical inputs (same
        // dequantized values) — only the summation tree differs.
        let abs_err = (scalar_dot - avx_dot).abs();
        let rel_err = abs_err / scalar_dot.abs().max(1e-9);
        assert!(
            abs_err < 1e-3 || rel_err < 1e-5,
            "AVX-512 dot {avx_dot} vs scalar {scalar_dot} (abs_err={abs_err}, rel_err={rel_err})"
        );
    }

    /// Full AVX-512 path (Sprint 41B) vs scalar reference. Same
    /// fixture as the hybrid test, plus an explicit cross-check
    /// that the full path matches the hybrid path bit-similarly
    /// (within FP32 reduction-order tolerance).
    #[test]
    fn avx512_full_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            eprintln!("AVX-512 F/BW/VL not all available — skipping full test");
            return;
        }

        let mut state: u64 = 0xFADE_5EED_2024_BABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let src: Vec<f32> = (0..QK_K).map(|_| next() * 0.5).collect();
        let blocks = quantize_to_q6k(&src);
        let hidden: Vec<f32> = (0..QK_K).map(|_| next() * 0.3).collect();

        // Scalar reference dot.
        let mut dequant = [0.0f32; QK_K];
        blocks[0].dequantize(&mut dequant);
        let scalar_dot: f32 = dequant
            .iter()
            .zip(hidden.iter())
            .map(|(a, b)| a * b)
            .sum();

        let full_dot = unsafe { dot_q6k_block_avx512_full(&blocks[0], &hidden) };
        let hybrid_dot = unsafe { dot_q6k_block_avx512(&blocks[0], &hidden) };

        // Full vs scalar: same tolerance as the hybrid test —
        // FP32 reduction-tree differences only.
        let abs_err = (scalar_dot - full_dot).abs();
        let rel_err = abs_err / scalar_dot.abs().max(1e-9);
        assert!(
            abs_err < 1e-3 || rel_err < 1e-5,
            "full AVX-512 dot {full_dot} vs scalar {scalar_dot} \
             (abs_err={abs_err}, rel_err={rel_err})"
        );

        // Full vs hybrid: should be within the same tolerance —
        // both are AVX-512 paths, only the dequant is vectorized
        // differently. Cross-checks that the SIMD bit-manipulation
        // produces the same dequantized values as the scalar
        // dequant inside the hybrid.
        let cross_err = (full_dot - hybrid_dot).abs();
        let cross_rel = cross_err / hybrid_dot.abs().max(1e-9);
        assert!(
            cross_err < 1e-3 || cross_rel < 1e-5,
            "full AVX-512 vs hybrid: {full_dot} vs {hybrid_dot} \
             (abs_err={cross_err}, rel_err={cross_rel})"
        );
    }
}
