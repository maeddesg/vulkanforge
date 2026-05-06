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
//! Sprint 41B (future): vectorize the nibble unpack itself —
//! `vpunpcklbw` for the low nibble, `vpermb` / `vpsrlvd` for the
//! high 2-bit, then `_mm512_cvtepi32_ps` for the int→float cast.
//! That would land us closer to the 76 GB/s DDR5 bandwidth limit
//! (≈ 175 tok/s theoretical for a 430 MB Q6_K weight matrix).
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
}
