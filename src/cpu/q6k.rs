//! Q6_K block codec — load-time quantizer + dequantizer.
//!
//! Q6_K is llama.cpp's 6-bit-per-weight super-block format,
//! 256 elements per block at ~210 B = 6.5625 bits/weight. Layout
//! matches the `block_q6_K` struct in upstream llama.cpp:
//!
//! ```text
//! struct block_q6_K {
//!     uint8_t  ql[QK_K/2];   // lower 4 bits, packed two per byte
//!     uint8_t  qh[QK_K/4];   // upper 2 bits, packed four per byte
//!     int8_t   scales[QK_K/16]; // 16 sub-block scales (one per 16 elements)
//!     ggml_fp16_t d;         // super-block scale
//! };
//! ```
//! with `QK_K = 256`. We mirror that field order so the on-disk
//! layout is interoperable with llama.cpp's GGUF loader, which is
//! useful when we eventually want to share `output.weight` Q6_K
//! between the GGUF path (already Q6_K on disk) and the FP8
//! requantize path.
//!
//! The quantization scheme follows llama.cpp's reference
//! `quantize_row_q6_K_reference` (ggml-quants.c):
//!
//! 1. Pick a super-block scale `d = max_abs(block) / 31`.
//! 2. Pick a sub-block scale per 16 elements: `scales[s] = round(max_abs(sub) / d)`,
//!    clamped to int8.
//! 3. Quantize each weight: `q = round(w / (d * scales[i/16]))`,
//!    clamped to [-32, 31], stored as unsigned [0, 63] in the
//!    packed `ql` + `qh` arrays.
//!
//! Dequantize is the trivial inverse. With Q6_K's 6-bit grid the
//! per-element rounding error is roughly `d * scales[i/16] / 64`,
//! which empirically lands at ~1e-3 for typical weight matrices —
//! more than enough for an `lm_head` whose downstream consumer is
//! an argmax / softmax.
//!
//! All routines here are scalar Rust. The CPU GEMV in
//! [`crate::cpu::lm_head`] consumes the dequantized values in an
//! inner loop that the compiler auto-vectorizes via AVX-512 on
//! Zen 4. A hand-tuned AVX-512 unpack + dot kernel can replace the
//! scalar dot product in a future sprint without touching this
//! file.

use half::f16;

/// Block size — fixed by the Q6_K format.
pub const QK_K: usize = 256;

/// One Q6_K super-block. `#[repr(C)]` pins the field order to the
/// llama.cpp on-disk layout. Size is 128 + 64 + 16 + 2 = 210 bytes;
/// the natural alignment is 2 (because of `d: f16`), so an array of
/// blocks is contiguous.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q6KBlock {
    /// Lower 4 bits of each quantized weight, packed two per byte.
    /// `ql[i / 2]` holds element `i`'s low nibble in bits
    /// `4 * (i & 1)..4 * (i & 1) + 4`.
    pub ql: [u8; QK_K / 2],
    /// Upper 2 bits of each quantized weight, packed four per byte.
    /// `qh[i / 4]` holds element `i`'s high pair in bits
    /// `2 * (i & 3)..2 * (i & 3) + 2`.
    pub qh: [u8; QK_K / 4],
    /// 16 sub-block scales (one per 16 elements). Signed so a
    /// sub-block dominated by negative weights gets the right sign
    /// out of the multiply.
    pub scales: [i8; QK_K / 16],
    /// Super-block scale, FP16. Multiplied by `scales[i/16]` gives
    /// the per-element scale.
    pub d: f16,
}

impl Q6KBlock {
    /// All-zero block. Useful for the round-trip tests.
    pub const ZERO: Self = Q6KBlock {
        ql: [0; QK_K / 2],
        qh: [0; QK_K / 4],
        scales: [0; QK_K / 16],
        d: f16::ZERO,
    };

    /// Dequantize this block into a 256-element FP32 buffer.
    /// Cost: 256 nibble unpacks + 256 multiplies. Used by tests
    /// and by the scalar reference GEMV path.
    pub fn dequantize(&self, out: &mut [f32; QK_K]) {
        let d = self.d.to_f32();
        for i in 0..QK_K {
            let lo = (self.ql[i / 2] >> (4 * (i & 1))) & 0x0F;
            let hi = (self.qh[i / 4] >> (2 * (i & 3))) & 0x03;
            let qu = lo | (hi << 4); // unsigned [0, 63]
            let q = qu as i32 - 32;  // signed   [-32, 31]
            let sub_scale = self.scales[i / 16] as f32;
            out[i] = d * sub_scale * q as f32;
        }
    }
}

/// Quantize an FP32 buffer of length `N * 256` into `N` Q6_K blocks.
/// Panics on a non-multiple-of-256 length — the caller (lm_head
/// loader) is responsible for padding the trailing partial block
/// if needed. For the supported models (Llama-8B, Qwen3-8B,
/// Qwen2.5-14B) `vocab × hidden` always divides 256 cleanly.
pub fn quantize_to_q6k(src: &[f32]) -> Vec<Q6KBlock> {
    assert!(
        src.len().is_multiple_of(QK_K),
        "quantize_to_q6k: input length {} must be a multiple of {}",
        src.len(),
        QK_K,
    );
    let n_blocks = src.len() / QK_K;
    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        blocks.push(quantize_block(&src[b * QK_K..(b + 1) * QK_K]));
    }
    blocks
}

/// Quantize one 256-element chunk to a Q6_K block.
/// Two-stage algorithm (matches llama.cpp `quantize_row_q6_K_reference`):
///
/// 1. Per-sub-block float scales `scales_f[sb] = sub_absmax / 31`.
///    Each is the per-element grid step that exactly hits the
///    sub-block extremum at `q = ±31`.
/// 2. Super-block scale `d = max(|scales_f|) / 127` quantizes the
///    16 sub-block scales into int8 with the full `[-127, 127]`
///    range used. Per-element scale is then `d * scales_int8[sb]`,
///    finer than `d` alone by up to 127× — that's where Q6_K's
///    accuracy comes from.
fn quantize_block(src: &[f32]) -> Q6KBlock {
    debug_assert_eq!(src.len(), QK_K);

    // Step 1 — per-sub-block float scales.
    let mut scales_f = [0.0f32; QK_K / 16];
    let mut max_scale_f = 0.0f32;
    for sb in 0..(QK_K / 16) {
        let sub = &src[sb * 16..(sb + 1) * 16];
        let sub_absmax = sub.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let s = sub_absmax / 31.0;
        scales_f[sb] = s;
        if s > max_scale_f {
            max_scale_f = s;
        }
    }
    if max_scale_f <= 0.0 {
        // Degenerate: a block of all zeros. Zero block dequantizes
        // to zeros regardless of the q payload.
        return Q6KBlock::ZERO;
    }

    // Step 2 — super-block scale. We use 127 (not 128) so the
    // round-to-nearest at the upper bound doesn't overflow int8.
    let d = max_scale_f / 127.0;
    let d_inv = 1.0 / d;

    // Compress the 16 float scales into int8 sub-block scales.
    let mut scales = [0i8; QK_K / 16];
    for sb in 0..(QK_K / 16) {
        let raw = (scales_f[sb] * d_inv).round();
        scales[sb] = raw.clamp(-127.0, 127.0) as i8;
    }

    // Step 3 — quantize each weight. Per-element scale = d * scales[sb].
    // Q6 signed range is [-32, 31] (six bits, signed). Clamp on the
    // upper end at 31 so round-to-nearest can't overflow.
    let mut ql = [0u8; QK_K / 2];
    let mut qh = [0u8; QK_K / 4];
    for i in 0..QK_K {
        let elem_scale = d * scales[i / 16] as f32;
        let q_signed = if elem_scale.abs() > 0.0 {
            (src[i] / elem_scale).round().clamp(-32.0, 31.0) as i32
        } else {
            0
        };
        let qu = (q_signed + 32) as u8; // unsigned [0, 63]

        // Pack low 4 bits.
        let byte_idx = i / 2;
        let nibble = i & 1;
        ql[byte_idx] |= (qu & 0x0F) << (4 * nibble);

        // Pack high 2 bits.
        let h_byte = i / 4;
        let h_shift = (i & 3) * 2;
        qh[h_byte] |= ((qu >> 4) & 0x03) << h_shift;
    }

    Q6KBlock {
        ql,
        qh,
        scales,
        d: f16::from_f32(d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a smooth block. Q6_K's per-element grid step is
    /// ≈ `sub_absmax / 31` (the Q6 signed grid covers each
    /// sub-block with 64 levels), so the max round-trip error is
    /// bounded by half of that. For weights in `[-1.28, 1.27]` the
    /// bound is roughly `0.04 / 2 ≈ 0.02`. We leave a small
    /// headroom for FP16 storage of `d`.
    #[test]
    fn round_trip_smooth_block() {
        let src: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let blocks = quantize_to_q6k(&src);
        assert_eq!(blocks.len(), 1);
        let mut out = [0.0f32; QK_K];
        blocks[0].dequantize(&mut out);
        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.025,
            "smooth-block max abs error {max_err} exceeded Q6_K bound"
        );
    }

    /// Round-trip a larger random buffer. With weights uniform on
    /// `[-5, 5]` the per-element grid is ≈ `5 / 31 ≈ 0.16`, so the
    /// max-abs bound is ~0.08 and RMS is roughly half that. Q6_K's
    /// downstream consumer is an argmax over `lm_head` logits; an
    /// 0.05-relative-precision per weight is well below what
    /// argmax cares about.
    #[test]
    fn round_trip_random_multi_block() {
        // Deterministic LCG so the test is reproducible without an
        // RNG dep.
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let src: Vec<f32> = (0..QK_K * 8).map(|_| next() * 5.0).collect();
        let blocks = quantize_to_q6k(&src);

        let mut out = vec![0.0f32; src.len()];
        for (b, block) in blocks.iter().enumerate() {
            let mut tmp = [0.0f32; QK_K];
            block.dequantize(&mut tmp);
            out[b * QK_K..(b + 1) * QK_K].copy_from_slice(&tmp);
        }

        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 5/31 / 2 ≈ 0.08 worst case; allow a bit more for FP16
        // rounding of `d` and the int8 scale compression.
        assert!(
            max_err < 0.15,
            "random multi-block max abs error {max_err} exceeded Q6_K bound"
        );

        let mse: f32 = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| {
                let e = a - b;
                e * e
            })
            .sum::<f32>()
            / src.len() as f32;
        let rms = mse.sqrt();
        assert!(
            rms < 0.05,
            "random multi-block RMS error {rms} exceeded Q6_K bound"
        );
    }

    /// All-zero block must dequantize cleanly (no NaN or div-by-zero).
    #[test]
    fn zero_block_round_trip() {
        let src = vec![0.0f32; QK_K];
        let blocks = quantize_to_q6k(&src);
        let mut out = [0.0f32; QK_K];
        blocks[0].dequantize(&mut out);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    /// Layout sanity. Q6_K blocks are stored contiguously in arrays;
    /// future AVX-512 kernels rely on `Q6KBlock` being plain old
    /// data with the C field order.
    #[test]
    fn block_size_is_210_bytes() {
        assert_eq!(std::mem::size_of::<Q6KBlock>(), 210);
    }
}
