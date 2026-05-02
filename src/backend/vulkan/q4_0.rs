//! Q4_0 block layout + CPU dequant.
//!
//! Sprint 17D — Q4_0 is the simplest 4-bit quant: a single fp16
//! scale and 32 unsigned 4-bit quants per 18-byte block. Used by
//! Qwen2.5 GGUFs (0.5B / 7B / 14B). All weights in a Q4_0 GGUF are
//! Q4_0 (no mixed-quant recipe), unlike Q3_K_M / Q5_K_M.
//!
//! Block layout (18 bytes / 32 weights, 4 bits/weight):
//! ```text
//!   fp16   d        // scale
//!   uint8  qs[16]   // 32 quants × 4 bits — packed half/half:
//!                   //   qs[j] low nibble  → output[j]      (j ∈ 0..16)
//!                   //   qs[j] high nibble → output[j + 16] (j ∈ 0..16)
//! ```
//!
//! Dequant: `output[i] = d * (q[i] - 8)` where `q[i]` is the
//! 4-bit unsigned quant for position `i` (range 0..15). Subtracting
//! 8 maps it to signed (-8..7).

use half::f16;

pub const QUANT_K: usize = 32;
pub const BLOCK_BYTES: usize = 18;

/// Dequantize one 18-byte Q4_0 block to 32 f32 weights. Mirrors
/// `dequantize_row_q4_0` in `ggml/src/ggml-quants.c` (single-block
/// form).
pub fn dequant_block(block: &[u8; BLOCK_BYTES]) -> [f32; QUANT_K] {
    // 0..2:   fp16 d
    // 2..18:  16 bytes qs (half/half packing, see module doc)
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..18];

    let mut out = [0.0f32; QUANT_K];
    for j in 0..16 {
        let lo = (qs[j] & 0x0F) as i32 - 8;
        let hi = ((qs[j] >> 4) & 0x0F) as i32 - 8;
        out[j] = lo as f32 * d;
        out[j + 16] = hi as f32 * d;
    }
    out
}
