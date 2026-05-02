//! Q3_K block layout + CPU dequant.
//!
//! Sprint 17B — Q3_K_M GGUFs ship `token_embd.weight` as Q3_K, so
//! `decode::embedding_row` (which dequantizes one embedding row per
//! generated token, on the CPU) needs a Q3_K reference path. The GPU
//! GEMV / GEMM kernels for Q3_K are byte-identical to llama.cpp; this
//! module is the matching CPU implementation, ported from
//! ggml-quants.c `dequantize_row_q3_K`.
//!
//! Block layout (110 bytes / 256 weights, 3.4375 bits/weight):
//! ```text
//!   uint8  hmask[32]     // high bit of each 3-bit quant (256 bits total)
//!   uint8  qs[64]        // low 2 bits per quant (256 × 2 / 8 = 64 B)
//!   uint8  scales[12]    // 16 sub-block scales packed at 6 bits each
//!   fp16   d             // super-block scale
//! ```
//!
//! Each 256-weight block is split into 16 sub-blocks of 16 weights.
//! Each weight is reconstructed as
//! `output = d * (scale - 32) * (((qs >> shift) & 3) - ((hmask & m) ? 0 : 4))`.

use half::f16;

pub const QUANT_K: usize = 256;
pub const BLOCK_BYTES: usize = 110;

/// Dequantize one 110-byte Q3_K block to 256 f32 weights. Mirrors
/// `dequantize_row_q3_K` in `ggml/src/ggml-quants.c` (single-block
/// form).
pub fn dequant_block(block: &[u8; BLOCK_BYTES]) -> [f32; QUANT_K] {
    // Layout offsets:
    //   0..32   hmask
    //  32..96   qs
    //  96..108  scales (12 bytes, packed 6-bit × 16)
    // 108..110  d (fp16)
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales_raw = &block[96..108];
    let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();

    // Unpack the 16 × 6-bit scales into 16 i8s. The packed format is
    // identical to llama.cpp's: bytes 0..3 are the low-4-bit scales
    // for sub-blocks 0..3, bytes 4..7 are the low-4-bit scales for
    // 4..7, byte 8..11 hold the high-2-bit fragments interleaved.
    let mut aux: [u32; 4] = [0; 4];
    aux[0] = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
    aux[1] = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
    aux[2] = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: [i8; 16] = bytemuck::cast(aux);

    // Mirror ggml-quants.c: `is`, `m`, and `q_off` all live across
    // both halves of the 256-weight block; only `shift` resets per
    // half. `m` walks bits 0..7 across the 8 scales-pair iterations
    // (4 per half).
    let mut out = [0.0f32; QUANT_K];
    let mut y = 0usize;
    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut m = 1u8;
    for _block_half in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let dl = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let q_lo = ((qs[q_off + l] >> shift) & 0x3) as i8;
                let h_bit = if (hmask[l] & m) != 0 { 0i8 } else { 4i8 };
                out[y] = dl * ((q_lo - h_bit) as f32);
                y += 1;
            }
            let dl2 = d_all * (scales[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let q_lo = ((qs[q_off + l + 16] >> shift) & 0x3) as i8;
                let h_bit = if (hmask[l + 16] & m) != 0 { 0i8 } else { 4i8 };
                out[y] = dl2 * ((q_lo - h_bit) as f32);
                y += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q_off += 32;
    }
    out
}
