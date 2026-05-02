//! Q5_K block layout + CPU dequant.
//!
//! Sprint 17C — Q5_K is the 5.5-bit-per-weight K-quant. Q3_K_M
//! GGUFs use Q5_K for `attn_v.weight` and `ffn_down.weight`
//! (the bug that broke Sprint 17B); Q5_K_M / Q5_K_S use it for
//! the bulk of the model. CPU dequant is needed when Q5_K appears
//! as `token_embd.weight` (defensive — uncommon in practice but
//! cheap to implement).
//!
//! Block layout (176 bytes / 256 weights, 5.5 bits/weight):
//! ```text
//!   fp16   d                // super-block scale
//!   fp16   dmin             // super-block min
//!   uint8  scales[12]       // 8 sub-block scales + 8 sub-block mins, 6 bits each
//!   uint8  qh[32]           // high bit (5th bit) of each quant — 256 bits total
//!   uint8  qs[128]          // low 4 bits per quant, 2 quants per byte
//! ```
//!
//! Each weight is reconstructed as
//! `output = d * scale * (low4 + (high_bit ? 16 : 0)) - dmin * min`
//! where `low4 = qs[l] & 0xF` (or `>> 4` for the high nibble) and
//! `high_bit = (qh[l] >> u_bit) & 1`.

use half::f16;

pub const QUANT_K: usize = 256;
pub const BLOCK_BYTES: usize = 176;

/// Decode the (scale, min) pair for sub-block index `j` ∈ 0..8 from
/// the 12-byte packed `scales` array. Mirrors `get_scale_min_k4`
/// in `ggml-quants.c` (used by both Q4_K and Q5_K).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one 176-byte Q5_K block to 256 f32 weights. Mirrors
/// `dequantize_row_q5_K` in `ggml/src/ggml-quants.c` (single-block
/// form).
pub fn dequant_block(block: &[u8; BLOCK_BYTES]) -> [f32; QUANT_K] {
    // Layout offsets:
    //   0..4    : f16 d, f16 dmin
    //   4..16   : 12 bytes scales (6-bit packed × 16 = 8 scales + 8 mins)
    //  16..48   : 32 bytes qh (one high bit per weight)
    //  48..176  : 128 bytes qs (low 4 bits, 2 quants per byte)
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: [u8; 12] = block[4..16].try_into().unwrap();
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut out = [0.0f32; QUANT_K];
    let mut y = 0usize;
    let mut ql_off = 0usize;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    // 4 iterations × 64 weights = 256 weights total. Each iteration
    // pulls two (scale, min) pairs from `scales[is, is+1]` and uses
    // two qh-bit masks (u1, u2).
    for _ in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, &scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;

        for l in 0..32 {
            let q4 = (qs[ql_off + l] & 0x0F) as i32;
            let qh_bit = if (qh[l] & u1) != 0 { 16i32 } else { 0i32 };
            out[y] = d1 * (q4 + qh_bit) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            let q4 = (qs[ql_off + l] >> 4) as i32;
            let qh_bit = if (qh[l] & u2) != 0 { 16i32 } else { 0i32 };
            out[y] = d2 * (q4 + qh_bit) as f32 - m2;
            y += 1;
        }
        ql_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    out
}
