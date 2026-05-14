//! Q6_K block layout + CPU dequant.
//!
//! Sprint 52E — mirrors `q5k.rs` but for the 6.5-bits-per-weight
//! K-quant. Gemma-4-E4B (and likely 26B-A4B) emits
//! `per_layer_token_embd.weight` as Q6_K instead of E2B's Q5_K — the
//! imatrix quantiser keeps the larger PLE tables at higher precision.
//! The GGUF PLE load path (`build_gemma4_ple_from_gguf`) needs this
//! dequant to produce the BF16-byte buffer
//! `PleData::build_per_layer_inputs` reads at runtime.
//!
//! Block layout (210 bytes / 256 weights, 6.5 bits/weight):
//! ```text
//!   uint8  ql[128]          // low 4 bits per quant, 2 quants per byte
//!   uint8  qh[64]           // high 2 bits per quant, 4 quants per byte
//!   int8   scales[16]       // 16 sub-block scales (8-bit signed)
//!   fp16   d                // super-block scale
//! ```
//!
//! Each weight is reconstructed as
//! `output = d * scale * (q - 32)`
//! where `q = low4 + (high2 << 4)` ∈ `[0, 63]` (so `q - 32 ∈ [-32, 31]`)
//! and `scale` is the int8 sub-block scale selected by element index.
//!
//! Reference: `dequantize_row_q6_K` in `ggml/src/ggml-quants.c`
//! (single-block form).

use half::f16;

pub const QUANT_K: usize = 256;
pub const BLOCK_BYTES: usize = 210;

/// Dequantize one 210-byte Q6_K block to 256 f32 weights. Mirrors
/// `dequantize_row_q6_K` in `ggml-quants.c` exactly: each 128-element
/// half consumes 8 sub-block scales (`scales[0..8]` or `scales[8..16]`),
/// where the 4 reconstruction positions within a half draw from
/// `sc[is]`, `sc[is + 2]`, `sc[is + 4]`, `sc[is + 6]` with
/// `is = l / 16`. Sprint 52X fixed an earlier bug where only the first
/// 4 scales per half were used (sub-blocks were 32-wide instead of 16),
/// which produced sign flips at ~17 indices per 2816-d Q6_K row.
pub fn dequant_block(block: &[u8; BLOCK_BYTES]) -> [f32; QUANT_K] {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales_raw = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    let mut scales = [0i8; 16];
    for (i, &b) in scales_raw.iter().enumerate() {
        scales[i] = b as i8;
    }

    let mut out = [0.0f32; QUANT_K];

    for half in 0..2 {
        let ql_base = half * 64;
        let qh_base = half * 32;
        let sc_base = half * 8;
        let out_base = half * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1_low = (ql[ql_base + l] & 0x0F) as i32;
            let q2_low = (ql[ql_base + l + 32] & 0x0F) as i32;
            let q3_low = (ql[ql_base + l] >> 4) as i32;
            let q4_low = (ql[ql_base + l + 32] >> 4) as i32;
            let qh_byte = qh[qh_base + l];
            let q1_hi = ((qh_byte >> 0) & 0x03) as i32;
            let q2_hi = ((qh_byte >> 2) & 0x03) as i32;
            let q3_hi = ((qh_byte >> 4) & 0x03) as i32;
            let q4_hi = ((qh_byte >> 6) & 0x03) as i32;
            let q1 = q1_low | (q1_hi << 4);
            let q2 = q2_low | (q2_hi << 4);
            let q3 = q3_low | (q3_hi << 4);
            let q4 = q4_low | (q4_hi << 4);
            let s1 = scales[sc_base + is] as i32;
            let s2 = scales[sc_base + is + 2] as i32;
            let s3 = scales[sc_base + is + 4] as i32;
            let s4 = scales[sc_base + is + 6] as i32;
            out[out_base + l] = d * (s1 * (q1 - 32)) as f32;
            out[out_base + l + 32] = d * (s2 * (q2 - 32)) as f32;
            out[out_base + l + 64] = d * (s3 * (q3 - 32)) as f32;
            out[out_base + l + 96] = d * (s4 * (q4 - 32)) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: a zeroed Q6_K block (d=0) should dequantise to all-zero
    /// FP32. Guards the layout offsets (a misindexed read of `d`
    /// would pull non-zero bytes and the test would fail).
    #[test]
    fn dequant_zero_block_yields_zero() {
        let block = [0u8; BLOCK_BYTES];
        let out = dequant_block(&block);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "elem {i}");
        }
    }

    /// Synthesise a block with d=1, scales=[1; 16], and a known low-bit
    /// pattern, then verify a handful of dequantised values match the
    /// expected `(q - 32)` reconstruction.
    #[test]
    fn dequant_known_block_recovers_q_minus_32() {
        let mut block = [0u8; BLOCK_BYTES];
        // ql: each byte holds two 4-bit values. Set ql[0] = 0xBA →
        // low pair = (10, 11). After `q = low4 | (hi<<4)` with hi=0,
        // q1=10, q3=11. Element 0 → d*scale*(10-32) = 1*1*(-22) = -22.
        block[0] = 0xBA;
        // scales[0] = 1
        block[192] = 1;
        // scales[1..16] = 1 too (defensive — other groups touched
        // by elements out of focus, but we only assert on element 0).
        for i in 193..208 {
            block[i] = 1;
        }
        // d (fp16) = 1.0  → bits = 0x3C00
        block[208] = 0x00;
        block[209] = 0x3C;
        let out = dequant_block(&block);
        assert_eq!(out[0], -22.0, "elem 0: 0xBA low-nibble = 10 → 10-32 = -22");
        // ql[0] = 0xBA, low-nibble = 0xA = 10. So elem 0 = -22.
        // ql[0] >> 4 = 0xB = 11. Elem 64 = (11-32)*scale[4] = -21*1.
        assert_eq!(out[64], -21.0, "elem 64: 0xBA high-nibble = 11 → 11-32 = -21");
    }

    /// Sprint 52X regression: distinguish the correct scale-indexing
    /// from the pre-52X bug where `out[l + 32/64/96]` read
    /// `scales[sc_base + 1/2/3]` instead of `scales[sc_base + is + 2/4/6]`.
    /// Setting all ql/qh to zero (so every `q = 0`, `q - 32 = -32`) and
    /// using a distinct value per scale lets us observe exactly which
    /// scale index was applied at each output position.
    #[test]
    fn dequant_uses_full_8_scales_per_half() {
        let mut block = [0u8; BLOCK_BYTES];
        // ql + qh stay zero → q = 0 at every position → q - 32 = -32.
        // scales[0..8] = 1, 2, ... 8 ; scales[8..16] = 10, 20, ... 80.
        for i in 0..8 {
            block[192 + i] = (i + 1) as u8;
        }
        for i in 0..8 {
            block[192 + 8 + i] = ((i + 1) * 10) as u8;
        }
        block[208] = 0x00;
        block[209] = 0x3C; // d = fp16(1.0)
        let out = dequant_block(&block);

        // First half, l=0 → is=0 → scales[0], [2], [4], [6].
        assert_eq!(out[0], -32.0, "l=0 → sc[0]=1 → 1*-32");
        assert_eq!(out[32], -96.0, "l=0 → sc[2]=3 → 3*-32 (BUG would give sc[1]=2 → -64)");
        assert_eq!(out[64], -160.0, "l=0 → sc[4]=5 → 5*-32 (BUG would give sc[2]=3 → -96)");
        assert_eq!(out[96], -224.0, "l=0 → sc[6]=7 → 7*-32 (BUG would give sc[3]=4 → -128)");

        // First half, l=16 → is=1 → scales[1], [3], [5], [7].
        assert_eq!(out[16], -64.0, "l=16 → sc[1]=2 (BUG would give sc[0]=1 → -32)");
        assert_eq!(out[48], -128.0, "l=16 → sc[3]=4");
        assert_eq!(out[80], -192.0, "l=16 → sc[5]=6");
        assert_eq!(out[112], -256.0, "l=16 → sc[7]=8");

        // Second half, l=0 → is=0, sc_base=8 → scales[8], [10], [12], [14].
        assert_eq!(out[128], -320.0, "second half l=0 → sc[8]=10");
        assert_eq!(out[160], -960.0, "second half l=0 → sc[10]=30");
        assert_eq!(out[192], -1600.0, "second half l=0 → sc[12]=50");
        assert_eq!(out[224], -2240.0, "second half l=0 → sc[14]=70");

        // Second half, l=16 → is=1, sc_base=8 → scales[9], [11], [13], [15].
        assert_eq!(out[144], -640.0, "second half l=16 → sc[9]=20");
        assert_eq!(out[176], -1280.0, "second half l=16 → sc[11]=40");
        assert_eq!(out[208], -1920.0, "second half l=16 → sc[13]=60");
        assert_eq!(out[240], -2560.0, "second half l=16 → sc[15]=80");
    }
}
