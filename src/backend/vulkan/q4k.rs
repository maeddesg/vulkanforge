//! Q4_K block encoder, dequantizer, and CPU GEMV reference.
//!
//! Phase 1 / Step 1.3: builds the synthetic Q4_K weight blocks for
//! the smoke-test, and provides a CPU-side dequant + GEMV that the
//! GPU output (Step 1.4) is checked against.
//!
//! The block layout matches `block_q4_K` in `vk_shaders/types.glsl`
//! and `ggml-common.h` byte-for-byte (see step 1.0 §4):
//!   offset  0..2   : f16 d
//!   offset  2..4   : f16 dmin
//!   offset  4..16  : 12 bytes packed scales (8 sub-block scales +
//!                    8 sub-block mins, 6 bits each, packed per
//!                    `quantize_row_q4_K_ref` in ggml-quants.c)
//!   offset 16..144 : 128 bytes nibbles (2 per byte, sub-blocks 0..3
//!                    in low nibbles, sub-blocks 4..7 in high)
//!
//! Total: 144 bytes per block, holding 256 (= QUANT_K) weights.

use half::f16;

pub const QUANT_K: usize = 256;
pub const BLOCK_BYTES: usize = 144;

pub struct Q4KBlockSpec {
    pub d: f32,
    pub dmin: f32,
    /// 6-bit values, per sub-block (0..=63).
    pub sub_scales: [u8; 8],
    /// 6-bit values, per sub-block (0..=63).
    pub sub_mins: [u8; 8],
    /// 4-bit nibbles per element (0..=15).
    pub nibbles: [u8; QUANT_K],
}

/// Pack a `Q4KBlockSpec` into the 144-byte llama.cpp wire format.
pub fn encode_block(b: &Q4KBlockSpec) -> [u8; BLOCK_BYTES] {
    let mut out = [0u8; BLOCK_BYTES];

    out[0..2].copy_from_slice(&f16::from_f32(b.d).to_bits().to_le_bytes());
    out[2..4].copy_from_slice(&f16::from_f32(b.dmin).to_bits().to_le_bytes());

    // Scales/mins: per ggml-quants.c quantize_row_q4_K_ref.
    //   j in 0..4: byte (4+j) low6 = sub_scales[j], byte (8+j) low6 = sub_mins[j]
    //   j in 4..8: byte (12+j-4) = (sub_scales[j]&0xF) | ((sub_mins[j]&0xF)<<4)
    //              high 2 bits of sub_scales[j] OR-into byte (4+j-4) bits 6-7
    //              high 2 bits of sub_mins[j]   OR-into byte (4+j)   bits 6-7
    for j in 0..4 {
        out[4 + j] = b.sub_scales[j] & 0x3F;
        out[4 + j + 4] = b.sub_mins[j] & 0x3F;
    }
    for j in 4..8 {
        let ls = b.sub_scales[j];
        let lm = b.sub_mins[j];
        out[4 + j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
        out[4 + (j - 4)] |= ((ls >> 4) & 0x03) << 6;
        out[4 + j] |= ((lm >> 4) & 0x03) << 6;
    }

    // Nibble pair layout — see ggml-quants.c quantize_row_q4_K_ref:
    //   for (j = 0; j < QK_K; j += 64)
    //     for (l = 0; l < 32; ++l)
    //       qs[l] = L[j+l] | (L[j+l+32] << 4)
    //
    // i.e. pair p (0..3) holds sub-blocks 2p (low nibble) and 2p+1
    // (high nibble) in bytes 16+p*32 .. 16+p*32+31. NOT "low half =
    // sb0..3, high half = sb4..7" — that interleaving only looks right
    // for uniform-nibble blocks (smoke test passed under it because
    // every nibble was identical).
    for p in 0..4 {
        for l in 0..32 {
            let lo = b.nibbles[(2 * p) * 32 + l] & 0x0F;
            let hi = b.nibbles[(2 * p + 1) * 32 + l] & 0x0F;
            out[16 + p * 32 + l] = lo | (hi << 4);
        }
    }

    out
}

/// Inverse of `encode_block`. Returns the 256 dequantized f32 weights.
pub fn dequant_block(block: &[u8; BLOCK_BYTES]) -> [f32; QUANT_K] {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales_bytes = &block[4..16];
    let qs = &block[16..144];

    let mut sub_scales = [0u8; 8];
    let mut sub_mins = [0u8; 8];
    for j in 0..4 {
        sub_scales[j] = scales_bytes[j] & 0x3F;
        sub_mins[j] = scales_bytes[j + 4] & 0x3F;
    }
    for j in 4..8 {
        let lo_byte = scales_bytes[j + 4];
        let scale_lo4 = lo_byte & 0x0F;
        let min_lo4 = (lo_byte >> 4) & 0x0F;
        let scale_hi2 = (scales_bytes[j - 4] >> 6) & 0x03;
        let min_hi2 = (scales_bytes[j] >> 6) & 0x03;
        sub_scales[j] = scale_lo4 | (scale_hi2 << 4);
        sub_mins[j] = min_lo4 | (min_hi2 << 4);
    }

    let mut out = [0.0f32; QUANT_K];
    for sb in 0..8 {
        let scale = sub_scales[sb] as f32;
        let min = sub_mins[sb] as f32;
        // Pair p (= sb / 2) lives in bytes p*32..p*32+31. Even sub-blocks
        // are in the low nibble of each byte, odd sub-blocks in the high.
        let pair = sb / 2;
        let high_nibble = sb % 2 == 1;
        for k in 0..32 {
            let pos = sb * 32 + k;
            let byte = qs[pair * 32 + k];
            let nibble = if high_nibble {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            };
            out[pos] = d * scale * (nibble as f32) - dmin * min;
        }
    }
    out
}

/// Reference GEMV: `output[r] = Σ_e dequant(W[r,*])[e] · input[e]`.
///
/// Layout follows what the shader expects: weights are `n_rows`
/// rows × `k`-elements-per-row, packed per-row as
/// `(k / QUANT_K) * BLOCK_BYTES` bytes; input is `k` f32; output is
/// `n_rows` f32.
pub fn cpu_gemv(weights: &[u8], n_rows: usize, k: usize, input: &[f32]) -> Vec<f32> {
    assert!(k % QUANT_K == 0, "k must be a multiple of QUANT_K");
    let blocks_per_row = k / QUANT_K;
    assert_eq!(weights.len(), n_rows * blocks_per_row * BLOCK_BYTES);
    assert_eq!(input.len(), k);

    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * BLOCK_BYTES;
            let block: &[u8; BLOCK_BYTES] = (&weights[off..off + BLOCK_BYTES])
                .try_into()
                .expect("block slice must be exactly BLOCK_BYTES");
            let dq = dequant_block(block);
            for e in 0..QUANT_K {
                out[r] += dq[e] * input[b * QUANT_K + e];
            }
        }
    }
    out
}

/// Tiny deterministic xorshift RNG. Used for reproducible random
/// Q4_K weights and inputs in step 1.5; not a serious PRNG but fine
/// for benchmark fixtures.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s as u32
    }
    /// Float in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / ((1u32 << 24) as f32)
    }
    /// Float in `[-range, range)`.
    fn next_signed(&mut self, range: f32) -> f32 {
        (self.next_unit() * 2.0 - 1.0) * range
    }
}

/// Random Q4_K weight matrix with `m` rows × `k` elements per row
/// (`k` must be a multiple of QUANT_K). Uses LLM-realistic small
/// `d`/`dmin` so the resulting outputs stay in a few-units range —
/// this keeps absolute-error thresholds meaningful when checked
/// against the CPU reference.
pub fn build_random_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % QUANT_K == 0, "k must be a multiple of QUANT_K");
    let blocks_per_row = k / QUANT_K;
    let total_blocks = m * blocks_per_row;

    let mut rng = XorShift64::new(seed);
    let mut bytes = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    for _ in 0..total_blocks {
        // d ∈ [0.001, 0.011], dmin ∈ [0.0005, 0.0055] — small,
        // similar in scale to real Q4_K weights.
        let d = 0.001 + rng.next_unit() * 0.01;
        let dmin = 0.0005 + rng.next_unit() * 0.005;
        let mut sub_scales = [0u8; 8];
        let mut sub_mins = [0u8; 8];
        let mut nibbles = [0u8; QUANT_K];
        for i in 0..8 {
            sub_scales[i] = (rng.next_u32() % 64) as u8;
            sub_mins[i] = (rng.next_u32() % 64) as u8;
        }
        for i in 0..QUANT_K {
            nibbles[i] = (rng.next_u32() % 16) as u8;
        }
        let block = encode_block(&Q4KBlockSpec {
            d,
            dmin,
            sub_scales,
            sub_mins,
            nibbles,
        });
        bytes.extend_from_slice(&block);
    }
    bytes
}

/// Random input vector in `[-range, range)`.
pub fn build_random_input(k: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    (0..k).map(|_| rng.next_signed(range)).collect()
}

/// Smoke-test weight matrix:
/// - Row 0: d=1.0, dmin=0.0, all sub-block scales=1, all mins=0,
///   all nibbles=1 → every dequantized weight = 1.0
/// - Row 1: as row 0 but all nibbles=2 → every weight = 2.0
///
/// Combined with the all-ones input vector this gives an analytical
/// expected output of `[256.0, 512.0]`, which the dispatch in step
/// 1.4 has to reproduce exactly (modulo Q4_K's quantization noise,
/// which is zero here because the values fit exactly).
pub fn build_smoke_weights() -> Vec<u8> {
    let row0 = encode_block(&Q4KBlockSpec {
        d: 1.0,
        dmin: 0.0,
        sub_scales: [1; 8],
        sub_mins: [0; 8],
        nibbles: [1; QUANT_K],
    });
    let row1 = encode_block(&Q4KBlockSpec {
        d: 1.0,
        dmin: 0.0,
        sub_scales: [1; 8],
        sub_mins: [0; 8],
        nibbles: [2; QUANT_K],
    });
    let mut bytes = Vec::with_capacity(2 * BLOCK_BYTES);
    bytes.extend_from_slice(&row0);
    bytes.extend_from_slice(&row1);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_weights_dequant_to_known_values() {
        let bytes = build_smoke_weights();
        assert_eq!(bytes.len(), 2 * BLOCK_BYTES);

        let row0: &[u8; BLOCK_BYTES] = (&bytes[..BLOCK_BYTES]).try_into().unwrap();
        let row1: &[u8; BLOCK_BYTES] = (&bytes[BLOCK_BYTES..]).try_into().unwrap();
        let dq0 = dequant_block(row0);
        let dq1 = dequant_block(row1);
        assert!(dq0.iter().all(|&w| (w - 1.0).abs() < 1e-6));
        assert!(dq1.iter().all(|&w| (w - 2.0).abs() < 1e-6));
    }

    #[test]
    fn dequant_recovers_per_subblock_distinct_nibbles() {
        // Distinct constant nibble per sub-block — a pair-layout bug
        // would shuffle them and this test would fail.
        let mut nibbles = [0u8; QUANT_K];
        for sb in 0..8 {
            for k in 0..32 {
                nibbles[sb * 32 + k] = (sb as u8) + 1;
            }
        }
        let block = encode_block(&Q4KBlockSpec {
            d: 1.0,
            dmin: 0.0,
            sub_scales: [1; 8],
            sub_mins: [0; 8],
            nibbles,
        });
        let dq = dequant_block(&block);
        for sb in 0..8 {
            let expected = (sb as f32) + 1.0;
            for k in 0..32 {
                let pos = sb * 32 + k;
                assert!(
                    (dq[pos] - expected).abs() < 1e-6,
                    "sb {sb} pos {pos}: got {got}, expected {expected}",
                    got = dq[pos]
                );
            }
        }
    }

    #[test]
    fn dequant_recovers_per_subblock_distinct_scales() {
        // Distinct scale per sub-block, uniform nibble = 1 → output
        // weight equals scale per sub-block.
        let mut sub_scales = [0u8; 8];
        for sb in 0..8 {
            sub_scales[sb] = (sb as u8) + 1;
        }
        let block = encode_block(&Q4KBlockSpec {
            d: 1.0,
            dmin: 0.0,
            sub_scales,
            sub_mins: [0; 8],
            nibbles: [1; QUANT_K],
        });
        let dq = dequant_block(&block);
        for sb in 0..8 {
            let expected = (sb as f32) + 1.0;
            for k in 0..32 {
                let pos = sb * 32 + k;
                assert!(
                    (dq[pos] - expected).abs() < 1e-6,
                    "sb {sb} pos {pos}: got {got}, expected {expected}",
                    got = dq[pos]
                );
            }
        }
    }

    #[test]
    fn smoke_gemv_matches_analytical() {
        let weights = build_smoke_weights();
        let input = vec![1.0f32; QUANT_K];
        let out = cpu_gemv(&weights, 2, QUANT_K, &input);
        assert!((out[0] - 256.0).abs() < 1e-3, "row 0 = {}", out[0]);
        assert!((out[1] - 512.0).abs() < 1e-3, "row 1 = {}", out[1]);
    }
}
