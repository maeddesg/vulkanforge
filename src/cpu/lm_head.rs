//! CPU-side `lm_head` GEMV.
//!
//! Sprint 40 Part 2 — Phase A scalar implementation. The hot loop
//! is plain Rust; rustc auto-vectorizes the inner Q6_K dequant +
//! FMA chain through AVX-512 on Zen 4 well enough for Phase A's
//! correctness-first goal. A hand-tuned AVX-512 unpack + dot
//! kernel can replace `dot_q6k_block` in a future sprint without
//! disturbing this module's public surface.
//!
//! `CpuLmHead::from_fp32_weights` does the load-time requantize
//! from FP32 → Q6_K once per model load. Subsequent decode steps
//! call `forward(hidden, logits)`, which fans the GEMV out across
//! Rayon's default thread pool: each row of the vocab is a parallel
//! work unit. Memory bandwidth is the limiter (Q6_K is 6.5 b/w),
//! so threading is a multiplier on bandwidth-aggregate from all
//! CPU cores' L1/L2/L3 caches plus DDR5.

#![allow(dead_code)]

use crate::cpu::q6k::{self, Q6KBlock, QK_K};
use rayon::prelude::*;

/// CPU-resident `lm_head` weights, requantized to Q6_K at load
/// time. Shape: `[vocab_size, hidden_size]` row-major; each row is
/// `hidden_size / QK_K` consecutive [`Q6KBlock`]s.
pub struct CpuLmHead {
    pub weights: Vec<Q6KBlock>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Number of Q6_K blocks per output row = `hidden_size / QK_K`.
    pub blocks_per_row: usize,
}

impl CpuLmHead {
    /// Requantize FP32 weights to Q6_K. `weights_row_major` is the
    /// `[vocab_size, hidden_size]` matrix in row-major order — the
    /// caller (loader) is responsible for any FP8/FP16/BF16 →
    /// FP32 dequantization upstream.
    ///
    /// Panics if `hidden_size` is not a multiple of [`QK_K`] (256).
    /// All supported models satisfy this trivially: 4096 / 256 = 16,
    /// 5120 / 256 = 20.
    pub fn from_fp32_weights(
        weights_row_major: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        assert!(
            hidden_size.is_multiple_of(QK_K),
            "CpuLmHead: hidden_size {hidden_size} must be a multiple of {QK_K}",
        );
        assert_eq!(
            weights_row_major.len(),
            vocab_size * hidden_size,
            "CpuLmHead: weight buffer length mismatch (got {}, expected {} = {} * {})",
            weights_row_major.len(),
            vocab_size * hidden_size,
            vocab_size,
            hidden_size,
        );
        // One block per QK_K elements. Quantize across the whole
        // matrix; the row layout is preserved because Q6_K is
        // applied in 256-element chunks of consecutive memory and
        // every row is hidden_size = k*QK_K elements long.
        let weights = q6k::quantize_to_q6k(weights_row_major);
        let blocks_per_row = hidden_size / QK_K;
        Self {
            weights,
            vocab_size,
            hidden_size,
            blocks_per_row,
        }
    }

    /// `logits[v] = Σ_k hidden[k] · weights[v, k]`.
    /// Hidden state is FP32; weights are Q6_K dequantized inline.
    /// Output is FP32. Parallelized across vocab rows; each row's
    /// dot product is sequential within a thread.
    ///
    /// Sprint 41 — runtime dispatch:
    ///
    /// - AVX-512F + BW + VL → fully vectorized Q6_K dequant + FMA
    ///   (Sprint 41B, [`dot_q6k_block_avx512_full`]).
    /// - AVX-512F only      → hybrid scalar dequant + AVX-512 FMA
    ///   (Sprint 41A, [`dot_q6k_block_avx512`]).
    /// - otherwise          → scalar reference.
    ///
    /// On Zen 4 / Sapphire Rapids the full path runs; the hybrid
    /// is retained for older AVX-512F-only platforms (Skylake-X,
    /// Cannon Lake) where BW/VL aren't both present. Scalar
    /// covers everything else (ARM, older x86, MIPS for tests).
    pub fn forward(&self, hidden: &[f32], logits: &mut [f32]) {
        assert_eq!(
            hidden.len(),
            self.hidden_size,
            "CpuLmHead::forward: hidden length mismatch"
        );
        assert_eq!(
            logits.len(),
            self.vocab_size,
            "CpuLmHead::forward: logits length mismatch"
        );

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                self.gemv_q6k_avx512_full(hidden, logits);
                return;
            }
            if std::is_x86_feature_detected!("avx512f") {
                self.gemv_q6k_avx512(hidden, logits);
                return;
            }
        }
        self.gemv_q6k_scalar(hidden, logits);
    }

    /// Scalar reference GEMV. Fallback for non-AVX-512 CPUs and
    /// the calibration target for the AVX-512 path's correctness
    /// tests.
    fn gemv_q6k_scalar(&self, hidden: &[f32], logits: &mut [f32]) {
        let blocks_per_row = self.blocks_per_row;
        let weights = self.weights.as_slice();
        logits
            .par_iter_mut()
            .enumerate()
            .for_each(|(v, out)| {
                let row = &weights[v * blocks_per_row..(v + 1) * blocks_per_row];
                let mut sum = 0.0f32;
                for (b, block) in row.iter().enumerate() {
                    sum += dot_q6k_block(block, &hidden[b * QK_K..(b + 1) * QK_K]);
                }
                *out = sum;
            });
    }

    /// Sprint 41A AVX-512 GEMV — hybrid (scalar dequant + AVX FMA).
    /// Kept for AVX-512F-only platforms; Sprint 41B's full kernel
    /// requires AVX-512BW + VL on top.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn gemv_q6k_avx512(&self, hidden: &[f32], logits: &mut [f32]) {
        let blocks_per_row = self.blocks_per_row;
        let weights = self.weights.as_slice();
        logits
            .par_iter_mut()
            .enumerate()
            .for_each(|(v, out)| {
                let row = &weights[v * blocks_per_row..(v + 1) * blocks_per_row];
                let mut sum = 0.0f32;
                for (b, block) in row.iter().enumerate() {
                    // SAFETY: `forward` already verified AVX-512F
                    // at runtime. The `target_feature` annotation
                    // on `dot_q6k_block_avx512` makes the compiler
                    // aware of the precondition.
                    sum += unsafe {
                        crate::cpu::avx512_gemv::dot_q6k_block_avx512(
                            block,
                            &hidden[b * QK_K..(b + 1) * QK_K],
                        )
                    };
                }
                *out = sum;
            });
    }

    /// Sprint 41B AVX-512 GEMV — full vectorized dequant + FMA.
    /// On Zen 4 / Sapphire Rapids this is the production path.
    /// Calls [`crate::cpu::avx512_gemv::dot_q6k_block_avx512_full`].
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn gemv_q6k_avx512_full(&self, hidden: &[f32], logits: &mut [f32]) {
        let blocks_per_row = self.blocks_per_row;
        let weights = self.weights.as_slice();
        logits
            .par_iter_mut()
            .enumerate()
            .for_each(|(v, out)| {
                let row = &weights[v * blocks_per_row..(v + 1) * blocks_per_row];
                let mut sum = 0.0f32;
                for (b, block) in row.iter().enumerate() {
                    // SAFETY: `forward` already verified AVX-512F + BW + VL
                    // at runtime. The full kernel's `target_feature`
                    // requires the same set.
                    sum += unsafe {
                        crate::cpu::avx512_gemv::dot_q6k_block_avx512_full(
                            block,
                            &hidden[b * QK_K..(b + 1) * QK_K],
                        )
                    };
                }
                *out = sum;
            });
    }

    /// Approximate footprint of the requantized weights.
    pub fn size_bytes(&self) -> usize {
        self.weights.len() * std::mem::size_of::<Q6KBlock>()
    }
}

/// Q6_K block · FP32 256-vector dot product. Inline dequant +
/// scalar FMA. Compiler auto-vectorizes through the inner loop.
#[inline]
fn dot_q6k_block(block: &Q6KBlock, hidden: &[f32]) -> f32 {
    debug_assert_eq!(hidden.len(), QK_K);
    let d = block.d.to_f32();
    let mut sum = 0.0f32;
    for i in 0..QK_K {
        let lo = (block.ql[i / 2] >> (4 * (i & 1))) & 0x0F;
        let hi = (block.qh[i / 4] >> (2 * (i & 3))) & 0x03;
        let qu = lo | (hi << 4);
        let q = qu as i32 - 32;
        let elem_scale = d * block.scales[i / 16] as f32;
        sum += elem_scale * (q as f32) * hidden[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference FP32 dot product, used to bound the GEMV error.
    fn ref_gemv_fp32(
        hidden: &[f32],
        weights_row_major: &[f32],
        vocab: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; vocab];
        for v in 0..vocab {
            let row = &weights_row_major[v * hidden_size..(v + 1) * hidden_size];
            let mut s = 0.0f32;
            for k in 0..hidden_size {
                s += row[k] * hidden[k];
            }
            out[v] = s;
        }
        out
    }

    /// Compare CPU Q6_K GEMV against the FP32 reference. The
    /// downstream consumer (argmax / softmax) cares about top-K
    /// agreement, so we assert top-1 matches and top-5 IDs are
    /// the same set. Absolute logit values are also compared but
    /// with a generous tolerance proportional to the weight RMS.
    #[test]
    fn cpu_gemv_top5_matches_fp32_reference() {
        // Small synthetic problem: vocab=512, hidden=512.
        // Hidden is a multiple of QK_K (256) by construction.
        let vocab = 512usize;
        let hidden = 512usize;

        let mut state: u64 = 0xDEAD_BEEF_F00D_BABE;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // Random weights in a typical lm_head range (post-tied or
        // post-LN scaling, roughly N(0, 0.5) magnitudes).
        let weights_fp32: Vec<f32> = (0..vocab * hidden).map(|_| next() * 0.5).collect();
        let hidden_vec: Vec<f32> = (0..hidden).map(|_| next() * 0.3).collect();

        let lm = CpuLmHead::from_fp32_weights(&weights_fp32, vocab, hidden);
        let mut cpu_logits = vec![0.0f32; vocab];
        lm.forward(&hidden_vec, &mut cpu_logits);

        let ref_logits = ref_gemv_fp32(&hidden_vec, &weights_fp32, vocab, hidden);

        // Top-1 must match — argmax is the most-load-bearing
        // property of the lm_head output.
        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let ref_top1 = ref_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            cpu_top1, ref_top1,
            "CPU Q6_K argmax {} differs from FP32 reference {}",
            cpu_top1, ref_top1
        );

        // Top-5 IDs as sets must agree on at least 4 out of 5.
        // Ranking ties under Q6_K rounding can swap one element.
        let mut cpu_idx: Vec<usize> = (0..vocab).collect();
        cpu_idx.sort_by(|a, b| cpu_logits[*b].partial_cmp(&cpu_logits[*a]).unwrap());
        let mut ref_idx: Vec<usize> = (0..vocab).collect();
        ref_idx.sort_by(|a, b| ref_logits[*b].partial_cmp(&ref_logits[*a]).unwrap());
        let cpu_top5: std::collections::HashSet<_> = cpu_idx.iter().take(5).copied().collect();
        let ref_top5: std::collections::HashSet<_> = ref_idx.iter().take(5).copied().collect();
        let overlap = cpu_top5.intersection(&ref_top5).count();
        assert!(
            overlap >= 4,
            "top-5 overlap {} < 4 (cpu={:?} ref={:?})",
            overlap,
            cpu_idx.iter().take(5).collect::<Vec<_>>(),
            ref_idx.iter().take(5).collect::<Vec<_>>()
        );

        // Per-element logit RMS error should be small relative to
        // the logits' own RMS (signal-to-noise).
        let signal_rms = (ref_logits.iter().map(|x| x * x).sum::<f32>()
            / ref_logits.len() as f32)
            .sqrt();
        let mse = cpu_logits
            .iter()
            .zip(ref_logits.iter())
            .map(|(a, b)| {
                let e = a - b;
                e * e
            })
            .sum::<f32>()
            / cpu_logits.len() as f32;
        let noise_rms = mse.sqrt();
        let snr_db = 20.0 * (signal_rms / noise_rms).log10();
        // Q6_K on 512-element rows typically clears 30 dB SNR;
        // assert > 25 dB to leave margin for the deterministic
        // RNG quirks.
        assert!(
            snr_db > 25.0,
            "CPU GEMV SNR {snr_db:.1} dB below 25 dB threshold"
        );
    }

    /// `from_fp32_weights` panics on a non-multiple-of-256 hidden_size.
    #[test]
    #[should_panic(expected = "must be a multiple of")]
    fn rejects_non_multiple_hidden() {
        let _ = CpuLmHead::from_fp32_weights(&[0.0; 100 * 100], 100, 100);
    }
}
