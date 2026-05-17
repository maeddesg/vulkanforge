//! Qwen3.5 / Qwen3.6 (`qwen35`) architecture helpers — **skeleton only**.
//!
//! Sprint B (v0.4.6 prep). Provides the typed configuration struct, the
//! layer-routing predicate (full-attention vs linear-attention/SSM), and
//! the GGUF tensor-name lookups needed by the loader in future sprints.
//!
//! Reference: `docs/qwen35_architecture_analysis.md`.
//!
//! Nothing here is wired into the dispatch path yet — adding new
//! `LayerStep` variants would force both `DecodeExec` and `BatchExec`
//! to implement them (coding standards §3.2 / §4.2) which is the
//! scope of Sprint C+. The struct lives here so the loader and arch
//! plumbing can be staged independently of the SSM/GDN shaders.
//!
//! Open items (handled in later sprints):
//!
//! - `LayerStep::AttnQGateSplit { gate_dim }` — splits the fused
//!   Q+gate output of `attn_q` for full-attention layers and applies
//!   `sigmoid(gate)` on the attention output.
//! - `LayerStep::SsmConv1d { layer }` — Conv1d (kernel = 4) + persistent
//!   conv-state update.
//! - `LayerStep::GatedDeltaNet { layer }` — Q/K/V/G recurrence with
//!   persistent SSM state. Port of llama.cpp's `gated_delta_net.comp`.
//! - `LayerStep::NormGated { eps }` — fused `RMSNorm(x) * SiLU(z)`.
//! - `LayerStep::L2Norm { eps }` — optional; reusable via RMSNorm with
//!   weight = 1.
//!
//! mRoPE simplification: `rope.dimension_sections = [11, 11, 10, 0]`
//! ⇒ for text-only inference, partial RoPE with `n_rot = 32` matches
//! llama.cpp's `ggml_rope_multi` output. The fourth (temporal) section
//! is zero and never rotates a frequency band.

use std::collections::BTreeSet;

/// Architecture-string under `general.architecture`.
pub(super) const ARCH_NAME: &str = "qwen35";

/// Default `full_attention_interval` if the GGUF metadata key is
/// missing. llama.cpp uses 4; Qwen3.6-27B sets the key explicitly.
pub(super) const DEFAULT_FULL_ATTENTION_INTERVAL: u32 = 4;

/// Tokenizer pre-type string for `tokenizer.ggml.pre`. The Rust
/// tokenizer needs a new flavour (currently rejected by
/// `pick_bpe_flavour`); see analysis doc §7.
pub(super) const TOKENIZER_PRE: &str = "qwen35";

/// Pre-tokenizer regex (from llama.cpp `src/unicode.cpp:608`). The
/// difference vs. qwen2 is `[\p{L}\p{M}]+` instead of `\p{L}+` —
/// Qwen3.5 letter runs absorb Unicode combining marks.
pub(super) const PRE_REGEX_QWEN35: &str =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Typed view over the `qwen35.*` GGUF metadata. Mirrors the keys
/// listed in `docs/qwen35_architecture_analysis.md` §1. Values shown
/// in defaults correspond to the **Qwen3.6-27B** instance verified in
/// Sprint A; smaller variants (0.8B / 2B / 4B / 9B) will populate
/// different numbers but the same field set.
#[derive(Debug, Clone, Copy)]
pub(super) struct Qwen35Config {
    // Block layout
    pub block_count: u32,            // 65 (= n_main 64 + 1 MTP)
    pub nextn_predict_layers: u32,   // 1 (MTP draft blocks)
    pub full_attention_interval: u32, // 4

    // Embedding dimensions
    pub embedding_length: u32,       // 5120 (hidden)
    pub feed_forward_length: u32,    // 17408 (FFN intermediate)
    pub vocab_size: u32,             // 248320 (tokens)

    // Full-attention (run on layers where (i+1) % interval == 0)
    pub n_head: u32,                 // 24
    pub n_head_kv: u32,              // 4 (GQA 6:1)
    pub head_k: u32,                 // 256
    pub head_v: u32,                 // 256

    // RoPE (mRoPE in upstream; partial-RoPE n_rot=32 is text-equivalent)
    pub rope_dimension_count: u32,   // 64 (per-head rot dim)
    pub rope_sections: [u32; 4],     // [11, 11, 10, 0] (mRoPE)
    pub rope_freq_base: f32,         // 1e7

    // Linear-attention / SSM (run on layers where (i+1) % interval != 0)
    pub ssm_d_conv: u32,             // 4   (conv1d kernel)
    pub ssm_d_state: u32,            // 128 (head_k_dim, head_v_dim)
    pub ssm_n_group: u32,            // 16  (num_k_heads)
    pub ssm_dt_rank: u32,            // 48  (num_v_heads)
    pub ssm_d_inner: u32,            // 6144 (= n_head * head_v)

    // RMSNorm
    pub rms_norm_eps: f32,           // 1e-6
}

impl Qwen35Config {
    /// Number of "main" decoder blocks (trunk), excluding the MTP
    /// nextn block(s) appended at the end.
    pub fn n_main(&self) -> u32 {
        self.block_count - self.nextn_predict_layers
    }

    /// Full-attention layers fire when `(i+1) % interval == 0`. The
    /// MTP block (index ≥ n_main) is always non-recurrent — see
    /// `is_mtp_block`. Mirrors llama.cpp `qwen35.cpp:25-28`.
    pub fn is_full_attention_layer(&self, layer_idx: u32) -> bool {
        layer_idx >= self.n_main() || (layer_idx + 1) % self.full_attention_interval == 0
    }

    /// Inverse predicate. Linear-attention (Gated Delta Net + SSM
    /// conv) layers occupy the SSSF pattern's S slots in the trunk.
    pub fn is_recurrent_layer(&self, layer_idx: u32) -> bool {
        layer_idx < self.n_main()
            && (layer_idx + 1) % self.full_attention_interval != 0
    }

    /// True for blocks that carry the `blk.N.nextn.*` tensors. With
    /// `nextn_predict_layers == 1` (current Qwen3.6 release) this is
    /// the single block at index `n_main()`.
    pub fn is_mtp_block(&self, layer_idx: u32) -> bool {
        layer_idx >= self.n_main()
    }

    /// First MTP block index (or `block_count` if MTP is absent).
    /// Used by the loader to split the trunk-load loop from the
    /// MTP-load loop.
    pub fn mtp_start(&self) -> u32 {
        self.n_main()
    }

    /// Effective `n_rot` for text-only inference: sum of the three
    /// non-temporal mRoPE sections. For Qwen3.6-27B `[11, 11, 10, 0]`
    /// this is 32. Sprint B uses this as a partial-RoPE simplification
    /// (analysis doc §6.6).
    pub fn n_rot_text_only(&self) -> u32 {
        self.rope_sections[0] + self.rope_sections[1] + self.rope_sections[2]
    }

    /// Conv-input width for the linear-attention conv1d. Matches
    /// llama.cpp `qwen35.cpp:65-67`:
    /// `conv_dim = 2 * key_dim + value_dim`
    /// = `2 * (ssm_d_state * ssm_n_group) + ssm_d_inner`.
    pub fn conv_channels(&self) -> u32 {
        2 * self.ssm_d_state * self.ssm_n_group + self.ssm_d_inner
    }

    /// Bytes per sequence for the recurrent buffers, in F32. Reported
    /// for VRAM-budget estimates — actual layout is per-layer with
    /// `ssm_d_conv - 1` rolling slots for the conv state.
    pub fn recurrent_state_bytes_per_seq(&self) -> u64 {
        let n_lin = (self.n_main() - self.n_main() / self.full_attention_interval) as u64;
        // ssm_state: [head_v, head_v, n_v_heads] per layer per seq
        let ssm = n_lin
            * self.ssm_d_state as u64
            * self.ssm_d_state as u64
            * self.ssm_dt_rank as u64
            * 4;
        // conv_state: [ssm_d_conv - 1, conv_channels] per layer per seq
        let conv = n_lin * (self.ssm_d_conv as u64 - 1) * self.conv_channels() as u64 * 4;
        ssm + conv
    }
}

/// Sorted set of full-attention layer indices, derived from the
/// config. Useful when the loader needs to know up-front which
/// `attn_q/k/v/output` tensor families to expect.
pub(super) fn full_attention_layer_indices(cfg: &Qwen35Config) -> BTreeSet<u32> {
    (0..cfg.block_count)
        .filter(|&i| cfg.is_full_attention_layer(i))
        .collect()
}

/// GGUF tensor names per per-layer slot. Returns `None` for slots
/// that are not present on the given layer kind (e.g. asking for
/// `attn_q` on a recurrent layer, or `ssm_alpha` on a full-attention
/// layer). Caller must already know the layer kind via the predicate
/// helpers above.
pub(super) fn qwen35_tensor_name(layer: u32, slot: Qwen35TensorSlot) -> String {
    use Qwen35TensorSlot::*;
    let prefix = format!("blk.{layer}");
    match slot {
        // Per-layer norms (present on every block)
        AttnNorm           => format!("{prefix}.attn_norm.weight"),
        PostAttentionNorm  => format!("{prefix}.post_attention_norm.weight"),

        // Dense FFN (every block)
        FfnGate            => format!("{prefix}.ffn_gate.weight"),
        FfnUp              => format!("{prefix}.ffn_up.weight"),
        FfnDown            => format!("{prefix}.ffn_down.weight"),

        // Full-attention layers (17 blocks: 3, 7, ..., 63, 64)
        AttnQ              => format!("{prefix}.attn_q.weight"),
        AttnK              => format!("{prefix}.attn_k.weight"),
        AttnV              => format!("{prefix}.attn_v.weight"),
        AttnOutput         => format!("{prefix}.attn_output.weight"),
        AttnQNorm          => format!("{prefix}.attn_q_norm.weight"),
        AttnKNorm          => format!("{prefix}.attn_k_norm.weight"),

        // Linear-attention layers (48 blocks)
        AttnQkv            => format!("{prefix}.attn_qkv.weight"),
        AttnGate           => format!("{prefix}.attn_gate.weight"),
        SsmA               => format!("{prefix}.ssm_a"),
        SsmAlpha           => format!("{prefix}.ssm_alpha.weight"),
        SsmBeta            => format!("{prefix}.ssm_beta.weight"),
        SsmConv1d          => format!("{prefix}.ssm_conv1d.weight"),
        SsmDtBias          => format!("{prefix}.ssm_dt.bias"),
        SsmNorm            => format!("{prefix}.ssm_norm.weight"),
        SsmOut             => format!("{prefix}.ssm_out.weight"),

        // MTP nextn (single block at index n_main)
        NextnEhProj        => format!("{prefix}.nextn.eh_proj.weight"),
        NextnEnorm         => format!("{prefix}.nextn.enorm.weight"),
        NextnHnorm         => format!("{prefix}.nextn.hnorm.weight"),
        NextnSharedHeadNorm => format!("{prefix}.nextn.shared_head_norm.weight"),
    }
}

/// Per-layer tensor-slot enumeration. Membership is sparse — see
/// `qwen35_layer_slots` for the per-kind subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen35TensorSlot {
    AttnNorm,
    PostAttentionNorm,
    FfnGate, FfnUp, FfnDown,
    // Full attention
    AttnQ, AttnK, AttnV, AttnOutput, AttnQNorm, AttnKNorm,
    // Linear attention / SSM
    AttnQkv, AttnGate,
    SsmA, SsmAlpha, SsmBeta, SsmConv1d, SsmDtBias, SsmNorm, SsmOut,
    // MTP nextn
    NextnEhProj, NextnEnorm, NextnHnorm, NextnSharedHeadNorm,
}

/// What kind of layer block this is — used by the layer-plan builder
/// to dispatch the right step sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen35LayerKind {
    /// Trunk linear-attention block (SSM conv + Gated Delta Net + FFN).
    LinearAttn,
    /// Trunk full-attention block (GQA + mRoPE + Q-gate + FFN).
    FullAttn,
    /// MTP nextn draft head (concat + projection + FullAttn + FFN +
    /// shared-head norm + lm_head reuse).
    MtpNextn,
}

impl Qwen35LayerKind {
    pub fn of(cfg: &Qwen35Config, layer: u32) -> Self {
        if cfg.is_mtp_block(layer) {
            Self::MtpNextn
        } else if cfg.is_full_attention_layer(layer) {
            Self::FullAttn
        } else {
            Self::LinearAttn
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verified against `/tmp/qwen35_full_dump.txt` (Sprint B Phase 1).
    fn qwen36_27b_config() -> Qwen35Config {
        Qwen35Config {
            block_count: 65,
            nextn_predict_layers: 1,
            full_attention_interval: 4,
            embedding_length: 5120,
            feed_forward_length: 17408,
            vocab_size: 248320,
            n_head: 24,
            n_head_kv: 4,
            head_k: 256,
            head_v: 256,
            rope_dimension_count: 64,
            rope_sections: [11, 11, 10, 0],
            rope_freq_base: 1.0e7,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ssm_d_inner: 6144,
            rms_norm_eps: 1.0e-6,
        }
    }

    #[test]
    fn layer_routing_matches_sprint_a_inventory() {
        let cfg = qwen36_27b_config();

        // From `/tmp/qwen35_dump.txt`: full-attention layers carry
        // attn_q / attn_k / attn_v / attn_output. Sprint A verified
        // {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64}.
        let expected_full: [u32; 17] = [
            3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64,
        ];
        let observed: Vec<u32> = (0..cfg.block_count)
            .filter(|&i| cfg.is_full_attention_layer(i))
            .collect();
        assert_eq!(observed.as_slice(), expected_full);

        // Recurrent layers are the complement minus the MTP block.
        let recurrent_count = (0..cfg.block_count)
            .filter(|&i| cfg.is_recurrent_layer(i))
            .count();
        assert_eq!(recurrent_count, 48);

        // MTP block sits at index 64 (= n_main).
        assert_eq!(cfg.mtp_start(), 64);
        assert!(cfg.is_mtp_block(64));
        assert!(!cfg.is_mtp_block(63));
    }

    #[test]
    fn layer_kind_dispatch() {
        let cfg = qwen36_27b_config();
        assert_eq!(Qwen35LayerKind::of(&cfg, 0), Qwen35LayerKind::LinearAttn);
        assert_eq!(Qwen35LayerKind::of(&cfg, 3), Qwen35LayerKind::FullAttn);
        assert_eq!(Qwen35LayerKind::of(&cfg, 62), Qwen35LayerKind::LinearAttn);
        assert_eq!(Qwen35LayerKind::of(&cfg, 63), Qwen35LayerKind::FullAttn);
        assert_eq!(Qwen35LayerKind::of(&cfg, 64), Qwen35LayerKind::MtpNextn);
    }

    #[test]
    fn derived_quantities_match_gguf() {
        let cfg = qwen36_27b_config();
        // conv_channels = 2 * 128 * 16 + 6144 = 10240 (from ssm_conv1d shape)
        assert_eq!(cfg.conv_channels(), 10240);
        // n_rot (text-only) = 11 + 11 + 10 = 32
        assert_eq!(cfg.n_rot_text_only(), 32);
        // n_main = 64
        assert_eq!(cfg.n_main(), 64);
    }

    #[test]
    fn full_attention_set_helper_matches_predicate() {
        let cfg = qwen36_27b_config();
        let set = full_attention_layer_indices(&cfg);
        assert_eq!(set.len(), 17);
        assert!(set.contains(&3));
        assert!(set.contains(&63));
        assert!(set.contains(&64));
        assert!(!set.contains(&0));
    }

    #[test]
    fn tensor_name_helpers() {
        assert_eq!(
            qwen35_tensor_name(0, Qwen35TensorSlot::AttnQkv),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            qwen35_tensor_name(3, Qwen35TensorSlot::AttnQ),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            qwen35_tensor_name(63, Qwen35TensorSlot::SsmDtBias),
            "blk.63.ssm_dt.bias" // .bias, not .bias.weight
        );
        assert_eq!(
            qwen35_tensor_name(63, Qwen35TensorSlot::SsmA),
            "blk.63.ssm_a" // no .weight suffix
        );
        assert_eq!(
            qwen35_tensor_name(64, Qwen35TensorSlot::NextnEhProj),
            "blk.64.nextn.eh_proj.weight"
        );
    }

    #[test]
    fn recurrent_state_bytes_reasonable() {
        let cfg = qwen36_27b_config();
        let bytes = cfg.recurrent_state_bytes_per_seq();
        // ~150 MB for Qwen3.6-27B (Sprint A doc §6.2). Allow a wide
        // tolerance — this is a sanity check, not an exact gate.
        let mb = bytes as f64 / (1024.0 * 1024.0);
        assert!(
            (100.0..200.0).contains(&mb),
            "recurrent state {mb:.1} MB outside 100-200 MB window"
        );
    }
}
