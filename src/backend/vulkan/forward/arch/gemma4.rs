//! Sprint 44B-1 — extracted from `forward/mod.rs` (pure code-move).
//!
//! Gemma-4-only helpers consumed by the dispatch paths:
//! - `rope_params_for_layer`: per-layer (head_dim, rotary_dim, freq_base,
//!   theta_scale). Falls back to the uniform RoPE params for non-Gemma-4
//!   stacks.
//! - `gemma4_kv_read_layer`, `gemma4_layer_owns_kv`: cross-layer KV-share
//!   plumbing for Gemma-4's Own/Subscribes layout.
//! - `gemma4_kv_start`: sliding-window-attention lower bound.
//! - `apply_final_logit_softcap`: post-lm_head logit soft-cap.

use super::super::super::gguf::{Gemma4KvSource, Gemma4LayerKind, ModelConfig};

/// Sprint 43D-2 — per-layer RoPE parameters. Returns
/// `(head_dim_geom, rotary_dim, freq_base, theta_scale)` where:
/// - `head_dim_geom` is the physical row stride (256 sliding / 512 full
///   for Gemma-4 E2B; uniform `cfg.head_dim` otherwise);
/// - `rotary_dim` is the rotation extent (= head_dim_geom × p_factor for
///   p-RoPE full layers; 0.25 × 512 = 128 for E2B full layers);
/// - `freq_base` is the per-layer θ (10 000 sliding / 1 000 000 full
///   for Gemma-4 E2B; `cfg.rope_freq_base` otherwise);
/// - `theta_scale` = `(1/freq_base)^(2/rotary_dim)`, i.e. the RoPE
///   per-pair frequency scale matching `freq_base` and `rotary_dim`.
///
/// Falls back to the uniform `(head_dim, head_dim, rope_freq_base,
/// rope_theta_scale)` for non-Gemma-4 stacks. Bit-identical math on
/// the existing Llama / Qwen path.
pub(crate) fn rope_params_for_layer(
    cfg: &ModelConfig,
    rope_theta_scale_default: f32,
    layer: u32,
) -> (u32, u32, f32, f32) {
    if let Some(g) = cfg.gemma4.as_ref() {
        let s = &g.layers[layer as usize];
        // Sprint 43D-4 bisect — `VF_DISABLE_PROPE=1` rotates the FULL
        // head_dim (no partial-RoPE). All other Gemma-4 RoPE params
        // (per-layer θ for sliding vs. full layers) are preserved.
        let disable_prope = std::env::var("VF_DISABLE_PROPE").is_ok();
        let rotary_dim = if disable_prope {
            s.head_dim
        } else {
            match s.rope_partial_factor {
                Some(f) => ((s.head_dim as f32) * f).round() as u32,
                None => s.head_dim,
            }
        };
        let theta_scale = (1.0_f32 / s.rope_theta).powf(2.0 / rotary_dim as f32);
        (s.head_dim, rotary_dim, s.rope_theta, theta_scale)
    } else {
        (
            cfg.head_dim,
            cfg.head_dim,
            cfg.rope_freq_base,
            rope_theta_scale_default,
        )
    }
}

/// Sprint 43D-2 — KV-source mapping for Gemma-4 cross-layer KV-sharing.
/// Returns the layer index whose KV-cache slab this layer should read
/// from. For "Own" / "OwnAndPublishesSliding" / "OwnAndPublishesFull"
/// layers this is `layer` itself; for "SubscribesSliding" /
/// "SubscribesFull" layers it's the publisher's layer index (the first
/// layer with the matching `OwnAndPublishesXxx` source). Non-Gemma-4
/// stacks always return `layer`.
///
/// Sprint 43D-4 bisect — `VF_DISABLE_KV_SHARE=1` forces every layer to
/// read its own slab (Subscribe layers get treated as Own). Useful to
/// isolate whether a coherence regression comes from cross-layer KV-
/// sharing semantics vs. some other feature.
pub(crate) fn gemma4_kv_read_layer(cfg: &ModelConfig, layer: u32) -> u32 {
    if std::env::var("VF_DISABLE_KV_SHARE").is_ok() {
        return layer;
    }
    let g = match cfg.gemma4.as_ref() {
        Some(g) => g,
        None => return layer,
    };
    let s = &g.layers[layer as usize];
    let want_publish = match s.kv_source {
        Gemma4KvSource::Own
        | Gemma4KvSource::OwnAndPublishesSliding
        | Gemma4KvSource::OwnAndPublishesFull => return layer,
        Gemma4KvSource::SubscribesSliding => Gemma4KvSource::OwnAndPublishesSliding,
        Gemma4KvSource::SubscribesFull => Gemma4KvSource::OwnAndPublishesFull,
    };
    g.layers
        .iter()
        .enumerate()
        .find(|(_, ls)| ls.kv_source == want_publish)
        .map(|(i, _)| i as u32)
        .unwrap_or(layer)
}

/// Sprint 43D-2 — does this layer compute and write its own K/V
/// projections, or does it subscribe to a published slab from an
/// earlier layer? Returns `true` for non-Gemma-4 stacks (they always
/// own their KV) and for Gemma-4 layers in `Own` /
/// `OwnAndPublishesSliding` / `OwnAndPublishesFull`.
///
/// Sprint 43D-4 bisect — `VF_DISABLE_KV_SHARE=1` forces every layer to
/// own its KV (= compute + write its own K/V projections every step).
pub(crate) fn gemma4_layer_owns_kv(cfg: &ModelConfig, layer: u32) -> bool {
    if std::env::var("VF_DISABLE_KV_SHARE").is_ok() {
        return true;
    }
    let g = match cfg.gemma4.as_ref() {
        Some(g) => g,
        None => return true,
    };
    !matches!(
        g.layers[layer as usize].kv_source,
        Gemma4KvSource::SubscribesSliding | Gemma4KvSource::SubscribesFull
    )
}

/// Sprint 43D-2 — sliding-window-attention lower bound. For Gemma-4
/// sliding layers returns `max(0, position+1 - sliding_window)`; for
/// every other layer (Gemma-4 full layers AND non-Gemma-4 stacks)
/// returns 0 (= attend to the full causal history).
///
/// Sprint 43D-4 bisect — `VF_DISABLE_SLIDING_WINDOW=1` always returns 0
/// (= every layer attends to the full causal history, no window mask).
pub(crate) fn gemma4_kv_start(cfg: &ModelConfig, layer: u32, position: u32) -> u32 {
    if std::env::var("VF_DISABLE_SLIDING_WINDOW").is_ok() {
        return 0;
    }
    let g = match cfg.gemma4.as_ref() {
        Some(g) => g,
        None => return 0,
    };
    let s = &g.layers[layer as usize];
    if !matches!(s.kind, Gemma4LayerKind::Sliding) {
        return 0;
    }
    let seq_len = position + 1;
    seq_len.saturating_sub(g.sliding_window)
}

/// Sprint 43C — Gemma-4 final logit soft-cap. Applied CPU-side
/// after the lm_head GEMV (or CPU lm_head Q6_K GEMV) and before
/// sampling. No-op for every other architecture (`config.gemma4 ==
/// None` or `final_logit_softcapping == None`).
///
/// Formula: `logits[i] = cap × tanh(logits[i] / cap)` — caps the
/// magnitude of every logit at `cap` (≈ 30.0 for Gemma-4 E2B). At
/// greedy temperature this shouldn't change the argmax for typical
/// post-`lm_head` distributions, but it keeps top-k / top-p / temp ≠ 0
/// sampling consistent with HF reference behaviour.
pub(crate) fn apply_final_logit_softcap(config: &ModelConfig, logits: &mut [f32]) {
    let cap = match config.gemma4.as_ref().and_then(|g| g.final_logit_softcapping) {
        Some(c) if c > 0.0 => c,
        _ => return,
    };
    let inv = 1.0 / cap;
    for x in logits.iter_mut() {
        *x = cap * (*x * inv).tanh();
    }
}
