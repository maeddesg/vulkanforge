//! Sprint 44B-1 — extracted from `forward/mod.rs` (pure code-move).
//! Sprint 44C-3 — `rope_params_for_layer` and `gemma4_layer_owns_kv`
//! removed; the layer-plan builder duplicates their logic inline at
//! plan-build time, and the dispatch path no longer queries them.
//!
//! Gemma-4-only helpers consumed by the runtime path:
//! - `gemma4_kv_read_layer`: cross-layer KV-share publisher resolution
//!   (called from `runs::run_scalar_attn`).
//! - `gemma4_kv_start`: sliding-window-attention lower bound (same caller).
//! - `apply_final_logit_softcap`: post-lm_head logit soft-cap.

use super::super::super::gguf::{Gemma4KvSource, Gemma4LayerKind, ModelConfig};


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
