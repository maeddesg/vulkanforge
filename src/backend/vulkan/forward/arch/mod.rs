//! Sprint 44B-1 — architecture-specific helpers split out of
//! `forward/mod.rs`. Pure code-move:
//!
//! - `common.rs` holds the layer-weight lookup helpers, the GEMM/GEMV
//!   shader selection, the dispatch-dim helpers, and the two compute
//!   pipeline barriers — none of these are arch-specific (they cover
//!   Llama / Qwen / Mistral / Gemma alike).
//! - `gemma4.rs` holds the Gemma-4-only helpers (per-layer RoPE
//!   parameters, KV-source mapping, sliding-window window start, and the
//!   final-logit soft-cap).
//!
//! Both submodules `pub(super)` re-export their items so the parent
//! `forward/mod.rs` can reach them via `use arch::*;`.

pub(super) mod common;
pub(super) mod gemma4;
/// Sprint B (v0.4.6 prep) — Qwen3.5 / Qwen3.6 (`qwen35`) skeleton.
/// Config + layer-kind routing + tensor-name helpers, no dispatch
/// wiring yet. See `docs/qwen35_architecture_analysis.md` and
/// `results/sprint_b_qwen35_prechecks.md`.
#[allow(dead_code)]
pub(super) mod qwen35;

pub(super) use common::{
    GemmKind, compute_barrier, is_f32_layer_weight, is_fp8_layer_weight, layer_dims,
    layer_weight, layer_weight_indexed_shader, layer_weight_mm_id_shader,
    layer_weight_mmq_id_shader, layer_weight_opt, layer_weight_scale_block,
    layer_weight_scale_buf, layer_weight_scale_scalar, layer_weight_shader,
    layer_weight_shader_gemm, n_kv_heads_for, transfer_to_compute_barrier,
};
pub(super) use gemma4::{
    apply_final_logit_softcap, gemma4_kv_read_layer, gemma4_kv_start,
};
