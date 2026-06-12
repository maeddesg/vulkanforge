//! Auto context-size sizing for `vulkanforge serve`.
//!
//! When `--ctx-size` is omitted, the server picks the largest *safe* KV
//! context for the loaded model from three bounds:
//!
//! 1. **VRAM**: `(free − weights − reserve) / kv_bytes_per_token`, where
//!    `free` is live VRAM headroom (`VK_EXT_memory_budget`), `weights` is
//!    the GGUF on-disk footprint, and `reserve` is a conservative
//!    constant for everything allocated *after* the KV cache (pipeline
//!    registry, Forward scratch, MoE-router scratch, and the transient
//!    prefill activation buffers).
//! 2. **Model max context** (GGUF training horizon) — a hard ceiling.
//! 3. **Hardware LDS cap** — `scalar_attn.comp` allocates `shared float
//!    scores[MAX_SEQ]` (4 B/entry) and is compiled unconditionally by the
//!    pipeline registry, so `MAX_SEQ × 4` must fit in
//!    `maxComputeSharedMemorySize` (65536 on RDNA4 → cap 16384) or the
//!    load aborts at pipeline creation. This is a real, device-queried
//!    hardware limit, not a policy knob.
//! 4. **Sane cap** — so a 128k-context model doesn't allocate an absurd
//!    KV cache just because VRAM happens to allow it.
//!
//! The result is rounded down to a multiple of [`ROUND_TO`] and floored
//! at [`MIN_CTX`]. The policy is deliberately conservative: we would
//! rather hand back some context than OOM mid-load. An explicit
//! `--ctx-size N` bypasses all of this and is used verbatim.

/// Compute/scratch reserve (MiB) held back beyond weights + KV: the
/// pipeline registry, Forward scratch buffers, MoE-router scratch, and
/// the transient prefill activation buffers — all allocated *after* the
/// KV cache, so they must be subtracted up front. Conservative by design.
/// Tunable via `VF_AUTO_CTX_RESERVE_MIB`.
pub const DEFAULT_RESERVE_MIB: u64 = 1536;

/// Upper bound on the auto-selected context. A model advertising 128k+
/// context shouldn't burn all of VRAM on KV just because it fits.
pub const SANE_CAP: u32 = 32768;

/// Round the chosen context down to a multiple of this.
pub const ROUND_TO: u32 = 256;

/// Never auto-pick below this. A computed context this small means VRAM
/// is pathologically tight; we floor here and let the subsequent KV
/// allocation / VRAM gate surface a real OOM loudly rather than silently
/// picking a useless 128-token window.
pub const MIN_CTX: u32 = 512;

/// Bytes of LDS consumed per context token by `scalar_attn.comp`'s
/// `shared float scores[MAX_SEQ]` (one f32 per token).
pub const SCALAR_ATTN_LDS_BYTES_PER_TOKEN: u32 = 4;

/// Largest `max_seq` whose `scalar_attn.comp` LDS allocation fits in the
/// device's `maxComputeSharedMemorySize`. Above this, pipeline creation
/// rejects the shader and the load aborts.
pub fn lds_ctx_cap(max_compute_shared_memory_bytes: u32) -> u32 {
    max_compute_shared_memory_bytes / SCALAR_ATTN_LDS_BYTES_PER_TOKEN
}

/// Which of the four bounds determined the chosen context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtxBound {
    /// Limited by available VRAM.
    Vram,
    /// Limited by the model's training context length.
    ModelMax,
    /// Limited by the device LDS budget (`scalar_attn.comp` `scores[]`).
    HwLds,
    /// Limited by [`SANE_CAP`].
    SaneCap,
}

impl CtxBound {
    pub fn label(self) -> &'static str {
        match self {
            CtxBound::Vram => "VRAM",
            CtxBound::ModelMax => "model max context",
            CtxBound::HwLds => "hardware LDS limit",
            CtxBound::SaneCap => "sane cap",
        }
    }
}

/// Outcome of [`compute_auto_ctx`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoCtx {
    /// The chosen `max_seq_len` (rounded, floored).
    pub ctx: u32,
    /// Which bound was binding (for the transparency line).
    pub bound: CtxBound,
    /// Bytes left for the KV cache after weights + reserve (for the
    /// transparency line). `free − weights − reserve`, saturating.
    pub avail_for_kv: u64,
}

/// Reserve in bytes, honoring `VF_AUTO_CTX_RESERVE_MIB`.
pub fn reserve_bytes() -> u64 {
    let mib = std::env::var("VF_AUTO_CTX_RESERVE_MIB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_RESERVE_MIB);
    mib.saturating_mul(1024 * 1024)
}

/// Pure auto-ctx formula. All byte quantities are bytes;
/// `kv_bytes_per_token` is the per-token KV footprint (both buffers).
/// Returns the chosen context, the binding constraint, and the bytes
/// left for KV. Saturating throughout — no panic on absurd inputs.
pub fn compute_auto_ctx(
    free_vram: u64,
    weights: u64,
    reserve: u64,
    kv_bytes_per_token: u64,
    model_max_ctx: u32,
    hw_lds_cap: u32,
    sane_cap: u32,
) -> AutoCtx {
    let avail = free_vram.saturating_sub(weights).saturating_sub(reserve);
    // Guard against a zero divisor (degenerate config): treat as 1 so the
    // VRAM bound becomes "huge" and the other bounds win.
    let kv_bpt = kv_bytes_per_token.max(1);
    let vram_ctx = u32::try_from(avail / kv_bpt).unwrap_or(u32::MAX);

    // The binding constraint is whichever is smallest. Each `if cap < ctx`
    // only fires when `cap` is below the running minimum, so the final
    // `bound` is always the label of the true argmin (ties → earlier).
    let mut ctx = vram_ctx;
    let mut bound = CtxBound::Vram;
    if model_max_ctx < ctx {
        ctx = model_max_ctx;
        bound = CtxBound::ModelMax;
    }
    if hw_lds_cap < ctx {
        ctx = hw_lds_cap;
        bound = CtxBound::HwLds;
    }
    if sane_cap < ctx {
        ctx = sane_cap;
        bound = CtxBound::SaneCap;
    }

    // Round down to a multiple of ROUND_TO, then floor at MIN_CTX.
    let rounded = (ctx / ROUND_TO) * ROUND_TO;
    let ctx = rounded.max(MIN_CTX);
    AutoCtx { ctx, bound, avail_for_kv: avail }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * 1024 * 1024;
    /// A "no-op" cap larger than anything the other bounds produce, so a
    /// test can isolate one specific bound.
    const HUGE: u32 = 1 << 20;

    #[test]
    fn vram_bound_when_plenty_of_model_context() {
        // 16G free, 9G weights, 1.5G reserve → 5.5G for KV. KV ≈ 0.16
        // MiB/tok → ~35k tokens; raise the other caps so VRAM binds.
        let kv_bpt = 160 * 1024; // 0.16 MiB/tok
        let a = compute_auto_ctx(16 * G, 9 * G, 3 * G / 2, kv_bpt, 131072, HUGE, HUGE);
        assert_eq!(a.bound, CtxBound::Vram);
        assert_eq!(a.avail_for_kv, 16 * G - 9 * G - 3 * G / 2);
        assert!(a.ctx as u64 <= a.avail_for_kv / kv_bpt);
        assert_eq!(a.ctx % ROUND_TO, 0);
    }

    #[test]
    fn model_max_bound_when_vram_is_huge() {
        // Tons of VRAM, small training context → model_max wins.
        let a = compute_auto_ctx(64 * G, 4 * G, G, 80 * 1024, 4096, HUGE, SANE_CAP);
        assert_eq!(a.bound, CtxBound::ModelMax);
        assert_eq!(a.ctx, 4096);
    }

    #[test]
    fn hw_lds_bound_when_lds_is_the_tightest() {
        // Plenty of VRAM, 128k model, big sane cap, but the device LDS
        // caps MAX_SEQ at 16384 (RDNA4: 65536 / 4) → LDS binds.
        let lds_cap = lds_ctx_cap(65536);
        assert_eq!(lds_cap, 16384);
        let a = compute_auto_ctx(64 * G, 4 * G, G, 40 * 1024, 131072, lds_cap, SANE_CAP);
        assert_eq!(a.bound, CtxBound::HwLds);
        assert_eq!(a.ctx, 16384);
    }

    #[test]
    fn sane_cap_bound_for_huge_context_model() {
        // 128k-context model, huge LDS budget, plenty of VRAM → sane cap.
        let a = compute_auto_ctx(64 * G, 4 * G, G, 40 * 1024, 131072, HUGE, SANE_CAP);
        assert_eq!(a.bound, CtxBound::SaneCap);
        assert_eq!(a.ctx, SANE_CAP);
    }

    #[test]
    fn result_is_rounded_down_to_multiple() {
        // avail = 10003 * kv_bpt + slack → vram_ctx = 10003 (non-multiple).
        let kv_bpt = 1_000_000;
        let avail = 10_003u64 * kv_bpt + 123;
        let a = compute_auto_ctx(avail, 0, 0, kv_bpt, 131072, HUGE, HUGE);
        assert_eq!(a.bound, CtxBound::Vram);
        assert_eq!(a.ctx, 9984); // 10003 rounded down to a multiple of 256
        assert_eq!(a.ctx % ROUND_TO, 0);
    }

    #[test]
    fn floors_at_min_ctx_when_vram_pathologically_tight() {
        // Only ~100 tokens' worth of VRAM after weights+reserve → floor.
        let kv_bpt = 1_000_000;
        let a = compute_auto_ctx(9 * G + 100 * kv_bpt, 9 * G, 0, kv_bpt, 131072, HUGE, SANE_CAP);
        assert_eq!(a.bound, CtxBound::Vram);
        assert_eq!(a.ctx, MIN_CTX);
    }

    #[test]
    fn saturates_when_weights_exceed_free() {
        // Weights bigger than free VRAM → avail 0 → floored to MIN_CTX
        // (the real OOM is then surfaced by the KV alloc / VRAM gate).
        let a = compute_auto_ctx(8 * G, 12 * G, G, 100_000, 131072, HUGE, SANE_CAP);
        assert_eq!(a.avail_for_kv, 0);
        assert_eq!(a.ctx, MIN_CTX);
    }

    #[test]
    fn lds_ctx_cap_divides_by_f32_size() {
        assert_eq!(lds_ctx_cap(65536), 16384); // RDNA4
        assert_eq!(lds_ctx_cap(32768), 8192);
    }

    #[test]
    fn reserve_bytes_default_is_1536_mib() {
        // Env-independent assertion on the constant (the env var is
        // process-global; we don't mutate it here to avoid test races).
        assert_eq!(DEFAULT_RESERVE_MIB * 1024 * 1024, 1536 * 1024 * 1024);
    }
}
