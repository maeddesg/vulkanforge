//! KV prefix-reuse byte-identity gate (GPU integration; self-skips when
//! the Qwen3-8B model isn't on disk, like `regression.rs`).
//!
//! The server's cross-request KV prefix-reuse (`ServerSession::generate_reuse`,
//! `state.rs`) keeps the prior request's resident KV `[0..k)` and prefills only
//! the suffix at `base_pos = k`, where `k = lcp(cached, full)` (capped to
//! `len-1`). The `longest_common_prefix` boundary is unit-tested in `state.rs`;
//! what was NOT tested before this file is the load-bearing GPU claim:
//!
//!   prefilling the suffix on top of a *retained* KV prefix produces the same
//!   next-token logits as a fresh single-shot prefill of the full sequence —
//!   including when the suffix OVERWRITES positions that held different (stale)
//!   tokens from the prior request, and NOT reusing stale KV past `k`.
//!
//! This is the correctness gate for defaulting reuse ON. Three scenarios drive
//! the real batched prefill kernel (`Forward::prefill_batch`, the path Qwen3-8B
//! takes in production) and compare logits:
//!   1. CLEAN     — full = cachedPrefix + suffix (no divergence).
//!   2. DIVERGENT — cached and full share a prefix then diverge; the suffix
//!                  overwrites stale positions. Proves no stale-KV leak.
//!   3. OVERSHOOT — deliberately reuse PAST the true lcp (the `VF_KV_REUSE_
//!                  OVERSHOOT` fault-injection case). Proves the boundary is
//!                  load-bearing: a wrong reuse is detectably different.
//!
//! Assertion level: logit argmax bit-equality + top-5 overlap + max-abs-diff
//! report. Logits are read with `Forward::logits()`, so the comparison sees the
//! full distribution (covers any sampling temperature, not just greedy). Raw
//! f32 bit-identity across different prefill *batch shapes* is NOT asserted —
//! the existing `sprint5b_chunked_prefill_parity` test establishes that this
//! engine relies on argmax+top-5 (different batch decompositions can differ at
//! the ULP level); see the report for the measured diff.

use std::path::PathBuf;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;

fn qwen3_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VF_MODEL_PATH").map(PathBuf::from) {
        return Some(p);
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models").join("Qwen3-8B-Q4_K_M.gguf");
    if p.exists() { Some(p) } else { None }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx.truncate(k);
    idx
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Harness owning the GPU stack for one model load. All scenarios run on the
/// same `Forward` (reset between fresh computations) to amortise the ~5 s load.
struct Harness {
    dev: VulkanDevice,
    allocator: Allocator,
    registry: PipelineRegistry,
    cmd_ctx: CommandContext,
    gguf: GgufFile,
    cfg: ModelConfig,
    model: LoadedModel,
    fwd: Forward,
}

impl Harness {
    fn new(path: &PathBuf) -> Self {
        let dev = VulkanDevice::new().expect("VulkanDevice::new");
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: dev.instance.clone(),
            device: dev.device.clone(),
            physical_device: dev.physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .expect("Allocator::new");
        let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
        let gguf = GgufFile::open(path).expect("open");
        let cfg = ModelConfig::from_gguf(&gguf).expect("config");
        let model = LoadedModel::load(&dev, &mut allocator, &gguf, None).expect("load");
        let kv = KvCache::new(
            &dev.device,
            &mut allocator,
            KvCacheConfig {
                n_layers: cfg.n_layers,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
                max_seq_len: 256,
                // Mirror the server (main.rs / serve.rs): Gemma-4 ships
                // heterogeneous per-layer head_dim (sliding 256 / full 512)
                // and KV-head counts (8 / 2). Passing None here would build a
                // uniform-stride cache → wrong layout for Gemma-4 → garbage
                // logits (an artifact, not a reuse bug). Other architectures
                // (e.g. Qwen3) keep None and use the uniform stride.
                per_layer_head_dim: cfg
                    .gemma4
                    .as_ref()
                    .map(|g| g.layers.iter().map(|s| s.head_dim).collect()),
                per_layer_n_kv_heads: cfg
                    .gemma4
                    .as_ref()
                    .map(|g| g.layers.iter().map(|s| s.n_kv_heads).collect::<Vec<_>>()),
            },
        )
        .expect("kv_cache");
        // max_prefill_tokens large enough that every sequence here is one batch.
        let fwd = Forward::new_with_prefill(&dev, &mut allocator, kv, cfg.clone(), None, 64)
            .expect("forward");
        // Prove which KV dtype is actually live (guards against a "false green"
        // where VULKANFORGE_KV_FP8=1 is set but the cache silently stays F16).
        eprintln!(
            "[kv_reuse_ident] KV dtype = {} (is_fp8={})",
            fwd.kv_cache.kv_dtype.label(),
            fwd.kv_cache.is_fp8(),
        );
        Self { dev, allocator, registry, cmd_ctx, gguf, cfg, model, fwd }
    }

    fn embeds(&self, tokens: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * self.cfg.hidden_dim as usize);
        for &t in tokens {
            out.extend(embedding_row(&self.gguf, &self.cfg, t).expect("embedding_row"));
        }
        out
    }

    /// One `prefill_batch` of `tokens` written at `base_pos`. Does NOT reset —
    /// the caller controls `current_seq_len` to mirror `generate_reuse`. The
    /// `tokens` slice is also passed as `token_ids` so the Gemma-4 PLE
    /// pre-stage fires for that architecture (it's a no-op for models without
    /// per-layer embeddings, e.g. Qwen3 — `prefill_batch` gates it on
    /// `model.ple_data.is_some()`); production `generate_from_tokens` likewise
    /// feeds real ids, so this keeps the FP8/26B path faithful.
    fn prefill(&mut self, tokens: &[u32], base_pos: u32) {
        let embeds = self.embeds(tokens);
        self.fwd
            .prefill_batch(
                &self.dev, &self.registry, &self.cmd_ctx, &self.model,
                &embeds, tokens.len() as u32, base_pos, tokens,
            )
            .expect("prefill_batch");
    }

    /// Fresh single-shot prefill of the whole sequence from position 0.
    fn fresh_logits(&mut self, full: &[u32]) -> Vec<f32> {
        self.fwd.kv_cache.reset();
        self.prefill(full, 0);
        self.fwd.logits().expect("logits")
    }

    /// Reuse path: populate the prior request's KV (`prior`), then mirror
    /// `generate_reuse` by setting `current_seq_len = keep_k` and prefilling the
    /// suffix `full[keep_k..]` at `base_pos = keep_k`. `keep_k` is the caller's
    /// chosen reuse depth (= lcp for a legit reuse, > lcp to force over-reuse).
    fn reuse_logits(&mut self, prior: &[u32], full: &[u32], keep_k: u32) -> Vec<f32> {
        self.fwd.kv_cache.reset();
        self.prefill(prior, 0);
        // Mirror generate_reuse: drop the counter to the reuse depth, keep the
        // resident KV bytes [0..keep_k), prefill the suffix over [keep_k..).
        self.fwd.kv_cache.current_seq_len = keep_k;
        self.prefill(&full[keep_k as usize..], keep_k);
        self.fwd.logits().expect("logits")
    }

    fn destroy(self) {
        let Harness { dev, mut allocator, registry, cmd_ctx, model, fwd, .. } = self;
        fwd.destroy(&dev.device, &mut allocator);
        cmd_ctx.destroy(&dev.device);
        model.destroy(&dev.device, &mut allocator);
        registry.destroy(&dev.device);
        drop(allocator);
    }
}

// Token ids (all < vocab). The exact ids are irrelevant; what matters is that
// `stale_tail` and `new_tail` differ so DIVERGENT/OVERSHOOT actually exercise a
// stale-KV overwrite. lcp(prior, full) = shared.len() = 7 in every scenario.
const SHARED: [u32; 7] = [9707, 1879, 264, 1207, 11, 358, 2776];
const STALE_TAIL: [u32; 2] = [16, 17];
const NEW_TAIL: [u32; 3] = [220, 221, 222];

/// Returns `(argmax_eq, max_abs_diff, bit_identical)`.
fn report(tag: &str, fresh: &[f32], other: &[f32]) -> (bool, f32, bool) {
    let a_f = argmax(fresh);
    let a_o = argmax(other);
    let diff = max_abs_diff(fresh, other);
    let t5_f = top_k(fresh, 5);
    let t5_o = top_k(other, 5);
    let overlap = t5_f.iter().filter(|t| t5_o.contains(t)).count();
    let bitid = bit_identical(fresh, other);
    let argmax_eq = a_f == a_o;
    eprintln!(
        "[kv_reuse_ident] {tag}: argmax fresh={a_f} other={a_o} eq={argmax_eq} \
         top5_overlap={overlap}/5 max_abs_diff={diff:.6e} bit_identical={bitid}",
    );
    (argmax_eq, diff, bitid)
}

/// CLEAN: full = cachedPrefix + suffix. Reuse keeps the prefix, prefills the
/// suffix. Must match a fresh full prefill.
#[test]
fn reuse_clean_prefix_matches_fresh() {
    let Some(path) = qwen3_path() else { return };
    let mut h = Harness::new(&path);

    let full: Vec<u32> = SHARED.iter().chain(NEW_TAIL.iter()).copied().collect();
    let fresh = h.fresh_logits(&full);
    // prior = just the shared prefix; lcp = 7; reuse keeps [0..7).
    let reuse = h.reuse_logits(&SHARED, &full, SHARED.len() as u32);

    let (argmax_eq, diff, bitid) = report("CLEAN", &fresh, &reuse);
    h.destroy();
    assert!(argmax_eq, "CLEAN: reuse argmax must equal fresh single-shot prefill");
    // Measured: logits are bit-identical (max_abs_diff == 0.0), stable
    // run-to-run. Assert the strongest level — bit-identity covers EVERY
    // sampling temperature, not just greedy. A future kernel change that
    // introduces even ULP drift here fires the gate and must be re-justified.
    assert!(
        bitid,
        "CLEAN: reuse logits must be bit-identical to fresh (max_abs_diff={diff:.6e})",
    );
}

/// DIVERGENT (the key no-stale-leak test): prior = shared+STALE_TAIL, full =
/// shared+NEW_TAIL. lcp = 7. Reuse keeps [0..7) and the suffix prefill at pos 7
/// must OVERWRITE the stale STALE_TAIL KV at positions 7,8 with NEW_TAIL — never
/// reuse the stale entries. Result must equal a fresh prefill of `full`.
#[test]
fn reuse_divergent_prefix_overwrites_stale_kv() {
    let Some(path) = qwen3_path() else { return };
    let mut h = Harness::new(&path);

    let prior: Vec<u32> = SHARED.iter().chain(STALE_TAIL.iter()).copied().collect();
    let full: Vec<u32> = SHARED.iter().chain(NEW_TAIL.iter()).copied().collect();
    let fresh = h.fresh_logits(&full);
    // lcp(prior, full) = 7 (shared); legit reuse keeps exactly [0..7).
    let reuse = h.reuse_logits(&prior, &full, SHARED.len() as u32);

    let (argmax_eq, diff, bitid) = report("DIVERGENT", &fresh, &reuse);
    h.destroy();
    assert!(
        argmax_eq,
        "DIVERGENT: stale KV from the prior request leaked — suffix prefill did \
         not fully overwrite positions past the common prefix",
    );
    // Measured: bit-identical even though positions 7,8 were overwritten from
    // STALE_TAIL to NEW_TAIL. Any stale-KV leak would perturb these logits
    // (cf. the OVERSHOOT case: max_abs_diff ~9.58). Bit-identity here is the
    // hard no-leak proof at byte level, all temperatures.
    assert!(
        bitid,
        "DIVERGENT: reuse logits must be bit-identical to a fresh prefill of the \
         full sequence (max_abs_diff={diff:.6e}) — any nonzero diff means stale KV leaked",
    );
}

/// OVERSHOOT teeth: same prior/full as DIVERGENT, but deliberately reuse PAST
/// the true lcp (keep_k = lcp+2 = 9), retaining the stale STALE_TAIL KV at
/// positions 7,8 instead of overwriting them. This is the `VF_KV_REUSE_OVERSHOOT
/// =2` fault. The result MUST differ from a fresh prefill — proving the lcp
/// boundary is load-bearing and a wrong reuse is detectable (not a silent
/// corruption). The legit reuse (keep_k = 7) on the same inputs must match.
#[test]
fn overshoot_past_lcp_corrupts_logits() {
    let Some(path) = qwen3_path() else { return };
    let mut h = Harness::new(&path);

    let prior: Vec<u32> = SHARED.iter().chain(STALE_TAIL.iter()).copied().collect();
    let full: Vec<u32> = SHARED.iter().chain(NEW_TAIL.iter()).copied().collect();
    let fresh = h.fresh_logits(&full);

    // Legit reuse at the true lcp — must match fresh (control), bit-identical.
    let legit = h.reuse_logits(&prior, &full, SHARED.len() as u32);
    let (_legit_eq, _legit_diff, legit_bitid) = report("OVERSHOOT/legit-control", &fresh, &legit);

    // Over-reuse: keep_k = 9 retains stale positions 7,8 (STALE_TAIL) and only
    // prefills full[9..] = [222] at pos 9. Positions 7,8 now hold 16,17 instead
    // of 220,221 → corrupted attention → logits must diverge from fresh.
    let overshoot = h.reuse_logits(&prior, &full, (SHARED.len() + STALE_TAIL.len()) as u32);
    let (_overshoot_eq, overshoot_diff, overshoot_bitid) =
        report("OVERSHOOT/over-reuse", &fresh, &overshoot);

    h.destroy();
    assert!(
        legit_bitid,
        "OVERSHOOT control: legit reuse at the true lcp must be bit-identical to fresh",
    );
    // Teeth: over-reuse must visibly corrupt the logits (measured max_abs_diff
    // ~9.58). If it were silently bit-identical, the lcp boundary would NOT be
    // load-bearing and the byte-ident gate above would be vacuous.
    assert!(
        !overshoot_bitid && overshoot_diff > 1e-2,
        "OVERSHOOT teeth: over-reusing past the lcp must corrupt logits \
         (bit_identical={overshoot_bitid}, max_abs_diff={overshoot_diff:.6e}) — if this \
         passes silently, the byte-ident boundary is NOT load-bearing",
    );
}
