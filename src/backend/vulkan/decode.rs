//! Token-by-token decode loop driving [`Forward::forward_token`].
//!
//! Phase 2D introduced a single `generate(prompt)` entry point that
//! resets the KV cache, applies the chat template, prefills, and
//! decodes greedily. Phase 3B splits that into:
//!
//!   1. [`generate_from_tokens`] — the low-level driver that takes
//!      already-tokenised prefill ids and a starting position. Does
//!      NOT touch `kv_cache.reset()`. Used by [`crate::backend::vulkan::chat::ChatSession`]
//!      to extend an existing conversation.
//!   2. [`generate`] — the back-compat wrapper that resets the KV
//!      cache, applies the chat template, and forwards to
//!      `generate_from_tokens`. Existing call sites (Phase-2D
//!      regression test, validation suite) keep working unchanged.
//!
//! Streaming is exposed via a `&mut dyn FnMut(u32, &str)` callback.
//! [`GenerateConfig::print_stream`] is the simple "print every token
//! to stdout" path; callers that want to filter or render differently
//! pass an explicit callback to `generate_from_tokens`.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::{Duration, Instant};

use super::chat_template::ChatTemplate;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::forward::{gpu_direct_moe_enabled, Forward};
use super::gguf::{GgufFile, ModelConfig};
use super::loader::LoadedModel;
use super::pipeline_registry::PipelineRegistry;
use super::q4k;
use super::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: u32,
    pub print_stream: bool,
    /// Strip `<think>...</think>` blocks from the visible output (only
    /// affects streaming; the raw token sequence is unchanged).
    pub think_filter: bool,
    /// Phase 6 v0.1.2 sampling. `None` (or `Sampling::greedy()`) keeps
    /// the legacy argmax-only behaviour every previous phase ran with;
    /// the bench / regression tests pin temperature=0 so their
    /// outputs remain deterministic and comparable to v0.1.1.
    pub sampling: Sampling,
    /// v0.4 Sprint 3 — optional cooperative cancellation. When set,
    /// the decode loop checks the flag once per token and exits
    /// gracefully (treated as a soft EOS) when the bit is `true`.
    /// `None` (the default) keeps the legacy unconditional run-to-
    /// `max_tokens`-or-EOS behaviour — every CLI/bench call site is
    /// unaffected. The server's streaming handler sets this so a
    /// client TCP-disconnect cancels GPU work within ~1 token.
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            print_stream: false,
            think_filter: false,
            sampling: Sampling::greedy(),
            cancel_token: None,
        }
    }
}

/// Sampling configuration for the next-token decision.
///
/// `temperature == 0.0` short-circuits to argmax (greedy decoding) —
/// every benchmark and regression test in the project pins this so
/// outputs stay byte-deterministic. Any other temperature applies the
/// standard `softmax(logits / T)` pipeline, optionally filtered by
/// top-k (keep the K highest-prob candidates) and top-p (smallest
/// candidate set whose cumulative probability exceeds P), with a
/// repetition penalty over the previously emitted tokens applied
/// before temperature scaling.
#[derive(Debug, Clone)]
pub struct Sampling {
    /// Temperature applied to logits before softmax. `0.0` ⇒ greedy.
    pub temperature: f32,
    /// Keep at most `top_k` highest-probability candidates after
    /// softmax. `0` disables the filter.
    pub top_k: u32,
    /// Top-p (nucleus) cutoff in `(0.0, 1.0]`. `1.0` disables.
    pub top_p: f32,
    /// Multiplicative penalty applied to previously generated tokens'
    /// raw logits before temperature scaling. `1.0` disables;
    /// `>1.0` discourages repetition.
    pub repetition_penalty: f32,
    /// Seed for the deterministic xorshift RNG used by the sampler.
    /// Different seeds produce different (still-valid) outputs at
    /// `temperature > 0.0`.
    pub seed: u64,
}

impl Sampling {
    /// Greedy decoding — argmax of the logits, every step. The sampler
    /// short-circuits on `temperature == 0.0` so every other field is
    /// ignored; this is the default everywhere it isn't explicitly
    /// overridden.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl Default for Sampling {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Pick the next token id from a logits row.
///
/// Greedy short-circuit on `temperature == 0.0` keeps the test suite
/// byte-deterministic: every prior phase's regression test runs
/// through this function with greedy sampling and produces the same
/// argmax it did in v0.1.1.
pub fn sample_next_token(
    logits: &mut [f32],
    history: &[u32],
    sampling: &Sampling,
    rng_state: &mut u64,
) -> u32 {
    if sampling.temperature == 0.0 {
        return argmax(logits) as u32;
    }
    // Repetition penalty — divide by `penalty` for previously emitted
    // tokens (positive logits become smaller, negatives become larger
    // — that's the standard Hugging Face formulation).
    if sampling.repetition_penalty > 1.0 {
        for &t in history {
            let i = t as usize;
            if i < logits.len() {
                let v = logits[i];
                logits[i] = if v > 0.0 {
                    v / sampling.repetition_penalty
                } else {
                    v * sampling.repetition_penalty
                };
            }
        }
    }
    // Temperature.
    let inv_t = 1.0 / sampling.temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
    // Softmax over the (post-temperature) logits.
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax(logits) as u32;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
    // Build (id, prob) pairs and sort descending by prob for top-k /
    // top-p filtering.
    let mut pairs: Vec<(u32, f32)> = probs.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if sampling.top_k > 0 {
        pairs.truncate(sampling.top_k as usize);
    }
    if sampling.top_p < 1.0 {
        let mut cumul = 0.0f32;
        let mut keep = pairs.len();
        for (i, &(_, p)) in pairs.iter().enumerate() {
            cumul += p;
            if cumul >= sampling.top_p {
                keep = i + 1;
                break;
            }
        }
        pairs.truncate(keep.max(1));
    }
    // Re-normalise the kept distribution.
    let kept_sum: f32 = pairs.iter().map(|&(_, p)| p).sum();
    if kept_sum <= 0.0 {
        return pairs.first().map(|&(id, _)| id).unwrap_or(0);
    }
    let r = next_rand_unit(rng_state) * kept_sum;
    let mut acc = 0.0f32;
    for &(id, p) in &pairs {
        acc += p;
        if r <= acc {
            return id;
        }
    }
    pairs.last().map(|&(id, _)| id).unwrap_or(0)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    fn synth_logits(values: &[(u32, f32)]) -> Vec<f32> {
        let n = 8;
        let mut v = vec![-10.0f32; n];
        for &(i, x) in values {
            v[i as usize] = x;
        }
        v
    }

    #[test]
    fn greedy_matches_argmax() {
        // Temperature=0 must short-circuit to argmax of the raw logits,
        // ignoring every other sampling field. This is what every
        // benchmark and regression test in the project relies on.
        let mut logits = synth_logits(&[(0, 1.0), (3, 5.0), (7, 2.0)]);
        let mut rng = 0u64;
        let s = Sampling::greedy();
        let id = sample_next_token(&mut logits, &[], &s, &mut rng);
        assert_eq!(id, 3);
    }

    #[test]
    fn temperature_picks_from_softmax() {
        // With temperature=1.0 and no top-k / top-p filter, every
        // token has a positive probability — but the highest-logit
        // one should still dominate. Run many trials and assert the
        // top-logit token wins the plurality.
        let logits = synth_logits(&[(0, 0.5), (3, 4.0), (7, 0.5)]);
        let s = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 1234 };
        let mut counts = [0u32; 8];
        let mut rng = s.seed;
        for _ in 0..2000 {
            let mut l = logits.clone();
            let id = sample_next_token(&mut l, &[], &s, &mut rng);
            counts[id as usize] += 1;
        }
        // Token 3 (logit 4.0) dominates; tokens 0 / 7 (logit 0.5) are
        // ~equal, both far behind. The 4.0-logit-vs-0.5 odds at
        // temperature=1 are ~exp(3.5) ≈ 33×.
        assert!(counts[3] > counts[0] * 5);
        assert!(counts[3] > counts[7] * 5);
        assert!(counts.iter().filter(|&&c| c > 0).count() >= 2,
            "expected the sampler to actually sample, got {counts:?}");
    }

    #[test]
    fn top_k_limits_candidates() {
        // top_k=1 must pick the highest-logit token deterministically,
        // regardless of seed (only one candidate survives the filter).
        let logits = synth_logits(&[(0, 1.0), (3, 4.0), (7, 2.0)]);
        let s = Sampling { temperature: 0.7, top_k: 1, top_p: 1.0,
            repetition_penalty: 1.0, seed: 99 };
        for seed in 0..5u64 {
            let mut l = logits.clone();
            let mut rng = seed;
            let s2 = Sampling { seed, ..s.clone() };
            let id = sample_next_token(&mut l, &[], &s2, &mut rng);
            assert_eq!(id, 3, "top_k=1 must always pick the argmax");
        }
    }

    #[test]
    fn top_p_keeps_minimal_set() {
        // With one dominant token (probability > 0.95) and rest tiny,
        // top_p=0.9 must keep only the dominant one.
        let logits = synth_logits(&[(0, 0.0), (3, 6.0), (7, 0.0)]);
        let s = Sampling { temperature: 1.0, top_k: 0, top_p: 0.9,
            repetition_penalty: 1.0, seed: 7 };
        let mut rng = s.seed;
        for _ in 0..50 {
            let mut l = logits.clone();
            let id = sample_next_token(&mut l, &[], &s, &mut rng);
            assert_eq!(id, 3);
        }
    }

    #[test]
    fn repetition_penalty_discourages_history() {
        // Token 3 has the highest logit, but when it's already in the
        // history the penalty should kick its logit below token 7's.
        // Use top_k=1 to make the test deterministic — whoever has the
        // highest *adjusted* logit wins.
        let logits = synth_logits(&[(3, 4.0), (7, 3.0)]);
        let history = [3u32];
        let s = Sampling { temperature: 0.7, top_k: 1, top_p: 1.0,
            repetition_penalty: 2.0, seed: 0 };
        let mut l = logits.clone();
        let mut rng = 0;
        let id = sample_next_token(&mut l, &history, &s, &mut rng);
        // 4.0 / 2.0 = 2.0 < 3.0  ⇒ token 7 wins.
        assert_eq!(id, 7);
        // Without history, token 3 wins.
        let mut l = logits.clone();
        let mut rng2 = 0;
        let id2 = sample_next_token(&mut l, &[], &s, &mut rng2);
        assert_eq!(id2, 3);
    }
}

/// xorshift64* — small, deterministic, no dependencies. Returns a
/// uniform `f32` in `[0, 1)` and updates the state in place.
fn next_rand_unit(state: &mut u64) -> f32 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15; // golden-ratio splitter, avoid x=0 trap
    }
    x ^= x << 12;
    x ^= x >> 25;
    x ^= x << 27;
    *state = x;
    let m = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    // Take the top 24 bits → uniform in [0, 1) at f32 precision.
    let bits = (m >> 40) as u32; // 24 bits
    (bits as f32) / ((1u32 << 24) as f32)
}

#[derive(Debug)]
pub struct GenerateResult {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub generated_text: String,
    /// Same as `generated_text` but with `<think>...</think>` removed
    /// when `config.think_filter` was set. Otherwise equals `generated_text`.
    pub visible_text: String,
    pub stopped_on_eos: bool,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl GenerateResult {
    pub fn prefill_tok_s(&self) -> f64 {
        let s = self.prefill_time.as_secs_f64();
        if s == 0.0 { 0.0 } else { self.prompt_tokens as f64 / s }
    }
    pub fn decode_tok_s(&self) -> f64 {
        let s = self.decode_time.as_secs_f64();
        if s == 0.0 { 0.0 } else { self.generated_tokens as f64 / s }
    }
}

/// Phase-2D compat wrapper: chat-template, reset KV, prefill+decode,
/// stdout streaming if requested. Does not preserve cross-call
/// state — use [`crate::backend::vulkan::chat::ChatSession`] for
/// multi-turn.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    gguf: &GgufFile,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<GenerateResult, Box<dyn std::error::Error>> {
    let template = ChatTemplate::detect(gguf, tokenizer);
    let prompt_tokens =
        template.render_first_turn(tokenizer, "You are a helpful assistant.", prompt);
    forward.kv_cache.reset();
    if config.print_stream {
        let mut filter = ThinkFilter::new();
        let do_filter = config.think_filter;
        let mut on_token = move |_id: u32, raw: &str| {
            let visible = if do_filter { filter.push(raw) } else { raw.to_string() };
            if !visible.is_empty() {
                print!("{visible}");
                std::io::stdout().flush().ok();
            }
        };
        let mut r = generate_from_tokens(
            forward, dev, registry, cmd_ctx, model, EmbeddingSource::Gguf(gguf),
            cfg, tokenizer, &prompt_tokens, 0, config, false, &mut on_token,
        )?;
        // Flush any tail still buffered in the filter.
        // We can't borrow the closure-local filter back here, so
        // post-process the raw text instead — same result.
        if config.think_filter {
            r.visible_text = ThinkFilter::strip_all(&r.generated_text);
        }
        println!();
        Ok(r)
    } else {
        let mut on_token = |_: u32, _: &str| {};
        generate_from_tokens(
            forward, dev, registry, cmd_ctx, model, EmbeddingSource::Gguf(gguf),
            cfg, tokenizer, &prompt_tokens, 0, config, false, &mut on_token,
        )
    }
}

/// Sprint 20-M3 — abstract source for token-embedding rows. GGUF
/// models dequantize on the fly from the file mmap; SafeTensors FP8
/// models keep an FP32 host cache (BF16 → FP32 expanded at load
/// time) because the GPU buffer isn't host-readable.
pub enum EmbeddingSource<'a> {
    Gguf(&'a GgufFile),
    /// Pre-expanded FP32 vocab × hidden_dim, row-major.
    Host(&'a [f32]),
}

fn embed_lookup(
    src: &EmbeddingSource<'_>,
    cfg: &ModelConfig,
    tid: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match src {
        EmbeddingSource::Gguf(gguf) => embedding_row(gguf, cfg, tid),
        EmbeddingSource::Host(cache) => {
            let hidden = cfg.hidden_dim as usize;
            let start = (tid as usize) * hidden;
            if start + hidden > cache.len() {
                return Err(format!("token_id {tid} out of range").into());
            }
            Ok(cache[start..start + hidden].to_vec())
        }
    }
}

/// Lower-level driver: prefills `prefill_tokens` starting at
/// `start_pos` (does NOT reset the KV cache), then runs greedy decode
/// up to `config.max_tokens` or EOS. Streams each generated token
/// through `on_token(id, decoded_text)`.
///
/// `force_per_token_prefill = true` (Sprint 20-M3) bypasses
/// `prefill_batch` and feeds each prompt token through `forward_token`
/// instead. Used by SafeTensors FP8 models, which don't (yet) ship
/// an FP8 GEMM prefill kernel — the per-token GEMV path works because
/// `run_gemv` is already FP8-aware.
///
/// Returns the elapsed prefill / decode timings and the raw +
/// think-filtered concatenated text.
#[allow(clippy::too_many_arguments)]
pub fn generate_from_tokens(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    embed_src: EmbeddingSource<'_>,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    prefill_tokens: &[u32],
    start_pos: u32,
    config: &GenerateConfig,
    force_per_token_prefill: bool,
    on_token: &mut dyn FnMut(u32, &str),
) -> Result<GenerateResult, Box<dyn std::error::Error>> {
    let max_seq = forward.kv_cache.config.max_seq_len;
    let prefill_end = start_pos as u64 + prefill_tokens.len() as u64;
    if prefill_end + config.max_tokens as u64 > max_seq as u64 {
        return Err(format!(
            "prefill {} from pos {} + max_tokens {} > max_seq_len {}",
            prefill_tokens.len(),
            start_pos,
            config.max_tokens,
            max_seq,
        )
        .into());
    }

    // Sprint G-4 — per-shader GPU-time accumulator. Enabled by
    // `VF_GPU_TIMER=1` plus a ShaderProfiler installed at Forward::new.
    // Aggregates per-label totals across every forward_token call in
    // both prefill (per-token path) and decode, then prints a sorted
    // breakdown at the end. Zero overhead when the env-gate is off
    // (Forward's profile() helper short-circuits on profiler=None).
    let gpu_timer = std::env::var("VF_GPU_TIMER").map(|v| v == "1").unwrap_or(false);
    let mut acc_per_shader: BTreeMap<String, (Duration, u32)> = BTreeMap::new();
    let mut acc_per_layer_decode: Vec<Vec<Duration>> = Vec::new();
    let mut decode_tokens_profiled: u32 = 0;
    let mut prefill_tokens_profiled: u32 = 0;

    // Sprint G-7 — CPU-side per-stage breakdown. Aggregates reset, begin,
    // record, end, submit, gpu_wait (from `one_shot_profiled`) plus the
    // readback time (logits_staging.read_bytes()) plus a wrap-time
    // (everything else in the decode-loop iteration) so we can localise
    // exactly where the 8-17 % CPU/idle wall-gap sits.
    let cpu_timer = std::env::var("VF_CPU_TIMER").map(|v| v == "1").unwrap_or(false);
    let mut acc_reset = Duration::ZERO;
    let mut acc_begin = Duration::ZERO;
    let mut acc_record = Duration::ZERO;
    let mut acc_end = Duration::ZERO;
    let mut acc_submit = Duration::ZERO;
    let mut acc_wait = Duration::ZERO;
    let mut acc_readback = Duration::ZERO;
    let mut acc_other = Duration::ZERO;
    // Graph-analysis Teil 3 — barrier-count delta per decode token.
    // VF emits 361 (Qwen3-8B) → 1840 (Gemma-4-26B-A4B) barriers/tok,
    // versus llama.cpp's ~50-70/tok (byte-range memory-overlap tracking).
    // Surfacing the delta in VF_CPU_TIMER's printout lets future sprints
    // verify barrier-reduction work (e.g. G-8 byte-range tracking) by
    // re-running this bench.
    let barrier_baseline = if cpu_timer { Some(forward.barrier_stats()) } else { None };
    let mut acc_barriers_issued: u64 = 0;
    let mut acc_barriers_checked: u64 = 0;

    // ---- Prefill ----
    let prefill_start = Instant::now();
    let mut pos = start_pos;
    let prefill_len = prefill_tokens.len() as u32;
    if prefill_len > 0 {
        // Sprint 5B — chunked batched prefill. Slice the prefill into
        // pieces of `max_prefill_tokens` and dispatch each piece
        // through `prefill_batch` with a bumped `base_pos`. Replaces
        // the prior token-by-token fallback that collapsed long
        // prompts to decode rate (~90 tok/s). prefill_batch already
        // takes `base_pos` and uses it for (a) RoPE positions, (b) KV
        // write offsets via `pos_offset_bytes`, (c) flash_attn_batch
        // bounds (q_start = base_pos, n_kv = base_pos + seq_len), so
        // the second-and-later chunks see the prior chunks' KV
        // entries automatically. The two_tiles regression test
        // already covers the multi-tile-within-one-prefill case;
        // chunked prefill is the same shape extended across multiple
        // prefill_batch submits.
        // Sprint F.2a — MTP shadow-pass: when VF_MTP_DRAFT, also run the
        // draft block at each prompt position so block-64's (cold) KV is
        // populated over the FULL context (prompt + decode), not just the
        // decode positions. Otherwise the decode-phase match-rate is
        // confounded by missing prompt-position block-64 KV (§1b / Phase 2).
        let mtp_shadow = std::env::var("VF_MTP_DRAFT").as_deref() == Ok("1");
        // Probe (gated, default-off, zero production cost): log which
        // prefill dispatch path this run takes. Resolves the qwen35
        // "BatchExec vs per-token decode" routing question empirically —
        // chat.rs forces `force_per_token_prefill=true` for qwen35, so
        // qwen35 prefill takes the per-token forward_token loop here and
        // never reaches prefill_batch/BatchExec (the b_step_* token-0
        // stubs are dormant for this path).
        let dbg_prefill_path =
            std::env::var("VF_DEBUG_PREFILL_PATH").as_deref() == Ok("1");
        if force_per_token_prefill {
            if dbg_prefill_path {
                eprintln!(
                    "[VF_DEBUG_PREFILL_PATH] per-token DECODE prefill loop \
                     (force_per_token_prefill=true): {prefill_len} tokens via \
                     forward_token — prefill_batch/BatchExec/b_step_* bypassed",
                );
            }
            // Sprint 20-M3 — SafeTensors FP8 models reach this branch:
            // batched prefill would hit the (FP16) `mul_mm.comp` GEMM
            // which has no `DATA_A_FP8` variant, so we fall back to
            // the per-token GEMV path. That's slow at long pp but
            // proves the end-to-end pipeline; an FP8 GEMM port is
            // future work (Sprint 19A-style coopmat coverage).
            // Diagnostic (gated, default-off): dump the per-token DECODE
            // oracle's logits at every prefill position so they can be
            // diffed against BatchExec's per-position output (VF_BATCH_DUMP_POS).
            let per_tok_logits =
                std::env::var("VF_PREFILL_PER_TOKEN_LOGITS").as_deref() == Ok("1");
            for (i, &tid) in prefill_tokens.iter().enumerate() {
                let embd = embed_lookup(&embed_src, cfg, tid)?;
                let stats = forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos, tid)?;
                if per_tok_logits {
                    if let Ok(l) = forward.logits() {
                        let mut amax = 0usize;
                        let mut mv = f32::NEG_INFINITY;
                        for (j, &v) in l.iter().enumerate() {
                            if v > mv { mv = v; amax = j; }
                        }
                        eprintln!(
                            "[ORACLE_PREFILL] pos={i} argmax_idx={amax} max={mv:.4}",
                        );
                    }
                }
                if gpu_timer {
                    for (k, (d, n)) in stats.per_shader {
                        let e = acc_per_shader.entry(k).or_insert((Duration::ZERO, 0));
                        e.0 += d;
                        e.1 += n;
                    }
                    prefill_tokens_profiled += 1;
                }
                // Shadow-pass: write block-64 KV at pos+1 using h_pos +
                // the (known) next prompt token. Skipped for the LAST
                // prompt token (no next; and its trunk logits must survive
                // to seed the first decode sample — the shadow-pass would
                // overwrite logits_buf).
                if mtp_shadow {
                    if let Some(&next_tid) = prefill_tokens.get(i + 1) {
                        let e = embed_lookup(&embed_src, cfg, next_tid)?;
                        let _ = forward
                            .mtp_draft_logits(dev, registry, cmd_ctx, model, &e, pos + 1)?;
                    }
                }
                pos += 1;
            }
        } else {
            let chunk_size = forward.max_prefill_tokens.max(1) as usize;
            if dbg_prefill_path {
                eprintln!(
                    "[VF_DEBUG_PREFILL_PATH] batched prefill_batch/BatchExec \
                     path: {prefill_len} tokens in chunks of {chunk_size}",
                );
            }
            for chunk in prefill_tokens.chunks(chunk_size) {
                let chunk_len = chunk.len() as u32;
                let mut chunk_embeds: Vec<f32> =
                    Vec::with_capacity(chunk.len() * cfg.hidden_dim as usize);
                for &tid in chunk {
                    chunk_embeds.extend(embed_lookup(&embed_src, cfg, tid)?);
                }
                forward.prefill_batch(
                    dev, registry, cmd_ctx, model, &chunk_embeds, chunk_len, pos,
                    chunk,
                )?;
                pos += chunk_len;
            }
        }
    }
    let mut last_logits = forward.logits()?;
    let prefill_time = prefill_start.elapsed();

    // ---- Sprint MTP-Orch S3 GATE: partial-accept reconciliation bit-ident ----
    // Risk-front-loaded proof (VF_MTP_RECONCILE_TEST=1; needs VF_MTP=1 for the
    // recurrent-state snapshot buffers). Forces M=0 (full reject), M=1 (PARTIAL
    // accept — the bug-class trigger) and M=2 (full accept) for an n=2 verify
    // and asserts the post-reconcile recurrent state (ssm_state + conv_state)
    // AND the KV-len counter are bit-identical (rel=0) to a plain decode of
    // exactly the accepted tokens. Reconcile method = snapshot-pre-verify +
    // restore + replay-accepted via the verified decode path (the brief's
    // sanctioned "restore-pre-draft + d_1..d_M nachfahren"; the per-position
    // hidden the next draft needs is re-derived by the replay's mtp_h_buf hook).
    // Diagnostic only: runs once after prefill, then early-returns.
    if std::env::var("VF_MTP_RECONCILE_TEST").as_deref() == Ok("1") {
        let ok = run_mtp_reconcile_gate(
            forward, dev, registry, cmd_ctx, model, &embed_src, cfg, &last_logits, pos,
        )?;
        eprintln!(
            "[MTP_RECONCILE] VERDICT: {}",
            if ok {
                "PASS — partial-accept reconciliation bit-ident (rel=0) → S3 gate cleared"
            } else {
                "FAIL — reconciliation NOT bit-ident → STOP, do not build the loop on this glue"
            }
        );
        return Ok(GenerateResult {
            prompt_tokens: prefill_tokens.len(),
            generated_tokens: 0,
            generated_text: String::new(),
            visible_text: String::new(),
            stopped_on_eos: true,
            prefill_time,
            decode_time: Duration::ZERO,
        });
    }

    // ---- Sprint MTP-Orch S4: self-speculative decode loop (gated VF_MTP) ----
    // Draft n tokens with the nextn head → verify in ONE batched forward
    // (the v0.5.6 deterministic batched-prefill) → accept the longest
    // argmax-matching prefix +1 → reconcile recurrent state. Output is
    // byte-identical to plain greedy decode by construction (accept only
    // tokens equal to the verify argmax). Default-off; the bit-ident S3
    // gate (VF_MTP_RECONCILE_TEST) proved the partial-accept reconciliation.
    let mtp_enabled = cfg.qwen35.is_some()
        && std::env::var("VF_MTP").as_deref() == Ok("1")
        && std::env::var("VF_MTP_RECONCILE_TEST").as_deref() != Ok("1")
        && forward.mtp_snapshot_ready();
    if mtp_enabled {
        let decode_start = Instant::now();
        let (generated, stopped_on_eos, gen_text, visible) = run_mtp_decode_loop(
            forward, dev, registry, cmd_ctx, model, &embed_src, cfg, tokenizer,
            config, &last_logits, pos, on_token,
        )?;
        let decode_time = decode_start.elapsed();
        return Ok(GenerateResult {
            prompt_tokens: prefill_tokens.len(),
            generated_tokens: generated.len(),
            generated_text: gen_text,
            visible_text: visible,
            stopped_on_eos,
            prefill_time,
            decode_time,
        });
    }

    // ---- Decode ----
    let decode_start = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut stopped_on_eos = false;
    // Phase 6 v0.1.2 sampling RNG seed. The greedy path (default,
    // temperature=0) never reads it, so existing tests stay
    // byte-deterministic; non-greedy callers seed via
    // `config.sampling.seed`.
    let mut rng_state = config.sampling.seed.wrapping_add(start_pos as u64);

    // Sprint 16B — UTF-8 stream buffer. A single token can encode just
    // a fragment of a multi-byte codepoint (e.g. one of the four bytes
    // of an emoji), so we accumulate raw token bytes here and emit only
    // the valid-UTF-8 prefix on each step. The trailing partial bytes
    // stay in the buffer until the next token completes the codepoint;
    // any unfinished tail at end-of-stream is lossy-flushed.
    let mut utf8_buf: Vec<u8> = Vec::new();
    let emit = |id: u32, on_token: &mut dyn FnMut(u32, &str), buf: &mut Vec<u8>, bytes: &[u8]| {
        buf.extend_from_slice(bytes);
        let valid_up_to = match std::str::from_utf8(buf) {
            Ok(_) => buf.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_up_to > 0 {
            // SAFETY: from_utf8's valid_up_to() guarantees the prefix is valid UTF-8.
            let s = unsafe { std::str::from_utf8_unchecked(&buf[..valid_up_to]) };
            on_token(id, s);
            buf.drain(..valid_up_to);
        }
    };

    // Sprint 15E — async pipelined decode loop, default-on. Hides the
    // ~1836 µs CPU recording phase inside the ~9034 µs GPU compute
    // window of the previous token by alternating two CB+fence pairs:
    //   stage 1: pre_record(slot=cur, pos)   [during GPU(prev)]
    //   stage 2: wait(prev) + read logits + sample + embedding lookup
    //   stage 3: fill_embed + submit(cur)    [GPU(cur) starts]
    //   slot rotates 0/1, repeat
    // Opt-out: VULKANFORGE_DISABLE_ASYNC_DECODE=1 falls back to the
    // serial path (single CB, record+submit+wait per token).
    //
    // Sprint 56C-3 — async safe again for MoE when the GPU-direct
    // expert FFN is active. Sprint 56C-2 eliminated the
    // `mid_frame_submit_and_wait` from `step_moe_route` (no more CPU
    // readback of routing decisions), so the Sprint 54I stale-scratch_a
    // race condition cannot fire. The legacy CPU-readback path is
    // still selectable via `VF_GPU_DIRECT_MOE=0` — in that case async
    // remains disabled for MoE to preserve the 54I workaround.
    // Sprint G-2j — Qwen3.6 originally had a Heisenberg sync bug between
    // async decode pipeline stages (gibberish at q_idx > 0). G-2g→G-2i
    // landed the barrier fixes and G-2j confirmed the path was coherent
    // again; the workaround was kept while G-3 reported no async
    // speedup. Sprint G-7 — measured 18.3 → 20.5 tok/s (+12 %) for
    // Qwen3.6 after G-6's redundant-trailing-barrier strip; the 8-shaped
    // dispatch chain now actually overlaps token N+1's CPU record with
    // token N's GPU wait. Re-enabled async-decode default-on. Opt-out
    // via `VULKANFORGE_DISABLE_ASYNC_DECODE=1` retained as escape hatch.
    // Sprint F.1 — the MTP Phase-4 rollback self-test injects a
    // throwaway forward + state restore per token, which only the
    // serial path supports; force serial when it is active.
    let mtp_rollback_test = std::env::var("VF_MTP_ROLLBACK_TEST").as_deref() == Ok("1");
    // Sprint F.2a — the MTP draft-head match-rate harness runs the draft
    // between trunk tokens (serial path only); force serial when active.
    let mtp_draft_test = std::env::var("VF_MTP_DRAFT").as_deref() == Ok("1");
    let async_decode = if mtp_rollback_test || mtp_draft_test {
        false
    } else {
        match std::env::var("VULKANFORGE_DISABLE_ASYNC_DECODE") {
            Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
            Err(_) => {
                cfg.gemma4
                    .as_ref()
                    .map(|g| !g.enable_moe_block || gpu_direct_moe_enabled())
                    .unwrap_or(true)
            }
        }
    };

    // Sprint F.2a — MTP draft-head match-rate accumulators (VF_MTP_DRAFT).
    let mut mtp_pending: Option<u32> = None;
    let mut mtp_matches: u64 = 0;
    let mut mtp_total: u64 = 0;
    let mut mtp_history: Vec<bool> = Vec::new();

    if async_decode {
        // ---- Async 3-stage pipeline ----
        // First decode token: cold start the pipe.
        let first_id = sample_next_token(
            &mut last_logits, &generated, &config.sampling, &mut rng_state,
        );
        if tokenizer.is_eos(first_id) {
            stopped_on_eos = true;
        } else if generated.len() < config.max_tokens as usize && pos < max_seq {
            let bytes = tokenizer.decode_token_bytes(first_id);
            emit(first_id, on_token, &mut utf8_buf, &bytes);
            generated.push(first_id);

            let embd = embed_lookup(&embed_src, cfg, first_id)?;
            forward.pre_record(dev, registry, model, 0, pos)?;
            forward.fill_embed_and_submit(dev, 0, &embd, pos, model, first_id)?;
            let mut cur_slot = 1usize;
            pos += 1;

            loop {
                if generated.len() >= config.max_tokens as usize || pos >= max_seq {
                    break;
                }
                // v0.4 Sprint 3 — cooperative cancel. Treated as
                // soft-EOS so the caller still gets a finish_reason
                // and the token sequence written so far is intact.
                if config
                    .cancel_token
                    .as_ref()
                    .is_some_and(|t| t.load(std::sync::atomic::Ordering::Acquire))
                {
                    stopped_on_eos = true;
                    break;
                }
                let prev_slot = 1 - cur_slot;

                // Stage 1: pre-record the current slot's CB. References
                // slots[cur_slot] handles only — embedding contents are
                // written below, after sampling.
                forward.pre_record(dev, registry, model, cur_slot, pos)?;

                // Stage 2: wait for previous GPU work, read logits, sample.
                last_logits = forward.wait_and_read_logits(dev, prev_slot, model)?;
                let next_id = sample_next_token(
                    &mut last_logits, &generated, &config.sampling, &mut rng_state,
                );
                if tokenizer.is_eos(next_id) {
                    stopped_on_eos = true;
                    // The pre-recorded CB at cur_slot is left unsubmitted;
                    // it'll be reset on the next session's pre_record call.
                    break;
                }
                let bytes = tokenizer.decode_token_bytes(next_id);
                emit(next_id, on_token, &mut utf8_buf, &bytes);
                generated.push(next_id);

                // Stage 3: write embedding + submit.
                let embd = embed_lookup(&embed_src, cfg, next_id)?;
                forward.fill_embed_and_submit(dev, cur_slot, &embd, pos, model, next_id)?;

                cur_slot = 1 - cur_slot;
                pos += 1;
            }

            // Drain: read logits from the last submitted slot.
            let last_slot = 1 - cur_slot;
            last_logits = forward.wait_and_read_logits(dev, last_slot, model)?;
        }
    } else {
        // ---- Serial path (pre-15E behaviour) ----
        loop {
            // v0.4 Sprint 3 — cooperative cancel (same shape as the
            // async path above). Set before the next sampling so a
            // disconnect on token N never gets the wasted compute
            // for token N+1.
            if config
                .cancel_token
                .as_ref()
                .is_some_and(|t| t.load(std::sync::atomic::Ordering::Acquire))
            {
                stopped_on_eos = true;
                break;
            }
            let next_id = sample_next_token(
                &mut last_logits, &generated, &config.sampling, &mut rng_state,
            );
            if tokenizer.is_eos(next_id) {
                stopped_on_eos = true;
                break;
            }
            if generated.len() >= config.max_tokens as usize || pos >= max_seq {
                break;
            }
            let bytes = tokenizer.decode_token_bytes(next_id);
            emit(next_id, on_token, &mut utf8_buf, &bytes);
            generated.push(next_id);

            // Sprint F.1 — MTP Phase-4 rollback self-test. Snapshot the
            // recurrent state, run a throwaway forward with a DIFFERENT
            // token (genuinely advances + corrupts the GDN/conv state and
            // writes garbage KV at `pos`), then restore. If snapshot/
            // restore is complete, the generated stream stays
            // token-identical to a normal serial run — the proof that the
            // R3 recurrent-state rollback mechanism is correct.
            if mtp_rollback_test && forward.mtp_snapshot_ready() {
                // Diagnostics: VF_MTP_RB_NO_THROW skips the throwaway
                // forward (snapshot+restore only → isolates the copy
                // mechanism). VF_MTP_RB_SAMETOK uses next_id as the
                // throwaway token (state advance matches the real one →
                // isolates mechanism drift from un-restored state).
                let no_throw = std::env::var("VF_MTP_RB_NO_THROW").as_deref() == Ok("1");
                let sametok = std::env::var("VF_MTP_RB_SAMETOK").as_deref() == Ok("1");
                let hash_dbg = std::env::var("VF_MTP_RB_HASH").as_deref() == Ok("1");
                let saved_seq = forward.kv_seq_len();
                forward.snapshot_recurrent_state(dev, cmd_ctx)?;
                let h_snap = if hash_dbg { forward.mtp_debug_ssm_hash(dev, cmd_ctx)? } else { 0 };
                if !no_throw {
                    let garbage_id = if sametok {
                        next_id
                    } else if next_id == 0 {
                        1
                    } else {
                        next_id - 1
                    };
                    let garbage_embd = embed_lookup(&embed_src, cfg, garbage_id)?;
                    forward.forward_token(
                        dev, registry, cmd_ctx, model, &garbage_embd, pos, garbage_id,
                    )?;
                }
                let h_corrupt = if hash_dbg { forward.mtp_debug_ssm_hash(dev, cmd_ctx)? } else { 0 };
                forward.restore_recurrent_state(dev, cmd_ctx)?;
                let h_restore = if hash_dbg { forward.mtp_debug_ssm_hash(dev, cmd_ctx)? } else { 0 };
                forward.set_kv_seq_len(saved_seq);
                if hash_dbg {
                    eprintln!(
                        "[MTP_RB] pos={pos} h_snap={h_snap:#018x} h_corrupt={h_corrupt:#018x} \
                         h_restore={h_restore:#018x} corrupt_changed={} restore_ok={}",
                        h_corrupt != h_snap,
                        h_restore == h_snap,
                    );
                }
            }
            let embd = embed_lookup(&embed_src, cfg, next_id)?;
            let iter_start = if cpu_timer { Some(Instant::now()) } else { None };
            let stats = forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos, next_id)?;
            if gpu_timer || cpu_timer {
                decode_tokens_profiled += 1;
            }
            if gpu_timer {
                for (k, (d, n)) in stats.per_shader {
                    let e = acc_per_shader.entry(k).or_insert((Duration::ZERO, 0));
                    e.0 += d;
                    e.1 += n;
                }
                if !stats.per_layer.is_empty() {
                    acc_per_layer_decode.push(stats.per_layer);
                }
            }
            if cpu_timer {
                if let Some(t) = forward.last_one_shot_timings.take() {
                    acc_reset += t.reset;
                    acc_begin += t.begin;
                    acc_record += t.record;
                    acc_end += t.end;
                    acc_submit += t.submit;
                    acc_wait += t.wait;
                }
                if let Some(rb) = forward.last_readback_time.take() {
                    acc_readback += rb;
                }
                // iter_total — full forward_token call (reset+begin+record+
                // end+submit+wait+readback) plus whatever Rust glue runs
                // between iter_start and here. Difference vs the sum is
                // the "between" wrap time.
                if let Some(s) = iter_start {
                    acc_other += s.elapsed();
                }
            }
            last_logits = forward.logits()?;
            // Sprint F.2a — MTP draft-head match-rate. Verify the previous
            // draft's prediction against the trunk's greedy argmax, then
            // compute the next draft from h_t (captured during the forward
            // above) + the trunk's predicted token. argmax is NaN-safe.
            if mtp_draft_test {
                let argmax = |v: &[f32]| -> u32 {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| {
                            a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                };
                let trunk_argmax = argmax(&last_logits);
                if let Some(p) = mtp_pending.take() {
                    mtp_total += 1;
                    let hit = p == trunk_argmax;
                    if hit {
                        mtp_matches += 1;
                    }
                    mtp_history.push(hit);
                }
                let e = embed_lookup(&embed_src, cfg, trunk_argmax)?;
                let dl =
                    forward.mtp_draft_logits(dev, registry, cmd_ctx, model, &e, pos + 1)?;
                mtp_pending = Some(argmax(&dl));
            }
            pos += 1;
        }
    }
    let _ = last_logits;
    if mtp_draft_test {
        let rate = if mtp_total > 0 {
            100.0 * mtp_matches as f64 / mtp_total as f64
        } else {
            0.0
        };
        eprintln!(
            "[MTP_DRAFT] n=1 standalone match-rate: {mtp_matches}/{mtp_total} = {rate:.1}% \
             (draft.argmax == trunk next token; decode-incremental block-64 KV)"
        );
        // Windowed: does the rate climb as block-64 decode-KV accumulates?
        // (climb → KV-warmup/prompt-gap; flat → wiring/quant.)
        let n = mtp_history.len();
        if n >= 8 {
            let q = n / 4;
            let win = |lo: usize, hi: usize| -> String {
                let s: usize = mtp_history[lo..hi].iter().filter(|&&b| b).count();
                let t = hi - lo;
                format!("{s}/{t}={:.0}%", 100.0 * s as f64 / t.max(1) as f64)
            };
            eprintln!(
                "[MTP_DRAFT] quartiles: Q1 {} | Q2 {} | Q3 {} | Q4 {}",
                win(0, q), win(q, 2 * q), win(2 * q, 3 * q), win(3 * q, n),
            );
        }
    }
    // Drain the UTF-8 buffer: any trailing partial bytes get a lossy
    // flush (the model produced an incomplete codepoint at EOS / max
    // tokens — surface it as U+FFFD rather than dropping bytes).
    if !utf8_buf.is_empty() {
        let last_id = generated.last().copied().unwrap_or(0);
        let lossy = String::from_utf8_lossy(&utf8_buf).into_owned();
        on_token(last_id, &lossy);
        utf8_buf.clear();
    }
    let decode_time = decode_start.elapsed();

    // Sprint G-4 — print accumulated per-shader GPU-time breakdown.
    if gpu_timer && (decode_tokens_profiled > 0 || prefill_tokens_profiled > 0) {
        let total_tokens = decode_tokens_profiled + prefill_tokens_profiled;
        eprintln!();
        eprintln!("=== VF_GPU_TIMER per-shader breakdown ===");
        eprintln!("(prefill_tok={prefill_tokens_profiled}, decode_tok={decode_tokens_profiled}; values are TOTAL across all profiled forward_token calls)");
        let mut rows: Vec<(String, Duration, u32)> = acc_per_shader
            .iter()
            .map(|(k, (d, n))| (k.clone(), *d, *n))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let total_ns: u128 = rows.iter().map(|r| r.1.as_nanos()).sum();
        let total_ms = total_ns as f64 / 1e6;
        let per_tok_ms = if total_tokens > 0 {
            total_ms / total_tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "  {:<32} {:>10} {:>10} {:>8}   {:>10}",
            "shader", "total ms", "calls", "%", "ms/tok"
        );
        eprintln!("  {}", "-".repeat(78));
        for (name, dur, calls) in &rows {
            let ms = dur.as_nanos() as f64 / 1e6;
            let pct = if total_ns > 0 {
                100.0 * (dur.as_nanos() as f64) / (total_ns as f64)
            } else {
                0.0
            };
            let mstok = if total_tokens > 0 {
                ms / total_tokens as f64
            } else {
                0.0
            };
            eprintln!(
                "  {:<32} {:>10.3} {:>10} {:>7.2}%   {:>10.3}",
                name, ms, calls, pct, mstok
            );
        }
        eprintln!("  {}", "-".repeat(78));
        eprintln!(
            "  {:<32} {:>10.3} ms ({:>5.2} ms/tok)",
            "TOTAL GPU TIME", total_ms, per_tok_ms
        );
        let wall_decode_ms = decode_time.as_nanos() as f64 / 1e6;
        if decode_tokens_profiled > 0 {
            let wall_per_tok = wall_decode_ms / decode_tokens_profiled as f64;
            eprintln!(
                "  {:<32} {:>10.3} ms  ({:>5.2} ms/tok decode wall)",
                "WALL DECODE", wall_decode_ms, wall_per_tok
            );
            let gpu_decode_ns: u128 = acc_per_layer_decode
                .iter()
                .flat_map(|v| v.iter().map(|d| d.as_nanos()))
                .sum::<u128>();
            let gpu_decode_ms = gpu_decode_ns as f64 / 1e6;
            let gpu_per_tok = if decode_tokens_profiled > 0 {
                gpu_decode_ms / decode_tokens_profiled as f64
            } else {
                0.0
            };
            eprintln!(
                "  {:<32} {:>10.3} ms  ({:>5.2} ms/tok per_layer sum, decode-only)",
                "GPU DECODE (per_layer sum)", gpu_decode_ms, gpu_per_tok
            );
            if wall_decode_ms > 0.0 {
                let busy_pct = 100.0 * gpu_decode_ms / wall_decode_ms;
                eprintln!(
                    "  GPU-busy fraction (decode wall): {:.1}%   CPU/idle gap: {:.1}%",
                    busy_pct,
                    100.0 - busy_pct,
                );
            }
        }
        eprintln!();
    }

    // Sprint G-7 — CPU-side per-stage breakdown printout.
    if cpu_timer && decode_tokens_profiled > 0 {
        // Snapshot barrier-stats delta over the whole decode loop.
        if let Some((checked0, issued0)) = barrier_baseline {
            let (checked1, issued1) = forward.barrier_stats();
            acc_barriers_checked = checked1.saturating_sub(checked0);
            acc_barriers_issued = issued1.saturating_sub(issued0);
        }
        let n = decode_tokens_profiled as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let total = acc_reset + acc_begin + acc_record + acc_end
            + acc_submit + acc_wait + acc_readback;
        let between = if acc_other > total { acc_other - total } else { Duration::ZERO };
        eprintln!();
        eprintln!("=== VF_CPU_TIMER per-stage breakdown ===");
        eprintln!("(decode_tok={decode_tokens_profiled}; values are TOTAL ms / cumulative + per-token ms/tok)");
        eprintln!("  {:<14} {:>12} {:>14}", "stage", "total ms", "ms/tok");
        eprintln!("  {}", "-".repeat(46));
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "reset",    ms(acc_reset),    ms(acc_reset)    / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "begin",    ms(acc_begin),    ms(acc_begin)    / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "record",   ms(acc_record),   ms(acc_record)   / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "end",      ms(acc_end),      ms(acc_end)      / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "submit",   ms(acc_submit),   ms(acc_submit)   / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "GPU wait", ms(acc_wait),     ms(acc_wait)     / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "readback", ms(acc_readback), ms(acc_readback) / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "between",  ms(between),      ms(between)      / n);
        eprintln!("  {}", "-".repeat(46));
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "CPU TOTAL (excl GPU wait)",
            ms(total - acc_wait + between), (ms(total - acc_wait + between)) / n);
        eprintln!("  {:<14} {:>12.3} {:>14.4}", "ITER TOTAL", ms(acc_other), ms(acc_other) / n);
        eprintln!();
        eprintln!("  Async-decode potential overlap:");
        eprintln!("    - GPU wait ({:.2} ms/tok) hides record+submit+end of next token", ms(acc_wait) / n);
        eprintln!("    - readback ({:.2} ms/tok) is post-wait CPU work; could hide if N+1 pre-records", ms(acc_readback) / n);
        eprintln!();
        eprintln!("  Barrier-elision counters (delta over decode loop):");
        eprintln!("    - checked: {:>8}  ({:>7.1} /tok)",
            acc_barriers_checked, acc_barriers_checked as f64 / n);
        eprintln!("    - issued : {:>8}  ({:>7.1} /tok)  ({:.1}% elided)",
            acc_barriers_issued, acc_barriers_issued as f64 / n,
            100.0 * (acc_barriers_checked.saturating_sub(acc_barriers_issued)) as f64
                / acc_barriers_checked.max(1) as f64);
        eprintln!();
    }

    let generated_text = tokenizer.decode(&generated);
    let visible_text = if config.think_filter {
        ThinkFilter::strip_all(&generated_text)
    } else {
        generated_text.clone()
    };

    Ok(GenerateResult {
        prompt_tokens: prefill_tokens.len(),
        generated_tokens: generated.len(),
        generated_text,
        visible_text,
        stopped_on_eos,
        prefill_time,
        decode_time,
    })
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

/// Sprint MTP-Orch S3 GATE — prove the MTP partial-accept rollback
/// reconciliation is bit-identical to a plain decode of the accepted
/// tokens. This is the make-or-break correctness risk: a reconcile that
/// restores the pre-draft state but fails to re-advance to `S_{t+M}`
/// leaks a stale recurrent state on partial accept — INVISIBLE to full
/// accept (M=n, no restore) and full reject (M=0, trivial), so it MUST be
/// demonstrated at `0 < M < n`.
///
/// Method (the brief's sanctioned "restore-pre-draft + replay accepted"):
///   1. snapshot `S_p` (the pre-verify recurrent state).
///   2. reference = plain decode of `x, a_0, a_1` from `S_p`, hashing the
///      recurrent state after each step → `S_{p+1}, S_{p+2}, S_{p+3}`.
///   3. for each forced M ∈ {0,1,2}: construct drafts that yield exactly
///      that M, restore `S_p`, run the batched verify (`prefill_batch` over
///      `[x,d_1,d_2]` — the v0.5.6 deterministic path), read per-position
///      argmax (`mtp_verify_argmax`), accept (`d_i == v_{i-1}`), then
///      reconcile = restore `S_p` + replay the M+1 accepted real tokens via
///      the decode path, and assert the resulting `(ssm,conv)` hashes + KV
///      counter equal the matching reference.
///   4. "teeth" control on M=1: the BUGGY reconcile (restore `S_p`, rewind
///      counter to `p+2`, NO replay) MUST mismatch `S_{p+2}` — proving the
///      gate would catch the bug-class.
#[allow(clippy::too_many_arguments)]
fn run_mtp_reconcile_gate(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    embed_src: &EmbeddingSource<'_>,
    cfg: &ModelConfig,
    last_logits: &[f32],
    p: u32,
) -> Result<bool, Box<dyn std::error::Error>> {
    if !forward.mtp_snapshot_ready() {
        eprintln!(
            "[MTP_RECONCILE] SKIP: recurrent-state snapshot buffers not allocated. \
             Re-run with VF_MTP=1 (qwen35 only)."
        );
        return Ok(false);
    }
    let vocab = cfg.vocab_size;
    let other = |t: u32| -> u32 { if vocab > 1 { (t + 1) % vocab } else { 0 } };
    let x = argmax(last_logits) as u32;
    eprintln!(
        "\n=== MTP partial-accept reconciliation gate (n=2, start pos p={p}, x={x}) ==="
    );

    let decode_one = |fwd: &mut Forward, tok: u32, pos: u32|
        -> Result<u32, Box<dyn std::error::Error>> {
        let e = embed_lookup(embed_src, cfg, tok)?;
        fwd.forward_token(dev, registry, cmd_ctx, model, &e, pos, tok)?;
        Ok(argmax(&fwd.logits()?) as u32)
    };

    // --- Reference: plain decode x, a0, a1 from S_p, hashing each state. ---
    forward.snapshot_recurrent_state(dev, cmd_ctx)?; // snapshot buffers ← S_p
    let h_sp = forward.mtp_state_hashes(dev, cmd_ctx)?;
    let a0 = decode_one(forward, x, p)?;
    let ref1 = forward.mtp_state_hashes(dev, cmd_ctx)?; // S_{p+1}
    let ref1_kv = forward.kv_seq_len();
    let a1 = decode_one(forward, a0, p + 1)?;
    let ref2 = forward.mtp_state_hashes(dev, cmd_ctx)?; // S_{p+2}
    let ref2_kv = forward.kv_seq_len();
    let a2 = decode_one(forward, a1, p + 2)?;
    let ref3 = forward.mtp_state_hashes(dev, cmd_ctx)?; // S_{p+3}
    let ref3_kv = forward.kv_seq_len();
    eprintln!(
        "  reference decode: a0={a0} a1={a1} a2={a2}; \
         S_p ssm={:#018x} conv={:#018x}",
        h_sp.0, h_sp.1
    );

    let replay_toks = [x, a0, a1];
    let mut all_ok = true;
    for target in 0u32..=2 {
        // Construct drafts (d1,d2) that yield exactly this M.
        let (d1, d2) = match target {
            0 => (other(a0), other(a1)),         // d1 != a0 → reject immediately
            1 => (a0, other(a1)),                // d1 matches, d2 rejected
            _ => (a0, a1),                        // both match
        };
        // Restore S_p, run batched verify of [x, d1, d2].
        forward.restore_recurrent_state(dev, cmd_ctx)?;
        forward.set_kv_seq_len(p);
        let mut embeds: Vec<f32> = Vec::with_capacity(3 * cfg.hidden_dim as usize);
        for &t in &[x, d1, d2] {
            embeds.extend(embed_lookup(embed_src, cfg, t)?);
        }
        forward.prefill_batch(dev, registry, cmd_ctx, model, &embeds, 3, p, &[x, d1, d2])?;
        let v = forward.mtp_verify_argmax(dev, registry, cmd_ctx, model, 3)?;
        // Accept: longest matching prefix of d_i == v_{i-1}.
        let mut m = 0u32;
        if d1 == v[0] {
            m = 1;
            if d2 == v[1] {
                m = 2;
            }
        }
        // Reconcile: restore S_p + replay the M+1 accepted real tokens.
        forward.restore_recurrent_state(dev, cmd_ctx)?;
        forward.set_kv_seq_len(p);
        for i in 0..=(m as usize) {
            let t = replay_toks[i];
            let e = embed_lookup(embed_src, cfg, t)?;
            forward.forward_token(dev, registry, cmd_ctx, model, &e, p + i as u32, t)?;
        }
        let got = forward.mtp_state_hashes(dev, cmd_ctx)?;
        let got_kv = forward.kv_seq_len();
        let (ref_h, ref_kv) = match m {
            0 => (ref1, ref1_kv),
            1 => (ref2, ref2_kv),
            _ => (ref3, ref3_kv),
        };
        // Sanity: verify argmax must match the decode oracle (v0.5.6
        // byte-ident) — but ONLY at positions whose verify INPUT matches the
        // reference decode's input. Position 0's input is `x` in both, so
        // v[0]==a0 always. Position 1's input is d1, which equals a0 only
        // when the first draft was accepted (target>=1); position 2's input
        // is d2==a1 only at full accept (target==2). Comparing at mismatched
        // inputs is meaningless (different token → different next-token).
        let v_ok = v[0] == a0
            && (target < 1 || v[1] == a1)
            && (target < 2 || v[2] == a2);
        let m_ok = m == target;
        let state_ok = got == ref_h && got_kv == ref_kv;
        let ok = v_ok && m_ok && state_ok;
        all_ok &= ok;
        let kind = match target {
            0 => "M=0 full-reject ",
            1 => "M=1 PARTIAL    ",
            _ => "M=2 full-accept",
        };
        eprintln!(
            "  {kind}: drafts=[{d1},{d2}] verify=[{},{},{}] M={m} \
             | committed ssm={:#018x} conv={:#018x} kv={got_kv} \
             | ref ssm={:#018x} conv={:#018x} kv={ref_kv} \
             | v_ok={v_ok} M_ok={m_ok} state_bit_ident={state_ok} -> {}",
            v[0], v[1], v[2], got.0, got.1, ref_h.0, ref_h.1,
            if ok { "PASS (rel=0)" } else { "FAIL" },
        );
        // Teeth on the PARTIAL case: a buggy "restore-without-replay" must
        // be DETECTABLY wrong (state stays at S_p, ≠ S_{p+2}).
        if target == 1 {
            forward.restore_recurrent_state(dev, cmd_ctx)?;
            forward.set_kv_seq_len(p + 2);
            let buggy = forward.mtp_state_hashes(dev, cmd_ctx)?;
            let teeth = buggy != ref2; // must MISMATCH the M=1 reference
            all_ok &= teeth;
            eprintln!(
                "  teeth (buggy restore-no-replay): committed ssm={:#018x} (= S_p) vs ref \
                 ssm={:#018x} (S_p+2) -> {} (gate {} detect the bug-class)",
                buggy.0, ref2.0,
                if teeth { "MISMATCH" } else { "MATCH" },
                if teeth { "DOES" } else { "FAILS to" },
            );
        }
    }
    // Leave a clean recurrent state for process teardown.
    forward.restore_recurrent_state(dev, cmd_ctx)?;
    forward.set_kv_seq_len(p);
    eprintln!(
        "=== MTP RECONCILE GATE: {} ===",
        if all_ok {
            "ALL PASS — accept selects M correctly, reconcile lands at S_{t+M} bit-ident, teeth detect the bug"
        } else {
            "FAIL — see rows above"
        }
    );
    Ok(all_ok)
}

/// Sprint MTP-Orch S4 — the self-speculative decode loop (gated `VF_MTP`).
///
/// Each iteration, from committed state `S_pos` with the next real token
/// `cur` (= the previous bonus, not yet emitted/processed):
///   1. draft `d_1..d_n` with the nextn head, chaining each draft's own
///      hidden into the next (`mtp_chain_capture_hidden`).
///   2. snapshot `S_pos`, then run the batched verify of `[cur, d_1..d_n]`
///      (`prefill_batch`, the v0.5.6 deterministic path) and read the
///      per-position argmax `v_0..v_n` (`mtp_verify_argmax`).
///   3. accept M = longest prefix with `d_{i+1} == v_i`; bonus = `v_M`.
///   4. reconcile: FULL accept (M==n) keeps the verify's already-computed
///      live state (the speedup case, no replay); PARTIAL/reject restores
///      `S_pos` and replays `cur, d_1..d_M` via the decode path (the
///      S3-proven byte-ident glue). Both land at `S_{pos+M+1}`.
///   5. emit `cur` + accepted drafts; carry `bonus` as next `cur`.
/// The emitted token stream is byte-identical to plain greedy decode
/// (accept admits only tokens equal to the verify argmax).
///
/// Returns `(generated_ids, stopped_on_eos, generated_text, visible_text)`
/// and prints an `[MTP]` acceptance/throughput summary to stderr.
#[allow(clippy::too_many_arguments)]
fn run_mtp_decode_loop(
    forward: &mut Forward,
    dev: &VulkanDevice,
    registry: &PipelineRegistry,
    cmd_ctx: &CommandContext,
    model: &LoadedModel,
    embed_src: &EmbeddingSource<'_>,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    config: &GenerateConfig,
    last_logits: &[f32],
    start_pos: u32,
    on_token: &mut dyn FnMut(u32, &str),
) -> Result<(Vec<u32>, bool, String, String), Box<dyn std::error::Error>> {
    let n = std::env::var("VF_MTP_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| (1..=8).contains(&v))
        .unwrap_or(3);
    let max_seq = forward.kv_cache.config.max_seq_len;
    let max_tokens = config.max_tokens as usize;
    let mut generated: Vec<u32> = Vec::new();
    let mut text_bytes: Vec<u8> = Vec::new();
    let mut stopped_on_eos = false;
    let mut pos = start_pos;

    // Acceptance / iteration stats.
    let mut iters: u64 = 0;
    let mut accepted_sum: u64 = 0; // Σ M (accepted drafts)
    let mut drafted_sum: u64 = 0; // Σ n
    let mut full_accepts: u64 = 0;

    // Push + stream one committed token; returns true when generation must
    // stop after it (EOS → not emitted, matching plain greedy; or cap hit).
    macro_rules! commit {
        ($id:expr) => {{
            let id = $id;
            if tokenizer.is_eos(id) {
                stopped_on_eos = true;
                true
            } else if generated.len() >= max_tokens {
                true
            } else {
                let b = tokenizer.decode_token_bytes(id);
                let s = String::from_utf8_lossy(&b);
                on_token(id, &s);
                text_bytes.extend_from_slice(&b);
                generated.push(id);
                generated.len() >= max_tokens || pos >= max_seq
            }
        }};
    }

    // --- Seed: plain-decode the first token (establishes mtp_h_buf = the
    // trunk hidden the first draft needs; prefill's batched path does not
    // run the decode-only hook). ---
    let x0 = argmax(last_logits) as u32;
    if commit!(x0) {
        let text = String::from_utf8_lossy(&text_bytes).into_owned();
        return Ok((generated, stopped_on_eos, text.clone(), text));
    }
    let e0 = embed_lookup(embed_src, cfg, x0)?;
    forward.forward_token(dev, registry, cmd_ctx, model, &e0, pos, x0)?;
    let next_logits = forward.logits()?;
    pos += 1;
    let mut cur = argmax(&next_logits) as u32; // token@pos, not yet emitted

    loop {
        if generated.len() >= max_tokens || pos >= max_seq {
            break;
        }
        if config
            .cancel_token
            .as_ref()
            .is_some_and(|t| t.load(std::sync::atomic::Ordering::Acquire))
        {
            stopped_on_eos = true;
            break;
        }
        // Guard: an n-draft verify needs positions pos..pos+n in range.
        if pos as u64 + n as u64 >= max_seq as u64 {
            // Tail: plain-decode `cur` and finish next loop turn.
            if commit!(cur) {
                break;
            }
            let e = embed_lookup(embed_src, cfg, cur)?;
            forward.forward_token(dev, registry, cmd_ctx, model, &e, pos, cur)?;
            cur = argmax(&forward.logits()?) as u32;
            pos += 1;
            continue;
        }

        // 1. Draft chain d_1..d_n from (mtp_h_buf = hidden@(pos-1), cur@pos).
        let mut drafts: Vec<u32> = Vec::with_capacity(n);
        let mut e = embed_lookup(embed_src, cfg, cur)?;
        let mut dpos = pos;
        for i in 0..n {
            let dl = forward.mtp_draft_logits(dev, registry, cmd_ctx, model, &e, dpos)?;
            let d = argmax(&dl) as u32;
            drafts.push(d);
            if i + 1 < n {
                forward.mtp_chain_capture_hidden(dev, cmd_ctx)?;
                e = embed_lookup(embed_src, cfg, d)?;
                dpos += 1;
            }
        }

        // 2. Snapshot S_pos, batched verify of [cur, d_1..d_n], per-pos argmax.
        forward.snapshot_recurrent_state(dev, cmd_ctx)?;
        let mut verify_toks: Vec<u32> = Vec::with_capacity(n + 1);
        verify_toks.push(cur);
        verify_toks.extend_from_slice(&drafts);
        let mut embeds: Vec<f32> = Vec::with_capacity((n + 1) * cfg.hidden_dim as usize);
        for &t in &verify_toks {
            embeds.extend(embed_lookup(embed_src, cfg, t)?);
        }
        forward.prefill_batch(
            dev, registry, cmd_ctx, model, &embeds, (n + 1) as u32, pos, &verify_toks,
        )?;
        let v = forward.mtp_verify_argmax(dev, registry, cmd_ctx, model, (n + 1) as u32)?;

        // 3. Accept: longest prefix d_{i+1} == v_i.
        let mut m = 0usize;
        while m < n && drafts[m] == v[m] {
            m += 1;
        }
        let bonus = v[m];
        iters += 1;
        accepted_sum += m as u64;
        drafted_sum += n as u64;

        // 4. Reconcile to S_{pos+m+1}.
        if m == n {
            full_accepts += 1;
            // Keep the verify's live state (= committed; prefill_batch
            // already set kv = pos+n+1 = pos+m+1). Refresh the draft h_t
            // from the last committed row of batch_residual (= hidden@pos+m).
            forward.set_kv_seq_len(pos + m as u32 + 1);
            forward.mtp_set_h_from_batch_row(dev, cmd_ctx, m as u32)?;
        } else {
            forward.restore_recurrent_state(dev, cmd_ctx)?;
            forward.set_kv_seq_len(pos);
            // Replay cur, d_1..d_M via the verified decode path.
            let mut rp = pos;
            let replay: Vec<u32> = std::iter::once(cur)
                .chain(drafts[..m].iter().copied())
                .collect();
            for t in replay {
                let e = embed_lookup(embed_src, cfg, t)?;
                forward.forward_token(dev, registry, cmd_ctx, model, &e, rp, t)?;
                rp += 1;
            }
            // last forward_token left mtp_h_buf = hidden@(pos+m). ✓
        }

        // 5. Emit cur + accepted drafts; carry bonus as next cur.
        let mut stop = commit!(cur);
        if !stop {
            for &d in &drafts[..m] {
                if commit!(d) {
                    stop = true;
                    break;
                }
            }
        }
        cur = bonus;
        pos += m as u32 + 1;
        if stop {
            break;
        }
    }

    let rate = if drafted_sum > 0 {
        100.0 * accepted_sum as f64 / drafted_sum as f64
    } else {
        0.0
    };
    let mean_commit = if iters > 0 {
        // tokens committed per spec iteration = M+2 (cur + M drafts + bonus),
        // but cur/bonus chain across iters → net new = M+1 per iter.
        1.0 + accepted_sum as f64 / iters as f64
    } else {
        0.0
    };
    eprintln!(
        "[MTP] n={n} iters={iters} accept={accepted_sum}/{drafted_sum}={rate:.1}% \
         full_accepts={full_accepts}/{iters} mean_new_tokens/iter={mean_commit:.2} \
         gen={} (byte-ident to plain greedy by construction)",
        generated.len()
    );
    let text = String::from_utf8_lossy(&text_bytes).into_owned();
    Ok((generated, stopped_on_eos, text.clone(), text))
}

/// CPU dequant of one row of `token_embd.weight` straight out of the
/// mmap'd GGUF.
///
/// Sprint 17B — `token_embd.weight` is Q4_K in Q4_K_M GGUFs and Q3_K
/// in Q3_K_M GGUFs (llama-quantize converts the embedding tensor
/// alongside the weights). Block layout differs (144 B vs 110 B), so
/// dispatch on the tensor's `ggml_type`.
pub fn embedding_row(
    gguf: &GgufFile,
    cfg: &ModelConfig,
    token_id: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use super::gguf::GgmlType;
    use super::q3k;
    use super::q4_0;
    use super::q5k;
    use super::q6k;
    let info = gguf
        .tensor("token_embd.weight")
        .ok_or("token_embd.weight not in GGUF")?;
    let bytes = gguf.tensor_bytes(info);
    match info.ggml_type {
        GgmlType::Q4K => {
            let blocks_per_row = (cfg.hidden_dim as usize) / q4k::QUANT_K;
            let row_bytes = blocks_per_row * q4k::BLOCK_BYTES;
            let row_off = (token_id as usize) * row_bytes;
            if row_off + row_bytes > bytes.len() {
                return Err(format!("token_id {token_id} out of range").into());
            }
            let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
            for b in 0..blocks_per_row {
                let blk_off = row_off + b * q4k::BLOCK_BYTES;
                let block: &[u8; q4k::BLOCK_BYTES] =
                    (&bytes[blk_off..blk_off + q4k::BLOCK_BYTES]).try_into().unwrap();
                out.extend_from_slice(&q4k::dequant_block(block));
            }
            Ok(out)
        }
        GgmlType::Q3K => {
            let blocks_per_row = (cfg.hidden_dim as usize) / q3k::QUANT_K;
            let row_bytes = blocks_per_row * q3k::BLOCK_BYTES;
            let row_off = (token_id as usize) * row_bytes;
            if row_off + row_bytes > bytes.len() {
                return Err(format!("token_id {token_id} out of range").into());
            }
            let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
            for b in 0..blocks_per_row {
                let blk_off = row_off + b * q3k::BLOCK_BYTES;
                let block: &[u8; q3k::BLOCK_BYTES] =
                    (&bytes[blk_off..blk_off + q3k::BLOCK_BYTES]).try_into().unwrap();
                out.extend_from_slice(&q3k::dequant_block(block));
            }
            Ok(out)
        }
        GgmlType::Q5K => {
            let blocks_per_row = (cfg.hidden_dim as usize) / q5k::QUANT_K;
            let row_bytes = blocks_per_row * q5k::BLOCK_BYTES;
            let row_off = (token_id as usize) * row_bytes;
            if row_off + row_bytes > bytes.len() {
                return Err(format!("token_id {token_id} out of range").into());
            }
            let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
            for b in 0..blocks_per_row {
                let blk_off = row_off + b * q5k::BLOCK_BYTES;
                let block: &[u8; q5k::BLOCK_BYTES] =
                    (&bytes[blk_off..blk_off + q5k::BLOCK_BYTES]).try_into().unwrap();
                out.extend_from_slice(&q5k::dequant_block(block));
            }
            Ok(out)
        }
        GgmlType::Q6K => {
            // Sprint 52F — E4B GGUFs emit `token_embd.weight` as Q6_K
            // (E2B uses Q4_K, 26B likely Q6_K too). Mirrors the Q5_K
            // arm above — same block-iteration shape, just calls the
            // `q6k::dequant_block` we shipped in Sprint 52E P3 for
            // the PLE-table dequant.
            let blocks_per_row = (cfg.hidden_dim as usize) / q6k::QUANT_K;
            let row_bytes = blocks_per_row * q6k::BLOCK_BYTES;
            let row_off = (token_id as usize) * row_bytes;
            if row_off + row_bytes > bytes.len() {
                return Err(format!("token_id {token_id} out of range").into());
            }
            let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
            for b in 0..blocks_per_row {
                let blk_off = row_off + b * q6k::BLOCK_BYTES;
                let block: &[u8; q6k::BLOCK_BYTES] =
                    (&bytes[blk_off..blk_off + q6k::BLOCK_BYTES]).try_into().unwrap();
                out.extend_from_slice(&q6k::dequant_block(block));
            }
            Ok(out)
        }
        GgmlType::Q4_0 => {
            // block_size = 32 (NOT 256 like K-quants), so a hidden=4096
            // row spans 128 blocks vs 16 for the K-quants.
            let blocks_per_row = (cfg.hidden_dim as usize) / q4_0::QUANT_K;
            let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
            let row_off = (token_id as usize) * row_bytes;
            if row_off + row_bytes > bytes.len() {
                return Err(format!("token_id {token_id} out of range").into());
            }
            let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
            for b in 0..blocks_per_row {
                let blk_off = row_off + b * q4_0::BLOCK_BYTES;
                let block: &[u8; q4_0::BLOCK_BYTES] =
                    (&bytes[blk_off..blk_off + q4_0::BLOCK_BYTES]).try_into().unwrap();
                out.extend_from_slice(&q4_0::dequant_block(block));
            }
            Ok(out)
        }
        other => Err(format!("token_embd.weight type {other:?} not yet supported").into()),
    }
}

// =======================================================================
// ThinkFilter — strips `<think>…</think>` blocks from streamed output.
// Operates on decoded text rather than token ids, so it works whether
// the model emits `<think>` as the special id 151667 or as the BPE
// sequence ["<th","ink",">"] — both decode to the same string.
// =======================================================================

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Default)]
pub struct ThinkFilter {
    in_think: bool,
    /// Accumulator for partial-match boundaries across token chunks.
    pending: String,
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a chunk of decoded text. Returns the visible portion (text
    /// not inside a `<think>…</think>` block, with partial open/close
    /// tags held back until the next chunk so they can be matched
    /// across token boundaries).
    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find(THINK_CLOSE) {
                    let cut = idx + THINK_CLOSE.len();
                    self.pending.drain(..cut);
                    self.in_think = false;
                    continue;
                } else if let Some(safe_end) = partial_tag_split(&self.pending, THINK_CLOSE) {
                    // Discard everything before the partial tail; keep the tail.
                    self.pending.drain(..safe_end);
                    break;
                } else {
                    self.pending.clear();
                    break;
                }
            } else {
                if let Some(idx) = self.pending.find(THINK_OPEN) {
                    out.push_str(&self.pending[..idx]);
                    let cut = idx + THINK_OPEN.len();
                    self.pending.drain(..cut);
                    self.in_think = true;
                    continue;
                } else if let Some(safe_end) = partial_tag_split(&self.pending, THINK_OPEN) {
                    out.push_str(&self.pending[..safe_end]);
                    self.pending.drain(..safe_end);
                    break;
                } else {
                    out.push_str(&self.pending);
                    self.pending.clear();
                    break;
                }
            }
        }
        out
    }

    /// Finalise at end of stream. If we're still inside a `<think>`
    /// block, anything pending is dropped; otherwise it's emitted.
    pub fn flush(&mut self) -> String {
        if self.in_think {
            self.pending.clear();
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }

    /// Whole-string convenience — runs the filter to completion and
    /// returns the visible text. Used for post-processing
    /// `GenerateResult::visible_text` from the raw `generated_text`.
    pub fn strip_all(text: &str) -> String {
        let mut f = ThinkFilter::new();
        let mut out = f.push(text);
        out.push_str(&f.flush());
        out
    }
}

/// Returns the byte offset within `text` at which the longest suffix
/// that could be a *prefix* of `tag` begins, i.e. the safe split
/// point: bytes before it can be emitted, bytes from it on must be
/// held back. Returns `None` if no suffix of `text` is a prefix of
/// `tag` (so all of `text` is safe to emit).
fn partial_tag_split(text: &str, tag: &str) -> Option<usize> {
    let max_k = (tag.len().saturating_sub(1)).min(text.len());
    for k in (1..=max_k).rev() {
        let start = text.len() - k;
        if !text.is_char_boundary(start) {
            continue;
        }
        if tag.starts_with(&text[start..]) {
            return Some(start);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_filter_passthrough_when_no_tags() {
        let mut f = ThinkFilter::new();
        assert_eq!(f.push("Hello world"), "Hello world");
        assert_eq!(f.flush(), "");
    }

    #[test]
    fn think_filter_strips_full_block_in_one_chunk() {
        let mut f = ThinkFilter::new();
        let v = f.push("Hello <think>internal</think>World");
        let after = f.flush();
        assert_eq!(format!("{v}{after}"), "Hello World");
    }

    #[test]
    fn think_filter_handles_token_split_boundaries() {
        // Mirrors the actual Qwen3 BPE sequence ["<th","ink",">"]/["</","think",">"].
        let mut f = ThinkFilter::new();
        let mut visible = String::new();
        for chunk in ["Hello ", "<th", "ink", ">", "internal", "</", "think", ">", "World"] {
            visible.push_str(&f.push(chunk));
        }
        visible.push_str(&f.flush());
        assert_eq!(visible, "Hello World");
    }

    #[test]
    fn think_filter_empty_block() {
        assert_eq!(ThinkFilter::strip_all("<think></think>foo"), "foo");
    }

    #[test]
    fn think_filter_only_open_no_close_drops_tail() {
        // Streaming was cut short while still inside <think>: drop
        // any post-open content rather than leaking it.
        let mut f = ThinkFilter::new();
        let v = f.push("ok <think>thinking but never closed");
        let tail = f.flush();
        assert_eq!(format!("{v}{tail}"), "ok ");
    }

    #[test]
    fn think_filter_partial_open_at_end_held_back() {
        // "<th" alone could become "<think>" — must hold it back.
        let mut f = ThinkFilter::new();
        assert_eq!(f.push("hi <th"), "hi ");
        // Continuation as plain text means the partial wasn't a tag.
        assert_eq!(f.push("ello"), "<thello");
    }

    #[test]
    fn partial_tag_split_basic() {
        assert_eq!(partial_tag_split("hi <th", "<think>"), Some(3));
        assert_eq!(partial_tag_split("hi <", "<think>"), Some(3));
        assert_eq!(partial_tag_split("hi", "<think>"), None);
        assert_eq!(partial_tag_split("", "<think>"), None);
    }
}
