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

use std::io::Write;
use std::time::{Duration, Instant};

use super::chat_template::ChatTemplate;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::forward::Forward;
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
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            print_stream: false,
            think_filter: false,
            sampling: Sampling::greedy(),
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
        if force_per_token_prefill {
            // Sprint 20-M3 — SafeTensors FP8 models reach this branch:
            // batched prefill would hit the (FP16) `mul_mm.comp` GEMM
            // which has no `DATA_A_FP8` variant, so we fall back to
            // the per-token GEMV path. That's slow at long pp but
            // proves the end-to-end pipeline; an FP8 GEMM port is
            // future work (Sprint 19A-style coopmat coverage).
            for &tid in prefill_tokens {
                let embd = embed_lookup(&embed_src, cfg, tid)?;
                forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos, tid)?;
                pos += 1;
            }
        } else {
            let chunk_size = forward.max_prefill_tokens.max(1) as usize;
            for chunk in prefill_tokens.chunks(chunk_size) {
                let chunk_len = chunk.len() as u32;
                let mut chunk_embeds: Vec<f32> =
                    Vec::with_capacity(chunk.len() * cfg.hidden_dim as usize);
                for &tid in chunk {
                    chunk_embeds.extend(embed_lookup(&embed_src, cfg, tid)?);
                }
                forward.prefill_batch(
                    dev, registry, cmd_ctx, model, &chunk_embeds, chunk_len, pos,
                )?;
                pos += chunk_len;
            }
        }
    }
    let mut last_logits = forward.logits()?;
    let prefill_time = prefill_start.elapsed();

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
    let async_decode = std::env::var("VULKANFORGE_DISABLE_ASYNC_DECODE")
        .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

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

            let embd = embed_lookup(&embed_src, cfg, next_id)?;
            forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos, next_id)?;
            last_logits = forward.logits()?;
            pos += 1;
        }
    }
    let _ = last_logits;
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
