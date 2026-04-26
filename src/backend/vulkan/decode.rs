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

use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::forward::Forward;
use super::gguf::{GgufFile, ModelConfig};
use super::loader::LoadedModel;
use super::pipeline_registry::PipelineRegistry;
use super::q4k;
use super::tokenizer::{apply_chat_template, Tokenizer};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: u32,
    pub print_stream: bool,
    /// Strip `<think>...</think>` blocks from the visible output (only
    /// affects streaming; the raw token sequence is unchanged).
    pub think_filter: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            print_stream: false,
            think_filter: false,
        }
    }
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
    let prompt_tokens = apply_chat_template(tokenizer, prompt, None);
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
            forward, dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
            &prompt_tokens, 0, config, &mut on_token,
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
            forward, dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
            &prompt_tokens, 0, config, &mut on_token,
        )
    }
}

/// Lower-level driver: prefills `prefill_tokens` starting at
/// `start_pos` (does NOT reset the KV cache), then runs greedy decode
/// up to `config.max_tokens` or EOS. Streams each generated token
/// through `on_token(id, decoded_text)`.
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
    gguf: &GgufFile,
    cfg: &ModelConfig,
    tokenizer: &Tokenizer,
    prefill_tokens: &[u32],
    start_pos: u32,
    config: &GenerateConfig,
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
    for &tid in prefill_tokens {
        let embd = embedding_row(gguf, cfg, tid)?;
        forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos)?;
        pos += 1;
    }
    let mut last_logits = forward.logits()?;
    let prefill_time = prefill_start.elapsed();

    // ---- Decode ----
    let decode_start = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut stopped_on_eos = false;

    loop {
        let next_id = argmax(&last_logits) as u32;
        if tokenizer.is_eos(next_id) {
            stopped_on_eos = true;
            break;
        }
        if generated.len() >= config.max_tokens as usize || pos >= max_seq {
            break;
        }
        let raw = tokenizer.decode_token(next_id);
        on_token(next_id, &raw);
        generated.push(next_id);

        let embd = embedding_row(gguf, cfg, next_id)?;
        forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos)?;
        last_logits = forward.logits()?;
        pos += 1;
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
pub fn embedding_row(
    gguf: &GgufFile,
    cfg: &ModelConfig,
    token_id: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let info = gguf
        .tensor("token_embd.weight")
        .ok_or("token_embd.weight not in GGUF")?;
    let blocks_per_row = (cfg.hidden_dim as usize) / q4k::QUANT_K;
    let row_bytes = blocks_per_row * q4k::BLOCK_BYTES;
    let bytes = gguf.tensor_bytes(info);
    let row_off = (token_id as usize) * row_bytes;
    if row_off + row_bytes > bytes.len() {
        return Err(format!("token_id {token_id} out of range").into());
    }
    let mut out = Vec::with_capacity(cfg.hidden_dim as usize);
    for b in 0..blocks_per_row {
        let blk_off = row_off + b * q4k::BLOCK_BYTES;
        let block: &[u8; q4k::BLOCK_BYTES] = (&bytes[blk_off..blk_off + q4k::BLOCK_BYTES])
            .try_into()
            .unwrap();
        out.extend_from_slice(&q4k::dequant_block(block));
    }
    Ok(out)
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
