//! Token-by-token decode loop driving [`Forward::forward_token`].
//!
//! Phase 2D / Schritt 2.9. Prefill and decode are both single-token
//! through the same forward pass — Phase 3 will replace prefill with
//! a batched GEMM path, but for now correctness wins over throughput.

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

pub struct GenerateConfig {
    pub max_tokens: u32,
    pub print_stream: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            print_stream: false,
        }
    }
}

#[derive(Debug)]
pub struct GenerateResult {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub generated_text: String,
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
    let max_seq = forward.kv_cache.config.max_seq_len;
    if prompt_tokens.len() as u32 + config.max_tokens > max_seq {
        return Err(format!(
            "prompt {} + max_tokens {} > max_seq_len {}",
            prompt_tokens.len(),
            config.max_tokens,
            max_seq,
        )
        .into());
    }
    forward.kv_cache.reset();

    // ---- Prefill ----
    let prefill_start = Instant::now();
    for (pos, &tid) in prompt_tokens.iter().enumerate() {
        let embd = embedding_row(gguf, cfg, tid)?;
        forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos as u32)?;
    }
    let mut last_logits = forward.logits()?;
    let prefill_time = prefill_start.elapsed();

    // ---- Decode ----
    let decode_start = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = prompt_tokens.len() as u32;
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

        if config.print_stream {
            let s = tokenizer.decode_token(next_id);
            print!("{}", s);
            std::io::stdout().flush().ok();
        }
        generated.push(next_id);

        let embd = embedding_row(gguf, cfg, next_id)?;
        forward.forward_token(dev, registry, cmd_ctx, model, &embd, pos)?;
        last_logits = forward.logits()?;
        pos += 1;
    }
    if config.print_stream {
        println!();
    }
    let decode_time = decode_start.elapsed();

    Ok(GenerateResult {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated.len(),
        generated_text: tokenizer.decode(&generated),
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
/// mmap'd GGUF. Identical to the helper that the Phase-2C demo binary
/// inlines, lifted out so the decode loop can reuse it.
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
