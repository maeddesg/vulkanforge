//! Multi-turn chat session — keeps the KV cache populated across
//! turns so Turn 2 doesn't re-prefill Turn 1's prompt + response.
//!
//! Phase 3B / Schritt 2. Built on top of [`crate::backend::vulkan::decode::generate_from_tokens`],
//! which is the only existing entry point that lets us prefill at a
//! non-zero `start_pos` without resetting the KV cache.
//!
//! Turn boundaries:
//!
//! * **First turn** — full chat template:
//!   `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`
//! * **Subsequent turns** — the model emitted `<|im_end|>` to end the
//!   previous response which we caught as EOS and did *not* commit
//!   to the KV cache. So Turn 2's prefill prepends `<|im_end|>\n` to
//!   re-establish the boundary, then frames the new user message:
//!   `<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`
//!
//! Context overflow is reported as a structured error (no sliding
//! window in Phase 3B per prompt §2.3).

use std::time::Duration;

use super::chat_template::ChatTemplate;
use super::commands::CommandContext;
use super::decode::{generate_from_tokens, GenerateConfig};
use super::device::VulkanDevice;
use super::forward::Forward;
use super::gguf::{GgufFile, ModelConfig};
use super::loader::LoadedModel;
use super::pipeline_registry::PipelineRegistry;
use super::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
    pub token_count: u32,
}

#[derive(Debug)]
pub enum ChatError {
    ContextOverflow {
        current_pos: u32,
        needed: u32,
        max_seq_len: u32,
    },
    Generation(Box<dyn std::error::Error>),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::ContextOverflow { current_pos, needed, max_seq_len } => write!(
                f,
                "context overflow: current_pos={current_pos} + needed={needed} > max_seq_len={max_seq_len}. Use /reset to clear history."
            ),
            ChatError::Generation(e) => write!(f, "generation error: {e}"),
        }
    }
}

impl std::error::Error for ChatError {}

/// One Multi-Turn chat session. Owns the [`Forward`] instance and
/// tracks the next-free position in its KV cache.
pub struct ChatSession {
    pub forward: Forward,
    pub history: Vec<ChatTurn>,
    pub system_prompt: String,
    pub template: ChatTemplate,
    /// Next free position in the KV cache. After Turn 1's prefill+gen,
    /// this is `prompt_tokens + generated_tokens`. Increments
    /// monotonically across turns until [`reset`] is called.
    pub current_pos: u32,
    /// Whether Turn 1 has been sent yet — used to decide between the
    /// full first-turn chat template and the leaner continuation form.
    pub turn_count: u32,
}

impl ChatSession {
    pub fn new(forward: Forward, system_prompt: impl Into<String>) -> Self {
        Self::new_with_template(forward, system_prompt, ChatTemplate::ChatML)
    }

    pub fn new_with_template(
        forward: Forward,
        system_prompt: impl Into<String>,
        template: ChatTemplate,
    ) -> Self {
        Self {
            forward,
            history: Vec::new(),
            system_prompt: system_prompt.into(),
            template,
            current_pos: 0,
            turn_count: 0,
        }
    }

    pub fn max_seq_len(&self) -> u32 {
        self.forward.kv_cache.config.max_seq_len
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.max_seq_len().saturating_sub(self.current_pos)
    }

    /// Discard KV state + history. Next `send()` is treated as Turn 1.
    pub fn reset(&mut self) {
        self.forward.kv_cache.reset();
        self.history.clear();
        self.current_pos = 0;
        self.turn_count = 0;
    }

    /// Build the prefill-token sequence for `user_message` based on
    /// whether this is the first turn or a continuation. Dispatches
    /// to the configured [`ChatTemplate`].
    pub fn build_prefill_tokens(&self, tokenizer: &Tokenizer, user_message: &str) -> Vec<u32> {
        if self.turn_count == 0 {
            self.template.render_first_turn(tokenizer, &self.system_prompt, user_message)
        } else {
            self.template.render_continuation(tokenizer, user_message)
        }
    }

    /// Send a user message, prefill it on top of the existing KV
    /// state, then decode greedily. Streams generated tokens through
    /// `on_token` (post-think-filter when `config.think_filter` is set).
    #[allow(clippy::too_many_arguments)]
    pub fn send_streaming(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        gguf: &GgufFile,
        cfg: &ModelConfig,
        tokenizer: &Tokenizer,
        user_message: &str,
        config: &GenerateConfig,
        mut on_visible: impl FnMut(&str),
    ) -> Result<TurnResult, ChatError> {
        let prefill = self.build_prefill_tokens(tokenizer, user_message);
        let need = prefill.len() as u32 + config.max_tokens;
        if self.current_pos + need > self.max_seq_len() {
            return Err(ChatError::ContextOverflow {
                current_pos: self.current_pos,
                needed: need,
                max_seq_len: self.max_seq_len(),
            });
        }

        // Wire streaming through ThinkFilter when requested.
        let mut filter = if config.think_filter {
            Some(super::decode::ThinkFilter::new())
        } else {
            None
        };
        let mut on_token = |_id: u32, raw: &str| {
            if let Some(f) = filter.as_mut() {
                let visible = f.push(raw);
                if !visible.is_empty() {
                    on_visible(&visible);
                }
            } else {
                on_visible(raw);
            }
        };

        let result = generate_from_tokens(
            &mut self.forward,
            dev, registry, cmd_ctx, model,
            super::decode::EmbeddingSource::Gguf(gguf),
            cfg, tokenizer,
            &prefill, self.current_pos, config, false, &mut on_token,
        )
        .map_err(ChatError::Generation)?;

        // Flush any text held back across the boundary (e.g. partial
        // `</think>` tail at end of stream).
        if let Some(f) = filter.as_mut() {
            let tail = f.flush();
            if !tail.is_empty() {
                on_visible(&tail);
            }
        }

        let prompt_tok = result.prompt_tokens as u32;
        let gen_tok = result.generated_tokens as u32;
        self.current_pos += prompt_tok + gen_tok;
        self.turn_count += 1;

        // Track turn history (best-effort: we recover the user
        // message from the input arg, the assistant text from the
        // generation result).
        self.history.push(ChatTurn {
            role: Role::User,
            content: user_message.to_string(),
            token_count: prompt_tok,
        });
        self.history.push(ChatTurn {
            role: Role::Assistant,
            content: result.visible_text.clone(),
            token_count: gen_tok,
        });

        Ok(TurnResult {
            prompt_tokens: prompt_tok,
            generated_tokens: gen_tok,
            generated_text: result.generated_text,
            visible_text: result.visible_text,
            stopped_on_eos: result.stopped_on_eos,
            prefill_time: result.prefill_time,
            decode_time: result.decode_time,
        })
    }

    /// Convenience non-streaming send — all generated text returned in
    /// one go, no per-token callbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn send(
        &mut self,
        dev: &VulkanDevice,
        registry: &PipelineRegistry,
        cmd_ctx: &CommandContext,
        model: &LoadedModel,
        gguf: &GgufFile,
        cfg: &ModelConfig,
        tokenizer: &Tokenizer,
        user_message: &str,
        config: &GenerateConfig,
    ) -> Result<TurnResult, ChatError> {
        self.send_streaming(
            dev, registry, cmd_ctx, model, gguf, cfg, tokenizer,
            user_message, config, |_| {},
        )
    }
}

#[derive(Debug)]
pub struct TurnResult {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub generated_text: String,
    pub visible_text: String,
    pub stopped_on_eos: bool,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl TurnResult {
    pub fn prefill_tok_s(&self) -> f64 {
        let s = self.prefill_time.as_secs_f64();
        if s == 0.0 { 0.0 } else { self.prompt_tokens as f64 / s }
    }
    pub fn decode_tok_s(&self) -> f64 {
        let s = self.decode_time.as_secs_f64();
        if s == 0.0 { 0.0 } else { self.generated_tokens as f64 / s }
    }
}
