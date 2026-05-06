//! Chat-template detection + rendering for the Phase-4D / Phase-5C
//! model zoo.
//!
//! VulkanForge ships its own minimal renderer rather than running the
//! Jinja template embedded in each GGUF — every supported model maps
//! to one of a small number of canonical layouts that are easy to
//! emit by hand. The renderer takes the same inputs (system prompt,
//! user message, assistant-priming flag) and produces the prefill
//! token sequence.
//!
//! Auto-detection (see [`ChatTemplate::detect`]) prefers the GGUF
//! `tokenizer.chat_template` string over the architecture name when
//! the two disagree (DeepSeek-R1-Distill-Llama is `arch=llama` but
//! ships the DeepSeek-R1 template; Mistral and Llama-3 both share
//! `arch=llama` but use very different layouts).

use std::path::Path;

use super::gguf::GgufFile;
use super::tokenizer::{Tokenizer, TokenizerFlavour};

/// One of the canonical chat-template layouts supported in Phase 4D /
/// Phase 5C.
///
/// `Raw` is the no-template fallback — used when the user explicitly
/// asks for plain-text completion or for a base model with no chat
/// template at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Qwen / Qwen2 / Qwen3 ChatML form:
    /// `<|im_start|>role\n{content}<|im_end|>\n…<|im_start|>assistant\n`
    ChatML,
    /// Meta-Llama-3 chat form:
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n{content}<|eot_id|>…`
    Llama3,
    /// DeepSeek-R1 thinking form (Distill-Llama variant uses
    /// `<｜begin▁of▁sentence｜>` + `<｜User｜>` / `<｜Assistant｜>`).
    DeepSeekR1,
    /// Mistral Instruct form: `<s>[INST] {user} [/INST]`. Multi-turn
    /// re-uses the same `[INST]…[/INST]` brackets per turn.
    Mistral,
    /// Plain text — system + user concatenated, no role markers.
    Raw,
}

impl ChatTemplate {
    /// Detect the right template for this GGUF. Order: explicit
    /// match against the embedded Jinja `tokenizer.chat_template`
    /// string first (covers DeepSeek-on-Llama-arch and Mistral-on-
    /// Llama-arch), then fall back to the tokenizer flavour.
    pub fn detect(gguf: &GgufFile, tokenizer: &Tokenizer) -> Self {
        let template_str = gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // DeepSeek-R1's template ships these exact glyphs.
        if template_str.contains("<｜User｜>") || template_str.contains("<｜Assistant｜>") {
            return ChatTemplate::DeepSeekR1;
        }
        // Llama-3 ships <|start_header_id|>...<|end_header_id|> and
        // an `<|eot_id|>` turn separator — both are unique to that
        // family.
        if template_str.contains("<|start_header_id|>") || template_str.contains("<|eot_id|>") {
            return ChatTemplate::Llama3;
        }
        // Qwen3 (and any other ChatML model) drops `<|im_start|>` /
        // `<|im_end|>` literals into the template.
        if template_str.contains("<|im_start|>") || template_str.contains("<|im_end|>") {
            return ChatTemplate::ChatML;
        }
        // Mistral Instruct: `[INST]` + `[/INST]` brackets in the
        // template. Mistral is the canonical SPM template, so any
        // SPM model that doesn't match the others falls through here
        // via the tokenizer flavour fallback below as well.
        if template_str.contains("[INST]") {
            return ChatTemplate::Mistral;
        }

        // Fallback: pick from the tokenizer flavour. SPM (Mistral,
        // Llama-2 family) returns `None` from `flavour()` — default
        // those to the Mistral [INST] template, which is the only
        // SPM-style template we currently render.
        match tokenizer.flavour() {
            Some(TokenizerFlavour::Qwen2) => ChatTemplate::ChatML,
            Some(TokenizerFlavour::Llama3) => ChatTemplate::Llama3,
            None => ChatTemplate::Mistral,
        }
    }

    /// v0.3.13 — detect from a HuggingFace SafeTensors model
    /// directory's `tokenizer_config.json`. Same string heuristics as
    /// [`Self::detect`], same flavour fallback. The renderers
    /// themselves are unchanged — they target the canonical layouts,
    /// not the upstream Jinja string.
    pub fn detect_hf(model_dir: &Path, tokenizer: &Tokenizer) -> Self {
        let tcfg_path = model_dir.join("tokenizer_config.json");
        let template_str = std::fs::read_to_string(&tcfg_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                v.get("chat_template")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        if template_str.contains("<｜User｜>") || template_str.contains("<｜Assistant｜>") {
            return ChatTemplate::DeepSeekR1;
        }
        if template_str.contains("<|start_header_id|>") || template_str.contains("<|eot_id|>") {
            return ChatTemplate::Llama3;
        }
        if template_str.contains("<|im_start|>") || template_str.contains("<|im_end|>") {
            return ChatTemplate::ChatML;
        }
        if template_str.contains("[INST]") {
            return ChatTemplate::Mistral;
        }
        match tokenizer.flavour() {
            Some(TokenizerFlavour::Qwen2) => ChatTemplate::ChatML,
            Some(TokenizerFlavour::Llama3) => ChatTemplate::Llama3,
            None => ChatTemplate::Mistral,
        }
    }

    /// Build the prefill token sequence for a single-turn message.
    /// Equivalent to `add_generation_prompt=True` in HuggingFace
    /// terms — ends right where the assistant should start writing.
    pub fn render_first_turn(
        self,
        tokenizer: &Tokenizer,
        system: &str,
        user: &str,
    ) -> Vec<u32> {
        match self {
            ChatTemplate::ChatML => render_chatml_first(tokenizer, system, user),
            ChatTemplate::Llama3 => render_llama3_first(tokenizer, system, user),
            ChatTemplate::DeepSeekR1 => render_deepseek_first(tokenizer, system, user),
            ChatTemplate::Mistral => render_mistral_first(tokenizer, system, user),
            ChatTemplate::Raw => render_raw_first(tokenizer, system, user),
        }
    }

    /// Build the prefill token sequence for a continuation turn —
    /// caller has already retained the prior turn's KV state up to
    /// (but not including) the previous assistant-turn terminator.
    pub fn render_continuation(self, tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
        match self {
            ChatTemplate::ChatML => render_chatml_continuation(tokenizer, user),
            ChatTemplate::Llama3 => render_llama3_continuation(tokenizer, user),
            ChatTemplate::DeepSeekR1 => render_deepseek_continuation(tokenizer, user),
            ChatTemplate::Mistral => render_mistral_continuation(tokenizer, user),
            ChatTemplate::Raw => render_raw_continuation(tokenizer, user),
        }
    }
}

// ---------- ChatML (Qwen3) ----------

fn render_chatml_first(tokenizer: &Tokenizer, system: &str, user: &str) -> Vec<u32> {
    let im_start = tokenizer
        .im_start_id
        .expect("ChatML render needs <|im_start|>");
    let im_end = tokenizer
        .im_end_id
        .expect("ChatML render needs <|im_end|>");

    let mut tokens = Vec::new();
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("system\n"));
    tokens.extend(tokenizer.encode(system));
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("assistant\n"));
    tokens
}

fn render_chatml_continuation(tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
    let im_start = tokenizer.im_start_id.expect("ChatML render needs <|im_start|>");
    let im_end = tokenizer.im_end_id.expect("ChatML render needs <|im_end|>");

    let mut tokens = Vec::new();
    // The previous turn was cut at the model-emitted `<|im_end|>` we
    // caught as EOS but didn't commit; replay it now.
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("assistant\n"));
    tokens
}

// ---------- Llama-3 ----------

fn llama3_special(tokenizer: &Tokenizer, name: &str) -> u32 {
    tokenizer
        .special_id(name)
        .unwrap_or_else(|| panic!("Llama-3 chat template missing special token {name:?}"))
}

fn render_llama3_first(tokenizer: &Tokenizer, system: &str, user: &str) -> Vec<u32> {
    let bot = llama3_special(tokenizer, "<|begin_of_text|>");
    let start_h = llama3_special(tokenizer, "<|start_header_id|>");
    let end_h = llama3_special(tokenizer, "<|end_header_id|>");
    let eot = llama3_special(tokenizer, "<|eot_id|>");

    let mut tokens = Vec::new();
    tokens.push(bot);
    if !system.is_empty() {
        tokens.push(start_h);
        tokens.extend(tokenizer.encode("system"));
        tokens.push(end_h);
        tokens.extend(tokenizer.encode("\n\n"));
        tokens.extend(tokenizer.encode(system));
        tokens.push(eot);
    }
    tokens.push(start_h);
    tokens.extend(tokenizer.encode("user"));
    tokens.push(end_h);
    tokens.extend(tokenizer.encode("\n\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(eot);
    tokens.push(start_h);
    tokens.extend(tokenizer.encode("assistant"));
    tokens.push(end_h);
    tokens.extend(tokenizer.encode("\n\n"));
    tokens
}

fn render_llama3_continuation(tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
    let start_h = llama3_special(tokenizer, "<|start_header_id|>");
    let end_h = llama3_special(tokenizer, "<|end_header_id|>");
    let eot = llama3_special(tokenizer, "<|eot_id|>");

    let mut tokens = Vec::new();
    tokens.push(eot);
    tokens.push(start_h);
    tokens.extend(tokenizer.encode("user"));
    tokens.push(end_h);
    tokens.extend(tokenizer.encode("\n\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(eot);
    tokens.push(start_h);
    tokens.extend(tokenizer.encode("assistant"));
    tokens.push(end_h);
    tokens.extend(tokenizer.encode("\n\n"));
    tokens
}

// ---------- DeepSeek-R1 (Distill-Llama variant) ----------
// Template (simplified, no tools):
//   <｜begin▁of▁sentence｜>{system}
//   <｜User｜>{user}<｜Assistant｜><think>\n
//
// `<think>` is part of the priming so the reasoning stage starts
// automatically — DeepSeek's R1 series always thinks before answering.

fn render_deepseek_first(tokenizer: &Tokenizer, system: &str, user: &str) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or_else(|| {
        tokenizer
            .special_id("<｜begin▁of▁sentence｜>")
            .unwrap_or(0)
    });
    let user_tok = tokenizer
        .special_id("<｜User｜>")
        .expect("DeepSeek-R1 chat template missing <｜User｜>");
    let asst_tok = tokenizer
        .special_id("<｜Assistant｜>")
        .expect("DeepSeek-R1 chat template missing <｜Assistant｜>");

    let mut tokens = Vec::new();
    tokens.push(bos);
    if !system.is_empty() {
        tokens.extend(tokenizer.encode(system));
    }
    tokens.push(user_tok);
    tokens.extend(tokenizer.encode(user));
    tokens.push(asst_tok);
    tokens.extend(tokenizer.encode("<think>\n"));
    tokens
}

fn render_deepseek_continuation(tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
    let user_tok = tokenizer
        .special_id("<｜User｜>")
        .expect("DeepSeek-R1 chat template missing <｜User｜>");
    let asst_tok = tokenizer
        .special_id("<｜Assistant｜>")
        .expect("DeepSeek-R1 chat template missing <｜Assistant｜>");
    let eot = tokenizer.special_id("<｜end▁of▁sentence｜>");

    let mut tokens = Vec::new();
    if let Some(id) = eot {
        tokens.push(id);
    }
    tokens.push(user_tok);
    tokens.extend(tokenizer.encode(user));
    tokens.push(asst_tok);
    tokens.extend(tokenizer.encode("<think>\n"));
    tokens
}

// ---------- Mistral Instruct ([INST]...[/INST]) ----------
// Template (Mistral-7B-Instruct-v0.3, single turn, no tools):
//   <s>[INST] {user_content} [/INST]
// Multi-turn (continuation):
//   ...{prev_assistant}</s>[INST] {user_content} [/INST]
//
// The HF Mistral chat template ignores the system role unless wrapped
// inside the user message. We replicate that: for the first turn, if
// `system` is non-empty we splice it ahead of `user` separated by a
// blank line.
//
// `[INST]` and `[/INST]` are real vocab entries (single ids) in the
// Mistral GGUF — using `special_id` keeps them as literal tokens
// rather than re-tokenizing the brackets as ASCII.

fn mistral_special(tokenizer: &Tokenizer, name: &str) -> u32 {
    tokenizer
        .special_id(name)
        .unwrap_or_else(|| panic!("Mistral chat template missing special token {name:?}"))
}

fn render_mistral_first(tokenizer: &Tokenizer, system: &str, user: &str) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or(1);
    let inst_open = mistral_special(tokenizer, "[INST]");
    let inst_close = mistral_special(tokenizer, "[/INST]");

    let mut tokens = Vec::new();
    tokens.push(bos);
    tokens.push(inst_open);
    // After the [INST] special token we emit a single leading space
    // (matches HF Jinja `'[INST] '`). Use `encode_no_prefix` so the
    // SPM normaliser doesn't add its own ▁ prefix on top — that would
    // produce a double ▁. We deliberately do NOT emit a trailing
    // space before `[/INST]`: the HF template's `' [/INST]'` would
    // tokenise to a `▁` token right before the special id, which the
    // model treats as a confusing post-content space and pushes the
    // first generated token off-distribution.
    let body = if system.is_empty() {
        format!(" {}", user)
    } else {
        format!(" {}\n\n{}", system, user)
    };
    tokens.extend(tokenizer.encode_no_prefix(&body));
    tokens.push(inst_close);
    tokens
}

fn render_mistral_continuation(tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
    let inst_open = mistral_special(tokenizer, "[INST]");
    let inst_close = mistral_special(tokenizer, "[/INST]");
    // Mistral expects the previous assistant turn to end with `</s>`
    // (the EOS we caught last turn but didn't commit). Replay it.
    let eos = tokenizer.eos_id;

    let mut tokens = Vec::new();
    tokens.push(eos);
    tokens.push(inst_open);
    tokens.extend(tokenizer.encode_no_prefix(&format!(" {}", user)));
    tokens.push(inst_close);
    tokens
}

// ---------- Raw / no-template ----------

fn render_raw_first(tokenizer: &Tokenizer, system: &str, user: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_id {
        tokens.push(bos);
    }
    if !system.is_empty() {
        tokens.extend(tokenizer.encode(system));
        tokens.extend(tokenizer.encode("\n"));
    }
    tokens.extend(tokenizer.encode(user));
    tokens
}

fn render_raw_continuation(tokenizer: &Tokenizer, user: &str) -> Vec<u32> {
    tokenizer.encode(user)
}
