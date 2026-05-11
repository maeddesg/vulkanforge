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
    /// Sprint 43B-2 — Gemma-4 chat form (asymmetric `<|turn>` /
    /// `<turn|>` boundary tokens, completely different from the
    /// `<start_of_turn>` / `<end_of_turn>` of Gemma-1/2/3):
    /// `<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n`.
    Gemma4,
    /// Sprint 51D-K — Gemma-4 variant that appends
    /// `<|channel>thought\n<channel|>` after the assistant header when
    /// `add_generation_prompt=true && enable_thinking=false` (the
    /// upstream-jinja default for Gemma-4-26B-A4B). Without these 4
    /// tokens the model sees a malformed prompt and collapses on
    /// `<|channel>` (id 100). E2B's chat_template.jinja does NOT emit
    /// the block; 26B's does.
    Gemma4WithThoughtChannel,
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
            Some(TokenizerFlavour::Gemma2) => ChatTemplate::Gemma4,
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
        // Sprint 51D-K — Gemma-4 ships its chat template as a separate
        // `chat_template.jinja` file rather than embedding it in
        // `tokenizer_config.json`. The `<|channel>thought<channel|>`
        // suffix lives only in there (added by the 26B variant when
        // `add_generation_prompt=true && enable_thinking=false`), so
        // we read both and concatenate before sniffing.
        let jinja_path = model_dir.join("chat_template.jinja");
        let jinja_str = std::fs::read_to_string(&jinja_path).unwrap_or_default();
        let template_str = if !jinja_str.is_empty() {
            format!("{template_str}\n{jinja_str}")
        } else {
            template_str
        };

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
        // Sprint 43B-2 — Gemma-4's chat template is large (~16 KB
        // Jinja with macros). Two heuristics catch it: the asymmetric
        // `<|turn>` / `<turn|>` boundary tokens are unique to Gemma-4
        // (Gemma-1/2/3 use `<start_of_turn>` / `<end_of_turn>`).
        // The chat_template may not always be in tokenizer_config.json
        // (Gemma-4 ships it as a separate `chat_template.jinja` file)
        // — fall through to the flavour check below in that case.
        if template_str.contains("<|turn>") && template_str.contains("<turn|>") {
            // Sprint 51D-K — sub-detect the Gemma-4-26B-A4B variant
            // that appends `<|channel>thought\n<channel|>` after the
            // assistant header. Match the LITERAL jinja string
            // `'<|channel>thought\n<channel|>'` (single concatenated
            // form, no `+ thinking_text +`). E2B's chat_template only
            // contains the concatenated form
            // `'<|channel>thought\n' + thinking_text + '\n<channel|>'`
            // for explicit-thinking-history paths, never the
            // single-string form — so the substring discriminates.
            if template_str.contains("<|channel>thought\\n<channel|>") {
                return ChatTemplate::Gemma4WithThoughtChannel;
            }
            return ChatTemplate::Gemma4;
        }
        match tokenizer.flavour() {
            Some(TokenizerFlavour::Qwen2) => ChatTemplate::ChatML,
            Some(TokenizerFlavour::Llama3) => ChatTemplate::Llama3,
            Some(TokenizerFlavour::Gemma2) => ChatTemplate::Gemma4,
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
            ChatTemplate::Gemma4 => render_gemma4_first(tokenizer, system, user, false),
            ChatTemplate::Gemma4WithThoughtChannel
                => render_gemma4_first(tokenizer, system, user, true),
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
            ChatTemplate::Gemma4 => render_gemma4_continuation(tokenizer, user, false),
            ChatTemplate::Gemma4WithThoughtChannel
                => render_gemma4_continuation(tokenizer, user, true),
            ChatTemplate::Raw => render_raw_continuation(tokenizer, user),
        }
    }

    /// v0.4 Sprint 5 — render the entire `messages[]` history from
    /// scratch in one shot. Used by the stateless OpenAI-compatible
    /// API server, which resets the KV cache per request rather than
    /// carrying state across requests like the interactive CLI does.
    ///
    /// `messages` MUST end with a `User` entry — the caller of this
    /// function is responsible for that check; this function appends
    /// the assistant generation-prompt header after the final user
    /// turn so the model knows it's its turn to speak.
    pub fn render_full_history(
        self,
        tokenizer: &Tokenizer,
        messages: &[RenderMessage<'_>],
    ) -> Vec<u32> {
        match self {
            ChatTemplate::ChatML => render_chatml_full(tokenizer, messages),
            ChatTemplate::Llama3 => render_llama3_full(tokenizer, messages),
            ChatTemplate::DeepSeekR1 => render_deepseek_full(tokenizer, messages),
            ChatTemplate::Mistral => render_mistral_full(tokenizer, messages),
            ChatTemplate::Gemma4 => render_gemma4_full(tokenizer, messages, false),
            ChatTemplate::Gemma4WithThoughtChannel
                => render_gemma4_full(tokenizer, messages, true),
            ChatTemplate::Raw => render_raw_full(tokenizer, messages),
        }
    }
}

/// One message in the conversation history. Borrows the content so
/// the caller can hold off on string-allocations until the rendered
/// token vector is built.
#[derive(Debug, Clone, Copy)]
pub struct RenderMessage<'a> {
    pub role: HistoryRole,
    pub content: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRole {
    System,
    User,
    Assistant,
}

// ---------- Gemma-4 (Sprint 43B-2) ----------
//
// Format extracted from the upstream `chat_template.jinja`
// (lines 179–344): three boundary forms,
//
//   `<|turn>system\n{system}<turn|>\n`
//   `<|turn>user\n{user}<turn|>\n`
//   `<|turn>model\n{model}<turn|>\n`
//
// The system block is *optional* per the upstream guard
// `if enable_thinking or tools or messages[0]['role'] in
//  ['system', 'developer']` — we always emit it when the caller
// passes a non-empty system prompt and skip it otherwise.

fn gemma4_special(tokenizer: &Tokenizer, name: &str) -> u32 {
    tokenizer
        .special_id(name)
        .unwrap_or_else(|| panic!("Gemma-4 chat template missing special token {name:?}"))
}

/// Sprint 51D-K — append the `<|channel>thought\n<channel|>` block
/// that 26B's chat_template.jinja emits when
/// `add_generation_prompt=true && enable_thinking=false`. Without
/// this 4-token suffix the model sees a malformed prompt and
/// collapses on `<|channel>` (id 100).
fn append_gemma4_thought_channel(tokenizer: &Tokenizer, tokens: &mut Vec<u32>) {
    tokens.push(gemma4_special(tokenizer, "<|channel>"));
    tokens.extend(tokenizer.encode("thought\n"));
    tokens.push(gemma4_special(tokenizer, "<channel|>"));
}

fn render_gemma4_first(
    tokenizer: &Tokenizer,
    system: &str,
    user: &str,
    with_thought_channel: bool,
) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or_else(|| gemma4_special(tokenizer, "<bos>"));
    let sot = gemma4_special(tokenizer, "<|turn>");
    let eot = gemma4_special(tokenizer, "<turn|>");

    let mut tokens = Vec::new();
    tokens.push(bos);
    if !system.is_empty() {
        tokens.push(sot);
        tokens.extend(tokenizer.encode("system\n"));
        tokens.extend(tokenizer.encode(system));
        tokens.push(eot);
        tokens.extend(tokenizer.encode("\n"));
    }
    tokens.push(sot);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(eot);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(sot);
    tokens.extend(tokenizer.encode("model\n"));
    if with_thought_channel {
        append_gemma4_thought_channel(tokenizer, &mut tokens);
    }
    tokens
}

fn render_gemma4_continuation(
    tokenizer: &Tokenizer,
    user: &str,
    with_thought_channel: bool,
) -> Vec<u32> {
    let sot = gemma4_special(tokenizer, "<|turn>");
    let eot = gemma4_special(tokenizer, "<turn|>");

    let mut tokens = Vec::new();
    // The previous turn's `<turn|>` may not have been committed yet
    // (the caller may have stopped at it as EOS). Replay it here so
    // the model sees a clean turn-boundary before the new user
    // message.
    tokens.push(eot);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(sot);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user));
    tokens.push(eot);
    tokens.extend(tokenizer.encode("\n"));
    tokens.push(sot);
    tokens.extend(tokenizer.encode("model\n"));
    if with_thought_channel {
        append_gemma4_thought_channel(tokenizer, &mut tokens);
    }
    tokens
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

// =========================================================================
// v0.4 Sprint 5 — full-history renderers (stateless server path)
// =========================================================================
//
// Each `render_*_full` walks the entire `messages[]` array and emits
// the same on-the-wire form HuggingFace's `apply_chat_template` would
// produce, ending with the assistant generation-prompt header. The
// resulting `Vec<u32>` is what the GPU prefills against position 0
// after a `kv_cache.reset()`.
//
// All renderers assume the last message has `role: User` (validated
// by the caller — `server::handlers::chat::validate_and_normalise`).
// Empty system content is treated as "no system message".

/// Pick the first system message (if any) from the head of the list.
/// All other messages (user/assistant alternation) follow.
fn split_system<'a>(messages: &'a [RenderMessage<'a>]) -> (Option<&'a str>, &'a [RenderMessage<'a>]) {
    if let Some(first) = messages.first() {
        if first.role == HistoryRole::System {
            return (Some(first.content), &messages[1..]);
        }
    }
    (None, messages)
}

// ---------- ChatML (Qwen3 / generic) full ----------

fn render_chatml_full(tokenizer: &Tokenizer, messages: &[RenderMessage<'_>]) -> Vec<u32> {
    let im_start = tokenizer.im_start_id.expect("ChatML render needs <|im_start|>");
    let im_end = tokenizer.im_end_id.expect("ChatML render needs <|im_end|>");

    let mut tokens = Vec::new();
    let (system, rest) = split_system(messages);
    if let Some(sys) = system {
        tokens.push(im_start);
        tokens.extend(tokenizer.encode("system\n"));
        tokens.extend(tokenizer.encode(sys));
        tokens.push(im_end);
        tokens.extend(tokenizer.encode("\n"));
    }
    for msg in rest {
        let role_str = match msg.role {
            HistoryRole::User => "user\n",
            HistoryRole::Assistant => "assistant\n",
            HistoryRole::System => continue, // duplicate system → ignored
        };
        tokens.push(im_start);
        tokens.extend(tokenizer.encode(role_str));
        tokens.extend(tokenizer.encode(msg.content));
        tokens.push(im_end);
        tokens.extend(tokenizer.encode("\n"));
    }
    // Generation prompt: assistant header without content.
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("assistant\n"));
    tokens
}

// ---------- Llama-3 full ----------

fn render_llama3_full(tokenizer: &Tokenizer, messages: &[RenderMessage<'_>]) -> Vec<u32> {
    let bot = llama3_special(tokenizer, "<|begin_of_text|>");
    let start_h = llama3_special(tokenizer, "<|start_header_id|>");
    let end_h = llama3_special(tokenizer, "<|end_header_id|>");
    let eot = llama3_special(tokenizer, "<|eot_id|>");

    let mut tokens = Vec::new();
    tokens.push(bot);
    let (system, rest) = split_system(messages);
    if let Some(sys) = system {
        tokens.push(start_h);
        tokens.extend(tokenizer.encode("system"));
        tokens.push(end_h);
        tokens.extend(tokenizer.encode("\n\n"));
        tokens.extend(tokenizer.encode(sys));
        tokens.push(eot);
    }
    for msg in rest {
        let role_str = match msg.role {
            HistoryRole::User => "user",
            HistoryRole::Assistant => "assistant",
            HistoryRole::System => continue,
        };
        tokens.push(start_h);
        tokens.extend(tokenizer.encode(role_str));
        tokens.push(end_h);
        tokens.extend(tokenizer.encode("\n\n"));
        tokens.extend(tokenizer.encode(msg.content));
        tokens.push(eot);
    }
    // Generation prompt.
    tokens.push(start_h);
    tokens.extend(tokenizer.encode("assistant"));
    tokens.push(end_h);
    tokens.extend(tokenizer.encode("\n\n"));
    tokens
}

// ---------- Gemma-4 full ----------

fn render_gemma4_full(
    tokenizer: &Tokenizer,
    messages: &[RenderMessage<'_>],
    with_thought_channel: bool,
) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or_else(|| gemma4_special(tokenizer, "<bos>"));
    let sot = gemma4_special(tokenizer, "<|turn>");
    let eot = gemma4_special(tokenizer, "<turn|>");

    let mut tokens = Vec::new();
    tokens.push(bos);
    let (system, rest) = split_system(messages);
    if let Some(sys) = system {
        tokens.push(sot);
        tokens.extend(tokenizer.encode("system\n"));
        tokens.extend(tokenizer.encode(sys));
        tokens.push(eot);
        tokens.extend(tokenizer.encode("\n"));
    }
    for msg in rest {
        let role_str = match msg.role {
            HistoryRole::User => "user\n",
            HistoryRole::Assistant => "model\n",
            HistoryRole::System => continue,
        };
        tokens.push(sot);
        tokens.extend(tokenizer.encode(role_str));
        tokens.extend(tokenizer.encode(msg.content));
        tokens.push(eot);
        tokens.extend(tokenizer.encode("\n"));
    }
    // Generation prompt.
    tokens.push(sot);
    tokens.extend(tokenizer.encode("model\n"));
    if with_thought_channel {
        append_gemma4_thought_channel(tokenizer, &mut tokens);
    }
    tokens
}

// ---------- DeepSeek-R1 full ----------

fn render_deepseek_full(tokenizer: &Tokenizer, messages: &[RenderMessage<'_>]) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or_else(|| {
        tokenizer.special_id("<｜begin▁of▁sentence｜>").unwrap_or(0)
    });
    let user_tok = tokenizer
        .special_id("<｜User｜>")
        .expect("DeepSeek-R1 chat template missing <｜User｜>");
    let asst_tok = tokenizer
        .special_id("<｜Assistant｜>")
        .expect("DeepSeek-R1 chat template missing <｜Assistant｜>");
    let eos = tokenizer.special_id("<｜end▁of▁sentence｜>");

    let mut tokens = Vec::new();
    tokens.push(bos);
    let (system, rest) = split_system(messages);
    if let Some(sys) = system {
        tokens.extend(tokenizer.encode(sys));
    }
    for msg in rest {
        match msg.role {
            HistoryRole::User => {
                tokens.push(user_tok);
                tokens.extend(tokenizer.encode(msg.content));
            }
            HistoryRole::Assistant => {
                tokens.push(asst_tok);
                tokens.extend(tokenizer.encode(msg.content));
                // The model normally emits EOS after a finished
                // assistant turn; in a stateless replay we have to
                // insert it ourselves so the next user-turn boundary
                // matches the training distribution.
                if let Some(id) = eos {
                    tokens.push(id);
                }
            }
            HistoryRole::System => continue,
        }
    }
    // Generation prompt with R1's mandatory `<think>` priming.
    tokens.push(asst_tok);
    tokens.extend(tokenizer.encode("<think>\n"));
    tokens
}

// ---------- Mistral Instruct full ----------

fn render_mistral_full(tokenizer: &Tokenizer, messages: &[RenderMessage<'_>]) -> Vec<u32> {
    let bos = tokenizer.bos_id.unwrap_or(1);
    let inst_open = mistral_special(tokenizer, "[INST]");
    let inst_close = mistral_special(tokenizer, "[/INST]");
    let eos = tokenizer.eos_id;

    let mut tokens = Vec::new();
    tokens.push(bos);

    // Mistral's HF template ignores system role unless wrapped inside
    // the FIRST user message. Pre-extract the system text; we'll
    // splice it into the first user turn.
    let (system, rest) = split_system(messages);
    let system_owned = system.unwrap_or("");

    let mut first_user_done = false;
    let mut i = 0;
    while i < rest.len() {
        let msg = rest[i];
        match msg.role {
            HistoryRole::User => {
                if !first_user_done && !system_owned.is_empty() {
                    let body = format!(" {}\n\n{}", system_owned, msg.content);
                    tokens.push(inst_open);
                    tokens.extend(tokenizer.encode_no_prefix(&body));
                    tokens.push(inst_close);
                } else {
                    tokens.push(inst_open);
                    tokens.extend(tokenizer.encode_no_prefix(&format!(" {}", msg.content)));
                    tokens.push(inst_close);
                }
                first_user_done = true;
            }
            HistoryRole::Assistant => {
                tokens.extend(tokenizer.encode(msg.content));
                tokens.push(eos);
            }
            HistoryRole::System => {}
        }
        i += 1;
    }
    tokens
}

// ---------- Raw / no-template full ----------

fn render_raw_full(tokenizer: &Tokenizer, messages: &[RenderMessage<'_>]) -> Vec<u32> {
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_id {
        tokens.push(bos);
    }
    for msg in messages {
        if matches!(msg.role, HistoryRole::System) && msg.content.is_empty() {
            continue;
        }
        tokens.extend(tokenizer.encode(msg.content));
        tokens.extend(tokenizer.encode("\n"));
    }
    tokens
}
