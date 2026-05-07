//! Tokenizer dispatch — wraps the GPT-2 byte-level BPE (Qwen3,
//! Llama-3 / DeepSeek family) and the SentencePiece Unigram path
//! (Mistral, Llama-2) behind a single API. Both are selected from
//! `tokenizer.ggml.model` at load time and surface the same
//! `encode`/`decode`/`is_eos`/`special_id` surface so callers don't
//! need to know which one they have.
//!
//! BPE pipeline (encode):
//!   text (UTF-8)
//!     → pre-split via the architecture-specific PAT regex
//!     → for each chunk: byte-level encode (byte → unicode char)
//!     → BPE-merge using the merges table (lowest priority wins)
//!     → vocab lookup → token ids
//!
//! SPM pipeline (encode):
//!   text (UTF-8)
//!     → SPM normalise (prepend ▁, swap ' ' for ▁)
//!     → Viterbi best-path over the unigram vocabulary, with
//!       byte-fallback as a competing edge per char.
//!
//! Special tokens are not scanned for inside `encode()`. Callers that
//! need them (chat-template application) emit the ids directly via
//! [`Tokenizer::special_id`] (or the convenience `im_start_id` /
//! `im_end_id` accessors which return the Qwen3 ids when present),
//! and call `encode()` only on the regular text between them.

use std::collections::HashMap;
use std::path::Path;

use fancy_regex::Regex;

use super::gguf::GgufFile;
use super::spm::SpmTokenizer;

/// Qwen2/3 pre-tokenizer regex (matches the canonical `tokenizer.json`
/// PAT_STR). Uses `\p{L}`, `\p{N}` for Unicode letter/number classes
/// and a `(?!\S)` negative lookahead for the trailing-whitespace rule.
const QWEN2_PRE_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)\
|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\
|\\p{N}+\
| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\
|\\s*[\\r\\n]+\
|\\s+(?!\\S)\
|\\s+";

/// Llama-3 pre-tokenizer regex. Differs from Qwen2 in two ways:
///   * digit runs are split into chunks of length 1..=3 (not greedy),
///   * the apostrophe contractions are case-insensitive only on ASCII
///     (we keep `(?i:…)` — same observable behaviour for ASCII).
/// Same PAT used by `tiktoken_bpe`'s `cl100k_base` is the basis but
/// llama-bpe matches the exact form used in `tokenizer.json` for
/// Meta-Llama-3.
const LLAMA3_PRE_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)\
|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\
|\\p{N}{1,3}\
| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\
|\\s*[\\r\\n]+\
|\\s+(?!\\S)\
|\\s+";

#[derive(Debug)]
pub enum TokenizerError {
    MissingMetadata(&'static str),
    UnexpectedType(&'static str),
    BadModel(String),
    BadMerge(String),
    BadRegex(String),
    UnknownToken(String),
    UnmappableChar(char),
    Malformed(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::MissingMetadata(k) => write!(f, "missing metadata key '{k}'"),
            TokenizerError::UnexpectedType(k) => write!(f, "metadata key '{k}' has wrong type"),
            TokenizerError::BadModel(m) => write!(f, "unsupported tokenizer model: {m}"),
            TokenizerError::BadMerge(s) => write!(f, "malformed merge '{s}'"),
            TokenizerError::BadRegex(e) => write!(f, "regex compile failed: {e}"),
            TokenizerError::UnknownToken(s) => write!(f, "BPE produced unknown token '{s}'"),
            TokenizerError::UnmappableChar(c) => {
                write!(f, "decode hit non-byte-encoded char '{}' (u+{:04x})", c, *c as u32)
            }
            TokenizerError::Malformed(s) => write!(f, "malformed tokenizer data: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Which pre-tokenizer / special-token namespace the GPT-2 BPE was
/// built for. Only meaningful when [`Tokenizer::flavour`] returns
/// `Some(_)`; SPM-flavoured models return `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFlavour {
    /// Qwen2 / Qwen3 — `pre="qwen2"`. ChatML special tokens.
    Qwen2,
    /// Llama-3 family — `pre="llama-bpe"`. Header-id / eot_id specials.
    Llama3,
    /// Sprint 43B-2 — Gemma-4 family. `▁`-prefixed SP-style BPE with
    /// `byte_fallback: true`. Special tokens are `<bos>` (id=2),
    /// `<eos>` (id=1), `<|turn>` / `<turn|>` for chat boundaries,
    /// plus the multimodal markers we don't consume yet.
    Gemma2,
}

/// Internal storage discriminator.
enum TokenizerInner {
    Bpe(BpeData),
    Spm(SpmTokenizer),
}

/// GPT-2 byte-level BPE state.
struct BpeData {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// concatenated-pair → priority (lower = applied first).
    merges: HashMap<String, u32>,
    pre_split: Regex,
    flavour: TokenizerFlavour,
    /// GPT-2 byte-level encoding tables. `byte_to_char[b]` is the
    /// unicode char that represents byte `b` in the vocab.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

pub struct Tokenizer {
    inner: TokenizerInner,
    pub bos_id: Option<u32>,
    /// Primary EOS id from `tokenizer.ggml.eos_token_id` — varies per
    /// model (Qwen3=151645 `<|im_end|>`, Llama-3.1=128009 `<|eot_id|>`,
    /// DeepSeek-R1-Distill-Llama=128001 `<|end_of_text|>`,
    /// Mistral=2 `</s>`).
    pub eos_id: u32,
    /// Extra ids treated as end-of-stream by `is_eos`. For Qwen3 this
    /// holds `<|endoftext|>`; for Llama-3 it holds the alternates from
    /// the chat-template (e.g. `<|eom_id|>`); empty for SPM.
    extra_eos_ids: Vec<u32>,
    /// Qwen3 ChatML ids. Populated for `Qwen2` flavour, `None`
    /// otherwise. Kept here so the ChatML chat-template code can stay
    /// terse; non-Qwen callers go through [`Tokenizer::special_id`].
    pub im_start_id: Option<u32>,
    pub im_end_id: Option<u32>,
    pub endoftext_id: Option<u32>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.model"))?;
        match model {
            "gpt2" => Self::from_gguf_bpe(gguf),
            "llama" => Self::from_gguf_spm(gguf),
            other => Err(TokenizerError::BadModel(other.to_string())),
        }
    }

    fn from_gguf_bpe(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let pre = gguf
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let flavour = match pre {
            "qwen2" => TokenizerFlavour::Qwen2,
            "llama-bpe" => TokenizerFlavour::Llama3,
            other => return Err(TokenizerError::BadModel(format!("pre={other}"))),
        };
        let pre_regex = match flavour {
            TokenizerFlavour::Qwen2 => QWEN2_PRE_REGEX,
            TokenizerFlavour::Llama3 => LLAMA3_PRE_REGEX,
            // Sprint 43B-2 — Gemma-4 isn't reachable from the GGUF
            // path (no llama.cpp Gemma-4 support); arm exists for
            // exhaustiveness only.
            TokenizerFlavour::Gemma2 => "[\\s\\S]+",
        };

        // Vocab — Array<string>.
        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.tokens"))?
            .as_array()
            .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.tokens"))?;
        let mut vocab: Vec<String> = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id: HashMap<String, u32> =
            HashMap::with_capacity(tokens_arr.len());
        for (i, v) in tokens_arr.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.tokens[i]"))?
                .to_string();
            token_to_id.insert(s.clone(), i as u32);
            vocab.push(s);
        }

        // Merges — Array<string>, "left right".
        let merges_arr = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.merges"))?
            .as_array()
            .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.merges"))?;
        let mut merges: HashMap<String, u32> = HashMap::with_capacity(merges_arr.len());
        for (rank, v) in merges_arr.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.merges[i]"))?;
            let space = s
                .find(' ')
                .ok_or_else(|| TokenizerError::BadMerge(s.to_string()))?;
            // The two halves never contain a literal ' ' (space is
            // encoded as Ġ in byte-level form), so concatenation is
            // unambiguous.
            let left = &s[..space];
            let right = &s[space + 1..];
            let mut joined = String::with_capacity(left.len() + right.len());
            joined.push_str(left);
            joined.push_str(right);
            merges.entry(joined).or_insert(rank as u32);
        }

        // Special-token ids — looked up by name so the Qwen3 GGUF
        // quirk (`eos_token_id = <|im_end|>`, `bos_token_id =
        // <|endoftext|>`) doesn't bleed into our naming.
        let bos_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.eos_token_id"))?;
        let endoftext_id = token_to_id.get("<|endoftext|>").copied();
        let im_start_id = token_to_id.get("<|im_start|>").copied();
        let im_end_id = token_to_id.get("<|im_end|>").copied();

        // For Llama-3 we want both `<|eot_id|>` (turn end) and
        // `<|end_of_text|>` (sequence end) to terminate decode; the
        // primary `eos_id` covers one, the other goes into extras.
        let mut extra_eos_ids: Vec<u32> = Vec::new();
        match flavour {
            TokenizerFlavour::Qwen2 => {
                if let Some(et) = endoftext_id {
                    if et != eos_id {
                        extra_eos_ids.push(et);
                    }
                }
            }
            TokenizerFlavour::Llama3 => {
                for name in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"] {
                    if let Some(&id) = token_to_id.get(name) {
                        if id != eos_id && !extra_eos_ids.contains(&id) {
                            extra_eos_ids.push(id);
                        }
                    }
                }
            }
            // Sprint 43B-2 — Gemma-4 isn't reachable through the
            // GGUF path (no llama.cpp Gemma-4 support yet); arm
            // listed for exhaustiveness only.
            TokenizerFlavour::Gemma2 => {}
        }

        let (byte_to_char, char_to_byte) = build_byte_unicode_tables();

        let pre_split = Regex::new(pre_regex)
            .map_err(|e| TokenizerError::BadRegex(format!("{e}")))?;

        let inner = TokenizerInner::Bpe(BpeData {
            vocab,
            token_to_id,
            merges,
            pre_split,
            flavour,
            byte_to_char,
            char_to_byte,
        });

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            extra_eos_ids,
            im_start_id,
            im_end_id,
            endoftext_id,
        })
    }

    /// v0.3.13 — load a HuggingFace BPE tokenizer directly from
    /// `<model_dir>/tokenizer.json` (+ `tokenizer_config.json` for the
    /// special-token literals). Builds the *same* internal `BpeData`
    /// struct as `from_gguf_bpe`, so all the existing chat-template
    /// renderers (`ChatTemplate::render_*`) keep working unchanged.
    ///
    /// Two upstream `merges` formats are handled:
    ///   * `["Ġ Ġ", "Ġ ĠĠĠ", …]` — space-joined strings (Llama-3.1).
    ///   * `[["Ġ","Ġ"], ["ĠĠ","ĠĠ"], …]` — pair lists (Qwen3 / newer
    ///     `tokenizers` crate output).
    ///
    /// Flavour is detected from the presence of `<|im_start|>`
    /// (Qwen2 / Qwen3 ChatML) or `<|begin_of_text|>` (Llama-3). SPM
    /// SafeTensors models (Mistral) aren't yet covered — they'd need
    /// a different upstream payload (`tokenizer.model` SentencePiece
    /// proto) and aren't in the v0.3.13 scope.
    pub fn from_hf_dir(model_dir: &Path) -> Result<Self, TokenizerError> {
        let tjson_path = model_dir.join("tokenizer.json");
        let tjson_bytes = std::fs::read(&tjson_path).map_err(|e| {
            TokenizerError::Malformed(format!(
                "read {}: {e}",
                tjson_path.display()
            ))
        })?;
        let tjson: serde_json::Value = serde_json::from_slice(&tjson_bytes)
            .map_err(|e| TokenizerError::Malformed(format!("parse tokenizer.json: {e}")))?;

        let model = tjson
            .get("model")
            .ok_or_else(|| TokenizerError::Malformed("tokenizer.json: missing model".into()))?;
        let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if model_type != "BPE" {
            return Err(TokenizerError::BadModel(format!(
                "tokenizer.json model.type = {model_type:?}, only BPE supported"
            )));
        }

        // Vocab — `model.vocab` is `{token_string: id}`. Some IDs may
        // be claimed by `added_tokens` instead (specials live there
        // for Qwen3). Allocate a `vocab` Vec sized for `max_id + 1`
        // and fill from both sources.
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TokenizerError::Malformed("model.vocab missing or not object".into()))?;
        let added: Vec<(u32, String)> = tjson
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let id = t.get("id")?.as_u64()? as u32;
                        let content = t.get("content")?.as_str()?.to_string();
                        Some((id, content))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut max_id: u32 = 0;
        for v in vocab_obj.values() {
            if let Some(i) = v.as_u64() {
                max_id = max_id.max(i as u32);
            }
        }
        for (id, _) in &added {
            max_id = max_id.max(*id);
        }
        let mut vocab: Vec<String> = vec![String::new(); (max_id + 1) as usize];
        let mut token_to_id: HashMap<String, u32> =
            HashMap::with_capacity(vocab_obj.len() + added.len());
        for (k, v) in vocab_obj.iter() {
            let id = v
                .as_u64()
                .ok_or_else(|| TokenizerError::Malformed("vocab id not u64".into()))?
                as u32;
            vocab[id as usize] = k.clone();
            token_to_id.insert(k.clone(), id);
        }
        for (id, content) in &added {
            // added_tokens beats vocab on collision (specials must
            // own their slot).
            vocab[*id as usize] = content.clone();
            token_to_id.insert(content.clone(), *id);
        }

        // Merges — handle both string and array forms.
        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TokenizerError::Malformed("model.merges missing".into()))?;
        let mut merges: HashMap<String, u32> = HashMap::with_capacity(merges_arr.len());
        for (rank, m) in merges_arr.iter().enumerate() {
            let (left, right) = if let Some(s) = m.as_str() {
                let space = s
                    .find(' ')
                    .ok_or_else(|| TokenizerError::BadMerge(s.to_string()))?;
                (&s[..space], &s[space + 1..])
            } else if let Some(arr) = m.as_array() {
                let l = arr
                    .first()
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| TokenizerError::BadMerge("merge[].0 missing".into()))?;
                let r = arr
                    .get(1)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| TokenizerError::BadMerge("merge[].1 missing".into()))?;
                (l, r)
            } else {
                return Err(TokenizerError::BadMerge(format!(
                    "merge[{rank}] not str or array"
                )));
            };
            let mut joined = String::with_capacity(left.len() + right.len());
            joined.push_str(left);
            joined.push_str(right);
            merges.entry(joined).or_insert(rank as u32);
        }

        // Flavour from added_tokens. Both Qwen2/Qwen3 use ChatML
        // specials, so a single `<|im_start|>` check covers both.
        // Sprint 43B-2: Gemma-4 detected via the `<|turn>` chat
        // boundary marker (asymmetric tag, unique to Gemma-4).
        let flavour = if token_to_id.contains_key("<|im_start|>") {
            TokenizerFlavour::Qwen2
        } else if token_to_id.contains_key("<|begin_of_text|>") {
            TokenizerFlavour::Llama3
        } else if token_to_id.contains_key("<|turn>") && token_to_id.contains_key("<turn|>") {
            TokenizerFlavour::Gemma2
        } else {
            return Err(TokenizerError::BadModel(
                "HF BPE tokenizer flavour unknown (no <|im_start|>, no <|begin_of_text|>, no <|turn>)"
                    .into(),
            ));
        };
        let pre_regex = match flavour {
            TokenizerFlavour::Qwen2 => QWEN2_PRE_REGEX,
            TokenizerFlavour::Llama3 => LLAMA3_PRE_REGEX,
            // Gemma-4 doesn't actually need a pre-split regex — its
            // pre_tokenizer is `Split` on " " with `MergedWithPrevious`
            // semantics (each word inherits its leading space as `▁`).
            // We give the regex a permissive default so the BpeData
            // struct stays uniform; the encode path branches on
            // flavour and bypasses the regex entirely.
            TokenizerFlavour::Gemma2 => "[\\s\\S]+",
        };
        let pre_split = Regex::new(pre_regex)
            .map_err(|e| TokenizerError::BadRegex(format!("{e}")))?;

        // bos / eos literals from tokenizer_config.json. The fields
        // can be either a plain string or an object `{ "content": …,
        // "lstrip": …, … }`.
        let tcfg_path = model_dir.join("tokenizer_config.json");
        let lookup_token_str = |val: &serde_json::Value| -> Option<String> {
            if let Some(s) = val.as_str() {
                return Some(s.to_string());
            }
            if let Some(obj) = val.as_object() {
                return obj.get("content").and_then(|v| v.as_str()).map(String::from);
            }
            None
        };
        let (bos_id, eos_id_opt) = match std::fs::read(&tcfg_path) {
            Ok(bytes) => {
                let tcfg: serde_json::Value =
                    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
                let bos = tcfg
                    .get("bos_token")
                    .and_then(lookup_token_str)
                    .and_then(|s| token_to_id.get(&s).copied());
                let eos = tcfg
                    .get("eos_token")
                    .and_then(lookup_token_str)
                    .and_then(|s| token_to_id.get(&s).copied());
                (bos, eos)
            }
            Err(_) => (None, None),
        };
        // EOS fallback by flavour if tokenizer_config.json didn't
        // pin one (some upstream models leave eos_token unset).
        let eos_id = eos_id_opt
            .or_else(|| match flavour {
                TokenizerFlavour::Qwen2 => token_to_id.get("<|im_end|>").copied(),
                TokenizerFlavour::Llama3 => token_to_id.get("<|eot_id|>").copied(),
                // Gemma-4 ships `<eos>` (id=1) as the canonical end-
                // of-stream and `<turn|>` (id=106) as the per-turn
                // boundary. Default to `<eos>`.
                TokenizerFlavour::Gemma2 => token_to_id.get("<eos>").copied(),
            })
            .ok_or_else(|| {
                TokenizerError::Malformed("no eos_token resolvable from tokenizer_config.json".into())
            })?;

        let endoftext_id = token_to_id.get("<|endoftext|>").copied();
        let im_start_id = token_to_id.get("<|im_start|>").copied();
        let im_end_id = token_to_id.get("<|im_end|>").copied();

        let mut extra_eos_ids: Vec<u32> = Vec::new();
        match flavour {
            TokenizerFlavour::Qwen2 => {
                if let Some(et) = endoftext_id {
                    if et != eos_id {
                        extra_eos_ids.push(et);
                    }
                }
            }
            TokenizerFlavour::Llama3 => {
                for name in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"] {
                    if let Some(&id) = token_to_id.get(name) {
                        if id != eos_id && !extra_eos_ids.contains(&id) {
                            extra_eos_ids.push(id);
                        }
                    }
                }
            }
            // Sprint 43B-2 — Gemma-4: stop on `<turn|>` (per-turn
            // boundary, id=106 in E2B) and `<eos>` (model-level end,
            // id=1). The primary eos_id covers one; the other lands
            // here.
            TokenizerFlavour::Gemma2 => {
                if let Some(&id) = token_to_id.get("<turn|>") {
                    if id != eos_id { extra_eos_ids.push(id); }
                }
                if let Some(&id) = token_to_id.get("<eos>") {
                    if id != eos_id && !extra_eos_ids.contains(&id) {
                        extra_eos_ids.push(id);
                    }
                }
            }
        }

        let (byte_to_char, char_to_byte) = build_byte_unicode_tables();

        let inner = TokenizerInner::Bpe(BpeData {
            vocab,
            token_to_id,
            merges,
            pre_split,
            flavour,
            byte_to_char,
            char_to_byte,
        });

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            extra_eos_ids,
            im_start_id,
            im_end_id,
            endoftext_id,
        })
    }

    fn from_gguf_spm(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let spm = SpmTokenizer::from_gguf(gguf)?;
        let bos_id = spm.bos_id;
        let eos_id = spm.eos_id;
        // SPM models don't carry ChatML literals; leave those empty.
        Ok(Self {
            inner: TokenizerInner::Spm(spm),
            bos_id,
            eos_id,
            extra_eos_ids: Vec::new(),
            im_start_id: None,
            im_end_id: None,
            endoftext_id: None,
        })
    }

    /// Returns the BPE flavour, or `None` for SPM-backed tokenizers.
    pub fn flavour(&self) -> Option<TokenizerFlavour> {
        match &self.inner {
            TokenizerInner::Bpe(b) => Some(b.flavour),
            TokenizerInner::Spm(_) => None,
        }
    }

    /// Look up a special-token id by its literal vocab string
    /// (e.g. `"<|begin_of_text|>"` for Llama-3, `"[INST]"` for Mistral).
    pub fn special_id(&self, name: &str) -> Option<u32> {
        match &self.inner {
            TokenizerInner::Bpe(b) => b.token_to_id.get(name).copied(),
            TokenizerInner::Spm(s) => s.special_id(name),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            TokenizerInner::Bpe(b) => b.vocab.len(),
            TokenizerInner::Spm(s) => s.vocab_size(),
        }
    }

    /// Plain-text encode (no special-token scanning). For SPM this
    /// applies the ▁-prefix normalisation; chat-template code that wants
    /// to glue text onto a special token without the leading marker
    /// should call [`Tokenizer::encode_no_prefix`].
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match &self.inner {
            TokenizerInner::Bpe(b) => match b.flavour {
                TokenizerFlavour::Gemma2 => gemma_encode(b, text),
                _ => bpe_encode(b, text),
            },
            TokenizerInner::Spm(s) => s.encode(text),
        }
    }

    /// Encode without prepending the SPM `▁` space marker. Identical to
    /// `encode` for BPE flavours (BPE has no leading-space convention),
    /// but for SPM it skips the implicit leading space — used by chat
    /// templates that already emit a special token whose representation
    /// ends mid-word.
    pub fn encode_no_prefix(&self, text: &str) -> Vec<u32> {
        match &self.inner {
            TokenizerInner::Bpe(b) => match b.flavour {
                TokenizerFlavour::Gemma2 => gemma_encode(b, text),
                _ => bpe_encode(b, text),
            },
            TokenizerInner::Spm(s) => s.encode_no_prefix(text),
        }
    }

    /// Decode a token-id slice. Concatenates the byte-level encoded
    /// vocab strings, then maps each char back to its byte (BPE), or
    /// expands `<0xHH>` byte tokens and replaces `▁` with space (SPM).
    pub fn decode(&self, ids: &[u32]) -> String {
        match &self.inner {
            TokenizerInner::Bpe(b) => match b.flavour {
                TokenizerFlavour::Gemma2 => gemma_decode(b, ids),
                _ => bpe_decode(b, ids),
            },
            TokenizerInner::Spm(s) => s.decode(ids),
        }
    }

    pub fn decode_token(&self, id: u32) -> String {
        match &self.inner {
            TokenizerInner::Bpe(b) => match b.flavour {
                TokenizerFlavour::Gemma2 => gemma_decode(b, &[id]),
                _ => bpe_decode_token(b, id),
            },
            TokenizerInner::Spm(s) => s.decode_token(id),
        }
    }

    /// Sprint 16B — return the raw decoded bytes for one token without
    /// the lossy `String::from_utf8_lossy` step that
    /// [`Self::decode_token`] performs. A single token can be a partial
    /// UTF-8 codepoint (e.g. one byte of a 4-byte emoji), so the
    /// streaming caller is expected to buffer the bytes across tokens
    /// and emit valid UTF-8 prefixes as they complete. SPM's `▁` → ' '
    /// rewrite is applied here (it's intra-token in the vocab and never
    /// straddles a token boundary).
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        match &self.inner {
            TokenizerInner::Bpe(b) => match b.flavour {
                TokenizerFlavour::Gemma2 => {
                    // Gemma decode is "render the vocab string with `▁`
                    // → ' '" + "byte tokens turn into raw bytes". For a
                    // single token we just take the bytes via the joint
                    // decoder.
                    gemma_decode(b, &[id]).into_bytes()
                }
                _ => {
                    let mut buf = Vec::with_capacity(8);
                    bpe_decode_into(b, id, &mut buf);
                    buf
                }
            },
            TokenizerInner::Spm(s) => s.decode_token_bytes(id),
        }
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        match &self.inner {
            TokenizerInner::Bpe(b) => b.vocab.get(id as usize).map(|s| s.as_str()),
            TokenizerInner::Spm(s) => s.token_str(id),
        }
    }

    /// True for the model's primary EOS plus any architecture-specific
    /// alternates (Qwen3 also stops on `<|endoftext|>`; Llama-3 also
    /// stops on `<|end_of_text|>` / `<|eom_id|>`).
    pub fn is_eos(&self, id: u32) -> bool {
        id == self.eos_id || self.extra_eos_ids.contains(&id)
    }

    /// True if this tokenizer is the SPM (SentencePiece Unigram) variant
    /// — Mistral / Llama-2 style. Used by callers that need to choose
    /// between BPE-style and SPM-style chat templating defaults.
    pub fn is_spm(&self) -> bool {
        matches!(self.inner, TokenizerInner::Spm(_))
    }
}

// ---------- BPE encode/decode (free functions over `BpeData`) ----------

fn bpe_encode(b: &BpeData, text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for m in b.pre_split.find_iter(text) {
        let m = match m {
            Ok(x) => x,
            Err(_) => continue,
        };
        if m.start() > cursor {
            bpe_encode_chunk(b, &text[cursor..m.start()], &mut out);
        }
        bpe_encode_chunk(b, m.as_str(), &mut out);
        cursor = m.end();
    }
    if cursor < text.len() {
        bpe_encode_chunk(b, &text[cursor..], &mut out);
    }
    out
}

fn bpe_encode_chunk(b: &BpeData, chunk: &str, out: &mut Vec<u32>) {
    if chunk.is_empty() {
        return;
    }
    let mut byte_chars = String::with_capacity(chunk.len());
    for byte in chunk.as_bytes() {
        byte_chars.push(b.byte_to_char[*byte as usize]);
    }
    let pieces = bpe_merge(b, &byte_chars);
    for p in pieces {
        let id = *b
            .token_to_id
            .get(p.as_str())
            .unwrap_or_else(|| panic!("BPE produced unknown token: {:?}", p));
        out.push(id);
    }
}

/// Iteratively merge adjacent token-strings with the lowest merge
/// priority until no merge applies.
fn bpe_merge(b: &BpeData, word: &str) -> Vec<String> {
    let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    if tokens.len() < 2 {
        return tokens;
    }
    loop {
        let mut best: Option<(usize, u32)> = None;
        for i in 0..tokens.len() - 1 {
            let mut joined =
                String::with_capacity(tokens[i].len() + tokens[i + 1].len());
            joined.push_str(&tokens[i]);
            joined.push_str(&tokens[i + 1]);
            if let Some(&prio) = b.merges.get(&joined) {
                if best.map_or(true, |(_, p)| prio < p) {
                    best = Some((i, prio));
                }
            }
        }
        match best {
            None => break,
            Some((idx, _)) => {
                let right = tokens.remove(idx + 1);
                tokens[idx].push_str(&right);
            }
        }
    }
    tokens
}

fn bpe_decode(b: &BpeData, ids: &[u32]) -> String {
    let mut buf: Vec<u8> = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        bpe_decode_into(b, id, &mut buf);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn bpe_decode_token(b: &BpeData, id: u32) -> String {
    let mut buf = Vec::with_capacity(8);
    bpe_decode_into(b, id, &mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

fn bpe_decode_into(b: &BpeData, id: u32, out: &mut Vec<u8>) {
    let s = match b.vocab.get(id as usize) {
        Some(s) => s,
        None => return,
    };
    // Special tokens like `<|im_end|>` are pure ASCII — every char
    // round-trips through char_to_byte (since ASCII bytes >= 33
    // map to themselves in the byte-level table). Same for `<`,
    // `|`, etc. Falling back to UTF-8 bytes covers any leftover.
    for c in s.chars() {
        if let Some(&byte) = b.char_to_byte.get(&c) {
            out.push(byte);
        } else {
            let mut tmp = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
        }
    }
}

// ---------- Gemma-4 BPE encode/decode (Sprint 43B-2) ----------
//
// Gemma-4's BPE is *not* GPT-2 byte-level. Three differences from the
// `bpe_*` family above:
//
//   1. **No byte-level pre-encoding.** Each Unicode codepoint is
//      either looked up directly in `vocab` (for ASCII / extended
//      Latin / CJK / etc.) or — when not present — falls back to
//      its UTF-8 byte sequence emitted as `<0xHH>` tokens.
//   2. **`▁` (U+2581) marks word boundaries.** Spaces in the input
//      are normalised to `▁` *before* the BPE merge graph runs, so
//      the merge graph naturally combines `▁` + `the` → `▁the`,
//      `▁` + `hello` → `▁hello`.
//   3. **Standard rank-ordered BPE merges** then collapse the
//      char-level pieces into the final vocab tokens.
//
// Decode reverses (1) and (2): tokens whose string starts with `<0x`
// are interpreted as raw bytes; everything else has its `▁` runs
// rewritten back to spaces.
//
// 43B-2 minimal-viable scope: this is enough to encode standard
// chat prompts (English, basic punctuation, the Gemma-4 special-
// turn markers). Esoteric Unicode that isn't in the 262 K vocab as a
// single character will still take the byte-fallback path. We do *not*
// try to escape outermost added-tokens here — the chat template is
// expected to call `special_id()` and `encode()` separately and stitch
// the result.

fn gemma_encode(b: &BpeData, text: &str) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }

    // Step 1: normalise spaces to `▁`. We do this on a `String` rather
    // than streaming because the merge graph operates on full Unicode
    // codepoints — char-by-char, but with stable boundaries.
    let normalised: String = text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }).collect();

    // Step 2: produce a `Vec<String>` of single-codepoint pieces
    // (BPE works at codepoint granularity, not byte granularity, for
    // Gemma's vocab). Codepoints not present in `vocab` get expanded
    // via byte-fallback into `<0xHH>` tokens up front.
    let mut pieces: Vec<String> = Vec::with_capacity(normalised.len());
    for c in normalised.chars() {
        let s = c.to_string();
        if b.token_to_id.contains_key(&s) {
            pieces.push(s);
        } else {
            // Byte-fallback: emit one `<0xHH>` per UTF-8 byte.
            let mut tmp = [0u8; 4];
            let bytes = c.encode_utf8(&mut tmp).as_bytes();
            for &byte in bytes {
                pieces.push(format!("<0x{:02X}>", byte));
            }
        }
    }
    if pieces.is_empty() {
        return Vec::new();
    }

    // Step 3: rank-ordered BPE merge over the codepoint pieces. Same
    // algorithm as `bpe_merge`, just operating on a different starting
    // alphabet — `bpe_merge` takes a single &str and re-splits it into
    // `chars()`, so we'd lose the `<0xHH>` byte tokens that are made
    // of multiple chars each. Re-implement the merge loop here so the
    // initial tokenisation we computed above is preserved.
    let mut tokens = pieces;
    while tokens.len() >= 2 {
        let mut best: Option<(usize, u32)> = None;
        for i in 0..tokens.len() - 1 {
            let mut joined = String::with_capacity(tokens[i].len() + tokens[i + 1].len());
            joined.push_str(&tokens[i]);
            joined.push_str(&tokens[i + 1]);
            if let Some(&prio) = b.merges.get(&joined) {
                if best.map_or(true, |(_, p)| prio < p) {
                    best = Some((i, prio));
                }
            }
        }
        match best {
            None => break,
            Some((idx, _)) => {
                let right = tokens.remove(idx + 1);
                tokens[idx].push_str(&right);
            }
        }
    }

    // Step 4: lookup. Unknown pieces get a single byte-fallback expansion
    // as a last resort (shouldn't happen if the merge graph is well-
    // behaved — every merge result must be in the vocab).
    let mut out = Vec::with_capacity(tokens.len());
    for tok in &tokens {
        if let Some(&id) = b.token_to_id.get(tok.as_str()) {
            out.push(id);
        } else {
            // Last-ditch: re-decompose into bytes and emit fallbacks.
            for byte in tok.as_bytes() {
                let key = format!("<0x{:02X}>", byte);
                if let Some(&id) = b.token_to_id.get(key.as_str()) {
                    out.push(id);
                } else {
                    // Truly unrecoverable — this would be a malformed
                    // vocab. We push 0 (`<pad>` for Gemma) and warn
                    // through the test smoke; the Forward will run
                    // with garbage instead of panicking.
                    out.push(0);
                }
            }
        }
    }
    out
}

fn gemma_decode(b: &BpeData, ids: &[u32]) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
    for &id in ids {
        let s = match b.vocab.get(id as usize) {
            Some(s) => s.as_str(),
            None => continue,
        };
        // `<0xHH>` byte-fallback tokens decode to a single raw byte.
        if s.len() == 6
            && s.as_bytes()[0] == b'<'
            && s.as_bytes()[1] == b'0'
            && s.as_bytes()[2] == b'x'
            && s.as_bytes()[5] == b'>'
        {
            if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
                bytes.push(byte);
                continue;
            }
        }
        // Special tokens like `<bos>`, `<eos>`, `<|turn>`, `<turn|>`
        // are not byte-fallback shapes; they stay verbatim and serve
        // as visible markers if the chat code prints them. The
        // `▁ → space` substitution applies inline.
        for ch in s.chars() {
            if ch == '\u{2581}' {
                bytes.push(b' ');
            } else {
                let mut tmp = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Apply Qwen3's ChatML template — matches the `add_generation_prompt
/// =True` rendering of the Jinja template embedded in the GGUF. For
/// other architectures use [`super::chat_template::apply_chat_template`].
///
/// Panics if invoked on a non-Qwen tokenizer (the ChatML special ids
/// are absent there).
pub fn apply_chat_template(
    tokenizer: &Tokenizer,
    user_message: &str,
    system_prompt: Option<&str>,
) -> Vec<u32> {
    let im_start = tokenizer
        .im_start_id
        .expect("apply_chat_template: tokenizer is not ChatML / Qwen2 flavour");
    let im_end = tokenizer
        .im_end_id
        .expect("apply_chat_template: tokenizer is not ChatML / Qwen2 flavour");

    let system = system_prompt.unwrap_or("You are a helpful assistant.");
    let mut tokens = Vec::new();

    // <|im_start|>system\n{system}<|im_end|>\n
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("system\n"));
    tokens.extend(tokenizer.encode(system));
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));

    // <|im_start|>user\n{user}<|im_end|>\n
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user_message));
    tokens.push(im_end);
    tokens.extend(tokenizer.encode("\n"));

    // <|im_start|>assistant\n  ← model continues from here
    tokens.push(im_start);
    tokens.extend(tokenizer.encode("assistant\n"));

    tokens
}

/// GPT-2 byte-to-unicode mapping. Identical to `bytes_to_unicode()`
/// in `openai/gpt2/encoder.py`. 188 printable bytes (b'!'..='~',
/// b'\xa1'..='\xac', b'\xae'..='\xff') map to themselves; the
/// remaining 68 bytes are mapped to chars 256..324 in the order
/// they're encountered.
fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut taken = [false; 256];
    let printable: [(u8, u8); 3] = [(b'!', b'~'), (0xa1, 0xac), (0xae, 0xff)];
    for &(lo, hi) in &printable {
        for b in lo..=hi {
            byte_to_char[b as usize] = b as char;
            taken[b as usize] = true;
        }
    }
    let mut n: u32 = 0;
    for b in 0u32..256 {
        if !taken[b as usize] {
            let c = char::from_u32(256 + n).expect("valid scalar");
            byte_to_char[b as usize] = c;
            n += 1;
        }
    }
    let mut char_to_byte = HashMap::with_capacity(256);
    for (b, &c) in byte_to_char.iter().enumerate() {
        char_to_byte.insert(c, b as u8);
    }
    (byte_to_char, char_to_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_unicode_roundtrip_covers_all_256() {
        let (b2c, c2b) = build_byte_unicode_tables();
        for b in 0..256 {
            let c = b2c[b];
            let back = *c2b.get(&c).unwrap();
            assert_eq!(back as usize, b, "byte {b} did not round-trip");
        }
    }

    #[test]
    fn space_maps_to_g_dot() {
        let (b2c, _) = build_byte_unicode_tables();
        // Byte 32 (space) → 33rd missing byte → char(256 + 32) = U+0120 = Ġ
        assert_eq!(b2c[32], '\u{0120}');
        // Byte 10 (\n) → 11th missing byte → char(256 + 10) = U+010A = Ċ
        assert_eq!(b2c[10], '\u{010A}');
    }

    #[test]
    fn pre_split_handles_simple_sentence() {
        let re = Regex::new(QWEN2_PRE_REGEX).unwrap();
        let pieces: Vec<&str> = re
            .find_iter("Hello world!")
            .filter_map(|m| m.ok().map(|x| x.as_str()))
            .collect();
        assert_eq!(pieces, vec!["Hello", " world", "!"]);
    }
}
