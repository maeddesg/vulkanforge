//! Byte-level BPE tokenizer for Qwen3 (model="gpt2", pre="qwen2").
//!
//! Phase 2D / Schritt 2.8. The vocabulary, merges, and special-token
//! ids are pulled out of the GGUF metadata table (`tokenizer.ggml.*`).
//!
//! Pipeline (encode):
//!   text (UTF-8)
//!     → pre-split via Qwen2 regex (Unicode-aware, with `(?!\S)` lookahead)
//!     → for each chunk: byte-level encode (byte → unicode char)
//!     → BPE-merge using the merges table (lowest priority wins)
//!     → vocab lookup → token ids
//!
//! Decode is the inverse: token id → vocab byte-encoded string,
//! concatenated across the run, then byte-decoded back to UTF-8.
//!
//! Special tokens (`<|im_start|>`, `<|im_end|>`, …) are not scanned
//! for inside `encode()`. Callers that need them (chat-template
//! application) emit the ids directly via [`Tokenizer::im_start_id`]
//! etc., and call `encode()` only on the regular text between them.

use std::collections::HashMap;

use fancy_regex::Regex;

use super::gguf::GgufFile;

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

#[derive(Debug)]
pub enum TokenizerError {
    MissingMetadata(&'static str),
    UnexpectedType(&'static str),
    BadModel(String),
    BadMerge(String),
    BadRegex(String),
    UnknownToken(String),
    UnmappableChar(char),
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
        }
    }
}

impl std::error::Error for TokenizerError {}

pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// concatenated-pair → priority (lower = applied first).
    merges: HashMap<String, u32>,
    pre_split: Regex,

    /// GPT-2 byte-level encoding tables. `byte_to_char[b]` is the
    /// unicode char that represents byte `b` in the vocab.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,

    pub bos_id: Option<u32>,
    pub endoftext_id: u32,      // <|endoftext|>
    pub im_start_id: u32,       // <|im_start|>
    pub im_end_id: u32,         // <|im_end|>
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(TokenizerError::BadModel(model.to_string()));
        }

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
        let endoftext_id = *token_to_id
            .get("<|endoftext|>")
            .ok_or(TokenizerError::MissingMetadata("<|endoftext|> token"))?;
        let im_start_id = *token_to_id
            .get("<|im_start|>")
            .ok_or(TokenizerError::MissingMetadata("<|im_start|> token"))?;
        let im_end_id = *token_to_id
            .get("<|im_end|>")
            .ok_or(TokenizerError::MissingMetadata("<|im_end|> token"))?;

        let (byte_to_char, char_to_byte) = build_byte_unicode_tables();

        let pre_split = Regex::new(QWEN2_PRE_REGEX)
            .map_err(|e| TokenizerError::BadRegex(format!("{e}")))?;

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            pre_split,
            byte_to_char,
            char_to_byte,
            bos_id,
            endoftext_id,
            im_start_id,
            im_end_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Plain-text encode (no special-token scanning).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        // fancy-regex returns iterator over Result<Match>.
        for m in self.pre_split.find_iter(text) {
            let m = match m { Ok(x) => x, Err(_) => continue };
            // The regex covers the input contiguously when applied via
            // find_iter (with whitespace fallback), but be defensive:
            // skip over any unmatched gap by encoding it as bytes too.
            if m.start() > cursor {
                self.encode_chunk(&text[cursor..m.start()], &mut out);
            }
            self.encode_chunk(m.as_str(), &mut out);
            cursor = m.end();
        }
        if cursor < text.len() {
            self.encode_chunk(&text[cursor..], &mut out);
        }
        out
    }

    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        if chunk.is_empty() {
            return;
        }
        // Byte-level encode each UTF-8 byte to the GPT-2 unicode char.
        let mut byte_chars = String::with_capacity(chunk.len());
        for b in chunk.as_bytes() {
            byte_chars.push(self.byte_to_char[*b as usize]);
        }
        // BPE-merge.
        let pieces = self.bpe(&byte_chars);
        for p in pieces {
            // Vocab lookup must succeed — byte-level BPE guarantees
            // every leaf char is in vocab and the merge tree only
            // produces vocab entries.
            let id = *self
                .token_to_id
                .get(p.as_str())
                .unwrap_or_else(|| panic!("BPE produced unknown token: {:?}", p));
            out.push(id);
        }
    }

    /// Iteratively merge adjacent token-strings with the lowest merge
    /// priority until no merge applies.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if tokens.len() < 2 {
            return tokens;
        }
        loop {
            let mut best: Option<(usize, u32)> = None;
            // Build pair concat once per (i,j) — small allocations are
            // cheap for words of typical length 1..30.
            for i in 0..tokens.len() - 1 {
                let mut joined =
                    String::with_capacity(tokens[i].len() + tokens[i + 1].len());
                joined.push_str(&tokens[i]);
                joined.push_str(&tokens[i + 1]);
                if let Some(&prio) = self.merges.get(&joined) {
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

    /// Decode a token-id slice. Concatenates the byte-level encoded
    /// vocab strings, then maps each char back to its byte.
    /// Non-mappable chars (special-token glyphs like `<|im_end|>`)
    /// pass through verbatim — they are pure ASCII so byte-decoding
    /// is the identity.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut buf: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.decode_into(id, &mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub fn decode_token(&self, id: u32) -> String {
        let mut buf = Vec::with_capacity(8);
        self.decode_into(id, &mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn decode_into(&self, id: u32, out: &mut Vec<u8>) {
        let s = match self.vocab.get(id as usize) {
            Some(s) => s,
            None => return,
        };
        // Special tokens like `<|im_end|>` are pure ASCII — every char
        // round-trips through char_to_byte (since ASCII bytes >= 33
        // map to themselves in the byte-level table). Same for `<`,
        // `|`, etc. Falling back to UTF-8 bytes covers any leftover.
        for c in s.chars() {
            if let Some(&b) = self.char_to_byte.get(&c) {
                out.push(b);
            } else {
                let mut tmp = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
            }
        }
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// True for `<|im_end|>` and `<|endoftext|>` — both end an
    /// assistant turn in Qwen3.
    pub fn is_eos(&self, id: u32) -> bool {
        id == self.im_end_id || id == self.endoftext_id
    }
}

/// Apply Qwen3's chat template manually. We don't run the Jinja2
/// template embedded in the GGUF — for Phase 2 the simple form below
/// matches the `add_generation_prompt=True` rendering that
/// llama.cpp emits.
pub fn apply_chat_template(
    tokenizer: &Tokenizer,
    user_message: &str,
    system_prompt: Option<&str>,
) -> Vec<u32> {
    let system = system_prompt.unwrap_or("You are a helpful assistant.");
    let mut tokens = Vec::new();

    // <|im_start|>system\n{system}<|im_end|>\n
    tokens.push(tokenizer.im_start_id);
    tokens.extend(tokenizer.encode("system\n"));
    tokens.extend(tokenizer.encode(system));
    tokens.push(tokenizer.im_end_id);
    tokens.extend(tokenizer.encode("\n"));

    // <|im_start|>user\n{user}<|im_end|>\n
    tokens.push(tokenizer.im_start_id);
    tokens.extend(tokenizer.encode("user\n"));
    tokens.extend(tokenizer.encode(user_message));
    tokens.push(tokenizer.im_end_id);
    tokens.extend(tokenizer.encode("\n"));

    // <|im_start|>assistant\n  ← model continues from here
    tokens.push(tokenizer.im_start_id);
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
