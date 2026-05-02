//! SentencePiece tokenizer for `tokenizer.ggml.model="llama"` GGUFs
//! (Mistral, Llama-2, Falcon-3 family).
//!
//! Phase 5C / v0.1.1 — `tokenizer.rs` handles the GPT-2 byte-level BPE
//! flavours (Qwen3, Llama-3); this module covers SentencePiece. Both
//! are wired behind the public [`super::tokenizer::Tokenizer`]
//! dispatcher so call sites don't need to know which one they're
//! holding.
//!
//! Algorithm: **greedy bigram merge** (mirrors llama.cpp's
//! `llm_tokenizer_spm`). The GGUF `tokenizer.ggml.scores` array stores
//! a per-token merge priority — higher score = merge first. We start
//! with one symbol per UTF-8 char, then repeatedly pop the highest-
//! scoring adjacent pair whose concatenation is in vocab and merge it.
//! When the queue empties, every remaining symbol is emitted as its
//! vocab id (or as `<0xHH>` byte-fallback / `<unk>` if unmappable).
//!
//! `▁` (U+2581) is the SPM space marker: we replace `' '` with `▁` and
//! prepend a leading `▁` (for `encode`; `encode_no_prefix` skips the
//! prepend so chat templates can glue text onto a special token without
//! a synthetic leading space).
//!
//! Decode is the inverse: concatenate vocab strings, expand `<0xHH>`
//! byte tokens back to raw bytes, swap `▁` for space, and drop the
//! SPM-conventional leading space.

use std::collections::{BinaryHeap, HashMap};

use super::gguf::{GgufFile, MetadataValue};
use super::tokenizer::TokenizerError;

/// SPM space marker — every token that starts a word begins with this
/// character in the vocab.
pub(super) const SPM_SPACE: char = '\u{2581}';

/// `tokenizer.ggml.token_type` enum values from llama.cpp's
/// `LLAMA_TOKEN_TYPE_*`. UNUSED tokens are skipped entirely; BYTE
/// tokens are skipped during the merge step (they exist only for
/// fallback at output time when a symbol's string isn't in vocab).
const TOKEN_TYPE_UNUSED: u32 = 5;
const TOKEN_TYPE_BYTE: u32 = 6;

pub(super) struct SpmTokenizer {
    pub(super) vocab: Vec<String>,
    pub(super) token_to_id: HashMap<String, u32>,
    /// `tokenizer.ggml.scores[i]` — merge priority. Higher = merged
    /// first by the greedy bigram algorithm.
    scores: Vec<f32>,
    /// `tokenizer.ggml.token_type[i]`.
    token_types: Vec<u32>,
    pub(super) bos_id: Option<u32>,
    pub(super) eos_id: u32,
    pub(super) unk_id: u32,
    /// `byte_fallback[b]` is the vocab id of `<0xHH>` for byte `b`, or
    /// `None` when the model has no byte-fallback tokens.
    byte_fallback: [Option<u32>; 256],
}

impl SpmTokenizer {
    pub(super) fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.tokens"))?
            .as_array()
            .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.tokens"))?;
        let scores_arr = gguf
            .metadata
            .get("tokenizer.ggml.scores")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.scores"))?
            .as_array()
            .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.scores"))?;
        let types_arr = gguf
            .metadata
            .get("tokenizer.ggml.token_type")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.token_type"))?
            .as_array()
            .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.token_type"))?;

        if tokens_arr.len() != scores_arr.len() || tokens_arr.len() != types_arr.len() {
            return Err(TokenizerError::Malformed(format!(
                "SPM vocab arrays length mismatch: tokens={} scores={} token_type={}",
                tokens_arr.len(),
                scores_arr.len(),
                types_arr.len()
            )));
        }

        let n = tokens_arr.len();
        let mut vocab: Vec<String> = Vec::with_capacity(n);
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(n);
        let mut scores: Vec<f32> = Vec::with_capacity(n);
        let mut token_types: Vec<u32> = Vec::with_capacity(n);
        let mut byte_fallback: [Option<u32>; 256] = [None; 256];

        for i in 0..n {
            let s = tokens_arr[i]
                .as_str()
                .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.tokens[i]"))?
                .to_string();
            let score = scores_arr[i]
                .as_f32()
                .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.scores[i]"))?;
            let kind = parse_token_type(&types_arr[i])?;
            if let Some(b) = parse_byte_token(&s) {
                byte_fallback[b as usize] = Some(i as u32);
            }
            token_to_id.insert(s.clone(), i as u32);
            vocab.push(s);
            scores.push(score);
            token_types.push(kind);
        }

        let bos_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.eos_token_id"))?;
        let unk_id = gguf
            .metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);

        Ok(Self {
            vocab,
            token_to_id,
            scores,
            token_types,
            bos_id,
            eos_id,
            unk_id,
            byte_fallback,
        })
    }

    pub(super) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn token_str(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    pub(super) fn special_id(&self, name: &str) -> Option<u32> {
        self.token_to_id.get(name).copied()
    }

    /// Encode plain text. Replaces `' '` with `▁` and prepends `▁`
    /// (SPM convention). Does NOT add BOS — chat templates emit it
    /// explicitly.
    pub(super) fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = spm_normalize(text);
        self.greedy_merge(&normalized)
    }

    /// Encode a chunk that should not get a leading `▁` prepended —
    /// used by chat templates that need to glue text onto the trailing
    /// edge of a special token. Spaces inside the chunk still get
    /// rewritten to `▁` for vocab matching.
    pub(super) fn encode_no_prefix(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized: String = text
            .chars()
            .map(|c| if c == ' ' { SPM_SPACE } else { c })
            .collect();
        self.greedy_merge(&normalized)
    }

    /// SentencePiece greedy bigram merge. Symbols start as one per
    /// UTF-8 char; the priority queue holds adjacent-pair candidates
    /// ordered by their merged-token score. Highest score wins; ties
    /// broken by smaller left index (mirrors llama.cpp's
    /// `llm_bigram_spm::operator<`).
    fn greedy_merge(&self, text: &str) -> Vec<u32> {
        // Build the initial doubly-linked list of one symbol per char.
        let mut symbols = self.build_initial_symbols(text);
        let n = symbols.len();
        if n == 0 {
            return Vec::new();
        }

        // Priority queue of pending bigrams. We push every adjacent
        // pair whose concatenation exists in the vocab.
        let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
        for i in 1..n {
            self.try_add_bigram(text, &symbols, i - 1, i, &mut queue);
        }

        // Greedy merge until no more bigrams.
        while let Some(bg) = queue.pop() {
            // Skip stale bigrams: either endpoint already merged into a
            // neighbour (its `n` byte length now differs from when the
            // bigram was queued), or the right symbol no longer
            // immediately follows the left.
            let left = &symbols[bg.left];
            let right = &symbols[bg.right];
            if left.n == 0 || right.n == 0 {
                continue;
            }
            if left.n + right.n != bg.size {
                continue;
            }
            if left.next != bg.right as i32 {
                continue;
            }

            // Merge: extend `left` to span `right`, mark `right` dead.
            let new_n = left.n + right.n;
            let new_next = right.next;
            symbols[bg.left].n = new_n;
            symbols[bg.left].next = new_next;
            // Detach the right symbol.
            symbols[bg.right].n = 0;
            // Re-link right's successor's prev pointer.
            if new_next != -1 {
                symbols[new_next as usize].prev = bg.left as i32;
            }
            // Try new bigrams that the merge has exposed.
            let prev = symbols[bg.left].prev;
            if prev != -1 {
                self.try_add_bigram(text, &symbols, prev as usize, bg.left, &mut queue);
            }
            if new_next != -1 {
                self.try_add_bigram(
                    text,
                    &symbols,
                    bg.left,
                    new_next as usize,
                    &mut queue,
                );
            }
        }

        // Walk the linked list and emit ids. Each surviving symbol is
        // either (a) directly in vocab → emit its id, or (b) only
        // present as raw chars → byte-fallback per UTF-8 byte.
        let mut out: Vec<u32> = Vec::with_capacity(n);
        let bytes = text.as_bytes();
        let mut i: i32 = 0;
        while i != -1 {
            let sym = &symbols[i as usize];
            if sym.n > 0 {
                let s = std::str::from_utf8(&bytes[sym.offset..sym.offset + sym.n]).unwrap_or("");
                if let Some(&tid) = self.token_to_id.get(s) {
                    out.push(tid);
                } else {
                    // Byte fallback: emit each UTF-8 byte as <0xHH>.
                    for &b in &bytes[sym.offset..sym.offset + sym.n] {
                        let tid = self.byte_fallback[b as usize].unwrap_or(self.unk_id);
                        out.push(tid);
                    }
                }
            }
            i = sym.next;
        }
        out
    }

    fn build_initial_symbols(&self, text: &str) -> Vec<Symbol> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0usize;
        while offset < n {
            let len = utf8_char_len(bytes[offset]).min(n - offset);
            let idx = symbols.len();
            symbols.push(Symbol {
                offset,
                n: len,
                prev: idx as i32 - 1,
                next: idx as i32 + 1,
            });
            offset += len;
        }
        // Tail link: last symbol's next = -1.
        if let Some(last) = symbols.last_mut() {
            last.next = -1;
        }
        symbols
    }

    fn try_add_bigram(
        &self,
        text: &str,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        queue: &mut BinaryHeap<Bigram>,
    ) {
        let l = &symbols[left];
        let r = &symbols[right];
        if l.n == 0 || r.n == 0 {
            return;
        }
        let bytes = text.as_bytes();
        let combined_offset = l.offset;
        let combined_len = l.n + r.n;
        let combined = match std::str::from_utf8(&bytes[combined_offset..combined_offset + combined_len]) {
            Ok(s) => s,
            Err(_) => return,
        };
        let tid = match self.token_to_id.get(combined) {
            Some(&t) => t,
            None => return,
        };
        let kind = self.token_types[tid as usize];
        if kind == TOKEN_TYPE_UNUSED || kind == TOKEN_TYPE_BYTE {
            return;
        }
        queue.push(Bigram {
            left,
            right,
            score: self.scores[tid as usize],
            size: combined_len,
        });
    }

    /// Decode a token-id slice. Concatenates vocab entries, expands
    /// `<0xHH>` byte tokens back to raw bytes, swaps `▁` for space,
    /// and drops the leading SPM space.
    pub(super) fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.append_decoded(id, &mut bytes);
        }
        let raw = String::from_utf8_lossy(&bytes).into_owned();
        let mut out = raw.replace(SPM_SPACE, " ");
        // SPM convention: leading ▁ became a leading space — strip it.
        if out.starts_with(' ') {
            out.remove(0);
        }
        out
    }

    /// Single-token decode used for streaming. Does NOT strip the
    /// leading space; the streaming caller wants every char as it arrives.
    pub(super) fn decode_token(&self, id: u32) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        self.append_decoded(id, &mut bytes);
        let raw = String::from_utf8_lossy(&bytes).into_owned();
        raw.replace(SPM_SPACE, " ")
    }

    /// Sprint 16B — byte-exact streaming variant. Returns the token's
    /// raw bytes with the `▁` (E2 96 81) → space (0x20) rewrite done
    /// at the byte level, so the streaming caller can buffer partial
    /// UTF-8 codepoints across token boundaries without going through
    /// `String::from_utf8_lossy`. The `▁` glyph only appears as a
    /// complete 3-byte sequence inside a single vocab entry, never
    /// straddling a token boundary, so this byte-level replace is safe.
    pub(super) fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        self.append_decoded(id, &mut bytes);
        let mut tmp = [0u8; 4];
        let spm_bytes: &[u8] = SPM_SPACE.encode_utf8(&mut tmp).as_bytes(); // E2 96 81
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if i + spm_bytes.len() <= bytes.len() && &bytes[i..i + spm_bytes.len()] == spm_bytes {
                out.push(b' ');
                i += spm_bytes.len();
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        out
    }

    fn append_decoded(&self, id: u32, out: &mut Vec<u8>) {
        let s = match self.vocab.get(id as usize) {
            Some(s) => s,
            None => return,
        };
        if let Some(b) = parse_byte_token(s) {
            out.push(b);
        } else {
            out.extend_from_slice(s.as_bytes());
        }
    }
}

/// Doubly-linked-list symbol used during the merge.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    /// Byte offset into the normalised input.
    offset: usize,
    /// Byte length of this symbol; `0` once the symbol has been merged
    /// into a neighbour and is no longer part of the active chain.
    n: usize,
    /// Previous-symbol index in the chain, or `-1` for the head.
    prev: i32,
    /// Next-symbol index in the chain, or `-1` for the tail.
    next: i32,
}

/// Pending merge candidate. `BinaryHeap` is a max-heap, so `Ord` here
/// returns `Greater` for the bigram we want to pop first: highest
/// `score`, ties broken by smaller `left` (the leftmost merge wins on
/// equal scores — mirrors llama.cpp).
#[derive(Debug, Clone, Copy)]
struct Bigram {
    left: usize,
    right: usize,
    score: f32,
    /// Combined byte length at the time the bigram was pushed; used to
    /// detect stale entries (one of the endpoints has since been
    /// extended by another merge).
    size: usize,
}

impl Eq for Bigram {}
impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score → greater (popped first).
        match self.score.partial_cmp(&other.score) {
            Some(std::cmp::Ordering::Equal) | None => other.left.cmp(&self.left),
            Some(ord) => ord,
        }
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn parse_token_type(v: &MetadataValue) -> Result<u32, TokenizerError> {
    v.as_i32()
        .map(|x| x.max(0) as u32)
        .or_else(|| v.as_u32())
        .ok_or(TokenizerError::UnexpectedType("tokenizer.ggml.token_type[i]"))
}

/// Number of UTF-8 bytes in the codepoint that starts with `b`.
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        // Continuation byte at the start — treat as one byte to make
        // forward progress; should not happen on valid UTF-8.
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

fn spm_normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    out.push(SPM_SPACE);
    for c in text.chars() {
        if c == ' ' {
            out.push(SPM_SPACE);
        } else {
            out.push(c);
        }
    }
    out
}

/// Parses `<0xHH>` (six ASCII bytes, exactly that form). Returns `None`
/// for any other string.
fn parse_byte_token(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() == 6 && b[0] == b'<' && b[1] == b'0' && b[2] == b'x' && b[5] == b'>' {
        let hi = hex_digit(b[3])?;
        let lo = hex_digit(b[4])?;
        Some((hi << 4) | lo)
    } else {
        None
    }
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_byte_token_works() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0xab>"), Some(0xAB));
        assert_eq!(parse_byte_token("<0x>"), None);
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xZZ>"), None);
    }

    #[test]
    fn utf8_char_len_basic() {
        assert_eq!(utf8_char_len(b'A'), 1);
        assert_eq!(utf8_char_len(0xC3), 2); // start of 'ä'
        assert_eq!(utf8_char_len(0xE2), 3); // start of '▁'
        assert_eq!(utf8_char_len(0xF0), 4); // start of '🦀'
    }

    #[test]
    fn spm_normalize_prepends_and_replaces() {
        assert_eq!(spm_normalize(""), "\u{2581}");
        assert_eq!(spm_normalize("a"), "\u{2581}a");
        assert_eq!(spm_normalize("a b"), "\u{2581}a\u{2581}b");
        assert_eq!(spm_normalize("hi  there"), "\u{2581}hi\u{2581}\u{2581}there");
    }

    #[test]
    fn bigram_max_heap_pops_highest_score_first() {
        let mut q: BinaryHeap<Bigram> = BinaryHeap::new();
        q.push(Bigram { left: 0, right: 1, score: -100.0, size: 2 });
        q.push(Bigram { left: 5, right: 6, score: -10.0, size: 2 });
        q.push(Bigram { left: 2, right: 3, score: -50.0, size: 2 });
        let first = q.pop().unwrap();
        assert_eq!(first.score, -10.0);
        let second = q.pop().unwrap();
        assert_eq!(second.score, -50.0);
        let third = q.pop().unwrap();
        assert_eq!(third.score, -100.0);
    }

    #[test]
    fn bigram_tiebreak_smaller_left_first() {
        let mut q: BinaryHeap<Bigram> = BinaryHeap::new();
        q.push(Bigram { left: 5, right: 6, score: -10.0, size: 2 });
        q.push(Bigram { left: 1, right: 2, score: -10.0, size: 2 });
        let first = q.pop().unwrap();
        assert_eq!(first.left, 1);
    }
}
