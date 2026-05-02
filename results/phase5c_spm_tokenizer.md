# Phase 5C — SPM Tokenizer + Mistral Support

**Date:** 2026-04-27
**Version:** v0.1.1 (after v0.1.0)
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Model under test:** `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`

---

## 1 — GGUF metadata (Mistral probe)

```
architecture       = "llama"
name               = "models--mistralai--Mistral-7B-Instruct-v0.3"

llama.attention.head_count               = 32
llama.attention.head_count_kv            = 8       (GQA 32:8)
llama.attention.layer_norm_rms_epsilon   = 1e-5
llama.block_count                        = 32
llama.context_length                     = 32768
llama.embedding_length                   = 4096
llama.feed_forward_length                = 14336
llama.rope.dimension_count               = 128
llama.rope.freq_base                     = 1000000
llama.vocab_size                         = 32768

tokenizer.ggml.model                     = "llama"   ← SPM
tokenizer.ggml.pre                       = "default"
tokenizer.ggml.tokens                    = Array<string> len=32768
tokenizer.ggml.scores                    = Array<f32>    len=32768
tokenizer.ggml.token_type                = Array<i32>    len=32768
tokenizer.ggml.bos_token_id              = 1
tokenizer.ggml.eos_token_id              = 2
tokenizer.ggml.unknown_token_id          = 0
tokenizer.ggml.add_bos_token             = true
tokenizer.chat_template                  = ".... '[INST] ' + content + ' [/INST]' ...."

[INST]   = id 3 (CONTROL)
[/INST]  = id 4 (CONTROL)

blk.0.attn_q_norm.weight                 absent     ← no Q/K-norm
blk.0.attn_k_norm.weight                 absent
RoPE variant                             Norm       ← arch=llama → adjacent-pair
```

No merges array, no qwen2/llama-bpe regex — confirms SentencePiece.

---

## 2 — SPM tokenizer (`src/backend/vulkan/spm.rs`)

### 2.1 Algorithm: greedy bigram-merge

The first-pass attempt (Viterbi best-path over a per-char DAG, summing
log-probabilities) was wrong for this format. The Mistral GGUF stores
`scores` as **merge priorities**, not log-probabilities — `at` has score
-11 and `▁What` has score -1565, so the Viterbi sum picks `▁Wh + at`
(-695) over `▁What` (-1565). The model was trained against a tokenizer
that uses **greedy bigram merge** (mirrors llama.cpp's
`llm_tokenizer_spm`):

1. Split input into one symbol per UTF-8 char.
2. Build a max-heap of every adjacent pair whose concatenation is in
   vocab, keyed on the merged token's score.
3. Pop highest-scoring pair, merge it (extend left symbol, mark right
   dead, re-link the doubly-linked list).
4. Push new bigrams that the merge has exposed (left's predecessor +
   left, left + right's successor).
5. Repeat until queue empties.
6. Walk the surviving chain — each symbol is either a vocab id, or
   byte-fallback `<0xHH>` per UTF-8 byte.

Tiebreak rule: equal scores → smaller `left` index pops first (matches
`llm_bigram_spm::operator<` in llama.cpp).

### 2.2 Normalisation

`encode(text)`: prepend `▁` (U+2581, SPM space marker) and replace
every `' '` with `▁` before merging.

`encode_no_prefix(text)`: same but skip the leading `▁`. Used by chat
templates to glue text directly onto a special-token boundary (e.g.
after `[INST]`) without a synthetic extra space.

### 2.3 Byte fallback

The 256 `<0xHH>` byte tokens (TYPE=BYTE, score=0) are recorded in a
fixed `[Option<u32>; 256]` lookup at load time. They're skipped during
the regular merge loop (so the score-0 bytes don't sweep the priority
queue and beat real merges) and only used at output time for symbols
whose final string isn't in vocab — exactly llama.cpp's behaviour.

### 2.4 LOC

- `src/backend/vulkan/spm.rs` — 422 lines (incl. tests + doc).
- `src/backend/vulkan/tokenizer.rs` — refactored to dispatch struct,
  ~520 lines.
- `src/backend/vulkan/chat_template.rs` — +60 lines for `Mistral` arm.

---

## 3 — Roundtrip + byte-fallback tests

```
phase5c_spm_tokenizer_loads_mistral        ok
phase5c_spm_encode_decode_roundtrip        ok    5 strings
phase5c_spm_byte_fallback_handles_unicode  ok    "こんにちは" via <0xHH>
phase5c_mistral_chat_template_brackets     ok    BOS+[INST]…[/INST]
```

`encode_decode_roundtrip` exercises: `"Hello World"`, `"1+1=2"`,
`"Hallo Welt!"`, `"The quick brown fox jumps over the lazy dog."`,
`"Mistral-7B-Instruct-v0.3"`. All round-trip exactly.

For Japanese (`"こんにちは"`) the 5 codepoints aren't whole-token vocab
entries; greedy-merge leaves them as raw chars and output emits each
UTF-8 byte as `<0xHH>`. Decode reassembles the bytes and gets the
original text back.

---

## 4 — Mistral chat template

Layout:

```
<s>[INST] {user_content} [/INST]
```

(or `[INST] {system}\n\n{user} [/INST]` when a system prompt is set —
the HF Jinja folds system into the user turn).

`[INST]` and `[/INST]` are emitted as their dedicated special-token
ids (3 / 4) rather than re-tokenized as ASCII. The body is encoded
with `encode_no_prefix` so the leading space `' '` becomes a single
`▁` (which the SPM merge step then concatenates with the first user
word).

We deliberately omit the trailing `' [/INST]'` space the HF Jinja
inserts — without it, the body's last token sits flush against
`[/INST]`, which matches what the model expects to see at training
time and keeps the first generated token on-distribution.

### 4.1 Detection

`ChatTemplate::detect` recognises Mistral by `[INST]` in the embedded
Jinja string; it ranks BEFORE the BPE-flavour fallback so the order is
DeepSeek → Llama-3 → ChatML → Mistral → flavour-fallback. SPM-flavoured
GGUFs without a Jinja template default to `ChatTemplate::Mistral`.

### 4.2 Tokenisation example

`<s>[INST] What is 2 + 2? [/INST]` →

```
[ 1=<s>] [3=[INST]] [2592=▁What] [1117=▁is] [29473=▁] [29518=2]
[1416=▁+] [29473=▁] [29518=2] [29572=?] [4=[/INST]]
```

Note `▁2` is not in the v0.3 vocab — it splits as `▁` + `2`. That's
the correct SPM Mistral output, not a bug.

---

## 5 — 5-prompt benchmark (Mistral-7B-Instruct-v0.3 Q4_K_M)

```
# 1 Mutex Explanation                pp= 22 gen=134  prefill=333.6  decode= 99.6 ✓
# 2 Haiku                            pp= 19 gen= 24  prefill=308.0  decode=104.7 ✓
# 3 Two Plus Two                     pp= 19 gen= 12  prefill=306.2  decode=104.1 ✓
# 4 Translate to German              pp= 28 gen= 19  prefill=390.1  decode=102.7 ✓
# 5 Prime Check (Python)             pp= 23 gen=220  prefill=346.1  decode= 98.9 ✓
```

| Metric              | Value           |
|---------------------|-----------------|
| Decode median tok/s | **102.7**       |
| Prefill median tok/s| 333.6           |
| Decode aggregate    | 99.8 tok/s      |
| Prefill aggregate   | 338.5 tok/s     |
| Coherence           | 5/5             |

Hand-spot-checked outputs:

- "Explain what a mutex is in one paragraph." →
  `A mutex (short for "mutual exclusion object") is a programming
  construct used to control access to shared resources …`
- "Write a haiku about the ocean." →
  `Endless blue expanse, / Whispers of ancient secrets, / Life's
  cradle, ebb, and flow.`
- "What is 2 + 2?" → `The sum of 2 + 2 is 4.`
- "Translate to German: The quick brown fox jumps over the lazy dog." →
  `Der schnelle braune Fuchs springt über den faulen Hund.`
- "Write a Python function that checks if a number is prime." →
  Idiomatic `is_prime(n)` with edge-case handling for n<2 and the
  `range(2, sqrt(n)+1)` loop.

---

## 6 — 4-model overview after v0.1.1

| Model | Arch | Tokenizer | Template | Decode tok/s | Prefill tok/s |
|---|---|---|---|---:|---:|
| Qwen3-8B Q4_K_M | qwen3 | BPE / qwen2 | ChatML | 88.5 | 405 |
| Meta-Llama-3.1-8B Q4_K_M | llama | BPE / llama-bpe | Llama3 | 94.6 | 490 |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M | llama | BPE / llama-bpe | DeepSeekR1 | 94.8 | 434 |
| Mistral-7B-Instruct-v0.3 Q4_K_M | llama | **SPM** | **Mistral** | **102.7** | 334 |

The first three numbers come from the Phase 5A 15-prompt suite; the
Mistral row is the new 5-prompt smoke (above).

---

## 7 — Test count

```
unit (lib)         19   (+5 vs v0.1.0: spm parse_byte_token, utf8_char_len,
                          spm_normalize, bigram heap, bigram tiebreak)
correctness         25
regression          21   (+4: phase5c_*)
doctests             0
total              65
```

All 65 green on `cargo test --release`.

---

## 8 — Files touched

```
NEW   src/backend/vulkan/spm.rs                    SPM tokenizer
NEW   examples/spm_dump.rs                         vocab/encoding diagnostic
NEW   inference_test_prompts_mistral_5.json        5-prompt suite
NEW   results/phase5c_spm_tokenizer.md             this report

EDIT  Cargo.toml                                   0.1.0 → 0.1.1
EDIT  src/backend/vulkan/mod.rs                    pub mod spm;
EDIT  src/backend/vulkan/tokenizer.rs              dispatch struct
EDIT  src/backend/vulkan/chat_template.rs          ChatTemplate::Mistral
EDIT  tests/regression.rs                          +4 phase5c_* tests
EDIT  CHANGELOG.md                                 v0.1.1 entry
EDIT  README.md                                    Mistral row + 5-prompt table
```

---

## 9 — Commit

Single squashed commit on `main`:

```
v0.1.1: SPM tokenizer + Mistral-7B support
```

No push — held for the user.

---

## 10 — Out of scope

- **Llama-2 chat support** — same SPM tokenizer family, but a
  different chat template (`<<SYS>>` / `[INST]` interleave with
  carriage returns). Not added in this phase.
- **Prefill optimisation** — Phase 5B target. Mistral prefill at 334
  tok/s vs llama.cpp Vulkan ~4 300 tok/s remains the largest delta.
- **Sampling** — still greedy argmax. Temperature / top-k / top-p is
  on the v0.2 backlog.
- **Quantised KV cache** — would unlock Mistral's full 32k context in
  16 GiB of VRAM. Not in this release.
