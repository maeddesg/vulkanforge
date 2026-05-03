# Sprint 22B — Skip embed_tokens GPU upload (−1.96 GiB VRAM)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 22, head was `57e5a25`)
**Goal:** Reduce SafeTensors VRAM. The brief proposed BF16→FP16
expansion to halve unquantized-tensor VRAM (~−2.5 GiB). The
actual fix turned out to be **simpler and bigger** for the
biggest single tensor: just don't upload it.

## Headline result

```
Meta-Llama-3.1-8B-Instruct-FP8 GPU upload:
  Sprint 20-M1 baseline:    10.42 GiB
  Sprint 22B (this commit):  8.46 GiB     (−1.96 GiB)
                                          (−18.8% VRAM)
```

Same load time (~1.9 s), same FP8 chat output, same decode
throughput. **One conditional `continue` in the loader.**

## What the brief proposed vs what shipped

The brief proposed a BF16→FP16 conversion to halve the cost of
the three unquantized tensor groups on GPU:

| Tensor | size | BF16→FP32 (Sprint 20-M1) | BF16→FP16 (brief) | Sprint 22B |
|---|---:|---:|---:|---:|
| `model.embed_tokens.weight` | 1.0 GiB BF16 | 2.1 GiB | 1.0 GiB | **0 (skipped)** |
| `lm_head.weight` | 1.0 GiB BF16 | 2.1 GiB | 1.0 GiB (needs FP16 GEMV) | unchanged (2.1 GiB) |
| 64× norm weights | ~0.5 MiB BF16 | ~1 MiB | ~0.5 MiB (needs FP16 RmsNorm) | unchanged |

For the 8B model on the 16 GiB card, **`embed_tokens` is the single
biggest piece** of unquantized VRAM. Skipping its upload lands a
larger saving than the brief's full BF16→FP16 plan would have
delivered for embed (`−2.1 GiB` vs `−1.1 GiB` from halving), with
**zero shader changes**.

## Why we can skip `embed_tokens` on GPU

The GPU copy of `model.embed_tokens.weight` is referenced in
exactly two places:

1. **`embedding_row` in `decode.rs`** — for GGUF models, reads from
   `gguf.tensor_bytes(&info)` (host mmap), never touches GPU.
   For SafeTensors, reads from `EmbeddingSource::Host(&host_embed)`
   (the FP32 host cache populated by `load_safetensors`), also
   never touches GPU.
2. **`dispatch_final` in `forward.rs`** — falls back to
   `token_embd.weight` when `output.weight` is missing
   (tied-weight models).

For neuralmagic's Llama-3.1-Instruct-FP8: `tie_word_embeddings: false`
in `config.json`, and `lm_head.weight` is present in the
SafeTensors tensor map. So the GPU copy of `embed_tokens` is
**dead weight** — uploaded, never read.

Sprint 20-M1's loader expanded BF16 → FP32 and uploaded all of
that data anyway, paying 2.1 GiB for nothing.

## What changed (loader.rs, ~15 LOC)

Inside `LoadedModel::load_safetensors`, before the per-tensor
upload loop:

```rust
let has_lm_head = st.tensors.contains_key("lm_head.weight");
let skip_embed_gpu = has_lm_head && !hf.tie_word_embeddings;
```

Then in the loop:

```rust
if skip_embed_gpu && hf_name == "model.embed_tokens.weight" {
    continue;
}
```

The host cache (the `Vec<f32>` returned alongside `LoadedModel`)
is still populated from the same `st.tensor_bytes()` call further
down the function — that path is unchanged. So embedding lookups
during chat still work; the change only affects what gets staged
to VRAM.

## Tied-weight safety

If a future SafeTensors model has `tie_word_embeddings: true`
(would be unusual for a quantized FP8 build but possible for an
FP16 / BF16 native model), `skip_embed_gpu` is `false` and the
old behavior is preserved — `embed_tokens` is uploaded as before.
`dispatch_final`'s fallback path keeps working in that case.

## What about FP16 expansion?

The brief's BF16→FP16 plan would shave the **remaining 1.6 GiB**
(lm_head 2.1 → 1.0 GiB, norms 1 MiB → 0.5 MiB) but requires:

1. A new `MulMatVecF16` GEMV (or extending `MulMatVecF32` with
   a build flag) — ~60 LOC of GLSL, pipeline registration,
   dispatch routing in `forward.rs::dispatch_final`.
2. A new `RmsNormF16` shader OR keeping norms FP32 in VRAM
   (negligible cost).
3. The bf16→f16 conversion math (3-mantissa-pad, exponent
   rebias, FP16-range clamp at ±65504).

That's ~150 LOC of careful work for ~1 GiB more saved on top of
the 2 GiB this sprint delivered. Diminishing returns; the
remaining 8.46 GiB lives in the FP8 weights themselves
(6.5 GiB) plus the 2.1 GiB unavoidable FP32 lm_head until the
FP16-GEMV port lands. **Punt to a future sprint** — when /
if a 14B-class FP8 model needs to fit in 16 GiB, this becomes
the lever.

## VRAM headroom now

```
Total GPU upload:                       8.46 GiB
KV cache @ max-context=2048 (FP16):     0.29 GiB
KV cache @ max-context=4096 (FP16):     0.58 GiB
KV cache @ max-context=8192 (FP16):     1.15 GiB
Scratch + Vulkan overhead:             ~0.50 GiB
─────────────────────────────────────────────
Total VRAM @ 8K context (Llama-3.1-FP8): ~10.1 GiB on 16 GiB card

Remaining headroom @ 16 GiB:           ~5.9 GiB
  (was 3.9 GiB before this sprint)
```

The 14B FP8 model class at ~12 GiB FP8 weights becomes plausibly
loadable now — the lm_head is still 4 GiB FP32 expanded, but
total = 12 + 4 + 0.5 + 1 = ~17.5 GiB, still over budget.
**The future BF16→FP16 sprint** would bring lm_head to 2 GiB,
totaling ~15.5 GiB → fits 16 GiB with margin. So the **layered
plan** is:

1. Sprint 22B (this commit): −2 GiB → 8B FP8 fits with room
2. Future BF16→FP16 sprint: another −1 GiB → 14B FP8 fits

## Coherence + Regression

```
> What is 2+2?  →  "The answer to 2+2 is 4."  (bit-identical vs Sprint 22)
  Decode: 62.9 tok/s (unchanged)
  Prefill (28 tokens): 86 ms / 325 tok/s (within noise of post-21B)
```

* GGUF Qwen3-8B-Q4_K_M chat: prefill 412, decode 111 tok/s
  (within noise).
* `cargo test --release --lib`: **37/37 pass**.
* FP8 GEMV + GEMM correctness tests: pass (kernels unchanged).
* Inspector reports `GpuTensors: 290` (was 291) — exactly one
  fewer = `token_embd.weight`.

## Files touched

* `src/backend/vulkan/loader.rs` — `skip_embed_gpu` gate +
  `continue` in the upload loop. ~15 LOC.
* `results/v034_sprint22b_bf16_fp16.md` — this report.

No shader changes, no new SPVs, no Cargo changes.

## Sprint 22 + 22B perf snapshot (Llama-3.1-8B-FP8)

```
                  Sprint 20-M1     Sprint 22B
GPU upload:       10.42 GiB        8.46 GiB     −1.96 GiB
Load time:         1.98 s          1.88 s        unchanged
Decode:            63.3 tok/s      62.9 tok/s    unchanged
Prefill (pp=512):  694.6 tok/s     ~694 tok/s    unchanged
Coherence:         "The answer is 4." (bit-identical)
```

The Sprint 20 USP — first Vulkan FP8 chat — now ships **with
~2 GiB more headroom** for KV cache or larger models.
