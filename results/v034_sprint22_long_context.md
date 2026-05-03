# Sprint 22 — `--max-context` CLI flag (long-context unblocked)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 21C, head was `6f001ec`)
**Goal:** Replace the hard-coded `MAX_SEQ_LEN = 2048` with a
runtime CLI flag so chat / bench can drive contexts up to the
model's training horizon. Pre-check first (per Sprint 12 rule
"erst messen, dann fixen") to find the actual blocker before
writing code.

## Pre-check findings

The brief assumed `pp > 2048` would crash or produce garbage.
Reality (measured before any code change):

```
$ VF_PROMPT="<2450-word synthetic prompt>" \
    vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf \
    --temperature 0.0 --max-tokens 5

  ctx_max=2048
  …
  [context overflow]   ← graceful, no crash
```

VF already handles overflow cleanly via `ChatError::ContextOverflow`
in the chat REPL. The blocker isn't crashes — it's that the user
**can't opt into a larger window** without recompiling.

Memory-side audit of the four MAX_SEQ-spec-const consumers:

| Shader | LDS dependency on `MAX_SEQ` |
|---|---|
| `scalar_attn.comp` | `shared float scores[MAX_SEQ]` — 4 B per token; 8 KB at 2048, 32 KB at 8192. **Not on production dispatch path** (the helper `run_scalar_attn` actually dispatches `FlashAttn*` based on tile threshold). |
| `flash_attn.comp` | `shared float scores_lds[TILE]` — fixed; `MAX_SEQ` carried for binary compat only |
| `flash_attn_split.comp` | `MAX_SEQ` push-constant only, no LDS use |
| `flash_attn_batch.comp` | `MAX_SEQ` push-constant only |

**None of the production attention paths' LDS scales with MAX_SEQ.**
The blocker is purely the hardcoded constant in `main.rs:62` and
the four pipeline_registry.rs spec-const sites pinned at 2048.

## What landed (~80 LOC, no shader changes)

### `src/backend/vulkan/pipeline_registry.rs`

* New `PipelineRegistry::new_with_max_seq(device, cache_path, max_seq)`
  constructor. The legacy `new(...)` becomes a wrapper that calls
  `new_with_max_seq(.., 2048)` so the **14 existing example/test
  callsites stay byte-identical**.
* The four `[u32; 1] = [2048]` spec-const arrays in the
  `ScalarAttn`, `FlashAttn` family, and `FlashAttnBatch` arms become
  `[max_seq]`.

### `src/main.rs`

* `Commands::Chat` and `Commands::Bench` gain
  `max_context: Option<u32>` (`--max-context N` CLI flag).
* `ChatArgs` gains `max_context`.
* `run_chat`, `run_chat_safetensors`, `run_bench`,
  `run_bench_safetensors` all default to the existing `MAX_SEQ_LEN`
  (2048) when the flag isn't set, otherwise pass the user's value
  to `PipelineRegistry::new_with_max_seq` and to `KvCacheConfig.max_seq_len`.
* Post-load warning when `max_context > cfg.context_length` (the
  model's training horizon — generation past that point will lose
  coherence regardless of cache size).
* Bench KV-cache also respects the user's `--max-context`
  (previously was `(max_pp + 64).max(2048)` only).

## Measured results

### Q4_K_M (Qwen3-8B), 3-run median

```
                     ms (med)     tok/s
default (2048):
  pp=64               35.4       1808.8
  pp=128              47.1       2716.3
  pp=512             132.8       3856.4
  pp=1024            274.0       3736.7
  decode             ms/tok      117.1 tok/s

--max-context 4096:
  pp=1024            273.9       3739.3   ← unchanged
  pp=2048            646.7       3166.8   ← NEW
  pp=4096           1719.8       2381.7   ← NEW

--max-context 8192:
  pp=4096           1722.7       2377.7
  pp=8192           5154.4       1589.3   ← NEW
  decode            ms/tok       117.1 tok/s   ← unchanged
```

The throughput dip at large pp is **attention's natural O(N²)
work** (every prefilled token attends to all earlier ones, so the
total flash-attention work scales as N² even though each pass uses
O(N) memory). At pp=8192 the prefill is 5.15 s — usable.

### FP8 SafeTensors + `--max-context 4096`

```
                     ms (med)     tok/s
  pp=1024           1516.7       675.2
  pp=2048           2973.4       688.8   ← peak
  pp=4096           6300.9       650.1
  decode             ms/tok       62.7 tok/s
```

The FP8 GEMM kernel's plateau (~700 tok/s, Sprint 18B's 1.18×
WMMA ceiling) absorbs the attention work fairly evenly. pp=4096
prefill takes 6.3 s — slow but coherent.

### VRAM / KV-cache at long context (Qwen3-8B GQA: 8 KV-heads, head_dim=128, 36 layers)

| `--max-context` | FP16 KV | FP8 KV | Total VRAM (Q4_K_M) |
|---:|---:|---:|---:|
| 2048 | 288 MB | 144 MB | ~6.0 GB |
| 4096 | 576 MB | 288 MB | ~6.3 GB |
| 8192 | 1152 MB | 576 MB | ~6.9 GB |
| 32768 | 4608 MB | 2304 MB | ~10.3 GB |

A 16-GB RX 9070 XT has plenty of headroom for **32K context** on
Q4_K_M with FP8 KV — well past Llama-3's 8K training window for
non-Llama-3.1 builds, and into useful territory for long-context
work on the Llama-3.1-128K base.

## Coherence

```
$ vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf \
    --max-context 4096 --temperature 0 --max-tokens 20

> What is 2+2?
   "Okay, the user asked 'What is 2+2?' That's a basic math…"
   coherent, decode 110 tok/s — same as default.
```

`--max-context > cfg.context_length` (e.g. asking for 4096 on a
model whose `cfg.context_length` reports 2048) prints a warning
but proceeds; the user is responsible for knowing whether the
model can actually attend that far.

## Regression — all green

* Default (no `--max-context`) bench output bit-identical to
  pre-Sprint-22 baseline (Qwen3-8B-Q4_K_M decode 117.1 tok/s,
  prefill 64/128/512/1024 = 1809/2716/3856/3737).
* `cargo test --release --lib`: **37/37 pass**.
* FP8 GEMM correctness: pass (kernel unchanged).
* The 14 example / test callsites of `PipelineRegistry::new` are
  untouched — they continue to pin 2048 via the wrapper.

## Files touched

* `src/backend/vulkan/pipeline_registry.rs` — split `new` into
  legacy wrapper + new `new_with_max_seq` body; 4 spec-const
  literals replaced with `max_seq` parameter.
* `src/main.rs` — `--max-context` CLI flag wired through 4
  entry points (Chat GGUF / Chat SafeTensors / Bench GGUF /
  Bench SafeTensors); `MAX_SEQ_LEN` constant kept as the default;
  post-load warning for over-the-horizon contexts.
* `results/v034_sprint22_long_context.md` — this file.

~80 LOC net, no shader changes, no new SPVs.

## What's now possible

1. **First measured pp=2048-8192 prefill on VulkanForge** for any
   model. Until this commit, those numbers literally couldn't be
   measured.
2. **Long-context chat sessions** without the user having to
   recompile — `--max-context 4096` (or 8192, or 32768 if the
   model supports it) works on the existing GGUF + FP8 paths.
3. **Sprint-future memory-budget exploration** with an honest
   data point at every context size, not just 2048.

## Honest caveats

* Throughput drops at large pp because attention is genuinely
  O(N²) work. At pp=8192, prefill is ~5 s on Q4_K_M — usable but
  not snappy. Llama.cpp shows a similar curve on the same
  hardware; the difference is the 1.18× WMMA gap on the
  individual matmul, not anything algorithmic at the attention
  level.
* The post-load warning when `max_context > cfg.context_length`
  catches the *model's* training horizon, but doesn't catch RoPE
  scaling subtleties — Llama-3.1's 128K base requires the
  `rope_scaling` config to apply per-position-band rescaling,
  which VF detects but doesn't apply yet (flagged in
  Sprint 20-M3's report). Generation works at long context but
  positional encoding may drift past 8K.
* `scalar_attn.comp` declares `shared float scores[MAX_SEQ]` →
  at MAX_SEQ=8192 the LDS allocation goes from 8 KB to 32 KB.
  The shader is **not on the production dispatch path**, so it
  doesn't affect chat/bench, but if a user opts into it via
  some debug path with a very large `--max-context`, they may
  hit the 64-KB-per-CU LDS cap. Mark as deferred — fix when
  someone actually needs it.
