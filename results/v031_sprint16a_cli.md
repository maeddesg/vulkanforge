# v0.3.1 Sprint 16A — clap CLI port

**Result:** clap-driven `chat` / `bench` / `info` subcommands shipped.
Build clean, 27/27 lib tests, 15/15 coherence, decode median
**109.2 tok/s** (matches v0.3.0 baseline within run-to-run noise).
No changes to the decode/prefill hot paths.

## What changed

`src/main.rs` was refactored from an env-var-driven REPL into a
clap-dispatched binary with three subcommands. Every existing
behaviour of the old REPL is preserved verbatim — flags route to the
same `ChatSession`, the same banner / slash-commands / streaming
output, the same coopmat / async-decode plumbing.

```
$ vulkanforge --help
VulkanForge — Vulkan-backed LLM inference engine targeting AMD RDNA4 (gfx1201).
Decode 109 tok/s (0.95x llama.cpp Vulkan), prefill 0.89x at pp=512.

Usage: vulkanforge <COMMAND>

Commands:
  chat   Interactive multi-turn chat REPL (single-turn via VF_PROMPT="...")
  bench  Run a small bench: 5-prompt decode + 4-point pp sweep …
  info   Show GGUF model metadata + GPU/Vulkan information
```

### Subcommands

- **`chat`** — model path + sampling args (`--temperature`, `--top-k`,
  `--top-p`, `--repetition-penalty`, `--seed`, `--max-tokens`,
  `--system`, `--think-filter`). Single-turn fallback via `VF_PROMPT="…"`.
- **`bench`** — small in-binary smoke bench: 32-token decode +
  configurable `--pp-list` sweep. The full 15-prompt and full pp-sweep
  remain in `examples/run_15prompt_bench.rs` and
  `examples/run_pp_bench.rs` (called out in `--help` text).
- **`info`** — opens the GGUF without uploading weights, prints model
  metadata (architecture, layers, hidden dim, heads, RoPE base, vocab,
  file size) and GPU/Vulkan info (device name, API version,
  device-local heap size). Warns on unsupported architecture and on
  VRAM that cannot fit the file.

### Implementation choices

- `Cli` / `Commands` derive `clap::Parser` / `clap::Subcommand` with
  `--help` text quoting v0.3.0 numbers and pointing to the example
  binaries for full benchmarks.
- `default_model_path()` resolves `$VF_MODEL_PATH` →
  `~/models/Qwen3-8B-Q4_K_M.gguf`, used for every subcommand's
  `--model` default. Avoids forcing the path on every invocation.
- `run_chat(args: ChatArgs)` is the literal old `run()` body with
  env-var lookups replaced by `args.X` field reads — the chat REPL
  pre/post is bit-identical.
- `run_info()` uses `instance.get_physical_device_properties` and
  `…_memory_properties` directly from the `VulkanDevice` rather than
  re-doing physical-device selection.
- `run_bench()` builds the full forward stack (allocator, registry,
  kernel pool, model upload, KV cache, async fences) and calls
  `generate_from_tokens(…)` in a tight loop, taking the per-run median
  of `r.decode_time / r.generated_tokens` and `r.prefill_time`. Greedy
  sampling (temperature=0.0) for reproducibility. `forward.kv_cache.reset()`
  between runs.

### Bekannte Fallstricke encountered

1. **`tokenizer.bos_id`** — public *field* on `Tokenizer`, not a
   method (`Option<u32>`). First compile pass tried `bos_id().unwrap_or(1)`
   → fixed to `bos_id.unwrap_or(1)`. Two call sites in `run_bench`.
2. The `Cargo.toml` added `clap = { version = "4", features = ["derive"] }`
   — no MSRV bump needed (clap 4.x supports Rust 1.85).

No other API mismatches. The `generate_from_tokens(…)` signature,
`GenerateResult { decode_time, prefill_time, generated_tokens }`,
`Forward.kv_cache.reset()`, and `KvCache::reset()` were all correct
on the first attempt.

## Verification

| Check                      | Result |
|----------------------------|--------|
| `cargo build --release`    | clean (after `bos_id` fix)              |
| `cargo test --release --lib` | **27 / 27 passed**                    |
| `vulkanforge --help`       | three subcommands listed, banner OK     |
| `vulkanforge info …`       | qwen3 / 36 layers / 4.68 GiB / RX 9070 XT |
| `vulkanforge bench …`      | decode 116.6 tok/s, pp=128 2484 tok/s, pp=256 3481 tok/s (1 run, warmup-noisy) |
| `cargo run --release --example run_15prompt_bench` | **15 / 15 coherent**, median decode **109.2 tok/s** |

The 15-prompt median decode (109.2 tok/s) is within ±0.5 % of the
v0.3.0 release-noted 109.0 tok/s — the CLI shell adds zero hot-path
overhead, as expected.

## Files

- `Cargo.toml` — `+1` (clap dep)
- `src/main.rs` — refactor, `+356 / -30` (net +326 lines: clap structs,
  `default_model_path`, three `run_*` dispatch functions; chat REPL
  helpers preserved verbatim)
- `results/v031_sprint16a_cli.md` — this file

## Why this is v0.3.1, not v0.3.0+

v0.3.0 shipped the async decode pipeline behind `cargo run` env-vars
and `cargo run --example`. v0.3.1 makes the binary distributable —
`vulkanforge info <gguf>` is the single command a user needs to know
whether their model + GPU combo will work, and `vulkanforge chat …`
is the same command shape as `llama-cli` / `ollama run`. Surface-area
parity with ROCmForge's CLI is now back.

## Next

- Tag `v0.3.1` once user confirms.
- Consider exposing the async-decode kill-switch as a `chat --serial`
  / `bench --serial` flag rather than a `VULKANFORGE_DISABLE_ASYNC_DECODE`
  env var — minor ergonomics, not blocking.
