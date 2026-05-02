# v0.3.1 Sprint 16B — GGUF auto-detection (info + preflight)

**Result:** `vulkanforge info` reads quant + tokenizer + context length
on every GGUF in `~/models/` (9 of 10 — gemma-4 is blocked by an
unrelated Q4_K tensor-type parser issue, not metadata). `vulkanforge
chat` and `vulkanforge bench` now bail out **before** Vulkan device
init when the architecture or quantization isn't wired into the
forward pass, with a one-line pointer to `vulkanforge info`.

## Pre-check before code

The brief assumed VF was Qwen3-hardcoded and proposed rewriting
`ModelConfig::from_gguf()`. The pre-check disproved that:

- `ModelConfig::from_gguf()` (`src/backend/vulkan/gguf.rs:368`) already
  reads `general.architecture` and uses it as a metadata-key prefix
  for every dim (`{arch}.block_count`, `{arch}.embedding_length`, …).
- Live test on 5 non-Qwen3 GGUFs: 4 of 5 already produced clean
  `info` output. The actual gaps were in **what `info` shows**, not
  in detection.

Per `feedback_sprint_hypothesis_check.md` memory: ship the gap, not
the brief verbatim.

## What changed

### `src/main.rs`

- Added `file_type_name(u32) -> &'static str` — the `LLAMA_FTYPE_*`
  enum (file-level quant tag), spanning `F32` … `Q6_K` … `BF16`.
- Added `inference_support(arch, file_type) -> (arch_ok, quant_ok)`.
  Truth-table:

  | arch         | quant   | result |
  |--------------|---------|--------|
  | qwen2 / qwen3 | Q4_K_M | ✓ supported |
  | qwen2 / qwen3 | other  | quant warning |
  | other         | any    | arch warning |

- Added `preflight_supported(model_path, subcommand)` — opens the GGUF
  header (no weight upload, no Vulkan init), runs the support check,
  and returns a clean `Err` with a user-facing message before chat /
  bench can spin up the device.
- `run_chat()` and `run_bench()` now call `preflight_supported(...)`
  on the first line — costs ~30 ms vs. seconds for full setup before
  failure.
- `run_info()` now displays:
  - `Quantization   Q4_K_M (file_type=15)`
  - `Tokenizer      gpt2` (or `llama`)
  - `Context        40960`
  - `Status         ✓ inference supported` /  ⚠ architecture / quant
- Removed the misleading `["qwen3", "qwen2", "llama"]` whitelist —
  Llama models ran past the architecture check and crashed during
  forward pass setup.

## Sweep across `~/models/`

| File                                       | Arch | Quant | Status |
|--------------------------------------------|------|-------|--------|
| Qwen3-8B-Q4_K_M                            | qwen3 | Q4_K_M | ✓ supported (production) |
| qwen2.5-0.5b-instruct-q4_k_m               | qwen2 | Q4_K_M | ✓ supported |
| qwen2.5-0.5b-instruct-q4_0                 | qwen2 | Q4_0   | ⚠ quant |
| Qwen2.5-7B-Instruct-Q4_0                   | qwen2 | Q4_0   | ⚠ quant |
| Qwen2.5-7B-Instruct-Q4_0-Pure              | qwen2 | Q4_0   | ⚠ quant |
| Qwen2.5-14B-Instruct-Q4_0                  | qwen2 | Q4_0   | ⚠ quant |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M          | llama | Q4_K_M | ⚠ arch |
| Mistral-7B-Instruct-v0.3.Q4_K_M            | llama | Q4_K_M | ⚠ arch (the GGUF self-reports `llama`) |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M        | llama | Q4_K_M | ⚠ arch (Llama-distill) |
| gemma-4-E4B-it-Q4_K_M                      | gemma3 | —    | tensor parse fail (`unknown ggml tensor type 30`) |

The Mistral and DeepSeek-R1-Distill GGUFs both self-report
`architecture=llama` — confirms the brief's "Bekannte Fallstrick #4":
DeepSeek-R1-Distill is a Llama-architecture model, and Mistral's GGUF
authoring convention reuses the Llama metadata layout.

The `gemma-4` failure is an unrelated tensor-type parser issue in
`gguf.rs::GgmlType::from_u32` — Gemma's Q4_K tensors use ggml type
ID 30, which isn't in the enum. Out of scope for auto-detection;
needs a separate dequant kernel addition.

## Verification

| Check                              | Result |
|------------------------------------|--------|
| `cargo build --release`            | clean |
| `cargo test --release --lib`       | **27 / 27** |
| `info` sweep across 10 GGUFs       | 9 / 10 (gemma blocked by unrelated parser) |
| `chat` preflight on Llama          | ⚠ message + exit, no device init |
| `chat` preflight on Qwen2.5 Q4_0   | ⚠ message + exit, no device init |
| `chat` on Qwen3 Q4_K_M             | works as before |
| `run_15prompt_bench`               | **15 / 15 coherent**, decode 109.0 tok/s |

## What's NOT in this sprint

- No new `ModelConfig` rewrite (already generic; would have churned
  every consumer for zero benefit).
- No Llama / Mistral forward-pass implementation — that's the v0.4
  multi-arch sprint; preflight gives a clean error message in the
  meantime.
- No Q4_0 dequant kernel — same reasoning. `info` shows the warning
  so users know exactly why.
- No Gemma3 tensor-type-30 fix — separate concern from auto-detection.

## Files

- `src/main.rs` — `+78 / -10` (helpers + display + preflight)
- `results/v031_sprint16b_auto_detection.md` — this file

Total LOC delta: ~80, single file. The brief's proposed
`ModelConfig::from_gguf()` rewrite would have been ~300 LOC across
3 files for behaviour the codebase already had.
