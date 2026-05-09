# Installing VulkanForge

## Requirements

- **GPU:** AMD RX 9070 XT (RDNA4 / `gfx1201`) — other GPUs untested.
  Other RDNA generations may work but the spec-constants are tuned
  for `gfx1201` Wave64.
- **Driver:** RADV (Mesa **26.1+** recommended; system default on
  current Arch / CachyOS) advertising `VK_KHR_cooperative_matrix`.
  Mesa 26.0.6 still works via the BF16 conversion path but does not
  support native FP8 WMMA.
- **OS:** Linux. Tested on Arch Linux / CachyOS; other distros should
  work as long as they ship the same Mesa.
- **Rust:** stable toolchain, MSRV **1.85** (edition 2024).
- **`shaderc`:** the build script invokes `shaderc-sys` to compile
  GLSL → SPIR-V. On Arch / CachyOS: `pacman -S shaderc`.

## Build

```bash
git clone https://github.com/maeddesg/vulkanforge.git
cd vulkanforge
cargo build --release
```

First build compiles 81 SPIR-V binaries (a few seconds on top of
the Rust compile). Subsequent `cargo build` re-runs are ~2 s as
SPV is cached.

The binary is at `target/release/vulkanforge`.

## Install (optional)

```bash
# Symlink into PATH:
ln -sf "$(pwd)/target/release/vulkanforge" ~/.local/bin/vulkanforge

# Or copy:
cp target/release/vulkanforge ~/.local/bin/
```

## Quick start

Download a GGUF model, then:

```bash
# Inspect first (no VRAM upload):
vulkanforge info --model ~/models/Qwen3-8B-Q4_K_M.gguf

# Greedy chat:
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf

# 5-prompt decode + pp sweep:
vulkanforge bench --model ~/models/Qwen3-8B-Q4_K_M.gguf
```

`chat` accepts `--temperature / --top-k / --top-p /
--repetition-penalty / --seed / --max-tokens / --system /
--no-think-filter`. Each flag has a `VF_*` env-var fallback.

## Supported models

Auto-detected from GGUF metadata (`general.architecture` +
`tokenizer.ggml.pre` + chat template). `chat` and `bench` exit
with a clear message when a GGUF doesn't match a wired arch/quant.

| Architecture       | Tested models                                          | Status |
|--------------------|--------------------------------------------------------|--------|
| `qwen3`            | Qwen3-8B (Q3_K_M, Q4_K_M)                              | reference |
| `qwen2`            | Qwen2.5 (Q4_K_M)                                       | works   |
| `llama`            | Meta-Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B | works |
| `llama` (Mistral)  | Mistral-7B-Instruct-v0.3 (reports as `llama`)          | works   |

Qwen2.5 Q4_0 GGUFs need bias-add support which isn't shipped yet —
the Q4_0 shader infrastructure is in `main` but `file_type=2` is
gated out of preflight.

## Supported quantizations

| File type | Name      | Bits | Status                          |
|-----------|-----------|------|---------------------------------|
| 12        | Q3_K_M    | 3.4  | supported (decode 133.7 tok/s on Qwen3-8B with FP8 KV) |
| 15        | Q4_K_M    | 4.5  | supported (default, 118.5 tok/s with FP8 KV)  |
| 16        | Q5_K_S    | 5.5  | supported                        |
| 17        | Q5_K_M    | 5.5  | supported                        |
| 2         | Q4_0      | 4.0  | shader ready, **gated out** (needs bias-add for Qwen2.5) |

## FP8 KV cache (optional, v0.3.3)

Halves KV-cache VRAM via native FP8 E4M3 (`VK_EXT_shader_float8`):

```bash
VULKANFORGE_KV_FP8=1 vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf
```

| Model         | FP16 KV (default) | FP8 KV     | Decode bonus |
|---------------|------------------:|-----------:|-------------:|
| Qwen3-8B      |          288 MB   |   144 MB   |    +1.4 %    |
| Llama-3.1-8B  |          256 MB   |   128 MB   |    +1.2 %    |
| Mistral-7B    |          256 MB   |   128 MB   |    +1.4 %    |

Quality: 15 / 15 prompts coherent on the regression suite —
indistinguishable from FP16 in spot checks. Setting
`VULKANFORGE_KV_FP8=1` implies `VULKANFORGE_ENABLE_FP8=1` so the
device feature is auto-wired. Requires RDNA4 + a Mesa version
advertising `VK_KHR_cooperative_matrix` (26.1+ recommended).

## Environment variables

A small subset; full list in `README.md` § Configuration.

| Variable                              | Default | Effect                                    |
|---------------------------------------|---------|-------------------------------------------|
| `VF_MODEL_PATH`                       | —       | Default `--model` path                    |
| `VF_TEMPERATURE`                      | `0`     | `0` = greedy; `>0` enables sampling       |
| `VF_NO_THINK_FILTER`                  | unset   | Disable `<think>…</think>` stripping      |
| `VULKANFORGE_DISABLE_ASYNC_DECODE=1`  | off     | Serial decode (parity / debugging)        |
| `VULKANFORGE_DISABLE_MM_COOPMAT=1`    | off     | Scalar prefill (parity / debugging)       |
| `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` | off     | LDS tree-reduction GEMV instead of subgroup |

## Verify the install

```bash
# 1) GGUF parses + GPU detected:
vulkanforge info --model ~/models/Qwen3-8B-Q4_K_M.gguf
# expect: "Status         ✓ inference supported"

# 2) End-to-end coherent generation:
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf --max-tokens 30
# expect: a sensible answer to your prompt

# 3) Performance check:
vulkanforge bench --model ~/models/Qwen3-8B-Q4_K_M.gguf --runs 3
# expect: ~109 tok/s decode, ~3 800 tok/s pp=512
```

## Troubleshooting

- `Status: ⚠ architecture <X> ...` — the GGUF's architecture or
  `file_type` isn't in the preflight whitelist. `info` still works;
  `chat` / `bench` won't run until support lands.
- `VK_ERROR_FEATURE_NOT_PRESENT` at startup — driver doesn't expose
  `VK_KHR_cooperative_matrix`. Update Mesa.
- Garbage output / `!!!!!!` — almost always indicates an unsupported
  quant in a model that nominally is supported. Run
  `vulkanforge info` to confirm `file_type` is in the supported
  table; otherwise this is likely a mixed-quant tensor (see Sprint
  17B-debug / 17C / 17D reports in `results/`).

## Reports

Each sprint's findings live in `results/`. Recent landmarks:
`v032_sprint17a_llama.md` (multi-arch), `v032_sprint17c_q5k.md`
(Q3_K_M unblock + Q5_K_M unlock), `v032_sprint17d_q4_0.md` (Q4_0
infra + Qwen2.5 bias dependency surfaced).
