# VulkanForge

LLM inference engine for AMD RDNA4 GPUs. Pure Rust + Vulkan compute
shaders. ~14 MB static binary; no runtime dependencies beyond the
system Vulkan loader.

**First engine doing native FP8 WMMA over Vulkan on consumer AMD
hardware** (`V_WMMA_F32_16X16X16_FP8_FP8` via Mesa 26.1+
`shaderFloat8CooperativeMatrix`).

> This project builds on the foundational work of
> [oldnordic](https://github.com/oldnordic/ROCmForge).
> Without his original ROCmForge implementation — the model loader,
> the CPU inference path, the GGUF parser, and the overall
> architecture — none of the WMMA matrix-core optimisations, the
> multi-model support, or the interactive chat CLI would have been
> possible. Thank you for making this project a reality.

## Highlights

- **Wins decode on every direct comparison on RDNA4** — beats
  llama.cpp (Vulkan + ROCm) on Q4_K_M 8B, beats vLLM 0.20.1 ROCm
  on FP8 single-user decode (1.3–2× ahead).
- **Native FP8 E4M3** loader that ingests HuggingFace SafeTensors
  directly, no FP16 round-trip on disk.
- **All three FP8 scaling strategies auto-detected**:
  per-tensor, per-channel, block-wise `[128, 128]`.
- **Native FP8 WMMA on Mesa 26.1+** (`VF_FP8_NATIVE_WMMA=1`)
  — +45–58 % FP8 prefill across all three sub-types.
- **CPU `lm_head` offload (v0.3.10)** — Q6_K weights on CPU RAM,
  hand-tuned AVX-512 GEMV (Zen 4). Frees ~970 MB VRAM and on
  **14B FP8 it's 32 % faster than the GPU baseline**
  (17.8 vs 13.5 tok/s).
- **2× better power efficiency** (tok/s/W) on decode vs llama.cpp.
- **Llama-3, Qwen2.5, Qwen3, Mistral, DeepSeek-R1-Distill, Gemma-4**
  model families covered (Gemma-4 SafeTensors path produces coherent
  English with full Markdown structure post-v0.3.13; see
  [docs/MODELS.md](docs/MODELS.md) and the Unreleased section in
  [CHANGELOG.md](CHANGELOG.md) for the 8-bug coherence fix-up).
- **90 / 90 coherent (100 %)** on the deterministic 15-prompt suite
  across all six production configurations — GGUF, FP8 native WMMA
  (per-tensor / per-channel / block-wise), and CPU `lm_head` offload
  (see [Quality](#quality-15-prompt-benchmark) below).

## Quick start

```bash
# GGUF — no flag needed, default-everywhere path (Mesa 26.0.6+)
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf

# FP8 SafeTensors — one flag, no --tokenizer-from needed (v0.3.13)
# Auto-detects: FP8 model (config.json), native WMMA (Mesa 26.1+),
# AVX-512 (host), model size → CPU lm_head offload (≥ 12 B).
# Auto-loads: tokenizer.json + chat_template from the model dir.
VF_FP8=auto vulkanforge chat --model ~/models/Qwen3-8B-FP8/

# 14 B FP8 with auto CPU lm_head — saves 970 MB VRAM, +9 % decode
VF_FP8=auto vulkanforge chat --model ~/models/Qwen2.5-14B-Instruct-FP8/
```

The legacy v0.3.10 flags (`VULKANFORGE_ENABLE_FP8=1`,
`VF_FP8_NATIVE_WMMA=1`, `VF_CPU_LM_HEAD=1`) and `--tokenizer-from
<gguf>` still work as explicit overrides — handy when you want CPU
`lm_head` on an 8 B model for VRAM headroom, or want to force a
specific tokenizer source for a regression check.

Verify Mesa 26.1+ before setting `VF_FP8_NATIVE_WMMA=1`:

```bash
vulkaninfo 2>/dev/null | grep shaderFloat8CooperativeMatrix
# → must show "true"
```

Build from source:

```bash
cargo build --release   # Rust 1.85+, Vulkan headers required
```

## Performance at a glance

All numbers on AMD Radeon RX 9070 XT (gfx1201, RDNA4), Mesa 26.1-rc3
RADV unless noted. Full tables with power data and methodology in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### GGUF decode (single-user, batch=1)

| Engine        | Model        | Format | Backend  | Decode tok/s | tok/s/W |
|---------------|--------------|--------|----------|-------------:|--------:|
| **VF v0.3.9** | 8B Llama     | Q4_K_M | Vulkan   |      **121** | **0.58** |
| llama.cpp     | 8B Llama     | Q4_K_M | Vulkan   |          114 |   0.37  |
| llama.cpp     | 8B Llama     | Q4_K_M | ROCm     |           94 |   0.30  |

### Native FP8 prefill pp=512 (Mesa 26.1+, `VF_FP8_NATIVE_WMMA=1`)

| Model            | Scale type           | VulkanForge | vLLM 0.20.1 ROCm* |
|------------------|----------------------|------------:|------------------:|
| Llama-3.1-8B FP8 | per-tensor           |        1130 |             14757 |
| Qwen2.5-14B FP8  | per-channel          |         428 |             (n/a) |
| Qwen3-8B FP8     | block-wise [128,128] |        1118 |              2776 |

\* vLLM 0.20.1 is **not optimized for gfx1201** — model load logs
`Using default W8A8 Block FP8 kernel config. Performance might be
sub-optimal!`. Per-tensor uses `ROCmFP8ScaledMMLinearKernel` (specialized);
block-wise uses `TritonFp8BlockScaledMMKernel` (untuned). Run with
`VLLM_ROCM_USE_AITER=0 --enforce-eager` (only working configuration on RDNA4).

### Native FP8 decode (single-user, batch=1, decode-only power)

| Model            | VulkanForge (tok/s @ Avg W) | vLLM 0.20.1 ROCm (tok/s @ Avg W) | VF tok/s/W gain |
|------------------|----------------------------:|---------------------------------:|----------------:|
| Llama-3.1-8B FP8 |        **70 t/s @ 166 W**   |              53 t/s @ 159 W      | **+27 %**       |
| Qwen3-8B FP8     |        **62 t/s @ 125 W**   |              22 t/s @ 167 W      | **+267 %**      |

**VF wins decode 1.3–2×; vLLM wins prefill 2.5–12×.** Pick the engine
that fits the workload — single-user chat is VulkanForge, batch
serving is vLLM.

### CPU `lm_head` offload (v0.3.10, AVX-512)

`VF_CPU_LM_HEAD=1` moves the vocabulary projection onto the CPU as
Q6_K, freeing ~970 MB of VRAM. Hand-tuned AVX-512 kernel (Zen 4 /
Ice Lake+ runtime-detected; scalar fallback otherwise).

| Model              | GPU `lm_head`      | CPU `lm_head` (AVX-512) | VRAM saved | Verdict |
|--------------------|-------------------:|------------------------:|-----------:|---------|
| Llama-3.1-8B-FP8   | 70 tok/s           |  47.6 tok/s             |  −970 MB   | use for VRAM, not speed |
| **Qwen2.5-14B-FP8**| 13.5 tok/s         | **17.8 tok/s (+32 %)**  | **−970 MB**| **CPU wins both axes** |

The 14B win is structural: the GPU `lm_head` GEMV is bandwidth-bound
on 644 GB/s VRAM, and offloading it lets DDR5 (32 threads × L3 →
DDR5-5600 76 GB/s) carry the work in parallel with the rest of the
GPU pipeline freed up. Combined with the 970 MB VRAM saving, the
flag is a default-on candidate for 14B FP8 on Zen 4.

## Optional features

| Feature              | Flag                       | Requires                     | Effect                              |
|----------------------|----------------------------|------------------------------|-------------------------------------|
| FP8 model loading    | `VULKANFORGE_ENABLE_FP8=1` | Mesa 26.0.6+                 | Load HuggingFace FP8 SafeTensors    |
| Native FP8 WMMA      | `VF_FP8_NATIVE_WMMA=1`     | Mesa 26.1+                   | +45–58 % FP8 prefill                |
| CPU `lm_head` offload| `VF_CPU_LM_HEAD=1`         | AVX-512F + BW + VL (Zen 4 / Ice Lake+) | −970 MB VRAM, 14B +32 % decode |

All features are opt-in. Without flags, VulkanForge runs GGUF models
on any Mesa 26.0.6+ with no special requirements.

## Quality (15-prompt benchmark)

The deterministic 15-prompt suite (greedy decoding, temperature = 0)
on all six production paths:

| Configuration                                    | Coherent   | Median decode |
|--------------------------------------------------|-----------:|--------------:|
| Qwen3-8B Q4_K_M GGUF                             |  **15/15** |     109 tok/s |
| Llama-3.1-8B Q4_K_M GGUF                         |  **15/15** |     112 tok/s |
| Qwen3-8B-FP8 native WMMA + activation quant      |  **15/15** |      62 tok/s |
| Qwen2.5-14B-FP8 native WMMA + CPU `lm_head`      |  **15/15** |      17 tok/s |
| Llama-3.1-8B-FP8 native WMMA                     |  **15/15** |      70 tok/s |
| Llama-3.1-8B-FP8 native WMMA + CPU `lm_head`     |  **15/15** |      46 tok/s |

**90 / 90 prompts (100 %) coherent across the full suite.**
v0.3.11 closes the v0.3.10 Llama-FP8 per-tensor edge case (2/15
code-gen prompts collapsing to `!`) by porting the Sprint 39
per-block activation-absmax + rescale pattern to the per-tensor
WMMA path. The fix costs ~5 % on prefill (1197 → 1130 tok/s on
8B-FP8 pp=512) — an unavoidable trade for keeping post-RMS-norm
activations inside the FP8 E4M3 ±448 envelope.

## Driver requirements

| Mesa version | Capabilities                                                                  |
|--------------|-------------------------------------------------------------------------------|
| **26.0.6+**  | Full GGUF + FP8 SafeTensors via the BF16 conversion WMMA path                 |
| **26.1+**    | Adds `shaderFloat8CooperativeMatrix` → `VF_FP8_NATIVE_WMMA=1` is safe to set  |

For 14B+ models, set `amdgpu.lockup_timeout=10000,10000` on the
kernel command line — the default 2 s compute timeout is too short
for long prefill submits. Setup details and troubleshooting in
[docs/INSTALLATION.md](docs/INSTALLATION.md).

## Documentation

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — full VulkanForge vs llama.cpp vs vLLM tables, power data, methodology
- [docs/INSTALLATION.md](docs/INSTALLATION.md) — Mesa setup, kernel parameter, environment variables, troubleshooting
- [docs/MODELS.md](docs/MODELS.md) — supported GGUF / FP8 formats, model architectures, FP8 scaling strategies
- [CHANGELOG.md](CHANGELOG.md) — release history with per-sprint performance deltas

## CLI

```
vulkanforge chat   --model <PATH> [--tokenizer-from <GGUF>] ...
vulkanforge bench  --model <PATH> [--tokenizer-from <GGUF>] [--runs N]
```

`vulkanforge chat --help` lists every flag (sampling, max-tokens,
think-filter, max-context). The chat REPL accepts `/help`, `/quit`,
and a single-shot mode via `VF_PROMPT="..."`.

## Limitations

- Single-stream only — no batch inference, no concurrent sessions on
  one `Forward` instance.
- Decode at 0.80–1.06× llama.cpp Vulkan (model-dependent); coopmat
  is prefill-only on this codebase.
- FP8 prefill structurally behind ROCm-specialized kernels (vLLM's
  `ROCmFP8ScaledMMLinearKernel` is in a different class).
- `vulkanforge bench` accepts only Q4_K_M GGUF; Q8_0 chat works but
  does not bench.
- Mistral / Llama-2 SPM tokenizer not yet wired for FP8 SafeTensors
  (only `gpt2` tokenizer family).

For the full architectural notes and the v0.2.x optimization audit
(nine falsified hypotheses against the residual gap to llama.cpp),
see [CHANGELOG.md](CHANGELOG.md).

## License

VulkanForge is licensed under the
[GNU General Public License v3.0](LICENSE).
