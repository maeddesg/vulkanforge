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
- **2× better power efficiency** (tok/s/W) on decode vs llama.cpp.
- **Llama-3, Qwen2.5, Qwen3, Mistral, DeepSeek-R1-Distill** model
  families covered (see [docs/MODELS.md](docs/MODELS.md)).

## Quick start

```bash
# GGUF (works on Mesa 26.0.6+, the default-everywhere path)
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf

# Native FP8 SafeTensors (Mesa 26.0.6+, BF16 conversion path)
VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Qwen3-8B-FP8/ \
  --tokenizer-from ~/models/Qwen3-8B-Q4_K_M.gguf

# Native FP8 WMMA (Mesa 26.1+, +45–58 % FP8 prefill)
VF_FP8_NATIVE_WMMA=1 VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Qwen3-8B-FP8/ \
  --tokenizer-from ~/models/Qwen3-8B-Q4_K_M.gguf
```

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
| Llama-3.1-8B FP8 | per-tensor           |        1197 |             14757 |
| Qwen2.5-14B FP8  | per-channel          |         450 |             (n/a) |
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
