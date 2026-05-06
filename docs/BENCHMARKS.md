# VulkanForge Benchmarks

All numbers measured on **AMD Radeon RX 9070 XT (gfx1201, RDNA4, 16 GB)**
unless noted otherwise. Power sampled at 10 Hz from
`/sys/class/drm/card1/device/hwmon/hwmon*/power1_average`, first 20
samples discarded as warm-up.

VulkanForge build: v0.3.9 + Sprint 39 (commit `95185aa`). Mesa 26.1-rc3
RADV unless a specific run is tagged 26.0.6. ROCm 7.2.2 / Linux 7.0.3.

---

## Quick verdict

| Workload                              | Best engine                          |
|---------------------------------------|--------------------------------------|
| Single-user chat decode (b=1)         | **VulkanForge** (1.06–2.0× ahead)    |
| Power efficiency (tok/s/W) — decode   | **VulkanForge** everywhere on 8B    |
| GGUF Q4_K_M decode                    | **VulkanForge** (121 vs 114 vs 94)   |
| GGUF prefill (Q4_K_M / Q8_0)          | llama.cpp (Vulkan ≥ ROCm on RDNA4)   |
| FP8 prefill (per-tensor / block-wise) | vLLM (untuned but still 2.5–12× ahead) |
| Batch serving (b≥4)                   | vLLM (VF is single-stream)           |

---

## GGUF — VulkanForge vs llama.cpp

llama.cpp build `23b8cc4`, built with `-DGGML_HIP=ON -DGGML_VULKAN=ON
-DAMDGPU_TARGETS=gfx1201`.

### Decode (tok/s, single-user, batch=1)

| Engine        | Model        | Format | Backend     | Decode | Avg W | tok/s/W |
|---------------|--------------|--------|-------------|-------:|------:|--------:|
| **VF v0.3.9** | 8B Llama     | Q4_K_M | Vulkan      |    **121** | 209 | **0.58** |
| llama.cpp     | 8B Llama     | Q4_K_M | Vulkan      |        114 | 312 |   0.37  |
| llama.cpp     | 8B Llama     | Q4_K_M | ROCm/HIP    |         94 | 310 |   0.30  |
| llama.cpp     | 8B Llama     | Q8_0   | Vulkan      |         73 | 251 |   0.29  |
| llama.cpp     | 8B Llama     | Q8_0   | ROCm/HIP    |         64 | 246 |   0.26  |
| llama.cpp     | 14B Qwen2.5  | Q8_0   | Vulkan      |         40 | 215 |   0.19  |
| llama.cpp     | 14B Qwen2.5  | Q8_0   | ROCm/HIP    |         38 | 226 |   0.17  |

**VF wins decode 1.06× over the closest llama.cpp build, 1.93× over
llama.cpp ROCm.** Power-efficiency lead is structural (≈ −33 % power
at higher throughput), not a backend artifact.

### Prefill pp=512 (tok/s)

| Engine        | Model    | Format | Backend  | pp=64 | pp=128 | pp=512 | pp=1024 |
|---------------|----------|--------|----------|------:|-------:|-------:|--------:|
| llama.cpp     | 8B Llama | Q8_0   | Vulkan   |  1620 |   2510 | **4972** |    — |
| llama.cpp     | 8B Llama | Q8_0   | ROCm     |  1545 |   2380 |   4790 |    — |
| llama.cpp     | 8B Llama | Q4_K_M | Vulkan   |  1600 |   2360 |   4458 |    — |
| VF v0.3.9     | 8B Llama | Q4_K_M | Vulkan   |  1802 |   2806 |   3992 |   3923 |
| llama.cpp     | 8B Llama | Q4_K_M | ROCm     |  1490 |   2240 |   3936 |    — |
| llama.cpp     | 14B Qwen | Q8_0   | ROCm     |  1333 |   1820 |   2535 |    — |
| llama.cpp     | 14B Qwen | Q8_0   | Vulkan   |  1272 |   2134 |   1660 |    — |

llama.cpp wins prefill on GGUF formats (rocBLAS / coopmat-tuned tile
selection). VF Q4_K_M sits at 0.89–0.90× vs llama.cpp Vulkan; the gap
is structural at the graph level (verified via the nine-hypothesis
falsification table in v0.2.x sprints).

### Honest-negatives

- **VF cannot bench Q8_0 GGUF** — `vulkanforge bench` rejects it as
  unsupported. Q8_0 is loadable for chat (`vulkanforge chat --model
  *.Q8_0.gguf`) but the benchmark harness is Q4_K_M-only.
- **14B Q8_0 prefill drops on llama.cpp Vulkan** at long context
  (1660 t/s @ pp=512 vs 2535 ROCm) — KV-pressure / TDR-window
  interaction at the 16 GB VRAM ceiling. ROCm is monotonic.

---

## Native FP8 — VulkanForge vs vLLM 0.20.1 ROCm

> **vLLM caveats — read first.**
> 1. **vLLM 0.20.1 is not optimized for gfx1201.** On model load it warns
>    `Using default W8A8 Block FP8 kernel config. Performance might be
>    sub-optimal! Config file not found at
>    …/N=4096,K=4096,device_name=AMD_Radeon_RX9070XT,…`. There are no
>    device-specific kernel configs shipped for the RX 9070 XT.
> 2. **`VLLM_ROCM_USE_AITER=0`** required — AITER's gfx942-tuned ASM
>    path crashes on gfx1201.
> 3. **`--enforce-eager` required** — ROCm 7.2.2 CUDAGraph capture is
>    unstable on gfx1201; eager mode pays per-step Python dispatch.
> 4. **Per-tensor uses `ROCmFP8ScaledMMLinearKernel` (specialized);
>    block-wise uses `TritonFp8BlockScaledMMKernel` (untuned).** The two
>    paths are not equally optimized — the per-tensor numbers reflect
>    a dedicated ROCm kernel; block-wise is more comparable.

VulkanForge is also an out-of-the-box configuration: no shape-aware
auto-tuning, single-stream only.

### Decode (tok/s, batch=1, single-user) — measured 2026-05-09

Measurement methodology (Sprint 39C):
- VF: `vulkanforge chat --max-tokens 256 --temperature 0.0` with a
  long-response prompt; power sampled across the decode window.
- vLLM: `vllm bench latency --batch-size 1 --input-len 1
  --output-len 128 --enforce-eager --num-iters 5`; decode tok/s
  derived as 128 / avg_latency.
- Both measurements are **decode-dominated** (≥ 117 generated tokens
  for VF runs, 128 for vLLM) so the W value is decode steady-state,
  not bench-wide.

| Engine         | Model            | Scale type   | Decode | Avg W | tok/s/W | Notes                       |
|----------------|------------------|--------------|-------:|------:|--------:|-----------------------------|
| **VF v0.3.9**  | 8B Llama FP8     | per-tensor   |  **69.8** | 166.2 | **0.42** | Native FP8 WMMA, Mesa 26.1+ |
| vLLM 0.20.1    | 8B Llama FP8     | per-tensor   |     53.2 | 159.2 |   0.33  | `ROCmFP8ScaledMMLinearKernel` |
| **VF v0.3.9**  | 8B Qwen3 FP8     | block-wise   |  **62.0** | 125.3 | **0.49** | Native FP8 WMMA + act-quant |
| vLLM 0.20.1    | 8B Qwen3 FP8     | block-wise   |     22.4 | 166.5 |   0.13  | `TritonFp8BlockScaledMMKernel` (untuned for gfx1201) |
| **VF v0.3.9**  | 14B Qwen2.5 FP8  | per-channel  |  **13.5** | 151.2 | **0.089** | vLLM 14B not benched (16 GiB VRAM tight) |

**VF wins decode 1.31× (Llama) up to 2.77× (Qwen3) at equal-or-lower
power.** Note vLLM's Qwen3 number specifically: at `input-len=1`,
vLLM falls into a slower scheduler path (22.4 t/s vs 31.7 t/s when
the prefill is non-trivial). Even comparing against the more
favourable 31.7 t/s subtraction methodology used in the
Sprint 38-Bench report, VF's 62 t/s lead stays.

Decode is memory-bandwidth-bound; VF's Vulkan-native GEMV beats
vLLM's eager-mode Python dispatch overhead consistently across all
three FP8 sub-types.

#### Earlier (Sprint 34D) decode + Avg-W reading

The previous version of this table cited `135 W / 0.51 tok/s/W` for
VF Llama-8B-FP8. That figure came from a Sprint 34D bench on
Mesa 26.0.6 + VF v0.3.7 (BF16 conversion path) and used a
*bench-wide* Avg W (mixed prefill + decode). The Sprint 39C
measurement above is **decode-only** with a long-response prompt and
v0.3.9 native FP8 WMMA — the cleaner signal.

### Prefill pp=512 (tok/s)

| Engine         | Model            | Scale type   |  pp=512 | Avg W |
|----------------|------------------|--------------|--------:|------:|
| vLLM 0.20.1    | 8B Llama FP8     | per-tensor   | **14757** | 147 |
| vLLM 0.20.1    | 8B Qwen3 FP8     | block-wise   |    2776 | 185 |
| VF v0.3.9      | 8B Llama FP8     | per-tensor   |    1197 | 204 |
| VF v0.3.9      | 8B Qwen3 FP8     | block-wise   |    1118 | 273 |

**vLLM wins prefill 2.5× (block-wise) to 12× (per-tensor).** ROCm's
specialized `ROCmFP8ScaledMMLinearKernel` is a different class of GEMM
than what a single Vulkan compute shader can match. For batch-serving
or long-prompt prefill, vLLM is the right tool.

### Use-case split

- **Single-user latency / chat (decode-dominated):** VulkanForge.
- **Batch serving / long prompts (prefill-dominated):** vLLM.
- **GGUF-only (no FP8 model on disk):** llama.cpp Vulkan.

### Throughput (batch=10, in=512, out=128) — vLLM strength

| Engine         | Model        | Total t/s | Output t/s | Avg W |
|----------------|--------------|----------:|-----------:|------:|
| vLLM 0.20.1    | 8B Llama FP8 |      1908 |        382 |    94 |
| vLLM 0.20.1    | 8B Qwen3 FP8 |       539 |        108 |   129 |
| VF v0.3.9      | (any FP8)    |        n/a |        n/a |   n/a |

VulkanForge is single-stream; batched throughput is out of scope.

---

## Native FP8 — Mesa 26.1+ vs Mesa 26.0.x

`VF_FP8_NATIVE_WMMA=1` enables `V_WMMA_F32_16X16X16_FP8_FP8` on
Mesa 26.1+. Mesa 26.0.x silently runs the BF16 conversion path.

| Model                    | Scale type           | Mesa 26.0.x / no flag | Mesa 26.1+ native | Δ        |
|--------------------------|----------------------|----------------------:|------------------:|---------:|
| Llama-3.1-8B-FP8         | per-tensor           |                   757 |          **1197** | **+58 %** |
| Qwen2.5-14B-FP8          | per-channel          |                   325 |           **450** | **+39 %** |
| Qwen3-8B-FP8             | block-wise [128,128] |                   770 |          **1118** | **+45 %** |

The block-wise win required a per-k_block dynamic activation absmax
(Sprint 39) — the naive FP32→FP8 cast lost too much dynamic range
against block-wise-calibrated weights. See
`results/v039_sprint39_blockwise_act_quant.md` for the algorithm + ISA.

---

## Methodology

- **VulkanForge:** `vulkanforge bench --runs 3` (median over 3 runs).
- **llama.cpp:** `llama-bench -n 128 -p 64,128,512 -ngl 99 -dev <RoCm0|Vulkan0>`,
  build `23b8cc4`.
- **vLLM:** 0.20.1.dev0+g88d34c640.
  - Decode/prefill isolation: latency benches with `--output-len 1`
    (prefill) and `--output-len 128` (full) — subtract to derive
    per-step decode time. Engine-logger throughput is averaged over
    a 10 s window and is not a reliable per-stage signal (see
    `results/v038_bench_comparison.md` for the methodology bug we
    avoided).
- **Power:** 10 Hz hwmon sampling, first 20 samples skipped as warm-up.
- **Decode metric:** 1-token prompt + 32 generated tokens, greedy.
- **Prefill metric:** `pp=N` = N-token prompt, no generation.
- **Coherence:** every native FP8 path verified with `vulkanforge chat`
  before reporting performance numbers (Sprint 38 P2 honest-negative
  showed why bench-tok/s is not a correctness signal).

Raw bench logs and power CSVs are persisted in `results/`:

- `results/v038_bench_comparison.md` — full Sprint 34C/D + Sprint 38-Bench data
- `results/v039_sprint38_fp8_native_wmma.md` — Sprint 38 Part 1 (per-tensor / per-channel)
- `results/v039_sprint38p2_blockwise_native.md` — Sprint 38 Part 2 honest-negative
- `results/v039_sprint39_blockwise_act_quant.md` — Sprint 39 fix + activation-quant
- `results/sprint34cd_logs/`, `results/sprint38_part1_logs/`, `results/sprint38p2_logs/`,
  `results/sprint38_vllm_logs/`, `results/sprint39_logs/` — bench + power raw data
- `results/sprint39c_decode_power_logs/` — Sprint 39C decode-only power
  measurements (chat-driven for VF, latency `in=1 out=128` for vLLM)
