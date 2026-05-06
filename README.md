# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

> This project builds on the foundational work of
> [oldnordic](https://github.com/oldnordic/ROCmForge).
> Without his original ROCmForge implementation — the model
> loader, the CPU inference path, the GGUF parser, and the
> overall architecture — none of the WMMA matrix-core
> optimisations, the multi-model support, or the interactive
> chat CLI would have been possible. Thank you for making
> this project a reality.

## Status

**v0.3.9 — Native FP8 WMMA on Mesa 26.1+ — 1197 tok/s prefill on 8B.**
Per-tensor and per-channel FP8 models (Llama-3.1-FP8, Qwen2.5-FP8) get
a new `V_WMMA_F32_16X16X16_FP8_FP8` cooperative-matrix path that
removes the FP8→BF16 conversion step from the K-loop. Llama-3.1-8B-FP8
prefill: **757 → 1197 tok/s (+58 %)** at pp=512. Qwen2.5-14B-FP8:
**325 → 450 tok/s (+39 %)**. Decode is unchanged (GEMV at M=1, no WMMA).
Block-wise FP8 (Qwen3-FP8) keeps the Sprint-36 BF16 scale-fold path —
a native block-wise shader was prototyped (Sprint 38 Part 2) but stays
disabled because the naive FP32→FP8 cast on the activation tile loses
too much dynamic range for block-wise weights (vLLM and llama.cpp's
W8A8 Block FP8 paths quantize activations dynamically; we don't yet).
The native shader is in tree, ready for the activation-quantization
pass to land. Opt-in via `VF_FP8_NATIVE_WMMA=1`, default OFF for
Mesa 26.0.x compatibility.

### Driver Requirements

- **Mesa 26.0.6+** — full GGUF + FP8 support (BF16 WMMA path on every
  FP8 sub-type, including block-wise).
- **Mesa 26.1+** — enables native FP8 WMMA for per-tensor and
  per-channel FP8 prefill (+37–58 %) via `VF_FP8_NATIVE_WMMA=1`.
  Required Vulkan feature: `shaderFloat8CooperativeMatrix = true`
  (check via `vulkaninfo | grep shaderFloat8CooperativeMatrix`).
- Kernel parameter `amdgpu.lockup_timeout=10000,10000` recommended for
  14B+ models (avoids the 2 s compute timeout on long prefill submits).

On Mesa 26.0.x the native flag is unsafe (pipeline build fails on the
missing FP8 cooperative-matrix capability). VulkanForge does **not**
auto-detect the Mesa version — leave `VF_FP8_NATIVE_WMMA` unset on
26.0.x and the BF16 conversion path runs everywhere.

**v0.3.8 — Block-wise FP8 (Qwen3-FP8) — 770 tok/s prefill.** Adds
support for `weight_block_size: [128, 128]` FP8 quantization, the
format used by every official Qwen3 / Qwen3.5 / DeepSeek-V3 FP8
release on HuggingFace. Qwen3-8B-FP8 runs at **64.5 tok/s decode,
770 tok/s prefill at pp=512**, matching the per-tensor 8B Llama path
on prefill (770 vs 757) — block-scale fold during the A-tile load
has no measurable steady-state cost. All three FP8 scaling strategies
(per-tensor, per-channel, block-wise) auto-detected from SafeTensors
metadata. Recommended driver: **Mesa 26.1+** with
`amdgpu.lockup_timeout=10000,10000` for 14B+ models (avoids the 2 s
compute-timeout on long prefill submits).

**v0.3.7 — FP8 prefill +113%.** Cooperative-matrix tiling rewrite
of the FP8 GEMM kernel takes 14B Qwen2.5-FP8 prefill at pp=512 from
159 to 338 tok/s — the first real performance sprint since
Sprint 25B. Phase 0 per-shape diagnostics identified FFN GEMMs at
7-18% TFLOPS efficiency; Phase 1's BN=32 / BLOCK_SIZE=512 / 4×2
subgroup-grid kernel doubles WMMA work per activation read and
recovers most of that headroom. A BN=64 follow-up was tested and
kept as `VF_FP8_GEMM_BN=64` opt-in (occupancy-limited at this
shape; ~3% slower than BN=32). Decode and GGUF unchanged.

**v0.3.6 — architecture cleanup release.** SubgroupAdd reduction
in all 4 GEMV shaders (LDS 4096 → 0 B per WG, occupancy ceiling
6/16 → 16/16 wavefronts/SIMD), fp8pc descriptor-set cache (pool
524 288 → 1024 sets, −33 MiB descriptor metadata), lm_head harness
pipeline with `VF_LMHEAD_HARNESS=0` A/B toggle. Three optimization
sprints (29B, 30, 31) confirmed the 14B FP8 decode bottleneck sits
below the application layer; a Mesa bug report is filed upstream
and the runtime gap is monitored, not optimised further at the
application level.

**v0.3.5 — first 14B FP8 model on a 16 GiB consumer GPU.**
Qwen2.5-14B-Instruct-FP8 (per-channel scaling, Qwen2 architecture)
runs coherently at **13.77 GiB VRAM, 14.1 tok/s decode, 169 tok/s
prefill @ pp=512** — 15/15 on the canonical coherence suite,
including exact arithmetic (17×23=391), C++/Go/Python code
generation, multilingual prose, and emoji input. No other
open-source inference engine ships this model on this hardware.
Adds per-channel FP8 quantization (SSBO scale-vector kernels),
Qwen2 architecture support (Q/K/V bias-add, ChatML detection,
rope_theta from `config.json`), and a logits-buffer architecture
refactor (GpuOnly + host-mapped staging copy). All v0.3.4 GGUF
Q4_K_M / Q3_K_M decode wins carry forward; Llama-3.1-8B-FP8 stays
at 68 tok/s. **14B FP8 decode is correctness-first, not yet
optimized** — see "FP8 Performance Notes" below for the honest
breakdown.

### Decode performance (tok/s, higher is better)

| Model + quant                       | VF FP16-KV | VF FP8-KV  | llama.cpp Vulkan | VF best / lc.cpp |
|-------------------------------------|-----------:|-----------:|-----------------:|-----------------:|
| Qwen3-8B Q3_K_M                     |    131.7   |  **133.7** |     128.7        |   **1.04 ×**     |
| Mistral-7B-Instruct-v0.3 Q4_K_M     |    130.0   |  **131.8** |     124.2        |   **1.06 ×**     |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M |    121.2   |  **122.9** |     117.7        |   **1.04 ×**     |
| Meta-Llama-3.1-8B-Instruct Q4_K_M   |    121.4   |  **122.8** |     117.6        |   **1.04 ×**     |
| Qwen3-8B Q4_K_M                     |    116.9   |  **118.5** |     113.1        |   **1.05 ×**     |

Bench: `vulkanforge bench --runs 3` Decode vs `llama-bench tg128 -r 3`,
RX 9070 XT (gfx1201, RDNA4), RADV Mesa 26.0.6, llama.cpp build
23b8cc4 with `-ngl 99`. Greedy / decode-only, median of 3 runs.

### Prefill performance (tok/s @ pp=512)

| Model + quant                       | VF FP16-KV | VF FP8-KV | llama.cpp |  VF / lc.cpp |
|-------------------------------------|-----------:|----------:|----------:|-------------:|
| Meta-Llama-3.1-8B-Instruct Q4_K_M   |    3 945   |   4 153   |   4 445   |    0.93 ×    |
| Mistral-7B-Instruct-v0.3 Q4_K_M     |    3 963   |   4 052   |   4 491   |    0.90 ×    |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M |    3 958   |   4 198   |   4 426   |    0.95 ×    |
| Qwen3-8B Q4_K_M                     |    3 778   |   3 835   |   4 315   |    0.89 ×    |
| Qwen3-8B Q3_K_M                     |    2 253   |   2 258   |   3 844   |    0.59 ×    |

Q4_K_M family hits 0.89–0.95 × of llama.cpp's prefill. Q3_K_M is
the outlier — Sprint 17B shipped Mmq-only Q3_K (no coopmat
coverage); coopmat-Q3_K is a follow-up.

### Native FP8 E4M3 KV cache

VulkanForge is the first Vulkan LLM engine with **native FP8 KV
cache** via `VK_EXT_shader_float8`. One byte per element instead
of two; 4 packed FP8 values per `uint32` in storage; native
`floate4m3_t` reads in five attention shaders (`flash_attn`,
`flash_attn_split`, `flash_attn_batch`, `flash_attn_tiled`,
`flash_attn_coopmat`). FP32 accumulator unchanged.

| Model         | FP16 KV  | FP8 KV   | VRAM saved | Decode bonus |
|---------------|---------:|---------:|-----------:|-------------:|
| Qwen3-8B      |   288 MB |  144 MB  |    −50 %   |    +1.4 %    |
| Llama-3.1-8B  |   256 MB |  128 MB  |    −50 %   |    +1.2 %    |
| Mistral-7B    |   256 MB |  128 MB  |    −50 %   |    +1.4 %    |

Enable: `VULKANFORGE_KV_FP8=1`. Quality indistinguishable from
FP16: 15 / 15 coherent on `run_15prompt_bench`, multi-turn KV
recall verified end-to-end.

### Native FP8 LLM (SafeTensors)

VulkanForge runs FP8 SafeTensors models end-to-end without unpacking
to FP16/BF16. **All three FP8 scaling strategies** are auto-detected
from `config.json` + the SafeTensors header — just point at the model
directory.

**Mesa 26.1+ with `VF_FP8_NATIVE_WMMA=1` (recommended path on RDNA4):**

| Model                                  | Scaling                  | VRAM (GPU)  | Decode    | Prefill @ pp=512    | Coherent |
|----------------------------------------|--------------------------|------------:|----------:|--------------------:|---------:|
| Meta-Llama-3.1-8B-Instruct-FP8         | per-tensor               |   7.48 GiB  |  69 t/s   | **1197 t/s** (+58 %) |  15/15   |
| Qwen2.5-14B-Instruct-FP8               | per-channel              |  13.77 GiB  |  14 t/s   |  **450 t/s** (+39 %) |  15/15   |
| Qwen3-8B-FP8                           | block-wise [128,128]     |   8.5 GiB   |  64 t/s   |  770 t/s (BF16 path)|    ✓     |

**Mesa 26.0.x fallback (BF16 conversion path) — also default if `VF_FP8_NATIVE_WMMA` is unset:**

| Model                                  | Scaling                  | VRAM (GPU)  | Decode    | Prefill @ pp=512 | Coherent |
|----------------------------------------|--------------------------|------------:|----------:|-----------------:|---------:|
| Meta-Llama-3.1-8B-Instruct-FP8         | per-tensor               |   7.48 GiB  |  69 t/s   |  757 t/s         |  15/15   |
| Qwen2.5-14B-Instruct-FP8               | per-channel              |  13.77 GiB  |  14 t/s   |  325 t/s         |  15/15   |
| Qwen3-8B-FP8 (new in v0.3.8)           | block-wise [128,128]     |   8.5 GiB   |  64 t/s   |  770 t/s         |    ✓     |

Block-wise FP8 (`weight_block_size: [128, 128]`) — used by every
official Qwen3 / Qwen3.5 / DeepSeek-V3 FP8 release — keeps the Sprint
36 BF16 scale-fold path on every Mesa version *and* with the native
flag set. A native block-wise shader using a partial-accumulator scale
trick was prototyped (Sprint 38 Part 2, in tree at
`vk_shaders/mul_coopmat_fp8_native_bn32_blockwise.comp`) — bench
throughput hit +59 % but coherence failed: a naive FP32→FP8 cast on
the activation tile destroys too much dynamic range for block-wise
weights, which were calibrated against per-token-quantized activations
(the way vLLM and llama.cpp's W8A8 Block FP8 paths handle it).
The shader and pipeline stay built so the next sprint can swap them
in once a separate activation-quantization dispatch lands. Per-tensor
and per-channel FP8 win the +37–58 % prefill on Mesa 26.1+ today.
Decode is unchanged across all paths (memory-bandwidth-bound GEMV, no
WMMA at M=1). (Mesa 26.1-rc3, RX 9070 XT. Decode median of 3, prefill
median of 3.)

#### Quick start — Qwen3-8B-FP8

```bash
# Download the model (HF CLI v0.36+: the verb is `hf download`)
hf download Qwen/Qwen3-8B-FP8 --local-dir ~/models/Qwen3-8B-FP8

# Tokenizer comes from the matching GGUF (FP8 SafeTensors don't ship
# the BPE that VF consumes). Any Qwen3 GGUF works.
hf download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf \
  --local-dir ~/models

VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Qwen3-8B-FP8/ \
  --tokenizer-from ~/models/qwen3-8b-q4_k_m.gguf \
  --temperature 0.6
```

#### Quick start — Llama-3.1-8B-FP8 with native FP8 WMMA (Mesa 26.1+)

```bash
# Confirm Mesa 26.1+ (must show `true`)
vulkaninfo | grep shaderFloat8CooperativeMatrix

# Native FP8 WMMA path (+58 % prefill on 8B vs the BF16 fallback)
VF_FP8_NATIVE_WMMA=1 VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Meta-Llama-3.1-8B-Instruct-FP8/ \
  --tokenizer-from ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Without the flag (or on Mesa 26.0.x): BF16 fallback, still works,
# just slower on FP8 prefill.
VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Meta-Llama-3.1-8B-Instruct-FP8/ \
  --tokenizer-from ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Block-wise (Qwen3-FP8): the flag is silently ignored on this path
# until the next sprint lands the activation-quant pass — runs on
# the Sprint 36 BF16 scale-fold path (770 tok/s).
VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Qwen3-8B-FP8/ \
  --tokenizer-from ~/models/Qwen3-8B-Q4_K_M.gguf \
  --temperature 0.6
```

### FP8 Performance Notes

**Prefill is now 2× faster than v0.3.5** thanks to BN=32 cooperative-
matrix tiling (v0.3.7). Qwen2.5-14B FP8 went from 159 to 338 tok/s
at pp=512 — a +113% improvement from one kernel rewrite. llama.cpp
still cannot load FP8 SafeTensors models at all.

- **8B FP8 prefill (Mesa 26.1+, native): 1197 tok/s** — v0.3.9 uses
  `V_WMMA_F32_16X16X16_FP8_FP8` directly; the FP8→BF16 conversion
  step is gone from the K-loop. +58 % over the 757-tok/s BF16
  fallback. Activations are converted FP32→FP8 on the B-side via
  `v_cvt_pk_fp8_f32` (RADV/ACO does not support mixed-type FP8/BF16
  cooperative matrix — both A and B must be FP8).
- **Qwen3-8B FP8 block-wise prefill: still 770 tok/s.** A native
  partial-accumulator shader (Sprint 38 Part 2) was prototyped and
  benched at 1218 tok/s, but coherence failed because the naive
  FP32→FP8 activation cast loses too much dynamic range against
  block-wise-calibrated weights. Shipping it would produce garbage
  output, so the routing falls back to the Sprint 36 BF16 scale-fold
  path even when `VF_FP8_NATIVE_WMMA=1` is set. The native shader is
  in tree, ready for a future sprint that adds per-token activation
  quantization (the way vLLM and llama.cpp do it).
- **14B FP8 prefill (Mesa 26.1+, native): 450 tok/s** — +39 % over
  the 325-tok/s BF16 fallback. The remaining gap to the FP8 peak is
  partly LDS staging (per-step `ds_store_b8` + `ds_load_u8`) and
  partly tile selection — a `GLOBAL_LOAD_TR_B64` pattern-match in
  ACO would let the activation tile bypass LDS entirely.
- **FP8 decode (14-73 tok/s)** — bandwidth-bound. The 14B/8B decode
  gap is a runtime-state interaction confirmed by RGP analysis at
  the wavefront level — not fixable at the application layer. A
  Mesa bug report is filed upstream and the runtime gap is
  monitored, not chased further here.
- **BN=64 opt-in** (`VF_FP8_GEMM_BN=64`): tested in Sprint 33,
  ~3% slower than BN=32 due to occupancy drop (5→3 WGs/CU at
  BLOCK_SIZE=1024 = RDNA4 max). Kept in tree as opt-in for future
  tuning on shapes where the trade-off may flip.

For comparison: GGUF Q4_K_M prefill runs at ~3870 tok/s — the
remaining ~11× gap reflects kernel maturity (GGUF has had months of
tiling optimisation; FP8 got its first real tiling pass in v0.3.7).

Components:

- HuggingFace SafeTensors loader (`compressed-tensors` per-tensor
  format) — single-file or sharded (`*.safetensors.index.json`)
- FP8 GEMV decode kernel (`mul_mat_vec_fp8.comp`, Sprint 20-M2)
- FP8 GEMM prefill kernels (naive + aligned + multi-WG variants,
  Sprints 20-GEMM / 21A / 21B); multi-WG path gates on `m ≥ 64
  && n ≥ 64` to avoid pp ≤ 32 regression
- BF16 → FP16 narrow-load `lm_head` GEMV (`mul_mat_vec_f16.comp`,
  Sprint 22C) — halves the lm_head GEMV's VRAM bandwidth and
  yields the +9 % decode bonus

Run an FP8 chat:

```bash
# 8B FP8 (per-tensor, neuralmagic):
VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Meta-Llama-3.1-8B-Instruct-FP8/ \
  --tokenizer-from ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# 14B FP8 (per-channel, larryvrh/compressed-tensors) — first 14B
# FP8 model on a 16 GiB consumer GPU:
VULKANFORGE_ENABLE_FP8=1 vulkanforge chat \
  --model ~/models/Qwen2.5-14B-Instruct-FP8/ \
  --tokenizer-from ~/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --max-context 1024
```

Per-channel FP8 (`strategy: "channel"`, used by `larryvrh/Qwen2.5-14B-Instruct-FP8`
and other community Qwen2.5-FP8 builds) is **supported as of v0.3.5**
via SSBO scale-vector kernels and dedicated dispatch resources for
the per-channel GEMV path (Sprint 24-Inline). **Block-wise FP8**
(`weight_block_size: [128, 128]`, used by Qwen3-FP8 / DeepSeek-V3-FP8
and the Qwen3.5 family) is **supported as of v0.3.8** via a dedicated
GEMV (`mul_mat_vec_fp8_blockwise.comp`) and a BN=32 cooperative-matrix
GEMM (`mul_coopmat_fp8_bn32_blockwise.comp`) that folds the per-block
scale into the A-tile load (Sprints 35 + 36).

### Multi-architecture support

| Model                                  | Arch / format       | Tokenizer        | Chat template | Status      |
|----------------------------------------|---------------------|------------------|---------------|-------------|
| Qwen3-8B Q3_K_M / Q4_K_M               | qwen3 / GGUF        | gpt2 / qwen2     | ChatML        | ✅ reference |
| Qwen2.5-{0.5B, 7B, 14B} Q4_K_M         | qwen2 / GGUF        | gpt2 / qwen2     | ChatML        | ✅           |
| Meta-Llama-3.1-8B-Instruct Q4_K_M      | llama / GGUF        | gpt2 / llama-bpe | Llama3        | ✅           |
| Meta-Llama-3.1-8B-Instruct-FP8         | llama / SafeTensors | gpt2 / llama-bpe | Llama3        | ✅ native FP8 |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M    | llama / GGUF        | gpt2 / llama-bpe | DeepSeek-R1   | ✅           |
| Mistral-7B-Instruct-v0.3 Q4_K_M        | llama / GGUF        | llama (SPM)      | Mistral       | ✅           |
| Qwen2.5-14B-Instruct-FP8 (per-channel) | qwen2 / SafeTensors | gpt2 / qwen2     | ChatML        | ✅ native FP8 (new in v0.3.5) |
| Qwen3-8B-FP8 (block-wise [128,128])    | qwen3 / SafeTensors | gpt2 / qwen2     | ChatML        | ✅ native FP8 (new in v0.3.8) |

**109 SPIR-V pipelines, 37 lib tests + 40+ GPU correctness tests,
15 / 15 prompts coherent on Qwen3-8B Q4_K_M, Llama-3.1-8B-FP8,
and Qwen2.5-14B-FP8; Qwen3-8B-FP8 coherent on the `<think>`-mode
smoke set.** See `INSTALL.md` for setup.

### Power efficiency vs llama.cpp on RX 9070 XT

GPU power sampled at 10 Hz from `/sys/class/drm/card1/device/hwmon/.../power1_average`,
steady-state average after warmup; full numbers in
`results/v038_bench_comparison.md`.

| Config              | VF tok/s | VF Avg W | llama.cpp tok/s | llama.cpp Avg W | VF tok/s/W | llama.cpp tok/s/W |
|---------------------|---------:|---------:|----------------:|----------------:|-----------:|------------------:|
| 8B Q4_K_M (decode)  |     121  |    209 W |       94 (ROCm) |          310 W  |   **0.58** |              0.30 |
|                     |          |          |       114 (Vk)  |          312 W  |            |              0.37 |
| 8B FP8 (decode)     |      69  |    135 W |       64 (Q8_0 ROCm) |     246 W  |   **0.51** |              0.26 |
|                     |          |          |       73 (Q8_0 Vk)   |     251 W  |            |              0.29 |
| Qwen3-8B FP8 (decode, v0.3.8) | 64 |   156 W |    n/a*         |    n/a          |   **0.41** |              n/a  |

Across every directly comparable 8B config, VF wins decode `tok/s/W`
by 1.6× to 1.9×. Prefill: llama.cpp ROCm wins 6× at 8B-Q8_0 (rocBLAS
GEMM is hard to beat at this shape); VF block-wise FP8 prefill
(770 t/s) is in the same range as VF per-tensor 8B FP8 (757 t/s).

\* Qwen3-FP8 has no llama.cpp equivalent — block-wise FP8 SafeTensors
loading isn't supported by llama.cpp.

### vs vLLM (Qwen3-8B-FP8, single-user decode)

| Engine                  | Decode      | Notes                                            |
|-------------------------|------------:|--------------------------------------------------|
| **VulkanForge v0.3.8**  | **64.5 t/s**| Native block-wise FP8                            |
| vLLM 0.20.1 ROCm        |   ~30 t/s   | Default kernel configs, untuned for gfx1201      |

vLLM is designed for datacenter batch serving and ships no tuned
kernel configs for the RX 9070 XT — it warns "Using default W8A8
Block FP8 kernel config" at startup. With device-specific tuning
and CUDAGraphs (currently disabled on gfx1201) vLLM would land
higher. Numbers reflect out-of-the-box experience on consumer
RDNA4. Methodology in `results/v038_bench_comparison.md`.

### What VulkanForge does that llama.cpp Vulkan doesn't

- **Block-wise FP8 SafeTensors models** (new in v0.3.8) — Qwen3-FP8 /
  Qwen3.5-FP8 / DeepSeek-V3-FP8 with `weight_block_size: [128, 128]`,
  loaded directly without any conversion step. llama.cpp doesn't load
  these formats at all.
- **First 14B FP8 LLM on a 16 GiB consumer GPU** (new in v0.3.5)
  — Qwen2.5-14B-Instruct-FP8, 13.77 GiB VRAM, 14.1 tok/s decode,
  15/15 coherent. Per-channel scale-vector kernels and Qwen2
  bias-add support land here. Performance is correctness-first
  (see "FP8 Performance Notes" above); v0.3.6 will close the
  gap to the 8B's BW efficiency.
- **Native FP8 LLM end-to-end** (v0.3.4) — load HuggingFace
  SafeTensors with `compressed-tensors` per-tensor FP8, run chat
  on a single 16 GiB consumer GPU at 7.48 GiB VRAM /
  68.1 tok/s decode. No FP8→BF16 unpack at load time.
- **Native FP8 E4M3 KV cache** via `VK_EXT_shader_float8` — half
  the cache VRAM, +1–4 % decode, equal coherence (Sprint 18A).
- **3-stage async-pipelined decode** — CPU command-recording
  hidden in GPU compute (Sprint 15E, the +19 % over v0.2.4 that
  put VulkanForge over the llama.cpp line).
- **Single-binary deployment** — one `vulkanforge` binary,
  ~10 MB, no external dependencies beyond Mesa.

### CLI surface (v0.3.1+)

- **`vulkanforge` CLI** with three subcommands — `chat` (REPL with
  sampling flags + `rustyline` editing), `bench` (decode + pp
  sweep), `info` (GGUF metadata + GPU info, no weight upload).
- **GGUF auto-detection + preflight** — `info` works on every GGUF;
  `chat` / `bench` exit cleanly when the architecture or quant
  isn't wired through the forward pass.
- **Sampling** — temperature / top-K / top-P / repetition-penalty
  with auto-seed-from-clock when `--seed` is unset.

### Key features (v0.3.0 engine, v0.3.1 surface)

- **Async pipelined decode loop** (default ON, new in v0.3.0) — the
  CPU records the next token's command buffer while the GPU runs the
  previous token's. 3-stage rolling pipeline:
  ```
  Stage 1: pre_record(CB[N+1])  ← during GPU(CB[N]), 1.8 ms hidden
  Stage 2: wait(CB[N]) → readback → sample → token[N+1]
  Stage 3: write_embed → submit(CB[N+1])
  ```
  Per-token wall drops from 10.9 ms to 9.1 ms; decode goes from
  91 tok/s to **109 tok/s** (+19.3 %, 0.95 × llama.cpp). Vulkan
  records buffer *handles* not contents, so the embedding can be
  written after recording but before submission. Opt-out:
  `VULKANFORGE_DISABLE_ASYNC_DECODE=1`.
- **Double-buffered intermediates** (Sprint 15D infrastructure) —
  17 per-forward scratch buffers (`scratch_a`/`b`, `hidden_norm`,
  Q/K/V projections, attention scratch, FFN scratch, RoPE-pos,
  flash-attention split scratch) extracted into an
  `IntermediateSlot × 2` struct so two CBs can be in different
  pipeline stages without buffer races.

- **KHR cooperative matrix WMMA prefill** (default ON) — Q4_K and Q6_K
  GEMM dispatched through RDNA4's 128 AI Accelerators via
  `VK_KHR_cooperative_matrix`. **S-tile (BM=32) + M-tile (BM=64) +
  L-tile (BM=128)** pipelines with a runtime selector that mirrors
  llama.cpp's `ggml_vk_guess_matmul_pipeline` (`n ≤ 32 → S`,
  `n ≤ 64 → M`, else L). Aligned variant uses `LOAD_VEC_B=8` with
  `B_TYPE=mat2x4` for 4× wider B-matrix loads. Opt out with
  `VULKANFORGE_DISABLE_MM_COOPMAT=1`.
- **f16-accumulator coopmat path** (opt-in via
  `VULKANFORGE_COOPMAT_F16ACC=1`) — FP16 accumulator instead of FP32.
  Default OFF. RDNA4-neutral-to-slightly-negative because the FP16
  fragment is emulated on top of `v_wmma_f32_16x16x16_fp16`. Retained
  for hardware with native f16 accumulator support (NVIDIA Ampere+,
  Intel XMX).
- **Subgroup-arithmetic GEMV reduction** (default ON, new in v0.2.4) —
  K-quant decode GEMVs use `subgroupAdd` over the 64-lane wave instead
  of an LDS tree-reduction. Removes 6 LDS barrier levels from the
  reduction step, matching llama.cpp's RDNA4 GEMV recipe. Wall-time
  delta on this hardware is within noise (the reduction was < 0.2 % of
  per-dispatch time at BLOCK_SIZE=64), but the path is the prerequisite
  for any future GEMV change that depends on a fixed subgroup size.
  Pipeline pins `requiredSubgroupSize=64` via Sprint 14A's plumbing.
  Opt out with `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1`.
- **coopmat QK attention** — KHR cooperative matrix WMMA replaces the
  scalar inner loop in `flash_attn_coopmat.comp`. ~85 % faster prefill
  at pp=2048 vs scalar; resolves the pp=4096 TDR crash.
- **FP16 KV-cache** (default ON) — half the cache VRAM, +21 % prefill
  at pp=2048. Opt out with `VULKANFORGE_FP16_KV=0`.
- **5 fused kernels** — `swiglu`, `multi_add_rms` (×2 sites),
  `rms_norm_mul_rope` — −5 dispatches per layer.
- **Tiled flash-attention** — Br=16 / Bc=32 with online softmax.
- **pp=4096 supported** — previously crashed with TDR.

Gemma-4 is out of scope (different arch, requires Gemma-specific
tensor layout work).

## Performance (RX 9070 XT, gfx1201, RDNA 4)

### Prefill throughput sweep (Qwen3-8B-Q4_K_M, RUNS=5 median)

| pp / decode | v0.2.0 | v0.2.4 | **v0.3.0** | llama.cpp | Ratio (v0.3.0) |
|-------------|-------:|-------:|-----------:|----------:|---------------:|
| Decode      |   90.5 |   91.1 |  **109.0** |     114.2 | **0.95×** |
| pp=32       |    —   |    975 |        975 |       —   | — |
| pp=64       |   1511 |   1678 |       1678 |     2285  | 0.73× |
| pp=128      |   2001 |   2560 |       2570 |     3637  | 0.71× |
| pp=256      |   2200 |   3558 |       3558 |     3995  | 0.89× |
| pp=512      |   2255 |   3863 |   **3865** |     4326  | **0.89×** |
| pp=1024     |   2204 |   3748 |       3742 |     4173  | 0.90× |
| pp=2048     |   1997 |   3172 |       3172 |     3765  | 0.84× |

llama.cpp reference: build 23b8cc4 with `-fa 1` on the same hardware.
**Decode at 109 tok/s = 0.95 × llama.cpp** is the v0.3.0 headline
gain (Sprint 15E async pipeline, +19.3 % over v0.2.4's 91.1).
Prefill peak 3 865 tok/s @ pp=512 is unchanged from v0.2.2 (Sprint
12L's aligned coopmat shipped that figure; v0.3.0's async pipeline
only touches the decode GEMV path). The pp ≤ 128 gap (0.70–0.73 ×)
lives in pipeline-creation infrastructure (subgroup-arithmetic
reduction); the remaining ~5 % decode gap is dedicated `lm_head`
coopmat + buffer-aliasing — see "Limitations".

### 4-system comparison (Qwen3-8B, same hardware)

| System                     | Decode tok/s | Prefill peak tok/s | Decode ratio | Prefill ratio |
|----------------------------|-------------:|-------------------:|-------------:|--------------:|
| llama.cpp Vulkan           |      114.2   |              4326  |       1.00×  |        1.00×  |
| **VulkanForge v0.3.0**     |  **109.0**   |          **3865**  |    **0.95×** |     **0.89×** |
| VulkanForge v0.2.4         |       91.1   |              3863  |       0.80×  |        0.89×  |
| VulkanForge v0.2.0         |       90.5   |              2255  |       0.79×  |        0.52×  |
| llama.cpp ROCm             |       87.5   |              3684  |       0.77×  |        0.85×  |
| ROCmForge (HIP)            |       95.4   |               769  |       0.84×  |        0.18×  |

vs v0.2.4: decode **+19.3 %** (91.1 → 109.0); prefill flat (3 863 →
3 865, run-to-run noise). The decode gain comes from the Sprint 15E
async pipelined decode loop — CPU command-recording (~1 836 µs/token)
now runs in parallel with GPU compute (~9 034 µs/token) of the
previous token, dropping per-token wall from 10.9 ms to 9.1 ms.
**0.95 × llama.cpp Vulkan decode** is the headline figure; the
remaining 5 % gap lives in dedicated `lm_head` coopmat + buffer
aliasing (analysis in Sprint 15B / 15C). ROCm / ROCmForge HIP rows
carry forward from v0.2.0; not re-measured.

## Build

```bash
cargo build --release             # ~2-3 s after first build (SPIR-V is cached)
cargo run --release               # Phase 0 device-init smoke
cargo test --release              # 176 tests across 7 binaries (27 lib, 149 integration)
```

The build compiles 102 SPIR-V binaries (53 in v0.2.0, 65 in v0.2.1,
68 in v0.2.2, 70 in v0.2.3, 72 in v0.2.4, 87 in v0.3.3, +15 in
v0.3.4: FP8 GEMV + 3 FP8 GEMM variants + Q3_K/Q5_K coopmat S/M/L
tiles + FP16 lm_head GEMV).

MSRV is **Rust 1.85** (edition 2024). Build dependencies require a working
`shaderc` install (the `shaderc-sys` crate); on Arch / CachyOS this is
`shaderc` from the official repos. `VK_KHR_cooperative_matrix` must be
advertised by the driver — RADV gfx1201 with Mesa 26.0.5+ qualifies.
For 14B+ models on long prefill, **recommended: Mesa 26.1+ with
`amdgpu.lockup_timeout=10000,10000` on the kernel command line** —
the default 2 s amdgpu compute timeout fires on `pp=1024` 14B prefill
submits (~2.9 s wall). Mesa 26.0.6 also works at smaller pp.

## Run

Three subcommands ship in the `vulkanforge` binary (Sprint 16A):

```bash
vulkanforge --help                 # subcommand list
vulkanforge info  --model <gguf>   # GGUF metadata + GPU info, no weight upload
vulkanforge bench --model <gguf>   # short decode + prefill sweep (greedy)
vulkanforge chat  --model <gguf>   # interactive multi-turn REPL
```

`info` is the safe first call on a new GGUF — it prints architecture,
quantization, dimensions, tokenizer, context length and a support
status without uploading weights to VRAM:

```bash
vulkanforge info --model ~/models/Qwen3-8B-Q4_K_M.gguf
```

`chat` accepts the standard sampling flags (Sprint 16C). Default is
greedy decoding; `--temperature N` (with optional `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`) switches to weighted
sampling:

```bash
# greedy / deterministic (default — same as VF v0.2.x)
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf

# creative, fresh seed each run
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --temperature 0.7 --top-p 0.9 --top-k 40

# creative AND reproducible (pin the seed)
vulkanforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --temperature 0.7 --seed 42
```

| Flag                   | Default | Effect                                        |
|------------------------|---------|-----------------------------------------------|
| `--model`              | `$VF_MODEL_PATH` or `~/models/Qwen3-8B-Q4_K_M.gguf` | Path to GGUF |
| `--system`             | `"You are a helpful assistant."` | System prompt    |
| `--max-tokens`         | 400     | Max tokens generated per turn                 |
| `--temperature`        | 0.0     | `0` ⇒ greedy / argmax; `>0` enables sampling   |
| `--top-k`              | 0       | Keep top-K candidates after softmax (0 = off) |
| `--top-p`              | 1.0     | Nucleus cutoff (1.0 = off)                    |
| `--repetition-penalty` | 1.0     | `>1.0` discourages repeating prior tokens     |
| `--seed`               | clock   | RNG seed; explicit value pins reproducibility |
| `--no-think-filter`    | (on)    | Disable the `<think>…</think>` filter         |
| `--tokenizer-from`     | —       | Borrow `tokenizer.json` from a sibling repo (FP8 SafeTensors only, v0.3.4) |
| `--max-context`        | model default | Override KV-cache capacity for long-context chat (v0.3.4) |

Each flag has a `VF_*` env-var fallback (`VF_TEMPERATURE`, `VF_SEED`,
…) so containerised setups don't need argv plumbing.

`bench` always runs greedy regardless of env state — the 15-prompt
and pp-sweep examples remain the canonical performance harness:

```bash
VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
  cargo run --release --example run_15prompt_bench

VF_MODEL_PATH=$HOME/models/Qwen3-8B-Q4_K_M.gguf \
  cargo run --release --example run_pp_bench
```

## Configuration (environment variables)

### Default-on toggles (set to `0` / `false` / `true` to override)

| Variable | Default | Effect |
|---|---|---|
| `VULKANFORGE_DISABLE_MM_COOPMAT=1` | off (coopmat ON) | Falls back to scalar `mul_mmq` GEMM (v0.2.1 behaviour). |
| `VULKANFORGE_USE_MM_COOPMAT=0` | (legacy alias) | Same effect as `DISABLE_MM_COOPMAT=1`. |
| `VULKANFORGE_COOPMAT_F16ACC=1` | off (FP32 acc) | Opt-in FP16 accumulator for the aligned-L-tile coopmat path. **RDNA4-neutral-to-slightly-negative** (emulated, not native). May benefit NVIDIA Ampere+ / Intel XMX hardware. New in v0.2.3. |
| `VULKANFORGE_DISABLE_SUBGROUP_GEMV=1` | off (Path A on) | Disables the subgroupAdd GEMV reduction (Path A) and falls back to the LDS tree-reduction (Path B). Both paths produce identical results within FP precision. The Path A pipeline pins `requiredSubgroupSize=64` via Sprint 14A's plumbing. New in v0.2.4. |
| `VULKANFORGE_DISABLE_ASYNC_DECODE=1` | off (async ON) | Disables the 3-stage async pipelined decode loop and falls back to the serial path (record → submit → wait → readback per token). Output is bit-identical between modes; the async mode just hides CPU recording inside GPU compute. **New in v0.3.0** — this is the +19.3 % decode lever. |
| `VULKANFORGE_FP16_KV=0` | on | Use FP32 KV cache (2× VRAM, parity with pre-v0.2.0). |
| `VULKANFORGE_KV_FP8=1` | off (FP16 KV on) | **New in v0.3.3.** Use native FP8 E4M3 KV cache via `VK_EXT_shader_float8`. Halves cache VRAM (Qwen3-8B: 288 MB → 144 MB), +1–4 % decode, 15 / 15 prompts coherent on the regression suite. Implies `VULKANFORGE_ENABLE_FP8=1` so device.rs auto-wires the FP8 device feature. Requires RDNA4 + Mesa 26.0+. |
| `VULKANFORGE_ENABLE_FP8=1` | off | **New in v0.3.3.** Enable `VK_EXT_shader_float8` at device-create. Implied by `VULKANFORGE_KV_FP8=1`; set independently for FP8 coopmat smoke testing (`cargo run --release --example fp8_smoke`). |
| `VULKANFORGE_COOPMAT_ATTN=0` | on | Disable coopmat QK attention; falls back to scalar tiled. **DEVICE_LOSTs at pp ≥ 4096** — debugging only. |
| `VULKANFORGE_BATCH_ATTN=0` | on | Per-token attention loop instead of batched. Parity testing only. |
| `VULKANFORGE_CB_REUSE=0` | on | Disable descriptor-set cache; pre-v0.1.0 codepath. |

### Driver-side flags (Mesa 26.1+)

| Variable | Effect |
|---|---|
| `RADV_PERFTEST=cswave32` | Compile compute shaders to Wave32 (enables RDNA4 VOPD dual-issue). Tested in Sprint 13D: ACO emits 3 546 dual-issue instructions, but wall-time is neutral on this workload (memory-bandwidth-bound, not VALU-bound). |

### Sampling (per-run, mirrors `chat` flags)

| Variable | Default | Effect |
|---|---|---|
| `VF_TEMPERATURE` | `0` (greedy) | `0` ⇒ argmax (deterministic); `>0` enables sampling |
| `VF_TOP_K` | `0` (off) | Keep top-K candidates after softmax |
| `VF_TOP_P` | `1.0` (off) | Nucleus cutoff after the post-softmax sort |
| `VF_REPETITION_PENALTY` | `1.0` (off) | `>1.0` discourages prior tokens |
| `VF_SEED` | clock-derived | Pin to make a `>0` temperature reproducible |

The sampler runs repetition-penalty → temperature → softmax → top-K →
top-P → renormalize → weighted draw, in that order (matches
llama.cpp). `temperature=0` short-circuits to argmax; the other
fields are inert in that case.

### GEMM tile-tuning (advanced)

`VULKANFORGE_GEMM_{BLOCK_SIZE,BM,BN,WM,WN,WMITER,TM,TN}` override the
spec-constants used to instantiate `mul_mmq` pipelines. Useful for A/B
tile sweeps without rebuilding SPV.

## Architecture

* `src/backend/vulkan/device.rs` — physical-device pick + queue family.
* `src/backend/vulkan/gguf.rs` — GGUF v3 parser + `ModelConfig` (auto-detects
  rope variant, qk-norm presence, vocab size, etc).
* `src/backend/vulkan/tokenizer.rs` — byte-level BPE for the `gpt2` tokenizer
  model. Picks the correct pre-split regex per `tokenizer.ggml.pre` (`qwen2` or
  `llama-bpe`).
* `src/backend/vulkan/spm.rs` — SentencePiece Unigram tokenizer (Mistral).
* `src/backend/vulkan/chat_template.rs` — `ChatTemplate` enum (ChatML / Llama3
  / DeepSeekR1 / Mistral / Raw) with auto-detection from the GGUF metadata.
* `src/backend/vulkan/forward.rs` — single-token + batched prefill graph.
  `layer_weight_shader_gemm` routes coopmat dispatches across S/M/L
  tiles, aligned/unaligned, and the f16acc opt-in path.
* `src/backend/vulkan/pipeline_registry.rs` — pipeline-layout +
  spec-constants, including the `mul_mm` S/M/L tile warptile blocks
  and the GEMV `MMV_NUM_ROWS` (= 1; NUM_ROWS=2 was tested with
  both LDS and subgroupAdd reductions and reverted in both cases
  on RDNA4).

## Conventions

* Keep `unsafe` blocks scoped to single FFI calls.
* No swapchain, no graphics-queue paths.
* Spec-constants for the GEMV / GEMM / coopmat shaders are pinned in
  `pipeline_registry.rs` — RADV silently produces wrong results when a
  pipeline relies on GLSL defaults.
* Vulkan compute shaders ported from llama.cpp (`mul_mm.comp`,
  `mul_mmq.comp`, `mul_mat_vec_q*_k.comp`) are kept md5-identical to
  upstream HEAD. Performance differences are resolved through
  build-defines, spec-constants, SPV variants, and runtime routing
  rather than shader-source forks.

## Limitations

* Single batch — concurrent sessions need separate `Forward` instances.
* **Decode at 0.80× llama.cpp Vulkan** — coopmat is prefill-only.
  Decode-side coopmat (e.g. `lm_head` GEMV) remains a v0.3 candidate.
* **Remaining ~0.10–0.15× prefill / ~0.20× decode gap** to llama.cpp
  is **structural at the graph level**, not at the shader or
  pipeline-config level. Sprints 12–14 systematically tested and
  falsified **nine** "port llama.cpp's config" hypotheses on RDNA4 +
  this codebase. The remaining levers — multi-submit / command-buffer
  reuse decode loop, dedicated `lm_head` coopmat dispatch,
  buffer-aliasing / live-set reduction, `quantize_q8_1` fusion into
  the GEMM dispatch — are v0.3-class architectural changes.

  | # | Hypothesis | Sprint | Result |
  |---|---|---|---|
  | 1 | Barrier elision (dirty-flag tracker) | 12D | 0 % impact |
  | 2 | Norm + RoPE fusion | 12E | +1 % (run-to-run noise) |
  | 3 | Q6_K shader optimisation | 12H | upstream-identical |
  | 4 | Mesa 26.0.6 → 26.1-rc3 driver upgrade | 13B | ±2 % noise |
  | 5 | f16-accumulator coopmat shader | 13C | −2 % (emulated on RDNA4) |
  | 6 | Wave32 / VOPD dual-issue codegen | 13D | 0 % decode |
  | 7 | `MMV_NUM_ROWS=2` with LDS reduction (Path B) | 13E | −2.9 % |
  | 8 | `subgroupAdd` GEMV reduction (Path A) | 14B | +0.16 % noise |
  | 9 | `MMV_NUM_ROWS=2` with Path A | 14C | −1.5 % |
* All compute shaders ported from llama.cpp (`mul_mm.comp`,
  `mul_mmq.comp`, `mul_mat_vec_q*_k.comp`) are **byte-identical to
  upstream HEAD `23b8cc4`**. Performance differences are configuration,
  not source.
* `VULKANFORGE_COOPMAT_ATTN=0` (explicit opt-out) still DEVICE_LOSTs at
  pp ≥ 4096 — scalar attention exceeds the RADV TDR window at long
  contexts. Default-ON works; opt-out is debugging-only.
