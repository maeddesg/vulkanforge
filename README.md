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

- **Server-side memory (v1.0)** — `vulkanforge serve` now has a **persistent, project-scoped, semantic memory**
  embedded in the API process. VF-native endpoints write notes on purpose and read them back by meaning:
  `POST /memory/remember`, `POST /memory/recall` (`{hits:[{id,kind,name,text,status,score}]}`),
  `POST`/`GET /memory/projects`. [SQLiteGraph](https://github.com/oldnordic/sqlitegraph) (nodes + edges +
  per-project HNSW indexes in one SQLite file) + a CPU embedder ([fastembed](https://github.com/maeddesg/fastembed-rs),
  Nomic-Embed v1.5-Q, 768-dim, AVX-512/VNNI). Each project gets its own index — recall in one project **cannot**
  return another's notes — and the store survives restarts (vectors restore with no re-embed). The memory path
  never takes the GPU permit. **First start** downloads the ONNX model into `~/.vulkanforge/embed-cache` (then
  offline); the two native deps add ~34 MB to the binary (static ONNX Runtime + bundled SQLite). What it is, what
  it isn't, and the roadmap: the wiki's **[Memory](https://github.com/maeddesg/vulkanforge/wiki/Memory)** page.
  See `CHANGELOG.md`.
- **`vf-clide` REPL permission ceiling + denial wording (v0.9.4)** — `vf-clide` (0.3.1): in the agent **REPL**,
  tool calls at or below the active `--yes` / `--allow-mutating` / `--allow-shell` ceiling are now
  **auto-approved** (still printed) and only calls **above** it prompt `y/N` — consistent with headless, not
  laxer (workspace confinement still bounds reads/writes independently; **headless `-p` is unchanged**, denying
  above the ceiling). The agent constitution now separates a *permission* denial (lifted by re-running with
  `--allow-*`) from an absolute *workspace-confinement* denial. No engine change. See `CHANGELOG.md`.
- **`vf-clide` token meter + clean `serve` shutdown (v0.9.2)** — `vf-clide` (0.3.0) now shows **live token
  usage** (real server counts on the non-streaming, tool-calling, and streaming paths) and a **pinned status
  line** with a token meter and the current action — a no-op off-TTY, so **headless `-p` stays byte-for-byte
  unchanged**. Engine bugfix: Ctrl+C / SIGTERM on `vulkanforge serve` used to leak GPU objects and **SIGSEGV**;
  shutdown now idles the device, runs the resource teardown in order, and exits cleanly with **0 leaked
  objects** (shutdown-path only — decode is untouched). See `CHANGELOG.md`.
- **Agentic `vf-clide` + engine test-infra hardening (v0.9.0)** — the `vf-clide` client grows from a chat
  client into an **agentic coding client**: an opt-in `--agent` tool loop with **read_file / write_file /
  search / shell**, a **three-tier permission model** (ReadOnly / Mutating / Exec, opt-in via
  `--yes` → `--allow-mutating` → `--allow-shell`, cumulative), **workspace confinement** for the file tools
  (`../` and symlink escapes rejected), and a **constitution** (built-in system prompt + project `AGENTS.md`).
  `shell` is deliberately *not* confined — `--allow-shell` is the explicit opt-in. The engine's end-to-end
  regression + per-shader correctness suites are reactivated and guarded against drift; **no decode/behavior
  change**. See [`vf-clide/README.md`](vf-clide/README.md), the wiki's *vf-clide* page, and `CHANGELOG.md`.
- **Automatic context sizing + Gemma-4 tool-calling + the `vf-clide` client (v0.8.0)** —
  `serve` without `--ctx-size` now picks the largest safe KV context from live VRAM + the model and
  prints what it chose and why (explicit `--ctx-size` still overrides; hardware-capped at 16384 on
  RDNA4). The OpenAI `tools` API now works with **Gemma-4**'s native tool-call format (Qwen/Hermes
  path unchanged). And a new standalone CLI chat client, **`vf-clide/`** (streaming REPL + headless,
  no engine deps), ships alongside. Inference output unchanged. See `CHANGELOG.md`.
- **Prefill parity with llama.cpp Vulkan (v0.7.0, default-on)** — coopmat flash-attention now
  also covers **dense `head_dim=128`** (Qwen3 / Llama-3.1 / Mistral-v0.3 / DeepSeek-R1), and the
  Gemma-4 MoE router gate-projection is batched through the dense GEMM. Dense prefill @p2048
  **0.71–0.78× → 0.96–1.04× llama** (parity; Mistral ahead); Gemma-4-26B-A4B MoE **Q3 0.80→0.89× /
  QAT 0.73→0.83×**. Decode unchanged. Three new default-on paths (`VF_FA_COOPMAT_HD128`,
  prefill-separate MoE router, `VF_MOE_ROUTER_BATCHED`), each with an opt-out. The batched MoE
  router is value-preserving on factual output (a borderline top-k flip can diverge a free-form
  tail; opt out via `VF_MOE_ROUTER_BATCHED=0`). See the matrix below + `CHANGELOG.md`.
- **Gemma-4 prefill 612 → 2629 t/s (v0.6.2, default-on)** — the whole Gemma-4
  attention now runs on a KHR-cooperative-matrix flash-attention kernel (a
  faithful port of llama.cpp's `flash_attn_cm1`: 16×16×16 coopmat QK + PV,
  Br16/Bc64/row_split4, K/V coopMatLoad'd direct-from-global f16). Attention
  drops from 76 % of the prefill wall to ~0 → prefill hits the attention-free
  ceiling: Gemma-4-26B-A4B Q3_K_M **612 → 2629 t/s @p2048** (4.3× over v0.6.1;
  **0.80× llama.cpp Vulkan** at p2048 on the unified matrix below); QAT-Q4_0 hits
  3107 t/s. Occupancy matches
  llama's budget (hd256 occ 12 / hd512 occ 8, 0 spills). Default change
  (coopmat f16-PV vs fa_batch f32, rel ~5e-4, ctx4096-coherent); `VF_FA_COOPMAT_RS=0`
  escapes to fa_batch. Non-Gemma-4 / hd128 untouched (15-prompt regression across
  11 models: no flip regression). See `CHANGELOG.md`.
- **Big-MoE decode +89 % (v0.5.2)** — adaptive load-staging eliminates a
  fixed-buffer VRAM load-transient that was evicting weights to GTT, plus a
  parallel MoE-router top-K (both value-preserving, bit-identical to v0.5.1).
  Gemma-4-26B-A4B Q3_K_M decode **54 → 102.6 tok/s** (was 41 % of llama.cpp;
  v0.6.2 reaches **0.87× llama.cpp** on the same GGUF — see matrix below);
  Qwen3.6-27B **24 → 39.7**. See `CHANGELOG.md`.
- **Software Graph dispatch pipeline (v0.5.0, default-on)** — the
  `LayerStep`-based imperative executor is now scheduled through a
  topologically-sorted dependency graph (`SubDispatch` enum, byte-
  range edge resolution, high-water-mark barrier pass). Validated
  15/15 coherent across 6 release-target architectures (Llama,
  Mistral, Qwen3, DeepSeek-R1, Qwen3.6 GDN, Gemma-4-26B MoE) with
  0 Vulkan synchronization hazards under `validate_sync`. Decode
  +28 % on Qwen3.6 (SG-3 SSM step decomposition); see
  `results/v050_release_bench.md`.
- **Near-parity decode vs llama.cpp Vulkan on RDNA4** — 0.87–0.97×
  on the unified matrix below (same backend); beats vLLM 0.20.1 ROCm
  on FP8 single-user decode (1.3–2× ahead).
- **Native FP8 E4M3** loader that ingests HuggingFace SafeTensors
  directly, no FP16 round-trip on disk.
- **All three FP8 scaling strategies auto-detected**:
  per-tensor, per-channel, block-wise `[128, 128]`.
- **Native FP8 WMMA on Mesa 26.1+** (auto-enabled when the driver
  advertises `shaderFloat8CooperativeMatrix`) — +45–58 % FP8 prefill
  across all three sub-types.
- **CPU `lm_head` offload (v0.3.10)** — Q6_K weights on CPU RAM,
  hand-tuned AVX-512 GEMV (Zen 4). Frees ~970 MB VRAM and on
  **14B FP8 it's 32 % faster than the GPU baseline**
  (17.8 vs 13.5 tok/s).
- **2× better power efficiency** (tok/s/W) on decode vs llama.cpp.
- **Llama-3, Qwen2.5, Qwen3, Mistral, DeepSeek-R1-Distill, Gemma-4**
  model families covered (Gemma-4 SafeTensors path produces coherent
  English with full Markdown structure as of v0.3.14; see
  [docs/MODELS.md](docs/MODELS.md) and the v0.3.14 entry in
  [CHANGELOG.md](CHANGELOG.md) for the 8-bug coherence fix-up plus
  the `forward.rs` refactor that ships alongside it). **v0.6.1 adds
  Q4_0 GGUF support for Gemma-4 — the QAT line (E2B…31B) runs**,
  byte-gated vs llama.cpp (26B-A4B QAT: prefill 1143/646 t/s
  @p512/p2048, decode 117 t/s out of the box). Qwen2.5-Q4_0 stays
  deliberately gated (missing arch features, not the quant).
- **`forward.rs` Refactor (v0.3.14):** the 7816-LOC dispatch file
  splits into 13 sibling modules with a `LayerStep` enum + two
  `LayerExecutor` impls. The Sprint 43F bug class — "added a
  per-layer step in decode but forgot prefill" — becomes a compile
  error in both executors until the new variant is handled.
- **104 / 105 coherent** on the deterministic 15-prompt suite across
  all production configurations — six 8B+ paths plus Gemma-4-E2B
  SafeTensors (14/15; the one miss is a Gemma-tokenizer-emoji
  surrogate). See [Quality](#quality-15-prompt-benchmark) below.

## Quick start

```bash
# GGUF — no flag needed, default-everywhere path (Mesa 26.1+ recommended)
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
`VF_CPU_LM_HEAD=1`) and `--tokenizer-from <gguf>` still work as
explicit overrides — handy when you want CPU `lm_head` on an 8 B
model for VRAM headroom, or want to force a specific tokenizer
source for a regression check.

Native FP8 WMMA was a flag in v0.3.10–v0.3.15
(`VF_FP8_NATIVE_WMMA=1`); v0.3.16 (Sprint 47B) removes the flag and
makes the routing capability-driven — VulkanForge picks the native
FP8 path automatically iff the driver advertises
`shaderFloat8CooperativeMatrix`. Check with:

```bash
vulkaninfo 2>/dev/null | grep shaderFloat8CooperativeMatrix
```

Build from source:

```bash
cargo build --release   # Rust 1.85+, Vulkan headers required
```

## Performance at a glance

All numbers on AMD Radeon RX 9070 XT (gfx1201, RDNA4), RADV/Vulkan.
Full tables with power data and methodology in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### VulkanForge vs llama.cpp — unified bench matrix (v0.7.0)

One run, identical conditions per row, no cherry-picking (Sprint 11i, measured fresh at the
v0.7.0 HEAD). **HW:** RX 9070 XT, gfx1201 (RDNA4). **Driver:** RADV / Mesa 26.1.2-arch2.1
(Vulkan 1.4.303). **VF:** v0.7.0 — coopmat flash-attention (dense hd128 + Gemma-4 hd256/512) +
batched MoE-router gate-proj, all default-on. **llama.cpp:** Vulkan build (`GGML_VULKAN`,
`KHR_coopmat`, *not* HIP), `b9174-g0253fb21f`, `-fa 1 -ngl 99`. **ctx** 4096, greedy, validation
off, warm, median ≥3–5; runs strictly sequential (never both engines at once). Measured 2026-06-09.

| Model | Quant | KV | VF p≈512 | llama p≈512 | VF/ll | VF p≈2048 | llama p≈2048 | VF/ll | VF dec | llama dec | VF/ll |
|---|---|---|--:|--:|:-:|--:|--:|:-:|--:|--:|:-:|
| Qwen3-8B | Q4_K_M | f16 | 4291 | 4472 | 0.96 | 4280 | 4479 | **0.96** | 109.4 | 114.9 | **0.95** |
| Llama-3.1-8B-Instruct | Q4_K_M | f16 | 4474 | 4802 | 0.93 | 4470 | 4644 | **0.96** | 114.5 | 119.6 | **0.96** |
| Mistral-7B-v0.3 | Q4_K_M | f16 | 4845 | 4826 | 1.00 | 4825 | 4654 | **1.04** | 124.2 | 127.5 | **0.97** |
| DeepSeek-R1-Distill-Llama-8B | Q4_K_M | f16 | 4461 | 4785 | 0.93 | 4464 | 4628 | **0.96** | 114.7 | 118.6 | **0.97** |
| gemma-4-26B-A4B (MoE) | Q3_K_M | FP8/q8_0 | 2085 | 3251 | 0.64 | 2862 | 3219 | **0.89** | 110.2 | 127.1 | 0.87 |
| gemma-4-26B-A4B (MoE) | Q4_0 ⁑ | FP8/q8_0 | 2653 | 4140 | 0.64 | 3436 | 4119 | **0.83** | 121.2 | 132.7 | **0.91** |

Prefill is tok/s, decode is generated tok/s. **Prefill lengths (matched per row):** dense exact
512 / 2048; MoE 513 / 2049. **KV:** 8B both f16; 26B = VF-FP8 vs llama-`q8_0` (both 8-bit, nearest
equivalent — the one config difference). **⁑ QAT** = the same file
`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` for both engines; VF labels it `Q4_K_XL` by filename,
llama reads the tensors as `Q4_0`. **MoE prefill is measured with varied tokens** — *not* the
`run_pp_bench` micro-bench, whose single-token-repeat degenerates MoE routing and overstates MoE
prefill by 1.2–1.7×. Net (v0.7.0): **dense prefill 0.93–1.04× llama (parity), Gemma-MoE prefill
0.64× @512 → 0.83–0.89× @2048, decode 0.87–0.97×**. Dense attention (coopmat-FA on hd128, v0.7.0)
and the Gemma router/gate-proj are no longer the bottleneck; the remaining MoE prefill gap is the
grouped expert-GEMM (`MUL_MAT_ID`). **Behaviour note:** the batched MoE router is llama-aligned and
value-preserving on factual/structural output, but a borderline top-k expert flip can make a
free-form generation tail diverge from the pre-v0.7.0 per-token router (deliberate; opt out with
`VF_MOE_ROUTER_BATCHED=0`).

**Gemma quant: `Q3_K_M` vs QAT (`Q4_0`) — no single default.** Gemma-4-26B-A4B runs as both `Q3_K_M`
(3-bit) and the QAT `Q4_0` line (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, quantization-aware-trained,
4-bit); they trade quality, speed, and context — pick by your priority. Measured on the 16 GB reference
card: QAT decode **~119 vs ~108 tok/s** and prefill **~1.25–1.27×** `Q3_K_M`, and QAT is stronger on
harder coding tasks; `Q3_K_M` is smaller, with more context headroom. vs llama.cpp-Vulkan (same Mesa):
QAT prefill 0.63× @512 → 0.86× @~1.8k, decode 0.85× (≈ the v0.7.0 matrix QAT row above). For a three-way
comparison that also includes the dense Qwen3.6-27B, see the wiki:
[Choosing a Model for Coding](https://github.com/maeddesg/vulkanforge/wiki/Choosing-a-Model-for-Coding).
**Context (engine setting, not a model default):** `--max-context 3072` is a good general coding value —
same VRAM as 2048 but enough generation budget for a substantial function (2048 can truncate ~180-line
outputs); `4096` is a spare-VRAM opt-in. The pre-load free-VRAM gate (`VF_VRAM_GATE=1`) guards against a
prior process's un-freed VRAM or compositor pressure. **`VULKANFORGE_KV_FP8=1` is required for the
gemma-4-26B-A4B MoE models** — with a non-FP8 KV cache (F16/F32) these MoE models produce invalid output,
so loading one without it now aborts with a clear, actionable error (override with
`VULKANFORGE_ALLOW_BROKEN_KV=1`); it also helps the 26B models fit 16 GB. Dense and Qwen3.5/3.6 models are
unaffected.

### v0.3.16 15-prompt mixed-workload benchmark

The decode column in the matrix above is `vulkanforge bench`-style tg128
(1-token prompt, constant-low KV). The 15-prompt suite is a real-workload mix (smoke /
code / prose / reasoning / context-stress / numerics / tokenizer) with
generations up to 1024 tokens — KV grows during decode, so steady-state
numbers are below `tg128`. Mesa 26.1.0 RADV.

| Model                              | Prefill avg | Decode avg | Avg W | tok/s/W | Quality |
|------------------------------------|------------:|-----------:|------:|--------:|--------:|
| Qwen3-8B Q4_K_M                    |     **701 t/s** ² |   104.0 t/s |  258 W |  0.40   | 15/15 ✓ |
| Llama-3.1-8B Q4_K_M                |     824 t/s |   **110.0 t/s** |  271 W |  0.41   | 15/15 ✓ |
| Qwen3-8B FP8                       |     559 t/s |    60.7 t/s |  193 W |  0.32   | 15/15 ✓ |
| Gemma-4-E2B-it (FP32 SafeTensors)  |     96 t/s ¹ |    33.7 t/s |  64 W  |  0.53   | 15/15 ✓ |
| **Gemma-4-E2B-it (Q4_K on-load³)** |    **106 t/s** | **52.0 t/s** | **37 W** | **1.39** | 15/15 ✓ |

¹ v0.3.15 lifted the v0.3.14 `force_per_token_prefill` workaround
(33 → 89 → 96 t/s on v0.3.14 / v0.3.15 / v0.3.16). The batch path is
bit-identical to the per-token reference
(`VULKANFORGE_FORCE_PER_TOKEN=1` keeps the v0.3.14 path available
as a bisect fallback). Decode-side is on par with the larger models
on a tok/s/W basis (best in the test) thanks to the 2 B parameter
count keeping power draw at 64 W.

² v0.3.16 closes the v0.3.15 Sprint-46H barrier regression on
Owner-only models (Qwen3, Llama). The Q-side barriers are now
gated on the Gemma-4 subscriber predicate; Qwen3-Q4_K_M prefill
recovers from 638 to **701 t/s** (+9.9 %).

³ v0.3.17 adds on-the-fly Q4_K quantization at model load
(`VF_QUANTIZE_ON_LOAD=1`). Gemma-4 SafeTensors weights are
quantized FP32 → Q4_K_M on the CPU (rayon-parallelized,
`Loaded in 13.2 s`) and routed through the existing Q4_K shader
pipeline. Decode +54 %, power −41 %, **tok/s/W 1.39 — best in the
suite**. VRAM 8.51 → 2.49 GiB (7.1× compression on the quantized
tensors; norms / embeddings stay FP32). Coherence identical to
the FP32 baseline.

### Native FP8 prefill pp=512 (Mesa 26.1+, native FP8 WMMA path)

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
| FP8 model loading    | `VULKANFORGE_ENABLE_FP8=1` | Mesa 26.1+ (or 26.0.6 BF16 path) | Load HuggingFace FP8 SafeTensors    |
| Native FP8 WMMA      | (auto)                     | `shaderFloat8CooperativeMatrix` (Mesa 26.1+) | +45–58 % FP8 prefill |
| CPU `lm_head` offload| `VF_CPU_LM_HEAD=1`         | AVX-512F + BW + VL (Zen 4 / Ice Lake+) | −970 MB VRAM, 14B +32 % decode |
| On-the-fly Q4_K      | `VF_QUANTIZE_ON_LOAD=1`    | SafeTensors model with FP32 / BF16 weights | Quantize 2D weights to Q4_K_M at load; ~7× VRAM compression on quantized tensors, routes through the Q4_K shader pipeline. Gemma-4-E2B: decode +54 %, power −41 %, tok/s/W 1.39 |
| FP8 KV-cache         | `VULKANFORGE_KV_FP8=1`     | Mesa 26.1+ (heterogeneous head_dim auto-handled) | −50 % KV-cache VRAM. Gemma-4-26B-A4B: 880 → 440 MB. |
| KV prefix-reuse (API server, **default OFF**) | `VF_KV_PREFIX_REUSE=1` | `vulkanforge serve` (chat + completions) | Cross-request KV reuse: keeps the last request's KV and re-prefills only the suffix after the longest common token prefix → multi-turn / agentic (growing context) skips re-prefilling the shared history. Value-preserving (byte-identical to full re-prefill @temp 0). Single retained session (last request); errors/cancel invalidate. OFF = v0.4 stateless behavior, bit-identical. |
| MTP timing instrumentation (**default OFF**) | `VF_MTP_TIMER=1` | Qwen3.6-27B-MTP (`VF_MTP=1`) | Per-phase wall-clock + submit breakdown for the (parked, gated-off) MTP self-spec decode loop. Measurement only; MTP itself stays default-OFF. |

> Debug/test-only env flags (never set in production): `VF_KV_REUSE_DEBUG=1`
> (logs the prefix-reuse `k`/prefill-token count per request) and
> `VF_KV_REUSE_OVERSHOOT=N` (the byte-ident gate's teeth-test — intentionally
> over-reuses; do not use outside testing).
| Expert-Grouped MoE prefill (**default ON** since v0.5.3) | (auto); opt-out `VF_MOE_GROUPED=0` | MoE model (Gemma-4-26B-A4B et al.) | +33–39 % prefill on Gemma-4-26B-A4B Q3_K_M (146 → ~199 t/s @pp512), value-preserving, decode-neutral. +~265 MB VRAM (MoE models only; dense/GDN unaffected). `VF_MOE_GROUPED=0` = legacy GPU-direct GEMV prefill. |
| Batched Full-attention prefill layers (**default ON** since v0.6.0) | (auto); opt-out `VF_PREFILL_FULL_BATCHED=0` | Gemma-4-26B-A4B (MoE) | The 5 Full-attention layers stay on the batched prefill path instead of the 51D-AN per-token workaround (1 GPU drain per token per Full layer). Validated bit-identical to an independent per-query attention reference; note: changes default greedy output vs ≤v0.5.9 (FP-reorder, both paths correct). |
| MoE prefill GLU batching (**default ON** since v0.6.0) | (auto); opt-out `VF_MOE_GLU_BATCHED=0` | MoE model (Gemma-4-26B-A4B) | Per-(token,slot) GLU loop (125k mini-dispatches/chunk) → 1 dispatch/layer. Byte-identical output. |
| MoE prefill gather-combine (**default ON** since v0.6.0) | (auto); opt-out `VF_MOE_GATHER_COMBINE=0` | MoE model (Gemma-4-26B-A4B) | Per-(token,slot) scatter-FMA loop + per-dispatch barriers (125k+125k/chunk) → 1 gather dispatch/layer (hazard removed by design). Byte-identical output. **Combined with the two flags above: Gemma-4-26B prefill 198 → 1073 t/s @p512 (5.4×), 178 → 604 @p2048 (3.4×).** |
| Q-tiled flash attention hd256/512 (**experimental, default OFF — NOT recommended**) | `VF_FA_TILED_SLIDING=1` / `VF_FA_TILED_FULL=1` | Gemma-4 | Numerically validated (byte-identical single-tile, long-context recall green) but **slower than the default** (−31…−36 % prefill: KV-traffic goal reached, LDS occupancy lost — see `docs/prefill_arc_2026_06.md`). Research platform for occupancy tuning only. |
| qwen35 decode determinism fix (**default ON** since v0.5.4) | (auto); opt-out `VF_QWEN35_DEC_GATE_BARRIER=0` | Qwen3.6-27B-MTP (qwen35) | Correctness fix: adds the missing gated-output → o-proj barrier so decode reads the fully-gated `attn_out`. Makes greedy decode bit-deterministic (was a RAW race, widened by `head_dim=256`). Decode-neutral, qwen35-only (no effect on other models). |
| qwen35 batched prefill (**default ON** since v0.5.6) | (auto); opt-out `VF_QWEN35_BATCHED=0` | Qwen3.6-27B-MTP (qwen35) | qwen35 prefill runs through the batched executor (GDN + Full-Attn over seq_len=N; cross-token cores serial with state-carry, projections batched GEMM). **4–11× prefill speedup** (grows with prompt length; e.g. 1098-tok 32 → 358 t/s), deterministic + byte-identical to per-token (greedy), decode-neutral. `VF_QWEN35_BATCHED=0` = per-token prefill. (v0.5.5 shipped this opt-in; v0.5.6 fixed a conv-loop barrier-elision race that made it non-deterministic on near-tie prompts, and flipped it on by default.) |
| Batched MoE decode (v0.4.5)         | `VF_MOE_BATCHED_DECODE=1` | MoE model | +4.8 % decode on Gemma-4-26B-A4B Q3_K_M (27.3 → 28.6 t/s). 800 → 450 MoE dispatches/token. |
| VRAM budget probe (v0.4.5)          | (auto)                    | Linux sysfs `/sys/class/drm/card*/device/mem_info_vram_*` | Diagnostic. Warns when free VRAM < `VF_VRAM_HEADROOM_GIB` (default 1.0). |
| Tensor-load progress bar (v0.4.5)   | (auto, suppress with `VF_NO_LOAD_PROGRESS=1`) | — | `\r`-overwritten stderr progress bar during GGUF / SafeTensors upload. |

Most features are opt-in; the v0.5.x defaults (adaptive staging, parallel
MoE-router top-K, Expert-Grouped MoE prefill, qwen35 decode determinism fix)
are on by default and value-preserving. Without flags, VulkanForge runs GGUF models
on Mesa 26.1+ with no special configuration. `VF_FP8=auto` picks
the right FP8 path based on what the driver actually advertises.

## Quality (15-prompt benchmark)

The deterministic 15-prompt suite (greedy decoding, temperature = 0)
on all six production paths:

| Configuration                                    | Coherent   | Median decode |
|--------------------------------------------------|-----------:|--------------:|
| Qwen3-8B Q4_K_M GGUF                             |  **15/15** |     107 tok/s |
| Llama-3.1-8B Q4_K_M GGUF                         |  **15/15** |     112 tok/s |
| Qwen3-8B-FP8 native WMMA + activation quant      |  **15/15** |      62 tok/s |
| Qwen2.5-14B-FP8 native WMMA + CPU `lm_head`      |  **15/15** |      17 tok/s |
| Llama-3.1-8B-FP8 native WMMA                     |  **15/15** |      70 tok/s |
| Llama-3.1-8B-FP8 native WMMA + CPU `lm_head`     |  **15/15** |      46 tok/s |
| **Gemma-4-E2B-it SafeTensors** (v0.3.14, new)    |    **14/15** |      34 tok/s |

**104 / 105 prompts (99 %) coherent across the full suite.**
v0.3.14 adds the Gemma-4-E2B-it SafeTensors path; 14/15 coherent on
the suite, the single miss is the emoji-identification prompt where
the Gemma-4 tokenizer's surrogate-pair handling drops the input
emojis before the model sees them.

v0.3.11 closes the v0.3.10 Llama-FP8 per-tensor edge case (2/15
code-gen prompts collapsing to `!`) by porting the Sprint 39
per-block activation-absmax + rescale pattern to the per-tensor
WMMA path. The fix costs ~5 % on prefill (1197 → 1130 tok/s on
8B-FP8 pp=512) — an unavoidable trade for keeping post-RMS-norm
activations inside the FP8 E4M3 ±448 envelope.

## Driver requirements

| Mesa version | Capabilities                                                                  |
|--------------|-------------------------------------------------------------------------------|
| **26.1+**    | Default. Native FP8 WMMA via `shaderFloat8CooperativeMatrix`                  |
| **26.0.6**   | Legacy. GGUF + FP8 SafeTensors via the BF16 conversion path (no native FP8 WMMA) |

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
vulkanforge serve  --model <PATH> [--host 127.0.0.1] [--port 8080] [--cors]
```

`vulkanforge chat --help` lists every flag (sampling, max-tokens,
think-filter, max-context). The chat REPL accepts `/help`, `/quit`,
`/reset` (clear KV cache + history without reloading the model), and
a single-shot mode via `VF_PROMPT="..."`.

## API Server (v0.4)

OpenAI-compatible HTTP server. Drop-in backend for Open WebUI,
SillyTavern, Continue.dev, the OpenAI Python SDK, LangChain, and
any other client that speaks Chat Completions.

```
vulkanforge serve --model ~/models/Qwen3-8B-Q4_K_M.gguf --port 8080
```

### Endpoints

| Path                         | Method | Description                          |
|------------------------------|--------|--------------------------------------|
| `/v1/chat/completions`       | POST   | Chat (streaming via SSE + sync JSON); **function/tool calling** (Qwen3/Hermes) |
| `/v1/completions`            | POST   | Legacy text-completion (raw prompt, no chat template; SSE + sync JSON) |
| `/v1/models`                 | GET    | List loaded model                    |
| `/health`                    | GET    | Liveness + KV-cache status           |

The non-prefixed aliases `/chat/completions`, `/completions` and
`/models` are also routed for clients that omit the `/v1/` prefix.

**`/v1/completions`** generates from a **raw `prompt` string** with **no
chat template applied** (the only difference from `/v1/chat/completions`;
the sampling surface, streaming, stop, and usage are identical). The
prompt is tokenized with special-token parsing, so a pre-rendered
chat-template string round-trips to the same tokens the chat endpoint
would produce. `prompt` is a single string in v0.5 (array-of-strings and
token-id-array prompts are planned). Accepted-but-**ignored** (warn-logged)
fields: `suffix`, `best_of`, `n>1`, `logprobs`, `logit_bias`, `echo`,
`presence_penalty`. Responses use the `text_completion` object with a
`cmpl-…` id and `choices[].text`.

### Examples

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4_k_m",
    "messages": [{"role": "user", "content": "What is a mutex?"}],
    "max_tokens": 100
  }'

# Streaming (SSE)
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4_k_m",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 50
  }'

# Legacy text-completion — raw prompt, no chat template (non-streaming)
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4_k_m",
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0
  }'

# Legacy text-completion (streaming SSE)
curl -N http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "17 * 23 =",
    "max_tokens": 16,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'

# Function/tool calling (OpenAI-compatible, Qwen3/Hermes models)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-q4_k_m",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "description":"Get current weather for a location",
      "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
    }}],
    "max_tokens": 128, "temperature": 0
  }'
# → choices[0].message.tool_calls:[{id,"type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}], finish_reason:"tool_calls"
# Send the result back as {"role":"tool","tool_call_id":"<id>","content":"15C sunny"} for the final answer.
```

**Function/tool calling** (`/v1/chat/completions`) is OpenAI-compatible for
**Qwen3/Hermes** models: pass `tools` + optional `tool_choice` (`"auto"`/`"none"`);
calls come back as `message.tool_calls` + `finish_reason:"tool_calls"`; send
results back via `role:"tool"` messages. v1 covers the Qwen3/Hermes
`<tool_call>` format; Llama-3.1/Mistral formats, `tool_choice:"required"`/named
forcing, and char-incremental argument streaming are planned. Requests **without**
`tools` are unaffected.

### Options

| Flag                   | Default       | Notes                                              |
|------------------------|---------------|----------------------------------------------------|
| `--host`               | `127.0.0.1`   | Use `0.0.0.0` for remote/Docker (no auth in v0.4)  |
| `--port`               | `8080`        | TCP listen port                                    |
| `--cors`               | off           | Enable CORS (browser UIs on different ports)       |
| `--ctx-size`           | `2048`        | KV-cache capacity in tokens                        |
| `--served-model-name`  | basename      | Override the `model` id reported by `/v1/models`   |
| `--tokenizer-from`     | —             | Reserved for SafeTensors-serve (v0.4.1)            |

### Scope

- **In:** Streaming + non-streaming chat, multi-turn history,
  **`/v1/completions`** (raw-prompt), **OpenAI function/tool calling**
  (Qwen3/Hermes), **cross-request KV prefix-reuse** (`VF_KV_PREFIX_REUSE=1`,
  opt-in), `frequency_penalty` → repetition-penalty mapping,
  `stream_options.include_usage`, `developer` role alias for `system`,
  `chat_template_kwargs.enable_thinking` toggle for `<think>` filtering.
- **Out:** Vision content, embeddings, auth, SafeTensors directory models
  (use `vulkanforge chat` for those); tool-calling beyond the Qwen3/Hermes
  format and `tool_choice` beyond auto/none (follow-ups).

### Using VulkanForge with OpenCode (and OpenAI-compatible agents)

VulkanForge works as a backend for [OpenCode](https://opencode.ai) and other
OpenAI-compatible coding agents (function/tool calling). Validated end-to-end on
**Qwen3-8B Q4_K_M**: real tool calls execute and files are written to disk.

**1. Serve with a raised context** — the 2048 default is too small for an
agent's large system prompt (~7,500 tokens), so set `--ctx-size`:

```fish
vulkanforge serve -m ~/models/Qwen3-8B-Q4_K_M.gguf --port 8080 --ctx-size 16384
```

**2. `~/.config/opencode/opencode.json`:**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "vulkanforge/qwen3-8b-q4_k_m",
  "provider": {
    "vulkanforge": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "VulkanForge (local)",
      "options": { "baseURL": "http://localhost:8080/v1" },
      "models": { "qwen3-8b-q4_k_m": { "name": "Qwen3-8B (VF)" } }
    }
  }
}
```

**Status / Limitations.** Functional: tool calls execute and files are written
(validated end-to-end on Qwen3-8B Q4_K_M). **Not yet optimized:** per-turn
latency is high because VulkanForge re-prefills the agent's large system prompt
each turn — prefill is the current bottleneck (per-turn latency in the tens of
seconds, dominated by prefill, not generation). Recommended model: **Qwen3-8B
Q4_K_M**. The 27B (Q3_K_S) is impractical for agentic use today (prefill cost +
aggressive quantization). Set **`VF_KV_PREFIX_REUSE=1`** to speed up multi-turn
(opt-in; mitigates later turns, not the first).

> **v0.5.8 — Gemma-4 (MoE) now coherent on serve.** Earlier serve builds produced
> garbage on Gemma-4 specifically (the API path skipped the GPU MoE-router init the
> CLI ran, and the GGUF path missed the channel-thought template detection). Fixed in
> v0.5.8; dense models (Llama/Qwen) were never affected. The prefill-latency caveat
> above still applies to the 26B for agentic use.

## Limitations

- Single-stream only — no batch inference, no concurrent sessions on
  one `Forward` instance.
- vs llama.cpp Vulkan (same backend, v0.7.0 matrix): dense models reach
  prefill **parity** (~0.93–1.04×) and near-parity decode (~0.95–0.97×);
  Gemma-4 MoE prefill is still behind (~0.64× at short context, ~0.83–0.89×
  @p2048 — the grouped expert-GEMM), MoE decode ~0.87–0.91×. Coopmat is
  prefill-only on this codebase.
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
