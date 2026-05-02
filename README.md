# VulkanForge

A Vulkan-based LLM inference engine in Rust, targeting AMD RDNA 4 (`gfx1201`).
Compute-only — no swapchain, no graphics queues — built directly on `ash 0.38`
(Vulkan 1.3) rather than a higher-level wrapper.

## Status

**v0.3.2** — multi-arch + K-quant family. **Decode 131.1 tok/s
on Qwen3-8B-Q3_K_M = 1.15 × llama.cpp Vulkan** (and 1.14 × on
Mistral-7B Q4_K_M, 1.06 × on Llama-3.1-8B Q4_K_M); the Qwen3-8B
Q4_K_M baseline stays at 109.0 tok/s = 0.95 × from v0.3.0. Prefill
peak 3 865 tok/s @ pp=512 unchanged.

| Model + quant                       | Decode (tok/s) | vs llama.cpp |
|-------------------------------------|---------------:|-------------:|
| Qwen3-8B-Q3_K_M                     |      **131.1** |   **1.15 ×** |
| Mistral-7B-Instruct-v0.3 Q4_K_M     |          130.0 |       1.14 × |
| Meta-Llama-3.1-8B-Instruct Q4_K_M   |          121.1 |       1.06 × |
| Qwen3-8B-Q4_K_M                     |          109.0 |       0.95 × |

Sprints 17A → 17D extended the engine across architectures and
the K-quant family without touching the v0.3.0 decode-async hot
path:

- **Multi-arch (17A)** — Qwen3 / Qwen2.5 / Llama-3.1 /
  Mistral-7B / DeepSeek-R1-Distill-Llama. The forward pass and
  loader were already generic over tensor names; RoPE variant
  arch-selected; Q/K-norm gated by tensor presence;
  `ChatTemplate::Llama3 / Mistral / DeepSeekR1` had shipped at
  v0.2.x without runtime wiring. One line in
  `inference_support()` + one call to `ChatTemplate::detect()`
  unblocked the family.
- **Q3_K (17B)** — decode GEMV + Mmq prefill shaders, byte-identical
  to llama.cpp upstream. Initial Q3_K_M ship was broken (Sprint
  17B-debug found Q3_K_M actually contains Q5_K weights for
  attn_v / ffn_down — the brief had it wrong). Latent
  `MulMmqQ3KL` L-tile dispatch bug fixed during the debug session.
- **Q5_K (17C)** — completes the K-quant chain (Q3 → Q4 → Q5 →
  Q6). Simultaneously fixed Q3_K_M end-to-end and unlocked Q5_K_S
  / Q5_K_M file_types.
- **Q4_0 infra (17D)** — shader is bit-exact correct, but Qwen2.5
  Q4_0 GGUFs need missing Q/K/V bias-add (architectural) plus
  Q4_1 / Q8_0 shaders for the mixed-quant variants. Q4_0
  preflight stays gated until the Qwen2.5 sprint.

| Model                                | Arch    | Tokenizer        | Chat template | Status      |
|--------------------------------------|---------|------------------|---------------|-------------|
| Qwen3-8B Q3_K_M / Q4_K_M             | qwen3   | gpt2 / qwen2     | ChatML        | ✅ reference |
| Qwen2.5-{0.5B, 7B, 14B} Q4_K_M       | qwen2   | gpt2 / qwen2     | ChatML        | ✅           |
| Qwen2.5-* Q4_0                       | qwen2   | gpt2 / qwen2     | ChatML        | infra ready, gated (Qwen2.5 needs bias-add) |
| Meta-Llama-3.1-8B-Instruct Q4_K_M    | llama   | gpt2 / llama-bpe | Llama3        | ✅           |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M  | llama   | gpt2 / llama-bpe | DeepSeek-R1   | ✅           |
| Mistral-7B-Instruct-v0.3 Q4_K_M      | llama   | llama (SPM)      | Mistral       | ✅           |

**81 SPIR-V pipelines, 40+ GPU correctness tests, 27/27 lib tests,
15/15 prompts coherent on Qwen3-8B Q4_K_M @ 109 tok/s.** See
`INSTALL.md` for setup; sprint reports are in `results/`.

### v0.3.1 surface (still current)

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

The build compiles 72 SPIR-V binaries (53 in v0.2.0, 65 in v0.2.1,
68 in v0.2.2, 70 in v0.2.3 with f16-accumulator coopmat, +2 in
v0.2.4: Q4_K and Q6_K subgroup GEMV variants).

MSRV is **Rust 1.85** (edition 2024). Build dependencies require a working
`shaderc` install (the `shaderc-sys` crate); on Arch / CachyOS this is
`shaderc` from the official repos. `VK_KHR_cooperative_matrix` must be
advertised by the driver — RADV gfx1201 with Mesa 26.0.5+ qualifies.
Mesa 26.1-rc3 is functionally fine (Sprint 13B) but does not improve
performance vs 26.0.6; recommended driver remains **Mesa 26.0.6**.

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
  and the GEMV `MMV_NUM_ROWS` (= 1 — see
  `results/v023_sprint13e_mmv_numrows.md` and
  `results/v024_sprint14c_numrows2_redux.md` for why NUM_ROWS=2
  was tested with both LDS and subgroupAdd reductions and reverted
  in both cases).

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

## Reports

Phase write-ups live in `results/`. Notable v0.2 series:

* `v02_sprint10c_coopmat_qk.md` — coopmat QK attention bring-up.
* `v021_sprint12c_gap_analysis.md` — pre-coopmat-GEMM gap analysis.
* `v021_sprint12gc_rgp_profiling.md` — per-dispatch GPU profiling.
* `v022_sprint12i_prefill_rgp.md` — prefill bottleneck root cause.
* `v022_sprint12k_q6k_coopmat.md` — Q6_K coopmat shader port.
* `v022_sprint12l_sml_tiles.md` — aligned LOAD_VEC_B=8 mat2x4.
* `v022_sprint12m_mtile.md` — M-tile + coopmat default-on.
* `v023_sprint13a_stile.md` — S-tile coopmat (BM=32) for pp ≤ 32.
* `v023_sprint13b_mesa26.1_test.md` — Mesa 26.1-rc3 driver test (neutral).
* `v023_sprint13c_f16acc.md` — f16-accumulator coopmat (opt-in, RDNA4-neutral).
* `v023_sprint13d_wave32_probe.md` — Wave32 / VOPD probe (neutral).
* `v023_sprint13e_mmv_numrows.md` — `MMV_NUM_ROWS=2` GEMV (slight regression).
* `v024_sprint14a_subgroup_size.md` — `requiredSubgroupSize=64` plumbing.
* `v024_sprint14b_subgroup_gemv.md` — subgroupAdd GEMV (Path A, default-on).
* `v024_sprint14c_numrows2_redux.md` — `MMV_NUM_ROWS=2` re-tested with Path A (still regresses, reverted).

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
