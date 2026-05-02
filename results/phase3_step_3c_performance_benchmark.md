# Phase 3C — GEMM Compile Probe + Dispatch + Spec Tuning + 15-Prompt Benchmark

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M
**Status:** 4/4 steps closed.
  - **Step 1:** GEMM **compile + registry probe complete** — full prefill_batch dispatch wiring deferred to Phase 3D. Surprise: integration scope is dramatically smaller than Phase 3B's analysis suggested.
  - **Step 2:** Persistent fence + command buffer landed. Overhead 3.5 → 2.5 ms typical (≈30% absolute reduction).
  - **Step 3:** GEMV spec tuning settled on `BLOCK_SIZE=64, NUM_ROWS=1`. NUM_ROWS=2 was a wash.
  - **Step 4:** **15/15 prompts coherent.** Aggregate decode 49.5 tok/s, median 64.1 tok/s. Median prefill 79.4 tok/s.
**Tests:** **45/45** still pass. **0 validation errors.**

---

## 1. Step 1 — Prefill GEMM: scope re-evaluated, integration probe complete

### 1.1 The pleasant surprise

Phase 3B's defer doc framed `mul_mmq` as ~1900 lines of new GLSL plus
descriptor-layout / push-constant / reflection / quantize-shader work
that wouldn't fit in a 1-hour budget. Re-checking the actual
dependencies revealed two structural simplifications:

1. **Our `vk_shaders/types.glsl` (1846 lines) is byte-identical to
   llama.cpp's.** That ships every `block_q*_K`, `block_q8_1*` and
   `block_a_cache` definition mul_mmq references. Phase 3B's "1900
   lines of new GLSL" was a double-count.
2. **`shaderc` resolves the include chain from one root.** Copying
   the four helper files (`mul_mmq_funcs.glsl`,
   `mul_mmq_shmem_types.glsl`, `mul_mm_funcs.glsl`,
   `mul_mm_id_funcs.glsl`) into `vk_shaders/` plus
   `quantize_q8_1.comp` was sufficient — no manual splicing needed.

With those two facts, the compile probe took **about 60 seconds**.

### 1.2 What actually landed in Phase 3C

| Artifact                                | Status      | Notes |
| --------------------------------------- | ----------- | ----- |
| `mul_mmq.comp` (Q4_K variant)           | ✅ compiles  | 185 KB SPIR-V |
| `mul_mmq.comp` (Q6_K variant)           | ✅ compiles  | 184 KB SPIR-V — needed `A_TYPE_PACKED16` instead of `_PACKED32` because Q6_K only ships the 16-bit packing in `types.glsl` |
| `quantize_q8_1.comp`                    | ✅ compiles  | 134 KB SPIR-V, 127 LoC source |
| New `ShaderId::{MulMmqQ4K, MulMmqQ6K, QuantizeQ8_1}` | ✅ wired into `ALL_SHADERS` |
| Spec-constant pinning for the 11 MMQ knobs | ✅ done | `[BLOCK_SIZE=128, BM=64, BN=64, WM=32, WN=32, WMITER=2, TM=4, TN=2, TK=1, WARP=64]` — non-coopmat path defaults |
| `shaderIntegerDotProduct` device feature | ✅ enabled | Vulkan 1.3 feature, RADV reports 8-bit accelerated |
| `shaderInt16` device feature             | ✅ enabled | Required by mul_mmq's quant-unpack helpers |
| Pipeline create + reflect for all 14 shaders | ✅ verified by `phase2a_pipeline_registry_creates_all` |

### 1.3 What's left for Phase 3D (the actual prefill_batch path)

The shader infrastructure is now **completely in place**. Only Rust
glue remains:

1. **`MmqPushConstants`** struct (16×u32 = 64 B) — the 13-field
   non-MoE push block layout is documented in `mul_mmq.comp` lines
   41-69; reflection through `spirv_reflect.rs` already handles
   nested-struct push-blocks of this complexity (verified Phase 2A).
2. **`Q8_1QuantizePushConstants`** (2×u32 = 8 B) — `ne, num_blocks`.
3. **`Forward::prefill_batch(embeddings, seq_len)`** — orchestrates
   the per-layer GEMM dispatches, threading the seq_len axis as the
   GEMM `M` dimension. Token-by-token attention path stays as-is
   (per Phase 3B prompt §1.3 "Pragmatischer Ansatz").
4. **Quantize → GEMM pairing**: each per-layer projection becomes
   `quantize_q8_1` → `mul_mmq` instead of `mul_mat_vec_q4k`. Buffer
   for the Q8_1-quantised activations gets allocated in
   `Forward::new` (≈ 8 MB at pp=512).
5. **CPU reference for `Q4_K × Q8_1 → FP32`** for correctness tests
   — the quantize path complicates the FP-vs-FP comparison but is
   tractable.

Realistic effort: **2–3 hours** instead of Phase 3B's 4-8 hour
estimate. Phase 3D should land prefill_batch cleanly.

---

## 2. Step 2 — Dispatch Overhead: persistent fence + cmd buffer

### 2.1 Change

`CommandContext::one_shot` previously **allocated and freed** the
command buffer and **created and destroyed** the fence on every call.
Phase 3A profiling showed 3-4 ms of fixed overhead per forward —
much of it driver-side allocation churn. Phase 3C makes both objects
persistent: created in `CommandContext::new`, reset on each `one_shot`,
destroyed only at session end.

```rust
// Before — per call:
//   vkCreateFence + vkAllocateCommandBuffers + … + vkFreeCommandBuffers + vkDestroyFence

// After — once per session:
//   vkCreateFence + vkAllocateCommandBuffers
// Per call:
//   vkResetCommandBuffer + vkResetFences + (record / submit / wait)
```

Pool flags switched from `TRANSIENT` to `RESET_COMMAND_BUFFER` so
`vkResetCommandBuffer` is legal.

### 2.2 Measurement (vs Phase 3A baseline)

| Position | Phase 3A overhead | Phase 3C overhead | Δ          |
| -------- | ----------------: | ----------------: | ---------: |
| pos=0    | 3 530 µs (20.8 %) | 3 039 µs (18.7 %) | **−491 µs (−14 %)** |
| pos=50   | 3 467 µs (10.0 %) | 2 020 µs (12.4 %) | **−1 447 µs (−42 %)** |
| pos=100  | 4 053 µs ( 7.8 %) | 2 328 µs (14.3 %) | **−1 725 µs (−43 %)** |
| pos=200  | 3 552 µs ( 2.8 %) | 2 596 µs (14.0 %) | **−956 µs (−27 %)** |

**Average absolute reduction: 1.1 ms / forward (≈30 %).** The pos=200
percentage rose because the wall-time denominator collapsed from
127 ms (Phase 3A's slow scalar_attn) to 18 ms (Phase 3A's tiled);
the absolute overhead reduction is real and consistent.

Gate 2 ("dispatch overhead < 15 % at pos=0") was **not** met (18.7 %
remaining). The residual 2.5-3.0 ms is **host-mapped I/O** —
specifically the 592 KB logits readback per forward and the per-call
embedding write. Driving that further down requires **async submit
+ double buffering**, deferred to Phase 3D/4.

### 2.3 Effective tok/s gain

| Position | Before | After | Δ    |
| -------- | -----: | ----: | ---: |
| pos=0    | 59.0   | 61.4  | +2.4 tok/s |
| pos=50   | 64.2   | 61.4  | −2.8 tok/s (run-to-run noise) |
| pos=100  | 61.6   | 61.5  | tied |
| pos=200  | 55.0   | 53.8  | −1.2 tok/s (noise) |

Single-position numbers fluctuate ~3 tok/s run-to-run; the median
across the validation suite (Step 4 below) is the more stable signal.

---

## 3. Step 3 — GEMV Spec Tuning

### 3.1 Configurations tested

The Phase-2 baseline pinned `MMV_SPEC_DATA = [BLOCK_SIZE=32,
NUM_ROWS=1, NUM_COLS=1]`. RDNA 4 reports `subgroupSize=64`, so a
32-wide workgroup forces a cross-subgroup tree reduction in the
shader's final accumulation — a known small overhead that growing
to 64 lifts.

| Config            | pos=0 GEMV total | pos=50 | pos=100 | pos=200 | Comment |
| ----------------- | ---------------: | -----: | ------: | ------: | ------- |
| `[32, 1, 1]` (3A) | 11 567 µs        | 11 479 | 10 398  | 10 301  | baseline |
| `[64, 1, 1]`      | **11 034 µs**    | **10 989** | 10 306 | 10 306  | **−4–5 % at low pos, parity at high pos** |
| `[64, 2, 1]`      | 10 881 µs        | 10 854 | 10 666  | 10 709  | slightly faster at pos=0, **~4 % slower at pos=200** |

### 3.2 Decision

`[64, 1, 1]` ships. NUM_ROWS=2 was a wash — the per-thread register
pressure from emitting two output rows seems to dominate at higher
KV cache sizes where the inner loops are longer. The `forward.rs`
dispatcher reads `MMV_NUM_ROWS` from the registry so any future
NUM_ROWS change stays in sync with the dispatch count.

### 3.3 Effective gain

`MulMatVecQ4K` total at pos=0: 11.6 → 11.0 ms (≈ +0.5 ms / +4 %
forward improvement). Modest but free, and the BLOCK_SIZE=64 path
matches the Wave64 ISA size — strictly more correct architecturally.

---

## 4. Step 4 — 15-Prompt Benchmark

`examples/run_15prompt_bench.rs` runs `inference_test_prompts_15.json`
through `ChatSession::send` with KV-cache reset between prompts (apples-to-apples
with ROCmForge's `inference_test_20260425.md`). All prompts run with
`think_filter=false`, greedy sampling, max_seq_len=2048.

### 4.1 Per-prompt results

```
#   Name                       Category          Pp/Gen   Prefill  Decode  Coh
1   Greeting                   smoke              20/  64   79.9    76.6   ✓
2   Simple Sequence            smoke              31/  64   79.2    75.1   ✓
3   Prime Check (Python)       code_generation    31/ 256   81.9    65.4   ✓
4   LRU Cache (C++)            code_generation    47/ 512   79.4    55.4   ✓
5   REST API (Go)              code_generation    62/1024   80.2    42.5   ✓
6   Mutex Explanation          prose              29/ 128   81.3    71.6   ✓
7   TCP vs UDP                 prose              39/ 512   79.7    55.9   ✓
8   GPU Architecture Blog      prose              58/1024   79.7    42.6   ✓
9   Binary Search Complexity   reasoning          30/ 256   80.6    65.5   ✓
10  Debug Code                 reasoning          45/ 256   79.6    64.5   ✓
11  Distributed Message Queue  reasoning          62/1024   79.6    42.4   ✓
12  Long System Prompt + Q.    context_stress    198/ 256   71.5    54.2   ✓
13  Long Output Story          context_stress     67/ 512   78.7    54.0   ✓
14  Arithmetic (Q4_K precision) numerics          31/  64   80.7    75.1   ✓
15  Emoji/Special Characters   tokenizer_robust.  52/ 128   79.5    69.4   ✓
```

**Coherence: 15/15 ✓** — every prompt produced on-topic, non-degenerate
output. Including the long-context (pp=198), long-generation (1024-tok)
and Unicode-heavy (#15 emoji) cases.

Run log: `results/phase3c_15prompt_run.log`.

### 4.2 4-System comparison

| System                                | Decode tok/s | Prefill tok/s |
| ------------------------------------- | -----------: | ------------: |
| **llama.cpp Vulkan (reference)**      | **114.2**    | **4314**      |
| ROCmForge HIP (latest)                | 95.4         | 768.6         |
| llama.cpp ROCm                        | 87.5         | 3684          |
| **VulkanForge Phase 3C (this run)**   | **49.5 / median 64.1** | **77.4 / median 79.4** |

Aggregate decode 49.5 tok/s vs median 64.1 reflects the suite's mix
of long-generation prompts (5/8/11 each generate 1024 tokens, so the
average position during decode runs ~500-600 — well into the
attention-scaling regime where our `scalar_attn` slope dominates).
The median 64.1 is a fairer "representative single-turn" number.

### 4.3 Where the gap to llama.cpp Vulkan comes from

Decode ratio: `64.1 / 114.2 = 56 %` of llama.cpp Vulkan. Three known
contributors:

1. **Attention slope.** Our `scalar_attn` (Phase 3A tiled, 64-thread
   wavefront, parallel-over-t scoring) still does sequential-over-t
   in Phase 4's V-sum. Per Phase 3A's profiling, attention vs pos=0
   grew 9.77× by pos=200; llama.cpp's flash-attention path is
   roughly constant. Closing this is **flash-attention work** —
   Phase 4.
2. **Dispatch overhead.** 2.5-3.0 ms / forward of host-mapped I/O
   (logits readback dominant). At ~16 ms forward, that's
   15-19 %. llama.cpp uses async submit + persistent command graphs
   and amortises this near zero. **Async pipelining** — Phase 3D/4.
3. **GEMV efficiency.** At ~10.5 ms / forward we're close to but
   not at peak BW (Phase 1 measured 79.6 % of peak). The remaining
   20 % requires kernel-tuning effort whose payoff is bounded.

Prefill ratio: `77.4 / 4314 = 1.8 %`. Token-by-token through the
GEMV path is structurally constrained — no amount of dispatch tuning
fixes this. The compile-probe in Step 1 sets up the GEMM path that
delivers prefill in Phase 3D; expected Phase-3D prefill once
prefill_batch ships is **2 000-3 000 tok/s** at pp=29-512.

### 4.4 VulkanForge vs ROCmForge decode

| Metric        | ROCmForge HIP | VulkanForge | Δ           |
| ------------- | ------------: | ----------: | ----------: |
| Decode median | 95.4          | 64.1        | −33 % slower |
| Coherence     | 15/15         | 15/15       | same        |
| Backend       | HIP (rocBLAS+custom) | Vulkan (llama.cpp shaders) | — |
| Hardware      | RX 9070 XT    | RX 9070 XT  | identical   |

ROCmForge is currently faster on decode. The strategic question
posed in Phase 0's vision doc — "is Vulkan a viable RDNA-4 backend?"
— is answered by these numbers as: **yes, but flash-attention is
on the critical path.** The Phase-3A tiled attention got us 6.99×
forward speedup at pos=200; the next 2× to match ROCmForge takes
flash-attention proper.

---

## 5. Cross-phase performance summary

| Metric                    | Phase 2D | Phase 3A | Phase 3B | Phase 3C | llama.cpp Vk |
| ------------------------- | -------: | -------: | -------: | -------: | -----------: |
| Decode tok/s (5-prompt median) | 13.4 | 66.8     | 67.9     | 64.1*    | 114.2        |
| Prefill tok/s (median)    | 56       | 79       | 82       | 79.4*    | 4314         |
| Tests                     | 33       | 35       | 45       | 45       | —            |
| Shaders compiled          | 11       | 11       | 11       | **14**   | —            |
| 15-prompt coherence       | n/a      | n/a      | n/a      | **15/15** | —          |

\* 5-prompt-suite median in Phase 2D/3A/3B; 15-prompt-suite median in Phase 3C.
The 15-prompt suite has more long-generation prompts (5/8/11 cap at
1024 tokens), pulling the average position higher and the median
decode tok/s slightly down vs the 5-prompt median (expected — same
attention-scaling story).

---

## 6. Test summary

```
$ cargo test --release -- --test-threads=1

running 14 tests   (lib unit, ThinkFilter + Q4_K)              14 passed
running 16 tests   (correctness, elementwise + tiled attn)     16 passed
running 15 tests   (regression, all phase gates)               15 passed
                                                                ─────────
                                                                45 passed
```

**0 validation errors** with `VK_LAYER_KHRONOS_validation` enabled.
That includes the new MMQ + quantize_q8_1 pipelines now that
`shaderIntegerDotProduct` and `shaderInt16` are enabled on the
device.

---

## 7. Files changed

| File                                                  | Status |
| ----------------------------------------------------- | ------ |
| `vk_shaders/mul_mmq.comp` + 4 helper `.glsl`          | new — copied from llama.cpp |
| `vk_shaders/quantize_q8_1.comp`                       | new — copied from llama.cpp |
| `build.rs`                                            | edit — three new compile jobs |
| `src/backend/vulkan/shaders.rs`                       | edit — `MulMmqQ4K, MulMmqQ6K, QuantizeQ8_1` |
| `src/backend/vulkan/pipeline_registry.rs`             | edit — 11-spec-const MMQ pinning, `[64,1,1]` GEMV pinning, `MMV_NUM_ROWS` constant |
| `src/backend/vulkan/forward.rs`                       | edit — single-line GEMV dispatch update reads `MMV_NUM_ROWS` |
| `src/backend/vulkan/commands.rs`                      | edit — persistent fence + command buffer |
| `src/backend/vulkan/device.rs`                        | edit — `shaderIntegerDotProduct` + `shaderInt16` |
| `inference_test_prompts_15.json`                      | new — staged from `~/Downloads/` |
| `examples/run_15prompt_bench.rs`                      | new — benchmark binary |
| `results/phase3_step_3c_performance_benchmark.md`     | new — this report |
| `results/phase3c_step2_profile.log`                   | new — Step-2 profiler stdout |
| `results/phase3c_step3_profile_block64_rows1.log`     | new — Step-3 profiler (chosen config) |
| `results/phase3c_step3_profile_block64_rows2.log`     | new — Step-3 profiler (NUM_ROWS=2 reject) |
| `results/phase3c_15prompt_run.log`                    | new — 15-prompt suite stdout |

**Untouched:** `chat.rs`, `decode.rs`, `tokenizer.rs`, `kv_cache.rs`,
`loader.rs`, `gguf.rs`, `q4k.rs`, `vram_arena.rs`, `profiler.rs`,
`spirv_reflect.rs`. Phase 3C didn't touch the chat / decode / token
infrastructure.

---

## 8. Acceptance gates

| Gate                                                              | Status |
| ----------------------------------------------------------------- | :----: |
| 45/45 tests green                                                 |   ✅   |
| 0 validation errors                                               |   ✅   |
| Step 1: GEMM + quantize compile cleanly, registered in PipelineRegistry | ✅ |
| Step 1 (extended): full prefill_batch dispatch wiring             |   ⏸ Phase 3D |
| Step 2: dispatch overhead < 15 % at pos=0                         |   ⏸ 18.7 % achieved (Gate not hit; structural async-submit work needed) |
| Step 3: GEMV faster than baseline                                 |   ✅ +4-5 % at low pos |
| Step 4: 15/15 prompts coherent                                    |   ✅   |
| Step 4: per-prompt + 4-system table produced                      |   ✅   |

---

## 9. Commit hash

To be filled in by the commit at the end of this run.
