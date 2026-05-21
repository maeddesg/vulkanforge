# VulkanForge v0.4.6 — Qwen3.6 Gated-Delta-Net Support

**First Vulkan inference engine with Gated-Delta-Net (GDN) linear attention support.**

## 🏆 Highlights

### Qwen3.6-27B Support (NEW)

Full port of Qwen3.6 / qwen35 architecture (Qwen3Next):

- **Gated-Delta-Net** linear attention on 48 recurrent layers
- **16 full-attention layers** with Q-Gate-Split projection
- **1 MTP draft head** (loader-aware; runtime use planned for v0.5)
- Persistent SSM state (`conv_state` + `ssm_state`) across decode tokens
- ChatML with thinking-skip template (`<|im_start|>assistant\n<think>\n</think>\n\n`)
- Validated **5/5 coherent** on standard prompts (Paris, 2+2, Berlin, GPU, haiku)
- **20.5 tok/s** decode on AMD RX 9070 XT (gfx1201) with Q3_K_S quantization

8 new GLSL shaders required: `gated_delta_net.comp` (2 variants — w/ + w/o subgroup
clustering), `softplus.comp`, `repeat_interleave.comp`, `ssm_conv_setup.comp`,
`ssm_conv.comp`, `l2_norm.comp`, `sigmoid.comp`. Plus 10 new `LayerStep`
variants extending the executor enum for recurrent attention.

### Performance — Sprint G series

- **G-6: Barrier-elision (+4.6 % Qwen3.6 decode)** — removed 10 redundant
  trailing `VkMemoryBarrier`s in SSM step functions that drained the GPU
  dispatch pipeline between RAW-independent dispatches. GPU-busy fraction
  jumped from 80 % to 92 %.
- **G-7: Async-decode re-enabled (+12.6 % Qwen3.6 decode)** — the G-2j
  "async produces 0 speedup" workaround was caused by those same trailing
  barriers stalling the pipeline. Post-G-6 strip, async-decode finally
  overlaps the CPU command-buffer record (5.84 ms/tok) with the GPU
  compute window (48 ms/tok).

### Diagnostic infrastructure

- `VF_GPU_TIMER=1` → per-dispatch GPU TIMESTAMP breakdown printed at decode
  end (aggregated across all `forward_token` calls). Reveals per-shader
  ms/tok + GPU-busy fraction + total CPU/idle gap.
- `VF_CPU_TIMER=1` → per-stage CPU breakdown (reset, begin, record, end,
  submit, gpu_wait, readback). Reveals where the per-token CPU work sits.
- `VF_GPU_TIMER_DIAG_FFN=1` → opt-in barrier between gemv_gate + gemv_up to
  diagnose RADV adjacent-dispatch timestamp inflation (Sprint G-4 finding).
- Both timers zero-overhead when env unset.

### Stability

- **0 Vulkan synchronization hazards** verified via
  `VK_LAYER_KHRONOS_validation` + `validate_sync = true`.
- 217 / 217 lib tests passing.
- 154 SPIR-V compute shaders.

## 📊 Verified Performance (release-bench, warm cache)

| Model | Decode | Coherence |
|---|---:|---|
| **Qwen3.6-27B-MTP-Q3_K_S** (NEW) | **20.5 tok/s** | 5/5 ✅ |
| Qwen3-8B-Q4_K_M | 109.4 tok/s | "2 + 2 equals 4." ✅ |
| Gemma-4-26B-A4B-it-Q3_K_M | 24.7 tok/s | full paragraphs ✅ |
| Llama-3.1-8B-Instruct-Q4_K_M | 112.7 tok/s | "2 + 2 = 4" ✅ |

*Hardware: AMD RX 9070 XT (RDNA4, gfx1201), Ryzen 9, Linux 7.0, Mesa 26.1,
Rust 1.95. llama.cpp Vulkan baseline on Qwen3.6: 41.93 tok/s
(VF / llama.cpp gap: 2.04× — orchestration-side, see `results/sprint_g7_cpu_gap.md`).*

## 🔧 Environment Variables (v0.4.6 surface)

| Variable | Effect |
|---|---|
| `VF_LMHEAD_ALLOC_FIRST=1` | Hoist `output.weight` to tensor-upload position 0 (improves VRAM placement on ≥12 GB models) |
| `VULKANFORGE_KV_FP8=1` | FP8 KV-cache (-50 % KV VRAM, recommended for 26B+ models) |
| `VF_GPU_TIMER=1` | Per-dispatch GPU timing breakdown at decode end |
| `VF_CPU_TIMER=1` | Per-stage CPU timing breakdown at decode end |
| `VF_GPU_TIMER_DIAG_FFN=1` | Diagnostic barrier between FFN gate+up (G-4 timestamp-inflation diag) |
| `VF_GEMV_NO_SHMEM=1` | Use `_subgroup_no_shmem` K-quant GEMV variants (G-5 honest-negative) |
| `VF_GEMV_NUM_ROWS_K=N` | Override NUM_ROWS spec const for K-quant GEMVs (G-5 honest-negative, breaks Qwen3.6) |
| `VULKANFORGE_DISABLE_ASYNC_DECODE=1` | Force synchronous decode (debug; opt-out from G-7 default) |

## 📦 Build

```bash
git clone https://github.com/maeddesg/vulkanforge.git
cd vulkanforge
git checkout v0.4.6
cargo build --release
```

Requires: Rust 1.85+, Vulkan 1.3, AMD RDNA4 GPU (gfx1201), Mesa 26.1+.

## 🔄 Changes since v0.4.5

26 commits across Sprints G-2a through G-7. Major bugs found + fixed during
Qwen3.6 bring-up:

1. **L2-norm semantics** — Q-K post-conv L2-norm uses ggml's bias-free
   L2-norm (not RMSNorm-with-weight).
2. **Q-Gate interleaved-per-head layout** — `attn_q.weight` outputs
   `[Q_h(256), Gate_h(256), Q_h+1, Gate_h+1, …]` with `nb1 = 2 * head_dim`
   stride, not concat-halves.
3. **40 GDN WAW synchronization hazards** — `cmd_copy_buffer` writes
   into `ssm_state` needed a `SHADER_WRITE|SHADER_READ|TRANSFER_WRITE` →
   `TRANSFER_READ|TRANSFER_WRITE` pre-copy barrier. Resolved by Sprint G-2i.
4. **BatchExec Q-Gate-Deinterleave** — per-token-prefill workaround active
   for Qwen3.6 (BatchExec prefill still broken; G-5/G-6 didn't lift this).
5. **ChatML thinking-skip template** — `tokenizer.special_id("<think>")`
   integer lookup (BPE encode-string fragments `<think>` into 3 tokens).

## 📋 Known Limitations

- **Gemma-4-E2B / E4B incoherent** — pre-existing regression that predates
  Sprint G (verified by checkout). Tracked separately, not a v0.4.6 blocker.
- **Qwen3.6 BatchExec prefill** — emits gibberish; per-token-prefill is the
  production workaround. Each prompt-token decoded individually. G-5 §3 +
  G-3 §3 attempted fixes but the underlying b_step_* divergence remains.
- **Qwen3.6 vs llama.cpp 2.04× gap** — VF GPU-compute is ~2.3× slower than
  llama.cpp's on identical shader sources. Suspect: ggml-vulkan uses
  pre-recorded primary command buffers, secondary queue families, or a
  different descriptor-indexing pattern. G-8+ investigation.
- **MTP speculative decoding** — Qwen3.6 GGUF ships the draft head; runtime
  use planned for v0.5 (+29-63 % decode if/when wired).

## 🗺️ Roadmap (v0.5 directions)

- ggml-vulkan command-buffer recording deep-dive (Sprint G-8) — close the
  remaining 2× gap to llama.cpp on Qwen3.6.
- Qwen3.6 BatchExec prefill fix — currently the 15× prefill speedup is left
  on the table.
- MTP speculative decoding (Qwen3.6 ships the draft head).
- E2B/E4B Gemma-4 coherence repair.
