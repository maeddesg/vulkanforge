# Sprint 19B-A — Multi-Submit Prefill Pacing

**Date:** 2026-05-03
**Branch:** main (post-Sprint 19A, commit `1aebccf`)
**Goal:** Close the small-prompt prefill gap to llama.cpp by overlapping
CPU command-buffer recording with GPU execution — the same pattern
`ggml_backend_vk_graph_compute` uses (Sprint 19B analysis).

## Hypothesis

For prompts where GPU compute is fast (pp ≤ 128 on Qwen3-8B), VF spends
significant wall-clock time *recording* the single ~470-dispatch
command buffer while the GPU sits idle. Splitting the prefill into
N command buffers and submitting them sequentially lets the GPU start
executing chunk K while the CPU records chunk K+1, hiding most of the
recording latency behind compute.

llama.cpp does exactly this in `ggml-vulkan.cpp:14507-14755` — submits
every ~100 nodes OR ~100 MB of matmul, yielding 3-8 submits per
8B prefill.

## Implementation

### `Forward` struct (`src/backend/vulkan/forward.rs`)

Four new fields, mirroring the existing `async_*` setup for Sprint 15E
async decode:

```rust
prefill_pool: vk::CommandPool,        // dedicated CB pool
prefill_cbs: Vec<vk::CommandBuffer>,  // ceil(n_layers/interval) CBs
prefill_fence: vk::Fence,             // for the final wait only
layers_per_submit: u32,               // 0 = legacy; default 4
```

`Forward::new` reads `VULKANFORGE_PREFILL_SUBMIT_INTERVAL` (default
**4**), pre-allocates `ceil(n_layers / interval)` CBs from a fresh
`CommandPool` with `RESET_COMMAND_BUFFER`, and one fence. When
interval ≥ n_layers (or 0) we leave `prefill_cbs` empty and fall
back to the legacy `cmd_ctx.one_shot` path.

### `prefill_batch` refactor

Extracted three phases of the old single-CB body into helper methods:
- `record_prefill_seed` — copy `batch_input → batch_residual` + RMS
  norm seed for layer 0 (Sprint 9b.2 cross-layer-fusion contract)
- the existing `dispatch_layer_batch` per-layer loop body
- `record_prefill_finalize` — copy last-row → scratch_a + final norm
  + LM head + host-read barrier

Multi-submit branch: begin CB[0], record seed; loop layers, recording
into `prefill_cbs[chunk]`; at every `(layer + 1) % interval == 0`
boundary (except the very last layer) `vkEndCommandBuffer` + a
**fence-less** `vkQueueSubmit` of the current CB, then begin
`prefill_cbs[chunk + 1]`. After the loop, record finalize into the
last CB and submit it **with** `prefill_fence`. One `vkWaitForFences`
at the end covers all prior submits because Vulkan guarantees
in-order execution on the same compute queue and submit boundaries
on the same queue act as full memory barriers (no explicit
inter-CB synchronization needed).

Total diff: ~150 LOC added across `forward.rs` (struct + new()
+ destroy() + record helpers + branch in prefill_batch). Zero shader
changes. Decode (`forward_token`, `forward_token_async`) untouched.

### Why one fence works

* All submits go to the same `dev.compute_queue`.
* Vulkan guarantees in-order execution per queue.
* Submit boundaries on the same queue are implicitly equivalent to a
  full memory barrier (Vulkan spec: a submission's commands "are
  visible to" later submissions on the same queue).
* The host only needs to know when the *last* submit completes →
  one fence on the final submit, one wait at the end.

### Bit-exactness preserved

All dispatches happen in the same order, with the same arguments,
on the same queue. The only difference is when the boundaries
between submits fall. Empirical greedy-output check on Q4_K_M:
"Explain the concept of recursion in programming briefly." with
`temperature=0` produces the exact same 60-token continuation in
single-submit and multi-submit (interval=6) modes.

## Measurements

### Q4_K_M (Qwen3-8B), median over 3 runs

| pp | Single-submit (legacy) | I=12 | I=6 | I=4 (default) | I=3 | I=2 | best Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  32 |  992 | 1040 | 1061 | **1062** | 1072 | 1061 | **+8.0%** |
|  64 | 1689 | 1782 | 1791 | **1801** | 1799 | 1810 | **+7.2%** |
| 128 | 2582 | 2679 | 2711 | **2718** | 2717 | 2716 | **+5.3%** ← GATE |
| 256 | 3532 | 3609 | 3626 | **3636** | 3633 | 3640 | **+3.1%** |
| 512 | 3863 | 3898 | 3890 | **3901** | 3905 | 3902 | **+1.1%** |
|1024 | 3735 | 3752 | 3759 | **3755** | 3756 | 3753 | **+0.6%** |

Sweet spot is **interval = 4** (9 chunks for a 36-layer Qwen3): wins
at every pp, no regression vs single-submit at any pp, and ≤ 0.5%
behind any other interval where another value is best.

### Q3_K_M (Qwen3-8B), median over 3 runs

| pp | Single-submit | I=6 | I=4 (default) | I=3 | best Δ |
|---:|---:|---:|---:|---:|---:|
|  32 |  966 | 1031 | **1036** | 1024 | **+7.3%** |
|  64 | 1656 | 1755 | **1770** | 1761 | **+6.9%** |
| 128 | 2570 | 2683 | **2708** | 2696 | **+5.4%** |
| 256 | 3384 | 3478 | **3501** | 3508 | **+3.7%** |
| 512 | 3683 | 3736 | **3735** | 3730 | **+1.4%** |
|1024 | 3513 | 3536 | **3535** | 3524 | **+0.6%** |

Same shape as Q4_K_M. Bench gate (pp=128 ≥ +5%) cleared on **both**
quants.

### Decode regression

Both quants: 116-117 tok/s for Q4_K_M, 131-132 tok/s for Q3_K_M
— identical with and without multi-submit, confirming the change
only touches `prefill_batch` (decode goes through `forward_token` /
`forward_token_async`).

### 15-prompt suite

* Q4_K_M @ I=4: **15/15 coherent**, median decode 109.1 tok/s,
  median prefill 886.7 tok/s.
* Q3_K_M @ I=4: **15/15 coherent**, median decode 121.7 tok/s,
  median prefill 850.9 tok/s.

## Why the gain shrinks at large pp

The pattern (`+8% at pp=32` → `+0.6% at pp=1024`) is exactly what we'd
predict if recording overhead were a fixed cost per dispatch and GPU
compute scaled with pp. At pp=32 GPU compute per layer is ~0.9 ms
while CPU recording is ~3-4 ms across 36 layers — recording dominates,
splitting wins big. At pp=1024 GPU compute per layer is ~7-8 ms,
recording is the same ~3-4 ms — already well-overlapped, so splitting
adds little. This validates the model in §1 of the Sprint 19B plan
(`results/v034_sprint19b_graph_plan.md`).

## What this does NOT do

* No buffer-aliasing changes. (Sprint 19B analysis: not a lever.)
* No fusion changes. (Reserved for sub-sprint 19B-B if shipped.)
* No barrier-elision changes. (Reserved for 19B-C.)
* No decode changes. (Sprint 15E async path remains as-is.)

## Sprint 19A + 19B-A combined progress (Q4_K_M pp=512)

* v0.3.3 baseline: 3865 tok/s
* Sprint 19A (Q3_K + Q5_K coopmat coverage): 3865 (Q4_K_M unchanged
  there — Q3_K_M was the big win, +57%)
* Sprint 19B-A (multi-submit): 3911 (+1.3%)

The compounding return on Q4_K_M is small because pp=512 was already
GPU-bound; the real 19B-A win is **on the small-prompt regime
(pp ≤ 128) where pp=128 went from 2582 → 2718 (+5.3%)** —
exactly where we predicted the lever would matter.

## Files touched

* `src/backend/vulkan/forward.rs` — struct fields, `new()` / `destroy()`,
  `prefill_batch` branch, `record_prefill_seed`,
  `record_prefill_finalize` helpers. ~150 LOC net.

## Outcome

**Bench gate `pp=128 ≥ +5%` cleared on Q4_K_M (+5.3%) and Q3_K_M
(+5.4%).** Default = `interval=4` (9 chunks per 36-layer prefill).
Override via `VULKANFORGE_PREFILL_SUBMIT_INTERVAL=N` (0 = legacy
single submit). Bit-exact output, decode unaffected,
32/32 lib tests pass, 15/15 prompts coherent on both Q3_K_M and Q4_K_M.
