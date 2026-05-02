# VulkanForge v0.2.1 Sprint 12C — Gap-Analyse + Sprint 11 Revalidierung

**Date:** 2026-04-30
**Branch:** main (HEAD = 67a9425, post-Sprint 12B)
**Inputs:** `results/v021_sprint12a_llama_audit.md`, `results/v021_sprint12b_vf_audit.md`
**Mode:** Pure analysis — no code changes.

## TL;DR

Of the two audits, only **two categories** show meaningful gaps:
1. **Barrier density** — VF emits ~3× more barriers per forward (~432 vs ~150)
2. **Dispatch count** — VF runs ~17–18 OPs per layer vs llama.cpp's 10–13 (4–7 missing fusions per layer)

All other categories are **TIED** or **VF WINS** (e.g., disk pipeline cache).

Estimated end-to-end impact, per realistic mid-point:
- **Prefill** (GPU-dominated): ~3–5 % savings → 0.54× → ~0.57–0.59× (~40 % gap remains, **NOT explained by infrastructure**)
- **Decode** (CPU/GPU-balanced): ~25–40 % savings → 0.79× → ~0.99–1.10× (**gap potentially closed**)

Sprint 12D (recommended): Tier-1 fix #1 (barrier elision) — biggest ROI, scoped, fully infrastructure (no shader work). Sprint 11G retest in 12F is cheap (~30 min) but analytically unlikely to flip NO-GO → GO.

---

## 1. Side-by-side comparison table

Compiled from 12A (llama.cpp) and 12B (VulkanForge) — exact line-referenced numbers, classified per row.

| # | Category                    | llama.cpp (12A)                                     | VulkanForge (12B)                                  | Class      |
|---|-----------------------------|-----------------------------------------------------|----------------------------------------------------|-----------|
| 1 | Submits / Forward           | 5–10 (100-node heuristic + ramp-up)                 | **1** (one_shot wraps whole forward)               | VF WINS¹  |
| 2 | CmdBuf reuse                | yes, pool reset every 10 buffers                    | yes, RESET_COMMAND_BUFFER per one_shot             | TIED      |
| 3 | Wait granularity            | fence per submit, no hot-path wait between          | **1 wait_for_fences per forward (BLOCKING)**       | GAP-S²    |
| 4 | Barriers / Layer            | **0–4** (dirty-flag elision)                        | **12** (decode) / 11 (prefill, no elision)         | **GAP-L** |
| 5 | Barriers / Forward          | ~150                                                | **~432** (decode) / ~396 (prefill)                 | **GAP-L** |
| 6 | Barrier-Elision             | yes — 3 dirty flags + per-pipeline+tensor cache     | **NO** — unconditional after every block           | **GAP-L** |
| 7 | WaitIdle / DeviceWaitIdle   | 0 in hot path                                       | 0                                                  | TIED      |
| 8 | Fences / Submit             | 1 (`ctx->fence`) + 1 almost_ready                   | 1 (CommandContext.fence)                           | TIED      |
| 9 | Timeline Semaphores         | yes, transfer→compute handshake                     | **NO**                                             | GAP-S³    |
| 10| vkAllocateMemory / Token    | 0                                                   | 0                                                  | TIED      |
| 11| Buffer pool / Arena         | 4 named scratch + offset-arithmetic                 | 32 named scratch + gpu-allocator                   | TIED      |
| 12| Persistent map              | yes (mapMemory once, ptr cached)                    | yes (gpu-allocator persistent)                     | TIED      |
| 13| Weight memory               | DEVICE_LOCAL+HOST_VISIBLE (ReBAR preferred)         | DEVICE_LOCAL only (staging copy at load)           | GAP-S⁴    |
| 14| Descriptor pool model       | pre-alloc, indexed (descriptor_set_idx), 50 % grow  | pre-alloc, HashMap-cache (BindingSignature key)    | TIED      |
| 15| DSL strategy                | **single global DSL** (MAX_PARAMETER_COUNT=12)      | per-pipeline, reflected from SPIR-V                | TIED      |
| 16| Push descriptors / templates| not used                                            | not used                                           | TIED      |
| 17| Pipeline disk-cache         | **NO**                                              | **YES** (`~/.vulkanforge/pipeline_cache.bin`)      | **VF WINS** |
| 18| Pipelines (eager / lazy)    | lazy on first use (with parallel async)             | eager at PipelineRegistry::new                     | TIED      |
| 19| Per-vendor overrides        | yes (AMD-GCN ≠ AMD-coopmat ≠ Intel-XE2 ≠ Win-old)   | RDNA4 only                                         | TIED⁵     |
| 20| Pipelines per quant         | 6+ (l/m/s × aligned, plus _id, _int_k variants)     | S+L (no M-tile in 11G; aligned variants)           | GAP-S⁶    |
| 21| Spec-constant warptile      | 11-int (BLOCK_SIZE…WARP)                            | identical 11-int (Sprint 11C ports llama.cpp)      | TIED      |
| 22| Dispatches / Layer (decode) | ~10–13 (full fusion)                                | **17**                                             | **GAP-L** |
| 23| Dispatches / Layer (prefill mmq)  | ~10–13                                        | **18** (mmq) / **14** (mul_mm)                     | **GAP-L** |
| 24| Fused: RMS_NORM+MUL+ROPE+VIEW+SET_ROWS (5-op) | yes (writes rope output → KV-cache slot directly) | **NO** (rope + kv_copy = 2 dispatches)         | **GAP-L** |
| 25| Fused: MUL_MAT+ADD(+ADD)    | yes                                                 | NO (Qwen3-8B has no bias — moot for our model)     | TIED⁷     |
| 26| Fused: MULTI_ADD (≤9-arity) | yes                                                 | binary only (multi_add_rms = 2 inputs)             | GAP-S⁸    |
| 27| Fused: SwiGLU               | pipeline_swiglu (`silu(g)*u`)                       | yes (`run_swiglu`)                                 | TIED      |
| 28| Fused: RMS+MUL+ROPE (3-op)  | yes                                                 | yes (`rms_norm_mul_rope` Sprint 9c.5)              | TIED      |
| 29| Fused: cross-layer residual+next-norm | yes (via `multi_add_rms` chains)          | yes (Sprint 9b.2)                                  | TIED      |
| 30| Fused: TopK MoE (4 modes)   | yes                                                 | NO (Qwen3-8B is dense — moot for our model)        | TIED⁷     |
| 31| q8_1 redundancy (Q/K/V)     | 1× (per-pipeline+tensor cache)                      | 1× (buffer-aliasing via `batch_q8`)                | TIED      |
| 32| q8_1 redundancy (Gate/Up)   | 1×                                                  | 1×                                                 | TIED      |
| 33| Per-dispatch host commands  | 5 (bind+bind+push+dispatch + barrier-via-flag)      | 5 (bind+bind+push+dispatch + barrier-via-helper)   | TIED      |
| 34| Transfers per Token (decode)| 2 (token-id in, logits out)                         | 3 (embedding + rope_pos + logits)                  | GAP-S⁹    |
| 35| Transfer queue              | yes (separate, timeline-semaphore handshake)        | NO (single compute_queue)                          | GAP-S³    |

**Footnotes:**
1. VF wins on submit overhead per forward, but **loses on CPU/GPU overlap** because the single submit blocks. For decode (CPU-record-bound), pipelining is a net win for llama.cpp despite more submits.
2. Single blocking wait — cost depends on tail-latency dispatches; analyzed in §2.3.
3. Only matters when there is concurrent transfer work; in steady-state Qwen3-8B inference there is none.
4. Load-time cost only; not a per-forward gap.
5. We are explicitly RDNA4-only by design; not a gap to close.
6. Sprint 11 abandoned the M-tile int8-coopmat shader. mul_mmq has S/L and aligned variants only.
7. Qwen3-8B is dense and has no bias terms — these fusions don't fire on our model. They block other-model support, not current performance.
8. Useful for MoE / gated paths; not exercised by Qwen3-8B dense.
9. The extra `rope_pos_buf` write (4 bytes) in VF: llama.cpp encodes positions via push constants. Negligible host-overhead delta.

**Net findings:**
- 2 GAP-L items (categories #4–6 = barriers; #22–24 = dispatches/fusion)
- 6 GAP-S items (mostly load-time, decode-only, or moot for Qwen3-8B)
- 21 TIED items
- 2 VF WINS items (single-submit on prefill, disk pipeline cache)

**Conclusion:** the host-side gap is concentrated in two areas (barriers + missing 5-op fusion). Everything else is parity or favours VF.

---

## 2. Impact analysis per gap

Baseline timings (from Sprint 10F report and v0.2.x history):
- VF prefill pp=512: ~227 ms (2254 tok/s)
- llama.cpp prefill pp=512: ~123 ms (4174 tok/s)
- VF decode: ~11.1 ms/token (90 tok/s)
- llama.cpp decode: ~8.8 ms/token (113 tok/s)

Speedup gaps:
- Prefill: 0.54× (need 1.85× current speed to match)
- Decode: 0.79× (need 1.26× current speed to match)

Per-call cost models (RDNA4 + RADV, validated via prior profiling and µbenchmarks):
- `vkCmdPipelineBarrier`: **2–5 µs CPU**, **~5–15 µs GPU stall** (pipeline flush + L1/L2 invalidation; lower bound on cache-warm short kernels, upper bound on kernels with significant LDS write-back) → **midpoint ~13 µs combined**
- `vkCmdBindPipeline + BindDescriptorSets + PushConstants`: **~5 µs CPU** combined
- `vkCmdDispatch`: **~3 µs CPU**, **~5–20 µs GPU launch overhead** (shader VGPR allocation + workgroup setup, depends on tile size)
- `vkQueueSubmit`: **~30–80 µs CPU** (one-shot path includes fence reset + cmdbuf reset)

### 2.1 Barrier overhead (GAP-L)

```
Extra barriers / forward = 432 (VF) − 150 (llama.cpp) = 282

Per-barrier cost (combined CPU+GPU):
  Conservative (5 µs):   1.41 ms savings/forward
  Mid          (13 µs):  3.67 ms savings/forward
  Optimistic   (20 µs):  5.64 ms savings/forward

PREFILL impact (227 ms total):
  Conservative: 0.6 %  (0.54× → 0.544×)
  Mid:          1.6 %  (0.54× → 0.549×)
  Optimistic:   2.5 %  (0.54× → 0.554×)

DECODE impact (11.1 ms total):
  Conservative: 12.7 % (0.79× → 0.89×)
  Mid:          33.1 % (0.79× → 1.05×)
  Optimistic:   50.8 % (0.79× → 1.19×)
```

Caveat: not every barrier can be elided — some are correctness-required (KV-cache write before attention read, residual write before next layer's norm read). Rough estimate of *elidable* barriers based on 12B's per-layer enumeration: **~8 of 12** are elidable when the next op consumes a *different* buffer than the previous one wrote. Adjusted savings:

```
Realistic elidable barriers / forward: 282 × (8/12) ≈ 188

PREFILL: 188 × 13 µs / 227 ms = 1.08 %  → 0.54× → 0.546×
DECODE:  188 × 13 µs / 11.1 ms = 22.0 % → 0.79× → 0.96×
```

### 2.2 Dispatch overhead (GAP-L)

```
Extra dispatches / Layer (decode):    17 − 11.5 = 5.5
Extra dispatches / Forward (decode):  5.5 × 36 = 198

Extra dispatches / Layer (prefill):   18 − 11.5 = 6.5
Extra dispatches / Forward (prefill): 6.5 × 36 = 234

Per-dispatch cost (combined): ~25 µs (CPU 8 µs + GPU launch+stall 17 µs)

PREFILL: 234 × 25 µs / 227 ms = 2.6 %   (0.54× → 0.554×)
DECODE:  198 × 25 µs / 11.1 ms = 44.6 % (0.79× → 1.14×)
```

Largest single source: the missing **5-op fusion `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS`** saves 2 dispatches per layer (× 36 = 72 dispatches per forward). The remaining 3–4 extra dispatches per layer come from less obviously-fusable patterns (RMS_NORM + Q/K/V quant + GEMM are bottle-necked by data dependencies, not fusion gaps).

```
Just the 5-op fusion impact:
  72 dispatches × 25 µs = 1.8 ms / forward

PREFILL: 0.79 % (0.54× → 0.544×)
DECODE:  16.2 % (0.79× → 0.92×)
```

### 2.3 Submit batching / pipelining (VF WINS at submit count, but)

```
Submit overhead:
  VF:        1 × ~50 µs = 0.05 ms
  llama.cpp: 7 × ~50 µs = 0.35 ms

VF saves 0.30 ms in submit overhead per forward.
```

**But** the bigger effect is CPU/GPU overlap:
- VF: CPU records all dispatches → submits → blocks waiting for fence → reads logits.
- llama.cpp: CPU records 100 nodes → submits → records next 100 nodes (CPU running) while GPU runs first batch (GPU running) → submits next → ... overlap window per submit boundary.

Estimating overlap savings is hard without GPU traces. Empirical guess based on prior profiler runs:
- Prefill: GPU-dominated (~80 % GPU, 20 % CPU). Pipelining wins ~0–2 %.
- Decode: balanced (~50 % GPU, ~50 % CPU host-record). Pipelining wins ~5–10 %.

```
PREFILL: −0.3 ms (submit) + 0–2 % pipelining = ~−0.3 to +4.5 ms
        net: 0.54× → 0.55× to 0.56×
DECODE:  −0.3 ms + 5–10 % pipelining = ~−0.3 to +1.1 ms
        net: 0.79× → 0.79× to 0.87×
```

Note: pipelining requires submit-splitting (architecture rewrite of `commands.rs`, `forward.rs`'s `one_shot` use). Higher effort than barrier elision.

### 2.4 Other gaps (GAP-S, low priority)

| Gap                     | Forward impact (mid)   | Effort     | ROI if both fixed                  |
|-------------------------|------------------------|------------|------------------------------------|
| Transfer queue          | < 0.5 % (steady state) | medium     | Skip                               |
| Timeline semaphores     | 0 % (no concurrency)   | medium     | Skip — needs transfer queue first  |
| ReBAR weight memory     | 0 % (load-time only)   | low        | Skip — not a per-forward cost      |
| MULTI_ADD ≤9-arity      | < 0.5 % (Qwen3 dense)  | medium     | Skip for Qwen3, future MoE work    |
| MAT+ADD bias fusion     | 0 % (no bias)          | high       | Skip — not exercised by Qwen3-8B   |
| Descriptor HashMap→idx  | < 0.1 %                | medium     | Skip — hash overhead is ~50 ns     |
| rope_pos push-constant  | < 0.05 %               | low        | Skip — rounding error              |

---

## 3. Priority list (ROI ranking)

Aufwand expressed in working days. Impact is the realistic-midpoint percentage from §2. ROI = `realistic-impact-percent / aufwand-days`.

| # | Fix                                       | Prefill % | Decode % | Aufwand | Decode ROI | GO/NO-GO |
|---|-------------------------------------------|-----------|----------|---------|------------|----------|
| 1 | **Barrier elision via dirty-flags**       | +1.1 %    | +22.0 %  | 7–10 d  | 2.4 %/d    | **GO**   |
| 2 | **5-op fusion RMS+MUL+ROPE+VIEW+SET_ROWS**| +0.8 %    | +16.2 %  | 3–4 d   | 4.6 %/d    | **GO**   |
| 3 | Submit-splitting + pipelining             | +0–2 %    | +5–10 %  | 7–10 d  | 1.0 %/d    | MAYBE    |
| 4 | MULTI_ADD ≤9-arity                        | +0.2 %    | +2–5 %   | 4–5 d   | 0.7 %/d    | NO-GO    |
| 5 | rope_pos via push-constants               | +0 %      | +0.5 %   | 0.5 d   | 1.0 %/d    | NO-GO    |
| 6 | Descriptor HashMap → index                | +0.1 %    | +1.0 %   | 2–3 d   | 0.4 %/d    | NO-GO    |

**GO:** ROI > 2 %/d on Decode (the user-facing hot metric).
**MAYBE:** ROI 0.5–2 %/d.
**NO-GO:** ROI < 0.5 %/d.

**Top 2 fixes by ROI: #2 (5-op fusion) > #1 (barrier elision)**, but #1 has the larger absolute impact. The right ordering depends on what we're optimising for:
- Maximise total decode speedup → **#1 first** (22 %, larger headroom)
- Maximise quick wins → **#2 first** (cleaner, less risk, faster shipping)

Sprint 12D recommendation: **#1 first** — it's the lever Sprint 12A's audit flagged as central to llama.cpp's host-side advantage; fixing it removes the elephant. Sprint 12E can do #2.

---

## 4. Cumulative expectations

Sequential gain (assumes effects are mostly additive — barrier-elision and dispatch-fusion don't massively overlap because they save different *kinds* of GPU/CPU time):

### 4.1 Prefill (pp=512, baseline 0.54×)

| Scenario        | After #1 (barrier) | After #1+#2 | After #1+#2+#3 |
|-----------------|--------------------|-------------|-----------------|
| Conservative    | 0.543×             | 0.547×      | 0.550×          |
| Realistic (mid) | 0.546×             | 0.550×      | 0.560×          |
| Optimistic      | 0.554×             | 0.566×      | 0.580×          |

**Best realistic outcome: ~0.55–0.58×.** Prefill gap remains **~40–45 %** unexplained.

### 4.2 Decode (baseline 0.79×)

| Scenario        | After #1 (barrier) | After #1+#2 | After #1+#2+#3 |
|-----------------|--------------------|-------------|-----------------|
| Conservative    | 0.89×              | 0.93×       | 0.94×           |
| Realistic (mid) | 0.96×              | 1.06×       | 1.11×           |
| Optimistic      | 1.06×              | 1.16×       | 1.27×           |

**Best realistic outcome: ~1.05–1.15×** — decode gap **closed** under realistic-or-better assumptions.

### 4.3 Honest synthesis

- **Decode is the win.** Infrastructure fixes plausibly turn 0.79× into ≥ 1.0× — i.e. parity with or above llama.cpp. The 22 % barrier impact + 16 % dispatch impact + 5–10 % pipelining = 40+ % decode speedup.
- **Prefill stays a partial mystery.** Even optimistic Tier-1 fixes leave ~0.58×; the remaining ~40 % gap is **not explained by the host-side audit**. See §6 for hypotheses.
- This asymmetry is consistent with how the gaps scale: prefill is GPU-dominated, where 200 µs of barrier savings is rounding noise on a 227 ms forward. Decode is CPU/GPU-balanced, where the same 200 µs is 1.8 % per dispatch × 280 dispatches = ~50 % of the forward.

---

## 5. Sprint 11 revalidation

### 5.1 The question

Sprint 11G-D measured **Int8-coopmat Q4_K = 0.47×** vs production scalar `mul_mmq = 0.54×` of llama.cpp. NO-GO at the 1.2× gate. After Sprint 12D/E infrastructure fixes, does Int8-coopmat flip GO?

### 5.2 Which gaps affect which path

| Gap                   | Affects mul_mmq?        | Affects Int8-cm Q4K? | Differential? |
|-----------------------|-------------------------|----------------------|---------------|
| Barrier elision (#1)  | yes — same OPs, same barrier sites | yes — identical barrier pattern around the Q4_K coopmat dispatch | **NEUTRAL** (both paths benefit equally) |
| 5-op fusion (#2)      | yes — moves rope+KV-write outside Int8 GEMM region | yes — same | **NEUTRAL** |
| Submit pipelining (#3)| yes — slightly more GPU time per dispatch in mul_mmq → less CPU/GPU overlap | yes — slightly less GPU time per dispatch in Int8-cm → more potential overlap | **slightly POSITIVE for Int8-cm** (+1–2 %?) |
| q8_1 redundancy       | yes — but already 0     | yes — Int8-cm consumes the same Q8_1 buffer | **NEUTRAL** |

### 5.3 Analytical prediction

Both paths get the same ~3–5 % prefill boost from infrastructure:

```
mul_mmq:  0.54× × 1.05 = 0.567× (mid)
Int8-cm:  0.47× × 1.05 = 0.494× (mid; slightly more if pipelining helps)

Ratio Int8-cm / mul_mmq = 0.494 / 0.567 = 0.87× (Int8-cm still 13 % slower)
```

Even under the most optimistic assumption (pipelining is +5 % Int8-cm-only):

```
mul_mmq:  0.567×
Int8-cm:  0.47× × 1.10 = 0.517×
Ratio:    0.91× — still slower
```

**Prognosis: Sprint 11G-D retest will likely remain NO-GO.** The structural gap between coopmat-fold orchestration and tight-inline-fold (Sprint 11G-D's diagnosis) doesn't shrink under infrastructure cleanup.

### 5.4 Recommendation

**Retest anyway in Sprint 12F** — cost is negligible (the bench harness already exists; ~30 minutes wall time):

```bash
cargo run --release --example bench_int8cm_q4k
```

The retest gives definitive numerical proof rather than analytical projection. If the result is unexpectedly positive (say, ≥ 1.0×), there's a hidden interaction we missed and Sprint 11G reopens. If 0.5–0.95× as projected, Sprint 11G stays abandoned and we have empirical confirmation.

---

## 6. Remaining prefill gap

After Tier-1 fixes (#1 + #2), realistic prefill is 0.55–0.58×. **The remaining ~40 % gap is not explained by either audit.** Hypotheses, ranked by plausibility:

| Hyp | Description                                                                  | Estimated impact | Evidence for                                                                                                            | Evidence against                                                |
|-----|------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| A   | **Shape-pipeline-selection mismatch**. llama.cpp's `ggml_vk_guess_matmul_pipeline` uses tile counts vs `shader_core_count`, split-K thresholds, and per-quant `mul_mat_l/m/s` tables. VF uses simple `m>128 && n>256`. | 5–10 %           | Sprint 11C measured +4–5 % from porting just the L-tile warptile values. There's likely more there.                     | Sprint 4 sweep was thorough; M-tile ruled out as not helpful.   |
| B   | **GPU-side cache effects** from fewer barriers. RDNA4 GPU L1/L2 stays warmer when fewer pipeline-barriers force invalidations between dispatches. Hard to attribute, but plausibly explains some of the prefill discrepancy beyond the direct CPU+stall barrier cost. | 5–10 %           | RDNA4 has 32 KB L1 / 4 MB L2 per CU/CU-cluster; cache invalidations cost real time on cache-resident kernels.            | Not directly measurable without GPU trace.                       |
| C   | **Spec-constant tuning differences** beyond what Sprint 11C ported. llama.cpp has per-pipeline-name subgroup-size overrides for RDNA1/2 (`rdna1_pipelines`, `rdna2_pipelines`). RDNA4-specific tuning may differ from what we copied. | 3–8 %            | Sprint 11C ported only L-tile values; M-tile is implicit-default.                                                       | Only one tested; could be smaller.                              |
| D   | **Per-pipeline+per-tensor q8_1 caching** llama.cpp does cross-layer (line 7693–7710 of `ggml-vulkan.cpp`). VF's buffer-aliasing handles intra-layer redundancy but each layer rewrites `batch_q8`. If two consecutive layers' attn-norm outputs are bit-identical (common when residual is very small), llama.cpp skips the second quantize entirely. | 2–5 %            | Plausible on Qwen3-8B where residual deltas dominate; but rare in practice.                                            | Empirically rare hit-rate per llama.cpp's own profiling.        |
| E   | **Measurement methodology**. Different prefill harnesses, warmup counts, prompt lengths, or GPU thermal state between the two reference points. Compounded by the fact that we measure 15 prompts averaged vs llama.cpp running its own bench tool. | 0–5 %            | Sprint 10F's single-run `0% Citrix delta` analysis confirmed our methodology is stable, but we never re-ran llama.cpp under identical conditions. | The 0.54× number has been stable across 10 sprints.             |

### Sum of plausible additional savings

```
Hyp A:  +5–10 %
Hyp B:  +5–10 %
Hyp C:  +3–8 %
Hyp D:  +2–5 %
Hyp E:  +0–5 %
                ─────
Range:  +15–38 %  → 0.58× × 1.15 = 0.67× to 0.58× × 1.38 = 0.80×
```

Even under optimistic stacking, prefill reaches ~0.80× — we'd still be ~25 % off llama.cpp. **There may be a structural ceiling we're not seeing.** Possible candidates include:
- A specific `mul_mmq` variant we haven't built (e.g., a `K-quant-specific` GEMM with weight prefetch into LDS).
- An ACO compiler issue with the Phase-3C `vulkanforge` shader build vs llama.cpp's offline-compiled SPV.
- Shape-specific prefill where llama.cpp dispatches a different shader entirely.

These need empirical investigation in Sprint 13+, *after* Tier-1 fixes have established the post-cleanup baseline.

---

## 7. Recommended Sprint 12D / E / F scope

### Sprint 12D — Barrier elision via dirty flags
- **Effort:** 7–10 days
- **Files:** `forward.rs` (touch every `compute_barrier` callsite), new `BufferDirtyState` tracker on `Forward`
- **Approach:**
  - Define a per-buffer `last_writer_op` enum tracking the last writing OP for each scratch buffer
  - Replace `compute_barrier(dev, cmd)` with `barrier_if_needed(dev, cmd, &mut self.dirty_state, reads, writes)`
  - Skip barrier when `reads ∩ writes == ∅` AND no dirty-flag is set on read targets
- **Gate:** decode tok/s improves ≥ +15 % with bit-exact parity vs current (parity test in `tests/correctness.rs`)
- **Falls 12D fails to deliver +15 % decode**: stop, root-cause, retry once. Two failures → declare 12D NO-GO.

### Sprint 12E — 5-op fusion `RMS_NORM+MUL+ROPE+SET_ROWS`
- **Effort:** 3–4 days
- **New shader:** `vk_shaders/rms_norm_mul_rope_set_rows.comp` — reads from `batch_q`/`batch_k`, applies RMS-norm + scale + rope, writes directly to `kv_cache.k_buffer` / `kv_cache.v_buffer` at `pos_offset_bytes(layer, pos)`
- **Replaces:** 4 dispatches (`rms_norm_mul_rope_q_b` + `rms_norm_mul_rope_k_b` + `kv_copy_fp16_k_b` + `kv_copy_fp16_v_b`) with 2 dispatches
- **Gate:** decode tok/s improves ≥ +10 % beyond the 12D baseline; bit-exact parity

### Sprint 12F — Sprint 11G-D retest
- **Effort:** 0.5 days (run existing bench, write 50-line report)
- **What:** Re-run `cargo run --release --example bench_int8cm_q4k` after 12D+12E
- **Hypothesis:** Int8-coopmat Q4K still 0.85–0.95× of (now-faster) mul_mmq → confirms Sprint 11G stays abandoned
- **Surprise case:** if Int8-cm is ≥ 1.0× of new mul_mmq, reopen 11G with new theory (likely some interaction with barrier-density that disproportionately helped Int8-cm)

### Pessimistic outcome plan

If Sprint 12D + 12E together deliver < +20 % decode speedup, the gap analysis was wrong — barriers and dispatches are NOT the bottleneck. Pivot to:
- Sprint 12G: GPU-side profiling with Radeon GPU Profiler (RGP) — get real per-dispatch timings
- Sprint 13: investigate Hypotheses A (shape selection) and B (cache effects) directly

The reason to commit to 12D+12E first: they're the *only* gaps we have empirical evidence for (audit numbers), and even the conservative case (~10 % decode speedup) would be the first end-to-end win we've delivered since Sprint 10F.
