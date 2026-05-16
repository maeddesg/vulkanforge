# Phase 2' — Expert-Grouped Dispatch for Gemma-4 26B-A4B

**Status:** Plan (Sprint 60A).
**Predecessor:** Phase 1' (Sprints 56A → 56C-3, v0.4.4) shipped GPU
router + GPU-direct indexed-GEMV expert FFN, taking prefill from 40
→ 65 t/s and decode from 20 → 27 t/s.

**Premise to verify (Phase 2'-0):** at v0.4.4, expert-FFN dispatches
dominate 26B prefill — confirm with ShaderProfiler before writing
new shaders.

---

## 1. Current Dispatch Pattern (v0.4.4, after 56C-3)

### Decode (1 token)
`executor/moe.rs:432` — `step_moe_expert_ffn_gpu_direct`:

```
for slot in 0..top_k {            // 8 iterations
    run_gemv_indexed_at_offset(gate_up_w, …, indices_buf, slot, …)
    GeluPytorchTanhGlu dispatch
    run_gemv_indexed_at_offset(down_w, …, indices_buf, slot, …)
    run_fma_add_indexed(o_buf → ffn_hidden, weights_buf, slot, …)
}
```

Dispatches per DEC layer: `top_k × 4 = 32`.
For 30 MoE layers, one token: **960 dispatches.**

### Prefill (N tokens)
`executor/moe.rs:975` — `b_step_moe_expert_ffn_gpu_direct`:

```
for t in 0..seq_len {             // 25
    for k in 0..top_k {           // 8
        run_gemv_indexed_at_offset(gate_up_w, …, flat_slot=t*top_k+k)
        GeluPytorchTanhGlu
        run_gemv_indexed_at_offset(down_w, …, flat_slot)
        run_fma_add_indexed(…, flat_slot)
    }
}
```

Dispatches per BAT layer: `seq_len × top_k × 4 = 25 × 8 × 4 = 800`.
For 30 MoE layers, N=25 prefill: **24 000 dispatches** for expert FFN alone.

**Note:** routing indices+weights already live on GPU at
`router.indices_scratch.handle` and `router.weights_scratch.handle`
after 56B. No CPU readback in the GPU-direct path. The flat-slot
index `t * top_k + k` selects which expert this dispatch pulls.

---

## 2. The Pre-Check Win — llama.cpp Already Has the Kernel

`vk_shaders/mul_mm.comp` and `vk_shaders/mul_mmq.comp` (both ports
of llama.cpp's Vulkan GEMM kernels) **already contain a complete
MUL_MAT_ID branch** that implements expert-grouped batched GEMM:

```glsl
// mul_mm.comp:69-72 — MUL_MAT_ID bindings
#ifdef MUL_MAT_ID
layout (binding = 3) readonly buffer IDS    { int data_ids[]; };
layout (binding = 4) readonly buffer Counts { int data_expert_count[]; };
#endif

// mul_mm.comp:74-94 — push-constant block
layout (push_constant) uniform parameter {
    uint M; uint N; uint K;
    uint stride_a; uint stride_b; uint stride_d;
    uint batch_stride_a; uint batch_stride_b; uint batch_stride_d;
#ifdef MUL_MAT_ID
    uint nei0;   // = top_k       (cols of IDS, must be power-of-2 fast path)
    uint nei1;   // = seq_len     (rows of IDS)
    uint nbi1;   // = top_k       (stride of IDS row)
    uint ne11;   // = seq_len
#else
    …
#endif
};
```

`mul_mm_id_funcs.glsl` and the `#ifdef MUL_MAT_ID` blocks inside the
two main shaders implement: per-expert workgroup launch
(`gl_WorkGroupID.z`), IDS-buffer scan to gather only the rows
routed to that expert, partial GEMM over the matching rows, then
output to `D` at the gathered token positions.

This is **exactly the expert-grouped pattern** the original
`docs/p1_batch_prefill_plan.md` proposed as "Phase 1 — Expert-
Grouped Dispatch" (~80 LOC new shader). Pre-check confirms: that
shader already exists. **No new GLSL is required.**

What's missing in the repo is the SPV build with `-DMUL_MAT_ID`:

| Existing SPV | Needed Phase 2' SPV |
|---|---|
| `mul_mmq_q3_k_f32.spv`  | `mul_mmq_q3_k_f32_id.spv`  |
| `mul_mmq_q4_k_f32.spv`  | `mul_mmq_q4_k_f32_id.spv`  |
| `mul_mmq_q4_0_f32.spv`  | `mul_mmq_q4_0_f32_id.spv`  |
| `mul_mmq_q5_0_f32.spv`  | `mul_mmq_q5_0_f32_id.spv`  |
| `mul_mm_q4_k_f32.spv`   | `mul_mm_q4_k_f32_id.spv`   (optional, COOPMAT) |

(Q3_K, Q4_K, Q4_0, Q5_0 are the four quants 26B's expert tensors
actually use — Q3_K_M's gate_up is Q3_K, down is mixed Q4_0+Q5_0;
SafeTensors variant is Q4_K.)

This shifts the sprint shape: instead of "write 80 LOC of GEMM
GLSL + scatter shader", Phase 2' becomes **"build N new MUL_MAT_ID
SPV variants + wire one mul_mmq_id dispatch helper + scatter-FMA"**.

---

## 3. Token-Grouping Strategy

The MUL_MAT_ID kernel itself does no grouping — it scans the IDS
buffer per-expert and matches rows. Three things must happen
around the kernel:

### 3.1 IDS buffer (already exists)

`router.indices_scratch.handle` already holds an `[seq_len × top_k]`
u32 array indexed as `indices[t * top_k + k] = expert_id`. The
shader expects `data_ids[ii1 * nbi1 + ii0]` with `ii1 ∈ [0, nei1)`
(= seq_len) and `ii0 ∈ [0, nei0)` (= top_k). Layout matches; no
re-pack needed. Push `nei0=top_k`, `nei1=seq_len`, `nbi1=top_k`.

### 3.2 Expert-count buffer (new)

`data_expert_count[expert_idx]` tells the shader how many (token,
slot) entries route to `expert_idx`. Three options:

**Option A — CPU readback + count + upload (pragmatic, Phase 2'-1):**
```
mid_frame_submit_and_wait();
indices = router.indices_scratch.read();          // 800 bytes for N=25
counts = vec![0u32; n_experts];
for &expert in indices { counts[expert] += 1; }
upload(counts, expert_count_buf);
```
- Cost: 1 mid_frame_submit per MoE layer = 30 per prefill (~15-30 ms stall).
- The router-readback path existed in 56B (50 t/s) — performance
  envelope is known.
- Implementation: ~30 LOC in `b_step_moe_expert_ffn_gpu_direct`.

**Option B — GPU histogram shader (Phase 2'-2 if Option A bottlenecks):**
- New shader: `moe_expert_count.comp` (~20 LOC). One thread per IDS
  entry, `atomicAdd(counts[indices[i]], 1)`.
- Dispatch right after `run_moe_router_gpu`, before the MMQ_ID call.
- Cost: 1 trivial GPU dispatch, no host sync.
- Total per layer: count → gate_up_mmq_id → glu → down_mmq_id →
  scatter_fma = **5 dispatches** (vs 800 today).

**Recommendation:** ship Option A first (validates the kernel path
on real data), then upgrade to Option B if mid_frame_submit
becomes the new bottleneck.

### 3.3 Dispatch launch

```
push_constants = MmqIdPushConstants {
    M: 2 * mi,                              // output rows (gate_up)
    N: seq_len,                             // input rows
    K: hidden,
    stride_a: K, stride_b: K, stride_d: M,
    batch_stride_a: elems_per_expert,       // weights stride between experts
    batch_stride_b: 0,                      // inputs not strided by expert
    batch_stride_d: 0,                      // outputs gathered
    nei0: top_k, nei1: seq_len,
    nbi1: top_k, ne11: seq_len,
};
dispatch(workgroups_x = ceil(M/BM),
         workgroups_y = ceil(N/BN),
         workgroups_z = n_experts);         // ONE workgroup-z per expert
```

One dispatch covers **all experts × all matching token rows**. The
shader internally early-exits workgroups for experts with zero
matching tokens (via `expert_count[expert_idx] == 0`).

---

## 4. Scatter-FMA

After `down_mmq_id` writes `[seq_len × hidden]` (where each row is
weighted by ONE of the top_k slots for that token), the result must
be multiplied by the per-(token, slot) routing weight and summed
across slots into `batch_ffn_hidden`.

**Current GPU-direct path:** `run_fma_add_indexed` per (t, slot).
**Phase 2' grouped path:** the MMQ_ID output is naturally per-slot
already — but with grouping, the "rows" of the output are
indexed not by token but by the position in the gathered group.

Two options:

**Option X — Output back to per-(t, slot) slots:**
- Allocate `[seq_len × top_k × hidden]` intermediate (≈25×8×2816×4
  = 2.2 MB, fits in `o_buf` if resized, or new scratch).
- MMQ_ID writes each row to its `(t, slot)` slot using IDS-to-slot
  mapping.
- Reuse existing `fma_add_indexed` but as ONE dispatch covering
  all `[seq_len × top_k × hidden]` invocations (single big dispatch
  with workgroup over `seq_len × top_k`).
- ~1 dispatch.

**Option Y — Pre-weighted GEMM output:**
- Add a `weights` SSBO binding to the MMQ_ID kernel (modify shader),
  or fold the weights into a small post-pass.
- New 30-LOC `scatter_fma_grouped.comp`: reads `o_buf`, scatters
  to `batch_ffn_hidden[token]` with `weights[t*top_k+slot]`.
- ~1 dispatch.

**Recommendation:** Option X for the first land — reuses
`fma_add_indexed` and keeps the MMQ_ID kernel unmodified. Promote
to Option Y if benchmarks show the intermediate buffer is a
bandwidth issue.

---

## 5. Expected Performance

### Dispatch-count reduction
| Path | Per layer | Per prefill (30 layers) |
|---|---|---|
| v0.4.4 GPU-direct GEMV | `N × top_k × 4 = 800` | 24 000 |
| Phase 2'-1 (Option A+X) | `~5` (count + gate_up_mmq_id + glu + down_mmq_id + fma) | 150 |
| Phase 2'-2 (Option B+X) | `5` | 150 |

≈160× dispatch reduction.

### Arithmetic intensity
The MMQ_ID kernel processes `n_active_per_expert ≈ N × top_k /
n_active_experts` rows per expert dispatch. For N=25, top_k=8,
~40 active experts → **5 rows/expert on average**. Each expert's
weight matrix is read once and amortized across 5 token rows —
roughly **5× better cache reuse** vs the per-(t, k) GEMV path.

### Headline target
- Phase 2'-1: 65 → **130–180 t/s prefill** (2-3× from cache reuse
  on top of dispatch reduction).
- Phase 2'-2: 130–180 → **180–220 t/s prefill** (eliminating
  remaining mid_frame_submit stalls).
- Phase 2'-1 also helps **decode** marginally — DEC top_k=8 GEMVs
  collapse to one MMQ_ID dispatch, ≈+1-2 t/s. Not the headline
  improvement, but free.

Decode-only Phase 2' is questionable — DEC's 8 GEMVs at N=1 are
cheap and the MMQ_ID overhead may eat the gain. Recommend
**land Phase 2' BAT first**, evaluate DEC separately.

---

## 6. Sprint Breakdown

### Sprint 61A — Phase 2'-0 Profile (½ day)
1. Attach `ShaderProfiler` to a 26B prefill_batch run via
   `Forward::new(.., Some(p))`.
2. Run prompt "The capital of France is" at N=25.
3. Dump `stats.per_shader` for labels prefixed `moe_*_id_b`.
4. Confirm > 60 % of GPU time is in the 800-dispatch chain.
5. Produce `results/sprint_61a_profile.md`.

If < 40 % → bottleneck is elsewhere (e.g. attention KV stride);
**stop here and re-plan**.

### Sprint 61B — Build + Register MMQ_ID SPVs (1 day)
1. Add 4 (or 5) `ShaderJob` entries to `build.rs` mirroring the
   existing `mul_mmq_q3_k_f32` etc., with `-DMUL_MAT_ID`.
2. New `ShaderId::MulMmqQ3KMatId`, `MulMmqQ4KMatId`,
   `MulMmqQ4_0MatId`, `MulMmqQ5_0MatId` in `shaders.rs`.
3. Pipeline registration in `pipeline_registry.rs`. Push-constants
   are 52 B (vs MatVecId's 48 B) — new struct `MmqIdPushConstants`
   in `pipeline.rs`.
4. `MmqIdPushConstants` size assert.
5. **No behavior change.** Smoke test all 4 models still pass.

### Sprint 61C — MMQ_ID Helper + Option A Counts (1 day)
1. `runs.rs`: `run_mmq_id_at_offset` (~70 LOC, mirrors
   `run_gemv_indexed_at_offset`'s parameter list).
2. `executor/moe.rs`: new helper
   `b_step_moe_expert_ffn_grouped` — gated by `VF_MOE_GROUPED=1`.
3. Inside: read indices from staging (1 mid_frame_submit),
   compute counts on CPU, upload to a new `moe_expert_count_buf`
   (allocated in `setup.rs`).
4. Dispatch: gate_up_mmq_id → glu (one big dispatch on
   `[seq_len × top_k × mi]`) → down_mmq_id → fma_add_indexed
   (one big dispatch on `[seq_len × top_k × hidden]`).
5. Regression: all 4 models match Sprint 60A baseline within ±2 %.
6. New 26B prefill ≥ 90 t/s.

### Sprint 61D — Default-flip + Async (½ day)
1. If 61C lands a clean speedup, flip default ON
   (`VF_MOE_GROUPED` default-true).
2. Verify async-decode still safe (currently gated on
   `gpu_direct_moe_enabled()` — needs OR-arm for the grouped path).
3. Update `docs/p1_batch_prefill_plan.md` Phase 2' status row.

### Sprint 61E — Option B GPU Histogram (1 day, optional)
1. `vk_shaders/moe_expert_count.comp` (~20 LOC).
2. `ShaderId::MoeExpertCount`, registry, runs helper.
3. Replace the readback+count+upload in 61C with one shader call.
4. Removes the last `mid_frame_submit_and_wait` from MoE prefill
   → re-enables full async-decode without conditional gating.

### Sprint 61F — DEC port (½ day, optional)
1. Apply the grouped path to `step_moe_expert_ffn_gpu_direct`
   (decode, N=1).
2. Measure: if +1 t/s, keep; else revert and document.

**Phase 2' total:** 4-5 days for 61A → 61D, +1.5 days for 61E+61F.

---

## 7. Risks

### 7.1 nei0 power-of-2 requirement
`mul_mm.comp:161` has a fast path `if (bitCount(p.nei0) == 1)`
using subgroup ballot. Gemma-4 26B has `top_k = 8` ✓. Document
this — any future MoE with non-pow-2 top_k falls onto the slower
scalar path (still correct, ~2× slower per kernel call).

### 7.2 Push-constant size assert
MMQ has 13 u32 (52 B), MMQ_ID has 13 u32 (52 B) — same physical
size, different field semantics. Need a `MmqIdPushConstants`
struct in `pipeline.rs` with `assert!(size_of == 52)` to lock
the layout.

### 7.3 Indexed-GEMV path stays as fallback
Don't delete `run_gemv_indexed_at_offset` and the 8 indexed-GEMV
shader variants. Keep them as the `VF_MOE_GROUPED=0` fallback
through at least v0.5.0. Only consider deletion once Phase 2'
ships in 2+ releases without regression.

### 7.4 Per-quant validation
Q3_K, Q4_K, Q4_0, Q5_0 each need their own MMQ_ID SPV variant.
The shader's MUL_MAT_ID path was written against the same dequant
machinery as the non-ID path — should "just work" — but
quantization-specific bugs (e.g. block alignment K%256) have bitten
us before (see `feedback_q4k_per_row_alignment.md`). Smoke each
quant.

### 7.5 ne11 vs N alignment
`mul_mm.comp:230`'s `pos_b_ib` formula uses `% p.ne11` to wrap
the row index. If `ne11 != seq_len` we'd silently read wrong
input rows. Set `ne11 = seq_len` and assert at dispatch time.

### 7.6 Coopmat option
`mul_mm_q4_k_f32_coopmat.spv` already exists (Sprint 11E port).
Adding a `_id_coopmat` variant is +1 spec-constant flag. Could
land in 61B for the Q4_K SafeTensors path. Q3_K coopmat (GGUF
26B) does NOT exist — would be ~2 days extra work, deferred.

### 7.7 Numerical drift vs CPU baseline
Phase 2' changes the accumulation order: previously per-token
across slots, now per-slot across tokens. FP32 sum is associative
for these magnitudes, but a `forward_layer_debug` cosine check vs
the v0.4.4 path is the safe gate before flipping defaults.

---

## 8. Files Likely Touched

| File | Lines | Reason |
|---|---|---|
| `build.rs` | +60 | 4-5 new MUL_MAT_ID SPV ShaderJobs |
| `vk_shaders/moe_expert_count.comp` | +25 | Option B histogram (Sprint 61E) |
| `vk_shaders/` (no other changes) | 0 | mul_mm/mul_mmq already have MUL_MAT_ID |
| `src/backend/vulkan/shaders.rs` | +30 | 4-5 ShaderId variants + ALL_SHADERS |
| `src/backend/vulkan/pipeline.rs` | +20 | `MmqIdPushConstants` struct |
| `src/backend/vulkan/pipeline_registry.rs` | +40 | match arms for new IDs |
| `src/backend/vulkan/forward/runs.rs` | +90 | `run_mmq_id_at_offset` |
| `src/backend/vulkan/forward/setup.rs` | +15 | `moe_expert_count_buf` allocation |
| `src/backend/vulkan/forward/state.rs` | +3 | field on Forward |
| `src/backend/vulkan/forward/executor/moe.rs` | +150 | `b_step_moe_expert_ffn_grouped` |
| `src/backend/vulkan/forward/mod.rs` | +5 | `moe_grouped_enabled()` env flag |

No public API. No model-format changes. No tests deleted; new
unit test for `moe_expert_count` builder.

---

## 9. Decision

Recommended sequencing for the next sprint chain (v0.5.0):

1. **Sprint 61A** (½ day) — Profile, confirm hypothesis.
2. **Sprint 61B** (1 day) — MMQ_ID SPV variants land, zero behavior change.
3. **Sprint 61C** (1 day) — Grouped path env-gated, validation.
4. **Sprint 61D** (½ day) — Default-flip if 61C is clean.
5. **Sprint 61E** (optional, 1 day) — GPU histogram.
6. **Sprint 61F** (optional, ½ day) — DEC port.

Tag at the end of 61D as v0.5.0-rc1 if prefill ≥ 130 t/s.

Phase 3 (coopmat / FP16 / FP8 on expert weights) becomes a
**parameter sweep on the MMQ_ID kernel** rather than a fresh
shader project — Phase 2' lays the right plumbing for it.

---

## 10. References

- `docs/p1_batch_prefill_plan.md` — original analysis (kept for
  context; Section "Original v0.4.3 Analysis" still describes the
  v0.4.3 baseline).
- `docs/gemma4_26b_moe_solution.md` — v0.4.4 perf table.
- `vk_shaders/mul_mm.comp:69-94` — MUL_MAT_ID push-constants + bindings.
- `vk_shaders/mul_mm_id_funcs.glsl` — IDS-scan + per-expert dispatch.
- `src/backend/vulkan/forward/executor/moe.rs:432,975` — current
  GPU-direct DEC + BAT helpers (the code Phase 2' replaces).
- `src/backend/vulkan/forward/runs.rs:1531,1591` — existing
  indexed-GEMV + FMA helpers (the API shape Phase 2's helpers
  mirror).
