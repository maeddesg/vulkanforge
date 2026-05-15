# P1: Faster Batch Prefill for Gemma-4 26B-A4B (MoE)

**Status:** v0.4.3 ships at ~40 tok/s prefill for 26B (vs 580 tok/s for
Qwen3-8B). This document analyzes the bottleneck and proposes the
v0.5 fix.

**TL;DR — the brief's premise was partly off.** 26B already runs the
batched prefill path (`prefill_batch` + `dispatch_layer_batch` +
`BatchExec`). What's missing is **per-expert dispatch batching** in
the MoE FFN — currently each (token × expert) pair launches its own
4-dispatch GEMV chain, producing ~24,000 dispatches per prefill.
Fix is "expert-grouping", not "batch the router".

---

## Current Prefill Path (verified from source)

### Dispatcher

`src/backend/vulkan/decode.rs:487-514` — `generate_from_tokens`:

```rust
if force_per_token_prefill {
    // 1 forward_token per prefill token (slow fallback for FP8 SafeTensors)
    for &tid in prefill_tokens {
        forward.forward_token(...)?;
        pos += 1;
    }
} else {
    // BATCHED PATH (default; what 26B uses)
    let chunk_size = forward.max_prefill_tokens.max(1) as usize;
    for chunk in prefill_tokens.chunks(chunk_size) {
        forward.prefill_batch(dev, registry, cmd_ctx, model,
                              &chunk_embeds, chunk_len, pos, chunk)?;
    }
}
```

`force_per_token_prefill` defaults to `false` (only set by
`VULKANFORGE_FORCE_PER_TOKEN=1` env var). **26B chat goes through
`prefill_batch`.** The observed ~40 tok/s IS the batched speed.

### Batch MoE Routing — ALREADY batched

`src/backend/vulkan/forward/executor.rs:2414` — `b_step_moe_route`:

```rust
let total_bytes = (seq_len as u64) * (hidden as u64) * 4;
// Copy ALL N tokens × hidden to host-visible staging
cmd_copy_buffer(batch_in, staging, total_bytes);
mid_frame_submit_and_wait(...);   // ONE drain per layer, not per token

let raw = moe_route_staging.read_bytes()?;
let mut all_routing: Vec<Vec<(u32, f32)>> = Vec::with_capacity(seq_len);
for t in 0..seq_len {
    let token_hidden = &raw[t*h*4 .. (t+1)*h*4];
    let routing = cpu_moe_route(token_hidden, layer_data, ...);
    all_routing.push(routing);
}
fwd.moe_routing_batch = Some(all_routing);
```

CPU routing for ALL N tokens runs after a single mid_frame_submit per
layer. So routing overhead is `30 layers × (drain ~1ms + CPU GEMM
N × ~700µs)` ≈ `30ms + N × 21ms` for N tokens.

For N=25: 30 + 525 = **555 ms just for routing**. Already a major
chunk of the 625ms total prefill, but the GPU work below is bigger.

### Batch Expert FFN — NOT batched per-expert

`src/backend/vulkan/forward/executor.rs:2472` — `b_step_moe_expert_ffn`:

```rust
for t in 0..seq_len {
    for &(expert_idx, weight) in &routing_batch[t] {  // top-8 = 8 experts
        run_gemv_q4k_at_offset_inout(gate_up_w[expert_idx], ...);   // dispatch 1
        glu_dispatch(...);                                            // dispatch 2
        run_gemv_q4k_at_offset_inout(down_w[expert_idx], ...);       // dispatch 3
        run_fma_add_at_offset(weight, o_buf, ffn_hidden[t], ...);   // dispatch 4
    }
}
```

**Dispatch count per layer: `N × top_k × 4 = N × 32`.** For N=25
tokens × 30 layers = **24,000 GPU dispatches per prefill**, each with
its own descriptor-set bind + push-constants + ~10–50µs driver
overhead even when the kernel itself is tiny.

That's the bottleneck.

## Why is Qwen3 / Llama so much faster?

Qwen3-8B and Llama-3.1-8B have **no MoE block**. Their `dispatch_layer_batch`
issues one big GEMM per FFN step covering all N tokens × hidden_dim →
1 dispatch per FFN layer step instead of N×32. For N=25, that's
~30 GEMMs/layer (~900 total per prefill) vs 24,000 for 26B-A4B. The
~25× dispatch-count gap explains most of the 580→40 tok/s gap.

## Bottleneck Confirmation Path

Profile the existing batch path to confirm the dispatch-count
hypothesis quantitatively. VulkanForge has a `ShaderProfiler` already
wired up (`feedback_shaderprofiler_already_wired.md`):

```rust
let p = ShaderProfiler::new(&dev.device)?;
let mut fwd = Forward::new(.., Some(p))?;
fwd.forward_token(... or prefill_batch ...);
let stats = fwd.stats();
for s in &stats.per_shader {
    println!("{}: total {:?} count {}", s.label, s.total, s.count);
}
```

Expected (hypothesis): `moe_gate_up_b`, `moe_down_b`, `moe_glu_b`,
`moe_fma_b` total = 60–80% of prefill_time. If true → expert
batching is the right fix.

## Proposed v0.5 Plan: Expert-Grouped Dispatch

### Phase 0 — Profile (1 sprint)

Run prefill_batch with ShaderProfiler attached on 26B Q3_K_M, N=25.
Confirm the per-expert dispatches dominate (>50% of GPU time).
Produces a real cost breakdown to prioritize the next phases against.

### Phase 1 — Expert-Grouped Dispatch (2–3 sprints)

Replace the `(token × expert)` double loop with `(expert × token-group)`:

```rust
// build per-expert token-list from routing_batch (CPU)
let mut expert_groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
for (t, routing) in routing_batch.iter().enumerate() {
    for &(e, w) in routing {
        expert_groups[e as usize].push((t, w));
    }
}

for (expert_idx, group) in expert_groups.iter().enumerate() {
    if group.is_empty() { continue; }
    // ONE gate_up GEMM covering all tokens routed to this expert.
    // group.len() rows × hidden -> group.len() rows × 2*mi
    // Re-uses the existing Q3_K/Q4_K GEMM shaders with a different
    // row-count parameter.
    run_gemm_q4k_at_offset_grouped(
        gate_up_w, expert_offset(expert_idx),
        batch_in, &group,   // gather: input rows indexed by group[i].0
        gate_buf,
        h, 2*mi,
        ...
    );
    glu_dispatch(...);    // operates on group.len() rows
    run_gemm_q4k_at_offset_grouped(down_w[e], ...);
    // weighted accumulate into batch_ffn_hidden[group[i].0]
    run_scatter_fma_add(group, ...);   // scatter back to ffn_hidden[t]
}
```

Expected dispatch count: **30 layers × n_experts_used × 4** where
`n_experts_used ≤ min(n_experts, N × top_k)`. For N=25, top_k=8,
n_experts=128 → up to 25×8=200 expert-token pairs, distributed across
≤128 experts. Practical count: ~50–80 experts used → 50 × 30 × 4 =
6,000 dispatches (vs 24,000 today, ~4× reduction in dispatch count).

The GEMV→GEMM transition also unlocks higher arithmetic intensity per
expert (~3× rows means ~3× better cache reuse) — modeling this is
Phase 0's job.

### Phase 2 — Coopmat / FP16 Expert GEMM (1 sprint, optional)

If Phase 1 lands well, the per-expert GEMM is large enough to benefit
from cooperative matrix paths (existing `mul_mm.comp` coopmat variants
for FP16 / FP8). Q3_K activations would need dequant-to-FP16 first, or
add a Q3_K coopmat path (~2 days of shader work).

### Phase 3 — GPU-side router (LONG TERM, separate sprint chain)

Move `cpu_moe_route` to a compute-shader pipeline:
- rms_norm + scale + inv_sqrt: 1 dispatch (already exists)
- router GEMM (hidden × n_experts): existing `mul_mat_vec` for N=1, or
  small GEMM for batch
- top_k: new shader (radix/bitonic top-k on n_experts=128 is well-
  studied; ~50 lines GLSL)
- renormalize: 1 dispatch

Eliminates **all 30 `mid_frame_submit_and_wait` drains per prefill**
(~30ms saved at N=25). More importantly: **re-enables async decode
for MoE models** (currently disabled by v0.4.2 fix). The async decode
gain is ~+1 tok/s on 26B (4% recovery) — small but free once the
router runs entirely on GPU.

## Risks

- **GEMM with row-gather**: existing Q3_K/Q4_K shaders are GEMV
  (single-row). Phase 1 needs a small extension to support
  `n_rows > 1` with a row-indirection array. Might land as a fresh
  shader rather than parameterizing the existing one.
- **Validation**: routing decisions must remain identical to the
  current path. A bit-id smoke test (compare top_k indices before/
  after change for a fixed prompt) gates merge.
- **Per-token vs per-batch numerical drift**: Phase 1 changes
  accumulate-order (now: per-token across experts; after: per-expert
  across tokens). FP32 sum is associative for our magnitudes; spot-
  check L29 output cos vs the pre-change path.

## Decision

**Recommended sequencing:**
1. **Profile first** (Phase 0): no code change, ~1h work. Settle the
   bottleneck question with measurement instead of inference.
2. **If Phase 0 confirms**: Phase 1 (expert-grouped dispatch) for v0.5.
3. **If 4×–5× prefill speedup is enough**: stop at Phase 1.
4. **If chasing parity with non-MoE prefill speeds**: continue with
   Phase 2 (coopmat) and Phase 3 (GPU router) in v0.5.x patches.

Targets:
- Phase 1 alone: ~150–200 tok/s prefill for 26B (4×–5× current).
- Phase 1 + 2: 250–400 tok/s.
- Full GPU router (Phase 3): no significant prefill gain (drains are
  ~5% of prefill time), but unlocks async-decode for MoE → +1 tok/s
  decode.

## Files Likely Affected

- `src/backend/vulkan/forward/executor.rs` — `b_step_moe_expert_ffn`
  (~80 LOC rewrite)
- `src/backend/vulkan/forward/runs.rs` — new
  `run_gemv_q4k_grouped` / `run_gemm_q4k_grouped` helpers (~100 LOC)
- `vk_shaders/mul_mat_q4k_grouped.comp` — new shader with row-gather
  (~80 LOC)
- `vk_shaders/scatter_fma_add.comp` — new (~30 LOC)
- Tests: bit-id smoke for routing-decision parity (~50 LOC)

No public API changes. No model-format changes. No release-blocker
risks identified.
