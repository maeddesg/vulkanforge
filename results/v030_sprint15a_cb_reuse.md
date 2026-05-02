# Sprint 15A — CB-Reuse Decode Loop: ANALYSIS, no code changes

**Premise.** After Sprints 12-14 closed the shader-and-pipeline-config
branch of the decode-optimisation tree (9 hypotheses falsified), the
remaining 0.80× decode gap to llama.cpp was attributed to graph-level
infrastructure. The brief's working hypothesis: llama.cpp records the
decode command buffer **once** as a template and per-token only updates
small parameters (position / KV-offset) before re-submitting,
eliminating ~720 `vkCmd*` calls per token. Bench-gate: decode ≥ 95 tok/s
(+4 % vs the 91.1 baseline).

**Verdict.** **Pre-check failed — STOP per "BEI UNKLARHEITEN SOFORT
STOP".** The measurement is real and the lever has the right
magnitude (CB-record = **1 836 µs / token** at steady-state, ~17 % of
the 11 ms wall), but reading llama.cpp's source code shows the brief's
central hypothesis is wrong: **llama.cpp also re-records per token.**
Their `ggml_vk_dispatch_pipeline` calls
`pushConstants + bindPipeline + bindDescriptorSets + dispatch` fresh
for every dispatch every forward, just like we do
(`ggml-vulkan.cpp:6622-6629`).

llama.cpp's lead is therefore **not** "template CB-reuse with
parameter UBO". It must be either (a) async multi-submit / CPU-GPU
pipelining via timeline semaphores, (b) GPU-kernel differences we
haven't yet measured, or (c) some combination. Without source-level
clarity on which, **a multi-week shader-side push-constant → UBO
refactor (the brief's Option B) cannot be justified by "we're
copying llama.cpp's approach"** — they're not doing that.

No code changes shipped. 27 / 27 lib tests, 15 / 15 coherent (current
v0.2.4 default config). This is an analysis-only sprint.

## 1. Measurement — CB-record time is real

`profile_forward` on Qwen3-8B-Q4_K_M, all-positions sweep:

```
   pos  pre_setup     reset     begin    RECORD       end    submit  GPU_WAIT  readback     TOTAL
     0          0        78         2      1604         0        16      8707        21     10432
    50          0        21         1      1751         0        13      8810        22     10621
   100          0        22         1      1782         0        15      9051        21     10894
   150          1        21         1      1904         0        16      9024        25     10996
   200          1        24         3      1637         0        17      9030        19     10735

Median over 50-token windows:
 50-99           0        22         1      1784         0        14      9055        22     10902
100-149          0        22         1      1868         0        14      9049        22     10980
150-199          0        22         1      1836         0        14      9034        22     10934

Floor: empty record block (no dispatches at all)        82 µs
Drill-down at pos=100 — per-layer time inside RECORD:
  RECORD wall                1593 µs  (one-shot timer)
  Σ per-layer dispatches     1586 µs  (99.6% of RECORD)
  per-layer min/med/max        40 /     41 /     72 µs
  dispatch_final + barrier      6 µs
```

(All figures in µs; median over 50-token windows.)

**Steady-state breakdown per token (pos=150-199 window):**

| Phase | µs | % of wall |
|---|---:|---:|
| `reset` (`vkResetCommandBuffer + reset_fences`) | 22 | 0.2 % |
| `begin` (`vkBeginCommandBuffer`) | 1 | 0.0 % |
| **`RECORD`** (per-layer dispatches, host-side) | **1 836** | **16.8 %** |
| `end` (`vkEndCommandBuffer`) | 0 | 0.0 % |
| `submit` (`vkQueueSubmit`, host-only) | 14 | 0.1 % |
| **CPU subtotal** | **~1 873** | **17.1 %** |
| `GPU_WAIT` (`vkWaitForFences`) | 9 034 | 82.7 % |
| `readback` (logits → host) | 22 | 0.2 % |
| **TOTAL** | **10 934** | **100 %** |

`1/10 934 µs = 91.5 tok/s` — matches our 91 tok/s 15-prompt median.

**The 1 836 µs RECORD figure is well above the brief's 500 µs ROI
threshold.** A perfect CB-reuse system that could skip the entire
RECORD phase would, *in isolation*, give:

- Naive: `1 / (10 934 - 1 836) µs = 110 tok/s` (CPU and GPU stay
  serial, just shorter CPU phase) — **+20 %**.
- Pipelined: `1 / max(1 873, 9 034) µs = 110 tok/s` — same upper
  bound, since CPU is hidden inside GPU_WAIT.

So the *theoretical* upper bound on this lever is ~110 tok/s,
**still short of llama.cpp's 114**. Some part of llama.cpp's lead
is GPU-side, not CPU-side. CB-reuse alone cannot close the gap.

## 2. llama.cpp source review — no template CB-reuse

The brief assumed llama.cpp records the decode CB once and re-submits
it with updated parameters. **That is not what they do.** Their
`ggml_vk_dispatch_pipeline` (`ggml-vulkan.cpp:6601-6630`):

```cpp
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx,
        vk_context& subctx, vk_pipeline& pipeline, /* … */ const T &push_constants,
        std::array<uint32_t, 3> elements) {
    // … workgroup-size math, asserts …
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    subctx->s->buffer->buf.pushConstants(pipeline->layout, …, push_constant_data(push_constants));
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout, 0, { descriptor_set }, {});
    subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
}
```

This is the **exact same per-dispatch pattern we use**: push
constants + bind pipeline + bind descriptors + dispatch, fresh
into a recording CB. Called for every dispatch in every forward.
There is no "record once, update params, re-submit" pathway.

Their `ggml_vk_get_compute_ctx` (`ggml-vulkan.cpp:6659-6676`) does
re-use a "compute context" via a `weak_ptr` for the lifetime of a
graph, but the underlying CB inside it is freshly recorded each
graph evaluation (their `ggml_vk_ctx_begin` calls
`ggml_vk_begin_submission` which begins a new CB).

**The CB-reuse hypothesis is falsified at the source level.** The
brief's Option A (template CB with push-constants) and Option B
(template CB with UBO) are not llama.cpp's design.

## 3. Where llama.cpp's lead actually lives

If not CB-reuse, then what? Reading their decode path more
carefully and cross-referencing what the brief sketched, three
candidates remain:

### 3.1 Async multi-submit via timeline semaphores

`ggml_vk_submit` (`ggml-vulkan.cpp:2409-2417`) builds a
`std::vector<vk::SubmitInfo>` with `tl_wait_semaphores` and
`tl_signal_semaphores` (timeline semaphores). They submit
multiple CBs with a chain of timeline-semaphore dependencies and
**don't fence-wait between submits**. The wait happens lazily.

This *could* allow CPU recording of token N+1 to overlap with GPU
compute of token N, IF the next token's recording doesn't need
data from the previous token's logits.

It doesn't — there is a hard dependency through `sample(logits[N])
→ embedding(token[N+1]) → write to scratch_a → forward(N+1)`. The
embedding-upload step needs the previous logits in CPU memory.

**But:** the *recording* phase doesn't read those values; it only
references the buffer handles. So in principle:

```
T=0:  sample[N-1] → embed[N] → scratch_a[N] → record CB[N] → submit CB[N]
            ↘ CPU-side bookkeeping for sample[N+1]:          [hidden in GPU_WAIT]
T=g:  GPU runs CB[N] (g = ~9 ms)
            ↘ CPU records CB[N+1] (with NEW push-constants for pos=N+1):
T=g+r: wait CB[N] → read logits[N] → sample[N+1] → embed[N+1] → scratch_a[N+1] → submit CB[N+1]
```

If `r ≤ g` (CPU recording fits inside the GPU window), the wall
time per token compresses to ~`max(record + readback + sample + embed + submit, GPU)`
= ~`max(1.9 ms + ε, 9 ms)` = ~9 ms → **111 tok/s**.

This requires:
- Two persistent CBs (alternating)
- Submit-without-fence-wait + lazy fence-wait at the right moment
- A way to record CB[N+1] without knowing the embedding values
  (they're already in `scratch_a` — we just don't *write* to
  `scratch_a` until after sampling, but recording references the
  buffer handle, not its contents)

This is a real architectural change but **does not require shader
modifications**. The push-constants stay where they are; only the
host-side submission orchestration changes. Estimated scope: 200-400
LOC across `commands.rs`, `forward.rs`, and the calling main loop.
Estimated effort: 1-2 weeks to do safely, including correctness
tests across all 4 supported model families.

### 3.2 Dedicated `lm_head` coopmat path

`lm_head` is a vocab-major GEMV at N=151 936. Sprint 12G-D measured
it at ~6 % of the decode forward (740 µs / 12 ms). A coopmat dispatch
there could shave ~3 % decode wall.

Smaller scope (~50-100 LOC), faster to ship and validate (~1 sprint),
clearer correctness story (just port the coopmat recipe to one more
shape). Doesn't close the full gap but is a real, contained win.

### 3.3 Buffer-aliasing / live-set reduction

We hold ~20 SSBOs live per layer (scratch_a, scratch_b, batch_q8,
batch_norm, multiple per-layer caches, etc); llama.cpp recycles
into 3-4 via their `ggml_vk_pool` mechanism. May matter for L2
thrashing — Sprint 12G-D didn't measure cache-side effects
directly. Unmeasured lever; could be 0 % or could be 5 %.

## 4. Why I am not implementing this sprint

The brief framed Sprint 15A as a single-sprint deliverable with a
specific bench-gate (decode ≥ 95 tok/s) and a specific lever
("CB-reuse decode loop"). After measuring (✓) and reading
upstream (✗), the lever as specified does not exist in the
reference implementation. Three honest paths remain:

1. **Pivot Sprint 15A scope**: implement async multi-submit
   instead. Requires a redesign of the brief, not just a redesign
   of the implementation. Section 3.1 sketches the architecture;
   the scope is 1-2 weeks of careful work, not a single sprint.
2. **Pivot Sprint 15A target**: implement the smaller `lm_head`
   coopmat path (Section 3.2). One sprint, ~3 % decode lift,
   doesn't reach the ≥ 95 gate but is real progress.
3. **Stop and document** (this report). No code changes. The
   user (next session) decides whether to fund the multi-week
   async-multisubmit refactor or pivot to the smaller `lm_head`
   sprint.

I am taking path 3 because:

- The brief's Fallstrick #1 (Sprint 12 lesson: "IMMER messen, nie
  schätzen") explicitly maps to "if the assumed lever isn't real,
  STOP and report".
- Touching `forward.rs`'s `forward_token` + `commands.rs`'s
  `one_shot` for an incremental async-multisubmit prototype that
  may or may not gate-pass is high-risk for the kernel of the
  engine — exactly the case the brief flags with "BEI
  UNKLARHEITEN SOFORT STOP. Das ist KERN-Infrastruktur!".
- The honest measurement (CB-record = 1 836 µs) and the honest
  upper bound (~+20 % wall reduction → ~111 tok/s) are now
  documented and reproducible from this report. A future sprint
  with the right scope can build on them.

## 5. What this sprint contributes

- **Hard measurement** of the per-token CPU recording cost
  (1 836 µs steady-state, well above the brief's ROI floor).
- **Source-level falsification** of the "llama.cpp uses template
  CB-reuse" hypothesis — they re-record per token too.
- **Architectural sketch** for an async multi-submit redesign
  (Section 3.1) that captures most of the lever without shader
  changes.
- **Smaller-scope alternative** identified (`lm_head` coopmat,
  Section 3.2) that fits a single sprint.

## 6. Recommendations

For the next sprint:

- **If the team has ~2 weeks of focused capacity for v0.3-A**:
  pivot Sprint 15 to async multi-submit (Section 3.1). Estimated
  lift: +5–15 % decode (depending on how cleanly we can hide CPU
  recording inside GPU_WAIT).
- **If the team wants a single-sprint deliverable**: pivot to
  Sprint 15B (`lm_head` coopmat, Section 3.2). Estimated lift:
  +2–4 % decode. Doesn't reach the ≥ 95 gate but is a real,
  measurable win on a contained scope.
- **In all cases**: revise this report's Section 3.1 estimate
  with a 1-day prototype that measures whether `record(CB[N+1])`
  can actually overlap GPU(`CB[N]`). If our descriptor-set
  cache (Sprint 5A) prevents that overlap (because both CBs would
  reference the same descriptor pool / same cached sets), the
  lift drops from "+15 %" to "+5 %" and the Section 3.2
  alternative becomes more attractive.

## 7. Outputs

- This report.
- **No code changes.** v0.2.4 binary unchanged. 27 / 27 lib
  tests, 15 / 15 coherent (current default-config baseline).
- `examples/profile_forward.rs` (existing, unchanged) is the
  reproducible measurement harness for Section 1.
- The decode-gap analysis ledger now has **10** entries (the 9
  Sprint 12-14 falsified hypotheses + this one): the brief's
  CB-reuse-as-llama.cpp-recipe hypothesis is the 10th false
  starting point. The remaining unfalsified candidates are
  Sections 3.1 (async multi-submit), 3.2 (`lm_head` coopmat),
  and 3.3 (buffer-aliasing) — none of them are "single
  config-flip" levers.

## 8. Honest framing

This is what the v0.2 → v0.3 boundary actually looks like. Every
"copy the llama.cpp config" lever has been tried; the remaining
work is **architectural** (async submit, dedicated kernels,
buffer-pool redesign) and is fundamentally multi-sprint. The
v0.2 series got us from 0.52× (v0.2.0) → 0.89× (v0.2.2) prefill
and ironed out a complete coopmat tile matrix; v0.3 will need
graph-level work to push decode from 0.80× toward parity.

There's no shame in the brief's hypothesis being wrong. It was
*plausible* — every prior sprint that opened with "let's port
llama.cpp's recipe X" found something concrete to either ship or
falsify. This one falsified at the source-reading step, which is
faster and cheaper than at the implementation step. That is the
test methodology working as designed.
