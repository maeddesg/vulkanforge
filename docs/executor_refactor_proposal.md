# `executor.rs` Refactor Proposal — Sprint 57A Analysis

**Methodology:** mirrors the Sprint 44A `forward.rs` refactor pattern
(7816 LOC → 13 modules under `forward/`). No code change in this
sprint — analysis + proposal only.

---

## 0. Brief Premise Correction

The brief opens with "executor.rs = 130k LOC". **The actual file is
2995 LOC** (verified with `wc -l`). Likely a stale estimate or
typo-by-three-orders-of-magnitude.

This changes the refactor cost/benefit conversation: a 3k-line file
is awkward but not catastrophic. The 44A refactor was on a file 2.6×
larger. The case for splitting `executor.rs` is real but more about
**cohesion** than **navigability**.

## 1. Bestandsaufnahme

| Metric | Value |
|--------|-------|
| Total LOC | 2995 |
| Total `fn`s | 88 |
| `pub fn` (external API) | 0 |
| `pub(super) fn` (module-internal API) | 3 (`cpu_moe_route`, `compute_to_transfer_barrier`, `transfer_to_host_barrier`) |
| `fn step_*` (DEC) | 33 |
| `fn b_step_*` (BAT) | 33 |
| DEC↔BAT name symmetry | 100% (all 33 names match) |
| `impl` blocks | 2 (`DecodeExec`, `BatchExec`) |
| Largest function | `step_moe_expert_ffn` — 252 LOC |
| Top 10 functions share | 1426 LOC = **48%** of file |
| Top 5 are all MoE | `step_moe_expert_ffn` (252), `b_step_ple_block` (236), `b_step_moe_expert_ffn` (195), `b_run_proj` (189), `step_moe_route` (133) |
| Baseline `cargo build --release` after `touch executor.rs` | **5.71 s** |

### External Surface

What sibling modules import from `executor`:

```
prefill.rs    super::executor::{ExecCtx, ExecMode, BatchExec}
decode.rs     super::executor::{ExecCtx, ExecMode, DecodeExec}
```

That's it. 5 items total (3 types, 1 enum, 1 struct). Re-exporting
from a future `executor/mod.rs` is trivial.

### `impl` Block Boundaries

```
   1- 240  Top level (use statements, ExecCtx, ExecMode, cpu_moe_route,
                       compute_to_transfer_barrier, transfer_to_host_barrier)
 242-1452  impl DecodeExec   (DEC path, 1211 LOC)
 1455-2995 impl BatchExec    (BAT path, 1540 LOC)
```

## 2. Function Categorization (LOC totals)

Each function classified by purpose. Sizes from line-diff between
adjacent function starts.

| Category | DEC LOC | BAT LOC | Combined | Functions |
|----------|---------|---------|----------|-----------|
| **A. Attention** | 417 | 425 | **842** | 18 each |
| **B. Dense FFN** | 150 | 144 | **294** | 7 each |
| **C. MoE** | 457 | 377 | **834** | 6 each |
| **D. Layer-Control** | 202 | 329 | **531** | 4 each + execute_step |
| **E. BAT-helpers** | — | 214 | **214** | `b_run_proj`, `b_subscriber_q_barrier`, `b_fused_attn_residual_norm` |
| **F. Top-level** | — | — | **156** | `cpu_moe_route`, barrier helpers, types |
| **TOTAL** | 1226 | 1489 | **2871** + 124 (decl/comments) |

### Category A — Attention (842 LOC, 36 fns total)

```
attn_norm, q_proj, k_proj, v_proj, v_from_k_raw,
q_bias, k_bias, v_bias,
q_norm_rope, k_norm_rope, q_rope, k_rope,
v_norm, kv_write,
attention (flash_attn dispatch), o_proj,
post_attn_norm, attn_residual_add
```

DEC and BAT pendants exist for all 18; bodies differ in `seq_len`
handling but follow the same skeleton.

### Category B — Dense FFN (294 LOC, 14 fns total)

```
pre_ffn_norm, gate_proj, up_proj, activation,
down_proj, post_ffn_norm, ffn_residual_add
```

Cleanest 1:1 category. The 7-step dense block is identical in
structure DEC ↔ BAT.

### Category C — MoE (834 LOC, 12 fns + 1 top-level)

```
moe_route, moe_expert_ffn, moe_branch_add,
post_dense_mlp_norm, pre_moe_norm, post_moe_norm
+ cpu_moe_route (top-level helper)
```

**Single biggest category by LOC**. `step_moe_expert_ffn` (252) and
`b_step_moe_expert_ffn` (195) carry ~54% of the category. The Sprint
56A/56B GPU-router branches now live in both `step_moe_route` and
`b_step_moe_route`. **Touching this category is the highest-risk
change in the codebase** — the 29-sprint bug hunt (53A-54I) lived
entirely here.

### Category D — Layer Control (531 LOC, 7 fns + 2 execute_step)

```
ple_block (97 in DEC, 236 in BAT — BAT is 2.4× longer),
layer_scalar_mul,
execute_step (DEC: 65, BAT: 69) — the match dispatcher
fused_attn_residual_norm (DEC: 25, BAT: 17) — Llama-only lookahead fusion
```

`step_ple_block` for BAT is unusually large (236 LOC) compared to DEC
(97 LOC) because the prefill PLE path applies the 5-dispatch sequence
per token across the batch.

### Category E — BAT-only helpers (214 LOC)

```
b_run_proj                  189 LOC — the heart of batched K/V/Q projection
b_subscriber_q_barrier      8 LOC
b_fused_attn_residual_norm  17 LOC — but counted in D above
```

`b_run_proj` is the per-token dispatch entry point for batched Q/K/V
projection; it has no DEC analogue because DEC uses direct per-step
helpers.

## 3. Split Proposal

### Recommended layout: `executor/` directory mirror of `forward/`

```
forward/executor.rs (2995 LOC)
   ↓ becomes ↓
forward/executor/
    mod.rs        (~200 LOC: ExecCtx, ExecMode, DecodeExec, BatchExec
                   struct decls, top-level barrier helpers, cpu_moe_route,
                   module declarations + re-exports for sibling modules)
    dispatch.rs   (~150 LOC: BOTH execute_step impls + fused-norm helpers —
                   the cohesion-critical match-dispatchers that enforce
                   the "no missing LayerStep arm" invariant)
    attention.rs  (~842 LOC: step_* + b_step_* for 18 attention names)
    ffn.rs        (~294 LOC: step_* + b_step_* for 7 dense FFN names)
    moe.rs        (~834 LOC: step_* + b_step_* for 6 MoE names)
    control.rs    (~430 LOC: step_ple_block (DEC+BAT), step_layer_scalar_mul
                   (DEC+BAT), and b_run_proj — the per-batch K/V/Q proj
                   entry point that doesn't fit elsewhere)
```

### Why this carving

**1. DEC and BAT live in the SAME file per category.** The whole
point of the executor pattern, per the file's own docstring, is that
adding a `LayerStep` variant compiles-fails in BOTH `execute_step`
matches until handled. Splitting by category (rather than by
DEC-vs-BAT) preserves that invariant: a future contributor adding,
say, `step_grouped_moe_dispatch` finds `step_moe_expert_ffn` and
`b_step_moe_expert_ffn` in the same file — they cannot miss the BAT
side.

This is exactly the lesson saved in
`memory/feedback_layer_dispatch_paths.md`: "a new per-layer step
must land in dispatch_layer, dispatch_layer_batch, AND
dispatch_layer_partial; skipping one ships a silently broken path."

**2. MoE gets its own module.** It's the largest category (834 LOC),
the most fragile (29-sprint bug hunt), and growing (56A/56B added the
GPU-router branches; 56C will add GPU-direct expert dispatch).
Isolating it under `executor/moe.rs` makes the next major change
reviewable.

**3. Dense FFN is the easiest extraction (294 LOC).** It has the
cleanest 1:1 DEC/BAT pendants. Use it as the **first** migration to
validate the impl-split mechanics work, before touching attention or
MoE.

**4. `execute_step` matchers stay together in `dispatch.rs`.** They're
small (134 LOC combined) but they're the keystone — both impls need
to enumerate the same `LayerStep` variants. Putting them in their own
file makes that pair visible side-by-side.

### Files NOT introduced

- ❌ `executor_dec_*.rs` + `executor_bat_*.rs` — would split the
  match-pair invariant across files (anti-pattern per 44A's lesson).
- ❌ `executor_runs.rs` — `runs.rs` already exists at the parent
  level; the existing `Forward::run_*` family lives there.
- ❌ `executor_barriers.rs` — the 2 barrier helpers
  (compute_to_transfer, transfer_to_host) are ~30 LOC total. Keep in
  `mod.rs`.
- ❌ `executor_debug.rs` — debug helpers already in
  sibling `debug.rs`.

## 4. DEC ↔ BAT Symmetry & Duplication

100% symmetric by **name** (33/33 functions in each impl block match
exactly). But how much **body** is identical?

Spot-checks on a few similarly-sized DEC↔BAT pairs:

| Pair | DEC LOC | BAT LOC | Bodies |
|------|---------|---------|--------|
| `step_q_norm_rope` ↔ `b_step_q_norm_rope` | 23 | 25 | ~85% same; BAT loops over `seq_len` |
| `step_kv_write` ↔ `b_step_kv_write` | 53 | 58 | ~70% same; BAT computes per-token offsets |
| `step_moe_expert_ffn` ↔ `b_step_moe_expert_ffn` | 252 | 195 | ~50% same; BAT is shorter because it shares more inner helpers |
| `step_ple_block` ↔ `b_step_ple_block` | 97 | 236 | Bodies diverge significantly; BAT applies the 5-PLE-dispatch sequence per-token |

**Could DEC be expressed as BAT with `seq_len = 1`?** Theoretically
yes for ~80% of the steps. In practice the existing code paths have
diverged for good reasons (BAT uses `batch_residual` / `batch_q`
buffers, DEC uses per-slot `res1` / `q_buf`). Collapsing them is a
**separate, deeper refactor** that would:
1. Unify the slot/batch buffer layouts (~100s LOC change to
   `state.rs` + `setup.rs`).
2. Re-thread every per-step caller.
3. Break bit-identity unless extremely carefully done.

**Not in scope for this refactor.** This proposal is a pure
code-move that preserves DEC↔BAT separateness; collapse-as-special-
case is a "Sprint 60+" topic that should be approached only after
57A lands and a fresh shape of the executor is visible.

## 5. Migration Order (4 commits, smallest → riskiest)

**Each step is a pure code-move; behaviour must be bit-identical
after each commit. Tests + 4-model regression after every commit.**

1. **`executor/mod.rs` skeleton + `dispatch.rs`** (≈150 LOC moved)
   - Create `executor/` directory.
   - Move top-level types (`ExecCtx`, `ExecMode`, `DecodeExec`,
     `BatchExec` struct decls, barrier helpers, `cpu_moe_route`) into
     `mod.rs`.
   - Move both `execute_step` impls + Llama-fusion helpers into
     `dispatch.rs`.
   - Rest of `executor.rs` body becomes a temporary sibling block in
     `mod.rs` for the in-flight migration. Verify build green.

2. **`executor/ffn.rs`** (≈294 LOC moved)
   - Lowest-risk category (1:1 DEC/BAT, no MoE complexity).
   - Validates the impl-split mechanism for ALL further categories.

3. **`executor/attention.rs`** (≈842 LOC moved)
   - Larger but well-trodden. Most-changed in pre-v0.4 sprints, so
     the diff is reviewable.

4. **`executor/moe.rs`** (≈834 LOC moved) + **`executor/control.rs`** (≈430 LOC moved)
   - Highest-risk category. Move LAST after all the simpler ones
     are in place + verified.
   - `control.rs` (ple_block, layer_scalar, b_run_proj) can fold
     into the same commit since it's small.

After step 4: `mod.rs` is back to its skeleton role (~200 LOC).
Total file count goes 1 → 6.

## 6. Risiken (Risks)

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Bit-identity regression from impl-split | Low | Each commit is a pure code-move; full 4-model regression + 26B coherence between each. Rust's impl-can-be-split-across-files semantics are mature. |
| Hidden CFG-attribute or private-item leak | Low | `cargo build --release` errors are loud; the only `pub(super)` items export from siblings already, which works whether they live in `mod.rs` or a submodule. |
| Refactor cost > refactor benefit at 3k LOC | **Medium** | This is the honest concern. A 3k file is awkward but tractable. The case for splitting is strongest if Sprint 56C (GPU-direct expert FFN) is coming next — it touches MoE heavily and benefits from a focused `executor/moe.rs`. |
| Loss of "everything is here" navigation | Low | Modern Rust tooling (rust-analyzer goto-def) makes file boundaries invisible. The current 3k file already requires search anyway. |

## 7. Compile-time Baseline

```
$ touch src/backend/vulkan/forward/executor.rs && time cargo build --release
... 5.71 s
```

After the split, ONLY the file containing the changed function would
need recompiling (assuming no public-API changes between submodules).
For a typical MoE-only change, that's `executor/moe.rs` (~834 LOC)
instead of the full 2995. Estimated incremental rebuild after splitting:
**~3-4 s** for category-local changes; the 5.71 s baseline only applies
when `mod.rs` itself changes.

This is a **modest** win at 3k LOC — savings ~1-2 s per build. The
real value is **cohesion**, not compile time.

## 8. Recommendation

**Carve, but only as a precursor to a substantive change** (e.g.,
Sprint 56C's GPU-direct expert FFN, or a Q4_K_M coherence push).
Pure refactoring of a 3k file for its own sake is hard to justify
given:
- The current navigation is awkward but tractable.
- Each migration commit carries small but non-zero regression risk.
- The 29-sprint bug hunt is fresh — the codebase rewards
  conservatism.

If the next sprint IS 56C, do the refactor **first** (in 4 commits over
1 session, mostly mechanical), so 56C lands in a clean, focused
`executor/moe.rs`. If the next sprint is unrelated (e.g., new model
support, longer context test), defer the refactor and revisit when
MoE work resumes.

**Concrete proposal for next steps:**

- If the next 1-2 sprints touch MoE → **do the refactor now**
  (Sprint 57B = mechanical migration in 4 commits).
- If the next 1-2 sprints DON'T touch MoE → **defer**; the file is
  tractable as-is. Re-evaluate in 2-3 sprints.

## 9. What This Document Is NOT

- Not an authorization to start coding. **Sprint 57A is analysis-only
  per the brief regulations.**
- Not a commitment to collapse DEC ↔ BAT into a unified `seq_len`-
  generic executor. That's separate, deeper, and out of scope here.
- Not a critique of the current code. The single-file `executor.rs`
  was a deliberate choice in Sprint 44C-2 (per its docstring) and
  served well through Sprints 51-56. The refactor proposal here is
  about preparing for the **next** wave of MoE changes, not fixing
  something broken.
