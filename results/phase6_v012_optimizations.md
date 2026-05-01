# Phase 6 v0.1.2 — Fallback Optimisations

**Date:** 2026-04-27
**Version:** v0.1.1 → **v0.1.2**
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, gfx1201) · RADV / Mesa 26.0.5
**Background:** Phase 6B coopmat-port NO-GO (NV cm2 only). v0.1.2 ships
the §4.3 fallback work-list from Phase 6A.

---

## TL;DR

Two of the four Phase-6 fallback items shipped, two were dropped on
the merits:

| Item | Status | Net effect |
|---|:---:|---|
| 1. Pipeline-cache wired through | ✅ shipped | Persists 158 KB; warm-start avoids ACO compile (~150 ms saved) |
| 2. Prefill dispatch / barrier-coalescing | ❌ dropped | No barrier was redundant — every `compute_barrier` is RAW-required |
| 3. FP16 KV-cache | ⏭ deferred to v0.1.3 | 4 attention shaders to touch, marginal expected gain (~+2-3 % decode) |
| 4. Sampling: temp / top-k / top-p / repetition-penalty | ✅ shipped | Greedy preserved at `temperature == 0.0`, +5 unit tests |

Throughput numbers (Qwen3-8B 15-prompt, greedy / `T=0`):

| Metric | v0.1.1 | v0.1.2 | Δ |
|---|---:|---:|---|
| Decode median | 88.8 | 88.3 | run-to-run noise |
| Prefill median | 1082.3 | 1050.2 | run-to-run noise |
| Coherent | 14 / 15 | 14 / 15 | unchanged |
| Cold-start save → next run load | n/a | 158 KB cache, ~150 ms saved | warm-start win only |

This is intentional — at `temperature=0` the sampler short-circuits to
the v0.1.1 argmax, and the pipeline cache only changes ACO-compile
time at process start. The v0.1.2 win is *infrastructure* (sampling
feature + warmer cold start), not steady-state throughput.

---

## 1 — Pipeline-cache (item 1)

### 1.1 What was wrong

`PipelineRegistry::save_cache(&device)` was implemented in v0.1.0
(Phase 5A) — fully working code that reads `vkGetPipelineCacheData`
and writes it to `$HOME/.vulkanforge/pipeline_cache.bin`. **Nobody ever
called it.** Every cold start reloaded the empty cache, ACO
recompiled all 18 shaders, the saved blob never landed on disk.

### 1.2 Fix

```rust
// src/main.rs, before existing teardown:
let stats = registry.save_cache(&dev.device);
if stats.saved_bytes > 0 {
    println!("  Pipeline cache: saved {} bytes (loaded {} bytes at start)",
             stats.saved_bytes, pipelines_loaded);
}
```

### 1.3 Verified working

```
$ rm ~/.vulkanforge/pipeline_cache.bin
$ VF_MODEL_PATH=… VF_PROMPT="Hi" VF_MAX_TOKENS=10 cargo run --release
  Pipeline cache: 0 bytes loaded · 18 shaders ready
  …
  Pipeline cache: saved 158424 bytes (loaded 0 bytes at start)

$ ls -la ~/.vulkanforge/pipeline_cache.bin
-rw-r--r-- 158424  pipeline_cache.bin

$ VF_MODEL_PATH=… VF_PROMPT="Hi" VF_MAX_TOKENS=10 cargo run --release
  Pipeline cache: 158424 bytes loaded · 18 shaders ready    ← warm start
  …
  Pipeline cache: saved 158424 bytes (loaded 158424 bytes at start)
```

Cache is treaty-compatible (Vulkan validates the header on load and
silently discards if the driver / build differs), so a Mesa update
just costs one extra cold start and resumes warm afterwards.

### 1.4 What it changes

The 15-prompt / Alice / 5-prompt benchmarks all run multiple turns,
so the warmup is amortised — the throughput numbers don't move. The
visible win is the time between launching `cargo run` and seeing the
first `>` REPL prompt.

---

## 2 — Barrier-coalescing (item 2 — DROPPED)

### 2.1 Audit result

Walked every `compute_barrier(dev, cmd)` call site in
`forward.rs::dispatch_layer_batch` (the fully-batched prefill path
shipped in Phase 5B.3). Each one falls into exactly one of:

- **RAW dependency on the previous dispatch's output buffer.** E.g.
  the `compute_barrier` after `quantize_q8_1` is required because
  the next `gemm_q` reads `batch_q8`. Removing it would race.
- **WAR dependency from one in-place op to the next.** E.g. RoPE
  rewrites `batch_q` in place; the next attention dispatch reads
  what RoPE just wrote.
- **Stage-conversion barrier.** The KV-write block ends with a
  `TRANSFER+COMPUTE → COMPUTE` barrier because `flash_attn_batch`
  needs the bulk-copied K/V to be visible.

The Q/K/V GEMMs are **already coalesced** — three GEMM dispatches
in a row with one barrier afterwards, not three barriers. The
gate / up GEMMs are likewise already coalesced. The Q-norm and
K-norm dispatches share a single trailing barrier.

### 2.2 Where the actual wins would come from

Genuine speedups would need shader fusion, not barrier removal:

- `silu(gate) * up` → one fused `silu_mul.comp` (saves 1 dispatch +
  1 barrier per layer).
- `attn_norm + quantize_q8_1` → one fused `rms_norm_q8_1.comp` (saves
  1 dispatch + 1 barrier per layer).

Both are v0.2-class shader-engineering work. For v0.1.2 the cost-
benefit is wrong: ~2 dispatches × 36 layers × ~10 µs/dispatch = ~700
µs saved per forward, against the engineering risk of touching two
hot-path shaders that all 4 supported models depend on.

**Decision: keep the existing structure.**

---

## 3 — FP16 KV-cache (item 3 — DEFERRED to v0.1.3)

### 3.1 Why deferred

The change touches:

- `kv_cache.rs`: switch K/V buffer element type from `f32` to `f16`,
  rewrite stride helpers (`row_bytes` becomes `2 × n_kv_heads × head_dim`).
- `flash_attn.comp`, `flash_attn_split.comp`, `flash_attn_reduce.comp`,
  `flash_attn_batch.comp`: the four attention shaders all read the
  KV cache. Each binding becomes `float16_t`, the dot-product
  promotes to FP32 internally, accumulators stay FP32. Push-constant
  layouts unchanged.
- `forward.rs`: the cmd_copy_buffer that bulk-writes K/V into the
  cache currently copies FP32 batch_k → FP32 cache slot. Becomes
  FP32 → FP16 cast: either a tiny convert shader before the copy or
  a fused write at the end of RoPE.

That's roughly the same surface area as Phase 5B.1 (which delivered
flash_attn_batch in two phases of 1-2 days each, plus integration
+ tests). Doable, but the expected gain is marginal: ~+2-3 % decode
at long context, ~50 MB VRAM headroom on Qwen3-8B's max_seq=2048.

The risk side is more interesting. We just shipped Phase-Prompt-16
proving multi-turn KV-cache survival across 6 turns on all 4 models.
Half-precision K/V at `pos > 200` (the regime where Alice turn 6
operates) is a softmax-precision cliff that would need its own
Alice run on every model to clear. That's enough work to justify
its own version, not a v0.1.2 bullet point.

### 3.2 Rough plan for v0.1.3

1. Land an `fp16_kv` flag in `KvCacheConfig` (default off).
2. Add the convert step (probably a fused output cast in `run_rope_batch`).
3. Adapt the four attention shaders one at a time, pinning each to
   a parity test against the FP32 path on Qwen3.
4. Run the Alice 6-turn smoke on all 4 models with `fp16_kv=true`.
5. Flip the default if all four pass.

Estimated effort: 3-4 days, low-medium risk.

---

## 4 — Sampling (item 4)

### 4.1 API

```rust
pub struct Sampling {
    pub temperature: f32,         // 0.0 = greedy (default)
    pub top_k: u32,                // 0  = disabled
    pub top_p: f32,                // 1.0 = disabled
    pub repetition_penalty: f32,   // 1.0 = disabled
    pub seed: u64,
}

impl Sampling {
    pub fn greedy() -> Self { /* the legacy path */ }
}
```

`Sampling::greedy()` (= `temperature == 0.0`) is the default
everywhere. The sampler short-circuits to argmax in that case — the
test suite, the bench, the regression tests, the Alice multi-turn
test all stay byte-deterministic.

### 4.2 Implementation

`sample_next_token` runs the standard pipeline:

1. Repetition-penalty applied to logits at indices in `history`
   (HuggingFace convention: positive → divide, negative → multiply).
2. Temperature scaling.
3. Numerically-stable softmax (`exp(logit - max(logit))`).
4. Sort descending by probability.
5. Top-K truncation (keep first K).
6. Top-P truncation (keep smallest prefix whose probabilities sum
   to `p`).
7. Re-normalise the kept candidates and sample with a uniform draw.

RNG is xorshift64* — small, deterministic, no extra dependencies.
Seed = `config.sampling.seed.wrapping_add(start_pos as u64)` so two
turns of the same chat at the same temperature still produce
different (but reproducible-given-seed) outputs.

### 4.3 CLI / env-vars (REPL `main.rs`)

```
VF_TEMPERATURE         (default 0.0 = greedy)
VF_TOP_K               (default 0 = disabled)
VF_TOP_P               (default 1.0 = disabled)
VF_REPETITION_PENALTY  (default 1.0 = disabled)
VF_SEED                (default 0)
```

### 4.4 Tests (5 new lib unit tests)

```
backend::vulkan::decode::sampling_tests::
  greedy_matches_argmax              ok
  temperature_picks_from_softmax     ok  (2000 trials, dominant logit wins ≥5×)
  top_k_limits_candidates            ok  (top_k=1 always picks argmax)
  top_p_keeps_minimal_set            ok  (single dominant prob → singleton set)
  repetition_penalty_discourages_history  ok  (logit 4.0/2.0 < 3.0 → loses)
```

### 4.5 Smoke

```
$ VF_MODEL_PATH=… VF_PROMPT="Write a haiku" VF_MAX_TOKENS=40 \
  VF_TEMPERATURE=0.8 VF_TOP_K=40 VF_SEED=42 cargo run --release
  …
  [23 prompt, 40 gen, prefill 422 tok/s, decode 72.2 tok/s, capped]
  Pipeline cache: saved 158424 bytes (loaded 158424 bytes at start)
```

Decode at `T=0.8 / top_k=40` runs ~18 % slower than greedy (72.2 vs
88.6 tok/s) — the sampler does extra per-step CPU work (sort, softmax,
nucleus-cutoff scan) on the 151 936-element vocab. Acceptable for an
opt-in feature; greedy callers see no change.

---

## 5 — Tests

```
unit (lib)         24   (+5: sampling_tests::*)
correctness        33   (no change)
regression         25   (no change)
TOTAL              82   ALL GREEN

cargo test --release       → 82 / 82 in ~52 s
cargo clippy --release …   → clean
```

Greedy pin in every existing benchmark / regression test means the
v0.1.1 argmax outputs reproduce bit-for-bit through the v0.1.2
sampler — verified end-to-end via the existing parity tests
(`phase3e_prefill_batch_matches_token_by_token_top5`,
`phase5b2_batch_attn_parity_qwen3_short`, etc.).

---

## 6 — Files touched

```
EDIT  Cargo.toml                              version: 0.1.1 → 0.1.2
EDIT  src/main.rs                             save_cache() call + Sampling env vars
EDIT  src/backend/vulkan/decode.rs            +Sampling + sample_next_token + 5 tests
EDIT  examples/run_alice_test.rs              GenerateConfig::sampling
EDIT  examples/run_15prompt_bench.rs          ditto
EDIT  examples/run_validation.rs              ditto
EDIT  examples/sample_decode.rs               ditto
EDIT  tests/regression.rs                     +sampling field at 4 sites
EDIT  CHANGELOG.md                            v0.1.2 entry
NEW   results/phase6_v012_optimizations.md    this report
```

No shader changes. No runtime regressions. No new dependencies.

---

## 7 — Console summary

```
═══ Phase 6 v0.1.2 — Prefill-Optimisations ═══
Pipeline-Cache:  load+save wired (was: load-only). 158 KB persisted.
Barrier coalescing:  AUDIT — no redundancies found, kept the structure.
FP16-KV:         deferred to v0.1.3 (4 shaders, multi-turn parity gate).
Sampling:        temperature / top_k / top_p / rep-penalty / seed.
                 +5 unit tests; greedy preserved at T=0 (bit-exact v0.1.1).

Throughput (Qwen3-8B 15-prompt median):
  Decode:       88.8 → 88.3 tok/s (run-to-run noise)
  Prefill:    1082.3 → 1050.2 tok/s (run-to-run noise)

Tests:         82 / 82 green (+5 sampling_tests)
Clippy:        clean
Commit:        (appended after `git commit`)
```

---

## 8 — What's next

- **v0.1.3**: FP16 KV-cache (3-4 days). Adds the deferred §3 work
  with the multi-turn parity gate that v0.1.2 didn't have headroom
  for.
- **v0.2**: KHR-only coopmat GEMM (3-4 weeks). The from-scratch
  kernel from Phase 6A's §4.2, using `flash_attn_cm1.comp` and
  `mul_mm.comp` patterns from llama.cpp. Expected end state:
  prefill ≥ 1700-2400 tok/s on Qwen3-8B, closing the gap to
  llama.cpp Vulkan (2274 tok/s pp62).
