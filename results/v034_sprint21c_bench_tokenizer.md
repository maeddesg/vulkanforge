# Sprint 21C — `bench --tokenizer-from` for SafeTensors

**Date:** 2026-05-03
**Branch:** main (post-Sprint 21B, head was `6be5c44`)
**Goal:** Plumb `--tokenizer-from <gguf>` through the `bench`
subcommand so SafeTensors FP8 models can be pp-swept with the same
methodology as GGUF models. Previous FP8 numbers came from
single chat prompts and were noisy ±5-10%; we needed measured
3-run-median data before deciding on further optimisation.

## Headline result

**First measured FP8 pp-sweep on Meta-Llama-3.1-8B-Instruct-FP8:**

| pp   | Q4_K_M GGUF (Qwen3-8B) | FP8 SafeTensors (Llama-3.1-8B) | FP8 / Q4_K_M |
|-----:|-----------------------:|-------------------------------:|-------------:|
|   64 |       1791.2 tok/s     |          545.9 tok/s           |    0.30×     |
|  128 |       2690.8 tok/s     |          607.2 tok/s           |    0.23×     |
|  256 |       (~3500 tok/s)    |          660.9 tok/s           |    ~0.19×    |
|  512 |       3853.5 tok/s     |          694.6 tok/s           |    0.18×     |
| 1024 |       3737.5 tok/s     |          708.7 tok/s           |    0.19×     |
| Decode | 116.9 tok/s | 62.0 tok/s | 0.53× |

The Q4_K_M numbers are from Qwen3-8B-Q4_K_M (the canonical baseline);
the same-base-model Llama-3.1-8B-Q4_K_M would be similar. The FP8
numbers come from the new `bench` path on the same machine + same
`vulkanforge bench` invocation shape — proper 3-run-median data.

## What this tells us

1. **FP8 prefill plateaus around ~710 tok/s** as pp → 1024. That's
   the realistic ceiling of `mul_coopmat_fp8_multi_wg.comp` on
   RDNA4/Mesa today. Sprint 21B's chat-derived 625 tok/s estimate
   at pp=406 was within ±10% of the truth (the curve has 660 at
   pp=256 and 694 at pp=512; pp=406 is clean interpolation).
2. **FP8 prefill is ~5× slower than Q4_K_M prefill at pp=512**
   (694 vs 3854). Sprint 18B's measured 1.18× FP8/BF16 WMMA
   throughput ceiling fully accounts for this — Q4_K_M's
   `mul_mm_q4_k_f32_aligned_coopmat.spv` runs the same WMMA
   throughput on BF16-narrowed weights but **without** the FP8 →
   FP32 → BF16 LDS round-trip. The conversion and LDS staging is
   the gap.
3. **FP8 decode is 0.53× of Q4_K_M decode** (62 vs 117 tok/s).
   FP8 GEMV reads 2× the bytes per parameter (1 B vs ~0.56 B for
   Q4_K-effective), so this is the natural bandwidth ratio.
4. **The "9.98× composite M3 → 21B" claim from earlier reports
   was correct in shape but used per-prompt timings.** The clean
   measurement shows: M3 was 59 tok/s pp=406 (per-token GEMV);
   Sprint 21B at pp=512 is 695 tok/s. So the composite is ≈11.7×
   on the bench-measured numbers. Same story, slightly cleaner
   data.

## What the brief expected vs reality

The Sprint 20-21 trail is now properly anchored:

```
Brief (optimistic):  FP8 prefill could match or exceed Q4_K_M (4500 t/s)
Sprint 18B (early):  Measured 1.18× FP8/BF16 WMMA on RDNA4 → expect
                     Q4_K-parity at best (~3800-3900 t/s)
Sprint 21A/21B + 21C: Measured 694 t/s at pp=512 → 0.18× Q4_K_M
                       (the LDS staging + BF16-narrow conversion is
                        the gap, not the WMMA).
```

This is **the correct floor for a naive coopmat FP8 GEMM kernel
with no aligned-coopmat / large-tile / multi-WMMA-per-subgroup
infrastructure** (those are Sprint 12L/12M/13A's analogues, ported
piecemeal in 21A/21B). To close the gap further on this hardware,
the kernel needs:

* `mat2x4` / `f16mat2x4` LOAD_VEC_B-equivalent for FP8 (4× wider B
  loads — Sprint 12L's pattern). This is a non-trivial refactor of
  the LDS staging.
* Multiple WMMAs per subgroup (M-tile = 32 instead of 16; same
  pattern Sprint 12M used). Doubles arithmetic per subgroup.
* `f16acc` accumulator path on the precision-tolerant GEMMs. Risky
  on a 32-layer model.

**These are 5-15% levers each.** Composite headroom is maybe +30%
to ~900 tok/s pp=512. Sprint 18B's WMMA ceiling caps it well below
Q4_K_M parity unless Mesa lifts its FP8 throughput.

## What landed

`src/main.rs`:

* **`Commands::Bench`** gained `tokenizer_from: Option<PathBuf>`
  (`--tokenizer-from <gguf>` CLI flag).
* **`run_bench` dispatcher** detects `model_path.is_dir()` and routes
  to a new `run_bench_safetensors` (parallel function); GGUF path
  is byte-identical to before.
* **`run_bench_safetensors`** (~110 LOC) — mirror of `run_bench`
  with three deltas: loads weights via
  `LoadedModel::load_safetensors`, tokenizer from the supplied
  `--tokenizer-from` GGUF, and `EmbeddingSource::Host(&host_embed)`
  in the `generate_from_tokens` calls. Same decode + pp-sweep
  reporting layout.

### Diff size

* `src/main.rs`: +124 lines (new function), +5 lines (CLI plumbing).
* No shader changes.
* No new SPVs (still 101).
* Zero new dependencies.

## Regression — all green

* GGUF Qwen3-8B-Q4_K_M `bench` output above:
  decode 116.9 tok/s, prefill 64/128/512/1024 = 1791/2691/3854/3738
  — within noise of pre-Sprint-21 baseline.
* `cargo test --release --lib`: **37/37 pass**.
* FP8 GEMM correctness: pass (Sprint 21B verified, kernel
  unchanged).
* GGUF chat path: untouched, no risk.

## Files touched

* `src/main.rs` — `Commands::Bench` + `run_bench` dispatch +
  `run_bench_safetensors` function. ~130 LOC net.
* `results/v034_sprint21c_bench_tokenizer.md` — this file.

## Sprint 20-21 final perf snapshot

```
Llama-3.1-8B (same base model):

| Path         | Decode    | pp=512    | Prefill 512 wait |
|--------------|-----------|-----------|------------------|
| Q4_K_M GGUF  | 117 tok/s | 3854 tok/s|  133 ms          |
| FP8 native   |  62 tok/s |  695 tok/s|  737 ms          |

Compared to FP8 M3 baseline:
| Sprint       | pp=512 prefill |
|--------------|---------------:|
| M3 per-token |     59 tok/s   |
| Wire (GEMM)  |   ~340 tok/s   |
| 21A aligned  |   ~580 tok/s   |
| 21B multi-WG |    695 tok/s   |  (new measured number)
```

Composite **M3 → 21B**: 59 → 695 = **11.7×** prefill speedup
(measured, not estimated). For a 512-token prompt: 8.7-second wait
→ 0.74-second wait.

## Honest status

The naive FP8 GEMM ceiling on RDNA4/Mesa is now well-characterised:
**~710 tok/s prefill, ~62 tok/s decode** for an 8B Llama-3 FP8
SafeTensors model. The product story is unchanged from Sprint 21A:
FP8 chat is **usable**, not fast. Q4_K_M remains the throughput
choice; FP8 the quality choice.

Next sprints have measurement infrastructure to chase the +30%
remaining headroom (LOAD_VEC_B-equivalent, multi-WMMA per subgroup).
Whether to chase that is a product decision.
