# VulkanForge Tests & Diagnostic Tools

Index of test and diagnostic artefacts. Public-facing so external
references (e.g. Mesa/RADV/ACO RFEs) can link directly into the repo.

## Correctness tests (`tests/*.rs`)

Run with `cargo test`.

| Test | Purpose |
|------|---------|
| `correctness.rs`               | End-to-end forward-pass correctness vs reference |
| `regression.rs`                | Cross-shape / cross-quant regression matrix |
| `coopmat_q4k.rs`               | Q4_K cooperative-matrix GEMV correctness |
| `coopmat_tiled.rs`             | Tiled coopmat kernel correctness |
| `dequant_q4k.rs`               | Q4_K dequant unit test |
| `q3k_dequant_sanity.rs`        | Q3_K dequant sanity |
| `q3k_gemv_correctness.rs`      | Q3_K GEMV correctness |
| `q4_0_dequant_sanity.rs`       | Q4_0 dequant sanity |
| `q4_0_gemv_correctness.rs`     | Q4_0 GEMV correctness |
| `q5k_dequant_sanity.rs`        | Q5_K dequant sanity |
| `q5k_gemv_correctness.rs`      | Q5_K GEMV correctness |
| `fp8_gemm_correctness.rs`      | FP8 GEMM correctness |
| `fp8_gemv_correctness.rs`      | FP8 GEMV correctness |
| `flash_attn_tiled_ref.rs`      | Flash-attention tiled reference |

## Benchmark + diagnostic examples (`examples/*.rs`)

Run with `cargo run --release --example <name>`.

| Example | Sprint | Purpose |
|---------|--------|---------|
| `fp8_gemv_standalone.rs`         | 24-Harness | Isolated FP8 GEMV correctness + perf reproducer (per-channel scale) |
| `gemv_f16_shape_bench.rs`        | 27         | F16 GEMV shape sweep with VkQueryPool TIMESTAMP per dispatch |
| `cb_backpressure_test.rs`        | 28 Phase 1 | CB cardinality vs `lm_head` timing — backpressure hypothesis |
| `vram_pressure_test.rs`          | 28B        | VRAM-occupancy impact on bandwidth-bound shapes |
| `profile_decode_safetensors.rs`  | 26         | Per-dispatch decode profiling on SafeTensors FP8 models |
| `fp8_prefill_shape_bench.rs`     | 32 Phase 0 | Per-shape FP8 prefill TFLOPS diagnostic |

Older / supporting probes also live under `examples/`
(e.g. `bench_coopmat.rs`, `probe_coopmat_layout.rs`,
`profile_forward.rs`). Sprint provenance is in each file's header.

## ISA-level diagnostics

| Directory | Description |
|-----------|-------------|
| [`sprint34_load_tr/`](sprint34_load_tr/README.md) | GLSL test shaders + RGA ISA dumps proving ACO does **not** emit `GLOBAL_LOAD_TR_*` for `coopMatLoad` from SSBO on gfx1201. Useful for upstream Mesa/ACO RFEs. |

## FP8 cooperative-matrix shader variants (`vk_shaders/`)

| Shader | BM | BN | BK | Threads/WG | Status |
|--------|----|----|----|------------|--------|
| `mul_coopmat_fp8.comp`          | 16 | 16 | 16 | 64   | Legacy reference |
| `mul_coopmat_fp8_naive.comp`    | 16 | 16 | 16 | 64   | Legacy |
| `mul_coopmat_fp8_multi_wg.comp` | 64 | 16 | 16 | 256  | Legacy multi-WG |
| `mul_coopmat_fp8_bn32.comp`     | 64 | 32 | 16 | 512  | **Default (v0.3.7)** |
| `mul_coopmat_fp8_bn64.comp`     | 64 | 64 | 16 | 1024 | Opt-in via `VF_FP8_GEMM_BN=64` |

## Local-only artefacts (not in repo)

- `~/sprint29b_captures/*.rgp` — RGP/SQTT traces, ~3 GiB total
- `results/*.md` — per-sprint memos (gitignored under `results/`)
- `docs/` — design notes (gitignored)
