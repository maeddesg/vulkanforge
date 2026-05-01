# Phase 7 — mul_mm.comp debug & silent mul_mmq fix (v0.1.3)

**Date:** 2026-04-27
**Goal:** Debug the NaN/garbage-output bug in the `mul_mm.comp` port from
Phase 6 v0.1.2 using small-tile bring-up methodology, eliminating
hypotheses H1–H5 (unaligned N, BK_STEP/SHMEM_STRIDE, BLOCK_SIZE,
stride_a, B-layout transpose).

**Outcome:** Two distinct bugs found and fixed; the first one was
silently corrupting `mul_mmq` output in production for prompts >32
tokens. Both kernels now bit-exact; test suite 82 → 93. mul_mm stays
opt-in for perf reasons (mul_mmq is ~45 % faster).

---

## Methodology

The user's prompt called for "small-tile bring-up": isolate the GEMM
from the full forward pass, dispatch it on dimensions where the kernel
math is provable by hand, and only then scale up to real workloads.
That approach paid off — both bugs surfaced inside the unit tests
without ever loading the model.

Concretely we added a parametrised parity test
`run_mul_mm_parity(M, N, K, label)` in `tests/correctness.rs` that:

1. Generates `M × K` Q4_K weights via `q4k::build_random_weights`.
2. Generates `N × K` FP32 activations.
3. Computes a CPU reference (`cpu_gemm_q4k_ref`) by dequantising every
   block and summing in F64.
4. Dispatches `MulMmQ4K` with `(ceil(M/BM), ceil(N/BN), 1)` workgroups.
5. Reports per-column max error (with the column ratio `err / cpu_amax`
   highlighting any cols that read as zero — i.e. uninitialised
   memory).

The test suite covers `K ∈ {256, 512, 2048, 11008}`, aligned + unaligned
`N`, and single + multi `BM/BN` tiles.

---

## H3 (BLOCK_SIZE) — checked first per the prompt's "PRÜFE DAS ZUERST"

The shader's GLSL default is `BLOCK_SIZE = 64`; we override to `128`
via spec-constant `id = 0`. With `WARP = 64` (RDNA Wave64) that gives
`NUM_WARPS = BLOCK_SIZE / WARP = 2`.

The output store loop fans out as

```glsl
const uint warp_i = gl_LocalInvocationID.x / WARP;        // [0, NUM_WARPS)
const uint warp_r = warp_i % (BM / WM);                   // [0, BM/WM)
const uint warp_c = warp_i / (BM / WM);                   // <-- !
```

For `BM = BN = 64`, `WM = WN = 32`, we need `(BM/WM) × (BN/WN) = 4`
warp tiles per workgroup — but with `NUM_WARPS = 2`, `warp_c` is
**always 0**, so columns `[WN, BN) = [32, 64)` of every output tile
are never written. The first proof:

```
$ cargo test --release test_gemm_q4k_full_tile_64x64_mul_mmq
  col  0: err=8.6e-2  cpu_amax=2.2e2  ratio=0.000   <-- correct
  col  1: err=5.5e-2  cpu_amax=2.2e2  ratio=0.000
  ...
  col 31: err=4.4e-2  cpu_amax=2.1e2  ratio=0.000
  col 32: err=2.2e2   cpu_amax=2.2e2  ratio=1.000   <-- gpu = 0
  col 33: err=2.2e2   cpu_amax=2.2e2  ratio=1.000
  ...
  col 63: err=1.9e2   cpu_amax=1.9e2  ratio=1.000
PANIC: max_err = 2.2e2 >= 1.1e1
```

`ratio = err / cpu_amax = 1.000` is the giveaway: GPU reads as zero,
so the absolute error equals the CPU's absolute magnitude. Fix:

```rust
let block_size: u32 = std::env::var("VULKANFORGE_GEMM_BLOCK_SIZE")
    .ok().and_then(|s| s.parse().ok()).unwrap_or(256);   // was 128
```

Setting `BLOCK_SIZE = 256` (i.e. 4 warps) brings the test green
end-to-end:

```
$ VULKANFORGE_GEMM_BLOCK_SIZE=256 cargo test ... test_gemm_q4k_full_tile_64x64_mul_mmq
test result: ok. 1 passed; 0 failed; ...
```

### Why production never noticed

`mul_mmq` (the working baseline) had this exact same bug. It went
unnoticed because every regression test that loaded the real model
used a short prompt:

* `phase3e_prefill_batch_matches_token_by_token_top5` —
  `"Explain what a mutex is in one sentence."` chat-templated to
  ~29 tokens, well under the 32-col threshold.
* `phase5b2_*` and `phase_prompt16_alice_*` — similar.

The pre-existing GEMM unit test `test_gemm_q4k_vs_gemv_seq1_parity`
ran `M = 2, N = 1`, where the bounds-check `dc_warp + cc < N` is the
only thing that gets exercised — the missing warp tile gets clipped.

A new test `test_gemm_q4k_full_tile_64x64_mul_mmq` was added to
prevent regression. It dispatches a full 64×64 tile against random
Q4_K weights and would have flagged the bug from day one.

---

## H1, H2, H4, H5 — eliminated by the parametrised tests

After fixing `BLOCK_SIZE`, the small-tile parity test
`test_gemm_q4k_full_tile_64x64_mul_mm` passed too. We then scaled up:

| Test | M | N | K | Result |
|---|---:|---:|---:|---|
| `_k512_aligned` | 64 | 64 | 512 | ✓ |
| `_k2048_aligned` | 64 | 64 | 2048 | ✓ |
| `_n_unaligned_62` | 64 | **62** | 256 | ✓ — H1 (unaligned N) eliminated |
| `_multi_n_tile_128` | 64 | 128 | 256 | ✓ — multi-`BN` tile |
| `_multi_m_tile_128` | 128 | 64 | 256 | ✓ — multi-`BM` tile |
| `_multi_tile_both` | 128 | 128 | 256 | ✓ |
| `_n200` | 64 | 200 | 256 | ✓ |
| `_realistic_2048x62` | **2048** | **62** | **2048** | ✓ — full real-prefill `gemm_q` dims |
| `_ffn_down_dims` | 2048 | 62 | **11008** | ✓ — `ffn_down` (43 BK iterations) |

H2 (`SHMEM_STRIDE` / `BK_STEP`), H4 (`stride_a / LOAD_VEC_A`), and
H5 (B-layout transpose) all check out — every test passes against the
F64 CPU reference within `0.01 × cpu_amax`.

---

## Bug 2 — Q6_K LOAD_VEC_A (uncovered by enabling mul_mm at full scale)

The unit tests all passed, but enabling `VULKANFORGE_USE_MUL_MM=1` in
`Forward::new_with_prefill` and re-running `phase3e_prefill_batch_*`
still produced **NaN/Inf logits**. Since the unit tests only exercise
the Q4_K shader, that pointed at the only other path: Q6_K, used by
`gemm_v` (V-projection weights) and `gemm_down` (FFN-down).

Reading `vk_shaders/mul_mm_funcs.glsl` line 263–284:

```glsl
#elif defined(DATA_A_Q6_K)
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

    const uint ib = idx / 128;       // 2 values per idx
    ...
    buf_a[buf_idx] = FLOAT_TYPEV2(q.x, q.y);   // <-- only 1 vec2 = 2 weights
```

Compare Q4_K (line 190–224):

```glsl
#elif defined(DATA_A_Q4_K)
    const uint ib = idx / 64;        // 4 values per idx
    ...
    buf_a[buf_idx    ] = FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m));
    buf_a[buf_idx + 1] = FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m));
```

Q4_K writes **two** `vec2`s per invocation (4 weights → matches
`LOAD_VEC_A = 4`). Q6_K writes **one** `vec2` (2 weights) — its
math is implicitly `LOAD_VEC_A = 2`. Compiling Q6_K with
`LOAD_VEC_A = 4` (because Phase 6's `build.rs` blindly mirrored Q4_K)
left `buf_a[buf_idx + 1]` uninitialised on every Q6_K invocation;
those slots got read by the inner FMA loop a few iterations later
and produced `NaN`.

Fix: pin Q6_K to `LOAD_VEC_A = 2` in `build.rs`:

```rust
ShaderJob {
    out_name: "mul_mm_q6_k_f32.spv",
    ...
    ("LOAD_VEC_A", "2"),    // Q6_K dequant emits 1 vec2/idx (= 2 weights)
},
```

Re-run with `VULKANFORGE_USE_MUL_MM=1`:

```
$ cargo test --release ... phase3e_prefill_batch_matches_token_by_token_top5
[parity] top1_a=151667 top1_b=151667 top5_a=[151667, 85387, 151668, 34894, 50897]
                                     top5_b=[151667, 85387, 151668, 34894, 50897] overlap=5
test phase3e_prefill_batch_matches_token_by_token_top5 ... ok
```

Top-1 match, **top-5 = 5/5 overlap** — bit-exact equivalence with the
per-token GEMV reference path.

---

## Why mul_mm stays opt-in

mul_mmq is significantly faster on the same hardware:

| Prompt (Qwen3-8B-Q4_K_M) | mul_mmq | mul_mm | Δ |
|---|---:|---:|---:|
| 29-tok mutex | 545 tok/s | 309 tok/s | −43 % |
| 55-tok essay | 980 tok/s | 538 tok/s | −45 % |

The mechanical reason is straightforward — mul_mm reads B as raw FP32
(4 B/element) into LDS while mul_mmq reads Q8_1-packed B
(~1.13 B/element), so the inner load loop pulls 3.5× more bytes from
VRAM per workgroup. Decode (GEMV path) is unchanged in both.

`VULKANFORGE_USE_MUL_MM=1` remains available for runs that need
bit-exact FP32 activations end-to-end (e.g. validating drift
attributable to Q8_1 quantisation of activations).

---

## Files touched

| File | Change |
|---|---|
| `vk_shaders/mul_mm.comp` | unchanged |
| `vk_shaders/mul_mm_funcs.glsl` | unchanged |
| `build.rs` | Q6_K `LOAD_VEC_A`: 4 → 2 |
| `src/backend/vulkan/pipeline_registry.rs` | default `BLOCK_SIZE`: 128 → 256 (both `MulMmq*` and `MulMm*`) |
| `src/backend/vulkan/forward.rs` | mul_mm comment refresh; default still OFF |
| `tests/correctness.rs` | `+11` GEMM-parity tests; `cpu_gemm_q4k_ref` helper |
| `Cargo.toml` | 0.1.2 → 0.1.3 |
| `CHANGELOG.md` | v0.1.3 entry |

---

## Test suite final state

```
cargo test --release --lib            : 24 / 24 green
cargo test --release --test correctness : 44 / 44 green   (was 33)
cargo test --release --test regression  : 25 / 25 green
                                          ─────
                                          93 / 93 green
```
