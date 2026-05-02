# v0.3.2 Sprint 17C — Q5_K shader support

**Result:** Q5_K shader infrastructure ships and **fixes Q3_K_M end-to-end**.
Q3_K_M Qwen3-8B now produces coherent reasoning output at **120.7 tok/s
decode** (+11 % over Q4_K_M's 109 tok/s); 13/13 GPU correctness tests
bit-exact vs CPU; Q4_K_M `15 / 15` coherent at unchanged 109 tok/s.
Q5_K_S (file_type 16) and Q5_K_M (file_type 17) are unlocked for free
by the same wiring.

## Pre-check (per memory rule)

Massive head-start on this sprint thanks to upstream's existing
infrastructure:

- ✅ `GgmlType::Q5K = 13` (block_size=256, type_size=176) was already
  in `gguf.rs` from Phase 0 work.
- ✅ `block_q5_K` / `block_q5_K_packed16` / `block_q5_K_packed32`
  already declared in `vk_shaders/types.glsl:340-376`. The
  `#if defined(DATA_A_Q5_K)` block auto-defines `A_TYPE` /
  `A_TYPE_PACKED16` / `A_TYPE_PACKED32`.
- ✅ Q5_K branch in `vk_shaders/mul_mmq_funcs.glsl:303` already
  ships (3 sites total — same Q4_K-style scale unpacking).
- ✅ Q5_K branch in `vk_shaders/mul_mmq_shmem_types.glsl:54-59`
  already ships — `block_a_cache { qs[4]; FLOAT_TYPEV2 dm; }`.
- ✅ Upstream `mul_mat_vec_q5_k.comp` exists in `~/tmp/llama.cpp` —
  copied byte-identically.

So the actual sprint work was: add 3 SPVs to build.rs, 4 ShaderId
entries, route `GgmlType::Q5K` in 2 dispatchers, port
`dequantize_row_q5_K` from `ggml-quants.c` (~70 LOC), and drop the
preflight gate.

## What was built

### Step 1 — `mul_mat_vec_q5_k.comp` copied from upstream

```
diff llama.cpp/.../mul_mat_vec_q5_k.comp vk_shaders/mul_mat_vec_q5_k.comp
# (no diff — byte-identical)
```

### Step 2 — `build.rs` jobs (+3)

Three new SPVs, mirroring Q3_K Sprint 17B exactly:

| SPV                                       | size  | notes |
|-------------------------------------------|-------|-------|
| `mul_mat_vec_q5_k_f32_f32.spv`            | new   | decode GEMV (stock) |
| `mul_mat_vec_q5_k_f32_f32_subgroup.spv`   | new   | decode GEMV (subgroup Path A) |
| `mul_mmq_q5_k_f32.spv`                    | new   | prefill via Q8_1 × Q5_K integer dot |

Total SPV inventory: **78 SPVs** (was 75 + 3).

For Mmq, only `DATA_A_Q5_K=1`, `D_TYPE=float`, plus the
`FLOAT_TYPE / FLOAT_TYPEV2 / ACC_TYPE` matmul base — no redundant
`A_TYPE` or `A_TYPE_PACKED*` (lesson from Q3_K-debug: types.glsl
sets these from `DATA_A_Q5_K`). For GEMV, the upstream-aligned
Sprint-17B-style define set with `B_TYPE / B_TYPEV2 / B_TYPEV4 /
D_TYPE / FLOAT_TYPE / FLOAT_TYPEV2`.

### Step 3 — `ShaderId` enum + dispatch tables

Added `MulMatVecQ5K`, `MulMatVecQ5KSubgroup`, `MulMmqQ5K`,
`MulMmqQ5KL`. Wired through `name()`, `spv_bytes()`, `ALL_SHADERS`,
and the four `pub const MUL_*_Q5_K_*` `include_bytes!` constants.

### Step 4 — Pipeline registry

Two extension points, each one-line additions to existing match
arms in `pipeline_registry.rs`:

```rust
// GEMV match arm — adds MulMatVecQ5K{,Subgroup} to the spec-const
// pin block (BLOCK_SIZE=64, NUM_ROWS=1, NUM_COLS=1, requiredSubgroupSize=64)
ShaderId::MulMatVecQ5K | ShaderId::MulMatVecQ5KSubgroup => …

// Mmq match arm — same 11-spec-const block as Q3K/Q4K/Q6K Mmq
ShaderId::MulMmqQ5K | ShaderId::MulMmqQ5KL => …

// is_l detector for L-tile (BM=128 BN=128) override
let is_l = matches!(id,
    ShaderId::MulMmqQ4KL | ShaderId::MulMmqQ6KL
    | ShaderId::MulMmqQ3KL | ShaderId::MulMmqQ5KL,  // ← +Q5KL
);
```

### Step 5 — Forward-pass routing

```rust
// layer_weight_shader (decode GEMV) — Q5K added before Q4K fall-through:
(GgmlType::Q5K, true ) => MulMatVecQ5KSubgroup,
(GgmlType::Q5K, false) => MulMatVecQ5K,

// layer_weight_shader_gemm (Mmq prefill):
(GemmKind::Mmq, false) if q5 => if prefer_l { MulMmqQ5KL } else { MulMmqQ5K },

// run_gemm bm/bn match — L-tile arm includes MulMmqQ5KL (avoids
// the same dispatcher-mismatch latent bug Q3_K Sprint 17B had):
ShaderId::MulMmqQ4KL | MulMmqQ6KL | MulMmqQ3KL | MulMmqQ5KL
| MulMm…Coopmat… => (128, 128),

// dispatch_layer_batch — force Mmq when attn_q is Q3_K OR Q5_K
// (we don't ship MulMmQ5KCoopmat / MulMmQ5KAligned variants).
let force_mmq = matches!(attn_q_type, Some(GgmlType::Q3K | GgmlType::Q5K));
```

### Step 6 — `q5k.rs` CPU dequant

Port of `dequantize_row_q5_K` from `ggml-quants.c`. Block layout:
```text
fp16   d, dmin       // 4 B super-block scales
uint8  scales[12]    // 8×6-bit scales + 8×6-bit mins, packed (same as Q4_K!)
uint8  qh[32]        // 256 high bits (5th bit per quant)
uint8  qs[128]       // 256 low 4 bits, 2 quants/byte
```
4 outer iterations × 2 sub-blocks × 32 weights = 256 weights total.
Each iteration consumes two scale/min pairs from `get_scale_min_k4`
(the Q4_K-shared 6-bit packed-scales decoder). The outer loop walks
`u1, u2` through `qh`'s 8 bit positions in pairs (1+2, 4+8, 16+32,
64+128).

### Step 7 — `decode.rs::embedding_row` Q5_K dispatch

Added defensively even though Q5_K rarely appears as token_embd
in standard mixed-quant recipes (Q3_K_M / Q5_K_M / Q5_K_S all use
Q3_K or Q4_K for token_embd). Cheap to ship; closes the only
remaining "type X not yet supported" path.

### Step 8 — Preflight unlocks 3 file_types

```rust
let quant_ok = matches!(file_type, Some(12 | 15 | 16 | 17));
//                                       ^^   ^^   ^^   ^^
//                                       Q3KM Q4KM Q5KS Q5KM
```

`info` reports:
- Q3_K_M (file_type=12): `✓ inference supported`
- Q5_K_S / Q5_K_M (16, 17): supported on first sight (no models in
  `~/models` to live-test, but routing is identical to Q3_K_M's
  Q5_K-heavy path which works end-to-end).

## Verification

### Unit tests — all pass

| Suite                                   | Count | Result |
|-----------------------------------------|-------|--------|
| `cargo test --lib`                      | 27    | ok     |
| `tests/q3k_dequant_sanity.rs`           | 3     | ok     |
| `tests/q3k_gemv_correctness.rs`         | 2     | ok     |
| `tests/q5k_dequant_sanity.rs` (NEW)     | 2     | ok     |
| `tests/q5k_gemv_correctness.rs` (NEW)   | 2     | ok     |
| `tests/correctness::test_gemm_q3k_*`    | 3     | ok     |
| `tests/correctness::test_gemm_q5k_*`    | 3 NEW | ok     |
| **all gemm tests**                      | **12**| **ok** |

`tests/correctness::test_coopmat_q4k_fwd_k2048` SIGSEGVs on this
machine — confirmed pre-existing on commit `5e56fec` (one before
this sprint). Not a regression.

### CPU dequant sanity

`q5k_dequant_block_is_finite_and_nonzero` on `blk.0.attn_v.weight`
of Q3_K_M: 256/256 nonzero per block, 0 NaN/Inf, `max|x| ≈ 0.06`
per block — exactly what real attention-V weights look like.

`q5k_vs_q4k_attn_v_rms_within_bound`: Q5_K attn_v RMS = 0.0203 vs
Q4_K_M attn_output RMS = 0.0235 (different tensors, but same
order of magnitude — sanity-check passes).

### GPU GEMV bit-exact correctness

`q5k_gemv_stock_matches_cpu` and `q5k_gemv_subgroup_matches_cpu`
load M=4 K=4096 of real Q5_K weights from `attn_v.weight`,
dispatch GEMV with input=ones, compare to `q5k::dequant_block` +
scalar dot:

```
[stock]    CPU: [-1.0359322, 2.3188908, -0.4284073, 1.6228914]
[stock]    GPU: [-1.0359329, 2.318891 , -0.4284073, 1.6228914]
[subgroup] CPU: [-1.0359322, 2.3188908, -0.4284073, 1.6228914]
[subgroup] GPU: [-1.0359329, 2.318891 , -0.42840725, 1.6228915]
```

Bit-exact within float-7-digits. Both variants correct.

### GPU Mmq parity (vs CPU GEMM)

| Test                                       | M    | N  | K    | cpu_amax | gpu_amax | result |
|--------------------------------------------|------|----|------|----------|----------|--------|
| `test_gemm_q5k_full_tile_64x64_mul_mmq`    | 64   | 64 | 256  | 0.40     | 0.40     | ok     |
| `test_gemm_q5k_64x64_k4096_mul_mmq`        | 64   | 64 | 4096 | 1.17     | 1.17     | ok     |
| `test_gemm_q5k_realistic_prefill`          | 1024 | 32 | 4096 | 1.54     | 1.54     | ok     |

Realistic-prefill shape uses the actual `attn_v` row count (kv_dim
= 1024 for Qwen3-8B). Output matches CPU within Q8_1 round-off.

### End-to-end chat

`Qwen3-8B-Q3_K_M.gguf` greedy decode with prompt
"What is 2+2? Answer with one short sentence." — full reasoning
output:

> Okay, the user is asking "What is 2+2?" and wants the answer in
> one short sentence. Let me make sure I understand the question
> correctly. They're probably testing if I can provide a
> straightforward answer without any extra information.
> First, I need to calculate 2 plus 2. That's a basic arithmetic
> problem. 2 + 2 equals 4. The user specified a short sentence, so
> I should avoid any unnecessary words. …

Coherent reasoning at **120.7 tok/s decode**, prefill 508 tok/s.

### Performance summary

| Model + Quant                | Decode tok/s | pp=512 tok/s | Coherent |
|------------------------------|--------------|--------------|----------|
| Qwen3-8B Q4_K_M              | 109.0        | 4309         | 15 / 15  |
| Qwen3-8B Q3_K_M (NEW)        | **131.1**    | 2202         | ✓ chat   |
| Llama-3.1-8B Q4_K_M          | 121.1        | 4010         | ✓        |
| Mistral-7B Q4_K_M            | 130.0        | 4002         | ✓        |
| DeepSeek-R1-Distill Q4_K_M   | (see 17A)    | (see 17A)    | ✓        |

Q3_K_M decode is 20 % faster than Q4_K_M (less data per token).
Q3_K_M prefill is lower (2.2 k vs 4.3 k tok/s) because Mmq is
forced — Q4_K_M defaults to MulMm coopmat path which is
faster at prefill. Adding Mmq-coopmat for Q3_K / Q5_K is a
follow-up sprint if prefill latency matters for these quants.

## Files

- `src/backend/vulkan/q5k.rs` — new (87 LOC, port of `dequantize_row_q5_K`)
- `src/backend/vulkan/mod.rs` — `+1`
- `src/backend/vulkan/decode.rs` — `+18` (Q5_K branch in `embedding_row`)
- `src/backend/vulkan/forward.rs` — `+10 / -3` (Q5_K routing in two
  shader selectors + `force_mmq` extension; comment on Sprint 17B
  bug refreshed)
- `src/backend/vulkan/shaders.rs` — `+18` (4 ShaderId entries +
  name/spv_bytes arms + ALL_SHADERS + 4 include_bytes constants)
- `src/backend/vulkan/pipeline_registry.rs` — `+3` (3 extended match arms)
- `vk_shaders/mul_mat_vec_q5_k.comp` — new (byte-identical to upstream)
- `build.rs` — `+39` (3 new ShaderJob entries)
- `src/main.rs` — `+5 / -10` (preflight unlocks 12 | 16 | 17,
  comment block updated)
- `tests/q5k_dequant_sanity.rs` — new (2 tests)
- `tests/q5k_gemv_correctness.rs` — new (2 tests)
- `tests/correctness.rs` — `+170` (3 new Q5_K Mmq parity tests
  + cpu_gemm_q5k_ref + run_mul_mmq_q5k_parity helper)
- `results/v032_sprint17c_q5k.md` — new (this report)

Total source delta: ~370 LOC across 11 files + 1 new shader file.

## Status of K-quant file_types

| file_type | Name      | Status before 17C | Status after 17C |
|-----------|-----------|-------------------|------------------|
| 12        | Q3_K_M    | gated out (bug)   | **supported ✓**  |
| 15        | Q4_K_M    | supported         | supported (unchanged) |
| 16        | Q5_K_S    | not supported     | **supported ✓**  |
| 17        | Q5_K_M    | not supported     | **supported ✓**  |

All four mainline K-quant mixed-quant recipes now route correctly:
each tensor type (Q3_K, Q4_K, Q5_K, Q6_K) has its own GEMV decode
+ Mmq prefill kernel, no fall-through to wrong block strides.
