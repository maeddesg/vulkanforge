# v0.3.2 Sprint 17B follow-up — Q3_K debug session

**Result:** **root cause identified**. Q3_K_M GGUFs ship `attn_v` and
`ffn_down` as **Q5_K**, not Q4_K — Sprint 17B's brief was wrong. We
have no Q5_K shader, so those weights silently fall through to
`MulMmqQ4K` / `MulMatVecQ4K` and read 144-byte blocks out of a
176-byte buffer → garbage compute → `!!!` output.

The Q3_K shader path itself is **bit-exactly correct vs CPU** for
both decode (GEMV) and prefill (Mmq) at every shape we tested,
including the realistic `M=4096, N=32, K=4096` chat prefill.

**Action shipped:** revert Q3_K_M from `inference_support` (one-liner,
`Some(12 | 15)` → `Some(15)`). The Q3_K shader / Mmq / CPU dequant
infrastructure stays in `main` and is ready for re-activation once
Q5_K joins it. Also fixed a latent bug in the L-tile dispatcher
that was hiding behind the bigger Q5_K issue.

## Hypothesis tracker

| #  | Hypothesis                                  | Verdict |
|----|---------------------------------------------|---------|
| 1  | CPU `q3k::dequant_block` bug                | ruled out |
| 2  | Mmq Q3_K build-defines drift                | ruled out |
| 3a | GPU `MulMatVecQ3K` (decode GEMV)            | ruled out (bit-exact) |
| 3b | GPU `MulMmqQ3K{,L}` (prefill compute)       | ruled out (bit-exact at every shape) |
| 4  | **Mixed-quant Q3_K_M ships Q5_K weights**   | **ROOT CAUSE** |

## What we knew before this session

Sprint 17B brief assumed Q3_K_M = "bulk Q3_K + sprinkling of Q4_K +
Q6_K `output.weight`". Built shader/dispatch/CPU-dequant
infrastructure for Q3_K. Verified output is `!!!`. Three speculative
hypotheses left unverified.

## What this session learnt

### Q3_K is correct end-to-end

`tests/q3k_dequant_sanity.rs` (CPU): 3 tests
- `q3k_dequant_block_is_finite_and_nonzero` — first 8 blocks of
  `token_embd.weight` produce sane values (max|x| ≈ 0.15, no NaN,
  ~190/256 nonzero).
- `q3k_vs_q4k_embedding_rms_within_bound` — Q3_K vs Q4_K row 0:
  RMS(Q3) = 0.0292, RMS(Q4) = 0.0290, RMS(diff) = 0.0048 (16 % rel —
  exactly the lossy-requant noise floor).
- `dump_q3k_m_layer0_quant_types` — *this is the test that surfaced
  the real bug.* See below.

`tests/q3k_gemv_correctness.rs` (GPU): 2 tests
- `q3k_gemv_stock_matches_cpu`: bit-exact match for M=4 K=4096.
- `q3k_gemv_subgroup_matches_cpu`: bit-exact match for M=4 K=4096.

`tests/correctness.rs::test_gemm_q3k_*` (GPU Mmq prefill): 3 tests
- `64x64 K=256` (1 block/row): cpu_amax 0.54 vs gpu_amax 0.54 ✓
- `64x64 K=4096` (16 blocks/row): cpu_amax 1.70 vs gpu_amax 1.70 ✓
- `realistic M=4096 N=32 K=4096`: cpu_amax 2.55 vs gpu_amax 2.55 ✓

GPU output matches CPU within Q8_1 round-off at every dimension we
tested. The Q3_K compute path is correct.

### The actual bug — Q3_K_M is mixed-quant **Q5_K**, not Q4_K

`dump_q3k_m_layer0_quant_types` dumps every weight in layer 0:

```
=== Q3_K_M Qwen3-8B layer 0 weight types ===
  blk.0.attn_q.weight     → Q3K dims=[4096, 4096]
  blk.0.attn_k.weight     → Q3K dims=[4096, 1024]
  blk.0.attn_v.weight     → Q5K dims=[4096, 1024]    ← !!!
  blk.0.attn_output.weight → Q4K dims=[4096, 4096]
  blk.0.ffn_gate.weight   → Q3K dims=[4096, 12288]
  blk.0.ffn_up.weight     → Q3K dims=[4096, 12288]
  blk.0.ffn_down.weight   → Q5K dims=[12288, 4096]   ← !!!
=== Top-level tensors ===
  token_embd.weight       → Q3K dims=[4096, 151936]
  output.weight           → Q6K dims=[4096, 151936]
```

`attn_v.weight` and `ffn_down.weight` are **Q5_K** in every layer.
Llama.cpp's `Q3_K_M` recipe (FTYPE 18 → file_type 12) is in fact
*Q3_K bulk + Q5_K for V/down + Q6_K for output*, not what the brief
stated.

### How Q5_K reaches the wrong shader

`forward.rs::layer_weight_shader_gemm` (Mmq prefill):

```rust
match (gemm_kind, q6) {
    (GemmKind::Mmq, true)  => Q6K shader,
    (GemmKind::Mmq, false) if q3 => Q3K shader,
    (GemmKind::Mmq, false) => Q4K shader,        // ← Q5K lands here
}
```

`forward.rs::layer_weight_shader` (decode GEMV):

```rust
match (ggml_type, subgroup) {
    (Q6K, _) => Q6K GEMV,
    (Q3K, _) => Q3K GEMV,
    (_,   _) => Q4K GEMV,                         // ← Q5K lands here
}
```

Q5_K block layout: 176 B (`f16vec2 dm + scales[12] + qh[32] + qs[128]`).
Q4_K block layout: 144 B (`f16vec2 dm + scales[12] + qs[128]`).

When MulMmqQ4K / MulMatVecQ4K dispatches against a Q5_K-stride buffer,
`data_a_packed*[ib_k]` reads the Q4_K-shaped 144-byte struct out of
buffer rows where each block is actually 176 B. After block 0,
every subsequent read straddles the previous block's tail and
the next block's head → progressively more wrong → final logits
collapse to a constant (the `!` token argmax).

This explains every observation:
- `decode 122 tok/s` — bandwidth-bound shader runs at full speed
  reading the right *amount* of data even though it's the wrong
  shape (reads still hit the buffer, just at wrong strides).
- Output is identical garbage on every prompt — the layer-0
  KV-cache state is deterministically corrupted, so every
  downstream forward pass produces the same uniform-logit pattern.
- Q4_K_M routed through the same Mmq path works fine — it actually
  contains Q4_K weights end-to-end.

### Bonus: latent dispatcher bug

While searching, found that `forward.rs::run_gemm` was dispatching
`MulMmqQ3KL` with `bm/bn=64,64` even though `pipeline_registry` pins
its spec constants to BM=BN=128. The L-tile match arm listed Q4KL
and Q6KL but skipped Q3KL — addded it (`forward.rs:3086`).

This bug only fires when `prefer_l == true` (M>128 and N>256), so
it doesn't fire on the short prompts in normal chat — but it would
have produced silent corruption on long-prompt prefill once the
Q5_K issue was fixed. Worth keeping shipped.

## Files changed

- `src/main.rs` — Q3_K_M (file_type=12) removed from preflight
  whitelist with a comment explaining the Q5_K dependency.
- `src/backend/vulkan/forward.rs` — `MulMmqQ3KL` added to
  L-tile bm/bn match arm; comment in `layer_weight_shader`
  updated to flag the Q5_K silent fall-through.
- `tests/q3k_dequant_sanity.rs` — new (3 tests, CPU dequant
  + Q5_K detection dump).
- `tests/q3k_gemv_correctness.rs` — new (2 tests, GPU GEMV
  bit-exact vs CPU).
- `tests/correctness.rs` — `+3 tests` for `MulMmqQ3K` Mmq parity
  (64×64×256, 64×64×4096, realistic 4096×32×4096).

All 8 new tests pass + 27 / 27 lib tests + Q4_K_M chat coherent.

## Status of Q3_K_M support

The shader / Mmq / CPU-dequant infrastructure for Q3_K is in `main`
and verified correct. Re-activating Q3_K_M needs:

1. **Q5_K shader sprint** (similar effort to 17B):
   - `mul_mat_vec_q5_k.comp` (decode GEMV)
   - `mul_mmq_q5_k_f32.spv` build entry (mul_mmq.comp + DATA_A_Q5_K)
   - `block_q5_K` / `block_q5_K_packed16` already in `types.glsl`
     (lines 333-371) — no shader-source change needed there
   - `q5k.rs` — port `dequant_row_q5_K` from ggml-quants.c (~80 LOC)
   - `ShaderId::{MulMatVecQ5K, MulMatVecQ5KSubgroup, MulMmqQ5K, MulMmqQ5KL}`
   - Wire into `forward.rs::layer_weight_shader{,_gemm}` and
     `decode.rs::embedding_row` (Q5_K never appears as token_embd
     in standard Q3_K_M, but adding the path is cheap).
   - Re-enable file_type=12 in `inference_support`.

2. **Validate**: a `dump_q3k_m_layer0_quant_types`-style sanity test
   for the Q5_K_S / Q5_K_M file_types too, in case those land in
   future user models.
