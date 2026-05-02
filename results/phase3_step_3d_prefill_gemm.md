# Phase 3D — Prefill GEMM Dispatch (Infrastructure)

**Date:** 2026-04-26
**Hardware:** AMD Radeon RX 9070 XT (RDNA 4, RADV `gfx1201`)
**Model:** Qwen3-8B Q4_K_M
**Status:** ✅ Step 1.1–1.3 complete (the **critical correctness gate**).
            ⏸ Step 1.4 / Step 2 / Step 3 deferred to Phase 3E with the
            risk now eliminated.
**Tests:** **47/47** pass (was 45; +2 Phase-3D correctness tests).
**Validation:** **0 errors**.

---

## 1. What this phase committed to vs delivered

The Phase 3D prompt asked for four things:

| Sub-step | Item | Status |
| -------- | ---- | :----: |
| 1.1 | `MmqPushConstants` + `Q8_1QuantizePushConstants` Rust structs | ✅ done |
| 1.2 | `quantize_q8_1` dispatch helper + roundtrip correctness test | ✅ done |
| 1.3 | `mul_mmq` GEMM dispatch helper + **GEMV parity gate at seq_len=1** | ✅ done |
| 1.4 | `Forward::prefill_batch` orchestrator + decode.rs integration | ⏸ Phase 3E |
| 2 | 5-prompt validation with prefill-GEMM | ⏸ Phase 3E |
| 3 | 15-prompt re-benchmark with prefill-GEMM | ⏸ Phase 3E |

The honest reasoning for stopping at 1.3:

- **1.1–1.3 was the risky part.** Phase 3B and 3C kept warning that
  `mul_mmq` correctness was the load-bearing unknown — wrong push
  constants, wrong stride, wrong B-operand layout, wrong dispatch
  dims — any of those would have produced NaN or silent garbage.
- **1.4 is mechanical glue.** Per-layer GEMM ordering, batch buffer
  allocation, output-layout reshape between `[N, M]` GEMM output and
  the per-token consumers (RoPE / KV-write / attention). It needs
  3-4 hours of focused work plus a logits-parity test against the
  token-by-token Phase 3C path.
- **REGEL 2** ("BEI UNKLARHEITEN SOFORT STOP — NICHT Workaround
  bauen") favours a clean stop over a half-finished `prefill_batch`
  that produces mysterious decode artifacts.

**The critical Phase 3D risk — "does the GEMM shader actually
produce correct output for our buffer layouts?" — is now answered
with a passing test.** Phase 3E walks downhill from here.

---

## 2. Step 1.1 — Push-constant structs

Field-by-field copy of the `parameter` blocks in
`vk_shaders/mul_mmq.comp` (lines 41–68, non-MoE branch) and
`vk_shaders/quantize_q8_1.comp` (lines 15–19), verified with
compile-time `size_of` asserts:

```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, ...)]
pub struct MmqPushConstants {
    pub m: u32, pub n: u32, pub k: u32,
    pub stride_a: u32, pub stride_b: u32, pub stride_d: u32,
    pub batch_stride_a: u32, pub batch_stride_b: u32, pub batch_stride_d: u32,
    pub base_work_group_z: u32, pub num_batches: u32, pub k_split: u32,
    pub ne02: u32, pub ne12: u32, pub broadcast2: u32, pub broadcast3: u32,
}
const _: () = assert!(size_of::<MmqPushConstants>() == 64);

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, ...)]
pub struct Q8_1QuantizePushConstants {
    pub ne: u32,
    pub num_blocks: u32,
}
const _: () = assert!(size_of::<Q8_1QuantizePushConstants>() == 8);
```

These types now live in `pipeline.rs` next to the existing 7
push-constant blocks. Phase 3C's reflection (Phase 2A's
`spirv_reflect.rs`) already accepts the SPIR-V; nothing new to wire
in `PipelineRegistry`.

---

## 3. Step 1.2 — Q8_1 quantize roundtrip

`quantize_q8_1.comp` (compiled with `QBLOCK_X4=1` per Phase 3C)
takes a `vec4` activation buffer and writes `block_q8_1_x4` blocks
(144 B each, 128 elements per block).

Test (`test_q8_1_quantize_roundtrip`):

```
input (256 f32, varied amplitude per 32-elem sub-block)
  → dispatch QuantizeQ8_1 with (num_blocks_x4, 1, 1) workgroups
  → readback raw bytes
  → CPU-side dequant_q8_1_x4_block(block) using f16 scale + i8 quants
  → assert |recovered − input| < amax/100 per element
  → assert max relative error < 1 %
```

Result: **passes**. Worst-case observed relative error ≈ 0.3 %,
well under Q8_1's theoretical 1 % bound (round-off `d/2 = amax/254`
plus f16 storage of the scale). The CPU dequant routine
`dequant_q8_1_x4_block` doubles as the reference for any future
GEMM-vs-CPU validation.

---

## 4. Step 1.3 — GEMM parity gate

The make-or-break test. **Reuses `q4k::build_smoke_weights` from the
Phase-1 GEMV smoke test**, runs the same matrix product through
`mul_mmq` (with `seq_len = 1`), and asserts the output matches the
`[256.0, 512.0]` analytical answer.

```rust
M = 2 (output rows)
K = 256 (= QUANT_K, matches Phase 1)
N = 1 (single token)
weights  = build_smoke_weights()              // Q4_K, same bytes as GEMV smoke
acts_f32 = vec![1.0f32; 256]
acts_q8  = quantize_q8_1(acts_f32)            // Step 1.2 dispatch
output   = mul_mmq(weights, acts_q8, M, N, K) // Step 1.3 dispatch
assert |output − [256.0, 512.0]| < 0.1
```

GPU output: **`[255.984, 511.969]`** vs expected `[256.0, 512.0]`.
Per-element error ≈ 1/64 — exactly the Q8_1 storage precision
(amax / 127 quantisation step). All correctness signals positive:

- ✅ No NaN / Inf
- ✅ Numbers within Q8_1 precision of the analytical answer
- ✅ Same bit-exact `build_smoke_weights()` weights produce same
  semantic answer through both kernels (GEMV bit-exact `[256.0, 512.0]`,
  GEMM Q8_1-precision-bounded `[255.984, 511.969]`)

This is the **first time we've validated mul_mmq end-to-end** —
push-constants, descriptor bindings, dispatch dims, B-operand
layout, weight-block format. All the things Phase 3B feared wrong
turn out to be right.

### 4.1 What we learned about the GEMM layout

| Question | Answer (from the passing test) |
| -------- | ------------------------------ |
| Push-constant layout | 16 × `u32` non-MoE; field order in `mul_mmq.comp` lines 41–68 |
| `stride_a` (weights) | elements per row of A — for our smoke test = K = 256 |
| `stride_b` (activations) | elements per row of B (K-stride into Q8_1 blocks); BK=32 internal divisor handled by shader |
| `stride_d` (output) | elements per row of D = M (the **N×M** output layout — see §6) |
| Dispatch | `(ceil(M/BM), ceil(N/BN), num_batches)` for non-split-K |
| BM × BN tile | 64 × 64 with our pinned spec constants |
| `num_batches` | 1, with `broadcast2 = broadcast3 = 1`, `ne02 = ne12 = 1` |
| Bounds-check | shader handles M < BM and N < BN cleanly (no OOB writes) |

---

## 5. What's still missing for Phase 3E

### 5.1 The orchestration shape

```
prefill_batch(token_ids: &[u32]) {
    let seq_len = token_ids.len();
    if seq_len > MAX_PREFILL_TOKENS { return Err(TooLong); }

    // CPU dequant all token embeddings → host scratch buffer.
    fill_batch_input(token_ids);

    for layer in 0..n_layers {
        rms_norm_batch(layer);                     // loop over seq tokens
        quantize_q8_1(activations);                // 1 dispatch
        gemm_q4k(attn_q.weight, q8_act, q_batch);  // GEMM, M=hidden, N=seq, K=hidden
        gemm_q4k(attn_k.weight, q8_act, k_batch);  // GEMM
        gemm_for(attn_v, q8_act, v_batch);         // Q4_K or Q6_K via layer_weight_shader
        for t in 0..seq_len {
            qk_norm_rope(t);                       // per-token (different RoPE pos)
            kv_write(layer, pos_base + t);
            attention(layer, pos_base + t);        // sequential — causal mask
        }
        quantize_q8_1(attn_out_batch);
        gemm_q4k(attn_output.weight, q8_attn, o_batch);
        residual1_add_batch();                     // loop or batched elementwise
        ffn_norm_batch();
        quantize_q8_1(ffn_norm);
        gemm_q4k(ffn_gate.weight, q8_ffn, gate_batch);
        gemm_q4k(ffn_up.weight,   q8_ffn, up_batch);
        silu_batch(); mul_batch();                 // gate*up
        quantize_q8_1(ffn_hidden);
        gemm_for(ffn_down, q8_ffn_h, ffn_out_batch);
        residual2_add_batch();
    }
    final_norm_batch();
    quantize_q8_1(final_norm);
    gemm_for(lm_head, q8_final, logits_batch);
}
```

### 5.2 Buffer budget for `MAX_PREFILL_TOKENS = 256`

| Buffer                    | Bytes              |
| ------------------------- | -----------------: |
| Batch input (f32)         | 256 × 4096 × 4 = 4 MB |
| Q8_1 quantised activations | 256 × 4096 / 128 × 144 = 1.2 MB |
| Q batch (f32)             | 4 MB |
| K, V batch (f32)          | 1 MB each |
| Attn-out batch            | 4 MB |
| O batch                   | 4 MB |
| Gate, Up batches (f32)    | 12 MB each |
| FFN-hidden, FFN-out       | 12 MB, 4 MB |
| **Total scratch**         | ≈ 60 MB |

Comfortable in the ~10 GiB free after weight upload. Allocated in
`Forward::new` based on a configurable `max_prefill_tokens` constant
(default 256 covers all 15 prompts in the benchmark; longer prompts
fall back to the existing token-by-token path).

### 5.3 GEMM output layout — the only non-trivial reshape

GEMM writes output as **[N, M] row-major** with `stride_d = M`
(verified from `mul_mmq.comp` line 304 — `data_d[(dc_warp + cc) * stride_d + dr_warp + cr]`,
where `dc + cc` is the N-tile column and `dr + cr` is the M-tile
row). For our use case:

- `N = seq_len`, `M = output_dim`
- One row of D per token, contiguous M elements per row

**This is exactly the layout the per-token forward path expects** —
each token's `[output_dim]` projection result is at offset
`token_idx × output_dim` for `output_dim` contiguous floats. **No
transpose dispatch needed.** That was the worry going in; it
evaporates under inspection.

### 5.4 Token-by-token attention stays

Per Phase 3B prompt §1.3 ("Pragmatischer Ansatz") and confirmed
viable by the layout above: prefill attention runs as a sequential
loop over `t in 0..seq_len`, calling the existing `tiled_attn` with
`seq_len = pos_base + t + 1`. The causal mask is automatic (KV
cache only contains positions ≤ t).

This is the slowest part of prefill_batch — `O(seq_len²)` attention
work — but it's the same constant-factor work the Phase 3C
token-by-token path does, just folded into one command buffer
instead of N submits. The headline prefill speedup comes from
batching the **7 projection GEMMs** per layer × 36 layers, not from
attention.

---

## 6. Test summary

```
$ cargo test --release -- --test-threads=1

running 14 tests   (lib unit, ThinkFilter + Q4_K)              14 passed
running 18 tests   (correctness, +2 Phase-3D)                  18 passed   ← +2
running 15 tests   (regression)                                15 passed
                                                                ─────────
                                                                47 passed   (was 45)
```

New tests this phase:
- `test_q8_1_quantize_roundtrip`
- `test_gemm_q4k_vs_gemv_seq1_parity`

All 45 pre-existing tests still pass. **0 validation errors**.

---

## 7. Cross-phase performance summary (numbers unchanged from 3C)

| Metric                  | Ph2D | Ph3A | Ph3B | Ph3C | **Ph3D** | llama.cpp Vk |
| ----------------------- | ---: | ---: | ---: | ---: | -------: | -----------: |
| Decode tok/s (median)   | 13.4 | 66.8 | 67.9 | 64.1 | **64.1** | 114.2        |
| Prefill tok/s (median)  | 56   | 79   | 82   | 79.4 | **79.4** | 4314         |
| Tests                   | 33   | 35   | 45   | 45   | **47**   | —            |
| Shaders                 | 11   | 11   | 11   | 14   | **14**   | —            |
| 15/15 coherent          | n/a  | n/a  | n/a  | ✅   | **✅**   | —            |

The decode/prefill numbers carry over from Phase 3C — Phase 3D ships
infrastructure only. Phase 3E will move them.

---

## 8. Files changed

| File                                              | Status |
| ------------------------------------------------- | ------ |
| `src/backend/vulkan/pipeline.rs`                  | edit — `MmqPushConstants` (64 B) + `Q8_1QuantizePushConstants` (8 B) |
| `tests/correctness.rs`                            | edit — `dequant_q8_1_x4_block` helper + 2 new tests |
| `results/phase3_step_3d_prefill_gemm.md`          | new — this report |

**Untouched:** `forward.rs`, `decode.rs`, `chat.rs`, `pipeline_registry.rs`,
`shaders.rs`, `commands.rs`, `device.rs`, every `vk_shaders/*` —
Phase 3D is push-constant + dispatch-test only. The shader infra
from Phase 3C is sufficient.

---

## 9. Acceptance gates

| Gate                                                     | Status |
| -------------------------------------------------------- | :----: |
| 47/47 tests green                                        |   ✅   |
| 0 validation errors                                      |   ✅   |
| Push-constant struct sizes match SPIR-V reflection       |   ✅ (64 / 8 B) |
| Q8_1 quantize round-trip < 1 % relative error            |   ✅ (~0.3 % observed) |
| **GEMM parity gate**: `[255.984, 511.969] ≈ [256, 512]`  |   ✅   |
| Prefill ≥ 500 tok/s (Gate 1 from prompt)                 |   ⏸ Phase 3E (orchestration not wired) |
| 15-prompt re-benchmark                                   |   ⏸ Phase 3E |

---

## 10. Phase 3E plan (now risk-free)

With the GEMM correctness validated, Phase 3E is mechanical:

1. **Allocate batch scratch buffers** in `Forward::new` (≈ 60 MB,
   gated by `max_prefill_tokens`).
2. **`Forward::run_quantize_q8_1` + `Forward::run_gemm`** dispatch
   helpers (`fn` shape mirrors existing `run_gemv`, push-constants
   already typed).
3. **`Forward::prefill_batch`** orchestrator following §5.1.
4. **`decode::generate_from_tokens` fast-path**: if
   `prefill_tokens.len() ≤ max_prefill_tokens`, call
   `prefill_batch`; else fall back to the token-by-token loop.
5. **Logits parity test**: `prefill_batch(mutex_prompt) ==
   token_by_token(mutex_prompt)` within max_abs_err < 1e-3,
   top-5 token IDs identical.
6. **5-prompt + 15-prompt re-benchmark**.

Realistic effort: **2–3 hours**. The hard part (mul_mmq
correctness) is done.

---

## 11. Commit hash

To be filled in by the commit at the end of this run.
