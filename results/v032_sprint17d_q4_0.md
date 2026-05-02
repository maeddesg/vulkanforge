# v0.3.2 Sprint 17D — Q4_0 shader (incomplete: blocked on Qwen2.5 arch)

**Result:** Q4_0 shader infrastructure ships and is **bit-exact vs
CPU** at every shape we tested. **No Qwen2.5 Q4_0 GGUF runs yet** —
each variant has at least one orthogonal blocker beyond Q4_0
itself: missing biases (architectural), missing Q4_1 shader, or
missing Q8_0 shader. file_type=2 stays gated out of preflight; the
Q4_0 infrastructure is preserved in `main` for the future Qwen2.5
sprint.

This is an honest-negative result like Sprint 17B was — Q4_0 the
shader is correct, but the brief's framing assumed Q4_0 alone would
unlock the four ~/models GGUFs, and the ggml_type dump showed it
wouldn't.

## What shipped

### Q4_0 shader infrastructure (3 new SPVs, 81 total)

- `mul_mat_vec_q4_0_f32_f32{,_subgroup}.spv` — decode GEMV. Built
  from the **generic** `mul_mat_vec.comp` (not a per-quant shader —
  upstream ships per-quant only for K-quants, so this is the first
  time we're using the generic path). Required copying
  `mul_mat_vec.comp` and `dequant_funcs.glsl` from upstream
  byte-identically.
- `mul_mmq_q4_0_f32.spv` — prefill via Q8_1 × Q4_0 integer dot.
  `mul_mmq_funcs.glsl` already had the Q4_0 dequant branch (line
  9-58); we just had to add the build define.

### `q4_0.rs` — CPU dequant

Simplest dequant in the codebase: 18 B / 32 weights, single fp16
`d`, half/half nibble packing. Port of `dequantize_row_q4_0` from
`ggml-quants.c`.

### Routing

Identical to Sprint 17C's Q5_K wiring — `forward.rs::layer_weight_shader{,_gemm}`,
`run_gemm` L-tile bm/bn match, `dispatch_layer_batch::force_mmq`,
pipeline_registry GEMV/Mmq spec-const arms, all extended by one
arm each.

## Why this didn't unlock the four Qwen2.5 GGUFs

`tests/q4_0_dequant_sanity.rs::dump_qwen25_layer0_quant_types`
(new) walks `blk.0.*.weight` for each of the four Qwen2.5 variants
and reveals what they actually contain:

```
=== 0.5B layer 0 ===
  blk.0.attn_q.weight     → Q4_0 dims=[896, 896]
  ...
  output.weight           → Q8_0 dims=[896, 151936]   ← !
  All-tensor counts: {F32: 121, Q4_0: 169, Q8_0: 1}
=== 7B layer 0 ===
  blk.0.attn_q.weight     → Q4_0 dims=[3584, 3584]
  blk.0.ffn_down.weight   → Q4_1 dims=[18944, 3584]   ← !
  output.weight           → Q6K dims=[3584, 152064]
  All-tensor counts: {F32: 141, Q4_0: 194, Q4_1: 3, Q6K: 1}
=== 7B-Pure layer 0 ===
  blk.0.attn_q.weight     → Q4_0 dims=[3584, 3584]
  blk.0.ffn_down.weight   → Q4_0 dims=[18944, 3584]   ← all Q4_0!
  output.weight           → Q4_0 dims=[3584, 152064]
  All-tensor counts: {F32: 141, Q4_0: 198}
=== 14B layer 0 ===
  blk.0.ffn_down.weight   → Q4_1 dims=[13824, 5120]   ← !
  output.weight           → Q6K dims=[5120, 152064]
  All-tensor counts: {F32: 241, Q4_0: 331, Q4_1: 6, Q6K: 1}
```

Plus, every Qwen2.5 layer ships F32 biases that Qwen3 doesn't have:
```
blk.0.attn_q.bias       → F32 dims=[3584]
blk.0.attn_k.bias       → F32 dims=[512]
blk.0.attn_v.bias       → F32 dims=[512]
```

VulkanForge's forward pass has no bias-add path (commented in
`main.rs::inference_support` since Sprint 17A — "the bias-add path
that some Qwen GGUFs ship attn_q/k/v.bias for is also not
implemented for Qwen").

| Model         | Quant blocker         | Architectural blocker     | Net status  |
|---------------|-----------------------|---------------------------|-------------|
| 0.5B          | Q8_0 output           | Q/K/V biases              | ✗           |
| 7B            | Q4_1 ffn_down         | Q/K/V biases              | ✗           |
| 7B-Pure       | (none — pure Q4_0)    | Q/K/V biases              | ✗           |
| 14B           | Q4_1 ffn_down         | Q/K/V biases              | ✗           |

### Live evidence

| Model         | Output                                            |
|---------------|---------------------------------------------------|
| 0.5B          | `JACK booze噂裤 jumping教育教学的基础缘分 …` (multi-lang junk) |
| 7B            | `!!!!!!!!!!!` (logit collapse — Q4_1 fall-through to Q4_K) |
| 7B-Pure       | `Sac Community Community Community looking …` (locked tokens — bias missing) |
| 14B           | (not tested — same Q4_1 issue as 7B)              |

The 7B-Pure failure mode is illuminating: it's pure Q4_0 (no
Q4_1, no Q8_0), so the only missing piece is the bias-add. Without
biases the Q/K/V projections are systematically wrong, which
collapses generation into a small token repertoire ("Community")
rather than the uniform-logit `!!!` we see when the matmul itself
is broken.

## Why the Q4_0 shader is still ready to ship

`tests/q4_0_gemv_correctness.rs` (2 tests) loads M=4 K=3584 of real
Q4_0 weights from `Qwen2.5-7B-Pure`'s `token_embd.weight`, runs
both `MulMatVecQ4_0` stock and `MulMatVecQ4_0Subgroup` against an
all-ones input, and compares to a CPU GEMV via
`q4_0::dequant_block`:

```
[stock]    CPU expected: [0.21186829, 0.15536499, 1.0279541, 0.77983093]
[stock]    GPU got     : [0.21186829, 0.15536499, 1.0279541, 0.77983093]
[subgroup] CPU expected: [0.21186829, 0.15536499, 1.0279541, 0.77983093]
[subgroup] GPU got     : [0.21186829, 0.15536499, 1.0279541, 0.77983093]
```

Bit-exact. K=3584 = 112 Q4_0 blocks per row exercises the
`it_size > 1` main-loop path of `mul_mat_vec.comp`. The shader and
its routing are correct; what's missing is the Qwen2.5 architecture
support around it.

## Verification

| Suite                                    | Count | Result |
|------------------------------------------|-------|--------|
| `cargo test --lib`                       | 27    | ok     |
| `tests/q3k_*` + `tests/q5k_*`            | 9     | ok     |
| `tests/q4_0_dequant_sanity.rs` (NEW)     | 3     | ok     |
| `tests/q4_0_gemv_correctness.rs` (NEW)   | 2     | ok     |
| `tests/correctness::test_gemm_q3k_*`     | 3     | ok     |
| `tests/correctness::test_gemm_q5k_*`     | 3     | ok     |

Q4_K_M chat: coherent (unchanged). Q3_K_M chat: coherent (unchanged).

## What it would take to unlock Qwen2.5

1. **Q/K/V bias support** (largest blocker — affects every Qwen2.5
   variant). Read F32 biases from GGUF, dispatch elementwise add
   on `batch_q / batch_k / batch_v` after each Q/K/V GEMM in
   `dispatch_layer_batch`. Detect via tensor presence
   (`blk.0.attn_q.bias`).
2. **Q4_1 shader** for 7B + 14B's `ffn_down`. Same scope as
   Sprint 17D — copy from upstream, build define, route, CPU
   dequant. `mul_mmq_funcs.glsl` already has the Q4_1 branch
   (shared `#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)`),
   `mul_mmq_shmem_types.glsl` already declares Q4_1's
   `block_a_cache`, `types.glsl` already declares
   `block_q4_1{,_packed16}`. Smallest of the three Qwen2.5
   blockers.
3. **Q8_0 shader** for 0.5B's `output.weight`. Similar shape to
   Q4_1 but a different infrastructure path (block_size = 32,
   34 B/block, single fp16 scale, 32 i8 quants). Q8_0 has its
   own pre-built infrastructure throughout — `block_q8_0` in
   types.glsl, branches in mul_mmq_funcs, etc — same pattern as
   Q4_0 / Q5_K / Q4_1: copy/route/test.

The bias-add is the architectural lift; the two extra quants are
cookie-cutter Sprint 17D-style work.

## Files

- `src/backend/vulkan/q4_0.rs` — new (52 LOC)
- `src/backend/vulkan/mod.rs` — `+1`
- `src/backend/vulkan/decode.rs` — `+19` (Q4_0 branch in `embedding_row`)
- `src/backend/vulkan/forward.rs` — `+10 / -3` (Q4_0 routing in two
  selectors + `force_mmq` + L-tile bm/bn arm)
- `src/backend/vulkan/shaders.rs` — `+18` (4 ShaderId entries +
  name/spv_bytes arms + ALL_SHADERS + 4 include_bytes constants)
- `src/backend/vulkan/pipeline_registry.rs` — `+3` (3 extended match arms)
- `vk_shaders/mul_mat_vec.comp` — new (byte-identical to upstream,
  generic non-K-quant GEMV)
- `vk_shaders/dequant_funcs.glsl` — new (byte-identical, dependency
  of `mul_mat_vec.comp`)
- `vk_shaders/mul_mat_vec_q4_0.comp` — created earlier and removed
  (we use the generic `mul_mat_vec.comp` instead)
- `build.rs` — `+39` (3 new ShaderJob entries)
- `src/main.rs` — `+5 / -7` (preflight comment block updated;
  file_type=2 stays GATED OUT)
- `tests/q4_0_dequant_sanity.rs` — new (3 tests, includes the
  Qwen2.5 layer-type dump that surfaced the real blockers)
- `tests/q4_0_gemv_correctness.rs` — new (2 tests, bit-exact vs CPU)
- `results/v032_sprint17d_q4_0.md` — new (this report)

Total source delta: ~330 LOC across 11 files + 3 new shader files.

## Lessons re-learnt

The Sprint 17B → 17B-debug pattern repeated exactly:
- **Pre-check correctly identified what infrastructure was already
  in place** — types.glsl + mul_mmq_funcs + shmem_types had the
  Q4_0 branches, GgmlType::Q4_0 was in the parser. Saved the
  shader-source-side work.
- **End-to-end test surfaced what pre-check couldn't see** — the
  per-tensor `ggml_type` dump on real GGUFs revealed the
  mixed-quant patterns (Q4_1 + Q6_K + Q8_0) and the bias presence
  that the brief's "all weights are Q4_0" assumption missed.

Adding `dump_qwen25_layer0_quant_types` to the test suite up front
turned this from a "shader's broken!" debug session into a 30-min
"shader's fine, the model has features we don't support yet" one.
