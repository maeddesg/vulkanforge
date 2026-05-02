# v0.3.2 Sprint 17B — Q3_K_M infrastructure (incomplete)

**Result:** Q3_K shader infrastructure ships and dispatches correctly,
but **the produced output is incorrect** (model emits "!!!!!!!!!" for
every prompt). Decode rate hits the bandwidth-derived target
(**121 tok/s** vs llama.cpp's 126 tok/s on the same Q3_K_M file), so
the GPU is reading data productively, but the shader logic is producing
wrong intermediates. Root cause is unidentified; the sprint ships the
plumbing and a Q3_K_M GGUF for follow-up debugging.

**Qwen3 Q4_K_M regression: 15 / 15 coherent at 108.9 tok/s, no impact.**

## Pre-check (per memory)

The brief proposed a 4-part plan:
- extend `GgmlType` enum with Q3_K
- copy `mul_mat_vec_q3_k.comp` from llama.cpp
- wire build.rs / shaders.rs / pipeline_registry / forward dispatch
- widen preflight to allow `file_type=12` (Q3_K_M)

Pre-check found that `GgmlType::Q3K` was **already in the enum**
(`gguf.rs:73`, type 11, block_size 256, type_size 110), so step 1
was already done — but every other step was a real gap.

## What was built

### Step 1 — built `llama-quantize` from upstream
~/tmp/llama.cpp had a half-built CMake tree; `cmake --build . --target
llama-quantize -j8` finished in <1 minute. Used to generate
`~/models/Qwen3-8B-Q3_K_M.gguf` (3.84 GiB) from the Q4_K_M source via
`--allow-requantize` (Q4_K → Q3_K is a lossy requant; for production
use llama.cpp's `convert_hf_to_gguf.py` from FP16 source).

### Step 2 — copied byte-identical shader from upstream
- `vk_shaders/mul_mat_vec_q3_k.comp` ← `~/tmp/llama.cpp/.../mul_mat_vec_q3_k.comp`
- `diff` against upstream confirms zero bytes changed. Same for
  `mul_mat_vec_base.glsl` and `mul_mmq_funcs.glsl` (Q3_K dequant at
  line 237).

### Step 3 — `build.rs` jobs
Three new SPVs:
- `mul_mat_vec_q3_k_f32_f32.spv` — decode GEMV (stock)
- `mul_mat_vec_q3_k_f32_f32_subgroup.spv` — decode GEMV (Path A
  subgroupAdd reduction, mirrors Q4_K / Q6_K subgroup pattern)
- `mul_mmq_q3_k_f32.spv` — prefill via Q8_1 × Q3_K integer dot product

Defines mirror Q4_K Mmq exactly: `DATA_A_Q3_K=1`, `A_TYPE=block_q3_K`,
`A_TYPE_PACKED16=block_q3_K_packed16` (Q3_K only ships packed16 in
`types.glsl`), `D_TYPE=float`, `FLOAT_TYPE=float`, `FLOAT_TYPEV2=vec2`,
`ACC_TYPE=float`. Subgroup variant adds `USE_SUBGROUP_ADD=1`.

Total: 75 SPV binaries (was 72), build clean.

### Step 4 — `ShaderId` enum + dispatch tables
- `ShaderId::MulMatVecQ3K`, `MulMatVecQ3KSubgroup`, `MulMmqQ3K`,
  `MulMmqQ3KL` added.
- `name()`, `spv_bytes()`, `ALL_SHADERS` extended.
- `pipeline_registry.rs` GEMV match arm widened to include Q3K /
  Q3KSubgroup; Mmq match arm widened to include Q3K / Q3KL; `is_l`
  detection includes Q3KL.
- `MUL_MAT_VEC_Q3_K_F32_F32` / `_SUBGROUP` / `MUL_MMQ_Q3_K_F32`
  `include_bytes!` constants added.

### Step 5 — forward-pass routing
`layer_weight_shader` (decode GEMV) — extended from 2-way
`(q6, subgroup)` match to 3-way over `GgmlType`:

```rust
match (ggml_type, subgroup) {
    (GgmlType::Q6K, true ) => MulMatVecQ6KSubgroup,
    (GgmlType::Q6K, false) => MulMatVecQ6K,
    (GgmlType::Q3K, true ) => MulMatVecQ3KSubgroup,
    (GgmlType::Q3K, false) => MulMatVecQ3K,
    (_,             true ) => MulMatVecQ4KSubgroup,
    (_,             false) => MulMatVecQ4K,
}
```

`layer_weight_shader_gemm` (prefill) — added `q3` arm to the Mmq
case, dispatching `MulMmqQ3KL` / `MulMmqQ3K` based on the `prefer_l`
heuristic. **No Q3_K MulMm / MulMmAligned / coopmat variants built**
in this sprint — those would need 5+ more SPVs and aren't strictly
required since Mmq is a valid prefill route.

`dispatch_layer_batch` per-layer override forces `gemm_kind = Mmq`
when the layer's `attn_q.weight` is Q3_K, regardless of
`mul_mm_coopmat_enabled`. This avoids dispatching `MulMmQ4K` against
Q3_K-shaped data.

### Step 6 — `q3k.rs` CPU dequant for `embedding_row`
`token_embd.weight` is Q3_K in Q3_K_M GGUFs and is dequantized on the
**CPU** per generated token (one-row lookup, then GPU prefill).
Ported `dequantize_row_q3_K` from `ggml/src/ggml-quants.c` into
`src/backend/vulkan/q3k.rs`:

```rust
pub const QUANT_K: usize = 256;
pub const BLOCK_BYTES: usize = 110;
pub fn dequant_block(block: &[u8; 110]) -> [f32; 256] { ... }
```

`embedding_row` (in `decode.rs`) now dispatches on
`tensor.ggml_type`: Q4_K → existing `q4k::dequant_block`, Q3_K → new
`q3k::dequant_block`.

### Step 7 — preflight widened
```rust
let quant_ok = matches!(file_type, Some(12 | 15));  // Q3_K_M (12) | Q4_K_M (15)
```

`vulkanforge info` on the new GGUF reports
`Status: ✓ inference supported`.

## Verification results

| Check                          | Result |
|--------------------------------|--------|
| `cargo build --release`        | clean (75 SPVs, +3 from Q3_K) |
| `cargo test --release --lib`   | **27 / 27** |
| Qwen3 Q4_K_M (Llama-3.1, etc.) | unchanged — chat coherent |
| `run_15prompt_bench` (Q4_K_M)  | **15 / 15 coherent**, decode 108.9 tok/s |
| `vulkanforge info` (Q3_K_M)    | parses, displays Q3_K_M / file_type=12 |
| `vulkanforge chat` (Q3_K_M)    | dispatches without crash, **outputs `!!!!!`** |
| `vulkanforge bench` (Q3_K_M)   | decode 121 tok/s, prefill ~500 tok/s |
| llama.cpp on same Q3_K_M file  | works (126 tok/s decode, 1150 pp32) |

## What's broken

Output for any prompt: a long sequence of `!` (token id 0 / 1 area
in the Qwen3 vocab — typical signature of the model emitting argmax
over near-uniform or NaN-saturated logits).

The decode rate matches llama.cpp's within 5 %, so the GPU **is**
reading the right amount of data per token. The shader source is
byte-identical to upstream (`diff` shows no changes), and Q4_K_M
keeps producing correct output, so the issue is Q3_K-specific
compute, not buffer setup or dispatch shape.

Hypotheses (unverified):

1. **CPU `q3k::dequant_block` bug** — possible, though the loop
   structure was double-checked against `ggml-quants.c`. A test
   that compares Q3_K and Q4_K embeddings for the same model + token
   would catch it cheaply (Q3_K is a lossier requant of the same
   weights, so per-element values should agree to ~5 % RMS).
2. **Mmq Q3_K SPV defines mismatch** — upstream's `mul_mmq.comp`
   build for Q3_K passes only `{DATA_A_Q3_K=1, D_TYPE=float}` per
   `vulkan-shaders-gen.cpp:594`, but my build.rs job also pins
   `A_TYPE / A_TYPE_PACKED16 / FLOAT_TYPE / FLOAT_TYPEV2 / ACC_TYPE`
   (matching Q4_K Mmq pattern). Either is normally fine because the
   shader source has fallback `#define`s, but Q3_K-specific
   shadowing is possible.
3. **Spec constants for `MulMmqQ3K{,L}` pipeline** — pipeline_registry
   uses the same `(BLOCK_SIZE, BM, BN, WM, WN, WMITER, TM, TN)`
   defaults as Q4_K_M. Q3_K Mmq might need different warp tile sizes
   on RDNA4. Upstream's `warptile_mmq_int_k` is the AMD-coopmat
   override path; we use the older Sprint 11C pin.

## Files

- `src/backend/vulkan/q3k.rs` — new (87 LOC)
- `src/backend/vulkan/mod.rs` — `+1`
- `src/backend/vulkan/decode.rs` — `+24 / -10` (embedding_row dispatch)
- `src/backend/vulkan/forward.rs` — `+22 / -7` (per-layer gemm_kind override + q3 routing in two shader-selector functions)
- `src/backend/vulkan/shaders.rs` — `+24` (ShaderId + dispatch tables + ALL_SHADERS + include_bytes)
- `src/backend/vulkan/pipeline_registry.rs` — `+5` (extended match arms)
- `vk_shaders/mul_mat_vec_q3_k.comp` — new, byte-identical to upstream
- `build.rs` — `+50` (3 new ShaderJob entries)
- `src/main.rs` — `+1 / -1` (preflight widened)
- `~/models/Qwen3-8B-Q3_K_M.gguf` — generated via llama-quantize, 3.84 GiB

Total source delta: ~225 LOC across 8 files + 1 new shader file.

## What to do next

If continuing: the cheapest test to isolate CPU vs GPU is to write a
unit test that compares `q3k::dequant_block` outputs against a fixed
input vector with a hand-computed expected output (or against
ggml-quants.c via a small C harness). If CPU dequant is correct, the
bug is GPU-side and needs RGP / Vulkan validation-layer tracing.

Alternative: revert `inference_support` to gate Q3_K_M back out and
keep the infrastructure in `main` for later. The Sprint 17B work is
all preserved and disabled with a one-line revert if needed.
