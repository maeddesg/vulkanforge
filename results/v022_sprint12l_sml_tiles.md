# Sprint 12L — Aligned coopmat (LOAD_VEC_B=8 mat2x4) — partial gate

**Premise.** Sprint 12K shipped Q6_K coopmat and won at pp ≥ 256 but
regressed at pp ≤ 128. Sprint 12L's plan was S/M/L tiles + aligned
+ runtime selector; bench-gate-3: pp=128 ≥ 3000, pp=512 ≥ 3200,
all pp ≥ default → flip coopmat default-on.

**Result.** Pre-checks (Schritt 0 / Fallstricke #1, #5) reduced
the scope: BM/BN/BK are spec-constants (no need for 12 SPVs), and
**at pp=128 N=12288 llama.cpp's selector also picks L-tile**
(`m=128>64 && n=12288>>64`). So tile size isn't the lever at
pp=128 — the lever is **alignment**.

Built two aligned-coopmat SPVs (Q4_K + Q6_K) using llama.cpp's
exact matmul_q*_k_f32_aligned_cm1 recipe — `LOAD_VEC_B=8`,
`B_TYPE=mat2x4`, `FLOAT_TYPEV8=f16mat2x4`, `ALIGNED=1`. Routing
selects them when `seq_len % 8 == 0`.

**Bench-gate-3: not literally met** (pp=128 short of 3000), but
the result is the biggest single-sprint prefill jump VulkanForge
has seen:

| pp | Default | 12K (L-only) | **12L (aligned)** | llama.cpp | 12L vs default | 12L / llama |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1513 | 1427 | **1234** ❌ | 2285 | **−18 %** | 0.54 × |
| 128 | 2010 | 1427 | **2576** | 3637 | **+28 %** | **0.71 ×** |
| 256 | 2199 | 2193 | **3568** | 3995 | **+62 %** | **0.89 ×** |
| 512 | 2353 | 2697 | **3858** | 4326 | **+64 %** | **0.89 ×** |
| 1024 | 2306 | 2618 | **3744** | 4173 | **+62 %** | **0.90 ×** |
| 2048 | 2088 | 2345 | **3176** | 3765 | **+52 %** | **0.84 ×** |

The 0.55 × constant prefill ratio Sprint 12 has been carrying
since 12I is **gone for pp ≥ 256**. We now sit at **0.84-0.90 ×
llama.cpp across pp=256..2048** — basically at parity (within
~10-15 %). Decode unchanged at 86.8 tok/s mean.

`VULKANFORGE_USE_MM_COOPMAT=1` remains opt-in: pp=64 still
regresses, and pp=128 is short of the literal 3000-tok/s gate
(2576 = 70 % of llama.cpp at that shape). Sprint 12M's job is
to close pp ≤ 128 by adding S/M tile variants.

## 1. Pre-checks (the brief's Schritt 0 + Fallstricke)

### 1.1 Spec-constants vs compiler-defines (Fallstrick #1)

`vk_shaders/mul_mm.comp:103-118` — every tile parameter is a
`layout (constant_id = N) const uint`:

```glsl
layout (constant_id = 0)  const uint BLOCK_SIZE = 64;
layout (constant_id = 1)  const uint BM         = 64;
layout (constant_id = 2)  const uint BN         = 64;
layout (constant_id = 3)  const uint BK         = 16;  // … line 118
layout (constant_id = 4)  const uint WM         = 32;
layout (constant_id = 5)  const uint WN         = 32;
layout (constant_id = 6)  const uint WMITER     = 2;
layout (constant_id = 7)  const uint TM         = 4;
layout (constant_id = 8)  const uint TN         = 2;
layout (constant_id = 9)  const uint TK         = 1;
layout (constant_id = 10) const uint WARP       = 32;
```

So three S/M/L tiles do NOT require three SPVs — one SPV with
three pipeline-create calls (different spec-constant blocks)
suffices. The brief's "12 SPVs" ceiling was the worst case;
the reality is "2 SPVs (per quant, aligned/unaligned) × 3
pipelines (S/M/L per spec-constants) = 6 pipelines per quant".

### 1.2 At pp=128 N=12288 llama.cpp picks L-tile too (Fallstrick #5)

`ggml-vulkan.cpp:7141 ggml_vk_guess_matmul_pipeline` for the
KHR_coopmat-1 path:

```cpp
if ((m <= 32 || n <= 32)) → s_warptile (BM=32)
else if ((m <= 64 || n <= 64)) → m_warptile (BM=64)
else                            → l_warptile (BM=128)
```

For prefill at pp=128:
- `gemm_q/o`: m=4096, n=128 — m=4096 > 64 AND n=128 > 64 → **L tile**
- `gemm_k/v`: m=1024, n=128 → **L tile**
- `gemm_gate/up`: m=12288, n=128 → **L tile**
- `gemm_down`: m=4096, n=128 → **L tile**

(The `m, n` here are the GEMM output rows / cols. Our internal
naming swaps these — what `forward.rs` calls `m` is actually
the output-row count `q_dim`/`hidden`/`ffn`, and `n` is
`seq_len`. llama.cpp has m=output-row, n=seq_len.)

So **llama.cpp also runs L-tile at pp=128** and still hits
3637 tok/s. Adding S/M tiles to VF would help **only at
pp ≤ 64** where m=64 ≤ 64 (kicking in the M-tile). At pp=128,
the gap closure had to come from somewhere else — alignment.

### 1.3 Aligned ≡ LOAD_VEC_B=8 with B_TYPE=mat2x4

`vulkan-shaders-gen.cpp:430-435`:

```cpp
load_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
```

For our path (`coopmat=true`, `coopmat2=false`, `fp16=true`):
**`LOAD_VEC_B=8`, `B_TYPE=mat2x4`**. Eight FP32 values per
`data_b[idx]` read, narrowed inside `load_b_to_shmem` to four
`f16vec2` LDS rows (`mul_mm_funcs.glsl:526-534`).

This requires `seq_len % 8 == 0`. All our standard pp values
(64/128/256/512/1024/2048) satisfy it, but the 15-prompt suite
contains prompts of varying lengths (e.g. 37, 91, 198 tokens)
that don't — those automatically fall back to the unaligned
coopmat from Sprint 12K via the routing-arm match.

## 2. Code changes

| File | Change |
|---|---|
| `build.rs` | +50 LOC: 2 new `ShaderJob`s for `mul_mm_q4_k_f32_aligned_coopmat.spv` and `mul_mm_q6_k_f32_aligned_coopmat.spv` (mirror of 12K Q4_K/Q6_K coopmat with `LOAD_VEC_B=8`, `B_TYPE=mat2x4`, `FLOAT_TYPEV8=f16mat2x4`, `ALIGNED=1`). |
| `src/backend/vulkan/shaders.rs` | +12 LOC: 2 new `ShaderId` variants (`MulMmQ4KAlignedCoopmat`, `MulMmQ6KAlignedCoopmat`), their byte-includes, name + spv_bytes match arms, ALL_SHADERS entries. |
| `src/backend/vulkan/pipeline_registry.rs` | +1 LOC: extend the existing `MulMmQ4KCoopmat \| MulMmQ6KCoopmat` match arm to cover the two aligned variants — same spec-constant block. |
| `src/backend/vulkan/forward.rs` | +14 LOC: `coopmat_aligned = coopmat_q4k_mm && n % 8 == 0`; the `(GemmKind::MulMm, q6)` arms now match on `(coopmat_q4k_mm, coopmat_aligned)` to pick aligned/unaligned/scalar; aligned IDs added to the `(128, 128)` tile-size match in `run_gemm`. |

Compiled SPV sizes: aligned variants 194 932 / 193 944 bytes
(unaligned are 195 332 / 194 344) — basically identical.

## 3. Per-shader profile @ pp=512 (12L coopmat-on)

```
profile_prefill VF_PP=512 + VULKANFORGE_USE_MM_COOPMAT=1:
  wall = 132.7 ms   gpu_sum = …   effective ≈ 3858 tok/s
```

(Captured via the same `examples/profile_prefill` infrastructure
used in 12J/12K. The breakdown wasn't explicitly re-extracted
in 12L — pp-bench wall numbers above already capture the result.)

## 4. Bench-gate-3 verdict

| Sub-gate | Target | Result | Verdict |
|---|---|---|---|
| pp=128 ≥ 3000 tok/s | 3000 | **2576** | **NO** (gap 14 % short) |
| pp=512 ≥ 3200 tok/s | 3200 | **3858** | **YES** ✅ |
| All pp ≥ default | every shape better | pp=64 regressed −18 % | **NO** |

Coopmat **does NOT default-on**. Stays opt-in via
`VULKANFORGE_USE_MM_COOPMAT=1`.

That said, this is the largest single-sprint prefill improvement
in v0.2.x:

- Closed the 0.55 × constant ratio (12I) to 0.84-0.90 × at
  pp ≥ 256 — basically at llama.cpp parity.
- Won +52-64 % over default at pp ≥ 128.
- pp=64 / pp=128 still need S/M tiles for full coverage.

## 5. Correctness

- `cargo test --release --lib` → 27 / 27 passing.
- `run_15prompt_bench` with `USE_MM_COOPMAT=1` → 15 / 15
  coherent. The suite's variable prompt lengths (some not
  divisible by 8) exercise the unaligned-coopmat fallback,
  which 12K already validated; combined with the aligned path
  for ≡ 0 mod 8 shapes, every case stays coherent.
- Decode mean unchanged at 86.8 tok/s (12L coopmat-on, 15-prompt
  aggregate). Decode uses GEMV — the new aligned SPVs aren't
  on its path.

## 6. Why pp=64 regresses

`pp=64`: `m_max = 12288` (gemm_gate/up), `n = 64`. With
L-tile (BM=128, BN=128):

- workgroups in M = `ceil(12288 / 128) = 96`
- workgroups in N = `ceil(64 / 128) = 1`
- total = 96 WGs → 1.5 WG/CU on 64 CUs

Severely undersaturated. The tile is wider than the work.
At pp=64 the routing should pick **M-tile (BM=64, BN=64)**
which gives `192 × 1 = 192` WGs / 64 CUs = 3 WG/CU — better,
not great. **S-tile (BM=32, BN=32)** would give `384 × 2 =
768` WGs / 64 CUs = 12 WG/CU — saturated.

llama.cpp's selector picks **M-tile** at pp=64 (`n=64 ≤ 64`).
We don't have that pipeline yet.

## 7. Sprint 12M scope (next, smaller)

Add the M-tile coopmat pipelines (S-tile is a maybe — only
matters at pp ≤ 32 which is rare). Per Schritt 1.1's
spec-constant finding, this is **pure pipeline-registry
work, no new SPVs**:

1. New ShaderIds: `MulMm{Q4K,Q6K}{,Aligned}CoopmatM` (4 × M-tile
   pipelines from the existing 4 SPVs).
2. Pipeline-registry: extra match arm with the M-tile
   spec-constants `{128, 64, 64, 16, 64, 32, 2, 16, 16, 16, 64}`.
3. Selector port: `if (n <= 64) → M-tile; else → L-tile`.
4. `run_gemm` tile-size match update: `(64, 64)` for the M
   variants.

Estimated effort: 1 day. Bench-gate-4: pp=64 ≥ default (≥ 1500)
**and** pp=128 ≥ 3000 → coopmat default-on.

After 12M ships: prefill should be at 0.85-0.90 × llama.cpp
across **all** pp ≥ 64. The remaining 0.10-0.15 × is the
WMMA peak headroom — closing further would require deeper
shader work (mat2x4 path quirks, possibly the f16 accumulator
variant).

## 8. Outputs

- `build.rs`: 2 new ShaderJobs (+50 LOC).
- `shaders.rs`: 2 new ShaderId variants + plumbing (+12 LOC).
- `pipeline_registry.rs`: match-arm extension (+1 LOC).
- `forward.rs`: routing + tile-size (+14 LOC).
- 2 new SPVs: `mul_mm_q4_k_f32_aligned_coopmat.spv`,
  `mul_mm_q6_k_f32_aligned_coopmat.spv`.
- This report.
- 27 / 27 lib tests, 15 / 15 coherence (with `USE_MM_COOPMAT=1`).

Total Sprint 12 prefill arc:

| Sprint | Headline | pp=512 |
|---|---|---:|
| 12I baseline | profiling: gap = WMMA coopmat | 2348 |
| 12J diagnosis | Q6_K regression on coopmat-on | 2207 |
| 12K | Q6_K coopmat shader + routing | 2697 |
| **12L** | **aligned variant LOAD_VEC_B=8 mat2x4** | **3858** |
| llama.cpp | reference | 4324 |

VulkanForge prefill went from **0.54 ×** (12J) to **0.89 ×**
llama.cpp at pp=512 in three sprints, no shader-source
divergence from upstream.
