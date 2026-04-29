#![allow(clippy::ptr_arg, clippy::op_ref)]

//! Build-time GLSL → SPIR-V compilation for VulkanForge.
//!
//! Phase 2A — compiles every shader the decode pipeline needs into a
//! SPIR-V blob in `OUT_DIR`. `src/backend/vulkan/shaders.rs` embeds the
//! results via `include_bytes!`. Defines per shader mirror what
//! llama.cpp's `vulkan-shaders-gen` passes to glslc for the same
//! variant.

use std::env;
use std::fs;
use std::path::PathBuf;

use shaderc::{
    CompileOptions, Compiler, EnvVersion, OptimizationLevel, ResolvedInclude, ShaderKind,
    SourceLanguage, TargetEnv,
};

struct ShaderJob {
    out_name: &'static str,
    entry_source: &'static str,
    defines: &'static [(&'static str, &'static str)],
}

const JOBS: &[ShaderJob] = &[
    // Q4_K decode GEMV (Phase 1 baseline). Defines per
    // results/phase1_step_1.0_shader_analysis.md §3.3.
    ShaderJob {
        out_name: "mul_mat_vec_q4_k_f32_f32.spv",
        entry_source: "mul_mat_vec_q4_k.comp",
        defines: &[
            ("DATA_A_Q4_K", "1"),
            ("B_TYPE", "float"),
            ("B_TYPEV2", "vec2"),
            ("B_TYPEV4", "vec4"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
        ],
    },
    // Q6_K decode GEMV. Same scaffolding as Q4_K, just a different
    // weight storage format.
    ShaderJob {
        out_name: "mul_mat_vec_q6_k_f32_f32.spv",
        entry_source: "mul_mat_vec_q6_k.comp",
        defines: &[
            ("DATA_A_Q6_K", "1"),
            ("B_TYPE", "float"),
            ("B_TYPEV2", "vec2"),
            ("B_TYPEV4", "vec4"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
        ],
    },
    // RMSNorm. generic_binary_head: 3 SSBOs (input A, weight B, output D).
    ShaderJob {
        out_name: "rms_norm_f32.spv",
        entry_source: "rms_norm.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
        ],
    },
    // v0.2 Sprint 9c.5 — fused rms_norm+mul+RoPE. Same shader source
    // as rms_norm.comp, but built with RMS_NORM_ROPE_FUSION=1 so the
    // rope_neox/rope_norm path runs after the rms_norm+mul step
    // (sharing the normalized intermediate via `rope_data_a` LDS).
    // 7 SSBO bindings (A, B, [skip 2], pos, ff, output, set_rows_idx).
    // Mirrors llama.cpp's `rms_norm_mul_rope_f32_f32` SPV.
    ShaderJob {
        out_name: "rms_norm_mul_rope_f32.spv",
        entry_source: "rms_norm.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("ROPE_D_TYPE", "float"),
            ("RMS_NORM_ROPE_FUSION", "1"),
        ],
    },
    // RoPE (rotated position embedding) — Llama-style "norm" variant.
    // rope_head.glsl: 5 SSBOs (data, pos, ff, output, indices).
    ShaderJob {
        out_name: "rope_norm_f32.spv",
        entry_source: "rope_norm.comp",
        defines: &[("A_TYPE", "float"), ("ROPE_D_TYPE", "float")],
    },
    // RoPE NeoX-style (split-half rotation). llama.cpp returns
    // LLAMA_ROPE_TYPE_NEOX for LLM_ARCH_QWEN3 — see Phase-2B report
    // §2 — so this is the variant the Qwen3 decode path actually uses.
    ShaderJob {
        out_name: "rope_neox_f32.spv",
        entry_source: "rope_neox.comp",
        defines: &[("A_TYPE", "float"), ("ROPE_D_TYPE", "float")],
    },
    // Element-wise add (residual). generic_binary_head, 3 SSBOs.
    ShaderJob {
        out_name: "add_f32.spv",
        entry_source: "add.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
        ],
    },
    // Element-wise multiply (gate * up in SwiGLU). generic_binary_head.
    ShaderJob {
        out_name: "mul_f32.spv",
        entry_source: "mul.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
        ],
    },
    // SiLU activation. generic_head: 2 SSBOs (input, output).
    ShaderJob {
        out_name: "silu_f32.spv",
        entry_source: "silu.comp",
        defines: &[("A_TYPE", "float"), ("D_TYPE", "float")],
    },
    // v0.2 Sprint 9a — fused SwiGLU (silu(gate) * up → out). 3 SSBOs,
    // single u32 push-constant `n`. Replaces the separate silu+mul
    // pair in the FFN block; -1 dispatch, -1 barrier per layer.
    ShaderJob {
        out_name: "swiglu_f32.spv",
        entry_source: "swiglu.comp",
        defines: &[],
    },
    // v0.2 Sprint 9b — fused residual-add + RMSNorm-mul. 5 SSBOs
    // (a, b, weight, sum, norm_out); 1 WG per row, BLOCK_SIZE=512.
    // Replaces (add → barrier → rms_norm) at Stelle 1 (add_res1 +
    // rms_norm_ffn). -1 dispatch, -1 barrier per layer.
    ShaderJob {
        out_name: "multi_add_rms_f32.spv",
        entry_source: "multi_add_rms.comp",
        defines: &[],
    },
    // v0.2 Sprint 9d.2 — FP32 → packed-FP16 conversion compute shader.
    // Replaces vkCmdCopyBuffer for prefill KV-cache writes when
    // VULKANFORGE_FP16_KV=1 is enabled. Each thread converts 2
    // FP32 elements into 1 packed uint via packHalf2x16.
    ShaderJob {
        out_name: "kv_copy_fp16.spv",
        entry_source: "kv_copy_fp16.comp",
        defines: &[],
    },
    // Softmax (attention scores). Own header, ~3 SSBOs.
    ShaderJob {
        out_name: "soft_max_f32.spv",
        entry_source: "soft_max.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
        ],
    },
    // Buffer copy / dtype conversion. generic_unary_head: 2 SSBOs.
    ShaderJob {
        out_name: "copy_f32_f32.spv",
        entry_source: "copy.comp",
        defines: &[("A_TYPE", "float"), ("D_TYPE", "float")],
    },
    // VulkanForge Phase-2C scalar single-token attention.
    ShaderJob {
        out_name: "scalar_attn_f32.spv",
        entry_source: "scalar_attn.comp",
        defines: &[],
    },
    // VulkanForge Phase-4B online-softmax decode attention. Drop-in
    // compatible bindings + push-constants with scalar_attn; one WG
    // per Q-head, TILE=64 K positions per pass, no big scores[] LDS.
    ShaderJob {
        out_name: "flash_attn_f32.spv",
        entry_source: "flash_attn.comp",
        defines: &[],
    },
    // VulkanForge Phase-4C split-K worker. Dispatched as
    // (n_heads, n_tiles, 1); writes per-tile partials to scratch.
    ShaderJob {
        out_name: "flash_attn_split_f32.spv",
        entry_source: "flash_attn_split.comp",
        defines: &[],
    },
    // VulkanForge Phase-4C split-K reducer. Dispatched as (n_heads,
    // 1, 1); combines partials with online-softmax correction.
    ShaderJob {
        out_name: "flash_attn_reduce_f32.spv",
        entry_source: "flash_attn_reduce.comp",
        defines: &[],
    },
    // VulkanForge Phase-5B.1 batched-Q flash attention for prefill.
    // Dispatched as (n_heads, M, 1); each WG handles one (head, query)
    // and applies the causal mask via causal_len = q_start + q_idx + 1.
    ShaderJob {
        out_name: "flash_attn_batch_f32.spv",
        entry_source: "flash_attn_batch.comp",
        defines: &[],
    },
    // VulkanForge Sprint 7 — Br>1 tiled-Q flash attention. Identical
    // bindings + push-constants to flash_attn_batch; dispatched as
    // (n_heads, ceil(M/BR), 1) with BR queries sharing one K-tile.
    // Sprint 7.5 — three SPVs for the BR sweep (4 / 8 / 16). GLSL
    // shared arrays can't be sized by spec constants, so BR is a
    // compile-time #define and we emit one SPV per Br.
    ShaderJob {
        out_name: "flash_attn_tiled_br4.spv",
        entry_source: "flash_attn_tiled.comp",
        defines: &[("BR", "4")],
    },
    ShaderJob {
        out_name: "flash_attn_tiled_br8.spv",
        entry_source: "flash_attn_tiled.comp",
        defines: &[("BR", "8")],
    },
    ShaderJob {
        out_name: "flash_attn_tiled_br16.spv",
        entry_source: "flash_attn_tiled.comp",
        defines: &[("BR", "16"), ("BC", "64")],
    },
    // Sprint 7.6 — Br=16 with Bc=32 K-tile. Halves the K-LDS row
    // count (16 KB vs 32 KB), enabling higher WG-occupancy at the
    // cost of half-utilised score-compute threads (32 active out
    // of 64 in the wave).
    ShaderJob {
        out_name: "flash_attn_tiled_br16_bc32.spv",
        entry_source: "flash_attn_tiled.comp",
        defines: &[("BR", "16"), ("BC", "32")],
    },
    // v0.2 Sprint 9d.2 — FP16 KV-cache reader variants. Same source
    // as the FP32 build but compiled with -DFP16_KV=1 so K/V binding
    // switch to `uint[]` and the load_k/load_v helpers decode via
    // unpackHalf2x16. Used when VULKANFORGE_FP16_KV=1 routes the
    // prefill attention through these SPVs.
    ShaderJob {
        out_name: "flash_attn_tiled_br16_bc32_fp16kv.spv",
        entry_source: "flash_attn_tiled.comp",
        defines: &[("BR", "16"), ("BC", "32"), ("FP16_KV", "1")],
    },
    ShaderJob {
        out_name: "flash_attn_batch_fp16kv.spv",
        entry_source: "flash_attn_batch.comp",
        defines: &[("FP16_KV", "1")],
    },
    // v0.2 Sprint 9d.3 — FP16 KV variants for the decode-side
    // attention shaders. flash_attn (single-WG) covers seq_len ≤ 64,
    // flash_attn_split (multi-WG split-K) covers > 64. Same source
    // SPVs as the FP32 builds, just with FP16_KV=1.
    ShaderJob {
        out_name: "flash_attn_fp16kv.spv",
        entry_source: "flash_attn.comp",
        defines: &[("FP16_KV", "1")],
    },
    ShaderJob {
        out_name: "flash_attn_split_fp16kv.spv",
        entry_source: "flash_attn_split.comp",
        defines: &[("FP16_KV", "1")],
    },
    // Phase-6A probe: confirms shaderc 0.8 + Mesa glslang ship a
    // coopmat + bfloat16 toolchain that produces SPV without warnings.
    // Output is unused at runtime — purely a build-time GO gate. See
    // results/phase6a_coopmat_benchmark.md.
    ShaderJob {
        out_name: "_probe_coopmat.spv",
        entry_source: "_probe_coopmat.comp",
        defines: &[],
    },
    // Phase-6A pure-WMMA throughput benchmark (BF16 × BF16 → FP32).
    // Used by examples/bench_coopmat. Not loaded by the runtime forward
    // pass.
    ShaderJob {
        out_name: "bench_coopmat_pure_f32.spv",
        entry_source: "bench_coopmat_pure.comp",
        defines: &[],
    },
    // v0.2 smoke-test FP8 throughput bench (E4M3 × E4M3 → FP32).
    // Twin of bench_coopmat_pure for the FP8 path. Used by
    // examples/bench_coopmat when VF_BENCH_FP8=1.
    ShaderJob {
        out_name: "bench_coopmat_fp8_e4m3.spv",
        entry_source: "bench_coopmat_fp8.comp",
        defines: &[],
    },
    // v0.2 Sprint 1A — tiled BF16 coopmat GEMM (BN=64 default).
    // 4 subgroups × 2x2 WMMA per subgroup over a 64x64 output tile,
    // BK=16 K-step, LDS-staged with bank-conflict padding. Used by
    // examples/bench_coopmat when VF_BENCH_TILED=1, and by the
    // tiled-coopmat correctness suite. Not loaded by the runtime
    // forward pass yet (Sprint 3 task).
    ShaderJob {
        out_name: "mul_coopmat_bf16_f32.spv",
        entry_source: "mul_coopmat_bf16.comp",
        defines: &[("BN", "64")],
    },
    // v0.2 Sprint 1A.5 — Skinny-N variants with smaller BN. Same
    // source, different macro: BN=32 splits the output tile across
    // 4 subgroups in a 4x1 grid (each subgroup 16x32 = 2 WMMAs);
    // BN=16 is the same 4x1 grid but each subgroup emits a single
    // 16x16 WMMA. Both increase WG-count along N for short-seq
    // prefill workloads (N == seq_len ≤ 64).
    ShaderJob {
        out_name: "mul_coopmat_bf16_bn32.spv",
        entry_source: "mul_coopmat_bf16.comp",
        defines: &[("BN", "32")],
    },
    ShaderJob {
        out_name: "mul_coopmat_bf16_bn16.spv",
        entry_source: "mul_coopmat_bf16.comp",
        defines: &[("BN", "16")],
    },
    // v0.2 Sprint 1B — tiled FP8 (E4M3) coopmat GEMM. Lift-and-shift
    // from the BF16 skeleton: same tile geometry and parametric BN, but
    // input element type is `floate4m3_t`. Three SPVs for BN ∈
    // {16, 32, 64}. Used by examples/bench_coopmat under the
    // TiledFp8Bn{16,32,64} / TiledFp8Auto modes.
    ShaderJob {
        out_name: "mul_coopmat_fp8_bn64.spv",
        entry_source: "mul_coopmat_fp8.comp",
        defines: &[("BN", "64")],
    },
    ShaderJob {
        out_name: "mul_coopmat_fp8_bn32.spv",
        entry_source: "mul_coopmat_fp8.comp",
        defines: &[("BN", "32")],
    },
    ShaderJob {
        out_name: "mul_coopmat_fp8_bn16.spv",
        entry_source: "mul_coopmat_fp8.comp",
        defines: &[("BN", "16")],
    },
    // v0.2 Sprint 2A — isolated Q4_K dequant probe (no GEMM, no LDS
    // tiling). Three output-type variants share the same shader source
    // so the only difference between SPVs is the f32→{f32,bf16,fp8}
    // narrowing convert at the store. Used by tests/dequant_q4k.rs
    // for correctness vs CPU and by examples/bench_dequant for the
    // throughput comparison.
    ShaderJob {
        out_name: "dequant_q4k_fp32.spv",
        entry_source: "dequant_q4k_debug.comp",
        defines: &[("OUT_FP32", "1")],
    },
    ShaderJob {
        out_name: "dequant_q4k_bf16.spv",
        entry_source: "dequant_q4k_debug.comp",
        defines: &[("OUT_BF16", "1")],
    },
    ShaderJob {
        out_name: "dequant_q4k_fp8.spv",
        entry_source: "dequant_q4k_debug.comp",
        defines: &[("OUT_FP8", "1")],
    },
    // v0.2 Sprint 2B — Q4_K dequant-fusion into the tiled FP8 coopmat
    // GEMM. A is bound as `block_q4_K[]` (144 B/256 weights), B is
    // FP32 activations, output is FP32. Each K-step does Q4_K → FP32
    // → FP8 in registers, then coopMatMulAdd in FP8. Three BN
    // variants for the same per-shape Sprint-1A.5 selector.
    ShaderJob {
        out_name: "mul_coopmat_q4k_bn64.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "64")],
    },
    ShaderJob {
        out_name: "mul_coopmat_q4k_bn32.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "32")],
    },
    ShaderJob {
        out_name: "mul_coopmat_q4k_bn16.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "16")],
    },
    // v0.2 Sprint 3A — Q4_K coopmat with forward-pass-compatible
    // memory layout. B is read as [N, K] row-major (matches the
    // runtime batch_norm activations) and C is written as [N, M]
    // row-major (matches the runtime mul_mmq output convention so
    // downstream RoPE/attention consumers stay unchanged). Three BN
    // variants for the same per-shape selector used in Sprint 1A.5/1B.
    ShaderJob {
        out_name: "mul_coopmat_q4k_fwd_bn64.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "64"), ("FORWARD_LAYOUT", "1")],
    },
    ShaderJob {
        out_name: "mul_coopmat_q4k_fwd_bn32.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "32"), ("FORWARD_LAYOUT", "1")],
    },
    ShaderJob {
        out_name: "mul_coopmat_q4k_fwd_bn16.spv",
        entry_source: "mul_coopmat_q4k.comp",
        defines: &[("BN", "16"), ("FORWARD_LAYOUT", "1")],
    },
    // v0.2 Sprint 3B — naive Q4_K coopmat with BF16 narrowing
    // (1 Wave64/WG, single 16x16 output tile). Targets skinny-N
    // prefill where the tiled kernels of Sprint 2B/3A regress.
    // BF16 (7 mantissa bits) replaces FP8 (3 bits) so the precision
    // is sufficient for full-stack Qwen3 forward.
    ShaderJob {
        out_name: "mul_coopmat_q4k_naive_bf16.spv",
        entry_source: "mul_coopmat_q4k_naive.comp",
        defines: &[],
    },
    // v0.2 Sprint 3C — naive Q4_K coopmat with N-padding. Same
    // shader source as the Sprint 3B BF16 variant, but compiled with
    // -DPADDED_OUTPUT so the output store is a single ColumnMajor
    // coopMatStore (no LDS staging, no per-thread bounds check). The
    // runtime is responsible for rounding pc.n up to a multiple of
    // 16 and zero-filling the activation tail rows.
    ShaderJob {
        out_name: "mul_coopmat_q4k_naive_padded_bf16.spv",
        entry_source: "mul_coopmat_q4k_naive.comp",
        defines: &[("PADDED_OUTPUT", "1")],
    },
    // v0.2 Sprint 3C — FP8 retry. Sprint 3A's "FP8 precision
    // failure" was actually a partial-tile-store bug. With
    // PADDED_OUTPUT eliminating partial tiles, FP8 should produce
    // logits parity. ELEM_TYPE switches to floate4m3_t under
    // -DFP8_MODE; convert is one native v_cvt_pk_fp8_f32 (vs BF16's
    // 5-VALU-op software pack).
    ShaderJob {
        out_name: "mul_coopmat_q4k_naive_padded_fp8.spv",
        entry_source: "mul_coopmat_q4k_naive.comp",
        defines: &[("PADDED_OUTPUT", "1"), ("FP8_MODE", "1")],
    },
    // Phase 6 v0.1.2 cont. — mul_mm.comp port from llama.cpp
    // (MIT-licensed). Same shader runtime as mul_mmq.comp but takes
    // FP32 activations directly (no Q8_1 quantize step in front), uses
    // a different shared-memory layout (vec2-packed buf with +1 stride
    // padding instead of mul_mmq's per-block buf), and hits ~2× the
    // GEMM throughput on RDNA4 in llama.cpp's reference. Used by the
    // Phase-6 prefill path; mul_mmq stays as the gated fallback.
    ShaderJob {
        out_name: "mul_mm_q4_k_f32.spv",
        entry_source: "mul_mm.comp",
        defines: &[
            ("DATA_A_Q4_K", "1"),
            ("A_TYPE", "block_q4_K"),
            ("A_TYPE_PACKED32", "block_q4_K_packed32"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            // llama.cpp's vulkan-shaders-gen pins LOAD_VEC_A=4 for the
            // Q4_K / Q6_K unaligned mul_mm builds (load_vec_quant=4 in
            // gen.cpp:560). Without this the dequant index math
            // de-syncs by a factor of 4 and the GEMM produces garbage.
            ("LOAD_VEC_A", "4"),
        ],
    },
    ShaderJob {
        out_name: "mul_mm_q6_k_f32.spv",
        entry_source: "mul_mm.comp",
        defines: &[
            ("DATA_A_Q6_K", "1"),
            ("A_TYPE", "block_q6_K"),
            ("A_TYPE_PACKED16", "block_q6_K_packed16"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            // Phase 7 — Q6_K's load_a_to_shmem branch in
            // mul_mm_funcs.glsl emits "// 2 values per idx" and writes a
            // single FLOAT_TYPEV2 to buf_a[buf_idx] per invocation. Q4_K
            // ("// 4 values per idx") writes two — LOAD_VEC_A=4 lines up
            // there but leaves half of buf_a uninitialised on the Q6_K
            // path, surfacing as NaN logits at scale. Pin Q6_K to 2.
            ("LOAD_VEC_A", "2"),
        ],
    },
    // Phase 7 (cont.) — aligned mul_mm variants. Mirror llama.cpp's
    // vulkan-shaders-gen.cpp:583 logic: ALIGNED=1, LOAD_VEC_B=4,
    // B_TYPE=vec4 → load_b_to_shmem takes the vec4-of-floats path
    // at mul_mm_funcs.glsl:535. Only safe when shader N (= seq_len)
    // is divisible by LOAD_VEC_B; runtime falls back to mul_mmq when
    // not aligned.
    ShaderJob {
        out_name: "mul_mm_q4_k_f32_aligned.spv",
        entry_source: "mul_mm.comp",
        defines: &[
            ("DATA_A_Q4_K", "1"),
            ("A_TYPE", "block_q4_K"),
            ("A_TYPE_PACKED32", "block_q4_K_packed32"),
            ("B_TYPE", "vec4"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("FLOAT_TYPEV4", "vec4"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            ("LOAD_VEC_A", "4"),
            ("LOAD_VEC_B", "4"),
            ("ALIGNED", "1"),
        ],
    },
    ShaderJob {
        out_name: "mul_mm_q6_k_f32_aligned.spv",
        entry_source: "mul_mm.comp",
        defines: &[
            ("DATA_A_Q6_K", "1"),
            ("A_TYPE", "block_q6_K"),
            ("A_TYPE_PACKED16", "block_q6_K_packed16"),
            ("B_TYPE", "vec4"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("FLOAT_TYPEV4", "vec4"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            ("LOAD_VEC_A", "2"),
            ("LOAD_VEC_B", "4"),
            ("ALIGNED", "1"),
        ],
    },
    // Phase-3C: Q4_K integer-MMQ GEMM. Mirrors the defines
    // llama.cpp's vulkan-shaders-gen passes for a non-MoE, non-coopmat
    // Q4_K mul_mmq build. Used by Forward::prefill_batch (TBD) and
    // microbenchmarks.
    ShaderJob {
        out_name: "mul_mmq_q4_k_f32.spv",
        entry_source: "mul_mmq.comp",
        defines: &[
            ("DATA_A_Q4_K", "1"),
            ("A_TYPE", "block_q4_K"),
            ("A_TYPE_PACKED32", "block_q4_K_packed32"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("ACC_TYPE", "float"),
        ],
    },
    // Q6_K integer-MMQ GEMM (mixed-quant: attn_v + ffn_down).
    // Q6_K only ships a packed16 variant in types.glsl — match it.
    ShaderJob {
        out_name: "mul_mmq_q6_k_f32.spv",
        entry_source: "mul_mmq.comp",
        defines: &[
            ("DATA_A_Q6_K", "1"),
            ("A_TYPE", "block_q6_K"),
            ("A_TYPE_PACKED16", "block_q6_K_packed16"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("ACC_TYPE", "float"),
        ],
    },
    // Quantize FP32 activations → block_q8_1_x4 — needed before each
    // mul_mmq dispatch since B is consumed as Q8_1.
    ShaderJob {
        out_name: "quantize_q8_1_f32.spv",
        entry_source: "quantize_q8_1.comp",
        defines: &[("QBLOCK_X4", "1")],
    },
];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("vk_shaders");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    // Be conservative: rebuild if any tracked shader source changes.
    for entry in fs::read_dir(&shader_dir).expect("read vk_shaders/") {
        let path = entry.expect("entry").path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext == "comp" || ext == "glsl" {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let compiler = Compiler::new().expect("shaderc Compiler::new");
    let mut total_bytes: usize = 0;

    for job in JOBS {
        let mut options = CompileOptions::new().expect("shaderc CompileOptions::new");
        options.set_source_language(SourceLanguage::GLSL);
        options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(OptimizationLevel::Performance);
        options.set_generate_debug_info();

        for (k, v) in job.defines {
            options.add_macro_definition(k, Some(v));
        }

        let include_root = shader_dir.clone();
        options.set_include_callback(move |requested, _kind, _from, _depth| {
            let path = include_root.join(requested);
            let content = fs::read_to_string(&path).map_err(|e| {
                format!("failed to read include '{}': {}", path.display(), e)
            })?;
            Ok(ResolvedInclude {
                resolved_name: path.to_string_lossy().into_owned(),
                content,
            })
        });

        let entry_path = shader_dir.join(job.entry_source);
        let source = fs::read_to_string(&entry_path)
            .unwrap_or_else(|e| panic!("read {}: {}", entry_path.display(), e));

        let artifact = compiler
            .compile_into_spirv(
                &source,
                ShaderKind::Compute,
                job.entry_source,
                "main",
                Some(&options),
            )
            .unwrap_or_else(|e| panic!("shaderc compile {} failed:\n{}", job.entry_source, e));

        let warnings = artifact.get_warning_messages();
        if !warnings.is_empty() {
            println!("cargo:warning=shaderc warnings for {}: {}", job.entry_source, warnings);
        }

        let spv_bytes = artifact.as_binary_u8();
        assert!(spv_bytes.len() % 4 == 0, "SPIR-V byte length must be % 4 == 0");
        assert!(
            spv_bytes.len() >= 4 && &spv_bytes[0..4] == [0x03, 0x02, 0x23, 0x07],
            "SPIR-V magic missing for {}",
            job.entry_source
        );

        let out_path = out_dir.join(job.out_name);
        fs::write(&out_path, spv_bytes)
            .unwrap_or_else(|e| panic!("write {}: {}", out_path.display(), e));

        total_bytes += spv_bytes.len();
        println!(
            "cargo:warning=compiled {} -> {} ({} bytes)",
            job.entry_source,
            job.out_name,
            spv_bytes.len()
        );
    }

    println!(
        "cargo:warning=total SPIR-V: {} bytes across {} shader(s)",
        total_bytes,
        JOBS.len()
    );
}
