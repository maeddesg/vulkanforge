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
