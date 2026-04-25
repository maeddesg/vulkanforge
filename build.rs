//! Build-time GLSL → SPIR-V compilation for VulkanForge.
//!
//! Phase 1, Step 1.1: compiles the Q4_K GEMV shader (and only that one)
//! to a SPIR-V blob in `OUT_DIR`, ready to be embedded via
//! `include_bytes!`. Defines mirror llama.cpp's `vulkan-shaders-gen`
//! invocation for the `mul_mat_vec_q4_k_f32_f32` variant — see
//! results/phase1_step_1.0_shader_analysis.md §3.3.

use std::env;
use std::fs;
use std::path::PathBuf;

use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeType, OptimizationLevel, ResolvedInclude,
    ShaderKind, SourceLanguage, TargetEnv,
};

struct ShaderJob {
    out_name: &'static str,
    entry_source: &'static str,
    defines: &'static [(&'static str, &'static str)],
}

const JOBS: &[ShaderJob] = &[ShaderJob {
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
}];

const TRACKED_FILES: &[&str] = &[
    "vk_shaders/mul_mat_vec_q4_k.comp",
    "vk_shaders/mul_mat_vec_base.glsl",
    "vk_shaders/mul_mat_vec_iface.glsl",
    "vk_shaders/types.glsl",
];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("vk_shaders");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    for f in TRACKED_FILES {
        println!("cargo:rerun-if-changed={}", f);
    }

    let compiler = Compiler::new().expect("shaderc Compiler::new failed");

    for job in JOBS {
        let mut options = CompileOptions::new().expect("shaderc CompileOptions::new failed");
        options.set_source_language(SourceLanguage::GLSL);
        options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(OptimizationLevel::Performance);
        options.set_generate_debug_info();

        for (k, v) in job.defines {
            options.add_macro_definition(k, Some(v));
        }

        // #include resolution: relative paths resolve against vk_shaders/.
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
        assert!(spv_bytes.len() % 4 == 0, "SPIR-V byte length must be multiple of 4");
        assert!(
            spv_bytes.len() >= 4 && &spv_bytes[0..4] == [0x03, 0x02, 0x23, 0x07],
            "SPIR-V magic number missing"
        );

        let out_path = out_dir.join(job.out_name);
        fs::write(&out_path, spv_bytes)
            .unwrap_or_else(|e| panic!("write {}: {}", out_path.display(), e));

        println!(
            "cargo:warning=compiled {} -> {} ({} bytes, {} u32 words)",
            job.entry_source,
            job.out_name,
            spv_bytes.len(),
            spv_bytes.len() / 4
        );
    }

    // Touch include type so unused-import warnings stay quiet under future tweaks.
    let _ = IncludeType::Relative;
}
