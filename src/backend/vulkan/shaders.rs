//! Pre-compiled SPIR-V blobs, embedded at build time.
//!
//! The `build.rs` script produces one `.spv` per entry per variant under
//! `$OUT_DIR/`; we re-export the raw byte slices and a small helper that
//! returns properly-aligned `u32` words for `vk::ShaderModuleCreateInfo`.

/// Q4_K GEMV, f32 input × f32 output, default specialization constants.
/// See results/phase1_step_1.0_shader_analysis.md §5 for the configuration.
pub const MUL_MAT_VEC_Q4_K_F32_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_k_f32_f32.spv"));

/// Decode a SPIR-V byte blob into u32 words. Vulkan consumes SPIR-V as
/// `&[u32]`; `include_bytes!` only gives us a `&[u8]` whose alignment is
/// not guaranteed, so we copy into an owned, naturally-aligned `Vec<u32>`.
pub fn spv_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V length not 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
