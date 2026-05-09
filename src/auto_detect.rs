//! `VF_FP8=auto` runtime detection (v0.3.12).
//!
//! Replaces the three-flag opt-in surface
//! (`VULKANFORGE_ENABLE_FP8` / `VF_FP8_NATIVE_WMMA` /
//! `VF_CPU_LM_HEAD`) with a single env var and an auto-detection
//! chain. The old flags still work as explicit overrides; this
//! module decides what they should be when the user picks `auto`.
//!
//! Detection is split into two phases because half the inputs come
//! from the model directory (config.json, fast) and the other half
//! from the GPU driver (FP8 cooperative-matrix support, only known
//! after `VulkanDevice::new()`).
//!
//! Phase A — pre-device, in `main()`:
//! * `detect_fp8_model_dir(model_path)` reads `config.json` and looks
//!   for `quantization_config.quant_method == "fp8"` (or
//!   `"compressed-tensors"` with `fmt == "e4m3"`). On a hit we set
//!   `VULKANFORGE_ENABLE_FP8=1` so the existing device-init path
//!   pushes `VK_EXT_shader_float8` into the device-create chain.
//!
//! Phase B — post-device, before `Forward::new()`:
//! * `apply_post_device(...)` reads back `VulkanDevice::native_fp8_wmma`,
//!   the AVX-512 capability of the host CPU, and the model's parameter
//!   count. If `auto`, it sets `VF_FP8_NATIVE_WMMA=1` when the device
//!   actually advertises `shaderFloat8CooperativeMatrix`, and
//!   `VF_CPU_LM_HEAD=1` when the model is ≥ 12 B parameters and the
//!   host has AVX-512F+BW+VL. Smaller models stay on the GPU lm_head
//!   because CPU offload is ~32 % slower for 8 B (it's a VRAM-savings
//!   feature, not a speed feature, below 12 B).
//!
//! Backward-compat: if the user explicitly sets `VF_FP8_NATIVE_WMMA=0`
//! / `VF_CPU_LM_HEAD=0` / `VF_FP8_NATIVE_WMMA=1` / `VF_CPU_LM_HEAD=1`,
//! that wins regardless of the auto decision. Same for the legacy
//! `VULKANFORGE_ENABLE_FP8` flag — see `parse_fp8_mode()`.

use std::path::Path;

use crate::hf_config::HfConfig;

/// Tri-state for the central `VF_FP8` env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Mode {
    /// `VF_FP8=0` (or unset on a GGUF path with no legacy flag).
    Off,
    /// `VF_FP8=1` (or legacy `VULKANFORGE_ENABLE_FP8=1`).
    On,
    /// `VF_FP8=auto` — run the detection chain.
    Auto,
}

/// Parse `VF_FP8`. Falls back to the legacy flag when `VF_FP8` is
/// unset so v0.3.10 invocations keep working untouched.
pub fn parse_fp8_mode() -> Fp8Mode {
    match std::env::var("VF_FP8").as_deref() {
        Ok(v) if v.eq_ignore_ascii_case("auto") => Fp8Mode::Auto,
        Ok("1") | Ok("true") | Ok("True") | Ok("TRUE") => Fp8Mode::On,
        Ok("0") | Ok("false") | Ok("False") | Ok("FALSE") => Fp8Mode::Off,
        _ => match std::env::var("VULKANFORGE_ENABLE_FP8").as_deref() {
            Ok("1") | Ok("true") => Fp8Mode::On,
            _ => Fp8Mode::Off,
        },
    }
}

/// Returns true when the model directory carries an FP8 SafeTensors
/// payload. Two upstream conventions need to be handled:
///
/// * `quant_method = "fp8"` — Qwen3-FP8, DeepSeek-V3-FP8. Has
///   `fmt = "e4m3"` and `weight_block_size`.
/// * `quant_method = "compressed-tensors"` — neuralmagic FP8
///   (Llama-3.1-8B-FP8 per-tensor, Qwen2.5-14B-FP8 per-channel).
///   No top-level `fmt` field; the FP8 signal lives in
///   `format = "naive-quantized" | "float-quantized"` (and the
///   nested `config_groups.weights.type = "float"` with 8 bits).
///   We check `format` because that's the field already present in
///   our `QuantizationConfig` struct. INT8 W8A8 compressed-tensors
///   models use `format = "int-quantized"`, so they're correctly
///   excluded.
///
/// A missing `config.json` (GGUF, plain SafeTensors) returns `false`,
/// so the auto path is a no-op for non-FP8 inputs.
pub fn detect_fp8_model_dir(model_path: &Path) -> bool {
    if !model_path.is_dir() {
        return false;
    }
    let cfg = match HfConfig::from_dir(model_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let Some(qc) = cfg.quantization_config.as_ref() else {
        return false;
    };
    let method = qc.quant_method.as_deref().unwrap_or("");
    if method.eq_ignore_ascii_case("fp8") {
        return true;
    }
    if method.eq_ignore_ascii_case("compressed-tensors") {
        let format = qc.format.as_deref().unwrap_or("");
        return format.eq_ignore_ascii_case("naive-quantized")
            || format.eq_ignore_ascii_case("float-quantized");
    }
    false
}

/// Quick parameter-count estimate from `config.json`. The transformer
/// rule of thumb (~12 × hidden² per layer + vocab × hidden) is good
/// to within ~10 % across the 7–70 B range — enough to draw the
/// 12 B threshold cleanly. Returns 0.0 if `config.json` is missing
/// (GGUF path / unknown size → never trigger auto CPU lm_head).
pub fn estimate_model_size_billions(model_path: &Path) -> f32 {
    if !model_path.is_dir() {
        return 0.0;
    }
    let cfg = match HfConfig::from_dir(model_path) {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let h = cfg.hidden_size as f64;
    let l = cfg.num_hidden_layers as f64;
    let v = cfg.vocab_size as f64;
    let i = cfg.intermediate_size as f64;
    // attn ≈ 4 h², ffn ≈ 3 h i (gate+up+down for SwiGLU, 2 h i for plain FFN).
    // Use 3 h i as the upper bound for SwiGLU stacks; it dominates.
    let per_layer = 4.0 * h * h + 3.0 * h * i;
    let params = l * per_layer + 2.0 * v * h; // embed + lm_head (tied or not)
    (params / 1e9) as f32
}

/// Returns true on hosts with AVX-512F + BW + VL — the three sub-ISAs
/// the v0.3.10 CPU `lm_head` AVX-512 GEMV depends on. Zen 4
/// (7945HX / 9950X / EPYC Genoa+) and Sapphire Rapids / Ice Lake
/// Server / Tiger Lake-U all qualify; older Skylake-X advertises the
/// foundation only, so we check all three explicitly.
#[cfg(target_arch = "x86_64")]
pub fn detect_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn detect_avx512() -> bool {
    false
}

/// Threshold above which we auto-enable CPU `lm_head` offload. Below
/// this, the GPU `lm_head` is faster (8 B is ~32 % faster on GPU);
/// above this, DDR5 + AVX-512 wins on tok/s and saves ~970 MB of
/// VRAM. The 12 B value is conservative — Qwen2.5-14B (14 B) and
/// Mistral-Small-24B-Instruct (24 B) both clear it.
pub const CPU_LM_HEAD_THRESHOLD_B: f32 = 12.0;

/// Outcome of Phase A (pre-device). Carried into Phase B so we can
/// log what we saw without re-reading `config.json`.
#[derive(Debug, Clone)]
pub struct PhaseAResult {
    pub mode: Fp8Mode,
    pub is_fp8_model: bool,
    pub model_params_billions: f32,
    pub avx512: bool,
}

/// Phase A — runs *before* `VulkanDevice::new()`. Inspects
/// `config.json`, sets `VULKANFORGE_ENABLE_FP8=1` if `auto` resolved
/// `is_fp8_model = true`, and returns a snapshot to feed Phase B.
pub fn apply_pre_device(model_path: &Path) -> PhaseAResult {
    let mode = parse_fp8_mode();
    let is_fp8 = detect_fp8_model_dir(model_path);
    let size = estimate_model_size_billions(model_path);
    let avx = detect_avx512();

    let enable_fp8 = match mode {
        Fp8Mode::Auto => is_fp8,
        Fp8Mode::On => true,
        Fp8Mode::Off => false,
    };
    if enable_fp8 && std::env::var("VULKANFORGE_ENABLE_FP8").is_err() {
        // SAFETY: env mutation in single-threaded `main` before any
        // Vulkan / Forward init. The legacy flag readers in device.rs
        // / loader.rs / forward.rs see this on their first read.
        unsafe { std::env::set_var("VULKANFORGE_ENABLE_FP8", "1") };
    }

    PhaseAResult {
        mode,
        is_fp8_model: is_fp8,
        model_params_billions: size,
        avx512: avx,
    }
}

/// Phase B — runs *after* `VulkanDevice::new()`. Takes the device's
/// real `native_fp8_wmma` capability (so we don't promise native
/// WMMA on a driver that doesn't advertise the FP8 cooperative-matrix
/// extension) and writes the legacy flags that `forward.rs` /
/// `loader.rs` will read next.
pub fn apply_post_device(phase_a: &PhaseAResult, native_fp8_wmma: bool) {
    if phase_a.mode != Fp8Mode::Auto {
        return; // Explicit On/Off — user is in charge.
    }
    if !phase_a.is_fp8_model {
        return; // GGUF path. Nothing to do.
    }

    // VF_FP8_NATIVE_WMMA: respect explicit override, else auto-decide.
    if std::env::var("VF_FP8_NATIVE_WMMA").is_err() && native_fp8_wmma {
        unsafe { std::env::set_var("VF_FP8_NATIVE_WMMA", "1") };
    }

    // VF_CPU_LM_HEAD: respect explicit override, else auto-decide for ≥ 12 B + AVX-512.
    if std::env::var("VF_CPU_LM_HEAD").is_err()
        && phase_a.avx512
        && phase_a.model_params_billions >= CPU_LM_HEAD_THRESHOLD_B
    {
        unsafe { std::env::set_var("VF_CPU_LM_HEAD", "1") };
    }
}

/// Pretty user-facing summary, printed after Phase B. Shows the raw
/// detection outputs *and* the resolved env vars so a "why did it
/// pick X" question is answerable from the chat banner alone.
pub fn print_summary(phase_a: &PhaseAResult, native_fp8_wmma: bool) {
    if phase_a.mode != Fp8Mode::Auto {
        return;
    }
    let on = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
    println!("VF_FP8=auto detected:");
    println!("  FP8 model:      {}", if phase_a.is_fp8_model { "yes" } else { "no (GGUF)" });
    println!(
        "  Native WMMA:    {}",
        if native_fp8_wmma {
            "yes (VK_EXT_shader_float8)"
        } else {
            "no (BF16 conversion path)"
        }
    );
    println!("  AVX-512:        {}", if phase_a.avx512 { "yes" } else { "no" });
    println!("  Model size:     ~{:.1} B params", phase_a.model_params_billions);
    println!("  → Native WMMA:  {}", if on("VF_FP8_NATIVE_WMMA") { "ON" } else { "OFF" });
    println!(
        "  → CPU lm_head:  {}",
        if on("VF_CPU_LM_HEAD") {
            if phase_a.model_params_billions >= CPU_LM_HEAD_THRESHOLD_B {
                "ON (auto: ≥ 12 B + AVX-512)"
            } else {
                "ON (explicit override)"
            }
        } else {
            "OFF"
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fp8_mode_auto_wins() {
        unsafe {
            std::env::set_var("VF_FP8", "auto");
            std::env::set_var("VULKANFORGE_ENABLE_FP8", "0");
        }
        assert_eq!(parse_fp8_mode(), Fp8Mode::Auto);
        unsafe {
            std::env::remove_var("VF_FP8");
            std::env::remove_var("VULKANFORGE_ENABLE_FP8");
        }
    }

    #[test]
    fn parse_fp8_mode_legacy_fallback() {
        unsafe {
            std::env::remove_var("VF_FP8");
            std::env::set_var("VULKANFORGE_ENABLE_FP8", "1");
        }
        assert_eq!(parse_fp8_mode(), Fp8Mode::On);
        unsafe { std::env::remove_var("VULKANFORGE_ENABLE_FP8") };
    }

    #[test]
    fn estimate_size_handles_missing_dir() {
        let path = std::path::Path::new("/tmp/__nonexistent_vf_test_dir__");
        assert_eq!(estimate_model_size_billions(path), 0.0);
        assert!(!detect_fp8_model_dir(path));
    }
}
