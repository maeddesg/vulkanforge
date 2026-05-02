//! `VK_EXT_shader_float8` raw FFI shim.
//!
//! ash 0.38.0+1.3.281 (latest crates.io as of 2026-05) tracks
//! Vulkan SDK 1.3.281 and predates `VK_EXT_shader_float8` (added
//! in SDK 1.4). The system Vulkan headers ship the extension and
//! the RADV driver advertises both feature bits — the gap is purely
//! in our Rust bindings. This module fills the gap with ~30 LOC
//! of `repr(C)` structs and string constants. Replace with the
//! ash-provided types as soon as ash bumps to a SDK 1.4 generation.
//!
//! Reference: `vulkan-headers 1.4.341` ⇒ `vulkan_core.h`:
//!
//! ```c
//! VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT = 1000567000;
//! VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT = 1000491002;
//! VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT = 1000491003;
//!
//! typedef struct VkPhysicalDeviceShaderFloat8FeaturesEXT {
//!     VkStructureType    sType;
//!     void*              pNext;
//!     VkBool32           shaderFloat8;
//!     VkBool32           shaderFloat8CooperativeMatrix;
//! } VkPhysicalDeviceShaderFloat8FeaturesEXT;
//! ```

use ash::vk;
use std::ffi::{c_void, CStr};

/// Extension name as a NUL-terminated C string. Goes into
/// `VkDeviceCreateInfo::ppEnabledExtensionNames`.
pub const SHADER_FLOAT8_EXT_NAME: &CStr = c"VK_EXT_shader_float8";

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT`.
/// Note: the brief had `1000491000` here — that's the unrelated NV
/// cooperative-vector extension; the correct EXT FP8 sType is
/// `1000567000` per vulkan-headers 1.4.341.
pub const STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: vk::StructureType =
    vk::StructureType::from_raw(1000567000);

/// `VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT` — for cooperative-matrix
/// component types.
pub const COMPONENT_TYPE_FLOAT8_E4M3_EXT: vk::ComponentTypeKHR =
    vk::ComponentTypeKHR::from_raw(1000491002);
/// `VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT`.
pub const COMPONENT_TYPE_FLOAT8_E5M2_EXT: vk::ComponentTypeKHR =
    vk::ComponentTypeKHR::from_raw(1000491003);

/// Mirror of `VkPhysicalDeviceShaderFloat8FeaturesEXT`. C struct
/// layout (verified via `cc + offsetof`): 24 bytes total,
/// `sType`@0 / `pNext`@8 / `shaderFloat8`@16 /
/// `shaderFloat8CooperativeMatrix`@20.
///
/// The first two fields exactly match `VkBaseOutStructure`, which
/// is what ash's `ptr_chain_iter` casts to when walking `pNext`.
/// That makes it safe to use this struct via `push_next()` after
/// the `unsafe impl ExtendsDeviceCreateInfo` below.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PhysicalDeviceShaderFloat8FeaturesEXT {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub shader_float8: vk::Bool32,
    pub shader_float8_cooperative_matrix: vk::Bool32,
}

impl Default for PhysicalDeviceShaderFloat8FeaturesEXT {
    fn default() -> Self {
        Self {
            s_type: STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
            p_next: std::ptr::null_mut(),
            shader_float8: vk::FALSE,
            shader_float8_cooperative_matrix: vk::FALSE,
        }
    }
}

impl PhysicalDeviceShaderFloat8FeaturesEXT {
    pub fn shader_float8(mut self, enable: bool) -> Self {
        self.shader_float8 = if enable { vk::TRUE } else { vk::FALSE };
        self
    }

    pub fn shader_float8_cooperative_matrix(mut self, enable: bool) -> Self {
        self.shader_float8_cooperative_matrix = if enable { vk::TRUE } else { vk::FALSE };
        self
    }
}

// SAFETY: the first two fields (`s_type`: 4 bytes @ offset 0,
// `p_next`: 8 bytes @ offset 8) are layout-compatible with
// `VkBaseOutStructure`, which is what ash's `push_next()` casts
// to when walking the `pNext` chain. The struct is `repr(C)`,
// so field ordering and offsets are guaranteed to match the C
// VkPhysicalDeviceShaderFloat8FeaturesEXT verbatim.
unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceShaderFloat8FeaturesEXT {}

// =====================================================================
// CPU-side FP8 E4M3 conversion helpers — used by the smoke test to
// validate GPU `uintBitsToFloate4m3EXT(...)` behaviour bit-for-bit.

/// Decode a raw u8 bit pattern as IEEE-754 FP8 E4M3 ("FN" variant —
/// only one NaN encoding, no Inf, max value 448.0). Mirrors what
/// the GPU's `float(uintBitsToFloate4m3EXT(b))` produces.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;
    let sign_f = if sign != 0 { -1.0_f32 } else { 1.0_f32 };

    // The single NaN encoding: S.1111.111
    if exp == 0xF && mant == 0x7 {
        return sign_f * f32::NAN;
    }
    // Subnormal: value = sign * 2^-6 * mant/8
    if exp == 0 {
        return sign_f * (mant as f32 / 8.0) * 2.0_f32.powi(-6);
    }
    // Normal: value = sign * 2^(exp-7) * (1 + mant/8)
    let m_frac = 1.0 + (mant as f32) / 8.0;
    let exp_actual = (exp as i32) - 7;
    sign_f * m_frac * 2.0_f32.powi(exp_actual)
}

/// Encode an f32 as FP8 E4M3 ("FN" variant). Round-to-nearest-even
/// per IEEE-754 default rounding.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let ax = x.abs();
    if ax == 0.0 {
        return sign_bit;
    }
    // Largest E4M3 magnitude is 448.0 (S=0, E=15, M=110 = 0x7E).
    // Anything larger clamps to that (no Inf in E4M3 FN).
    if ax >= 448.0 {
        return sign_bit | 0x7E;
    }

    // f32 ↔ bits decomposition.
    let bits = ax.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32; // biased
    let f32_mant = bits & 0x007F_FFFF;          // 23-bit mantissa

    // Re-target: f32 bias 127 → e4m3 bias 7.
    let e4m3_exp = f32_exp - 127 + 7;

    if e4m3_exp >= 1 && e4m3_exp <= 15 {
        // Normal range. Round 23-bit mantissa to 3 bits.
        let mant3 = round_mantissa_to_3(f32_mant, 0);
        // Mantissa overflow → carry into exponent.
        if mant3 == 0x8 {
            let new_exp = e4m3_exp + 1;
            if new_exp == 16 {
                // Would be NaN encoding; clamp.
                return sign_bit | 0x7E;
            }
            return sign_bit | ((new_exp as u8) << 3);
        }
        return sign_bit | ((e4m3_exp as u8) << 3) | (mant3 as u8);
    }

    if e4m3_exp <= 0 {
        // Subnormal in E4M3. Value = (1 + mant/2^23) * 2^(f32_exp - 127).
        // Express as mant3 such that result = mant3/8 * 2^-6.
        // shift = 1 - e4m3_exp tells us how many extra bits beyond
        // the normal mantissa to drop.
        let shift = (1 - e4m3_exp) as u32;
        if shift > 23 {
            return sign_bit; // underflow → ±0
        }
        // Reconstruct full implicit-1 mantissa, then shift right by
        // `shift` and round 3 bits.
        let full_mant = 0x0080_0000 | f32_mant; // implicit 1
        let mant3 = round_mantissa_to_3(full_mant, shift);
        if mant3 == 0x8 {
            // Promotes to smallest normal: E=1, M=0
            return sign_bit | (1u8 << 3);
        }
        return sign_bit | (mant3 as u8);
    }

    // e4m3_exp > 15: clamp to max.
    sign_bit | 0x7E
}

/// Round a 23-bit f32 mantissa to a 3-bit FP8 mantissa using
/// round-to-nearest-even. `extra_shift` adds extra right-shifts
/// for the subnormal path. Returns 0..=8 (8 == carry).
fn round_mantissa_to_3(mant23: u32, extra_shift: u32) -> u32 {
    // We want the top 3 bits after shifting; total shift right is
    // 20 (23 → 3) plus extra. Round bit is at (shift-1).
    let shift = 20u32 + extra_shift;
    if shift >= 32 {
        return 0;
    }
    let truncated = mant23 >> shift;
    let round_bit = (mant23 >> (shift - 1)) & 1;
    let sticky = if shift >= 2 {
        (mant23 & ((1u32 << (shift - 1)) - 1)) != 0
    } else {
        false
    };
    let mut rounded = truncated;
    // Round-to-nearest-even.
    if round_bit == 1 && (sticky || (truncated & 1) == 1) {
        rounded += 1;
    }
    rounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_size_matches_c() {
        // C `sizeof(VkPhysicalDeviceShaderFloat8FeaturesEXT) == 24`,
        // verified at sprint-write time via cc + offsetof.
        assert_eq!(std::mem::size_of::<PhysicalDeviceShaderFloat8FeaturesEXT>(), 24);
    }

    #[test]
    fn e4m3_round_trip_zero() {
        // ±0.0 round-trips bit-for-bit.
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80);
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0);
    }

    #[test]
    fn e4m3_one_round_trips() {
        // 1.0 in E4M3 = S=0, E=7, M=0 = 0b0_0111_000 = 0x38.
        assert_eq!(f32_to_fp8_e4m3(1.0), 0x38);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
    }

    #[test]
    fn e4m3_max_value() {
        // E4M3 max = 448.0 = S=0, E=15, M=110 = 0b0_1111_110 = 0x7E.
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(f32_to_fp8_e4m3(448.0), 0x7E);
        // Anything above clamps (no Inf in E4M3 FN).
        assert_eq!(f32_to_fp8_e4m3(1000.0), 0x7E);
    }

    #[test]
    fn e4m3_decode_table_known_values() {
        // Spot-check a few canonical values that match
        // `__nv_fp8_e4m3` and the GLSL `floate4m3_t` reference.
        // S=0, E=8, M=0  = 2.0
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        // S=0, E=9, M=0  = 4.0
        assert_eq!(fp8_e4m3_to_f32(0x48), 4.0);
        // S=1, E=7, M=0 = -1.0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        // S=0, E=6, M=0 = 0.5
        assert_eq!(fp8_e4m3_to_f32(0x30), 0.5);
        // S=0, E=7, M=4 = 1.5  (1 + 4/8)
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
    }
}
