//! `VK_KHR_shader_bfloat16` raw FFI shim.
//!
//! ash 0.38.0+1.3.281 (latest crates.io as of 2026-05) tracks
//! Vulkan SDK 1.3.281 and predates `VK_KHR_shader_bfloat16` (added
//! in SDK 1.4). The system Vulkan headers ship the extension and
//! the RADV driver advertises all three feature bits — the gap is
//! purely in our Rust bindings, identical situation to `fp8_ext`.
//!
//! Why we care: VulkanForge's BF16-narrow FP8 GEMM cousins
//! (`mul_coopmat_fp8_bn32.comp`, `_blockwise.comp`, `_multi_wg.comp`,
//! `_naive.comp`) declare `OpCapability BFloat16TypeKHR` and
//! `OpCapability BFloat16CooperativeMatrixKHR`. Without the device
//! extension enabled, validation layers fire `VUID-…-pCode-08742`
//! warnings on every pipeline build (RADV runs the pipeline anyway,
//! but the noise is unhelpful).
//!
//! Reference: `vulkan-headers 1.4.341` ⇒ `vulkan_core.h`:
//!
//! ```c
//! #define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME "VK_KHR_shader_bfloat16"
//! VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR = 1000141000;
//!
//! typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
//!     VkStructureType    sType;
//!     void*              pNext;
//!     VkBool32           shaderBFloat16Type;
//!     VkBool32           shaderBFloat16DotProduct;
//!     VkBool32           shaderBFloat16CooperativeMatrix;
//! } VkPhysicalDeviceShaderBfloat16FeaturesKHR;
//! ```

use ash::vk;
use std::ffi::{c_void, CStr};

/// Extension name as a NUL-terminated C string. Goes into
/// `VkDeviceCreateInfo::ppEnabledExtensionNames`.
pub const SHADER_BFLOAT16_KHR_NAME: &CStr = c"VK_KHR_shader_bfloat16";

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR`.
pub const STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: vk::StructureType =
    vk::StructureType::from_raw(1000141000);

/// Mirror of `VkPhysicalDeviceShaderBfloat16FeaturesKHR`. Layout matches
/// `VkBaseOutStructure` for the first two fields, so it can be safely
/// pushed onto the device-create `pNext` chain via the
/// `ExtendsDeviceCreateInfo` impl below.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PhysicalDeviceShaderBfloat16FeaturesKHR {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub shader_bfloat16_type: vk::Bool32,
    pub shader_bfloat16_dot_product: vk::Bool32,
    pub shader_bfloat16_cooperative_matrix: vk::Bool32,
}

impl Default for PhysicalDeviceShaderBfloat16FeaturesKHR {
    fn default() -> Self {
        Self {
            s_type: STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        }
    }
}

impl PhysicalDeviceShaderBfloat16FeaturesKHR {
    pub fn shader_bfloat16_type(mut self, enable: bool) -> Self {
        self.shader_bfloat16_type = if enable { vk::TRUE } else { vk::FALSE };
        self
    }

    pub fn shader_bfloat16_cooperative_matrix(mut self, enable: bool) -> Self {
        self.shader_bfloat16_cooperative_matrix = if enable { vk::TRUE } else { vk::FALSE };
        self
    }
}

// SAFETY: `s_type` (4 B @ offset 0) + `p_next` (8 B @ offset 8) match
// `VkBaseOutStructure`, which is what ash's `push_next()` casts to
// when walking the `pNext` chain. `repr(C)` pins the field ordering
// and offsets to the C VkPhysicalDeviceShaderBfloat16FeaturesKHR.
unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceShaderBfloat16FeaturesKHR {}
