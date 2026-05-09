//! Minimal Vulkan device init for VulkanForge.
//!
//! Phase 1 scope: enumerate physical devices, pick the first discrete
//! GPU, locate a compute-capable queue family, enable validation
//! layers + the storage / int8 features that the Q4_K GEMV shader
//! needs, and wire up a debug messenger so any later validation error
//! lands in stderr instead of being silent.
//!
//! The `Drop` impl tears messenger → device → instance down in the
//! right order so a smoke-run leaves no orphan handles.

use std::ffi::{CStr, c_void};

use ash::ext::debug_utils;
use ash::vk;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    /// True when the driver advertises `VK_EXT_shader_float8` *and*
    /// `shaderFloat8CooperativeMatrix` — i.e. native FP8 WMMA is
    /// usable. `Forward::new` copies this into
    /// `Forward::native_fp8_wmma`, which the FP8 GEMM routing in
    /// `runs.rs` reads (Sprint 47B; before that the routing went
    /// through a `VF_FP8_NATIVE_WMMA` env var).
    pub native_fp8_wmma: bool,
    debug_loader: Option<debug_utils::Instance>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanDevice {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"VulkanForge")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Validation: enabled if the layer is installed. Missing layer
        // is not fatal (CI / minimal images), only loud.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let validation_available = available_layers.iter().any(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
            name == VALIDATION_LAYER
        });
        if !validation_available {
            eprintln!(
                "VulkanForge: VK_LAYER_KHRONOS_validation not installed — running without validation"
            );
        }

        let layer_ptrs: Vec<*const i8> = if validation_available {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let mut instance_extensions: Vec<*const i8> = Vec::new();
        if validation_available {
            instance_extensions.push(debug_utils::NAME.as_ptr());
        }

        // Attach a debug messenger to the create-info chain so the
        // very first instance-level errors (if any) land in our
        // callback instead of going to default loader logging.
        let mut debug_create_info = debug_messenger_create_info();

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&instance_extensions);
        if validation_available {
            instance_create_info = instance_create_info.push_next(&mut debug_create_info);
        }

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        let (debug_loader, debug_messenger) = if validation_available {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let messenger =
                unsafe { loader.create_debug_utils_messenger(&debug_messenger_create_info(), None)? };
            (Some(loader), messenger)
        } else {
            (None, vk::DebugUtilsMessengerEXT::null())
        };

        // Pick the first discrete GPU. If none is present, fall back
        // to whatever physical device is enumerated first — handy on
        // CI / iGPU-only setups, but we log it loud.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err("No Vulkan-capable physical device found".into());
        }

        let physical_device = physical_devices
            .iter()
            .copied()
            .find(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "VulkanForge: no DISCRETE_GPU found, falling back to first available device"
                );
                physical_devices[0]
            });

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        println!(
            "VulkanForge: GPU = {}",
            name.to_str().unwrap_or("<non-utf8 device name>")
        );

        // First queue family with COMPUTE bit. On AMD the same family
        // is usually GRAPHICS+COMPUTE+TRANSFER; we only need COMPUTE.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("No compute-capable queue family")? as u32;

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        // Feature chain mandated by the Q4_K GEMV shader (see
        // results/phase1_step_1.1_shader_extraction.md §4 "Capabilities"):
        //   StorageBuffer16BitAccess  — Vulkan 1.1 feature
        //   StorageBuffer8BitAccess   — Vulkan 1.2 feature
        //   shaderInt8                — Vulkan 1.2 feature
        // Phase 3C adds Vulkan 1.3 `shaderIntegerDotProduct` so the
        // mul_mmq.comp SPIR-V (compile-probe in this phase, full
        // dispatch in 3D) can declare its `DotProduct` capabilities
        // cleanly — RADV/gfx1201 reports both 8-bit accelerated paths
        // available.
        // Sprint 14A — subgroupSizeControl + computeFullSubgroups are
        // both Vulkan 1.3 core. Required for
        // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo +
        // PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
        // which the GEMV pipelines opt into to enable subgroup-arithmetic
        // reductions (Path A in mul_mat_vec_base.glsl) at Sprint 14B.
        // RDNA4 advertises both as supported (vulkaninfo confirms
        // subgroupSizeControl=true, computeFullSubgroups=true,
        // requiredSubgroupSizeStages=COMPUTE_BIT, min=32, max=64).
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .shader_integer_dot_product(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        // Sprint 16B — shaderFloat16 + vulkanMemoryModel are declared by
        // the coopmat / flash-attn SPIR-V (Float16 + VulkanMemoryModel
        // capabilities). RADV runs the binary anyway but the validation
        // layer correctly flags them as missing-feature errors. Both are
        // 1.2 core features.
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .storage_buffer8_bit_access(true)
            .shader_int8(true)
            .shader_float16(true)
            .vulkan_memory_model(true);
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true);
        // shaderInt16 is also pulled in by mul_mmq's i16 unpack helpers.
        // It's a base 1.0 feature so it lives on PhysicalDeviceFeatures,
        // not in the 1.1/1.2/1.3 extension chains.
        let core_features = vk::PhysicalDeviceFeatures::default().shader_int16(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(core_features);

        // Sprint 16B — VK_KHR_cooperative_matrix is the device extension
        // that backs the CooperativeMatrixKHR SPIR-V capability declared
        // by every coopmat GEMM / flash_attn shader. We must both enable
        // the extension and request `cooperativeMatrix` on the feature
        // struct, otherwise the validation layer (correctly) errors.
        //
        // Sprint 18-M1 — VK_EXT_shader_float8 lets shaders use the
        // native E4M3 / E5M2 types and the FP8 cooperative-matrix
        // path. The extension is opt-in: only enabled when
        // `VULKANFORGE_ENABLE_FP8=1`, since ash 0.38 doesn't ship
        // bindings yet (we use a raw FFI shim) and we don't want
        // the smoke-test path to affect every chat/bench run. The
        // chain is built the same way the existing features are.
        // Sprint 18A — VULKANFORGE_KV_FP8=1 implies the device-feature
        // opt-in too, since the FP8 KV-cache shaders declare Float8EXT
        // and need shaderFloat8 enabled on the device.
        let fp8_opt_in = std::env::var("VULKANFORGE_ENABLE_FP8")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || std::env::var("VULKANFORGE_KV_FP8")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

        let mut device_extensions: Vec<*const i8> =
            vec![vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr()];

        // Sprint 42C / v0.3.12 — probe `VK_EXT_shader_float8` availability
        // before requesting it. Pre-v0.3.12 device.rs pushed the extension
        // unconditionally on `fp8_opt_in`, which crashed with
        // `VK_ERROR_EXTENSION_NOT_PRESENT` on drivers that didn't
        // advertise it. Probing makes `VF_FP8=auto` safe everywhere
        // (no native WMMA, but no hard crash either) and feeds
        // `native_fp8_wmma` so the FP8 GEMM routing reflects the
        // device's *actual* capability (Sprint 47B made the routing
        // capability-driven; the legacy `VF_FP8_NATIVE_WMMA` env-var
        // was removed).
        let fp8_ext_available = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        let name = std::ffi::CStr::from_ptr(e.extension_name.as_ptr());
                        name == crate::backend::vulkan::fp8_ext::SHADER_FLOAT8_EXT_NAME
                    })
                })
                .unwrap_or(false)
        };
        let push_fp8_ext = fp8_opt_in && fp8_ext_available;
        if push_fp8_ext {
            device_extensions.push(crate::backend::vulkan::fp8_ext::SHADER_FLOAT8_EXT_NAME.as_ptr());
        } else if fp8_opt_in && !fp8_ext_available {
            eprintln!(
                "VulkanForge: VK_EXT_shader_float8 not advertised by driver — \
                 falling back to BF16 conversion path. Update to a driver \
                 that exposes shaderFloat8CooperativeMatrix for native FP8 WMMA."
            );
        }

        // Sprint 39 — `VK_KHR_shader_bfloat16`. The BF16-narrow FP8 GEMM
        // cousins (`mul_coopmat_fp8_bn32(.comp|_blockwise.comp)`,
        // `_multi_wg.comp`, `_naive.comp`) declare BFloat16TypeKHR +
        // BFloat16CooperativeMatrixKHR capabilities. RADV runs the
        // pipelines without the extension being enabled, but
        // validation layers fire VUID-…-pCode-08742 on every
        // shader-module create. Enable the extension when the runtime
        // advertises it and silence the warnings. We probe the
        // available device extensions explicitly so runtimes without
        // it keep working untouched (they just see the warnings as
        // before).
        let bfloat16_available = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        let name = std::ffi::CStr::from_ptr(e.extension_name.as_ptr());
                        name == crate::backend::vulkan::bfloat16_ext::SHADER_BFLOAT16_KHR_NAME
                    })
                })
                .unwrap_or(false)
        };
        if bfloat16_available {
            device_extensions.push(
                crate::backend::vulkan::bfloat16_ext::SHADER_BFLOAT16_KHR_NAME.as_ptr(),
            );
        }

        let mut coopmat_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut fp8_features =
            crate::backend::vulkan::fp8_ext::PhysicalDeviceShaderFloat8FeaturesEXT::default()
                .shader_float8(true)
                .shader_float8_cooperative_matrix(true);
        let mut bfloat16_features =
            crate::backend::vulkan::bfloat16_ext::PhysicalDeviceShaderBfloat16FeaturesKHR::default()
                .shader_bfloat16_type(true)
                .shader_bfloat16_cooperative_matrix(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut features13)
            .push_next(&mut coopmat_features);
        if push_fp8_ext {
            device_create_info = device_create_info.push_next(&mut fp8_features);
        }
        if bfloat16_available {
            device_create_info = device_create_info.push_next(&mut bfloat16_features);
        }

        let device =
            unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        println!("VulkanForge: Compute Queue Family = {queue_family_index}");
        if validation_available {
            println!("VulkanForge: validation layer ACTIVE");
        }

        Ok(VulkanDevice {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
            native_fp8_wmma: push_fp8_ext,
            debug_loader,
            debug_messenger,
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Tear down in reverse construction order. The compute queue
        // does not need an explicit destroy call; it is implicitly
        // owned by the logical device. Messenger must die before the
        // instance that produced it.
        unsafe {
            self.device.destroy_device(None);
            if let Some(loader) = &self.debug_loader {
                if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                    loader.destroy_debug_utils_messenger(self.debug_messenger, None);
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn debug_messenger_create_info<'a>() -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Sev;
    use vk::DebugUtilsMessageTypeFlagsEXT as Ty;
    // INFO is firehose-level on Linux loaders (every device-discovery
    // pass, every layer-manifest find, etc.); keep only WARN/ERROR
    // so any real validation issue stands out.
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(Sev::WARNING | Sev::ERROR)
        .message_type(Ty::GENERAL | Ty::VALIDATION | Ty::PERFORMANCE)
        .pfn_user_callback(Some(vk_debug_callback))
}

unsafe extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { &*callback_data };
    let message = if data.p_message.is_null() {
        "<null message>"
    } else {
        unsafe { CStr::from_ptr(data.p_message) }
            .to_str()
            .unwrap_or("<non-utf8 message>")
    };
    let sev = match severity {
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) => "ERROR",
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) => "WARN",
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) => "INFO",
        _ => "VERBOSE",
    };
    let ty = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "validation"
    } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "perf"
    } else {
        "general"
    };
    eprintln!("[vk {sev}/{ty}] {message}");
    vk::FALSE
}
