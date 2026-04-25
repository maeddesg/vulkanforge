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
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .storage_buffer8_bit_access(true)
            .shader_int8(true);
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .push_next(&mut features2)
            .push_next(&mut features11)
            .push_next(&mut features12);

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
