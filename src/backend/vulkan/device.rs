//! Minimal Vulkan device init for VulkanForge.
//!
//! Phase 0 / smoke-test scope: enumerate physical devices, pick the
//! first discrete GPU, locate a compute-capable queue family, and
//! hand back the device + queue. No swap-chain (compute-only), no
//! validation layers (yet), no extensions beyond the implicit core.
//!
//! The `Drop` impl tears the device + instance down in the right
//! order so a smoke-run leaves no orphan handles even on early-exit
//! paths.

use ash::vk;

pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
}

impl VulkanDevice {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"VulkanForge")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None)? };

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
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
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

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device =
            unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        println!("VulkanForge: Compute Queue Family = {queue_family_index}");

        Ok(VulkanDevice {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Tear down in reverse construction order. The compute queue
        // does not need an explicit destroy call; it is implicitly
        // owned by the logical device.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
