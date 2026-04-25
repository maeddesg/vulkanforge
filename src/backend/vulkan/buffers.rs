//! Thin GPU-buffer wrapper around `gpu_allocator::vulkan::Allocator`.
//!
//! Phase 1 / Step 1.3: each `GpuBuffer` owns one `VkBuffer` + its
//! suballocation. Memory location follows gpu-allocator's enum
//! (`GpuOnly` = DEVICE_LOCAL, `CpuToGpu` = staging-upload,
//! `GpuToCpu` = readback). Mapped pointers are exposed via
//! `mapped_*` for host-visible buffers; otherwise the data has to
//! flow through a staging buffer + `vkCmdCopyBuffer`.
//!
//! Cleanup is explicit (`destroy(self, device, allocator)`); there
//! is no `Drop` impl because the buffer does not hold a reference
//! to either the device or the allocator. Teardown order is:
//! all buffers → drop(allocator) → drop(VulkanDevice).

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

pub struct GpuBuffer {
    pub handle: vk::Buffer,
    pub size: u64,
    allocation: Option<Allocation>,
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.create_buffer(&info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };

        let allocation_result = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation_result {
            Ok(a) => a,
            Err(e) => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(Box::new(e));
            }
        };

        unsafe { device.bind_buffer_memory(handle, allocation.memory(), allocation.offset())? };

        Ok(Self {
            handle,
            size,
            allocation: Some(allocation),
        })
    }

    /// Write up to `bytes.len()` host-visible bytes to the buffer.
    /// Errors if the buffer is GPU-only (not host-mappable).
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let alloc = self.allocation.as_mut().ok_or("buffer already freed")?;
        let slice = alloc
            .mapped_slice_mut()
            .ok_or("buffer is not host-visible")?;
        if bytes.len() > slice.len() {
            return Err(format!(
                "write of {} bytes exceeds buffer capacity {}",
                bytes.len(),
                slice.len()
            )
            .into());
        }
        slice[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Read host-visible bytes (typically for output readback).
    pub fn read_bytes(&self) -> Result<&[u8], Box<dyn std::error::Error>> {
        let alloc = self.allocation.as_ref().ok_or("buffer already freed")?;
        alloc
            .mapped_slice()
            .ok_or_else(|| "buffer is not host-visible".into())
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            // free is best-effort; if it fails we still need to destroy
            // the VkBuffer to avoid a leak the validation layer would
            // catch on instance teardown.
            let _ = allocator.free(alloc);
        }
        unsafe { device.destroy_buffer(self.handle, None) };
    }
}
