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
    /// Sprint B Phase 1 — when `true`, this GpuBuffer is a non-owning
    /// view into a shared backing buffer (a bucket). `destroy()` then
    /// skips both `allocator.free()` AND `device.destroy_buffer()` to
    /// avoid double-free on the shared handle. The bucket's own
    /// GpuBuffer (with `shared=false`) is owned by the LoadedModel and
    /// frees the memory at teardown.
    shared: bool,
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
            shared: false,
        })
    }

    /// Sprint B Phase 1 — construct a non-owning view that aliases an
    /// existing `vk::Buffer` (a bucket). Returned `GpuBuffer` has
    /// `shared=true` so `destroy()` is a no-op; the underlying bucket
    /// is owned elsewhere (LoadedModel::buckets).
    pub fn shared_view(handle: vk::Buffer, size: u64) -> Self {
        Self { handle, size, allocation: None, shared: true }
    }

    /// Sprint v0.5.2 (coalesced backing) — wrap a `vk::Buffer` that the
    /// caller has already bound to externally-owned `VkDeviceMemory`
    /// (a shared coalesced backing block, `LoadedModel::bucket_backing`).
    /// `allocation=None` + `shared=false` so `destroy()` runs
    /// `vkDestroyBuffer` (this GpuBuffer owns the VkBuffer) but performs
    /// NO `allocator.free()` (it holds no gpu-allocator allocation; the
    /// backing memory is freed separately, AFTER all such buffers).
    pub fn from_bound_buffer(handle: vk::Buffer, size: u64) -> Self {
        Self { handle, size, allocation: None, shared: false }
    }

    /// Write up to `bytes.len()` host-visible bytes to the buffer.
    /// Errors if the buffer is GPU-only (not host-mappable).
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.write_bytes_at(0, bytes)
    }

    /// Write at an offset — used by batched staging-buffer uploads.
    pub fn write_bytes_at(
        &mut self,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alloc = self.allocation.as_mut().ok_or("buffer already freed")?;
        let slice = alloc
            .mapped_slice_mut()
            .ok_or("buffer is not host-visible")?;
        let off = offset as usize;
        let end = off
            .checked_add(bytes.len())
            .ok_or("write offset+len overflows usize")?;
        if end > slice.len() {
            return Err(format!(
                "write of {} bytes at offset {} exceeds buffer capacity {}",
                bytes.len(),
                offset,
                slice.len()
            )
            .into());
        }
        slice[off..end].copy_from_slice(bytes);
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
        if self.shared {
            // Sprint B Phase 1 — non-owning view into a bucket; skip
            // both allocator.free and device.destroy_buffer so we don't
            // double-free the shared handle.
            return;
        }
        if let Some(alloc) = self.allocation.take() {
            // free is best-effort; if it fails we still need to destroy
            // the VkBuffer to avoid a leak the validation layer would
            // catch on instance teardown.
            let _ = allocator.free(alloc);
        }
        unsafe { device.destroy_buffer(self.handle, None) };
    }

    /// Sprint B Phase 2 — `true` when this `GpuBuffer` is a non-owning
    /// view into a shared bucket (set by `shared_view`). Consumers use
    /// this to decide whether descriptor bindings need an explicit
    /// sub-range (`byte_offset`, `byte_size`) instead of WHOLE_SIZE.
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Sprint G.6b diag accessor — returns the underlying VkDeviceMemory
    /// handle (as u64) and the suballocation offset within that memory.
    /// Pure read-only; used by `VF_VRAM_DIAG` to correlate physical
    /// placement with per-call GEMV BW. Behaviour-free (only the caller's
    /// `eprintln` consumes it); returns None if the buffer has been freed.
    pub fn debug_placement(&self) -> Option<(u64, u64)> {
        use ash::vk::Handle as _;
        let alloc = self.allocation.as_ref()?;
        // SAFETY: Allocation::memory() is marked unsafe only because the
        // returned handle's lifetime is tied to the Allocation; we use the
        // raw u64 immediately for logging and never store/dereference it,
        // so the lifetime constraint cannot be violated here.
        let mem = unsafe { alloc.memory() };
        Some((mem.as_raw(), alloc.offset()))
    }
}
