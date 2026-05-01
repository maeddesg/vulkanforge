//! Monolithic VRAM arena for VulkanForge.
//!
//! Option A from the Phase 2A prompt: one `vkAllocateMemory` for the
//! whole working set, three zones (weights, KV-cache, scratch) carved
//! by offset arithmetic. The Phase-2A constructor checks
//! `maxMemoryAllocationSize` up-front and bails with a structured
//! error when the request doesn't fit, so the caller can fall back
//! to a sub-allocator (Option B / `gpu-allocator`) without crashing.
//!
//! Buffer views: callers ask for `(offset, size, usage)` and get a
//! `vk::Buffer` bound to `(arena.memory, offset)`. Alignment is
//! checked against `vkGetBufferMemoryRequirements`; an offset that
//! doesn't satisfy a buffer's alignment requirements returns a
//! structured error rather than tripping the validation layer.
//!
//! Cleanup is explicit (`destroy(self, &Device)`); buffers created
//! out of the arena must be destroyed by their callers before the
//! arena.

use ash::vk;

#[derive(Clone, Copy, Debug)]
pub struct ArenaConfig {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    /// Total scratch — split internally into 2 equal ping-pong halves
    /// for [`VramArena::scratch_for_layer`].
    pub scratch_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaZone {
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaLayout {
    pub weights: ArenaZone,
    pub kv_cache: ArenaZone,
    pub scratch: ArenaZone,
}

pub struct VramArena {
    pub memory: vk::DeviceMemory,
    pub total_bytes: u64,
    pub layout: ArenaLayout,
    /// Memory-type index used by the underlying allocation. Buffers
    /// created from this arena must be compatible with this type
    /// (`memory_type_bits & (1 << idx) != 0`).
    pub memory_type_index: u32,
    /// Alignment used to round zone sizes up. At least
    /// `minStorageBufferOffsetAlignment`, but we bump it to 4 KiB so
    /// per-zone padding is large enough to absorb any per-buffer
    /// alignment requirement we'll encounter in practice.
    pub zone_alignment: u64,
}

#[derive(Debug)]
pub enum ArenaError {
    /// Total request exceeds `maxMemoryAllocationSize` — caller should
    /// fall back to a sub-allocator.
    AllocationTooLarge { requested: u64, max: u64 },
    NoDeviceLocalMemoryType,
    Vk(vk::Result),
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArenaError::AllocationTooLarge { requested, max } => write!(
                f,
                "arena allocation {requested} B exceeds maxMemoryAllocationSize {max} B"
            ),
            ArenaError::NoDeviceLocalMemoryType => {
                write!(f, "no DEVICE_LOCAL memory type available")
            }
            ArenaError::Vk(r) => write!(f, "Vulkan error: {r}"),
        }
    }
}

impl std::error::Error for ArenaError {}

#[derive(Debug)]
pub enum BufferViewError {
    OutOfBounds { offset: u64, size: u64, arena: u64 },
    AlignmentMismatch { offset: u64, required: u64 },
    IncompatibleMemoryType { requirements_bits: u32, arena_bit: u32 },
    Vk(vk::Result),
}

impl std::fmt::Display for BufferViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferViewError::OutOfBounds { offset, size, arena } => write!(
                f,
                "buffer view {offset}..{} exceeds arena {arena} B",
                offset + size
            ),
            BufferViewError::AlignmentMismatch { offset, required } => write!(
                f,
                "offset {offset} is not a multiple of buffer alignment {required}"
            ),
            BufferViewError::IncompatibleMemoryType {
                requirements_bits,
                arena_bit,
            } => write!(
                f,
                "buffer accepts memory types {requirements_bits:b}, arena memory type bit is {arena_bit:b}"
            ),
            BufferViewError::Vk(r) => write!(f, "Vulkan error: {r}"),
        }
    }
}

impl std::error::Error for BufferViewError {}

impl VramArena {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        config: ArenaConfig,
    ) -> Result<Self, ArenaError> {
        let limits = unsafe { instance.get_physical_device_properties(physical_device).limits };
        let mem_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let max_alloc_size = query_max_memory_allocation_size(instance, physical_device);

        let zone_alignment =
            (limits.min_storage_buffer_offset_alignment.max(4096)) as u64;
        let weights = align_up(config.weights_bytes, zone_alignment);
        let kv_cache = align_up(config.kv_cache_bytes, zone_alignment);
        let scratch = align_up(config.scratch_bytes, zone_alignment);
        let total = weights + kv_cache + scratch;

        if total > max_alloc_size {
            return Err(ArenaError::AllocationTooLarge {
                requested: total,
                max: max_alloc_size,
            });
        }

        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(ArenaError::NoDeviceLocalMemoryType)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(total)
            .memory_type_index(memory_type_index);
        let memory =
            unsafe { device.allocate_memory(&alloc_info, None) }.map_err(ArenaError::Vk)?;

        let layout = ArenaLayout {
            weights: ArenaZone { offset: 0, size: weights },
            kv_cache: ArenaZone { offset: weights, size: kv_cache },
            scratch: ArenaZone {
                offset: weights + kv_cache,
                size: scratch,
            },
        };

        Ok(Self {
            memory,
            total_bytes: total,
            layout,
            memory_type_index,
            zone_alignment,
        })
    }

    /// Create a `vk::Buffer` bound at `offset` for `size` bytes inside
    /// the arena. Returns structured errors for the three things that
    /// can go wrong without ever tripping the validation layer:
    /// out-of-bounds, alignment mismatch, incompatible memory type.
    pub fn create_buffer(
        &self,
        device: &ash::Device,
        offset: u64,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::Buffer, BufferViewError> {
        if offset.checked_add(size).map_or(true, |end| end > self.total_bytes) {
            return Err(BufferViewError::OutOfBounds {
                offset,
                size,
                arena: self.total_bytes,
            });
        }

        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }.map_err(BufferViewError::Vk)?;

        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let arena_bit = 1u32 << self.memory_type_index;
        if req.memory_type_bits & arena_bit == 0 {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(BufferViewError::IncompatibleMemoryType {
                requirements_bits: req.memory_type_bits,
                arena_bit,
            });
        }
        if offset % req.alignment != 0 {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(BufferViewError::AlignmentMismatch {
                offset,
                required: req.alignment,
            });
        }

        unsafe { device.bind_buffer_memory(buffer, self.memory, offset) }
            .map_err(BufferViewError::Vk)?;
        Ok(buffer)
    }

    /// Return `(offset, size)` for the layer's scratch slice. Two
    /// halves alternate so layer i reads what layer i-1 wrote.
    pub fn scratch_for_layer(&self, layer_idx: usize) -> (u64, u64) {
        let half_size = self.layout.scratch.size / 2;
        let half = (layer_idx % 2) as u64;
        let offset = self.layout.scratch.offset + half * half_size;
        (offset, half_size)
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe { device.free_memory(self.memory, None) };
    }
}

fn align_up(n: u64, alignment: u64) -> u64 {
    (n + alignment - 1) & !(alignment - 1)
}

/// Query `VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize`
/// (Vulkan 1.1 core). Returns a conservative 1 GiB fallback if the
/// query fails for any reason — it never silently overstates the real
/// limit.
pub fn query_max_memory_allocation_size(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> u64 {
    let mut maint3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maint3);
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    let raw = maint3.max_memory_allocation_size;
    if raw == 0 { 1 << 30 } else { raw }
}
