//! VRAM model loader — wires the GGUF parser to the buffer layer.
//!
//! Phase 2B / Schritt 2.4. One `vk::Buffer` per tensor (suballocated
//! by gpu-allocator), uploaded in batches through a 256-MiB staging
//! buffer. Per Phase-2A's 4-GiB `maxMemoryAllocationSize` limit on
//! RADV, gpu-allocator handles the underlying `vkAllocateMemory`
//! splitting transparently — we don't need our own multi-allocation
//! arena for the weights yet.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::buffers::GpuBuffer;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::gguf::{GgmlType, GgufFile, ModelConfig};

/// Staging-buffer size used for batched uploads. 1 GiB easily fits
/// the largest single tensor in a Qwen3-8B Q4_K_M GGUF (the Q6_K
/// LM-head `output.weight` is ~510 MiB) and keeps the load to ≤ 5
/// submits. Stays well under the 4-GiB `maxMemoryAllocationSize`
/// limit on RADV (see Phase-2A report §3.1).
const STAGING_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
pub enum LoaderError {
    Gguf(super::gguf::GgufError),
    Vk(vk::Result),
    Buffer(String),
    TensorTooLarge { name: String, size: u64, max: u64 },
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::Gguf(e) => write!(f, "GGUF parse: {e}"),
            LoaderError::Vk(r) => write!(f, "Vulkan: {r}"),
            LoaderError::Buffer(s) => write!(f, "buffer: {s}"),
            LoaderError::TensorTooLarge { name, size, max } => write!(
                f,
                "tensor '{name}' is {size} B, exceeds {max} B staging buffer"
            ),
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<super::gguf::GgufError> for LoaderError {
    fn from(e: super::gguf::GgufError) -> Self {
        LoaderError::Gguf(e)
    }
}

impl From<vk::Result> for LoaderError {
    fn from(r: vk::Result) -> Self {
        LoaderError::Vk(r)
    }
}

pub struct GpuTensor {
    pub buffer: GpuBuffer,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    pub byte_size: u64,
}

pub struct LoadedModel {
    pub config: ModelConfig,
    pub tensors: HashMap<String, GpuTensor>,
    pub bytes_uploaded: u64,
    pub upload_duration: Duration,
}

impl LoadedModel {
    pub fn load(
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        gguf: &GgufFile,
    ) -> Result<Self, LoaderError> {
        let config = ModelConfig::from_gguf(gguf)?;

        // Sort tensor names for deterministic upload order — helpful
        // when debugging which tensor a panic came from.
        let mut tensor_names: Vec<&str> =
            gguf.tensors.keys().map(|s| s.as_str()).collect();
        tensor_names.sort_unstable();

        let started = Instant::now();
        let mut tensors: HashMap<String, GpuTensor> =
            HashMap::with_capacity(gguf.tensors.len());
        let mut bytes_uploaded: u64 = 0;

        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;

        // One persistent staging buffer; rewound at every flush.
        let mut staging = GpuBuffer::new(
            &dev.device,
            allocator,
            STAGING_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "loader_staging",
        )
        .map_err(|e| LoaderError::Buffer(format!("staging buffer: {e}")))?;

        let mut staging_off: u64 = 0;
        // (dst_buffer_handle, src_offset_in_staging, dst_offset, size)
        let mut pending: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();

        for name in tensor_names {
            let info = gguf.tensor(name).expect("name from same map");
            let size = info.byte_size();
            if size > STAGING_BYTES {
                // Fail loud; no per-tensor staging fallback today.
                staging.destroy(&dev.device, allocator);
                cmd_ctx.destroy(&dev.device);
                for (_, t) in tensors.drain() {
                    t.buffer.destroy(&dev.device, allocator);
                }
                return Err(LoaderError::TensorTooLarge {
                    name: name.to_string(),
                    size,
                    max: STAGING_BYTES,
                });
            }

            // Flush the batch if this tensor wouldn't fit.
            if staging_off + size > STAGING_BYTES {
                if let Err(e) = Self::flush_batch(dev, &cmd_ctx, &staging, &pending) {
                    staging.destroy(&dev.device, allocator);
                    cmd_ctx.destroy(&dev.device);
                    for (_, t) in tensors.drain() {
                        t.buffer.destroy(&dev.device, allocator);
                    }
                    return Err(e);
                }
                pending.clear();
                staging_off = 0;
            }

            // Allocate the destination buffer.
            let dst = match GpuBuffer::new(
                &dev.device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                name,
            ) {
                Ok(b) => b,
                Err(e) => {
                    staging.destroy(&dev.device, allocator);
                    cmd_ctx.destroy(&dev.device);
                    for (_, t) in tensors.drain() {
                        t.buffer.destroy(&dev.device, allocator);
                    }
                    return Err(LoaderError::Buffer(format!(
                        "tensor '{name}' alloc: {e}"
                    )));
                }
            };

            // Copy mmap → staging.
            let src_bytes = gguf.tensor_bytes(info);
            staging
                .write_bytes_at(staging_off, src_bytes)
                .map_err(|e| LoaderError::Buffer(e.to_string()))?;

            pending.push((
                dst.handle,
                vk::BufferCopy::default()
                    .src_offset(staging_off)
                    .dst_offset(0)
                    .size(size),
            ));
            staging_off += size;
            bytes_uploaded += size;

            tensors.insert(
                name.to_string(),
                GpuTensor {
                    buffer: dst,
                    shape: info.dimensions.clone(),
                    ggml_type: info.ggml_type,
                    byte_size: size,
                },
            );
        }

        // Final flush.
        if !pending.is_empty() {
            if let Err(e) = Self::flush_batch(dev, &cmd_ctx, &staging, &pending) {
                staging.destroy(&dev.device, allocator);
                cmd_ctx.destroy(&dev.device);
                for (_, t) in tensors.drain() {
                    t.buffer.destroy(&dev.device, allocator);
                }
                return Err(e);
            }
        }

        let upload_duration = started.elapsed();

        staging.destroy(&dev.device, allocator);
        cmd_ctx.destroy(&dev.device);

        Ok(Self {
            config,
            tensors,
            bytes_uploaded,
            upload_duration,
        })
    }

    fn flush_batch(
        dev: &VulkanDevice,
        cmd_ctx: &CommandContext,
        staging: &GpuBuffer,
        batch: &[(vk::Buffer, vk::BufferCopy)],
    ) -> Result<(), LoaderError> {
        if batch.is_empty() {
            return Ok(());
        }
        cmd_ctx
            .one_shot(&dev.device, dev.compute_queue, |cmd| {
                for (dst, copy) in batch {
                    unsafe {
                        dev.device.cmd_copy_buffer(
                            cmd,
                            staging.handle,
                            *dst,
                            std::slice::from_ref(copy),
                        );
                    }
                }
            })
            .map_err(|e| LoaderError::Buffer(e.to_string()))?;
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        for (_, t) in self.tensors.drain() {
            t.buffer.destroy(device, allocator);
        }
    }
}
