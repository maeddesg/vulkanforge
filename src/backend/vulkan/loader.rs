//! VRAM model loader — wires the GGUF parser to the buffer layer.
//!
//! Phase 2B / Schritt 2.4. One `vk::Buffer` per tensor (suballocated
//! by gpu-allocator), uploaded in batches through a 256-MiB staging
//! buffer. Per Phase-2A's 4-GiB `maxMemoryAllocationSize` limit on
//! RADV, gpu-allocator handles the underlying `vkAllocateMemory`
//! splitting transparently — we don't need our own multi-allocation
//! arena for the weights yet.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use super::buffers::GpuBuffer;
use super::commands::CommandContext;
use super::device::VulkanDevice;
use super::gguf::{GgmlType, GgufFile, ModelConfig};
use crate::hf_config::{HfConfig, Llama3RopeScaling};
use crate::safetensors::{hf_to_vf_name, SafeTensorsFile, TensorDtype, TensorInfo};

/// Staging-buffer size used for batched uploads. Sprint 20-M1
/// bumped this from 1 GiB → 2.5 GiB so the 8B Llama lm_head
/// (128256 × 4096 × 4 B = 2.1 GiB after BF16→FP32 expansion) fits
/// in a single staging slot. Still under RADV's 4-GiB
/// `maxMemoryAllocationSize` (Phase-2A report §3.1). For GGUF
/// loads the headroom is only paid as virtual memory until pages
/// actually touch — `gpu-allocator` doesn't pin the whole slot.
const STAGING_BYTES: u64 = 2_560 * 1024 * 1024;

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
    /// Sprint 20 — per-tensor FP32 dequantization scale for FP8 weights.
    /// `Some` for per-tensor models (Llama-3.1-FP8, `strategy: "tensor"`)
    /// and `None` for per-channel and block-wise models (which use
    /// `scale_buffer` only). The FP8 GEMV decode path consumes this
    /// scalar via push constant; the FP8 GEMM prefill path uses
    /// `scale_buffer` regardless.
    pub weight_scale: Option<f32>,
    /// Sprint 24A — FP32 dequantization scale buffer for FP8 weights.
    /// Per-channel: length = `shape[0]` (= output dim). Per-tensor
    /// models populate the buffer by broadcasting the on-disk scalar
    /// to all M positions. Sprint 35: also used for block-wise scales,
    /// flat row-major `[N/block_n, K/block_k]` (see `scale_block`).
    /// `None` for unquantized tensors (norms, embeddings, `lm_head`)
    /// and for GGUF tensors.
    pub scale_buffer: Option<GpuBuffer>,
    /// Sprint 35 — block dimensions for block-wise FP8 weights
    /// (Qwen3-FP8, DeepSeek-V3-FP8). `Some((block_n, block_k))` means
    /// `scale_buffer` is a flat row-major 2D `[N/block_n, K/block_k]`
    /// FP32 grid; one scale entry covers an entire `block_n × block_k`
    /// weight tile. `None` for per-tensor / per-channel scales — those
    /// consumers continue to read `scale_buffer` as a 1D `[N]` vector.
    pub scale_block: Option<(u32, u32)>,
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
                    weight_scale: None,
                    scale_buffer: None,
                    scale_block: None,
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
            if let Some(s) = t.scale_buffer {
                s.destroy(device, allocator);
            }
        }
    }

    /// Sprint 20-M1 — load a HuggingFace SafeTensors model directory.
    /// Reads `config.json` + `model.safetensors` (or sharded
    /// `model.safetensors.index.json`), parses the
    /// `compressed-tensors`/`naive-quantized` FP8 schema, and uploads
    /// to GPU through the same staging path as the GGUF loader.
    ///
    /// Conversions applied at load time:
    /// * `F8_E4M3` weights → uploaded as raw bytes (1 B / elem),
    ///   `ggml_type = F8E4M3`. Per-tensor `weight_scale` (BF16 scalar
    ///   in the file) is read into FP32 and stored on `GpuTensor`.
    /// * `BF16` tensors (norms, embeddings, lm_head) → expanded to
    ///   FP32 on the host before upload, `ggml_type = F32`. VF's
    ///   shaders consume FP32 for those today; a follow-up sprint can
    ///   keep them as BF16 if the lm_head VRAM cost (≈1 GiB at 8B
    ///   scale) becomes a concern.
    /// * `F16` / `F32` tensors → uploaded raw.
    /// * `*.input_scale` tensors are skipped (vLLM activation
    ///   quantization metadata; VF runs activations in FP32 / FP16).
    /// Returns `(model, host_embed_cache, hf_config)` — the host
    /// cache holds the BF16→FP32-expanded token_embd row-major so
    /// `EmbeddingSource::Host` can look up rows without GPU readback.
    /// The HfConfig is returned alongside so the caller can apply
    /// quirks (Llama-3 RoPE scaling, chat-template selection) that
    /// VF's `ModelConfig` doesn't capture.
    pub fn load_safetensors(
        dev: &VulkanDevice,
        allocator: &mut Allocator,
        dir: &Path,
    ) -> Result<(Self, Vec<f32>, HfConfig), LoaderError> {
        let st = SafeTensorsFile::open(dir)
            .map_err(LoaderError::Buffer)?;
        let hf = HfConfig::from_dir(dir)
            .map_err(LoaderError::Buffer)?;
        let config = hf_to_model_config(&hf)?;

        // Sprint 24A + 35 — pre-pass collects FP32 weight_scale data
        // keyed by the VF tensor name. The upload-pass classifies the
        // scale into per-tensor / per-channel / block-wise based on
        // the parsed shape. Three on-disk encodings are handled:
        //   * `*.weight_scale`       — `strategy: "tensor"` (scalar)
        //                              or `"channel"` (`[out_dim]`).
        //                              BF16 (Llama-3.1-FP8, Qwen2.5-FP8).
        //   * `*.weight_scale_inv`   — block-wise 2D `[N/bn, K/bk]`
        //                              (Qwen3-FP8, DeepSeek-V3-FP8).
        //                              BF16 too on Qwen3.
        // The dequant math is identical (`weight × scale`) — the `_inv`
        // suffix is just upstream naming for the precomputed inverse
        // of the absolute-max-based scale. We record the suffix only
        // for diagnostics.
        struct ParsedScale {
            flat: Vec<f32>,
            shape: Vec<u32>,
            #[allow(dead_code)]
            is_inverse: bool,
        }
        let mut weight_scales: HashMap<String, ParsedScale> =
            HashMap::with_capacity(256);
        for (hf_name, info) in &st.tensors {
            let (stem, is_inverse) = if let Some(s) = hf_name.strip_suffix(".weight_scale_inv") {
                (s, true)
            } else if let Some(s) = hf_name.strip_suffix(".weight_scale") {
                (s, false)
            } else {
                continue;
            };
            let weight_hf_name = format!("{stem}.weight");
            let Some(vf_weight_name) = hf_to_vf_name(&weight_hf_name) else { continue; };
            let bytes = st.tensor_bytes(info);
            let flat = match info.dtype {
                TensorDtype::BF16 => bf16_scale_to_f32_vec(bytes, &weight_hf_name)?,
                TensorDtype::F32 => {
                    if bytes.len() % 4 != 0 {
                        return Err(LoaderError::Buffer(format!(
                            "F32 weight_scale '{weight_hf_name}': byte length {} is not a multiple of 4",
                            bytes.len(),
                        )));
                    }
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
                TensorDtype::F16 => {
                    if bytes.len() % 2 != 0 {
                        return Err(LoaderError::Buffer(format!(
                            "F16 weight_scale '{weight_hf_name}': byte length {} is not a multiple of 2",
                            bytes.len(),
                        )));
                    }
                    bytes
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect()
                }
                other => {
                    return Err(LoaderError::Buffer(format!(
                        "weight_scale '{weight_hf_name}' has unsupported dtype {:?} \
                         (expected BF16/F16/F32)",
                        other,
                    )));
                }
            };
            let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();
            weight_scales.insert(vf_weight_name, ParsedScale { flat, shape, is_inverse });
        }

        // Plan the upload list: (vf_name, source_dtype, source_bytes,
        // target_ggml_type, owned_fp32_storage_for_bf16). Owned-storage
        // keeps the converted FP32 alive across the staging-flush loop;
        // raw F8/F16/F32 tensors borrow the mmap directly via `Cow`.
        enum Source<'a> {
            Borrowed(&'a [u8]),
            Owned(Vec<u8>),
        }
        impl<'a> Source<'a> {
            fn as_slice(&self) -> &[u8] {
                match self {
                    Source::Borrowed(b) => b,
                    Source::Owned(v) => v.as_slice(),
                }
            }
        }
        struct Plan<'a> {
            vf_name: String,
            shape: Vec<u64>,
            target_dtype: GgmlType,
            bytes: Source<'a>,
            /// Sprint 20 — per-tensor scalar (None for per-channel /
            /// block-wise models). Consumed by the FP8 GEMV decode
            /// path via push constant.
            weight_scale_scalar: Option<f32>,
            /// Sprint 24A — Pre-built FP32 scale vector. Length is
            /// `out_dim` (per-channel / per-tensor-broadcast) or
            /// `(N/block_n) * (K/block_k)` (block-wise, row-major).
            scale_vec: Option<Vec<f32>>,
            /// Sprint 35 — block dimensions when the scale is 2D.
            /// `Some((block_n, block_k))` triggers the block-wise
            /// FP8 GEMV path; `None` keeps the per-channel route.
            scale_block: Option<(u32, u32)>,
        }
        // Sprint 22B — VRAM saver: when `output.weight` (the lm_head)
        // is present, the GPU copy of `model.embed_tokens.weight` is
        // dead weight. The chat / bench paths read the embedding from
        // the host cache (`EmbeddingSource::Host`), and `dispatch_final`
        // only falls through to `token_embd.weight` for tied-weight
        // models (`tie_word_embeddings: true`). neuralmagic's
        // Llama-3.1-FP8 sets `tie_word_embeddings: false`, so skipping
        // the GPU upload here saves ~2 GiB on an 8B vocab without any
        // shader change.
        let has_lm_head = st.tensors.contains_key("lm_head.weight");
        let skip_embed_gpu = has_lm_head && !hf.tie_word_embeddings;
        let mut plans: Vec<Plan> = Vec::with_capacity(st.tensors.len());
        for (hf_name, info) in &st.tensors {
            // Skip scale metadata — already harvested above.
            if hf_name.ends_with(".weight_scale") || hf_name.ends_with(".input_scale") {
                continue;
            }
            // Sprint 22B — skip the embedding GPU upload (host cache
            // covers all reads).
            if skip_embed_gpu && hf_name == "model.embed_tokens.weight" {
                continue;
            }
            let Some(vf_name) = hf_to_vf_name(hf_name) else {
                // Tensor we don't (yet) consume — skip silently rather
                // than failing the whole load.
                continue;
            };

            let raw = st.tensor_bytes(info);
            // Sprint 22C — narrow BF16 lm_head to FP16 instead of
            // expanding to FP32 (saves ~1.1 GiB on Llama-3.1's
            // 128 256 × 4 096 lm_head). Other BF16 tensors (the
            // 32 input/output layernorms, ~1 MiB total) stay FP32
            // because RmsNorm consumes FP32 weights and the shader
            // change isn't worth a few hundred KiB.
            let is_lm_head = hf_name == "lm_head.weight";
            let (target_dtype, bytes) = match info.dtype {
                TensorDtype::F8E4M3 => (GgmlType::F8E4M3, Source::Borrowed(raw)),
                TensorDtype::F16 => (GgmlType::F16, Source::Borrowed(raw)),
                TensorDtype::F32 => (GgmlType::F32, Source::Borrowed(raw)),
                TensorDtype::BF16 if is_lm_head => {
                    let fp16 = bf16_to_f16_vec(raw, info, hf_name)?;
                    (GgmlType::F16, Source::Owned(fp16))
                }
                TensorDtype::BF16 => {
                    let fp32 = bf16_to_f32_vec(raw, info, hf_name)?;
                    (GgmlType::F32, Source::Owned(fp32))
                }
                TensorDtype::F8E5M2 => {
                    return Err(LoaderError::Buffer(format!(
                        "tensor '{hf_name}': F8_E5M2 not supported in M1 (E4M3 only)"
                    )));
                }
            };
            let shape: Vec<u64> = info.shape.iter().map(|&s| s as u64).collect();
            // Sprint 24A — build the per-output-row scale vector. The
            // scale is stored either as a scalar (`strategy: "tensor"`)
            // or as a `[out_dim]` vector (`strategy: "channel"`). For
            // FP8 weights the GEMV/GEMM kernels index `scale[row]`,
            // so scalars are broadcast to `[out_dim]` here.
            let out_dim = shape.first().copied().unwrap_or(0) as usize;
            let in_dim = shape.get(1).copied().unwrap_or(0) as usize;
            let raw_scale = weight_scales.get(&hf_name_to_vf(hf_name));
            // Sprint 35 — branch on the parsed scale shape:
            //   shape=[]  / [1]                  → per-tensor scalar
            //   shape=[out_dim]                  → per-channel
            //   shape=[scale_n, scale_k]         → block-wise; block_n
            //                                      and block_k are inferred
            //                                      from `weight.shape /
            //                                      scale.shape`.
            let (weight_scale_scalar, scale_vec, scale_block) = match raw_scale {
                None => (None, None, None),
                Some(ps) if ps.shape.is_empty() || (ps.shape.len() == 1 && ps.flat.len() == 1) => {
                    if out_dim == 0 {
                        return Err(LoaderError::Buffer(format!(
                            "tensor '{hf_name}': scalar weight_scale on a tensor without out_dim"
                        )));
                    }
                    let scalar = ps.flat[0];
                    (Some(scalar), Some(vec![scalar; out_dim]), None)
                }
                Some(ps) if ps.shape.len() == 1 && ps.flat.len() == out_dim => {
                    (None, Some(ps.flat.clone()), None)
                }
                Some(ps) if ps.shape.len() == 2 => {
                    let scale_n = ps.shape[0] as usize;
                    let scale_k = ps.shape[1] as usize;
                    if scale_n == 0 || scale_k == 0 {
                        return Err(LoaderError::Buffer(format!(
                            "tensor '{hf_name}': 2D weight_scale has zero dimension {:?}",
                            ps.shape,
                        )));
                    }
                    if out_dim == 0 || in_dim == 0 {
                        return Err(LoaderError::Buffer(format!(
                            "tensor '{hf_name}': 2D weight_scale on tensor without rank-2 shape"
                        )));
                    }
                    if out_dim % scale_n != 0 || in_dim % scale_k != 0 {
                        return Err(LoaderError::Buffer(format!(
                            "tensor '{hf_name}': 2D scale shape {:?} does not tile weight shape \
                             [{out_dim}, {in_dim}] cleanly",
                            ps.shape,
                        )));
                    }
                    if ps.flat.len() != scale_n * scale_k {
                        return Err(LoaderError::Buffer(format!(
                            "tensor '{hf_name}': 2D weight_scale element count {} != \
                             scale_n*scale_k = {}*{}",
                            ps.flat.len(), scale_n, scale_k,
                        )));
                    }
                    // Sprint 35 — collapse trivially-2D shapes to per-channel.
                    // Qwen2.5-14B-FP8 stores per-channel as `[out_dim, 1]`
                    // (one K-block of size in_dim). That's mathematically
                    // identical to per-channel, but the per-channel GEMM
                    // path covers prefill while the block-wise path
                    // doesn't yet — so we route this case to per-channel.
                    if scale_k == 1 && scale_n == out_dim {
                        (None, Some(ps.flat.clone()), None)
                    } else {
                        let block_n = (out_dim / scale_n) as u32;
                        let block_k = (in_dim / scale_k) as u32;
                        (None, Some(ps.flat.clone()), Some((block_n, block_k)))
                    }
                }
                Some(ps) => {
                    return Err(LoaderError::Buffer(format!(
                        "tensor '{hf_name}': unsupported weight_scale shape {:?} \
                         (length {}, expected scalar / [out_dim] / [N/bn, K/bk])",
                        ps.shape, ps.flat.len(),
                    )));
                }
            };
            plans.push(Plan {
                vf_name,
                shape,
                target_dtype,
                bytes,
                weight_scale_scalar,
                scale_vec,
                scale_block,
            });
        }
        // Deterministic upload order — easier to read panics.
        plans.sort_by(|a, b| a.vf_name.cmp(&b.vf_name));

        // Upload through the same staging path as the GGUF loader.
        let started = Instant::now();
        let mut tensors: HashMap<String, GpuTensor> =
            HashMap::with_capacity(plans.len());
        let mut bytes_uploaded: u64 = 0;
        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)?;
        let mut staging = GpuBuffer::new(
            &dev.device,
            allocator,
            STAGING_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "loader_staging_st",
        )
        .map_err(|e| LoaderError::Buffer(format!("staging buffer: {e}")))?;
        let mut staging_off: u64 = 0;
        let mut pending: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();

        for plan in plans.iter() {
            let src = plan.bytes.as_slice();
            let size = src.len() as u64;
            if size > STAGING_BYTES {
                staging.destroy(&dev.device, allocator);
                cmd_ctx.destroy(&dev.device);
                for (_, t) in tensors.drain() {
                    t.buffer.destroy(&dev.device, allocator);
                }
                return Err(LoaderError::TensorTooLarge {
                    name: plan.vf_name.clone(),
                    size,
                    max: STAGING_BYTES,
                });
            }
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
            let dst = match GpuBuffer::new(
                &dev.device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                &plan.vf_name,
            ) {
                Ok(b) => b,
                Err(e) => {
                    staging.destroy(&dev.device, allocator);
                    cmd_ctx.destroy(&dev.device);
                    for (_, t) in tensors.drain() {
                        t.buffer.destroy(&dev.device, allocator);
                    }
                    return Err(LoaderError::Buffer(format!(
                        "tensor '{}' alloc: {e}",
                        plan.vf_name,
                    )));
                }
            };
            staging
                .write_bytes_at(staging_off, src)
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

            // Sprint 24A — upload the per-output-row scale alongside
            // the weight tensor. Scale is FP32; size = `out_dim * 4`.
            // Total VRAM budget for scales is tiny — Llama-3.1-8B FP8
            // is ~3 MiB and Qwen2.5-14B FP8 is ~5 MiB across all linears.
            let scale_buffer = if let Some(svec) = &plan.scale_vec {
                let scale_bytes_size = (svec.len() * std::mem::size_of::<f32>()) as u64;
                if staging_off + scale_bytes_size > STAGING_BYTES {
                    if let Err(e) = Self::flush_batch(dev, &cmd_ctx, &staging, &pending) {
                        staging.destroy(&dev.device, allocator);
                        cmd_ctx.destroy(&dev.device);
                        for (_, t) in tensors.drain() {
                            t.buffer.destroy(&dev.device, allocator);
                            if let Some(s) = t.scale_buffer { s.destroy(&dev.device, allocator); }
                        }
                        return Err(e);
                    }
                    pending.clear();
                    staging_off = 0;
                }
                let scale_buf = GpuBuffer::new(
                    &dev.device,
                    allocator,
                    scale_bytes_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    MemoryLocation::GpuOnly,
                    &format!("{}.weight_scale", plan.vf_name),
                ).map_err(|e| LoaderError::Buffer(format!(
                    "scale buffer alloc for '{}': {e}", plan.vf_name,
                )))?;
                let scale_bytes: &[u8] = bytemuck::cast_slice(svec);
                staging
                    .write_bytes_at(staging_off, scale_bytes)
                    .map_err(|e| LoaderError::Buffer(e.to_string()))?;
                pending.push((
                    scale_buf.handle,
                    vk::BufferCopy::default()
                        .src_offset(staging_off)
                        .dst_offset(0)
                        .size(scale_bytes_size),
                ));
                staging_off += scale_bytes_size;
                bytes_uploaded += scale_bytes_size;
                Some(scale_buf)
            } else {
                None
            };

            tensors.insert(
                plan.vf_name.clone(),
                GpuTensor {
                    buffer: dst,
                    shape: plan.shape.clone(),
                    ggml_type: plan.target_dtype,
                    byte_size: size,
                    weight_scale: plan.weight_scale_scalar,
                    scale_buffer,
                    scale_block: plan.scale_block,
                },
            );
        }
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

        // Llama-3 RoPE scaling is parsed but not yet plumbed into
        // ModelConfig (Sprint 20-M1 scope). Surface it in stderr so
        // the user knows we saw it; the Llama-3.1-FP8 release uses
        // theta=500_000 + a 4-band scale that pre-multiplies the RoPE
        // frequencies. M3 will need to wire this through.
        if let Some(scaling) = Llama3RopeScaling::from_config(&hf) {
            eprintln!(
                "VulkanForge: Llama-3 RoPE scaling detected (factor={}, original_max_pos={}). \
                 Decode/prefill will use unscaled freqs in M1; M3 should plumb this through.",
                scaling.factor, scaling.original_max_position_embeddings,
            );
        }

        // Pull the embedding cache out of the SafeTensors mmap one
        // more time. It's small enough (vocab × hidden × 4 B ≈ 2 GiB
        // for an 8B Llama vocab) to keep on host alongside the GPU
        // copy, and EmbeddingSource::Host avoids a per-token GPU
        // readback during decode.
        let host_embed = {
            let info = st.tensor("model.embed_tokens.weight")
                .ok_or_else(|| LoaderError::Buffer(
                    "model.embed_tokens.weight not present in SafeTensors".into()
                ))?;
            let raw = st.tensor_bytes(info);
            match info.dtype {
                TensorDtype::F32 => {
                    let mut out = vec![0.0_f32; info.n_elements()];
                    out.copy_from_slice(bytemuck::cast_slice(raw));
                    out
                }
                TensorDtype::F16 => {
                    let n = info.n_elements();
                    let mut out = vec![0.0_f32; n];
                    for i in 0..n {
                        let h = u16::from_le_bytes([raw[2*i], raw[2*i+1]]);
                        out[i] = half::f16::from_bits(h).to_f32();
                    }
                    out
                }
                TensorDtype::BF16 => {
                    let n = info.n_elements();
                    let mut out = vec![0.0_f32; n];
                    for i in 0..n {
                        let bf = u16::from_le_bytes([raw[2*i], raw[2*i+1]]);
                        out[i] = bf16_to_f32(bf);
                    }
                    out
                }
                other => return Err(LoaderError::Buffer(format!(
                    "unsupported embedding dtype {:?}", other
                ))),
            }
        };

        Ok((
            Self {
                config,
                tensors,
                bytes_uploaded,
                upload_duration,
            },
            host_embed,
            hf,
        ))
    }
}

/// Map an HF tensor name (as it appears in the SafeTensors header) to
/// the VF tensor name used as the lookup key in [`weight_scales`].
/// Returns the VF "weight" name even when called with a stem that
/// already misses the trailing `.weight` — defensive against future
/// callers.
fn hf_name_to_vf(hf_name: &str) -> String {
    hf_to_vf_name(hf_name).unwrap_or_else(|| hf_name.to_string())
}

/// BF16 → FP32 conversion (host-side). BF16 layout (1+8+7 bits) is
/// the upper half of the matching FP32 value, so the conversion is a
/// 16-bit left shift; NaN/Inf/denormal/zero all round-trip exactly.
#[inline]
fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

/// Sprint 24A — Convert a `weight_scale` BF16 byte slice (either a
/// single scalar for `strategy: "tensor"` models or an `[out_dim]`
/// vector for `strategy: "channel"` models) into an FP32 `Vec<f32>`.
/// The caller decides whether to broadcast a length-1 result up to
/// `out_dim` (per-tensor case) or pass the vector through unchanged
/// (per-channel case).
fn bf16_scale_to_f32_vec(bytes: &[u8], context: &str) -> Result<Vec<f32>, LoaderError> {
    if bytes.is_empty() || bytes.len() % 2 != 0 {
        return Err(LoaderError::Buffer(format!(
            "BF16 weight_scale '{context}': byte length {} is not a non-zero multiple of 2",
            bytes.len(),
        )));
    }
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bf = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(bf16_to_f32(bf));
    }
    Ok(out)
}

/// Expand a BF16 byte slice to FP32 on the host. Used at load time
/// for SafeTensors tensors that we don't (yet) consume in BF16 form
/// natively (norms, token_embd, lm_head).
fn bf16_to_f32_vec(
    raw: &[u8],
    info: &TensorInfo,
    name: &str,
) -> Result<Vec<u8>, LoaderError> {
    if raw.len() % 2 != 0 {
        return Err(LoaderError::Buffer(format!(
            "BF16 tensor '{name}': byte length {} is not a multiple of 2",
            raw.len(),
        )));
    }
    let n = raw.len() / 2;
    let expected = info.n_elements();
    if n != expected {
        return Err(LoaderError::Buffer(format!(
            "BF16 tensor '{name}': byte length implies {n} elements, shape implies {expected}"
        )));
    }
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        let f = bf16_to_f32(bf);
        out[4 * i .. 4 * i + 4].copy_from_slice(&f.to_le_bytes());
    }
    Ok(out)
}

/// Sprint 22C — narrow a BF16 byte slice to FP16 on the host.
/// Used at load time to halve `lm_head`'s GPU footprint (Llama-3.1
/// 128 256 × 4 096 = 2.1 GiB FP32 → 1.0 GiB FP16). FP16's range
/// (±65 504) covers normal LLM weight magnitudes; the BF16 → FP32
/// → FP16 round-trip via the `half` crate handles NaN / Inf /
/// subnormal correctly. Norms (≪ 1 MiB total) stay FP32 — no
/// VRAM payoff for the extra shader work.
fn bf16_to_f16_vec(
    raw: &[u8],
    info: &TensorInfo,
    name: &str,
) -> Result<Vec<u8>, LoaderError> {
    if raw.len() % 2 != 0 {
        return Err(LoaderError::Buffer(format!(
            "BF16 tensor '{name}': byte length {} is not a multiple of 2",
            raw.len(),
        )));
    }
    let n = raw.len() / 2;
    let expected = info.n_elements();
    if n != expected {
        return Err(LoaderError::Buffer(format!(
            "BF16 tensor '{name}': byte length implies {n} elements, shape implies {expected}"
        )));
    }
    let mut out = vec![0u8; n * 2];
    for i in 0..n {
        let bf_bits = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        let f32_val = half::bf16::from_bits(bf_bits).to_f32();
        let f16_bits = half::f16::from_f32(f32_val).to_bits();
        out[2 * i .. 2 * i + 2].copy_from_slice(&f16_bits.to_le_bytes());
    }
    Ok(out)
}

/// Bridge between HuggingFace `config.json` and VulkanForge's
/// `ModelConfig`. Sprint 20-M1 scope: enough fields to drive the
/// existing forward pass on a Llama-style architecture.
fn hf_to_model_config(hf: &HfConfig) -> Result<ModelConfig, LoaderError> {
    // Sprint 24C — accept Llama-style and Qwen2-style architectures.
    // Both use the same forward-pass scaffolding; the only behavioural
    // differences are (a) Qwen2 carries Q/K/V projection biases that
    // Llama omits, handled by Sprint 24B's bias-add dispatch, and (b)
    // a different rope_theta default (1e6 for Qwen2.5 vs 5e5 for
    // Llama-3) — both read from `config.json` directly, so no code
    // branch needed.
    let arch = match hf.model_type.as_str() {
        "llama" => "llama",
        "qwen2" => "qwen2",
        // Sprint 35 — Qwen3 SafeTensors (Qwen3-FP8 family). Uses the
        // qwen2/qwen3 forward path with q_norm/k_norm enabled; bias is
        // gated on the actual presence of `attn_*.bias` tensors in the
        // SafeTensors archive (Qwen3 omits them, Qwen2.5 carries them).
        "qwen3" => "qwen3",
        other => {
            return Err(LoaderError::Buffer(format!(
                "SafeTensors model_type '{other}' not yet supported \
                 (have: 'llama', 'qwen2', 'qwen3')"
            )));
        }
    };
    use super::gguf::RopeVariant;
    // Sprint 35 — Qwen3 ships per-head Q/K norms (`q_norm` / `k_norm`)
    // that the forward pass dispatches when `has_qk_norm` is true.
    // Qwen2/Llama set this `false`; Qwen3 keeps it `true`.
    let has_qk_norm = hf.model_type == "qwen3";
    // Important: SafeTensors / PyTorch carries Q/K weights in the
    // *un-permuted* HuggingFace layout, where RoPE rotates the
    // [i, i + head_dim/2] pairs (NeoX / GPT-NeoX style). llama.cpp's
    // GGUF for Llama re-permutes the weights at conversion so its
    // `LLM_ROPE_TYPE_NORM` (adjacent-pair) shader produces the same
    // numerics. We don't permute on load — instead we route through
    // `RopeVariant::Neox` (the existing `rope_neox.comp` shader) so
    // the math matches HF semantics on the raw weights.
    Ok(ModelConfig {
        architecture: arch.to_string(),
        hidden_dim: hf.hidden_size,
        ffn_dim: hf.intermediate_size,
        n_heads: hf.num_attention_heads,
        n_kv_heads: hf.n_kv_heads(),
        head_dim: hf.head_dim(),
        n_layers: hf.num_hidden_layers,
        vocab_size: hf.vocab_size,
        rms_norm_eps: hf.rms_norm_eps,
        rope_freq_base: hf.rope_theta,
        rope_dim: hf.head_dim(),
        rope_variant: RopeVariant::Neox,
        context_length: hf.max_position_embeddings.unwrap_or(2048),
        has_qk_norm,
    })
}
