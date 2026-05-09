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
use crate::quantize::{quantize_f32_to_q4k, QK_K};
use crate::safetensors::{hf_to_vf_name, SafeTensorsFile, TensorDtype, TensorInfo};

/// Staging-buffer size used for batched uploads. Sprint 20-M1
/// bumped this from 1 GiB → 2.5 GiB so the 8B Llama lm_head
/// (128256 × 4096 × 4 B = 2.1 GiB after BF16→FP32 expansion) fits
/// in a single staging slot. Sprint 51D-A bumped to 3.5 GiB so
/// Gemma-4-26B-A4B's `token_embd.weight` (262144 × 2816 × 4 B =
/// 2.95 GiB) fits — tied-weight model where `skip_embed_gpu` can't
/// drop the upload because lm_head's fallback path reads the same
/// GPU tensor (proper Embed-Lookup-Host-Pfad is a follow-up
/// sub-sprint). Still under RADV's 4-GiB `maxMemoryAllocationSize`
/// (Phase-2A report §3.1). For GGUF loads the headroom is only paid
/// as virtual memory until pages actually touch — `gpu-allocator`
/// doesn't pin the whole slot.
const STAGING_BYTES: u64 = 3_584 * 1024 * 1024;

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
    /// Sprint 40 Part 2 — `lm_head` weights requantized to Q6_K and
    /// resident in CPU pinned RAM, populated when `VF_CPU_LM_HEAD=1`.
    /// `Forward::dispatch_final` branches on this field: when `Some`,
    /// the GPU `lm_head` GEMV is skipped and the post-norm hidden
    /// state is downloaded for a CPU GEMV instead. `None` keeps the
    /// status-quo GPU path.
    pub cpu_lm_head: Option<crate::cpu::lm_head::CpuLmHead>,
    /// Sprint 43D-3 — per-layer-input embedding state for Gemma-4.
    /// `Some` when the model is Gemma-4 AND the SafeTensors file has
    /// the `embed_tokens_per_layer` + `per_layer_projection_norm`
    /// tensors. CPU-resident; lookup + scale + RMSNorm runs on the host
    /// per-token before each `forward_token`, then the per-layer-inputs
    /// vector ([num_layers × hps] FP32) is uploaded to a per-slot
    /// CpuToGpu buffer and consumed by `dispatch_layer`'s PLE block.
    pub ple_data: Option<PleData>,
}

/// Sprint 43D-3 + 43D-4 — Per-Layer Embeddings runtime state for Gemma-4.
///
/// HF transformers `Gemma4TextModel.{get,project}_per_layer_inputs`:
///
/// 1. **token_identity** = `embed_tokens_per_layer[token_id]` × √hps.
///    Reshape to `[num_layers, hps]`. NO RMSNorm here. (The
///    `Gemma4TextScaledWordEmbedding` applies the √hps scale inline.)
/// 2. **context_projection** =
///       `per_layer_model_projection @ inputs_embeds`     # [num_layers × hps]
///       × (1 / √hidden_size)
///       reshape `[num_layers, hps]`
///       per-row RMSNorm with `per_layer_projection_norm.weight`.
///    The `× 1/√hidden_size` before RMSNorm is mathematically a no-op
///    (RMSNorm normalises out any constant scalar) — VF skips it.
/// 3. **merge** = `(context_projection + token_identity) × (1 / √2)`.
///
/// All CPU-side per token (35 × 256 = 8960 floats; the GEMV is
/// 13.8 M MACs / token = ~50 µs on AVX-512 — negligible vs. per-token
/// GPU forward).
///
/// The result is a `[num_layers × hps]` FP32 vector that gets uploaded
/// to a per-slot CpuToGpu staging buffer and consumed by
/// `Forward::dispatch_layer`'s PLE block at slice
/// `[layer * hps .. (layer+1) * hps]`.
pub struct PleData {
    /// `embed_tokens_per_layer.weight` raw BF16 bytes, owned (cloned
    /// from the SafeTensors mmap at load). Shape:
    /// `[vocab_size_per_layer_input, num_layers × hps]`. Row-major;
    /// `byte_offset(tok) = tok * num_layers * hps * 2`. Sized 4.7 GB
    /// for Gemma-4-E2B (262 144 × 35 × 256 × 2 B).
    pub embed_table_bf16: Vec<u8>,
    /// Vocab dim of the PLE table (262 144 for E2B; usually equal to
    /// the main vocab but stored separately for clarity).
    pub vocab: u32,
    /// 35 for E2B.
    pub num_layers: u32,
    /// 256 for E2B (`hidden_size_per_layer_input`).
    pub hps: u32,
    /// `hidden_size` (1536 on E2B). Needed as the GEMV input dim.
    pub hidden_size: u32,
    /// `sqrt(hps)` — applied to the BF16→F32 token-identity lookup.
    pub embed_per_layer_scale: f32,
    /// `per_layer_projection_norm.weight` host vector, length `hps`.
    /// Sprint 43D-4 — applied to the context_projection (row-major
    /// per-layer slice), NOT to the token_identity (43D-3 wrong place).
    pub projection_norm_gamma: Vec<f32>,
    /// `cfg.rms_norm_eps` — same eps as the main RMSNorms in the layer.
    pub rms_norm_eps: f32,
    /// Sprint 43D-4 — `per_layer_model_projection.weight` expanded to
    /// FP32, row-major `[num_layers × hps, hidden_size]` =
    /// `[8960, 1536]`. Used as the context-aware GEMV that combines
    /// with the token-identity lookup. Sized ~55 MB on E2B.
    pub per_layer_model_projection: Vec<f32>,
}

impl PleData {
    /// CPU-side per-token PLE prep (Sprint 43D-4 build).
    ///
    /// Inputs:
    /// - `token_id`: current token's id (drives the `embed_tokens_per_layer` lookup).
    /// - `inputs_embeds`: the post-`embed_scale` token embedding for the
    ///   current token (length = `hidden_size`). Matches HF's `inputs_embeds`
    ///   = `embed_tokens(input_ids) * sqrt(hidden_size)`.
    ///
    /// Returns a `[num_layers × hps]` FP32 vector ready for GPU upload.
    pub fn build_per_layer_inputs(
        &self,
        token_id: u32,
        inputs_embeds: &[f32],
    ) -> Vec<f32> {
        let nl = self.num_layers as usize;
        let hps = self.hps as usize;
        let h = self.hidden_size as usize;
        let row_elems = nl * hps;
        debug_assert_eq!(inputs_embeds.len(), h);

        // (1) token_identity = embed_tokens_per_layer[token_id] × √hps.
        let row_off = (token_id as usize) * row_elems * 2;
        let row_bytes = &self.embed_table_bf16[row_off..row_off + row_elems * 2];
        let mut token_id_part = vec![0.0_f32; row_elems];
        for (i, chunk) in row_bytes.chunks_exact(2).enumerate() {
            let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
            token_id_part[i] = bf16_to_f32(bf) * self.embed_per_layer_scale;
        }

        // (2) context_projection = per_layer_model_projection @ inputs_embeds.
        //     Weight shape [out=row_elems, in=hidden_size], row-major. Output
        //     is [row_elems] FP32. Skip the HF `× 1/√hidden_size` scale —
        //     RMSNorm below makes it a no-op.
        let mut ctx = vec![0.0_f32; row_elems];
        let w = self.per_layer_model_projection.as_slice();
        debug_assert_eq!(w.len(), row_elems * h);
        for out_i in 0..row_elems {
            let row = &w[out_i * h..(out_i + 1) * h];
            let mut acc = 0.0_f64;
            for j in 0..h {
                acc += (row[j] as f64) * (inputs_embeds[j] as f64);
            }
            ctx[out_i] = acc as f32;
        }

        //     Per-row RMSNorm with projection_norm_gamma (length = hps).
        let eps = self.rms_norm_eps as f64;
        for layer in 0..nl {
            let slice = &mut ctx[layer * hps..(layer + 1) * hps];
            let mut sum_sq = 0.0_f64;
            for &v in slice.iter() {
                sum_sq += (v as f64) * (v as f64);
            }
            let mean_sq = sum_sq / (hps as f64);
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for (i, v) in slice.iter_mut().enumerate() {
                *v = ((*v as f64) * scale * (self.projection_norm_gamma[i] as f64)) as f32;
            }
        }

        // (3) merge = (ctx + token_id_part) × (1/√2).
        let inv_sqrt2 = 1.0_f32 / std::f32::consts::SQRT_2;
        let mut out = vec![0.0_f32; row_elems];
        for i in 0..row_elems {
            out[i] = (ctx[i] + token_id_part[i]) * inv_sqrt2;
        }
        out
    }
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
            // GGUF path does not yet support CPU lm_head offload; the
            // SafeTensors path does. Future sprint extends this to
            // GGUF Q6_K (which is already 6-bit on disk and would
            // map directly into a `CpuLmHead` without requantize).
            cpu_lm_head: None,
            ple_data: None,
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
        // Sprint 40 Part 2 — `VF_CPU_LM_HEAD=1` opt-in: skip the
        // GPU upload of `lm_head.weight` and instead requantize it
        // to Q6_K on the CPU. The forward.rs decode path branches
        // on `LoadedModel::cpu_lm_head` to pick CPU vs GPU lm_head.
        let cpu_lm_head_enabled = std::env::var("VF_CPU_LM_HEAD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let skip_lm_head_gpu = cpu_lm_head_enabled && has_lm_head;
        // Sprint 50B — opt-in on-the-fly Q4_K quantization. Gated by
        // `VF_QUANTIZE_ON_LOAD=1`; default OFF until Sprint 50D
        // validates coherence on each architecture.
        let quantize_on_load = std::env::var("VF_QUANTIZE_ON_LOAD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut q_stats = QuantStats::default();
        let mut plans: Vec<Plan> = Vec::with_capacity(st.tensors.len());
        for (hf_name, info) in &st.tensors {
            // Skip scale metadata — already harvested above.
            if hf_name.ends_with(".weight_scale") || hf_name.ends_with(".input_scale") {
                continue;
            }
            // Sprint 40 Part 2 — when CPU lm_head is enabled, the
            // GPU copy of `lm_head.weight` is unused (dispatch_final
            // takes the CPU branch). Skip the upload to free 0.5–
            // 0.8 GiB of VRAM. The lm_head bytes are dequantized
            // and requantized to Q6_K for the CPU path further
            // below — we still mmap them via `st.tensor_bytes`.
            if skip_lm_head_gpu && hf_name == "lm_head.weight" {
                continue;
            }
            // Sprint 22B — skip the embedding GPU upload (host cache
            // covers all reads).
            if skip_embed_gpu && (hf_name == "model.embed_tokens.weight"
                || hf_name == "model.language_model.embed_tokens.weight") {
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
            // Sprint 43E-2 — Gemma-4 ties lm_head to embed_tokens. The
            // tied buffer's `is_lm_head` heuristic in Sprint 22B was
            // limited to literal `lm_head.weight`; for Gemma-4 the
            // tied tensor lives at `model.language_model.embed_tokens.
            // weight`. VF_GEMMA_EMBED_F16=1 routes Gemma-4 embed via
            // BF16→F16 (= MulMatVecF16 path) instead of BF16→F32
            // (= MulMatVecF32). Diagnose-only: lets us isolate
            // whether MulMatVecF32 has a bug specific to vocab=262144
            // by routing through the well-exercised F16 harness path.
            let is_lm_head = hf_name == "lm_head.weight"
                || (hf.gemma4.is_some()
                    && hf_name == "model.language_model.embed_tokens.weight"
                    && std::env::var("VF_GEMMA_EMBED_F16").is_ok());
            // Sprint 50B — quantize 2D weights to Q4_K_M when
            // `VF_QUANTIZE_ON_LOAD=1` and the tensor passes
            // `should_quantize_st`. Norms / embeddings / lm_head /
            // 1D scalars / non-multiple-of-256 tensors stay at their
            // native dtype.
            let do_quant = quantize_on_load
                && !is_lm_head
                && should_quantize_st(hf_name, info);
            let (target_dtype, bytes) = match info.dtype {
                TensorDtype::F8E4M3 => (GgmlType::F8E4M3, Source::Borrowed(raw)),
                TensorDtype::F16 => (GgmlType::F16, Source::Borrowed(raw)),
                TensorDtype::F32 => {
                    if do_quant {
                        let f32_vec: Vec<f32> = raw
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let q4k = quantize_f32_to_q4k(&f32_vec);
                        q_stats.record(raw.len() as u64, q4k.len() as u64);
                        (GgmlType::Q4K, Source::Owned(q4k))
                    } else {
                        q_stats.record_skipped(hf_name, info.dtype);
                        (GgmlType::F32, Source::Borrowed(raw))
                    }
                }
                TensorDtype::BF16 if is_lm_head => {
                    let fp16 = bf16_to_f16_vec(raw, info, hf_name)?;
                    (GgmlType::F16, Source::Owned(fp16))
                }
                TensorDtype::BF16 => {
                    if do_quant {
                        let n = info.n_elements();
                        let mut f32_vec = vec![0f32; n];
                        for i in 0..n {
                            let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                            f32_vec[i] = bf16_to_f32(bf);
                        }
                        let q4k = quantize_f32_to_q4k(&f32_vec);
                        // FP32-equivalent original size for compression ratio
                        // (BF16-on-disk would understate the saving since
                        // the runtime would have expanded to FP32 anyway).
                        q_stats.record((n * 4) as u64, q4k.len() as u64);
                        (GgmlType::Q4K, Source::Owned(q4k))
                    } else {
                        // Sprint 43E investigation note — earlier hypothesis
                        // was that Gemma-4 RMSNorm uses `(1 + weight)` and
                        // requires a load-time +1.0 fix. Verification against
                        // HF transformers `Gemma3nRMSNorm` (Gemma-4 inherits
                        // from this) showed the semantics are *Llama-style*
                        // again (`x * weight / sqrt(...)`, init=`torch.ones`).
                        // The +1 patch was WRONG and got reverted. The NaN
                        // root cause is *not* in RMSNorm semantics; further
                        // bisect needed (lm_head GEMV with vocab=262144
                        // remains the prime suspect).
                        let fp32 = bf16_to_f32_vec(raw, info, hf_name)?;
                        q_stats.record_skipped(hf_name, info.dtype);
                        (GgmlType::F32, Source::Owned(fp32))
                    }
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

        // Sprint 50B — surface the quantization summary.
        if quantize_on_load {
            q_stats.print();
        }

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
            // Sprint 43B-2 — Gemma-4 nests embed under `model.language_model.`.
            // Probe both names so the same loader path covers Llama / Qwen
            // (`model.embed_tokens.weight`) and Gemma-4
            // (`model.language_model.embed_tokens.weight`).
            let info = st
                .tensor("model.embed_tokens.weight")
                .or_else(|| st.tensor("model.language_model.embed_tokens.weight"))
                .ok_or_else(|| LoaderError::Buffer(
                    "embed_tokens.weight not present in SafeTensors \
                     (tried model.embed_tokens.weight and \
                     model.language_model.embed_tokens.weight)".into()
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

        // Sprint 40 Part 2 — build the CPU `lm_head` (Q6_K) when the
        // `VF_CPU_LM_HEAD=1` opt-in flag was set. We dequantize
        // whatever native dtype the SafeTensors file uses to FP32,
        // apply the FP8 weight scale if present, and hand the FP32
        // matrix to `CpuLmHead::from_fp32_weights` for the on-load
        // requantize. The corresponding GPU upload was already
        // skipped above (`skip_lm_head_gpu`).
        let cpu_lm_head = if cpu_lm_head_enabled {
            let info = st
                .tensor("lm_head.weight")
                .ok_or_else(|| {
                    LoaderError::Buffer(
                        "VF_CPU_LM_HEAD=1 but lm_head.weight not in SafeTensors".into(),
                    )
                })?;
            let raw = st.tensor_bytes(info);
            // Shape is [vocab, hidden]. Both must be present.
            let vocab_size = *info.shape.first().ok_or_else(|| {
                LoaderError::Buffer("lm_head.weight: missing vocab dim".into())
            })? as usize;
            let hidden_size = *info.shape.get(1).ok_or_else(|| {
                LoaderError::Buffer("lm_head.weight: missing hidden dim".into())
            })? as usize;

            // Dequantize bytes → FP32. Llama-3.1, Qwen2.5, Qwen3 all
            // ship lm_head as BF16; F16 / F32 / F8E4M3 are accepted
            // for completeness.
            let n_elements = vocab_size * hidden_size;
            let mut fp32 = vec![0.0_f32; n_elements];
            match info.dtype {
                TensorDtype::F32 => {
                    fp32.copy_from_slice(bytemuck::cast_slice(raw));
                }
                TensorDtype::F16 => {
                    for i in 0..n_elements {
                        let h = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                        fp32[i] = half::f16::from_bits(h).to_f32();
                    }
                }
                TensorDtype::BF16 => {
                    for i in 0..n_elements {
                        let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                        fp32[i] = bf16_to_f32(bf);
                    }
                }
                TensorDtype::F8E4M3 => {
                    // FP8 lm_head — rare but valid. Use ash bindings
                    // for the conversion; the loader does the same in
                    // the GPU upload path via the shader.
                    use crate::backend::vulkan::fp8_ext::fp8_e4m3_to_f32;
                    for i in 0..n_elements {
                        fp32[i] = fp8_e4m3_to_f32(raw[i]);
                    }
                    // Apply scale if present (per-tensor scalar; the
                    // exotic per-channel / block-wise lm_head cases
                    // are not in the wild on the supported models).
                    let vf_name = hf_name_to_vf("lm_head.weight");
                    if let Some(ps) = weight_scales.get(&vf_name) {
                        if !ps.flat.is_empty() {
                            let scale = ps.flat[0];
                            for v in fp32.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }
                other => {
                    return Err(LoaderError::Buffer(format!(
                        "VF_CPU_LM_HEAD=1: lm_head.weight has unsupported dtype {other:?}"
                    )));
                }
            };

            let t0 = Instant::now();
            let lm = crate::cpu::lm_head::CpuLmHead::from_fp32_weights(
                &fp32,
                vocab_size,
                hidden_size,
            );
            eprintln!(
                "VF_CPU_LM_HEAD=1: lm_head {}×{} requantized to Q6_K ({:.1} MB, {:.0} ms)",
                vocab_size,
                hidden_size,
                lm.size_bytes() as f64 / 1e6,
                t0.elapsed().as_secs_f64() * 1000.0,
            );
            Some(lm)
        } else {
            None
        };

        // Sprint 43D-3 — PLE setup for Gemma-4 SafeTensors models.
        // Two top-level tensors that the standard hf_to_vf_name path
        // returns None for:
        //   1. `model.language_model.embed_tokens_per_layer.weight`
        //      Shape: [vocab_size_per_layer_input, num_layers × hps]
        //      BF16; 4.7 GB on E2B (262144 × 35 × 256 × 2 B).
        //      Cloned into a host Vec<u8> for runtime row lookup.
        //   2. `model.language_model.per_layer_projection_norm.weight`
        //      Shape: [hps]. BF16 on disk; expanded to FP32 host vec.
        //
        // Per-layer pieces (`per_layer_input_gate`, `per_layer_projection`,
        // `post_per_layer_input_norm`) ship via the standard upload path
        // since 43D-3's safetensors.rs change.
        let ple_data = if hf.gemma4.is_some() {
            let pl_table_info = st.tensor("model.language_model.embed_tokens_per_layer.weight");
            let pl_norm_info = st.tensor("model.language_model.per_layer_projection_norm.weight");
            // Sprint 43D-4 — context-aware projection. Required for the
            // (token_identity + context) merge inside PLE.
            let pl_proj_info = st.tensor("model.language_model.per_layer_model_projection.weight");
            match (pl_table_info, pl_norm_info, pl_proj_info, hf.gemma4.as_ref()) {
                (Some(table_info), Some(norm_info), Some(proj_info), Some(gm)) => {
                    if !matches!(table_info.dtype, TensorDtype::BF16) {
                        return Err(LoaderError::Buffer(format!(
                            "embed_tokens_per_layer dtype {:?} not supported (BF16 only)",
                            table_info.dtype
                        )));
                    }
                    let table_bytes = st.tensor_bytes(table_info);
                    let nl = config.n_layers;
                    let hps = gm.hidden_size_per_layer_input;
                    let vocab = gm.vocab_size_per_layer_input;
                    let hidden = config.hidden_dim;
                    let expected = (vocab as u64) * (nl as u64) * (hps as u64) * 2;
                    if table_bytes.len() as u64 != expected {
                        return Err(LoaderError::Buffer(format!(
                            "embed_tokens_per_layer size {} != expected {} \
                             (vocab={vocab} × n_layers={nl} × hps={hps} × 2 B)",
                            table_bytes.len(), expected,
                        )));
                    }
                    let projection_norm_gamma: Vec<f32> = match norm_info.dtype {
                        TensorDtype::BF16 => {
                            let raw = st.tensor_bytes(norm_info);
                            // `bf16_to_f32_vec` returns Vec<u8> (FP32
                            // bytes); reinterpret as f32 values.
                            let bytes = bf16_to_f32_vec(
                                raw, norm_info, "per_layer_projection_norm.weight",
                            )?;
                            bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        }
                        TensorDtype::F32 => {
                            let raw = st.tensor_bytes(norm_info);
                            raw.chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        }
                        other => {
                            return Err(LoaderError::Buffer(format!(
                                "per_layer_projection_norm dtype {other:?} not supported"
                            )));
                        }
                    };
                    if projection_norm_gamma.len() != hps as usize {
                        return Err(LoaderError::Buffer(format!(
                            "per_layer_projection_norm has {} elements, expected {}",
                            projection_norm_gamma.len(), hps,
                        )));
                    }
                    // Sprint 43D-4 — context-aware projection weight.
                    // Shape: [num_layers × hps, hidden_size] = [8960, 1536].
                    // Validate dtype + size; expand BF16 → FP32 host vec.
                    let proj_expected_elems =
                        (nl as u64) * (hps as u64) * (hidden as u64);
                    let per_layer_model_projection: Vec<f32> = match proj_info.dtype {
                        TensorDtype::BF16 => {
                            let raw = st.tensor_bytes(proj_info);
                            if (raw.len() as u64) != proj_expected_elems * 2 {
                                return Err(LoaderError::Buffer(format!(
                                    "per_layer_model_projection size {} != expected {} \
                                     (out={}×{}={}, in={}, BF16 ⇒ ×2 B)",
                                    raw.len(), proj_expected_elems * 2,
                                    nl, hps, nl * hps, hidden,
                                )));
                            }
                            let bytes = bf16_to_f32_vec(
                                raw, proj_info, "per_layer_model_projection.weight",
                            )?;
                            bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        }
                        TensorDtype::F32 => {
                            let raw = st.tensor_bytes(proj_info);
                            if (raw.len() as u64) != proj_expected_elems * 4 {
                                return Err(LoaderError::Buffer(format!(
                                    "per_layer_model_projection size {} != expected {} \
                                     (out={}×{}={}, in={}, F32 ⇒ ×4 B)",
                                    raw.len(), proj_expected_elems * 4,
                                    nl, hps, nl * hps, hidden,
                                )));
                            }
                            raw.chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        }
                        other => {
                            return Err(LoaderError::Buffer(format!(
                                "per_layer_model_projection dtype {other:?} not supported"
                            )));
                        }
                    };
                    let t0 = Instant::now();
                    let embed_table_bf16 = table_bytes.to_vec();
                    eprintln!(
                        "Sprint 43D-3 PLE: embed_tokens_per_layer cloned ({:.1} GB BF16, {:.0} ms)",
                        embed_table_bf16.len() as f64 / 1e9,
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                    eprintln!(
                        "Sprint 43D-4 PLE: per_layer_model_projection expanded \
                         ({:.1} MB FP32 host, [{}, {}])",
                        (per_layer_model_projection.len() * 4) as f64 / 1e6,
                        nl * hps, hidden,
                    );
                    Some(PleData {
                        embed_table_bf16,
                        vocab,
                        num_layers: nl,
                        hps,
                        hidden_size: hidden,
                        embed_per_layer_scale: (hps as f32).sqrt(),
                        projection_norm_gamma,
                        rms_norm_eps: config.rms_norm_eps,
                        per_layer_model_projection,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        Ok((
            Self {
                config,
                tensors,
                bytes_uploaded,
                upload_duration,
                cpu_lm_head,
                ple_data,
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

/// Sprint 50B — accumulator for `VF_QUANTIZE_ON_LOAD=1` reporting.
/// Tracks how many tensors got quantized vs. left at native dtype,
/// and the byte counts (FP32-equivalent for the source side so the
/// compression ratio reflects the in-VRAM saving).
#[derive(Default)]
struct QuantStats {
    n_quantized: u32,
    n_skipped_norm: u32,
    n_skipped_embed: u32,
    n_skipped_other: u32,
    bytes_orig_fp32_eq: u64,
    bytes_q4k: u64,
}

impl QuantStats {
    fn record(&mut self, orig_bytes: u64, q4k_bytes: u64) {
        self.n_quantized += 1;
        self.bytes_orig_fp32_eq += orig_bytes;
        self.bytes_q4k += q4k_bytes;
    }
    fn record_skipped(&mut self, hf_name: &str, _dtype: TensorDtype) {
        if hf_name.contains("norm") {
            self.n_skipped_norm += 1;
        } else if hf_name.contains("embed") {
            self.n_skipped_embed += 1;
        } else {
            self.n_skipped_other += 1;
        }
    }
    fn print(&self) {
        let n_skipped = self.n_skipped_norm + self.n_skipped_embed + self.n_skipped_other;
        let orig_gib = self.bytes_orig_fp32_eq as f64 / (1024.0 * 1024.0 * 1024.0);
        let q4k_gib = self.bytes_q4k as f64 / (1024.0 * 1024.0 * 1024.0);
        let ratio = if self.bytes_q4k > 0 {
            self.bytes_orig_fp32_eq as f64 / self.bytes_q4k as f64
        } else {
            0.0
        };
        println!(
            "VulkanForge: On-the-fly Q4_K quantization ENABLED (VF_QUANTIZE_ON_LOAD=1)"
        );
        println!(
            "  Quantized {} tensors ({:.2} GiB FP32-eq → {:.2} GiB Q4_K, {:.2}× compression)",
            self.n_quantized, orig_gib, q4k_gib, ratio,
        );
        println!(
            "  Skipped   {} tensors (norms: {}, embeddings: {}, other: {})",
            n_skipped, self.n_skipped_norm, self.n_skipped_embed, self.n_skipped_other,
        );
    }
}

/// Sprint 50B — should the SafeTensors tensor `hf_name` be quantized
/// to Q4_K_M at load? Heuristic mirrors the brief: 2D weights with
/// `n_elements % 256 == 0` only, blacklisting the norms / embeddings /
/// lm_head paths that consume FP32 in shaders today.
fn should_quantize_st(hf_name: &str, info: &TensorInfo) -> bool {
    // Q4_K block requires `numel % 256 == 0`. No padding implemented;
    // tensors that don't fit cleanly stay at their native dtype.
    if info.n_elements() % QK_K != 0 {
        return false;
    }
    // Embeddings: lookup tables, never enter a GEMM.
    if hf_name.contains("embed_tokens") {
        return false;
    }
    // Any norm weight (RMSNorm, LayerNorm, q/k norms, post-attention)
    // is precision-sensitive and typically tiny. Defensive even though
    // most are 1D and already filtered by the `shape.len() != 2` test.
    if hf_name.contains("norm") {
        return false;
    }
    // lm_head has its own paths (CPU-Offload via VF_CPU_LM_HEAD,
    // FP16-narrow at upload time, tied-weight skipping). Don't compete.
    if hf_name.ends_with("lm_head.weight") {
        return false;
    }
    // Per-layer scalars (Gemma-4 `layer_scalar` [1]) — already filtered
    // by `n_elements % 256 != 0`, but defensive.
    if hf_name.ends_with("layer_scalar") {
        return false;
    }
    // Sprint 51D-A — Gemma-4-26B MoE router weights stay FP32. The
    // 2D `router.proj.weight` is 2816×128 = 360448 (% 256 == 0)
    // which would otherwise be picked up by the 2D rule below; the
    // 1D `router.scale` and `router.per_expert_scale` already get
    // skipped by `shape.len() != 2`, but the blacklist keeps the
    // intent explicit.
    if hf_name.contains("router") {
        return false;
    }
    // Q4_K is a GEMM/GEMV input format. 2D weight tensors are the
    // common case (Q/K/V/O proj, MLP gate/up/down, embeddings).
    // Sprint 51D-A — Gemma-4-26B's `experts.gate_up_proj` /
    // `experts.down_proj` are packed 3D tensors `[n_experts, …, …]`
    // whose total numel is a multiple of 256 (verified in 51A);
    // route them through the same Q4_K quantizer (the block layout
    // is rank-agnostic, only numel matters).
    if info.shape.len() == 3 && hf_name.contains("experts.") {
        return true;
    }
    if info.shape.len() != 2 {
        return false;
    }
    true
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
    use super::gguf::{Gemma4KvSource, Gemma4LayerKind, Gemma4LayerSpec, Gemma4Spec, RopeVariant};

    let arch = match hf.model_type.as_str() {
        "llama" => "llama",
        "qwen2" => "qwen2",
        // Sprint 35 — Qwen3 SafeTensors (Qwen3-FP8 family). Uses the
        // qwen2/qwen3 forward path with q_norm/k_norm enabled; bias is
        // gated on the actual presence of `attn_*.bias` tensors in the
        // SafeTensors archive (Qwen3 omits them, Qwen2.5 carries them).
        "qwen3" => "qwen3",
        // Sprint 43B-2 — Gemma-4 weights load through this path. The
        // forward refactor in 43B-2 picks it up via `cfg.gemma4 =
        // Some(Gemma4Spec)`. Sprint 43D adds PLE.
        "gemma4" => "gemma4",
        other => {
            return Err(LoaderError::Buffer(format!(
                "SafeTensors model_type '{other}' not yet supported \
                 (have: 'llama', 'qwen2', 'qwen3', 'gemma4')"
            )));
        }
    };
    // Sprint 35 — Qwen3 ships per-head Q/K norms (`q_norm` / `k_norm`)
    // that the forward pass dispatches when `has_qk_norm` is true.
    // Sprint 43B-2 — Gemma-4 also ships them.
    let has_qk_norm = matches!(hf.model_type.as_str(), "qwen3" | "gemma4");

    // Sprint 43B-2 — for Gemma-4, allocate buffers sized to the
    // *maximum* head_dim and intermediate_size across all layers
    // (sliding=256 / full=512, plain=6144 / double-wide=12288 in E2B).
    // Per-layer push-constants pick the actual size at dispatch time.
    let (head_dim, ffn_dim, gemma4_spec) = if let Some(gm) = hf.gemma4.as_ref() {
        let max_head_dim = gm.head_dim_sliding.max(gm.head_dim_full);
        let max_ffn = if gm.use_double_wide_mlp {
            gm.intermediate_size * 2
        } else {
            gm.intermediate_size
        };
        let first_kv_shared = hf.num_hidden_layers.saturating_sub(gm.num_kv_shared_layers);

        // Find the last sliding / last full layer index in
        // `[0, first_kv_shared)`. Those two layers carry the extra
        // "publishes shared KV" responsibility, mirroring HF's
        // `store_full_length_kv = True` flag.
        let mut last_sliding_pre: Option<u32> = None;
        let mut last_full_pre: Option<u32> = None;
        for i in 0..first_kv_shared {
            match gm.layer_types[i as usize] {
                crate::hf_config::Gemma4LayerType::Sliding => last_sliding_pre = Some(i),
                crate::hf_config::Gemma4LayerType::Full => last_full_pre = Some(i),
            }
        }

        let mut layer_specs = Vec::with_capacity(hf.num_hidden_layers as usize);
        for i in 0..hf.num_hidden_layers {
            let is_shared = i >= first_kv_shared;
            let kind = match gm.layer_types[i as usize] {
                crate::hf_config::Gemma4LayerType::Sliding => Gemma4LayerKind::Sliding,
                crate::hf_config::Gemma4LayerType::Full => Gemma4LayerKind::Full,
            };
            let head_dim_layer = match kind {
                Gemma4LayerKind::Sliding => gm.head_dim_sliding,
                Gemma4LayerKind::Full => gm.head_dim_full,
            };
            let intermediate_layer = if is_shared && gm.use_double_wide_mlp {
                gm.intermediate_size * 2
            } else {
                gm.intermediate_size
            };
            let kv_source = if is_shared {
                match kind {
                    Gemma4LayerKind::Sliding => Gemma4KvSource::SubscribesSliding,
                    Gemma4LayerKind::Full => Gemma4KvSource::SubscribesFull,
                }
            } else if Some(i) == last_sliding_pre {
                Gemma4KvSource::OwnAndPublishesSliding
            } else if Some(i) == last_full_pre {
                Gemma4KvSource::OwnAndPublishesFull
            } else {
                Gemma4KvSource::Own
            };
            let (theta, partial) = match kind {
                Gemma4LayerKind::Sliding => (gm.sliding_rope_theta, None),
                Gemma4LayerKind::Full => (gm.full_rope_theta, gm.full_rope_partial_factor),
            };
            layer_specs.push(Gemma4LayerSpec {
                kind,
                head_dim: head_dim_layer,
                intermediate_size: intermediate_layer,
                has_kv_proj: !is_shared,
                kv_source,
                rope_theta: theta,
                rope_partial_factor: partial,
                // Sprint 51B-pre — E2B has uniform kv_heads (=1
                // num_key_value_heads). 26B-A4B will diverge here:
                // 8 for sliding (`num_key_value_heads`) and 2 for
                // full (`num_global_key_value_heads`). For E2B the
                // value is the same on every layer so the Vec we feed
                // into KvCacheConfig is no-op-equivalent to the global
                // `cfg.n_kv_heads`, but routing through the per-layer
                // helper keeps the code path uniform.
                n_kv_heads: hf.n_kv_heads(),
                // Sprint 51B — full-attention layers under
                // `attention_k_eq_v: true` (26B-A4B) skip the v_proj
                // weight and derive V from K_raw. Sliding layers
                // always have their own v_proj. E2B has
                // `attention_k_eq_v=false` → every layer keeps its
                // v_proj.
                has_v_proj: !(gm.attention_k_eq_v
                    && matches!(kind, Gemma4LayerKind::Full)),
            });
        }

        // Sprint 51D-A — Failsafe von 51C entfernt. Loader kann jetzt
        // MoE-Tensoren laden:
        //   - 3D experts.gate_up_proj / experts.down_proj durch den
        //     Q4_K-Quantizer (`should_quantize_st` erkennt sie als
        //     packed-3D mit numel % 256 == 0)
        //   - router.proj.weight bleibt FP32 (router-blacklist in
        //     should_quantize_st)
        //   - router.scale / router.per_expert_scale sind 1D und
        //     fallen automatisch durch (shape.len() != 2 + router-
        //     blacklist defensive)
        //   - 3 zusätzliche Norms (`ffn_post_norm_1` /
        //     `ffn_pre_norm_2` / `ffn_post_norm_2`) durch
        //     "norm"-blacklist ebenfalls FP32.
        // Forward-Pfad ist NOCH im todo!()-Stub (Sprint 51D-B/C),
        // ein 26B-Inference-Aufruf crasht weiterhin — aber jetzt mit
        // einer klaren `not yet implemented`-Meldung statt einem
        // Loader-Error.

        let embed_scale = (hf.hidden_size as f32).sqrt();
        let spec = Gemma4Spec {
            sliding_window: gm.sliding_window,
            final_logit_softcapping: gm.final_logit_softcapping,
            embed_scale,
            hidden_activation: gm.hidden_activation.clone(),
            tie_word_embeddings: hf.tie_word_embeddings,
            first_kv_shared,
            layers: layer_specs,
            // layer_scalars get filled by the loader in a second pass
            // once the BF16 [1] tensors have been read off disk.
            layer_scalars: vec![1.0; hf.num_hidden_layers as usize],
            hidden_size_per_layer_input: gm.hidden_size_per_layer_input,
            enable_moe_block: gm.enable_moe_block,
            n_experts: gm.n_experts,
            top_k_experts: gm.top_k_experts,
            moe_intermediate_size: gm.moe_intermediate_size,
        };
        (max_head_dim, max_ffn, Some(spec))
    } else {
        (hf.head_dim(), hf.intermediate_size, None)
    };

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
        ffn_dim,
        n_heads: hf.num_attention_heads,
        n_kv_heads: hf.n_kv_heads(),
        head_dim,
        n_layers: hf.num_hidden_layers,
        vocab_size: hf.vocab_size,
        rms_norm_eps: hf.rms_norm_eps,
        rope_freq_base: hf.rope_theta,
        rope_dim: head_dim,
        rope_variant: RopeVariant::Neox,
        context_length: hf.max_position_embeddings.unwrap_or(2048),
        has_qk_norm,
        gemma4: gemma4_spec,
    })
}
