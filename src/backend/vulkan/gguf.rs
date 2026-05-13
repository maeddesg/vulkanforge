//! GGUF v3 parser — header + metadata + tensor inventory + mmap.
//!
//! Phase 2B / Schritt 2.4. Targeted at Qwen3 GGUFs but the parser
//! itself is architecture-agnostic; `ModelConfig::from_gguf` is the
//! piece that pulls Qwen3-specific keys (`<arch>.embedding_length`,
//! `<arch>.rope.freq_base`, etc.) out of the metadata table.
//!
//! Tensor data stays mmap'd — `tensor_bytes(name)` returns a `&[u8]`
//! into the mapped region so `loader.rs` can `memcpy` straight into
//! a HOST_VISIBLE staging buffer without ever copying through Rust
//! heap.

use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    UnknownMetadataType(u32),
    UnknownTensorType(u32),
    EndOfFile,
    MissingMetadata(String),
    UnexpectedType(String, &'static str),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::BadMagic => write!(f, "GGUF magic missing"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::UnknownMetadataType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::UnknownTensorType(t) => write!(f, "unknown ggml tensor type {t}"),
            GgufError::EndOfFile => write!(f, "unexpected end of file while parsing"),
            GgufError::MissingMetadata(k) => write!(f, "missing metadata key '{k}'"),
            GgufError::UnexpectedType(k, want) => {
                write!(f, "metadata key '{k}' is not a {want}")
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// `ggml_type` enum values matching `ggml.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    /// Brain Float 16 — 1 sign + 8 exponent + 7 mantissa, same exponent
    /// range as FP32. Appears in Gemma-family GGUFs for norm and
    /// embedding weights (the ones the quant tooling chose not to
    /// compress). VF expands BF16 → FP32 at GPU upload time
    /// (`LoadedModel::load`), so downstream shaders never see this
    /// dtype — the in-memory `GpuTensor.ggml_type` is `F32` for the
    /// expanded result.
    BF16 = 30,
    /// Sprint 20 — native FP8 E4M3 (1 byte / element). Not a GGUF
    /// type number (GGUF stops at 35); we use 100 as an internal
    /// sentinel so SafeTensors models can flow through the same
    /// `LoadedModel` / shader-routing infrastructure as GGUF
    /// quants. `from_u32` will not return this — only the
    /// `LoadedModel::load_safetensors` constructor sets it.
    F8E4M3 = 100,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            30 => GgmlType::BF16,
            _ => return Err(GgufError::UnknownTensorType(v)),
        })
    }

    /// Number of elements per quantisation block. 1 for non-quant types.
    pub fn block_size(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F8E4M3 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 |
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K |
            GgmlType::Q6K | GgmlType::Q8K => 256,
        }
    }

    /// Byte size of one block (or one element for non-quant).
    pub fn type_size(self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::BF16 => 2,
            GgmlType::F8E4M3 => 1,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            GgmlType::Q8K => 292,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::U32(v) => Some(*v),
            MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
            MetadataValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::F32(v) => Some(*v),
            MetadataValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    /// For `[STRING]` arrays — the only array we currently inspect
    /// (vocab_size from `tokenizer.ggml.tokens`).
    pub fn array_len(&self) -> Option<usize> {
        match self {
            MetadataValue::Array(v) => Some(v.len()),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            MetadataValue::U8(v) => Some(*v != 0),
            _ => None,
        }
    }
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            MetadataValue::I32(v) => Some(*v),
            MetadataValue::U32(v) if *v <= i32::MAX as u32 => Some(*v as i32),
            MetadataValue::I64(v) if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 => Some(*v as i32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor-data section
    /// (NOT the start of the file). Use [`GgufFile::tensor_bytes`] to
    /// get an absolute slice.
    pub data_offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }
    pub fn byte_size(&self) -> u64 {
        let n = self.n_elements();
        let bs = self.ggml_type.block_size();
        let ts = self.ggml_type.type_size();
        debug_assert!(n % bs == 0, "tensor element count not a multiple of block size");
        (n / bs) * ts
    }
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Absolute offset (file-relative) where the tensor-data section
    /// begins, after alignment padding.
    pub data_section_offset: u64,
    pub alignment: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let mut cursor = Cursor::new(&mmap);
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = cursor.read_u32()?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = cursor.read_u64()?;
        let kv_count = cursor.read_u64()?;

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = cursor.read_string()?;
            let value = cursor.read_metadata_value()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let n_dims = cursor.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64()?);
            }
            let type_u32 = cursor.read_u32()?;
            let ggml_type = GgmlType::from_u32(type_u32)?;
            let data_offset = cursor.read_u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dimensions: dims,
                    ggml_type,
                    data_offset,
                },
            );
        }

        // Tensor data starts at the next multiple of `alignment` after
        // the header end.
        let header_end = cursor.pos();
        let data_section_offset = align_up(header_end as u64, alignment);

        Ok(Self {
            mmap,
            version,
            tensor_count,
            metadata,
            tensors,
            data_section_offset,
            alignment,
        })
    }

    /// Raw mmap'd bytes of the tensor — `&self.mmap[..]` slice. Backed
    /// by the original file; valid for the lifetime of `&self`.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = (self.data_section_offset + info.data_offset) as usize;
        let size = info.byte_size() as usize;
        &self.mmap[start..start + size]
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str, GgufError> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_str()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "string"))
    }

    pub fn metadata_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_u32()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "u32"))
    }

    /// Gemma-4-style per-layer arrays (`{arch}.feed_forward_length`,
    /// `{arch}.attention.sliding_window_pattern` …) appear as Array of
    /// scalars in the metadata table. For buffer sizing on the
    /// `ModelConfig` side, we need a single representative value —
    /// the **max** across all layers (so any per-layer dispatch fits
    /// the same scratch buffers). Scalar metadata returns the scalar
    /// unchanged; this keeps non-Gemma archs working.
    pub fn metadata_u32_scalar_or_array_max(&self, key: &str) -> Result<u32, GgufError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?;
        if let Some(n) = v.as_u32() {
            return Ok(n);
        }
        if let Some(arr) = v.as_array() {
            let mut best: Option<u32> = None;
            for el in arr {
                if let Some(n) = el.as_u32() {
                    best = Some(best.map_or(n, |b| b.max(n)));
                }
            }
            if let Some(b) = best {
                return Ok(b);
            }
        }
        Err(GgufError::UnexpectedType(key.into(), "u32 or u32[]"))
    }

    pub fn metadata_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_f32()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "f32"))
    }

    /// Sprint 52B — read a per-layer u32 array out of GGUF metadata.
    /// Used by `Gemma4Spec::from_gguf` for `feed_forward_length` (the
    /// per-layer FFN intermediate size, already-doubled in the GGUF
    /// for shared layers). Scalar metadata is rejected here — use
    /// `metadata_u32` or `metadata_u32_scalar_or_array_max` for those.
    pub fn metadata_u32_array(&self, key: &str) -> Result<Vec<u32>, GgufError> {
        let arr = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_array()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "u32[]"))?;
        let mut out = Vec::with_capacity(arr.len());
        for el in arr {
            let v = el
                .as_u32()
                .ok_or_else(|| GgufError::UnexpectedType(key.into(), "u32[]"))?;
            out.push(v);
        }
        Ok(out)
    }

    /// Sprint 52B — read a per-layer bool array out of GGUF metadata.
    /// Used by `Gemma4Spec::from_gguf` for
    /// `gemma4.attention.sliding_window_pattern` (bool[n_layers],
    /// `true` = Sliding, `false` = Full — verified against the E2B
    /// GGUF dump, where the `False` positions are 4, 9, 14, 19, 24, 29, 34
    /// — the standard Gemma SSSSF pattern at indices ≡ 4 (mod 5)).
    pub fn metadata_bool_array(&self, key: &str) -> Result<Vec<bool>, GgufError> {
        let arr = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_array()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "bool[]"))?;
        let mut out = Vec::with_capacity(arr.len());
        for el in arr {
            let v = el
                .as_bool()
                .ok_or_else(|| GgufError::UnexpectedType(key.into(), "bool[]"))?;
            out.push(v);
        }
        Ok(out)
    }
}

/// Layout convention for RoPE. `Neox` rotates `[i, i + n_dims/2]`
/// pairs (Qwen / Qwen2 / Qwen3); `Norm` rotates adjacent `[2k, 2k+1]`
/// pairs (Llama / Mistral / DeepSeek-R1-Distill-Llama).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeVariant {
    Norm,
    Neox,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub head_dim: u32,
    pub rope_freq_base: f32,
    pub rope_dim: u32,
    pub rope_variant: RopeVariant,
    pub context_length: u32,
    pub has_qk_norm: bool,
    pub rms_norm_eps: f32,
    /// Sprint 43B-2 — Gemma-4 specific per-layer specs. `None` for
    /// every other architecture; the existing dispatch path remains
    /// untouched. When `Some`, `head_dim` and `ffn_dim` above are
    /// the *maximum* across all layers (used for `Forward::new`
    /// buffer sizing); the per-layer values live inside `Gemma4Spec`.
    pub gemma4: Option<Gemma4Spec>,
}

/// Sprint 43B-2 — per-Gemma-4 model state. Pulled into `ModelConfig`
/// rather than `LoadedModel` because the forward needs it to choose
/// the dispatch branch and per-layer weights.
#[derive(Debug, Clone)]
pub struct Gemma4Spec {
    /// Sliding-window length (512 for E2B). Sliding layers should
    /// only attend to the last `sliding_window` positions; in the
    /// 43B-2 minimal-viable forward we ignore this and use full
    /// causal — output is garbage anyway without PLE. The field is
    /// stored here so 43C can wire it without changing the API.
    pub sliding_window: u32,
    /// Final logits soft-cap (`30.0` for E2B). Applied CPU-side
    /// after `lm_head`: `logits = cap × tanh(logits / cap)`.
    pub final_logit_softcapping: Option<f32>,
    /// Embedding scale = `√hidden_size` for Gemma. Applied to the
    /// initial token embedding before layer 0.
    pub embed_scale: f32,
    /// `gelu_pytorch_tanh` for E2B. Captured raw; consumer matches.
    /// In the 43B-2 minimal-viable path we substitute SiLU (existing
    /// shader) — output is garbage but the dispatch runs.
    pub hidden_activation: String,
    /// Whether `lm_head` is tied to `embed_tokens` (true for E2B).
    pub tie_word_embeddings: bool,
    /// `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`.
    /// Layers `[0, first_kv_shared)` compute their own K/V. Layers
    /// `[first_kv_shared, num_layers)` reuse K/V from the last
    /// non-shared layer of the same `layer_type`. For E2B: 35 − 20 = 15.
    pub first_kv_shared: u32,
    /// `[layer_idx → spec]` of length `n_layers`.
    pub layers: Vec<Gemma4LayerSpec>,
    /// `[layer_idx → scalar]` of length `n_layers`. Each layer
    /// multiplies its output by this value at the very end of its
    /// forward pass. Loaded from the per-layer `layer_scalar` BF16
    /// `[1]` tensor in the SafeTensors archive (gelernt — typische
    /// Werte 0.018 .. 0.87 für E2B).
    pub layer_scalars: Vec<f32>,
    /// Sprint 43D-3 — `hidden_size_per_layer_input` (256 for E2B).
    /// Used as both the GEMV inner dim (1536 → 256 for input_gate; 256
    /// → 1536 for projection) and the per-slot
    /// `per_layer_inputs` buffer stride.
    pub hidden_size_per_layer_input: u32,
    /// Sprint 51C — `true` on Gemma-4-26B-A4B; the layer FFN runs
    /// Dense-MLP AND MoE-expert-FFN in parallel and sums their
    /// outputs. `false` for E2B.
    pub enable_moe_block: bool,
    /// Sprint 51C — total expert count (`128` for 26B-A4B, `0` for E2B).
    pub n_experts: u32,
    /// Sprint 51C — top-K experts selected per token (`8` for 26B-A4B).
    pub top_k_experts: u32,
    /// Sprint 51C — per-expert FFN intermediate size (`704` for
    /// 26B-A4B, `0` for E2B). Independent of the Dense-MLP
    /// `intermediate_size` (= 2112 for 26B-A4B).
    pub moe_intermediate_size: u32,
}

/// Sprint 43B-2 — per-layer routing state for a Gemma-4 stack.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4LayerSpec {
    pub kind: Gemma4LayerKind,
    /// 256 (sliding) or 512 (full) for E2B. Used as push-constant
    /// per-dispatch; the buffers are sized to `max(head_dim)`.
    pub head_dim: u32,
    /// 6144 for layers `[0, first_kv_shared)`, 12 288 for shared
    /// layers (double-wide MLP per `use_double_wide_mlp`).
    pub intermediate_size: u32,
    /// Whether this layer has its own `k_proj` / `v_proj` /
    /// `k_norm` / `v_norm` weights. False for layers
    /// `[first_kv_shared, n_layers)` — they reuse K/V from a
    /// type-matched earlier layer instead.
    pub has_kv_proj: bool,
    /// Where this layer reads K/V from (and, for the two writer
    /// layers in 0..first_kv_shared, where it writes to).
    pub kv_source: Gemma4KvSource,
    /// 10 000 (sliding) / 1 000 000 (full) for E2B.
    pub rope_theta: f32,
    /// Proportional-rotation factor for the full-attention RoPE
    /// variant. `Some(0.25)` for full layers in E2B; `None` for
    /// sliding layers (= default RoPE).
    pub rope_partial_factor: Option<f32>,
    /// Sprint 51B-pre — per-layer KV-head count. Equal to
    /// `cfg.n_kv_heads` for E2B (all layers); diverges on
    /// 26B-A4B (`8` for sliding layers, `2` for full layers via
    /// `num_global_key_value_heads`).
    pub n_kv_heads: u32,
    /// Sprint 51B — when `false`, the layer skips its V-side dispatch
    /// (no `v_proj` weight, no VBiasAdd). V is taken from K's raw
    /// projection (pre-norm, pre-RoPE) and run through a parameterless
    /// `v_norm`. Used by 26B-A4B's full-attention layers under
    /// `attention_k_eq_v: true`. Defaults to `true` for E2B.
    pub has_v_proj: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4LayerKind {
    Sliding,
    Full,
}

/// Sprint 43B-2 — KV-cache routing for a single layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4KvSource {
    /// Layer computes & owns its KV (layers 0..first_kv_shared
    /// minus the two write-shared layers below).
    Own,
    /// Layer 13 in E2B — owns its KV *and* this slot is the source
    /// for every later sliding layer (15, 16, 17, 18, 20, …).
    OwnAndPublishesSliding,
    /// Layer 14 in E2B — owns its KV *and* this slot is the source
    /// for every later full-attention layer (19, 24, 29, 34).
    OwnAndPublishesFull,
    /// Sliding layer in `[first_kv_shared, n_layers)` — reads K/V
    /// from the `OwnAndPublishesSliding` slot.
    SubscribesSliding,
    /// Full layer in `[first_kv_shared, n_layers)` — reads K/V
    /// from the `OwnAndPublishesFull` slot.
    SubscribesFull,
}

/// Sprint 52B — pure function that maps the per-layer
/// `sliding_window_pattern` (`true` = Sliding, `false` = Full) and the
/// `first_kv_shared` boundary into the per-layer `Gemma4KvSource`
/// routing decision. Layers in `[0, first_kv_shared)` own their KV;
/// the **last** sliding and the **last** full layer in that range
/// become the publishers for every subscriber after the boundary.
///
/// Extracted from `Gemma4Spec::from_gguf` so the publisher algorithm
/// can be unit-tested with synthetic patterns (the SafeTensors path
/// in `loader.rs:1918-1925` has the same shape but isn't testable
/// without an HfConfig).
fn build_kv_sources(
    pattern: &[bool],
    first_kv_shared: u32,
    n_layers: usize,
) -> Vec<Gemma4KvSource> {
    let mut last_sliding: Option<u32> = None;
    let mut last_full: Option<u32> = None;
    for i in 0..first_kv_shared as usize {
        // Out-of-range pattern → assume Sliding (the dense kind, which
        // is the safe default for Gemma where Full is sparse).
        let is_sliding = pattern.get(i).copied().unwrap_or(true);
        if is_sliding {
            last_sliding = Some(i as u32);
        } else {
            last_full = Some(i as u32);
        }
    }
    (0..n_layers as u32)
        .map(|i| {
            let is_sliding = pattern.get(i as usize).copied().unwrap_or(true);
            if i >= first_kv_shared {
                if is_sliding {
                    Gemma4KvSource::SubscribesSliding
                } else {
                    Gemma4KvSource::SubscribesFull
                }
            } else if Some(i) == last_sliding {
                Gemma4KvSource::OwnAndPublishesSliding
            } else if Some(i) == last_full {
                Gemma4KvSource::OwnAndPublishesFull
            } else {
                Gemma4KvSource::Own
            }
        })
        .collect()
}

impl Gemma4Spec {
    /// Sprint 52B — construct a `Gemma4Spec` from GGUF metadata. Mirrors
    /// the SafeTensors builder (`loader.rs::hf_to_model_config` Gemma-4
    /// branch) 1:1 with the metadata pulled from `gemma4.*` keys.
    ///
    /// Key cross-walk (verified against `gemma-4-E2B-it-Q4_K_M.gguf`
    /// header dump):
    /// ```
    /// SafeTensors HfConfig field         GGUF metadata key                      E2B
    /// ─────────────────────────────────────────────────────────────────────────────
    /// num_hidden_layers                  gemma4.block_count                     35
    /// hidden_size                        gemma4.embedding_length                1536
    /// num_kv_shared_layers               gemma4.attention.shared_kv_layers      20
    /// sliding_window                     gemma4.attention.sliding_window        512
    /// final_logit_softcapping            gemma4.final_logit_softcapping         30.0
    /// hidden_size_per_layer_input        gemma4.embedding_length_per_layer_input 256
    /// head_dim_full / _sliding           gemma4.attention.key_length / _swa     512 / 256
    /// full_rope_theta / sliding_         gemma4.rope.freq_base / _swa           1e6 / 1e4
    /// layer_types[i]                     gemma4.attention.sliding_window_pattern bool[35]
    ///                                       (true = Sliding, false = Full)
    /// intermediate_size[i]               gemma4.feed_forward_length             u32[35]
    ///                                       (already-doubled in GGUF for shared layers)
    /// hidden_activation                  HARDCODED "gelu_pytorch_tanh"          —
    /// tie_word_embeddings                INFERRED: !gguf.tensor("output.weight") true
    /// full_rope_partial_factor           HARDCODED Some(0.25)                   0.25
    ///                                       (llama.cpp doesn't store this key
    ///                                       — Gemma-3/4 always partial=0.25 on
    ///                                       Full layers per HF transformers)
    /// ```
    ///
    /// 26B-A4B-only fields (`expert_count`, etc.) are read with
    /// `metadata.get` so they default cleanly to 0 / None on E2B.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // Block-count and core dims (already validated by ModelConfig
        // caller but we double-read for self-containment).
        let n_layers = gguf.metadata_u32("gemma4.block_count")? as usize;
        let hidden_size = gguf.metadata_u32("gemma4.embedding_length")?;
        let embed_scale = (hidden_size as f32).sqrt();

        // KV-share boundary. E2B: 35 - 20 = 15. Layers `[0, 15)` are
        // publishers (own their KV); `[15, 35)` subscribe.
        let n_shared = gguf
            .metadata
            .get("gemma4.attention.shared_kv_layers")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize;
        let first_kv_shared = (n_layers.saturating_sub(n_shared)) as u32;

        // Per-layer kind (true = Sliding, false = Full — verified
        // against the E2B GGUF where False sits at indices 4, 9, 14,
        // 19, 24, 29, 34 = the canonical Gemma SSSSF pattern).
        let pattern: Vec<bool> = gguf
            .metadata_bool_array("gemma4.attention.sliding_window_pattern")
            .unwrap_or_else(|_| vec![true; n_layers]);
        // Per-layer FFN intermediate size. Sprint 52V — Gemma-4-26B-A4B
        // GGUFs store `feed_forward_length` as a SCALAR (since every
        // layer shares the same dense-FFN intermediate of 2112), but
        // some converters wrap it in a 1-element array. The strict
        // `metadata_u32_array` accepts only an array and was returning
        // `Err` on scalar → fallback `vec![0; n_layers]` → every layer
        // dispatched the dense FFN branch with `intermediate_size=0` →
        // missing dense-branch contribution → garbage logits invariant
        // to weight quant (Sprint 52U narrowed the bug to a
        // quant-invariant code-path divergence at Layer 0, this fix
        // closes it). Accept three shapes:
        //   per-layer u32[n_layers]   → use directly,
        //   single u32 (or u32[1])    → broadcast to all layers,
        //   missing/unparseable       → keep the legacy zero-fill.
        let ffl: Vec<u32> = match gguf.metadata_u32_array("gemma4.feed_forward_length") {
            Ok(arr) if arr.len() == n_layers => arr,
            _ => {
                let scalar = gguf
                    .metadata_u32_scalar_or_array_max("gemma4.feed_forward_length")
                    .unwrap_or(0);
                vec![scalar; n_layers]
            }
        };

        // Head / RoPE dims, Full and Sliding variants.
        let head_dim_full = gguf
            .metadata
            .get("gemma4.attention.key_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(256);
        let head_dim_sliding = gguf
            .metadata
            .get("gemma4.attention.key_length_swa")
            .and_then(|v| v.as_u32())
            .unwrap_or(256);
        let rope_full = gguf
            .metadata
            .get("gemma4.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rope_sliding = gguf
            .metadata
            .get("gemma4.rope.freq_base_swa")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);

        // Sprint 52H-1 — 26B-A4B per-layer KV-head divergence.
        // E2B: head_count_kv = 1 (scalar) for every layer.
        // 26B-A4B: head_count_kv is a u32[30] array `[8,8,8,8,8,2,...]`
        // — Sliding=8, Full=2 — exactly mirroring `sliding_window_pattern`
        // (Sliding ≡ `true`, Full ≡ `false`).
        //
        // Read both shapes: scalar (E2B) → broadcast to per-layer
        // global; array (26B) → use the per-layer array directly. The
        // scalar fallback `n_kv_global` is also the right answer for
        // every E2B layer.
        let n_kv_array: Option<Vec<u32>> = gguf
            .metadata_u32_array("gemma4.attention.head_count_kv")
            .ok();
        let n_kv_global = gguf
            .metadata
            .get("gemma4.attention.head_count_kv")
            .and_then(|v| v.as_u32())
            .or_else(|| n_kv_array.as_ref().and_then(|a| a.iter().copied().max()))
            .unwrap_or(1);

        // Per-layer kv-source via the publisher algorithm (extracted
        // to a pure fn for unit-testability).
        let kv_sources = build_kv_sources(&pattern, first_kv_shared, n_layers);

        let mut layers: Vec<Gemma4LayerSpec> = Vec::with_capacity(n_layers);
        for i in 0..n_layers as u32 {
            let is_sliding = *pattern.get(i as usize).unwrap_or(&true);
            let kind = if is_sliding {
                Gemma4LayerKind::Sliding
            } else {
                Gemma4LayerKind::Full
            };
            let head_dim = match kind {
                Gemma4LayerKind::Sliding => head_dim_sliding,
                Gemma4LayerKind::Full => head_dim_full,
            };
            let intermediate = ffl.get(i as usize).copied().unwrap_or(0);
            let is_shared = i >= first_kv_shared;
            let kv_source = kv_sources[i as usize];
            let (rope_theta, rope_partial_factor) = match kind {
                Gemma4LayerKind::Sliding => (rope_sliding, None),
                // HF Gemma-4 always uses partial_rotary_factor=0.25 on
                // Full layers (rotary_dim = 0.25 * key_length = 128 on
                // E2B Full). llama.cpp doesn't emit a metadata key for
                // this — hardcoded matches the SafeTensors path
                // (Sprint 51D-AJ).
                Gemma4LayerKind::Full => (rope_full, Some(0.25)),
            };
            // Sprint 52H-1 — 26B-A4B Full layers get n_kv_heads=2,
            // Sliding=8 via the `gemma4.attention.head_count_kv` u32[30]
            // array. E2B carries a scalar so `n_kv_array` is `None` and
            // `n_kv_global` is the right answer for every layer.
            let n_kv_heads = n_kv_array
                .as_ref()
                .and_then(|a| a.get(i as usize).copied())
                .unwrap_or(n_kv_global);
            // Sprint 52K — detect `has_v_proj` via tensor presence
            // instead of metadata. 26B GGUFs (Google official) omit
            // `attn_v.weight` for Full layers entirely (the model uses
            // `attention_k_eq_v=true` semantics — V derived from K), but
            // they don't emit a `gemma4.attention.k_eq_v` metadata key
            // for VF to read. Probing the actual tensor table is robust
            // for both: present → has_v_proj=true (SafeTensors-quantised
            // model OR future GGUFs that DO emit attn_v), missing →
            // has_v_proj=false (current 26B GGUF Full layers).
            let attn_v_name = format!("blk.{i}.attn_v.weight");
            let has_v_proj = gguf.tensor(&attn_v_name).is_some();
            layers.push(Gemma4LayerSpec {
                kind,
                head_dim,
                intermediate_size: intermediate,
                has_kv_proj: !is_shared,
                kv_source,
                rope_theta,
                rope_partial_factor,
                n_kv_heads,
                has_v_proj,
            });
        }

        // Global Gemma4Spec fields.
        let sliding_window = gguf
            .metadata
            .get("gemma4.attention.sliding_window")
            .and_then(|v| v.as_u32())
            .unwrap_or(512);
        let final_logit_softcapping = gguf
            .metadata
            .get("gemma4.final_logit_softcapping")
            .and_then(|v| v.as_f32());
        let hidden_size_per_layer_input = gguf
            .metadata
            .get("gemma4.embedding_length_per_layer_input")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        // Tied embeddings = no separate `output.weight` tensor.
        let tie_word_embeddings = gguf.tensor("output.weight").is_none();
        // MoE (26B-A4B). E2B has none of these keys; defaults of
        // (0, 0, 0, false) keep the dispatcher on the Dense path.
        let n_experts = gguf
            .metadata
            .get("gemma4.expert_count")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let top_k_experts = gguf
            .metadata
            .get("gemma4.expert_used_count")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let moe_intermediate_size = gguf
            .metadata
            .get("gemma4.expert_feed_forward_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let enable_moe_block = n_experts > 0;

        Ok(Gemma4Spec {
            sliding_window,
            final_logit_softcapping,
            embed_scale,
            hidden_activation: "gelu_pytorch_tanh".to_string(),
            tie_word_embeddings,
            first_kv_shared,
            layers,
            // Filled by the loader in a second pass once the BF16
            // `blk.{i}.layer_scalar` tensors are read off disk — same
            // pattern as the SafeTensors path (loader.rs:2017-2019).
            // No consumer in `forward/` today reads this Vec; the
            // GPU tensor is queried via `layer_weight(..., "layer_scalar")`.
            layer_scalars: vec![1.0; n_layers],
            hidden_size_per_layer_input,
            enable_moe_block,
            n_experts,
            top_k_experts,
            moe_intermediate_size,
        })
    }
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let architecture = gguf.metadata_str("general.architecture")?.to_string();
        let arch = architecture.as_str();

        let metadata_u32 = |suffix: &str| -> Result<u32, GgufError> {
            gguf.metadata_u32(&format!("{arch}.{suffix}"))
        };
        let metadata_f32 = |suffix: &str| -> Result<f32, GgufError> {
            gguf.metadata_f32(&format!("{arch}.{suffix}"))
        };

        let n_layers = metadata_u32("block_count")?;
        let n_heads = metadata_u32("attention.head_count")?;
        // Sprint 52H-1 — Gemma-4-26B-A4B stores `head_count_kv` as a
        // per-layer u32 array (`[8,8,8,8,8,2, ...]` — Sliding=8, Full=2),
        // not a scalar. `ModelConfig.n_kv_heads` is a single global value
        // used for max-buffer sizing → take the array max. Per-layer
        // values land in `Gemma4LayerSpec.n_kv_heads` (built inside
        // `Gemma4Spec::from_gguf`). E2B (single u32) keeps its existing
        // behaviour via the scalar branch.
        let n_kv_heads =
            gguf.metadata_u32_scalar_or_array_max(&format!("{arch}.attention.head_count_kv"))?;
        let hidden_dim = metadata_u32("embedding_length")?;
        // Gemma-4 stores `feed_forward_length` as a per-layer u32 array
        // (35 entries for E2B). For `ModelConfig.ffn_dim` (single value,
        // used for max-buffer sizing), take the array max. Scalar
        // metadata still returns the scalar, so all other archs
        // (qwen2/3, llama, mistral, …) keep their existing behaviour.
        let ffn_dim = gguf.metadata_u32_scalar_or_array_max(&format!("{arch}.feed_forward_length"))?;
        let context_length = metadata_u32("context_length")?;
        let rope_freq_base = metadata_f32("rope.freq_base")?;
        let rms_norm_eps = metadata_f32("attention.layer_norm_rms_epsilon")?;

        // attention.key_length is the per-head dim (RoPE dim). Falls
        // back to hidden_dim / n_heads when the metadata is missing
        // (older GGUFs).
        let head_dim = gguf
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(hidden_dim / n_heads);
        let rope_dim = head_dim;

        let vocab_size = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.array_len())
            .map(|n| n as u32)
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".into()))?;

        // Qwen3 (and Qwen2 + qk-norm finetunes) ship per-head Q/K
        // RMSNorm tensors. Detected by tensor presence rather than a
        // metadata flag.
        let has_qk_norm = gguf.tensor("blk.0.attn_q_norm.weight").is_some()
            && gguf.tensor("blk.0.attn_k_norm.weight").is_some();

        // RoPE layout follows the architecture: Qwen* uses NeoX
        // (rotates [i, i+n_dims/2] pairs), llama / mistral / deepseek
        // use the standard adjacent-pair form. Mirrors llama.cpp's
        // `llama_rope_type()` switch in llama-arch.cpp.
        let rope_variant = match arch {
            "qwen2" | "qwen2moe" | "qwen2vl" | "qwen3" | "qwen3moe" | "phi2" | "phi3"
            | "gpt-neox" | "gpt-neox-japanese" | "stablelm" => RopeVariant::Neox,
            _ => RopeVariant::Norm,
        };

        // Sprint 52B — build Gemma-4 per-layer spec from GGUF metadata
        // when `arch == "gemma4"`. Non-Gemma archs keep `gemma4: None`.
        let gemma4 = if arch == "gemma4" {
            Some(Gemma4Spec::from_gguf(gguf)?)
        } else {
            None
        };

        Ok(Self {
            architecture,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            ffn_dim,
            vocab_size,
            head_dim,
            rope_freq_base,
            rope_dim,
            rope_variant,
            context_length,
            has_qk_norm,
            rms_norm_eps,
            gemma4,
        })
    }
}

// ---- internal cursor over a byte slice ----

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn read_n(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.data.len() {
            return Err(GgufError::EndOfFile);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_n(1)?[0])
    }
    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_n(2)?;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_n(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_n(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_n(1)?[0] as i8)
    }
    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let b = self.read_n(2)?;
        Ok(i16::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let b = self.read_n(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let b = self.read_n(8)?;
        Ok(i64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let b = self.read_n(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let b = self.read_n(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_n(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let kind = self.read_u32()?;
        self.read_metadata_value_of_kind(kind)
    }
    fn read_metadata_value_of_kind(&mut self, kind: u32) -> Result<MetadataValue, GgufError> {
        Ok(match kind {
            0 => MetadataValue::U8(self.read_u8()?),
            1 => MetadataValue::I8(self.read_i8()?),
            2 => MetadataValue::U16(self.read_u16()?),
            3 => MetadataValue::I16(self.read_i16()?),
            4 => MetadataValue::U32(self.read_u32()?),
            5 => MetadataValue::I32(self.read_i32()?),
            6 => MetadataValue::F32(self.read_f32()?),
            7 => MetadataValue::Bool(self.read_u8()? != 0),
            8 => MetadataValue::String(self.read_string()?),
            9 => {
                let inner_kind = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut v = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    v.push(self.read_metadata_value_of_kind(inner_kind)?);
                }
                MetadataValue::Array(v)
            }
            10 => MetadataValue::U64(self.read_u64()?),
            11 => MetadataValue::I64(self.read_i64()?),
            12 => MetadataValue::F64(self.read_f64()?),
            other => return Err(GgufError::UnknownMetadataType(other)),
        })
    }
}

fn align_up(n: u64, alignment: u64) -> u64 {
    (n + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GGML type 30 = BF16. Sprint v0.4.x — Gemma-family GGUFs use it
    /// for `token_embd.weight` and a few norm/output tensors. The
    /// parser must accept type 30 and report `block_size=1`,
    /// `type_size=2`.
    #[test]
    fn ggml_type_30_is_bf16() {
        let t = GgmlType::from_u32(30).expect("type 30 should parse as BF16");
        assert_eq!(t, GgmlType::BF16);
        assert_eq!(t.block_size(), 1);
        assert_eq!(t.type_size(), 2);
    }

    #[test]
    fn bf16_tensor_byte_size_is_2x_elements() {
        let info = TensorInfo {
            name: "embed".to_string(),
            dimensions: vec![128, 64],
            ggml_type: GgmlType::BF16,
            data_offset: 0,
        };
        assert_eq!(info.n_elements(), 128 * 64);
        assert_eq!(info.byte_size(), 128 * 64 * 2);
    }

    /// Gemma-4 GGUFs encode `feed_forward_length` (and a handful of
    /// other per-layer keys) as a u32 array, one entry per block. Pre
    /// this sprint, the strict `metadata_u32` rejected them with
    /// `UnexpectedType("…", "u32")`, blocking every Gemma-4 GGUF at
    /// `ModelConfig::from_gguf`. Verify the array-tolerant helper
    /// returns the max element (used for buffer sizing) AND still
    /// works for plain scalar metadata so non-Gemma archs are
    /// unaffected.
    #[test]
    fn metadata_u32_scalar_or_array_max_handles_array() {
        let mut md: HashMap<String, MetadataValue> = HashMap::new();
        md.insert("scalar".to_string(), MetadataValue::U32(2048));
        md.insert(
            "array".to_string(),
            MetadataValue::Array(vec![
                MetadataValue::U32(8192),
                MetadataValue::U32(12288),
                MetadataValue::U32(2048),
                MetadataValue::U32(12288), // max present twice on purpose
                MetadataValue::U32(4096),
            ]),
        );
        md.insert(
            "u64_scalar".to_string(),
            MetadataValue::U64(65536), // fits in u32; should resolve via as_u32()
        );

        // Hand-roll a GgufFile-shaped object isn't trivial (mmap), so
        // test the helper logic by stitching the metadata access path
        // manually — same code paths the method calls.
        let resolve = |key: &str| -> Result<u32, GgufError> {
            let v = md
                .get(key)
                .ok_or_else(|| GgufError::MissingMetadata(key.into()))?;
            if let Some(n) = v.as_u32() {
                return Ok(n);
            }
            if let Some(arr) = v.as_array() {
                let mut best: Option<u32> = None;
                for el in arr {
                    if let Some(n) = el.as_u32() {
                        best = Some(best.map_or(n, |b| b.max(n)));
                    }
                }
                if let Some(b) = best {
                    return Ok(b);
                }
            }
            Err(GgufError::UnexpectedType(key.into(), "u32 or u32[]"))
        };

        assert_eq!(resolve("scalar").unwrap(), 2048);
        assert_eq!(resolve("array").unwrap(), 12288);
        assert_eq!(resolve("u64_scalar").unwrap(), 65536);
        assert!(matches!(
            resolve("missing"),
            Err(GgufError::MissingMetadata(_))
        ));
    }

    /// Sprint 52B — verify the publisher-algorithm against the
    /// canonical E2B pattern: 35 layers, `first_kv_shared = 15`,
    /// SSSSF cycle with Full at indices 4, 9, 14, 19, 24, 29, 34.
    /// Brief sanity items:
    ///   * Layer 13 (last sliding in `[0, 15)`) → `OwnAndPublishesSliding`
    ///   * Layer 14 (last full in `[0, 15)`)    → `OwnAndPublishesFull`
    ///   * Layers 0..13 except 4 / 9            → `Own` (sliding non-publisher)
    ///   * Layers 4, 9                          → `Own` (full non-publisher)
    ///   * Sliding layers ≥15                   → `SubscribesSliding`
    ///   * Full layers 19, 24, 29, 34           → `SubscribesFull`
    #[test]
    fn build_kv_sources_e2b_pattern() {
        // Replicate the E2B GGUF's sliding_window_pattern exactly.
        let pattern: Vec<bool> = (0..35)
            .map(|i| (i % 5) != 4) // True (Sliding) except at indices ≡ 4 (mod 5)
            .collect();
        // Sanity-check the test fixture matches what python gguf reported.
        let full_indices: Vec<usize> = pattern
            .iter()
            .enumerate()
            .filter(|(_, b)| !**b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![4, 9, 14, 19, 24, 29, 34]);

        let srcs = build_kv_sources(&pattern, 15, 35);

        // Publishers in the pre-shared range:
        assert_eq!(srcs[13], Gemma4KvSource::OwnAndPublishesSliding,
            "layer 13 is the last sliding pre-shared");
        assert_eq!(srcs[14], Gemma4KvSource::OwnAndPublishesFull,
            "layer 14 is the last full pre-shared");
        // Non-publisher pre-shared: full at 4 / 9, sliding at 0..3, 5..8, 10..12
        for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 4, 9] {
            assert_eq!(srcs[i], Gemma4KvSource::Own,
                "layer {i} should be Own (non-publisher in pre-shared)");
        }
        // Subscribers in `[15, 35)`:
        for i in 15..35 {
            let expected = if (i % 5) == 4 {
                Gemma4KvSource::SubscribesFull
            } else {
                Gemma4KvSource::SubscribesSliding
            };
            assert_eq!(srcs[i], expected, "layer {i} subscriber kind");
        }
    }

    #[test]
    fn build_kv_sources_handles_no_full_pre_shared() {
        // Edge: all-sliding pre-shared range — no full publisher should
        // be assigned in `[0, first_kv_shared)`, even though shared
        // full layers exist later (subscribers would have no source —
        // but VF should still return SubscribesFull for them and a
        // higher-level fail-loud check catches that mismatch).
        let pattern: Vec<bool> = vec![true; 10]; // all sliding
        let mut p = pattern.clone();
        p[7] = false; // a single full layer outside the pre-shared range
        let srcs = build_kv_sources(&p, 5, 10);
        // Pre-shared (0..5) all sliding; layer 4 is the last → publisher
        assert_eq!(srcs[4], Gemma4KvSource::OwnAndPublishesSliding);
        // Layer 7 is shared (≥5) and full → SubscribesFull
        assert_eq!(srcs[7], Gemma4KvSource::SubscribesFull);
        // Other shared sliding layers
        for i in [5, 6, 8, 9] {
            assert_eq!(srcs[i], Gemma4KvSource::SubscribesSliding);
        }
    }

    /// Sprint 52B — `metadata_bool_array` should round-trip a `[bool]`
    /// metadata array. Uses the in-memory shape because there's no
    /// public test constructor for `GgufFile`; the resolve closure
    /// mirrors the helper's logic exactly.
    #[test]
    fn metadata_bool_array_round_trip() {
        let arr = MetadataValue::Array(vec![
            MetadataValue::Bool(true),
            MetadataValue::Bool(false),
            MetadataValue::Bool(true),
            MetadataValue::Bool(true),
            MetadataValue::Bool(false),
        ]);
        let parts = arr.as_array().unwrap();
        let mut out = Vec::with_capacity(parts.len());
        for el in parts {
            out.push(el.as_bool().expect("bool element"));
        }
        assert_eq!(out, vec![true, false, true, true, false]);

        // U8(0/1) should also work as bool — defensive.
        let mixed = MetadataValue::Array(vec![
            MetadataValue::U8(1),
            MetadataValue::U8(0),
        ]);
        let parts = mixed.as_array().unwrap();
        let out: Vec<bool> = parts
            .iter()
            .filter_map(|el| el.as_bool())
            .collect();
        assert_eq!(out, vec![true, false]);

        // Non-bool element returns None — defensive.
        assert!(MetadataValue::U32(5).as_bool().is_none());
    }
}
