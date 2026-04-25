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
            _ => return Err(GgufError::UnknownTensorType(v)),
        })
    }

    /// Number of elements per quantisation block. 1 for non-quant types.
    pub fn block_size(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
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

    pub fn metadata_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.into()))?
            .as_f32()
            .ok_or_else(|| GgufError::UnexpectedType(key.into(), "f32"))
    }
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
    pub context_length: u32,
    pub has_qk_norm: bool,
    pub rms_norm_eps: f32,
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
        let n_kv_heads = metadata_u32("attention.head_count_kv")?;
        let hidden_dim = metadata_u32("embedding_length")?;
        let ffn_dim = metadata_u32("feed_forward_length")?;
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
            context_length,
            has_qk_norm,
            rms_norm_eps,
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
