//! HuggingFace SafeTensors loader (Sprint 20-M1).
//!
//! Supports single-file (`model.safetensors`) and sharded multi-file
//! (`model.safetensors.index.json` + `model-NNNNN-of-MMMMM.safetensors`)
//! layouts. Tensor data is memory-mapped — `tensor_bytes` returns a
//! `&[u8]` view into the mmap, no copies.
//!
//! Format reference (HuggingFace, MIT-licensed):
//!   [u64 LE: header_size][header_size bytes: JSON][raw tensor data]
//!
//! The JSON header is a flat `Map<String, TensorEntry>` plus an optional
//! `__metadata__` key. Each entry has `dtype`, `shape`, and
//! `data_offsets: [start, end]` in bytes relative to the start of the
//! raw-data block (i.e. byte offset 8 + header_size into the file).
//!
//! Multi-file: `model.safetensors.index.json` carries `weight_map:
//! {"tensor": "model-00001-of-00002.safetensors"}` to route each tensor
//! to its host shard. Each shard has its own header + offsets.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    /// FP8 E4M3 (1 byte/elem, 4 exp + 3 mantissa). Native target dtype
    /// for Sprint 20.
    F8E4M3,
    /// FP8 E5M2 (1 byte/elem, 5 exp + 2 mantissa). Recognized but not
    /// yet routed to a shader; SafeTensors models seen in the wild
    /// typically use E4M3 for weights.
    F8E5M2,
    F16,
    BF16,
    F32,
}

impl TensorDtype {
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "F8_E4M3" => TensorDtype::F8E4M3,
            "F8_E5M2" => TensorDtype::F8E5M2,
            "F16" => TensorDtype::F16,
            "BF16" => TensorDtype::BF16,
            "F32" => TensorDtype::F32,
            _ => return None,
        })
    }

    pub fn bytes_per_elem(self) -> usize {
        match self {
            TensorDtype::F8E4M3 | TensorDtype::F8E5M2 => 1,
            TensorDtype::F16 | TensorDtype::BF16 => 2,
            TensorDtype::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    /// Index into `SafeTensorsFile::shards` where the bytes live.
    pub shard_idx: usize,
    /// Byte offset of the first element relative to the shard's
    /// raw-data block (i.e. into `Shard::data`).
    pub start: usize,
    /// Byte offset one past the last element.
    pub end: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Deserialize)]
struct RawHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One memory-mapped `.safetensors` shard. The raw tensor bytes start
/// at `8 + header_size` into the file; we carve that suffix once and
/// keep just the data slice's offset for cheap lookups.
struct Shard {
    mmap: Mmap,
    data_start: usize,
}

impl Shard {
    fn open(path: &Path) -> Result<(Self, HashMap<String, RawHeaderEntry>), String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        if mmap.len() < 8 {
            return Err(format!("safetensors {}: file shorter than 8-byte header", path.display()));
        }
        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if mmap.len() < 8 + header_size {
            return Err(format!(
                "safetensors {}: declared header_size {header_size} exceeds file length {}",
                path.display(), mmap.len(),
            ));
        }
        let header_slice = &mmap[8..8 + header_size];
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header_slice)
            .map_err(|e| format!("safetensors {}: header JSON parse: {e}", path.display()))?;
        let mut entries = HashMap::with_capacity(raw.len());
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let parsed: RawHeaderEntry = serde_json::from_value(value)
                .map_err(|e| format!("safetensors {}: tensor '{name}' header malformed: {e}", path.display()))?;
            entries.insert(name, parsed);
        }
        let shard = Shard {
            mmap,
            data_start: 8 + header_size,
        };
        Ok((shard, entries))
    }

    fn slice(&self, start: usize, end: usize) -> &[u8] {
        &self.mmap[self.data_start + start .. self.data_start + end]
    }
}

pub struct SafeTensorsFile {
    /// Tensor name → metadata (including which shard hosts it).
    pub tensors: HashMap<String, TensorInfo>,
    shards: Vec<Shard>,
}

impl SafeTensorsFile {
    /// Open a SafeTensors model from a directory. Auto-detects:
    /// * `<dir>/model.safetensors` (single-file)
    /// * `<dir>/model.safetensors.index.json` + shard files
    ///   (multi-file)
    pub fn open(dir: &Path) -> Result<Self, String> {
        let single = dir.join("model.safetensors");
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            Self::open_multi_file(dir, &index)
        } else if single.exists() {
            Self::open_single_file(&single)
        } else {
            Err(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                dir.display(),
            ))
        }
    }

    fn open_single_file(path: &Path) -> Result<Self, String> {
        let (shard, entries) = Shard::open(path)?;
        let mut tensors = HashMap::with_capacity(entries.len());
        for (name, raw) in entries {
            let dtype = TensorDtype::from_str(&raw.dtype)
                .ok_or_else(|| format!("safetensors {}: unsupported dtype {} for tensor '{name}'", path.display(), raw.dtype))?;
            tensors.insert(name, TensorInfo {
                dtype,
                shape: raw.shape,
                shard_idx: 0,
                start: raw.data_offsets[0],
                end: raw.data_offsets[1],
            });
        }
        Ok(Self { tensors, shards: vec![shard] })
    }

    fn open_multi_file(dir: &Path, index_path: &Path) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let index_bytes = std::fs::read(index_path)
            .map_err(|e| format!("read {}: {e}", index_path.display()))?;
        let index: Index = serde_json::from_slice(&index_bytes)
            .map_err(|e| format!("parse {}: {e}", index_path.display()))?;

        // Collect distinct shard filenames preserving first-seen order.
        let mut shard_paths: Vec<PathBuf> = Vec::new();
        let mut shard_idx_for: HashMap<String, usize> = HashMap::new();
        for shard_name in index.weight_map.values() {
            if !shard_idx_for.contains_key(shard_name) {
                shard_idx_for.insert(shard_name.clone(), shard_paths.len());
                shard_paths.push(dir.join(shard_name));
            }
        }

        // Open each shard, gather raw entries.
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut per_shard_entries: Vec<HashMap<String, RawHeaderEntry>> =
            Vec::with_capacity(shard_paths.len());
        for path in &shard_paths {
            let (shard, entries) = Shard::open(path)?;
            shards.push(shard);
            per_shard_entries.push(entries);
        }

        // Build tensors, cross-checking that the index's claimed shard
        // matches the shard whose header actually carries the entry.
        let mut tensors = HashMap::with_capacity(index.weight_map.len());
        for (name, shard_name) in index.weight_map {
            let idx = *shard_idx_for.get(&shard_name).unwrap();
            let raw = per_shard_entries[idx].get(&name).ok_or_else(|| {
                format!("safetensors index claims '{name}' lives in {shard_name}, but its header doesn't list it")
            })?;
            let dtype = TensorDtype::from_str(&raw.dtype).ok_or_else(|| {
                format!("safetensors '{name}': unsupported dtype {}", raw.dtype)
            })?;
            tensors.insert(name, TensorInfo {
                dtype,
                shape: raw.shape.clone(),
                shard_idx: idx,
                start: raw.data_offsets[0],
                end: raw.data_offsets[1],
            });
        }

        Ok(Self { tensors, shards })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        self.shards[info.shard_idx].slice(info.start, info.end)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

/// Map a HuggingFace transformer tensor name to VulkanForge's internal
/// `blk.N.foo` / `output_norm` / `output` / `token_embd` schema. Returns
/// `None` for tensors we don't (yet) consume — most notably FP8
/// `*_scale_inv` / `*_scale` calibration tensors that vLLM emits but
/// VulkanForge ignores in M1.
///
/// Sprint 43B-1: Gemma-4 nests every text-decoder tensor under
/// `model.language_model.*` and adds layer-locals VF doesn't yet
/// consume (`layer_scalar`, `pre_feedforward_layernorm`, the
/// `per_layer_*` PLE tensors). The big PLE table
/// `embed_tokens_per_layer` (4.7 GB on E2B) and the projection /
/// gate / norm tensors return `None` here — they get loaded by the
/// PLE-aware path landing in Sprint 43D. The vision and audio towers
/// are skipped wholesale (text-only path).
pub fn hf_to_vf_name(hf: &str) -> Option<String> {
    // Gemma-4: strip the `model.language_model.` prefix and recurse
    // into the standard mapping. The recursive call may add Gemma-4
    // specific suffixes via the `gemma4_extra_layer_suffix` branch.
    if let Some(rest) = hf.strip_prefix("model.language_model.") {
        // Multimodal / PLE top-level tensors that VF doesn't (yet)
        // consume — return None so the loader skips them silently.
        if rest == "embed_tokens_per_layer.weight"
            || rest == "per_layer_model_projection.weight"
            || rest == "per_layer_projection_norm.weight"
        {
            return None;
        }
        // Re-route as if it were a flat `model.*` name.
        return hf_to_vf_name(&format!("model.{rest}"));
    }
    // Skip everything in the vision / audio towers (Gemma-4 multimodal).
    if hf.starts_with("model.vision_tower.")
        || hf.starts_with("model.audio_tower.")
        || hf.starts_with("model.embed_vision.")
        || hf.starts_with("model.embed_audio.")
    {
        return None;
    }

    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let vf_suffix = match suffix {
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        // Optional Q/K-norm (Qwen3-style models). Emit if they appear;
        // Llama-3 doesn't carry these.
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        // Sprint 24B — Qwen2-style attention biases. Expanded to FP32
        // at load time and applied after Q/K/V GEMV/GEMM, before RoPE.
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        // Sprint 43B-1 — Gemma-4-specific extra layer norms +
        // residual scalar. Mapped here so the loader uploads them
        // alongside the standard tensors; Sprint 43B-2 forward will
        // consume them.
        "pre_feedforward_layernorm.weight" => "ffn_pre_norm.weight",
        "post_feedforward_layernorm.weight" => "ffn_post_norm.weight",
        "self_attn.v_norm.weight" => "attn_v_norm.weight",
        "layer_scalar" => "layer_scalar",
        // Sprint 43B-1 — Gemma-4 per-layer-input modulation pieces.
        // Sprint 43D-3 wires the PLE forward block; map to `blk.N.*`
        // names so they go through the standard tensor-upload path.
        "per_layer_input_gate.weight" => "per_layer_input_gate.weight",
        "per_layer_projection.weight" => "per_layer_projection.weight",
        "post_per_layer_input_norm.weight" => "post_per_layer_input_norm.weight",
        // Sprint 51C — Gemma-4-26B-A4B MoE-block weights (parallel
        // Dense-MLP + MoE FFN per layer). Three additional norms
        // around the MoE branch + the packed expert/router tensors.
        // Sprint 51D will wire the actual loader uploads + dispatch;
        // 51C registers the names so the loader sees them as known.
        "post_feedforward_layernorm_1.weight" => "ffn_post_norm_1.weight",
        "pre_feedforward_layernorm_2.weight" => "ffn_pre_norm_2.weight",
        "post_feedforward_layernorm_2.weight" => "ffn_post_norm_2.weight",
        "experts.gate_up_proj" => "moe_experts.gate_up_proj",
        "experts.down_proj" => "moe_experts.down_proj",
        "router.proj.weight" => "moe_router.proj.weight",
        "router.scale" => "moe_router.scale",
        "router.per_expert_scale" => "moe_router.per_expert_scale",
        _ => return None,
    };
    Some(format!("blk.{layer}.{vf_suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_round_trip() {
        assert_eq!(TensorDtype::from_str("F8_E4M3"), Some(TensorDtype::F8E4M3));
        assert_eq!(TensorDtype::from_str("F8_E5M2"), Some(TensorDtype::F8E5M2));
        assert_eq!(TensorDtype::from_str("F16"), Some(TensorDtype::F16));
        assert_eq!(TensorDtype::from_str("BF16"), Some(TensorDtype::BF16));
        assert_eq!(TensorDtype::from_str("F32"), Some(TensorDtype::F32));
        assert_eq!(TensorDtype::from_str("nonsense"), None);
        assert_eq!(TensorDtype::F8E4M3.bytes_per_elem(), 1);
        assert_eq!(TensorDtype::F32.bytes_per_elem(), 4);
    }

    #[test]
    fn name_mapping_basics() {
        assert_eq!(
            hf_to_vf_name("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("blk.0.attn_q.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.layers.31.mlp.down_proj.weight").as_deref(),
            Some("blk.31.ffn_down.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.layers.5.input_layernorm.weight").as_deref(),
            Some("blk.5.attn_norm.weight"),
        );
        assert_eq!(hf_to_vf_name("model.layers.0.self_attn.q_proj.weight_scale"), None);
        assert_eq!(hf_to_vf_name("lm_head.weight").as_deref(), Some("output.weight"));
    }

    #[test]
    fn gemma4_prefix_stripping() {
        // Gemma-4 nests every text-decoder tensor under `language_model`.
        assert_eq!(
            hf_to_vf_name("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.4.self_attn.q_proj.weight").as_deref(),
            Some("blk.4.attn_q.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.7.mlp.down_proj.weight").as_deref(),
            Some("blk.7.ffn_down.weight"),
        );
    }

    #[test]
    fn gemma4_new_per_layer_suffixes() {
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.0.pre_feedforward_layernorm.weight").as_deref(),
            Some("blk.0.ffn_pre_norm.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.34.post_feedforward_layernorm.weight").as_deref(),
            Some("blk.34.ffn_post_norm.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.0.layer_scalar").as_deref(),
            Some("blk.0.layer_scalar"),
        );
        // q_norm + k_norm exist on Gemma-4 too (per-head dim).
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.0.self_attn.q_norm.weight").as_deref(),
            Some("blk.0.attn_q_norm.weight"),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.0.self_attn.v_norm.weight").as_deref(),
            Some("blk.0.attn_v_norm.weight"),
        );
    }

    #[test]
    fn gemma4_ple_and_multimodal_routing() {
        // Sprint 43D-3 — PLE top-level tensors stay None; the loader
        // pulls embed_tokens_per_layer + per_layer_projection_norm via
        // a separate path (PleData) and never via hf_to_vf_name.
        // per_layer_model_projection is for the AltUp path which VF
        // doesn't implement; intentionally still skipped.
        assert_eq!(
            hf_to_vf_name("model.language_model.embed_tokens_per_layer.weight"),
            None,
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.per_layer_model_projection.weight"),
            None,
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.per_layer_projection_norm.weight"),
            None,
        );
        // Sprint 43D-3 — per-layer PLE tensors NOW go through the
        // standard `blk.N.*` upload path so dispatch_layer can sample
        // them via layer_weight().
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.5.per_layer_input_gate.weight"),
            Some("blk.5.per_layer_input_gate.weight".into()),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.5.per_layer_projection.weight"),
            Some("blk.5.per_layer_projection.weight".into()),
        );
        assert_eq!(
            hf_to_vf_name("model.language_model.layers.5.post_per_layer_input_norm.weight"),
            Some("blk.5.post_per_layer_input_norm.weight".into()),
        );
        // Vision / audio towers — never wanted in text-only mode.
        assert_eq!(hf_to_vf_name("model.vision_tower.layers.0.attn.weight"), None);
        assert_eq!(hf_to_vf_name("model.audio_tower.encoder.0.weight"), None);
        assert_eq!(hf_to_vf_name("model.embed_vision.embedding_projection.weight"), None);
        assert_eq!(hf_to_vf_name("model.embed_audio.embedding_projection.weight"), None);
    }
}
