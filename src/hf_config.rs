//! HuggingFace `config.json` parser (Sprint 20-M1).
//!
//! Reads the architectural metadata that GGUF carries inline but
//! SafeTensors models keep in a separate `config.json`. Output is
//! a plain struct that mirrors the fields VulkanForge already cares
//! about so the loader can synthesise a `ModelConfig` without
//! reaching into HF-specific names elsewhere in the codebase.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct HfConfig {
    pub architectures: Option<Vec<String>>,
    pub model_type: String,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    /// Often absent for non-GQA models — defaults to `num_attention_heads`.
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Optional Llama-3 RoPE scaling (`{factor, low_freq_factor,
    /// high_freq_factor, original_max_position_embeddings, rope_type}`).
    /// Stored generically; only the fields VF currently consumes are
    /// extracted by `Llama3RopeScaling::from_value`.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub max_position_embeddings: Option<u32>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    /// Compressed-tensors quantization metadata for FP8 models. Carries
    /// the `naive-quantized` info we need to know when a Linear has a
    /// per-tensor `weight_scale`.
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    /// Token IDs for the BOS / EOS / PAD specials. Llama-3 emits an
    /// EOS *array* (`[128001, 128008, 128009]`) — we keep the raw value
    /// so the caller can pick.
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct QuantizationConfig {
    pub format: Option<String>,
    pub quant_method: Option<String>,
    /// Tensors *exempt* from quantization (e.g. lm_head on neuralmagic
    /// FP8 models). Names match the HF tensor-name schema.
    #[serde(default)]
    pub ignore: Vec<String>,
}

impl HfConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let cfg: HfConfig = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        Ok(cfg)
    }

    pub fn n_kv_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Returns `true` iff `tensor_name` is on the quantization-ignore
    /// list (carried unquantized in the SafeTensors files even though
    /// the model is otherwise FP8).
    pub fn is_quant_excluded(&self, tensor_name: &str) -> bool {
        self.quantization_config
            .as_ref()
            .map(|q| q.ignore.iter().any(|n| n == tensor_name))
            .unwrap_or(false)
    }
}

/// Llama-3 RoPE scaling — only emitted by the 3.x family. VulkanForge
/// applies the rescaled frequency at GEMV-side RoPE; this struct just
/// carries the four scalars we need from `config.rope_scaling`.
#[derive(Debug, Clone)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
}

impl Llama3RopeScaling {
    pub fn from_config(cfg: &HfConfig) -> Option<Self> {
        let v = cfg.rope_scaling.as_ref()?;
        if v.get("rope_type").and_then(|x| x.as_str()) != Some("llama3") {
            return None;
        }
        Some(Self {
            factor: v.get("factor")?.as_f64()? as f32,
            low_freq_factor: v.get("low_freq_factor")?.as_f64()? as f32,
            high_freq_factor: v.get("high_freq_factor")?.as_f64()? as f32,
            original_max_position_embeddings: v
                .get("original_max_position_embeddings")?
                .as_u64()? as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_llama_config() {
        let raw = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        }"#;
        let cfg: HfConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.n_kv_heads(), 8);
        assert_eq!(cfg.head_dim(), 128);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn parses_llama3_rope_scaling() {
        let raw = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let cfg: HfConfig = serde_json::from_str(raw).unwrap();
        let scaling = Llama3RopeScaling::from_config(&cfg).unwrap();
        assert_eq!(scaling.factor, 8.0);
        assert_eq!(scaling.original_max_position_embeddings, 8192);
    }

    #[test]
    fn parses_quantization_ignore_list() {
        let raw = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "quantization_config": {
                "format": "naive-quantized",
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head"]
            }
        }"#;
        let cfg: HfConfig = serde_json::from_str(raw).unwrap();
        assert!(cfg.is_quant_excluded("lm_head"));
        assert!(!cfg.is_quant_excluded("model.layers.0.self_attn.q_proj"));
    }
}
