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
    /// Gemma-4 text-config metadata. Populated only for `model_type ==
    /// "gemma4"`; `None` for every other architecture. The flat fields
    /// above (`hidden_size`, `num_attention_heads`, …) carry the *text*
    /// values for Gemma-4 too — `from_dir` flattens `text_config` into
    /// the top level before deserialisation so the rest of the loader
    /// stays uniform.
    #[serde(default, skip)]
    pub gemma4: Option<Gemma4TextMeta>,
}

/// Gemma-4 text-decoder-specific metadata read from `config.json`'s
/// nested `text_config` block. Sprint 43B-1: parsed and surfaced;
/// consumers added incrementally over 43B-2 and 43D.
#[derive(Debug, Clone)]
pub struct Gemma4TextMeta {
    /// `["sliding_attention", "full_attention", …]` — exactly
    /// `num_hidden_layers` entries. Sprint 43B-2 routes the per-layer
    /// attention dispatch off this.
    pub layer_types: Vec<Gemma4LayerType>,
    /// 512 for E2B-it. Applied only on `Sliding` layers.
    pub sliding_window: u32,
    /// 256 for E2B (sliding layers).
    pub head_dim_sliding: u32,
    /// 512 for E2B (full layers).
    pub head_dim_full: u32,
    /// 6144 for E2B (layers 0..first_kv_shared).
    pub intermediate_size: u32,
    /// Whether KV-shared layers double the MLP intermediate size to
    /// compensate for skipped K/V compute. `true` for E2B (12288 on
    /// shared layers).
    pub use_double_wide_mlp: bool,
    /// 256 for E2B — slice width of the per-layer-embedding for each
    /// decoder layer.
    pub hidden_size_per_layer_input: u32,
    /// Equal to `vocab_size` for E2B (262 144). Stored separately
    /// because Gemma's HF code allows divergence in principle.
    pub vocab_size_per_layer_input: u32,
    /// Number of *layers from the tail* that share KV state with an
    /// earlier layer of the same `layer_types[i]`. 20 for E2B → first
    /// shared layer is `num_hidden_layers - 20 = 15`.
    pub num_kv_shared_layers: u32,
    /// 30.0 — applied to the final logits before sampling:
    /// `logits = cap × tanh(logits / cap)`.
    pub final_logit_softcapping: Option<f32>,
    /// `gelu_pytorch_tanh` for E2B. Captured raw; consumer matches.
    pub hidden_activation: String,
    /// RoPE θ for sliding-attention layers (10 000 for E2B).
    pub sliding_rope_theta: f32,
    /// RoPE θ for full-attention layers (1 000 000 for E2B).
    pub full_rope_theta: f32,
    /// Proportional-rotation factor for full-attention RoPE (0.25
    /// for E2B → only the first 25 % of the head dim is rotated).
    pub full_rope_partial_factor: Option<f32>,
}

/// Per-layer attention kind in a Gemma-4 stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4LayerType {
    Sliding,
    Full,
}

#[derive(Debug, Deserialize)]
pub struct QuantizationConfig {
    pub format: Option<String>,
    pub quant_method: Option<String>,
    /// Tensors *exempt* from quantization (e.g. lm_head on neuralmagic
    /// FP8 models). Names match the HF tensor-name schema.
    #[serde(default)]
    pub ignore: Vec<String>,
    /// Sprint 35 — block-wise FP8 (Qwen3-FP8, DeepSeek-V3-FP8). When
    /// present, weights ship with a 2D scale grid `[N/block_n, K/block_k]`
    /// instead of a per-tensor or per-channel scale. Stored verbatim
    /// from `config.json`'s `quantization_config.weight_block_size`.
    #[serde(default)]
    pub weight_block_size: Option<[u32; 2]>,
    /// Sprint 35 — `e4m3` for Qwen3-FP8 / DeepSeek-V3-FP8. Carried
    /// through verbatim; the loader rejects anything that isn't
    /// `e4m3` to fail loud rather than silently mis-dequantize.
    #[serde(default)]
    pub fmt: Option<String>,
}

impl HfConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;

        let model_type = raw
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: missing model_type", path.display()))?;

        // Gemma-4 nests the language-model fields under `text_config`,
        // alongside `vision_config` and `audio_config`. Flatten those
        // fields onto a synthetic top-level object so the rest of the
        // loader (which assumes a flat Llama / Qwen-style schema) keeps
        // working unchanged. We extract the Gemma-4 specific extras
        // separately into `Gemma4TextMeta`.
        if model_type == "gemma4" {
            let text = raw
                .get("text_config")
                .ok_or_else(|| format!("{}: gemma4 missing text_config", path.display()))?;
            let gemma_meta = parse_gemma4_text_meta(text)
                .map_err(|e| format!("{}: {e}", path.display()))?;

            let mut flat = serde_json::Map::new();
            flat.insert("model_type".into(), serde_json::Value::String("gemma4".into()));
            if let Some(a) = raw.get("architectures") {
                flat.insert("architectures".into(), a.clone());
            }
            for k in [
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "vocab_size",
                "rms_norm_eps",
                "max_position_embeddings",
                "tie_word_embeddings",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "rope_scaling",
                "torch_dtype",
            ] {
                if let Some(v) = text.get(k) {
                    flat.insert(k.into(), v.clone());
                }
            }
            // rope_theta lives under `rope_parameters.sliding_attention.rope_theta`
            // in Gemma-4. The flat field is consumed only for
            // single-RoPE-config architectures; for Gemma-4 we publish
            // the sliding θ here so the legacy `cfg.rope_theta` is at
            // least populated, and ship the full pair via `gemma4`.
            flat.insert(
                "rope_theta".into(),
                serde_json::json!(gemma_meta.sliding_rope_theta),
            );
            // top-level `dtype` (Gemma-4) → torch_dtype synonym
            if !flat.contains_key("torch_dtype") {
                if let Some(d) = raw.get("dtype").and_then(|v| v.as_str()) {
                    flat.insert("torch_dtype".into(), serde_json::Value::String(d.into()));
                }
            }
            // tie_word_embeddings can also live at top level.
            if !flat.contains_key("tie_word_embeddings") {
                if let Some(t) = raw.get("tie_word_embeddings") {
                    flat.insert("tie_word_embeddings".into(), t.clone());
                }
            }
            let synth = serde_json::Value::Object(flat);
            let mut cfg: HfConfig = serde_json::from_value(synth)
                .map_err(|e| format!("{}: synthesised flat parse: {e}", path.display()))?;
            cfg.gemma4 = Some(gemma_meta);
            return Ok(cfg);
        }

        let cfg: HfConfig = serde_json::from_value(raw)
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

fn parse_gemma4_text_meta(text: &serde_json::Value) -> Result<Gemma4TextMeta, String> {
    let layer_types_raw = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or("gemma4 text_config.layer_types missing or not an array")?;
    let mut layer_types = Vec::with_capacity(layer_types_raw.len());
    for v in layer_types_raw {
        match v.as_str() {
            Some("sliding_attention") => layer_types.push(Gemma4LayerType::Sliding),
            Some("full_attention") => layer_types.push(Gemma4LayerType::Full),
            other => {
                return Err(format!(
                    "gemma4 layer_types entry not recognised: {other:?}"
                ))
            }
        }
    }

    let u32_field = |key: &str| -> Result<u32, String> {
        text.get(key)
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .ok_or_else(|| format!("gemma4 text_config.{key} missing or not u64"))
    };
    let f32_field_opt = |key: &str| -> Option<f32> {
        text.get(key).and_then(|v| v.as_f64()).map(|x| x as f32)
    };
    let bool_field = |key: &str, default: bool| -> bool {
        text.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    };

    let head_dim_sliding = u32_field("head_dim")?;
    // global_head_dim is optional in the schema; falls back to head_dim
    // when the model uses a single head dim everywhere.
    let head_dim_full = text
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(head_dim_sliding);
    let intermediate_size = u32_field("intermediate_size")?;
    let use_double_wide_mlp = bool_field("use_double_wide_mlp", false);
    let hidden_size_per_layer_input = u32_field("hidden_size_per_layer_input")?;
    let vocab_size_per_layer_input = u32_field("vocab_size_per_layer_input")?;
    let num_kv_shared_layers = text
        .get("num_kv_shared_layers")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(0);
    let final_logit_softcapping = f32_field_opt("final_logit_softcapping");
    let sliding_window = u32_field("sliding_window")?;

    let hidden_activation = text
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| "gelu_pytorch_tanh".into());

    // Two-RoPE-block: one per layer-type.
    let rope_params = text
        .get("rope_parameters")
        .ok_or("gemma4 text_config.rope_parameters missing")?;
    let sliding_rope_theta = rope_params
        .get("sliding_attention")
        .and_then(|s| s.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .ok_or("gemma4 rope_parameters.sliding_attention.rope_theta missing")?
        as f32;
    let full_block = rope_params
        .get("full_attention")
        .ok_or("gemma4 rope_parameters.full_attention missing")?;
    let full_rope_theta = full_block
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .ok_or("gemma4 rope_parameters.full_attention.rope_theta missing")?
        as f32;
    let full_rope_partial_factor = full_block
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .map(|x| x as f32);

    Ok(Gemma4TextMeta {
        layer_types,
        sliding_window,
        head_dim_sliding,
        head_dim_full,
        intermediate_size,
        use_double_wide_mlp,
        hidden_size_per_layer_input,
        vocab_size_per_layer_input,
        num_kv_shared_layers,
        final_logit_softcapping,
        hidden_activation,
        sliding_rope_theta,
        full_rope_theta,
        full_rope_partial_factor,
    })
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
    fn parses_gemma4_text_config_via_from_dir() {
        // Use a tempdir so we exercise from_dir's nested-flatten path
        // (parse_gemma4_text_meta is invoked indirectly).
        let dir = std::env::temp_dir().join("vf_gemma4_cfg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_json = r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "dtype": "bfloat16",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "num_hidden_layers": 35,
                "vocab_size": 262144,
                "rms_norm_eps": 1.0e-6,
                "max_position_embeddings": 131072,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 512,
                "use_double_wide_mlp": true,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "num_kv_shared_layers": 20,
                "final_logit_softcapping": 30.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "tie_word_embeddings": true,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        }"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let cfg = HfConfig::from_dir(&dir).expect("gemma4 config parses");
        assert_eq!(cfg.model_type, "gemma4");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.n_kv_heads(), 1);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rope_theta, 10000.0);

        let g = cfg.gemma4.expect("gemma4 meta populated");
        assert_eq!(g.layer_types.len(), 35);
        assert_eq!(g.layer_types[0], Gemma4LayerType::Sliding);
        assert_eq!(g.layer_types[4], Gemma4LayerType::Full);
        assert_eq!(g.layer_types[34], Gemma4LayerType::Full);
        assert_eq!(g.head_dim_sliding, 256);
        assert_eq!(g.head_dim_full, 512);
        assert_eq!(g.sliding_window, 512);
        assert_eq!(g.intermediate_size, 6144);
        assert!(g.use_double_wide_mlp);
        assert_eq!(g.num_kv_shared_layers, 20);
        assert_eq!(g.hidden_size_per_layer_input, 256);
        assert_eq!(g.vocab_size_per_layer_input, 262144);
        assert_eq!(g.final_logit_softcapping, Some(30.0));
        assert_eq!(g.hidden_activation, "gelu_pytorch_tanh");
        assert_eq!(g.sliding_rope_theta, 10000.0);
        assert_eq!(g.full_rope_theta, 1000000.0);
        assert_eq!(g.full_rope_partial_factor, Some(0.25));

        std::fs::remove_dir_all(&dir).ok();
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
