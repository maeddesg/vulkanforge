//! Sprint 20-M1 — inspect a SafeTensors model directory:
//! list tensors, dtypes, shapes, sizes; group by HF→VF name mapping.

use std::path::PathBuf;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::hf_config::HfConfig;
use vulkanforge::safetensors::{SafeTensorsFile, hf_to_vf_name};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: safetensors_inspect <model-dir>")?;
    let do_load = std::env::args().any(|a| a == "--load");

    println!("== Config ({}) ==", dir.display());
    let cfg = HfConfig::from_dir(&dir)?;
    println!(
        "  arch={:?} hidden={} layers={} heads={} kv_heads={} ffn={} vocab={} rope_theta={}",
        cfg.architectures.as_deref().unwrap_or(&[]),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.n_kv_heads(),
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.rope_theta,
    );
    if let Some(qc) = &cfg.quantization_config {
        println!(
            "  quant: format={:?} method={:?} ignore={:?}",
            qc.format, qc.quant_method, qc.ignore,
        );
    }

    println!();
    println!("== SafeTensors ==");
    let st = SafeTensorsFile::open(&dir)?;
    let mut by_dtype: std::collections::BTreeMap<&'static str, (usize, u64)> =
        Default::default();
    let mut mapped = 0usize;
    let mut unmapped = 0usize;
    let mut scale_count = 0usize;
    let mut input_scale_count = 0usize;
    let mut total_bytes = 0u64;
    for (name, info) in &st.tensors {
        let dtype_label = match info.dtype {
            vulkanforge::safetensors::TensorDtype::F8E4M3 => "F8_E4M3",
            vulkanforge::safetensors::TensorDtype::F8E5M2 => "F8_E5M2",
            vulkanforge::safetensors::TensorDtype::F16 => "F16",
            vulkanforge::safetensors::TensorDtype::BF16 => "BF16",
            vulkanforge::safetensors::TensorDtype::F32 => "F32",
        };
        let entry = by_dtype.entry(dtype_label).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += info.byte_len() as u64;
        total_bytes += info.byte_len() as u64;

        if name.ends_with(".weight_scale") {
            scale_count += 1;
        } else if name.ends_with(".input_scale") {
            input_scale_count += 1;
        } else if hf_to_vf_name(name).is_some() {
            mapped += 1;
        } else {
            unmapped += 1;
        }
    }
    println!("  tensors total: {}", st.tensors.len());
    for (label, (count, bytes)) in &by_dtype {
        println!(
            "  dtype {label:<8}: {count:>4} tensors, {:.2} GiB",
            *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    println!(
        "  total raw bytes: {:.2} GiB",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  mapped to VF names: {mapped}, weight_scale: {scale_count}, \
         input_scale: {input_scale_count}, unmapped: {unmapped}",
    );

    if do_load {
        println!();
        println!("== Loading to VRAM (LoadedModel::load_safetensors) ==");
        let dev = VulkanDevice::new()?;
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: dev.instance.clone(),
            device: dev.device.clone(),
            physical_device: dev.physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })?;
        let (model, _embed_cache, _hf) = LoadedModel::load_safetensors(&dev, &mut allocator, &dir)?;
        println!(
            "  loaded in {:.2} s, {:.2} GiB uploaded",
            model.upload_duration.as_secs_f64(),
            model.bytes_uploaded as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        let total = model.tensors.len();
        let with_scale = model.tensors.values()
            .filter(|t| t.scale_buffer.is_some())
            .count();
        println!("  GpuTensors: {total} (with weight_scale: {with_scale})");

        // Spot-check layer 0
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_norm.weight",
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
        ] {
            match model.tensor(name) {
                Some(t) => {
                    let scale = if t.scale_buffer.is_some() { "buf".to_string() } else { "-".to_string() };
                    println!(
                        "  {name:<28} {:?} {:?} {:>10} B  scale={}",
                        t.ggml_type, t.shape, t.byte_size, scale,
                    );
                }
                None => println!("  {name:<28} NOT FOUND"),
            }
        }
        // Tear down to silence validation layers
        model.destroy(&dev.device, &mut allocator);
    }

    println!();
    println!("== Layer 0 (mapped sample) ==");
    let mut keys: Vec<&str> = st.tensors.keys()
        .filter(|n| n.contains("layers.0."))
        .map(|s| s.as_str())
        .collect();
    keys.sort();
    for k in keys {
        let info = &st.tensors[k];
        let dtype_label = match info.dtype {
            vulkanforge::safetensors::TensorDtype::F8E4M3 => "F8_E4M3",
            vulkanforge::safetensors::TensorDtype::F8E5M2 => "F8_E5M2",
            vulkanforge::safetensors::TensorDtype::F16 => "F16",
            vulkanforge::safetensors::TensorDtype::BF16 => "BF16",
            vulkanforge::safetensors::TensorDtype::F32 => "F32",
        };
        let mapped = hf_to_vf_name(k).unwrap_or_else(|| "(skipped)".into());
        println!(
            "  {dtype_label:<8} {:?} {:>10} B  {} → {}",
            info.shape, info.byte_len(), k, mapped,
        );
    }
    Ok(())
}
