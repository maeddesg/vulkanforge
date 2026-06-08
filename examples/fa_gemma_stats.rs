//! Sprint 10b Phase-1a — occupancy self-check helper.
//!
//! Builds ONLY the coopmat-FA Gemma pipeline(s) so `RADV_DEBUG=shaderstats`
//! (with `MESA_SHADER_CACHE_DISABLE=true`) emits exactly one shader-stats
//! block per pipeline — unambiguous VGPR / LDS / "Subgroups per SIMD"
//! (occupancy) without mapping block indices through the full registry.
//!
//!   RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true \
//!     cargo run --release --example fa_gemma_stats
//!
//! Gate: VGPR ≤ 192, LDS ≤ 26 KB, occ (hd256) ≥ 12, 0 spills. Sprint-7-class
//! (occ 2 / VGPR 256) → STOP.

use ash::vk;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::ComputeKernel;
use vulkanforge::backend::vulkan::shaders::{self, ShaderId};

fn main() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    for id in [ShaderId::FlashAttnCmGemmaHd256, ShaderId::FlashAttnCmGemmaHd256Fp8,
               ShaderId::FlashAttnCmGemmaRsHd256] {
        eprintln!("=== building {} ===", id.name());
        let words = shaders::spv_words(id.spv_bytes());
        let k = ComputeKernel::from_spv(&dev.device, &words, vk::PipelineCache::null())
            .expect("from_spv");
        // Keep it alive past the stats print, then tear down.
        unsafe {
            dev.device.destroy_pipeline(k.pipeline, None);
            dev.device.destroy_pipeline_layout(k.pipeline_layout, None);
            dev.device.destroy_descriptor_set_layout(k.descriptor_set_layout, None);
        }
    }
    eprintln!("=== done ===");
}
