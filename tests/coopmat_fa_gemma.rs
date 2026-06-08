//! Sprint 10b Phase-1a — numerical gate for the HEAD_DIM-parametric coopmat
//! flash-attention kernel (`flash_attn_cm_gemma.comp`, hd256).
//!
//! Dispatches the FP32-KV variant on a controlled causal self-attention case
//! and compares against a CPU reference that mirrors the kernel's arithmetic:
//! QK is accumulated from f16-rounded Q/K (the coopmat operands) in f32, then
//! a standard softmax and an f32 P·V. This isolates kernel correctness from
//! f16 input precision, so the tolerance is tight (rel-to-max < 2e-3, covering
//! only the online-softmax reassociation + coopmat f32-accumulate vs the
//! reference's straight f32-accumulate).

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use half::f16;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const HD: usize = 256;
const BR: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FaPush {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    m: u32,
    n_kv: u32,
    q_start: u32,
    scale: f32,
    kv_start: u32,
}

struct Harness {
    allocator: Allocator,
    registry: PipelineRegistry,
    cmd_ctx: CommandContext,
    dev: VulkanDevice,
}

impl Harness {
    fn new() -> Self {
        let dev = VulkanDevice::new().expect("VulkanDevice::new");
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: dev.instance.clone(),
            device: dev.device.clone(),
            physical_device: dev.physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .expect("Allocator::new");
        let (registry, _) =
            PipelineRegistry::new(&dev.device, None).expect("PipelineRegistry::new");
        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).unwrap();
        Self { allocator, registry, cmd_ctx, dev }
    }

    fn buf(&mut self, data: &[f32], loc: MemoryLocation, name: &str) -> GpuBuffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let mut b = GpuBuffer::new(&self.dev.device, &mut self.allocator,
            bytes.len().max(4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, loc, name).unwrap();
        if loc != MemoryLocation::GpuToCpu { b.write_bytes(bytes).unwrap(); }
        b
    }

    #[allow(clippy::too_many_arguments)]
    fn fa(&mut self, q: &[f32], k: &[f32], v: &[f32],
          m: usize, n_heads: usize, n_kv_heads: usize, n_kv: usize) -> Vec<f32> {
        let (pipeline, pipeline_layout, dsl) = {
            let k = self.registry.get(ShaderId::FlashAttnCmGemmaHd256);
            (k.pipeline, k.pipeline_layout, k.descriptor_set_layout)
        };
        let o_len = m * n_heads * HD;
        let q_buf = self.buf(q, MemoryLocation::CpuToGpu, "fa_q");
        let k_buf = self.buf(k, MemoryLocation::CpuToGpu, "fa_k");
        let v_buf = self.buf(v, MemoryLocation::CpuToGpu, "fa_v");
        let o_buf = self.buf(&vec![0f32; o_len], MemoryLocation::GpuToCpu, "fa_o");
        let dev = &self.dev;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 }];
        let pool = unsafe { dev.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None) }.unwrap();
        let layouts = [dsl];
        let set = unsafe { dev.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.unwrap()[0];
        let infos = [q_buf.handle, k_buf.handle, v_buf.handle, o_buf.handle]
            .map(|h| vk::DescriptorBufferInfo { buffer: h, offset: 0, range: vk::WHOLE_SIZE });
        let writes: [vk::WriteDescriptorSet; 4] = std::array::from_fn(|i| {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&infos[i..i + 1])
        });
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

        let pc = FaPush {
            n_heads: n_heads as u32, n_kv_heads: n_kv_heads as u32, head_dim: HD as u32,
            m: m as u32, n_kv: n_kv as u32, q_start: 0,
            scale: 1.0 / (HD as f32).sqrt(), kv_start: 0,
        };
        let n_qtiles = m.div_ceil(BR);
        self.cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                pipeline_layout, 0, &[set], &[]);
            dev.device.cmd_push_constants(cmd, pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
            dev.device.cmd_dispatch(cmd, n_heads as u32, n_qtiles as u32, 1);
            let post = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            dev.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), std::slice::from_ref(&post), &[], &[]);
        }).unwrap();

        let bytes = o_buf.read_bytes().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&bytes[..o_len * 4]).to_vec();
        unsafe { dev.device.destroy_descriptor_pool(pool, None) };
        out
    }
}

// CPU reference mirroring the kernel: QK from f16-rounded operands accumulated
// in f32, causal softmax, f32 P·V.
fn cpu_attn(q: &[f32], k: &[f32], v: &[f32],
            m: usize, n_heads: usize, n_kv_heads: usize, n_kv: usize) -> Vec<f32> {
    let scale = 1.0f32 / (HD as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let mut o = vec![0f32; m * n_heads * HD];
    for h in 0..n_heads {
        let kvh = h / group;
        for qi in 0..m {
            let q_pos = qi; // q_start = 0
            let qo = (qi * n_heads + h) * HD;
            let lim = (q_pos + 1).min(n_kv);
            let mut sc = vec![f32::NEG_INFINITY; lim];
            for (t, s) in sc.iter_mut().enumerate() {
                let ko = (t * n_kv_heads + kvh) * HD;
                let mut acc = 0f64;
                for d in 0..HD {
                    let qf = f16::from_f32(q[qo + d]).to_f32();
                    let kf = f16::from_f32(k[ko + d]).to_f32();
                    acc += (qf as f64) * (kf as f64);
                }
                *s = (acc as f32) * scale;
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for (t, &p) in sc.iter().enumerate() {
                let vo = (t * n_kv_heads + kvh) * HD;
                let w = p * inv;
                for d in 0..HD { o[qo + d] += w * v[vo + d]; }
            }
        }
    }
    o
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((self.0 >> 40) as u32) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn check(m: usize, n_heads: usize, n_kv_heads: usize) {
    let n_kv = m;
    let mut r = Lcg(0x1234_5678_9abc_def0 ^ (m as u64));
    let q: Vec<f32> = (0..m * n_heads * HD).map(|_| r.f()).collect();
    let k: Vec<f32> = (0..n_kv * n_kv_heads * HD).map(|_| r.f()).collect();
    let v: Vec<f32> = (0..n_kv * n_kv_heads * HD).map(|_| r.f()).collect();

    let mut h = Harness::new();
    let gpu = h.fa(&q, &k, &v, m, n_heads, n_kv_heads, n_kv);
    let cpu = cpu_attn(&q, &k, &v, m, n_heads, n_kv_heads, n_kv);

    let maxabs = cpu.iter().cloned().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    let mut max_rel = 0f32;
    let mut worst = 0usize;
    for i in 0..gpu.len() {
        let rel = (gpu[i] - cpu[i]).abs() / maxabs;
        if rel > max_rel { max_rel = rel; worst = i; }
    }
    println!("[fa_cm_gemma m={m} nh={n_heads} nkv={n_kv_heads}] max_rel={max_rel:.2e} \
              worst(i={worst} gpu={} cpu={})", gpu[worst], cpu[worst]);
    assert!(max_rel < 2e-3,
        "coopmat-FA hd256 numerics off vs reference: max_rel={max_rel:.3e} (m={m})");
}

#[test]
fn fa_cm_gemma_hd256_single_tile() { check(16, 4, 2); }   // one Q-tile, GQA 2

#[test]
fn fa_cm_gemma_hd256_multi_tile() { check(40, 4, 1); }    // 3 Q-tiles (partial last), MHA

#[test]
fn fa_cm_gemma_hd256_gqa8() { check(48, 8, 1); }          // GQA group 8 (sliding-like)
