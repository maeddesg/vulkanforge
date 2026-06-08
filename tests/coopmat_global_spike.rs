//! Sprint 10b Phase-0b — KEYSTONE SPIKE.
//!
//! Validates the one capability the whole coopmat-FA build rests on and
//! that NO existing VF shader exercises: coopMatLoad a K-fragment DIRECTLY
//! from a *global* storage buffer in the real flash-attention per-head
//! layout (positions strided by `n_kv_heads*head_dim`), with no LDS
//! staging, producing a value-correct Q·Kᵀ.
//!
//! This is the occupancy-preserving load llama's `flash_attn_*_aligned_cm1`
//! relies on (shmem_staging=0 on RADV) and that Sprint 7 inverted (full
//! K-tiles in 41 KB LDS + VGPR-256 accumulators → occ 2). If this FAILS,
//! the recipe is not buildable as specified in VF's stack → Phase-0 STOP.
//!
//! Inputs are small integers (exactly f16-representable, |Σ| < 2^24) so the
//! f16×f16→f32 coopmat result is EXACT and the comparison is unambiguous —
//! the one-hot/exact-arithmetic trick from Sprint 9. K positions OUTSIDE
//! the tested head are poison-filled (1e3) so any stray read blows the
//! result far past tolerance.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use half::f16;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

const MAT: usize = 16; // coopmat tile (16×16), one Q-tile of 16 queries

struct Harness {
    // Field order = drop order: device-child objects drop BEFORE `dev`.
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

    /// Dispatch the spike: scores[16×16] = Q(16×D) · Kᵀ, K read direct from
    /// the global buffer in FA layout. Returns the 256 f32 scores.
    fn qk(&mut self, head_dim: usize, n_kv_heads: usize, kvh: usize,
          q: &[f32], k_fa: &[f32]) -> Vec<f32> {
        let k_stride = n_kv_heads * head_dim;
        let head_off = kvh * head_dim;
        let dev = &self.dev;
        let kernel = self.registry.get(ShaderId::SpikeCmGlobalQk);

        let q_bytes: Vec<u8> = q.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect();
        let k_bytes: Vec<u8> = k_fa.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect();
        let s_size = (MAT * MAT * 4) as u64;

        let mut q_buf = GpuBuffer::new(&dev.device, &mut self.allocator,
            q_bytes.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu, "spike_q").unwrap();
        let mut k_buf = GpuBuffer::new(&dev.device, &mut self.allocator,
            k_bytes.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu, "spike_k").unwrap();
        let s_buf = GpuBuffer::new(&dev.device, &mut self.allocator,
            s_size, vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu, "spike_s").unwrap();
        q_buf.write_bytes(&q_bytes).unwrap();
        k_buf.write_bytes(&k_bytes).unwrap();

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1).pool_sizes(&pool_sizes);
        let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
        let layouts = [kernel.descriptor_set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(&layouts);
        let set = unsafe { dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];
        let infos = [
            vk::DescriptorBufferInfo { buffer: q_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: k_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: s_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        ];
        let writes: [vk::WriteDescriptorSet; 3] = std::array::from_fn(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        });
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

        let pc: [u32; 3] = [head_dim as u32, k_stride as u32, head_off as u32];
        let pc_bytes: &[u8] = bytemuck::cast_slice(&pc);

        self.cmd_ctx.one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout, 0, &[set], &[]);
            dev.device.cmd_push_constants(cmd, kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            dev.device.cmd_dispatch(cmd, 1, 1, 1);
            let post = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            dev.device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), std::slice::from_ref(&post), &[], &[]);
        }).unwrap();

        let bytes = s_buf.read_bytes().unwrap();
        let out: Vec<f32> = (0..MAT * MAT)
            .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        unsafe { dev.device.destroy_descriptor_pool(pool, None) };
        out
    }
}

// Deterministic small-integer test values, exactly f16-representable.
fn qval(qi: usize, d: usize) -> f32 { (((qi * 3 + d) % 5) as i32 - 2) as f32 }      // -2..2
fn kval(t: usize, d: usize) -> f32 { (((t * 2 + d * 2 + 1) % 7) as i32 - 3) as f32 } // -3..3

fn run_case(head_dim: usize, n_kv_heads: usize, kvh: usize) {
    let mut h = Harness::new();
    let k_stride = n_kv_heads * head_dim;
    let head_off = kvh * head_dim;

    // Q: 16 queries × head_dim, contiguous.
    let mut q = vec![0f32; MAT * head_dim];
    for qi in 0..MAT {
        for d in 0..head_dim { q[qi * head_dim + d] = qval(qi, d); }
    }
    // K: FA layout [16 positions × k_stride], poison everywhere except this
    // head's columns so a stray read (wrong stride/head) is unmissable.
    let mut k_fa = vec![1.0e3f32; MAT * k_stride];
    for t in 0..MAT {
        for d in 0..head_dim { k_fa[t * k_stride + head_off + d] = kval(t, d); }
    }

    let scores = h.qk(head_dim, n_kv_heads, kvh, &q, &k_fa);

    // Exact reference: scores[qi][t] = Σ_d Q[qi][d]·K[t][d].
    let mut max_err = 0f32;
    let mut worst = (0usize, 0usize, 0f32, 0f32);
    for qi in 0..MAT {
        for t in 0..MAT {
            let mut acc = 0f64;
            for d in 0..head_dim { acc += (qval(qi, d) as f64) * (kval(t, d) as f64); }
            let r = acc as f32;
            let g = scores[qi * MAT + t];
            let e = (g - r).abs();
            if e > max_err { max_err = e; worst = (qi, t, g, r); }
        }
    }
    println!(
        "[spike hd{head_dim} nkv{n_kv_heads} kvh{kvh}] stride={k_stride} head_off={head_off} \
         max_abs_err={max_err}  worst(qi={},t={}) gpu={} ref={}",
        worst.0, worst.1, worst.2, worst.3
    );
    // Integer math ⇒ exact; allow <0.5 (rounds to the integer) as headroom.
    assert!(max_err < 0.5,
        "coopMatLoad-direct-global produced wrong Q·Kᵀ (max_abs_err={max_err}) — \
         KEYSTONE FAIL, recipe not buildable as specified");
}

#[test]
fn spike_hd256_sliding_direct_global() {
    // Gemma-4 sliding layer: head_dim 256, 8 KV-heads. Probe head 3.
    run_case(256, 8, 3);
}

#[test]
fn spike_hd512_full_direct_global() {
    // Gemma-4 full layer: head_dim 512, 2 KV-heads (GQA 16:2). Probe head 1.
    run_case(512, 2, 1);
}

#[test]
fn spike_hd16_minimal_direct_global() {
    // Minimal single-chunk case (one 16×16 coopMatMulAdd), stride > head_dim.
    run_case(16, 4, 2);
}
