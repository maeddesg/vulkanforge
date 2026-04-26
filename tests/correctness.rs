//! Phase-2B correctness tests for every elementwise / utility shader
//! the decode pipeline depends on.
//!
//! Each test allocates host-visible input/output buffers, dispatches
//! the shader once, reads the output back, and compares against a
//! tiny CPU reference. Thresholds match the prompt §2.5.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::pipeline::{
    init_fastdiv_values, ComputeKernel, GenericBinaryPushConstants,
    GenericHeadPushConstants, GenericUnaryPushConstants, MmqPushConstants,
    Q8_1QuantizePushConstants, RopePushConstants, ScalarAttnPushConstants,
    SoftMaxPushConstants,
};
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::shaders::ShaderId;

// -----------------------------------------------------------------
// Test fixture — one Vulkan device + allocator + registry per test.

struct Fixture {
    dev: VulkanDevice,
    allocator: Option<Allocator>,
    registry: Option<PipelineRegistry>,
    cmd_ctx: Option<CommandContext>,
    pending_buffers: Vec<GpuBuffer>,
}

impl Fixture {
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
        let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("PipelineRegistry");
        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)
            .expect("CommandContext::new");
        Self {
            dev,
            allocator: Some(allocator),
            registry: Some(registry),
            cmd_ctx: Some(cmd_ctx),
            pending_buffers: Vec::new(),
        }
    }

    fn track(&mut self, buf: GpuBuffer) -> vk::Buffer {
        let h = buf.handle;
        self.pending_buffers.push(buf);
        h
    }

    fn registry(&self) -> &PipelineRegistry {
        self.registry.as_ref().unwrap()
    }
    fn cmd_ctx(&self) -> &CommandContext {
        self.cmd_ctx.as_ref().unwrap()
    }

    fn host_buffer_f32(&mut self, data: &[f32], name: &str) -> vk::Buffer {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        let device = self.dev.device.clone();
        let allocator = self.allocator.as_mut().unwrap();
        let mut buf = GpuBuffer::new(
            &device,
            allocator,
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            name,
        )
        .expect("alloc host buffer");
        buf.write_bytes(bytes).expect("write host buffer");
        self.track(buf)
    }

    fn host_buffer_u32(&mut self, data: &[u32], name: &str) -> vk::Buffer {
        let bytes = bytemuck::cast_slice::<u32, u8>(data);
        let device = self.dev.device.clone();
        let allocator = self.allocator.as_mut().unwrap();
        let mut buf = GpuBuffer::new(
            &device,
            allocator,
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            name,
        )
        .expect("alloc host buffer");
        buf.write_bytes(bytes).expect("write host buffer");
        self.track(buf)
    }

    fn output_buffer_f32(&mut self, count: usize, name: &str) -> vk::Buffer {
        let device = self.dev.device.clone();
        let allocator = self.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device,
            allocator,
            (count * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu,
            name,
        )
        .expect("alloc output buffer");
        self.track(buf)
    }

    fn read_output(&self, handle: vk::Buffer, count: usize) -> Vec<f32> {
        let buf = self
            .pending_buffers
            .iter()
            .find(|b| b.handle == handle)
            .expect("buffer not tracked");
        let bytes = buf.read_bytes().expect("read_bytes");
        bytemuck::cast_slice::<u8, f32>(&bytes[..count * 4]).to_vec()
    }

    fn dispatch(
        &mut self,
        shader_id: ShaderId,
        buffers: &[vk::Buffer],
        push_constants: &[u8],
        groups: (u32, u32, u32),
    ) {
        let kernel: &ComputeKernel = self.registry().get(shader_id);
        let (gx, gy, gz) = groups;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: buffers.len() as u32,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = unsafe {
            self.dev
                .device
                .create_descriptor_pool(&pool_info, None)
        }
        .expect("descriptor pool");
        let layouts = [kernel.descriptor_set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let set =
            unsafe { self.dev.device.allocate_descriptor_sets(&alloc) }.expect("desc set")[0];

        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&b| vk::DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { self.dev.device.update_descriptor_sets(&writes, &[]) };

        let last_buf = *buffers.last().expect("at least one buffer");
        let device = self.dev.device.clone();
        let queue = self.dev.compute_queue;
        let pipeline = kernel.pipeline;
        let layout = kernel.pipeline_layout;
        self.cmd_ctx().one_shot(&device, queue, |cmd| unsafe {
                device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                if !push_constants.is_empty() {
                    device.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constants,
                    );
                }
                device.cmd_dispatch(cmd, gx, gy, gz);
                let post = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(last_buf)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&post),
                    &[],
                );
            })
            .expect("one_shot");

        unsafe { self.dev.device.destroy_descriptor_pool(pool, None) };
    }

    /// Explicit teardown — call at end of every test.
    fn teardown(mut self) {
        let device = self.dev.device.clone();
        let mut allocator = self.allocator.take().unwrap();
        for b in self.pending_buffers.drain(..) {
            b.destroy(&device, &mut allocator);
        }
        if let Some(ctx) = self.cmd_ctx.take() {
            ctx.destroy(&device);
        }
        if let Some(reg) = self.registry.take() {
            reg.destroy(&device);
        }
        drop(allocator);
        // self.dev drops here, after all owned resources have been freed.
    }
}


// -----------------------------------------------------------------
// Tests

const N: usize = 4096;

#[test]
fn test_silu_vs_cpu() {
    let mut fix = Fixture::new();
    let input: Vec<f32> = (0..N).map(|i| -5.0 + 10.0 * (i as f32) / (N as f32 - 1.0)).collect();
    let cpu: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let in_h = fix.host_buffer_f32(&input, "silu_in");
    let out_h = fix.output_buffer_f32(N, "silu_out");

    let pc = GenericHeadPushConstants {
        kx: N as u32, ky: 1,
        param1: 0.0, param2: 0.0, param3: 0.0, param4: 0.0,
    };
    let local_x: u32 = 512;
    let dispatch_x = (N as u32 + local_x - 1) / local_x;

    fix.dispatch(ShaderId::Silu, &[in_h, out_h], bytemuck::bytes_of(&pc), (dispatch_x, 1, 1));
    let gpu = fix.read_output(out_h, N);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(max_abs < 1e-6, "silu max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_add_exact() {
    let mut fix = Fixture::new();
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 0.1 * i as f32).collect();
    let cpu: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let a_h = fix.host_buffer_f32(&a, "add_a");
    let b_h = fix.host_buffer_f32(&b, "add_b");
    let o_h = fix.output_buffer_f32(N, "add_d");

    let pc = binary_pc_1d(N as u32);
    // generic_binary_head's get_idx() pulls wy*512 + lx, so the
    // wg count goes on the Y axis (per llama.cpp's elements layout
    // for ADD/MUL: { 512, ceil(N/512), 1 } against wg_denoms
    // {512, 1, 1}).
    let dispatch_y = (N as u32 + 511) / 512;
    fix.dispatch(ShaderId::Add, &[a_h, b_h, o_h], bytemuck::bytes_of(&pc), (1, dispatch_y, 1));
    let gpu = fix.read_output(o_h, N);
    for i in 0..N {
        assert_eq!(gpu[i], cpu[i], "add[{i}]: gpu={} cpu={}", gpu[i], cpu[i]);
    }
    fix.teardown();
}

#[test]
fn test_mul_exact() {
    let mut fix = Fixture::new();
    let a: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 - i as f32 * 0.0005).collect();
    let cpu: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

    let a_h = fix.host_buffer_f32(&a, "mul_a");
    let b_h = fix.host_buffer_f32(&b, "mul_b");
    let o_h = fix.output_buffer_f32(N, "mul_d");

    let pc = binary_pc_1d(N as u32);
    let dispatch_y = (N as u32 + 511) / 512;
    fix.dispatch(ShaderId::Mul, &[a_h, b_h, o_h], bytemuck::bytes_of(&pc), (1, dispatch_y, 1));
    let gpu = fix.read_output(o_h, N);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(max_abs < 1e-6, "mul max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_copy_identity() {
    let mut fix = Fixture::new();
    let input: Vec<f32> = (0..N).map(|i| (i as f32) * 0.0123 - 17.0).collect();

    let in_h = fix.host_buffer_f32(&input, "cpy_in");
    let out_h = fix.output_buffer_f32(N, "cpy_out");

    let pc = unary_pc_1d(N as u32);
    let dispatch_y = (N as u32 + 511) / 512;
    fix.dispatch(ShaderId::Copy, &[in_h, out_h], bytemuck::bytes_of(&pc), (1, dispatch_y, 1));
    let gpu = fix.read_output(out_h, N);
    assert_eq!(gpu, input, "copy must be byte-exact");
    fix.teardown();
}

#[test]
fn test_rmsnorm_vs_cpu() {
    let mut fix = Fixture::new();
    let input: Vec<f32> = (0..N).map(|i| -1.0 + 2.0 * (i as f32) / (N as f32 - 1.0)).collect();
    let weight: Vec<f32> = vec![1.0; N];
    let eps: f32 = 1e-6;

    let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / N as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let cpu: Vec<f32> = input.iter().zip(&weight).map(|(x, w)| x * inv_rms * w).collect();

    let a_h = fix.host_buffer_f32(&input, "rms_in");
    let b_h = fix.host_buffer_f32(&weight, "rms_w");
    let o_h = fix.output_buffer_f32(N, "rms_out");

    // RMSNorm dispatches one workgroup per row. 1D vector → (1, 1, 1).
    let mut pc = binary_pc_1d(N as u32);
    pc.param1 = eps;
    fix.dispatch(ShaderId::RmsNorm, &[a_h, b_h, o_h], bytemuck::bytes_of(&pc), (1, 1, 1));
    let gpu = fix.read_output(o_h, N);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(max_abs < 1e-5, "rmsnorm max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_softmax_sums_to_one() {
    softmax_case(/*kx=*/ 128);
}

#[test]
fn test_softmax_vs_cpu_512() {
    softmax_case(/*kx=*/ 512);
}

fn softmax_case(kx: u32) {
    let mut fix = Fixture::new();
    // deterministic seed
    let mut rng_state = 0xCAFEBABEu32;
    let mut next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 8) as f32) / ((1u32 << 24) as f32) * 10.0 - 5.0
    };
    let input: Vec<f32> = (0..kx as usize).map(|_| next()).collect();

    let max_v = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let cpu: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let a_h = fix.host_buffer_f32(&input, "sm_in");
    // soft_max declares 4 SSBO bindings; only the first (in) and last
    // (out) are used in this configuration. We still need a real
    // buffer at every binding the layout declares — pass dummies for
    // the unused ones.
    let b_h = fix.host_buffer_f32(&[0.0; 4], "sm_b_dummy");
    let c_h = fix.host_buffer_f32(&[0.0; 4], "sm_c_dummy");
    let o_h = fix.output_buffer_f32(kx as usize, "sm_out");

    let pc = SoftMaxPushConstants {
        kx, ky: 0,
        ne00: kx, ne01: 1, ne02: 1, ne12: 1, ne13: 1,
        nb11: kx, nb12: kx, nb13: kx,
        scale: 1.0, max_bias: 0.0, m0: 1.0, m1: 1.0,
        n_head_log2: 0, nrows_x: 1, has_sinks: 0,
    };
    // soft_max binding order: 0=a, 1=b (mask), 2=c (sinks), 3=d (output).
    // Reflection sorts bindings by index, so the buffer slice must
    // mirror that order.
    fix.dispatch(ShaderId::SoftMax, &[a_h, b_h, c_h, o_h], bytemuck::bytes_of(&pc), (1, 1, 1));
    let gpu = fix.read_output(o_h, kx as usize);

    let gpu_sum: f32 = gpu.iter().sum();
    assert!((gpu_sum - 1.0).abs() < 1e-5, "softmax sum = {gpu_sum}");
    assert!(gpu.iter().all(|&v| v >= 0.0), "softmax has negative entries");
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(max_abs < 1e-5, "softmax(kx={kx}) max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_rope_neox_pos0_identity() {
    rope_neox_case(0, 1e-5);
}

#[test]
fn test_rope_neox_pos100_nontrivial() {
    rope_neox_case(100, 1e-4);
}

#[test]
fn test_rope_neox_pos4096_large() {
    // Larger threshold for pos=4096 because cos(theta) loses
    // precision when theta = pos·θ_scale^i at small i — RADV's
    // single-precision trig has ~1e-4 ULP at theta≈4096.
    rope_neox_case(4096, 5e-4);
}

const HEAD_DIM: u32 = 128;
const FREQ_BASE: f32 = 1_000_000.0;

fn rope_neox_case(pos: i32, abs_threshold: f32) {
    let mut fix = Fixture::new();

    // Single head, single token.
    let input: Vec<f32> = (0..HEAD_DIM as usize)
        .map(|i| 0.1 + 3.4 * (i as f32) / (HEAD_DIM as f32 - 1.0))
        .collect();

    // CPU NeoX-style RoPE: split-half rotation.
    let half = HEAD_DIM as usize / 2;
    let theta_scale = 1.0_f32 / FREQ_BASE.powf(2.0 / HEAD_DIM as f32);
    let mut cpu = vec![0.0f32; HEAD_DIM as usize];
    for i in 0..half {
        let freq = theta_scale.powi(i as i32);
        let theta = pos as f32 * freq;
        let c = theta.cos();
        let s = theta.sin();
        cpu[i] = input[i] * c - input[i + half] * s;
        cpu[i + half] = input[i] * s + input[i + half] * c;
    }

    let a_h = fix.host_buffer_f32(&input, "rope_in");
    let pos_h = fix.host_buffer_u32(&[pos as u32], "rope_pos");
    let ff_h = fix.host_buffer_f32(&[1.0; 4], "rope_ff");
    let o_h = fix.output_buffer_f32(HEAD_DIM as usize, "rope_out");
    let idx_h = fix.host_buffer_u32(&[0u32; 4], "rope_idx");

    // rope_norm.glsl / rope_neox.glsl share the same push-constant
    // block. n_dims is the number of paired dimensions (= head_dim
    // for full rotation). Set sections=[0,0,0,0] for non-Mrope.
    let pc = RopePushConstants {
        rope_mode: 2, // GGML_ROPE_TYPE_NEOX
        nrows: 1,
        n_dims: HEAD_DIM,
        freq_scale: 1.0,
        freq_base: FREQ_BASE,
        ext_factor: 0.0,
        attn_factor: 1.0,
        corr_dims: [0.0, 0.0],
        theta_scale,
        has_ff: 0,
        sections: [0, 0, 0, 0],
        is_imrope: 0,
        is_back: 0,
        set_rows_stride: 0,
        ne00: HEAD_DIM,
        ne01: 1,
        ne02: 1,
        nb01: HEAD_DIM,
        nb02: HEAD_DIM,
        nb03: HEAD_DIM,
        nb11: HEAD_DIM,
        nb12: HEAD_DIM,
        nb13: HEAD_DIM,
    };
    // local_size = (1, 256, 1). For n_dims=128, the loop covers
    // i0=0..n_dims/2 with step gl_GlobalInvocationID.y. (1, 1, 1)
    // workgroup → 256 threads, of which the first 64 cover the 64 pairs.
    fix.dispatch(
        ShaderId::RopeNeox,
        &[a_h, pos_h, ff_h, o_h, idx_h],
        bytemuck::bytes_of(&pc),
        (1, 1, 1),
    );
    let gpu = fix.read_output(o_h, HEAD_DIM as usize);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(
        max_abs < abs_threshold,
        "rope_neox(pos={pos}) max_abs_err = {max_abs:e} >= {abs_threshold:e}\n\
         GPU first 8: {:?}\nCPU first 8: {:?}",
        &gpu[..8],
        &cpu[..8]
    );
    fix.teardown();
}

/// Phase-2C diagnostic: same as `test_scalar_attn_qwen3_dims_seq1`
/// but bind K and V with a non-zero descriptor offset and an
/// explicit range (matches what `Forward` does to slice per-layer
/// K/V out of the multi-layer cache buffer).
#[test]
fn test_scalar_attn_qwen3_dims_with_binding_offset() {
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = vec![0.1f32; (n_heads * head_dim) as usize];
    // 2-layer-equivalent buffer; bind starting at layer 1's offset.
    let layer_size = (max_seq * n_kv_heads * head_dim) as usize;
    let mut k_full = vec![999.0f32; layer_size * 2]; // layer 0 = garbage
    let mut v_full = vec![999.0f32; layer_size * 2];
    for i in 0..layer_size {
        k_full[layer_size + i] = 0.1; // layer 1 K
        v_full[layer_size + i] = 1.0; // layer 1 V
    }
    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k_full, "k");
    let v_h = fix.host_buffer_f32(&v_full, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");

    // Manually craft the descriptor set: offset=layer 1, range=layer_size_bytes.
    let kernel: &ComputeKernel = fix.registry().get(ShaderId::ScalarAttn);
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 4,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { fix.dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { fix.dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];

    let layer_off_bytes = (layer_size * 4) as u64;
    let layer_range_bytes = (layer_size * 4) as u64;
    let infos = [
        vk::DescriptorBufferInfo { buffer: q_h, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: k_h, offset: layer_off_bytes, range: layer_range_bytes },
        vk::DescriptorBufferInfo { buffer: v_h, offset: layer_off_bytes, range: layer_range_bytes },
        vk::DescriptorBufferInfo { buffer: o_h, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes: [vk::WriteDescriptorSet; 4] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos[i..i + 1])
    });
    unsafe { fix.dev.device.update_descriptor_sets(&writes, &[]) };

    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len: 1,
        max_seq, scale,
    };
    let device = fix.dev.device.clone();
    let queue = fix.dev.compute_queue;
    let pipeline = kernel.pipeline;
    let layout = kernel.pipeline_layout;
    fix.cmd_ctx().one_shot(&device, queue, |cmd| unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
            layout, 0, &[set], &[]);
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE,
            0, bytemuck::bytes_of(&pc));
        device.cmd_dispatch(cmd, n_heads, 1, 1);
        // post-barrier
        let post = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(o_h)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[], std::slice::from_ref(&post), &[]);
    }).unwrap();
    unsafe { fix.dev.device.destroy_descriptor_pool(pool, None) };

    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    let max_abs_from_one = gpu.iter().map(|&x| (x - 1.0).abs()).fold(0.0f32, f32::max);
    assert!(!nan,
        "scalar_attn with binding offset produced NaN/Inf — first 8: {:?}",
        &gpu[..8]);
    assert!(max_abs_from_one < 1e-5,
        "scalar_attn with binding offset wrong — max_abs_from_one = {max_abs_from_one}");
    fix.teardown();
}

/// Phase-2C diagnostic: scalar_attn at full Qwen3 dimensions
/// (32 heads, 8 kv-heads, head_dim=128) — match the forward path
/// to expose dim-dependent bugs.
#[test]
fn test_scalar_attn_qwen3_dims_seq1() {
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = vec![0.1f32; (n_heads * head_dim) as usize];
    let kv_size = (max_seq * n_kv_heads * head_dim) as usize;
    let k = vec![0.1f32; kv_size];
    let v = vec![1.0f32; kv_size];

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");
    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len: 1,
        max_seq, scale,
    };
    fix.dispatch(ShaderId::ScalarAttn, &[q_h, k_h, v_h, o_h],
                 bytemuck::bytes_of(&pc), (n_heads, 1, 1));
    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    let max_abs_from_one = gpu.iter().map(|&x| (x - 1.0).abs()).fold(0.0f32, f32::max);
    assert!(!nan, "scalar_attn(qwen3 dims, seq=1) produced NaN/Inf\nfirst 8: {:?}", &gpu[..8]);
    assert!(max_abs_from_one < 1e-5, "max_abs_from_one = {max_abs_from_one}");
    fix.teardown();
}

/// Phase-2C diagnostic: scalar_attn with seq_len=2 to see if the
/// across-token loop produces NaN.
#[test]
fn test_scalar_attn_two_tokens() {
    let mut fix = Fixture::new();
    let n_heads: u32 = 4;
    let n_kv_heads: u32 = 2;
    let head_dim: u32 = 32;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = vec![0.1f32; (n_heads * head_dim) as usize];
    let kv_size = (max_seq * n_kv_heads * head_dim) as usize;
    let mut k = vec![0.1f32; kv_size];
    let mut v = vec![1.0f32; kv_size];
    // pos=1 has different values to make the test non-trivial.
    let pos1_off = (n_kv_heads * head_dim) as usize;
    for i in 0..(n_kv_heads * head_dim) as usize {
        k[pos1_off + i] = 0.2;
        v[pos1_off + i] = 2.0;
    }

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");
    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len: 2,
        max_seq, scale,
    };
    fix.dispatch(ShaderId::ScalarAttn, &[q_h, k_h, v_h, o_h],
                 bytemuck::bytes_of(&pc), (n_heads, 1, 1));
    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "scalar_attn(seq=2) produced NaN/Inf");
    fix.teardown();
}

/// Phase-2C diagnostic: scalar_attn with single-token KV (seq_len=1)
/// and known Q/K/V values. With Q=0.1, K=0.1, V=1.0:
///   score[0] = dot(Q, K) * scale = 128*0.01 / sqrt(128) ≈ 0.113
///   softmax(scores) = [1.0]   (single element)
///   output[d] = 1.0 * V[0, d] = 1.0 for all d.
#[test]
fn test_scalar_attn_single_token() {
    let mut fix = Fixture::new();
    let n_heads: u32 = 4;
    let n_kv_heads: u32 = 2;
    let head_dim: u32 = 32;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = vec![0.1f32; (n_heads * head_dim) as usize];
    // Pos-major K/V layout: [pos, kv_head, dim]
    let kv_size = (max_seq * n_kv_heads * head_dim) as usize;
    let k = vec![0.1f32; kv_size];
    let v = vec![1.0f32; kv_size];

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");

    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len: 1,
        max_seq, scale,
    };
    fix.dispatch(ShaderId::ScalarAttn, &[q_h, k_h, v_h, o_h],
                 bytemuck::bytes_of(&pc), (n_heads, 1, 1));
    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "scalar_attn produced NaN/Inf with bounded input");
    let max_abs = gpu
        .iter()
        .map(|&x| (x - 1.0).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-5,
            "scalar_attn(seq=1, K=0.1, V=1.0) max_abs from 1.0 = {max_abs:e}\n\
             gpu first 8: {:?}",
            &gpu[..8]);
    fix.teardown();
}

// -----------------------------------------------------------------
// Phase-3D — Q8_1 quantize roundtrip + GEMM-vs-GEMV parity tests.

/// CPU dequantise one `block_q8_1_x4` (144 bytes covering 128 floats).
/// Layout: `f16vec2 ds[4]` (16 B) followed by `int8 qs[128]` (the
/// shader writes them as `int32_t qs[32]` = same bytes).
/// Each of the 4 sub-blocks of 32 floats has its own f16 scale `d`
/// (the `.x` of `ds[i]`) and was quantised as `round(val / d)` with
/// `d = amax / 127`.
fn dequant_q8_1_x4_block(bytes: &[u8; 144]) -> [f32; 128] {
    let mut out = [0.0f32; 128];
    let mut ds = [0.0f32; 4];
    for i in 0..4 {
        // f16vec2 = 4 bytes; .x is the scale, .y is the f16 sum*d (unused for dequant).
        let off = i * 4;
        let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        ds[i] = half::f16::from_bits(bits).to_f32();
    }
    let qs = &bytes[16..16 + 128];
    for sub in 0..4 {
        let d = ds[sub];
        for j in 0..32 {
            let q = qs[sub * 32 + j] as i8;
            out[sub * 32 + j] = (q as f32) * d;
        }
    }
    out
}

#[test]
fn test_q8_1_quantize_roundtrip() {
    let mut fix = Fixture::new();
    // 256 elements (= 2 x4 blocks). Mix of magnitudes per sub-block
    // so each scale is non-trivial and the round-trip exercises real
    // dequant work.
    let n: u32 = 256;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let sub = (i / 32) as f32;
            let amp = 0.1 + 0.5 * sub;
            ((i as f32 - 128.0) * 0.05).sin() * amp
        })
        .collect();

    let a_h = fix.host_buffer_f32(&input, "q8_in");
    // Output buffer: ceil(n/128) x4 blocks × 144 bytes each.
    let blocks_x4 = ((n + 127) / 128) as usize;
    let out_bytes = (blocks_x4 * 144) as u64;
    let out_h = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device, allocator, out_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu,
            "q8_out",
        ).expect("alloc q8 out");
        fix.track(buf)
    };

    let pc = Q8_1QuantizePushConstants {
        ne: n,
        num_blocks: blocks_x4 as u32,
    };
    fix.dispatch(
        ShaderId::QuantizeQ8_1,
        &[a_h, out_h],
        bytemuck::bytes_of(&pc),
        // local_size_x = GROUP_SIZE = 32; one workgroup per x4 block.
        (blocks_x4 as u32, 1, 1),
    );

    // Pull the raw bytes back, dequantise, compare.
    let buf = fix
        .pending_buffers
        .iter()
        .find(|b| b.handle == out_h)
        .expect("out_h tracked");
    let raw = buf.read_bytes().expect("read q8 out");
    let mut recovered = vec![0.0f32; n as usize];
    for b in 0..blocks_x4 {
        let block: &[u8; 144] = (&raw[b * 144..b * 144 + 144]).try_into().unwrap();
        let block_out = dequant_q8_1_x4_block(block);
        recovered[b * 128..b * 128 + 128].copy_from_slice(&block_out);
    }

    // Q8_1 round-trip error is bounded by quantisation round-off
    // (d/2 = amax/254) plus f16 scale storage error. The empirical
    // worst case on our test data is <1% of the per-sub-block amax
    // — anything more would mean a layout/stride mismatch.
    let mut max_rel = 0.0f64;
    for sub in 0..(n as usize / 32) {
        let amax = input[sub * 32..(sub + 1) * 32]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let tol = (amax / 100.0).max(1e-6);
        for j in 0..32 {
            let i = sub * 32 + j;
            let err = (recovered[i] - input[i]).abs();
            assert!(
                err <= tol,
                "q8_1 roundtrip exceeded 1%-of-amax at i={i}: input={} recovered={} err={} tol={}",
                input[i], recovered[i], err, tol
            );
            let rel = (err / amax.max(1e-9)) as f64;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    assert!(max_rel < 0.01, "max relative roundtrip error {max_rel} > 1%");
    fix.teardown();
}

/// Phase 3D — GEMM(weights, activations_seq1) must match the existing
/// GEMV smoke output `[256.0, 512.0]` for `build_smoke_weights()`.
/// This is the parity gate: if the dispatch / push-constants / buffer
/// layout aren't right, output diverges (or NaNs) and the rest of
/// `prefill_batch` can't ship.
///
/// Setup mirrors `phase1_q4k_smoke_dispatch_bit_exact`:
///   M = 2 (output rows), K = QUANT_K = 256, N = 1 (single token)
///   Weights = `build_smoke_weights()` (Q4_K)
///   Activations = `[1.0; K]`, quantised through `quantize_q8_1`.
#[test]
fn test_gemm_q4k_vs_gemv_seq1_parity() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 2;
    let k: u32 = q4k::QUANT_K as u32; // 256
    let n: u32 = 1;

    // 1. Q4_K weights (same bytes the GEMV smoke uses).
    let weights_bytes = q4k::build_smoke_weights();
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights_bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "gemm_weights",
        ).expect("alloc weights");
        b.write_bytes(&weights_bytes).expect("write weights");
        fix.track(b)
    };

    // 2. Activations FP32 — single token of K=256 ones.
    let act: Vec<f32> = vec![1.0; k as usize];
    let act_buf = fix.host_buffer_f32(&act, "gemm_act_f32");

    // 3. Quantise activations → Q8_1_x4 (one 128-element x4 block per
    //    sub-row; K=256 → 2 x4 blocks).
    let q8_blocks_x4 = ((k * n + 127) / 128) as usize;
    let q8_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device, allocator, (q8_blocks_x4 * 144) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "gemm_q8",
        ).expect("alloc q8 act");
        fix.track(buf)
    };
    let q8_pc = Q8_1QuantizePushConstants {
        ne: k * n,
        num_blocks: q8_blocks_x4 as u32,
    };
    fix.dispatch(
        ShaderId::QuantizeQ8_1, &[act_buf, q8_buf],
        bytemuck::bytes_of(&q8_pc),
        (q8_blocks_x4 as u32, 1, 1),
    );

    // 4. Output buffer — M × N f32 = 2 floats.
    let out_buf = fix.output_buffer_f32((m * n) as usize, "gemm_out");

    // 5. Dispatch GEMM. Push constants per mul_mmq.comp lines 41-68
    //    (non-MoE variant). With BM=BN=64, M=2 < BM and N=1 < BN, so
    //    blocks_m = 1, blocks_n = 1, total dispatch = (1, 1, 1). The
    //    shader bounds-checks before writing rows past M.
    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,           // K elements per row of weights
        stride_b: k,            // K elements per row of activations (1 row of K elements)
        stride_d: n,            // N elements per row of D = 1
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,             // no split-K
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };
    fix.dispatch(
        ShaderId::MulMmqQ4K, &[w_buf, q8_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (1, 1, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    eprintln!("GEMM seq=1 output: {:?}", gpu);
    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "GEMM seq=1 produced NaN/Inf — first 8: {:?}", &gpu[..gpu.len().min(8)]);
    // GEMV smoke produces exactly [256.0, 512.0]. GEMM should match
    // within Q8_1 round-off (input is all 1.0 → quantises to exactly
    // 127 with d=1/127 → dequant 1.0 exactly → matrix product matches).
    // Q8_1 storage of activations + f16 storage of the per-block scale
    // introduces round-off bounded by amax/127 per term. For input
    // amax=1.0 the per-term error is ≈1/127, so the matrix-product
    // error of K=256 ones × constant weights stays within ≈0.5% of
    // the analytical answer. We assert <0.1 absolute which is well
    // above the observed ≈0.02 round-off.
    let g0_err = (gpu[0] - 256.0).abs();
    let g1_err = (gpu[1] - 512.0).abs();
    assert!(
        g0_err < 0.1 && g1_err < 0.1,
        "GEMM seq=1 ≠ GEMV: got [{}, {}], expected [256.0, 512.0] (errs {} {})",
        gpu[0], gpu[1], g0_err, g1_err
    );
    fix.teardown();
}

// -----------------------------------------------------------------
// Phase-3A tiled attention regression — seq_len=64 (one wavefront)
// and seq_len=200 (≈ Phase 2 baseline pos=200) compared against a
// CPU reference. The shader file `vk_shaders/scalar_attn.comp` was
// rewritten as a 64-thread tiled implementation; the four `test_scalar_attn_*`
// tests above already verify drop-in compatibility on the original
// Phase-2C fixtures.

/// CPU attention reference. Inputs in pos-major K/V layout, GQA aware.
fn cpu_attention_reference(
    q: &[f32],
    k_full: &[f32],
    v_full: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    scale: f32,
) -> Vec<f32> {
    let group_size = n_heads / n_kv_heads;
    let pos_stride = (n_kv_heads * head_dim) as usize;
    let head_dim_us = head_dim as usize;
    let mut out = vec![0.0f32; (n_heads * head_dim) as usize];
    for h in 0..n_heads {
        let kvh = h / group_size;
        let q_off = (h * head_dim) as usize;
        let head_off = (kvh * head_dim) as usize;

        let mut scores = vec![0.0f32; seq_len as usize];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let kt = (t as usize) * pos_stride + head_off;
            let mut s = 0.0f64;
            for d in 0..head_dim_us {
                s += (q[q_off + d] as f64) * (k_full[kt + d] as f64);
            }
            let s = s as f32 * scale;
            scores[t as usize] = s;
            if s > max_score {
                max_score = s;
            }
        }
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s as f64;
        }
        let inv = 1.0 / sum as f32;
        let o_off = (h * head_dim) as usize;
        for d in 0..head_dim_us {
            let mut acc = 0.0f64;
            for t in 0..seq_len {
                let vt = (t as usize) * pos_stride + head_off;
                acc += (scores[t as usize] as f64) * (v_full[vt + d] as f64);
            }
            out[o_off + d] = (acc * inv as f64) as f32;
        }
    }
    out
}

fn run_tiled_attn_seqlen(seq_len: u32, abs_threshold: f32) {
    let mut fix = Fixture::new();
    // Use Qwen3 dimensions exactly so we exercise the same code path
    // the forward pass takes.
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (max_seq as usize) * pos_stride;
    let mut q = vec![0.0f32; (n_heads * head_dim) as usize];
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];

    // Deterministic non-trivial inputs that exercise both head_dim
    // and pos dependencies. No two heads/positions are identical, so
    // a buggy wavefront layout would surface.
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i as f32) * 0.001).sin();
    }
    for t in 0..(seq_len as usize) {
        for kvh in 0..(n_kv_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = t * pos_stride + kvh * (head_dim as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + d as f32 * 0.001).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.7 + d as f32 * 0.0007).sin();
            }
        }
    }

    let cpu = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, scale);

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");
    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len, max_seq, scale,
    };
    fix.dispatch(
        ShaderId::ScalarAttn,
        &[q_h, k_h, v_h, o_h],
        bytemuck::bytes_of(&pc),
        (n_heads, 1, 1),
    );
    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "tiled scalar_attn(seq={seq_len}) produced NaN/Inf");
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(
        max_abs < abs_threshold,
        "tiled scalar_attn(seq={seq_len}) vs CPU max_abs_err = {max_abs:e} >= {abs_threshold:e}\n\
         GPU first 8: {:?}\nCPU first 8: {:?}",
        &gpu[..8],
        &cpu[..8],
    );
    fix.teardown();
}

#[test]
fn test_tiled_attn_seq64_vs_cpu() {
    // One full wavefront of work — every thread does exactly one t.
    // 1e-3 threshold accounts for the f32-vs-f64 rounding of the
    // softmax + accumulation chain (CPU reference uses f64 internally).
    run_tiled_attn_seqlen(64, 1e-3);
}

#[test]
fn test_tiled_attn_seq200_vs_cpu() {
    // ≈ Phase 2 baseline `pos=200`. seq_len=200 is 3.125 × WGSIZE,
    // so threads do uneven work (some 3, some 4 t-iterations) — this
    // is the regime that broke the Phase-2 scalar shader's perf and
    // is the reason the tiled rewrite exists.
    run_tiled_attn_seqlen(200, 1e-3);
}

// -----------------------------------------------------------------
// Phase-4B flash-attention parity tests.
// flash_attn.comp is a drop-in replacement for scalar_attn.comp with
// the same bindings + push constants but online-softmax internals
// (no scores[2048] LDS array, single pass instead of three). It must
// produce numerically equivalent output for every seq_len we use in
// production.

fn run_flash_attn_seqlen(seq_len: u32, abs_threshold: f32) {
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (max_seq as usize) * pos_stride;
    let mut q = vec![0.0f32; (n_heads * head_dim) as usize];
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];

    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i as f32) * 0.001).sin();
    }
    for t in 0..(seq_len as usize) {
        for kvh in 0..(n_kv_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = t * pos_stride + kvh * (head_dim as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + d as f32 * 0.001).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.7 + d as f32 * 0.0007).sin();
            }
        }
    }

    let cpu = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, scale);

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");
    let o_h = fix.output_buffer_f32((n_heads * head_dim) as usize, "o");
    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len, max_seq, scale,
    };
    fix.dispatch(
        ShaderId::FlashAttn,
        &[q_h, k_h, v_h, o_h],
        bytemuck::bytes_of(&pc),
        (n_heads, 1, 1),
    );
    let gpu = fix.read_output(o_h, (n_heads * head_dim) as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "flash_attn(seq={seq_len}) produced NaN/Inf — first 8: {:?}", &gpu[..8]);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(
        max_abs < abs_threshold,
        "flash_attn(seq={seq_len}) vs CPU max_abs_err = {max_abs:e} >= {abs_threshold:e}\n\
         GPU first 8: {:?}\n CPU first 8: {:?}",
        &gpu[..8], &cpu[..8],
    );
    fix.teardown();
}

#[test]
fn test_flash_attn_seq1_vs_cpu() {
    // seq_len=1 is the minimum (only the new token's KV is in cache).
    // softmax(1 score) = 1.0, output should be exactly V[0].
    // Exact same input as scalar_attn's test, easy correctness gate.
    run_flash_attn_seqlen(1, 1e-3);
}

#[test]
fn test_flash_attn_seq64_vs_cpu() {
    // One full TILE — exercises the tile-loop boundary exactly once.
    run_flash_attn_seqlen(64, 1e-3);
}

#[test]
fn test_flash_attn_seq200_vs_cpu() {
    // 3.125 × TILE — tail-tile bounds path exercised. Phase-2-baseline
    // pos=200 regime; the very situation that motivates the rewrite.
    run_flash_attn_seqlen(200, 1e-3);
}

#[test]
fn test_flash_attn_matches_scalar_attn_seq200() {
    // The strongest signal: same Q/K/V through scalar_attn (Phase-3A
    // tiled, three-pass) and through flash_attn (Phase-4B online
    // softmax). Both must produce numerically equivalent output —
    // softmax is mathematically order-invariant so the only difference
    // is f32 rounding through different operation orders.
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let seq_len: u32 = 200;

    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (max_seq as usize) * pos_stride;
    let mut q = vec![0.0f32; (n_heads * head_dim) as usize];
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i as f32) * 0.001).sin();
    }
    for t in 0..(seq_len as usize) {
        for kvh in 0..(n_kv_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = t * pos_stride + kvh * (head_dim as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + d as f32 * 0.001).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.7 + d as f32 * 0.0007).sin();
            }
        }
    }
    let q_a = fix.host_buffer_f32(&q, "q");
    let k_a = fix.host_buffer_f32(&k, "k");
    let v_a = fix.host_buffer_f32(&v, "v");
    let o_a = fix.output_buffer_f32((n_heads * head_dim) as usize, "o_scalar");
    let q_b = fix.host_buffer_f32(&q, "q2");
    let k_b = fix.host_buffer_f32(&k, "k2");
    let v_b = fix.host_buffer_f32(&v, "v2");
    let o_b = fix.output_buffer_f32((n_heads * head_dim) as usize, "o_flash");
    let pc = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim, seq_len, max_seq, scale,
    };
    fix.dispatch(
        ShaderId::ScalarAttn, &[q_a, k_a, v_a, o_a],
        bytemuck::bytes_of(&pc), (n_heads, 1, 1),
    );
    fix.dispatch(
        ShaderId::FlashAttn, &[q_b, k_b, v_b, o_b],
        bytemuck::bytes_of(&pc), (n_heads, 1, 1),
    );
    let scalar = fix.read_output(o_a, (n_heads * head_dim) as usize);
    let flash  = fix.read_output(o_b, (n_heads * head_dim) as usize);
    let max_abs = max_abs_err(&flash, &scalar);
    assert!(
        max_abs < 1e-3,
        "flash_attn vs scalar_attn at seq=200: max_abs_err = {max_abs:e} ≥ 1e-3\n\
         scalar first 8: {:?}\n flash  first 8: {:?}",
        &scalar[..8], &flash[..8],
    );
    fix.teardown();
}

// -----------------------------------------------------------------
// Helpers for push-constant boilerplate.

fn binary_pc_1d(n: u32) -> GenericBinaryPushConstants {
    GenericBinaryPushConstants {
        ne: n,
        ne00: n, ne01: 1, ne02: 1, ne03: 1,
        nb00: 1, nb01: n, nb02: n, nb03: n,
        ne10: n, ne11: 1, ne12: 1, ne13: 1,
        nb10: 1, nb11: n, nb12: n, nb13: n,
        ne20: n, ne21: 1, ne22: 1, ne23: 1,
        nb20: 1, nb21: n, nb22: n, nb23: n,
        misalign_offsets: 0,
        param1: 0.0, param2: 0.0, param3: 0,
    }
}

fn unary_pc_1d(n: u32) -> GenericUnaryPushConstants {
    let (ne0_012mp, ne0_012l) = init_fastdiv_values(n);
    let (ne0_01mp, ne0_01l) = init_fastdiv_values(n);
    let (ne0_0mp, ne0_0l) = init_fastdiv_values(n);
    let (ne1_012mp, ne1_012l) = init_fastdiv_values(n);
    let (ne1_01mp, ne1_01l) = init_fastdiv_values(n);
    let (ne1_0mp, ne1_0l) = init_fastdiv_values(n);
    GenericUnaryPushConstants {
        ne: n,
        ne00: n, ne01: 1, ne02: 1, ne03: 1,
        nb00: 1, nb01: n, nb02: n, nb03: n,
        ne10: n, ne11: 1, ne12: 1, ne13: 1,
        nb10: 1, nb11: n, nb12: n, nb13: n,
        misalign_offsets: 0,
        param1: 0.0, param2: 0.0,
        ne0_012mp, ne0_012l,
        ne0_01mp, ne0_01l,
        ne0_0mp, ne0_0l,
        ne1_012mp, ne1_012l,
        ne1_01mp, ne1_01l,
        ne1_0mp, ne1_0l,
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}
