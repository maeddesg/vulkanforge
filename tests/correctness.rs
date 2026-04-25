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
    GenericHeadPushConstants, GenericUnaryPushConstants, RopePushConstants,
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
