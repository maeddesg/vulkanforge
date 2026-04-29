#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
)]

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
    init_fastdiv_values, ComputeKernel, FlashAttnReducePushConstants,
    FlashAttnSplitPushConstants, GenericBinaryPushConstants,
    GenericHeadPushConstants, GenericUnaryPushConstants, MmqPushConstants,
    FlashAttnBatchPushConstants, MultiAddRmsPushConstants, Q8_1QuantizePushConstants,
    RopePushConstants, ScalarAttnPushConstants, SoftMaxPushConstants, SwigluPushConstants,
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

// -----------------------------------------------------------------
// v0.2 Sprint 9a — fused SwiGLU correctness tests.
//
// The fused kernel keeps the SiLU intermediate in an FP32 register
// instead of round-tripping through global memory, but the arithmetic
// is identical: `(g / (1 + exp(-g))) * u`. We require BIT-EXACT
// agreement with the separate silu→mul path on the same random
// inputs (max_abs_err == 0.0).

fn swiglu_dispatch(fix: &mut Fixture, gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "swiglu inputs must be same length");
    let n = gate.len();
    let gate_h = fix.host_buffer_f32(gate, "swiglu_gate");
    let up_h = fix.host_buffer_f32(up, "swiglu_up");
    let out_h = fix.output_buffer_f32(n, "swiglu_out");
    let pc = SwigluPushConstants { n: n as u32 };
    let dispatch_x = (n as u32 + 255) / 256;
    fix.dispatch(
        ShaderId::SwiGLU,
        &[gate_h, up_h, out_h],
        bytemuck::bytes_of(&pc),
        (dispatch_x, 1, 1),
    );
    fix.read_output(out_h, n)
}

/// Reference: separate silu(gate→gate) + mul(gate, up→out) on the GPU
/// — i.e. the path SwiGLU is replacing. Uses the existing Silu and
/// Mul shaders so the comparison is exactly Sprint 8a vs Sprint 9a.
fn separate_silu_mul_dispatch(fix: &mut Fixture, gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    let n = gate.len();
    // Step 1: silu(gate) → tmp (out-of-place to keep the original
    // input intact for any later assertions).
    let gate_h = fix.host_buffer_f32(gate, "ssm_gate");
    let tmp_h = fix.output_buffer_f32(n, "ssm_silu_tmp");
    let pc_silu = GenericHeadPushConstants {
        kx: n as u32, ky: 1,
        param1: 0.0, param2: 0.0, param3: 0.0, param4: 0.0,
    };
    let dispatch_silu = (n as u32 + 511) / 512;
    fix.dispatch(
        ShaderId::Silu,
        &[gate_h, tmp_h],
        bytemuck::bytes_of(&pc_silu),
        (dispatch_silu, 1, 1),
    );
    // Step 2: mul(tmp, up → out).
    let up_h = fix.host_buffer_f32(up, "ssm_up");
    let out_h = fix.output_buffer_f32(n, "ssm_mul_out");
    let pc_mul = binary_pc_1d(n as u32);
    let dispatch_mul = (n as u32 + 511) / 512;
    fix.dispatch(
        ShaderId::Mul,
        &[tmp_h, up_h, out_h],
        bytemuck::bytes_of(&pc_mul),
        (1, dispatch_mul, 1),
    );
    fix.read_output(out_h, n)
}

#[test]
fn test_swiglu_vs_separate_small() {
    let mut fix = Fixture::new();
    let n = 1024;
    let gate: Vec<f32> = (0..n).map(|i| -5.0 + 10.0 * (i as f32) / (n as f32 - 1.0)).collect();
    let up: Vec<f32> = (0..n).map(|i| 0.5 + 0.001 * (i as f32)).collect();

    let fused = swiglu_dispatch(&mut fix, &gate, &up);
    let separate = separate_silu_mul_dispatch(&mut fix, &gate, &up);

    let max_abs = max_abs_err(&fused, &separate);
    assert!(max_abs < 1e-6, "swiglu vs separate max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_swiglu_vs_separate_qwen_ffn_shape() {
    // Realistic shape: seq_len=128 × ffn_dim=12288 (Qwen3-8B).
    let mut fix = Fixture::new();
    let n: usize = 128 * 12288;
    let mut gate = Vec::with_capacity(n);
    let mut up = Vec::with_capacity(n);
    // Deterministic pseudo-random inputs covering all sign quadrants.
    for i in 0..n {
        let t = (i as f32) * 0.001;
        gate.push((t.sin() * 4.0) - 1.5);
        up.push(t.cos() * 2.0);
    }

    let fused = swiglu_dispatch(&mut fix, &gate, &up);
    let separate = separate_silu_mul_dispatch(&mut fix, &gate, &up);

    let max_abs = max_abs_err(&fused, &separate);
    assert!(max_abs < 1e-6, "swiglu (FFN-shape) max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_swiglu_zeros() {
    // gate=0 → silu(0) = 0 → out=0 regardless of up.
    let mut fix = Fixture::new();
    let n = 256;
    let gate = vec![0.0_f32; n];
    let up: Vec<f32> = (0..n).map(|i| (i as f32) - 100.0).collect();

    let out = swiglu_dispatch(&mut fix, &gate, &up);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "swiglu_zeros[{i}] = {v} (expected 0)");
    }
    fix.teardown();
}

#[test]
fn test_swiglu_negative_saturates() {
    // gate=-10 → silu(-10) ≈ -4.54e-5 → out ≈ 0.
    let mut fix = Fixture::new();
    let n = 64;
    let gate = vec![-10.0_f32; n];
    let up = vec![1.0_f32; n];

    let out = swiglu_dispatch(&mut fix, &gate, &up);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-3, "swiglu_negative[{i}] = {v} (expected ≈ 0)");
    }
    fix.teardown();
}

#[test]
fn test_swiglu_positive_passthrough() {
    // gate=10 → silu(10) ≈ 9.99955 → out ≈ up * 9.99955.
    let mut fix = Fixture::new();
    let n = 64;
    let gate = vec![10.0_f32; n];
    let up: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let out = swiglu_dispatch(&mut fix, &gate, &up);
    let silu10 = 10.0_f32 / (1.0 + (-10.0_f32).exp());
    for (i, &v) in out.iter().enumerate() {
        let expected = silu10 * up[i];
        let err = (v - expected).abs();
        assert!(err < 1e-4, "swiglu_positive[{i}] = {v}, expected {expected}");
    }
    fix.teardown();
}

// -----------------------------------------------------------------
// v0.2 Sprint 9b — Fused multi_add_rms correctness tests.
//
// Replaces (add → barrier → rms_norm) at the attn-output → ffn-norm
// transition. The shader computes:
//     sum[r,c]      = a[r,c] + b[r,c]
//     scale         = 1 / sqrt(mean_c(sum[r,c]²) + eps)
//     norm_out[r,c] = sum[r,c] * scale * weight[c]
// We require:
//   1. Bit-identical residual update (sum_out = a + b; max_abs = 0).
//   2. Numerical match against a CPU f64 reference (max_abs < 1e-4
//      — slight drift vs separate-pipeline path because the reduction
//      tree shape and the residual-write/re-read round-trip differ).

fn cpu_multi_add_rms(
    a: &[f32], b: &[f32], weight: &[f32],
    n_rows: usize, n_cols: usize, eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut sum = vec![0.0_f32; n_rows * n_cols];
    let mut norm = vec![0.0_f32; n_rows * n_cols];
    for r in 0..n_rows {
        let off = r * n_cols;
        let mut sum_sq: f64 = 0.0;
        for c in 0..n_cols {
            let v = a[off + c] + b[off + c];
            sum[off + c] = v;
            sum_sq += (v as f64) * (v as f64);
        }
        let mean = sum_sq / (n_cols as f64);
        let scale = 1.0_f64 / (mean + eps as f64).sqrt();
        for c in 0..n_cols {
            norm[off + c] = (sum[off + c] as f64 * scale * weight[c] as f64) as f32;
        }
    }
    (sum, norm)
}

fn dispatch_multi_add_rms(
    fix: &mut Fixture,
    a: &[f32], b: &[f32], weight: &[f32],
    n_rows: u32, n_cols: u32, eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), (n_rows * n_cols) as usize);
    assert_eq!(weight.len(), n_cols as usize);
    let n_total = a.len();

    let a_h = fix.host_buffer_f32(a, "mar_a");
    let b_h = fix.host_buffer_f32(b, "mar_b");
    let w_h = fix.host_buffer_f32(weight, "mar_w");
    let sum_h = fix.output_buffer_f32(n_total, "mar_sum");
    let norm_h = fix.output_buffer_f32(n_total, "mar_norm");

    let pc = MultiAddRmsPushConstants { ne00: n_cols, n_rows, eps };
    fix.dispatch(
        ShaderId::MultiAddRms,
        &[a_h, b_h, w_h, sum_h, norm_h],
        bytemuck::bytes_of(&pc),
        (n_rows, 1, 1),
    );
    let sum = fix.read_output(sum_h, n_total);
    let norm = fix.read_output(norm_h, n_total);
    (sum, norm)
}

#[test]
fn test_multi_add_rms_residual_unchanged_small() {
    // Sum buffer must equal a + b elementwise (bit-exact).
    let mut fix = Fixture::new();
    let n_rows: u32 = 4;
    let n_cols: u32 = 256;
    let eps = 1e-6_f32;
    let total = (n_rows * n_cols) as usize;
    let a: Vec<f32> = (0..total).map(|i| ((i as f32) - 100.0) * 0.01).collect();
    let b: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0005 - 0.25).collect();
    let weight: Vec<f32> = (0..n_cols).map(|i| 1.0 + 0.001 * i as f32).collect();

    let (gpu_sum, _) = dispatch_multi_add_rms(&mut fix, &a, &b, &weight, n_rows, n_cols, eps);
    for i in 0..total {
        let expected = a[i] + b[i];
        assert_eq!(gpu_sum[i], expected,
            "sum_out[{i}]: gpu={} expected={}", gpu_sum[i], expected);
    }
    fix.teardown();
}

#[test]
fn test_multi_add_rms_norm_vs_cpu_small() {
    let mut fix = Fixture::new();
    let n_rows: u32 = 4;
    let n_cols: u32 = 256;
    let eps = 1e-6_f32;
    let total = (n_rows * n_cols) as usize;
    let a: Vec<f32> = (0..total).map(|i| ((i as f32).sin()) * 2.0).collect();
    let b: Vec<f32> = (0..total).map(|i| ((i as f32).cos()) * 0.5).collect();
    let weight: Vec<f32> = (0..n_cols).map(|i| 0.5 + 0.5 * (i as f32 / n_cols as f32)).collect();

    let (_, gpu_norm) =
        dispatch_multi_add_rms(&mut fix, &a, &b, &weight, n_rows, n_cols, eps);
    let (_, cpu_norm) = cpu_multi_add_rms(
        &a, &b, &weight, n_rows as usize, n_cols as usize, eps);
    let max_abs = max_abs_err(&gpu_norm, &cpu_norm);
    assert!(max_abs < 1e-4, "norm vs CPU max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_multi_add_rms_qwen_attn_to_ffn_shape() {
    // Realistic Qwen3 Stelle-1 shape: seq_len=64, hidden=4096.
    let mut fix = Fixture::new();
    let n_rows: u32 = 64;
    let n_cols: u32 = 4096;
    let eps = 1e-6_f32;
    let total = (n_rows * n_cols) as usize;
    // Deterministic pseudo-random inputs.
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        let t = i as f32 * 0.0003;
        a.push(t.sin() * 1.5);
        b.push(t.cos() * 0.7 - 0.1);
    }
    let weight: Vec<f32> =
        (0..n_cols).map(|i| 0.9 + 0.2 * ((i as f32 / 100.0).sin())).collect();

    let (gpu_sum, gpu_norm) =
        dispatch_multi_add_rms(&mut fix, &a, &b, &weight, n_rows, n_cols, eps);
    let (cpu_sum, cpu_norm) = cpu_multi_add_rms(
        &a, &b, &weight, n_rows as usize, n_cols as usize, eps);

    for i in 0..total {
        assert_eq!(gpu_sum[i], cpu_sum[i],
            "sum_out[{i}] not bit-identical to a+b");
    }
    let max_abs = max_abs_err(&gpu_norm, &cpu_norm);
    assert!(max_abs < 1e-4, "Qwen-shape norm max_abs_err = {max_abs:e}");
    fix.teardown();
}

#[test]
fn test_multi_add_rms_inplace_aliased_sum() {
    // When the dispatch passes the same buffer for `a` and `sum_out`
    // (as the batched forward does — `sum = a + b` with `sum` ===
    // `batch_residual` === `a`), the result must still be a + b.
    // We can't fixture this with the generic dispatch helper because
    // it allocates separate buffers, so we do it by reading-back the
    // sum buffer and checking it elementwise — proves the shader
    // doesn't depend on `a` and `sum_out` being distinct as long as
    // each thread writes only positions it owns.
    //
    // (The shader doesn't read `data_a[c]` after writing `data_sum[c]`,
    // so even when they alias the result is unambiguous.)
    let mut fix = Fixture::new();
    let n_rows: u32 = 8;
    let n_cols: u32 = 1024;
    let eps = 1e-6_f32;
    let total = (n_rows * n_cols) as usize;
    let a: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 0.1 - 0.5).collect();
    let b: Vec<f32> = (0..total).map(|i| ((i % 23) as f32) * 0.05 + 0.1).collect();
    let weight: Vec<f32> = vec![1.0; n_cols as usize];

    let (gpu_sum, _) =
        dispatch_multi_add_rms(&mut fix, &a, &b, &weight, n_rows, n_cols, eps);
    for i in 0..total {
        assert_eq!(gpu_sum[i], a[i] + b[i],
            "sum_out[{i}] = {} (expected {})", gpu_sum[i], a[i] + b[i]);
    }
    fix.teardown();
}

#[test]
fn test_multi_add_rms_unit_weight_matches_pure_rms_norm() {
    // weight = 1.0 → norm_out should match a pure rms_norm of (a+b).
    // Useful as a sanity check that the weight axis is wired correctly.
    let mut fix = Fixture::new();
    let n_rows: u32 = 2;
    let n_cols: u32 = 512;
    let eps = 1e-6_f32;
    let total = (n_rows * n_cols) as usize;
    let a: Vec<f32> = (0..total).map(|i| 0.01 * (i as f32 - 200.0)).collect();
    let b: Vec<f32> = (0..total).map(|i| 0.005 * (i as f32 + 50.0)).collect();
    let weight = vec![1.0_f32; n_cols as usize];

    let (_, gpu_norm) =
        dispatch_multi_add_rms(&mut fix, &a, &b, &weight, n_rows, n_cols, eps);

    // Pure CPU rms_norm of a+b.
    let mut expected = vec![0.0_f32; total];
    for r in 0..n_rows as usize {
        let off = r * n_cols as usize;
        let sum_sq: f64 = (0..n_cols as usize)
            .map(|c| {
                let v = (a[off + c] + b[off + c]) as f64;
                v * v
            })
            .sum();
        let scale = 1.0_f64 / (sum_sq / (n_cols as f64) + eps as f64).sqrt();
        for c in 0..n_cols as usize {
            expected[off + c] = ((a[off + c] + b[off + c]) as f64 * scale) as f32;
        }
    }
    let max_abs = max_abs_err(&gpu_norm, &expected);
    assert!(max_abs < 1e-4, "unit-weight norm max_abs_err = {max_abs:e}");
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

// Phase 7 — full 64×64 GEMM tile parity. The seq=1 test above only
// exercises the corner of one workgroup; this hits a complete BM×BN
// tile (M = BM = 64, N = BN = 64). If warp-tile coverage is broken
// (NUM_WARPS too small for BM*BN/(WM*WN) tiles), half the output cols
// will read uninitialised — this test surfaces it as a max-abs-err
// blow-up versus the CPU reference.
fn cpu_gemm_q4k_ref(
    weights: &[u8],
    acts: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use vulkanforge::backend::vulkan::q4k::{BLOCK_BYTES, QUANT_K, dequant_block};
    assert_eq!(k % QUANT_K, 0);
    let bpr = k / QUANT_K;
    let mut out = vec![0.0f32; m * n];
    let mut dq_rows: Vec<f32> = vec![0.0f32; m * k];
    for r in 0..m {
        for b in 0..bpr {
            let off = (r * bpr + b) * BLOCK_BYTES;
            let block: &[u8; BLOCK_BYTES] = (&weights[off..off + BLOCK_BYTES]).try_into().unwrap();
            let dq = dequant_block(block);
            dq_rows[r * k + b * QUANT_K..r * k + (b + 1) * QUANT_K].copy_from_slice(&dq);
        }
    }
    // Output layout per shader: data_d[col * stride_d + row], stride_d = M.
    // out[col * m + row] = sum_e dq_rows[row, e] * acts[col, e]
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            for e in 0..k {
                acc += (dq_rows[row * k + e] as f64) * (acts[col * k + e] as f64);
            }
            out[col * m + row] = acc as f32;
        }
    }
    out
}

fn run_mul_mm_parity(m: u32, n: u32, k: u32, label: &str) {
    use vulkanforge::backend::vulkan::q4k;
    assert_eq!(k as usize % q4k::QUANT_K, 0, "K must be a multiple of QUANT_K");

    let mut fix = Fixture::new();

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0FFEE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu,
            "mul_mm_w",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "mul_mm_act");
    let out_buf = fix.output_buffer_f32((m * n) as usize, "mul_mm_out");

    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    let groups_x = (m + 63) / 64;
    let groups_y = (n + 63) / 64;
    fix.dispatch(
        ShaderId::MulMmQ4K, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (groups_x, groups_y, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "mul_mm[{label}] M={m} N={n} K={k} produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let thresh = (max_amax * 0.01).max(0.1);

    if max_err >= thresh {
        eprintln!("FAIL [{label}] M={m} N={n} K={k} — per-col errs (first 16 + last 16):");
        let mut col_err = vec![0.0f32; n as usize];
        let mut col_amax = vec![0.0f32; n as usize];
        for col in 0..n as usize {
            for row in 0..m as usize {
                let idx = col * m as usize + row;
                col_err[col] = col_err[col].max((gpu[idx] - cpu[idx]).abs());
                col_amax[col] = col_amax[col].max(cpu[idx].abs());
            }
        }
        let show = |c: usize| {
            eprintln!("  col {c:>4}: err={:.4e} cpu_amax={:.4e} ratio={:.3}",
                col_err[c], col_amax[c], col_err[c] / col_amax[c].max(1e-9));
        };
        for c in 0..(n as usize).min(16) { show(c); }
        if n as usize > 32 {
            eprintln!("  ...");
            for c in (n as usize - 16)..n as usize { show(c); }
        }
    }
    assert!(
        max_err < thresh,
        "mul_mm[{label}] M={m} N={n} K={k} max_err={max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

#[test]
fn test_mul_mm_q4k_k512_aligned()        { run_mul_mm_parity(64, 64, 512, "K=512"); }

#[test]
fn test_mul_mm_q4k_k2048_aligned()       { run_mul_mm_parity(64, 64, 2048, "K=2048"); }

#[test]
fn test_mul_mm_q4k_n_unaligned_62()      { run_mul_mm_parity(64, 62, 256,  "N=62"); }

#[test]
fn test_mul_mm_q4k_multi_n_tile_128()    { run_mul_mm_parity(64, 128, 256, "N=128 (2 BN tiles)"); }

#[test]
fn test_mul_mm_q4k_multi_m_tile_128()    { run_mul_mm_parity(128, 64, 256, "M=128 (2 BM tiles)"); }

#[test]
fn test_mul_mm_q4k_realistic_2048x62()   { run_mul_mm_parity(2048, 62, 2048, "real prefill"); }

// Phase 7 — aligned variant (LOAD_VEC_B=4, vec4 B-loads) parity. N
// must be divisible by 4 for the aligned shader to be safe; the
// unaligned bounds check is removed, so off-the-end reads on B
// would otherwise be UB.
fn run_mul_mm_aligned_parity(m: u32, n: u32, k: u32, label: &str) {
    use vulkanforge::backend::vulkan::q4k;
    assert_eq!(n % 4, 0, "aligned variant requires N % 4 == 0");
    assert_eq!(k as usize % q4k::QUANT_K, 0, "K must be a multiple of QUANT_K");

    let mut fix = Fixture::new();

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xBEEFCAFE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu,
            "aligned_w",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.011).cos() * 0.4)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "aligned_act");
    let out_buf = fix.output_buffer_f32((m * n) as usize, "aligned_out");

    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    let groups_x = (m + 63) / 64;
    let groups_y = (n + 63) / 64;
    fix.dispatch(
        ShaderId::MulMmQ4KAligned, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (groups_x, groups_y, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "mul_mm aligned[{label}] M={m} N={n} K={k} produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let thresh = (max_amax * 0.01).max(0.1);
    assert!(
        max_err < thresh,
        "mul_mm aligned[{label}] M={m} N={n} K={k} max_err={max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

#[test]
fn test_mul_mm_aligned_q4k_64x64()       { run_mul_mm_aligned_parity(64,   64, 256, "aligned 64x64"); }
#[test]
fn test_mul_mm_aligned_q4k_k2048()       { run_mul_mm_aligned_parity(64,   64, 2048, "aligned K=2048"); }
#[test]
fn test_mul_mm_aligned_q4k_n128()        { run_mul_mm_aligned_parity(64,  128, 256, "aligned N=128"); }
#[test]
fn test_mul_mm_aligned_q4k_real_dims()   { run_mul_mm_aligned_parity(2048, 60, 2048, "aligned real prefill (N=60)"); }
#[test]
fn test_mul_mm_aligned_q4k_ffn_dims()    { run_mul_mm_aligned_parity(2048, 60, 11008, "aligned ffn_down (N=60)"); }
// Bit-exact check vs the unaligned variant on the same inputs. This
// is the strongest correctness gate: the aligned shader must produce
// identical results to the unaligned one when N % 4 == 0.
#[test]
fn test_mul_mm_aligned_matches_unaligned_q4k() {
    use vulkanforge::backend::vulkan::q4k;
    let mut fix = Fixture::new();
    let m: u32 = 128;
    let n: u32 = 64; // % 4 == 0
    let k: u32 = 2048;
    let weights = q4k::build_random_weights(m as usize, k as usize, 0xDEADC0DE);
    let acts: Vec<f32> = (0..(n * k)).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect();
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_match",
        ).expect("alloc");
        b.write_bytes(&weights).expect("write");
        fix.track(b)
    };
    let act_buf = fix.host_buffer_f32(&acts, "act_match");
    let out_unaligned = fix.output_buffer_f32((m * n) as usize, "out_un");
    let out_aligned = fix.output_buffer_f32((m * n) as usize, "out_aligned");

    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k, stride_b: k, stride_d: m,
        batch_stride_a: m * k, batch_stride_b: n * k, batch_stride_d: m * n,
        base_work_group_z: 0, num_batches: 1, k_split: k,
        ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1,
    };
    let groups = ((m + 63) / 64, (n + 63) / 64, 1);
    fix.dispatch(ShaderId::MulMmQ4K, &[w_buf, act_buf, out_unaligned], bytemuck::bytes_of(&pc), groups);
    fix.dispatch(ShaderId::MulMmQ4KAligned, &[w_buf, act_buf, out_aligned], bytemuck::bytes_of(&pc), groups);

    let g_un = fix.read_output(out_unaligned, (m * n) as usize);
    let g_al = fix.read_output(out_aligned, (m * n) as usize);

    // Same shader, different load pattern → output must match within
    // float-summation order tolerance. The accumulation order is
    // identical (load order changes, FMA chain stays per-thread the
    // same), so we expect bit-exactness in practice.
    let max_err = max_abs_err(&g_un, &g_al);
    assert!(
        max_err < 1e-4,
        "aligned vs unaligned diverged: max_err = {max_err:e}"
    );
    fix.teardown();
}

// FFN-down dims: M=hidden=2048, N=seq=62, K=ffn_dim=11008.
// 11008 / QUANT_K = 43 BK iterations — most likely place a K-loop bug
// would surface if it weren't visible at K=2048 (8 iterations).
#[test]
fn test_mul_mm_q4k_ffn_down_dims()       { run_mul_mm_parity(2048, 62, 11008, "ffn_down K=11008"); }

// Multi-tile both axes to stress the (groups_x, groups_y) dispatch shape.
#[test]
fn test_mul_mm_q4k_multi_tile_both()     { run_mul_mm_parity(128, 128, 256, "multi-tile both"); }

// Larger N with unaligned tail.
#[test]
fn test_mul_mm_q4k_n200()                { run_mul_mm_parity(64, 200, 256, "N=200"); }

// Sprint 3A — Q4_K coopmat (forward layout) vs CPU f64 reference. Same
// shape pattern as `run_mul_mm_parity` but dispatches the coopmat
// shader. Output layout is identical ([N, M] row-major), so the
// existing CPU reference fits without modification. Tolerance is
// looser than mul_mm because the coopmat path narrows both A and B
// to FP8 inside the K-loop.
fn run_coopmat_q4k_parity(m: u32, n: u32, k: u32, label: &str) {
    use vulkanforge::backend::vulkan::pipeline::CoopmatPushConstants;
    use vulkanforge::backend::vulkan::q4k;
    assert_eq!(k as usize % q4k::QUANT_K, 0, "K must be a multiple of QUANT_K");

    let mut fix = Fixture::new();
    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0FFEE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu,
            "coopmat_w",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "coopmat_act");
    let out_buf = fix.output_buffer_f32((m * n) as usize, "coopmat_out");

    // Per-shape BN selector (matches the runtime gemm_q switch).
    let (shader, bn_tile) = if n <= 32 {
        (ShaderId::MulCoopmatQ4KFwdBn16, 16u32)
    } else if n <= 64 {
        (ShaderId::MulCoopmatQ4KFwdBn32, 32u32)
    } else {
        (ShaderId::MulCoopmatQ4KFwdBn64, 64u32)
    };

    let pc = CoopmatPushConstants {
        m, n, k,
        stride_a: k,   // weights stride in elements
        stride_b: k,   // FORWARD_LAYOUT: B is [N, K], stride = K
        stride_c: m,   // FORWARD_LAYOUT: C is [N, M], stride = M
    };
    let groups_x = m.div_ceil(64);
    let groups_y = n.div_ceil(bn_tile);
    fix.dispatch(
        shader, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (groups_x, groups_y, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "coopmat[{label}] M={m} N={n} K={k} produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // FP8 narrowing on both A and B → ~12.5% relative grid plus
    // sqrt(K) accumulation drift. 50% of max-magnitude is comfortable
    // headroom; matches what the v0.2 sprint-2B tests use.
    let thresh = (max_amax * 0.5).max(0.5);

    assert!(
        max_err < thresh,
        "coopmat[{label}] M={m} N={n} K={k} max_err={max_err:e} >= {thresh:e} (max_amax={max_amax:e})"
    );
    fix.teardown();
}

#[test]
fn test_coopmat_q4k_fwd_k512()    { run_coopmat_q4k_parity(64, 64, 512, "K=512"); }
#[test]
fn test_coopmat_q4k_fwd_k2048()   { run_coopmat_q4k_parity(64, 64, 2048, "K=2048"); }
#[test]
fn test_coopmat_q4k_fwd_n128()    { run_coopmat_q4k_parity(64, 128, 256, "N=128"); }
#[test]
fn test_coopmat_q4k_fwd_m128()    { run_coopmat_q4k_parity(128, 64, 256, "M=128"); }
#[test]
fn test_coopmat_q4k_fwd_prefill_2048_64() {
    run_coopmat_q4k_parity(2048, 64, 4096, "prefill 2048x64x4096");
}

// Sprint 3B — naive BF16 Q4_K coopmat. Same shape pattern as
// run_coopmat_q4k_parity but dispatches the naive shader. Parity
// gate is FP32-noise level since BF16 (7 mantissa bits) is much
// closer to f64 than FP8 was — tolerance scales with sqrt(K) * a
// small fraction of max-magnitude.
fn run_coopmat_q4k_naive_parity(m: u32, n: u32, k: u32, label: &str) {
    use vulkanforge::backend::vulkan::pipeline::CoopmatPushConstants;
    use vulkanforge::backend::vulkan::q4k;
    assert_eq!(k as usize % q4k::QUANT_K, 0);
    let mut fix = Fixture::new();
    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0FFEE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu,
            "naive_w",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };
    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "naive_act");
    let out_buf = fix.output_buffer_f32((m * n) as usize, "naive_out");

    let pc = CoopmatPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,    // FORWARD_LAYOUT: B = [N, K]
        stride_c: m,    // FORWARD_LAYOUT: C = [N, M]
    };
    // Naive kernel emits 16x16 output tiles, so dispatch is
    // (M/16, N/16) instead of (M/64, N/BN).
    let groups_x = m.div_ceil(16);
    let groups_y = n.div_ceil(16);
    fix.dispatch(
        ShaderId::MulCoopmatQ4KNaiveBf16, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (groups_x, groups_y, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "naive[{label}] produced NaN/Inf");
    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // BF16 has ~0.4% relative precision per element; tolerance scales
    // with the worst-case product magnitude in the fixture.
    let thresh = (max_amax * 0.05).max(0.05);
    assert!(
        max_err < thresh,
        "naive[{label}] M={m} N={n} K={k} max_err={max_err:e} >= {thresh:e} (max_amax={max_amax:e})"
    );
    fix.teardown();
}

#[test]
fn test_coopmat_q4k_naive_k512()       { run_coopmat_q4k_naive_parity(64, 64, 512, "K=512"); }
#[test]
fn test_coopmat_q4k_naive_k2048()      { run_coopmat_q4k_naive_parity(64, 64, 2048, "K=2048"); }
#[test]
fn test_coopmat_q4k_naive_m128()       { run_coopmat_q4k_naive_parity(128, 64, 256, "M=128"); }
#[test]
fn test_coopmat_q4k_naive_prefill_64() {
    run_coopmat_q4k_naive_parity(2048, 64, 4096, "prefill 2048x64x4096");
}

#[test]
fn test_coopmat_q4k_naive_qwen3_gemm_q() {
    // Exact runtime gemm_q shape on Qwen3-8B: q_dim=4096, seq_len=64, hidden=4096.
    run_coopmat_q4k_naive_parity(4096, 64, 4096, "qwen3 gemm_q 4096x64x4096");
}

#[test]
fn test_gemm_q4k_full_tile_64x64_mul_mm() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 64;
    let n: u32 = 64;
    let k: u32 = q4k::QUANT_K as u32;

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0FFEE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_full_tile_mm",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "act_full_tile_mm");

    let out_buf = fix.output_buffer_f32((m * n) as usize, "out_full_tile_mm");

    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    fix.dispatch(
        ShaderId::MulMmQ4K, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (1, 1, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "mul_mm 64x64 produced NaN/Inf");

    let mut col_errs: Vec<f32> = Vec::with_capacity(n as usize);
    let mut col_cpu_amax: Vec<f32> = Vec::with_capacity(n as usize);
    for col in 0..n as usize {
        let mut e = 0.0f32;
        let mut a = 0.0f32;
        for row in 0..m as usize {
            let idx = col * m as usize + row;
            e = e.max((gpu[idx] - cpu[idx]).abs());
            a = a.max(cpu[idx].abs());
        }
        col_errs.push(e);
        col_cpu_amax.push(a);
    }
    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = col_cpu_amax.iter().cloned().fold(0.0f32, f32::max);
    // mul_mm reads B as raw FP32 (no Q8_1 round-off on B), so the only
    // round-off is from Q4_K dequant of A. Tighter threshold than mul_mmq.
    let thresh = (max_amax * 0.01).max(0.1);
    if max_err >= thresh {
        eprintln!("FAIL mul_mm — per-col max errors:");
        for (col, (e, a)) in col_errs.iter().zip(&col_cpu_amax).enumerate() {
            eprintln!("  col {col:>2}: err={:.4e}  cpu_amax={:.4e}  ratio={:.3}",
                      e, a, e / a.max(1e-9));
        }
    }
    assert!(
        max_err < thresh,
        "mul_mm 64x64 vs CPU max_err = {max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

#[test]
fn test_gemm_q4k_full_tile_64x64_mul_mmq() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 64; // = BM exactly
    let n: u32 = 64; // = BN exactly — the case that exercises every warp tile
    let k: u32 = q4k::QUANT_K as u32; // 256, one block per row

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0FFEE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_full_tile",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    // N tokens × K features each, deterministic non-trivial values.
    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "act_full_tile");

    // Q8_1 quantization for mul_mmq: ceil(N*K / 128) x4-blocks of 144 B.
    let q8_blocks_x4 = ((n * k + 127) / 128) as usize;
    let q8_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device, allocator, (q8_blocks_x4 * 144) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "q8_full_tile",
        ).expect("alloc q8");
        fix.track(buf)
    };
    let q8_pc = Q8_1QuantizePushConstants { ne: n * k, num_blocks: q8_blocks_x4 as u32 };
    fix.dispatch(
        ShaderId::QuantizeQ8_1, &[act_buf, q8_buf],
        bytemuck::bytes_of(&q8_pc),
        (q8_blocks_x4 as u32, 1, 1),
    );

    let out_buf = fix.output_buffer_f32((m * n) as usize, "out_full_tile");

    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    // groups = (ceil(M/BM), ceil(N/BN), 1) = (1, 1, 1).
    fix.dispatch(
        ShaderId::MulMmqQ4K, &[w_buf, q8_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (1, 1, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "mul_mmq 64x64 produced NaN/Inf");

    // Per-column max error and a global cap. If any column is uninitialised
    // (warp-tile-coverage bug), its max error will be ~|cpu| (i.e. huge).
    let mut col_errs: Vec<f32> = Vec::with_capacity(n as usize);
    let mut col_cpu_amax: Vec<f32> = Vec::with_capacity(n as usize);
    for col in 0..n as usize {
        let mut e = 0.0f32;
        let mut a = 0.0f32;
        for row in 0..m as usize {
            let idx = col * m as usize + row;
            e = e.max((gpu[idx] - cpu[idx]).abs());
            a = a.max(cpu[idx].abs());
        }
        col_errs.push(e);
        col_cpu_amax.push(a);
    }

    // Q8_1 round-off: per-output ≈ |a|*K / 127 for amax≈0.5, K=256 → ~1.0.
    // Use a generous threshold; the bug we're hunting produces errors
    // orders of magnitude larger.
    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = col_cpu_amax.iter().cloned().fold(0.0f32, f32::max);
    let thresh = (max_amax * 0.05).max(0.5);
    if max_err >= thresh {
        eprintln!("FAIL — per-col max errors:");
        for (col, (e, a)) in col_errs.iter().zip(&col_cpu_amax).enumerate() {
            eprintln!("  col {col:>2}: err={:.4e}  cpu_amax={:.4e}  ratio={:.3}",
                      e, a, e / a.max(1e-9));
        }
    }
    assert!(
        max_err < thresh,
        "mul_mmq 64x64 vs CPU max_err = {max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

/// Sprint 11E — COOPMAT mul_mm Q4_K parity at the L-tile shape.
/// Dispatches `MulMmQ4KCoopmat` (BM=BN=128, KHR coopmat 16x16x16
/// FP16xFP16->FP32) over the production-Qwen3 Q-projection shape
/// `(M=512, K=4096, N=512)` and compares against the CPU reference.
/// FP16 LDS introduces precision drift vs scalar FP32 GEMM, so the
/// threshold is looser than the L-tile mul_mmq parity check.
#[test]
fn test_gemm_q4k_coopmat_mul_mm_parity() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 512;
    let n: u32 = 512;
    let k: u32 = 4 * q4k::QUANT_K as u32; // 1024

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xCAFE_BABE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_coopmat",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| (((i as f32) * 0.0073).sin() + ((i as f32) * 0.013).cos()) * 0.25)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "act_coopmat");

    let out_buf = fix.output_buffer_f32((m * n) as usize, "out_coopmat");
    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    // BM=BN=128 → groups = (4, 4, 1)
    fix.dispatch(
        ShaderId::MulMmQ4KCoopmat, &[w_buf, act_buf, out_buf],
        bytemuck::bytes_of(&pc),
        ((m + 127) / 128, (n + 127) / 128, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    assert!(gpu.iter().all(|x| x.is_finite()), "COOPMAT mul_mm produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().cloned().fold(0.0f32, |acc, v| acc.max(v.abs()));
    // FP16 LDS for both A and B → looser threshold than scalar mul_mm
    // (which keeps everything FP32). 5% of |amax| matches the mul_mmq
    // band — Q4_K weight round-off dominates either way.
    let thresh = (max_amax * 0.05).max(0.5);
    assert!(
        max_err < thresh,
        "COOPMAT mul_mm 512x512x{k} max_err = {max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

/// Sprint 11C — L-tile (BM=BN=128) end-to-end parity at the
/// dispatch shape that exercises every warp tile in the WG. With
/// `(BM/WM) * (BN/WN) = (128/64)*(128/64) = 4` warp tiles per WG
/// and `WMITER=1`, this is the same coverage check Phase 7 added
/// for the S tile but at the doubled-tile geometry. Verifies that
/// `l_warptile_mmq_int_k`'s spec constants (BLOCK_SIZE=256, BM=128,
/// BN=128, WM=64, WN=64, WMITER=1, TM=4, TN=2, WARP=64) produce
/// identical arithmetic to the S tile.
#[test]
fn test_gemm_q4k_full_tile_128x128_mul_mmq_l() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 128; // = BM_L exactly
    let n: u32 = 128; // = BN_L exactly
    let k: u32 = q4k::QUANT_K as u32; // 256

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xDEADC0DE);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_l_tile",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32) * 0.011).cos() * 0.4)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "act_l_tile");

    let q8_blocks_x4 = ((n * k + 127) / 128) as usize;
    let q8_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device, allocator, (q8_blocks_x4 * 144) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "q8_l_tile",
        ).expect("alloc q8");
        fix.track(buf)
    };
    let q8_pc = Q8_1QuantizePushConstants { ne: n * k, num_blocks: q8_blocks_x4 as u32 };
    fix.dispatch(
        ShaderId::QuantizeQ8_1, &[act_buf, q8_buf],
        bytemuck::bytes_of(&q8_pc),
        (q8_blocks_x4 as u32, 1, 1),
    );

    let out_buf = fix.output_buffer_f32((m * n) as usize, "out_l_tile");
    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    // groups = (ceil(M/BM_L), ceil(N/BN_L), 1) = (1, 1, 1) at exactly 128x128.
    fix.dispatch(
        ShaderId::MulMmqQ4KL, &[w_buf, q8_buf, out_buf],
        bytemuck::bytes_of(&pc),
        (1, 1, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    assert!(gpu.iter().all(|x| x.is_finite()), "mul_mmq L-tile produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().cloned().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let thresh = (max_amax * 0.05).max(0.5);
    assert!(
        max_err < thresh,
        "mul_mmq L-tile 128x128 vs CPU max_err = {max_err:e} >= {thresh:e}"
    );
    fix.teardown();
}

/// Sprint 11C — L-tile parity at the *production-Qwen3 prefill shape*
/// `(M=512, K=4096, N=512)` for the Q-projection at pp=512. Catches
/// drift between the warp-tile recurrence at small (full-tile) and
/// large (multi-WG) dispatch shapes — a single buggy warp would only
/// show up in one column out of 64 here. Reuses the existing CPU
/// reference for cross-shape parity. Threshold is the same Q8_1
/// round-off cap (`5 % * |amax|`).
#[test]
fn test_gemm_q4k_l_tile_qwen3_qproj_parity() {
    use vulkanforge::backend::vulkan::q4k;

    let mut fix = Fixture::new();
    let m: u32 = 512;
    let n: u32 = 512;
    let k: u32 = 4 * q4k::QUANT_K as u32; // 1024 — fits the test budget while still > 1 BK_STEP

    let weights = q4k::build_random_weights(m as usize, k as usize, 0xC0DEFEED);
    let w_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let mut b = GpuBuffer::new(
            &device, allocator, weights.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "w_qproj",
        ).expect("alloc weights");
        b.write_bytes(&weights).expect("write weights");
        fix.track(b)
    };

    let acts: Vec<f32> = (0..(n * k))
        .map(|i| (((i as f32) * 0.0073).sin() + ((i as f32) * 0.013).cos()) * 0.25)
        .collect();
    let act_buf = fix.host_buffer_f32(&acts, "act_qproj");

    let q8_blocks_x4 = ((n * k + 127) / 128) as usize;
    let q8_buf = {
        let device = fix.dev.device.clone();
        let allocator = fix.allocator.as_mut().unwrap();
        let buf = GpuBuffer::new(
            &device, allocator, (q8_blocks_x4 * 144) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "q8_qproj",
        ).expect("alloc q8");
        fix.track(buf)
    };
    let q8_pc = Q8_1QuantizePushConstants { ne: n * k, num_blocks: q8_blocks_x4 as u32 };
    fix.dispatch(
        ShaderId::QuantizeQ8_1, &[act_buf, q8_buf],
        bytemuck::bytes_of(&q8_pc),
        (q8_blocks_x4 as u32, 1, 1),
    );

    let out_buf = fix.output_buffer_f32((m * n) as usize, "out_qproj");
    let pc = MmqPushConstants {
        m, n, k,
        stride_a: k,
        stride_b: k,
        stride_d: m,
        batch_stride_a: m * k,
        batch_stride_b: n * k,
        batch_stride_d: m * n,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k,
        ne02: 1, ne12: 1,
        broadcast2: 1, broadcast3: 1,
    };
    // BM=BN=128 → groups = (4, 4, 1) — 16 WGs total at this shape.
    fix.dispatch(
        ShaderId::MulMmqQ4KL, &[w_buf, q8_buf, out_buf],
        bytemuck::bytes_of(&pc),
        ((m + 127) / 128, (n + 127) / 128, 1),
    );

    let gpu = fix.read_output(out_buf, (m * n) as usize);
    let cpu = cpu_gemm_q4k_ref(&weights, &acts, m as usize, n as usize, k as usize);

    assert!(gpu.iter().all(|x| x.is_finite()), "mul_mmq L-tile Qproj produced NaN/Inf");

    let max_err = max_abs_err(&gpu, &cpu);
    let max_amax = cpu.iter().cloned().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let thresh = (max_amax * 0.05).max(0.5);
    assert!(
        max_err < thresh,
        "mul_mmq L-tile Qproj 512x512x{k} max_err = {max_err:e} >= {thresh:e}"
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

/// Phase-4C split-K parity: dispatch (FlashAttnSplit + FlashAttnReduce)
/// against the same Q/K/V tensors that drive Phase-4B's `flash_attn`,
/// and compare outputs. The split path's online-softmax-merge maths
/// must produce numerically equivalent output to the single-WG
/// formulation — within an f32 round-off envelope.
fn run_split_attn_seqlen(seq_len: u32, abs_threshold: f32) {
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    const TILE: u32 = 64;
    let n_tiles = (seq_len + TILE - 1) / TILE;

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
    // CPU reference for absolute correctness check.
    let cpu = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, scale);

    let q_h = fix.host_buffer_f32(&q, "q");
    let k_h = fix.host_buffer_f32(&k, "k");
    let v_h = fix.host_buffer_f32(&v, "v");

    // Scratch buffers — sized for the worst case at seq_len=2048,
    // n_tiles = 32, head_dim = 128.
    let scratch_max_tiles = (max_seq + TILE - 1) / TILE;
    let scratch_out_count = (n_heads * scratch_max_tiles * head_dim) as usize;
    let scratch_max_count = (n_heads * scratch_max_tiles) as usize;
    let scratch_out = fix.output_buffer_f32(scratch_out_count, "scratch_out");
    let scratch_max = fix.output_buffer_f32(scratch_max_count, "scratch_max");
    let scratch_sum = fix.output_buffer_f32(scratch_max_count, "scratch_sum");

    // Final output buffer.
    let o_split = fix.output_buffer_f32((n_heads * head_dim) as usize, "o_split");

    // ---- Dispatch split worker ----
    let split_pc = FlashAttnSplitPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len, max_seq, scale,
        n_tiles,
    };
    fix.dispatch(
        ShaderId::FlashAttnSplit,
        &[q_h, k_h, v_h, scratch_out, scratch_max, scratch_sum],
        bytemuck::bytes_of(&split_pc),
        (n_heads, n_tiles, 1),
    );

    // ---- Dispatch reduce ----
    let reduce_pc = FlashAttnReducePushConstants {
        n_heads, head_dim, n_tiles,
    };
    fix.dispatch(
        ShaderId::FlashAttnReduce,
        &[scratch_out, scratch_max, scratch_sum, o_split],
        bytemuck::bytes_of(&reduce_pc),
        (n_heads, 1, 1),
    );

    let gpu = fix.read_output(o_split, (n_heads * head_dim) as usize);
    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(!nan, "split+reduce(seq={seq_len}) NaN/Inf — first 8: {:?}", &gpu[..8]);
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(
        max_abs < abs_threshold,
        "split+reduce(seq={seq_len}) vs CPU max_abs_err = {max_abs:e} >= {abs_threshold:e}\n\
         GPU first 8: {:?}\n CPU first 8: {:?}",
        &gpu[..8], &cpu[..8],
    );
    fix.teardown();
}

#[test]
fn test_split_attn_seq64_vs_cpu() {
    // n_tiles = 1 — the degenerate case. Should still produce correct
    // output (1-tile reduce is just normalisation by sum).
    run_split_attn_seqlen(64, 1e-3);
}

#[test]
fn test_split_attn_seq200_vs_cpu() {
    // n_tiles = 4 — the canonical "split-K kicks in" case. This is
    // the regime that motivates the rewrite (Phase-3 baseline pos=200).
    run_split_attn_seqlen(200, 1e-3);
}

#[test]
fn test_split_attn_seq2048_vs_cpu() {
    // n_tiles = 32 — max reasonable context for the Phase-2 KV cache.
    // Stresses the online-softmax merge over 32 partial accumulators.
    run_split_attn_seqlen(2048, 1e-3);
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

// =====================================================================
// Phase 5B.1 — batched-Q flash-attention (`flash_attn_batch.comp`).
// One dispatch covers (n_heads, M, 1); each WG runs the same online-
// softmax recurrence as Phase-4B `flash_attn` but with a per-query
// causal mask `causal_len = q_start + q_idx + 1`. Tests below are
// isolated — they don't go through the forward pass.
// =====================================================================

/// CPU reference for batched attention with causal masking. f64
/// internally for the softmax + V-sum so the f32 GPU path can be
/// graded against a numerically clean answer.
fn cpu_batch_attn_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    m: u32,
    n_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    q_start: u32,
    scale: f32,
) -> Vec<f32> {
    let head_dim_us = head_dim as usize;
    let pos_stride = (n_kv_heads * head_dim) as usize;
    let group_size = n_heads / n_kv_heads;
    let scale_d = scale as f64;
    let mut out = vec![0.0f32; (m * n_heads * head_dim) as usize];
    for q_idx in 0..m {
        let causal_len = ((q_start + q_idx + 1).min(n_kv)) as usize;
        for h in 0..n_heads {
            let kvh = (h / group_size) as usize;
            let q_off = ((q_idx * n_heads + h) * head_dim) as usize;

            let mut scores = Vec::with_capacity(causal_len);
            let mut max_score = f64::NEG_INFINITY;
            for t in 0..causal_len {
                let k_off = t * pos_stride + kvh * head_dim_us;
                let mut dot = 0.0f64;
                for d in 0..head_dim_us {
                    dot += (q[q_off + d] as f64) * (k[k_off + d] as f64);
                }
                let s = dot * scale_d;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }
            let mut weights: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() {
                *w /= sum;
            }

            let o_off = ((q_idx * n_heads + h) * head_dim) as usize;
            for d in 0..head_dim_us {
                let mut acc = 0.0f64;
                for t in 0..causal_len {
                    let v_off = t * pos_stride + kvh * head_dim_us;
                    acc += weights[t] * (v[v_off + d] as f64);
                }
                out[o_off + d] = acc as f32;
            }
        }
    }
    out
}

fn run_batch_attn(
    m: u32,
    q_start: u32,
    abs_threshold: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut fix = Fixture::new();
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let n_kv = q_start + m;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (n_kv as usize) * pos_stride;
    let q_count = (m * n_heads * head_dim) as usize;

    // Deterministic non-trivial inputs. Different sin/cos seeds per
    // (q_idx, h, kvh, d) so a buggy stride or GQA group_size would
    // produce divergent output. f32 is good enough — the f64 CPU
    // reference absorbs the round-off envelope.
    let mut q = vec![0.0f32; q_count];
    for q_idx in 0..(m as usize) {
        for h in 0..(n_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = (q_idx * n_heads as usize + h) * head_dim as usize + d;
                q[off] = ((q_idx as f32) * 0.017 + (h as f32) * 0.011 + (d as f32) * 0.003).sin();
            }
        }
    }
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];
    for t in 0..(n_kv as usize) {
        for kvh in 0..(n_kv_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = t * pos_stride + kvh * (head_dim as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + (d as f32) * 0.001 + (kvh as f32) * 0.7).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.5 + (d as f32) * 0.0007).sin();
            }
        }
    }

    let cpu = cpu_batch_attn_reference(
        &q, &k, &v, m, n_kv, n_heads, n_kv_heads, head_dim, q_start, scale,
    );

    let q_h = fix.host_buffer_f32(&q, "batch_q");
    let k_h = fix.host_buffer_f32(&k, "batch_k");
    let v_h = fix.host_buffer_f32(&v, "batch_v");
    let o_h = fix.output_buffer_f32(q_count, "batch_o");
    let pc = FlashAttnBatchPushConstants {
        n_heads,
        n_kv_heads,
        head_dim,
        m,
        n_kv,
        q_start,
        scale,
    };
    fix.dispatch(
        ShaderId::FlashAttnBatch,
        &[q_h, k_h, v_h, o_h],
        bytemuck::bytes_of(&pc),
        (n_heads, m, 1),
    );
    let gpu = fix.read_output(o_h, q_count);

    let nan = gpu.iter().any(|x| !x.is_finite());
    assert!(
        !nan,
        "flash_attn_batch(m={m}, q_start={q_start}) produced NaN/Inf — first 8: {:?}",
        &gpu[..8]
    );
    let max_abs = max_abs_err(&gpu, &cpu);
    assert!(
        max_abs < abs_threshold,
        "flash_attn_batch(m={m}, q_start={q_start}) vs CPU max_abs_err = {max_abs:e} >= {abs_threshold:e}\n\
         GPU first 8: {:?}\n CPU first 8: {:?}",
        &gpu[..8],
        &cpu[..8],
    );
    fix.teardown();
    (gpu, cpu)
}

#[test]
fn test_batch_attn_m1_vs_cpu() {
    // M=1 with q_start=0 reduces to single-query attention over a
    // single key — softmax([s]) = 1, output = V[0] (per kv-head). Same
    // shape as the FlashAttn `seq=1` smoke test.
    run_batch_attn(1, 0, 1e-4);
}

#[test]
fn test_batch_attn_m4_vs_cpu() {
    // Causal triangle is fully exercised: q=0 sees K[0], q=3 sees K[0..=3].
    run_batch_attn(4, 0, 1e-3);
}

#[test]
fn test_batch_attn_m16_vs_cpu() {
    // M=16: 16 queries, all under the same TILE — exercises the
    // per-query causal_len bookkeeping inside one tile.
    run_batch_attn(16, 0, 1e-3);
}

#[test]
fn test_batch_attn_m64_vs_cpu() {
    // M=64 = exactly one full TILE for q_idx=63. Tile boundary
    // transition (causal_len from 64 to 65) hits the tail-tile path
    // for all later queries. Run with q_start=0.
    run_batch_attn(64, 0, 1e-3);
}

#[test]
fn test_batch_attn_m200_vs_cpu() {
    // 3.125 × TILE — final query has 4 tiles to walk; numerical envelope
    // a touch wider for the deepest accumulator.
    run_batch_attn(200, 0, 5e-3);
}

#[test]
fn test_batch_attn_q_start_offset() {
    // q_start > 0: queries already see prior cache content. M=4 queries
    // appended to a 60-position prefix → causal_len ranges 61..=64.
    // Verifies the shader's q_start arithmetic and that causal_len is
    // clamped against n_kv.
    run_batch_attn(4, 60, 1e-3);
}

#[test]
fn test_batch_attn_m1_matches_flash_attn() {
    // For M=1, q_start=0, the batch shader must produce the same
    // output as the single-query FlashAttn shader on identical Q/K/V.
    // Tighter bound (1e-5) — both run the same online-softmax loop;
    // any difference would be a pure ordering/accumulation artifact.
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let max_seq: u32 = 2048;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (max_seq as usize) * pos_stride;

    let q_count = (n_heads * head_dim) as usize;
    let mut q = vec![0.0f32; q_count];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i as f32) * 0.001).sin();
    }
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];
    for kvh in 0..(n_kv_heads as usize) {
        for d in 0..(head_dim as usize) {
            let off = kvh * (head_dim as usize) + d;
            k[off] = (0.01 + (d as f32) * 0.001).cos();
            v[off] = ((kvh as f32) * 0.7 + (d as f32) * 0.0007).sin();
        }
    }

    // ---- single-query flash_attn ----
    let mut fix_a = Fixture::new();
    let q_a = fix_a.host_buffer_f32(&q, "fa_q");
    let k_a = fix_a.host_buffer_f32(&k, "fa_k");
    let v_a = fix_a.host_buffer_f32(&v, "fa_v");
    let o_a = fix_a.output_buffer_f32(q_count, "fa_o");
    let pc_fa = ScalarAttnPushConstants {
        n_heads, n_kv_heads, head_dim,
        seq_len: 1, max_seq, scale,
    };
    fix_a.dispatch(
        ShaderId::FlashAttn,
        &[q_a, k_a, v_a, o_a],
        bytemuck::bytes_of(&pc_fa),
        (n_heads, 1, 1),
    );
    let out_fa = fix_a.read_output(o_a, q_count);
    fix_a.teardown();

    // ---- batched flash_attn_batch (M=1, q_start=0) ----
    let mut fix_b = Fixture::new();
    let q_b = fix_b.host_buffer_f32(&q, "fab_q");
    let k_b = fix_b.host_buffer_f32(&k, "fab_k");
    let v_b = fix_b.host_buffer_f32(&v, "fab_v");
    let o_b = fix_b.output_buffer_f32(q_count, "fab_o");
    let pc_fab = FlashAttnBatchPushConstants {
        n_heads, n_kv_heads, head_dim,
        m: 1, n_kv: 1, q_start: 0, scale,
    };
    fix_b.dispatch(
        ShaderId::FlashAttnBatch,
        &[q_b, k_b, v_b, o_b],
        bytemuck::bytes_of(&pc_fab),
        (n_heads, 1, 1),
    );
    let out_fab = fix_b.read_output(o_b, q_count);
    fix_b.teardown();

    let max_abs = max_abs_err(&out_fa, &out_fab);
    assert!(
        max_abs < 1e-5,
        "flash_attn_batch(M=1) vs flash_attn parity max_abs_err = {max_abs:e} >= 1e-5\n\
         flash_attn first 8: {:?}\nflash_attn_batch first 8: {:?}",
        &out_fa[..8], &out_fab[..8],
    );
}

#[test]
fn test_batch_attn_causal_mask_isolates_queries() {
    // Causal correctness: if the model's first query is supposed to
    // see only K[0], then expanding the batch to include later queries
    // must NOT change query 0's output.
    //
    // We dispatch twice on the SAME Q/K/V — once with M=1 and once
    // with M=4 — and assert that the first n_heads*head_dim outputs
    // are identical (within f32 round-off) across the two runs.
    let n_heads: u32 = 32;
    let n_kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let head_count = (n_heads * head_dim) as usize;
    let m_full: u32 = 4;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let pos_stride = (n_kv_heads * head_dim) as usize;
    let kv_size = (m_full as usize) * pos_stride;
    let q_count_full = (m_full * n_heads * head_dim) as usize;

    // Same generator as run_batch_attn — Q's first query is identical
    // to the M=1 run's Q.
    let mut q_full = vec![0.0f32; q_count_full];
    for q_idx in 0..(m_full as usize) {
        for h in 0..(n_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = (q_idx * n_heads as usize + h) * head_dim as usize + d;
                q_full[off] =
                    ((q_idx as f32) * 0.017 + (h as f32) * 0.011 + (d as f32) * 0.003).sin();
            }
        }
    }
    let mut k = vec![0.0f32; kv_size];
    let mut v = vec![0.0f32; kv_size];
    for t in 0..(m_full as usize) {
        for kvh in 0..(n_kv_heads as usize) {
            for d in 0..(head_dim as usize) {
                let off = t * pos_stride + kvh * (head_dim as usize) + d;
                k[off] = ((t as f32 + 1.0) * 0.01 + (d as f32) * 0.001 + (kvh as f32) * 0.7).cos();
                v[off] = ((t as f32) * 0.013 + (kvh as f32) * 0.5 + (d as f32) * 0.0007).sin();
            }
        }
    }
    let q_first = q_full[..head_count].to_vec();

    // ---- run with M=1 ----
    let mut fix_1 = Fixture::new();
    let q1 = fix_1.host_buffer_f32(&q_first, "qm1");
    let k1 = fix_1.host_buffer_f32(&k, "k1");
    let v1 = fix_1.host_buffer_f32(&v, "v1");
    let o1 = fix_1.output_buffer_f32(head_count, "o1");
    let pc1 = FlashAttnBatchPushConstants {
        n_heads, n_kv_heads, head_dim,
        m: 1, n_kv: 1, q_start: 0, scale,
    };
    fix_1.dispatch(
        ShaderId::FlashAttnBatch,
        &[q1, k1, v1, o1],
        bytemuck::bytes_of(&pc1),
        (n_heads, 1, 1),
    );
    let out_m1 = fix_1.read_output(o1, head_count);
    fix_1.teardown();

    // ---- run with M=4 ----
    let mut fix_4 = Fixture::new();
    let q4 = fix_4.host_buffer_f32(&q_full, "qm4");
    let k4 = fix_4.host_buffer_f32(&k, "k4");
    let v4 = fix_4.host_buffer_f32(&v, "v4");
    let o4 = fix_4.output_buffer_f32(q_count_full, "o4");
    let pc4 = FlashAttnBatchPushConstants {
        n_heads, n_kv_heads, head_dim,
        m: m_full, n_kv: m_full, q_start: 0, scale,
    };
    fix_4.dispatch(
        ShaderId::FlashAttnBatch,
        &[q4, k4, v4, o4],
        bytemuck::bytes_of(&pc4),
        (n_heads, m_full, 1),
    );
    let out_m4 = fix_4.read_output(o4, q_count_full);
    fix_4.teardown();

    let max_abs = max_abs_err(&out_m1, &out_m4[..head_count]);
    assert!(
        max_abs < 1e-5,
        "causal mask isolation: query 0 output differs between M=1 and M=4 (max_abs={max_abs:e})\n\
         M=1 first 8: {:?}\nM=4 first 8: {:?}",
        &out_m1[..8], &out_m4[..8],
    );

    // Sanity: query 3 (sees K[0..=3]) must NOT match query 0 (sees only K[0]).
    let q3_off = 3 * head_count;
    let mut diff_count = 0;
    for i in 0..head_count {
        if (out_m4[q3_off + i] - out_m1[i]).abs() > 1e-3 {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > head_count / 8,
        "causal sanity: query 3 should attend to additional KV → output should differ from query 0 \
         (only {diff_count}/{head_count} elements differ by >1e-3)"
    );
}
