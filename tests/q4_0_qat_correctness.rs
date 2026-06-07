//! Sprint 9 (Q4_0 wireup) — byte-gates on the real Gemma-4-26B-A4B QAT
//! GGUF (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, file_type=2: pure
//! Q4_0 weights + F32 norms).
//!
//! Evidence layers (correctness anchor = byte identity vs llama.cpp):
//!
//! 1. **One-hot GEMV column extraction, BIT-EXACT** — a one-hot input
//!    `e_k` turns the GEMV into a pure `d*(q-8)` dequant read of weight
//!    column `k` (every other product is exactly 0.0, and `0.0 + x == x`
//!    in IEEE-754), so there is NO FP-accumulation-order ambiguity and
//!    byte-equality of GPU output vs the CPU `q4_0::dequant_block`
//!    mirror (itself a line-for-line port of ggml's
//!    `dequantize_row_q4_0`) is well-defined and REQUIRED.
//!    With `VF_Q40_DUMP_DIR=<dir>` the GPU outputs are additionally
//!    dumped as f32-LE binaries for the external `np.array_equal`
//!    cross-check against gguf-py's reference dequant (llama.cpp repo).
//! 2. **Random-input GEMV** vs an f64 CPU reference (error bounded
//!    relative to the absolute product mass, not the cancellable sum).
//! 3. **Dense MMQ GEMM** (`QuantizeQ8_1` → `MulMmqQ4_0`) vs an f64 CPU
//!    reference. The Q8_1 activation quantization bounds the achievable
//!    accuracy here (llama.cpp's own MMQ path has the same property),
//!    so this gate is a tight relative-mass tolerance, not bit-equality.
//!    Source fidelity of the integer path is anchored separately: VF's
//!    `mul_mmq_funcs.glsl` / `dequant_funcs.glsl` / `mul_mm_funcs.glsl`
//!    Q4_0 branches diff byte-identical vs llama.cpp.
//! 4. **File-composition guard** — asserts the QAT file is exactly the
//!    pure Q4_0 + F32 layout the wireup was validated against.
//!
//! All tests skip (with a notice) when the QAT GGUF is absent, matching
//! the Sprint-17D test convention.

use std::path::PathBuf;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile};
use vulkanforge::backend::vulkan::pipeline::{
    MatVecPushConstants, MmqPushConstants, Q8_1QuantizePushConstants,
    PUSH_CONSTANT_BYTES,
};
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::q4_0;
use vulkanforge::backend::vulkan::shaders::ShaderId;

fn qat_gguf_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME unset");
    PathBuf::from(home)
        .join("models")
        .join("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
}

fn open_or_skip() -> Option<GgufFile> {
    let p = qat_gguf_path();
    if !p.exists() {
        eprintln!("skip — {} not found", p.display());
        return None;
    }
    Some(GgufFile::open(&p).expect("open QAT gguf"))
}

/// CPU dequant of weight element `W[row, col]` from raw Q4_0 row-major
/// block bytes. Mirrors ggml's `dequantize_row_q4_0`.
fn cpu_dequant_at(bytes: &[u8], k: usize, row: usize, col: usize) -> f32 {
    let blocks_per_row = k / q4_0::QUANT_K;
    let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
    let b = col / q4_0::QUANT_K;
    let e = col % q4_0::QUANT_K;
    let off = row * row_bytes + b * q4_0::BLOCK_BYTES;
    let block: &[u8; q4_0::BLOCK_BYTES] =
        bytes[off..off + q4_0::BLOCK_BYTES].try_into().unwrap();
    q4_0::dequant_block(block)[e]
}

/// Deterministic LCG → f32 in [-1, 1].
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (self.0 >> 40) as u32; // 24 bits
        (v as f32 / (1u32 << 23) as f32) - 1.0
    }
}

/// Shared GPU context for the GEMV tests (device + 5-binding GEMV
/// descriptor wiring, copied from the Sprint-17D test).
struct GemvHarness {
    // Field order = drop order: everything that holds device-child
    // objects must drop BEFORE `dev` (which destroys the VkDevice).
    allocator: Allocator,
    registry: PipelineRegistry,
    cmd_ctx: CommandContext,
    dev: VulkanDevice,
}

impl GemvHarness {
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
        let cmd_ctx =
            CommandContext::new(&dev.device, dev.queue_family_index).unwrap();
        Self { dev, allocator, registry, cmd_ctx }
    }

    /// Run one GEMV dispatch: `out[0..m] = W(m×k, Q4_0 bytes) · input`.
    /// Returns the raw GPU output floats.
    fn gemv(
        &mut self,
        shader: ShaderId,
        weights_bytes: &[u8],
        input: &[f32],
        m: usize,
    ) -> Vec<f32> {
        let k = input.len();
        let kernel = self.registry.get(shader);
        let dev = &self.dev;

        let storage_dst =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
        let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

        let weights_size = weights_bytes.len() as u64;
        let input_bytes: &[u8] = bytemuck::cast_slice(input);
        let input_size = input_bytes.len() as u64;
        let output_size = (m * 4) as u64;

        let weights_buf = GpuBuffer::new(
            &dev.device, &mut self.allocator, weights_size, storage_dst,
            MemoryLocation::GpuOnly, "weights",
        ).unwrap();
        let input_buf = GpuBuffer::new(
            &dev.device, &mut self.allocator, input_size, storage_dst,
            MemoryLocation::GpuOnly, "input",
        ).unwrap();
        let output_buf = GpuBuffer::new(
            &dev.device, &mut self.allocator, output_size, storage_only,
            MemoryLocation::GpuToCpu, "output",
        ).unwrap();
        let fuse0 = GpuBuffer::new(
            &dev.device, &mut self.allocator, 16, storage_only,
            MemoryLocation::GpuOnly, "fuse0",
        ).unwrap();
        let fuse1 = GpuBuffer::new(
            &dev.device, &mut self.allocator, 16, storage_only,
            MemoryLocation::GpuOnly, "fuse1",
        ).unwrap();
        let mut staging_w = GpuBuffer::new(
            &dev.device, &mut self.allocator, weights_size, staging_src,
            MemoryLocation::CpuToGpu, "staging_w",
        ).unwrap();
        let mut staging_i = GpuBuffer::new(
            &dev.device, &mut self.allocator, input_size, staging_src,
            MemoryLocation::CpuToGpu, "staging_i",
        ).unwrap();
        staging_w.write_bytes(weights_bytes).unwrap();
        staging_i.write_bytes(input_bytes).unwrap();

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 5,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool =
            unsafe { dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
        let layouts = [kernel.descriptor_set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let set = unsafe { dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];
        let infos = [
            vk::DescriptorBufferInfo { buffer: weights_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: input_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: output_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: fuse0.handle, offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: fuse1.handle, offset: 0, range: vk::WHOLE_SIZE },
        ];
        let writes: [vk::WriteDescriptorSet; 5] = std::array::from_fn(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i..i + 1])
        });
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

        let pc = MatVecPushConstants {
            ncols: k as u32,
            stride_a: k as u32,
            stride_b: k as u32,
            stride_d: m as u32,
            batch_stride_a: (k * m) as u32,
            batch_stride_b: k as u32,
            batch_stride_d: m as u32,
            fusion_flags: 0,
            base_work_group_y: 0,
            ne02: 1,
            ne12: 1,
            broadcast2: 1,
            broadcast3: 1,
        };
        let pc_bytes: &[u8] = bytemuck::bytes_of(&pc);
        assert_eq!(pc_bytes.len(), PUSH_CONSTANT_BYTES as usize);

        self.cmd_ctx
            .one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
                let copy_w = vk::BufferCopy::default().size(weights_size);
                let copy_i = vk::BufferCopy::default().size(input_size);
                dev.device.cmd_copy_buffer(
                    cmd, staging_w.handle, weights_buf.handle,
                    std::slice::from_ref(&copy_w),
                );
                dev.device.cmd_copy_buffer(
                    cmd, staging_i.handle, input_buf.handle,
                    std::slice::from_ref(&copy_i),
                );
                let bar = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&bar), &[], &[],
                );
                dev.device.cmd_bind_pipeline(
                    cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline,
                );
                dev.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline_layout, 0, &[set], &[],
                );
                dev.device.cmd_push_constants(
                    cmd, kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
                );
                dev.device.cmd_dispatch(cmd, m as u32, 1, 1);
                let post = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                dev.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&post), &[], &[],
                );
            })
            .unwrap();

        let out_bytes = output_buf.read_bytes().unwrap();
        let mut out = vec![0.0f32; m];
        for r in 0..m {
            out[r] = f32::from_le_bytes(out_bytes[r * 4..r * 4 + 4].try_into().unwrap());
        }
        unsafe { dev.device.destroy_descriptor_pool(pool, None) };
        out
    }
}

/// The Q4_0 tensors exercised: dense attn proj, dense FFN down
/// (K=2112, non-256-multiple K regression guard), the tied lm_head
/// table, and the head of the MoE expert gate_up table (same bytes the
/// `*Id` / MMQ_ID variants index into).
const TENSORS: [&str; 4] = [
    "blk.0.attn_q.weight",
    "blk.0.ffn_down.weight",
    "token_embd.weight",
    "blk.0.ffn_gate_up_exps.weight",
];

const M: usize = 64;

#[test]
fn q4_0_qat_onehot_gemv_bitexact() {
    let Some(gguf) = open_or_skip() else { return };
    let mut h = GemvHarness::new();
    let dump_dir = std::env::var("VF_Q40_DUMP_DIR").ok();
    if let Some(d) = &dump_dir {
        std::fs::create_dir_all(d).unwrap();
    }

    for name in TENSORS {
        let info = gguf.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(info.ggml_type, GgmlType::Q4_0, "{name} not Q4_0");
        let k = info.dimensions[0] as usize;
        assert_eq!(k % q4_0::QUANT_K, 0);
        let row_bytes = k / q4_0::QUANT_K * q4_0::BLOCK_BYTES;
        let bytes = &gguf.tensor_bytes(info)[..M * row_bytes];

        // Columns covering: first elem, low/high nibble of block 0,
        // block boundary, mid-tensor, last column.
        let cols = [0usize, 7, 15, 16, 31, 32, 49, k / 2 + 1, k - 1];
        let mut dump: Vec<f32> = Vec::new();
        for shader in [ShaderId::MulMatVecQ4_0, ShaderId::MulMatVecQ4_0Subgroup] {
            for &c in &cols {
                let mut input = vec![0.0f32; k];
                input[c] = 1.0;
                let gpu = h.gemv(shader, bytes, &input, M);
                for r in 0..M {
                    let cpu = cpu_dequant_at(bytes, k, r, c);
                    assert_eq!(
                        gpu[r].to_bits(),
                        cpu.to_bits(),
                        "{name} {shader:?} col {c} row {r}: GPU {} != CPU {} (bit-exact required)",
                        gpu[r], cpu,
                    );
                }
                if shader == ShaderId::MulMatVecQ4_0 {
                    dump.extend_from_slice(&gpu);
                }
            }
        }
        eprintln!("[onehot] {name}: {} cols × {M} rows × 2 shaders bit-exact ✓", cols.len());
        if let Some(d) = &dump_dir {
            let safe = name.replace('.', "_");
            std::fs::write(
                format!("{d}/{safe}.bin"),
                bytemuck::cast_slice::<f32, u8>(&dump),
            ).unwrap();
            std::fs::write(
                format!("{d}/{safe}.cols"),
                cols.map(|c| c.to_string()).join(","),
            ).unwrap();
        }
    }
}

#[test]
fn q4_0_qat_random_gemv_matches_f64_reference() {
    let Some(gguf) = open_or_skip() else { return };
    let mut h = GemvHarness::new();

    for name in TENSORS {
        let info = gguf.tensor(name).unwrap();
        let k = info.dimensions[0] as usize;
        let row_bytes = k / q4_0::QUANT_K * q4_0::BLOCK_BYTES;
        let bytes = &gguf.tensor_bytes(info)[..M * row_bytes];

        let mut rng = Lcg(0x5eed_5eed_5eed_5eed);
        let input: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();

        for shader in [ShaderId::MulMatVecQ4_0, ShaderId::MulMatVecQ4_0Subgroup] {
            let gpu = h.gemv(shader, bytes, &input, M);
            let mut max_rel = 0.0f64;
            for r in 0..M {
                let mut acc = 0.0f64;
                let mut mass = 0.0f64;
                for c in 0..k {
                    let w = cpu_dequant_at(bytes, k, r, c) as f64;
                    let p = w * input[c] as f64;
                    acc += p;
                    mass += p.abs();
                }
                let err = (gpu[r] as f64 - acc).abs();
                let rel = err / mass.max(1e-12);
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 1e-3,
                    "{name} {shader:?} row {r}: GPU {} vs f64 ref {acc} (err {err:.3e}, rel-to-mass {rel:.3e})",
                    gpu[r],
                );
            }
            eprintln!("[random] {name} {shader:?}: max rel-to-mass {max_rel:.3e} ✓");
        }
    }
}

#[test]
fn q4_0_qat_dense_mmq_gemm_matches_f64_reference() {
    let Some(gguf) = open_or_skip() else { return };
    let mut h = GemvHarness::new();

    let name = "blk.0.attn_q.weight";
    let info = gguf.tensor(name).unwrap();
    let k = info.dimensions[0] as usize; // 2816
    let row_bytes = k / q4_0::QUANT_K * q4_0::BLOCK_BYTES;
    let weights_bytes = &gguf.tensor_bytes(info)[..M * row_bytes];

    const N: usize = 8; // tokens
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let input: Vec<f32> = (0..N * k).map(|_| rng.next_f32()).collect();

    let dev_handle = &h.dev;
    let q8_kernel = h.registry.get(ShaderId::QuantizeQ8_1);
    let mmq_kernel = h.registry.get(ShaderId::MulMmqQ4_0);

    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;

    let weights_size = weights_bytes.len() as u64;
    let input_bytes: &[u8] = bytemuck::cast_slice(&input);
    // Q8_1 packed: 36 B per 32-element block (+ slack for x4 packing).
    let q8_size = ((N * k / 32) * 36 + 256) as u64;
    let output_size = (N * M * 4) as u64;

    let weights_buf = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, weights_size, storage_dst,
        MemoryLocation::GpuOnly, "weights",
    ).unwrap();
    let input_buf = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, input_bytes.len() as u64, storage_dst,
        MemoryLocation::GpuOnly, "input",
    ).unwrap();
    let q8_buf = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, q8_size, storage_only,
        MemoryLocation::GpuOnly, "q8",
    ).unwrap();
    let output_buf = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, output_size, storage_only,
        MemoryLocation::GpuToCpu, "output",
    ).unwrap();
    let mut staging_w = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, weights_size,
        vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "sw",
    ).unwrap();
    let mut staging_i = GpuBuffer::new(
        &dev_handle.device, &mut h.allocator, input_bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "si",
    ).unwrap();
    staging_w.write_bytes(weights_bytes).unwrap();
    staging_i.write_bytes(input_bytes).unwrap();

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 8,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2)
        .pool_sizes(&pool_sizes);
    let pool = unsafe {
        dev_handle.device.create_descriptor_pool(&pool_info, None)
    }.unwrap();

    // Set 1: quantize (input → q8).
    let l1 = [q8_kernel.descriptor_set_layout];
    let a1 = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&l1);
    let set_q = unsafe { dev_handle.device.allocate_descriptor_sets(&a1) }.unwrap()[0];
    let qi = [
        vk::DescriptorBufferInfo { buffer: input_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: q8_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let qw: [vk::WriteDescriptorSet; 2] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set_q).dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&qi[i..i + 1])
    });
    unsafe { dev_handle.device.update_descriptor_sets(&qw, &[]) };

    // Set 2: MMQ (weights, q8, output).
    let l2 = [mmq_kernel.descriptor_set_layout];
    let a2 = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool).set_layouts(&l2);
    let set_m = unsafe { dev_handle.device.allocate_descriptor_sets(&a2) }.unwrap()[0];
    let mi = [
        vk::DescriptorBufferInfo { buffer: weights_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: q8_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: output_buf.handle, offset: 0, range: vk::WHOLE_SIZE },
    ];
    let mw: [vk::WriteDescriptorSet; 3] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set_m).dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&mi[i..i + 1])
    });
    unsafe { dev_handle.device.update_descriptor_sets(&mw, &[]) };

    let ne = (N * k) as u32;
    let q8_pc = Q8_1QuantizePushConstants { ne, num_blocks: (ne + 127) / 128 };
    let mmq_pc = MmqPushConstants {
        m: M as u32,
        n: N as u32,
        k: k as u32,
        stride_a: k as u32,
        stride_b: k as u32,
        stride_d: M as u32,
        batch_stride_a: (M * k) as u32,
        batch_stride_b: (N * k) as u32,
        batch_stride_d: (M * N) as u32,
        base_work_group_z: 0,
        num_batches: 1,
        k_split: k as u32,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };

    h.cmd_ctx
        .one_shot(&dev_handle.device, dev_handle.compute_queue, |cmd| unsafe {
            let cw = vk::BufferCopy::default().size(weights_size);
            let ci = vk::BufferCopy::default().size(input_bytes.len() as u64);
            dev_handle.device.cmd_copy_buffer(
                cmd, staging_w.handle, weights_buf.handle, std::slice::from_ref(&cw),
            );
            dev_handle.device.cmd_copy_buffer(
                cmd, staging_i.handle, input_buf.handle, std::slice::from_ref(&ci),
            );
            let b1 = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            dev_handle.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&b1), &[], &[],
            );
            // Quantize activations.
            dev_handle.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, q8_kernel.pipeline,
            );
            dev_handle.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                q8_kernel.pipeline_layout, 0, &[set_q], &[],
            );
            dev_handle.device.cmd_push_constants(
                cmd, q8_kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&q8_pc),
            );
            dev_handle.device.cmd_dispatch(cmd, q8_pc.num_blocks, 1, 1);
            let b2 = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            dev_handle.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&b2), &[], &[],
            );
            // MMQ GEMM.
            dev_handle.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, mmq_kernel.pipeline,
            );
            dev_handle.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                mmq_kernel.pipeline_layout, 0, &[set_m], &[],
            );
            dev_handle.device.cmd_push_constants(
                cmd, mmq_kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&mmq_pc),
            );
            dev_handle.device.cmd_dispatch(
                cmd,
                ((M as u32) + 63) / 64,
                ((N as u32) + 63) / 64,
                1,
            );
            let b3 = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            dev_handle.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&b3), &[], &[],
            );
        })
        .unwrap();

    let out_bytes = output_buf.read_bytes().unwrap();
    let gpu: &[f32] = bytemuck::cast_slice(&out_bytes[..N * M * 4]);

    let mut max_rel = 0.0f64;
    for t in 0..N {
        for r in 0..M {
            let mut acc = 0.0f64;
            let mut mass = 0.0f64;
            for c in 0..k {
                let w = cpu_dequant_at(weights_bytes, k, r, c) as f64;
                let p = w * input[t * k + c] as f64;
                acc += p;
                mass += p.abs();
            }
            // Output layout: D[N × M] row-major, stride_d = M.
            let g = gpu[t * M + r] as f64;
            let rel = (g - acc).abs() / mass.max(1e-12);
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-2,
                "token {t} row {r}: GPU {g} vs f64 ref {acc} (rel-to-mass {rel:.3e}; \
                 Q8_1 activation quantization bounds this path)",
            );
        }
    }
    eprintln!("[mmq] {name}: {N}×{M} outputs, max rel-to-mass {max_rel:.3e} ✓ (Q8_1-bounded path)");
    unsafe { dev_handle.device.destroy_descriptor_pool(pool, None) };
}

#[test]
fn q4_0_qat_file_composition_guard() {
    let Some(gguf) = open_or_skip() else { return };
    let mut q4_0 = 0usize;
    let mut f32_ = 0usize;
    let mut other = 0usize;
    for (name, info) in &gguf.tensors {
        match info.ggml_type {
            GgmlType::Q4_0 => q4_0 += 1,
            GgmlType::F32 => f32_ += 1,
            t => {
                eprintln!("unexpected tensor type {t:?}: {name}");
                other += 1;
            }
        }
    }
    eprintln!("[composition] Q4_0={q4_0} F32={f32_} other={other}");
    assert_eq!(other, 0, "QAT file should be pure Q4_0 + F32");
    assert_eq!(q4_0, 266);
    assert_eq!(f32_, 392);
    let embd = gguf.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.ggml_type, GgmlType::Q4_0, "tied lm_head table must be Q4_0");
    assert!(gguf.tensor("output.weight").is_none(), "lm_head is tied (token_embd)");
}
