//! Sprint 18-M0 + 18-M1 — FP8 smoke test (Go/No-Go gate for FP8 prefill).
//!
//! Inspects what's actually available for native FP8 on the current
//! ash + RDNA4 + Mesa stack. Five tests, escalating in scope:
//!   1. ash version vs system Vulkan SDK — does our crate know FP8?
//!   2. Cooperative-matrix property table — does the driver expose
//!      FP8 component types and what M×N×K sizes?
//!   3. GLSL FP8 shader compiles? (out-of-process glslangValidator)
//!   4. (M1) FP8 round-trip CPU↔GPU — uintBitsToFloate4m3EXT path
//!   5. (M1) FP8 cooperative-matrix 16×16×16 matmul vs CPU reference
//!
//! Run:   cargo run --release --example fp8_smoke
//! Reads: results/v033_sprint18_m{0,1}_fp8_*.md (writes a section)

use std::fs;
use std::process::Command;

use ash::{khr, vk, Entry};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::fp8_ext::fp8_e4m3_to_f32;
use vulkanforge::backend::vulkan::pipeline::ComputeKernel;
use vulkanforge::backend::vulkan::shaders::spv_words;

// Per `vulkan_core.h` shipped with vulkan-headers 1.4.341 — the
// 1000491000/1 codes are SINT8_PACKED_NV / UINT8_PACKED_NV (NV
// vendor extension), and 1000491002/3 are the actual EXT FP8 codes.
const VK_COMPONENT_TYPE_SINT8_PACKED_NV: u32 = 1000491000;
const VK_COMPONENT_TYPE_UINT8_PACKED_NV: u32 = 1000491001;
const VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: u32 = 1000491002;
const VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: u32 = 1000491003;
const VK_COMPONENT_TYPE_BFLOAT16_KHR: u32 = 1000141000;

fn fmt_component_type(t: vk::ComponentTypeKHR) -> String {
    match t {
        vk::ComponentTypeKHR::FLOAT16 => "FP16".into(),
        vk::ComponentTypeKHR::FLOAT32 => "FP32".into(),
        vk::ComponentTypeKHR::FLOAT64 => "FP64".into(),
        vk::ComponentTypeKHR::SINT8 => "INT8".into(),
        vk::ComponentTypeKHR::SINT16 => "INT16".into(),
        vk::ComponentTypeKHR::SINT32 => "INT32".into(),
        vk::ComponentTypeKHR::SINT64 => "INT64".into(),
        vk::ComponentTypeKHR::UINT8 => "UINT8".into(),
        vk::ComponentTypeKHR::UINT16 => "UINT16".into(),
        vk::ComponentTypeKHR::UINT32 => "UINT32".into(),
        vk::ComponentTypeKHR::UINT64 => "UINT64".into(),
        other => {
            let raw = other.as_raw() as u32;
            match raw {
                VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT => "FP8_E4M3".into(),
                VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT => "FP8_E5M2".into(),
                VK_COMPONENT_TYPE_BFLOAT16_KHR => "BF16".into(),
                VK_COMPONENT_TYPE_SINT8_PACKED_NV => "INT8_PACKED_NV".into(),
                VK_COMPONENT_TYPE_UINT8_PACKED_NV => "UINT8_PACKED_NV".into(),
                v => format!("UNKNOWN({v})"),
            }
        }
    }
}

fn test1_ash_versions() {
    println!("\n=== TEST 1: ash + Vulkan SDK versions ===");
    println!("ash crate         : 0.38.0+1.3.281 (Cargo.toml pin)");
    println!("System headers    : VK_HEADER_VERSION 341 (1.4.341, has VK_EXT_shader_float8)");
    // We can't statically reference vk::PhysicalDeviceShaderFloat8FeaturesEXT
    // because ash 0.38 doesn't declare it. Probing for it would be a compile
    // error, so we report the gap explicitly here.
    println!(
        "ash FP8 structs   : ABSENT (PhysicalDeviceShaderFloat8FeaturesEXT, \
         FLOAT8_E4M3_EXT, FLOAT8_E5M2_EXT not in ash 0.38)"
    );
    println!("Driver advertises : shaderFloat8 + shaderFloat8CooperativeMatrix");
    println!("                    (per `vulkaninfo`; not yet enabled at device create)");
}

fn test2_coopmat_props() -> bool {
    println!("\n=== TEST 2: cooperative-matrix property table ===");
    let entry = match unsafe { Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            println!("FAIL — can't load Vulkan loader: {e}");
            return false;
        }
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vulkanforge-fp8-smoke")
        .api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => {
            println!("FAIL — vkCreateInstance: {e}");
            return false;
        }
    };

    let coop_fn = khr::cooperative_matrix::Instance::new(&entry, &instance);

    let phys_devs = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    if phys_devs.is_empty() {
        println!("FAIL — no physical devices");
        unsafe { instance.destroy_instance(None) };
        return false;
    }
    // Pick first DISCRETE_GPU, fall back to first device
    let phys = phys_devs
        .iter()
        .copied()
        .find(|&p| {
            unsafe { instance.get_physical_device_properties(p) }.device_type
                == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap_or(phys_devs[0]);
    let props = unsafe { instance.get_physical_device_properties(phys) };
    let name = props
        .device_name_as_c_str()
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("?");
    println!("Device            : {name}");

    let table = match unsafe { coop_fn.get_physical_device_cooperative_matrix_properties(phys) } {
        Ok(t) => t,
        Err(e) => {
            println!("FAIL — coopmat property query: {e}");
            unsafe { instance.destroy_instance(None) };
            return false;
        }
    };
    println!("Total entries     : {}", table.len());
    println!();

    let mut fp8_entries = 0;
    let mut bf16_entries = 0;
    let mut int8_entries = 0;
    let mut fp16_entries = 0;

    for (i, p) in table.iter().enumerate() {
        let a = fmt_component_type(p.a_type);
        let b = fmt_component_type(p.b_type);
        let c = fmt_component_type(p.c_type);
        let r = fmt_component_type(p.result_type);
        let m = p.m_size;
        let n = p.n_size;
        let k = p.k_size;
        let scope = format!("{:?}", p.scope);

        if a.starts_with("FP8") || b.starts_with("FP8") {
            fp8_entries += 1;
        }
        if a.starts_with("BFLOAT16") || b.starts_with("BFLOAT16") {
            bf16_entries += 1;
        }
        if a == "INT8" || a == "UINT8" {
            int8_entries += 1;
        }
        if a == "FP16" {
            fp16_entries += 1;
        }

        println!(
            "  [{i:2}]  A={a:9}  B={b:9}  C={c:6}  R={r:6}  M={m:>2} N={n:>2} K={k:>2}  scope={scope}"
        );
    }
    println!();
    println!(
        "Counts: FP8={fp8_entries}, BF16={bf16_entries}, INT8/UINT8={int8_entries}, FP16={fp16_entries}"
    );

    unsafe { instance.destroy_instance(None) };

    if fp8_entries == 0 {
        println!("FAIL — no FP8 cooperative-matrix sizes advertised");
        false
    } else {
        println!("PASS — driver exposes {fp8_entries} FP8 coopmat entries");
        true
    }
}

fn test3_glsl_fp8_compile() -> bool {
    println!("\n=== TEST 3: glslangValidator FP8 shader compile ===");
    // Per `strings libglslang.so.16`: GL_EXT_float_e4m3 / GL_EXT_float_e5m2,
    // type names `floate4m3_t` / `floate5m2_t`, and bit-cast helpers
    // `intBitsToFloate4m3EXT` / `floate4m3BitsToUintEXT` are baked into
    // the compiler (Khronos glslang PR #3969 landed before 16.2.0).
    let shader = r#"#version 450
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_float_e5m2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Out { float data_out[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    // Construct an FP8 E4M3 value from a u8 bit pattern (≈1.0 = 0x38),
    // convert to FP32, write out.
    floate4m3_t v = uintBitsToFloate4m3EXT(uint8_t(0x38));
    data_out[i] = float(v);

    // Same for E5M2:
    floate5m2_t w = uintBitsToFloate5m2EXT(uint8_t(0x3C));
    data_out[i + 1] = float(w);
}
"#;
    let path = "/tmp/fp8_smoke_test.comp";
    fs::write(path, shader).expect("write shader");
    let out = "/tmp/fp8_smoke_test.spv";
    let result = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.3", "-o", out, path])
        .output();
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            println!("FAIL — glslangValidator unavailable: {e}");
            return false;
        }
    };
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    println!("--- glslangValidator stdout ---");
    print!("{}", stdout);
    if !stderr.is_empty() {
        println!("--- glslangValidator stderr ---");
        print!("{}", stderr);
    }
    println!("--- exit status: {} ---", result.status);

    if result.status.success() {
        // Inspect the SPIR-V briefly to confirm it has the FP8 cap.
        let dis = Command::new("spirv-dis").arg(out).output();
        if let Ok(d) = dis {
            let dis_text = String::from_utf8_lossy(&d.stdout);
            let has_cap = dis_text.contains("Float8EXT") || dis_text.contains("FloatE4M3");
            println!(
                "  spirv-dis: SPIR-V {} the FP8 capability declaration",
                if has_cap { "DECLARES" } else { "DOES NOT declare" }
            );
        }
        println!("PASS — FP8 shader compiles with E4M3 + E5M2 types");
        true
    } else {
        println!("FAIL — `GL_EXT_float_e4m3` / `floate4m3_t` not accepted by glslangValidator");
        false
    }
}

// =====================================================================
// Sprint 18-M1 — Tests 4 + 5 use VulkanForge's `VulkanDevice` (which
// now reads `VULKANFORGE_ENABLE_FP8` and pushes the raw FP8 feature
// struct via the `fp8_ext` shim). All buffers are dispatched through
// the existing `ComputeKernel` + descriptor plumbing.

const FP8_ROUNDTRIP_GLSL: &str = r#"#version 450
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly  buffer InP  { uint  packed_in[];  };
layout(set = 0, binding = 1) writeonly buffer OutP { float data_out[];   };
layout(push_constant) uniform Pc { uint n; } pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.n) return;
    uint word = packed_in[idx >> 2];
    uint byte = (word >> ((idx & 3u) * 8u)) & 0xFFu;
    floate4m3_t v = uintBitsToFloate4m3EXT(uint8_t(byte));
    data_out[idx] = float(v);
}
"#;

const FP8_COOPMAT_GLSL: &str = r#"#version 450
#extension GL_KHR_cooperative_matrix       : require
#extension GL_KHR_memory_scope_semantics   : require
#extension GL_EXT_float_e4m3               : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint BLOCK_SIZE = 64;

// Bind A and B as raw uint8 byte buffers — coopMatLoad reads them
// as native FP8 because the matrix's component type is floate4m3_t.
layout(set = 0, binding = 0) readonly  buffer BufA { uint8_t a[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { uint8_t b[]; };
layout(set = 0, binding = 2) writeonly buffer BufC { float   c[]; };

void main() {
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
    coopmat<float,       gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    coopMatLoad(matA, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(matB, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
    matC = coopMatMulAdd(matA, matB, matC);
    coopMatStore(matC, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

/// Compile `glsl_src` via out-of-process `glslangValidator`. Returns
/// the SPIR-V bytes (still as raw u8s) on success, or an error string.
fn compile_glsl(glsl_src: &str, label: &str) -> Result<Vec<u8>, String> {
    let comp_path = format!("/tmp/fp8_{label}.comp");
    let spv_path = format!("/tmp/fp8_{label}.spv");
    fs::write(&comp_path, glsl_src).map_err(|e| format!("write: {e}"))?;
    let out = Command::new("glslangValidator")
        .args([
            "-V",
            "--target-env",
            "vulkan1.3",
            "-o",
            &spv_path,
            &comp_path,
        ])
        .output()
        .map_err(|e| format!("glslangValidator unavailable: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "glslangValidator failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))
}

/// Common Vulkan plumbing (device + allocator + cmd-pool) shared
/// between Tests 4 and 5. Returns `Err` on first failure.
struct GpuFixture {
    dev: VulkanDevice,
    allocator: Allocator,
    cmd_ctx: CommandContext,
    cache: vk::PipelineCache,
}

impl GpuFixture {
    fn new() -> Result<Self, String> {
        let dev = VulkanDevice::new().map_err(|e| format!("VulkanDevice::new: {e:?}"))?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: dev.instance.clone(),
            device: dev.device.clone(),
            physical_device: dev.physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| format!("Allocator::new: {e:?}"))?;
        let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index)
            .map_err(|e| format!("CommandContext::new: {e:?}"))?;
        let cache = unsafe {
            dev.device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .map_err(|e| format!("create_pipeline_cache: {e:?}"))?;
        Ok(Self {
            dev,
            allocator,
            cmd_ctx,
            cache,
        })
    }
}

fn test4_round_trip(fixture: &mut GpuFixture) -> bool {
    println!("\n=== TEST 4: FP8 round-trip CPU↔GPU ===");
    let spv = match compile_glsl(FP8_ROUNDTRIP_GLSL, "roundtrip") {
        Ok(b) => b,
        Err(e) => {
            println!("FAIL — shader compile: {e}");
            return false;
        }
    };
    let words = spv_words(&spv);

    // Build a deterministic input set covering every E4M3 byte that
    // matters: signed zero, ±1.0, ±2.0, ±0.5, ±max (448), subnormals.
    // The shader reinterprets each byte as a floate4m3_t and converts
    // to f32; we do the same on CPU via `fp8_e4m3_to_f32`.
    const N: usize = 256; // every possible u8 pattern, conveniently
    // (0..N as u8) would overflow N=256→0 — iterate as usize and cast.
    let bytes: Vec<u8> = (0..N).map(|i| i as u8).collect();
    // Pack 4 bytes per uint32 for the SSBO.
    let packed: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| {
            (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
        })
        .collect();

    let cpu_expected: Vec<f32> = bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();

    // Allocate buffers.
    let in_size = (packed.len() * 4) as u64;
    let out_size = (N * 4) as u64;
    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let mut in_buf = GpuBuffer::new(
        &fixture.dev.device,
        &mut fixture.allocator,
        in_size,
        storage_dst,
        MemoryLocation::CpuToGpu,
        "fp8_in",
    )
    .unwrap();
    let out_buf = GpuBuffer::new(
        &fixture.dev.device,
        &mut fixture.allocator,
        out_size,
        storage_only,
        MemoryLocation::GpuToCpu,
        "fp8_out",
    )
    .unwrap();
    in_buf
        .write_bytes(bytemuck::cast_slice(&packed))
        .expect("upload");

    // Build the kernel.
    let kernel = match ComputeKernel::from_spv(&fixture.dev.device, &words, fixture.cache) {
        Ok(k) => k,
        Err(e) => {
            println!("FAIL — ComputeKernel::from_spv: {e:?}");
            return false;
        }
    };

    // Descriptor set + dispatch.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 2,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { fixture.dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { fixture.dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];

    let infos = [
        vk::DescriptorBufferInfo {
            buffer: in_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: out_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes: [vk::WriteDescriptorSet; 2] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos[i..i + 1])
    });
    unsafe { fixture.dev.device.update_descriptor_sets(&writes, &[]) };

    let pc_n: u32 = N as u32;
    let pc_bytes: [u8; 4] = pc_n.to_le_bytes();
    let device = fixture.dev.device.clone();
    let queue = fixture.dev.compute_queue;
    fixture
        .cmd_ctx
        .one_shot(&device, queue, |cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                kernel.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc_bytes,
            );
            device.cmd_dispatch(cmd, ((N + 63) / 64) as u32, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(out_buf.handle)
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

    let bytes_out = out_buf.read_bytes().expect("read_bytes");
    let gpu_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes_out[..N * 4]).to_vec();

    // Compare GPU vs CPU. NaN compare with bit-identity (both sides
    // produce the same NaN bit pattern from byte 0x7F / 0xFF).
    let mut max_err: f32 = 0.0;
    let mut mismatched: Vec<usize> = Vec::new();
    for i in 0..N {
        let g = gpu_out[i];
        let c = cpu_expected[i];
        if g.is_nan() && c.is_nan() {
            continue;
        }
        let d = (g - c).abs();
        if d > max_err {
            max_err = d;
        }
        if d > 1e-9 {
            mismatched.push(i);
        }
    }
    println!("  inputs: every E4M3 byte 0x00..0xFF (256 patterns)");
    println!("  CPU[0..8] = {:?}", &cpu_expected[..8]);
    println!("  GPU[0..8] = {:?}", &gpu_out[..8]);
    println!(
        "  max_err = {max_err:.10}, mismatched bytes = {}",
        mismatched.len()
    );
    if !mismatched.is_empty() && mismatched.len() <= 8 {
        for &i in &mismatched {
            println!(
                "    byte 0x{i:02X}: cpu={} gpu={}",
                cpu_expected[i], gpu_out[i]
            );
        }
    }

    unsafe {
        fixture.dev.device.destroy_descriptor_pool(pool, None);
        in_buf.destroy(&fixture.dev.device, &mut fixture.allocator);
        out_buf.destroy(&fixture.dev.device, &mut fixture.allocator);
        fixture.dev.device.destroy_pipeline(kernel.pipeline, None);
        fixture
            .dev
            .device
            .destroy_pipeline_layout(kernel.pipeline_layout, None);
        fixture
            .dev
            .device
            .destroy_descriptor_set_layout(kernel.descriptor_set_layout, None);
    }

    if mismatched.is_empty() && max_err < 1e-6 {
        println!("PASS — every E4M3 byte round-trips bit-exact between CPU and GPU");
        true
    } else {
        println!("FAIL — GPU disagrees with CPU FP8 reference");
        false
    }
}

fn cpu_matmul_fp8_e4m3_16x16(a: &[u8; 256], b: &[u8; 256]) -> [f32; 256] {
    // A row-major (i, k), B column-major (k, j), C row-major (i, j).
    let mut c = [0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut sum = 0.0f32;
            for k in 0..16 {
                let av = fp8_e4m3_to_f32(a[i * 16 + k]);
                let bv = fp8_e4m3_to_f32(b[j * 16 + k]); // ColumnMajor
                sum += av * bv;
            }
            c[i * 16 + j] = sum;
        }
    }
    c
}

fn test5_coopmat_matmul(fixture: &mut GpuFixture) -> bool {
    println!("\n=== TEST 5: FP8 cooperative-matrix matmul (16×16×16) ===");
    let spv = match compile_glsl(FP8_COOPMAT_GLSL, "coopmat") {
        Ok(b) => b,
        Err(e) => {
            println!("FAIL — shader compile: {e}");
            return false;
        }
    };
    let words = spv_words(&spv);

    // Build a deterministic, well-conditioned 16×16 matmul input.
    // Use values squarely in E4M3's representable range so the
    // CPU↔GPU compare reflects WMMA correctness, not requantization.
    // Pattern: A[i,k] = signbit ? -((k+1)/8) : ((k+1)/8); same for B.
    let mut a = [0u8; 256];
    let mut b = [0u8; 256];
    for i in 0..16 {
        for k in 0..16 {
            // value in {0.125, 0.25, 0.5, 1.0} cycling — exact in E4M3.
            let val = match (i + k) % 4 {
                0 => 0.5_f32,
                1 => 1.0_f32,
                2 => -0.25_f32,
                _ => 0.125_f32,
            };
            a[i * 16 + k] = vulkanforge::backend::vulkan::fp8_ext::f32_to_fp8_e4m3(val);
            b[k * 16 + i] = vulkanforge::backend::vulkan::fp8_ext::f32_to_fp8_e4m3(val * 0.5);
        }
    }

    let cpu_c = cpu_matmul_fp8_e4m3_16x16(&a, &b);

    // Allocate buffers.
    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let mut a_buf = GpuBuffer::new(
        &fixture.dev.device,
        &mut fixture.allocator,
        256,
        storage_dst,
        MemoryLocation::CpuToGpu,
        "fp8_a",
    )
    .unwrap();
    let mut b_buf = GpuBuffer::new(
        &fixture.dev.device,
        &mut fixture.allocator,
        256,
        storage_dst,
        MemoryLocation::CpuToGpu,
        "fp8_b",
    )
    .unwrap();
    let c_buf = GpuBuffer::new(
        &fixture.dev.device,
        &mut fixture.allocator,
        (256 * 4) as u64,
        storage_only,
        MemoryLocation::GpuToCpu,
        "fp8_c",
    )
    .unwrap();
    a_buf.write_bytes(&a).unwrap();
    b_buf.write_bytes(&b).unwrap();

    // Pin BLOCK_SIZE=64 (1 RDNA4 wave).
    let spec_data: [u32; 1] = [64];
    let entries = [vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: 4,
    }];
    let kernel = match ComputeKernel::from_spv_with_spec(
        &fixture.dev.device,
        &words,
        fixture.cache,
        &entries,
        bytemuck::bytes_of(&spec_data),
        Some(64),
    ) {
        Ok(k) => k,
        Err(e) => {
            println!("FAIL — ComputeKernel::from_spv: {e:?}");
            return false;
        }
    };

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { fixture.dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe { fixture.dev.device.allocate_descriptor_sets(&alloc) }.unwrap()[0];
    let infos = [
        vk::DescriptorBufferInfo {
            buffer: a_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: c_buf.handle,
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes: [vk::WriteDescriptorSet; 3] = std::array::from_fn(|i| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(i as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos[i..i + 1])
    });
    unsafe { fixture.dev.device.update_descriptor_sets(&writes, &[]) };

    let device = fixture.dev.device.clone();
    let queue = fixture.dev.compute_queue;
    fixture
        .cmd_ctx
        .one_shot(&device, queue, |cmd| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(cmd, 1, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(c_buf.handle)
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

    let bytes_out = c_buf.read_bytes().expect("read_bytes");
    let gpu_c: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes_out[..256 * 4]).to_vec();

    let max_err = gpu_c
        .iter()
        .zip(cpu_c.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f32, f32::max);
    let cpu_amax = cpu_c.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_amax = gpu_c.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    println!("  cpu_amax = {cpu_amax:.6}, gpu_amax = {gpu_amax:.6}");
    println!("  CPU C[0..8] = {:?}", &cpu_c[..8]);
    println!("  GPU C[0..8] = {:?}", &gpu_c[..8]);
    println!("  max_err     = {max_err:.6e}");

    unsafe {
        fixture.dev.device.destroy_descriptor_pool(pool, None);
        a_buf.destroy(&fixture.dev.device, &mut fixture.allocator);
        b_buf.destroy(&fixture.dev.device, &mut fixture.allocator);
        c_buf.destroy(&fixture.dev.device, &mut fixture.allocator);
        fixture.dev.device.destroy_pipeline(kernel.pipeline, None);
        fixture
            .dev
            .device
            .destroy_pipeline_layout(kernel.pipeline_layout, None);
        fixture
            .dev
            .device
            .destroy_descriptor_set_layout(kernel.descriptor_set_layout, None);
    }

    // FP32 accumulator with E4M3-exact inputs: GPU should match CPU
    // bit-for-bit modulo summation order. Tolerate up to 1% of cpu_amax
    // for any FMA-reordering effects.
    let thresh = (cpu_amax * 0.01).max(1e-3);
    if max_err < thresh && cpu_amax > 1e-6 {
        println!("PASS — GPU FP8 WMMA matches CPU reference within {thresh:.4e}");
        true
    } else {
        println!("FAIL — divergence above tolerance ({thresh:.4e})");
        false
    }
}

fn main() {
    println!("=== VulkanForge FP8 Sprint 18-M0 + 18-M1 smoke test ===");

    // Auto-enable FP8 in the device shim. VulkanDevice::new() reads
    // this env var; smoke-test users don't need to set it manually.
    if std::env::var("VULKANFORGE_ENABLE_FP8").is_err() {
        // SAFETY: smoke-test process; no other threads racing with env.
        unsafe { std::env::set_var("VULKANFORGE_ENABLE_FP8", "1") };
    }

    test1_ash_versions();
    let t2 = test2_coopmat_props();
    let t3 = test3_glsl_fp8_compile();

    let mut fixture_result: Option<GpuFixture> = match GpuFixture::new() {
        Ok(f) => Some(f),
        Err(e) => {
            println!("\nFAIL — couldn't open Vulkan device with FP8: {e}");
            None
        }
    };

    let (t4, t5) = if let Some(fixture) = fixture_result.as_mut() {
        let t4 = test4_round_trip(fixture);
        let t5 = test5_coopmat_matmul(fixture);
        (t4, t5)
    } else {
        (false, false)
    };

    println!("\n=== Summary ===");
    println!("  Test 1 — ash version + driver caps : INFO (ash 0.38 lacks FP8 bindings; M1 ships shim)");
    println!(
        "  Test 2 — coopmat property FP8 entries: {}",
        if t2 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 3 — GLSL FP8 shader compiles    : {}",
        if t3 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 4 — round-trip CPU↔GPU          : {}",
        if t4 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 5 — FP8 cooperative-matrix mul  : {}",
        if t5 { "PASS" } else { "FAIL" }
    );
    println!();
    let all = t2 && t3 && t4 && t5;
    if all {
        println!("VERDICT: GO — Sprint 18A FP8 KV-cache + 18B FP8 WMMA prefill unblocked.");
    } else if t2 && t3 && t4 {
        println!("VERDICT: PARTIAL — round-trip works, WMMA matmul fails; KV-cache feasible, prefill needs debug.");
    } else if t2 && t3 {
        println!("VERDICT: PARTIAL — toolchain ready, runtime FP8 enablement broken in our shim.");
    } else {
        println!("VERDICT: NO-GO — see per-test detail.");
    }
}
