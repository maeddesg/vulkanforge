//! Phase 6A diagnostic — query
//! `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and print every
//! (M, N, K, AType, BType, CType, ResultType, scope) tuple the driver
//! advertises. This is the GO/NO-GO gate for using the AI accelerators
//! on RDNA4: if BF16×BF16→FP32 isn't here, the rest of Phase 6 is a
//! non-starter.
//!
//! Run with `cargo run --release --example probe_coopmat`.

use std::ffi::CStr;

use ash::{khr, vk};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load()? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"VulkanForge probe_coopmat")
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = physical_devices
        .iter()
        .copied()
        .find(|&pd| {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap_or(physical_devices[0]);

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    println!("Device: {}", name.to_string_lossy());
    println!(
        "API:    {}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version),
    );

    // ---- Verify the device advertises VK_KHR_cooperative_matrix. ----
    let device_exts =
        unsafe { instance.enumerate_device_extension_properties(physical_device)? };
    let has_coopmat = device_exts.iter().any(|e| {
        let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
        n == khr::cooperative_matrix::NAME
    });
    println!(
        "VK_KHR_cooperative_matrix:  {}",
        if has_coopmat { "PRESENT" } else { "ABSENT (NO-GO)" }
    );
    let has_bf16 = device_exts.iter().any(|e| {
        let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
        n.to_bytes() == b"VK_KHR_shader_bfloat16"
    });
    println!(
        "VK_KHR_shader_bfloat16:     {}",
        if has_bf16 { "PRESENT" } else { "ABSENT" }
    );

    if !has_coopmat {
        eprintln!("\nDriver does not advertise VK_KHR_cooperative_matrix on this device.");
        unsafe { instance.destroy_instance(None) };
        std::process::exit(2);
    }

    // ---- Query the property table. ----
    let cm = khr::cooperative_matrix::Instance::new(&entry, &instance);
    let entries = unsafe { cm.get_physical_device_cooperative_matrix_properties(physical_device)? };
    println!(
        "\nVK_KHR_cooperative_matrix properties: {} entry(ies)",
        entries.len()
    );
    println!(
        "{:<3}  {:>3} {:>3} {:>3}  {:<10} {:<10} {:<10} {:<10}  {:<10}  saturating",
        "#", "M", "N", "K", "AType", "BType", "CType", "ResultType", "scope"
    );
    println!("{}", "─".repeat(90));
    let mut has_bf16_to_f32 = false;
    let mut has_f16_to_f32 = false;
    let mut has_f16_to_f16 = false;
    let mut has_i8_to_i32 = false;
    for (i, p) in entries.iter().enumerate() {
        println!(
            "{:<3}  {:>3} {:>3} {:>3}  {:<10} {:<10} {:<10} {:<10}  {:<10}  {}",
            i,
            p.m_size,
            p.n_size,
            p.k_size,
            type_name(p.a_type),
            type_name(p.b_type),
            type_name(p.c_type),
            type_name(p.result_type),
            scope_name(p.scope),
            if p.saturating_accumulation == vk::TRUE { "yes" } else { "no" },
        );
        // Track headline combos for the GO/NO-GO summary.
        if is_bf16(p.a_type) && is_bf16(p.b_type)
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32 {
            has_bf16_to_f32 = true;
        }
        if p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32 {
            has_f16_to_f32 = true;
        }
        if p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT16
            && p.result_type == vk::ComponentTypeKHR::FLOAT16 {
            has_f16_to_f16 = true;
        }
        if p.a_type == vk::ComponentTypeKHR::SINT8
            && p.b_type == vk::ComponentTypeKHR::SINT8
            && p.c_type == vk::ComponentTypeKHR::SINT32
            && p.result_type == vk::ComponentTypeKHR::SINT32 {
            has_i8_to_i32 = true;
        }
    }

    println!("\nGO-gate summary:");
    println!(
        "  BF16 × BF16 → FP32 → FP32:  {}",
        if has_bf16_to_f32 { "YES (GO)" } else { "NO" }
    );
    println!(
        "  FP16 × FP16 → FP32 → FP32:  {}",
        if has_f16_to_f32 { "YES" } else { "NO" }
    );
    println!(
        "  FP16 × FP16 → FP16 → FP16:  {}",
        if has_f16_to_f16 { "YES" } else { "NO" }
    );
    println!(
        "  I8   × I8   → I32  → I32:   {}",
        if has_i8_to_i32 { "YES" } else { "NO" }
    );

    unsafe { instance.destroy_instance(None) };
    Ok(())
}

/// Vulkan ComponentType raw enum values from extensions ash
/// 0.38.0+1.3.281 doesn't ship constants for. BFLOAT16_KHR is from
/// `VK_KHR_shader_bfloat16` (added 2025); E4M3/E5M2 are from
/// `VK_NV_cooperative_matrix2` / shader-FP8 extensions. Confirmed
/// against the values RADV reports for gfx1201 (entries 17 / 19 of
/// the property table).
const COMPONENT_TYPE_BFLOAT16_KHR: i32 = 1000141000;
const COMPONENT_TYPE_FLOAT_E4M3_EXT: i32 = 1000491002;
const COMPONENT_TYPE_FLOAT_E5M2_EXT: i32 = 1000491003;
const COMPONENT_TYPE_SINT8_PACKED_NV: i32 = 1000491000;
const COMPONENT_TYPE_UINT8_PACKED_NV: i32 = 1000491001;

fn is_bf16(c: vk::ComponentTypeKHR) -> bool {
    c.as_raw() == COMPONENT_TYPE_BFLOAT16_KHR
}

fn type_name(c: vk::ComponentTypeKHR) -> String {
    match c {
        vk::ComponentTypeKHR::FLOAT16 => "FP16".into(),
        vk::ComponentTypeKHR::FLOAT32 => "FP32".into(),
        vk::ComponentTypeKHR::FLOAT64 => "FP64".into(),
        vk::ComponentTypeKHR::SINT8 => "I8".into(),
        vk::ComponentTypeKHR::SINT16 => "I16".into(),
        vk::ComponentTypeKHR::SINT32 => "I32".into(),
        vk::ComponentTypeKHR::SINT64 => "I64".into(),
        vk::ComponentTypeKHR::UINT8 => "U8".into(),
        vk::ComponentTypeKHR::UINT16 => "U16".into(),
        vk::ComponentTypeKHR::UINT32 => "U32".into(),
        vk::ComponentTypeKHR::UINT64 => "U64".into(),
        _ => match c.as_raw() {
            COMPONENT_TYPE_BFLOAT16_KHR => "BF16".into(),
            COMPONENT_TYPE_FLOAT_E4M3_EXT => "E4M3".into(),
            COMPONENT_TYPE_FLOAT_E5M2_EXT => "E5M2".into(),
            COMPONENT_TYPE_SINT8_PACKED_NV => "I8x4".into(),
            COMPONENT_TYPE_UINT8_PACKED_NV => "U8x4".into(),
            x => format!("?{x}"),
        },
    }
}

fn scope_name(s: vk::ScopeKHR) -> String {
    match s {
        vk::ScopeKHR::DEVICE => "Device".into(),
        vk::ScopeKHR::WORKGROUP => "Workgroup".into(),
        vk::ScopeKHR::SUBGROUP => "Subgroup".into(),
        vk::ScopeKHR::QUEUE_FAMILY => "QueueFamily".into(),
        x => format!("?{}", x.as_raw()),
    }
}
