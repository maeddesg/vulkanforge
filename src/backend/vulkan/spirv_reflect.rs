//! Minimal SPIR-V reflection for compute pipelines.
//!
//! Walks the SPIR-V word stream once and extracts what the
//! [`crate::backend::vulkan::pipeline::ComputeKernel`] builder needs
//! to construct a matching `VkDescriptorSetLayout` and
//! `VkPipelineLayout`:
//!
//! - the unique `(set, binding, descriptor_type)` tuples for every
//!   resource variable that has both `DescriptorSet` and `Binding`
//!   decorations,
//! - the byte size of the push-constant block (max member offset +
//!   member size; 0 if the shader has no push constants),
//! - the IDs of every specialization constant (so callers know how
//!   many `VkSpecializationMapEntry` slots to wire up),
//! - the entry point's `LocalSize` execution mode (default
//!   workgroup size; will be overridden by spec constants when the
//!   shader uses `local_size_*_id`).
//!
//! No external dependencies — small enough to read in one sitting.
//! The parser tolerates ops it doesn't understand by stepping over
//! them via the instruction-length field, so it works on SPIR-V
//! produced by both glslang and shaderc.

use std::collections::HashMap;

use ash::vk;

#[derive(Debug, Clone)]
pub struct ReflectedShader {
    pub push_constant_size: u32,
    pub bindings: Vec<BindingInfo>,
    pub spec_constants: Vec<u32>,
    pub local_size: [u32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct BindingInfo {
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
}

const SPIRV_MAGIC: u32 = 0x07230203;

const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_VARIABLE: u16 = 59;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_EXECUTION_MODE: u16 = 16;

const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_SPEC_ID: u32 = 1;

const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

pub fn reflect(words: &[u32]) -> ReflectedShader {
    assert!(words.len() >= 5, "SPIR-V too short for header");
    assert_eq!(words[0], SPIRV_MAGIC, "SPIR-V magic missing");

    // OpDecorate / OpMemberDecorate aggregation.
    let mut decorations: HashMap<u32, Vec<(u32, Vec<u32>)>> = HashMap::new();
    let mut member_offsets: HashMap<(u32, u32), u32> = HashMap::new();
    let mut array_strides: HashMap<u32, u32> = HashMap::new();

    // Type system snapshots.
    let mut type_size: HashMap<u32, u32> = HashMap::new();
    let mut type_struct: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut type_pointer: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut type_array: HashMap<u32, (u32, u32)> = HashMap::new(); // id -> (element_type, length_const_id)
    let mut int_constants: HashMap<u32, u32> = HashMap::new(); // id -> u32 value (used for array lengths)
    // Variables and their declared (storage_class, pointer_type_id).
    let mut variables: HashMap<u32, (u32, u32)> = HashMap::new();

    let mut local_size = [1u32; 3];

    let mut i = 5;
    while i < words.len() {
        let header = words[i];
        let opcode = (header & 0xFFFF) as u16;
        let length = (header >> 16) as usize;
        if length == 0 || i + length > words.len() {
            // Malformed; bail out. spirv-val should have caught this.
            break;
        }
        let payload = &words[i + 1..i + length];

        match opcode {
            OP_DECORATE => {
                if payload.len() >= 2 {
                    let target = payload[0];
                    let decoration = payload[1];
                    if decoration == DECORATION_ARRAY_STRIDE && payload.len() >= 3 {
                        array_strides.insert(target, payload[2]);
                    }
                    let args = payload[2..].to_vec();
                    decorations.entry(target).or_default().push((decoration, args));
                }
            }
            OP_MEMBER_DECORATE => {
                if payload.len() >= 4 && payload[2] == DECORATION_OFFSET {
                    let struct_id = payload[0];
                    let member_idx = payload[1];
                    let offset = payload[3];
                    member_offsets.insert((struct_id, member_idx), offset);
                }
            }
            OP_TYPE_INT | OP_TYPE_FLOAT => {
                // (id, width, [signedness])
                if payload.len() >= 2 {
                    type_size.insert(payload[0], payload[1] / 8);
                }
            }
            OP_TYPE_VECTOR => {
                // (id, component_type, count)
                if payload.len() >= 3 {
                    if let Some(&csz) = type_size.get(&payload[1]) {
                        type_size.insert(payload[0], csz * payload[2]);
                    }
                }
            }
            OP_TYPE_MATRIX => {
                // (id, column_type, column_count)
                if payload.len() >= 3 {
                    if let Some(&csz) = type_size.get(&payload[1]) {
                        type_size.insert(payload[0], csz * payload[2]);
                    }
                }
            }
            OP_TYPE_ARRAY => {
                // (id, element_type, length_const_id)
                if payload.len() >= 3 {
                    type_array.insert(payload[0], (payload[1], payload[2]));
                }
            }
            OP_CONSTANT => {
                // (result_type_id, result_id, value [, value_high for 64-bit])
                if payload.len() >= 3 {
                    int_constants.insert(payload[1], payload[2]);
                }
            }
            OP_TYPE_STRUCT => {
                // (id, member_type_ids...)
                if !payload.is_empty() {
                    let id = payload[0];
                    let members = payload[1..].to_vec();
                    type_struct.insert(id, members);
                }
            }
            OP_TYPE_POINTER => {
                // (id, storage_class, target_type)
                if payload.len() >= 3 {
                    type_pointer.insert(payload[0], (payload[1], payload[2]));
                }
            }
            OP_VARIABLE => {
                // (result_type_id, result_id, storage_class, [initializer])
                if payload.len() >= 3 {
                    let type_id = payload[0];
                    let id = payload[1];
                    let sc = payload[2];
                    variables.insert(id, (type_id, sc));
                }
            }
            OP_EXECUTION_MODE => {
                // (entry_point_id, mode, args...)
                if payload.len() >= 5 && payload[1] == EXECUTION_MODE_LOCAL_SIZE {
                    local_size = [payload[2], payload[3], payload[4]];
                }
            }
            _ => {}
        }
        i += length;
    }

    // Push-constant block size: scan every PushConstant variable, walk
    // its pointer to the struct type, and recursively size members
    // (handles nested structs like rope_params and arrays like
    // `int sections[4]`).
    let mut push_constant_size = 0u32;
    for &(type_id, sc) in variables.values() {
        if sc != STORAGE_CLASS_PUSH_CONSTANT {
            continue;
        }
        let Some(&(_, struct_id)) = type_pointer.get(&type_id) else {
            continue;
        };
        let size = sized(struct_id, &type_size, &type_struct, &type_array, &int_constants, &array_strides, &member_offsets);
        push_constant_size = push_constant_size.max(size);
    }
    // Round up to the nearest multiple of 4 — Vulkan push constant
    // ranges must be 4-byte aligned.
    push_constant_size = (push_constant_size + 3) & !3;

    // Bindings: every variable whose storage class is StorageBuffer
    // (modern SPIR-V) or Uniform-with-BufferBlock (legacy SSBO) and
    // that carries DescriptorSet + Binding decorations.
    let mut bindings_map: HashMap<(u32, u32), vk::DescriptorType> = HashMap::new();
    for (&var_id, &(type_id, sc)) in &variables {
        if sc != STORAGE_CLASS_STORAGE_BUFFER && sc != STORAGE_CLASS_UNIFORM {
            continue;
        }
        let Some(decs) = decorations.get(&var_id) else {
            continue;
        };
        let mut set = None;
        let mut binding = None;
        for (dec, args) in decs {
            if *dec == DECORATION_DESCRIPTOR_SET && !args.is_empty() {
                set = Some(args[0]);
            }
            if *dec == DECORATION_BINDING && !args.is_empty() {
                binding = Some(args[0]);
            }
        }
        let (Some(s), Some(b)) = (set, binding) else {
            continue;
        };
        let descriptor_type = if sc == STORAGE_CLASS_STORAGE_BUFFER {
            vk::DescriptorType::STORAGE_BUFFER
        } else {
            // Uniform: distinguish UBO vs legacy SSBO by the struct's
            // BufferBlock decoration.
            let struct_id = type_pointer.get(&type_id).map(|p| p.1).unwrap_or(0);
            let is_ssbo = decorations
                .get(&struct_id)
                .map(|decs| decs.iter().any(|(d, _)| *d == DECORATION_BUFFER_BLOCK))
                .unwrap_or(false);
            if is_ssbo {
                vk::DescriptorType::STORAGE_BUFFER
            } else {
                vk::DescriptorType::UNIFORM_BUFFER
            }
        };
        // Multiple aliased SSBOs at the same binding (e.g. block_q4_K
        // / _packed16 / _packed32) collapse to a single descriptor
        // slot — keep the first descriptor type we see.
        bindings_map.entry((s, b)).or_insert(descriptor_type);
    }
    let mut bindings: Vec<BindingInfo> = bindings_map
        .into_iter()
        .map(|((s, b), t)| BindingInfo {
            set: s,
            binding: b,
            descriptor_type: t,
        })
        .collect();
    bindings.sort_by_key(|b| (b.set, b.binding));

    // Specialization-constant IDs, deduplicated and sorted.
    let mut spec_constants: Vec<u32> = decorations
        .values()
        .flat_map(|decs| {
            decs.iter().filter_map(|(d, args)| {
                if *d == DECORATION_SPEC_ID && !args.is_empty() {
                    Some(args[0])
                } else {
                    None
                }
            })
        })
        .collect();
    spec_constants.sort_unstable();
    spec_constants.dedup();

    ReflectedShader {
        push_constant_size,
        bindings,
        spec_constants,
        local_size,
    }
}

/// Recursive type-size walker covering primitives, vectors/matrices
/// (already in `type_size`), arrays (with explicit `ArrayStride` or
/// fallback element size × length), and structs (max member offset +
/// member size).
fn sized(
    type_id: u32,
    type_size: &HashMap<u32, u32>,
    type_struct: &HashMap<u32, Vec<u32>>,
    type_array: &HashMap<u32, (u32, u32)>,
    int_constants: &HashMap<u32, u32>,
    array_strides: &HashMap<u32, u32>,
    member_offsets: &HashMap<(u32, u32), u32>,
) -> u32 {
    if let Some(&size) = type_size.get(&type_id) {
        return size;
    }
    if let Some(members) = type_struct.get(&type_id) {
        let mut max_end = 0u32;
        for (idx, &mtype) in members.iter().enumerate() {
            let offset = *member_offsets
                .get(&(type_id, idx as u32))
                .unwrap_or(&0);
            let msize = sized(
                mtype,
                type_size,
                type_struct,
                type_array,
                int_constants,
                array_strides,
                member_offsets,
            );
            max_end = max_end.max(offset + msize);
        }
        return max_end;
    }
    if let Some(&(elem_ty, len_id)) = type_array.get(&type_id) {
        let length = int_constants.get(&len_id).copied().unwrap_or(0);
        let stride = array_strides.get(&type_id).copied().unwrap_or_else(|| {
            sized(
                elem_ty,
                type_size,
                type_struct,
                type_array,
                int_constants,
                array_strides,
                member_offsets,
            )
        });
        return length * stride;
    }
    0
}
