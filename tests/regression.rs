//! Phase-1 + Phase-2A regression tests. Run with `cargo test`.
//!
//! These talk to a real Vulkan device; each test creates its own
//! `VulkanDevice` so they don't share state and can be run in any
//! order (or in parallel).

use std::path::PathBuf;
use std::time::Duration;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use vulkanforge::backend::vulkan::buffers::GpuBuffer;
use vulkanforge::backend::vulkan::chat::ChatSession;
use vulkanforge::backend::vulkan::decode::embedding_row;
use vulkanforge::backend::vulkan::commands::CommandContext;
use vulkanforge::backend::vulkan::decode::{generate, GenerateConfig, ThinkFilter};
use vulkanforge::backend::vulkan::device::VulkanDevice;
use vulkanforge::backend::vulkan::forward::Forward;
use vulkanforge::backend::vulkan::gguf::{GgmlType, GgufFile, ModelConfig};
use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
use vulkanforge::backend::vulkan::loader::LoadedModel;
use vulkanforge::backend::vulkan::pipeline::{MatVecPushConstants, PUSH_CONSTANT_BYTES};
use vulkanforge::backend::vulkan::q4k as q4k_module;
use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;
use vulkanforge::backend::vulkan::q4k;
use vulkanforge::backend::vulkan::shaders::{self, ShaderId, ALL_SHADERS};
use vulkanforge::backend::vulkan::spirv_reflect;
use vulkanforge::backend::vulkan::tokenizer::{apply_chat_template, Tokenizer};
use vulkanforge::backend::vulkan::vram_arena::{
    ArenaConfig, ArenaError, BufferViewError, VramArena,
};

const SPIRV_MAGIC: [u8; 4] = [0x03, 0x02, 0x23, 0x07];

#[test]
fn phase2a_all_shaders_compile_to_spirv() {
    // Cheap sanity: every embedded blob is a valid SPIR-V binary in
    // structure (magic number + 4-byte aligned). Catches a build.rs
    // regression that would otherwise only surface as a runtime
    // pipeline-create error.
    for &id in ALL_SHADERS {
        let bytes = id.spv_bytes();
        assert!(
            bytes.len() >= 5 * 4,
            "{:?}: SPIR-V too short ({})",
            id,
            bytes.len()
        );
        assert_eq!(
            &bytes[0..4],
            &SPIRV_MAGIC,
            "{:?}: SPIR-V magic missing",
            id
        );
        assert_eq!(bytes.len() % 4, 0, "{:?}: SPIR-V not 4-byte aligned", id);

        // Reflection must succeed and report at least one binding for
        // every shader in the inventory (none of them are "empty").
        let reflection = spirv_reflect::reflect(&shaders::spv_words(bytes));
        assert!(
            !reflection.bindings.is_empty(),
            "{:?}: reflection found 0 bindings",
            id
        );
    }
}

#[test]
fn phase2a_q4k_unit_tests_referenced() {
    // Phase-1 unit tests live in src/backend/vulkan/q4k.rs and run
    // automatically with `cargo test`. This integration test just
    // verifies the data-generation entry points still work and the
    // CPU GEMV produces the analytical answer — protecting the
    // pair-layout bug (Q4_K nibble-bug) from regressing.
    let weights = q4k::build_smoke_weights();
    let input: Vec<f32> = vec![1.0; q4k::QUANT_K];
    let out = q4k::cpu_gemv(&weights, 2, q4k::QUANT_K, &input);
    assert!((out[0] - 256.0).abs() < 1e-3);
    assert!((out[1] - 512.0).abs() < 1e-3);
}

#[test]
fn phase2a_pipeline_registry_creates_all() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let cache_path: Option<PathBuf> = None; // never touch user cache from tests
    let (registry, loaded) = PipelineRegistry::new(&dev.device, cache_path.as_deref())
        .expect("PipelineRegistry::new");
    assert_eq!(loaded, 0, "no cache path → no bytes loaded");
    assert_eq!(
        registry.count(),
        ALL_SHADERS.len(),
        "registry must hold one pipeline per ShaderId"
    );
    // Every ShaderId must be retrievable.
    for &id in ALL_SHADERS {
        let _ = registry.get(id);
    }
    // Sanity: cold-start pipeline creation should be fast (< 5 s).
    // This catches a hang in vkCreateComputePipelines.
    assert!(
        registry.create_duration < Duration::from_secs(5),
        "pipeline creation took {:?}",
        registry.create_duration
    );
    registry.destroy(&dev.device);
}

#[test]
fn phase2a_vram_arena_zones_and_pingpong() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let config = ArenaConfig {
        weights_bytes: 4 * 1024 * 1024,
        kv_cache_bytes: 2 * 1024 * 1024,
        scratch_bytes: 256 * 1024,
    };
    let arena = VramArena::new(&dev.instance, dev.physical_device, &dev.device, config)
        .expect("VramArena::new");
    let l = arena.layout;
    // No overlap, no gaps beyond the 4 KiB zone alignment.
    assert_eq!(l.weights.offset, 0);
    assert_eq!(l.kv_cache.offset, l.weights.size);
    assert_eq!(l.scratch.offset, l.weights.size + l.kv_cache.size);
    assert!(arena.total_bytes >= l.scratch.offset + l.scratch.size);
    assert!(l.weights.size >= config.weights_bytes);
    assert!(l.kv_cache.size >= config.kv_cache_bytes);
    assert!(l.scratch.size >= config.scratch_bytes);

    // Ping-pong: even/odd layers map to the two halves of scratch.
    let (off0, size0) = arena.scratch_for_layer(0);
    let (off1, size1) = arena.scratch_for_layer(1);
    let (off2, _) = arena.scratch_for_layer(2);
    assert_eq!(size0, size1);
    assert_eq!(size0, l.scratch.size / 2);
    assert_ne!(off0, off1, "ping-pong halves must differ");
    assert_eq!(off0, off2, "even layers reuse the same half");
    assert!(off0 + size0 <= l.scratch.offset + l.scratch.size);
    assert!(off1 + size1 <= l.scratch.offset + l.scratch.size);

    // Buffer views in each zone should bind cleanly.
    let weights_view = arena
        .create_buffer(
            &dev.device,
            l.weights.offset,
            128 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in weights zone");
    let kv_view = arena
        .create_buffer(
            &dev.device,
            l.kv_cache.offset,
            64 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in kv zone");
    let scratch_view = arena
        .create_buffer(
            &dev.device,
            off0,
            size0,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .expect("create_buffer in scratch zone");

    // Out-of-bounds is rejected with a structured error.
    let oob = arena.create_buffer(
        &dev.device,
        arena.total_bytes - 16,
        128,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    );
    assert!(matches!(oob, Err(BufferViewError::OutOfBounds { .. })));

    unsafe {
        dev.device.destroy_buffer(weights_view, None);
        dev.device.destroy_buffer(kv_view, None);
        dev.device.destroy_buffer(scratch_view, None);
    }
    arena.destroy(&dev.device);
}

#[test]
fn phase2a_vram_arena_oom_clean_error() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    // Request more than maxMemoryAllocationSize ever could be.
    let huge = ArenaConfig {
        weights_bytes: u64::MAX / 2,
        kv_cache_bytes: u64::MAX / 4,
        scratch_bytes: 0,
    };
    let result = VramArena::new(&dev.instance, dev.physical_device, &dev.device, huge);
    assert!(
        matches!(result, Err(ArenaError::AllocationTooLarge { .. })),
        "expected AllocationTooLarge, got {:?}",
        result.as_ref().err()
    );
}

/// Phase-2B regression — the parser itself, no GPU. Skipped silently
/// if the model file isn't present so CI machines don't fail.
#[test]
fn phase2b_gguf_parses_qwen3_8b() {
    let Some(path) = qwen3_path() else { return };
    let gguf = GgufFile::open(&path).expect("GgufFile::open");
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.tensor_count, 399);
    let cfg = ModelConfig::from_gguf(&gguf).expect("ModelConfig::from_gguf");

    assert_eq!(cfg.architecture, "qwen3");
    assert_eq!(cfg.n_layers, 36);
    assert_eq!(cfg.n_heads, 32);
    assert_eq!(cfg.n_kv_heads, 8);
    assert_eq!(cfg.hidden_dim, 4096);
    assert_eq!(cfg.ffn_dim, 12288);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.context_length, 40960);
    assert!((cfg.rope_freq_base - 1_000_000.0).abs() < 1e-3);
    assert!(cfg.has_qk_norm, "Qwen3 should expose attn_q_norm/attn_k_norm");

    // Spot check a few expected tensors and their dtypes.
    let q = gguf.tensor("blk.0.attn_q.weight").expect("attn_q present");
    assert_eq!(q.ggml_type, GgmlType::Q4K);
    assert_eq!(q.dimensions, vec![4096, 4096]);

    let down = gguf.tensor("blk.0.ffn_down.weight").expect("ffn_down present");
    assert_eq!(down.ggml_type, GgmlType::Q6K);
}

/// Phase-2B regression — full VRAM upload of the Qwen3-8B GGUF.
/// Skipped silently when the model isn't on disk; ~5 s on RX 9070 XT.
#[test]
fn phase2b_qwen3_loads_to_vram() {
    let Some(path) = qwen3_path() else { return };
    let gguf = GgufFile::open(&path).expect("open");
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");

    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("LoadedModel::load");
    assert_eq!(model.tensors.len(), gguf.tensors.len());
    // 4.68 GiB ± a bit — exact byte total can shift across builds; assert >= 4 GiB.
    assert!(
        model.bytes_uploaded >= 4 * 1024 * 1024 * 1024,
        "uploaded only {} bytes",
        model.bytes_uploaded
    );
    // A quick sanity buffer-handle check on a tensor we know exists.
    let q = model.tensor("blk.0.attn_q.weight").expect("missing attn_q");
    assert!(q.byte_size > 0);
    assert_ne!(q.buffer.handle, vk::Buffer::null());

    model.destroy(&dev.device, &mut allocator);
    drop(allocator);
}

/// Phase-2C regression — full forward pass through Qwen3-8B with a
/// real Q4_K embedding row produces finite logits with a non-trivial
/// distribution. Skipped silently when the model file isn't present.
#[test]
fn phase2c_forward_token_qwen3_finite_logits() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = vulkanforge::backend::vulkan::commands::CommandContext::new(
        &dev.device, dev.queue_family_index,
    )
    .expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("config");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");

    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: 512, // small to keep allocator pressure low
        },
    )
    .expect("kv_cache");

    let mut forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None).expect("forward");

    // CPU dequant the token-0 row of token_embd.weight.
    let info = gguf.tensor("token_embd.weight").expect("token_embd");
    let blocks_per_row = (cfg.hidden_dim as usize) / q4k_module::QUANT_K;
    let row_bytes = blocks_per_row * q4k_module::BLOCK_BYTES;
    let bytes = gguf.tensor_bytes(info);
    let token_id: u32 = 9707;
    let row_off = (token_id as usize) * row_bytes;
    let mut embd = Vec::with_capacity(cfg.hidden_dim as usize);
    for b in 0..blocks_per_row {
        let blk_off = row_off + b * q4k_module::BLOCK_BYTES;
        let block: &[u8; q4k_module::BLOCK_BYTES] = (&bytes[blk_off..blk_off + q4k_module::BLOCK_BYTES])
            .try_into().unwrap();
        embd.extend_from_slice(&q4k_module::dequant_block(block));
    }

    forward
        .forward_token(&dev, &registry, &cmd_ctx, &model, &embd, 0)
        .expect("forward_token");
    let logits = forward.logits().expect("logits");

    assert_eq!(logits.len(), cfg.vocab_size as usize);
    let any_nan = logits.iter().any(|v| !v.is_finite());
    assert!(!any_nan, "logits contain NaN/Inf");
    let all_zero = logits.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "logits are all zero");
    let max_abs = logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.1, "logits look saturated to ~0 (max_abs={max_abs})");

    // Top-1 should be a real token id, not garbage.
    let top1 = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert!(
        top1.0 < cfg.vocab_size as usize,
        "top-1 token id out of range: {}",
        top1.0
    );

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

fn qwen3_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VF_MODEL_PATH").map(PathBuf::from) {
        return Some(p);
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models").join("Qwen3-8B-Q4_K_M.gguf");
    if p.exists() { Some(p) } else { None }
}

// -----------------------------------------------------------------
// Phase 2D — tokenizer + decode loop regression tests.

/// Tokenizer roundtrips simple ASCII strings exactly. Encodes through
/// the Qwen2 pre-tokenizer + GPT-2 byte-level BPE, decodes back.
#[test]
fn phase2d_tokenizer_roundtrip() {
    let Some(path) = qwen3_path() else { return };
    let gguf = GgufFile::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    assert_eq!(tok.vocab_size(), 151936);
    for s in [
        "Hello world",
        "Hello world!",
        "Explain what a mutex is.",
        " leading space",
        "Two\nlines",
    ] {
        let ids = tok.encode(s);
        let back = tok.decode(&ids);
        assert_eq!(back, s, "roundtrip mismatch for {s:?}: got {back:?}");
    }
    // Known Qwen3 tokenisation: "Hello" → 9707, " world" → 1879.
    let ids = tok.encode("Hello world");
    assert_eq!(ids, vec![9707, 1879]);
}

/// Chat template emits `<|im_start|>` / `<|im_end|>` as single ids,
/// places them where the spec calls for, and renders back to the
/// expected Qwen3 prompt string.
#[test]
fn phase2d_chat_template_qwen3() {
    let Some(path) = qwen3_path() else { return };
    let gguf = GgufFile::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let prompt = apply_chat_template(&tok, "Hi", None);
    let im_start = tok.im_start_id.expect("Qwen3 tokenizer has <|im_start|>");
    let im_end = tok.im_end_id.expect("Qwen3 tokenizer has <|im_end|>");
    // Must start with <|im_start|>system and end with <|im_start|>assistant\n.
    assert_eq!(prompt[0], im_start);
    assert!(
        prompt.iter().filter(|&&id| id == im_start).count() == 3,
        "expected 3 <|im_start|> in chat template, got {prompt:?}",
    );
    assert!(
        prompt.iter().filter(|&&id| id == im_end).count() == 2,
        "expected 2 <|im_end|> in chat template",
    );
    let rendered = tok.decode(&prompt);
    assert!(
        rendered.contains("<|im_start|>system\n")
            && rendered.contains("<|im_start|>user\nHi<|im_end|>\n")
            && rendered.ends_with("<|im_start|>assistant\n"),
        "chat-template renders unexpectedly:\n{rendered}",
    );
}

/// End-to-end decode: greedy generation on "Explain what a mutex is in
/// one sentence." must produce coherent English containing one of the
/// expected mutex-related keywords. Catches future regressions that
/// flip from "looks plausible" to "Garbage".
#[test]
fn phase2d_decode_produces_coherent_text() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("config");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            // Generous enough for a 30-token prompt + 80 generated tokens.
            max_seq_len: 256,
        },
    )
    .expect("kv_cache");
    let mut forward =
        Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None).expect("forward");

    let result = generate(
        &mut forward,
        &dev,
        &registry,
        &cmd_ctx,
        &model,
        &gguf,
        &cfg,
        &tokenizer,
        "Explain what a mutex is in one sentence.",
        &GenerateConfig {
            max_tokens: 80,
            print_stream: false,
            think_filter: false, sampling: Default::default(),
        },
    )
    .expect("generate");

    assert!(result.generated_tokens > 0, "no tokens generated");
    let text_lower = result.generated_text.to_lowercase();
    let keywords = ["mutex", "mutual", "lock", "thread", "synchron", "exclus"];
    assert!(
        keywords.iter().any(|k| text_lower.contains(k)),
        "decoded text contains none of {keywords:?}:\n{}",
        result.generated_text,
    );

    forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

// -----------------------------------------------------------------
// Phase 3B — ChatSession (multi-turn) + ThinkFilter regression tests.

/// `ThinkFilter::strip_all` is the unit-level check; this test
/// exercises the same filter applied to a real generated stream that
/// contains a complete `<think>...</think>` block. Cheap (no GPU).
#[test]
fn phase3b_think_filter_strips_real_text() {
    let raw = "<think>\nOkay, the user asked about mutexes.\n</think>\n\nA mutex is a synchronization primitive.";
    let visible = ThinkFilter::strip_all(raw);
    assert!(
        !visible.contains("<think>") && !visible.contains("</think>"),
        "filtered text still has tags: {visible:?}"
    );
    assert!(
        !visible.contains("Okay, the user asked"),
        "filtered text leaked think content: {visible:?}"
    );
    assert!(
        visible.contains("synchronization primitive"),
        "filtered text dropped the answer: {visible:?}"
    );
}

/// Multi-Turn carries context across turns. Tells the assistant a
/// fact in Turn 1, asks for it in Turn 2, then `/reset`s and verifies
/// the fact is gone.
///
/// Greedy decode is deterministic, so the model's first response
/// fixes the second-turn context exactly. Asking "What is my name?"
/// after telling it "My name is Alice" must produce text containing
/// "Alice"; after `/reset` it must not.
#[test]
fn phase3b_chat_session_multi_turn_carries_and_resets() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("config");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            // 2 turns × ≈70 tokens each fits comfortably in 768.
            max_seq_len: 768,
        },
    )
    .expect("kv_cache");
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None).expect("forward");
    let mut session = ChatSession::new(forward, "You are a helpful assistant.");
    let cfg_g = GenerateConfig {
        max_tokens: 80,
        print_stream: false,
        // Filter <think> chatter out of the *visible* text we assert on
        // — the model frequently muses inside <think>; we want the
        // post-think answer.
        think_filter: true, sampling: Default::default(),
    };

    // Turn 1: tell it the user's name.
    let _r1 = session
        .send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            "My name is Alice. Please remember it.",
            &cfg_g,
        )
        .expect("turn 1");
    assert_eq!(session.turn_count, 1, "turn count after first send");
    assert!(session.current_pos > 0, "current_pos should advance after turn 1");

    // Turn 2: ask for it back.
    let r2 = session
        .send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            "What is my name?",
            &cfg_g,
        )
        .expect("turn 2");
    assert_eq!(session.turn_count, 2, "turn count after second send");
    let visible_lower = r2.visible_text.to_lowercase();
    let raw_lower = r2.generated_text.to_lowercase();
    assert!(
        visible_lower.contains("alice") || raw_lower.contains("alice"),
        "Multi-turn lost context — Turn 2 answer:\n  visible: {:?}\n  raw:     {:?}",
        r2.visible_text, r2.generated_text,
    );

    // /reset wipes both KV state and history.
    session.reset();
    assert_eq!(session.current_pos, 0);
    assert_eq!(session.turn_count, 0);
    assert!(session.history.is_empty());

    // Turn 3 (post-reset): asking the same question should NOT find
    // "alice" because the session has no memory of it.
    let r3 = session
        .send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            "What is my name?",
            &cfg_g,
        )
        .expect("turn 3");
    let post_reset_lower = r3.visible_text.to_lowercase();
    assert!(
        !post_reset_lower.contains("alice"),
        "post-reset answer leaked Turn 1's context — answer was {:?}",
        r3.visible_text,
    );

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

/// Phase 3E — `prefill_batch` (GEMM path) vs token-by-token (GEMV path)
/// must produce equivalent logits for the same prompt. Equivalence
/// is checked at two levels:
/// * `argmax(logits)` must agree (the actual sampled token id)
/// * `top-5 ids` must overlap by at least 4
///
/// We don't assert tight numerical equality — Q8_1 activations + f16
/// scale storage accumulate ~1% per layer × 36 layers, enough to shift
/// distant tokens. Top-5 stability is the practical guarantee.
#[test]
fn phase3e_prefill_batch_matches_token_by_token_top5() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("dev");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    }).expect("alloc");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tok");

    // Use the same prompt the Phase-2D coherence test uses.
    let mut prompt_tokens =
        vulkanforge::backend::vulkan::tokenizer::apply_chat_template(
            &tokenizer, "Explain what a mutex is in one sentence.", None,
        );
    // Trim to fit the small KV cache below.
    if prompt_tokens.len() > 64 { prompt_tokens.truncate(64); }
    let seq_len = prompt_tokens.len() as u32;

    // -- Path A: token-by-token (forward_token) --
    let kv_a = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 256,
    }).expect("kv_a");
    let mut fwd_a = Forward::new_with_prefill(
        &dev, &mut allocator, kv_a, cfg.clone(), None, /* max_pp = */ 1,
    ).expect("forward_a");
    fwd_a.kv_cache.reset();
    for (pos, &tid) in prompt_tokens.iter().enumerate() {
        let embd = embedding_row(&gguf, &cfg, tid).expect("embd");
        fwd_a.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos as u32)
            .expect("fwd_token");
    }
    let logits_a = fwd_a.logits().expect("logits_a");
    fwd_a.destroy(&dev.device, &mut allocator);

    // -- Path B: prefill_batch (GEMM) --
    let kv_b = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 256,
    }).expect("kv_b");
    let mut fwd_b = Forward::new_with_prefill(
        &dev, &mut allocator, kv_b, cfg.clone(), None, /* max_pp = */ seq_len,
    ).expect("forward_b");
    fwd_b.kv_cache.reset();
    let mut all_embeds: Vec<f32> = Vec::new();
    for &tid in &prompt_tokens {
        all_embeds.extend(embedding_row(&gguf, &cfg, tid).expect("embd"));
    }
    fwd_b.prefill_batch(
        &dev, &registry, &cmd_ctx, &model, &all_embeds, seq_len, 0,
    ).expect("prefill_batch");
    let logits_b = fwd_b.logits().expect("logits_b");
    fwd_b.destroy(&dev.device, &mut allocator);

    assert_eq!(logits_a.len(), logits_b.len());
    let nan_b = logits_b.iter().any(|v| !v.is_finite());
    assert!(!nan_b, "prefill_batch produced NaN/Inf logits");

    // Top-1 (argmax) — practical "did we sample the same token?".
    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let top1_a = argmax(&logits_a);
    let top1_b = argmax(&logits_b);
    let top_k = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let top5_a = top_k(&logits_a, 5);
    let top5_b = top_k(&logits_b, 5);
    let overlap = top5_a.iter().filter(|t| top5_b.contains(t)).count();
    eprintln!(
        "[parity] top1_a={} top1_b={} top5_a={:?} top5_b={:?} overlap={}",
        top1_a, top1_b, top5_a, top5_b, overlap,
    );
    // Greedy-decode practical gate: top-1 must match. The Q8_1
    // activations + f16 scale storage drift the rest of the top-5
    // ordering — ~1% of logit values shifts each layer × 36 layers
    // is enough to swap distant ranks but not the top-1 (when the
    // top-1 has a non-trivial logit gap, which it does on the
    // mutex-prompt: 151667 wins by a wide margin in both paths).
    assert_eq!(
        top1_a, top1_b,
        "argmax differs: token-by-token={} prefill_batch={}\n\
         Top-5 a: {:?}\n\
         Top-5 b: {:?}",
        top1_a, top1_b, top5_a, top5_b,
    );
    // Phase-3E-drift-fix tightened gate: with the per-token RoPE
    // position bug fixed (each token now reads its own slot in
    // rope_pos_buf instead of all collapsing to the last host write),
    // top-5 overlap should be ≥ 4/5. Pre-fix it was 1/5 — path B's
    // top-5 was dominated by structural tokens (<|im_start|>,
    // <|im_end|>, …) because every position decoded as the last one.
    assert!(
        overlap >= 4,
        "Top-5 overlap {}/5 too low — RoPE drift bug regressed?\n  A: {:?}\n  B: {:?}",
        overlap, top5_a, top5_b,
    );

    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

/// Sprint 3A — `gemm_q` through coopmat must produce the same top-1
/// token (and a stable top-5) as the mul_mmq baseline. Both Forward
/// instances run in the same process with the rest of the prefill
/// path identical; only `coopmat_q4k_enabled` differs.
#[test]
fn sprint3a_coopmat_gemm_q_logits_parity() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("dev");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    }).expect("alloc");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tok");

    let mut prompt_tokens =
        vulkanforge::backend::vulkan::tokenizer::apply_chat_template(
            &tokenizer, "Explain what a mutex is in one sentence.", None,
        );
    if prompt_tokens.len() > 64 { prompt_tokens.truncate(64); }
    let seq_len = prompt_tokens.len() as u32;
    let mut all_embeds: Vec<f32> = Vec::new();
    for &tid in &prompt_tokens {
        all_embeds.extend(embedding_row(&gguf, &cfg, tid).expect("embd"));
    }

    let make_forward = |allocator: &mut Allocator, coopmat: bool| -> Forward {
        let kv = KvCache::new(&dev.device, allocator, KvCacheConfig {
            n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim, max_seq_len: 256,
        }).expect("kv");
        let mut fwd = Forward::new_with_prefill(
            &dev, allocator, kv, cfg.clone(), None, seq_len,
        ).expect("forward");
        fwd.kv_cache.reset();
        fwd.set_coopmat_q4k_enabled(coopmat);
        fwd
    };

    // Path A: mul_mmq baseline.
    let mut fwd_a = make_forward(&mut allocator, false);
    fwd_a.prefill_batch(&dev, &registry, &cmd_ctx, &model, &all_embeds, seq_len, 0)
        .expect("prefill mmq");
    let logits_a = fwd_a.logits().expect("logits_a");
    fwd_a.destroy(&dev.device, &mut allocator);

    // Path B: gemm_q via coopmat (rest stays mul_mmq).
    let mut fwd_b = make_forward(&mut allocator, true);
    fwd_b.prefill_batch(&dev, &registry, &cmd_ctx, &model, &all_embeds, seq_len, 0)
        .expect("prefill coopmat");
    let logits_b = fwd_b.logits().expect("logits_b");
    fwd_b.destroy(&dev.device, &mut allocator);

    assert_eq!(logits_a.len(), logits_b.len());
    let nan_b = logits_b.iter().any(|v| !v.is_finite());
    assert!(!nan_b, "coopmat gemm_q produced NaN/Inf logits");

    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let top1_a = argmax(&logits_a);
    let top1_b = argmax(&logits_b);
    let top_k = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let top5_a = top_k(&logits_a, 5);
    let top5_b = top_k(&logits_b, 5);
    let overlap5 = top5_a.iter().filter(|t| top5_b.contains(t)).count();

    // Rank of the mul_mmq top-1 token in the coopmat logits — measures
    // how much signal survives the FP8 narrowing across 36 layers.
    let mut sorted_b: Vec<(usize, f32)> = logits_b
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    sorted_b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mmq_top1_rank_in_coopmat = sorted_b.iter().position(|(i, _)| *i == top1_a).unwrap_or(usize::MAX);

    eprintln!(
        "[sprint3-parity] top1_mmq={} top1_coopmat={} top5_mmq={:?} top5_coopmat={:?} top5_overlap={} mmq_top1_rank_in_coopmat={}",
        top1_a, top1_b, top5_a, top5_b, overlap5, mmq_top1_rank_in_coopmat,
    );
    // Sprint 3B gate (after the BF16 pivot + partial-tile-store fix
    // landed): top-1 must agree, and top-5 overlap >= 3/5. Sprint 3A's
    // FP8 attempt failed both checks; the failure trace was
    //   top1_mmq=151667 top1_coopmat=13 overlap=0 rank=12322
    // After Sprint 3B the same prompt yields
    //   top1=151667/151667 overlap=4/5 rank=0
    assert_eq!(
        top1_a, top1_b,
        "coopmat gemm_q changed top-1 token: mmq={} coopmat={}",
        top1_a, top1_b,
    );
    assert!(
        overlap5 >= 3,
        "top-5 overlap {}/5 too low\n  mmq:     {:?}\n  coopmat: {:?}",
        overlap5, top5_a, top5_b,
    );
    let _ = mmq_top1_rank_in_coopmat;

    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

/// Context-overflow surfaces as a structured `ChatError::ContextOverflow`,
/// not a panic.
#[test]
fn phase3b_chat_session_context_overflow_clean_error() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("config");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    // Tiny KV cache so a single send overflows immediately.
    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: 64,
        },
    )
    .expect("kv_cache");
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None).expect("forward");
    let mut session = ChatSession::new(forward, "You are a helpful assistant.");

    let err = session
        .send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            // The chat template alone is ~25 tokens; with max_tokens=80
            // we ask for ~110, well over max_seq_len=64.
            "Tell me everything you know about distributed systems.",
            &GenerateConfig { max_tokens: 80, print_stream: false, think_filter: false, sampling: Default::default() },
        )
        .expect_err("expected ChatError::ContextOverflow");
    let msg = format!("{err}");
    assert!(
        msg.contains("context overflow"),
        "expected context overflow error, got: {msg}"
    );

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

/// Phase-1 regression: the original 1.4 smoke test. Bit-exact output
/// from the Q4_K GEMV pipeline → [256.0, 512.0]. This is the test the
/// prompt's regression strategy explicitly calls out as `BEHALTEN`.
#[test]
fn phase1_q4k_smoke_dispatch_bit_exact() {
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");

    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("PipelineRegistry::new");
    let kernel = registry.get(ShaderId::MulMatVecQ4K);

    const M: usize = 2;
    const K: usize = q4k::QUANT_K;

    let weights_bytes = q4k::build_smoke_weights();
    let input: Vec<f32> = vec![1.0; K];
    let input_bytes_slice: &[u8] = bytemuck::cast_slice(&input);

    let weights_size = weights_bytes.len() as u64;
    let input_size = input_bytes_slice.len() as u64;
    let output_size = (M * std::mem::size_of::<f32>()) as u64;
    let dummy_size: u64 = 16;

    let storage_dst =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let storage_only = vk::BufferUsageFlags::STORAGE_BUFFER;
    let staging_src = vk::BufferUsageFlags::TRANSFER_SRC;

    let weights_buf = GpuBuffer::new(&dev.device, &mut allocator, weights_size, storage_dst, MemoryLocation::GpuOnly, "weights").unwrap();
    let input_buf = GpuBuffer::new(&dev.device, &mut allocator, input_size, storage_dst, MemoryLocation::GpuOnly, "input").unwrap();
    let output_buf = GpuBuffer::new(&dev.device, &mut allocator, output_size, storage_only, MemoryLocation::GpuToCpu, "output").unwrap();
    let fuse0 = GpuBuffer::new(&dev.device, &mut allocator, dummy_size, storage_only, MemoryLocation::GpuOnly, "fuse0").unwrap();
    let fuse1 = GpuBuffer::new(&dev.device, &mut allocator, dummy_size, storage_only, MemoryLocation::GpuOnly, "fuse1").unwrap();

    let mut staging_w = GpuBuffer::new(&dev.device, &mut allocator, weights_size, staging_src, MemoryLocation::CpuToGpu, "staging_w").unwrap();
    let mut staging_i = GpuBuffer::new(&dev.device, &mut allocator, input_size, staging_src, MemoryLocation::CpuToGpu, "staging_i").unwrap();
    staging_w.write_bytes(&weights_bytes).unwrap();
    staging_i.write_bytes(input_bytes_slice).unwrap();

    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).unwrap();
    cmd_ctx
        .one_shot(&dev.device, dev.compute_queue, |cmd| {
            let copy_w = vk::BufferCopy::default().size(weights_size);
            let copy_i = vk::BufferCopy::default().size(input_size);
            unsafe {
                dev.device.cmd_copy_buffer(cmd, staging_w.handle, weights_buf.handle, std::slice::from_ref(&copy_w));
                dev.device.cmd_copy_buffer(cmd, staging_i.handle, input_buf.handle, std::slice::from_ref(&copy_i));
            }
        })
        .unwrap();

    // Descriptor pool + set + writes.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
    let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None) }.unwrap();
    let layouts = [kernel.descriptor_set_layout];
    let alloc = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts);
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
        ncols: K as u32, stride_a: K as u32, stride_b: K as u32, stride_d: M as u32,
        batch_stride_a: (K * M) as u32, batch_stride_b: K as u32, batch_stride_d: M as u32,
        fusion_flags: 0, base_work_group_y: 0, ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1,
    };
    let pc_bytes: &[u8] = bytemuck::bytes_of(&pc);
    assert_eq!(pc_bytes.len(), PUSH_CONSTANT_BYTES as usize);

    cmd_ctx
        .one_shot(&dev.device, dev.compute_queue, |cmd| unsafe {
            let pre = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(weights_buf.handle).offset(0).size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(input_buf.handle).offset(0).size(vk::WHOLE_SIZE),
            ];
            dev.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[], &pre, &[],
            );
            dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline_layout, 0, &[set], &[]);
            dev.device.cmd_push_constants(cmd, kernel.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            dev.device.cmd_dispatch(cmd, 2, 1, 1);
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output_buf.handle).offset(0).size(vk::WHOLE_SIZE);
            dev.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[], &[post], &[],
            );
        })
        .unwrap();

    let output_bytes = output_buf.read_bytes().unwrap();
    let g0 = f32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
    let g1 = f32::from_le_bytes(output_bytes[4..8].try_into().unwrap());
    assert_eq!(g0, 256.0, "Phase-1 smoke regression: output[0]");
    assert_eq!(g1, 512.0, "Phase-1 smoke regression: output[1]");

    // Cleanup
    unsafe { dev.device.destroy_descriptor_pool(pool, None) };
    staging_i.destroy(&dev.device, &mut allocator);
    staging_w.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    fuse1.destroy(&dev.device, &mut allocator);
    fuse0.destroy(&dev.device, &mut allocator);
    output_buf.destroy(&dev.device, &mut allocator);
    input_buf.destroy(&dev.device, &mut allocator);
    weights_buf.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

/// Phase 5A-2 Stage 2D — descriptor-set cache parity.
///
/// Two `Forward` instances run the same 16-token sequence on Qwen3-8B:
/// one with `cache_enabled=false` (Direct-Path), one with
/// `cache_enabled=true` (CB-Reuse). Per-position logits must agree
/// within `< 1e-6` max abs error, and the argmax must be identical
/// at every step.
#[test]
fn phase5a_cb_reuse_parity_qwen3() {
    let Some(path) = qwen3_path() else { return };
    let dev = VulkanDevice::new().expect("VulkanDevice::new");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("Allocator::new");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd_ctx");
    let gguf = GgufFile::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&gguf).expect("config");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("load");

    // Use a varied token sequence so we exercise different embedding
    // values and different attention positions.
    let token_seq: Vec<u32> = (0..16u32).map(|i| (i * 1009 + 13) % 100_000).collect();

    // ---- Path A: cache disabled (Direct-Path) ----
    let kv_a = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 256,
    }).expect("kv_a");
    let mut fwd_a = Forward::new(&dev, &mut allocator, kv_a, cfg.clone(), None)
        .expect("forward_a");
    fwd_a.set_cache_enabled(false);
    let mut logits_a: Vec<Vec<f32>> = Vec::with_capacity(token_seq.len());
    for (pos, &tid) in token_seq.iter().enumerate() {
        let embd = embedding_row(&gguf, &cfg, tid).expect("embd_a");
        fwd_a.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos as u32)
            .expect("fwd_a step");
        logits_a.push(fwd_a.logits().expect("logits_a"));
    }
    fwd_a.destroy(&dev.device, &mut allocator);

    // ---- Path B: cache enabled (CB-Reuse) ----
    let kv_b = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 256,
    }).expect("kv_b");
    let mut fwd_b = Forward::new(&dev, &mut allocator, kv_b, cfg.clone(), None)
        .expect("forward_b");
    fwd_b.set_cache_enabled(true);
    assert!(fwd_b.cache_enabled(), "set_cache_enabled(true) didn't stick");
    let mut logits_b: Vec<Vec<f32>> = Vec::with_capacity(token_seq.len());
    for (pos, &tid) in token_seq.iter().enumerate() {
        let embd = embedding_row(&gguf, &cfg, tid).expect("embd_b");
        fwd_b.forward_token(&dev, &registry, &cmd_ctx, &model, &embd, pos as u32)
            .expect("fwd_b step");
        logits_b.push(fwd_b.logits().expect("logits_b"));
    }
    fwd_b.destroy(&dev.device, &mut allocator);

    // ---- Compare ----
    assert_eq!(logits_a.len(), logits_b.len());
    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0
    };
    for (pos, (la, lb)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
        assert_eq!(la.len(), lb.len(), "logits len mismatch at pos {pos}");
        let mut max_abs = 0f32;
        for (&a, &b) in la.iter().zip(lb.iter()) {
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
        }
        let am_a = argmax(la);
        let am_b = argmax(lb);
        eprintln!(
            "[parity] pos={:>2}  max_abs_err={:.3e}  argmax_a={}  argmax_b={}",
            pos, max_abs, am_a, am_b,
        );
        assert!(
            max_abs < 1e-6,
            "Stage 2D parity break at pos {pos}: max abs err {max_abs:.6e} ≥ 1e-6 \
             — the cache must produce IDENTICAL logits to the direct path",
        );
        assert_eq!(am_a, am_b, "argmax differs at pos {pos}: {am_a} vs {am_b}");
    }

    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

// =====================================================================
// Phase 5C — SPM tokenizer + Mistral support.
// =====================================================================

fn mistral_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VF_MISTRAL_PATH").map(PathBuf::from) {
        return Some(p);
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home)
        .join("models")
        .join("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf");
    if p.exists() { Some(p) } else { None }
}

#[test]
fn phase5c_spm_tokenizer_loads_mistral() {
    let Some(path) = mistral_path() else { return };
    let gguf = GgufFile::open(&path).expect("open Mistral");
    let tok = Tokenizer::from_gguf(&gguf).expect("Mistral tokenizer");
    assert_eq!(tok.vocab_size(), 32768, "Mistral vocab_size");
    assert_eq!(tok.eos_id, 2, "Mistral </s>");
    assert_eq!(tok.bos_id, Some(1), "Mistral <s>");
    assert!(tok.flavour().is_none(), "Mistral is SPM, has no BPE flavour");
    assert!(tok.is_spm());
    assert!(
        tok.special_id("[INST]").is_some(),
        "Mistral vocab must contain [INST] as a special token"
    );
    assert!(
        tok.special_id("[/INST]").is_some(),
        "Mistral vocab must contain [/INST] as a special token"
    );
}

#[test]
fn phase5c_spm_encode_decode_roundtrip() {
    let Some(path) = mistral_path() else { return };
    let gguf = GgufFile::open(&path).expect("open Mistral");
    let tok = Tokenizer::from_gguf(&gguf).expect("Mistral tokenizer");

    // Strings the SPM tokeniser must round-trip exactly through
    // encode→decode. Includes ASCII, multibyte, and a non-Latin script
    // that exercises byte-fallback if those code points aren't in the
    // 32k vocab.
    for s in [
        "Hello World",
        "1+1=2",
        "Hallo Welt!",
        "The quick brown fox jumps over the lazy dog.",
        "Mistral-7B-Instruct-v0.3",
    ] {
        let ids = tok.encode(s);
        assert!(!ids.is_empty(), "encode {s:?} produced no tokens");
        let back = tok.decode(&ids);
        assert_eq!(back, s, "SPM roundtrip mismatch for {s:?}: got {back:?}");
    }
}

#[test]
fn phase5c_spm_byte_fallback_handles_unicode() {
    let Some(path) = mistral_path() else { return };
    let gguf = GgufFile::open(&path).expect("open Mistral");
    let tok = Tokenizer::from_gguf(&gguf).expect("Mistral tokenizer");

    // Japanese (likely not in the 32k Mistral vocab as whole words —
    // exercises the <0xHH> byte-fallback path).
    let s = "こんにちは";
    let ids = tok.encode(s);
    let back = tok.decode(&ids);
    assert_eq!(back, s, "byte-fallback roundtrip mismatch: got {back:?}");
}

#[test]
fn phase5c_mistral_chat_template_brackets() {
    use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
    let Some(path) = mistral_path() else { return };
    let gguf = GgufFile::open(&path).expect("open Mistral");
    let tok = Tokenizer::from_gguf(&gguf).expect("Mistral tokenizer");

    let template = ChatTemplate::detect(&gguf, &tok);
    assert_eq!(
        template,
        ChatTemplate::Mistral,
        "Mistral chat-template must auto-detect as Mistral"
    );

    let bos = tok.bos_id.expect("Mistral has BOS");
    let inst_open = tok.special_id("[INST]").expect("[INST]");
    let inst_close = tok.special_id("[/INST]").expect("[/INST]");
    let prompt = template.render_first_turn(&tok, "", "What is 2+2?");
    assert_eq!(prompt[0], bos, "Mistral first-turn starts with <s>");
    assert_eq!(prompt[1], inst_open, "after BOS comes [INST]");
    assert_eq!(
        prompt.last().copied(),
        Some(inst_close),
        "first-turn ends with [/INST]"
    );
    // Exactly one of each bracket id in a single-turn prompt.
    assert_eq!(
        prompt.iter().filter(|&&id| id == inst_open).count(),
        1
    );
    assert_eq!(
        prompt.iter().filter(|&&id| id == inst_close).count(),
        1
    );
}

// =====================================================================
// Prompt 16 — Alice multi-turn context retention.
//
// Six-turn `ChatSession` exchange with NO `reset()` between turns. On a
// working KV-cache + chat-template-continuation path the model
// recalls the user-stated facts (name, city) in later turns. The three
// "critical" turns (3, 5, 6) MUST pass; soft turns (1, 2, 4) are not
// asserted here (e.g. Qwen3's <think> block can hit max_tokens before
// emitting "4" for `2+2` — that's a budgeting detail, not a KV-cache
// bug). A failure on a critical turn means the multi-turn pipeline is
// broken (KV-cache being reset between turns, position offset wrong,
// chat-template continuation rendered with the wrong boundary, …).
// =====================================================================

#[test]
fn phase_prompt16_alice_context_retention_qwen3() {
    use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
    use vulkanforge::backend::vulkan::decode::GenerateConfig;
    use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
    use vulkanforge::backend::vulkan::loader::LoadedModel;
    use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;

    const MAX_SEQ_LEN: u32 = 2048;
    let Some(path) = qwen3_path() else { return };

    let dev = VulkanDevice::new().expect("device");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .expect("allocator");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("registry");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd");

    let gguf = GgufFile::open(&path).expect("open Qwen3");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("upload");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let kv_cache = KvCache::new(
        &dev.device,
        &mut allocator,
        KvCacheConfig {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len: MAX_SEQ_LEN,
        },
    )
    .expect("kv");
    let forward = Forward::new(&dev, &mut allocator, kv_cache, cfg.clone(), None)
        .expect("forward");
    let template = ChatTemplate::detect(&gguf, &tokenizer);
    let mut session = ChatSession::new_with_template(
        forward,
        "You are a helpful assistant.",
        template,
    );

    // (user_msg, expect_keyword_lowercase, max_tokens, critical)
    // Qwen3 streams a long <think> block before the answer; bump the
    // max-tokens cap on critical turns so the model has room to emit
    // the recall keyword AFTER `</think>`.
    let turns: &[(&str, &[&str], u32, bool)] = &[
        ("My name is Alice.",                       &["alice"],            120, false),
        ("What is 2 + 2?",                          &["4"],                120, false),
        ("What is my name?",                        &["alice"],            220, true),
        ("I live in Berlin.",                       &["berlin"],           120, false),
        ("Where do I live?",                        &["berlin"],           220, true),
        ("What is my name and where do I live?",    &["alice", "berlin"],  300, true),
    ];

    let mut critical_pass = 0usize;
    let mut critical_total = 0usize;
    let mut last_pos = 0u32;
    for (i, (user, expect, max_tok, critical)) in turns.iter().enumerate() {
        if *critical {
            critical_total += 1;
        }
        let cfg_g = GenerateConfig {
            max_tokens: *max_tok,
            print_stream: false,
            think_filter: false, sampling: Default::default(),
        };
        let result = session.send(
            &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
            user, &cfg_g,
        ).unwrap_or_else(|e| panic!("Alice turn {} failed: {e}", i + 1));

        let lower = result.generated_text.to_lowercase();
        let passed = expect.iter().all(|k| lower.contains(*k));
        eprintln!(
            "[alice] turn {} '{}' kw={:?} pass={} pos={} gen={}",
            i + 1, user, expect, passed, session.current_pos, result.generated_tokens,
        );
        if *critical {
            assert!(
                passed,
                "CRITICAL Alice turn {} failed: expected {:?} in response, got: {}",
                i + 1, expect, &result.generated_text,
            );
            critical_pass += 1;
        }
        last_pos = session.current_pos;
    }

    assert_eq!(
        critical_pass, critical_total,
        "Alice critical-turn score should be {}/{}, got {}/{}",
        critical_total, critical_total, critical_pass, critical_total,
    );
    assert!(
        last_pos > 0 && last_pos < MAX_SEQ_LEN,
        "Alice should leave the KV-cache populated within bounds (got pos={last_pos})"
    );

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

// =====================================================================
// Phase 5B.2 — batched-Q prefill attention.
//
// `dispatch_layer_batch` now folds the M-fold per-token attention
// loop into a single `flash_attn_batch` dispatch when
// `batch_attn_enabled` is set. The tests below build two `Forward`
// instances with explicit `batch_attn_enabled` settings and confirm
// that:
//   1. argmax of the prefill logits matches between the two paths.
//   2. top-5 overlap is ≥ 4/5 (same gate as `phase3e_…_top5`).
//   3. multi-turn decode after a batched prefill produces a coherent
//      continuation (verifies the KV cache survives the new path).
//   4. continuation prefill at q_start > 0 still recalls earlier
//      facts from the prior turn (multi-turn KV-cache + q_start arith).
// =====================================================================

fn batched_prefill_logits(
    path: &PathBuf,
    user_prompt: &str,
    batch_attn: bool,
    max_seq_len: u32,
) -> (Vec<f32>, u32) {
    use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
    use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;

    let dev = VulkanDevice::new().expect("dev");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    }).expect("alloc");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("reg");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd");
    let gguf = GgufFile::open(path).expect("gguf");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("upload");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tok");

    let mut prompt_tokens =
        vulkanforge::backend::vulkan::tokenizer::apply_chat_template(
            &tokenizer, user_prompt, None,
        );
    if prompt_tokens.len() > 64 { prompt_tokens.truncate(64); }
    let seq_len = prompt_tokens.len() as u32;

    let kv = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len,
    }).expect("kv");
    let mut fwd = Forward::new_with_prefill(
        &dev, &mut allocator, kv, cfg.clone(), None, seq_len,
    ).expect("fwd");
    fwd.set_batch_attn_enabled(batch_attn);
    fwd.kv_cache.reset();

    let mut all_embeds: Vec<f32> = Vec::new();
    for &tid in &prompt_tokens {
        all_embeds.extend(embedding_row(&gguf, &cfg, tid).expect("embd"));
    }
    fwd.prefill_batch(
        &dev, &registry, &cmd_ctx, &model, &all_embeds, seq_len, 0,
    ).expect("prefill_batch");
    let logits = fwd.logits().expect("logits");

    fwd.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);

    (logits, seq_len)
}

#[test]
fn phase5b2_batch_attn_parity_qwen3_short() {
    let Some(path) = qwen3_path() else { return };
    let prompt = "Explain what a mutex is in one sentence.";
    let (logits_off, seq_off) = batched_prefill_logits(&path, prompt, false, 256);
    let (logits_on, seq_on) = batched_prefill_logits(&path, prompt, true, 256);
    assert_eq!(seq_off, seq_on, "tokenization should be deterministic");

    assert!(
        !logits_on.iter().any(|v| !v.is_finite()),
        "batched prefill produced NaN/Inf"
    );

    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let top1_off = argmax(&logits_off);
    let top1_on = argmax(&logits_on);
    let top_k = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let top5_off = top_k(&logits_off, 5);
    let top5_on = top_k(&logits_on, 5);
    let overlap = top5_off.iter().filter(|t| top5_on.contains(t)).count();
    eprintln!(
        "[phase5b2] short top1_off={top1_off} top1_on={top1_on} \
         top5_off={top5_off:?} top5_on={top5_on:?} overlap={overlap}"
    );
    assert_eq!(top1_off, top1_on,
        "argmax differs: per-token={top1_off} batch={top1_on}\n\
         top5 off: {top5_off:?}\n\
         top5 on:  {top5_on:?}");
    assert!(overlap >= 4,
        "top-5 overlap {overlap}/5 too low (off={top5_off:?} on={top5_on:?})");
}

#[test]
fn phase5b2_batch_attn_parity_qwen3_two_tiles() {
    let Some(path) = qwen3_path() else { return };
    // Long enough that seq_len caps at 64 in `batched_prefill_logits`
    // — last query has two TILE iterations, exercises the per-query
    // causal triangle past the TILE=64 boundary.
    let prompt = "Write a long, detailed paragraph explaining what a \
                  mutex is, why it is needed, when you use it, what \
                  alternatives exist, and how it differs from a \
                  semaphore in low-level systems programming.";
    let (logits_off, seq_off) = batched_prefill_logits(&path, prompt, false, 256);
    let (logits_on, seq_on) = batched_prefill_logits(&path, prompt, true, 256);
    assert_eq!(seq_off, seq_on);
    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let top1_off = argmax(&logits_off);
    let top1_on = argmax(&logits_on);
    let top_k = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let top5_off = top_k(&logits_off, 5);
    let top5_on = top_k(&logits_on, 5);
    let overlap = top5_off.iter().filter(|t| top5_on.contains(t)).count();
    eprintln!(
        "[phase5b2] two_tiles seq={seq_on} top1_off={top1_off} top1_on={top1_on} \
         top5_off={top5_off:?} top5_on={top5_on:?} overlap={overlap}"
    );
    assert_eq!(top1_off, top1_on,
        "argmax differs (two-tile prompt): per-token={top1_off} batch={top1_on}");
    assert!(overlap >= 4, "top-5 overlap {overlap}/5 too low at seq={seq_on}");
}

#[test]
fn phase5b2_decode_after_batched_prefill_qwen3() {
    use vulkanforge::backend::vulkan::chat::ChatSession;
    use vulkanforge::backend::vulkan::chat_template::ChatTemplate;
    use vulkanforge::backend::vulkan::decode::GenerateConfig;
    use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
    use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;

    const MAX_SEQ_LEN: u32 = 2048;
    let Some(path) = qwen3_path() else { return };

    let dev = VulkanDevice::new().expect("dev");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    }).expect("alloc");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("reg");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd");
    let gguf = GgufFile::open(&path).expect("gguf");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("upload");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tok");
    let kv = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: MAX_SEQ_LEN,
    }).expect("kv");
    let mut fwd = Forward::new(&dev, &mut allocator, kv, cfg.clone(), None).expect("fwd");
    fwd.set_batch_attn_enabled(true);
    let template = ChatTemplate::detect(&gguf, &tokenizer);
    let mut session = ChatSession::new_with_template(
        fwd, "You are a helpful assistant.", template,
    );

    let cfg_g = GenerateConfig {
        max_tokens: 220,
        print_stream: false,
        think_filter: false, sampling: Default::default(),
    };
    // Turn 1: introduce a fact via batched prefill.
    let r1 = session.send(
        &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
        "I live in Berlin.", &cfg_g,
    ).expect("turn1");
    assert!(r1.generated_tokens > 0, "turn1 produced no tokens");

    // Turn 2: continuation prefill at q_start = current_pos > 0.
    // Must recall "Berlin" from the prior turn — fails fast if the new
    // prefill path leaves the KV cache in a broken state.
    let r2 = session.send(
        &dev, &registry, &cmd_ctx, &model, &gguf, &cfg, &tokenizer,
        "Where do I live?", &cfg_g,
    ).expect("turn2");
    let lower = r2.generated_text.to_lowercase();
    assert!(
        lower.contains("berlin"),
        "decode after batched prefill failed to recall 'Berlin' — got: {}",
        r2.generated_text
    );
    eprintln!(
        "[phase5b2] batched-prefill recall pos={} gen={} reply={:?}",
        session.current_pos, r2.generated_tokens,
        phase5b2_first_chars(&r2.generated_text, 80)
    );

    session.forward.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);
}

fn phase5b2_first_chars(s: &str, n: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= n { out.push('…'); break; }
        out.push(c);
    }
    out
}

// =====================================================================
// Sprint 5B — chunked-prefill parity
// `decode.rs` now slices any prompt longer than `max_prefill_tokens`
// into chunks and feeds them through `prefill_batch` one after the
// other (replacing the old token-by-token fallback). The chunked path
// must produce identical final logits to a single-shot prefill_batch
// of the same prompt — argmax bit-equal, top-5 ≥ 4/5 overlap. If the
// KV-cache offsets, RoPE positions, or attention bounds drift between
// chunks this test catches it.
// =====================================================================
fn sprint5b_chunked_logits(
    path: &PathBuf,
    user_prompt: &str,
    max_prefill_tokens: u32,
) -> (Vec<f32>, u32) {
    use vulkanforge::backend::vulkan::kv_cache::{KvCache, KvCacheConfig};
    use vulkanforge::backend::vulkan::pipeline_registry::PipelineRegistry;

    let dev = VulkanDevice::new().expect("dev");
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: dev.instance.clone(),
        device: dev.device.clone(),
        physical_device: dev.physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    }).expect("alloc");
    let (registry, _) = PipelineRegistry::new(&dev.device, None).expect("reg");
    let cmd_ctx = CommandContext::new(&dev.device, dev.queue_family_index).expect("cmd");
    let gguf = GgufFile::open(path).expect("gguf");
    let cfg = ModelConfig::from_gguf(&gguf).expect("cfg");
    let model = LoadedModel::load(&dev, &mut allocator, &gguf).expect("upload");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tok");

    let mut prompt_tokens =
        vulkanforge::backend::vulkan::tokenizer::apply_chat_template(
            &tokenizer, user_prompt, None,
        );
    if prompt_tokens.len() > 120 { prompt_tokens.truncate(120); }
    let seq_len = prompt_tokens.len() as u32;

    let kv = KvCache::new(&dev.device, &mut allocator, KvCacheConfig {
        n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim, max_seq_len: 256,
    }).expect("kv");
    let mut fwd = Forward::new_with_prefill(
        &dev, &mut allocator, kv, cfg.clone(), None, max_prefill_tokens,
    ).expect("fwd");
    fwd.set_batch_attn_enabled(true);
    fwd.kv_cache.reset();

    // Walk the same chunks the production decode.rs path walks.
    let chunk_size = max_prefill_tokens.max(1) as usize;
    let mut pos: u32 = 0;
    for chunk in prompt_tokens.chunks(chunk_size) {
        let chunk_len = chunk.len() as u32;
        let mut chunk_embeds: Vec<f32> = Vec::with_capacity(
            chunk.len() * cfg.hidden_dim as usize,
        );
        for &tid in chunk {
            chunk_embeds.extend(embedding_row(&gguf, &cfg, tid).expect("embd"));
        }
        fwd.prefill_batch(
            &dev, &registry, &cmd_ctx, &model, &chunk_embeds, chunk_len, pos,
        ).expect("prefill_batch chunk");
        pos += chunk_len;
    }
    let logits = fwd.logits().expect("logits");

    fwd.destroy(&dev.device, &mut allocator);
    cmd_ctx.destroy(&dev.device);
    model.destroy(&dev.device, &mut allocator);
    registry.destroy(&dev.device);
    drop(allocator);

    (logits, seq_len)
}

#[test]
fn sprint5b_chunked_prefill_parity_qwen3() {
    let Some(path) = qwen3_path() else { return };
    // Long enough that the small chunk_size forces ≥3 chunks.
    let prompt = "Write a long, detailed paragraph explaining what a \
                  mutex is, why it is needed, when you use it, what \
                  alternatives exist, and how it differs from a \
                  semaphore in low-level systems programming.";
    // Reference: single-shot prefill_batch (cap = full prompt).
    let (logits_single, seq_single) = sprint5b_chunked_logits(&path, prompt, 120);
    // Test: 3+ chunks of size 32 over the same 120-token prompt.
    let (logits_chunked, seq_chunked) = sprint5b_chunked_logits(&path, prompt, 32);
    assert_eq!(seq_single, seq_chunked, "tokenization should be deterministic");
    assert!(
        !logits_chunked.iter().any(|v| !v.is_finite()),
        "chunked prefill produced NaN/Inf"
    );

    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let top1_single = argmax(&logits_single);
    let top1_chunked = argmax(&logits_chunked);
    let top_k = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let top5_single = top_k(&logits_single, 5);
    let top5_chunked = top_k(&logits_chunked, 5);
    let overlap = top5_single
        .iter()
        .filter(|t| top5_chunked.contains(t))
        .count();
    eprintln!(
        "[sprint5b] chunked_parity seq={seq_single} top1_single={top1_single} \
         top1_chunked={top1_chunked} top5_single={top5_single:?} \
         top5_chunked={top5_chunked:?} overlap={overlap}"
    );
    assert_eq!(top1_single, top1_chunked,
        "argmax differs (chunked vs single): single={top1_single} chunked={top1_chunked}\n\
         top5 single:  {top5_single:?}\n\
         top5 chunked: {top5_chunked:?}");
    assert!(overlap >= 4,
        "top-5 overlap {overlap}/5 too low (single={top5_single:?} chunked={top5_chunked:?})");
}
