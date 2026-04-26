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
            think_filter: false,
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
        think_filter: true,
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
            &GenerateConfig { max_tokens: 80, print_stream: false, think_filter: false },
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
