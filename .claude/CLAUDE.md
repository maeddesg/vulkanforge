# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VulkanForge is a Vulkan-based LLM inference engine targeting AMD RDNA 4 (`gfx1201`). The Vulkan backend is **compute-only** — no swapchain, no graphics queues — and uses `ash` 0.38 (Vulkan 1.3) directly rather than a higher-level wrapper.

The codebase is in **Phase 0** (device-init smoke test). Phase 1 work — memory allocator, descriptor sets, compute pipelines, and a Q4_K matrix-vector multiply dispatch — is the next milestone. Several `src/` subdirectories (`core/`, `introspection/`, `model/`, `monitor/`, `runtime/`) are empty placeholders for those phases.

## Common commands

```bash
cargo run                    # Phase 0 smoke: enumerate GPU, init device, dump heaps
cargo build --release        # release build (LTO=thin, opt-level=3)
cargo check                  # fast type-check
cargo test                   # tests directory currently empty
```

MSRV is **Rust 1.85** (edition 2024).

## Architecture notes

### Vulkan device init (`src/backend/vulkan/device.rs`)
- Picks the **first DISCRETE_GPU** physical device, falling back loudly to `physical_devices[0]` for iGPU/CI setups.
- Picks the **first queue family with the COMPUTE bit** — on AMD this is typically the unified GRAPHICS+COMPUTE+TRANSFER family, but only COMPUTE is required.
- `VulkanDevice::Drop` tears down `device` then `instance` in reverse order; the queue is implicitly owned by the device. Preserve this ordering when extending the struct.
- No validation layers and no instance/device extensions beyond core are enabled yet. Add them as Phase 1 features need them.

### Shader pipeline (`vk_shaders/`)
- Compute shaders are GLSL (`*.comp`) and target SPIR-V; compiled outputs go to `vk_shaders/spirv/` and are gitignored (`*.spv`).
- `shaderc` 0.8 is declared as a `build-dependencies` entry, but **no `build.rs` exists yet** — adding it is part of Phase 1. When you add it, drive shader compilation from there so SPIR-V is rebuilt on `.comp`/`.glsl` changes.
- The two `.comp` files (`dequant_q4_k.comp`, `mul_mat_vec_q4_k.comp`) are ports of llama.cpp's Vulkan Q4_K kernels and `#include` headers (`dequant_head.glsl`, `mul_mat_vec_base.glsl`) that have **not been added to the repo yet**. Those headers must accompany any attempt to compile the shaders.

### Dependencies and their roles
- `ash` — raw Vulkan bindings.
- `gpu-allocator` — Vulkan memory suballocator (Phase 1: replaces ad-hoc `vkAllocateMemory`).
- `bytemuck` (with `derive`) — POD casts for descriptor/push-constant structs and quantized buffer layouts.
- `half` — `f16` for activation tensors.
- `memmap2` — mmap GGUF / model weight files.

## Conventions

- Keep `unsafe` blocks scoped to single FFI calls and comment any non-obvious lifetime/ordering assumption (see `device.rs` Drop).
- Log device/queue selection decisions to stdout — the smoke test's value is its visibility.
- `main.rs` is intentionally a thin demo driver; real entry points belong under `src/runtime/` once that module materialises.
