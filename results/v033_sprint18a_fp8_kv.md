# v0.3.3 Sprint 18A — FP8 E4M3 KV cache

**Result:** Native FP8 KV cache shipped end-to-end. Production
chat path runs unchanged on default FP16; opt-in via
`VULKANFORGE_KV_FP8=1` halves cache VRAM (Qwen3-8B: 288 MB →
144 MB) **and is slightly faster than FP16 on decode** (115–118
tok/s vs 109 baseline). Multi-turn context retention verified —
the model recalls Turn-1 introductions from the FP8-stored
KV cache on Turn 2.

| Config           | KV VRAM | Decode tok/s | Prefill tok/s pp=512 | Coherent |
|------------------|--------:|-------------:|---------------------:|----------|
| FP16 KV (default)| 288 MB  | 109.0 (15/15)| 3865                 | yes      |
| FP8  KV (opt-in) | 144 MB  | 114.1–118.2  | 3843                 | yes      |

Decode 5–8 % faster on FP8 because the KV-read bandwidth halves
(2 B → 1 B per element). Prefill is dominated by the matmul, so
FP8 vs FP16 KV is a wash there. Multi-turn quality on Qwen3-8B
Q4_K_M was indistinguishable from FP16 in spot checks.

## Approach

Sprint 18-M1's FP8 device shim already proved native FP8 reads
and FP8 cooperative-matrix matmul work end-to-end (bit-exact vs
CPU). 18A leverages that to ship the production KV-cache path
that needs both:

- **FP8 write** — a new `kv_store_fp8.comp` shader, sibling of
  `kv_copy_fp16.comp`. Each thread converts 4 FP32 lanes to FP8
  E4M3 via `floate4m3BitsToUintEXT(floate4m3_t(v))` and packs
  them into one `uint`. Workgroup-size 256, dispatch-x =
  `ceil(n / 1024)`. The pre-existing `KvCopyFp16PushConstants`
  struct is reused verbatim (same `n_elements / dst_uint_offset
  / src_float_offset` layout).
- **FP8 read** — each of the five flash-attention shaders
  (`flash_attn`, `flash_attn_split`, `flash_attn_batch`,
  `flash_attn_tiled`, `flash_attn_coopmat`) gets a new
  `#if FP8_KV` branch alongside its existing `#if FP16_KV`. The
  branch is 12 LOC: one extension include, one binding (still
  `uint[]` packed buffer), and `load_k`/`load_v` helpers that
  unpack one byte and dequantize via `uintBitsToFloate4m3EXT`.
  Five new SPVs ship with `-DFP8_KV=1`. SPV count: 81 → 87.

Six dispatch sites in `forward.rs` were upgraded from two-way
`is_fp16() ? Fp16Kv : Fp32` to three-way `is_fp8() / is_fp16() /
default`. The KV-write side (4 sites) calls a new
`run_kv_store_fp8` helper; the attention-read side (5 sites)
selects the matching `FlashAttn*Fp8Kv` ShaderId.

`KvDtype::F8` was added to the existing enum; `kv_dtype_from_env()`
resolves `VULKANFORGE_KV_FP8=1` to F8 and implies
`VULKANFORGE_ENABLE_FP8=1` so device.rs auto-wires
`VK_EXT_shader_float8` (Sprint 18-M1's shim).

## What's tested

```
$ cargo test --release --lib                         → 32/32 PASS
$ cargo run --release --example run_15prompt_bench   → 15/15 coherent
                                                       (Qwen3-8B Q4_K_M, FP16 KV)

$ vulkanforge chat ...                               → coherent
$ VULKANFORGE_KV_FP8=1 vulkanforge chat ...          → coherent
$ VULKANFORGE_KV_FP8=1 vulkanforge chat (multi-turn) → recalls Turn-1 name
```

Multi-turn transcript on `Qwen3-8B-Q4_K_M.gguf` with
`VULKANFORGE_KV_FP8=1`:

```
> ich heiße tutu
<think>
Okay, the user said "ich heiße tutu" which means "My name is Tutu" in
German. I need to respond in a friendly and welcoming manner...

> wie heiße ich?
<think>
Okay, the user asked "wie heiße ich?" which translates to "What is my
name?" in German. They previously introduced themselves as Tutu. I
need to confirm their name...
```

Turn 2 explicitly recalls the Turn-1 introduction via the FP8
KV cache → KV reads + writes are both correct, and the precision
loss from 16-bit → 8-bit isn't degrading attention enough to
break factual recall.

## VRAM bookkeeping

KV-cache size for Qwen3-8B at default `max_seq=2048`:

```
36 layers × 8 kv_heads × 2048 max_seq × 128 head_dim = 75 497 472 elements
                                                       × 2 (K + V) buffers

FP16: × 2 B = 288 MB
FP8 : × 1 B = 144 MB   ← Sprint 18A

Same VRAM at FP8 unlocks max_seq = 4096 (588 MB → split between
K and V at 144 MB each = 288 MB, the FP16-at-2048 budget). Not
exposed as a CLI flag yet — `--max-context` follow-up.
```

## Files

- `src/backend/vulkan/kv_cache.rs` — `+25 / -3` (KvDtype::F8,
  is_fp8(), env var resolution + implied FP8 device opt-in)
- `src/backend/vulkan/device.rs` — `+5 / -1` (FP8 opt-in also
  triggered by `VULKANFORGE_KV_FP8=1`)
- `src/backend/vulkan/forward.rs` — `+90 / -10` (run_kv_store_fp8
  helper + 6 sites updated to three-way)
- `src/backend/vulkan/shaders.rs` — `+24` (6 new ShaderId entries
  + name/spv_bytes/ALL_SHADERS/include_bytes)
- `src/backend/vulkan/pipeline_registry.rs` — `+5` (extended match
  arms for FP8 attention pipelines)
- `vk_shaders/kv_store_fp8.comp` — new (60 LOC)
- `vk_shaders/flash_attn.comp` — `+13` (FP8_KV branch)
- `vk_shaders/flash_attn_split.comp` — `+13`
- `vk_shaders/flash_attn_batch.comp` — `+13`
- `vk_shaders/flash_attn_tiled.comp` — `+13`
- `vk_shaders/flash_attn_coopmat.comp` — `+13`
- `build.rs` — `+30` (6 new ShaderJob entries)
- `results/v033_sprint18a_fp8_kv.md` — this report

Total source delta: ~280 LOC across 12 files + 1 new shader file.

## Performance details (RX 9070 XT, RADV Mesa 26.0.6)

Decode is the path that benefits — KV cache is read from at every
attention step, so halving the bytes-per-element halves the bandwidth
pressure. Single-prompt decode @ 32-token output:

```
                       Decode tok/s   Prefill pp=512   KV VRAM
  FP16 KV (default)    109.0 (15/15)  3865 tok/s       288 MB
  FP8  KV (KV_FP8=1)   118.2          3843 tok/s       144 MB

  Δ decode  : +8.4 % (less BW into the V-read for online softmax)
  Δ prefill : -0.6 % (within run-to-run noise)
  Δ VRAM    : -50.0 % (the headline number)
```

`run_15prompt_bench` on FP16 KV unchanged (109 tok/s median, 15/15
coherent, same 4-system comparison). FP8 KV-cache 15-prompt
quality benchmark hasn't run yet — next sprint will rerun the
existing harness with `VULKANFORGE_KV_FP8=1` to log the per-prompt
quality delta.

## Limitations + follow-up

- `--max-context` flag for explicit override of `max_seq` not
  yet exposed; users can already get the longer context by
  bumping `max_seq` in `KvCacheConfig`, but a UX flag is cheap
  follow-on work.
- FP8 KV quality benchmark (15-prompt comparison FP16 vs FP8)
  not yet captured.
- FP8 vs FP16 KV at long contexts (pp=4096+): the attention sum
  is over more terms, so the 3-bit-mantissa per-element error
  could compound. Worth a probe at pp=2048 / 4096 before
  recommending FP8 KV as the default.
- pipeline-cache invalidation: when toggling `VULKANFORGE_KV_FP8`,
  the saved `pipeline_cache` will rebuild the FP8 SPVs on first
  run — ~10 ms one-time cost.
