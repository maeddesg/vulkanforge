# Gemma-4 26B-A4B MoE: Async Decode Race Condition on Vulkan

## Summary

VulkanForge achieves coherent Gemma-4 26B-A4B-it inference on consumer
AMD RDNA 4 hardware (Radeon RX 9070 XT) via Vulkan compute shaders,
with no graphics queue and no swapchain. The model produces coherent
multi-token output — factual answers, multi-sentence explanations, and
even rhyming poetry — at ~20 tok/s decode on a single 16 GB consumer
GPU.

Two critical bugs were uncovered and fixed during the 29-sprint
bring-up:

1. **Async MoE Race Condition** — `mid_frame_submit_and_wait` in the
   MoE router reads stale GPU buffers when the decode pipeline
   pre-records command buffers asynchronously.
2. **V-Race Barrier** — A missing `SHADER_WRITE → TRANSFER_READ`
   pipeline barrier before `cmd_copy_buffer` in the K→V copy path
   leaks the previous position's K into the current V cache.

Both fixes are tiny (a few lines each) but the symptoms cascade so
catastrophically through the rest of the forward pass that the model
output was hard-stuck on `"Paris Paris Paris…"` or multilingual
garbage — looking for all the world like a model-quality or
quantization issue rather than a GPU synchronization issue.

## Results

With both fixes applied, Gemma-4 26B-A4B-it produces coherent
multi-token output on a single RX 9070 XT, all measured on Q3_K_M GGUF
(12 GiB VRAM), with natural EOS, no `VULKANFORGE_DISABLE_ASYNC_DECODE`
env var (the fix applies automatically):

| Prompt | Output | Tokens | Decode |
|--------|--------|--------|--------|
| `"The capital of France is"` | `"Paris."` | 2 | 20.2 tok/s |
| `"What is 2+2? Answer in one sentence."` | `"Two plus two equals four."` | 6 | 20.3 tok/s |
| `"Was ist die Hauptstadt von Deutschland?"` | `"Die Hauptstadt von Deutschland ist **Berlin**."` | 8 | 20.6 tok/s |
| `"Explain what a neural network is in three sentences."` (temp=0.6) | `"A neural network is a computational model inspired by the structure and function of the human brain. It consists of interconnected layers of artificial neurons that process data by assigning mathematical weights to different inputs. By adjusting these weights through a process called training, the network learns to recognize complex patterns and make accurate predictions."` | 59 | 18.5 tok/s |
| `"Write a short poem about the moon."` (temp=0.6) | An 8-line AABB-rhymed poem (see Sample Output section below) | 67 | 18.7 tok/s |

## Performance (RX 9070 XT, RDNA 4)

| Model | Quant | VRAM | Decode | Prefill | Status |
|-------|-------|------|--------|---------|--------|
| Qwen3-8B | Q4_K_M | 6 GiB | 107 tok/s | 581 tok/s | coherent |
| Llama-3.1-8B | Q4_K_M | 6 GiB | 113 tok/s | 284 tok/s | coherent |
| Gemma-4-E2B | Q4_K_M | 2.5 GiB | 62 tok/s | 74 tok/s | coherent |
| **Gemma-4-26B-A4B** | **Q3_K_M** | **12 GiB** | **20 tok/s** | **40 tok/s** | **coherent (new in v0.4.2)** |

Prefill numbers are for short prompts (~17–25 tokens) measured in
this release. 26B uses `force_per_token_prefill` mode, which caps
prefill at decode-like speeds; batched MoE prefill is planned for v0.5.

## 15-Prompt Coherence Bench (Gemma-4 26B Q3_K_M, greedy)

All 15 prompts run with `temperature=0.0` (greedy), `--max-tokens 200`,
no `VULKANFORGE_DISABLE_ASYNC_DECODE` env var (the v0.4.2 auto-disable
applies). Result: **15/15 semantically coherent** (12 clean, 3 with
minor Q3_K token-level artifacts where the model picked a near-tied
alternative — `"e.1."` instead of `"e.g."` etc.; semantics correct in
all cases).

| # | Prompt | Output (summary) | Tokens | Verdict |
|---|--------|------------------|--------|---------|
| 1 | "The capital of France is" | "Paris." | 2 | PASS |
| 2 | "What is 2+2? Answer in one sentence." | "Two plus two equals four." | 6 | PASS |
| 3 | "Explain what a neural network is in three sentences." | 3 coherent sentences (brain analogy, layers/weights, training) | 64 | PASS |
| 4 | "Write a short poem about the moon." | 8-line AABB-rhymed poem (see below) | 65 | PASS |
| 5 | "What is the speed of light in meters per second?" | "299,792,458 meters per second" — exact | 25 | PASS |
| 6 | "Translate 'Good morning' to German, French, and Japanese." | "Guten Morgen / Bonjour / おはようございます (Ohayō gozaimasu)" — all correct, native script | 41 | PASS |
| 7 | "Name three benefits of regular exercise." | 3 structured bullets (cardio, mental, weight) | 125 | PASS |
| 8 | "What is photosynthesis? Explain briefly." | Detailed answer with chemical equation; one "**" → "together" token-swap glitch in bullet 2 | 200 (cap) | PARTIAL |
| 9 | "Write a haiku about rain." | "Soft gray clouds descend, / Silver droplets kiss the earth, / Life drinks from the sky." (5-7-5) | 19 | PASS |
| 10 | "Who wrote Romeo and Juliet?" | "**William Shakespeare** wrote *Romeo and Juliet*." | 10 | PASS |
| 11 | "What are the three states of matter?" | Solid/Liquid/Gas with particle behaviour; "### 1. Liquid" numbering bug after "### 1. Solid" | 200 (cap) | PARTIAL |
| 12 | "Explain the difference between a virus and a bacterium in two sentences." | Bacterium = single-celled living; virus = non-living genetic package needing host; antibiotic vs antiviral note | 61 | PASS |
| 13 | "What is the largest planet in our solar system?" | "The largest planet in our solar system is **Jupiter**." | 11 | PASS |
| 14 | "Write a one-paragraph summary of World War II." | 184-token paragraph: 1939-1945, Allies/Axis, theatres, Holocaust, atomic bombs, UN founding | 184 | PASS |
| 15 | "What is machine learning?" | Definition + comparison to traditional programming + apple-recognition analogy; minor `"e.1."` and `"e. longer"` token-artifacts | 200 (cap) | PARTIAL |

### Highlights

**Prompt 4 — Moon poem (greedy, AABB rhyme, two stanzas):**

```
A silver coin in velvet deep,
Watching while the weary sleep.
A silent guide through velvet night,
Draped in borrowed, ghostly light.

She pulls the tides with gentle hand,
Across the sea and sleeping land,
Changing shape from thin to round,
Without a word, without a sound.
```

**Prompt 9 — Haiku (5-7-5):**

```
Soft gray clouds descend,
Silver droplets kiss the earth,
Life drinks from the sky.
```

**Prompt 6 — Trilingual translation** with correct native Japanese
script (`おはようございます` plus romaji `(Ohayō gozaimasu)`) shows
the 262K-vocab Gemma-4 tokenizer + Q3_K weights preserve multilingual
fidelity.

**Prompt 14 — WW2 paragraph** (184 tokens, natural EOS) covers
allies/axis composition, theatres, Holocaust, nuclear weapons, and
the post-war UN founding — factually accurate, well-structured.

The 3 PARTIAL cases are all Q3_K-quantization token-selection
artifacts where the model picked a near-equiprobable alternative at
one or two token positions (typo-like, e.g. `"e.1."` for `"e.g."`).
Semantics are correct throughout. At Q4_K_M these typically clear up.

## Hardware / Stack

- GPU: AMD Radeon RX 9070 XT (RDNA 4, `gfx1201`, 16 GB VRAM)
- Driver: RADV (Mesa 26.1), Vulkan 1.4.341
- OS: Arch Linux (CachyOS)
- Backend: Vulkan compute-only, `ash` 0.38 directly, no graphics queue
- Quantization: Q3_K_M GGUF (12 GiB) or Q4_K from SafeTensors
- Inference engine: [VulkanForge](https://github.com/maeddesg/vulkanforge)

## Bug 1: Async Decode + MoE Mid-Frame Submit

### Symptom

Loading Gemma-4 26B-A4B-it (Q3_K_M, Q4_K_M, or SafeTensors) and running
greedy decode produced:

- Stuck-token loops (`"Paris Paris Paris …"`)
- Multilingual / non-sense tokens after the first correct token
- 18× too-large pre-softcap logits (max ~593 vs expected ~31)
- Saturated post-softcap distributions (5+ tokens at the `30.0` cap)

The first decoded token was correct (e.g. `"Paris"` after
`"The capital of France is"`) because the prefill path took a
different code path. Every subsequent decode token was garbage.

### Root Cause

Gemma-4 26B-A4B uses a Mixture-of-Experts FFN block: 128 experts with
top-8 routing per token. VulkanForge runs the router on the CPU: the
post-attention residual is copied to a host-visible staging buffer,
the GPU is drained (`vkQueueSubmit` + `vkWaitForFences`), the CPU
computes the routing decision (RMSNorm + scale + GEMM + softmax +
top-k), and the result is then used by the GPU side to select which
expert weights to multiply against. This GPU→CPU→GPU round-trip is
implemented as `mid_frame_submit_and_wait` and fires from inside
`record_decode_dispatches` — i.e., *while* the command buffer is
being recorded:

```rust
fn step_moe_route(&self, fwd, ..., ctx) {
    // ... copy res1 to host-visible staging ...
    fwd.mid_frame_submit_and_wait(ctx.dev, ctx.cmd)?;   // ←  end CB, submit, wait, reset+begin
    let raw = fwd.moe_route_staging.read_bytes()?;
    let routing = cpu_moe_route(&raw, ...);
    fwd.moe_routing = Some(routing);
    // ... rest of layer recorded into CB after this ...
}
```

VulkanForge's decode loop is async-pipelined by default. The CPU
records the command buffer for token N+1 *while the GPU executes
token N*, then writes the embedding and submits. The CB references
buffer **handles** (not contents), so handle-only operations are safe;
the data lands in the buffer at submit time:

```
ASYNC pipeline (broken for MoE):
    pre_record(slot, pos)            ← records CB, NO embedding written yet
        record_decode_dispatches
            ... attn block ...
            step_moe_route
                mid_frame_submit_and_wait  ← submits partial CB, reads stale scratch_a
                cpu_moe_route(stale_data)  ← picks wrong experts
                fwd.moe_routing = Some(WRONG)
            ... rest of layer uses WRONG routing ...
        end_command_buffer
    fill_embed_and_submit(slot, embd_for_token_N, pos)
        write embedding to scratch_a   ← CORRECT embedding, but TOO LATE
        queue_submit                    ← CB runs against wrong routing decisions
```

So the GPU computes attention against the *correct* embedding (because
that's written to `scratch_a` before submit) but then dispatches the
wrong expert weights (because routing was decided against the
*previous* slot occupant's stale data, two iterations ago since slots
alternate 0/1). The wrong-expert FFN output then cascades through the
remaining layers, producing 18× too-large logits at the output.

### The Race Visualized

```
SYNC (correct):
  ┌──────────────┐    ┌──────────────────────┐    ┌────────┐
  │ Write embed  │───▶│      Record CB       │───▶│ Submit │
  │ to scratch_a │    │   mid_frame_submit   │    │  CB    │
  └──────────────┘    │   reads CURRENT      │    └────────┘
                      │   scratch_a → right  │
                      │   expert routing     │
                      └──────────────────────┘

ASYNC (broken for MoE):
  ┌──────────────────────┐    ┌──────────────┐    ┌────────┐
  │      Record CB       │───▶│ Write embed  │───▶│ Submit │
  │   mid_frame_submit   │    │ to scratch_a │    └────────┘
  │   reads STALE        │    └──────────────┘
  │   scratch_a (prev    │                       (CB now runs against
  │   slot occupant)     │                        wrong expert routing
  │                      │                        baked in at record-
  │   → WRONG routing    │                        time → cascade of
  │     baked into CB    │                        wrong FFN outputs →
  └──────────────────────┘                        18× logits explosion)
```

### Fix

For MoE models, force the legacy synchronous decode loop, which
writes the embedding *before* recording the command buffer:

```rust
let async_decode = match std::env::var("VULKANFORGE_DISABLE_ASYNC_DECODE") {
    Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
    Err(_) => cfg
        .gemma4
        .as_ref()
        .map(|g| !g.enable_moe_block)
        .unwrap_or(true),
};
```

Non-MoE models (Llama, Qwen3, Mistral, Gemma-4-E2B without MoE) keep
async pipelining and their 100+ tok/s decode speed. MoE models eat a
~4% decode hit (21 → 20.2 tok/s on 26B) for correctness. A future
GPU-side router would eliminate the `mid_frame_submit_and_wait` and
allow async to come back.

## Bug 2: V-Race Pipeline Barrier

Gemma-4 uses `attention_k_eq_v = true` — the V projection for each
token is derived from the *raw* (pre-norm, pre-RoPE) K projection.
The VulkanForge implementation copies `k_buf` to `v_buf` via
`cmd_copy_buffer` right after the K projection compute dispatch:

```rust
fn step_v_from_k_raw(&self, fwd, cfg, ctx) {
    // ... no pre-barrier ...
    cmd_copy_buffer(cmd, k_buf, v_buf, &copy);     // ← TRANSFER_READ on k_buf
    fwd.mark_written(&[v_buf]);
}
```

The K projection (immediately upstream) is a `SHADER_WRITE` on `k_buf`.
The framework's `maybe_compute_barrier` only handles
`COMPUTE → COMPUTE` (SHADER_WRITE → SHADER_READ); it doesn't cover
`COMPUTE → TRANSFER`. Without an explicit pipeline barrier, the
copy can run before the GEMV finishes — reading stale `k_buf`
contents left over from the previous token's command buffer (the
*post*-RoPE state written by `step_k_norm_rope` at the end of that
CB).

### Signature

The bug had a clean, almost-too-good mathematical signature: a
per-position cosine comparison of VF's `V[pos=N]` against its own
`K[pos=N-1]` gave **`cos = 1.000000` exactly** for every position
N ≥ 1, and VF's `V[pos=0]` was *exactly all zeros* (no previous CB →
the buffer's initial zero state was what the racing copy read):

```
pos=1 kv=0  V vs K[same]=+0.8292   V vs K[prev]=+1.0000
pos=1 kv=1  V vs K[same]=+0.6067   V vs K[prev]=+1.0000
pos=2 kv=0  V vs K[same]=+0.9094   V vs K[prev]=+1.0000
pos=2 kv=1  V vs K[same]=+0.5571   V vs K[prev]=+1.0000
...
```

The `1.0000` exact match told us: this is a data race, not a math
bug. There is no arithmetic source of an exact off-by-one identity.

### Fix

Add two memory barriers around the copy:

```rust
let pre = vk::MemoryBarrier::default()
    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
cmd_pipeline_barrier(COMPUTE_SHADER, TRANSFER, [pre]);

cmd_copy_buffer(cmd, k_buf, v_buf, &copy);

let post = vk::MemoryBarrier::default()
    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
    .dst_access_mask(vk::AccessFlags::SHADER_READ);
cmd_pipeline_barrier(TRANSFER, COMPUTE_SHADER, [post]);
```

The post-barrier ensures the downstream `step_v_norm` compute read of
`v_buf` sees the transfer write (which the framework's standard
`COMPUTE → COMPUTE` barrier wouldn't have covered).

This fix alone moved the L5/11/17/23 hidden-state cosines from
`0.98/0.97/0.91/0.84` to `0.997/0.993/0.97/0.95` against an HF BF16
reference, but the model output still loops on `"Paris"` because of
Bug 1 above.

## Sample Output (after both fixes)

All measured on a single Radeon RX 9070 XT, 12 GiB VRAM used by the
model, no `VULKANFORGE_DISABLE_ASYNC_DECODE` env var set (the fix
applies automatically).

```
Prompt: "The capital of France is"
Output: "Paris."
        [18 prompt, 2 gen, prefill 40 tok/s, decode 20.2 tok/s]  natural EOS

Prompt: "What is 2+2? Answer in one sentence."
Output: "Two plus two equals four."
        [25 prompt, 6 gen, decode 20.3 tok/s]  natural EOS

Prompt: "Explain what a neural network is in three sentences."
Output: "A neural network is a computational model inspired by the
         structure and function of the human brain. It consists of
         interconnected layers of artificial neurons that process data
         by assigning mathematical weights to different inputs. By
         adjusting these weights through a process called training,
         the network learns to recognize complex patterns and make
         accurate predictions."
        [23 prompt, 59 gen, decode 18.5 tok/s]  natural EOS

Prompt: "Write a short poem about the moon."
Output:  A silver pearl in velvet night,
         Spilling pale and ghostly light,
         She watches while the weary sleep,
         And guards the secrets shadows keep.
         A silent orb, a glowing crest,
         She guides the tides with gentle rest,
         Changing shape from thin to grand,
         A lonely lantern o'er the land.
        [21 prompt, 67 gen, decode 18.7 tok/s]  AABB rhyme + meter

Prompt: "Was ist die Hauptstadt von Deutschland?"
Output: "Die Hauptstadt von Deutschland ist **Berlin**."
        [20 prompt, 8 gen, decode 20.6 tok/s]  natural EOS, German
```

## Regression — Non-MoE models keep async pipelining

```
Qwen3-8B Q4_K_M:    "2 + 2 equals 4."                       105 tok/s
Llama-3.1-8B Q4_K_M: "2 + 2 = 4"                            113 tok/s
Gemma-4-E2B Q4_K_M: "\"2+2\" is a very simple arithmetic     62 tok/s
                     problem. **The answer is 4.**"
```

All three keep their pre-fix decode speed because the fix only
disables async for `gemma4.enable_moe_block == true`.

## Relevance to Other Vulkan Inference Engines

The async decode + MoE mid-frame-submit race condition is not specific
to VulkanForge. Any Vulkan-based inference engine that:

1. **Pre-records command buffers** before writing the current token's
   input data (a standard latency-hiding pattern), AND
2. **Uses GPU→CPU readback during CB recording** for MoE expert
   selection (because Vulkan compute lacks a single-pass GPU top-k
   primitive at the time of writing)

will hit the same race. The symptoms are deceptive: the output looks
like a model-quality, quantization, or attention-math bug — not a
synchronization issue. We spent 29 sprints (and falsified 12+
hypotheses, including RoPE pair convention, attention softcapping,
attention scaling, post-norm γ outliers, MoE-router math, embed_scale,
and L29-specific divergence) before locating the actual root cause.

### Possible Mitigations

- **Force synchronous decode for MoE** (our current fix; –4% decode
  for the affected models, no change for non-MoE).
- **GPU-side MoE routing** — implement the whole router (parameterless
  RMSNorm + per-channel scale + GEMM + softmax + top-k + per-expert
  scale) as compute shaders, eliminating the CPU round-trip entirely.
  This is the proper long-term solution and re-enables async
  pipelining for MoE models.
- **Double-buffered routing input** — copy the pre-attention residual
  to a dedicated staging buffer in a slot-stable location *before* CB
  recording begins, so `mid_frame_submit_and_wait` always reads the
  right data regardless of slot rotation.

The V-race (Bug 2) is a more conventional missing-barrier issue and
affects any code path that mixes a `SHADER_WRITE` upstream with a
`TRANSFER_READ` downstream on the same buffer without an explicit
pipeline barrier — easy to introduce, easy to miss in validation
because most drivers serialize compute-queue submissions far enough
that the race only triggers reliably under specific scheduling
patterns (like the async pipeline above, where slot rotation
guarantees the stale data has known structure).

## Related Issues

- **[llama.cpp Issue #21516](https://github.com/ggml-org/llama.cpp/issues/21516)**
  — *"Eval bug: Gemma 4 generates &lt;unused&gt; tokens in infinite
  loop on Vulkan backend."* Same end-user-visible symptom (Gemma-4
  Vulkan infinite loops), still open. The async + mid-frame-submit
  pattern described above may be relevant if ggml's Vulkan backend
  uses similar CB pre-recording for MoE models. We have not audited
  the ggml Vulkan path; the writeup here is shared as a data point
  for that investigation.

## Source

- Fix commits: [`32695e3`](https://github.com/maeddesg/vulkanforge/commit/32695e3) (async/MoE) and [`1d904b7`](https://github.com/maeddesg/vulkanforge/commit/1d904b7) (V-race)
- Repo: <https://github.com/maeddesg/vulkanforge>
- Tag: `v0.4.2`

## Acknowledgements

The bug hunt that uncovered both of these spanned 29 numbered sprints
(53A–54I), most of which chased a different (red herring) explanation
based on a misreading of HuggingFace's `output_hidden_states` (which
includes the post-`model.norm` final RMSNorm — *not* the last decoder
layer's forward return). Sprint 54G untangled that
methodology-artifact; 54H located the async/MoE race condition; 54I
shipped the fix.
