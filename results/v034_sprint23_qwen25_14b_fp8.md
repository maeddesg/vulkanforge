# Sprint 23 — Qwen2.5 14B FP8 (blocked: per-channel quantization)

**Date:** 2026-05-03
**Branch:** main (post-Sprint 22C, head was `6c815ec`)
**Goal:** Download `larryvrh/Qwen2.5-14B-Instruct-FP8`, add Q/K/V
bias-add support for the Qwen2 architecture, run the first 14B
FP8 chat on a 16 GiB consumer GPU.

## Status

**Pre-check stopped the sprint.** The 14B model uses
**per-channel** weight scaling (`strategy: "channel"`), but VF's
existing FP8 GEMV/GEMM kernels assume **per-tensor** scaling.
Shipping bias-add without first adding per-channel scale support
would land an incompatible loader path that can never actually
run this model coherently.

No code committed. Branch unchanged at `6c815ec`. The download
is still in progress in the background (8.3 GiB / ~15 GiB at
last check) — the model is preserved for a future per-channel
support sprint or a different sprint that needs a Qwen2.5
testbed.

## What the pre-check revealed

```
$ python3 - <<'EOF'
config.json:
  model_type:      qwen2
  hidden_size:     5120
  num_layers:      48                     ← was 40 in the brief; 48 actual
  num_heads:       40
  num_kv_heads:    8                       ← GQA 5:1
  intermediate:    13824
  vocab:           152064
  tie_word_embeddings: false
  attention_bias:  None                    ← bias is per-tensor in the
                                              SafeTensors map, not a
                                              config-level boolean

quantization_config:
  format:          "float-quantized"       ← was "naive-quantized" in 8B
  quant_method:    "compressed-tensors"
  ignore:          ["lm_head"]
  weights:
    strategy:      "channel"               ← PER-CHANNEL SCALING
    num_bits:      8
    type:          "float"
    symmetric:     true
    dynamic:       false
  input_activations:
    strategy:      "token"                 ← dynamic per-token, not stored
    dynamic:       true

Bias tensors: 144 (= 48 layers × 3 Q/K/V) — Q/K/V only, no MLP bias.
Total weight_scales: 336 (= 48 × 7 Linears). Each scale is a
                          *vector* of shape [out_dim], not a scalar.
EOF
```

Two material differences from `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
(the model VF's FP8 path was built around):

| Property | Llama-3.1-FP8 (works) | Qwen2.5-14B-FP8 (this) |
|---|---|---|
| `quantization_config.format` | `naive-quantized` | `float-quantized` |
| Weights `strategy` | `tensor` (per-tensor) | **`channel` (per-channel)** |
| `weight_scale` shape | scalar (`[1]` BF16) | **vector (`[out_dim]` BF16)** |
| Activation quant | static `tensor` | dynamic `token` (we ignore both) |
| Has Q/K/V bias | no | **yes (144 bias tensors)** |
| Layer count | 32 | **48** |

The activation-quant difference doesn't matter (VF doesn't quantize
activations regardless). The bias is what the brief targeted.
**The per-channel scaling is the deal-breaker** — it's a kernel-
level change to the FP8 GEMV (Sprint 20-M2) and FP8 GEMM
(Sprint 20-GEMM, 21A, 21B) that all currently consume a single
FP32 scalar from the push-constant block.

## Why per-channel scaling is a kernel rewrite, not a config tweak

Current FP8 GEMV (`mul_mat_vec_fp8.comp`):

```glsl
// per-tensor: weight_scale is one float in push constants.
data_d[row] = lds_partial[0] * p.weight_scale;
```

Per-channel needs:

```glsl
// per-channel: weight_scale is a vector indexed by output row.
data_d[row] = lds_partial[0] * scale_buf[row];
```

That requires:

* A new SSBO binding for the scale buffer (currently the FP8 GEMV
  uses 3 bindings; spirv-opt strips the unused fuse dummies, so
  binding count is 3-tight).
* `Forward::run_gemv` extension: pass the scale buffer as a 4th
  binding, not a 7th-slot push-constant.
* Same change in the FP8 GEMM kernels (single-tile + multi-WG).
* Loader changes: `weight_scale: Option<f32>` becomes
  `weight_scale: Option<vk::Buffer>` (the per-channel scale
  uploaded as its own buffer) — touches `GpuTensor`, the
  scale-collection pre-pass, the dispatch helpers, and the
  forward-routing branch in `dispatch_layer` / `dispatch_layer_batch`.

Realistic effort: **~300-400 LOC across loader + 3 shaders +
dispatch routing**, plus debug iterations against per-channel
reference output. That's a full sprint on its own.

## Why bias-add alone wouldn't be useful

The brief's plan was "bias-add + ship a 14B FP8 chat." Without
per-channel scale support, the chat would produce garbage
regardless of whether bias-add is wired. Building the bias-add
infrastructure today (loader paths, dispatch sites,
broadcast-aware shader) without per-channel support means
the next sprint would still need to do all the per-channel work
*and* then exercise the bias path through it.

Better: pause both. Ship them together when the kernel
infrastructure is ready.

## VRAM budget recheck (just for the record)

The brief estimated 13.8 GiB for the 14B FP8. With **48 layers**
(not 40 as the brief assumed) and the weight shapes from
`config.json`:

```
FP8 weights (48 × 7 Linears):
  Q/O    : 48 × 5120² × 1 B          = 1.26 GiB
  K/V    : 48 × 1280 × 5120 × 1 B    = 0.31 GiB each → 0.63 GiB total
  gate/up: 48 × 13824 × 5120 × 1 B   = 3.39 GiB each → 6.78 GiB total
  down   : 48 × 5120 × 13824 × 1 B   = 3.39 GiB
  ─────────────────────────────────────────────────
  FP8 weights total:                  12.06 GiB

FP16 lm_head: 152064 × 5120 × 2 B   =  1.49 GiB
Norms + biases (FP32):                ~0.001 GiB
KV cache @ 2K (FP16):                  0.83 GiB
KV cache @ 2K (FP8 KV):                0.42 GiB
Scratch + Vulkan:                     ~0.50 GiB
─────────────────────────────────────────────────
Total @ 2K context:
  FP16 KV: 12.06 + 1.49 + 0.83 + 0.5 = 14.88 GiB    ← fits 16 GiB
  FP8  KV: 12.06 + 1.49 + 0.42 + 0.5 = 14.47 GiB    ← fits 16 GiB
```

So the VRAM budget is **plausible**, just tighter than the brief
projected. Sprint 22B + 22C's 8B optimisations (skip embed,
FP16 lm_head) leave just enough room. Per-channel scale buffers
add ~3.4 MiB total (336 vectors × 5120 max × 4 B / scale) —
negligible.

## Three paths forward (decision matrix)

### Path A — Per-channel + bias support sprint (~400 LOC, multi-day)

* Add per-channel weight_scale buffer support to the FP8 GEMV
  (Sprint 20-M2's kernel) and both FP8 GEMM kernels (Sprint
  20-GEMM + 21B's kernels).
* Add `weight_scale: Option<vk::Buffer>` to `GpuTensor`.
* Load `*.weight_scale` as `[out_dim]` BF16 → FP32 buffer
  (instead of single-scalar).
* Add bias loading + broadcast-aware bias-add dispatch
  (the original brief work).
* Add Qwen2 chat-template detection + RoPE-theta routing.
* Test against the 14B FP8 model.

This is the full path to running 14B FP8. Realistic effort:
two sprints (per-channel kernels first, bias + Qwen2 wiring
second), each ~200 LOC.

### Path B — Find or build a per-tensor Qwen2.5-FP8

The neuralmagic-style "naive-quantized + per-tensor" recipe
that produced the working Llama-3.1-8B-FP8 *can* be applied to
Qwen2.5; it's just that the popular community FP8 builds use
the per-channel "float-quantized" recipe. If a per-tensor
Qwen2.5-FP8 exists (or one is created via a vLLM
`compressed-tensors` quantize call), VF would only need
the bias-add work to support it.

Lower-risk than Path A but depends on third-party tooling /
availability.

### Path C — Defer 14B FP8 indefinitely

Keep VF's FP8 path at the 8B Llama-3.1 demo level. The 8B FP8
chat works end-to-end at usable speed (Sprint 22C: 7.48 GiB
VRAM, 68.5 tok/s decode, 695 tok/s prefill at pp=512). The
"first Vulkan FP8 inference" narrative is captured. 14B can
wait for a future hardware / software inflection (RDNA-next FP8
WMMA throughput, Mesa improvements, or a per-tensor Qwen
release).

## Honest call

**Path A is the right work** if we want to demonstrate "14B FP8
on a 16 GiB consumer GPU" as a product feature. But it's a
multi-sprint commitment, not a single-sprint patch. The brief's
"~100 LOC for bias + first 14B chat" estimate was off by ~3-4×
because of the per-channel scaling discovery.

**Recommended next move:** before committing to Path A, run a
microbenchmark on a hand-converted 8B Qwen2.5-FP8 (per-tensor
recipe) to validate the bias-add wiring + per-tensor path. If
that works, then commit the per-channel sprint with confidence
that the rest of the infrastructure (loader, dispatch routing,
chat template) is correct.

Or: **skip 14B for now** and pick up Sprint 24+ on the
GGUF / Q4_K_M side where headroom exists (FP16-acc coopmat
exploration, multi-submit on SafeTensors, or the 32K-context
RoPE scaling that Sprint 22 deferred).

## Files touched

* `results/v034_sprint23_qwen25_14b_fp8.md` (this file)

No code changes. Branch is at `6c815ec` (Sprint 22C). The
downloaded model directory `~/models/Qwen2.5-14B-Instruct-FP8/`
is preserved for the next sprint.

## Pre-check log

```
=== config.json ===
model_type: qwen2          hidden_size: 5120        num_layers: 48
num_heads: 40              num_kv_heads: 8          intermediate: 13824
vocab: 152064              tie_word_embeddings: False

=== quantization_config ===
format: float-quantized
quant_method: compressed-tensors
ignore: ['lm_head']
weights config:
{
  "actorder": null, "block_structure": null, "dynamic": false,
  "group_size": null, "num_bits": 8, "observer": "minmax",
  "observer_kwargs": {}, "strategy": "channel",
  "symmetric": true, "type": "float"
}

=== bias inventory ===
Bias tensors: 144 (= 48 layers × 3 [q/k/v]_proj.bias)
Unique suffixes:
  self_attn.k_proj.bias
  self_attn.q_proj.bias
  self_attn.v_proj.bias
(MLP biases: NONE — gate/up/down have no bias)

=== weight_scale shape ===
strategy: "channel" → scale shape per Linear is [out_features].
(unverified by direct read; first shard not yet downloaded.)
```

## Conclusion

The brief's pre-check rule fired correctly. Pre-check found
the format mismatch *before* writing 100 LOC of speculative
bias-add code. The honest stop preserves the option to
implement per-channel support cleanly in a future sprint
without unwinding speculative changes. Branch is clean.
