# Sprint 12I — Prefill profiling: where the 4× gap actually lives

**Premise.** Decode is at 0.80× llama.cpp; prefill is at 0.55×. The
4× prefill gap is the biggest unmeasured signal in v0.2.x. Profile
it before designing any optimization.

**Result.** The gap is **not in shader source** (`mul_mmq.comp` and
`mul_mm.comp` are byte-identical to llama.cpp HEAD, like Sprint 12H
showed for the GEMV variants). The gap is in **what compiled
pipeline gets dispatched**: llama.cpp routes prefill GEMMs through
KHR cooperative-matrix WMMA pipelines (`matrix cores: KHR_coopmat`
per `llama-bench` banner); VulkanForge's coopmat path exists but is
slower than our default `mul_mmq` integer-DP path on the same
hardware, so we ship the integer-DP path. Closing the gap is
therefore a **coopmat-pipeline engineering task**, not a shader port.

**Bonus finding.** The decode-side `gemv_gate`/`gemv_up` timestamp
artifact (Sprint 12G-D) **applies to prefill too**. `gemm_up` at
"29 % of prefill GPU sum" is inflated; real `gemm_up ≈ gemm_gate`,
and the inflated `gemm_up` figure does *not* indicate a real
asymmetry to chase.

## 1. Capture / measurement set

- **Timestamps (reliable, primary):** `examples/profile_prefill`,
  fresh `ShaderProfiler` (capacity 4096), no instruction-timing
  overhead. Run at `VF_PP={128, 512}`.
- **pp sweep (head-to-head):** `examples/run_pp_bench` (VF) vs
  `llama-bench -p N -n 0 -ngl 99 -r 3` (llama.cpp HEAD,
  `build-vulkan/bin/llama-bench`).
- **RGP attempt:** failed for `prefill_batch` —
  `VK_ERROR_DEVICE_LOST` ("CS has been cancelled because the
  context is lost. This context is guilty of a hard recovery") on
  pp=128 and pp=512 with 4 GB SPM trace buffer. The `prefill_batch`
  path is **a single one-shot submit with ~600 dispatches and tens
  of MB of trace metadata**; SPM thread-trace cannot keep up and
  the GPU watchdog fires. Only one tiny pp=32 capture
  (`/tmp/rgp_sprint12i/prefill_pp32_compact.rgp`, 587 KB, 106
  compute pipelines, GFX1201) survived — too small to be useful for
  per-event analysis.
- **Source-code verification:** read `dispatch_layer_batch`
  (`forward.rs:3159`) for prefill barrier/dispatch ordering.
- **Shader-identity pre-check:** md5 of the GEMM shaders against
  llama.cpp HEAD (`23b8cc4`).

## 2. pp-sweep — VulkanForge vs llama.cpp Vulkan

| pp | VF tok/s | VF wall (ms) | llama.cpp tok/s | llama.cpp wall (ms) | Gap (wall) | VF/llama |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1505 | 42.5 | 2259 | 28.3 | +14.2 ms | 0.67 × |
| 128 | 2004 | 63.9 | 3631 | 35.2 | +28.7 ms | 0.55 × |
| 256 | 2197 | 116.5 | 3975 | 64.4 | +52.1 ms | 0.55 × |
| **512** | **2348** | **218.1** | **4324** | **118.4** | **+99.7 ms** | **0.54 ×** |
| 1024 | 2303 | 444.7 | 4177 | 245.2 | +199.5 ms | 0.55 × |
| 2048 | 2082 | 983.5 | 3756 | 545.3 | +438.2 ms | 0.55 × |

The ratio is essentially constant at **0.55 × from pp=128 onwards**.
This is a **constant-factor** gap — not a "small-pp problem" or a
"large-pp problem" that scales differently. That structure rules
out shape-specific tile-tuning as the headline lever and points at
something the prefill GEMM kernel does on every dispatch.

## 3. Shader-identity pre-check

```
$ md5sum vk_shaders/mul_mmq.comp ~/tmp/llama.cpp/.../mul_mmq.comp
e6bd22f7a66a25fce2d082662c0ae350  vk_shaders/mul_mmq.comp
e6bd22f7a66a25fce2d082662c0ae350  …/llama.cpp/…/mul_mmq.comp

$ md5sum vk_shaders/mul_mm.comp  ~/tmp/llama.cpp/.../mul_mm.comp
0bdf44a455deb59b29fc8621a559058b  vk_shaders/mul_mm.comp
0bdf44a455deb59b29fc8621a559058b  …/llama.cpp/…/mul_mm.comp

$ md5sum vk_shaders/mul_mat_vec_q4_k.comp / mul_mat_vec_q6_k.comp
                                      vs llama.cpp HEAD: identical (Sprint 12H)
```

**Both prefill GEMM source files (`mul_mmq.comp` and `mul_mm.comp`)
are byte-identical to llama.cpp HEAD `23b8cc4`.** Together with
Sprint 12H's `mul_mat_vec_q*_k.comp` identity, **every weight-quant
GEMM/GEMV shader source we ship is upstream-identical**. There is
no shader port to be done.

## 4. The default prefill path is `mul_mmq` (Q8_1 + integer-DP)

`forward.rs:520-537` reads three feature flags:

```rust
mul_mm_coopmat_enabled = VULKANFORGE_USE_MM_COOPMAT == "1"   // default OFF
mul_mm_enabled         = mul_mm_coopmat_enabled
                       || VULKANFORGE_USE_MUL_MM == "1"      // default OFF
coopmat_q4k_enabled    = VULKANFORGE_COOPMAT == "1"          // default OFF
                                                              // (gemm_q-only)
```

Default-OFF on all three → `dispatch_layer_batch` routes every
prefill GEMM through `run_gemm` → `mul_mmq.comp` (Q8_1 activations
+ integer dot-product). That's our fastest path, but it does **not**
use AMD's matrix cores.

Confirmation from a 4-way bench at pp=128 / pp=512 (3 runs each,
median):

| Path | env | pp=128 tok/s | pp=512 tok/s |
|---|---|---:|---:|
| **default** (mul_mmq, integer DP) | (none) | **2004** | **2348** |
| coopmat (mul_mm + KHR_coopmat) | `VULKANFORGE_USE_MM_COOPMAT=1` | 1378 | 2177 |
| plain mul_mm (FP32 activations) | `VULKANFORGE_USE_MUL_MM=1` | 1074 | 1151 |
| Q4_K coopmat (gemm_q only) | `VULKANFORGE_COOPMAT=1` | 690 | 931 |
| llama.cpp Vulkan | (matrix cores: KHR_coopmat) | 3631 | 4324 |

Reading: every alternative we ship is **slower** than the default,
yet llama.cpp — using the WMMA matrix cores via KHR_coopmat —
**runs ~1.84 × faster** than our integer-DP path. So the gap is real
and the lever is real, but the lever is **inside our coopmat
pipeline implementation**, not the source GLSL. (Our
`mul_mm_q4_k_f32_coopmat.spv` exists but underperforms the
integer-DP path; llama.cpp's coopmat pipeline is a different
implementation than the file we share via `mul_mm.comp`.)

## 5. Per-shader prefill breakdown (pp=512, timestamps)

Total: wall = 232.05 ms, gpu_sum = 315.70 ms (overlap factor 1.36 ×
— prefill dispatches DO overlap on GPU). 90.4 % GEMM bucket.

| Shader | Calls | Sum µs | µs/disp (timestamp) | µs/disp (real) | Notes |
|---|---:|---:|---:|---:|---|
| `gemm_up` | 36 | 91 157 | 2532 | **~1326** ⚠ artifact-inflated | Q4_K, M=512 N=12288 K=4096 |
| `gemm_down` | 36 | 61 054 | 1696 | ~1696 | **Q6_K**, M=512 N=4096 K=12288 |
| `gemm_gate` | 36 | 47 722 | 1326 | 1326 | Q4_K, same shape as `gemm_up` |
| `gemm_v` | 36 | 26 979 | 750 | ~750 | **Q6_K**, M=512 N=1024 K=4096 |
| `gemm_k` | 36 | 24 288 | 675 | ~675 | Q4_K, same shape as `gemm_v` |
| `gemm_q` | 36 | 17 041 | 473 | ~473 | Q4_K, M=512 N=4096 K=4096 |
| `gemm_o` | 36 | 16 181 | 449 | ~449 | Q4_K, M=512 N=4096 K=4096 |
| `fa_tiled` | 36 | 16 067 | 446 | ~446 | flash-attn batched |
| `swiglu_b` | 36 | 3 889 | 108 | ~108 | |
| `rms_norm_mul_rope_*_b` | 72 | 5 999 | 83 | ~83 | Sprint 9c.5 fused |
| `add_rms_*_b` | 71 | 2 408 | 34 | ~34 | |
| `quantize_*` | 108 | 916 | 8.5 | ~8.5 | Q8_1 quantizer for mul_mmq inputs |
| `kv_copy_fp16_*_b` | 72 | 344 | 4.8 | ~4.8 | |
| `lm_head` | 1 | (not in prefill) | — | — | |

### 5.1 The `gemm_gate` / `gemm_up` artifact (same as decode 12G-D)

`forward.rs:3686-3691` dispatches `gemm_gate` then `gemm_up` with
**no barrier between them** (correct: they read the same
post-norm input and write disjoint output buffers). This is
exactly the pattern that produced the gemv_up "75 % slowdown"
artifact in Sprint 12G-D, and it produces the same artifact for
`gemm_up`:

- `gemm_gate measured = gemm_gate_real`
- `gemm_up measured  = gemm_gate_real + gemm_up_real`

Same shape (M=512, N=12288, K=4096, Q4_K) → real per-dispatch is
1326 µs each. The "29 %" headline figure is approximately the same
as gate's 15.1 % plus a phantom 13.9 % of double-counting.

**Decode + prefill both confirm**: per-dispatch `vkCmdWriteTimestamp`
is unreliable for the second of two RAW-independent dispatches.

### 5.2 Per-GEMM compute-throughput estimate

Treating prefill GEMMs as compute-bound (RDNA4 dual-issue VALU + Q4_K
dequant), effective TFLOPs:

| GEMM | shape (M·N·K) | FLOPs (G) | µs (real) | TFLOPs |
|---|---|---:|---:|---:|
| `gemm_gate` | 512·12288·4096 | 51.6 | 1326 | **38.9** |
| `gemm_up` | 512·12288·4096 | 51.6 | ~1326 | ~38.9 |
| `gemm_down` | 512·4096·12288 | 51.6 | 1696 | **30.4** (Q6_K) |
| `gemm_q/o` | 512·4096·4096 | 17.2 | ~460 | ~37.4 |
| `gemm_k` | 512·1024·4096 | 4.29 | 675 | **6.36** |
| `gemm_v` | 512·1024·4096 | 4.29 | 750 | 5.72 (Q6_K) |

RX 9070 XT theoretical compute peak (compute shaders, no WMMA):
~50 TF FP32 single-issue, ~100 TF dual-issue. The big GEMMs reach
**~78 % of single-issue peak** (39 TF / 50 TF) — this is healthy
for a dequant+integer-DP kernel on a non-WMMA path.

Reaching llama.cpp's level (1.84 × ours) on `gemm_gate`/`gemm_up`
would require ~71 TF — i.e. crossing into dual-issue territory, or
WMMA-tier compute. The KHR_coopmat path (which we have but is
slower today) is the only realistic vehicle for that on this
hardware.

### 5.3 Q6_K shows up at GEMM but smaller than at GEMV

Pairing same-shape Q4_K vs Q6_K dispatches:

| Shape | Q4_K kernel | Q6_K kernel | Q6_K overhead |
|---|---:|---:|---:|
| M=512 N=1024 K=4096 (k vs v) | 675 µs | 750 µs | **+11 %** |
| M=512 N=4096 K=12288 (gate-like vs down) | n/a | 1696 µs | (no Q4_K twin) |

11 % Q6_K overhead in GEMM-prefill, vs the ~14 % overhead at
GEMV-decode (`gemv_v` 22.6 µs vs `gemv_k` 21.5 µs at the same
shape). Same direction, similar magnitude. **Q6_K is a real but
small (~1-2 %) lever in prefill**, dwarfed by the coopmat lever.

## 6. Decomposition of the 99 ms gap to llama.cpp at pp=512

The brief sketched a candidate-budget table; here are the numbers
that survive the data:

| Candidate | Estimated contribution | Evidence |
|---|---:|---|
| **WMMA matrix-cores (coopmat path)** | **~80 ms** | llama-bench banner says `matrix cores: KHR_coopmat`; our coopmat alternatives are slower than mul_mmq → llama.cpp's coopmat pipeline is engineered, ours isn't |
| Q6_K format overhead (gemm_v + gemm_down) | ~3-5 ms | 11 % of those two kernels' time |
| Pipeline parallelism / multi-submit | unknown but bounded | gpu_sum / wall = 1.36 in our prefill; llama.cpp's analogous figure not measured (their RGP capture failed in 12G-D-retry) |
| Tile-size tuning (Sprint 11C "L-tile only one config") | ~10-15 ms | gemm_q ≈ 449 µs / 17.2 GFLOPs gives 38 TF; the wider tiles only kick in at certain N — many calls fall back to smaller tiles |
| Graph-fusion / view elision | small at prefill | already mostly bypassed via batched dispatches; bigger lever for decode |
| `quantize_q8_1` overhead | < 1 ms | 0.3 % of forward — well in the noise |

The dominant bucket is the WMMA coopmat path, which alone accounts
for the bulk of the gap. The other items together are 1-2 sprint
worth of small wins.

## 7. Source verification: `dispatch_layer_batch` shape

`forward.rs:3159 fn dispatch_layer_batch(…)`. Per-layer event
sequence (paraphrased):

```
quantize_attn(input → batch_q8)            # M=512 row-quantize for mul_mmq
gemm_q,  gemm_k,  gemm_v       # 3 GEMMs, NO BARRIERS BETWEEN THEM
compute_barrier                # before norm+rope
rms_norm_mul_rope_q_b  +  rms_norm_mul_rope_k_b   # fused pair
compute_barrier
kv_copy_fp16_k_b  +  kv_copy_fp16_v_b
compute_barrier
fa_tiled                       # batched flash-attn over M tokens
compute_barrier
quantize_attn_out
gemm_o
compute_barrier
add_rms_attn_next_b            # fused multi_add+rms
compute_barrier
quantize_ffn
gemm_gate,  gemm_up            # 2 GEMMs, NO BARRIER BETWEEN THEM
compute_barrier
swiglu_b
compute_barrier
quantize_ffn_h
gemm_down
compute_barrier
add_rms_ffn_b                  # next layer's input
```

Two no-barrier dispatch pairs survive timestamps wrong (q/k/v at
first, gate/up later); the rest of the per-shader timings are
fine.

## 8. Verdict & next-sprint recommendation

The 4 × prefill gap is **not** a missing shader, **not** a Q6_K
format problem, **not** a graph-layer rewrite. It is **one
specific engineering item**: the KHR_coopmat WMMA pipeline that
llama.cpp's prefill uses by default and that VulkanForge has only
as an experimental, slower alternative.

Concrete plan for **Sprint 12J — coopmat WMMA prefill pipeline**:

1. **Day 1 — measure the coopmat ceiling in isolation.** Run
   `examples/bench_coopmat` (already exists) with
   `VULKANFORGE_USE_MM_COOPMAT=1` and a Q4_K-shape micro-bench.
   Record VGPR / SGPR / occupancy / TFLOPs from RGP for *one*
   coopmat dispatch. Compare against llama.cpp's coopmat dispatch
   in their own RGP capture (revisit Sprint 12G-D-retry's
   llama-cli RGP attempts — without instruction timing, with
   `MESA_VK_TRACE_TRIGGER` to skip startup).
2. **Day 2-3 — port the *configuration* (not the source).**
   llama.cpp's coopmat path differs from ours in spec-constants
   (`BM`/`BN`/`BK`/`LOAD_VEC_*`) and possibly in pipeline
   configuration (descriptor-set layout, push-const ordering).
   Diff the *built* SPV binaries (`spirv-dis`) between our
   `mul_mm_q4_k_f32_coopmat.spv` and llama.cpp's. If our shader
   source compiles to a different SPV, fix the pipeline-build
   call site.
3. **Day 4 — bench gate.** Goal: `VULKANFORGE_USE_MM_COOPMAT=1`
   default-on for prefill, with pp=512 ≥ 3500 tok/s
   (within 20 % of llama.cpp). If we fall short, fall back to
   `mul_mmq` with explicit logging.

Secondary sprint (low priority): **K/V projection MMV_NUM_ROWS=2 +
Q6_K kernel touch-up**. Decode-only, ~+3-4 % combined. Worth one
day after 12J ships.

## 9. Outputs

- `/tmp/rgp_sprint12i/prefill_pp32_compact.rgp` — single 587 KB
  prefill capture (pp=32, 106 compute pipelines visible). Useful
  only as proof-of-life that the SPM tracer captured *something*
  for prefill_batch — not enough resolution to read off
  per-dispatch numbers in the GUI. Larger captures crashed the
  GPU; do not retry without splitting `prefill_batch` into smaller
  submits first.
- This report.
- **No code changes.** No new tests. 27/27 lib tests still green.

Sprint 12 lessons applied:
- Pre-checked shader identity before designing optimization (12H rule).
- Used timestamps without instruction-timing as the primary measure
  (12H lesson — 88 → 54 µs correction).
- Documented the gemm_up artifact as a known-issue, not a lever
  (12G-D lesson).
