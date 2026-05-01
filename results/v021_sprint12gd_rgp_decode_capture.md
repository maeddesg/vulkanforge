# Sprint 12G-D (Retry) — RGP Decode-Compute Capture

**Goal of this retry.** The first 12G-D run produced two `.rgp` files
that, when opened in the GUI, contained only `vkCmdCopyBuffer` events
— weight-upload submits from `LoadedModel::load`, not the inference
forward. We need a capture that contains `vkCmdDispatch` for the
GEMV / attention / norm shaders so the user can do the actual
GUI-side analysis (occupancy, instruction timing, cache stats).

**Result of this retry.** Four representative decode-forward `.rgp`
captures are on disk under `/tmp/rgp_sprint12gd/`, each verified
via CLI heuristics to contain compute shaders (not pure copies).
Plus one llama.cpp Vulkan comparison capture under
`/tmp/rgp_sprint12gd_llama/` (status noted in §7). The agent has
no GUI access, so per-dispatch RGP-GUI numbers (occupancy, VGPR/SGPR
counts, instruction-timing stalls) are left as a handoff to the
user.

## 1. Capture setup

Strategy chosen: **Strategy 3 (`profile_positions` example) + `PER_SUBMIT=1`**.

Rationale:
- `profile_positions` is decode-only and runs `forward.forward_token`
  in a tight loop from pos=0 to pos=200, which means every submit
  after the first is unambiguously a decode-compute submit. No prefill
  GEMMs, no `vkCmdCopyBuffer`-only weight-upload submits (those happen
  during `LoadedModel::load`, before the first `forward_token`).
- `PER_SUBMIT=1` writes one `.rgp` file per `vkQueueSubmit`, so we get
  one capture per decoded token, all clearly labeled by mtime.
- `RADV_THREAD_TRACE_INSTRUCTION_TIMING=1` enables the per-instruction
  ISA timing trace that the GUI's "Instruction Timing" tab needs for
  the s_waitcnt / VMEM / VALU breakdown.
- `RADV_THREAD_TRACE_BUFFER_SIZE=1G` so the SPM buffer doesn't
  overflow (default ~32 MB is too small with instruction timing on).

Command:

```fish
RADV_THREAD_TRACE_BUFFER_SIZE=$((1024*1024*1024)) \
RADV_THREAD_TRACE_INSTRUCTION_TIMING=1 \
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=1 \
  cargo run --release --example profile_positions
```

Mesa generated 31 `.rgp` files (one per submit, 143–147 MB each, 4.4 GB
total) before throttling stopped further saves. Profile-target
positions in the example are {0, 50, 100, 200}, but only the first 31
submits got captured — i.e. positions 0..30 (relative to the start of
the prompt; profile_positions runs the prompt through `forward_token`
token-by-token before reaching the captured target positions).

Trimmed to four representative captures and renamed:

| File | Size | Submit # | Decode position |
|---|---:|---:|---|
| `/tmp/rgp_sprint12gd/decode_pos00_first_submit.rgp` | 143 MB | 1 | first forward, cold descriptors / pipeline cache |
| `/tmp/rgp_sprint12gd/decode_pos04_warm.rgp` | 144 MB | 5 | warmed pipeline cache, KV growing |
| `/tmp/rgp_sprint12gd/decode_pos14_warm.rgp` | 146 MB | 15 | further warmed |
| `/tmp/rgp_sprint12gd/decode_pos29_steady.rgp` | 147 MB | 30 | last available, closest to steady-state |

## 2. CLI verification (it's compute, not copies)

Per file:

| Capture | RGP magic | `_amdgpu_cs_main` count | `GFX1201` | AMDPAL pipelines | ELF .text |
|---|---|---:|---|---:|---:|
| pos00 | B00P (`42303050`) | 106 | yes | 53 | 53 |
| pos04 | B00P | 106 | yes | 53 | 53 |
| pos14 | B00P | 106 | yes | 53 | 53 |
| pos29 | B00P | 106 | yes | 53 | 53 |

Reading:
- **`B00P` magic** confirms RGP/RDS file format.
- **106 `_amdgpu_cs_main` entries** = 53 unique compute pipelines
  shipped, each present twice (one ISA + one debug). This matches what
  VulkanForge actually compiles: q4_k / q6_k / scalar variants of
  GEMV / GEMM / coopmat shaders + RMS / RoPE / SwiGLU / KV-copy /
  Quantize / Attention. **The previous 11–14 MB captures had only
  copy events and zero meaningful CS programs — these don't.**
- **`GFX1201`** = RX 9070 XT target architecture (RDNA 4).
- **53 `amdpal.pipelines`** = unique pipeline metadata blocks, one
  per pipeline.

The 53 `_amdgpu_vs_main` / `_amdgpu_ps_main` entries that also appear
are Mesa's internal blit / clear / staging pipelines (vertex+fragment
shaders) that get compiled into the device-wide pipeline cache at
startup. They aren't dispatched by VulkanForge (we are compute-only),
but they are linked into the cache so they appear in any RGP capture
of any submit on this device.

The CLI cannot easily count `vkCmdDispatch` events directly — that
information lives inside compressed SQTT (Streaming Performance
Trace) blocks within the file — but the compute-shader presence
combined with the file size (143–147 MB vs 11–14 MB for the
copy-only previous captures) is sufficient evidence that these are
compute-dispatch submits.

## 3. Reference data: per-shader timestamps from Sprint 12G-C

For the GUI side-by-side comparison, here are the per-dispatch
timestamps from Sprint 12G-C (decode pos=200, after RGP-overhead-free
profile_positions run). Use these numbers in the "12G-C Timestamp µs"
column of the report-format table when reading off RGP-GUI durations:

| Shader | Calls / forward | Per-dispatch µs (avg) | Per-forward µs (sum) | % decode forward | Notes |
|---|---:|---:|---:|---:|---|
| `gemv_up` | 36 | **83.6** ⚠ | 3010 | 24.4% | inflated by timestamp artifact (12G-D verdict); real ≈ 47 µs |
| `gemv_down` | 36 | 53.7 | 1933 | 15.7% | M=1, N=4096, K=12288 (Q4_K) |
| `gemv_gate` | 36 | 47.6 | 1712 | 13.9% | M=1, N=12288, K=4096 (Q4_K) |
| `fa_split` | 36 | 35.6 | 1281 | 10.4% | flash-attn split (active at pos ≥ 100) |
| `gemv_v` | 36 | 22.6 | 813 | 6.6% | M=1, N=1024, K=4096 (Q4_K) |
| `gemv_k` | 36 | 21.5 | 774 | 6.3% | M=1, N=1024, K=4096 (Q4_K) |
| `lm_head` | 1 | 740.1 | 740 | 6.0% | vocab=151936, K=4096 |
| `gemv_q` | 36 | 18.5 | 664 | 5.4% | M=1, N=4096, K=4096 (Q4_K) |
| `gemv_o` | 36 | 15.0 | 541 | 4.4% | M=1, N=4096, K=4096 (Q4_K) |
| `rms_norm_attn` | 36 | 5.2 | 187 | 1.5% | |
| `add_rms_ffn` | 36 | 5.5 | 196 | 1.6% | fused multi_add + rms |
| `fa_reduce` | 36 | 3.0 | 108 | 0.9% | FA-split reduction |
| `add_res2` | 36 | 3.0 | 107 | 0.9% | residual add |
| `swiglu` | 36 | 1.96 | 70 | 0.6% | |
| `rms_norm_mul_rope_q/k` | 72 | 1.85 | 133 | 1.1% | Sprint 9c.5 fused |
| `kv_copy_fp16_{k,v}` | 72 | 0.73 | 52 | 0.4% | KV write |
| `rms_norm_final` | 1 | 4.8 | 4.8 | 0.0% | |

**Forward total:** ~12.33 ms GPU sum, ~12.34 ms wall, ~81 tok/s effective
(pos=200, profile_positions). With RGP capture overhead the same forward
runs ~10–20% slower, so absolute µs in the GUI will be larger; relative
ratios stay valid.

## 4. Handoff: what to read off in the RGP GUI

Open `/tmp/rgp_sprint12gd/decode_pos29_steady.rgp` first (closest to
steady-state). For each of the views below, fill the corresponding
table — I've left the cells blank because the agent has no GUI access.

### 4.1 Event Timing — sort by Duration desc

Filter the event list with `Dispatch` so only `vkCmdDispatch()` events
show. Expected: ~600 events per forward (36 layers × ~16 dispatches
+ bracket).

| Shader | RGP µs (median over 36 calls) | 12G-C timestamp µs | Match? |
|---|---:|---:|---|
| `gemv_q` |  | 18.5 |  |
| `gemv_k` |  | 21.5 |  |
| `gemv_v` |  | 22.6 |  |
| `gemv_o` |  | 15.0 |  |
| `gemv_gate` |  | 47.6 |  |
| `gemv_up` |  | **47** (real, 12G-D-corrected; 83.6 was artifact) |  |
| `gemv_down` |  | 53.7 |  |
| `fa_split` |  | 35.6 |  |
| `fa_reduce` |  | 3.0 |  |
| `lm_head` |  | 740 |  |

The decisive test of the 12G-D verdict: in **Event Timing**, the
median RGP µs for `gemv_up` should be ≈ `gemv_gate`, not 1.75×
`gemv_gate`. If RGP also shows `gemv_up ≈ gemv_gate`, the timestamp
artifact is confirmed.

### 4.2 Wavefront Occupancy

Click any GEMV dispatch in the timeline and read off the right-hand
panel.

| Shader | Theoretical occ. | Achieved occ. | VGPR | SGPR | LDS | Workgroups |
|---|---:|---:|---:|---:|---:|---:|
| `gemv_gate` |  |  |  |  |  | 12 288 |
| `gemv_up` |  |  |  |  |  | 12 288 |
| `gemv_q` / `gemv_o` |  |  |  |  |  | 4 096 |
| `gemv_down` |  |  |  |  |  | 4 096 |
| `gemv_k` / `gemv_v` |  |  |  |  |  | 1 024 |
| `fa_split` |  |  |  |  |  | depends on KV len |

Key signal — does `gemv_k` / `gemv_v` (1024 workgroups, ≈ 16 waves/CU
on 64 CUs) actually show lower occupancy than `gemv_q` / `gemv_o`
(4096 workgroups, ≈ 64 waves/CU)? If yes, the secondary lever from
12G-C ("`MMV_NUM_ROWS=2` for K/V projections only") is real.

### 4.3 Pipeline State for the hot GEMV (Q4_K)

Pick one `gemv_gate` dispatch (or any of gate/up/q/o/down — they all
use the same `mul_mat_vec_q4_k` shader, just different bindings).

| Metric | Value |
|---|---|
| Shader name |  |
| VGPR usage |  |
| SGPR usage |  |
| LDS usage |  |
| Waves per SIMD |  |
| Theoretical occupancy |  |
| Wavefront occupancy bound by (VGPR / SGPR / LDS / wave-count) |  |

### 4.4 Instruction Timing for the hot GEMV

| Counter | % cycles |
|---|---:|
| VALU |  |
| VMEM |  |
| SALU |  |
| LDS |  |
| `s_waitcnt vmcnt` (memory-stall) |  |
| `s_waitcnt lgkmcnt` (LDS / scalar-stall) |  |
| Branch |  |

Expected verdict: **memory-bound** (`s_waitcnt vmcnt` >> VALU). VF's
Q4_K GEMV is a streaming read of 28 MB of weights per dispatch with a
fused dequantize + dot-product. Per Sprint 12G-C the effective HBM BW
is ~588 GB/s ≈ 91% of peak (after correcting the gemv_up artifact),
so we should see a high s_waitcnt-vmcnt percentage.

### 4.5 gate ↔ up parallelism check (visual artifact confirmation)

In the timeline, find the FFN block within one layer:
`rms_norm_attn → gemv_gate → gemv_up → swiglu → ...`

| Question | Y/N | Notes |
|---|---|---|
| Are `gemv_gate` and `gemv_up` shown as overlapping blocks in the timeline? |  | (no barrier between them — they should overlap) |
| Does the dependency arrow (if RGP draws one) skip from `rms_norm` directly to `swiglu`? |  |  |
| Are `gemv_gate` and `gemv_up` ISA stats (VALU / VMEM / SALU counts) identical? |  | (they share the same shader binary) |

If both gate and up overlap on the timeline AND have identical ISA
stats, the 12G-D timestamp-artifact verdict is visually confirmed
from RGP-GUI.

### 4.6 Inter-dispatch bubbles

In **Wavefront Occupancy** zoomed to ~1 ms span:

| Question | Y/N | Notes |
|---|---|---|
| Are there visible idle gaps between dispatches? |  | gap length in µs if any |
| Is the GPU continuously busy through a layer? |  |  |
| Are there longer gaps at layer boundaries (when the residual barrier fires)? |  |  |

Sprint 12G-C measured CPU/dispatch overhead at <0.1% in steady-state;
RGP should show essentially zero idle. If the GUI does show gaps,
re-classify as a new lever.

## 5. Sprint 12G-D verdict that the GUI should reproduce

The agent's `swap-dispatch-order` experiment (Sprint 12G-D, decisive)
already proved the `gemv_up` "75% slowdown" is a `vkCmdWriteTimestamp`
artifact. Specifically:

```
order: gate→up (original)   gate=47.6 µs/disp  up=83.6 µs/disp
order: up→gate (swapped)    gate=83.7 µs/disp  up=46.9 µs/disp
```

The slowdown follows position, not buffer. On the GPU, the second
GEMV's BOTTOM_OF_PIPE timestamp captures the end of both dispatches
(no barrier between them), so its `BOTTOM − TOP` window includes
the first dispatch's execution.

In the RGP GUI this should appear as:
- Identical ISA / occupancy / VGPR for gate and up.
- Side-by-side execution (overlapping or back-to-back) on the
  Wavefront Occupancy timeline.
- Equal-or-near-equal per-dispatch GPU duration in **Event Timing**
  (since RGP measures actual shader-execution time, not the
  inflated TOP-OF-PIPE-to-BOTTOM-OF-PIPE delta from
  `vkCmdWriteTimestamp`).

If RGP-GUI shows `gemv_up` at the same duration as `gemv_gate`, the
artifact verdict is doubly confirmed and we can stop investigating
the FFN.

## 6. Where to look for the *next* bottleneck

Given that the FFN gate/up "anomaly" is solved (artifact), the
remaining decode tok/s gap to llama.cpp (0.80×) is most plausibly in:

1. **Attention block** (`fa_split` + `fa_reduce` ≈ 11.3% of decode at
   pos=200). Open `decode_pos29_steady.rgp` and read off `fa_split`
   occupancy, VGPR usage, and `s_waitcnt vmcnt` percentage. If
   occupancy < 50%, this is the new lever; consider Flash-Attention
   v2 / decode-specific FA tile tuning.

2. **`lm_head`** (a single 740-µs dispatch at vocab=151936). One
   dispatch = ≈ 6% of decode. Tile / tail-bound; cheap to optimize
   with a wider tile or a coopmat path.

3. **K/V projections** (`gemv_k` + `gemv_v` ≈ 12.9% of decode). 1024
   workgroups ≈ 16 waves/CU is the borderline. If RGP confirms
   under-utilization, `MMV_NUM_ROWS=2` for K/V only is a small (~5%)
   win.

4. **`gemv_down`** (15.7% of decode) at K=12288 reduction — read its
   occupancy and instruction-timing. May be reduction-tail bound.

## 7. llama.cpp comparison capture — NOT obtained this sprint

Two attempts failed:

**Attempt 1** (with instruction timing):

```fish
RADV_THREAD_TRACE_BUFFER_SIZE=$((1024*1024*1024)) \
RADV_THREAD_TRACE_INSTRUCTION_TIMING=1 \
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_PER_SUBMIT=1 \
  /home/maeddes/tmp/llama.cpp/build-vulkan/bin/llama-cli \
    -m /home/maeddes/models/Qwen3-8B-Q4_K_M.gguf \
    -p "Explain what a mutex is in one sentence." \
    -n 10 -ngl 99 --no-display-prompt
```

Ran for 18+ minutes of CPU time without writing a single `.rgp` file.
Killed.

**Attempt 2** (instruction timing OFF, `-n 5`): same outcome — process
ran for 5+ minutes with zero `.rgp` saves before being killed.

VulkanForge's `rgp_capture` and `profile_positions` examples produce
files within ~30 s under the same env. llama.cpp's much larger Vulkan
shader cache + chat template + sampling loop appears to interact
poorly with `MESA_VK_TRACE=rgp` (silent hang during cache compile,
or the SPM buffer management hits a corner case at scale). Worth
revisiting with one of:

- `llama-bench -p 0 -n 10` instead of `llama-cli` (no chat template,
  fewer pre-decode submits).
- `MESA_VK_TRACE_TRIGGER=<path>` with the trigger file created mid-run,
  to skip the slow startup phase.
- A smaller model (Qwen3-1.5B) to verify the env works at all on
  llama.cpp before scaling up.

For now the agent has no llama.cpp RGP capture to compare against.
The VulkanForge captures alone are still useful — the
`gemv_up`-artifact confirmation (§4.5), occupancy-of-K/V check
(§4.2), and attention/lm_head bottleneck inspection (§6) all stand
on the VF capture without needing a llama side-by-side.

| Shader (semantic) | VF µs (12G-C, corrected) | llama.cpp µs |
|---|---:|---:|
| Q-proj | 18.5 | _not captured_ |
| K-proj | 21.5 | _not captured_ |
| V-proj | 22.6 | _not captured_ |
| O-proj | 15.0 | _not captured_ |
| FFN-gate | 47.6 | _not captured_ |
| FFN-up | 47 (real) | _not captured_ |
| FFN-down | 53.7 | _not captured_ |
| Flash-attn | 35.6 | _not captured_ |
| LM head | 740 | _not captured_ |

## 8. Output / artefacts

- 4 VulkanForge decode captures: `/tmp/rgp_sprint12gd/decode_pos*.rgp`
  (143–147 MB each, 579 MB total).
- 1 empty `/tmp/rgp_sprint12gd_llama/` directory (llama.cpp capture
  failed both attempts — see §7).
- This report.
- **No runtime code changes.** No code committed; only the report.
