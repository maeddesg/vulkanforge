# Sprint 12G-D — RGP GUI Screenshot Analysis

**Capture inspected.** `decode_pos29_steady.rgp` (147 MB), produced by
Sprint 12G-D Retry from `examples/profile_positions` under
`MESA_VK_TRACE=rgp` + `RADV_THREAD_TRACE_INSTRUCTION_TIMING=1`.
Single forward-token submit, 53 unique compute pipelines, 12.4 ms
of GPU time.

**Screenshots used:**

| File (`results/`) | Tab | Selected event |
|---|---|---|
| `vr-v0.2.1-wqavefront.png` | Wavefront Occupancy | full timeline |
| `vr-v0.2.1-event-timing.png` | Event Timing | full event list |
| `vr-v0.2.1-pipelin-state-event20.png` | Pipeline State | event 20 — `gemv_gate` |
| `vr-v0.2.1-instruction-timing-event25.png` | Instruction Timing | event 25 — `gemv_down` |
| `most-expensive-events.png` (older) | Most-Expensive Events | top dispatches (gemv_down × 36) |
| `pipelines.png` (older) | Pipelines | per-pipeline rollup |
| `vr-v0.2.1-instruvtion-timing.png` | Instruction Timing | small dispatch (rms_norm) |
| `vr-v0.2.1-pipelin-state.png` | Pipeline State | barrier event (no pipeline) |

## 1. Dispatch sequence — RGP events ↔ `forward.rs` correlation

Reading the Event Timing screenshot together with `forward.rs:1339-1551`
(`dispatch_layer`), one decoder layer maps to ~26 RGP events
(dispatch + barrier). The user pre-extracted µs values; below they are
mapped to specific source-line numbers for Qwen3-8B (`has_qk_norm =
true`, `is_fp16` KV cache):

| Event | Dims | µs | Shader / op | `forward.rs` line |
|---:|---|---:|---|---|
| 0 | (1,1,1) | 8.7 | `rms_norm_attn` | 1366 |
| 1 | barrier | 3.9 | `[hidden_norm]` | 1370 |
| 2 | (4096,1,1) | 28.9 | `gemv_q` (M=1, N=4096, K=4096, Q4_K) | 1381 |
| 3 | (1024,1,1) | 17.5 | `gemv_k` (M=1, N=1024, K=4096, Q4_K) | 1384 |
| 4 | (1024,1,1) | 16.4 | `gemv_v` (M=1, N=1024, K=4096, Q6_K) | 1387 |
| 5 | barrier | **31.1** | `[q_buf, k_buf]` (Q/K → norm+rope) | 1392 |
| 6 | (8,1,1)¹ | small | `rms_norm_mul_rope_q` (n_heads tile, fused) | 1406 |
| 7 | (8,1,1)¹ | small | `rms_norm_mul_rope_k` | 1412 |
| 8 | barrier | small | `[k_buf, v_buf]` (→ kv_copy) | 1426 |
| 9 | (8,1,1)¹ | small | `kv_copy_fp16_k` | 1440 |
| 10 | (8,1,1)¹ | small | `kv_copy_fp16_v` | 1443 |
| 11 | barrier | 1.3 | inline kv_bar (TRANSFER+COMPUTE → COMPUTE) | 1462 |
| 12 | (32,4,1) | 49.5 | `fa_split` (32 q-heads × 4 KV splits) | (in `run_scalar_attn` 1474) |
| 13 | (32,1,1) | small | `fa_reduce` (32 q-heads, scalar reduce) | (in `run_scalar_attn`) |
| 14 | barrier | small | `[attn_out]` | 1477 |
| 15 | barrier | 2.3 | (kv_bar's pair / lingering) | — |
| 16 | (4096,1,1) | 25.3 | `gemv_o` (M=1, N=4096, K=4096, Q4_K) | 1482 |
| 17 | barrier | **21.4** | `[input, o_buf]` → add_rms_ffn | 1487 |
| 18 | (1,1,1) | small | `add_rms_ffn` (fused multi_add+rms) | 1494 |
| 19 | barrier | small | `[hidden_norm]` → gate/up | 1503 |
| 20 | (12288,1,1) | **66.0** | `gemv_gate` (M=1, N=12288, K=4096, Q4_K) | 1510 |
| 21 | (12288,1,1) | **65.7** | `gemv_up` (M=1, N=12288, K=4096, Q4_K) | 1513 |
| 22 | barrier | small | `[gate_buf, up_buf]` → swiglu | 1518 |
| 23 | (192,1,1)¹ | small | `swiglu` (12288 / 64 = 192 WGs) | 1523 |
| 24 | barrier | small | `[ffn_hidden]` → gemv_down | 1530 |
| 25 | (4096,1,1) | **88.2** | `gemv_down` (M=1, N=4096, K=12288, **Q6_K**) | 1535 |
| 26 | barrier | **83.5** | `[res1, ffn_out]` → add_res2 | 1540 |
| 27 | (64,1,1)¹ | small | `add_res2` (4096 elems / 64 thread group) | 1543 |
| 28 | barrier | small | end-of-layer `[output]` | 1550 |

¹ Some dim values are inferred from shape arithmetic; the screenshot
text resolution doesn't render every dim cleanly. Confirm in GUI by
clicking the event.

**Cross-checks:**
- Events 20+21 (gemv_gate, gemv_up) at **66.0 / 65.7 µs** are the RGP
  visual confirmation of the Sprint 12G-D verdict: identical work,
  identical duration. The 12G-C `vkCmdWriteTimestamp` reading of
  47.6 / 83.6 µs was the artifact; RGP measures actual shader
  execution and shows them equal.
- Event 25 (gemv_down) at 88.2 µs uses the **Q6_K** shader
  (`mul_mat_vec_q6_k.comp`, line 1532 comment confirms), not Q4_K.
  K=12288 (the FFN intermediate) is the largest reduction in the
  decode forward — naturally the longest single shader.
- Event 5 (post-Q/K/V barrier) at 31.1 µs is heavy because three
  destination buffers (q/k/v) just got large fan-out writes from
  three GEMVs that overlapped in flight; the barrier waits for *all
  three* drains.

## 2. Pipeline State analysis (event 20 — gemv_gate)

From `vr-v0.2.1-pipelin-state-event20.png`:

| Property | Value |
|---|---|
| Total thread groups | 12,288 |
| Thread group dimensions | (64, 1, 1) |
| Total wavefronts | 12,288 (one wave64 per workgroup) |
| Total threads | 786,432 |
| Average wavefront duration | **9.889 µs** |
| Wavefront mode | wave64 |
| Vector registers (VGPR) | 48 (48 allocated) |
| Scalar registers (SGPR) | **108 (128 allocated)** |
| LDS bytes per thread group | 1024 |
| **Theoretical occupancy** | **13 / 16 wave64 per SIMD** |
| Occupancy bound by | **scalar register usage** |
| RGP hint | "If you reduce scalar register usage by 64 you could run another wavefront" |

### Where the 108 SGPRs come from

`mul_mat_vec_base.glsl:17-39` — one push-constant block of **13 uint
scalars**:

```glsl
ncols, stride_a, stride_b, stride_d,
batch_stride_a, batch_stride_b, batch_stride_d,
fusion_flags, base_work_group_y,
ne02, ne12, broadcast2, broadcast3
```

Plus 5 buffer bindings (`forward.rs:1828-1834`):

```rust
(0, weights, …), (1, input, …), (2, output, …),
(3, fuse0, …),  (4, fuse1, …)
```

Each buffer descriptor (V#) is 4 SGPRs on RDNA, so 5 × 4 = 20 SGPRs
for buffer addresses alone. 13 push-const u32s map to ~13 SGPRs
loaded from user-data. The shader prologue / batch-index math (lines
46-86 of `mul_mat_vec_base.glsl`) sinks more into SGPRs because the
expressions are uniform across the wave (workgroup-ID-based).
Together: ~50 "real" SGPRs + ~50 from compiler-introduced
intermediates (constants, masks for the fused-Q4_K dequantize) = 108.

### Is the SGPR lever realistic?

The RGP hint "reduce by 64" means going **108 → 44 SGPRs** would
unlock 14/16 occupancy. That's a 60 % cut. Given that:

- 5 buffer descriptors alone = 20 SGPRs (cannot reduce without
  buffer-device-addressing rewrite).
- 13 push-const scalars = 13 SGPRs (fully used by `get_offsets` and
  the fusion path; can't drop without changing the dispatch API).
- Compiler-generated SGPRs are not directly controllable.

→ **One-extra-wave-per-SIMD is unreachable without a structural shader
rewrite.** The two paths that could move the needle:
1. Switch to `VK_KHR_buffer_device_address` for the weight / input /
   output buffers — frees ~12-16 SGPRs (descriptors collapse to
   64-bit VGPR pointers).
2. Pack push constants — `ne02/ne12/broadcast2/broadcast3` are all 1
   for non-batched decode; could be stripped from the shader path
   for the M=1 case via a specialization-constant guard.

Even both together likely yield only ~24-30 SGPRs back; not the 64
RGP wants. **Filing this as a low-priority, high-effort lever**: the
expected gain is "one more wave/SIMD ≈ 5-8 % memory-latency hiding"
on an already-91 %-of-peak-BW shader, so the absolute upside is at
best 2-3 % of decode wall-time.

## 3. Pipeline State delta — gate (event 20) vs down (event 25)

| | gemv_gate (Q4_K) | gemv_down (Q6_K) |
|---|---:|---:|
| Workgroups | 12,288 | 4,096 |
| Avg wavefront duration | 9.89 µs | n/a (timing screenshot) |
| Shader duration (sum) | 65.7 µs | 88.2 µs |
| VGPR | **48** | **60** |
| SGPR | 108 | 108 |
| LDS | 1024 B | 1024 B |
| Occupancy | **13/16** | **12/16** |

`gemv_down` uses `mul_mat_vec_q6_k.comp` because Qwen3-8B-Q4_K_M ships
`ffn_down.weight` as Q6_K (forward.rs:1532 comment). Q6_K's
super-block layout is denser and the dequant unrolling exposes more
live values, hence +12 VGPRs over Q4_K. The extra VGPR pressure costs
1 wave/SIMD of occupancy (12/16 vs 13/16). At 88.2 µs for K=12288
(3× the Q4_K K-dimension), the per-byte BW is similar — Q6_K is just
a heavier kernel.

## 4. Instruction Timing — gemv_down (event 25)

From `vr-v0.2.1-instruction-timing-event25.png` (right-pane statistics):

| Wavefront statistics | Value |
|---|---:|
| Total instructions | 217,000 |
| Branches taken | 4,290 |

| Instruction class | Count |
|---|---:|
| VALU | 148,200 |
| SALU | 25,220 |
| VMEM | 14,170 |
| LDS | 6,630 |
| WMMA | **0** |

| Hardware utilization | % cycles |
|---|---:|
| VALU | **27.5 %** |
| SALU | 3.2 % |
| VMEM | 12.6 % |
| LDS | 8.8 % |

| Hot stalls | Latency (clk) |
|---|---:|
| `s_wait_loadcnt` (VMEM wait) | **339** |
| `s_wait_kmcnt` (scalar/LDS wait) | 257 |

### Reading

- **VALU = 27.5 %** with **`s_wait_loadcnt` = 339 clk per occurrence**
  is the textbook fingerprint of a **memory-latency-bound** kernel
  with insufficient occupancy to hide the DRAM round-trip. 12/16
  wavefronts/SIMD aren't enough to keep the ALU busy across the HBM
  fetch latency.
- **WMMA = 0** confirms no Wave-Matrix-Multiply path — this is
  exactly the M=1 GEMV case where coopmat is unprofitable
  (Sprint 11A-G's conclusion). The compute is plain `v_fma_f32` /
  `v_dot4_i32_iu8` per the dequantize+FMA loop in
  `mul_mat_vec_q4_k.comp:73-82` (Q6_K is similar shape).
- **VMEM = 12.6 %** vs **`s_wait_loadcnt = 339 clk`** says the VMEM
  units are not saturating; what saturates is the **wait** for HBM,
  not the issue rate of memory instructions.
- **LDS = 8.8 %** with 6,630 LDS hits is the cross-lane reduction in
  `reduce_result` (the partial-sum tree over the 64 threads of the
  wave; subgroupAdd path).

**Effective HBM BW estimate** from this dispatch: 28 MB / 88.2 µs ≈
**320 GB/s ≈ 50 % of the 644 GB/s peak**. Less than `gemv_gate`'s
~91 %. Two factors plausibly explain the gap:

1. **Q6_K is denser**: 6.5625 bits per weight (vs Q4_K's 4.5625).
   Same K=12288 → ~28 MB just like Q4_K's K=4096 N=12288 (which is
   why the byte total looks the same). But Q6_K's per-block scale
   layout requires an extra `data_a_packed16[…].scales[v_im+4]` fetch
   per super-block; that's a serial dependency on the inner loop.
2. **K=12288** vs gate's K=4096 means **3× more inner-loop
   iterations**. Each iteration needs an HBM round-trip for fresh
   weight bytes; that's 3× more `s_wait_loadcnt` events to absorb on
   the same 12-wave-per-SIMD occupancy.

**This is a real lever** — closing gemv_down from 50 % to ~80 % peak
BW is **+ ~30 µs/dispatch × 36 = ~1.1 ms = ~9 % decode wall-time**.

## 5. Wavefront Occupancy timeline

From `vr-v0.2.1-wqavefront.png`:

- Forward span: ~12,400 µs (consistent with the 12.34 ms wall from
  Sprint 12G-C).
- 36 visually distinct repetitive blocks across the timeline =
  36 layers of Qwen3-8B.
- A large pink/magenta block at the very end of the timeline, lasting
  ~750 µs and showing very high sustained occupancy across all 4 SE
  rows — this is **`lm_head`** (vocab=151,936 → 151,936 wave64
  workgroups, single dispatch).
- Within each layer-block: visible micro-structure — peaks at the
  large GEMVs (gate/up/down at 12,288 / 4,096 wave fronts each fill
  the SIMDs to occupancy ≈ 75 %), troughs during the small
  scalar/reduction dispatches (rms_norm, fa_reduce, swiglu).
- **Inter-dispatch gaps:** the timeline shows a continuous "noise
  floor" of activity between the main peaks — there are no visibly
  large idle gaps, consistent with Sprint 12G-C's measurement that
  CPU/dispatch overhead is < 1 % at steady-state decode.
- **SE balance:** the four shader-engine rows (SE0/1/2/3) appear
  similarly populated. RGP doesn't expose an explicit SE-imbalance
  number from this screenshot, but visually no SE looks systematically
  starved relative to the others.

## 6. Most-expensive events

`most-expensive-events.png` (older capture but same shape) shows the
top events of the forward concentrate in a narrow µs band:

| Rank | Event | Duration |
|---:|---|---:|
| 1 | `vkCmdDispatch(1228, 1, 1)` (LM head approx.) | 974.97 µs |
| 2-37 | `vkCmdDispatch(4096, 1, 1)` (gemv_down × 36) | **87.34 – 87.40 µs** each |
| 38-… | gemv_gate, gemv_up (12288×) | 65-66 µs each |

The histogram in the same screenshot reads "the most expensive 5 % of
events take 97 % of the total GPU time of the frame" — concretely:
**the 36 gemv_down + 36 (gate+up) + 1 lm_head dispatches = 109 events
≈ 6 % of total events absorb almost all the time.** Every µs we want
to recover lives in those events.

## 7. Barrier costs — the previously hidden term

Per-dispatch `vkCmdWriteTimestamp` cannot see barrier-wait time
(barriers don't get profiled by `self.profile()`). RGP shows them
explicitly. Per-layer barrier sum (decode pos=29):

| Barrier event | µs | Drains |
|---|---:|---|
| 1 (post `rms_norm_attn`) | 3.9 | `hidden_norm` |
| 5 (post Q/K/V) | **31.1** | 3 buffers × 4096/1024 outputs |
| 8 (post norm+rope) | small | k/v |
| 11 (kv_bar) | 1.3 | TRANSFER+COMPUTE → COMPUTE global |
| 14 (post attn) | small | `attn_out` |
| 17 (post `gemv_o`) | **21.4** | `input` + `o_buf` for fused add+rms |
| 19 (post `add_rms_ffn`) | small | `hidden_norm` |
| 22 (post gate/up) | small | `gate_buf` + `up_buf` |
| 24 (post swiglu) | small | `ffn_hidden` |
| 26 (post `gemv_down`) | **83.5** | `res1` + `ffn_out` for residual2 |
| 28 (end-of-layer) | small | `output` |

The three **expensive** barriers per layer (5, 17, 26) sum to **~136 µs
per layer × 36 layers ≈ 4.9 ms** if these are real serial-stall costs.

### Are they actually serial?

Two reasons to think the RGP barrier µs is **largely overlapped with
shader work**, not pure GPU idle:

1. The pos=29 forward total is ~12.3 ms wall. If barriers were
   serial, the shader sum (Sprint 12G-C: ~12.3 ms aggregate
   timestamps) plus 4.9 ms of barrier serial-stall would be > 17 ms,
   not 12.3 ms. The arithmetic doesn't fit — barriers must be
   substantially overlapping the *next* dispatch's prologue or the
   *previous* dispatch's tail.
2. Sprint 12C / 12D already did the serial-stall test by counting
   "barrier-elision rate" = 0 % (every barrier fires) and yet
   wall-time changed by < 1 % when fusing dispatches. That null
   result is consistent with barrier-time being mostly overlapped.

The most plausible reading of RGP's barrier µs is **"how long the
hardware-side fence took to retire"**, which on RDNA4 is dominated by
the final wave drain of the *preceding* dispatch + cache-flush. So
event 26's 83.5 µs is essentially "wait for the last 4096-WG
gemv_down wave to retire and L2 to flush" — and that wait *would
have happened anyway* before `add_res2` could read `ffn_out`, with
or without the explicit barrier.

**Practical conclusion:** the barrier numbers in RGP are not a hidden
"missing 5 %" lever; they are the visualisation of dependency
boundaries that any correct execution must respect. The earlier
Sprint 12C model that counted "2 µs/barrier" was already correctly
treating these as small CPU costs rather than serial GPU stalls.

The one place where the barrier number is *interestingly* high is
event 26 (post-`gemv_down`, 83.5 µs ≈ the gemv_down dispatch itself
takes 88.2 µs). That tells us **gemv_down's tail is serializing the
end of the layer**: the next dispatch (`add_res2`) cannot start until
all 4096 wave64 work groups of gemv_down finish, and at 12-waves/SIMD
occupancy the long-tailed final waves dominate. **Reducing
gemv_down's wall time** (per §4) would automatically reduce this
barrier's apparent cost.

## 8. Decode-time decomposition (RGP-derived, definitive)

Per layer, summing the RGP event µs we have hard numbers for:

| Bucket | µs / layer | × 36 layers | % of forward |
|---|---:|---:|---:|
| GEMVs (q+k+v+o+gate+up+down) | ~308 | 11 088 | **89.4 %** |
| Attention (`fa_split + fa_reduce`) | ~52 | 1 872 | 15.1 %* |
| Norm + RoPE + KV-copy | ~10 | 360 | 2.9 % |
| SwiGLU + add_rms_ffn + add_res2 | ~6 | 216 | 1.7 % |
| `lm_head` (1×) | — | 975 | 7.9 % |
| Forward total (RGP) | — | ~12 400 | 100 % |
| **Sum here** | — | ~14 511 | — |

* Marked: percentages don't sum to 100 because dispatches **overlap**
  on the GPU (especially the gate/up pair within a layer, and adjacent
  norms/copies between layers). The "sum here" overshoot of ~17 %
  vs the 12.4 ms wall is exactly the parallelism that
  `vkCmdWriteTimestamp` was charging incorrectly to a single
  dispatch. RGP shows it as separate events that can run in parallel.

Headline ratio matches Sprint 12G-C:
**GEMVs ≈ 80-90 % of decode time, single biggest bucket.**

The decode wall is **lower** than the sum because the GPU pipelines
dispatches without barriers between them (most notably gate↔up and
the small ops between layers). RGP shows this visually as overlapping
event durations on the 4 shader-engine rows.

## 9. Recommendation — where the next sprint goes

After 12G-C and 12G-D-retry the picture is: **GEMV is 80-90 %, and
within GEMV the asymmetry is gemv_down (Q6_K, 88 µs, 50 % BW) vs
gemv_gate/up (Q4_K, 65 µs, 91 % BW)**. The single highest-ROI lever
in the decode forward is therefore **closing the gemv_down BW gap**,
not the SGPR-occupancy lever (too expensive for too little gain) and
not the barrier lever (already overlapped).

Concrete plan (estimate 2-3 days, profiling-first):

1. **Day 1.** Re-profile a Q4_K-only model (e.g. Qwen3-8B-Q4_K_S, where
   ffn_down is also Q4_K) to confirm gemv_down's "low BW" goes away
   when the shader is the same as gate/up. If yes → the lever is
   `mul_mat_vec_q6_k.comp` specifically.
2. **Day 2.** Inspect `mul_mat_vec_q6_k.comp` for the per-block scale
   indirection that doesn't exist in Q4_K (the extra
   `data_a_packed16[…].scales[v_im+4]` fetch). Compare ISA in RGP for
   the inner loop — is there a `s_wait_loadcnt` between the two scale
   fetches that doesn't appear in Q4_K? If yes, prefetch the second
   scale earlier.
3. **Day 3.** Re-run `profile_positions` and `run_validation`. Gate the
   sprint on Q6_K_path → ≥ 70 % peak BW (currently 50 %). +9 % decode
   target.

Secondary lever: `lm_head` is a single 740-975 µs dispatch (≈ 6-8 %
of decode). It uses the same Q4_K GEMV with N=151,936 and reads ~110
MB of weights. Wide tile / coopmat path investigated in Sprints
11A-G never won at M=1 GEMV shape, but `lm_head` is **not** M=1 in
the same way (the output is 151k logits, the *input* is M=1, K=4096).
A dedicated lm_head shader that batches across the vocab dimension
might help; budget 1 day to scope.

## 10. Closed earlier

Sprint 12G-D's swap experiment proved the gemv_up "75 % slowdown" was
a `vkCmdWriteTimestamp` artifact. RGP **visually confirms**: events
20+21 are 66.0 / 65.7 µs respectively (parity within 0.5 %), with
identical Pipeline State (48 VGPR / 108 SGPR / 1024 LDS / 13/16
occupancy) — same shader, same shape, same execution profile. The
1.76× ratio in Sprint 12G-C was timestamps, not GPU.

Path A (graph-layer rewrite, 4 weeks) — still rejected (CPU overhead
< 1 % at steady-state).
Path B (six fusion shaders, 6 days) — extrapolates to +5-6 %; lower
ROI than gemv_down BW work.
**Path D — gemv_down (Q6_K) BW recovery — is the new top lever.**
