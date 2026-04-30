# Sprint 12H — Q6_K BW recovery: HONEST NEGATIVE

**Premise.** Sprint 12G-D RGP analysis showed `gemv_down` (Q6_K, M=1,
N=4096, K=12288) at 88.2 µs / "~50 % peak BW" / 60 VGPRs / 12-of-16
occupancy, vs `gemv_gate` (Q4_K) at 66 µs / "91 % peak BW". The
sprint hypothesis: port a faster Q6_K shader from llama.cpp (or
optimise our own) to recover the gap, target +9 % decode wall-time.

**Verdict (Step 0 + Step 1, before any code change):**
**HYPOTHESIS FALSIFIED.** Our `mul_mat_vec_q6_k.comp` is **byte-
identical** to llama.cpp HEAD's. There is no upstream optimisation
to port, and the per-format cost we see (Q6_K runs slower than Q4_K)
is intrinsic to the format — llama.cpp pays the same price.

No shader code changed. No new tests. Current numbers reproduced as
the baseline; future Q6_K BW work, if any, requires inventing
something the upstream community hasn't.

## 1. The shader-identity check (Step 0+1, decisive)

```
$ md5sum vk_shaders/mul_mat_vec_q6_k.comp \
    ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
2fe68d21485384afc1384fdbca4b4bb8  vk_shaders/mul_mat_vec_q6_k.comp
2fe68d21485384afc1384fdbca4b4bb8  …/llama.cpp/…/mul_mat_vec_q6_k.comp

$ md5sum vk_shaders/mul_mat_vec_q4_k.comp \
    ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp
9112134d596638833468e6b8f3fd974b  vk_shaders/mul_mat_vec_q4_k.comp
9112134d596638833468e6b8f3fd974b  …/llama.cpp/…/mul_mat_vec_q4_k.comp

$ diff vk_shaders/mul_mat_vec_q6_k.comp \
    ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
(no output — files identical)

llama.cpp HEAD:  23b8cc4 (android : libcommon -> libllama-common, #22076)
```

Both Q4_K and Q6_K Vulkan GEMV shaders are byte-identical to llama.cpp
HEAD. There is no "newer / better" version upstream to port.

Per the project memory's "Sprint Hypothesis Pre-Check" rule, this
sprint joins 8b / 9c / 11A / 11D / 11E-A as honest negatives caught
*after* the brief was written but *before* any code was touched.

## 2. Why Q6_K is structurally slower than Q4_K (format-level)

Reading both shaders in `vk_shaders/mul_mat_vec_q4_k.comp` (135 LOC)
and `vk_shaders/mul_mat_vec_q6_k.comp` (130 LOC):

| Aspect | Q4_K | Q6_K |
|---|---|---|
| Bits per weight | 4 | 6 |
| Per-block weight arrays | 1 (`qs`) | **2 (`ql` + `qh`)** |
| Per-block scale layout | inline-packed bytes (line 19-26) | **separate `scales[]` array** |
| Scale fetch | per-thread, register-only | **LDS-cached, shared across threads** |
| `barrier()` in hot loop | **none** | **two** (line 24, line 56) |
| `csel ^= 1` LDS double-buffer | n/a | yes (line 19) |
| Inner-loop temporaries (vec4) | 4 (`q0..q3`) | 4 (`q0..q3`) but built from 2 sources |

The Q6_K format is denser (6 bits vs 4 bits, +37 % bits-per-weight)
and stores the high 2 bits of every weight in a separate `qh` array
that must be fetched independently and bit-blended with `ql`.
Additionally the scales are too numerous to fit in registers, so the
shader uses LDS as a scale cache populated by the first 16 threads of
the workgroup and then read back — this requires `barrier()` calls
per super-block, which are workgroup-wide synchronisations.

The double-buffered `csel ^= 1` scheme is a partial mitigation
(allows the *next* block's scale-load to proceed while the *current*
block's compute runs), but it doesn't eliminate the fundamental
overhead.

These are deliberate tradeoffs of the Q6_K format, not bugs:

- llama.cpp's `Q4_K_M` quantisation specifically uses Q6_K for
  `attn_v.weight` and `ffn_down.weight` because the precision delta
  matters for those tensors. The user *paid* for higher precision
  via slower kernels.
- Anyone targeting decode speed over precision can pick `Q4_K_S`
  (uniform Q4_K everywhere) and skip the Q6_K kernel entirely —
  Q4_K_S is not on this machine but is freely downloadable.

## 3. The "50 % peak BW" claim — re-examined

Sprint 12G-D's "gemv_down at 50 % peak BW" came from the RGP-GUI
event-timing reading of **88.2 µs** with
`RADV_THREAD_TRACE_INSTRUCTION_TIMING=1` enabled. **Instruction-
timing tracing perturbs absolute durations.**

Re-running `profile_positions` today (no instruction timing) gives
a tighter steady-state number:

| pos | wall (ms) | gemv_down µs/dispatch (sum/36) | tok/s |
|---:|---:|---:|---:|
| 0 | 16.83 | 77.9 | 59.4 |
| 50 | 11.89 | 53.9 | 84.1 |
| 100 | 11.87 | 53.9 | 84.3 |
| 200 | 11.81 | **53.7** | 84.7 |

At 53.7 µs / dispatch:
- bytes read ≈ K × N × bpw = 12 288 × 4 096 × (6.5625 / 8) = 26.5 MB
- effective BW = 26.5 MB / 53.7 µs ≈ **494 GB/s ≈ 77 % of 644 GB/s peak**

The "50 %" headline number was an artifact of the 12G-D capture
running with instruction timing on (~50–60 % overhead). At 77 % peak
the BW is **good, not bad** — the real gap to close, if it exists,
is at most ~14 % BW (77 % → ~91 % matching gate), not 41 %.

That translates to:
- 53.7 µs → ~46 µs → save ~7.7 µs / dispatch
- 7.7 × 36 = 277 µs / forward
- 277 µs / 11 810 µs wall = **+2.3 % decode wall-time**, not 9 %.

A 2 % lever requiring shader rewrites that diverge from upstream is
not a sprint we should run.

## 4. 15-prompt bench baseline (this run)

```
=== Aggregate ===
  Prefill total:   728.5 ms  →  1100.9 tok/s
  Decode total:  71030.1 ms  →  85.6 tok/s
  MEDIAN decode:  90.7 tok/s
  MEDIAN prefill: 1083.1 tok/s
  Coherent prompts: 15/15

=== 4-System Comparison ===
  llama.cpp Vulkan:           114.2 tok/s decode  4314 tok/s prefill
  VulkanForge (this run):      85.6 mean / 90.7 median  1100.9 prefill
  Ratio decode (median):       0.79 ×
  Ratio decode (mean):         0.75 ×
  Ratio prefill:               0.26 ×
```

Tests: 27 passed, 0 failed (`cargo test --release --lib`).

Decode is at the same 0.75-0.80 × llama.cpp ratio Sprint 12 has
reported throughout. Since our Q6_K shader is *llama.cpp's*
Q6_K shader, that gap cannot be primarily Q6_K. It must live
elsewhere — most plausibly in:

- **Prefill** (we are at 0.26 × llama.cpp, the much bigger gap),
  where llama.cpp uses graph-level fusion / scheduling that we don't
  (Sprint 12G-A audit). But prefill is its own sprint, not 12H's
  premise.
- **Per-token-CPU sampling/dispatch** (gap that doesn't show in the
  GPU gpu_sum but does show in wall — we measured ~0.1 % steady-
  state, so this is small).
- **A constant-factor difference in how llama.cpp pipelines
  dispatches** (their forward might overlap dispatches more
  aggressively — gpu_sum > wall observed in our prefill, suggesting
  some overlap is happening; might be more on llama.cpp's side).

None of these are addressable by editing `mul_mat_vec_q6_k.comp`.

## 5. What the sprint did (and did not) produce

| Activity | Outcome |
|---|---|
| Read `mul_mat_vec_q6_k.comp` (130 LOC) | done, §2 |
| Compare to llama.cpp HEAD (md5sum + diff) | done, **identical**, §1 |
| Compare to `mul_mat_vec_q4_k.comp` | done, format-level diff only, §2 |
| Re-run baseline benchmarks | done, §3-4 |
| Implement an optimization | **NOT DONE** — premise falsified |
| Write parity tests | **NOT DONE** — no code change |
| Commit shader change | **NOT DONE** — no code change |

The brief authorised a shader rewrite but explicitly anticipated
this failure mode in §2.4 and the "Bekannte Fallstricke" #1: "Falls
IDENTISCH: gleicher Shader, gleiche BW! → Der Gap ist NICHT
shader-level …". This is exactly the case.

## 6. Where the next sprint should go

Given Q6_K is at ~77 % peak BW (not 50 %) and the shader matches
upstream, the remaining decode gap to llama.cpp lives elsewhere.
Re-prioritising the Sprint 12 path tree:

1. **NEW Path E — Prefill GEMM fusion / scheduling.** Our prefill is
   at 0.26 × llama.cpp (1100 vs 4314 tok/s). That's a 4 × gap, much
   bigger than decode's 1.25 × gap. llama.cpp uses graph-level
   scheduling for prefill that we don't (Sprint 12G-A). Even
   recovering 50 % of that gap is a far bigger win than any
   decode-side micro-optimisation. Estimated effort: 1-2 weeks for
   the minimum viable graph layer that just covers prefill.
2. **Path B (specific fusion shaders, decode)** still extrapolates
   to +5-6 %; reasonable as a fill-in sprint.
3. **Path D (this sprint, gemv_down BW recovery): closed.**
4. **Path A (full graph-layer rewrite, 4 weeks): rejected.**

A user-side option that bypasses Q6_K entirely is to ship / use
**Q4_K_S** instead of Q4_K_M. That removes Q6_K from `attn_v` and
`ffn_down`, replacing them with Q4_K. Expected gain on this
hardware: gemv_down at ~46-47 µs (matching gate) → ~+2.3 % decode,
plus gemv_v improvement of similar relative magnitude scaled by its
12.9 % share → ~+1 % decode. Total ~3-4 % decode for free, at the
cost of slightly worse output quality on attention-V and FFN-down
projections. **No code change required**, only a different model
file. Not a sprint, just a download. Worth mentioning in the
project README.

## 7. Outputs

- This report.
- **No code changes** committed (`git diff` empty for `src/`,
  `vk_shaders/`).
- **No tests added** (premise falsified before any new test was
  worth writing).
- 27 / 27 lib tests pass after this sprint, identical to before.
- Memory updated to extend the Sprint Hypothesis Pre-Check entry
  with the "diff against upstream" specific case.
