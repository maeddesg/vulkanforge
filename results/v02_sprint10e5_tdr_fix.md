# VulkanForge v0.2 Sprint 10E.5 — pp=4096 TDR-Crash Investigation

**Date:** 2026-04-29
**Branch:** main (HEAD = 7d9ea12 v0.2 sprint8b.1 + Sprint 10A–10E commits queued)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV, kernel 7.0.1-cachyos
**Model:** Qwen3-8B-Q4_K_M

## TL;DR

Sprint 10E flipped `VULKANFORGE_COOPMAT_ATTN` to **default ON**. With that
default, **pp=4096 now runs cleanly** at **1652 tok/s** (4 prefill chunks of
1024 tokens each). The TDR (Timeout Detection and Recovery) crash that
plagued the scalar attention path at long contexts is gone.

The crash still reproduces when users explicitly opt out
(`VULKANFORGE_COOPMAT_ATTN=0`) at pp ≥ 4096 — scalar attention's per-WG
work scales with `n_kv` and exceeds the kernel's ~5 s GPU watchdog at
kv_len = 4096. **No code fix is needed beyond what Sprint 10E already
shipped:** the default config is safe; opt-out at very long context is
"hold it differently."

This sprint is an **honest investigation report**. The pp-sweep, the
chunking audit, and the bisection are documented; no shader or runtime
change is committed. 167/167 regression tests stay green.

## Context

**Pre-existing problem (Sprint 9d.2 era):** pp=4096 returned `DEVICE_LOST`
on the scalar `flash_attn_tiled.comp` path. The chunked-prefill loop
(`decode.rs:443-453`) splits the prompt into 1024-token chunks, but the
*last* chunk's attention dispatch sees `n_kv = prior_pos + chunk_len`,
i.e. up to 4096 KV positions × 1024 query tokens. Per-WG inner-loop work
is `O(BR × n_kv × head_dim)` scalar FMAs — at kv_len = 4096 that crosses
the Linux RADV TDR threshold.

**Sprint 10C** introduced `flash_attn_coopmat.comp` (KHR-coopmat QK + scalar
softmax + scalar PV via per-thread `my_out0/my_out1` registers), gated
behind `VULKANFORGE_COOPMAT_ATTN=1` (opt-in, default OFF).

**Sprint 10E** flipped that default to ON (opt-out via `=0`).

This sprint asks: **does the new default actually fix the pp=4096 crash?**

## Method

1. **Reproduce** the crash on the current HEAD (Sprint 10E default-ON).
2. **Bisect** which env-var matters: `COOPMAT_ATTN` and/or `FP16_KV`.
3. **Audit** the chunking logic in `forward.rs` / `decode.rs` to confirm
   `chunk_size` is enforced for all paths.
4. **Full pp-sweep** with the default-ON config to validate.
5. **Regression** — 167 tests must stay green.

### Bench harness

`examples/run_pp_bench.rs` (Sprint 9d.2). Driver:
```
VF_PP_LIST=<list> VF_PP_RUNS=3 VF_PP_WARMUP=1 cargo run --release --example run_pp_bench
```

## Results

### Headline pp-sweep (default config: coopmat ON + FP16 KV ON)

| pp   | median_ms | tok/s   | chunks |
|------|----------:|--------:|-------:|
| 64   | 41.69     | 1535.3  | 1      |
| 128  | 63.70     | 2009.4  | 1      |
| 256  | 117.37    | 2181.1  | 1      |
| 512  | 228.14    | 2244.2  | 1      |
| 1024 | 467.72    | 2189.3  | 1      |
| 2048 | 1030.17   | 1988.0  | 2      |
| **4096** | **2479.46** | **1652.0** | **4** |

Peak prefill throughput: **2244 tok/s @ pp=512**.
pp=4096 crash: **resolved** by Sprint 10E's default-ON.

### Opt-out comparison (`VULKANFORGE_COOPMAT_ATTN=0`)

| pp   | scalar tok/s | coopmat tok/s | Δ        |
|------|-------------:|--------------:|---------:|
| 64   | 1479.0       | 1535.3        | +3.8 %   |
| 128  | 1890.4       | 2009.4        | +6.3 %   |
| 256  | 1928.1       | 2181.1        | +13.1 %  |
| 512  | 1771.2       | 2244.2        | +26.7 %  |
| 1024 | 1484.1       | 2189.3        | +47.5 %  |
| 2048 | 1071.6       | 1988.0        | +85.5 %  |
| 4096 | **DEVICE_LOST** | 1652.0    | crash → works |

The coopmat win grows with kv_len exactly as expected — QK is `O(n_kv)`
per query, so the per-tile coopmat speedup compounds as the context
lengthens. At pp=2048 coopmat is +85 %; at pp=4096 it's the difference
between "works" and "TDR".

### Bisection of the pp=4096 crash

| Config                                    | pp=4096 result   |
|-------------------------------------------|------------------|
| coopmat ON + FP16 KV ON  (Sprint 10E def.) | ✅ 1652 tok/s   |
| coopmat ON + FP16 KV OFF                  | ✅ 1665 tok/s   |
| coopmat OFF + FP16 KV ON                  | ❌ DEVICE_LOST  |
| coopmat OFF + FP16 KV OFF                 | ❌ DEVICE_LOST  |

`COOPMAT_ATTN` is the **only** determining factor. `FP16_KV` is irrelevant
to the crash (it changes K/V load bandwidth but not per-tile compute).

### Boundary sweep (default config)

| pp   | tok/s  |
|------|-------:|
| 2560 | 1846   |
| 3072 | 1806   |
| 3584 | 1684   |
| 4096 | 1652   |

Smooth degradation through the boundary; no cliff. The drop from 2244
@ pp=512 to 1652 @ pp=4096 is the expected `O(n_kv²)` softmax tail —
attention compute dominates the prefill at long contexts.

## Chunking audit

`decode.rs:443-453`:
```rust
let chunk_size = forward.max_prefill_tokens.max(1) as usize;  // 1024
for chunk in prefill_tokens.chunks(chunk_size) {
    let chunk_len = chunk.len() as u32;
    ...
    forward.prefill_batch(dev, registry, cmd_ctx, model, &chunk_embeds,
                          chunk_len, pos, ...)?;
    pos += chunk_len;
}
```

**Confirmed:** every prefill path goes through `decode::run_prefill_chunked`,
which slices into ≤1024-token chunks. `prefill_batch` itself asserts
`seq_len ≤ max_prefill_tokens` (`forward.rs:2837`). The single attention
dispatch per chunk-per-layer (`run_flash_attn_tiled`,
`forward.rs:2161`) is `cmd_dispatch(n_heads, m.div_ceil(br), 1)` —
`m ≤ 1024`, so `q_tiles ≤ 64`. The dispatch's `n_kv` push-constant,
however, scales with the **cumulative** prior context: at the last
chunk of pp=4096 it is 4096.

**Why scalar fails and coopmat succeeds at n_kv=4096:**

Each WG of `flash_attn_tiled` runs an outer loop over `tile_base = 0
..n_kv` step Bc. For Br=16, Bc=32, that's 128 tiles at n_kv=4096.
Per-tile inner loop: 16 queries × 128 head_dim × 64 threads of scalar
FMA ≈ 130 K FMA-ops per tile per WG. ×128 tiles × 36 layers × 32 q_heads
≈ ~20 G scalar-FMA-equivalents per chunk — enough to cross the 5 s
TDR window with RADV's scheduling.

`flash_attn_coopmat.comp` replaces the inner `head_dim` scalar loop with
a chain of 8 `coopMatMulAdd` calls (`vk_shaders/flash_attn_coopmat.comp:157-165`),
each consuming a 16×16×16 WMMA tile. The Sprint 10B microbench measured
this as 47.5× scalar FMA on Br=Bc=16. End-to-end speedup is smaller
because softmax + PV remain scalar, but **the QK term — the dominant
cost at long n_kv — is what was pushing the dispatch past TDR.**
Coopmat brings it back under the watchdog.

## Decision

**No code change committed.** Sprint 10E's default-ON already prevents
the crash. Specifically:

- ✅ Default users (no env-vars set) are safe at pp=4096.
- ⚠️ Users who explicitly opt out via `VULKANFORGE_COOPMAT_ATTN=0`
   will still crash at pp ≥ 4096. This is acceptable: opt-out is an
   explicit choice, and the only reasons to opt out are debugging or
   parity-isolation (where the user already knows what they're doing).

Considered but rejected:

- **Auto-fallback to coopmat when scalar would TDR.** Adds runtime
  state-machine complexity for a path nobody should take. The opt-out
  exists for debugging; making it silently re-enable coopmat defeats
  the purpose.
- **Reduce `max_prefill_tokens` for scalar opt-out.** Doesn't help —
  the last-chunk attention sees `n_kv = total_prefill`, regardless of
  chunk size. The fix would have to chunk the *attention* itself
  (split-K), which is what `flash_attn_split` already does and which
  is a separate architectural path.
- **Runtime warning for `COOPMAT_ATTN=0` + long-context users.** Possibly
  useful but premature — we have no telemetry that this opt-out is in
  use anywhere, and adding a stderr warning to a compute-only library
  is ugly.

## Regression

```
test result: ok. 27 passed; 0 failed
test result: ok.  9 passed; 0 failed
test result: ok. 18 passed; 0 failed
test result: ok. 70 passed; 0 failed
test result: ok.  8 passed; 0 failed
test result: ok.  8 passed; 0 failed
test result: ok. 27 passed; 0 failed
                ----
total           167 passed; 0 failed
```

All 167/167 green. No code changed; this is purely an investigation
report.

## Files touched

- `results/v02_sprint10e5_tdr_fix.md` (this report — new)

No source changes. No SPV rebuilds. No new dependencies.

## Forward look

- Sprint 10F candidate: **PV-coopmat retry** with a different scheme
  than Sprint 10D's LDS-scratch hybrid. Possible: keep `my_out` as a
  per-thread register array but consume `pscore` from a coopmat fragment
  via an A-fragment pull (no LDS round-trip). Risk: same FP16/FP32
  fragment-mixing gotcha that 10D ran into.
- Or: **decode-path coopmat** — currently `forward_token` uses the
  same `flash_attn_tiled` cascade as prefill, so coopmat already
  applies at decode. Verify there's no decode regression and call it.
- Long-context (pp ≥ 8192) will eventually need split-K attention or
  a longer TDR window. Out of scope for v0.2.

## Take-aways

1. **Sprint 10E's default-flip is the real fix for pp=4096.** Sprint 10E.5
   is the verification, not the fix.
2. **End-to-end speedup that makes a dispatch cross/under TDR is
   different from end-to-end speedup that just makes things faster.**
   Coopmat at +47–85 % at pp=1024–2048 is "nice." At pp=4096 it's
   "works vs doesn't." Both are real wins; they're worth different
   things in the changelog.
3. **Honest negatives are commits too.** The "no code change" outcome
   is the right outcome here, and saying so explicitly is more useful
   than inventing a fix to justify the sprint.
