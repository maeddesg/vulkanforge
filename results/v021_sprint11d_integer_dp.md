# VulkanForge v0.2.1 Sprint 11D — Integer Dot-Product Pre-Check

**Date:** 2026-04-29
**Branch:** main (HEAD = a4a950f, post-Sprint 11C)
**GPU:** AMD RX 9070 XT (RDNA4, gfx1201), Mesa RADV
**Mode:** Analysis-only (third hypothesis pre-check hit in three sprints)

## TL;DR — already shipped

The brief proposes porting an integer-dot-product GEMM shader (using
`dot4_i8packed` / `dotPacked4x8EXT`) as the "endgame" GEMM lever.

**We already use it.** Three independent verifications:

1. **Shader source:** `vk_shaders/mul_mmq.comp:7` declares
   `#extension GL_EXT_integer_dot_product : require`. The Q4_K and
   Q6_K inner-loop dot products go through `dotPacked4x8EXT` —
   16 call sites in `mul_mmq_funcs.glsl` (lines 47–407, plus all
   the K-quant `mmq_dot_product` definitions).
2. **Compiled SPV:** `spirv-dis mul_mmq_q4_k_f32.spv` shows
   `OpCapability DotProduct`, `OpCapability DotProductInput4x8BitPacked`,
   `OpExtension "SPV_KHR_integer_dot_product"`. The 4×INT8→INT32
   instruction is in the SPIR-V we ship.
3. **Device feature:** `device.rs:139-140` enables
   `VkPhysicalDeviceVulkan13Features::shaderIntegerDotProduct = true`
   (Phase 3C, v0.1.x). `vulkaninfo` confirms RADV gfx1201 advertises
   `VK_KHR_shader_integer_dot_product : extension revision 1` with
   `shaderIntegerDotProduct = true`.

Sprint 11C's +4–5 % at pp ≥ 512 was already on top of an
integer-DP-active baseline. We've been running with `dot4_i8packed`
since the v0.1.x mul_mmq port.

The brief's framing (`mul_mmq.comp` is "scalar", a separate
`matmul_q4_k_q8_1_int_dp.comp` exists in upstream that we'd port) is
incorrect on both points: (a) `mul_mmq.comp` is the integer-DP shader
upstream, and (b) there is no separate `*_int_dp.comp` file — the
file `matmul_q4_k_q8_1_int_dp.comp` does not exist in current
llama.cpp.

This is the **third** sprint in three sprints to hit the same
pre-check pattern (8b conditional barriers, 9c rms_norm_mul, 11A
Q8_1+mul_mm port — and now 11D integer-DP). The hypothesis-check
memory paid for itself again. **No code changes; no commit needed
beyond this report.**

## Pre-check evidence in detail

### 1. Shader source uses `dotPacked4x8EXT`

`vk_shaders/mul_mmq.comp` line 7:
```glsl
#extension GL_EXT_integer_dot_product : require
```

`vk_shaders/mul_mmq_funcs.glsl` — Q4_K dot product
(lines 349–363, the `mmq_dot_product` for `DATA_A_Q4_K`):
```glsl
ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = int32_t(
            (cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);  // ← THIS
    }
    return ACC_TYPE(...);
}
```

`vk_shaders/mul_mmq_funcs.glsl:407` — Q6_K equivalent:
```glsl
[[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
    const int32_t qs_a = cache_a[ib_a].qs[iqs];
    q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
}
```

`dotPacked4x8EXT` is the GLSL frontend for the SPIR-V
`OpSDot`/`OpSDotKHR` instruction with `PackedVectorFormat4x8Bit` —
which on RDNA4 maps to the `v_dot4_i32_iu8` instruction (verified
via the SPV disasm + RGA on prior sprints).

### 2. SPIR-V confirms the capability

```
$ spirv-dis target/release/build/.../out/mul_mmq_q4_k_f32.spv | head -10
   OpCapability Shader
   OpCapability Int8
   OpCapability StorageBuffer16BitAccess
   OpCapability StorageBuffer8BitAccess
   OpCapability DotProductInput4x8BitPacked    ← packed 4×INT8 input
   OpCapability DotProduct                     ← KHR integer dot product
   OpExtension "SPV_KHR_integer_dot_product"   ← extension declared
```

If the brief's premise were correct (we were running scalar FMA),
this SPV would show neither capability. It shows both.

### 3. Device-side activation (Phase 3C)

`src/backend/vulkan/device.rs:134-140`:
```rust
// Phase 3C adds Vulkan 1.3 `shaderIntegerDotProduct` so the
// mul_mmq.comp SPIR-V (compile-probe in this phase, full
// dispatch in 3D) can declare its `DotProduct` capabilities
// cleanly — RADV/gfx1201 reports both 8-bit accelerated paths
// available.
let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
    .shader_integer_dot_product(true);
```

Plus the dependent feature chain is enabled:
```rust
features12.storage_buffer8_bit_access(true).shader_int8(true);
features11.storage_buffer16_bit_access(true);
core_features.shader_int16(true);
```

`vulkaninfo` on this machine:
```
VK_KHR_shader_integer_dot_product : extension revision 1
shaderIntegerDotProduct                            = true
```

All three layers are wired through: GLSL extension required, SPV
capability declared, Vulkan device feature enabled. The integer dot
product runs every prefill GEMM call.

### 4. The shader the brief referenced does not exist upstream

```
$ find ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders -name '*int_dp*'
   (empty)
$ find ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders -name '*integer*'
   ~/tmp/llama.cpp/.../feature-tests/integer_dot.comp   (a capability probe, not a kernel)
```

The brief asks for "matmul_q4_k_q8_1_int_dp.comp" — that file is
not in upstream llama.cpp. Integer-DP for Q4_K × Q8_1 is implemented
*inside* `mul_mmq.comp` via the `mmq_dot_product` function in
`mul_mmq_funcs.glsl`, conditional on `DATA_A_Q4_K` and the
`GL_EXT_integer_dot_product` extension. Both flags are set in our
build (`build.rs:560+` and the bit-identical shader source).

## Why is the gap still 0.54× then?

This is the more useful question. With Q8_1 activations, integer-DP
GEMM, S+L tile pipelines, and KV-cache FP16 all shipped, where is
the remaining ~0.46× gap to llama.cpp Vulkan?

The gap is **structural**, not single-feature:

### A. Tile-pipeline coverage: 3 missing variants

Sprint 11C added `L`. Llama.cpp ships **6 pipelines per quant type**:
`{S, M, L} × {aligned, unaligned}`. We ship 2 (`S`, `L` unaligned only):

| Variant            | We ship | llama.cpp ships | Gap impact (estimated)              |
|--------------------|:-------:|:---------------:|:------------------------------------|
| S unaligned        |    ✓    |        ✓        | —                                   |
| S aligned          |    ✗    |        ✓        | small (decode-adjacent shapes)      |
| **M unaligned**    |    ✗    |        ✓        | **medium** — 64 < n ≤ 256 + GQA K/V |
| M aligned          |    ✗    |        ✓        | medium                              |
| **L unaligned**    |    ✓    |        ✓        | covered by 11C                      |
| L aligned          |    ✗    |        ✓        | medium — large pp with seq_len%4=0  |

The K/V projections at typical Qwen3 prefill (m=1024 GQA, n=512) sit
on the L threshold but with `groups_x · groups_y = 8 · 4 = 32` —
half-empty GPU. An M-tile (`BM=BN=64` with longer K-chain) would
keep WG count at 8 · 8 = 64 and amortise the K-loop better. *This*
is the biggest 11E candidate.

### B. `mul_mm.comp` COOPMAT path not enabled

`mul_mm.comp:17-19` has:
```glsl
#ifdef COOPMAT
#extension GL_KHR_cooperative_matrix : enable
```

llama.cpp compiles `mul_mm.comp` with `-DCOOPMAT` when
`device->coopmat_support == true` (RDNA4 advertises KHR coopmat 1).
Our `build.rs` does **not** define COOPMAT for `mul_mm.comp`. SPV
disasm confirms our `mul_mm_q4_k_f32.spv` declares neither
`OpCapability CooperativeMatrixKHR` nor `OpExtension
"SPV_KHR_cooperative_matrix"`.

But `mul_mm.comp` is **not our prefill path** — `mul_mmq.comp` is.
The COOPMAT path on `mul_mm.comp` only matters if we ever route
prefill through it (currently ~45 % slower per CHANGELOG v0.1.3).
Likely a smaller win than M-tile, but if the COOPMAT-`mul_mm` path
is materially faster than scalar `mul_mmq` at large M+N+K, it's a
sprint candidate.

### C. Spec-constants on S-tile may be locally optimal but globally
suboptimal

Our S-tile uses `BM=64 BN=64 WM=WN=32 WMITER=2 TM=2 TN=4`,
empirically tuned in Sprint 4 of v0.2 (`pipeline_registry.rs:200-212`
sweep log) to 740 tok/s on RDNA4. Llama.cpp's S-tile is
`BM=BN=32 BLOCK_SIZE=64 WMITER=2 TM=2 TN=1` (very different shape).
Sprint 4 already tested several llama.cpp-shaped configurations and
they regressed:

```
BS=256 WMITER=2 TM=4 TN=2  → 691 tok/s (-7%)   [llama.cpp's M-tile shape]
BS=128 WMITER=1 WM=WN=32   → 596 tok/s (-19%)  [llama.cpp's S-tile shape]
BS=64  WMITER=1            → 532 tok/s (-28%)
BS=256 WMITER=2 TM=2 TN=4  → 740 tok/s         [v0.1.3 default — current]
```

Our S-tile is locally optimal **for its specific (m, n, k) shape
distribution** (small-mixed-prompt 15-prompt bench at v0.1.3). It may
not be optimal for the L-tile-cousin shape (n in 64..256 zone).
That's M-tile territory — a *new* spec-constant set, not a sweep of
the existing one.

### D. We don't ship `mul_mat_vecq.comp` for decode

Llama.cpp's decode path uses `mul_mat_vecq.comp` (a separate GEMV
shader for Q8_1-vec inputs, generic K-quants), not their per-quant
`mul_mat_vec_q4_k.comp`. We use the per-quant version. This affects
decode, not prefill — and our decode is already at 0.79× llama.cpp,
so the impact here is much smaller than the prefill gap.

### E. K-tile depth (`BK=32`) is hardcoded

`vk_shaders/mul_mmq.comp:82`:
```glsl
#define BK 32
```

For the down-projection (K=12288), 12288/32 = 384 K-loop
iterations per WG. A wider BK (64 or 128) would halve or quarter
the K-loop overhead. Llama.cpp's K-quant tiles use BK=32 in
`l_warptile_mmq_int_k` too — but the `_int` (non-K-quant) path uses
BK=64 in `m_warptile_mmq_int`. K=12288 is the down-projection only;
the gain is bounded.

## Sprint 11 phase plan — revised after 11D pre-check

The brief's plan ("port int_dp shader") is moot. Replacement plan,
ordered by expected impact:

### Phase 11E — **M-tile** mul_mmq

Add a second new pipeline (BM=BN=64, BK=32, BLOCK_SIZE=128, WMITER=2,
TM=2, TN=2). Different from S-tile (BLOCK_SIZE=128 not 256, WMITER=2
not 2 with smaller WM/WN, TN=2 not 4). Threshold: `m > 64 && n > 64`
without the L gate (i.e., M routes when neither S nor L do).

Same SPV (`mul_mmq.comp` unchanged). Same approach as 11C: add
ShaderId, add pipeline_registry branch, extend selector. Estimated
1 day. Expected gain: +5–15 % at the 64<n<256 zone (which we already
left on the table per Sprint 11C analysis).

### Phase 11F — **Aligned** mul_mmq variants

`mul_mm` already has aligned variants in our codebase (`MulMmQ4KAligned`,
`MulMmQ6KAligned`) — `LOAD_VEC_B=4` skip-the-K-tail-bounds-check at
`seq_len % 4 == 0`. `mul_mmq` does not. Mirror the build.rs entry
for mul_mmq, add the runtime selector for `seq_len % 4 == 0`.

Estimated 1 day. Expected gain: +2–4 % at aligned shapes (typical
prefill is `seq_len = 512, 1024, 2048` → all aligned).

### Phase 11G — **Wider BK** for down-projection

Investigate compiling a separate `mul_mmq.spv` with `BK=64` or
`BK=128` for the down-projection (K=12288). `BK` is a `#define` not
a spec-constant, so this needs a separate SPV via build.rs.

Estimated 2 days. Expected gain: +3–5 % on the down-projection only,
which is one of seven prefill GEMMs per layer — so end-to-end ~0.5–1 %.
**Probably not worth it** unless 11E+11F leave us still with a real
gap.

### Phase 11H — **`mul_mm.comp` with COOPMAT define**

Compile an extra `mul_mm` SPV with `-DCOOPMAT`. Same shader,
different define. Run the same Sprint 4 sweep against `mul_mmq` to
see if `mul_mm + COOPMAT` beats `mul_mmq + integer-DP` on RDNA4.

Estimated 2–3 days (build.rs + bench + selector + parity tests).
**Risk:** llama.cpp's preference of `mul_mm + COOPMAT` over
`mul_mmq` for large prefill on coopmat-enabled hardware suggests
this is the actual win — but our v0.1.3 changelog already says
"mul_mm is ~45% slower than mul_mmq at prefill." The gap is from
when mul_mm was scalar-only; with COOPMAT enabled the picture
might invert. **High variance, large potential payoff.**

### Phase 11I — Eigenbau KHR coopmat GEMM (parked)

Sprint 11A's parked candidate. Defer until 11E–H numbers are in.

## What I did this sprint

- Read `mul_mmq.comp`, `mul_mmq_funcs.glsl`, `device.rs`, llama.cpp's
  `vulkan-shaders/` directory.
- `spirv-dis`'d our compiled SPVs.
- Ran `vulkaninfo` to confirm device-side support.
- Compared file lists between repos.
- Wrote this report.

**No source / shader / build / runtime changes.**

## Files touched

- `results/v021_sprint11d_integer_dp.md` (this report — new)

## Tests / regression

Not run — no code changes. Sprint 11C's 169/169 still applies.

## Take-aways

1. **Pre-check streak: 8b → 9c → 11A → 11D.** Four sprints in three
   weeks where the brief's premise didn't match the codebase. The
   hypothesis-check memory keeps paying for itself.
2. **Bit-identical port covers more ground than expected.** The
   v0.1.x mul_mmq.comp port brought integer-DP, Q8_1 activations,
   and 8-bit weight unpacking all together — three separately-named
   "endgame levers." None of them are ports we still need to do.
3. **The 0.46× gap is structural, not single-feature.** Closing it
   needs M-tile + aligned + maybe COOPMAT-mul_mm, in that order.
   No single sprint will get us to 1.0× — but Phase 11E (M-tile)
   alone could shave another 5–15 % at common prefill shapes for a
   day's work.
4. **`grep` your repo before reading the brief.** Two minutes saved
   four weeks of reimplementation. Worth memorialising the workflow
   in MEMORY.md.
