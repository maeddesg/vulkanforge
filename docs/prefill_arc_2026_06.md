# Prefill Arc (2026-06) — Gemma-4-26B-A4B: 198 → 1073 t/s, and one documented dead end

State as of **v0.6.0**. All numbers: Gemma-4-26B-A4B Q3_K_M, `vulkanforge serve`, KV-FP8, ctx 4096,
RX 9070 XT (RADV gfx1201), real prompts, cold prefill, median of 3. Reference: llama.cpp `llama-server`
on identical GGUFs/prompts ≈ 2785 t/s @p512 / 2837 @p2048.

## The arc

| Step | Change | p512 | p2048 |
|---|---|---:|---:|
| v0.5.8 baseline | (51D-AN per-token Full layers + MoE mini-dispatch loops) | 198 | 178 |
| v0.5.9 `VF_PREFILL_FULL_BATCHED` | Full layers batched (51D-AK/AM defects no longer reproduce) | 317 | 266 |
| + `VF_MOE_GLU_BATCHED` + `VF_MOE_GATHER_COMBINE` | MoE loops → 2 dispatches/layer (byte-identical) | 1025 | 602 |
| **v0.6.0 default** (all three ON) | | **1073** | **604** |

Mechanisms removed, per the Sprint-2/4 profiles (`results/sprint2_prefill_batchpath_profile.md`,
`results/sprint4_expert_ffn_profile.md`):
- **Per-token Full layers** (5/30): batch=1 GEMVs + 1 full GPU drain *per token per layer* (10,300 drains
  @p2048) = 76–80 % of prefill GPU time. Removed by `VF_PREFILL_FULL_BATCHED` — the historical batched-path
  bugs (51D-AK cos-0.13 attention, 51D-AM FFN) were fixed in the interim (52K weight-type fix, KV-layout work,
  v0.5.3 bucket-offset audits) and verified gone (bit-identity vs an independent per-query reference).
- **MoE expert-FFN mini-dispatch loops**: 251,520 dispatches + 125,760 accumulation barriers per 524-token
  chunk — 58 % CPU-record-bound + 35 % barrier-serialization, only 6 % GPU compute. Removed by one batched
  GLU dispatch + one gather-weighted-sum dispatch per layer (`FmaReduceBatch`), byte-identical by
  FP-order-preserving construction.

What remains @p2048 (Sprint-6 decomposition, `results/sprint6_rest_wall_profile.md`): **attention 77 %**
(sliding 50 % — O(n×512) across 25 layers; full 28 % — clean O(n²) on 5 layers), expert GEMM 16 %
(MMQ, amortizing — CoopMat headroom ≤16 %), rest 6 %. The attention core is **mem-bound on the KV stream**
(KV FP8→FP16 ⇒ attention wall +64…93 %).

## The documented dead end: Q-tiled flash attention (Sprint 7, commit `eeef090`)

**Do not naively repeat this.** The obvious fix for the mem-bound attention — Br-query-tiled flash attention
(stage one K-tile in LDS, share it across Br queries; `flash_attn_tiled.comp` generalized to HEAD_DIM 256/512,
gated `VF_FA_TILED_SLIDING` / `VF_FA_TILED_FULL`) — is **numerically perfect and 31–36 % SLOWER**:

- Correctness fully validated: single-tile prompts byte-identical to `fa_batch`; multi-tile cos 0.984–1.0
  with no layer cliff; long-context recall (fact at position ~10 retrieved after ~1400 tokens) green.
- The traffic goal **was achieved**: KV-format sensitivity drops 5× (+345 ms vs +1706 ms @p2048 for FP8→FP16).
- But the **41/40 KB LDS tiles starve workgroup occupancy** (1–3 resident WGs/WGP vs `fa_batch`'s 256 B LDS
  with massive WG parallelism + L2 absorption), and the barrier-separated synchronous K-staging exposes
  latency. hd512's BC=16 additionally idles 75 % of score threads. Net: sliding +19 % wall, full ×2.6.

**Lesson (RDNA4): KV-traffic reduction that costs occupancy is a net loss.** `fa_batch`'s redundant KV
streaming is hidden better by parallelism than the roofline suggests. Consequences:
- **CoopMat in the attention core is NOT the next step** — the tiled path is occupancy-bound, not math-bound.
- If attention is attacked again, the directions are: much smaller staged tiles / double-buffered (async)
  K-staging, a no-LDS variant (Q in registers, K straight from L2), GQA-group dedup to cut the WG count
  (8 Q-heads re-read the same KV on full layers, 2 on sliding), or subgroup shuffles instead of LDS.
- The validated HEAD_DIM-parametric tiled family (FP8/FP16 KV) stays in-tree, default-OFF, as the platform
  for that tuning — the numerics never have to be re-proven.

## Flags (v0.6.0)

Default-ON (opt-out with `=0`, coupled set): `VF_PREFILL_FULL_BATCHED`, `VF_MOE_GLU_BATCHED`,
`VF_MOE_GATHER_COMBINE`. Note: `VF_PREFILL_FULL_BATCHED` changes default greedy output vs ≤v0.5.9
(FP-reorder between per-token and batched paths, amplified by MoE routing flips; both validated correct).
Experimental, default-OFF, not recommended: `VF_FA_TILED_SLIDING`, `VF_FA_TILED_FULL`.
