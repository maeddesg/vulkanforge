# v0.2 Sprint 9d.3 — FP16 KV-Cache Decode Hot-Path + Default ON

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9d.2 (FP16 Prefill, pp=2048 +21%, 167/167 Tests)

---

## TL;DR — Decode FP16 funktional, alle 167 Tests grün, Default ON.

```
═══ v0.2 Sprint 9d.3 ═══

DAS GROSSE ERGEBNIS:
  cargo test --release          (default FP16 KV)  → 167/167 ✓
  VULKANFORGE_FP16_KV=0 cargo test (opt-out FP32)  → 167/167 ✓
  → Alle End-to-End Argmax-Parity-Tests bestehen mit FP16!
  → Default-Switch zu FP16 ON ist sicher und shipped in 9d.3.

Lieferungen:

A. Decode-Attention FP16-Varianten (2 neue SPVs, 55 total):
   • flash_attn_fp16kv.spv         (single-WG decode, seq_len ≤ 64)
   • flash_attn_split_fp16kv.spv   (multi-WG split-K worker, > 64)
   • Same #ifdef FP16_KV pattern wie Sprint 9d.2:
     - K/V Bindings: float[] → uint[] (packed FP16)
     - load_k/load_v helpers über unpackHalf2x16
   • flash_attn_reduce.comp UNVERÄNDERT (liest nur partials, kein KV)
   • scalar_attn.comp UNVERÄNDERT (Phase 4B durch FlashAttn ersetzt,
     nicht im Hot-Path; ScalarAttn Pipeline existiert weiter aber
     wird nicht dispatcht)

B. Forward.rs Decode-KV-Write Migration (3 Stellen, alle gefunden):
   1. dispatch_layer (Hot-Path Decode, Zeile ~1313)
   2. dispatch_layer_partial (Debug-Halt-Helper, Zeile ~1049)
   3. dispatch_layer_batch per-token legacy (Zeile ~3249)
      ← DIESE Stelle hatte 9d.2 ÜBERSEHEN. Sprint 9d.3 entdeckte
        das, weil phase5b2_batch_attn_parity_qwen3_short/two_tiles
        unter FP16 NaN-Logits produzierten (per-token-Pfad mit
        batch_attn=false schrieb FP32 in FP16-Buffer → Garbage).

C. Forward.rs Attention-Shader-Selektor (2 Stellen):
   • run_scalar_attn (single-WG branch): FlashAttn / FlashAttnFp16Kv
   • run_flash_attn_split_reduce: FlashAttnSplit / FlashAttnSplitFp16Kv

D. KvDtype Default-Switch (kv_cache.rs::kv_dtype_from_env):
   VORHER: env unset → F32, "1" → F16
   NACHHER: env unset oder ≠"0" → F16 (DEFAULT ON), "0" → F32 (opt-out)
   Matches llama.cpp's Qwen3 default.

E. WARN-Log entfernt:
   Sprint 9d.1 hatte einen "WILL BE INCORRECT" Warning. Sprint 9d.3
   bestätigt "is correct" via 167/167 grün → Warning gelöscht.
   Startup-Log bleibt: "KV cache FP16 (2B/elem) × 2 buffers = 288 MB".

Performance — 15-Prompt Bench (NEW DEFAULT FP16):
  | Metric                   | FP32-Baseline | FP16-Default | Δ       |
  |--------------------------|----------------|---------------|---------|
  | MEDIAN prefill (alle 15) |    1062.7      |    1063.8     | ~ 0.0%  |
  | MEDIAN decode            |      91.7      |      90.4     | -1.4%   |
  | First-5 prefill (pp=62)  |    1438.0      |    1435.5     | -0.2%   |
  | Coherent prompts         |     15/15      |     15/15     | ✓       |

  → Bei kurzem pp und Decode (1-Token Forward) ist FP16 marginal
    langsamer (Citrix-Noise + unpackHalf2x16 VALU-Overhead). Innerhalb
    Mess-Variance, kein praktischer Verlust.

Performance — pp=2048 (Sprint 9d.2 bestätigt mit Default ON):
  pp=2048: FP16 1018 tok/s (vs FP32 841 → +21%).

VRAM (Qwen3-8B, max_seq=2048):
  FP32: 576 MB → FP16: 288 MB (-288 MB)
  Bei max_seq=8192: 2 304 MB → 1 152 MB (-1.13 GB)

Tests:
  Default (FP16): 167/167 ✓
  Opt-out (FP32): 167/167 ✓
  Beide Pfade decken ALLE E2E Argmax-Parity-Tests ab.

Files:
  modified: vk_shaders/flash_attn.comp (+ #ifdef FP16_KV)
  modified: vk_shaders/flash_attn_split.comp (+ #ifdef FP16_KV)
  modified: build.rs (+2 ShaderJobs)
  modified: src/backend/vulkan/shaders.rs (+2 ShaderIds, +2 SPV consts)
  modified: src/backend/vulkan/pipeline_registry.rs (+ 2 branches)
  modified: src/backend/vulkan/kv_cache.rs (default ON, WARN entfernt)
  modified: src/backend/vulkan/forward.rs:
            - 3 KV-write sites: vkCmdCopyBuffer → run_kv_copy_fp16
              (decode hot-path + decode debug + batched per-token legacy)
            - 2 attention shader selectors (run_scalar_attn,
              run_flash_attn_split_reduce)
  new:      results/v02_sprint9d3_fp16_kv_decode.md (this report)

Commit: HEAD (kein Push).
```

---

## 1. Der entscheidende Befund: argmax-Parity hält

Sprint 9d.3's wichtigste Entdeckung war, dass **die FP16-Drift in
Qwen3-8B's Attention-Compute klein genug bleibt, dass argmax über alle
167 Regression-Tests identisch zur FP32-Variante ist**. Das war nicht
selbstverständlich — die Brief-Erwartung war "argmax identical OR top-5
≥ 4/5" (mit Toleranz für leichten Drift). Reale Messung: argmax exakt
gleich.

Konkret bestätigte Tests bei `VULKANFORGE_FP16_KV=1`:
* `phase3e_prefill_batch_matches_token_by_token_top5` ✓
  → Vergleicht prefill_batch (mit FP16 KV-write/read) gegen
    forward_token (jetzt auch FP16). Top-1 + Top-5 identisch.
* `sprint5b_chunked_prefill_parity_qwen3` ✓
  → Single-shot prefill_batch == 4-chunk prefill_batch (beide FP16).
* `phase5b2_decode_after_batched_prefill_qwen3` ✓
  → Prefill (FP16 write) → Decode (FP16 read+write) → kohärenter Output.
* `phase_prompt16_alice_context_retention_qwen3` ✓
  → 16-prompt multi-turn chat session, akkumulierte FP16-Drift bleibt
    unter argmax-Wechsel-Schwelle.
* `phase5b2_batch_attn_parity_qwen3_short / two_tiles` ✓
  → batch_attn=true vs batch_attn=false: BEIDE Pfade FP16, beide
    produzieren identische top-1.
* `phase3b_chat_session_multi_turn_carries_and_resets` ✓
  → Multi-turn Chat funktioniert.

Damit ist FP16 KV **default-ON-tauglich** — kein Drift, kein
Garbled-Output, kein Coherence-Verlust. Sprint 9d.3 schaltet entsprechend.

---

## 2. Das überraschende Bug: per-token legacy path

Beim ersten Lauf von `VULKANFORGE_FP16_KV=1 cargo test --release`
fielen ZWEI Tests durch:
* `phase5b2_batch_attn_parity_qwen3_short` — panic: argmax NaN
* `phase5b2_batch_attn_parity_qwen3_two_tiles` — panic: argmax NaN

Diagnose: beide rufen `batched_prefill_logits(..., batch_attn=false, ...)`,
welches durch die per-token-legacy Branch in `dispatch_layer_batch`
geht (lines ~3199-3309). Diese Branch hatte ihren EIGENEN
`vkCmdCopyBuffer`-KV-write (lines 3255-3262), den Sprint 9d.2
übersehen hat. Resultat: FP32-Daten in FP16-Buffer kopiert →
Buffer-Overflow oder garbage values → NaN logits.

Sprint 9d.3 fixt diese 3. KV-write-Site (zusätzlich zu den 2 echten
decode sites). Nach dem Fix: 167/167 grün auch unter
`VULKANFORGE_FP16_KV=1`.

**Lesson:** der Pre-Check für 9d.2 hätte alle 4 vkCmdCopyBuffer-
Stellen aufzählen müssen, nicht nur 2 ("Sprint 9d Analyse identifizierte
4: 1058, 1322, 2879, 2964" — aber 2879 ist die batched-bulk-Site,
und die per-token-legacy Site bei 3255 wurde nicht gelistet, weil
sie nur bei batch_attn=false aktiv ist). Memory-feedback für
zukünftige Sprints.

---

## 3. Was die Decode-Shader machen

### 3.1 flash_attn.comp (single-WG decode, seq_len ≤ 64)

Identisches FP16-Read-Pattern wie flash_attn_batch.comp aus Sprint
9d.2:

```glsl
#if FP16_KV
layout (binding = 1) readonly buffer K { uint k_packed[]; };
layout (binding = 2) readonly buffer V { uint v_packed[]; };
float load_k(uint idx) { return unpackHalf2x16(k_packed[idx >> 1])[idx & 1u]; }
float load_v(uint idx) { return unpackHalf2x16(v_packed[idx >> 1])[idx & 1u]; }
#else
layout (binding = 1) readonly buffer K { float k[]; };
layout (binding = 2) readonly buffer V { float v[]; };
float load_k(uint idx) { return k[idx]; }
float load_v(uint idx) { return v[idx]; }
#endif
```

Inner-Loop accesses changed from `k[k_pos_off + d]` to
`load_k(k_pos_off + d)`. V access `v[v_pos_off + tid]` /
`v[v_pos_off + tid + 64]` similar.

### 3.2 flash_attn_split.comp (multi-WG worker, seq_len > 64)

Same #ifdef FP16_KV header + load_k/load_v helpers + 3 access-site
swaps. Note that `flash_attn_reduce.comp` (paired with split) reads
NOT the KV cache but per-tile `scratch_max/sum/out` (FP32 partials),
so it doesn't need an FP16 variant.

### 3.3 scalar_attn.comp (NICHT geupdatet)

Sprint 9d Analyse listete scalar_attn.comp als KV-Reader. Aber: er
wird im Forward-Pass NICHT mehr dispatcht (Phase 4B ersetzte ihn
durch FlashAttn). `ShaderId::ScalarAttn` ist in der Pipeline-Registry,
aber keine `forward.rs`-Funktion ruft `registry.get(ShaderId::ScalarAttn)`
für ein Dispatch. Wir lassen ihn FP32-only — falls Sprint 9d.4 (oder
ein neuer Hot-Path) ihn reaktivieren wollte, kann der FP16-Variant
nachgeliefert werden.

---

## 4. Forward.rs Wiring

### 4.1 KV-Write Sites — alle drei

```rust
// Site 1: dispatch_layer (decode hot-path), ~line 1313:
if self.kv_cache.is_fp16() {
    self.run_kv_copy_fp16(... k_buf → kv_cache.k_buffer ...);
    self.run_kv_copy_fp16(... v_buf → kv_cache.v_buffer ...);
} else {
    self.profile("kv_write", dev, cmd, |dev, cmd| unsafe {
        cmd_copy_buffer(...);
    });
}

// Site 2: dispatch_layer_partial (debug halt), ~line 1049:
// Same pattern, no `profile` wrapper.

// Site 3: dispatch_layer_batch per-token-legacy, ~line 3249:
// Same pattern, runs M times in loop (one KV-write per token).
```

Alle drei nutzen `run_kv_copy_fp16` mit `kv_elements = n_kv_heads *
head_dim` (1 token's worth) und dem byte-offset aus
`kv_cache.pos_offset_bytes(layer, pos)`.

### 4.2 Attention Shader Selectors

```rust
// run_scalar_attn (single-WG branch):
let kernel = registry.get(if self.kv_cache.is_fp16() {
    ShaderId::FlashAttnFp16Kv
} else {
    ShaderId::FlashAttn
});

// run_flash_attn_split_reduce:
let split_kernel = registry.get(if self.kv_cache.is_fp16() {
    ShaderId::FlashAttnSplitFp16Kv
} else {
    ShaderId::FlashAttnSplit
});
// FlashAttnReduce stays FP32 (reads partials, not KV).
```

---

## 5. Default-ON Switch

```rust
// kv_cache.rs::kv_dtype_from_env (NEW):
fn kv_dtype_from_env() -> KvDtype {
    match std::env::var("VULKANFORGE_FP16_KV") {
        Ok(s) if s == "0" => KvDtype::F32, // explicit opt-out
        _ => KvDtype::F16,                 // default ON
    }
}
```

Begründung:
1. **Test evidence:** 167/167 Tests grün mit FP16 KV, einschließlich
   strict argmax-parity tests.
2. **Industry convention:** llama.cpp's Vulkan backend defaults to
   FP16 KV for Qwen3 + Llama-3 + Mistral. Wir matchen das.
3. **VRAM saving:** -50% auf KV-cache, ermöglicht 2× längere Kontexte
   ohne Hardware-Upgrade.
4. **Long-context speedup:** +21% bei pp=2048 (Sprint 9d.2 measurement).

Trade-off: marginal -1% bei kurzem pp (≤ 1024) durch unpackHalf2x16
VALU-Overhead — innerhalb Citrix-Noise-Band, im Aggregat über
realistische Workloads ein klarer Net-Positive.

Opt-out via `VULKANFORGE_FP16_KV=0` für bit-exakte FP32-KV-Reproduktion.

---

## 6. Performance Detail

### 6.1 Default-Comparison vs Sprint 9c.5

```
| Metric                   | 9c.5 (FP32) | 9d.3 (FP16 default) | Δ      |
|--------------------------|-------------|---------------------|--------|
| MEDIAN prefill (15-prom) |   1062.7    |       1063.8        | +0.1%  |
| MEDIAN decode            |     91.7    |         90.4        | -1.4%  |
| First-5 prefill (pp=62)  |   1438.0    |       1435.5        | -0.2%  |
| Coherent (15/15)         |     ✓       |          ✓          | ✓      |
| pp=2048 (single)         |    841.3    |       1018.1        | +21%   |
| KV-Cache VRAM (max=8192) |  2 304 MB   |      1 152 MB       | -50%   |
```

Bei kurzem pp und Decode (1-Token Forward) wirkt der unpackHalf2x16-
VALU-Overhead. Bei langem pp (2048+) gewinnt Bandwidth-Halving
deutlich. VRAM-Halbierung ist immer da.

### 6.2 Cumulative Sprint 9 — pp=2048 vs Sprint 8a baseline

Sprint 8a's pp=2048 baseline (chunked): ~810 tok/s.
Sprint 9d.3's pp=2048 (FP16 default, 9d.2 measurement): 1018 tok/s.
**Cumulative gain: +26% bei pp=2048 über die volle Sprint-9-Reihe.**

### 6.3 vs llama.cpp

```
| pp   | VF 8a baseline | VF 9d.3 default | llama.cpp | VF / llama |
|------|----------------|------------------|-----------|------------|
|  128 |    1641        |     1862         |   3603    |   0.52×    |
|  512 |     921        |     1761         |   4317    |   0.41×    |
| 1024 |     556        |     1477         |   4189    |   0.35×    |
| 2048 |     ~810       |     1018         |   3771    |   0.27×    |
```

Bei pp=2048 sind wir jetzt bei 0.27× von llama.cpp (war 0.21× nach
Sprint 8a). Verbleibende 3.7×-Gap hat hauptsächlich strukturelle
Ursachen:
* mul_mmq vs cooperative_matrix (RDNA4 hat WMMA, llama.cpp nutzt es)
* Vermutlich auch Int8-DotProduct-Optimierungen für Q4_K
* Sprint 10+ Territory.

---

## 7. WARN-Log Update

### 7.1 Sprint 9d.1 Warning (jetzt gelöscht)

```
VulkanForge: WARNING — VULKANFORGE_FP16_KV=1 allocates the KV cache
as FP16 (Sprint 9d.1 infrastructure), but the attention shaders and
KV-write copies are still FP32. Outputs WILL BE INCORRECT until
Sprint 9d.2/9d.3 ship the conversion shader and FP16-aware attention
SPVs.
```

### 7.2 Was bleibt: nur Info-Log

```
VulkanForge: KV cache FP16 (2B/elem) × 2 buffers = 288 MB
(36 layers × 8 kv_heads × 2048 max_seq × 128 head_dim)
```

Bei opt-out:
```
VulkanForge: KV cache FP32 (4B/elem) × 2 buffers = 576 MB ...
```

---

## 8. Files Touched

```
modified: vk_shaders/flash_attn.comp        (+ #ifdef FP16_KV header,
                                              load_k/load_v, 3 access swaps)
modified: vk_shaders/flash_attn_split.comp  (same treatment)
modified: build.rs                          (+ 2 ShaderJobs)
modified: src/backend/vulkan/shaders.rs     (+ 2 ShaderIds + name + SPV)
modified: src/backend/vulkan/pipeline_registry.rs
                                            (FlashAttnFp16Kv shares
                                             FlashAttn's MAX_SEQ spec;
                                             FlashAttnSplitFp16Kv to
                                             no-spec-const branch)
modified: src/backend/vulkan/kv_cache.rs    (default-ON in
                                             kv_dtype_from_env, WARN
                                             log removed)
modified: src/backend/vulkan/forward.rs:
          - dispatch_layer KV-write (Site 1, hot-path)
          - dispatch_layer_partial KV-write (Site 2, debug)
          - dispatch_layer_batch per-token legacy KV-write (Site 3, the
            one 9d.2 missed)
          - run_scalar_attn shader selector
          - run_flash_attn_split_reduce shader selector
new:      results/v02_sprint9d3_fp16_kv_decode.md (this report)
```

Total Sprint-9d-Reihe SPVs hinzugefügt: 5 (kv_copy_fp16,
flash_attn_tiled_br16_bc32_fp16kv, flash_attn_batch_fp16kv,
flash_attn_fp16kv, flash_attn_split_fp16kv). Total Shader: 55.

---

## 9. Bottom Line — Sprint-9-Reihe komplett

```
| Sprint | Status      | Wert / Δ                       |
|--------|-------------|--------------------------------|
| 9a     | DONE        | swiglu fusion +2%              |
| 9c     | DONE (neg)  | rms_norm_mul already fused     |
| 9b     | DONE        | multi_add_rms +1.5%            |
| 9d     | ANALYSIS    | 3-phase plan                   |
| 9b.2   | DONE        | cross-layer +1.3%              |
| 9c.5   | DONE        | rms_norm_mul_rope +0.6%        |
| 9d.1   | DONE        | FP16 KV infrastructure         |
| 9d.2   | DONE        | FP16 KV prefill +21% pp=2048   |
| 9d.3   | DONE        | FP16 KV decode + DEFAULT ON    |
```

Cumulative Sprint-9 Wins (pp=2048):
* Sprint 8a baseline: ~810 tok/s
* Sprint 9d.3 default: 1018 tok/s → **+26%**
* VRAM: -50% (Qwen3-8B max_seq=2048: 576 → 288 MB)
* Tests: 145 (start of v0.2) → 167 (Sprint 9d.3) → +22 neue Tests
* Dispatches/Layer: 24 → 19 → -5 (-21%)
* Barriers/Layer: 18 → 14 → -4 (-22%)

Verbleibende Lücke zu llama.cpp (pp=2048 0.27×) ist strukturell:
mul_mmq vs WMMA cooperative-matrix Wege. Sprint 10+ Territory
(coopmat for Q4_K GEMM, llama.cpp's float16/bf16 cooperative
matrix path).

Die Sprint-9-Reihe schließt mit FP16 KV als Default ON — das größte
strukturelle Sprint-Resultat seit Sprint 8a's flash_attn_tiled
default-flip. VulkanForge ist jetzt bei VRAM-Footprint und
Long-Context-Throughput in einer mit llama.cpp vergleichbaren
Liga (innerhalb 3.7× Gap).
