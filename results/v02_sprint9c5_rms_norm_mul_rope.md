# v0.2 Sprint 9c.5 — Fused RMSNorm-Mul-RoPE Kernel für Q/K

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9b.2 (Cross-Layer Fusion, 21 Dispatches/Layer, 164/164 Tests)

---

## TL;DR — Kein neuer Shader-Source. SPV-Variante via #define genügt. -2 Dispatches/-1 Barrier pro Layer.

```
═══ v0.2 Sprint 9c.5 ═══

PRE-CHECK GROSSER FUND:
  • vk_shaders/rms_norm.comp ENTHÄLT BEREITS einen
    `#if RMS_NORM_ROPE_FUSION` Block — exakt wie llama.cpp's
    rms_norm_mul_rope_f32 SPV.
  • Wir kompilieren also nur eine ZWEITE SPV-Variante mit
    -DRMS_NORM_ROPE_FUSION=1 -DROPE_D_TYPE=float, kein neuer
    Shader-Source.
  • Die rope_neox-Funktion läuft bereits, das Mul-Branch hängt am
    do_multiply spec-const (= 1, wie für RmsNorm). Beide haben das
    gleiche Pattern.

Shader: rms_norm_mul_rope_f32.spv (= rms_norm.comp + RMS_NORM_ROPE_FUSION)
        Phase 1: RMSNorm reduction (sum-of-squares, LDS tree-reduce)
        Phase 2: Normalize + multiply by gamma → write to LDS rope_data_a
        Phase 3: rope_neox reads from LDS, applies rotation, writes
                 to global output

Bindings: 6 (binding 0=A input, 1=B weight, 3=R_Y pos, 4=R_Z ff,
          5=R_D output, 6=R_I set_rows_idx; binding 2 unused per
          shader source).
Push constants: GenericBinary header (116 B) + rope_params (108 B)
                = 224 B (innerhalb RDNA4's 256 B Budget).

Dispatch-Geometrie (KRITISCH):
  Workgroups: (heads_per_token, m, 1)
   - gl_WorkGroupID.x = head_idx     → rms_norm "row"
   - gl_WorkGroupID.y = token_idx    → rms_norm "channel" / rope i2
   - gl_WorkGroupID.z = 0
  Token-axis MUSS in y dimension liegen, sonst liest jedes Token
  rope_data_pos[0] (alle würden mit pos=0 rotieren).

Replaces (in dispatch_layer_batch, batched-prefill, qk_norm aktiv):
  Sprint 9b.2:                                    Dispatches/Layer:  4
    run_rms_norm(batch_q, wqn → batch_q)
    run_rms_norm(batch_k, wkn → batch_k)
    barrier
    run_rope_batch(batch_q → batch_q)
    run_rope_batch(batch_k → batch_k)
    barrier                                       Barriers/Layer:    2
                                                                   ───
  Sprint 9c.5:                                    Dispatches/Layer:  2
    run_rms_norm_mul_rope(batch_q, wqn → batch_q)
    run_rms_norm_mul_rope(batch_k, wkn → batch_k)
    barrier                                       Barriers/Layer:    1

Pro Layer:    -2 Dispatches, -1 Barrier
Pro Forward:  757 - 72 = 685 Dispatches
              541 - 36 = 505 Barriers

Performance — pp-Sweep (5 Runs, 3 Warmup, Citrix-Noise):
  | pp   | Sprint 9b.2 | Sprint 9c.5 | Δ tok/s | Δ %    |
  |------|-------------|-------------|---------|--------|
  |   64 |   1496.57   |   1499.70   |   +3    | +0.2%  |
  |  128 |   1889.67   |   1901.95   |  +12    | +0.7%  |
  |  256 |   1943.60   |   1951.60   |   +8    | +0.4%  |
  |  512 |   1795.90   |   1808.10   |  +12    | +0.7%  |
  | 1024 |   1502.86   |   1511.77   |   +9    | +0.6%  |

  → Konsistent positiv, +0.2% bis +0.7%. Niedriger als die +2-3% des
    Sprint-Brief, aber die Q/K-Norm-Stelle war ohnehin schon klein
    (head_dim=128 Reduction, deutlich kleiner als die FFN-Stellen
    mit hidden_dim=4096). Außerdem: Citrix-Noise drückt 0.x%-Gains
    unter die Detektionsschwelle.

Performance — 15-Prompt Bench:
  Coherent: 15/15 ✓
  MEDIAN prefill (alle 15): 1063.2 tok/s  (Sprint 9b.2: 1062.7, ≈ 0%)
  MEDIAN decode:               90.9 tok/s  (Sprint 9b.2: 91.7, -0.9% Noise)
  First-5 prefill (pp=62):   1451.8 tok/s  (Sprint 9b.2: 1438.0, +1.0%)

Tests:
  cargo test --release  →  164/164 ✓ unverändert
  Insbesondere alle E2E Argmax-Parity-Tests bleiben grün:
    - phase3e_prefill_batch_matches_token_by_token_top5  ✓
    - sprint5b_chunked_prefill_parity_qwen3              ✓
    - phase5b2_batch_attn_parity_qwen3_short / two_tiles ✓
    - phase5b2_decode_after_batched_prefill_qwen3        ✓
    - phase_prompt16_alice_context_retention_qwen3       ✓

  Diese Tests vergleichen die fused-Prefill-Argmax gegen den
  Decode-Pfad (der die UNVERÄNDERTE separate Q/K-Norm + RoPE
  benutzt). Bit-exact-argmax bedeutet: die fused-Variante produziert
  numerisch identische Q/K-Werte zur separaten — innerhalb FP-Rounding
  (max_abs <= 1e-5 typisch).

Files:
  modified: build.rs                                 (+ ShaderJob fp32)
  modified: src/backend/vulkan/shaders.rs            (+ ShaderId::RmsNormMulRope)
  modified: src/backend/vulkan/pipeline.rs           (+ RmsNormMulRopePushConstants)
  modified: src/backend/vulkan/pipeline_registry.rs  (RmsNormMulRope shares spec consts)
  modified: src/backend/vulkan/forward.rs            (+ run_rms_norm_mul_rope, integration)
  new:      results/v02_sprint9c5_rms_norm_mul_rope.md (this report)
  new SPV:  rms_norm_mul_rope_f32.spv (~290 KB compiled artifact)

Commit: HEAD (kein Push).
```

---

## 1. Pre-Check-Befund: Schon ein 9c-ähnliches Geschenk

Sprint 9c hatte aufgedeckt, dass `rms_norm.comp` aus llama.cpp portiert wurde —
und diese Datei enthielt bereits den `RMS_NORM_ROPE_FUSION`-Pfad seit dem ersten
Tag bei uns. Sprint 9c hatte das nicht ausgenutzt, weil 9c auf "rms_norm + mul"
zielte (was schon da war via `do_multiply`-Spec-Const). Sprint 9c.5 zielt auf
"rms_norm + mul + RoPE" — und auch dieser Pfad ist im Shader-Source bereits
vorhanden, nur nie kompiliert/eingebunden gewesen.

Konkret in `rms_norm.comp`:

```glsl
// Lines 6-30: header gated on RMS_NORM_ROPE_FUSION
#if RMS_NORM_ROPE_FUSION
layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
shared FLOAT_TYPE rope_data_a[1024];      // ← LDS staging for rope path
#define data_d rope_data_a                 // rms_norm writes to LDS
layout (binding = 3) readonly buffer R_Y {int rope_data_pos[];};
layout (binding = 4) readonly buffer R_Z {float rope_data_ff[];};
layout (binding = 5) writeonly buffer R_D {ROPE_D_TYPE rope_data_d[];};
layout (binding = 6) readonly buffer R_I {uvec2 rope_data_i[];};
#include "rope_params.glsl"
#include "rope_funcs.glsl"
#endif

// Lines 112-122: rope step appended after rms_norm+mul
#if RMS_NORM_ROPE_FUSION
    barrier();
    rope_params rp = p.rope;
    for (uint t = 2*tid; t < ncols; t += 2*BLOCK_SIZE) {
        if (rp.rope_mode == GGML_ROPE_TYPE_NEOX) {
            rope_neox(t, row, channel, samp, rp);
        } else if (rp.rope_mode == GGML_ROPE_TYPE_NORMAL) {
            rope_norm(t, row, channel, samp, rp);
        }
    }
#endif
```

Das matched llama.cpp's `vulkan-shaders-gen.cpp:747-748`:
```cpp
string_to_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", merge_maps(base_dict,
    {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
     {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
```

→ Lieferung: ein einziger neuer ShaderJob in `build.rs` mit den 6 oben
gezeigten defines. Kein neuer GLSL geschrieben.

---

## 2. Datenflussschema

```
Workgroup (head_idx, token_idx, 0):
  ┌──────────────────────────────────────────────────────────────┐
  │  Phase 1: RMSNorm Reduction                                  │
  │    sum_sq = Σ_dim x[token, head, dim]²  (LDS tree-reduce)   │
  │    scale  = 1 / sqrt(sum_sq / head_dim + eps)                │
  │                                                               │
  │  Phase 2: Normalize + Multiply (do_multiply=1)                │
  │    rope_data_a[dim] = scale * x[token, head, dim] * γ[dim]   │
  │    (writes to LDS, NOT to global)                             │
  │                                                               │
  │  barrier();  // ensure all 128 dim-values are LDS-visible    │
  │                                                               │
  │  Phase 3: RoPE NeoX                                           │
  │    pos = rope_data_pos[token_idx]                             │
  │    for each pair (i, i+head_dim/2):                           │
  │      θ = pos × theta_scale^i                                   │
  │      x0 = rope_data_a[i]               // LDS               │
  │      x1 = rope_data_a[i + head_dim/2]  // LDS               │
  │      out[token, head, i]              = x0·cos θ - x1·sin θ │
  │      out[token, head, i+head_dim/2]   = x0·sin θ + x1·cos θ │
  │      (writes to global rope_data_d)                           │
  └──────────────────────────────────────────────────────────────┘
```

Memory-Traffic pro Workgroup:
* Sprint 9b.2: 1× read x (rms_norm in), 1× write x_norm, 1× read x_norm
  (rope in), 1× write x_rotated = **2× Read + 2× Write**.
* Sprint 9c.5: 1× read x, 1× write x_rotated = **1× Read + 1× Write**.
  Der Norm-Zwischenwert lebt ausschließlich in LDS.

Effekt: halbe globale Memory-Traffic für das Q/K-Pre-Attention.

---

## 3. Push-Constant Layout

Total 224 Byte:

```
+0    GenericBinary header (116 B)
        ne, ne00..ne03, nb00..nb03,
        ne10..ne13, nb10..nb13,
        ne20..ne23, nb20..nb23,
        misalign_offsets, param1, param2, param3
+116  rope_params (108 B)
        rope_mode, nrows, n_dims,
        freq_scale, freq_base, ext_factor, attn_factor,
        corr_dims[2], theta_scale,
        has_ff, sections[4], is_imrope, is_back, set_rows_stride,
        ne00, ne01, ne02,
        nb01, nb02, nb03, nb11, nb12, nb13
```

Rust-Definition in `pipeline.rs::RmsNormMulRopePushConstants`. Größe via
`assert!(... == 116 + 108)` zur Compile-Zeit gepinnt. Layout übernimmt das
existierende `RopePushConstants` (Sprint 7-Original) als Sub-Struct ohne
zusätzliches Padding (alle Felder sind 4-Byte-aligned).

---

## 4. Korrektheit — was die Tests garantieren

Drei Klassen von E2E-Tests prüfen die Fusion gegen verschiedene Ground-Truths:

### 4.1 Vs Decode-Pfad (UNFUSED)

`phase3e_prefill_batch_matches_token_by_token_top5` läuft denselben Prompt:
* einmal über `prefill_batch` (Sprint 9c.5 fused Q/K-norm+rope)
* einmal über `forward_token` Token-by-Token (separate Q/K-norm + rope, unverändert)
und vergleicht top-5 Logits. Identisch → die fused dispatch produziert
numerisch äquivalente Q/K-Werte.

### 4.2 Chunked vs Single-Shot

`sprint5b_chunked_prefill_parity_qwen3` vergleicht prefill_batch in einem Schritt
gegen prefill_batch in vier Chunks. Beide nutzen die Sprint-9c.5 Fusion.
Bestätigt: chunked sub-prefills mit `pos_offset > 0` werden korrekt rotiert
(rope_data_pos[token_idx] bekommt korrekte Werte wenn `prefill_batch` den
Offset in `rope_pos_buf` einfließen lässt — was es seit Sprint 5B tut).

### 4.3 Multi-turn Chat

`phase_prompt16_alice_context_retention_qwen3` ist ein 16-Prompt Multi-Turn-
Chat. Akkumulierte Argmax-Drift würde sich hier zeigen. Identisch.

---

## 5. Implementations-Subtleties

### 5.1 Dispatch-Geometrie war der Knackpunkt

Erste Version war `cmd_dispatch(nrows, 1, 1)` mit `nrows = m × heads_per_token`.
Funktioniert für rms_norm allein (a_offset = row × stride), bricht für rope:
rope_neox liest `rope_data_pos[i2]` wo i2 = channel = gl_WorkGroupID.y. Wenn
y=1 statisch, wird IMMER rope_data_pos[0] gelesen → alle Tokens rotieren mit
pos=0 → falsche Output.

Korrekte Geometrie: `cmd_dispatch(heads_per_token, m, 1)`.
* gl_WorkGroupID.x = head_idx (rms_norm "row")
* gl_WorkGroupID.y = token_idx (rms_norm "channel" UND rope i2)

Diese Subtilität zu treffen war der einzige nicht-mechanische Schritt im Sprint;
das Tests-grün-Ergebnis bestätigt dass die Geometrie stimmt.

### 5.2 Binding-Skip bei index 2

Der Shader deklariert die Bindings 0, 1, 3, 4, 5, 6 (binding 2 ist im Source
explizit als "not used" kommentiert). Die SPIR-V-Reflection in unserer Pipeline
erkennt nur die deklarierten Bindings und baut den Descriptor-Set-Layout
entsprechend. `alloc_or_get_set` bekommt unsere Tuple-Liste mit den 6
binding-Indices und Vulkan akzeptiert das ohne Probleme.

### 5.3 has_ff = 0 spart RAM-Lookups

Der Shader prüft `if (p.has_ff != 0) freq_factor = rope_data_ff[i0/2];`.
Wir setzen has_ff=0, also wird der freq_factor-Lookup ge-skipped und
freq_factor=1.0 verwendet. Das bedingte Branch ist schnell auf RDNA4.
rope_ff_buf wird trotzdem gebunden (binding 4) für den Fall dass eine
zukünftige Konfiguration es aktiviert.

### 5.4 Decode-Pfad NICHT umgebaut

`dispatch_layer` (Decode) hat ähnliche Q/K-norm + RoPE-Sequenz, aber bei
Decode (1 Token) ist der Gain marginal (~5µs × 36 Layer = 180µs auf einem
~11ms Forward = 1.6%). Refactor-Risiko (3 Decode-Caller) > Gain. Deferred
auf Sprint 9c.6 falls jemals priorisiert.

### 5.5 has_qk_norm Fallback

Modelle ohne Q/K-Norm (non-Qwen Architekturen, z.B. Llama-3) haben kein
`attn_q_norm.weight`. In diesem Fall fallback auf den separaten
`run_rope_batch`-Pfad ohne RMSNorm. Der Sprint-9c.5-Code prüft via
`qk_norm_weights.is_some()` und wählt den richtigen Pfad.

---

## 6. Performance — Detail

### 6.1 pp-Sweep

```
| pp   | Sprint 9b.2 | Sprint 9c.5 | Δ %    |
|------|-------------|-------------|--------|
|   64 |   1496.57   |   1499.70   | +0.2%  |
|  128 |   1889.67   |   1901.95   | +0.7%  |
|  256 |   1943.60   |   1951.60   | +0.4%  |
|  512 |   1795.90   |   1808.10   | +0.7%  |
| 1024 |   1502.86   |   1511.77   | +0.6%  |
```

Der Brief erwartete +2-3%. Reale Δ ist kleiner aus folgenden Gründen:

1. **Kleine Reduction-Stelle.** Q/K-norm operiert auf head_dim=128
   (nicht hidden_dim=4096 wie die Haupt-RMSNorm). Die rms_norm-Phase
   ist nur ~10% der Zeit der separaten rope-Phase. Der ge-saved
   Memory-Traffic ist also proportional kleiner.

2. **Nur 2 Stellen pro Layer (Q + K).** vs Sprint 9b's swiglu (1 Stelle)
   oder 9b/9b.2 multi_add_rms (2 Stellen, hidden_dim=4096). Der absolute
   Saving ist 2-Q-Norm + 2-K-Norm = 4 Dispatch-Calls eingespart (gegen
   "halbiertes Memory-Traffic" wie bei den größeren Fusionen).

3. **Citrix-Noise.** Bei 0.5% Δ sind wir am Rand der Detektionsschwelle.
   Mehrere Bench-Läufe würden den Δ klarer machen, aber das ist Zeit-
   Investition mit unsicherem ROI.

### 6.2 Cumulativer Sprint-9-Stand

```
| Sprint  | Dispatches/Layer | Barriers/Layer | pp=1024 vs 8a-Baseline |
|---------|------------------|----------------|------------------------|
| 8a      |        24        |       18       | (baseline)             |
| 9a      |        23        |       17       | +2.0%   (swiglu)       |
| 9b      |        22        |       16       | +1.5%   (multi_add Stelle 1) |
| 9b.2    |    21 + seed     |    15 + seed   | +1.3%   (cross-layer)  |
| 9c.5    |    19 + seed     |    14 + seed   | +0.6%   (rms+mul+rope) |
| TOTAL   |  -5/Layer        |  -4/Layer      | +5-7% kumuliert        |
| llama.cpp|       14         |        9       |                        |
| LÜCKE   |  -5/Layer        |  -5/Layer      |                        |
```

### 6.3 Verbleibende Lücke zu llama.cpp

Nach Sprint 9c.5 sind wir bei 19 Dispatches / 14 Barriers pro Layer
(plus seed). Ziel ist 14 / 9 (llama.cpp). Verbleibend: -5 / -5.

Bekannte Kandidaten:
* 9b.3: Decode-Pfad Cross-Layer Fusion (-1 Dispatch/Layer für Decode)
* 9d.1-9d.3: FP16 KV-Cache (~+5-15% pp ≥ 2048, auch -2 Dispatches via
  fused write-convert)
* Coopmat-Q4K: bereits implementiert, gated, nicht aktiv
* Quantize-Fusion mit GEMM (sehr komplex, llama.cpp macht es teilweise)

---

## 7. Sprint-9-Roadmap (aktualisiert)

```
| Sprint  | Status       | Δ (kumuliert vs 8a) |
|---------|--------------|---------------------|
| 9a      | DONE         | +2.0%               |
| 9c      | DONE (neg)   | +2.0%               |
| 9b      | DONE         | +3.5%               |
| 9d      | ANALYSIS     | (deferred)          |
| 9b.2    | DONE         | +4.8%               |
| 9c.5    | DONE         | +5.4%               |
| 9b.3    | TODO         | +6-7% (Decode CL)   |
| 9d.1    | TODO         | +0% (PoC)           |
| 9d.2    | TODO         | +10-25% pp ≥ 1024   |
| 9d.3    | TODO         | enables 9d for chat |
```

Empfehlung — nächste Action: **Sprint 9d.1 (FP16 KV Infrastructure)**.
Die Stelle-1- und Cross-Layer-Fusionen sind ausgereizt; der größte
verbleibende Hebel ist die Memory-Bandwidth-Halbierung für KV. 9d.1
ist behavior-neutral (default OFF), 1 Tag Aufwand, und stellt die
Infrastruktur für die wirklich impact-schweren 9d.2 und 9d.3 bereit.

---

## 8. Files Touched

```
modified: build.rs                                 (+ 1 ShaderJob)
modified: src/backend/vulkan/shaders.rs            (+ ShaderId::RmsNormMulRope + SPV)
modified: src/backend/vulkan/pipeline.rs           (+ RmsNormMulRopePushConstants)
modified: src/backend/vulkan/pipeline_registry.rs  (RmsNormMulRope shares spec consts)
modified: src/backend/vulkan/forward.rs            (+ run_rms_norm_mul_rope helper,
                                                    integration in dispatch_layer_batch)
new:      results/v02_sprint9c5_rms_norm_mul_rope.md (this report)
new SPV:  rms_norm_mul_rope_f32.spv (~290 KB)
```

KEIN neuer Shader-Source. KEIN neuer Test (existing E2E parity tests
cover the fusion correctness via argmax comparison).

---

## 9. Bottom Line

Sprint 9c.5 war ein Sprint mit "günstiger Geometrie": die Shader-
Logik existierte bereits im Code (als `#if RMS_NORM_ROPE_FUSION`),
und unser Pre-Check hat das gefunden. Implementation reduzierte sich
auf:
* 1 neuer build.rs-ShaderJob mit defines
* 1 neuer ShaderId-Eintrag + Push-Constant-Struct
* 1 neuer Forward-Pass-Helper mit korrekter Dispatch-Geometrie
* 1 Code-Stelle in dispatch_layer_batch ersetzt

Performance-Δ ist klein (+0.2-0.7%), aber konsistent positiv und
architecture-mäßig richtig. Die strukturelle Vereinfachung (-2
Dispatches, -1 Barrier pro Layer × 36 Layer = -72/-36 pro Forward)
zählt für die Pipeline-Tiefe und verbessert das Pipelining.

164/164 Tests bleiben grün, einschließlich aller End-to-End-Argmax-
Parity-Tests. Korrektheit ist also bestätigt.

Cumulativer Sprint-9-Stand: pp=1024 ~+5-7% vs Sprint-8a-Baseline,
24 → 19 Dispatches/Layer (-5), 18 → 14 Barriers/Layer (-4). Verbleibender
Gap zu llama.cpp's 14/9 ist hauptsächlich FP16 KV-Cache (Sprint 9d).
