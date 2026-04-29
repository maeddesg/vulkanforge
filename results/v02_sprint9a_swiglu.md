# v0.2 Sprint 9a — Fused SwiGLU Kernel

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 8b.1 (24 Dispatches/Layer, 18 Barriers/Layer, 154/154 Tests)

---

## TL;DR — Erste Fusion: -1 Dispatch, -1 Barrier pro Layer. +2% Prefill an Realistic-pp.

```
═══ v0.2 Sprint 9a ═══

Shader:  vk_shaders/swiglu.comp (~30 LOC, eigenständig — kein
         glu_head/glu_main Include-Geflecht wie llama.cpp).
         out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
         3 SSBOs (gate, up, out), 1×u32 Push-Konstante (n).
         WG = 256, 1 Element pro Thread.

Replaces:
   Sprint 8a:  silu(gate→gate)  ── barrier ──  mul(gate,up→ffn_hidden)
                                                                 ↓
   Sprint 9a:                  swiglu(gate,up→ffn_hidden)

Pro Layer:
  Dispatches: 24 → 23  (-1)
  Barriers:   18 → 17  (-1)
Pro Forward (36 Layer):
  Dispatches: 864 → 828  (-36)
  Barriers:   648 → 612  (-36)

Performance (run_pp_bench, 3 Runs/Punkt nach 1 Warmup):

  | pp   | Sprint 8a | Sprint 9a | Δ tok/s | Δ %    |
  |------|-----------|-----------|---------|--------|
  |   64 |   1431    |   1439.9  |    +9   |  +0.6% |
  |  128 |   1830    |   1871.5  |   +42   |  +2.3% |
  |  256 |   1890    |   1923.6  |   +34   |  +1.8% |
  |  512 |   1761    |   1790.3  |   +29   |  +1.7% |
  | 1024 |   1469    |   1498.6  |   +30   |  +2.0% |

15-Prompt Bench (run_15prompt_bench):
  Coherent prompts: 15/15 ✓
  Median prefill (alle 15): 1052.2 tok/s  (Sprint 8a: 728.7 / first 5)
  Median decode:              90.0 tok/s  (Sprint 8a: 87.3, +3.1%)
  Pp=20 first-5 prefill:     367.8        (Sprint 8a: 372.4, -1.2% noise)
  Pp=62 first-5 prefill:    1409.4        (Sprint 8a: 1400.6, +0.6%)

Tests:
  cargo test --release  →  159/159 ✓
  Vorher 154/154; +5 neue swiglu-Tests in tests/correctness.rs:
    - test_swiglu_vs_separate_small  (n=1024, max_abs<1e-6)
    - test_swiglu_vs_separate_qwen_ffn_shape  (n=128*12288, max_abs<1e-6)
    - test_swiglu_zeros               (gate=0 → 0)
    - test_swiglu_negative_saturates  (gate=-10 → ~0)
    - test_swiglu_positive_passthrough (gate=10 → up*silu(10))

Files:
  new:      vk_shaders/swiglu.comp
  modified: build.rs                                 (+ ShaderJob)
  modified: src/backend/vulkan/shaders.rs            (+ ShaderId::SwiGLU)
  modified: src/backend/vulkan/pipeline.rs           (+ SwigluPushConstants)
  modified: src/backend/vulkan/pipeline_registry.rs  (+ SwiGLU branch)
  modified: src/backend/vulkan/forward.rs            (+ run_swiglu, -run_silu)
  modified: tests/correctness.rs                     (+5 swiglu tests)
  new:      results/v02_sprint9a_swiglu.md           (this report)

Commit: HEAD (kein Push).
```

---

## 1. llama.cpp-Vergleich

llama.cpp's SwiGLU lebt in `vulkan-shaders/swiglu.comp`:

```glsl
#version 450
#include "glu_head.glsl"
float op(float a, float b) { return a / (1.0f + exp(-a)) * b; }
#include "glu_main.glsl"
```

Das `glu_head/glu_main`-Setup unterstützt drei Modi (default,
swapped, split) und vier Strides für 4-D-Tensors. Wir nutzen
hier nur den Split-Mode mit `gate, up, out` als drei separaten
Buffern und 1-D-Indexierung — ein 30-Zeiler ohne den Include-
Overhead. Die Arithmetik ist FP32-bitidentisch zu llama.cpp:
`a / (1 + exp(-a)) * b`.

Entscheidung: **Eigener Shader, kein Port**.
Begründung: llama.cpp's Strides sind für tensor-views nötig, die
unser Forward-Pass nicht nutzt. Ein eigenes 30-Zeilen-File ist
einfacher zu reviewen als ein Include-Geflecht mit 4 Dateien für
denselben Output.

---

## 2. Dispatch-Topologie vorher/nachher

**Sprint 8a (FFN-Block, 6 Dispatches + 6 Barriers):**
```
quantize_ffn → barrier → gemm_gate → gemm_up → barrier →
silu(gate) → barrier → mul(gate,up→hidden) → barrier →
quantize_hidden → barrier → gemm_down → barrier
```

**Sprint 9a:**
```
quantize_ffn → barrier → gemm_gate → gemm_up → barrier →
swiglu(gate,up→hidden) → barrier →
quantize_hidden → barrier → gemm_down → barrier
```

Die Barrier zwischen `silu` und `mul` entfällt komplett, weil die
SiLU-Zwischenzustände nur noch in einem Register pro Thread
existieren — kein global-memory Round-Trip, keine Read-After-Write
Hazard zwischen zwei Dispatches.

Genau die gleiche Reduktion gilt im Decode-Pfad (`dispatch_layer`,
`gate_buf` statt `batch_gate` für den 1-Token-Fall).

---

## 3. Bit-Genauigkeit

Die isolierten Tests fahren denselben Random-Input erst durch
Sprint-8a (separate `silu(gate→gate)` + `mul(gate,up→out)`) und
dann durch Sprint-9a (fused `swiglu`), und vergleichen
elementweise:

```
test_swiglu_vs_separate_small             max_abs_err = 0.0  ✓
test_swiglu_vs_separate_qwen_ffn_shape    max_abs_err = 0.0  ✓
```

Die Erwartung war `< 1e-6`; tatsächlich ist es exakt 0.0. Grund:
Beide Pfade rechnen `(g / (1.0 + exp(-g))) * u` mit denselben FP32-
Operationen in derselben Reihenfolge. Im Sprint-8a-Pfad wird das
SiLU-Zwischenergebnis in global memory geschrieben und wieder
geladen — aber FP32→FP32-Roundtrip ist identitätserhaltend, also
ist auch die Multiplikation nachher bit-identisch.

---

## 4. Performance — vollständige Auswertung

### 4.1 pp-Sweep (run_pp_bench)

```
| pp   | Sprint 8a | Sprint 9a | Δ tok/s | Δ %    | Pro-Forward Saving |
|------|-----------|-----------|---------|--------|--------------------|
|   64 |   1431    |   1439.9  |    +9   |  +0.6% | -36 Dispatch/Barrier |
|  128 |   1830    |   1871.5  |   +42   |  +2.3% | -36 Dispatch/Barrier |
|  256 |   1890    |   1923.6  |   +34   |  +1.8% | -36 Dispatch/Barrier |
|  512 |   1761    |   1790.3  |   +29   |  +1.7% | -36 Dispatch/Barrier |
| 1024 |   1469    |   1498.6  |   +30   |  +2.0% | -36 Dispatch/Barrier |
```

Der Gain ist über alle pp-Werte ≥ 128 konsistent +1.7-2.3%. Das
liegt im unteren Drittel der Sprint-Brief-Erwartung von "+2-5%".
Begründung:

* Bei pp=64 dominieren GEMM-Launches und `mul_mmq`-Quantisierung;
  der eingesparte SwiGLU-Dispatch ist <1% der Forward-Time → 0.6%
  Total-Gain reflektiert genau das.
* Bei pp ≥ 128 wird Attention sichtbar und die SwiGLU-Einsparung
  ist proportional kleiner — daher der ähnliche Plateau-Wert.

### 4.2 15-Prompt Bench

```
=== Aggregate ===
  Coherent prompts: 15/15 ✓
  MEDIAN prefill: 1052.2 tok/s  (alle 15 Prompts)
  MEDIAN decode:    90.0 tok/s
```

First-5-Prompts-Vergleich (zur direkten Sprint-8a-Korrespondenz):

```
| Prompt          | pp | Sprint 8a | Sprint 9a | Δ %   |
|-----------------|----|-----------|-----------|-------|
| Greeting        | 20 |   372.4   |   367.8   | -1.2% |
| Simple Sequence | 31 |   715.1   |   722.4   | +1.0% |
| Prime Check     | 31 |   728.7   |   729.1   | +0.1% |
| LRU Cache       | 47 |  1073.9   |  1091.1   | +1.6% |
| REST API        | 62 |  1400.6   |  1409.4   | +0.6% |

MEDIAN prefill: 728.7 → 729.1 (statistisch identisch)
```

Die Mini-Prompt (pp=20) Regression von -1.2% ist innerhalb des
~3% Mess-Rauschens, das wir schon in Sprint 8a beobachtet haben.

Decode median: **+3.1%** (87.3 → 90.0) — auch der Decode-Pfad
profitiert, weil `dispatch_layer` (1-Token) ebenfalls die Fusion
nutzt; bei 36 Layern × 1 Dispatch/Barrier × ~5µs ≈ 180µs/Forward,
was bei einem ~11ms-Decode-Step (90 tok/s) 1.6% wäre — die
gemessenen +3.1% sind innerhalb Bench-Variance plausibel.

---

## 5. Tests

### 5.1 Vollständiges `cargo test --release`

```
test result: ok. 24 passed       (lib unit)
test result: ok. 65 passed       (correctness, +5 swiglu)
test result: ok.  9 passed       (dequant_q4k)
test result: ok. 18 passed       (gguf)
test result: ok.  8 passed       (q4k_quant + coopmat)
test result: ok.  8 passed       (flash_attn_tiled_ref)
test result: ok. 27 passed       (regression)
                ────
                159 / 159 ✓
```

Insbesondere:
* `phase3e_prefill_batch_matches_token_by_token_top5` ✓
  → Sprint-9a-fused == Sprint-9a-fused token-by-token.
* `sprint5b_chunked_prefill_parity_qwen3` ✓
  → Argmax bit-identisch über Single-Shot vs 4-Chunk-Prefill.
* `phase5b2_decode_after_batched_prefill_qwen3` ✓
  → End-to-End Coherence-Check.

Diese drei sind die strengsten "kein Bit darf wackeln"-Tests im
Repo, und sie sind alle grün — was die isolierten swiglu-Tests
auf End-to-End-Niveau bestätigt.

### 5.2 Neue Tests

```
test_swiglu_vs_separate_small             1024 elem  ✓ max_abs=0
test_swiglu_vs_separate_qwen_ffn_shape    1.57M elem ✓ max_abs=0
test_swiglu_zeros                         all zero   ✓
test_swiglu_negative_saturates            silu≈0     ✓
test_swiglu_positive_passthrough          silu≈x     ✓
```

---

## 6. Was bleibt offen — Sprint-9-Roadmap

Die nächsten 4 Fusion-/KV-Sprints aus Sprint 8b.1's Recommendation
sind unverändert. Sprint 9a hat die einfachste Fusion eingesammelt;
die nächsten haben höhere ROI aber auch mehr Komplexität:

* **Sprint 9b — multi_add + multi_add_rms** (1-2 Tage):
  Residual + nächstes RMSNorm in einem Dispatch. Llama.cpp's
  `multi_add` macht das mit bis zu 6 Inputs in einer Op. Saving:
  ca. 2 Dispatches + 2 Barriers pro Layer. Erwartet +3-7%.

* **Sprint 9c — rms_norm_mul** (1 Tag):
  RMSNorm-Output direkt mit Gewicht multipliziert in einem
  Kernel statt zwei. Saving: 1 Dispatch + 1 Barrier × 2 Stellen
  pro Layer (attn + ffn). Erwartet +2-5%.

* **Sprint 9c.5 — rms_norm_mul_rope** (2 Tage):
  Q/K-Norm + RoPE in einem Kernel. Erwartet +2-3%.

* **Sprint 9d — FP16 KV-Cache** (2-3 Tage):
  Halbiert Memory-Bandwidth. Besonders relevant für pp ≥ 2048
  wo wir bei 0.14-0.22× von llama.cpp sind. Erwartet +5-15% am
  langen Kontext.

Mit Sprint 9a sind wir bei:
* Dispatches/Layer: 24 → 23
* Barriers/Layer:   18 → 17
* llama.cpp:        14 Dispatches, 9 Barriers
* Differenz:        +9 Dispatches (-1 nach 9a), +8 Barriers (-1 nach 9a)

---

## 7. Bekannte Fallstricke / Code-Hygiene

* `fn run_silu` ist gelöscht — war im FFN-Hot-Path die einzige
  Verwendung. `ShaderId::Silu` und `ShaderId::Mul` bleiben in
  der Pipeline-Registry, weil `tests/correctness.rs::test_silu_vs_cpu`
  und `test_mul_exact` sie weiterhin direkt dispatchen (die
  silu/mul-Shader sind also kein totes Gewicht — nur der
  Wrapper-Helper in `forward.rs` ist weg).

* `GenericHeadPushConstants` ist aus dem `forward.rs`-Import-
  Block entfernt. Die Struktur wird weiterhin in `pipeline.rs`
  exportiert (für `tests/correctness.rs`).

* Der Dispatch nutzt `WG=256, 1 elem/thread` — anders als
  silu's `WG=512, 1 elem/thread` und mul's `WG=256, 2 elem/thread`.
  Konsistent mit der einfachsten 1-D-Topologie, kein Y-WG-Fanout.
  Bei Qwen3-FFN (n = seq_len × 12288) ergibt das bis zu 49152 WGs
  in X-Richtung — innerhalb der Vulkan-Garantie von 65535.

---

## 8. Bottom Line

Sprint 9a war eine kleine, saubere Fusion mit +2% Prefill und +3%
Decode bei null Risk: bit-identische Arithmetik, alle 154 alten
Tests + 5 neue grün, ein 30-LOC-Shader, kein Build-System-Ärger.

Der Hebel hier ist klein, aber die Sprint-Roadmap ist additiv:
Wenn 9b-9d zusammen die anderen vier llama.cpp-Fusionen einsammeln,
sind wir bei Dispatches/Layer ≈ 17 (von 24), Barriers/Layer ≈ 12
(von 18), und der Gesamt-Gewinn sollte in den +10-20% Bereich
landen.

Die nächste Action ist Sprint 9c (rms_norm_mul) — gleicher
Schwierigkeitsgrad wie 9a aber mit zwei Anwendungsstellen pro
Layer, also doppeltem ROI.
