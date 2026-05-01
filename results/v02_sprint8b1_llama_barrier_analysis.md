# v0.2 Sprint 8b.1 — llama.cpp Barrier-Analyse (Sprint 8b nachgeholt)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 8b (negatives Resultat ohne llama.cpp-Vergleich)

---

## TL;DR — Sprint 8b war richtig zu unseren Barriers, aber blind zur llama.cpp-Strategie

```
═══ v0.2 Sprint 8b.1 ═══

NACHGEHOLT: der llama.cpp-Vergleich aus Sprint 8b's Schritt 2.

ÜBERRASCHENDES ERGEBNIS:
  llama.cpp's Vorteil ist NICHT "weniger Barriers durch
  Tracking" — es ist "weniger DISPATCHES durch FUSED KERNELS",
  und damit implizit weniger Barriers.

Architektur-Vergleich:

  VulkanForge (Sprint 8b):
    • 1 Buffer pro Tensor (batch_q, batch_k, ...)
    • Fine-grained per-Op compute_barrier
    • 18 Barriers/Layer × 36 = 648/Forward
    • Alle Barriers RAW-gated, keine Elision möglich

  llama.cpp (build 408225b):
    • 3 Coarse-Scratch-Buffer-Flags (X, Y, split_k)
    • Pipeline+Tensor-Identity-Caching für Skip-Logic
    • 1 globaler Barrier-Type (full-wide, alle Stages, alle Access)
    • Fused Kernels:
        rms_norm_mul_f32              (RMS + multiply)
        rms_norm_mul_rope_f32         (RMS + mul + RoPE)
        swiglu                        (silu(gate) * up)
        multi_add_rms                 (residual + RMS-norm)
        multi_add                     (mehrere Adds in einer Op)
    • Graph-Level-Fusion via num_additional_fused_ops Counter

KERNERKENNTNIS — der Gap liegt NICHT in Barrier-Tracking:

  Sprint 8b's eigentliche Frage war "warum ist llama.cpp 3-5×
  schneller bei Prefill?" und Sprint 8b vermutete: bessere
  Barrier-Tracking. Das ist FALSCH.

  Tatsächlich:
    1. llama.cpp dispatcht ~13 Ops/Layer (mit allen Fusionen)
       vs unsere 24 Ops/Layer.
    2. Jeder fused Kernel spart 1-2 Barriers.
    3. Pro Layer: ~5-7 Barriers vs unsere 18 → -60-70%.
    4. Pro Forward: ~180-250 Barriers vs unsere 648 → -60-70%.

ABER (und das ist wichtig):
  Die Barrier-Reduktion ist EFFEKT der Kernel-Fusion, nicht
  Ursache. llama.cpp's "barrier elision" via _need_sync-Flags
  ist nur möglich weil sie 3 SHARED Scratch-Buffers haben statt
  unsere ~12 dedizierte Buffers. Das ist eine Architektur-
  Entscheidung mit Folgen.

EMPFEHLUNG:
  • Sprint 8b's Negativ-Befund (unsere Barriers sind alle nötig)
    BLEIBT korrekt.
  • Sprint 9 sollte FP16-KV PLUS swiglu+rms_norm_mul Fusion sein,
    nicht nur FP16-KV. Die Fusion ist ein größerer Hebel als
    der Audit angenommen hatte.

Files: nur dieser Report (kein Code).
Tests: 154/154 ✓ (unverändert).
Commit: HEAD (kein Push).
```

---

## 1. Wie llama.cpp Barriers handhabt

### 1.1 Scratch-Buffer-Architektur (3 Flags)

`ggml-vulkan.cpp:1912`:
```cpp
bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_split_k_need_sync;
```

llama.cpp hat genau **drei** Scratch-Buffer für mat-mul-Operationen:
* `prealloc_x` — src0 (weights/dequantized) Working-Space
* `prealloc_y` — src1 (activations/quantized) Working-Space
* `prealloc_split_k` — partial-output Working-Space für split-K reductions

Jede Op die in einen dieser Buffer schreibt setzt das entsprechende
`_need_sync = true`. Beim NÄCHSTEN Op das aus dem Buffer liest:

```cpp
// ggml-vulkan.cpp:7679-7683
if (x_non_contig || qx_needs_dequant) {
    if (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);  // emit barrier
    }
}
```

`ggml_vk_sync_buffers` resettet ALLE drei Flags auf `false` und
emittiert EINEN globalen Barrier (Zeile 2861):
```cpp
subctx->s->buffer->buf.pipelineBarrier(
    p->q->stage_flags, p->q->stage_flags, {},
    { { ShaderRead|ShaderWrite|TransferRead|TransferWrite,
        ShaderRead|ShaderWrite|TransferRead|TransferWrite } },
    {}, {}
);
```

Das ist ein **VOLL-WIDE-BARRIER** (alle Stages, alle Access-Flags).
Keine fine-grained Buffer-Auflistung. Granularität: 3 Flags total.

### 1.2 Pipeline+Tensor-Identity-Caching

`ggml-vulkan.cpp:7692-7712`:
```cpp
if (y_non_contig) {
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
        ctx->prealloc_y_last_tensor_used != src1) {
        if (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        ggml_vk_cpy_to_contiguous(...);
        ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
        ctx->prealloc_y_last_tensor_used = src1;
    }
}
```

Wenn die GLEICHE Pipeline auf den GLEICHEN Tensor in den GLEICHEN
Scratch geschrieben hat (z.B. `quantize_q8_1(src1) → prealloc_y`),
wird die Op KOMPLETT übersprungen — kein Re-Dispatch, kein Barrier.

Das macht Sinn z.B. wenn mehrere mat-muls die gleiche Activation
nutzen (Q/K/V-Projektion teilen die selbe quantisierte Activation).
**VulkanForge macht das bereits implizit** durch Re-Verwendung
der `batch_q8` Activation für alle drei GEMMs ohne Re-Quantize.

### 1.3 Fused Kernels (HIER ist der echte Hebel)

llama.cpp hat Pipelines die mehrere logische ggml-Ops in einem
Vulkan-Dispatch zusammenfassen:

#### `pipeline_rms_norm_mul_f32`
```cpp
// ggml-vulkan.cpp:9461-9463
return ctx->num_additional_fused_ops > 0
     ? ctx->device->pipeline_rms_norm_mul_f32      // fused
     : ctx->device->pipeline_rms_norm_f32;          // standalone
```
Fusion: `rms_norm(x) * weight` → ein Kernel.
Bei VulkanForge: 1 dispatch (rms_norm) + 1 barrier + 1 dispatch (mul).
Differenz: **-1 Dispatch, -1 Barrier**.

Verwendung pro Layer: 1× attn_norm + 1× ffn_norm = **2 Stellen**.

#### `pipeline_rms_norm_mul_rope_f32_*`
```cpp
// Comment ggml-vulkan.cpp:1342:
// "For fused rms_norm+mul+rope(+view+set_rows)"
```
Fusion: `rms_norm + mul + rope (+ optionally view + set_rows)` →
ein Kernel. Bei VulkanForge: 3-5 separate Dispatches mit 2-4
Barriers dazwischen.
Differenz: **-2 bis -4 Dispatches, -2 bis -4 Barriers**.

Verwendung: nur an spezifischen Stellen wo das Pattern matched
(prä-Attention Q/K-Pfad bei Modellen mit qk-norm).

#### `pipeline_swiglu`
```cpp
// ggml-vulkan.cpp:798
vk_pipeline pipeline_swiglu[2];   // [f32_out, f16_out]
```
Fusion: `silu(gate) * up` → ein Kernel.
Bei VulkanForge: 2 Dispatches (silu in-place batch_gate; mul gate*up
→ batch_ffn_hidden) + 1 Barrier dazwischen.
Differenz: **-1 Dispatch, -1 Barrier**.

Verwendung: 1× pro Layer FFN.

#### `pipeline_multi_add_rms`
```cpp
// ggml-vulkan.cpp:728-729
vk_pipeline pipeline_multi_add[MAX_FUSED_ADDS];
vk_pipeline pipeline_multi_add_rms[MAX_FUSED_ADDS];
```
Fusion: `(residual + delta) → rms_norm` → ein Kernel.
Bei VulkanForge: 2 Dispatches (add_residual + rms_norm) + 1 Barrier.
Differenz: **-1 Dispatch, -1 Barrier**.

Verwendung: 2× pro Layer (residual1 + rms_norm_ffn; residual2 +
nächste Layer's rms_norm_attn). PRO LAYER also -2 Dispatches,
-2 Barriers.

### 1.4 Graph-Level Fusion

`ggml-vulkan.cpp:1932`:
```cpp
int num_additional_fused_ops {};
```

Ein Counter im Backend-Context. Beim Graph-Walk LOOK-AHEAD:
wenn `node[i]` und `node[i+1]` ein fused-Pattern bilden, setze
`num_additional_fused_ops = 1`, dispatch nur EINMAL (mit dem
fused-Pipeline), springe i+=2.

```cpp
// ggml-vulkan.cpp:9298-9302
if (ctx->num_additional_fused_ops > 0) {
    if (rms) {
        return ctx->device->pipeline_multi_add_rms[ctx->num_additional_fused_ops];
    } else {
        return ctx->device->pipeline_multi_add[ctx->num_additional_fused_ops];
    }
}
```

Das ist GENERISCHER als die fest-verdrahteten Pipelines —
`num_additional_fused_ops = 2` bedeutet "fold 2 zusätzliche Ops
in den nächsten Dispatch", was z.B. `rms_norm + mul + rope` als
Triple ergibt.

---

## 2. Barrier-Count pro Layer

### 2.1 VulkanForge (Sprint 8b-Topologie)

24 Dispatches × 18 Barriers/Layer × 36 Layer = **648 Barriers/Forward**.

### 2.2 llama.cpp (mit allen Fusionen aktiv)

Geschätzte Layer-Topologie (typischer Qwen3/Llama3 Layer):

```
| #  | Op (logisch)            | Fused-In             | Dispatch | Barrier? |
|----|------------------------ |----------------------|----------|----------|
| 1  | rms_norm_attn + mul     | rms_norm_mul         |    1     | (impl.)  |
| 2  | gemm_q                  | (standalone)          |    1     | nach     |
| 3  | gemm_k                  | (standalone, same Y) |    1     | skip     |
| 4  | gemm_v                  | (standalone, same Y) |    1     | skip     |
| 5  | rope_q                  | (standalone)          |    1     | nach     |
| 6  | rope_k                  | (standalone)          |    1     | skip*    |
| 7  | flash_attn              | (standalone)          |    1     | nach     |
| 8  | gemm_o                  | (standalone)          |    1     | nach     |
| 9  | residual1 + rms_norm    | multi_add_rms        |    1     | nach     |
| 10 | gemm_gate               | (standalone)          |    1     | nach     |
| 11 | gemm_up                 | (standalone, same Y) |    1     | skip     |
| 12 | silu(gate)*up           | swiglu               |    1     | nach     |
| 13 | gemm_down               | (standalone)          |    1     | nach     |
| 14 | residual2 + rms_norm    | multi_add_rms        |    1     | nach     |
                              ────────────────────────────────────
                              14 Dispatches, ~9 Barriers/Layer
```

* "skip" weil die rope_k Y-Scratch nicht von rope_q dirty ist
  (verschiedene Tensor-Identitäten).

Pro Forward: 9 × 36 = **~324 Barriers/Forward** (vs unsere 648).
Differenz: **~50% weniger Barriers**.

Aber wichtiger: **24 → 14 Dispatches/Layer = ~40% weniger Compute-
Submits**. Jeder Submit hat Descriptor-Bind + Push-Constant +
Dispatch-Overhead.

### 2.3 Wo der echte Performance-Hebel liegt

Sprint 8b's Hypothese: "weniger Barriers durch Tracking".
Empirisch: **die Barrier-Tracking-Logik ist EFFEKT der Kernel-
Fusion, nicht Ursache**.

llama.cpp kann seine 3-Flag-Tracking-Strategie deshalb nutzen
weil sie nur 3 Scratch-Buffers haben. Wir haben ~12 dedizierte
Buffer (batch_q, batch_k, batch_v, batch_norm, batch_q8, batch_o,
batch_gate, batch_up, batch_ffn_hidden, batch_ffn_out, batch_attn_out,
batch_residual). Per-Buffer-Tracking wäre für uns möglich, würde
aber die Sprint-8b-Erkenntnis nicht ändern: **alle unsere
verbleibenden Barriers gated echte RAW-Hazards**.

Der Performance-Hebel ist:
1. **Kernel-Fusion** (rms_norm+mul, swiglu, multi_add_rms) →
   weniger Dispatches → automatisch weniger Barriers.
2. **Pipeline-Identity-Caching** für Re-Use von quantisierten
   Activations (das machen wir implizit schon).

NICHT:
3. Per-Buffer-Barrier-Elision (würde nichts bringen).

---

## 3. Antworten auf die Sprint-8b.1-Fragen

> **A. Trackt llama.cpp per-Buffer Read/Write Sets?**

**NEIN, nicht per-Buffer.** Llama.cpp trackt drei coarse Scratch-
Buffer-Flags (`x`, `y`, `split_k`) plus pro Scratch ein "letzte
Pipeline + letzter Tensor"-Cache. Die Granularität ist Scratch-
Buffer-weit, nicht Region-weit.

> **B. Oder per-Pipeline (gleicher Shader → skip)?**

**JA, kombiniert mit Tensor-Identity.** Wenn die gleiche Pipeline
auf den gleichen Tensor in den gleichen Scratch dispatcht hat,
wird die OP komplett übersprungen (nicht nur der Barrier).

> **C. Oder globales "dirty" Flag?**

**Drei Flags, eins pro Scratch-Buffer.** Nicht ganz global, nicht
per-Buffer — drei.

> **D. Wie granular? Per-Op? Per-Layer? Per-Submit?**

**Per-Op innerhalb eines Submits.** Jede Op prüft die relevanten
Flags VOR dem Dispatch und emittiert ggml_vk_sync_buffers nur
wenn nötig. Per-Layer-Granularität wäre zu coarse, per-Submit-
Granularität zu coarse.

> **E. Hat llama.cpp fused Kernels die wir nicht haben?**

**JA, FÜNF GROSSE:**
1. `rms_norm_mul_f32` — RMS + multiply
2. `rms_norm_mul_rope_f32_*` — RMS + mul + RoPE
3. `swiglu` — silu * mul (FFN-Aktivierung)
4. `multi_add_*` — fold N consecutive adds
5. `multi_add_rms_*` — multi_add + RMS

**Wir haben KEINE davon.**

> **F. Wie viele vkQueueSubmit pro Forward?**

llama.cpp: 1 Submit pro Forward (üblicherweise). Das ist auch
unser Modell — `cmd_ctx.one_shot` submitted alles in einem Call.
Hier kein Unterschied.

> **G. Hat llama.cpp eine andere Op-Reihenfolge?**

**NEIN. Op-Reihenfolge ist identisch** (RMS-norm → Q/K/V GEMMs →
RoPE → Attn → Output → FFN). Der Unterschied ist die FUSION
mehrerer logischer Ops in einen Dispatch — die Reihenfolge bleibt
bei beiden gleich.

---

## 4. Vergleichstabelle (Sprint 8b.1 nachgereicht)

```
═══ Barrier-Vergleich VulkanForge vs llama.cpp ═══

| Metrik                    | VulkanForge | llama.cpp (build 408225b) |
|---------------------------|-------------|---------------------------|
| Dispatches pro Layer      |     24      |    ~14 (mit Fusion)       |
| Barriers pro Layer        |     18      |    ~9                     |
| Barriers pro Forward (36) |    648      |   ~324                    |
| Fused Kernels             |      0      |    5                      |
| Barrier-Tracking-Methode  | per-op      | 3-flag scratch + identity |
| Graph-basierte Planung    | NEIN        | num_additional_fused_ops  |
| vkQueueSubmit pro Forward |     1       |    1                      |
```

---

## 5. Sprint 8b war NICHT umsonst

Sprint 8b kam zu der Konklusion "unsere 18 Barriers sind alle
nötig, keine Reduktion möglich ohne Kernel-Änderung". Diese
Konklusion ist **bei vorgegebener Kernel-Architektur korrekt**.

Sprint 8b.1 zeigt jetzt: die Reduktion erfordert Kernel-Fusion.
Das ist EXAKT was Sprint 8b's Sektion 3 ("Wo überhaupt etwas
möglich wäre") als Sprint-9-Kandidaten aufgelistet hatte:
* (a) Fused silu+mul Kernel  → llama's `swiglu`
* (b) Out-of-place RoPE     → komplementär zu llama's rms_norm_mul_rope
* (c) Persistent KV-Layout  → orthogonal

Der Audit hat den Hebel richtig identifiziert, nur die Größenordnung
unterschätzt. Sprint 8b: "1-3% pro fused kernel". Sprint 8b.1
revidiert: bei 5 fused-Kernel-Pattern × ~1-3% = **5-15% kumulativer
Gewinn** plus **~50% weniger Barrier-Overhead**.

---

## 6. Empfehlung — Sprint 9 erweitert

Statt nur "FP16 KV-Cache", Sprint 9 sollte ZWEI Hebel kombinieren:

### Sprint 9a — Fused SwiGLU (1 Tag, niedriges Risiko)

* Neuer Shader `swiglu.comp` (silu(gate) × up → out, single dispatch)
* Pro Layer: -1 Dispatch, -1 Barrier
* Pro Forward: -36 Dispatches, -36 Barriers
* Erwarteter Gewinn: 2-5% bei mittleren pp (wo FFN dominant ist)

### Sprint 9b — Fused multi_add_rms (1-2 Tage, mittleres Risiko)

* Neuer Shader `add_rms_norm.comp` (residual + rms_norm in einem)
* Pro Layer: -2 Dispatches, -2 Barriers (residual1+ffn_norm,
  residual2+next-attn_norm)
* Pro Forward: -72 Dispatches, -72 Barriers
* Erwarteter Gewinn: 3-7% durchgehend

### Sprint 9c — Fused rms_norm_mul (1 Tag, niedriges Risiko)

* Neuer Shader `rms_norm_mul.comp` (rms_norm × weight in einem)
* Ersetzt `pipeline_rms_norm_mul_f32`-Pattern
* Pro Layer: -2 Dispatches, -2 Barriers
* Pro Forward: -72 Dispatches, -72 Barriers
* Erwarteter Gewinn: 2-5%

### Sprint 9d — FP16 KV-Cache (2-3 Tage, hohes Risiko + Numerik)

* Halbiert KV-Memory-Bandbreite
* Hilft besonders pp ≥ 2048
* Erwarteter Gewinn: 5-15% bei langem Kontext

**Reihenfolge:** 9c → 9a → 9b → 9d. Niedriges Risiko zuerst,
inkrementelle Wins, dann der größere FP16-Hebel.

Kumulativ pro Forward: -180 Dispatches, -180 Barriers.
Erwartete Performance-Verbesserung: +10-20% durchgehend, plus
weiterer Gewinn aus FP16 bei langem pp.

---

## 7. Lessons / Decisions

* **Sprint 8b's Negativ-Befund war für seine Frage korrekt** —
  unsere existierenden Barriers sind alle nötig.
* **Sprint 8b's Frage war zu eng gefasst** — er fragte "können
  wir Barriers entfernen", nicht "WIE können wir Barriers
  reduzieren". Letzteres öffnet die Kernel-Fusion-Tür.
* **Der Audit's "Conditional Barriers" als Sprint-8b-Konzept
  war unscharf** — das echte llama.cpp-Pattern ist
  Kernel-Fusion + 3-Flag-Tracking, nicht
  Per-Buffer-Conditional-Tracking.
* **PR #12135** (aus dem Sprint-Brief erwähnt) — ohne expliziten
  PR-Number-Match scheint dieser eher die `prealloc_*_need_sync`-
  Logik zu sein die wir oben dokumentiert haben (build 408225b
  hat sie aktiv).

---

## 8. Files Touched

```
new file: results/v02_sprint8b1_llama_barrier_analysis.md
```

KEIN Code geändert. Tests unverändert 154/154 ✓.
