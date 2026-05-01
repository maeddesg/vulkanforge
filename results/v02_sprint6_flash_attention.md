# v0.2 Sprint 6 — Flash-Attention-Prefill (Analyse + Negativer Optimierungs-Versuch)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 5B (Chunked Prefill, pp=4096 erreichbar, 146/146 Tests)
**Ziel:** O(N²) Attention auf O(N)-Memory mappen → pp-Skalierung
fixen.

---

## TL;DR — Wir HABEN bereits Flash-Attention. Der Gap liegt anderswo.

```
═══ v0.2 Sprint 6 ═══

ANALYSE-ERGEBNIS (vor Code-Änderung):
  flash_attn_batch.comp ist bereits ECHTES Flash-Attention:
    • Streaming/online-softmax mit running-max + running-sum
    • Tile-Loop über K/V (TILE=64 Wave64)
    • LDS-Verbrauch konstant (256 B scores_lds), NICHT O(N²)
    • Keine voll-materialisierte QK^T-Matrix in Global Memory
    • Causal-Mask per-Tile via causal_len = q_start + q_idx + 1

  → Die O(N²) COMPUTE ist fundamental für Attention (nicht
    behebbar mit Flash-Attention oder anderem Algorithmus).
    Was Flash-Attention fixt ist O(N) MEMORY — das haben wir.
  → Die Sprint-5B-pp-Decay-Kurve ist genau das was Flash-Attention
    LIEFERT: tok/s ∝ 1/N im langen pp-Limit.
  → llama.cpp's flachere Kurve liegt NICHT an einem fehlenden
    Algorithmus, sondern an konstant-faktor-Optimierungen die wir
    nicht haben:
      (a) Br > 1: Q-Tile mit mehreren Queries pro Workgroup,
          K/V-Tile via LDS gemeinsam genutzt
      (b) f16 K/V: halbe Memory-Bandbreite
      (c) coopmat-Pfad (cm1/cm2): WMMA-beschleunigte QK + PV
      (d) Br/Bc/D_split spec-constants gegen-pipeline getuned

OPTIMIERUNGS-VERSUCH (Code-Änderung):
  K-Tile in LDS staging — strided global K-Reads ersetzen durch
  einmalige LDS-Befüllung pro Tile. Algorithmus identisch.

ERGEBNIS — NEGATIV (durchgehend):
  | pp   | Sprint 5B  | Sprint 6 (K-LDS) | Δ      |
  |------|------------|------------------|--------|
  |  64  |    1489    |     1248         | -16%   |
  |  128 |    1641    |     1283         | -22%   |
  |  256 |    1337    |     1042         | -22%   |
  |  512 |     921    |      733         | -20%   |

  Begründung: RDNA4's L1-Cache absorbiert die strided K-Reads
  schon (jeder Thread liest 128 sequentielle K-Floats — das wird
  vom Hardware-Prefetcher erkannt). Die LDS-Staging-Schicht
  verdoppelt das Memory-Trafic ohne den Bottleneck zu bewegen.

REVERTET. Shader unverändert vs Sprint 5B.

WAS BLEIBT:
  • flash_attn_batch.comp unverändert
  • 146/146 Tests grün (Revert ist clean)
  • Dokumentation des wahren Bottlenecks (Br=1, fp32, kein coopmat)

EHRLICHE EMPFEHLUNG SPRINT 7 (oder neuer Sprint-6):
  Echter Br>1 Tiled-Q-Flash-Attention-Kernel. Das ist eine
  ~300-LOC GLSL-Neuschreibung mit nicht-trivialer Korrektheits-
  Risiko (Online-Softmax across Q-Tile, GQA-Indexing, Causal-Mask
  innerhalb Q-Tile). Verdient eigenes mehrtägiges Projekt mit
  vorab-Korrektheitspfad (parity-test gegen scalar attention bei
  jeder Tile-Größe), nicht einen einzelnen Sprint.
```

---

## 1. Analyse — Ist `flash_attn_batch.comp` echtes Flash-Attention?

### 1.1 Die Antwort: **Ja, bereits.**

Quelle: `vk_shaders/flash_attn_batch.comp` (157 LOC).

**Streaming-Softmax über K-Tiles** (Zeilen 98-145):
```glsl
for (uint tile_base = 0; tile_base < causal_len; tile_base += TILE) {
    // ... QK score per thread for this K-tile
    float tile_max = subgroupMax(score);
    float new_max  = max(my_max, tile_max);
    float kfac     = exp(my_max - new_max);
    my_out0 *= kfac;            // running-O rescale
    my_sum  *= kfac;            // running-sum rescale
    my_max   = new_max;
    // ... pscore = exp(score - new_max)
    // ... weighted V-sum into my_out0/my_out1
}
float inv_sum = 1.0 / my_sum;
o[o_off + tid] = my_out0 * inv_sum;  // final normalize
```

Das ist textbook Dao-et-al.-Flash-Attention: running max, running
sum, scale-correct accumulation. Kein "Materialize-then-Softmax-
then-V"-Pattern.

**LDS-Verbrauch ist konstant**:
```glsl
const uint TILE = WGSIZE;       // 64
shared float scores_lds[TILE];  // 256 B
```

Kein LDS-Buffer der mit `seq_len` oder `causal_len` skaliert. Kein
`shared float scores[seq_len][kv_len]` o.ä. — das wäre die
Anti-Pattern.

**Keine Global-Memory-Materialisierung der QK-Matrix**:
* QK-Scores entstehen pro Wave-Iteration, leben in VGPRs/LDS.
* Werden direkt in `pscore = exp(...)` umgewandelt, dann sofort
  konsumiert für V-Akkumulation, dann verworfen.
* Es gibt keinen Buffer der `[seq, kv_len]`-große Scores hält.

### 1.2 Warum dann der Decay?

Die pp-Kurve (Sprint 5B):
```
pp=64:    1489 tok/s
pp=128:   1641 tok/s   ← Peak
pp=512:    921 tok/s   (-44%)
pp=1024:   556 tok/s   (-66%)
pp=4096:   303 tok/s   (-82%)
```

Das ist **nicht** das Symptom eines fehlenden Flash-Attention.
Das ist das **erwartete** Verhalten von Flash-Attention.

**Mathematik**: Attention-Compute ist O(N²) (jeder Query gegen jede
Key). Über N Tokens prefilled in O(N²) Wall-Time → tok/s = N/N² =
1/N. Lineare Abnahme im Throughput.

**Plus** GEMM-Anteil der konstanter ist (skaliert mit O(N) compute
durch O(N) Tokens = O(1) pro Token):
```
total_time(N) = T_g · N + T_a · N²
tok/s(N)      = N / total_time(N)
              = 1 / (T_g + T_a · N)
```

Für unsere Messungen:
* pp=128: 1641 tok/s → total 78 ms
* pp=1024: 556 tok/s → total 1842 ms
* Setze Modell an:
  * 128: T_g · 128 + T_a · 128² = 78 ms
  * 1024: T_g · 1024 + T_a · 1024² = 1842 ms

Solving: T_g ≈ 0.44 ms/token, T_a ≈ 0.0014 ms/token² (≈ 1.4 µs/token²).

Bei pp=4096 vorhergesagt: 0.44·4096 + 0.0014·4096² = 1802 + 23488
= 25290 ms vs gemessen 13511 ms. Das Modell überschätzt — d.h.
unsere Attention wächst tatsächlich SUB-quadratisch. Vermutlich
weil bei sehr großen Tiles GPU-Caches besser greifen, ODER weil
GEMM bei großem pp doch schneller wird (besser ausgelastet).

In jedem Fall: die Kurve ist **konsistent mit O(N²) Compute**.

### 1.3 Wo llama.cpp wirklich gewinnt

Quelle: `~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`
* `flash_attn_base.glsl` (367 LOC)
* `flash_attn.comp` (788 LOC)
* `flash_attn_cm1.comp` (642 LOC, coopmat-1)
* `flash_attn_cm2.comp` (390 LOC, coopmat-2 mit KHR_coopmat)

**Kritische Spec-Constants** (`flash_attn_base.glsl:1-7`):
```glsl
layout (constant_id = 1) const uint32_t Br = 1;        // Q-Tile rows
layout (constant_id = 2) const uint32_t Bc = 32;       // K/V-Tile cols
layout (constant_id = 6) const uint32_t D_split = 16;  // d-dim parallelism
layout (constant_id = 7) const uint32_t row_split = 1;
```

Drei Differenz-Achsen zu uns:

(a) **Br > 1 möglich**: llama.cpp kann mehrere Queries pro
    Workgroup verarbeiten, sodass eine geladene K-Tile von
    mehreren Q-Rows wiederverwendet wird. Das ist die
    klassische 2-D Flash-Attention-Tiling. Wir haben Br=1
    (eine Query pro WG).

(b) **f16 K/V** (`#extension GL_EXT_shader_explicit_arithmetic_types_float16`):
    Halber Memory-Footprint, halbe Bandbreite. Wir nutzen FP32 K/V
    in der KV-Cache.

(c) **coopmat-Pfade**: cm1 (basic_coopmat) und cm2 (KHR_coopmat
    mit BF16-Support). Hardware-WMMA für QK und PV. Wir machen
    das alles in Skalar-FP32-FMAs.

llama.cpp wählt zur Laufzeit (Build-Banner zeigt `matrix cores:
KHR_coopmat`) den cm2-Pfad. Das ist der Hebel — nicht der
Algorithmus, sondern die Hardware-Beschleunigung.

---

## 2. Optimierungs-Versuch — K-LDS-Staging

### 2.1 Hypothese

Im aktuellen Kernel liest jeder Thread im Inner-Loop:
```glsl
for (uint d = 0; d < p.head_dim; ++d) {
    s += q[q_off + d] * k[k_pos_off + d];
}
```
* `q[q_off + d]`: alle 64 Threads lesen das gleiche Q-Element →
  caching/broadcast.
* `k[k_pos_off + d]`: thread `tid` liest K-Row bei position
  `tile_base + tid`. Adjacent threads access non-adjacent memory
  (stride = `pos_stride` = 1024 floats = 4 KB).

Das sieht "non-coalesced" aus. Wenn man K in LDS staged (Wave
collaborativ lädt 64 K-Rows × 128 dims = 32 KB), wird der globale
Read coalesced (adjacent threads access adjacent dims), und der
Inner-Loop liest nur LDS (~4 cycles vs ~100+ für global).

### 2.2 Implementierung

Eingebaut: 32 KB `k_tile_lds[TILE * HEAD_DIM]`. Vor dem Inner-Loop:
Wave kollaborativ lädt 64 K-Rows in LDS via 256-B-coalesced Bursts.
Inner-Loop liest aus LDS statt global. V-Staging gleichzeitig wäre
64 KB → LDS-Limit überschritten (RDNA4 64 KB pro CU), also nur K.

(Erster Versuch staged BEIDES, V auch — das produzierte
SIGFPE-Crashes durch LDS-Overflow auf 64.25 KB. Reduziert auf K-only,
parity-tests grün.)

### 2.3 Messung — durchgehend NEGATIV

```
| pp   | Sprint 5B (kein LDS-stage) | Sprint 6 (K-LDS-stage) | Δ      |
|------|---------------------------|------------------------|--------|
|  64  |          1489 tok/s       |       1248 tok/s       | -16%   |
|  128 |          1641 tok/s       |       1283 tok/s       | -22%   |
|  256 |          1337 tok/s       |       1042 tok/s       | -22%   |
|  512 |           921 tok/s       |        733 tok/s       | -20%   |
```

Parity-Tests grün (146/146), aber Performance ist konsistent 16-22%
schlechter.

### 2.4 Begründung — warum die Hypothese falsch war

(a) **RDNA4's L1-Cache war schon effektiv**. Jeder Thread liest 128
    sequentielle K-Floats nacheinander (`for d in 0..head_dim`).
    Der Hardware-Prefetcher erkennt das Pattern. Die "non-coalesced
    by tid"-These übersieht, dass innerhalb eines Threads die Reads
    streng linear sind.

(b) **Staging verdoppelt Memory-Traffic**. Original: 1 Read pro
    K-Element global. Mit Staging: 1 Read global + 1 Write LDS +
    1 Read LDS = 3× IO. LDS ist schneller als global — aber nur
    wenn der Read-Pattern global SCHLECHT war. Wenn global schon
    gut war, ist Staging reine Mehrarbeit.

(c) **LDS-Bank-Conflicts**. Layout `k_tile_lds[local_pos *
    HEAD_DIM + d]` mit HEAD_DIM=128 = 512 Bytes. RDNA4 LDS hat 32
    Banks à 4 Bytes = 128 Bytes pro Reihe. Stride 512 Bytes = 4
    Banks → 4-way Bank-Conflicts beim Inner-Loop-Read. Theoretisch
    sollte das nur 4× LDS-Latenz sein, aber kombiniert mit dem
    Compiler-Reschedule sieht es so aus dass man die Wins verliert.

(d) **Tile-Setup-Cost**. Das Pre-Loading läuft in einem nested
    Loop: für jedes der 64 local_pos läuft jeder Thread über 4 LDS-
    Schreibops. Das sind 64×4 = 256 Loop-Iterationen pro Tile pro
    Thread → ~1024 Cycles, vs der ursprüngliche Inner-Loop ohne
    Staging: 128 Iterationen pro Tile pro Thread. Mit dem Staging
    haben wir mehr Loop-Overhead UND nicht weniger globalen
    Memory-Verkehr.

### 2.5 Ergebnis: Reverted

`git checkout vk_shaders/flash_attn_batch.comp` — Shader ist
identisch zu Sprint 5B. Keine Code-Änderung in diesem Sprint.

---

## 3. Was wäre der echte Sprint 6?

Drei mögliche Hebel, sortiert nach erwartetem Gewinn:

### 3.1 Br > 1 (Tiled-Q Flash-Attention) — geschätzt +50-100% bei pp ≥ 1024

Eigentliches "vollständiges" Flash-Attention. Ein Workgroup
verarbeitet Br Queries gleichzeitig gegen die gleiche K-Tile.
LDS-Layout:
* Q-Tile: Br × HEAD_DIM = 16 × 128 × 4 = 8 KB (Br=16)
* K-Tile: Bc × HEAD_DIM = 32 × 128 × 4 = 16 KB (Bc=32)
* V-Tile: Bc × HEAD_DIM = 16 KB
* Score-Tile: Br × Bc = 16 × 32 × 4 = 2 KB
* Total: 42 KB → passt in 64 KB LDS

Algorithmische Komplikation:
* Online-Softmax ist **per-Query**: jede der Br Queries hat eigenen
  running max + running sum. Br × 2 floats in registers/LDS pro WG.
* Causal-Mask **innerhalb** des Q-Tiles: query i im Q-Tile sieht
  KV-Position 0..(q_start + i). Per-Element-Check oder per-Tile-
  Skip-Logik.
* GQA: Br Queries können unterschiedliche KV-Heads haben, oder alle
  den gleichen. Aktuelle GQA-Ratio Qwen3=4 (32 Q-Heads / 8 KV-Heads).
  Falls Br ≤ 4: Q-Tile teilt sich KV-Head → trivial. Falls Br > 4:
  müssen Mehrere KV-Heads im LDS sein → komplizierter.

Geschätzt 250-400 LOC GLSL plus Pipeline-Spec-Constant-Tuning.
Korrektheits-Risiko: hoch (Online-Softmax across Br ist nicht-trivial,
viele off-by-one Falle).

**Sollte als eigener mehrtägiger Sprint laufen, nicht als
single-pass Improvement.**

### 3.2 f16 K/V — geschätzt +30-50% bei langem pp

Memory-Bandbreite halbiert. Erfordert:
* KV-Cache-Format-Änderung: f32 → f16 (3 GB → 1.5 GB bei
  max_seq_len=8192 für Qwen3-8B)
* Alle Reader-Shader anpassen (Attention, Read-Back-Logic)
* Numerik-Validierung: f16 hat ±0.001 RoPE-Drift bei langen Sequenzen
* Shader-Extensions: GL_EXT_shader_16bit_storage, GL_EXT_shader_explicit_arithmetic_types_float16

**Eigener Sprint, ~1-2 Tage Aufwand.**

### 3.3 coopmat-Attention — geschätzt +50-150% bei langem pp

WMMA für QK und PV. Erfordert:
* Coopmat-fähigen Q-Tile/K-Tile Layout (16×16 oder 16×32 Tiles)
* Coopmat-Akkumulator-Konversion in/aus Online-Softmax (FP32
  Akkumulator beibehalten, BF16/FP16 für Inputs)
* Test gegen Coopmat-fähige RDNA4 (KHR_coopmat aktiviert)

**Eigener Sprint, ~2-3 Tage Aufwand. Kombiniert mit f16 K/V wäre
das die llama.cpp-cm2-Strategie.**

### 3.4 Niedrige Hängende Früchte: Conditional Barriers

Aus dem Audit: llama.cpp PR #12135-Pattern. Statt nach jedem
Layer-Op `vkCmdPipelineBarrier(SHADER_WRITE→SHADER_READ)`, prüfen
ob die Buffer-Reads/Writes wirklich überlappen. Geschätzter Gewinn:
5-15% über alle pp.

**Niedrigster ROI, aber niedrigster Aufwand. ~1 Tag.**

---

## 4. Tests & Files

```
cargo test --release

test result: ok. 24 passed
test result: ok.  9 passed
test result: ok. 18 passed
test result: ok. 60 passed
test result: ok.  8 passed
test result: ok. 27 passed
                ────
                146 / 146 ✓
```

Files Touched: KEINE Code-Änderung in diesem Sprint (Shader-Edit
revertiert). Nur dieser Report neu.

```
new file:   results/v02_sprint6_flash_attention.md
```

---

## 5. Empfehlung

**Sprint 7 (oder Sprint 6-Reboot)**: Wähle EINEN der drei großen
Hebel (Br>1 / f16-KV / coopmat) und führe ihn als eigener
mehrtägiger Mini-Sprint mit:
* Schritt 1: Korrektheits-Skelett (Reference-Implementation in Rust
  oder NumPy)
* Schritt 2: Parity-Test-Harness (alle Tile-Größen, alle GQA-Ratios)
* Schritt 3: GLSL-Implementierung
* Schritt 4: A/B-Bench mit Sprint-5B-Baseline

Erwartung bei korrekt umgesetztem Br=16-Flash-Attention:
* pp=1024: 555 → ~1000-1300 tok/s (+80-130%)
* pp=4096: 303 → ~500-700 tok/s (+65-130%)

Damit wäre VF/llama bei pp=4096 ~0.15-0.21× (von aktuell 0.09×) —
immer noch 5× Gap, aber halbiert. Kombiniert mit f16-KV wäre VF
realistisch bei ~0.3× von llama.cpp bei langem pp.

**KEIN Sprint sollte die O(N²) Compute-Untergrenze als "Bug" oder
"missing feature" framen — sie ist ein mathematisches Gesetz für
exact Attention. Die Lücke zu llama.cpp ist messbarer
Konstantfaktor-Overhead, nicht Algorithmus-Klasse.**
