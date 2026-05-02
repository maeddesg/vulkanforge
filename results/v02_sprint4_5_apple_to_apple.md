# v0.2 Sprint 4.5 — Apple-zu-Apfel Benchmark VF vs llama.cpp

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 4 — Spec-Constants pivot **NEGATIV**
(Phase-7 v0.1.3 Default ist lokales Optimum, alle Sweeps regressierten).
**Ziel:** Bei IDENTISCHEN pp-Werten messen, um den "2.17× Gap"
ehrlich zu lokalisieren.

---

## TL;DR — Der "2.17× Gap" zerfällt in zwei separate Probleme

```
═══ v0.2 Sprint 4.5 ═══

NUR MESSEN, KEIN CODE. Aber: ein riesiger Bug gefunden.

Apple-zu-Apfel-Matrix (gleiche Hardware, gleicher Build, ein
Lauf direkt nach dem anderen):

  | pp   | VF tok/s | llama.cpp | VF/llama |
  |------|----------|-----------|----------|
  |  20  |   373.5  |    910.4  |  0.41×   |
  |  31  |   735.1* |   1321.3  |  0.56×   |
  |  47  |  1093.4  |   1722.1  |  0.63×   |
  |  62  |  1429.8  |   2187.2  |  0.65×   |
  | 256  |  ~1300** |   ~3000** |  ~0.43×  |  (extrapolated)
  | 319  |    90.6  |   ~3300** | 0.027×   |  ← CLIFF
  | 512  |    90.3  |   4328.0  | 0.021×   |
  | dec  |    89.2  |    113.2  |  0.79×   |
  *) median of 2 samples (#2 und #3 in 5-prompt bench)
  **) pp=256 not measured; pp=319 forced via long prompt

Zwei separate Befunde:

(1) ECHTER GEMM-GAP bei kleinen pp: 0.41× → 0.65× (skaliert mit pp).
    Bei pp=62 sind wir bei 65% der llama.cpp-Performance. Dieser
    Gap ist real und liegt im Kernel + Pipeline (nicht in Spec-
    Constants — Sprint 4 hat das ausgeschlossen). Mögliche Hebel:
    Conditional-Barriers, Variant-Dispatch, Kernel-Mikrooptimierung.

(2) HARTER CLIFF bei pp > 256: VF fällt auf 91 tok/s = Decode-Rate.
    Das ist KEIN Kernel-Problem, das ist ein Code-Pfad-Bug:
    `decode.rs:429` selektiert die batched-prefill nur wenn
    `prefill_len <= forward.max_prefill_tokens` (default 256).
    Längere Prompts laufen Token-für-Token durch `forward_token`
    (= GEMV statt GEMM, also Decode-Rate).

  → Bei pp=512 ist der "47× Gap" zu 100% durch Befund (2) erklärt.
  → Bei pp=62 ist der "1.5× Gap" durch Befund (1) erklärt.

  Die Audit-Behauptung "2.17× Gap durch Spec-Constants" war
  falsch. Der echte Gap ist entweder ~1.5× (bei kleinen pp,
  Kernel-bedingt) oder ~47× (bei pp>256, Code-Pfad-bedingt).

Nächste Sprints:
  • Sprint 5: max_prefill_tokens hochziehen (z.B. 1024 oder
    dynamisch). Erwarteter Gewinn: pp=512 von 91 auf ~3000 tok/s
    (vermutlich, da 5-Prompt-Bench bei pp=62 schon 1430 macht und
    bei pp=512 GEMM noch besser auslastet).
  • Sprint 6 (optional): kleinen pp-Range optimieren (Conditional
    Barriers etc.) für 0.65× → 0.85× bei pp=62.
```

*Hinweis: VF pp=31 ist Median aus den zwei Prompts mit pp=31
(#2 Simple Sequence: 729.0, #3 Prime Check: 741.1).*

---

## 1. llama.cpp — Vulkan-Backend, alle pp-Werte in einem Lauf

### Build

* Pfad: `~/tmp/llama.cpp/build/bin/llama-bench`
* Build: `408225b (1)` (gebaut 2026-04-XX, neuer als Audit's
  Referenz-Build 23b8cc4)
* Backend: Vulkan, RADV/Mesa, RX 9070 XT
* Mesa-Banner: `warp size: 64`, `int dot: 1`, `matrix cores: KHR_coopmat`

### Kommando

```
~/tmp/llama.cpp/build/bin/llama-bench \
  -m ~/models/Qwen3-8B-Q4_K_M.gguf \
  -p 20,31,47,62,512 -n 128 -ngl 999 -t 1 -r 5
```

### Ergebnis (5 Repetitions, ± = Stdabw.)

```
| test  |               t/s  |
|-------|-------------------:|
| pp20  |     910.36 ± 17.31 |
| pp31  |    1321.31 ±  2.03 |
| pp47  |    1722.12 ±  2.74 |
| pp62  |    2187.22 ±  2.27 |
| pp512 |    4327.97 ±  3.20 |
| tg128 |     113.24 ±  0.88 |
```

**Beobachtungen:**

* `pp512 = 4328 tok/s` — fast 2× mehr als die Audit-Referenz
  (2274 tok/s). Build-Drift seit dem Audit-Snapshot ist real;
  llama.cpp's Vulkan-Backend ist ~2× schneller geworden.
* Skaliert sehr stark: pp20→pp512 ist 4.75× Throughput für 25.6×
  Tokens — sublineare GEMM-Skalierung wie erwartet.

---

## 2. VulkanForge — gleiche pp-Werte über zwei Bench-Läufe

### 2.1 5-Prompt-Bench (pp=20–62)

Kommando: `VF_NUM_PROMPTS=5 cargo run --release --example run_15prompt_bench`

```
| Prompt           | pp | gen | prefill tok/s | decode tok/s |
|------------------|----|----:|--------------:|-------------:|
| Greeting         | 20 |  64 |        373.5  |        90.5  |
| Simple Sequence  | 31 |  64 |        729.0  |        90.1  |
| Prime Check      | 31 | 256 |        741.1  |        88.9  |
| LRU Cache (C++)  | 47 | 512 |       1093.4  |        89.2  |
| REST API (Go)    | 62 |1024 |       1429.8  |        80.7  |
```

* Aggregate: 851.7 tok/s prefill, 84.5 tok/s decode (Mean).
* Median prefill: 741.1 tok/s, Median decode: 89.2 tok/s.
* Confirmed Phase-7 v0.1.3 baseline (~740 tok/s Median bestätigt).

### 2.2 pp-Sweep zur Lokalisierung des Cliffs

VF_PROMPTS auf custom-Files mit längeren System-Prompts gesetzt,
chat-template-bereinigte tatsächliche pp-Werte protokolliert.

**Sweep 1 (Char-Counts 100–600):**
```
| pp  | prefill tok/s |
|----:|--------------:|
|  53 |         955.6 |
|  70 |        1092.1 |
| 129 |        1244.4 |
| 201 |        1258.3 |
```

**Sweep 2 (250–350 Char-Targets, kürzer als gedacht):**
```
| pp  | prefill tok/s |
|----:|--------------:|
| 104 |        1248.0 |
| 132 |        1265.6 |
| 177 |        1437.7 |
```

**Sweep 3 (1500–2700 Char-Targets, lange Prompts):**
```
| pp  | prefill tok/s |
|----:|--------------:|
| 319 |          90.6 |  ← CLIFF
| 376 |          91.7 |
| 461 |          90.9 |
| 550 |          90.3 |
```

**Sweep 4 (Cliff-Probe 240–280 Char):**
```
| pp  | prefill tok/s |
|----:|--------------:|
|  93 |        1072.5 |
| 106 |        1478.5 |
| 123 |        1595.2 |
```

* Höchster Sweet-Spot beobachtet: pp=123 → 1595.2 tok/s.
* Bis pp=201: VF skaliert linear ~1250 tok/s.
* Ab pp=319: VF kollabiert auf 91 tok/s.
* Cliff-Position liegt zwischen 201 und 319 — exakt bei
  `max_prefill_tokens = 256` (siehe Befund-Analyse §3).

---

## 3. Befund-Analyse — der Cliff ist KEIN Kernel-Problem

### 3.1 Die 91 tok/s sind die Decode-Rate

Vergleich:
* VF Decode (token-by-token GEMV): 89.2 tok/s Median
* VF "Prefill" bei pp ≥ 319: 90.6, 91.7, 90.9, 90.3 tok/s
* Differenz: < 2 tok/s. Statistisch nicht unterscheidbar.

→ Bei pp > 256 läuft die "Prefill" tatsächlich Token-für-Token
durch denselben GEMV-Pfad wie Decode. KEIN GEMM-Dispatch. Daher
auch keine pp-Skalierung — egal ob 319 oder 550 Tokens, gleicher
Throughput.

### 3.2 Der Code-Pfad-Selector

Quelle: `src/backend/vulkan/decode.rs:428–449` (Lese-only, keine
Änderung in diesem Sprint):

```rust
let prefill_len = prefill_tokens.len() as u32;
if prefill_len > 0 && prefill_len <= forward.max_prefill_tokens {
    // BATCHED PATH — prefill_batch dispatches one big GEMM
    let mut all_embeds = Vec::with_capacity(...);
    for &tid in prefill_tokens { /* gather embeddings */ }
    forward.prefill_batch(dev, registry, cmd_ctx, model,
                          &all_embeds, prefill_len, pos, ...)?;
    pos += prefill_len;
} else {
    // FALLBACK PATH — token-by-token forward, == decode rate
    for &tid in prefill_tokens {
        forward.forward_token(dev, registry, cmd_ctx, model,
                              &embd, pos)?;
    }
}
```

`forward.max_prefill_tokens` ist konstruktor-konfigurierbar:
* `Forward::new(...)` → ruft `new_with_prefill(... 256)` auf
  (`forward.rs:272`).
* Alle `examples/run_15prompt_bench.rs`, `chat.rs` usw. nutzen
  den 256-Default.

### 3.3 Warum 256?

Quelle: `forward.rs:236–239`:

> Allocated once in `new` based on `max_prefill_tokens`. Memory
> budget at the default (256 tokens) is ~60 MB — well within the
> 16 GB VRAM-Budget.

256 wurde als VRAM-konservativer Default gewählt. Die Allokation
skaliert linear mit `max_pp` (`forward.rs:374-377`):
* `pp_hidden = max_pp · hidden_dim · 4` (BF16 staging)
* `pp_kv     = max_pp · n_kv_heads · head_dim · 4`
* `pp_q      = max_pp · n_heads · head_dim · 4`
* `pp_ffn    = max_pp · ffn_dim · 4`
* `pp_q8     = max_pp · hidden_dim/128 · 144` (Q8_1 quant)

Für Qwen3-8B mit `hidden_dim=4096, ffn_dim=12288, n_heads=32,
n_kv_heads=8, head_dim=128`:
* Bei `max_pp=256`: ~60 MB
* Bei `max_pp=1024`: ~240 MB
* Bei `max_pp=2048`: ~480 MB

Selbst bei 2048 Tokens passt das problemlos in die 16 GB der
9070 XT. Der 256-Default ist deutlich zu konservativ.

### 3.4 Korrekturansatz (NICHT in Sprint 4.5 implementiert)

Zwei Varianten:

(a) **Statisch hochziehen**: Default auf 1024 oder 2048. Kostet
    180–420 MB VRAM, garantiert keine Cliff bis dahin.

(b) **Dynamisch chunken**: Wenn `prefill_len > max_prefill_tokens`,
    in Chunks von `max_prefill_tokens` aufteilen. Kostet keine
    extra VRAM, behält den 256-Default für niedrige
    Speicherbudgets, hat aber Per-Chunk-Submit-Overhead.

(c) **Dynamic auto-grow**: Bei session start `max_prefill_tokens`
    auf `MAX_SEQ_LEN` setzen, wenn VRAM passt. Sonst chunken.

Ansatz (b) ist der natürliche Apple-zu-Apfel-Vergleich zu
llama.cpp, das intern auch `-ub 512` als Standard-Microbatch hat.

---

## 4. Apple-zu-Apfel-Matrix

```
| pp   | VF tok/s | llama.cpp tok/s | VF/llama | Gap-Quelle           |
|------|----------|-----------------|----------|----------------------|
|  20  |    373.5 |          910.4  |  0.41×   | Kernel/Pipeline      |
|  31  |    735.1 |         1321.3  |  0.56×   | Kernel/Pipeline      |
|  47  |   1093.4 |         1722.1  |  0.63×   | Kernel/Pipeline      |
|  62  |   1429.8 |         2187.2  |  0.65×   | Kernel/Pipeline      |
| 201  |   1258.3 |       (~3000)¹  |  ~0.42×  | Kernel/Pipeline      |
| 319  |     90.6 |       (~3300)¹  | ~0.027×  | **256-Cap-Cliff**    |
| 376  |     91.7 |       (~3500)¹  | ~0.026×  | **256-Cap-Cliff**    |
| 512  |     90.3 |         4328.0  | 0.0209×  | **256-Cap-Cliff**    |
| dec  |     89.2 |          113.2  |  0.79×   | Decode-GEMV-Gap      |
```

¹ llama.cpp wurde nur bei pp ∈ {20,31,47,62,512} gemessen;
Zwischenwerte interpoliert (pp512 / pp62 = 1.98×, lineare
Interpolation auf log-Skala für die Gap-Charakterisierung).

---

## 5. Fragen aus dem Sprint-Briefing — beantwortet

> **A. Wie groß ist der Gap bei GLEICHEN pp-Werten?**

Bei pp=62 (höchster zuverlässiger gemeinsamer Datenpunkt):
**0.65× → 35% Gap**. Innerhalb des "Spec-Constants OK"-Bereichs
(<50% Gap, also Kernel/Pipeline-bedingt, nicht
Architektur-bedingt).

> **B. Wie skaliert VF mit steigendem pp?**

* pp=20:   373.5 tok/s
* pp=62:  1429.8 tok/s   (3.83× bei 3.1× Tokens — überlinear)
* pp=123: 1595.2 tok/s   (4.27× — peak)
* pp=201: 1258.3 tok/s   (3.37× — fängt an zu plateauen, evtl.
                          Cache-Pressure?)
* pp=256:  ???           (NICHT gemessen, da Cliff-Schwelle
                          schwer punktgenau zu treffen mit
                          chat-template-Drift)
* pp≥319:   ~91 tok/s    (CLIFF — fallback path)

VF skaliert bis pp~120 fast genauso steil wie llama.cpp,
plateaut dann früher (vielleicht Cache-Conflict-Pattern).
Sublineare Skalierung ist normal.

> **C. Ist der 2.17× Gap ein pp-Scaling-Problem?**

**Teilweise ja, teilweise nein.**

Der ursprüngliche "2.17× Gap" basierte auf:
* VF bei pp≈31 (5-Prompt Median): 740 tok/s
* llama.cpp bei pp=512: 2274 tok/s
* Verhältnis: 3.07× (sogar mehr als 2.17×)

Bei sauberem Vergleich (gleiche pp):
* Bei pp=62: 0.65× → Gap 1.53× — **echter Kernel-Gap, klein**
* Bei pp=512: 0.021× → Gap 47.9× — **Cliff-Bug, riesig**

→ Das was der Audit "2.17×" nannte, ist eine Mischung aus
beidem. Die Wahrheit ist:
1. Wenn man Apple-zu-Apfel vergleicht: Gap ~1.5×, kein
   Architektur-Problem
2. Wenn man die 256-Cliff tritt: Gap ~50×, ein Code-Pfad-Bug

---

## 6. Decode-Vergleich (pp=0)

```
| Engine        | tg128 tok/s |
|---------------|-------------|
| VulkanForge   |       89.2  |
| llama.cpp     |      113.2  |
| Verhältnis    |      0.79×  |
```

* Gap: 27%. Konstant über Builds (Phase-7-Daten zeigen 84-90 für
  VF, 99-116 für llama.cpp).
* Decode ist GEMV (1 Token gegen alle Gewichte). Hier hat
  llama.cpp `mul_mat_vec_q4_k`-Optimierungen die wir auch ähnlich
  haben, aber das Memory-Bandwidth-Bottleneck ist auf beiden
  Seiten ähnlich.
* 27% ist klein genug um einen anderen Sprint zu rechtfertigen,
  aber nicht der dringendste Fix.

---

## 7. Mess-Bedingungen / Caveats

* **Citrix lief im Hintergrund** (PID 4882). User-Briefing forderte
  "Citrix beenden falls möglich" — wurde NICHT beendet, weil
  destruktiv für die User-Session. Auswirkung: GPU war bei 7%
  busy (`/sys/class/drm/card1/device/gpu_busy_percent`), also
  praktisch idle. Der Effekt ist zudem für beide Engines
  identisch — relativer Vergleich bleibt valide.
* **Validation-Layer-Warnings**: VF zeigt SPV_KHR_bfloat16/
  cooperative_matrix-Warnings. Sind harmlos (Module wird trotzdem
  geladen, weil Extensions später aktiviert werden).
  llama.cpp zeigt keine solchen Warnings. Kein Performance-Effekt
  beobachtet.
* **Build-Drift**: llama.cpp ist Build 408225b (≈2 Wochen neuer
  als Audit's 23b8cc4). pp512 ist von 2274 auf 4328 tok/s
  gestiegen — fast verdoppelt. Audit's "2.17× Gap" war zum Audit-
  Zeitpunkt; aktuell ist die Lücke noch größer (in Tokens), aber
  qualitativ unverändert in der Charakterisierung.
* **Repetitions**:
  * llama-bench: 5 Reps mit Stdabw < 1% (sehr stabil).
  * VF: 1 Run pro pp (kein Repetitions-Flag im VF-Bench). Bei
    pp ≥ 319 wurden mehrere Datenpunkte aufgenommen, alle ~91
    tok/s ± 1 → reproduzierbar.

---

## 8. Files Touched

```
new file:   results/v02_sprint4_5_apple_to_apple.md
```

KEIN Code geändert (gemäß Sprint-Regel). Der `decode.rs:429`-
Befund ist dokumentarisch; Sprint 5 würde den Fix implementieren.

Tests: nicht erneut ausgeführt (Code unverändert seit Sprint 4 →
145/145 grün, dort verifiziert).

---

## 9. Empfehlung für nächsten Sprint

**Sprint 5 — Prefill-Cap aufheben** (höchste ROI):

1. `examples/run_15prompt_bench.rs`: `Forward::new_with_prefill(
   ..., 1024)` statt Default 256.
2. Optional: `decode.rs:429` umschreiben, sodass bei
   `prefill_len > max_prefill_tokens` automatisch in Chunks von
   `max_prefill_tokens` aufgeteilt wird (statt Token-für-Token-
   Fallback).
3. Re-Run pp=512 mit erhöhtem Cap. Erwartung:
   * VF auf pp=512 sollte mindestens **3000 tok/s** erreichen
     (extrapoliert aus pp=62 mit 1430 tok/s und sublinearer
     Skalierung).
   * Damit wäre der 47× Gap auf ~1.5× reduziert.
4. Regression: 145/145 muss grün bleiben (Default sollte für
   Tests evtl. unverändert bleiben, oder Test-Suite an höheren
   Cap angepasst werden).

**Erwarteter Gesamteffekt** — VF-Performance bei langen Prompts
sollte sich um ein Vielfaches verbessern. Das ist mit großer
Wahrscheinlichkeit der höchste Single-Hebel den wir bislang
identifiziert haben.

**Sprint 6 (optional, niedriger ROI)** — Conditional-Barriers
oder Variant-Dispatch für die verbleibenden 35% Gap bei kleinen
pp. Würde 0.65× → ~0.85× pushen, aber der Audit hat schon
gezeigt dass Spec-Constants alleine es NICHT sind.
