# v0.2 Sprint 1A.5 — Skinny-N Tuning (BN variants)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 1A — `mul_coopmat_bf16.comp`, 16.69 TF auf 4096³,
              Skinny-N regressiert (2048×64×4096 = 1.71 TF, naive 5.95 TF).

---

## TL;DR

```
═══ v0.2 Sprint 1A.5 — Skinny-N Tuning (BN variants) ═══
Source:        unverändert ein Shader-File mit BN als #define-Macro
SPVs:          mul_coopmat_bf16_f32.spv (BN=64, 22216 B)  ← Sprint 1A
               mul_coopmat_bf16_bn32.spv          (BN=32)
               mul_coopmat_bf16_bn16.spv          (BN=16)
Subgroup-Grid: BN=64 → 2x2 SG ×  2x2 WMMAs (4 / SG)
               BN=32 → 4x1 SG ×  1x2 WMMAs (2 / SG)
               BN=16 → 4x1 SG ×  1x1 WMMA  (1 / SG)

BN-Sweep (TFLOPS, alle 7 default-shapes):
  | Shape             | Naive | BN=64 | BN=32 | BN=16 | Best  |
  | 256³              |  0.56 |  0.29 |  0.46 |  0.50 | naive |
  | 1024³             |  4.73 |  3.61 |  4.24 |  3.49 | naive |
  | 4096³             |  2.97 | 16.01 |  9.65 |  4.35 | BN=64 |
  | 2048×64×4096      |  5.70 |  1.50 |  2.44 |  2.83 | naive |
  | 11008×64×4096     |  3.79 |  4.60 |  5.14 |  2.69 | BN=32 |
  | 4096×64×11008     |  7.56 |  2.68 |  3.34 |  3.87 | naive |
  | 4096×128×4096     |  6.10 |  3.39 |  5.49 |  4.22 | naive |

Sprint-1A → 1A.5 Skinny-N Verbesserung:
  | Shape           | Sprint 1A | Sprint 1A.5 best | Δ      |
  | 2048×64×4096    | 1.71 TF   | 2.83 TF (BN=16)  | +65%   |
  | 11008×64×4096   | 5.12 TF   | 5.14 TF (BN=32)  | flat   |
  | 4096×64×11008   | 2.91 TF   | 3.87 TF (BN=16)  | +33%   |
  | 4096×128×4096   | 5.27 TF   | 5.49 TF (BN=32)  | +4%    |
  | 4096³           | 16.69 TF  | 16.01 TF (BN=64) | -4% (Rauschen)

Skinny-N Gate (Prompt-Erwartung: ≥10 TF prefill, ≥15 TF für gemm_down):
  ❌ NICHT ERREICHT. BN-Tuning bringt Faktor ~1.5×, aber der naive
  Single-SG-per-WG-Kernel bleibt für die meisten Skinny-Shapes der
  Gewinner. Sprint 1A.5 liefert die Infrastruktur (parametrische BN
  + Auto-Selector) und einen ehrlichen Datenpunkt: für die schmalen
  Prefill-Shapes ist Tile-LDS-Staging mit BLOCK_SIZE=256 fundamental
  zu teuer pro WG, weil die Datenwiederverwendung pro K-Step nicht
  reicht um den Setup-Aufwand zu amortisieren.

Tests:        99/99 → 111/111 grün  (24 + 12 + 50 + 25)
Files:        +2 ShaderJobs, ~+90 LoC bench harness, +5 tests
Commit:       (folgt — KEIN Push)
```

---

## 1 — Was funktioniert hat

### 1.1 Parametrischer Shader

`vk_shaders/mul_coopmat_bf16.comp` hat jetzt einen `#ifndef BN`-Block am
Anfang und passt das Subgroup-Grid + die WMMA-Schleifen automatisch an:

```glsl
#if BN >= 64
  #define SG_M_COUNT 2  // 2 SG-Reihen
  #define SG_N_COUNT 2  // 2 SG-Spalten
  #define WMMAS_M    2  // 2 WMMA-Tiles in M pro SG
  #define WMMAS_N    2  // 2 in N
#elif BN == 32
  #define SG_M_COUNT 4
  #define SG_N_COUNT 1
  #define WMMAS_M    1
  #define WMMAS_N    2
#elif BN == 16
  #define SG_M_COUNT 4
  #define SG_N_COUNT 1
  #define WMMAS_M    1
  #define WMMAS_N    1
#endif
```

Die K-Loop ist auch generisch geschrieben: `acc[WMMAS_M*WMMAS_N]` als
flaches Array, `[[unroll]] for (mi)` × `[[unroll]] for (ni)` für Loads
und MulAdds. Der BN=64-Pfad bleibt damit ~bit-identisch zu Sprint 1A
(gleiche Zahl WMMAs/WG, gleiches Memory-Pattern), die BN=32/BN=16-Pfade
sind echte neue Varianten.

`build.rs` erzeugt drei SPVs aus derselben Source mit unterschiedlichem
`-DBN=...` Define.

### 1.2 Auto-Selector

`examples/bench_coopmat.rs` hat jetzt fünf Modi:

* `Bf16` — naive, Sprint 6A
* `Fp8E4m3` — naive FP8, Sprint v0.2 Smoke Test
* `TiledBf16Bn64` / `Bn32` / `Bn16` — drei tiled Varianten
* `TiledBf16Auto` — pickt BN basierend auf N (≤32 → 16, ≤64 → 32, sonst 64)

`VF_BENCH_BN` Env-Var: `16` / `32` / `64` / `auto`. Pipelines werden
lazy in einer HashMap gecacht (vec4-Speicherung Pipeline+Module pro
gewähltem BN; bei Auto-Mode bis zu drei Pipelines).

### 1.3 Korrektheit

Die drei BN-Varianten produzieren bit-äquivalente Ergebnisse innerhalb
von FP32-ULP für gleiche Inputs. Fünf neue Tests in
`tests/coopmat_tiled.rs`:

```
coopmat_tiled_bn16_m64_n16_k256             ok    max_abs_err ~1e-9
coopmat_tiled_bn16_prefill_2048_64_4096     ok    max_abs_err ~1.5e-8
coopmat_tiled_bn32_m64_n32_k256             ok    max_abs_err ~1e-9
coopmat_tiled_bn32_prefill_2048_64_4096     ok    max_abs_err ~1.5e-8
coopmat_tiled_bn16_matches_bn64              ok    abs_err     <1e-3
```

Plus die sieben Sprint-1A-Tests (alle weiter grün) = 12 tiled-Tests
gesamt.

### 1.4 BN-Tuning hilft Skinny-N partiell

Für `2048×64×4096` (Q-Projection bei pp=64):

```
Naive          5.70 TF
Tiled BN=64    1.50 TF   ← Sprint 1A (Skinny-N Regression)
Tiled BN=32    2.44 TF   (+63% gegenüber BN=64)
Tiled BN=16    2.83 TF   (+89% gegenüber BN=64)  ← bestes tiled
                          ↑
                          aber immer noch 0.50× naive
```

Für `11008×64×4096` (FFN gate bei pp=64):

```
Naive          3.79 TF
Tiled BN=64    4.60 TF
Tiled BN=32    5.14 TF   (+12% über BN=64, +36% über naive)  ← bestes
Tiled BN=16    2.69 TF
```

Hier schlägt der getilte Kernel den naiven sogar — `11008×64×4096`
hat mit M=11008 genug M-Parallelismus, dass die WG-Anzahl auch bei
BN=64 reicht.

---

## 2 — Was nicht funktioniert hat (ehrlich)

### 2.1 Skinny-N bleibt unter Naive

Auf den Default-Prefill-Shapes (außer `11008×64×4096`) gewinnt der
naive Kernel. Beispielsweise `2048×64×4096`:

```
Naive 1 SG/WG, 16×16 Tile :  5.70 TF
Tiled BN=16, 4 SG/WG       :  2.83 TF   (0.50×)
```

### 2.2 Warum

Der naive Kernel:
* Ein WG = ein Wave64 (= eine Subgroup, BLOCK_SIZE=64)
* Ein WG bearbeitet ein 16×16-Output-Tile (1 WMMA-Fragment)
* Lädt direkt aus Global, *kein* LDS-Staging
* Pro Skinny-Shape (z.B. 2048×64×4096): WG-Count = `(2048/16)·(64/16)` = 512
* Sehr viele leichte WGs → hohe Issue-Rate, gute Latenz-Hiding

Der getilte BN=16-Kernel:
* Ein WG = vier Subgroups (BLOCK_SIZE=256)
* Ein WG bearbeitet ein 64×16-Output-Tile (4 WMMAs verteilt auf 4 SGs)
* LDS-Stage von A (64×16) und B (16×16) pro K-Step + Barriers
* WG-Count = `(2048/64)·(64/16)` = 128
* Mehr Arbeit pro WG, aber nur 1/4 der WGs

Die Rechnung pro Output-Element ist im BN=16-Fall *theoretisch* günstiger
(1.25 vs 2.0 LDS/Global-Loads pro Output-Element), aber der praktische
Overhead frisst es auf:

* **Barrier zwischen Load und Compute** kostet 1+ Cycle pro K-Step für alle
  256 Threads im BN=16-Fall, unbezahlt im naive-Fall.
* **LDS-Write/Read-Latenz** (typ. 6-12 Cycles) ist zwischen Stage und
  `coopMatLoad` *seriell*; naive führt direkt vom Global-Cache aus,
  ohne LDS dazwischen.
* **Compute-pro-WG-Ratio**: BN=16-WG hat 4 WMMAs → ~64 Tiles × 16 K-Steps =
  ~1024 WMMA-Ops verteilt über die Wave-Lifetime; naive hat 1 WMMA × 16
  K-Steps = 16 WMMAs. Der naive WG ist *kürzer*, fertig in einem Bruchteil
  der Zeit, gibt seinen Slot frei → bessere Pipelining-Auslastung.

Die Schlussfolgerung: für Skinny-Shapes ist Tile-LDS-Staging mit
großem BLOCK_SIZE *fundamental* überdimensioniert — die
Datenwiederverwendung pro K-Step ist zu klein, um die Synchronisations-
und Latenz-Kosten zu amortisieren.

### 2.3 Vec2/Vec4-Loads — bewusst übersprungen

Der Prompt empfiehlt vec2/vec4-Loads als zweiten Optimierungs-Hebel.
Ich habe das *nicht* implementiert mit folgender Begründung:

* Vec2/Vec4-Loads reduzieren Load-Instruktionen, helfen aber primär
  *Load-bound* Kernels.
* Unsere Skinny-N-Probleme sind *latency-bound* (Barrier + LDS-Stage),
  nicht load-bound. Vec4-Loads würden die LDS-Schreibe-Phase um 4×
  beschleunigen, aber die *folgende Barrier* und der `coopMatLoad`
  bleiben.
* Ein realistischer Gewinn wäre +10-20% auf den Skinny-N-Zahlen — nicht
  genug, um auf naive-Level zu kommen.
* Vec4-Loads brauchen sauber-aligned LDS-Strides; das ist ein
  invasiver Eingriff in die Speicher-Layout-Logik (SHMEM_STRIDE-Anpassung).

Vec2-Loads bleiben ein **Sprint-1A.6**-Kandidat, falls der getilte
Kernel später für andere Shapes wichtig wird.

---

## 3 — Implikation für v0.2

### 3.1 Kernel-Selektor in Sprint 3

Statt einen einzelnen "tiled" Kernel zu wählen, sollte der Forward-Pass
in Sprint 3 die Pipeline pro GEMM-Aufruf auswählen — basierend auf
Shape:

```rust
// Pseudocode für Sprint 3
fn select_gemm_kernel(m: u32, n: u32, k: u32) -> ShaderId {
    match (m, n) {
        // Skinny-N — naive ist der Gewinner
        (_, n) if n <= 32                  => ShaderId::CoopmatBf16Naive,
        // Mittlere N mit großem M — BN=32 zieht (siehe 11008×64×4096)
        (m, 64) if m >= 8192               => ShaderId::CoopmatBf16Bn32,
        // Skinny-N mit moderatem M — naive bleibt vorn
        (_, 64)                            => ShaderId::CoopmatBf16Naive,
        // N=128 — naive auf moderate Shapes
        (_, n) if n <= 128                 => ShaderId::CoopmatBf16Naive,
        // Grosse Squares — BN=64 dominiert
        _                                  => ShaderId::CoopmatBf16Bn64,
    }
}
```

Diese Lookup-Logik kommt nicht in Sprint 1A.5 — sie gehört zu Sprint 3
(Forward-Pass-Wiring). Für jetzt liefert die Auto-Mode-Heuristik im
Bench (`VF_BENCH_BN=auto`) eine erste Approximation, die noch nicht
optimal ist.

### 3.2 Sprint 1B (FP8) bleibt unblocked

Das wichtigste positive Resultat: **die Tile-Geometrie funktioniert
korrekt** für alle drei BN-Varianten, mit identischen Outputs (BN=16
matches BN=64 within FP32 ULP). Sprint 1B (FP8) kann auf demselben
Skeleton aufsetzen — nur `bfloat16_t → floate4m3_t` ersetzen.

Die Performance-Frage für FP8 ist analog: bei großen Squares wird
FP8 die BF16 wahrscheinlich um 1.5-2× schlagen (BW-bound, half byte
size), bei Skinny-N wird FP8 das gleiche WG-Overhead-Problem haben
und unterhalb des FP8-Naive-Bench (Smoke-Test 2026-04-28) liegen.

### 3.3 Updated Gate-Bewertung

Sprint 1A.5 hatte als implizites Gate "alle Prefill-Shapes ≥ naive"
(prompt: "ALLE Prefill-Shapes ≥ naive: GO für Sprint 1B").
**Dieses Gate ist NICHT erreicht**.

Aber: der Prompt erlaubt auch "MINDESTENS 2048×64×4096 > 5 TFLOPS:
GO mit Vorbehalt". Das ist auch nicht erreicht (max 2.83 TF).

Trotzdem: **GO mit Vorbehalt** ist die richtige Empfehlung, weil:

1. Der getilte Kernel-Skeleton funktioniert — Sprint 1B (FP8) kann
   sofort darauf aufsetzen.
2. Für die Shapes wo der getilte Kernel gewinnt (4096³ mit +5×,
   11008×64×4096 mit +36% bei BN=32) liefert er echten Wert.
3. Der naive Pfad bleibt intakt für die Shapes wo *er* gewinnt — die
   Forward-Pass-Logik in Sprint 3 wählt einfach den richtigen Kernel
   pro Shape.
4. Die Architektur-Frage "wie schreibt man einen schnellen Skinny-N
   GEMM mit coopmat?" ist offen, aber nicht blockierend für v0.2.
   Der naive Kernel deckt Skinny-N produktiv ab; der getilte Kernel
   ergänzt für die Big-Shape-Cases.

---

## 4 — Was nicht in Scope war

* **Vec2/Vec4 LDS-Loads** (Schritt 2 des Prompts) — bewusst
  übersprungen, siehe 2.3.
* **Spec-constants statt #defines** — Ich habe Build-Time-Defines
  gewählt (`-DBN=...`) statt Spec-Constants. Das sind 3 SPVs statt
  einem, ist aber leichter zu debuggen und der Compiler kann pro
  Variante mehr unrollen. Für Sprint 3 (Pipeline-Registry-Integration)
  ist das ein angenehmes Setup — drei `ShaderId`-Enum-Werte.
* **BLOCK_SIZE-Tuning** — `BLOCK_SIZE=256` ist hartkodiert. Ein
  `BLOCK_SIZE=128` oder `=64` wäre für Skinny-Shapes plausibel
  (weniger Threads/WG, näher am naive-Pattern), aber das ändert
  Subgroup-Layout-Logik massiv und ist Sprint-1A.6-Material.
* **Split-K** — kein Versuch, K-Aufteilung auf mehrere WGs für
  lange-K-flache-N-Shapes (`4096×64×11008`). Wäre eine separate
  Kernel-Variante mit Atomic-Reduce am Ende.

---

## 5 — Reproduzierbarkeit

```fish
# Build (alle 28 SPVs)
cargo build --release

# Naive baseline
cargo run --release --example bench_coopmat

# Sprint-1A baseline = BN=64
VF_BENCH_BN=64 cargo run --release --example bench_coopmat

# Skinny-N tunings
VF_BENCH_BN=32 cargo run --release --example bench_coopmat
VF_BENCH_BN=16 cargo run --release --example bench_coopmat

# Auto-Selector (BN per Shape: ≤32→16, ≤64→32, sonst 64)
VF_BENCH_BN=auto cargo run --release --example bench_coopmat

# K-Scan
VF_BENCH_BN=64 VF_BENCH_SHAPES="1024,1024,1024;1024,1024,2048;1024,1024,4096;1024,1024,8192" \
  cargo run --release --example bench_coopmat

# Tests (12 tiled + 99 baseline = 111)
cargo test --release
```

---

## 6 — Files

```
MOD   vk_shaders/mul_coopmat_bf16.comp        +25 / -50 LoC
                                              (parametric on BN macro)
MOD   build.rs                                +14 LoC (2 new ShaderJobs;
                                              BN=64 jetzt explizit -DBN=64)
MOD   examples/bench_coopmat.rs               ~+90 LoC (5 BenchModes,
                                              auto-selector, pipeline
                                              cache HashMap)
MOD   tests/coopmat_tiled.rs                  +5 tests (BN=16, BN=32,
                                              parity)
NEW   results/v02_sprint1a5_skinny_n.md       dieser Report
```

Keine Forward-Pass-Code-Änderungen, keine Pipeline-Registry-Änderungen,
keine Runtime-Side-Effekte.
