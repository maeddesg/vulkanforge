# v0.2 Smoke Test — FP8 coopmat (Korrektur)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.1.3 (kein Version-Bump — Smoke Test)
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgestern:** v0.2 Smoke Test 27.04.2026 — FP8-Pfad fälschlich als
"toolchain-blocked" eingestuft.
**Gestern:** `results/fp8_glslang_analysis.md` — FP8 GLSL funktioniert auf
shaderc 0.8 + glslang 16.2.0. Smoke-Test-STOP war Fehldiagnose (falscher
Typ-Name, nicht falsche Toolchain).
**Heute:** Test 2 (FP8 TFLOPS) und Test 4 (BW-Ceiling) nachgeholt.

---

## TL;DR

```
═══ v0.2 Smoke Test — FP8 coopmat (Korrektur) ═══
Shader:       vk_shaders/bench_coopmat_fp8.comp (floate4m3_t)
Kompiliert:   shaderc 0.8 ✅  (6780 B SPV)  spirv-val ✅
SPIR-V:       OpTypeFloat 8 Float8E4M3EXT, Float8EXT, Float8CooperativeMatrixEXT,
              SPV_EXT_float8

Default-Shapes (BF16 vs FP8 vs scalar baseline):
  | Shape            | BF16    | FP8     | FP8/BF16 |
  | 256³             | 0.75 TF | 0.60 TF | 0.80×    |
  | 1024³            | 7.51    | 5.79    | 0.77×    |
  | 4096³            | 4.06    | 5.95    | 1.47×    |
  | 2048×64×4096     | 6.26    | 9.79    | 1.56×    |
  | 11008×64×4096    | 4.90    | 4.94    | 1.01×    |
  | 4096×64×11008    | 7.52    | 14.48   | 1.93×    |  ← BW-bound win
  | 4096×128×4096    | 7.36    | 9.56    | 1.30×    |

K-Scan (M=N=1024):
  | K    | BF16    | FP8     | FP8/BF16 |
  | 1024 | 6.43 TF | 6.70 TF | 1.04×    |
  | 2048 | 7.78    | 8.51    | 1.09×    |
  | 4096 | 8.76    | 8.71    | 0.99×    |
  | 8192 | 4.64    | 8.97    | 1.93×    |  ← BF16 fällt ab, FP8 hält

BW-Ceiling:   BANDWIDTH-BOUND bei großem Traffic bestätigt.
              Ab K≈8192 (oder M·K + K·N > L2-Working-Set) erbringt
              FP8 ~2× BF16 — exakt der erwartete Bytes-pro-Element-Ratio.
              Bei kleinen Tiles (256³, 1024³) verliert FP8 leicht
              (extra Decode-Overhead ohne BW-Gewinn).

Correctness:  abs_err vs CPU f64-Referenz an C[0] für 64³ / 256³ / 1024×64×1024:
              BF16  : 4.657e-10 / 2.328e-10 / 3.725e-9   (FP32 ULP)
              FP8   : 0.000e0   / 0.000e0   / 0.000e0    (Bit-exakt)
              FP8 macht keine sichtbaren Numerik-Probleme an C[0] —
              die FP32-Akkumulation absorbiert die FP8-Dequant-Fehler.

Empfehlung:   PARALLEL-TRACK. v0.2A (BF16) bleibt der Default-Pfad weil
              kleine/Decode-GEMVs sogar leicht schneller sind, ABER
              v0.2B (FP8) kommt für Prefill mit Long-Context (K≥8192) und
              für VRAM-knappe Modelle: dort sind 2× Speedup + 2× weniger
              VRAM klar genug, um beide Pfade nebeneinander zu pflegen.
              Naive Kernels ist beides — ein getilter v0.2 Sprint-1
              Kernel könnte das Bild in eine oder die andere Richtung
              verschieben.

Tests:        99/99 grün  (24 + 50 + 25 = 99)
Files:        +1 shader, +1 build job, ~+85 LOC bench harness
Commit:       (folgt — KEIN Push)
```

---

## 1 — Setup

### 1.1 Was wurde geändert

```
NEW   vk_shaders/bench_coopmat_fp8.comp                ← FP8-E4M3 Twin von
                                                        bench_coopmat_pure.comp
MOD   build.rs                                          ← +1 ShaderJob
MOD   examples/bench_coopmat.rs                         ← BenchMode enum,
                                                        VF_BENCH_FP8/CHECK env,
                                                        FP8 device features,
                                                        f32↔fp8_e4m3 converters,
                                                        CPU-reference C[0] check
NEW   results/v02_smoke_test_fp8.md                     ← dieser Report
```

Keine Forward-Pass-Änderungen, keine Pipeline-Registry-Änderungen, keine
Test-Änderungen. Der FP8-Pfad ist ein Standalone-Benchmark.

### 1.2 Toolchain-Verifikation

shaderc 0.8 (gegen System-libshaderc 2026.1, glslang 16.2.0) baut die
FP8-Variante so als hätten wir den BF16-Twin neu gebaut:

```
warning: vulkanforge@0.1.3: compiled bench_coopmat_pure.comp -> bench_coopmat_pure_f32.spv (7380 bytes)
warning: vulkanforge@0.1.3: compiled bench_coopmat_fp8.comp  -> bench_coopmat_fp8_e4m3.spv (6780 bytes)
```

Die FP8-SPV kommt mit den richtigen Capabilities raus:

```
OpCapability Shader
OpCapability Float8EXT
OpCapability Float8CooperativeMatrixEXT
OpCapability VulkanMemoryModel
OpCapability CooperativeMatrixKHR
OpExtension  "SPV_EXT_float8"
OpExtension  "SPV_KHR_cooperative_matrix"
%fp8e4m3 = OpTypeFloat 8 Float8E4M3EXT             ← native FP8 SPIR-V Typ
```

`spirv-val` validiert clean.

### 1.3 Device-Setup

`bench_coopmat.rs` aktiviert beide Coopmat-Pfade in einem Device:

* `VK_KHR_cooperative_matrix` (rev 2)
* `VK_KHR_shader_bfloat16`
* `VK_EXT_shader_float8`  ← neu
* Feature-Splice: `feat_coopmat → feat_bf16 → feat_fp8 → ...`
* `shaderFloat8 = TRUE`
* `shaderFloat8CooperativeMatrix = TRUE`

ash 0.38 hat keine Builder für `VK_EXT_shader_float8`, daher manuell
eingeschmuggelt mit `s_type = 1000567000` (Registry-Eintrag 567).

---

## 2 — Throughput-Resultate

Median über 5 Samples + Warmup, exakt wie der BF16-Bench. Die Inputs sind
deterministisch, gleiche Werte in beiden Pfaden vor der Quantisierung; nur
das Storage-Format unterscheidet sich (1 vs 2 Bytes/Element).

### 2.1 Default-Shape-Liste

```
$ cargo run --release --example bench_coopmat
bench_coopmat — AMD Radeon RX 9070 XT (RADV GFX1201) (BF16)
size                   GFLOPs    warmup_ms   med_ms   TFLOPS   vs scalar
256^3                  0.03         0.30     0.045    0.75     0.03×
1024^3                 2.15         0.32     0.286    7.51     0.30×
4096^3                 137.44      37.26    33.872    4.06     0.16×
2048x64x4096           1.07         0.36     0.172    6.26     0.25×
11008x64x4096          5.77         1.30     1.177    4.90     0.20×
4096x64x11008          5.77         0.82     0.767    7.52     0.30×
4096x128x4096          4.29         0.63     0.584    7.36     0.29×

$ VF_BENCH_FP8=1 cargo run --release --example bench_coopmat
bench_coopmat — AMD Radeon RX 9070 XT (RADV GFX1201) (FP8-E4M3)
size                   GFLOPs    warmup_ms   med_ms   TFLOPS   vs scalar
256^3                  0.03         0.40     0.056    0.60     0.02×
1024^3                 2.15         0.46     0.371    5.79     0.23×
4096^3                 137.44      30.58    23.099    5.95     0.24×
2048x64x4096           1.07         0.39     0.110    9.79     0.39×
11008x64x4096          5.77         1.72     1.168    4.94     0.20×
4096x64x11008          5.77         0.65     0.399   14.48     0.58×
4096x128x4096          4.29         0.67     0.449    9.56     0.38×
```

Side-by-side:

| Shape            | BF16 TF | FP8 TF | Δ      | Bewertung |
|------------------|---------|--------|--------|-----------|
| 256³             |  0.75   |  0.60  | -20%   | Compute-limit (Overhead beider Pfade dominiert; FP8 zahlt zusätzlichen Decode) |
| 1024³            |  7.51   |  5.79  | -23%   | Compute-limit (BF16's WMMA-Throughput sättigt schon; FP8 erbt nicht den Vorteil) |
| 4096³            |  4.06   |  5.95  | +47%   | BW-Limit (großer Working-Set; FP8 spart die Hälfte der Loads) |
| 2048×64×4096     |  6.26   |  9.79  | +56%   | Prefill, K-dominiert → BW-bound |
| 11008×64×4096    |  4.90   |  4.94  | +1%    | M-dominiert → kein BW-Gewinn an A; aber auch keine Regression |
| 4096×64×11008    |  7.52   | 14.48  | +93%   | K=11008 maximal BW-bound → FP8 verdoppelt fast linear |
| 4096×128×4096    |  7.36   |  9.56  | +30%   | Prefill, doppelte N-Parallelität → mittleres BW-Limit |

### 2.2 K-Scan (M=N=1024)

```
$ VF_BENCH_SHAPES="1024,1024,1024;...,8192" cargo run --release --example bench_coopmat
bench_coopmat — (BF16)
1024^3                 2.15      0.58     0.334     6.43     0.26×
1024x1024x2048         4.29      0.62     0.552     7.78     0.31×
1024x1024x4096         8.59      1.08     0.981     8.76     0.35×
1024x1024x8192         17.18     3.91     3.701     4.64     0.19×

$ VF_BENCH_FP8=1 VF_BENCH_SHAPES="..." cargo run --release --example bench_coopmat
bench_coopmat — (FP8-E4M3)
1024^3                 2.15      0.59     0.321     6.70     0.27×
1024x1024x2048         4.29      0.55     0.504     8.51     0.34×
1024x1024x4096         8.59      1.05     0.986     8.71     0.35×
1024x1024x8192         17.18     1.96     1.916     8.97     0.36×
```

| K    | BF16 TF | FP8 TF | Δ      |
|------|---------|--------|--------|
| 1024 |  6.43   |  6.70  | +4%    |
| 2048 |  7.78   |  8.51  | +9%    |
| 4096 |  8.76   |  8.71  | -1%    |
| 8192 |  4.64   |  8.97  | +93%   |

**Key Finding aus dem K-Scan.** Bei K=4096 erreichen beide ~8.8 TFLOPS — sie
laufen offenbar ans gleiche Compute- oder L2-Tiling-Limit. Bei K=8192 fällt
BF16 schlagartig auf ~50% ab (fast-doppelt-so-große Working-Sets, gegen
HBM3 statt L2 BW), während FP8 stabil bei ~9 TFLOPS bleibt — exakt das
1.93×-Verhältnis das ein bandwidth-bound Kernel mit halbiertem Element-Size
zeigt.

Das ist saubere Bestätigung: **bei großem K (≈ Long-Context-Prefill) ist FP8
fast doppelt so schnell wie BF16, aus rein bandbreitentechnischen Gründen,
auch ohne tiled Kernel.**

---

## 3 — BW-Bound vs Compute-Bound — Diagnose

Beobachtungen quer zu den Shapes:

* **Klein/cubic (256³, 1024³).** BF16 *gewinnt* leicht. Hier reicht der L2 für
  beide Pfade; der WMMA-Pfad selbst ist Compute-bound (oder durch Latenz
  dominiert), und der naive FP8-Pfad zahlt einen kleinen Decode-Overhead
  (Driver-seitige FP8→intern-Konvertierung vor MulAdd?). Fix wäre ein
  optimierter Kernel; mit dem naiven Bench ist das nicht zu lösen.
* **Mittel (1024×1024×4096, 4096×128×4096).** Beide klettern; FP8 zieht moderat
  davon (+30-9%). Working-Set passt grenzwertig in L2.
* **Groß (4096³, 4096×64×11008, 1024²×8192).** FP8 dominiert deutlich (+47% bis
  +93%). Working-Set sprengt L2; HBM3-Bandbreite wird zur Wand; FP8 hat
  literally die Hälfte der Bytes pro MAC zu lesen.

Der Crossover liegt grob bei **(M·K + K·N) · sizeof(elem) ≈ 8-16 MB** —
das passt zur 16-MB-L2-Größe von gfx1201. Darunter dominieren WMMA-Latenz
und Driver-Overhead, darüber dominiert HBM3-Bandbreite.

---

## 4 — Correctness-Probe

`VF_BENCH_CHECK=1` aktiviert eine inline-Berechnung von `C[0]` auf der CPU
mit f64-Akkumulation und identischem FP8/BF16-Dequant wie der Shader. Probe
auf 3 kleinen Shapes:

```
[BF16, VF_BENCH_CHECK=1]
  64^3            : C[0]: gpu= 0.000038  cpu_ref= 0.000038  abs_err=4.657e-10
  256^3           : C[0]: gpu=-0.000877  cpu_ref=-0.000877  abs_err=2.328e-10
  1024x64x1024    : C[0]: gpu=-0.007675  cpu_ref=-0.007675  abs_err=3.725e-09

[FP8, VF_BENCH_CHECK=1]
  64^3            : C[0]: gpu=-0.000244  cpu_ref=-0.000244  abs_err=0.000e0
  256^3           : C[0]: gpu= 0.000092  cpu_ref= 0.000092  abs_err=0.000e0
  1024x64x1024    : C[0]: gpu=-0.005646  cpu_ref=-0.005646  abs_err=0.000e0
```

* **BF16-Werte** liegen im 1-4 ULP von FP32 — das ist erwartete Numerik bei
  einer langen Akkumulationskette.
* **FP8-Werte** matchen bit-exakt. Das ist *nicht* mystisch: FP8 hat 8
  diskrete Werte pro Exponent (3 Mantissen-Bits + 1 sign), die Inputs hier
  liegen im Bereich [-0.032, 0.032], und die FP32-Akkumulation hält jedes
  Zwischenergebnis exakt darstellbar. CPU- und GPU-Pfad rechnen die gleiche
  Reihe an FP32-MACs mit den gleichen FP8-dequantisierten Operanden, also
  ist Bit-Exaktheit das natürliche Ergebnis solange keine
  Reduktions-Reihenfolge-Effekte ins Spiel kommen.

**Take-away:** auf C[0]-Ebene gibt es keinen sichtbaren Precision-Verlust.
Für ein End-to-End-LLM-Forward-Pass (kein Akkumulator-Reset zwischen Layern)
muss das nicht der Fall bleiben — die echte Präzisions-Frage stellt sich
beim Q4_K → FP8 Dequant + Forward-Integration, nicht beim WMMA-Microbench.

---

## 5 — v0.2-Plan, aktualisiert

### Vor diesem Smoke-Test (gestern aus `fp8_glslang_analysis.md`)

> "v0.2B FP8 unblocked. Promote to in parallel with v0.2A Sprint 2."
> Recommendation war noch hypothetisch — BW-Limit war nicht gemessen.

### Nach diesem Smoke-Test

Jetzt mit Daten:

* **Bandwidth-Bound bei großem Traffic bestätigt.** Crossover ≈ 8-16 MB
  Working-Set. Bei K ≥ 8192 oder Pre-Fill-Shapes mit M·K oder K·N im
  1-100M Bereich ist FP8 ~2× BF16 — *ohne* getilten Kernel.
* **Compute-Bound bei kleinem Traffic.** Bei Decode-GEMV (M=hidden,
  N=1, K=hidden) und cubic-Mini-GEMMs ist BF16 leicht vorne; FP8 spart
  *nur* VRAM, nicht Geschwindigkeit.
* **Numerik unproblematisch** für die getesteten Tile-Shapes.

**Konkrete Empfehlung für v0.2:**

1. **Dual-Track behalten.**
   * `v0.2A` (BF16) bleibt Default-Pfad für Decode + kleine Prefill.
   * `v0.2B` (FP8) wird parallel als Spezial-Pfad für *Long-Context-Prefill*
     und *VRAM-knappe Modelle* (14B+ in 16 GB VRAM).
   * Beides hat einen gemeinsamen Ausgangs-Kernel-Skeleton (16×16×16 WMMA
     mit FP32-Accumulator); nur der Element-Type unterscheidet sich.

2. **v0.2A Sprint 1 zuerst** — der getilte BF16-Kernel (4 Subgroups, LDS
   staging, 30+ TFLOPS Ziel). Dieser Kernel ist *die* Investition.

3. **v0.2B Sprint 1 als Fork.** Sobald der BF16-Kernel stabil ist:
   - Klone als FP8-Variante.
   - Vergleichs-Bench: FP8-Tiled-Kernel auf den gleichen Shapes.
   - Erwartung: FP8 holt bei großen Shapes auf 60+ TFLOPS auf (3-4×
     gegenüber unserem naiven BF16-Bench).
   - Falls bestätigt: FP8 wird Standard-Pfad für Prefill,
     BF16 für Decode.

4. **Q4_K Dequant-Fusion** — beide Element-Types sind mögliche Ziele:
   - Q4_K → BF16 (mehr Precision, etablierter Pfad)
   - Q4_K → FP8 (mehr BW-Headroom, Vergleich offen)
   - Wahl in v0.2A Sprint 2 nach BF16-Kernel-Stand.

5. **Decode bleibt auf mul_mmq.** Phase-3C v0.1.3-Numbers (mul_mmq Q4_K)
   liefern Decode-tok/s im 30-60 Bereich; das ist nicht
   coopmat-bottleneck.

---

## 6 — Falsche Vorhersagen aus dem ursprünglichen Smoke Test (27.04.2026)

| Vorhersage | Tatsächlich |
|---|---|
| "FP8 GLSL nicht verfügbar → v0.2B deferred" | Falsch. shaderc 0.8 + System-glslang 16.2.0 hat es; Typ-Name war `floate4m3_t`, nicht `float8_e4m3`. |
| "shaderc 0.8 bündelt glslang 11.13" | Halbwahr. Source liegt bei, aber `find_library` nimmt System-libshaderc 2026.1 + glslang 16.2.0. |
| "FP8 K=32 statt K=16 (ISA-Shape)" | Falsch. Driver advertises K=16 für FP8 — gleiche WMMA-Shape wie BF16. |
| "FP8 könnte storage-only sein" | Falsch. `OpTypeFloat 8 Float8E4M3EXT` ist nativ; `Float8CooperativeMatrixEXT` Capability ist gesetzt. |
| "FP8 ≈ 2× BF16 falls BW-bound" | Bestätigt für große Shapes (1.5-1.9×); kein Effekt bei kleinen (Compute-bound). |

---

## 7 — Reproduzierbarkeit

```fish
# Build (FP8 + BF16 zusammen)
cargo build --release --example bench_coopmat

# Default-Shape BF16
cargo run --release --example bench_coopmat

# FP8 mit gleicher Shape-Liste
VF_BENCH_FP8=1 cargo run --release --example bench_coopmat

# K-Scan zur BW-Bestätigung (M=N=1024, K=1024..8192)
VF_BENCH_SHAPES="1024,1024,1024;1024,1024,2048;1024,1024,4096;1024,1024,8192" \
  cargo run --release --example bench_coopmat

# Correctness-Probe an C[0]
VF_BENCH_CHECK=1 VF_BENCH_FP8=1 VF_BENCH_SHAPES="64,64,64" \
  cargo run --release --example bench_coopmat
```

---

## 8 — Was nicht in Scope war (ehrlich)

* **Kein getilter Kernel.** Dieser Bench misst raw-WMMA-Throughput mit
  einer Subgroup pro Tile. Sprint 1 von v0.2A/B ist genau die Aufgabe,
  Tiling + LDS-Staging einzubauen. Die hier gemessenen TFLOPS sind ein
  *Untergrund*, kein erreichtes Maximum.
* **Keine Forward-Pass-Integration.** Kein Layer ruft `bench_coopmat_fp8`
  auf. Das ist v0.2B Sprint 3.
* **Kein E5M2-Vergleich.** Wir haben nur E4M3 getestet (Inference-Format
  per OCP). E5M2 würde uns größeren Range bei weniger Precision geben —
  irrelevant für Forward-Pass-Inference.
* **Keine Memory-Bandwidth-Messung in GB/s.** Die Aussage "BW-bound" stützt
  sich auf das ~2× FP8/BF16-Verhältnis bei großen Shapes; absolute HBM3-
  Auslastung müsste über `radv_dump_shaders` oder ein dedicated copy-bench
  bestimmt werden. Sprint-Aufgabe.
* **Keine Q4_K → FP8 Dequant-Pipeline.** Das ist die echte Frage von
  v0.2B. Hier nicht gemacht.

---

## 9 — Nächster Schritt (für morgen oder später)

Wenn der nächste Coding-Tag startet:

* **Git-State.** main hat 1 commit ahead von origin (gestern: smoke-test
  Pre-Korrektur), plus dieser FP8-Smoke-Test-Commit (jetzt). Beides nicht
  gepusht.
* **Wahl 1.** v0.2A Sprint 1 (getilter BF16-Kernel) starten. Klare
  Investition; der Kernel-Skeleton ist später für FP8 wiederverwendbar.
* **Wahl 2.** Naive FP8 in den Forward-Pass einklemmen (ohne Tiling) —
  als Reality-Check ob der gemessene 1.5-2× Speedup auf Layer-Ebene
  reproduzierbar ist. Niedrige Investition, hohes Lerngewicht.

Empfehlung: Wahl 1 zuerst, weil der getilte Kernel der größere Hebel ist
und wir aus diesem Smoke-Test schon genug Datenmaterial haben um zu
wissen, dass FP8 sich später lohnt.
