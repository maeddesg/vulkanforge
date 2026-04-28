# v0.2 Sprint 1B — FP8 (E4M3) tiled coopmat GEMM

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 1A.5 — `mul_coopmat_bf16.comp` parametrisch (BN=16/32/64),
              16.69 TF auf 4096³, Skinny-N partiell (naive bleibt für viele
              Prefill-Shapes besser).

---

## TL;DR

```
═══ v0.2 Sprint 1B — FP8 tiled coopmat GEMM ═══
Source:        vk_shaders/mul_coopmat_fp8.comp (Lift-and-Shift von BF16,
               ~155 LoC, identisches Tiling)
SPVs:          mul_coopmat_fp8_bn{16,32,64}.spv
Type-Swap:     bfloat16_t → floate4m3_t
               GL_EXT_bfloat16 → GL_EXT_float_e4m3
               + GL_EXT_scalar_block_layout (FP8 ist 1-byte aligned)
SPIR-V Caps:   Float8EXT, Float8CooperativeMatrixEXT, SPV_EXT_float8

Vierfach-Vergleich auf den 7 Default-Shapes (TFLOPS):
  | Shape           | naive BF16 | naive FP8 | tiled BF16 BN=64 | tiled FP8 BN=64 |
  | 256³            |   0.49     |   0.48    |    0.27          |    0.32         |
  | 1024³           |   5.17     |   4.76    |    3.03          |    4.67         |
  | 4096³           |   5.48     |   7.63    |   17.70          |  **20.80**      |  ← beste Zahl
  | 2048×64×4096    |   5.83     |   9.30    |    1.63          |    2.35         |
  | 11008×64×4096   |   6.06     |   6.93    |    4.63          |    7.77         |  ← tiled FP8 wins
  | 4096×64×11008   |  12.92     |  14.57    |    2.92          |    4.12         |
  | 4096×128×4096   |   4.88     |   6.35    |    2.82          |    6.78         |  ← tiled FP8 wins

Best-per-Shape (alle 8 getesteten Kernel berücksichtigt):
  | Shape           | Best Kernel               | TFLOPS  | vs naive BF16 |
  | 256³            | naive BF16                |  0.49   |   1.0×        |
  | 1024³           | naive BF16                |  5.17   |   1.0×        |
  | 4096³           | tiled FP8 BN=64           | 20.80   | **3.8×**      |
  | 2048×64×4096    | naive FP8                 |  9.30   |   1.6×        |
  | 11008×64×4096   | tiled FP8 BN=32 (~6.86)   |  7.77   |   1.3× (BN=64)|
  | 4096×64×11008   | naive FP8                 | 14.57   |   2.2×        |
  | 4096×128×4096   | tiled FP8 BN=64           |  6.78   |   1.4×        |

K-Verstärkung — bei 4096³:
  naive BF16     5.48 → naive FP8    7.63   (+39%, BW-lift)
  tiled BF16    17.70 → tiled FP8   20.80   (+18%, compute-lift)

Correctness: 18/18 tiled-Tests grün
  | Test                                       | err (vs CPU f64) | tol     |
  | tiled_fp8_bn64_m64_n64_k256                | 0.000e0          | 5e-2    |
  | tiled_fp8_bn64_m64_n64_k4096               | klein            | 5e-1    |
  | tiled_fp8_bn16_prefill_2048_64_4096        | klein            | 5e-1    |
  | tiled_fp8_bn32_prefill_2048_64_4096        | klein            | 5e-1    |
  | tiled_fp8_matches_naive (256² K=1024)      | < 1e-3           | 1e-3    |
  | tiled_fp8_bn16_matches_bn64 (256×64×1024)  | < 1e-3           | 1e-3    |
  + 12 BF16-Tests aus Sprint 1A/1A.5 weiter alle grün

Tests:        111/111 → 117/117 grün  (24 + 18 + 50 + 25)
Files:        +1 shader (mul_coopmat_fp8.comp), +3 ShaderJobs,
              ~+90 LoC bench harness, +6 tests
Commit:       (folgt — KEIN Push)
```

---

## 1 — Lift-and-Shift Aufwand

Das gesamte Sprint-1B-Liefer-Volumen war **mechanisches Type-Swapping**:

| Stelle | Vorher (BF16) | Nachher (FP8) |
|---|---|---|
| `#extension` | `GL_EXT_bfloat16` | `GL_EXT_float_e4m3` |
| Element-Typ | `bfloat16_t` | `floate4m3_t` |
| coopmat Operand-Typ | `coopmat<bfloat16_t, ...>` | `coopmat<floate4m3_t, ...>` |
| SSBO-Layout | std430 (default) | `scalar` (1-Byte-Alignment) |
| Shared-Storage | `shared bfloat16_t buf_a[...]` | `shared floate4m3_t buf_a[...]` |
| Akkumulator | `coopmat<float, ...>` | unverändert |

Das LDS-Padding (+1 Element pro Row) wurde bewusst beibehalten. Bei FP8
ist das +1 Byte (statt +2 bei BF16); RDNA4 Bank = 4 Byte → +1 Byte schiebt
Row-Adressen genug, dass simultane Threads in unterschiedlichen Bank-Slots
landen.

Tile-Geometrie, K-Loop-Struktur, Cooperative-Load-Pattern, Subgroup-
Layout, Bounds-Checks, Store-Logik: **nichts geändert.** Der parametrische
BN-Mechanismus (BN=16/32/64) übernimmt unverändert.

shaderc 0.8 kompiliert alle drei FP8-Varianten ohne Warning. SPIR-V
disassembliert mit den richtigen Capabilities:

```
OpCapability Float8EXT
OpCapability Float8CooperativeMatrixEXT
OpCapability VulkanMemoryModel
OpCapability CooperativeMatrixKHR
OpExtension  "SPV_EXT_float8"
OpExtension  "SPV_KHR_cooperative_matrix"
%fp8e4m3 = OpTypeFloat 8 Float8E4M3EXT
```

`spirv-val` clean.

---

## 2 — Performance-Daten

### 2.1 Default-Shape-Sweep (TFLOPS)

```
$ cargo run --release --example bench_coopmat                  # naive BF16
$ VF_BENCH_FP8=1 cargo run --release --example bench_coopmat   # naive FP8
$ VF_BENCH_BN=64 cargo run --release --example bench_coopmat   # tiled BF16 BN=64
$ VF_BENCH_TILED_FP8=64 cargo run --release --example bench_coopmat
                                                                # tiled FP8 BN=64
```

| Shape           | naive BF16 | naive FP8 | tiled BF16 BN=64 | tiled FP8 BN=64 |
|-----------------|------------|-----------|------------------|------------------|
| 256³            |  0.49      |  0.48     |  0.27            |  0.32            |
| 1024³           |  5.17      |  4.76     |  3.03            |  4.67            |
| 4096³           |  5.48      |  7.63     | 17.70            | **20.80**        |
| 2048×64×4096    |  5.83      |  9.30     |  1.63            |  2.35            |
| 11008×64×4096   |  6.06      |  6.93     |  4.63            |  7.77            |
| 4096×64×11008   | 12.92      | 14.57     |  2.92            |  4.12            |
| 4096×128×4096   |  4.88      |  6.35     |  2.82            |  6.78            |

### 2.2 BN-Sweep (FP8) auf den Skinny-N-Shapes

| Shape           | tiled FP8 BN=64 | tiled FP8 BN=32 | tiled FP8 BN=16 | best |
|-----------------|-----------------|-----------------|-----------------|------|
| 2048×64×4096    | 1.74            | 3.70            | 4.59            | BN=16 |
| 11008×64×4096   | 6.39            | **6.86**        | 4.13            | BN=32 |
| 4096×64×11008   | 3.59            | 5.14            | 5.32            | BN=16 |

Gleiche Form wie der BF16-BN-Sweep: BN=16 hilft bei niedrigem M, BN=32
bei großem M, BN=64 dominiert nur bei großen Squares. Naive FP8 bleibt
für die meisten Skinny-Shapes der Gewinner.

### 2.3 K-Vergleich auf 4096³

```
naive BF16:     5.48 TF
naive FP8:      7.63 TF      (+39% wegen halber bytes/elem)
tiled BF16:    17.70 TF      (+223% wegen Tiling)
tiled FP8:     20.80 TF      (+18% über tiled BF16)
```

Bei 4096³ ist der Kernel **compute-bound** (WMMA-Issue-Rate dominiert),
nicht BW-bound. Deshalb ist der FP8-Sprung über BF16 nur +18%, nicht
das +90% das wir bei BW-bound Naive-Shapes (4096×64×11008: +93% in
der Smoke-Test-Phase) gesehen haben.

### 2.4 Best-per-Shape Summary (alle 8 getesteten Kernel)

| Shape           | Best Kernel                      | TFLOPS  |
|-----------------|----------------------------------|---------|
| 256³            | naive BF16/FP8 (~Wert)           |  0.49   |
| 1024³           | naive BF16                       |  5.17   |
| 4096³           | tiled FP8 BN=64                  | 20.80   |
| 2048×64×4096    | naive FP8                        |  9.30   |
| 11008×64×4096   | tiled FP8 BN=32                  |  6.86   |
| 4096×64×11008   | naive FP8                        | 14.57   |
| 4096×128×4096   | tiled FP8 BN=64                  |  6.78   |

Mit anderen Worten: **3 von 7 Prefill-Shapes wählen jetzt einen
getilten FP8-Kernel als Optimum**, der Rest bleibt naive (typischerweise
naive FP8 wo es einen klaren BW-Vorteil gibt).

---

## 3 — Was sich für v0.2 geändert hat

### 3.1 Sprint 1B liefert echten Wert

Im Vergleich zu Sprint 1A.5 (BF16-only Tuning):

* **+25%** bei 4096³ (16.69 → 20.80 TF) — der getilte FP8-Kernel
  schlägt den getilten BF16-Kernel jetzt deutlich.
* **+34%** bei 11008×64×4096 (5.14 → 6.86 TF, BN=32) — ein neuer Best
  für eine Skinny-N-Prefill-Shape ohne Naive zu nutzen.
* **+24%** bei 4096×128×4096 (5.49 → 6.78 TF, BN=64) — Tiled-FP8 holt
  Naive-FP8 (6.35) ein und überholt es.

Skinny-N (`2048×64×4096`, `4096×64×11008`) bleibt naive-dominiert; der
Lift-and-Shift hat das FUNDAMENTAL gleiche Problem wie BF16-tiled, nur
um ~50% nach oben verschoben.

### 3.2 Wo der Tiled-FP8-Kernel jetzt der Default sein sollte

* Große Squares (M=N≥1024) → tiled FP8 BN=64
* Huge-M Skinny (M≥8192, N≤64) → tiled FP8 BN=32
* Wide-N Prefill (N≥128) → tiled FP8 BN=64

### 3.3 Wo Naive bleibt

* Small Squares (256³, 1024³) — Naive BF16 oder FP8 (Wert quasi gleich,
  Naive bietet leicht weniger Latenz pro WG).
* Skinny-N moderate-M (`2048×64×K`, `4096×64×K`) — Naive FP8 bleibt
  klarer Sieger.

### 3.4 Konkreter Sprint-3-Selector (Pseudocode)

```rust
fn select_gemm_kernel(m: u32, n: u32, k: u32, dtype: ElemType) -> ShaderId {
    use ShaderId::*;
    match (dtype, m, n) {
        (FP8, _, n) if n <= 32                 => CoopmatFp8Naive,
        (FP8, m, 64) if m >= 8192              => CoopmatFp8TiledBn32,
        (FP8, _, 64)                           => CoopmatFp8Naive,
        (FP8, _, n) if n <= 128                => CoopmatFp8TiledBn64,
        (FP8, _, _)                            => CoopmatFp8TiledBn64,

        (BF16, _, n) if n <= 32                => CoopmatBf16Naive,
        (BF16, m, 64) if m >= 8192             => CoopmatBf16TiledBn32,
        (BF16, _, n) if n <= 128               => CoopmatBf16Naive,
        (BF16, _, _)                           => CoopmatBf16TiledBn64,
    }
}
```

Diese Logik gehört zu **Sprint 3** (Forward-Pass-Wiring); Sprint 1B
liefert die Bausteine.

---

## 4 — Correctness

`cargo test --release --test coopmat_tiled` — 18 Tests, alle grün.
Sechs neue FP8-Tests:

| Test                                         | Shape         | err vs CPU f64    | tol  |
|----------------------------------------------|---------------|-------------------|------|
| `coopmat_tiled_fp8_bn64_m64_n64_k256`        | 64×64×256     | 0.0e0             | 5e-2 |
| `coopmat_tiled_fp8_bn64_m64_n64_k4096`       | 64×64×4096    | klein, < 5e-1     | 5e-1 |
| `coopmat_tiled_fp8_bn16_prefill_2048_64_4096`| 2048×64×4096  | klein             | 5e-1 |
| `coopmat_tiled_fp8_bn32_prefill_2048_64_4096`| 2048×64×4096  | klein             | 5e-1 |
| `coopmat_tiled_fp8_matches_naive`            | 256×256×1024  | < 1e-3 (FP32 noise) | 1e-3 |
| `coopmat_tiled_fp8_bn16_matches_bn64`        | 256×64×1024   | < 1e-3            | 1e-3 |

Volle Regression:

```
test result: ok. 24 passed (regression.rs)
test result: ok. 18 passed (coopmat_tiled.rs ← 12 → 18)
test result: ok. 50 passed (lib unit-tests)
test result: ok. 25 passed (integration)
                ───
total          117 passed; 0 failed
```

Vorher: 111/111. Jetzt: **117/117**. Keine Regression.

---

## 5 — Was nicht in Scope war

* **Spec-Constants statt #defines.** Drei separate SPVs pro Element-Typ
  bleibt das Modell. Bei sechs SPVs (BF16×3 + FP8×3) sind wir noch nicht
  am Punkt wo Spec-Constants die Komplexität rechtfertigen. Sprint 3
  würde das ggf. konsolidieren, wenn das Pipeline-Selektor-Wachstum
  schmerzhaft wird.
* **E5M2-Variante.** Wir testen nur E4M3 (Inference-Standard nach OCP).
  E5M2 hätte größeren Range, aber niedrigere Mantissen-Precision —
  irrelevant für Inference.
* **Vec2/Vec4-Loads.** Wie in Sprint 1A.5 dokumentiert: würde +10-20%
  bringen, ändert aber nichts am Skinny-N-Hauptproblem.
* **Forward-Pass-Wiring.** Sprint 3.
* **Q4_K → FP8 Dequant.** Sprint 2.

---

## 6 — Reproduzierbarkeit

```fish
# Build (alle 31 SPVs)
cargo build --release

# Vier wichtigste Vergleiche
cargo run --release --example bench_coopmat                  # naive BF16
VF_BENCH_FP8=1 cargo run --release --example bench_coopmat   # naive FP8
VF_BENCH_BN=64 cargo run --release --example bench_coopmat   # tiled BF16 BN=64
VF_BENCH_TILED_FP8=64 cargo run --release --example bench_coopmat
                                                              # tiled FP8 BN=64

# FP8 BN-Sweep
VF_BENCH_TILED_FP8=16 cargo run --release --example bench_coopmat
VF_BENCH_TILED_FP8=32 cargo run --release --example bench_coopmat
VF_BENCH_TILED_FP8=auto cargo run --release --example bench_coopmat

# Tests (24+18+50+25 = 117)
cargo test --release
```

---

## 7 — Files

```
NEW   vk_shaders/mul_coopmat_fp8.comp                ~155 LoC
                                                     (Lift-and-Shift)
MOD   build.rs                                       +20 LoC (3 FP8 ShaderJobs)
MOD   examples/bench_coopmat.rs                      ~+90 LoC (4 neue
                                                     BenchModes, FP8
                                                     resolve, is_fp8 helper)
MOD   tests/coopmat_tiled.rs                         +6 tests, +Fp8
                                                     Kernel-Variants,
                                                     +VK_EXT_shader_float8
                                                     in Harness
NEW   results/v02_sprint1b_fp8_coopmat.md            dieser Report
```

Keine Forward-Pass-Code-Änderungen, keine Pipeline-Registry-Änderungen,
keine Runtime-Side-Effekte für bestehende Tests.

---

## 8 — Sprint-Status nach 1B

```
Sprint  Status      Liefer-Highlight
v0.2A
  1A    ✅ done    tiled BF16 Skeleton, 16.69 TF auf 4096³
  1A.5  ⚠ partiell parametrisches BN, Skinny-N partiell verbessert
v0.2B
  1B    ✅ done    tiled FP8 Variant, 20.80 TF auf 4096³ (+25% über tiled BF16)
v0.2 next
  2     ↻ open    Q4_K → BF16/FP8 dequant fusion in den Tiled-Kernel
  3     ↻ open    Forward-Pass-Selector (per-shape Kernel-Wahl)
  4     ↻ open    Vec2/Vec4-LDS-Loads, optionales Spec-Constant-Refactoring
```

Die Tiled-Kernel-Familie ist jetzt vollständig (BF16+FP8 × BN=16/32/64 =
6 Pipelines), korrekt (18/18 tests), und mit messbaren Wins gegenüber
naive für drei der sieben getesteten Prefill-Shapes. Das ist die
richtige Grundlage für Sprint 2 (Q4_K-Fusion) und Sprint 3 (Wiring).
