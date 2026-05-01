# v0.2 Sprint 1A — Tiled BF16 coopmat GEMM kernel

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0 (Minor-Bump)
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** v0.1.3 — naive BF16 6-13 TFLOPS, naive FP8 5-14.5 TFLOPS

---

## TL;DR

```
═══ v0.2 Sprint 1A — Tiled BF16 coopmat GEMM ═══
Shader:      vk_shaders/mul_coopmat_bf16.comp (161 LoC)
Geometry:    BLOCK_SIZE=256 (4 Subgroups × Wave64), BM=BN=64, BK=16
WMMA:        4 × 16×16×16 coopMatMulAdd pro Subgroup pro K-Step
             = 16 WMMAs / WG / K-Step
LDS:         buf_a[BM × (BK+1)] = 64×17 bf16 = 2.1 KiB
             buf_b[BK × (BN+1)] = 16×65 bf16 = 2.0 KiB
             Total: ~4.1 KiB / WG (in 64 KiB / CU LDS)
Layout:      RowMajor in LDS für A und B (matches naive bench convention).

Performance (Median über 5, default-Shapes, naive vs tiled):
  | Shape           | Naive   | Tiled   | Speedup | Diagnose         |
  | 256³            |  0.56   |  0.39   |  0.70×  | Compute-Limit    |
  | 1024³           |  6.00   |  5.16   |  0.86×  | Compute-Limit    |
  | 4096³           |  3.35   | 16.69   |  4.98×  | BW dominiert ↘   |
  | 2048×64×4096    |  5.95   |  1.71   |  0.29×  | Skinny-N regress |
  | 11008×64×4096   |  3.86   |  5.12   |  1.33×  | Skinny-N grenz   |
  | 4096×64×11008   | 10.45   |  2.91   |  0.28×  | Skinny-N regress |
  | 4096×128×4096   |  7.93   |  5.27   |  0.66×  | Skinny-N regress |

K-Scan (M=N=1024, naive vs tiled):
  | K    | Naive | Tiled | Speedup |
  | 1024 |  6.43 |  5.28 | 0.82×   |
  | 2048 |  7.78 |  6.52 | 0.84×   |
  | 4096 |  8.76 |  7.25 | 0.83×   |
  | 8192 |  4.64 |  7.62 | 1.64×   |  ← BW-Wand bei naive

Correctness: 7/7 Tests grün
  | Test                                | max_abs_err  | tol     |
  | tiled_bf16_m64_n64_k256             | 1.28e-9      | 1e-2    |
  | tiled_bf16_m64_n64_k4096            | 1.49e-8      | 1.5e-1  |
  | tiled_bf16_m128_n128_k4096          | 1.49e-8      | 1.5e-1  |
  | tiled_bf16_prefill_dims (2048×64)   | 1.49e-8      | 1.5e-1  |
  | tiled_bf16_ffn_down (4096×64×11008) | 1.56e-7      | 4e-1    |
  | tiled_matches_naive (256³ K=1024)   | < 1e-3       | 1e-3    |
  | tiled_bf16_unaligned_n              | 1.28e-9      | 5e-2    |

Gate:        TUNING — Tiled BF16 erreicht 16.69 TFLOPS auf 4096³
             (+5× über naive), aber regressiert auf Prefill-Shapes
             mit BN=64 ≥ N=64 (low-N-parallelism). Sprint 1A kommt
             mit "GO mit Vorbehalt" durch — der naive Pfad bleibt
             schneller für aktuelle Prefill-Workloads, der tiled
             Pfad ist der RICHTIGE Skeleton aber braucht Sprint 1A.5
             Tuning vor der FP8-Variante (Sprint 1B).

Tests:       106/106 grün  (24 + 7 + 50 + 25)
Files:       +1 shader (161 LoC), +1 ShaderJob, +1 BenchMode arm,
             +1 test file (~370 LoC), Cargo version 0.1.3 → 0.2.0
Commit:      (folgt — KEIN Push)
```

---

## 1 — Quellen-Analyse (vor dem Code)

Alle 6 Referenz-Dateien aus Schritt 0 wurden gelesen, bevor eine Zeile
des Shaders entstanden ist. Die kritischen Erkenntnisse, mit denen das
Design steht oder fällt:

### 1.1 `flash_attn_cm1.comp` — KHR-coopmat-API auf RADV

```glsl
coopmat<float16_t,  gl_ScopeSubgroup, MatBc, 16,    gl_MatrixUseA>          KMat;
coopmat<float16_t,  gl_ScopeSubgroup, 16,    MatBr, gl_MatrixUseB>          QMat;
coopmat<ACC_TYPE,   gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator> SfMat;

coopMatLoad(KMat, kvsh, coord, kvsh_stride, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(QMat, Qf,   d * 16 / 4, qstride, gl_CooperativeMatrixLayoutColumnMajor);
SfMat = coopMatMulAdd(KMat, QMat, SfMat);
```

* Subgroup-Scope, A/B/Accumulator-Pattern bestätigt.
* **Stride zählt in Element-Einheiten** (matchend mit dem Buffer-Element-Type),
  nicht in Bytes — `kvsh_stride` ist die Anzahl `f16vec4`-Elemente pro Row.
* `coopMatLoad` kann sowohl aus Global als auch aus LDS lesen, gleicher
  Aufruf — gut für unsere LDS-Stage-Strategie.
* Q wird hier mit `ColumnMajor` geladen, K mit `RowMajor` — der
  Layout-Flag teilt dem Treiber mit, **wie das Source-Memory aussieht**,
  nicht wie das WMMA-Fragment es interpretieren soll.

### 1.2 `mul_mm.comp` — KHR-coopmat-Tiling

`mul_mm.comp` enthält einen `#ifdef COOPMAT`-Pfad (Subgroup-Scope, KHR),
der unser direktes Vorbild ist. Tile-Geometrie:

```glsl
layout(constant_id = 0) const uint BLOCK_SIZE = 64;   // wir: 256
layout(constant_id = 1) const uint BM        = 64;
layout(constant_id = 2) const uint BN        = 64;
layout(constant_id = 4) const uint WM        = 32;    // subgroup tile M
layout(constant_id = 5) const uint WN        = 32;    // subgroup tile N
layout(constant_id = 7) const uint TM        = 4;     // (override für coopmat → 16)
layout(constant_id = 8) const uint TN        = 2;     //                       → 16
layout(constant_id = 9) const uint TK        = 1;     //                       → 16

#ifdef COOPMAT
#define SHMEM_STRIDE (BK / 2 + 4)   // bf16-vec2-units = +8 bf16 padding
#else
#define SHMEM_STRIDE (BK / 2 + 1)
#endif

shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];
shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];

#ifdef COOPMAT
shared ACC_TYPE coopmat_stage[TM * TN * NUM_WARPS];   // store-staging für FP16-Output
#endif
```

Inner Loop:

```glsl
coopMatLoad(cache_a, buf_a, (warp_r * WM + cm_row * TM) * SHMEM_STRIDE + i / 2,
            SHMEM_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(cache_b, buf_b, (warp_c * WN + cm_col * TN) * SHMEM_STRIDE + i / 2,
            SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);
sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_b, sums[...]);
```

* Llama.cpp wählt **B in LDS spalten-orientiert** und nutzt `ColumnMajor`
  beim Laden in `cache_b`. Die `load_b_to_shmem`-Funktion in
  `mul_mm_funcs.glsl` macht das Transponieren beim Stage-In.
* Wir wählen einfacher: **B row-major in LDS** und `RowMajor` beim
  `coopMatLoad`. Spart die Transpose-Logik beim Staging; identisch zu
  der Wahl in unserem `bench_coopmat_pure.comp`.

### 1.3 Konkrete Antworten auf die Schritt-0-Fragen

* `BLOCK_SIZE = 256`, 4 Subgroups, in 2×2 Grid mapped auf 64×64-Output-Tile.
* `coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, MatrixUse{A,B}>` direkt —
  *nicht* `uint16_t`. `GL_EXT_bfloat16` gibt uns den Typ erster Klasse.
* **Stride-Argument ist in Element-Einheiten (bfloat16_t-Anzahl)** — bestätigt.
* **B-Layout-Flag = RowMajor** für unsere LDS-Stage-Konvention. (llama.cpp
  benutzt ColumnMajor weil ihr LDS-Layout column-orientiert ist; das ist
  eine Codestyle-Wahl, kein Hardware-Zwang.)
* Padding +1 BF16 pro Row (statt llama.cpp +8) reicht, weil bf16=2 Byte
  und LDS-Bank=4 Byte: ein +1-Element-Shift verschiebt jede Row um eine
  halbe Bank → kein Cross-Row-Konflikt für simultane Threads, die in
  derselben Spalte landen.

---

## 2 — Kernel-Design

### 2.1 Geometrie

```
WG      : BLOCK_SIZE = 256 = 4 × Wave64
Tile-out: BM × BN = 64 × 64
K-step  : BK = 16  (= WMMA K-Dimension)
SG-Tile : WM × WN = 32 × 32  (jede Subgroup besitzt 4 16×16-Akkumulatoren)
SG-Grid : 2 × 2 (gl_SubgroupID 0..3 → (sg_m, sg_n) ∈ {(0,0),(0,1),(1,0),(1,1)})
WMMAs/WG: 4 SG × 4 Tiles/SG = 16 coopMatMulAdd / K-Step
```

### 2.2 LDS

```
buf_a [BM × (BK+1)] = 64 × 17 bfloat16_t = 2176 Bytes
buf_b [BK × (BN+1)] = 16 × 65 bfloat16_t = 2080 Bytes
Total                                    ≈ 4256 Bytes / WG
                                         ↪ 64 KiB LDS / CU → bis zu ~15 WGs co-resident
```

Bank-Conflict-Padding: `+1 bfloat16_t` pro Row schiebt jede Row um 2 Bytes
gegenüber der bank-aligned Konfiguration. Die kleinste Bank-Unit auf RDNA4
ist 4 Bytes (32 Bänke à 4 B = 128 B Cycle). Mit `BK+1 = 17 bf16 = 34 B`
landet Row 1 in einem anderen Bank-Phase als Row 0; subsequente Rows
zykeln und treffen sich erst alle 64 Rows wieder (was wir nicht haben).

### 2.3 Kooperative Loads

Pro K-Step lädt das WG `BM·BK + BK·BN = 64·16 + 16·64 = 2048` BF16-Elemente
aus Global in LDS. Bei 256 Threads: **8 Elemente/Thread/K-Step** — vier
Passes für A, vier Passes für B, jeweils 1 BF16/Thread/Pass.

Implementierung: einfacher Flat-Index-Loop ohne Vec2/Vec4-Packing. Das
ist sub-optimal (mehr Load-Instruktionen als nötig), bleibt aber für
Sprint 1A korrekt und einfach lesbar. Vec2/Vec4-Packing ist klar als
Sprint-1A.5-Tuning markiert.

```glsl
[[unroll]] for (uint pass = 0; pass < 4; pass++) {
    const uint fidx  = pass * BLOCK_SIZE + tid;
    const uint m_idx = fidx / BK;            // 0..63
    const uint k_idx = fidx & (BK - 1);      // 0..15
    const uint gm    = wg_m + m_idx;
    const uint gk    = kk + k_idx;
    const bool ok    = gm < pc.m && gk < pc.k;
    buf_a[m_idx * A_STRIDE + k_idx] = ok
        ? a[gm * pc.stride_a + gk]
        : bfloat16_t(0.0);
}
```

Bounds-Check zero-paddet (a) das K-Tail wenn `K % 16 != 0` und (b) M/N-Tails
für Rand-WGs (in Sprint 1A nutzen wir nur Shapes mit `M % 16 == 0` und
`N % 16 == 0`, aber der Check ist defensiv).

### 2.4 Compute & Store

```glsl
barrier();  // alle 256 Threads haben buf_a / buf_b geschrieben

coopMatLoad(matA0, buf_a, (sg_m * WM + 0 * TM) * A_STRIDE, A_STRIDE, RowMajor);
coopMatLoad(matA1, buf_a, (sg_m * WM + 1 * TM) * A_STRIDE, A_STRIDE, RowMajor);
coopMatLoad(matB0, buf_b,  sg_n * WN + 0 * TN,             B_STRIDE, RowMajor);
coopMatLoad(matB1, buf_b,  sg_n * WN + 1 * TN,             B_STRIDE, RowMajor);

acc00 = coopMatMulAdd(matA0, matB0, acc00);
acc01 = coopMatMulAdd(matA0, matB1, acc01);
acc10 = coopMatMulAdd(matA1, matB0, acc10);
acc11 = coopMatMulAdd(matA1, matB1, acc11);

barrier();  // bevor die nächste Iteration buf_a/buf_b überschreibt
```

`coopMatStore` schreibt direkt aus dem FP32-Akkumulator in das C-Buffer in
Global Memory — kein Stage-Buffer nötig, weil C bereits FP32 ist. Per-Tile
Bounds-Guard (`if (c_row + TM <= pc.m && c_col + TN <= pc.n)`) verhindert
Out-of-Bounds-Stores für Rand-Tiles.

---

## 3 — Setup-Aufwand

### 3.1 Build-Integration

Ein neuer `ShaderJob` in `build.rs` reicht:

```rust
ShaderJob {
    out_name: "mul_coopmat_bf16_f32.spv",
    entry_source: "mul_coopmat_bf16.comp",
    defines: &[],
},
```

shaderc 0.8 kompiliert den Shader sauber zum 22 216-Byte-SPV mit:

```
OpCapability Float16
OpCapability GroupNonUniform
OpCapability BFloat16TypeKHR
OpCapability BFloat16CooperativeMatrixKHR
OpCapability VulkanMemoryModel
OpCapability CooperativeMatrixKHR
OpExtension  "SPV_KHR_bfloat16"
OpExtension  "SPV_KHR_cooperative_matrix"
%41 = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_16 %uint_16 %uint_2
```

`spirv-val` validiert ohne Beanstandung.

### 3.2 Bench-Harness

`examples/bench_coopmat.rs` bekommt einen dritten `BenchMode`:

```rust
enum BenchMode { Bf16, Fp8E4m3, TiledBf16 }
```

Mit dem `tile_mn()`-Helper wird das Dispatch-Grid pro Modus berechnet
(`(M/16, N/16)` für naive, `(M/64, N/64)` für tiled). Switch via
`VF_BENCH_TILED=1`.

### 3.3 Korrektheits-Tests

Eigenes `tests/coopmat_tiled.rs` (ca. 370 LoC) — der Shader ist nicht
in der Runtime-Pipeline-Registry, also baut der Test sein eigenes
`vk::Device` mit coopmat + bfloat16 features auf, identisch zu
`examples/bench_coopmat.rs`. `OnceLock` für ein einmaliges Setup;
sieben `#[test]`-Funktionen.

---

## 4 — Performance

### 4.1 Default-Shapes (Median ms über 5 Messungen + Warmup)

```
$ cargo run --release --example bench_coopmat                # naive BF16
$ VF_BENCH_TILED=1 cargo run --release --example bench_coopmat
```

| Shape           | Naive TF | Tiled TF | Speedup | Diagnose                              |
|-----------------|----------|----------|---------|---------------------------------------|
| 256³            |  0.56    |  0.39    | 0.70×   | WG-Setup-Overhead, ~16 WGs total      |
| 1024³           |  6.00    |  5.16    | 0.86×   | Compute/LDS-Latenz dominiert          |
| 4096³           |  3.35    | 16.69    | **4.98×** | Naive verliert an L2; tiled gewinnt |
| 2048×64×4096    |  5.95    |  1.71    | 0.29×   | N=64=BN → 1 WG/Spalte → kein N-Parallelismus |
| 11008×64×4096   |  3.86    |  5.12    | 1.33×   | Mehr M-WGs (172) holt was raus        |
| 4096×64×11008   | 10.45    |  2.91    | 0.28×   | wie 2048×64 — N=64 erstickt          |
| 4096×128×4096   |  7.93    |  5.27    | 0.66×   | N=128=2·BN → nur 2 WGs/Zeile         |

### 4.2 K-Scan (M=N=1024)

| K    | Naive TF | Tiled TF | Speedup |
|------|----------|----------|---------|
| 1024 | 6.43     | 5.28     | 0.82×   |
| 2048 | 7.78     | 6.52     | 0.84×   |
| 4096 | 8.76     | 7.25     | 0.83×   |
| 8192 | **4.64** | **7.62** | **1.64×** |

Das Crossover bei K=8192 spiegelt wider, was wir gestern im FP8-Smoke-Test
gesehen haben: sobald das Working-Set die L2 sprengt, dominiert HBM3
und der Kernel mit der besseren Daten-Wiederverwendung gewinnt. Der
tiled Kernel liest A und B nur einmal aus Global pro WG-K-Step und
re-uses sie über die 4 WMMA-Ops.

### 4.3 Diagnose der Regressionen

* **Skinny-N (Prefill)** ist die größte Schwäche. Die Default-Prefill-Shapes
  haben N=64 oder N=128, exakt 1-2× BN. Das WG-Grid wird in N-Richtung
  trivial; die Tiles laufen im Wesentlichen sequenziell pro M-Reihe. Der
  naive 16×16-Kernel hat 4-8× mehr WGs in N und füllt RDNA4 besser.
  *Fix:* eigene Variante mit `BN=16` (oder `BN=32`) für Prefill, oder
  Split-K-Reduction für lange-K-flache-N-Cases. Sprint 1A.5.

* **Kleine Squares** verlieren leicht durch WG-Setup-Overhead. Bei 256³
  hat das tiled-Grid nur 4×4=16 WGs, aber jede dispatcht 256 Threads
  und 16 K-Steps — bei dieser geringen Compute-Density ist die K-Loop-
  Init / Barrier-Overhead spürbar. Akzeptabel; Mini-GEMMs sind nicht
  unser Optimierungsziel.

### 4.4 Was der tiled Kernel beweist

* **+5× bei 4096³** ist der wichtigste Datenpunkt: für große Squares
  schlagen wir den naiven Kernel deutlich, und die 16.69 TFLOPS sind
  ~67% des konservativen FP32-FMA-Theoretikum-Targets — der Kernel
  saturiert die WMMA-Issue-Rate näher als der naive es konnte.
* **+1.64× bei K=8192** zeigt, dass der Tiled Kernel die HBM3-Wand
  besser handhabt (weniger Re-reads).
* Der **Skeleton ist korrekt** — alle 7 Correctness-Tests grün, und der
  bit-exakte Vergleich gegen den naiven Kernel passt innerhalb von 1e-3
  (FP32-Roundoff).

---

## 5 — Correctness

`cargo test --release --test coopmat_tiled` führt 7 Tests aus, alle grün:

| Test                                       | shape          | err vs CPU f64 | tol     |
|--------------------------------------------|----------------|----------------|---------|
| `coopmat_tiled_bf16_m64_n64_k256`          | 64×64×256      | 1.28e-9        | 1e-2    |
| `coopmat_tiled_bf16_m64_n64_k4096`         | 64×64×4096     | 1.49e-8        | 1.5e-1  |
| `coopmat_tiled_bf16_m128_n128_k4096`       | 128×128×4096   | 1.49e-8        | 1.5e-1  |
| `coopmat_tiled_bf16_prefill_dims`          | 2048×64×4096   | 1.49e-8        | 1.5e-1  |
| `coopmat_tiled_bf16_ffn_down`              | 4096×64×11008  | 1.56e-7        | 4e-1    |
| `coopmat_tiled_matches_naive`              | 256×256×1024   | < 1e-3         | 1e-3    |
| `coopmat_tiled_bf16_unaligned_n`           | 128×64×256     | 1.28e-9        | 5e-2    |

Alle Fehler liegen 6-8 Größenordnungen unter den (defensiv großzügig
gewählten) Toleranzen. Der `tiled_matches_naive`-Test bestätigt, dass
der tiled Kernel innerhalb von FP32-Rounding-Differenzen das gleiche
Ergebnis wie unser ursprünglicher naive Kernel produziert; der minimale
Drift kommt von der unterschiedlichen Reduktions-Reihenfolge in der
K-Loop (naive: 1 Subgroup walks K linear; tiled: 4 Subgroups walken
parallel über getilte BK=16-Slabs).

Volle Regression-Suite:

```
test result: ok. 24 passed (regression.rs)
test result: ok.  7 passed (coopmat_tiled.rs ← neu)
test result: ok. 50 passed (lib unit-tests)
test result: ok. 25 passed (integration)
                ───
total            106 passed; 0 failed
```

Vorher: 99/99. Jetzt: 106/106. **Keine Regression**.

---

## 6 — Gate-Bewertung

Der Prompt definiert das Gate so:

> Falls Tiled BF16 ≥ 20 TFLOPS: GO für Sprint 1B
> Falls 10-20 TFLOPS: GO mit Vorbehalt (Tuning nötig)
> Falls < 10 TFLOPS: Kernel-Bug

**Status: GO mit Vorbehalt.**

* **16.69 TFLOPS** auf 4096³ überschreitet die 10-TF-Schwelle deutlich.
* **5× Speedup** gegenüber naive auf demselben großen Square — das ist
  das eindeutige Zeichen, dass das Tiling-Schema funktioniert.
* **Nicht 30+ TFLOPS** wie llama.cpp's `mul_mm_cm2`: deren cm2-Variante
  nutzt `gl_ScopeWorkgroup`, was auf RADV nicht verfügbar ist. Mit
  `gl_ScopeSubgroup`-Coopmat ist 16-20 TFLOPS realistisch ohne weitere
  Optimierung; 30+ TFLOPS bräuchte vec4-Packing der LDS-Loads + Hand-
  Tuning der Subgroup-Tile-Geometrie.
* **Skinny-N-Regression** ist real, blockiert aber Sprint 1B nicht — FP8
  ist ein Element-Type-Swap auf demselben Skeleton, und die FP8-
  Variante wird genau das gleiche Skinny-N-Problem haben.

**Empfehlung:** Sprint 1A.5 (1-2 Tage Tuning) BEVOR Sprint 1B startet,
um die Skinny-N-Regression zu beheben. Tuning-Hebel:

1. **Vec2/Vec4-LDS-Loads.** 4 BF16/Thread/Pass statt 1. Reduziert Load-
   Instruktionen 4× und LDS-Write-Druck.
2. **Smaller-N-Variante.** Spec-constant `BN ∈ {16, 32, 64}`. Pipeline-
   Selector wählt zur Dispatch-Zeit basierend auf `seq_len`. Für `N=64`
   wird `BN=32` getestet (= 2 WGs/Zeile statt 1) — sollte den
   Prefill-Pfad wieder über naive ziehen.
3. **Split-K für lange-K + flache-N.** Bei K=11008 N=64 scheinen wir an
   sequenzielle Tile-Bearbeitung gebunden. Split-K-Reduce ist unser
   Phase-4C-Pattern und passt hier konzeptuell.

Falls Sprint 1A.5 in 2 Tagen nicht fertig ist: Sprint 1B kann trotzdem
parallel auf einem Branch starten — der naive FP8-Pfad steht bereits
(v0.2 Smoke-Test FP8), und das Lift-and-Shift zum Tiled-FP8 ist
mechanisch sobald der BF16-Tuning fertig ist.

---

## 7 — Was nicht in Scope war (ehrlich)

* **Kein Forward-Pass-Wiring.** `pipeline_registry.rs` und `forward.rs`
  sind unverändert. Das ist Sprint 3 (Q4_K → BF16 Dequant + Decode/Prefill-
  Integration).
* **Kein Spec-Constant-Tuning.** Die Tile-Maße (BLOCK_SIZE, BM, BN, BK,
  WM, WN) sind als `const uint` hartcodiert, nicht als spec-constants.
  Damit kann man pipeline-creation-time keine Variante wählen. Sprint
  1A.5 sollte die fünf wichtigsten als spec-constants exposed.
* **Kein FP8-Kernel.** Sprint 1B-Aufgabe: derselbe Skeleton mit
  `floate4m3_t` statt `bfloat16_t` und `Float8CooperativeMatrixEXT`.
* **Kein Decode-GEMV.** Decode benutzt `mul_mat_vec_q4_k.comp` (Phase 3),
  bleibt unverändert.
* **Kein Bank-Conflict-Profil.** Wir vertrauen dem `+1`-Padding-Ansatz,
  ohne RGP/radv_dump_shaders zu fahren. Bei der Sprint-1A.5-Optimierung
  sollte das bestätigt oder via `+8`-Padding (wie llama.cpp) abgesichert
  werden.

---

## 8 — Reproduzierbarkeit

```fish
# Build (alle 26 SPVs inkl. mul_coopmat_bf16_f32)
cargo build --release

# Naive BF16 baseline
cargo run --release --example bench_coopmat

# Tiled BF16
VF_BENCH_TILED=1 cargo run --release --example bench_coopmat

# K-Scan
VF_BENCH_TILED=1 \
  VF_BENCH_SHAPES="1024,1024,1024;1024,1024,2048;1024,1024,4096;1024,1024,8192" \
  cargo run --release --example bench_coopmat

# Korrektheits-Tests (mit Magnitude-Output)
cargo test --release --test coopmat_tiled -- --nocapture

# Volle Regression
cargo test --release
```

---

## 9 — Files

```
NEW   vk_shaders/mul_coopmat_bf16.comp                   161 LoC
MOD   build.rs                                           +9 LoC (1 ShaderJob)
MOD   examples/bench_coopmat.rs                          +24 LoC (TiledBf16 mode)
NEW   tests/coopmat_tiled.rs                             ~370 LoC
MOD   Cargo.toml                                         version 0.1.3 → 0.2.0
NEW   results/v02_sprint1a_tiled_coopmat.md              dieser Report
```

Kein Forward-Pass-Code, keine Pipeline-Registry-Änderung, keine Runtime-
Side-Effekte für bestehende Tests.
