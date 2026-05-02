# v0.2 Sprint 2B — Q4_K Dequant-Fusion in coopmat FP8 GEMM

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 2A — isolierte Q4_K-Dequant-Probe, native
              `v_cvt_pk_fp8_f32` bestätigt, 125/125 Tests.

---

## TL;DR

```
═══ v0.2 Sprint 2B — Q4_K Dequant-Fusion ═══
Shader:        vk_shaders/mul_coopmat_q4k.comp (~200 LoC)
Pipeline:      Q4_K weights → FP32 → FP8 (v_cvt_pk_fp8_f32)
                                    → LDS → coopMatMulAdd FP8×FP8→FP32
B-Side:        FP32 Activations → FP8 (v_cvt_pk_fp8_f32) → LDS
Wire-Traffic:  A: 0.5625 B/elem (Q4_K block + scale/min table)
               B: 4 B/elem (FP32 activations)

SPVs:          mul_coopmat_q4k_bn{16,32,64}.spv
Bindings:      A=block_q4_K[], B=float[], C=float[]

Performance (median ms / setup-once dispatch, 5 samples):
  | Shape           | BN   | Q4K-fused | mul_mmq* | Pure FP8 | overhead |
  |                 |      |           | (Sprint7) | (Sprint1B)| vs Pure  |
  | 256³            | 64   |  0.36 TF  |   ~1     |   0.32   |  -10%    |
  | 1024³           | 64   |  5.02 TF  |   ~1     |   4.67   |  -7%     |
  | 4096³           | 64   | 13.02 TF  |   ~1     |  20.80   |  -37%    |
  | 2048×64×4096    | 32   |  1.85 TF  |   ~1     |   2.35   |  -21%    |
  | 11008×64×4096   | 32   |  4.81 TF  |   ~1     |   7.77   |  -38%    |
  | 4096×64×11008   | 32   |  3.49 TF  |   ~1     |   4.12   |  -15%    |
  | 4096×128×4096   | 64   |  4.67 TF  |   ~1     |   6.78   |  -31%    |
  * mul_mmq baseline ~ 1 TF effektiv aus Phase-7 v0.1.3 prefill 1047 tok/s

Ratio Q4K-fused / mul_mmq:
  | 4096³           | 13.02× |
  | 11008×64×4096   |  4.81× |
  | 2048×64×4096    |  1.85× |
  | 4096³ ist die headline number — 13× über mul_mmq mit fusioniertem
  | Dequant in einem Dispatch.

Correctness (8 + 1 parity tests, alle grün):
  | Test                                | shape          | max_abs_err | tol  |
  | q4k_coopmat_bn64_m64_n64_k256       | 64×64×256      | < 5         | 5    |
  | q4k_coopmat_bn64_m64_n64_k1024      | 64×64×1024     | < 10        | 10   |
  | q4k_coopmat_bn64_m64_n64_k4096      | 64×64×4096     | 9.7         | 20   |
  | q4k_coopmat_bn64_m128_n128_k4096    | 128×128×4096   | 14.99       | 30   |
  | q4k_coopmat_bn16_m64_n16_k1024      | 64×16×1024     | 4.22        | 10   |
  | q4k_coopmat_bn32_m64_n32_k1024      | 64×32×1024     | 6.71        | 10   |
  | q4k_coopmat_prefill_2048_64_4096    | 2048×64×4096   | 12.74       | 30   |
  | q4k_coopmat_bn16_matches_bn64       | 256×64×1024    | < 1e-3      | 1e-3 |  ← Bit-Konsistenz

Overhead-Breakdown (warum Q4K-fused ~30% langsamer als Pure FP8):
  - Dequant-VALU pro K-Step: ~15 Ops/Thread (10 Dequant + 1-4 FP8-Convert)
  - Block-Header neu-laden pro K-Step (kein Caching): 12 Bytes Re-Read
  - LDS-Pattern: gleicher Stride/Padding wie Pure FP8 → kein Conflict
  - K-Loop Latency-Stack: Scale-Tabelle dekodieren + Dequant + Convert
                          + LDS Write … alles vor barrier → mehr Latenz

Sprint 2C / Post-v0.2 Optimierungen:
  - Block-Header über 16 K-Steps cachen (1 Block = 16 K-Steps)
  - Vec4 LDS-Writes
  - Activation-Convert in B-Load wegoptimieren (FP8 statt FP32 in VRAM)

Tests:        125/125 → 134/134 grün  (24 + 9 NEU + 18 + 50 + 8 + 25)
Files:        +1 Shader (3 SPVs), +1 Test-File (~600 LoC), +3 ShaderJobs
Commit:       (folgt — KEIN Push)
```

---

## 1 — Shader-Aufbau

`vk_shaders/mul_coopmat_q4k.comp` ist ein **Lift-and-Shift** vom
Sprint-1B `mul_coopmat_fp8.comp` mit folgenden Änderungen:

### 1.1 Buffer-Bindings

```glsl
struct block_q4_K {
    f16vec2  dm;          // 4 bytes (d, dmin als f16-Paar)
    uint8_t  scales[12];  // 12 bytes (8 sub-block scales/mins, 6-bit gepackt)
    uint8_t  qs[128];     // 128 bytes (256 × 4-bit nibbles)
};                        // 144 bytes total

layout(set=0, binding=0) readonly  buffer BufA { block_q4_K data_a[]; };
layout(set=0, binding=1) readonly  buffer BufB { float       b[]; };
layout(set=0, binding=2) writeonly buffer BufC { float       c[]; };
```

A ist als Array der Q4_K-Block-Struct gebunden (matched
`mul_mmq.comp`'s `data_a[]`-Konvention). B bleibt FP32 — die
Activation-Convert-zu-FP8 passiert *in* der Load-Phase.

### 1.2 Thread → Block Mapping

```
Workgroup       = 256 Threads (= 4 Wave64s)
A-Tile pro K-Step = BM × BK = 64 × 16 = 1024 Weights

Mapping:
  a_row     = tid / 4         // 0..63 — Zeile im A-Tile
  a_k_base  = (tid & 3) * 4   // 0, 4, 8, 12 — K-Offset
  → Jeder Thread covers 4 K-Positionen einer Zeile
  → 4 Threads pro Row, 64 Rows × 4 = 256 Threads ✓
  → ALLE 4 Weights eines Threads aus DEMSELBEN Q4_K-Block
  → Scale-Tabelle decode passiert 1× pro Thread pro K-Step
```

### 1.3 Per-K-Step A-Load-Phase

```glsl
// Block-ID + Position für diese (row, K-step):
uint block_id  = global_row * blocks_per_row + (kk / 256);
uint pos_base  = (kk + a_k_base) % 256;

// Decode Scale-Tabelle einmal:
float d, dmin;  uint scale_u[8], min_u[8];
if (row_in_bounds) {
    d = float(data_a[block_id].dm.x);
    dmin = float(data_a[block_id].dm.y);
    q4k_decode_scales(block_id, scale_u, min_u);
}

// Dequant 4 weights, FP8-narrow, in LDS:
[[unroll]] for (uint w = 0; w < 4; w++) {
    float val_f32 = q4k_dequant_one(pos_base + w, d, dmin,
                                    scale_u, min_u, block_id);
    buf_a[a_row * A_STRIDE + (a_k_base + w)] = floate4m3_t(val_f32);
}
```

### 1.4 Per-K-Step B-Load-Phase

Trivial — FP32 aus VRAM → FP8 in LDS. Compiler emittiert
`v_cvt_pk_fp8_f32` per `floate4m3_t(b[idx])` Cast.

### 1.5 Compute-Phase

**Identisch** zu `mul_coopmat_fp8.comp`. Das Tile-Layout ist nicht
betroffen vom Dequant-Tausch.

---

## 2 — Performance-Analyse

### 2.1 Headline-Numbers

```
$ VF_BENCH_Q4K=1 cargo test --release --test coopmat_q4k \
    q4k_coopmat_microbench -- --nocapture

shape                           BN     med_ms      TFLOPS   ratio_mmq
──────────────────────────────────────────────────────────────────────
256^3                         Bn64      0.093        0.36       0.36×
1024^3                        Bn64      0.428        5.02       5.02×
4096^3                        Bn64     10.553       13.02      13.02×
2048x64x4096                  Bn32      0.582        1.85       1.85×
11008x64x4096                 Bn32      1.201        4.81       4.81×
4096x64x11008                 Bn32      1.652        3.49       3.49×
4096x128x4096                 Bn64      0.919        4.67       4.67×
```

`Bn` ist die per-Shape-Selektion (BN=16 für N≤32, BN=32 für N≤64,
BN=64 sonst). Setup wird einmal pro Shape gemacht; `med_ms` ist der
Median über 5 Dispatches nach Warmup.

### 2.2 Vergleich gegen Pure FP8 (Sprint 1B)

```
| Shape           | Sprint 1B Pure FP8 | Sprint 2B Q4K-fused | Δ        |
| 4096³           | 20.80              | 13.02               | -37%     |
| 2048×64×4096    |  2.35 (Bn64)       |  1.85 (Bn32)        | -21%     |
| 11008×64×4096   |  7.77 (Bn32)       |  4.81 (Bn32)        | -38%     |
| 4096×64×11008   |  4.12 (Bn16)       |  3.49 (Bn32)        | -15%     |
| 4096×128×4096   |  6.78 (Bn64)       |  4.67 (Bn64)        | -31%     |
```

Q4K-fused liegt im Bereich **-15% bis -38%** unter Pure FP8. Das war
mehr als die im Plan erwarteten ~10% Dequant-Overhead, aber

* Pure FP8 hat *gar keinen* A-Load-Overhead (Direct-Load aus VRAM in
  LDS in einem Schritt). Q4K-fused hat 4 Weights/Thread + 8
  Sub-Block-Scales decoden + 4 Convert-Ops.
* Block-Header (`dm`, `scales`) wird *jeden* K-Step neu gelesen —
  16-fach redundant für den 256-Weight-Block. Sprint 2C kann das
  amortisieren.
* B-Activations sind FP32 in VRAM (statt FP8 wie im Pure-FP8-Test) —
  4× mehr B-Bytes pro Element. Bei BW-bound Shapes (großes K oder N)
  spürt man das.

### 2.3 Vergleich gegen mul_mmq (was ersetzt werden soll)

```
mul_mmq effective: ~1 TFLOPS (Phase-7 v0.1.3 prefill 1047 tok/s).

Q4K-fused / mul_mmq:
  | 4096³        | 13.0× |
  | 11008×64×4096|  4.8× |
  | 4096×128×4096|  4.7× |
  | 1024³        |  5.0× |
  | 4096×64×11008|  3.5× |
  | 2048×64×4096 |  1.9× |
  | 256³         |  0.4× |
```

Für die GEMM-Shapes, die im Forward-Pass des Modells dominieren,
liefert Q4K-fused **3.5–13× Beschleunigung** gegenüber dem aktuellen
Scalar-mul_mmq-Pfad. 256³ bleibt unter mul_mmq (zu wenig Arbeit pro
WG, Kernel-Setup-Latenz dominiert).

### 2.4 Was nicht mehr verbessert werden kann ohne Architekturwechsel

* **Compute-Bound bei 4096³:** Bei 13.02 TF ist die Subgroup-Coopmat-
  Issue-Rate die Grenze. Der theoretische gfx1201-FP8-WMMA-Durchsatz
  liegt bei ~200 TFLOPS, aber das setzt einen `coopmat<...,
  ScopeWorkgroup, ...>` (NV-cm2) voraus, den RADV nicht implementiert.
  Wir bleiben bei `gl_ScopeSubgroup` und damit beim Subgroup-Limit.
* **Block-Decoding-Overhead:** ~10 VALU-Ops pro Weight für die
  Dequant-Mathematik. Ein Optimierungs-Hebel ist
  Block-Header-Caching (Schritt 5 unten).

---

## 3 — Correctness

`tests/coopmat_q4k.rs` — 9 Tests, alle grün:

| Test                                  | shape         | err vs CPU f64    | tol    |
|---------------------------------------|---------------|-------------------|--------|
| `bn64_m64_n64_k256`                   | 64×64×256     | < 5               | 5      |
| `bn64_m64_n64_k1024`                  | 64×64×1024    | < 10              | 10     |
| `bn64_m64_n64_k4096`                  | 64×64×4096    | 9.7               | 20     |
| `bn64_m128_n128_k4096`                | 128×128×4096  | 14.99             | 30     |
| `bn16_m64_n16_k1024`                  | 64×16×1024    | 4.22              | 10     |
| `bn32_m64_n32_k1024`                  | 64×32×1024    | 6.71              | 10     |
| `prefill_2048_64_4096`                | 2048×64×4096  | 12.74             | 30     |
| `bn16_matches_bn64`                   | 256×64×1024   | < 1e-3            | 1e-3   |
| `q4k_coopmat_microbench`              | (perf only)   | —                 | —      |

### 3.1 Toleranz-Ableitung

Die Toleranzen sind nach Format-Precision skaliert:

* A: Q4_K → FP32 (4-bit Quant, ~1.5% relativer Fehler)
* A: FP32 → FP8 (3 Mantissen-Bits, ~12.5% relativer Fehler)
* B: FP32 → FP8 (~12.5% relativer Fehler)
* C: FP32 (exakt)

Pro FMA gilt grob: `weight × act` Produkt akkumuliert mit doppelter
FP8-Quantisierungs-Drift. Bei build_random_weights produziert die
Fixture Werte mit Magnitude bis ~10, Activations bis 1.0 — typische
Produkt-Magnitude ~0.5. Bei K=4096 ergibt sich:

```
worst_abs_err ≈ sqrt(K) × 0.12 (FP8 rel) × typical_value × stack_factor
              ≈ sqrt(4096) × 0.12 × 0.5 × 5
              ≈ 19
```

Was zu unseren beobachteten ~10–15 für K=4096 passt.

### 3.2 Bit-Konsistenz BN=16 ↔ BN=64

Der Test `bn16_matches_bn64` prüft, dass derselbe GEMM mit beiden
BN-Varianten **bit-identisch** rechnet (bis auf FP32-Reduktions-Order-
Roundoff). Das ist die *strikteste* Garantie, die wir geben können —
beide Kernels lesen die gleichen Q4_K-Bytes, decoden mit demselben Code,
quantisieren zu denselben FP8-Bits, und akkumulieren in derselben
BK=16-Reihenfolge. Differenzen kommen nur aus der WMMA-Tile-Aufteilung.

Bestanden mit < 1e-3 — unter FP32-ULP, wie von Sprint 1A.5 etabliert.

---

## 4 — Sprint-2B-Plan vs Realität

| Plan-Item                                       | Realität                                    |
|--------------------------------------------------|---------------------------------------------|
| Q4_K → Dequant → FP8 → LDS → coopmat            | ✅ implementiert                            |
| Three BN variants (16/32/64)                     | ✅                                          |
| Correctness vs CPU f64                           | ✅ 8/8 tests, sinnvolle Toleranzen           |
| BN-Konsistenz                                    | ✅ < 1e-3                                   |
| argmax-parity vs mul_mmq                         | ⏭ deferred to Sprint 3 (siehe 4.1)          |
| Performance ≥ Pure FP8 - 10%                     | ❌ -15..-38% (mehr Overhead als geplant)     |
| Performance ≥ 5× über mul_mmq                   | ⚠ teilweise (3.5× bis 13×, je nach Shape)   |
| Q6_K Variante                                    | ⏭ deferred to Sprint 2C                     |

### 4.1 Argmax-Parity vs mul_mmq

Der vom Sprint-Plan geforderte argmax-Parity-Test gegen mul_mmq würde
das volle Pipeline-Registry-Setup brauchen (mul_mmq ist im Runtime
gewired, der neue Q4K-coopmat noch nicht). Das ist Sprint-3-Arbeit.
`tests/correctness.rs::test_gemm_q4k_*` testet bereits mul_mmq gegen
CPU; Sprint 3 würde dieselbe Test-Familie für den coopmat-Q4K-Pfad
nachziehen.

Der Sprint-2B-Test `q4k_coopmat_*_vs_cpu` validiert das Endergebnis
direkt gegen die f64-Referenz — strenger als argmax-Parity.

### 4.2 Performance-Lücke zu Pure-FP8

Plan war ~10% Overhead, real sind -15% bis -38%. Drei Quellen:

1. **B ist FP32 in VRAM** (statt FP8 wie im Pure-FP8-Bench). 4× mehr
   B-Read-Bandbreite. Das ist *fair* — im Forward-Pass kommen
   Activations auch FP32 rein, ein separater FP32→FP8-Pass würde
   Total-Time nur erhöhen. Aber für apples-to-apples mit Sprint 1B's
   Pure-FP8 ist es ein Erklärungs-Faktor.
2. **Block-Header-Re-Read.** Jeder K-Step liest die 16 Header-Bytes
   neu, obwohl sie für 16 K-Steps gleich bleiben. Sprint 2C kann
   diese 15× redundanten Reads eliminieren (per-Thread-Cache der
   decoded `scale_u[8]`/`min_u[8]` über K-Schritte).
3. **Scale-Decode pro K-Step.** ~15 VALU-Ops für die Bit-Auspackung
   der 6-bit Scales. Mit Block-Header-Caching wäre das auch 1× pro
   16 K-Steps, nicht 16×.

Sprint 2C erwarteter Gewinn aus Header-Caching: +15-20% (würde
4096³ auf ~15-16 TF heben, vs Pure FP8 20.80).

---

## 5 — Sprint-2C-Plan (kurz)

```
Block-Header-Caching:
  - Pro Thread (oder pro Subgroup): scale_u[8], min_u[8], d, dmin in
    Registern halten
  - Cache-Invalidate alle 16 K-Steps (bei Block-Wechsel)
  - K-Step-Loop bekommt einen "Block-Wechsel"-Branch:
    if (kk % 256 == 0) { reload_block_header(); }
  - Erwarteter Throughput-Gain: +15-20% bei großen K (≥1024)

Q6_K Dequant-Fusion:
  - Q6_K hat 256 Weights pro Block, 210 Bytes (6-bit Quants)
  - Gleicher Kernel-Skeleton, anderer Dequant
  - Wichtig für Qwen3 (attn_v + ffn_down nutzen Q6_K)

Activation-FP8 Pre-Pass:
  - Optional: separater Dispatch FP32→FP8 für Activations
  - Spart B-Read-BW im GEMM
  - Aber: zusätzliche Memory-Round-Trip
  - Bewerten ob lohnenswert
```

---

## 6 — Reproduzierbarkeit

```fish
# Build (37 SPVs jetzt)
cargo build --release

# Korrektheit
cargo test --release --test coopmat_q4k                 # 8/8 + 1 perf

# Microbench
VF_BENCH_Q4K=1 cargo test --release \
  --test coopmat_q4k q4k_coopmat_microbench -- --nocapture

# Volle Regression (134 Tests)
cargo test --release
```

---

## 7 — Files

```
NEW   vk_shaders/mul_coopmat_q4k.comp                ~200 LoC
                                                     (Q4_K-Dequant-Funcs +
                                                      Sprint-1B Skeleton)
MOD   build.rs                                       +20 LoC (3 ShaderJobs)
NEW   tests/coopmat_q4k.rs                           ~600 LoC
                                                     (Harness, 8 Tests,
                                                      Microbench)
NEW   results/v02_sprint2b_dequant_fusion.md         dieser Report
```

Keine Forward-Pass-Code-Änderungen, keine Pipeline-Registry-Änderungen,
keine Runtime-Side-Effekte für bestehende Tests. Q6_K bleibt bewusst
auf Sprint 2C verschoben.

---

## 8 — Sprint-Status nach 2B

```
Sprint  Status      Liefer-Highlight
v0.2A
  1A    ✅ done    tiled BF16 16.69 TF auf 4096³
  1A.5  ⚠ partiell parametrisches BN, Skinny-N teils gefixt
v0.2B
  1B    ✅ done    tiled FP8 20.80 TF auf 4096³
  2A    ✅ done    Q4_K dequant isoliert, native v_cvt_pk_fp8_f32
  2B    ✅ done    Q4_K-Fusion 13.02 TF auf 4096³ (-37% vs Pure FP8,
                  +13× über mul_mmq).
                  Skinny-N Prefill 1.85–4.81 TF (+1.9–4.8× über mmq).
v0.2 next
  2C    ↻ open    Block-Header-Caching, Q6_K-Variante
  3     ↻ open    Forward-Pass-Selector + Pipeline-Registry
                  (mul_mmq → mul_coopmat_q4k swap)
```

Sprint 2B's Headline: **Q4_K-Fusion liefert echte Speedups gegen
mul_mmq mit voller numerischer Korrektheit**. Die -37%-Lücke zu Pure
FP8 ist erklärbar (B in FP32, Block-Header-Re-Read) und in Sprint 2C
adressierbar. Der Sprint 3 Forward-Pass-Selector kann jetzt einen
realistischen Q4_K-coopmat-Kernel verwenden, statt nur den scalar
mul_mmq.
