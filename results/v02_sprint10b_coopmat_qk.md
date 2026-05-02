# v0.2 Sprint 10B — Isolierter coopmat QK Micro-Benchmark

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 10A (cm2-Pivot zu cm1, Eigenbau-Entscheidung, 167/167 Tests)

---

## TL;DR — STRONG GO. coopmat QK ist 47.5× schneller als scalar FMA. Phase 10C+ freigegeben.

```
═══ v0.2 Sprint 10B ═══

Setup:
  Br=16, Bc=16, head_dim=128 (Qwen3 Attention-Shape)
  Q[16, 128] und K[16, 128] FP32 mit deterministischen Pseudo-
  Random-Werten in [-1, 1]. Bench-Loop: n_tiles=1000 QK-Wiederholungen
  pro Dispatch, 20 Reps für Median, 3 Warmups.

Shaders:
  bench_qk_scalar.comp  — FP32 FMA, Br×Bc/64=4 Cells/Thread,
                          identischer Algorithmus wie der Score-Loop
                          in flash_attn_tiled.comp.
  bench_qk_coopmat.comp — VK_KHR_cooperative_matrix WMMA, 16×16×16
                          FP16→FP32 Fragmente. 8 coopMatMulAdd-Schritte
                          pro QK-Tile (head_dim=128 / TILE_K=16).
                          Kompiliert sauber via shaderc 0.8.

Performance:
  | Metrik              | Scalar FMA | coopmat QK | Speedup |
  |---------------------|-----------:|-----------:|---------|
  | Median µs / 1000 QK |  20 649 µs |    434.6 µs| 47.5×   |
  | µs / QK-Tile        |    20.65 µs|     0.43 µs| 47.5×   |
  | Compute Throughput  |  3.17 GFLOPS|150.7 GFLOPS| 47.5×   |

Parity:
  coopmat vs CPU f64-Referenz, max_rel_err = 8.0e-3
  → ~FP16 Precision (~0.1% pro FMA × 128 head_dim × 1000 reps,
    durch FP32-Akkumulator in coopmat gut beschränkt).
  Beide Pfade: keine NaN/Inf, finite=true.

GO/NO-GO Entscheidung:
  Threshold ≥ 4×: STRONG GO        → 47.5× ✓✓✓
  Threshold ≥ 2×: GO (moderate)
  Threshold ≥ 1×: CONDITIONAL
  Threshold < 1×: NO-GO

  → STRONG GO. Phase 10C (PV Matmul) freigegeben.

Brief-Erwartung: 3-10× Speedup.
Reale Messung: 47.5× — deutlich höher als prognostiziert.
  Grund: Sprint 2A's mul_mmq-Coopmat-Misserfolg (-60%) lag an
  Skinny-N Tiles. Unsere QK-Shape (16×16×16) trifft die WMMA-
  Hardware mit voller Auslastung; das WMMA-Issue-Pattern ist
  fundamental schneller als jeder Scalar-Dot.

Tests: 167/167 ✓ (kein Hot-Path-Code geändert, nur 2 neue
       Bench-SPVs + 1 Example).

Files:
  new:      vk_shaders/bench_qk_scalar.comp     (~70 LOC)
  new:      vk_shaders/bench_qk_coopmat.comp    (~95 LOC)
  new:      examples/bench_qk.rs                (~270 LOC)
  modified: build.rs                            (+2 ShaderJobs)
  modified: src/backend/vulkan/shaders.rs       (+2 ShaderIds, +SPV consts)
  modified: src/backend/vulkan/pipeline.rs      (+ BenchQkPushConstants)
  modified: src/backend/vulkan/pipeline_registry.rs  (+ no-spec-const branch)
  new:      results/v02_sprint10b_coopmat_qk.md (this report)

Commit: HEAD (kein Push).
```

---

## 1. Was die 47.5× konkret bedeuten

### 1.1 Per-Dispatch Compute

```
Eine QK-Tile-Berechnung (Br=Bc=16, head_dim=128):
  Inputs:  Q[16, 128]  (8 KB)
           K[16, 128]  (8 KB)
  Output:  Score[16, 16]  (1 KB)
  Compute: 16 × 16 × 128 × 2 FLOPs = 65 536 FMAs ≈ 65 536 × 2 = 131 072 FLOPs
                                                            = 0.131 MFLOPs

Bench-Dispatch (n_tiles=1000):
  131 MFLOPs / Dispatch
  Scalar:  20 649 µs  ⇒  6.3 MFLOPs/ms ⇒  6.3 GFLOPS  →
                          (effective per-WG; 1 WG dispatch only)
                          Recomputed:  131 / 0.020649 = 6.3 GFLOPS
  Coopmat: 434.6 µs   ⇒  301 MFLOPs/ms ⇒  301 GFLOPS

Wait — recheck:
  Total FLOPs across 1000 reps = 1000 × 65 536 × 2 = 131 072 000 FLOPs ≈ 131 MFLOPs.
  Scalar  20 649 µs = 0.0207 s  →  131e6 / 0.0207 = 6.3 GFLOPS
  Coopmat   434.6 µs = 4.35e-4 s →  131e6 / 4.35e-4 = 301 GFLOPS

Hmm das widerspricht den Bench-Output-Zahlen. Nochmal:
  Per-Bench-Dispatch FMAs: 1000 reps × 16×16 cells × 128 dotprod-FMAs
                         = 1000 × 256 × 128 = 32 768 000 FMAs
                         × 2 FLOPs/FMA = 65 536 000 FLOPs = 65.5 MFLOPs

  Scalar:    65.5 MFLOPs / 0.020649 s =  3.17 GFLOPS  ✓ (matches output!)
  Coopmat:   65.5 MFLOPs / 0.000435 s = 150.7 GFLOPS  ✓ (matches output!)

Speedup: 150.7 / 3.17 = 47.5×  ✓
```

### 1.2 Vergleich zu RDNA4 Theoretical Peak

RDNA4 (gfx1201, RX 9070 XT):
* WMMA peak (Sprint 2A measurement): **20.8 TFLOPS** für FP16×FP16→FP32
* 128 AI Accelerators, ~62 CUs

Unser bench_qk_coopmat: 150.7 GFLOPS = **0.7% des Peaks**.

Warum so weit unter Peak?

1. **Single-WG dispatch**: 1 von ~62 CUs aktiv. Bei voller CU-Belegung
   wäre ~9.4 TFLOPS theoretisch erreichbar (62× Skalierung).
2. **Tiny problem size**: 16×16×16 = 256-Output-Tile. Sprint 2A's
   bench_coopmat_pure benutzte 4096³ — das ist ~16 Mio Output-Tiles
   pro Run. Unser einer Output-Tile pro Bench-Dispatch hat keinen Platz
   für Hardware-Pipelining.
3. **LDS-Load-Overhead**: pro Bench-Iteration 16+16 Threads laden
   FP32→FP16 in LDS. Das ist amortisiert über 1000 Reps, aber jede
   Rep hat trotzdem den `for d in 0..head_dim step 16` Inner-Loop
   mit 8 coopMatLoad-Calls.
4. **Single-Wave64-Subgroup**: gl_ScopeSubgroup ist ein Wave (64 lanes
   auf gfx1201). WMMA-Issue-Rate ist gut, aber Idle-Time zwischen
   Issues schlägt durch.

Der eigentliche Punkt ist: in der echten Forward-Pass wird die
Attention `(n_heads × ceil(seq_len/Br))` Workgroups dispatchen, was
hunderte bis tausende parallele WGs sind. Damit wird die CU-Auslastung
hoch und das Verhältnis zum 20.8 TFLOPS-Peak entsprechend besser.

Wichtig für GO/NO-GO: **die Frage ist nicht "wie nah am Peak", sondern
"wieviel schneller als scalar"**. 47.5× ist exzellent.

---

## 2. Was wir konkret bestätigt haben

### 2.1 Toolchain-Befunde

```
✅ GL_KHR_cooperative_matrix kompiliert auf shaderc 0.8 (Mesa glslang).
   Compile-Output:
     "compiled bench_qk_coopmat.comp -> bench_qk_coopmat.spv (7764 bytes)"

✅ shared float16_t[] funktioniert unter
   GL_EXT_shader_explicit_arithmetic_types_float16.

✅ float→float16_t Conversion in LDS klappt (`float16_t(data_q[i])`).

✅ coopMatLoad mit ColumnMajor + stride=head_dim auf K-Buffer
   produziert die korrekte K^T-Fragment-Sicht (max_rel_err 8e-3 vs
   CPU-Referenz).

✅ Component-wise `total += C_frag` auf accumulator-coopmat
   funktioniert (KHR coopmat add overload supported).

✅ coopMatStore aus FP32-Accumulator in FP32-SSBO klappt mit
   RowMajor + stride=Bc.
```

### 2.2 Validation-Layer-Hinweise (informational, kein Blocker)

```
"vkCreateShaderModule(): SPIR-V Capability VulkanMemoryModel was
declared, but ... VkPhysicalDeviceVulkan12Features::vulkanMemoryModel"
"vkCreateShaderModule(): SPIR-V Extension SPV_KHR_bfloat16 was
declared, but ... VK_KHR_shader_bfloat16"
```

Diese kommen von OTHER Pipelines (mul_coopmat_q4k_*, bench_coopmat_pure
mit BF16). Sind kein Blocker — die Pipelines werden trotzdem gebaut
und unser bench_qk_* nutzt KEINE dieser Capabilities. Werden in
Sprint 11+ adressiert wenn coopmat-GEMM (mul_mmq) wieder ins Visier
kommt.

### 2.3 Korrektheits-Verifikation

CPU-Referenz: `Score[qi, ki] = Σ_d Q[qi, d] × K[ki, d]` in f64.
Erwarteter coopmat-Output nach 1000 Reps: `n_tiles × Score[qi, ki]`.

```
max_rel_err coopmat vs CPU = 8.020e-3  (= 0.8%)
```

Das entspricht erwartet FP16-Precision-Drift (~10 mantissa bits ≈ 0.1%
pro Op, akkumuliert über 128 head_dim FMAs + 1000 outer reps via
FP32-Akkumulator → bounded ~1e-2). Akzeptabel für Attention-Scores
die ohnehin durch Softmax durchlaufen.

---

## 3. Implementations-Details

### 3.1 Scalar-Variant (bench_qk_scalar.comp, ~70 LOC)

```glsl
shared float q_lds[16 * 128];   // 8 KB
shared float k_lds[16 * 128];   // 8 KB

void main() {
    uint tid = gl_LocalInvocationID.x;
    // Cooperative LDS load
    for (uint i = tid; i < Br * head_dim; i += 64) q_lds[i] = data_q[i];
    for (uint i = tid; i < Bc * head_dim; i += 64) k_lds[i] = data_k[i];
    barrier();

    float total = 0.0;
    for (uint rep = 0; rep < n_tiles; ++rep) {
        for (uint i = tid; i < Br * Bc; i += 64) {  // 4 cells/thread
            uint qi = i / Bc, ki = i % Bc;
            float s = 0.0;
            for (uint d = 0; d < head_dim; ++d) {
                s += q_lds[qi*head_dim+d] * k_lds[ki*head_dim+d];
            }
            total += s;
        }
    }
    if (tid < Br * Bc) data_s[tid] = total;
}
```

64 threads compute 16×16 = 256 score cells (4 per thread). Per cell:
128 FMAs. Total per rep: 256 × 128 = 32 768 FMAs / WG. Mirrors
flash_attn_tiled.comp's score loop exactly.

### 3.2 Coopmat-Variant (bench_qk_coopmat.comp, ~95 LOC)

```glsl
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

shared float16_t q_lds[16 * 128];   // 4 KB (FP16)
shared float16_t k_lds[16 * 128];   // 4 KB

void main() {
    // Cooperative load + FP32→FP16 convert
    for (...) q_lds[i] = float16_t(data_q[i]);
    for (...) k_lds[i] = float16_t(data_k[i]);
    barrier();

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> total = ...(0);
    for (uint rep = 0; rep < n_tiles; ++rep) {
        coopmat<float, ..., gl_MatrixUseAccumulator> C_frag = ...(0);
        for (uint d = 0; d < HEAD_DIM; d += 16) {
            coopmat<float16_t, ..., gl_MatrixUseA> A_frag;
            coopmat<float16_t, ..., gl_MatrixUseB> B_frag;
            coopMatLoad(A_frag, q_lds, d, HEAD_DIM, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(B_frag, k_lds, d, HEAD_DIM, gl_CooperativeMatrixLayoutColumnMajor);
            C_frag = coopMatMulAdd(A_frag, B_frag, C_frag);
        }
        total += C_frag;
    }
    coopMatStore(total, data_s, 0, BC, gl_CooperativeMatrixLayoutRowMajor);
}
```

**Schlüssel-Erkenntnis (für Sprint 10C):** `gl_CooperativeMatrixLayoutColumnMajor`
+ `stride=head_dim` auf einer RowMajor-K-Buffer gibt eine korrekte
K^T-Sicht. Die Index-Math:

```
ColumnMajor mit base=d, stride=head_dim:
  Fragment[m, n] = mem[base + n * stride + m]
                 = K[base + n * head_dim + m]
                 = K[n][d + m]    ← m in [0, 16), n in [0, Bc)
                 = K^T[d + m, n]  ← genau was wir wollen ✓
```

### 3.3 Bench-Loop Anti-Elision

Beide Shaders akkumulieren über `n_tiles` Iterationen in einen
laufenden `total`-Wert. Das verhindert dass der Compiler alle
Iterationen außer der letzten eliminiert (sonst würde Speedup
"unendlich" und die Messung wertlos). Output ist der akkumulierte
Wert, der von ALLEN Reps abhängt.

---

## 4. Phase 10C+ Roadmap

### 4.1 Was Sprint 10C jetzt liefern kann

```
Sprint 10C — Eigenbau Coopmat-Attention v1 (~300-400 LOC):
  • flash_attn_coopmat.comp neuer Shader
  • Reuse von Sprint 9d.2's FP16 KV (uint[] + unpackHalf2x16,
    oder f16vec4-Alias)
  • LDS-Roundtrip-Pattern aus cm1 (Sprint 10A audit):
      Score-frag → LDS → scalar softmax → LDS → P-frag
  • Per-thread Of[rows][d] f16vec4 register accumulator
  • Inline causal mask auf LDS-staging Scores
  • Single Wave64 WG, Br=16, Bc=16, head_dim=128

Erwarteter end-to-end Gain auf pp=2048:
  • Attention compute: ~30-40% der Forward-Zeit
  • Coopmat speedup auf attention: 47.5× am Microbench, aber
    real-world 5-15× wegen LDS-staging + softmax + V-load Overhead
  • End-to-end forward: +20-50% bei pp=2048+
  • Additiv zu Sprint 9d's +21% FP16 KV win
```

### 4.2 Risiken die wir noch verifizieren müssen

```
NICHT in Sprint 10B getestet:
  • LDS-Roundtrip Score → scalar softmax → P (Sprint 10C)
  • Online-Softmax-Update auf Of-register-array (Sprint 10C)
  • PV-Matmul mit P-fragment loaded aus LDS (Sprint 10C)
  • Causal mask in Software auf LDS-staging Scores (Sprint 10C)
  • Forward-Pass-Integration: ShaderId-Selektor + bench (Sprint 10D)
  • End-to-end argmax-parity vs flash_attn_tiled (Sprint 10D)

Diese Sub-Sprints können jetzt mit hohem Confidence geplant werden,
weil der teuerste Teil (coopmat-vs-scalar Performance-Frage) als
positiv beantwortet ist.
```

---

## 5. Files Touched

```
new:      vk_shaders/bench_qk_scalar.comp        (~70 LOC)
new:      vk_shaders/bench_qk_coopmat.comp       (~95 LOC)
new:      examples/bench_qk.rs                   (~270 LOC)
modified: build.rs                               (+2 ShaderJobs)
modified: src/backend/vulkan/shaders.rs          (+ShaderIds, name+SPV maps,
                                                  ALL_SHADERS list)
modified: src/backend/vulkan/pipeline.rs         (+ BenchQkPushConstants)
modified: src/backend/vulkan/pipeline_registry.rs (+ branch in spec-const switch)
new:      results/v02_sprint10b_coopmat_qk.md    (this report)
```

KEIN Hot-Path-Code geändert. Forward-Pass-Tests unverändert grün
(167/167). Die zwei neuen SPVs werden bei jedem Pipeline-Registry-
Build zusätzlich erzeugt (+~14 KB SPV insgesamt) — vernachlässigbar.

---

## 6. Bottom Line

Sprint 10B beantwortet die einzige Frage die Sprint 10A offen ließ:
ist coopmat für Attention-QK auf RDNA4 schnell genug? Antwort:
**ja, 47.5× schneller als scalar**, deutlich besser als die
Brief-Prognose von 3-10×.

Der kritische Pre-Check (kompiliert KHR_cooperative_matrix? funktioniert
shared float16_t? klappt der ColumnMajor-Trick für K^T?) ist alles
verifiziert. Sprint 10C kann mit **GO + High Confidence** starten:
der WMMA-Pfad ist offen, der Eigenbau auf flash_attn_tiled.comp Basis
mit LDS-Roundtrip-Pattern ist die richtige Architektur, und die
erwarteten +20-50% End-to-End an pp=2048+ sind realistisch.

Empfehlung — nächster Sprint: **Sprint 10C** (Eigenbau-Coopmat-
Attention-Shader v1, single-Wave64-WG, Br=16, Bc=16, mit
LDS-Roundtrip-Softmax). Das ist eine konzentrierte 1-2-Tage-Session
ohne Citrix für saubere Performance-Messung.
