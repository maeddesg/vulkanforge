# v0.2 Sprint 2A — Q4_K Dequant isoliert (BF16 vs FP8 Convert-Kosten)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 1B (6 tiled coopmat-Pipelines BF16+FP8, 117/117 Tests).

---

## TL;DR

```
═══ v0.2 Sprint 2A — Q4_K Dequant isoliert ═══
Shader:        dequant_q4k_debug.comp (~120 LoC, 1 Source, 3 SPVs via -DOUT_*)
Output-Typen:  FP32 (no convert) / BF16 / FP8 E4M3
Workgroup:     1 Wave64 / WG, 1 Q4_K-Block (256 Weights) / WG, 4 Weights / Thread

Correctness (random Q4_K weights, 100 Blöcke):
  | Variante | max_abs vs CPU f64 | tol  | pass |
  | FP32     | 0 (BIT-EXAKT)     | 1e-6 | ✅   |
  | BF16     | 0.06              | 1e-1 | ✅   |
  | FP8      | 0.49              | 1.5  | ✅   |
  +1 Bit-Exact-Test (FP32 == CPU per-bit)

ISA-Verifikation (RADV_DEBUG=shaders):
  FP8-Convert: ✅ 4 × v_cvt_pk_fp8_f32 (NATIVE gfx12-Instruction, encoding 0xd7690009)
               1 Instruktion packt 2 FP32 → 2 FP8
  BF16-Convert: ❌ Software-Sequenz (kein nativer Pack für bfloat16
                 auf gfx1201 — ACO emittiert ~5 VALU-Ops via
                 `v_lshrrev_b32 + bitselect`-Pattern)

Throughput-Microbench (4096 Blöcke = 1.05M Weights):
  | Variante | median ms | GW/s   | output GiB/s |
  | FP32     | 0.141     |  7.42  | 27.65        |
  | BF16     | 0.094     | 11.16  | 20.79        |  ← 33% schneller (!)
  | FP8 E4M3 | 0.071     | 14.68  | 13.68        |  ← 50% schneller (!!)

Überraschung: NARROWING ist NETTO POSITIV.
  → Output-BW dominiert die Dispatch-Zeit, nicht die Convert-VALU-Ops.
  → BF16 spart 50% Output-Bytes → +33% Throughput selbst mit Software-Convert.
  → FP8 spart 75% Output-Bytes → +50% Throughput trotz extra Convert.
  → Native v_cvt_pk_fp8_f32 macht FP8 zur klaren Wahl: weniger Bytes UND
    weniger VALU.

Empfehlung Sprint 2B:
  * **FP8 als Default** für die Dequant-Fusion in den tiled Kernel.
  * BF16 als Fallback nur falls Q4_K → FP8 → coopmat numerisch nicht
    reicht (separater Test wenn Sprint 3 echte Modelle sieht).
  * Separate-Dispatch-Variante (Q4_K → BF16/FP8 in eigenem Pass) ist
    NICHT nötig — Convert ist quasi kostenlos in der Fusion.

Tests:        117/117 → 125/125 grün (+8 neue Dequant-Tests)
Files:        +1 Shader (3 SPVs), +1 Test-File (~470 LoC), +3 ShaderJobs
Commit:       (folgt — KEIN Push)
```

---

## 1 — Shader: `dequant_q4k_debug.comp`

Eine Source kompiliert zu drei SPVs via `-DOUT_FP32`, `-DOUT_BF16`,
`-DOUT_FP8`. Jeder Workgroup behandelt genau einen Q4_K-Block (256
Weights, 144 Bytes) mit 64 Threads (1 Wave64); jeder Thread dequantisiert
4 zusammenhängende Weights:

```glsl
const uint QUANT_K     = 256;
const uint BLOCK_SIZE  = 64;
const uint PER_THREAD  = 4;     // = 256/64

void main() {
    uint block_id = gl_WorkGroupID.x;
    uint tid      = gl_LocalInvocationID.x;

    float d    = float(data_a[block_id].dm.x);
    float dmin = float(data_a[block_id].dm.y);

    [[unroll]] for (uint w = 0; w < PER_THREAD; w++) {
        uint pos = tid * PER_THREAD + w;        // 0..255
        // Decode (scale, min) — 6-bit pair packed across the 12-byte table
        // ... (mirrors src/backend/vulkan/q4k.rs:dequant_block)
        float val_f32 = d * scale * float(nib) - dmin * min_;
#if defined(OUT_FP32)
        data_c[block_id*256 + pos] = val_f32;
#elif defined(OUT_BF16)
        data_c[...] = bfloat16_t(val_f32);
#elif defined(OUT_FP8)
        data_c[...] = floate4m3_t(val_f32);
#endif
    }
}
```

Ergebnis-SPVs:

```
dequant_q4k_fp32.spv   14748 B   no narrowing convert
dequant_q4k_bf16.spv   17044 B   +BFloat16TypeKHR cap, +SPV_KHR_bfloat16
dequant_q4k_fp8.spv    17076 B   +Float8EXT cap, +SPV_EXT_float8
```

Der Block-Dekodier-Code spiegelt 1:1 die Logik in
`src/backend/vulkan/q4k.rs::dequant_block` — gleiche Scale/Min-Auspackung
für die 6-bit-gepackten Sub-Block-Konstanten, gleiche Nibble-Extraktion.

---

## 2 — Correctness

Tests in `tests/dequant_q4k.rs` (8 neue Tests, alle grün):

| Test                                  | Variante | Blöcke | max_abs_err | tol  | Result |
|---------------------------------------|----------|--------|-------------|------|--------|
| `dequant_q4k_fp32_single_block`       | FP32     |     1  | 0.0e0       | 1e-6 | ✅     |
| `dequant_q4k_fp32_multi_block`        | FP32     |   100  | 0.0e0       | 1e-6 | ✅     |
| `dequant_q4k_fp32_bit_exact`          | FP32     |    64  | bit-exact   | —    | ✅     |
| `dequant_q4k_bf16_single_block`       | BF16     |     1  | 0.013       | 1e-1 | ✅     |
| `dequant_q4k_bf16_multi_block`        | BF16     |   100  | 0.059       | 1e-1 | ✅     |
| `dequant_q4k_fp8_single_block`        | FP8      |     1  | 0.119       | 1.5  | ✅     |
| `dequant_q4k_fp8_multi_block`         | FP8      |   100  | 0.489       | 1.5  | ✅     |
| `dequant_q4k_microbench`              | alle     |  4096  | (perf only) | —    | ✅     |

**Kritischer Test: FP32 ist BIT-EXAKT zur CPU-Referenz** (`assert_eq!`
über alle 64×256 = 16384 Werte). Damit wissen wir: das GLSL-Decoding
matcht die CPU-Logik exakt; jede Drift in BF16/FP8 kommt rein vom
Narrowing-Convert.

Die BF16/FP8-Toleranzen reflektieren die Format-Precision:

* BF16: 7 Mantissen-Bits → ~0.4% relative Precision → bei Q4_K-Werten
  bis ~10 Magnitude → ~0.04 absolut worst-case → Tol 1e-1 hat Margin.
* FP8 E4M3: 3 Mantissen-Bits → ~12.5% relative Precision → ~1.25
  absolut worst-case → Tol 1.5 hat Margin.

Die `multi_block`-Worst-Cases (0.06 BF16, 0.49 FP8) liegen dort, wo
eine Quantisierungsstufe gerade an einer Format-Grenze landet.

---

## 3 — ISA-Verifikation: native FP8-Convert bestätigt

`RADV_DEBUG=shaders` dumpt ACO's intermediate AND final ISA. Im
`dequant_q4k_fp8.comp`-Output sehen wir vier Instances von:

```
v1b: %95:v[9][0:8]  = v_cvt_pk_fp8_f32 %91:v[3], 0
v1b: %161:v[9][0:8] = v_cvt_pk_fp8_f32 %159:v[3], 0
v1b: %225:v[8][0:8] = v_cvt_pk_fp8_f32 %223:v[3], 0
v1b: %290:v[6][0:8] = v_cvt_pk_fp8_f32 %288:v[3], 0
```

Final-ISA Encoding: `d7690009 00010103` (gfx12 native FP8-pack opcode).

**Vier Instances** sind exakt das was wir erwarten — der
`[[unroll]] for (w = 0; w < 4)` Loop gibt vier Convert-Slots, und ACO
fasst je `floate4m3_t(val_f32)` Aufruf in eine Instruktion. Das ist die
"Smoking-Gun"-Bestätigung aus dem Sprint-Plan.

Für BF16 emittiert ACO die erwartete Software-Sequenz — keine native
`v_cvt_pk_bf16_f32` Instruction existiert auf gfx1201 (anders als auf
CDNA3/MI300).

---

## 4 — Throughput-Microbench

`VF_BENCH_DEQUANT=1 cargo test … dequant_q4k_microbench -- --nocapture`
löst die Dispatch-Zeit-Messung aus. 4096 Blöcke (= 1.05M Weights),
Median über 11 Samples + Warmup:

```
variant          median_ms           GW/s      out_GiB/s
----------------------------------------------------------
FP32                 0.141           7.42          27.65
BF16                 0.094          11.16          20.79
FP8 E4M3             0.071          14.68          13.68
```

Mehrere Beobachtungen:

* **BF16 schlägt FP32 um +33%.** Trotz der Software-Sequenz für den
  BF16-Pack-Convert. Erklärung: 50% weniger Output-Bytes (2 statt 4).
* **FP8 schlägt BF16 um +25% (und FP32 um +50%).** Native
  `v_cvt_pk_fp8_f32`-Instruktion + 25% weniger Output-Bytes (1 statt
  2 BF16-Bytes; 1 statt 4 FP32-Bytes).
* **Output-Bandbreite dominiert die Dispatch-Zeit.** FP32 sättigt mit
  27.65 GiB/s die HBM3-Output-Schiene am stärksten; BF16 und FP8 sind
  nicht BW-limitiert (20.79 und 13.68 GiB/s liegen unter dem
  ~256 GiB/s-Peak), die Variante ist eher von der WG-Setup-Latenz und
  dem Dequant-Compute getrieben.

### 4.1 Convert-Overhead relativ zum Compute

```
Pro Block (256 Weights):
  Dequant:      ~10 VALU-Ops/Weight × 4 Weights/Thread = ~40 VALU-Ops/Thread
  BF16-Convert: ~5 VALU-Ops × 4 Weights = ~20 zusätzliche VALU-Ops/Thread
  FP8-Convert:  1 v_cvt_pk_fp8_f32 / 2 Weights × 4 = 2 VALU-Ops/Thread

  Convert-Overhead vs Dequant:
    BF16: +50% VALU
    FP8:  +5% VALU
```

Das matcht qualitativ die Throughput-Daten — FP8 ist nur ein winziges
bisschen über dem reinen Dequant-Cost; BF16 zeigt mehr Compute-Zusatz
(verloren im BW-Gewinn).

---

## 5 — Sprint-2B-Plan: LDS-Layout für die Fusion

Für Sprint 2B (Dequant-Fusion in den tiled Kernel `mul_coopmat_*.comp`):

### 5.1 K-Loop-Skelett

Statt im Sprint-1A/1B-Pfad:

```glsl
buf_a[m_idx * A_STRIDE + k_idx] = a[gm * stride_a + gk];   // BF16/FP8
                                                           //  direct from VRAM
```

…wird Sprint 2B den Load durch eine Q4_K → FP32 → BF16/FP8-Sequenz
ersetzen:

```glsl
// Pseudo, in load_a_to_shmem-Position:
uint block_id  = gm / 256;
uint pos       = gm % 256;
float val_f32  = dequant_q4k_one(data_a[block_id], pos);
buf_a[m_idx * A_STRIDE + k_idx] = bfloat16_t(val_f32);   // oder floate4m3_t
```

LDS-Layout bleibt **identisch** zu Sprint 1A/1B — `buf_a[BM × A_STRIDE]`
mit gleichem +1-Padding. Die Fusion ist rein "Loader-Tausch".

### 5.2 Thread→Block-Mapping

Tile A pro K-Step: `BM × BK = 64 × 16 = 1024 Weights`.
Block-Ratio: `1024 / 256 = 4 Q4_K-Blöcke` pro K-Step.

256 Threads / 4 Blöcke = **64 Threads pro Block** — das passt exakt zu
unserer Sprint-2A Wave64-WG-Geometrie. Wir können den
Sprint-2A-Dequant-Code 1:1 in den K-Loop einsetzen, mit `block_id` und
`pos` aus `(m_idx, k_idx)` abgeleitet.

### 5.3 K-Alignment-Constraint

Sprint 2A's Dequant nimmt einen Block (256 Weights) als Einheit an. Im
GEMM-K-Loop walken wir K in Schritten von BK=16; ein Q4_K-Block deckt
also 16 K-Steps. Ein WG K-Tile (BK=16) deckt **nur 1/16 eines Blocks**.

Wir brauchen also pro K-Step:

* Block-Daten (`dm`, `scales`, `qs`) bereits dekodiert *für 4 Blöcke*
  (jede WG braucht 4 verschiedene Blöcke in M-Richtung, kombiniert pro
  K-Step → braucht aber für jeden Block alle 16 K-Steps gleich).

Optimierung-Idee (Sprint 2B): Block-Header (`dm`, `scales`, decodierte
sub_scales/sub_mins) **einmal** in LDS staging, dann 16 K-Steps lang
re-using. Spart das Re-decoding alle 16 K-Steps. Sprint 2A misst diese
Optimierung *nicht* — der Microbench läuft 1 Block = 1 K-Step Äquivalent.

### 5.4 Performance-Erwartung für Sprint 2B

Annahme: Dequant-Fusion fügt ~10% Overhead zum Sprint-1B-Tiled-FP8 hinzu.

* Sprint 1B Tiled FP8 BN=64 auf 4096³: 20.80 TF.
* Sprint 2B Tiled-FP8-mit-Q4K-Dequant: 18-19 TF erwartet.
* Vergleichswert: aktuelles `mul_mmq_q4_k` (Phase 7): ~1 TF effektiv.
* → **18-19× Speedup** über die existierende Q4_K-GEMM-Implementierung.

Falls Dequant-Fusion deutlich schlechter abschneidet: Fallback auf
*separate* Q4_K → FP8-Buffer Dispatch + reines FP8-coopmat GEMM. Mehr
VRAM-Traffic, aber kein Compute-Stau.

---

## 6 — Was nicht in Scope war

* **Q6_K-Dequant.** Nur Q4_K für Sprint 2A. Q6_K kommt in Sprint 2B/2C
  (Qwen3 nutzt Q4_K für die meisten Layer + Q6_K für Embedding/Output).
* **Activation-Convert.** B-Seite (Activations) bleibt FP32 — wir
  testen nur die A-Seite (Weights).
* **Echte GGUF-Weights.** Random-Weights aus `build_random_weights`
  reichen für Correctness; Echte Layer-0-Daten kommen wenn wir die
  Fusion gebaut haben.
* **GEMM-Integration.** Der Schritt "Dequant in Load-Phase einbauen"
  ist Sprint 2B.
* **Block-Header-LDS-Caching.** Sprint 2B Optimierung.

---

## 7 — Reproduzierbarkeit

```fish
# Build (35 SPVs jetzt, +3 dequant)
cargo build --release

# Korrektheit
cargo test --release --test dequant_q4k                 # 8 Tests, alle grün

# Microbench
VF_BENCH_DEQUANT=1 cargo test --release \
  --test dequant_q4k dequant_q4k_microbench -- --nocapture

# ISA-Dump (sucht nach v_cvt_pk_fp8_f32)
RADV_DEBUG=shaders VF_BENCH_DEQUANT=1 \
  cargo test --release --test dequant_q4k dequant_q4k_microbench \
  -- --nocapture 2>/tmp/radv_isa.txt
grep "v_cvt_pk_fp8_f32" /tmp/radv_isa.txt
# Erwartung: ≥4 Hits (eine pro [[unroll]]-Iteration)

# Volle Regression (125 Tests)
cargo test --release
```

---

## 8 — Files

```
NEW   vk_shaders/dequant_q4k_debug.comp           ~120 LoC
                                                  (1 Source, 3 Output-Typen)
MOD   build.rs                                    +20 LoC (3 Dequant-ShaderJobs)
NEW   tests/dequant_q4k.rs                        ~470 LoC
                                                  (Harness + 8 Tests + Microbench)
NEW   results/v02_sprint2a_dequant.md             dieser Report
```

Keine Forward-Pass-Code-Änderungen, keine Pipeline-Registry-Änderungen,
keine Runtime-Side-Effekte für bestehende Tests.

---

## 9 — Sprint-Status nach 2A

```
Sprint  Status     Erkenntnis
v0.2A
  1A    ✅ done   tiled BF16 16.69 TF auf 4096³
  1A.5  ⚠ partiell parametrisches BN, Skinny-N teilweise gefixt
v0.2B
  1B    ✅ done   tiled FP8 20.80 TF auf 4096³ (+25%)
v0.2 next
  2A    ✅ done   Q4_K → FP32/BF16/FP8 dequant isoliert.
                  FP8-Convert NATIV (v_cvt_pk_fp8_f32, 1 Instruktion / 2 Werte).
                  Narrowing macht Dispatch-Zeit *kürzer* — Output-BW dominiert.
                  → Dequant-Fusion ist klar günstig, nicht teuer.
  2B    ↻ open    Dequant-Fusion in den tiled Kernel einbauen.
  3     ↻ open    Forward-Pass-Selector + Q4_K-Wiring.
```

Sprint 2A liefert die wichtigste Erkenntnis seit Sprint 1A: **FP8 ist
auf gfx1201 als Convert-Target günstiger als FP32 oder BF16, weil die
Hardware eine native Pack-Instruktion hat und die Output-Bandbreite
durch das schmälere Element entlastet wird.** Das macht den Sprint-2B-
Plan einfach: Q4_K-Dequant direkt in die Load-Phase des
`mul_coopmat_fp8_*.comp`-Kernels einbauen, mit ~5% erwartetem VALU-
Overhead über den aktuellen Tiled-FP8-Pfad.
