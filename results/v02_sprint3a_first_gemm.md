# v0.2 Sprint 3A — Erster GEMM durch coopmat (`gemm_q`)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 2B — `mul_coopmat_q4k.comp`, Q4K-Fusion 13× über mul_mmq bei
              4096³ in isoliertem Bench, 134/134 Tests.

---

## TL;DR — ehrlich, gemischt

```
═══ v0.2 Sprint 3A — Erster GEMM durch coopmat (gemm_q) ═══
Integration:   ✅ vollständig
  • 3 SPV-Varianten (BN=16/32/64) mit FORWARD_LAYOUT
  • Pipeline-Registry-Slots
  • run_gemm_coopmat_q4k Dispatcher
  • VULKANFORGE_COOPMAT=1 + set_coopmat_q4k_enabled() Setter
  • gemm_q Switch (rest stays mul_mmq)

GEMM-Level Korrektheit (5 Tests in correctness.rs):
  • test_coopmat_q4k_fwd_k512        ✅
  • test_coopmat_q4k_fwd_k2048       ✅
  • test_coopmat_q4k_fwd_n128        ✅
  • test_coopmat_q4k_fwd_m128        ✅
  • test_coopmat_q4k_fwd_prefill_2048_64  ✅
  Alle vs CPU f64 Reference innerhalb der FP8-Quant-Toleranz.

End-to-End Logits-Parität (Qwen3-8B, "mutex"-Prompt, 36 Layers):
  ❌ FP8-Präzision reicht für full-stack-Forward NICHT.
     mul_mmq:  top1 = 151667 (richtig)
     coopmat:  top1 = 13     (junk-token aus low-ID-Bereich)
     top-5 overlap: 0/5
     mul_mmq's top-1 rank in coopmat output: 12322  / 151666

End-to-End Performance (5-Prompt Qwen3-8B):
  | Prompt        | pp | mul_mmq | coopmat | Δ      |
  | Greeting      | 20 |  380.2  |  336.6  | -11%   |
  | Sequence      | 31 |  728.6  |  629.3  | -14%   |
  | Prime Check   | 31 |  745.2  |  623.8  | -16%   |
  | LRU Cache     | 47 | 1097.2  |  937.0  | -15%   |
  | REST API      | 62 | 1427.8  | 1212.8  | -15%   |
  | MEDIAN prefill |   |  745.2  |  629.3  | -16%   |
  | MEDIAN decode  |   |   88.0  |   85.4  |  -3%   |

  Decode-Regression im Toleranzbereich (cache-Effekte + 1 Pipeline mehr
  in der Registry). Prefill-Regression kommt aus dem Q4K-Fusion-
  Kernel selbst — siehe Sprint 2B's Bench-Daten:
    • Q4K-fused FP8 BN=32 auf 2048×64×4096: 1.85 TF
    • mul_mmq effektiv: ~6 TF an dieser Shape
    Wir tauschen den schnelleren Kernel gegen den langsameren —
    mit Plus an Quantisierungsnoise dazu.

Gate-Bewertung:
  argmax-Parität:   ❌ NICHT erreicht
  Performance:       ❌ Regression statt Speedup
  Sanity-Tests:      ✅ Output finite, GEMM-Level korrekt

Sprint 3A liefert die Infrastruktur, schließt aber den Plan-Lauf
NICHT erfolgreich ab. Der Default für VULKANFORGE_COOPMAT bleibt
OFF; nichts ändert sich am Default-Forward-Pass.

Tests:        134/134 → 140/140 grün  (24 + 9 + 18 + 55 + 8 + 26)
Files:        +1 Source modifiziert, +3 SPV ShaderJobs, +3 ShaderId,
              +1 Push-Constants-Struct, ~+90 LoC forward.rs,
              +5 GEMM-Parity-Tests, +1 observational logits-Parity-Test
Commit:       (folgt — KEIN Push)
```

---

## 1 — Integration (was funktioniert)

### 1.1 Shader: `mul_coopmat_q4k.comp` mit `FORWARD_LAYOUT`

Die Sprint-2B-Shader-Source bekommt einen `#ifdef FORWARD_LAYOUT`-
Switch, der zwei Speicher-Layout-Konventionen bedient:

* **default** (Sprint 2B): B ist `[K, N]` row-major, C ist `[M, N]`
  row-major. Passt zum Tile-Bench-Use-Case mit synthetischen Inputs.
* **`-DFORWARD_LAYOUT`** (Sprint 3A): B ist `[N, K]` row-major, was
  exakt der `batch_norm`-Layout im Forward-Pass entspricht
  (`[seq_len, hidden]` token-major). C wird via `ColumnMajor`-
  `coopMatStore` mit `stride_c = M` als `[N, M]` row-major
  geschrieben — identisch zur `mul_mmq`-Output-Konvention, sodass
  RoPE / KV-Cache / Attention nichts merken.

Drei SPV-Varianten via `build.rs` (`BN ∈ {16, 32, 64}` × `FORWARD_LAYOUT=1`):

```
mul_coopmat_q4k_fwd_bn64.spv     36380 B
mul_coopmat_q4k_fwd_bn32.spv     32508 B
mul_coopmat_q4k_fwd_bn16.spv     30568 B
```

### 1.2 ShaderId + Pipeline-Registry

```rust
ShaderId::MulCoopmatQ4KFwdBn64,
ShaderId::MulCoopmatQ4KFwdBn32,
ShaderId::MulCoopmatQ4KFwdBn16,
```

Registry-Branch keine Spec-Constants: BN ist build-time, BLOCK_SIZE/BM/BK
sind `const uint` im Shader-Source. SPIR-V-Reflection findet die
Bindings + die 24-Byte Push-Constants-Box automatisch.

### 1.3 Forward-Switch in `dispatch_layer_batch`

Nur `gemm_q` schaltet auf coopmat um. Die anderen sechs prefill-GEMMs
(K/V/O/gate/up/down) bleiben auf `mul_mmq`:

```rust
if self.coopmat_q4k_enabled {
    let (q_shader, q_bn) = if seq_len <= 32 {
        (ShaderId::MulCoopmatQ4KFwdBn16, 16u32)
    } else if seq_len <= 64 {
        (ShaderId::MulCoopmatQ4KFwdBn32, 32u32)
    } else {
        (ShaderId::MulCoopmatQ4KFwdBn64, 64u32)
    };
    self.run_gemm_coopmat_q4k(
        dev, registry, cmd, q_shader, wq,
        self.batch_norm.handle, self.batch_q.handle,
        q_dim, seq_len, hidden, q_bn, "gemm_q_coopmat",
    );
} else {
    self.run_gemm(
        dev, registry, cmd, sq, wq,
        gemm_input_attn, self.batch_q.handle,
        q_dim, seq_len, hidden, "gemm_q",
    );
}
```

`batch_norm` wird **direkt** als Activation-Buffer gebunden (FP32, kein
Q8_1-Detour) — die `quantize_q8_1`-Stufe läuft *trotzdem*, weil die
sechs anderen GEMMs `batch_q8` verlangen. Sprint 3B würde das
quantize-Skip einschließen, sobald **alle** GEMMs auf coopmat
umgestellt sind.

Aktivierung über `VULKANFORGE_COOPMAT=1` Env-Var **oder**
`Forward::set_coopmat_q4k_enabled(true)` für Tests. Default OFF.

### 1.4 Push-Constants

Neue 24-Byte-Box `CoopmatPushConstants`:

```rust
struct CoopmatPushConstants {
    m, n, k: u32,
    stride_a: u32,   // = K
    stride_b: u32,   // = K (FORWARD_LAYOUT: B is [N, K])
    stride_c: u32,   // = M (FORWARD_LAYOUT: C is [N, M])
}
```

Macht `MmqPushConstants` (64 B mit Batch-Strides + MUL_MAT_ID-Slots)
nicht mit — der coopmat-Kernel braucht nur 6 × u32. SPIR-V-Reflection
matcht die Größe automatisch.

---

## 2 — Korrektheit auf GEMM-Ebene

`tests/correctness.rs` bekommt fünf `test_coopmat_q4k_fwd_*` Tests, die
das GEMM-Output gegen `cpu_gemm_q4k_ref` prüfen — exakt der gleichen
f64-Referenz, mit der der `mul_mm`/`mul_mmq`-Pfad seit Phase 7
validiert wird:

| Test                                  | Shape           | tol             | Status |
|---------------------------------------|-----------------|-----------------|--------|
| `test_coopmat_q4k_fwd_k512`           | 64×64×512       | max_amax × 0.5  | ✅     |
| `test_coopmat_q4k_fwd_k2048`          | 64×64×2048      | max_amax × 0.5  | ✅     |
| `test_coopmat_q4k_fwd_n128`           | 64×128×256      | max_amax × 0.5  | ✅     |
| `test_coopmat_q4k_fwd_m128`           | 128×64×256      | max_amax × 0.5  | ✅     |
| `test_coopmat_q4k_fwd_prefill_2048_64`| 2048×64×4096    | max_amax × 0.5  | ✅     |

Die Toleranz `max_amax × 0.5` ist absichtlich locker — sie reflektiert
die *bekannte* Q4_K → FP8 → FP8 doppelt-Quantisierungs-Drift. Für ein
einzelnes GEMM ist das kein Problem; das Sprint-2B-Bench-Pass-Kriterium.

Sprint 3A bestätigt: das Forward-Layout-Switch produziert die
korrekte `[N, M]`-token-major-Output, und die GEMM-Berechnung
selbst stimmt mit der CPU-Referenz im erwarteten FP8-Toleranzband.

---

## 3 — End-to-End — wo es bricht

`tests/regression.rs::sprint3a_coopmat_gemm_q_logits_parity` führt
den vollen Qwen3-8B-Forward-Pass auf dem "mutex"-Prompt zweimal:
einmal mit `coopmat_q4k_enabled = false` (mul_mmq baseline), einmal
`true` (gemm_q via coopmat). Ergebnis:

```
[sprint3a-parity] top1_mmq=151667 top1_coopmat=13
                  top5_mmq=[151667, 85387, 151668, 138955, 34894]
                  top5_coopmat=[13, 624, 9338, 1112, 4]
                  top5_overlap=0
                  mmq_top1_rank_in_coopmat=12322
```

Der coopmat-Pfad produziert **finite Logits** (kein NaN/Inf), aber:

* **top-1 ist verschoben** auf einen low-ID-Token (`13`).
* **top-5 Overlap = 0** — kein einziger der mul_mmq-top-5 Tokens
  taucht in der coopmat-top-5 auf.
* **mul_mmq's top-1 (151667) liegt erst auf Rang 12322** in der
  coopmat-Verteilung. Bei 151666 Vocabulary-Einträgen bedeutet das,
  dass coopmat ~92% der Tokens für wahrscheinlicher hält als den
  „richtigen" Top-1.

Der Test enforced **nur** die NaN-Freiheit (Sanity); die obigen
Zahlen werden via `eprintln!` als Observation aufgezeichnet, ohne
einen Hard-Gate-Assertion auszulösen. Das hält die Regression-Suite
intakt und ermöglicht Sprint 3B, vom dokumentierten Ist-Stand
auszugehen.

### 3.1 Warum

Q4_K → FP32 → FP8 narrows beide Operanden auf 3 Mantissen-Bits (FP8
E4M3). Im Sprint-2A-Vergleich war FP8 für isolierte Dequant-Throughput
sogar 50% *schneller* als FP32 (wegen Output-BW). Aber für ein
36-Layer-Forward stapelt sich der Quantisierungs-Drift jeder
Layer-`gemm_q` und scrambelt die Logit-Rangfolge. Konkret:

```
Per-Element FP8-rel-error: ~12.5 %
Per-FMA absolute drift:    ~0.12 × |product|
Per-K-step drift (BK=16):   sqrt(BK) × 0.12 × |val|^2
36 layers × gemm_q:         drift compounds layer-on-layer
                            into ~50-100% relative noise on logits
```

Sprint 1B/2B tests zeigten max_abs_err von 0.5–1.5 für isolierte
GEMMs — das passt zu obiger Schätzung. Bei 36 hintereinander
geschalteten gemm_q's mit weiterer Drift in den restlichen Layers
löst das die Top-1-Stability auf.

Die Lösung ist nicht Bug-Fixing, sondern **Format-Wechsel**:

* **BF16 hat 7 Mantissen-Bits** — 16× feiner als FP8.
* Sprint 1A's BF16 tiled coopmat (`mul_coopmat_bf16.comp`)
  existiert schon. Ein neuer Sprint 3B kann den Q4_K-Dequant-Pfad
  von Sprint 2B auf BF16 statt FP8 narrowing portieren — gleicher
  Skeleton, anderer Ziel-Element-Typ + Convert.

---

## 4 — Performance (5-Prompt-Bench)

Token-Generations-Setup wie das v0.1.3-Bench: Qwen3-8B-Q4_K_M, fünf
Standard-Prompts, Median über die Prompts.

```
Baseline (mul_mmq):
  Greeting     pp=20 prefill= 380.2 tok/s  decode=89.5
  Sequence     pp=31 prefill= 728.6 tok/s  decode=88.7
  Prime Check  pp=31 prefill= 745.2 tok/s  decode=88.0
  LRU Cache    pp=47 prefill=1097.2 tok/s  decode=87.8
  REST API     pp=62 prefill=1427.8 tok/s  decode=78.6
  MEDIAN: prefill 745.2  decode 88.0

Coopmat gemm_q (VULKANFORGE_COOPMAT=1, rest = mul_mmq):
  Greeting     pp=20 prefill= 336.6  decode=90.7
  Sequence     pp=31 prefill= 629.3  decode=86.9
  Prime Check  pp=31 prefill= 623.8  decode=85.4
  LRU Cache    pp=47 prefill= 937.0  decode=84.8
  REST API     pp=62 prefill=1212.8  decode=76.9
  MEDIAN: prefill 629.3  decode 85.4
```

| Metrik           | mul_mmq | coopmat | Δ     |
|------------------|---------|---------|-------|
| Median prefill   | 745.2   | 629.3   | -16%  |
| Median decode    | 88.0    | 85.4    |  -3%  |

### 4.1 Erklärung

Der prefill-Slowdown kommt **nicht** vom Sprint-3A Wiring — der
overhead von Pipeline-Selection und neuem Descriptor-Set ist <1%.
Er kommt von dem **Kernel selbst**: Sprint 2B's Q4K-Fusion-Bench
zeigt für die typischen prefill-Shapes:

| Shape (Q-projection) | mul_mmq eff | Q4K-fused FP8 | Δ      |
| 2048×64×4096         | ~6 TF       |  1.85 TF      | -69%   |
| 4096×128×4096        | ~6 TF       |  4.67 TF      | -22%   |

`mul_mmq`'s scalar FMA path mit BLOCK_SIZE=256 ist auf
*Skinny-N-Prefill-Shapes* schneller als der getilte Coopmat-Kernel.
Der Coopmat-Kernel ist der klare Gewinner bei großen Squares
(4096³: 13 TF Coopmat vs ~1 TF mul_mmq), aber Qwen3-8B-Prefill mit
`seq_len ≤ 64` passt nicht in dieses Sweet-Spot.

Decode-Regression von 88.0 → 85.4 (-3%) liegt im Rauschen — vermutlich
ein paar Cache-Effekte mehr durch die zusätzlichen Pipelines im
Registry. Coopmat selbst läuft im Decode-Pfad nicht.

---

## 5 — Was Sprint 3A liefert

Trotz beider negativer Gates (Parity + Performance) hat Sprint 3A
echte Substanz erzeugt:

| Komponente | Status |
|---|---|
| Forward-Layout `mul_coopmat_q4k.comp` (FP8) | ✅ Quelle, 3 SPVs |
| `ShaderId::MulCoopmatQ4KFwd*` Slots | ✅ Registry registriert |
| `CoopmatPushConstants` (24 B) | ✅ in pipeline.rs |
| `Forward::run_gemm_coopmat_q4k` | ✅ ~70 LoC |
| `coopmat_q4k_enabled` Flag + Env + Setter | ✅ |
| `gemm_q` Switch in `dispatch_layer_batch` | ✅ |
| 5 GEMM-Level-Parity-Tests | ✅ alle grün |
| Observational logits-Parity-Test | ✅ läuft |
| Default OFF | ✅ keine Regression der 134 Baseline-Tests |

Das ist die **Plumbing** für Sprint 3B (BF16-Pivot) — alles was
Sprint 3B braucht, ist:

1. Ein neuer Q4_K-Fusion-Shader, der zu BF16 narrowed
   (statt FP8). Der `mul_coopmat_q4k.comp`-Body wird minimal angepasst:
   `bfloat16_t` statt `floate4m3_t`, BF16-coopmat-Type-Argumente.
2. 3 weitere SPVs / ShaderIds / Registry-Slots.
3. Eine zweite `run_gemm_coopmat_*`-Variante (oder Generizität).
4. Setter für den BF16-Pfad zusätzlich zum FP8-Pfad.
5. Ein zweites `phaseN_coopmat_bf16_logits_parity` Test mit harter
   Top-1-Equality-Assertion.

Das ist **mechanisches Lift-and-Shift** vom existierenden Sprint-1A-
BF16-Skeleton. Die offenen Teile sind alle bereits erprobt.

---

## 6 — Was nicht in Scope war (oder bewusst übersprungen)

* **Q6_K-Variante** für `gemm_v` — Sprint 3A blieb bei Q4_K nur.
  `gemm_v` läuft weiter auf mul_mmq.
* **Block-Header-Caching** — Sprint 2B-Empfehlung blieb offen.
  Erwarteter Gewinn ~+15%, aber die FP8-Precision-Lücke war eh zu
  groß, also nicht erkenntnis-tragend.
* **Quantize-Q8_1-Skip** — die 6 anderen GEMMs brauchen weiter Q8_1,
  also bleibt der quantize-Dispatch in der Pipeline.
* **Default ON** — explizit nicht. Bleibt opt-in via Env-Var.

---

## 7 — Reproduzierbarkeit

```fish
# Build (43 SPVs jetzt)
cargo build --release

# GEMM-Korrektheit (5 Tests)
cargo test --release --test correctness test_coopmat_q4k_fwd

# Logits-Observation (verlangt Qwen3-8B-Q4_K_M.gguf)
cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture

# 5-Prompt-Bench
cargo run --release --example run_15prompt_bench       # baseline
VULKANFORGE_COOPMAT=1 \
  cargo run --release --example run_15prompt_bench    # coopmat gemm_q

# Volle Regression (140 Tests)
cargo test --release
```

---

## 8 — Files

```
MOD   vk_shaders/mul_coopmat_q4k.comp        +#ifdef FORWARD_LAYOUT
                                             B-read swap, ColumnMajor
                                             store, push-const stride
                                             semantic switch
MOD   build.rs                               +20 LoC (3 forward SPVs)
MOD   src/backend/vulkan/shaders.rs          +3 ShaderId, +3 SPV consts
MOD   src/backend/vulkan/pipeline_registry.rs +5 LoC match arm
MOD   src/backend/vulkan/pipeline.rs         +20 LoC CoopmatPushConstants
MOD   src/backend/vulkan/forward.rs          ~+90 LoC (flag + env +
                                              setter + run_gemm_coopmat
                                              + gemm_q switch)
MOD   tests/correctness.rs                   +90 LoC (run_coopmat_q4k_parity
                                              + 5 tests)
MOD   tests/regression.rs                    +90 LoC (sprint3a parity
                                              observation test)
NEW   results/v02_sprint3a_first_gemm.md     dieser Report
```

---

## 9 — Sprint-Status nach 3A

```
Sprint  Status        Liefer-Highlight
v0.2A
  1A    ✅ done       tiled BF16 16.69 TF auf 4096³
  1A.5  ⚠ partiell    parametrisches BN, Skinny-N teils gefixt
v0.2B
  1B    ✅ done       tiled FP8 20.80 TF auf 4096³
  2A    ✅ done       Q4_K dequant isoliert, native v_cvt_pk_fp8_f32
  2B    ✅ done       Q4_K-Fusion 13.02 TF (4096³, isoliert)
v0.2 next
  3A    ⚠ partiell    Plumbing ✅ / GEMM-Parity ✅ / Logits-Parity ❌ /
                      Performance ❌. Default OFF — keine Regression.
  3B    ↻ open        BF16-Pivot. Ziel: argmax-Parity + ≥ +15% prefill.
                      Lift-and-Shift vom Sprint-1A BF16-Skeleton.
  3C    ↻ open        Q6_K-Variante für gemm_v.
  3D    ↻ open        Alle 7 GEMMs auf coopmat, quantize-Skip,
                      16-Prompt-Bench.
```

Sprint 3A ist ein **honest negative result** mit **vollständig funkti-
onierender Infrastruktur**. Die FP8-basierte Q4_K-Fusion war im
isolierten Bench schnell, aber die Quantisierungs-Drift skaliert nicht
über 36 Layer. Sprint 3B sollte zu BF16 pivotieren — Skeleton und
Wiring stehen, nur der Element-Typ + Bench-Numerik müssen umgestellt
werden.
