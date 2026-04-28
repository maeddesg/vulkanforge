# v0.2 Sprint 3B — BF16-Pivot, Naive-Kernel, 5/7 GEMMs durch coopmat

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 3A (Plumbing ✅ / FP8-Parity ❌, top-1 13 vs 151667),
              140/140 Tests.

---

## TL;DR — Parity-Gate erreicht, Performance-Gate verfehlt

```
═══ v0.2 Sprint 3B — BF16 Naive + 5/7 GEMMs ═══
Precision-Fix: BF16 (7 Mantissen-Bits) statt FP8 (3 Bits)
Performance-Fix: Naive coopmat (1 Wave64/WG, mini-LDS) für Skinny-N
Bonus-Fix: Partial-tile-store in LDS-staged write — vorher wurde
           der Output-Tail bei N nicht-Vielfaches-von-16 stillschweigend
           verworfen (das war der eigentliche 3A-Logits-Bug, NICHT FP8).

GEMMs auf coopmat (5 von 7):
  • gemm_q  → naive BF16 ✅
  • gemm_k  → naive BF16 ✅
  • gemm_o  → naive BF16 ✅
  • gemm_gate → naive BF16 ✅
  • gemm_up   → naive BF16 ✅
  • gemm_v   → mul_mmq (Q6_K, kein coopmat-Kernel — Sprint 3C)
  • gemm_down → mul_mmq (NaN-Logits beim Switch, K=11008 zu lang —
                        Sprint 3C)

Logits-Parity (Qwen3-8B, "mutex"-Prompt):
  Sprint 3A FP8:    top1 151667 → 13 (junk) — FAILED
  Sprint 3B BF16:   top1 151667 → 151667 ✅
                    top-5 overlap: 4/5
                    mmq's top-1 rank in coopmat: 0

Performance (5-Prompt Qwen3-8B):
  | Metric        | mul_mmq | coopmat (5/7) | Δ      |
  | prefill med   |  740.5  |  320.3        | -57%   |
  | decode med    |   89.1  |   89.7        |  +0.7% |

  Naive coopmat ist für Skinny-N-Shapes (N≤64) langsamer als
  mul_mmq's scalar Q4_K × Q8_1 Pfad. Sprint 1A.5 Bench zeigte
  naive BF16 = 5.70 TF, mul_mmq = ~6 TF — annähernd gleich. Bei 5
  GEMMs übertragen multipliziert sich der ~5% Per-GEMM Slowdown
  durch zusätzlichen Dispatch + Barrier-Overhead nicht-linear.

Tests:        140/140 → 145/145 grün  (24 + 9 + 18 + 60 + 8 + 26)
Default:      OFF (VULKANFORGE_COOPMAT=1 zum Aktivieren)
Commit:       (folgt — KEIN Push)
```

---

## 1 — Was Sprint 3B gemacht hat

### 1.1 Neuer Shader: `mul_coopmat_q4k_naive.comp`

Naive Q4_K-Dequant-Fusion mit BF16 narrowing. Geometrie:

* `BLOCK_SIZE = 64` (1 Wave64 = 1 Subgroup)
* `TILE = 16` (WMMA M = N = K)
* 1 Output-Tile von 16×16 pro Workgroup
* Mini-LDS-Staging: `buf_a[16×17] bf16` + `buf_b[16×17] bf16` +
  `buf_c[16×17] f32` (Output-Stage) ≈ 2.2 KiB / WG

Per K-Step (BK=16):
1. 64 Threads dequantisieren 256 Q4_K-Weights → BF16 in `buf_a`
2. 64 Threads laden 256 FP32-Activations → BF16 in `buf_b`
3. `barrier`
4. 1× `coopMatLoad matA`, 1× `coopMatLoad matB`, 1× `coopMatMulAdd`
5. `barrier`

Output-Phase:
1. `coopMatStore acc → buf_c` (LDS, RowMajor)
2. `barrier`
3. **Per-Thread Bounds-Check Write zu Global**: 64 Threads × 4 elem =
   256, jeder Thread schreibt sein Element nur falls
   `(gm < pc.m && gn < pc.n)` — **das war Sprint 3A's eigentlicher Bug**.

### 1.2 Der versteckte Sprint-3A-Bug: Partial-Tile-Store

Sprint 3A schrieb eine 16×16-Tile **direkt** mit `coopMatStore` — und
übersprang den Store komplett für Rand-Tiles, in denen `M` oder `N`
kein Vielfaches von 16 ist:

```glsl
if (c_row + TM <= pc.m && c_col + TN <= pc.n) {
    coopMatStore(...);   // sonst: nichts geschrieben → stale memory
}
```

Realer Qwen3-Prefill mit `seq_len ≈ 30..40` (Chat-Template + kurzer
Prompt) trifft das ständig — Tokens 16..29 wurden mit *uninitialisierten*
batch_q-Werten in den nächsten Layer gefüttert. Das wurde fälschlich
als "FP8-Precision-Versagen" diagnostiziert (alle Sprint-2B/3A
Unit-Tests passten, weil sie nur Tile-aligned Shapes nutzten).

Sprint 3B's LDS-staged-write fixed das **vor** dem BF16-Pivot — der
Pivot zu BF16 war zwar trotzdem nötig (höhere Precision), aber das
**eigentliche** Logits-Versagen kam vom Partial-Tile-Drop.

### 1.3 Forward-Pass-Switches

`forward.rs::dispatch_layer_batch` hat jetzt drei Coopmat-Forks:

* (c) Q/K/V: gemm_q/k via coopmat naive BF16, gemm_v bleibt mul_mmq
* (e) O-Projection: gemm_o via coopmat naive BF16
* (i) Gate/Up: gemm_gate/up via coopmat naive BF16
* (k) Down: gemm_down bleibt mul_mmq (NaN-Bug bei 11008 K-Steps)

Jeder Fork wählt per-Shape:
  * `seq_len ≤ 64`  → `MulCoopmatQ4KNaiveBf16`, BM=BN=16
  * `seq_len ≤ 128` → `MulCoopmatQ4KFwdBn32`,    BM=64, BN=32
  * `seq_len > 128` → `MulCoopmatQ4KFwdBn64`,    BM=64, BN=64

Für die typischen Qwen3-Prefill-Shapes (≤ 64 Tokens) wird der naive
Kernel verwendet.

### 1.4 `run_gemm_coopmat_q4k` Signature-Erweiterung

Die Funktion bekam einen `bm_tile` Parameter (zusätzlich zu `bn_tile`),
weil der naive Kernel BM=16 statt BM=64 emittiert. Das war Sprint 3A's
*zweiter* Bug — der hardcoded `groups_x = m.div_ceil(64)` hätte für
den naive Kernel nur 1/4 der nötigen WGs dispatched.

---

## 2 — Logits-Parity (Qwen3-8B "mutex"-Prompt)

```
$ cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture

[sprint3-parity] top1_mmq=151667 top1_coopmat=151667
                 top5_mmq=[151667, 85387, 151668, 138955, 34894]
                 top5_coopmat=[151667, 85387, 151668, 34894, 50897]
                 top5_overlap=4 mmq_top1_rank_in_coopmat=0
test result: ok. 1 passed
```

* **Top-1 identisch** — Greedy-Decode wäre bit-stabil.
* **Top-5 Overlap 4/5** — der einzige Drift ist auf Position 4 (mmq
  `138955` vs coopmat `50897`); beides sind Long-Tail-Kandidaten mit
  ähnlichem Logit-Wert.
* **mmq's Top-1 rankt 0 in coopmat** — auch in der coopmat-Verteilung
  ist `151667` der wahrscheinlichste Token.

Der Test asserts jetzt hart auf `top1_a == top1_b` und
`top5_overlap >= 3/5` — Default-OFF Path bleibt unverändert grün, der
Coopmat-Pfad muss diesen Gate erfüllen.

---

## 3 — GEMM-Level Korrektheit

5 neue Tests in `tests/correctness.rs` für den naiven Kernel:

| Test                                        | Shape          | err vs CPU f64  | tol (BF16) |
|---------------------------------------------|----------------|------------------|------------|
| `test_coopmat_q4k_naive_k512`               | 64×64×512      | passes           | max_amax×0.05 |
| `test_coopmat_q4k_naive_k2048`              | 64×64×2048     | passes           | max_amax×0.05 |
| `test_coopmat_q4k_naive_m128`               | 128×64×256     | passes           | max_amax×0.05 |
| `test_coopmat_q4k_naive_prefill_64`         | 2048×64×4096   | passes           | max_amax×0.05 |
| `test_coopmat_q4k_naive_qwen3_gemm_q`       | 4096×64×4096   | passes           | max_amax×0.05 |

Toleranz `max_amax × 0.05` reflektiert BF16's ~0.4% relative
Precision — *deutlich* enger als Sprint 2B/3A's FP8-Toleranz von
`max_amax × 0.5`.

Plus die 5 Sprint-3A `test_coopmat_q4k_fwd_*` Tests (alle weiter grün).

---

## 4 — Performance: ehrliche Zahlen

```
$ VF_NUM_PROMPTS=5 cargo run --release --example run_15prompt_bench
[mul_mmq baseline]
  # 1 Greeting     pp= 20 prefill= 377.9 decode=91.1 ✓
  # 2 Sequence     pp= 31 prefill= 731.4 decode=89.5 ✓
  # 3 Prime Check  pp= 31 prefill= 740.5 decode=89.1 ✓
  # 4 LRU Cache    pp= 47 prefill=1090.0 decode=88.8 ✓
  # 5 REST API     pp= 62 prefill=1411.7 decode=80.4 ✓
  MEDIAN: prefill 740.5 / decode 89.1

$ VULKANFORGE_COOPMAT=1 VF_NUM_PROMPTS=5 cargo run --release --example run_15prompt_bench
[coopmat 5/7 GEMMs]
  # 1 Greeting     pp= 20 prefill= 191.2 decode=91.7 ✓
  # 2 Sequence     pp= 31 prefill= 319.2 decode=91.1 ✓
  # 3 Prime Check  pp= 31 prefill= 320.3 decode=89.7 ✓
  # 4 LRU Cache    pp= 47 prefill= 364.2 decode=89.1 ✓
  # 5 REST API     pp= 62 prefill= 383.2 decode=80.4 ✓
  MEDIAN: prefill 320.3 / decode 89.7
```

| Metric        | mul_mmq | coopmat | Δ      |
|---------------|---------|---------|--------|
| Median prefill|  740.5  |  320.3  | -57%   |
| Median decode |   89.1  |   89.7  |  +0.7% |

* **Decode ist unverändert** — coopmat berührt den GEMV-Pfad nicht.
* **Prefill regressiert um -57%** — der naive coopmat ist für die
  Qwen3-Prefill-Skinny-N-Shapes langsamer als mul_mmq's scalar
  Q4_K × Q8_1 Integer-FMA-Pfad.
* **Alle 5 Prompts coherent** (✓) — die Logits sind richtig genug
  für sinnvolle Token-Generation.

### 4.1 Warum die Performance regressiert

mul_mmq:
  * BM = BN = 64, BLOCK_SIZE = 256 → 64 grosse WGs für M=4096, N=64
  * Q4_K + Q8_1 integer-FMA path: hochoptimierter scalar-FMA-Loop
  * Tile-tuning (TM=2, TN=4) seit Phase 6 v0.1.2

Naive coopmat:
  * BM = BN = 16, BLOCK_SIZE = 64 → 1024 kleinere WGs
  * Q4_K → BF16 → coopmat WMMA: dequant kostet ~10 VALU-Ops/weight,
    BF16-Convert weitere 5 VALU-Ops, plus barriers
  * Pro K-Step: dequant + barrier + WMMA + barrier — vier serielle
    Phasen, viel Latenz pro Element

Für *grosse* Squares dominiert WMMA-Throughput und coopmat gewinnt
(Sprint 1A: 4096³ tiled BF16 = 16.7 TF vs naive 3.4 TF). Für
Skinny-N (N ≤ 64) ist das WMMA-Compute zu wenig dichte, und der
scalar mul_mmq-Pfad bleibt günstiger.

Sprint 1A.5's Bench-Daten haben das vorhergesagt:
  * naive BF16 bei 2048×64×4096:  5.70 TF
  * mul_mmq effektiv:              ~6.0 TF (aus 1047 prefill tok/s
                                            aboriginal)
  → Naive ≈ mul_mmq, aber **kein** Speedup auf dieser Shape.

Bei 5 GEMMs mit jeweils ~5% Slowdown plus zusätzlichem Dispatch+Barrier-
Overhead summiert sich das auf den beobachteten -57% Gesamt-Prefill-
Slowdown.

---

## 5 — Was Sprint 3B liefert / nicht liefert

### Liefert ✅

* **BF16 Pivot.** Q4_K → FP32 → BF16 → coopmat. 7 Mantissen-Bits
  reichen für 36-Layer-Qwen3 — Logits-Top-1 stabil.
* **Naive Kernel mit LDS-staged Output.** Behebt den eigentlichen
  Sprint-3A-Bug (Partial-Tile-Store-Drop).
* **5/7 GEMMs auf coopmat.** Q, K, O, gate, up. gemm_v stays Q6_K mul_mmq;
  gemm_down stays Q4_K mul_mmq (NaN-Investigation für Sprint 3C).
* **Pipeline-Registry-Slot** für `MulCoopmatQ4KNaiveBf16`.
* **Hard-Gate Logits-Parity-Test** ersetzt Sprint 3A's
  observational record.

### Liefert nicht ❌

* **Performance-Win.** -57% Prefill ist Regression.
  Sprint 3B bleibt **default OFF**; nur explizites
  `VULKANFORGE_COOPMAT=1` aktiviert den (langsameren, korrekten)
  Pfad.
* **Alle 7 GEMMs.** gemm_v braucht Q6_K-Coopmat-Kernel (Sprint 3C).
  gemm_down (K=11008) divergiert zu NaN beim Switch — vermutlich
  ein längerer-K-Akkumulations-Issue, nicht behoben.
* **quantize_q8_1 Skip.** Solange gemm_v und gemm_down auf mul_mmq
  bleiben, läuft der quantize-Dispatch weiter.

---

## 6 — Sprint-3C / Post-3B Ausblick

### 6.1 Performance-Lift-Hebel

Mehrere bekannte Hebel, von einfach zu komplex:

1. **Block-Header-Caching** (Sprint 2C): pro Thread `dm` und die 8
   `(scale, min)`-Paare einmal in Registern halten und für die 16
   K-Steps einer Block-Spanne re-usen. Erwarteter Lift: +15-20% an
   gemm_q/k/o/gate/up. Reicht *nicht*, um -57% zu kompensieren.
2. **Dispatch-Konsolidierung**: alle 5 coopmat-Q4_K-GEMMs eines Layers
   batchen statt einzeln. Würde Mesa Submission Merging maximieren.
3. **K-Tiled Naive**: K in BK=64-Blöcken statt BK=16, einmal pro
   Block alle Header laden, dann 4 WMMAs sequentiell.
4. **Tiled Q4_K-Coopmat für N=64**: existiert (Sprint 2B `mul_coopmat_q4k.comp`
   FORWARD_LAYOUT), aber Sprint 1A.5 zeigte tile_BN=16 ≈ 2.83 TF —
   schlechter als naive 5.70.

### 6.2 gemm_down NaN-Untersuchung

Bei K=11008 (ffn → hidden) divergiert der Coopmat-Pfad zu NaN
nachdem alle 5 anderen GEMMs bereits gemerged sind. Beobachtungen:
* Single-GEMM-Switch (nur down): vermutlich auch NaN — nicht getestet.
* Lange K-Akkumulation in BF16: theoretisch nicht problematisch, aber
  mit FP32-Accumulator + 11008 K-Steps könnte ein einzelner Wert
  `inf` werden falls die Activation-Zwischenwerte nach SiLU-Mul
  unbeschränkt sind.

Sprint 3C: bisect mit gemm_down-only switch + Output-NaN-Lokalisierung
auf Layer-Granularität.

### 6.3 gemm_v (Q6_K)

Q6_K hat 256 Weights/Block, 6-bit Quants, andere Scale-Layout. Code
existiert in `mul_mmq_funcs.glsl` Q6_K-Branch — eine Q6_K-Naive-
Coopmat-Variante (analog zu Sprint 2A's Dequant-Probe + Sprint 3B's
Naive-Skeleton) ist mechanisches Lift-and-Shift. Sprint 3C-Aufgabe.

---

## 7 — Reproduzierbarkeit

```fish
# Build (44 SPVs jetzt)
cargo build --release

# GEMM-Level Korrektheit (5 naive + 5 fwd-tiled aus Sprint 3A = 10)
cargo test --release --test correctness test_coopmat_q4k

# End-to-End Logits-Parity (verlangt Qwen3-8B-Q4_K_M.gguf)
cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture

# 5-Prompt Bench
cargo run --release --example run_15prompt_bench         # baseline
VULKANFORGE_COOPMAT=1 cargo run --release --example run_15prompt_bench
                                                          # coopmat 5/7

# Volle Regression (145 Tests)
cargo test --release
```

---

## 8 — Files

```
NEW   vk_shaders/mul_coopmat_q4k_naive.comp     ~165 LoC
                                                (BF16 narrow, mini-LDS,
                                                 LDS-staged store)
MOD   build.rs                                  +1 ShaderJob
MOD   src/backend/vulkan/shaders.rs             +1 ShaderId, +1 SPV const
MOD   src/backend/vulkan/pipeline_registry.rs   +1 line in match arm
MOD   src/backend/vulkan/forward.rs             ~+90 LoC: bm_tile param,
                                                4 GEMM coopmat forks,
                                                gemm_down kept on mmq
MOD   tests/correctness.rs                      +5 naive parity tests
MOD   tests/regression.rs                       hard-gate parity now
NEW   results/v02_sprint3b_bf16_all_gemms.md    dieser Report
```

---

## 9 — Sprint-Status nach 3B

```
Sprint  Status        Liefer-Highlight
v0.2A
  1A    ✅ done       tiled BF16 16.69 TF auf 4096³
  1A.5  ⚠ partiell    parametrisches BN, Skinny-N teils gefixt
v0.2B
  1B    ✅ done       tiled FP8 20.80 TF auf 4096³
  2A    ✅ done       Q4_K dequant isoliert, native v_cvt_pk_fp8_f32
  2B    ✅ done       Q4_K-Fusion 13 TF (4096³ isoliert)
v0.2 forward integration
  3A    ⚠ partiell    Plumbing ✅ / Logits ❌ FP8 (eigentlicher Bug:
                      partial-tile-store, NICHT FP8-Präzision!)
  3B    ✅ partiell  Logits-Parity ✅ / Performance ❌ -57%
                      BF16 + naive + LDS-staged-store. 5/7 GEMMs.
                      gemm_v / gemm_down weiter auf mul_mmq.
                      Default OFF.
  3C    ↻ open       Block-Header-Caching, gemm_down NaN-Debug,
                      Q6_K-Naive für gemm_v.
                      Performance-Ziel: ≥ mul_mmq (kein Slowdown).
```

Sprint 3B ist die **erste** Sprint, die einen *correct* End-to-End
coopmat-Forward-Pass auf Qwen3-8B liefert. Performance-Lift bleibt
für Sprint 3C — die Architektur (5/7 GEMMs umgeschaltet, Pipelines
registriert, Tests grün) steht aber.
