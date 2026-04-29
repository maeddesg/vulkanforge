# v0.2 Sprint 10D — PV-coopmat (Honest Negative Result)

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 10C (QK-coopmat + PV-scalar, +85.8% pp=2048, 167/167 Tests)

---

## TL;DR — PV-coopmat ist KORREKT (167/167 Tests grün) aber LANGSAMER (−1% bis −24%) als Sprint 10C. Reverted, Sprint 10C bleibt der Production-Winner.

```
═══ v0.2 Sprint 10D ═══

VERSUCH: Erweitere flash_attn_coopmat.comp so dass auch der PV-
         Schritt via VK_KHR_cooperative_matrix WMMA läuft, nicht
         mehr scalar wie in Sprint 10C.

DESIGN (Ansatz C — Hybrid):
  • coopmat PV via 8 × coopMatMulAdd über head_dim/16 Segmente
  • P-Fragment EINMAL aus p_lds laden (FP16) — invariant für alle 8 V-Segmente
  • V-Tile in v_lds (FP16, 4 KB) staged
  • Pro Segment: coopMatStore(PV_frag → pv_scratch[8 KB FP32])
  • barrier
  • Per-thread accumulate: my_out0/my_out1 += pv_scratch[qi*HEAD_DIM + tid/+64]
  • Existierende per-thread my_out + Online-Softmax-Rescale UNVERÄNDERT
    → vermeidet die KHR-coopmat-* scalar Frage komplett

LDS-Budget Sprint 10D: 21.5 KB (vs Sprint 10C's 9 KB)
  q_lds:      4 KB (FP16, unchanged)
  k_lds:      4 KB (FP16, unchanged)
  v_lds:      4 KB (FP16, NEW)
  scores_lds: 1 KB (FP32, unchanged)
  p_lds:    0.5 KB (FP16, NEW — softmax output for coopmat PV)
  pv_scratch: 8 KB (FP32, NEW — coopMatStore output, transient)

KORREKTHEIT: ✓ ✓ ✓
  Build: flash_attn_coopmat.spv (FP32 KV) + flash_attn_coopmat_fp16kv.spv
         beide kompilierten clean (64 KB SPV jeweils, +8 KB vs 10C).

  cargo test --release                              → 167/167 ✓
  VULKANFORGE_COOPMAT_ATTN=1 cargo test --release   → 167/167 ✓
  → Alle E2E argmax-parity Tests grün:
       phase3e_prefill_batch_matches_token_by_token_top5
       sprint5b_chunked_prefill_parity_qwen3
       phase5b2_decode_after_batched_prefill_qwen3
       phase_prompt16_alice_context_retention_qwen3
       phase5b2_batch_attn_parity_qwen3_short / two_tiles
  → FP16 P × FP16 V → FP32 Akkumulation produziert top-1
    bit-äquivalente Logits zur scalar PV-Variante.

PERFORMANCE: ✗ — REGRESSION

  | pp   | 10C (QK-cm) | 10D (QK+PV-cm) | Δ tok/s | Δ %     |
  |------|------------:|---------------:|--------:|--------:|
  |  128 |        2010 |           1982 |     -28 |  -1.4%  |
  |  512 |        2241 |           2065 |    -176 |  -7.9%  |
  | 1024 |        2193 |           1888 |    -305 | -13.9%  |
  | 2048 |        1989 |           1511 |    -478 | -24.0%  |

  Note: bei pp=2048 ist 10D (1511) immer noch +41% vs Sprint 9d.3
  baseline (1070) — coopmat ALLEIN ist immer noch ein Win, NUR die
  PV-Erweiterung über QK hinaus regrediert.

ROOT CAUSE (im Brief antizipiert):
  Bei Bc=16 hat die PV-Phase nur 1 P-Fragment-K-Schritt × 8 V-
  Output-Segmente = 8 coopMatMulAdd. Die Compute-Last (8 × WMMA)
  ist klein — kleiner als die Overhead-Last die durch das
  KHR-only-Pattern entsteht:

  1. **pv_scratch FP32 8 KB write+read pro K-Tile** — 16 KB extra
     LDS-Bandbreite die in 10C's scalar PV nicht existierte
     (10C las V direkt aus Global ohne LDS-Stage).
  2. **V-LDS-Staging Pflicht** — coopMatLoad braucht V in LDS;
     in 10C wurde V direkt aus Global gelesen via load_v().
     +4 KB LDS-Schreibvorgang plus 1 zusätzlicher Barrier.
  3. **Barriers/K-Tile: 6 (10D) vs 4 (10C)** — 50% mehr
     workgroup-wide Sync-Punkte.
  4. **Occupancy reduziert**: 21.5 KB LDS/WG → 2 WGs/CU bei
     RDNA4's 64 KB pro CU (vs 9 KB in 10C → bis zu 7 WGs/CU,
     ~3.5× mehr in-flight WGs). Der Occupancy-Verlust schlägt
     bei längeren Sequenzen stärker durch (mehr K-Tiles =
     mehr Synchronisation = höhere Latenz pro Forward).

ENTSCHEIDUNG: REVERT zur Sprint 10C-Konfiguration.
  Per Brief Schritt 7:
    "Falls Performance < 10C bei manchen pp:
       → Default: 10C Konfiguration (QK-coopmat + PV-scalar)
       → PV-coopmat als opt-in"

  Wir haben < 10C bei ALLEN gemessenen pp. Der PV-coopmat-Pfad
  würde nicht nur "manchmal" sondern "immer" verlieren.

  REVERT-Aktion:
    • flash_attn_coopmat.comp via `git checkout` zurück auf 10C
    • SPVs neu kompiliert (zurück auf 56 KB)
    • Tests 167/167 grün auf BEIDEN Pfaden bestätigt
    • Production-Code unverändert seit Sprint 10C

DEFAULT für VULKANFORGE_COOPMAT_ATTN: bleibt OFF.
  Begründung: bei Citrix-Noise + lückenhafter Production-Bench-
  Coverage ist eine bewusste Default-Switch-Entscheidung erst
  nach einem sauberen Bench ohne Citrix sinnvoll. Sprint 10E
  könnte das angehen.

Tests: 167/167 ✓ (kein Code committed außer dieser Report)
Files: nur results/v02_sprint10d_pv_coopmat.md
Commit: HEAD (KEIN Push)
```

---

## 1. Was implementiert wurde (und dann reverted)

### 1.1 Modified flash_attn_coopmat.comp

Die Modifikation umfasste 99 hinzugefügte Zeilen, 20 entfernt (grob
aus +80 LOC für PV-coopmat bestehend). Konkret:

**LDS-Erweiterungen (Header):**

```glsl
shared float16_t v_lds[BC * HEAD_DIM];      // 4 KB (NEW)
shared float16_t p_lds[BR * BC];            // 0.5 KB (NEW)
shared float     pv_scratch[BR * HEAD_DIM]; // 8 KB (NEW)
```

**Softmax-Anpassung:**

```glsl
// Statt FP32 in scores_lds:
//   scores_lds[qi * BC + tid] = pscore;
// schreibt FP16 in p_lds für coopmat-PV:
if (tid_in_tile) {
    p_lds[qi * BC + tid] = float16_t(pscore);
} else if (tid < BC) {
    p_lds[qi * BC + tid] = float16_t(0);  // tile-boundary pad
}
```

**Scalar PV-Loop ersetzt durch coopmat:**

```glsl
// (2d) V-tile load into FP16 LDS for coopmat PV
for (uint local_pos = 0; local_pos < TILE; ++local_pos) {
    uint global_t = tile_base + local_pos;
    if (global_t < causal_len_max) {
        uint kv_off = global_t * pos_stride + head_off;
        v_lds[local_pos * HEAD_DIM + tid]      = float16_t(load_v(kv_off + tid));
        v_lds[local_pos * HEAD_DIM + tid + 64] = float16_t(load_v(kv_off + tid + 64));
    } else {
        v_lds[local_pos * HEAD_DIM + tid]      = float16_t(0);
        v_lds[local_pos * HEAD_DIM + tid + 64] = float16_t(0);
    }
}
barrier();

// (2e) PV coopmat: 1 P-fragment × 8 V-fragments → 8 O-segments
{
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> P_frag;
    coopMatLoad(P_frag, p_lds, 0, BC, gl_CooperativeMatrixLayoutRowMajor);

    for (uint d = 0; d < HEAD_DIM; d += 16) {
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> V_frag;
        coopMatLoad(V_frag, v_lds, d, HEAD_DIM, gl_CooperativeMatrixLayoutRowMajor);

        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> PV_frag =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        PV_frag = coopMatMulAdd(P_frag, V_frag, PV_frag);

        coopMatStore(PV_frag, pv_scratch, d, HEAD_DIM,
                     gl_CooperativeMatrixLayoutRowMajor);
    }
}
barrier();

// (2f) Per-thread accumulate from pv_scratch
[[unroll]] for (uint qi = 0; qi < BR; ++qi) {
    my_out0[qi] += pv_scratch[qi * HEAD_DIM + tid];
    my_out1[qi] += pv_scratch[qi * HEAD_DIM + tid + 64];
}
barrier();   // guard pv_scratch + v_lds + p_lds reuse for next tile
```

**Diff-Eigenschaft:** lokal beschränkt auf den PV-Block; QK + Softmax
unverändert. Gut reviewbarer Patch.

### 1.2 Warum Ansatz C (statt Brief's Ansatz B)

Brief's empfohlener Ansatz B war: O als 8 coopmat-Fragmente in
Registern halten, mit Online-Softmax-Rescale via `O_frag *= kfac`.

**Problem mit Ansatz B:** kfac ist per-row (per qi), nicht uniform
across das Fragment. KHR coopmat hat keinen elementweisen
Per-Row-Scale ohne NV-Erweiterung. Eine Diagonal-Matrix-Multiplikation
wäre möglich aber teuer (zusätzliche coopMatMulAdd nur für Rescale).

**Ansatz C (was wir gemacht haben):** Per-thread-Akkumulator
`my_out0[qi] / my_out1[qi]` aus Sprint 10C beibehalten. PV-coopmat
schreibt in transienten LDS-Scratch; Threads addieren ihre Slots.
Rescale `my_out *= kfac` bleibt scalar — trivial.

Ansatz C ist konzeptionell sauber, aber die LDS-Round-Trip-Kosten
für `pv_scratch` sind hoch (8 KB Write + 8 KB Read pro K-Tile).

---

## 2. Performance-Analyse

### 2.1 pp-Sweep (3 runs / 1 warmup, Citrix-noisy)

```
| pp   | Scalar (9d.3) | 10C (QK-cm) | 10D (QK+PV-cm) | 10D Δ vs 9d.3 | 10D Δ vs 10C |
|------|--------------:|------------:|---------------:|--------------:|-------------:|
|  128 |        1876   |       2010  |          1982  |        +5.6%  |       -1.4%  |
|  512 |        1757   |       2241  |          2065  |       +17.5%  |       -7.9%  |
| 1024 |        1484   |       2193  |          1888  |       +27.2%  |      -13.9%  |
| 2048 |        1070   |       1989  |          1511  |       +41.2%  |      -24.0%  |
```

Beobachtungen:

1. **10D ist immer noch besser als scalar (9d.3)**: das beweist dass
   coopmat-Attention ein Win ist. Aber 10D ist KONSISTENT schlechter
   als 10C (QK-only-coopmat).

2. **Regression skaliert mit pp**: bei pp=128 verliert 10D nur 1.4%
   (im Citrix-Noise-Band), bei pp=2048 24%. Je mehr K-Tiles, desto
   mehr akkumuliert sich der Per-Tile-Overhead von 10D.

3. **Brief's Erwartung war "+10-30% über 10C"** — wir messen
   "−1% bis −24%". Vorzeichen-Vorhersage falsch. Größenordnung
   stimmt: der Effekt ist substantiell, aber in die andere Richtung.

### 2.2 Per-K-Tile Bilanz

```
Sprint 10C (QK-cm + PV-scalar):
  K-Load:               1 LDS-write phase + 1 barrier
  QK coopMatStore:      1 phase + 1 barrier
  Softmax:              writes scores_lds[BC] elements + 1 barrier
  Scalar V-sum:         32 global FP16 reads + 256 scalar FMAs/thread
                        no barrier (per-thread independent state)
  End-of-tile barrier:  1 barrier
  Total: 4 barriers, 9 KB LDS

Sprint 10D (QK-cm + PV-cm):
  K-Load:               1 LDS-write phase + 1 barrier
  QK coopMatStore:      1 phase + 1 barrier
  Softmax:              writes p_lds[BC] FP16 + 1 barrier
  V-Load:               32 global FP16 reads + 32 LDS writes + 1 barrier
  PV coopmat (8 stores): 8 KB LDS write to pv_scratch + 1 barrier
  Per-thread accumulate: 32 LDS reads from pv_scratch
  End-of-tile barrier:  1 barrier
  Total: 6 barriers, 21.5 KB LDS

Δ pro Tile:
  +2 barriers
  +12.5 KB LDS pressure (Q+K+S layout reused, +V+P+pv_scratch new)
  +8 KB LDS-writes (pv_scratch coopMatStore)
  +8 KB LDS-reads (per-thread accumulate from pv_scratch)
  +4 KB LDS-writes (V-load to v_lds)
  -- but: PV compute much faster (WMMA vs scalar FMA)
```

Bei **kv_len=2048, Bc=16**: 128 K-Tiles. 128 × 2 = 256 zusätzliche
Barriers pro Q-Tile. Bei **n_heads=32, m=2048**: das sind
∼8.2 Mio Barrier-Operationen pro Forward. Plus die LDS-Bandwidth-
Last.

### 2.3 LDS-Occupancy als Hauptproblem

RDNA4 hat **64 KB LDS pro CU**. WGs schauen sich CU-Resourcen-Pool:

```
Sprint 10C (9 KB/WG):     64/9 = 7 WGs/CU possible
Sprint 10D (21.5 KB/WG):  64/21.5 = 2 WGs/CU
```

Bei 62 CUs auf der RX 9070 XT:

```
Sprint 10C: bis zu 434 WGs in flight
Sprint 10D: 124 WGs in flight (3.5× weniger Parallelität)
```

Die Occupancy-Reduktion erklärt den größten Teil der pp-skalierenden
Regression: bei kürzeren Sequenzen (pp=128) ist auch 10C's 434
WGs nicht voll genutzt (n_heads × q_tiles = 32 × 8 = 256 WGs), bei
pp=2048 ist 10D auf 124 limitiert während 16384 WGs bereit wären.

---

## 3. Was wir gelernt haben

### 3.1 Compute-Win allein reicht nicht

Sprint 10B's microbench zeigte 47.5× WMMA-vs-scalar Compute-Speedup.
Aber: bei einem einzelnen 16×16×16-Fragment (= 1 K-Schritt im
PV-Matmul) ist die WMMA-Issue-Frequenz der einzige Win-Hebel; LDS-
Bandwidth, Barriers und Occupancy bleiben Bottlenecks.

Die Lehre: **Microbench-TFLOPS ≠ End-to-End-Speedup**. Der echte
Hebel war in 10C (QK-coopmat) wo HEAD_DIM/16=8 Inner-Loops die
Per-Thread-Tile-Init-Kosten amortisieren.

### 3.2 KHR-only Pattern hat strukturelle Limits

Ohne `coopMatPerElementNV`/`coopMatReduceNV` (NV-only, nicht
portabel) gibt es nur 3 Wege zwischen coopmat-Fragmenten und
scalar-Code:

a. coopMatStore (Fragment → LDS) + LDS-Scalar-Code + coopMatLoad
   → 2 Round-Trips, viele Barriers
b. Component-wise Ops (Fragment + Fragment, Fragment * Fragment)
   → braucht dass kfac etc. AUCH als Fragment vorliegt
c. Scalar broadcast (`coopmat<...>(scalar_value)`)
   → konstruiert eine Fragment voller `scalar_value`-Werte
   → uniform pro Fragment, nicht per-row

Sprint 10C nutzt (a) für QK-Score-Storage, dann scalar Softmax.
Sprint 10D wollte das auch für PV — aber bei Bc=16 ist die
Compute-Phase zu kurz um die LDS-Roundtrip-Kosten zu amortisieren.

### 3.3 Bc=16 ist zu klein für PV-coopmat

Bei Bc=16 hat das PV-Matmul:
* 1 P-Fragment (16×16) × 8 V-Fragmente = 8 coopMatMulAdd
* P wird einmal geladen, V achtmal
* Der gesamte PV-Compute ist 8 WMMA-Issues

Bei Bc=32 wäre:
* 2 P-Fragmente × 8 V-Fragmente = 16 coopMatMulAdd (2× compute)
* P-Load amortisiert über mehr V-Stores
* Aber: V-LDS würde 8 KB statt 4 KB, scores 2 KB statt 1 KB,
  pv_scratch unverändert bei 8 KB
* Total LDS bei Bc=32: ~31 KB → immer noch 2 WGs/CU

**Bc=64**: noch mehr compute, aber LDS würde 50+ KB → 1 WG/CU.

Es ist möglich dass ein optimaler Punkt zwischen Bc=16 und Bc=32
existiert — Sprint 10F könnte einen Sweep machen.

### 3.4 Sprint 10C's Architektur ist bemerkenswert robust

**Sprint 10C ist der Sweet-Spot:**
* QK-coopmat amortisiert über head_dim=128 (8 Inner-Loops) —
  da kommt der WMMA-Win raus.
* Scalar PV nutzt per-thread Register-Accumulator — kein LDS-
  Roundtrip, kein extra Barrier.
* LDS-Footprint klein (9 KB) → hohe Occupancy.

Die Brief-Vorhersage "PV ist der neue Bottleneck" stimmt
quantitativ: nach 10C ist PV ein größerer Anteil der Attention-
Zeit. Aber den PV mit coopmat zu beschleunigen ist nicht die
gleiche Win-Strategie wie QK-coopmat, weil PV strukturell anders
parallelisiert.

---

## 4. Action Items / Folge-Sprints

### 4.1 Sprint 10E (production polish, kein Code-Win)

* Bench Sprint 10C ohne Citrix für saubere Δ-Validierung.
* End-to-End 5-Prompt + 15-Prompt mit COOPMAT_ATTN=1 für stabiles
  Coherence-Profil.
* Default-ON-Decision basierend auf bereinigten Bench-Zahlen
  (Citrix-frei).

### 4.2 Sprint 10F (Bc-Sweep mit coopmat)

* Variante mit Bc=32: PV hat mehr Inner-Compute → könnte
  Overhead amortisieren.
* Variante mit Bc=64: noch mehr compute, aber LDS-Druck.
* Architektur-Frage: kann man pv_scratch ELIMINIEREN durch
  cleveres Aufteilen der per-thread my_out auf Wave-Lanes?
  (8 O-Segmente × 16 Cells = 128 Cells / 64 Threads = 2 Cells/Thread
  → wave-shuffle?)

### 4.3 Sprint 10G (alternative coopmat-PV-Topologie)

Statt Ansatz C (LDS-Scratch) zu wiederholen: prüfe ob es
möglich ist, **P-Fragment in MULTIPLE coopmat-Fragmenten zu
halten** (Bc=64 würde 4 P-Fragmente bedeuten) und PV als 8 ×
coopMatMulAdd mit längerer K-Achse zu schreiben:

```glsl
for (uint d = 0; d < HEAD_DIM; d += 16) {
    coopmat<float, ..., 16, 16, gl_MatrixUseAccumulator> O_seg = ...(0.0);
    for (uint b = 0; b < Bc / 16; b++) {
        coopMatLoad(P_frag, p_lds, b * 16, BC, RowMajor);
        coopMatLoad(V_frag, v_lds, b * 16 * HEAD_DIM + d, HEAD_DIM, RowMajor);
        O_seg = coopMatMulAdd(P_frag, V_frag, O_seg);
    }
    coopMatStore(O_seg, pv_scratch, d, HEAD_DIM, RowMajor);
}
```

Mit Bc=64: 4 Inner-Iterations × 8 Outer-Loops = 32 coopMatMulAdd
(4× compute), aber pv_scratch-Store-Frequenz halbiert (8 stores
total). Könnte das Verhältnis kippen.

---

## 5. Production-Status nach Revert

```
flash_attn_coopmat.comp  →  identisch zum Sprint-10C-Stand
                            (git checkout reverted die PV-coopmat-Erweiterung)
SPVs in OUT_DIR:
  flash_attn_coopmat.spv         → 56 KB (10C-size, ohne PV-coopmat)
  flash_attn_coopmat_fp16kv.spv  → 56 KB

VULKANFORGE_COOPMAT_ATTN env var:
  unset / "0" → scalar tiled Pfad (default behavior)
  "1"         → 10C's QK-coopmat-Pfad (PV bleibt scalar)

Tests: 167/167 ✓ auf BEIDEN Pfaden
```

Production unverändert seit Sprint 10C bf256cf — nichts gerieben,
nichts schlimmer. Sprint 10D's Lieferung ist:
- **Korrektheits-Beweis** dass PV-coopmat funktioniert
- **Performance-Negativbefund** dass es bei Bc=16 nicht lohnt
- **Architektur-Erkenntnis** über LDS-Roundtrip-Kosten

---

## 6. Files Touched

```
new:      results/v02_sprint10d_pv_coopmat.md (this report)
```

**KEIN Shader-Code committed.** Die experimentelle PV-coopmat-
Modifikation wurde vor dem Commit reverted (`git checkout
vk_shaders/flash_attn_coopmat.comp`). Die exakten Code-Snippets
sind oben in §1.1 für künftige Referenz dokumentiert.

---

## 7. Bottom Line

Sprint 10D war ein **honest-negative Performance-Sprint** —
analog zu Sprint 8b (conditional barriers) und Sprint 9c (rms_norm_mul).
Die Implementierung war erfolgreich (167/167 Tests grün, korrekte
argmax-bit-äquivalente Logits), aber das Ergebnis ist ein
Regression gegenüber Sprint 10C an allen pp.

Die Brief-Vorhersage "PV-coopmat sollte ähnlichen Speedup wie QK
bringen" stimmt nicht für **unsere Bc=16-Konfiguration**. Bei
Bc=16 hat die PV-Phase nur 8 coopMatMulAdd (vs 8 für QK aber QK
hat 1 Output-Fragment vs PV's 8); der Overhead von LDS-Staging +
6 Barriers/Tile + halbierter Occupancy frisst den Compute-Win auf.

Sprint 10C bleibt der Production-Winner. COOPMAT_ATTN Default OFF
bis Sprint 10E (sauberer Bench ohne Citrix). Sprint 10F (Bc-Sweep)
könnte PV-coopmat in einer anderen Konfiguration sinnvoll machen.

**Cumulative Status pp=2048 nach revert:**
* Sprint 8a baseline:                ~810 tok/s
* Sprint 9d.3 (FP16 KV):              1070 tok/s   (+32%)
* Sprint 10C (QK-coopmat, opt-in):    1989 tok/s   (+146% gesamt)
* vs llama.cpp 3771:                  0.53× (war 0.21× nach Sprint 8a)

Sprint 10D ändert daran nichts — der 10C-Win bleibt unangetastet.
