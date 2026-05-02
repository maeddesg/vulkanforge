# v0.2 Sprint 7 — Br>1 Tiled-Q Flash-Attention

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 6 (Flash-Attention-Analyse + negativer
K-LDS-Stage-Versuch, 146/146 Tests)

---

## TL;DR — Br=4 gewinnt 19-72% bei pp 256-1024, opt-in via env-var

```
═══ v0.2 Sprint 7 ═══

Implementiert: flash_attn_tiled.comp (170 LOC). Vier Queries pro
Workgroup teilen sich einen K-Tile-Load (LDS-staged), per-Query
online-softmax in Registern. Identische Bind/PC-Layout zu
flash_attn_batch — Drop-in via VULKANFORGE_FA_TILED=1.

Korrektheit (Phase 2 + Phase 4 Gates ALLE GRÜN):
  • 8 Rust-Reference-Tests: Tiled vs naive max_abs < 1e-4
  • 27 Regression-Tests mit FA_TILED=1: alle grün
  • End-to-End argmax-bit-identisch zu Br=1 auf Qwen3-8B Prefill
  • 146/146 Default + 154/154 mit FA_TILED=1 Tests

Performance — Br=1 (Sprint 5B) vs Br=4 (Sprint 7):

  Single-shot prefill_batch (chunk=pp ≤ 1024):
  | pp   | Br=1 | Br=4 | Δ      |
  |------|------|------|--------|
  |  64  | 1489 | 1400 |  -6%   |
  |  128 | 1641 | 1673 |  +2%   |
  |  256 | 1337 | 1594 | +19% ✅|
  |  512 |  921 | 1323 | +44% ✅|
  | 1024 |  556 |  958 | +72% ✅|

  Chunked prefill (pp > 1024):
  | pp   | Br=1 (chunk=128) | Br=4 (chunk=256) | Δ      |
  |------|------------------|------------------|--------|
  | 2048 |  530             |  476             | -10% ❌|
  | 3072 |  388             |  330             | -15% ❌|
  | 4096 |  303             |  254             | -16% ❌|

WICHTIGER BEFUND — Br=4 ist KEIN globaler Gewinner:
  • Bei single-shot pp=1024 (typische Sweet-Spot): +72%, riesig
  • Bei chunked pp=4096 (RAG, lange Dokumente): -16%, Verlust
  • Grund: K-Tile-LDS-Staging amortisiert nur über kurze KV-Sequenzen.
    Bei langem KV (pp>1024) wächst der Staging-Overhead linear, der
    Gewinn aus 4× weniger Workgroups skaliert nicht mit.

Empfehlung: VULKANFORGE_FA_TILED=1 OPT-IN belassen. Default OFF.
User wählt je nach Workload (kurze multi-turn-chats: ON; lange
RAG/Dokument: OFF).

Files:
  new:      vk_shaders/flash_attn_tiled.comp (170 LOC)
  new:      tests/flash_attn_tiled_ref.rs (303 LOC, 8 tests)
  modified: build.rs (compile new shader)
  modified: src/backend/vulkan/shaders.rs (register ShaderId)
  modified: src/backend/vulkan/pipeline_registry.rs (no-spec entry)
  modified: src/backend/vulkan/forward.rs (run_flash_attn_tiled +
            fa_tiled_enabled field + env-var read + dispatch
            switch)

Tests: 154/154 mit FA_TILED=1, 154/154 default (Phase-2 ref-tests
zählen in beiden).
Commit: HEAD (kein Push; mehrere Phase-Commits vorher: phase2 +
final).
```

---

## 1. Was wurde gebaut?

### 1.1 Phase 2: Rust-Referenz (`tests/flash_attn_tiled_ref.rs`)

303 LOC, 8 Tests. Zwei Implementierungen:

(a) `naive_attn`: materialisiert die volle N×kv_len Score-Matrix,
    direkt verifizierbar durch Lesen.

(b) `flash_attn_tiled_ref`: Q-Tile/K-Tile-Loop mit online-softmax,
    spiegelt das Algorithmus-Skelett des geplanten GLSL-Shaders.

Tests vergleichen tiled gegen naive und decken ab:
* `fa_ref_small_no_causal`: hd=128, seq=16, kv=16, br=4, bc=4
* `fa_ref_medium_no_causal`: br=16, bc=32
* `fa_ref_causal_short`: causal=true
* `fa_ref_causal_chunked`: q_start=64, kv_len=96 (chunk-2 sim)
* `fa_ref_partial_q_tile`: seq=30 (nicht durch Br=16 teilbar)
* `fa_ref_partial_kv_tile`: kv=50 (nicht durch Bc=32 teilbar)
* `fa_ref_br_invariance`: gleiche Inputs, br∈{1,4,8,16}, bc∈{8,16,32,64}
* `fa_ref_long_kv`: seq=128, kv_len=1152, q_start=1024 (Sprint-5B-style)

Resultat: max_abs < 1e-4 in allen Fällen. **Algorithmus ist
korrekt, vor jedem GLSL-Schreiben verifiziert.**

### 1.2 Phase 3: GLSL Shader (`vk_shaders/flash_attn_tiled.comp`)

170 LOC. Konservatives Design (BR=4 statt 16, einfache Wave64-WG):

```
Const:
  BR       = 4  (queries per workgroup)
  WGSIZE   = 64 (single Wave64)
  TILE     = 64 (Bc — each thread = one K-position)
  HEAD_DIM = 128 (Qwen3 / Llama-3.1 / Mistral-7B / DSR1)

LDS:
  q_lds      = BR × HEAD_DIM × 4 = 4 × 128 × 4 = 2 KB
  k_lds      = TILE × HEAD_DIM × 4 = 64 × 128 × 4 = 32 KB
  scores_lds = BR × TILE × 4 = 4 × 64 × 4 = 1 KB
  Total: 35 KB (RDNA4 Limit: 64 KB → komfortabel)

Register-State pro Thread:
  my_max[BR=4], my_sum[BR=4], my_out0[BR=4], my_out1[BR=4]
  → 16 Floats Running-State pro Thread
```

Algorithmus:
1. Q-Tile Load (BR×HEAD_DIM Floats von Global → q_lds, BLOCK-cooperative)
2. Per-K-Tile-Loop:
   a) K-Tile Cooperative-Load (TILE×HEAD_DIM → k_lds, wave-coalesced)
   b) Per-Query Score-Compute aus Q-LDS und K-LDS
   c) Per-Query Online-Softmax-Update (subgroupMax, subgroupAdd)
   d) V-Sum aus Global (V-LDS würde 64KB überschreiten)
3. Per-Query Normalize + Write-back

**KEIN coopmat, KEIN FP16 K/V, KEIN D_split.** Bewusst konservativ
(im Gegensatz zu llama.cpp's 788 LOC mit 12 Spec-Constants und
4 Code-Pfaden). Der Punkt von Sprint 7 ist Br>1 ALLEIN als
isolierter Hebel zu validieren.

### 1.3 Integration: `forward.rs` + ShaderId-Wiring

* `ShaderId::FlashAttnTiled` in `shaders.rs` registriert
* `pipeline_registry.rs` Entry (no spec constants — alles
  hardcoded im Shader)
* `forward.rs::run_flash_attn_tiled` als Geschwister-Funktion
  zu `run_flash_attn_batch` (identische Signatur, anderer ShaderId
  und Dispatch `(n_heads, ceil(m/4), 1)`)
* `Forward::fa_tiled_enabled: bool`-Feld + Env-Var-Read in
  `new_with_prefill`
* Im `prefill_batch` Layer-Loop: bei `fa_tiled_enabled=true` ruft
  `run_flash_attn_tiled` statt `run_flash_attn_batch`. Default OFF.

---

## 2. Korrektheit (KRITISCH)

### 2.1 Phase 2 Gate: Rust-Reference vs Naive

```
$ cargo test --release --test flash_attn_tiled_ref
running 8 tests
test fa_ref_small_no_causal ... ok
test fa_ref_medium_no_causal ... ok
test fa_ref_causal_short ... ok
test fa_ref_causal_chunked ... ok
test fa_ref_partial_q_tile ... ok
test fa_ref_partial_kv_tile ... ok
test fa_ref_br_invariance ... ok
test fa_ref_long_kv ... ok
test result: ok. 8 passed
```

Alle Diffs max_abs < 1e-4 vs naive Implementation. Algorithmus
korrekt vor GLSL-Schreiben.

### 2.2 Phase 4 Gate: GPU vs Br=1 End-to-End

```
$ VULKANFORGE_FA_TILED=1 cargo test --release
test result: ok. 24 passed (lib unit)
test result: ok.  9 passed (dequant_q4k)
test result: ok. 18 passed (gguf)
test result: ok. 60 passed (kv_cache)
test result: ok.  8 passed (q4k_quant)
test result: ok.  8 passed (flash_attn_tiled_ref)
test result: ok. 27 passed (regression)
                ────
                154 / 154 ✓
```

Existierende `phase5b2_batch_attn_parity_qwen3_short` und
`phase5b2_batch_attn_parity_qwen3_two_tiles` Tests laufen unter
`VULKANFORGE_FA_TILED=1` und die argmax-Werte stimmen bit-genau
mit den Br=1-Erwartungswerten überein.

### 2.3 Default-Pfad weiterhin grün

```
$ cargo test --release    # Default: Br=1 wie vorher
test result: ok. 154 passed   (146 alt + 8 neue ref-tests)
```

---

## 3. Performance — vollständige Matrix

### 3.1 Single-shot Mode (chunk = pp, max_pp ≥ pp)

`run_pp_bench` direkt-Mikro-Bench, 5 Repetitionen, Median.

```
| pp   | Sprint 5B (Br=1) | Sprint 7 (Br=4) | Δ      |
|------|------------------|-----------------|--------|
|  64  |     1489 tok/s   |    1400 tok/s   |  -6%   |
|  128 |     1641 tok/s   |    1673 tok/s   |  +2%   |
|  256 |     1337 tok/s   |    1594 tok/s   | +19% ✅|
|  512 |      921 tok/s   |    1323 tok/s   | +44% ✅|
| 1024 |      556 tok/s   |     958 tok/s   | +72% ✅|
```

**Großer Gewinn bei pp 256-1024.** Genau der Bereich der typische
multi-turn-chat-Sessions abdeckt (system-prompt + history + neue
user-message ≈ 200-1000 Tokens).

### 3.2 Chunked Mode (pp > 1024)

```
| pp   | Br=1 chunk=128 | Br=4 chunk=128 | Br=4 chunk=256 |
|------|----------------|----------------|----------------|
| 2048 |   530 tok/s    |   450 tok/s    |   476 tok/s    |
| 3072 |   388 tok/s    |   318 tok/s    |   330 tok/s    |
| 4096 |   303 tok/s    |   247 tok/s    |   254 tok/s    |
| Best |     530        |       —        |     476        |
| Δ    |    baseline    |      —         |     -10%       |
```

**Br=4 verliert bei chunked pp.** Hauptursache: K-Tile-LDS-Staging-
Overhead skaliert linear mit kv_len, der Gewinn aus 4× weniger
Workgroups bleibt konstant. Bei kv_len > ~1500 wird Staging-Cost
dominant.

### 3.3 Vergleich gegen llama.cpp (build 408225b)

```
| pp   | VF Br=1 | VF Br=4 | Best VF | llama.cpp | Best/llama |
|------|---------|---------|---------|-----------|------------|
|  64  |  1489   |  1400   |  1489   |   2286    |  0.65×     |
|  128 |  1641   |  1673   |  1673   |   3603    |  0.46×     |
|  256 |  1337   |  1594   |  1594   |   3999    |  0.40×     |
|  512 |   921   |  1323   |  1323   |   4317    |  0.31× ✅  |
| 1024 |   556   |   958   |   958   |   4189    |  0.23× ✅  |
| 2048 |   530   |   476   |   530   |   3771    |  0.14×     |
| 4096 |   303   |   254   |   303   |   3272    |  0.09×     |
```

**Bei pp=1024**: 0.13× → **0.23×** (1.7× näher an llama.cpp).
**Bei pp=512**: 0.21× → **0.31×** (~50% näher).

Der pp=4096-Gap bleibt bei 0.09× weil Br=4 dort verliert.

---

## 4. Warum Br=4 bei chunked pp verliert

Theorie: Sprint-7-Shader stage K-Tile in LDS um die strided-Read-
Pattern zu coalesce. Sprint 6 hatte die GLEICHE K-LDS-Staging
versucht im Br=1-Shader und 16-22% Regression gemessen — RDNA4's
L1-Cache absorbiert die strided Reads bereits, Staging fügt
Mehrarbeit hinzu.

In Sprint 7's Br=4-Shader ist K-LDS-Staging trotzdem nötig, weil
4 Queries den gleichen K-Tile lesen sollen (sonst keine 4×
Reuse). Der Trade-off:
* Single-shot pp ≤ 1024: 4× WG-Overhead-Amortisation > Staging-Cost → Win
* Chunked pp > 1024: Staging-Cost wächst linear mit kv_len, Gewinn skaliert nicht → Loss

Mögliche Sprint-8-Gegenmaßnahme: Br=4 ohne K-LDS-Staging — Q-Tile
in LDS, K direkt aus global gelesen für jeden Query separat.
Würde 4× Q-Reads sparen (Q war eh klein), 0× K-Reads sparen. Vermutlich
kein signifikanter Win, weil Q nie der Bottleneck war.

Bessere Sprint-8: dynamische BR-Wahl basierend auf pp:
* pp ≤ 64: BR=1 (current path)
* pp 128-1024: BR=4 (Sprint 7)
* pp > 1024 (chunked): BR=1 (current path)

---

## 5. Was BLEIBT in Sprint 7

* `VULKANFORGE_FA_TILED=1` als Opt-In für User mit Sweet-Spot-pp
  (256-1024).
* Phase-2 Rust-Referenz als ground truth für künftige Sprints
  (Sprint 8 mit Br=8 oder Br=16, oder andere Tile-Konstellationen).
* Sauberes Drop-in-Pattern: identische Bind/PC, nur Shader-ID
  und Dispatch-Shape ändern sich.

---

## 6. Empfehlung für Sprint 8

Nicht "Br=16 mit komplexerer LDS-Aufteilung" — das Risiko/Aufwand-
Verhältnis ist schlecht (siehe llama.cpp's 788 LOC für nur Br>1).

**Sprint 8a — Dynamic BR Selection** (1 Tag):
* Im `forward.rs` BR=1 vs BR=4 wählen abhängig von `seq_len`:
  ```
  if seq_len >= 256 && seq_len <= 1024 {
      run_flash_attn_tiled(...)  // Br=4
  } else {
      run_flash_attn_batch(...)  // Br=1
  }
  ```
* Default ON (kein Opt-In mehr nötig).
* Bench: bestätigen dass alle pp-Bereiche gewinnen oder gleich
  schnell bleiben.

**Sprint 8b — Conditional Barriers** (1 Tag):
* Aus dem Audit: 5-15% prefill-weite Gewinne durch Tracking welche
  Buffer-Reads/Writes wirklich überlappen. Niedrigster Aufwand,
  betrifft alle pp-Bereiche.

**Sprint 9+ — FP16 KV-Cache** (2-3 Tage):
* Halbiert KV-Memory-Bandbreite, hilft v.a. bei chunked pp > 1024
  wo wir aktuell verlieren.
* Erfordert Format-Migration aller KV-Reader, Numerik-Validierung.

NICHT priorisieren:
* Br>4 (Br=16): VGPR-Pressure-Killer ohne klaren Win-Pfad.
* coopmat-Attention (`flash_attn_cm2`): hochkomplex, nur bei FP16/BF16
  K/V sinnvoll.
