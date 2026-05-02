# v0.2 Sprint 8a — Flash-Attention Tiled Default ON

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 7.6 (Br=16 Bc=32 wins everywhere ≥128, 154/154 Tests)

---

## TL;DR — 1-Zeilen-Default-Flip. +91% bei pp=512, +164% bei pp=1024 ohne Env-Var.

```
═══ v0.2 Sprint 8a ═══

Änderung:  forward.rs::new_with_prefill — fa_tiled_enabled
           default false → true. Env-Var-Semantik invertiert
           (vorher: =1 → ON; jetzt: =0 → OFF).
Opt-out:   VULKANFORGE_FA_TILED=0

Spot-Check (default ON, ohne env-var):
  pp=128:  1830 tok/s  (Sprint 7.6 mit FA_TILED=1: 1816 → noise)
  pp=512:  1761 tok/s  (exakt match)
  pp=1024: 1482 tok/s  (vs 1469 → noise)

5-Prompt-Bench (Sprint 5B-style pp=20-62):
  Median prefill: 729 tok/s   (Sprint 5 Br=1 baseline: ~745)
  Median decode:   87 tok/s   (unchanged)
  → kleine Regression (~2%) bei winzigen pp, akzeptabel.

Regression:
  default ON:                    154/154 ✓
  VULKANFORGE_FA_TILED=0 (off):  154/154 ✓ (= Sprint 7.6 OFF Pfad)

Net effect für User die nichts setzen:
  pp=128:   1641 → 1830 (+12%)
  pp=512:    921 → 1761 (+91%)
  pp=1024:   556 → 1482 (+164%)
  pp=2048:   530 → ~810 (chunked, +53%)
  pp=4096:   303 → ~450 (chunked, +49%)

Files:
  modified: src/backend/vulkan/forward.rs (1 line + comment)
  new:      results/v02_sprint8a_default_on.md

Commit: HEAD (kein Push).
```

---

## 1. Was wurde geändert

Genau eine Code-Stelle, `forward.rs::new_with_prefill`:

```rust
// Vorher (Sprint 7.6):
let fa_tiled_enabled = match std::env::var("VULKANFORGE_FA_TILED") {
    Ok(s) => s == "1",          // ON nur bei explizit "1"
    Err(_) => false,            // Default OFF
};

// Nachher (Sprint 8a):
let fa_tiled_enabled = match std::env::var("VULKANFORGE_FA_TILED") {
    Ok(s) => s != "0",          // OFF nur bei explizit "0"
    Err(_) => true,             // Default ON
};
```

Migrationskompatibilität:
* User mit `VULKANFORGE_FA_TILED=1` in ihren Skripten → weiterhin
  ON (s != "0" ist true).
* User die nichts setzen → vorher OFF, jetzt ON.
* User mit `VULKANFORGE_FA_TILED=0` → OFF (neue Opt-Out-Schiene).
* Andere Werte (`true`, `yes`, leerer String): pragmatisch ON
  (alle != "0").

---

## 2. Begründung

Sprint-7.6-Daten zeigen eindeutig dass Br=16 Bc=32 in jeder
realistischen pp-Range gewinnt:

```
| pp   | Br=1 (off) | Br=16 Bc=32 (on) | Δ        |
|------|------------|------------------|----------|
|  16  |   386      |    375           |  -2.8%   |
|  32  |   762      |    744           |  -2.4%   |
|  48  |  1115      |   1074           |  -3.7%   |
|  64  |  1489      |   1431           |  -3.9%   |
| 128  |  1641      |   1816           | +10.7%   |
| 256  |  1337      |   1890           | +41.4%   |
| 512  |   921      |   1761           | +91.2%   |
| 1024 |   556      |   1469           | +164%    |
```

Trade-off:
* pp ≤ 64: -2.4 bis -3.9% (innerhalb 5% Mess-Rauschen, kein
  realistischer Workload-Effekt — pp=64 prefill braucht ~45 ms
  also ~2 ms Verlust).
* pp ≥ 128: +11% bis +164% (substantiell, betrifft jede
  Multi-Turn-Chat-Session und jede RAG/Document-Query).

Der Cross-over-Punkt liegt zwischen pp=64 und pp=128. User mit
ausschließlich pp ≤ 64 Workloads (sehr kurze Prompts, ohne System-
Prompt) können `VULKANFORGE_FA_TILED=0` setzen.

---

## 3. Tests

### 3.1 Default ON

```
$ cargo test --release
test result: ok. 24 passed       (lib unit)
test result: ok.  9 passed       (dequant_q4k)
test result: ok. 18 passed       (gguf)
test result: ok. 60 passed       (kv_cache)
test result: ok.  8 passed       (q4k_quant)
test result: ok.  8 passed       (flash_attn_tiled_ref)
test result: ok. 27 passed       (regression)
                ────
                154 / 154 ✓
```

Diese 27 Regression-Tests beinhalten u.a. die End-to-End
phase5b2-Parity (Multi-Tile-Causal) und sprint5b-Chunked-Parity
(q_start>0). Beide vergleichen argmax — Br=16 Bc=32 produziert
bit-identische Vorhersagen wie Br=1.

### 3.2 Explicit OFF

```
$ VULKANFORGE_FA_TILED=0 cargo test --release
                ────
                154 / 154 ✓
```

Opt-Out-Pfad funktioniert: User die explizit OFF wollen bekommen
das Sprint-7.6-OFF Verhalten (= Sprint-5B-Br=1).

---

## 4. Bench

### 4.1 5-Prompt Bench (default ON)

```
| Prompt          | pp | gen | prefill tok/s | decode tok/s |
|-----------------|----|----|---------------|--------------|
| Greeting        | 20 |  64 |     372.4     |     88.1     |
| Simple Sequence | 31 |  64 |     715.1     |     88.0     |
| Prime Check     | 31 | 256 |     728.7     |     87.3     |
| LRU Cache       | 47 | 512 |    1073.9     |     86.6     |
| REST API        | 62 |1024 |    1400.6     |     79.7     |

MEDIAN prefill: 728.7 tok/s
MEDIAN decode:   87.3 tok/s
```

Vergleich zu Sprint-5-Baseline (Br=1):
* Median prefill: 728 (heute) vs 745 (Sprint 5) → -2%
* Median decode: 87 (heute) vs 89 (Sprint 5) → -2%

Beides innerhalb Mess-Variance. Keine signifikante Regression
für die kurzen 5-Prompt-Bench-Workloads.

### 4.2 pp Spot-Check (default ON)

```
| pp   | tok/s (8a) | tok/s (Sprint 7.6 FA_TILED=1) |
|------|------------|-------------------------------|
|  128 |   1829.5   |          1816                 |
|  512 |   1761.4   |          1761                 |
| 1024 |   1482.0   |          1469                 |
```

Identisch zu Sprint 7.6 mit explizitem FA_TILED=1 — der Default-
Flip funktioniert wie erwartet.

---

## 5. Was bleibt offen

Sprint 8a war eine 1-Zeilen-Änderung. Die nächsten Hebel sind
substantielle Algorithmen-Sprünge:

* **Sprint 8b — Conditional Barriers** (1 Tag): aus dem Audit
  5-15% predicted über alle pp.
* **Sprint 9 — FP16 KV-Cache** (2-3 Tage): besonders relevant
  für pp ≥ 2048 wo wir bei 0.14-0.22× von llama.cpp sind.
* **Sprint 10 — coopmat_cm2 für Attention** (mehrere Tage): nur
  sinnvoll nach FP16-KV.

Aktueller Stand vs llama.cpp (mit Default ON):
```
| pp   | VF Default (8a) | llama.cpp | VF/llama |
|------|-----------------|-----------|----------|
|   64 |  1431           |   2286    |  0.63×   |
|  128 |  1830           |   3603    |  0.51×   |
|  256 |  1890           |   3999    |  0.47×   |
|  512 |  1761           |   4317    |  0.41×   |
| 1024 |  1482           |   4189    |  0.35×   |
| 2048 |  ~810           |   3771    |  0.21×   |
| 4096 |  ~450           |   3272    |  0.14×   |
```

Sprint 5B → Sprint 8a hat den pp=1024-Gap von 0.13× auf 0.35×
geschoben — **2.7× näher an llama.cpp** ohne Algorithmus-Wechsel,
nur durch incrementelle Tile-Tuning + Cap-Lift + Chunked-Prefill.
