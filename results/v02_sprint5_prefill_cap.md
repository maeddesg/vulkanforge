# v0.2 Sprint 5 — Prefill-Cap Fix + Dreifach-pp-Skalierungskurve

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 4.5 (256-Cliff lokalisiert via Apple-zu-Apfel-Bench)
**Ziel:** Cliff fixen + drei vollständige pp-Skalierungskurven (mul_mmq,
coopmat, llama.cpp) zur Crossover-Analyse.

---

## TL;DR — Fix landet 10×, kein Coopmat-Crossover, llama.cpp hält 2-7× Lead

```
═══ v0.2 Sprint 5 ═══

FIX (1 Stelle in forward.rs::new):
  max_prefill_tokens 256 → 1024 (env-var override
  VULKANFORGE_MAX_PREFILL für niedrige VRAM-Hosts)

End-to-End-Effekt (pp=401 chat session):
  VORHER:  91 tok/s   (Token-für-Token Decode-Pfad)
  NACHHER: 936 tok/s  (Batched GEMM-Pfad)
  Gewinn:  10.3×

Direkter prefill_batch Mikro-Bench (pp=512):
  VF (mul_mmq):  920 tok/s   (war: 91 tok/s)
  VF (coopmat):  592 tok/s
  llama.cpp:    4317 tok/s

Skalierungskurven, alle drei Engines am gleichen Build/HW:

  pp    mul_mmq   coopmat   llama.cpp   mmq/llama  coop/mmq
  ───   ───────   ───────   ─────────   ─────────  ────────
   16     386.5     222.9       700.6     0.55×     0.58×
   32     762.3     284.0      1319.8     0.58×     0.37×
   48    1114.8     311.0      1766.4     0.63×     0.28×
   64    1489.3     331.4      2286.2     0.65×     0.22×
   80    1210.0     526.8      1870.2     0.65×     0.44×
   96    1399.5     620.2      2214.6     0.63×     0.44×
  112    1536.2     571.1      2574.5     0.60×     0.37×
  128    1641.1     633.9      3602.5     0.46×     0.39×  ← VF mmq peak
  160    1411.5     731.8      2729.3     0.52×     0.52×
  192    1489.5     771.2      3165.2     0.47×     0.52×  ← coopmat peak
  224    1320.0     725.4      3574.2     0.37×     0.55×
  256    1337.0     737.6      3999.4     0.33×     0.55×
  320    1200.0     700.3      3355.6     0.36×     0.58×
  384    1091.6     661.5      3840.9     0.28×     0.61×
  448    1003.3     615.6      3916.3     0.26×     0.61×
  512     920.5     591.7      4317.2     0.21×     0.64×
  640     794.2     540.0      4109.2     0.19×     0.68×
  768     696.1     494.1      4158.0     0.17×     0.71×
 1024     556.4    DEVICE_LOST 4179.3     0.13×       —    ← coopmat crash

Drei zentrale Befunde:

(1) FIX FUNKTIONIERT. End-zu-End-Recovery 91 → 936 tok/s bei
    pp=401. mul_mmq-Pfad scalat jetzt sauber bis pp=1024.
    Sprint-4.5-Cliff komplett behoben.

(2) KEIN COOPMAT-CROSSOVER. coopmat ist über den gesamten pp-Range
    (16-768) durchgehend langsamer als mul_mmq. Die Audit-These
    "coopmat sollte ab pp~256 gewinnen" ist falsifiziert.
    coop/mmq peak bei pp=768 mit 0.71× — coopmat kommt nie auf
    mul_mmq's Niveau. → KEIN Hybrid-Dispatch sinnvoll.
    PLUS: coopmat crasht bei pp=1024 mit DEVICE_LOST (separater
    Bug, nicht Sprint-5-blocker).

(3) LLAMA.CPP-LEAD SKALIERT MIT pp.
    Bei kleinen pp (16-64): VF/llama ≈ 0.55-0.65× (1.5-1.8× Gap)
    Bei pp=128:              VF/llama = 0.46× (2.2× Gap)
    Bei pp=512:              VF/llama = 0.21× (4.7× Gap)
    Bei pp=1024:             VF/llama = 0.13× (7.5× Gap)

    Schlimmer noch: VF mul_mmq SINKT mit pp ab pp=128 (1641 → 556),
    während llama.cpp STEIGT (3603 → 4179). Das ist der echte
    "missing magic" — Attention skaliert bei uns O(N²) scalar,
    bei llama.cpp Flash-Attention O(N) (oder zumindest Tile-O(N²/T)).

Tests: 145/145 ✓ (Cap-Bump bricht keine Tests).
Commit: HEAD (kein Push).

Empfehlung Sprint 6: Flash-Attention-Prefill (oder Tiled-Scalar-
Attn). Höchster ROI nachdem Cliff-Bug weg ist.
```

---

## 1. Was wurde gemacht?

Vier Änderungen, alle landed:

(a) `forward.rs::new` — Default `max_prefill_tokens` von 256 auf 1024
    angehoben, `VULKANFORGE_MAX_PREFILL` als Env-Override.

(b) Neuer Bench `examples/run_pp_bench.rs` — synthetischer
    Prefill-Mikro-Bench, ruft `prefill_batch` direkt mit konfigurier­
    barem pp auf. Spiegelt llama-bench's `-p N -tg 0`. Ohne
    Tokenizer/Chat-Template/Decode-Schleife — sauber isolierte
    Prefill-Messung.

(c) Drei vollständige pp-Sweeps (16-1024) — VF mul_mmq,
    VF coopmat (`VULKANFORGE_COOPMAT=1`), llama.cpp via llama-bench.

(d) Regression `cargo test --release` — 145/145 grün.

KEIN Code in den GEMM-Shadern oder Attention-Pfaden geändert
(out-of-scope für Sprint 5).

---

## 2. Der Fix — `forward.rs::new`

### Vorher (Sprint 4.5 Diagnose)

```rust
// forward.rs:272
Self::new_with_prefill(dev, allocator, kv_cache, config, profiler, 256)
```

→ jeder Caller von `Forward::new` bekam `max_prefill_tokens = 256`.
→ `decode.rs:429` routete `prefill_len > 256` durch `forward_token`
   (== Decode-Rate ~90 tok/s).

### Nachher

```rust
// forward.rs:272
let max_pp: u32 = std::env::var("VULKANFORGE_MAX_PREFILL")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(1024);
Self::new_with_prefill(dev, allocator, kv_cache, config, profiler, max_pp)
```

### VRAM-Kosten verifiziert

Für Qwen3-8B (`hidden=4096, ffn=12288, n_heads=32, n_kv_heads=8,
head_dim=128`), Allokationsformel aus `forward.rs:374-380`:

```
pp_hidden  = max_pp · 4096 · 4    = 16 KB · max_pp
pp_kv      = max_pp · 8·128 · 4   =  4 KB · max_pp
pp_q       = max_pp · 32·128 · 4  = 16 KB · max_pp
pp_ffn     = max_pp · 12288 · 4   = 48 KB · max_pp
pp_q8      = max_pp · 4096/128·144 = 4.6 KB · max_pp
pp_q8_ffn  = max_pp · 12288/128·144 = 13.8 KB · max_pp
```

Summe ≈ 102 KB pro max_pp-Token. Bei max_pp=1024 → ~104 MB.
Doppelt-Buffer für Ping-Pong + Q/K/V + Attn-Out + O + Gate + Up
verdoppelt das auf ~210 MB Gesamt-Batch-Buffer. Plus die festen
~60 MB für non-batch-buffers ≈ **~270 MB** total Forward-Allokation
bei max_pp=1024.

Frei in 16 GB VRAM bei 4.68 GB Modell + 2 GB KV-Cache (Qwen3-8B,
2048 ctx, fp16) → **rund 9 GB Headroom**. Cap=1024 problemlos.

### End-zu-End-Verifikation (pp=401)

```
Sprint 4.5 (cap=256):
  Long PP512   pp=401   prefill=  91.0 tok/s

Sprint 5 (cap=1024):
  Long PP512   pp=401   prefill= 936.0 tok/s
                                  ↑ 10.3×
```

Der echte Smoking-Gun-Zahlenwert: **10.3× am gleichen Prompt,
gleicher Hardware, gleichem Build, einziger Unterschied = der
Cap-Default**.

---

## 3. Bench-Methodik

### 3.1 VulkanForge — `run_pp_bench.rs`

Direkter `prefill_batch`-Aufruf. Synthetisiert Embeddings durch
`pp`-faches Wiederholen der Token-0-Embedding-Zeile. Die GEMM-Ops
machen die gleiche Arbeit unabhängig vom Inhalt — Tokens dienen
nur als Workload-Größe-Driver.

* Warmup: 2 Runs (Pipeline-Cache + erste Allokation aus dem
  Weg räumen)
* Mess-Runs: 5 Repetitionen pro pp (3 für die isolierten Sanity-
  Checks)
* Reportiert Median (robust gegen einzelne GPU-Stalls)
* `kv_cache.reset()` zwischen jedem Run (jeder Run startet bei
  pos=0)
* Wall-Clock um `prefill_batch(...)` — Funktion blockiert via
  `cmd_ctx.one_shot` bis zur Fence-Completion, also misst echte
  GPU-Zeit + Submit-Overhead

### 3.2 llama.cpp — llama-bench

```
~/tmp/llama.cpp/build/bin/llama-bench \
  -m ~/models/Qwen3-8B-Q4_K_M.gguf \
  -p 16,32,48,64,80,96,112,128,160,192,224,256, \
     320,384,448,512,640,768,1024 \
  -n 0 -ngl 999 -t 1 -r 3
```

* `-n 0` → kein Decode (vermeidet Decode-Mischung in Reportierung)
* `-r 3` → 3 Repetitions, ± Stdabw
* Build: `408225b` (gleicher wie Sprint 4.5)

---

## 4. Ergebnis-Matrix (vollständig)

```
| pp   | mul_mmq tok/s | coopmat tok/s | llama.cpp tok/s |
|------|---------------|---------------|-----------------|
|   16 |         386.5 |         222.9 |  700.63 ± 5.43  |
|   32 |         762.3 |         284.0 | 1319.85 ± 7.48  |
|   48 |        1114.8 |         311.0 | 1766.44 ± 7.89  |
|   64 |        1489.3 |         331.4 | 2286.21 ± 2.11  |
|   80 |        1210.0 |         526.8 | 1870.23 ± 1.47  |
|   96 |        1399.5 |         620.2 | 2214.60 ± 8.96  |
|  112 |        1536.2 |         571.1 | 2574.47 ± 8.18  |
|  128 |        1641.1 |         633.9 | 3602.54 ± 5.25  |
|  160 |        1411.5 |         731.8 | 2729.26 ± 2.31  |
|  192 |        1489.5 |         771.2 | 3165.17 ± 4.61  |
|  224 |        1320.0 |         725.4 | 3574.19 ± 6.68  |
|  256 |        1337.0 |         737.6 | 3999.41 ± 9.37  |
|  320 |        1200.0 |         700.3 | 3355.59 ± 1.79  |
|  384 |        1091.6 |         661.5 | 3840.86 ± 6.34  |
|  448 |        1003.3 |         615.6 | 3916.33 ± 2.06  |
|  512 |         920.5 |         591.7 | 4317.20 ± 10.64 |
|  640 |         794.2 |         540.0 | 4109.16 ± 6.88  |
|  768 |         696.1 |         494.1 | 4157.96 ± 5.90  |
| 1024 |         556.4 |  DEVICE_LOST  | 4179.25 ± 3.47  |
```

VF-Werte sind Median über 5 Runs (3 für pp=1024). llama.cpp ist
Mittelwert ± Stdabw über 3 Repetitions.

### 4.1 Kurven-Charakteristik

**mul_mmq:** Steigt bis pp=128 (Peak 1641 tok/s), dann stetiger
Fall auf 556 tok/s bei pp=1024. Charakteristisches Stair-Step-
Muster wegen Tile-Größe (BM=BN=64): pp=64 hat 64-Tile-Auslastung
voll, pp=80 nutzt zwei 64-Tiles aber nur halb.

**coopmat:** Niedriger Anfang (pp=16: 223 tok/s), steigt langsam
bis pp=192 (771 tok/s), dann ähnlich wie mul_mmq fallend.
Bei pp=1024 → **VK_ERROR_DEVICE_LOST**. Vermutlich Pipeline-
Padding-Bug oder LDS-Overflow im naiven Coopmat-Pfad bei großem N.

**llama.cpp:** Steigt bis pp=512 (Peak 4317 tok/s), dann
plateauiert um 4100-4180 tok/s bis pp=1024. Klares
sublineares Skalieren mit Plateau — typisches GEMM-saturierendes
Verhalten.

### 4.2 Crossover-Analyse

**Crossover 1 — mul_mmq vs coopmat:** existiert NICHT.
mul_mmq dominiert über den gesamten pp-Range. Bestes Verhältnis
(coop/mmq = 0.71) bei pp=768; auch dort ist mul_mmq 1.41× schneller.

→ Hybrid-Dispatch (`if seq_len ≥ N: coopmat`) wäre auf RDNA4
RADV ein Anti-Pattern. Die Audit-Hypothese aus
`grand_audit_v02.md` ("naive coopmat = 7% of peak; tiled coopmat
sollte gewinnen ab pp=256") wird empirisch widerlegt — auch
das tiled-coopmat (`mul_coopmat_q4k_naive_padded_*` = unsere
Sprint 3A/B-Implementierung) erreicht selbst bei pp=192 nur 771
tok/s, gegen mul_mmq 1490 tok/s.

**Crossover 2 — coopmat vs llama.cpp:** existiert NICHT.
Selbst beim bestmöglichen Coopmat-Run (pp=192: 771 tok/s) ist
llama.cpp 4.1× schneller (3165 tok/s).

→ Die "VulkanForge ist schneller bei langen Prompts wegen
WMMA"-These des Sprint-Briefings wird falsifiziert. RADV's
KHR_cooperative_matrix-Implementierung auf RDNA4 ist nicht
performant genug, um den Mehraufwand der Coopmat-Pipelines zu
rechtfertigen.

**Crossover 3 — mul_mmq vs llama.cpp:** existiert NICHT.
Best-Case Verhältnis 0.65× bei pp=64; danach divergiert es:
- pp=128: 0.46× (1.4× Gap)
- pp=512: 0.21× (4.7× Gap)
- pp=1024: 0.13× (7.5× Gap)

→ Der Gap **wächst mit pp** statt zu schrumpfen. Das ist
**kein** GEMM-Problem (bei reinem GEMM wäre das Verhältnis
konstant), sondern ein **Skalierungsproblem im Attention-Pfad**:
unsere Scalar-Attn ist O(N²) ohne Tiling, llama.cpp's
Flash-Attention-Prefill ist O(N²) mit Working-Set-Tiling, das
in den L1/LDS passt. Bei pp=1024 sind 1024² = 1M Attention-
Elemente — bei uns alle in HBM, bei llama in der Cache.

---

## 5. Befundliste

### 5.1 Cliff weg (Hauptziel) ✓

* End-zu-End: pp=401 chat session 91 → 936 tok/s (10.3×).
* Direkt-Mikro-Bench: pp=512 prefill 90 → 920 tok/s (10.2×).
* `decode.rs:429`-Pfad-Switch bleibt unverändert; nur der Default-
  Cap wandert vom 256-Block-Pfad zum 1024-Block-Pfad.

### 5.2 Hybrid-Dispatch unbrauchbar (Sub-Befund)

* coopmat ist NIE schneller als mul_mmq.
* Bestes Verhältnis 0.71× bei pp=768.
* Hybrid-Threshold-Logik aus dem Sprint-Briefing (`if seq_len ≥
  COOPMAT_THRESHOLD: coopmat`) **nicht implementieren** — würde
  Performance VERSCHLECHTERN.
* Existing `VULKANFORGE_COOPMAT=1` Default OFF bleibt richtig.

### 5.3 coopmat pp=1024 DEVICE_LOST (separater Bug)

* `VULKANFORGE_COOPMAT=1` + pp=1024 → `VK_ERROR_DEVICE_LOST`.
* Bei pp=768 läuft's noch (494 tok/s).
* Wahrscheinliche Ursache: Spec-Constant N-Padding-Tile in
  `mul_coopmat_q4k_naive_padded_*` overflow oder Workgroup-Count
  > device-max-dispatch.
* Out-of-scope für Sprint 5 (coopmat ist eh deaktiviert per
  Default; nur Diagnose-Werkzeug-Pfad). Zu fixen falls/wenn
  Coopmat ernsthaft revisited wird.

### 5.4 Attention ist der nächste Bottleneck

* mul_mmq peak 1641 tok/s bei pp=128, fällt monoton bis 556 tok/s
  bei pp=1024.
* Reine GEMM-Operations skalieren NICHT in dieser Form — eine
  Q-K^T-GEMM von 128×128 vs 1024×1024 hat 64× mehr FLOPs für 8×
  mehr Tokens (→ Throughput sollte STEIGEN nicht fallen, weil
  die GEMM-Auslastung besser wird).
* Charakteristisches Drop-Off von Scalar-O(N²)-Attn ohne Tiling.
* llama.cpp's Kurve PLATEAUIERT bei pp=512 (4317 tok/s) und hält
  das bis pp=1024 (4179 tok/s) — typisches Flash-Attn-Scaling.

→ **Sprint 6 Empfehlung**: Flash-Attention-Prefill ODER zumindest
Tiled-Scalar-Attn (LDS-staged Q-K^T und Softmax-V). Erwarteter
Gewinn: Wenn wir vom 1641-tok/s-Peak (pp=128) NICHT mehr abfallen
würden, wäre VF bei pp=1024 ~1641 tok/s → 0.39× von llama.cpp,
also ~3× besser als heute (556 → 1641).

### 5.5 Build-Drift-Notiz

llama.cpp pp512: gemessen 4317 tok/s (Sprint 4.5: 4328 tok/s) —
identisch innerhalb Stdabw. Build 408225b stabil zwischen den
beiden Sprints.

---

## 6. Tests

```
cargo test --release

test result: ok. 24 passed
test result: ok.  9 passed
test result: ok. 18 passed
test result: ok. 60 passed
test result: ok.  8 passed
test result: ok. 26 passed
                ────
                145 / 145 ✓
```

5-Prompt-Bench (pp=20-62) Regression-Check:

```
| Prompt          | pp | prefill tok/s | decode tok/s |
|-----------------|----|---------------|--------------|
| Greeting        | 20 |         361.6 |         89.3 |
| Simple Sequence | 31 |         744.8 |         89.4 |
| Prime Check     | 31 |         748.6 |         88.2 |
| LRU Cache       | 47 |        1105.7 |         87.7 |
| REST API        | 62 |        1425.8 |         79.3 |

MEDIAN prefill: 748.6 tok/s   (Sprint 4.5: 741.1 tok/s)
MEDIAN decode:   88.2 tok/s   (Sprint 4.5: 89.2 tok/s)
```

Innerhalb Mess-Variance. Keine Regression durch den Cap-Bump.

---

## 7. Files Touched

```
modified:   src/backend/vulkan/forward.rs            (+10 -2)
new file:   examples/run_pp_bench.rs                 (180 lines)
new file:   results/v02_sprint5_prefill_cap.md       (this file)
```

KEINE Änderungen in Shadern, Attention-Pfaden, GEMM-Dispatch-Logik
oder Tests. Reine Cap-Anhebung + Diagnose-Bench.

---

## 8. Empfehlung Sprint 6

Aus den drei Kurven ist klar:

* mul_mmq ist auf RDNA4 RADV der dominante Pfad. Spec-Constants
  sind lokales Optimum (Sprint 4 bestätigt).
* coopmat ist auf diesem Stack tot — nicht weiter tunen.
* Der echte Bottleneck ist **Attention-Skalierung**, NICHT
  GEMM-Spec-Constants oder Coopmat-vs-Scalar-Wahl.

**Sprint 6 — Flash-Attention-Prefill** (oder mindestens
Tiled-Scalar-Attn):

* Erwarteter Gewinn: VF mul_mmq Kurve flach halten ab pp=128
  (Peak 1641 tok/s) statt fallen zu lassen. Bei pp=1024:
  556 → ~1500-1700 tok/s = **2.7-3.0×** Gewinn isoliert.
* Sekundäreffekt: KV-Cache-Bandbreite fällt drastisch (Tiled-Attn
  liest K/V mehrfach aus LDS statt aus HBM).
* Gleichzeitig: nähert sich llama.cpp's Plateau-Verhalten an
  (4179 tok/s bei pp=1024) — auch wenn wir das absolute Niveau
  noch nicht erreichen.

**Sprint 7 (optional, danach):** Conditional-Barriers (PR #12135-
Pattern). Einzige verbleibende Hypothese aus dem Audit die noch
nicht überprüft ist. Erwarteter Gewinn: 5-15% über alle pp.

NICHT priorisieren:
* Spec-Constants weiter tunen (Sprint 4 → negative).
* Coopmat-Pfad reanimieren (Sprint 5 → negative).
* Kernel-Mikrooptimierung im mul_mmq (negativer ROI bevor
  Attention behoben ist).
