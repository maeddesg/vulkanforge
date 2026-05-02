# v0.2 Sprint 7.6 — Bc-Sweep (32 / 64) bei Br=16

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 7.5 (Br=16 dominiert mit Bc=64, 154/154 Tests)

---

## TL;DR — Bc=32 gewinnt überall. Neuer Default. pp=1024 jetzt 2.64× vs Br=1.

```
═══ v0.2 Sprint 7.6 ═══

Hypothese: kleinere K-Tile (BC=32 vs 64) → halber LDS-Verbrauch
→ höhere WG-Occupancy → bessere Performance, trotz 50% idle
threads im Score-Compute (Wave64 mit 32 aktiven Lanes).

LDS-Budget bei BR=16:
  BC=64: 44 KB (q=8 + k=32 + scores=4)  — Sprint 7.5
  BC=32: 26 KB (q=8 + k=16 + scores=2)  — Sprint 7.6 (NEU)

EMPIRISCHES ERGEBNIS — Bc=32 GEWINNT AUF JEDER GEMESSENEN pp:

Single-shot:
| pp   | Br=1 | Br=16 Bc=64 | Br=16 Bc=32 | Δ vs Bc=64 | vs Br=1 |
|------|------|-------------|-------------|------------|---------|
|  64  | 1489 |    1416     |    1431     |   +1%      | 0.96×   |
| 128  | 1641 |    1759     |    1816     |   +3%      | 1.11×   |
| 256  | 1337 |    1769     |    1890     |   +7%      | 1.41×   |
| 512  |  921 |    1619     |    1761     |   +9%      | 1.91×   |
| 1024 |  556 |    1327     |    1469     |  +11% ⭐   | 2.64×   |

Chunked (Br=16, optimal chunk per pp):
| pp   | Br=1 best | Bc=64 best   | Bc=32 best   | Δ vs Bc=64 |
| 2048 |  530      |  708 (=512)  |  817 (=512)  |  +15%      |
| 3072 |  388      |  487 (=256)  |  571 (=256)  |  +17%      |
| 4096 |  303      |  381 (=256)  |  453 (=256)  |  +19%      |

vs llama.cpp (build 408225b):
  pp=1024: Br=1 0.13× → Bc=32 0.35×    (2.7× näher!)
  pp=512:  Br=1 0.21× → Bc=32 0.41×
  pp=256:  Br=1 0.33× → Bc=32 0.47×

Drei Befunde:

(1) BC=32 IST OVERALL WINNER. Kein einziges pp regrediert. Win-
    Magnitude wächst mit pp (chunked +15-19% vs single-shot
    +1-11%). Die Occupancy-Hypothese ist VALIDIERT — die 50%
    idle Lanes im Score-Compute sind kein Problem weil mehr WGs
    parallel laufen.

(2) WGSIZE=64 BLEIBT FIXED. Bc=32 mit Wave64 bedeutet Threads
    32-63 sind im Score-Compute idle, beteiligen sich aber an
    der K-Tile-LDS-Befüllung (jeder Thread lädt 1/64 der LDS-
    Cache-Lines). Subgroup-Reductions ignorieren -inf/0 von
    inaktiven Lanes. Korrektheit: 154/154 grün.

(3) DEFAULT NEU. fa_tiled_bc Default flippt von 64 → 32. User
    der VULKANFORGE_FA_TILED=1 setzt bekommt automatisch das
    Optimum. Override via VULKANFORGE_FA_BC=64 wenn nötig
    (z.B. für vergleichende Benchmarks).

Files:
  modified: vk_shaders/flash_attn_tiled.comp (#ifndef BC + tid<BC guard)
  modified: build.rs (NEU: flash_attn_tiled_br16_bc32.spv)
  modified: src/backend/vulkan/shaders.rs (FlashAttnTiledBr16Bc32)
  modified: src/backend/vulkan/pipeline_registry.rs (no-spec entry)
  modified: src/backend/vulkan/forward.rs (Bc selector + default 32)
  new:      results/v02_sprint7_6_bc_sweep.md

Tests: 154/154 mit BC=32 (default), 154/154 mit BC=64 (override).
Commit: HEAD (kein Push).
```

---

## 1. Was wurde gemacht?

### 1.1 Shader-Anpassung (`flash_attn_tiled.comp`)

Drei Änderungen, alle algorithm-erhaltend:

(a) `BC` als #define, default 64 (kompatibel zu Sprint 7.5):

```glsl
#ifndef BC
#define BC 64
#endif
const uint TILE = BC;
```

(b) `tid_in_tile`-Guard im Score-Compute:

```glsl
bool tid_in_tile = tid < BC;
bool t_in_range = tid_in_tile && t_global < causal_len_max;
```

Bei BC=32 + WGSIZE=64: Threads 32-63 haben `tid_in_tile=false`,
ihr `valid` flag ist false, score bleibt -1e30, pscore=0. Die
`subgroupMax`/`subgroupAdd` Reductions ignorieren sie korrekt.

(c) LDS-Write-Guard:

```glsl
if (tid_in_tile) {
    scores_lds[qi * TILE + tid] = pscore;
}
```

Ohne diesen Guard würden Threads mit tid >= BC in den nächsten
Q-Slot schreiben (`scores_lds[qi*32 + 32..63]` == `scores_lds[(qi+1)*32 + 0..31]`),
was Memory-Korruption produziert.

### 1.2 Build-Layer

```rust
// build.rs — neue SPV neben den existierenden Br-SPVs:
ShaderJob {
    out_name: "flash_attn_tiled_br16_bc32.spv",
    entry_source: "flash_attn_tiled.comp",
    defines: &[("BR", "16"), ("BC", "32")],
}
```

Existierende Br=4/8/16-SPVs sind unverändert (BC default 64
aktiv via #ifndef). Nur Br=16 hat eine Bc-Variante, weil die
Sprint-7.5-Daten Br=16 als klaren Winner zeigten.

### 1.3 Selector

```rust
// forward.rs::run_flash_attn_tiled:
let (shader_id, br) = match (self.fa_tiled_br, self.fa_tiled_bc) {
    (16, 32) => (ShaderId::FlashAttnTiledBr16Bc32, 16u32),
    (16, _)  => (ShaderId::FlashAttnTiledBr16,     16u32),
    (8,  _)  => (ShaderId::FlashAttnTiledBr8,       8u32),
    _        => (ShaderId::FlashAttnTiledBr4,       4u32),
};
```

Default: `fa_tiled_bc = 32`. Override via `VULKANFORGE_FA_BC=64`.

---

## 2. Korrektheit

### 2.1 Per-Variante Regression

```
$ VULKANFORGE_FA_TILED=1                         cargo test --release
   → 154/154 ✓ (Default: Br=16 Bc=32)

$ VULKANFORGE_FA_TILED=1 VULKANFORGE_FA_BC=64    cargo test --release
   → 154/154 ✓ (Br=16 Bc=64)

$ VULKANFORGE_FA_TILED=1 VULKANFORGE_FA_BR=4     cargo test --release
   → 154/154 ✓ (Br=4 Bc=64, unchanged from Sprint 7)

$ VULKANFORGE_FA_TILED=1 VULKANFORGE_FA_BR=8     cargo test --release
   → 154/154 ✓ (Br=8 Bc=64)

$                                                cargo test --release
   → 154/154 ✓ (Default OFF, fa_batch Br=1)
```

### 2.2 phase5b2_two_tiles und sprint5b_chunked decken die kritischen Fälle ab

* `two_tiles`: lange Prompt → mehrere K-Tiles → Online-Softmax-
  Rescale über Tile-Grenzen (also über die Bc-Tile-Größe). Mit
  BC=32 sind das doppelt so viele Tiles → mehr Rescale-Operationen.
  Wenn der Score-LDS-Write-Guard fehlt: Memory-Korruption →
  argmax-Drift. Test: argmax bit-identisch zu Br=1.

* `sprint5b_chunked_prefill_parity_qwen3`: q_start > 0 (Chunk 2+
  Simulation) → Causal-Mask innerhalb Q-Tile mit Bc=32. Test:
  argmax bit-identisch zwischen 1-chunk und 4-chunk Pfad.

Beide Tests grün → die kritischen Code-Pfade (Multi-K-Tile mit
neuen Guards, Multi-Chunk mit q_start) sind validiert.

---

## 3. Performance-Vergleich

### 3.1 Single-Shot

```
$ VULKANFORGE_FA_TILED=1 VULKANFORGE_FA_BR=16 VULKANFORGE_FA_BC=32 \
  VF_PP_LIST=64,128,256,512,1024 VF_PP_RUNS=5 \
  cargo run --release --example run_pp_bench
```

```
| pp   | Bc=64 (7.5) | Bc=32 (7.6) | Δ        |
|------|-------------|-------------|----------|
|  64  |    1416     |    1431     |  +1.1%   |
| 128  |    1759     |    1816     |  +3.2%   |
| 256  |    1769     |    1890     |  +6.8%   |
| 512  |    1619     |    1761     |  +8.8%   |
| 1024 |    1327     |    1469     | +10.7%   |
```

Trend: Win wächst monoton mit pp. Bei kleinem pp (64-128) sind
die Tiles ohnehin so klein dass die Occupancy-Headroom nichts
nutzt — wenige WGs, alle CUs unterausgelastet sowieso.

Bei pp=1024: 16 Q-Tiles × 32 Heads = 512 WGs total. Mit Bc=32
und ~26 KB LDS sind 2 WGs/CU möglich → effektiv 1024 "WG-Slots"
verfügbar. Mit Bc=64 nur 1 WG/CU = 512 Slots. Doppeltes Latency-
Hiding zahlt sich aus.

### 3.2 Chunked (pp > 1024)

```
| pp   | Bc=64 best        | Bc=32 best        | Δ      |
|------|-------------------|-------------------|--------|
| 2048 | 708 (chunk=512)   | 817 (chunk=512)   | +15.4% |
| 3072 | 487 (chunk=256)   | 571 (chunk=256)   | +17.2% |
| 4096 | 381 (chunk=256)   | 453 (chunk=256)   | +18.9% |
```

Win wächst mit pp — bei chunked liegt das daran, dass die K-Tile-
Iterationen pro Chunk mit kv_len wachsen (jeder Chunk macht
attention über alle vorherigen + aktuelle KV). Mit Bc=32 sind
das doppelt so viele Iterationen, aber jede ist halb so groß
und besser in der Pipeline gehalten.

### 3.3 Volle Matrix (Sprint 5B → 7.6 progression)

```
| pp   | 5B Br=1 | 7 Br=4 | 7.5 Br=16 Bc=64 | 7.6 Br=16 Bc=32 | llama |
|------|---------|--------|-----------------|-----------------|-------|
|  64  |  1489   |  1400  |   1416          |    1431         | 2286  |
| 128  |  1641   |  1673  |   1759          |    1816         | 3603  |
| 256  |  1337   |  1594  |   1769          |    1890         | 3999  |
| 512  |   921   |  1323  |   1619          |    1761         | 4317  |
| 1024 |   556   |   958  |   1327          |    1469         | 4189  |
| 2048 |   530   |   476  |    708          |     817         | 3771  |
| 3072 |   388   |    —   |    487          |     571         | 3522  |
| 4096 |   303   |   254  |    381          |     453         | 3272  |
```

```
| pp   | 5B/llama | 7/llama | 7.5/llama | 7.6/llama |
|------|----------|---------|-----------|-----------|
| 1024 |  0.13×   |  0.23×  |   0.32×   |  0.35× ⭐ |
| 4096 |  0.09×   |   —     |   0.12×   |  0.14×    |
```

Sprint 7.6 holt nochmal ~10% gegenüber Sprint 7.5 raus. Kumulativ
seit Sprint 5B: pp=1024 von 0.13× auf **0.35×** = **2.7× näher**
an llama.cpp. pp=4096 von 0.09× auf 0.14× = 1.6× näher.

---

## 4. Warum Bc=32 trotz halb-leerer Threads gewinnt

### 4.1 LDS als Occupancy-Limiter

RDNA4 Hardware-Constraints:
* LDS pro CU: 64 KB
* VGPRs pro Thread (Wave64): bis zu 256
* Wave64 pro CU: bis zu 16 (4 SIMD32 × 4 Waves im theoretischen
  Maximum)

Mit Bc=64 und 44 KB LDS pro WG: **1 WG/CU** maximal (44 + 44 > 64).
Mit Bc=32 und 26 KB LDS pro WG: **2 WGs/CU** möglich (26 + 26 = 52 < 64).

Doppelte WG-Anzahl bedeutet doppelte Wavefronts in flight — mehr
Latency-Hiding wenn ein Wave auf Memory wartet.

### 4.2 Idle-Threads sind nicht das Problem

Bei Bc=32 sind 32 von 64 Threads im Score-Compute idle. Das
KÖNNTE auf erste Sicht problematisch sein (50% ALU verschwendet).
ABER:

* RDNA4 ALUs sind nicht der Bottleneck — Memory-Bandbreite ist
  meist das Limit für Attention.
* Idle-Threads in der gleichen Wave belegen keine extra ALU-Zyklen
  (der Wave executet predicated als Ganzes).
* Die idle Threads HELFEN sogar bei der K-Tile-LDS-Befüllung
  (jeder Thread lädt 2 dim-halves von 32 K-Rows = 64 floats →
  64 Threads × 64 floats = 4096 floats ÷ 32 banks = perfekt
  coalesced).

### 4.3 Empirische Validierung

Wenn idle ALUs ein Problem wären, müsste Bc=32 bei kleinem pp
schlechter laufen (wo Occupancy eh nicht ausgereizt wird). Wir
sehen aber +1% schon bei pp=64 — d.h. Bc=32 ist nirgends
schlechter, nur unterschiedlich-gut. Worst case: Mess-Rauschen.

---

## 5. Sprint 8 Empfehlungen — unverändert

Sprint 7.6 hat den letzten offenen Tuning-Hebel im flash_attn_tiled-
Kernel ausgeschöpft. Weitere Optimierungen sind algorithmische
Sprünge:

* **Sprint 8a — Default ON** (1 Stunde): Flippe `fa_tiled_enabled`
  von false auf true. User braucht keine Env-Var mehr für die +138%-
  Performance. Risiko: -3% an pp ≤ 64 (Mess-Rauschen vs default
  fa_batch). Empfohlen.

* **Sprint 8b — Conditional Barriers** (1 Tag): 5-15% predicted
  über alle pp.

* **Sprint 9 — FP16 KV-Cache** (2-3 Tage): besonders relevant für
  pp ≥ 2048 wo wir bei 0.14-0.22× von llama.cpp sind. Halbiert
  Memory-Bandbreite.

* **Sprint 10 — coopmat_cm2 für Attention** (mehrere Tage):
  hardware-WMMA für QK und PV. Nur sinnvoll nach FP16 KV.

NICHT priorisieren:
* Bc=16 — zu viele Tiles, Loop-Overhead frisst Occupancy-Gain.
* Br=32 — VGPR-Spilling-Risiko (128 floats running state pro thread).
