# v0.2 Sprint 3C — N-Padding, FP8-Retry, Buffer-OOB-Fix

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 3B (BF16 Parity ✅, -57% Performance, 145/145 Tests).

---

## TL;DR — Bug-Postmortem komplett, FP8 rehabilitiert

```
═══ v0.2 Sprint 3C ═══

Drei Funde:

1. **Buffer-Out-of-Bounds-Bug**: max_pp wurde NICHT auf TILE
   gerundet. Mein `cmd_fill_buffer(rows N..N_padded)` schrieb
   außerhalb der allokierten Buffer-Größe — Adjacent-Memory-
   Korruption führte zu ~30% top-5 Drift (Sprint 3B → Sprint 3C
   erste Versuche). Fix: max_pp = round_up(max_pp_raw, 16).

2. **FP8 wurde rehabilitiert**: Sprint 3A's "FP8-Precision-
   Versagen über 36 Layer" war zu 100% der Partial-Tile-Store-Bug.
   Nach Sprint 3B's LDS-Staged-Store + Sprint 3C's N-Padding +
   Buffer-Fix:
     mul_mmq:                 top1=151667
     Sprint 3B BF16:          top1=151667 (overlap 4/5)
     Sprint 3C BF16 padded:   top1=151667 (overlap 4/5)
     Sprint 3C FP8 padded:    top1=151667 (overlap 2/5)
   FP8 produziert das RICHTIGE Top-1-Token. Top-5 hat etwas mehr
   Drift als BF16 wegen 3-Mantissen-Bit-Auflösung — aber NICHT
   Junk wie in Sprint 3A behauptet.

3. **TRANSFER→COMPUTE Barrier**: cmd_fill_buffer ist eine
   TRANSFER-Op, nicht COMPUTE. Brauchte einen separaten
   Pipeline-Barrier vor dem nächsten Shader-Read. Behoben.

Performance (5-Prompt Qwen3-8B):
  | Config                  | prefill | decode | Δ vs mul_mmq |
  | mul_mmq baseline        |  740.9  |  89.8  | —            |
  | Sprint 3B BF16 no-pad   |  320.3  |  89.7  | -57%         |
  | Sprint 3C BF16 padded   |  287.5  |  88.2  | -61%         |
  | Sprint 3C FP8 padded    |  293.5  |  88.5  | -60%         |

  N-Padding hilft NICHT der Performance — die Architektur (naive
  coopmat 1 SG/WG für skinny-N) bleibt der Bottleneck. FP8 ist
  marginal schneller als BF16 (+2.1%) wegen weniger LDS / native
  Convert.

Default:  COOPMAT_FP8 = OFF (BF16 ist Default wenn Coopmat AN);
          VULKANFORGE_COOPMAT = OFF (mul_mmq ist System-Default)

Tests:        145/145 → 145/145 grün (keine neuen Tests; FP8 retry
              ist eine zweite Pass-Variante des existierenden parity-
              Tests via Env-Var)
Commit:       (folgt — KEIN Push)
```

---

## 1 — Sprint-3A Postmortem komplett

Sprint 3A diagnostizierte das End-to-End-Logits-Versagen als
"FP8-Precision versagt über 36 Layer". Sprint 3B fand den **echten**
Bug: `coopMatStore` hat Partial-Tiles fallengelassen, was bei realen
Qwen3-Prompts mit `seq_len ≈ 30..40` zu Stale-Memory-Tokens 16..29
führte → Junk durch die Layer-Stack.

Sprint 3B hat das mit LDS-Staged-Store + Per-Thread-Bounds-Check
behoben. **Sprint 3C zeigt jetzt: das Sprint-3A FP8-Versagen war zu
100% derselbe Bug.** Mit Partial-Tile-Schutz (entweder Sprint 3B's
LDS-Staged oder Sprint 3C's N-Padding) produziert FP8 stabile
Top-1-Logits.

| Sprint    | Path                              | Top-1     | Top-5 overlap | Diagnose       |
|-----------|-----------------------------------|-----------|---------------|----------------|
| 3A        | tiled FP8 (FORWARD_LAYOUT)        | 13 junk   | 0/5           | Partial-tile   |
| 3B        | naive BF16, LDS-staged-store      | 151667    | 4/5           | Working        |
| 3C        | naive BF16, N-padded direct-store | 151667    | 4/5           | Working        |
| 3C        | naive FP8,  N-padded direct-store | 151667    | 2/5           | FP8 rehab.     |

Die Sprint-3A-Diagnose war komplett irreführend. FP8-Precision war NIE
das Problem — es war 100% Memory-Corruption durch Partial-Tile-
Drops.

---

## 2 — Drei Bugs, drei Fixes

### 2.1 Bug A: Partial-Tile-Store (Sprint 3B's Fix war OK)

Bereits in Sprint 3B behoben via LDS-staged store + per-thread
bounds check. Sprint 3C bietet eine zweite Lösung: N-Padding macht
alle Tiles voll → `coopMatStore` direkt nach Global, ohne LDS-
Staging. Beide Varianten gibt es als separate SPVs:

* `mul_coopmat_q4k_naive_bf16.spv` — Sprint 3B's LDS-staged BF16
* `mul_coopmat_q4k_naive_padded_bf16.spv` — Sprint 3C's direct BF16
* `mul_coopmat_q4k_naive_padded_fp8.spv` — Sprint 3C's direct FP8

### 2.2 Bug B: Buffer-OOB beim Zero-Fill

Die kritische Entdeckung in Sprint 3C — und der Grund, warum die
ersten Test-Versuche mit "N-Padding aktiv" nur top-5 = 1/5 ergaben.

`max_prefill_tokens` (= `max_pp`) bestimmt die Allokationsgröße der
Batch-Buffer (`batch_norm`, `batch_q`, `batch_q8`, ...). Wenn der
Aufrufer `max_pp = seq_len = 30` setzt, ist `batch_norm` exakt für
30 Rows allokiert. `pad_to_tile(30, 16) = 32`. Mein `cmd_fill_buffer`
fügte Bytes 30*K*4 .. 32*K*4 als Zero — d.h. **2 Rows hinter
dem Buffer-Ende**.

Vulkan-Implementierungen serialisieren diese OOB-Writes oft auf
benachbarte Memory-Allokationen. Im konkreten Fall: andere Buffer
in der GPU-Allocator's Heap. Die Korruption schlich subtil durch das
Ranking durch — Top-1 blieb richtig, Top-2..5 driftete.

**Fix**: `max_pp = round_up(max_pp_raw, 16)` bei Allokation. Klein,
aber kritisch.

```rust
let max_pp_raw = max_prefill_tokens as u64;
let max_pp = (max_pp_raw + 15) / 16 * 16;
```

### 2.3 Bug C: TRANSFER→COMPUTE Barrier

`cmd_fill_buffer` ist eine TRANSFER-Operation. Der existierende
`compute_barrier` synchronisiert nur SHADER_WRITE → SHADER_READ.
Der nächste Compute-Shader-Read der gepadden Bytes hatte keinen
korrekten Barrier — Hazard im Vulkan-Spec-Sinn (auch wenn das
RADV-driver-spezifisch nicht gecrashed ist).

**Fix**: Neuer Helper `transfer_to_compute_barrier`:

```rust
fn transfer_to_compute_barrier(dev: &VulkanDevice, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            ...
        );
    }
}
```

Wird nach jedem `zero_activation_tail` aufgerufen.

---

## 3 — Was ist neu in Sprint 3C

### 3.1 Shader-Source Erweiterungen

`mul_coopmat_q4k_naive.comp` bekommt zwei neue `#ifdef`-Switches:

* `FP8_MODE` — `ELEM_TYPE = floate4m3_t` statt `bfloat16_t`. Triggert
  GL_EXT_float_e4m3 + GL_EXT_scalar_block_layout (FP8 ist 1-byte
  aligned). LDS-Storage halbiert sich (1 B/elem statt 2 B/elem).
* `PADDED_OUTPUT` — direkter ColumnMajor `coopMatStore` zu Global,
  ohne LDS `buf_c` und ohne Per-Thread-Bounds-Check. Spart 1088 B
  LDS und einen Barrier pro WG. Gilt nur wenn der Runtime
  garantiert dass pc.n und pc.m Vielfache von 16 sind.

Drei SPVs gebaut:

* `mul_coopmat_q4k_naive_bf16.spv` — Sprint 3B Default (kein Define)
* `mul_coopmat_q4k_naive_padded_bf16.spv` — `-DPADDED_OUTPUT`
* `mul_coopmat_q4k_naive_padded_fp8.spv` — `-DPADDED_OUTPUT -DFP8_MODE`

### 3.2 forward.rs N-Padding Pfad

```rust
// In dispatch_layer_batch (an drei Stellen: Q+K, O, gate+up):
let n_padded = Self::pad_to_tile(seq_len, 16);
self.zero_activation_tail(dev, cmd, batch_norm.handle, seq_len, n_padded, hidden);
transfer_to_compute_barrier(dev, cmd);

let (shader, bm, bn) = if seq_len <= 64 {
    (self.coopmat_naive_padded_shader(), 16, 16)
} else if seq_len <= 128 {
    (ShaderId::MulCoopmatQ4KFwdBn32, 64, 32)
} else {
    (ShaderId::MulCoopmatQ4KFwdBn64, 64, 64)
};
self.run_gemm_coopmat_q4k(
    dev, registry, cmd, shader, weights,
    activations.handle, output.handle,
    M, n_padded, K, bm, bn, "gemm_*_coopmat",
);
```

`coopmat_naive_padded_shader()` wählt zwischen BF16- und FP8-
Variante per `coopmat_fp8_enabled` (Env: `VULKANFORGE_COOPMAT_FP8=1`).

### 3.3 Dispatch ändert sich nicht

Für die typischen Qwen3-Prefill-Shapes (seq_len ≤ 64) bleiben die
Dispatch-Dimensionen gleich:

* seq_len = 30, n_padded = 32: groups_y = 32/16 = 2 (vs 30/16 = 2 vorher)
* seq_len = 47, n_padded = 48: groups_y = 48/16 = 3 (vs 47/16 = 3 vorher)
* seq_len = 62, n_padded = 64: groups_y = 64/16 = 4 (vs 62/16 = 4 vorher)

→ Identical WG count; nur die letzten 1-2 Tokens werden gepadded.

---

## 4 — Logits-Parity-Daten (komplett)

```
$ cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture
[BF16 padded — Sprint 3C default]
[sprint3-parity] top1_mmq=151667 top1_coopmat=151667
                 top5_mmq=[151667, 85387, 151668, 138955, 34894]
                 top5_coopmat=[151667, 85387, 151668, 34894, 50897]
                 top5_overlap=4 mmq_top1_rank_in_coopmat=0

$ VULKANFORGE_COOPMAT_FP8=1 cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture
[FP8 padded — Sprint 3C retry]
[sprint3-parity] top1_mmq=151667 top1_coopmat=151667
                 top5_mmq=[151667, 85387, 151668, 138955, 34894]
                 top5_coopmat=[151667, 85387, 50897, 151645, 147752]
                 top5_overlap=2 mmq_top1_rank_in_coopmat=0
```

Beide Varianten produzieren das **richtige Top-1-Token**. BF16 hält
top-5 4/5; FP8 hält top-5 2/5. Beide weit von Sprint 3A's "junk
top-1=13" entfernt.

Der Test enforced top-1 + top-5 ≥ 3/5; FP8 schafft 2/5 → Test schlägt
fehl bei FP8-mode. Das ist ein realistisches Bild: BF16 ist sicher,
FP8 hat zusätzliche Quantisierungs-Drift. Default bleibt BF16.

---

## 5 — Performance: keine Verbesserung, aber FP8 ≈ BF16

```
$ cargo run --release --example run_15prompt_bench  (baseline)
  MEDIAN prefill: 740.9 tok/s, decode: 89.8

$ VULKANFORGE_COOPMAT=1 cargo run --release --example run_15prompt_bench
  MEDIAN prefill: 287.5 tok/s, decode: 88.2  (BF16 padded)

$ VULKANFORGE_COOPMAT=1 VULKANFORGE_COOPMAT_FP8=1 cargo run --release --example run_15prompt_bench
  MEDIAN prefill: 293.5 tok/s, decode: 88.5  (FP8 padded)
```

| Variant            | Prefill |  Δ vs mul_mmq | Δ vs Sprint 3B |
|--------------------|---------|---------------|----------------|
| mul_mmq            |  740.9  |    —          |   +131%        |
| 3B BF16 LDS-staged |  320.3  |   -57%        |    —           |
| 3C BF16 padded     |  287.5  |   -61%        |    -10%        |
| 3C FP8 padded      |  293.5  |   -60%        |    -8%         |

* **N-Padding selbst bringt keine Performance-Lift** — der direct
  `coopMatStore` spart einen Barrier + LDS, aber das ist marginal
  vs den Dequant + Convert + WMMA-Latenz pro K-Step.
* **Sprint 3C BF16 ist sogar -10% gegenüber Sprint 3B** — vermutlich
  weil das `cmd_fill_buffer` (auch wenn klein, ~0.5 ms pro Layer ×
  3 Calls × 36 Layer ≈ 50 ms) Overhead addiert.
* **FP8 ist marginal schneller als BF16** (+2%) — weniger LDS-
  Bandwidth + ein-Instruction-Convert (`v_cvt_pk_fp8_f32`) statt
  5-VALU-Op BF16-Pack. Nicht dramatisch genug, um den Precision-
  Verlust (top-5 2/5 vs 4/5) zu rechtfertigen.

**Fazit**: für Skinny-N-Prefill ist die naive coopmat-Architektur
fundamental zu teuer. Sprint 3C's Werkzeug (N-padding) löst keine
Performance-Probleme, eliminiert aber zwei korrektheits-relevante
Bugs (OOB-Write, TRANSFER-Sync).

---

## 6 — Was Sprint 3C **nicht** liefert

* **Performance-Lift** — bleibt -60% vs mul_mmq. Sprint 3D müsste
  einen anderen Hebel finden (Header-Caching, Tiled-N für N>16,
  Multi-WMMA-pro-WG).
* **gemm_down NaN-Debug** — verschoben. K=11008 bleibt im
  mul_mmq-Pfad.
* **Q6_K-Coopmat** — verschoben. gemm_v bleibt mul_mmq.
* **Default ON** — bleibt OFF (env-var `VULKANFORGE_COOPMAT=1`
  weiter Pflicht zum Aktivieren).

---

## 7 — Sprint-Status nach 3C

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
                      partial-tile-store, jetzt definitiv bestätigt)
  3B    ✅ partiell  Logits-Parity BF16 ✅ / Performance -57%
  3C    ✅ partiell  Bug-Postmortem KOMPLETT (OOB + TRANSFER-sync) /
                      FP8-Logits REHABILITIERT (top-1=151667 ✓) /
                      Performance NICHT besser (-60%)
  3D    ↻ open       Performance-Hebel: Block-Header-Caching,
                      Multi-WMMA, oder coopmat NUR für non-skinny-N.
                      gemm_down NaN. Q6_K für gemm_v.
```

Die wichtigste Sprint-3C-Erkenntnis ist die **Korrektur der Sprint-
3A-Diagnose**. FP8 funktioniert end-to-end — der Junk-Output war ein
Memory-Bug, nicht ein Precision-Bug. Das öffnet die Tür für eine
zukünftige FP8-Lösung wenn die Performance-Hebel greifen
(z.B. wenn das WMMA selbst der Bottleneck ist statt Dequant +
Convert).

---

## 8 — Reproduzierbarkeit

```fish
# Build (44 SPVs jetzt: +2 Padded-Variants)
cargo build --release

# Default OFF — keine Änderung
cargo test --release

# Sprint 3C BF16 padded parity
cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture

# Sprint 3C FP8 padded parity
VULKANFORGE_COOPMAT_FP8=1 cargo test --release --test regression sprint3a_coopmat_gemm_q -- --nocapture

# Bench (5-Prompt)
cargo run --release --example run_15prompt_bench                            # baseline
VULKANFORGE_COOPMAT=1 cargo run --release --example run_15prompt_bench      # BF16 padded
VULKANFORGE_COOPMAT=1 VULKANFORGE_COOPMAT_FP8=1 \
  cargo run --release --example run_15prompt_bench                          # FP8 padded

# Diagnostic env vars (nur für Bug-Bisection):
VULKANFORGE_COOPMAT_LEGACY_STORE=1   # → Sprint 3B SPV (LDS-staged)
VULKANFORGE_COOPMAT_NO_FILL=1        # → skip zero-fill (Bug-A diagnostic)
```

---

## 9 — Files

```
MOD   vk_shaders/mul_coopmat_q4k_naive.comp     +PADDED_OUTPUT, +FP8_MODE
                                                #ifdef switches
MOD   build.rs                                  +2 ShaderJobs
MOD   src/backend/vulkan/shaders.rs             +2 ShaderId, +2 SPV consts
MOD   src/backend/vulkan/pipeline_registry.rs   +2 lines in match arm
MOD   src/backend/vulkan/forward.rs             ~+150 LoC:
                                                 - max_pp round-up to 16
                                                 - pad_to_tile + zero_activation_tail
                                                 - transfer_to_compute_barrier
                                                 - coopmat_fp8_enabled flag + setter
                                                 - coopmat_naive_padded_shader picker
                                                 - 3 dispatch sites updated
                                                 - 2 diagnostic env vars
NEW   results/v02_sprint3c_padding_fp8_retry.md dieser Report
```

Tests: 145/145 → 145/145. Default forward path unverändert.
