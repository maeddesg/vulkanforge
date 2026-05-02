# v0.2 Sprint 8b — Conditional Barriers (Honest Negative Result)

**Datum:** 2026-04-28
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 8a (default ON, 154/154 Tests)

---

## TL;DR — Codebase ist bereits barrier-optimal. Keine sicheren Entfernungen.

```
═══ v0.2 Sprint 8b ═══

Hypothese (Sprint-Briefing):
  ~612 Barriers/Forward → ~468 (-24%) durch Entfernung der
  Barriers zwischen unabhängigen GEMMs (gemm_q→gemm_k usw.).

EMPIRISCHE ANALYSE:
  Die Annahme stimmt nicht. dispatch_layer_batch hat bereits
  exakt 18 explizite `compute_barrier`-Calls pro Layer (im
  Default-Pfad: mul_mmq + batch_attn + qk_norm + !coopmat).

  Konkret: Q/K/V GEMMs teilen bereits EINE Barrier (Zeile 2756),
  Gate+Up GEMMs teilen EINE Barrier (3124), RoPE Q/K teilen EINE
  Barrier (2819), Q/K-Norm teilen EINE Barrier (2800). Das ist
  was der Audit-Sprint 8b vorgeschlagen hat — und es ist alles
  bereits implementiert.

  Jeder verbleibende `compute_barrier` gated einen echten
  Read-After-Write Hazard zwischen aufeinanderfolgenden Ops:

    Pos | Nach Op             | Schreibt          | Nächste Op liest        | RAW |
    ----+--------------------+-------------------+-------------------------+-----+
    1   | rms_norm_attn       | batch_norm        | quantize_attn           |  ✓  |
    2   | quantize_attn       | batch_q8          | gemm_q/k/v              |  ✓  |
    3   | gemm_q/k/v          | batch_q,k,v       | rms_norm_q+k            |  ✓  |
    4   | rms_norm_q+k        | batch_q,k (in-pl) | rope_q+k                |  ✓  |
    5   | rope_q+k            | batch_q,k (in-pl) | flash_attn / kv_copy    |  ✓  |
    6   | kv_bar (T→C)        | kv_cache          | flash_attn              |  ✓  |
    7   | flash_attn          | batch_attn_out    | quantize_attn_out       |  ✓  |
    8   | quantize_attn_out   | batch_q8          | gemm_o                  |  ✓  |
    9   | gemm_o              | batch_o           | add_res1                |  ✓  |
    10  | add_res1            | batch_residual    | rms_norm_ffn            |  ✓  |
    11  | rms_norm_ffn        | batch_norm        | quantize_ffn            |  ✓  |
    12  | quantize_ffn        | batch_q8          | gemm_gate+up            |  ✓  |
    13  | gemm_gate+up        | batch_gate, up    | silu (in-place gate)    |  ✓  |
    14  | silu                | batch_gate (in-pl)| mul_gate_up             |  ✓  |
    15  | mul_gate_up         | batch_ffn_hidden  | quantize_ffn_h          |  ✓  |
    16  | quantize_ffn_h      | batch_q8          | gemm_down               |  ✓  |
    17  | gemm_down           | batch_ffn_out     | add_res2                |  ✓  |
    18  | add_res2            | batch_residual    | next layer's rms_norm   |  ✓  |

  Alle 18 Stellen sind echte Hazards. Keine sichere Entfernung
  möglich ohne Algorithmus-Änderung (z.B. silu+mul fusion,
  out-of-place silu).

ENTSCHEIDUNG: KEIN CODE GEÄNDERT.
  • Risiko (GPU-Hang, undefined behavior) > vermuteter Gewinn (5-15%)
  • Vermuteter Gewinn ist bereits zum großen Teil im Codebase realisiert
  • Sprint-8b-Hypothese basierte auf veralteter Mental-Map

Tests: 154/154 ✓ (unverändert).
Files: nur dieser Report (keine Code-Änderung).

Bottom line: Sprint 8b's premise war falsch. Das Conditional-
Barrier-Pattern aus llama.cpp PR #12135 ist hier bereits
manuell-instantiiert — nicht via Graph-Tracking, sondern
fest-verdrahtet im dispatch_layer_batch Code.
```

---

## 1. Detail-Analyse — Barrier-Topologie

### 1.1 Was der Audit/Sprint-Briefing annahm

Aus `results/v02_sprint8b_conditional_barriers.md` (Sprint-Brief):

> "Pro Layer (36×):
>   RMSNorm → barrier
>   quantize_q8_1 → barrier
>   gemm_q → barrier
>   gemm_k → barrier
>   gemm_v → barrier
>   RoPE Q → barrier
>   RoPE K → barrier
>   ..."

Das implizit ein "Pattern wo jeder Shader seinen eigenen
Barrier emittiert". Das wäre tatsächlich ineffizient.

### 1.2 Was der Code wirklich tut

`forward.rs::dispatch_layer_batch` (default config: mul_mmq +
batch_attn + qk_norm + !coopmat_q4k) hat exakt diese Struktur:

```rust
// Step 1
self.run_rms_norm(...batch_residual → batch_norm...);
compute_barrier(dev, cmd);  // 2656

// Step 2
self.run_quantize_q8_1(...batch_norm → batch_q8...);
compute_barrier(dev, cmd);  // 2678

// Step 3 — DREI GEMMs OHNE BARRIERS DAZWISCHEN
self.run_gemm(...sq, wq, batch_q8 → batch_q...);
self.run_gemm(...sk, wk, batch_q8 → batch_k...);
self.run_gemm(...sv, wv, batch_q8 → batch_v...);
compute_barrier(dev, cmd);  // 2756 — EINE Barrier nach allen drei

// Step 4 — Q/K-norm OHNE Barrier dazwischen
self.run_rms_norm(...batch_q in-place...);
self.run_rms_norm(...batch_k in-place...);
compute_barrier(dev, cmd);  // 2800

// Step 5 — RoPE Q+K OHNE Barrier dazwischen
self.run_rope_batch(...batch_q in-place...);
self.run_rope_batch(...batch_k in-place...);
compute_barrier(dev, cmd);  // 2819

// ... usw.
```

Die "naive" Variante (Sprint-Brief-Annahme) wäre 22 Barriers
pro Layer. Die TATSÄCHLICHE Implementation hat 18 Barriers pro
Layer. Differenz von 4 ist exakt was Sprint 8b vorschlug zu
entfernen — nur ist es bereits passiert.

Total: 18 × 36 = **648 Barriers/Forward**, nicht ~612.

### 1.3 Verifikation: alle 18 sind RAW-gated

Tabelle aus dem TL;DR oben — jede Stelle ist:
* Schreibt-dann-Liest auf demselben Buffer (in-place ops),
  ODER
* Schreibt-dann-Liest auf demselben Logical-Buffer (z.B.
  batch_norm wird von rms_norm geschrieben und von quantize
  gelesen)

KEIN einziger ist "write-only-1 → write-only-2 disjoint" wie
der Audit annahm. Solche Stellen existieren nicht im code —
sie wurden bereits eliminiert in früheren Sprints (Phase 5B,
3E, 7).

---

## 2. Buffer-Allocation Check

`Forward::new_with_prefill` (ab Zeile 305-380) zeigt:

```rust
let scratch_a = mk_storage(...);    // separate VRAM allocation
let scratch_b = mk_storage(...);
let hidden_norm = mk_storage(...);
let q_buf = mk_storage(...);
let k_buf = mk_storage(...);
let v_buf = mk_storage(...);
// ... batch_q, batch_k, batch_v jeweils eigene mk_storage(...)
```

Jeder Buffer hat eigene `vkAllocateMemory` (via gpu-allocator).
Kein Aliasing. Daher ist meine Tabellen-Analyse oben valide:
verschiedene Buffer-Namen = verschiedener VRAM = kein
versteckter Hazard.

---

## 3. Wo überhaupt etwas möglich wäre

Theoretische Sprint-9-Kandidaten (jeder erfordert
Algorithmus-/Kernel-Änderungen, NICHT nur Barrier-Entfernung):

(a) **Fused silu+mul Kernel** — würde Step 14 + Step 15 zu
einer Op zusammenführen, eine Barrier sparen pro Layer.
Schätzung: -36 Barriers, ~1-3% Gewinn. **Aufwand**: neuer
Shader + Spec-Constants + Tests. Lohnt sich nicht für 1-3%.

(b) **Out-of-place RoPE** — RoPE schreibt aktuell IN-PLACE,
was eine Barrier zwischen rope_q/k und flash_attn nötig macht.
Wenn RoPE in einen separaten Buffer schreiben würde, könnte
flash_attn parallel zu mehr von RoPE laufen. **Aufwand**:
2× VRAM für Q/K, plus Kernel-Änderung. Lohnt sich nicht.

(c) **Persistent KV-Cache-Buffer-Layout** — gemm_k schreibt in
batch_k, dann cmd_copy_buffer kopiert nach kv_cache. Wenn gemm_k
DIREKT in kv_cache schreiben könnte, würden wir die Copy + ein
Barrier sparen. **Aufwand**: Custom-GEMM-Shader mit anderem
Output-Layout. Mittlere Komplexität, geschätzter Gewinn: 5-10%
auf Prefill, 0 auf Decode.

(d) **Conditional barrier elision via runtime tracking**
(llama.cpp PR #12135) — würde NICHTS BRINGEN auf unserem
hot-path weil es keine elidierbaren Barriers gibt. Würde nur
helfen wenn neue Algorithmen-Fälle hinzukommen die das Pattern
brechen. Lohnt nicht.

---

## 4. Files Touched

```
new file: results/v02_sprint8b_conditional_barriers.md (this report)
```

KEIN Code geändert. KEIN Test geändert. KEIN Commit der
Funktionsbits.

---

## 5. Fazit

Die Sprint-8b-Hypothese war eine **alte Mentalkarte**: sie ging
von einem naiven dispatch_layer_batch aus mit Barriers nach JEDER
Op, und projizierte dann eine 24%-Reduktion durch Gruppierung
unabhängiger GEMMs.

Der Code wurde bereits in Phase 5B (batched-Q attention) und
Phase 3E (batched GEMM) so umgebaut dass diese Gruppierung
implizit ist. Die "144 Barriers eliminieren"-Hypothese aus dem
Sprint-Brief ist nicht erreichbar — die 144 sind nicht da zu
eliminieren.

Wir sind seit Phase 5B bei **648 Barriers/Forward**, alle real
benötigt. Weitere Reduktion erfordert Algorithmus-Änderungen
(fused-Kernels, out-of-place ops, layout-changes), nicht
"barrier elision".

---

## 6. Empfehlung — Sprint 9 unverändert

Aus Sprint 7.6/8a Recommendations:

* **Sprint 9 — FP16 KV-Cache** (2-3 Tage): besonders relevant für
  pp ≥ 2048 wo wir bei 0.14-0.22× von llama.cpp sind. Halbiert
  Memory-Bandbreite → 5-15% Prefill-Gewinn auf langem Kontext.

* **Sprint 10 — coopmat_cm2 für Attention** (mehrere Tage): nach
  FP16-KV.

NICHT priorisieren:
* Sprint 8b reboot mit fused-Kernels — niedriger ROI für hohen
  Aufwand (1-3% pro fused kernel × ~3 Kandidaten = 3-9% gesamt).
* Conditional-barrier-tracking Framework — adressiert ein Problem
  das wir nicht haben.
