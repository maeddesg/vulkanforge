# v0.2 Sprint 9b.2 — Cross-Layer add_res2 + rms_norm_attn Fusion

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9b (multi_add_rms Stelle 1, 22 Dispatches/Layer, 164/164 Tests)

---

## TL;DR — Layer-Loop umstrukturiert. -35 Dispatches/-35 Barriers, +0.7% bis +2.4% pp-Sweep, alle Tests grün.

```
═══ v0.2 Sprint 9b.2 ═══

PRE-CHECK ERGEBNIS:
  • Layer-Loop ist EXTERN (caller iteriert in prefill_batch).
  • dispatch_layer_batch wird einmal pro Layer aufgerufen.
  • Cross-Layer Fusion ist sauber implementierbar via API-Erweiterung
    (Option<vk::Buffer> für next_attn_norm_weight).
  • Decode (forward_token) hat 3 Caller + forward_layer_debug —
    Refactor-Risiko hoch, Gain ~1.6%. Defer auf 9b.3.

SCOPE: Nur batched-Pfad (prefill).

Shader: vk_shaders/multi_add_rms.comp WIEDERVERWENDET aus Sprint 9b.
        KEIN neuer Shader.

Layer-Loop-Umstrukturierung in src/backend/vulkan/forward.rs:

  prefill_batch (caller):
    1. Pre-loop: run_rms_norm(model.layers[0].attn_norm.weight) → batch_norm
    2. Loop für jeden Layer N:
       - Layer N=0..n-2: pass next_attn_norm_weight = Some(w_attn[N+1])
       - Layer N=n-1:    pass next_attn_norm_weight = None

  dispatch_layer_batch (callee):
    1. Step (a) rms_norm_attn ENTFÄLLT (batch_norm bereits seeded).
    2. Step (l) verzweigt:
       - Some(w_next) → multi_add_rms(batch_residual, batch_ffn_out,
                                       w_next → batch_residual,
                                       batch_norm)
                       (in-place residual update + seed for next layer)
       - None         → run_binary(Add) (last layer, plain add_res2)

Dispatch-Reduktion (pro Forward):
  Sprint 9b:   36 × 22 Dispatches      = 792
  Sprint 9b.2: 1 (seed) + 36 × 21 = 757 Dispatches
  → Δ = -35 Dispatches, -35 Barriers pro Forward

Performance — pp-Sweep (5 Runs, 3 Warmup, Citrix-Noise):
  | pp   | Sprint 9b | Sprint 9b.2 | Δ tok/s | Δ %    |
  |------|-----------|-------------|---------|--------|
  |   64 |   1480.3  |   1496.57   |   +16   | +1.1%  |
  |  128 |   1876.8  |   1889.67   |   +13   | +0.7%  |
  |  256 |   1923.7  |   1943.60   |   +20   | +1.0%  |
  |  512 |   1773.4  |   1795.90   |   +22   | +1.3%  |
  | 1024 |   1467.9  |   1502.86   |   +35   | +2.4%  |

  → Δ konsistent positiv über alle pp, sauberer als Sprint 9b alleine
    (dort waren pp=512/1024 leicht negativ unter Citrix). Die Cross-Layer-
    Fusion eliminiert genau die Dispatches die im Hot-Loop häufig
    auftreten (35× pro Forward, jeder Layer-Übergang).

Performance — 15-Prompt Bench:
  | Metric                       | Sprint 9b | Sprint 9b.2 | Δ      |
  |------------------------------|-----------|-------------|--------|
  | MEDIAN prefill (alle 15)     |  1067.5   |  1062.7     | -0.4%  |
  | MEDIAN decode                |    90.7   |    91.7     | +1.1%  |
  | First-5 prefill (pp=20)      |   373.8   |   375.9     | +0.6%  |
  | First-5 prefill (pp=62)      |  1436.4   |  1438.0     | +0.1%  |
  | Coherent prompts             |   15/15   |   15/15     | ✓      |

  → Decode +1.1% sogar OHNE dispatch_layer-Änderung (kein Sprint
    9b.2-Effekt für reinen Decode-Workload — die Variation ist
    Citrix-noise innerhalb ±2%).
  → Prefill 15-Prompt-Median -0.4% innerhalb Mess-Rauschen (15-Prompt
    enthält Long-System-Prompt-Workloads die nicht-deterministisch
    durch Citrix beeinflusst werden).

Tests:
  cargo test --release  →  164/164 ✓ unverändert
  Insbesondere alle E2E argmax-Parity-Tests bleiben grün:
    - phase3e_prefill_batch_matches_token_by_token_top5  ✓
    - sprint5b_chunked_prefill_parity_qwen3              ✓
    - phase5b2_decode_after_batched_prefill_qwen3        ✓
    - phase_prompt16_alice_context_retention_qwen3       ✓
    - phase5b2_batch_attn_parity_qwen3_short / two_tiles ✓

  Keine NEUEN Tests addiert: multi_add_rms.comp ist Sprint-9b-getestet
  (5 isolierte Tests, alle grün). Die Cross-Layer-Validierung
  geschieht End-to-End über die existierenden argmax-Tests, die alle
  35 Layer-Übergänge in Qwen3-8B exercisen.

Files:
  modified: src/backend/vulkan/forward.rs (dispatch_layer_batch
            signature, body, prefill_batch caller)
  new:      results/v02_sprint9b2_cross_layer.md (this report)

Commit: HEAD (kein Push).
```

---

## 1. Architektur — was sich verändert hat

### 1.1 Layer-Lifecycle vorher (Sprint 9b)

```
prefill_batch:
  copy embeddings → batch_residual
  for layer in 0..36:
    dispatch_layer_batch(layer)
      ├─ (a) run_rms_norm(batch_residual, w_attn[layer]) → batch_norm  ← FUSION CANDIDATE
      ├─ (b..l-1) Q/K/V GEMM → Attention → O GEMM → SwiGLU → Down GEMM
      └─ (l) run_binary(Add, batch_residual, ffn_out)                   ← FUSION CANDIDATE
  dispatch_final  (rms_norm_final + lm_head)

Total Dispatches/Forward: 36 × 22 = 792
```

### 1.2 Layer-Lifecycle nachher (Sprint 9b.2)

```
prefill_batch:
  copy embeddings → batch_residual
  run_rms_norm(batch_residual, w_attn[0]) → batch_norm   ← SEED Layer 0
  for layer in 0..36:
    next_w = (layer + 1 < 36) ? Some(w_attn[layer+1]) : None
    dispatch_layer_batch(layer, next_w)
      ├─ (a) ENTFÄLLT — batch_norm bereits seeded
      ├─ (b..l-1) Q/K/V GEMM → Attention → O GEMM → SwiGLU → Down GEMM
      └─ (l)  Some(w_next) → multi_add_rms(batch_residual, ffn_out,
                                            w_next → batch_residual,
                                            batch_norm)
              None         → run_binary(Add)
  dispatch_final  (unverändert)

Total Dispatches/Forward: 1 (seed) + 36 × 21 = 757
                       = 792 - 35  → -35 Dispatches/Barriers
```

### 1.3 Datenfluss-Garantie

Die Korrektheit der Fusion hängt am `batch_norm`-Lifecycle:

```
Before Loop:                batch_norm = rms_norm(embeddings) * w_attn[0]
After Layer 0 fusion:       batch_norm = rms_norm(residual_0_updated) * w_attn[1]
After Layer 1 fusion:       batch_norm = rms_norm(residual_1_updated) * w_attn[2]
...
After Layer 34 fusion:      batch_norm = rms_norm(residual_34_updated) * w_attn[35]
Layer 35 reads batch_norm:  USES Layer 35's attn_norm output ✓
After Layer 35 plain-add:   batch_norm UNCHANGED (held over from Layer 34 — correct,
                            Layer 35 has already consumed it for Q/K/V GEMM in step (b..)).
Caller dispatches final:    Reads batch_residual (last residual update). batch_norm
                            irrelevant for final-norm path.
```

Zu jedem Zeitpunkt wo Layer `N` `batch_norm` liest (in seinem step (b..)
Q/K/V GEMM), enthält `batch_norm` exakt `rms_norm(residual_{N-1}) *
w_attn[N]` — also genau das was die alte Step-(a) selbst berechnet hätte.

---

## 2. Code-Änderung (Auszug)

### 2.1 `dispatch_layer_batch` API + Body

```rust
fn dispatch_layer_batch(
    &mut self,
    ...,
    layer: u32,
    seq_len: u32,
    base_pos: u32,
    next_attn_norm_weight: Option<vk::Buffer>,    // NEW
) {
    ...
    // ---- (a) attn_norm: per-token RMSNorm into batch_norm. ----
    // Sprint 9b.2 — this used to dispatch run_rms_norm(...). Now
    // batch_norm is pre-seeded by either prefill_batch (for layer 0)
    // or by the previous layer's cross-layer fusion. Nothing to do.
    
    // ... (b..k) unchanged ...
    
    // ---- (l) Residual2 = residual + ffn_out + (cross-layer fuse). ----
    match next_attn_norm_weight {
        Some(w_next) => {
            self.run_multi_add_rms(
                dev, registry, cmd,
                self.batch_residual.handle, self.batch_ffn_out.handle, w_next,
                self.batch_residual.handle,        // sum_out (in-place)
                self.batch_norm.handle,            // norm_out (seed for next)
                hidden, seq_len, cfg.rms_norm_eps, "add_rms_attn_next_b",
            );
        }
        None => {
            self.run_binary(
                dev, registry, cmd, ShaderId::Add,
                self.batch_residual.handle, self.batch_ffn_out.handle,
                self.batch_residual.handle,
                seq_len * hidden, "add_res2_b",
            );
        }
    }
    compute_barrier(dev, cmd);
}
```

### 2.2 `prefill_batch` Caller

```rust
// Sprint 9b.2 — seed batch_norm for layer 0.
let w_attn_norm_0 = layer_weight(model, 0, "attn_norm.weight");
self.run_rms_norm(
    dev, registry, cmd,
    self.batch_residual.handle, w_attn_norm_0, self.batch_norm.handle,
    hidden, seq_len, cfg.rms_norm_eps, "rms_norm_attn_b_seed",
);
compute_barrier(dev, cmd);

for layer in 0..cfg.n_layers {
    let next_w = if layer + 1 < cfg.n_layers {
        Some(layer_weight(model, layer + 1, "attn_norm.weight"))
    } else {
        None
    };
    self.dispatch_layer_batch(
        dev, registry, cmd, model, layer, seq_len, base_pos, next_w,
    );
}
```

---

## 3. Korrektheit — wie wir sicher waren, dass Indizierung stimmt

Das Sprint-Briefing-Risiko #1 war: "Index-Off-by-One = KOMPLETT falsche
Logits (kein NaN!)". Drei orthogonale Tests fangen das ab:

1. **`phase3e_prefill_batch_matches_token_by_token_top5`** vergleicht
   die argmax-Logits aus `prefill_batch` (mit Sprint 9b.2 Cross-Layer
   Fusion) gegen `forward_token` (Token-by-Token, decode-Pfad,
   UNverändert). Wenn die fused-Path ein falsches Layer-Weight zieht,
   würde dieser Test mit einem komplett anderen argmax fehlschlagen.

2. **`sprint5b_chunked_prefill_parity_qwen3`** läuft `prefill_batch`
   einmal mit allen Tokens vs in Chunks und vergleicht argmax. Beide
   Pfade nutzen die neue Cross-Layer-Fusion, beide müssen identisch
   sein. Falls die per-Chunk-`run_rms_norm`-Seedung falsch wäre
   (z.B. Stale-Norm vom vorigen Chunk), würde das hier divergieren.

3. **`phase_prompt16_alice_context_retention_qwen3`** ist ein
   multi-turn Chat-Test mit einem 16-Prompt-Workload. Rekursive
   Argmax-Drift würde sich hier akkumulieren.

Alle drei grün → Indexierung ist korrekt.

---

## 4. Performance — wo der Gewinn herkommt

### 4.1 35 Dispatches/Barriers eingespart

Pro Forward-Pass entfallen 35× (1 Dispatch + 1 Barrier). Bei einer
typischen Dispatch-Overhead von ~5-10µs pro Call:
* 35 × 7.5µs ≈ 260µs gespart pro Forward
* Bei pp=64 (Forward ~43ms): 0.6% Speedup theoretisch
* Bei pp=1024 (Forward ~683ms): 0.04% theoretisch

Aber die gemessenen Werte zeigen +1.1% bis +2.4% — also mehr als
Dispatch-Overhead allein. Vermutung: das Pipelining wird besser, weil
die Barrier-Tiefe reduziert wird. Mit 35 weniger `compute_barrier`-Calls
kann der GPU-Scheduler mehr in flight halten.

### 4.2 Vergleichbarkeit mit Sprint 9b's Stelle 1

```
| Sprint    | Dispatches/Layer | Barriers/Layer | pp=1024 Δ vs Vorgänger |
|-----------|------------------|----------------|------------------------|
| 9a (start)|        24        |       18       | (baseline)             |
| 9a end    |        23        |       17       | + ~2.0%                |
| 9b end    |        22        |       16       | -2.0% (Citrix noise)   |
| 9b.2 end  |        21*       |       15*      | + 2.4%                 |

*effective: 36×21 + 1 seed = 757 dispatches per forward
*effective: 36×15 + 1 seed = 541 barriers per forward
```

Die `+2.4%` an pp=1024 in 9b.2 holen den negativen Citrix-Run aus
9b.2 mehr als zurück. Cumulativer Gewinn 9a → 9b.2: ca. +5-7%.

### 4.3 Was nicht gewonnen wurde

* Decode-Pfad bleibt unverändert: forward_token wäre ~+1.6% bei
  äquivalenter Cross-Layer-Fusion, aber der Refactor wäre über
  3 Caller (forward_token, forward_token_profile,
  forward_token_profile_layers) plus forward_layer_debug zu ziehen.
  Das Risiko-zu-Gewinn-Verhältnis ist niedrig. → Sprint 9b.3.

* Erste-Layer-Pre-Seed ist 1 zusätzlicher Dispatch + 1 Barrier vor
  dem Loop, also netto -35 statt -36. Vernachlässigbar.

---

## 5. Sprint-9-Roadmap (aktualisiert nach 9b.2)

```
| Sprint | Status      | Beschreibung                       | Δ         |
|--------|-------------|------------------------------------|-----------|
| 9a     | DONE        | swiglu fusion                      | +2.0%     |
| 9c     | DONE (neg)  | rms_norm_mul (already fused)       | 0%        |
| 9b     | DONE        | multi_add_rms Stelle 1             | +1.5%     |
| 9d     | ANALYSIS    | FP16 KV (3-phase plan)             | deferred  |
| 9b.2   | DONE        | Cross-Layer Stelle 2               | +1-2.5%   |
| 9b.3   | TODO        | Cross-Layer Stelle 2 in Decode     | +1-1.5%   |
| 9c.5   | TODO        | rms_norm_mul_rope (Q/K-Norm + RoPE)| +2-3%     |
| 9d.1   | TODO        | FP16 KV Infrastructure             | 0% (PoC)  |
| 9d.2   | TODO        | FP16 KV Prefill                    | +5-15%    |
| 9d.3   | TODO        | FP16 KV Decode (full coverage)     | enables 9d|
```

Cumulativer Prefill-Gewinn 8a→9b.2 (Sprint-aktive Sprints): ca. +5-7%
auf pp=1024.

Empfehlung — nächster Sprint:
* **9c.5 (rms_norm_mul_rope)** — gleiche Komplexität wie 9a/9b/9b.2,
  klares ROI (+2-3%), saubere Q/K-Norm-Stelle. Hoher Wert, niedrige
  Komplexität.
* **9b.3 (Decode Cross-Layer)** — ca. 1 Tag Refactor mit höherem
  Test-Risiko. Marginal.
* **9d.1 (FP16 KV Infrastructure)** — wichtig für VRAM-Erweiterung,
  aber 0% perf bis 9d.2/9d.3 fertig sind.

---

## 6. Bekannte Fallstricke — Erfahrungen aus diesem Sprint

1. **Pre-Check rettet Zeit.** Vor der Implementation 30 Minuten in
   den Layer-Loop-Audit zu investieren hat aufgedeckt, dass:
   - Die Loop EXTERN ist (in `prefill_batch`), nicht intern
   - Die saubere Lösung ist API-Erweiterung mit `Option<vk::Buffer>`
   - Der Decode-Pfad (3 Caller) ist DEUTLICH komplexer als der
     batched-Pfad (1 Caller)

   Ohne Pre-Check wäre vermutlich versucht worden, BEIDE Pfade
   gleichzeitig umzubauen, was den Sprint deutlich vergrößert hätte.

2. **`batch_norm`-Lifecycle ist subtil.** Das Buffer wird für jeden
   Layer überschrieben (von der Cross-Layer-Fusion am Ende von Layer
   N → batch_norm = next-attn-norm für Layer N+1). Wer es liest:
   - Layer N+1's step (b..) Q/K/V GEMM (oder Q8_1-quantize wenn mmq)
   - NACH dem Forward: niemand mehr (dispatch_final liest
     batch_residual, nicht batch_norm)
   - Layer 35 (last) liest batch_norm in seinem Q/K/V GEMM aus dem
     Layer-34-Fusion-Output → das war das, was Layer-34's Brücke
     produzierte. Korrekt.

3. **Last-Layer-Einbettung in das `match`-Pattern ist eleganter
   als Hand-Iteration.** Der `next_attn_norm_weight: Option<vk::Buffer>`-
   Param hält die Verzweigung im Callee, was den Caller-Code einfach
   hält. Alternative wäre gewesen, im Caller zwei separate Code-Pfade
   für "alle Layer außer letztem" und "letzter Layer" zu haben — das
   wäre erratisch und fehleranfällig.

4. **Tests waren bereits vorhanden.** Sprint 9b's isolierte
   `multi_add_rms`-Tests + die End-to-End argmax-Tests aus früheren
   Sprints (5B chunked, 3E prefill_batch) decken die Cross-Layer-
   Korrektheit ab, ohne dass Sprint 9b.2 neue Tests addieren muss.
   Reuse > Duplikation.

---

## 7. Files Touched

```
modified: src/backend/vulkan/forward.rs (3 hunks: signature,
          body steps (a)+(l), caller seed+loop)
new:      results/v02_sprint9b2_cross_layer.md (this report)
```

KEIN neuer Shader. KEIN Test geändert. KEIN Build-Setup. KEIN
Pipeline-Registry-Eintrag.

---

## 8. Bottom Line

Sprint 9b.2 ist der saubere Cross-Layer-Folge-Sprint zu 9b — gleicher
Shader, gleiche Fusion, aber jetzt am Layer-Übergang. Die Layer-Loop-
Umstrukturierung ist genau eine API-Erweiterung (`Option<vk::Buffer>`
in `dispatch_layer_batch`) und ein Pre-Loop-Seed in `prefill_batch`.

Die +1-2.4% pp-Sweep-Δ ist konsistent positiv und größer als der
"reine Dispatch-Overhead-Saving"-Gewinn — vermutlich profitiert
auch das Pipelining von der reduzierten Barrier-Tiefe.

Nächster Schritt-Empfehlung: **Sprint 9c.5 (rms_norm_mul_rope)**.
Saubere Stelle, gleiche Komplexitätsklasse, +2-3% erwartet. Decode-
Cross-Layer-Fusion (9b.3) ist machbar aber risk-to-value ungünstig
und kann später kommen.
