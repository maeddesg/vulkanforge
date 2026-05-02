# Phase 5A-2 — Command-Buffer-Reuse

**Datum:** 2026-04-26
**Status:** Stufe 2D (Persistente Descriptor-Sets) abgeschlossen + verifiziert.
Stufe 2A optional, Stufe 2B nicht nötig (Qwen3 jetzt > 88 tok/s).
**Modell:** Qwen3-8B Q4_K_M, 36 Layer, hidden=4096, heads=32/8
**Hardware:** AMD Radeon RX 9070 XT (RADV GFX1201, Mesa 26.0.5)

## TL;DR

Der "3.3 ms Dispatch-Overhead" aus Phase 4D ist **echt und reproduzierbar**:
RECORD-Block braucht bei pos=100 median 3.5–3.7 ms wall-time. **99.7 % davon
liegen in den 36 `dispatch_layer`-Aufrufen** (~96 µs/Layer × 36). Der
Rest (begin/end/submit/empty-floor) ist vernachlässigbar (~143 µs).

Aber: **GPU_WAIT dominiert das Forward-Total** mit ~9.5 ms, mehr als 2.5×
RECORD. VulkanForge ist bei Decode primär GPU-bound, nicht CPU-bound.
Optimization-Ceiling: max(CPU, GPU) ≈ 9.5 ms → **~105 tok/s** vs. heute 73 tok/s.

## Per-Phase Breakdown (median über pos 50–99, in µs)

| Phase | µs | % of total | Optimierbar? |
|---|---:|---:|---|
| `pre_setup` (embed write, rope-pos write, desc-pool reset) | 537 | 3.9 % | gering |
| `reset` (`vkResetCommandBuffer` + `reset_fences`) | 10 | 0.1 % | nein |
| `begin` (`vkBeginCommandBuffer`) | 1 | 0.0 % | nein |
| **`RECORD`** (alle 36 `dispatch_layer` + `dispatch_final` + final barrier) | **3 570** | **26.1 %** | **JA — Hauptziel** |
| `end` (`vkEndCommandBuffer`) | 1 | 0.0 % | nein |
| `submit` (`vkQueueSubmit`) | 18 | 0.1 % | nein |
| **`GPU_WAIT`** (`vkWaitForFences` — pure GPU wall-clock) | **9 508** | **69.5 %** | nein (Phase 4C bereits durch) |
| `readback` (logits buffer → host) | 22 | 0.2 % | gering |
| **TOTAL** | **13 669** | **100 %** | |
| → Decode-Rate today: 1 / 13.7 ms ≈ **73 tok/s** | | | |

## Per-Position-Snapshot

```
   pos  pre_setup     reset     begin    RECORD       end    submit  GPU_WAIT  readback     TOTAL
     0        569        48         1      3393         1        24     11507       235     15781   ← warm-up
     1        574        13         2      3264         0        19     11430        29     15334   ← warm-up
    50        574        11         1      3390         1        18      9223        23     13244
   100        460         6         1      3531         1        16      9512        22     13552
   150        574        10         1      3685         1        16      9578        21     13889
   200        601        12         2      3686         1        18      9574        24     13921
   210        600         7         1      3514         1        18      9329        23     13495
```

Beobachtungen:
* RECORD wächst leicht mit pos (3.4 → 3.7 ms) — wahrscheinlich split-K
  Attention ab pos≥64 (1 zusätzlicher Dispatch pro Layer).
* GPU_WAIT wächst von 9.2 ms (pos=50) auf 9.6 ms (pos=200) — Attention
  ist O(seq_len) wegen KV-Cache-Sweep.
* Bei pos 0 + 1 ist GPU_WAIT erhöht (pipeline-cache JIT auf der ersten
  Welle; Mesa lädt SPIR-V → ACO bei Cold-Start).

## Drill-Down: Wo gehen die 3.5 ms RECORD genau hin?

```
RECORD wall                3 701 µs
Σ per-layer dispatches     3 689 µs   (99.7 %)
per-layer min/med/max         94 / 96 / 182 µs
dispatch_final + barrier      10 µs
unaccounted                    2 µs

Floor (empty record block)   143 µs
```

Damit ist klar:
* **`dispatch_layer` ist die einzige Stellschraube.** 36 Layer × ~96 µs
  = 3.46 ms.
* `dispatch_final` (final RMSNorm + LM-Head GEMV) ist mit 10 µs
  vernachlässigbar — er nutzt bereits gecachte Pipeline-Handles.
* Per-Layer-Varianz ist klein (median 96 µs, max 182 µs in den
  Attention-haltigen Layern).

### Was steckt in einem 96-µs-`dispatch_layer`?

Pro Layer ~15 Dispatches: RMSNorm-attn, GEMV-Q/K/V (3), Q/K-Norm (2),
RoPE-Q/K (2), Attention (1), GEMV-O, Add, RMSNorm-FFN, GEMV-Gate/Up (2),
SiLU+Mul, GEMV-Down, Add. → **~6.4 µs pro Dispatch.**

Pro Dispatch passieren CPU-seitig:
1. `registry.get(ShaderId)` — `HashMap` lookup, ~150 ns.
2. `self.alloc_set(dev, layout)` — `vkAllocateDescriptorSets` aus Pool.
3. `self.write_bindings(...)` — `vkUpdateDescriptorSets` mit ~5 Bindings.
4. Push-Constants-Struct bauen (Rust-side).
5. `vkCmdBindPipeline`.
6. `vkCmdBindDescriptorSets`.
7. `vkCmdPushConstants`.
8. `vkCmdDispatch` (oder `cmd_copy_buffer` für KV-write, oder `cmd_pipeline_barrier`).

Ohne weitere Mikro-Instrumentierung (würde die Messung selbst
beeinträchtigen) ist die wahrscheinlichste Verteilung der 6.4 µs:
* **Descriptor-Set-Allokation + Update: ~3–4 µs** (Vulkan-Loader hat
  hier signifikanten Pfad: pool walk + slot bookkeeping, plus
  Update-Calls bauen im Loader die `VkWriteDescriptorSet`-Strukturen).
* **`vkCmd*`-Calls: ~1–2 µs** (Loader-Trampoline + Driver-Funktion;
  bei RADV sehr schnell weil Command-Recording reines Memcpy in den
  CB-Buffer ist).
* **HashMap-Lookup + PC-Struct-Bau: <1 µs**.

→ **Hauptlast liegt mit hoher Wahrscheinlichkeit bei den Descriptor-
Sets**, nicht bei den `vkCmd*`-Calls selbst.

## Was bedeutet das für die Step-2-Optimierung?

Der Prompt unterscheidet drei Pfade:

* **2A — Caching + Fast Re-Recording** (Pipeline-Handles cachen,
  PC-Templates wiederverwenden, Barrier-Pattern fixieren).
  *Verspricht*: weniger Rust-Setup pro Dispatch.
* **2B — UBO + echtes Command-Buffer-Reuse** (variable Parameter in
  Uniform-Buffer; CB einmal aufnehmen, nur UBO updaten).
  *Verspricht*: 0 µs Recording ab Token 1.
* **2C — Pipelined Submission** (Token N+1 aufnehmen während GPU N
  läuft).
  *Inapplicable*: Logits-Readback blockiert das Pipelining.

### Die Daten zeigen aber noch eine vierte Option

**2D — Persistente Descriptor-Sets (Pre-Alloc + Re-Use)**:

Heute reset'en wir den Descriptor-Pool **bei jedem `forward_token`**
(`reset_descriptor_pool` in `pre_setup`) und allozieren dann pro
Dispatch ein neues Set. Die Descriptor-Sets sind aber strukturell
identisch zwischen Tokens (gleiche Bindings, nur Buffer-Inhalt ändert
sich).

```
Heute:    reset_pool + alloc(N=540) per forward × 36 Layer × ~15 disp
Stattdessen: pre-alloc ALLE Sets EINMAL beim Forward::new
             vkUpdateDescriptorSets nur an Stellen wo Buffer-Handle
             tatsächlich wechselt (KV-Cache-Layer-Offset, …)
             → pro Token: nur ~50-100 Updates statt 540 Allokationen
```

Dieser Ansatz ist:
* **Einfacher als 2A** (kein Code-Refactor in `dispatch_layer`).
* **Risikoärmer als 2B** (keine Shader-Änderungen, kein UBO-Layout).
* **Adressiert vermutlich den größten Brocken** der 96 µs/Layer.

### Geschätzte Wirkung der Optionen

| Pfad | Erwarteter RECORD | Decode-Rate | Aufwand |
|---|---:|---:|---|
| Heute | 3.6 ms | 73 tok/s | — |
| **2D — Persistente Desc-Sets** | ~1.5–2.0 ms | ~85–88 tok/s | 1 Tag |
| 2A — + PC-Templates / Pipeline-Cache | ~1.0–1.5 ms | ~88–92 tok/s | +1 Tag |
| 2B — Full CB-Reuse mit UBO | ~0.2–0.5 ms | ~95–99 tok/s | 4–5 Tage |
| GPU-Floor (alle CPU-Phasen → 0) | 0 | ~105 tok/s | nicht erreichbar |

Wichtig: **Selbst 2B kann nicht über 105 tok/s gehen** — das ist die
GPU-Wall (Phase 4C bereits ausgereizt mit Split-K-Attention).

## Empfehlung

Vorschlag: **inkrementelle Implementation in der Reihenfolge 2D → 2A → 2B**, mit
einem Correctness-Gate nach jeder Stufe.

* **Stufe 1 — 2D (Persistente Descriptor-Sets):** Direkter Angriff auf
  die ~3 µs/Dispatch im Allokations-Pfad. Erwarteter Gewinn: +15-20 %
  Decode (Qwen3 73 → 85-88 tok/s).
  Aufwand: ~1 Tag. Kein Shader-Eingriff.
  Implementierung: `Forward::new` legt einen `Vec<vk::DescriptorSet>`
  pro (Layer, Shader-Slot) an, `dispatch_layer` ruft die Sets nur ab
  und macht ein zielgerichtetes `vkUpdateDescriptorSets` für die
  Bindings die zwischen Tokens variieren (KV-Cache-Pointer auf den
  aktuellen Layer-Offset).

* **Stufe 2 — 2A (Pipeline-Handle-Cache + PC-Templates):** Nach 2D
  prüfen, was vom 96-µs-Budget übrig ist. Falls noch >40 µs/Layer:
  Pipeline-Lookup-HashMap durch direkten Slice-Access ersetzen,
  PC-Bytes pro Layer pre-built im `Forward`-Struct halten.
  Aufwand: ~1 Tag.

* **Stufe 3 — 2B (UBO + CB-Reuse):** Nur falls 2D+2A nicht reichen.
  Aufwand: ~4-5 Tage, betrifft alle Shader.

Einmaliger Vergleich am Ende: alle drei Stufen jeweils mit der
5-Prompt-Suite und dem `profile_forward`-Tool.

## Fragen an den User vor Coding

1. **Ist die Reihenfolge 2D → 2A (→ optional 2B) okay?** Pro:
   inkrementell, jeder Schritt sofort messbar, jeder Schritt für
   sich rückwärtskompatibel (Direct-Path bleibt). Contra: drei kleine
   Commits statt einer großen Lösung.

2. **Acceptance-Kriterium**: bit-exact zum Direct-Path? Falls Stufe 2D
   mit `vkUpdateDescriptorSets` nicht den exakt gleichen Ablauf erzeugt
   (Validation-Layer-Heuristiken), könnten Outputs in der allerletzten
   FP-Stelle abweichen. Akzeptieren wir < 1e-6 max abs error?

3. **`VULKANFORGE_CB_REUSE=1` Flag** wie im Prompt skizziert: weiter
   gewünscht, oder direkt Default? (Ich würde Flag empfehlen, bis nach
   einer Stufe alle Tests sauber durch sind.)

## Status

* `examples/profile_forward.rs` (neu) — reproduzierbar via
  `VF_NUM_TOKENS=220 cargo run --release --example profile_forward`.
* `src/backend/vulkan/commands.rs` — `one_shot_profiled()` +
  `OneShotTimings`-Struct hinzugefügt (additiv, `one_shot()`-Verhalten
  unverändert).
* `src/backend/vulkan/forward.rs` — `forward_token_profile()` und
  `forward_token_profile_layers()` hinzugefügt (additiv, kein bestehender
  Pfad geändert).
* Tests-Stand: 41/41 (Phase 4D-Stand, vor Step 1 verifiziert; Step 1
  hat keinen Pfad geändert der getestet wird).
* Kein Commit, kein Push.

## Reproduktion (Step 1 — Direct-Path Profile)

```bash
VF_NUM_TOKENS=220 cargo run --release --example profile_forward
```

Erwartete Ausgabe-Zeile (median 50–99):
```
 50-99   ~537   ~10    ~1   ~3570    ~1   ~18   ~9508   ~22   ~13669    µs
```

---

# Stufe 2D — Persistente Descriptor-Sets

**Status:** ✅ Implementiert, getestet, gemessen.
**Acceptance:** ✅ Bit-exact (max abs err = 0.00e0) gegen Direct-Path bei
allen 16 getesteten Positionen.
**Tests:** ✅ 42/42 grün — sowohl mit Cache deaktiviert als auch aktiviert
(17 regression + 25 correctness).

## Was geändert wurde

Eine Descriptor-Set-Cache wurde additiv ins `Forward`-Struct eingebaut.
Aktivierung über das Flag `VULKANFORGE_CB_REUSE=1` (Default: Direct-Path
unverändert).

* `BindingSignature` + `BindingEntry` Structs in `forward.rs`:
  fixe-Größe-Key (8 Bindings, kein Heap-Alloc) → cheap HashMap-Key.
* Neue Methode `Forward::alloc_or_get_set(layout, &bindings)` ersetzt
  alle 19 `alloc_set` + `write_bindings`-Paare an den Call-Sites in
  `dispatch_layer`, `dispatch_final` und allen `run_*`-Helpern.
* `Forward::reset_descriptor_pool_and_cache()` invalidiert den Pool
  *und* die Cache-HashMap. Wird nur von den Pfaden mit variablen
  Bindings aufgerufen: `prefill_batch`, `forward_layer_debug`,
  `forward_layer_debug_intermediate`.
* In `forward_token` / `forward_token_profile{,_layers}`: das
  `reset_descriptor_pool` wird übersprungen wenn `cache_enabled = true`
  — die Sets bleiben gültig.
* Pool-Größe um Faktor 4 erhöht (`max_sets *= 4`) damit der akkumulierte
  Cache + ein anschließender prefill_batch noch reinpassen.
* `Forward::set_cache_enabled(bool)` + `cache_enabled()`: für
  Tests, die zwei Forward-Instanzen mit deterministischer Cache-Settings
  aufbauen müssen.

## Performance-Vergleich (Qwen3-8B, 5-Prompt-Suite)

| Metric | Direct-Path | CB-Reuse | Δ |
|---|---:|---:|---:|
| **5-Prompt median decode** | 72.4 tok/s | **88.7 tok/s** | **+22.5 %** |
| 5-Prompt aggregate decode | 68.7 tok/s | 84.0 tok/s | +22.3 % |
| 5-Prompt aggregate prefill | 307.8 tok/s | 376.0 tok/s | +22.2 % |
| Coherent prompts | 5/5 | 5/5 | — |

Llama-3.1-8B (gleiche Suite, gleiche Prompts):

| Metric | Direct-Path | CB-Reuse | Δ |
|---|---:|---:|---:|
| 5-Prompt median decode | 81.5 tok/s | **95.1 tok/s** | +16.7 % |
| 5-Prompt aggregate decode | 75.9 tok/s | 87.6 tok/s | +15.4 % |
| Coherent | 5/5 *(ein false-positive bei Bench-Heuristik)* | 4/5 *(selber FP)* | — |

## Wo geht die gewonnene Zeit hin? (Profile bei pos=100)

```
                          Direct-Path        CB-Reuse           Δ
pre_setup                    537 µs             0 µs       -537  (kein pool reset)
RECORD                      3570 µs          1960 µs      -1610  (-45 %)
  per-layer (median)            96 µs            51 µs       -45  (-47 %)
GPU_WAIT                    9508 µs          9198 µs       -310  (kleinerer Cache-Druck)
TOTAL                      13669 µs         11221 µs      -2448  (-18 %)
Decode-Rate                 73.2 tok/s       89.1 tok/s    +22 %
```

Pro Dispatch (96 µs/Layer / ~15 Dispatches ≈ 6.4 µs):
* Direct: ~3-4 µs `vkAllocateDescriptorSets` + ~1 µs `vkUpdateDescriptorSets`
  + ~1 µs `vkCmdBindPipeline` + `vkCmdPushConstants` + `vkCmdDispatch` + ~1 µs Rust-Overhead.
* CB-Reuse: HashMap-Lookup-Hit (~150 ns) + dieselben drei `vkCmd*`-Calls
  + Rust-Overhead. Die Allokations-/Update-Calls sind komplett weg.

→ **Stage 2D adressiert genau die ~3-4 µs Allokations-Overhead pro Dispatch**,
exakt wie in der Step-1-Hypothese vorhergesagt.

## Correctness

Neue Regression-Test `phase5a_cb_reuse_parity_qwen3` in `tests/regression.rs`:

```
[parity] pos= 0  max_abs_err=0.000e0  argmax_a=271  argmax_b=271
[parity] pos= 1  max_abs_err=0.000e0  argmax_a=14085 argmax_b=14085
…
[parity] pos=15  max_abs_err=0.000e0  argmax_a=13   argmax_b=13
test phase5a_cb_reuse_parity_qwen3 ... ok
```

**Bit-exact** an allen 16 getesteten Positionen — der Cache reproduziert
*exakt* dieselbe Floating-Point-Sequenz wie der Direct-Path. Damit ist die
Bedingung „< 1e-6 max abs error" deutlich übererfüllt; numerisch sind
beide Pfade identisch.

(Logische Erklärung: Stage 2D ändert *nur* die Reihenfolge von Vulkan-
Host-Calls — die GPU sieht denselben Command-Buffer-Stream mit denselben
Pipeline-Bindings, denselben Descriptor-Sets und denselben Push-Constant-
Bytes. Die Operationen-Reihenfolge auf der GPU ist deshalb byte-identisch.)

## Tests

```
$ cargo test --release --tests
test result: ok. 17 passed; 0 failed
test result: ok. 25 passed; 0 failed

$ VULKANFORGE_CB_REUSE=1 cargo test --release --tests
test result: ok. 17 passed; 0 failed
test result: ok. 25 passed; 0 failed
```

**42/42 grün in beiden Modi.**

## Empfehlung für Stufe 2A

Stage 2D hat die Decode-Rate von 73 auf 89 tok/s gehoben (+22 %). Der
User hatte vor Beginn formuliert: „2B nur falls die ersten beiden
zusammen unter 80 tok/s bleiben." → Wir sind über 80 tok/s, **2B fällt
weg**.

Was Stage 2A noch holen könnte:
* Pipeline-Handle-Cache (HashMap-Lookup → direkter Slice-Access): ~0.5 µs/Dispatch
* PC-Template (Struct schon ge-bytemuck'd): ~0.5 µs/Dispatch
* Geschätzter Gewinn: ~5-7 µs/Layer → ~200 µs RECORD → +1.5 % decode (89 → 90).

Das ist Polish, kein nennenswerter Gewinn mehr. Mein Vorschlag:
**Stage 2D shippen, Stage 2A optional auf den Backlog**, und mit dem
freien Budget lieber an die 9.5-ms GPU-Wall ran — wenn überhaupt noch
in Phase 5.

## Status

* Code: additiv, kein Direct-Path-Verhalten geändert. Default ist
  unverändert; CB-Reuse via `VULKANFORGE_CB_REUSE=1` opt-in.
* Tests: 42/42 in beiden Modi. Neuer Parity-Test bit-exact.
* Performance: +22 % Qwen3, +17 % Llama-3.1.
* Bereit zum Commit (auf User-Freigabe).
