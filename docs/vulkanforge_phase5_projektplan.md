# VulkanForge — Phase 5 Projektplan

**Erstellt:** 26.04.2026
**Basis:** Phase 4D Ergebnisse (81.5 tok/s Llama-3.1, 72.4 Qwen3, 55/55 Tests, 17 Shader)
**Hardware:** RX 9070 XT (gfx1201, RDNA4), Ryzen 9 7945HX, 64GB RAM, Arch Linux
**Strategie:** Performance-Lücke zu llama.cpp schließen + strukturelle Vorteile ausspielen

---

## Ausgangslage nach Phase 4

```
                    Qwen3-8B    Llama-3.1   DeepSeek-R1   llama.cpp Vk
Decode (tok/s):     72.4        81.5        81.3          114.2
Prefill (tok/s):    288.1       358.2       306.7         4314
Kohärenz:           5/5         5/5         5/5           —
Tests:              55/55
Shader:             17
Modelle:            3 (+ Mistral blocked auf SPM)

Verbleibende Gaps:
  Decode:   81.5 / 114.2 = 71% von llama.cpp Vulkan
  Prefill:  358 / 4314 = 8% von llama.cpp Vulkan

Bottleneck-Analyse (pos=200, Llama-3.1 geschätzt):
  GEMV:              ~10.9 ms  (65-78%)
  Attention Split+Red: ~1.9 ms  (13%)
  Dispatch-Overhead:  ~2.7 ms  (16%)
  Rest (Norm etc.):   ~1.2 ms  (8%)
```

---

## Lehren aus Phase 4 die den Plan formen

```
Lektion 1: RGP-Interpretation war falsch (VGPR ≠ Bottleneck)
  → KONSEQUENZ: Jede Hypothese MESSEN, nicht annehmen

Lektion 2: Negativ-Ergebnisse sind wertvoll (Phase 4A, 4B)
  → KONSEQUENZ: Honest STOP statt Workaround

Lektion 3: Infrastruktur vor Performance (4B → 4C)
  → KONSEQUENZ: Flash-Attn war nötig für Split-K

Lektion 4: Llama-3.1 Bug reproduzierte NICHT
  → KONSEQUENZ: VulkanForge Neuimpl ist sauberer als ROCmForge-Port

Lektion 5: llama.cpp nutzt DGC NICHT
  → KONSEQUENZ: Struktureller Vorteil möglich
```

---

## Übersicht

```
Phase 5A: DGC — Device Generated Commands (3-5 Tage)
  → Dispatch-Overhead 3.3ms → ~0.2ms
  → Erwartung: 81.5 → ~90+ tok/s
  → llama.cpp hat das NICHT

Phase 5B: Prefill Batch-Attention (3-5 Tage)
  → Prefill 358 → 2000+ tok/s
  → Token-für-Token-Loop durch batched Flash-Attention ersetzen

Phase 5C: SPM-Tokenizer (1-2 Tage)
  → Mistral-7B, Llama-2, Falcon-3 freischalten
  → Unigram-Tokenizer ~200-300 LOC

Phase 5D: FP16-KV + vec4 GEMV (3-5 Tage)
  → KV-Cache Memory-Traffic halbieren
  → GEMV Memory-Transaktionen 4× breiter
  → Erwartung: +10-15% Decode

Phase 5E: Polish + Release (2-3 Tage)
  → 15-Prompt-Benchmark alle Modelle
  → README finalisieren
  → v1.0.0 Tag vorbereiten

Gesamt: ~15-20 Tage (~3-4 Wochen)
```

---

# Phase 5A: VK_EXT_device_generated_commands (3-5 Tage)

## Ziel

```
Dispatch-Overhead:  3.3ms → ~0.2ms
Decode Llama-3.1:   81.5 → ~90+ tok/s
Differenzierung:    llama.cpp kann das NICHT (Portabilität)
```

## Hintergrund

```
Aktuell: CPU zeichnet 250+ Dispatches pro Token auf
  → Pipeline binden, Push-Constants, Barrier, vkCmdDispatch
  → 3.3ms CPU-Overhead (16-21% der Forward-Zeit)

DGC: GPU führt vorbereitete Command-Sequenz selbst aus
  → CPU: 1× Submit pro Token
  → GPU: iteriert über Dispatch-Tabelle intern
  → Push-Constants aus GPU-Buffer statt Host-Writes

Confirmed: vulkaninfo zeigt VK_EXT_device_generated_commands rev 1
```

## Schritte

```
5A.1: Extension-Analyse (Tag 1)
  - VK_EXT_device_generated_commands API studieren
  - Welche Structs: VkIndirectCommandsLayout, VkGeneratedCommandsInfo
  - Wie werden Push-Constants aus GPU-Buffern gespeist?
  - Limitations: max indirect commands, supported operations
  - RADV-Implementierung in Mesa-Source checken (wie reif?)
  - Smoke-Test: triviale Indirect-Command-Sequenz (1 Dispatch)
  Gate: Extension funktioniert auf RADV/gfx1201

5A.2: Forward-Pass als Indirect-Command-Sequenz (Tag 2-3)
  - 1 Layer als Indirect-Command-Template aufbauen
  - Push-Constants in GPU-Buffer stagen
  - Barrier-Integration (DGC unterstützt Barriers?)
  - Falls DGC keine Barriers unterstützt: Hybrid-Ansatz
    (DGC für Dispatch-Gruppen zwischen Barriers)
  Gate: 1 Layer via DGC, Correctness vs direct Dispatch

5A.3: Full Forward via DGC (Tag 3-4)
  - 36 Layer × ~7 Dispatches = ~250 Indirect Commands
  - Position-abhängige Push-Constants dynamisch updaten
  - Attention-Branching (flash_attn vs split_k) in DGC?
    Falls nicht: Hybrid (DGC für GEMV-Block, direct für Attention)
  Gate: Full Forward via DGC, tok/s ≥ Baseline

5A.4: Performance-Messung + Optimierung (Tag 4-5)
  - ShaderProfiler pos=0, pos=100, pos=200
  - Dispatch-Overhead Vorher/Nachher
  - 5-Prompt-Suite alle Modelle
  - Falls Overhead > 1ms: analysieren warum, DGC-Struct optimieren
  Gate: Dispatch-Overhead < 1ms, Decode > Baseline

STOP-Bedingungen:
  - DGC auf RADV instabil/buggy → dokumentieren, Fallback behalten
  - DGC unterstützt keine Compute-Pipelines → STOP (nur Graphics?)
  - Performance-Regression → Revert, Direct-Path bleibt Default
```

## Risiken

```
RISIKO 1: RADV DGC Implementierung ist unreif
  → Extension exposed ≠ funktioniert
  → Smoke-Test am Tag 1 klärt das
  → Fallback: Command-Buffer-Template-Reuse (Weg 2)

RISIKO 2: DGC unterstützt keine Compute-Dispatches
  → VK_EXT_device_generated_commands spezifiziert Compute
  → Aber RADV-Implementierung könnte unvollständig sein
  → Tag 1 Smoke-Test

RISIKO 3: Push-Constants nicht dynamisch aus GPU-Buffer
  → Alternative: Uniform-Buffer statt Push-Constants
  → Größerer Umbau, aber machbar

RISIKO 4: Barriers zwischen Dispatches nicht in DGC
  → Hybrid: DGC-Blöcke zwischen Barriers, CPU setzt Barriers
  → Reduziert Gewinn aber eliminiert ihn nicht
```

## Fallback (falls DGC scheitert)

```
Weg 2: Command-Buffer-Template + Reuse
  - Einmal aufnehmen (alle 250 Dispatches)
  - Pro Token: nur Push-Constants updaten, re-submit
  - Dynamic Uniform Buffers für variable Offsets
  - Aufwand: ~2 Tage, portabel, kein Extension-Dependency
  - Erwartung: 3.3ms → ~1ms (nicht so gut wie DGC, aber solid)
```

---

# Phase 5B: Prefill Batch-Attention (3-5 Tage)

## Ziel

```
Prefill:  358 → 2000+ tok/s
Problem:  prefill_batch nutzt Token-für-Token Attention-Loop
Lösung:   Batched-Q Flash-Attention (alle Query-Tokens parallel)
```

## Hintergrund

```
Aktuell in prefill_batch:
  for pos in 0..seq_len:
    run_scalar_attn(pos)   // 1 Query gegen pos K/V-Einträge
  
  → O(seq_len²) Attention-Dispatches
  → Bei pp=500: 500 × Attention-Dispatch = langsam

Batched-Q Flash-Attention:
  1 Dispatch für ALLE Query-Tokens gleichzeitig
  → O(1) Dispatches für Attention (plus O(seq_len²) Compute intern)
  → Massiv weniger Dispatch-Overhead
  → Kausale Maske nötig (Token i sieht nur Token 0..i)
```

## Schritte

```
5B.1: Batched-Q Flash-Attention Shader (Tag 1-2)
  - flash_attn_batch.comp: M Query-Tokens × N KV-Positionen
  - Kausale Maske: score = -inf falls k_pos > q_pos
  - Dispatch: (n_heads, n_tiles_kv, n_tiles_q) oder ähnlich
  - Online-SoftMax mit Maske
  Gate: Correctness vs Token-für-Token bei pp=4

5B.2: Integration in prefill_batch (Tag 2-3)
  - Alle Q-Tokens als Matrix statt Vektor
  - Q-Projection: GEMM (haben wir schon)
  - K/V-Projection: GEMM (haben wir schon)
  - Attention: flash_attn_batch statt Loop
  - O-Projection: GEMM
  Gate: prefill_batch mit Batch-Attention, Logits-Parity

5B.3: Performance + Optimierung (Tag 3-5)
  - Prefill tok/s bei pp=29, pp=100, pp=500
  - Vergleich mit Token-für-Token Baseline
  - Tile-Size tuning für Batch-Attention
  Gate: Prefill ≥ 1500 tok/s bei pp=500

STOP-Bedingungen:
  - Kausale Maske numerisch instabil → Token-für-Token Fallback
  - Batch-Attention VRAM-Explosion bei großem seq_len → Cap
  - Correctness-Regression → Revert
```

---

# Phase 5C: SPM-Tokenizer (1-2 Tage)

## Ziel

```
Mistral-7B-Instruct freischalten
Llama-2 / Falcon-3 ebenfalls (gleicher Tokenizer-Typ)
```

## Schritte

```
5C.1: Unigram-Tokenizer implementieren (Tag 1)
  - Token-Tabelle: (token_string, log_probability) aus GGUF
  - Segmentierung: Viterbi Best-Path
    (gegeben Text, finde Tokenisierung mit höchster Gesamt-Log-Prob)
  - Decode: ▁ (U+2581) → Leerzeichen
  - ~200-300 LOC

5C.2: Mistral Integration + Test (Tag 1-2)
  - Tokenizer::from_gguf erkennt model="llama" → SPM-Pfad
  - Mistral Chat-Template: [INST]...[/INST]
  - 5-Prompt-Test auf Mistral-7B
  - Roundtrip-Tests (encode → decode = original)
  Gate: Mistral-7B 5/5 kohärent

Tests:
  [spm] encode("Hello world") = erwartete Token-IDs
  [spm] roundtrip für 10 verschiedene Strings
  [spm] Mistral 5-Prompt-Suite: kohärent + tok/s
```

---

# Phase 5D: FP16-KV + vec4 GEMV (3-5 Tage)

## Ziel

```
GEMV ist 78% der Forward-Zeit. Zwei Hebel:

1. FP16-KV-Cache: Attention liest K/V als f16 statt f32
   → 2× weniger Memory-Traffic bei Attention
   → KV-Cache halb so groß → mehr Kontext in 16GB

2. vec4 GEMV Loads: GEMV liest Weights als vec4 statt float
   → 4× breitere Memory-Transaktionen
   → Bessere Memory-Bus-Auslastung
```

## Schritte

```
5D.1: FP16-KV-Cache (Tag 1-2)
  - KV-Write: f32 → f16 Conversion beim Schreiben
  - Attention: f16 → f32 Conversion beim Lesen
  - Oder: Attention direkt in f16 rechnen (weniger präzise)
  - flash_attn_split.comp anpassen: K/V als float16_t
  - Parity-Gate: max_abs vs f32-KV < 1e-2
  Gate: Decode tok/s ≥ Baseline, Attention-Zeit sinkt

5D.2: vec4 GEMV Loads (Tag 2-4)
  - mul_mat_vec_q4_k.comp: B-Operand als vec4 laden
    statt 4× float Load
  - Prüfe: llama.cpp Shader nutzt möglicherweise schon vec4?
    → Falls ja: warum nutzen WIR es nicht? (Spec-Constants?)
  - Falls eigene Modifikation nötig: minimaler GLSL-Eingriff
  Gate: GEMV BW% steigt, Decode tok/s steigt

5D.3: Kombiniert messen (Tag 4-5)
  - Beide Optimierungen zusammen
  - 5-Prompt-Suite alle Modelle
  - 15-Prompt-Benchmark
  Gate: Decode ≥ 90 tok/s (Llama-3.1)
```

## Risiken

```
RISIKO: FP16-KV Qualitätsverlust
  → FP16 hat nur 10 Mantissen-Bits (vs 23 bei f32)
  → Bei großem Kontext (>1000 Tokens): Attention-Drift?
  → Parity-Gate fängt das ab
  → Fallback: f32-KV bleibt als Option
```

---

# Phase 5E: Polish + Release (2-3 Tage)

## Schritte

```
5E.1: 15-Prompt-Benchmark (Tag 1)
  - Alle unterstützten Modelle (Qwen3, Llama-3.1, DeepSeek-R1,
    + Mistral falls 5C fertig)
  - 4-System-Vergleich aktualisiert
  - Performance-Tabelle für README

5E.2: README + Docs finalisieren (Tag 1-2)
  - Installation (Arch Linux, Rust, Vulkan-Treiber)
  - Quick Start
  - Performance-Tabelle
  - Supported Models
  - Architecture Overview
  - CHANGELOG komplett
  - NOTICE (llama.cpp Attribution)
  - LICENSE (MIT/Apache-2.0)

5E.3: Release-Vorbereitung (Tag 2-3)
  - cargo clippy (0 Warnings)
  - cargo test (alle grün)
  - Unused-Code aufräumen (cpu_embedding_lookup etc.)
  - Version in Cargo.toml: 1.0.0
  - v1.0.0 Tag vorbereiten (KEIN Push ohne User-OK)
  Gate: Release-ready, User bestätigt
```

---

## Zusammenfassung

```
Phase 5A: DGC                       Tag 1-5       ~3 Tests
Phase 5B: Prefill Batch-Attention   Tag 6-10      ~5 Tests
Phase 5C: SPM-Tokenizer             Tag 11-12     ~4 Tests
Phase 5D: FP16-KV + vec4 GEMV       Tag 13-17     ~5 Tests
Phase 5E: Polish + Release          Tag 18-20     Audit
                                                  ─────────
                                     Subtotal     ~17 Tests
═══════════════════════════════════════════════════════════
GESAMT: 5 Schritte, ~20 Tage (~3-4 Wochen), ~17 neue Tests
        Jeder Schritt: Report → STOP → User bestätigt.
═══════════════════════════════════════════════════════════
```

## Performance-Ziele Phase 5

```
                    Phase 4D    Phase 5 Ziel    llama.cpp Vk
Decode Llama-3.1:   81.5        ≥100 tok/s      114.2
Decode Qwen3:       72.4        ≥90 tok/s       114.2
Prefill:            358         ≥2000 tok/s     4314
Modelle:            3           4+ (Mistral)    —
Context:            2048        2048 (5E: 8192?) —

Falls alle Phase-5-Ziele erreicht:
  Decode: 100/114 = 88% von llama.cpp Vulkan
  Prefill: 2000/4314 = 46% von llama.cpp Vulkan
  + DGC-Vorteil den llama.cpp NICHT hat
  + Intelligenz-Features (Monitor, Introspection) als Phase 6
```

## Abhängigkeiten

```
5A (DGC) → unabhängig, kann sofort starten
5B (Batch-Attn) → unabhängig von 5A
5C (SPM) → unabhängig, niedrig hängend
5D (FP16+vec4) → unabhängig, aber NACH 5A/5B sinnvoller
                  (dann sieht man den neuen Bottleneck klarer)
5E (Release) → NACH 5A-5D

Empfohlene Reihenfolge:
  5A → 5B → 5C → 5D → 5E
  (5C kann parallel zu 5A/5B, da unabhängig und klein)
```
