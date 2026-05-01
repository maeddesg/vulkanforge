# VulkanForge — Phase 4 Plan
# Datengetrieben aus RGP-Analyse (26.04.2026)

**Basis:** `results/phase3_rgp_analysis.md` + korrigierte scalar_attn Daten
**Ausgangslage:** Decode 61.8 tok/s, Prefill 289.5 tok/s, 15/15 kohärent

---

## RGP-Befunde die den Plan treiben

```
Shader          VGPRs    Occupancy    GPU-Anteil (pos=200)    Problem
──────────────────────────────────────────────────────────────────────────
GEMV Q4_K       88       12/16 (75%)  65%                     8 VGPRs → nächster Wave
GEMV Q6_K       88       12/16 (75%)  ~12%                    (gleicher Shader, gleiche VGPRs)
scalar_attn     192       4/16 (25%)  29%                     NICHT fixbar, braucht Ersatz
LM-Head         60       12/16 (75%)  ~5%                     3 VGPRs → nächster Wave
RMSNorm/RoPE    ?        ?            <5%                     unkritisch

Barriers:       < 1% Frame-Zeit      → KEIN Optimierungsziel
Dispatch-Gaps:  < 1 µs pro Gap       → KEIN Optimierungsziel
BW-Peak:        644.1 GB/s (nicht 608 wie angenommen)
Achieved BW:    484 GB/s = 75.2% Peak (nicht 79.6%)
Memory-bound:   bestätigt (WAITS ≈ Total cycles)
```

---

## Phase-4-Ziele

```
Decode:   61.8 → 90+ tok/s    (Stretch: 100+)
Prefill:  289.5 → 2000+ tok/s (via Batch-Attention im Prefill)
```

---

## Phase 4A: GEMV VGPR-Reduktion (1-2 Tage)

**Der billigste große Hebel: 88 → ≤80 VGPRs → 16/16 Occupancy**

### Warum zuerst

- Betrifft 65% der GPU-Zeit (GEMV Q4_K) + 12% (GEMV Q6_K) = 77%
- 12/16 → 16/16 Occupancy = +33% Wavefronts = ~25% mehr BW
- Bei 75% achieved BW → theoretisch 75% × 1.25 = 94% Peak
- Erwarteter Decode-Impact: 61.8 → ~75 tok/s

### Ansätze

```
A1: Compiler-Flags
    → shaderc -O size statt -O performance
    → SPIR-V 1.4 target
    → Aufwand: 30 Min, Ergebnis: wahrscheinlich < 5 VGPRs gespart

A2: Spec-Constants tunen
    → BLOCK_SIZE, NUM_ROWS, NUM_COLS beeinflussen Unroll-Faktor
    → Weniger Unrolling = weniger Register = bessere Occupancy
    → Aber: weniger Unrolling = weniger ILP = langsamerer Shader
    → Trade-Off: messen ob Occupancy-Gewinn > ILP-Verlust

A3: GLSL modifizieren
    → Accumulator-Variablen reduzieren (split loops)
    → Lokale Variablen in Shared Memory verschieben
    → vec4-Loads statt einzelne f32-Loads
    → Aufwand: 1-2 Tage, potenziell 10-20 VGPRs gespart

A4: RGA (Radeon GPU Analyzer) offline
    → SPIR-V → gfx1201 ISA-Disassembly
    → Zeigt exakt welche GLSL-Zeile welche VGPRs verursacht
    → Gezielte Optimierung statt blind
```

### Gate

```
Vorher:  GEMV 88 VGPRs, 12/16 Occupancy, ~484 GB/s
Nachher: GEMV ≤80 VGPRs, ≥14/16 Occupancy (ideal 16/16)
Messung: ShaderProfiler + RGP Re-Capture
Decode:  ≥ 70 tok/s (Median, 15 Prompts)
```

---

## Phase 4B: Flash-Attention (1-2 Wochen)

**Der einzige Fix für scalar_attn (192 VGPRs, 4/16, 25% Breadth)**

### Warum nach VGPR

- scalar_attn ist nur 29% der GPU-Zeit bei pos=200 (GEMV ist 65%)
- Flash-Attention ist deutlich mehr Aufwand (1155 Zeilen GLSL vs VGPR-Tweak)
- VGPR-Reduktion hilft GEMV sofort, Flash-Attention hilft nur Attention
- Aber: Flash-Attention löst ZWEI Probleme gleichzeitig:
  (1) VGPR: 192 → ~60 (Flash-Attn ist Register-effizient)
  (2) Breadth: 32 WGs → 128+ WGs (Tiling über Kontext-Chunks)

### Ansätze

```
B1: llama.cpp flash_attn.comp portieren
    → 1155 Zeilen GLSL, coopmat-Varianten
    → Bewährt, Community-optimiert
    → Risiko: Integration komplex (neues Push-Constants-Layout,
      andere Descriptor-Bindings, kausale Maske)

B2: Eigenen Flash-Attention-Shader schreiben
    → ~300 Zeilen, ohne coopmat, Wave64-optimiert
    → Chunk-basiert: K/V in TILE_SEQ Chunks, Online-SoftMax
    → Einfacher zu debuggen als llama.cpp's Version
    → Risiko: Performance möglicherweise schlechter

B3: Hybrid: llama.cpp's Decode-Pfad aus flash_attn.comp
    → Nur den Decode-Pfad (seq_len=1 new token, all-past KV)
    → Prefill-Pfad separat (Token-für-Token bleibt)
    → Weniger Code als B1, besser als B2
```

### Impact

```
scalar_attn bei pos=200:    4.6 ms → ~0.5-1.0 ms
Attention-Skalierung:       10× (aktuell) → ~2× (Flash)
Decode bei pos=200:         55 tok/s → ~65-70 tok/s
Decode bei pos=500:         42 tok/s → ~55-60 tok/s
Decode bei pos=1000:        ~25 tok/s → ~50 tok/s (geschätzt)

Prefill-Attention (in prefill_batch):
  Aktuell: Token-für-Token-Loop, O(n²)
  Mit Flash-Attention: O(n), ~2-3× Prefill-Speedup
  289 tok/s → ~600-800 tok/s Prefill (nur Attention-Anteil)
```

### Gate

```
Vorher:  scalar_attn 192 VGPRs, 4/16, 4.6 ms bei pos=200
Nachher: Flash-Attn ≤80 VGPRs, ≥12/16, < 1 ms bei pos=200
Messung: ShaderProfiler + RGP Re-Capture
Decode:  ≥ 80 tok/s bei pos=200 (Median über 15 Prompts)
Parity:  Logits Top-5 identisch zu scalar_attn
```

---

## Phase 4C: Prefill Batch-Attention (3-5 Tage)

**Den O(n²) Attention-Loop im Prefill eliminieren**

### Problem

```
Aktueller Prefill:
  GEMM für Projections: O(n) — schnell ✅
  Attention: Token-für-Token-Loop, O(n²) — langsam ❌
  
Bei pp=200:
  GEMM: ~konstant (~2 ms für alle Projections)
  Attention-Loop: 200 × tiled_attn(pos=t) = 200 × ~50 µs = 10 ms
  → Attention dominiert Prefill bei langem Prompt
```

### Lösung

Flash-Attention aus Phase 4B im Prefill nutzen:
- Alle Q-Vektoren als Batch
- KV-Cache wächst inkrementell (kausale Maske implizit)
- Ein Dispatch pro Layer statt 200 Dispatches

### Voraussetzung

Flash-Attention (Phase 4B) muss funktionieren.
Deshalb NACH 4B, nicht parallel.

### Gate

```
Prefill bei pp=200: 289 tok/s → 1500+ tok/s
Prefill bei pp=500: → 2000+ tok/s
Logits-Parity vs Token-für-Token
```

---

## Phase 4D: Multi-Modell + Polish (1 Woche)

**Feature-Erweiterung + Release-Vorbereitung**

### Multi-Modell

```
Modell 2: Llama-3.1-8B Q4_K_M
  → Anderes Chat-Template, andere Dimensionen
  → Bekannt: instruction-blind bei ROCmForge (Root Cause offen)
  → Test: funktioniert VulkanForge besser als ROCmForge?

Modell 3: DeepSeek-R1-Distill-Qwen-7B Q4_K_M
  → Think-Filter-Integration (Reasoning-Modell)

Modell 4: Mistral-7B-Instruct Q4_K_M
  → SPM-Tokenizer nötig (falls nicht deferred)
```

### Quick-Wins aus RGP

```
- LM-Head VGPR: 60 → 57 (3 VGPRs) = 13/16 Occupancy
  → Quick check ob der Compiler das schafft
  
- Elementwise VGPR-Audit: RMSNorm, RoPE, SiLU, Add, Mul
  → Pipeline-State Screenshots für jeden
  → Falls einer überraschend VGPR-limitiert: fixen

- VK_EXT_debug_utils Labels für RGP
  → Shader-Namen in RGP statt anonyme Hashes
  → 30 Min Aufwand, spart Stunden bei zukünftigem Profiling
```

### Release-Vorbereitung

```
- README.md mit Benchmarks
- CLI-Referenz (--model, --max-tokens, /reset, /quit, /stats, /think)
- Build-Anleitung (Arch Linux, Rust, Mesa, ash)
- CHANGELOG.md
- Lizenz-Attribution (llama.cpp GLSL Shader, MIT)
- GitHub Release (KEIN Tag ohne User-Bestätigung)
```

---

## Phase 4E: Backlog (nicht committed, priorisieren nach Messung)

```
Aus früheren Phasen gesammeltes Backlog:

Performance:
  □ RDNA4 Spec-Tuning für neue Flash-Attn Shader
  □ Async-Submit (< 5% Gewinn laut RGP, niedrige Prio)
  □ Batch-Elementwise im Prefill (RMSNorm/RoPE/SiLU als Batch-Dispatch)
  □ GPU Work Graphs (VK_AMDX_shader_enqueue) — experimentell, Phase 5
  □ FP8-KV-Cache mit Quality-Gate

Intelligenz-Layer (Ports aus ROCmForge):
  □ Model Introspection (SNR-Risk, Critical-Token)
  □ Quality Monitor (z-score Drift, Halluzinations-Warnung)
  □ Bandit Runtime (UCB1, Kernel-Varianten-Auswahl)
  □ Arch-Aware Sampling (SNR-basiertes repeat_penalty)

Modelle:
  □ SPM-Tokenizer (Mistral, Gemma)
  □ Gemma-4 Support
  □ MoE-Support (Qwen3-30B-A3B)
  □ 14B-Modelle

CPU:
  □ AVX-512 VNNI für GA Parity-Validation (15k Checks/GA-Run)
  □ AVX-512 FMA für CPU-Referenz-Forward (Proxy-Loss)
```

---

## Zeitplan-Schätzung

```
Phase 4A: GEMV VGPR-Reduktion        1-2 Tage     → 75 tok/s Decode
Phase 4B: Flash-Attention             1-2 Wochen   → 90 tok/s Decode
Phase 4C: Prefill Batch-Attention     3-5 Tage     → 2000 tok/s Prefill
Phase 4D: Multi-Modell + Polish       1 Woche      → Release-Ready
                                      ─────────
                                      3-4 Wochen

Erwartete Endperformance:
                          Phase 3 Final   Phase 4 Ziel   llama.cpp Vk
  Decode tok/s (median):  61.8            90+            114.2
  Prefill tok/s:          289.5           2000+          4314
  Modelle:                1 (Qwen3)       3-4            alle
  15/15 kohärent:         ✅              ✅             —
```

---

## Risiken

```
R1: VGPR-Reduktion bringt nichts weil Trade-Off ILP > Occupancy
    → Wahrscheinlichkeit: 30%
    → Mitigation: RGA-Analyse VOR Shader-Änderung zeigt ob es sich lohnt
    → Fallback: Phase 4B (Flash-Attn) zuerst

R2: Flash-Attention-Integration > 2 Wochen
    → Wahrscheinlichkeit: 40%
    → Mitigation: B3 (Hybrid, nur Decode-Pfad) als Fallback
    → Mindest-Impact: scalar_attn bei pos=200 von 4.6ms → 1ms

R3: VGPR-Reduktion bricht Correctness
    → Wahrscheinlichkeit: 10%
    → Mitigation: GEMM-Parity-Tests, Logits Top-5, 15-Prompt-Suite
    → Jede Änderung: cargo test + ShaderProfiler + Logits-Vergleich

R4: Multi-Modell deckt neue Bugs auf
    → Wahrscheinlichkeit: 50% (ROCmForge hatte instruction-blind Bug)
    → Mitigation: Per-Modell 5-Prompt-Suite, systematisches Debugging
```

---

## Meilensteine

```
M1: GEMV ≤80 VGPRs, Decode ≥70 tok/s              → Phase 4A Gate
M2: Flash-Attn funktioniert, Decode ≥80 tok/s      → Phase 4B Gate
M3: Prefill ≥1500 tok/s mit Batch-Attention         → Phase 4C Gate
M4: 3+ Modelle, README, Release-Ready               → Phase 4D Gate

15-Prompt-Benchmark nach JEDEM Meilenstein.
RGP Re-Capture nach M1 und M2.
```
