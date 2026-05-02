# VulkanForge — Projektplan

**Erstellt:** 25.04.2026
**Basis:** vulkanforge_vision.md
**Strategie:** llama.cpp GLSL Vulkan-Shader + ROCmForge Rust-Infrastruktur + eigener Vulkan-Dispatch (Option C)

---

## Überblick

```
Phase 1: PoC — 1 Kernel (1–2 Wochen, 6 Schritte)
  → EIN llama.cpp GLSL Shader (mul_mat_vec Q4_K) via ash dispatchen
  → Parity + Performance-Messung
  → GATE: Kernel korrekt, BW > 30%

Phase 2: Minimaler Inference-Loop (2–3 Wochen, 10 Schritte)
  → Qwen3-8B Q4_K_M laden + Tokens generieren
  → Alle Decode-Shader + Attention + Norm + Sampling
  → GATE: Kohärenter Text, Decode ≥ 100 tok/s

Phase 3: Feature-Parity mit ROCmForge (2–3 Wochen, 9 Schritte)
  → Multi-Turn, Streaming, Prefill, Bandit, Monitor, Introspection
  → 15-Prompt-Suite
  → GATE: 15/15 kohärent, Decode ≥ 110 tok/s, Prefill ≥ 3000 tok/s

Gesamt: ~6–8 Wochen, 25 Schritte
```

---

## Lehren aus ROCmForge die den Plan formen

```
Lektion 1: Eigene Kernel schreiben = 33% Trefferquote
  → KONSEQUENZ: llama.cpp Shader DIREKT nutzen, KEINE eigenen Kernel

Lektion 2: JEDER große Perf-Gewinn kam aus llama.cpp Kernel-Ports
  → KONSEQUENZ: Gleich mit llama.cpp Shader starten, nicht portieren

Lektion 3: HIP/ROCm crasht ab pp>4096 auf Consumer-GPUs
  → KONSEQUENZ: Vulkan ist stabiler (getestet bis 16k+)

Lektion 4: Profiling invertiert IMMER die Schätzungen
  → KONSEQUENZ: Messen nach JEDEM Schritt

Lektion 5: Integration ist wo Bugs entstehen
  → KONSEQUENZ: Shader-Interface EXAKT analysieren (Schritt 1.0) bevor Code

Lektion 6: 6 von 9 Optimierungen waren negativ
  → KONSEQUENZ: Keine Optimierung in Phase 1+2, erst korrekt dann schnell

Lektion 7: Vulkan = 30% schneller + 36% effizienter als ROCm auf RDNA4
  → KONSEQUENZ: Vulkan als primäres Backend, nicht ROCm
```

---

## Was VulkanForge NICHT braucht (weil llama.cpp Shader es lösen)

```
❌ Dequant IR              → Shader dequantisiert intern
❌ Eigene GPU-Kernel       → llama.cpp GLSL (MIT-lizenziert)
❌ HIP FFI Bindings        → ash (1:1 Vulkan-API)
❌ GA Kernel-Tuning        → Shader sind Community-optimiert
❌ FP8-WMMA Codegen        → Shader nutzen KHR_coopmat wo verfügbar
❌ hipcc Build-System      → shaderc GLSL→SPIR-V
```

## Was VulkanForge WIEDERVERWENDET (80% aus ROCmForge)

```
✅ GGUF-Parser             → Tensor-Inventar, Modell-Config, Weight-Loading
✅ Tokenizer + Chat-Templates → BPE, 7+ Template-Varianten, Disambiguation
✅ Model Introspection     → SNR-Risk, Critical-Token-Erkennung
✅ Quality Monitor         → z-score Drift, Halluzinations-Warnung
✅ Bandit Runtime (UCB1)   → Kernel-Varianten-Auswahl (Shader-Pipelines statt HIP-Kernel)
✅ Streaming + Think-Filter → Token-für-Token, <think>-Tag Filterung
✅ CLI + Chat-REPL         → /reset, /quit, Modell-Report
✅ Multi-Turn KV-Persistence → Session-übergreifend
✅ Arch-Aware Sampling     → SNR-basiertes repeat_penalty
```

## Was VulkanForge NEU schreibt (5% — Backend-Swap)

```
🆕 Vulkan-Device (ash)     → VkInstance, VkDevice, VkQueue
🆕 Vulkan-Memory (gpu-alloc) → Arena-Allokation auf VkDeviceMemory
🆕 Vulkan-Pipeline         → SPIR-V laden, VkPipeline, Descriptor-Sets
🆕 Vulkan-Dispatch         → Command-Buffer, vkCmdDispatch, Sync
🆕 Pipeline-Cache          → Shader-Kompilierung einmal, dann cached
🆕 Shader-Extraktion       → llama.cpp GLSL → eigene vk_shaders/ Kopie
```

---

# Phase 1: PoC — 1 Kernel (1–2 Wochen)

**Ziel:** BEWEIS dass llama.cpp GLSL Shader via ash in Rust korrekt dispatcht werden können.
**Detaillierter Prompt:** `vulkanforge_phase1_poc_prompt.md`

---

## Schritt 1.0: Shader-Analyse (Tag 1, NUR lesen)

```
- llama.cpp Vulkan-Shader-Quellen analysieren:
  ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/
- Richtigen Shader finden (mul_mat_vec Q4_K)
- Schnittstelle dokumentieren:
  Push-Constants, Descriptor-Set-Layout, Buffer-Formate,
  Workgroup-Size, Dispatch-Dimensionen
- Alle Dependencies identifizieren (#includes, #defines, Generator-Scripts)
- Q4_K Block-Layout dokumentieren (scales, mins, nibbles, byte-offsets)
```

**STOP-Bedingung:** Falls Shader durch Python-Generator erzeugt wird → STOP, Ansatz überdenken.

**Gate:** Vollständige Shader-Analyse als Report. User bestätigt.

**Report:** `results/phase1_step_1.0_shader_analysis.md`

---

## Schritt 1.1: Shader-Extraktion + SPIR-V (Tag 2)

```
- Shader + Dependencies nach vk_shaders/ kopieren
- build.rs: GLSL → SPIR-V via shaderc mit korrekten #defines
- SPIR-V einbetten (include_bytes!) oder zur Laufzeit laden
```

**Tests:**
```
[build] cargo build kompiliert fehlerfrei
[build] SPIR-V wird erzeugt (Datei existiert, > 0 Bytes)
[build] Keine Shader-Compile-Warnings
```

**Gate:** Build grün, SPIR-V erzeugt.

**Report:** `results/phase1_step_1.1_shader_extraction.md`

---

## Schritt 1.2: Vulkan Pipeline Setup (Tag 2–3)

```
- VkDescriptorSetLayout (Bindings aus Schritt 1.0)
- Push-Constant-Range (Rust struct, #[repr(C)])
- VkShaderModule aus SPIR-V
- VkPipelineLayout + VkComputePipeline
- VK_LAYER_KHRONOS_validation MUSS aktiv sein
```

**Tests:**
```
[vk] Pipeline erstellt ohne Validation-Errors
[vk] Validation-Layers aktiv (Log-Output bestätigt)
```

**Gate:** Pipeline ohne Errors. 0 Validation-Warnings.

**Report:** `results/phase1_step_1.2_pipeline_setup.md`

---

## Schritt 1.3: Buffer-Allokation + Testdaten (Tag 3)

```
- Synthetische Q4_K-Blöcke (bekannte Scales/Mins/Nibbles)
- Einfacher f32-Input-Vektor (z.B. all-ones oder [1,2,3,...])
- GPU-Buffer via gpu-alloc (weights, input, output)
- Staging-Upload (HOST_VISIBLE → DEVICE_LOCAL)
- CPU-Referenz: Q4_K Dequant + MatVec in f32
```

**Tests:**
```
[mem] Buffers alloziert (DEVICE_LOCAL = tatsächlich VRAM)
[mem] Staging-Upload funktioniert (readback = original)
[cpu] CPU-Referenz berechnet (Ergebnis plausibel)
```

**Gate:** Buffers ready, CPU-Referenz existiert.

**Report:** `results/phase1_step_1.3_buffer_testdata.md`

---

## Schritt 1.4: Dispatch + Smoke-Test (Tag 3–4)

```
- Descriptor-Set schreiben (Buffers → Bindings)
- Command-Buffer: Bind Pipeline → Push Constants → Dispatch
- Submit + Fence-Wait
- Output readback + Validierung
```

**Smoke-Test-Checks:**
```
[smoke] Output ist NICHT all-zeros
[smoke] Output ist NICHT NaN/Inf
[smoke] Output-Dimensionen stimmen
[smoke] Output-Werte im erwarteten Bereich
[parity] max_abs_error vs CPU-Referenz < 1e-2
[parity] max_rel_error < 5%
```

**STOP-Bedingung:** Output all-zeros oder NaN → STOP, systematisch debuggen, NICHT blind probieren.

**Gate:** Alle Smoke-Checks bestanden. Finite, nicht-triviale Ergebnisse.

**Report:** `results/phase1_step_1.4_dispatch_smoke_test.md`
  → Output-Werte (erste 16), CPU-Referenz (erste 16), Errors, Kernel-Zeit (µs)

---

## Schritt 1.5: Skalierungstest (Tag 4–5)

```
- Realistische Dimensionen (Qwen3-8B: N=3584, K=3584)
- Random-Weights (gültiges Q4_K-Format)
- Smoke-Test wiederholen mit großen Dimensionen
- 100× Dispatch: Median, P95, Stddev
- Effektive BW berechnen: bytes_read / kernel_time
- BW-Auslastung vs Peak (RX 9070 XT: ~608 GB/s)
```

**Tests:**
```
[scale] Output != zeros, != NaN bei realistischen Dimensionen
[perf]  BW-Auslastung > 30% (< 30% = fundamentales Problem)
[perf]  Varianz über 100 Runs < 10% (stabile Performance)
```

**Gate:** Kernel skaliert. BW > 30%.

**Report:** `results/phase1_step_1.5_scaling_test.md`
  → Dimensionen, Kernel-Zeit (Median/P95/Stddev), BW (GB/s), BW-Auslastung (%)
  → Vergleich: llama.cpp Vulkan erreicht ~80% BW

---

**Phase 1 ABGESCHLOSSEN wenn Schritt 1.5 Gate bestanden.**

```
Phase-1-Erfolgskriterien (aus Vision):
  ☐ 1 GLSL Shader läuft korrekt via ash in Rust
  ☐ Parity mit CPU-Referenz (< 1e-2)
  ☐ Kernel-Zeit innerhalb plausiblem Bereich
  ☐ BW-Auslastung > 30%
```

---

# Phase 2: Minimaler Inference-Loop (2–3 Wochen)

**Ziel:** Qwen3-8B Q4_K_M laden und kohärenten Text generieren.
**Performance-Ziel:** Decode ≥ 100 tok/s

---

## Schritt 2.1: Shader-Inventar + SPIR-V Bulk-Kompilierung (Tag 6–7)

```
Alle benötigten Shader aus llama.cpp extrahieren:

Decode-Pfad (GEMV):
  mul_mat_vec_q4_k.comp     ← Q4_K Decode (aus Phase 1)
  mul_mat_vec_q6_k.comp     ← Q6_K Decode

Attention:
  flash_attn_f16.comp       ← Flash-Attention (Decode-Pfad)
  ODER: eigene Scalar-Attention als Fallback (wie ROCmForge)

Elementweise:
  norm.comp / rms_norm.comp ← RMSNorm
  rope.comp                 ← RoPE Embedding
  add.comp                  ← Residual-Addition
  mul.comp                  ← Element-Multiply (SiLU gate)
  silu.comp                 ← SiLU Activation

Utility:
  quantize_q8_1.comp        ← Activation-Quantize (falls GEMV Q8-Input braucht)
  dequant_q4_k.comp         ← Falls separat von GEMV nötig
  copy.comp                 ← Buffer-Copy

Analyse PRO Shader:
  → Push-Constants, Descriptor-Layout, Workgroup-Size, Dispatch-Dims
  → Dependencies (#includes)
  → Zusammenfassung in Tabelle
```

**Gate:** Alle Shader kompilieren zu SPIR-V. Analyse-Tabelle komplett.

**Report:** `results/phase2_step_2.1_shader_inventory.md`
  → Tabelle: Shader | Bindings | Push-Constants | Workgroup | Dispatch-Formel

---

## Schritt 2.2: Pipeline-Registry (Tag 7–8)

```
- PipelineRegistry: HashMap<ShaderId, ComputeKernel>
- Alle Shader als VkComputePipeline registrieren
- Einheitlicher Dispatch-Aufruf:
    registry.dispatch(ShaderId::MulMatVecQ4K, &buffers, &push_constants, dims)
- Pipeline-Cache auf Disk (VkPipelineCache → Datei)
  → Erster Start: Shader kompilieren (~Sekunden)
  → Folgestarts: Cache laden (< 100ms)
```

**Tests:**
```
[reg]   Alle Shader registriert (count == erwartete Anzahl)
[reg]   Dispatch-Aufruf für jeden Shader ohne Validation-Error
[cache] Pipeline-Cache geschrieben (Datei existiert)
[cache] Zweiter Start lädt Cache (Zeitvergleich: 10× schneller)
```

**Gate:** Registry funktioniert. Pipeline-Cache aktiv.

**Report:** `results/phase2_step_2.2_pipeline_registry.md`

---

## Schritt 2.3: VRAM-Arena für Vulkan (Tag 8–9)

```
- Analogie zu ROCmForge VRAM-Arena, aber mit VkDeviceMemory:
  1× vkAllocateMemory für bulk (z.B. 14 GB)
  Zonen: A (Weights), B (KV-Cache), C (Scratch/Activations)
  Offset-Arithmetik, Alignment (minStorageBufferOffsetAlignment)
- VkBuffer-Views: Unter-Buffer aus Arena-Offsets
  → vkBindBufferMemory mit Offset
- Ping-Pong-Scratchpad für Zwischen-Ergebnisse (Zone C)
```

**Tests:**
```
[arena] 14 GB alloziert auf DEVICE_LOCAL heap
[arena] 3 Zonen passen ohne Überlappung
[arena] Alignment respektiert (minStorageBufferOffsetAlignment)
[arena] Scratch Ping-Pong wechselt korrekt
[arena] OOM → sauberer Error
```

**Gate:** Arena alloziert, Zonen korrekt.

**Report:** `results/phase2_step_2.3_vram_arena.md`

---

## Schritt 2.4: GGUF-Loader (Port aus ROCmForge) (Tag 9–11)

```
Wiederverwendung aus ROCmForge:
  → GGUF-Header-Parser (Magic, Version, Metadata, Tensor-Info)
  → Modell-Konfiguration aus Metadaten
  → Tensor-Inventar + Layer-Gruppierung

Anpassung:
  → Weights laden in Vulkan-Arena (Zone A) statt HIP-Arena
  → Staging-Upload per VkCommandBuffer statt hipMemcpy
  → Tensor → Arena-Offset-Map
```

**Tests:**
```
[gguf] Qwen3-8B Q4_K_M Header korrekt geparst
[gguf] n_layers=36, hidden=3584, n_heads=28, n_kv_heads=4
[gguf] Tensor-Count korrekt
[gguf] Q4_K und Q6_K Quant-Typen erkannt
[load] Alle Tensoren in VRAM-Arena (Zone A)
[load] 256-Byte-Alignment
[load] Spot-Check: 10 zufällige Blöcke readback = Disk
[load] Load-Zeit < 5s
```

**Gate:** Modell geladen, Tensor-Map korrekt.

**Report:** `results/phase2_step_2.4_gguf_loader.md`

---

## Schritt 2.5: Elementweise Shader Validierung (Tag 11–12)

```
Jeden elementweisen Shader einzeln testen:

RMSNorm:
  → Bekannter Input → erwarteter Output (CPU-Referenz)
  → Toleranz: max_abs_err < 1e-5

RoPE:
  → Position 0, 1, 100 → erwartete sin/cos Werte
  → Prüfe rope_freq_base aus GGUF

SiLU:
  → silu(x) = x * sigmoid(x), CPU-Referenz trivial

Residual-Add:
  → a + b = c, trivial

Activation-Quantize (q8_1):
  → Float → Q8_1 → Dequant zurück ≈ Original (Toleranz)
```

**Tests (pro Shader):**
```
[norm]  RMSNorm Output vs CPU-Referenz < 1e-5
[rope]  RoPE Position 0 korrekt
[rope]  RoPE Position 100 korrekt
[silu]  SiLU vs CPU < 1e-6
[add]   Residual-Add exakt
[quant] Q8_1 round-trip error < 0.5%
```

**Gate:** Alle elementweisen Shader korrekt.

**Report:** `results/phase2_step_2.5_elementwise_validation.md`

---

## Schritt 2.6: Single-Layer Forward Pass (Tag 12–14)

```
EINEN Transformer-Layer durchrechnen:

  Input: Random Activations (passende Dimensionen)

  Ablauf:
  1. RMSNorm (attn_norm)
  2. Q/K/V Projection (3× GEMV Q4_K)
  3. Q/K Norm (Qwen3-spezifisch)
  4. RoPE auf Q und K
  5. KV-Cache Write
  6. Attention (Score + Softmax + V-Multiply)
  7. Output Projection (GEMV Q4_K)
  8. Residual Add
  9. RMSNorm (ffn_norm)
  10. Gate + Up Projection (2× GEMV Q4_K)
  11. SiLU(gate) * up
  12. Down Projection (GEMV Q6_K)
  13. Residual Add

  → Alles in EINEM VkCommandBuffer aufnehmen
  → Pipeline-Barriers zwischen Compute-Schritten
  → Korrektes Memory-Barrier-Handling (RAW-Dependencies)
```

**Tests:**
```
[layer] Output != zeros, != NaN
[layer] Output-Dimensionen = Input-Dimensionen (hidden_dim)
[layer] Keine Validation-Layer-Errors
[layer] Korrekte Barrier-Platzierung (kein Race-Condition)
[perf]  Single-Layer-Zeit messen (Baseline für 36-Layer-Extrapolation)
```

**STOP-Bedingung:** Validation-Layer-Error bei Barriers → STOP. Memory-Barrier-Bugs sind subtil und gefährlich.

**Gate:** 1 Layer korrekt durchgerechnet. Keine Validation-Errors.

**Report:** `results/phase2_step_2.6_single_layer.md`
  → Layer-Zeit, Breakdown pro Shader, Barrier-Schema

---

## Schritt 2.7: Full Forward Pass — 36 Layer (Tag 14–16)

```
Kette von 36 Layern:

  1. Token-Embedding Lookup
  2. 36× Transformer-Layer (Schritt 2.6 Ablauf)
  3. Final RMSNorm
  4. LM-Head (Embedding → Logits, GEMV oder tied weights)

  → Embedding: Lookup-Tabelle, kein Compute-Shader nötig
    (CPU-seitig oder einfacher Copy-Shader)
  → LM-Head: GEMV mit vocab_size Output-Dimension

Command-Buffer-Strategie:
  → Option A: 1 großer Command-Buffer für alle 36 Layer
  → Option B: 1 Command-Buffer pro Layer (einfacher zu debuggen)
  → Phase 2: Option B (Debugging). Phase 3: Option A (Performance).
```

**Tests:**
```
[fwd] Logits-Output: vocab_size Elemente (151936 für Qwen3)
[fwd] Logits != all-zeros, != NaN
[fwd] Logits-Summe ist endlich und im erwarteten Bereich
[fwd] Top-5 Token-IDs sind plausibel (nicht alle gleich)
[perf] Forward-Pass-Zeit für 1 Token messen
       → Erwartung: < 10ms (= 100+ tok/s)
```

**Gate:** Logits korrekt. Keine Validation-Errors.

**Report:** `results/phase2_step_2.7_full_forward.md`
  → Logits-Statistik, Forward-Zeit, Layer-Breakdown

---

## Schritt 2.8: Tokenizer + Sampling (Tag 16–17)

```
Wiederverwendung aus ROCmForge:
  → BPE-Tokenizer (GGUF-Vokabular)
  → Chat-Template Disambiguation (7+ Varianten)
  → Greedy-Sampling (argmax über Logits)
  → Temperature-Sampling (optional, Greedy reicht für Gate)

Anpassung:
  → Logits readback: GPU → CPU (VkBuffer map)
  → Sampling auf CPU (wie ROCmForge)
  → Nächstes Token → Embedding → nächster Forward-Pass
```

**Tests:**
```
[tok] Tokenizer lädt Vokabular aus GGUF
[tok] "Hello world" → Token-IDs → "Hello world" (roundtrip)
[tok] Chat-Template erkannt (Qwen3)
[tok] Special Tokens korrekt (BOS, EOS, <|im_start|>)
[sample] Greedy auf bekannten Logits → erwartetes Token
```

**Gate:** Tokenizer + Sampling funktioniert.

**Report:** `results/phase2_step_2.8_tokenizer_sampling.md`

---

## Schritt 2.9: Decode-Loop — Erste Tokens (Tag 17–19)

```
End-to-End Decode-Loop:

  1. User-Prompt tokenisieren
  2. Prefill: Forward-Pass über alle Prompt-Tokens
     (Phase 2: Token-für-Token, kein Batch-Prefill)
  3. KV-Cache befüllt
  4. Decode-Loop:
     a. Forward-Pass (1 Token)
     b. Logits readback
     c. Greedy-Sample
     d. Token in KV-Cache eintragen
     e. Token ausgeben
     f. Repeat bis EOS oder max_tokens

  Test-Prompt: "Explain what a mutex is in one sentence."
```

**Tests:**
```
[e2e] Output ist nicht leer
[e2e] Output enthält englische Wörter (nicht Garbage)
[e2e] Output endet (EOS oder max_tokens erreicht)
[e2e] Kein VRAM-Leak (Arena-Nutzung stabil über 100 Tokens)
[e2e] Keine Validation-Layer-Errors
```

**STOP-Bedingung:** Garbage-Output → STOP. Mögliche Ursachen systematisch prüfen:
  (1) Embedding-Lookup falsch, (2) RoPE-Position falsch, (3) KV-Cache-Indexing falsch,
  (4) Attention-Masking falsch, (5) LM-Head falsch.

**Gate:** Kohärenter englischer Text als Antwort auf den Test-Prompt.

**Report:** `results/phase2_step_2.9_first_tokens.md`
  → Prompt + Output (komplett), Decode-Zeit pro Token, Gesamt-tok/s

---

## Schritt 2.10: Phase-2-Validierung (Tag 19–21)

```
Systematischer Test über mehrere Prompts:

  Prompt 1: "Explain what a mutex is in one sentence."
  Prompt 2: "Write a haiku about programming."
  Prompt 3: "What is 2 + 2?"
  Prompt 4: "Translate 'hello world' to German."
  Prompt 5: "List three prime numbers."

Performance-Messung:
  → Decode tok/s (Median über 5 Prompts)
  → Forward-Pass-Breakdown (Attention vs GEMV vs Elementwise)
  → VRAM-Nutzung
  → Vergleich mit llama.cpp Vulkan (114 tok/s Decode)
```

**Tests:**
```
[val] 5/5 Prompts: kohärenter Output
[val] Keine Crashes, keine Validation-Errors
[perf] Decode ≥ 100 tok/s (Phase-2-Ziel)
[mem] VRAM < 14 GB (Qwen3-8B Q4_K_M + KV-Cache)
```

**Gate:** 5/5 kohärent, Decode ≥ 100 tok/s.

**Report:** `results/phase2_step_2.10_validation.md`
  → Performance-Tabelle, Prompt/Output-Paare, VRAM-Breakdown

---

**Phase 2 ABGESCHLOSSEN wenn Schritt 2.10 Gate bestanden.**

```
Phase-2-Erfolgskriterien (aus Vision):
  ☐ Qwen3-8B Q4_K_M: Decode ≥ 100 tok/s
  ☐ Kohärenter Text-Output (5/5 Prompts)
  ☐ GGUF-Laden funktioniert
```

---

# Phase 3: Feature-Parity mit ROCmForge (2–3 Wochen)

**Ziel:** Alle ROCmForge-Differenzierungs-Features portiert + Prefill + 15-Prompt-Suite.
**Performance-Ziel:** Decode ≥ 110 tok/s, Prefill ≥ 3000 tok/s

---

## Schritt 3.1: Prefill — GEMM-Shader + Flash-Attention (Tag 22–25)

```
Decode war Token-für-Token (GEMV). Prefill ist Batch (GEMM):

Shader:
  mul_mat_q4_k.comp         ← Q4_K GEMM (Batch-Prefill)
  mul_mat_q6_k.comp         ← Q6_K GEMM
  flash_attn_f16.comp       ← Flash-Attention (Prefill-Pfad)

Integration:
  → Prompt-Tokens als Batch verarbeiten (nicht einzeln)
  → KV-Cache in einem Durchlauf befüllen
  → Flash-Attention statt Scalar-Attention

Performance-Erwartung:
  → Prefill ≥ 3000 tok/s bei pp=512
  → llama.cpp Vulkan: 4314 tok/s pp=512
```

**Tests:**
```
[gemm]  Q4_K GEMM: Output vs GEMV (Token-für-Token) identisch
[gemm]  Q6_K GEMM: ditto
[flash] Flash-Attention Output vs Scalar-Attention < 1e-3
[e2e]   Prefill-Output (Logits nach Prompt) identisch zu Decode-Pfad
[perf]  Prefill tok/s bei pp=512
```

**Gate:** Prefill korrekt und ≥ 3000 tok/s.

**Report:** `results/phase3_step_3.1_prefill.md`

---

## Schritt 3.2: Multi-Turn + Streaming + Think-Filter (Tag 25–27)

```
Port aus ROCmForge (Rust-Code, Backend-unabhängig):

Multi-Turn:
  → KV-Cache zwischen Turns persistieren
  → Context-Window-Management (Sliding oder Truncation)

Streaming:
  → Token-für-Token Ausgabe (callback/channel)
  → Partial UTF-8 Handling

Think-Filter:
  → <think>...</think> Tags erkennen und filtern
  → Nur sichtbaren Content streamen

CLI:
  → Interactive Chat-REPL (/reset, /quit, /stats)
  → Modell-Info beim Start anzeigen
```

**Tests:**
```
[mt]     Turn 1 → Turn 2: Kontext bleibt erhalten
[mt]     /reset: KV-Cache geleert, frischer Start
[stream] Tokens kommen einzeln an (nicht als Block)
[think]  <think>reasoning</think>Answer → nur "Answer" gestreamt
[cli]    /quit beendet sauber (kein VRAM-Leak)
```

**Gate:** Multi-Turn + Streaming + Think-Filter funktionieren.

**Report:** `results/phase3_step_3.2_multi_turn_streaming.md`

---

## Schritt 3.3: Model Introspection (Port aus ROCmForge) (Tag 27–28)

```
Port aus ROCmForge:
  → Automatische Modell-Erkennung aus GGUF-Metadata
  → SNR-Risk-Score pro Layer (Quantisierungs-Qualität)
  → Critical-Token-Erkennung (Embedding-Anomalien)
  → PrecisionHint pro Layer (FP8 vs FP16 Empfehlung)
  → Modell-Qualitäts-Report beim Laden

Anpassung:
  → Tensor-Daten aus Vulkan-Arena lesen (statt HIP-Buffer)
  → SNR-Berechnung auf CPU (readback nötig, einmalig beim Laden)
```

**Tests:**
```
[snr]   Qwen3-8B: SNR > 2.0 für alle Layer (bekannt gut)
[snr]   Llama-3.1-8B: Special-Token-Layer SNR < 1.0 erkannt
[detect] Qwen3 als "qwen2" Architecture erkannt
[report] Modell-Report wird beim Laden angezeigt
```

**Gate:** Introspection korrekt. Bekannte Modell-Eigenheiten erkannt.

**Report:** `results/phase3_step_3.3_introspection.md`

---

## Schritt 3.4: Quality Monitor (Port aus ROCmForge) (Tag 28–29)

```
Port aus ROCmForge:
  → z-score Drift-Detection pro Token
  → Halluzinations-Warnung bei statistischer Abweichung
  → RepetitionDetector (Loop-Erkennung)
  → Automatische Warnung in Streaming-Output

Anpassung:
  → Logits aus Vulkan-Buffer statt HIP-Buffer
  → Monitor läuft auf CPU (Logits-Readback pro Token)
```

**Tests:**
```
[drift]  Normaler Output: 0 Monitor-Events
[drift]  Erzwungene Repetition: RepetitionDetector schlägt an
[drift]  Stark gestörte Logits: z-score Alert
[warn]   Warnung erscheint im Streaming-Output
```

**Gate:** Monitor erkennt Anomalien korrekt.

**Report:** `results/phase3_step_3.4_quality_monitor.md`

---

## Schritt 3.5: Bandit Runtime (Port aus ROCmForge) (Tag 29–30)

```
Port aus ROCmForge:
  → UCB1 Bandit wählt optimale Kernel-Variante pro Shape
  → Lernt pro Modell + GPU-Kombination
  → Pipeline-Caching nach Konvergenz
  → Bandit-State-Persistenz über Sessions

Anpassung:
  → "Kernel-Variante" = VkPipeline-ID (statt HIP-Kernel-Pointer)
  → Timing via vkCmdWriteTimestamp (statt hipEvent)
  → Varianten: z.B. verschiedene Workgroup-Sizes oder Shader-Versionen

Hinweis: In Phase 3 hat VulkanForge möglicherweise NUR eine Variante
pro Shader (die llama.cpp Default-Variante). Bandit-Infrastruktur
trotzdem aufbauen — Varianten kommen in Phase 4.
```

**Tests:**
```
[bandit] Bandit registriert Varianten korrekt
[bandit] UCB1-Auswahl konvergiert (bei 1 Variante: trivial)
[bandit] State-Persistenz: Save → Load → gleiche Auswahl
[bandit] Timing-Integration: vkCmdWriteTimestamp liefert plausible Werte
```

**Gate:** Bandit-Infrastruktur funktioniert.

**Report:** `results/phase3_step_3.5_bandit_runtime.md`

---

## Schritt 3.6: Arch-Aware Sampling (Port aus ROCmForge) (Tag 30–31)

```
Port aus ROCmForge:
  → SNR-basiertes Sampling (repeat_penalty automatisch)
  → SNR < 2.0 → penalty = 1.1
  → FP8-KV-Cache mit Quality-Gate
  → Temperature, Top-K, Top-P Sampling
```

**Tests:**
```
[sample] repeat_penalty automatisch für Qwen3 (SNR > 2.0 → penalty = 1.0)
[sample] repeat_penalty automatisch für Llama-3.1 (SNR-Edge → penalty = 1.1)
[sample] Temperature-Sampling: höhere Temp → mehr Varianz
[sample] Top-K: nur Top-K Tokens haben Wahrscheinlichkeit > 0
```

**Gate:** Sampling-Strategien korrekt.

**Report:** `results/phase3_step_3.6_sampling.md`

---

## Schritt 3.7: Multi-Modell-Support (Tag 31–33)

```
Über Qwen3-8B hinaus:

Modell 2: Llama-3.1-8B Q4_K_M
  → Anderer Chat-Template, andere Dimensionen
  → Bekanntes Problem: instruction-blind bei ROCmForge (Root Cause offen)

Modell 3: DeepSeek-R1-Distill-Qwen-7B Q4_K_M
  → Think-Filter-Integration (Reasoning-Modell)

Modell 4: Mistral-7B-Instruct Q4_K_M
  → Anderer Tokenizer-Typ (SPM vs BPE)
  → Falls SPM noch nicht implementiert: DEFERRED notieren

Pro Modell:
  → 5-Prompt-Test (aus Schritt 2.10)
  → Kohärenz-Check
  → Performance-Messung
```

**Tests:**
```
[llama]   5/5 kohärent (oder instruction-blind wie ROCmForge → dokumentieren)
[ds]      5/5 kohärent, Think-Filter aktiv
[mistral] 5/5 kohärent ODER SPM-Blocker dokumentiert
```

**Gate:** ≥ 3 Modelle funktionieren.

**Report:** `results/phase3_step_3.7_multi_model.md`

---

## Schritt 3.8: Performance-Optimierung (Tag 33–35)

```
Optimierungen ohne Shader-Änderungen:

1. Command-Buffer-Strategie:
   → 1 großer Command-Buffer statt 36 kleine
   → Oder: Secondary Command-Buffers, wiederverwendbar

2. Pipeline-Barriers minimieren:
   → Nur wo echte RAW-Dependencies bestehen
   → Batch-Barriers wo möglich

3. Descriptor-Set-Pooling:
   → Pre-allokierte Descriptor-Sets, nicht pro-Dispatch neu

4. Pipeline-Cache prüfen:
   → Startup-Zeit mit Cache < 1s

5. Profiling:
   → vkCmdWriteTimestamp pro Shader-Kategorie
   → Bottleneck identifizieren
```

**Messung:**
```
Ziele:
  Decode Qwen3-8B Q4_K_M:  ≥ 110 tok/s (llama.cpp: 114)
  Prefill pp=512:           ≥ 3000 tok/s (llama.cpp: 4314)
  Startup (mit Cache):      < 2s
  VRAM:                     < 14 GB

Falls Decode < 100 tok/s nach Optimierung:
  → Profiling-Report: wo geht die Zeit hin?
  → Vergleich: Dispatch-Overhead vs Kernel-Zeit vs Readback
```

**Report:** `results/phase3_step_3.8_optimization.md`
  → Performance-Tabelle, Bottleneck-Analyse, Vorher/Nachher

---

## Schritt 3.9: Finale Validierung — 15-Prompt-Suite (Tag 35–37)

```
Vollständiger Benchmark mit ROCmForge 15-Prompt-Suite:

  → Alle 15 Prompts aus inference_test_prompts_15.json
  → Qwen3-8B Q4_K_M
  → Decode tok/s pro Prompt + Median
  → Prefill tok/s (pp=512)
  → Kohärenz: 15/15 Prompts müssen kohärenten Output erzeugen
  → Multi-Turn: 3 Prompts als Multi-Turn-Konversation
  → Streaming: Visueller Check (Tokens kommen einzeln)
  → Monitor: 0 false-positive Warnungen bei normalem Betrieb
  → Introspection: Modell-Report korrekt

Performance-Vergleich:
  → VulkanForge vs ROCmForge vs llama.cpp Vulkan vs llama.cpp ROCm
  → Tabelle mit allen 4 Systemen
```

**Tests:**
```
[suite]  15/15 Prompts: kohärenter Output
[perf]   Decode ≥ 110 tok/s (Median über 15 Prompts)
[perf]   Prefill ≥ 3000 tok/s (pp=512)
[mt]     Multi-Turn: Kontext bleibt über 3 Turns
[stream] Streaming funktioniert für alle 15 Prompts
[mon]    0 false-positive Monitor-Events
[intro]  Modell-Report wird angezeigt
```

**Gate:** ALLE Tests bestanden. Performance-Ziele erreicht.

**Report:** `results/phase3_step_3.9_final_validation.md`
  → Performance-Tabelle (4-System-Vergleich)
  → 15 Prompt/Output-Paare
  → Modell-Report
  → VRAM-Nutzung

---

**Phase 3 ABGESCHLOSSEN wenn Schritt 3.9 Gate bestanden.**

```
Phase-3-Erfolgskriterien (aus Vision):
  ☐ Decode ≥ 110 tok/s (vs llama.cpp Vulkan 114)
  ☐ Prefill ≥ 3000 tok/s
  ☐ 15/15 Prompt-Kohärenz
  ☐ Multi-Turn + Streaming + Think-Filter
  ☐ Quality Monitor + Model Introspection aktiv
  ☐ ≥ 3 Modelle funktionieren
```

---

# Phase 4: Differentiating Features (nach Release, ongoing)

```
NICHT im initialen Plan — wird nach Phase 3 priorisiert.

Mögliche Features (Backlog, NICHT committet):
  → Bandit mit echten Shader-Varianten (verschiedene Workgroup-Sizes)
  → FP8-KV-Cache mit Quality-Gate
  → MoE-Support (Qwen3-30B-A3B)
  → SentencePiece-Tokenizer (Mistral, Gemma)
  → Gemma-4 Support
  → 14B-Modelle
  → Windows-Support
  → Plugin-System für Modell-Adaptoren
  → README + Dokumentation + Release

Entscheidung NACH Phase 3: Backlog-Dokument, Priorisierung nach Messung.
(Lektion aus ROCmForge: Agile Releases v1.01/v1.02 statt Big-Bang)
```

---

# Zusammenfassung

```
Phase 1: PoC — 1 Kernel
  Schritt 1.0:  Shader-Analyse                   Tag 1        Report
  Schritt 1.1:  Shader-Extraktion + SPIR-V       Tag 2        ~3 Checks
  Schritt 1.2:  Vulkan Pipeline Setup             Tag 2–3      ~2 Checks
  Schritt 1.3:  Buffer + Testdaten                Tag 3        ~3 Checks
  Schritt 1.4:  Dispatch + Smoke-Test             Tag 3–4      ~6 Checks
  Schritt 1.5:  Skalierungstest                   Tag 4–5      ~3 Checks
                                                              ─────────
                                                   Subtotal   ~17 Checks

Phase 2: Minimaler Inference-Loop
  Schritt 2.1:  Shader-Inventar                   Tag 6–7      Report
  Schritt 2.2:  Pipeline-Registry                 Tag 7–8      ~4 Tests
  Schritt 2.3:  VRAM-Arena                        Tag 8–9      ~5 Tests
  Schritt 2.4:  GGUF-Loader                       Tag 9–11     ~8 Tests
  Schritt 2.5:  Elementweise Shader               Tag 11–12    ~6 Tests
  Schritt 2.6:  Single-Layer Forward              Tag 12–14    ~5 Tests
  Schritt 2.7:  Full Forward Pass                 Tag 14–16    ~5 Tests
  Schritt 2.8:  Tokenizer + Sampling              Tag 16–17    ~5 Tests
  Schritt 2.9:  Decode-Loop                       Tag 17–19    ~5 Tests
  Schritt 2.10: Phase-2-Validierung               Tag 19–21    ~4 Tests
                                                              ─────────
                                                   Subtotal   ~47 Tests

Phase 3: Feature-Parity
  Schritt 3.1:  Prefill (GEMM + Flash-Attn)       Tag 22–25    ~5 Tests
  Schritt 3.2:  Multi-Turn + Streaming             Tag 25–27    ~5 Tests
  Schritt 3.3:  Model Introspection                Tag 27–28    ~4 Tests
  Schritt 3.4:  Quality Monitor                    Tag 28–29    ~4 Tests
  Schritt 3.5:  Bandit Runtime                     Tag 29–30    ~4 Tests
  Schritt 3.6:  Arch-Aware Sampling                Tag 30–31    ~4 Tests
  Schritt 3.7:  Multi-Modell                       Tag 31–33    ~3 Tests
  Schritt 3.8:  Performance-Optimierung            Tag 33–35    Messung
  Schritt 3.9:  Finale Validierung                 Tag 35–37    ~7 Tests
                                                              ─────────
                                                   Subtotal   ~36 Tests

═══════════════════════════════════════════════════════════════════════
GESAMT: 25 Schritte, ~37 Tage (~6–8 Wochen), ~100 Tests
        Jeder Schritt: Report → STOP → User bestätigt → weiter.
        KEIN Schritt ohne User-OK. KEINE Ausnahme.
═══════════════════════════════════════════════════════════════════════
```

---

## Risiken + Mitigationen (aus Vision)

```
RISIKO 1: Shader-Generator statt standalone GLSL
  → Schritt 1.0 klärt das SOFORT (Tag 1)
  → Falls Generator: Script ausführen, generierten GLSL nehmen

RISIKO 2: Vulkan-Boilerplate aufwändig
  → ash ist 1:1 Vulkan, gpu-alloc löst Allocation
  → Ähnlicher Aufwand wie HIP-FFI in ROCmForge

RISIKO 3: Performance-Verlust durch eigenen Dispatch
  → Schritt 1.5 misst das SOFORT (Tag 4–5)
  → Falls BW < 30%: Dispatch-Strategie überdenken

RISIKO 4: Shader-Kompilierung dauert Minuten
  → Pipeline-Cache ab Schritt 2.2
  → Pre-kompilierte SPIR-V im Build

RISIKO 5: Feature-Drift bei llama.cpp Shader-Updates
  → Eigene Kopie in vk_shaders/ (pinned)
  → Periodischer Sync (1×/Quartal)
```
