# VulkanForge — Projekt-Vision

**Erstellt:** 25.04.2026
**Basis:** ROCmForge v1.0 Erfahrungen + Triple-Benchmark Ergebnisse
**Strategie:** Rust-Infrastruktur von ROCmForge + llama.cpp Vulkan GLSL Shader + eigener Vulkan-Dispatch

---

## Warum VulkanForge

### Die Erkenntnis aus ROCmForge v1.0

```
ROCmForge v1.0 hat BEWIESEN:
  ✅ Rust-Architektur funktioniert (Bandit, Monitor, Introspection)
  ✅ Multi-Turn + Streaming + Think-Filter ist wertvoller Differentiator
  ✅ Model-Introspection (SNR-Risk) existiert NIRGENDWO sonst
  ✅ Quality-Monitor (z-score Drift) existiert NIRGENDWO sonst
  ✅ Self-Tuning Runtime (UCB1 Bandit) existiert NIRGENDWO sonst

ROCmForge v1.0 hat AUCH bewiesen:
  ❌ Eigene GPU-Kernel sind SCHLECHTER als llama.cpp (33% Trefferquote)
  ❌ JEDER große Perf-Gewinn kam aus llama.cpp Kernel-Ports
  ❌ HIP/ROCm auf RDNA4 ist 30% langsamer als Vulkan
  ❌ HIP/ROCm crasht ab pp>4096 auf Consumer-GPUs
  ❌ 6 von 9 Optimierungsversuche waren NEGATIV
```

### Triple-Benchmark (25.04.2026)

```
Hardware: RX 9070 XT (gfx1201, RDNA4)
Modell:   Qwen3-8B Q4_K_M

                    Decode tok/s    Prefill tok/s (pp=512)    Effizienz
llama.cpp Vulkan:   114.2           4314                      0.34 tok/s/W
ROCmForge (HIP):     96.2            751                      ~0.27 tok/s/W
llama.cpp ROCm:      87.5           3684                      0.25 tok/s/W

Vulkan ist:
  → 30% schneller im Decode als ROCm
  → 36% effizienter (tok/s/W)
  → Stabil bis 16k+ Tokens (ROCm crasht ab 4k)
  → 80% BW-Auslastung (vs 62% ROCm, 68% ROCmForge)
```

### Die strategische Schlussfolgerung

```
STATT:  Eigene Kernel schreiben (schlecht bei uns, 33% Trefferquote)
        und am Ende doch llama.cpp portieren

BESSER: llama.cpp's BEWIESENE Vulkan-Kernel DIREKT nutzen
        + Unsere Rust-Infrastruktur OBENDRAUF
        = BESTES aus beiden Welten
```

---

## Architektur

```
┌──────────────────────────────────────────────────────────────┐
│  VulkanForge Rust Layer (UNSER Code, UNSER Wert)             │
│                                                              │
│  Features die NUR VulkanForge hat (llama.cpp hat sie NICHT): │
│                                                              │
│  ✅ Model Introspection                                      │
│     → Automatische Modell-Erkennung aus GGUF Metadata        │
│     → SNR-Risk-Score für Quantisierungs-Qualität             │
│     → Critical-Token-Erkennung (Embedding-Probleme)          │
│     → PrecisionHint pro Layer                                │
│                                                              │
│  ✅ Self-Tuning Runtime                                      │
│     → UCB1 Bandit wählt optimale Kernel-Variante pro Shape   │
│     → Lernt pro Modell + GPU-Kombination                     │
│     → HIP-Graph/Pipeline-Caching nach Konvergenz             │
│     → Bandit-State-Persistenz über Sessions                  │
│                                                              │
│  ✅ Quality Monitor                                          │
│     → z-score Drift-Detection pro Token                      │
│     → Halluzinations-Warnung bei statistischer Abweichung    │
│     → Automatische Precision-Eskalation (FP8→FP16→FP32)     │
│     → RepetitionDetector (Loop-Erkennung)                    │
│                                                              │
│  ✅ Arch-Aware Inference                                     │
│     → SNR-basiertes Sampling (repeat_penalty auto)           │
│     → FP8-KV-Cache mit Quality-Gate                          │
│     → Chat-Template Disambiguation (7+ Varianten)            │
│     → Multi-Turn KV-Cache-Persistenz                         │
│     → Streaming + <think>-Tag Filterung                      │
│                                                              │
│  ✅ User-Experience                                          │
│     → Interactive Chat-REPL (/reset, /quit)                  │
│     → Token-für-Token Streaming                              │
│     → Modell-Qualitäts-Report beim Laden                     │
│     → Automatische Warnung bei problematischen Modellen      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Vulkan-Dispatch (Rust, via ash crate)                       │
│                                                              │
│  → VkDevice, VkQueue, VkCommandBuffer Management            │
│  → VkBuffer für Weights, Activations, KV-Cache              │
│  → VkPipeline für jeden Compute-Shader                       │
│  → VkDescriptorSet für Kernel-Parameter                      │
│  → Memory-Arena (wie ROCmForge, aber VkDeviceMemory)         │
│  → Pipeline-Cache (Shader-Kompilierung 1× dann cached)      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  GLSL Compute Shaders (aus llama.cpp ggml-vulkan/)           │
│                                                              │
│  → BEWIESENE Performance: 114 tok/s, 80% BW                 │
│  → Community-optimiert über Jahre                            │
│  → MIT-lizenziert (frei nutzbar)                             │
│  → Alle Quant-Formate: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ    │
│  → MoE, Sliding Window, Flash-Attention                      │
│  → KHR_coopmat für Matrix-Multiply auf RDNA4                │
│  → Plattformübergreifend (Linux, Windows, Android)           │
│                                                              │
│  Kernel-Quellen:                                             │
│  ~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/        │
│  → mul_mat_vec_*.comp (Decode GEMV)                          │
│  → mul_mat_*.comp (Prefill GEMM)                             │
│  → flash_attn_*.comp (Flash-Attention)                       │
│  → quantize_*.comp (Activation-Quantize)                     │
│  → dequant_*.comp (Weight-Dequantisierung)                   │
│  → Alle als SPIR-V pre-kompilierbar                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Was VulkanForge SOFORT gewinnt (ohne eigene Kernel-Arbeit)

```
Von llama.cpp Vulkan-Kernel GRATIS:

Performance:
  → 114 tok/s Decode (vs ROCmForge 96.2)           = +19%
  → 4314 tok/s Prefill pp=512 (vs ROCmForge 751)   = +474%
  → Stabil bis 16k+ Tokens (vs 8192 Cap)
  → 80% BW-Auslastung (vs 68%)
  → 36% bessere Effizienz (tok/s/W)

Modell-Support:
  → ALLE Quant-Formate (Q4_0, Q5_0, Q5_K, IQ, etc.)
  → MoE-Support (Qwen3-30B-A3B, DeepSeek)
  → Sliding-Window Attention (Mistral)
  → Flash-Attention (Prefill + Decode)
  → SentencePiece Tokenizer (über llama.cpp's Tokenizer oder eigen)
  → Gemma-4 (wenn llama.cpp Support hat)
  → 14B+ Modelle

Hardware:
  → Plattformübergreifend: Linux, Windows, Android
  → Jede Vulkan-GPU: AMD, NVIDIA, Intel, Apple (MoltenVK)
  → Nicht mehr auf ROCm/RDNA4 beschränkt!

ROCmForge Backlog das sich AUFLÖST:
  PERF-01 Flash-Attention Prefill    → GRATIS
  PERF-02 Flash-Attention Decode     → GRATIS (hebt 8k-Cap auf!)
  PERF-09 Vulkan-Backend             → IST das Backend
  MODEL-01 SPM-Tokenizer             → GRATIS (oder eigene Impl)
  MODEL-02 MoE-Support               → GRATIS
  MODEL-03 Gemma-4                   → GRATIS (wenn llama.cpp)
  MODEL-04 Sliding-Window            → GRATIS
  MODEL-05 14B Modelle               → GRATIS
  MODEL-06 Qwen2.5 (Q4_0/Q5_0)      → GRATIS
  MODEL-08 Arena-Sizer               → Neues Design
  BUG-02 8192-Token-Cap              → GRATIS
  BUG-04 Mistral Tokenizer           → GRATIS
```

---

## Technischer Ansatz — Option C (GLSL Shader + ash crate)

### Warum Option C

```
Option A (llama.cpp als C-Library FFI):
  ❌ ggml-API ist instabil (ändert sich häufig)
  ❌ Schwer zu debuggen (C↔Rust Boundary)
  ❌ Kein Low-Level-Zugriff für Bandit + Monitor

Option B (llama.cpp als Subprocess):
  ❌ Latenz durch IPC
  ❌ Kein Zugriff auf Kernel-Zeiten für Bandit
  ❌ Kein Zugriff auf Logits für Monitor

Option C (GLSL Shader extrahieren + eigener Vulkan-Dispatch): ✅
  ✅ Volle Kontrolle über Scheduling + Memory
  ✅ Bandit kann DIREKT Kernel-Varianten switchen
  ✅ Monitor hat DIREKT Zugriff auf Logits
  ✅ ash crate ist dünn + stabil (1:1 Vulkan-API)
  ✅ GLSL Shader sind MIT-lizenziert
  ✅ Gleicher FFI-Pattern wie ROCmForge (HIP → Vulkan)
  ✅ ~80% des Rust-Codes wiederverwendbar
```

### Technischer Stack

```
Rust Crates:
  ash        = Vulkan-API Bindings (dünn, 1:1 mapping)
  gpu-alloc  = VkDeviceMemory Allocator
  shaderc    = GLSL → SPIR-V Compiler (build-time)

GLSL Shader (aus llama.cpp extrahiert):
  vk_kernels/
    mul_mat_vec_q4_k.comp     ← Decode GEMV Q4_K
    mul_mat_vec_q6_k.comp     ← Decode GEMV Q6_K
    mul_mat_q4_k.comp         ← Prefill GEMM Q4_K (Integer-WMMA!)
    flash_attn_f16.comp       ← Flash-Attention
    quantize_q8_1.comp        ← Activation-Quantize
    dequant_q4_k.comp         ← Weight-Dequant
    ...
  → Pre-kompiliert zu SPIR-V beim Build (shaderc)
  → Pipeline-Cache für Runtime-Shader-Caching

Vulkan-Dispatch (Rust):
  src/backend/vulkan/
    device.rs       ← VkDevice + Queue Setup
    memory.rs       ← Arena-Allokation (VkDeviceMemory)
    pipeline.rs     ← Shader-Pipeline + Descriptor-Sets
    dispatch.rs     ← Kernel-Launch (vkCmdDispatch)
    sync.rs         ← Fences + Semaphores
```

### Migration von ROCmForge

```
WIEDERVERWENDBAR (80% des Rust-Codes):
  src_v1/core/inference.rs      → Multi-Turn, Streaming
  src_v1/core/streaming.rs      → Think-Filter
  src_v1/core/tokenizer.rs      → Chat-Templates, BPE
  src_v1/introspection/         → SNR-Risk, Model-Detect
  src_v1/runtime/bandit.rs      → UCB1 Self-Tuning
  src_v1/monitor/               → Quality Monitor
  src_v1/cli/                   → CLI, Chat-REPL

ANPASSUNG NÖTIG (15%):
  src_v1/graph/                 → Computation Graph (Backend-abstrahieren)
  src_v1/core/model_loader.rs   → GGUF-Parser (bleibt, Tensor-Routing ändert sich)

NEU SCHREIBEN (5%):
  src/backend/vulkan/           → Vulkan-Device, Memory, Pipeline, Dispatch
  → Ersetzt: src_v1/backend/gpu/ (HIP FFI)
  → GLEICHER Pattern, andere API
```

---

## Roadmap

### Phase 0: ROCmForge v1.0 Release (3-4 Tage)

```
→ README.md + CHANGELOG.md
→ v1.0.0 Tag (lokal)
→ ROCmForge ist FERTIG und DOKUMENTIERT
→ Dient als Referenz-Implementation + Benchmark-Baseline
```

### Phase 1: VulkanForge PoC — 1 Kernel (1-2 Wochen)

```
Ziel: BEWEIS dass der Ansatz funktioniert

1. ash + gpu-alloc Setup (Vulkan-Device, Queue, Memory)
2. EINEN Shader extrahieren: mul_mat_vec_q4_k.comp (Decode GEMV)
3. SPIR-V kompilieren + Pipeline erstellen
4. Dispatch: Q4_K GEMV auf einem Test-Vektor
5. Parity vs llama.cpp Vulkan Output

GATE: Kernel produziert korrekte Ergebnisse → Ansatz validiert
MESSUNG: tok/s des einzelnen Kernels vs llama.cpp
```

### Phase 2: Minimaler Inference-Loop (2-3 Wochen)

```
Ziel: Ein Modell laden + Tokens generieren

1. GGUF-Parser (aus ROCmForge wiederverwendet)
2. Alle GEMV-Shader (Q4_K, Q6_K) + Attention + Norm + RoPE
3. Embedding + LM-Head
4. Decode-Loop: Token für Token generieren
5. Greedy-Sampling

GATE: "Explain what a mutex is" → kohärenter Text
MESSUNG: Decode tok/s → Erwartung: ~110+ tok/s
```

### Phase 3: Feature-Parity mit ROCmForge (2-3 Wochen)

```
1. Multi-Turn + Streaming + Think-Filter (Rust-Code portieren)
2. Prefill (GEMM-Shader + Flash-Attention)
3. Bandit Runtime (Kernel-Varianten-Auswahl)
4. Quality Monitor
5. Model Introspection
6. FP8-KV-Cache
7. Chat-Template Disambiguation

GATE: 15-Prompt Suite, 15/15, ~114 tok/s
```

### Phase 4: Differentiating Features (ongoing)

```
Features die NUR VulkanForge hat:

1. Automatische Modell-Qualitäts-Erkennung
   → "Dieses Modell hat SNR-Probleme bei Q4_K, empfehle Q5_K"
   → Kein anderes Tool macht das

2. Runtime-Halluzinations-Warnung
   → z-score Drift → "⚠ Output-Qualität degradiert"
   → Kein anderes Tool macht das

3. Self-Tuning (Bandit lernt pro GPU)
   → Erste Inferenz: exploriert Kernel-Varianten
   → Ab Inferenz 10+: optimale Variante automatisch
   → Kein anderes Tool macht das

4. Quality-gated FP8
   → FP8-KV-Cache AN, aber Monitor überwacht
   → Falls Qualität sinkt → automatisch FP16 Fallback
   → Kein anderes Tool macht das

5. Plugin-System für Modell-Adaptoren
   → Neue Modell-Familie: nur Tensor-Routing + Template definieren
   → Kernel bleiben gleich (llama.cpp GLSL)
   → Schneller Support für neue Modelle als llama.cpp CLI
```

---

## Differenzierung vs llama.cpp

```
WARUM SOLLTE JEMAND VULKANFORGE STATT LLAMA.CPP NUTZEN?

llama.cpp ist ein TOOL — VulkanForge ist ein INTELLIGENTES Tool.

llama.cpp:
  → Startet, lädt, inferiert, fertig
  → Keine Qualitäts-Überwachung
  → Keine automatische Modell-Erkennung
  → Keine Warnung bei schlechten Quant-Formaten
  → Kein Self-Tuning
  → Kein Halluzinations-Detektor

VulkanForge:
  → "Dieses Modell hat SNR 0.023 — Q4_K ist problematisch"
  → "⚠ Output-Drift detektiert (z-score 5.8) — mögliche Halluzination"
  → "Kernel-Variante MMVQ-Fused ist 21% schneller auf dieser GPU"
  → "FP8-KV-Cache aktiv, Qualität stabil (0 Monitor-Events)"
  → "Chat-Template als Llama-3 erkannt (vocab=128256, bos=128000)"

VulkanForge = llama.cpp Performance + Intelligenz-Layer obendrauf
```

---

## Lizenz-Betrachtung

```
llama.cpp:        MIT License → GLSL Shader frei nutzbar
ash crate:        MIT/Apache-2.0
gpu-alloc:        MIT/Apache-2.0
shaderc:          Apache-2.0
ROCmForge Rust:   [eigene Lizenz]

→ Keine Lizenz-Konflikte bei Option C
→ GLSL Shader dürfen kopiert und modifiziert werden
→ Attribution in README/NOTICE erforderlich
```

---

## Risiken

```
RISIKO 1: llama.cpp GLSL-Shader sind NICHT stabil
  → Shader-Interfaces können sich ändern
  → Mitigation: Pinned Version + eigene Kopie (nicht live-link)
  → Wir extrahieren Shader EINMAL und pflegen sie selbst

RISIKO 2: Vulkan-Boilerplate in Rust ist aufwändig
  → Device-Setup, Memory-Management, Pipeline-Caching
  → Mitigation: ash ist dünn, gpu-alloc löst Allocation
  → ROCmForge hatte ähnlichen Aufwand (HIP-FFI, Arena)

RISIKO 3: Performance-Verlust durch eigenen Dispatch
  → llama.cpp hat optimierten ggml-Dispatch
  → Unser Dispatch könnte langsamer sein
  → Mitigation: Vulkan Pipeline-Cache, Command-Buffer-Reuse
  → PoC (Phase 1) misst das SOFORT

RISIKO 4: Shader-Kompilierung dauert Minuten beim ersten Start
  → Triple-Benchmark hat das gezeigt (>14 Min first-run)
  → Mitigation: Pre-kompilierte SPIR-V + Pipeline-Cache auf Disk
  → Second-run ist sofort

RISIKO 5: Feature-Drift zwischen llama.cpp Shader-Updates
  → Neue Quant-Formate, neue Kernel-Varianten
  → Mitigation: Periodischer Shader-Sync (1×/Quartal)
  → Wir müssen nicht JEDEN Shader haben — Q4_K + Q6_K reicht für Start
```

---

## Erfolgskriterien

```
VulkanForge v1.0 PoC (Phase 1 Gate):
  ☐ 1 GLSL Shader läuft korrekt via ash in Rust
  ☐ Parity mit llama.cpp Vulkan Output
  ☐ Kernel-Zeit innerhalb 10% von llama.cpp

VulkanForge v1.0 MVP (Phase 2 Gate):
  ☐ Qwen3-8B Q4_K_M: Decode ≥ 100 tok/s
  ☐ Kohärenter Text-Output
  ☐ GGUF-Laden funktioniert

VulkanForge v1.0 Release (Phase 3 Gate):
  ☐ Decode ≥ 110 tok/s (vs llama.cpp Vulkan 114)
  ☐ Prefill ≥ 3000 tok/s
  ☐ 15/15 Prompt-Kohärenz
  ☐ Multi-Turn + Streaming + Think-Filter
  ☐ Quality Monitor + Model Introspection aktiv
  ☐ README + CLI-Referenz
```
