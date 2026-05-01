# RDNA4 Cooperative Matrix — Verifizierte Referenzen

**Erstellt:** 27.04.2026
**Zweck:** Referenz für Phase 6A Agent — coopmat + BF16 auf gfx1201

---

## Kernaussage

RADV mappt VK_KHR_cooperative_matrix auf NATIVE WMMA-Instruktionen.
Keine Software-Emulation. Die AI Accelerators werden direkt angesprochen.

Die Trennung ist:
1. **Datenvorbereitung:** Software (5 VALU-Instruktionen BF16-Convert)
2. **Berechnung:** Hardware (v_wmma_f32_16x16x16_bf16, nativ auf AI Accel)

---

## Verifizierte Links

### 1. RADV Cooperative Matrix Implementation (Phoronix, Feb 2025)

```
https://www.phoronix.com/news/RADV-Lands-RDNA4-Coop-Matrix

Inhalt:
  - RADV merged coopmat Support für GFX12/RDNA4
  - Baut auf WMMA-Infrastruktur von RDNA3 auf
  - Entwickler: Samuel Pitoiset (Valve)
  - Keine Soft-Emulation, direkte Hardware-Nutzung
```

### 2. Mesa 25.1 Release Notes

```
https://docs.mesa3d.org/relnotes/25.1.0.html

Inhalt:
  - GFX12 Feature-Support gelistet
  - RADV nutzt für Matrix-Operationen auf WMMA-Architekturen
    (GFX11/GFX12) direkt die Hardware-Beschleuniger
```

### 3. vLLM RDNA4 FP8/BF16 WMMA Diskussion (Nov 2025)

```
https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900

Inhalt:
  - Entwickler diskutieren RDNA4 AI-Beschleuniger für LLM-Inference
  - RDNA4 hat native WMMA für FP8 UND BF16
  - Datenvorbereitung (Shuffling/Casting) braucht Software-Sequenzen
  - Berechnung ist nativ auf den AI Accelerators
```

### 4. Mesa/ACO Compiler Source

```
Mesa Source: src/amd/compiler/aco_instruction_selection.cpp
  - Lowering-Regeln für f2bf16 auf GFX12
  - ACO generiert Bit-Shift + Maskierungs-Sequenz (5 Ops)
  - Ergebnis wird an WMMA-Einheit übergeben

Verifizierbar mit:
  umr (User Mode Register Debugger)
  amdgpu-objdump (Shader disassembly)
  Suche nach: v_wmma_f32_16x16x16_bf16
```

---

## Hardware-Architektur (RDNA4 / gfx1201)

```
Pro Compute Unit (CU):
  - 2 SIMD32-Einheiten (= 1 Wave64)
  - 2 AI Accelerators (WMMA-Units)
  - Shared LDS

64 CU × 2 AI Accel = 128 AI Accelerators total

WMMA Tile-Size: 16×16×16
  - Input: BF16, FP16, FP8 (E4M3, E5M2)
  - Output: FP32
  - Instruktion: v_wmma_f32_16x16x16_bf16 (1 Takt pro Tile)

Theoretische Peakleistung:
  128 AI Accel × 16×16×16 × 2 (FMA) / Takt
  = ~200 TOPS (BF16)
  vs ~25 TFLOPS (scalar FP32 FMA auf VALU)
  = ~8× Compute-Vorteil
```

---

## Implikation für Phase 6

```
POSITIV:
  - coopmat ist KEIN Emulation → echte Hardware-Beschleunigung
  - v_wmma_f32_16x16x16_bf16 existiert und wird von RADV genutzt
  - ~8× theoretischer Compute-Vorteil
  - vLLM arbeitet bereits daran → Machbarkeit bestätigt

RISIKO (unverändert):
  - BF16-Convert: 5 VALU-Ops pro Element
  - Q4_K Dequant + BF16-Convert = ~10-15 Instruktionen pro Element
    BEVOR WMMA rechnen kann
  - ROCmForge-Erfahrung: FP8-WMMA auf Q4_K = 0.75× (NEGATIV)
    obwohl FP8-Convert nur 1 Op war (nicht 5)

ABER:
  - WMMA ist 8× schneller als scalar → mehr Spielraum für Overhead
  - FP8 hatte: 1 Op Convert + Dequant = negativ
  - BF16 hat: 5 Op Convert + Dequant
  - Die Frage ist: kompensieren 8× WMMA die 5× teurere Konvertierung?
  - Micro-Benchmark in Phase 6A beantwortet das definitiv

ENTSCHEIDEND:
  - Phase 6A muss die Dequant-Pipeline INKLUSIVE BF16-Convert messen
  - Nicht nur WMMA-TFLOPS isoliert (das wäre ~200 TOPS, toll aber irrelevant)
  - Sondern: Q4_K-Block → Dequant → BF16 → WMMA → FP32 End-to-End
```
## Hardware & Treiber-Support (RDNA 4 / GFX12)

Diese Links belegen, dass die Hardware-Features (WMMA) für deine Karte bereits im Linux-Ökosystem gelandet sind:

    Phoronix: RADV Lands RDNA 4 Cooperative Matrix Support

        https://www.phoronix.com/news/RADV-Lands-RDNA4-Coop-Matrix

        Bestätigt, dass der Vulkan-Treiber (RADV) die AI-Beschleuniger der GFX12-Architektur nativ anspricht.  

    Mesa 25.1.0 Release Notes (GFX12 Features)

        https://docs.mesa3d.org/relnotes/25.1.0.html

        Listet die Unterstützung für kooperative Matrizen auf GFX12-Hardware.
        
## Community-Diskussionen zu LLM-Performance auf GFX12

Hier wird aktiv über die Implementierung von BF16/FP8-WMMA für Karten wie die RX 9070 XT diskutiert:

    vLLM Discussion: Native FP8/BF16 WMMA Support for RDNA 4

        https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900

        Thematisiert die Notwendigkeit von Software-Sequenzen für die Datenvorbereitung (Dequant), bevor die WMMA-Hardware die Berechnung übernimmt.

## Shader-Code & Compiler-Internals

Um zu verstehen, warum unser Q6_K Shader aktuell "nur" so schnell wie der von llama.cpp ist (da er identisch ist):

    llama.cpp Vulkan Shader Source (Upstream)

        https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp

        Dies ist der Code, mit dem wir aktuell byte-identisch sind. Du kannst dort in Zeile 24 und 56 die barrier()-Aufrufe sehen, die wir für die gfx1201 eliminieren wollen.

    Mesa/ACO Compiler Source (f2bf16 Lowering)

        Mesa GitLab - aco_instruction_selection.cpp

        Hier lässt sich nachverfolgen, wie der Compiler BF16-Konvertierungen auf GFX12 in 5 VALU-Instruktionen übersetzt.
        
## Analyse-Tools für dein System

Falls du selbst prüfen möchtest, wie die GPU die Shader aktuell ausführt:

    UMR (User Mode Register Debugger): https://gitlab.freedesktop.org/tomstdenis/umr – Unverzichtbar, um die Wellenfronten (Waves) auf der gfx1201 live zu sehen.
