# VulkanForge — Test- und Profiling-Strategie

**Basis:** ROCmForge v1.0 Erfahrungen
**Eingebettet in:** vulkanforge_projektplan.md

---

## Lehren aus ROCmForge

```
Bug                              Wie entdeckt              Kosten         Lektion
─────────────────────────────────────────────────────────────────────────────────────
Q4_K Nibble-Layout (VF Phase 1)  Random-Daten in 1.5       1h Fix         Uniform-Smoke reicht NICHT
Scalar-Fallback (RF v0.x)        Zufällig nach Wochen      Tage Debug     Regression-Tests ab Tag 1
Q6_K Dispatch-Bug (RF v0.x)      rocprof: 15× zu langsam   2 Tage         Profiling NACH jedem Schritt
Bandit "33.2 tok/s" (RF Phase 2) Fehlende calibrate()      1 Tag          E2E-Tests brauchen VOLLEN Stack
RoPE Identity-Bug (RF Phase 1)   pos=0 ist Identität       Spät erkannt   Teste mit pos>0
```

---

## Drei Kategorien, drei Zeitpunkte

### 1. Correctness-Tests (ab Schritt 2.5)

**Problem:** Ab Phase 2 haben wir 10+ Shader. Für jeden einzeln eine
CPU-Referenz schreiben ist aufwändig und fehleranfällig — die CPU-Referenz
selbst kann Bugs haben (wie der Nibble-Bug in Phase 1 gezeigt hat).

**Lösung: llama.cpp als Golden Reference.**

```
Schritt 2.4 (GGUF-Loader):
  → Gleiche GGUF-Datei in llama.cpp UND VulkanForge laden
  → Basis für alle späteren Vergleiche

Schritt 2.5 (Elementwise):
  → CPU-Referenz PRO Shader (RMSNorm, RoPE, SiLU — trivial zu implementieren)
  → RoPE: Teste mit pos=0 UND pos=100 UND pos=4096
    (pos=0 ist Identitätsrotation → findet keine RoPE-Bugs!)
  → Toleranzen: RMSNorm < 1e-5, SiLU < 1e-6, Add = exakt

Schritt 2.6 (Single-Layer): ← KRITISCH
  → llama.cpp mit Debug-Flag: Hidden-States nach Layer 0 dumpen
    Methode: llama.cpp GGML_DEBUG=1 oder Custom-Patch (einmalig)
  → VulkanForge: Hidden-States nach Layer 0 readback
  → Vergleich: max_abs_err < 1e-3 (Q4_K Akkumulation)
  → Falls > 1e-3: SOFORT STOP — Bug in Barrier, Push-Constants, oder Buffer-Layout

Schritt 2.7 (Full Forward):
  → Logits-Vergleich: VulkanForge vs llama.cpp für gleichen Prompt
  → Top-5 Token-IDs müssen identisch sein
  → KL-Divergenz der Logits-Distribution < 1e-4
  → DAS ist der ultimative Correctness-Test
```

**Implementierung: `cargo test` Test-Suite**

```rust
// tests/correctness.rs — ab Schritt 2.5 einführen

#[test] fn test_rmsnorm_vs_cpu_reference()         // pos-unabhängig
#[test] fn test_rope_pos0_identity()               // Sanity
#[test] fn test_rope_pos100_nontrivial()           // PFLICHT — findet echte RoPE-Bugs
#[test] fn test_rope_pos4096_large()               // Kontext-Skala
#[test] fn test_silu_vs_cpu()                      // trivial
#[test] fn test_gemv_q4k_random_data()             // aus Phase 1, BEHALTEN
#[test] fn test_gemv_q4k_distinct_subblocks()      // aus Phase 1, BEHALTEN
#[test] fn test_gemv_q6k_random_data()             // neu ab 2.1

// Ab Schritt 2.6:
#[test] fn test_single_layer_vs_llama_cpp()        // Golden Reference

// Ab Schritt 2.7:
#[test] fn test_logits_top5_vs_llama_cpp()          // Ultimativer Test
#[test] fn test_logits_kl_divergence()              // Statistischer Test
```

---

### 2. Regression-Tests (ab Schritt 2.1)

**Problem:** Jeder neue Shader oder jede Code-Änderung kann bestehende
Funktionalität brechen. In ROCmForge blieb der Scalar-Fallback-Bug
wochenlang unentdeckt weil es keine automatische Regression gab.

**Lösung: `cargo test` als Gate vor JEDEM Commit.**

```
Ab Schritt 2.1:
  → ALLE Phase-1-Tests MÜSSEN weiter grün bleiben
  → Jeder neue Shader bekommt mindestens 1 Correctness-Test
  → `cargo test` vor jedem Commit ist PFLICHT

Ab Schritt 2.6:
  → Single-Layer-Test wird Regression-Test
  → Jede Änderung am Dispatch-Code → `cargo test` MUSS grün

Ab Schritt 2.9:
  → End-to-End Decode von 1 Prompt wird Regression-Test
  → "Explain what a mutex is" → Output muss kohärent sein
  → Das ist der "Scalar-Fallback-Bug-Killer" — wenn der Decode
    plötzlich Müll produziert, fängt es dieser Test

Ab Schritt 2.10:
  → Performance-Regression-Test:
    "Decode tok/s darf nicht unter 90% der Baseline fallen"
  → Verhindert den "33.2 tok/s"-Bug (fehlende Bandit-Kalibrierung)
```

**Implementierung: Regression-Test-Datei**

```rust
// tests/regression.rs — wächst mit jedem Schritt

// Phase 1 (BEHALTEN):
#[test] fn test_phase1_smoke_256_512()             // Bit-exakt
#[test] fn test_phase1_stress_bw_above_30pct()     // BW-Regression
#[test] fn test_q4k_nibble_pair_layout()           // Nibble-Bug-Killer

// Ab Phase 2:
#[test] fn test_all_shaders_compile_to_spirv()     // Build-Regression
#[test] fn test_pipeline_registry_all_registered()  // Kein Shader vergessen
#[test] fn test_no_validation_errors()              // Vulkan-Validation-Regression

// Ab Schritt 2.9:
#[test] fn test_decode_produces_coherent_text()     // E2E-Regression

// Ab Schritt 2.10:
#[test] fn test_decode_tok_s_above_baseline()       // Performance-Regression
```

**Regel für den Agent:**
```
Vor jedem Commit: cargo test --release
Falls ein Test fehlschlägt: STOP — nicht committen, Bug zuerst fixen.
Neue Funktionalität = neuer Test. Kein Test = kein Commit.
```

---

### 3. Performance-Profiling (ab Schritt 2.6)

**Problem:** "Profiling invertiert IMMER die Schätzungen" (ROCmForge Lektion #1).
Ohne Messung weiß man nicht wo die Zeit wirklich hingeht.

**Vulkan-Profiling-Tools (Äquivalent zu rocprof):**

```
vkCmdWriteTimestamp          ← Bereits in Phase 1 implementiert
  → Pro Shader-Dispatch: Start + End Timestamp
  → Resolution: 10 ns auf RX 9070 XT (timestampPeriod)
  → KEIN Tool-Overhead (Hardware-Counter)

VkQueryPool (PIPELINE_STATISTICS) ← Ab Phase 2
  → Compute Invocations pro Dispatch
  → Nützlich um zu prüfen ob die richtige Anzahl Workgroups läuft

RADV_DEBUG=info              ← Bei Bedarf
  → Memory-Allokation Details
  → Pipeline-Cache Hit/Miss

RGP (AMD GPU Profiler)       ← Phase 3 bei Bedarf
  → Detaillierte Wavefront-Analyse
  → Occupancy, Cache-Hit-Rates
  → Nur bei spezifischen Performance-Problemen
```

**Wann welches Profiling:**

```
Schritt 2.6 (Single Layer): ← PROFILING EINFÜHREN
  → Per-Shader Timestamps für 1 Layer
  → Breakdown: Attention vs GEMV vs Norm vs RoPE vs Residual
  → Das zeigt SOFORT ob ein Shader unerwartet langsam ist
  → Erwartung: GEMV dominiert (~70%), Attention ~20%, Rest ~10%
  → Falls Norm plötzlich 30%: Bug in Barrier oder Workgroup-Size

Schritt 2.7 (Full Forward):
  → Per-Layer Timestamps (36 Layer)
  → Erwartung: alle Layer ~gleich schnell (±5%)
  → Falls Layer 0 2× langsamer: Embedding-Bug oder Cold-Cache
  → Forward-Pass Gesamtzeit → erste tok/s Schätzung

Schritt 2.10 (Validierung):
  → Formale Baseline etablieren:
    - Decode tok/s (Median über 5 Prompts)
    - Prefill tok/s (pp=512)
    - Per-Shader-Breakdown (Tabelle)
    - Vergleich vs llama.cpp Vulkan (114 tok/s)
  → Diese Baseline wird in results/ committed
  → Alle zukünftigen Änderungen messen GEGEN diese Baseline

Phase 3 Schritt 3.8 (Optimierung):
  → Detailliertes Profiling mit Shader-Kategorien
  → Dispatch-Overhead vs Kernel-Zeit vs Readback
  → Barrier-Overhead quantifizieren
  → Pipeline-Cache Startup-Zeit
```

**Profiling-Infrastruktur (einmal bauen, überall nutzen):**

```rust
// src/backend/vulkan/profiler.rs — ab Schritt 2.6 einführen

pub struct ShaderProfiler {
    query_pool: vk::QueryPool,
    timestamp_period_ns: f64,
    entries: Vec<ProfileEntry>,
}

pub struct ProfileEntry {
    shader_name: &'static str,
    start_tick: u64,
    end_tick: u64,
}

impl ShaderProfiler {
    /// Insert timestamp pair around a dispatch
    pub fn begin(&mut self, cmd: vk::CommandBuffer, name: &'static str);
    pub fn end(&mut self, cmd: vk::CommandBuffer);

    /// Read all results after submit+fence
    pub fn collect(&mut self, device: &ash::Device) -> Vec<TimingResult>;

    /// Print breakdown table
    pub fn report(&self) -> String;
    // Output:
    //   Shader              Time (µs)    %      BW (GB/s)
    //   ─────────────────────────────────────────────────
    //   gemv_q4k (attn_q)    14.8       12.3%   484
    //   gemv_q4k (attn_k)    14.9       12.4%   483
    //   gemv_q6k (ffn_down)  15.1       12.5%   478
    //   attention             22.3       18.5%   —
    //   rmsnorm               1.2        1.0%   —
    //   rope                  0.8        0.7%   —
    //   ...
    //   TOTAL                120.5      100%
}
```

---

## Einbettung in den Projektplan

```
Phase 2 Schritte — Test/Profiling-Ergänzungen:

Schritt 2.1 (Shader-Inventar):
  + Regression: cargo test darf nicht brechen (Phase-1-Tests grün)

Schritt 2.5 (Elementwise Validation):
  + Correctness: CPU-Referenz pro Shader
  + Correctness: RoPE pos=0 UND pos=100 UND pos=4096
  + Regression: tests/correctness.rs einführen

Schritt 2.6 (Single Layer): ← KRITISCHSTER PUNKT
  + Correctness: Hidden-States vs llama.cpp (Golden Reference)
  + Profiling: ShaderProfiler einführen, Per-Shader-Breakdown
  + Regression: Single-Layer-Test wird permanenter Regression-Test

Schritt 2.7 (Full Forward):
  + Correctness: Logits Top-5 vs llama.cpp
  + Profiling: Per-Layer Timing, Gesamt-Forward-Zeit
  + Regression: Logits-Vergleich wird Regression-Test

Schritt 2.9 (Decode-Loop):
  + Regression: "mutex prompt" → kohärenter Text (E2E-Regression)
  + Profiling: Decode tok/s messen

Schritt 2.10 (Validierung):
  + Profiling: Formale Baseline etablieren + committen
  + Regression: Performance-Regression-Test (tok/s > 90% Baseline)

Phase 3 Schritt 3.8 (Optimierung):
  + Profiling: Detaillierte Analyse, Dispatch-Overhead, Barrier-Overhead
  + Alle Änderungen gegen Schritt-2.10-Baseline messen
```

---

## Golden-Reference-Strategie: llama.cpp Output dumpen

Für den Logits-Vergleich (ab Schritt 2.7) brauchen wir eine
llama.cpp-Referenzdatei. Einmaliger Aufwand:

```fish
# 1. llama.cpp mit Logits-Dump bauen (einmalig)
cd ~/tmp/llama.cpp
# Option A: GGML_DEBUG Flag (zeigt Tensor-Werte)
# Option B: Kleiner Patch in llama_decode() der Logits nach Datei schreibt
# Option C: llama-cli --logits-out (falls verfügbar in aktueller Version)

# 2. Referenz-Logits erzeugen
./llama-cli -m ~/models/Qwen3-8B-Q4_K_M.gguf \
  -p "Explain what a mutex is" \
  --logits-out /tmp/llama_cpp_logits_mutex.bin \
  -n 1  # nur 1 Token generieren reicht für Logits-Vergleich

# 3. Referenz-Datei ins VulkanForge-Repo
cp /tmp/llama_cpp_logits_mutex.bin tests/golden/qwen3_8b_mutex_logits.bin
```

Die Referenz-Datei ist ~600 KB (151936 × f32 für Qwen3 vocab_size).
Wird einmal erzeugt und dann als Golden-Reference in `cargo test` genutzt.

Für Hidden-States (Schritt 2.6) analog, aber nach Layer 0 abfangen.

---

## Zusammenfassung: Wann was einbauen

```
Schritt 2.1  │ Regression-Tests beginnen (Phase-1-Tests als Basis)
             │
Schritt 2.5  │ Correctness-Tests pro Shader (CPU-Referenz)
             │ → tests/correctness.rs einführen
             │
Schritt 2.6  │ ★ GROSSER EINSCHNITT ★
             │ Golden Reference vs llama.cpp (Hidden-States)
             │ ShaderProfiler einführen (Per-Shader-Breakdown)
             │ Regression: Single-Layer-Test permanent
             │
Schritt 2.7  │ Logits-Vergleich vs llama.cpp
             │ Per-Layer-Profiling
             │
Schritt 2.9  │ E2E-Regression (kohärenter Text)
             │
Schritt 2.10 │ Baseline committen
             │ Performance-Regression-Test
             │
Phase 3      │ Gegen Baseline messen
             │ Detailliertes Profiling bei Optimierung
```
