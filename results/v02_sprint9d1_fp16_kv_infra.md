# v0.2 Sprint 9d.1 — FP16 KV-Cache Infrastructure (Behavior-Neutral)

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9c.5 (19 Dispatches/Layer, 14 Barriers, 164/164 Tests)
**Vorarbeit:** Sprint 9d Architektur-Analyse (results/v02_sprint9d_fp16_kv.md)

---

## TL;DR — Infrastruktur fertig, default OFF, kein Hot-Path. 167/167 Tests grün.

```
═══ v0.2 Sprint 9d.1 ═══

Lieferung:
  • KvDtype { F32, F16 } enum mit element_size() / label()
  • kv_dtype Field in KvCache (NICHT in KvCacheConfig — sonst
    25+ Literal-Init-Sites in src/, examples/, tests/ zu ändern)
  • Env-Var-Read in KvCache::new (default F32)
  • Allokation, pos_offset_bytes, row_bytes, layer_offset_bytes
    skalieren via element_size()
  • Neue Methode: layer_size_bytes() — ersetzt 4× hardcoded
    `max_seq * n_kv_heads * head_dim * 4` Berechnungen in den
    Attention-Dispatches
  • Startup-Log mit KV-Cache-Konfiguration
  • Loud WARNING wenn FP16_KV=1: "outputs WILL BE INCORRECT
    until 9d.2/9d.3 ship the conversion shader and FP16-aware
    attention SPVs"
  • 3 neue Unit-Tests in kv_cache.rs (element_size, label,
    layer_size_scales_with_dtype)

Behavior:
  • Default (env-var unset oder ≠"1"):  IDENTISCH zu Sprint 9c.5
  • VULKANFORGE_FP16_KV=1: KV-Buffer halb so groß alloziert,
    aber Shader lesen/schreiben FP32 → Garbage-Output
    (deshalb der WARN-Log; nicht für Inferenz nutzen)

VRAM-Vergleich (Qwen3-8B max_seq=2048):
  FP32: 576 MB (36 layers × 2 × 8 × 128 × 2048 × 4 B)
  FP16: 288 MB (gleiche Form, halbe Element-Size)
  Δ = 288 MB Ersparnis BEI funktionsfähigem 9d.2/9d.3

Tests:
  cargo test --release  →  167/167 ✓
  Sprint 9c.5 baseline: 164.  Δ = +3 (kv_dtype unit tests).
  Alle E2E-Argmax-Parity-Tests bleiben grün:
    - phase3e_prefill_batch_matches_token_by_token_top5  ✓
    - sprint5b_chunked_prefill_parity_qwen3              ✓
    - phase5b2_decode_after_batched_prefill_qwen3        ✓
    - phase_prompt16_alice_context_retention_qwen3       ✓
  → Default-Pfad verhält sich bit-identisch zu vor 9d.1.

Files:
  modified: src/backend/vulkan/kv_cache.rs (+enum, +field,
            +method, +tests, +warn-log)
  modified: src/backend/vulkan/forward.rs (4× layer_size
            replacements: hardcoded `*4` → layer_size_bytes())
  new:      results/v02_sprint9d1_fp16_kv_infra.md (this report)

Commit: HEAD (kein Push).
Bench: nicht ausgeführt — behavior-neutral, kein Performance-
       Effekt zu erwarten oder messen.
```

---

## 1. Design-Entscheidung: dtype im KvCache, nicht in KvCacheConfig

Das Sprint-Brief schlug zwei Optionen vor:
* (a) `kv_dtype` in `KvCacheConfig` aufnehmen (ergibt sauberere API)
* (b) `kv_dtype` direkt im `KvCache` Struct, env-var inside `KvCache::new`

Audit zeigte: `KvCacheConfig` wird an **mindestens 25 Stellen** mit
struct-literal-Syntax konstruiert (1× in `src/main.rs`, 9× in
`examples/*.rs`, 13× in `tests/regression.rs`, plus die Definition).
Option (a) hätte alle 25 Stellen erforderlich angepasst — ein massiver
Diff für ein behavior-neutrales Feature.

→ **Option (b) gewählt.** Existierende Konstruktoren bleiben unverändert.
Die env-var `VULKANFORGE_FP16_KV` wird automatisch beim ersten
`KvCache::new`-Aufruf gelesen.

Trade-off:
* Pro: Null Änderungen an 25 Konstruktor-Sites.
* Pro: Sprint 9d.1 bleibt klein (~80 Zeilen Diff).
* Contra: Tests können nicht explizit FP16 anfordern ohne env-var.
  Akzeptabel — Sprint 9d.1 hat sowieso keine FP16-Pfad-Tests
  (FP16 ist bewusst nicht funktional bis 9d.2).

---

## 2. Buffer-Sizing-Mathematik

### 2.1 Allokationsformel

```rust
let elem_size = kv_dtype.element_size();   // 4 (F32) oder 2 (F16)
let bytes = (config.n_layers as u64)
    * (config.n_kv_heads as u64)
    * (config.max_seq_len as u64)
    * (config.head_dim as u64)
    * elem_size;
```

### 2.2 Offset-Methoden

Vorher (FP32 hardcoded):
```rust
elems * (std::mem::size_of::<f32>() as u64)
```

Nachher (parametriert):
```rust
elems * self.kv_dtype.element_size()
```

Aktualisiert in:
* `pos_offset_bytes(layer, pos)` — Byte-Offset für vkCmdCopyBuffer-dst
* `row_bytes()` — Zeilenstride für vkCmdCopyBuffer-size
* `layer_offset_bytes(layer)` — Layer-Offset für Attention-SSBO-Range

### 2.3 Neue Methode: layer_size_bytes()

```rust
pub fn layer_size_bytes(&self) -> u64 {
    (self.config.max_seq_len as u64)
        * (self.config.n_kv_heads as u64)
        * (self.config.head_dim as u64)
        * self.kv_dtype.element_size()
}
```

Diese Methode ersetzt vier identische `let layer_size = max_seq *
n_kv_heads * head_dim * 4` Berechnungen in den Attention-Dispatches
(forward.rs Zeilen 1810, 1965, 2031, 2088). Effekt: bei FP16 wird die
SSBO-Range jetzt halbiert mit-skaliert, statt fix-FP32-Range zu binden.

### 2.4 Was NICHT skaliert

* `q_bytes` / `q_bytes_total` (Q-Buffers) — FP32 bleibt FP32.
  Q-Aktivierungen werden im Shader als FP32 erzeugt (GEMM-Output) und
  nicht im Cache gespeichert. Sprint 9d nimmt das nicht ins Visier.
* `fa_scratch_out_bytes` / `fa_scratch_red_bytes` (Sprint 4C
  Scratch-Buffer für split-K Decode-Reduce) — FP32, kein Cache-Bezug.
* RoPE/RMS-Norm/SwiGLU-Buffers — alles FP32.

→ Nur die KV-Cache-Buffers ändern Größe; alles andere bleibt FP32.

---

## 3. Env-Var Semantik + Warning

```rust
fn kv_dtype_from_env() -> KvDtype {
    match std::env::var("VULKANFORGE_FP16_KV") {
        Ok(s) if s == "1" => KvDtype::F16,
        _ => KvDtype::F32,
    }
}
```

* Default: F32 (entspricht Sprint 9c.5 Verhalten genau).
* Opt-in: `VULKANFORGE_FP16_KV=1`.
* Andere Werte (`true`, `yes`, `0`, leer): F32 (defensiv).

Bei FP16-Wahl gibt `KvCache::new` einen lauten Warn-Log aus:
```
VulkanForge: WARNING — VULKANFORGE_FP16_KV=1 allocates the KV cache
as FP16 (Sprint 9d.1 infrastructure), but the attention shaders and
KV-write copies are still FP32. Outputs WILL BE INCORRECT until
Sprint 9d.2/9d.3 ship the conversion shader and FP16-aware attention
SPVs.
```

Plus immer: ein Info-Log mit der Konfiguration + Total-MB:
```
VulkanForge: KV cache FP32 (4B/elem) × 2 buffers = 576 MB
(36 layers × 8 kv_heads × 2048 max_seq × 128 head_dim)
```

---

## 4. Korrektheit — wie verifiziert

### 4.1 Behavior-Neutralität

Der Default-Pfad (`VULKANFORGE_FP16_KV` unset) erzeugt:
* KvCache.kv_dtype = F32
* element_size() = 4
* Alle Offset-Berechnungen ergeben **bit-identisch dieselben Werte**
  wie vor 9d.1 (vorher: hardcoded `* sizeof::<f32>() = * 4`,
  jetzt: `* element_size() = * 4` für F32).

→ Default-Verhalten ist mathematisch beweisbar identisch.

### 4.2 Regression-Tests

```
test result: ok. 27 passed (lib unit, including +3 new kv_dtype tests)
test result: ok.  9 passed (dequant_q4k)
test result: ok. 18 passed (gguf)
test result: ok. 70 passed (correctness)
test result: ok.  8 passed (q4k_quant)
test result: ok.  8 passed (flash_attn_tiled_ref)
test result: ok. 27 passed (regression — INCL. all E2E argmax tests)
                ────
                167 / 167 ✓
```

E2E-Argmax-Parity-Tests (die kritischsten "kein Bit darf wackeln"-Tests):
* `phase3e_prefill_batch_matches_token_by_token_top5` ✓
* `sprint5b_chunked_prefill_parity_qwen3` ✓
* `phase5b2_decode_after_batched_prefill_qwen3` ✓
* `phase_prompt16_alice_context_retention_qwen3` ✓

Diese vier produzieren bit-identische Logits zu Sprint 9c.5. Behavior-
neutralität bestätigt.

### 4.3 Neue Unit-Tests (3, in `src/backend/vulkan/kv_cache.rs`)

```rust
#[test] fn kv_dtype_element_size() {
    assert_eq!(KvDtype::F32.element_size(), 4);
    assert_eq!(KvDtype::F16.element_size(), 2);
}

#[test] fn kv_dtype_label() {
    assert_eq!(KvDtype::F32.label(), "FP32");
    assert_eq!(KvDtype::F16.label(), "FP16");
}

#[test] fn layer_size_scales_with_dtype() {
    // Qwen3-8B realistic config: 36 layers × 8 kv_heads × 2048 max_seq × 128 head_dim
    // FP32 total = 576 MB, FP16 total = 288 MB (half).
    ...
}
```

Pure-Rust-Arithmetik, kein Vulkan-Device benötigt.

---

## 5. FP16-Pfad ist NICHT funktional in 9d.1 — bewusst

Wenn ein Nutzer trotz Warning `VULKANFORGE_FP16_KV=1` setzt:
1. KvCache wird als **halb-so-große Buffers** alloziert (288 MB statt 576 MB).
2. Die KV-Write-Sites in `forward.rs` (4× `vkCmdCopyBuffer`) kopieren
   FP32-Daten als raw bytes in den FP16-Buffer → **erste 50% des Buffers
   enthalten FP32-bytes** (interpretiert als doppelt so viele FP16-Werte
   = Garbage).
3. Die Attention-Shader (`flash_attn_*.comp`) deklarieren K/V als
   `float[]` — sie lesen 4-Byte-Werte aus dem half-allocierten Buffer
   → **lesen über die Buffer-Grenze hinaus** (UB / Crash) ODER lesen
   Garbage-Werte für Positionen ≥ max_seq/2 → falsche Logits.

Beides ist erwartetes Verhalten für 9d.1. Sprint 9d.2 löst beide:
* (a) ersetzt die 4× `vkCmdCopyBuffer` durch einen `kv_copy_fp16`
  Compute-Shader der FP32 → FP16 packed schreibt.
* (b) Erzeugt FP16-aware SPV-Varianten der Attention-Shader die
  `unpackHalf2x16(uint)` zum Lesen verwenden.

Bis dahin: **niemals** mit `VULKANFORGE_FP16_KV=1` produktiv inferieren.
Die WARNING im Log ist explizit so formuliert.

---

## 6. Code-Diff (gekürzt)

```rust
// src/backend/vulkan/kv_cache.rs (NEU):
+pub enum KvDtype { F32, F16 }
+impl KvDtype { fn element_size(self) -> u64 {...} fn label(...)}
+fn kv_dtype_from_env() -> KvDtype { ... }
 pub struct KvCache {
     pub k_buffer: GpuBuffer,
     pub v_buffer: GpuBuffer,
     pub config: KvCacheConfig,
     pub current_seq_len: u32,
+    pub kv_dtype: KvDtype,
 }
 impl KvCache {
     pub fn new(...) -> Result<...> {
+        let kv_dtype = kv_dtype_from_env();
+        let elem_size = kv_dtype.element_size();
-        let bytes = ... * (std::mem::size_of::<f32>() as u64);
+        let bytes = ... * elem_size;
+        if kv_dtype == KvDtype::F16 { eprintln!("WARNING ..."); }
+        eprintln!("KV cache {} ...", kv_dtype.label());
         ...
     }
-    pub fn pos_offset_bytes(...) { elems * size_of::<f32>() }
+    pub fn pos_offset_bytes(...) { elems * self.kv_dtype.element_size() }
-    pub fn row_bytes(...) { ... * size_of::<f32>() }
+    pub fn row_bytes(...) { ... * self.kv_dtype.element_size() }
-    pub fn layer_offset_bytes(...) { ... * size_of::<f32>() }
+    pub fn layer_offset_bytes(...) { ... * self.kv_dtype.element_size() }
+    pub fn layer_size_bytes(&self) -> u64 { ... * self.kv_dtype.element_size() }
 }

// src/backend/vulkan/forward.rs (4 identische Replacements):
-    let layer_size = (self.kv_cache.config.max_seq_len as u64)
-        * (cfg.n_kv_heads as u64)
-        * (cfg.head_dim as u64)
-        * 4;
+    let layer_size = self.kv_cache.layer_size_bytes();
```

---

## 7. Files Touched

```
modified: src/backend/vulkan/kv_cache.rs (+72 LOC: enum, env-fn,
          field, method, log, tests)
modified: src/backend/vulkan/forward.rs (4× layer_size hunks,
          -16 lines net via layer_size_bytes() helper)
new:      results/v02_sprint9d1_fp16_kv_infra.md (this report)
```

KEIN neuer Shader-Code. KEIN Pipeline-Registry-Eintrag. KEIN
build.rs-Eintrag. Pure Rust-side infrastructure.

---

## 8. Verbleibende Sprint-9d-Schritte

### Sprint 9d.2 (2-3 Tage) — Prefill Hot-Path

* Neuer Shader `vk_shaders/kv_copy_fp16.comp`: konvertiert FP32 →
  packed-FP16 via `packHalf2x16`. Ersetzt die 2 batched `vkCmdCopyBuffer`
  Sites in `dispatch_layer_batch`.
* Neue SPV-Varianten der Attention-Shader mit `#define FP16_KV`:
  - `flash_attn_tiled_br16_bc32_fp16kv.spv` (Sprint-8a-Default)
  - `flash_attn_batch_fp16kv.spv` (Br=1 Fallback)
* GLSL-Pattern: Buffer als `uint[]` deklarieren, `unpackHalf2x16`
  beim Read. Keine neuen Vulkan-Features nötig (storage_buffer16_bit_access
  ist seit Phase 1 enabled).
* Tests: isolierter Round-Trip + isolierte FA-Parity + run_pp_bench
  mit FP16_KV=1 (prefill-only — decode noch nicht).

### Sprint 9d.3 (1-2 Tage) — Decode Hot-Path

* Decode-Attention-Shader-Varianten: `flash_attn_fp16kv.spv`,
  `flash_attn_split_fp16kv.spv`, `scalar_attn_fp16kv.spv`.
* 2 weitere `vkCmdCopyBuffer` Sites in `dispatch_layer` umgestellt.
* End-to-End-Argmax-Tests mit FP16_KV=1 + chat-session.
* Default-Decision: ON falls argmax ≥ top-5/5 vs FP32 (llama.cpp
  default).

### Sprint 9d.4 (optional) — Bc=64 Tiled-Coverage

* Br=4/8/16 mit Bc=64 FP16-Varianten (für seq_len ≤ 64 Sweet Spots).
* Optimierung: 2-element-pro-thread Inner-Loop.

---

## 9. Bottom Line

Sprint 9d.1 ist ein **disziplinierter Infrastruktur-Sprint**: kleines
Diff (~80 LOC), behavior-neutral, alle Tests grün. Die scharfe
Konfigurationskante (env-var off → identisches Verhalten; env-var on
→ Garbage mit Warning) ist intentional: 9d.1 macht NUR die Allokations-
und Offset-Mathematik, der Hot-Path bleibt für 9d.2 reserviert.

Architektonisch ist das wertvoller als es performance-mäßig aussieht:
nach 9d.1 sind die 4 hardcoded `* 4` Byte-Stride-Berechnungen in
`forward.rs` durch eine zentrale `KvCache::layer_size_bytes()`-
Methode ersetzt. Sprint 9d.2 muss dann nur Shader + KV-Write berühren,
nicht mehr die Attention-Dispatch-Wiring (das ist bereits richtig
parametriert).

VRAM-Vorteil bei späterer Aktivierung (Qwen3-8B):
* max_seq=2048: -288 MB pro Run
* max_seq=8192: -1.13 GB pro Run

Empfehlung — nächster Sprint: **Sprint 9d.2 (Prefill Hot-Path)**.
Das Infrastruktur-Fundament steht; der Shader-Teil ist
~3-tägig konzentrierte Arbeit, am besten ohne Citrix-Hintergrund-
last für saubere Δ-Messung.
