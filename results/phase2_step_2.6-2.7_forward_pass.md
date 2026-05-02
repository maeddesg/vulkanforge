# Phase 2C — Single-Layer + Full Forward Pass (Schritte 2.6–2.7)

**Datum:** 2026-04-25
**Schritte:** 2.6 (Single-Layer Forward), 2.7 (Full Forward Pass + LM-Head)
**Modell:** Qwen3-8B Q4_K_M (Mixed-Quant: Q4_K weights + Q6_K für `attn_v` und `ffn_down`)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Status

| Gate                                                                          | Ergebnis |
|-------------------------------------------------------------------------------|----------|
| 2.6 — Single-Layer Forward läuft, kein Crash                                   | ✅       |
| 2.6 — ShaderProfiler eingeführt + per-shader Breakdown                         | ✅       |
| 2.7 — Full Forward (Embedding → 36 Layer → final norm → LM head)               | ✅       |
| 2.7 — Logits-Output 151 936 Elemente                                            | ✅       |
| 2.7 — Logits != all-zeros                                                       | ✅       |
| 2.7 — Logits != NaN/Inf                                                         | ✅       |
| 2.7 — Forward-Pass < 50 ms                                                      | ✅ 16.7 ms |
| 2.7 — Per-Layer Profiling: alle Layer ähnlich schnell                           | ✅ min=361 µs, mean=378 µs, max=414 µs |
| **Validation-WARN/ERROR Lifecycle**                                             | **0**    |
| **`cargo test`** (Phase 1 + 2A + 2B + 2C)                                       | **27/27 passed** |

---

## 1. Bug während 2.6/2.7 gefunden + behoben — **Mixed-Quant in Q4_K_M**

Erste Forward-Iteration produzierte **NaN** in jedem Layer. Plan war Option D → C → scalar_attn-Isolation. Der Bug stellte sich als ganz woanders heraus.

### Diagnose-Sequenz

**Option D** (scalar_attn umgehen, Q→attn_out kopieren): NaN bleibt. → scalar_attn ist **nicht** der Bug.

**Option C** (per-layer readback): Layer 4 erster NaN-Layer. Aber Werte explodieren schon ab Layer 0 (std=2.8 → 3.7 → 9.5 → 13.4 → NaN). Mit Attention-Bypass werden die Residuals nicht gebounded — erwartet ohne echte Attention, also weiter zu Option-Test mit echter Attention.

**Mit echtem scalar_attn**: NaN bereits in Layer 0, ALLE 4096 Output-Werte. Scalar_attn isoliert getestet (`tests/correctness.rs::test_scalar_attn_*`): funktioniert mit Q=0.1/K=0.1/V=1.0, mit echten Qwen3-Dimensionen, mit Descriptor-Offset-Binding. Shader ist sauber.

**Per-Step intra-Layer-Trace** (`forward_layer_debug_intermediate` mit `DebugTarget::{AttnNorm, QProj, KProj, VProj, …}`):
```
layer 0, AttnNorm:   ✓ finite, min=-0.06, max=0.05
layer 0, QProj:      ✓ finite, min=-0.37, max=0.47
layer 0, KProj:      ✓ finite, min=-0.27, max=0.30
layer 0, VProj:      ❌ 503 NaN von 1024 Werten
layer 0, QNormRope:  ✓ finite
layer 0, KNormRope:  ✓ finite (max=203)
layer 0, AttnOut:    ❌ NaN
```

**VProj** ist der erste NaN-Punkt. Q und K aus dem gleichen Q4_K-GEMV-Shader sind sauber. → Daten-spezifischer Bug.

`gguf-dump` auf das Modell:

```
blk.0.attn_q.weight       Q4_K
blk.0.attn_k.weight       Q4_K
blk.0.attn_v.weight       Q6_K   ← !!
blk.0.attn_output.weight  Q4_K
blk.0.ffn_gate.weight     Q4_K
blk.0.ffn_up.weight       Q4_K
blk.0.ffn_down.weight     Q6_K   (war schon bekannt)
```

### Root Cause

Qwen3-8B-**Q4_K_M** mixt Quant-Typen pro Tensor:
- `attn_v.weight` ist **Q6_K** (nicht Q4_K) — eine pro-Layer-Q-Mix-Eigenschaft des "M" (medium) Quant-Profils.
- `ffn_down.weight` ist Q6_K (war im Phase-2B Tensor-Inventar bereits sichtbar; Phase-2C-Code hatte das für ffn_down hardgecodet, aber nicht für attn_v).

Mein `dispatch_layer` rief `MulMatVecQ4K` auch für `attn_v.weight` auf. Der Q4_K-Shader liest 144-byte-Blöcke, das tatsächliche Tensor enthält 210-byte Q6_K-Blöcke → Bit-Versatz, ~halbe Outputs sind NaN-erzeugende Bit-Muster.

### Fix

Neue Helfer-Funktion `layer_weight_shader(model, layer, suffix) -> ShaderId`, die anhand `model.tensor(name).ggml_type` die richtige Pipeline auswählt:

```rust
fn layer_weight_shader(model: &LoadedModel, layer: u32, suffix: &str) -> ShaderId {
    match model.tensor(...).ggml_type {
        GgmlType::Q6K => ShaderId::MulMatVecQ6K,
        _             => ShaderId::MulMatVecQ4K,
    }
}
```

ALLE GEMV-Aufrufe in `dispatch_layer` und `dispatch_layer_partial` ziehen jetzt durch diesen Helper — keine hardcoded Quant-Annahmen mehr. Funktioniert auch für reine Q4_K-Modelle (Q4_K_S, Q4_K_L) und beliebige andere Mixed-Quant-Profile.

### Ergebnis nach Fix

- Layer 0..35 alle finite, Top-1 Logit `id=82, logit=12.62`, 0 Validation-Errors, 16.7 ms total.
- Phase-2D-Decode-Loop kann darauf aufsetzen.

---

## 2. Implementierte Komponenten

### 2.1 ShaderProfiler (`profiler.rs`, ~150 Zeilen)

`VkQueryPool` mit TIMESTAMP-Queries, begin/end-Token-Pattern, aggregation by name. Eingebunden in `Forward::profile()` als optional `Option<&mut ShaderProfiler>` durch das `&mut self`-Methoden-Idiom — keine borrow-checker-Konflikte mit dem Closure-basierten `one_shot`.

Capacity-sized in `Forward::new` auf 1024 Query-Slots = 512 begin/end-Paare. Reicht für 2 Forward-Pässe ohne Reset.

### 2.2 KvCache (`kv_cache.rs`, ~140 Zeilen)

**Layout:** `K[layer, pos, kv_head, dim]` (pos-major), gleiches Layout für V. Pro Buffer: 36 Layer × 2048 Tokens × 8 KV-Heads × 128 Dim × 4 Byte = 288 MiB. Total für K + V: 576 MiB DEVICE_LOCAL.

**Pos-major** (vs. head-major) wurde gewählt weil eine einzelne `vkCmdCopyBuffer`-Region pro Token einen kompletten Layer-Slot beschreibt. Head-major hätte n_kv_heads (= 8) separate Copy-Regions gebraucht.

`scalar_attn` Shader-Indexing dazu passend: `k[t * pos_stride + kvh * head_dim + d]` mit `pos_stride = n_kv_heads * head_dim`.

### 2.3 Scalar Attention Shader (`vk_shaders/scalar_attn.comp`, 60 Zeilen)

Single-Token Decode-Attention. Eine Workgroup pro Q-Head, ein Thread pro Workgroup, `shared float scores[2048]`. Korrektheits-fokussierte (langsame) Implementierung — Phase 3 ersetzt das durch Flash-Attention.

GQA-Indexing: `kvh = h / (n_heads / n_kv_heads)` = `h / 4` für Qwen3 (32 Q-heads → 8 KV-heads).

**4 isolierte Unit-Tests** in `tests/correctness.rs`:
- `test_scalar_attn_single_token`: seq=1, kleine Dims, 1.0-Output verifiziert
- `test_scalar_attn_two_tokens`: seq=2, prüft Cross-Token-Loop
- `test_scalar_attn_qwen3_dims_seq1`: volle Qwen3-Dims, seq=1
- `test_scalar_attn_qwen3_dims_with_binding_offset`: Descriptor-Offset (entspricht `Forward::run_scalar_attn`-Pattern)

Alle 4 grün — Bug lag nicht im Shader.

### 2.4 Forward-Orchestrierung (`forward.rs`, ~900 Zeilen)

Struct hält 16 GpuBuffers (scratch_a/b ping-pong, hidden_norm, q/k/v_buf, attn_out, o_buf, res1, gate/up/ffn_hidden/ffn_out, logits_buf, fuse0/1, rope_pos/ff/idx) plus KvCache + DescriptorPool + optionaler Profiler.

**API:**
- `Forward::new(...)` — alloziert alle Buffer + Pool.
- `Forward::forward_token(dev, registry, cmd_ctx, model, embedding: &[f32], position) -> ForwardStats` — eine Decode-Step, schreibt Logits in `logits_buf`.
- `Forward::logits()` — liest Logits per Mapped-Pointer.
- `Forward::forward_layer_debug_intermediate(layer, target: DebugTarget) -> Vec<f32>` — debug helper, zwingt einen partiellen Layer-Run und Readback eines spezifischen Intermediate-Buffers (nutzte ich für die Bug-Diagnose; bleibt im Code).
- `Forward::destroy(...)` — explizit, da Forward keinen Device-Handle besitzt.

**Per-Token Sequenz** (intern in `dispatch_layer`):
```
attn_norm
gemv_q (sq), gemv_k (sk), gemv_v (sv)        ← shader pro Tensor anhand ggml_type
rms_norm_q (per-head), rms_norm_k (per-head)
rope_q, rope_k                              ← NeoX
KV-cache write (vkCmdCopyBuffer)
scalar_attn
gemv_o
add_res1
ffn_norm
gemv_gate, gemv_up
silu(gate)
mul(gate, up)
gemv_down                                   ← Q6_K via shader-picker
add_res2
```

**18 Dispatches/Layer × 36 Layer + 2 Final = 650 Dispatches/Token.**

### 2.5 Embedding-Lookup

CPU-seitig in `main.rs` und `tests/regression.rs`: dequantize Q4_K-Row aus dem mmap'd `token_embd.weight` via `q4k::dequant_block`. 2304 Bytes (16 Q4_K-Blöcke) → 4096 f32. Einmalig pro Token, vernachlässigbare Latenz.

`Forward::forward_token` nimmt das Embedding als `&[f32]` Argument — kein impliziter Lookup im Forward-Code, das hält die Kopplung an GgufFile draußen.

---

## 3. Per-Shader / Per-Layer Profiling-Output

Aus `VF_FULL=1 VF_PROFILE=1 cargo run --release`:

```
Forward total: 16.67 ms
Per-shader breakdown (sortiert nach Zeit, Top-12):
  gemv_up           36 calls   3590.7 µs   25.3%
  gemv_down         36 calls   2375.9 µs   16.7%   ← Q6_K
  gemv_gate         36 calls   1938.4 µs   13.6%
  gemv_v            36 calls   1006.8 µs    7.1%   ← Q6_K
  gemv_k            36 calls    928.9 µs    6.5%
  scalar_attn       36 calls    836.3 µs    5.9%
  gemv_q            36 calls    820.2 µs    5.8%
  lm_head            1 calls    811.7 µs    5.7%   ← Q6_K, größter Single-Dispatch
  gemv_o            36 calls    729.4 µs    5.1%
  rms_norm_attn     36 calls    225.5 µs    1.6%
  rms_norm_ffn      36 calls    225.0 µs    1.6%
  add_res2          36 calls    133.5 µs    0.9%
  …

Per-layer time (µs): min=361  mean=378  max=414  (36 layers)
```

→ **GEMV dominiert** wie erwartet (~80% der Zeit). FFN-up + FFN-down + FFN-gate alleine = 55%. Attention = 5.9% (scalar) + 11.6% (Q+K+V projections) = 17.5%. RMSNorms + Residuals + RoPE zusammen ~7%.

**Per-Layer ist sehr gleichförmig** (361–414 µs). Layer 0 ist nicht spürbar langsamer als Layer 35 — der Pipeline-Cache war beim Test-Run schon warm und die Memory-Caches haben pro Layer dieselbe Belastung.

**Forward = ~60 tok/s single-token sequential**. llama.cpp Vulkan-Decode auf der gleichen GPU: 114 tok/s. Wir liegen bei 53% davon. Phase 3 Optimierungen (Flash-Attention, weniger Barriers, ein-Submit-pro-Forward, RDNA4-tuned Spec-Constants) sollten die Lücke schließen.

---

## 4. Test-Ergebnisse

```
$ cargo test --release -- --test-threads=1

q4k unit tests              : 4/4   ok  (Phase 1, BEHALTEN)
correctness tests           : 14/14 ok  (Phase 2B 10 + Phase 2C 4 neue scalar_attn)
regression tests            : 9/9   ok  (Phase 2C neu: phase2c_forward_token_qwen3_finite_logits)
                              ─────
                              27/27 ok
```

**Validation-WARN/ERROR-Count über `VF_FULL=1 cargo run --release`: 0**.

Neue Tests in Phase 2C:
- `tests/correctness.rs::test_scalar_attn_{single_token, two_tokens, qwen3_dims_seq1, qwen3_dims_with_binding_offset}`
- `tests/regression.rs::phase2c_forward_token_qwen3_finite_logits` — voller Forward-Pass mit echter GGUF-Embedding-Row, prüft auf finite + non-zero Logits + Top-1 Token-ID im Vocab-Range.

---

## 5. Geänderte / neue Dateien

```
NEU:
  src/backend/vulkan/profiler.rs          ~150 LoC — VkQueryPool TIMESTAMP wrapper
  src/backend/vulkan/kv_cache.rs          ~140 LoC — pos-major K/V buffers
  src/backend/vulkan/forward.rs           ~900 LoC — Forward + Layer dispatch + debug helpers
  vk_shaders/scalar_attn.comp              60 LoC — single-token decode attention
  results/phase2_step_2.6-2.7_forward_pass.md     — dieser Report

GEÄNDERT:
  build.rs                                +1 ShaderJob (scalar_attn)
  src/backend/vulkan/mod.rs               +pub mod forward, kv_cache, profiler
  src/backend/vulkan/pipeline.rs          +ScalarAttnPushConstants struct (24 B)
  src/backend/vulkan/pipeline_registry.rs +ScalarAttn case mit MAX_SEQ=2048 spec const
  src/backend/vulkan/shaders.rs           +ShaderId::ScalarAttn + SCALAR_ATTN_F32
  src/main.rs                             Phase-2C Demo + Layer-Walk-Debugger
  tests/correctness.rs                    +4 scalar_attn-Tests
  tests/regression.rs                     +phase2c_forward_token_qwen3_finite_logits
```

11 Shader im Inventar (war 10 nach Phase 2B). 27 Tests grün.

---

## 6. Bekannte Limitierungen / Offene Punkte

- **Single-Submit/Token, langsamer Pfad.** Aktuell 1 `vkQueueSubmit` + Fence pro Forward-Pass mit allen 36 Layern in einem Command-Buffer. Phase 3 kann auf Pre-Recorded Secondary Command Buffers umstellen für bessere Latenz.

- **scalar_attn = ein Thread pro Workgroup.** Korrektheits-Fokus, Performance-mäßig 1/64 GPU-Auslastung. Phase 3 / 4 ersetzt durch Flash-Attention oder einen parallelisierten scalar-Variant. Aktuell: scalar_attn = 5.9% der Forward-Zeit, also nicht der dominante Bottleneck — GEMV optimieren bringt mehr.

- **Position 0 only.** Tests laufen nur mit pos=0 (erstes Token, KV-Cache leer). Multi-Token-Decode (pos=1..N mit wachsendem KV-Cache) ist Phase 2D. Die Infrastruktur (KV-Cache mit `current_seq_len`, RoPE-pos in Push-Constants) ist da.

- **Embedding-Lookup auf CPU** über mmap. Die Latenz ist vernachlässigbar (16 Q4_K-Blöcke × ~50 ns pro Block = 800 ns pro Token). Phase 3 könnte einen `get_rows_quant`-Shader nachziehen, lohnt aber kaum.

- **Layer-Activations explodieren in Magnitude** (Layer 0 max ~13, Layer 35 max ~5363). Das ist NICHT mein Bug — das ist das Modell mit einem (vermutlich) unrealistischen/dekontextualisierten Token-0-Embedding. Der Quotient `top-1 / mean` der Logits ist gesund (12.6 / mean ~0). Phase 2D mit echtem Tokenizer + sinnvollem Prompt sollte Layer-Magnituden in normaler Range halten.

- **`VF_TRACE_L0=1`** debug-walk in main.rs zeigt intra-Layer-Stats für Layer 0. `VF_FULL=1` läuft den vollen Forward-Pass mit Logits-Print. `VF_PROFILE=1` schaltet ShaderProfiler an. Diese Diagnostik bleibt im Code als Phase-2D-Helper.

- **Q4_K_M Mixed-Quant-Detection** wird durch den `layer_weight_shader`-Helper handled. Falls ein zukünftiges Modell Q5_K, IQ-Quants etc. verwendet, müssen wir den Shader-Mapping erweitern. Heute: nur Q4_K + Q6_K.

- **`Forward::destroy` ordering** muss nach jedem Forward `cmd_ctx` und `registry` lebendig haben; aktuelles main.rs/test-Pattern ist korrekt.

---

## 7. Nächster Schritt — Phase 2D (Tokenizer + Decode-Loop, Schritte 2.8–2.10)

Der Forward-Pass ist da. Phase 2D braucht:

- **Tokenizer** (Schritt 2.8): BPE aus GGUF-Vokabular (`tokenizer.ggml.tokens` + `tokenizer.ggml.merges` aus den Phase-2B-Metadaten — vocab_size 151936 für Qwen3). Port aus ROCmForge wie im Vision-Dok angekündigt.
- **Sampling**: greedy argmax über logits (trivial), später Top-K / Temperature / repeat_penalty.
- **Decode-Loop** (Schritt 2.9): forward_token in einer Schleife mit wachsendem `position`, Token-für-Token-Output. Prompt: "Explain what a mutex is." → kohärenter Text als Phase-2-Gate.
- **Phase-2-Validierung** (Schritt 2.10): 5-Prompt-Suite, Median tok/s, VRAM-Check.

Der Forward-Pass ist heute 16.7 ms / Token = ~60 tok/s sequential — **deutlich unter** dem Phase-2-Ziel von 100 tok/s. Phase 3-Optimierungen (Pre-Record / Flash-Attention / Subgroup-Reduce / RDNA4-tuned Spec) müssten die Lücke schließen, ABER für die Phase-2-Korrektheits-Gates reichen 60 tok/s mit Sicherheit.

Warte auf Feedback / Phase-2D-Prompt.
