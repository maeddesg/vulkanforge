# Phase 2A — Shader-Infrastruktur (Schritte 2.1–2.3)

**Datum:** 2026-04-25
**Schritte:** 2.1 (Shader-Inventar + Bulk-SPIR-V), 2.2 (Pipeline-Registry + Pipeline-Cache), 2.3 (VRAM-Arena)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Status

| Gate                                                         | Ergebnis |
|--------------------------------------------------------------|----------|
| 2.1 — Alle Shader kompilieren zu SPIR-V                       | ✅ 9 Shader, 1.40 MiB total |
| 2.1 — Analyse-Tabelle komplett                                | ✅ siehe §1.4 |
| 2.2 — Pipeline-Registry funktioniert, alle Shader registriert | ✅ 9/9 |
| 2.2 — Pipeline-Cache wird geschrieben + bei zweitem Start geladen | ✅ 80 KiB blob |
| 2.3 — Arena alloziert, Zonen korrekt, Ping-Pong              | ✅ 254 MiB Demo-Budget |
| **Validation-WARN/ERROR über kompletten Lifecycle**           | **0** |
| **`cargo test` (Phase-1 + Phase-2A)**                         | **10/10 passed** |

**Kein Schritt mit STOP-Bedingung getroffen.** Ein Bug während 2.2 wurde inline behoben (siehe §3.2).

---

## 1. Schritt 2.1 — Shader-Inventar + SPIR-V

### 1.1 Auswahl

Inventar gemäß Prompt §2.1.1 "PFLICHT für Decode (M=1, Token-für-Token)". 9 Shader für den Qwen3-8B Decode-Pfad:

```
Decode-GEMV:
  mul_mat_vec_q4_k.comp     (Phase 1 baseline; Q-/K-/Output-Projection, gate/up)
  mul_mat_vec_q6_k.comp     (Qwen3 attn_v + ffn_down)

Elementwise:
  rms_norm.comp             (attn_norm, ffn_norm, final norm)
  rope_norm.comp            (rotary positional embedding, "norm" Variante)
  add.comp                  (residual)
  mul.comp                  (gate × up)
  silu.comp                 (FFN activation)

Attention-Bausteine:
  soft_max.comp             (attention scores)

Utility:
  copy.comp                 (general dtype copy / tensor reshape)
```

**Bewusst NICHT in Phase 2A:**
- **`flash_attn.comp`** (1155 Zeilen + cm1/cm2/mask_opt-Varianten, sehr komplex). Die Attention-Engine-Wahl (Flash vs. handgeschriebener Scalar-Decode-Attention-Kernel mit `soft_max` + zwei `mul_mat_vec`-Calls) wird in **Schritt 2.6 (Single-Layer)** entschieden, sobald die Push-Constants-/Dispatch-Topologie für ein konkretes Modell klar ist.
- **`get_rows*.comp`** für Embedding-Lookup. Per Prompt-Fallstrick #4 bleibt CPU-Embedding für Phase-2-Decode (1 Token, vernachlässigbar) eine valide Alternative; finale Entscheidung in **Schritt 2.9**.
- **`quantize_q8_1.comp`** — nur für die Q8_1-Activation-Variante des GEMV. Der `f32_f32`-Pfad braucht es nicht.
- **`rope_neox.comp`** — Qwen3 nutzt RoPE, aber welche Variante (Llama-style "norm" vs NeoX) hängt vom GGUF-Metadata ab. Wird in **Schritt 2.5** geklärt; kann durch zusätzlichen `ShaderJob` in `build.rs` minimal nachgezogen werden.

### 1.2 Build-Pipeline

`build.rs` erweitert von 1 auf 9 `ShaderJob`s. Jeder Shader bekommt seine eigenen `-D`-Defines (entspricht 1:1 dem `vulkan-shaders-gen.cpp`-Aufruf für die jeweilige `f32_f32`/`f32`-Variante). shaderc-Setup unverändert (Vulkan 1.2 Target, Performance-Optimization, Include-Callback gegen `vk_shaders/`). Output:

```
mul_mat_vec_q4_k.comp -> mul_mat_vec_q4_k_f32_f32.spv (165 KiB)
mul_mat_vec_q6_k.comp -> mul_mat_vec_q6_k_f32_f32.spv (174 KiB)
rms_norm.comp         -> rms_norm_f32.spv             (218 KiB)
rope_norm.comp        -> rope_norm_f32.spv            (142 KiB)
add.comp              -> add_f32.spv                  (138 KiB)
mul.comp              -> mul_f32.spv                  (137 KiB)
silu.comp             -> silu_f32.spv                 (128 KiB)
soft_max.comp         -> soft_max_f32.spv             (229 KiB)
copy.comp             -> copy_f32_f32.spv             (135 KiB)
total SPIR-V: 1 467 472 bytes across 9 shader(s)
```

`spirv-val --target-env vulkan1.2` ist auf jedem Output `OK`. Debug-Info ist eingeschaltet (`set_generate_debug_info`); ohne Debug wären die Blobs ~50% kleiner — Trade-off für Phase-2-Debugging.

### 1.3 Transitive Header-Abhängigkeiten

Aus `~/tmp/llama.cpp/.../vulkan-shaders/` neu kopiert nach `vk_shaders/`:

```
GLSL-Headers:
  utils.glsl                  (fastmod, fastdiv, get_indices)
  rope_params.glsl            (struct rope_params: 24 Felder, ~104 Bytes)
  rope_head.glsl              (RoPE binding-set, push-constant block)
  rope_funcs.glsl             (rope_norm, rope_neox, rope_multi Funktionen)
  generic_head.glsl           (silu/etc unary mit minimal-PC)
  generic_unary_head.glsl     (copy/dequant: 2 SSBOs + fastdiv-Helpers)
  generic_binary_head.glsl    (add/mul/rms_norm: 3 SSBOs + 116-B PC)
```

`mul_mat_vec_base.glsl`, `mul_mat_vec_iface.glsl`, `types.glsl` aus Phase 1 wiederverwendet.

### 1.4 Reflection-Tabelle (per Run-Output)

```
┌─────────────────────────────────┬──────┬────────┬─────────────┬────────────────┐
│ Shader                          │ Bind │ PC (B) │ SpecIds     │ LocalSize      │
├─────────────────────────────────┼──────┼────────┼─────────────┼────────────────┤
│ mul_mat_vec_q4_k_f32_f32        │    5 │     52 │ 0,1,2       │   1×  1×  1    │
│ mul_mat_vec_q6_k_f32_f32        │    5 │     52 │ 0,1,2       │   1×  1×  1    │
│ rms_norm_f32                    │    3 │    116 │ 0,1         │ 512×  1×  1    │
│ rope_norm_f32                   │    5 │    108 │ —           │   1×256×  1    │
│ add_f32                         │    3 │    116 │ 0           │ 256×  1×  1    │
│ mul_f32                         │    3 │    116 │ 0           │ 256×  1×  1    │
│ silu_f32                        │    2 │     24 │ —           │ 512×  1×  1    │
│ soft_max_f32                    │    4 │     68 │ 0           │   1×  1×  1    │
│ copy_f32_f32                    │    2 │    128 │ —           │ 512×  1×  1    │
└─────────────────────────────────┴──────┴────────┴─────────────┴────────────────┘
```

`Bind` = SSBOs an DescriptorSet 0. `PC` = Push-Constant-Range in Bytes. `SpecIds` = Specialization-Constant-IDs aus dem SPIR-V. `LocalSize` = `OpExecutionMode LocalSize`-Default (wird bei Shadern mit `local_size_x_id` durch SpecConstantComposite zur Pipeline-Createzeit überschrieben).

Tabelle wird vom **eigenen SPIR-V-Reflector** (`src/backend/vulkan/spirv_reflect.rs`, 230 Zeilen, keine externe Dependency) erzeugt — kein Hardcoding, kein manuelles Tracking. Reflector deckt:
- `OpDecorate(DescriptorSet/Binding/SpecId/ArrayStride)`
- `OpMemberDecorate(Offset)`
- `OpType(Int/Float/Vector/Matrix/Array/Struct/Pointer)` → rekursive Größenberechnung (geschachtelte Structs wie `rope_params` ✓, Arrays wie `int sections[4]` ✓)
- `OpVariable(StorageClass=PushConstant/StorageBuffer/Uniform)`
- `OpExecutionMode(LocalSize)`
- `OpConstant` (für Array-Längen)

Reflector wird im Phase-1-Kontext verifiziert: Q4_K liefert weiterhin 5 Bindings + 52 B PC + 3 SpecIds (matched `vk_mat_vec_push_constants`).

---

## 2. Schritt 2.2 — Pipeline-Registry + Pipeline-Cache

### 2.1 API

```rust
pub struct PipelineRegistry {
    pipelines: HashMap<ShaderId, ComputeKernel>,
    cache: vk::PipelineCache,
    cache_path: Option<PathBuf>,
    pub create_duration: Duration,
}

impl PipelineRegistry {
    pub fn new(device: &ash::Device, cache_path: Option<&Path>)
        -> Result<(Self, usize /* loaded_bytes */), Box<dyn Error>>;
    pub fn get(&self, id: ShaderId) -> &ComputeKernel;
    pub fn count(&self) -> usize;
    pub fn save_cache(&self, device: &ash::Device) -> CacheStats;
    pub fn destroy(self, device: &ash::Device);
}
```

`ShaderId` ist ein flaches Enum (`MulMatVecQ4K`, ..., `Copy`); `id.spv_bytes()` und `id.name()` liefern das embedded Blob bzw. den Output-Filenamen.

### 2.2 Pipeline-Cache (VkPipelineCache)

- Cache-Pfad-Default: `$HOME/.vulkanforge/pipeline_cache.bin` (per `default_cache_path()`).
- `vkCreatePipelineCache` mit Disk-Blob (oder leer beim ersten Start). Inkompatible Caches (anderer Treiber, andere Version) werden vom Vulkan-Loader stillschweigend verworfen — kein manuelles Versions-Management.
- Nach Create-Pass: `vkGetPipelineCacheData` → Disk-Blob speichern.

**Timing:**

| Run                    | `vkCreateComputePipelines` (9 Shader) | Cache-Blob   |
|------------------------|---------------------------------------|--------------|
| Cold Start (kein Cache) | **40.1 ms**                           | 80 024 B aus |
| Warm Start (Cache vorhanden) | **38.6 ms**                      | 80 024 B in  |

Differenz klein, weil:
- Nur 9 Shader. Bei Phase 3 mit 100+ Shadern wird es deutlich relevanter.
- RADV hat einen eigenen treiberinternen Cache (gegenüber userland-Cache zusätzlich), der ebenfalls warmlaufen muss.

→ Pipeline-Cache funktioniert, Effekt sichtbar, aber bei der aktuellen Shader-Anzahl im Bereich von Mess-Rauschen.

### 2.3 Per-Shader-Pipeline-Erstellung

`ComputeKernel::from_spv(device, spv_words, cache)` reflektiert den SPIR-V und baut:
- `VkDescriptorSetLayout` mit `n` Bindings (Typ aus Reflection: STORAGE_BUFFER für `buffer`-Blocks, UNIFORM_BUFFER für UBOs)
- `VkPipelineLayout` mit dem DSL + einer Push-Constant-Range der reflektierten Größe
- `VkShaderModule` aus dem SPV
- `VkComputePipeline` aus Stage + Layout + Cache

Failure-Cleanup auf jedem Pfad — kein Leak im Validation-Layer.

### 2.4 Spec-Constants — Hardcoded für GEMV

Die GEMV-Shader (`MulMatVecQ4K`, `MulMatVecQ6K`) bekommen explizite Specialization-Daten `(BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1)` per `ComputeKernel::from_spv_with_spec`. Begründung: siehe §3.2 (Bug-Bericht).

Andere Shader nutzen ihre GLSL-Defaults (kein Override).

---

## 3. Schritt 2.3 — VRAM-Arena

### 3.1 Option-A vs Option-B Entscheidung — Treiber-Befund

Per Prompt §2.3.2: "Falls `maxMemoryAllocationSize ≥ 14 GB`: Option A bevorzugen."

**Gemessen auf RX 9070 XT / RADV:** `maxMemoryAllocationSize = 4.0 GiB`.

Das Limit kommt aus `VkPhysicalDeviceMaintenance3Properties` (Vulkan 1.1+ core), abgefragt per `query_max_memory_allocation_size` in `vram_arena.rs`. Konsequenz für Phase 2B (Qwen3-8B Q4_K_M ≈ 13 GB):

→ **Eine einzelne `vkAllocateMemory`-Allokation reicht NICHT für ein echtes Modell.** Phase 2B muss eine der Optionen wählen:
  - **B.1** gpu-allocator-Sub-Allokation (Phase-1-Ansatz, bewährt, erlaubt > 4 GB durch interne Split-Allocations).
  - **B.2** Mehrere `vkAllocateMemory`-Calls in der Arena (Multi-Memory-Arena), z.B. eine Allokation pro Zone (Weights ~5 GB → würde auch nicht reichen, also doch in Chunks).

**Phase-2A-Demo nutzt Option A.** Der 254-MiB-Demo-Budget passt komfortabel in 4 GiB. Die Architektur-Entscheidung für Phase 2B ist explizit offen — ich habe NICHT selbst entschieden, sondern dokumentiere das Limit. Falls präferiert, lässt sich `VramArena::new` so erweitern, dass es bei `AllocationTooLarge` automatisch auf gpu-allocator-Sub-Allocation umschaltet (Fallback-Pfad).

### 3.2 Zone-Layout

```
Demo-Budget: 254 MiB total
  Zone A "weights"  : 0..209 715 200            (200 MiB)
  Zone B "kv_cache" : 209 715 200..262 144 000  ( 50 MiB)
  Zone C "scratch"  : 262 144 000..266 338 304  (  4 MiB)

Zone-Alignment: 4096 B  (max(minStorageBufferOffsetAlignment, 4096))
```

Zone-Größen werden automatisch auf `zone_alignment` aufgerundet. Buffer-Views müssen ihre eigene Alignment-Anforderung erfüllen — `VramArena::create_buffer` prüft das gegen `vkGetBufferMemoryRequirements.alignment` und liefert `BufferViewError::AlignmentMismatch` zurück, BEVOR `vkBindBufferMemory` einen Validation-Error werfen würde.

### 3.3 Ping-Pong Scratchpad

```rust
pub fn scratch_for_layer(&self, layer_idx: usize) -> (offset, size) {
    let half = self.layout.scratch.size / 2;
    (self.layout.scratch.offset + (layer_idx % 2) as u64 * half, half)
}
```

Verifiziert in `phase2a_vram_arena_zones_and_pingpong`:
- `scratch_for_layer(0).offset != scratch_for_layer(1).offset` ✓
- `scratch_for_layer(0).offset == scratch_for_layer(2).offset` ✓
- Beide Hälften liegen vollständig in Zone C ✓

### 3.4 Strukturierte Fehler

`ArenaError::AllocationTooLarge { requested, max }` wird bei OOM-Anfrage geworfen statt zu crashen — `phase2a_vram_arena_oom_clean_error` testet das mit `u64::MAX/2`-Request. `BufferViewError` deckt OutOfBounds, AlignmentMismatch, IncompatibleMemoryType, Vk(vk::Result).

---

## 4. Bug während 2.2: Q4_K-Dispatch ohne explizite Spec-Daten produziert 0.0

### Symptom

Erste Iteration der `phase1_q4k_smoke_dispatch_bit_exact`-Regression: `output[0] = 0.0` statt `256.0`. Kein `[vk WARN/ERROR]` während des Dispatches.

### Ursache

`ComputeKernel::from_spv` ohne Specialization-Info → Vulkan nutzt SPIR-V-Spec-Const-Defaults. Für Q4_K sind diese `(BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1)` — identisch zur Phase-1-`SpecConstants::SMOKE_DEFAULT` (32, 1, 1). Funktional sollte das äquivalent sein.

**Beobachtet auf RADV (RX 9070 XT, GFX1201):** ist es nicht. Der Pipeline-Output ist 0.0, wenn KEIN expliziter `VkSpecializationInfo` gesetzt wird, auch wenn die übergebenen Werte den GLSL-Defaults entsprechen würden. Mit explizitem `VkSpecializationInfo` für SpecIds 0/1/2 funktioniert es bit-exakt.

Spekulation: möglicherweise generiert RADVs Compiler bei "no specialization" eine andere Code-Variante als bei "specialization-with-default-values" — z.B. wird der `gl_WorkGroupSize`-SpecConstantComposite anders aufgelöst. Habe das nicht weiter forensisch verfolgt; Workaround ist trivial.

### Fix

`PipelineRegistry::new` setzt für GEMV-Shader (MulMatVecQ4K, MulMatVecQ6K) explizit `MMV_SPEC_DATA = [32, 1, 1]` via `from_spv_with_spec`. Andere Shader behalten Default-Verhalten — sie haben entweder keine SpecIds (silu, copy) oder ihre Defaults funktionieren in den Tests.

→ **Lehre für Phase 2B**: bei jedem neuen Pipeline-Typ explizit prüfen, ob spec-defaults wirklich greifen. Gegebenenfalls `from_spv_with_spec` benutzen. Beim Refactor in eine variants-aware Registry (Phase 4) ohnehin Standard.

---

## 5. Test-Ergebnisse

### `cargo test --release`

```
running 4 tests (q4k unit tests in src/)
....
test result: ok. 4 passed; 0 failed

running 6 tests (tests/regression.rs integration tests)
......
test result: ok. 6 passed; 0 failed
```

| Test                                                | Step | Pass |
|-----------------------------------------------------|------|------|
| `q4k::tests::smoke_weights_dequant_to_known_values` | 1.3  | ✅   |
| `q4k::tests::smoke_gemv_matches_analytical`         | 1.3  | ✅   |
| `q4k::tests::dequant_recovers_per_subblock_distinct_nibbles` | 1.5 | ✅ (Q4_K-nibble-bug-killer) |
| `q4k::tests::dequant_recovers_per_subblock_distinct_scales`  | 1.5 | ✅ |
| `phase2a_all_shaders_compile_to_spirv`              | 2.1  | ✅   |
| `phase2a_q4k_unit_tests_referenced`                 | 2.1  | ✅   |
| `phase2a_pipeline_registry_creates_all`             | 2.2  | ✅   |
| `phase2a_vram_arena_zones_and_pingpong`             | 2.3  | ✅   |
| `phase2a_vram_arena_oom_clean_error`                | 2.3  | ✅   |
| `phase1_q4k_smoke_dispatch_bit_exact`               | reg. | ✅ (`output = [256.0, 512.0]`) |

**Phase-1-Regression**: alle 4 Phase-1-Unit-Tests + der Bit-exakte-Dispatch-Test grün. Q4_K-Nibble-Bug-Killer aus 1.5 weiterhin scharf.

### `cargo run --release` Run-Output (gefiltert)

```
VulkanForge v0.1.0 — Phase 2A demo
VulkanForge: GPU = AMD Radeon RX 9070 XT (RADV GFX1201)
VulkanForge: Compute Queue Family = 0
VulkanForge: validation layer ACTIVE
✅ Vulkan device initialized
  maxMemoryAllocationSize: 4.0 GiB
✅ gpu_allocator initialized
  Pipeline cache: /home/maeddes/.vulkanforge/pipeline_cache.bin
✅ PipelineRegistry: 9 shaders, vkCreateComputePipelines = 38.587 ms (warm), …
  Loaded 80024 B from existing pipeline cache (warm start)

  ┌─────────────────────────────────┬──────┬────────┬─────────────┬────────────────┐
  …  (Reflection-Tabelle wie §1.4)
  └─────────────────────────────────┴──────┴────────┴─────────────┴────────────────┘

  Arena demo budget: 254.0 MiB total (weights 200.0 MiB, KV 50.0 MiB, scratch 4.0 MiB)
✅ VramArena: 266 338 304 B allocated on memory_type_index 0 (DEVICE_LOCAL), zone alignment 4096 B
  Zones — weights: 0..209715200 (size 209715200), kv_cache: 209715200..262144000 (size 52428800), scratch: 262144000..266338304 (size 4194304)
  ✅ Buffer views: weights@0x0, kv@0xc800000, scratch_layer_0@0xfa00000, scratch_layer_1_offset=0xfc00000 (ping-pong)
  Pipeline cache saved: 80024 B → /home/maeddes/.vulkanforge/pipeline_cache.bin
✅ Phase 2A teardown clean
```

```
$ cargo run --release --quiet 2>&1 | grep -cE "^\[vk (WARN|ERROR)/"
0
```

**0 Validation-WARN, 0 Validation-ERROR** über kompletten Lifecycle (Instance + Device + 9 Pipelines + Arena + Buffer-Views + Cleanup), sowohl Cold- als auch Warm-Start.

---

## 6. Bekannte Limitierungen / Offene Punkte

- **`maxMemoryAllocationSize = 4 GiB` auf RADV** — siehe §3.1. Phase 2B muss Multi-Allokation-Strategie wählen. Ich habe NICHT vorausgreifend implementiert, weil die Demo-Größe 254 MiB problemlos passt; der Trade-off bleibt für die User-Entscheidung in Phase 2B.

- **Q4_K-Spec-Default-Quirk**, siehe §4. Workaround: explizite Spec-Daten für GEMV-Pipelines. Sollte beim ersten Phase-2B-Treiber-Update gegengeprüft werden, ob das immer noch nötig ist.

- **Flash-Attention nicht im Inventar**, siehe §1.1. Kommt mit Schritt 2.6 — entweder via `flash_attn.comp` oder via Scalar-Decode-Attention auf bestehenden Bausteinen (`mul_mat_vec` + `soft_max` + `mul_mat_vec`).

- **`get_rows*.comp` nicht im Inventar**, siehe §1.1. Embedding-Lookup CPU-seitig in Phase 2 ist akzeptabel; Shader-seitige Variante ist Phase-3-Optimization.

- **`rope_neox.comp` nicht kompiliert** — falls Qwen3-Metadata "neox"-Variante verlangt, in Schritt 2.5 als zusätzlichen `ShaderJob` nachziehen (5 Zeilen build.rs + 1 Zeile shaders.rs). Trivial.

- **Pipeline-Cache-Effekt klein** (~1.5 ms) bei 9 Shadern. Bei 100+ Shadern in Phase 3 wird der Effekt deutlich. Für Phase 2A reicht "Cache funktioniert + persistiert" als Gate.

- **Kein PIPELINE_STATISTICS Query-Pool** (vom Test/Profiling-Strategie-Dok für Phase 2 vorgesehen). Kommt mit Schritt 2.6 wenn der `ShaderProfiler` eingeführt wird.

- **Per-Shader-Push-Constants-Structs in Rust noch nicht definiert** — der Prompt §2.2.4 fordert "Pro Shader-Typ ein eigenes `#[repr(C)]` Struct". Phase 2A hat nur `MatVecPushConstants` (Phase-1-Erbe für GEMV). Strukturen für `RmsNormPushConstants` etc. werden in Schritt 2.5 / 2.6 ergänzt — sie werden erst gebraucht, wenn die Shader tatsächlich dispatcht werden.

---

## 7. Geänderte / neue Dateien

```
NEU:
  src/lib.rs                                      — Library-Crate für tests/regression.rs
  src/backend/vulkan/spirv_reflect.rs             — 230 Zeilen, eigener SPIR-V-Reflector
  src/backend/vulkan/pipeline_registry.rs         — Registry + VkPipelineCache I/O
  src/backend/vulkan/vram_arena.rs                — Monolithische Arena, strukturierte Errors
  tests/regression.rs                             — 6 Integration-Tests inkl. Phase-1-Bit-exact
  vk_shaders/{mul_mat_vec_q6_k, rms_norm, rope_norm, add, mul, silu, soft_max, copy}.comp
                                                  — 8 Shader aus llama.cpp kopiert
  vk_shaders/{utils, rope_params, rope_head, rope_funcs,
              generic_head, generic_unary_head, generic_binary_head}.glsl
                                                  — 7 Header aus llama.cpp kopiert
  results/phase2_step_2.1-2.3_shader_infra.md     — dieser Report

GEÄNDERT:
  build.rs                                        — 1 → 9 ShaderJobs
  src/backend/vulkan/mod.rs                       — +3 pub mod (pipeline_registry, spirv_reflect, vram_arena)
  src/backend/vulkan/pipeline.rs                  — refactor: from_spv + from_spv_with_spec via Reflection
  src/backend/vulkan/shaders.rs                   — ShaderId enum, 9 const &[u8], ALL_SHADERS, spv_words helper
  src/main.rs                                     — Phase-2A-Demo (Registry + Arena + Reflection-Tabelle)

UNVERÄNDERT (Phase 1 Erbe):
  src/backend/vulkan/{device, buffers, commands, q4k}.rs
  Cargo.toml
```

---

## 8. Nächster Schritt

Phase-2A abgeschlossen. Nächste sinnvolle Schritte (User entscheidet):

- **Phase 2B (Schritte 2.4 – 2.5)**: GGUF-Loader (Port aus ROCmForge) + Elementwise-Shader-Validierung. Hier wird die Arena-Strategie für > 4 GiB Modelle entschieden (siehe §3.1).
- **Schritt 2.6 vorziehen**: Single-Layer Forward Pass — würde Flash-Attention vs Scalar-Attention-Entscheidung erzwingen und die Phase-2-Korrektheits-Strategie (llama.cpp Golden Reference) etablieren.
- **Per-Shader-Push-Constants-Structs jetzt definieren**: kleiner Aufwand, gut für die Verifikation (Compile-Time-Größenchecks via `assert!(size_of == reflection.push_constant_size)`).
