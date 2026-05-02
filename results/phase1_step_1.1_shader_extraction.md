# Phase 1 — Schritt 1.1: Shader-Extraktion + SPIR-V-Kompilierung

**Datum:** 2026-04-25
**Schritt:** 1.1 (Shader nach `vk_shaders/` kopieren, build.rs mit shaderc, SPIR-V generieren)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Zusammenfassung

- 4 GLSL-Quelldateien aus `~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/` nach `vk_shaders/` kopiert (3 neu, 1 schon identisch vorhanden).
- `build.rs` erstellt: shaderc kompiliert `mul_mat_vec_q4_k.comp` mit den 7 Defines aus Schritt 1.0 zu `$OUT_DIR/mul_mat_vec_q4_k_f32_f32.spv`.
- Embedding via `include_bytes!` in `src/backend/vulkan/shaders.rs` (Decision: embed statt Runtime-Load — siehe §3).
- `cargo build` grün, **165 424 Bytes SPIR-V (41 356 u32-Words)**.
- `spirv-val --target-env vulkan1.2` passt.
- `cargo run` läuft weiterhin (Phase-0-Smoke unverändert: GPU = AMD Radeon RX 9070 XT, Heap 1 = 16304 MB DEVICE_LOCAL).

**GATE: ✅ erfüllt.** SPIR-V wird erzeugt, keine Compile-Fehler.

---

## 1. Kopierte Dateien

```
vk_shaders/
  mul_mat_vec_q4_k.comp         (134 Zeilen, identisch zur llama.cpp-Vorlage; war bereits im Repo)
  mul_mat_vec_base.glsl         (230 Zeilen, NEU)
  mul_mat_vec_iface.glsl        ( 35 Zeilen, NEU)
  types.glsl                    (653 Zeilen, NEU)
```

`dequant_q4_k.comp` aus dem Initial-Commit bleibt unangetastet — gehört zum separaten Weight-Dequant-Pfad und ist für Phase 1 nicht erforderlich.

---

## 2. build.rs — Implementierungsdetails

### Job-Struktur

`JOBS`-Tabelle in `build.rs:23-39` listet pro SPIR-V-Output: `out_name`, `entry_source`, `defines`. Im Moment nur 1 Eintrag. Erweiterung für f16_f32 / Subgroup-Varianten in Schritt 1.5 erfolgt durch zusätzliche `ShaderJob`-Einträge — kein Code-Change.

### shaderc-Optionen

| Option                                             | Wert                            | Quelle / Begründung                                          |
|----------------------------------------------------|---------------------------------|--------------------------------------------------------------|
| `set_source_language`                              | `GLSL`                          | Default, explizit für Klarheit                               |
| `set_target_env`                                   | `Vulkan, Vulkan1_2`             | llama.cpp-Default (`vulkan-shaders-gen.cpp:328`)             |
| `set_optimization_level`                           | `Performance`                   | entspricht glslc `-O`                                        |
| `set_generate_debug_info`                          | enabled                         | `OpString`/`OpLine` für `spirv-dis`-Debugging                |
| `add_macro_definition` × 7                         | siehe §3.3 in Step 1.0          | identisch zum llama.cpp-Generator                            |
| `set_include_callback`                             | `vk_shaders/`-relative          | resolve `#include "mul_mat_vec_base.glsl"` etc.              |

### Defines (verifiziert gegen `vulkan-shaders-gen.cpp:688, 695`)

```
DATA_A_Q4_K=1
B_TYPE=float        B_TYPEV2=vec2     B_TYPEV4=vec4
D_TYPE=float
FLOAT_TYPE=float    FLOAT_TYPEV2=vec2
```

### Sanity-Checks im Build

`build.rs:84-87`:
- Length % 4 == 0 (SPIR-V-Words sind u32-aligned)
- Magic-Number `[0x03, 0x02, 0x23, 0x07]` (LE-Encoding von 0x07230203)

Würde shaderc Müll oder einen leeren Output produzieren, schlägt der Build sofort fehl statt das Problem an Schritt 1.2 weiterzureichen.

### `cargo:rerun-if-changed`-Tracking

Vier Quelldateien + `build.rs` selbst. Editieren einer `.glsl`/`.comp`-Datei reicht für Re-Compile.

---

## 3. Design-Entscheidung: Embed vs. Runtime-Load

**Gewählt:** Embed via `include_bytes!(concat!(env!("OUT_DIR"), "/mul_mat_vec_q4_k_f32_f32.spv"))`.

| Aspekt                       | Embed (gewählt)               | Runtime-Load                          |
|------------------------------|-------------------------------|---------------------------------------|
| Deployment                   | Single-Binary, keine Pfade    | `.spv` muss neben Binary liegen       |
| Erste Iteration / Smoke-Test | trivial                       | Search-Path-Bugs möglich              |
| Hot-Reload bei Shader-Änderungen | Recompile nötig            | Replace `.spv`, Binary unverändert    |
| Binary-Größe                 | +165 KB pro Variante (debug)  | unverändert                           |
| Runtime-Filesystem-IO        | 0                             | `mmap` oder `fs::read` beim Start     |

Für PoC-Zwecke und Phase-2-Inferenz-Loop ist Embed klar überlegen. Der Hot-Reload-Vorteil von Runtime-Load wird erst bei größeren Shader-Bibliotheken (Phase 3) relevant; falls nötig, ist die Umstellung lokal in `shaders.rs` trivial.

---

## 4. SPIR-V-Inspektion (TODO aus Schritt 1.0 erfüllt)

`spirv-val --target-env vulkan1.2 mul_mat_vec_q4_k_f32_f32.spv` → **OK**.

### Kennwerte (aus `spirv-dis`)

| Eigenschaft           | Wert                                                                      |
|-----------------------|---------------------------------------------------------------------------|
| SPIR-V Version        | 1.5 (Vulkan 1.2 default)                                                  |
| Generator             | `Google Shaderc over Glslang; 11`                                         |
| Bound (max ID)        | 2505                                                                      |
| Memory Model          | Logical / GLSL450                                                         |
| Entry Point           | `OpEntryPoint GLCompute %main "main"`                                     |
| ExecutionMode         | `LocalSize 1 1 1` (wird durch SpecId-0 → BLOCK_SIZE überschrieben)        |
| `gl_WorkGroupSize.x`  | spec const (SpecId 0) — überschreibt LocalSize zur Pipeline-Createzeit    |

### Capabilities (declared)

```
OpCapability Shader
OpCapability Int8
OpCapability StorageBuffer16BitAccess
OpCapability StorageBuffer8BitAccess
```

→ Für **Schritt 1.2** muss `device.rs` aktivieren:
- `VkPhysicalDevice8BitStorageFeatures.storageBuffer8BitAccess`
- `VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess`
- `VkPhysicalDeviceFeatures.shaderInt8` (bzw. via `Vulkan12Features.shaderInt8` in 1.2-Style)

`StorageInputOutput16/PushConstant16` werden **nicht** verlangt — die f16-Daten leben ausschließlich in SSBOs.

### Bindings (Antwort auf TODO "welche aliased Bindings überleben `glslc -O`?")

| DescriptorSet | Binding | Anzahl SSBO-Aliase im SPIR-V | Bedeutung                                           |
|---------------|---------|------------------------------|-----------------------------------------------------|
| 0             | 0       | **3** (`%__2`, `%__3`, `%__4`) | `block_q4_K` + `_packed16` + `_packed32` — alle 3 werden referenziert (`data_a`, `data_a_packed16`, `data_a_packed32`), `glslc -O` pruned NICHTS |
| 0             | 1       | 1 (`%__5`)                   | `B` (durch B_TYPEV4-Alias `data_b_v4`)              |
| 0             | 2       | 1 (`%__1`)                   | `D` (output)                                        |
| 0             | 3       | 1 (`%_`)                     | `Fuse0` — deklariert, aber bei `fusion_flags=0` ungelesen |
| 0             | 4       | 1 (`%__0`)                   | `Fuse1` — dito                                      |

→ **5 Bindings, 7 SSBO-Decorations.** Pipeline-Layout in Schritt 1.2: `VkDescriptorSetLayout` mit 5 Bindings (Binding 0 deckt alle 3 Aliase ab — Vulkan-Layout zählt nur unique Binding-Slots, nicht die SPIR-V-Variablen).

### Specialization Constants

```
OpDecorate %BLOCK_SIZE  SpecId 0
OpDecorate %NUM_ROWS    SpecId 1
OpDecorate %NUM_COLS    SpecId 2
OpDecorate %853         SpecId 0   (zweite Referenz: gl_WorkGroupSize.x)
```

Bestätigt §2.3 aus Schritt-1.0-Report: drei Spec-Constants, BLOCK_SIZE wird sowohl als Schleifen-Schranke als auch als `local_size_x` genutzt.

### Push Constant Block

```
%parameter = OpTypeStruct %uint × 13
%p         = OpVariable PushConstant
```

→ 13 × `u32` = 52 Bytes, Layout-konsistent zu `vk_mat_vec_push_constants`.

---

## 5. Test-Ergebnisse

| Check                                                        | Ergebnis                                  |
|--------------------------------------------------------------|-------------------------------------------|
| `cargo build` erfolgreich                                    | ✅ (4.76 s, dev profile)                   |
| build.rs warning-Output                                      | `compiled mul_mat_vec_q4_k.comp -> mul_mat_vec_q4_k_f32_f32.spv (165424 bytes, 41356 u32 words)` |
| SPIR-V Magic-Number korrekt                                  | ✅ (Build-Assertion passt)                 |
| SPIR-V Längen-Alignment (% 4 == 0)                           | ✅                                        |
| `spirv-val --target-env vulkan1.2`                           | ✅ OK                                     |
| Entry Point `main`                                           | ✅                                        |
| 5 Bindings (DescriptorSet 0, Binding 0–4)                    | ✅ alle deklariert                        |
| 3 Aliased SSBOs an Binding 0                                 | ✅ alle 3 überleben `-O`                   |
| 3 Specialization Constants (SpecId 0/1/2)                    | ✅                                        |
| Push Constant struct = 13 × u32                              | ✅                                        |
| `cargo run` (Phase-0-Smoke) läuft unverändert                | ✅ "GPU = AMD Radeon RX 9070 XT"           |
| **STOP-Bedingung "shaderc kompiliert nicht" ausgelöst?**     | **❌ Nein**                               |

---

## 6. Bekannte Limitierungen / Offene Punkte

- **Debug-Info aktiviert** → ~165 KB SPIR-V. Ohne `set_generate_debug_info` schätzungsweise 50–60 KB. Kein Performance-Impact (Treiber strippt Debug-Info beim Pipeline-Create), aber Binary-Größe. Switch auf release-only-debug ist trivial; warte auf Schritt 1.5.
- **Nur 1 Variante kompiliert** (`f32_f32`). f16_f32, Subgroup-Varianten und der Q8_1-quantize-input-Pfad kommen erst in Schritt 1.5 dazu.
- **Capability `StorageBuffer16BitAccess`** verlangt explizite Aktivierung der entsprechenden Device-Feature-Struct in Schritt 1.2 — `device.rs` muss erweitert werden.
- **shaderc bundled vs. system-link:** shaderc-rs 0.8 hat hier zur System-`libshaderc_combined.so` gelinkt (Build-Output zeigt keine cmake/cc-Builds für shaderc selbst — sondern nur shaderc-sys/-rs Wrapper). Das ist auf Arch mit `pacman -Qi shaderc` (2026.1-2.1) sauber. Bei CI / anderen Distros muss man ggf. SHADERC_LIB_DIR setzen oder das `build-from-source`-Feature aktivieren.
- **Binding 0 dreifach im SPIR-V:** Vulkan-Validation-Layer könnten beim `vkCreateDescriptorSetLayout` warnen, dass mehrere SSBO-Decorations dasselbe Binding teilen. llama.cpp tut das produktiv, also funktioniert es — wir prüfen Validation-Output explizit in Schritt 1.2.

---

## 7. Geänderte / neue Dateien

```
build.rs                                  (NEU,  108 Zeilen)
src/backend/vulkan/shaders.rs             (NEU,   18 Zeilen)
src/backend/vulkan/mod.rs                 (+1 Zeile: pub mod shaders;)
vk_shaders/mul_mat_vec_base.glsl          (NEU,  230 Zeilen, kopiert)
vk_shaders/mul_mat_vec_iface.glsl         (NEU,   35 Zeilen, kopiert)
vk_shaders/types.glsl                     (NEU,  653 Zeilen, kopiert)
results/phase1_step_1.1_shader_extraction.md  (NEU, dieser Report)
```

`Cargo.toml` unverändert — `shaderc = "0.8"` war bereits als build-dependency deklariert (siehe Cargo.toml:14-15).
`vk_shaders/mul_mat_vec_q4_k.comp` unverändert (war bereits identisch zur llama.cpp-Vorlage im Initial-Commit).

---

## 8. Nächster Schritt

**Schritt 1.2** — Vulkan-Pipeline-Setup:
- `device.rs` um `VkPhysicalDevice8BitStorageFeatures` + `16BitStorageFeatures` + `shaderInt8` erweitern.
- Validation-Layer (`VK_LAYER_KHRONOS_validation`) aktivieren.
- `VkDescriptorSetLayout` für die 5 Bindings (alle `STORAGE_BUFFER`).
- `VkPipelineLayout` mit dem Layout + 52-Byte Push-Constant-Range (COMPUTE).
- `VkShaderModule` aus `MUL_MAT_VEC_Q4_K_F32_F32` (`shaders::spv_words`).
- `VkComputePipeline` mit Spec-Constants `[BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1]`.
- `pub struct ComputeKernel { pipeline, pipeline_layout, descriptor_set_layout }` exakt wie im Prompt §1.2.4 spezifiziert.

**Wartet auf User-Bestätigung gemäß Regel 0.**
