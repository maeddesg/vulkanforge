# Phase 1 — Schritt 1.2: Vulkan Pipeline Setup

**Datum:** 2026-04-25
**Schritt:** 1.2 (DescriptorSetLayout, PipelineLayout, ShaderModule, ComputePipeline)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Zusammenfassung

- `device.rs` erweitert: Validation-Layer-Aktivierung mit Manifest-Lookup, `VK_EXT_debug_utils` + Debug-Messenger, Vulkan-1.1/1.2-Feature-Chain für die SPIR-V-Capabilities aus Schritt 1.1 (`StorageBuffer16BitAccess`, `StorageBuffer8BitAccess`, `shaderInt8`).
- `pipeline.rs` (NEU): `ComputeKernel`-Wrapper exakt per Prompt §1.2.4. Erzeugt 5-Binding-DescriptorSetLayout, 52-Byte-PushConstantRange, ShaderModule, ComputePipeline mit 3 Specialization-Constants.
- `main.rs` baut die Q4_K-GEMV-Pipeline mit `SpecConstants::SMOKE_DEFAULT` (BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1) und reißt sie wieder ab.
- **GATE:** ✅ Pipeline created, **0 Validation-WARN, 0 Validation-ERROR** beim vollständigen Lebenszyklus (Instance → Device → Pipeline → Destroy).

**STOP-Bedingung Validation-Layer fehlt** war bei der ersten Run-Iteration getroffen — User hat `vulkan-validation-layers` (Khronos 1.4.341) installiert, dann re-run, jetzt grün.

---

## 1. Änderungen `device.rs`

### Validation-Layer-Aktivierung

Neuer Lookup-Pfad:
1. `entry.enumerate_instance_layer_properties()` durchsuchen nach `VK_LAYER_KHRONOS_validation`.
2. Falls vorhanden: Layer + `VK_EXT_debug_utils`-Extension der `InstanceCreateInfo` hinzufügen, **plus** `DebugUtilsMessengerCreateInfoEXT` per `push_next` einhängen, damit instance-Create-Time-Errors bereits unseren Callback erreichen.
3. Falls nicht vorhanden: lautes `eprintln!` und ohne Validation weiter (CI-friendly).

Lebenszyklus des Messengers: `debug_loader: Option<debug_utils::Instance>` + `debug_messenger: vk::DebugUtilsMessengerEXT` als Felder von `VulkanDevice`. `Drop` zerstört Messenger **vor** Instance — Reihenfolge in `device.rs:171-184` dokumentiert.

### Debug-Callback

`vk_debug_callback` (`device.rs:201-227`) klassifiziert Severity (ERROR/WARN/INFO/VERBOSE) und Type (validation/perf/general), schreibt nach `eprintln!` als `[vk SEV/TY] message`. Gibt `vk::FALSE` zurück → Vulkan setzt nach jeder Meldung normal fort (kein Abbruch).

**Severity-Maske: WARNING + ERROR.** INFO ist auf dem Linux-Loader Firehose-Level (Layer-Discovery, ICD-Sortierung, jeder physical-device-Pass), das war beim ersten Test-Run unbrauchbar. Die "Validation Layer ACTIVE"-Bestätigung läuft jetzt über unseren eigenen `println!` in `VulkanDevice::new`.

### Feature-Chain (Q4_K-Capabilities)

```rust
let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
    .storage_buffer8_bit_access(true)
    .shader_int8(true);
let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
    .storage_buffer16_bit_access(true);
let mut features2 = vk::PhysicalDeviceFeatures2::default();

DeviceCreateInfo::default()
    .queue_create_infos(...)
    .push_next(&mut features2)
    .push_next(&mut features11)
    .push_next(&mut features12);
```

Mapping zu den vier SPIR-V-Capabilities (Step 1.1 §4):

| SPIR-V Capability             | Vulkan-Feature                                        |
|-------------------------------|-------------------------------------------------------|
| `Shader`                      | core (immer aktiv)                                    |
| `Int8`                        | `Vulkan12Features.shader_int8`                        |
| `StorageBuffer16BitAccess`    | `Vulkan11Features.storage_buffer16_bit_access`        |
| `StorageBuffer8BitAccess`     | `Vulkan12Features.storage_buffer8_bit_access`         |

**Kein** `enabled_features()`-Aufruf — das wäre eine Doppelspezifikation und löst einen Validation-Error aus, sobald `pNext`-Chain `PhysicalDeviceFeatures2` enthält. ash forciert die Trennung schon durch das Typsystem.

---

## 2. Neuer `pipeline.rs`

### `ComputeKernel`-Struct

```rust
pub struct ComputeKernel {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}
```

Exakt das Layout aus Prompt §1.2.4. Kein Drop-Impl — explizite `destroy(self, device)`-Methode, weil `ComputeKernel` keine eigene `ash::Device`-Referenz hält und die Lifetime-Verflechtung zur `VulkanDevice` unnötig Komplexität schafft. Aufrufer (`main.rs`) muss `destroy` vor dem `VulkanDevice`-Drop aufrufen — momentan trivial sicherzustellen.

### Descriptor-Set-Layout

5 Bindings, alle:
- `descriptor_type = STORAGE_BUFFER`
- `descriptor_count = 1`
- `stage_flags = COMPUTE`

Aus `std::array::from_fn` generiert (`pipeline.rs:71-77`). Die drei aliased SSBOs an Binding 0 (`block_q4_K`, `_packed16`, `_packed32` aus Step 1.1 §4) **kollabieren zu einem Vulkan-Slot**: Vulkan zählt Bindings, nicht SPIR-V-Variablen. Ein einzelner Storage-Buffer wird zur Laufzeit (Step 1.3) gleichzeitig allen drei SPIR-V-Aliasen zur Verfügung stehen.

### Push-Constant-Range

```rust
PushConstantRange {
    stage_flags: COMPUTE,
    offset: 0,
    size: 52,                  // 13 × u32 — siehe Step 1.0 §2.1
}
```

`PUSH_CONSTANT_BYTES` als `pub const` exportiert für späteren Sanity-Check beim `vkCmdPushConstants`.

### Specialization-Constants

`SpecConstants { block_size, num_rows, num_cols }` mit `#[repr(C)]` — die Feldreihenfolge bestimmt die Offsets, die in den `SpecializationMapEntry`s landen (0/4/8). `bytemuck::cast` macht den `[u32; 3] → [u8; 12]` Konvert. `constant_id` 0/1/2 entspricht den GLSL `SpecId`s aus `mul_mat_vec_base.glsl:89-91`.

Konstante `SpecConstants::SMOKE_DEFAULT = (32, 1, 1)` — die GLSL-Defaults, hardware-agnostisch. Für RDNA4 BLOCK_SIZE=64 NUM_ROWS=2 (Step 1.5) reicht es, den Wert dieser Konstante zu ändern; kein API-Change.

### ShaderModule + Pipeline

ShaderModule wird aus `&[u32]` (Aligned via `shaders::spv_words`) erzeugt, **vor** dem Pipeline-Create existiert sie nur kurz und wird **direkt nach** `vkCreateComputePipelines` zerstört (`pipeline.rs:153`) — nach Vulkan-Spec ist die Pipeline danach autonom, das Modul muss nicht weiterleben.

Cleanup-Pfade auf jedem Fehlerzweig: bei `create_pipeline_layout`-Fail wird DescriptorSetLayout zerstört, bei `create_shader_module`-Fail beide, bei `create_compute_pipelines`-Fail Layouts plus etwaige teilerfolgreiche Pipelines aus dem Tuple-Result. Damit gibt es bei keinem Vulkan-Error ein Leak, das Validation-Layer beim Programmexit melden würde.

---

## 3. Test-Ergebnisse

### Run-Output (relevante Zeilen, Filter: ohne INFO-Spam)

```
VulkanForge v0.1.0
WARNING: radv is not a conformant Vulkan implementation, testing use only.
VulkanForge: GPU = AMD Radeon RX 9070 XT (RADV GFX1201)
VulkanForge: Compute Queue Family = 0
VulkanForge: validation layer ACTIVE
✅ Vulkan device initialized
  API Version: 1.4.335
  Heap 0: 32000 MB
  Heap 1: 16304 MB DEVICE_LOCAL
  SPIR-V: mul_mat_vec_q4_k_f32_f32 — 165424 bytes / 41356 words
✅ Q4_K GEMV pipeline created (spec: BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1)
✅ Pipeline destroyed cleanly
```

Die "RADV is not a conformant Vulkan implementation"-Warnung kommt vom Mesa RADV-Driver selbst (libvulkan_radeon.so → stderr, nicht über unseren Debug-Callback). Auf Mesa erwartet, harmlos für unsere Zwecke.

### Validation-Check (GATE-Kriterium)

```bash
$ cargo run --quiet 2>&1 | grep -cE "^\[vk (WARN|ERROR)/"
0
```

**0 Validation-Warnings, 0 Validation-Errors** während kompletter Sequenz:
- Instance erstellen (mit Layer + Debug-Messenger)
- Physical-Device-Auswahl
- Device erstellen (mit Vulkan11/12-Feature-Chain)
- DescriptorSetLayout erstellen
- PipelineLayout erstellen
- ShaderModule erstellen
- ComputePipeline erstellen
- ShaderModule zerstören
- Pipeline + Layouts zerstören
- Device + Messenger + Instance zerstören

| Check                                                        | Ergebnis                                  |
|--------------------------------------------------------------|-------------------------------------------|
| `cargo build`                                                | ✅ grün                                   |
| `cargo run` exit-status 0                                    | ✅                                        |
| `VK_LAYER_KHRONOS_validation` aktiv                          | ✅ "validation layer ACTIVE"              |
| Instance-Create ohne Validation-Errors                        | ✅                                        |
| Device-Create mit Feature-Chain ohne Errors                  | ✅                                        |
| DescriptorSetLayout (5 STORAGE_BUFFER, COMPUTE) ohne Errors  | ✅                                        |
| PipelineLayout (DSL + 52B PC) ohne Errors                    | ✅                                        |
| ShaderModule (41356 SPIR-V Words) ohne Errors                | ✅                                        |
| ComputePipeline (Spec 32/1/1) ohne Errors                    | ✅                                        |
| Teardown ohne Errors                                          | ✅                                        |
| **WARN/ERROR-Count**                                         | **0**                                     |

---

## 4. Design-Entscheidungen

1. **Vulkan11Features + Vulkan12Features** statt der älteren Einzel-Feature-Strukturen (`PhysicalDevice8BitStorageFeatures`, `PhysicalDevice16BitStorageFeatures`, `PhysicalDeviceShaderFloat16Int8Features`): kompakter, weniger pNext-Glieder, Standard für Vulkan 1.2+ Code. Wir laufen auf API-Version 1.3 (siehe `device.rs:33`), also voll unterstützt.
2. **Kein Drop-Impl auf `ComputeKernel`** — explizite `destroy(self, device)`-Methode. Begründet in §2 oben. Falls später mehrere ComputeKernels gleichzeitig leben, lohnt sich ein zentraler `Drop`-fähiger Wrapper mit `Arc<ash::Device>` — Phase-3-Refactor.
3. **Severity-Maske WARN+ERROR** (kein INFO/VERBOSE im Default). Loader-Discovery-Spam ist informativ aber unbrauchbar im Day-to-Day. Bei Bedarf einmalig auf INFO hochziehen.
4. **ShaderModule-Lebenszeit kurz** (nur während Pipeline-Create). Vermeidet ein viertes Feld in `ComputeKernel` und einen vierten Destroy-Pfad.

---

## 5. Bekannte Limitierungen / Offene Punkte

- **`debug_loader: Option<...>`-Feld** — funktioniert, aber sieht in `device.rs` nach Boilerplate aus. Wenn wir Validation in CI immer voraussetzen können, kann das `Option` weg. Aktuell zu konservativ → Phase-3-Cleanup.
- **Spec-Constants nicht via Pipeline-Cache deduped** — bei mehreren Spec-Variants entstehen N Pipelines. Akzeptabel solange der Cache (kommt Schritt 1.5 / Phase 2) ihn deckt.
- **`VkPhysicalDeviceFeatures2` ohne explizite Feature-Bits** — lassen wir leer (nur den 1.1/1.2-Pointer-Chain). Vulkan-Spec garantiert dass `pEnabledFeatures = NULL` und `pNext->PhysicalDeviceFeatures2` exklusiv genutzt werden müssen. Kein Konflikt.
- **Keine Pipeline-Cache** (`VkPipelineCache::null()`). Erste Pipeline-Erstellung ist daher gegen die Treiber-Cache-Disk; second-run ist trotzdem schnell weil RADV intern cached. Explicit cache via `vkCreatePipelineCache` lohnt erst bei mehreren Pipelines (Step 1.5+).
- **Zwei `WARNING: radv is not a conformant Vulkan implementation`-Zeilen vom Driver** — kein Validation-Issue, kommt direkt aus libvulkan_radeon. Auf Mesa-Standard.

---

## 6. Geänderte / neue Dateien

```
src/backend/vulkan/device.rs    (~erweitert: 95 → 227 Zeilen)
src/backend/vulkan/pipeline.rs  (NEU,  179 Zeilen)
src/backend/vulkan/mod.rs       (+1 Zeile: pub mod pipeline;)
src/main.rs                     (~+25 Zeilen: Pipeline-Create + Destroy)
results/phase1_step_1.2_pipeline_setup.md  (NEU, dieser Report)
```

`Cargo.toml` unverändert. `bytemuck` war bereits Dependency.

---

## 7. Nächster Schritt

**Schritt 1.3** — Buffer-Allokation + Testdaten:
- `gpu-allocator` einbinden (Dependency bereits vorhanden), `Allocator` hinter `VulkanDevice` aufbauen.
- 5 GPU-Buffer allozieren: weights (288 B Q4_K), input (1024 B f32), output (8 B f32), fuse0 + fuse1 (je 16 B Dummy).
- Staging-Upload-Helfer (HOST_VISIBLE → DEVICE_LOCAL via `vkCmdCopyBuffer`).
- Synthetische Q4_K-Blöcke generieren (deterministische dm/scales/qs).
- CPU-Referenz: Q4_K-Dequant + GEMV in f32, gespeichert für Vergleich in Schritt 1.4.

**Wartet auf User-Bestätigung gemäß Regel 0.**
