# Phase 1 — Schritt 1.0: Shader-Analyse

**Datum:** 2026-04-25
**Schritt:** 1.0 (read-only Shader-Analyse, kein Code)
**Quelle:** `~/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Zusammenfassung

- Der Q4_K GEMV ist **eine eigenständige Datei** `mul_mat_vec_q4_k.comp`. Kein Template-System, kein Python-Generator.
- llama.cpp hat einen **C++ glslc-Wrapper** (`vulkan-shaders-gen.cpp`), der für jeden Shader Permutationen von `-D`-Defines durchgeht und glslc aufruft. **Keine Quelltext-Transformation** außerhalb des C-Preprozessors. → Risiko 1 aus dem Vision-Dokument ist **kontrolliert**: shaderc + identische Defines reproduziert das gleiche SPIR-V.
- Wir brauchen **4 Dateien** zum Kompilieren des Shaders:
  - `mul_mat_vec_q4_k.comp`
  - `mul_mat_vec_base.glsl`
  - `mul_mat_vec_iface.glsl`
  - `types.glsl`
- Push-Constants: 13 × uint32 = 52 Bytes (`vk_mat_vec_push_constants`).
- Descriptor-Set: **5 Bindings** (Binding 0=Weights A, 1=Input B, 2=Output D, 3=Fuse0, 4=Fuse1). Fuse0/1 werden bei `fusion_flags=0` nicht gelesen, müssen aber als gültige Storage-Buffer gebunden sein.
- 3 Specialization-Constants: BLOCK_SIZE (constant_id=0), NUM_ROWS (constant_id=1), NUM_COLS (constant_id=2).
- Workgroup-Local-Size: `(BLOCK_SIZE, 1, 1)` — über `local_size_x_id = 0` (specialization).
- Dispatch (für AMD non-GCN, Smoke-Test-Default): X = `ceil(M / NUM_ROWS)`, Y = batch (1), Z = 1.

**Keine STOP-Bedingung getroffen.** Der Generator ist ein glslc-Wrapper, kein Shader-Code-Generator. Der Ansatz aus dem Prompt bleibt gültig.

---

## 1. Shader-Auswahl

### Gefundene Datei

`vk_shaders/mul_mat_vec_q4_k.comp` (134 Zeilen, llama.cpp-Quelle).
Eigenständig pro Quant-Format — die generischen Varianten (`mul_mat_vec.comp` mit Defines) gelten nur für legacy quants. K-quants (`*_k`) und i-quants (`iq*`) haben jeweils einen eigenen Shader. Bestätigt durch `vulkan-shaders-gen.cpp:693`:

```cpp
std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || ...)
    ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";
```

### Generator-Status (Risiko 1 aus Vision-Dok)

`vulkan-shaders-gen.cpp` ist **kein Code-Generator**, sondern ein **paralleler glslc-Wrapper**. Ablauf:
1. Iteration über `(shader, dtype, B-Type, fp16/fp32, subgroup-Mode, …)`-Permutationen.
2. Pro Permutation: `glslc --target-env=vulkan1.2 -fshader-stage=compute <shader>.comp -O -DDATA_A_Q4_K=1 -DB_TYPE=float … -o <name>.spv`.
3. Anschließend werden alle `.spv`-Files in eine `.cpp`-Tabelle einkompiliert (für die Runtime in `ggml-vulkan.cpp`).

Quellzeilen: `vulkan-shaders-gen.cpp:327–401` (`string_to_spv_func`), `vulkan-shaders-gen.cpp:687–719` (mmv-Iteration).

→ **Implikation für VulkanForge:** wir können shaderc (oder externes glslc/glslangValidator) aufrufen, dieselben Defines setzen, identisches SPIR-V erhalten. Kein Python-Tooling notwendig.

---

## 2. Shader-Schnittstelle

### 2.1 Push-Constants (`mul_mat_vec_base.glsl:16-41`)

GLSL-Definition (ohne `MUL_MAT_ID`, was wir nicht brauchen):

```glsl
layout (push_constant) uniform parameter {
    uint ncols;             // K (= ne00 = ne10), Spalten in A / Länge B
    uint stride_a;           // Element-Stride zwischen A-Zeilen (in fp16-Equiv.-Elementen, dann /QUANT_K)
    uint stride_b;           // Element-Stride zwischen B-Zeilen
    uint stride_d;           // Anzahl Output-Zeilen (dient als Schranke!)
    uint batch_stride_a;     // Batch-Stride A
    uint batch_stride_b;     // Batch-Stride B
    uint batch_stride_d;     // Batch-Stride D
    uint fusion_flags;       // Bits: BIAS0=0x1, BIAS1=0x2, SCALE0=0x4, SCALE1=0x8
    uint base_work_group_y;  // Offset in Y (für split-dispatch bei großen Batches)
    uint ne02;               // Batch-Dim
    uint ne12;               // Batch-Dim
    uint broadcast2;         // Batch-Broadcast
    uint broadcast3;         // Batch-Broadcast
} p;
```

**Größe:** 13 × `uint32_t` = **52 Bytes**.
Validiert in `ggml-vulkan.cpp:992-1006` (`struct vk_mat_vec_push_constants`).

Werte für unseren Smoke-Test (M=2, K=256, batch=1):
| Feld              | Wert            | Bedeutung                                             |
|-------------------|-----------------|-------------------------------------------------------|
| ncols             | 256             | K = 1 × QUANT_K_Q4_K                                  |
| stride_a          | 256             | = ne10 (Länge der A-Zeile in Elementen)               |
| stride_b          | 256             | = ne10                                                |
| stride_d          | 2               | = ne01 (Output-Zeilen) — wirkt als Out-of-bounds-Cut  |
| batch_stride_a    | 256 × 2 = 512   | = ne00 × ne01                                         |
| batch_stride_b    | 256             | = ne10 × ne11                                         |
| batch_stride_d    | 2               | = ne20 × ne21                                         |
| fusion_flags      | 0               | Keine Bias/Scale-Fusion                               |
| base_work_group_y | 0               | Erster Batch-Slice                                    |
| ne02, ne12        | 1, 1            | Keine Batch-Dimension                                 |
| broadcast2/3      | 1, 1            | Identitäts-Broadcast                                  |

### 2.2 Descriptor-Set Layout (`mul_mat_vec_iface.glsl`)

5 Bindings, alle `STORAGE_BUFFER`:

| Binding | Name        | Zugriff   | Inhalt (Q4_K f32_f32)                                       |
|---------|-------------|-----------|-------------------------------------------------------------|
| 0       | A           | readonly  | Weights als `block_q4_K[]` (auch aliased als `_packed16/32`)|
| 1       | B           | readonly  | Input-Aktivierungen `float[]` (auch aliased `BV2/BV4`)      |
| 2       | D           | writeonly | Output `float[]`                                            |
| 3       | Fuse0       | readonly  | Bias/Scale 0 — bei `fusion_flags=0` ungelesen, aber gebunden|
| 4       | Fuse1       | readonly  | Bias/Scale 1 — dito                                         |

**Wichtig:** Binding 0 wird **dreifach aliased deklariert** (`block_q4_K`, `block_q4_K_packed16`, `block_q4_K_packed32`). Im Q4_K-GEMV-Hauptpfad wird primär `data_a_packed16` und `data_a_packed32` gelesen (siehe `mul_mat_vec_q4_k.comp:19,37,38`). Vulkan akzeptiert mehrfach aliased SSBOs auf identischem Binding — **alle drei Aliase müssen im Shader-Reflection-Output sichtbar sein, aber teilen sich denselben Buffer**.

Bindings 3 + 4 sind **immer deklariert**, aber bei `fusion_flags=0` wird kein Lesezugriff ausgeführt. Für unseren PoC binden wir dort den **Output-Buffer als Dummy** (oder einen 16-Byte-Dummy-Buffer, sicherer wegen Validation-Layer).

### 2.3 Specialization-Constants

Drei `layout (constant_id = N) const uint NAME = default;` (`mul_mat_vec_base.glsl:89-91`):

| ID | Name        | Default | Wert für Q4_K f32_f32 (RDNA4, AMD non-GCN)                     |
|----|-------------|---------|----------------------------------------------------------------|
| 0  | BLOCK_SIZE  | 32      | `subgroup_size16` = 64 (Wave64 RDNA4) bei SUBGROUP-Mode        |
| 1  | NUM_ROWS    | 1       | `rm_kq` = 2 (AMD non-GCN); 4 bei AMD_GCN                       |
| 2  | NUM_COLS    | 1       | i+1 (typisch 1 für Decode-GEMV; max `mul_mat_vec_max_cols`)    |

Quelle `ggml-vulkan.cpp:4128`, `4148-4158`, `4180`. BLOCK_SIZE entspricht **gleichzeitig** der `local_size_x` (siehe `mul_mat_vec_q4_k.comp:7`: `layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;`).

**Smoke-Test-Wahl:** BLOCK_SIZE=32 (default, geräteunabhängig, kein Subgroup-Add nötig), NUM_ROWS=1, NUM_COLS=1. Damit dispatchen wir 1 Workgroup pro Output-Zeile, ohne `USE_SUBGROUP_ADD`-Pfad. Die Subgroup-optimierten Pfade kommen erst in Schritt 1.5 ins Spiel.

### 2.4 Workgroup-Size & Dispatch-Dimensionen

**Local Size:** `(BLOCK_SIZE, 1, 1)` über Specialization Constant 0.

**Workgroup-Mapping im Shader** (`mul_mat_vec_q4_k.comp:122-133`):

```glsl
const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
if (first_row + NUM_ROWS <= p.stride_d) compute_outputs(first_row, NUM_ROWS);
else if (first_row >= p.stride_d) return;
else compute_outputs(first_row, p.stride_d - first_row);
```

→ **Jede Workgroup berechnet `NUM_ROWS` aufeinanderfolgende Output-Zeilen** (über X- und Z-Workgroup-ID linearisiert). Y-Dim trägt den Batch-Index.

**Dispatch-Berechnung** (aus `ggml-vulkan.cpp:7996-8002, 8044` und `ggml_vk_dispatch_pipeline:6601-6630`):

```
elements   = { ne01, ne12*ne13, 1 }     // M, batch, 1
wg_denoms  = { rm_kq, 1, 1 }             // {NUM_ROWS, 1, 1}
groupCount = { ceil(elements[0]/wg_denoms[0]), elements[1], 1 }
           = { ceil(M / NUM_ROWS),               batch,      1 }

// Falls M > maxComputeWorkGroupCount[0]: groupCount.z = 64, groupCount.x = ceil(M/(NUM_ROWS*64))
```

**Smoke-Test-Konkretisierung:**
- M=2, NUM_ROWS=1: `groupCount = (2, 1, 1)` → 2 Workgroups, jede 32 Threads.
- Insgesamt 2 × 32 = 64 Threads, was auf RDNA4 genau **1 Wavefront** ist.

### 2.5 Threading-Modell innerhalb einer Workgroup

`mul_mat_vec_q4_k.comp:87-119` (`compute_outputs`):

- Pro Block werden **16 Threads** verwendet (`itid = tid % 16`, `it_size = BLOCK_SIZE/16`).
- Bei BLOCK_SIZE=32: `it_size=2` → 2 parallele Block-Streams.
- Bei BLOCK_SIZE=64 (RDNA4 SUBGROUP): `it_size=4` → 4 Block-Streams.
- Jeder Thread bearbeitet `n=4` Q4_K-Nibbles → 16 Werte pro Thread, 16 Threads × 16 = **256 = QUANT_K** Elemente pro Block (passt).
- Akkumulation in `temp[NUM_COLS][NUM_ROWS]` (Thread-lokal), Reduktion über Shared Memory oder Subgroup-Add in `reduce_result`.

---

## 3. Dependencies

### 3.1 Include-Kette

```
mul_mat_vec_q4_k.comp
  └─ #include "mul_mat_vec_base.glsl"
      └─ #include "mul_mat_vec_iface.glsl"
          └─ #include "types.glsl"
```

**Vier Dateien total.** Keine weiteren transitive Includes für den f32_f32-Pfad. `dequant_funcs.glsl` und `dequant_head.glsl` werden **NICHT** benötigt — der Q4_K-MMV inlined die Dequantisierung selbst (Lines 19-65).

### 3.2 Required GLSL-Extensions

| Extension                                              | Quelle                          | Zweck                                |
|--------------------------------------------------------|---------------------------------|--------------------------------------|
| `GL_EXT_shader_explicit_arithmetic_types_int32`        | `mul_mat_vec_q4_k.comp:3`       | uint32_t / unpack8                   |
| `GL_EXT_shader_explicit_arithmetic_types_int64/16/8`   | `types.glsl:4-7`                | int64_t / uint8_t etc.               |
| `GL_EXT_shader_16bit_storage`                          | `types.glsl:8`, `base:2`        | f16vec2 in SSBOs                     |
| `GL_EXT_shader_8bit_storage`                           | `mul_mat_vec_base.glsl:3`       | uint8_t in SSBOs (scales[12])        |
| `GL_EXT_control_flow_attributes`                       | `mul_mat_vec_base.glsl:1`       | `[[unroll]]`                         |
| `GL_KHR_shader_subgroup_basic` + `_arithmetic`         | conditional `USE_SUBGROUP_ADD*` | optional; nicht für Smoke-Test       |

→ Vulkan 1.2 Target. shaderc muss mit `EnvVersion::Vulkan_1_2` kompilieren.

**RDNA4-Anforderungen (Phase-1-relevant):**
- 16-bit & 8-bit storage: VkPhysicalDevice16BitStorageFeatures / VkPhysicalDevice8BitStorageFeatures müssen aktiviert werden in `VkDeviceCreateInfo.pNext`. → **TODO Schritt 1.2**.
- shaderInt8 / shaderInt16 / shaderFloat16 PhysicalDeviceFeatures.

### 3.3 Required `-D`-Defines (Q4_K f32_f32, Smoke-Test)

Aus `vulkan-shaders-gen.cpp:688, 695`:

```
DATA_A_Q4_K=1
B_TYPE=float
B_TYPEV2=vec2
B_TYPEV4=vec4
D_TYPE=float
FLOAT_TYPE=float
FLOAT_TYPEV2=vec2
```

**Nicht setzen für Smoke-Test:**
- `MUL_MAT_ID` (Multi-Expert MoE)
- `USE_SUBGROUP_ADD`, `USE_SUBGROUP_ADD_NO_SHMEM` (optional)
- `LOAD_VEC_A`, `LOAD_VEC_B`, `ALIGNED` (für Matmul, nicht MMV)
- `FLOAT16` (FP16-Akkumulation)

### 3.4 glslc-Aufruf-Vorlage

```
glslc --target-env=vulkan1.2 -fshader-stage=compute -O \
  -DDATA_A_Q4_K=1 -DB_TYPE=float -DB_TYPEV2=vec2 -DB_TYPEV4=vec4 \
  -DD_TYPE=float -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 \
  vk_shaders/mul_mat_vec_q4_k.comp -o mul_mat_vec_q4_k_f32_f32.spv
```

Mit shaderc-Rust API → `CompileOptions::add_macro_definition(name, Some(value))` für jeden `-D`.

---

## 4. Q4_K-Datenformat

### 4.1 Block-Struktur (`types.glsl:301-322`)

```glsl
#define QUANT_K_Q4_K 256

struct block_q4_K {
    f16vec2 dm;                              //  4 Bytes  (d, dmin als f16)
    uint8_t scales[3*QUANT_K_Q4_K/64];       // 12 Bytes  = 3*256/64 = 12
    uint8_t qs[QUANT_K_Q4_K/2];              //128 Bytes  = 256/2    (4 bit pro Element)
};                                           //=144 Bytes pro Block (256 Elemente)

struct block_q4_K_packed16 { f16vec2 dm; uint16_t scales[6]; uint16_t qs[64]; };
struct block_q4_K_packed32 { f16vec2 dm; uint32_t scales[3]; uint32_t qs[32]; };
```

**Identisch zu `block_q4_K` aus `ggml-common.h`** in llama.cpp (144 Bytes, fp16 d/dmin, 12 packed scales+mins, 128 nibble-Bytes).

### 4.2 Byte-Layout (Offset → Bedeutung)

| Offset (Bytes) | Größe   | Inhalt                                                              |
|----------------|---------|---------------------------------------------------------------------|
| 0              | 2       | `d` (float16) — globaler Scale für Sub-Block-Scales                 |
| 2              | 2       | `dmin` (float16) — globaler Scale für Sub-Block-Mins                |
| 4              | 12      | `scales[12]` — 8 sub-block scales + 8 sub-block mins, bit-packed   |
| 16             | 128     | `qs[128]` — 256 × 4-bit Nibbles, je 2 Werte pro Byte               |
| **= 144**      | (total) |                                                                     |

### 4.3 Sub-Block-Schema

Q4_K teilt einen 256-Element-Superblock in **8 Sub-Blocks à 32 Elemente**. Jeder Sub-Block hat:
- 6-bit `scale_l` und 6-bit `min_l` für Sub-Blocks 0–3 (in den unteren 6 Bits jedes scales-Bytes 0–7),
- 6-bit `scale_h` und 6-bit `min_h` für Sub-Blocks 4–7, bit-packed in den oberen 2 Bits von scales[0–7] sowie in scales[8–11].

Dequant-Formel pro Element `e` in Sub-Block `s`:
```
weight = d * scale[s] * nibble - dmin * min[s]
```

### 4.4 Nibble-Packing in `qs[128]`

Aus `mul_mat_vec_q4_k.comp:37-65`:
- Niedrige 4 Bits jedes Bytes → Sub-Blocks 0–3 (Indizes 0..127)
- Hohe 4 Bits → Sub-Blocks 4–7 (Indizes 128..255)
- `data_a_packed32.qs[i]` liefert 4 Nibbles parallel (Half-Word-Zugriff).

### 4.5 Validierung gegen llama.cpp

`ggml-common.h` definiert `block_q4_K` mit identischem 144-Byte-Layout. `f16vec2` (GLSL) ↔ `ggml_fp16_t d, dmin` (C). `uint8_t scales[12]` ↔ `uint8_t scales[K_SCALE_SIZE=12]`. `uint8_t qs[128]` ↔ `uint8_t qs[QK_K/2 = 128]`. **Bit-Identisch.**

→ **VulkanForge kann GGUF-Q4_K-Tensoren 1:1 als `[u8; 144]`-Stream in den GPU-Buffer kopieren**, ohne Re-Pack.

---

## 5. Smoke-Test Konfiguration (für Schritt 1.3+)

| Parameter           | Wert                          | Begründung                                          |
|---------------------|-------------------------------|-----------------------------------------------------|
| Variante            | `mul_mat_vec_q4_k_f32_f32`    | Einfachste, kein Subgroup, kein FP16                |
| BLOCK_SIZE          | 32                            | Specialization-Const 0 = default; HW-agnostisch     |
| NUM_ROWS            | 1                             | Specialization-Const 1 = default                    |
| NUM_COLS            | 1                             | Specialization-Const 2 = default                    |
| M (= ne01)          | 2                             | 2 Output-Zeilen, beide validierbar                  |
| K (= ne00 = ne10)   | 256                           | 1 × QUANT_K_Q4_K                                    |
| Batch               | 1                             | Kein Batch                                          |
| Weight-Buffer       | 2 × 144 Bytes = 288 Bytes     | 2 Q4_K-Blöcke (1 pro Output-Zeile)                  |
| Input-Buffer        | 256 × 4 Bytes = 1024 Bytes    | f32-Vektor                                          |
| Output-Buffer       | 2 × 4 Bytes = 8 Bytes         | f32                                                 |
| Fuse0/1-Buffer      | 16 Bytes Dummy                | Pflicht-Binding, ungelesen bei fusion_flags=0       |
| Push-Constants      | 52 Bytes (s. 2.1)             |                                                     |
| Dispatch            | (2, 1, 1)                     | groupCount.x = ceil(M/NUM_ROWS) = 2                 |

**Erwartung:**
output[0] = Σ_{e=0..255} dequant_q4_k(weights[0], e) × input[e]
output[1] = Σ_{e=0..255} dequant_q4_k(weights[1], e) × input[e]

CPU-Referenz wird in Schritt 1.3 implementiert, vergleicht gegen den GPU-Output mit `max_abs_error < 1e-2`.

---

## 6. Design-Entscheidungen (für Phase 1)

1. **Variante `f32_f32` als Smoke-Test** statt `f16_f32`: vermeidet `VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16` als zusätzliche Anforderung in Schritt 1.2. Reicht für den Korrektheits-Beweis.
2. **Defaults für Specialization-Constants** (BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1) statt RDNA4-tuned (64/2/1): einfacher Dispatch, keine Subgroup-Annahme, vergleichbarer Output. Optimierung in Schritt 1.5.
3. **Fuse0/1 mit Dummy-Buffer binden** statt das Shader-Source zu modifizieren: Shader bleibt 1:1 llama.cpp, Bindings sind Vulkan-Validation-konform.
4. **shaderc als Build-Dep** (nicht externes glslangValidator): build.rs-integriert, reproduzierbar, plattformübergreifend.
5. **SPIR-V Embedding via `include_bytes!`** statt Runtime-Load: PoC läuft self-contained, kein Search-Path-Bug.

---

## 7. Bekannte Limitierungen / Offene Punkte

- **Multi-Aliased Bindings (Binding 0 dreifach):** Vulkan-konform, aber wir müssen sicherstellen dass shaderc/SPIR-V die drei Decorations (`block_q4_K`, `_packed16`, `_packed32`) korrekt emittiert. Bei `glslc -O` werden ungenutzte Aliase üblicherweise gepruned — wir müssen prüfen welche tatsächlich im finalen SPIR-V referenziert werden. **TODO Schritt 1.1 / 1.2:** SPIR-V mit `spirv-dis` inspizieren.
- **8-bit / 16-bit Storage Features:** Müssen explizit in `VkDeviceCreateInfo.pNext` aktiviert werden, sonst schlägt Pipeline-Erstellung fehl. Aktuelles `device.rs` aktiviert sie nicht. → **TODO Schritt 1.2**.
- **Dummy-Buffer für Fuse0/1:** 16 Bytes minimum (Vulkan SSBO-Min-Range). Sollten `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` und im Validation-Layer zugriffsfrei bleiben.
- **Subgroup-Size:** Smoke-Test umgeht Wave64-Spezifika durch BLOCK_SIZE=32 + kein Subgroup-Pfad. Aber der `gl_NumSubgroups`-Code in `reduce_result` Hybrid-Pfad ist trotzdem im Shader — bei BLOCK_SIZE=32 und Subgroup-Size=64 wird `gl_NumSubgroups = 1`, die Schleife läuft 1×, korrekt.
- **`stride_d` als Out-of-Bounds-Cut:** Der Shader nutzt `stride_d` als Schranke (`first_row + NUM_ROWS <= p.stride_d`). Über-Dispatch (mehr Workgroups als Zeilen) ist daher safe.
- **`p.ncols` Alignment:** `ncols` MUSS Vielfaches von `QUANT_K_Q4_K = 256` sein (`num_blocks_per_row = p.ncols / QUANT_K`). Gilt in unserem Smoke-Test (256/256=1) und in realen Modellen (Qwen3-8B hidden=3584=14·256).
- **Push-Constant-Größe:** 52 Bytes < 128 Byte Vulkan-Mindestlimit → safe auf jeder Vulkan-1.2-GPU.

---

## 8. Test-Ergebnisse (Schritt 1.0 = Analyse-only)

| Check                                                               | Ergebnis |
|---------------------------------------------------------------------|----------|
| Q4_K GEMV-Shader gefunden                                           | ✅ `mul_mat_vec_q4_k.comp` |
| Push-Constants identifiziert                                        | ✅ 52 Bytes, 13 felder |
| Descriptor-Set-Layout identifiziert                                 | ✅ 5 Bindings |
| Specialization-Constants identifiziert                              | ✅ 3 (BLOCK_SIZE/NUM_ROWS/NUM_COLS) |
| Dispatch-Berechnung dokumentiert                                    | ✅ ceil(M/NUM_ROWS) × batch × 1 |
| Q4_K-Block-Layout dokumentiert                                      | ✅ 144 Bytes, identisch llama.cpp |
| Generator-Status geklärt                                            | ✅ glslc-Wrapper, kein Code-Gen |
| Required `-D`-Defines extrahiert                                    | ✅ 7 Defines für f32_f32 |
| Required GLSL-Extensions identifiziert                              | ✅ 6 Extensions |
| Include-Kette aufgelöst                                             | ✅ 4 Dateien |
| **STOP-Bedingung "komplexer Generator" getroffen?**                 | **❌ Nein** |

---

## 9. Nächster Schritt

**Schritt 1.1** — Shader-Dateien (`mul_mat_vec_q4_k.comp` + 3 Includes) nach `vk_shaders/` kopieren, `build.rs` mit shaderc-Integration und den 7 `-D`-Defines erstellen, SPIR-V-Bytecode via `include_bytes!` einbetten.

**Wartet auf User-Bestätigung gemäß Regel 0.**
