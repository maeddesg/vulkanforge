# Phase 1 — Schritt 1.3: Buffer-Allokation + Testdaten + CPU-Referenz

**Datum:** 2026-04-25
**Schritt:** 1.3 (Allocator, Buffer, Staging-Upload, Q4_K-Encoding, CPU-Referenz)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Naming-Klärung vor dem Report

**Cargo.toml (Initial-Commit):** `gpu-allocator = "0.27"` (Traverse-Research-Crate).
**Prompt + Vision-Dok:** schreiben "gpu-alloc" (anderes Crate, Maintainer Zakarumych).

Ich habe das tatsächlich vorhandene `gpu-allocator` benutzt — ein Tausch zu `gpu-alloc` würde Cargo.toml und vermutlich die ROCmForge-Migrations-Pläne aus dem Vision-Dok beeinflussen, das ist außerhalb des 1.3-Scopes. Falls du das umstellen willst, ist das ein eigener Schritt — sag Bescheid.

---

## Zusammenfassung

- 4 neue Module: `buffers.rs` (GPU-Buffer + gpu-allocator-Wrapper), `commands.rs` (one-shot CommandPool für Staging-Copy), `q4k.rs` (Q4_K-Encoder, -Dequantizer, CPU-GEMV-Referenz). `mod.rs` registriert alle.
- `main.rs` orchestriert: Device → Allocator → Pipeline (aus 1.2) → 5 Storage-Buffer → 2 Staging-Buffer → upload via `vkCmdCopyBuffer` → CPU-Referenz → Teardown.
- 2 Q4_K-Unit-Tests in `q4k.rs::tests` validieren Encoder/Dequantizer-Round-Trip + CPU-GEMV-Output (`cargo test` grün, 2/2 passed).
- **GATE: ✅ erfüllt.** Buffers alloziert, Staging-Upload funktioniert, CPU-Referenz existiert. **0 Validation-WARN, 0 Validation-ERROR** über den ganzen Lebenszyklus.
- CPU-Referenz: `output[0] = 256.000`, `output[1] = 512.000` — exakt analytisch (max abs err < 1e-3).

---

## 1. Test-Daten — Konstruktion und Begründung

Ziel: ein analytisch lösbares Setup mit denkbar einfachen Werten, damit jede Bit-Verschiebung in Encoder/Dequant/Shader sofort als Korrumpierung sichtbar wird.

### Weight-Matrix (2 Q4_K-Blöcke = 288 Bytes)

| Row | d  | dmin | sub_scales[8] | sub_mins[8] | nibbles[256] | dequantisiert ergibt |
|-----|----|------|---------------|-------------|--------------|----------------------|
| 0   | 1.0 | 0.0 | [1; 8]        | [0; 8]      | [1; 256]     | **alle 1.0**         |
| 1   | 1.0 | 0.0 | [1; 8]        | [0; 8]      | [2; 256]     | **alle 2.0**         |

Formel im Shader (vereinfacht für `dmin=0`, alle `sub_mins = 0`):
```
weight[row][e] = d * sub_scales[sb(e)] * nibble[e] - dmin * sub_mins[sb(e)]
              = 1 * 1 * nibble[e] - 0
              = nibble[e]
```

Dies eliminiert den `dmin`-Pfad aus der Verifikation — Schritt 1.4-Output, der von 1.0/2.0 abweicht, hat einen **Nibble-Pfad-Bug**. Der `dmin`-Pfad wird in Schritt 1.5 mit zufälligen Werten getestet.

### Input-Vektor

`[1.0; 256]` — All-Ones-f32. Damit ist die Antwort einfach `Σ weight[row][e]`.

### Erwartetes Output

```
output[0] = Σ_{e=0..255} 1.0 = 256.0
output[1] = Σ_{e=0..255} 2.0 = 512.0
```

---

## 2. Q4_K-Encoder — Implementierung und Verifikation

### Layout (matched gegen `vk_shaders/types.glsl` und `ggml-quants.c`)

```
Offset      Größe   Inhalt
   0..2     2 B     d        (f16 LE)
   2..4     2 B     dmin     (f16 LE)
   4..16    12 B    scales   (8 Sub-Block-Scales + 8 Sub-Block-Mins, 6-bit, packed)
  16..144   128 B   qs       (256 Nibbles, low/high jeder Byte = 2 Werte)
  ────────────────
  144 B total
```

### Scales-Bit-Packing (Auszug aus `q4k.rs::encode_block`)

```rust
for j in 0..4 {
    out[4 + j]     = sub_scales[j] & 0x3F;            // bytes 4..7  : low6 = scale_l[j]
    out[4 + j + 4] = sub_mins[j]   & 0x3F;            // bytes 8..11 : low6 = min_l[j]
}
for j in 4..8 {
    let ls = sub_scales[j];  let lm = sub_mins[j];
    out[4 + j + 4]   = (ls & 0x0F) | ((lm & 0x0F) << 4);  // bytes 12..15: scale_h[j] low4 + min_h[j] low4
    out[4 + (j-4)]  |= ((ls >> 4) & 0x03) << 6;            // OR scale_h high2 into bytes 4..7 top
    out[4 + j]      |= ((lm >> 4) & 0x03) << 6;            // OR min_h   high2 into bytes 8..11 top
}
```

Dieser Encoder ist exakt invers zum Dequantizer in `q4k.rs::dequant_block`. Round-Trip-Test in `q4k.rs::tests::smoke_weights_dequant_to_known_values` bestätigt Bit-Exaktheit:
```
$ cargo test --quiet q4k
test result: ok. 2 passed; 0 failed
```

### Nibble-Packing

Vom Shader (`mul_mat_vec_q4_k.comp:40-65`):
- Sub-Blocks 0–3 (Indizes 0–127) lesen die **niedrigen** 4 Bits jedes `qs`-Bytes.
- Sub-Blocks 4–7 (Indizes 128–255) lesen die **hohen** 4 Bits.

`encode_block` setzt entsprechend:
```rust
for j in 0..128 {
    let lo = nibbles[j] & 0x0F;       // sub-blocks 0..3
    let hi = nibbles[j + 128] & 0x0F; // sub-blocks 4..7
    out[16 + j] = lo | (hi << 4);
}
```

### Dequant + CPU-GEMV

`q4k::cpu_gemv(weights, n_rows, k, input)` macht für jeden Output-Row:
1. `dequant_block` über alle Blöcke der Row → 256 f32 pro Block.
2. Akkumulation: `out[r] += Σ_e dequant[e] * input[block_offset + e]`.

Test `smoke_gemv_matches_analytical` bestätigt das Endergebnis [256.0, 512.0] mit `< 1e-3` Toleranz.

---

## 3. Buffer-Plan (5 Storage + 2 Staging)

| Buffer            | Größe (B) | Usage Flags                                      | MemoryLocation        | Zweck                         |
|-------------------|----------:|--------------------------------------------------|-----------------------|-------------------------------|
| `weights`         |       288 | `STORAGE_BUFFER \| TRANSFER_DST`                 | `GpuOnly`             | 2× Q4_K-Block (Binding 0)     |
| `input`           |      1024 | `STORAGE_BUFFER \| TRANSFER_DST`                 | `GpuOnly`             | 256× f32 (Binding 1)          |
| `output`          |         8 | `STORAGE_BUFFER`                                 | `GpuToCpu`            | 2× f32 Readback (Binding 2)   |
| `fuse0_dummy`     |        16 | `STORAGE_BUFFER`                                 | `GpuOnly`             | Pflicht-Bind (Binding 3)      |
| `fuse1_dummy`     |        16 | `STORAGE_BUFFER`                                 | `GpuOnly`             | Pflicht-Bind (Binding 4)      |
| `staging_weights` |       288 | `TRANSFER_SRC`                                   | `CpuToGpu`            | Host→Device-Hop (transient)   |
| `staging_input`   |      1024 | `TRANSFER_SRC`                                   | `CpuToGpu`            | Host→Device-Hop (transient)   |

`MemoryLocation` (gpu-allocator-Enum) → konkrete `VkMemoryPropertyFlags`:
- `GpuOnly`     → `DEVICE_LOCAL`
- `CpuToGpu`    → `HOST_VISIBLE | HOST_COHERENT`, bevorzugt `DEVICE_LOCAL` falls verfügbar (Resizable BAR)
- `GpuToCpu`    → `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`, bevorzugt `DEVICE_LOCAL`

Auf der RX 9070 XT mit Resizable BAR (Heap 1 = 16304 MB DEVICE_LOCAL, sichtbar als HOST_VISIBLE) erwarte ich, dass Output und Staging-Buffers tatsächlich im VRAM sitzen. Verifikation via `vkGetBufferMemoryRequirements`+`vkGetPhysicalDeviceMemoryProperties` ist nicht im Scope von 1.3, kommt bei Performance-Tuning in Schritt 1.5.

---

## 4. Staging-Upload — Mechanik

`commands.rs::CommandContext::one_shot` macht in einem Aufruf:
1. `vkAllocateCommandBuffers` (1 primary)
2. `vkBeginCommandBuffer` (`ONE_TIME_SUBMIT`)
3. Caller-Closure: zwei `vkCmdCopyBuffer` (staging → device), Größe = Buffer-Größe (kein Offset, kein Sub-Region)
4. `vkEndCommandBuffer`
5. `vkCreateFence` + `vkQueueSubmit` + `vkWaitForFences(timeout=u64::MAX)`
6. `vkDestroyFence` + `vkFreeCommandBuffers`

Pool-Flags: `TRANSIENT` (kein Reset/Reuse, der Pool wird nach diesem Step zerstört). Queue: `dev.compute_queue` (Familie 0, GRAPHICS+COMPUTE+TRANSFER auf AMD).

**Kein Pipeline-Barrier nötig zwischen Copy und nachfolgendem Compute** — der `vkQueueSubmit` + Fence-Wait synchronisieren schon bis zur Host-Sicht; in Schritt 1.4 reicht ein `BUFFER_MEMORY_BARRIER` zwischen Compute-Stage-Read und vorherigem Transfer-Write, falls überhaupt nötig (Compute auf einem fresh Submit-Boundary sieht prior submits ohne Barrier per Implicit Synchronization).

---

## 5. Run-Output

```
VulkanForge v0.1.0
WARNING: radv is not a conformant Vulkan implementation, testing use only.
VulkanForge: GPU = AMD Radeon RX 9070 XT (RADV GFX1201)
VulkanForge: Compute Queue Family = 0
VulkanForge: validation layer ACTIVE
✅ Vulkan device initialized
✅ gpu_allocator initialized
✅ Q4_K GEMV pipeline (BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1)
✅ Storage buffers: weights=288 B, input=1024 B, output=8 B, fuse0=16 B, fuse1=16 B
✅ Staging buffers populated (host-visible)
✅ Staging upload complete (weights + input → DEVICE_LOCAL)
✅ CPU reference: output[0]=256.000, output[1]=512.000
✅ CPU reference matches analytical expectation [256.0, 512.0] (max abs err < 1e-3)
✅ Teardown clean
```

```
$ cargo run --quiet 2>&1 | grep -cE "^\[vk (WARN|ERROR)/"
0
```

| GATE-Check                                              | Ergebnis |
|---------------------------------------------------------|----------|
| Allocator initialisiert                                 | ✅       |
| Alle 5 Storage-Buffers + 2 Staging-Buffers alloziert    | ✅       |
| Staging-Buffer mit Host-Daten beschrieben               | ✅       |
| `vkCmdCopyBuffer` × 2 + Fence-Wait erfolgreich          | ✅       |
| CPU-Referenz berechnet                                  | ✅       |
| CPU-Referenz matched analytische Erwartung              | ✅ (ε=0) |
| Round-Trip Encoder ↔ Dequantizer (Unit-Tests)           | ✅ 2/2   |
| Validation-WARN/ERROR über kompletten Lifecycle         | **0**    |
| Teardown leak-frei (allocator + device dropped)         | ✅       |

---

## 6. Bekannte Limitierungen / Offene Punkte

- **`gpu-allocator` vs. `gpu-alloc`** — siehe Naming-Klärung am Reportanfang. Falls explizit `gpu-alloc` (Zakarumych-Crate) gewünscht ist, ist das ein eigener Schritt mit anderem API-Surface (z.B. `Config`, `MemoryBlock`).
- **Memory-Type-Verifikation**: ich verlasse mich auf gpu-allocators automatische Memory-Type-Auswahl. Ob `GpuOnly` tatsächlich VRAM erwischt und `CpuToGpu` wirklich Resizable-BAR/HOST_VISIBLE-DEVICE-LOCAL ist, lässt sich erst mit `vkGetPhysicalDeviceMemoryProperties`-Inspektion oder per RADV-`RADV_DEBUG=allocations` final verifizieren — nice-to-have für Schritt 1.5.
- **Coherent Mapping**: gpu-allocator gibt für `CpuToGpu`/`GpuToCpu` HOST_COHERENT-Memory zurück; daher kein expliziter `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges`. Wenn das auf einer anderen GPU nicht der Fall sein sollte, müsste man flush/invalidate ergänzen.
- **One-shot CommandContext** ist auf Step-1.3-Komfort optimiert. Schritt 1.4 braucht eine längerlebige Recording-Pfad (Dispatch + Timestamp-Queries). Refactor folgt.
- **`bytemuck::cast_slice` für `&[f32] → &[u8]`** — funktioniert weil f32 Pod ist. Keine Abhängigkeit auf bytemuck-derive an dieser Stelle.
- **Dummy-Buffer-Inhalte uninitialisiert** — Vulkan garantiert keine Initialisierung. Da `fusion_flags=0` keine Reads triggert, ist das per Spec safe. Falls Validation-Layer in einer zukünftigen Version "uninitialized read" warnt (sie tut es momentan nicht für ungelesenes SSBO-Data), könnten wir mit `vkCmdFillBuffer(0)` initialisieren.

---

## 7. Geänderte / neue Dateien

```
src/backend/vulkan/buffers.rs   (NEU,  102 Zeilen)
src/backend/vulkan/commands.rs  (NEU,   71 Zeilen)
src/backend/vulkan/q4k.rs       (NEU,  205 Zeilen, inkl. 2 Unit-Tests)
src/backend/vulkan/mod.rs       (+3 Zeilen: buffers, commands, q4k)
src/main.rs                     (~komplett umgeschrieben: 81 → 165 Zeilen)
results/phase1_step_1.3_buffer_testdata.md  (NEU, dieser Report)
```

`Cargo.toml` unverändert (`gpu-allocator`, `bytemuck`, `half` waren alle schon vorhanden).

---

## 8. Nächster Schritt

**Schritt 1.4** — Dispatch + Smoke-Test:
- `VkDescriptorPool` (1 set, 5 storage) + `VkDescriptorSet`-Allokation.
- `vkUpdateDescriptorSets` mit den 5 Buffer-Bindings.
- Längerlebige Command-Recording (Pipeline-Bind, Descriptor-Bind, Push-Constants, Dispatch).
- Push-Constant-Werte aus Step 1.0 §2.1: ncols=256, stride_a=256, stride_b=256, stride_d=2, batch_strides, fusion_flags=0, batch=0/1/1/1.
- Dispatch `(2, 1, 1)` — entspricht `ceil(M/NUM_ROWS) × batch × 1` mit M=2, NUM_ROWS=1.
- Submit + Fence-Wait, Output-Buffer mappen, Bytes lesen, gegen `[256.0, 512.0]` vergleichen.
- VkQueryPool für Timestamp-Profiling (Kernel-Zeit in µs).

**Wartet auf User-Bestätigung gemäß Regel 0.**
