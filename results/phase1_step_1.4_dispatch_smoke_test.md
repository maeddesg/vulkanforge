# Phase 1 — Schritt 1.4: Dispatch + Smoke-Test

**Datum:** 2026-04-25
**Schritt:** 1.4 (DescriptorPool/Set, vkCmdDispatch, Readback, Parity vs CPU-Referenz)
**Commit-Hash:** *(siehe `git log -1` nach diesem Commit)*

---

## Zusammenfassung

**🎯 Phase-1-PoC-Kern-Erfolg:** Der erste llama.cpp-GLSL-Shader läuft via ash in Rust und produziert **bit-exakte Ergebnisse** auf der RX 9070 XT.

| Größe                | GPU              | CPU-Referenz     | abs. Fehler |
|----------------------|------------------|------------------|-------------|
| `output[0]`          | 256.000000       | 256.000000       | **0.0**     |
| `output[1]`          | 512.000000       | 512.000000       | **0.0**     |

- `max_abs_err = 0`, `max_rel_err = 0` — Q4_K kodiert die synthetischen Werte exakt, der GPU-Pfad reproduziert sie ohne FP-Drift.
- Kernel-Zeit: **6.36 µs** (636 Ticks × 10.0 ns/Tick laut `timestampPeriod`).
- **0 Validation-WARN, 0 Validation-ERROR** über kompletten Lifecycle.

GATE: ✅ erfüllt. Alle Smoke-Test-Checks bestanden.

---

## 1. Was hinzugekommen ist

`pipeline.rs` bekommt `MatVecPushConstants` (`#[repr(C)]`, `bytemuck::Pod`, exakt 52 Bytes mit Compile-Time-Assertion):
```rust
#[repr(C)]
#[derive(..., Pod, Zeroable)]
pub struct MatVecPushConstants {
    pub ncols, stride_a, stride_b, stride_d,
    batch_stride_a, batch_stride_b, batch_stride_d,
    fusion_flags, base_work_group_y,
    ne02, ne12, broadcast2, broadcast3: u32,  // 13 × u32
}
const _: () = assert!(size_of::<MatVecPushConstants>() == PUSH_CONSTANT_BYTES as usize);
```

`main.rs` bekommt die Dispatch-Pipeline:
1. `VkDescriptorPool` (max_sets=1, STORAGE_BUFFER × 5)
2. `VkDescriptorSet` aus Pool, mit `kernel.descriptor_set_layout`
3. `vkUpdateDescriptorSets` mit 5 `VkDescriptorBufferInfo` (Bindings 0–4)
4. `VkQueryPool` (TIMESTAMP, 2 Queries)
5. Recording in `cmd_ctx.one_shot`:
   - `cmd_reset_query_pool`
   - Pre-Barrier: `TRANSFER_WRITE → SHADER_READ` auf weights+input
   - `cmd_write_timestamp(TOP_OF_PIPE, 0)`
   - `cmd_bind_pipeline(COMPUTE)`, `cmd_bind_descriptor_sets`, `cmd_push_constants`
   - `cmd_dispatch(2, 1, 1)`
   - `cmd_write_timestamp(BOTTOM_OF_PIPE, 1)`
   - Post-Barrier: `SHADER_WRITE → HOST_READ` auf output
6. `get_query_pool_results(WAIT)` → Kernel-Zeit
7. `output_buf.read_bytes()` → 2 × `f32::from_le_bytes`
8. Smoke-Checks (all-zeros / NaN / abs / rel) + CPU-Vergleich

---

## 2. Konkrete Push-Constant-Werte (für Smoke-Test)

Aus Step 1.0 §2.1 mit M=2, K=256, batch=1:

| Feld              | Wert | Begründung                                     |
|-------------------|-----:|------------------------------------------------|
| `ncols`           |  256 | K = 1 × QUANT_K                                |
| `stride_a`        |  256 | = ne10                                         |
| `stride_b`        |  256 | = ne10                                         |
| `stride_d`        |    2 | = ne01 (Out-of-bounds-Cut für Workgroups)      |
| `batch_stride_a`  |  512 | = ne00 × ne01 = 256 × 2 (irrelevant bei batch=1) |
| `batch_stride_b`  |  256 | = ne10 × ne11                                  |
| `batch_stride_d`  |    2 | = ne20 × ne21                                  |
| `fusion_flags`    |    0 | keine Bias/Scale-Fusion                        |
| `base_work_group_y` | 0  | erster Batch-Slice                            |
| `ne02`            |    1 |                                                |
| `ne12`            |    1 |                                                |
| `broadcast2`      |    1 |                                                |
| `broadcast3`      |    1 |                                                |

**Dispatch-Dimensionen:** `(ceil(M / NUM_ROWS), batch, 1) = (ceil(2/1), 1, 1) = (2, 1, 1)`. Mit `BLOCK_SIZE=32` als local_size.x ergibt das **2 Workgroups × 32 Threads = 64 Threads insgesamt** = 1 RDNA4-Wavefront.

---

## 3. Sync-Strategie

Zwei Submits auf der gleichen Compute-Queue: erst Staging-Upload, dann Dispatch — separiert durch `vkWaitForFences` auf der Host-Seite.

**Vulkan-Spec-Subtilität:** Submission-Order garantiert *Execution-Order* zwischen Submits, aber **nicht automatisch Memory-Visibility**. Ohne explizite Barrier können die Compute-Stages der zweiten Submission die TRANSFER-Stage-Writes der ersten als noch nicht sichtbar sehen — und Validation Layer's Sync-Validation-Subset (wenn aktiviert) flagged das.

Lösung im Dispatch-`one_shot`:
- **Pre-Barrier** am Anfang: `TRANSFER_WRITE → SHADER_READ` mit `srcStage=TRANSFER`, `dstStage=COMPUTE_SHADER`. Macht die Upload-Writes für die Dispatch sichtbar.
- **Post-Barrier** am Ende: `SHADER_WRITE → HOST_READ` mit `srcStage=COMPUTE_SHADER`, `dstStage=HOST`. Macht die Output-Writes für den `read_bytes()`-Mapped-Pointer sichtbar.

Beides nicht nur defensiv: ohne den Post-Barrier auf `HOST_READ` ist der Mapped-Pointer-Read auch auf coherent Memory technisch UB per Spec.

---

## 4. Run-Output

```
VulkanForge v0.1.0
WARNING: radv is not a conformant Vulkan implementation, testing use only.
VulkanForge: GPU = AMD Radeon RX 9070 XT (RADV GFX1201)
VulkanForge: Compute Queue Family = 0
VulkanForge: validation layer ACTIVE
✅ Vulkan device initialized
  Timestamp: 64 valid bits, 10.000 ns/tick
✅ gpu_allocator initialized
✅ Q4_K GEMV pipeline (BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1)
✅ Storage buffers: weights=288 B, input=1024 B, output=8 B, fuse0=16 B, fuse1=16 B
✅ Staging upload complete
  CPU reference: [256.000000, 512.000000]
✅ Descriptor set updated (5 storage-buffer bindings)
✅ Dispatch executed (2, 1, 1)
  Kernel time: 6.360 µs (636 ticks × 10.000 ns)
  GPU output: [256.000000, 512.000000]
  CPU ref   : [256.000000, 512.000000]
  max_abs_err = 0.000000e0, max_rel_err = 0.000000e0
✅ Smoke test PASSED — GPU output matches CPU reference
✅ Teardown clean
```

---

## 5. GATE-Check

| Smoke-Test-Check (per Prompt §1.4.4)              | Threshold             | Ergebnis                | Pass |
|---------------------------------------------------|-----------------------|-------------------------|------|
| Output ist NICHT all-zeros                        | —                     | [256.0, 512.0]          | ✅   |
| Output ist NICHT NaN/Inf                          | `is_finite`           | beide finite            | ✅   |
| Output-Dimensionen stimmen (M=2 Elemente)         | `len == 2`            | 2 (8 Bytes / 4 B/f32)   | ✅   |
| Output-Werte im erwarteten Bereich                | analytisch [256, 512] | exakt                   | ✅   |
| `max_abs_error < 1e-2`                            | 1e-2                  | **0.0**                 | ✅   |
| `max_rel_error < 5%`                              | 5%                    | **0.0**                 | ✅   |
| Kernel-Zeit gemessen                              | µs-Auflösung          | 6.360 µs                | ✅   |
| Validation-WARN/ERROR Count                       | 0                     | **0**                   | ✅   |

**Erste 2 Output-Elemente** (Prompt §Report fordert „erste 16"; wir haben nur 2 Elemente):
```
GPU: [256.000000, 512.000000]
CPU: [256.000000, 512.000000]
```

**Validation-Output:** Stille — keine WARN/ERROR. Das eine `WARNING:` aus dem Run-Output stammt vom RADV-Driver selbst (libvulkan_radeon.so → stderr) und meldet nur, dass RADV nicht konformitäts-zertifiziert ist; kein Validation-Issue.

---

## 6. Kernel-Zeit-Einordnung

Kernel-Zeit 6.36 µs bei 1 Wavefront / 64 Threads / 2 Output-Zeilen / 256 Elementen / 2 Q4_K-Blöcken ist **nicht aussagekräftig** — Setup-Cost (Pipeline-Bind, Descriptor-Bind, Wave-Launch, Cache-Warmup) dominiert gegenüber der eigentlichen Compute-Arbeit.

Effektive BW-Berechnung würde hier `~300 B / 6.36 µs ≈ 47 MB/s` ergeben — was Größenordnungen unter der Peak-BW liegt und schlicht reflektiert, dass 300 B kleiner als eine Cache-Line × 2 sind. **Performance-Aussagen kommen erst in Schritt 1.5** mit Qwen-3-realistischen Dimensionen (K=3584, M=3584, ~1.6 MB Weights pro GEMV, ~14 KB Input).

Was die 6.36 µs trotzdem zeigen:
- Timestamps werden korrekt geliefert (64 valid bits, period 10 ns).
- Der Dispatch-Pfad (Submit + Fence + Query-Read) hat keine pathologisch hohen Latenzen.
- Kein Crash, kein Timeout, kein Hang.

---

## 7. Bekannte Limitierungen / Offene Punkte

- **Pre-Barrier nutzt `TRANSFER → COMPUTE_SHADER`** — alternativ wäre `BOTTOM_OF_PIPE → COMPUTE_SHADER` defensiver, aber die exakte Stage gibt dem Treiber mehr Optimierungsspielraum. Validation-Layer akzeptiert beide.
- **Nur ein Run, keine Variabilitäts-Messung.** Step 1.5 macht 100× Dispatches und reportet Median + p95 + Stddev.
- **Smoke-Daten triggern keinen `dmin`-Pfad** (alle `sub_mins=0`, `dmin=0`). Gewollt für 1.4 (siehe 1.3-Report-§1), aber ein Mins-Bug wäre hier unsichtbar. Step 1.5 mit Random-Daten triggert beides.
- **Output-Werte sind kleine Integers (256, 512)** — keine Floating-Point-Subnormalitäten oder Akkumulations-Drift sichtbar. Step 1.5 mit großen K und Random-Inputs deckt das ab.
- **`bytemuck::Pod` derive auf `MatVecPushConstants`** funktioniert weil alle Felder u32 sind. Wenn wir in Phase 2 ein nicht-Pod-Feld einbauen (z.B. ein enum), müssen wir manuell auf `unsafe { std::slice::from_raw_parts }` ausweichen.
- **Kein Pipeline-Cache** — Pipeline-Erstellung dauert spürbar. Bei ~100 Pipelines in Phase 2 lohnt es sich, in Schritt 1.5 schon einen `VkPipelineCache` einzuführen.
- **`vk::WHOLE_SIZE` bei DescriptorBufferInfo.range:** Vulkan-Spec erlaubt das, das Validation-Layer ist zufrieden. Saubere alternative wäre die exakte Buffer-Größe — kein Funktionsunterschied auf RADV.

---

## 8. Geänderte / neue Dateien

```
src/backend/vulkan/pipeline.rs   (~+30 Zeilen: MatVecPushConstants struct + size assertion)
src/main.rs                      (~+155 Zeilen: descriptor pool/set, query pool,
                                   dispatch one_shot, smoke checks, cleanup)
results/phase1_step_1.4_dispatch_smoke_test.md  (NEU, dieser Report)
```

`Cargo.toml` und Shader-Dateien unverändert.

---

## 9. Nächster Schritt

**Schritt 1.5** — Skalierungstest:
- Realistische Dimensionen (Vorschlag: Qwen3-8B-style K=3584, M=3584; entspricht ~1× HiddenDim×HiddenDim Tensor).
- Random-Q4_K-Daten (gültiges Format, zufällige `dm`/`scales`/`mins`/`nibbles`).
- 100× Dispatch hintereinander, Median/p95/Stddev über Kernel-Zeit reporten.
- Effektive Bandbreite (GB/s) berechnen, Anteil an Peak-BW (~608 GB/s für 9070 XT).
- Toleranz für CPU-Referenz lockern (Q4_K hat bei Random-Daten inhärente Quantisierungsfehler bis ~1e-2 absolute / paar Prozent relativ).

**Wartet auf User-Bestätigung gemäß Regel 0.**
