# v0.2 Sprint 10A — flash_attn_cm2 Deep-Dive (Pivot zu cm1)

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9d.3 (FP16 KV Default ON, 167/167 Tests)

---

## TL;DR — cm2 ist NV-only, NICHT portierbar. cm1 ist die echte Referenz, und sie ist komplexer als gedacht.

```
═══ v0.2 Sprint 10A ═══

KRITISCHER BEFUND (vor jeder weiteren Arbeit):

  flash_attn_cm2.comp Zeile 13:  #extension GL_NV_cooperative_matrix2 : enable

  cm2 nutzt EXTENSIVE NVIDIA-spezifische Features:
    • coopMatLoadTensorNV     — Tensor-Descriptor-basierte Loads
    • coopMatReduceNV         — Row/Col Reduction über coopmat
    • coopMatPerElementNV     — Per-Element-Funktion auf coopmat
    • tensorLayoutNV<>        — typed tensor descriptors
    • tensorViewNV<>          — Transponierte Views
    • gl_ScopeWorkgroup       (NV-erweitert; KHR primär gl_ScopeSubgroup)

  RDNA4/RADV unterstützt NUR VK_KHR_cooperative_matrix rev2.
  → vulkaninfo bestätigt: "VK_KHR_cooperative_matrix : extension revision 2"
  → KEINE Listung von VK_NV_cooperative_matrix2

  KONSEQUENZ: flash_attn_cm2.comp KANN NICHT auf unserer Hardware laufen.
              Der Sprint-Brief ging von einer falschen Annahme aus.

PIVOT: flash_attn_cm1.comp ist die KHR-portable Variante (642 LOC,
       deutlich komplexer als cm2's 390 LOC).
  • Verwendet nur GL_KHR_cooperative_matrix
  • Echte coopmat-Attention auf RDNA4 möglich
  • ABER: viel komplexerer Code wegen fehlender NV-Helpers

DEEP-DIVE-ERGEBNISSE — wie cm1 die schwierigen Probleme löst:

  Frage 1 (Online-Softmax + coopmat):
    NICHT via direkter Fragment-Manipulation. Stattdessen:
    1. coopMatStore(Score, sfsh) — Score-Fragment → LDS
    2. barrier()
    3. Per-Thread Scalar-Code: row-max via subgroupMax über sfsh
    4. exp(s - max) → Psh in LDS (per-thread)
    5. coopMatLoad(P_frag, Psh) — P aus LDS zurück in Fragment
    → 2× LDS-Roundtrip pro K-Tile (Score-Store + P-Load)
    → KEIN direkter Fragment-Element-Zugriff (nicht-portabel)

  Frage 2 (Causal Mask in coopmat):
    cm1 hat KEINEN Causal-Mask-Code. Es liest stattdessen eine
    PRECOMPUTED Mask-Matrix aus Buffer-Binding 3 (`data_m`):
    1. Host füllt mask[Br × KV] mit 0 / -inf vor jedem Forward.
    2. Shader lädt mask-tile in LDS, prüft ob ALLE -inf (skip block).
    3. Falls nicht: addiert `slope * mask` zu sfsh nach QK-MulAdd.
    → Causal-Mask ist EXTERNE Vorberechnung, kein Shader-Logik.

  O-Akkumulator:
    NICHT in coopmat. Held als per-Thread-Register-Array
    `f16vec4 Of[rows_per_thread][d_per_thread]`.
    → Online-Softmax-Rescale ist trivial: `Of[r][d] *= float16_t(eMf[r])`
    → PV-Result wird über LDS staged (`pvsh`) und in Of akkumuliert.

  LDS-Footprint (HSK=128, Br=16, Bc=??): ~33-35 KB
    Qf:   ~4.5 KB (Q-tile, einmal pro WG)
    Psh:  ~3 KB (P-staging für PV)
    sfsh: ~6 KB (Score-staging)
    kvsh: ~17 KB (K oder V, alternierend)
    pvsh: ~2 KB (PV-result-staging)
    slope/tmpsh: ~100 B
    → Innerhalb 64 KB-Budget, aber Tight für Occupancy.

  Multi-Subgroup-WG: cm1 nutzt `row_split` Subgroups per WG für
    Parallelität über die Br Q-Rows. Bei row_split=4 (Wave64): jede
    Subgroup behandelt 4 Q-Rows.

ENTSCHEIDUNG: Eigenbau. Aufbauen auf unserem flash_attn_tiled.comp,
              das LDS-Roundtrip-Pattern von cm1 übernehmen.
  Begründung:
    - cm1 ist 642 LOC, zu komplex für 1:1-Port (D_split, row_split,
      ALiBi, sinks, split-K, mask-matrix, soft-cap)
    - Unser flash_attn_tiled.comp hat bereits gute Br/Bc-Topologie
      und FP16-KV-Read via unpackHalf2x16 (Sprint 9d.2)
    - Wir brauchen NUR die zwei coopmat-Stellen: QK-MulAdd + PV-MulAdd
    - Der Rest (online-softmax, causal mask, output write) bleibt
      aus tiled.comp — das ist getestet und funktioniert
  Geschätzter Aufwand: ~300-400 LOC neuer Shader-Code.

GATE FÜR PHASE 10B: GO mit Scope-Down.
  Sprint 10B = Eigenbau-Coopmat-Attention für Br=16, Bc=16 (single-
  subgroup WG, kein row_split, kein D_split, kein mask-buffer).

Tests: 167/167 unverändert (kein Code in 10A).
Files: nur dieser Report.
Commit: HEAD (kein Push).
```

---

## 1. Der NV-Extension-Blocker bei cm2

### 1.1 cm2's Extension-Liste (Zeilen 1-17)

```glsl
#version 450
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_NV_cooperative_matrix2 : enable     ← KILLER
#extension GL_EXT_buffer_reference : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_EXT_null_initializer : enable
```

### 1.2 NV-spezifische Features die cm2 stark nutzt

| Feature | Wo in cm2 | KHR-Äquivalent? |
|---------|-----------|------------------|
| `coopMatLoadTensorNV(M, ptr, off, sliceTensorLayoutNV(...))` | Zeile 137, 203, 233, 294 | NEIN — nur basic `coopMatLoad(M, ptr, off, stride, layout)` |
| `coopMatReduceNV(out, in, gl_CooperativeMatrixReduceRowNV, fn)` | Zeile 205, 220, 260, 304, 334, 343 | NEIN — muss via LDS + subgroupAdd |
| `coopMatPerElementNV(out, in, fn, args...)` | Zeile 160, 255, 269-271, 281, 318, 321-322, 325-326, 338, 380 | NEIN — muss via LDS-Roundtrip |
| `tensorLayoutNV<dims, ClampMode>` mit `setTensorLayoutDimensionNV` etc. | Zeile 104-117 | NEIN — basic stride/layout in coopMatLoad |
| `tensorViewNV<>` mit Transpose | Zeile 108, 386 | NEIN — Transpose via Load mit ColMajor |
| `coopmat<T, gl_ScopeWorkgroup, ...>` | überall | KHR primär gl_ScopeSubgroup; gl_ScopeWorkgroup NICHT garantiert |

Diese Features machen cm2 elegant: die ganze Online-Softmax-Logik
(max → exp → mul) läuft direkt auf den coopmat-Fragmenten via
`coopMatPerElementNV` und `coopMatReduceNV`. Diese Features fehlen
auf RDNA4. Damit ist cm2 keine Option.

### 1.3 RDNA4-Capabilities (vulkaninfo)

```
$ vulkaninfo | grep -i cooperative
	VK_KHR_cooperative_matrix                     : extension revision 2
```

Nur KHR. Keine NV-Variante.

---

## 2. flash_attn_cm1 — die KHR-portable Variante

### 2.1 Größenvergleich

```
flash_attn_cm2.comp  →  390 LOC  (NV-only, nicht portabel)
flash_attn_cm1.comp  →  642 LOC  (KHR-only, portabel; +65% LOC weil
                                  NV-Helpers manuell nachgebaut werden)
flash_attn_base.glsl →  367 LOC  (geteilte Push-Constants + Helpers)
```

cm1 ist 1.65× größer als cm2 — das ist der "Cost" für NV→KHR-Translation.

### 2.2 cm1 Extension-Liste

```glsl
#version 450
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable    ← KHR only
```

Alle Extensions auf RDNA4 verfügbar. Portabel.

### 2.3 Coopmat-Geometrie

```glsl
const uint32_t MatBr = 16;        // coopmat-tile rows
const uint32_t MatBc = 16;        // coopmat-tile cols
// (Br, Bc) sind variable Spec-Constants, Vielfache von Mat-Größen
```

Coopmat-Typen die cm1 nutzt (alles `gl_ScopeSubgroup` — wave64-skaliert):

```glsl
// QK-Matmul:
coopmat<float16_t, gl_ScopeSubgroup, MatBc, 16, gl_MatrixUseA> KMat;
coopmat<float16_t, gl_ScopeSubgroup, 16, MatBr, gl_MatrixUseB> QMat;
coopmat<ACC_TYPE,  gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator> SfMat;
// Hinweis: Score ist [MatBc, MatBr] — TRANSPONIERT!
//          (cm1 berechnet K * Q^T statt Q * K^T, "to allow N=8 if needed")

// PV-Matmul:
coopmat<float16_t, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseA>           KMat;  // = P
coopmat<float16_t, gl_ScopeSubgroup, ?, ?, gl_MatrixUseB>                   QMat;  // = V
coopmat<float16_t, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator> PVMat;
```

KHR garantiert `(M, N, K) = (16, 16, 16)` für FP16×FP16→FP32 auf
gfx1201 (per Sprint 2A bench). Andere Sizes optional, vendor-spezifisch.

---

## 3. Die ZENTRALE Antwort: wie cm1 Online-Softmax mit coopmat macht

### 3.1 Das Pattern: LDS-Roundtrip

cm1 macht NICHT, was cm2 mit `coopMatPerElementNV` direkt auf
Fragment-Elementen macht. Stattdessen:

```
QK Phase:
  coopmat<...> SfMat = MulAdd(KMat, QMat, ...);
  
  ↓ Fragment → LDS:
  
  coopMatStore(SfMat, sfsh, coord, sfshstride, RowMajor);
  barrier();
  
  ↓ Scalar code (per-thread, row-organized):
  
  for (each row r assigned to this thread):
    rowmaxf = -∞;
    for (each col c):
      rowmaxf = max(rowmaxf, sfsh[r, c]);
    rowmaxf = subgroupMax(rowmaxf);    // cross-thread row reduction
    Mf[r] = max(rowmaxf, Mold);
    eMf[r] = exp(Mold - Mf[r]);
    Of[r] *= eMf[r];                    // O accumulator rescale
    for (each col c):
      Pf = exp(sfsh[r, c] - Mf[r]);
      Lf[r] += Pf;
      Psh[r, c] = Pf;                    // store P for PV matmul
  barrier();
  
  ↓ LDS → Fragment für PV:
  
  coopMatLoad(KMat, Psh, ..., ColMajor);  // KMat = P-fragment
  coopmat<...> PVMat = MulAdd(KMat, V_frag, PVMat);
  
  ↓ Fragment → LDS für Output:
  
  coopMatStore(PVMat, pvsh, ...);
  barrier();
  
  ↓ Per-thread accumulate to Of register array:
  
  for (each output row r and dim d in this thread):
    Of[r][d] += pvsh[r, d];
```

### 3.2 Der Trick: O ist NICHT in coopmat

```glsl
// Per-thread Register Array (FP16 vec4):
f16vec4 Of[rows_per_thread][d_per_thread];
```

Damit ist Online-Softmax-Rescale (`O *= exp(M_old - M)`) trivial:
direkte Multiplikation auf Register-Array. Kein
`coopMatPerElementNV` nötig.

Der Trade-off: Output muss aus dem PV-Coopmat-Fragment HERAUSGENOMMEN
und in die Register-Akkumulatoren GESCHRIEBEN werden — das ist der
zweite LDS-Roundtrip (`pvsh` staging). Aber: kein Coopmat-Fragment-
Element-Zugriff nötig, alles funktioniert mit nur den 3 KHR-Primitiven
(coopMatLoad, coopMatStore, coopMatMulAdd).

### 3.3 Code-Stellen in cm1

| Operation | Zeilen |
|-----------|--------|
| QK Tile-Loop | 253-308 |
| Score → LDS (sfsh) | 310-312 |
| Mask + Soft-Cap auf sfsh | 314-339 |
| Row-Max + Online-Softmax-Update + O-Rescale | 341-372 |
| P → LDS (Psh) + Lf-Akkumulation | 374-391 |
| V-staging | 393-414 |
| PV-Loop (über num_hsv_tiles) | 419-496 |
| PV-Result → Of (register accumulator) | 506-519 |
| Final-Norm + Write-Back | 525-641 |

---

## 4. Causal-Mask: cm1 hat keine Inline-Logik

### 4.1 Was cm1 stattdessen macht

cm1 erwartet eine **vorberechnete Mask-Matrix `data_m`** (binding 3,
FP16, shape `[seq_len × KV]`). Der Host ist verantwortlich, diese
Matrix VOR dem Forward zu füllen mit:

* `mask[q, k] = 0` falls Position k ≤ Position q (kausal erlaubt)
* `mask[q, k] = -∞` (sentinel `0xfc00` als FP16) falls k > q (verboten)

Im Shader:
1. Lade `mask`-Tile in `mask_cache` (Zeile 170-220)
2. Berechne `max_mask` über das Tile, skip wenn alle -∞ (Zeile 207-219)
3. Nach QK-Matmul: `sfsh += slope * mask_cache` (Zeile 325-339)

### 4.2 Konsequenz für unseren Port

Wir müssten entweder:

**Option A (cm1-Stil):** Vor jedem prefill_batch eine
mask-Matrix berechnen + uploaden. Mehr CPU-Arbeit + extra GPU-Buffer.

**Option B (unser Stil, einfacher):** Unsere bestehende inline
Causal-Mask in `flash_attn_tiled.comp` (per-Element-Check
`t_global <= q_pos_qi`) bleibt erhalten. Die Mask wird auf den
LDS-staging Scores angewendet, BEVOR der Row-Max berechnet wird.

```glsl
// Pseudo (Option B):
coopMatStore(SfMat, sfsh, ...);
barrier();

// Per-thread scalar mask + softmax pipeline:
for (each row r assigned to thread):
    for (each col c):
        bool valid = (j*Bc + c) <= (q_start + i*Br + r);
        float s = valid ? sfsh[r, c] : -∞;
        rowmaxf = max(rowmaxf, s);
    ...
```

Option B ist weniger Arbeit und matched unser Pattern. Empfehlung:
**Option B verwenden.**

---

## 5. LDS-Layout-Vergleich

### 5.1 cm1 (HSK=128, ?Br?, ?Bc?)

```
  Qf      = Br * (HSK/4 + 2) * 8 B  (f16vec4)
  sfsh    = Bc * (Br/4 + 2) * 16 B  (ACC_TYPEV4 FP32)
  Psh     = Bc * (Br/4 + 2) * 8 B   (f16vec4)
  kvsh    = Bc * max(stride) * 8 B  (f16vec4, K oder V alternierend)
  pvsh    = MatBc * row_split*4 * 8 B (f16vec4)
  slope   = Br * 4 B
  tmpsh   = row_split * 4 B
```

Konkret bei Br=16, Bc=64, HSK=128, row_split=4 (Wave64 RDNA4):
* Qf:   16 × 34 × 8 = **4 352 B**
* sfsh: 64 × 6 × 16 = **6 144 B** (kein Padding-Trick bei HSK=128, anderer stride)
* Psh:  64 × 6 × 8 = **3 072 B**
* kvsh: 64 × 34 × 8 = **17 408 B**
* pvsh: 16 × 16 × 8 = **2 048 B**
* slope/tmpsh: ~80 B
* **Total: ~33 KB** (passt in 64 KB)

### 5.2 Unser flash_attn_tiled.comp (Br=16, Bc=32, HEAD_DIM=128)

```
  q_lds:      Br * HEAD_DIM * 4 B  = 8 KB
  k_lds:      TILE * HEAD_DIM * 4 B = 16 KB
  scores_lds: Br * TILE * 4 B = 2 KB
  V: aus Global, kein LDS
  Total: 26 KB
```

### 5.3 Unser geplanter Coopmat-Eigenbau (Br=16, Bc=16)

Kann den cm1-Pattern adaptieren, aber kleiner (single-subgroup, kein
row_split):

```
  q_lds:    Br * HSK * 2 B (FP16)  = 4 KB
  k_lds:    Bc * HSK * 2 B (FP16)  = 1 KB  (Bc=16!)
  v_lds:    Bc * HSV * 2 B (FP16)  = 1 KB
  sfsh:     Br * Bc * 4 B (FP32)   = 1 KB
  Psh:      Br * Bc * 2 B (FP16)   = 0.5 KB
  pvsh:     Br * HSV * 2 B (FP16)  = 4 KB (write-stage für O-update)
  Total: ~11.5 KB
```

Deutlich kompakter als cm1 weil:
* Bc=16 statt 64 (kleinere Tiles, mehr Tile-Iterations)
* Single-subgroup, kein row_split
* HEAD_DIM=128 fix (kein D_split-Logik)

---

## 6. Was wir aus dem Deep-Dive mitnehmen

### 6.1 Bestätigte Annahmen

* ✅ KHR `coopMatLoad / coopMatStore / coopMatMulAdd` reicht für eine
  funktionsfähige Flash-Attention.
* ✅ FP16 KV passt: `f16vec4` Buffer-Aliase auf unser `uint[]` storage
  (oder direkt `float16_t[]` über GL_EXT_shader_16bit_storage)
* ✅ Online-Softmax + causal mask sind machbar OHNE NV-Extensions,
  aber MIT LDS-Roundtrip.
* ✅ LDS-Budget passt komfortabel in 64 KB.

### 6.2 Korrektur der Sprint-Brief-Annahmen

* ❌ "390 LOC coopmat-Attention" — gilt nur für NV-cm2.
  Für KHR-portable cm1: 642 LOC.
* ❌ "Score = coopmat(Q, K^T) → coopmat-Fragment, Softmax direkt
  darauf" — KHR erlaubt das NICHT direkt. Stattdessen:
  Fragment → LDS → Scalar → LDS → Fragment.
* ❌ "Causal Mask in coopmat-Fragmenten" — cm1 macht das via
  externe Mask-Matrix. Wir sollten unsere bestehende inline-Mask
  beibehalten (auf den LDS-Scores).

### 6.3 Was wir NICHT brauchen aus cm1

* row_split / D_split (Multi-Subgroup-Komplexität)
* ALiBi (slope multiplication; nicht in Qwen3)
* Sink tokens (kein Anwendungsfall bei uns)
* Soft-cap (logit_softcap; nicht in Qwen3)
* Split-K (separater reduce-shader)
* Mask-Matrix-Loading (wir haben inline causal)
* Block-Quantized KV (BLOCK_SIZE > 1; wir haben FP16)
* GQA-Storage (nb01/02/03 Komplexität; wir haben simpler Layout)

### 6.4 Was wir aus cm1 ÜBERNEHMEN

* Das LDS-Roundtrip-Pattern für Score↔Softmax↔P
* Die Idee: O-Akkumulator als per-thread Register-Array (f16vec4)
* PV-result → LDS-pvsh → register-accumulate
* Q-tile einmal pro WG laden, K/V pro Tile laden

---

## 7. Port vs Eigenbau — Entscheidung

### 7.1 Vergleichstabelle

| Aspekt | Port cm1 | Eigenbau auf tiled.comp |
|--------|----------|-------------------------|
| LOC | 642 → ~400 (nach Bereinigung) | ~300-400 neu |
| Komplexität | llama.cpp-D_split-Logik verstehen | Eigene Architektur |
| ALiBi/sinks/softcap rauswerfen | Riskant (subtile Code-Stellen) | NICHT da |
| row_split rausziehen | Aufwendig (Multi-Subgroup-Sync) | NICHT da (single SG) |
| Mask-Matrix → inline | Refactor in vielen Stellen | Bereits inline-Stil |
| FP16 KV Read-Pattern | Different (`f16vec4 data_kv4[]`) | Bereits `uint[]` + unpack |
| Debugging | Fremder Code-Stil | Unser Code-Stil |
| Korrektheits-Risiko | Niedriger (durchgetestet) | Mittel (neu) |
| Performance-Risiko | Niedriger (optimiert) | Mittel-Hoch (naive v1) |

### 7.2 Empfehlung: **Eigenbau**

Begründung:
1. **cm1 ist 1.65× komplexer als cm2** wegen NV→KHR-Translation. Ein
   sauberer Port erfordert das Rausschneiden von row_split, ALiBi,
   sinks, soft-cap, split-K, mask-matrix, BLOCK_SIZE>1 — am Ende
   bleibt nicht viel mehr als die zwei coopMatMulAdd-Stellen.
2. **Unser flash_attn_tiled.comp** hat eine getestete Topologie (Br=16,
   Bc=32, default-on seit Sprint 8a). Die zwei coopMatMulAdd-Stellen
   einbauen ist konzentrierter als cm1-Port.
3. **FP16-KV-Read-Pattern** (uint+unpackHalf2x16) aus Sprint 9d.2
   ist bereits da. cm1 nutzt `f16vec4` Buffer-Aliasing — anderer Pfad,
   wäre Refactor.
4. **Inline causal mask** ist unser Stil und einfacher als
   precomputed mask-matrix.

### 7.3 Geplante Sprint 10B Architektur

Neuer Shader: `vk_shaders/flash_attn_coopmat.comp` (~300-400 LOC)

```
Spec-Constants:  Br=16 (fix), Bc=16 (fix für v1; Bc=32 in v2 wenn LDS reicht)
                 HEAD_DIM=128 (Qwen3-fix)
WG-Size:         64 (1 Wave64 = 1 Subgroup)
Bindings:        Q (binding 0, FP32 oder f16vec4), K_packed (binding 1, uint),
                 V_packed (binding 2, uint), O (binding 3, FP32)
                 Same as flash_attn_batch — drop-in compatible

LDS:
  q_lds[Br * HEAD_DIM]:           f16vec4-aligned, ~4 KB
  kv_stage_lds[Bc * HEAD_DIM]:    re-used für K und V, ~4 KB  
  s_lds[Br * Bc]:                 FP32 score staging, ~1 KB
  p_lds[Br * Bc]:                 FP16 P staging, ~0.5 KB
  pv_lds[Br * HEAD_DIM]:          FP16 PV-result staging, ~4 KB
  Total: ~14 KB  (komfortabel unter 64 KB)

Per-thread state:
  float Mf[rows_per_thread];                   // running max
  float Lf[rows_per_thread];                   // running sum
  f16vec4 Of[rows_per_thread][HEAD_DIM/4/...]; // output accumulator

Algorithmus:
  1. Q-tile loaden (Q_lds als f16vec4)
  2. for each K-tile in 0..n_kv step Bc:
     a. K-tile loaden (kv_stage_lds via unpackHalf2x16)
     b. coopMatLoad(Q_frag, q_lds) und coopMatLoad(K_frag, kv_stage_lds)
     c. score_frag = coopMatMulAdd(Q_frag, K_frag^T, 0)
     d. coopMatStore(score_frag, s_lds, RowMajor)  ← LDS roundtrip 1
     e. barrier
     f. Per-thread: causal mask + row_max + online_softmax → p_lds
     g. coopMatLoad(P_frag, p_lds, RowMajor)         ← LDS roundtrip 2
     h. V-tile loaden in kv_stage_lds (überschreibt K)
     i. coopMatLoad(V_frag, kv_stage_lds)
     j. pv_frag = coopMatMulAdd(P_frag, V_frag, 0)
     k. coopMatStore(pv_frag, pv_lds)                ← LDS roundtrip 3
     l. barrier
     m. Per-thread: Of += pv_lds (mit eM rescale wenn nötig)
  3. Final normalize Of via Lf, schreibe O

Erwarteter Gewinn vs flash_attn_tiled.comp:
  • RDNA4 WMMA: 20.8 TFLOPS coopmat vs scalar FMA (Sprint 2A)
  • Bei pp=2048: Attention ist 30-40% der Forward-Zeit
  • coopmat speedup auf attention compute alone: 3-5×?
  • End-to-End impact: pp=2048 ~+30-50% erwartet (additiv zu FP16 KV's +21%)
```

---

## 8. Gate für Phase 10B

**GO**, mit folgenden Scope-Constraints:

1. ✅ Eigenbau, NICHT Port. Cm1 zu komplex zum Bereinigen.
2. ✅ Single-Subgroup-WG (Wave64). Kein row_split.
3. ✅ Br=16, Bc=16 für v1. Bc=32 später wenn v1 läuft.
4. ✅ HEAD_DIM=128 fix (nicht parametrisiert, aber via Spec-Constant
   bei Bedarf).
5. ✅ Inline causal mask auf LDS-staging Scores.
6. ✅ FP16 KV via `f16vec4` Buffer-Alias auf unser `uint[]` storage
   (kompatibel mit Sprint 9d.2).
7. ✅ Drop-in compatible: gleiche Bindings + Push-Constants wie
   flash_attn_tiled.comp.
8. ❌ KEIN ALiBi, sinks, soft-cap, split-K, mask-matrix, BLOCK_SIZE>1.

Erwarteter LOC: 300-400 für den Shader. Plus ~50 LOC Rust-Wiring
(neue ShaderId, Pipeline-Registry, FP16-Pfad-Selektor).

Zeitbudget Sprint 10B: 1-2 Tage konzentrierte Arbeit, am besten ohne
Citrix für saubere Performance-Messungen.

Risiken:
* coopMatLoad-Layout-Match (RowMajor vs ColMajor) — debug-intensiv
* LDS-Bank-Conflicts in den drei Roundtrip-Stages
* FP16-Akkumulation bei langen Sequenzen (FP32 accumulator empfohlen)

Mitigations:
* Inkrementell: erst nur QK-coopmat (PV bleibt scalar) — bisect-fähig
* Spirv-val auf jedem build
* Isolated test: synthetic Q/K/V → bit-equivalent zu scalar reference
* Argmax-Parity vs flash_attn_tiled (existing E2E tests)

---

## 9. Files Touched

```
new:      results/v02_sprint10a_cm2_deepdive.md (this report)
```

KEIN Code geändert. KEIN Test geändert. KEIN Pipeline. KEIN
Forward-Pass. KEIN Build-Setup. Tests: 167/167 unverändert.

---

## 10. Bottom Line

Sprint 10A entdeckte einen kritischen Fehler im Sprint-Brief: cm2
nutzt NVIDIA-spezifische Extensions, die auf RDNA4 nicht
verfügbar sind. Das gesamte ursprüngliche "390 LOC sauber porten"-
Mental-Modell stimmt nicht.

cm1 ist die KHR-portable Variante (642 LOC, +65% wegen Workarounds).
Es funktioniert via:
* **LDS-Roundtrip-Pattern** für Score ↔ Softmax ↔ P (3 Stages
  pro Tile-Iteration)
* **O-Akkumulator als per-thread Register-Array**, NICHT als
  coopmat-Fragment — Online-Softmax-Rescale ist trivial
* **Pre-computed mask matrix** statt inline-causal (bei uns
  überflüssig — wir behalten inline)

Empfehlung: **Eigenbau auf flash_attn_tiled.comp Basis**. Adaptiert
das LDS-Roundtrip-Pattern von cm1, aber single-subgroup, fixed
Br=16/Bc=16, kein row_split, kein D_split, kein
Mask-Matrix-Loading. Geschätzt 300-400 LOC.

Erwarteter Gain bei pp=2048: +30-50% additiv zu FP16 KV's +21%
(wenn coopmat den Attention-Compute auf ~20 TFLOPS hebt).

Sprint 10B GO mit dem dokumentierten Scope. Sprint 10C+ können
dann row_split, Bc=32, und multi-coopmat-Block PV-Topologie
adressieren falls v1 unterperformt.
