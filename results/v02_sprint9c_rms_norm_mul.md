# v0.2 Sprint 9c — Fused RMSNorm-Mul (Honest Negative Result)

**Datum:** 2026-04-29
**Projekt:** VulkanForge v0.2.0
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV/Mesa
**Vorgänger:** Sprint 9a (159/159 Tests)

---

## TL;DR — RMSNorm + Weight-Multiply ist bereits in einem Dispatch fused.

```
═══ v0.2 Sprint 9c ═══

HYPOTHESE (Sprint-Brief):
  rms_norm_mul ist eine 2-Dispatch + 1-Barrier-Sequenz im Forward-
  Pass. An 2 Stellen pro Layer (attn-norm + ffn-norm) lässt sich
  je 1 Dispatch + 1 Barrier sparen → -72 Dispatches, -72 Barriers
  pro Forward, +2-5% Prefill.

EMPIRISCHE ANALYSE:
  Falsch. Unser `vk_shaders/rms_norm.comp` enthält den Weight-
  Multiply BEREITS direkt im Shader, gesteuert durch eine
  Spec-Constant `do_multiply`. Die Pipeline wird in
  `pipeline_registry.rs` mit `do_multiply=1` instanziiert, also
  läuft jeder rms_norm-Dispatch automatisch als rms_norm_mul.
  Das Pattern ist 1:1 identisch zu llama.cpp.

  Konkret:
    vk_shaders/rms_norm.comp:35
      layout (constant_id = 1) const bool do_multiply = false;

    vk_shaders/rms_norm.comp:88-103   (= rms_norm_mul-Branch)
      if (do_multiply) {
          data_d[col] = scale * data_a[col] * data_b[col];
      } else {
          data_d[col] = scale * data_a[col];
      }

    src/backend/vulkan/pipeline_registry.rs:107
      ShaderId::RmsNorm => {
          let data: [u32; 2] = [0, 1];   // (norepeat=0, do_multiply=1)
          ...
      }

    src/backend/vulkan/forward.rs:1641
      fn run_rms_norm(
          ..., input, weight, output, cols, rows, eps, label,
      ) { ... }
      ↑ nimmt einen weight-Buffer als zweiten Parameter, bindet ihn an
        Slot 1, das ist genau data_b im Shader → der Multiply läuft.

  Forward-Pass-Topologie nach JEDEM rms_norm-Dispatch:
    Stelle              | Folge-Op                          | Mul dazwischen?
    --------------------+-----------------------------------+----------------
    rms_norm_attn_b     | quantize_q8_1 ODER GEMM (mul_mm)  |  NEIN
    rms_norm_ffn_b      | quantize_q8_1 ODER GEMM (mul_mm)  |  NEIN
    rms_norm_q_b        | run_rope_neox                     |  NEIN
    rms_norm_k_b        | run_rope_neox                     |  NEIN
    rms_norm_attn       | run_gemv (×3 für Q/K/V)           |  NEIN
    rms_norm_ffn        | run_gemv (gate + up)              |  NEIN
    rms_norm_q/k        | run_rope_neox                     |  NEIN
    rms_norm_final      | embed-output GEMV                 |  NEIN

  An KEINER Stelle existiert ein separater Multiply-Dispatch nach
  rms_norm. Es gibt also keinen "rms_norm + mul" Pair zu fusionieren.

llama.cpp-VERGLEICH:
  llama.cpp's rms_norm_mul_f32 ist DERSELBE SPV-Blob wie sein
  rms_norm_f32, nur mit verschieden gesetzten Spec-Constants.
  Quelle: ggml-vulkan.cpp:4410
    ggml_vk_create_pipeline(..., pipeline_rms_norm_mul_f32,
        "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data,
        "main", 4, sizeof(vk_op_binary_push_constants),
        {1,1,1}, {0, 1}, 1, true);
                          ↑↑↑↑↑↑
                          spec_consts = (norepeat=0, do_multiply=1)
  Wir nutzen exakt denselben Mechanismus.

ENTSCHEIDUNG: KEIN CODE GEÄNDERT.
  • Die postulierte Fusion existiert bereits, es gibt nichts mehr
    zusammenzulegen.
  • Eine alternative Fusion (rms_norm + quantize_q8_1) wäre möglich
    aber unwirtschaftlich — siehe Abschnitt 4.

Tests: 159/159 ✓ (unverändert seit Sprint 9a; kein Code touched).
Files: nur dieser Report.

Bottom line: Sprint 9c's Prämisse stammt aus einer veralteten Mental-
Map, gleiche Klasse von Befund wie Sprint 8b. Die "rms_norm_mul"-
Fusion aus llama.cpp PR #14903 ist bei uns seit Phase 2A live —
ungeachtet, weil sie über einen Spec-Constant statt einen separaten
Shader-Namen aktiviert wird.
```

---

## 1. Detail-Analyse — Wo der Sprint-Brief abbiegt

### 1.1 Was der Sprint-Brief annimmt (Schritt 0.1)

> "Stelle 1: Attention-Block
>   rms_norm_attn(batch_residual) → batch_norm     [Dispatch 1]
>   barrier
>   quantize_q8_1(batch_norm) → batch_q8           [Dispatch 2]
>   FRAGE: Wird batch_norm VOR quantize noch mit einem Weight
>   multipliziert?"

Implizit: Die Annahme ist, dass `rms_norm` _nur_ normalisiert
(also `data_d = scale * data_a`) und dass es eine separate
Multiply-Stelle für das `attn_norm.weight`-Tensor gibt.

### 1.2 Was der Code wirklich tut

Der Weight-Multiply ist Bestandteil des `rms_norm`-Kernels selbst,
gesteuert durch die Spec-Constant `do_multiply` auf id 1. Wenn
diese auf `true` gesetzt ist (was in unserer Pipeline immer der
Fall ist), läuft der Branch:

```glsl
// vk_shaders/rms_norm.comp:88-103
if (do_multiply) {
    if (ncols > p.ne10) {
        ...
        data_d[d_offset + col] = D_TYPE(
            scale * FLOAT_TYPE(data_a[a_offset + col])
                  * FLOAT_TYPE(data_b[b_offset + fastmod(col, p.ne10)])
        );
    } else {
        data_d[d_offset + col] = D_TYPE(
            scale * FLOAT_TYPE(data_a[a_offset + col])
                  * FLOAT_TYPE(data_b[b_offset + col])
        );
    }
}
```

`data_a` ist der Input (residual), `data_b` ist das Weight, `data_d`
ist der Output. Eine Op, ein Dispatch, eine Barrier nach dem Dispatch.

### 1.3 Beweis: pipeline_registry.rs setzt `do_multiply=1`

```rust
// src/backend/vulkan/pipeline_registry.rs:103-111
ShaderId::RmsNorm => {
    // SpecId 0 = norepeat (false → broadcast-safe), 1 =
    // do_multiply (true → norm-with-gamma, which is the
    // standard transformer use case).
    let data: [u32; 2] = [0, 1];
    let entries = [entry(0, 0, 4), entry(1, 4, 4)];
    let bytes = bytemuck::bytes_of(&data);
    ComputeKernel::from_spv_with_spec(device, &words, cache, &entries, bytes)
}
```

Die Pipeline wird mit `do_multiply=1` instanziiert. Damit emittiert
jeder `cmd_dispatch` auf der `RmsNorm`-Pipeline den fused
norm-mul-Branch.

### 1.4 Beweis: run_rms_norm bindet Weight an Binding 1

```rust
// src/backend/vulkan/forward.rs:1641-1681
fn run_rms_norm(
    &mut self, dev, registry, cmd,
    input: vk::Buffer,
    weight: vk::Buffer,         // ← weight ist Pflicht-Parameter
    output: vk::Buffer,
    cols, rows, eps, label,
) {
    let kernel = registry.get(ShaderId::RmsNorm);
    let set = self.alloc_or_get_set(
        dev, kernel.descriptor_set_layout,
        &[(0, input, 0, 0), (1, weight, 0, 0), (2, output, 0, 0)],
                              ↑ data_b = weight tensor
    );
    ...
}
```

Jeder Aufrufer übergibt einen echten Weight-Buffer (siehe Abschnitt 2).
Es gibt keinen "rms_norm without weight"-Pfad im Hot-Path-Code.

---

## 2. Forward-Pass — Was nach jedem rms_norm wirklich kommt

Vollständige Liste aller `run_rms_norm`-Callsites (`grep run_rms_norm`):

| forward.rs Zeile | Pfad                  | Label              | Folge-Op                     |
|------------------|-----------------------|--------------------|------------------------------|
|  997-998         | dispatch_layer        | rms_norm_attn      | barrier → 3× run_gemv (Q/K/V)|
| 1029-1031        | dispatch_layer (qk)   | rms_norm_q         | (kein barrier zwischen q+k)  |
| 1032-1034        | dispatch_layer (qk)   | rms_norm_k         | barrier → run_rope_neox      |
| 1268-1269        | dispatch_layer        | rms_norm_attn      | barrier → 3× run_gemv (alt)  |
| 1296-1298        | dispatch_layer (qk)   | rms_norm_q         | (kein barrier zwischen q+k)  |
| 1299-1301        | dispatch_layer (qk)   | rms_norm_k         | barrier → run_rope_neox      |
| 1358-1360        | dispatch_layer        | rms_norm_ffn       | barrier → 2× run_gemv        |
| 1426-1429        | forward_token         | rms_norm_final     | (lm_head GEMV)               |
| 2655-2658        | dispatch_layer_batch  | rms_norm_attn_b    | barrier → quantize ODER GEMM |
| 2792-2796        | dispatch_layer_batch  | rms_norm_q_batch   | (kein barrier zwischen q+k)  |
| 2798-2802        | dispatch_layer_batch  | rms_norm_k_batch   | barrier → RoPE               |
| 2893-2895        | dispatch_layer_batch  | rms_norm_q_b       | (alt path)                   |
| 2897-2899        | dispatch_layer_batch  | rms_norm_k_b       | (alt path)                   |
| 3061-3064        | dispatch_layer_batch  | rms_norm_ffn_b     | barrier → quantize ODER GEMM |

**An keiner Stelle folgt ein separater Multiply-Dispatch.** Die
direkte Folge ist immer eines von:
* GEMV (Q4_K mat-vec für Decode)
* Quantize_Q8_1 (für mul_mmq Pfad in Batched)
* GEMM (für mul_mm Pfad in Batched, FP32-direkt)
* RoPE (nach Q/K-Norm)
* Embed-Output GEMV (rms_norm_final)

---

## 3. llama.cpp-Vergleich — exakt das gleiche Pattern

llama.cpp's Vulkan-Backend hat KEINE separate `rms_norm_mul.comp`-
Datei (nur `rms_norm.comp`, `rms_norm_back.comp`, `rms_norm_partials.comp`).
Die "rms_norm_mul"-Pipeline ist via Spec-Constants konfigurierter
**derselbe SPV-Blob**:

```cpp
// ggml-vulkan.cpp:4408-4410 (gekürzt)
ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32,
    "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data,
    "main", 4, ..., {1,1,1}, {0, 0}, 1, true);
                                ↑ do_multiply=0
ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32,
    "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data,
                        ↑ SELBES SPV
    "main", 4, ..., {1,1,1}, {0, 1}, 1, true);
                                ↑ do_multiply=1
```

llama.cpp wählt zwischen den beiden Pipelines je nach Op-Graph
(via `ctx->num_additional_fused_ops`, ggml-vulkan.cpp:9461-9463).
**Wir wählen die fused-Variante immer**, weil unser Forward-Pass
hardcodiert den norm-mul-Pattern dispatcht — der nicht-fused
"nur normalize"-Mode (`do_multiply=0`) wird in unserem Code
nirgends erreicht.

---

## 4. Alternative Fusion — rms_norm_mul + quantize_q8_1?

Der einzig theoretisch mögliche neue Fusionskandidat im Hot-Path
wäre `rms_norm_mul + quantize_q8_1` für den batched-mmq-Pfad
(`dispatch_layer_batch` Stellen 2655-2683 und 3061-3076):

```rust
// VORHER:
self.run_rms_norm(... batch_residual, w_attn_norm → batch_norm);
compute_barrier(dev, cmd);
self.run_quantize_q8_1(... batch_norm → batch_q8);
compute_barrier(dev, cmd);

// HYPOTHETISCHE Fusion:
self.run_rms_norm_mul_quantize(... batch_residual, w_attn_norm → batch_q8);
compute_barrier(dev, cmd);
```

### 4.1 Warum das unwirtschaftlich ist

1. **Zwei Reductions verschiedener Granularität.**
   - rms_norm braucht `mean(x²)` über die ganze Row (4096 Elemente).
   - quantize_q8_1 braucht `max(|x|)` pro 128-Element-Block (32 Blöcke pro Row).
   Beide müssten in einer Pass berechnet werden. Das geht, ist aber
   nicht-trivial: man muss die rms-Sum über die ganze Row aufbauen
   UND parallel pro Block den Max sammeln. Mehr LDS, mehr Barriers
   im Inneren.

2. **Pfad-Divergenz.** Der mul_mm-Pfad und der coopmat-Pfad
   überspringen die quantize-Stufe komplett (sie nehmen FP32
   direkt). Nach Sprint 8a ist mul_mmq nur noch der gated
   Fallback. Eine fused-rms-quantize-Pipeline würde nur den
   Fallback beschleunigen; im Default-Pfad (mul_mm aligned bei
   seq_len%4==0) wäre sie nicht aktiv.

3. **llama.cpp baut sie nicht.** llama.cpp hat eine `rms_norm_partials`-
   Variante für split-K rms-Reductions, aber **keine** rms_norm-
   plus-quantize-Fusion. Wenn der Gewinn substantiell wäre,
   hätten sie ihn längst.

4. **Geschätzter Gewinn: 1-2%** auf den mmq-Fallback-Pfad bei
   großem Aufwand (kompletter neuer Shader, neue Push-Constants,
   neue Pipeline, Tests gegen FP32+Q8_1-Roundtrip). Niedriger
   ROI als jeder andere Sprint-9-Kandidat.

---

## 5. Sprint-9-Roadmap nach Sprint 9c

Mit dem 9c-Befund verschiebt sich die Priorität:

* **Sprint 9b — multi_add + multi_add_rms** (1-2 Tage):
  llama.cpp's `multi_add`-Kernel summiert bis zu 6 Inputs in
  einem Dispatch. Anwendung: `add_res1` und `add_res2` (Residual-
  Addition). Im aktuellen Code sind das 2 Dispatches/Layer.
  Falls multi_add_rms (Residual + nächstes RMSNorm in 1) machbar
  ist: -2 Dispatches, -2 Barriers pro Layer → 36×2 = -72 pro
  Forward. **Erwarteter Gain: +3-7%**. Höchster verbleibender
  ROI.

* **Sprint 9c.5 — rms_norm_mul_rope** (2 Tage):
  llama.cpp hat `pipeline_rms_norm_mul_rope_f32_f32` (siehe
  ggml-vulkan.cpp:4415). Fusioniert Q-Norm/K-Norm + RoPE in
  einem Kernel. **Erwarteter Gain: +2-3%** (nur Q/K-Pfad).

* **Sprint 9d — FP16 KV-Cache** (2-3 Tage):
  Halbiert Memory-Bandwidth. Besonders relevant für pp ≥ 2048
  wo wir bei 0.14-0.22× von llama.cpp sind. **Erwarteter Gain:
  +5-15% am langen Kontext**.

NICHT priorisieren:
* Sprint 9c reboot mit rms_norm + quantize-Fusion — niedriger ROI,
  llama.cpp hat es nicht, wir würden den Fallback-Pfad
  beschleunigen statt den Hot-Path.

---

## 6. Files Touched

```
new file: results/v02_sprint9c_rms_norm_mul.md (this report)
```

KEIN Shader-Code geändert. KEIN Test geändert. KEIN Pipeline
geändert. KEIN Forward-Pass geändert.

Tests: 159/159 ✓ (unverändert seit Sprint 9a; cargo test
nicht erneut gelaufen — kein Code touched).

---

## 7. Fazit

Sprint 9c ist die zweite "Mental-Map-Korrektur" in dieser
Optimierungs-Reihe (nach Sprint 8b). Wie 8b basiert die Hypothese
auf einer naiven Vorstellung des Forward-Passes; die postulierte
Fusion ist seit Phase 2A im Code (rms_norm.comp wurde von Anfang
an aus llama.cpp portiert, _inklusive_ des `do_multiply`
Spec-Constants).

Empfehlung: Nächster Sprint **9b (multi_add)** statt 9c-Reboot.
Die Residual-Addition (`add_res1`, `add_res2`) ist die nächste
realistisch fusionierbare Stelle im Forward-Pass.

---

## 8. Empfehlung — Sprint 9b/9c.5/9d

```
| Sprint | Beschreibung                       | Aufwand  | Erw. Gain | ROI |
|--------|------------------------------------|----------|-----------|-----|
| 9b     | multi_add (Residual+Residual)      | 1-2 Tage | +3-7%     | ★★★ |
| 9c.5   | rms_norm_mul_rope (Q/K-Norm+RoPE)  | 2 Tage   | +2-3%     | ★★  |
| 9d     | FP16 KV-Cache                      | 2-3 Tage | +5-15%    | ★★★ |
```

Sprint 9b ist der natürliche Nachfolger: gleiche Komplexität wie
9a, höherer Gewinn (2× Stellen pro Layer), und im Gegensatz zu
9c eine echte Lücke zwischen unserem Code und llama.cpp.
