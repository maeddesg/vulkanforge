# VulkanForge Grand Audit (v0.2.0, post-Sprint 3C)

**Datum:** 2026-04-28
**Hardware:** AMD RX 9070 XT, RDNA 4, gfx1201, RADV / Mesa 25.x
**Frage:** Wo verlieren wir die 2.17× gegen llama.cpp's Vulkan-Backend
(1047 tok/s prefill vs ≥ 2274 tok/s)? Und ist der coopmat-Weg auf
RDNA4 für Skinny-N Prefill überhaupt gewinnbar?

**Methodik:** Drei parallele Investigationen — VulkanForge-Source-
Audit, llama.cpp/ggml-vulkan-Vergleich, Web-Recherche zu RDNA4
coopmat / GEMM-Tuning / GGML-Architektur. Synthese unten.

---

## 1 — Headline: zwei voneinander unabhängige Probleme

```
Problem A — Coopmat at skinny-N is fundamentally underwater on RDNA4
            ohne NV_cooperative_matrix2 (das RADV nicht hat).
            → Coopmat aufgeben für den Hot-Path.

Problem B — Unser mul_mmq selbst ist ~2× langsamer als llama.cpp's,
            obwohl wir denselben Shader-Source porten.
            → Der Gap ist in der Host-Infrastruktur, nicht im Kernel.

Sprint 3A → 3C hat sich auf Problem A festgebissen (-57% bis -61%).
Problem B ist seit Phase 7 ungelöst und der eigentliche
Hauptengpass.
```

---

## 2 — Warum coopmat bei Skinny-N strukturell verliert

### 2.1 Public evidence (Vulkanised 2025, vLLM, llama.cpp R9700)

* **Jeff Bolz (NVIDIA), Vulkanised 2025**: ein „naiver"
  KHR_coopmat-GEMM auf RTX 4070 erreicht **8 TFLOPS gegen 116 TFLOPS
  Peak** = 7% des Geräts. Sein O-Ton: *„KHR_coopmat matrices are
  small enough that using them the natural way in a GEMM does not
  lead to good performance."* Workgroup-scope-Tiling (NV_coopmat2
  mit 256×256-Tiles, 8 subgroups, K=32 K-step) bringt das auf
  ~98 TFLOPS. RDNA4 / RADV haben **kein NV_coopmat2** — wir sitzen
  in der „naive subgroup-scope" Falle.
  → [vulkan.org/T47-Jeff-Bolz-NVIDIA.pdf](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf)

* **vLLM RDNA4**: nutzt FP8 WMMA mit hartem `M ≥ 16`-Floor; unter
  M=16 fällt der Backend auf FP32-Dequant zurück. → vLLM hat AUCH
  keinen brauchbaren coopmat-skinny-N-Pfad und löst das Problem
  durch Padding + WMMA-only-für-M≥16.
  → [discuss.vllm.ai/.../1900](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900)

* **R9700 Llama-Experimente** (gfx1201, AMDVLK): coopmat hilft
  ~17% bei MoE-pp2048, ist „neutral" auf Dense-Decode, **integer-dot
  hat negligibler Einfluss** = WMMA verliert seinen Vorteil bei
  kleinem N.
  → [github.com/ggml-org/llama.cpp/discussions/21043](https://github.com/ggml-org/llama.cpp/discussions/21043)

* **PR #12135 (DP4A MMQ in ggml-vulkan)**: AMD Vega20 Q4_0
  310→800 t/s (+158%), RX 6800 XT 921→1845 t/s (+100%), RTX 3090
  1020→2895 t/s (+184%). Der int-dot-Pfad **schlägt FP-coopmat
  konsistent** auf den Quantisierungs-Workloads.
  → [github.com/ggml-org/llama.cpp/pull/12135](https://github.com/ggml-org/llama.cpp/pull/12135)

### 2.2 Strukturelle Begründung

Naive coopmat hat ein 16×16×16 Tile pro Subgroup. Für Output-
Element-Dichte braucht es entweder:

1. **Viele Tiles parallel** (große M, große N) — wir haben N≤64.
2. **Workgroup-scope tiling** (NV_coopmat2) — RADV hat es nicht.
3. **Manuelles Multi-Subgroup-Tiling** mit shared-memory Staging —
   das ist genau, was wir in Sprint 1A/1B versucht haben (BN=64
   tiled = 16.7 TF auf 4096³, **aber 1.7 TF auf Skinny-N**).

DP4A/v_dot8 läuft auf der regulären VALU mit 4–8 Mults pro Cycle
**ohne 16er-Tile-Padding-Penalty**. Bei N≤64 saturiert mul_mmq
(BM=64, BN=64, scalar FMA + DP4A) die Wavefronts; coopmat verliert,
weil das 16×16-Tile wenig zu tun hat und die Subgroup-zu-Output-
Element-Mapping nicht effizient ist.

**Verdikt zu coopmat bei Skinny-N:** unwinnable ohne
NV_cooperative_matrix2. Wir haben 6 Wochen in einen Pfad
investiert, der von Hardware her gar nicht der richtige ist für
unsere Hot-Path-Shapes.

---

## 3 — Wo der mul_mmq-Gap (1047 vs 2274 tok/s) wirklich liegt

Wir nutzen **denselben Shader-Source** wie llama.cpp (`mul_mmq.comp`,
non-coopmat-non-MoE-Q4_K-Pfad). Der 2.17× Gap kann nicht im
Kernel-Algorithmus liegen. Wo sitzt er dann?

### 3.1 Spec constants — wir kompilieren den Shader anders!

VulkanForge `pipeline_registry.rs` lines 195–212 (Q4_K-Pfad):

```
BLOCK_SIZE = 256        TM = 2     WARP = 64    BM = BN = 64
                        TN = 4
```

llama.cpp `mul_mmq.comp` lines 70–80 (default constants, der
Shader-Generator pinnt sie shape-abhängig):

```
BLOCK_SIZE = 64         TM = 4     WARP = 32    BM = BN = 64
                        TN = 2
```

**Drei Unterschiede.** Wir haben `BLOCK_SIZE=256`, llama.cpp `=64`.
Wir haben `WARP=64`, llama.cpp `=32`. Wir haben `TM=2/TN=4`,
llama.cpp `TM=4/TN=2` — die Achsen sind **VERTAUSCHT**. WMITER ist
in beiden 2.

* **BLOCK_SIZE 256 vs 64**: wir laufen 4× größere Workgroups. Das
  reduziert die Zahl der WGs für M=4096 N=64 von ~520 (llama.cpp)
  auf 130 (uns) — wir verlieren Latenz-Hiding-Spielraum auf den
  RDNA4-CUs.
* **WARP 64 vs 32**: RDNA4 unterstützt sowohl Wave32 als auch
  Wave64. llama.cpp's `WARP=32` mit `BLOCK_SIZE=64` ergibt 2
  Wavefronts pro WG (Wave32-Pfad), was auf RDNA4 oft besser
  skaliert als unser Wave64-Pfad mit 4 Waves/WG.
* **TM/TN swap**: llama.cpp's `TM=4 TN=2` re-uses geladene
  Q4_K-Weight-Rows besser bei `WMITER=2`. Unser `TM=2 TN=4` ist die
  *umgedrehte* Phase-6-Optimierung — der 2026-Entscheid sollte
  überprüft werden.
* **3 Workgroup-Denoms (S/M/L)**: llama.cpp pickt eine von **drei
  spec-constant-Sätzen** abhängig von M/N (`m<=32 || n<=32` → S,
  `m<=64 || n<=64` → M, sonst L). Wir haben **EINE** Konfiguration
  für alle Shapes.

→ Quick-Win: Spec-Constants auf llama.cpp-defaults wechseln und
neu benchen. Erwartet: signifikanter Anteil des 2.17×-Gaps.

### 3.2 Barrier-Strategie

VulkanForge `dispatch_layer_batch` (mul_mmq-Pfad, lines 2521-2626):

```
Per-Layer Sequence:
  rms_norm_attn → barrier
  quantize_q8_1 (attn) → barrier
  gemm_q, gemm_k, gemm_v → barrier
  Q-norm, K-norm, RoPE-Q, RoPE-K → barrier
  flash_attn_batch → barrier
  gemm_o → barrier
  add_res1 → barrier
  rms_norm_ffn → barrier
  quantize_q8_1 (ffn) → barrier
  gemm_gate, gemm_up → silu+mul → barrier
  gemm_down → barrier
  add_res2 → barrier
  ─────────────────────────────────────────
  ≈ 11 unconditional `compute_barrier` per layer
  × 36 layers = ~400 barriers per forward pass
```

llama.cpp `ggml_vk_sync_buffers`: **49 calls in the entire host
codepath**, but **conditional on pipeline-change**: lines 7693-7712
in `ggml-vulkan.cpp` cache `prealloc_y_last_pipeline_used` and
`prealloc_y_last_tensor_used`. **Wenn die nächste Op dasselbe
Pipeline + denselben Tensor benutzt → KEIN Barrier.**

Beispiel: gemm_q, gemm_k, gemm_v laufen alle gegen `batch_q8` mit
DERSELBEN mul_mmq-Pipeline. llama.cpp emittiert dort **null**
Barrier zwischen den dreien (sie schreiben in disjunkte
Output-Buffer; Reads vom gleichen Input-Buffer brauchen keine
Synchronisation). Wir emittieren einen post-attn-norm + post-quantize
+ post-Q/K/V Barrier — alle drei sind in unserem Code unconditional.

→ Optimistische Schätzung: pro Layer ~5 Barriers eliminierbar →
~180 Barriers / Forward → vermutlich 5-15% Prefill-Lift, der
größte Einzel-Hebel ohne Architektur-Umbau.

### 3.3 quantize_q8_1: zweimal pro Layer ohne Caching

* Wir rufen `run_quantize_q8_1` **zweimal pro Layer** (nach
  attn_norm und nach ffn_norm), unconditional.
* Pro Aufruf: ~2048 WGs für seq_len=64, hidden=4096. ~576 KiB
  geschriebene Q8_1-Bytes pro Token pro Layer.
* llama.cpp: `quantize_y` is cached (lines 7703-7712). Wenn der
  gleiche Tensor mit der gleichen Pipeline schon quantisiert wurde
  → SKIP. Beim Q/K/V GEMM-Triple wird die Q8_1-Quantisierung
  **einmal** gemacht und für alle drei wiederverwendet.

In unserem Code: `gemm_q`, `gemm_k`, `gemm_v` lesen alle
batch_q8 (richtig — dieselbe Quantisierung wird wiederverwendet),
aber WIR rufen quantize_q8_1 trotzdem nur einmal vor Q/K/V — der
Cache ist also implizit OK für Q/K/V. Das gleiche gilt für Gate/Up
nach FFN-Norm (eine Quantisierung, zwei GEMMs). Aber: Gate+Up
quantisiert nochmal die FFN-Norm-Output, dann gemm_down
quantisiert das silu(gate)·up Result. **Drei** quantize_q8_1
pro Layer, nicht zwei wie ich vorhin schrieb.

Noch einmal nachgezählt mit forward.rs lines 2543, 2870, 2941:
* quantize_attn (line 2543): batch_norm → batch_q8 (für Q/K/V)
* quantize_attn_out (line 2772): batch_attn_out → batch_q8 (für O)
* quantize_ffn (line 2887): batch_norm (FFN-norm output) → batch_q8 (für Gate/Up)
* quantize_ffn_h (line 2952): batch_ffn_hidden → batch_q8 (für Down)

= **4 quantize_q8_1 dispatches per layer × 36 layer = 144 separate
quantize-Dispatches per Forward-Pass.** Für ein 64-Token-Prefill
sind das ~4ms reine Quantize-Compute (geschätzt aus
~30µs/dispatch × 144).

→ llama.cpp's per-pipeline Caching dürfte 50% davon eliminieren
(Q/K/V teilt sich den ersten quantize, aber gemm_o und gemm_down
brauchen je einen eigenen). Aber: **wir teilen den auch schon** —
batch_q8 wird wiederverwendet zwischen Q/K/V. Der Gewinn ist klein.

### 3.4 Command-Buffer Submission

llama.cpp PR #10499 (Bolz): batch ramp-up von 1 → 100 nodes pro
`vkQueueSubmit` startet den ersten Submit früher. Erwarteter
Throughput-Gain: ~1-2%. Klein.

VulkanForge submitted die KOMPLETTE prefill_batch in **einem**
VkCommandBuffer (wir haben ein einziges `cmd_ctx.one_shot`). Das
ist eigentlich besser als llama.cpp's 100-Node-Batches — wir
saturieren die GPU-Pipeline-Submission gar nicht erst.

→ **Hier liegt KEIN Gap.** Wir batchen aggressiver als llama.cpp.

### 3.5 Pipeline-Cache

Wir laden + speichern eine `~547 KB` Vulkan-Pipeline-Cache von Disk
(`pipeline_registry.rs` line 70-78). llama.cpp: **kein
persistenter Cache** (grep findet kein `vkCreatePipelineCache` mit
`initial_data`). Pipelines werden bei jedem Start neu compiliert.

→ Wir sparen Pipeline-Compile-Zeit (millisekunden-bereich) gegen
llama.cpp's Cold-Start. Pro Forward-Pass irrelevant.

### 3.6 Buffer-Aliasing

llama.cpp's GGML hat **drei Prealloc-Buffer** (`prealloc_x`,
`prealloc_y`, `prealloc_split_k`) die für ALLE Intermediate-
Tensoren wiederverwendet werden, lifetime-tracked.

VulkanForge: **dedizierte Batch-Buffer pro Tensor-Klasse**
(`batch_norm`, `batch_q`, `batch_k`, `batch_v`, `batch_o`,
`batch_attn_out`, `batch_q8`, `batch_gate`, `batch_up`,
`batch_ffn_hidden`, `batch_ffn_out`). Jeder ist auf
`max_prefill_tokens × dim × 4` allokiert.

VRAM-Verbrauch ist höher (~80 MB Batch-Buffer), aber Performance-
seitig macht das vermutlich nichts: jeder Buffer ist persistent
und wird nicht reallokiert.

→ Kein Performance-Gap, vermutlich.

---

## 4 — ROI-Tabelle

| # | Hebel                                          | Aufwand | Erwartet. Gain | ROI  |
|---|------------------------------------------------|---------|----------------|------|
| 1 | Spec-Constants auf llama.cpp-Werte (BLOCK_SIZE 256→64, WARP 64→32, TM/TN swap) | 1 Tag | **~30-50%** auf Prefill | extrem hoch |
| 2 | 3-Workgroup-Denom-Variants (S/M/L) wie llama.cpp | 2-3 Tage | weitere ~10% | hoch |
| 3 | Conditional barriers (pipeline-change tracking) | 3-5 Tage | ~5-15% | hoch |
| 4 | quantize_q8_1 caching (skip wenn (tensor, pipeline) gleich)   | 1-2 Tage | 1-3% | mittel |
| 5 | Coopmat-Pfad als Default OFF lassen, Code-Pfad **entfernen** wenn Sprint 4 bestätigt: keine Win-Shapes für unsere Workloads | 2-3 Tage Code-Cleanup | -1500 LoC, weniger Maintenance | reife Tech-Debt-Ent­scheidung |
| 6 | Multi-User Batching (M_batch=4-8 Sessions kombiniert) für Server-Use-Case | 2 Wochen | bei N=128+: coopmat tiled wird relevant | nur falls Server-Mode |
| 7 | NV_coopmat2 Adapter falls RADV es jemals shipped (Mesa 26.x?) | RADV-Roadmap | Bolz' 7% → 80%+ Lift | langfristig |
| 8 | Disk Pipeline-Cache rauswerfen (Match llama.cpp, weniger Komplexität) | 1 Tag | 0% (rein Maintenance) | nur Cleanup |

---

## 5 — Empfehlung für v0.2

### Vor diesem Audit

Sprint 3D plante: Block-Header-Caching, gemm_down-NaN-Debug,
Q6_K-Coopmat — alles im Coopmat-Pfad.

### Nach diesem Audit

**Pivot.** Drei klare Hebel sichtbar, alle nicht-coopmat:

1. **Spec-constants angleichen** (Hebel #1). Match llama.cpp's
   BLOCK_SIZE=64, WARP=32, TM=4, TN=2. Re-bench.
2. **Drei Workgroup-Variants** (Hebel #2). S/M/L mit shape-abhängiger
   Pipeline-Auswahl bei dispatch.
3. **Conditional barriers** (Hebel #3). Track
   `last_pipeline_per_buffer` und skip Barrier wenn (Pipeline,
   Buffer) unverändert.

Diese drei zusammen können **realistisch +50-70% Prefill-Lift**
liefern. Damit wäre VulkanForge bei ~1500-1800 tok/s — immer noch
nicht 2274, aber 70-80% des llama.cpp-Niveaus mit deutlich
weniger Code (kein GGML-Graph-Planner-Refactoring).

**Coopmat-Pfad strategisch:** Default OFF lassen für Sprint 4.
Falls Sprint 4 keine Win-Shape findet (Multi-User-Batching ist die
einzige plausible Tür) → Coopmat-Code als „Reference / Future"
markieren und nicht mehr aktiv pflegen. Sprint 1A-3C haben den
Skeleton gebaut; er steht da, falls NV_coopmat2 jemals von RADV
geshipped wird.

### Was v0.2 wirklich sollte liefern

* **v0.2.0** (jetzt): Coopmat-Plumbing + Bug-Postmortem (Sprint
  3A-3C). Default OFF. Geliefert.
* **v0.2.1** (Sprint 4): Spec-Constants-Pivot + Variants-Selector
  + conditional Barriers. Ziel: ≥ 1500 tok/s prefill (70% von
  llama.cpp).
* **v0.2.2** (Sprint 5, optional): GGML-Style Memory-Planner falls
  Sprint 4 ≥ 80% von llama.cpp.

---

## 6 — Coopmat: behalten oder aufgeben?

**Empfehlung: behalten als API-/Pipeline-Skelet, aufgeben als
Hot-Path.**

Begründung:

* Coopmat ist auf RDNA4-RADV strukturell nicht
  konkurrenzfähig zu mul_mmq bei N≤64. Bolz, vLLM, R9700-
  Experimente und unsere eigenen Sprint-1A-Bench-Daten sind sich
  einig.
* Coopmat wird relevant bei N≥128 (Multi-User-Batching) oder mit
  NV_coopmat2 (RADV-Roadmap unklar). Falls einer der beiden
  eintritt, haben wir einen funktionierenden Skeleton.
* Default OFF kostet uns nichts (140/140 Tests grün ohne, 145/145
  mit). Sprint 3A-3C hat das Code-Volumen angelegt — die
  Wartungs-Last ist überschaubar wenn Default OFF bleibt.

**Konkret:** keine weiteren Coopmat-Optimierungen in v0.2 / v0.3.
Sprint 4-5 fokussiert ausschließlich auf den mul_mmq-Pfad und das
Host-side-Wiring. Das schließt den Sprint-1A bis 3C-Loop sauber ab.

---

## 7 — Was wir übersehen haben

1. **Bolz' Vulkanised-2025 Talk.** Der Talk vom Februar 2025 sagt
   wörtlich: naive coopmat = 7% von Peak. Das hätten wir vor Sprint
   1A lesen sollen. Sprint 0 (Smoke-Test) hatte den Talk in den
   Referenzen gehabt aber NICHT als „naive coopmat verliert"-
   Warnung interpretiert.
2. **DP4A PR #12135.** Die ggml PR von Februar 2025 zeigt
   2-3× int-dot-Speedup. Das ist die *Hauptoptimierung*, die
   llama.cpp's mul_mmq schnell macht — und die wir bereits portiert
   haben. Das erklärt warum unser mul_mmq überhaupt die 1047 t/s
   schafft. Aber wir haben den Pfad nicht weiter getuned.
3. **Spec-Constants als Variants.** llama.cpp hat 3 mul_mmq-
   Pipelines pro Quant-Type (S/M/L). Wir hatten überhaupt nicht
   überlegt, dass das ein Optimierungs-Hebel ist.
4. **WARP=32 auf RDNA4.** RDNA4 kann sowohl Wave32 als auch
   Wave64. Unser hartcodiertes WARP=64 ist für Wave64-only. Mit
   Wave32 (WARP=32) sind die Latenzen kürzer und Issue-Slots
   freier — llama.cpp nutzt das.
5. **Conditional Barriers.** Vulkan-Sync-Best-Practices (NVIDIA
   Dos&Don'ts; Khronos sample) sagen: weniger Barriers ist mehr.
   Wir haben das nicht eingebaut.

---

## 8 — Quellen

Source-Audit:
* `/home/maeddes/projects/vulkanforge/src/backend/vulkan/forward.rs`
  (3000+ LoC)
* `/home/maeddes/projects/vulkanforge/src/backend/vulkan/pipeline_registry.rs:195-212`
  (mul_mmq spec constants)
* `/home/maeddes/projects/vulkanforge/vk_shaders/mul_mmq.comp`
* `/home/maeddes/tmp/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:7141-7712`
  (mul_mat dispatch + barrier strategy)
* `/home/maeddes/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp:70-80`
  (default spec constants)
* `/home/maeddes/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:594`
  (`warptile_mmq_int_k`)

Web:
* [Vulkanised 2025 — Bolz on cooperative_matrix](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf)
* [GPUOpen — Matrix Cores on RDNA4](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
* [GPUOpen — WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/)
* [llama.cpp PR #12135 — DP4A MMQ + Q8_1](https://github.com/ggml-org/llama.cpp/pull/12135)
* [llama.cpp PR #10499 — first CB submitted sooner](https://github.com/ggml-org/llama.cpp/pull/10499)
* [llama.cpp #19890 — R9700 Qwen3 prefill numbers](https://github.com/ggml-org/llama.cpp/discussions/19890)
* [llama.cpp #21043 — RDNA4 R9700 coopmat-on/off](https://github.com/ggml-org/llama.cpp/discussions/21043)
* [vLLM forum — Native FP8 WMMA for RDNA4](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900)
* [Phoronix — RADV Vulkan beats ROCm on RX 9070 XT](https://www.phoronix.com/review/llama-cpp-windows-linux/5)
* [NVIDIA Vulkan Dos and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts/)
* [Khronos — Vulkan pipeline-barriers performance](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html)
* [DeepWiki — ggml Vulkan backend](https://deepwiki.com/ggml-org/ggml/3.3-vulkan-backend)
* [AMD RDNA4 ISA Reference](https://amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
