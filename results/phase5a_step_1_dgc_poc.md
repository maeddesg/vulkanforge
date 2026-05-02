# Phase 5A-1 — DGC API Study (Schritt 1)

**Datum:** 2026-04-26
**Status:** API-Studie abgeschlossen. STOP-Punkt vor Coding (Schritte 2–4) erreicht.
**System:** AMD Radeon RX 9070 XT (RADV GFX1201, Mesa 26.0.5-arch2.4, Vulkan 1.4.341)

---

## 1.1 / 1.2 — Spec + RADV-Implementierung

### Driver-Status (vulkaninfo)

```
deviceName     = AMD Radeon RX 9070 XT (RADV GFX1201)
driverName     = radv
driverInfo     = Mesa 26.0.5-arch2.4
extensionRev   = 1   (VK_EXT_device_generated_commands)
```

Das RX 9070 XT exponiert die Extension. Mesa 24.3.0 (Q4 2024) brachte den
Initial-Support für nvk + radv (Samuel Pitoiset, Valve);
Mesa 26.0.5 ist deutlich neuer und sollte alle Bugfixes enthalten.

### Properties (RADV-Limits)

```
maxIndirectPipelineCount                        = 4096
maxIndirectShaderObjectCount                    = 4096
maxIndirectSequenceCount                        = 1 048 576
maxIndirectCommandsTokenCount                   = 128
maxIndirectCommandsTokenOffset                  = 2047
maxIndirectCommandsIndirectStride               = 2048
supportedIndirectCommandsShaderStages           = ALL  (VERTEX, …, COMPUTE, RAYGEN, …)
supportedIndirectCommandsShaderStagesPipelineBinding = SHADER_STAGE_COMPUTE_BIT
supportedIndirectCommandsShaderStagesShaderBinding   = SHADER_STAGE_COMPUTE_BIT
deviceGeneratedCommandsTransformFeedback        = true
deviceGeneratedCommandsMultiDrawIndirectCount   = true
```

**Critical**: Compute-Pipeline-Binding via DGC ist explizit unterstützt. Der
Pipeline- und der Shader-Object-Binding-Pfad sind beide auf COMPUTE
limitiert (kein Graphics-Pipeline-Switch — irrelevant für uns).

### Token-Typen (Khronos-Proposal, bestätigt)

Für eine Compute-DGC-Sequenz nutzbar:

| Token-Type | Zweck |
|---|---|
| `TOKEN_TYPE_EXECUTION_SET_EXT` | Pipeline-Switch (32-bit Index in den IndirectExecutionSet) |
| `TOKEN_TYPE_PUSH_CONSTANT_EXT` | Push-Constants pro Sequenz |
| `TOKEN_TYPE_SEQUENCE_INDEX_EXT` | Schreibt Sequenz-Index in PC (selten nötig) |
| `TOKEN_TYPE_DISPATCH_EXT` | Compute-Dispatch (groupCountX/Y/Z) |

**Es gibt KEINEN Barrier-Token.** Die Spec bestätigt: Innerhalb einer
Sequenz sind keine Synchronization-Operations möglich. Intra-Sequenz
sind weder Memory-Barriers noch Pipeline-Barriers ausdrückbar.

### Sequenz-Struktur (kritisch)

Die Spec sagt: jede Sequenz hat genau **EIN DISPATCH-Token**. Die
homogene Struktur (gleiches Layout für alle Sequenzen) erlaubt mehrere
state-changing Tokens pro Sequenz, aber nur einen Dispatch:

```
Sequence-Layout (Compute):
  [EXECUTION_SET, PUSH_CONSTANT, …, PUSH_CONSTANT, DISPATCH]
   └─ optional ─┘└──────── 0 oder mehr ─────────┘└─ pflicht ┘
```

Daraus folgt für unseren Forward-Pass: jeder einzelne Dispatch
(RMSNorm, GEMV, RoPE, …) ist eine eigene Sequenz im Indirect-Buffer.

### Inter-Sequenz-Synchronization (auch kritisch)

> The specification does not provide explicit ordering guarantees
> between adjacent sequences.

Ohne `VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT`
sind Sequenzen "may be serialized" — aber **garantiert ist es nicht**.
Mit dem Flag dürfen sie explizit out-of-order laufen.

Konsequenz für VulkanForge:
* Sequenzen mit RAW-Abhängigkeit (z.B. RMSNorm → GEMV) **müssen** in
  separaten `vkCmdExecuteGeneratedCommandsEXT`-Aufrufen liegen, mit
  Pipeline-Barrier dazwischen.
* Sequenzen ohne Abhängigkeit (z.B. GEMV-Q, GEMV-K, GEMV-V) können in
  *einem* Execute zusammengefasst werden.

### Preprocessing

Die Spec macht Preprocessing **optional**:
```c
vkCmdExecuteGeneratedCommandsEXT(cmd, isPreprocessed = VK_FALSE, &info)
```
Wenn `isPreprocessed = VK_FALSE`, "any necessary processing will be
performed as part of this command." Wir können also für den Smoke-Test
ohne separaten Preprocess-Pass starten.

### Memory-Requirements

`VkGeneratedCommandsMemoryRequirementsInfoEXT` +
`vkGetGeneratedCommandsMemoryRequirementsEXT` liefern den Scratch-Bedarf
für eine konkrete (Layout, MaxSequenceCount, MaxDrawCount)-Kombination.
Pflicht-Allokation einer DEVICE_LOCAL Buffer mit
`STORAGE_BUFFER_BIT | INDIRECT_BUFFER_BIT` für den Preprocess-Output —
bei `isPreprocessed=VK_FALSE` immer noch nötig (Driver schreibt dort
intern).

### Feature-Bit

`VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT.deviceGeneratedCommands
= VK_TRUE` muss bei Device-Create im pNext stehen, sonst
`VK_ERROR_FEATURE_NOT_PRESENT`. Die optionale
`dynamicGeneratedPipelineLayout` brauchen wir nicht (wir kennen unsere
Pipeline-Layouts beim Setup).

---

## 1.3 — ash-Crate-Support

```
$ cargo tree -p ash
ash v0.38.0+1.3.281

$ grep -rE "GeneratedCommandsInfoEXT|IndirectCommandsLayoutEXT|deviceGeneratedCommands" \
       ~/.cargo/registry/src/index.crates.io-*/ash-0.38.0+1.3.281/
(keine Treffer)

$ ls ~/.cargo/registry/src/.../ash-0.38.0+1.3.281/src/extensions/nv/
device_generated_commands_compute.rs   ← nur die NV-Variante!
```

ash 0.38 (basiert auf Vulkan-Headers 1.3.281, ~Juni 2024) hat **nur die
NV-Variante** — die EXT-Promotion landete in den Headers ~1.3.286
(Q3 2024). Die NV-Structs sind ABI-inkompatibel zu EXT (anderer
sType, anderes Token-Enum-Layout, andere Funktionsnamen).

**Konsequenz**: Wir brauchen manuelle FFI-Bindings für EXT-DGC. Das ist
in CLAUDE.md / unserer Memory bereits als akzeptabler Pfad markiert
(ROCmForge: 19 HIP-Funktionen manuell). Aufwand:
* ~10 Struct-Definitionen mit `#[repr(C)]`
* ~6 Funktions-Pointer via `device.get_device_proc_addr`
* ein paar Enum-Konstanten
* geschätzt 150-200 LoC, ~2-3 h Arbeit für ein cleanes Modul

---

## Beantwortung der Eingangsfragen

| Frage | Antwort |
|---|---|
| Unterstützt DGC Compute-Dispatches laut Spec? | **JA** — `TOKEN_TYPE_DISPATCH_EXT`, `SHADER_STAGE_COMPUTE_BIT` |
| Hat RADV eine vollständige Implementierung? | **JA, exposed seit Mesa 24.3 / hier 26.0.5; Compute Pipeline-Binding bestätigt** |
| Hat ash die nötigen Typen? | **NEIN — nur NV-Variante. EXT muss manuell gebunden werden.** |
| Sind Barriers innerhalb DGC möglich? | **NEIN — keine Barrier-Tokens. Jede RAW-Dependency = neuer Execute-Call.** |
| Empfehlung: Weiter? | **Vorsichtig GO mit angepasster Erwartung — siehe unten.** |

---

## Implikation für die Performance-Erwartung

Die Original-Erwartung im Prompt war:
> 250 vkCmdDispatch (3.3 ms) → 1 vkCmdExecuteGeneratedCommands (~0.2 ms)

Die Spec-Studie zeigt: Wir können **keine** 250 Dispatches in einen
einzigen Execute-Call legen, weil:

1. RAW-Abhängigkeiten (RMSNorm → GEMV → Attention → …) brauchen
   Pipeline-Barriers.
2. Pipeline-Barriers sind nur **zwischen** Execute-Calls möglich,
   nicht innerhalb einer Sequenz oder eines Layouts.

Realistische Aufteilung für einen Qwen3-8B-Layer (gegated auf
`has_qk_norm=true`):

```
1.  RMSNorm-attn         (1 dispatch)              ┐
2.  GEMV Q/K/V           (3 parallel)              │
3.  Q-norm + K-norm      (2 parallel)              │
4.  RoPE Q/K             (2 parallel)              │  ≈ 13 Execute-Calls
5.  KV-Write             (1 transfer-cmd-copy)     │   pro Layer
6.  Attention            (1 dispatch)              │
7.  GEMV O               (1)                       │
8.  Add residual         (1)                       │
9.  RMSNorm-ffn          (1)                       │
10. Gate + Up            (2 parallel)              │
11. SiLU + Mul           (1-2)                     │
12. GEMV Down            (1)                       │
13. Add residual         (1)                       ┘
```

→ **~13 ExecuteGeneratedCommands × 36 Layer ≈ 470 Execute-Calls** pro
Forward.

Vergleich mit dem Direct-Path:
* Direct-Path heute: ~250 Dispatches × 3 Host-Calls (Bind + PC + Dispatch) ≈ **750 Host-Calls**
* DGC realistisch: **470 Execute-Calls** (Bind + PC bytes sind im Indirect-Buffer)

**~37 % Reduktion bei den Host-Calls**, nicht 99 %. Wenn das CPU-seitige
3.3 ms Dispatch-Overhead grob proportional zur Host-Call-Zahl ist,
landet DGC bei ~2.1 ms statt 0.2 ms. Immer noch eine Verbesserung,
aber **eine andere Größenordnung als der Prompt erwartete**.

Zusätzlicher Benefit, der das Bild aufwerten kann: der Indirect-Buffer
kann *vorab* gefüllt werden und über mehrere Forward-Passes
wiederverwendet werden, solange sich Pipeline-Set, Buffer-Bindings und
Push-Constants nicht ändern. Wir müssten messen.

---

## Bekannte Risiken

1. **RADV-Compute-DGC ist relativ neu (~18 Monate).** Graphics-DGC
   wird durch vkd3d-proton hart getestet (DX12 Indirect-Argument-
   Buffers); reine Compute-Pfade wahrscheinlich weniger. Wir sind
   möglicherweise einer der ersten Real-World-Workloads.

2. **maxIndirectCommandsTokenCount = 128.** Reicht problemlos für
   unsere Sequenzen (max ~5 Tokens pro Sequenz: EXEC_SET + 2-3 PC + DISPATCH).

3. **Push-Constant-Größen.** Unsere PC-Structs sind teils > 64 B
   (RoPE: 92 B, FlashAttn: 60 B). Die Spec verlangt PC-Token-Größe
   ≤ `maxIndirectCommandsTokenOffset = 2047` — uns reicht das, aber
   wir müssen pro Pipeline mehrere PC-Tokens layern, falls ein
   einzelner Token die `pushConstantRangeSize` der Pipeline nicht
   abdeckt.

4. **Validation-Layer-Lücken.** Bei DGC werden viele Konsistenz-Checks
   nicht abgedeckt. Falsche Buffer-Inhalte → falscher Output, kein
   Layer-Error. Wir validieren Output-Bit-Exact gegen Direct-Path.

5. **Memory-Allocation für Preprocess-Buffer.** RADV-spezifisch:
   wahrscheinlich DEVICE_LOCAL, alignment evtl. mehrere KiB. Wir
   müssen nach `vkGetGeneratedCommandsMemoryRequirementsEXT` den
   tatsächlichen Bedarf abfragen, nicht raten.

---

## GO / NO-GO an dieser Stelle

**Empfehlung: vorsichtig GO** — aber mit angepassten Erwartungen:

* DGC funktioniert auf der API-Ebene bei uns (Driver bestätigt).
* Die ursprüngliche 16× Speedup-Schätzung (3.3 ms → 0.2 ms) war zu
  optimistisch — realistisch sind ~30-50 % Reduktion beim CPU-Dispatch-
  Overhead, weil Barriers außerhalb der DGC-Sequenz bleiben müssen.
* Die NV-only ash-Bindings zwingen uns zu manuellem FFI; das ist Aufwand
  aber kein Showstopper.
* **Phase 5A-2 (Full Forward via DGC) wäre ein viel kleinerer Gewinn
  als geplant** — möglicherweise 1.5-2 tok/s Decode statt der erhofften
  10-15 tok/s.

Mögliche Alternativen, die bei dieser Erwartungsanpassung
attraktiver werden könnten:

* **Weg 2 — Command-Buffer-Reuse**: Wir bauen *einmal* einen Command
  Buffer pro (seq_len, position-Klasse) und reusen ihn über Tokens.
  Push-Constants werden nicht über `vkCmdPushConstants` sondern
  über einen Uniform-Buffer mit per-Token-Offset gebunden. Resultat:
  **0 ms Recording-Overhead** ab dem zweiten Token. Aufwand: ~2 Tage,
  keine neue Extension nötig, kein FFI-Manual-Work.

* **Weg 3 — Submission-Reduzierung**: Aktuell submitten wir pro Token
  ein Mal. Wenn wir 4-8 Tokens batchen und EINE Submission machen,
  schmelzen 4-8 × 3.3 ms auf ~3.3 ms. Aufwand: ~1 Tag.

---

## Fragen an den User vor Coding (Schritte 2-4)

1. **Akzeptiere ich die Erwartungsanpassung?** ~30-50 % Dispatch-Overhead-
   Reduktion via DGC statt der ursprünglich anvisierten ~95 %.

2. **Vor dem Smoke-Test in Schritt 2 weitermachen, oder doch erst
   Weg 2 (Command-Buffer-Reuse) als günstigeren Vergleichswert
   evaluieren?** Weg 2 wäre etwa 2 Tage Implementierung für vermutlich
   ähnliche Endnutzer-Performance.

3. Falls weiter mit DGC: **manuelle FFI-Bindings im Crate (`src/backend/vulkan/dgc.rs`) OK?**
   Das ist die einzige Option — ash 0.38 hat es nicht.

---

## Wenn weiter, dann mit folgendem Plan

* **Schritt 2** (Smoke): `examples/dgc_smoke.rs` + `src/backend/vulkan/dgc.rs`
  mit manuellen FFI-Bindings. Erste Ziel-Validierung: SiLU via DGC vs
  Direct-Path bit-exact.
* **Schritt 3** (Multi-Dispatch): 3 unabhängige SiLU-Dispatches in einer
  Sequenz, ohne Barrier zwischen ihnen — testen ob RADV out-of-order
  ausführt oder serialisiert.
* **Schritt 4** (1-Layer): RMSNorm + GEMV-Q + GEMV-K + GEMV-V als eine
  DGC-Sequenz mit Pipeline-Switch via EXECUTION_SET-Token. Output bit-
  exact vergleichen.
* **GO/NO-GO** dann am Ende mit echten Mess-Daten statt Spekulation.

---

## Status

**Schritt 1 abgeschlossen. STOP für User-Entscheidung.**
Keine Code-Änderungen, keine Commits. Tests-Stand: 41/41 (Phase 4D-Stand).
