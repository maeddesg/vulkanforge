# Qwen3.5 / Qwen3.6 (`qwen35`) Architektur-Analyse

**Modell-Sample:** `~/models/Qwen3.6-27B-MTP-Q3_K_S.gguf` (12.6 GB, mit MTP-Block)
**Referenz-Implementation:** `~/tmp/llama.cpp/src/models/qwen35.cpp` (627 LOC, dense Variante)
**llama.cpp Vulkan-Shader:** `gated_delta_net.comp`, `ssm_conv.comp`, `ssm_scan.comp`
**Sprint:** Sprint A (Analyse, kein Code)
**Status:** Vorbereitung v0.4.6 Qwen3.6-Support

> **Hinweis:** Das ohne-MTP GGUF `~/models/Qwen3.6-27B-Q3_K_S.gguf`
> ist auf dem System **nicht** vorhanden — nur die MTP-Variante.
> Die MTP-spezifischen Tensoren sind durch ihre `blk.N.nextn.*`
> Namen klar abgrenzbar, ein Diff gegen die Non-MTP-GGUF ist daher
> nicht zwingend nötig. Wenn später für eine v0.4.6-Verifizierung
> die Non-MTP-Variante gebraucht wird, kann sie auf HuggingFace
> nachgeladen werden.

---

## 1. Metadaten

Aus `gguf_dump.py --no-tensors`:

| Key                                          | Wert                  | Bedeutung                                     |
|----------------------------------------------|----------------------:|-----------------------------------------------|
| `general.architecture`                       | `qwen35`              | Dense-Variante (MoE-Pendant: `qwen35moe`)     |
| `qwen35.block_count`                         | 65                    | 64 trunk + 1 MTP nextn block                  |
| `qwen35.context_length`                      | 262 144               | 256 K context                                 |
| `qwen35.embedding_length`                    | 5 120                 | hidden                                        |
| `qwen35.feed_forward_length`                 | 17 408                | FFN intermediate                              |
| `qwen35.attention.head_count`                | 24                    | Q heads                                       |
| `qwen35.attention.head_count_kv`             | 4                     | KV heads (GQA 6:1)                            |
| `qwen35.attention.key_length`                | 256                   | head_k (≠ hidden / n_head ⇒ 24·256=6144)      |
| `qwen35.attention.value_length`              | 256                   | head_v                                        |
| `qwen35.attention.layer_norm_rms_epsilon`    | 1.0e-6                | RMSNorm eps                                   |
| `qwen35.rope.dimension_count`                | 64                    | Rot dim per head (Wikipedia mRoPE)            |
| `qwen35.rope.dimension_sections`             | `[11, 11, 10, 0]`     | mRoPE-Sektionen für mm/Vision                 |
| `qwen35.rope.freq_base`                      | 1.0e7                 | RoPE θ_base (höher als Llama-typisch 1e4)     |
| `qwen35.ssm.conv_kernel`                     | 4                     | conv1d kernel size                            |
| `qwen35.ssm.state_size`                      | 128                   | head_k_dim, head_v_dim für Linear-Attn        |
| `qwen35.ssm.group_count`                     | 16                    | num_k_heads (Linear-Attn-Gruppen)             |
| `qwen35.ssm.time_step_rank`                  | 48                    | num_v_heads                                   |
| `qwen35.ssm.inner_size`                      | 6 144                 | d_inner (= n_head × n_embd_head = 24·256)     |
| `qwen35.full_attention_interval`             | 4                     | Pattern: jede 4. Layer = volle Attention      |
| `qwen35.nextn_predict_layers`                | 1                     | 1 MTP-Draft-Block                             |
| `general.file_type`                          | 11                    | GGUF Q3_K_S                                   |
| `tokenizer.ggml.model` / `.pre`              | `gpt2` / `qwen35`     | Custom Qwen3.5 BPE pre-regex                  |
| `tokenizer.ggml.tokens` (Anzahl)             | 248 320               | Vokabular incl. multimodal-Tokens             |
| `eos_token_id` / `bos_token_id` / `pad`      | 248 046 / 248 044 / 248 055 |                                          |
| `general.tags`                               | `['unsloth', 'image-text-to-text']` | Multimodal-Hinweis              |

**Beobachtungen:**

- `key_length = value_length = 256`, `n_head = 24` → `n_embd_head·n_head = 6144 ≠ 5120 = n_embd`. Die Q/K-Projektion expandiert also Hidden → 6144 für Q (plus Gate). Q+Gate kommen aus einem **gemeinsamen `attn_qkv`-ähnlichen Tensor** (siehe §3.1).
- `ssm.inner_size = 6144 = n_head_k · key_length` → die Linear-Attention läuft im gleichen 6144er Raum wie die volle Attention.
- `rope.dimension_count = 64` ist die per-head-Rotationsdimension (`<` 256 = head_dim ⇒ Partial-RoPE).
- `rope.dimension_sections = [11, 11, 10, 0]` Summe 32 → mRoPE mit 4 Sektionen (text/x/y/z), nur 32 von 64 rot-dims werden tatsächlich genutzt. Inferenz auf reinem Text aktiviert die ersten 32 dims (Text-Sektion = 11+11+10).
- `rope.freq_base = 1e7` höher als bei Standard-Llama (1e4 / 5e5). Konsequenz: längere Wellenlängen, passt zum 256K context.
- **Multimodal-Modell** (`image-text-to-text` tag), aber das GGUF enthält **nur** das LLM-Backbone — keine Vision-Tensor-Familie (kein `vision.*` Key, kein `patch_embd.weight` etc.).

---

## 2. Tensor-Inventar

`gguf_ex_read_0: n_tensors = 866`.

### 2.1 Globale Tensoren (3 Stück)

| Name                  | Quant-Bytes | Implikation                  |
|-----------------------|------------:|------------------------------|
| `token_embd.weight`   | 546 304 000 | Q3_K_S `[n_embd=5120, vocab=248320]` |
| `output.weight`       | 1 042 944 000 | Q3_K_S — eigene lm_head (kein tied embedding) |
| `output_norm.weight`  | 20 480     | FP16 vector `[n_embd]`       |

### 2.2 Per-Layer-Tensoren

Pattern aus `cut -d'|' -f1 ... | sed 's/blk\.[0-9]*/blk.N/' | sort | uniq -c`:

| Tensor                            | Anzahl Layer | Zweck                                                |
|-----------------------------------|-------------:|------------------------------------------------------|
| `blk.N.attn_norm.weight`          | 65           | Pre-attn RMSNorm                                     |
| `blk.N.post_attention_norm.weight`| 65           | Pre-FFN RMSNorm                                      |
| `blk.N.ffn_gate.weight`           | 65           | Dense FFN gate proj (SiLU + ×up)                     |
| `blk.N.ffn_up.weight`             | 65           | Dense FFN up proj                                    |
| `blk.N.ffn_down.weight`           | 65           | Dense FFN down proj                                  |
| `blk.N.attn_qkv.weight`           | 48           | **Linear-Attn**: fused Q+K+V projection              |
| `blk.N.attn_gate.weight`          | 48           | **Linear-Attn**: gating projection (z)               |
| `blk.N.ssm_a`                     | 48           | A-log scalar (FP32, 48 floats per layer)             |
| `blk.N.ssm_alpha.weight`          | 48           | α projection `[n_embd, num_v_heads=48]`              |
| `blk.N.ssm_beta.weight`           | 48           | β projection `[n_embd, num_v_heads=48]`              |
| `blk.N.ssm_conv1d.weight`         | 48           | Conv1d kernel `[ssm_d_conv=4, conv_dim]`             |
| `blk.N.ssm_dt.bias`               | 48           | dt bias (FP32, 48 floats per layer)                  |
| `blk.N.ssm_norm.weight`           | 48           | Gated-Norm (`build_norm_gated`) weight, `[head_v_dim=128]` |
| `blk.N.ssm_out.weight`            | 48           | Output projection `[value_dim, n_embd]`              |
| `blk.N.attn_q.weight`             | 17           | **Full-Attn**: Q projection (incl. Gate, `n_embd_head·n_head·2 = 12288`) |
| `blk.N.attn_k.weight`             | 17           | Full-Attn: K projection                              |
| `blk.N.attn_v.weight`             | 17           | Full-Attn: V projection                              |
| `blk.N.attn_output.weight`        | 17           | Full-Attn: output projection                         |
| `blk.N.attn_q_norm.weight`        | 17           | Per-head Q RMSNorm `[head_dim=256]`                  |
| `blk.N.attn_k_norm.weight`        | 17           | Per-head K RMSNorm                                   |

### 2.3 MTP-spezifische Tensoren (nur Block 64)

| Tensor                                  | Bytes      | Zweck                                                                 |
|-----------------------------------------|-----------:|-----------------------------------------------------------------------|
| `blk.64.nextn.eh_proj.weight`           | 55 705 600 | Concat-Projection `[2·n_embd=10240 → n_embd=5120]`                    |
| `blk.64.nextn.enorm.weight`             | 20 480     | RMSNorm auf token-embedding-Input                                     |
| `blk.64.nextn.hnorm.weight`             | 20 480     | RMSNorm auf hidden-state-Input                                        |
| `blk.64.nextn.shared_head_norm.weight`  | 20 480     | Final RMSNorm vor MTP lm_head                                         |
| **(optional, fehlt im GGUF)**           |            | `blk.64.nextn.embed_tokens.weight` (sonst `model.tok_embd` reused)    |
| **(optional, fehlt im GGUF)**           |            | `blk.64.nextn.shared_head_head.weight` (sonst `model.output` reused)  |

Block 64 trägt **außerdem alle Full-Attn-Tensoren** (`attn_q/k/v/output/q_norm/k_norm`) plus die Dense-FFN-Tensoren. Die MTP-Logik (`graph_mtp` in qwen35.cpp:487-627) ist eine zusätzliche Schicht **mit eigenem Embedding-Concat-Vorlauf** vor dem klassischen Attn+FFN. Im Trunk-Forward wird Block 64 **nicht** ausgeführt (qwen35.cpp:162 `n_transformer_layers = n_layer - nextn_predict_layers = 64`). Der MTP-Pfad wird separat als `LLM_GRAPH_TYPE_DECODER_MTP` aufgerufen.

---

## 3. Layer-Pattern

### 3.1 Verteilung (verifiziert per `grep + sort`)

| Block-Index | Layer-Typ                | Anzahl |
|-------------|--------------------------|-------:|
| 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 | **Full Attention** (GQA + mRoPE + Q-Gate) | 16 |
| 0,1,2, 4,5,6, 8,9,10, …, 60,61,62                            | **Linear Attention / Gated Delta Net**     | 48 |
| 64                                                            | Full Attention + **MTP nextn head**       | 1  |

**Pattern-Formel** (aus qwen35.cpp:25-28):

```c
hparams.recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % full_attn_interval != 0);
// n_main = n_layer - nextn_predict_layers = 64
// full_attn_interval = 4
```

⇒ Block `i` ist **recurrent (Linear-Attn)** wenn `i < 64 && (i+1) % 4 ≠ 0`.
⇒ Full-Attn wenn `(i+1) % 4 == 0` (also `i ∈ {3, 7, …, 63}`) oder `i ≥ 64` (MTP).

Im Vergleich zu Gemma-4 (SSSF mit Sliding-Window-Attn auf "S", Full-Attn auf "F", `full_attention_interval = 5` analog): **gleiches Muster, andere Periode (4 statt 5) und Linear-Attn statt Sliding-Window-Attn auf den S-Layern**.

### 3.2 Gemeinsame Komponenten (in beiden Layer-Typen)

Beide Layer-Typen teilen sich:
- `attn_norm` (Pre-Sub-Block-1 RMSNorm)
- `post_attention_norm` (Pre-Sub-Block-2 RMSNorm — analog Gemma-4)
- Dense `ffn_{gate, up, down}` mit SwiGLU/SiLU + parallel
- Residuale Verbindungen: `attn_out + inpSA`, `ffn_out + ffn_residual`

Damit ist **alles außer der Attention-Sub-Block** für Full-Attn und Linear-Attn **identisch**.

---

## 4. MTP-Block (Detail)

`graph_mtp` (qwen35.cpp:487-627). Asserts in der Implementation: `nextn_predict_layers > 0` und `== 1`.

```
Eingaben:
  inp.tokens     : [n_tokens]  Draft-Token-IDs (int32)
  inp.embd       : [n_embd, n_tokens]  Vorheriger Hidden State (FP32)

Ablauf:
  1. tok_embd     = ggml_get_rows(layer.nextn.embed_tokens or model.tok_embd, inp.tokens)
  2. h_norm       = RMSNorm(inp.embd,  layer.nextn.hnorm)
  3. e_norm       = RMSNorm(tok_embd,  layer.nextn.enorm)
  4. concat       = concat[dim=0](e_norm, h_norm)        // [2*n_embd, n_tokens]
  5. cur          = layer.nextn.eh_proj · concat         // [n_embd, n_tokens]
  6. (cur ist der "embed_for_block" — von hier ab ein normaler Full-Attn-Block:)
  7. cur = RMSNorm(cur, layer.attn_norm)
  8. Q/K/V via wq/wk/wv (gleiche Tensoren wie ein regulärer Full-Attn-Block)
  9. Q-Gate split (siehe §5.1), Q/K-Norm, mRoPE, Attention
  10. attn_out × sigmoid(gate)
  11. wo, residual, post-norm, FFN, residual
  12. shared_head_norm  (nextn.shared_head_norm wenn vorhanden, sonst output_norm)
  13. lm_head           (nextn.shared_head_head wenn vorhanden, sonst output)
  14. logits → Sampler im AR-Draft-Loop
```

Im aktuellen GGUF fehlen `nextn.embed_tokens` und `nextn.shared_head_head`; der Code-Pfad fällt auf `model.tok_embd` und `model.output` zurück. Das **shared LM-Head + shared Embedding** ist also der Default und spart 1.6 GB.

---

## 5. Compute-Logik

### 5.1 Full-Attention-Layer (`build_layer_attn`, qwen35.cpp:256-335)

```
Qcur_full = wq · cur                           # [n_embd_head·n_head·2 = 12288, n_tokens]
Qcur      = Qcur_full[0       : 6144]          # view, stride-2
gate      = Qcur_full[6144    : 12288]
Qcur      = RMSNorm(Qcur, attn_q_norm)         # per-head, head_dim=256

Kcur = wk · cur                                # [n_head_kv·head_k = 1024, n_tokens]
Kcur = reshape[head_dim, n_head_kv, n_tokens]
Kcur = RMSNorm(Kcur, attn_k_norm)

Vcur = wv · cur                                # [n_head_kv·head_v = 1024, n_tokens]
Vcur = reshape[head_dim, n_head_kv, n_tokens]

Qcur, Kcur ← ggml_rope_multi(..., sections=[11,11,10,0])   # mRoPE
                                                            # (1D bei reinen Text-Tokens)

attn = build_attn(Qcur, Kcur, Vcur, kq_scale = 1/sqrt(256))
attn = attn × sigmoid(gate)                    # gated attention output
out  = wo · attn
```

**Neu vs. VF heute:**

- **Q-Gate-Split**: Q-projection produziert Q+Gate fused (×2 output-dim). Gate ist `sigmoid(gate) · attn_out`. **Neue Operation**, kein bisheriger VF-Layer-Typ macht das.
- **Per-Head Q/K-RMSNorm** (Qwen3 ähnlich, aber mit unterschiedlichem head_dim).
- **mRoPE** (`ggml_rope_multi` mit `sections[4]`). Bei reinen Text-Tokens entspricht das im Endeffekt 1D-RoPE auf den ersten `sum(sections[0:3]) = 32` Dimensionen, das letzte (z-/temporal-)Segment ist Null. Für Phase-1-Text-only-Support ist das **vereinfachbar zu einem Partial-RoPE mit n_rot=32**.

### 5.2 Linear-Attention-Layer (`build_layer_attn_linear`, qwen35.cpp:337-469)

```
# Projektionen
qkv_mixed = wqkv · cur                          # [conv_dim = 2·k_dim + v_dim, n_tokens]
                                                # k_dim = ssm_state·n_group = 128·16 = 2048
                                                # v_dim = ssm_state·dt_rank = 128·48 = 6144
                                                # conv_dim = 2·2048 + 6144 = 10240
z          = wqkv_gate · cur                    # [v_dim = 6144, n_tokens]  → SiLU-Gate

beta       = ssm_beta · cur                     # [num_v_heads=48, n_tokens]
beta       = sigmoid(beta)

alpha      = ssm_alpha · cur                    # [num_v_heads=48, n_tokens]
alpha     += ssm_dt.bias                        # bias-add
alpha      = softplus(alpha)
gate_decay = alpha · ssm_a                      # ssm_a is FP32 scalar-per-head, negative

# Conv1d update (state-update conv with kernel=4)
conv_state = (load recurrent conv_state for sequence)
conv_out   = ssm_conv(conv_input = qkv_mixed, conv_kernel = ssm_conv1d)
conv_out   = silu(conv_out)
# split conv_out → q_conv [k_dim], k_conv [k_dim], v_conv [v_dim]
q_conv     = L2Norm(q_conv)
k_conv     = L2Norm(k_conv)

# Recurrent state load
state      = (load recurrent ssm_state for sequence) [head_v, head_v, num_v_heads, n_seqs]

# Gated Delta Net — the actual recurrence
output     = build_recurrent_attn(q_conv, k_conv, v_conv, gate_decay, beta, state)
                                                # uses gated_delta_net.comp shader on Vulkan

# Gated norm (z-gated): RMSNorm(output, ssm_norm) · SiLU(z)
out_gnorm  = build_norm_gated(output, ssm_norm, z_2d)
                                                # = RMSNorm(out) · SiLU(z)

cur        = ssm_out · out_gnorm                # [n_embd, n_tokens]
```

**Recurrent State** (pro Sequence persistent über Tokens):
- `conv_state` (Shape `[ssm_d_conv-1=3, conv_channels=10240]` pro Sequence) — letzte 3 Conv-Inputs.
- `ssm_state` (Shape `[head_v_dim, head_v_dim, num_v_heads, n_seqs] = [128, 128, 48, n_seqs]`)
  — pro V-head eine `[128×128]` Matrix. Pro Sequence: **48 · 128 · 128 · 4 bytes = 3 MB FP32**, **× 48 SSM-Layer = 144 MB recurrent state pro Sequence**.

**Vulkan-Kernel in llama.cpp:**

| GGML-Op                       | llama.cpp Vulkan-Shader              | LOC | Aufgabe                       |
|-------------------------------|--------------------------------------|----:|-------------------------------|
| `ggml_ssm_conv`               | `ssm_conv.comp`                      | 50  | Conv1d-State-Update           |
| `build_recurrent_attn` (GDN)  | `gated_delta_net.comp`               | 190 | Gated-Delta-Net Recurrenz     |
| `ggml_softplus`               | (in `ggml-vulkan` als unary op)      |     | `log(1+exp(x))`               |
| `ggml_l2_norm`                | (existing unary)                     |     | `x / sqrt(sum(x²)+eps)`       |

`gated_delta_net.comp` nimmt Q, K, V, G (gate) als Inputs und einen persistenten State-Buffer. Subgroup-clustered/arithmetic für die Reduktion. State-Update + Output in einem Shader.

### 5.3 Dense FFN (beide Layer-Typen identisch)

Standard SwiGLU / SiLU-parallel:

```
ffn_out = ffn_down · (silu(ffn_gate · cur) ⊙ (ffn_up · cur))
```

⇒ Drei GEMVs/GEMMs + SiLU + elementwise mul. **Genau wie Qwen3-8B oder Llama-3.1-8B** — VF kann die existierenden `step_ffn_*` Pfade 1:1 wiederverwenden.

---

## 6. VulkanForge Implementation-Plan

### 6.1 Was wir wiederverwenden können

| Komponente                | Bestehend in VF                          |
|---------------------------|------------------------------------------|
| Dense FFN (SiLU)          | `executor/ffn.rs` step_ffn_* unverändert |
| GQA Attention             | `executor/attention.rs` step_q_norm_rope + step_attn_kqv + step_attn_softmax + step_attn_output  |
| RMSNorm                   | `run_rms_norm`                           |
| K/V-Cache (für Full-Attn-Layer) | KV-Cache infra; **nur** für die 17 Full-Attn-Layer (16 trunk + 1 MTP) — die Linear-Attn-Layer brauchen **keinen** KV-Cache. |
| Quant-Loader (Q3_K_S)     | `loader.rs` GGUF-Pfad                    |
| FP8 KV-Cache              | `VULKANFORGE_KV_FP8=1` würde nur auf den 17 Full-Attn-Layern wirken — Ersparnis kleiner als bei reinen Attn-Modellen. |

### 6.2 Was neu gebaut werden muss

**Tier 1: Architektur-Plumbing (notwendig, klein)**

- `src/backend/vulkan/forward/arch/qwen35.rs` — Layer-Builder, mit Tabelle ob Layer `Recurrent` oder `Attention`.
- `LayerStep` Enum-Erweiterungen (siehe §6.3) in `layer_plan.rs`.
- `cfg.qwen35.is_some()` flag in `state.rs`; Loader liest `qwen35.*` Metadaten.
- mRoPE-Variante: entweder ein neuer `RopeKind::Multi { sections: [u32; 4] }` Wert oder, für Phase-1, **direkt Partial-RoPE mit `n_rot=32`** als Vereinfachung (gibt bei reinem Text identische Outputs).
- **Q-Gate-Split**: neue Operation `Step::AttnQGateSplit` — splittet das `wq`-Output `[n_embd_head·n_head·2 = 12288]` in Q `[6144]` + Gate `[6144]` und appliziert `sigmoid(gate)` nachträglich auf den Attention-Output.

**Tier 2: Linear-Attention (groß, modellspezifisch)**

- **Shader 1: `ssm_conv1d.comp`** — Conv1d mit Kernel=4 + State-Update.
  Eingang: `[conv_channels = 10240, n_tokens]`. Ausgang: Conv-Output + neuer State.
- **Shader 2: `gated_delta_net.comp`** — Port aus llama.cpp (190 LOC). Eingang: Q/K/V/G + persistenter SSM-State. Ausgang: per-token V mit aktualisiertem State.
  Spec-constants: `S_V=128`, `KDA`, subgroup_size, lanes-per-column. Sollte sich relativ direkt portieren lassen.
- **Shader 3: `softplus.comp`** — `log(1+exp(x))` als unary kernel (alternativ in einen existierenden bias-add-norm-kernel inlinen).
- **Shader 4: `l2_norm.comp`** — `x / sqrt(sum(x²)+eps)`. Sehr ähnlich zu RMSNorm aber **ohne** per-channel weight. Kann eventuell durch RMSNorm mit `weight=1` ersetzt werden, mit Verlust einer FMA.
- **Shader 5: `norm_gated.comp`** — `RMSNorm(x) · SiLU(z)` fused. Optional als zwei existierende Kernel + extra `eltwise_mul` simulierbar, aber dann mit extra Allocation.

**Persistenter State** (neue Forward-Felder in `state.rs`):

```
pub conv_states: Vec<GpuBuffer>,   // 48 layers × [3, 10240] FP32 = 5.9 MB total
pub ssm_states:  Vec<GpuBuffer>,   // 48 layers × [128, 128, 48] FP32 = 144 MB total
```

Beide pro Sequence einmal, **persistent über Decode-Tokens** (Reset bei neuer Sequence). Das ist VF-untypisch — die KV-Cache-Infrastruktur ist auf "append per token" ausgelegt, hier brauchen wir "update in place per token".

**Tier 3: MTP-Draft-Head (optional für v0.4.6, kann v0.4.7 sein)**

- Separater Graph-Mode `LLM_GRAPH_TYPE_DECODER_MTP` (oder VF-Äquivalent).
- Eingang: vorheriger Hidden-State + Draft-Token-ID.
- Block-64-Tensoren laden (attn_q/k/v/output, ffn_*, nextn.eh_proj/enorm/hnorm/shared_head_norm).
- AR-Draft-Loop in `decode.rs` für Spec-Decoding (llama.cpp's `--spec-type draft-mtp` ergab Decode +29-63 % auf der MTP-GGUF; siehe `feedback_decode_dispatch_bound.md` + die llama.cpp-MTP-Bench-Notes).

### 6.3 LayerStep-Erweiterungen (Vorschlag)

In `layer_plan.rs` zu ergänzen (Coding-Standards §3 beachtet: generische Namen, parameterisiert wenn möglich):

```rust
pub enum LayerStep {
    // ... existing
    AttnQGateSplit { gate_dim: u32 },   // Qwen3.5/3.6: split fused QG, apply sigmoid(gate) on attn_out
    SsmConv1d { layer: u32 },            // Conv1d kernel=4 + state update
    GatedDeltaNet { layer: u32 },        // Recurrence (Q,K,V,G + state) → V_out
    NormGated { eps: f32 },              // RMSNorm(out) · SiLU(z) fused
    L2Norm { eps: f32 },                 // (optional, kann durch RMSNorm-weight=1 ersetzt werden)
}
```

`AttnQGateSplit` und die SSM-Steps gehen in den **gleichen `LayerStep`-Enum**, müssen aber im `BatchExec` UND `DecodeExec` implementiert werden (Coding-Standards §3.2 / §4.2; vgl. `feedback_layer_dispatch_paths.md`). Builder `build_qwen35_layer` in `arch/qwen35.rs` entscheidet pro Layer-Index ob Linear oder Attention (nach Formel aus §3.1).

### 6.4 Aufwandsschätzung

| Bereich                                                          | Geschätzte Sprints (à 2-4 h) |
|------------------------------------------------------------------|-----------------------------:|
| Arch-Plumbing (cfg, loader, layer_plan, builder)                 | 2                            |
| Q-Gate-Split + mRoPE-Simplification (Partial-RoPE n_rot=32)      | 1                            |
| Full-Attn-Layer end-to-end (17 Layer)                            | 1                            |
| `ssm_conv1d.comp` Port + recurrent buffer infra                  | 2                            |
| `gated_delta_net.comp` Port + spec-const tuning                  | 3                            |
| `norm_gated`, `l2_norm`, `softplus` Hilfs-Kernel                 | 1                            |
| Recurrent-State-Lifecycle (allocate, reset, per-decode-update)   | 2                            |
| Integration + 5-Prompt-Smoke + Coherence-Verifizierung           | 2                            |
| MTP-Draft-Head (optional, Phase 2)                               | +3                           |
| **Total v0.4.6 (ohne MTP)**                                      | **~14 Sprints**              |
| **Total v0.4.6 + MTP**                                           | **~17 Sprints**              |

### 6.5 Was uns hilft

- **llama.cpp hat alle drei Vulkan-Shader bereits** (`gated_delta_net.comp`, `ssm_conv.comp`, `ssm_scan.comp`). Direkter Port ist möglich.
- **Dense-FFN und Full-Attn sind ein "Standard"-Qwen3-Layer**, die existierende VF-Infra trägt.
- **MTP-Head ist ein voller Decode-Block + Embedding-Concat + lm_head**. Die LM-Head-Logik ist bereits da.
- **Recurrent State ist klein** (~150 MB FP32 für die 48 SSM-Layer pro Sequence) im Vergleich zum 12 GB Modell.

### 6.6 Was uns blockt / Risiken

- **Recurrent State Layout-Match**: llama.cpp's `ssm_state` ist `[head_v_dim, head_v_dim, num_v_heads, n_seqs]` mit Konvention "head_v_dim×head_v_dim matrix per V-head". VF hat keine entsprechende Buffer-Layout-Konvention — sauberes Mapping nötig.
- **mRoPE vs. Partial-RoPE**: Wenn wir Phase 1 mit `n_rot=32` Partial-RoPE simplifizieren und HF/llama.cpp mit echtem mRoPE rechnet, könnten **bei längeren Sequenzen Drift entstehen** (mRoPE-Sektionen sind für mm gedacht, aber das frequency-Layout der ersten 32 dims sollte bei Text identisch sein — verifizierungspflichtig).
- **conv_state-Update unter async-decode**: Sprint 52Q's `feedback_async_decode_dump_hazard.md` ist relevant. Recurrent buffers werden in jedem Token gelesen+geschrieben — Barrier-Disziplin ist kritisch (siehe Sprint 61G: nicht-isolierte FMAs vs. compute_barrier).
- **Batched Prefill (DEFAULT seit v0.5.6, opt-out `VF_QWEN35_BATCHED=0`)**: der batched-Prefill-Pfad (BatchExec) ist
  für qwen35 komplett gebaut, verifiziert und seit v0.5.6 Default (v0.5.5 shippte ihn opt-in; v0.5.6 fixte eine
  barrier-elision-Race). **Race-Fix:** der per-Token-conv-loop (`b_step_ssm_conv1d`) reuste einen 1-token-`conv_input`-
  Scratch über die N Token-Iterationen → `conv(t)`-Read ↔ `setup(t+1)`-Write = WAR; der elision-Dirty-Flag-Tracker
  modelliert nur Writes (RAW), nicht WAR → die elision-aware end-of-iteration-Barriere feuerte nicht → probabilistische
  run-to-run-Non-Determinismus auf near-tie-Prompts (NUR recurrent-Layer; full-attn kein conv → clean; via Layer-TYP-
  Bisect gepinnt). Fix = die eine end-of-iter-Barriere unkonditional (`compute_barrier`); targeted, KEIN global
  elision-disable (gemessen: Elision trägt 100% des Prefill-Speedups). Architektur = **per-Token-Reuse**: die
  seriellen Cross-Token-
  Cores (SSM-Conv `ssm_conv.comp` + GDN-Recurrence `gated_delta_net.comp`) laufen als N sequentielle Single-Token-
  Dispatches mit State-Carry (conv_state/ssm_state, wiederverwenden die verifizierten Decode-Bodies); die Projektionen
  (qkv/gate-z/beta/ssm_alpha/out-proj) werden batched GEMM (M=N) — dort sitzt der Speedup (4–11×, wächst mit Länge),
  da die Cross-Token-Cores ohnehin seriell sind. Verifikation pro Stage gegen den deterministischen Decode-Oracle:
  Cores bit-identisch, Full-Layer-Integration cos→1.0 @~1e-5 (N=2..32), E2E byte-identisch zu per-Token. Inter-Stage-
  Strides (V-in-conv_output Token-Stride = conv_channels=10240, GDN-V-Offset 4096, qrep/krep = head_k·H) sind die
  Risiko-Punkte — alle verifiziert. Default-Flip = separater owner-Call.
- **Gated-output → o-proj Barriere (BEHOBEN v0.5.4)**: Der Full-Attn gated output (`step_attn_gated_output`) schreibt `sigmoid_mul` **in-place** auf `attn_out`; das folgende `step_o_proj` liest `attn_out` per gemv. o-proj rief keine pre-read-`maybe_compute_barrier(&[attn_out])` und der Gate-Step hatte keine trailing-Barriere → unter `Imperative`+Elision (Default) las o-proj racy (RAW-Hazard), durch `head_dim=256` verbreitert → **bit-non-deterministischer Greedy-Decode** ("q36 differs run-to-run"). Fix = trailing `maybe_compute_barrier(&[attn_out])` im Decode-Gate-Step (spiegelt den Batch-Pfad `b_step_attn_gated_output`), Default-ON seit v0.5.4 (`VF_QWEN35_DEC_GATE_BARRIER=0` = Escape-Hatch). Lehre: jeder in-place-Write braucht eine pre-read-Barriere beim *nächsten Reader*, auch wenn der Reader in einem separaten Step lebt.
- **Build-Validierung**: kein non-MTP-GGUF auf dem System ⇒ wir testen anfangs nur die MTP-Variante (laden geht, MTP-Block einfach nicht ausführen; Trunk-Layers 0-63 reichen für Korrektheit).
- **Quant-Coverage**: Q3_K_S für `output.weight` (1 GB) — VF hat Q3_K bereits. Andere Q-Levels in den `ssm_*` Tensoren (siehe Bytes-Spalte) — `ssm_a` und `ssm_dt.bias` sind extrem klein (192 Bytes), wahrscheinlich FP32 oder F16. Genaue Quant-Type pro Tensor mit `gguf_dump.py` (mit `--tensors`) prüfen vor Implementierung.
- **mRoPE-Sections-Wert `[11, 11, 10, 0]`**: Die `0` an Position 3 deutet darauf hin, dass die "temporale" Sektion ungenutzt ist. Selbst wenn wir mRoPE später vollständig implementieren wollen, ist Phase 1 mit `n_rot=32` korrekt für Text-only-Inferenz.
- **Linear-Attn-Throughput**: Da 48 von 65 Layern Linear-Attn sind, dominieren die GDN-Shader die Performance-Charakteristik. Schlechter Port = signifikante Regression im Vergleich zu Qwen3-8B-Decode.

---

## 7. Pre-Check-Empfehlung vor Sprint B (Code)

Bevor irgendeine Zeile Code geschrieben wird:

1. **Quant-Type-Audit pro Tensor-Klasse**:
   `gguf_dump.py ~/models/Qwen3.6-27B-MTP-Q3_K_S.gguf` (ohne `--no-tensors`) und prüfen welche Tensoren in welchem Quant gespeichert sind. Insb. `ssm_*`, `attn_qkv`, `attn_gate`. **Hypothese**: 2D-Quant-Tensoren = Q3_K_S, kleine 1D-Tensoren (norms, ssm_a, ssm_dt.bias) = FP16/FP32.
2. **HF-Referenz**: Tokenizer- und mRoPE-Verhalten auf einem 27B-Sample reproduzieren (z. B. via `transformers` mit qwen35-Architektur, wenn die Klasse in `transformers >= 4.50` existiert) — Baseline-Tokens für 5-Prompt-Smoke generieren.
3. **GDN-Shader-Pre-Check**: `gated_delta_net.comp` ohne Modifikation gegen `glslc` (Mesa 26.1) → SPIR-V kompilieren, prüfen ob alle GLSL-Extensions verfügbar sind (`subgroup_basic`, `subgroup_clustered`, `subgroup_arithmetic` — alles bereits in VF genutzt).
4. **Non-MTP-GGUF beschaffen** (optional aber empfohlen): `huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q3_K_S.gguf` für saubere Trunk-Bring-up ohne MTP-Komplexität.

---

## 8. Zusammenfassung in zwei Sätzen

Qwen3.6-27B ist **kein Standard-Transformer**: es ist ein **Hybrid-Modell** mit 48 Linear-Attention (Gated Delta Net) Layern und 16+1 Full-Attention Layern in einem **SSSF-ähnlichen Muster** (jede 4. Layer ist Full-Attn), plus einem optionalen MTP-Draft-Block (1 Layer). Für VF-Support bringt llama.cpp die kompletten Vulkan-Shader (`gated_delta_net.comp`, `ssm_conv.comp`) bereits mit; der Aufwand liegt in **Recurrent-State-Infrastruktur**, **Q-Gate-Split**, **mRoPE-Vereinfachung** und **Layer-Routing** — geschätzt ~14 Sprints für v0.4.6 ohne MTP, ~17 mit.
