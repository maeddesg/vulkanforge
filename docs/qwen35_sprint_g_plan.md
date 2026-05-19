# Sprint G — Implementation Plan (Linear-Attention Live)

**Status:** PLAN (kein Code)
**Datum:** 2026-05-19
**Vorgängersprint:** Sprint F-1 (`bd74aad`) — SSM-Conv Shader portiert,
LayerStep::SsmConv1d eingebaut (no-op Body), `Forward::conv_state_buf`
allokiert (5.6 MB).
**Brief:** Sprint G — Gated-Delta-Net + Conv Dispatch
**Referenz:** llama.cpp 75 a4a3b (qwen35.cpp + delta-net-base.cpp +
gated_delta_net.comp)

---

## 1. Warum dieser Plan vor Code

Brief listet 6 Deliverables. **llama.cpp's `build_layer_attn_linear`
ruft 16 distincte Operationen pro Layer auf** (qwen35.cpp:337-469
verifiziert). Die 10 fehlenden Ops sind nicht optional — ohne sie
ist die Output-Inkohärenz nicht durch Barrier-Tuning rettbar.
Aufwand-Schätzung Brief 8-10 h vs. realistisch (mit
last-mile-Debugging analog zu Gemma-4-26B Sprint 53C-G): **25-40 h**.

Plan-Doc reduziert Risiko:
- Frühe Validierung der Shape-/Stride-/Offset-Annahmen
- Klare LayerStep-Surface (DEC+BAT Sympathy mit Coding-Standards §3.2)
- Realistische Sprint-G-Splits (G-2a/G-2b/...) statt einem Mega-Commit
- Eine Anlaufstelle für Coherence-Bisect wenn Sprint G-* daneben geht

---

## 2. Verifizierte Tensor-Shapes (Qwen3.6-27B-MTP)

Aus qwen35.cpp:80-91 (load_block_trunk recurrent branch):

| Tensor | Shape | Source-Zeile | VF-Lookup |
|---|---|---|---|
| `attn_qkv.weight` (wqkv) | `[5120, 10240]` | :83 | `blk.N.attn_qkv.weight` |
| `attn_gate.weight` (wqkv_gate) | `[5120, 6144]` | :84 | `blk.N.attn_gate.weight` |
| `ssm_conv1d.weight` | `[4, 10240]` | :85 | `blk.N.ssm_conv1d.weight` |
| `ssm_dt` (bias) | `[48]` | :86 | `blk.N.ssm_dt.bias` |
| `ssm_a` (no-scan) | `[48]` | :87 | `blk.N.ssm_a` |
| `ssm_beta.weight` | `[5120, 48]` | :88 | `blk.N.ssm_beta.weight` |
| `ssm_alpha.weight` | `[5120, 48]` | :89 | `blk.N.ssm_alpha.weight` |
| `ssm_norm.weight` | `[128]` | :90 | `blk.N.ssm_norm.weight` |
| `ssm_out.weight` | `[6144, 5120]` | :91 | `blk.N.ssm_out.weight` |

**ACHTUNG (Brief-Korrektur):** Brief schreibt `ssm_a: [128 × 48]`.
Verifiziert ist `ssm_a: [48]` (1D, scalar-per-v-head). Das spart
6144 Float-Konstanten pro Layer und entspricht dem
`g_val = exp(data_g[gb_off])` Pfad in gated_delta_net.comp:139
(KDA=0 ⇒ skalarer g-Eintrag pro Token).

**Conv-Channel-Formel** (qwen35.cpp:385):
```
conv_channels = d_inner + 2 * ssm_n_group * ssm_d_state
              = 6144     + 2 * 16          * 128
              = 10240
```

**Split-Offsets im Conv-Output** (qwen35.cpp:402-422):
```
qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        = 128 * 16 * 2 + 128 * 48
        = 4096          + 6144
        = 10240

Q-Slice: offset = 0,                size = head_k_dim*num_k_heads = 2048
K-Slice: offset = head_k_dim*num_k_heads = 2048, size = 2048
V-Slice: offset = 2*head_k_dim*num_k_heads = 4096, size = head_v_dim*num_v_heads = 6144
```

---

## 3. Die 16 Operationen (verifiziert aus qwen35.cpp:337-469)

| # | Op | Input(s) | Output | Source-Zeile | VF-Status |
|---|---|---|---|---|---|
| 1 | `AttnNorm` | hidden | hidden_norm | (vor `build_layer_attn_linear`) | ✅ vorhanden |
| 2 | `AttnQkvProj` GEMV | hidden_norm + wqkv `[5120,10240]` | qkv_mixed `[10240]` | :356 (build_qkvz) | ❌ neu |
| 3 | `AttnGateProj` GEMV | hidden_norm + wqkv_gate `[5120,6144]` | z `[6144]` | :356 (build_qkvz) | ❌ neu |
| 4 | `SsmBetaProj` GEMV | hidden_norm + ssm_beta `[5120,48]` | beta_raw `[48]` | :360 | ❌ neu |
| 5 | `SigmoidBeta` | beta_raw | beta `[48]` | :364 | ⚠️ Reuse Sigmoid (existing?) |
| 6 | `SsmAlphaProj` GEMV | hidden_norm + ssm_alpha `[5120,48]` | alpha_raw `[48]` | :367 | ❌ neu |
| 7 | `AlphaAddDtBias` | alpha_raw + ssm_dt.bias | alpha_biased `[48]` | :371 (`ggml_add`) | ⚠️ Reuse `run_binary(Add)` |
| 8 | `Softplus` | alpha_biased | alpha_softplus `[48]` | :372 | ❌ neu shader (23 LOC port) |
| 9 | `GateMulSsmA` | alpha_softplus × ssm_a | gate `[48]` | :375 (`ggml_mul`) | ⚠️ Reuse `run_binary(Mul)` |
| 10 | `SsmConv1d` | qkv_mixed + state(3) + ssm_conv1d.weight | conv_output `[10240]` | :393 | ⚠️ Sprint F shader vorhanden, Body fehlt |
| 11 | `Silu` | conv_output | conv_silu `[10240]` | :396 | ✅ existing Silu |
| 12 | `SplitQKV` (view only) | conv_silu | Q `[2048]`, K `[2048]`, V `[6144]` | :406-422 | ✅ keine Operation, nur Offset-Push-Const |
| 13 | `L2Norm` × 2 | q_slice, k_slice + eps | q_l2, k_l2 | :430-431 | ❌ neu shader (44 LOC port) — bzw RMSNorm w=1 trick |
| 14 | `RepeatHeads` | q_l2, k_l2 (16-head → 48-head) | q_rep, k_rep | :439-443 (conditional) | ❌ neu shader (26 LOC port) ODER fused-GDN-Mode |
| 15 | `GatedDeltaNet` | q_rep, k_rep, v_slice, gate, beta + ssm_state | gdn_out `[6144]`, new_state | :449 | ❌ neu shader (190 LOC port) + 144 MB Buffer |
| 16 | `NormGated` | gdn_out + ssm_norm + z | attn_out_norm `[6144]` | :455 | ⚠️ Fuseable: RMSNorm + Silu + Mul (3 existing dispatches) |
| 17 | `SsmOutProj` GEMV | attn_out_norm + ssm_out `[6144,5120]` | linear_attn_out `[5120]` | :462 | ❌ neu |
| 18 | `AttnResidualAdd` | linear_attn_out + inpSA | residual_out `[5120]` | :185 (post-call) | ✅ existing |

(18 statt 16 weil ich Sprint G-Brief's "GatedOutput" rausgelassen
habe — qwen35.cpp ruft nach SsmOut **keinen** sigmoid-gate-multiply
auf. Das `gate` aus Op 9 wird IN GDN via decay konsumiert, nicht
nach Output-Proj angewendet. Brief-Annahme korrigiert.)

---

## 4. Buffer-Allocations (NEU)

### 4.1 Persistent ssm_state (~144 MB)

```
ssm_state_buf: [n_recurrent × num_v_heads × head_v_dim × head_v_dim × 4]
             = [48          × 48           × 128         × 128         × 4]
             = 150 994 944 bytes
             ≈ 144 MiB
```

`MemoryLocation::GpuOnly`, zero-init beim ersten Token via
`cmd_fill_buffer(buf, 0, size, 0)` (Sprint F-1 deferred dies).

### 4.2 Scratch-Buffer pro Layer (per-token, NICHT persistent)

Wiederverwendbar über alle 48 recurrent layers (sequenziell
dispatched). Sized auf den größten benötigten Wert:

| Slot | Größe (decode) | Größe (prefill pp=64) | Verwendung |
|---|---|---|---|
| `ssm_qkv_buf` | 10240 × 4 = 40 KB | 10240 × 64 × 4 = 2.5 MB | qkv_mixed (Op 2 → Op 10) |
| `ssm_z_buf` | 6144 × 4 = 24 KB | 6144 × 64 × 4 = 1.5 MB | z gate (Op 3 → Op 16) |
| `ssm_beta_buf` | 48 × 4 = 192 B | 48 × 64 × 4 = 12 KB | beta (Op 4 → Op 15) |
| `ssm_alpha_buf` | 48 × 4 = 192 B | analog | alpha_raw → alpha_softplus → gate |
| `ssm_gate_buf` | 48 × 4 = 192 B | analog | gate für GDN (Op 9 → Op 15) |
| `ssm_conv_input_buf` | 4 × 10240 × 4 = 160 KB | (4+seq-1) × 10240 × 4 | Op 10 src0 |
| `ssm_conv_output_buf` | 10240 × 4 = 40 KB | 10240 × seq × 4 | Op 10 dst → Op 11 → Op 13 |
| `ssm_qrep_buf` | 2048 × 3 × 4 = 24 KB | analog | repeated Q (Op 14) |
| `ssm_krep_buf` | 2048 × 3 × 4 = 24 KB | analog | repeated K (Op 14) |
| `ssm_gdn_out_buf` | 6144 × 4 = 24 KB | 6144 × seq × 4 | Op 15 dst |
| `ssm_norm_out_buf` | 6144 × 4 = 24 KB | analog | Op 16 dst |

**Aggregate scratch decode:** ~400 KB
**Aggregate scratch prefill (pp=64):** ~30 MB

Allokationen in `Forward::new`, gegated auf `cfg.qwen35.is_some()`,
gecleaned in `Forward::destroy` parallel zu `conv_state_buf`.

### 4.3 Total VRAM-Impact

| Posten | Sprint F-1 | Sprint G |
|---|---|---|
| Model weights | 11.7 GiB | 11.7 GiB |
| KV-Cache (FP8) | 136 MB | 136 MB |
| conv_state_buf | 5.6 MB | 5.6 MB |
| **ssm_state_buf** | — | **144 MB** |
| Scratch (decode-pfad-Max) | — | ~30 MB |
| Pipeline cache + sonst. | ~250 MB | ~250 MB |
| **Total** | **~12.2 GiB** | **~12.4 GiB** |

Passt in 16 GiB. **0.2 GiB Margin** zu kritischer 10-GiB-Cliff-Grenze
(Sprint I VRAM-Scaling-Memory) ist **schon überschritten** —
`VF_LMHEAD_ALLOC_FIRST=1` Recommendation steht für alle Qwen3.6
Workloads.

---

## 5. Neue Shaders (Port-Liste)

| Shader | Upstream-LOC | Notes |
|---|---|---|
| `ssm_conv.comp` | 50 | **DONE Sprint F-1** |
| `gated_delta_net.comp` | 190 | 3 Build-Defines (`USE_SUBGROUP_CLUSTERED`, `USE_SUBGROUP_ADD`, `KDA=0`). Spec-Consts `[S_V=128, KDA=0, SUBGROUP_SIZE, LANES_PER_COLUMN]`. RDNA4 hat alle Subgroup-Ops, default-clustered ist OK. |
| `softplus.comp` | 23 | Trivial elementwise `log(1+exp(x))`. 1 SSBO inout. |
| `l2_norm.comp` | 44 | `x / sqrt(sum(x²)+eps)`. Pro Wert wie RMSNorm aber ohne weight. **Alternative:** Reuse `rms_norm.comp` mit `weight=1.0` Bias-Buffer (spart 1 Shader, kostet 1 Buffer-Konstante pro Layer). |
| `repeat.comp` | 26 | n_dim 0-3 Repeat-Faktoren in Push-Const. Für unseren Fall: nur dim-1 (heads) repeat-Faktor=3 (16→48). |
| `norm_gated.comp` (opt.) | NEU | Fused `rms_norm(x)*silu(z)`. Spart 2 dispatches+barriers gg 3-step naive. Optional Sprint-G3-Optimierung. |

**Empfehlung Sprint G:** L2Norm via RMSNorm-Trick (kein neuer Shader),
NormGated zunächst als 3-step naive (Sprint G3 optimiert). Bleibt:
GDN (190 LOC), Softplus (23 LOC), Repeat (26 LOC) — **3 neue Shaders**.

---

## 6. LayerStep-Surface (Coding-Standards §3.2)

Jeder Variant braucht: enum-Eintrag in `layer_plan.rs` + 2 Match-Arms
in `executor/dispatch.rs` + 2 Step-Bodies in
`executor/attention.rs` (DEC und BAT im gleichen File pro §4.2) +
Builder-Anpassung in `build_qwen35_layer` + 1 Unit-Test.

### 6.1 Neu in Sprint G

| LayerStep | Op-Nr | Param-Set |
|---|---|---|
| `AttnQkvProj { layer }` | 2 | — (uses `attn_qkv.weight`) |
| `AttnGateProjLin { layer }` | 3 | — (Linear-Attn gate, NICHT der Full-Attn-AttnQGateProj) |
| `SsmBetaProj { layer }` | 4 | — |
| `SsmAlphaProj { layer }` | 6 | — |
| `SsmGateCompose { layer }` | 7+8+9 fused | — (innerhalb des Step-Bodies: add+softplus+mul ohne extra LayerSteps) |
| `SsmConvBody { layer }` | 10 | — (existing `SsmConv1d` Variant, nur Body neu) |
| `SsmConvSilu` | 11 | — (Reuse: `Activation { Silu }` ?? — aktueller Variant ist GLU-fused, braucht reine Silu-Variant ODER inline in SsmConvBody) |
| `SsmQKL2Norm { layer }` | 13 | — (2 dispatches innerhalb des Body) |
| `SsmRepeatQK { layer }` | 14 | repeat_factor (= num_v_heads/num_k_heads = 3) |
| `GatedDeltaNet { layer }` | 15 | — |
| `NormGatedSsm { layer }` | 16 | eps (typically `cfg.rms_norm_eps`) |
| `SsmOutProj { layer }` | 17 | — |

**Brief vs Plan diff:** Brief erwähnt `NormGated`, `SsmOutProj`,
`AttnQkvProj`, `GatedDeltaNet`. Plan ergänzt: `AttnGateProjLin`,
`SsmBetaProj`, `SsmAlphaProj`, `SsmGateCompose`, `SsmQKL2Norm`,
`SsmRepeatQK`. **+6 Variants gegenüber Brief.**

### 6.2 Reused Existing Variants

- `AttnNorm` — Op 1
- `AttnResidualAdd` — Op 18
- Plus FFN-Block (6 Steps) unverändert

### 6.3 Builder-Sequence build_qwen35_layer

```rust
if spec.is_recurrent_layer(layer) {
    plan.push(LayerStep::AttnNorm);
    plan.push(LayerStep::AttnQkvProj      { layer });   // → ssm_qkv_buf
    plan.push(LayerStep::AttnGateProjLin  { layer });   // → ssm_z_buf
    plan.push(LayerStep::SsmBetaProj      { layer });   // → ssm_beta_buf (raw)
    // Sigmoid auf ssm_beta_buf via inline (kein eigener LayerStep wenn fwd.run_sigmoid existiert)
    plan.push(LayerStep::SsmAlphaProj     { layer });   // → ssm_alpha_buf
    plan.push(LayerStep::SsmGateCompose   { layer });   // alpha + dt → softplus → × ssm_a
    plan.push(LayerStep::SsmConv1d        { layer });   // qkv_mixed + state → conv_silu (fused silu)
    plan.push(LayerStep::SsmQKL2Norm      { layer });   // q/k slices in-place L2-norm
    plan.push(LayerStep::SsmRepeatQK      { layer, repeat_factor: 3 });
    plan.push(LayerStep::GatedDeltaNet    { layer });   // GDN dispatch + ssm_state update
    plan.push(LayerStep::NormGatedSsm     { layer });   // RMSNorm(gdn_out, ssm_norm) * silu(z)
    plan.push(LayerStep::SsmOutProj       { layer });   // → linear_attn_out
    plan.push(LayerStep::AttnResidualAdd);
    // FFN folgt (unverändert)
}
```

12 LayerSteps + AttnNorm + ResidualAdd + 6 FFN-Steps = **20 Steps pro
Linear-Attn-Layer × 48 Layer = 960 dispatches/token**. Plus die
17 × ~17 = 289 dispatches/token für Full-Attn Layer. **Decode total
~1250 dispatches/token** (Qwen3-8B hat ~36 layers × ~15 steps = 540).

Per Memory `[[feedback_decode_dispatch_bound]]`: pro Dispatch ~5-8 µs
Overhead auf RDNA4. 1250 × 6 µs = 7.5 ms/token → **~130 tok/s Floor
(dispatch-bound)**. Echte Decode wird unten landen wegen GDN-Compute
+ Memory-BW. Schätzung: **30-50 tok/s**.

---

## 7. Per-Step Implementation-Notes

### 7.1 AttnQkvProj / AttnGateProj / SsmBetaProj / SsmAlphaProj / SsmOutProj

Standard GEMVs. Reuse `fwd.run_gemv_dispatch` analog zu
`step_q_proj` / `step_k_proj`. Push consts identisch, nur Weight-
Buffer-Lookup ändert sich (`layer_weight(model, layer, "ssm_xxx.weight")`).

**Risiko:** Q-Type-Coverage von ssm_*-Tensoren. Per Sprint-A-Pre-Check-
Empfehlung (analysis-doc §7) sind die 2D-Quants vermutlich `Q3_K_S`
(gleicher Quant wie alle anderen 2D-Tensoren in dem GGUF). VF's
gemv-Q3_K-Pfad ist getestet. Verifizieren mit:
```bash
~/tmp/llama.cpp/gguf-py/scripts/gguf_dump.py \
  ~/models/Qwen3.6-27B-MTP-Q3_K_S.gguf | grep "blk.0.ssm_\|blk.0.attn_qkv\|blk.0.attn_gate"
```

### 7.2 SsmGateCompose (Op 7+8+9 fused im Body)

```rust
fn step_ssm_gate_compose(&self, fwd, ctx, layer) {
    let alpha = fwd.ssm_alpha_buf.handle;
    let ssm_dt = layer_weight(ctx.model, layer, "ssm_dt.bias");
    let ssm_a = layer_weight(ctx.model, layer, "ssm_a");  // NB: kein .weight Suffix
    let gate = fwd.ssm_gate_buf.handle;

    // 1. alpha += ssm_dt (in-place)
    fwd.run_binary(ctx.dev, ctx.registry, ctx.cmd, ShaderId::Add,
        alpha, ssm_dt, alpha, NUM_V_HEADS, "ssm_alpha_add_dt");
    fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[alpha]);

    // 2. alpha = softplus(alpha) (in-place via NEW shader)
    fwd.run_softplus(ctx.dev, ctx.registry, ctx.cmd, alpha, NUM_V_HEADS);
    fwd.maybe_compute_barrier(ctx.dev, ctx.cmd, &[alpha]);

    // 3. gate = alpha * ssm_a (elementwise)
    fwd.run_binary(ctx.dev, ctx.registry, ctx.cmd, ShaderId::Mul,
        alpha, ssm_a, gate, NUM_V_HEADS, "ssm_gate_mul");
    fwd.mark_written(&[gate]);
}
```

ssm_dt und ssm_a sind 1D-Vectoren `[48]` aus dem GGUF — gleicher
Lookup wie für 2D-Weights aber kein `.weight` Suffix für `ssm_a`.

### 7.3 SsmConv1d Body (Sprint F-1 Deferred)

```rust
fn step_ssm_conv1d(&self, fwd, ctx, layer) {
    let qkv = fwd.ssm_qkv_buf.handle;        // input from AttnQkvProj
    let weight = layer_weight(ctx.model, layer, "ssm_conv1d.weight");
    let conv_state = fwd.conv_state_buf.handle;
    let conv_input = fwd.ssm_conv_input_buf.handle;
    let conv_output = fwd.ssm_conv_output_buf.handle;

    let recurrent_idx = count_recurrent_before(spec, layer);
    let state_offset_bytes = recurrent_idx
        * (spec.ssm_d_conv - 1) as u64
        * spec.conv_channels() as u64 * 4;

    // 1. Bau conv_input = [past_state(3), current_qkv(1)]
    //    Layout: channel-major-time-inner. Per channel ch, 4 time slots.
    //    state[t=0..2, ch] → conv_input[ch*4 + 0..2]
    //    qkv[ch]            → conv_input[ch*4 + 3]
    //    → Braucht entweder einen Helper-Shader (build_conv_input.comp,
    //      1 dispatch) ODER 4 cmd_copy_buffer Calls.
    // ANSCHEIN-VERSION (4 cmd_copy_buffer):
    //   für i in 0..3: copy conv_state[offset + i*conv_channels*4..]
    //                       → conv_input[i*conv_channels*4..]
    //   copy qkv → conv_input[3*conv_channels*4..]
    // Aber das ist channel-OUTER-time-INNER falsch. Die richtige Form
    // braucht stride. → 1 dispatch eines neuen Mini-Shaders ist sauberer.

    // 2. Dispatch ssm_conv_f32 (Sprint F bereits gewired)
    fwd.run_ssm_conv(...);
    fwd.maybe_compute_barrier(...);

    // 3. Silu auf conv_output (in-place)
    fwd.run_silu(...);
    fwd.maybe_compute_barrier(...);

    // 4. State shift: conv_state[0..3] ← conv_input[1..4]
    //    1 cmd_copy_buffer mit src/dst overlap NICHT erlaubt — braucht
    //    Zwischenbuffer ODER channel-major Layout-Annahme.
    cmd_copy_buffer(conv_input + 1*stride → conv_state + 0*stride, 3*stride);
}
```

**Subtilität:** Die Layouts müssen passen. ssm_conv.comp erwartet
src0 mit `nb01 = bytes_per_channel = (kernel_size * sizeof(f32)) = 16`
(verifiziert in :32 `i1 * (nb01/4)`). Das ist **channel-major-time-inner**.

Das macht die conv_state-Speicherung sinnvoll als channel-major:
state[channel][time]. Per Channel 3 floats. Insgesamt `[channels=10240,
time=3]` Layout. Beim Bau von conv_input wird state pro Channel 3
floats gefolgt vom current input 1 float = 4 floats. Das ist ein
strided write (3-Stride im Source, 4-Stride im Target).

**Solution:** Wir brauchen einen NEUEN Helper-Shader
`ssm_conv_build_window.comp` (~30 LOC) der das strided merge macht.
ODER wir layouten conv_state mit time-major (state[time][channel])
und bauen conv_input mit 3 cmd_copy_buffer + 1 cmd_copy für current.

**Empfehlung:** Helper-Shader. Cleaner, weniger Barriers.

### 7.4 GatedDeltaNet

GDN-Shader hat 7 Bindings:
- Q, K, V (readonly, FP32, geshapet `[head_v_dim, num_v_heads, n_tokens, n_seqs]`)
- G (gate, readonly, FP32, `[1, num_v_heads, n_tokens, n_seqs]` für KDA=0)
- Beta (readonly, FP32, gleiche Shape wie G)
- State (readonly+rw, FP32, `[head_v_dim, head_v_dim, num_v_heads, n_seqs]` PERSISTENT)
- Dst (FP32, `[head_v_dim * num_v_heads * n_tokens * n_seqs + state_snapshot]`)

Push-Consts:
```
H, n_tokens, n_seqs, s_off, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
neq1, rq3, float scale, K
```

Spec-Consts: `S_V=128, KDA=0, SUBGROUP_SIZE=64 (RDNA4), LANES_PER_COLUMN=32`.

Workgroup: `(head_id, seq_id, col_block)`.

**Compute pattern**: Pro Workgroup ein Q-V-Head-Pair, Subgroup-
Cluster-Reduction über `head_v_dim` für Inner-Product, Per-Lane-Shard
hält `ROWS_PER_LANE = S_V/LANES_PER_COLUMN = 4` State-Rows.
**Persistenter State wird im Dst-Buffer am Ende geschrieben** (siehe
:185-189 — slot s_off + offset, ein einziger Snapshot bei K=1).

**Aktivierungs-Stride-Push-Consts** (sq1, sq2, sq3, sv1, ...):
Q hat 3D-Layout `[head_v_dim, num_v_heads, n_tokens]` (n_seqs=1
typisch). sq1=head_v_dim (= 128), sq2=128*48=6144 (?). Verifizierung
nötig durch llama.cpp Dispatch-Caller.

### 7.5 NormGatedSsm (Op 16, Fused?)

3 Dispatches naive (Sprint G):
```rust
fn step_norm_gated_ssm(&self, fwd, ctx, layer) {
    let gdn_out = fwd.ssm_gdn_out_buf.handle;
    let z = fwd.ssm_z_buf.handle;
    let weight = layer_weight(ctx.model, layer, "ssm_norm.weight");
    let out = fwd.ssm_norm_out_buf.handle;

    // 1. RMSNorm(gdn_out, ssm_norm) → out (per-head, head_dim=128)
    fwd.run_rms_norm(...);
    fwd.maybe_compute_barrier(...);
    // 2. Silu(z) → z (in-place)
    fwd.run_silu(...);
    fwd.maybe_compute_barrier(...);
    // 3. out *= z (elementwise)
    fwd.run_binary(Mul, out, z, out, ...);
    fwd.mark_written(&[out]);
}
```

Sprint G3 Optimierung: Fused norm_gated.comp = 1 dispatch + 0 barriers
intern. ~30-50 LOC neuer Shader. Save 2 dispatches × 48 layers =
96 dispatches/token = ~0.5-0.7 ms/token. Lohnt sich nach
Coherence-Gate.

### 7.6 L2Norm via RMSNorm-Trick

```rust
fn step_ssm_qk_l2_norm(&self, fwd, ctx, layer) {
    // RMSNorm(q) und RMSNorm(k) je mit einer 1.0-gefüllten weight.
    // Alternative: Reuse `rms_norm.comp` mit einem static-init
    // weight-buffer `static_ones_128 = [1.0; 128]`.
    // q_norm: per-head Norm über head_k_dim=128
    fwd.run_rms_norm_no_weight(q_buf, q_buf, num_k_heads * 2 * num_v_heads, head_k_dim);
    // k_norm analog
}
```

**Risiko:** RMSNorm-Shader von VF erwartet evtl. immer einen weight-
Buffer. Wenn ja: kleinen static-ones-buffer allokieren oder neuen
parameterlose Variante schreiben (~20 LOC).

### 7.7 SsmRepeatQK

Q (von l2_norm) hat shape `[head_k_dim=128, num_k_heads=16,
n_tokens, n_seqs]` = 16 heads × 128 = 2048 floats/token. GDN-Shader
erwartet `[head_v_dim=128, num_v_heads=48, n_tokens, n_seqs]` =
48 heads × 128 = 6144 floats/token. **Repeat-Factor = num_v_heads /
num_k_heads = 3.**

Repeat heads-axis: für jedes head_in 0..16, schreibe drei Kopien
auf head_out 3*in, 3*in+1, 3*in+2 (oder 0, 16, 32 — abhängig von
llama.cpp's `ggml_repeat_4d` Konvention).

llama.cpp Konvention: `ggml_repeat_4d(t, head_k_dim, num_v_heads, ...)`
— sie spannen entlang `num_v_heads`. Per ggml-Semantik wird Element
[i,j,...] in dim 1 = `j % src.ne[1]`. Also: head_out j → head_in
j % 16. Das ergibt Layout `[h0, h1, ..., h15, h0, h1, ..., h15, h0, ...,
h15]` (3× wiederholt).

repeat.comp Shader (26 LOC):
- 1 SSBO src readonly + 1 SSBO dst
- Push consts: src/dst strides (ne01, ne02, ne03, nb01, ...)
- Each thread copies one element with modulo-index into src

### 7.8 Persistent State: Zero-Init + /reset

```rust
// In Forward::new / Forward::new_persistent_init:
if let Some(ref buf) = self.ssm_state_buf {
    let bytes = self.ssm_state_bytes;
    unsafe { dev.device.cmd_fill_buffer(cmd, buf.handle, 0, bytes, 0); }
}
if let Some(ref buf) = self.conv_state_buf { /* analog */ }
// Pipeline barrier TRANSFER → COMPUTE_SHADER

// /reset Command (in chat REPL):
pub fn reset_recurrent_state(&mut self, cmd: vk::CommandBuffer, dev: &VulkanDevice) {
    // Gleiche cmd_fill_buffer Logik wie initial zero-init.
}
```

Ort des /reset-Hooks: `src/server/serve.rs` / `src/main.rs` chat-
REPL. Sucht nach `"/reset"` String und ruft die neue Methode auf.

---

## 8. Coherence-Test Pipeline

### 8.1 llama.cpp Referenz

```bash
# Build (existing build):
~/tmp/llama.cpp/build/bin/llama-cli --version  # ensure existed

# 5-Prompts referenzieren:
for prompt in \
  "The capital of France is" \
  "What is 2+2?" \
  "Berlin ist die Hauptstadt von" \
  "Write a haiku about rain." \
  "Explain a GPU in one sentence."; do
    echo "=== $prompt ==="
    ~/tmp/llama.cpp/build/bin/llama-cli \
        -m ~/models/Qwen3.6-27B-MTP-Q3_K_S.gguf \
        -p "$prompt" -n 30 --temp 0.0 -ngl 99
done > /tmp/qwen36_llama_ref.txt 2>&1
```

### 8.2 VF Output

```bash
for prompt in ...; do
    echo "=== $prompt ==="
    echo "$prompt" | VF_LMHEAD_ALLOC_FIRST=1 VULKANFORGE_KV_FP8=1 \
        RAYON_NUM_THREADS=4 vulkanforge chat \
        --model ~/models/Qwen3.6-27B-MTP-Q3_K_S.gguf \
        --temperature 0.0 --max-tokens 30 --max-context 4096
done > /tmp/qwen36_vf_out.txt 2>&1
```

### 8.3 Gate-Kriterien

| Kriterium | Min |
|---|---|
| KOHÄRENT (semantisch sinnvoll) | 5/5 ✓ |
| KORREKT (Paris/4/Berlin) | 4/5 (Haiku schwankt) |
| KEIN NaN/Inf | 5/5 |
| KEIN REP (Wiederholungs-Loop) | 5/5 |
| ÄHNLICH (erste 5 Tokens) | ≥ 3/5 |

Sprint G ist erst dann "live" wenn **alle 4 Hard-Gates (KOHÄRENT,
KORREKT, NO-NaN, NO-REP) erfüllt sind**. ÄHNLICH ist optional (echte
Vergleichbarkeit braucht bit-identisches Quantization-Pfad).

---

## 9. Realistic Effort-Breakdown

| Slice | LOC-Schätzung | Stunden |
|---|---|---|
| **G-2a: Shader-Ports** | | |
| · gated_delta_net.comp port + 3 build defines | ~200 | 1.5 |
| · softplus.comp port | ~25 | 0.5 |
| · repeat.comp port | ~30 | 0.5 |
| · build.rs + ShaderId + pipeline_registry (×3) | ~50 | 1 |
| **G-2b: Buffer + LayerStep + Builder** | | |
| · ssm_state_buf alloc + cleanup | ~30 | 0.5 |
| · Scratch-Buffer alloc (11 neue Felder) | ~80 | 1 |
| · 9 neue LayerStep-Variants (enum + 18 match arms) | ~120 | 2 |
| · 9 neue step_*/b_step_* Body-Pairs | ~400 | 4 |
| · build_qwen35_layer Update + 5 neue Unit-Tests | ~120 | 1.5 |
| **G-2c: Conv-Body (Sprint F deferred)** | | |
| · ssm_conv_build_window.comp helper | ~40 | 1 |
| · step_ssm_conv1d Body (DEC + BAT) | ~120 | 2 |
| **G-2d: GDN Wiring** | | |
| · run_gdn wrapper + push-const struct | ~100 | 1 |
| · step_gated_delta_net Body (DEC + BAT) | ~150 | 2 |
| · State-buffer offset math + per-token snapshot | ~50 | 1 |
| **G-2e: /reset + State Lifecycle** | | |
| · Lazy zero-fill on first token | ~40 | 0.5 |
| · /reset REPL hook | ~30 | 0.5 |
| **G-2f: Coherence + Debug** | | |
| · 5-Prompt-Reference vs VF compare | — | 1 |
| · Barrier-Bisect (worst case) | — | 4-12 |
| · Shape-Mismatch-Bisect (worst case) | — | 4-12 |
| **Total optimistic** | ~1700 LOC + 285 shader | **~24 h** |
| **Total pessimistic (last-mile)** | + 8-16 h debug | **~32-40 h** |

---

## 10. Sprint-G-Splits Vorschlag

### G-2a: Shader-Ports (1 Commit, ~3.5 h)

- `vk_shaders/gated_delta_net.comp`, `softplus.comp`, `repeat.comp`
  (3 new shaders, all directly ported)
- `vk_shaders/ssm_conv_build_window.comp` (1 new helper)
- build.rs (4 ShaderJobs)
- shaders.rs (4 new ShaderIds + bytes + names)
- pipeline_registry.rs (4 arms: GDN with spec consts, repeat
  no-spec, softplus no-spec, conv-build no-spec)
- **0 LayerStep changes** — alle Shader compiled aber nicht
  dispatched
- Test: build clean, lib-tests grün (216/216 nach UpdateF), 145
  shaders in chat-Header

Risiko: minimal. Pure additive.

### G-2b: Buffer + Plumbing (1 Commit, ~5 h)

- 9 neue LayerStep-Variants (mit no-op Bodies, alle in
  `executor/attention.rs`)
- `Forward::ssm_state_buf` + 11 Scratch-Felder
- Buffer-Allokation in setup.rs (alle `Option<GpuBuffer>` gegated
  auf `cfg.qwen35.is_some()`)
- Cleanup in destroy
- build_qwen35_layer emittiert die neuen LayerSteps in der
  korrekten Reihenfolge
- 5 neue Unit-Tests (plan-shape pro neue LayerStep)
- Smoke: Qwen3.6 läuft mit no-op-Bodies durch (same gibberish output
  wie Sprint F-1, same regression-gates pass)

Risiko: niedrig. Buffer-Alloc-Logic ist die einzige neue Lauffläche.

### G-2c: Conv Body + L2Norm (1 Commit, ~3 h)

- step_ssm_conv1d echter Body (DEC + BAT)
- step_ssm_qk_l2_norm echter Body
- state-shift cmd_copy_buffer chain
- Lazy zero-fill bei erstem Token
- Smoke: conv_output ist non-zero, state akkumuliert
- Output STILL incorrect (GDN downstream nicht wired)

Risiko: medium. State-Buffer-Offsets sind die häufigste Bug-Quelle.

### G-2d: SsmGateCompose + GEMVs (1 Commit, ~3 h)

- step_attn_qkv_proj, step_attn_gate_proj_lin, step_ssm_beta_proj,
  step_ssm_alpha_proj, step_ssm_out_proj (5 step bodies, alle reine
  GEMV-Dispatches)
- step_ssm_gate_compose Body (add + softplus + mul)
- Smoke: Ergebnis-Buffer haben non-zero Werte
- Output STILL incorrect (GDN missing)

Risiko: niedrig (GEMVs sind well-tested in VF). Softplus
shape-validation ist die kleine Unbekannte.

### G-2e: GDN Live (1 Commit, ~5 h)

- run_gdn wrapper
- step_gated_delta_net Body (mit ssm_state offset math)
- step_norm_gated_ssm Body (3-step naive)
- step_ssm_repeat_qk Body
- /reset Hook
- **Coherence-Test: 5-Prompt-Vergleich gegen llama.cpp**
- Hier ist last-mile-Debugging fällig (Stunden 4 oben sind Best-Case)

Risiko: **HOCH**. State-Layout-Mismatch, Barrier-Bugs, scale-
Mismatches in GDN-Push-Const, head-Repeat-Konvention — alle
plausibel. Coherence-Gate kann mehrere Iterationen kosten.

### G-3 (optional): Performance-Optimierung

- Fused norm_gated.comp (1 Shader statt 3 dispatches)
- Q/K/V-Split via Shader-Push-Const Offsets statt
  cmd_copy_buffer
- Conv input-build im Conv-Shader inline (eliminiert build_window
  helper)

---

## 11. Risiko-Register

| # | Risiko | Mitigation |
|---|---|---|
| R1 | GDN state-buffer-Layout mismatched mit Shader-Erwartung | Sprint G-2e: Build a standalone harness test (analog `[[feedback_standalone_kernel_harness]]`) bevor in forward path integrieren |
| R2 | Q/K/V split-offset off-by-one | Im step_ssm_conv1d Body explizit 3 dispatches mit getrennten Push-Const-Offsets statt 1 dispatch + Shader-internal Splits |
| R3 | softplus numerical stability für große negative alpha | Test mit alpha ∈ [-20, 20]; softplus.comp upstream nutzt `log1p(exp(-|x|)) + max(0,x)` Form für Stabilität — diese Form übernehmen |
| R4 | ssm_a-Konvention: positiv-multiply vs `-A_log.exp()` Inversion | qwen35.cpp:375 sagt explizit `gate = alpha_softplus * ssm_a` ohne Vorzeichen-Inversion — wir folgen 1:1, GGUF speichert bereits `-A_log.exp()` |
| R5 | Repeat-Konvention: `[h0,h0,h0,h1,h1,h1,...]` vs `[h0,h1,...,h0,h1,...]` | ggml_repeat_4d nutzt Modulo-Konvention: `dst[i] = src[i % src.ne]` → das ist die SECOND Variante (`[h0,...,h15,h0,...]`). repeat.comp 1:1 portieren |
| R6 | Decode-Speed-Regression auf Qwen3-8B durch neue Forward-Felder | Bench Qwen3-8B vor/nach G-2b; Forward-Felder sind Option<GpuBuffer> mit `None` für Qwen3 — Match-Cost = ein Compare = vernachlässigbar |
| R7 | Coherence-Bisect ohne Layer-by-Layer-Dump-Infra | Pre-G: Layer-0-Hidden-State-Dump aktivieren analog zu Gemma-4 53D-G Workflow (`feedback_hidden_state_bisect`). Sprint G-2e startet damit |
| R8 | `feedback_layer_dispatch_paths`: neue LayerSteps in BAT vergessen | Coding-Standards §3.2 erzwingt durch exhaustive Match — Compile-Gate fängt's |
| R9 | `feedback_subscriber_barrier_pattern`: zu wenig Barriers zwischen Conv-output → L2Norm → GDN | jede Op ENDET mit `mark_written` + nächste Op STARTET mit `maybe_compute_barrier` — Barrier-Elision-Tracker erfasst's automatisch |
| R10 | VRAM-Overflow durch 144 MB ssm_state | Sprint G-2b vor-validierung mit `mem_info_vram_used` — wenn 12.4 GiB überschreitet, halt mit Diagnose |

---

## 12. Coding-Standards Compliance Checkliste

- [✓] §2.1 GPU-Dispatch-Helper `run_*` → `runs.rs`
- [✓] §2.4 Neue Shader → shaders.rs + pipeline_registry + runs.rs
- [✓] §2.5 Buffer-Alloc → setup.rs + Field in state.rs
- [✓] §2.9 Attention-Code → executor/attention.rs (SSM = Teil
       des attn-sub-blocks)
- [✓] §3.2 Neue LayerSteps in **BEIDEN** Executors → dispatch.rs
       + step bodies in attention.rs (DEC + BAT co-located)
- [✓] §3.3 Variant-Naming generisch (z.B. `GatedDeltaNet`, nicht
       `Qwen35Gdn`)
- [✓] §4.1 KEIN Inline-Logic in decode.rs/prefill.rs
- [✓] §4.2 DEC + BAT Pendants im selben File
- [✓] §4.4 KEIN Wildcard `_ =>`
- [✓] §4.6 Barriers gegated (alle hier nur auf Qwen3.6-Pfad)
- [✓] §5.1 Unit-Tests für jede neue LayerStep
- [✓] §5.2 Regression-Gate: Qwen3-8B ≥ 100 t/s, Gemma-4-26B
       "Paris." natural EOS (G-2b smoke + G-2e final)

---

## 13. Was als Nächstes

**Owner-Entscheidung:**
1. Plan abnehmen → Sprint G-2a starten (3.5 h, low risk)
2. Plan modifizieren (z.B. l2_norm via separater Shader, fused
   norm_gated, andere Buffer-Strategie)
3. Plan ablehnen → Sprint G in andere Richtung skizzieren

**Pre-G-2a Pre-Check:** Quant-Type-Audit der ssm_*-Tensoren
(gguf_dump des Qwen3.6-Q3_K_S, verifiziert dass `attn_qkv`,
`attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_out` alle Q3_K_S sind und
`ssm_a`, `ssm_dt`, `ssm_norm`, `ssm_conv1d` FP32 oder FP16 sind).
Verzögert G-2a um 20 Minuten, fängt Q-Type-Mismatches im richtigen
GEMV-Shader-Dispatch ab.

---

## 14. Notes

- Diese Plan-Doc lag etwa ~2 h Aufwand. Sie ist NICHT der Punkt um
  Specs zu verfeinern — Schreibstand verifizierter llama.cpp-
  Snapshots + Sprint F-1 + Sprint A. Konkrete Implementations-
  Details (push-const layout pro Sprint, exact stride math)
  entstehen pro Sprint-G-Slice.
- Falls Sprint G-2a/b/c smoothy laufen und Coherence-Gate (G-2e)
  beim ersten Versuch hits, ist der gesamte Sprint G ~17-20 h.
  Last-mile-Debugging-Tail kann das aber leicht verdoppeln.
- Memory `[[feedback_sprint_hypothesis_check]]`: Bevor G-2e
  startet, validate dass GDN-Shader gegen llama.cpp's `--n-gpu-layers 0`
  CPU-Path Bit-identisch arbeitet (kleines Standalone-Harness mit
  Mock-State). Spart potentiell 8+ Stunden last-mile-Debug.

