# Phase 5A-3 — Ship CB-Reuse + Full 15-Prompt Benchmark

**Datum:** 2026-04-26
**Status:** Stage 2D als Default geshippt. 15-Prompt-Suite auf allen
3 Modellen gemessen. README + CHANGELOG aktualisiert.
**Hardware:** AMD Radeon RX 9070 XT (RADV GFX1201, RDNA 4, Mesa 26.0.5)

## TL;DR

* **CB-Reuse ist jetzt Default**. `VULKANFORGE_CB_REUSE=0` für Direct-
  Path-Fallback; jede andere Belegung (auch unset) aktiviert die Cache.
* **15-Prompt-Suite auf allen 3 Modellen**:
  * Qwen3-8B: **88.5 tok/s** median decode (war 72.4 in 4D, +22 %)
  * Llama-3.1-8B: **94.6 tok/s** (war 81.5, +16 %)
  * DeepSeek-R1-Llama-8B: **94.8 tok/s** (war 81.3, +17 %)
* **42/42 Tests grün** in beiden Modi (Cache an / Cache aus).
* **0 Rust-Warnings** im Release-Build (Dead-Code aufgeräumt).

## 1 — CB-Reuse als Default

```rust
// forward.rs::Forward::new
let cache_enabled = match std::env::var("VULKANFORGE_CB_REUSE") {
    Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
    _ => true,
};
```

Vorher: opt-in via `VULKANFORGE_CB_REUSE=1`.
Jetzt: opt-out via `VULKANFORGE_CB_REUSE=0`.

Bit-exact-Parity (Stage 2D, 16 Token Qwen3-8B): max abs err = 0.000e0.
Daher gefahrlos als Default zu setzen — der Direct-Path-Code bleibt
für Diagnose/Vergleich erhalten.

## 2 — 15-Prompt-Benchmark, alle Modelle

Reproduzierbar via:
```bash
VF_MODEL_PATH=$HOME/models/<MODEL>.gguf \
  cargo run --release --example run_15prompt_bench
```

### 2.1 Per-Modell-Tabelle

| Modell | Decode median | Decode aggregate | Prefill median | Coherent |
|---|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M | **88.5 tok/s** | 84.2 | 404.9 | 14/15 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | **94.6 tok/s** | 88.9 | 489.9 | 13/15 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | **94.8 tok/s** | 90.1 | 433.9 | 15/15 |

`Coherent` ist die automatische ✓/✗-Heuristik des Bench-Skripts. Die
Falsch-Negative auf Qwen3 (1) und Llama-3.1 (2) sind digits-only /
emoji-only Antworten (z. B. "1, 2, 3, …, 10."), die die "≥4 Buchstaben"-
Heuristik nicht passieren — die Outputs selbst sind korrekt.

### 2.2 Per-Prompt Highlights

Qwen3-8B (15 Prompts, alle ✓ außer #15 emoji-only):
```
# 1 Greeting             pp= 20 gen=  64  prefill= 261 tok/s  decode= 90.6
# 5 REST API (Go)        pp= 62 gen=1024  prefill= 453 tok/s  decode= 79.1
# 8 GPU Architecture     pp= 58 gen=1024  prefill= 437 tok/s  decode= 79.8
#11 Distrib. Msg Queue   pp= 62 gen= 895  prefill= 444 tok/s  decode= 82.2
#12 Long Sys Prompt+Q    pp=198 gen= 256  prefill= 428 tok/s  decode= 88.2
#14 Arithmetic           pp= 31 gen=  64  prefill= 347 tok/s  decode= 90.0
```
Decode bleibt zwischen 79.1 (lange 1024-Token-Outputs, Attention-Kosten
wachsen mit seq_len) und 90.6 tok/s (kurze Outputs).

DeepSeek-R1-Llama (alle 15/15 ✓):
```
# 1 Greeting             pp= 13 gen=  64  prefill= 194 tok/s  decode= 84.1
# 8 GPU Architecture     pp= 51 gen= 995  prefill= 505 tok/s  decode= 86.4
#12 Long Sys Prompt+Q    pp=191 gen= 256  prefill= 561 tok/s  decode= 94.2
#14 Arithmetic           pp= 21 gen=  64  prefill= 322 tok/s  decode= 96.9
```
Schwankung 84.1–96.9 tok/s; lange Generationen (≥ 1000 Tokens) verlieren
~10 % wegen Attention-Sweep-Wachstum.

## 3 — 4-System-Vergleich (aktualisiert)

Auf demselben Host (RX 9070 XT, RADV / ROCm), Qwen3-8B-Q4_K_M:

| System | Decode tok/s | Prefill tok/s | VF Ratio |
|---|---:|---:|---:|
| llama.cpp Vulkan | 114.2 | 4314 | 1.00 (Ref) |
| **VulkanForge 5A (Qwen3)** | **88.5** | **404.9** | **0.78 / 0.094** |
| **VulkanForge 5A (Llama-3.1)** | **94.6** | **489.9** | **0.83 / 0.114** |
| **VulkanForge 5A (DeepSeek)** | **94.8** | **433.9** | **0.83 / 0.101** |
| ROCmForge HIP | 95.4 | 768.6 | 0.84 / 0.18 |
| llama.cpp ROCm | 87.5 | 3684 | 0.77 / 0.85 |

VulkanForge ist beim Decode jetzt:
* **schneller als llama.cpp ROCm** (88.5 vs 87.5 für Qwen3, deutlich
  schneller für Llama-3.1 / DeepSeek);
* **etwa gleich auf mit ROCmForge HIP** (94.8 vs 95.4 für DeepSeek);
* **noch ~17 % hinter llama.cpp Vulkan** (88.5 vs 114.2 für Qwen3).

Beim Prefill ist die Lücke deutlich größer (~10× hinter llama.cpp
Vulkan). Das ist ein Phase-5B-Ziel — Prefill nutzt unsere GEMM-Pipeline
(`mul_mmq`), die bisher nicht für lange Token-Batches optimiert wurde.

## 4 — Profiling: vorher / nachher

`cargo run --release --example profile_forward` (median 50–99,
Qwen3-8B):

| Phase | Phase 4D (cache off) | Phase 5A (cache on) | Δ |
|---|---:|---:|---:|
| pre_setup | 537 µs | 0 µs | -537 (kein pool reset) |
| reset+begin+end+submit | 30 µs | 24 µs | -6 |
| **RECORD** | **3 570 µs** | **1 937 µs** | **-1 633 (-46 %)** |
| per-layer median | 96 µs | 48 µs | -48 (-50 %) |
| GPU_WAIT | 9 508 µs | 9 109 µs | -399 |
| readback | 22 µs | 21 µs | -1 |
| **TOTAL** | **13 669 µs** | **11 108 µs** | **-2 561 (-19 %)** |
| → Decode-Rate | **73.2 tok/s** | **90.0 tok/s** | **+23 %** |

Per-layer-Drill bei pos=100 mit Cache an:
```
RECORD wall              1871 µs
Σ per-layer dispatches   1864 µs   (99.6 %)
per-layer min/med/max      47 / 48 / 92 µs
dispatch_final + barrier    6 µs
unaccounted                 1 µs
Floor (empty record)       97 µs
```

Die ~3-4 µs/Dispatch Allokations-Overhead (Phase-4D-Schätzung im Step-1-
Report) wurden tatsächlich eingesammelt — per-layer-Zeit fällt von 96 µs
auf 48 µs, also exakt die erwartete Größenordnung.

## 5 — Cross-Phase-Progression (Qwen3-8B Decode)

| Phase | tok/s | Δ kumulativ | Headline |
|---|---:|---:|---|
| Phase 2D (initial decode) | ~14 | 0× | erste lauffähige Forward-Schleife |
| Phase 3A | ~32 | 2.3× | `cmd_pool` reuse, fence reuse |
| Phase 3C | ~48 | 3.4× | persistent CommandContext |
| Phase 4B | ~48 | 3.4× | flash-attention drop-in (tied) |
| Phase 4C | ~67 | 4.8× | split-K multi-WG attention (+41 % aggregate) |
| Phase 4D | 72.4 | 5.2× | multi-model + chat templates |
| **Phase 5A** | **88.5** | **6.3×** | **persistent descriptor sets** |
| (llama.cpp Vulkan reference) | 114.2 | (8.2×) | — |

## 6 — Code-Cleanup

* Removed: `Forward::cpu_embedding_lookup` (Phase-2C dummy, längst
  durch `decode::embedding_row` ersetzt).
* Removed: unused `hidden_bytes` local in `prefill_batch`.
* Annotated: `Row` fields in `examples/run_15prompt_bench.rs` mit
  `#[allow(dead_code)]` (Debug-Output behält sie zugänglich).
* `cargo build --release` und `cargo build --release --tests --examples`:
  **0 Rust-Warnings** (build-script `cargo:warning` Output ist nur
  informational SPIR-V-Logging).
* `cargo clippy --release --all-targets -- -W clippy::correctness`:
  keine zusätzlichen Findings.

## 7 — Tests

```
$ cargo test --release --tests
test result: ok. 17 passed; 0 failed   (regression)
test result: ok. 25 passed; 0 failed   (correctness)

$ VULKANFORGE_CB_REUSE=0 cargo test --release --tests
test result: ok. 17 passed; 0 failed   (regression, fallback)
test result: ok. 25 passed; 0 failed   (correctness, fallback)
```

**42/42 grün in beiden Modi** (Default + `VULKANFORGE_CB_REUSE=0`).

Der neue Parity-Test `phase5a_cb_reuse_parity_qwen3` läuft beide Modi
explizit und vergleicht Logits bit-exact:
```
[parity] pos= 0  max_abs_err=0.000e0  argmax_a=271  argmax_b=271
…
[parity] pos=15  max_abs_err=0.000e0  argmax_a=13   argmax_b=13
```

## 8 — Was bleibt

Auf dem Phase-5+ Backlog:

* **Stage 2A** (Pipeline-Cache + PC-Templates): geschätzt +1-2 % decode.
  Nicht prioritär.
* **Phase 5B — Prefill** (`mul_mmq` Optimierung, evtl. coopmat-Pfad):
  derzeit ~400-500 tok/s prefill vs llama.cpp Vulkan ~4300. Größtes
  noch offenes Delta.
* **Phase 5C — SPM-Tokenizer**: Mistral / Llama-2 freischalten.
* **Sampling**: temperature / top-k / top-p (heute nur greedy).
* **Quantized KV cache**: würde 32k-Kontext-Llama-3.1 in 16 GiB
  passen lassen.

## Status

Code: additiv für CB-Reuse-Pfad; Direct-Path-Code bleibt vollständig
erhalten als Fallback. Default geflippt von "false" auf "true". README
und CHANGELOG aktualisiert. Bereit zum Commit.

42/42 Tests grün, 0 Warnings, alle 3 Modelle ≥ 88 tok/s median decode.
