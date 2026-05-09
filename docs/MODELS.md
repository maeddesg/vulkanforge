# Supported Models & Formats

## GGUF (llama.cpp-compatible quantization)

| Format | Status | Decode 8B Llama | Notes                                    |
|--------|:------:|----------------:|------------------------------------------|
| Q4_K_M | ✅     |       121 t/s   | Primary GGUF format, all benches use it  |
| Q3_K_M | ✅     |       132 t/s   | Smaller VRAM, slight quality drop        |
| Q5_K   | ✅     |       105 t/s   |                                          |
| Q4_0   | ✅     |             —   | Less common; chat works                  |
| Q6_K   | ✅     |             —   | Less common; chat works                  |
| Q8_0   | ⚠️    |             —   | Loadable via `chat` but rejected by `bench` |

The GGUF loader covers `llama` and `qwen2` / `qwen3` architectures.
Tokenizer is read from the GGUF file; no extra setup needed.

## Native FP8 E4M3 (HuggingFace SafeTensors)

VulkanForge auto-detects all three FP8 scaling strategies from the
`config.json` `quantization_config` block + the SafeTensors header.

| Scale type           | Example model                                  | VRAM (GPU) | Decode (GPU) | Prefill pp=512* | CPU `lm_head` decode | 15-prompt coherence |
|----------------------|------------------------------------------------|-----------:|-------------:|----------------:|---------------------:|--------------------:|
| Per-tensor           | `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`   |   7.5 GiB  |     69 tok/s | **1197 tok/s**  |        47.6 tok/s    |              13/15  |
| Per-channel          | `larryvrh/Qwen2.5-14B-Instruct-FP8`            |  13.8 GiB  |     14 tok/s |   **450 tok/s** |    **17.8 tok/s** ✓  |              15/15  |
| Block-wise [128,128] | `Qwen/Qwen3-8B-FP8`                             |   8.5 GiB  |     62 tok/s |  **1118 tok/s** |   (not benched yet)  |              15/15  |

\*With native FP8 WMMA on Mesa 26.1+. Routing is capability-driven
since Sprint 47B / v0.3.16: VulkanForge picks the native path
automatically when the driver advertises
`shaderFloat8CooperativeMatrix`; on Mesa 26.0.x or any driver
without the extension, all three paths use the BF16 conversion
fallback at ~770 / 325 / 757 tok/s respectively.

The CPU `lm_head` column shows decode tok/s when
`VF_CPU_LM_HEAD=1` is set (AVX-512 Q6_K GEMV on Zen 4 7945HX,
v0.3.10). The ✓ marks the 14B win zone: per-channel FP8 with the
CPU offload **beats the GPU baseline by 32 %** while freeing
970 MB of VRAM. Per-tensor 8B trades 32 % decode for the same
VRAM saving — useful on 12 GB cards or when running multiple 8B
sessions, otherwise leave the flag off.

### FP8 quirks

- **Tokenizer not in the FP8 SafeTensors model.** Use
  `--tokenizer-from <gguf>` pointing at any GGUF from the same model
  family.
- **Block-wise needs `block_size` divisible by BM=64 and BK=16.** The
  Qwen3 / DeepSeek-V3 `[128, 128]` calibration satisfies both. Other
  block shapes silently fall through to the GEMV-loop fallback path
  (Sprint 35).
- **`weight_scale` vs `weight_scale_inv`** suffix conventions in the
  SafeTensors header are both auto-detected. Different vendors use
  different conventions; the loader handles either.

## Model architectures

| Architecture     | GGUF | FP8 (SafeTensors)        | Tokenizer | Notes                                |
|------------------|:----:|:------------------------:|-----------|--------------------------------------|
| Llama            | ✅   | ✅ per-tensor            | gpt2/llama| Llama-3.1, Llama-3.2 (8B + 14B)      |
| Qwen2 / Qwen2.5  | ✅   | ✅ per-channel           | gpt2      | ChatML, RoPE θ from `config.json`    |
| Qwen3            | ✅   | ✅ block-wise [128,128]  | gpt2      | `<think>` mode, Q/K-norm             |
| Mistral 7B       | ✅   | ⚠️                       | SPM       | GGUF works; FP8 SPM tokenizer not yet wired |
| DeepSeek-R1      | ✅   | —                        | gpt2      | R1-Distill-Llama works on GGUF; native MoE pending |

`gpt2` tokenizer covers Qwen, Llama-3 (which uses BPE), and the
DeepSeek family. SPM (Mistral / Llama-2 family) is GGUF-only at the
moment.

## What VulkanForge does **not** yet do

- **Mixture-of-experts** (DeepSeek-V3, Mixtral-style routing) — pending.
- **Speculative decode** / draft-model tokens — single-stream only.
- **Batch inference** (b > 1 concurrent prompts) — single-stream only.
- **Long-context optimizations beyond 4 k tokens** — RoPE works to
  the model's `ctx_max`, but no chunked prefill or paged-attention.
- **Quantization formats outside the table above** — IQ-formats,
  AWQ, GPTQ, GGML legacy formats are out of scope.

## Coherence test set

The full 15-prompt coherence suite (`results/sprint34cd_logs/bench_vf_15p_8b_q4.txt`)
covers:

1. Simple sequence (numerics)
2. Prime-check Python (code generation)
3. Multilingual prose (German / French / Mandarin)
4. Emoji input
5. Arithmetic with Q4_K precision (numerics edge case)
6. C++ snippet
7. Go function
8. Recursive algorithm explanation
9. Haiku composition
10. Long-context recall (in-context shadow)
11. SQL query construction
12. JSON schema generation
13. Chat persona consistency
14. Arithmetic Q4_K precision (second case)
15. Free-form storytelling

Llama-3.1-8B-FP8 + Qwen2.5-14B-FP8 are 15 / 15 on this set. Q4_K_M
8B is 12 / 15 (the three failures are the numeric Q4_K-precision
prompts — a known Q4_K format limitation, not an engine bug).
