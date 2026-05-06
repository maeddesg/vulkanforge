# Installation & Driver Requirements

## Hardware

| GPU                                    | Status              |
|----------------------------------------|---------------------|
| AMD Radeon RX 9070 XT (gfx1201, RDNA4) | **Primary target**, all features tested |
| Other RDNA4 (RX 9070, RX 9070 GRE, AI Pro R9700) | Should work — same `gfx1201` ISA — but untested in this repo |
| Pre-RDNA4 (RDNA3, RDNA2, GCN, …)       | Not supported. The native FP8 WMMA path requires the gfx1201 cooperative-matrix ops |

VRAM:
- 8B Q4_K_M: ~5 GiB
- 8B FP8: ~7.5 GiB
- 14B FP8: ~13.8 GiB (16 GB cards work; 12 GB cards do not)

## Mesa driver

| Mesa version | Capabilities                                                                 |
|--------------|------------------------------------------------------------------------------|
| **26.0.6+**  | Full GGUF + FP8 SafeTensors support via the BF16-conversion WMMA path        |
| **26.1+**    | Adds `shaderFloat8CooperativeMatrix` → `VF_FP8_NATIVE_WMMA=1` is safe to set |

Check your Mesa version:

```bash
vulkaninfo 2>/dev/null | grep driverInfo
```

Check FP8 cooperative-matrix support (required for native WMMA):

```bash
vulkaninfo 2>/dev/null | grep shaderFloat8CooperativeMatrix
# → must show "true" before setting VF_FP8_NATIVE_WMMA=1
```

VulkanForge does **not** auto-detect the Mesa version. Setting
`VF_FP8_NATIVE_WMMA=1` on a Mesa 26.0.x driver fails at pipeline build
because the FP8 cooperative-matrix capability isn't exposed. Leave the
flag unset on older Mesa and the BF16 path runs everywhere.

### Mesa setup recipe (Arch / btrfs example)

VulkanForge has been tested with the AUR `mesa-git` builds and with
locally-compiled `mesa-26.1.0-rc3`. A typical local build:

```bash
git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git ~/tmp/mesa-26.1
cd ~/tmp/mesa-26.1
meson setup build -Dvulkan-drivers=amd -Dgallium-drivers=radeonsi \
                  -Dprefix=$HOME/tmp/mesa-26.1
meson install -C build
cat > ~/tmp/mesa26.1.env.sh <<EOF
export VK_ICD_FILENAMES=$HOME/tmp/mesa-26.1/share/vulkan/icd.d/radeon_icd.x86_64.json
export LD_LIBRARY_PATH=$HOME/tmp/mesa-26.1/lib:\$LD_LIBRARY_PATH
EOF
# Activate per-shell:
source ~/tmp/mesa26.1.env.sh
```

## Kernel parameter

For 14B+ models, the default 2 s amdgpu compute timeout is too short
for long prefills. Set:

```
amdgpu.lockup_timeout=10000,10000
```

via your bootloader (GRUB / systemd-boot). Verify with:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep amdgpu
```

Without this, `pp=1024` on Qwen2.5-14B-FP8 (~2.9 s prefill submit)
trips the GPU TDR window and triggers a ring reset.

## Build

Rust 1.85+ and Vulkan SDK headers are required for the SPV compilation
step in `build.rs` (uses shaderc).

```bash
git clone https://github.com/maeddesg/vulkanforge.git
cd vulkanforge
cargo build --release
```

The release binary lands at `target/release/vulkanforge` (~14 MB
static binary, no runtime dependencies beyond the system Vulkan
loader).

## Environment variables

Boolean flags accept `1` / `0`, `true` / `false`, case-insensitive.

| Variable                       | Default | Effect                                                                            |
|--------------------------------|---------|-----------------------------------------------------------------------------------|
| `VULKANFORGE_ENABLE_FP8`       | `0`     | Allow FP8 SafeTensors loading (auto-detects per-tensor / per-channel / block-wise) |
| `VF_FP8_NATIVE_WMMA`           | `0`     | Use native FP8 WMMA (requires Mesa 26.1+; covers all three FP8 scaling strategies) |
| `VF_CPU_LM_HEAD`               | `0`     | Offload `lm_head` to CPU as Q6_K (saves ~970 MB VRAM; AVX-512 GEMV; v0.3.10) |
| `VF_FP8_GEMM_BN`               | `32`    | GEMM tile size for the FP8 BF16 path: `16` (naive) / `32` (default) / `64` (opt-in) |
| `VF_FP8_GEMM_BN32`             | unset   | Legacy opt-out: `0` falls back to BN=16                                           |
| `VULKANFORGE_KV_FP8`           | `0`     | FP8 E4M3 KV cache (half the KV memory; +tiny perplexity)                          |
| `VULKANFORGE_COOPMAT_FP8`      | `1`     | Enable FP8 coopmat code paths                                                     |
| `VULKANFORGE_DISABLE_BARRIER_ELISION` | `0` | Debug: keep all compute barriers (Sprint 12D dirty-flag tracker)                  |
| `VULKANFORGE_DISABLE_ASYNC_DECODE` | `0` | Debug: serial decode loop instead of pipelined `pre_record` + `submit` + `wait` |
| `VF_PROMPT`                    | unset   | Single-shot chat: `chat` reads this string and exits after one response          |
| `VF_LMHEAD_HARNESS`            | `1`     | Use the dedicated F16 lm_head pipeline (Sprint 29). Set `0` for the registry path |

### CPU `lm_head` offload (`VF_CPU_LM_HEAD=1`)

When set, the loader requantizes `lm_head.weight` from FP8/FP16/BF16
to **Q6_K** at startup and keeps it in CPU pinned RAM. The decode
loop downloads the post-norm hidden state into a 16 KB host-mapped
buffer (folded into the same Vulkan submit as the layer dispatches),
then runs an AVX-512 Q6_K GEMV on the CPU.

**Benefits:**

- ~970 MB freed from VRAM (model-dependent — see
  `docs/BENCHMARKS.md` for per-model numbers).
- 14B FP8 decode is **32 % faster** than the GPU baseline on
  Zen 4 / 7945HX (17.8 vs 13.5 tok/s). The 14B GPU `lm_head` GEMV
  is bandwidth-bound on VRAM; offloading it lets DDR5 (76 GB/s)
  share the work in parallel with the GPU pipeline.
- Enables 14B FP8 on 12 GB GPUs that wouldn't otherwise fit.

**Trade-offs:**

- 8B FP8 decode: 32 % slower than GPU (47.6 vs 70 tok/s) — use
  for VRAM savings, not speed.
- Requires AVX-512F + BW + VL (Zen 4, Ice Lake+). Older CPUs
  fall through to a scalar path that is several times slower.

**Runtime dispatch tiers** (auto-selected via
`is_x86_feature_detected!`):

| CPU features          | Path                                  | Sprint |
|-----------------------|---------------------------------------|--------|
| AVX-512F + BW + VL    | Full vectorized dequant + FMA          | 41B    |
| AVX-512F only         | Hybrid (scalar dequant + AVX FMA)      | 41A    |
| no AVX-512            | Scalar reference (multi-threaded)      | 40 P2  |

Confirm AVX-512 on your CPU:

```bash
grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u
# → wants avx512f, avx512bw, avx512vl for the full kernel
```

The flag is opt-in. On non-x86 / non-AVX-512 platforms the binary
still builds, but `VF_CPU_LM_HEAD=1` falls back to the slow scalar
path.

Path / model variables:

| Variable               | Effect                                                              |
|------------------------|---------------------------------------------------------------------|
| `VF_MODEL_PATH`        | Default model path (used when `--model` is omitted)                 |
| `VK_ICD_FILENAMES`     | Mesa ICD JSON (set this to switch between system Mesa and `~/tmp/`) |
| `LD_LIBRARY_PATH`      | Mesa lib dir (paired with `VK_ICD_FILENAMES`)                       |

## Recommended defaults

For most users running on Mesa 26.1+:

```bash
export VULKANFORGE_ENABLE_FP8=1
export VF_FP8_NATIVE_WMMA=1
```

Set them in your shell profile or `~/.config/environment.d/vulkanforge.conf`.
On Mesa 26.0.x leave `VF_FP8_NATIVE_WMMA` unset.

For 14B FP8 on Zen 4 / Sapphire Rapids (or 12 GB cards anywhere
with AVX-512), additionally:

```bash
export VF_CPU_LM_HEAD=1
```

This is a clean win on 14B (faster + saves 970 MB VRAM). On 8B FP8
it's a VRAM-vs-speed trade — leave it off unless you specifically
need the VRAM headroom.

## Troubleshooting

- **`vkCreateShaderModule: SPV_KHR_bfloat16 was declared but VK_KHR_shader_bfloat16 not enabled`** —
  cosmetic validation-layer warning from the BF16-cousin shaders in
  the tree. Pipelines still build; functionality unaffected.
- **`unsupported model for bench: arch=llama, quant=Q8_0`** — the
  bench harness is Q4_K_M-only. `vulkanforge chat` accepts Q8_0
  GGUF. To bench Q8_0, use llama.cpp.
- **`Selected TritonFp8BlockScaledMMKernel … Performance might be sub-optimal`** —
  this is vLLM's warning, not VulkanForge. Means vLLM has no
  pre-tuned config for gfx1201.
- **Chat output is `!!!!!!`** — likely a numerical issue in a new
  shader path. Sprint 38 P2's honest-negative is documented; the
  rule is **always smoke-test chat coherence before trusting bench
  numbers** (`vulkanforge bench` measures throughput, not output
  correctness).
