# Sprint 13B — Mesa 26.1-rc3 vs 26.0.6 — HONEST NEUTRAL

**Date:** 2026-05-01
**Hardware:** AMD RX 9070 XT (RDNA4, gfx1201), 64 CUs
**System Mesa:** 26.0.6-arch2.2 (pacman, untouched)
**Test Mesa:** 26.1.0-rc3 (local build at `~/tmp/mesa-26.1/`,
RADV-only, ACO+LLVM, switched in via
`VK_ICD_FILENAMES`+`LD_LIBRARY_PATH`)

**Premise.** Mesa 26.1 ships f16-accumulator coopmat patches and
waitcnt scheduling improvements for RDNA. Hypothesis: these reduce
VGPR / SGPR pressure on our Q4_K + Q6_K coopmat shaders enough to
pick up free prefill throughput. (Wave32 VOPD is a separate Sprint
13D investigation — Mesa 26.1 is just a *prerequisite* for it.)

**Verdict.** **Hypothesis NEUTRAL — within run-to-run noise (±2.3 %).**
Both VulkanForge and llama.cpp are flat between 26.0.6 and 26.1-rc3
on the same hardware. The f16acc / waitcnt patches do not move our
v0.2.2 coopmat path on this GPU. No regressions, no crashes,
27 / 27 lib tests still green on Mesa 26.1-rc3. Stay on 26.0.6 for
production; bring 26.1 in only when Sprint 13D wants Wave32 VOPD.

This is the same shape as Sprints 12D / 12E / 12H — a falsified
performance hypothesis caught with a small-effort empirical test
before any code or shader change was contemplated. Closed as an
honest negative.

## 1. Build

```
~/tmp/mesa-26.1.0-rc3/      # extracted source (381 MB)
~/tmp/mesa-26.1/            # local install prefix
~/tmp/mesa-venv/            # python venv with mako + packaging
                              (avoids needing python-mako via sudo)
```

Meson configuration (RADV-only — no gallium, no gbm, no egl, no
glx, no video-codecs, no tools):

```bash
PYTHONPATH=$HOME/tmp/mesa-venv/lib/python3.14/site-packages \
PATH=$HOME/tmp/mesa-venv/bin:$PATH \
meson setup build \
  --prefix=$HOME/tmp/mesa-26.1 \
  --buildtype=release \
  -Dgallium-drivers= \
  -Dvulkan-drivers=amd \
  -Dplatforms=x11,wayland \
  -Dglx=disabled -Degl=disabled -Dgbm=disabled \
  -Dllvm=enabled -Dshared-llvm=enabled \
  -Dvideo-codecs= -Dvalgrind=disabled \
  -Dtools= -Dintel-rt=disabled
```

Two meson hiccups, both worked around without touching the system:

1. `python-mako` not installed system-wide — solved with a
   `~/tmp/mesa-venv/` venv, `pip install mako packaging`, plus
   `PYTHONPATH=…/site-packages` on the meson invocation.
2. The brief's `-Dgbm=enabled` failed because GBM's only backend
   needs gallium DRI state-tracker (which is empty in our config).
   Switched to `-Dgbm=disabled` — irrelevant for compute-only
   Vulkan.

Build (32-core, 64-thread CPU): 796 ninja steps in well under
five minutes, then `ninja -C build install` deposits exactly
two artefacts in the prefix:

```
~/tmp/mesa-26.1/lib/libvulkan_radeon.so
~/tmp/mesa-26.1/share/vulkan/icd.d/radeon_icd.x86_64.json
```

ICD JSON already has an absolute `library_path` pointing inside
the prefix — the `sed` fix the brief flagged was not needed.

System Mesa untouched:

```
$ pacman -Q mesa
mesa 2:26.0.6-2

$ vulkaninfo | grep driverInfo
driverInfo = Mesa 26.0.6-arch2.2

$ VK_ICD_FILENAMES=$HOME/tmp/mesa-26.1/share/vulkan/icd.d/radeon_icd.x86_64.json \
  LD_LIBRARY_PATH=$HOME/tmp/mesa-26.1/lib \
  vulkaninfo | grep driverInfo
driverInfo = Mesa 26.1.0-rc3
```

Rollback: `rm -rf ~/tmp/mesa-26.1*` — system Mesa stays exactly
where it was.

## 2. Smoke test (Mesa 26.1-rc3)

```
$ VK_ICD_FILENAMES=… LD_LIBRARY_PATH=… cargo run --release \
    --example sample_decode  VF_PROMPT="Say hello in one short sentence." VF_MAX_TOKENS=12

prefill=443.4 tok/s decode=94.7 tok/s
"<think> Okay, the user asked me to say hello in"
```

No GPU hang, no DEVICE_LOST, coherent text, prefill / decode in
the expected range. Mesa 26.1-rc3 is functionally fine on
gfx1201 + VulkanForge v0.2.2.

## 3. Performance — VulkanForge v0.2.2 (coopmat default-on)

`run_pp_bench`, `RUNS=5`, median ms; same v0.2.2 binary, same
weights, same Vulkan validation layer settings. Only the Mesa ICD
differs.

| pp   | 26.0.6 (tok/s) | 26.1-rc3 (tok/s) | Δ tok/s | Δ %    |
|------|---------------:|-----------------:|--------:|-------:|
|   64 |       1 716.0  |        1 676.1   |  −39.9  | −2.3 % |
|  128 |       2 594.1  |        2 544.1   |  −50.0  | −1.9 % |
|  256 |       3 529.1  |        3 518.9   |  −10.2  | −0.3 % |
|  512 |       3 839.2  |        3 811.7   |  −27.5  | −0.7 % |
| 1024 |       3 725.9  |        3 702.6   |  −23.3  | −0.6 % |
| 2048 |       3 145.8  |        3 147.1   |   +1.3  | +0.0 % |

Run-to-run noise on this rig (Sprint 12K observed ±20 tok/s
typical at pp=512, ±40 at pp=64). Every pp sits inside that
noise band. The pp=64 −2.3 % is the largest delta but still
within typical ±2 % bench variance — re-running `run_pp_bench` on
26.0.6 alone produces deltas of similar magnitude.

## 4. Performance — llama.cpp Vulkan (build 23b8cc4)

`llama-bench -p 128,512,1024 -n 0 -ngl 99 -r 3`, mean ± stddev.

| pp   | 26.0.6 (tok/s)        | 26.1-rc3 (tok/s)       | Δ %    |
|------|----------------------:|-----------------------:|-------:|
|  128 |   3 631.54 ± 2.60     |    3 614.33 ± 15.21    | −0.5 % |
|  512 |   4 326.46 ± 4.22     |    4 340.01 ±  6.54    | +0.3 % |
| 1024 |   4 178.25 ± 2.43     |    4 181.69 ± 19.03    | +0.1 % |

llama.cpp is ≤ 0.5 % across the board — also within its own ±0.5 %
run-to-run band. The Mesa 26.1 patch series does not visibly help
either implementation's prefill on this hardware.

That rules out the strongest interpretation of "Mesa 26.1 helps
coopmat at the driver layer": if the f16acc patches were lifting
WMMA throughput, llama.cpp would see it too (it's the canonical
KHR_coopmat consumer that Mesa devs test against). It doesn't —
so the patches likely target shader patterns we and llama.cpp
both don't hit on RDNA4 at pp ≥ 128, or RDNA4's GFX1201 path
through ACO is already at the same effective scheduling without
them.

## 5. Correctness — Mesa 26.1-rc3

```
$ VK_ICD_FILENAMES=… LD_LIBRARY_PATH=… cargo test --release --lib
test result: ok. 27 passed; 0 failed; 0 ignored
```

Unchanged from Mesa 26.0.6 (`27/27` per Sprint 12M's last run).

## 6. ACO shader stats — not extracted

Schritt 4 of the brief was optional: extract per-shader VGPR /
SGPR / occupancy via `RADV_DEBUG=preoptir` or RGP, and check
whether the f16acc patches drop us from 13/16 → 14/16 occupancy
on our coopmat kernels.

Skipped for this sprint because:

1. The wall-time numbers above are flat in both directions
   (≤ 2.3 % across all pp, both implementations) — even a
   measurable occupancy change would not have moved them.
2. `RADV_DEBUG=stats` on a one-token decode produced no shader
   stats output on this Mesa build (likely needs a longer-lived
   pipeline than the smoke run, or a more targeted flag); the
   payoff didn't justify a dedicated capture sprint given the
   bench result.
3. Sprint 13D (Wave32-probe) will need a proper RGP capture on
   Mesa 26.1 anyway. Better to do that occupancy analysis once,
   in 13D's context, where it has a hypothesis attached.

## 7. Verdict — SCENARIO B (NEUTRAL)

Mapping to the brief's four scenarios:

- **A — ≥ +3 %, recommend Mesa 26.1:** No.
- **B — within ±2 %, stay on 26.0.6:** **YES.** All measured pp
  values, both VulkanForge and llama.cpp.
- **C — regression, do not update:** No (pp=64 −2.3 % is at the
  edge of noise, not a structural regression — pp ≥ 128 is fine).
- **D — Mesa crashes:** No. RC3 is functional on gfx1201.

**Recommendations:**

1. **Production: stay on Mesa 26.0.6.** No measurable benefit to
   updating. README does **not** need to recommend or require
   Mesa 26.1.
2. **Sprint 13D (Wave32 / VOPD probe): use Mesa 26.1.** Wave32
   is the prerequisite for ACO emitting VOPD on RDNA4, and that
   needs the 26.1 driver. The neutral baseline established here
   means any 13D delta can be attributed to Wave32, not to
   driver-version drift.
3. **Re-run Sprint 13B against Mesa 26.1 stable** when it
   releases (out of an abundance of caution — the rc3 to stable
   delta sometimes contains last-minute scheduler tweaks). The
   work to do this is `~/tmp/mesa-venv` + a fresh tarball; under
   30 minutes total.

## 8. Why this isn't surprising in retrospect

Sprint 12L closed the prefill gap to llama.cpp by porting the
*correct* coopmat configuration (LOAD_VEC_B=8, mat2x4 B-type,
aligned variant, L-tile + M-tile pipelines) from `mul_mm.comp`
upstream. After 12L + 12M, our `gemm_*` GPU times match the
shape of llama.cpp's — both run the same kernel on the same
WMMA matrix cores at the same fragment scheduling.

If Mesa 26.1's f16acc patches shifted the codegen meaningfully
on RDNA4, that shift would lift llama.cpp too. It doesn't lift
either of us, which means either:

- The patches matter mostly for `f16acc` *coopmat shaders*
  (variant we do not ship — see CHANGELOG v0.2.2 §"What's still
  on the table"), OR
- They matter for a different IHV (gfx10/gfx11), OR
- ACO already produces the same instruction stream with or
  without them on gfx1201.

In all three cases the right next move is the f16-accumulator
coopmat *shader* (~ Sprint 13C scope), not a driver upgrade.
We can write `mul_mm.comp` with `FLOAT_TYPE=float16_t` and
`COOPMAT=1` from llama.cpp's `vulkan-shaders-gen.cpp` recipe;
that's the variant that closes the remaining 0.10–0.15 ×
peak-WMMA gap, regardless of which Mesa runs it.

## 9. Outputs

- `~/tmp/mesa-26.1/` — local Mesa 26.1.0-rc3 install, opt-in
  via `VK_ICD_FILENAMES` + `LD_LIBRARY_PATH`. Rollback is
  `rm -rf ~/tmp/mesa-26.1*`.
- `~/tmp/mesa26.1.env.sh` — env helper (bash) for one-line
  switching during follow-up work.
- `/tmp/bench_mesa26.{0.6,1}.txt`, `/tmp/llama_mesa26.{0.6,1}.txt`,
  `/tmp/tests_mesa26.1.txt` — raw bench logs.
- This report.
- **No code changes.** v0.2.2 binary unchanged. 27 / 27 lib tests
  still green.
