# Sprint 34 — LOAD_TR Quick-Test Artefacts

Test whether ACO emits `GLOBAL_LOAD_TR_B64` / `GLOBAL_LOAD_TR_B128`
for `coopMatLoad` directly from a storage buffer on gfx1201 (RDNA4).

**Result: NO.** ACO emits `buffer_load_u16 × N` + `v_perm_b32`
software byte-permute as the transpose substitute, in all three
variants. The hardware transpose-load instruction is not reached
through the GLSL → SPIR-V → ACO path today.

## Files

| File | Description |
|------|-------------|
| `test_coopmat_direct_load.comp` | ColumnMajor A / RowMajor B / Wave64 (`local_size_x = 64`) |
| `test_coopmat_rowmajor.comp`    | RowMajor A / RowMajor B / Wave64 (control) |
| `test_coopmat_wave32.comp`      | ColumnMajor A / RowMajor B / Wave32 (`local_size_x = 32`) |
| `*.spv`                         | Compiled SPIR-V (glslang 1.4.341, vulkan1.3) |
| `gfx1201_*_isa_comp.txt`        | RGA 2.14.1.3 ISA disassembly (`vk-spv-offline`) |

## Reproduce

```bash
# 1. Compile
glslangValidator --target-env vulkan1.3 -V test_coopmat_direct_load.comp \
    -o test_coopmat_direct_load.spv

# 2. Disassemble to ISA
rga -s vk-spv-offline --isa /tmp/out.txt -c gfx1201 \
    --comp test_coopmat_direct_load.spv
# Output ends up at /tmp/gfx1201_out_comp.txt
# (RGA prepends arch and appends stage to the --isa path).

# 3. Search for the hardware transpose-load instruction
grep -inE "load_tr|ds_load_tr" /tmp/gfx1201_*_comp.txt   # → empty
```

## Observed ISA (gfx1201)

### `direct` (ColumnMajor A / Wave64)

```
buffer_load_u16 v2, v2, s[4:7], null offen
buffer_load_u16 v3, v3, s[4:7], null offen
buffer_load_u16 v6, v4, s[4:7], null offen
buffer_load_u16 v5, v5, s[4:7], null offen
v_perm_b32      v4, v3, v2, 0x5040100
v_perm_b32      v5, v5, v6, 0x5040100
v_wmma_f32_16x16x16_f16 v[0:7], v[4:7], v[4:7], 0
```

### Op counts

| Variant   | buffer_load_u16 | v_perm_b32 | v_wmma_* | ds_load/store | LOAD_TR |
|-----------|-----------------|------------|----------|---------------|---------|
| direct    | 4               | 2          | 1        | 0             | **0**   |
| rowmajor  | 4               | 2          | 1        | 0             | **0**   |
| wave32    | 8               | 4          | 1        | 0             | **0**   |

Wave32 doubles per-lane work (16×16 / 32 = 8 elements per lane vs
4 for Wave64), which lines up with the observed 2× counts.

## Why this matters

The optimisation hypothesis was: directly loading the WMMA A-matrix
from a row-major SSBO with `gl_CooperativeMatrixLayoutColumnMajor`
should *semantically* match what `GLOBAL_LOAD_TR_B128` does in
hardware (load + transpose in one instruction). If ACO recognised
that pattern, the LDS staging step in `mul_coopmat_fp8_*.comp`
would become removable.

**ACO does not currently recognise the pattern.** Instead it emits
plain `buffer_load_u16` per element and synthesises the transpose
via `v_perm_b32` byte-shuffle. So the LDS staging step we have today
is functionally equivalent to what ACO would produce if we removed
LDS — there is no hidden hardware win to unlock at the application
layer.

What is interesting: there are **zero `ds_load`/`ds_store` ops** in
any of the three ISA dumps. That confirms the LDS-bypass *is*
reachable through the standard GLSL path — it is just not faster
than LDS-staging because the transpose still happens in software
(VGPR permute), not in the load.

## Context

- GPU: RX 9070 XT (gfx1201, RDNA4)
- Driver: Mesa 26.0.6 (RADV / ACO)
- glslang: 1.4.341
- RGA: 2.14.1.3
- VulkanForge: <https://github.com/maeddesg/vulkanforge>
- RDNA4 ISA reference: §11.6 WMMA Matrix Load Ops with Transpose

## Upstream pointer

If you maintain Mesa/RADV/ACO and would like to wire this up, the
SPIR-V signal is `OpCooperativeMatrixLoadKHR` with
`MatrixLayout = ColumnMajor` (operand `int_1`) over a row-major
storage buffer, paired with a subsequent `OpCooperativeMatrixMulAddKHR`
where this matrix is the A operand of a 16×16×16 WMMA. The natural
ACO target is `GLOBAL_LOAD_TR_B128`.
