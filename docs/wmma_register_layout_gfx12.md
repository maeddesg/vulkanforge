# WMMA register layout — `v_wmma_f32_16x16x16_f16` on gfx12 / RDNA 4, Wave32

Ground-truth layout for Phase 2a. Generated with the AMD Matrix Instruction Calculator (`ROCm/amd_matrix_instruction_calculator`, `rdna4` architecture, `v_wmma_f32_16x16x16_f16` instruction). This file is the reference the kernel load/store patterns must match — mis-interpreting any of these three tables produces silently transposed 16×16 output (ROCm issue #6025).

## Builtin signature (from `/opt/rocm/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def`)

```
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(V8h A, V8h B, V8f C) -> V8f D
```

- `V8h` = 8 FP16 per lane (4 VGPRs)
- `V8f` = 8 FP32 per lane (8 VGPRs)
- Feature gate: `gfx12-insts,wavefrontsize32`
- gfx12 variant has **no lane duplication** — the RDNA 3 `V16h` form is absent here.

## A matrix — A[M][K], row-major

16 rows (M), 16 K columns. Each of the 32 lanes holds 8 FP16 values across 4 VGPRs `v0..v3`. The K-axis is split into two halves by lane: K ∈ {0,1,2,3, 8,9,10,11} lives in lanes 0..15, K ∈ {4,5,6,7, 12,13,14,15} lives in lanes 16..31. M is directly `lane % 16`.

Per-lane load rule:

```
row   = lane % 16
k_set = (lane / 16) * 4                 // 0 or 4

A_regs[0] = A[row][k_set + 0]           // v0.lo
A_regs[1] = A[row][k_set + 1]           // v0.hi
A_regs[2] = A[row][k_set + 2]           // v1.lo
A_regs[3] = A[row][k_set + 3]           // v1.hi
A_regs[4] = A[row][k_set + 8]           // v2.lo
A_regs[5] = A[row][k_set + 9]           // v2.hi
A_regs[6] = A[row][k_set + 10]          // v3.lo
A_regs[7] = A[row][k_set + 11]          // v3.hi
```

Full calculator output in [`wmma_gfx12_a_layout.txt`](wmma_gfx12_a_layout.txt).

## B matrix — B[K][N], row-major

16 K rows, 16 N columns. Mirror of A: N is `lane % 16`, K is split across lane halves.

Per-lane load rule:

```
col   = lane % 16
k_set = (lane / 16) * 4                 // 0 or 4

B_regs[0] = B[k_set + 0][col]           // v0.lo
B_regs[1] = B[k_set + 1][col]           // v0.hi
B_regs[2] = B[k_set + 2][col]           // v1.lo
B_regs[3] = B[k_set + 3][col]           // v1.hi
B_regs[4] = B[k_set + 8][col]           // v2.lo
B_regs[5] = B[k_set + 9][col]           // v2.hi
B_regs[6] = B[k_set + 10][col]          // v3.lo
B_regs[7] = B[k_set + 11][col]          // v3.hi
```

Full calculator output in [`wmma_gfx12_b_layout.txt`](wmma_gfx12_b_layout.txt).

## D matrix — D[M][N], FP32, 8 VGPRs per lane

This is the output / accumulator layout. **The schema that ROCm issue #6025 highlights as the most common source of bugs.** Lane index maps to columns, not rows.

Per-lane store rule (this is the one that matters for the store back to global memory):

```
col       = lane % 16
row_start = (lane / 16) * 8

// D_acc is the V8f returned from the intrinsic.
for v in 0..8:
    row = row_start + v
    D[row][col] = D_acc[v]
```

Equivalently:

- Lanes 0..15 hold column `lane` at rows 0..7.
- Lanes 16..31 hold column `lane - 16` at rows 8..15.
- `D_acc[v]` on any lane always lives in VGPR `v{lane}`.

Full calculator output in [`wmma_gfx12_d_layout.txt`](wmma_gfx12_d_layout.txt).

## Cross-check against ROCm issue #6025

The issue states for `v_wmma_f32_16x16x16_f16` on gfx12 Wave32:

> Lane i (0..31) holds:
>   Column: i % 16
>   Rows:   (i / 16) * 8  ..  (i / 16) * 8 + 7
>   acc[j] = Element at Row (i/16)*8 + j, Column i % 16

The calculator output matches this exactly. Both sources agree. `lane` indexes the **N** (column) axis, not **M** (row).

## The transpose pitfall

A naive store pattern that assumes `lane → row` (because WMMA docs sometimes describe "rows of output assigned to lanes") would produce `D^T` instead of `D`. Phase 2a's bit-exact correctness test against a CPU reference will catch this — and the identity-matrix diagnostic (`B = I`, expect `D = A`) isolates it to the store pattern.

## Data width summary

| Matrix | Shape | Per-lane VGPRs | Per-lane halfs/floats | Total values |
|--------|-------|---------------:|----------------------:|-------------:|
| A      | 16×16 FP16 |            4 |            8 (V8h)     |          256 |
| B      | 16×16 FP16 |            4 |            8 (V8h)     |          256 |
| C / D  | 16×16 FP32 |            8 |            8 (V8f)     |          256 |
