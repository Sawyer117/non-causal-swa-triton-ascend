# Ascend-NPU port — status & A3 roadmap

The SWA-non-causal-sink Triton operator, ported to the Triton-Ascend backend. Everything here
is **validated on CUDA GPU** (against the reference / the validated CUDA kernels) so only
backend lowering + on-device perf tuning remain for the A3. The CUDA kernels
(`swa_sink_fwd.py`, `swa_sink_bwd.py`, `torch_ref.py`) are the untouched, GPU-validated baseline.

## Files (Ascend = separate from CUDA)

| File | What |
|---|---|
| `swa_sink_ascend.py` | Ascend **forward** kernel + `swa_sink_attn_fwd_ascend` |
| `swa_sink_ascend_bwd.py` | Ascend **backward** kernels (dq / dkdv-MHA / dkdv-MLA) + `swa_sink_bwd_ascend` + the autograd op `swa_sink_attn_ascend` / `dense_sink_attn_ascend` |
| `../tests/test_ascend_fwd.py`, `test_ascend_bwd.py` | GPU validation (vs eager/gold and vs the validated CUDA kernels) |

## Done (GPU-validated)

- **1-D grid** (Ascend forbids multi-dim), **core-capped** via a grid-stride loop:
  `grid = min(NUM_TILES, num_cores)`, each program strides `for tile in range(pid, NUM_TILES,
  tl.num_programs(0))`. `num_cores` = `num_aicore` on Ascend else the CUDA SM count.
- Tile → `(b, h, block)` decode with **no `%`** (`a - (a//b)*b`).
- **fp32-cast window comparisons** (Ascend vectorizes fp32 compares).
- Forward matches eager/gold (fp32 ~1e-6, bf16 ~1e-2). Backward is **bit-identical (0.0)** to the
  validated CUDA backward across windowed/dense, MHA/MLA, real DSV4, and num_programs=1/3.
- Autograd op (fwd+bwd) + default tile sizes by head_dim (small for D>=256 so D=512 fits on-chip).

## TODO on the A3 (needs the hardware — measure, don't guess)

1. **Confirm lowering** on Triton-Ascend: `tl.dot`, `tl.trans`, `tl.math.exp2`, dynamic
   `range(lo, hi, step)`, `tl.num_programs`. If any won't lower, adjust (the eager fp32 path is
   the always-correct fallback).
2. **Precision**: `input_precision="ieee"` is kept for CUDA fp32 accuracy. The Ascend Cube uses
   a different precision path (fp16/bf16 native; fp32 is the slow `ieee`/tf32x3 route — see
   skills `ascend_conv_curated` SOURCE notes). Revisit: drop `input_precision`, or use the
   tf32x3 recipe for fast-and-correct. Keep the fp32 **softmax accumulation** regardless.
3. **Block-size autotune** — Ascend autotune supports **block size + multibuffer** (NOT
   num_warps/num_stages). Add `@triton.autotune` with block-size configs keyed on (LQ,LK,D) and
   let the A3 pick, budgeted to: **UB 192KB/VEC, L0A/L0B 64KB, L0C 128KB** (hw-ascend910-9362.md).
4. **D-tiling for D=512** (the memory lever): the PV accumulator `acc[BLOCK_M, D]` at D=512 is
   64KB (BLOCK_M=32) and pressures the UB. Options: tile the QK contraction over D (load q/k in
   D-chunks, accumulate qk — no result change) to shrink the q/k working set; and/or split-D the
   output (recompute p per D-chunk — trades compute for memory). Measure whether it helps.
5. **Grid = num_aicore**: `_num_cores` reads `num_aicore` via
   `triton.runtime.driver.active.utils.get_device_properties`; confirm on the A3 that grid ≤
   cube cores (mix op) and all cores are used.
6. **Real SAS-op validation**: plug `swa_sink_attn_ascend` / the fwd into
   `../fused_sas_vs_reference_parity.py` (replace the FUSED run) to compare against the compiled
   `npu_sparse_attn_sharedkv` op + `_dspark_attention_reference` on the A3 (DTYPE=float32 → ~1e-6).

## Ascend rules already applied (skills `latency-optimizer/references/checklist.md`)

1-D grid, grid ≤ core count (grid-stride), no `%`, no continue/break, int32 index math, fp32
comparisons, contiguous per-program tiles, fp32 softmax accumulation.
