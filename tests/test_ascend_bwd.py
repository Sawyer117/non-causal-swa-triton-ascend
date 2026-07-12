#!/usr/bin/env python3
"""Validate the Ascend-shaped (1-D grid) BACKWARD on a GPU, vs the validated CUDA backward.

The Ascend backward (triton_impl/swa_sink_ascend_bwd.py) has the NPU-shaped grid (1-D core-capped
grid-stride, no `%`, fp32 compares) but identical math and block sizes to the CUDA backward
(swa_sink_bwd, already validated vs the gold). So it must match the CUDA backward to ~machine eps
regardless of the grid size. This isolates the grid restructure. Grid-stride is checked with
num_programs=1/3 (one/few programs stride over all tiles). Run on the GPU box:

    python tests/test_ascend_bwd.py    (DTYPE=float32 also)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import dspark_sas_window, DSV4  # noqa: E402
try:
    from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend
    from triton_impl.swa_sink_ascend_bwd import swa_sink_bwd_ascend
    from triton_impl.swa_sink_bwd import swa_sink_bwd            # validated CUDA reference
except Exception as e:  # noqa: BLE001
    print(f"!! import failed: {type(e).__name__}: {e}"); raise SystemExit(1)

_DTYPES = ([{"bfloat16": torch.bfloat16, "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])
_DEV = "cuda"


def _cmp(nm, a, b):
    d = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-3)
    print(f"      d{nm:4} maxAbs={d.max().item():.2e}  meanAbs={d.mean().item():.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def run(tag, B, H, L, D, wl, wr, *, mla=False, dense=False, N=None, BS=None, KV=None,
        bm=32, bn=32):
    torch.manual_seed(0)
    if dense:
        scale = D ** -0.5
        q = torch.randn(N, H, BS, D, device=_DEV, dtype=torch.float32)      # kernel layout [N,H,BS,D]
        ksh = (N, KV, D) if mla else (N, H, KV, D)
        k = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
        v = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
        do = torch.randn(N, H, BS, D, device=_DEV, dtype=torch.float32)
    else:
        scale = D ** -0.5
        q = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
        ksh = (B, L, D) if mla else (B, H, L, D)
        k = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
        v = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
        do = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}  {'MLA' if mla else 'MHA'} {'dense' if dense else 'windowed'}  tile=({bm},{bn})")
    ok = True
    for dt in _DTYPES:
        qd, kd, vd, dod = q.to(dt), k.to(dt), v.to(dt), do.to(dt)
        o, lse = swa_sink_attn_fwd_ascend(qd, kd, vd, sink, wl, wr, scale=scale,
                                          dense=dense, BLOCK_M=bm, BLOCK_N=bn)
        ref = swa_sink_bwd(qd, kd, vd, sink, o, lse, dod, wl, wr, dense, scale, bm, bn)   # CUDA
        print(f"  dtype={str(dt).replace('torch.','')}  (Ascend bwd vs validated CUDA bwd)")
        for npg in (None, 1, 3):
            g = swa_sink_bwd_ascend(qd, kd, vd, sink, o, lse, dod, wl, wr, dense, scale,
                                    bm, bn, num_programs=npg)
            print(f"    num_programs={npg}")
            for nm, a, b in zip("q k v sink".split(), g, ref):
                ok &= _cmp(nm, a, b)
    return ok


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    print(">>> Ascend-shaped (1-D grid) BACKWARD vs the validated CUDA backward")
    BS, WIN = DSV4["block_size"], DSV4["window_size"]
    wl, wr = dspark_sas_window(BS, WIN)
    KV = WIN + BS
    ok = True
    ok &= run("[asym] windowed", 2, 8, 384, 64, wl, wr)
    ok &= run("[asym-mla] windowed MLA", 2, 8, 384, 64, wl, wr, mla=True)
    ok &= run("[gold] dense", None, 8, None, 64, 0, 0, dense=True, N=4, BS=BS, KV=KV, bm=16, bn=16)
    ok &= run("[gold-mla] dense MLA", None, 8, None, 64, 0, 0, dense=True, mla=True, N=4, BS=BS, KV=KV, bm=16, bn=16)
    ok &= run("[real] DSV4 H=64 D=512", 1, DSV4["num_heads"], 256, DSV4["head_dim"], wl, wr, bm=16, bn=16)
    print("\n" + ("PASS: 1-D-grid Ascend backward matches the CUDA backward."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
