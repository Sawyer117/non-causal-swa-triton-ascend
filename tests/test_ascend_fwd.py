#!/usr/bin/env python3
"""Validate the Ascend-shaped (1-D grid) forward kernel on a GPU, vs the eager reference.

The Ascend variant (triton_impl/swa_sink_ascend.py) has the NPU-shaped grid (1-D, no `%`) but
identical math, so it must match the eager reference exactly on CUDA. This proves the 1-D-grid
restructure is correct before moving to the A3 (where only backend lowering differs).

Run on the GPU box:  python tests/test_ascend_fwd.py    (DTYPE=float32 for the ~1e-6 gate)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, dspark_block_attention_ref, dspark_sas_window, DSV4,
)
try:
    from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend
except Exception as e:  # noqa: BLE001
    print(f"!! import failed: {type(e).__name__}: {e}"); raise SystemExit(1)

_TOL = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (2e-2, 2e-2)}
_DTYPES = ([{"bfloat16": torch.bfloat16, "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])
_DEV = "cuda"


def _row(dt, o, ref):
    atol, rtol = _TOL[dt]
    d = (o.float() - ref.float()).abs()
    close = torch.allclose(o.float(), ref.float(), atol=atol, rtol=rtol)
    print(f"  [{str(dt).replace('torch.',''):8}] allclose={close}  maxAbs={d.max().item():.2e}  "
          f"meanAbs={d.mean().item():.2e}  meanRel={(d/(ref.abs()+1e-6)).mean().item():.2e}  "
          f"{'OK' if close else 'FAIL'}")
    return close


def run_windowed(tag, B, H, L, D, wl, wr, *, mla=False):
    torch.manual_seed(0)
    scale = D ** -0.5
    q = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    ksh = (B, L, D) if mla else (B, H, L, D)
    k = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    v = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}  B={B} H={H} L={L} D={D} window=(L{wl},R{wr}) {'MLA' if mla else 'MHA'}")
    ok = True
    for dt in _DTYPES:
        qd, kd, vd = q.to(dt), k.to(dt), v.to(dt)
        ref = swa_sink_attention(qd.float(), kd.float(), vd.float(), sink, wl, wr,
                                 scale=scale, compute_dtype=torch.float32)
        o, _ = swa_sink_attn_fwd_ascend(qd, kd, vd, sink, wl, wr, scale=scale)
        ok &= _row(dt, o, ref)
    return ok


def run_dense(tag, N, BS, KV, H, D, *, mla=False):
    torch.manual_seed(0)
    scale = D ** -0.5
    qg = torch.randn(N, BS, H, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}  N={N} BS={BS} KV={KV} H={H} D={D} {'MLA' if mla else 'MHA'}  vs gold")
    ok = True
    for dt in _DTYPES:
        qg_d = qg.to(dt)
        if mla:
            kL = torch.randn(N, KV, D, device=_DEV, dtype=dt); vL = torch.randn(N, KV, D, device=_DEV, dtype=dt)
            gold = dspark_block_attention_ref(qg_d.float(), kL.float().unsqueeze(2).expand(N, KV, H, D),
                                              vL.float().unsqueeze(2).expand(N, KV, H, D), sink,
                                              scale=scale, compute_dtype=torch.float32)
            o, _ = swa_sink_attn_fwd_ascend(qg_d.permute(0, 2, 1, 3).contiguous(), kL, vL, sink,
                                            0, 0, scale=scale, dense=True, BLOCK_M=8, BLOCK_N=16)
        else:
            kg = torch.randn(N, KV, H, D, device=_DEV, dtype=dt); vg = torch.randn(N, KV, H, D, device=_DEV, dtype=dt)
            gold = dspark_block_attention_ref(qg_d.float(), kg.float(), vg.float(), sink,
                                              scale=scale, compute_dtype=torch.float32)
            o, _ = swa_sink_attn_fwd_ascend(qg_d.permute(0, 2, 1, 3).contiguous(),
                                            kg.permute(0, 2, 1, 3).contiguous(),
                                            vg.permute(0, 2, 1, 3).contiguous(), sink,
                                            0, 0, scale=scale, dense=True, BLOCK_M=8, BLOCK_N=16)
        ok &= _row(dt, o.permute(0, 2, 1, 3), gold)
    return ok


def run_gridstride(wl, wr):
    """The grid is core-capped, so each program strides over many tiles. Force tiny grids
    (num_programs=1, 3) so the stride loop iterates a lot, and confirm results are unchanged."""
    torch.manual_seed(0)
    B, H, L, D = 2, 8, 384, 64
    scale = D ** -0.5
    q = torch.randn(B, H, L, D, device=_DEV, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device=_DEV, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device=_DEV, dtype=torch.bfloat16)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    ref = swa_sink_attention(q.float(), k.float(), v.float(), sink, wl, wr,
                             scale=scale, compute_dtype=torch.float32)
    print("\n### grid-stride  (each program processes many tiles; must match regardless of grid)")
    ok = True
    for npg in (1, 3, None):    # None = the core-capped default
        o, _ = swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, num_programs=npg)
        d = (o.float() - ref.float()).abs()
        close = torch.allclose(o.float(), ref.float(), atol=2e-2, rtol=2e-2)
        ok &= close
        print(f"  num_programs={str(npg):5}  allclose={close}  maxAbs={d.max().item():.2e}  "
              f"{'OK' if close else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    print(">>> Ascend-shaped (1-D grid) forward — correctness on CUDA vs eager/gold")
    BS, WIN = DSV4["block_size"], DSV4["window_size"]
    wl, wr = dspark_sas_window(BS, WIN)
    KV = WIN + BS
    ok = True
    ok &= run_gridstride(wl, wr)
    ok &= run_windowed("[asym] windowed", 2, 8, 384, 64, wl, wr)
    ok &= run_windowed("[asym-mla] windowed MLA", 2, 8, 384, 64, wl, wr, mla=True)
    ok &= run_dense("[gold] dense", 4, BS, KV, 8, 64)
    ok &= run_dense("[gold-mla] dense MLA", 4, BS, KV, 8, 64, mla=True)
    ok &= run_windowed("[real] DSV4 H=64 D=512", 1, DSV4["num_heads"], 256, DSV4["head_dim"], wl, wr)
    print("\n" + ("PASS: 1-D-grid Ascend forward matches the reference."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
