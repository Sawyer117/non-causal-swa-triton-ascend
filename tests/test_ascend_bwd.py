#!/usr/bin/env python3
"""Validate the Ascend-shaped (1-D grid) BACKWARD on CUDA or NPU, vs eager autograd grads.

The Ascend backward (triton_impl/swa_sink_ascend_bwd.py) has the NPU-shaped grid (1-D core-capped
grid-stride, no `%`, fp32 compares). Here its grads for q, k, v, sink are compared to the eager
reference's autograd grads on the SAME inputs (windowed -> swa_sink_attention; dense -> the gold
dspark_block_attention_ref). Device-portable (runs on the A3's NPU too — the CUDA backward's 2-D
grid does NOT lower on Ascend, so we use the pure-torch eager grads as the reference). Grid-stride
is checked with num_programs=1/3 (few programs stride over all tiles).

    python tests/test_ascend_bwd.py    (DTYPE=float32 -> ~1e-6; bf16 -> ~1e-2 grad rounding)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, dspark_block_attention_ref, dspark_sas_window, DSV4,
)
try:  # Ascend: maps torch.cuda.* -> NPU so "cuda"/torch.cuda.* work transparently (vllm PR #775)
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:  # noqa: BLE001
    pass
try:
    from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend
    from triton_impl.swa_sink_ascend_bwd import swa_sink_bwd_ascend
except Exception as e:  # noqa: BLE001
    print(f"!! import failed: {type(e).__name__}: {e}"); raise SystemExit(1)

_DEV = "cuda" if torch.cuda.is_available() else None   # "cuda" == the Ascend NPU after the shim
_TOL = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (2e-2, 2e-2)}
_DTYPES = ([{"bfloat16": torch.bfloat16, "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])


def _cmp(nm, a, b, atol, rtol):
    d = (a.float() - b.float()).abs()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    print(f"      d{nm:4} allclose={ok}  meanAbs={d.mean().item():.2e}  "
          f"meanRel={(d / (b.float().abs() + 1e-6)).mean().item():.2e}  maxAbs={d.max().item():.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _eager_grads(kind, q, k, v, sink, do, wl, wr, N=None, KV=None, H=None, D=None):
    """Autograd grads of the fp32 eager reference on the same inputs. Returns [dq,dk,dv,dsink]
    in the kernel's input layout (q [B,H,LQ,D] windowed, or [N,H,BS,D]->grads for dense)."""
    if kind == "windowed":
        xs = [q.float().detach().requires_grad_(True), k.float().detach().requires_grad_(True),
              v.float().detach().requires_grad_(True), sink.detach().requires_grad_(True)]
        swa_sink_attention(*xs[:3], xs[3], wl, wr, scale=D ** -0.5,
                           compute_dtype=torch.float32).mul(do.float()).sum().backward()
        return [x.grad for x in xs]
    # dense: eager works in [N,BS,H,D]; kernel in [N,H,BS,D]. mla -> k/v are [N,KV,D].
    mla = (k.dim() == 3)
    qe = q.transpose(1, 2).float().detach().requires_grad_(True)          # [N,BS,H,D]
    ke = k.float().detach().requires_grad_(True)
    ve = v.float().detach().requires_grad_(True)
    se = sink.detach().requires_grad_(True)
    # ke/ve are the KERNEL-layout leaves (MHA [N,H,KV,D] or MLA [N,KV,D]); they're transposed/
    # expanded to the eager's [N,KV,H,D] INSIDE the call, so ke.grad is already kernel-layout.
    kk = ke.unsqueeze(2).expand(N, KV, H, D) if mla else ke.transpose(1, 2)
    vv = ve.unsqueeze(2).expand(N, KV, H, D) if mla else ve.transpose(1, 2)
    dspark_block_attention_ref(qe, kk, vv, se, scale=D ** -0.5,
                               compute_dtype=torch.float32).mul(do.transpose(1, 2).float()).sum().backward()
    dq = qe.grad.transpose(1, 2)                   # qe leaf is eager-layout -> transpose back
    return [dq, ke.grad, ve.grad, se.grad]         # dk/dv already kernel-layout


def run(tag, *, mla=False, dense=False, B=2, H=8, L=384, D=64, N=4, BS=7, KV=135,
        wl=134, wr=6, bm=32, bn=32):
    torch.manual_seed(0)
    scale = D ** -0.5
    if dense:
        q = torch.randn(N, H, BS, D, device=_DEV, dtype=torch.float32)     # kernel layout
        ksh = (N, KV, D) if mla else (N, H, KV, D)
        do = torch.randn(N, H, BS, D, device=_DEV, dtype=torch.float32)
    else:
        q = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
        ksh = (B, L, D) if mla else (B, H, L, D)
        do = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    k = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    v = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}  {'MLA' if mla else 'MHA'} {'dense' if dense else 'windowed'}  tile=({bm},{bn})")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _TOL[dt]
        qd, kd, vd, dod = q.to(dt), k.to(dt), v.to(dt), do.to(dt)
        ref = _eager_grads("dense" if dense else "windowed", qd, kd, vd, sink, dod,
                           0 if dense else wl, 0 if dense else wr, N=N, KV=KV, H=H, D=D)
        o, lse = swa_sink_attn_fwd_ascend(qd, kd, vd, sink, 0 if dense else wl, 0 if dense else wr,
                                          scale=scale, dense=dense, BLOCK_M=bm, BLOCK_N=bn)
        print(f"  dtype={str(dt).replace('torch.','')}  (Ascend bwd vs eager autograd, same inputs)")
        for npg in (None, 1, 3):
            g = swa_sink_bwd_ascend(qd, kd, vd, sink, o, lse, dod, 0 if dense else wl,
                                    0 if dense else wr, dense, scale, bm, bn, num_programs=npg)
            print(f"    num_programs={npg}")
            for nm, a, b in zip("q k v sink".split(), g, ref):
                ok &= _cmp(nm, a, b, atol, rtol)
    return ok


def main():
    if _DEV is None:
        print("!! no CUDA or NPU device found"); raise SystemExit(1)
    print(f">>> Ascend-shaped (1-D grid) BACKWARD vs eager autograd grads  (dev={_DEV})")
    BS, WIN = DSV4["block_size"], DSV4["window_size"]
    wl, wr = dspark_sas_window(BS, WIN)
    KV = WIN + BS
    ok = True
    ok &= run("[asym] windowed", B=2, H=8, L=384, D=64, wl=wl, wr=wr)
    ok &= run("[asym-mla] windowed MLA", mla=True, B=2, H=8, L=384, D=64, wl=wl, wr=wr)
    ok &= run("[gold] dense", dense=True, N=4, BS=BS, KV=KV, H=8, D=64, bm=16, bn=16)
    ok &= run("[gold-mla] dense MLA", dense=True, mla=True, N=4, BS=BS, KV=KV, H=8, D=64, bm=16, bn=16)
    ok &= run("[real] DSV4 H=64 D=512", B=1, H=DSV4["num_heads"], L=256, D=DSV4["head_dim"],
              wl=wl, wr=wr, bm=16, bn=16)
    print("\n" + ("PASS: 1-D-grid Ascend backward matches eager autograd grads."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
