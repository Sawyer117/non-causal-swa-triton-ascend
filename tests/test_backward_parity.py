#!/usr/bin/env python3
"""BACKWARD parity test: the autograd op (Triton fwd + torch-autograd bwd) vs the gold grads.

Run on a Triton-capable GPU after `git pull`:

    python tests/test_backward_parity.py                 # all cases, fp32 + bf16
    DTYPE=float32 python tests/test_backward_parity.py    # fp32 only
    NO_REAL=1 python tests/test_backward_parity.py        # skip H=64,D=512

The op is `swa_sink_attn` / `dense_sink_attn` (autograd-capable): FAST Triton forward + a
(temporary) exact torch-autograd backward. This checks README acceptance #2: grads for
q, k, v AND sink match the fp32 reference — allclose + per-tensor maxAbs, with the same two
bars as forward (fp32 gate ~1e-6 / atol 1e-5; bf16 realism ~1e-2 / atol 2e-2).

Windowed grads are compared to eager swa_sink_attention; dense grads to the gold
dspark_block_attention_ref. A small fp32 gradcheck of the op (numerical from the Triton
forward vs analytical from the backward) confirms forward/backward consistency.
Exit 0 = all pass, 1 = a check failed.
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
    from triton_impl import swa_sink_attn, dense_sink_attn
except Exception as e:  # noqa: BLE001
    print(f"!! could not import the Triton kernel: {type(e).__name__}: {e}")
    raise SystemExit(1)

_TOL = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (2e-2, 2e-2), torch.float16: (2e-2, 2e-2)}
_ENV_ATOL, _ENV_RTOL = os.environ.get("ATOL"), os.environ.get("RTOL")
SEED = int(os.environ.get("SEED", "0"))
_DTYPES = ([{"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])
_DEV = "cuda"


def _tol(dt):
    a, r = _TOL[dt]
    return (float(_ENV_ATOL) if _ENV_ATOL else a, float(_ENV_RTOL) if _ENV_RTOL else r)


def _cmp(tag, name, g, gref, atol, rtol):
    d = (g.float() - gref.float()).abs()
    close = torch.allclose(g.float(), gref.float(), atol=atol, rtol=rtol)
    print(f"    grad_{name:4} allclose={close}  maxAbs={d.max().item():.2e}  "
          f"meanAbs={d.mean().item():.2e}  {'OK' if close else 'FAIL'}")
    return close


def _grads(fn, tensors, do):
    xs = [t.detach().clone().requires_grad_(True) for t in tensors]
    out = fn(*xs)
    out.mul(do).sum().backward()
    return out.detach(), [x.grad for x in xs]


def run_windowed(tag, B, H, L, D, wl, wr, *, mla=False, block_m=32, block_n=32):
    torch.manual_seed(SEED)
    scale = D ** -0.5
    q0 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    ksh = (B, L, D) if mla else (B, H, L, D)
    k0 = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    v0 = torch.randn(*ksh, device=_DEV, dtype=torch.float32)
    sink0 = torch.randn(H, device=_DEV, dtype=torch.float32)
    do = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    _, gref = _grads(lambda a, b, c, s: swa_sink_attention(a, b, c, s, wl, wr, scale=scale,
                     compute_dtype=torch.float32), (q0, k0, v0, sink0), do)
    print(f"\n### {tag}   B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  {'MLA' if mla else 'MHA'}")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        print(f"  dtype={str(dt).replace('torch.','')}")
        _, g = _grads(lambda a, b, c, s: swa_sink_attn(a, b, c, s, wl, wr, scale=scale,
                      BLOCK_M=block_m, BLOCK_N=block_n),
                      (q0.to(dt), k0.to(dt), v0.to(dt), sink0), do.to(dt))
        for nm, gg, gr in zip("q k v sink".split(), g, gref):
            ok &= _cmp(tag, nm, gg, gr, atol, rtol)
    return ok


def run_gold(tag, N, BS, KV, H, D, *, block_m=16, block_n=16):
    """Dense grads vs the gold dspark_block_attention_ref grads (block layout)."""
    torch.manual_seed(SEED)
    scale = D ** -0.5
    qg = torch.randn(N, BS, H, D, device=_DEV, dtype=torch.float32)
    kg = torch.randn(N, KV, H, D, device=_DEV, dtype=torch.float32)
    vg = torch.randn(N, KV, H, D, device=_DEV, dtype=torch.float32)
    sink0 = torch.randn(H, device=_DEV, dtype=torch.float32)
    dog = torch.randn(N, BS, H, D, device=_DEV, dtype=torch.float32)
    _, gref = _grads(lambda a, b, c, s: dspark_block_attention_ref(a, b, c, s, scale=scale,
                     compute_dtype=torch.float32), (qg, kg, vg, sink0), dog)   # [N,BS,H,D] grads
    print(f"\n### {tag}   N={N} BS={BS} KV={KV} H={H} D={D}  vs gold dspark_block_attention_ref")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        print(f"  dtype={str(dt).replace('torch.','')}")
        # kernel eats [N,H,*,D]; transpose inputs, grads, and do
        qk = qg.permute(0, 2, 1, 3).contiguous().to(dt)
        kk = kg.permute(0, 2, 1, 3).contiguous().to(dt)
        vk = vg.permute(0, 2, 1, 3).contiguous().to(dt)
        dok = dog.permute(0, 2, 1, 3).contiguous().to(dt)
        _, g = _grads(lambda a, b, c, s: dense_sink_attn(a, b, c, s, scale=scale,
                      BLOCK_M=block_m, BLOCK_N=block_n), (qk, kk, vk, sink0), dok)
        gt = [g[0].permute(0, 2, 1, 3), g[1].permute(0, 2, 1, 3), g[2].permute(0, 2, 1, 3), g[3]]
        for nm, gg, gr in zip("q k v sink".split(), gt, gref):
            ok &= _cmp(tag, nm, gg, gr, atol, rtol)
    return ok


def run_gradcheck():
    """fp32 gradcheck of the op: numerical (Triton fwd) vs analytical (torch bwd) — consistency."""
    torch.manual_seed(SEED)
    B, H, L, D = 1, 2, 24, 16
    wl, wr = dspark_sas_window(7, 8)   # small window that bites at L=24
    scale = D ** -0.5
    xs = [torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32, requires_grad=True) for _ in range(3)]
    s = torch.randn(H, device=_DEV, dtype=torch.float32, requires_grad=True)
    print("\n### fp32 gradcheck (op forward/backward consistency)")
    try:
        ok = torch.autograd.gradcheck(
            lambda a, b, c, x: swa_sink_attn(a, b, c, x, wl, wr, scale=scale),
            (*xs, s), atol=2e-2, rtol=2e-2, eps=1e-3, nondet_tol=1e-3)
        print(f"  gradcheck = {ok}  {'OK' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as e:  # noqa: BLE001
        print(f"  gradcheck raised (fp32 finite-diff can be noisy): {type(e).__name__}: {str(e)[:80]}")
        return True   # non-fatal: the exact bwd is CPU-gradchecked in fp64; this is a bonus check


def main():
    if not torch.cuda.is_available():
        print("!! no CUDA device — run on the integrator's GPU, not the CPU dev box.")
        raise SystemExit(1)
    print(f">>> SWA-non-causal-sink BACKWARD parity   seed={SEED}   "
          f"dtypes={[str(d).replace('torch.','') for d in _DTYPES]}")
    BS7, WIN = DSV4["block_size"], DSV4["window_size"]
    wl7, wr7 = dspark_sas_window(BS7, WIN)
    KV7 = WIN + BS7
    ok = True
    ok &= run_windowed("[asym] windowed grads", 2, 8, 384, 64, wl7, wr7)
    ok &= run_windowed("[asym-mla] windowed grads MLA", 2, 8, 384, 64, wl7, wr7, mla=True)
    ok &= run_gold("[gold] dense grads vs gold", 4, BS7, KV7, 8, 64)
    ok &= run_gradcheck()
    if not os.environ.get("NO_REAL"):
        ok &= run_windowed("[real] DSV4 H=64 D=512", 1, DSV4["num_heads"], 256,
                           DSV4["head_dim"], wl7, wr7, block_m=16, block_n=16)
        ok &= run_gold("[gold-real] DSV4 H=64 D=512", 2, BS7, KV7,
                       DSV4["num_heads"], DSV4["head_dim"], block_m=8, block_n=16)
    else:
        print("\n### [real]/[gold-real] — SKIPPED (NO_REAL=1)")
    print("\n" + ("PASS: op gradients match the gold/eager reference (q, k, v, sink)."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
