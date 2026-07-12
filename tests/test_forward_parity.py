#!/usr/bin/env python3
"""FORWARD parity test: Triton swa_sink_attn_fwd vs the fp32 eager reference.

Run on a Triton-capable GPU (the integrator's box) after `git pull`:

    python tests/test_forward_parity.py                 # all cases, fp32 + bf16
    DTYPE=float32 python tests/test_forward_parity.py    # fp32 only (the correctness gate)
    ATOL=1e-6 RTOL=1e-6 python tests/test_forward_parity.py   # override tolerances
    NO_REAL=1 python tests/test_forward_parity.py        # skip the heavy H=64,D=512 case

Cases (each vs the fp32 eager ref, per README §4):
  1. [sym ]  symmetric microbench window (the first-step form) vs swa_noncausal_sink_attention
  2. [asym]  the REAL asymmetric window (dspark_sas_window) at toy H/D vs swa_sink_attention
  3. [real]  the asymmetric window at REAL DSV4 shapes (H=64, D=512) vs swa_sink_attention
Plus sink behaviour (README acceptance #3): sink->-inf == plain windowed softmax; finite
sink diverts mass.

Two precisions, two DIFFERENT bars (this is the point):
  * float32  -> CORRECTNESS gate. The kernel forces input_precision="ieee" (true fp32, no
    TF32), so maxAbs must be ~1e-6. Default gate atol=rtol=1e-5. ~1e-3 => TF32 leaked in;
    ~1e-2 => a real math bug.
  * bfloat16 -> DEPLOYMENT realism. bf16's ~8-bit mantissa (~4e-3 rel) makes ~1e-2 the
    FORMAT FLOOR, not looseness. Default 2e-2.

Exit 0 = all pass, 1 = a check failed.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, swa_noncausal_sink_attention, dspark_sas_window, DSV4,
)

try:
    from triton_impl import swa_sink_attn_fwd
except Exception as e:  # noqa: BLE001
    print(f"!! could not import the Triton kernel: {type(e).__name__}: {e}")
    print("   (needs torch + triton on a CUDA/Triton-capable device)")
    raise SystemExit(1)

_TOL = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (2e-2, 2e-2), torch.float16: (2e-2, 2e-2)}
_ENV_ATOL = os.environ.get("ATOL")
_ENV_RTOL = os.environ.get("RTOL")
SEED = int(os.environ.get("SEED", "0"))
_DTYPES = ([{"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])
_DEV = "cuda"


def _tol(dt):
    a, r = _TOL[dt]
    return (float(_ENV_ATOL) if _ENV_ATOL else a, float(_ENV_RTOL) if _ENV_RTOL else r)


def _stats(x, ref):
    d = (x.float() - ref.float()).abs()
    return d.max().item(), d.mean().item(), (d / (ref.abs() + 1e-6)).mean().item()


def _ref(q32, k32, v32, sink, wl, wr, scale):
    """fp32 eager reference for the asymmetric packed-SWA + sink form."""
    return swa_sink_attention(q32, k32, v32, sink, wl, wr, scale=scale,
                              compute_dtype=torch.float32)


def run_case(tag, B, H, L, D, wl, wr, *, block_m=32, block_n=32):
    """Run one shape/window case across dtypes; return (all_ok, kernel_fp32_out_or_None)."""
    torch.manual_seed(SEED)
    scale = D ** -0.5
    q32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    k32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    v32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    ref = _ref(q32, k32, v32, sink, wl, wr, scale)

    print(f"\n### {tag}   B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  tile=({block_m},{block_n})")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        q, k, v = q32.to(dt), k32.to(dt), v32.to(dt)
        try:
            o = swa_sink_attn_fwd(q, k, v, sink, wl, wr, scale=scale,
                                  BLOCK_M=block_m, BLOCK_N=block_n)
        except Exception as e:  # noqa: BLE001
            print(f"  [{str(dt).replace('torch.',''):8}] KERNEL RAISED: {type(e).__name__}: {str(e)[:70]}")
            ok = False
            continue
        mx, mae, mre = _stats(o, ref)
        close = torch.allclose(o.float(), ref.float(), atol=atol, rtol=rtol)
        ok &= close
        print(f"  [{str(dt).replace('torch.',''):8}] allclose={close}  maxAbs={mx:.2e}  "
              f"meanAbs={mae:.2e}  meanRel={mre:.2e}  (atol={atol:g})  {'OK' if close else 'FAIL'}")
    return ok


def run_sink_checks(B, H, L, D, wl, wr, *, block_m=32, block_n=32, dt=torch.bfloat16):
    """sink->-inf recovers windowed softmax; a finite sink diverts mass."""
    torch.manual_seed(SEED)
    scale = D ** -0.5
    q32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    k32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    v32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    sink_ninf = torch.full((H,), -1e9, device=_DEV, dtype=torch.float32)
    q, k, v = q32.to(dt), k32.to(dt), v32.to(dt)
    atol, rtol = _tol(dt)

    print(f"\n### sink behaviour   window=(L{wl},R{wr})  dtype={str(dt).replace('torch.','')}")
    ok = True
    ref_ninf = _ref(q32, k32, v32, sink_ninf, wl, wr, scale)
    o_ninf = swa_sink_attn_fwd(q, k, v, sink_ninf, wl, wr, scale=scale, BLOCK_M=block_m, BLOCK_N=block_n)
    mx0, _, _ = _stats(o_ninf, ref_ninf)
    close0 = torch.allclose(o_ninf.float(), ref_ninf.float(), atol=atol, rtol=rtol)
    ok &= close0
    print(f"  [sink0] sink->-inf == windowed softmax: allclose={close0}  maxAbs={mx0:.2e}  "
          f"{'OK' if close0 else 'FAIL'}")

    o_fin = swa_sink_attn_fwd(q, k, v, sink, wl, wr, scale=scale, BLOCK_M=block_m, BLOCK_N=block_n)
    delta = (o_fin.float() - o_ninf.float()).abs().mean().item()
    moved = delta > 1e-3
    ok &= moved
    print(f"  [sinkE] finite sink diverts mass: mean|o(sink)-o(-inf)|={delta:.3e}  "
          f"{'OK' if moved else 'FAIL (sink had no effect)'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("!! no CUDA device — this test runs on the integrator's GPU, not the CPU dev box.")
        raise SystemExit(1)
    print(f">>> SWA-non-causal-sink FORWARD parity   seed={SEED}   "
          f"dtypes={[str(d).replace('torch.','') for d in _DTYPES]}")

    BS, WIN = DSV4["block_size"], DSV4["window_size"]     # real 7, 128
    wl, wr = dspark_sas_window(BS, WIN)                   # (134, 6)

    ok = True
    # 1) symmetric microbench window (first-step form): win_left == win_right
    ok &= run_case("[sym ] symmetric microbench", B=2, H=8, L=512, D=64, wl=16, wr=16)
    # 2) REAL asymmetric window at toy H/D (fast math check at the real window formula)
    ok &= run_case("[asym] real window, toy H/D", B=2, H=8, L=384, D=64, wl=wl, wr=wr)
    ok &= run_sink_checks(B=2, H=8, L=384, D=64, wl=wl, wr=wr)
    # 3) REAL DSV4 shapes (H=64, D=512) — heavy; small tiles for shared-mem headroom
    if not os.environ.get("NO_REAL"):
        ok &= run_case("[real] DSV4 H=64 D=512", B=1, H=DSV4["num_heads"], L=256,
                       D=DSV4["head_dim"], wl=wl, wr=wr, block_m=16, block_n=16)
    else:
        print("\n### [real] DSV4 H=64 D=512  — SKIPPED (NO_REAL=1)")

    print("\n" + ("PASS: forward kernel matches the eager reference (symmetric + asymmetric)."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
