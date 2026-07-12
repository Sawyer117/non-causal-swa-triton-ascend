#!/usr/bin/env python3
"""FORWARD parity test: Triton swa_sink_fwd vs the fp32 eager reference.

Run on a Triton-capable GPU (the integrator's box) after `git pull`:

    python tests/test_forward_parity.py                 # runs BOTH fp32 and bf16
    DTYPE=float32 python tests/test_forward_parity.py    # only fp32
    ATOL=1e-6 RTOL=1e-6 python tests/test_forward_parity.py   # override tolerances

Two precisions, two DIFFERENT bars (this is the point):
  * float32  -> the CORRECTNESS gate. With input_precision="ieee" (true fp32, no TF32)
    the kernel must match the eager ref to ~1e-6 (max-abs). Default gate atol=rtol=1e-5.
    A miss here at ~1e-3 means TF32 leaked in; at ~1e-2 means a real math bug.
  * bfloat16 -> the DEPLOYMENT-realism check. bf16 has an ~8-bit mantissa (~4e-3 rel),
    so a ~1e-2 diff vs the fp32 ref is the FORMAT FLOOR, not looseness. Default 2e-2.

Also checks (README acceptance #3): sink->-inf reproduces plain windowed softmax, and a
finite sink diverts mass. Exit 0 = all pass, 1 = a check failed.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import swa_noncausal_sink_attention  # noqa: E402

try:
    from triton_impl import swa_noncausal_sink_attn_fwd
except Exception as e:  # noqa: BLE001
    print(f"!! could not import the Triton kernel: {type(e).__name__}: {e}")
    print("   (needs torch + triton on a CUDA/Triton-capable device)")
    raise SystemExit(1)

# per-dtype default tolerances; env ATOL/RTOL override both if set
_TOL = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (2e-2, 2e-2), torch.float16: (2e-2, 2e-2)}
_ENV_ATOL = os.environ.get("ATOL")
_ENV_RTOL = os.environ.get("RTOL")
SEED = int(os.environ.get("SEED", "0"))
B, H, L, D = 2, 8, 512, 64          # draft-block scale (matches dspark_swa_attn_bench)
W = 16                              # bidirectional half-width
SCALE = D ** -0.5

_DTYPES = ([{"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[os.environ["DTYPE"]]]
           if "DTYPE" in os.environ else [torch.float32, torch.bfloat16])


def _tol(dt):
    a, r = _TOL[dt]
    return (float(_ENV_ATOL) if _ENV_ATOL else a, float(_ENV_RTOL) if _ENV_RTOL else r)


def _stats(x, ref):
    """(max_abs, mean_abs, mean_rel) of x vs the fp32 reference tensor."""
    d = (x.float() - ref.float()).abs()
    return d.max().item(), d.mean().item(), (d / (ref.abs() + 1e-6)).mean().item()


def main():
    if not torch.cuda.is_available():
        print("!! no CUDA device — this test runs on the integrator's GPU, not the CPU dev box.")
        raise SystemExit(1)
    dev = "cuda"
    torch.manual_seed(SEED)

    print(f">>> SWA-non-causal-sink FORWARD parity   B={B} H={H} L={L} D={D} win=±{W}   seed={SEED}")

    # fp32 gold inputs; each dtype sees a cast of the SAME numbers
    q32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    k32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    v32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    sink_ninf = torch.full((H,), -1e9, device=dev, dtype=torch.float32)

    ref = swa_noncausal_sink_attention(q32, k32, v32, sink, window=W, scale=SCALE,
                                       compute_dtype=torch.float32)
    ref_ninf = swa_noncausal_sink_attention(q32, k32, v32, sink_ninf, window=W, scale=SCALE,
                                            compute_dtype=torch.float32)

    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        q, k, v = q32.to(dt), k32.to(dt), v32.to(dt)
        name = str(dt).replace("torch.", "")
        print(f"\n=== dtype={name}   allclose atol={atol:g} rtol={rtol:g} ===")

        # (1) forward parity vs fp32 eager ref
        o = swa_noncausal_sink_attn_fwd(q, k, v, sink, window=W, scale=SCALE)
        mx, mae, mre = _stats(o, ref)
        close = torch.allclose(o.float(), ref.float(), atol=atol, rtol=rtol)
        ok &= close
        print(f"[fwd ]  allclose={close}  maxAbs={mx:.2e}  meanAbs={mae:.2e}  meanRel={mre:.2e}"
              f"   {'OK' if close else 'FAIL'}")

        # (2a) sink -> -inf recovers plain windowed softmax
        o_ninf = swa_noncausal_sink_attn_fwd(q, k, v, sink_ninf, window=W, scale=SCALE)
        mx0, mae0, _ = _stats(o_ninf, ref_ninf)
        close0 = torch.allclose(o_ninf.float(), ref_ninf.float(), atol=atol, rtol=rtol)
        ok &= close0
        print(f"[sink0] sink->-inf == windowed softmax: allclose={close0}  "
              f"maxAbs={mx0:.2e}   {'OK' if close0 else 'FAIL'}")

        # (2b) finite sink diverts mass (output must move)
        delta = (o.float() - o_ninf.float()).abs().mean().item()
        moved = delta > 1e-3
        ok &= moved
        print(f"[sinkE] finite sink diverts mass: mean|o(sink)-o(-inf)|={delta:.3e}  "
              f"{'OK' if moved else 'FAIL (sink had no effect)'}")

    print("\n" + ("PASS: forward kernel matches the eager reference." if ok
                  else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
