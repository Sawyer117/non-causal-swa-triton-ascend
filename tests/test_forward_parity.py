#!/usr/bin/env python3
"""FORWARD parity test: Triton swa_sink_fwd vs the fp32 eager reference.

Run this on a Triton-capable GPU (the integrator's box) after `git pull`:

    python tests/test_forward_parity.py                 # bf16 kernel vs fp32 ref (atol=2e-2)
    DTYPE=float32 python tests/test_forward_parity.py    # fp32 kernel vs fp32 ref (~1e-6)
    ATOL=2e-2 RTOL=2e-2 SEED=0 python tests/test_forward_parity.py

What it checks (README §6 acceptance, forward only):
  1. Forward parity vs the fp32 eager ref: allclose + per-tensor mean-abs / mean-rel.
     bf16 diffs ~1e-2 are EXPECTED rounding; DTYPE=float32 collapses them to ~1e-6
     (that collapse is the real correctness signal — a math bug is ~1e-2 at BOTH).
  2. Sink behaviour: sink -> -inf reproduces plain windowed softmax (the kernel run
     with a huge-negative sink must match the eager ref with the same sink), and a
     finite sink diverts mass (output differs). README acceptance #3.

Exit code 0 = all checks pass, 1 = a check failed (so it's usable in CI / by eye).
"""
import os
import sys

import torch

# make ../eager_reference.py importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import swa_noncausal_sink_attention  # noqa: E402

try:
    from triton_impl import swa_noncausal_sink_attn_fwd
except Exception as e:  # noqa: BLE001
    print(f"!! could not import the Triton kernel: {type(e).__name__}: {e}")
    print("   (needs torch + triton on a CUDA/Triton-capable device)")
    raise SystemExit(1)

DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
SEED = int(os.environ.get("SEED", "0"))
B, H, L, D = 2, 8, 512, 64          # draft-block scale (matches dspark_swa_attn_bench)
W = 16                              # bidirectional half-width
SCALE = D ** -0.5


def _compare(x, ref):
    """(allclose, mean_abs, mean_rel) of x vs the fp32 reference tensor."""
    xf, rf = x.float(), ref.float()
    d = (xf - rf).abs()
    return (bool(torch.allclose(xf, rf, atol=ATOL, rtol=RTOL)),
            d.mean().item(), (d / (rf.abs() + 1e-6)).mean().item())


def main():
    if not torch.cuda.is_available():
        print("!! no CUDA device — this test runs on the integrator's GPU, not the CPU dev box.")
        raise SystemExit(1)
    dev = "cuda"
    torch.manual_seed(SEED)

    print(f">>> SWA-non-causal-sink FORWARD parity   B={B} H={H} L={L} D={D} win=±{W}")
    print(f">>> kernel dtype={DT}  vs fp32 eager ref   allclose atol={ATOL} rtol={RTOL}\n")

    # fp32 gold inputs; the kernel sees a (possibly) lower-precision cast of the same numbers
    q32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    k32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    v32 = torch.randn(B, H, L, D, device=dev, dtype=torch.float32)
    sink = torch.randn(H, device=dev, dtype=torch.float32)

    q, k, v = q32.to(DT), k32.to(DT), v32.to(DT)

    ok = True

    # ---- (1) forward parity vs fp32 eager ref ----
    ref = swa_noncausal_sink_attention(q32, k32, v32, sink, window=W, scale=SCALE,
                                       compute_dtype=torch.float32)
    o = swa_noncausal_sink_attn_fwd(q, k, v, sink, window=W, scale=SCALE)
    close, mae, mre = _compare(o, ref)
    ok &= close
    print(f"[fwd ]   allclose={close}  meanAbs={mae:.2e}  meanRel={mre:.2e}"
          f"   {'OK' if close else 'FAIL'}")

    # ---- (2a) sink -> -inf recovers plain windowed softmax ----
    sink_ninf = torch.full((H,), -1e9, device=dev, dtype=torch.float32)
    ref_ninf = swa_noncausal_sink_attention(q32, k32, v32, sink_ninf, window=W, scale=SCALE,
                                            compute_dtype=torch.float32)
    o_ninf = swa_noncausal_sink_attn_fwd(q, k, v, sink_ninf, window=W, scale=SCALE)
    close_ninf, mae_ninf, mre_ninf = _compare(o_ninf, ref_ninf)
    ok &= close_ninf
    print(f"[sink0]  sink->-inf == windowed softmax: allclose={close_ninf}  "
          f"meanAbs={mae_ninf:.2e}   {'OK' if close_ninf else 'FAIL'}")

    # ---- (2b) finite sink actually diverts mass (output must move) ----
    delta = (o.float() - o_ninf.float()).abs().mean().item()
    moved = delta > 1e-3
    ok &= moved
    print(f"[sinkE]  finite sink diverts mass: mean|o(sink)-o(-inf)|={delta:.3e}  "
          f"{'OK' if moved else 'FAIL (sink had no effect)'}")

    print("\n" + ("PASS: forward kernel matches the eager reference." if ok
                  else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
