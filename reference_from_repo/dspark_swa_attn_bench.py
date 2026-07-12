#!/usr/bin/env python3
"""Fallback bench for the DSpark draft's NON-CAUSAL sliding-window attention on Ascend NPU.

Training the draft needs FORWARD + BACKWARD. If SDPA can't do the non-causal (bidirectional)
sliding-window attention on NPU, we need a fallback that still has autograd. This benches the
candidates against an EAGER reference (the ground-truth math):
  - eager   : manual softmax(QK^T*scale + mask) @ V, fp32 internals -> the reference. PURE torch,
              ALWAYS has autograd on NPU. The guaranteed fallback (worst case = slower).
  - sdpa    : F.scaled_dot_product_attention(..., attn_mask=<bidir window bool mask>)
  - npu_fa  : torch_npu.npu_fusion_attention (Ascend native fused), best-effort.

For sdpa/npu_fa it reports, vs the eager reference: torch.allclose (atol/rtol), per-tensor
MEAN abs error, per-tensor MEAN rel error (for both the output and the q-grad), plus the
FORWARD and FORWARD+BACKWARD speedup RELATIVE TO EAGER (e.g. "3.5x / 1.7x").

A bf16 diff ~1e-2 vs the fp32 eager reference is EXPECTED (bf16 = 2^-7 ULP). Run DTYPE=float32
to see the diffs collapse to ~1e-6, proving it's dtype rounding, not a math bug.

Run in dspark-dsv4-base on ONE NPU:   python dspark_swa_attn_bench.py
  DTYPE=float32 python ...            # prove diffs are just bf16 rounding
  ATOL=2e-2 RTOL=2e-2 python ...      # allclose thresholds
"""
import os
import time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    print(f"!! torch_npu import failed: {e}")
    raise SystemExit(1)

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
B, H, L, D = 2, 8, 512, 64          # batch, heads, seq len, head dim (draft-block scale)
WIN = 16                            # bidirectional sliding-window half-width
SCALE = D ** -0.5
NITER = 30

qi = torch.arange(L, device=DEV).view(L, 1)
ki = torch.arange(L, device=DEV).view(1, L)
KEEP = ((ki >= qi - WIN) & (ki <= qi + WIN)).view(1, 1, L, L)     # bool [1,1,L,L], True=attend

Q0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)
K0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)
V0 = torch.randn(B, H, L, D, device=DEV, dtype=DT)


def fresh():
    return (Q0.clone().requires_grad_(True),
            K0.clone().requires_grad_(True),
            V0.clone().requires_grad_(True))


def attn_eager(q, k, v):
    s = (q.float() @ k.float().transpose(-2, -1)) * SCALE
    s = s.masked_fill(~KEEP, float("-inf"))
    return (s.softmax(dim=-1) @ v.float()).to(DT)


def attn_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=KEEP, scale=SCALE)


def attn_npu(q, k, v):
    am = (~KEEP).squeeze(0).squeeze(0).contiguous()              # [L,L] bool, True = masked out
    return torch_npu.npu_fusion_attention(
        q, k, v, H, "BNSD", atten_mask=am, scale=SCALE, keep_prob=1.0)[0]


def time_ms(step):
    for _ in range(3):
        step()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(NITER):
        step()
    torch.npu.synchronize()
    return (time.time() - t0) / NITER * 1e3


def fwd_step(fn):
    def s():
        with torch.no_grad():
            fn(*fresh())
    return s


def fb_step(fn):
    def s():
        q, k, v = fresh()
        fn(q, k, v).float().sum().backward()
    return s


def compare(x, ref):
    """(allclose, mean_abs_err, mean_rel_err) of x vs the fp32 reference tensor."""
    xf = x.float()
    d = (xf - ref).abs()
    mae = d.mean().item()
    mre = (d / (ref.abs() + 1e-6)).mean().item()
    close = torch.allclose(xf, ref, atol=ATOL, rtol=RTOL)
    return close, mae, mre


REF = {}


def bench(name, fn):
    try:
        q, k, v = fresh()
        out = fn(q, k, v)
        out.float().sum().backward()
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<7} FAILED (fwd/bwd): {type(e).__name__}: {str(e)[:60]}")
        return
    outf, gqf = out.detach().float(), q.grad.detach().float()
    fwd = time_ms(fwd_step(fn))
    fb = time_ms(fb_step(fn))
    if not REF:
        REF.update(out=outf, gq=gqf, fwd=fwd, fb=fb)
        print(f"  {name:<7} fwd={fwd:6.3f}ms  fwd+bwd={fb:6.3f}ms   speedup 1.00x / 1.00x   (reference)")
        return
    oc, oae, ore = compare(outf, REF["out"])
    gc, gae, gre = compare(gqf, REF["gq"])
    print(f"  {name:<7} fwd={fwd:6.3f}ms  fwd+bwd={fb:6.3f}ms   "
          f"speedup {REF['fwd'] / fwd:4.2f}x / {REF['fb'] / fb:4.2f}x   (fwd / fwd+bwd vs eager)")
    print(f"          out : allclose={str(oc):<5} meanAbs={oae:.2e} meanRel={ore:.2e}")
    print(f"          grad: allclose={str(gc):<5} meanAbs={gae:.2e} meanRel={gre:.2e}")


print(f">>> non-causal sliding-window attn  B={B} H={H} L={L} D={D} win=±{WIN}  dtype={DT}  iters={NITER}")
print(f">>> allclose atol={ATOL} rtol={RTOL}; meanAbs/meanRel are per-tensor averages vs the eager fp32 ref")
print(">>> speedup = eager_time / method_time (fwd and fwd+bwd)\n")
bench("eager", attn_eager)      # reference; the guaranteed autograd fallback
bench("sdpa", attn_sdpa)        # preferred fast path IF it works on NPU
bench("npu_fa", attn_npu)       # Ascend native fused, best-effort

print("\n>>> read: allclose=True with small meanAbs/meanRel -> numerically fine (bf16 rounding).")
print(">>>       speedup shows how much faster than eager; pick the fastest that's allclose=True.")
print(">>>       if a row FAILED -> that path is unsupported on NPU; eager always works.")
