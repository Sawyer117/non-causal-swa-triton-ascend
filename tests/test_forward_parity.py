#!/usr/bin/env python3
"""FORWARD parity test: Triton kernel vs the fp32 eager reference AND the gold block form.

Run on a Triton-capable GPU (the integrator's box) after `git pull`:

    python tests/test_forward_parity.py                 # all cases, fp32 + bf16
    DTYPE=float32 python tests/test_forward_parity.py    # fp32 only (the correctness gate)
    ATOL=1e-6 RTOL=1e-6 python tests/test_forward_parity.py   # override tolerances
    NO_REAL=1 python tests/test_forward_parity.py        # skip the heavy H=64,D=512 cases

Coverage (README §3/§4):
  WINDOWED self-attention (packed-SWA view) vs eager swa_sink_attention:
    [sym ]      symmetric microbench window (first-step form)
    [asym]      real asymmetric window dspark_sas_window(block=7,window=128)=(L134,R6)
    [asym-mla]  ^ but MLA-shared K/V (num_kv_heads=1 — the REAL model layout)
    [asym-b5]   block_size=5 -> window (L132,R4)
    [real]      real DSV4 shapes H=64 D=512 (skip with NO_REAL=1)
  DENSE cross-attention (the gold BLOCK form) vs dspark_block_attention_ref:
    [gold]      block shapes q[N,BS,H,D] x kv[N,KV,H,D], KV=window+BS, toy H/D
    [gold-mla]  ^ MLA-shared (num_kv_heads=1)
    [gold-real] block shapes at real DSV4 H=64 D=512 (skip with NO_REAL=1)
  sink behaviour: sink->-inf == plain (windowed/dense) softmax; a finite sink diverts mass.

Two precisions, two bars: float32 = correctness gate (ieee true fp32, maxAbs ~1e-6, atol 1e-5);
bfloat16 = deployment realism (~1e-2 is the mantissa floor, atol 2e-2). Exit 0=pass, 1=fail.
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
    from triton_impl import swa_sink_attn_fwd, dense_sink_attn_fwd
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


def _row(tag, dt, o, ref, atol, rtol):
    mx, mae, mre = _stats(o, ref)
    close = torch.allclose(o.float(), ref.float(), atol=atol, rtol=rtol)
    print(f"  [{str(dt).replace('torch.',''):8}] allclose={close}  maxAbs={mx:.2e}  "
          f"meanAbs={mae:.2e}  meanRel={mre:.2e}  (atol={atol:g})  {'OK' if close else 'FAIL'}")
    return close


def run_windowed(tag, B, H, L, D, wl, wr, *, mla=False, block_m=32, block_n=32):
    """Windowed self-attention vs eager swa_sink_attention (MHA or MLA-shared K/V).

    The reference is computed on the SAME (dtype-rounded) inputs the kernel sees, upcast to
    fp32 — so the error isolates the KERNEL's fidelity (fp32 accumulation + bf16 P@V + bf16
    output rounding), NOT the fp32->bf16 input rounding (which isn't the kernel's job)."""
    torch.manual_seed(SEED)
    scale = D ** -0.5
    q32 = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    kshape = (B, L, D) if mla else (B, H, L, D)
    k32 = torch.randn(*kshape, device=_DEV, dtype=torch.float32)
    v32 = torch.randn(*kshape, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}   B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  "
          f"{'MLA-shared' if mla else 'MHA'}  tile=({block_m},{block_n})")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        qd, kd, vd = q32.to(dt), k32.to(dt), v32.to(dt)
        ref = swa_sink_attention(qd.float(), kd.float(), vd.float(), sink, wl, wr,
                                 scale=scale, compute_dtype=torch.float32)  # same inputs, fp32 compute
        try:
            o = swa_sink_attn_fwd(qd, kd, vd, sink, wl, wr,
                                  scale=scale, BLOCK_M=block_m, BLOCK_N=block_n)
        except Exception as e:  # noqa: BLE001
            print(f"  [{str(dt).replace('torch.',''):8}] KERNEL RAISED: {type(e).__name__}: {str(e)[:60]}")
            ok = False; continue
        ok &= _row(tag, dt, o, ref, atol, rtol)
    return ok


def run_gold(tag, N, BS, KV, H, D, *, mla=False, block_m=16, block_n=16):
    """Dense cross-attention vs the gold dspark_block_attention_ref at BLOCK shapes.
    gold: q[N,BS,H,D], k/v[N,KV,H,D]; kernel eats q[N,H,BS,D], k/v[N,H,KV,D] (transpose).
    The gold is computed on the SAME (dtype-rounded) inputs the kernel sees, upcast to fp32,
    so the error isolates kernel fidelity, not the input's fp32->bf16 rounding."""
    torch.manual_seed(SEED)
    scale = D ** -0.5
    qg = torch.randn(N, BS, H, D, device=_DEV, dtype=torch.float32)
    if mla:   # one latent KV head shared across all H (num_kv_heads=1)
        kL = torch.randn(N, KV, D, device=_DEV, dtype=torch.float32)
        vL = torch.randn(N, KV, D, device=_DEV, dtype=torch.float32)
    else:
        kg = torch.randn(N, KV, H, D, device=_DEV, dtype=torch.float32)
        vg = torch.randn(N, KV, H, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    print(f"\n### {tag}   N={N} BS={BS} KV={KV} H={H} D={D}  {'MLA-shared' if mla else 'MHA'}  "
          f"tile=({block_m},{block_n})  vs gold dspark_block_attention_ref")
    ok = True
    for dt in _DTYPES:
        atol, rtol = _tol(dt)
        qg_d = qg.to(dt)
        if mla:
            kL_d, vL_d = kL.to(dt), vL.to(dt)
            kg_ref = kL_d.float().unsqueeze(2).expand(N, KV, H, D)     # [N,KV,H,D] for the gold
            vg_ref = vL_d.float().unsqueeze(2).expand(N, KV, H, D)
            k_kv, v_kv = kL_d, vL_d                                    # [N,KV,D] MLA path for the kernel
        else:
            kg_d, vg_d = kg.to(dt), vg.to(dt)
            kg_ref, vg_ref = kg_d.float(), vg_d.float()
            k_kv = kg_d.permute(0, 2, 1, 3).contiguous(); v_kv = vg_d.permute(0, 2, 1, 3).contiguous()
        gold = dspark_block_attention_ref(qg_d.float(), kg_ref, vg_ref, sink,
                                          scale=scale, compute_dtype=torch.float32)   # [N,BS,H,D]
        try:
            o = dense_sink_attn_fwd(qg_d.permute(0, 2, 1, 3).contiguous(), k_kv, v_kv, sink,
                                    scale=scale, BLOCK_M=block_m, BLOCK_N=block_n)
        except Exception as e:  # noqa: BLE001
            print(f"  [{str(dt).replace('torch.',''):8}] KERNEL RAISED: {type(e).__name__}: {str(e)[:60]}")
            ok = False; continue
        ok &= _row(tag, dt, o.permute(0, 2, 1, 3), gold, atol, rtol)   # -> [N,BS,H,D]
    return ok


def run_sink_checks():
    """sink->-inf recovers plain softmax; finite sink diverts mass — on both windowed & dense."""
    torch.manual_seed(SEED)
    dt = _DTYPES[-1]        # use the loosest dtype present (bf16 by default)
    atol, rtol = _tol(dt)
    print(f"\n### sink behaviour   dtype={str(dt).replace('torch.','')}")
    ok = True

    # windowed
    B, H, L, D = 2, 8, 384, 64
    wl, wr = dspark_sas_window(7, 128)
    scale = D ** -0.5
    q = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=_DEV, dtype=torch.float32)
    sink = torch.randn(H, device=_DEV, dtype=torch.float32)
    ninf = torch.full((H,), -1e9, device=_DEV, dtype=torch.float32)
    qd, kd, vd = q.to(dt), k.to(dt), v.to(dt)
    ref_ninf = swa_sink_attention(qd.float(), kd.float(), vd.float(), ninf, wl, wr,
                                  scale=scale, compute_dtype=torch.float32)   # same inputs
    o_ninf = swa_sink_attn_fwd(qd, kd, vd, ninf, wl, wr, scale=scale)
    c0 = torch.allclose(o_ninf.float(), ref_ninf.float(), atol=atol, rtol=rtol)
    o_fin = swa_sink_attn_fwd(qd, kd, vd, sink, wl, wr, scale=scale)
    d = (o_fin.float() - o_ninf.float()).abs().mean().item()
    ok &= c0 and d > 1e-3
    print(f"  [win  sink0] sink->-inf == windowed softmax: allclose={c0}  {'OK' if c0 else 'FAIL'}")
    print(f"  [win  sinkE] finite sink diverts mass: mean|Δ|={d:.3e}  {'OK' if d>1e-3 else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("!! no CUDA device — this test runs on the integrator's GPU, not the CPU dev box.")
        raise SystemExit(1)
    print(f">>> SWA-non-causal-sink FORWARD parity   seed={SEED}   "
          f"dtypes={[str(d).replace('torch.','') for d in _DTYPES]}")
    BS7, WIN = DSV4["block_size"], DSV4["window_size"]     # 7, 128
    wl7, wr7 = dspark_sas_window(BS7, WIN)                 # (134, 6)
    wl5, wr5 = dspark_sas_window(5, WIN)                   # (132, 4)
    KV7 = WIN + BS7                                        # 135
    ok = True

    # --- windowed self-attention ---
    ok &= run_windowed("[sym ] symmetric microbench", 2, 8, 512, 64, 16, 16)
    ok &= run_windowed("[asym] real window, toy H/D", 2, 8, 384, 64, wl7, wr7)
    ok &= run_windowed("[asym-mla] MLA-shared K/V", 2, 8, 384, 64, wl7, wr7, mla=True)
    ok &= run_windowed("[asym-b5] block=5 window", 2, 8, 320, 64, wl5, wr5)

    # --- dense = gold block-form parity ---
    ok &= run_gold("[gold] block form, toy H/D", 4, BS7, KV7, 8, 64)
    ok &= run_gold("[gold-mla] block form MLA", 4, BS7, KV7, 8, 64, mla=True)

    ok &= run_sink_checks()

    # --- real DSV4 shapes (heavy) ---
    if not os.environ.get("NO_REAL"):
        ok &= run_windowed("[real] DSV4 H=64 D=512", 1, DSV4["num_heads"], 256,
                           DSV4["head_dim"], wl7, wr7, block_m=16, block_n=16)
        ok &= run_gold("[gold-real] DSV4 block H=64 D=512", 2, BS7, KV7,
                       DSV4["num_heads"], DSV4["head_dim"], block_m=8, block_n=16)
    else:
        print("\n### [real]/[gold-real] DSV4 H=64 D=512  — SKIPPED (NO_REAL=1)")

    print("\n" + ("PASS: forward kernel matches eager (windowed) AND the gold block form (dense)."
                  if ok else "FAIL: see rows above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
