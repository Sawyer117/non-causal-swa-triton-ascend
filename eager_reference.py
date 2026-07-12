#!/usr/bin/env python3
"""Eager ground-truth reference for the SWA-NON-CAUSAL-SINK fused attention.

This is the *math contract* a Triton kernel must reproduce (forward AND backward).
It is DEPENDENCY-FREE (pure torch, CPU-runnable) so the kernel author can diff a
kernel against it directly, including a torch.autograd.gradcheck in fp64.

The operator = the union of three properties the DSpark draft block-attention needs
and that no single Ascend-NPU primitive provides today (see README.md "Why a kernel"):

  SWA          : each query i attends only to keys j within a window
  NON-CAUSAL   : the window is BIDIRECTIONAL  (j in [i-window, i+window]), not causal
  SINK         : a per-head learnable "attention sink" logit joins the softmax as one
                 extra column, then is DROPPED from the weighted value sum
                 (StreamingLLM-style off-ramp; lets a query put mass "nowhere").

Numerics contract (must match, this is what the real model does):
  * QK^T scaling and the softmax run in fp32 even when q/k/v are bf16.
  * the sink logit is a RAW fp32 logit: NOT multiplied by `scale`, NOT masked.
  * softmax is max-subtracted over [scores | sink] jointly (the sink participates in
    the normaliser), then the sink column is discarded before the P@V matmul.
  * P is cast back to v.dtype for P@V (the model accumulates PV in v's dtype).

Two layouts are supported by the same math; the kernel should target whichever the
integrator needs (the DSpark model uses MLA-shared KV, the SWA microbench uses MHA):
  * MHA        : k,v are [B, H, L, D]  (per-head keys/values)     <- primary target
  * MLA-shared : k,v are [B, L, D]     (one latent shared by all heads; DSV4 style)

Run:  python eager_reference.py          # self-test + fp64 gradcheck
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Masks (the SWA / non-causal structure). Returned as additive 0/-inf [Lq, Lk].
# A kernel should NOT materialize these -- it should test the predicate on the fly.
# They exist here only to make the eager reference obviously correct.
# ---------------------------------------------------------------------------
def bidirectional_window_mask(lq: int, lk: int, window: int, device) -> torch.Tensor:
    """keep[i, j] = (j >= i - window) & (j <= i + window). Non-causal SWA.

    Assumes query i and key i share a position (square-ish attention). For the
    packed anchor-block layout the predicate is richer (see README + repo mask);
    this is the isolated SWA microbench structure.
    """
    i = torch.arange(lq, device=device).unsqueeze(1)
    j = torch.arange(lk, device=device).unsqueeze(0)
    keep = (j >= i - window) & (j <= i + window)
    return torch.where(keep, 0.0, NEG_INF)


# ---------------------------------------------------------------------------
# The operator.
# ---------------------------------------------------------------------------
def swa_noncausal_sink_attention(
    q: torch.Tensor,               # [B, H, L, D]
    k: torch.Tensor,               # [B, H, L, D]  (MHA)  or [B, L, D] (MLA-shared)
    v: torch.Tensor,               # same layout as k
    sink: torch.Tensor,            # [H]  per-head sink logit (fp32)
    window: int,                   # bidirectional half-width; None/<=0 => full attn
    scale: float | None = None,    # default D**-0.5
    add_mask: torch.Tensor | None = None,   # optional extra additive mask [L, Lk]
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Returns o [B, H, L, D]. See module docstring for the exact numerics contract."""
    b, h, lq, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    mla = (k.dim() == 3)  # shared latent across heads

    # scores[b,h,i,j] = q . k  (fp32)
    if mla:
        lk = k.shape[1]
        scores = torch.einsum("bhid,bjd->bhij", q, k).to(compute_dtype) * scale
    else:
        lk = k.shape[2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k).to(compute_dtype) * scale

    # SWA / non-causal structure as an additive mask (0 keep, -inf drop)
    mask = torch.zeros(lq, lk, device=q.device, dtype=compute_dtype)
    if window is not None and window > 0:
        mask = mask + bidirectional_window_mask(lq, lk, window, q.device).to(compute_dtype)
    if add_mask is not None:
        mask = mask + add_mask.to(compute_dtype)
    scores = scores + mask  # broadcast [lq, lk]

    # per-head sink: one extra logit column, raw (unscaled, unmasked)
    sink_col = sink.view(1, h, 1, 1).expand(b, -1, lq, 1).to(compute_dtype)   # [b,h,lq,1]
    combined = torch.cat([scores, sink_col], dim=-1)                          # [b,h,lq,lk+1]
    combined = combined - combined.max(dim=-1, keepdim=True).values          # stable
    probs = combined.softmax(dim=-1)[..., :lk]                               # drop sink col

    # P @ V  (P cast back to v's dtype, matching the model)
    probs = probs.to(v.dtype)
    if mla:
        o = torch.einsum("bhij,bjd->bhid", probs, v)
    else:
        o = torch.einsum("bhij,bhjd->bhid", probs, v)
    return o


# ---------------------------------------------------------------------------
# Independent, loop-based oracle (no shared einsum) -> a real parity number.
# ---------------------------------------------------------------------------
def _naive_oracle(q, k, v, sink, window, scale=None):
    b, h, lq, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    mla = (k.dim() == 3)
    lk = k.shape[1] if mla else k.shape[2]
    out = torch.zeros(b, h, lq, d, dtype=torch.float64)
    for bi in range(b):
        for hi in range(h):
            kk = (k[bi] if mla else k[bi, hi]).double()
            vv = (v[bi] if mla else v[bi, hi]).double()
            for i in range(lq):
                sc = (q[bi, hi, i].double() @ kk.T) * scale        # [lk]
                for j in range(lk):
                    if window is not None and window > 0 and not (i - window <= j <= i + window):
                        sc[j] = NEG_INF
                logits = torch.cat([sc, sink[hi].double().view(1)])  # [lk+1]
                p = torch.softmax(logits, -1)[:-1]
                out[bi, hi, i] = p @ vv
    return out


def _selftest():
    torch.manual_seed(0)
    B, H, L, D, W = 2, 4, 12, 8, 3

    # --- MHA layout, fp64 exactness vs the independent oracle ---
    q = torch.randn(B, H, L, D, dtype=torch.double)
    k = torch.randn(B, H, L, D, dtype=torch.double)
    v = torch.randn(B, H, L, D, dtype=torch.double)
    sink = torch.randn(H, dtype=torch.double)
    o = swa_noncausal_sink_attention(q, k, v, sink, window=W, compute_dtype=torch.double)
    ref = _naive_oracle(q, k, v, sink, window=W)
    err = (o.double() - ref).abs().max().item()
    print(f"[MHA]  out{tuple(o.shape)}  max|eager - oracle|={err:.2e}  (expect < 1e-10)")
    assert err < 1e-10

    # --- MLA-shared KV layout runs and is sink-sensitive ---
    kL = torch.randn(B, L, D, dtype=torch.double)
    vL = torch.randn(B, L, D, dtype=torch.double)
    o_mla = swa_noncausal_sink_attention(q, kL, vL, sink, window=W, compute_dtype=torch.double)
    print(f"[MLA]  out{tuple(o_mla.shape)}  (shared latent across {H} heads)")

    # --- the sink actually changes the output (mass diverted off-ramp) ---
    o_no = swa_noncausal_sink_attention(q, k, v, sink - 1e6, window=W, compute_dtype=torch.double)
    o_hi = swa_noncausal_sink_attention(q, k, v, sink + 5.0, window=W, compute_dtype=torch.double)
    print(f"[sink] mean|o(sink) - o(sink+5)|={ (o - o_hi).abs().mean():.3e}  "
          f"(>0 => sink diverts probability mass; sink->-inf recovers plain softmax)")
    assert (o - o_hi).abs().mean() > 0

    # --- backward: gradcheck in fp64 (the kernel's bwd must match this) ---
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    sg = sink.clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda a, b, c, s: swa_noncausal_sink_attention(a, b, c, s, window=W, compute_dtype=torch.double),
        (qg, kg, vg, sg), atol=1e-6, rtol=1e-4,
    )
    print(f"[bwd]  gradcheck(q,k,v,sink) = {ok}  (autograd-clean; grads flow to the sink too)")
    print("OK: eager reference is self-consistent (fwd vs oracle, sink-sensitive, gradcheck).")


if __name__ == "__main__":
    _selftest()
