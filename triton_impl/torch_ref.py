#!/usr/bin/env python3
"""Self-contained differentiable torch reference for SWA-non-causal-sink attention.

Pure torch, NO triton import — so it runs on a CPU box and, crucially, provides the
BACKWARD for the autograd.Function while the Triton backward kernel is still TODO
(correctness-first: fast Triton forward + exact torch-autograd backward). It mirrors the
eager reference's math (README §2) exactly so the gradients are the contract's gradients:

  scores = q.k * scale (fp32);  windowed: scores[-inf] where not (j-i in [-win_left, win_right]);
  logits = [scores | sink]; softmax; drop the sink column; o = p @ v.

Kept byte-for-byte consistent with eager_reference.swa_sink_attention / the Triton kernel
(validated on CPU: forward match + fp64 gradcheck + grad match vs the eager reference).
"""
from __future__ import annotations

import torch


def torch_swa_sink_ref(q, k, v, sink, win_left, win_right, scale, dense=False):
    """Differentiable fp32 reference. q [B,H,Lq,D]; k,v [B,H,Lk,D] (MHA) or [B,Lk,D] (MLA-shared).
    dense=True -> no window (every query sees every key), = the gold block form. Returns o
    in q.dtype (compute is fp32; grads flow to q, k, v, and sink)."""
    b, h, lq, d = q.shape
    mla = (k.dim() == 3)
    lk = k.shape[1] if mla else k.shape[2]
    # fp32 softmax accumulation for the real (bf16/fp16/fp32) op, but KEEP fp64 inputs in fp64
    # so torch.autograd.gradcheck (which runs in fp64) sees the exact math, not an fp32 downcast.
    cdt = q.dtype if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)

    if mla:
        scores = torch.einsum("bhid,bjd->bhij", qf, kf) * scale       # [B,H,Lq,Lk]
    else:
        scores = torch.einsum("bhid,bhjd->bhij", qf, kf) * scale

    if not dense:
        i = torch.arange(lq, device=q.device).view(1, 1, lq, 1)
        j = torch.arange(lk, device=q.device).view(1, 1, 1, lk)
        keep = (j >= i - win_left) & (j <= i + win_right)
        scores = scores.masked_fill(~keep, float("-inf"))

    sink_col = sink.to(cdt).view(1, h, 1, 1).expand(b, -1, lq, 1)      # [B,H,Lq,1] RAW, unscaled
    combined = torch.cat([scores, sink_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values    # stable; sink in the max
    probs = combined.softmax(dim=-1)[..., :lk]                         # drop the sink column

    if mla:
        o = torch.einsum("bhij,bjd->bhid", probs, vf)
    else:
        o = torch.einsum("bhij,bhjd->bhid", probs, vf)
    return o.to(q.dtype)
