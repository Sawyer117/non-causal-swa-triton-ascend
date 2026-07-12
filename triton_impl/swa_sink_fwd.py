#!/usr/bin/env python3
"""Triton FORWARD kernel for SWA-non-causal-sink attention.

Forward-only fused operator specified in ../README.md and ../eager_reference.py. One kernel,
two modes (both with the per-head attention sink), matching the two views in README §4:

  * WINDOWED self-attention (the PACKED-SWA view an efficient kernel implements): q,k,v over a
    packed sequence [B,H,L,D], ASYMMETRIC sliding window keep[i,j]=(j>=i-win_left)&(j<=i+win_right).
    -> swa_sink_attn_fwd / swa_noncausal_sink_attn_fwd.
  * DENSE cross-attention (the gold BLOCK view = _dspark_attention_reference): per draft block,
    BS queries attend DENSELY to KV=window+BS keys [ctx | block]. q [B,H,Lq,D], k,v [B,H,Lk,D],
    Lq != Lk, no window. -> dense_sink_attn_fwd (the gold-parity entry point).

Both layouts also support MLA-shared K/V (num_kv_heads=1 -> k,v = [B,Lk,D], one latent shared
across all H query heads) — realized in the wrappers by passing head-stride 0 on K/V, so the
kernel body is layout-agnostic.

FORWARD ONLY (no autograd yet) — validated no-grad against the fp32 eager reference; the
backward pass is a separate step.

Numerics (README §2): scores = q.k*scale in fp32; per-head sink is a RAW fp32 logit that joins
the softmax max/denominator but never the P@V sum; P cast back to v.dtype for P@V. exp2 in the
log2 domain. The sink is injected by SEEDING the online softmax as a virtual key (m=sink*log2e,
l=1, acc=0) -> it stays in the denominator, out of the numerator, with no extra column. As
sink -> -inf its alpha collapses on the first real key => plain (windowed or dense) softmax.

Precision: QK^T and P@V use input_precision="ieee" so fp32 inputs get TRUE fp32 (CUDA tl.dot
defaults to TF32, capping accuracy at ~1e-3); bf16 inputs are unaffected (bf16*bf16 -> fp32 acc).
fp32 softmax accumulation regardless of input dtype (README criterion #1).

Backend note: CUDA Triton first (what the integrator validates on today). The Ascend-NPU port
is a distinct backend — see /workspace/skills/triton-ascend for its extra constraints.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .swa_sink_bwd import swa_sink_bwd      # fused Triton backward (dq/dk/dv + dsink)

# 1/ln(2); folds exp(x) into exp2(x*LOG2E). Must be a tl.constexpr: Triton >=3.x forbids
# @jit kernels from reading plain module globals (only constexpr globals are allowed).
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _swa_sink_fwd_kernel(
    Q, K, V, Sink, Out, Lse,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, LQ, LK, WIN_LEFT, WIN_RIGHT,
    scale,
    WINDOWED: tl.constexpr, FP32_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    # One program = one (batch*head, query-block). MLA-shared K/V is expressed by the caller
    # passing stride_kh = stride_vh = 0 (all heads read the same latent) — body is unchanged.
    pid_m = tl.program_id(0)       # which query block
    pid_bh = tl.program_id(1)      # which (batch, head)
    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)          # query rows [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                    # head dims  [BLOCK_D]
    d_mask = offs_d < D

    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < LQ) & d_mask[None, :]
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---- seed the online softmax WITH THE SINK as a virtual key ----
    sink_val = tl.load(Sink + h).to(tl.float32)      # per-head fp32 scalar, RAW (unscaled)
    # seed as a virtual key: m = sink*log2e, l = 1 (= exp2(sink-sink)), acc = 0.
    # (zeros + scalar, not tl.full with a runtime value, which some Triton versions reject.)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink_val * LOG2E   # running max incl. sink
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)               # denom incl. sink term
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)          # numerator (sink absent)

    qk_scale = scale * LOG2E     # fold scale AND log2(e) into the scores once

    # ---- key-block range ----
    if WINDOWED:
        # only blocks overlapping [m_start - win_left, m_start + BLOCK_M - 1 + win_right]
        lo = m_start - WIN_LEFT
        lo = tl.maximum(lo, 0)
        lo = (lo // BLOCK_N) * BLOCK_N               # align down to a key-block boundary
        hi = tl.minimum(m_start + BLOCK_M + WIN_RIGHT, LK)
    else:
        lo = 0                                        # dense: attend to every key
        hi = LK

    for n_start in range(lo, hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)      # key rows [BLOCK_N]

        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k_mask = (offs_n[:, None] < LK) & d_mask[None, :]
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        # scores = q . k^T, fp32 accumulate; masked d-lanes are 0*0 so the dot is exact over D.
        qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32)
        if not FP32_QK:
            # round QK to the input dtype -> mimic the production eager path (torch
            # einsum(bf16,bf16) rounds QK to bf16). No-op for fp32 inputs.
            qk = qk.to(q_block.dtype).to(tl.float32)
        qk = qk * qk_scale

        if WINDOWED:
            keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                    & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT)
                    & (offs_n[None, :] < LK))
        else:
            keep = offs_n[None, :] < LK               # dense, just drop padded tail lanes
        qk = tl.where(keep, qk, float("-inf"))

        # online softmax update (log2 domain)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])          # [BLOCK_M, BLOCK_N], fp32
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (V + b * stride_vb + h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v_mask = (offs_n[:, None] < LK) & d_mask[None, :]
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        # P@V: cast P back to v.dtype per the contract; accumulate in fp32 (ieee true fp32 path)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block, input_precision="ieee")
        m_i = m_ij

    o = acc / l_i[:, None]                            # sink is in l_i, absent from acc
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    o_mask = (offs_m[:, None] < LQ) & d_mask[None, :]
    tl.store(o_ptrs, o.to(Out.dtype.element_ty), mask=o_mask)

    # logsumexp (log2 domain, incl. the sink) -> the backward recomputes p = exp2(qk2 - lse).
    # Lse is [B, H, LQ] contiguous: offset = (b*H + h)*LQ + offs_m.
    lse = m_i + tl.log2(l_i)                          # [BLOCK_M]
    tl.store(Lse + (b * H + h) * LQ + offs_m, lse, mask=offs_m < LQ)


def _kv_strides(k):
    """Strides (b, h, n, d) for K/V that may be MHA [B,H,Lk,D] or MLA-shared [B,Lk,D].
    MLA-shared -> head-stride 0 so all query heads read the same latent."""
    if k.dim() == 4:
        return k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    if k.dim() == 3:                                   # [B, Lk, D] -> broadcast over heads
        return k.stride(0), 0, k.stride(1), k.stride(2)
    raise AssertionError("k/v must be [B,H,Lk,D] (MHA) or [B,Lk,D] (MLA-shared)")


def _launch(q, k, v, sink, LQ, LK, win_left, win_right, windowed, scale, BLOCK_M, BLOCK_N,
            fp32_qk=True):
    """Returns (o, lse). lse [B,H,LQ] fp32 = m + log2(l) (incl. sink) for the backward."""
    B, H, _, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(B, H, LQ, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    skb, skh, skn, skd = _kv_strides(k)
    svb, svh, svn, svd = _kv_strides(v)
    grid = (triton.cdiv(LQ, BLOCK_M), B * H)
    _swa_sink_fwd_kernel[grid](
        q, k, v, sink, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        skb, skh, skn, skd,
        svb, svh, svn, svd,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, LQ, LK, win_left, win_right,
        scale,
        WINDOWED=windowed, FP32_QK=fp32_qk,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )
    return o, lse


def swa_sink_attn_fwd(q, k, v, sink, win_left, win_right, scale=None, BLOCK_M=32, BLOCK_N=32,
                      fp32_qk=True):
    """Forward-only ASYMMETRIC windowed self-attention + sink (packed-SWA view). o [B,H,L,D].

    q [B,H,L,D]; k,v [B,H,L,D] (MHA) or [B,L,D] (MLA-shared). keep[i,j]=(j>=i-win_left)&(j<=
    i+win_right); use eager_reference.dspark_sas_window(block_size, window) for (win_left,
    win_right). fp32_qk=True (default) keeps fp32 QK accumulation; fp32_qk=False rounds QK to
    the input dtype to mimic the production torch-eager path (bf16 einsum). Validates against
    eager_reference.swa_sink_attention."""
    assert q.is_cuda, "kernel needs a CUDA (or Triton-capable) device"
    assert q.dim() == 4, "q must be [B,H,L,D]"
    assert win_left >= 0 and win_right >= 0, "window half-widths must be >= 0"
    L = q.shape[2]
    return _launch(q, k, v, sink, L, L, win_left, win_right, True, scale, BLOCK_M, BLOCK_N, fp32_qk)[0]


def dense_sink_attn_fwd(q, k, v, sink, scale=None, BLOCK_M=32, BLOCK_N=32, fp32_qk=True):
    """Forward-only DENSE cross-attention + sink = the gold BLOCK form. o [B,H,Lq,D].

    q [B,H,Lq,D]; k,v [B,H,Lk,D] (MHA) or [B,Lk,D] (MLA-shared), Lq != Lk allowed. Every query
    attends to ALL Lk keys (no window). This reproduces dspark_block_attention_ref /
    _dspark_attention_reference (feed the block as [N->B, H, BS->Lq, D] x [N, H, KV->Lk, D]).
    fp32_qk as in swa_sink_attn_fwd."""
    assert q.is_cuda, "kernel needs a CUDA (or Triton-capable) device"
    assert q.dim() == 4, "q must be [B,H,Lq,D]"
    Lq = q.shape[2]
    Lk = k.shape[-2] if k.dim() == 4 else k.shape[1]
    return _launch(q, k, v, sink, Lq, Lk, 0, 0, False, scale, BLOCK_M, BLOCK_N, fp32_qk)[0]


def swa_noncausal_sink_attn_fwd(q, k, v, sink, window, scale=None, BLOCK_M=32, BLOCK_N=32):
    """[COMPAT] SYMMETRIC window (win_left == win_right == window) — the first-step microbench
    form. Thin wrapper over swa_sink_attn_fwd; the REAL model uses the asymmetric window."""
    assert window is not None and window > 0, "symmetric wrapper needs a finite window > 0"
    return swa_sink_attn_fwd(q, k, v, sink, window, window, scale=scale,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


# ---------------------------------------------------------------------------
# Autograd wrapper: FAST Triton forward + (temporary) exact torch-autograd backward.
# The backward recomputes the fp32 differentiable reference and lets torch autograd produce
# grad_{q,k,v,sink} — correctness-first. Replacing this with a fused Triton backward kernel
# (dq/dk/dv on the fly + dsink) is the next step; the public API here stays the same.
# ---------------------------------------------------------------------------
class _SwaSinkAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, win_left, win_right, scale, dense, BLOCK_M, BLOCK_N):
        Lk = (k.shape[2] if k.dim() == 4 else k.shape[1])
        LQ = q.shape[2]
        wl, wr = (0, 0) if dense else (win_left, win_right)
        o, lse = _launch(q, k, v, sink, LQ, Lk, wl, wr, not dense, scale, BLOCK_M, BLOCK_N)
        ctx.save_for_backward(q, k, v, sink, o, lse)
        ctx.win_left, ctx.win_right, ctx.scale, ctx.dense = win_left, win_right, scale, dense
        ctx.BLOCK_M, ctx.BLOCK_N = BLOCK_M, BLOCK_N
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sink, o, lse = ctx.saved_tensors
        dq, dk, dv, ds = swa_sink_bwd(q, k, v, sink, o, lse, do, ctx.win_left, ctx.win_right,
                                      ctx.dense, ctx.scale, ctx.BLOCK_M, ctx.BLOCK_N)
        # non-tensor args (win_left..BLOCK_N) get None
        return (dq, dk, dv, ds, None, None, None, None, None, None)


def swa_sink_attn(q, k, v, sink, win_left, win_right, scale=None, BLOCK_M=32, BLOCK_N=32):
    """AUTOGRAD-CAPABLE asymmetric windowed self-attention + sink (fwd+bwd). o [B,H,L,D].
    Fast Triton forward, exact torch-autograd backward (grads to q, k, v AND sink)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return _SwaSinkAttnFn.apply(q, k, v, sink, win_left, win_right, scale, False, BLOCK_M, BLOCK_N)


def dense_sink_attn(q, k, v, sink, scale=None, BLOCK_M=32, BLOCK_N=32):
    """AUTOGRAD-CAPABLE dense cross-attention + sink = the gold BLOCK form (fwd+bwd). o [B,H,Lq,D]."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return _SwaSinkAttnFn.apply(q, k, v, sink, 0, 0, scale, True, BLOCK_M, BLOCK_N)
