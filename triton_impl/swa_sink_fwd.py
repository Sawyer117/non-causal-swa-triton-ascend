#!/usr/bin/env python3
"""Triton FORWARD kernel for SWA-non-causal-sink attention (MHA layout).

This is the first artifact of the fused operator specified in ../README.md and
../eager_reference.py. FORWARD ONLY (no autograd yet) — validated no-grad against
the fp32 eager reference; the backward pass is a separate step.

Math it reproduces (see README §2, the exact numerics contract):
  scores[i,j] = (q_i . k_j) * scale                       # fp32 accumulate
  scores[i,j] = -inf   if not (i-W <= j <= i+W)           # bidirectional SWA, on the fly
  logits      = [ scores_i | sink[h] ]                    # sink: RAW fp32, unscaled, unmasked
  p           = softmax(logits)[:-1]                       # sink in the normaliser, then dropped
  o_i         = sum_j p[j] * v_j                           # P cast to v.dtype for P@V

Key design points
-----------------
* SINK as a seeded virtual key. The per-head sink is a single extra softmax column
  that is constant across all keys and never enters the value sum. We inject it by
  SEEDING the online-softmax running stats as if the sink were the very first key:
      m = sink * log2(e)      (running max, in the log2 domain)
      l = 1.0                 (= exp2(sink - sink); the sink's term in the denominator)
      acc = 0                 (the sink contributes NOTHING to the numerator)
  Then real keys stream in with the standard online update. The sink therefore stays
  in the denominator and out of P@V for free — no extra column is ever materialized.
  As sink -> -inf its alpha collapses to 0 on the first real key => plain windowed
  softmax (README acceptance #3).

* Bidirectional window predicate, tested on the fly (README §4a) — never a dense
  [L,L] mask. Each (query-block, key-block) pair applies keep=(j>=i-W)&(j<=i+W), and
  the inner loop only visits key-blocks overlapping [m_start-W, m_end-1+W] (the SWA win).

* fp32 softmax accumulation regardless of input dtype (README criterion #1). QK^T and
  P@V accumulate in fp32; P is cast back to v.dtype before P@V (matching the model).

* exp2 instead of exp: every logit is pre-multiplied by log2(e) so softmax(x) via
  exp2 is bit-for-bit the same math (and the fast path on-device).

Backend note: written for CUDA Triton first (the hardware the integrator validates on
today). The Ascend-NPU port is a distinct backend — see /workspace/skills/triton-ascend
for its extra constraints (1-D grid, fp32 comparisons, no `%`, grid<=core count).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # 1/ln(2); folds exp(x) into exp2(x*LOG2E)


@triton.jit
def _swa_sink_fwd_kernel(
    Q, K, V, Sink, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, L, W,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    # One program = one (batch*head, query-block).
    pid_m = tl.program_id(0)       # which query block
    pid_bh = tl.program_id(1)      # which (batch, head)
    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)          # query rows [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                    # head dims  [BLOCK_D]
    d_mask = offs_d < D

    # ---- load this query block: q_block [BLOCK_M, BLOCK_D] ----
    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < L) & d_mask[None, :]
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---- seed the online softmax WITH THE SINK as a virtual key (see module docstring) ----
    sink_val = tl.load(Sink + h).to(tl.float32)      # per-head fp32 scalar, RAW (unscaled)
    m_i = tl.full([BLOCK_M], sink_val * LOG2E, dtype=tl.float32)   # running max incl. sink
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)               # denom incl. sink term
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)          # numerator (sink absent)

    qk_scale = scale * LOG2E     # fold scale AND log2(e) into the scores once

    # ---- key-block range: only blocks overlapping the query block's window union ----
    # union of [i-W, i+W] over i in [m_start, m_start+BLOCK_M) = [m_start-W, m_start+BLOCK_M-1+W]
    lo = m_start - W
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N                    # align down to a key-block boundary
    hi = tl.minimum(m_start + BLOCK_M + W, L)         # exclusive upper bound

    for n_start in range(lo, hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)      # key rows [BLOCK_N]

        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k_mask = (offs_n[:, None] < L) & d_mask[None, :]
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        # scores = q . k^T, fp32 accumulate; masked d-lanes are 0*0 so the dot is exact over D
        qk = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * qk_scale   # [BLOCK_M, BLOCK_N]

        # bidirectional window predicate, on the fly (comparisons in fp32 to vectorize on NPU)
        keep = ((offs_n[None, :] >= offs_m[:, None] - W)
                & (offs_n[None, :] <= offs_m[:, None] + W)
                & (offs_n[None, :] < L))
        qk = tl.where(keep, qk, float("-inf"))

        # online softmax update (log2 domain)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])          # [BLOCK_M, BLOCK_N], fp32
        alpha = tl.math.exp2(m_i - m_ij)              # rescale factor for prior stats
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (V + b * stride_vb + h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v_mask = (offs_n[:, None] < L) & d_mask[None, :]
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        # P@V: cast P back to v.dtype per the contract; accumulate in fp32
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        m_i = m_ij

    o = acc / l_i[:, None]                            # sink is in l_i, absent from acc
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    o_mask = (offs_m[:, None] < L) & d_mask[None, :]
    tl.store(o_ptrs, o.to(Out.dtype.element_ty), mask=o_mask)


def swa_noncausal_sink_attn_fwd(
    q: torch.Tensor,        # [B, H, L, D]
    k: torch.Tensor,        # [B, H, L, D]  (MHA)
    v: torch.Tensor,        # [B, H, L, D]
    sink: torch.Tensor,     # [H]  per-head fp32 logit
    window: int,            # bidirectional half-width W (>0)
    scale: float | None = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Forward-only Triton SWA-non-causal-sink attention (MHA). Returns o [B, H, L, D].

    No autograd — this validates the forward math against eager_reference.py. The
    window is bidirectional [i-W, i+W]; pass window>0 (full attention is out of scope
    for the first kernel). See module docstring for the numerics.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "kernel needs a CUDA (or Triton-capable) device"
    assert q.dim() == 4 and k.shape == q.shape and v.shape == q.shape, "MHA [B,H,L,D] expected"
    assert window is not None and window > 0, "first kernel targets a finite bidirectional window"
    B, H, L, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(L, BLOCK_M), B * H)
    _swa_sink_fwd_kernel[grid](
        q, k, v, sink, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, L, window,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )
    return o
