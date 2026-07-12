#!/usr/bin/env python3
"""Triton FORWARD kernel for SWA-non-causal-sink attention (MHA layout).

Forward-only fused operator specified in ../README.md and ../eager_reference.py. This
matches the PACKED-SWA view the efficient kernel implements: an ASYMMETRIC sliding window
+ per-head sink over a packed sequence. FORWARD ONLY (no autograd yet) — validated no-grad
against the fp32 eager reference; the backward pass is a separate step.

Math it reproduces (README §2, the exact numerics contract):
  scores[i,j] = (q_i . k_j) * scale                            # fp32 accumulate
  scores[i,j] = -inf   if not (i-win_left <= j <= i+win_right) # ASYMMETRIC SWA, on the fly
  logits      = [ scores_i | sink[h] ]                         # sink: RAW fp32, unscaled, unmasked
  p           = softmax(logits)[:-1]                            # sink in the normaliser, then dropped
  o_i         = sum_j p[j] * v_j                                # P cast to v.dtype for P@V

The window is ASYMMETRIC (this is the real DSpark form, not the naive symmetric one):
  win_left  = window + block_size - 1        # e.g. 128 + 7 - 1 = 134
  win_right = block_size - 1                 # e.g. 7 - 1 = 6
Use eager_reference.dspark_sas_window(block_size, window) to get (win_left, win_right).
A symmetric convenience wrapper (win_left == win_right) is kept for the first-step microbench.

Key design points
-----------------
* SINK as a seeded virtual key. The per-head sink is a single extra softmax column that is
  constant across all keys and never enters the value sum. We inject it by SEEDING the
  online-softmax running stats as if the sink were the very first key:
      m = sink * log2(e)      (running max, log2 domain)
      l = 1.0                 (= exp2(sink - sink); the sink's term in the denominator)
      acc = 0                 (the sink contributes NOTHING to the numerator)
  Real keys then stream in with the standard online update, so the sink stays in the
  denominator and out of P@V for free — no extra column is ever materialized. As
  sink -> -inf its alpha collapses to 0 on the first real key => plain windowed softmax.

* Asymmetric window predicate, tested on the fly (README §4) — never a dense [L,L] mask.
  The inner loop only visits key-blocks overlapping [m_start-win_left, m_end-1+win_right].

* fp32 softmax accumulation regardless of input dtype (README criterion #1). QK^T and P@V
  use input_precision="ieee" so fp32 inputs get TRUE fp32 (CUDA tl.dot defaults to TF32,
  which caps accuracy at ~1e-3); bf16 inputs are unaffected (already bf16*bf16 -> fp32 acc).
  P is cast back to v.dtype before P@V (matching the model). exp2 in the log2 domain.

Backend note: written for CUDA Triton first (the hardware the integrator validates on today).
The Ascend-NPU port is a distinct backend — see /workspace/skills/triton-ascend for its extra
constraints (1-D grid, fp32 comparisons, no `%`, grid<=core count).
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
    H, L, WIN_LEFT, WIN_RIGHT,
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
    # union of [i-win_left, i+win_right] over i in [m_start, m_start+BLOCK_M) is
    #   [m_start - win_left, m_start + BLOCK_M - 1 + win_right]
    lo = m_start - WIN_LEFT
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N                    # align down to a key-block boundary
    hi = tl.minimum(m_start + BLOCK_M + WIN_RIGHT, L)  # exclusive upper bound

    for n_start in range(lo, hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)      # key rows [BLOCK_N]

        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k_mask = (offs_n[:, None] < L) & d_mask[None, :]
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        # scores = q . k^T, fp32 accumulate; masked d-lanes are 0*0 so the dot is exact over D.
        # input_precision="ieee": true fp32 on fp32 inputs (ignored for bf16). See docstring.
        qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32) * qk_scale

        # asymmetric window predicate, on the fly (comparisons in fp32 to vectorize on NPU)
        keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT)
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

        # P@V: cast P back to v.dtype per the contract; accumulate in fp32 (ieee: true fp32 path)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block, input_precision="ieee")
        m_i = m_ij

    o = acc / l_i[:, None]                            # sink is in l_i, absent from acc
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    o_mask = (offs_m[:, None] < L) & d_mask[None, :]
    tl.store(o_ptrs, o.to(Out.dtype.element_ty), mask=o_mask)


def swa_sink_attn_fwd(
    q: torch.Tensor,        # [B, H, L, D]
    k: torch.Tensor,        # [B, H, L, D]  (MHA)
    v: torch.Tensor,        # [B, H, L, D]
    sink: torch.Tensor,     # [H]  per-head fp32 logit
    win_left: int,          # asymmetric window: attend to [i-win_left, i+win_right]
    win_right: int,
    scale: float | None = None,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
) -> torch.Tensor:
    """Forward-only Triton ASYMMETRIC sliding-window + sink attention (MHA). o [B,H,L,D].

    win_left/win_right define the window keep[i,j] = (j>=i-win_left)&(j<=i+win_right); get
    the real DSpark values from eager_reference.dspark_sas_window(block_size, window). No
    autograd — validates the forward math against eager_reference.swa_sink_attention.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "kernel needs a CUDA (or Triton-capable) device"
    assert q.dim() == 4 and k.shape == q.shape and v.shape == q.shape, "MHA [B,H,L,D] expected"
    assert win_left >= 0 and win_right >= 0, "window half-widths must be >= 0"
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
        H, L, win_left, win_right,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )
    return o


def swa_noncausal_sink_attn_fwd(q, k, v, sink, window, scale=None, BLOCK_M=32, BLOCK_N=32):
    """[COMPAT] SYMMETRIC window (win_left == win_right == window) — the first-step microbench
    form. Thin wrapper over swa_sink_attn_fwd; the REAL model uses the asymmetric window."""
    assert window is not None and window > 0, "symmetric wrapper needs a finite window > 0"
    return swa_sink_attn_fwd(q, k, v, sink, win_left=window, win_right=window,
                             scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
