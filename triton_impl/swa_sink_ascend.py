#!/usr/bin/env python3
"""Ascend-NPU-shaped FORWARD kernel for SWA-non-causal-sink attention.

Same math as swa_sink_fwd.py, but reshaped for the Triton-Ascend backend. It STILL runs and
validates on CUDA (identical results) so correctness can be nailed on the GPU before moving to
the A3 — only backend lowering remains NPU-specific. See /workspace/skills/triton-ascend
(latency-optimizer/references/checklist.md, kernel-generator/references/hw-ascend910-9362.md).

What changed vs the CUDA kernel (Ascend constraints):
  * 1-D GRID (Ascend forbids multi-dim grids). The (query-block, batch*head) tile index is
    flattened to a single program_id and decoded inside the kernel.
  * No `%` operator (Ascend forbids `a % b`): the tile/head decode uses only `//` and `-`
    (a - (a//b)*b), per the checklist.
  * int32 index math; no continue/break; fp32 softmax accumulation (unchanged).

Still TODO for the NPU (next steps, not needed for GPU correctness):
  * grid <= core count via a grid-stride loop (grid = num_aicore; each program strides over
    tiles) — here grid = NUM_TILES (one tile per program), which is 1-D and < 65536 but not yet
    core-capped.
  * cast the window comparisons to fp32 to vectorize on the NPU; confirm tl.dot / tl.trans /
    exp2 / dynamic range lower on Triton-Ascend; drop input_precision if unsupported.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _swa_sink_fwd_ascend_kernel(
    Q, K, V, Sink, Out, Lse,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    NUM_M_BLOCKS,                                   # = cdiv(LQ, BLOCK_M); for the 1-D decode
    WINDOWED: tl.constexpr, FP32_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    # ---- 1-D grid: decode program_id -> (b, h, query-block) with NO `%` (Ascend rule) ----
    pid = tl.program_id(0)                          # in [0, NUM_M_BLOCKS * B * H)
    bh = pid // NUM_M_BLOCKS
    m_block = pid - bh * NUM_M_BLOCKS               # == pid % NUM_M_BLOCKS, without `%`
    b = bh // H
    h = bh - b * H                                  # == bh % H, without `%`

    m_start = m_block * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m_valid = offs_m < LQ

    q_block = tl.load(Q + b * stride_qb + h * stride_qh
                      + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                      mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)

    sink_val = tl.load(Sink + h).to(tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink_val * LOG2E   # seed sink as a virtual key
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = scale * LOG2E

    if WINDOWED:
        lo = tl.maximum(m_start - WIN_LEFT, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = tl.minimum(m_start + BLOCK_M + WIN_RIGHT, LK)
    else:
        lo = 0
        hi = LK

    for n_start in range(lo, hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < LK
        k_block = tl.load(K + b * stride_kb + h * stride_kh
                          + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32)
        if not FP32_QK:
            qk = qk.to(q_block.dtype).to(tl.float32)
        qk = qk * qk_scale
        if WINDOWED:
            keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                    & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT) & (offs_n[None, :] < LK))
        else:
            keep = offs_n[None, :] < LK
        qk = tl.where(keep, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_block = tl.load(V + b * stride_vb + h * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block, input_precision="ieee")
        m_i = m_ij

    o = acc / l_i[:, None]
    tl.store(Out + b * stride_ob + h * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             o.to(Out.dtype.element_ty), mask=(m_valid[:, None] & d_mask[None, :]))
    lse = m_i + tl.log2(l_i)
    tl.store(Lse + (b * H + h) * LQ + offs_m, lse, mask=m_valid)


def _kv_strides(t):
    if t.dim() == 4:
        return t.stride(0), t.stride(1), t.stride(2), t.stride(3)
    return t.stride(0), 0, t.stride(1), t.stride(2)      # MLA-shared [B,Lk,D] -> head-stride 0


def swa_sink_attn_fwd_ascend(q, k, v, sink, win_left, win_right, scale=None,
                             dense=False, BLOCK_M=32, BLOCK_N=32, fp32_qk=True):
    """Ascend-shaped forward (1-D grid). Returns (o, lse). q [B,H,LQ,D]; k,v [B,H,LK,D] (MHA)
    or [B,LK,D] (MLA). dense=True -> no window (Lq!=Lk allowed). Runs on CUDA for validation."""
    assert q.dim() == 4, "q must be [B,H,LQ,D]"
    B, H, LQ, D = q.shape
    LK = k.shape[1] if k.dim() == 3 else k.shape[2]
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(B, H, LQ, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    num_m_blocks = triton.cdiv(LQ, BLOCK_M)
    skb, skh, skn, skd = _kv_strides(k)
    svb, svh, svn, svd = _kv_strides(v)
    grid = (num_m_blocks * B * H,)                       # 1-D grid
    _swa_sink_fwd_ascend_kernel[grid](
        q, k, v, sink, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        skb, skh, skn, skd, svb, svh, svn, svd,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, LQ, LK, win_left, win_right, scale,
        num_m_blocks,
        WINDOWED=not dense, FP32_QK=fp32_qk,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )
    return o, lse
