#!/usr/bin/env python3
"""Triton BACKWARD kernels for SWA-non-causal-sink attention (fused dq / dk / dv; dsink in torch).

Standard flash-attention backward + the per-head sink term. Given the forward's o and lse
(lse = m + log2(l), incl. the sink), the backward recomputes p on the fly:

  p_ij   = exp2(qk2_ij - lse_i)                    # qk2 = q.k * scale * log2e; masked -> 0
  D_i    = do_i . o_i                               # the "delta" (torch preprocess)
  dp_ij  = do_i . v_j
  ds_ij  = p_ij * (dp_ij - D_i)                     # softmax backward
  dq_i   = scale * sum_j ds_ij k_j                  # _bwd_dq_kernel  (query-parallel)
  dk_j   = scale * sum_i ds_ij q_i                  # _bwd_dkdv_kernel (key-parallel)
  dv_j   = sum_i p_ij do_i                          # _bwd_dkdv_kernel
  dsink_h= -sum_i exp2(sink_h*log2e - lse_i) * D_i  # torch reduction over lse + delta

Windows: dq visits key-blocks in [m-win_left, m+win_right]; dk/dv visits query-blocks in
[n-win_right, n+win_left] (win_left/win_right swap for the transposed pass). MLA-shared K/V:
dq reads the shared latent via head-stride 0 (per-head, unchanged); dk/dv use a dedicated
kernel (one program per key-block+batch) that loops ALL heads and accumulates in registers ->
[B,LK,D] directly (K/V loaded once, reused across heads; no [B,H,LK,D] buffer, no atomics).
Validated on CPU vs autograd to ~1e-15 (block algorithm + ranges, MHA and MLA).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)

# Autotune search space for the backward kernels (the bottleneck; ~3x the forward FLOPs +
# redundant recompute). BLOCK_M/BLOCK_N + num_warps/num_stages are tuned per (LQ,LK,D) so the
# large-D=512 register pressure is resolved on-device. NOTE: num_warps/num_stages are CUDA-only
# levers — the Ascend-Triton autotune ignores them (drop for the NPU port; see skills autotune.md).
_BWD_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_BWD_CONFIGS, key=["LQ", "LK", "D"])
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, Lse, Delta, DQ,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides
    stride_gb, stride_gh, stride_gm, stride_gd,      # DQ strides
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m_valid = offs_m < LQ

    q_block = tl.load(Q + b * stride_qb + h * stride_qh
                      + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                      mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
    do_block = tl.load(DO + b * stride_ob + h * stride_oh
                       + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                       mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
    lse_block = tl.load(Lse + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)
    delta_block = tl.load(Delta + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
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
        v_block = tl.load(V + b * stride_vb + h * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32) * qk_scale
        if WINDOWED:
            keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                    & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT) & (offs_n[None, :] < LK))
        else:
            keep = offs_n[None, :] < LK
        qk = tl.where(keep, qk, float("-inf"))
        p = tl.math.exp2(qk - lse_block[:, None])                     # masked -> 0
        dp = tl.dot(do_block, tl.trans(v_block), input_precision="ieee").to(tl.float32)
        ds = p * (dp - delta_block[:, None])
        dq_acc += tl.dot(ds.to(k_block.dtype), k_block, input_precision="ieee")

    dq_acc = dq_acc * scale
    tl.store(DQ + b * stride_gb + h * stride_gh
             + offs_m[:, None] * stride_gm + offs_d[None, :] * stride_gd,
             dq_acc.to(DQ.dtype.element_ty), mask=(m_valid[:, None] & d_mask[None, :]))


@triton.autotune(configs=_BWD_CONFIGS, key=["LQ", "LK", "D"])
@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, DO, Lse, Delta, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides
    stride_kgb, stride_kgh, stride_kgn, stride_kgd,  # DK strides ([B,H,LK,D])
    stride_vgb, stride_vgh, stride_vgn, stride_vgd,  # DV strides
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    n_valid = offs_n < LK

    k_block = tl.load(K + b * stride_kb + h * stride_kh
                      + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                      mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
    v_block = tl.load(V + b * stride_vb + h * stride_vh
                      + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                      mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    qk_scale = scale * LOG2E

    if WINDOWED:
        lo = tl.maximum(n_start - WIN_RIGHT, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
        hi = tl.minimum(n_start + BLOCK_N + WIN_LEFT, LQ)
    else:
        lo = 0
        hi = LQ

    for m_start in range(lo, hi, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_valid = offs_m < LQ
        q_block = tl.load(Q + b * stride_qb + h * stride_qh
                          + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                          mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
        do_block = tl.load(DO + b * stride_ob + h * stride_oh
                           + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                           mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
        lse_block = tl.load(Lse + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)
        delta_block = tl.load(Delta + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)

        qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32) * qk_scale
        if WINDOWED:
            keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                    & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT)
                    & (offs_m[:, None] < LQ) & (offs_n[None, :] < LK))
        else:
            keep = (offs_m[:, None] < LQ) & (offs_n[None, :] < LK)
        qk = tl.where(keep, qk, float("-inf"))
        p = tl.math.exp2(qk - lse_block[:, None])                     # [BM,BN], masked -> 0
        dp = tl.dot(do_block, tl.trans(v_block), input_precision="ieee").to(tl.float32)
        ds = p * (dp - delta_block[:, None])
        dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block, input_precision="ieee")
        dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block, input_precision="ieee")

    dk_acc = dk_acc * scale
    tl.store(DK + b * stride_kgb + h * stride_kgh
             + offs_n[:, None] * stride_kgn + offs_d[None, :] * stride_kgd,
             dk_acc.to(DK.dtype.element_ty), mask=(n_valid[:, None] & d_mask[None, :]))
    tl.store(DV + b * stride_vgb + h * stride_vgh
             + offs_n[:, None] * stride_vgn + offs_d[None, :] * stride_vgd,
             dv_acc.to(DV.dtype.element_ty), mask=(n_valid[:, None] & d_mask[None, :]))


@triton.autotune(configs=_BWD_CONFIGS, key=["LQ", "LK", "D"])
@triton.jit
def _bwd_dkdv_mla_kernel(
    Q, K, V, DO, Lse, Delta, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,                 # K/V/DK/DV are [B, LK, D] (shared latent)
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides [B,H,LQ,D]
    stride_kgb, stride_kgn, stride_kgd,              # DK strides [B,LK,D]
    stride_vgb, stride_vgn, stride_vgd,              # DV strides
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    """MLA-shared dk/dv: one program per (key-block, batch); loop ALL H heads and accumulate
    dk/dv in registers -> writes [B,LK,D] directly (no per-head buffer, no atomics). K/V loaded
    once and reused across heads."""
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    n_valid = offs_n < LK

    k_block = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                      mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
    v_block = tl.load(V + b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                      mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    qk_scale = scale * LOG2E

    if WINDOWED:
        lo = tl.maximum(n_start - WIN_RIGHT, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
        hi = tl.minimum(n_start + BLOCK_N + WIN_LEFT, LQ)
    else:
        lo = 0
        hi = LQ

    for h in range(0, H):
        for m_start in range(lo, hi, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            m_valid = offs_m < LQ
            q_block = tl.load(Q + b * stride_qb + h * stride_qh
                              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                              mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
            do_block = tl.load(DO + b * stride_ob + h * stride_oh
                               + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                               mask=(m_valid[:, None] & d_mask[None, :]), other=0.0)
            lse_block = tl.load(Lse + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)
            delta_block = tl.load(Delta + (b * H + h) * LQ + offs_m, mask=m_valid, other=0.0)

            qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32) * qk_scale
            if WINDOWED:
                keep = ((offs_n[None, :] >= offs_m[:, None] - WIN_LEFT)
                        & (offs_n[None, :] <= offs_m[:, None] + WIN_RIGHT)
                        & (offs_m[:, None] < LQ) & (offs_n[None, :] < LK))
            else:
                keep = (offs_m[:, None] < LQ) & (offs_n[None, :] < LK)
            qk = tl.where(keep, qk, float("-inf"))
            p = tl.math.exp2(qk - lse_block[:, None])
            dp = tl.dot(do_block, tl.trans(v_block), input_precision="ieee").to(tl.float32)
            ds = p * (dp - delta_block[:, None])
            dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block, input_precision="ieee")
            dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block, input_precision="ieee")

    dk_acc = dk_acc * scale
    tl.store(DK + b * stride_kgb + offs_n[:, None] * stride_kgn + offs_d[None, :] * stride_kgd,
             dk_acc.to(DK.dtype.element_ty), mask=(n_valid[:, None] & d_mask[None, :]))
    tl.store(DV + b * stride_vgb + offs_n[:, None] * stride_vgn + offs_d[None, :] * stride_vgd,
             dv_acc.to(DV.dtype.element_ty), mask=(n_valid[:, None] & d_mask[None, :]))


def _kv_strides(t):
    if t.dim() == 4:
        return t.stride(0), t.stride(1), t.stride(2), t.stride(3)
    return t.stride(0), 0, t.stride(1), t.stride(2)      # MLA-shared [B,Lk,D] -> head-stride 0


def swa_sink_bwd(q, k, v, sink, o, lse, do, win_left, win_right, dense, scale,
                 BLOCK_M=32, BLOCK_N=32):
    """Fused Triton backward. Returns (dq, dk, dv, dsink) in the inputs' dtypes.
    q,o,do [B,H,LQ,D]; k,v [B,H,LK,D] (MHA) or [B,LK,D] (MLA-shared); lse [B,H,LQ] fp32."""
    B, H, LQ, D = q.shape
    mla = (k.dim() == 3)
    LK = k.shape[1] if mla else k.shape[2]
    windowed = not dense
    do = do.contiguous()
    lse = lse.contiguous()
    delta = (do.float() * o.float()).sum(-1).contiguous()             # [B,H,LQ] fp32
    BLOCK_D = triton.next_power_of_2(D)
    skb, skh, skn, skd = _kv_strides(k)
    svb, svh, svn, svd = _kv_strides(v)
    # BLOCK_M/BLOCK_N + num_warps/num_stages come from @triton.autotune; the grid uses them.

    # dq (query-parallel; reads shared K/V via head-stride 0 for MLA)
    dq = torch.empty_like(q)
    _bwd_dq_kernel[lambda meta: (triton.cdiv(LQ, meta["BLOCK_M"]), B * H)](
        q, k, v, do, lse, delta, dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        skb, skh, skn, skd, svb, svh, svn, svd,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        H, LQ, LK, win_left, win_right, scale,
        WINDOWED=windowed, BLOCK_D=BLOCK_D, D=D,
    )

    # dk/dv (key-parallel). MLA: one program per (key-block, batch) loops all heads and
    # accumulates in registers -> [B,LK,D] directly (no per-head buffer). MHA: per (b,h).
    # Output buffers are the INPUT dtype (not fp32): each key-block is written exactly once
    # (accumulation is fp32 in-kernel), so no fp32 output buffer is needed.
    if mla:
        dk_f = torch.empty(B, LK, D, device=q.device, dtype=k.dtype)
        dv_f = torch.empty(B, LK, D, device=q.device, dtype=v.dtype)
        _bwd_dkdv_mla_kernel[lambda meta: (triton.cdiv(LK, meta["BLOCK_N"]), B)](
            q, k, v, do, lse, delta, dk_f, dv_f,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_f.stride(0), dk_f.stride(1), dk_f.stride(2),
            dv_f.stride(0), dv_f.stride(1), dv_f.stride(2),
            H, LQ, LK, win_left, win_right, scale,
            WINDOWED=windowed, BLOCK_D=BLOCK_D, D=D,
        )
    else:
        dk_f = torch.empty(B, H, LK, D, device=q.device, dtype=k.dtype)
        dv_f = torch.empty(B, H, LK, D, device=q.device, dtype=v.dtype)
        _bwd_dkdv_kernel[lambda meta: (triton.cdiv(LK, meta["BLOCK_N"]), B * H)](
            q, k, v, do, lse, delta, dk_f, dv_f,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            skb, skh, skn, skd, svb, svh, svn, svd,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_f.stride(0), dk_f.stride(1), dk_f.stride(2), dk_f.stride(3),
            dv_f.stride(0), dv_f.stride(1), dv_f.stride(2), dv_f.stride(3),
            H, LQ, LK, win_left, win_right, scale,
            WINDOWED=windowed, BLOCK_D=BLOCK_D, D=D,
        )
    dk = dk_f            # already in k/v dtype
    dv = dv_f

    # dsink_h = -sum_i exp2(sink_h*log2e - lse_i) * delta_i   (over batch + queries)
    LOG2E_F = 1.4426950408889634
    dsink = -(torch.exp2(sink.float().view(1, H, 1) * LOG2E_F - lse) * delta).sum(dim=(0, 2))
    return dq, dk, dv, dsink.to(sink.dtype)
