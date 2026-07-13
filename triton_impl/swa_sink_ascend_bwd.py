#!/usr/bin/env python3
"""Ascend-NPU-shaped BACKWARD kernels for SWA-non-causal-sink attention.

Same math as swa_sink_bwd.py (dq / dk / dv fused kernels + torch dsink), but reshaped for the
Triton-Ascend backend, and still GPU-testable. The only structural change vs the CUDA backward
is the GRID (identical to the Ascend forward, swa_sink_ascend.py):
  * 1-D grid, core-capped via a grid-stride loop: grid = min(NUM_TILES, num_cores); each program
    strides `for tile in range(pid, NUM_TILES, tl.num_programs(0))` over its tiles.
  * tile -> (b, h, block) decoded with NO `%` (a % b == a - (a//b)*b).
  * window comparisons cast to fp32 (Ascend vectorizes fp32 compares).
CPU-validated math is unchanged (~1e-15 vs autograd). input_precision kept for CUDA fp32
accuracy (revisit on the A3). MLA dk/dv loops all heads per key-block, accumulating in registers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .swa_sink_ascend import _kv_strides, _num_cores, _default_blocks, swa_sink_attn_fwd_ascend

LOG2E = tl.constexpr(1.4426950408889634)


def _bwd_safe_blocks(D, bm, bn):
    """The backward loads BOTH K[BN,D] and V[BN,D] per n-iter, and Ascend's auto multi-buffer
    ~doubles that on-chip; large BN at big D overflows the 512KB L1 ("cbuf overflow"). The FORWARD
    wants a big BN for speed (KV in one block) but the backward must stay small, so cap BN/BM here
    for large head_dim (fwd and bwd use independent tiles). See memory ascend-bwd-cbuf-limit."""
    if D >= 512:
        return min(bm, 16), min(bn, 32)
    if D >= 256:
        return min(bm, 32), min(bn, 64)
    return bm, bn


@triton.jit
def _bwd_dq_ascend_kernel(
    Q, K, V, DO, Lse, Delta, DQ,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides
    stride_gb, stride_gh, stride_gm, stride_gd,      # DQ strides
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    NUM_TILES, NUM_M_BLOCKS,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):
        bh = tile // NUM_M_BLOCKS
        m_block = tile - bh * NUM_M_BLOCKS
        b = bh // H
        h = bh - b * H
        m_start = m_block * BLOCK_M
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
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

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
            mf = offs_m[:, None].to(tl.float32)
            nf = offs_n[None, :].to(tl.float32)
            if WINDOWED:
                keep = (nf >= mf - WIN_LEFT) & (nf <= mf + WIN_RIGHT) & (nf < LK)
            else:
                keep = nf < LK
            qk = tl.where(keep, qk, float("-inf"))
            p = tl.math.exp2(qk - lse_block[:, None])
            dp = tl.dot(do_block, tl.trans(v_block), input_precision="ieee").to(tl.float32)
            ds = p * (dp - delta_block[:, None])
            dq_acc += tl.dot(ds.to(k_block.dtype), k_block, input_precision="ieee")

        dq_acc = dq_acc * scale
        tl.store(DQ + b * stride_gb + h * stride_gh
                 + offs_m[:, None] * stride_gm + offs_d[None, :] * stride_gd,
                 dq_acc.to(DQ.dtype.element_ty), mask=(m_valid[:, None] & d_mask[None, :]))


@triton.jit
def _bwd_dkdv_ascend_kernel(
    Q, K, V, DO, Lse, Delta, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides
    stride_kgb, stride_kgh, stride_kgn, stride_kgd,  # DK strides [B,H,LK,D]
    stride_vgb, stride_vgh, stride_vgn, stride_vgd,  # DV strides
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    NUM_TILES, NUM_N_BLOCKS,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):
        bh = tile // NUM_N_BLOCKS
        n_block = tile - bh * NUM_N_BLOCKS
        b = bh // H
        h = bh - b * H
        n_start = n_block * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < LK

        k_block = tl.load(K + b * stride_kb + h * stride_kh
                          + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        v_block = tl.load(V + b * stride_vb + h * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

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
            mf = offs_m[:, None].to(tl.float32)
            nf = offs_n[None, :].to(tl.float32)
            if WINDOWED:
                keep = (nf >= mf - WIN_LEFT) & (nf <= mf + WIN_RIGHT) & (mf < LQ) & (nf < LK)
            else:
                keep = (mf < LQ) & (nf < LK)
            qk = tl.where(keep, qk, float("-inf"))
            p = tl.math.exp2(qk - lse_block[:, None])
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


@triton.jit
def _bwd_dkdv_mla_ascend_kernel(
    Q, K, V, DO, Lse, Delta, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,                 # K/V/DK/DV [B,LK,D] shared latent
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,      # DO strides
    stride_kgb, stride_kgn, stride_kgd,
    stride_vgb, stride_vgn, stride_vgd,
    H, LQ, LK, WIN_LEFT, WIN_RIGHT, scale,
    NUM_TILES, NUM_N_BLOCKS,
    WINDOWED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):        # tile = (batch, key-block); loop heads inside
        b = tile // NUM_N_BLOCKS
        n_block = tile - b * NUM_N_BLOCKS
        n_start = n_block * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < LK

        k_block = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        v_block = tl.load(V + b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                          mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

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
                mf = offs_m[:, None].to(tl.float32)
                nf = offs_n[None, :].to(tl.float32)
                if WINDOWED:
                    keep = (nf >= mf - WIN_LEFT) & (nf <= mf + WIN_RIGHT) & (mf < LQ) & (nf < LK)
                else:
                    keep = (mf < LQ) & (nf < LK)
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


# ---------------------------------------------------------------------------
# D-TILED / ROW-TILED MLA-dense backward (mirrors the D-tiled forward; the DEFAULT dense-MLA path).
# Both kernels had the M=7 starvation of one head's BS queries. Here rows = flattened (head,query)
# pairs on the matmul M axis (BLOCK_M lever). dq D-tiles the D=512 head dim so nothing [*,512] is
# on chip (BLOCK_K); dk/dv tiles the KEY dim (small BLOCK_KV keeps dk_acc/dv_acc[BLOCK_KV,D] + Q/DO
# in the 192KB UB — same footprint as the validated head-batched version) and sums over row-tiles.
# Same math as the validated per-(b,h) kernels (per-row lse/delta). 1-D grid-stride, no `%`.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_dq_mla_dense_ascend_kernel(
    Q, K, V, DO, Lse, Delta, DQ,
    stride_qn, stride_qh, stride_qm, stride_qd,      # Q [N, H, BS, D]; flat row stride = stride_qm
    stride_kn, stride_kk, stride_kd,                 # K/V [N, KV, D] latent
    stride_vn, stride_vk, stride_vd,
    stride_on, stride_oh, stride_om, stride_od,      # DO strides
    stride_gn, stride_gh, stride_gm, stride_gd,      # DQ strides
    H, HR, KV, scale,                                # HR = H*BS rows per block
    NUM_TILES, NUM_M_TILES,
    BS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)                   # BLOCK_N == KV (all keys valid)
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):
        n = tile // NUM_M_TILES
        mt = tile - n * NUM_M_TILES
        m0 = mt * BLOCK_M
        rows = m0 + offs_m
        row_ok = rows < HR
        base_q = Q + n * stride_qn + m0 * stride_qm
        base_do = DO + n * stride_on + m0 * stride_om
        base_k = K + n * stride_kn
        base_v = V + n * stride_vn

        # phase 1: qk[M,KV] and dp[M,KV] via D-tiling (nothing [*,512] on chip)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_K):
            offs_k = d0 + tl.arange(0, BLOCK_K)
            kd = offs_k[None, :] < D
            q = tl.load(base_q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd,
                        mask=(row_ok[:, None] & kd), other=0.0)
            kk = tl.load(base_k + offs_n[:, None] * stride_kk + offs_k[None, :] * stride_kd,
                         mask=kd, other=0.0)
            qk += tl.dot(q, tl.trans(kk), input_precision="ieee").to(tl.float32)
            do = tl.load(base_do + offs_m[:, None] * stride_om + offs_k[None, :] * stride_od,
                         mask=(row_ok[:, None] & kd), other=0.0)
            vv = tl.load(base_v + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vd,
                         mask=kd, other=0.0)
            dp += tl.dot(do, tl.trans(vv), input_precision="ieee").to(tl.float32)
        qk = qk * qk_scale
        lse_block = tl.load(Lse + n * HR + rows, mask=row_ok, other=0.0)
        delta_block = tl.load(Delta + n * HR + rows, mask=row_ok, other=0.0)
        p = tl.math.exp2(qk - lse_block[:, None])
        ds = (p * (dp - delta_block[:, None])).to(K.dtype.element_ty)     # [M,KV] bf16

        # phase 2: dq[M,D] = (ds @ K) * scale, D-tiled output
        base_dq = DQ + n * stride_gn + m0 * stride_gm
        for d0 in range(0, D, BLOCK_K):
            offs_k = d0 + tl.arange(0, BLOCK_K)
            kd = offs_k[None, :] < D
            kk = tl.load(base_k + offs_n[:, None] * stride_kk + offs_k[None, :] * stride_kd,
                         mask=kd, other=0.0)
            dq_c = tl.dot(ds, kk, input_precision="ieee").to(tl.float32) * scale
            tl.store(base_dq + offs_m[:, None] * stride_gm + offs_k[None, :] * stride_gd,
                     dq_c.to(DQ.dtype.element_ty), mask=(row_ok[:, None] & kd))


@triton.jit
def _bwd_dkdv_mla_dense_ascend_kernel(
    Q, K, V, DO, Lse, Delta, DK, DV,
    stride_qn, stride_qh, stride_qm, stride_qd,      # Q [N, H, BS, D]
    stride_kn, stride_kk, stride_kd,                 # K/V/DK/DV [N, KV, D] latent
    stride_vn, stride_vk, stride_vd,
    stride_on, stride_oh, stride_om, stride_od,      # DO strides
    stride_kgn, stride_kgk, stride_kgd,
    stride_vgn, stride_vgk, stride_vgd,
    H, HR, KV, scale,
    NUM_TILES, NUM_KV_TILES, NUM_M_TILES,
    BS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_m = tl.arange(0, BLOCK_M)
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):     # tile = (block n, key-tile); sum over all rows
        n = tile // NUM_KV_TILES
        kt = tile - n * NUM_KV_TILES
        kv0 = kt * BLOCK_KV
        keys = kv0 + offs_kv
        kv_ok = keys < KV

        k_full = tl.load(K + n * stride_kn + keys[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                         mask=(kv_ok[:, None] & d_mask[None, :]), other=0.0)   # [BLOCK_KV, D]
        v_full = tl.load(V + n * stride_vn + keys[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                         mask=(kv_ok[:, None] & d_mask[None, :]), other=0.0)
        dk_acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

        for mt in range(0, NUM_M_TILES):             # row-tiles: BLOCK_M flattened (head,query) rows
            m0 = mt * BLOCK_M
            rows = m0 + offs_m
            row_ok = rows < HR
            base_q = Q + n * stride_qn + m0 * stride_qm
            base_do = DO + n * stride_on + m0 * stride_om
            q_full = tl.load(base_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                             mask=(row_ok[:, None] & d_mask[None, :]), other=0.0)   # [M, D]
            do_full = tl.load(base_do + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                              mask=(row_ok[:, None] & d_mask[None, :]), other=0.0)
            lse_block = tl.load(Lse + n * HR + rows, mask=row_ok, other=0.0)
            delta_block = tl.load(Delta + n * HR + rows, mask=row_ok, other=0.0)

            qk = tl.dot(q_full, tl.trans(k_full), input_precision="ieee").to(tl.float32) * qk_scale
            keep = row_ok[:, None] & kv_ok[None, :]
            qk = tl.where(keep, qk, float("-inf"))
            p = tl.math.exp2(qk - lse_block[:, None])                     # [M, BLOCK_KV]
            dp = tl.dot(do_full, tl.trans(v_full), input_precision="ieee").to(tl.float32)
            ds = p * (dp - delta_block[:, None])
            dv_acc += tl.dot(tl.trans(p.to(do_full.dtype)), do_full, input_precision="ieee")
            dk_acc += tl.dot(tl.trans(ds.to(q_full.dtype)), q_full, input_precision="ieee")

        dk_acc = dk_acc * scale
        tl.store(DK + n * stride_kgn + keys[:, None] * stride_kgk + offs_d[None, :] * stride_kgd,
                 dk_acc.to(DK.dtype.element_ty), mask=(kv_ok[:, None] & d_mask[None, :]))
        tl.store(DV + n * stride_vgn + keys[:, None] * stride_vgk + offs_d[None, :] * stride_vgd,
                 dv_acc.to(DV.dtype.element_ty), mask=(kv_ok[:, None] & d_mask[None, :]))


def _swa_sink_bwd_mla_dense(q, k, v, sink, o, lse, do, scale, num_programs=None,
                            BM_DQ=32, BK_DQ=64, BM_DKDV=16, BLOCK_KV=16):
    """D-tiled / row-tiled MLA-dense backward (the DEFAULT dense-MLA path). q [N,H,BS,D]; k,v latent
    [N,KV,D]. Returns (dq,dk,dv,dsink). dq D-tiles the head dim (big BLOCK_M rows); dk/dv tiles the
    key dim (BLOCK_KV) and sums over row-tiles (holds dk_acc/dv_acc[BLOCK_KV,D] + Q/DO in the UB)."""
    q = q.contiguous()
    N, H, BS, D = q.shape
    KV = k.shape[1]
    HR = H * BS
    do = do.contiguous()
    lse = lse.contiguous()
    delta = (do.float() * o.float()).sum(-1).contiguous()
    BLOCK_D = triton.next_power_of_2(D)
    ncores = _num_cores(q.device)

    def gsize(nt):
        return num_programs if num_programs else min(nt, ncores)

    # dq (D-tiled, row-tiled)
    dq = torch.empty_like(q)
    num_m_tiles = triton.cdiv(HR, BM_DQ)
    nt_dq = N * num_m_tiles
    _bwd_dq_mla_dense_ascend_kernel[(gsize(nt_dq),)](
        q, k, v, do, lse, delta, dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        H, HR, KV, scale, nt_dq, num_m_tiles,
        BS=BS, BLOCK_M=BM_DQ, BLOCK_N=KV, BLOCK_K=BK_DQ, D=D,
    )

    # dk/dv (key-tiled, row-tile inner loop)
    dk_f = torch.empty(N, KV, D, device=q.device, dtype=k.dtype)
    dv_f = torch.empty(N, KV, D, device=q.device, dtype=v.dtype)
    num_kv_tiles = triton.cdiv(KV, BLOCK_KV)
    num_m_tiles2 = triton.cdiv(HR, BM_DKDV)
    nt_kv = N * num_kv_tiles
    _bwd_dkdv_mla_dense_ascend_kernel[(gsize(nt_kv),)](
        q, k, v, do, lse, delta, dk_f, dv_f,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk_f.stride(0), dk_f.stride(1), dk_f.stride(2),
        dv_f.stride(0), dv_f.stride(1), dv_f.stride(2),
        H, HR, KV, scale, nt_kv, num_kv_tiles, num_m_tiles2,
        BS=BS, BLOCK_M=BM_DKDV, BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D, D=D,
    )

    LOG2E_F = 1.4426950408889634
    dsink = -(torch.exp2(sink.float().view(1, H, 1) * LOG2E_F - lse) * delta).sum(dim=(0, 2))
    return dq, dk_f, dv_f, dsink.to(sink.dtype)


def swa_sink_bwd_ascend(q, k, v, sink, o, lse, do, win_left, win_right, dense, scale,
                        BLOCK_M=None, BLOCK_N=None, num_programs=None, HG=None):
    """Ascend-shaped fused backward (1-D core-capped grid-stride). Returns (dq, dk, dv, dsink).
    Same signature/semantics as swa_sink_bwd; GPU-testable. BLOCK_M/BLOCK_N default by head_dim.

    The MLA-dense production shape (dense + k.dim()==3) uses the D-tiled / row-tiled backward by
    default (rows = flattened (head,query) pairs on the matmul M axis; dq D-tiles the head dim).
    Windowed / MHA keep the validated per-(b,h) kernels. HG is accepted for API compat (ignored)."""
    if dense and k.dim() == 3:
        return _swa_sink_bwd_mla_dense(q, k, v, sink, o, lse, do, scale, num_programs=num_programs)
    B, H, LQ, D = q.shape
    BLOCK_M, BLOCK_N = _default_blocks(D, BLOCK_M, BLOCK_N)
    BLOCK_M, BLOCK_N = _bwd_safe_blocks(D, BLOCK_M, BLOCK_N)   # keep K+V (x multi-buffer) in L1
    mla = (k.dim() == 3)
    LK = k.shape[1] if mla else k.shape[2]
    windowed = not dense
    do = do.contiguous()
    lse = lse.contiguous()
    delta = (do.float() * o.float()).sum(-1).contiguous()
    BLOCK_D = triton.next_power_of_2(D)
    num_m_blocks = triton.cdiv(LQ, BLOCK_M)
    num_n_blocks = triton.cdiv(LK, BLOCK_N)
    ncores = _num_cores(q.device)

    def gsize(num_tiles):
        return num_programs if num_programs else min(num_tiles, ncores)

    skb, skh, skn, skd = _kv_strides(k)
    svb, svh, svn, svd = _kv_strides(v)

    # dq (query-tiles)
    dq = torch.empty_like(q)
    nt_dq = num_m_blocks * B * H
    _bwd_dq_ascend_kernel[(gsize(nt_dq),)](
        q, k, v, do, lse, delta, dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        skb, skh, skn, skd, svb, svh, svn, svd,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        H, LQ, LK, win_left, win_right, scale, nt_dq, num_m_blocks,
        WINDOWED=windowed, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )

    # dk/dv (key-tiles)
    if mla:
        dk_f = torch.empty(B, LK, D, device=q.device, dtype=k.dtype)
        dv_f = torch.empty(B, LK, D, device=q.device, dtype=v.dtype)
        nt_kv = num_n_blocks * B
        _bwd_dkdv_mla_ascend_kernel[(gsize(nt_kv),)](
            q, k, v, do, lse, delta, dk_f, dv_f,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_f.stride(0), dk_f.stride(1), dk_f.stride(2),
            dv_f.stride(0), dv_f.stride(1), dv_f.stride(2),
            H, LQ, LK, win_left, win_right, scale, nt_kv, num_n_blocks,
            WINDOWED=windowed, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
        )
    else:
        dk_f = torch.empty(B, H, LK, D, device=q.device, dtype=k.dtype)
        dv_f = torch.empty(B, H, LK, D, device=q.device, dtype=v.dtype)
        nt_kv = num_n_blocks * B * H
        _bwd_dkdv_ascend_kernel[(gsize(nt_kv),)](
            q, k, v, do, lse, delta, dk_f, dv_f,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            skb, skh, skn, skd, svb, svh, svn, svd,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_f.stride(0), dk_f.stride(1), dk_f.stride(2), dk_f.stride(3),
            dv_f.stride(0), dv_f.stride(1), dv_f.stride(2), dv_f.stride(3),
            H, LQ, LK, win_left, win_right, scale, nt_kv, num_n_blocks,
            WINDOWED=windowed, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
        )
    dk, dv = dk_f, dv_f

    LOG2E_F = 1.4426950408889634
    dsink = -(torch.exp2(sink.float().view(1, H, 1) * LOG2E_F - lse) * delta).sum(dim=(0, 2))
    return dq, dk, dv, dsink.to(sink.dtype)


# ---------------------------------------------------------------------------
# Autograd wrapper: the complete Ascend-shaped fwd+bwd training op (mirrors the CUDA
# _SwaSinkAttnFn but uses the 1-D-grid Ascend kernels). GPU-testable; A3-ready.
# ---------------------------------------------------------------------------
class _SwaSinkAscendFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, win_left, win_right, scale, dense, BLOCK_M, BLOCK_N):
        o, lse = swa_sink_attn_fwd_ascend(q, k, v, sink, win_left, win_right, scale=scale,
                                          dense=dense, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        ctx.save_for_backward(q, k, v, sink, o, lse)
        ctx.win_left, ctx.win_right, ctx.scale, ctx.dense = win_left, win_right, scale, dense
        ctx.BLOCK_M, ctx.BLOCK_N = BLOCK_M, BLOCK_N
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sink, o, lse = ctx.saved_tensors
        dq, dk, dv, ds = swa_sink_bwd_ascend(q, k, v, sink, o, lse, do, ctx.win_left, ctx.win_right,
                                             ctx.dense, ctx.scale, ctx.BLOCK_M, ctx.BLOCK_N)
        return (dq, dk, dv, ds, None, None, None, None, None, None)


def swa_sink_attn_ascend(q, k, v, sink, win_left, win_right, scale=None, BLOCK_M=None, BLOCK_N=None):
    """AUTOGRAD Ascend-shaped windowed self-attention + sink (fwd+bwd). o [B,H,L,D].
    BLOCK_M/BLOCK_N default by head_dim (small for D>=256); fwd and bwd share the same tiles."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    bm, bn = _default_blocks(q.shape[-1], BLOCK_M, BLOCK_N)
    return _SwaSinkAscendFn.apply(q, k, v, sink, win_left, win_right, scale, False, bm, bn)


def dense_sink_attn_ascend(q, k, v, sink, scale=None, BLOCK_M=None, BLOCK_N=None):
    """AUTOGRAD Ascend-shaped dense cross-attention + sink = gold BLOCK form (fwd+bwd)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    bm, bn = _default_blocks(q.shape[-1], BLOCK_M, BLOCK_N)
    return _SwaSinkAscendFn.apply(q, k, v, sink, 0, 0, scale, True, bm, bn)
