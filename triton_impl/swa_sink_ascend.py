#!/usr/bin/env python3
"""Ascend-NPU-shaped FORWARD kernel for SWA-non-causal-sink attention.

Same math as swa_sink_fwd.py, but reshaped for the Triton-Ascend backend. It STILL runs and
validates on CUDA (identical results) so correctness can be nailed on the GPU before moving to
the A3 — only backend lowering remains NPU-specific. See /workspace/skills/triton-ascend
(latency-optimizer/references/checklist.md, kernel-generator/references/hw-ascend910-9362.md).

What changed vs the CUDA kernel (Ascend constraints):
  * 1-D GRID, core-capped via a GRID-STRIDE loop. Ascend forbids multi-dim grids AND wants the
    grid <= physical core count (mix cube+vector op -> <= cube cores). So we launch
    grid = min(NUM_TILES, num_cores) programs and each strides `for tile in range(pid, NUM_TILES,
    num_programs)` over its share of (query-block, batch*head) tiles. On CUDA we also cap the grid
    (to the SM count) so the stride loop actually iterates and is validated there.
  * No `%` operator (Ascend forbids `a % b`): the tile/head decode uses only `//` and `-`
    (a - (a//b)*b), per the checklist.
  * Window comparisons cast to fp32 (Ascend vectorizes fp32 compares; exact for these indices).
  * int32 index math; no continue/break; fp32 softmax accumulation (unchanged); lse emitted.

Still TODO on the actual A3 (needs the hardware; not testable on GPU):
  * confirm tl.dot / tl.trans / exp2 / dynamic range lower on Triton-Ascend; `input_precision`
    is kept for CUDA fp32 accuracy but Ascend's Cube uses a different precision path -- revisit
    (may drop / use tf32x3) there. Then the BACKWARD Ascend port.
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
    NUM_TILES, NUM_M_BLOCKS,                        # total tiles + cdiv(LQ,BLOCK_M) for the decode
    WINDOWED: tl.constexpr, FP32_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)                   # grid size (<= core count on NPU)
    offs_d = tl.arange(0, BLOCK_D)                  # loop-invariant, hoisted out of the tile loop
    d_mask = offs_d < D
    qk_scale = scale * LOG2E

    # ---- 1-D grid-stride loop over tiles; each is one (query-block, batch*head) ----
    for tile in range(pid, NUM_TILES, num_prog):
        # decode tile -> (b, h, query-block) with NO `%` (Ascend rule): a % b == a - (a//b)*b
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

        sink_val = tl.load(Sink + h).to(tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink_val * LOG2E   # seed sink as a virtual key
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

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
            # window predicate in fp32 (Ascend vectorizes fp32 compares; exact for these indices)
            mf = offs_m[:, None].to(tl.float32)
            nf = offs_n[None, :].to(tl.float32)
            if WINDOWED:
                keep = (nf >= mf - WIN_LEFT) & (nf <= mf + WIN_RIGHT) & (nf < LK)
            else:
                keep = nf < LK
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


@triton.jit
def _swa_sink_fwd_mla_dense_kernel(
    Q, K, V, Sink, Out, Lse,
    stride_qn, stride_qh, stride_qm, stride_qd,      # Q [N, H, BS, D]; flat row stride = stride_qm
    stride_kn, stride_kk, stride_kd,                 # K [N, KV, D] shared latent (MLA)
    stride_vn, stride_vk, stride_vd,
    stride_on, stride_oh, stride_om, stride_od,      # Out [N, H, BS, D]
    H, HR, KV, scale,                                # HR = H*BS = rows per block
    NUM_TILES, NUM_M_TILES,                          # tiles = N * cdiv(HR, BLOCK_M)
    BS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    D: tl.constexpr,
):
    """D-TILED MLA-dense forward (the production block-attention shape). One program owns a block n
    and a row-tile of BLOCK_M rows = flattened (head, query) pairs — because MLA shares one KV latent
    across all H heads, every row attends the SAME K/V[KV,D]. KV=135 is tiny so it's a SINGLE-PASS
    softmax (full scores[M,KV], no online rescale). The D=512 head dim is TILED by BLOCK_K in BOTH
    matmuls so nothing of shape [*,512] is ever on chip (fits the 192KB UB / 512KB L1 per the
    Triton-Ascend matmul guide: for k in range(0,D,BLOCK_K)), which lets M grow to fill the Cube.
    Per-row sink: head(row)=row//BS, an fp32 logit in the denominator, absent from P@V. 1-D
    grid-stride, no `%`; masked loads make the boundary `tl.where`/mask redundant so they're omitted."""
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)                   # BLOCK_N == KV (all keys valid; no key mask)
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):
        n = tile // NUM_M_TILES
        mt = tile - n * NUM_M_TILES
        m0 = mt * BLOCK_M
        rows = m0 + offs_m                           # global (head,query) row within block n
        row_ok = rows < HR

        # ---- QK^T with D-tiling -> scores[M, KV] ----
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        base_q = Q + n * stride_qn + m0 * stride_qm  # flat row stride spans heads (C-contiguous)
        base_k = K + n * stride_kn
        for d0 in range(0, D, BLOCK_K):
            offs_k = d0 + tl.arange(0, BLOCK_K)
            q = tl.load(base_q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd,
                        mask=(row_ok[:, None] & (offs_k[None, :] < D)), other=0.0)
            kk = tl.load(base_k + offs_n[:, None] * stride_kk + offs_k[None, :] * stride_kd,
                         mask=(offs_k[None, :] < D), other=0.0)
            scores += tl.dot(q, tl.trans(kk), input_precision="ieee").to(tl.float32)
        scores = scores * qk_scale

        # ---- softmax with per-row sink (single pass; KV all valid) ----
        sink_val = tl.load(Sink + (rows // BS), mask=row_ok, other=0.0).to(tl.float32)
        s = sink_val * LOG2E
        m = tl.maximum(tl.max(scores, axis=1), s)
        p = tl.math.exp2(scores - m[:, None])
        l = tl.sum(p, axis=1) + tl.math.exp2(s - m)  # denom keeps the sink term
        p = p.to(V.dtype.element_ty)

        # ---- P@V with D-tiling -> o[M, D] chunk by chunk ----
        base_v = V + n * stride_vn
        base_o = Out + n * stride_on + m0 * stride_om
        for d0 in range(0, D, BLOCK_K):
            offs_k = d0 + tl.arange(0, BLOCK_K)
            vv = tl.load(base_v + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vd,
                         mask=(offs_k[None, :] < D), other=0.0)
            o = tl.dot(p, vv, input_precision="ieee").to(tl.float32) / l[:, None]
            tl.store(base_o + offs_m[:, None] * stride_om + offs_k[None, :] * stride_od,
                     o.to(Out.dtype.element_ty), mask=(row_ok[:, None] & (offs_k[None, :] < D)))
        tl.store(Lse + n * HR + rows, m + tl.log2(l), mask=row_ok)


def swa_sink_attn_fwd_mla_dense_ascend(q, k, v, sink, scale=None, HG=None, BLOCK_M=None,
                                       BLOCK_K=None, num_programs=None):
    """D-tiled MLA-dense forward. q [N,H,BS,D]; k,v latent [N,KV,D]; sink [H]. Returns (o, lse) in the
    SAME layout/format as swa_sink_attn_fwd_ascend (o [N,H,BS,D], lse [N,H,BS]) so the backward
    consumes it unchanged. BLOCK_M = flattened (head,query) rows per program (M axis; the Cube-fill
    lever); BLOCK_K tiles the D=512 head dim so nothing of shape [*,512] is on chip. HG is a
    convenience alias: BLOCK_M = next_pow2(HG*BS)."""
    assert q.dim() == 4 and k.dim() == 3, "MLA-dense path: q[N,H,BS,D], kv[N,KV,D]"
    # rows are flattened into M via a single row stride (stride_qm), so [H,BS,D] must be C-contiguous.
    q = q.contiguous()
    N, H, BS, D = q.shape
    KV = k.shape[1]
    HR = H * BS
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(N, H, BS, device=q.device, dtype=torch.float32)
    if BLOCK_M is None:
        BLOCK_M = triton.next_power_of_2(HG * BS) if HG else 64
    BLOCK_M = min(BLOCK_M, triton.next_power_of_2(HR))
    BLOCK_K = 128 if BLOCK_K is None else BLOCK_K    # D-tile (512B-aligned; keeps [*,BK] on chip)
    BLOCK_N = KV                                     # KV=135 in one shot (tiny), single-pass softmax
    num_m_tiles = triton.cdiv(HR, BLOCK_M)
    num_tiles = N * num_m_tiles
    gsize = num_programs if num_programs else min(num_tiles, _num_cores(q.device))
    _swa_sink_fwd_mla_dense_kernel[(gsize,)](
        q, k, v, sink, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, HR, KV, scale, num_tiles, num_m_tiles,
        BS=BS, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, D=D,
    )
    return o, lse


def _kv_strides(t):
    if t.dim() == 4:
        return t.stride(0), t.stride(1), t.stride(2), t.stride(3)
    return t.stride(0), 0, t.stride(1), t.stride(2)      # MLA-shared [B,Lk,D] -> head-stride 0


def _default_blocks(D, block_m, block_n):
    """Pick safe default tile sizes when unspecified: large head_dim (e.g. D=512) needs small
    tiles to fit on-chip memory (GPU shared / Ascend UB). A3 perf tuning refines these."""
    if block_m is not None and block_n is not None:
        return block_m, block_n
    bm, bn = (16, 16) if D >= 256 else (32, 32)
    return (block_m or bm), (block_n or bn)


def _num_cores(device):
    """Physical core count for the grid cap: num_aicore on Ascend, else the CUDA SM count."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(device.index)
        for key in ("num_aicore", "num_vectorcore"):
            n = props.get(key)
            if n and n > 0:
                return int(n)
    except Exception:  # noqa: BLE001
        pass
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # noqa: BLE001
        return 64


def swa_sink_attn_fwd_ascend(q, k, v, sink, win_left, win_right, scale=None,
                             dense=False, BLOCK_M=None, BLOCK_N=None, fp32_qk=True, num_programs=None,
                             HG=None, BLOCK_K=None):
    """Ascend-shaped forward (1-D, core-capped grid-stride). Returns (o, lse). q [B,H,LQ,D];
    k,v [B,H,LK,D] (MHA) or [B,LK,D] (MLA). dense=True -> no window (Lq!=Lk allowed). Runs on CUDA
    for validation. BLOCK_M/BLOCK_N default by head_dim (small for D>=256). num_programs overrides
    the grid size (default = min(NUM_TILES, core count)).

    The MLA-dense production shape (dense + k.dim()==3) uses the D-tiled kernel: rows = flattened
    (head,query) pairs on the matmul M axis (BLOCK_M lever, HG=next_pow2(HG*BS) alias), D=512 tiled
    by BLOCK_K so nothing [*,512] is on chip. Windowed / MHA keep the plain flash kernel."""
    if dense and k.dim() == 3:
        return swa_sink_attn_fwd_mla_dense_ascend(q, k, v, sink, scale=scale, HG=HG, BLOCK_M=BLOCK_M,
                                                  BLOCK_K=BLOCK_K, num_programs=num_programs)
    assert q.dim() == 4, "q must be [B,H,LQ,D]"
    B, H, LQ, D = q.shape
    LK = k.shape[1] if k.dim() == 3 else k.shape[2]
    BLOCK_M, BLOCK_N = _default_blocks(D, BLOCK_M, BLOCK_N)
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(B, H, LQ, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    num_m_blocks = triton.cdiv(LQ, BLOCK_M)
    num_tiles = num_m_blocks * B * H
    gsize = num_programs if num_programs else min(num_tiles, _num_cores(q.device))
    skb, skh, skn, skd = _kv_strides(k)
    svb, svh, svn, svd = _kv_strides(v)
    _swa_sink_fwd_ascend_kernel[(gsize,)](              # 1-D grid, core-capped (grid-stride loop)
        q, k, v, sink, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        skb, skh, skn, skd, svb, svh, svn, svd,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, LQ, LK, win_left, win_right, scale,
        num_tiles, num_m_blocks,
        WINDOWED=not dense, FP32_QK=fp32_qk,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
    )
    return o, lse
