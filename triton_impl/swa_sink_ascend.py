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
    stride_qn, stride_qh, stride_qm, stride_qd,      # Q [N, H, BS, D]
    stride_kn, stride_kk, stride_kd,                 # K [N, KV, D] shared latent (MLA)
    stride_vn, stride_vk, stride_vd,
    stride_on, stride_oh, stride_om, stride_od,      # Out [N, H, BS, D]
    H, KV, scale,
    NUM_TILES, NUM_HG,                               # tiles = N * cdiv(H, HG)
    HG: tl.constexpr, BS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    """HEAD-BATCHED MLA-dense forward (the production block-attention shape). One program handles
    one (block n, head-group g): it flattens HG heads x BS queries into the matmul M axis (M=HG*BS
    instead of 7) so the Cube isn't starved on the M axis — because MLA shares one KV latent across
    all H heads, every row attends the SAME K/V[KV,D]. Per-row sink: row r -> head h0 + r//BS (fp32
    logit in the denominator, absent from P@V). Uses the SAME flash online-softmax + staged K/V
    n-loop as the validated kernel (small BLOCK_N so K/V don't blow the 512KB L1, and acc[M,D] fits
    the 192KB UB — which caps M ~= 32 at D=512, i.e. HG~=4). Same math/lse. 1-D grid-stride, no `%`."""
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_m = tl.arange(0, BLOCK_M)
    qk_scale = scale * LOG2E

    for tile in range(pid, NUM_TILES, num_prog):
        n = tile // NUM_HG
        g = tile - n * NUM_HG
        h0 = g * HG
        rows_valid = (tl.minimum(h0 + HG, H) - h0) * BS      # BS*heads present in this group
        row_ok = offs_m < rows_valid

        # Q for heads [h0:h0+HG] x BS rows is contiguous -> flat [HG*BS, D], row stride = stride_qm.
        base_q = Q + n * stride_qn + h0 * stride_qh
        q_block = tl.load(base_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                          mask=(row_ok[:, None] & d_mask[None, :]), other=0.0)
        # per-row sink logit: head(row) = h0 + row // BS  (no `%`; masked gather)
        sink_val = tl.load(Sink + (h0 + offs_m // BS), mask=row_ok, other=0.0).to(tl.float32)

        m_i = sink_val * LOG2E                                # seed sink as a per-row virtual key
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for n_start in range(0, KV, BLOCK_N):                # staged K/V (keeps L1/UB bounded)
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_valid = offs_n < KV
            k_block = tl.load(K + n * stride_kn + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                              mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
            qk = tl.dot(q_block, tl.trans(k_block), input_precision="ieee").to(tl.float32) * qk_scale
            qk = tl.where(n_valid[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_block = tl.load(V + n * stride_vn + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                              mask=(n_valid[:, None] & d_mask[None, :]), other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block, input_precision="ieee")
            m_i = m_ij

        o = acc / l_i[:, None]
        base_o = Out + n * stride_on + h0 * stride_oh
        tl.store(base_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 o.to(Out.dtype.element_ty), mask=(row_ok[:, None] & d_mask[None, :]))
        tl.store(Lse + (n * H + h0) * BS + offs_m, m_i + tl.log2(l_i), mask=row_ok)


def swa_sink_attn_fwd_mla_dense_ascend(q, k, v, sink, scale=None, HG=4, BLOCK_N=None,
                                       num_programs=None):
    """Head-batched MLA-dense forward. q [N,H,BS,D]; k,v latent [N,KV,D]; sink [H]. Returns (o, lse)
    in the SAME layout/format as swa_sink_attn_fwd_ascend (o [N,H,BS,D], lse [N,H,BS]) so the
    existing backward consumes it unchanged. HG = heads batched into the matmul M axis (M=HG*BS).
    Flash online softmax with staged K/V (BLOCK_N default 64) so it stays in L1/UB. Note: acc[M,D]
    is fp32 in the 192KB UB, so M (=HG*BS, padded) is capped ~32 at D=512 -> keep HG<=4 there."""
    assert q.dim() == 4 and k.dim() == 3, "head-batched path is MLA dense: q[N,H,BS,D], kv[N,KV,D]"
    # heads are flattened into M via a single row stride (stride_qm), so the [H,BS,D] block must be
    # C-contiguous (stride_qh == BS*stride_qm). The harness passes .contiguous(); enforce it here.
    q = q.contiguous()
    N, H, BS, D = q.shape
    KV = k.shape[1]
    scale = D ** -0.5 if scale is None else scale
    sink = sink.to(torch.float32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(N, H, BS, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 64 if BLOCK_N is None else BLOCK_N        # small staged key-block (fits L1/UB at D=512)
    HG = min(HG, H)
    BLOCK_M = triton.next_power_of_2(HG * BS)
    num_hg = triton.cdiv(H, HG)
    num_tiles = N * num_hg
    gsize = num_programs if num_programs else min(num_tiles, _num_cores(q.device))
    _swa_sink_fwd_mla_dense_kernel[(gsize,)](
        q, k, v, sink, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, KV, scale, num_tiles, num_hg,
        HG=HG, BS=BS, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, D=D,
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
                             HG=None):
    """Ascend-shaped forward (1-D, core-capped grid-stride). Returns (o, lse). q [B,H,LQ,D];
    k,v [B,H,LK,D] (MHA) or [B,LK,D] (MLA). dense=True -> no window (Lq!=Lk allowed). Runs on CUDA
    for validation. BLOCK_M/BLOCK_N default by head_dim (small for D>=256). num_programs overrides
    the grid size (default = min(NUM_TILES, core count)).

    HG (heads-per-tile, opt-in): for the MLA-dense production shape, dispatch to the head-batched
    kernel that flattens HG heads into the matmul M axis (M=HG*BS) — much better Cube utilisation
    on the NPU. Only valid for dense + MLA (k.dim()==3); ignored otherwise. Same (o, lse) format."""
    if HG and dense and k.dim() == 3:
        return swa_sink_attn_fwd_mla_dense_ascend(q, k, v, sink, scale=scale, HG=HG,
                                                  BLOCK_N=BLOCK_N, num_programs=num_programs)
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
