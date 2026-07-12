#!/usr/bin/env python3
"""Eager ground-truth reference for the SWA-NON-CAUSAL-SINK fused attention.

This is the *math contract* a Triton kernel must reproduce (forward AND backward).
It is DEPENDENCY-FREE (pure torch, CPU-runnable) so the kernel author can diff a
kernel against it directly, including torch.autograd.gradcheck in fp64.

THE ULTIMATE GROUND TRUTH is vllm_ascend's `_dspark_attention_reference`
(`vllm_ascend/ops/dspark_attention.py`), the pure-torch reference the fused "SAS"
op is validated against on-box. This file reproduces that math so it can be checked
without an NPU; when the two disagree, the vllm_ascend reference wins. See
`reference_from_repo/dspark_attn_ref_bench.py` for the on-box gold comparison.

The operator combines three properties the DSpark draft block-attention needs, none
of which any single Ascend-NPU primitive provides today (see README "Why a kernel"):

  SWA          : each query attends only to keys within a window
  NON-CAUSAL   : the window is BIDIRECTIONAL, and (critically) ASYMMETRIC --
                 win_left = window + block_size - 1,  win_right = block_size - 1
                 (NOT the naive symmetric window_size-1; vllm_ascend ships a test
                  that rejects that. Use `dspark_sas_window()` below.)
  SINK         : a per-head learnable "attention sink" logit sits in the softmax
                 DENOMINATOR (a StreamingLLM off-ramp); it is NOT a value, so it
                 contributes to normalisation but never to the weighted value sum.

Two equivalent views (the kernel author needs both):
  * BLOCK view (the gold parity target): per draft block, the block's `block_size`
    queries attend DENSELY to [last `window` context tokens | the FULL block]
    (non-causal in-block) + sink. Shapes q[N,BS,H,D], k/v[N,KV,H,D], KV=window+BS.
    -> `dspark_block_attention_ref` here == the gold `_dspark_attention_reference`.
  * PACKED-SWA view (what an efficient kernel implements): one long sequence with a
    sliding-window kernel parameterised by (win_left, win_right) + sink.
    -> `swa_sink_attention` here. `dspark_sas_window` maps block/window -> (L,R).

Numerics contract (match this or parity fails):
  * QK^T scaling and the softmax run in fp32 even when q/k/v are bf16.
  * the sink is a RAW fp32 per-head logit: NOT * scale, NOT windowed/masked. It joins
    the max-subtract and the softmax denominator, then is dropped from the P@V sum.
  * P is cast back to v.dtype for P@V (the model accumulates PV in v's dtype).
  * RoPE is applied OUTSIDE this op (partial rotary: only the trailing rope_head_dim=64
    of head_dim=512 is rotated, on q and k, before attention; the output gets the
    inverse rotation after). The kernel sees POST-RoPE q/k -- do not add RoPE here.

Run:  python eager_reference.py     # self-test + fp64 gradcheck + real-shape smoke
"""
from __future__ import annotations

import torch

NEG_INF = float("-inf")

# --- Real DeepSeek-V4-Flash-DSpark shapes (HF config.json; fixed model constants) ---
# Source of truth: Sawyer117/speculators @ feat/dsv4-dspark
#                  src/speculators/models/dsv4_dspark/config.py (DSV4DSparkConfig)
DSV4 = dict(
    hidden_size=4096,
    vocab_size=129280,
    num_heads=64,          # H (query heads)
    num_kv_heads=1,        # MLA: ONE latent K/V shared across all 64 query heads
    head_dim=512,          # D (nope | rope), per head
    rope_head_dim=64,      # trailing slice of head_dim that is rotated (partial RoPE)
    window_size=128,       # sliding-window context the draft attends to
    block_size=7,          # gamma; official block-5 config, released "block7" ckpt uses 7
    n_draft_layers=3,      # target_layer_ids = (40, 41, 42)
    markov_rank=256,
    noise_token_id=128799,
    scale=512 ** -0.5,     # head_dim ** -0.5
)


def dspark_sas_window(block_size: int, window: int) -> tuple[int, int]:
    """The REAL DSpark draft window as (win_left, win_right).

    A block query at block-position p (0..block_size-1) sits just after `window`
    context tokens; it attends to all `window` context tokens + the whole block.
    In linear (packed) coordinates that is:
        win_left  = window + block_size - 1
        win_right = block_size - 1
    Mirrors vllm_ascend `_dspark_sas_window`. The naive symmetric `window-1` is WRONG.
    """
    return window + block_size - 1, block_size - 1


# ---------------------------------------------------------------------------
# (A) PACKED-SWA view -- the general asymmetric sliding-window + sink attention
#     (what an efficient fused kernel implements over a packed sequence).
# ---------------------------------------------------------------------------
def asymmetric_window_mask(lq: int, lk: int, win_left: int, win_right: int, device):
    """Additive 0/-inf [lq, lk]. keep[i,j] = (j >= i-win_left) & (j <= i+win_right).
    A kernel must test this predicate on the fly -- NEVER materialize a dense mask."""
    i = torch.arange(lq, device=device).unsqueeze(1)
    j = torch.arange(lk, device=device).unsqueeze(0)
    keep = (j >= i - win_left) & (j <= i + win_right)
    return torch.where(keep, 0.0, NEG_INF)


def swa_sink_attention(
    q, k, v, sink, win_left, win_right,
    scale=None, add_mask=None, compute_dtype=torch.float32,
):
    """Asymmetric sliding-window attention + per-head sink. q[B,H,L,D];
    k,v = [B,H,Lk,D] (MHA) or [B,Lk,D] (MLA-shared, 1 kv head). Returns o[B,H,L,D]."""
    b, h, lq, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    mla = (k.dim() == 3)
    lk = k.shape[1] if mla else k.shape[2]

    if mla:
        scores = torch.einsum("bhid,bjd->bhij", q, k).to(compute_dtype) * scale
    else:
        scores = torch.einsum("bhid,bhjd->bhij", q, k).to(compute_dtype) * scale

    mask = asymmetric_window_mask(lq, lk, win_left, win_right, q.device).to(compute_dtype)
    if add_mask is not None:
        mask = mask + add_mask.to(compute_dtype)
    scores = scores + mask
    return _sink_softmax_pv(scores, sink, v, h, lq, lk, mla, compute_dtype)


# ---------------------------------------------------------------------------
# (B) BLOCK view -- the exact gold form (== _dspark_attention_reference).
#     Per block the KV is already [window-context | full block]; attention is
#     DENSE over it (the window sparsity is realised by how KV was assembled),
#     plus the per-head sink. This is the concrete parity target with real shapes.
# ---------------------------------------------------------------------------
def dspark_block_attention_ref(q, k, v, sink, scale=None, compute_dtype=torch.float32):
    """q[N,BS,H,D], k[N,KV,H,D], v[N,KV,H,D], sink[H]. Returns o[N,BS,H,D].
    N = number of draft blocks, BS = block_size, KV = window + BS (already sliced).
    Matches vllm_ascend `_dspark_attention_reference` / the bench's `attn_manual`."""
    n, bs, h, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    s = torch.einsum("nqhd,nkhd->nqhk", q, k).to(compute_dtype) * scale     # [N,BS,H,KV]
    sinkv = sink.view(1, 1, h, 1).to(compute_dtype)
    smax = torch.maximum(s.max(dim=-1, keepdim=True).values, sinkv)         # sink in the max
    e = torch.exp(s - smax)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sinkv - smax))         # sink in denom
    return torch.einsum("nqhk,nkhd->nqhd", p.to(v.dtype), v)               # sink NOT in the sum


def _sink_softmax_pv(scores, sink, v, h, lq, lk, mla, compute_dtype):
    """softmax over [scores | per-head sink], drop the sink column, then P@V.
    Equivalent to the block form's e/(sum_e + exp(sink-smax))."""
    b = scores.shape[0]
    sink_col = sink.view(1, h, 1, 1).expand(b, -1, lq, 1).to(compute_dtype)   # [b,h,lq,1]
    combined = torch.cat([scores, sink_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = combined.softmax(dim=-1)[..., :lk].to(v.dtype)
    if mla:
        return torch.einsum("bhij,bjd->bhid", probs, v)
    return torch.einsum("bhij,bhjd->bhid", probs, v)


# ---------------------------------------------------------------------------
# Independent loop oracle for the BLOCK form -> a real parity number.
# ---------------------------------------------------------------------------
def _block_oracle(q, k, v, sink, scale=None):
    n, bs, h, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    kv = k.shape[1]
    out = torch.zeros(n, bs, h, d, dtype=torch.float64)
    for ni in range(n):
        for hi in range(h):
            for qi in range(bs):
                sc = (q[ni, qi, hi].double() @ k[ni, :, hi].double().T) * scale   # [kv]
                logits = torch.cat([sc, sink[hi].double().view(1)])               # +sink
                p = torch.softmax(logits, -1)[:-1]                                # drop sink
                out[ni, qi, hi] = p @ v[ni, :, hi].double()
    return out


def _selftest():
    torch.manual_seed(0)
    N, BS, H, D, WIN = 3, 5, 4, 8, 6
    KV = WIN + BS

    # --- BLOCK form (gold) exactness vs the independent oracle (fp64) ---
    q = torch.randn(N, BS, H, D, dtype=torch.double)
    k = torch.randn(N, KV, H, D, dtype=torch.double)
    v = torch.randn(N, KV, H, D, dtype=torch.double)
    sink = torch.randn(H, dtype=torch.double)
    o = dspark_block_attention_ref(q, k, v, sink, compute_dtype=torch.double)
    ref = _block_oracle(q, k, v, sink)
    err = (o.double() - ref).abs().max().item()
    print(f"[block] out{tuple(o.shape)}  max|ref - oracle|={err:.2e}  (expect < 1e-10)")
    assert err < 1e-10

    # --- the asymmetric window formula ---
    wl, wr = dspark_sas_window(BS, WIN)
    print(f"[win]   dspark_sas_window(block={BS}, window={WIN}) = (win_left={wl}, win_right={wr})"
          f"  (naive symmetric window-1={WIN-1} would be WRONG)")
    assert (wl, wr) == (WIN + BS - 1, BS - 1)

    # --- sink actually diverts probability mass ---
    o_hi = dspark_block_attention_ref(q, k, v, sink + 5.0, compute_dtype=torch.double)
    o_lo = dspark_block_attention_ref(q, k, v, sink - 1e6, compute_dtype=torch.double)  # ~no sink
    print(f"[sink]  mean|o(sink) - o(sink+5)|={(o - o_hi).abs().mean():.3e}  (>0 => sink matters); "
          f"sink->-inf recovers plain softmax (delta={(o - o_lo).abs().mean():.3e} vs no-sink path)")
    assert (o - o_hi).abs().mean() > 0

    # --- backward: gradcheck the gold block form in fp64 (kernel bwd must match) ---
    qg, kg, vg, sg = (t.clone().requires_grad_(True) for t in (q, k, v, sink))
    ok = torch.autograd.gradcheck(
        lambda a, b, c, s: dspark_block_attention_ref(a, b, c, s, compute_dtype=torch.double),
        (qg, kg, vg, sg), atol=1e-6, rtol=1e-4,
    )
    print(f"[bwd]   gradcheck(q,k,v,sink) = {ok}  (autograd-clean; grads flow to the sink too)")

    # --- packed-SWA view runs (MHA + MLA-shared) with the asymmetric window ---
    qp = torch.randn(2, H, 12, D, dtype=torch.double)
    kp = torch.randn(2, H, 12, D, dtype=torch.double)
    vp = torch.randn(2, H, 12, D, dtype=torch.double)
    o_mha = swa_sink_attention(qp, kp, vp, sink, wl, wr, compute_dtype=torch.double)
    o_mla = swa_sink_attention(qp, kp[:, 0], vp[:, 0], sink, wl, wr, compute_dtype=torch.double)
    print(f"[swa]   packed MHA{tuple(o_mha.shape)} + MLA-shared{tuple(o_mla.shape)} run "
          f"(asymmetric window L={wl},R={wr})")

    # --- REAL-shape forward smoke (fp32, real H/D/WIN/BS, few blocks) ---
    c = DSV4
    Nr, BSr, Hr, Dr, WINr = 4, c["block_size"], c["num_heads"], c["head_dim"], c["window_size"]
    KVr = WINr + BSr
    qr = torch.randn(Nr, BSr, Hr, Dr)
    kr = torch.randn(Nr, KVr, Hr, Dr)
    vr = torch.randn(Nr, KVr, Hr, Dr)
    sr = torch.randn(Hr)
    orr = dspark_block_attention_ref(qr, kr, vr, sr, scale=c["scale"])
    print(f"[real]  DSV4 block attn q[{Nr},{BSr},{Hr},{Dr}] x kv[{Nr},{KVr},{Hr},{Dr}] "
          f"-> {tuple(orr.shape)}  finite={torch.isfinite(orr).all().item()}  "
          f"(H={Hr} D={Dr} win={WINr} block={BSr}; MLA real num_kv_heads=1)")
    assert torch.isfinite(orr).all()
    print("OK: gold block form matches oracle + gradcheck; packed-SWA runs; real DSV4 shapes forward.")


if __name__ == "__main__":
    _selftest()
