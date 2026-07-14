#!/usr/bin/env python3
"""SECONDARY diagnostic: does the compiled SAS op honor the non-causal win, or was it built from the
UPSTREAM causal-only source? RUN THIS ONLY IF the parity gap survives the harness fix (see below).

WHY / STATUS (read this first)
------------------------------
The original maxAbs~1.39 in `fused_sas_vs_reference_parity.py` was NOT a window bug — it was a
HARNESS bug: build_scenario fed per-head-INDEPENDENT K into shared_kv=True (MLA num_kv_heads=1). The
SAS op reads only head 0 (`dspark_attention.py:246` `k_ctx[:, :1, :]`) and broadcasts it to all H
query heads, while the reference uses each head's own K -> they disagreed on H-1 of H heads by
construction, nothing to do with the op. Fixed (parity build_scenario now shares one KV latent).
And the fork source IS non-causal: in va-src/va_fix the non-quant op's
`sparse_attn_sharedkv_tiling.cpp:1365,1368` reads `oriWinLeft_ < 0 / oriWinRight_ < 0` (PR #11196
relaxed the upstream `!= 127 / != 0` AND made the kernel honor win_right). The `.refsrc`/upstream
trees still show `!= 127` (causal) — that's a DIFFERENT tree, and the causal kv-quant op is not on
the draft path. So the op is expected correct.

THIS SCRIPT is only useful for the remaining BUILD question: if, AFTER the harness fix, PROD still
!= REF, is it because THIS NODE built the .so from the upstream (causal) source instead of the fork?
It computes the reference under BOTH window interpretations and reports which the .so matches.

Run at BS=5 (DSV4 inference block, win 132/4), not BS=7 (Qwen3 block7). Also just check the built
source directly:  grep -n 'oriWinRight_' <built csrc>/sparse_attn_sharedkv_tiling.cpp
  ->  '< 0'  = you built the fork (non-causal, correct)
  ->  '!= 0' = you built upstream (causal) -> rebuild from the dspark-dsv4 branch.

WHAT THIS DOES
--------------
Same scenario as the parity script, but computes the reference under BOTH window interpretations by
masking on absolute token positions (unambiguous — no guessing the op's internal alignment):
  * REF_noncausal : keep key kp for query qp if  qp-134 <= kp <= qp+6   (the intended DSV4 window)
  * REF_causal    : keep key kp for query qp if  qp-127 <= kp <= qp+0   (the upstream causal window)
Then compares the compiled op (PROD) to each. The one PROD matches tells you what the op computes.

VERDICT (run at the DSV4 block; win_left/win_right come from _dspark_sas_window(BS, WIN))
-------
- PROD ~= REF_noncausal (and >> REF_causal): the op is CORRECT (built from the fork). Expected result.
- PROD ~= REF_causal    (and >> REF_noncausal): this NODE built the UPSTREAM causal op. Rebuild
  vllm-ascend from the dspark-dsv4 branch whose sparse_attn_sharedkv_tiling.cpp:1365-1369 reads `< 0`.
- PROD matches NEITHER: not a window issue — a numeric/build-corruption artifact (or, if you skipped
  the harness fix, still the per-head-K scenario bug).

NOTE: our Triton kernel implements the true non-causal window and matches REF_noncausal at fp32
(5.96e-7), so this is purely a production-op build question, not a question about our kernel.

RUN (A3 NPU, env dspark-dsv4-*, vllm_ascend with the SAS op built):
    python diag_sas_window.py
    DTYPE=float32 python diag_sas_window.py    # removes bf16 noise; gaps become crisp
"""
import os

import torch  # noqa: E402  (parity import below pulls torch_npu + loads the .so)

# Reuse the EXACT scenario, overrides, and compare() from the trusted parity harness (its module-level
# _ensure_sas_op() also loads vllm_ascend_C.so). This guarantees we diagnose the same inputs.
from fused_sas_vs_reference_parity import (  # noqa: E402
    BS, DT, NBLK, SCALE, WIN, _override, _run, build_scenario, compare, dsa,
)
from vllm_ascend.ops.dspark_attention import _dspark_sas_window  # noqa: E402

DEV = "npu:0"


def masked_reference(s, win_left, win_right, prec="fp32"):
    """dspark_attention's per-block loop, but with an explicit position-based sliding-window mask.
    win_left/win_right are counts of past/future tokens (inclusive of the diagonal via win_left).
    prec='fp32': exact oracle. prec='bf16': mimic the op's internal Cube precision — round Q/K/V and
    the softmax P to bf16 (fp32 accumulate + fp32 softmax + fp32 sink), i.e. a standard bf16 attention
    path. If PROD matches the bf16 ref but not the fp32 one, the residual is just bf16 Cube rounding."""
    def r16(x):  # bf16-round-trip (bf16 matmul inputs, fp32 accumulate) when prec='bf16', else fp32
        return x.to(torch.bfloat16).float() if prec == "bf16" else x.float()
    q = s["q"]
    positions = s["positions"].to(torch.long)
    request_slots = s["request_slots"].to(torch.long)
    k_cache = s["k_cache"]
    cache_positions = s["cache_positions"]
    cache_valid = s["cache_valid"]
    draft_k = s["draft_k"]
    sink_all = s["attn_sink"]
    cap = k_cache.shape[1]
    out = torch.empty_like(q)

    for off in range(0, positions.numel(), BS):
        end = min(off + BS, positions.numel())
        block_pos = positions[off:end]                       # [bs] query positions
        block_start = int(block_pos.min().item())
        ctx_end = block_start - 1
        ctx_start = max(0, ctx_end + 1 - WIN)
        slot = int(request_slots[off].item())

        # gather valid context (mirrors _gather_context_kv), keep positions aligned with the keys
        if ctx_end >= ctx_start:
            ctx_positions = torch.arange(ctx_start, ctx_end + 1, device=DEV)
            ci = (ctx_positions % cap).long()
            cached_pos = cache_positions[slot, ci].to(torch.long)
            valid = cache_valid[slot, ci] & (cached_pos == ctx_positions)
            k_ctx = k_cache[slot, ci][valid]                 # [nctx, H, D]
            ctx_positions = ctx_positions[valid]
        else:
            k_ctx = k_cache.new_empty((0,) + k_cache.shape[2:])
            ctx_positions = torch.empty(0, dtype=torch.long, device=DEV)

        k_blk = draft_k[off:end]                              # [bs, H, D]  (shared_kv -> also V)
        packed_k = torch.cat([k_ctx, k_blk], dim=0)          # [KV, H, D]
        key_pos = torch.cat([ctx_positions, block_pos], dim=0)  # [KV]

        qf = r16(q[off:end])                                 # [bs, H, D]  (bf16-rounded if prec='bf16')
        kf = r16(packed_k)
        scores = torch.einsum("qhd,khd->qhk", qf, kf) * SCALE   # [bs, H, KV]  (fp32 accumulate)
        # sliding-window mask on absolute positions: qp - win_left <= kp <= qp + win_right
        qp = block_pos.view(-1, 1, 1)                        # [bs,1,1]
        kp = key_pos.view(1, 1, -1)                          # [1,1,KV]
        keep = (kp >= qp - win_left) & (kp <= qp + win_right)
        scores = scores.masked_fill(~keep, float("-inf"))

        sink = sink_all[: q.shape[1]].float().view(1, -1, 1)  # [1,H,1]  (op takes sink as fp32)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_s = torch.exp(scores - scores_max)
        probs = exp_s / (exp_s.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        p_pv = probs.to(torch.bfloat16).float() if prec == "bf16" else probs   # P rounded before P@V
        out[off:end] = torch.einsum("qhk,khd->qhd", p_pv, kf).to(q.dtype)      # V = K (shared_kv)
    return out


def main():
    s = build_scenario()
    _, wl, wr = _dspark_sas_window(BS, WIN)
    print(f">>> SAS window diagnostic   dtype={DT}   passed to op: win_left={wl}, win_right={wr} "
          f"(non-causal); upstream op only supports 127/0 (causal)")

    if dsa._get_dspark_sas_ops(s["q"]) is None:  # noqa: SLF001
        raise SystemExit("!! SAS op not registered — nothing to diagnose. Build/load vllm_ascend_C.so.")

    # PROD: the compiled SAS op (disable only the generic custom op so the entry takes the SAS path).
    with _override(_get_dspark_attention_custom_op=lambda q: None):
        prod = _run(s)
    # REF (entry's own reference loop) as a sanity anchor.
    with _override(_get_dspark_attention_custom_op=lambda q: None,
                   _get_dspark_sas_ops=lambda q: None):
        ref_entry = _run(s)
    torch.npu.synchronize()

    ref_nc = masked_reference(s, wl, wr)                  # intended non-causal window, fp32 oracle
    ref_ca = masked_reference(s, 127, 0)                  # upstream causal 127/0, fp32
    ref_nc_bf16 = masked_reference(s, wl, wr, prec="bf16")  # non-causal, mimic op internal bf16 Cube

    def line(tag, a, b):
        r = compare(a, b)
        print(f"[{tag:24}] allclose={str(r['allclose']):5}  maxAbs={r['maxAbs']:.2e}  "
              f"meanAbs={r['meanAbs']:.2e}  meanRel={r['meanRel']:.2e}")
        return r

    print()
    line("sanity ref_nc vs entry", ref_nc, ref_entry)   # my non-causal ref should equal the entry's loop
    r_nc = line("PROD vs ref_noncausal fp32", prod, ref_nc)     # intended window, exact
    r_ca = line("PROD vs ref_causal127 fp32", prod, ref_ca)     # upstream window, exact
    r_ncb = line("PROD vs ref_noncausal bf16", prod, ref_nc_bf16)  # intended window at op's bf16 prec

    # Per-block-position error breakdown. Block token i (0-indexed) needs its FUTURE block tokens
    # i+1..BS-1 (via win_right). If the op's non-causal (win_right) path is broken, the error is
    # CONCENTRATED at the early positions (many future tokens missed) and ~0 at the last (none needed).
    nblk = prod.shape[0] // BS
    d = (prod.float() - ref_nc.float()).abs().view(nblk, BS, *prod.shape[1:])
    per_pos = d.mean(dim=tuple(range(2, d.dim()))).mean(dim=0)   # [BS] mean|err| per within-block pos
    print("    per-block-position mean|PROD-ref_nc| (pos 0 sees the MOST future block tokens):")
    print("      " + "  ".join(f"p{i}={per_pos[i].item():.2e}" for i in range(BS)))
    print()

    d_nc, d_ca, d_ncb = r_nc["maxAbs"], r_ca["maxAbs"], r_ncb["maxAbs"]
    noncausal = d_nc < d_ca                       # closer to non-causal than to causal?
    bf16_closes = d_ncb < 0.5 * d_nc              # does the op's bf16 precision explain the fp32 gap?
    if noncausal and (d_ncb < 5e-2 or bf16_closes):
        print(">>> VERDICT: PROD is NON-CAUSAL (closer to non-causal than causal) and the bf16-internal "
              "reference CLOSES the gap -> the op is CORRECT; the residual vs the fp32 oracle is just the "
              "op's internal bf16 Cube precision (the same bf16 error any bf16 attention op has, incl. "
              "ours). Not a window/build bug. Confirm equivalence via `ours_vs_production.py`.")
    elif not noncausal and d_ca < 5e-2:
        print(">>> VERDICT: PROD matches the CAUSAL 127/0 window -> this build's SAS op is the UPSTREAM "
              "causal op. Rebuild vllm-ascend from the dspark-dsv4 commit whose "
              "sparse_attn_sharedkv_tiling.cpp:1365-1369 reads `< 0`.")
    elif noncausal:
        print(">>> VERDICT: PROD is directionally NON-CAUSAL but even the bf16-internal ref doesn't close "
              "the gap -> residual is neither window nor plain bf16 rounding. Investigate sink/scale "
              "handling; run `ours_vs_production.py` to compare the op against our validated kernel.")
    else:
        print(">>> VERDICT: PROD matches neither window cleanly -> numeric/build artifact; compare "
              "against our kernel with `ours_vs_production.py`.")


if __name__ == "__main__":
    main()
