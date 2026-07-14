#!/usr/bin/env python3
"""DECISIVE diagnostic: does the compiled SAS op honor the non-causal 134/6 window, or does it
silently compute the upstream CAUSAL 127/0 window?

WHY
---
`fused_sas_vs_reference_parity.py` shows PROD (npu_sparse_attn_sharedkv) != REF (maxAbs~1.39) on the
team's known-good scenario. Reading the AscendC source explains it:

  csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp:1365
      OP_CHECK_IF(oriWinLeft_  != 127, "ori_win_left should be 127", FAIL);
      OP_CHECK_IF(oriWinRight_ != 0,   "ori_win_right should be 0",  FAIL);
  op_kernel/sparse_attn_sharedkv_common.h:310  oriWinRight = 0;  (causal-only kernel)

The UPSTREAM op only implements a CAUSAL window-127. But dspark_attention.py passes the DSV4
NON-CAUSAL asymmetric window:  _dspark_sas_window(block=7, win=128) -> win_left=134, win_right=6.
So if the .so built on this node is the upstream (or an incomplete fork) op, it computes causal-127
instead of non-causal-134/6 -> wrong result -> the observed maxAbs~1.39. (If it were merely a dtype/
numeric bug, PROD would still track REF within ~1e-2; a 1.39 gap is a DIFFERENT MASK.)

WHAT THIS DOES
--------------
Same scenario as the parity script, but computes the reference under BOTH window interpretations by
masking on absolute token positions (unambiguous — no guessing the op's internal alignment):
  * REF_noncausal : keep key kp for query qp if  qp-134 <= kp <= qp+6   (the intended DSV4 window)
  * REF_causal    : keep key kp for query qp if  qp-127 <= kp <= qp+0   (the upstream causal window)
Then compares the compiled op (PROD) to each. The one PROD matches tells you what the op computes.

VERDICT
-------
- PROD ~= REF_noncausal (and >> REF_causal): the op is CORRECT; the parity failure was something else.
- PROD ~= REF_causal    (and >> REF_noncausal): CONFIRMED — the built op computes the CAUSAL 127/0
  window. The fork's non-causal (win_right>0) kernel patch is NOT in this build. Rebuild vllm-ascend
  from the dspark-dsv4 branch commit that adds win_right support (the same one that relaxed the
  oriWinLeft==127 / oriWinRight==0 tiling asserts), or diff this node's csrc against the old node's.
- PROD matches NEITHER: a numeric/build-corruption bug, not a window-semantics bug.

NOTE: our Triton kernel implements the true non-causal 134/6 window and matches REF_noncausal at fp32
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


def masked_reference(s, win_left, win_right):
    """dspark_attention's per-block loop, but with an explicit position-based sliding-window mask.
    win_left/win_right are counts of past/future tokens (inclusive of the diagonal via win_left)."""
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

        qf = q[off:end].float()                              # [bs, H, D]
        scores = torch.einsum("qhd,khd->qhk", qf, packed_k.float()) * SCALE   # [bs, H, KV]
        # sliding-window mask on absolute positions: qp - win_left <= kp <= qp + win_right
        qp = block_pos.view(-1, 1, 1)                        # [bs,1,1]
        kp = key_pos.view(1, 1, -1)                          # [1,1,KV]
        keep = (kp >= qp - win_left) & (kp <= qp + win_right)
        scores = scores.masked_fill(~keep, float("-inf"))

        sink = sink_all[: q.shape[1]].float().view(1, -1, 1)  # [1,H,1]
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_s = torch.exp(scores - scores_max)
        probs = exp_s / (exp_s.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        out[off:end] = torch.einsum("qhk,khd->qhd", probs, packed_k.float()).to(q.dtype)
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

    ref_nc = masked_reference(s, wl, wr)        # intended non-causal 134/6
    ref_ca = masked_reference(s, 127, 0)        # upstream causal 127/0

    def line(tag, a, b):
        r = compare(a, b)
        print(f"[{tag:22}] allclose={str(r['allclose']):5}  maxAbs={r['maxAbs']:.2e}  "
              f"meanAbs={r['meanAbs']:.2e}  meanRel={r['meanRel']:.2e}")

    print()
    line("sanity ref_nc vs entry", ref_nc, ref_entry)   # my non-causal ref should equal the entry's loop
    line("PROD vs ref_noncausal", prod, ref_nc)         # intended window
    line("PROD vs ref_causal127", prod, ref_ca)         # upstream window
    print()

    d_nc = compare(prod, ref_nc)["maxAbs"]
    d_ca = compare(prod, ref_ca)["maxAbs"]
    if d_nc <= d_ca and d_nc < 5e-2:
        print(">>> VERDICT: PROD matches the NON-CAUSAL 134/6 window -> op is correct; the parity gap "
              "is NOT a window bug (look elsewhere: dtype path, cache assembly).")
    elif d_ca < d_nc and d_ca < 5e-2:
        print(">>> VERDICT: PROD matches the CAUSAL 127/0 window, NOT 134/6. CONFIRMED: this build's "
              "SAS op computes the upstream causal window. The fork's non-causal (win_right>0) kernel "
              "patch is missing from this .so. Rebuild vllm-ascend from the dspark-dsv4 commit that "
              "adds win_right support (same one relaxing the oriWinLeft==127/oriWinRight==0 asserts).")
    else:
        print(">>> VERDICT: PROD matches NEITHER window cleanly -> not a pure window-semantics bug "
              "(suspect a numeric/build-corruption issue in the compiled op).")


if __name__ == "__main__":
    main()
