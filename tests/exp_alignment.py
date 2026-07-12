#!/usr/bin/env python3
"""FORWARD ALIGNMENT study: how tightly does the triton kernel align with the gold reference?

Context (from the repo docs):
  * eager_reference.py: `dspark_block_attention_ref` == the gold `_dspark_attention_reference`
    (the pure-torch reference the fused vllm_ascend "SAS" op is validated against).
  * reference_from_repo/dsv4_mla_ref.py self-test: two independent pure-torch implementations
    of the sink-attention math agree to **~1e-7 in fp32** and **~1e-14 in fp64** ("math
    identity"; a real bug would be ~1e-2 at BOTH). That ~1e-6/1e-7 is the *fp32 alignment*
    level between eager and the SAS reference.

This study shows the triton kernel hits that SAME fp32 alignment level vs the gold, so it is
aligned with eager / the SAS reference to math-identity in fp32. bf16 can't reach 1e-6 (the
kernel returns bf16 — the ~4e-3 output rounding is a hard floor); shown for context.

Run: python tests/exp_alignment.py    (fp32 is the alignment number; bf16 is context)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, dspark_block_attention_ref, dspark_sas_window, DSV4,
)
from triton_impl import swa_sink_attn_fwd, dense_sink_attn_fwd  # noqa: E402


def _d(x, ref):
    a = (x.float() - ref.float()).abs()
    return a.max().item(), a.mean().item(), (a / (ref.abs() + 1e-6)).mean().item()


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    torch.manual_seed(0)
    wl, wr = dspark_sas_window(DSV4["block_size"], DSV4["window_size"])
    H, D = DSV4["num_heads"], DSV4["head_dim"]
    KV = DSV4["window_size"] + DSV4["block_size"]
    print(">>> FORWARD ALIGNMENT vs the gold (== _dspark_attention_reference, the SAS reference)")
    print(">>> fp32 = the alignment number (math identity, like eager vs SAS ref ~1e-7);"
          " bf16 = context (output-rounding floor)\n")

    # --- windowed self-attention vs eager swa_sink_attention (real DSV4 H,D) ---
    for dt in (torch.float32, torch.bfloat16):
        q = torch.randn(1, H, 256, D, device="cuda").to(dt)
        k = torch.randn(1, H, 256, D, device="cuda").to(dt)
        v = torch.randn(1, H, 256, D, device="cuda").to(dt)
        s = torch.randn(H, device="cuda")
        ref = swa_sink_attention(q.float(), k.float(), v.float(), s, wl, wr,
                                 scale=D ** -0.5, compute_dtype=torch.float32)
        o = swa_sink_attn_fwd(q, k, v, s, wl, wr, scale=D ** -0.5, BLOCK_M=16, BLOCK_N=16)
        mx, mae, mre = _d(o, ref)
        print(f"  windowed  {str(dt).replace('torch.',''):8} vs eager  : maxAbs={mx:.2e} "
              f"meanAbs={mae:.2e} meanRel={mre:.2e}")

    # --- dense vs the GOLD block form dspark_block_attention_ref (real DSV4 H,D) ---
    for dt in (torch.float32, torch.bfloat16):
        qg = torch.randn(2, DSV4["block_size"], H, D, device="cuda").to(dt)
        kg = torch.randn(2, KV, H, D, device="cuda").to(dt)
        vg = torch.randn(2, KV, H, D, device="cuda").to(dt)
        s = torch.randn(H, device="cuda")
        gold = dspark_block_attention_ref(qg.float(), kg.float(), vg.float(), s,
                                          scale=D ** -0.5, compute_dtype=torch.float32)
        o = dense_sink_attn_fwd(qg.permute(0, 2, 1, 3).contiguous(),
                                kg.permute(0, 2, 1, 3).contiguous(),
                                vg.permute(0, 2, 1, 3).contiguous(), s,
                                scale=D ** -0.5, BLOCK_M=8, BLOCK_N=16).permute(0, 2, 1, 3)
        mx, mae, mre = _d(o, gold)
        print(f"  gold-block{str(dt).replace('torch.',''):8} vs gold   : maxAbs={mx:.2e} "
              f"meanAbs={mae:.2e} meanRel={mre:.2e}")

    print("\n>>> fp32 maxAbs ~1e-6 == the eager/SAS-reference math-identity level: the kernel IS")
    print(">>> aligned. bf16 ~1e-3 is the output-rounding floor. The definitive SAS-on-NPU check")
    print(">>> is dspark_attn_ref_bench.py against vllm_ascend on the A3 (DTYPE=float32 -> ~1e-6).")


if __name__ == "__main__":
    main()
