#!/usr/bin/env python3
"""EXPERIMENT (not an acceptance test): does disabling fp32 QK accumulation align the forward
with the production torch-eager(bf16) path?

The kernel keeps fp32 QK accumulation by default (fp32_qk=True), which is MORE precise than
torch eager run in bf16 (its einsum(bf16,bf16) rounds QK to bf16). This runs the kernel BOTH
ways at bf16 and reports the delta vs:
  * production-eager(bf16)  — torch eager at bf16 (the drop-in-replacement target)
  * fp32-truth              — eager on the same bf16 inputs, computed in fp32 (the correct answer)

Reading it: rounding QK to bf16 (fp32_qk=False) moves the MEAN toward eager but leaves the MAX
(other bf16 rounding points: P@V output, exp vs exp2, online vs full softmax) and costs accuracy
vs the truth. Keep fp32_qk=True unless you specifically need to mimic torch-eager-bf16 bitwise
(you can't fully, anyway). Run: python tests/exp_fp32_qk.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import swa_sink_attention, dspark_sas_window, DSV4  # noqa: E402
from triton_impl import swa_sink_attn_fwd  # noqa: E402


def _d(x, ref):
    a = (x.float() - ref.float()).abs()
    return a.max().item(), (a / (ref.abs() + 1e-6)).mean().item()


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    torch.manual_seed(0)
    wl, wr = dspark_sas_window(DSV4["block_size"], DSV4["window_size"])
    for (B, H, L, D) in [(2, 8, 384, 64), (1, DSV4["num_heads"], 256, DSV4["head_dim"])]:
        scale = D ** -0.5
        q = torch.randn(B, H, L, D, device="cuda")
        k = torch.randn(B, H, L, D, device="cuda")
        v = torch.randn(B, H, L, D, device="cuda")
        s = torch.randn(H, device="cuda")
        qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
        prod = swa_sink_attention(qb, kb, vb, s, wl, wr, scale=scale, compute_dtype=torch.float32)
        truth = swa_sink_attention(qb.float(), kb.float(), vb.float(), s, wl, wr,
                                   scale=scale, compute_dtype=torch.float32)
        bm = 16 if D >= 256 else 32
        print(f"\n### B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})")
        for flag in (True, False):
            o = swa_sink_attn_fwd(qb, kb, vb, s, wl, wr, scale=scale,
                                  BLOCK_M=bm, BLOCK_N=bm, fp32_qk=flag)
            mp, rp = _d(o, prod)
            mt, rt = _d(o, truth)
            print(f"  fp32_qk={str(flag):5}  vs prod-eager(bf16): maxAbs={mp:.2e} meanRel={rp:.2e}"
                  f"   |  vs fp32-truth: maxAbs={mt:.2e} meanRel={rt:.2e}")
    print("\n>>> Expect: fp32_qk=False moves meanRel toward prod-eager but not maxAbs, and worsens")
    print(">>> accuracy vs truth. Conclusion: keep fp32_qk=True (default).")


if __name__ == "__main__":
    main()
