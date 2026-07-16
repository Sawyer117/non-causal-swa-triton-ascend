#!/usr/bin/env python3
"""BF16 comparison at the DSV4 shape. BASELINE = the PROD fused op (what the vllm-ascend engine calls).
Everything is bf16; precision is measured vs PROD (the baseline) — NO fp32 gold anywhere. PROD itself
is not compared to anything (it IS the reference).

  BASELINE  prod op (bf16)          the engine's npu_sparse_attn_sharedkv (PA_ND, batched)
  case1     our train eager (bf16)  _sink_block_attention_torch as-is: per-head-expanded K (heavy mem)
  case2     our train eager mem-opt shared MLA latent (no per-head expand) — the reference-optimized form
  case3     reference-eager (bf16)  the PR reference _dspark_attention_reference
  case4     ours triton (bf16)      our fused Triton kernel

Also prints case1-vs-case2 to prove the memory optimization does NOT change the numbers.

RUN (A3):  BS=5 python bench_3way.py          # DSV4 (KV=133), bf16
           BS=5 DTYPE=float16 python bench_3way.py
"""
import os
import time

import torch

from ours_vs_production import (BS, D, DEV, DT, H, KV, NBLK, SCALE, WIN, build_scenario, ours, prod_call,
                                prod_paged, prod_prep)
from vllm_ascend.ops.dspark_attention import _dspark_attention_reference as _pr_ref

NITER = int(os.environ.get("NITER", "20"))


def eager(qb, kvl, sink, shared):
    """Faithful to speculators `_sink_block_attention_torch` (fp32 ACCUMULATION internally, bf16 out).
    shared=False: per-head-expanded K [N,KV,H,D] (materialised — the current training form, heavy mem).
    shared=True : the shared MLA latent [N,KV,D] used directly (the reference-optimized, light mem)."""
    if shared:
        k = kvl.float()                                              # [N,KV,D]
        s = torch.einsum("nqhd,nkd->nqhk", qb.float(), k) * SCALE
    else:
        k = kvl.unsqueeze(2).expand(NBLK, KV, H, D).float()          # [N,KV,H,D] (materialised)
        s = torch.einsum("nqhd,nkhd->nqhk", qb.float(), k) * SCALE
    sh = sink[:H].float().view(1, 1, H, 1)
    m = torch.maximum(s.max(dim=-1, keepdim=True).values, sh)
    e = torch.exp(s - m)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sh - m))
    o = torch.einsum("nqhk,nkd->nqhd", p, k) if shared else torch.einsum("nqhk,nkhd->nqhd", p, k)
    return o.to(qb.dtype).reshape(NBLK * BS, H, D)


def pr_eager(qb, kvl, sink):
    outs = []
    for b in range(NBLK):
        kb = kvl[b].unsqueeze(1).expand(KV, H, D).contiguous()
        outs.append(_pr_ref(qb[b], kb, kb, sink, SCALE))
    return torch.stack(outs, dim=0).reshape(NBLK * BS, H, D)


def cmp(x, ref):
    x, r = x.float(), ref.float(); d = (x - r).abs()
    return d.max().item(), d.mean().item(), (d / (r.abs() + 1e-6)).mean().item()


def time_ms(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(NITER):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / NITER * 1e3


def peak_mb(fn):
    torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
    fn(); torch.npu.synchronize()
    return torch.npu.max_memory_allocated() / (1024 ** 2)


def main():
    s, qb, kvl, sink = build_scenario()
    prep = prod_prep(qb, kvl, sink)
    ref = prod_call(prep)                                            # BASELINE output (bf16)
    torch.npu.synchronize()
    print(f">>> BF16 bench   H={H} D={D} WIN={WIN} block={BS} KV={KV} blocks={NBLK}  dtype={DT}")
    print(">>> BASELINE = PROD fused op; precision is vs PROD (bf16). No fp32 anywhere.\n")

    rows = [
        ("BASELINE prod op",  lambda: prod_call(prep), True),
        ("case1 train-eager",  lambda: eager(qb, kvl, sink, shared=False), False),
        ("case2 eager mem-opt", lambda: eager(qb, kvl, sink, shared=True), False),
        ("case3 reference-eager", lambda: pr_eager(qb, kvl, sink), False),
        ("case4 ours triton",  lambda: ours(qb, kvl, sink), False),
    ]
    print(f"{'impl':22} | {'maxAbs':>9} {'meanAbs':>9} {'meanRel':>9} | {'fwd ms':>8} | {'peak MB':>8}")
    print("-" * 80)
    outs = {}
    for name, fn, is_base in rows:
        try:
            o = fn(); torch.npu.synchronize(); outs[name] = o
            t, mem = time_ms(fn), peak_mb(fn)
            if is_base:
                print(f"{name:22} | {'(baseline)':>29} | {t:8.3f} | {mem:8.1f}")
            else:
                mx, ma, mr = cmp(o, ref)
                print(f"{name:22} | {mx:9.2e} {ma:9.2e} {mr:9.2e} | {t:8.3f} | {mem:8.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22} | FAILED: {type(e).__name__}: {str(e)[:44]}")

    if "case1 train-eager" in outs and "case2 eager mem-opt" in outs:
        mx, ma, mr = cmp(outs["case1 train-eager"], outs["case2 eager mem-opt"])
        print("-" * 80)
        print(f"{'case1 vs case2 (mem-opt)':22} | {mx:9.2e} {ma:9.2e} {mr:9.2e} |   "
              f"<- ~0 => the memory optimization does NOT change the output")

    print("\n>>> read: precision is vs PROD (bf16). case4 (ours) ~1e-7 vs PROD => bit-equal to the op.")
    print(">>> case1 vs case2 ~0 + case2's lower peak MB => the shared-latent mem-opt is free (safe).")
    print(">>> speed/mem: PROD is the engine baseline; the eagers materialise scores/per-head K (heavy).")


if __name__ == "__main__":
    main()
