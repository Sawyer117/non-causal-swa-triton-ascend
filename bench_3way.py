#!/usr/bin/env python3
"""3-way benchmark at the REAL DSV4 shape (KV=133): eager (bf16 REF) vs the PA_ND fused op (bf16) vs
OUR Triton kernel (bf16). For each: PRECISION (vs an fp32 gold), forward SPEED (ms), peak MEMORY (MB).

- gold        : dense sliding-window + per-head sink attention, computed in FP32 (the precision anchor).
- eager (REF) : the same math in bf16 (materialises the [.,.,H,KV] scores) — the pure-torch baseline.
- PROD PA_ND  : the compiled op npu_sparse_attn_sharedkv via the REAL serve convention (PA_ND paged +
                seqused_kv + ori_block_table), per draft block. (The impl-A TND path NaNs at KV>128 —
                see AUDIT §6.1 — so we use PA_ND, which matches at the bf16 floor.)
- OURS        : our fused Triton kernel.

bf16 vs the fp32 gold cannot beat ~4e-3 maxAbs (bf16 has an 8-bit mantissa AND the output is bf16),
so ~4e-3 maxAbs / ~1e-4 meanAbs / ~1% meanRel is the bf16 FLOOR = "as correct as bf16 gets".

RUN (A3):  python bench_3way.py
           DTYPE=float16 python bench_3way.py     # floor tightens ~8x (proves it's dtype, not logic)
           NBLK=256 python bench_3way.py
"""
import os
import time

import torch

# reuse the scenario + our kernel + the PA_ND prod call from ours_vs_production (its import also loads
# the SAS op and reads H/D/WIN/BS/NBLK/DT from env)
from ours_vs_production import (BS, D, DEV, DT, H, KV, NBLK, SCALE, WIN, build_scenario, ours, prod_paged)

NITER = int(os.environ.get("NITER", "20"))


def dense_attn(qb, kvl, sink, dtype):
    """Dense [ctx|block] attention + per-head sink over the SHARED latent (MLA — no per-head K expand),
    computed in `dtype`. qb [NBLK,BS,H,D], kvl [NBLK,KV,D]."""
    q = qb.to(dtype)                                                # [NBLK,BS,H,D]
    k = kvl.to(dtype)                                               # [NBLK,KV,D] shared across heads
    scores = torch.einsum("nqhd,nkd->nqhk", q, k).to(torch.float32) * SCALE   # [NBLK,BS,H,KV]
    s = sink[:H].float().view(1, 1, H, 1)
    m = torch.maximum(scores.max(dim=-1, keepdim=True).values, s)
    e = torch.exp(scores - m)
    p = (e / (e.sum(dim=-1, keepdim=True) + torch.exp(s - m))).to(dtype)
    o = torch.einsum("nqhk,nkd->nqhd", p, k)                        # [NBLK,BS,H,D]
    return o.reshape(NBLK * BS, H, D).to(qb.dtype)


def cmp(x, gold):
    x, g = x.float(), gold.float(); d = (x - g).abs()
    return d.max().item(), d.mean().item(), (d / (g.abs() + 1e-6)).mean().item()


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
    gold = dense_attn(qb, kvl, sink, torch.float32)                 # fp32 anchor
    torch.npu.synchronize()
    print(f">>> 3-way bench   H={H} D={D} WIN={WIN} block={BS} KV={KV} blocks={NBLK}  dtype={DT}")
    print(f">>> precision vs fp32 gold; bf16 floor ~4e-3 maxAbs / ~1e-4 meanAbs / ~1% meanRel\n")

    runners = {
        "eager (bf16 REF)": lambda: dense_attn(qb, kvl, sink, DT),
        "PROD PA_ND (op)":  lambda: prod_paged(qb, kvl, sink),
        "OURS (Triton)":    lambda: ours(qb, kvl, sink),
    }

    print(f"{'impl':18} | {'maxAbs':>9} {'meanAbs':>9} {'meanRel':>9} | {'fwd ms':>8} | {'peak MB':>8}")
    print("-" * 74)
    for name, fn in runners.items():
        try:
            out = fn(); torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"{name:18} | FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue
        mx, ma, mr = cmp(out, gold)
        t = time_ms(fn)
        mem = peak_mb(fn)
        print(f"{name:18} | {mx:9.2e} {ma:9.2e} {mr:9.2e} | {t:8.3f} | {mem:8.1f}")

    print("\n>>> read: eager/PROD/OURS should all sit at the bf16 floor (precision) — that's 'equivalent'.")
    print(">>> speed/mem: OURS = ONE fused batched kernel call; eager = ONE batched einsum (materialises")
    print("    the [NBLK,BS,H,KV] scores -> more peak MB) -> OURS-vs-eager is the FAIR head-to-head.")
    print(">>> PROD PA_ND time = a per-block PYTHON loop of op calls (dispatch-bound) -> integration")
    print("    overhead, NOT the op's fused batched throughput; don't read it as op speed.")


if __name__ == "__main__":
    main()
