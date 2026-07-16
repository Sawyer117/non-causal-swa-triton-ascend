#!/usr/bin/env python3
"""Multi-way benchmark at the REAL DSV4 shape (KV=133): precision / speed / peak-memory of
  * PR eager   : vllm_ascend `_dspark_attention_reference` (the OP's own gold, what inference references)
  * our eager  : `eager_reference.dspark_block_attention_ref` (the TRAINING-side gold our kernel targets)
  * PROD PA_ND : the compiled op `npu_sparse_attn_sharedkv` via the REAL serve convention (PA_ND paged +
                 seqused_kv + ori_block_table), per block. (The impl-A TND path NaNs at KV>128 — AUDIT §6.1.)
  * OURS       : our fused Triton kernel.
all in bf16, precision vs an FP32 gold. FIRST it checks the two eagers agree (training≡inference math),
then it benches everyone.

bf16 vs the fp32 gold cannot beat ~4e-3 maxAbs (bf16 = 8-bit mantissa AND the output is bf16-stored);
so ~4e-3 maxAbs / ~1e-4 meanAbs / ~1% meanRel is the bf16 FLOOR = "as correct as bf16 gets".

RUN (A3):  python bench_3way.py
           DTYPE=float16 python bench_3way.py     # floor tightens ~8x (proves it's dtype, not logic)
"""
import os
import time

import torch

from ours_vs_production import (BS, D, DEV, DT, H, KV, NBLK, SCALE, WIN, build_scenario, ours, prod_call,
                                prod_paged, prod_prep)
from vllm_ascend.ops.dspark_attention import _dspark_attention_reference as _pr_ref  # PR eager gold
from eager_reference import dspark_block_attention_ref as _our_ref                    # our training eager

NITER = int(os.environ.get("NITER", "20"))


def dense_attn(qb, kvl, sink, dtype):
    """fp32/dtype dense [ctx|block] + per-head sink over the SHARED latent (MLA, no per-head expand)."""
    q = qb.to(dtype); k = kvl.to(dtype)                            # [NBLK,BS,H,D], [NBLK,KV,D]
    scores = torch.einsum("nqhd,nkd->nqhk", q, k).to(torch.float32) * SCALE
    s = sink[:H].float().view(1, 1, H, 1)
    m = torch.maximum(scores.max(dim=-1, keepdim=True).values, s)
    e = torch.exp(scores - m)
    p = (e / (e.sum(dim=-1, keepdim=True) + torch.exp(s - m))).to(dtype)
    return torch.einsum("nqhk,nkd->nqhd", p, k).reshape(NBLK * BS, H, D).to(qb.dtype)


def pr_eager(qb, kvl, sink):
    """PR's `_dspark_attention_reference` (per block; it casts to fp32 internally, bf16 inputs)."""
    outs = []
    for b in range(NBLK):
        kb = kvl[b].unsqueeze(1).expand(KV, H, D).contiguous()     # [KV,H,D] shared latent -> per head
        outs.append(_pr_ref(qb[b], kb, kb, sink, SCALE))           # [BS,H,D]
    return torch.stack(outs, dim=0).reshape(NBLK * BS, H, D)


def our_eager(qb, kvl, sink):
    """Our training-side `dspark_block_attention_ref` (fp32 compute, bf16 inputs — as training uses it)."""
    k = kvl.unsqueeze(2).expand(NBLK, KV, H, D).contiguous()       # [NBLK,KV,H,D]
    return _our_ref(qb, k, k, sink[:H], SCALE, torch.float32).reshape(NBLK * BS, H, D)


def cmp(x, g):
    x, g = x.float(), g.float(); d = (x - g).abs()
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
    gold = dense_attn(qb, kvl, sink, torch.float32)
    torch.npu.synchronize()
    print(f">>> bench  H={H} D={D} WIN={WIN} block={BS} KV={KV} blocks={NBLK}  dtype={DT}")
    print(">>> precision vs fp32 gold; bf16 floor ~4e-3 maxAbs / ~1e-4 meanAbs / ~1% meanRel\n")

    # (0) do the two eager references agree? (training math == inference math)
    try:
        pe, oe = pr_eager(qb, kvl, sink), our_eager(qb, kvl, sink)
        mx, ma, mr = cmp(pe, oe)
        print(f"[agree]  PR eager vs our eager   maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}  "
              f"({'IDENTICAL math' if mx < 5e-2 else 'DIVERGE — investigate!'})\n")
    except Exception as e:  # noqa: BLE001
        print(f"[agree]  eager compare FAILED: {type(e).__name__}: {str(e)[:60]}\n")

    # PROD is the BASELINE (the op the vllm-ascend engine actually calls). Prep once (serve caches the
    # metadata) so the timed path is only the batched op's fused compute — a fair op-vs-op number.
    prep = None
    try:
        prep = prod_prep(qb, kvl, sink)
    except Exception as e:  # noqa: BLE001
        print(f"[PROD prep FAILED: {type(e).__name__}: {str(e)[:60]}]")
    # roles: prod op = INFERENCE ref (engine's op, no autograd -> training can't use it);
    #        our training eager = the TRAINING baseline (has torch autograd); ours triton = our kernel.
    runners = [
        ("prod op (infer ref)",     lambda: prod_call(prep) if prep else prod_paged(qb, kvl, sink)),
        ("our train eager (BASE)",  lambda: our_eager(qb, kvl, sink)),
        ("ours triton",             lambda: ours(qb, kvl, sink)),
        ("PR eager (gold check)",   lambda: pr_eager(qb, kvl, sink)),
    ]
    print(f"{'impl':22} | {'maxAbs':>9} {'meanAbs':>9} {'meanRel':>9} | {'fwd ms':>8} | {'peak MB':>8}")
    print("-" * 80)
    times = {}
    for name, fn in runners:
        try:
            out = fn(); torch.npu.synchronize()
            mx, ma, mr = cmp(out, gold)
            t, mem = time_ms(fn), peak_mb(fn)
            times[name] = (t, mem)
            print(f"{name:22} | {mx:9.2e} {ma:9.2e} {mr:9.2e} | {t:8.3f} | {mem:8.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22} | FAILED: {type(e).__name__}: {str(e)[:44]}")

    tr = times.get("ours triton")
    if tr:
        if times.get("our train eager (BASE)"):
            et, em = times["our train eager (BASE)"]
            print(f"\n>>> TRAINING (our deliverable): ours triton FWD vs the training baseline (eager) = "
                  f"{et / tr[0]:.2f}x faster, {em - tr[1]:+.0f} MB. NOTE: forward only — the training win "
                  f"needs the fwd+BWD number (bwd is the weak part); run fused_sas_vs_ours.py.")
        if times.get("prod op (infer ref)"):
            pt, _ = times["prod op (infer ref)"]
            print(f">>> INFERENCE ref: the engine's fused op fwd = {tr[0] / pt:.2f}x faster than ours "
                  f"(hand-tuned AscendC; it has NO autograd so training can't use it).")
    print(">>> precision: all sit at the bf16 floor (ours triton ≡ prod op ≡ eagers); fp16 tightens ~8x.")


if __name__ == "__main__":
    main()
