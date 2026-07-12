#!/usr/bin/env python3
"""DSpark draft attention benched against the vLLM-Ascend GOLD reference (not a hand-written mask).

The gold standard is `_dspark_attention_reference` in vllm_ascend/ops/dspark_attention.py — the
pure-torch reference the fused SAS op is validated against (the SAS-vs-PTA parity tests). It is
NOT a symmetric sliding-window mask. Per draft block, the block's queries attend DENSELY to
[the last `window_size` context tokens] + [the FULL current draft block] (non-causal within the
block), with an ATTENTION SINK in the softmax denominator. The real window is asymmetric:
win_left = window_size + block_size - 1, win_right = block_size - 1 (vllm_ascend even ships a test
rejecting the plain window_size-1 formula). An earlier hand-written bench got all three wrong; this
imports the REAL reference so we compare against ground truth.

Checks: (1) the reference runs FORWARD+BACKWARD on NPU (=> the training attention math is supported);
(2) candidate training impls reproduce it — torch.allclose (ATOL/RTOL), per-tensor mean abs / mean
rel error vs the gold, and forward / forward+backward speedup vs the gold. NB: vanilla SDPA has no
attention-sink, so "sdpa_nosink" is a speed reference that OMITS the sink (its diff shows the sink's
effect); the sink-correct training path is the einsum reference form (all standard NPU ops).

Run in dspark-dsv4-base on ONE NPU that has vllm_ascend installed (the A3):
  python dspark_attn_ref_bench.py
  DTYPE=float32 python dspark_attn_ref_bench.py     # diffs collapse to ~1e-6 -> dtype, not math
"""
import os
import time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    print(f"!! torch_npu import failed: {e}")
    raise SystemExit(1)

try:
    from vllm_ascend.ops.dspark_attention import (  # the GOLD reference + real window
        _dspark_attention_reference,
        _dspark_sas_window,
    )
except Exception as e:  # noqa: BLE001
    print(f"!! cannot import the vllm_ascend gold reference: {e}")
    print("   run this on a node with vllm_ascend installed (the A3), env dspark-dsv4-base.")
    raise SystemExit(1)

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
# DeepSeek-V4-Flash attention params — HARDCODED from the HF config.json (fixed model constants):
#   num_attention_heads=64, head_dim=512, num_key_value_heads=1 (MLA), sliding_window=128,
#   hidden_size=4096, 43 target layers.  DSpark draft: block_size=7 (block7 ckpt).
# NB: num_key_value_heads=1 (MLA) => real K/V are 1-head/tiny; this bench uses per-Q-head K/V so its
#   memory is a CONSERVATIVE OVER-estimate. Still env-overridable (NBLK/BS/WIN/H/D) for scans.
NBLK = int(os.environ.get("NBLK", "64"))       # draft blocks (~num_anchors; set NBLK=512 for full scale)
BS = int(os.environ.get("BS", "7"))            # DSpark block_size
WIN = int(os.environ.get("WIN", "128"))        # DSV4 sliding_window
Hh = int(os.environ.get("H", "64"))            # num_attention_heads
Dh = int(os.environ.get("D", "512"))           # head_dim
CTX = WIN
KV = CTX + BS
SCALE = Dh ** -0.5
NITER = 20

mm, wl, wr = _dspark_sas_window(BS, WIN)
print(f">>> _dspark_sas_window(block={BS}, window={WIN}) = mask_mode={mm}, win_left={wl}, win_right={wr}")
print(f">>> (real DSpark draft window: left={WIN}+{BS}-1={WIN + BS - 1}, right={BS}-1={BS - 1})")
print(f">>> scenario: {NBLK} blocks, q[{BS},{Hh},{Dh}] attends dense to k_ctx[{KV},{Hh},{Dh}] (ctx {CTX}+draft {BS}) + sink")
print(f">>> dtype={DT} allclose atol={ATOL} rtol={RTOL}\n")

Q0 = torch.randn(NBLK, BS, Hh, Dh, device=DEV, dtype=DT)
K0 = torch.randn(NBLK, KV, Hh, Dh, device=DEV, dtype=DT)
V0 = torch.randn(NBLK, KV, Hh, Dh, device=DEV, dtype=DT)
SINK = torch.randn(Hh, device=DEV, dtype=DT)


def fresh():
    return (Q0.clone().requires_grad_(True),
            K0.clone().requires_grad_(True),
            V0.clone().requires_grad_(True))


def attn_gold(q, k, v):
    # the imported vllm_ascend reference, per block (q[i] [BS,H,D], k[i] [KV,H,D])
    return torch.stack([_dspark_attention_reference(q[i], k[i], v[i], SINK, SCALE)
                        for i in range(q.shape[0])], dim=0)


def attn_manual(q, k, v):
    # batched re-derivation of the SAME math (einsum + sink) -> the training-side impl
    s = torch.einsum("nqhd,nkhd->nqhk", q.float(), k.float()) * SCALE
    sink = SINK.float().view(1, 1, Hh, 1)
    smax = torch.maximum(s.max(dim=-1, keepdim=True).values, sink)
    e = torch.exp(s - smax)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sink - smax))
    return torch.einsum("nqhk,nkhd->nqhd", p, v.float()).to(DT)


def attn_sdpa_nosink(q, k, v):
    # SDPA over [ctx+draft] with NO sink (dense, non-causal). Omits the sink on purpose.
    qb = q.transpose(1, 2)   # [N,H,BS,D]
    kb = k.transpose(1, 2)   # [N,H,KV,D]
    vb = v.transpose(1, 2)
    o = F.scaled_dot_product_attention(qb, kb, vb, scale=SCALE)
    return o.transpose(1, 2)


def time_ms(step):
    for _ in range(3):
        step()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(NITER):
        step()
    torch.npu.synchronize()
    return (time.time() - t0) / NITER * 1e3


def compare(x, ref):
    xf = x.detach().float().cpu()   # ref is kept on CPU to free NPU memory between candidates
    d = (xf - ref).abs()
    return (torch.allclose(xf, ref, atol=ATOL, rtol=RTOL), d.mean().item(),
            (d / (ref.abs() + 1e-6)).mean().item())


REF = {}


def bench(name, fn):
    torch.npu.empty_cache()
    try:
        torch.npu.reset_peak_memory_stats()
        q, k, v = fresh()
        out = fn(q, k, v)
        out.float().sum().backward()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() / 1e6  # MB, fwd+bwd peak
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<12} FAILED (fwd/bwd): {type(e).__name__}: {str(e)[:55]}")
        torch.npu.empty_cache()
        return
    outf = out.detach().float().cpu()
    gqf = q.grad.detach().float().cpu()
    del q, k, v, out
    torch.npu.empty_cache()
    fwd = time_ms(lambda: _fwd(fn))
    fb = time_ms(lambda: _fb(fn))
    if not REF:
        REF.update(out=outf, gq=gqf, fwd=fwd, fb=fb)   # kept on CPU
        print(f"  {name:<12} fwd={fwd:6.3f}ms fwd+bwd={fb:6.3f}ms  peak={peak:7.1f}MB  speedup 1.00x/1.00x  (GOLD)")
        torch.npu.empty_cache()
        return
    oc, oae, ore = compare(outf, REF["out"])
    gc, gae, gre = compare(gqf, REF["gq"])
    print(f"  {name:<12} fwd={fwd:6.3f}ms fwd+bwd={fb:6.3f}ms  peak={peak:7.1f}MB  speedup {REF['fwd']/fwd:4.2f}x/{REF['fb']/fb:4.2f}x")
    print(f"               out : allclose={str(oc):<5} meanAbs={oae:.2e} meanRel={ore:.2e}")
    print(f"               grad: allclose={str(gc):<5} meanAbs={gae:.2e} meanRel={gre:.2e}")
    torch.npu.empty_cache()


def _fwd(fn):
    with torch.no_grad():
        fn(*fresh())


def _fb(fn):
    q, k, v = fresh()
    fn(q, k, v).float().sum().backward()


bench("gold(ref)", attn_gold)         # imported vllm_ascend reference = ground truth
bench("manual", attn_manual)          # batched einsum+sink -> the training impl (should match gold)
bench("sdpa_nosink", attn_sdpa_nosink)  # SDPA, NO sink -> speed ref; its diff = the sink's effect

print("\n>>> read: manual should be allclose=True to gold (validates the math + that it runs fwd/bwd on NPU).")
print(">>>       sdpa_nosink's diff shows how much the attention SINK matters (vanilla SDPA can't do it).")
print(">>>       => training uses the einsum+sink form (standard NPU ops); it's the sink-correct path.")
