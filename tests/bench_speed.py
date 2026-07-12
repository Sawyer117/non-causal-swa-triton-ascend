#!/usr/bin/env python3
"""SPEED + MEMORY benchmark: the Triton kernel vs the EAGER reference (the real baseline).

We bench against eager (einsum+sink), NOT dense-mask SDPA: SDPA can't express the attention
sink, so it's not a usable training path — eager is the only correct fallback in production
today, so the meaningful win is triton-vs-eager. Eager materializes the [B,H,L,L] scores +
[L,L] mask (O(L^2) memory); the kernel tests the window predicate on the fly (O(L*window)),
so at large L eager OOMs while the kernel keeps running — that gap is the point.

Reports, per config: forward and forward+backward time (ms), speedup (eager/triton), and the
fwd+bwd peak memory (MB) for each. Default dtype bf16 (the training dtype).

Run: python tests/bench_speed.py
     DTYPE=float32 python tests/bench_speed.py
     LS=256,512,1024 python tests/bench_speed.py     # windowed L sweep
"""
import os
import time

import torch

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, dspark_block_attention_ref, dspark_sas_window, DSV4,
)
from triton_impl import swa_sink_attn, dense_sink_attn  # noqa: E402

DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
LS = [int(x) for x in os.environ.get("LS", "256,512,1024").split(",")]
ITERS, WARMUP = 20, 5


def _timed(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1e3   # ms


def _peak(step):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6   # MB


def _measure(name, fwd, fb):
    """Returns (fwd_ms, fb_ms, fwd_peak_MB, fb_peak_MB) or None on OOM/failure."""
    try:
        fb()   # smoke (also compiles the kernel)
        torch.cuda.synchronize()
        return _timed(fwd), _timed(fb), _peak(fwd), _peak(fb)
    except RuntimeError as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        msg = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
        print(f"    {name:8} FAILED: {msg}")
        return None


def _report(tri, eag):
    if tri is None:
        return
    tf, tb, tmf, tmb = tri
    if eag is None:
        print(f"    triton   fwd={tf:7.3f}ms  fwd+bwd={tb:7.3f}ms  peak fwd={tmf:8.1f}MB "
              f"fwd+bwd={tmb:8.1f}MB   (eager OOM/failed -> kernel runs where eager can't)")
        return
    ef, eb, emf, emb = eag
    print(f"    eager    fwd={ef:7.3f}ms  fwd+bwd={eb:7.3f}ms   peak fwd={emf:8.1f}MB  fwd+bwd={emb:8.1f}MB")
    print(f"    triton   fwd={tf:7.3f}ms  fwd+bwd={tb:7.3f}ms   peak fwd={tmf:8.1f}MB  fwd+bwd={tmb:8.1f}MB")
    print(f"    -> speedup {ef/tf:4.2f}x / {eb/tb:4.2f}x   mem  fwd {emf/tmf:4.2f}x  fwd+bwd {emb/tmb:4.2f}x")


def bench_windowed(B, H, L, D, wl, wr):
    scale = D ** -0.5
    q0 = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    k0 = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    v0 = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    sink = torch.randn(H, device="cuda", dtype=torch.float32)

    def fresh():
        return (q0.clone().requires_grad_(True), k0.clone().requires_grad_(True),
                v0.clone().requires_grad_(True))

    def tri_fwd():
        with torch.no_grad():
            swa_sink_attn(q0, k0, v0, sink, wl, wr, scale=scale)

    def tri_fb():
        q, k, v = fresh()
        swa_sink_attn(q, k, v, sink, wl, wr, scale=scale).float().sum().backward()

    def eag_fwd():
        with torch.no_grad():
            swa_sink_attention(q0, k0, v0, sink, wl, wr, scale=scale, compute_dtype=torch.float32)

    def eag_fb():
        q, k, v = fresh()
        swa_sink_attention(q, k, v, sink, wl, wr, scale=scale,
                           compute_dtype=torch.float32).float().sum().backward()

    print(f"\n### windowed  B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  dtype={DT}")
    _report(_measure("triton", tri_fwd, tri_fb), _measure("eager", eag_fwd, eag_fb))


def bench_dense(N, BS, KV, H, D, mla=False):
    """Dense block form. mla=True -> ONE shared KV latent (num_kv_heads=1, the real model);
    mla=False -> per-Q-head KV (MHA, the bench's conservative over-estimate). Returns (tri, eag)."""
    scale = D ** -0.5
    qg = torch.randn(N, BS, H, D, device="cuda", dtype=DT)      # gold q [N,BS,H,D]
    sink = torch.randn(H, device="cuda", dtype=torch.float32)
    qk = qg.permute(0, 2, 1, 3).contiguous()                   # kernel q [N,H,BS,D]
    if mla:
        kL = torch.randn(N, KV, D, device="cuda", dtype=DT)    # [N,KV,D] one latent head
        vL = torch.randn(N, KV, D, device="cuda", dtype=DT)

        def tri_fwd():
            with torch.no_grad():
                dense_sink_attn(qk, kL, vL, sink, scale=scale, BLOCK_M=8, BLOCK_N=16)

        def tri_fb():
            q = qk.clone().requires_grad_(True); k = kL.clone().requires_grad_(True); v = vL.clone().requires_grad_(True)
            dense_sink_attn(q, k, v, sink, scale=scale, BLOCK_M=8, BLOCK_N=16).float().sum().backward()

        def eag_fwd():
            with torch.no_grad():
                dspark_block_attention_ref(qg, kL.unsqueeze(2).expand(N, KV, H, D),
                                           vL.unsqueeze(2).expand(N, KV, H, D), sink,
                                           scale=scale, compute_dtype=torch.float32)

        def eag_fb():
            q = qg.clone().requires_grad_(True); k = kL.clone().requires_grad_(True); v = vL.clone().requires_grad_(True)
            dspark_block_attention_ref(q, k.unsqueeze(2).expand(N, KV, H, D),
                                       v.unsqueeze(2).expand(N, KV, H, D), sink,
                                       scale=scale, compute_dtype=torch.float32).float().sum().backward()
    else:
        kg = torch.randn(N, KV, H, D, device="cuda", dtype=DT)
        vg = torch.randn(N, KV, H, D, device="cuda", dtype=DT)
        kk = kg.permute(0, 2, 1, 3).contiguous(); vk = vg.permute(0, 2, 1, 3).contiguous()

        def tri_fwd():
            with torch.no_grad():
                dense_sink_attn(qk, kk, vk, sink, scale=scale, BLOCK_M=8, BLOCK_N=16)

        def tri_fb():
            q = qk.clone().requires_grad_(True); k = kk.clone().requires_grad_(True); v = vk.clone().requires_grad_(True)
            dense_sink_attn(q, k, v, sink, scale=scale, BLOCK_M=8, BLOCK_N=16).float().sum().backward()

        def eag_fwd():
            with torch.no_grad():
                dspark_block_attention_ref(qg, kg, vg, sink, scale=scale, compute_dtype=torch.float32)

        def eag_fb():
            q = qg.clone().requires_grad_(True); k = kg.clone().requires_grad_(True); v = vg.clone().requires_grad_(True)
            dspark_block_attention_ref(q, k, v, sink, scale=scale,
                                       compute_dtype=torch.float32).float().sum().backward()

    print(f"\n### dense/gold  N={N} BS={BS} KV={KV} H={H} D={D}  {'MLA (num_kv_heads=1)' if mla else 'MHA'}  dtype={DT}")
    tri, eag = _measure("triton", tri_fwd, tri_fb), _measure("eager", eag_fwd, eag_fb)
    _report(tri, eag)
    return tri, eag


def _prod_summary(N, BS, KV, H, D, prod):
    tri, eag = prod
    line = "=" * 74
    print("\n" + line)
    print(f"PRODUCTION SUMMARY  (dtype={str(DT).replace('torch.','')}, real DSV4-Flash-DSpark "
          f"BLOCK shape, MLA num_kv_heads=1)")
    print(f"  {N} draft blocks:  q[BS={BS}, H={H}, D={D}]  attends  kv[KV={KV}, D={D} shared latent]  + per-head sink")
    if tri is None:
        print("  triton FAILED to run — see error above.")
    elif eag is None:
        tf, tb, tmf, tmb = tri
        print(f"  triton fwd={tf:.3f}ms fwd+bwd={tb:.3f}ms  peak fwd={tmf:.0f}MB fwd+bwd={tmb:.0f}MB")
        print("  eager OOM/failed at this size -> the kernel runs where eager can't.")
    else:
        tf, tb, tmf, tmb = tri
        ef, eb, emf, emb = eag
        print(f"  forward      : {ef / tf:5.2f}x faster    peak {emf / tmf:4.2f}x smaller"
              f"   (triton {tmf:6.0f}MB  vs  eager {emf:6.0f}MB)")
        print(f"  forward+bwd  : {eb / tb:5.2f}x faster    peak {emb / tmb:4.2f}x smaller"
              f"   (triton {tmb:6.0f}MB  vs  eager {emb:6.0f}MB)")
    print("  baseline = eager einsum+sink (the only correct production path; SDPA has no sink)")
    print(line)


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    print(">>> SPEED+MEMORY  triton kernel vs EAGER (einsum+sink)  — SDPA excluded (no sink)")
    print(">>> speedup = eager_time / triton_time; mem = eager_peak / triton_peak")
    H, D = DSV4["num_heads"], DSV4["head_dim"]      # 64, 512
    BS, KV = DSV4["block_size"], DSV4["window_size"] + DSV4["block_size"]   # 7, 135
    wl, wr = dspark_sas_window(DSV4["block_size"], DSV4["window_size"])     # (134, 6)
    NBLK = int(os.environ.get("NBLK", "64"))        # draft blocks (~num anchors); the real scale
    for L in LS:                                     # packed-SWA view (context)
        bench_windowed(1, H, L, D, wl, wr)
    bench_dense(8, BS, KV, H, D, mla=False)          # block form, MHA over-estimate (context)
    prod = bench_dense(NBLK, BS, KV, H, D, mla=True)  # block form, MLA = the PRODUCTION case
    _prod_summary(NBLK, BS, KV, H, D, prod)


if __name__ == "__main__":
    main()
