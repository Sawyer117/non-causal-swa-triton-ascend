#!/usr/bin/env python3
"""GPU speed+memory of the ASCEND-shaped op vs eager (a sanity read before the A3).

IMPORTANT caveat: the Ascend op caps its grid to the core count (on GPU = SM count) because that
is the correct grid on the NPU (grid <= cube cores). On a GPU that ARTIFICIALLY limits
parallelism, so the "capped" numbers UNDER-state GPU throughput. We also report a "full" grid
(num_programs = NUM_TILES) which uses the whole GPU and reflects the kernel's own compute
efficiency (what transfers to per-core NPU efficiency):
  * full   -> is the kernel itself efficient? (GPU-fair)
  * capped -> the real NPU grid shape (pessimistic on GPU, correct on the A3)

Windowed packed-SWA at real DSV4 H=64/D=512, vs eager (einsum+sink; SDPA excluded, no sink).
Run: python tests/bench_ascend.py    (DTYPE=float32, LS=256,512,1024, NBLK env knobs)
"""
import os
import sys
import time

import torch
import triton

try:  # Ascend: maps torch.cuda.* -> NPU so this bench runs on the A3 too (vllm PR #775)
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:  # noqa: BLE001
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import (  # noqa: E402
    swa_sink_attention, dspark_block_attention_ref, dspark_sas_window, DSV4,
)
from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend, _default_blocks  # noqa: E402
from triton_impl.swa_sink_ascend_bwd import swa_sink_bwd_ascend  # noqa: E402

DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
LS = [int(x) for x in os.environ.get("LS", "256,512,1024").split(",")]
BM_ENV = int(os.environ["BM"]) if "BM" in os.environ else None   # override tile sizes to experiment
BN_ENV = int(os.environ["BN"]) if "BN" in os.environ else None
HG_ENV = int(os.environ["HG"]) if "HG" in os.environ else None   # head-batched MLA-dense fwd (M=HG*BS)
ITERS, WARMUP = 20, 5


def _timed(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1e3


def _peak(step):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def _prec(o, ref):
    """maxAbs / meanRel of an ascend output vs the fp32 reference on the same inputs."""
    d = (o.float() - ref.float()).abs()
    return d.max().item(), (d / (ref.abs() + 1e-6)).mean().item()


def _run(fwd, fb):
    try:
        fb(); torch.cuda.synchronize()
        return _timed(fwd), _timed(fb), _peak(fb)
    except RuntimeError as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        print(f"      FAILED: {'OOM' if 'out of memory' in str(e).lower() else type(e).__name__}")
        return None


def bench_windowed(B, H, L, D, wl, wr):
    scale = D ** -0.5
    q = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    k = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    v = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    do = torch.randn(B, H, L, D, device="cuda", dtype=DT)
    sink = torch.randn(H, device="cuda", dtype=torch.float32)
    bm, bn = _default_blocks(D, BM_ENV, BN_ENV)
    nt = triton.cdiv(L, min(bm, bn)) * B * H          # >= NUM_TILES -> full grid on GPU

    def eager_fwd():
        with torch.no_grad():
            swa_sink_attention(q, k, v, sink, wl, wr, scale=scale, compute_dtype=torch.float32)

    def eager_fb():
        a, b_, c = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        swa_sink_attention(a, b_, c, sink, wl, wr, scale=scale, compute_dtype=torch.float32).float().sum().backward()

    def asc_fwd(npg):
        def s():
            with torch.no_grad():
                swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, num_programs=npg,
                                         BLOCK_M=bm, BLOCK_N=bn)
        return s

    def asc_fb(npg):
        def s():
            o, lse = swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, num_programs=npg,
                                              BLOCK_M=bm, BLOCK_N=bn)
            swa_sink_bwd_ascend(q, k, v, sink, o, lse, do, wl, wr, False, scale,
                                BLOCK_M=bm, BLOCK_N=bn, num_programs=npg)
        return s

    print(f"\n### windowed  B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  dtype={DT}  tile=({bm},{bn})")
    with torch.no_grad():
        o_a, _ = swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, BLOCK_M=bm, BLOCK_N=bn)
    ref = swa_sink_attention(q.float(), k.float(), v.float(), sink, wl, wr, scale=scale,
                             compute_dtype=torch.float32)
    mx, mr = _prec(o_a, ref)
    print(f"    precision     vs eager(fp32, same inputs):  maxAbs={mx:.2e}  meanRel={mr:.2e}")
    eag = _run(eager_fwd, eager_fb)
    if eag:
        print(f"    eager         fwd={eag[0]:7.3f}ms  fwd+bwd={eag[1]:7.3f}ms  peak={eag[2]:8.1f}MB")
    for label, npg in (("ascend full  ", nt), ("ascend capped", None)):
        r = _run(asc_fwd(npg), asc_fb(npg))
        if r and eag:
            print(f"    {label} fwd={r[0]:7.3f}ms  fwd+bwd={r[1]:7.3f}ms  peak={r[2]:8.1f}MB   "
                  f"speedup fwd {eag[0]/r[0]:4.2f}x / fb {eag[1]/r[1]:4.2f}x   mem {eag[2]/r[2]:4.2f}x")


def bench_dense_mla(N, BS, KV, H, D):
    """The PRODUCTION shape: dense block form, MLA (num_kv_heads=1). q[N,H,BS,D], kv[N,KV,D]."""
    scale = D ** -0.5
    qk = torch.randn(N, H, BS, D, device="cuda", dtype=DT)          # ascend kernel layout
    kL = torch.randn(N, KV, D, device="cuda", dtype=DT)
    vL = torch.randn(N, KV, D, device="cuda", dtype=DT)
    do = torch.randn(N, H, BS, D, device="cuda", dtype=DT)
    sink = torch.randn(H, device="cuda", dtype=torch.float32)
    bm, bn = _default_blocks(D, BM_ENV, BN_ENV)
    nt = triton.cdiv(max(BS, KV), min(bm, bn)) * N * H

    def eager_fwd():
        with torch.no_grad():
            dspark_block_attention_ref(qk.transpose(1, 2), kL.unsqueeze(2).expand(N, KV, H, D),
                                       vL.unsqueeze(2).expand(N, KV, H, D), sink, scale=scale,
                                       compute_dtype=torch.float32)

    def eager_fb():
        a = qk.transpose(1, 2).detach().clone().requires_grad_(True)     # [N,BS,H,D]
        b_ = kL.detach().clone().requires_grad_(True); c = vL.detach().clone().requires_grad_(True)
        dspark_block_attention_ref(a, b_.unsqueeze(2).expand(N, KV, H, D),
                                   c.unsqueeze(2).expand(N, KV, H, D), sink, scale=scale,
                                   compute_dtype=torch.float32).float().sum().backward()

    def asc_fwd(npg):
        def s():
            with torch.no_grad():
                swa_sink_attn_fwd_ascend(qk, kL, vL, sink, 0, 0, scale=scale, dense=True,
                                         num_programs=npg, BLOCK_M=bm, BLOCK_N=bn, HG=HG_ENV)
        return s

    def asc_fb(npg):
        def s():
            o, lse = swa_sink_attn_fwd_ascend(qk, kL, vL, sink, 0, 0, scale=scale, dense=True,
                                              num_programs=npg, BLOCK_M=bm, BLOCK_N=bn, HG=HG_ENV)
            swa_sink_bwd_ascend(qk, kL, vL, sink, o, lse, do, 0, 0, True, scale, num_programs=npg)
        return s

    print(f"\n### dense MLA (production)  N={N} BS={BS} KV={KV} H={H} D={D}  dtype={DT}  tile=({bm},{bn})"
          f"{f'  HG={HG_ENV}' if HG_ENV else ''}")
    with torch.no_grad():
        o_a, _ = swa_sink_attn_fwd_ascend(qk, kL, vL, sink, 0, 0, scale=scale, dense=True,
                                          BLOCK_M=bm, BLOCK_N=bn, HG=HG_ENV)
    ref = dspark_block_attention_ref(qk.transpose(1, 2).float(),
                                     kL.float().unsqueeze(2).expand(N, KV, H, D),
                                     vL.float().unsqueeze(2).expand(N, KV, H, D), sink,
                                     scale=scale, compute_dtype=torch.float32)  # [N,BS,H,D]
    mx, mr = _prec(o_a.transpose(1, 2), ref)
    print(f"    precision     vs gold(fp32, same inputs):  maxAbs={mx:.2e}  meanRel={mr:.2e}")
    eag = _run(eager_fwd, eager_fb)
    if eag:
        print(f"    eager         fwd={eag[0]:7.3f}ms  fwd+bwd={eag[1]:7.3f}ms  peak={eag[2]:8.1f}MB")
    for label, npg in (("ascend full  ", nt), ("ascend capped", None)):
        r = _run(asc_fwd(npg), asc_fb(npg))
        if r and eag:
            print(f"    {label} fwd={r[0]:7.3f}ms  fwd+bwd={r[1]:7.3f}ms  peak={r[2]:8.1f}MB   "
                  f"speedup fwd {eag[0]/r[0]:4.2f}x / fb {eag[1]/r[1]:4.2f}x   mem {eag[2]/r[2]:4.2f}x")


def main():
    if not torch.cuda.is_available():
        print("!! run on the GPU box"); raise SystemExit(1)
    print(">>> Ascend op GPU speed/memory vs eager  (full = GPU-fair; capped = NPU grid shape)")
    print(">>> speedup = eager / ascend; a GPU 'capped' slowdown is the grid cap, corrected on the A3")
    print(">>> tile override: BM=.. BN=.. python tests/bench_ascend.py  (find a fast block for D=512)")
    if DT == torch.float32:
        print(">>> NOTE fp32: the kernel uses input_precision='ieee' (TRUE fp32 -> the SLOW matmul "
              "path) while eager fp32 uses fast TF32, so fp32 speed is NOT comparable and is NOT "
              "the production path. Production is bf16 (below). ieee is CUDA-only; revisited on the A3.")
    print()
    H, D = DSV4["num_heads"], DSV4["head_dim"]
    BS, WIN = DSV4["block_size"], DSV4["window_size"]
    KV = WIN + BS
    wl, wr = dspark_sas_window(BS, WIN)
    for L in LS:
        bench_windowed(1, H, L, D, wl, wr)
    bench_dense_mla(int(os.environ.get("NBLK", "64")), BS, KV, H, D)


if __name__ == "__main__":
    main()
