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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from eager_reference import swa_sink_attention, dspark_sas_window, DSV4  # noqa: E402
from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend, _default_blocks  # noqa: E402
from triton_impl.swa_sink_ascend_bwd import swa_sink_bwd_ascend  # noqa: E402

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
    return (time.time() - t0) / ITERS * 1e3


def _peak(step):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


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
    bm, bn = _default_blocks(D, None, None)
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
                swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, num_programs=npg)
        return s

    def asc_fb(npg):
        def s():
            o, lse = swa_sink_attn_fwd_ascend(q, k, v, sink, wl, wr, scale=scale, num_programs=npg)
            swa_sink_bwd_ascend(q, k, v, sink, o, lse, do, wl, wr, False, scale, num_programs=npg)
        return s

    print(f"\n### windowed  B={B} H={H} L={L} D={D}  window=(L{wl},R{wr})  dtype={DT}  tile=({bm},{bn})")
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
    print(">>> speedup = eager / ascend; a GPU 'capped' slowdown is the grid cap, corrected on the A3\n")
    H, D = DSV4["num_heads"], DSV4["head_dim"]
    wl, wr = dspark_sas_window(DSV4["block_size"], DSV4["window_size"])
    for L in LS:
        bench_windowed(1, H, L, D, wl, wr)


if __name__ == "__main__":
    main()
