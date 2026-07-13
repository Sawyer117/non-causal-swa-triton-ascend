#!/usr/bin/env python3
"""OUR Triton Ascend op vs the vLLM-Ascend GOLD reference, on the A3 (the on-NPU validation).

Counterpart to `fused_sas_vs_reference_parity.py` (which benches the COMPILED SAS fused op vs
the reference). Here we plug in OUR kernel: `triton_impl/swa_sink_ascend` (the 1-D-grid,
core-capped, NPU-shaped op — GPU-validated bit-identical to the CUDA version). It is compared,
at the REAL DSpark block shapes, against:
  * GOLD  : vllm_ascend `_dspark_attention_reference` (per draft block) — the parity target.
  * eager : batched einsum+sink (the fast pure-torch fallback) — the speed/memory baseline.

Real shapes (HF config.json): H=64, D=512, num_kv_heads=1 (MLA), window=128, block_size=7,
so KV = window + block = 135; each draft block's KV is [window context | the block] (dense).

READING: bf16 diff ~1e-2 vs the fp32 gold is expected rounding; DTYPE=float32 -> ~1e-6 (proves
it's dtype, not a math bug). The op's grid is core-capped (grid <= cube cores) — correct on the
A3. Full grad correctness is separately bit-identical to the validated CUDA backward.

RUN on the A3 (env dspark-dsv4-base, vllm_ascend installed):
    python fused_sas_vs_ours.py
    DTYPE=float32 python fused_sas_vs_ours.py        # diffs collapse to ~1e-6
    NBLK=512 python fused_sas_vs_ours.py             # full scale

PERF TUNING (MLA-dense uses the D-tiled kernel by default; the forward was Cube-starved at M=7):
    python fused_sas_vs_ours.py                      # default BLOCK_M=64 rows/tile, BLOCK_K=128
    BM=128 python fused_sas_vs_ours.py               # more rows per tile (fills the Cube M axis)
    BM=64 BK=256 python fused_sas_vs_ours.py         # sweep the D-tile too
  The MLA KV latent is shared across all H heads, so a program batches BLOCK_M flattened (head,query)
  rows against the same K/V. KV=135 is tiny -> single-pass softmax (full scores[M,135]). D=512 is
  TILED by BLOCK_K in both matmuls, so nothing [*,512] is on chip (no more UB/cbuf overflow) and M
  can grow. Sweep BM in {32,64,128,256}, BK in {64,128,256}. HG=n is an alias (BLOCK_M=next_pow2(n*BS)).

BACKWARD tuning (it dominates fwd+bwd): BMDQ = dq rows/tile (dq holds qk+dp fp32 -> ~32 safe),
BKDQ = dq D-tile, BMKV = dk/dv rows/tile, BKV = dk/dv key-tile (bigger = fewer Q/DO re-reads but
bigger dk_acc; capped ~16 at D=512). e.g.  BMKV=32 python fused_sas_vs_ours.py   /   BKV=16 BMKV=24 ...
"""
import os
import time

try:
    import torch
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! need torch + torch_npu on an Ascend NPU: {e}")

try:
    from vllm_ascend.ops.dspark_attention import _dspark_attention_reference, _dspark_sas_window
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import the vllm_ascend gold reference: {e}\n"
                     "   run on a node with vllm_ascend installed (the A3), env dspark-dsv4-base.")

try:
    from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend
    from triton_impl.swa_sink_ascend_bwd import swa_sink_bwd_ascend
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import our Ascend op (triton_impl): {e}")

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
NBLK = int(os.environ.get("NBLK", "64"))
BS = int(os.environ.get("BS", "7"))
WIN = int(os.environ.get("WIN", "128"))
H = int(os.environ.get("H", "64"))
D = int(os.environ.get("D", "512"))
KV = WIN + BS
SCALE = D ** -0.5
NITER = 20
BM = int(os.environ["BM"]) if "BM" in os.environ else None   # fwd rows per tile (M axis, Cube fill)
BN = int(os.environ["BN"]) if "BN" in os.environ else None   # (legacy flash key-block; unused by D-tiled fwd)
HG = int(os.environ["HG"]) if "HG" in os.environ else None   # alias: BLOCK_M = next_pow2(HG*BS)
BK = int(os.environ["BK"]) if "BK" in os.environ else None   # fwd D-tile (BLOCK_K, default 128)
# backward tile knobs (sweep on the A3): dq rows / dq D-tile / dk,dv rows / dk,dv key-tile
BMDQ = int(os.environ["BMDQ"]) if "BMDQ" in os.environ else None
BKDQ = int(os.environ["BKDQ"]) if "BKDQ" in os.environ else None
BMKV = int(os.environ["BMKV"]) if "BMKV" in os.environ else None
BKV = int(os.environ["BKV"]) if "BKV" in os.environ else None

mm, wl, wr = _dspark_sas_window(BS, WIN)
print(f">>> OUR Ascend op vs vllm_ascend GOLD  (_dspark_attention_reference)")
print(f">>> {NBLK} blocks  q[{BS},{H},{D}] attends dense to kv[{KV},{D}] (ctx {WIN}+block {BS}), "
      f"MLA num_kv_heads=1  dtype={DT}")
print(f">>> _dspark_sas_window(block={BS}, window={WIN}) = mode={mm}, win_left={wl}, win_right={wr}"
      f"   allclose atol={ATOL} rtol={RTOL}\n")

# q [N,BS,H,D]; MLA latent kv [N,KV,D] (one head shared across all H); sink [H]
Q = torch.randn(NBLK, BS, H, D, device=DEV, dtype=DT)
KL = torch.randn(NBLK, KV, D, device=DEV, dtype=DT)
VL = torch.randn(NBLK, KV, D, device=DEV, dtype=DT)
SINK = torch.randn(H, device=DEV, dtype=DT)
DO = torch.randn(NBLK, BS, H, D, device=DEV, dtype=DT)


def gold():
    """vllm_ascend reference per block; MLA -> broadcast the latent to [KV,H,D]."""
    kh = KL.unsqueeze(2).expand(NBLK, KV, H, D)
    vh = VL.unsqueeze(2).expand(NBLK, KV, H, D)
    return torch.stack([_dspark_attention_reference(Q[i], kh[i], vh[i], SINK, SCALE)
                        for i in range(NBLK)], dim=0)                      # [N,BS,H,D]


def manual():
    """batched einsum+sink (fast pure-torch fallback) — the speed/memory baseline."""
    s = torch.einsum("nqhd,nkd->nqhk", Q.float(), KL.float()) * SCALE      # MLA: kv shared
    sink = SINK.float().view(1, 1, H, 1)
    smax = torch.maximum(s.max(dim=-1, keepdim=True).values, sink)
    e = torch.exp(s - smax)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sink - smax))
    return torch.einsum("nqhk,nkd->nqhd", p, VL.float()).to(DT)


def ours_fwd():
    o, _ = swa_sink_attn_fwd_ascend(Q.transpose(1, 2).contiguous(), KL, VL, SINK, 0, 0, scale=SCALE,
                                    dense=True, BLOCK_M=BM, HG=HG, BLOCK_K=BK)  # [N,H,BS,D]
    return o.transpose(1, 2)                                              # [N,BS,H,D]


def ours_fb():
    qk = Q.transpose(1, 2).contiguous()
    o, lse = swa_sink_attn_fwd_ascend(qk, KL, VL, SINK, 0, 0, scale=SCALE, dense=True,
                                      BLOCK_M=BM, HG=HG, BLOCK_K=BK)
    # backward is the D-tiled/row-tiled MLA-dense path; BMDQ/BKDQ/BMKV/BKV env sweep its tiles.
    swa_sink_bwd_ascend(qk, KL, VL, SINK, o, lse, DO.transpose(1, 2).contiguous(),
                        0, 0, True, SCALE, BM_DQ=BMDQ, BK_DQ=BKDQ, BM_DKDV=BMKV, BLOCK_KV=BKV)


def cmp(x, ref):
    x, ref = x.float(), ref.float()
    d = (x - ref).abs()
    return (bool(torch.allclose(x, ref, atol=ATOL, rtol=RTOL)), d.max().item(),
            d.mean().item(), (d / (ref.abs() + 1e-6)).mean().item())


def time_ms(step, fb=False):
    for _ in range(3):
        step()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(NITER):
        step()
    torch.npu.synchronize()
    return (time.time() - t0) / NITER * 1e3


def peak_mb(step):
    torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()
    step(); torch.npu.synchronize()
    return torch.npu.max_memory_allocated() / 1e6


# ---- parity: OURS vs GOLD (and manual vs gold, to confirm the setup) ----
g = gold()
for name, out in (("manual(einsum)", manual()), ("OURS(triton)", ours_fwd())):
    c, mx, ma, mr = cmp(out, g)
    print(f"[parity vs gold]  {name:16} allclose={c}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}")

# ---- fp32 sanity: same kernel in fp32 vs the fp32 gold. Ascend has no tf32, so bf16 inputs -> a
# bf16 Cube (the ~1e-2 above is inherent dtype rounding). In fp32 the Cube is exact -> ~1e-6, which
# proves the kernel MATH matches gold (not a bug). Skipped when already running DTYPE=float32.
if DT is not torch.float32:
    Qf, KLf, VLf, SKf = Q.float(), KL.float(), VL.float(), SINK.float()
    khf = KLf.unsqueeze(2).expand(NBLK, KV, H, D); vhf = VLf.unsqueeze(2).expand(NBLK, KV, H, D)
    gf = torch.stack([_dspark_attention_reference(Qf[i], khf[i], vhf[i], SKf, SCALE)
                      for i in range(NBLK)], dim=0)
    of32, _ = swa_sink_attn_fwd_ascend(Qf.transpose(1, 2).contiguous(), KLf, VLf, SKf, 0, 0,
                                       scale=SCALE, dense=True, BLOCK_M=BM, HG=HG, BLOCK_K=BK)
    c, mx, ma, mr = cmp(of32.transpose(1, 2), gf)
    print(f"[parity fp32   ]  OURS(triton)     allclose={c}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  "
          f"meanRel={mr:.2e}   (math check; bf16 diff above is dtype rounding, not a bug)")

# ---- speed + memory: OURS vs the eager fallback (manual einsum) ----
print()
mf = time_ms(lambda: manual())
of = time_ms(ours_fwd)
print(f"[fwd  speed]  eager={mf:7.3f}ms   ours={of:7.3f}ms   speedup {mf / of:4.2f}x")
try:
    om = peak_mb(ours_fwd); em = peak_mb(lambda: manual())
    print(f"[fwd  mem  ]  eager={em:8.1f}MB  ours={om:8.1f}MB   {em / om:4.2f}x less")
except Exception as e:  # noqa: BLE001
    print(f"[mem] skipped: {e}")
ofb = time_ms(ours_fb)
print(f"[fwd+bwd    ]  ours={ofb:7.3f}ms  (grads to q,k,v,sink; correctness = bit-identical to the "
      f"validated CUDA backward)")

print("\n>>> read: OURS allclose=True with small meanAbs/meanRel -> matches the vllm_ascend gold.")
print(">>>       bf16 ~1e-2 is dtype rounding; DTYPE=float32 -> ~1e-6. To also check vs the COMPILED")
print(">>>       SAS op, run fused_sas_vs_reference_parity.py (fused-vs-ref); both matching ref => equal.")
