#!/usr/bin/env python3
"""The UNCHANGED MindSpeed sparse-flash-attention kernel, driven DENSE, vs the vLLM-Ascend GOLD.

This is the "adapt the production operator" path: `triton_impl/swa_sink_mindspeed.py` is MindSpeed-LLM's
DeepSeek-V4 `SparseFlashAttentionTriton` copied verbatim (Apache-2.0), and we feed it an IDENTITY topk
(all KV keys) so the sparse kernel computes DENSE attention over each block's KV latent — no kernel
edits. The kernel is K==V (a single latent is both key and value; confirmed for DSpark), so here the
latent KV drives both, and the gold is computed with kh==vh==KV.

Real shapes (HF config.json): H=64, D=512, num_kv_heads=1 (MLA), window=128, block_size=7 -> KV=135.
bf16 only (the kernel's on-chip buffers are bf16); it is NPU-only (uses the Ascend `al` cube/vector
sync extension), so this runs on the A3, not on a GPU.

RUN on the A3 (env with torch_npu + vllm_ascend + the MindSpeed triton-ascend `al` extension):
    python mindspeed_vs_gold.py
    NBLK=512 python mindspeed_vs_gold.py
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
    raise SystemExit(f"!! cannot import the vllm_ascend gold reference: {e}")

try:
    from triton_impl.swa_sink_mindspeed import mindspeed_dense_sink_attn
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import the MindSpeed adapter (needs the Ascend `al` extension): {e}")

torch.manual_seed(0)
DT = torch.bfloat16                                       # kernel buffers are bf16
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

mm, wl, wr = _dspark_sas_window(BS, WIN)
print(">>> UNCHANGED MindSpeed sparse kernel, driven DENSE (identity topk), vs vllm_ascend GOLD")
print(f">>> {NBLK} blocks  q[{BS},{H},{D}] attends dense to kv[{KV},{D}], MLA K==V (shared latent)  dtype={DT}")
print(f">>> _dspark_sas_window(block={BS}, window={WIN}) = mode={mm}, win_left={wl}, win_right={wr}"
      f"   allclose atol={ATOL} rtol={RTOL}\n")

# q [N,BS,H,D]; single MLA latent KVL [N,KV,D] (K==V -> both key and value); sink [H]
Q = torch.randn(NBLK, BS, H, D, device=DEV, dtype=DT)
KVL = torch.randn(NBLK, KV, D, device=DEV, dtype=DT)
SINK = torch.randn(H, device=DEV, dtype=DT)
DO = torch.randn(NBLK, BS, H, D, device=DEV, dtype=DT)


def gold():
    """vllm_ascend reference per block; K==V -> broadcast the ONE latent to both kh and vh."""
    kh = KVL.unsqueeze(2).expand(NBLK, KV, H, D)         # kh == vh (shared latent)
    return torch.stack([_dspark_attention_reference(Q[i], kh[i], kh[i], SINK, SCALE)
                        for i in range(NBLK)], dim=0)     # [N,BS,H,D]


def eager():
    """batched einsum+sink, K==V (the fast pure-torch baseline)."""
    s = torch.einsum("nqhd,nkd->nqhk", Q.float(), KVL.float()) * SCALE
    sink = SINK.float().view(1, 1, H, 1)
    smax = torch.maximum(s.max(dim=-1, keepdim=True).values, sink)
    e = torch.exp(s - smax)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sink - smax))
    return torch.einsum("nqhk,nkd->nqhd", p, KVL.float()).to(DT)


def ours_fwd():
    return mindspeed_dense_sink_attn(Q, KVL, SINK, SCALE)          # [N,BS,H,D]


def ours_fb():
    q = Q.detach().clone().requires_grad_(True)
    kv = KVL.detach().clone().requires_grad_(True)
    sk = SINK.detach().clone().float().requires_grad_(True)
    o = mindspeed_dense_sink_attn(q, kv, sk, SCALE)
    (o.float() * DO.float()).sum().backward()


def cmp(x, ref):
    x, ref = x.float(), ref.float()
    d = (x - ref).abs()
    return (bool(torch.allclose(x, ref, atol=ATOL, rtol=RTOL)), d.max().item(),
            d.mean().item(), (d / (ref.abs() + 1e-6)).mean().item())


def time_ms(step):
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


g = gold()
for name, out in (("eager(einsum)", eager()), ("MindSpeed(triton)", ours_fwd())):
    c, mx, ma, mr = cmp(out, g)
    print(f"[parity vs gold]  {name:18} allclose={c}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}")

print()
ef = time_ms(eager)
of = time_ms(ours_fwd)
print(f"[fwd  speed]  eager={ef:7.3f}ms   MindSpeed={of:7.3f}ms   speedup {ef / of:4.2f}x")
try:
    om = peak_mb(ours_fwd); em = peak_mb(eager)
    print(f"[fwd  mem  ]  eager={em:8.1f}MB  MindSpeed={om:8.1f}MB   {em / om:4.2f}x")
except Exception as e:  # noqa: BLE001
    print(f"[mem] skipped: {e}")
try:
    ofb = time_ms(ours_fb)
    print(f"[fwd+bwd    ]  MindSpeed={ofb:7.3f}ms  (grads to q, kv-latent, sink)")
except Exception as e:  # noqa: BLE001
    print(f"[fwd+bwd] FAILED: {type(e).__name__}: {e}")

print("\n>>> read: MindSpeed(triton) allclose=True -> the unchanged production kernel, driven dense,")
print(">>>       matches the gold. bf16 ~1e-2 is dtype rounding (the kernel is bf16 end-to-end).")
