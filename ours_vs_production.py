#!/usr/bin/env python3
"""OUR Triton kernel vs the COMPILED PRODUCTION op (npu_sparse_attn_sharedkv), on identical inputs.

This is the baseline comparison that matters: the baseline is the PRODUCTION operator, not eager/gold.
It reuses fused_sas_vs_reference_parity.py's scenario (the vllm_ascend paged-context setup) but forces
the KV latent to be SHARED across heads (MLA: num_kv_heads=1, per reference_from_repo/dsv4_mla_ref.py),
then runs THREE paths on the SAME inputs via vllm_ascend's `dspark_attention(..., shared_kv=True)`:
  * FUSED : the compiled SAS kernel  (npu_sparse_attn_sharedkv)  <- the PRODUCTION baseline
  * REF   : the per-block _dspark_attention_reference loop (fp32 internals)  <- the fp32 oracle
  * OURS  : our Triton kernel (swa_sink_attn_fwd_ascend), fed the SAME per-block KV assembled from the
            scenario cache+draft.

Reports parity OURS-vs-FUSED (do we match production?), OURS-vs-REF, FUSED-vs-REF, and the fwd SPEED
of OURS vs FUSED (is our Triton faster than the compiled AscendC op?). bf16 by default (production
dtype); DTYPE=float32 -> diffs collapse to ~1e-6 (proves it's dtype, not a math bug).

RUN (A3, env dspark-dsv4-base, vllm_ascend built WITH the sparse_attn_sharedkv kernel):
    python ours_vs_production.py
    DTYPE=float32 python ours_vs_production.py
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
    import vllm_ascend.ops.dspark_attention as dsa
    from vllm_ascend.ops.dspark_attention import _dspark_sas_window, dspark_attention
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import vllm_ascend.ops.dspark_attention: {e}\n"
                     "   run on the A3 with vllm_ascend installed (env dspark-dsv4-base).")

try:
    from triton_impl.swa_sink_ascend import swa_sink_attn_fwd_ascend
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import our Triton kernel: {e}")

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(os.environ.get("DTYPE", "bfloat16"),
                                                                torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
H = int(os.environ.get("H", "64")); D = int(os.environ.get("D", "512"))
WIN = int(os.environ.get("WIN", "128")); BS = int(os.environ.get("BS", "7"))
NBLK = int(os.environ.get("NBLK", "64"))
KV = WIN + BS
SCALE = D ** -0.5
CAP = 1 << (max(WIN + BS, 1) - 1).bit_length()
NITER = 20


def build_scenario():
    """Same paged layout as fused_sas_vs_reference_parity.py, but KV is SHARED across heads (MLA):
    each head gets the SAME cache/draft K, so the per-head op computes the shared-latent attention."""
    base = WIN
    n = NBLK * BS
    q = torch.randn(n, H, D, device=DEV, dtype=DT)
    attn_sink = torch.randn(H, device=DEV, dtype=DT)
    # ONE latent per (token), broadcast to all H heads -> MLA (num_kv_heads=1)
    draft_k1 = torch.randn(n, 1, D, device=DEV, dtype=DT)
    draft_k = draft_k1.expand(n, H, D).contiguous()
    positions = torch.empty(n, dtype=torch.int32, device=DEV)
    request_slots = torch.empty(n, dtype=torch.int32, device=DEV)
    cache_k = torch.zeros(NBLK, CAP, H, D, device=DEV, dtype=DT)
    cache_positions = torch.full((NBLK, CAP), -1, dtype=torch.int32, device=DEV)
    cache_valid = torch.zeros(NBLK, CAP, dtype=torch.bool, device=DEV)
    ctx_pos = torch.arange(base - WIN, base, device=DEV)
    idx = (ctx_pos % CAP).long()
    ctx_latent = torch.randn(NBLK, WIN, 1, D, device=DEV, dtype=DT)   # shared across heads
    for b in range(NBLK):
        sl = slice(b * BS, (b + 1) * BS)
        positions[sl] = torch.arange(base, base + BS, dtype=torch.int32, device=DEV)
        request_slots[sl] = b
        cache_k[b, idx] = ctx_latent[b].expand(WIN, H, D)
        cache_positions[b, idx] = ctx_pos.to(torch.int32)
        cache_valid[b, idx] = True
    s = dict(q=q, k_cache=cache_k, v_cache=cache_k, cache_positions=cache_positions,
             cache_valid=cache_valid, draft_k=draft_k, draft_v=draft_k, request_slots=request_slots,
             positions=positions, attn_sink=attn_sink, block_size=BS, window_size=WIN,
             softmax_scale=SCALE)
    # per-block assembled KV latent for OUR kernel: [WIN ctx | BS block], head 0 (all heads equal)
    kvl = torch.stack([torch.cat([ctx_latent[b, :, 0], draft_k1[b * BS:(b + 1) * BS, 0]], dim=0)
                       for b in range(NBLK)], dim=0)                  # [NBLK, KV, D]
    qb = q.view(NBLK, BS, H, D)
    return s, qb, kvl, attn_sink


def run_entry(s):
    return dspark_attention(s["q"], s["k_cache"], s["v_cache"], s["cache_positions"], s["cache_valid"],
                            s["draft_k"], s["draft_v"], s["request_slots"], s["positions"],
                            s["attn_sink"], s["block_size"], s["window_size"], s["softmax_scale"],
                            shared_kv=True)


class _override:
    def __init__(self, **kw): self.kw = kw; self.saved = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(dsa, k); setattr(dsa, k, v)
    def __exit__(self, *_):
        for k, v in self.saved.items():
            setattr(dsa, k, v)


def ours(qb, kvl, sink):
    o, _ = swa_sink_attn_fwd_ascend(qb.transpose(1, 2).contiguous(), kvl, kvl, sink, 0, 0,
                                    scale=SCALE, dense=True)          # [NBLK,H,BS,D]
    return o.transpose(1, 2).reshape(NBLK * BS, H, D)                 # [n,H,D] like the entry


def cmp(x, ref):
    x, ref = x.float(), ref.float(); d = (x - ref).abs()
    return (bool(torch.allclose(x, ref, atol=ATOL, rtol=RTOL)), d.max().item(),
            d.mean().item(), (d / (ref.abs() + 1e-6)).mean().item())


def time_ms(fn):
    for _ in range(3): fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(NITER): fn()
    torch.npu.synchronize()
    return (time.time() - t0) / NITER * 1e3


def main():
    s, qb, kvl, sink = build_scenario()
    mode, wl, wr = _dspark_sas_window(BS, WIN)
    print(f">>> OURS (Triton) vs PRODUCTION (npu_sparse_attn_sharedkv) vs REF   H={H} D={D} win={WIN} "
          f"block={BS} blocks={NBLK}  MLA shared-KV  dtype={DT}")
    print(f">>> _dspark_sas_window(block={BS}, window={WIN}) = mode={mode}, win_left={wl}, win_right={wr}\n")

    sas_available = dsa._get_dspark_sas_ops(s["q"]) is not None       # noqa: SLF001

    # REF (fp32 reference loop) + OURS are always available -> correctness check.
    with _override(_get_dspark_attention_custom_op=lambda q: None, _get_dspark_sas_ops=lambda q: None):
        ref = run_entry(s)
    ours_o = ours(qb, kvl, sink)
    torch.npu.synchronize()
    c, mx, ma, mr = cmp(ours_o, ref)
    print(f"[parity]  OURS vs REF   allclose={c}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}"
          f"   (correctness; bf16 ~1e-2 is dtype, DTYPE=float32 -> ~1e-6)")

    if not sas_available:
        # DO NOT print a "production" speedup here: with the op missing, the entry falls back to the
        # slow per-block reference LOOP, so any "production" number would be vs that loop (meaningless).
        to = time_ms(lambda: ours(qb, kvl, sink))
        print(f"[fwd speed]  ours={to:7.3f}ms   (our kernel alone)")
        print("\n!! production SAS op (npu_sparse_attn_sharedkv) is NOT built in this env, so there is\n"
              "   NO real production baseline to compare against. The FUSED entry would silently fall\n"
              "   back to the per-block fp32 reference loop (~100ms of Python) — comparing to THAT is\n"
              "   meaningless (it would print a fake ~100x+ 'speedup' and PROD-vs-REF=0.0).\n"
              "   Build vllm-ascend from source on the dspark-dsv4 branch (install step 4) so\n"
              "   torch.ops._C_ascend.npu_sparse_attn_sharedkv registers, then re-run for the REAL number.\n"
              "   Check:  python -c \"import torch,torch_npu; print(hasattr(torch.ops._C_ascend,"
              "'npu_sparse_attn_sharedkv'))\"")
        return

    # SAS op present -> real production comparison.
    with _override(_get_dspark_attention_custom_op=lambda q: None):
        fused = run_entry(s)
    torch.npu.synchronize()
    for name, a, b in (("OURS vs PROD", ours_o, fused), ("PROD vs REF ", fused, ref)):
        c, mx, ma, mr = cmp(a, b)
        print(f"[parity]  {name}  allclose={c}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}")
    print()
    tp = time_ms(lambda: run_entry(s))
    to = time_ms(lambda: ours(qb, kvl, sink))
    print(f"[fwd speed]  production={tp:7.3f}ms   ours={to:7.3f}ms   speedup {tp / to:4.2f}x")
    print("\n>>> OURS vs PROD allclose=True => our Triton kernel matches the production op. "
          "speedup>1 => our Triton is faster than the compiled op.")


if __name__ == "__main__":
    main()
