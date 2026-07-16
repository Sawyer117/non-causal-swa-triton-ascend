#!/usr/bin/env python3
"""Probe WHY the compiled SAS op mismatches the reference at KV>128 (DSV4 real KV=133).

Background (AUDIT_pr11196_op_2026-07-16.md §6): a KV sweep showed the op is CORRECT at KV<=128 and wrong
at KV=133/144 (NaN at 256) *when called the way impl A's `_call_dspark_sas_block` calls it*:
layout_kv="TND", seqused_kv=None, no block_table. The op's DESIGNED interface is PA_ND (paged) with
seqused_kv + ori_block_table (what the real serve `dsa_v1.py` uses -> AL 3.94). Biggest difference:
**seqused_kv** (real serve passes the actual KV length; impl A passes None). Sink was ruled out
(NOSINK unchanged). This isolates it: same B=1, KV=133 scenario, compiled op vs a fp32 dense+sink
reference under three call conventions:
  1. TND, seqused_kv=None  (= impl A, reuses vllm_ascend._call_dspark_sas_block) -> reproduces the bug
  2. TND, seqused_kv=[KV]   -> does telling the op the real length fix the multi-tile path?
  3. PA_ND, paged+seqused   -> the real serve convention

NOTE: we import vllm_ascend.ops.dspark_attention and get the ops via its own _get_dspark_sas_ops, and
run the KNOWN-GOOD _call_dspark_sas_block first, so the AICPU metadata op is loaded exactly as the
working harnesses load it (a cold direct torch.ops call otherwise fails "aclnn...Metadata not in
libopapi.so").

RUN (A3):  python probe_kv_path.py         # KV=133
           WIN=123 BS=5 python probe_kv_path.py   # KV=128 control
"""
import os

try:
    import torch
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! need torch + torch_npu on an Ascend NPU: {e}")

try:
    import vllm_ascend.ops.dspark_attention as dsa  # side effects register/expose the ops
    from vllm_ascend.ops.dspark_attention import (  # noqa: F401
        DSPARK_SAS_CMP_MASK_MODE, DSPARK_SAS_MASK_MODE, _call_dspark_sas_block, _get_dspark_sas_ops)
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! cannot import vllm_ascend.ops.dspark_attention: {e}")


def _ensure_sas_op():
    try:
        torch.ops._C_ascend.npu_sparse_attn_sharedkv
        return
    except (AttributeError, RuntimeError):
        pass
    import glob
    import vllm_ascend
    for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so); print(f">>> loaded {so}")
        except Exception as e:  # noqa: BLE001
            print(f">>> could not load {so}: {e}")


_ensure_sas_op()

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(os.environ.get("DTYPE", "bfloat16"),
                                                                torch.bfloat16)
H = int(os.environ.get("H", "64")); D = int(os.environ.get("D", "512"))
WIN = int(os.environ.get("WIN", "128")); BS = int(os.environ.get("BS", "5"))
KV = WIN + BS
SCALE = D ** -0.5
WIN_L = WIN + BS - 1
WIN_R = BS - 1


def reference(q, kv, sink):
    k = kv.unsqueeze(1).expand(-1, H, -1).float()            # [KV, H, D]  (shared latent -> all heads)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * SCALE
    s = sink[:H].float().view(1, H, 1)
    m = torch.maximum(scores.max(dim=-1, keepdim=True).values, s)
    e = torch.exp(scores - m)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(s - m))
    return torch.einsum("qhk,khd->qhd", p, k).to(q.dtype)


def cmp(x, ref):
    x, r = x.float(), ref.float(); d = (x - r).abs()
    return (bool(torch.allclose(x, r, atol=2e-2, rtol=2e-2)), d.max().item(), d.mean().item(),
            (d / (r.abs() + 1e-6)).mean().item())


def line(tag, out, ref):
    if out is None:
        return
    c, mx, ma, mr = cmp(out, ref)
    print(f"[{tag:26}] allclose={str(c):5}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}")


def op_tnd_seqused(q, packed, sink, metadata_op, attn_op):
    """Exactly _call_dspark_sas_block, but ALSO pass seqused_kv=[KV] to metadata_op AND attn_op."""
    cu_q = torch.tensor([0, BS], dtype=torch.int32, device=DEV)
    cu_kv = torch.tensor([0, KV], dtype=torch.int32, device=DEV)
    seqused_kv = torch.tensor([KV], dtype=torch.int32, device=DEV)
    sinks = sink[:H].float().contiguous()
    meta = metadata_op(num_heads_q=H, num_heads_kv=1, head_dim=D, cu_seqlens_q=cu_q,
                       cu_seqlens_ori_kv=cu_kv, cu_seqlens_cmp_kv=None, seqused_q=None,
                       seqused_kv=seqused_kv, batch_size=1, max_seqlen_q=BS, max_seqlen_kv=KV,
                       cmp_ratio=1, ori_mask_mode=DSPARK_SAS_MASK_MODE, cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
                       ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="TND",
                       has_ori_kv=True, has_cmp_kv=False, device=str(DEV))
    return attn_op(q, ori_kv=packed, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_kv, seqused_kv=seqused_kv,
                   sinks=sinks, metadata=meta, softmax_scale=SCALE, cmp_ratio=1,
                   ori_mask_mode=DSPARK_SAS_MASK_MODE, cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
                   ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="TND")[0]


def op_paged(q, kv, sink, metadata_op, attn_op):
    """PA_ND paged, real-serve convention: ori_kv=[num_blocks, page, 1, D] + ori_block_table + seqused."""
    page = 128
    nblk = (KV + page - 1) // page
    ori_kv = torch.zeros(nblk, page, 1, D, device=DEV, dtype=DT)
    ori_kv.view(nblk * page, 1, D)[:KV] = kv.unsqueeze(1)
    block_table = torch.arange(nblk, dtype=torch.int32, device=DEV).view(1, nblk)
    cu_q = torch.tensor([0, BS], dtype=torch.int32, device=DEV)
    cu_kv = torch.tensor([0, KV], dtype=torch.int32, device=DEV)
    seqused_kv = torch.tensor([KV], dtype=torch.int32, device=DEV)
    sinks = sink[:H].float().contiguous()
    meta = metadata_op(num_heads_q=H, num_heads_kv=1, head_dim=D, cu_seqlens_q=cu_q,
                       cu_seqlens_ori_kv=cu_kv, cu_seqlens_cmp_kv=None, seqused_q=None,
                       seqused_kv=seqused_kv, batch_size=1, max_seqlen_q=BS, max_seqlen_kv=KV,
                       cmp_ratio=1, ori_mask_mode=DSPARK_SAS_MASK_MODE, cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
                       ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="PA_ND",
                       has_ori_kv=True, has_cmp_kv=False, device=str(DEV))
    return attn_op(q, ori_kv=ori_kv, ori_block_table=block_table, cu_seqlens_q=cu_q,
                   seqused_kv=seqused_kv, sinks=sinks, metadata=meta, softmax_scale=SCALE, cmp_ratio=1,
                   ori_mask_mode=DSPARK_SAS_MASK_MODE, cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
                   ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="PA_ND")[0]


def main():
    q = torch.randn(BS, H, D, device=DEV, dtype=DT)
    kv = torch.randn(KV, D, device=DEV, dtype=DT)
    sink = torch.randn(H, device=DEV, dtype=DT)
    packed = kv.unsqueeze(1).contiguous()                    # [KV, 1, D]
    ref = reference(q, kv, sink)

    ops = _get_dspark_sas_ops(q)
    if ops is None:
        raise SystemExit("!! SAS op not available.")
    metadata_op, attn_op = ops
    torch.npu.synchronize()
    print(f">>> probe   H={H} D={D} WIN={WIN} BS={BS} KV={KV}  win_left={WIN_L} win_right={WIN_R}  dtype={DT}")
    print(f">>> KV={KV} spans {(KV + 127)//128} tile(s) of 128\n")

    variants = [
        ("TND seqused_kv=None (implA)",
         lambda: _call_dspark_sas_block(q, packed, sink, SCALE, BS, WIN, metadata_op, attn_op)),
        ("TND seqused_kv=[KV]", lambda: op_tnd_seqused(q, packed, sink, metadata_op, attn_op)),
        ("PA_ND paged+seqused (serve)", lambda: op_paged(q, kv, sink, metadata_op, attn_op)),
    ]
    for tag, fn in variants:
        try:
            out = fn(); torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"[{tag:26}] EXCEPTION: {type(e).__name__}: {str(e)[:130]}")
            continue
        line(tag, out, ref)

    print("\n>>> read: if 'TND seqused_kv=[KV]' or 'PA_ND' matches (~1e-2 bf16) but 'seqused=None' does "
          "not,\n    the op is CORRECT and impl A's seqused=None/TND call is the bug (small fix). "
          "If none match -> multi-tile compute broken.")


if __name__ == "__main__":
    main()
