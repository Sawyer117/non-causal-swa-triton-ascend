#!/usr/bin/env python3
"""Probe WHY the compiled SAS op mismatches the reference at KV>128 (DSV4 real KV=133).

Background (see AUDIT_pr11196_op_2026-07-16.md §6): a KV sweep showed the op is CORRECT at KV<=128 and
wrong at KV=133/144 (NaN at 256) *when called the way our harness calls it* — impl A's
`_call_dspark_sas_block`: layout_kv="TND", seqused_kv=None, no block_table. The op's DESIGNED interface
is PA_ND (paged) with seqused_kv + block_table (that's what the real serve `dsa_v1.py` uses, and what
produces AL 3.94). The single biggest difference is **seqused_kv** (real serve passes the actual KV
length; impl A passes None).

This isolates it: same B=1, KV=133 scenario, compare the compiled op against a fp32 dense+sink
reference under THREE calling conventions:
  1. TND, seqused_kv=None       -> reproduces our harness (expected: ~2e-2 wrong)
  2. TND, seqused_kv=[KV]       -> does telling the op the real length fix the multi-tile path?
  3. PA_ND, paged + seqused_kv  -> the REAL serve convention (paged ori_kv + ori_block_table)
If (2) or (3) matches the reference, the op is CORRECT and impl A's TND/seqused=None wrapper is the bug
(and the fix is small). If none match, the op's multi-tile compute is genuinely broken.

RUN (A3, env dspark-dsv4-*, vllm_ascend with the SAS op built):
    python probe_kv_path.py                 # KV=133 (WIN=128 BS=5), bf16
    WIN=123 BS=5 python probe_kv_path.py     # KV=128 control (all should match)
    DTYPE=float16 python probe_kv_path.py
"""
import os

try:
    import torch
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"!! need torch + torch_npu on an Ascend NPU: {e}")


def _ensure_sas_op():
    try:
        torch.ops._C_ascend.npu_sparse_attn_sharedkv
        return True
    except (AttributeError, RuntimeError):
        pass
    import glob
    import vllm_ascend
    for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so)
            print(f">>> loaded {so}")
        except Exception as e:  # noqa: BLE001
            print(f">>> could not load {so}: {e}")
    try:
        torch.ops._C_ascend.npu_sparse_attn_sharedkv
        return True
    except (AttributeError, RuntimeError):
        return False


if not _ensure_sas_op():
    raise SystemExit("!! SAS op not registered.")

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(os.environ.get("DTYPE", "bfloat16"),
                                                                torch.bfloat16)
H = int(os.environ.get("H", "64")); D = int(os.environ.get("D", "512"))
WIN = int(os.environ.get("WIN", "128")); BS = int(os.environ.get("BS", "5"))
KV = WIN + BS
SCALE = D ** -0.5
WIN_L = WIN + BS - 1   # _dspark_sas_window: ori_win_left
WIN_R = BS - 1         # ori_win_right
META = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
ATTN = torch.ops._C_ascend.npu_sparse_attn_sharedkv


def build():
    q = torch.randn(BS, H, D, device=DEV, dtype=DT)          # one draft block: [T1=BS, N1=H, D]
    kv = torch.randn(KV, D, device=DEV, dtype=DT)            # shared latent, [KV, D]  (num_kv_heads=1)
    sink = torch.randn(H, device=DEV, dtype=DT)
    return q, kv, sink


def reference(q, kv, sink):
    k = kv.unsqueeze(1).expand(-1, H, -1).float()            # [KV, H, D]
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * SCALE
    s = sink[:H].float().view(1, H, 1)
    m = torch.maximum(scores.max(dim=-1, keepdim=True).values, s)
    e = torch.exp(scores - m)
    p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(s - m))
    return torch.einsum("qhk,khd->qhd", p, k).to(q.dtype)    # [BS, H, D]


def cmp(x, ref):
    x, r = x.float(), ref.float()
    d = (x - r).abs()
    return (bool(torch.allclose(x, r, atol=2e-2, rtol=2e-2)), d.max().item(), d.mean().item(),
            (d / (r.abs() + 1e-6)).mean().item())


def line(tag, out, ref):
    if out is None:
        print(f"[{tag:26}] <failed>")
        return
    c, mx, ma, mr = cmp(out, ref)
    print(f"[{tag:26}] allclose={str(c):5}  maxAbs={mx:.2e}  meanAbs={ma:.2e}  meanRel={mr:.2e}")


def op_tnd(q, kv, sink, with_seqused):
    """impl A's _call_dspark_sas_block, TND. with_seqused toggles passing seqused_kv=[KV]."""
    packed = kv.unsqueeze(1).contiguous()                    # [KV, 1, D]
    cu_q = torch.tensor([0, BS], dtype=torch.int32, device=DEV)
    cu_kv = torch.tensor([0, KV], dtype=torch.int32, device=DEV)
    sinks = sink[:H].float().contiguous()
    seqused_kv = torch.tensor([KV], dtype=torch.int32, device=DEV) if with_seqused else None
    meta = META(num_heads_q=H, num_heads_kv=1, head_dim=D, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_kv,
                cu_seqlens_cmp_kv=None, seqused_q=None, seqused_kv=seqused_kv, batch_size=1,
                max_seqlen_q=BS, max_seqlen_kv=KV, cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
                ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="TND",
                has_ori_kv=True, has_cmp_kv=False, device=str(DEV))
    kw = dict(ori_kv=packed, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_kv, sinks=sinks, metadata=meta,
              softmax_scale=SCALE, cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
              ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="TND")
    if with_seqused:
        kw["seqused_kv"] = seqused_kv
    return ATTN(q, **kw)[0]


def op_paged(q, kv, sink):
    """PA_ND paged, the real-serve convention: ori_kv=[num_blocks, block_size, 1, D] + ori_block_table
    + seqused_kv=[KV]."""
    bs_page = 128                                            # page size (16-multiple; matches serve)
    nblocks = (KV + bs_page - 1) // bs_page                  # KV=133 -> 2 pages
    ori_kv = torch.zeros(nblocks, bs_page, 1, D, device=DEV, dtype=DT)
    flat = ori_kv.view(nblocks * bs_page, 1, D)
    flat[:KV] = kv.unsqueeze(1)                              # first KV slots hold the latent, rest 0
    block_table = torch.arange(nblocks, dtype=torch.int32, device=DEV).view(1, nblocks)
    cu_q = torch.tensor([0, BS], dtype=torch.int32, device=DEV)
    cu_kv = torch.tensor([0, KV], dtype=torch.int32, device=DEV)
    seqused_kv = torch.tensor([KV], dtype=torch.int32, device=DEV)
    sinks = sink[:H].float().contiguous()
    meta = META(num_heads_q=H, num_heads_kv=1, head_dim=D, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_kv,
                cu_seqlens_cmp_kv=None, seqused_q=None, seqused_kv=seqused_kv, batch_size=1,
                max_seqlen_q=BS, max_seqlen_kv=KV, cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
                ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND", layout_kv="PA_ND",
                has_ori_kv=True, has_cmp_kv=False, device=str(DEV))
    return ATTN(q, ori_kv=ori_kv, ori_block_table=block_table, cu_seqlens_q=cu_q, seqused_kv=seqused_kv,
                sinks=sinks, metadata=meta, softmax_scale=SCALE, cmp_ratio=1, ori_mask_mode=4,
                cmp_mask_mode=3, ori_win_left=WIN_L, ori_win_right=WIN_R, layout_q="TND",
                layout_kv="PA_ND")[0]


def main():
    q, kv, sink = build()
    ref = reference(q, kv, sink)
    torch.npu.synchronize()
    print(f">>> probe   H={H} D={D} WIN={WIN} BS={BS} KV={KV}  win_left={WIN_L} win_right={WIN_R}  dtype={DT}")
    print(f">>> KV={KV} spans {(KV + 127)//128} tile(s) of 128; reference = dense+sink over {KV} keys\n")

    for tag, fn in [
        ("TND seqused_kv=None (implA)", lambda: op_tnd(q, kv, sink, False)),
        ("TND seqused_kv=[KV]", lambda: op_tnd(q, kv, sink, True)),
        ("PA_ND paged+seqused (serve)", lambda: op_paged(q, kv, sink)),
    ]:
        try:
            out = fn()
            torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"[{tag:26}] EXCEPTION: {type(e).__name__}: {str(e)[:120]}")
            continue
        line(tag, out, ref)

    print("\n>>> read: if 'TND seqused_kv=[KV]' or 'PA_ND ...' matches (~1e-2 bf16) but 'seqused=None' "
          "does not,\n    the op is CORRECT and impl A's TND/seqused=None call is the bug (small fix). "
          "If none match -> op multi-tile is genuinely broken.")


if __name__ == "__main__":
    main()
