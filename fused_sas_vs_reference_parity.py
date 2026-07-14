#!/usr/bin/env python3
"""LEAD (for the kernel author): measure the FUSED SAS op vs the torch reference.

WHY THIS EXISTS
---------------
We have a recorded number for **our einsum+sink impl vs the torch reference**
(`_dspark_attention_reference`): meanAbs = meanRel = 0.00e+00, BIT-EXACT (A3,
2026-07-07, `reference_from_repo/dspark_attn_ref_bench.py`). We do NOT have a
recorded per-element error for the actual **compiled SAS fused kernel**
(`torch.ops._C_ascend.npu_sparse_attn_sharedkv`, the AscendC op in
`vllm_ascend/csrc/attention/sparse_attn_sharedkv/`) vs that reference — only a
binary "SAS-vs-PTA parity pass" from vllm-ascend PR #11196 and the end-to-end
inference match (DSV4-Flash DSpark AR 58.79% / AL 3.94 vs GPU ref AL 3.86).

This script produces that missing number. It also doubles as the template for
validating a NEW Triton kernel against the SAME fused op / reference: swap
`attn_fused` for your `swa_sink_attn_triton(...)` and keep the reference side.

WHAT IT DOES
------------
Builds a realistic paged-context scenario at the REAL DSV4 shapes
(H=64, D=512, window=128, block=7 — see ../README.md §3), then compares, on
IDENTICAL inputs, two runs of the public `dspark_attention(..., shared_kv=True)`
entry (`vllm_ascend/ops/dspark_attention.py`):
  * FUSED : generic custom op disabled -> routes to the SAS fused kernel
            (`npu_sparse_attn_sharedkv`).
  * REF   : both fused paths disabled  -> the per-block `_dspark_attention_reference`
            fallback loop (fp32 internals).
Any difference is therefore purely **fused kernel vs reference**. Reports
allclose / meanAbs / meanRel / maxAbs. FORWARD ONLY (the inference op has no autograd).

READING THE RESULT
------------------
- bf16 (default): a diff ~1e-2 vs the fp32 reference is EXPECTED bf16 rounding.
  Run `DTYPE=float32` and it should collapse to ~1e-6 (proving it's dtype, not a
  math bug). If it does NOT collapse, the fused kernel and the reference disagree.
- If the SAS op is not compiled/available, the script says so and STOPS (it will
  not print a misleading "0.0" from comparing the fallback to itself).

RUN (A3 NPU, env `dspark-dsv4-base`, vllm_ascend with the SAS kernel built):
    python fused_sas_vs_reference_parity.py
    DTYPE=float32 python fused_sas_vs_reference_parity.py   # diffs should collapse ~1e-6
    NBLK=64 WIN=128 BS=7 python fused_sas_vs_reference_parity.py
"""
import os

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
    raise SystemExit(
        f"!! cannot import vllm_ascend.ops.dspark_attention: {e}\n"
        "   run on a node with vllm_ascend installed (the A3), env dspark-dsv4-base."
    )


def _ensure_sas_op():
    """editable vllm-ascend installs don't always auto-load the compiled vllm_ascend_C.so, so the SAS
    op isn't registered even though it's built. Load it explicitly if missing."""
    try:
        torch.ops._C_ascend.npu_sparse_attn_sharedkv; return
    except (AttributeError, RuntimeError):
        pass
    import glob
    import vllm_ascend
    for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so); print(f">>> loaded vllm_ascend C-ext: {so}")
        except Exception as e:  # noqa: BLE001
            print(f">>> could not load {so}: {type(e).__name__}: {e}")


_ensure_sas_op()

torch.manual_seed(0)
DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
    os.environ.get("DTYPE", "bfloat16"), torch.bfloat16)
ATOL = float(os.environ.get("ATOL", "2e-2"))
RTOL = float(os.environ.get("RTOL", "2e-2"))
# REAL DeepSeek-V4-Flash-DSpark attention shapes (HF config.json; see ../README.md §3)
H = int(os.environ.get("H", "64"))       # heads
D = int(os.environ.get("D", "512"))      # head_dim
WIN = int(os.environ.get("WIN", "128"))  # sliding_window
BS = int(os.environ.get("BS", "7"))      # 7=Qwen3-block7 ckpt; DSV4: infer block=5 (win 132/4),
#                                          train block=6 (win 133/5) — see README §3. Diagnosis is
#                                          BS-agnostic (causal-vs-noncausal); override with BS=5/6.
NBLK = int(os.environ.get("NBLK", "8"))  # draft blocks (one request slot each)
SCALE = D ** -0.5
CAP = 1 << (max(WIN + BS, 1) - 1).bit_length()  # cache capacity, next pow2 >= win+block


def build_scenario():
    """One draft block per request slot; each block's context = the full `WIN`
    tokens before it. Cache layout mirrors vllm_ascend's own entry-vs-reference UT
    (test_dspark_attention_entry_matches_reference_with_request_cache)."""
    base = WIN                       # block starts at position WIN -> context = [0, WIN-1]
    n = NBLK * BS
    q = torch.randn(n, H, D, device=DEV, dtype=DT)
    # MLA (num_kv_heads=1): the KV latent is SHARED across all H query heads. Generate ONE head and
    # broadcast. (Bug fixed 2026-07-14: this used to be randn(n,H,D) — per-head-INDEPENDENT K. With
    # shared_kv=True the SAS op only reads head 0 [dspark_attention.py:246 `k_ctx[:, :1, :]`] while
    # the reference uses each head's own K, so per-head K made REF!=FUSED by construction — maxAbs
    # ~1.39 that had NOTHING to do with the op. Real inference shares the latent, hence good AL.)
    draft_k = torch.randn(n, 1, D, device=DEV, dtype=DT).expand(n, H, D).contiguous()
    attn_sink = torch.randn(H, device=DEV, dtype=DT)

    positions = torch.empty(n, dtype=torch.int32, device=DEV)
    request_slots = torch.empty(n, dtype=torch.int32, device=DEV)
    cache_k = torch.zeros(NBLK, CAP, H, D, device=DEV, dtype=DT)
    cache_positions = torch.full((NBLK, CAP), -1, dtype=torch.int32, device=DEV)
    cache_valid = torch.zeros(NBLK, CAP, dtype=torch.bool, device=DEV)

    ctx_pos = torch.arange(base - WIN, base, device=DEV)     # [0, WIN-1]
    idx = (ctx_pos % CAP).long()
    for b in range(NBLK):
        sl = slice(b * BS, (b + 1) * BS)
        positions[sl] = torch.arange(base, base + BS, dtype=torch.int32, device=DEV)
        request_slots[sl] = b
        cache_k[b, idx] = torch.randn(WIN, 1, D, device=DEV, dtype=DT).expand(WIN, H, D)  # shared latent
        cache_positions[b, idx] = ctx_pos.to(torch.int32)
        cache_valid[b, idx] = True
    # shared_kv=True => the entry uses k as v; pass k for the v args to satisfy the API.
    return dict(q=q, k_cache=cache_k, v_cache=cache_k, cache_positions=cache_positions,
                cache_valid=cache_valid, draft_k=draft_k, draft_v=draft_k,
                request_slots=request_slots, positions=positions, attn_sink=attn_sink,
                block_size=BS, window_size=WIN, softmax_scale=SCALE)


def _run(s):
    return dspark_attention(
        s["q"], s["k_cache"], s["v_cache"], s["cache_positions"], s["cache_valid"],
        s["draft_k"], s["draft_v"], s["request_slots"], s["positions"], s["attn_sink"],
        s["block_size"], s["window_size"], s["softmax_scale"], shared_kv=True,
    )


class _override:
    """Temporarily replace the op getters on the module to steer the entry's path."""
    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(dsa, k)
            setattr(dsa, k, v)
    def __exit__(self, *_):
        for k, v in self.saved.items():
            setattr(dsa, k, v)


def compare(x, ref):
    xf, rf = x.float(), ref.float()
    d = (xf - rf).abs()
    return dict(
        allclose=bool(torch.allclose(xf, rf, atol=ATOL, rtol=RTOL)),
        maxAbs=d.max().item(), meanAbs=d.mean().item(),
        meanRel=(d / (rf.abs() + 1e-6)).mean().item(),
    )


def main():
    s = build_scenario()
    mode, wl, wr = _dspark_sas_window(BS, WIN)
    print(f">>> fused SAS op vs torch reference   H={H} D={D} win={WIN} block={BS} "
          f"blocks={NBLK}  dtype={DT}")
    print(f">>> _dspark_sas_window(block={BS}, window={WIN}) = mask_mode={mode}, "
          f"win_left={wl}, win_right={wr}  (asymmetric; not window-1)")

    # Is the compiled SAS kernel actually present? (else the number would be trivial)
    sas_available = dsa._get_dspark_sas_ops(s["q"]) is not None  # noqa: SLF001
    if not sas_available:
        print("\n!! SAS fused op NOT available (torch.ops._C_ascend.npu_sparse_attn_sharedkv "
              "not registered).\n   Both runs would fall back to the reference -> the diff "
              "would be a meaningless 0.0.\n   Build vllm_ascend WITH the sparse_attn_sharedkv "
              "kernel on this A3, then re-run.")
        raise SystemExit(2)

    # FUSED: disable only the generic custom op so the entry takes the SAS path.
    with _override(_get_dspark_attention_custom_op=lambda q: None):
        fused = _run(s)
    # REF: disable both fused paths -> per-block _dspark_attention_reference loop.
    with _override(_get_dspark_attention_custom_op=lambda q: None,
                   _get_dspark_sas_ops=lambda q: None):
        ref = _run(s)
    torch.npu.synchronize()

    # FALSE-PASS GUARD: the SAS op has NO fp32 kernel (supports fp16/bf16 only). With DTYPE=float32 it
    # raises AclNN_Parameter_Error -> _maybe_call_dspark_sas_attention catches it and FALLS BACK to the
    # torch reference, so FUSED becomes bit-identical to REF and prints a meaningless 0.0. Detect that.
    if torch.equal(fused, ref):
        print("\n!! FUSED is BIT-IDENTICAL to REF -> the SAS op did NOT run; it fell back to the torch "
              "reference (see the 'DSpark SAS attention failed' warning above). Almost always because "
              "DTYPE=float32 is unsupported (op kernels are fp16/bf16 only). This 0.0 is meaningless.\n"
              "   Use the op's TIGHTEST supported dtype instead:  DTYPE=float16 (10-bit, 8x tighter "
              "than bf16) — that's the clean op-correctness test.")
        raise SystemExit(2)

    r = compare(fused, ref)
    print(f"\n[fused vs ref]  allclose={r['allclose']}  "
          f"maxAbs={r['maxAbs']:.2e}  meanAbs={r['meanAbs']:.2e}  meanRel={r['meanRel']:.2e}")
    print(">>> bf16 ~1e-2 is expected rounding. The op has NO fp32 kernel -> use DTYPE=float16 (8x "
          "tighter) for the clean test; float32 falls back to the reference (a false 0.0).")
    print(">>> if maxAbs stays high at fp16, run diag_sas_window.py to check causal-vs-noncausal.")
    raise SystemExit(0 if r["allclose"] else 1)


if __name__ == "__main__":
    main()
