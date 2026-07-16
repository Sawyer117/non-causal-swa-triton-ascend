# Handoff — DSV4-DSpark bf16 benchmark, the eager memory-opt, and kernel/op status (2026-07-16)

For a reviewing AI (esp. the training/speculators side). All numbers are real A3 runs; every claim →
`file:line`. TL;DR at top, then the one high-value action (the eager mem-opt), then the full picture.

## TL;DR

1. **HIGH-VALUE, DO BEFORE TRAINING RESTART — optimize the training eager to the shared MLA latent.**
   The current `_sink_block_attention_torch` expands the KV latent to per-head `[N,KV,H,D]` and then
   `.float()`s it (materialising ~2.2 GB of fp32 K+V). Using the shared latent `[N,KV,D]` directly is
   **bit-identical output (0.00e+00 diff), saves ~2.1 GB, and is ~20× faster**. Free win. Change =
   `speculators/.../dsv4_dspark/backbone/attention.py:147` (drop the `.expand`) + the two einsums in
   `_sink_block_attention_torch:54,62` (`nkhd`→`nkd`). Details in §2.
2. **Our Triton kernel ≡ the production op** (bit-equal, ~1e-7) and is correct fwd+bwd; but on
   fwd+bwd it's ~0.94× vs the eager baseline (the **backward is the weak spot**). §4.
3. **The compiled op is CORRECT in its real (PA_ND paged) path**; the earlier ~2e-2/NaN was our harness
   using the op's **TND** wrapper, which is broken for KV>128. §3.

## 1. The bf16 benchmark (bench_3way.py, `main @ d4e94cc`)

Baseline = the PROD fused op (what the vllm-ascend engine calls). Everything bf16/fp16; precision is
**vs PROD** (no fp32 gold). Real run, DSV4 shape `BS=5 KV=133 blocks=64`, **fp16**:

```
impl                   |    maxAbs   meanAbs   meanRel |   fwd ms |  peak MB
BASELINE prod op       |          (baseline)           |    0.096 |   1186.4
case1 train-eager      |  4.88e-04  1.91e-05  1.22e-03 |   14.116 |   3366.8   <- per-head expand (heavy)
case2 eager mem-opt    |  4.88e-04  1.91e-05  1.22e-03 |    0.721 |   1276.3   <- shared latent (light)
case3 reference-eager  |  4.88e-04  1.91e-05  1.22e-03 |   50.358 |   1256.8   <- PR _dspark_attention_reference
case4 ours triton      |  4.88e-04  1.02e-07  5.38e-06 |    0.468 |   1250.5   <- our fused kernel
case1 vs case2         |  0.00e+00  0.00e+00  0.00e+00                          <- mem-opt = bit-identical
```
Reads: all four sit at the fp16 output floor (`maxAbs 4.88e-4`); **case4 (ours) is ~1e-7 vs PROD =
bit-equal to the op**. **case1 vs case2 = 0.0** (the mem-opt does not change the output). At bf16 the
floor is ~8× looser (~4e-3 maxAbs) but the relations are identical.

Precision note (op-veteran-verified): bf16-vs-fp32 CANNOT be 1e-6 — bf16 has an 8-bit mantissa (~4e-3
rel) and the output is bf16-stored, so `maxAbs ~4e-3` = one bf16 ULP = the floor; the `~1% meanRel` is
small-output-denominator inflation (present in EVERY correct row). A `1e-2` *meanAbs* would be a real
bug — that's what flagged the broken TND path (§3).

## 2. THE EAGER MEMORY-OPT (do this before restarting training)

Training's attention core: `speculators/src/speculators/models/dsv4_dspark/backbone/attention.py`.
- `forward:146-148`:
  ```python
  kv = torch.cat([kv_ctx, kv_blk], dim=1)                       # [N, KV, D]  shared MLA latent (1 head)
  kv = kv.unsqueeze(2).expand(-1, -1, self.num_heads, -1)       # <-- expand to [N, KV, H, D]  (the cost)
  o  = sink_block_attention(q, kv, kv, self.attn_sink, self.scale, attn_bias)
  ```
- `_sink_block_attention_torch:54,62`: `einsum("nqhd,nkhd->nqhk", q.float(), k.float())` and
  `einsum("nqhk,nkhd->nqhd", p, v.float())`. The `k.float()`/`v.float()` on the *expanded* view
  MATERIALISE `[N,KV,H,D]` fp32 for K AND V (~1.1 GB each at N=64,KV=133,H=64,D=512).

**Fix (bit-identical for MLA, since every head shares the same latent):**
```python
# forward: drop the expand, pass the shared latent
kv = torch.cat([kv_ctx, kv_blk], dim=1)                          # [N, KV, D]
o  = sink_block_attention(q, kv, kv, self.attn_sink, self.scale, attn_bias)

# _sink_block_attention_torch: accept 3-D (shared) K/V, keep 4-D path for other callers
if k.dim() == 3:  # [N, Sk, D]
    s = torch.einsum("nqhd,nkd->nqhk", q.float(), k.float()) * scale
    ...
    return torch.einsum("nqhk,nkd->nqhd", p, v.float()).to(q.dtype)
else:             # [N, Sk, H, D]  (unchanged)
    s = torch.einsum("nqhd,nkhd->nqhk", q.float(), k.float()) * scale
    ...
```
**Proven safe & valuable (bench §1):** output bit-identical (`case1 vs case2 = 0.0`), peak MB
`3366.8 → 1276.3` (−2.1 GB), fwd `14.1 → 0.72 ms` (~20×). Backward is also identical — autograd sums
the gradient over heads whether K is expanded (a view) or shared. Keep the **fp32 accumulation**
(training wants the accurate reference); only the K/V materialisation shape changes.

## 3. The compiled op: PA_ND correct, TND broken (why our parity looked wrong for a while)

- The op `npu_sparse_attn_sharedkv` has two call conventions. The real serve (`dsa_v1.py:1976`) uses
  **PA_ND** (paged `ori_kv` + `ori_block_table` + `seqused_kv`). Our first harness used the op's **TND**
  eager wrapper `_call_dspark_sas_block` (`ops/dspark_attention.py:146`, `layout_kv="TND"`,
  `seqused_kv=None`).
- **TND is broken for KV>128**: `probe_kv_path.py` at KV=133 → `TND seqused=None` and `TND seqused=[KV]`
  both **NaN**; **PA_ND paged = allclose=True, maxAbs 3.91e-3 (bf16 floor)**. KV=128 (single 128-tile):
  all fine. So the op's multi-tile path is only correct via PA_ND. This resolves "the op can't be
  broken, inference gets AL 3.94": inference uses PA_ND (correct); our TND harness hit the bug.
- ⚠️ The default env `VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=0` routes the draft model's attention through
  that eager TND path (`deepseek_v4_dspark.py:361` else → `_run_dspark_attention` → `dspark_attention`)
  → a default-config DSpark serve would NaN at KV=133. **Serve via PA_ND (`=1`) or fix the TND path.**
- `ours_vs_production.py` now compares against the op via **PA_ND, batched** (fair op-vs-op); it prints
  `OURS vs PROD allclose=True meanAbs 1.26e-7` = our kernel ≡ the op.

## 4. Our Triton training kernel — status (fused_sas_vs_ours.py, BS=6 DSV4: block=6, win 133/5, KV=134)

- **Correct**: fwd fp32 `7.15e-7`; bwd fp32 dq/dk/dv/dsink all `~1e-6`; sink verified fwd+bwd.
- **fwd**: eager 0.684 ms, ours 0.429 ms → **1.60× faster**; mem 234 → 171 MB (**1.37× less**).
- **fwd+bwd**: eager 2.845 ms, ours 3.036 ms → **0.94×** (ours bwd ≈ 2.61 ms vs autograd bwd ≈ 2.16 ms
  → our **backward is ~1.2× slower**; it recomputes qk flash-style while autograd stores the scores).
  mem 395 → 345 MB (**1.14× less**).
- So the kernel is **correct + memory-lean + forward-faster, but ~parity on fwd+bwd** — the backward is
  the bottleneck. NOTE the Triton kernel IS forward-only faster than the eager but ~5× SLOWER than the
  PROD op on forward (op = hand-tuned AscendC; but the op has NO autograd, so training can't use it).

## 5. Roles (don't conflate — this caused confusion)

| thing | what | role |
|---|---|---|
| PROD op (`npu_sparse_attn_sharedkv`) | engine's fused op, PA_ND | INFERENCE baseline (no autograd → training can't use it) |
| our training eager (`_sink_block_attention_torch`) | fp32-accum torch attn (autograd) | what TRAINING uses now (optimize per §2) |
| ours triton (`swa_sink_ascend*.py`) | our fused kernel (fwd fast, bwd weak) | our contribution for training |

## 6. Actionables (ordered)

1. **[training] Apply the eager shared-latent mem-opt (§2)** — bit-identical, −2.1 GB, ~20× fwd, before
   restart. Kernel-session offered to make the speculators edit; awaiting go.
2. **[kernel] Optimize the backward** (§4) — the only thing keeping fwd+bwd at 0.94×. Tile sweep
   (`BMDQ/BKDQ/BMKV/BKV` in fused_sas_vs_ours.py) or store P/scores to skip the flash recompute at tiny KV.
3. **[serve] TND NaN heads-up (§3)** — serve DSpark via PA_ND (`VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=1`)
   or fix the op's TND multi-tile path; the default eager TND path NaNs at KV>128.

Files: `bench_3way.py`, `ours_vs_production.py` (prod_prep/prod_call = batched PA_ND), `probe_kv_path.py`,
`fused_sas_vs_ours.py`, `AUDIT_pr11196_op_2026-07-16.md`. Repo `main @ d4e94cc`.
