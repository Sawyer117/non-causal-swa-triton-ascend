# Status for review — SWA non-causal-sink Triton kernel + the PROD≠REF investigation (2026-07-14)

**Audience:** a reviewing AI/engineer. Every non-obvious claim is tied to a `file:line` or an actual
A3 run number so you can re-check, not trust. Two of my intermediate root-cause calls were WRONG and
are marked RETRACTED — please scrutinize the reasoning, not just the final claim.

Repo: `Sawyer117/non-causal-swa-triton-ascend`, branch `main`. Relevant commits this session:
`f9baf9a → 4bb9581 → f33be6a → 731cac0 → 90510ee → d5153ce → 6de0081` (latest).

---

## TL;DR

1. **Our Triton kernel is DONE and independently PROVEN correct.** On the real A3, shared-KV/MLA
   scenario, `OURS vs REF = allclose=True, meanAbs 1.53e-4` (bf16) — i.e. our kernel matches
   vllm_ascend's own `_dspark_attention_reference` to ~1e-4. fp32 parity elsewhere is 5.96e-7.
2. **The COMPILED production op (`npu_sparse_attn_sharedkv`) is wrong on THIS node.** Same scenario,
   same inputs: `PROD vs REF = allclose=False, meanAbs 1.60e-2, maxAbs 1.05` — ~100× worse than our
   bf16 kernel. Since a *correct* bf16 op would match REF like ours does, the op genuinely disagrees
   with its own reference.
3. **Ruled out** as the cause of (2): bf16 precision, the causal-vs-noncausal window, and the test
   harness. What remains is the **built `.so` on this node** (build provenance / a partial kernel).
4. This is a **vllm-ascend build question, not a question about our kernel.** The open item is for the
   node owner to verify/rebuild; our kernel needs nothing.

Two things a reviewer should independently double-check: **(A)** is the shared-KV scenario a faithful
model of the real op usage (so that "PROD≠REF" is meaningful)? **(B)** is there any chance REF itself
is not the op's intended semantics (i.e. could PROD be right and REF+OURS both wrong)? My analysis of
both is in §5.

---

## 1. What the operator is (the intended math)

Per-draft-block attention for DSpark/DSV4 speculative decoding. Each block's `block_size` query tokens
attend **densely** to `[ window context tokens | the whole block ]` (non-causal within the block),
with a per-head fp32 **sink** logit added to the softmax denominator (and dropped from `P@V`).
MLA: `num_kv_heads=1` — one KV latent shared across all H query heads.

- Reference (ground truth): `vllm_ascend/ops/dspark_attention.py::_dspark_attention_reference`
  (lines 75-87). fp32 internals; sink at line 86: `probs = e / (e.sum() + exp(sink - max))`.
- Window as an equivalent packed-SWA mask: `_dspark_sas_window(block, window)` (lines 32-37) →
  `win_left = window + block − 1`, `win_right = block − 1` (asymmetric, non-causal).
- Real DSV4 shapes: `H=64, D=512, window=128`. **block_size = 6 (training) / 5 (inference)** — see §6.

## 2. Our kernel — status: DONE, validated

- Files: `triton_impl/swa_sink_ascend.py` (fwd, D-tiled), `swa_sink_ascend_bwd.py` (bwd). CUDA
  baseline untouched (separate files, per the hard constraint).
- Correctness (real A3): `fused_sas_vs_ours.py` fp32 parity **5.96e-7**; `ours_vs_production.py`
  bf16 `OURS vs REF meanAbs 1.53e-4, maxAbs 7.81e-3, allclose=True`.
- Speed: forward ~0.5 ms; fwd 1.63× vs eager; fwd+bwd 0.85× (bwd is the honest weak spot).
- The kernel is **block-agnostic** — block_size 5/6/7 all work; validate at the geometry you mean.

## 3. The PROD≠REF investigation — evidence and decision trail

### 3.1 The original symptom
`fused_sas_vs_reference_parity.py` (forces the SAS op vs the reference loop through the same entry)
reported `allclose=False, maxAbs 1.39, meanRel 7.65` at BS=7.

### 3.2 RETRACTED call #1 — "the op computes a causal-127 window"
I read `csrc/.../sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp:1365,1368` in
`/workspace/drkernel-verl/.refsrc/...` and found it asserts `oriWinLeft_ != 127 / oriWinRight_ != 0`
(causal-only). I concluded the built op was causal. **Wrong**, because:
- That `.refsrc` tree is **upstream**. The **fork** (`va-src`/`va_fix`/`/tmp/user-va`, =
  `Sawyer117/vllm-ascend`) at the **same file:line reads `< 0 / < 0`** (non-causal; PR #11196 relaxed
  the asserts AND made the kernel honor `win_right`). Verified across every csrc tree on disk:
  fork = `< 0`, upstream (`.refsrc`, `/workspace/vllm-ascend`, `vllm-ascend-gh`, `mohammad-speculator`)
  = `!= 0`. (There are TWO SAS ops; the causal-only one is the **kv-quant** variant
  `kv_quant_sparse_attn_sharedkv_check_feature.cpp:27,31`, "当前不泛化", and the draft does NOT
  dispatch to it — `dspark_attention.py:140` uses the non-quant `npu_sparse_attn_sharedkv`.)
- The user's decisive counter-argument: real dspark inference gets good acceptance length, which a
  causal-windowed op could not produce.

### 3.3 The REAL harness bug (found, fixed — this was most of the 1.39)
`build_scenario` generated **per-head-independent** K (`randn(n, H, D)`) but ran `shared_kv=True`.
The SAS path reads **only head 0** (`dspark_attention.py:246` `k_ctx[:, :1, :]`, num_kv_heads=1) and
broadcasts it to all H query heads, while `_dspark_attention_reference` uses each head's own K. So
REF and FUSED disagreed on H−1 of H heads **by construction**, independent of the op. Fixed: the
scenario now builds one KV latent and broadcasts it (`randn(n,1,D).expand(n,H,D)`), matching MLA and
matching `ours_vs_production.py` (which was already shared-KV). After the fix, `meanAbs` dropped
**0.15 → 0.0198** — the per-head bug was the bulk of the old 1.39.

### 3.4 RETRACTED call #2 — "the residual is just the op's internal bf16 precision"
After the harness fix, BS=5 bf16 still showed `maxAbs 0.68, meanAbs 0.0198`. I hypothesized it was
the op's bf16 Cube precision vs the fp32 reference. **Wrong**, disproved three ways:
- **fp16 == bf16 error** (`maxAbs 6.81e-1` vs `6.82e-1`, `meanAbs 1.98e-2` both). Rounding would
  shrink ~8× at fp16; it didn't → the residual is **precision-independent**.
- A **bf16-internal reference** (round Q/K/V and P to bf16, fp32 accumulate + fp32 softmax/sink;
  `diag_sas_window.py::masked_reference(prec="bf16")`) gives the **same 0.682** as the fp32 oracle →
  bf16 rounding does not explain the gap.
- **Our own bf16 kernel matches REF to 1.53e-4** while PROD is 1.60e-2 off. A correct bf16 op would
  match REF like ours; PROD is ~100× worse → not a precision effect.

### 3.5 What's ruled OUT vs what's OPEN
- OUT — **causal window**: diag shows PROD closer to non-causal (`0.682`) than causal (`0.757`).
- OUT — **bf16 precision**: §3.4.
- OUT — **the harness/scenario**: `OURS vs REF = 1.53e-4` proves the scenario is sound and two
  independent implementations (our kernel + vllm's reference) agree on the intended math.
- OPEN — the **compiled `.so` on this node** (`/home/a00652497/dspark_austin/installation/
  vllm-ascend-v4/`): a stale/wrong/partial build, or a real op bug for `num_kv_heads=1` + this
  window. PROD is "non-causal-ish but 0.68–1.05 off from the exact non-causal reference," and
  precision-independent.

## 4. Exact A3 numbers (all BS=5 unless noted; commit `6de0081`)

```
# fused_sas_vs_reference_parity.py  (NBLK=8)
  bf16   [fused vs ref]  allclose=False  maxAbs=6.82e-1  meanAbs=1.98e-2  meanRel=1.27
  fp16   [fused vs ref]  allclose=False  maxAbs=6.81e-1  meanAbs=1.98e-2  meanRel=1.33   # == bf16
  fp32   -> op has NO fp32 kernel (AclNN_Parameter_Error; supported dtypes fp16/bf16) -> op FALLS
          BACK to the torch ref -> FUSED==REF -> a meaningless allclose=True maxAbs=0.00 (now guarded)

# diag_sas_window.py  (NBLK=8, bf16)
  [sanity ref_nc vs entry ]  maxAbs=0.00       # my non-causal ref == the entry's reference loop
  [PROD vs ref_noncausal fp32] maxAbs=6.82e-1  meanAbs=1.98e-2
  [PROD vs ref_causal127 fp32] maxAbs=7.57e-1  meanAbs=2.48e-2   # WORSE than non-causal
  [PROD vs ref_noncausal bf16] maxAbs=6.82e-1  meanAbs=1.98e-2   # == fp32 -> not bf16

# ours_vs_production.py  (NBLK=64, bf16)  <-- the decisive one
  [parity] OURS vs REF   allclose=True   maxAbs=7.81e-3  meanAbs=1.53e-4  meanRel=9.78e-3
  [parity] OURS vs PROD  allclose=False  maxAbs=1.05e+0  meanAbs=1.60e-2  meanRel=1.08
  [parity] PROD vs REF   allclose=False  maxAbs=1.05e+0  meanAbs=1.60e-2  meanRel=1.03
  [speed]  production=127.175ms  ours=0.500ms   # 127ms is a per-block PYTHON dispatch loop (see note)
```
Note on the "254×": the "production" time is the vllm_ascend entry = a per-block Python loop calling
the op once per block (dispatch-bound), NOT the op's fused compute. It is **not** a fair kernel-vs-
kernel number. (The script now says so and no longer prints a hardcoded "allclose=True" success.)

## 5. Two things the reviewer should independently pressure-test

- **(A) Is the shared-KV scenario faithful?** The real model is MLA (`num_kv_heads=1`); the scenario
  broadcasts one latent to all heads and feeds the op the same per-block `[window ctx | block]` KV
  the reference gathers. `OURS vs REF = 1.5e-4` shows the scenario is at least self-consistent across
  two implementations. Possible gap: real inference may route through the **generic**
  `torch.ops._C_ascend.dspark_attention` custom op, which the entry tries FIRST
  (`dspark_attention.py:290`) before the SAS path — and which is **not registered** in the csrc I
  inspected. If a production build has it, inference never hits the SAS op, which would explain
  "inference is fine" while the SAS path is broken. Worth confirming on the inference node:
  `hasattr(torch.ops._C_ascend, 'dspark_attention')`.
- **(B) Could PROD be right and REF wrong?** REF is the op's OWN documented reference
  (`_dspark_attention_reference`, "the Python reference for the future Ascend C kernel"), and our
  independent Triton kernel agrees with it to 1.5e-4. Two independent implementations pinning the same
  math, with PROD the outlier, is strong evidence PROD (or its build) is the wrong one. But if the
  reviewer knows of an intended semantic (e.g. a sink scaling, a V transform, a different diagonal
  alignment) that BOTH REF and OURS miss, that would flip it — please check.

## 6. block_size correction (independent of the above; already applied)

`feat/dsv4-dspark` @ `5834c9b` bumped `DSV4DSparkConfig.block_size` **5 → 6** (off-by-one). The three
regimes describe the same 5-draft attention, differing by the anchor (slot 0: loss-masked, but IS a
query and a key):

| context | block | win_left/win_right | KV |
|---|---|---|---|
| training (our kernel's path) | **6** = anchor + 5 drafts | 133 / 5 | 134 |
| inference (vllm-ascend SAS)  | **5** = γ = num_spec     | 132 / 4 | 133 |
| Qwen3 block7 ckpt (other model) | 7 | 134 / 6 | 135 |

README §3/§4 and `fused_sas_vs_ours.py` default (BS=6) updated; PROD-comparison scripts note BS=5 is
the inference-faithful value.

## 7. Recommended next actions (node owner / vllm-ascend side)

1. Build provenance on the **actually built** tree:
   `grep -n 'oriWinRight_' /home/a00652497/dspark_austin/installation/vllm-ascend-v4/csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp`
   → `!= 0` = built the upstream causal op (rebuild from the fork); `< 0` = source is the fork
   (non-causal) but the binary still misbehaves → clean rebuild (`rm -rf build csrc/build && pip
   install -e . --no-deps --no-build-isolation`), suspect a stale/partial kernel object.
2. `BS=5 python diag_sas_window.py` → the new per-block-position breakdown (`p0..p4`): error
   concentrated at early positions (which need FUTURE block tokens via `win_right`) ⇒ the `win_right`
   kernel path is the broken part.
3. `diff -r` this node's `csrc/attention/sparse_attn_sharedkv` against the known-good inference node.

Our kernel requires no changes in any branch of the above.
