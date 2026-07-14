# Handoff from the DSV4-DSpark trainer session — 2026-07-14 (AMENDED)

> **This supersedes an earlier same-day version** in which I wrote "your causal-127 diagnosis
> holds." After reading the **actual va-src source**, that was overclaimed. Below is the
> source-verified picture, **with my decision trail** (two corrections) so you can see how I got
> here and check every step yourself.

## TL;DR (calibrated)

- **The SAS op your draft path uses is `non-causal` in OUR fork's source (correct).** #11196 did
  that — the user's doubt about "#11196 fixed SAS" was misplaced on that point.
- Your diag docstring quotes the **upstream** `oriWinRight != 0` (causal) assert; but our va-src at
  that **same file:line is already `< 0`** (patched). So if your node's `.so` computes causal-127,
  it was built from the **wrong/upstream source** (or a partial build) — the source is non-causal.
- The **1.39 is NOT proven** to be a causal window bug. It's either a wrong-source/stale build **or**
  a non-window artifact (dtype path / cache assembly / scenario) — i.e. your diag's own "PROD matches
  non-causal → look elsewhere" branch. **Settle it: run `diag_sas_window.py` at `BS=5`** (DSV4's real
  block; `BS=7` is the Qwen3 block7 case). **Don't default to "causal."**
- Your **Triton kernel is unaffected and correct** (matches the non-causal ref at fp32 5.96e-7).

## Background — what I set out to verify

The main-session user (running the trainer) pushed back, sharply and correctly: *"Is the SAS op
right or wrong, really? Did #11196 'fix' SAS — sounds off? We tested SAS aligns with eager before,
right?"* So I stopped asserting from memory and read the actual source.

## Sources (every claim → file:line — check me, don't trust me)

1. **#11196 IS the non-causal SAS PR.** `dspark_extract/pr11196.diff`:
   ```
   - OP_CHECK_IF(oriWinLeft_  != 127, "ori_win_left should be 127" ...)
   + OP_CHECK_IF(oriWinLeft_  < 0,   "ori_win_left should be non-negative" ...)
   - OP_CHECK_IF(oriWinRight_ != 0,  "ori_win_right should be 0" ...)
   + OP_CHECK_IF(oriWinRight_ < 0,   "ori_win_right should be non-negative" ...)
   ```
   plus kernel `ProcessBalance`/`UpdateInnerLoopCond` changed to use `+ constInfo.oriWinRight`. So
   #11196 relaxes the causal asserts **and** makes the kernel honor `win_right`. (⇒ "#11196
   implemented non-causal SAS" is correct.)

2. **Our fork has TWO SAS ops — one patched, one not:**
   - non-quant `csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp:1365,1368`
     → `oriWinLeft_ < 0` / `oriWinRight_ < 0` = **PATCHED, non-causal** ✅
   - kv-quant `csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_feature.cpp:27,31`
     → `oriWinLeft_ != 127` ("only support 127") / `oriWinRight_ != 0` ("only support 0", comment
     `当前不泛化`) = **still CAUSAL-only** ❌

3. **The draft dispatches to the NON-QUANT op, passing `win_right > 0`:**
   `vllm_ascend/ops/dspark_attention.py:140` → `torch.ops._C_ascend.npu_sparse_attn_sharedkv`;
   `_call_dspark_sas_block` passes `ori_win_right = block_size − 1` from `_dspark_sas_window` (:32-36).
   ⇒ the draft uses the **patched (non-causal)** op; the causal kv-quant op is **not** on the draft path.

4. **The earlier "SAS aligns with eager" test tested the REFERENCE, not the `.so`.**
   `reference_from_repo/dspark_attn_ref_bench.py` benches eager against `_dspark_attention_reference`
   (imported pure-torch), never calling the compiled op. So it validated the **math** (consistent with
   the source being correct); it does **not** test the binary. Your `fused_sas_vs_reference_parity.py`
   is the first test of the actual `.so` — hence the new 1.39.

## My decision trail (the two corrections — so you don't inherit my errors)

- **v1 (wrong):** "the 1.39 is the sink." — that was about the *speculators-repo*
  `dspark_attn_ref_bench.py`, whose `sdpa_nosink` baseline drops the sink. In **your** harness the
  sink is held constant in both refs, so that didn't apply. Retracted.
- **v2 (overclaimed):** "your causal-127 diagnosis holds." — I hadn't yet checked (a) which op the
  draft dispatches to, or (b) our fork's actual assert at that file:line. Retracted.
- **v3 (this doc, source-verified):** the draft uses the non-quant op, which in our fork is already
  `< 0` (non-causal). So the **source is correct**; whether a given `.so` is causal is a **build**
  question, and it is **not proven** — could equally be a non-window artifact.

## Conclusion + action

- **SAS op (source, draft path) = non-causal = correct.** Don't refactor the op's semantics.
- The **PROD-vs-REF 1.39 is a compiled-binary question on that node, unresolved, not proven causal.**
  Two branches:
  1. Node built from the wrong/upstream source (or a partial build) → rebuild vllm-ascend from the
     `dspark-dsv4` commit whose `sparse_attn_sharedkv_tiling.cpp:1365-1369` reads `< 0`.
  2. Non-window artifact (dtype path / cache assembly / scenario) → your diag's alt-verdict.
- **Decisive check:** `BS=5 python diag_sas_window.py` (DSV4's block; not BS=7). And verify the
  built source: `grep -n 'oriWinRight_' .../sparse_attn_sharedkv_tiling.cpp` — `!= 0` = you built the
  upstream causal op; `< 0` = you built ours (non-causal).

## (still valid) DSV4 `block_size` 5→6 + train/infer windows

An off-by-one fix landed on `feat/dsv4-dspark` today: `DSV4DSparkConfig.block_size` **5 → 6**
(block WIDTH incl the anchor; drafts `block_size − 1` = 5). Your README §3 (`block=5`) is now stale.

| side | block | `win_left / win_right` | KV = window+block |
|---|---|---|---|
| **training** (speculators, your kernel's path) | **6** (anchor + 5 drafts) | **133 / 5** | 134 |
| **inference** (vllm-ascend SAS) | **5** (γ = num_spec) | **132 / 4** | 133 |

`134/6` is the Qwen3 **block7** case (`128+7−1`), not DSV4. Provenance for 5↔6: training
`speculative_tokens = block_size − 1` (speculators `models/dflash/core.py:186-188`, comment "First
block position is the anchor, not emitted"); inference `n_predict = dspark_block_size` no −1
(vllm-ascend `patch/platform/patch_speculative_config.py:21`); positions
`get_base_indices_for_anchored_blocks` (block=6 → anchor@A + drafts@A+1..A+5). Full draft provenance:
`docs/deployment/ascend-npu-dsv4-dspark-ep-training.md` §1.5 + §10 on `feat/dsv4-dspark`.
