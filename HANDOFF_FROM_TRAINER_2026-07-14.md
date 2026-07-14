# Handoff from the DSV4-DSpark trainer session — 2026-07-14

From the main trainer work (`Sawyer117/speculators @ feat/dsv4-dspark`). Two items: (1) your
causal-127 SAS build diagnosis is **sound — no correction**; (2) a `block_size` number changed
under you today (an off-by-one fix), so your README §3 is now slightly stale.

## 1. Your PROD-vs-REF (causal-127) diagnosis holds — confirmed, no change

Reviewed `diag_sas_window.py` + `fused_sas_vs_reference_parity.py`. The method is right and the
conclusion stands:

- PROD (compiled SAS op) and **both** reference windows carry the **same sink** (diag lines
  105-108), so the ~1.39 gap is **not** the sink — it's a genuine op discrepancy, and testing PROD
  against `REF_noncausal(134/6)` vs `REF_causal(127/0)` isolates causal-vs-noncausal unambiguously.
- Your Q2 answer is correct: if this node's `.so` is the upstream causal op (the
  `sparse_attn_sharedkv_tiling.cpp:1365` `oriWinLeft==127 / oriWinRight==0` asserts), it computes
  causal-127 → the **inference** SAS path on this node is also wrong. It's a **build** issue, not a
  training-specific one. Rebuild vllm-ascend from the `dspark-dsv4` commit that adds `win_right>0`
  (the one relaxing those two asserts).
- Your Triton kernel matching `REF_noncausal` at fp32 (5.96e-7) is the right target and is
  untouched by any of this.

(For the record: an earlier claim from my side that "the 1.39 is the sink" was about a *different*
script — the speculators-repo `dspark_attn_ref_bench.py`, whose `sdpa_nosink` baseline drops the
sink. In **your** harness the sink is constant, so the 1.39 is the op. Your framing was right.)

## 2. DSV4 `block_size` changed 5 → 6 today (off-by-one fix) — update README §3

Your README §3 lists `block_size = 5 (config) / 7 (block7 ckpt)`. The **config value is now 6**,
and the full train/infer picture (they differ by exactly the anchor) is:

| side | field | value | meaning | window `win_left/win_right` | KV = window+block |
|---|---|---|---|---|---|
| **training** (speculators — your kernel's path) | `--block-size` / `DSV4DSparkConfig.block_size` | **6** | block WIDTH = anchor(slot 0) + γ drafts | **133 / 5** (=128+6−1, 6−1) | **134** |
| **inference** (vllm-ascend SAS op) | `dspark_block_size` = `num_speculative_tokens` | **5** | γ = drafted-token count | **132 / 4** | 133 |

Why it matters for the kernel:
- **Training feeds the attention 6 query positions per block**: `[anchor, m1..m5]`. slot 0 (anchor)
  is loss-masked, but it IS a query (output computed, then discarded) **and** a key the 5 drafts
  attend to. So your **training kernel's block = 6, window 133/5, KV = 134** — not 5, not 7.
- Inference generates 5 tokens with the anchor sitting as the last of the 128-token context, so the
  SAS block = 5 → 132/4. Both describe the **same 5-draft attention** (the anchor is a visible key
  either way), and your kernel is block-agnostic, so nothing is "broken" — but:
  - **README §3:** config `block_size` is **6** now (was 5); the **training** block is **6**.
  - **DSV4 training parity scenario:** run at **`BS=6`** (win 133/5), not `BS=7` (134/6 = the Qwen3
    block7 ckpt) and not `BS=5` (that's the inference view).
  - The fork's non-causal SAS patch must accept **`win_right ∈ {4 (infer), 5 (train)}`**, not only 6.

### Provenance for the 5 ↔ 6 (so you can verify, not take my word)
- Training `speculative_tokens = block_size − 1`: speculators `models/dflash/core.py:186-188`
  (comment *"First block position is the anchor, not emitted during gen."*); DSpark's own forward
  masks the anchor slot: `models/dsv4_dspark/core.py:343,397`, `models/dspark/metrics.py:88,152`.
- Inference `n_predict = dspark_block_size` (no −1): vllm-ascend
  `patch/platform/patch_speculative_config.py:21`; test `tests/ut/spec_decode/test_dspark_config.py:36`
  asserts `n_predict == 5`.
- Position assignment: `models/dflash/utils.py::get_base_indices_for_anchored_blocks` — block=6 gives
  anchor@A + drafts@A+1..A+5, i.e. exactly the 5 positions inference generates (block=5 would give
  A+1..A+4 = only 4 → the off-by-one bug we just fixed).
- Cross-checked on Qwen3: upstream trains `BLOCK_SIZE=8`
  (`examples/train/dspark_qwen3_0_6b_sharegpt_online.sh:36`) ⇔ released
  `deepseek-ai/dspark_qwen3_4b_block7/config.json` `block_size=7` (Δ=1=anchor).

Pull `feat/dsv4-dspark` for the updated `DSV4DSparkConfig.block_size=6` (commit `5834c9b`). Full
provenance for the whole draft (architecture, block, HS) is in
`docs/deployment/ascend-npu-dsv4-dspark-ep-training.md` §1.5 + §10 on that branch.
