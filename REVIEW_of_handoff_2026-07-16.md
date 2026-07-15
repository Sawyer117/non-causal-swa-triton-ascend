# Critical review of `HANDOFF_FROM_TRAINER_2026-07-16.md`

Reviewer: kernel session. Method: every claim re-checked against the trees on disk; each finding →
`file:line` in a named tree. Verdict up front, then credit, then the correction with evidence.

## Verdict

The handoff is **factually accurate about the tree it examined** (`/workspace/vllm-ascend @ ee549165`),
but its **central alarm — "the serve may be a causal impl B, so your non-causal parity is moot" — does
NOT hold for our box.** It examined a **pre-#11196 external snapshot** (2026-06-30) that is *older* than
our box's fork (2026-07-03). The box's impl-B code is already **non-causal for the draft**, with the
**same `win_right = block-1`** as impl A. So the non-causal op-parity is on-target for **both** of the
box's draft paths. The handoff's genuinely valuable contribution — step 2, end-to-end `accept_len` as
ground truth — stands and should be done regardless.

## Credit (accurate / valuable)

- Its facts about `/workspace/vllm-ascend @ ee549165` are all verified: no `ops/dspark_attention.py`
  (absent from history), `dsa_v1.py` `ori_win_right=0` at every site, `_dspark_attention_reference`
  absent, and (I add) `get_draft_swa_window` absent (`grep -c` = 0). ✓
- It honestly retracted its first-pass "you're chasing a red herring" (its §4) and explicitly flagged
  the untraced caveat (whether impl B gets block-non-causality via the metadata builder). Good hygiene.
- **Step 2 (run the released DSV4 draft on the actual serve → gsm8k `accept_len` at `num_spec=5`) is the
  right ground truth** and sidesteps all of the impl archaeology. Do it regardless of the below.

## Correction — the core critique (with evidence)

I traced the caveat the handoff left open. It resolves *against* the handoff's headline.

1. **It examined a STALE external tree.** `/workspace/vllm-ascend @ ee549165` is dated **2026-06-30**;
   the box fork `va-src @ 6036507` is **2026-07-03** (newer). Ages via `git log -1 --format=%ci`.
2. **"box = impl A" is incomplete — the box carries BOTH impls, and BOTH draft paths are non-causal.**
   - impl A is *wired into the serve*: `va-src/vllm_ascend/models/deepseek_v4_dspark.py:50` imports and
     `:298`/`:388` calls `dspark_attention(...)` (`_dspark_sas_window` → `win_right = block-1`). Not orphaned.
   - impl B is also present: `va-src/vllm_ascend/attention/dsa_v1.py` + `ops/dsa.py` +
     `spec_decode/dflash_proposer.py` (`causal=False` at `:146,195`; `num_query_per_req = 1 + num_speculative_tokens` at `:81,174`).
3. **The box's impl-B draft is NON-CAUSAL — same geometry as impl A.** This is the caveat, now traced.
   PR #11196 added `va-src/vllm_ascend/attention/dsa_window.py` (the external `ee549165` tree LACKS it,
   `grep -c get_draft_swa_window` = 0). `dsa_v1.py:1195,1281` calls it to set the draft's op window:
   ```python
   # dsa_window.py:24-37
   if not common_attn_metadata.causal and is_dspark_speculative_config(speculative_config):
       block_size = num_speculative_tokens or dspark_block_size
       if block_size > 0:
           return window_size + block_size - 1, block_size - 1   # NON-causal: win_right = block-1
   return window_size - 1, 0                                     # else (main model) -> causal
   ```
   So `dflash_proposer.py`'s `causal=False` flows into `get_draft_swa_window`, which returns
   `win_right = block-1` (=4 at `num_spec=5`) — **identical to impl A's `_dspark_sas_window`**, feeding
   the SAME `npu_sparse_attn_sharedkv` op.
4. **The `ori_win_right=0` sites the handoff cited are the MAIN MODEL's DSA**, not the draft:
   `dsa_v1.py:769-830` are the `prefill_ratio_to_sas_metadata` / `compressor_ratio ∈ {1,4,128}` /
   `sliding_window-1` paths — the DeepSeek-V4 *main model*, which is *supposed* to be causal. Reading
   "win_right=0 everywhere" as "the draft is causal" conflates main-model vs draft attention.

**Therefore:** #11196 is exactly what makes the `dsa_v1.py` draft path non-causal (via `dsa_window.py`).
The box (post-#11196, 2026-07-03) has it; the external tree (pre-#11196, 2026-06-30) does not — so the
handoff described a *pre-fix* snapshot as if it were impl B's current state.

## One inference is also too strong

> "released draft gives good AL ⇒ the isolated PROD≠REF is a harness/build artifact, not a serve bug"

Too strong. The harness is validated (`OURS vs REF = 1.9e-5` fp16, two independent implementations
agree). And **both** box draft paths call the op with `win_right = block-1` — exactly what the parity
tests — so the compiled op's **systematic ~2e-2 vs its own reference** (uniform per block-position,
precision-independent; see `STATUS_FOR_REVIEW_2026-07-14.md` §3–§8 + the sink probe in
`diag_sas_window.py`) sits in the serve's real op invocation. Good AL would mean it is **tolerated**,
not that it is unreal or a harness artifact. "Serve is usable" and "op is bit-correct vs its reference"
are different questions; the handoff conflates them.

## Net / what to actually do

- **Drop the "serve might be causal" concern for the box.** Both draft paths (`dspark_attention.py` and
  `dsa_v1.py`) are non-causal with `win_right = block-1`. The non-causal parity is on-target for both.
- **Real open questions** (reframed): (a) which of the *two non-causal* draft paths the serve dispatches
  at runtime (config-dependent: `deepseek_v4_dspark.py` vs the `dflash_proposer`/`dsa_v1.py` builder) —
  it's non-causal either way; (b) the separate, confirmed compiled-op ~2e-2-vs-reference (likely sink),
  which affects whichever path uses `npu_sparse_attn_sharedkv`.
- **Adopt step 2** (end-to-end `accept_len`) as the pragmatic go/no-go; treat it as *complementary* to,
  not a replacement for, the op-vs-reference correctness question.

## Suggested TL;DR replacement for `HANDOFF_FROM_TRAINER_2026-07-16.md`

> - **There are multiple DSpark attention paths in vllm-ascend; they express the draft window
>   differently, but on OUR box both are NON-causal.** The box fork (`va-src @ 6036507`, 2026-07-03)
>   carries BOTH: impl A (`ops/dspark_attention.py`, wired into `models/deepseek_v4_dspark.py`,
>   `win_right = block-1`) and impl B (`attention/dsa_v1.py` + `spec_decode/dflash_proposer.py`
>   `causal=False`). Impl B's draft window is set by PR #11196's `attention/dsa_window.py::get_draft_swa_window`,
>   which returns `win_right = block-1` for a dspark draft — **the same non-causal geometry as impl A**.
> - **The older external tree I first examined (`/workspace/vllm-ascend @ ee549165`, 2026-06-30) is
>   PRE-#11196**: it lacks `dsa_window.py`, so its `dsa_v1.py` passes `win_right=0`. That snapshot's
>   draft looks causal, but it predates the box's fork. Do NOT read it as the serve's current state.
> - ⇒ The non-causal op-parity is **on-target for both of the box's draft paths**. The open question is
>   *which* non-causal path the serve dispatches at runtime (config), not non-causal-vs-causal.
> - **Ground truth regardless:** run the released draft on the serve → gsm8k `accept_len` at `num_spec=5`.
