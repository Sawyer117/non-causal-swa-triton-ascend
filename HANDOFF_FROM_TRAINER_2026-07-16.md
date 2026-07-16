# Handoff from the DSV4-DSpark trainer session — 2026-07-16

> The main-session user asked me to independently research **"how to correctly RUN the SAS op on
> vllm-ascend"** — detached from this kernel repo, looking at the vllm-ascend serve itself. I found a
> **version divergence** that decides whether your isolated SAS parity is even against the serve's path.
> Calibrated: every claim → `file:line` in a *named tree*. I flag clearly what I did **NOT** resolve, and
> I correct my own first-pass overclaim (see §4).

> **⚠ CORRECTED 2026-07-16 — after `REVIEW_of_handoff_2026-07-16.md` (kernel session). The reviewer was
> right; I independently re-verified and CONCEDE both points below. Read the original TL;DR through this.**
>
> 1. **My central alarm ("the serve may be a *causal* impl B") does NOT hold.** I examined a **stale
>    pre-#11196 tree** (`/workspace/vllm-ascend @ ee549165`, **dated 2026-06-30**; box fork is 2026-07-03).
>    It **lacks `attention/dsa_window.py::get_draft_swa_window`** (verified locally: `grep -c` = 0). That
>    file (added by #11196) is what sets the draft window `win_right = block-1`. On the box, **BOTH draft
>    paths are NON-causal** — impl A (`models/deepseek_v4_dspark.py` → `dspark_attention`) and impl B
>    (`dsa_v1.py` + `spec_decode/dflash_proposer.py causal=False` → `get_draft_swa_window`). **The
>    `ori_win_right=0` sites I cited (`dsa_v1.py:769-830`) are the DeepSeek-V4 MAIN MODEL's DSA (correctly
>    causal), NOT the draft.** I conflated main-model vs draft attention. → The non-causal op-parity is
>    **on-target for both** box draft paths.
> 2. **My "released-draft good AL ⇒ isolated PROD≠REF is a harness/build artifact" was too strong.** The
>    op's systematic ~2e-2 vs its own (validated) reference sits in the serve's REAL invocation (both paths
>    use `win_right = block-1`). Good AL = *tolerated*, not *bit-correct*. "Serve usable" ≠ "op correct vs
>    reference" — keep them separate.
>
> **Converged net (both sessions agree):** the draft is **non-causal** (settled). Open items = (a) *which*
> non-causal path the serve dispatches at runtime (config-dependent; non-causal either way), (b) the
> confirmed compiled-op **~2e-2 vs its reference** (likely sink), in whichever path calls
> `npu_sparse_attn_sharedkv`. **Step 2 (end-to-end released-draft `accept_len`) stands as the go/no-go, but
> is COMPLEMENTARY to — not a replacement for — the op-vs-reference correctness question.**

## TL;DR (original — superseded on the "impl B may be causal" point by the correction above)

- **There are (at least) TWO divergent DSpark attention impls in vllm-ascend; they compute the draft
  block attention DIFFERENTLY.** Your parity harness (+ the 2026-07-14 handoff) targets **impl A**; a
  newer refactor tree is **impl B**. *Which one is our serve* decides whether your PROD≠REF matters.
  - **Impl A (what you test, = #11196 / the 3.94 impl):** `vllm_ascend/ops/dspark_attention.py:140` →
    `npu_sparse_attn_sharedkv`, `_call_dspark_sas_block` passes **`ori_win_right = block-1`** (NON-causal
    block), reference `_dspark_attention_reference`.
  - **Impl B (`/workspace/vllm-ascend @ dspark-npu-fixes`, HEAD `ee549165`, origin vllm-project):**
    refactored to `ops/dsa.py` + `attention/dsa_v1.py` + `device/device_op.py`. **`ops/dspark_attention.py`
    does NOT exist** (`git log --all -- vllm_ascend/ops/dspark_attention.py` → empty; never in this repo's
    history); **`_dspark_attention_reference` absent**; **EVERY SAS-op call is `ori_win_right = 0`
    (CAUSAL SWA, `ori_mask_mode = 4`)** — a grep of the whole tree found **zero** non-zero `win_right`.
- ⇒ **The non-causal SAS parity you're stuck on (win_right=4 vs REF) is impl A's geometry. Impl B never
  invokes `win_right > 0` at all.** So before more isolated op-parity: pin which impl is our serve.

## Sources (file:line — re-check, don't trust)

Impl B (this sandbox's `/workspace/vllm-ascend @ dspark-npu-fixes`):
- No `ops/dspark_attention.py`; `git log --all` for it is empty. `ops/dsa.py` + `attention/dsa_v1.py` present.
- `dsa_v1.py:769-771, 798-799, 822-825, 1030-1085, 1246-1247, 1329-1330` → `ori_mask_mode=4`,
  `ori_win_left = sliding_window - 1`, **`ori_win_right = 0`** (every occurrence).
- SAS op wrapper: `device/device_op.py:514-516` `get_dsa_sparse_attn_op() → torch.ops._C_ascend.npu_sparse_attn_sharedkv`;
  `:518-521` base kwargs `{}` (filled by the caller = `dsa_v1.py`, win_right=0).
- `_dspark_attention_reference` — grep of `vllm_ascend/` = absent.

Impl A (your target; from the 2026-07-14 handoff §Sources 3 + this repo's csrc):
- `vllm_ascend/ops/dspark_attention.py:140` → non-quant `npu_sparse_attn_sharedkv`; `_dspark_sas_window`
  (:32-36) → `ori_win_right = block-1` (non-causal). The `csrc/attention/sparse_attn_sharedkv` op source
  exists in both trees; only the **Python dispatch** differs.

## What I did NOT resolve (be careful — this bounds the claim)

- **WHICH tree is the actual inference serve.** Your STATUS §8.1 shows the box install
  `/home/a00652497/dspark_austin/installation/vllm-ascend-v4/` has `dspark_attention.py` (**impl A**).
  If the serve is still impl A, then **your op-parity is correctly targeting the real serve** — impl B
  is a NEWER refactor tree that is **not (yet) the serve**, i.e. a heads-up about where the code is
  HEADED (the 12003-12006 decomposition), NOT a reframe of your current work. Verify at the box before
  acting.
- Whether impl B's `win_right=0` is a **causal regression** or it achieves block-non-causality via the
  metadata builder (`spec_decode/llm_base_proposer.py` draft metadata) instead of `win_right`. Untraced.

## §4 — my first-pass overclaim, RETRACTED

My initial read to the user was "the other AI is chasing a superseded path (red herring)." That was
**overstated**: I had not yet confirmed the box serve is impl A. The box install (per your §8.1) *is*
impl A, so your parity is most likely on-target and impl B is the *future* tree, not a dead path.
Corrected claim = the calibrated TL;DR above (a version divergence to pin, not a verdict that you're
testing the wrong thing).

## Conclusion + action (ordered)

1. **Confirm the serve's impl** at the box (one command):
   `ls /home/a00652497/dspark_austin/installation/vllm-ascend-v4/vllm_ascend/ops/dspark_attention.py`
   - **present** → impl A → your parity is on-target; the stale-AscendC-op lead (§8.1: force-recompile
     `_cann_ops_custom`) is the thing to close. "Correct way to run the op" = the non-quant
     `npu_sparse_attn_sharedkv` with `win_right = block-1`, from a **fully-recompiled** op package.
   - **absent** + `ops/dsa.py` present → the serve moved to impl B → the non-causal parity is moot;
     re-anchor to `dsa_v1.py` (win_right=0) and trace `llm_base_proposer.py` for how the block
     non-causality is expressed (metadata, not win_right).
2. **Decisive, impl-agnostic ground truth (do this regardless):** run the **RELEASED DSV4 draft** on the
   ACTUAL serve → gsm8k `accept_len` at `num_spec=5`. If ≈ 3.94 (or even just sane), **the serve's op
   path is correct end-to-end** — the isolated PROD≠REF is then a harness/build artifact, not a serve
   bug, and the trainer side proceeds to convert + eval OUR draft. If garbage → a real serve bug, debug
   against the serve's ACTUAL dispatch (impl A `win_right>0` OR impl B `win_right=0`), never a cross-impl
   reference.
3. **Heads-up (trainer→you):** upstream is decomposing DSV4-DSpark into PRs **12003** (attention DSA/SAS)
   / **12004** (draft model + loading hooks) / **12005** (eager decode) / **12006** (ACLGraph) / **11431**
   (refactor). Impl B (`dsa_v1.py`, win_right=0) is that direction. When the serve bumps to it, the draft
   attention geometry must be re-validated — that's when your kernel/parity work re-anchors.

## UPDATE 2026-07-16b — I re-did it on the CORRECT tree; you were right, and here's what I verified

I fetched the box's actual serve branch **`Sawyer117/vllm-ascend @ dspark-dsv4 = 60365071` (2026-07-03)**
into the sandbox (I had been reading `dspark-npu-fixes @ ee549165`, 2026-06-30 — an OLDER branch, my
mistake). On the correct tree I independently confirm your review:

1. **The draft IS non-causal.** `vllm_ascend/attention/dsa_window.py:17` `get_draft_swa_window`:
   ```python
   if not causal and is_dspark_speculative_config(...):
       return window_size + block_size - 1, block_size - 1   # win_right = block-1  (NON-causal)
   ```
   This file is **absent** in the ee549165 tree I first read (that's why I saw only `win_right=0`, which
   is the MAIN MODEL's DSA — exactly your point). Verified: `grep -c get_draft_swa_window` = 0 on the old
   tree, present here. **My "serve may be causal" alarm is dead.** Both your draft geometries are on-target.

2. **The op-with-sink IS in the real serve path.** `models/deepseek_v4_dspark.py:298-309`
   `_run_dspark_attention → dspark_attention(q,k,v, ..., self.attn_sink[:n_local_heads], self.block_size)`.
   So your confirmed compiled-op **~2e-2 vs its own reference (suspected sink)** sits in the serve's real
   `npu_sparse_attn_sharedkv` invocation — it is NOT a harness artifact. **That's the live item for you.**
   I fully concede my earlier "good AL ⇒ artifact" overreach.

3. **Ckpt conversion is now fully mapped** (so the trainer side is unblocked, independent of the op debug):
   `deepseek_v4_dspark.py::_remap_dspark_name` (`.attn.`->`.self_attn.`, `.ffn.`->`.mlp.`,
   `.w{1,2,3}.`->`.{gate,down,up}_proj.`) + the file-header "weights stored under the target ckpt's
   `mtp.*` namespace" pin the target as the RELEASED `mtp.*` layout = the inverse of our
   `weights.py::map_released_key`. Conversion script written + key-map unit-tested:
   `Sawyer117/speculators @ feat/dsv4-dspark-inference : scripts/convert_dspark_to_vllm.py`.

**Net for you:** the non-causal question is closed; the remaining, real one is the **~2e-2 sink deviation
in `npu_sparse_attn_sharedkv`** (your §8.1 stale-AscendC-op lead is still the best hypothesis — force-
recompile `_cann_ops_custom`). **Trainer-side go/no-go stays:** run the RELEASED draft on this serve →
gsm8k `accept_len` (main-session user is doing this in a separate context); if it lands ~3.94 the sink
~2e-2 is *tolerated* and we proceed to convert + eval OUR draft; if it's low, your sink item is what's
biting, and we fix the op before trusting any of our numbers.

## For the trainer side (context, not a request to you)

Our clean-room draft trains with **dense non-causal sink einsum** (`speculators .../backbone/attention.py`
`sink_block_attention`), which is impl A's math (`_dspark_attention_reference`). If the serve is impl A
and the released draft validates end-to-end, converting our `layers.*` ckpt → the `mtp.*` format is
straightforward (invert `speculators .../dsv4_dspark/weights.py::map_released_key` + unstack the
GroupedExperts). We do NOT proceed to that conversion until step 2 (released-draft end-to-end) is green.
