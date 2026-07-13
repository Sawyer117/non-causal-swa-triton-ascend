# Lead: MindSpeed's Ascend Triton sink flash-attention (a working fork base)

`g2_attention_kernel.py` is copied verbatim from **MindSpeed-LLM** (Huawei Ascend), file
`mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention_kernel.py`
(source: https://github.com/Ascend/MindSpeed-LLM , Apache-2.0). It is DeepSeek-V4's
attention kernel and it already contains **exactly the thing this repo is building** — an
Ascend **Triton** flash-attention with a **per-head learnable sink**, working on NPU.

## Why this is the strongest lead we have
- **`SparseFlashAttentionTriton(torch.autograd.Function)`** (~line 161): a full **forward +
  backward** Triton flash-attention on Ascend, with the sink term in the softmax denominator.
  It's sparse/windowed via `topk_idxs` (causal); the DSpark draft needs **block-non-causal**.
- It already solves the **Ascend-Triton tiling** problems you're fighting (UB/cbuf overflow,
  head-dim D-tiling): see the `@triton.jit` kernel — `BLOCK_H`/`HALF_H`, `HALF_N` sub-tiling,
  online-softmax `m_i`/`acc`, and crucially `import triton.language.extra.cann.extension as al`
  (the Ascend Triton extension). Study how it splits tiles to fit Ascend's memory hierarchy.
- **`G2CoreAttention.sparse_flash_attn`** (~line 39) is the pure-torch eager reference for the
  SAME math (MLA shared-latent kv, concat-sink → softmax → drop-sink → PV) — a second parity
  oracle alongside our `eager_reference.py`.

## What to change to get our kernel (block-non-causal + sink)
Fork `SparseFlashAttentionTriton`, keep the sink + the Ascend tiling scaffolding, and replace
the **causal `topk_idxs` masking** with the **asymmetric block-non-causal window**
(`win_left = window + block_size - 1`, `win_right = block_size - 1`; see the top-level README).
i.e. the attention math + Ascend tiling are already done here; only the mask predicate changes.

## Attribution / license
Copyright Huawei / Ascend, **Apache License 2.0**. Redistributed here unmodified for reference
under that license. Not our code; do not relicense.
