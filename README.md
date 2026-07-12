# SWA Non-Causal Sink attention — Triton fused-kernel spec

Handoff package for implementing a **Triton fused operator** for the DSpark draft's
block attention. This folder contains **the requirements + a dependency-free eager
reference** only. It contains **no kernel code** — writing the Triton kernel is your job.

- `eager_reference.py` — the math contract, pure torch, CPU/fp64 runnable, with an
  independent oracle + `torch.autograd.gradcheck`. **This is your diff target.**
- `reference_from_repo/` — verbatim ground-truth from the DSpark branch
  (`feat/dspark-confidence-head`, `examples/ascend_npu_dflash/`). These are the files
  the real model actually uses; the eager reference is distilled from them.

### Upstream / provenance

This repo is a **kernel-development spin-off**. The source of truth lives upstream in the
DSpark draft work — pull updates and check parity against it, not against this snapshot:

- **Upstream repo:** `Sawyer117/speculators` (a fork of `vllm-project/speculators`)
- **Reference files** (`reference_from_repo/`): branch `feat/dspark-confidence-head`,
  `examples/ascend_npu_dflash/` — verbatim copies; `sink_attention` (`dsv4_mla_ref.py`) is
  the canonical training math.
- **Real config / shapes** (§3): branch `feat/dsv4-dspark`,
  `src/speculators/models/dsv4_dspark/config.py` (`DSV4DSparkConfig`).
- **The ultimate GOLD** the fused op is validated against on-box:
  `vllm_ascend/ops/dspark_attention.py::_dspark_attention_reference` (and `_dspark_sas_window`
  for the window), a.k.a. the **SAS** op (Sliding-window Attention with Sink). The
  `reference_from_repo/dspark_attn_ref_bench.py` harness benches against it.

If the eager reference and the upstream / vllm_ascend gold ever disagree, **the gold wins**.

---

## 1. What the operator is

A single attention op that combines three properties, none of which any one Ascend-NPU
primitive gives us today:

| property     | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| **SWA**      | each query attends only to keys within a window                         |
| **NON-CAUSAL** | the window is **bidirectional and ASYMMETRIC** — `j ∈ [i-win_left, i+win_right]`, where `win_left = window+block_size-1`, `win_right = block_size-1` (**not** the naive symmetric `window-1`; see §4) |
| **SINK**     | a per-head learnable "attention sink" logit that sits in the softmax **denominator** (a StreamingLLM off-ramp); it normalises but is never a value in the weighted sum |

It runs in the **training** path of the DSpark draft, so it must be **forward + backward**
(autograd-clean, grads flowing to `q, k, v` **and** the sink parameter).

---

## 2. Exact numerics contract  (match this or parity fails)

For query `i`, head `h`:

```
scores[j] = (q_i · k_j) * scale                                # fp32, even if q/k bf16
scores[j] = -inf  if not (i-win_left <= j <= i+win_right)      # asymmetric SWA (§4)
logits    = concat( scores[0..Lk-1] , sink[h] )                # sink is a RAW fp32 logit:
                                                               #   NOT * scale, NOT masked
logits   -= max(logits)                                        # sink participates in the max/norm
p         = softmax(logits)[0..Lk-1]                            # DROP the sink column
o_i       = sum_j  p[j] * v_j                                   # P cast back to v.dtype for P@V

# equivalently (the gold form, vllm_ascend _dspark_attention_reference):
#   smax = max( scores.max(), sink[h] );  e = exp(scores - smax)
#   p    = e / ( e.sum() + exp(sink[h] - smax) );  o_i = p @ V
```

Gotchas that are easy to get wrong (all verified against the model in `reference_from_repo/`):

1. **fp32 softmax.** `scores` and the softmax are computed in fp32 even for bf16 inputs.
   A bf16-vs-fp32-ref diff of ~1e-2 is *expected* rounding, not a bug (run the bench in
   `DTYPE=float32` and diffs collapse to ~1e-6 — that's the correctness signal).
2. **Sink is unscaled and unmasked.** Do not multiply it by `scale`; do not apply the
   window mask to it. It is a single per-head fp32 scalar broadcast to every query.
3. **Sink is in the normaliser, then dropped.** It enters the max-subtract and the
   softmax denominator, but is **not** in the `P@V` sum. As `sink → -inf` the op reduces
   to plain windowed softmax attention — a good sanity check.
4. **P@V dtype.** The probabilities are cast back to `v.dtype` before `P@V` (the model
   accumulates PV in v's dtype). See `sink_attention` in `reference_from_repo/dsv4_mla_ref.py`.

`scale = head_dim ** -0.5` unless the integrator overrides it.

---

## 3. Shapes & layouts — REAL DeepSeek-V4-Flash-DSpark values

These are **fixed model constants** from the HF `config.json` (source: `DSV4DSparkConfig`,
branch `feat/dsv4-dspark`). Put these in the tests — don't use toy sizes as the "real" case.

| symbol        | value    | meaning                                                      |
|---------------|----------|--------------------------------------------------------------|
| `H` heads     | **64**   | query heads (`num_attention_heads`)                          |
| `num_kv_heads`| **1**    | **MLA** — one latent K/V **shared across all 64 query heads** |
| `D` head_dim  | **512**  | per-head q/k/v width (`nope | rope`)                          |
| `rope_head_dim` | **64** | trailing slice of `D` that is rotated (**partial** RoPE)     |
| `window`      | **128**  | sliding-window context tokens                                |
| `block_size` (γ) | **5** (config) / **7** (block7 ckpt) | draft tokens per forward |
| `hidden_size` | **4096** | model dim (for the projections, not the attention core)      |
| `vocab_size`  | **129280** | (`noise_token_id=128799`)                                  |
| `scale`       | **D**⁻⁰·⁵ = 512⁻⁰·⁵ | softmax scale                                    |
| n_draft_layers | **3**   | `target_layer_ids = (40, 41, 42)` — 3 attention layers/draft |

Per-invocation tensor shapes (the **block view**, = the gold parity target):

```
q    : [N, block_size, H, D]          # N = number of draft blocks (~num_anchors)
k,v  : [N, KV,         H, D]           # KV = window + block_size  (real K/V is 1 MLA head, broadcast)
sink : [H]                             # per-head, fp32
o    : [N, block_size, H, D]
```
Real magnitudes: `H=64, D=512, window=128, block_size∈{5,7}` → `KV = 133 or 135`.
`N` scales with sequence/anchors (the bench scans `NBLK=64 … 512`).

**Layouts** (same math): **MLA-shared** (real, `num_kv_heads=1`) vs **MHA** (K/V expanded
per-head — a conservative memory over-estimate the bench uses). Do MHA first if simpler;
MLA-shared just drops the head axis on K/V. **RoPE is applied outside this op** (partial:
only the last 64 of 512 dims, on q & k pre-attention; inverse-rotated on the output).

---

## 4. The mask — two equivalent views (get the window right!)

The DSpark draft attention has **one** real structure, expressible two ways.

**(a) Block view — the gold parity target.** Per draft block, the `block_size` queries attend
**densely** to `[ last window context tokens | the FULL block ]` (**non-causal within the
block** — intra-block causality is the Markov head's job, not the attention's), plus the
per-head sink. Because the K/V is *already* the windowed-context + block, there's no masking
**inside** a block — it's dense over `KV = window + block_size`. This is
`dspark_block_attention_ref` in `eager_reference.py` == the gold `_dspark_attention_reference`
== the bench's `attn_manual`.

**(b) Packed-SWA view — what an efficient kernel implements.** Over one long packed sequence,
this is a sliding-window attention with an **asymmetric** window + sink:

```
keep[i,j] = (j >= i - win_left) & (j <= i + win_right)
win_left  = window + block_size - 1        # e.g. 128 + 7 - 1 = 134
win_right = block_size - 1                 # e.g. 7 - 1 = 6
```

⚠️ **Do not use the naive symmetric `window-1`.** vLLM-Ascend ships a test that rejects it;
the real window is asymmetric. `dspark_sas_window(block_size, window)` (in `eager_reference.py`,
mirroring vllm_ascend `_dspark_sas_window`) returns `(win_left, win_right)`. This is
`swa_sink_attention` in `eager_reference.py`. **A kernel must test the window predicate on the
fly — never materialize a dense `[L,L]` mask** (that's the slow SDPA baseline we're replacing).

For completeness, the DFlash training path builds an even richer doc-aware packed mask
(`src/speculators/models/dflash/attention.py::create_anchor_block_mask_mod`: `same_doc &
before_anchor & in_window` on the context, full non-causal in-block). Keep the kernel general
(window params, optional additive-mask input via `add_mask`) so it can express that too.

> **Status note (symmetric first step → asymmetric gold).** The current `triton_impl/`
> forward kernel + `tests/test_forward_parity.py` target the **symmetric** microbench window
> (`swa_noncausal_sink_attention(window=W)`, `win_left = win_right = W`) — a valid first step.
> To match the **real model** it must move to the **asymmetric** window
> (`swa_sink_attention` + `dspark_sas_window`) at the real shapes (§3), and validate against
> the vllm_ascend gold (`dspark_attn_ref_bench.py`). Both eager entry points live in
> `eager_reference.py`; the symmetric one is a thin wrapper over the asymmetric one.

---

## 5. Why a fused kernel (the NPU gap)

On Ascend NPU (this is the whole motivation — see `reference_from_repo/dspark_npu_op_check.py`):

| path | status on NPU |
|------|---------------|
| `torch.nn.attention.flex_attention` | **does not run** on Ascend (the natural way to express this) |
| `torch_npu.npu_fusion_attention` SWA | **causal-only** (`ori_win_right=0`) — cannot do the `+window` right side, and has no sink |
| dense-mask `F.scaled_dot_product_attention` | works, autograd-clean, but **O(L²) mask + memory** — the slow baseline |
| **eager fp32 softmax** | always works, always has autograd — the guaranteed fallback, slowest |

So the fused Triton kernel's job: **asymmetric (win_left, win_right) windowed attention +
per-head sink, fwd+bwd, without an O(L²) dense mask** — faster than dense SDPA, numerically
matching the gold `_dspark_attention_reference` / the fp32 eager ref.

---

## 6. Acceptance criteria

A kernel is done when, vs `eager_reference.py` (the fp32 reference):

1. **Forward parity.** `allclose(atol=2e-2, rtol=2e-2)` in bf16; diffs collapse to ~1e-6 in
   fp32 (proving it's dtype rounding, not a math bug). Report per-tensor mean-abs & mean-rel.
2. **Backward parity.** Same thresholds on `grad_q, grad_k, grad_v, grad_sink`. In fp64 the
   kernel's bwd must pass the same `gradcheck` the eager ref passes.
3. **Sink behaviour.** `sink → -inf` reproduces plain windowed softmax; finite sink diverts
   mass (non-trivial output delta). The eager self-test asserts both.
4. **Speed.** Faster than dense-mask SDPA on the **real** draft-block shapes (§3), fwd and fwd+bwd.

Two harnesses to reuse (both report `allclose / mean-abs / mean-rel` on output **and** q-grad,
plus fwd and fwd+bwd speedup — add a `bench("triton", attn_triton)` row):
- `reference_from_repo/dspark_attn_ref_bench.py` — **benches against the vllm_ascend GOLD**
  (`_dspark_attention_reference`) at the **real DSV4 shapes** (`H=64, D=512, window=128,
  block∈{5,7}`, `NBLK` scannable). This is the authoritative parity check; run it on the A3.
- `reference_from_repo/dspark_swa_attn_bench.py` — the isolated SWA microbench (eager vs sdpa
  vs npu_fa) for quick iteration. Env knobs on both: `DTYPE=float32`, `ATOL`, `RTOL`, `BS`,
  `WIN`, `NBLK`, `H`, `D`.

**Recorded parity, and an open number (a lead).** Our einsum+sink impl vs the torch reference
`_dspark_attention_reference` is **bit-exact: meanAbs = meanRel = 0.00e+00** (out + grad; A3,
2026-07-07). What was **never recorded** is the per-element error of the actual **compiled SAS
fused kernel** (`npu_sparse_attn_sharedkv`) vs that reference — only a binary "parity pass" +
the end-to-end match (AR 58.79% / AL 3.94 vs GPU ref AL 3.86). `fused_sas_vs_reference_parity.py`
(top level) produces that missing number on the A3, and doubles as the template to validate a
Triton kernel against the same fused op / reference.

---

## 7. Platform notes

- Target env in the repo: `dspark-dsv4-base` on a single Ascend NPU (`torch_npu`, bf16).
- Triton-on-Ascend is a distinct backend from CUDA Triton — confirm the window-predicate,
  the extra sink column, and the fp32 softmax accumulation all lower on your Triton target
  before optimizing. If a construct won't lower, the eager fp32 path is the correctness
  fallback to fall back to (never wrong, just slow).
- Keep the fp32 softmax internal accumulation regardless of input dtype (criterion #1).

---

## 8. File manifest

```
swa_noncausal_sink_kernel/
├── README.md                        # this spec
├── LICENSE                          # MIT
├── eager_reference.py               # THE diff target: gold block form + packed-SWA, oracle + gradcheck + REAL shapes
├── fused_sas_vs_reference_parity.py # LEAD: compiled SAS fused op vs torch reference on A3 (the un-recorded number)
└── reference_from_repo/             # verbatim ground-truth
    ├── dspark_attn_ref_bench.py     # GOLD bench: real DSV4 shapes vs vllm_ascend _dspark_attention_reference + _dspark_sas_window
    ├── dsv4_mla_ref.py              # sink_attention (the exact softmax+sink math), RoPE, RMSNorm, MLAConfig
    ├── dspark_method.py             # dspark_block_mask (single-block window/non-causal mask), heads, loss
    ├── dspark_block_attention.py    # DSparkMTPSelfAttention: how sink_attention+mask compose in the model (MLA-shared)
    ├── dspark_swa_attn_bench.py     # isolated SWA microbench: eager vs sdpa vs npu_fa, parity + speedup
    └── dspark_npu_op_check.py       # proves flex fails / npu_fa is causal-only on Ascend (the "why a kernel")
```

**Start here:** read `dsv4_mla_ref.py::sink_attention` + `dspark_attn_ref_bench.py::attn_manual`
(the exact gold math + real shapes), run `python eager_reference.py` to see the contract pass
(oracle + gradcheck + real-shape smoke), then write the kernel against it and validate with
`dspark_attn_ref_bench.py` on the A3.
