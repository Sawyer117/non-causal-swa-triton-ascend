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
- **Branch:** `feat/dspark-confidence-head`
- **Path:** `examples/ascend_npu_dflash/`  — the files under `reference_from_repo/` here
  are verbatim copies from there; `sink_attention` (`dsv4_mla_ref.py`) is the canonical math.

If the eager reference and the upstream files ever disagree, **upstream wins**.

---

## 1. What the operator is

A single attention op that combines three properties, none of which any one Ascend-NPU
primitive gives us today:

| property     | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| **SWA**      | each query attends only to keys within a window                         |
| **NON-CAUSAL** | the window is **bidirectional** — `j ∈ [i-window, i+window]`, not `j ≤ i` |
| **SINK**     | a per-head learnable "attention sink" logit joins the softmax as one extra column, then is dropped from the value sum (StreamingLLM-style off-ramp) |

It runs in the **training** path of the DSpark draft, so it must be **forward + backward**
(autograd-clean, grads flowing to `q, k, v` **and** the sink parameter).

---

## 2. Exact numerics contract  (match this or parity fails)

For query `i`, head `h`:

```
scores[j] = (q_i · k_j) * scale                         # fp32, even if q/k bf16
scores[j] = -inf   if  not (i-window <= j <= i+window)  # bidirectional SWA
logits    = concat( scores[0..Lk-1] , sink[h] )         # sink is a RAW fp32 logit:
                                                        #   NOT * scale, NOT masked
logits   -= max(logits)                                 # sink participates in the max/norm
p         = softmax(logits)[0..Lk-1]                     # DROP the sink column
o_i       = sum_j  p[j] * v_j                            # P cast back to v.dtype for P@V
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

## 3. Shapes & layouts

Same math, two KV layouts — target whichever the integrator wires up:

- **MHA (primary target):** `q,k,v = [B, H, L, D]`, per-head keys/values.
  This is the isolated SWA microbench (`reference_from_repo/dspark_swa_attn_bench.py`).
- **MLA-shared (DSV4 model):** `q = [B, H, L, D]`, `k,v = [B, Lk, D]` — **one latent KV
  shared across all heads**. This is what `DSparkMTPSelfAttention` actually uses
  (`reference_from_repo/dspark_block_attention.py` → `sink_attention`). If you only do
  one, do MHA first; MLA-shared just drops the `H` axis on K/V.

`sink = [H]` fp32. Output `o = [B, H, L, D]`.
Representative draft-block sizes from the bench: `B=2, H=8, L=512, D=64, window=±16`.

---

## 4. The mask: isolated vs. real

There are **two** mask structures in play. Know both.

**(a) Isolated bidirectional SWA** — what "SWA non-causal" names, and the tractable first
kernel. `keep[i,j] = (j >= i-window) & (j <= i+window)`. Square attention, per-head KV.
This is `attn_eager` in `dspark_swa_attn_bench.py` and `bidirectional_window_mask` in
`eager_reference.py`. **A kernel must test this predicate on the fly — never materialize a
dense `[L,L]` mask** (that defeats the purpose; dense-mask SDPA is the slow baseline we're
replacing).

**(b) Real packed anchor-block mask** — the full training op the kernel ultimately plugs
into (`src/speculators/models/dflash/attention.py::create_anchor_block_mask_mod`, and its
single-block reduction `dspark_block_mask` in `reference_from_repo/dspark_method.py`).
KV = `[ packed base sequence | synthetic anchor blocks ]`; each query block `j` (one anchor)
may attend to:
   - base tokens in the **same document**, **before** its anchor, **within `window`** of it
     (`same_doc & before_anchor & in_window`) — this is the causal SWA part on the context,
   - **all** tokens of its **own** block (non-causal in-block; intra-block causality is the
     Markov head's job, not the attention's).
Plus the per-head sink. This is doc-aware and anchor-indexed — richer than (a). Build the
kernel general enough (window params + block layout, or an optional additive-mask input)
that it can express (b), even if you validate on (a) first. `eager_reference.py` accepts an
optional `add_mask` for exactly this.

---

## 5. Why a fused kernel (the NPU gap)

On Ascend NPU (this is the whole motivation — see `reference_from_repo/dspark_npu_op_check.py`):

| path | status on NPU |
|------|---------------|
| `torch.nn.attention.flex_attention` | **does not run** on Ascend (the natural way to express this) |
| `torch_npu.npu_fusion_attention` SWA | **causal-only** (`ori_win_right=0`) — cannot do the `+window` right side, and has no sink |
| dense-mask `F.scaled_dot_product_attention` | works, autograd-clean, but **O(L²) mask + memory** — the slow baseline |
| **eager fp32 softmax** | always works, always has autograd — the guaranteed fallback, slowest |

So the fused Triton kernel's job: **bidirectional windowed attention + per-head sink, fwd+bwd,
without an O(L²) dense mask**, faster than dense SDPA, numerically matching the fp32 eager ref.

---

## 6. Acceptance criteria

A kernel is done when, vs `eager_reference.py` (the fp32 reference):

1. **Forward parity.** `allclose(atol=2e-2, rtol=2e-2)` in bf16; diffs collapse to ~1e-6 in
   fp32 (proving it's dtype rounding, not a math bug). Report per-tensor mean-abs & mean-rel.
2. **Backward parity.** Same thresholds on `grad_q, grad_k, grad_v, grad_sink`. In fp64 the
   kernel's bwd must pass the same `gradcheck` the eager ref passes.
3. **Sink behaviour.** `sink → -inf` reproduces plain windowed softmax; finite sink diverts
   mass (non-trivial output delta). The eager self-test asserts both.
4. **Speed.** Faster than dense-mask SDPA on the draft-block shapes, fwd and fwd+bwd.

Reuse the harness in `reference_from_repo/dspark_swa_attn_bench.py`: it already reports
`allclose / mean-abs / mean-rel` for output **and** q-grad, plus fwd and fwd+bwd speedup vs
the eager reference. Add your kernel as another `bench("triton", attn_triton)` row and add a
sink column to its reference. Env knobs: `DTYPE=float32`, `ATOL`, `RTOL`.

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
├── eager_reference.py               # THE diff target: pure-torch fwd+bwd, oracle + gradcheck
└── reference_from_repo/             # verbatim ground-truth (feat/dspark-confidence-head)
    ├── dsv4_mla_ref.py              # sink_attention (the exact softmax+sink math), RoPE, RMSNorm, MLAConfig
    ├── dspark_method.py             # dspark_block_mask (single-block window/non-causal mask), heads, loss
    ├── dspark_block_attention.py    # DSparkMTPSelfAttention: how sink_attention+mask compose in the model (MLA-shared)
    ├── dspark_swa_attn_bench.py     # NPU bench harness: eager vs sdpa vs npu_fa, parity + speedup (reuse this)
    └── dspark_npu_op_check.py       # proves flex fails / npu_fa is causal-only on Ascend (the "why a kernel")
```

**Start here:** read `dsv4_mla_ref.py::sink_attention` (the exact math) and
`dspark_swa_attn_bench.py::attn_eager` (the window structure), then run
`python eager_reference.py` to see the contract pass, then write the kernel against it.
