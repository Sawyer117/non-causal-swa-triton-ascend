# Running the forward parity test on a GPU

The kernel is developed on a CPU-only box (no GPU/NPU there), so the Triton run happens
on your GPU machine. This is the full loop: **clone/pull → install → run → paste output back**.

## 0. Prereqs

- An NVIDIA GPU with a working CUDA driver (`nvidia-smi` prints your card).
- Python 3.9+.

## 1. Get the code

First time on this machine — clone:
```bash
git clone https://github.com/Sawyer117/non-causal-swa-triton-ascend.git
cd non-causal-swa-triton-ascend
```

Already cloned — just update:
```bash
cd non-causal-swa-triton-ascend
git pull                       # branch: main
```

Files that matter:
- `triton_impl/swa_sink_fwd.py`   — the forward Triton kernel
- `tests/test_forward_parity.py`  — this parity test
- `eager_reference.py`            — the fp32 diff target (already in the repo)

## 2. Install dependencies

You need **CUDA PyTorch + Triton**. Triton ships as a dependency of the Linux CUDA
PyTorch wheel, so if you already run torch on this GPU, `import triton` almost certainly
works already — check first:

```bash
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('triton', triton.__version__)"
```

- If that prints your torch/triton versions and `cuda True` → skip to step 3.
- If torch is missing or CPU-only, install a CUDA build matching your driver, e.g. CUDA 12.1:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
  (pick the cuXXX that matches your system; `nvidia-smi` shows the max CUDA version.)
- If `import triton` fails but torch is fine:
  ```bash
  pip install triton
  ```
- Optional (silences a harmless "Failed to initialize NumPy" warning): `pip install numpy`.

> Tip if `pip` dies with `No space left on device`: your `/tmp` may be a small tmpfs —
> run pip with `TMPDIR=/path/on/a/big/disk pip install ...`.

## 3. Run

```bash
python tests/test_forward_parity.py                 # forward: runs BOTH fp32 and bf16
python tests/test_backward_parity.py                # backward: grads for q,k,v,sink vs the gold
```

`test_backward_parity.py` exercises the autograd op (`swa_sink_attn` / `dense_sink_attn` —
fast Triton forward + **fused Triton backward** dq/dk/dv + torch dsink) and checks
grad_{q,k,v,sink} against the eager/gold reference, same two bars (fp32 ~1e-6, bf16 ~1e-2),
plus a small fp32 gradcheck. (The backward algorithm is CPU-validated vs autograd to ~1e-15.)

Optional variants:
```bash
DTYPE=float32 python tests/test_forward_parity.py    # fp32 only (the correctness gate)
ATOL=1e-6 RTOL=1e-6 python tests/test_forward_parity.py   # tighten tolerances yourself
SEED=1 python tests/test_forward_parity.py           # different random inputs
NO_REAL=1 python tests/test_forward_parity.py        # skip the heavy H=64,D=512 cases
```

Cases it runs:
- Windowed self-attention vs the eager `swa_sink_attention` (fp32):
  - `[sym ]` symmetric microbench window (first-step form)
  - `[asym]` the **real asymmetric** window `dspark_sas_window(block=7,window=128)=(L134,R6)`
  - `[asym-mla]` same, but **MLA-shared** K/V (`num_kv_heads=1` — the real model layout)
  - `[asym-b5]` `block_size=5` → window `(L132,R4)`
  - `[real]` real DSV4 shapes **H=64 D=512** (skip with `NO_REAL=1`)
- Dense = **gold block-form parity** vs `dspark_block_attention_ref` (the vllm_ascend gold):
  - `[gold]` block shapes `q[N,BS,H,D] × kv[N,KV,H,D]`, `KV=window+BS`, toy H/D
  - `[gold-mla]` same, MLA-shared
  - `[gold-real]` block shapes at real DSV4 **H=64 D=512** (skip with `NO_REAL=1`)
- `sink behaviour`: `sink->-inf` == plain softmax; a finite sink diverts mass

The `[real]`/`[gold-real]` cases use small tiles (BLOCK=8–16) for shared-memory headroom on
smaller GPUs.
If it raises a shared-memory/compile error (not a math error), rerun with `NO_REAL=1` and
tell me — that's a tile-size tuning issue, separate from correctness.

## 4. What to paste back

Copy the **entire output**. Each case (`[sym]`, `[asym]`, `[real]`, `sink behaviour`) prints
one row per dtype, then a final PASS/FAIL:

```
### [asym] real window, toy H/D   B=2 H=8 L=384 D=64  window=(L134,R6)  tile=(32,32)
  [float32 ] allclose=...  maxAbs=...  meanAbs=...  meanRel=...  (atol=1e-05)  OK/FAIL
  [bfloat16] allclose=...  maxAbs=...  meanAbs=...  meanRel=...  (atol=0.02)   OK/FAIL
### sink behaviour   window=(L134,R6)  dtype=bfloat16
  [sink0] sink->-inf == windowed softmax: allclose=...  maxAbs=...   OK/FAIL
  [sinkE] finite sink diverts mass: mean|o(sink)-o(-inf)|=...   OK/FAIL
...
PASS/FAIL
```

## 5. How to read it (the two bars are intentional)

Both bars compare the kernel to a reference computed **on the same (dtype-rounded) inputs**,
upcast to fp32 — so the error isolates the KERNEL's fidelity, not the fp32→bf16 *input*
rounding (which isn't the kernel's job).

- **float32 is the correctness gate.** The kernel forces `input_precision="ieee"` (true
  fp32, no TF32), so the `float32` row's `maxAbs` should be **~1e-6**. That's the real signal
  the math is right. If fp32 `maxAbs` is ~1e-3 → TF32 leaked in; ~1e-2 → a genuine math bug.
- **bfloat16 is deployment realism.** With identical inputs, the bf16 error reflects the
  kernel's fp32-accumulation + bf16 P@V + **bf16 output rounding** (~4e-3 relative is the
  inherent floor, since the kernel returns bf16). Expect `maxAbs` a few×1e-3, well under the
  2e-2 tolerance. (A bf16-vs-*fp32-original-input* comparison would instead be ~1e-2 — that
  extra gap is the input rounding, not a kernel error.)
- **`(info) vs production-eager(bf16)`** (not gated) — how far the kernel is from the torch
  eager op run at bf16, i.e. the drop-in-replacement delta. It's ~1e-2 **because eager-bf16
  rounds QK to bf16** (`einsum(bf16,bf16)->bf16`) while the kernel keeps fp32 QK accumulation
  — so this gap is mostly eager-bf16's own imprecision; the kernel is ~2× closer to the fp32
  truth. Against a fused fp32-QK production op (e.g. vllm_ascend SAS) the delta would be ~1e-3.
- `[sink0]` confirms `sink -> -inf` collapses to plain windowed softmax; `[sinkE]` confirms
  a finite sink actually changes the output.

If anything FAILs, paste the output — the maxAbs magnitude tells us whether it's TF32,
a math bug, or a Triton-lowering issue, and I'll fix from there.
