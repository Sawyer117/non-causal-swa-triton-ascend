# Problem record — stale AscendC SAS op after `install_npu_env_dspark.sh` (2026-07-14)

**For:** whoever owns `speculators/examples/ascend_npu_dflash/install_npu_env_dspark.sh`. **Question
to answer:** does the install script's clean step force the **AscendC operator kernels** to recompile,
or can it leave a **stale compiled op** while the C++/pybind layer and the source both look up to date?

Everything below is source/number-backed so you can re-check, not trust. Node:
`n84449292@…` / env `dspark-dsv4-austin`, install root
`/home/a00652497/dspark_austin/installation/vllm-ascend-v4`, CANN `9.0.0.0430`.

---

## TL;DR

The compiled production SAS op (`npu_sparse_attn_sharedkv`) on this node computes **wrong** results
(differs from its own torch reference by `maxAbs 1.05`, `meanAbs 1.6e-2`), even though:
- the **source has the correct fix** (PR #11196's `win_right` clamp) at both the build-time commit and
  the current checkout, and
- **CANN is the required version** (`==9.0.0`), and
- our independent Triton kernel matches the same reference to `1.5e-4`.

The evidence points to a **stale compiled AscendC operator package**: the op behaves exactly as if
PR #11196's kernel clamp is missing, and the install's version string shows the binary was built from
a **different, dirty** commit than the current tree. The install script (`:85`) only does
`rm -rf csrc/build` — which may **not** clear the AscendC op output, and `setup.py:297` states AscendC
kernels get **no ccache/ninja** incremental tracking. So a `pip install -e` "rebuild" can silently
keep a stale kernel. **Please review whether the clean step is sufficient.**

## The bug the stale kernel reproduces (PR #11196)

`win_right` (non-causal, future-block-token) support is added by vllm-ascend PR #11196 across THREE
separately-compiled units (diff cached at `/workspace/dspark_extract/pr11196.diff`):
1. `csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp:1365,1368` — relax the
   gate `oriWinLeft_!=127 / oriWinRight_!=0` → `< 0`.
2. `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h:692` — **the
   critical clamp** `oriMaskRight = Min(oriMaskRight, actOriS2Size − 1)`. The line
   `oriMaskRight = … + oriWinRight` already existed; without the clamp, `oriMaskRight` overshoots the
   KV length (win_right=4, KV=133 → right edge ≈136 > 132) and the kernel reads keys past the buffer
   → garbage.
3. `csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/…_aicpu.cpp` — `nextToken_ = winRight_`.

The node's op has (1) — it accepts `win_right=4` without erroring/falling back — and behaves as if it
is **missing (2)**: non-causal-ish (so it sees future tokens, (3) present) but numerically wrong in a
precision-independent way, worse on later query rows. That is the missing-clamp signature.

## Evidence it is a STALE BUILD, not source / CANN / our kernel

```
# source on the node HAS the clamp:
$ grep -n 'Min(tempLoopInfo.oriMaskRight' \
    .../vllm-ascend-v4/csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h
692:  tempLoopInfo.oriMaskRight = Min(tempLoopInfo.oriMaskRight, tempLoopInfo.actOriS2Size - 1);
# build-time commit 381bd0e ALSO has it (checked a second checkout). So source was never the problem.

# but the installed binary was built from a DIFFERENT, DIRTY commit than the current checkout:
$ pip show vllm_ascend  ->  0.1.dev3893+g381bd0e35.d20260714   # built @ 381bd0e35, tree dirty (.d)
$ git -C .../vllm-ascend-v4 branch  ->  * (HEAD detached from 60365071)   # current HEAD != 381bd0e35

# runtime behavior (BS=5, bf16, shared-KV/MLA scenario):
[parity] OURS vs REF   allclose=True   meanAbs=1.53e-4   # our Triton kernel == reference
[parity] PROD vs REF   allclose=False  meanAbs=1.60e-2  maxAbs=1.05   # the compiled op is wrong
# fp16 gives the SAME error as bf16 -> precision-independent -> not rounding.
```

- **CANN is not the cause**: branch requires `CANN==9.0.0` (`vllm-ascend/README.md:65`), node has
  `9.0.0.0430`. Too-low CANN fails to build or fails at dispatch (cf. the op's fp32
  `AclNN_Parameter_Error`), it does not silently miscompute.
- **Source is not the cause**: the clamp is present in the checked-out source and at the build-time
  commit.
- **Our kernel is not involved**: `OURS vs REF = 1.5e-4`.

## Why this points at the install script

`install_npu_env_dspark.sh:85`:
```bash
( cd "$VA_DIR" && rm -rf csrc/build && pip install -e . --no-deps --no-build-isolation -v )
```
- It removes only `csrc/build`. The **installed AscendC op package** for an editable build lives under
  `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/` (rpath at `CMakeLists.txt:198`;
  install destinations `packages/vendors/${VENDOR_NAME}_transformer/…` in `csrc/CMakeLists.txt`).
  `rm -rf csrc/build` does **not** delete that package, nor any output/cache used by the AscendC op
  build (`build_aclnn.sh`, referenced in the script's own comments at `:65`), nor a CANN opp custom
  vendor cache if one is used.
- `setup.py:297`: *"ccache and ninja can not be applied at ascendc kernels now"* — the AscendC kernels
  have **no incremental change-tracking**, so if their prior output is not deleted, a rebuild can
  reuse a stale kernel even though the `.cpp/.h` changed (or the tree moved commits).
- Net: an editable re-install after a `git checkout`/`git pull` recompiles the pybind `vllm_ascend_C.so`
  and the op_host C++, but can leave the **AscendC op_kernel package stale** — matching exactly what
  we see (op_host gate fresh → accepts win_right; op_kernel stale → missing clamp → wrong).

## What to check / candidate fix (reviewer's call)

1. Is `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/` present and **older** than the source
   (`ls -la` vs the `swa_kernel.h` mtime / the build-time date)? If older → stale, confirmed.
2. Where does the AscendC op build (`build_aclnn.sh` / the CMake AscendC targets) write its
   intermediate `.o`/operator package, and does anything survive `rm -rf csrc/build`? Is there a CANN
   opp custom-vendor cache (e.g. under `$ASCEND_HOME/opp/vendors/` or a per-user opp dir) that the
   build reuses?
3. Does the editable install skip op recompile when the package already exists?

Candidate fix for the script's clean step (verify before adopting):
```bash
( cd "$VA_DIR" \
  && rm -rf build csrc/build vllm_ascend/_cann_ops_custom \
  && pip install -e . --no-deps --no-build-isolation --no-cache-dir -v )
```
plus, if a CANN opp custom vendor cache is used, clear the `*_transformer` vendor there too. Then
**verify** the rebuilt op reflects the source:
```bash
BS=5 python ours_vs_production.py     # expect PROD vs REF to collapse from 1.05 -> ~1e-4..1e-2
```

## Note

This also reconciles "real dspark inference gets good acceptance length": that was measured against a
correctly/fully built op (elsewhere, or before the tree moved); THIS install's kernel package is
stale. Full technical context: `STATUS_FOR_REVIEW_2026-07-14.md` §3–§8.1 in this repo.
