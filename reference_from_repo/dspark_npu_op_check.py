#!/usr/bin/env python3
"""Minimal Ascend-NPU operator check for a DSpark-native draft port into speculators.

DSpark's draft (DeepSpec `deepspec/modeling/dspark/`) = standard transformer draft
layers (which DFlash already runs on NPU) + a **Markov head** + a **Confidence head**
+ **anchor-block sampling**. This script instantiates ONLY the DSpark-SPECIFIC pieces
and forwards them on the NPU, so we know the ops are supported BEFORE investing in a port.

Reading of DeepSpec (markov_head.py, common.py, qwen3/modeling.py): the ONLY non-standard
op is FlexAttention (`torch.nn.attention.flex_attention` + `create_block_mask`), which does
NOT run on Ascend — DFlash already replaces it with SDPA + a dense block mask. Everything
else (markov = embed+linear+sigmoid/tanh, confidence = one linear, sampling = gather/sort/
cumprod/where/scatter) is standard torch. This script proves that on the actual NPU.

Run on any single NPU (in the dspark-dsv4-base env):
    python dspark_npu_op_check.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    print(f"!! torch_npu import failed: {e}")
    raise SystemExit(1)

torch.manual_seed(0)
DT = torch.bfloat16
B, SEQ, NB, BS = 2, 64, 8, 7          # batch, ctx seq len, num anchor blocks, block_size
V, R, D = 4096, 256, 512             # vocab, markov_rank, hidden_size
QLEN = NB * BS
KVLEN = SEQ + NB * BS

results = []


def check(name, fn):
    try:
        out = fn()
        torch.npu.synchronize()
        shp = tuple(out.shape) if hasattr(out, "shape") else "-"
        print(f"  OK   {name:<44} out={shp}")
        results.append((name, True))
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name:<44} {type(e).__name__}: {str(e)[:90]}")
        results.append((name, False))


print(f">>> device={DEV} dtype={DT}  B={B} seq={SEQ} blocks={NB} bs={BS} V={V} R={R} D={D}\n")


# 1. Vanilla Markov head: Embedding + Linear (adds a logit bias from the prev token)
def markov_vanilla():
    w1 = nn.Embedding(V, R).to(DEV, DT)
    w2 = nn.Linear(R, V, bias=False).to(DEV, DT)
    tok = torch.randint(0, V, (B, QLEN), device=DEV)
    logits = torch.randn(B, QLEN, V, device=DEV, dtype=DT)
    return logits + w2(w1(tok))


check("markov vanilla (embed+linear)", markov_vanilla)


# 2. Gated Markov head: cat + Linear + sigmoid + elementwise gate
def markov_gated():
    w1 = nn.Embedding(V, R).to(DEV, DT)
    w2 = nn.Linear(R, V, bias=False).to(DEV, DT)
    gate = nn.Linear(D + R, R).to(DEV, DT)
    tok = torch.randint(0, V, (B, QLEN), device=DEV)
    h = torch.randn(B, QLEN, D, device=DEV, dtype=DT)
    pe = w1(tok)
    g = torch.sigmoid(gate(torch.cat([h, pe], dim=-1)))
    return w2(g * pe)


check("markov gated (cat+sigmoid+mul)", markov_gated)


# 3. RNN Markov head: GRU-like recurrence unrolled over block_size (chunk/sigmoid/tanh)
def markov_rnn():
    w1 = nn.Embedding(V, R).to(DEV, DT)
    w2 = nn.Linear(R, V, bias=False).to(DEV, DT)
    joint = nn.Linear(2 * R + D, 3 * R).to(DEV, DT)
    tok = torch.randint(0, V, (B, NB, BS), device=DEV)
    h = torch.randn(B, NB, BS, D, device=DEV, dtype=DT)
    base = torch.randn(B, NB, BS, V, device=DEV, dtype=DT)
    state = torch.zeros(B, NB, R, device=DEV, dtype=DT)
    outs = []
    for k in range(BS):
        pe = w1(tok[..., k])
        gr, cr, orr = joint(torch.cat([state, pe, h[..., k, :]], dim=-1)).chunk(3, dim=-1)
        gate = torch.sigmoid(gr)
        state = gate * state + (1.0 - gate) * torch.tanh(cr)
        outs.append(base[..., k, :] + w2(torch.tanh(orr)))
    return torch.stack(outs, dim=-2)


check("markov rnn (gru loop)", markov_rnn)


# 4. Confidence head: a single Linear -> accept-rate scalar
def confidence():
    proj = nn.Linear(D, 1).to(DEV, DT)
    feat = torch.randn(B, NB, BS, D, device=DEV, dtype=DT)
    return proj(feat).squeeze(-1)


check("confidence head (linear)", confidence)


# 5. Anchor-block sampling: arange/rand/where/sort/gather/cumprod + advanced-index scatter
def anchor_sampling():
    loss_mask = (torch.rand(B, SEQ, device=DEV) > 0.3).float()
    nc = SEQ - 1
    valid = (loss_mask[:, :nc] > 0.5) & (loss_mask[:, 1:nc + 1] > 0.5)
    idx = torch.arange(nc, device=DEV).unsqueeze(0).expand(B, -1)
    masked = torch.where(valid, idx, torch.full_like(idx, SEQ + 1))
    rv = torch.where(valid, torch.rand(B, nc, device=DEV), torch.full((B, nc), 2.0, device=DEV))
    _, si = rv.sort(dim=1)
    gathered = torch.gather(masked, 1, si)
    anchors = gathered[:, :NB].clamp(max=SEQ - 1).sort(dim=1).values
    keep = torch.arange(NB, device=DEV).unsqueeze(0) < valid.sum(1, keepdim=True).clamp(max=NB)
    em = (torch.rand(B, NB, BS, device=DEV) > 0.2).to(torch.int32).cumprod(dim=-1).bool()
    emb = nn.Embedding(V, D).to(DEV, DT)
    noise = torch.full((B, NB * BS), 3, dtype=torch.long, device=DEV)
    bstart = (torch.arange(NB, device=DEV) * BS).unsqueeze(0).expand(B, -1)
    bidx = torch.arange(B, device=DEV).unsqueeze(1).expand(B, NB)
    atok = torch.gather(torch.randint(0, V, (B, SEQ), device=DEV), 1, anchors)
    noise[bidx, bstart] = torch.where(keep, atok, torch.full_like(atok, 3))
    _ = em
    return emb(noise)


check("anchor sampling (gather/sort/cumprod/scatter)", anchor_sampling)


# 6. THE KEY ONE: DSpark anchor-block mask via SDPA (the flex_attention replacement)
def block_sdpa():
    H, hd = 4, D // 4
    q = torch.randn(B, H, QLEN, hd, device=DEV, dtype=DT)
    k = torch.randn(B, H, KVLEN, hd, device=DEV, dtype=DT)
    v = torch.randn(B, H, KVLEN, hd, device=DEV, dtype=DT)
    anchor_pos = torch.randint(1, SEQ, (B, NB), device=DEV)
    keep = torch.ones(B, NB, dtype=torch.bool, device=DEV)
    qb = torch.arange(QLEN, device=DEV) // BS                       # [QLEN] q block id
    kv = torch.arange(KVLEN, device=DEV)                            # [KVLEN]
    kvb = (kv - SEQ) // BS                                          # draft kv block id
    ap = anchor_pos[:, qb]                                          # [B, QLEN]
    mctx = (kv < SEQ).view(1, 1, KVLEN) & (kv.view(1, 1, KVLEN) < ap.unsqueeze(-1))
    mdrf = (kv >= SEQ).view(1, 1, KVLEN) & (qb.view(1, QLEN, 1) == kvb.view(1, 1, KVLEN))
    mask = ((mctx | mdrf) & keep[:, qb].unsqueeze(-1)).unsqueeze(1)  # [B,1,QLEN,KVLEN]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


check("BLOCK ATTENTION via SDPA (flex replacement)", block_sdpa)


# 6b. NON-CAUSAL SLIDING-WINDOW via SDPA — the colleague's "bidirectional SWA" concern.
# DSV4-Flash is a sliding-window arch and the DSpark draft is non-causal (is_causal=False),
# so the draft attends within a window BOTH ways (win_left>0 AND win_right>0). The fused
# NPU SWA kernel is causal-only (ori_win_right=0) — but a DENSE-mask SDPA can express the
# bidirectional window. This checks whether that dense non-causal-SWA path runs on NPU
# (the training path). If this passes, training isn't blocked; the fused-kernel bidirectional
# support (vllm-ascend #11125) is a separate INFERENCE-side question.
def swa_noncausal_sdpa():
    H, hd = 4, D // 4
    L = KVLEN
    win = 16
    q = torch.randn(B, H, L, hd, device=DEV, dtype=DT)
    k = torch.randn(B, H, L, hd, device=DEV, dtype=DT)
    v = torch.randn(B, H, L, hd, device=DEV, dtype=DT)
    qi = torch.arange(L, device=DEV).view(L, 1)
    ki = torch.arange(L, device=DEV).view(1, L)
    mask = ((ki >= qi - win) & (ki <= qi + win)).view(1, 1, L, L)  # BIDIRECTIONAL window (no causal cut)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


check("NON-CAUSAL sliding-window via SDPA (bidir SWA)", swa_noncausal_sdpa)


# 7. FlexAttention — EXPECTED to fail on NPU (documents why DFlash/DSpark use SDPA)
def flex():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    H, hd = 4, D // 4
    q = torch.randn(B, H, QLEN, hd, device=DEV, dtype=DT)
    k = torch.randn(B, H, KVLEN, hd, device=DEV, dtype=DT)
    v = torch.randn(B, H, KVLEN, hd, device=DEV, dtype=DT)

    def mm(b, h, qi, ki):
        return ki < SEQ

    bm = create_block_mask(mm, B=B, H=None, Q_LEN=QLEN, KV_LEN=KVLEN, device=DEV)
    return flex_attention(q, k, v, block_mask=bm)


check("flex_attention (EXPECTED to fail on NPU)", flex)


# ---- summary ----
ok = sum(1 for _, r in results if r)
print(f"\n>>> {ok}/{len(results)} op groups ran on NPU.")
critical = [n for n, r in results if not r and "flex" not in n.lower()]
if not critical:
    print(">>> RESULT: all DSpark-specific ops (markov / confidence / sampling / SDPA-block) run on NPU.")
    print(">>>         flex_attention failing is EXPECTED — use the SDPA block mask (as DFlash does).")
    print(">>>         => no basic-operator blocker for a DSpark-native port.")
else:
    print(f">>> RESULT: NON-flex failures that would block a port: {critical}")
