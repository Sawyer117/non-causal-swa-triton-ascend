#!/usr/bin/env python3
"""DSpark method modules (framework-agnostic torch) — the pieces we PORT onto the
MindSpeed DSV4 backbone (track B). Math mirrors the official DSpark
`inference/model.py` (deepseek-ai/DeepSeek-V4-Flash-DSpark) and upstream
speculators #677 loss semantics. All gated behind an `--enable-dspark` flag on-box.

What's here (all standard torch, CPU-testable):
  * DSparkMarkovHead      - low-rank vocab->vocab logit bias (markov_w1/w2)
  * DSparkConfidenceHead  - per-position accept-rate predictor (Linear([h, markov_embed])->1, fp32)
  * build_dspark_block    - block-gamma input ids: [anchor, noise*(gamma-1)]
  * dspark_block_mask     - sliding-window + block-non-causal additive mask (the DSpark window)
  * dspark_compound_loss  - w_k * [ (ce_a*CE + l1_a*L1) + conf_a*BCE(conf, 1-d_TV) ]

MEGATRON WIRING (on-box, track B): swap the plain layers for MindSpeed/megatron
parallel primitives — noted inline as `# MEGATRON:` . The MATH does not change.
The DSV4 backbone (MLA+256-MoE+mHC+sink attention + EP) is REUSED from MindSpeed;
these modules are added to its MTP layer under the dspark flag.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSparkConfig:
    vocab_size: int = 129280
    dim: int = 4096
    block_size: int = 5          # gamma
    markov_rank: int = 256
    noise_token_id: int = 128799
    window_size: int = 128
    target_layer_ids: tuple = (40, 41, 42)   # main_proj reads hidden_states[id+1]
    # loss coefficients (upstream #677 / DeepSpec): dist = 0.1*CE + 0.9*L1 ; + conf BCE
    ce_alpha: float = 0.1
    l1_alpha: float = 0.9
    conf_alpha: float = 1.0
    decay_gamma: float = 5.0     # position decay w_k = exp(-(k-1)/gamma)


class DSparkMarkovHead(nn.Module):
    """Low-rank sequential (Markov) logit bias. Official: markov_w1=Embed(vocab,rank),
    markov_w2=Head(vocab,rank); logits_bias = markov_w2(markov_w1(prev_token))."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)          # MEGATRON: ParallelEmbedding
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)  # MEGATRON: ParallelHead (col-parallel)

    def forward(self, token_ids: torch.Tensor):
        embed = self.markov_w1(token_ids)          # [..., rank]
        logits_bias = self.markov_w2(embed)        # [..., vocab]
        return logits_bias, embed


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance predictor. Official: Linear(dim+rank, 1, fp32) on
    cat([hidden, markov_embed]); returns a logit (squeeze)."""

    def __init__(self, dim: int, markov_rank: int):
        super().__init__()
        self.proj = nn.Linear(dim + markov_rank, 1).float()      # fp32 for stable confidence

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        # cast to the proj's dtype (fp32 in real use; matches module dtype under .double()/.half())
        x = torch.cat([hidden, markov_embed], dim=-1).to(self.proj.weight.dtype)
        return self.proj(x).squeeze(-1)            # [...] logit


def build_dspark_block(anchor_ids: torch.Tensor, block_size: int, noise_token_id: int):
    """[b] anchor token -> [b, block_size] = [anchor, noise, noise, ...]."""
    b = anchor_ids.size(0)
    ids = anchor_ids.new_full((b, block_size), noise_token_id)
    ids[:, 0] = anchor_ids
    return ids


def dspark_block_mask(ctx_len: int, block_size: int, window_size: int, device):
    """Additive mask [block_size, ctx_len + block_size]: each of the `block_size`
    draft queries attends to the last `window_size` context tokens + the FULL block
    (block-non-causal; intra-block causality comes from the Markov head, not here)."""
    kv = ctx_len + block_size
    m = torch.full((block_size, kv), float("-inf"), device=device)
    lo = max(0, ctx_len - window_size)
    m[:, lo:ctx_len] = 0.0          # sliding window over context
    m[:, ctx_len:] = 0.0            # full current block (non-causal)
    return m


def _accept_rate(p_draft: torch.Tensor, p_target: torch.Tensor):
    """Soft acceptance alpha = sum_v min(p, q) = 1 - d_TV. Detached (confidence target)."""
    return torch.minimum(p_draft, p_target).sum(-1)


def dspark_compound_loss(draft_logits, target_logits, target_tokens, conf_logits, cfg: DSparkConfig,
                         valid_mask=None):
    """Position-decayed compound loss over a gamma-block.
      draft_logits/target_logits: [b, gamma, vocab]   target_tokens: [b, gamma]
      conf_logits: [b, gamma]                          valid_mask: [b, gamma] bool
    L = sum_k w_k * [ ce_a*CE + l1_a*L1 + conf_a*BCE(conf, alpha) ] , w_k = exp(-(k-1)/gamma)
    """
    b, g, v = draft_logits.shape
    dev = draft_logits.device
    p = draft_logits.float().softmax(-1)
    q = target_logits.float().softmax(-1).detach()          # target is the teacher (detached)

    ce = F.cross_entropy(draft_logits.reshape(-1, v).float(), target_tokens.reshape(-1),
                         reduction="none").view(b, g)
    l1 = (p - q).abs().sum(-1)                              # full L1 = 2*TV
    alpha = _accept_rate(p, q).detach()                    # [b, g] soft accept rate = 1 - d_TV
    conf_bce = F.binary_cross_entropy_with_logits(conf_logits.float(), alpha, reduction="none")

    per_pos = cfg.ce_alpha * ce + cfg.l1_alpha * l1 + cfg.conf_alpha * conf_bce   # [b, g]

    k = torch.arange(g, device=dev).float()                # 0..gamma-1
    decay = torch.exp(-k / cfg.decay_gamma).view(1, g)     # w_k
    w = decay if valid_mask is None else decay * valid_mask.float()
    loss = (per_pos * w).sum() / w.sum().clamp_min(1e-6)
    return loss, {"ce": ce.mean().item(), "l1": l1.mean().item(),
                  "conf_bce": conf_bce.mean().item(), "accept_rate": alpha.mean().item()}


def _selftest():
    torch.manual_seed(0)
    # small config for CPU: noise_token_id must be < vocab_size (in the real model
    # vocab=129280 > 128799, so it's a valid token id; here we shrink both).
    cfg = DSparkConfig(vocab_size=512, dim=64, block_size=5, markov_rank=16,
                       window_size=6, noise_token_id=511)
    b, ctx = 3, 10

    mk = DSparkMarkovHead(cfg.vocab_size, cfg.markov_rank).double()
    cf = DSparkConfidenceHead(cfg.dim, cfg.markov_rank).double()

    anchor = torch.randint(0, cfg.vocab_size, (b,))
    ids = build_dspark_block(anchor, cfg.block_size, cfg.noise_token_id)
    print(f"[block]   anchor {anchor.tolist()} -> ids[0]={ids[0].tolist()} "
          f"(expect [anchor, {cfg.noise_token_id}*{cfg.block_size-1}])")
    assert (ids[:, 0] == anchor).all() and (ids[:, 1:] == cfg.noise_token_id).all()

    bias, emb = mk(ids)                       # [b, g, vocab], [b, g, rank]
    print(f"[markov]  bias {tuple(bias.shape)}  embed {tuple(emb.shape)}")
    assert bias.shape == (b, cfg.block_size, cfg.vocab_size)

    hidden = torch.randn(b, cfg.block_size, cfg.dim, dtype=torch.double)
    conf = cf(hidden, emb)                    # [b, g]
    print(f"[conf]    logits {tuple(conf.shape)}  (expect [{b},{cfg.block_size}])")
    assert conf.shape == (b, cfg.block_size)

    mask = dspark_block_mask(ctx, cfg.block_size, cfg.window_size, hidden.device)
    keep = (mask == 0).sum(-1)
    print(f"[mask]    each query attends to {keep.tolist()} keys "
          f"(expect min(window,ctx)={min(cfg.window_size,ctx)} + block {cfg.block_size} = "
          f"{min(cfg.window_size,ctx)+cfg.block_size})")

    dl = torch.randn(b, cfg.block_size, cfg.vocab_size, dtype=torch.double, requires_grad=True)
    tl = torch.randn(b, cfg.block_size, cfg.vocab_size, dtype=torch.double)
    tgt = torch.randint(0, cfg.vocab_size, (b, cfg.block_size))
    loss, stats = dspark_compound_loss(dl, tl, tgt, conf, cfg)
    loss.backward()
    print(f"[loss]    total={loss.item():.4f}  {stats}")
    print(f"[loss]    grad(draft_logits) finite={torch.isfinite(dl.grad).all().item()}  "
          f"decay w_k=exp(-(k-1)/g) applied, target detached")
    print("OK: DSpark method modules run fwd+bwd. Port to megatron per the `# MEGATRON:` notes; "
          "block attention reuses the validated einsum+sink (dspark_attn_ref_bench / dsv4_mla_ref).")


if __name__ == "__main__":
    _selftest()
