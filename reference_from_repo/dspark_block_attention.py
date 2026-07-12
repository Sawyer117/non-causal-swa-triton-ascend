#!/usr/bin/env python3
"""DSparkMTPSelfAttention — the DSpark draft's block attention (training path),
faithful to the official `DSparkAttention.forward` (DeepSeek-V4-Flash-DSpark
inference/model.py). The mechanism that makes it a *draft*:

  * CONTEXT keys/values come from `main_x` (= main_norm(main_proj(target hidden
    [40,41,42]))) — the draft attends BACK to the target's hidden context.
  * The BLOCK query + block keys/values come from the draft block `x` [b, gamma, dim].
  * Each of the gamma block queries attends to [windowed target-context] + [the FULL
    block] (NON-causal within the block; intra-block causality is the Markov head's
    job, not the attention's), with the per-head learnable sink.

Reuses the validated pieces: `sink_attention` + MLA projections (dsv4_mla_ref),
`dspark_block_mask` (dspark_method). On-box this becomes MindSpeed's
`DSparkMTPSelfAttention` (swap nn.Linear -> Column/RowParallelLinear).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from dsv4_mla_ref import (MLAConfig, RMSNorm, apply_rotary_emb, precompute_freqs_cis,
                          sink_attention)
from dspark_method import dspark_block_mask


class DSparkMTPSelfAttention(nn.Module):
    def __init__(self, cfg: MLAConfig, block_size: int = 5):
        super().__init__()
        self.cfg = cfg
        self.block_size = block_size
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)          # shared for context & block
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.wo_a = nn.Parameter(torch.empty(cfg.o_groups, cfg.o_lora_rank,
                                             cfg.n_heads * cfg.head_dim // cfg.o_groups))
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(cfg.n_heads, dtype=torch.float32))
        self.softmax_scale = cfg.head_dim ** -0.5
        nn.init.normal_(self.wo_a, std=0.02)

    def _rope_last(self, t, freqs, inverse=False):
        rd = self.cfg.rope_head_dim
        return torch.cat([t[..., :-rd], apply_rotary_emb(t[..., -rd:], freqs, inverse=inverse)], dim=-1)

    def forward(self, x, main_x, freqs_cis):
        """x: [b, gamma, dim] draft block ; main_x: [b, ctx, dim] target-hidden context.
        freqs_cis must cover ctx + gamma positions (context at 0..ctx-1, block right after)."""
        cfg = self.cfg
        b, ctx, _ = main_x.shape
        g = x.size(1)
        rd = cfg.rope_head_dim

        # context KV from main_x (positions 0..ctx-1), then keep the last `window` (sliding window)
        main_kv = self._rope_last(self.kv_norm(self.wkv(main_x)), freqs_cis[:ctx])
        ctx_kv = main_kv[:, -cfg.window_size:]
        ctx_len = ctx_kv.size(1)

        # block q/kv from x (positions ctx..ctx+gamma-1)
        bf = freqs_cis[ctx:ctx + g]
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.eps).to(q.dtype)
        q = self._rope_last(q, bf)
        blk_kv = self._rope_last(self.kv_norm(self.wkv(x)), bf)

        kv = torch.cat([ctx_kv, blk_kv], dim=1)                          # [b, ctx_len + gamma, head_dim]
        mask = dspark_block_mask(ctx_len, g, cfg.window_size, x.device).to(kv.dtype)
        o = sink_attention(q, kv, self.attn_sink, mask, self.softmax_scale)   # [b, gamma, h, head_dim]
        o = self._rope_last(o, bf, inverse=True)

        o = o.reshape(b, g, cfg.o_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a)
        return self.wo_b(o.flatten(2))                                   # [b, gamma, dim]


def _selftest():
    torch.manual_seed(0)
    cfg = MLAConfig(dim=128, n_heads=4, head_dim=32, rope_head_dim=8,
                    q_lora_rank=48, o_lora_rank=48, o_groups=2, window_size=6, eps=1e-6)
    b, ctx, g = 2, 10, 5
    attn = DSparkMTPSelfAttention(cfg, block_size=g).double()
    x = torch.randn(b, g, cfg.dim, dtype=torch.double, requires_grad=True)
    main_x = torch.randn(b, ctx, cfg.dim, dtype=torch.double, requires_grad=True)
    freqs = precompute_freqs_cis(cfg.rope_head_dim, ctx + g, cfg.rope_theta)

    o = attn(x, main_x, freqs)
    print(f"[shape]   x{tuple(x.shape)} + main_x{tuple(main_x.shape)} -> {tuple(o.shape)}  "
          f"(expect [{b},{g},{cfg.dim}])")
    assert o.shape == (b, g, cfg.dim)

    o.sum().backward()
    print(f"[bwd]     grad(x) finite={torch.isfinite(x.grad).all().item()}  "
          f"grad(main_x) finite={torch.isfinite(main_x.grad).all().item()}  "
          f"grad(sink) nonzero={attn.attn_sink.grad.abs().sum().item():.2e}")

    # the target-hidden context (main_x) MUST affect the draft's output
    with torch.no_grad():
        o0 = attn(x, main_x, freqs)
        o1 = attn(x, main_x + 1.0, freqs)
    d = (o0 - o1).abs().mean().item()
    print(f"[context] mean_abs |o(main_x) - o(main_x+1)|={d:.3e}  "
          f"(>0 => the draft block genuinely attends to the target-hidden context)")
    assert d > 0
    print("OK: DSpark block attention runs fwd+bwd; context from main_x, block-non-causal + sink. "
          "Real parity = vs official DSparkAttention on-box (bf16, real dims).")


if __name__ == "__main__":
    _selftest()
