"""Triton implementation of SWA-non-causal-sink attention (see ../README.md)."""
from .swa_sink_fwd import (
    swa_sink_attn,              # autograd fwd+bwd: asymmetric windowed self-attention + sink
    dense_sink_attn,           # autograd fwd+bwd: dense cross-attention + sink (gold BLOCK form)
    swa_sink_attn_fwd,         # forward-only: asymmetric windowed self-attention + sink
    dense_sink_attn_fwd,       # forward-only: dense cross-attention + sink (gold BLOCK form)
    swa_noncausal_sink_attn_fwd,  # forward-only symmetric-window compat wrapper (microbench)
)

__all__ = [
    "swa_sink_attn", "dense_sink_attn",
    "swa_sink_attn_fwd", "dense_sink_attn_fwd", "swa_noncausal_sink_attn_fwd",
]
