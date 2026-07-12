"""Triton implementation of SWA-non-causal-sink attention (see ../README.md)."""
from .swa_sink_fwd import (
    swa_sink_attn_fwd,          # asymmetric windowed self-attention + sink (packed-SWA view)
    dense_sink_attn_fwd,        # dense cross-attention + sink (the gold BLOCK form)
    swa_noncausal_sink_attn_fwd,  # symmetric-window compat wrapper (first-step microbench)
)

__all__ = ["swa_sink_attn_fwd", "dense_sink_attn_fwd", "swa_noncausal_sink_attn_fwd"]
