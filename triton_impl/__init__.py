"""Triton implementation of SWA-non-causal-sink attention (see ../README.md)."""
from .swa_sink_fwd import swa_sink_attn_fwd, swa_noncausal_sink_attn_fwd

__all__ = ["swa_sink_attn_fwd", "swa_noncausal_sink_attn_fwd"]
