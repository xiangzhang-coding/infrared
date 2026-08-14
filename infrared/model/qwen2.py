"""Qwen2.5 dense forward assembly (T0 — stub).

Wires the ``layers`` blocks into a decoder-only forward and loads weights from
HF ``safetensors`` (weights only — HF is the weight source + correctness oracle,
ADR-0005; never an execution path, ADR-0003). The weight-key → module mapping
and the tied-lm_head handling for the 0.5B model are documented in R2 §3.2.
"""

from __future__ import annotations

_T0 = "not implemented until T0 — see docs/spec/0001 §Goals(1) and R2 §3"


class Qwen2ForCausalLM:
    """Decoder-only Qwen2.5 model (RMSNorm/RoPE/GQA/SwiGLU) producing logits."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)

    @classmethod
    def from_safetensors(cls, *args: object, **kwargs: object) -> Qwen2ForCausalLM:
        """Load weights from a local safetensors checkpoint (T0)."""
        raise NotImplementedError(_T0)
