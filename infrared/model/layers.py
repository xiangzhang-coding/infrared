"""Qwen2.5 transformer building blocks (T0 — stubs).

Each class documents the math it must implement and the Qwen2-specific gotchas
surfaced by R2 (``docs/research/deps-and-qwen25-arch.md`` §3): QKV projections
carry a **bias**, O and MLP do **not**; weights are stored ``[out, in]``; the
0.5B model **ties** lm_head to the token embedding. No math lives here yet —
these are import-safe placeholders (no torch import) that the T0 forward-pass
ticket turns into ``torch.nn.Module``s.
"""

from __future__ import annotations

_T0 = "not implemented until T0 — see docs/spec/0001 and R1 blueprint §5"


class RMSNorm:
    """Root-mean-square norm (pre-norm), ``rms_norm_eps=1e-6`` on Qwen2.5."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)


class RotaryEmbedding:
    """Rotary position embedding (RoPE), ``rope_theta=1_000_000`` on Qwen2.5."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)


class Attention:
    """Grouped-query attention (GQA): Q heads share KV heads (group = 7).

    T0 uses a naive PyTorch/SDPA path; paged-KV read/write and a self-written
    Triton kernel arrive at T3/T4. Per ADR-0003 this path stays in-house — no
    ``flash_attn`` dependency.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)


class SwiGLU:
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (no bias on gate/up/down)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)
