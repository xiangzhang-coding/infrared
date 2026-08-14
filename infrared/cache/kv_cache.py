"""Physical KV cache tensors + profile-based sizing (T3 — stub).

Layout follows the R1 blueprint (§5):
``[2, n_layers, n_blocks, block_size, n_kv_heads, head_dim]``, with each
attention layer holding a k/v view. The block count is decided at startup by
profiling free GPU memory. No allocation happens in this scaffold — it never
touches a GPU (no-GPU-friendly, issue #4).
"""

from __future__ import annotations

_T3 = "not implemented until T3 — see docs/spec/0001 and R1 blueprint §5"


def allocate_kv_cache(*args: object, **kwargs: object) -> object:
    """Profile free memory, size the block pool, allocate KV tensors (T3)."""
    raise NotImplementedError(_T3)
