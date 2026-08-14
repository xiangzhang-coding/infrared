"""Per-request KV cache — T0 contiguous implementation.

T0 owns a simple **contiguous** cache: one preallocated ``[max_len, ...]`` slab
per layer, appended to as generation advances. The ``update`` call shape
(append this step's K/V, return the full history for the layer) is deliberately
the seam that the T3 PagedAttention block manager will reimplement — same
signature, block-backed storage instead of a contiguous slab.

Nothing here profiles GPU memory (that sizing is a T3 concern, R1 §5); T0 just
sizes to ``prompt + max_new_tokens`` so 0.5B runs on CPU.
"""

from __future__ import annotations

import torch


class KVCache:
    """Contiguous per-request key/value cache, one slab per layer."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (num_layers, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V at ``start_pos`` and return the full history.

        ``k`` / ``v`` are ``[S, num_kv_heads, head_dim]``. Returns the layer's
        cached K/V over positions ``[0, start_pos + S)``.
        """
        seq = k.shape[0]
        end = start_pos + seq
        if end > self.max_len:
            raise ValueError(
                f"KV cache overflow: need {end} slots, capacity {self.max_len}"
            )
        self.k[layer_idx, start_pos:end] = k
        self.v[layer_idx, start_pos:end] = v
        return self.k[layer_idx, :end], self.v[layer_idx, :end]
