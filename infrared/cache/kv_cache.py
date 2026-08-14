"""Per-request KV cache — contiguous, batch-first (T0 single-request = B=1).

T0 owns a simple **contiguous** cache: one preallocated slab per layer, appended
to as generation advances. Since T1 (static batching) all sequences in a batch
share one padded frame and advance columns in lockstep, so the cache carries a
batch dim ``[num_layers, B, max_len, num_kv_heads, head_dim]`` and every sequence
writes/reads the same column range.

The ``update`` call shape (append this step's K/V at a column, return the full
history) is deliberately the seam the T3 PagedAttention block manager will
reimplement — same signature, block-backed storage instead of a contiguous slab.
"""

from __future__ import annotations

import torch


class KVCache:
    """Contiguous batch-first key/value cache, one slab per layer."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (num_layers, batch_size, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.batch_size = batch_size
        self.max_len = max_len

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_col: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V at ``start_col`` and return the full history.

        ``k`` / ``v`` are ``[B, S, num_kv_heads, head_dim]``. Returns the layer's
        cached K/V over columns ``[0, start_col + S)`` as ``[B, T, ...]``.
        """
        seq = k.shape[1]
        end = start_col + seq
        if end > self.max_len:
            raise ValueError(
                f"KV cache overflow: need {end} columns, capacity {self.max_len}"
            )
        self.k[layer_idx, :, start_col:end] = k
        self.v[layer_idx, :, start_col:end] = v
        return self.k[layer_idx, :, :end], self.v[layer_idx, :, :end]
