"""Physical paged K/V pool + the write/gather seam (T3).

``PagedKVPool`` is the tensor side of PagedAttention: one big pool of fixed-size
blocks, shared by every sequence, that the ``BlockManager`` hands out ids into.
It is stored flat as ``[num_layers, num_blocks * block_size, num_kv_heads,
head_dim]`` so a token at physical slot ``s`` is just row ``s`` — a sequence's
(possibly non-contiguous) history is read with one **gather** over its slot ids,
and a step's new K/V is written with one **scatter**.

``PagedContext`` bundles the per-forward metadata the attention layer needs:
which flat slots this step writes (``write_slots``) and, per query row, which
slots to gather its history from (``gather_slots``, right-padded). This is the
naive-PyTorch paged path the issue asks for; the Triton ``store_kvcache`` /
paged-attn kernel that fuses scatter+gather+attention is T4 (R1 §5, §8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class PagedKVPool:
    """Shared flat K/V slot pool; addressed by physical slot id (block*size+off)."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        shape = (num_layers, self.num_slots, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor
    ) -> None:
        """Scatter this step's K/V into their physical slots.

        ``k`` / ``v`` are ``[N, num_kv_heads, head_dim]`` and ``slots`` is a
        ``[N]`` long tensor of flat physical slot ids (one per new token).
        """
        self.k[layer_idx, slots] = k
        self.v[layer_idx, slots] = v

    def gather(
        self, layer_idx: int, gather_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather per-query histories: ``gather_slots`` ``[B, T]`` -> ``[B, T, H, D]``.

        Padded (invalid) positions in ``gather_slots`` may point anywhere (slot 0
        by convention); the caller's additive mask blocks them out, so their
        contents never reach the softmax.
        """
        b, t = gather_slots.shape
        flat = gather_slots.reshape(-1)
        k = self.k[layer_idx, flat].reshape(b, t, *self.k.shape[2:])
        v = self.v[layer_idx, flat].reshape(b, t, *self.v.shape[2:])
        return k, v


@dataclass(slots=True)
class PagedContext:
    """Per-forward paged metadata handed to the attention layer.

    ``write_slots`` ``[N]`` — flat physical slots for this step's new K/V, in the
    row-major order of the query tokens (prefill: the one sequence's ``S`` tokens;
    decode: one slot per batched sequence). ``gather_slots`` ``[B, T]`` — for each
    query row, the slots of its full history (right-padded). The pool is shared;
    the attention layer indexes it per ``layer_idx``.
    """

    pool: PagedKVPool
    write_slots: torch.Tensor
    gather_slots: torch.Tensor
