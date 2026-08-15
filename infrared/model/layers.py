"""Qwen2.5 transformer building blocks (T0 math, batch-first since T1).

Hand-written to match HF ``transformers`` Qwen2 numerically (the Seam-A parity
gate depends on it). Tensors are batch-first ``[B, S, ...]``; single-request T0
is just ``B = 1``. Static batching (T1) pads a batch to a common width and drives
attention with an **additive mask** (causal + padding), which is why the core
attention takes a precomputed mask rather than positions.

Qwen2 facts baked in (R2 §3): QKV projections have a **bias**, O and MLP do
**not**; RoPE ``theta=1e6``; RMSNorm ``eps=1e-6``; GQA repeats each KV head
``num_heads // num_kv_heads`` times. We **own** the attention + KV path; the KV
``update`` call is the seam the T3 paged block manager will reimplement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from infrared.model.config import Qwen2Config
from infrared.model.triton_attention import paged_attention

if TYPE_CHECKING:
    from infrared.cache.kv_cache import KVCache
    from infrared.cache.paged_kv_cache import PagedContext


@dataclass(slots=True)
class ForwardContext:
    """Per-step inputs shared by every layer of one forward pass.

    Bundles what used to be threaded as loose args through ``Qwen2Model`` →
    ``DecoderLayer`` → ``Attention`` (the RoPE tables, the additive attention
    mask, the KV cache, and the shared start column). ``layer_idx`` stays a
    separate argument since it varies per layer.

    Two KV backends share this context. The **contiguous** path (T0/T1/T2) uses
    ``kv_cache`` + ``start_col`` — one padded frame, shared column. The **paged**
    path (T3) sets ``paged`` instead: a shared block pool addressed per-token by
    scatter/gather (``kv_cache``/``start_col`` are then unused). Attention picks
    the backend by whether ``paged`` is ``None``.
    """

    cos: torch.Tensor
    sin: torch.Tensor
    mask: torch.Tensor
    kv_cache: KVCache | None
    start_col: int
    paged: PagedContext | None = None


class RMSNorm(nn.Module):
    """Root-mean-square norm. Normalizes in fp32 then casts back (HF parity)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by halves: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k. q/k are ``[B, S, H, D]``; cos/sin are ``[B, S, D]``."""
    # Cast the (fp32) tables to the activation dtype, as HF does, so RoPE never
    # silently upcasts activations (a no-op in fp32; matters once T1 runs bf16).
    cos = cos.unsqueeze(2).to(q.dtype)  # [B, S, 1, D] broadcasts over heads
    sin = sin.unsqueeze(2).to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, S, H_kv, D]`` to ``[B, S, H_kv * n_rep, D]`` for GQA.

    Output head ``j`` maps to KV head ``j // n_rep`` (matches HF ``repeat_kv``).
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=2)


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Batched scaled-dot-product attention with an additive mask.

    Shapes: q ``[B, Sq, H, D]``; k, v ``[B, Sk, H, D]`` (already repeated to H
    heads); ``mask`` broadcasts to ``[B, H, Sq, Sk]`` (0 where allowed,
    ``finfo.min`` where disallowed — covers both causality and padding). Softmax
    is computed in fp32 for parity, then cast back.
    """
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale + mask
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE ``cos``/``sin`` for given absolute positions."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Not a learned weight and absent from HF checkpoints -> non-persistent.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions [B, S] -> freqs [B, S, D/2] -> emb [B, S, D]
        freqs = positions.to(torch.float32)[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Attention(nn.Module):
    """Grouped-query attention that reads/writes an external per-request KV cache."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        h = config.hidden_size
        # Qwen2: QKV carry a bias, O does not (R2 §3).
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)

    def forward(
        self, x: torch.Tensor, ctx: ForwardContext, layer_idx: int
    ) -> torch.Tensor:
        b, seq, _ = x.shape
        q = self.q_proj(x).view(b, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, seq, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, ctx.cos, ctx.sin)

        # Append this step's (rotated) K/V and read back the full history. Two
        # backends: contiguous (shared column) or paged (scatter/gather by slot).
        if ctx.paged is None:
            assert ctx.kv_cache is not None
            k_all, v_all = ctx.kv_cache.update(layer_idx, k, v, ctx.start_col)
            k_all = repeat_kv(k_all, self.n_rep)
            v_all = repeat_kv(v_all, self.n_rep)
            out = masked_attention(q, k_all, v_all, ctx.mask, self.scale)
        else:
            # Paged path: scatter K/V into the pool, then attend over the gathered
            # history — via the fused Triton kernel on CUDA (T4c), else the naive
            # PyTorch gather+masked_attention fallback (numerically equivalent).
            out = paged_attention(
                q, k, v, ctx.paged, ctx.mask, layer_idx, self.scale, self.n_rep
            )
        out = out.reshape(b, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (no biases)."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
