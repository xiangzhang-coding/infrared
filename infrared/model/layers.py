"""Qwen2.5 transformer building blocks (T0).

Hand-written to match HF ``transformers`` Qwen2 numerically (the Seam-A parity
gate depends on it). Qwen2-specific facts baked in (R2 §3): QKV projections have
a **bias**, O and MLP do **not**; RoPE ``theta=1e6``; RMSNorm ``eps=1e-6``;
grouped-query attention repeats each KV head ``num_heads // num_kv_heads`` times.

We **own** the attention + KV path (``causal_attention`` + the ``Attention``
module read/write an external KV cache); embedding/MLP/norm use standard
``torch`` ops. The KV interface here is what the T3 paged block manager will
reimplement behind the same call shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from infrared.model.config import Qwen2Config


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
    """Apply RoPE to q and k. q/k are ``[S, H, D]``; cos/sin are ``[S, D]``."""
    # Cast the (fp32) tables to the activation dtype, as HF does, so RoPE never
    # silently upcasts activations (a no-op in fp32; matters once T1 runs bf16).
    cos = cos.unsqueeze(1).to(q.dtype)  # [S, 1, D] broadcasts over heads
    sin = sin.unsqueeze(1).to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[S, H_kv, D]`` to ``[S, H_kv * n_rep, D]`` for GQA.

    Output head ``j`` maps to KV head ``j // n_rep`` (matches HF ``repeat_kv``).
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Single-request scaled-dot-product attention with an explicit causal mask.

    Shapes: q ``[Sq, H, D]``; k, v ``[Sk, H, D]`` (already repeated to H heads).
    ``q_pos`` / ``k_pos`` are absolute token positions, so this works for both
    prefill (Sq == Sk, triangular) and decode (Sq == 1 attending all history).
    Softmax is computed in fp32 for parity, then cast back.
    """
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale  # [H, Sq, Sk]
    mask = k_pos[None, :] > q_pos[:, None]  # [Sq, Sk]: a key after the query
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
    return torch.einsum("hqk,khd->qhd", attn, v)  # [Sq, H, D]


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
        # positions [S] -> freqs [S, D/2] -> emb [S, D]
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: object,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(seq, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Append this step's K/V and read back the full history for this layer.
        k_all, v_all = kv_cache.update(layer_idx, k, v, start_pos)
        k_all = repeat_kv(k_all, self.n_rep)
        v_all = repeat_kv(v_all, self.n_rep)

        total = k_all.shape[0]
        q_pos = torch.arange(start_pos, start_pos + seq, device=x.device)
        k_pos = torch.arange(total, device=x.device)
        out = causal_attention(q, k_all, v_all, q_pos, k_pos, self.scale)

        out = out.reshape(seq, self.num_heads * self.head_dim)
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
