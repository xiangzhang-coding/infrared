"""Qwen2.5 dense forward assembly + safetensors weight loader (T0).

Module attribute names mirror HF's key layout (``model.embed_tokens``,
``model.layers.{i}.self_attn.q_proj``, ``model.norm``, ``lm_head``) so a raw
``state_dict`` loads directly and the Seam-A parity gate can compare like for
like. HF ``safetensors`` supplies weights only (ADR-0003/0005) — we never call
``.generate()`` and never use HF's model as an execution path.

The forward is single-request (no batch dim): ``input_ids`` is ``[S]`` and it
returns logits ``[S, vocab]``. Continuous batching arrives at T1; this stays the
correctness baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from infrared.cache.kv_cache import KVCache
from infrared.model.config import Qwen2Config
from infrared.model.layers import MLP, Attention, RMSNorm, RotaryEmbedding


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: attn then SwiGLU MLP, each residual-added."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        attn = self.self_attn(
            self.input_layernorm(x), cos, sin, kv_cache, layer_idx, start_pos
        )
        x = x + attn
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    """Embedding + stack of decoder layers + final norm (no lm_head)."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache,
        start_pos: int,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache, i, start_pos)
        return self.norm(h)


class Qwen2ForCausalLM(nn.Module):
    """Full LM: Qwen2Model + lm_head, producing next-token logits."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    def forward(
        self, input_ids: torch.Tensor, kv_cache: KVCache, start_pos: int = 0
    ) -> torch.Tensor:
        """Run one forward over ``input_ids`` ``[S]`` -> logits ``[S, vocab]``.

        ``start_pos`` is the absolute position of the first token, so a prefill
        passes the whole prompt at ``start_pos=0`` and each decode step passes a
        single token at the growing position.
        """
        seq = input_ids.shape[0]
        positions = torch.arange(start_pos, start_pos + seq, device=input_ids.device)
        cos, sin = self.rotary(positions)
        h = self.model(input_ids, cos, sin, kv_cache, start_pos)
        return self.lm_head(h)

    def new_kv_cache(self, max_len: int) -> KVCache:
        """Allocate a per-request contiguous KV cache sized for this model."""
        return KVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_len=max_len,
            dtype=self.dtype,
            device=self.device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Qwen2ForCausalLM:
        """Build the model and load HF safetensors weights from a local dir.

        Defaults to fp32 on CPU — the deterministic path the parity gate uses.
        Handles the 0.5B tied lm_head (no ``lm_head.weight`` in the checkpoint).
        """
        model_dir = Path(model_dir)
        config = Qwen2Config.from_pretrained(model_dir)
        model = cls(config)

        state = _load_state_dict(model_dir)
        state = {k: t.to(dtype) for k, t in state.items()}

        if config.tie_word_embeddings:
            # Share storage so loading embed_tokens also fills lm_head.
            model.lm_head.weight = model.model.embed_tokens.weight

        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected weight keys: {unexpected}")
        allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
        extra_missing = set(missing) - allowed_missing
        if extra_missing:
            raise ValueError(f"missing weight keys: {sorted(extra_missing)}")

        model = model.to(dtype=dtype, device=device).eval()
        if config.tie_word_embeddings:
            # Re-establish sharing: .to() may return copied tensors, silently
            # breaking the tie set before load. Reassign so it always holds.
            model.lm_head.weight = model.model.embed_tokens.weight
        return model


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load weights from a single safetensors file or a sharded index."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(model_dir / shard))
        return state
    return load_file(model_dir / "model.safetensors")
