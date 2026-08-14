"""Qwen2.5 architecture config (data only — no torch).

Populated from the model's own ``config.json`` (and ``generation_config.json``
for stop tokens), so the hyperparameters are first-party, not hardcoded guesses.
Values for Qwen2.5-0.5B/7B are cross-checked in R2
(``docs/research/deps-and-qwen25-arch.md`` §2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Qwen2Config:
    """Dense Qwen2.5 hyperparameters (the subset the forward pass needs)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    # Chat stop tokens. Qwen2.5 stops on <|im_end|> (151645) or <|endoftext|>
    # (151643); the pair comes from generation_config.json (R2 §5).
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen2Config:
        """Read ``config.json`` (+ ``generation_config.json``) from a local dir."""
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())

        num_heads = cfg["num_attention_heads"]
        # Qwen2.5 configs omit head_dim; derive it (896/14=64, 3584/28=128).
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // num_heads

        # Prefer the (possibly multi-value) stop set from generation_config.
        eos: object = cfg.get("eos_token_id")
        gen_path = model_dir / "generation_config.json"
        if gen_path.exists():
            gen = json.loads(gen_path.read_text())
            eos = gen.get("eos_token_id", eos)
        if isinstance(eos, list):
            eos_ids = tuple(eos)
        elif eos is not None:
            eos_ids = (eos,)
        else:
            eos_ids = ()

        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg.get("rope_theta", 1_000_000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            bos_token_id=cfg.get("bos_token_id", 151643),
            eos_token_ids=eos_ids,
        )
