"""Token sampling (T0): greedy + temperature.

Greedy (``temperature == 0``) is deterministic argmax — the mode the Seam-A
parity gate uses so infrared and HF must agree token-for-token. Temperature
sampling scales logits then draws from the softmax; pass a seeded
``torch.Generator`` for reproducibility.
"""

from __future__ import annotations

import torch


class Sampler:
    """Maps a logits vector to the next token id."""

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> int:
        """Pick the next token from ``logits`` (shape ``[vocab]``)."""
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1))
        probs = torch.softmax(logits.to(torch.float32) / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=generator))
