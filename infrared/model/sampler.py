"""Token sampling (T0 — stub): greedy first, then temperature / top-p.

The correctness gate (Seam A, ``docs/spec/0001`` §Testing) runs greedy with a
fixed seed and compares logits against HF ``transformers`` (eager) on the same
weights — so the greedy path here must be numerically boring on purpose.
"""

from __future__ import annotations

_T0 = "not implemented until T0 — see docs/spec/0001 §Testing"


class Sampler:
    """Maps logits → next-token id per the request's sampling params."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T0)
