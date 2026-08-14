"""Continuous-batching scheduler — the CPU-side heart (T2 — stub).

Holds ``waiting`` / ``running`` deques plus a reference to the ``BlockManager``.
Each step picks a batch under a token budget (decode-first, then chunked
prefill), allocates/appends blocks, and **preempts** low-priority sequences by
recompute when blocks run out (R1 §3.2/§9). Pure decision-making — no tensor
math (that is the Worker's job, in ``infrared.model``).
"""

from __future__ import annotations

_T2 = "not implemented until T2 — see docs/spec/0001 and R1 blueprint §3.2"


class Scheduler:
    """Iteration-level admission/eviction: schedule → (worker) → postprocess."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T2)

    def schedule(self) -> object:
        """Pick this step's batch (+ per-seq block tables and token counts)."""
        raise NotImplementedError(_T2)

    def postprocess(self, *args: object, **kwargs: object) -> None:
        """Append sampled tokens, check stop conditions, recycle finished blocks."""
        raise NotImplementedError(_T2)
