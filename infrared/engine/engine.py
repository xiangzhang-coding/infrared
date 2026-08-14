"""Engine busy loop — owns the request lifecycle (T1 — stub).

``add_request`` turns a prompt into a ``Sequence`` on the waiting queue; ``step``
drives the three-stage cycle (schedule → worker forward+sample → postprocess);
``generate`` runs steps until every sequence finishes (R1 §3.1). The call into
the Worker is same-process for now, but the seam stays serializable for T6
(R1 §6).
"""

from __future__ import annotations

_T1 = "not implemented until T1 — see docs/spec/0001 and R1 blueprint §3.1"


class Engine:
    """Single-process inference engine: add_request / step / generate."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T1)

    def add_request(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T1)

    def step(self) -> object:
        """One iteration == one model forward pass (continuous batching)."""
        raise NotImplementedError(_T1)
