"""Per-request state machine (T1 — status enum + stub).

``SequenceStatus`` starts as the nano-vLLM three-state minimum (R1 §4.1); the
richer vLLM v1 states (PREEMPTED, FINISHED_*) are deferred. A ``Sequence``
carries its token ids, cached/scheduled token counts, its logical→physical
``block_table``, and sampling params (field list in R1 §4.1). Only the enum is
defined here; the data container is filled at T1.
"""

from __future__ import annotations

from enum import Enum, auto

_T1 = "not implemented until T1 — see docs/spec/0001 and R1 blueprint §4.1"


class SequenceStatus(Enum):
    """Lifecycle states. A preempted sequence returns to WAITING (recompute)."""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """A single request's state (token ids, block_table, sampling params)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T1)
