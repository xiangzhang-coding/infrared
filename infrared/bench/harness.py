"""Measurement harness (T5 — stub).

``measure`` runs a workload against a given engine config and returns one row of
the ladder: goodput@SLO / knee / GPU util / KV-block occupancy / batch-fill rate
(``CONTEXT.md`` §Metrics, ADR-0002). No load generation happens in the scaffold.
"""

from __future__ import annotations

_T5 = "not implemented until T5 — see docs/adr/0002 and CONTEXT.md §Metrics"


def measure(*args: object, **kwargs: object) -> object:
    """Return one (correctness, throughput, goodput, utilization) ladder row."""
    raise NotImplementedError(_T5)
