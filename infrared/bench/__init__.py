"""Metrics harness — the "done" spine (ADR-0002, ``CONTEXT.md`` §Metrics).

Produces one (correctness, throughput, goodput, utilization) row per config and
drives the before→after ladder (静态批 → 连续批 → +paged → +Triton), per ADR-0002
and ``CONTEXT.md`` §Metrics. Relative per-tier gains are the deliverable, not
absolute SOTA numbers.

The pure math lives in ``bench.metrics`` / ``bench.workload`` / ``bench.report``
(no torch); ``bench.harness`` is the driver, and ``python -m infrared.bench``
runs the whole thing (``bench.__main__``).
"""
