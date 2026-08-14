"""Metrics harness — the "done" spine (T5 — stub).

Produces one (correctness, throughput, goodput, utilization) row per config and
drives the before→after ladder (静态批 → 连续批 → +paged → +Triton), per ADR-0002
and ``CONTEXT.md`` §Metrics. Relative per-tier gains are the deliverable, not
absolute SOTA numbers.
"""
