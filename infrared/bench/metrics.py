"""The metrics spine — honest "done" numbers (ADR-0002, ``CONTEXT.md`` §Metrics).

Pure Python, **no torch**: everything here operates on already-recorded request
traces, so the definitions are deterministic and testable without an engine or a
GPU. ``bench/harness.py`` records real traces off a running engine and funnels
them through these same functions.

The three fuzzy north-star words become measurable here:

- **High concurrency → goodput / knee.** ``goodput`` is the rate of requests that
  individually meet the SLO; ``knee`` is the highest offered request-rate whose
  aggregate p99 latencies still fit the SLO — *not* saturated throughput.
- Latency is split into **TTFT** (time-to-first-token) and **TPOT** (time-per-
  output-token). A streaming engine gives both directly; a static batch returns
  all-or-nothing, so its TTFT is the *whole* latency (the head-of-line pathology
  T2 removes) and its TPOT can only be amortized — see ``RequestTrace.tpot``.

Times are in **seconds** (``time.perf_counter`` domain). ``SLO.from_ms`` is the
ergonomic constructor since serving SLOs are usually quoted in milliseconds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated ``p``-th percentile (``p`` in ``[0, 100]``).

    numpy-free so ``bench`` stays import-pure. Matches ``numpy.percentile``'s
    default ('linear') method on the interior; endpoints return min/max.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo))


@dataclass(slots=True)
class RequestTrace:
    """One request's timeline (seconds) as seen by the load driver.

    ``first_token`` is when the first output token became observable. A static
    batch has no streaming, so the driver sets ``first_token == completion``:
    that is faithful, not a bug — it is exactly why static batching's TTFT is so
    bad, and the knee curve is meant to show it.
    """

    arrival: float
    first_token: float
    completion: float
    num_output_tokens: int
    category: str = ""

    @property
    def ttft(self) -> float:
        """Time to first token."""
        return self.first_token - self.arrival

    @property
    def latency(self) -> float:
        """End-to-end latency (arrival → last token)."""
        return self.completion - self.arrival

    @property
    def is_streamed(self) -> bool:
        """True if tokens arrived incrementally (first token before completion)."""
        return self.completion > self.first_token

    @property
    def tpot(self) -> float:
        """Time per output token.

        Streamed: the inter-token gap ``(completion - first_token)/(n-1)``.
        All-at-once (static batch, ``first_token == completion``): amortized as
        ``latency / n`` — a proxy, since per-token times can't be observed.
        """
        n = self.num_output_tokens
        if n <= 0:
            return 0.0
        if self.is_streamed and n > 1:
            return (self.completion - self.first_token) / (n - 1)
        return self.latency / n


@dataclass(slots=True)
class LatencyStats:
    """Aggregate latency percentiles over a set of traces."""

    count: int
    p50_ttft_s: float
    p99_ttft_s: float
    p50_tpot_s: float
    p99_tpot_s: float


def summarize(traces: Sequence[RequestTrace]) -> LatencyStats:
    """Compute p50/p99 TTFT and TPOT over ``traces``."""
    if not traces:
        raise ValueError("cannot summarize an empty trace set")
    ttfts = [t.ttft for t in traces]
    tpots = [t.tpot for t in traces]
    return LatencyStats(
        count=len(traces),
        p50_ttft_s=percentile(ttfts, 50),
        p99_ttft_s=percentile(ttfts, 99),
        p50_tpot_s=percentile(tpots, 50),
        p99_tpot_s=percentile(tpots, 99),
    )


@dataclass(slots=True)
class SLO:
    """Service-level objective: p99 TTFT and p99 TPOT ceilings (seconds)."""

    ttft_s: float
    tpot_s: float

    @classmethod
    def from_ms(cls, ttft_ms: float, tpot_ms: float) -> SLO:
        return cls(ttft_s=ttft_ms / 1000.0, tpot_s=tpot_ms / 1000.0)


def slo_ok(stats: LatencyStats, slo: SLO) -> bool:
    """True if the aggregate p99 TTFT *and* p99 TPOT both fit the SLO."""
    return stats.p99_ttft_s <= slo.ttft_s and stats.p99_tpot_s <= slo.tpot_s


@dataclass(slots=True)
class SweepPoint:
    """One request-rate rung of the goodput sweep."""

    offered_rate: float  # req/s offered (open-loop arrival λ)
    achieved_rate: float  # completed req / wall time
    p99_ttft_s: float
    p99_tpot_s: float
    throughput_toks_per_s: float
    goodput_reqs_per_s: float  # per-request rate meeting the SLO
    slo_met: bool  # aggregate p99s within SLO
    num_requests: int


def sweep_point(
    offered_rate: float,
    traces: Sequence[RequestTrace],
    slo: SLO,
    wall_time: float,
) -> SweepPoint:
    """Fold one rate's traces into a ``SweepPoint``.

    ``goodput`` counts requests that *individually* meet the SLO (the honest
    high-concurrency number); ``slo_met`` is the aggregate p99 test that drives
    the knee.
    """
    if wall_time <= 0:
        raise ValueError("wall_time must be positive")
    if not traces:
        return SweepPoint(
            offered_rate=offered_rate,
            achieved_rate=0.0,
            p99_ttft_s=math.inf,
            p99_tpot_s=math.inf,
            throughput_toks_per_s=0.0,
            goodput_reqs_per_s=0.0,
            slo_met=False,
            num_requests=0,
        )
    stats = summarize(traces)
    good = sum(1 for t in traces if t.ttft <= slo.ttft_s and t.tpot <= slo.tpot_s)
    total_tokens = sum(t.num_output_tokens for t in traces)
    return SweepPoint(
        offered_rate=offered_rate,
        achieved_rate=len(traces) / wall_time,
        p99_ttft_s=stats.p99_ttft_s,
        p99_tpot_s=stats.p99_tpot_s,
        throughput_toks_per_s=total_tokens / wall_time,
        goodput_reqs_per_s=good / wall_time,
        slo_met=slo_ok(stats, slo),
        num_requests=len(traces),
    )


@dataclass(slots=True)
class CorrectnessReport:
    """A/B match of engine output vs. an oracle, per category (Seam A shape).

    ``passed`` requires *every* category to match exactly — the quality gate
    that catches a mechanism which keeps overall accuracy but collapses one
    category (``CONTEXT.md`` §Correctness, ADR-0002).
    """

    per_category: dict[str, float]  # category -> match rate in [0, 1]
    matched: int
    total: int

    @property
    def overall(self) -> float:
        return self.matched / self.total if self.total else 1.0

    @property
    def passed(self) -> bool:
        return bool(self.per_category) and all(
            rate >= 1.0 for rate in self.per_category.values()
        )


@dataclass(slots=True)
class Utilization:
    """Utilization evidence — "is the mechanism actually working?" (ADR-0002).

    For T1 static batching the evidence is the **batch-fill rate** (how much of
    the padded, lockstep decode grid was real work) and the **prompt-pad
    fraction**. ``fill_over_time`` is the per-batch fill sequence (batches run in
    order, so the index axis is time) — ADR-0002 §(b)'s "填充率随时间曲线".
    ``gpu_util_pct`` is filled only on CUDA (headline 4090 run; a coarse device-
    busy %, see ``harness.sample_gpu_util``) and ``kv_block_occupancy`` stays
    ``None`` until T3 introduces paged KV blocks.
    """

    batch_fill_rate: float | None = None
    prompt_pad_fraction: float | None = None
    gpu_util_pct: float | None = None
    kv_block_occupancy: float | None = None
    fill_over_time: list[float] = field(default_factory=list)


@dataclass(slots=True)
class LadderRow:
    """One rung of the ``static → continuous → +paged → +Triton`` ladder table."""

    tier: str
    correctness: CorrectnessReport | None = None
    throughput_toks_per_s: float | None = None
    goodput_reqs_per_s: float | None = None
    knee_rate: float | None = None
    utilization: Utilization | None = None
    notes: str = ""


def knee(points: Sequence[SweepPoint]) -> SweepPoint | None:
    """Highest *sustainable* offered rate: the knee before the SLO breaks.

    Scans rates low→high and stops at the first breach, so a spurious pass at a
    higher rate (CPU timing noise) can't inflate the knee. Returns ``None`` if
    even the lowest offered rate already violates the SLO.
    """
    best: SweepPoint | None = None
    for pt in sorted(points, key=lambda p: p.offered_rate):
        if not pt.slo_met:
            break
        best = pt
    return best
