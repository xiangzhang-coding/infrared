"""Pure metrics analysis (no torch): percentiles, TTFT/TPOT, goodput, knee.

These are the honest-number definitions of ADR-0002 / ``CONTEXT.md`` §Metrics,
tested on **synthetic** traces so the math is deterministic and independent of
any engine or GPU. The load driver in ``bench/harness.py`` feeds real traces
through the exact same functions.
"""

from __future__ import annotations

import math

import pytest

from infrared.bench.metrics import (
    SLO,
    RequestTrace,
    SweepPoint,
    knee,
    percentile,
    summarize,
    sweep_point,
)


def test_percentile_linear_interpolation() -> None:
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)
    assert percentile([10, 20, 30, 40], 0) == 10
    assert percentile([10, 20, 30, 40], 100) == 40
    # p99 of 1..100 interpolates between the 99th and 100th values.
    assert percentile(list(range(1, 101)), 99) == pytest.approx(99.01)


def test_percentile_single_and_empty() -> None:
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_trace_ttft_latency_and_streamed_tpot() -> None:
    # Streamed: first token at +0.1s, 5 tokens ending at +0.5s -> 4 inter-token gaps.
    t = RequestTrace(arrival=1.0, first_token=1.1, completion=1.5, num_output_tokens=5)
    assert t.ttft == pytest.approx(0.1)
    assert t.latency == pytest.approx(0.5)
    assert t.is_streamed
    assert t.tpot == pytest.approx((1.5 - 1.1) / 4)


def test_trace_all_at_once_tpot_is_amortized() -> None:
    # Static batch delivers all tokens at completion: first_token == completion.
    # TTFT is the *whole* latency (the pathology T2 fixes) and TPOT amortizes.
    t = RequestTrace(arrival=0.0, first_token=0.8, completion=0.8, num_output_tokens=4)
    assert not t.is_streamed
    assert t.ttft == pytest.approx(0.8)
    assert t.tpot == pytest.approx(0.8 / 4)  # latency / n


def test_summarize_p99() -> None:
    traces = [
        RequestTrace(
            arrival=0.0,
            first_token=float(i) / 100,
            completion=1.0 + i / 100,
            num_output_tokens=10,
        )
        for i in range(1, 101)
    ]
    stats = summarize(traces)
    assert stats.count == 100
    assert stats.p99_ttft_s == pytest.approx(percentile([t.ttft for t in traces], 99))
    assert stats.p99_tpot_s == pytest.approx(percentile([t.tpot for t in traces], 99))


def test_slo_from_ms() -> None:
    slo = SLO.from_ms(ttft_ms=200, tpot_ms=50)
    assert slo.ttft_s == pytest.approx(0.2)
    assert slo.tpot_s == pytest.approx(0.05)


def test_sweep_point_goodput_counts_per_request_slo() -> None:
    slo = SLO(ttft_s=0.5, tpot_s=0.1)
    # 4 requests over a 2s window: 3 meet SLO, 1 blows TTFT.
    good = [
        RequestTrace(0.0, 0.1, 0.6, num_output_tokens=6),  # ttft .1, tpot .1 -> ok
        RequestTrace(0.0, 0.2, 0.7, num_output_tokens=6),  # ttft .2, tpot .1 -> ok
        RequestTrace(0.0, 0.3, 0.8, num_output_tokens=6),  # ttft .3, tpot .1 -> ok
    ]
    bad = [RequestTrace(0.0, 0.9, 1.4, num_output_tokens=6)]  # ttft .9 > .5 -> bad
    pt = sweep_point(offered_rate=2.0, traces=good + bad, slo=slo, wall_time=2.0)
    assert pt.num_requests == 4
    assert pt.achieved_rate == pytest.approx(2.0)
    assert pt.goodput_reqs_per_s == pytest.approx(3 / 2.0)  # 3 good / 2s
    assert pt.throughput_toks_per_s == pytest.approx(24 / 2.0)  # 4*6 tokens / 2s


def test_knee_is_highest_sustainable_offered_rate() -> None:
    def pt(rate: float, met: bool) -> SweepPoint:
        return SweepPoint(
            offered_rate=rate,
            achieved_rate=rate,
            p99_ttft_s=0.0,
            p99_tpot_s=0.0,
            throughput_toks_per_s=0.0,
            goodput_reqs_per_s=0.0,
            slo_met=met,
            num_requests=1,
        )

    # Meets SLO up to rate=2, breaks at 3; a spurious pass at 4 must NOT count.
    points = [pt(1, True), pt(2, True), pt(3, False), pt(4, True)]
    assert knee(points).offered_rate == 2

    # SLO broken even at the lowest offered rate -> no sustainable knee.
    assert knee([pt(1, False), pt(2, False)]) is None

    # All rates meet SLO -> the knee is the highest one measured.
    assert knee([pt(1, True), pt(5, True), pt(3, True)]).offered_rate == 5


def test_knee_empty() -> None:
    assert knee([]) is None
    assert not math.isnan(percentile([1.0, 2.0], 50))
