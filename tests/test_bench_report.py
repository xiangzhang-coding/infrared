"""Report renderer tests (pure, no torch)."""

from __future__ import annotations

from infrared.bench.metrics import (
    CorrectnessReport,
    LadderRow,
    SweepPoint,
    Utilization,
)
from infrared.bench.report import (
    render_ladder_csv,
    render_ladder_markdown,
    render_sweep_markdown,
)


def _row() -> LadderRow:
    return LadderRow(
        tier="T1 static batch",
        correctness=CorrectnessReport(
            per_category={"decode-heavy": 1.0}, matched=8, total=8
        ),
        throughput_toks_per_s=123.4,
        goodput_reqs_per_s=2.5,
        knee_rate=4.0,
        utilization=Utilization(
            batch_fill_rate=0.5, prompt_pad_fraction=0.25, gpu_util_pct=None
        ),
        notes="CPU dev run",
    )


def test_ladder_markdown_has_header_and_row() -> None:
    md = render_ladder_markdown([_row()])
    assert "| Tier |" in md
    assert "T1 static batch" in md
    assert "123.4" in md  # throughput
    assert "4.0" in md  # knee
    # None values render as an em dash, not "None".
    assert "None" not in md


def test_ladder_markdown_empty() -> None:
    md = render_ladder_markdown([])
    assert "Tier" in md  # header still present


def test_ladder_csv_round_trips() -> None:
    csv = render_ladder_csv([_row()])
    lines = csv.strip().splitlines()
    assert lines[0].startswith("tier,")
    assert "T1 static batch" in lines[1]
    # One header + one data row.
    assert len(lines) == 2


def test_sweep_markdown_reports_ms_and_slo_flag() -> None:
    points = [
        SweepPoint(
            offered_rate=1.0,
            achieved_rate=1.0,
            p99_ttft_s=0.05,
            p99_tpot_s=0.01,
            throughput_toks_per_s=100.0,
            goodput_reqs_per_s=1.0,
            slo_met=True,
            num_requests=10,
        ),
        SweepPoint(
            offered_rate=8.0,
            achieved_rate=6.0,
            p99_ttft_s=2.0,
            p99_tpot_s=0.5,
            throughput_toks_per_s=300.0,
            goodput_reqs_per_s=0.0,
            slo_met=False,
            num_requests=40,
        ),
    ]
    md = render_sweep_markdown(points)
    assert "p99 TTFT" in md
    assert "50.0" in md  # 0.05s -> 50.0 ms
    assert "✓" in md and "✗" in md  # SLO met / breached markers
