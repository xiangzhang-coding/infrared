"""Render ladder rows and knee sweeps to Markdown / CSV (pure Python).

The killer artifact of ADR-0002 is the ``static → continuous → +paged →
+Triton`` before→after table. These renderers turn the computed ``LadderRow`` /
``SweepPoint`` data into a table you can paste into a PR or the README — the
harness stays about *measuring*, this stays about *presenting*.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from infrared.bench.metrics import LadderRow, SweepPoint

_DASH = "—"


def _fmt(value: float | None, spec: str = ".1f") -> str:
    return _DASH if value is None else format(value, spec)


def _ms(seconds: float | None) -> str:
    return _DASH if seconds is None else f"{seconds * 1000:.1f}"


def _pct(rate: float | None) -> str:
    return _DASH if rate is None else f"{rate * 100:.0f}%"


def _correctness_cell(row: LadderRow) -> str:
    c = row.correctness
    if c is None:
        return _DASH
    mark = "✓" if c.passed else "✗"
    return f"{mark} {c.matched}/{c.total}"


def _fill_cell(row: LadderRow) -> str:
    return _DASH if row.utilization is None else _pct(row.utilization.batch_fill_rate)


def _gpu_cell(row: LadderRow) -> str:
    if row.utilization is None or row.utilization.gpu_util_pct is None:
        return _DASH
    return f"{row.utilization.gpu_util_pct:.0f}%"


_LADDER_HEADERS = (
    "Tier",
    "Correctness",
    "Throughput tok/s",
    "Goodput req/s",
    "Knee req/s",
    "Batch-fill",
    "GPU util",
    "Notes",
)


def render_ladder_markdown(rows: Sequence[LadderRow]) -> str:
    """Render the before→after ladder as a GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(_LADDER_HEADERS) + " |",
        "| " + " | ".join("---" for _ in _LADDER_HEADERS) + " |",
    ]
    for row in rows:
        cells = [
            row.tier,
            _correctness_cell(row),
            _fmt(row.throughput_toks_per_s),
            _fmt(row.goodput_reqs_per_s, ".2f"),
            _fmt(row.knee_rate, ".2f"),
            _fill_cell(row),
            _gpu_cell(row),
            row.notes or _DASH,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_CSV_HEADERS = (
    "tier",
    "correctness_overall",
    "correctness_passed",
    "throughput_toks_per_s",
    "goodput_reqs_per_s",
    "knee_rate",
    "batch_fill_rate",
    "prompt_pad_fraction",
    "gpu_util_pct",
    "kv_block_occupancy",
    "notes",
)


def render_ladder_csv(rows: Sequence[LadderRow]) -> str:
    """Render the ladder as CSV (machine-readable, one row per tier)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_HEADERS)
    for row in rows:
        c = row.correctness
        u = row.utilization
        writer.writerow(
            [
                row.tier,
                "" if c is None else f"{c.overall:.4f}",
                "" if c is None else c.passed,
                _blank(row.throughput_toks_per_s),
                _blank(row.goodput_reqs_per_s),
                _blank(row.knee_rate),
                _blank(None if u is None else u.batch_fill_rate),
                _blank(None if u is None else u.prompt_pad_fraction),
                _blank(None if u is None else u.gpu_util_pct),
                _blank(None if u is None else u.kv_block_occupancy),
                row.notes,
            ]
        )
    return buf.getvalue()


def _blank(value: float | None) -> str:
    return "" if value is None else repr(value)


_SWEEP_HEADERS = (
    "Offered req/s",
    "Achieved req/s",
    "p99 TTFT (ms)",
    "p99 TPOT (ms)",
    "Throughput tok/s",
    "Goodput req/s",
    "SLO",
)


def render_sweep_markdown(points: Sequence[SweepPoint]) -> str:
    """Render a request-rate sweep (the knee curve) as a Markdown table."""
    lines = [
        "| " + " | ".join(_SWEEP_HEADERS) + " |",
        "| " + " | ".join("---" for _ in _SWEEP_HEADERS) + " |",
    ]
    for p in points:
        cells = [
            f"{p.offered_rate:.2f}",
            f"{p.achieved_rate:.2f}",
            _ms(p.p99_ttft_s),
            _ms(p.p99_tpot_s),
            f"{p.throughput_toks_per_s:.1f}",
            f"{p.goodput_reqs_per_s:.2f}",
            "✓" if p.slo_met else "✗",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
