"""Workload + arrival-process tests (pure, no torch)."""

from __future__ import annotations

import statistics

import pytest

from infrared.bench.workload import (
    Category,
    Workload,
    decode_heavy_category,
    poisson_arrivals,
)


def test_poisson_arrivals_deterministic_with_seed() -> None:
    a = poisson_arrivals(rate=5.0, n=50, seed=7)
    b = poisson_arrivals(rate=5.0, n=50, seed=7)
    c = poisson_arrivals(rate=5.0, n=50, seed=8)
    assert a == b
    assert a != c
    assert len(a) == 50


def test_poisson_arrivals_are_monotonic_and_nonnegative() -> None:
    a = poisson_arrivals(rate=10.0, n=200, seed=1)
    assert a[0] >= 0.0
    assert all(y >= x for x, y in zip(a, a[1:], strict=False))


def test_poisson_mean_inter_arrival_matches_rate() -> None:
    rate = 4.0
    a = poisson_arrivals(rate=rate, n=5000, seed=3)
    gaps = [y - x for x, y in zip([0.0, *a], a, strict=False)]
    # Mean inter-arrival of a Poisson process is 1/rate; loose bound for n=5000.
    assert statistics.mean(gaps) == pytest.approx(1 / rate, rel=0.1)


def test_poisson_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="rate"):
        poisson_arrivals(rate=0.0, n=10, seed=1)


def test_decode_heavy_category_shape_and_determinism() -> None:
    cat = decode_heavy_category(
        n=8, prompt_len=4, max_new_tokens=32, vocab_size=64, seed=0
    )
    assert cat.name == "decode-heavy"
    assert len(cat.prompts) == 8
    assert all(len(p) == 4 for p in cat.prompts)
    assert all(1 <= tok < 64 for p in cat.prompts for tok in p)
    assert cat.max_new_tokens == 32
    # Same seed reproduces the exact token ids.
    again = decode_heavy_category(
        n=8, prompt_len=4, max_new_tokens=32, vocab_size=64, seed=0
    )
    assert cat.prompts == again.prompts


def test_workload_items_flattens_categories() -> None:
    wl = Workload(
        categories=[
            Category(name="short", prompts=[[1, 2], [3, 4]], max_new_tokens=8),
            Category(name="long", prompts=[[5]], max_new_tokens=16),
        ]
    )
    items = list(wl.items())
    assert len(items) == 3
    assert items[0] == ("short", [1, 2], 8)
    assert items[-1] == ("long", [5], 16)
    assert wl.num_requests == 3
