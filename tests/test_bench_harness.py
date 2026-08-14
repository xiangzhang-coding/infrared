"""Metrics-harness integration (Seam B): the spine runs against a real engine.

Uses a tiny random model + a real ``StaticBatchEngine`` on CPU (no GPU, no
download), so the whole (correctness, throughput, goodput, utilization) row is
produced end to end. We assert the row is *well-formed and self-consistent* —
correctness is exact (batch-invariance vs the T0 oracle), throughput is
positive, a knee sweep is produced — not specific wall-clock numbers, which are
machine-dependent (spec §Testing: relative gains, not absolute SOTA).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infrared.bench.harness import (  # noqa: E402
    LoadRequest,
    check_correctness,
    measure,
    run_load,
    t0_oracle,
    utilization_from,
)
from infrared.bench.metrics import SLO  # noqa: E402
from infrared.bench.workload import Category, Workload  # noqa: E402
from infrared.engine.engine import StaticBatchEngine  # noqa: E402
from infrared.model.config import Qwen2Config  # noqa: E402
from infrared.model.qwen2 import Qwen2ForCausalLM  # noqa: E402


def _tiny_model() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_ids=(),
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def _workload() -> Workload:
    return Workload(
        categories=[
            Category(name="short", prompts=[[3, 9, 1], [8, 2, 5, 6]], max_new_tokens=6),
            Category(name="long", prompts=[[7, 4]], max_new_tokens=10),
        ]
    )


@pytest.fixture()
def engine():
    eng = StaticBatchEngine(_tiny_model(), max_batch_size=4, linger=0.005).start()
    yield eng
    eng.stop()


def test_run_load_records_a_trace_per_request(engine) -> None:
    reqs = [LoadRequest(prompt_ids=[1, 2, 3], max_new_tokens=4) for _ in range(5)]
    result = run_load(engine, reqs, arrivals=[0.0, 0.01, 0.02, 0.03, 0.04])
    assert len(result.traces) == 5
    assert result.wall_time > 0
    for t in result.traces:
        assert t.num_output_tokens == 4
        assert t.completion >= t.arrival
    # Static batch is all-at-once: TTFT == full latency (no early first token).
    assert all(not t.is_streamed for t in result.traces)
    assert result.batch_stats, "expected at least one BatchStats recorded"


def test_correctness_matches_t0_oracle_exactly(engine) -> None:
    model = engine.model
    report = check_correctness(engine, t0_oracle(model), _workload())
    # Batch-invariance: the static-batch greedy output equals the T0 path.
    assert report.total == 3
    assert report.matched == 3
    assert report.passed
    assert set(report.per_category) == {"short", "long"}
    assert all(rate == 1.0 for rate in report.per_category.values())


def test_correctness_flags_a_wrong_oracle(engine) -> None:
    def wrong(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        return [999] * max_new_tokens  # never what the engine produces

    report = check_correctness(engine, wrong, _workload())
    assert report.matched == 0
    assert not report.passed


def test_utilization_from_stats_is_a_fraction(engine) -> None:
    reqs = [
        LoadRequest(prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=2),
        LoadRequest(prompt_ids=[7], max_new_tokens=6),
    ]
    result = run_load(engine, reqs, arrivals=None)
    util = utilization_from(result.batch_stats)
    assert util.batch_fill_rate is not None
    assert 0.0 < util.batch_fill_rate <= 1.0
    assert util.prompt_pad_fraction is not None
    assert 0.0 <= util.prompt_pad_fraction <= 1.0
    assert util.kv_block_occupancy is None  # no paged blocks until T3


def test_measure_produces_a_wellformed_ladder_row(engine) -> None:
    result = measure(
        engine,
        oracle=t0_oracle(engine.model),
        workload=_workload(),
        slo=SLO.from_ms(ttft_ms=500, tpot_ms=200),
        rates=[2.0, 20.0, 200.0],
        seed=0,
        notes="tiny CPU model",
    )
    row = result.row
    assert row.tier == "T1 static batch"
    assert row.correctness is not None and row.correctness.passed
    assert row.throughput_toks_per_s is not None and row.throughput_toks_per_s > 0
    assert row.utilization is not None
    assert row.goodput_reqs_per_s is not None and row.goodput_reqs_per_s >= 0
    # A knee sweep with one point per offered rate was produced.
    assert [p.offered_rate for p in result.sweep] == [2.0, 20.0, 200.0]
