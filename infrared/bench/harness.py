"""Metrics harness — the "done" spine (ADR-0002, ``CONTEXT.md`` §Metrics).

``measure`` runs a workload against one engine config and returns a single
``LadderRow`` (correctness / throughput / goodput / utilization) plus the knee
sweep behind it. ``build_ladder`` stacks those rows into the ``static →
continuous → +paged → +Triton`` before→after table — the killer artifact.

**Seam B** (spec §Testing): this harness produces one honest row for *any*
config and drives the ladder; relative per-tier gains are the deliverable, not
absolute SOTA. The pure math lives in ``bench.metrics`` / ``bench.workload`` /
``bench.report`` (tested without torch); this module is the thin driver that
records real traces off a running engine.

**Import purity:** this module imports only stdlib + the pure ``bench`` helpers
at load time (the no-GPU smoke test imports it). torch, the engine's
``BatchRequest``, and the T0 oracle are imported lazily inside the functions
that actually run an engine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from infrared.bench.metrics import (
    SLO,
    CorrectnessReport,
    LadderRow,
    RequestTrace,
    SweepPoint,
    Utilization,
    knee,
    sweep_point,
)
from infrared.bench.report import render_ladder_markdown, render_sweep_markdown
from infrared.bench.workload import (
    Category,
    Workload,
    decode_heavy_category,
    poisson_arrivals,
)

if TYPE_CHECKING:  # hints only — never imported at runtime (keeps load torch-free)
    from infrared.engine.engine import (
        ContinuousBatchEngine,
        ContinuousPending,
        Pending,
        StaticBatchEngine,
    )
    from infrared.engine.static_batch import BatchStats
    from infrared.model.qwen2 import Qwen2ForCausalLM

    # Either engine drives the harness through the same submit/Pending surface.
    Engine = StaticBatchEngine | ContinuousBatchEngine
    Pendings = Pending | ContinuousPending

# A greedy reference: (prompt_ids, max_new_tokens) -> generated token ids.
Oracle = Callable[[list[int], int], list[int]]


@dataclass(slots=True)
class LoadRequest:
    """One request to offer the engine (torch-free; built into a ``BatchRequest``)."""

    prompt_ids: list[int]
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    category: str = ""


@dataclass(slots=True)
class LoadResult:
    """Everything one load run recorded: per-request traces + waste stats."""

    traces: list[RequestTrace]
    wall_time: float
    batch_stats: list[BatchStats] = field(default_factory=list)  # unique batches seen


def run_load(
    engine: Engine,
    requests: Sequence[LoadRequest],
    arrivals: Sequence[float] | None = None,
    timeout: float = 120.0,
) -> LoadResult:
    """Offer ``requests`` to a (started) engine, open-loop, and record traces.

    With ``arrivals`` (offsets in seconds, one per request) submissions follow a
    schedule regardless of whether the engine keeps up — the back-pressure that
    reveals the knee. With ``arrivals=None`` every request is offered at once (a
    burst), which is what ``measure_throughput`` uses.

    Each request gets a waiter thread that stamps its completion the moment the
    engine's event fires, so a slow neighbour can't smear another request's
    completion time. If the engine streams (T2 continuous batch exposes
    ``Pending.first_token_time``), the trace's ``first_token`` is that real TTFT;
    a static batch has no streaming, so ``first_token`` falls back to
    ``completion`` (faithful: static batching genuinely delivers all-or-nothing —
    the head-of-line pathology the knee curve exposes).
    """
    from infrared.engine.static_batch import BatchRequest

    n = len(requests)
    if arrivals is not None and len(arrivals) != n:
        raise ValueError("arrivals must have one offset per request")

    # Drop any per-step stats left over from a prior run (e.g. the correctness
    # A/B before this one) so this run's fill curve is scoped to this run only,
    # regardless of call order in ``measure``.
    if hasattr(engine, "pop_step_stats"):
        engine.pop_step_stats()

    # (completion_time, num_output_tokens, first_token_time_or_None)
    completions: list[tuple[float, int, float | None] | None] = [None] * n
    errors: list[BaseException | None] = [None] * n
    pendings: list[Pendings | None] = [None] * n
    threads: list[threading.Thread] = []

    def wait_one(idx: int, pending: Pending) -> None:
        try:
            output = pending.result(timeout)
            # Streaming engines stamp first_token_time before completion; static
            # ones don't expose it -> None -> first_token collapses to completion.
            ft = getattr(pending, "first_token_time", None)
            completions[idx] = (time.perf_counter(), len(output), ft)
        except BaseException as exc:  # noqa: BLE001 — recorded, re-raised below
            errors[idx] = exc

    arrival_times: list[float] = [0.0] * n
    t0 = time.perf_counter()
    for i, req in enumerate(requests):
        if arrivals is not None:
            due = t0 + arrivals[i]
            gap = due - time.perf_counter()
            if gap > 0:
                time.sleep(gap)
        arrival_times[i] = time.perf_counter()
        pending = engine.submit(
            BatchRequest(
                prompt_ids=req.prompt_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                seed=req.seed,
                eos_token_ids=req.eos_token_ids,
            )
        )
        pendings[i] = pending
        th = threading.Thread(target=wait_one, args=(i, pending), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join(timeout + 5.0)
    wall_time = time.perf_counter() - t0

    for err in errors:
        if err is not None:
            raise err

    traces: list[RequestTrace] = []
    for i, req in enumerate(requests):
        stamped = completions[i]
        if stamped is None:
            raise TimeoutError(f"request {i} did not complete within {timeout}s")
        completion, num_tokens, first_token_time = stamped
        traces.append(
            RequestTrace(
                arrival=arrival_times[i],
                # Real TTFT when the engine streams; else all-at-once (static).
                first_token=first_token_time
                if first_token_time is not None
                else completion,
                completion=completion,
                num_output_tokens=num_tokens,
                category=req.category,
            )
        )

    # Dedupe shared BatchStats (a static batch: every request points at one
    # object). A continuous-batch engine records per-step fill instead — drain it.
    seen: dict[int, BatchStats] = {}
    for pending in pendings:
        stats = getattr(pending, "stats", None)
        if stats is not None:
            seen.setdefault(id(stats), stats)
    batch_stats = list(seen.values())
    if hasattr(engine, "pop_step_stats"):
        batch_stats.extend(engine.pop_step_stats())
    return LoadResult(traces=traces, wall_time=wall_time, batch_stats=batch_stats)


def sample_gpu_util() -> float | None:
    """A coarse GPU-busy sample (nvml device utilization %), or ``None`` off CUDA.

    This is the cheap, always-available proxy. ADR-0002 §(a)'s ideal — torch-
    profiler / nsys **achieved occupancy / SM util** — is a T5 (efficiency-tier)
    refinement that needs a profiled CUDA run; ``torch.cuda.utilization()`` is
    only device busy-time %. On a Mac/CPU there is no GPU to read, so the batch-
    fill rate carries the T1 utilization story.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001 — util is best-effort evidence, never fatal
        return None


def utilization_from(batch_stats: Sequence[BatchStats]) -> Utilization:
    """Aggregate the T1 utilization evidence from the batches a run saw.

    ``batch_fill_rate`` = real decode work / padded-lockstep grid; the gap is the
    head-of-line waste T2 removes. ``fill_over_time`` is the per-batch fill in
    submission (time) order — ADR-0002's "填充率随时间曲线". ``prompt_pad_fraction``
    = left-pad tokens / prefill grid. ``kv_block_occupancy`` is the mean of the
    per-step paged-pool occupancy the engine recorded (``None`` for the
    contiguous T1/T2 caches, which have no block pool) — ADR-0002's KV-occupancy.
    """
    fill_over_time: list[float] = []
    for bs in batch_stats:
        slots = bs.batch_size * bs.decode_steps
        if slots:
            fill_over_time.append((slots - bs.decode_slack_tokens) / slots)
    total_slots = sum(bs.batch_size * bs.decode_steps for bs in batch_stats)
    total_slack = sum(bs.decode_slack_tokens for bs in batch_stats)
    total_prefill = sum(bs.batch_size * bs.max_prompt_len for bs in batch_stats)
    total_pad = sum(bs.prompt_pad_tokens for bs in batch_stats)
    fill = (total_slots - total_slack) / total_slots if total_slots else None
    pad_frac = total_pad / total_prefill if total_prefill else None
    kv_vals = [
        bs.kv_block_occupancy for bs in batch_stats if bs.kv_block_occupancy is not None
    ]
    kv_occ = sum(kv_vals) / len(kv_vals) if kv_vals else None
    return Utilization(
        batch_fill_rate=fill,
        prompt_pad_fraction=pad_frac,
        gpu_util_pct=sample_gpu_util(),
        kv_block_occupancy=kv_occ,
        fill_over_time=fill_over_time,
    )


def check_correctness(
    engine: Engine, oracle: Oracle, workload: Workload
) -> CorrectnessReport:
    """A/B every prompt (greedy) against ``oracle``, tallied per category.

    The default oracle for T1 is the T0 single-request path (``t0_oracle``),
    which is itself HF-parity-gated (``tests/test_parity.py``); matching it means
    the batched path matches HF transitively — the batch-invariance gate.
    ``passed`` needs *every* category perfect, so a category-collapse is caught.
    """
    from infrared.engine.static_batch import BatchRequest

    matched_by_cat: dict[str, int] = {}
    total_by_cat: dict[str, int] = {}
    matched = total = 0
    for category, prompt, max_new in workload.items():
        got = engine.generate(
            BatchRequest(
                prompt_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.0,
                eos_token_ids=(),
            )
        )
        want = oracle(prompt, max_new)
        ok = got == want
        matched_by_cat[category] = matched_by_cat.get(category, 0) + int(ok)
        total_by_cat[category] = total_by_cat.get(category, 0) + 1
        matched += int(ok)
        total += 1
    per_category = {
        cat: matched_by_cat[cat] / total_by_cat[cat] for cat in total_by_cat
    }
    return CorrectnessReport(per_category=per_category, matched=matched, total=total)


def t0_oracle(model: Qwen2ForCausalLM) -> Oracle:
    """Build a greedy oracle from the T0 single-request generate loop."""
    from infrared.model.generate import generate

    def _oracle(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        out = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_ids=(),
        )
        return out.generated_ids

    return _oracle


def hf_oracle(model_dir: str, dtype: str = "float32", device: str = "cpu") -> Oracle:
    """Build the literal Seam-A oracle: greedy generation from HF ``transformers``.

    This is the "A/B vs HF oracle" the issue asks for. HF is used *only* as the
    correctness reference here (ADR-0003/0005) — never as infrared's execution
    path. fp32 + eager attention on CPU is the deterministic parity setup
    (matches ``tests/test_parity.py``). ``t0_oracle`` is the no-download default;
    pass this when a run should A/B straight against HF on the same weights.
    """
    import torch
    from transformers import AutoModelForCausalLM

    hf = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=getattr(torch, dtype), attn_implementation="eager"
        )
        .to(device)
        .eval()
    )

    @torch.no_grad()
    def _oracle(prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        ids = torch.tensor([prompt_ids], device=device)
        out = hf.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0, len(prompt_ids) :].tolist()

    return _oracle


def measure_throughput(engine: Engine, category: Category) -> float:
    """Sustained output tok/s on a decode-heavy burst (fixed uniform shape)."""
    requests = [
        LoadRequest(
            prompt_ids=p, max_new_tokens=category.max_new_tokens, category=category.name
        )
        for p in category.prompts
    ]
    result = run_load(engine, requests, arrivals=None)
    total_tokens = sum(t.num_output_tokens for t in result.traces)
    return total_tokens / result.wall_time if result.wall_time > 0 else 0.0


def sweep_rates(
    engine: Engine,
    requests: Sequence[LoadRequest],
    rates: Sequence[float],
    slo: SLO,
    seed: int = 0,
    timeout: float = 120.0,
) -> list[SweepPoint]:
    """Offer ``requests`` at each rate (Poisson) and fold into ``SweepPoint``s."""
    points: list[SweepPoint] = []
    for rate in rates:
        arrivals = poisson_arrivals(rate=rate, n=len(requests), seed=seed)
        result = run_load(engine, requests, arrivals=arrivals, timeout=timeout)
        points.append(
            sweep_point(
                offered_rate=rate,
                traces=result.traces,
                slo=slo,
                wall_time=result.wall_time,
            )
        )
    return points


@dataclass(slots=True)
class MeasureResult:
    """One config's ladder row plus the knee sweep that produced it."""

    row: LadderRow
    sweep: list[SweepPoint]


def measure(
    engine: Engine,
    oracle: Oracle,
    workload: Workload,
    slo: SLO,
    rates: Sequence[float],
    *,
    tier: str = "T1 static batch",
    throughput_category: Category | None = None,
    sweep_requests: Sequence[LoadRequest] | None = None,
    seed: int = 0,
    notes: str = "",
) -> MeasureResult:
    """Produce one ``LadderRow`` (correctness/throughput/goodput/utilization).

    ``goodput`` is the best per-request SLO-meeting rate seen across the sweep;
    ``knee_rate`` is the highest offered rate whose aggregate p99s still fit the
    SLO. Defaults synthesize a decode-heavy category for throughput and reuse the
    workload's prompts for the sweep when the caller doesn't supply them.
    """
    correctness = check_correctness(engine, oracle, workload)

    if throughput_category is None:
        # Draw prompt tokens within the model's own vocab so a tiny test model
        # (small vocab) doesn't index past its embedding table.
        vocab_size = engine.model.config.vocab_size
        throughput_category = decode_heavy_category(
            n=engine.max_batch_size,
            prompt_len=8,
            max_new_tokens=32,
            vocab_size=vocab_size,
            seed=seed,
        )
    throughput = measure_throughput(engine, throughput_category)

    if sweep_requests is None:
        sweep_requests = [
            LoadRequest(prompt_ids=prompt, max_new_tokens=max_new, category=category)
            for category, prompt, max_new in workload.items()
        ]

    # Utilization evidence comes from a burst over the (mixed-length) workload,
    # not the uniform throughput shape — that's where a static batch's padding +
    # head-of-line waste actually shows up (a uniform shape trivially fills 100%).
    util_burst = run_load(engine, list(sweep_requests), arrivals=None)
    utilization = utilization_from(util_burst.batch_stats)

    sweep = sweep_rates(engine, sweep_requests, rates, slo, seed=seed)
    best = knee(sweep)
    goodput = max((p.goodput_reqs_per_s for p in sweep), default=None)

    row = LadderRow(
        tier=tier,
        correctness=correctness,
        throughput_toks_per_s=throughput,
        goodput_reqs_per_s=goodput,
        knee_rate=best.offered_rate if best is not None else None,
        utilization=utilization,
        notes=notes,
    )
    return MeasureResult(row=row, sweep=sweep)


# Tiers below don't exist yet; the ladder names them so the artifact reads as
# "one row per mechanism, filled in as it's built" rather than silently short.
# The whole T4 efficiency tier (paged KV, prefix caching, chunked prefill, Triton
# kernel, CUDA graphs) is now built — nothing pending until the T5/T6 arc lands.
_PENDING_TIERS: tuple[str, ...] = ()


def build_ladder(results: Sequence[MeasureResult], include_pending: bool = True) -> str:
    """Render the before→after ladder table (+ placeholder rows for T2–T4)."""
    rows = [r.row for r in results]
    if include_pending:
        rows = list(rows) + [
            LadderRow(tier=t, notes="not built yet") for t in _PENDING_TIERS
        ]
    return render_ladder_markdown(rows)


def render_report(result: MeasureResult) -> str:
    """A full text block: the ladder row, then its knee sweep."""
    return (
        "## Before→after ladder\n\n"
        + build_ladder([result])
        + "\n\n## Knee sweep (request-rate up-scan)\n\n"
        + render_sweep_markdown(result.sweep)
        + "\n"
    )
