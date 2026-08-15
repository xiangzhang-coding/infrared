"""Workloads and the open-loop arrival process (pure Python, no torch).

A **workload** is a set of categorised prompts (token-id lists, so no tokenizer
is needed to exercise the harness) plus how many tokens each should generate.
Categories let ``correctness`` report per-category — a mechanism that keeps
overall accuracy but collapses one category is caught, per the Seam-A quality
gate.

**Arrivals** model open-loop load: requests are offered at a target rate
regardless of whether the engine is keeping up (that back-pressure is exactly
what pushes latency past the SLO and reveals the knee). Inter-arrival times are
exponential (a Poisson process), seeded so a sweep is reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class Category:
    """A named bucket of prompts sharing a generation budget."""

    name: str
    prompts: list[list[int]]  # each prompt is a list of token ids
    max_new_tokens: int = 64


@dataclass(slots=True)
class Workload:
    """A set of categories driven as one benchmark run."""

    categories: list[Category] = field(default_factory=list)

    def items(self) -> Iterator[tuple[str, list[int], int]]:
        """Yield ``(category_name, prompt_ids, max_new_tokens)`` for every prompt."""
        for cat in self.categories:
            for prompt in cat.prompts:
                yield cat.name, prompt, cat.max_new_tokens

    @property
    def num_requests(self) -> int:
        return sum(len(cat.prompts) for cat in self.categories)


def decode_heavy_category(
    n: int,
    prompt_len: int = 16,
    max_new_tokens: int = 128,
    vocab_size: int = 1000,
    seed: int = 0,
) -> Category:
    """A short-prompt / long-generation category — the throughput shape.

    Decode-heavy (few prompt tokens, many generated) is what stresses the decode
    loop rather than prefill, so ``throughput`` measures the steady-state
    output tok/s the engine sustains. Token ids are drawn deterministically in
    ``[1, vocab_size)`` (0 is the static-batch pad id) so runs reproduce.
    """
    if n <= 0 or prompt_len <= 0:
        raise ValueError("n and prompt_len must be positive")
    rng = random.Random(seed)
    prompts = [
        [rng.randrange(1, vocab_size) for _ in range(prompt_len)] for _ in range(n)
    ]
    return Category(name="decode-heavy", prompts=prompts, max_new_tokens=max_new_tokens)


def shared_prefix_category(
    n: int,
    prefix_len: int = 32,
    tail_len: int = 4,
    max_new_tokens: int = 16,
    vocab_size: int = 1000,
    seed: int = 0,
) -> Category:
    """``n`` prompts that all begin with the *same* prefix, then a distinct tail.

    This is the shape prefix caching (T4) is built for — a shared system prompt /
    few-shot preamble across many requests. ``prefix_len`` should span at least
    one full KV block for reuse to trigger (the cache addresses whole blocks); the
    per-prompt ``tail_len`` differs so the requests aren't identical. Token ids are
    drawn deterministically in ``[1, vocab_size)`` (0 is the static-batch pad id)
    so runs reproduce; the shared prefix is drawn once, the tails per prompt.
    """
    if n <= 0 or prefix_len <= 0 or tail_len < 0:
        raise ValueError("n and prefix_len must be positive; tail_len non-negative")
    rng = random.Random(seed)
    prefix = [rng.randrange(1, vocab_size) for _ in range(prefix_len)]
    prompts = [
        prefix + [rng.randrange(1, vocab_size) for _ in range(tail_len)]
        for _ in range(n)
    ]
    return Category(
        name="shared-prefix", prompts=prompts, max_new_tokens=max_new_tokens
    )


def long_prefill_category(
    n: int = 2,
    prompt_len: int = 48,
    max_new_tokens: int = 16,
    vocab_size: int = 1000,
    seed: int = 0,
) -> Category:
    """Long prompts (many prefill blocks) — the shape chunked prefill (T4b) targets.

    A long prompt's prefill would, un-chunked, occupy the engine for one big step
    and stall concurrent decodes; chunked prefill spreads it across steps. Sized so
    ``prompt_len`` spans several KV blocks / chunks. Token ids are drawn
    deterministically in ``[1, vocab_size)`` (0 is the static-batch pad id).
    """
    if n <= 0 or prompt_len <= 0:
        raise ValueError("n and prompt_len must be positive")
    rng = random.Random(seed)
    prompts = [
        [rng.randrange(1, vocab_size) for _ in range(prompt_len)] for _ in range(n)
    ]
    return Category(name="long-prefill", prompts=prompts, max_new_tokens=max_new_tokens)


def poisson_arrivals(rate: float, n: int, seed: int = 0) -> list[float]:
    """Cumulative arrival offsets (seconds) for a rate-``rate`` Poisson process.

    Returns ``n`` non-decreasing offsets relative to ``t=0``; the driver sleeps
    until each one before submitting. Exponential inter-arrivals give a mean gap
    of ``1/rate``. Seeded for a reproducible sweep.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    arrivals: list[float] = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        arrivals.append(t)
    return arrivals
