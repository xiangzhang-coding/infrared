"""Per-request state machine (T2 — the continuous-batching state carrier).

A ``Sequence`` is the object the ``Scheduler`` moves between its ``waiting`` and
``running`` queues and the ``ContinuousBatchEngine`` advances one token per step.
It carries the request's token ids (prompt + generated), how many tokens are
already committed to its KV cache (``num_cached_tokens`` — the shared time axis),
its sampling params, and handles to its own KV cache + RNG (filled by the engine,
which owns the torch side).

Fields track the R1 §4.1 minimal set (``token_ids``, ``num_prompt_tokens``,
``num_cached_tokens``, sampling params, status). ``block_table`` (logical→physical
paging) is deliberately **absent** here: T2 gives each sequence its own
contiguous ``KVCache`` (``kv``), and the block-table indirection arrives with the
T3 paged block manager. Kept torch-free (torch types are ``TYPE_CHECKING`` only)
so the CPU-side scheduler/engine-core stays a pure-decision layer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch stays out of the engine-core import path
    import torch

    from infrared.cache.kv_cache import KVCache


class SequenceStatus(Enum):
    """Lifecycle states. A preempted sequence returns to WAITING (recompute).

    T2 uses the three-state minimum (WAITING → RUNNING → FINISHED). The
    PREEMPTED→WAITING recompute edge exists in the enum's spirit but has no
    trigger until T3: T2 preallocates one contiguous KV slab per sequence, so
    there is no shared block pool to run out of and force an eviction. Admission
    is bounded purely by the ``max_num_seqs`` concurrency cap.
    """

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_seq_counter = itertools.count()


@dataclass(slots=True)
class Sequence:
    """A single request's state: token ids, cached-token count, sampling params.

    Construction takes only the request-level inputs; the engine attaches ``kv``
    (a per-sequence :class:`~infrared.cache.kv_cache.KVCache`) and ``generator``
    lazily at prefill time. ``num_cached_tokens`` is the number of KV columns
    already written — ``0`` means "not prefilled yet", after prefill it equals
    ``num_prompt_tokens``, and each decode step bumps it by one.
    """

    prompt_ids: list[int]
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int | None = None
    eos_token_ids: tuple[int, ...] = ()

    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    status: SequenceStatus = SequenceStatus.WAITING
    num_cached_tokens: int = 0

    # Filled at construction from ``prompt_ids``; grows as tokens are generated.
    token_ids: list[int] = field(init=False)
    num_prompt_tokens: int = field(init=False)
    generated: list[int] = field(init=False, default_factory=list)

    # Logical→physical KV mapping (T3 paged engine). Empty for the T2 contiguous
    # cache; the paged ``BlockManager`` fills/clears it on allocate/preempt/free.
    block_table: list[int] = field(init=False, default_factory=list)

    # Torch-side handles, attached by the engine (kept out of the decision layer).
    kv: KVCache | None = field(default=None, repr=False)
    generator: torch.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("Sequence requires a non-empty prompt")
        self.token_ids = list(self.prompt_ids)
        self.num_prompt_tokens = len(self.prompt_ids)

    @property
    def needs_prefill(self) -> bool:
        """True until the prompt has been run through one prefill forward."""
        return self.num_cached_tokens == 0

    @property
    def last_token(self) -> int:
        """The most recent token — the single id a decode step forwards."""
        return self.token_ids[-1]

    @property
    def num_completion(self) -> int:
        """How many tokens have been generated so far."""
        return len(self.generated)

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append(self, token: int) -> None:
        """Record a freshly sampled token (both the full stream and the output)."""
        self.token_ids.append(token)
        self.generated.append(token)

    def should_stop(self, token: int) -> bool:
        """T0/T1 stop semantics: EOS hit, or the generation budget is reached.

        Mirrors ``infrared.model.generate.generate`` exactly — the sampled token
        is *always* appended first, so an EOS token is included in the output and
        ``max_new_tokens`` produces exactly that many tokens. Matching this keeps
        the continuous-batch output token-for-token identical to the T0 oracle.
        """
        return token in self.eos_token_ids or self.num_completion >= self.max_new_tokens
