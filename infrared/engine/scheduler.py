"""Continuous-batching scheduler — the CPU-side heart (T2).

Holds ``waiting`` / ``running`` deques and makes the **iteration-level**
admission/eviction decision that *is* continuous batching: every step it either
admits one waiting request (a prefill step) or advances all running sequences by
one token (a decode step). A finished sequence is retired the instant it stops,
freeing its slot for the next waiting request on the very next step — no waiting
for a whole batch to drain (the head-of-line blocking T1 exposes).

Pure decision-making — **no tensor math** (that is the engine/worker's job in
``infrared.model``). This module imports no torch.

**Scope (R1 §3.2, §8):**

- *One phase per step* (nano-vLLM's minimal form): a step is either all-prefill
  (one admission) or all-decode. vLLM v1's mixed prefill+decode step and
  *chunked* prefill are deferred to T4.
- *Admission* is bounded by ``max_num_seqs`` (the running-set / batch cap). With
  each sequence owning a preallocated contiguous KV slab (T2), there is no shared
  block pool to exhaust, so **recompute preemption** has no trigger yet — it
  arrives with the T3 paged ``BlockManager`` (R1 §8). ``max_num_batched_tokens``
  is accepted for forward-compat but only bites once chunked prefill exists (T4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from infrared.engine.sequence import Sequence, SequenceStatus


@dataclass(slots=True)
class ScheduledBatch:
    """One step's decision: which sequences to run, and in which phase.

    ``is_prefill`` True means ``seqs`` is a single freshly admitted sequence to
    prefill; False means ``seqs`` is the running set, each advancing one decode
    token. Empty ``seqs`` means the scheduler had nothing to do this step.
    """

    seqs: list[Sequence]
    is_prefill: bool


@dataclass(slots=True)
class SeqChunk:
    """One sequence's slice of work in a mixed prefill+decode step (T4 chunked prefill).

    ``num_query_tokens`` is how many tokens this step forwards for ``seq``: 1 for a
    decode, or a prefill chunk (``<= chunk_size``). ``completes_prefill`` is True
    only when this chunk lands the prompt's *last* prefill token — the point the
    sequence transitions to decode and its first output token is sampled. A prefill
    chunk that doesn't complete the prompt emits **no** token (still prefilling).
    """

    seq: Sequence
    num_query_tokens: int
    is_prefill: bool
    completes_prefill: bool


@dataclass(slots=True)
class MixedPlan:
    """One step's mixed schedule: which sequences run, decode-first then prefill.

    ``chunks`` lists decodes first (each 1 token) then prefill chunks, in the order
    the flattened forward should concatenate them. ``decode_tokens`` /
    ``prefill_tokens`` are the token counts the step spends (their sum is the step's
    real token load, bounded by the budget except that decodes are never dropped).
    """

    chunks: list[SeqChunk]
    decode_tokens: int
    prefill_tokens: int

    @property
    def is_mixed(self) -> bool:
        """True when this step carries both a decode and a prefill chunk."""
        return self.decode_tokens > 0 and self.prefill_tokens > 0


def plan_mixed_step(
    running: list[Sequence], *, token_budget: int, chunk_size: int
) -> MixedPlan:
    """Decide one mixed prefill+decode step over the running set (pure, no torch).

    **Decode-first** budgeting — the mechanism's whole point is that a long prefill
    must not stall in-flight requests, so every currently-decoding sequence gets its
    one token unconditionally (a decode can't be split, and dropping it would freeze
    that request). Prefill chunks then fill whatever token budget remains, FIFO over
    the sequences still prefilling, each taking ``min(chunk_size, remaining, budget)``
    tokens. With ``token_budget`` large this reduces to "decode all + one full prefill",
    i.e. today's behaviour; with a tight budget a long prompt is spread across steps
    so decode latency stays flat (R1's deferred mixed step; vLLM's chunked prefill).

    The semantics (a per-step token budget, prefill ``num_new_tokens`` clamped to
    ``<= chunk_size`` and to the remaining budget, decode + prefill sharing one step)
    were verified against vLLM's v1 scheduler / chunked-prefill docs before writing,
    per the ADR-0006 API-verification rule.

    The planner owns only the chunk/budget decision; block-allocation and admission
    (waiting → running) stay engine-side, where the paged block state lives.
    """
    chunks: list[SeqChunk] = []
    decode_tokens = 0
    for seq in running:
        if not seq.is_prefilling:
            chunks.append(
                SeqChunk(
                    seq, num_query_tokens=1, is_prefill=False, completes_prefill=False
                )
            )
            decode_tokens += 1

    prefill_budget = max(0, token_budget - decode_tokens)
    prefill_tokens = 0
    for seq in running:
        if prefill_budget <= 0:
            break
        if not seq.is_prefilling:
            continue
        take = min(chunk_size, seq.num_prefill_remaining, prefill_budget)
        if take <= 0:
            continue
        completes = seq.num_cached_tokens + take == seq.prefill_len
        chunks.append(
            SeqChunk(
                seq,
                num_query_tokens=take,
                is_prefill=True,
                completes_prefill=completes,
            )
        )
        prefill_budget -= take
        prefill_tokens += take

    return MixedPlan(
        chunks=chunks, decode_tokens=decode_tokens, prefill_tokens=prefill_tokens
    )


class Scheduler:
    """Iteration-level admission/eviction over ``waiting`` / ``running`` queues."""

    def __init__(
        self, max_num_seqs: int = 8, max_num_batched_tokens: int | None = None
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        self.max_num_seqs = max_num_seqs
        # Stored, but NOT consumed by this class — the paged engine bypasses
        # ``schedule()`` and enforces its own per-step token budget
        # (``PagedBatchEngine.token_budget``) via ``plan_mixed_step``. Kept here for
        # the T2 surface + forward-compat.
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """Enqueue a new request (FCFS tail of ``waiting``)."""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        """True when nothing is queued and nothing is decoding."""
        return not self.waiting and not self.running

    def schedule(self) -> ScheduledBatch:
        """Pick this step's batch: admit-if-room (prefill), else decode-all.

        Prefill is prioritised only while there is free capacity, so admissions
        front-load until ``running`` hits ``max_num_seqs`` or ``waiting`` drains;
        thereafter every step is a decode step until a sequence finishes and
        frees a slot — at which point the next step admits the next waiting
        request. That interleaving is the continuous-batching behaviour.
        """
        if self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            return ScheduledBatch(seqs=[seq], is_prefill=True)

        if self.running:
            # Snapshot the running set — postprocess may retire members mid-step.
            return ScheduledBatch(seqs=list(self.running), is_prefill=False)

        return ScheduledBatch(seqs=[], is_prefill=False)

    def retire(self, seq: Sequence) -> None:
        """Mark a finished sequence FINISHED and drop it from ``running``.

        (T3 will additionally return its KV blocks to the pool here; T2's
        per-sequence contiguous cache is simply dropped with the ``Sequence``.)
        """
        seq.status = SequenceStatus.FINISHED
        try:
            self.running.remove(seq)
        except ValueError:
            pass  # already retired (defensive; retire is called once per stop)
