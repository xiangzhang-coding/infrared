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


class Scheduler:
    """Iteration-level admission/eviction over ``waiting`` / ``running`` queues."""

    def __init__(
        self, max_num_seqs: int = 8, max_num_batched_tokens: int | None = None
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        self.max_num_seqs = max_num_seqs
        # Only meaningful once T4 adds chunked prefill; stored, not yet enforced.
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
