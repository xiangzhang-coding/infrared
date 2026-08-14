"""Static-batch engine: a request queue + a batching worker thread (T1).

Concurrent callers ``submit`` requests onto a thread-safe queue. A single worker
thread gathers whatever has arrived (up to ``max_batch_size``, after a short
``linger`` window so near-simultaneous requests batch together), runs them as one
static batch, and hands each caller its result. Because a static batch returns
all-or-nothing, a new batch can't start until the current one fully finishes —
the head-of-line blocking T1 is meant to expose.

This is the same-process, single-worker shape R1 calls the engine↔worker seam.
``ContinuousBatchEngine`` (below) is the T2 replacement: it swaps the "one static
batch at a time" worker for an iteration-level continuous-batching scheduler
behind the **same** ``submit`` / ``Pending`` surface, so the HTTP shell and the
bench harness drive either engine unchanged.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from infrared.engine.scheduler import Scheduler
from infrared.engine.sequence import Sequence
from infrared.engine.static_batch import (
    BatchRequest,
    BatchStats,
    run_static_batch,
)
from infrared.model.qwen2 import Qwen2ForCausalLM
from infrared.model.sampler import Sampler

_SHUTDOWN = object()  # sentinel to wake the worker for shutdown


@dataclass(slots=True)
class Pending:
    """A submitted request and its eventual result (set by the worker)."""

    request: BatchRequest
    event: threading.Event = field(default_factory=threading.Event)
    output: list[int] | None = None
    stats: BatchStats | None = None
    error: BaseException | None = None

    def result(self, timeout: float | None = None) -> list[int]:
        """Block until the worker completes this request; return its tokens."""
        if not self.event.wait(timeout):
            raise TimeoutError("request timed out")
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return self.output


class StaticBatchEngine:
    """Queue requests; a worker thread processes them one static batch at a time."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        max_batch_size: int = 8,
        linger: float = 0.01,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.linger = linger
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopping = False

    def start(self) -> StaticBatchEngine:
        """Start the background worker (idempotent)."""
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        return self

    def stop(self) -> None:
        """Signal the worker to exit, then fail any still-queued requests."""
        self._stopping = True
        self._queue.put(_SHUTDOWN)
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        self._fail_queued("engine stopped before this request ran")

    def _fail_queued(self, reason: str) -> None:
        """Drain leftover pendings and set an error so no waiter hangs forever."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                continue
            item.error = RuntimeError(reason)
            item.event.set()

    def submit(self, request: BatchRequest) -> Pending:
        """Enqueue a request; returns a handle to await its result."""
        if self._stopping:
            raise RuntimeError("engine is stopping; not accepting requests")
        pending = Pending(request=request)
        self._queue.put(pending)
        return pending

    def generate(
        self, request: BatchRequest, timeout: float | None = None
    ) -> list[int]:
        """Blocking convenience: submit one request and wait for its tokens."""
        return self.submit(request).result(timeout)

    def _collect(self) -> list[Pending]:
        """Block for one request, then drain more (up to the batch cap)."""
        first = self._queue.get()
        if first is _SHUTDOWN:
            return []
        batch = [first]
        if self.linger:
            time.sleep(self.linger)  # let concurrent arrivals accumulate
        while len(batch) < self.max_batch_size:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                self._stopping = True
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        while not self._stopping:
            batch = self._collect()
            if not batch:
                continue
            try:
                result = run_static_batch(self.model, [p.request for p in batch])
                for pending, output in zip(batch, result.outputs, strict=True):
                    pending.output = output
                    pending.stats = result.stats
                    pending.event.set()
            except BaseException as exc:  # deliver the failure to every waiter
                for pending in batch:
                    pending.error = exc
                    pending.event.set()


@dataclass(slots=True)
class ContinuousPending:
    """A submitted request and its streaming result (set by the busy loop).

    Unlike the static ``Pending`` (all-or-nothing), this handle stamps
    ``first_token_time`` the moment the sequence's first token is produced, so a
    caller — and the bench harness — can observe a real TTFT. ``result`` still
    blocks until the whole generation finishes; ``stats`` stays ``None`` because
    continuous batching has no per-request "batch" (per-step fill is recorded on
    the engine and drained via ``pop_step_stats``).
    """

    request: BatchRequest
    first_token: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    first_token_time: float | None = None
    output: list[int] | None = None
    error: BaseException | None = None
    stats: None = None  # kept for surface-compat with static Pending

    def result(self, timeout: float | None = None) -> list[int]:
        """Block until the busy loop finishes this request; return its tokens."""
        if not self.done.wait(timeout):
            raise TimeoutError("request timed out")
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return self.output

    def wait_first_token(self, timeout: float | None = None) -> bool:
        """Block until the first token is observable; True if it arrived."""
        return self.first_token.wait(timeout)

    def complete(self, output: list[int]) -> None:
        """Deliver a successful result and release any waiter (idempotent-safe).

        Sets ``first_token`` too so a first-token waiter is never left hanging on
        a zero-token generation (it may already be set from streaming).
        """
        self.output = output
        self.first_token.set()
        self.done.set()

    def fail(self, exc: BaseException) -> None:
        """Deliver an error and release every waiter (first-token and result)."""
        self.error = exc
        self.first_token.set()
        self.done.set()


class ContinuousBatchEngine:
    """T2 engine: an iteration-level scheduler behind the static submit surface.

    A single worker thread runs the busy loop: drain new submissions into the
    scheduler's ``waiting`` queue, then ``step()`` (schedule → forward → sample →
    append KV → retire) until everything drains, blocking on the queue when idle.
    Each ``step`` is one model forward — for a prefill step, the one just-admitted
    sequence; for a decode step, every running sequence advanced by one token.

    The forward path reuses the T0 single-request ``forward_single`` per sequence
    against that sequence's own contiguous KV cache, so a continuous-batch
    generation is **token-for-token identical** to the T0 oracle (the batch is a
    *scheduling* construct here, not a shared tensor frame). The flattened,
    right-pad-free varlen batched forward — the raw-throughput lever — needs
    paged KV to gather ragged histories and lands at T3/T4 (R1 §5, §8); T2's win
    is utilization (no padding, no head-of-line waste), goodput, and TTFT.
    """

    def __init__(self, model: Qwen2ForCausalLM, max_num_seqs: int = 8) -> None:
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.sampler = Sampler()
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs)
        self._pending: dict[int, ContinuousPending] = {}
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopping = False
        self._stats_lock = threading.Lock()
        self._step_stats: list[BatchStats] = []

    # ``StaticBatchEngine`` exposes ``max_batch_size``; mirror it so the bench
    # harness (``measure``) sizes its throughput category the same way.
    @property
    def max_batch_size(self) -> int:
        return self.max_num_seqs

    def start(self) -> ContinuousBatchEngine:
        """Start the background busy loop (idempotent)."""
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        return self

    def stop(self) -> None:
        """Signal the worker to exit, then fail any not-yet-finished requests."""
        self._stopping = True
        self._queue.put(_SHUTDOWN)
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        self._fail_pending("engine stopped before this request finished")

    def submit(self, request: BatchRequest) -> ContinuousPending:
        """Enqueue a request; returns a handle to await first token / result."""
        if self._stopping:
            raise RuntimeError("engine is stopping; not accepting requests")
        pending = ContinuousPending(request=request)
        self._queue.put(pending)
        return pending

    def generate(
        self, request: BatchRequest, timeout: float | None = None
    ) -> list[int]:
        """Blocking convenience: submit one request and wait for its tokens."""
        return self.submit(request).result(timeout)

    def pop_step_stats(self) -> list[BatchStats]:
        """Return per-step fill stats recorded since the last call, and clear.

        The bench harness pulls these after a load run to build the batch-fill-
        rate curve (one point per decode step). A prefill step records zero
        padding — the T1 pathology gone — so ``prompt_pad_fraction`` collapses.
        """
        with self._stats_lock:
            stats, self._step_stats = self._step_stats, []
            return stats

    def _fail_pending(self, reason: str) -> None:
        """Fail every still-queued and in-flight request so no waiter hangs."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                continue
            item.fail(RuntimeError(reason))
        for pending in list(self._pending.values()):
            if not pending.done.is_set():
                pending.fail(RuntimeError(reason))
        self._pending.clear()

    # --- busy loop ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stopping:
            if self.scheduler.is_finished():
                item = self._queue.get()  # block until work arrives
                if item is _SHUTDOWN:
                    break
                self._admit(item)
            self._drain_queue()  # pick up any concurrent arrivals
            if self.scheduler.is_finished():
                continue
            try:
                self._step()
            except BaseException as exc:  # noqa: BLE001
                # Per-sequence errors are already isolated in ``_step`` (fail +
                # retire just that seq). Reaching here means something outside a
                # single sequence broke unrecoverably — stop the loop and fail
                # everyone rather than spin re-running the same failing step.
                self._stopping = True
                self._fail_pending(f"engine step failed: {exc!r}")

    def _drain_queue(self) -> None:
        """Non-blocking: admit every arrival already waiting in the queue."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:
                self._stopping = True
                return
            self._admit(item)

    def _admit(self, pending: ContinuousPending) -> None:
        """Turn a submitted request into a WAITING ``Sequence`` (or complete it).

        A construction error (empty prompt, malformed request) fails *only* this
        request — never the worker. A zero-length generation budget completes
        immediately with no output, matching the T0/T1 oracles (which prefill but
        generate nothing for ``max_new_tokens <= 0``).
        """
        req = pending.request
        try:
            seq = Sequence(
                prompt_ids=list(req.prompt_ids),
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                seed=req.seed,
                eos_token_ids=tuple(req.eos_token_ids),
            )
        except Exception as exc:  # noqa: BLE001 — fail just this request
            pending.fail(exc)
            return
        if seq.max_new_tokens <= 0:
            pending.complete([])
            return
        self._pending[seq.seq_id] = pending
        self.scheduler.add(seq)

    # --- one step: schedule -> forward+sample -> postprocess ---------------

    @torch.no_grad()
    def _step(self) -> None:
        batch = self.scheduler.schedule()
        if not batch.seqs:
            return
        if batch.is_prefill:
            seq = batch.seqs[0]
            self._advance(seq)
            self._record(
                BatchStats(
                    batch_size=1,
                    max_prompt_len=seq.num_prompt_tokens,
                    prompt_pad_tokens=0,  # each prompt prefills at its own length
                    decode_steps=0,  # not a decode step -> excluded from fill grid
                    decode_slack_tokens=0,
                )
            )
            return

        # Decode step: advance every running sequence by one token. The "grid"
        # this step is exactly the running set, and every slot does real work —
        # continuous batching forwards no finished sequences (no head-of-line
        # slack) and pads nothing. So work-efficiency is 100% by construction;
        # the T1 baseline's shortfall (< 100%) is precisely the pad + HOL waste
        # this mechanism removes, which is what makes the fill delta attributable.
        num_running = len(batch.seqs)
        for seq in batch.seqs:
            self._advance(seq)
        self._record(
            BatchStats(
                batch_size=num_running,  # the real grid — matches T1's fill formula
                max_prompt_len=0,  # decode step contributes nothing to prefill grid
                prompt_pad_tokens=0,
                decode_steps=1,
                decode_slack_tokens=0,  # no finished-seq slots forwarded (no HOL)
            )
        )

    def _advance(self, seq: Sequence) -> None:
        """Forward + sample + postprocess one sequence, isolating its failures.

        A forward that raises (e.g. an out-of-vocab token id) fails **only** this
        request and retires the sequence, so one bad request can neither poison
        its batch-mates nor wedge the busy loop by lingering in ``running``.
        """
        try:
            token = self._forward_seq(seq)
            self._postprocess(seq, token)
        except Exception as exc:  # noqa: BLE001 — fail + retire just this seq
            pending = self._pending.pop(seq.seq_id, None)
            if pending is not None and not pending.done.is_set():
                pending.fail(exc)
            self.scheduler.retire(seq)

    def _forward_seq(self, seq: Sequence) -> int:
        """Run one forward for ``seq`` (prefill whole prompt, or one decode token).

        Reuses the T0 ``forward_single`` against the sequence's own KV cache, so
        the math is identical to running this prompt alone — the correctness seam.
        """
        device = self.model.device
        if seq.kv is None:  # lazily size the cache: prompt + full generation budget
            seq.kv = self.model.new_kv_cache(
                max_len=seq.num_prompt_tokens + seq.max_new_tokens
            )
            if seq.seed is not None:
                seq.generator = torch.Generator(device=device).manual_seed(seq.seed)

        if seq.needs_prefill:
            ids = torch.tensor(seq.prompt_ids, dtype=torch.long, device=device)
            logits = self.model.forward_single(ids, seq.kv, start_pos=0)
            seq.num_cached_tokens = seq.num_prompt_tokens
        else:
            ids = torch.tensor([seq.last_token], dtype=torch.long, device=device)
            logits = self.model.forward_single(
                ids, seq.kv, start_pos=seq.num_cached_tokens
            )
            seq.num_cached_tokens += 1

        return self.sampler.sample(logits[-1], seq.temperature, seq.generator)

    def _postprocess(self, seq: Sequence, token: int) -> None:
        """Append the sampled token, stream first-token, retire if stopped."""
        pending = self._pending.get(seq.seq_id)
        seq.append(token)
        if pending is not None and pending.first_token_time is None:
            pending.first_token_time = time.perf_counter()
            pending.first_token.set()
        if seq.should_stop(token):
            if pending is not None:
                pending.complete(seq.generated)
                self._pending.pop(seq.seq_id, None)
            self.scheduler.retire(seq)

    def _record(self, stats: BatchStats) -> None:
        with self._stats_lock:
            self._step_stats.append(stats)
