"""Static-batch engine: a request queue + a batching worker thread (T1).

Concurrent callers ``submit`` requests onto a thread-safe queue. A single worker
thread gathers whatever has arrived (up to ``max_batch_size``, after a short
``linger`` window so near-simultaneous requests batch together), runs them as one
static batch, and hands each caller its result. Because a static batch returns
all-or-nothing, a new batch can't start until the current one fully finishes —
the head-of-line blocking T1 is meant to expose.

This is the same-process, single-worker shape R1 calls the engine↔worker seam;
T2 replaces the "one static batch at a time" worker with a continuous-batching
scheduler behind the same submit/queue surface.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from infrared.engine.static_batch import (
    BatchRequest,
    BatchStats,
    run_static_batch,
)
from infrared.model.qwen2 import Qwen2ForCausalLM

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
