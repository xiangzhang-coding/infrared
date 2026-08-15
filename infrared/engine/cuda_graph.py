"""T4d — CUDA-graph decode capture/replay (GPU-only, R3 #10 §3).

Decode is launch-bound: each step fires dozens of tiny kernels (RoPE, the paged
scatter/gather + attention, RMSNorms, the MLP GEMMs, the lm_head) whose per-launch
overhead dwarfs their compute at batch-of-1-token width. A **CUDA graph** records
that whole kernel sequence once and replays it as a single submission, erasing the
per-step launch overhead — the decode-phase TPOT/throughput lever (R1 §5/§8).

**What's captured, and how variable-length decode is made graph-safe** (R3 §3.4):

- **Decode only.** Prefill stays eager (its seq-len varies per request, hard to
  graph); only the fixed-per-step decode forward is captured.
- **One graph per batch-size bucket.** The running-set size ``b`` varies, but a
  graph needs a fixed shape — so we capture graphs for a set of buckets
  (``1,2,4,…,max_num_seqs``) and, for a real ``b``, replay the smallest bucket
  ``B ≥ b``.
- **Fixed key axis ``t_max``.** History grows every step, but the captured
  ``gather_slots``/``mask`` shape can't. We size the key axis to a fixed ``t_max``
  and let each row's ``context_len`` (encoded in the additive mask) select how much
  is valid — so growing history needs **no re-capture**, only crossing a batch-size
  bucket does (R3 §3.4: "variable context length is fine without re-capture").
- **Pad-by-repeat.** Rows ``[b:B]`` replicate row 0 (a valid request) rather than a
  ``-1`` sentinel: their scatter writes row 0's own K to row 0's own slot
  (idempotent), their gather/logits are row 0's, and the caller slices ``[:b]`` so
  they're discarded. No pool corruption, no negative-index scatter.

**Cross-platform 铁律 (the ticket's key rule).** CUDA graphs need a CUDA device, so
``CudaGraphDecoder`` is **only ever constructed on CUDA** (the engine gates on
``torch.cuda.is_available()``); on Mac/CPU decode stays eager and this class is
never touched. All ``torch.cuda`` graph calls are made inside methods, never at
import. The pure input-building + bucketing helpers below are device-agnostic and
CPU-tested.

**Honesty (ADR-0006).** This box has no GPU: the capture/replay path was **not run
here**. Its parity-vs-eager + speedup are CUDA-gated (`tests/test_cuda_graph.py::
...on_cuda`) and validated on AutoDL. What *is* verified on CPU is that the
graph-shaped padded forward, sliced ``[:b]``, equals the eager decode — i.e. the
captured *workload* is numerically identical; only the CUDAGraph wrapper is GPU-only.
Every ``torch.cuda`` API used (``CUDAGraph``, ``graph(g, pool=…)``, ``g.replay()``,
``g.pool()``, side-stream warmup, ``static.copy_()``) is verified in R3 §3.1/3.3/3.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from infrared.cache.paged_kv_cache import PagedKVPool
    from infrared.model.qwen2 import Qwen2ForCausalLM


# --- pure bucketing ---------------------------------------------------------


def default_buckets(max_num_seqs: int) -> tuple[int, ...]:
    """Powers of two up to ``max_num_seqs``, plus ``max_num_seqs`` itself.

    e.g. ``max_num_seqs=8`` -> ``(1, 2, 4, 8)``; ``6`` -> ``(1, 2, 4, 6)``. A capture
    per bucket bounds how many graphs (and how much static memory) we hold while
    keeping the pad-up waste (``B - b`` dummy rows) under 2x.
    """
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    buckets: list[int] = []
    p = 1
    while p < max_num_seqs:
        buckets.append(p)
        p *= 2
    buckets.append(max_num_seqs)
    return tuple(buckets)


def pick_bucket(b: int, buckets: tuple[int, ...]) -> int:
    """Smallest bucket ``>= b``. Raises if ``b`` exceeds the largest bucket."""
    for bucket in buckets:
        if bucket >= b:
            return bucket
    raise ValueError(f"batch size {b} exceeds largest bucket {buckets[-1]}")


# --- pure fixed-shape decode input construction -----------------------------


@dataclass(slots=True)
class DecodeInputs:
    """Fixed-shape decode inputs for a bucket ``B`` and key axis ``t_max``.

    ``ids`` ``[B, 1]``, ``positions`` ``[B, 1]``, ``write_slots`` ``[B]``,
    ``gather_slots`` ``[B, t_max]``, ``mask`` ``[B, 1, 1, t_max]`` additive. These
    are the exact tensors ``Qwen2ForCausalLM.forward(..., paged=...)`` consumes for a
    batched decode step — just sized to fixed ``(B, t_max)`` so a CUDA graph can
    capture them once and replay as history grows.
    """

    ids: torch.Tensor
    positions: torch.Tensor
    write_slots: torch.Tensor
    gather_slots: torch.Tensor
    mask: torch.Tensor


def build_decode_static_inputs(
    block_tables: list[list[int]],
    num_cached: list[int],
    last_tokens: list[int],
    bucket_b: int,
    t_max: int,
    block_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> DecodeInputs:
    """Build one decode step's fixed-shape inputs for ``bucket_b`` rows over ``t_max``.

    Mirrors the eager ``PagedBatchEngine._decode_step`` per-sequence build (new token
    at absolute position ``num_cached``, written to ``slot(block_table, num_cached)``,
    attending its history ``[0, num_cached]`` — i.e. ``context_len = num_cached + 1``),
    but at fixed shape: the key axis is padded to ``t_max`` (extra columns masked) and
    the batch is padded to ``bucket_b`` by **repeating row 0** (see module docstring).
    Slicing the forward's output ``[:len(last_tokens)]`` recovers exactly the real rows.
    """
    b = len(last_tokens)
    if not 0 < b <= bucket_b:
        raise ValueError(f"need 0 < b ({b}) <= bucket_b ({bucket_b})")

    ids = torch.zeros(bucket_b, 1, dtype=torch.long, device=device)
    positions = torch.zeros(bucket_b, 1, dtype=torch.long, device=device)
    write_slots = torch.zeros(bucket_b, dtype=torch.long, device=device)
    gather_slots = torch.zeros(bucket_b, t_max, dtype=torch.long, device=device)
    context_lens: list[int] = []

    def slot(bt: list[int], pos: int) -> int:
        return bt[pos // block_size] * block_size + (pos % block_size)

    for i in range(bucket_b):
        j = i if i < b else 0  # padded rows replicate row 0
        bt, nc = block_tables[j], num_cached[j]
        ctx = nc + 1  # history including the new token
        if ctx > t_max:
            raise ValueError(f"context_len {ctx} exceeds t_max {t_max}")
        ids[i, 0] = last_tokens[j]
        positions[i, 0] = nc
        write_slots[i] = slot(bt, nc)
        for p in range(ctx):
            gather_slots[i, p] = slot(bt, p)
        context_lens.append(ctx)

    cols = torch.arange(t_max, device=device)
    lens = torch.tensor(context_lens, device=device)[:, None]  # [B, 1]
    invalid = cols[None, :] >= lens  # [B, t_max]
    mask = torch.zeros(bucket_b, 1, 1, t_max, dtype=dtype, device=device)
    mask = mask.masked_fill(invalid[:, None, None, :], torch.finfo(dtype).min)
    return DecodeInputs(ids, positions, write_slots, gather_slots, mask)


# --- the capture/replay decoder (GPU-only) ----------------------------------


@dataclass(slots=True)
class _Captured:
    """A captured decode graph for one bucket + the static buffers it reads/writes."""

    graph: object  # torch.cuda.CUDAGraph
    buf: DecodeInputs
    logits: torch.Tensor  # [B, vocab] — written in place by replay()


class CudaGraphDecoder:
    """Per-bucket CUDA-graph decode: capture once, then replay each step (CUDA-only).

    Constructed lazily by the engine **only when ``torch.cuda.is_available()``**. Holds
    one captured graph per batch-size bucket (all sharing a single memory pool, R3
    §3.5) plus their fixed-shape static buffers. ``decode`` picks the bucket, copies the
    step's real data into the static buffers, replays, and returns the real rows.
    """

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        pool: PagedKVPool,
        *,
        max_num_seqs: int,
        t_max: int,
        block_size: int,
        use_triton: bool = True,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.pool = pool
        self.buckets = default_buckets(max_num_seqs)
        self.t_max = t_max
        self.block_size = block_size
        self.use_triton = use_triton
        self.warmup_steps = warmup_steps
        self.device = model.device
        self.dtype = model.dtype
        self._graphs: dict[int, _Captured] = {}
        self._mempool: object | None = None  # shared across buckets (R3 §3.5)

    def decode(
        self,
        block_tables: list[list[int]],
        num_cached: list[int],
        last_tokens: list[int],
    ) -> torch.Tensor:
        """Advance the running set one token via graph replay; return ``[b, vocab]``."""
        b = len(last_tokens)
        bucket = pick_bucket(b, self.buckets)
        inp = build_decode_static_inputs(
            block_tables,
            num_cached,
            last_tokens,
            bucket,
            self.t_max,
            self.block_size,
            self.device,
            self.dtype,
        )
        if bucket not in self._graphs:
            self._capture(bucket, inp)
        else:
            self._copy_inputs(self._graphs[bucket].buf, inp)
        self._graphs[bucket].graph.replay()
        return self._graphs[bucket].logits[:b]

    def _forward(self, buf: DecodeInputs) -> torch.Tensor:
        """Run the batched decode forward over ``buf``; return last-token logits."""
        from infrared.cache.paged_kv_cache import PagedContext

        logits = self.model.forward(
            buf.ids,
            buf.positions,
            buf.mask,
            paged=PagedContext(
                self.pool, buf.write_slots, buf.gather_slots, use_triton=self.use_triton
            ),
        )
        return logits[:, -1]  # [B, vocab]

    @staticmethod
    def _copy_inputs(dst: DecodeInputs, src: DecodeInputs) -> None:
        """Overwrite the captured static buffers in place (a graph's input path)."""
        dst.ids.copy_(src.ids)
        dst.positions.copy_(src.positions)
        dst.write_slots.copy_(src.write_slots)
        dst.gather_slots.copy_(src.gather_slots)
        dst.mask.copy_(src.mask)

    def _capture(self, bucket: int, inp: DecodeInputs) -> None:
        """Warm up on a side stream, then capture the decode forward for ``bucket``.

        The static buffers are seeded with this step's real data (``inp``); warmup
        runs the forward a few times so lazy allocs / Triton autotuning settle (the
        paged scatter is idempotent — it re-writes identical K to the same slots), then
        the graph is captured into the shared pool. Follows R3 §3.3's required
        side-stream warmup + rejoin and §3.5's pool sharing.
        """
        import torch  # noqa: F401 — local ref; torch is module-level too

        buf = DecodeInputs(
            inp.ids.clone(),
            inp.positions.clone(),
            inp.write_slots.clone(),
            inp.gather_slots.clone(),
            inp.mask.clone(),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_steps):
                self._forward(buf)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        ctx = (
            torch.cuda.graph(graph)
            if self._mempool is None
            else torch.cuda.graph(graph, pool=self._mempool)
        )
        with ctx:
            logits = self._forward(buf)
        if self._mempool is None:
            self._mempool = graph.pool()
        self._graphs[bucket] = _Captured(graph=graph, buf=buf, logits=logits)
