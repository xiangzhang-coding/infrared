"""Worker / ModelRunner — GPU-side pure execution (T1 / T3 — stub).

Holds model weights + KV cache tensors + sampler. ``execute`` is the **narrow
engine↔worker seam**: given the scheduler's output it flattens the batch, builds
the ``slot_mapping`` / ``block_tables`` tensors, runs one forward (naive at T0,
paged at T3, Triton at T4), gathers last-token logits, and samples (R1 §5/§6).

The seam is kept serializable so T6 can split it across processes/GPUs without
touching the scheduler or block manager (R1 §9.1).
"""

from __future__ import annotations

_T1 = "not implemented until T1 — see docs/spec/0001 and R1 blueprint §5"


class ModelRunner:
    """Prepares inputs, runs forward, samples — the GPU executor."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T1)

    def execute(self, *args: object, **kwargs: object) -> object:
        """The seam: ``execute(scheduler_output) -> per-seq sampled token ids``."""
        raise NotImplementedError(_T1)
