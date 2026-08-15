"""Engine configuration surface (data only — no logic).

Mirrors the ``config.py`` node of the R1 blueprint (§8). These knobs are the
inputs a future ``Engine`` consumes; the scaffold defines them as a plain
dataclass so tests and tooling have a real, import-safe artifact to touch
without pulling in torch or downloading a model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """Static configuration for the inference engine.

    Defaults follow the R1 blueprint's "起步" recommendations — e.g.
    ``block_size=16`` (§8, finer paging on a single card) and
    ``enforce_eager=True`` (stay correct + measurable before any CUDA-graph
    work). No validation or derivation lives here yet; that arrives with the
    tiers that read these fields.
    """

    # Weight source / correctness-oracle model (ADR-0005): 0.5B for dev, 7B for
    # the headline benchmark.
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"

    # KV paging (T3).
    block_size: int = 16

    # Prefix caching (T4): share a request's cached prompt-prefix KV blocks with
    # later requests that repeat the prefix (system prompt / few-shot). A pure
    # no-op when nothing is shared; see ``engine/paged_engine.py``.
    enable_prefix_caching: bool = True

    # Continuous-batching budgets (T2).
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # Worker sizing / execution mode.
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    enforce_eager: bool = True
