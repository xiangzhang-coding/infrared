"""Self-implemented Qwen2.5 forward + its GPU worker (T0 / T1 — stubs).

Contents (filled starting at T0 — see ``docs/spec/0001`` and R1 §2/§5):

- ``layers``       — RMSNorm, RoPE, GQA attention, SwiGLU building blocks.
- ``qwen2``        — assembles the blocks into a ``Qwen2ForCausalLM``-equivalent
                     forward, loading weights from HF safetensors (weights only,
                     ADR-0003/0005).
- ``model_runner`` — the Worker: flattens the batch, builds slot_mapping /
                     block_tables, runs forward, gathers logits.
- ``sampler``      — greedy / temperature / top-p.

This is the **GPU-side, pure-execution** half of the engine↔worker seam
(R1 §9.1); the CPU-side decision half lives in ``infrared.engine``.
"""
