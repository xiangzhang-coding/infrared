"""Engine core — CPU-side orchestration (T1 / T2 — stubs).

Contents (filled across T1–T2 — see ``docs/spec/0001`` and R1 §2/§3):

- ``sequence``  — ``Sequence`` state machine + ``SequenceStatus``.
- ``scheduler`` — waiting/running deques; ``schedule`` / ``preempt`` /
                  ``postprocess`` (continuous batching, the engine's heart).
- ``engine``    — the busy loop: ``add_request`` / ``step`` / ``generate``.

This is the **CPU-side, pure-decision** half of the engine↔worker seam
(R1 §9.1); the GPU execution half (ModelRunner, Sampler) lives in
``infrared.model``, and KV block accounting lives in ``infrared.cache``.
"""
