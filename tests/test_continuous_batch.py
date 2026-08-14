"""Continuous batching (T2) — correctness seam + the mechanism's observable wins.

Two layers of gate:

- **Seam A (correctness).** A continuous-batch generation must be *token-for-
  token identical* to running each prompt alone through the T0 single-request
  path — even when more requests are in flight than there are running slots (so
  the scheduler genuinely admits/retires across steps). Since T2 reuses the T0
  ``forward_single`` against a per-sequence KV cache, this holds by construction;
  the test pins it against silent regressions.
- **Mechanism.** The pure-decision ``Scheduler`` admits up to the cap and retires
  on stop; and against the *same* mixed workload the continuous engine removes
  static batching's two wastes — prompt **padding** (fill's prefill grid) and
  **head-of-line** slack (finished seqs still forwarded) — driving batch-fill to
  100% and prompt-pad to 0, and it **streams** a real first token (TTFT) instead
  of returning all-or-nothing.
"""

from __future__ import annotations

import pytest

from infrared.engine.scheduler import Scheduler
from infrared.engine.sequence import Sequence, SequenceStatus

# --- Scheduler / Sequence: pure decision layer (no torch) -------------------


def test_sequence_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Sequence(prompt_ids=[])


def test_scheduler_admits_up_to_cap_then_decodes() -> None:
    sched = Scheduler(max_num_seqs=2)
    for p in ([1, 2], [3], [4, 5, 6]):  # three requests, only two slots
        sched.add(Sequence(prompt_ids=p, max_new_tokens=4))
    assert not sched.is_finished()

    # First two steps admit (prefill) one waiting seq each, filling the cap.
    b0 = sched.schedule()
    assert b0.is_prefill and len(b0.seqs) == 1
    assert b0.seqs[0].status is SequenceStatus.RUNNING
    b1 = sched.schedule()
    assert b1.is_prefill and len(b1.seqs) == 1
    assert len(sched.running) == 2

    # Cap reached (and one request still waiting): now steps decode the whole
    # running set — the third request stays queued until a slot frees.
    b2 = sched.schedule()
    assert not b2.is_prefill
    assert len(b2.seqs) == 2
    assert len(sched.waiting) == 1


def test_scheduler_retire_frees_a_slot_for_the_next_admission() -> None:
    sched = Scheduler(max_num_seqs=1)
    first = Sequence(prompt_ids=[1, 2], max_new_tokens=4)
    second = Sequence(prompt_ids=[3], max_new_tokens=4)
    sched.add(first)
    sched.add(second)

    sched.schedule()  # admits `first` (cap now full)
    decode = sched.schedule()  # decode step over the running set
    assert decode.seqs == [first]
    assert sched.schedule().seqs == [first]  # still full: `second` waits

    sched.retire(first)  # `first` stops -> slot frees
    assert first.status is SequenceStatus.FINISHED
    admit = sched.schedule()  # next step admits `second`
    assert admit.is_prefill and admit.seqs == [second]

    sched.retire(second)
    assert sched.is_finished()


# --- Engine: correctness + mechanism (needs torch + a tiny model) -----------


def _tiny_model():
    import torch

    from infrared.model.config import Qwen2Config
    from infrared.model.qwen2 import Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_ids=(),
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.lm_head.weight = model.model.embed_tokens.weight  # tie, like 0.5B
    return model.eval()


def test_continuous_batch_matches_single_request() -> None:
    """Seam A: batch-invariance vs the T0 path, with admission across steps."""
    pytest.importorskip("torch")
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest
    from infrared.model.generate import generate

    model = _tiny_model()
    # More requests (4) than running slots (2) forces continuous admit/retire.
    prompts = [[3, 9, 1, 27, 5], [8, 2], [10, 11, 12], [7]]
    max_new = [12, 8, 6, 10]

    engine = ContinuousBatchEngine(model, max_num_seqs=2).start()
    try:
        pendings = [
            engine.submit(BatchRequest(p, max_new_tokens=m, eos_token_ids=()))
            for p, m in zip(prompts, max_new, strict=True)
        ]
        got = [p.result(timeout=30) for p in pendings]
    finally:
        engine.stop()

    for i, (prompt, m) in enumerate(zip(prompts, max_new, strict=True)):
        single = generate(
            model, prompt, max_new_tokens=m, temperature=0.0, eos_token_ids=()
        )
        assert got[i] == single.generated_ids, f"seq {i} diverged from T0"


def test_continuous_batch_eliminates_padding_and_hol_waste() -> None:
    """Same mixed workload: continuous fill=100% / pad=0; static <100% / pad>0."""
    pytest.importorskip("torch")
    from infrared.bench.harness import LoadRequest, run_load, utilization_from
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest, run_static_batch

    model = _tiny_model()
    # Mixed shapes so static batching pays both wastes: seq 0 is a long prompt /
    # short generation (padding victim of the short prompt; HOL victim of the
    # long generation), seq 1 the opposite.
    requests = [
        LoadRequest(prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=2),
        LoadRequest(prompt_ids=[7], max_new_tokens=8),
    ]

    # The static-batch "before": one batch over both prompts (run the mechanism
    # directly so its inherent pad + HOL waste is deterministic, not subject to
    # the engine's batching-window timing).
    static_result = run_static_batch(
        model,
        [
            BatchRequest(
                r.prompt_ids, max_new_tokens=r.max_new_tokens, eos_token_ids=()
            )
            for r in requests
        ],
    )
    static_util = utilization_from([static_result.stats])

    cont = ContinuousBatchEngine(model, max_num_seqs=8).start()
    try:
        cont_util = utilization_from(run_load(cont, requests).batch_stats)
    finally:
        cont.stop()

    # Static batch wastes work on padding + head-of-line idling.
    assert static_util.batch_fill_rate is not None
    assert static_util.batch_fill_rate < 1.0
    assert static_util.prompt_pad_fraction is not None
    assert static_util.prompt_pad_fraction > 0.0

    # Continuous batch forwards only live sequences and pads nothing.
    assert cont_util.batch_fill_rate == pytest.approx(1.0)
    assert cont_util.prompt_pad_fraction == pytest.approx(0.0)
    assert cont_util.batch_fill_rate > static_util.batch_fill_rate  # the jump


def test_continuous_batch_streams_first_token() -> None:
    """A multi-token request exposes a real TTFT (first token before completion)."""
    pytest.importorskip("torch")
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest

    model = _tiny_model()
    engine = ContinuousBatchEngine(model, max_num_seqs=4).start()
    try:
        pending = engine.submit(BatchRequest([5, 6, 7], max_new_tokens=8))
        assert pending.wait_first_token(timeout=30)
        first_token_time = pending.first_token_time
        pending.result(timeout=30)
    finally:
        engine.stop()

    assert first_token_time is not None  # streamed, not all-or-nothing


def test_continuous_batch_rejects_empty_prompt_in_isolation() -> None:
    """An empty prompt fails only its own request; the engine keeps running."""
    pytest.importorskip("torch")
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest

    model = _tiny_model()
    engine = ContinuousBatchEngine(model, max_num_seqs=4).start()
    try:
        bad = engine.submit(BatchRequest([], max_new_tokens=4))
        with pytest.raises(ValueError, match="non-empty"):
            bad.result(timeout=30)
        # A subsequent valid request still completes.
        good = engine.submit(BatchRequest([1, 2], max_new_tokens=4))
        assert len(good.result(timeout=30)) == 4
    finally:
        engine.stop()


def test_bad_request_fails_alone_without_bricking_the_engine() -> None:
    """A forward that raises fails only that request; batch-mates still finish.

    Regression guard: an out-of-vocab token makes prefill raise mid-step. That
    must fail *only* its own request and retire the sequence — never take down
    concurrent requests or wedge the busy loop re-running the failing step.
    """
    pytest.importorskip("torch")
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest

    model = _tiny_model()  # vocab_size=64
    engine = ContinuousBatchEngine(model, max_num_seqs=4).start()
    try:
        good_before = engine.submit(BatchRequest([1, 2, 3], max_new_tokens=6))
        bad = engine.submit(BatchRequest([999], max_new_tokens=6))  # id >= vocab
        good_after = engine.submit(BatchRequest([4, 5], max_new_tokens=6))

        with pytest.raises(Exception):  # noqa: B017 — IndexError from embed lookup
            bad.result(timeout=30)
        assert len(good_before.result(timeout=30)) == 6
        assert len(good_after.result(timeout=30)) == 6
    finally:
        engine.stop()


def test_max_new_tokens_zero_matches_oracle() -> None:
    """A zero generation budget yields no tokens, like the T0/T1 oracles."""
    pytest.importorskip("torch")
    from infrared.engine.engine import ContinuousBatchEngine
    from infrared.engine.static_batch import BatchRequest

    model = _tiny_model()
    engine = ContinuousBatchEngine(model, max_num_seqs=4).start()
    try:
        assert engine.submit(BatchRequest([1, 2], max_new_tokens=0)).result(30) == []
    finally:
        engine.stop()
