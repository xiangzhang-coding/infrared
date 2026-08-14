"""Static batching correctness + observability (torch-only, tiny random model).

The core T1 gate is **batch-invariance**: running N prompts as one left-padded
static batch must produce exactly the same greedy tokens as running each prompt
alone through the T0 single-request path. If that holds and the single path
matches HF (test_parity), the batched path matches HF too — without needing the
real model here. Also checks the observable waste stats (padding + HOL slack).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infrared.engine.static_batch import BatchRequest, run_static_batch  # noqa: E402
from infrared.model.config import Qwen2Config  # noqa: E402
from infrared.model.generate import generate  # noqa: E402
from infrared.model.qwen2 import Qwen2ForCausalLM  # noqa: E402


def _tiny_model() -> Qwen2ForCausalLM:
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


def test_static_batch_matches_single_request() -> None:
    model = _tiny_model()
    prompts = [[3, 9, 1, 27, 5], [8, 2]]  # different lengths -> left-padding
    max_new = 12

    result = run_static_batch(
        model,
        [BatchRequest(p, max_new_tokens=max_new, eos_token_ids=()) for p in prompts],
    )

    for i, prompt in enumerate(prompts):
        single = generate(
            model, prompt, max_new_tokens=max_new, temperature=0.0, eos_token_ids=()
        )
        assert result.outputs[i] == single.generated_ids, f"seq {i} diverged"


def test_static_batch_reports_padding_and_hol_waste() -> None:
    model = _tiny_model()
    # Seq 0: long prompt, short generation; seq 1: short prompt, long generation.
    requests = [
        BatchRequest([1, 2, 3, 4, 5], max_new_tokens=2, eos_token_ids=()),
        BatchRequest([7], max_new_tokens=6, eos_token_ids=()),
    ]
    result = run_static_batch(model, requests)
    stats = result.stats

    assert stats.batch_size == 2
    assert stats.max_prompt_len == 5
    assert stats.prompt_pad_tokens == 4  # seq 1 padded from 1 -> 5
    assert stats.decode_steps == 6  # lockstep runs until the longest finishes
    # HOL blocking: seq 0 (2 tokens) idles for the remaining 4 steps.
    assert stats.decode_slack_tokens == (6 - 2) + (6 - 6)


def test_empty_batch() -> None:
    result = run_static_batch(_tiny_model(), [])
    assert result.outputs == []
    assert result.stats.batch_size == 0


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_static_batch(_tiny_model(), [BatchRequest([], max_new_tokens=2)])
