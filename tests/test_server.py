"""HTTP-shell tests: OpenAI-compatible /v1/completions over the static-batch engine.

Uses a tiny random model + a stub tokenizer + a real StaticBatchEngine, so the
FastAPI + queue + batching plumbing is exercised with no GPU and no model
download. Concurrency check confirms simultaneous requests batch together.
"""

from __future__ import annotations

import concurrent.futures as cf

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from infrared.engine.engine import StaticBatchEngine  # noqa: E402
from infrared.model.config import Qwen2Config  # noqa: E402
from infrared.model.qwen2 import Qwen2ForCausalLM  # noqa: E402
from infrared.server.app import create_app  # noqa: E402


class StubTokenizer:
    """Deterministic char-level stub (ids stay within the tiny model's vocab)."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 40) + 1 for c in text][:8]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


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
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


@pytest.fixture(scope="module")
def client():
    engine = StaticBatchEngine(_tiny_model(), max_batch_size=4, linger=0.03).start()
    app = create_app(engine, StubTokenizer(), eos_token_ids=())
    with TestClient(app) as test_client:
        yield test_client
    engine.stop()


def test_completions_response_shape(client) -> None:
    resp = client.post(
        "/v1/completions", json={"prompt": "hello world", "max_tokens": 5}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert isinstance(body["choices"][0]["text"], str)
    assert body["usage"]["completion_tokens"] == 5
    assert body["infrared_batch"]["batch_size"] >= 1


def test_multiple_prompts_form_one_batch(client) -> None:
    resp = client.post(
        "/v1/completions", json={"prompt": ["aaaa", "bb"], "max_tokens": 3}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["choices"]) == 2
    assert body["infrared_batch"]["batch_size"] == 2
    # Padding waste is observable: "bb" is padded up to len("aaaa").
    assert body["infrared_batch"]["prompt_pad_tokens"] == 2


def test_concurrent_requests_batch_together(client) -> None:
    prompts = ["alpha", "beta", "gamma", "delta"]

    def call(p: str):
        return client.post("/v1/completions", json={"prompt": p, "max_tokens": 4})

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(call, prompts))

    assert all(r.status_code == 200 for r in responses)
    max_batch = max(r.json()["infrared_batch"]["batch_size"] for r in responses)
    assert max_batch >= 2  # concurrent arrivals were batched by the worker


def test_empty_prompt_returns_400(client) -> None:
    resp = client.post("/v1/completions", json={"prompt": "", "max_tokens": 4})
    assert resp.status_code == 400


def test_engine_shutdown_does_not_hang_waiters() -> None:
    # A never-started engine has no worker; stopping must fail queued waiters
    # instead of leaving them blocked forever, and reject new submissions.
    from infrared.engine.engine import StaticBatchEngine
    from infrared.engine.static_batch import BatchRequest

    engine = StaticBatchEngine(_tiny_model())  # not started
    pending = engine.submit(BatchRequest([1, 2, 3], max_new_tokens=2))
    engine.stop()
    with pytest.raises(RuntimeError):
        pending.result(timeout=1.0)
    with pytest.raises(RuntimeError):
        engine.submit(BatchRequest([1], max_new_tokens=1))
