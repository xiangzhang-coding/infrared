"""Seam A — correctness parity against HF transformers (the T0 acceptance gate).

On the same weights, infrared must match HF's Qwen2 forward: identical greedy
output and first-step logits within numerical tolerance. HF is used **only** as
the weight source (tokenizer) and the reference oracle — never via ``.generate()``
(ADR-0003). Loaded fp32 on CPU with ``attn_implementation="eager"`` so the
reference takes the closest numerical path to our hand-written attention.

Skips cleanly when torch/transformers or the cached model are unavailable, so
the default no-GPU test run stays green.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _cached_model_dir() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_NAME, local_files_only=True)
    except Exception:
        return None


MODEL_DIR = _cached_model_dir()

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None, reason=f"{MODEL_NAME} not cached locally"
)


@pytest.fixture(scope="module")
def prompt_ids() -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = [
        {
            "role": "user",
            "content": "Give me a short introduction to large language models.",
        }
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids[0]  # 1D LongTensor


@pytest.fixture(scope="module")
def infra_model():
    from infrared.model.qwen2 import Qwen2ForCausalLM

    return Qwen2ForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, device="cpu"
    )


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    )
    return model.eval()


@torch.no_grad()
def test_first_step_logits_match_hf(infra_model, hf_model, prompt_ids) -> None:
    ids = prompt_ids
    kv = infra_model.new_kv_cache(max_len=ids.shape[0] + 1)
    infra_logits = infra_model.forward_single(ids, kv, start_pos=0)[-1]
    hf_logits = hf_model(ids.unsqueeze(0)).logits[0, -1]

    max_abs = (infra_logits - hf_logits).abs().max().item()
    assert int(infra_logits.argmax()) == int(hf_logits.argmax())
    # Same top-5 candidate set (order-insensitive: guards against near-tie flips).
    assert set(infra_logits.topk(5).indices.tolist()) == set(
        hf_logits.topk(5).indices.tolist()
    )
    assert max_abs < 1e-2, f"first-step logits diverge: max |Δ| = {max_abs:.3e}"


@torch.no_grad()
def test_greedy_generation_matches_hf(infra_model, hf_model, prompt_ids) -> None:
    from infrared.model.generate import generate

    ids = prompt_ids
    k = 24

    # infrared greedy (eos disabled to force a fixed-length, comparable run).
    out = generate(
        infra_model, ids.tolist(), max_new_tokens=k, temperature=0.0, eos_token_ids=()
    )
    infra_tokens = out.generated_ids

    # HF greedy via an explicit KV-cache loop — NOT model.generate().
    hf_tokens: list[int] = []
    step = hf_model(ids.unsqueeze(0), use_cache=True)
    past = step.past_key_values
    nxt = int(step.logits[0, -1].argmax())
    for _ in range(k):
        hf_tokens.append(nxt)
        step = hf_model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = step.past_key_values
        nxt = int(step.logits[0, -1].argmax())

    assert infra_tokens == hf_tokens


def test_tied_lm_head_shares_embedding_storage(infra_model) -> None:
    # 0.5B ties lm_head to the token embedding; the tie must survive load + .to().
    assert infra_model.config.tie_word_embeddings
    assert (
        infra_model.lm_head.weight.data_ptr()
        == infra_model.model.embed_tokens.weight.data_ptr()
    )


@torch.no_grad()
def test_generate_text_is_coherent(infra_model) -> None:
    # Exercises the engine's own tokenizer path (load_tokenizer + generate_text).
    from infrared.model.generate import generate_text, load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    text = generate_text(
        infra_model,
        tokenizer,
        "In one sentence, what is a KV cache?",
        max_new_tokens=32,
    )
    assert isinstance(text, str) and text.strip()
