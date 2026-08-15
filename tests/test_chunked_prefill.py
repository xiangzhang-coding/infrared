"""Chunked prefill (T4b) — the mixed prefill+decode step.

Layers of gate, built bottom-up:

- **Planner (pure, no torch).** ``plan_mixed_step`` decides one step's work:
  decode-first (every in-flight request keeps advancing) then prefill chunks
  filling the leftover token budget — so a long prefill is spread over steps
  instead of blocking the decode queue as one monolithic forward.
- **Varlen mask (pure tensor).** ``build_varlen_mask`` expresses the flattened
  multi-sequence frame — each packed query token attends only its own
  sequence's causal history, so the mixed forward is standalone-identical.
- **Engine (Seam A).** The mixed step's greedy output must stay token-for-token
  identical to one-shot prefill / the T0 oracle across chunk sizes, oversized
  prompts, block boundaries, preemption, and prefix-cache reuse — the reuse of
  the paged read/write means chunking changes *scheduling*, never *output*.
"""

from __future__ import annotations

import pytest

from infrared.engine.scheduler import plan_mixed_step
from infrared.engine.sequence import Sequence

# --- Pure planner: decode-first budgeting (no torch) ------------------------


def _seq(prompt_len: int, num_cached: int, max_new: int = 8) -> Sequence:
    s = Sequence(prompt_ids=list(range(1, prompt_len + 1)), max_new_tokens=max_new)
    s.num_cached_tokens = num_cached  # simulate prefill progress / decode state
    return s


def test_plan_decode_first_reserves_decodes_then_fills_prefill() -> None:
    decoding = _seq(prompt_len=3, num_cached=3)  # fully prefilled → decodes
    prefilling = _seq(prompt_len=10, num_cached=0)  # fresh → needs prefill
    plan = plan_mixed_step([decoding, prefilling], token_budget=4, chunk_size=8)

    assert plan.decode_tokens == 1
    assert plan.prefill_tokens == 3  # budget 4 - 1 decode = 3 left for prefill
    assert plan.is_mixed
    d, p = plan.chunks  # decode first, then prefill
    assert (d.seq is decoding, d.num_query_tokens, d.is_prefill) == (True, 1, False)
    assert (p.seq is prefilling, p.num_query_tokens, p.is_prefill) == (True, 3, True)
    assert p.completes_prefill is False  # 0 + 3 < 10


def test_plan_chunk_size_one_advances_one_token() -> None:
    plan = plan_mixed_step([_seq(5, 0)], token_budget=100, chunk_size=1)
    (c,) = plan.chunks
    assert c.num_query_tokens == 1
    assert c.completes_prefill is False


def test_plan_chunk_size_ge_remaining_is_one_full_chunk() -> None:
    plan = plan_mixed_step([_seq(5, 0)], token_budget=100, chunk_size=8)
    (c,) = plan.chunks
    assert c.num_query_tokens == 5  # capped by remaining, not chunk_size
    assert c.completes_prefill is True  # 0 + 5 == 5
    assert plan.decode_tokens == 0 and not plan.is_mixed


def test_plan_budget_below_decode_count_still_runs_all_decodes() -> None:
    running = [_seq(3, 3), _seq(3, 3), _seq(3, 3)]  # three decodes
    plan = plan_mixed_step(running, token_budget=2, chunk_size=4)
    assert plan.decode_tokens == 3  # decodes are never dropped...
    assert plan.prefill_tokens == 0  # ...and squeeze prefill to nothing


def test_plan_single_token_prompt_completes_immediately() -> None:
    plan = plan_mixed_step([_seq(1, 0)], token_budget=8, chunk_size=8)
    (c,) = plan.chunks
    assert c.num_query_tokens == 1
    assert c.completes_prefill is True  # 0 + 1 == 1


def test_plan_chunks_sum_to_prompt_length_across_steps() -> None:
    """Stepping a lone long prompt in chunks covers exactly num_prompt tokens."""
    seq = _seq(prompt_len=10, num_cached=0)
    taken = 0
    steps = 0
    while seq.is_prefilling:
        plan = plan_mixed_step([seq], token_budget=100, chunk_size=3)
        (c,) = plan.chunks
        seq.num_cached_tokens += c.num_query_tokens  # engine would do this
        taken += c.num_query_tokens
        steps += 1
        if c.completes_prefill:
            assert seq.num_cached_tokens == 10
    assert taken == 10
    assert steps == 4  # 3 + 3 + 3 + 1


# --- Varlen mask: flattened per-sequence causal frame (torch) ---------------


def test_build_varlen_mask_is_block_causal() -> None:
    torch = pytest.importorskip("torch")
    from infrared.model.inputs import build_varlen_mask

    # seq 0: a 2-token prefill chunk at positions [0, 1]
    # seq 1: a decode query at position 2, history positions [0, 1, 2]
    q_seq_ids = torch.tensor([0, 0, 1])
    q_pos = torch.tensor([0, 1, 2])
    k_seq_ids = torch.tensor([0, 0, 1, 1, 1])
    k_pos = torch.tensor([0, 1, 0, 1, 2])
    mask = build_varlen_mask(q_seq_ids, q_pos, k_seq_ids, k_pos, torch.float32, "cpu")

    assert mask.shape == (1, 1, 3, 5)
    allowed = mask[0, 0] == 0.0
    blocked = mask[0, 0] < 0.0
    # q0 (seq0,pos0): only seq0 key at pos<=0 → k0
    assert allowed[0].tolist() == [True, False, False, False, False]
    # q1 (seq0,pos1): seq0 keys at pos<=1 → k0,k1
    assert allowed[1].tolist() == [True, True, False, False, False]
    # q2 (seq1,pos2): seq1 keys at pos<=2 → k2,k3,k4 (never seq0's keys)
    assert allowed[2].tolist() == [False, False, True, True, True]
    assert blocked.sum().item() == 3 * 5 - (1 + 2 + 3)


# --- Engine: Seam-A parity + interleave (torch, tiny model) -----------------


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
        max_position_embeddings=256,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_ids=(),
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def _oracle(model, prompt, max_new):
    from infrared.model.generate import generate

    return generate(
        model, prompt, max_new_tokens=max_new, temperature=0.0, eos_token_ids=()
    ).generated_ids


def _run(model, items, **engine_kwargs):
    """Submit (prompt, max_new) items to a chunked engine; return (engine, outputs).

    Items are submitted as a burst (all at once) so prefill chunks and decodes
    genuinely overlap in the running set — the mixed step under test.
    """
    from infrared.engine.paged_engine import PagedBatchEngine
    from infrared.engine.static_batch import BatchRequest

    engine = PagedBatchEngine(
        model, enable_chunked_prefill=True, **engine_kwargs
    ).start()
    try:
        pendings = [
            engine.submit(BatchRequest(p, max_new_tokens=m, eos_token_ids=()))
            for p, m in items
        ]
        outputs = [p.result(timeout=30) for p in pendings]
    finally:
        engine.stop()
    return engine, outputs


def test_chunked_matches_oracle_across_chunk_sizes() -> None:
    """Seam A: a prompt chunked at any size reproduces one-shot / the T0 oracle."""
    pytest.importorskip("torch")
    model = _tiny_model()
    prompt, max_new = [3, 9, 1, 27, 5, 2, 14, 8, 6, 19], 8  # 10 tokens
    want = _oracle(model, prompt, max_new)
    # block_size 4; chunk sizes below, at, and above block/prompt boundaries.
    for chunk_size in (1, 2, 3, 4, 5, 8, 100):
        _, outputs = _run(
            model,
            [(prompt, max_new)],
            max_num_seqs=4,
            block_size=4,
            num_blocks=64,
            chunk_size=chunk_size,
        )
        assert outputs[0] == want, f"chunk_size={chunk_size} diverged from oracle"


def test_oversized_prompt_spans_chunks_without_bogus_tokens() -> None:
    """A prompt forced into many chunks emits exactly max_new tokens (no mid-prompt)."""
    pytest.importorskip("torch")
    model = _tiny_model()
    prompt = list(range(1, 21))  # 20 tokens
    max_new = 6
    _, outputs = _run(
        model,
        [(prompt, max_new)],
        max_num_seqs=4,
        block_size=4,
        num_blocks=64,
        chunk_size=3,  # ceil(20/3) = 7 prefill chunks before the first token
    )
    assert len(outputs[0]) == max_new  # no token sampled during prefill chunks
    assert outputs[0] == _oracle(model, prompt, max_new)


def test_non_aligned_chunk_tiny_pool_matches_oracle_and_recovers() -> None:
    """chunk_size not a multiple of block_size, multi-block prompts, tight pool."""
    pytest.importorskip("torch")
    model = _tiny_model()
    items = [([3, 9, 1, 27, 5], 8), ([8, 2, 15, 6, 7, 11], 6), ([10, 11, 12], 10)]
    engine, outputs = _run(
        model, items, max_num_seqs=4, block_size=4, num_blocks=32, chunk_size=3
    )
    for (prompt, m), out in zip(items, outputs, strict=True):
        assert out == _oracle(model, prompt, m)
    assert engine.block_manager.num_used_blocks == 0  # pool fully recovered
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_chunked_survives_preemption() -> None:
    """A pool too small forces recompute preemption; chunked output still matches."""
    pytest.importorskip("torch")
    model = _tiny_model()
    items = [([3, 9, 1, 27, 5], 12), ([8, 2], 8), ([10, 11, 12], 6), ([7, 4, 4, 4], 9)]
    engine, outputs = _run(
        model, items, max_num_seqs=4, block_size=4, num_blocks=6, chunk_size=2
    )
    for (prompt, m), out in zip(items, outputs, strict=True):
        assert out == _oracle(model, prompt, m)
    assert engine.block_manager.num_used_blocks == 0


def test_chunked_composes_with_prefix_caching() -> None:
    """Chunked + prefix caching together: reuse still happens, output still correct."""
    pytest.importorskip("torch")
    model = _tiny_model()
    prefix = [5, 5, 1, 2, 7, 3, 9, 4]  # 2 shared blocks at block_size 4
    a, b = prefix + [11], prefix + [12, 13]
    # Submit sequentially so B definitely sees A's registered prefix.
    from infrared.engine.paged_engine import PagedBatchEngine
    from infrared.engine.static_batch import BatchRequest

    engine = PagedBatchEngine(
        model,
        max_num_seqs=4,
        block_size=4,
        num_blocks=64,
        chunk_size=3,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    ).start()
    try:
        out_a = engine.submit(
            BatchRequest(a, max_new_tokens=6, eos_token_ids=())
        ).result(30)
        out_b = engine.submit(
            BatchRequest(b, max_new_tokens=6, eos_token_ids=())
        ).result(30)
    finally:
        engine.stop()
    assert out_a == _oracle(model, a, 6)
    assert out_b == _oracle(model, b, 6)
    assert engine.prefix_reused_blocks > 0  # B reused A's cached prefix blocks


def test_long_prefill_interleaves_with_decode() -> None:
    """The mechanism: a long prefill runs mixed with short requests' decode steps."""
    pytest.importorskip("torch")
    model = _tiny_model()
    long_prompt = list(range(1, 41))  # 40 tokens — many prefill chunks
    shorts = [[2, 3], [4, 5], [6, 7]]
    items = [(long_prompt, 8)] + [(s, 10) for s in shorts]
    engine, outputs = _run(
        model,
        items,
        max_num_seqs=8,
        block_size=4,
        num_blocks=128,
        chunk_size=4,
        max_num_batched_tokens=64,  # generous: shorts finish prefill early, then decode
    )
    for (prompt, m), out in zip(items, outputs, strict=True):
        assert out == _oracle(model, prompt, m)
    # Structural evidence (non-flaky): at least one step carried a prefill chunk of
    # the long prompt together with a decode of a short request.
    assert engine.mixed_steps > 0
