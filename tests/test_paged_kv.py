"""Paged KV block manager + paged engine (T3) — allocator, Seam A, occupancy.

Layers of gate:

- **Allocator (pure, no torch).** ``BlockManager`` hands out fixed-size blocks
  from a shared pool, grows a block table on demand, and returns blocks whole on
  free — so there is no external fragmentation and a completed run recovers the
  whole pool.
- **Seam A (correctness).** The paged engine's greedy output must stay
  token-for-token identical to the T0 single-request oracle — with an ample pool
  *and* with a pool so small it forces recompute **preemption** (the batched
  paged decode + preempt/resume must reproduce each sequence exactly).
- **Mechanism.** The engine reports **KV block occupancy**, and its on-demand
  paging completes a workload whose worst-case contiguous reservation would not
  fit the pool at once — the concurrency / no-fragmentation-OOM win.
"""

from __future__ import annotations

import pytest

from infrared.cache.block_manager import BlockManager

# --- BlockManager: the allocator (no torch) ---------------------------------


def test_block_manager_allocate_append_free_roundtrip() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    assert bm.num_free_blocks == 4
    assert bm.blocks_for(0) == 0
    assert bm.blocks_for(1) == 1
    assert bm.blocks_for(4) == 1
    assert bm.blocks_for(5) == 2

    table = bm.allocate(5)  # 5 tokens -> 2 blocks
    assert len(table) == 2
    assert bm.num_free_blocks == 2
    assert bm.num_used_blocks == 2

    # Position 8 is a block boundary (blocks hold 0-3, 4-7): appending needs one.
    bm.append(table, cur_len=8)
    assert len(table) == 3
    assert bm.num_free_blocks == 1
    # Position 9 fits in the last block: no growth.
    bm.append(table, cur_len=9)
    assert len(table) == 3

    bm.free(table)
    assert bm.num_free_blocks == 4  # fully recovered — no leak, no fragmentation
    assert bm.num_used_blocks == 0


def test_block_manager_can_allocate_gates_on_capacity() -> None:
    bm = BlockManager(num_blocks=2, block_size=4)
    assert bm.can_allocate(8) is True  # exactly 2 blocks
    assert bm.can_allocate(9) is False  # 3 blocks, only 2 exist
    bm.allocate(8)
    assert bm.can_allocate(1) is False  # pool full
    with pytest.raises(ValueError, match="cannot allocate"):
        bm.allocate(1)


# --- Engine: Seam A + mechanism (needs torch + a tiny model) ----------------


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
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def _oracle(model, prompt, max_new):
    from infrared.model.generate import generate

    return generate(
        model, prompt, max_new_tokens=max_new, temperature=0.0, eos_token_ids=()
    ).generated_ids


_PROMPTS = [[3, 9, 1, 27, 5], [8, 2], [10, 11, 12], [7], [4, 4, 4, 4]]
_MAX_NEW = [12, 8, 6, 10, 9]


def _run_paged(model, *, block_size, num_blocks, max_num_seqs):
    from infrared.engine.paged_engine import PagedBatchEngine
    from infrared.engine.static_batch import BatchRequest

    engine = PagedBatchEngine(
        model,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        num_blocks=num_blocks,
    ).start()
    try:
        pendings = [
            engine.submit(BatchRequest(p, max_new_tokens=m, eos_token_ids=()))
            for p, m in zip(_PROMPTS, _MAX_NEW, strict=True)
        ]
        outputs = [p.result(timeout=30) for p in pendings]
    finally:
        engine.stop()
    return engine, outputs


def test_paged_engine_matches_t0_with_ample_pool() -> None:
    """Seam A: batched paged decode reproduces the T0 oracle token-for-token."""
    pytest.importorskip("torch")
    model = _tiny_model()
    _, outputs = _run_paged(model, block_size=8, num_blocks=64, max_num_seqs=4)
    for i, (prompt, m) in enumerate(zip(_PROMPTS, _MAX_NEW, strict=True)):
        assert outputs[i] == _oracle(model, prompt, m), f"seq {i} diverged from T0"


def test_paged_engine_matches_t0_under_preemption() -> None:
    """Seam A survives recompute preemption: a pool too small forces evict/resume."""
    pytest.importorskip("torch")
    model = _tiny_model()
    # Small blocks + few of them: the running set outgrows the pool mid-decode,
    # forcing recompute preemption. Output must still match the oracle exactly.
    engine, outputs = _run_paged(model, block_size=4, num_blocks=6, max_num_seqs=4)
    for i, (prompt, m) in enumerate(zip(_PROMPTS, _MAX_NEW, strict=True)):
        assert outputs[i] == _oracle(model, prompt, m), (
            f"seq {i} diverged under preempt"
        )
    # Pool fully recovered once everything finished — no leak, no fragmentation.
    assert engine.block_manager.num_used_blocks == 0
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_paged_engine_reports_kv_block_occupancy() -> None:
    """The +paged tier fills the KV-occupancy metric T1/T2 leave as None."""
    pytest.importorskip("torch")
    from infrared.bench.harness import LoadRequest, run_load, utilization_from
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _tiny_model()
    requests = [
        LoadRequest(prompt_ids=p, max_new_tokens=m)
        for p, m in zip(_PROMPTS, _MAX_NEW, strict=True)
    ]
    engine = PagedBatchEngine(
        model, max_num_seqs=4, block_size=8, num_blocks=64
    ).start()
    try:
        util = utilization_from(run_load(engine, requests).batch_stats)
    finally:
        engine.stop()

    assert util.kv_block_occupancy is not None
    assert 0.0 < util.kv_block_occupancy <= 1.0
    assert util.batch_fill_rate == pytest.approx(1.0)  # batched decode, no HOL/pad


def test_paged_on_demand_beats_worst_case_reservation() -> None:
    """On-demand paging completes a workload that worst-case reservation can't hold.

    Each of the 5 requests would reserve ceil((prompt+max_new)/block_size) blocks
    if it grabbed its whole generation up front; summed, that exceeds the pool.
    Paged allocation (grow on demand, free on finish, preempt under pressure) runs
    them all correctly anyway — the concurrency / no-OOM win over T2's per-sequence
    contiguous reservation.
    """
    pytest.importorskip("torch")
    model = _tiny_model()
    block_size, num_blocks = 4, 8
    worst_case = sum(
        (len(p) + m + block_size - 1) // block_size
        for p, m in zip(_PROMPTS, _MAX_NEW, strict=True)
    )
    assert worst_case > num_blocks  # reserving each seq's max up front would not fit

    engine, outputs = _run_paged(
        model, block_size=block_size, num_blocks=num_blocks, max_num_seqs=5
    )
    for i, (prompt, m) in enumerate(zip(_PROMPTS, _MAX_NEW, strict=True)):
        assert outputs[i] == _oracle(model, prompt, m), f"seq {i} diverged"
    assert engine.block_manager.num_free_blocks == num_blocks  # pool recovered


def test_paged_bad_request_fails_alone_without_bricking_the_engine() -> None:
    """An out-of-vocab prompt fails only its own request; batch-mates still finish.

    Regression guard: the paged prefill must isolate a per-sequence forward error
    (and free its blocks) rather than let it reach the busy loop's fatal handler
    and fail every concurrent request — the same contract the T2 engine keeps.
    """
    pytest.importorskip("torch")
    from infrared.engine.paged_engine import PagedBatchEngine
    from infrared.engine.static_batch import BatchRequest

    model = _tiny_model()  # vocab_size=64
    engine = PagedBatchEngine(
        model, max_num_seqs=4, block_size=8, num_blocks=64
    ).start()
    try:
        good_before = engine.submit(BatchRequest([1, 2, 3], max_new_tokens=6))
        bad = engine.submit(BatchRequest([999], max_new_tokens=6))  # id >= vocab
        good_after = engine.submit(BatchRequest([4, 5], max_new_tokens=6))

        with pytest.raises(Exception):  # noqa: B017 — IndexError from embed lookup
            bad.result(timeout=30)
        assert len(good_before.result(timeout=30)) == 6
        assert len(good_after.result(timeout=30)) == 6
        # The failed request leaked no blocks.
        assert engine.block_manager.num_used_blocks == 0
    finally:
        engine.stop()
