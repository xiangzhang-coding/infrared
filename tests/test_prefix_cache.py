"""Prefix caching (T4) — the content-hashed block pool + engine reuse.

Two layers of gate, mirroring how the mechanism is built:

- **Allocator (pure, no torch).** ``BlockManager`` content-addresses each *full*
  block by a **chained** hash, so ``match_prefix`` reuses a shared prompt prefix
  (ref-counted up), the chain rules out cross-prefix collisions, blocks recycle
  only when their last referrer frees them, and a cached-but-free block is
  re-hittable until it is actually reused (evicted). The match is capped so a
  fully-cached prompt still leaves ≥1 token to compute the first logits.
- **Engine (Seam A).** The paged engine reusing a prefix must not change a single
  output token — greedy output stays identical to the T0 oracle with caching on,
  off, and versus the no-shared-prefix no-op. The reuse is *observable* (the
  engine counts the physical blocks / tokens served from cache) — the T4 win.
"""

from __future__ import annotations

import pytest

from infrared.cache.block_manager import BlockManager

# --- Allocator: content-hashed prefix cache (no torch) ----------------------


def test_match_prefix_is_a_noop_with_nothing_cached() -> None:
    bm = BlockManager(num_blocks=8, block_size=4)
    reused, num_cached = bm.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert reused == []
    assert num_cached == 0
    assert bm.num_used_blocks == 0  # a miss touches nothing


def test_register_then_match_reuses_blocks_and_refcounts() -> None:
    bm = BlockManager(num_blocks=8, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2 full blocks + 1 partial
    table = bm.allocate(9)
    assert len(table) == 3
    bm.register_full_blocks(table, toks)
    assert len(bm.hash_to_block_id) == 2  # only the 2 *full* blocks are cached

    reused, num_cached = bm.match_prefix(toks)  # cap = (9-1)//4 = 2 blocks
    assert reused == table[:2]
    assert num_cached == 8
    # Reused blocks were already owned (ref 1) → touched to ref 2 (shared).
    assert bm.blocks[table[0]].ref_count == 2
    assert bm.blocks[table[1]].ref_count == 2
    # The partial block was never registered, so it is never a hit.
    assert bm.blocks[table[2]].hash is None


def test_match_prefix_leaves_at_least_one_token_to_compute() -> None:
    """A fully-cached prompt still needs ≥1 query token for the first logits."""
    bm = BlockManager(num_blocks=8, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8]  # exactly 2 full blocks, no remainder
    table = bm.allocate(8)
    bm.register_full_blocks(table, toks)
    assert len(bm.hash_to_block_id) == 2

    reused, num_cached = bm.match_prefix(toks)  # cap = (8-1)//4 = 1, not 2
    assert reused == table[:1]  # the last full block is *not* reused
    assert num_cached == 4  # so 4 tokens remain to forward


def test_chained_hash_rules_out_cross_prefix_collision() -> None:
    """A block hash folds in the parent, so identical content under a different
    prefix is a different key — no false reuse of the wrong sequence's KV."""
    bm = BlockManager(num_blocks=16, block_size=4)
    a = [1, 1, 1, 1, 9, 9, 9, 9, 0]  # first block [1,1,1,1], second [9,9,9,9]
    b = [2, 2, 2, 2, 9, 9, 9, 9, 0]  # same second block, different first
    ta = bm.allocate(9)
    bm.register_full_blocks(ta, a)
    tb = bm.allocate(9)
    bm.register_full_blocks(tb, b)

    # Query [1,1,1,1][9,9,9,9]…: the [9,9,9,9] block must resolve under parent
    # hash([1,1,1,1]) → ta's block, never tb's (whose [9,9,9,9] chained off [2,2,2,2]).
    reused, num_cached = bm.match_prefix([1, 1, 1, 1, 9, 9, 9, 9, 7])
    assert reused == [ta[0], ta[1]]
    assert num_cached == 8


def test_shared_block_recycles_only_when_last_referrer_frees() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    toks = [1, 2, 3, 4, 5]  # 1 full block + partial
    t1 = bm.allocate(5)
    bm.register_full_blocks(t1, toks)

    reused, num_cached = bm.match_prefix(toks)  # cap = (5-1)//4 = 1 → reuse block 0
    assert reused == [t1[0]]
    assert num_cached == 4
    assert bm.blocks[t1[0]].ref_count == 2  # shared by two block tables
    t2 = reused + bm.allocate_new(1)  # seq2 = shared prefix + its own tail

    bm.free(t1)  # seq1 done: shared block 2→1 (still live), its tail 1→0 freed
    assert bm.blocks[t1[0]].ref_count == 1
    assert t1[0] in bm.used  # still held by seq2 — not recycled
    bm.free(t2)  # seq2 done: shared block 1→0 now
    assert bm.blocks[t1[0]].ref_count == 0
    assert t1[0] not in bm.used
    assert bm.num_free_blocks == bm.num_blocks  # whole pool recovered


def test_cached_but_free_block_is_resurrected_then_evicted() -> None:
    """Eviction ↔ prefix interaction: a freed cache entry stays hittable until it
    is actually reused for a new allocation, at which point it is evicted."""
    bm = BlockManager(num_blocks=2, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 full blocks — fills the tiny pool
    table = bm.allocate(8)
    bm.register_full_blocks(table, toks)
    assert bm.num_free_blocks == 0

    bm.free(table)  # both blocks ref→0 → free, but keep their cache hashes
    assert bm.num_free_blocks == 2
    assert len(bm.hash_to_block_id) == 2  # cached-but-free, still re-hittable

    reused, num_cached = bm.match_prefix(toks)  # cap 1 → resurrect table[0]
    assert reused == [table[0]]
    assert bm.num_free_blocks == 1  # pulled back out of the free queue
    assert bm.blocks[table[0]].ref_count == 1

    # Now take a fresh block under pressure: the only free block is table[1]
    # (still cached) → allocating it evicts its stale cache entry.
    got = bm.allocate_new(1)
    assert got == [table[1]]
    assert bm.blocks[table[1]].hash is None  # evicted
    assert len(bm.hash_to_block_id) == 1  # only the live (reused) entry remains


def test_disabled_via_never_registering_stays_a_plain_allocator() -> None:
    """Sanity: without registration, the pool is exactly the T3 allocator."""
    bm = BlockManager(num_blocks=4, block_size=4)
    table = bm.allocate(5)
    reused, num_cached = bm.match_prefix([1, 2, 3, 4, 5])  # nothing registered
    assert reused == [] and num_cached == 0
    bm.free(table)
    assert bm.num_free_blocks == 4


def test_admission_gate_credits_a_resident_shared_prefix() -> None:
    """A prefix a *live* sequence still holds costs no new blocks at admission."""
    bm = BlockManager(num_blocks=8, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 3 blocks (2 full + partial)
    seq1 = bm.allocate(9)  # seq1 holds all 3 (ref 1)
    bm.register_full_blocks(seq1, toks)
    # A second identical prompt: its 2 full prefix blocks are resident → credited,
    # so it needs only the 1 non-prefix block, not the whole-prompt count of 3.
    assert bm.blocks_for(len(toks)) == 3  # what the un-credited gate would demand
    assert bm.blocks_needed_with_prefix(toks) == 1


def test_admission_gate_does_not_credit_a_free_cached_prefix() -> None:
    """A cached-but-free prefix block still consumes a free slot on resurrection."""
    bm = BlockManager(num_blocks=8, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    seq1 = bm.allocate(9)
    bm.register_full_blocks(seq1, toks)
    bm.free(seq1)  # every block now cached-but-free (ref 0)
    # No live holder → no credit: the prefill resurrects + allocates from free.
    assert bm.blocks_needed_with_prefix(toks) == 3


# --- Engine: Seam A + observable reuse (needs torch + a tiny model) ---------


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


# A shared 2-block prefix (block_size 4) plus per-request tails.
_PREFIX = [5, 5, 1, 2, 7, 3, 9, 4]
_A = _PREFIX + [11]
_B = _PREFIX + [12, 13]


def _run_sequential(engine, prompts_and_max):
    """Submit prompts one-at-a-time so the block manager state is deterministic."""
    from infrared.engine.static_batch import BatchRequest

    outputs = []
    for prompt, m in prompts_and_max:
        out = engine.submit(
            BatchRequest(prompt, max_new_tokens=m, eos_token_ids=())
        ).result(timeout=30)
        outputs.append(out)
    return outputs


def test_shared_prefix_is_reused_and_output_is_unchanged() -> None:
    """The second request reuses the first's prefix blocks, bit-for-bit oracle-clean."""
    pytest.importorskip("torch")
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _tiny_model()
    engine = PagedBatchEngine(
        model, max_num_seqs=4, block_size=4, num_blocks=64, enable_prefix_caching=True
    ).start()
    try:
        outputs = _run_sequential(engine, [(_A, 6), (_B, 6)])
    finally:
        engine.stop()

    assert outputs[0] == _oracle(model, _A, 6)
    assert outputs[1] == _oracle(model, _B, 6)
    # Only the second request could reuse — it hit both shared prefix blocks.
    assert engine.prefix_reused_blocks == 2
    assert engine.prefix_reused_tokens == 8


def test_no_shared_prefix_is_a_noop() -> None:
    """Distinct prompts share no full block → zero reuse, still oracle-clean."""
    pytest.importorskip("torch")
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _tiny_model()
    distinct = [([3, 9, 1, 27, 5], 6), ([8, 2, 15, 6], 6), ([10, 11, 12, 13, 14], 6)]
    engine = PagedBatchEngine(
        model, max_num_seqs=4, block_size=4, num_blocks=64, enable_prefix_caching=True
    ).start()
    try:
        outputs = _run_sequential(engine, distinct)
    finally:
        engine.stop()

    for (prompt, m), out in zip(distinct, outputs, strict=True):
        assert out == _oracle(model, prompt, m)
    assert engine.prefix_reused_blocks == 0  # no-op when nothing is shared


def test_caching_on_off_produce_identical_output() -> None:
    """Reuse changes *allocation*, never *output*: on and off must agree (and both
    agree with the oracle) — the Seam-A invariance the feature must preserve."""
    pytest.importorskip("torch")
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _tiny_model()
    work = [(_A, 8), (_B, 8), (_A, 8)]  # a repeat so caching definitely bites

    def run(enable: bool):
        engine = PagedBatchEngine(
            model,
            max_num_seqs=4,
            block_size=4,
            num_blocks=64,
            enable_prefix_caching=enable,
        ).start()
        try:
            return _run_sequential(engine, work), engine.prefix_reused_blocks
        finally:
            engine.stop()

    on_outputs, on_reused = run(True)
    off_outputs, off_reused = run(False)

    assert on_outputs == off_outputs  # caching is output-invariant
    for (prompt, m), out in zip(work, on_outputs, strict=True):
        assert out == _oracle(model, prompt, m)
    assert on_reused > 0  # caching actually engaged
    assert off_reused == 0  # the control reused nothing
