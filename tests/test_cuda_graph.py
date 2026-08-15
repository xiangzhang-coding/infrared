"""T4d — CUDA-graph decode capture/replay + its CPU-verifiable seams.

The capture/replay itself is **GPU-only** (`torch.cuda.CUDAGraph` needs a CUDA
device), so on this Mac/CPU box it never runs — the graph-vs-eager parity check is
CUDA-gated and `skip`s here, validated on AutoDL (the ticket's cross-platform rule).
What *is* checkable on CPU, and pinned here:

1. **Bucketing** (`default_buckets`, `pick_bucket`) — pure integer logic.
2. **Fixed-shape input construction** (`build_decode_static_inputs`) — real rows match
   the eager per-sequence build; padded rows replicate row 0; the key axis pads to
   ``t_max`` with the tail masked.
3. **The captured *workload* is numerically identical.** Running the graph-shaped
   padded forward eagerly and slicing ``[:b]`` equals the tight eager decode forward —
   so padding-to-bucket + padding-the-key-axis-to-``t_max`` don't perturb the real
   rows. Only the CUDAGraph wrapper around this workload is GPU-only.
4. **The engine gate** — ``enable_cuda_graph=True`` is a no-op on CPU (stays eager).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infrared.cache.paged_kv_cache import PagedContext, PagedKVPool  # noqa: E402
from infrared.engine.cuda_graph import (  # noqa: E402
    build_decode_static_inputs,
    default_buckets,
    pick_bucket,
)

# --- 1. pure bucketing ------------------------------------------------------


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [(1, (1,)), (2, (1, 2)), (6, (1, 2, 4, 6)), (8, (1, 2, 4, 8)), (5, (1, 2, 4, 5))],
)
def test_default_buckets(max_num_seqs, expected) -> None:
    assert default_buckets(max_num_seqs) == expected


def test_default_buckets_rejects_zero() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        default_buckets(0)


def test_pick_bucket_rounds_up_and_bounds() -> None:
    buckets = (1, 2, 4, 8)
    assert pick_bucket(1, buckets) == 1
    assert pick_bucket(3, buckets) == 4
    assert pick_bucket(8, buckets) == 8
    with pytest.raises(ValueError, match="exceeds largest bucket"):
        pick_bucket(9, buckets)


# --- 2. fixed-shape decode input construction -------------------------------


def test_build_static_inputs_shapes_and_real_rows() -> None:
    """Real rows carry each seq's id/position/write-slot/history; shapes are fixed."""
    block_tables = [[0, 1], [2]]  # block_size 4: seq0 slots 0-7, seq1 slots 8-11
    num_cached = [5, 2]  # seq0 next token at pos 5 (slot 1), seq1 at pos 2 (slot 10)
    last_tokens = [7, 9]
    inp = build_decode_static_inputs(
        block_tables,
        num_cached,
        last_tokens,
        bucket_b=4,
        t_max=16,
        block_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    assert inp.ids.shape == (4, 1)
    assert inp.positions.shape == (4, 1)
    assert inp.write_slots.shape == (4,)
    assert inp.gather_slots.shape == (4, 16)
    assert inp.mask.shape == (4, 1, 1, 16)

    # seq0: pos 5 -> block_table[5//4=1]=1, 1*4 + 5%4=1 -> slot 5; history [0..5].
    assert inp.positions[0, 0].item() == 5
    assert inp.write_slots[0].item() == 1 * 4 + 1  # slot 5
    assert inp.ids[0, 0].item() == 7
    # seq1: pos 2 -> block 2, 2*4 + 2 = slot 10; history [0..2] via block 2.
    assert inp.write_slots[1].item() == 2 * 4 + 2  # slot 10
    # mask: seq0 valid for cols [0..5], blocked from 6; seq1 valid [0..2].
    neg = torch.finfo(torch.float32).min
    assert (inp.mask[0, 0, 0, :6] == 0).all()
    assert (inp.mask[0, 0, 0, 6:] == neg).all()
    assert (inp.mask[1, 0, 0, :3] == 0).all()
    assert (inp.mask[1, 0, 0, 3:] == neg).all()


def test_build_static_inputs_pads_by_repeating_row_zero() -> None:
    """Padded rows [b:B] replicate row 0 exactly (ids/positions/slots/mask)."""
    inp = build_decode_static_inputs(
        [[0], [1]],
        [1, 0],
        [3, 4],
        bucket_b=4,
        t_max=8,
        block_size=8,
        device="cpu",
        dtype=torch.float32,
    )
    for pad in (2, 3):
        assert inp.ids[pad, 0] == inp.ids[0, 0]
        assert inp.positions[pad, 0] == inp.positions[0, 0]
        assert inp.write_slots[pad] == inp.write_slots[0]
        assert torch.equal(inp.gather_slots[pad], inp.gather_slots[0])
        assert torch.equal(inp.mask[pad], inp.mask[0])


def test_build_static_inputs_rejects_overlong_context() -> None:
    with pytest.raises(ValueError, match="exceeds t_max"):
        build_decode_static_inputs(
            [[0]],
            [10],
            [1],
            bucket_b=1,
            t_max=8,
            block_size=8,
            device="cpu",
            dtype=torch.float32,
        )


# --- 3. the captured workload is numerically identical to eager decode ------


def _tiny_model():
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


def _decode_logits(model, pool, block_tables, num_cached, last_tokens, bucket_b, t_max):
    """Build fixed-shape inputs, run one eager decode forward -> logits [b, vocab]."""
    inp = build_decode_static_inputs(
        block_tables,
        num_cached,
        last_tokens,
        bucket_b,
        t_max,
        block_size=8,
        device="cpu",
        dtype=model.dtype,
    )
    logits = model.forward(
        inp.ids,
        inp.positions,
        inp.mask,
        paged=PagedContext(pool, inp.write_slots, inp.gather_slots, use_triton=False),
    )
    return logits[:, -1]


@torch.no_grad()
def test_graph_shaped_forward_matches_tight_eager_decode() -> None:
    """Padding to a bucket + padding the key axis to t_max must not move real rows.

    This is the CPU proxy for the CUDA graph's numerical-consistency gate: the exact
    workload a graph would capture (fixed [B, t_max] buffers), run eagerly and sliced
    ``[:b]``, equals the tight eager decode over just the b real rows at their true
    context length. The only thing left GPU-only is the CUDAGraph capture/replay.
    """
    model = _tiny_model()
    num_kv_heads, head_dim, n_layers = 2, 8, 2
    pool = PagedKVPool(
        num_layers=n_layers,
        num_blocks=4,
        block_size=8,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
    )
    # Seed some ragged history: seq0 (block 0) has 3 tokens, seq1 (block 1) has 5.
    torch.manual_seed(1)
    for layer in range(n_layers):
        pool.k[layer, 0:3] = torch.randn(3, num_kv_heads, head_dim)
        pool.v[layer, 0:3] = torch.randn(3, num_kv_heads, head_dim)
        pool.k[layer, 8:13] = torch.randn(5, num_kv_heads, head_dim)
        pool.v[layer, 8:13] = torch.randn(5, num_kv_heads, head_dim)
    block_tables = [[0], [1]]
    num_cached = [3, 5]  # next tokens at slots 3 and 13
    last_tokens = [7, 11]

    tight = _decode_logits(model, pool, block_tables, num_cached, last_tokens, 2, 6)
    padded = _decode_logits(model, pool, block_tables, num_cached, last_tokens, 4, 32)
    assert torch.allclose(padded[:2], tight, atol=1e-5)


# --- 4. the engine gate is a no-op on CPU -----------------------------------


def test_enable_cuda_graph_is_noop_on_cpu() -> None:
    """enable_cuda_graph=True with no CUDA stays eager (output identical to off)."""
    from infrared.engine.static_batch import BatchRequest

    def run(enable_cuda_graph):
        from infrared.engine.paged_engine import PagedBatchEngine

        model = _tiny_model()
        engine = PagedBatchEngine(
            model,
            max_num_seqs=4,
            block_size=8,
            num_blocks=32,
            enable_cuda_graph=enable_cuda_graph,
        ).start()
        try:
            assert engine._use_cuda_graph() is False  # no CUDA on this box
            pend = engine.submit(
                BatchRequest([3, 9, 1], max_new_tokens=8, eos_token_ids=())
            )
            return pend.result(timeout=30)
        finally:
            engine.stop()

    assert run(True) == run(False)  # graph flag makes no difference on CPU


def test_graph_key_axis_is_capped_below_full_context_window() -> None:
    """t_max caps at graph_max_seq_len, and never exceeds the model's context window.

    Guards the review fix: a decode graph must not fix its key axis to the full
    (e.g. 32k) window — that would gather/attend a huge masked span every step.
    """
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _tiny_model()  # max_position_embeddings = 128
    engine = PagedBatchEngine(model, graph_max_seq_len=64)
    assert engine._graph_t_max == 64  # capped by the requested max
    engine = PagedBatchEngine(model, graph_max_seq_len=4096)
    assert engine._graph_t_max == 128  # capped by the model's context window


# --- 5. real capture/replay vs eager — CUDA-only (skips on Mac/CPU) ---------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are GPU-only")
def test_cuda_graph_decode_matches_eager_on_cuda() -> None:
    """On CUDA: greedy output with graphs on must match graphs off, token-for-token."""
    from infrared.engine.paged_engine import PagedBatchEngine
    from infrared.engine.static_batch import BatchRequest

    prompts = [[3, 9, 1, 27, 5], [8, 2], [10, 11, 12]]
    max_new = [10, 8, 6]

    def run(enable_cuda_graph):
        model = _tiny_model().to("cuda")
        engine = PagedBatchEngine(
            model,
            max_num_seqs=4,
            block_size=8,
            num_blocks=64,
            enable_cuda_graph=enable_cuda_graph,
        ).start()
        try:
            pends = [
                engine.submit(BatchRequest(p, max_new_tokens=m, eos_token_ids=()))
                for p, m in zip(prompts, max_new, strict=True)
            ]
            return [p.result(timeout=60) for p in pends]
        finally:
            engine.stop()

    assert run(True) == run(False)
