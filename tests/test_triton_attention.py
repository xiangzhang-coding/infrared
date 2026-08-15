"""T4c — the fused Triton paged-attention kernel + its CPU-verifiable seams.

The Triton kernel is **GPU-only** (triton ships Linux wheels; `tl.dot` wants a
CUDA device), so on this Mac/CPU box it can never run — the real kernel-vs-naive
parity check is CUDA-gated and `skip`s here, running on the AutoDL/4090 box (the
ticket's cross-platform rule). What *is* checkable on CPU, and pinned here:

1. **The kernel's numerics, without a GPU.** ``paged_attention_blockwise_reference``
   is a pure-torch twin of the kernel's block-walk online softmax (running
   max/sum, rescale-accumulate). Proving it equals the one-shot ``masked_attention``
   for many key-block sizes (incl. non-divisors), GQA, and padded histories
   validates the *algorithm* the Triton kernel transliterates — the strongest
   correctness signal obtainable off-GPU, and the diff-oracle for the GPU port.
2. **The dispatch.** On CPU, ``paged_attention`` must select the naive path and
   reproduce it bit-for-bit — the fallback the no-GPU tests + CI exercise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infrared.cache.paged_kv_cache import PagedContext, PagedKVPool  # noqa: E402
from infrared.model.layers import masked_attention, repeat_kv  # noqa: E402
from infrared.model.triton_attention import (  # noqa: E402
    paged_attention,
    paged_attention_blockwise_reference,
)


def _gather_case(*, b, sq, t, h_kv, n_rep, seed=0):
    """Random q + a gathered/repeated history + a causal-ish additive mask.

    Returns ``(q, k_all, v_all, mask, scale)`` in ``masked_attention``'s contract:
    q ``[B, Sq, H, D]``; k_all/v_all ``[B, T, H, D]`` (already repeated to H heads);
    mask ``[B, 1, Sq, T]`` additive. Every query row attends at least one key (no
    fully-masked rows), so softmax is well-defined for both implementations.
    """
    torch.manual_seed(seed)
    h, d = h_kv * n_rep, 8
    q = torch.randn(b, sq, h, d)
    k_kv = torch.randn(b, t, h_kv, d)
    v_kv = torch.randn(b, t, h_kv, d)
    k_all, v_all = repeat_kv(k_kv, n_rep), repeat_kv(v_kv, n_rep)
    # Each query row i (absolute position t-sq+i) attends keys [0 .. t-sq+i].
    q_pos = torch.arange(t - sq, t)[:, None]
    k_pos = torch.arange(t)[None, :]
    allowed = k_pos <= q_pos  # [Sq, T]
    mask = torch.zeros(b, 1, sq, t)
    mask = mask.masked_fill(~allowed[None, None], torch.finfo(torch.float32).min)
    return q, k_all, v_all, mask, d**-0.5


# --- 1. the online-softmax reference == one-shot masked_attention -----------


@pytest.mark.parametrize("block_n", [1, 2, 3, 4, 7, 8, 100])
def test_blockwise_reference_matches_masked_attention(block_n: int) -> None:
    """Streaming the key axis in blocks (any size) must equal the one-shot softmax."""
    q, k_all, v_all, mask, scale = _gather_case(b=2, sq=5, t=8, h_kv=2, n_rep=2)
    ref = paged_attention_blockwise_reference(q, k_all, v_all, mask, scale, block_n)
    one_shot = masked_attention(q, k_all, v_all, mask, scale)
    assert torch.allclose(ref, one_shot, atol=1e-5), f"block_n={block_n} diverged"


def test_blockwise_reference_decode_single_query_row() -> None:
    """Decode shape (Sq=1, the mat-vec case) still matches the one-shot softmax."""
    q, k_all, v_all, mask, scale = _gather_case(b=3, sq=1, t=16, h_kv=2, n_rep=3)
    ref = paged_attention_blockwise_reference(q, k_all, v_all, mask, scale, block_n=4)
    assert torch.allclose(
        ref, masked_attention(q, k_all, v_all, mask, scale), atol=1e-5
    )


def test_blockwise_reference_mha_no_gqa() -> None:
    """n_rep=1 (no GQA repeat) is the plain-MHA special case."""
    q, k_all, v_all, mask, scale = _gather_case(b=1, sq=6, t=6, h_kv=4, n_rep=1)
    ref = paged_attention_blockwise_reference(q, k_all, v_all, mask, scale, block_n=3)
    assert torch.allclose(
        ref, masked_attention(q, k_all, v_all, mask, scale), atol=1e-5
    )


# --- 2. the dispatcher falls back to the naive path on CPU ------------------


def _paged_ctx(k_kv, v_kv, *, use_triton):
    """A single-block pool holding this step's K/V, gathered back in order.

    Writes the ``[B*Sq, H_kv, D]`` step K/V into flat slots ``0..N-1`` and gathers
    each query row's own single slot — the minimal PagedContext that lets
    ``paged_attention`` run end to end on CPU (B=1, so slots are contiguous).
    """
    b, sq, h_kv, d = k_kv.shape
    n = b * sq
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=1,
        block_size=n,
        num_kv_heads=h_kv,
        head_dim=d,
        dtype=torch.float32,
    )
    write_slots = torch.arange(n)
    gather_slots = torch.arange(n).reshape(b, sq)
    return pool, PagedContext(pool, write_slots, gather_slots, use_triton=use_triton)


def test_paged_attention_falls_back_to_naive_on_cpu() -> None:
    """With use_triton=True but no CUDA, dispatch must run the naive path exactly."""
    b, sq, h_kv, n_rep, d = 1, 4, 2, 2, 8
    torch.manual_seed(0)
    q = torch.randn(b, sq, h_kv * n_rep, d)
    k = torch.randn(b, sq, h_kv, d)
    v = torch.randn(b, sq, h_kv, d)
    # Causal mask over this step's own tokens (T == Sq here).
    q_pos = torch.arange(sq)[:, None]
    allowed = torch.arange(sq)[None, :] <= q_pos
    mask = torch.zeros(b, 1, sq, sq).masked_fill(
        ~allowed[None, None], torch.finfo(torch.float32).min
    )
    scale = d**-0.5

    _, ctx_want_triton = _paged_ctx(k, v, use_triton=True)
    out = paged_attention(
        q, k, v, ctx_want_triton, mask, layer_idx=0, scale=scale, n_rep=n_rep
    )
    # Reference: naive gather + repeat + masked_attention over the same history.
    expected = masked_attention(
        q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), mask, scale
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_dispatch_matches_regardless_of_use_triton_on_cpu() -> None:
    """use_triton True/False are identical on CPU (both take the fallback)."""
    b, sq, h_kv, n_rep, d = 1, 3, 2, 1, 8
    torch.manual_seed(1)
    q = torch.randn(b, sq, h_kv * n_rep, d)
    k, v = torch.randn(b, sq, h_kv, d), torch.randn(b, sq, h_kv, d)
    mask = torch.zeros(b, 1, sq, sq)
    scale = d**-0.5

    _, off = _paged_ctx(k, v, use_triton=False)
    out_off = paged_attention(q, k, v, off, mask, 0, scale, n_rep)
    _, on = _paged_ctx(k, v, use_triton=True)
    out_on = paged_attention(q, k, v, on, mask, 0, scale, n_rep)
    assert torch.allclose(out_off, out_on, atol=1e-6)


# --- 3. the real Triton kernel vs naive — CUDA-only (skips on Mac/CPU) ------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel is GPU-only")
def test_triton_kernel_matches_naive_on_cuda() -> None:
    """On CUDA: the fused kernel output must match the naive paged path (parity).

    This is the ticket's numerical-consistency gate; it only runs where Triton
    can compile (AutoDL/4090). head_dim=64 satisfies ``tl.dot``'s ≥16 constraint.
    """
    dev = "cuda"
    b, sq, h_kv, n_rep, d = 2, 4, 2, 2, 64
    torch.manual_seed(0)
    q = torch.randn(b, sq, h_kv * n_rep, d, device=dev)
    k = torch.randn(b, sq, h_kv, d, device=dev)
    v = torch.randn(b, sq, h_kv, d, device=dev)
    q_pos = torch.arange(sq, device=dev)[:, None]
    allowed = torch.arange(sq, device=dev)[None, :] <= q_pos
    mask = torch.zeros(b, 1, sq, sq, device=dev).masked_fill(
        ~allowed[None, None], torch.finfo(torch.float32).min
    )
    scale = d**-0.5

    n = b * sq
    pool = PagedKVPool(1, 1, n, h_kv, d, torch.float32, device=dev)
    write_slots = torch.arange(n, device=dev)
    gather_slots = torch.arange(n, device=dev).reshape(b, sq)
    triton_ctx = PagedContext(pool, write_slots, gather_slots, use_triton=True)
    got = paged_attention(q, k, v, triton_ctx, mask, 0, scale, n_rep)

    pool_ref = PagedKVPool(1, 1, n, h_kv, d, torch.float32, device=dev)
    naive_ctx = PagedContext(
        pool_ref, write_slots, gather_slots.clone(), use_triton=False
    )
    want = paged_attention(q, k, v, naive_ctx, mask, 0, scale, n_rep)
    assert torch.allclose(got, want, atol=1e-2, rtol=1e-2)
