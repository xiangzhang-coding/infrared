"""T4c — the self-written Triton paged-attention kernel (GPU-only, R3 #10).

Replaces T3's naive PyTorch paged path (`PagedKVPool.gather` → materialize the
full ``[B, T, H, D]`` history → ``masked_attention`` over a full score matrix)
with **one fused Triton kernel** that walks each query row's slot ids, gathers K/V
tiles by masked pointer loads, and runs a numerically-stable **online softmax** —
never materializing the gathered history or the full scores. This is the ticket's
"招牌高效 / write-your-own-kernel" tier (R1 §5, §8).

**Learned the *shape* from vLLM / FlashAttention-2, rewrote from scratch with
teaching comments — no copy-paste (ADR-0004).** Every Triton primitive used here
is one verified in the R3 findings (`docs/research/t4-triton-cudagraph-api.md` §5).

**Cross-platform rule (the ticket's 铁律).** Triton ships **Linux wheels only** and
``tl.dot`` needs a CUDA device, so:

- ``triton`` is **lazy-imported** inside the GPU path — never at module top level —
  so ``infrared`` imports cleanly on macOS/CPU and in torch-only CI.
- ``paged_attention`` dispatches to the fused kernel only when the tensors are on
  CUDA *and* triton is importable *and* the caller opted in
  (``PagedContext.use_triton``); otherwise it runs the **naive PyTorch paged path**
  (the T3 fallback), the correctness oracle the no-GPU tests exercise and the kernel
  is diffed against.

**Honesty (ADR-0006).** This box has no GPU, so the ``@triton.jit`` kernel below was
**not compiled or numerically checked here** — its parity vs the naive path is
CUDA-gated (`tests/test_triton_attention.py::...on_cuda`) and validated on AutoDL.
What *is* verified on CPU is the algorithm's numerics, via
``paged_attention_blockwise_reference`` (the kernel's pure-torch twin) proven equal
to ``masked_attention``. Read the resolved triton version off the box
(``pip show triton``) before pinning — do not assert it here.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from infrared.cache.paged_kv_cache import PagedContext

# 1/ln2: multiplying the score scale by this lets the kernel use hardware ``exp2``
# in place of ``exp`` (the FlashAttention/Triton-tutorial-06 trick) — mathematically
# identical, since 2^(x·log2e) = e^x. The torch reference below uses plain ``exp``.
_LOG2E = 1.4426950408889634


# --- path selection ---------------------------------------------------------


@cache
def _triton_available() -> bool:
    """Whether ``triton`` (+ ``triton.language``) can be imported. Cached once.

    Kept out of module top level so importing ``infrared`` never hard-depends on
    triton (Linux-wheel-only). The result is stable for a process, hence ``cache``.
    """
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "no triton here"
        return False
    return True


def _should_use_triton(q: torch.Tensor, want: bool) -> bool:
    """Gate the fused kernel on opt-in + a CUDA tensor + importable triton + head_dim.

    ``tl.dot`` requires the contraction dim (``head_dim``) to be a multiple of 16,
    so tiny toy models (head_dim 8) fall back even on CUDA — the naive path is
    always correct. Real Qwen2.5 head_dim is 64, so the kernel engages there.
    """
    return want and q.is_cuda and q.shape[-1] % 16 == 0 and _triton_available()


# --- the dispatcher (what Attention.forward calls) --------------------------


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedContext,
    mask: torch.Tensor,
    layer_idx: int,
    scale: float,
    n_rep: int,
) -> torch.Tensor:
    """Write this step's K/V into the paged pool, then attend over gathered history.

    Shapes: ``q`` ``[B, Sq, H, D]`` (already RoPE'd); ``k``/``v`` ``[B, Sq, H_kv, D]``
    (this step's new, RoPE'd K/V); ``mask`` ``[B, 1, Sq, T]`` additive (0 / ``-inf``).
    Returns ``[B, Sq, H, D]``. The K/V scatter into the pool happens **here, once**,
    before either read path branches — so ``_paged_attention_naive`` /
    ``_paged_attention_triton`` only *read* the pool and differ solely in how they
    attend. Dispatches to the fused Triton kernel on CUDA (opt-in), else the naive
    PyTorch paged path — the two are numerically equivalent (Seam A).
    """
    h, d = k.shape[-2], k.shape[-1]  # num_kv_heads, head_dim
    paged.pool.write(
        layer_idx, k.reshape(-1, h, d), v.reshape(-1, h, d), paged.write_slots
    )
    if _should_use_triton(q, paged.use_triton):
        return _paged_attention_triton(q, paged, mask, layer_idx, scale, n_rep)
    return _paged_attention_naive(q, paged, mask, layer_idx, scale, n_rep)


def _paged_attention_naive(
    q: torch.Tensor,
    paged: PagedContext,
    mask: torch.Tensor,
    layer_idx: int,
    scale: float,
    n_rep: int,
) -> torch.Tensor:
    """The T3 read path, unchanged: gather full history, masked one-shot softmax.

    Assumes this step's K/V is already scattered into the pool (done by the
    ``paged_attention`` dispatcher). Imported lazily to avoid a
    layers↔triton_attention import cycle (``layers`` imports ``paged_attention``).
    """
    from infrared.model.layers import masked_attention, repeat_kv

    k_all, v_all = paged.pool.gather(layer_idx, paged.gather_slots)
    k_all = repeat_kv(k_all, n_rep)
    v_all = repeat_kv(v_all, n_rep)
    return masked_attention(q, k_all, v_all, mask, scale)


# --- the pure-torch twin of the kernel's online softmax (CPU-verifiable) ----


def paged_attention_blockwise_reference(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    block_n: int | None = None,
) -> torch.Tensor:
    """FlashAttention-style block-walk online softmax, in plain torch (the twin).

    Same contract as ``masked_attention``: ``q`` ``[B, Sq, H, D]``; ``k_all``/``v_all``
    ``[B, T, H, D]`` (already repeated to H heads); ``mask`` ``[B, 1, Sq, T]`` additive.
    Streams the key axis in blocks of ``block_n`` keeping per-query running max ``m``
    and sum ``l`` and a rescaled accumulator — the *exact* numerics the Triton kernel
    runs (§2.3 of R3), so proving this equals the one-shot ``masked_attention`` on CPU
    validates the kernel's algorithm before it ever compiles on a GPU. Uses natural
    ``exp`` (the kernel's ``exp2`` with the ``_LOG2E`` scale is the identical HW form).
    """
    b, sq, h, d = q.shape
    t = k_all.shape[1]
    block_n = block_n or t
    # Work in SDPA's [B, H, S, *] layout so the matmuls broadcast over heads.
    qh = q.permute(0, 2, 1, 3).to(torch.float32)  # [B, H, Sq, D]
    kh = k_all.permute(0, 2, 1, 3).to(torch.float32)  # [B, H, T, D]
    vh = v_all.permute(0, 2, 1, 3).to(torch.float32)

    neg_inf = float("-inf")
    m_i = torch.full((b, h, sq, 1), neg_inf)
    l_i = torch.zeros((b, h, sq, 1))
    acc = torch.zeros((b, h, sq, d))
    for j in range(0, t, block_n):
        k_blk = kh[:, :, j : j + block_n]  # [B, H, n, D]
        v_blk = vh[:, :, j : j + block_n]
        m_blk = mask[:, :, :, j : j + block_n]  # [B, 1, Sq, n] broadcasts over H
        qk = (qh @ k_blk.transpose(-1, -2)) * scale + m_blk  # [B, H, Sq, n]
        m_ij = torch.maximum(m_i, qk.max(dim=-1, keepdim=True).values)  # new max
        p = torch.exp(qk - m_ij)  # rescaled block probabilities
        alpha = torch.exp(m_i - m_ij)  # correction factor for the old state
        l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + p @ v_blk  # rescale the accumulator, add this block
        m_i = m_ij
    out = acc / l_i  # final normalize
    return out.permute(0, 2, 1, 3).to(q.dtype)  # back to [B, Sq, H, D]


# --- the fused Triton kernel (GPU-only; not compiled/tested on this CPU box) -


@cache
def _paged_attn_kernel():
    """Build + cache the ``@triton.jit`` kernel. Lazy so triton stays Linux/GPU-only.

    One program per ``(batch, query-head, query-block)``. It loads a ``[BLOCK_M, D]``
    query tile, then walks the key axis in ``BLOCK_N`` chunks: for each chunk it reads
    the slot ids from ``gather_slots`` (masked ``tl.load``), gathers the K/V tiles
    non-contiguously from the flat pool at those slots (GQA folds ``head → head //
    N_REP`` into the K/V index), scores with ``tl.dot``, adds the precomputed additive
    mask, and updates the online-softmax running state (``m_i``/``l_i``/``acc``). K is
    loaded transposed (``[D, BLOCK_N]``) so ``tl.dot`` needs no ``tl.trans``.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        q_ptr,
        kc_ptr,
        vc_ptr,
        o_ptr,
        g_ptr,
        m_ptr,
        sqb,
        sqs,
        sqh,
        sqd,  # q strides [B, Sq, H, D]
        kcs,
        kch,
        kcd,  # k/v cache slot strides [num_slots, H_kv, D]
        ob,
        os_,
        oh,
        od,  # out strides [B, Sq, H, D]
        gb,
        gt,  # gather_slots strides [B, T]
        mb,
        ms,
        mt,  # mask strides [B, Sq, T] (head dim squeezed — broadcasts)
        scale,
        SQ: tl.constexpr,
        T: tl.constexpr,
        N_REP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)
        kv_h = pid_h // N_REP  # GQA: this query head reads its KV group's head

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] query rows
        offs_d = tl.arange(0, HEAD_DIM)  # [D]
        q_valid = offs_m < SQ

        # Load this program's query tile [BLOCK_M, D] (padded rows masked to 0).
        q = tl.load(
            q_ptr
            + pid_b * sqb
            + offs_m[:, None] * sqs
            + pid_h * sqh
            + offs_d[None, :] * sqd,
            mask=q_valid[:, None],
            other=0.0,
        )
        qk_scale = scale * _LOG2E  # fold 1/ln2 in so exp2 replaces exp

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        for j0 in range(0, T, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)  # [BLOCK_N] key positions
            n_valid = offs_n < T
            # slot id of each key position for this sequence (padding -> 0, masked).
            slots = tl.load(g_ptr + pid_b * gb + offs_n * gt, mask=n_valid, other=0)

            # Gather K transposed ([D, BLOCK_N]) and V ([BLOCK_N, D]) from the flat
            # pool at the (non-contiguous) slots — the paged gather, fused in.
            kT = tl.load(
                kc_ptr + slots[None, :] * kcs + kv_h * kch + offs_d[:, None] * kcd,
                mask=n_valid[None, :],
                other=0.0,
            )  # [D, BLOCK_N]
            v = tl.load(
                vc_ptr + slots[:, None] * kcs + kv_h * kch + offs_d[None, :] * kcd,
                mask=n_valid[:, None],
                other=0.0,
            )  # [BLOCK_N, D]

            qk = tl.dot(q, kT) * qk_scale  # [BLOCK_M, BLOCK_N] scores
            # Additive mask (0 allowed / -inf blocked) already encodes causal +
            # padding + varlen same-seq; -inf dominates regardless of log2 units.
            mval = tl.load(
                m_ptr + pid_b * mb + offs_m[:, None] * ms + offs_n[None, :] * mt,
                mask=q_valid[:, None] & n_valid[None, :],
                other=float("-inf"),
            )
            qk = qk + mval

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))  # running max
            p = tl.math.exp2(qk - m_ij[:, None])  # rescaled probs [BLOCK_M, BLOCK_N]
            alpha = tl.math.exp2(m_i - m_ij)  # correction for the old state
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_ij

        # Defensive guard: a query row that saw no valid key (fully-masked row)
        # leaves l_i == 0; floor it to 1 so the normalize is 0/1 == 0 instead of
        # 0/0 == NaN. Real callers never hit this (every query attends ≥ itself,
        # causally), but it keeps the kernel robust to any mask convention.
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / l_safe[:, None]  # final normalize
        tl.store(
            o_ptr
            + pid_b * ob
            + offs_m[:, None] * os_
            + pid_h * oh
            + offs_d[None, :] * od,
            acc.to(o_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )

    return _kernel


def _paged_attention_triton(
    q: torch.Tensor,
    paged: PagedContext,
    mask: torch.Tensor,
    layer_idx: int,
    scale: float,
    n_rep: int,
) -> torch.Tensor:
    """Run the fused gather+attention kernel over the already-scattered pool.

    This step's K/V is scattered by the ``paged_attention`` dispatcher; the scatter
    stays the pool's torch ``index_copy`` (already efficient on CUDA — R3 §2.5's
    ``store_kvcache`` Triton kernel is a documented, low-value extension). The kernel
    reads the flat pool slice for ``layer_idx`` at the precomputed ``gather_slots`` —
    reusing infrared's existing paged seam so one kernel covers prefill / decode /
    mixed via the caller's metadata + mask.
    """
    import triton

    kc = paged.pool.k[layer_idx]  # [num_slots, H_kv, D] contiguous
    vc = paged.pool.v[layer_idx]
    gather_slots = paged.gather_slots  # [B, T] long
    b, sq, heads, d = q.shape  # d == head_dim, passed as the HEAD_DIM constexpr
    t = gather_slots.shape[1]
    m2 = mask.reshape(b, sq, t)  # squeeze the head-broadcast dim -> [B, Sq, T]
    out = torch.empty_like(q)

    block_m, block_n = 16, 64
    grid = (b, heads, triton.cdiv(sq, block_m))
    _paged_attn_kernel()[grid](
        q,
        kc,
        vc,
        out,
        gather_slots,
        m2,
        *q.stride(),
        *kc.stride(),
        *out.stride(),
        *gather_slots.stride(),
        *m2.stride(),
        scale,
        SQ=sq,
        T=t,
        N_REP=n_rep,
        HEAD_DIM=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
