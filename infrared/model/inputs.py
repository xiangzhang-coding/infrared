"""Forward-input preparation: positions + additive attention masks (batch-first).

The model is a pure batched executor — it consumes precomputed ``positions`` and
an additive ``mask`` (this is R1's ``prepare_inputs`` split). Both the
single-request path (``Qwen2ForCausalLM.forward_single``) and the T1 static-batch
runner build their inputs here.

Left-padding convention (T1): a batch padded to width ``P`` puts ``pad_i = P -
len_i`` pad tokens on the **left**, so every sequence's real tokens end at the
same column and decode advances in lockstep. Columns are the shared time axis
(KV slot = column); a key column ``k`` is valid for sequence ``i`` only when
``k >= pad_i``. RoPE positions are per-sequence (``column - pad_i``) so real
tokens still see positions ``0..len_i-1``.
"""

from __future__ import annotations

import torch


def build_positions(
    pad_lens: list[int], start_col: int, length: int, device: torch.device | str
) -> torch.Tensor:
    """Per-sequence RoPE positions for columns ``[start_col, start_col+length)``.

    Returns ``[B, length]``; pad columns clamp to 0 (they are masked out anyway).
    """
    cols = torch.arange(start_col, start_col + length, device=device)  # [S]
    pad = torch.tensor(pad_lens, device=device)[:, None]  # [B, 1]
    return (cols[None, :] - pad).clamp(min=0)


def build_attention_mask(
    pad_lens: list[int],
    q_start_col: int,
    q_len: int,
    total_k: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Additive mask ``[B, 1, q_len, total_k]`` (0 allowed, ``finfo.min`` blocked).

    Combines causality (key column ``<=`` query column) with left-pad validity
    (key column ``>= pad_i``). Broadcasts over heads.
    """
    q_cols = torch.arange(q_start_col, q_start_col + q_len, device=device)  # [Sq]
    k_cols = torch.arange(total_k, device=device)  # [Sk]
    causal = k_cols[None, :] <= q_cols[:, None]  # [Sq, Sk]
    pad = torch.tensor(pad_lens, device=device)[:, None]  # [B, 1]
    key_valid = k_cols[None, :] >= pad  # [B, Sk]
    allowed = causal[None, :, :] & key_valid[:, None, :]  # [B, Sq, Sk]

    mask = torch.zeros(len(pad_lens), 1, q_len, total_k, dtype=dtype, device=device)
    return mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)


def build_varlen_mask(
    q_seq_ids: torch.Tensor,
    q_pos: torch.Tensor,
    k_seq_ids: torch.Tensor,
    k_pos: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Additive mask ``[1, 1, Q, K]`` for a **flattened** multi-sequence frame (T4).

    The T4 mixed prefill+decode step packs every scheduled query token of every
    sequence into one row (``B=1``, ``Sq=Q``) and gathers the union of their KV
    slots into one key axis (``Sk=K``). A query token may attend a key iff they
    belong to the **same sequence** and the key is **causally at or before** it —
    which, per sequence, is exactly the standalone causal mask, so the flattened
    forward is token-for-token identical to running each sequence alone (Seam A).

    ``q_seq_ids`` / ``q_pos`` are ``[Q]`` (each query token's sequence id + absolute
    position); ``k_seq_ids`` / ``k_pos`` are ``[K]`` (same, per gathered key). Unlike
    ``build_attention_mask`` (one contiguous q/k range + a left-pad offset) this
    expresses arbitrary per-token (seq, pos), which is what the packed frame needs.
    """
    same_seq = q_seq_ids[:, None] == k_seq_ids[None, :]  # [Q, K]
    causal = k_pos[None, :] <= q_pos[:, None]  # [Q, K]
    allowed = same_seq & causal
    mask = torch.zeros(
        1, 1, q_seq_ids.shape[0], k_seq_ids.shape[0], dtype=dtype, device=device
    )
    return mask.masked_fill(~allowed[None, None, :, :], torch.finfo(dtype).min)
