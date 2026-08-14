"""PagedAttention-style KV block allocator (T3 — stub).

``BlockManager`` owns a fixed-size block pool (free deque + used set) and fills
each sequence's ``block_table`` (logical block → physical block_id). It exposes
``can_allocate`` / ``allocate`` (prefill), ``can_append`` / ``may_append``
(decode), and ``deallocate`` (ref-counted return). Prefix caching (hash→block)
is deferred to T4 (R1 §4.3).
"""

from __future__ import annotations

_T3 = "not implemented until T3 — see docs/spec/0001 and R1 blueprint §4.3"


class Block:
    """A fixed-length slab of KV slots: {block_id, ref_count, hash, token_ids}."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T3)


class BlockManager:
    """Paged allocator: free deque + used set + ref-counted (de)allocation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_T3)
