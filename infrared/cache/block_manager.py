"""PagedAttention-style KV block allocator (T3).

``BlockManager`` owns a fixed-size pool of physical block ids (a ``free`` deque +
a ``used`` set) and hands them out to fill each sequence's **block table**
(logical block index → physical ``block_id``). It is the address layer of
PagedAttention: a sequence's KV need not be contiguous, so short sequences no
longer reserve a worst-case contiguous slab (the T2 waste) and there is no
external fragmentation — every block is the same size, freed blocks go straight
back to the pool.

Pure Python, **no torch**: this is the allocator (block *ids* only). The physical
K/V tensors those ids index into live in ``PagedKVPool`` (torch). Ref counting is
carried on ``Block`` but stays 1-per-block at T3; prefix-cache sharing
(``hash → block_id``, ``ref_count > 1``) is deferred to T4 (R1 §4.3).
"""

from __future__ import annotations

from collections import deque


class Block:
    """A fixed-length slab of KV slots, identified by ``block_id``.

    ``ref_count`` is how many sequences' block tables point at this block; at T3
    it is always 0 (free) or 1 (owned by one sequence). Prefix caching (T4) is
    what makes ``ref_count > 1`` — multiple sequences sharing a prompt prefix.
    """

    __slots__ = ("block_id", "ref_count")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0


class BlockManager:
    """Fixed pool of ``num_blocks`` blocks of ``block_size`` token-slots each."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return len(self.used)

    def blocks_for(self, num_tokens: int) -> int:
        """How many blocks ``num_tokens`` tokens occupy (ceil-divide)."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """True if the pool can back a fresh sequence of ``num_tokens`` tokens."""
        return self.blocks_for(num_tokens) <= len(self.free_block_ids)

    def allocate(self, num_tokens: int) -> list[int]:
        """Reserve blocks for ``num_tokens`` and return the new block table.

        Raises if the pool can't satisfy it — callers gate with ``can_allocate``
        (admission) so this only fires on a logic error.
        """
        need = self.blocks_for(num_tokens)
        if need > len(self.free_block_ids):
            raise ValueError(
                f"cannot allocate {need} blocks; only {len(self.free_block_ids)} free"
            )
        table = [self._take() for _ in range(need)]
        return table

    def can_append(self, block_table: list[int], cur_len: int) -> bool:
        """True if appending one token at position ``cur_len`` can be housed.

        A new block is needed only when the current blocks are exactly full
        (``cur_len`` is a block boundary); otherwise the token fits in the last
        block's next slot for free.
        """
        if cur_len % self.block_size != 0:
            return True
        return len(self.free_block_ids) > 0

    def append(self, block_table: list[int], cur_len: int) -> None:
        """Grow ``block_table`` by one block iff position ``cur_len`` needs it.

        Mutates ``block_table`` in place (the sequence owns the list). No-op when
        the token fits in the last block. Raises if a block is needed but none is
        free — callers ensure capacity (append-time preemption) first.
        """
        if cur_len % self.block_size != 0:
            return
        if not self.free_block_ids:
            raise ValueError("cannot append: no free blocks (preempt first)")
        block_table.append(self._take())

    def free(self, block_table: list[int]) -> None:
        """Return every block in ``block_table`` to the pool (ref-counted)."""
        for block_id in block_table:
            block = self.blocks[block_id]
            if block.ref_count > 0:
                block.ref_count -= 1
            if block.ref_count == 0 and block_id in self.used:
                self.used.remove(block_id)
                self.free_block_ids.append(block_id)

    def _take(self) -> int:
        block_id = self.free_block_ids.popleft()
        self.used.add(block_id)
        self.blocks[block_id].ref_count = 1
        return block_id
