"""PagedAttention-style KV block allocator + prefix cache (T3 / T4).

``BlockManager`` owns a fixed-size pool of physical block ids (a ``free`` deque +
a ``used`` set) and hands them out to fill each sequence's **block table**
(logical block index → physical ``block_id``). It is the address layer of
PagedAttention: a sequence's KV need not be contiguous, so short sequences no
longer reserve a worst-case contiguous slab (the T2 waste) and there is no
external fragmentation — every block is the same size, freed blocks go straight
back to the pool.

Pure Python, **no torch**: this is the allocator (block *ids* only). The physical
K/V tensors those ids index into live in ``PagedKVPool`` (torch).

**Prefix caching (T4, R1 §4.3).** Sequences that share a prompt prefix (a system
prompt / few-shot preamble) share the *same physical blocks* for it. Each **full**
block is content-addressed by a **chained hash** — ``hash(parent_block_hash,
block_token_ids)`` — so a hit guarantees the entire preceding token sequence
matches, not just this block's 16 tokens (the chain is what makes a block hash a
*prefix* hash). ``register_full_blocks`` publishes a sequence's full blocks into
``hash_to_block_id`` after their KV is computed; ``match_prefix`` looks a new
prompt's blocks up and **reuses** the hits (``ref_count += 1`` — "touch"), so the
prefix's KV is neither reallocated nor recomputed. Only full blocks are cached
(a partial last block's contents aren't final); the hash chain and the semantics
were verified against vLLM's ``design/prefix_caching`` + nano-vLLM before writing
(ADR-0006 API-verification rule).

A block whose ``ref_count`` falls to 0 returns to the free pool but **keeps its
hash** — it is *cached-but-free* and can still be re-hit (resurrected) until it is
actually reused for a new allocation, at which point ``_take`` **evicts** it (LRU:
freed blocks append to the tail, allocation pops the head). This is the
eviction ↔ prefix-block interaction PagedAttention has to get right.
"""

from __future__ import annotations

from collections import deque


class Block:
    """A fixed-length slab of KV slots, identified by ``block_id``.

    ``ref_count`` is how many sequences' block tables point at this block: 0
    (free) or 1 (owned by one sequence) at T3; prefix caching (T4) is what makes
    ``ref_count > 1`` — multiple sequences sharing a prompt prefix. ``hash`` /
    ``token_ids`` are set only while the block is **registered** in the prefix
    cache (it is full and content-addressable); they are cleared on eviction.
    ``hash is None`` means "not a cache entry".
    """

    __slots__ = ("block_id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None
        self.token_ids: tuple[int, ...] = ()


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
        # Prefix cache: content-hash of a full block → the physical block holding
        # it. A mapped block may be ``used`` (ref>0) or cached-but-free (ref==0).
        self.hash_to_block_id: dict[int, int] = {}

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
        (admission) so this only fires on a logic error. This is the no-prefix
        path; the prefix-aware path is ``match_prefix`` + ``allocate_new``.
        """
        need = self.blocks_for(num_tokens)
        if need > len(self.free_block_ids):
            raise ValueError(
                f"cannot allocate {need} blocks; only {len(self.free_block_ids)} free"
            )
        return self.allocate_new(need)

    def allocate_new(self, num_blocks: int) -> list[int]:
        """Take ``num_blocks`` fresh blocks from the pool (evicting as needed).

        Each block comes off the free-queue head; if that block is a stale cache
        entry (cached-but-free) it is evicted first (``_take``). Used for the
        *non-reused* tail after ``match_prefix`` has claimed the shared prefix.
        """
        if num_blocks > len(self.free_block_ids):
            raise ValueError(
                f"cannot allocate {num_blocks} blocks; "
                f"only {len(self.free_block_ids)} free"
            )
        return [self._take() for _ in range(num_blocks)]

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
        """Return every block in ``block_table`` to the pool (ref-counted).

        A block whose ``ref_count`` reaches 0 goes back on the free queue but
        **keeps its cache hash** — it is cached-but-free and can be re-hit by a
        later ``match_prefix`` until ``_take`` actually reuses (and evicts) it.
        """
        for block_id in block_table:
            block = self.blocks[block_id]
            if block.ref_count > 0:
                block.ref_count -= 1
            if block.ref_count == 0 and block_id in self.used:
                self.used.remove(block_id)
                self.free_block_ids.append(block_id)  # tail = most-recently-freed

    # --- prefix caching (T4) -----------------------------------------------

    def _block_hash(self, parent_hash: int | None, chunk: tuple[int, ...]) -> int:
        """Chained content hash of one full block (R1 §4.3, vLLM ``hash_block_tokens``).

        Folding ``parent_hash`` in makes this a hash of the whole prefix up to and
        including ``chunk``, so a collision would require the entire preceding
        token stream to collide — a block hash is a *prefix* hash. ``token_ids``
        equality is still checked on lookup as a belt-and-braces collision guard.
        """
        return hash((parent_hash, chunk))

    def _iter_block_hashes(
        self, token_ids: list[int], max_blocks: int
    ) -> list[tuple[int, tuple[int, ...], int]]:
        """``(block_index, chunk, block_hash)`` for the first ``max_blocks`` blocks.

        The single source of the prefix-hash **chain** (each block's hash folds in
        its predecessor's), so lookup (``match_prefix`` / ``blocks_needed_with_prefix``)
        and publish (``register_full_blocks``) can never disagree on how a prefix
        is keyed. Returns a materialised list (block counts are tiny) to keep the
        callers plain loops.
        """
        out: list[tuple[int, tuple[int, ...], int]] = []
        parent: int | None = None
        for i in range(max_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            h = self._block_hash(parent, chunk)
            out.append((i, chunk, h))
            parent = h
        return out

    def _cache_hit(self, chunk: tuple[int, ...], block_hash: int) -> int | None:
        """The block id caching exactly ``chunk`` under ``block_hash``, else ``None``.

        The ``token_ids`` equality check is the belt-and-braces guard against an
        (astronomically unlikely) hash collision — a mismatch is treated as a miss.
        """
        block_id = self.hash_to_block_id.get(block_hash)
        if block_id is None or self.blocks[block_id].token_ids != chunk:
            return None
        return block_id

    def _max_reuse_blocks(self, num_tokens: int) -> int:
        """Whole blocks reusable while leaving ≥1 query token for the forward.

        Capping at ``num_tokens - 1`` (rounded down to a block) guarantees a
        fully-cached prompt still forwards one token's worth of compute to produce
        its first decode logits — a prefill with zero query tokens has no output.
        """
        return (num_tokens - 1) // self.block_size

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest cached full-block prefix of ``token_ids`` (reused, ref-counted).

        Walks the prompt's full blocks in order, stopping at the first miss (a
        prefix must be contiguous). Every hit is **touched** (``ref_count += 1``);
        a hit that was cached-but-free is pulled back out of the free queue. The
        match is capped by ``_max_reuse_blocks`` so at least one query token always
        remains.

        Returns ``(reused_block_ids, num_cached_tokens)``; both empty/0 when
        nothing matches (the no-shared-prefix no-op).
        """
        reused: list[int] = []
        for _i, chunk, h in self._iter_block_hashes(
            token_ids, self._max_reuse_blocks(len(token_ids))
        ):
            block_id = self._cache_hit(chunk, h)
            if block_id is None:
                break
            block = self.blocks[block_id]
            if block.ref_count == 0:  # cached-but-free: resurrect it
                self.free_block_ids.remove(block_id)
                self.used.add(block_id)
            block.ref_count += 1
            reused.append(block_id)
        return reused, len(reused) * self.block_size

    def blocks_needed_with_prefix(self, token_ids: list[int]) -> int:
        """Free blocks a prefix-aware prefill of ``token_ids`` consumes (no mutation).

        The admission gate: ``total_blocks - (matched prefix blocks already
        resident)``. A matched block that another live sequence still holds
        (``ref_count > 0``) costs zero free blocks — the prefill just refs it up —
        so a shared prefix lets the pool admit a request its whole-prompt block
        count wouldn't fit. A matched-but-*free* block is not credited: resurrecting
        it still consumes a free-queue slot (it is counted in ``num_free_blocks``).
        Pure peek — no ref changes — so it is safe to call before deciding to admit.
        """
        matched_used = 0
        for _i, chunk, h in self._iter_block_hashes(
            token_ids, self._max_reuse_blocks(len(token_ids))
        ):
            block_id = self._cache_hit(chunk, h)
            if block_id is None:
                break
            if self.blocks[block_id].ref_count > 0:
                matched_used += 1
        return self.blocks_for(len(token_ids)) - matched_used

    def register_full_blocks(
        self, block_table: list[int], token_ids: list[int]
    ) -> None:
        """Publish this sequence's full blocks into the prefix cache.

        Called after a prefill has computed KV for all of ``token_ids``: every
        block that is completely full (all ``block_size`` slots written) becomes
        content-addressable so a later shared-prefix request can reuse it. Blocks
        already registered (e.g. ones this sequence itself reused) are skipped;
        the partial last block (if any) is never cached — its contents aren't
        final. The parent-hash chain is rebuilt from block 0 so newly-registered
        blocks chain onto reused ones seamlessly.
        """
        for i, chunk, h in self._iter_block_hashes(
            token_ids, len(token_ids) // self.block_size
        ):
            block_id = block_table[i]
            block = self.blocks[block_id]
            # Only register a not-yet-cached block into a not-yet-taken hash slot.
            # A duplicate (two blocks computed the same prefix before either
            # registered) keeps the first winner; the loser stays uncached and is
            # freed normally.
            if block.hash is None and h not in self.hash_to_block_id:
                block.hash = h
                block.token_ids = chunk
                self.hash_to_block_id[h] = block_id

    def _take(self) -> int:
        """Pop the free-queue head, evicting it from the prefix cache if stale."""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        if block.hash is not None:  # evict a cached-but-free block (LRU: oldest)
            self.hash_to_block_id.pop(block.hash, None)
            block.hash = None
            block.token_ids = ()
        self.used.add(block_id)
        block.ref_count = 1
        return block_id
