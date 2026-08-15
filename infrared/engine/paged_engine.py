"""Paged continuous-batching engine (T3) + prefix caching (T4) — PagedAttention.

Extends the T2 ``ContinuousBatchEngine`` with a paged KV backend, keeping the
same submit/``Pending`` surface and busy loop. Two things change:

1. **Storage.** Instead of a per-sequence contiguous slab sized for the whole
   worst-case generation (T2's reservation waste), each sequence draws
   fixed-size blocks from a shared ``BlockManager`` pool **on demand** as it
   grows. Freed blocks return to the pool immediately, so more sequences fit in
   the same KV budget and there is no external fragmentation.
2. **Compute.** The decode step is now **batched across the whole running set**
   in one forward — each sequence's ragged history is gathered from its block
   table (``PagedKVPool.gather``), padded, and masked per-sequence. This is the
   throughput lever T2 deferred: one batched matmul instead of a per-sequence
   loop. (The Triton paged-attn kernel that fuses the gather is T4 — R1 §5/§8.)

**Prefix caching (T4, ``enable_prefix_caching``).** When two requests share a
prompt prefix (a system prompt / few-shot preamble), the second reuses the first's
already-computed KV blocks instead of reallocating and recomputing them:
``_prefill`` calls ``BlockManager.match_prefix`` to claim the cached prefix blocks
(ref-counted up) and then forwards **only the un-cached suffix** — a partial
prefill whose queries attend back over the full history. Output stays
bit-identical (the prefix hash chain guarantees the reused KV is exactly this
prompt's prefix; §_prefill). It is a pure no-op when nothing is shared.

When the pool can't grant a block mid-decode, a **recompute preemption** evicts
the newest running sequence (frees its blocks, returns it to the waiting head);
when re-admitted it re-prefills its ``token_ids`` (prompt + tokens generated so
far) — re-hitting any still-cached prefix — so no output is lost, just recomputed.
Greedy output stays token-for-token identical to the T0 oracle: the paged
read/write reproduces each sequence's exact history, and batched decode is just
independent per-sequence attentions masked apart (validated in
tests/test_paged_kv.py).
"""

from __future__ import annotations

import torch

from infrared.cache.block_manager import BlockManager
from infrared.cache.paged_kv_cache import PagedContext, PagedKVPool
from infrared.engine.engine import ContinuousBatchEngine
from infrared.engine.scheduler import MixedPlan, SeqChunk, plan_mixed_step
from infrared.engine.sequence import Sequence, SequenceStatus
from infrared.engine.static_batch import BatchStats
from infrared.model.inputs import (
    build_attention_mask,
    build_positions,
    build_varlen_mask,
)
from infrared.model.qwen2 import Qwen2ForCausalLM


class PagedBatchEngine(ContinuousBatchEngine):
    """Continuous batching over a paged KV pool, with batched decode + preemption."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        max_num_seqs: int = 8,
        block_size: int = 16,
        num_blocks: int = 256,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = False,
        chunk_size: int = 512,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        super().__init__(model, max_num_seqs=max_num_seqs)
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        # Observable reuse evidence (the T4 win): how many physical prefix blocks
        # / prompt tokens were served from cache instead of allocated + computed.
        self.prefix_reused_blocks = 0
        self.prefix_reused_tokens = 0
        # Chunked prefill (T4b): when on, a step mixes prefill chunks + decode within
        # ``token_budget`` tokens. Default the budget to ``max_num_seqs + chunk_size``
        # so a full decode batch (one token each, decodes are never dropped) AND one
        # prefill chunk fit together — otherwise a busy decode set would starve the
        # chunk and the interleave wouldn't fire. ``mixed_steps`` counts steps that
        # actually carried both — structural evidence the interleave happens.
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunk_size = chunk_size
        self.token_budget = max_num_batched_tokens or (max_num_seqs + chunk_size)
        self.mixed_steps = 0
        self.block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.pool = PagedKVPool(
            num_layers=model.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            dtype=model.dtype,
            device=model.device,
        )

    # --- one step: paged schedule -> forward -> postprocess ----------------

    @torch.no_grad()
    def _step(self) -> None:
        if self.enable_chunked_prefill:
            self._step_mixed()
            return
        sched = self.scheduler
        # Prefill admission: a free running slot AND enough free blocks. Admitting
        # only prompt-sized blocks (not the worst-case generation) is what lets
        # more sequences share the pool than T2's contiguous reservation would.
        if sched.waiting and len(sched.running) < self.max_num_seqs:
            seq = sched.waiting[0]
            # Prefix-aware admission: a shared prefix already resident in the pool
            # (held by a live sequence) costs no new blocks, so credit it here —
            # otherwise the whole-prompt block count would defer a request the pool
            # could actually house (the occupancy win, claimed at admission too).
            if self.enable_prefix_caching:
                need = self.block_manager.blocks_needed_with_prefix(seq.token_ids)
            else:
                need = self.block_manager.blocks_for(len(seq.token_ids))
            if self.block_manager.num_free_blocks >= need:
                sched.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                sched.running.append(seq)
                self._prefill(seq)
                return
            if not sched.running:
                # Can't house even one sequence and nothing is running to drain:
                # the pool is too small for this prompt. Fail just this request.
                sched.waiting.popleft()
                self._fail_seq(seq, RuntimeError("KV pool too small for this prompt"))
                return
            # Otherwise fall through: decode the running set to free blocks, then
            # admit on a later step (continuous behaviour under block pressure).

        if sched.running:
            self._decode_step()

    def _prefill(self, seq: Sequence) -> None:
        """Reuse any cached prefix, prefill only the un-cached suffix, sample.

        With prefix caching, a shared prompt prefix (system prompt / few-shot) is
        served from blocks a prior request already computed: ``match_prefix``
        hands back those physical blocks (ref-counted up) and the count of cached
        tokens, so this forward runs over **only** ``token_ids[num_cached:]``. The
        reused prefix's K/V already sits in the pool; the suffix attends back over
        the full ``[0:length]`` history (gather), writes only its own slots, and
        RoPE positions resume at ``num_cached`` — so the result is bit-identical to
        prefilling the whole prompt (Seam A holds; the prefix hash chain guarantees
        the reused KV is exactly this prompt's prefix). The match is capped below
        ``length`` so there is always ≥1 suffix token to produce the first logits.
        Freshly-computed full blocks are then published for downstream reuse.
        """
        if seq.seed is not None and seq.generator is None:
            seq.generator = torch.Generator(device=self.model.device).manual_seed(
                seq.seed
            )
        length = len(seq.token_ids)
        if self.enable_prefix_caching:
            reused, num_cached = self.block_manager.match_prefix(seq.token_ids)
        else:
            reused, num_cached = [], 0
        try:
            total_blocks = self.block_manager.blocks_for(length)
            new_blocks = self.block_manager.allocate_new(total_blocks - len(reused))
            seq.block_table = reused + new_blocks
            q_len = length - num_cached  # only the un-cached suffix is forwarded
            write_slots = self._slots(
                [self._slot(seq.block_table, p) for p in range(num_cached, length)]
            )
            gather_slots = self._slots(
                [self._slot(seq.block_table, p) for p in range(length)]
            ).reshape(1, length)
            positions = build_positions([0], num_cached, q_len, self.model.device)
            mask = build_attention_mask(
                [0], num_cached, q_len, length, self.model.dtype, self.model.device
            )
            ids = torch.tensor(
                seq.token_ids[num_cached:], dtype=torch.long, device=self.model.device
            ).reshape(1, q_len)
            logits = self.model.forward(
                ids,
                positions,
                mask,
                paged=PagedContext(self.pool, write_slots, gather_slots),
            )
            seq.num_cached_tokens = length
            if self.enable_prefix_caching:
                # Now that this prompt's KV is in the pool, make its full blocks
                # reusable by the next request that shares the prefix.
                self.block_manager.register_full_blocks(seq.block_table, seq.token_ids)
            token = self.sampler.sample(logits[0, -1], seq.temperature, seq.generator)
            self.prefix_reused_blocks += len(reused)
            self.prefix_reused_tokens += num_cached
            self._postprocess(seq, token)
        except Exception as exc:  # noqa: BLE001 — isolate: fail just this request
            # A single-sequence prefill can fail on its own (e.g. an out-of-vocab
            # prompt id); that must not poison batch-mates or wedge the loop, so we
            # fail+free+retire only this sequence — the base engine's contract.
            self._fail_seq(seq, exc)
            return
        self._record(
            BatchStats(
                batch_size=1,
                max_prompt_len=length,
                prompt_pad_tokens=0,
                decode_steps=0,
                decode_slack_tokens=0,
                kv_block_occupancy=self._occupancy(),
            )
        )

    def _decode_step(self) -> None:
        """Advance the whole running set by one token in a single batched forward."""
        self._ensure_blocks(list(self.scheduler.running))
        running = list(self.scheduler.running)
        if not running:
            return

        device, dtype = self.model.device, self.model.dtype
        context_lens = [
            s.num_cached_tokens + 1 for s in running
        ]  # history incl. new token
        max_len = max(context_lens)

        write_slots = self._slots(
            [self._slot(s.block_table, s.num_cached_tokens) for s in running]
        )
        gather_slots = torch.zeros(
            len(running), max_len, dtype=torch.long, device=device
        )
        for i, seq in enumerate(running):
            bt = torch.tensor(seq.block_table, device=device)
            pos = torch.arange(context_lens[i], device=device)
            gather_slots[i, : context_lens[i]] = bt[
                pos // self.block_size
            ] * self.block_size + (pos % self.block_size)
        mask = self._decode_mask(context_lens, max_len, dtype, device)
        positions = torch.tensor(
            [[s.num_cached_tokens] for s in running], device=device
        )
        ids = torch.tensor([[s.last_token] for s in running], device=device)

        logits = self.model.forward(
            ids,
            positions,
            mask,
            paged=PagedContext(self.pool, write_slots, gather_slots),
        )
        next_logits = logits[:, -1]
        for i, seq in enumerate(running):
            seq.num_cached_tokens += 1  # the token just written is now cached
            try:
                token = self.sampler.sample(
                    next_logits[i], seq.temperature, seq.generator
                )
                self._postprocess(seq, token)
            except Exception as exc:  # noqa: BLE001 — isolate this seq, keep the rest
                self._fail_seq(seq, exc)
        self._record(
            BatchStats(
                batch_size=len(running),
                max_prompt_len=0,
                prompt_pad_tokens=0,
                decode_steps=1,
                decode_slack_tokens=0,  # no finished seq forwarded (no HOL)
                kv_block_occupancy=self._occupancy(),
            )
        )

    # --- chunked prefill: the mixed prefill+decode step (T4b) ---------------

    def _step_mixed(self) -> None:
        """One mixed step: admit → plan (decode-first) → ensure blocks → one forward.

        The chunked-prefill path (``enable_chunked_prefill``). Unlike the two-phase
        default (`_step`), a single step both decodes the running set **and** advances
        one or more in-flight prefills by a chunk, so a long prompt never blocks the
        decode queue as a monolithic prefill. The whole batch — prefill chunks (many
        query tokens) + decodes (one each) — runs in one flattened forward
        (`_run_mixed_forward`), which reduces to the standalone prefill/decode paths
        as special cases, so greedy output is token-for-token identical (Seam A).
        """
        self._admit_waiting_mixed()
        if not self.scheduler.running:
            return
        plan = plan_mixed_step(
            list(self.scheduler.running),
            token_budget=self.token_budget,
            chunk_size=self.chunk_size,
        )
        if not plan.chunks:
            return
        self._ensure_blocks_mixed(plan)
        # Preemption (block pressure) may have evicted a chunk's sequence — build the
        # forward only over the survivors still RUNNING.
        chunks = [c for c in plan.chunks if c.seq.status is SequenceStatus.RUNNING]
        if chunks:
            self._run_mixed_forward(chunks)
        if plan.is_mixed:
            self.mixed_steps += 1  # this step carried both a prefill chunk and decode

    def _admit_waiting_mixed(self) -> None:
        """Admit waiting requests, gating on the first chunk's blocks, not the prompt.

        Chunking's point is to admit a long prompt the pool couldn't house all at
        once, so the gate reserves only ``num_cached + first_chunk`` tokens' blocks.
        Prefix caching (T4a) composes: ``match_prefix`` seeds ``num_cached`` and the
        reused block table. A prompt with an out-of-vocab id is failed here (the
        batched mixed forward can't isolate a bad row the way single-seq prefill can).

        Multiple requests may be admitted per step; ``reserved`` tracks the blocks
        already promised to this step's earlier admits (allocated later in
        ``_ensure_blocks_mixed``) so the gate stays honest and never over-admits into
        immediate preemption.
        """
        sched = self.scheduler
        vocab = self.model.config.vocab_size
        reserved = 0  # first-chunk blocks promised to already-admitted seqs this step
        while sched.waiting and len(sched.running) < self.max_num_seqs:
            seq = sched.waiting[0]
            if any(not (0 <= t < vocab) for t in seq.token_ids):
                sched.waiting.popleft()
                self._fail_seq(seq, RuntimeError("prompt token id out of vocab range"))
                continue
            if self.enable_prefix_caching:
                reused, num_cached = self.block_manager.match_prefix(seq.token_ids)
            else:
                reused, num_cached = [], 0
            first_chunk = min(self.chunk_size, len(seq.token_ids) - num_cached)
            need = self.block_manager.blocks_for(num_cached + first_chunk) - len(reused)
            if self.block_manager.num_free_blocks - reserved >= need:
                sched.waiting.popleft()
                seq.num_cached_tokens = num_cached
                seq.block_table = list(reused)
                self.prefix_reused_blocks += len(reused)
                self.prefix_reused_tokens += num_cached
                seq.status = SequenceStatus.RUNNING
                sched.running.append(seq)
                reserved += need
                continue
            # Not enough blocks for even the first chunk. Undo the prefix touch (the
            # seq stays queued and will re-match next step), then either fail it (pool
            # too small and nothing to drain) or wait for the running set to free up.
            self.block_manager.free(reused)
            if not sched.running:
                sched.waiting.popleft()
                self._fail_seq(seq, RuntimeError("KV pool too small for this prompt"))
                continue
            break

    def _ensure_blocks_mixed(self, plan: MixedPlan) -> None:
        """Grow each chunk's block table to cover its new tokens; preempt if dry.

        A prefill chunk may need several new blocks (not just one like decode), so we
        grow ``block_table`` up to ``blocks_for(num_cached + num_query_tokens)``,
        reusing the LIFO ``_pick_victim`` / ``_preempt`` recompute-preemption loop
        under block pressure. A sequence that needs a block while it is the only one
        left and the pool is empty is failed (pool too small).
        """
        for chunk in plan.chunks:
            seq = chunk.seq
            if seq.status is not SequenceStatus.RUNNING:
                continue  # already evicted as a victim this pass
            target = self.block_manager.blocks_for(
                seq.num_cached_tokens + chunk.num_query_tokens
            )
            while len(seq.block_table) < target:
                while self.block_manager.num_free_blocks == 0:
                    victim = self._pick_victim(exclude=seq)
                    if victim is None:
                        self._fail_seq(seq, RuntimeError("KV pool exhausted"))
                        break
                    self._preempt(victim)
                if seq.status is not SequenceStatus.RUNNING:
                    break  # this seq was the one failed above
                seq.block_table.append(self.block_manager.allocate_new(1)[0])

    def _run_mixed_forward(self, chunks: list[SeqChunk]) -> None:
        """Flatten every chunk into one varlen forward; sample completed rows.

        All scheduled query tokens (prefill chunks + decodes) are packed into a
        single ``[1, Q]`` frame; each sequence's full history ``[0, start+q)`` is
        gathered into a shared ``[1, K]`` key axis; ``build_varlen_mask`` keeps every
        query token attending only its own sequence's causal history — so the packed
        forward equals running each sequence alone. Positions carry each token's true
        absolute index (RoPE correctness). Only a decode or a prefill chunk that
        *completes* the prompt samples a token; a mid-prompt chunk emits nothing.
        """
        device, dtype = self.model.device, self.model.dtype
        input_ids: list[int] = []
        positions: list[int] = []
        write_slots: list[int] = []
        gather_slots: list[int] = []
        q_seq_ids: list[int] = []
        q_pos: list[int] = []
        k_seq_ids: list[int] = []
        k_pos: list[int] = []
        for idx, chunk in enumerate(chunks):
            seq = chunk.seq
            start, q = seq.num_cached_tokens, chunk.num_query_tokens
            for p in range(start, start + q):  # this step's new query tokens
                input_ids.append(seq.token_ids[p])
                positions.append(p)
                write_slots.append(self._slot(seq.block_table, p))
                q_seq_ids.append(idx)
                q_pos.append(p)
            for p in range(start + q):  # full history to attend (incl. new tokens)
                gather_slots.append(self._slot(seq.block_table, p))
                k_seq_ids.append(idx)
                k_pos.append(p)

        num_q = len(input_ids)
        ids = torch.tensor(input_ids, dtype=torch.long, device=device).reshape(1, num_q)
        pos = torch.tensor(positions, dtype=torch.long, device=device).reshape(1, num_q)
        mask = build_varlen_mask(
            torch.tensor(q_seq_ids, device=device),
            torch.tensor(q_pos, device=device),
            torch.tensor(k_seq_ids, device=device),
            torch.tensor(k_pos, device=device),
            dtype,
            device,
        )
        logits = self.model.forward(
            ids,
            pos,
            mask,
            paged=PagedContext(
                self.pool,
                self._slots(write_slots),
                self._slots(gather_slots).reshape(1, -1),
            ),
        )
        rows = logits[0]  # [Q, vocab] — one row per packed query token

        num_decode = 0
        q_off = 0
        for chunk in chunks:
            seq = chunk.seq
            last = q_off + chunk.num_query_tokens - 1
            q_off += chunk.num_query_tokens
            seq.num_cached_tokens += chunk.num_query_tokens
            if self.enable_prefix_caching and chunk.is_prefill:
                # Publish the prompt blocks this chunk just filled (idempotent; caps
                # at the prompt so generated tokens aren't cached).
                self.block_manager.register_full_blocks(
                    seq.block_table, seq.token_ids[: seq.num_cached_tokens]
                )
            if not chunk.is_prefill:
                num_decode += 1
            if chunk.is_prefill and not chunk.completes_prefill:
                continue  # still prefilling — no token emitted this step
            try:
                token = self.sampler.sample(rows[last], seq.temperature, seq.generator)
                self._postprocess(seq, token)
            except Exception as exc:  # noqa: BLE001 — isolate this seq, keep the rest
                self._fail_seq(seq, exc)

        self._record(
            BatchStats(
                batch_size=num_decode,  # the decode grid (prefill-only step → 0)
                max_prompt_len=0,
                prompt_pad_tokens=0,
                decode_steps=1 if num_decode else 0,
                decode_slack_tokens=0,
                kv_block_occupancy=self._occupancy(),
            )
        )

    # --- block accounting + preemption -------------------------------------

    def _ensure_blocks(self, running: list[Sequence]) -> None:
        """Give each running seq a block for its next token; preempt if the pool is dry.

        Recompute preemption: evict the newest running sequence (LIFO, so FIFO
        fairness is preserved on re-admission), free its blocks, and requeue it at
        the waiting head. A sequence that needs a block while it is the only one
        left and the pool is empty can't be served — fail it (pool too small).
        """
        for seq in running:
            if seq.status is not SequenceStatus.RUNNING:
                continue  # already preempted as a victim this pass
            if seq.num_cached_tokens < len(seq.block_table) * self.block_size:
                continue  # its current blocks still have room for the next token
            while self.block_manager.num_free_blocks == 0:
                victim = self._pick_victim(exclude=seq)
                if victim is None:
                    self._fail_seq(seq, RuntimeError("KV pool exhausted"))
                    break
                self._preempt(victim)
            if seq.status is SequenceStatus.RUNNING:
                self.block_manager.append(seq.block_table, seq.num_cached_tokens)

    def _pick_victim(self, exclude: Sequence) -> Sequence | None:
        """Newest running sequence that isn't ``exclude`` (LIFO preemption)."""
        for seq in reversed(self.scheduler.running):
            if seq is not exclude:
                return seq
        return None

    def _preempt(self, victim: Sequence) -> None:
        """Recompute preemption: free blocks, reset, return to the waiting head."""
        self.block_manager.free(victim.block_table)
        victim.block_table = []
        victim.num_cached_tokens = (
            0  # -> needs_prefill; re-prefills token_ids on return
        )
        self.scheduler.running.remove(victim)
        victim.status = SequenceStatus.WAITING
        self.scheduler.waiting.appendleft(victim)

    def _fail_seq(self, seq: Sequence, exc: BaseException) -> None:
        """Fail one request in isolation and return its blocks to the pool."""
        pending = self._pending.pop(seq.seq_id, None)
        if pending is not None and not pending.done.is_set():
            pending.fail(exc)
        self.block_manager.free(seq.block_table)
        seq.block_table = []
        self.scheduler.retire(seq)
        try:
            self.scheduler.waiting.remove(seq)
        except ValueError:
            pass

    def _release(self, seq: Sequence) -> None:
        """Return a finished sequence's blocks to the pool (``_postprocess`` hook)."""
        self.block_manager.free(seq.block_table)
        seq.block_table = []

    def _slot(self, block_table: list[int], pos: int) -> int:
        """Flat physical slot for logical position ``pos`` via the block table."""
        return block_table[pos // self.block_size] * self.block_size + (
            pos % self.block_size
        )

    def _slots(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=self.model.device)

    def _decode_mask(
        self, context_lens: list[int], max_len: int, dtype, device
    ) -> torch.Tensor:
        """Additive mask ``[B, 1, 1, max_len]`` — each query attends its own history."""
        cols = torch.arange(max_len, device=device)
        lens = torch.tensor(context_lens, device=device)[:, None]
        invalid = cols[None, :] >= lens  # [B, max_len]
        mask = torch.zeros(len(context_lens), 1, 1, max_len, dtype=dtype, device=device)
        return mask.masked_fill(invalid[:, None, None, :], torch.finfo(dtype).min)

    def _occupancy(self) -> float:
        """Used token-slots / allocated block-slots — KV block occupancy (ADR-0002).

        The gap from 1.0 is internal fragmentation (partly-filled last blocks),
        never external fragmentation — freed blocks always return whole to the
        pool. Rises as the running set grows to fill its allocated blocks.
        """
        used = self.block_manager.num_used_blocks * self.block_size
        if used == 0:
            return 0.0
        filled = sum(s.num_cached_tokens for s in self.scheduler.running)
        return min(filled / used, 1.0)
