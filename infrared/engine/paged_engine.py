"""Paged continuous-batching engine (T3) — PagedAttention + batched decode.

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
   loop. (Prefill stays one sequence per step; chunked/flattened-varlen prefill
   and the Triton paged-attn kernel are T4 — R1 §5/§8.)

When the pool can't grant a block mid-decode, a **recompute preemption** evicts
the newest running sequence (frees its blocks, returns it to the waiting head);
when re-admitted it re-prefills its ``token_ids`` (prompt + tokens generated so
far), so no output is lost — just recomputed. Greedy output stays token-for-token
identical to the T0 oracle: the paged read/write reproduces each sequence's exact
history, and batched decode is just independent per-sequence attentions masked
apart (validated in tests/test_paged_kv.py).
"""

from __future__ import annotations

import torch

from infrared.cache.block_manager import BlockManager
from infrared.cache.paged_kv_cache import PagedContext, PagedKVPool
from infrared.engine.engine import ContinuousBatchEngine
from infrared.engine.sequence import Sequence, SequenceStatus
from infrared.engine.static_batch import BatchStats
from infrared.model.inputs import build_attention_mask, build_positions
from infrared.model.qwen2 import Qwen2ForCausalLM


class PagedBatchEngine(ContinuousBatchEngine):
    """Continuous batching over a paged KV pool, with batched decode + preemption."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        max_num_seqs: int = 8,
        block_size: int = 16,
        num_blocks: int = 256,
    ) -> None:
        super().__init__(model, max_num_seqs=max_num_seqs)
        self.block_size = block_size
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
        sched = self.scheduler
        # Prefill admission: a free running slot AND enough free blocks. Admitting
        # only prompt-sized blocks (not the worst-case generation) is what lets
        # more sequences share the pool than T2's contiguous reservation would.
        if sched.waiting and len(sched.running) < self.max_num_seqs:
            seq = sched.waiting[0]
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
        """Allocate blocks, run one paged prefill over ``token_ids``, sample."""
        if seq.seed is not None and seq.generator is None:
            seq.generator = torch.Generator(device=self.model.device).manual_seed(
                seq.seed
            )
        length = len(seq.token_ids)
        seq.block_table = self.block_manager.allocate(length)
        try:
            write_slots = self._slots(
                [self._slot(seq.block_table, p) for p in range(length)]
            )
            gather_slots = write_slots.reshape(1, length)
            positions = build_positions([0], 0, length, self.model.device)
            mask = build_attention_mask(
                [0], 0, length, length, self.model.dtype, self.model.device
            )
            ids = torch.tensor(
                seq.token_ids, dtype=torch.long, device=self.model.device
            ).reshape(1, length)
            logits = self.model.forward(
                ids,
                positions,
                mask,
                paged=PagedContext(self.pool, write_slots, gather_slots),
            )
            seq.num_cached_tokens = length
            token = self.sampler.sample(logits[0, -1], seq.temperature, seq.generator)
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
