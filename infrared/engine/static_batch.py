"""Static batching (T1) — the continuous-batching "before" baseline.

Gathers a fixed set of requests, **left-pads** their prompts to a common width,
runs one batched prefill, then decodes the whole batch in **lockstep** until
*every* sequence finishes (all-return-together). This is deliberately the naive
scheme whose costs T2 removes:

- **Padding waste** — short prompts are padded to the longest and those pad
  tokens still flow through the prefill (``stats.prompt_pad_tokens``).
- **Head-of-line blocking** — a sequence that finishes early keeps being
  forwarded every step until the slowest one is done (``stats.decode_slack``).

Left-padding aligns every sequence's last real token on the right edge, so decode
steps share one growing KV column — see ``infrared.model.inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from infrared.model.inputs import build_attention_mask, build_positions
from infrared.model.qwen2 import Qwen2ForCausalLM
from infrared.model.sampler import Sampler


@dataclass(slots=True)
class BatchRequest:
    """One request in a static batch."""

    prompt_ids: list[int]
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int | None = None
    eos_token_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class BatchStats:
    """Observable static-batch costs (the T2 baseline)."""

    batch_size: int
    max_prompt_len: int
    prompt_pad_tokens: int  # left-pad tokens forced through prefill
    decode_steps: int  # lockstep steps run (== the longest generation)
    decode_slack_tokens: int  # finished-seq slots still forwarded (HOL blocking)


@dataclass(slots=True)
class BatchResult:
    """Per-request generations plus the batch's waste stats."""

    outputs: list[list[int]] = field(default_factory=list)
    stats: BatchStats | None = None


@torch.no_grad()
def run_static_batch(
    model: Qwen2ForCausalLM,
    requests: list[BatchRequest],
    pad_id: int = 0,
    sampler: Sampler | None = None,
) -> BatchResult:
    """Run one static batch: left-pad, batched prefill, lockstep decode."""
    if not requests:
        return BatchResult(outputs=[], stats=BatchStats(0, 0, 0, 0, 0))

    sampler = sampler or Sampler()
    device, dtype = model.device, model.dtype
    batch = len(requests)

    lens = [len(r.prompt_ids) for r in requests]
    if min(lens) == 0:
        raise ValueError("static batch requires non-empty prompts")
    width = max(lens)
    pad_lens = [width - n for n in lens]
    max_new = max(r.max_new_tokens for r in requests)

    # Left-pad prompts to a common width.
    padded = [
        [pad_id] * pad + r.prompt_ids for pad, r in zip(pad_lens, requests, strict=True)
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # [B, P]

    kv = model.new_kv_cache(max_len=width + max_new, batch_size=batch)
    generators = [
        torch.Generator(device=device).manual_seed(r.seed)
        if r.seed is not None
        else None
        for r in requests
    ]
    stops = [set(r.eos_token_ids) for r in requests]

    # --- Prefill: one batched forward over the padded prompts. ---
    positions = build_positions(pad_lens, 0, width, device)
    mask = build_attention_mask(pad_lens, 0, width, width, dtype, device)
    logits = model.forward(input_ids, positions, mask, kv, start_col=0)
    next_logits = logits[:, -1, :]  # left-pad => last column is real for all seqs

    # --- Decode: lockstep until every sequence finishes. ---
    outputs: list[list[int]] = [[] for _ in requests]
    finished = [False] * batch
    col = width
    steps = 0
    while steps < max_new:
        tokens = []
        for i, req in enumerate(requests):
            if finished[i]:
                tokens.append(pad_id)  # still forwarded, but ignored (HOL waste)
                continue
            tok = sampler.sample(next_logits[i], req.temperature, generators[i])
            outputs[i].append(tok)
            tokens.append(tok)
            if tok in stops[i] or len(outputs[i]) >= req.max_new_tokens:
                finished[i] = True
        steps += 1
        if all(finished):
            break

        step_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
        positions = build_positions(pad_lens, col, 1, device)
        mask = build_attention_mask(pad_lens, col, 1, col + 1, dtype, device)
        logits = model.forward(step_ids, positions, mask, kv, start_col=col)
        next_logits = logits[:, -1, :]
        col += 1

    slack = sum(steps - len(out) for out in outputs)
    stats = BatchStats(
        batch_size=batch,
        max_prompt_len=width,
        prompt_pad_tokens=sum(pad_lens),
        decode_steps=steps,
        decode_slack_tokens=slack,
    )
    return BatchResult(outputs=outputs, stats=stats)
