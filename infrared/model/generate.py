"""Single-request generation loop (T0): prefill once, then decode step by step.

This is the phase boundary the whole engine is built around. **Prefill** runs one
forward over the entire prompt (filling the KV cache at positions ``0..S-1``);
**decode** then runs one forward per new token, each appending a single K/V slot
and reading the full history. Greedy (``temperature=0``) is the parity mode;
temperature sampling takes an optional seed.

Continuous batching (many requests sharing steps) is T1 — this loop is the
correctness baseline and drives exactly one request.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from infrared.model.qwen2 import Qwen2ForCausalLM
from infrared.model.sampler import Sampler


@dataclass(slots=True)
class GenerationOutput:
    """Result of a generation run."""

    prompt_ids: list[int]
    generated_ids: list[int]  # newly produced tokens (may end with an EOS token)


@torch.no_grad()
def generate(
    model: Qwen2ForCausalLM,
    input_ids: Sequence[int],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    eos_token_ids: Sequence[int] | None = None,
    seed: int | None = None,
    sampler: Sampler | None = None,
) -> GenerationOutput:
    """Generate up to ``max_new_tokens`` tokens for a single prompt."""
    sampler = sampler or Sampler()
    if eos_token_ids is None:
        eos_token_ids = model.config.eos_token_ids
    stop = set(eos_token_ids)
    device = model.device

    prompt = list(input_ids)
    kv = model.new_kv_cache(max_len=len(prompt) + max_new_tokens)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # --- Prefill: one forward over the whole prompt. ---
    ids = torch.tensor(prompt, dtype=torch.long, device=device)
    logits = model(ids, kv, start_pos=0)
    next_logits = logits[-1]
    pos = len(prompt)

    # --- Decode: one forward per new token. ---
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = sampler.sample(next_logits, temperature, generator)
        generated.append(token)
        if token in stop or len(generated) >= max_new_tokens:
            break  # stop, or we just sampled the last requested token
        step_ids = torch.tensor([token], dtype=torch.long, device=device)
        logits = model(step_ids, kv, start_pos=pos)
        next_logits = logits[-1]
        pos += 1

    return GenerationOutput(prompt_ids=prompt, generated_ids=generated)


def load_tokenizer(model_dir: str):
    """Load the HF tokenizer for a local model dir (tokenizer only, ADR-0003)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def generate_text(
    model: Qwen2ForCausalLM,
    tokenizer,
    prompt: str,
    *,
    system: str | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    """Text-in / text-out: apply the chat template, generate, decode.

    The engine owns tokenization/detokenization here (via the HF tokenizer),
    so a caller gets coherent text end to end without touching HF's model.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    out = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    return tokenizer.decode(out.generated_ids, skip_special_tokens=True)
