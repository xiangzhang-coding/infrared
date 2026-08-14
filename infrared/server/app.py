"""OpenAI-compatible HTTP shell (T1): non-streaming ``POST /v1/completions``.

``create_app(engine, tokenizer, ...)`` builds the ASGI app around an injected
``StaticBatchEngine`` + tokenizer, so tests can wire fakes without a GPU. The
endpoint is ``async`` and awaits each request's result in a threadpool, so many
concurrent clients enqueue and the engine's worker batches them — the request
queue T1 is about. ``build_app()`` is the ``--factory`` entry point that loads
the real model/tokenizer at server start (never at import).

FastAPI/torch are imported lazily inside the functions, so importing this module
stays dependency-free (no-GPU smoke test).

Note: this module intentionally does NOT use ``from __future__ import
annotations`` — FastAPI must see the request-body model as a real class (not a
stringized, closure-local name) to route it as a body rather than a query param.
"""

import os
import time
import uuid


def create_app(
    engine,
    tokenizer,
    eos_token_ids: tuple[int, ...] = (),
    model_name: str = "infrared-qwen2.5",
):
    """Build the FastAPI app around an engine + tokenizer (both injectable)."""
    import asyncio

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    from infrared.engine.static_batch import BatchRequest

    class CompletionRequest(BaseModel):
        prompt: str | list[str]
        model: str | None = None
        max_tokens: int = 64
        temperature: float = 0.0
        seed: int | None = None

    app = FastAPI(title="infrared", version="0.0.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> dict:
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt must be non-empty")

        # Tokenize + enqueue every prompt, then await them (the worker batches).
        pendings = []
        for text in prompts:
            prompt_ids = tokenizer.encode(text)
            if not prompt_ids:
                raise HTTPException(status_code=400, detail="prompt must be non-empty")
            pending = engine.submit(
                BatchRequest(
                    prompt_ids=prompt_ids,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    seed=req.seed,
                    eos_token_ids=eos_token_ids,
                )
            )
            pendings.append((prompt_ids, pending))

        loop = asyncio.get_running_loop()
        choices = []
        prompt_tokens = completion_tokens = 0
        last_stats = None
        for index, (prompt_ids, pending) in enumerate(pendings):
            try:
                output = await loop.run_in_executor(None, pending.result)
            except Exception as exc:  # worker failure -> 500
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finished_on_eos = bool(output) and output[-1] in eos_token_ids
            text = tokenizer.decode(output, skip_special_tokens=True)
            choices.append(
                {
                    "index": index,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": "stop" if finished_on_eos else "length",
                }
            )
            prompt_tokens += len(prompt_ids)
            completion_tokens += len(output)
            last_stats = pending.stats

        response = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if last_stats is not None:
            # Non-standard: surface the static-batch waste so `curl` can see it
            # (head-of-line blocking / padding — the T2 "before" baseline).
            response["infrared_batch"] = {
                "batch_size": last_stats.batch_size,
                "max_prompt_len": last_stats.max_prompt_len,
                "prompt_pad_tokens": last_stats.prompt_pad_tokens,
                "decode_steps": last_stats.decode_steps,
                "decode_slack_tokens": last_stats.decode_slack_tokens,
            }
        return response

    return app


def build_app():
    """``--factory`` entry point: load the real model/tokenizer and start the engine.

    Config via env: ``INFRARED_MODEL`` (HF id or local dir, default
    Qwen2.5-0.5B-Instruct), ``INFRARED_MAX_BATCH`` (default 8).
    """
    import torch

    from infrared.engine.engine import StaticBatchEngine
    from infrared.model.generate import load_tokenizer
    from infrared.model.qwen2 import Qwen2ForCausalLM

    model_id = os.environ.get("INFRARED_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    max_batch = int(os.environ.get("INFRARED_MAX_BATCH", "8"))

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(model_id)
    model = Qwen2ForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, device="cpu"
    )
    tokenizer = load_tokenizer(model_dir)
    engine = StaticBatchEngine(model, max_batch_size=max_batch).start()
    return create_app(
        engine,
        tokenizer,
        eos_token_ids=model.config.eos_token_ids,
        model_name=model_id,
    )
