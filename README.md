# infrared

> A **from-scratch LLM inference engine** — built to learn inference-serving optimization by *implementing* the mechanisms (continuous batching, a paged KV cache, a scheduler, Triton kernels) rather than tuning someone else's flags. Plus a thin serving layer, so it runs end-to-end.

`infrared` is the hands-on companion to [`inference-learning-path`](https://github.com/xiangzhang-coding/inference-learning-path) (a bilingual tutorial site that teaches you to *understand and tune* vLLM). Where that site deliberately stops at "read the source, tune the knobs," `infrared` crosses the line and **builds the engine** — the place high-concurrency / high-utilization stops being knobs you turn and becomes machinery you own.

## The arc (T0 → T6)

Built in rungs; each rung is a mechanism you implement and then **measure** against the one below it:

| Rung | What you build | Teaches |
|---|---|---|
| **T0** | Single-request correct generation — Qwen2.5 forward (RMSNorm/RoPE/GQA/SwiGLU), per-request KV, sampling loop | prefill/decode, KV growth |
| **T1** | Static batching + request queue + HTTP server (OpenAI-ish) | batching, padding waste, why static under-utilizes |
| **T2** | **Continuous batching** scheduler | the #1 throughput lever; the scheduler as the heart |
| **T3** | **Paged KV cache** (PagedAttention-style block manager) | the vLLM core innovation; memory utilization |
| **T4** | Efficiency: Triton paged-attention kernel, prefix caching, chunked prefill, CUDA graphs | the performance half |
| **T5** | Serving layer: streaming, metrics, load-test harness | the ops / serving-systems half |
| **T6** | Beyond: speculative decoding, quantization, multi-LoRA, tensor parallelism | advanced serving |

## What "done" measures

- **High concurrency** = **goodput**: req/s meeting an SLO (p99 TTFT / TPOT), and the concurrency **knee**.
- **High utilization** = GPU compute utilization + **KV-block occupancy / batch-fill** over time — evidence the scheduler and paging actually work.
- **Efficient** = output tok/s per GPU-second, and the Triton-kernel speedup over a naive gather.
- The killer artifact: a **`static → continuous → +paged → +Triton`** before→after table, where every rung is a mechanism *you built*.

## The "from-scratch" boundary

**Build** the serving machinery (forward wiring + attention/KV path, KV cache, scheduler, batching, server, Triton kernels, prefix caching, …). **Stand on** primitives — PyTorch (GEMM/tensors), HF `transformers`/`safetensors` (weight loading + tokenizer only, never `.generate()`), Triton (to write *our own* kernels), FastAPI (HTTP shell). **Never** use an inference *engine* (vLLM / ONNX Runtime / TensorRT-LLM / TGI) as the execution path — that would hollow out the whole point; they are allowed only as **benchmark yardsticks**.

## Baseline

Single RTX 4090 (24 GB) via AutoDL, ¥500 budget (inherited from `inference-learning-path` ADR-0001). Correctness dev on `Qwen2.5-0.5B-Instruct` (CPU/Mac-friendly); headline benchmark on `Qwen2.5-7B`.

## Repo layout

T0 (single-request forward) and T1 (static batching + HTTP) are implemented; later rungs are still stubs. Names mirror the R1 architecture blueprint (`docs/research/vllm-v1-nano-vllm-blueprint.md`), split along the one seam that matters — request orchestration vs model execution:

```
infrared/
  config.py            # EngineConfig (block_size, budgets, …) — data only
  model/               # model execution (batch-first; B=1 is the T0 path)
    config.py          #   Qwen2Config, read from HF config.json               (T0)
    layers.py          #   RMSNorm / RoPE / GQA attention / SwiGLU             (T0)
    inputs.py          #   positions + additive causal/padding masks           (T1)
    qwen2.py           #   Qwen2.5 forward + safetensors loader (tied lm_head) (T0)
    sampler.py         #   greedy / temperature                               (T0)
    generate.py        #   single-request prefill→decode + tokenizer path     (T0)
    model_runner.py    #   Worker seam (stub)                                 (T3+)
  engine/              # request orchestration
    static_batch.py    #   static batch: left-pad prefill + lockstep decode   (T1)
    engine.py          #   StaticBatchEngine: request queue + worker thread   (T1)
    scheduler.py       #   continuous-batching scheduler (stub)               (T2)
    sequence.py        #   Sequence state machine (stub)                      (T2)
  cache/
    kv_cache.py        #   contiguous per-request KV cache (batch-first)      (T0/T1)
    block_manager.py   #   PagedAttention block manager (stub)                (T3)
  server/app.py        #   FastAPI OpenAI /v1/completions (non-streaming)     (T1)
  bench/harness.py     #   metrics spine / before→after ladder (stub)         (T5)
```

## Development

Runtime deps (torch / triton / transformers / …) are pinned to the versions in `docs/research/deps-and-qwen25-arch.md` and are **Linux + GPU** (triton ships Linux wheels only). On a no-GPU box, use the dev path — it skips them but still runs the tests and linter green:

```bash
make install-dev   # editable install + pytest/ruff, no torch/triton
make test          # unit + smoke tests (parity tests skip unless the model is cached)
make lint          # ruff check (via uvx)
make install       # full runtime install — Linux + GPU only
```

The scaffold-level tests never touch a GPU or download a model, so `make test` / `make lint` pass anywhere.

### Correctness gate (Seam A)

T0's acceptance is that infrared's forward matches HF `transformers` on the same weights (greedy output + first-step logits). That gate lives in `tests/test_parity.py` and **skips** unless `Qwen2.5-0.5B-Instruct` is cached locally. To run it (downloads ~1 GB once, runs on CPU):

```bash
make parity        # fetch the 0.5B weights, then run the HF parity test
```

HF is used only as the weight source (+ tokenizer) and the reference oracle — never via `.generate()` (ADR-0003).

### Serving (T1)

Run the OpenAI-compatible server (loads `Qwen2.5-0.5B-Instruct` by default; override with `INFRARED_MODEL`):

```bash
uvicorn infrared.server.app:build_app --factory --port 8000
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

Static batching is deliberately the naive baseline for T2: the response carries a non-standard `infrared_batch` field (`prompt_pad_tokens`, `decode_slack_tokens`, …) so you can *see* the padding waste and head-of-line blocking a continuous-batching scheduler will remove.

## Status

🛠️ **Building** — T0 (single-request forward, HF-parity gated) and T1 (static batching + OpenAI-compatible HTTP) are in; T2 (continuous batching) is next. The plan lives as a [wayfinder map issue](https://github.com/xiangzhang-coding/infrared/issues) with build tickets. See `docs/spec/`, `docs/adr/`, and `CONTEXT.md` for the settled decisions and glossary.
