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

## Status

🗺️ **Charting** — the plan lives as a [wayfinder map issue](https://github.com/xiangzhang-coding/infrared/issues) with build tickets. See `docs/spec/`, `docs/adr/`, and `CONTEXT.md` for the settled decisions and glossary.
