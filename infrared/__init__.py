"""infrared — a from-scratch LLM inference engine (Qwen2.5), built tier by tier.

This is the scaffold from issue #4: package skeleton only, **no engine or model
logic**. Every subpackage is an import-safe stub whose public seams mirror the
R1 architecture blueprint (``docs/research/vllm-v1-nano-vllm-blueprint.md`` §8)
so the later tiers (T0–T6, see ``docs/spec/0001``) can fill them in place.

Subpackages
-----------
- ``infrared.model``  — self-implemented Qwen2.5 forward (RMSNorm/RoPE/GQA/SwiGLU)
                        + the GPU worker that runs it and samples.
- ``infrared.engine`` — Engine busy loop, continuous-batching Scheduler, and the
                        Sequence state machine (CPU-side orchestration).
- ``infrared.cache``  — PagedAttention-style KV block manager + KV cache tensors.
- ``infrared.server`` — thin FastAPI serving shell (T5).
- ``infrared.bench``  — metrics harness / before→after ladder (T5).

Nothing here imports torch, triton, or transformers at module load — the whole
package is importable in "no-GPU" / dev mode (issue #4 acceptance).
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
