"""``python -m infrared.bench`` — run the metrics spine and print the ladder.

Defaults to a **tiny random model** on CPU so ``make bench`` runs anywhere with
torch installed (no GPU, no download) and shows the whole spine end to end:
correctness (vs the T0 oracle), throughput, the goodput/knee sweep, and the
utilization evidence. Point ``--model`` at a local dir / HF id (e.g. the cached
``Qwen2.5-0.5B-Instruct``) for real weights; the workload is synthetic token ids
within the model's vocab, so no tokenizer is needed either way.

This is the unified entry the issue asks for: it measures **T1 static batch** and
**T2 continuous batch** on the identical model + workload + SLO and stacks them
into the ``static → continuous`` before→after ladder — every future tier plugs
the same ``measure`` call behind a different engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from infrared.bench.harness import build_ladder, measure, t0_oracle
from infrared.bench.metrics import SLO
from infrared.bench.report import render_sweep_markdown
from infrared.bench.workload import (
    Category,
    Workload,
    decode_heavy_category,
    long_prefill_category,
    shared_prefix_category,
)

if TYPE_CHECKING:  # hints only — torch/model imported lazily inside the functions
    from infrared.model.qwen2 import Qwen2ForCausalLM


def _tiny_model() -> Qwen2ForCausalLM:
    """A small random Qwen2.5 — fast, deterministic, no download (dev/demo)."""
    import torch

    from infrared.model.config import Qwen2Config
    from infrared.model.qwen2 import Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_ids=(),
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def _real_model(model_ref: str) -> Qwen2ForCausalLM:
    """Load real HF safetensors weights (fp32 on CPU) from a dir or HF id."""
    import torch

    from infrared.model.qwen2 import Qwen2ForCausalLM

    model_dir = model_ref
    if not Path(model_ref).exists():  # not a local dir — resolve as a hub id
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_ref)
    return Qwen2ForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, device="cpu"
    )


def _demo_workload(vocab_size: int, seed: int, block_size: int) -> Workload:
    """A small mixed workload within the model's vocab (token ids, no tokenizer).

    Includes a **shared-prefix** category (many prompts, one common preamble
    spanning ≥1 KV block) so the +prefix-caching tier has something to reuse, and a
    **long-prefill** category (multi-block prompts) so the +chunked-prefill tier has
    a long prefill to spread across steps; the prefix is sized to ``block_size`` so
    at least one whole block is cacheable.
    """
    short = decode_heavy_category(
        n=4, prompt_len=4, max_new_tokens=8, vocab_size=vocab_size, seed=seed
    )
    long = decode_heavy_category(
        n=2, prompt_len=8, max_new_tokens=24, vocab_size=vocab_size, seed=seed + 1
    )
    shared = shared_prefix_category(
        n=4,
        prefix_len=2 * block_size,
        tail_len=4,
        max_new_tokens=16,
        vocab_size=vocab_size,
        seed=seed + 2,
    )
    long_prefill = long_prefill_category(
        n=2,
        prompt_len=3 * block_size,
        max_new_tokens=16,
        vocab_size=vocab_size,
        seed=seed + 3,
    )
    return Workload(
        categories=[
            Category(name="short", prompts=short.prompts, max_new_tokens=8),
            Category(name="long", prompts=long.prompts, max_new_tokens=24),
            shared,
            long_prefill,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m infrared.bench")
    parser.add_argument(
        "--model",
        default=None,
        help="local model dir or HF id for real weights (default: tiny random model)",
    )
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument(
        "--block-size", type=int, default=16, help="paged KV block size (T3)"
    )
    parser.add_argument(
        "--num-blocks", type=int, default=128, help="paged KV pool size in blocks (T3)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="prefill chunk size for the +chunked-prefill tier (T4b)",
    )
    parser.add_argument(
        "--rates",
        default="5,25,100,400",
        help="comma-separated offered request rates (req/s) for the knee sweep",
    )
    parser.add_argument("--ttft-ms", type=float, default=500.0)
    parser.add_argument("--tpot-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "bench needs the runtime deps (torch). On a no-GPU box see the "
            "Makefile: `make install` is Linux+GPU; use base torch to run this.",
            file=sys.stderr,
        )
        return 0

    from infrared.engine.engine import ContinuousBatchEngine, StaticBatchEngine
    from infrared.engine.paged_engine import PagedBatchEngine

    model = _real_model(args.model) if args.model else _tiny_model()
    label = args.model or "tiny random model (CPU)"
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    slo = SLO.from_ms(ttft_ms=args.ttft_ms, tpot_ms=args.tpot_ms)
    workload = _demo_workload(model.config.vocab_size, args.seed, args.block_size)
    oracle = t0_oracle(model)

    # Measure each tier on the identical model + workload + SLO so the ladder's
    # deltas are attributable to the *mechanism* alone, nothing else. Every engine
    # shares the submit/Pending surface the harness drives.
    def _measure(make_engine, tier: str, notes: str):
        engine = make_engine().start()
        try:
            return measure(
                engine,
                oracle=oracle,
                workload=workload,
                slo=slo,
                rates=rates,
                tier=tier,
                seed=args.seed,
                notes=notes,
            )
        finally:
            engine.stop()

    t1 = _measure(
        lambda: StaticBatchEngine(model, max_batch_size=args.max_batch, linger=0.005),
        tier="T1 static batch",
        notes=label,
    )
    t2 = _measure(
        lambda: ContinuousBatchEngine(model, max_num_seqs=args.max_batch),
        tier="T2 continuous batch",
        notes=f"{label} · per-seq forward (batched fwd → T3)",
    )
    t3 = _measure(
        lambda: PagedBatchEngine(
            model,
            max_num_seqs=args.max_batch,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            enable_prefix_caching=False,  # pure paging — the T4 row adds caching
            enable_triton_attention=False,  # naive paged — the +Triton row is the A/B
        ),
        tier="T3 +paged KV",
        notes=f"{label} · paged {args.num_blocks}×{args.block_size} · batched decode",
    )

    # T4: same paged engine, prefix caching on. Keep the engine handle so its
    # reuse counters (physical blocks / prompt tokens served from cache) can go
    # into the row — the observable "it actually reused" evidence.
    t4_engine = PagedBatchEngine(
        model,
        max_num_seqs=args.max_batch,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        enable_prefix_caching=True,
        enable_triton_attention=False,  # naive paged — the +Triton row is the A/B
    ).start()
    try:
        t4 = measure(
            t4_engine,
            oracle=oracle,
            workload=workload,
            slo=slo,
            rates=rates,
            tier="T4 +prefix caching",
            seed=args.seed,
            notes="",
        )
    finally:
        t4_engine.stop()
    t4.row.notes = (
        f"{label} · reused {t4_engine.prefix_reused_blocks} prefix blocks "
        f"({t4_engine.prefix_reused_tokens} tok, engine-lifetime total) on "
        f"shared-prefix workload"
    )

    # T4b: prefix caching + chunked prefill. Keep the handle so ``mixed_steps`` (steps
    # that carried a prefill chunk and a decode together) goes into the row — the
    # structural evidence a long prefill interleaved with decode instead of blocking.
    t4b_engine = PagedBatchEngine(
        model,
        max_num_seqs=args.max_batch,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        chunk_size=args.chunk_size,
        enable_triton_attention=False,  # naive paged — the +Triton row is the A/B
    ).start()
    try:
        t4b = measure(
            t4b_engine,
            oracle=oracle,
            workload=workload,
            slo=slo,
            rates=rates,
            tier="T4 +chunked prefill",
            seed=args.seed,
            notes="",
        )
    finally:
        t4b_engine.stop()
    t4b.row.notes = (
        f"{label} · chunk={args.chunk_size}, budget={t4b_engine.token_budget} · "
        f"{t4b_engine.mixed_steps} mixed prefill+decode steps on long-prefill workload"
    )

    # T4c: same paged engine, the fused Triton paged-attn kernel on. On CUDA this
    # replaces the naive gather+attention with one fused kernel (the throughput
    # win); on a no-GPU box it transparently falls back to the naive path, so the
    # row equals T4b here — the note says so. The GPU speedup is measured on AutoDL.
    on_gpu = torch.cuda.is_available()
    t4c_engine = PagedBatchEngine(
        model,
        max_num_seqs=args.max_batch,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        chunk_size=args.chunk_size,
        enable_triton_attention=True,
    ).start()
    try:
        t4c = measure(
            t4c_engine,
            oracle=oracle,
            workload=workload,
            slo=slo,
            rates=rates,
            tier="T4 +Triton kernel",
            seed=args.seed,
            notes="",
        )
    finally:
        t4c_engine.stop()
    t4c.row.notes = f"{label} · fused paged-attn kernel " + (
        "engaged (CUDA)"
        if on_gpu
        else "falls back to naive on CPU (no CUDA) — GPU speedup measured on AutoDL"
    )

    print(f"# infrared metrics spine — {label}\n")
    print("## Before→after ladder\n")
    print(build_ladder([t1, t2, t3, t4, t4b, t4c]))
    print(
        "\n> **Reading the ladder.** **T2** (continuous batch) removes static"
        " batching's prompt padding + head-of-line waste (batch-fill → 100%) and"
        " streams a real TTFT, but forwards each sequence alone — so raw"
        " throughput can lag T1's single batched GEMM. **T3** (+paged KV) bundles"
        " two mechanisms, so read its deltas accordingly: the **KV-occupancy** and"
        " **concurrent-sequences-per-KV-budget** gains are attributable to *paging*"
        " (on-demand fixed-size blocks, no worst-case reservation, no"
        " fragmentation); the **throughput/goodput** gain over T2 is attributable"
        " to *batched decode* (one matmul across the running set), which paging"
        " enables. **T4 +prefix caching** reuses shared prompt-prefix KV blocks"
        " across requests (see the reused-blocks note): the win shows on"
        " shared-prefix workloads (system prompt / few-shot) as skipped prefill"
        " compute + fewer blocks held; it is a no-op when prompts share nothing."
        " **T4 +chunked prefill** splits a long prefill into chunks mixed with decode"
        " in one step (see the mixed-steps note), so a long prompt no longer blocks"
        " the decode queue — the TTFT/TPOT protection for in-flight requests."
        " **T4 +Triton kernel** fuses the paged gather + scaled-dot-product +"
        " online-softmax into one self-written Triton kernel (R1 §5, §8): on CUDA it"
        " skips materializing the gathered KV + full score matrix (the throughput"
        " win); on a no-GPU box it falls back to the naive path, so its row matches"
        " the prior one here and the GPU speedup is measured on AutoDL (see its note)."
    )
    print("\n## T1 static batch — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t1.sweep))
    print("\n## T2 continuous batch — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t2.sweep))
    print("\n## T3 +paged KV — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t3.sweep))
    print("\n## T4 +prefix caching — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t4.sweep))
    print("\n## T4 +chunked prefill — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t4b.sweep))
    print("\n## T4 +Triton kernel — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t4c.sweep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
