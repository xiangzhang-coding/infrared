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
from infrared.bench.workload import Category, Workload, decode_heavy_category

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


def _demo_workload(vocab_size: int, seed: int) -> Workload:
    """A small mixed workload within the model's vocab (token ids, no tokenizer)."""
    short = decode_heavy_category(
        n=4, prompt_len=4, max_new_tokens=8, vocab_size=vocab_size, seed=seed
    )
    long = decode_heavy_category(
        n=2, prompt_len=8, max_new_tokens=24, vocab_size=vocab_size, seed=seed + 1
    )
    return Workload(
        categories=[
            Category(name="short", prompts=short.prompts, max_new_tokens=8),
            Category(name="long", prompts=long.prompts, max_new_tokens=24),
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
    workload = _demo_workload(model.config.vocab_size, args.seed)
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
        ),
        tier="T3 +paged KV",
        notes=f"{label} · paged {args.num_blocks}×{args.block_size} · batched decode",
    )

    print(f"# infrared metrics spine — {label}\n")
    print("## Before→after ladder\n")
    print(build_ladder([t1, t2, t3]))
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
        " enables. The Triton paged-attn kernel that fuses the gather is T4 (R1"
        " §5, §8)."
    )
    print("\n## T1 static batch — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t1.sweep))
    print("\n## T2 continuous batch — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t2.sweep))
    print("\n## T3 +paged KV — knee sweep (request-rate up-scan)\n")
    print(render_sweep_markdown(t3.sweep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
