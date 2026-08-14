"""Smoke test (issue #4).

The whole ``infrared`` package tree must import cleanly with **no GPU and no
model download**, and we report whatever pinned runtime deps happen to be
present. infrared's modules never import torch/triton/transformers at load
time, so this passes in "no-GPU" / dev mode where those (R2-pinned) deps are
absent.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

from pytest import importorskip

import infrared

# Modules that stay pure Python (import with no heavy deps) — always importable.
PURE_SUBMODULES = [
    "infrared",
    "infrared.config",
    "infrared.model",
    "infrared.model.config",
    "infrared.model.model_runner",
    "infrared.engine",
    "infrared.engine.sequence",
    "infrared.engine.scheduler",
    "infrared.engine.engine",
    "infrared.cache",
    "infrared.cache.block_manager",
    "infrared.server",
    "infrared.server.app",
    "infrared.bench",
    "infrared.bench.harness",
]

# Modules that legitimately import torch (real forward-pass code lands at T0).
TORCH_SUBMODULES = [
    "infrared.model.layers",
    "infrared.model.qwen2",
    "infrared.model.sampler",
    "infrared.model.generate",
    "infrared.cache.kv_cache",
]

# Pinned runtime deps (R2 / issue #3) — optional in no-GPU dev mode.
OPTIONAL_DEPS = ["torch", "triton", "transformers", "safetensors", "fastapi", "uvicorn"]


def test_pure_submodules_import() -> None:
    for name in PURE_SUBMODULES:
        importlib.import_module(name)


def test_torch_submodules_import() -> None:
    importorskip("torch")
    for name in TORCH_SUBMODULES:
        importlib.import_module(name)


def test_version_exposed() -> None:
    assert isinstance(infrared.__version__, str)
    assert infrared.__version__


def test_config_is_pure_data() -> None:
    # EngineConfig is data only (no engine logic) — safe to construct anywhere.
    from infrared.config import EngineConfig

    cfg = EngineConfig()
    assert cfg.block_size == 16
    assert cfg.model.startswith("Qwen/Qwen2.5")


def test_importing_infrared_pulls_no_heavy_deps() -> None:
    # Checked in a fresh interpreter so the result is independent of whatever
    # else this test session has already imported.
    code = (
        "import sys, infrared\n"
        "heavy = ('torch', 'triton', 'transformers', 'fastapi', 'uvicorn')\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "print(','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"importing infrared dragged in heavy deps: {result.stdout.strip()}"
    )


def test_print_dependency_versions() -> None:
    # "打印依赖版本" — report what's actually installed (no-GPU-friendly).
    lines = [f"infrared {infrared.__version__}"]
    for name in OPTIONAL_DEPS:
        try:
            mod = importlib.import_module(name)
            lines.append(f"  {name}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            lines.append(f"  {name}: not installed (no-GPU/dev mode)")
    report = "\n".join(lines)
    print("\n" + report)  # visible with `pytest -s`
    assert "infrared" in report
