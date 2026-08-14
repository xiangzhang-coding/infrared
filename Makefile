# infrared — dev harness (issue #4 scaffold). Prefers `uv`.
#
# Full runtime deps (torch/triton/...) are Linux + GPU only (see pyproject.toml
# and R2). On a no-GPU box use `make install-dev`, which skips them but still
# lets `make test` / `make lint` run green.

.DEFAULT_GOAL := help
.PHONY: help install install-dev test lint fmt bench parity clean

PY ?= python3

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Full editable install incl. pinned runtime deps (Linux + GPU)
	uv pip install -e ".[dev]"

install-dev:  ## No-GPU install: infrared + test/lint tools, skips torch/triton
	uv pip install -e . --no-deps
	# Dev-tool versions are pinned once in pyproject's [dev] extra; install by
	# name here so the no-GPU path can't drift from that single source.
	uv pip install pytest ruff

test:  ## Run the test suite (parity tests skip unless the model is cached)
	$(PY) -m pytest

parity:  ## Seam-A gate: fetch Qwen2.5-0.5B (~1GB) + run HF parity (needs torch+transformers)
	$(PY) -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-0.5B-Instruct')"
	$(PY) -m pytest tests/test_parity.py -v

lint:  ## Lint with ruff (uvx = no separate install needed)
	uvx ruff check .

fmt:  ## Auto-fix + format with ruff
	uvx ruff check --fix .
	uvx ruff format .

bench:  ## Metrics harness (stub until T5)
	@echo "bench is a stub until T5 — see infrared/bench/harness.py"

clean:  ## Remove caches / build artifacts
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
