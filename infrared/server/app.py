"""FastAPI application factory (T5 — stub).

``create_app`` will build the ASGI app (OpenAI-compatible routes, streaming).
FastAPI is imported **lazily inside the factory** so importing this module stays
dependency-free in no-GPU / dev mode (issue #4 acceptance) — the pinned
``fastapi`` is only needed once T5 actually builds the app.
"""

from __future__ import annotations

_T5 = "not implemented until T5 — see docs/spec/0001 §Goals(7)"


def create_app(*args: object, **kwargs: object) -> object:
    """Build and return the FastAPI app that serves the engine (T5).

    T5 will start with ``from fastapi import FastAPI`` inside this function.
    """
    raise NotImplementedError(_T5)
