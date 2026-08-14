"""Thin serving shell (T5 — stub).

An OpenAI-compatible FastAPI front end (tokenize · assemble Request · stream
detokenize) that drives the engine. Demonstration-grade only — no production
fleet (``docs/spec/0001`` §Out of Scope). Nothing is wired to the engine yet,
and FastAPI is imported lazily so this package stays import-safe without it.
"""
