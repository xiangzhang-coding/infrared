# Research · T4 GPU-only API surfaces — Triton paged-attn kernel + torch CUDA graphs

> **Ticket**: R3 · Research (#10), part of the infrared map (#1). Blocks **T4c (#13)** Triton kernel and **T4d (#14)** CUDA graphs.
> **Policy (ADR-0004 / ADR-0006)**: learn the *shape* from vLLM/FlashAttention, **rewrite with teaching comments, never copy-paste**. **Only APIs actually verified via Context7 or Sonar in the target versions are reported here; every API is tagged with its source + the version it was verified in. Anything unverified is flagged explicitly.**
> **Target stack**: real `torch==2.12.0` (ADR-0006 verified-green) + `triton` (transitively resolved by torch's Linux CUDA wheel — we do NOT pin it) + Linux/CUDA only.

## 0. TL;DR

_(outline — filled in below)_

## 1. Version matrix — what pairs with torch 2.12

## 2. Triton paged-attention kernel — the real API

### 2.1 Decorators / launch / program id
### 2.2 Loads, stores, block pointers, masking
### 2.3 `tl.dot` + numerically-stable online softmax
### 2.4 Minimal GPU-compilable paged-attn kernel skeleton
### 2.5 The `store_kvcache` (scatter) kernel skeleton

## 3. torch CUDA graphs — capture & replay a decode step

### 3.1 Low-level `torch.cuda.CUDAGraph` + `torch.cuda.graph(...)`
### 3.2 `make_graphed_callables`
### 3.3 Static-input-buffer constraints + warmup
### 3.4 Variable-length / paged batches under graphs

## 4. Linux/CUDA-only compat notes (lazy import, no CPU path)

## 5. API provenance table (source + verified version)

## 6. What I could NOT verify (honest gaps)

---

_↩ Back to tracking issue: [infrared#10 — R3 · Research: Triton paged-attn kernel API + torch CUDA graphs API](https://github.com/xiangzhang-coding/infrared/issues/10)_
