# Research · T4 GPU-only API surfaces — Triton paged-attn kernel + torch CUDA graphs

> **Ticket**: R3 · Research (#10), part of the infrared map (#1). Blocks **T4c (#13)** Triton kernel and **T4d (#14)** CUDA graphs.
> **Policy (ADR-0004 / ADR-0006)**: learn the *shape* from vLLM/FlashAttention, **rewrite with teaching comments, never copy-paste**. **Only APIs actually verified via Context7 or Sonar in the target versions are reported here; every API is tagged with its source + the version it was verified in. Anything unverified is flagged explicitly.**
> **Target stack**: real `torch==2.12.0` (ADR-0006 verified-green) + `triton` (transitively resolved by torch's Linux CUDA wheel — we do NOT pin it) + Linux/CUDA only.

## 0. TL;DR

- **Triton paged-attn kernel** = one `@triton.jit` function, gridded over `(seq, head, [q-block])` via `tl.program_id`; it **gathers** paged KV by walking a `block_table` with plain pointer arithmetic + masked `tl.load` (paged KV is non-contiguous, so `tl.make_block_ptr` fits the *contiguous* Q/O tiles, not the paged gather), computes scores with `tl.dot`, and runs a **numerically-stable online softmax** (running `m_i`/`l_i`, `exp2`, rescale-and-accumulate) so it never materializes the full score row. A separate tiny `store_kvcache` `@triton.jit` kernel scatters this step's K/V into `slot_mapping`.
- **CUDA graphs** = warm up on a side `torch.cuda.Stream`, capture the decode forward once into a `torch.cuda.CUDAGraph()` under the `torch.cuda.graph(g)` context, then per step `static_*.copy_(real)` → `g.replay()`. Buffers are **fixed-shape and long-lived**; inference engines capture **one graph per batch-size bucket**, **pad up** to the bucket, keep **prefill eager**, and **share one memory pool** across the per-bucket graphs.
- **Honesty flags (ADR-0006)**: the exact **triton version** paired with `torch 2.12.0` is **NOT first-party verified** (see §1) — do not pin it. Every other API below is tagged with its source + verified version in §5.

## 1. Version matrix — what pairs with torch 2.12

| Package | Version | How verified |
|---|---|---|
| `torch` | **2.12.0** | ADR-0006 verified-green pin (`pyproject.toml`); the 2.12 release is real (PyTorch 2.12 release blog, via Sonar). |
| `triton` | **unpinned — NOT first-party verified** | We do not pin it (ADR-0006): torch's Linux CUDA wheel resolves it transitively. Sonar found only that torch 2.12 depends on the **upstream `triton` PyPI package** (not `pytorch-triton`); the *exact* version is not stated in first-party release notes. A third-party repackaging (`torch 2.12.0+cu133`) shipped `triton 3.7.0`, but that is **not** an official pin — treat as a hint, not a fact. **Observe the real version with `pip show triton` on the GPU box when T4c lands, then pin.** |

> **Why this matters (ADR-0006)**: R2 fabricated `triton 3.7.1`. The Triton APIs in §2 are verified to exist in the **current Triton `main` docs/source** (Context7 `/triton-lang/triton` + Sonar on triton-lang.org). They are stable, long-standing primitives — but the *version number* torch 2.12 pulls must be read off the machine, never asserted here.

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
