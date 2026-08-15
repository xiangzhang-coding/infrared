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

- **`@triton.jit`** — decorator that compiles a Python fn into a GPU kernel. `constexpr` params are compile-time (tile sizes). [Context7 `/triton-lang/triton`, `docs/python-api/triton.rst`]
- **`tl.program_id(axis=0|1|2)`** / **`tl.num_programs(axis)`** — this program's coordinate in the launch grid. Paged-attn grids over `(head, seq, [q_block])`; vLLM documents the decode grid as `(num_heads, num_seqs, max_num_partitions)`. [Context7 test_bindings.py; Sonar → docs.vllm.ai paged_attention]
- **Launch / grid** — call the kernel as `kernel[grid](args..., BLOCK=..., num_warps=..., num_stages=...)` where `grid` is a tuple (or a lambda of `meta`). Kernels return nothing; they write via output pointers. [Context7 `/triton-lang/triton`]
- **`tl.constexpr`** — annotate tile sizes / compile-time constants; math constants (e.g. `log2e: tl.constexpr = 1.44269504`) must be assigned to a `constexpr` var inside the kernel. [Context7 test_bindings.py]


### 2.2 Loads, stores, block pointers, masking

Two addressing styles, both real; paged-attn uses **both**:

- **Plain pointer + `tl.arange` + mask** (the paged-gather path). Build offsets `offs = base + tl.arange(0, BLOCK)`, then `tl.load(ptr + offs, mask=offs < n, other=0.0)` and `tl.store(ptr + offs, val, mask=...)`. Masking is how you (a) stay in-bounds on ragged tails and (b) keep padded/invalid KV out of the math. This is what a paged kernel uses to gather non-contiguous blocks via a `block_table`. [Context7 test_bindings.py; Sonar → triton-lang.org]
- **Block pointer** (the contiguous Q/O/tile path). `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` returns a tiled view; `tl.load(blk_ptr, boundary_check=(0,1), padding_option="zero")` reads a tile with auto bounds-padding; `tl.advance(blk_ptr, offsets)` slides it to the next tile; `tl.store(blk_ptr, val, boundary_check=...)`. Verified signature: `triton.language.make_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic=None)`; `tl.advance(ptr, offsets)`. [Sonar → triton-lang.org/main/python-api/generated/make_block_ptr; github triton.language.rst]

> **Key shape decision**: paged KV is scattered across physical blocks, so `make_block_ptr` (which assumes a single strided base) does **not** describe the KV gather — vLLM's paged kernels use computed offsets + masked `tl.load` over the `block_table` for KV, and reserve block pointers for the contiguous query/output tiles. [Sonar → vLLM Triton backend deep-dive; arXiv "Anatomy of a Triton Attention Kernel"]

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
