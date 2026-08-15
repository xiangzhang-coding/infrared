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

- **`tl.dot(a, b)`** — the tiled matmul primitive; used twice per block: `qk = tl.dot(q, k)` (scores) and `acc += tl.dot(p, v)` (weighted values). Listed under `triton.language` Linear-Algebra ops. [Context7 `/triton-lang/triton` triton.language.rst]
- **Online softmax** (FlashAttention-2 style, verified against Triton tutorial 06). Keep per-query-row running state `m_i` (max) and `l_i` (sum-of-exp); stream over KV blocks without ever building the full score row. Uses `tl.maximum`, `tl.max(.., axis=1)`, `tl.exp2`, `tl.sum` — all in `triton.language` math/reduction ops. [Sonar → triton-lang.org tutorial 06 + github `python/tutorials/06-fused-attention.py`; Context7 triton.language.rst]

```python
# per-row running state
m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
l_i = tl.zeros([BLOCK_M], tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
qk_scale = sm_scale * 1.44269504  # 1/ln2, so exp2 replaces exp

for blk in range(num_kv_blocks):          # walk this seq's block_table
    k, v = load_kv_block(...)              # masked tl.load via block_table
    qk = tl.dot(q, k) * qk_scale           # [BLOCK_M, BLOCK_N]
    qk = tl.where(valid_mask, qk, -float("inf"))   # causal / context-len mask
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))     # new running max
    p = tl.math.exp2(qk - m_ij[:, None])           # rescaled probs
    alpha = tl.math.exp2(m_i - m_ij)               # correction for old state
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_ij

acc = acc / l_i[:, None]                    # final normalize
```

> Multiplying the scale by `1/ln2 ≈ 1.44269504` lets the kernel use hardware `exp2` instead of `exp` (a standard tutorial-06 trick). The `-inf` masking + running-max subtraction is exactly the numerical-stability guarantee the ticket asks for. [Sonar → triton-lang.org tutorial 06]


### 2.4 Minimal GPU-compilable paged-attn kernel skeleton

**Shape, not copy (ADR-0004).** Every API call below is a real, verified primitive (§5); the *assembly* is illustrative and must be `triton.jit`-compiled + parity-tested on a GPU. Matches infrared's seam (`PagedKVPool` flat `[layers, num_blocks*block_size, kv_heads, head_dim]`, `write_slots`=slot_mapping, per-row history via `block_table`).

```python
import triton
import triton.language as tl

@triton.jit
def paged_decode_attn(
    q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,          # tensors
    block_tables_ptr, context_lens_ptr,              # paged metadata
    scale,
    stride_qs, stride_qh, stride_qd,                 # q strides (row-major)
    stride_kb, stride_kh, stride_kd,                 # k/v cache slot strides
    stride_bt_s, stride_bt_b,                        # block_tables strides
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
):
    seq = tl.program_id(0)                            # one program per (seq, head)
    head = tl.program_id(1)
    ctx_len = tl.load(context_lens_ptr + seq)         # how much history is valid

    # load this seq/head's single decode query row -> [HEAD_DIM]
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq * stride_qs + head * stride_qh + d * stride_qd)
    qk_scale = scale * 1.44269504                      # 1/ln2 -> use exp2

    m_i = -float("inf"); l_i = 0.0
    acc = tl.zeros([HEAD_DIM], tl.float32)

    for blk in range(MAX_BLOCKS):
        blk_start = blk * BLOCK_SIZE
        # block_table[seq, blk] -> physical block id -> flat slot base
        phys = tl.load(block_tables_ptr + seq * stride_bt_s + blk * stride_bt_b)
        slots = tl.arange(0, BLOCK_SIZE)
        tok_pos = blk_start + slots
        valid = tok_pos < ctx_len                      # mask past real context
        slot_base = phys * BLOCK_SIZE + slots          # flat physical rows

        # gather K/V tile for this physical block (masked, non-contiguous)
        k = tl.load(k_cache_ptr + (slot_base[:, None] * stride_kb
                    + head * stride_kh + d[None, :] * stride_kd),
                    mask=valid[:, None], other=0.0)     # [BLOCK_SIZE, HEAD_DIM]
        v = tl.load(v_cache_ptr + (slot_base[:, None] * stride_kb
                    + head * stride_kh + d[None, :] * stride_kd),
                    mask=valid[:, None], other=0.0)

        qk = tl.sum(q[None, :] * k, axis=1) * qk_scale  # [BLOCK_SIZE] scores
        qk = tl.where(valid, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.math.exp2(qk - m_ij)                     # [BLOCK_SIZE]
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    acc = acc / l_i
    tl.store(o_ptr + seq * stride_qs + head * stride_qh + d * stride_qd, acc)
```

- Grid: `paged_decode_attn[(num_seqs, num_heads)](...)`. For long contexts, add a 3rd `program_id` axis to split the block loop into partitions (vLLM's `max_num_partitions`) and combine partials — deferred; the single-program loop above is the minimal correct shape. [Sonar → docs.vllm.ai paged_attention]
- GQA: map `head` → `kv_head = head // n_rep` when indexing K/V. (infrared already does `repeat_kv`; the kernel folds it into the index.)
- Uses `tl.sum(q*k)` rather than `tl.dot` because decode has a single query row (a mat-vec); a **prefill** kernel with `BLOCK_M>1` query rows uses `tl.dot(q, k)` per §2.3. Both `tl.sum` and `tl.dot` are verified `triton.language` ops.

### 2.5 The `store_kvcache` (scatter) kernel skeleton

Before attention reads history, this step's rotated K/V must be scattered into the pool at `slot_mapping` (infrared's `write_slots`). One `@triton.jit` kernel, one program per token:

```python
@triton.jit
def store_kvcache(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    stride_t, stride_h, stride_d,                     # source [N, kv_heads, head_dim]
    stride_cb, stride_ch, stride_cd,                  # cache slot strides
    KV: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    tok = tl.program_id(0)                            # one token per program
    slot = tl.load(slot_mapping_ptr + tok)            # flat physical slot
    if slot < 0:                                      # -1 = padding row, skip
        return
    hd = tl.arange(0, KV * HEAD_DIM)
    k = tl.load(k_ptr + tok * stride_t + hd)          # this token's K/V
    v = tl.load(v_ptr + tok * stride_t + hd)
    tl.store(k_cache_ptr + slot * stride_cb + hd, k)
    tl.store(v_cache_ptr + slot * stride_cb + hd, v)
```

- Grid: `store_kvcache[(num_tokens,)](...)`. Prefill scatters a whole prompt; decode scatters one slot per sequence. This is the Triton analogue of infrared's `PagedKVPool.write` naive scatter. [Sonar → nano-vLLM `layers/attention.py` `store_kvcache` shape]
- `slot < 0` guard lets CUDA-graph decode pad the batch with sentinel `-1` slots (§3.4) without corrupting the pool. All calls (`tl.program_id`, `tl.load`, `tl.store`, `tl.arange`) are verified `triton.language` ops (§5).



### 3.1 Low-level `torch.cuda.CUDAGraph` + `torch.cuda.graph(...)`

Real API (all from PyTorch's official CUDA notes, `docs/source/notes/cuda.md`, via Context7 `/pytorch/pytorch`):

- **`g = torch.cuda.CUDAGraph()`** — the graph object.
- **`with torch.cuda.graph(g): ...`** — capture context; it auto-sets a side stream as current for the duration.
- **`g.replay()`** — re-execute the captured work on the *same* memory addresses.
- **`static_input.copy_(new_data)`** — the only way to feed new data: overwrite the captured buffers in place, then `replay()`.

```python
g = torch.cuda.CUDAGraph()
static_in = torch.empty((5,), device="cuda")

# warmup on a side stream (see §3.3), then capture
with torch.cuda.graph(g):
    static_out = static_in * 2

static_in.copy_(torch.full((5,), 3, device="cuda"))
g.replay()          # static_out now holds 6s
static_in.copy_(torch.full((5,), 4, device="cuda"))
g.replay()          # static_out now holds 8s
```

[Context7 `/pytorch/pytorch` notes/cuda.md]

### 3.2 `make_graphed_callables`

- **`torch.cuda.make_graphed_callables(callable, sample_args)`** — wraps an `nn.Module` (or callable) so its forward/backward run as a graph, while it manages its own internal `CUDAGraph` objects + warmup for you. `sample_args` must match the real inputs' shapes and `requires_grad` state. [Context7 `/pytorch/pytorch` notes/cuda.md]
- Passing a **tuple of callables** lets them **share one memory pool** (the inference-server variable-batch pattern), but they must then be invoked in the same order they were captured — outputs live in burned-in addresses. [Context7 `/pytorch/pytorch` notes/cuda.md + torch.compiler_cudagraph_trees.md]

```python
graphed = torch.cuda.make_graphed_callables(module, (sample_input,))
out = graphed(real_input)   # forward runs as a graph
```

> For infrared's decode (inference-only, no autograd), the **low-level `CUDAGraph` + explicit static buffers** (§3.1/§3.4) gives more control over the paged metadata than `make_graphed_callables`; the latter is the quick path when a plain module just needs graphing.

### 3.3 Static-input-buffer constraints + warmup
### 3.4 Variable-length / paged batches under graphs

## 4. Linux/CUDA-only compat notes (lazy import, no CPU path)

## 5. API provenance table (source + verified version)

## 6. What I could NOT verify (honest gaps)

---

_↩ Back to tracking issue: [infrared#10 — R3 · Research: Triton paged-attn kernel API + torch CUDA graphs API](https://github.com/xiangzhang-coding/infrared/issues/10)_
