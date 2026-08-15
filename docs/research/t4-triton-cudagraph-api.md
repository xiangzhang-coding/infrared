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

Verified constraints [Context7 `/pytorch/pytorch` notes/cuda.md]:

1. **Warmup before capture, on a side stream.** Run the workload a few times so lazy allocations / autotuning settle, then capture. Required pattern:
   ```python
   s = torch.cuda.Stream()
   s.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(s):
       for _ in range(3):
           _ = model(static_in)          # warmup
   torch.cuda.current_stream().wait_stream(s)
   with torch.cuda.graph(g):
       static_out = model(static_in)     # capture
   ```
2. **Fixed shapes + fixed addresses.** Every captured tensor's shape and allocation must be identical at replay. Keep long-lived references to the static in/out buffers.
3. **No new allocations inside the captured region.** Model params + KV cache must already be resident; the graph captures their *use*, not fresh allocs.
4. **No data-dependent shapes/control-flow.** You may branch on values, but shapes must not change; mask padded rows via data, not by reshaping.
5. **Multi-stream rule.** Any side stream used *inside* capture must branch from and rejoin the initial capture stream (`s.wait_stream(...)` / `current_stream().wait_stream(s)`), or capture is rejected.

### 3.4 Variable-length / paged batches under graphs

The problem: CUDA graphs need fixed shapes, but a decode batch's size and context lengths vary every step. The verified real-world pattern (vLLM + nano-vLLM) [Sonar → docs.vllm.ai/design/cuda_graphs + nano-vLLM CUDA-graph walkthroughs]:

- **Decode-only graphs.** Prefill stays eager (variable seq-len, hard to graph); only the fixed-per-step decode is captured. An `enforce_eager` flag disables all graphs for debugging.
- **One graph per batch-size bucket.** Capture graphs for a set of sizes, e.g. `[1, 2, 4, 8, ..., 512]`. Each owns fixed-shape static buffers: `input_ids [B,1]`, `positions [B,1]`, `slot_mapping [B]`, `block_tables [B, max_blocks]`, `context_lens [B]`.
- **Pad up at runtime.** For real batch `b`, pick the smallest bucket `B ≥ b`, `copy_` real data into rows `[:b]`, and pad rows `[b:B]` with sentinels (e.g. `slot_mapping = -1` so the §2.5 scatter skips them; dummy block ids). Then `graph[B].replay()`, read back rows `[:b]`.
- **Variable context length is fine without re-capture.** `context_lens` is *data* in a fixed-shape buffer; the kernel loops to `max_blocks` and masks by `context_len`. Growing history needs no new graph — only crossing a batch-size bucket does.
- **Share one memory pool across buckets** via `pool=` (§3.5) — many graphs, bounded memory; replay order must be respected.

```python
b = len(seqs); B = next_bucket(b)                 # smallest bucket >= b
buf = graphs[B].buffers
buf.input_ids[:b].copy_(input_ids); buf.input_ids[b:].fill_(0)
buf.slot_mapping[:b].copy_(slots);  buf.slot_mapping[b:].fill_(-1)
buf.block_tables[:b].copy_(block_tables)
buf.context_lens[:b].copy_(context_lens); buf.context_lens[b:].fill_(0)
graphs[B].graph.replay()
logits = buf.logits[:b]                            # consume real rows only
```

### 3.5 Sharing one memory pool across the per-bucket graphs

Capturing ~10 batch-size buckets wastes memory if each gets a private pool. PyTorch lets graphs **share a pool** [Context7 `/pytorch/pytorch` notes/cuda.md]:

- **`with torch.cuda.graph(g2, pool=g1.pool()): ...`** — capture `g2` hinting it may reuse `g1`'s pool. `CUDAGraph.pool()` returns the opaque pool handle.
- Explicit pools also exist: **`torch.cuda.MemPool()`** + **`with torch.cuda.use_mem_pool(pool): ...`**; `g.pool()` / `g.pools()` read them back.
- Docs note pool sharing is **"common in inference servers with variable batch sizes,"** but shared-pool graphs must be **replayed in capture order and never concurrently**, or one graph's replay overwrites another's outputs.

```python
graphs = {}; shared = None
for B in buckets:                       # capture in a fixed order
    g = torch.cuda.CUDAGraph()
    ctx = torch.cuda.graph(g) if shared is None else torch.cuda.graph(g, pool=shared)
    with ctx:
        static_logits[B] = model(static_in[B])
    shared = g.pool()                   # subsequent buckets reuse this pool
    graphs[B] = g
```

> Flag: `torch.cuda.graph_pool_handle()` (a standalone way to mint a shareable pool handle) is a real documented helper but I verified only the **`g.pool()` / `pool=` / `MemPool` / `use_mem_pool`** forms above via Context7 — prefer those, or confirm `graph_pool_handle()` on the box first.

## 4. Linux/CUDA-only compat notes (lazy import, no CPU path)

Both surfaces are **GPU-only**; infrared keeps a naive PyTorch attention fallback (the existing `masked_attention` in `infrared/model/layers.py`) as the CPU/no-GPU path (T4c ticket). Rules:

- **Lazy-import triton.** `triton` ships **Linux wheels only** (ADR-0006 / `pyproject.toml` marker `platform_system == 'Linux'`). Never `import triton` at module top level — import inside the GPU code path (or guard with `torch.cuda.is_available()`), so macOS/CPU dev + CI (torch without triton) still import `infrared` cleanly.
  ```python
  def paged_attn(...):
      if not torch.cuda.is_available():
          return masked_attention(...)      # naive fallback (existing seam)
      import triton, triton.language as tl   # lazy, GPU-only
      ...
  ```
- **CUDA graphs are CUDA-only.** `torch.cuda.CUDAGraph` / `graph()` need a CUDA device; gate capture behind `torch.cuda.is_available()` + an `enforce_eager` config flag. On CPU, decode runs eager.
- **Naive path is the correctness oracle.** Triton kernel and fallback must produce matching logits (HF parity, ADR-0005); the fallback is what no-GPU CI exercises.
- **Don't pin triton** (ADR-0006): read the version off the GPU box (`pip show triton`) when T4c lands, then pin.

## 5. API provenance table (source + verified version)

| API | Verified via | Version / commit |
|---|---|---|
| `@triton.jit`, `tl.program_id`, `tl.num_programs`, `tl.constexpr` | Context7 `/triton-lang/triton` (docs/python-api/triton.rst, test_bindings.py) | Triton `main` docs/source |
| `tl.load` / `tl.store` (+ `mask=`, `other=`) | Context7 `/triton-lang/triton` (test_bindings.py) | Triton `main` |
| `tl.arange`, `tl.zeros`, `tl.full`, `tl.where` | Context7 `/triton-lang/triton` (triton.language.rst categories) | Triton `main` |
| `tl.dot`, `tl.sum`, `tl.max`, `tl.maximum`, `tl.exp2` | Context7 `/triton-lang/triton` (triton.language.rst: linalg/reduction/math ops) | Triton `main` |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`, `tl.advance(ptr, offsets)`, `load(..., boundary_check, padding_option)` | Sonar → triton-lang.org/main/python-api + github triton.language.rst | Triton `main` docs |
| Online-softmax shape (m_i/l_i, exp2, rescale-accumulate) | Sonar → triton-lang.org tutorial 06 + github `python/tutorials/06-fused-attention.py` | Triton `main` tutorial |
| vLLM paged-attn grid `(num_heads, num_seqs, max_num_partitions)`, block_table/context_lens/slot_mapping roles | Sonar → docs.vllm.ai paged_attention + vLLM Triton backend deep-dive + arXiv "Anatomy of a Triton Attention Kernel" | vLLM docs (design-level) |
| `store_kvcache` scatter kernel shape | Sonar → nano-vLLM `layers/attention.py` | nano-vLLM source |
| `torch.cuda.CUDAGraph()`, `torch.cuda.graph(g)`, `g.replay()`, `static.copy_()` | Context7 `/pytorch/pytorch` (notes/cuda.md) | PyTorch docs `main` (API stable across 2.x; used under torch 2.12.0) |
| Side-stream warmup + multi-stream capture rule | Context7 `/pytorch/pytorch` (notes/cuda.md) | PyTorch docs `main` |
| `torch.cuda.make_graphed_callables(callable, sample_args)` | Context7 `/pytorch/pytorch` (notes/cuda.md, torch.compiler_cudagraph_trees.md) | PyTorch docs `main` |
| Pool sharing: `torch.cuda.graph(g, pool=g1.pool())`, `CUDAGraph.pool()/pools()`, `torch.cuda.MemPool()`, `torch.cuda.use_mem_pool()` | Context7 `/pytorch/pytorch` (notes/cuda.md) | PyTorch docs `main` |
| Bucketed decode graphs + pad-up + prefill-eager + `enforce_eager` | Sonar → docs.vllm.ai/design/cuda_graphs + nano-vLLM CUDA-graph walkthroughs | vLLM/nano-vLLM (design-level) |

> **Version honesty (ADR-0006)**: Triton/PyTorch API entries are verified to *exist in current first-party docs/source*; they are long-standing primitives, but Context7/Sonar pull from `main`, so I do **not** claim a per-symbol "added in vX" line. `torch==2.12.0` is the repo's verified-green pin; the exact **triton** version is unverified (§1).

## 6. What I could NOT verify (honest gaps)

---

_↩ Back to tracking issue: [infrared#10 — R3 · Research: Triton paged-attn kernel API + torch CUDA graphs API](https://github.com/xiangzhang-coding/infrared/issues/10)_
