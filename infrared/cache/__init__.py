"""KV cache — PagedAttention-style block manager + physical KV tensors.

Contents (filled at T3 — see ``docs/spec/0001`` and R1 §4/§5):

- ``block_manager`` — ``Block`` + ``BlockManager`` (free deque + used set +
                      ref_count; can_allocate/append + deallocate; prefix cache
                      deferred to T4).
- ``kv_cache``      — physical KV tensor layout and profile-based sizing.

The block table (logical→physical page table) and ``slot_mapping`` are the data
structures T3/T4 hang the paged-attention kernel on (R1 §4.2/§9).
"""
