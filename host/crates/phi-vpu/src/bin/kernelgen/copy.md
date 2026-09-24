# copy.rs: moving a request's data through the mapping, a vector at a time

`phi_copy64(dst, src, count)` copies `count` 64-byte vectors with whole
`vmovaps` loads and stores, four at a time. It exists for one reason:
the card worker's mapping of the host window (`/dev/phihost`) is
uncached (`pgprot_noncached`, the stack's kernel patch 0026), so every
access is its own transaction across the link and the width of the
access is the width of the transaction. The card's `memcpy` moves at
most 8 bytes at a time (Knights Corner has no SSE); a 64-byte vector
store moves eight times the bytes for the same transaction.

Measured on card 0 with `phi-vpu matmul-check --probe`, 2026-09-23:

| through the mapping, card to host | 16 KiB |
| --- | --- |
| `memcpy` (8-byte stores) | 224 us, 73 MB/s |
| `phi_copy64` (64-byte stores), one thread | **29.4 us, 557 MB/s** |
| the block device, back to back (its best case) | 94 us |

Loads are round trips the in-order core waits for, 87 MB/s from one
thread, but the pool can have one in flight per core: split across 57
threads (`copy_pool` in `card/vpu/vpu_matmul.c`) the same copy reaches
2.6 GB/s at 1 MiB, ahead of the block device up to about 2 MiB. That is
what the matrix-multiply service now moves its activations and results
with (`card/vpu/vpu_matmul.md`, "How a request's data crosses").

Uncached stores are strongly ordered, so the reply the worker writes
after a copy cannot be seen by the host before the data.
