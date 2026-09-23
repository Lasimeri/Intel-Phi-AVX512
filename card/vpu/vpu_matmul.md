# vpu_matmul.c and vpu_matmul.h: the card as a matrix-multiply engine

The service behind the ggml backend (`host/crates/phi-ggml`): three
request kinds on the worker's doorbell, with a descriptor
(`struct vpu_matmul`, 128 bytes) in the control area at
`VPU_OFF_MATMUL`.

- `VPU_K_UPLOAD`: the host has put a tensor's rows in the window; the
  card copies them into its own memory (huge pages while it has them,
  else 4 KiB pages; each mapping remembers its own length and kind, see
  below) and keeps them under an id. A tensor uploaded again under the
  same id replaces the old one.
- `VPU_K_MATMUL`: `d[n][m] = a[m][k] . b[n][k]`, ggml's `MUL_MAT`
  (the result transposed, as ggml lays it out). `a` is a resident tensor
  (`a_id`) or, for a tensor the host does not keep, in the window; `b`
  (float32, n rows) comes through the window every time; `d` (n rows of
  m floats, contiguous) goes back through it. The rows of `a` are split
  across the pool's threads (`vpu_pool_map`).
- `VPU_K_FREE`: drop a tensor, or all of them.

Weight types (`VPU_MM_*`, `a_type`): float32 and float16 rows use the
dot kernels `phi_dot*_f16/f32` (one row against one or four activation
rows, float16 up-converted by the load). The quantized formats of
llama.cpp, Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS, use the superblock kernels
`phi_<fmt>_<1|4|8>` (`host/crates/phi-vpu/src/bin/kernelgen/quant.md`):
one 256-weight superblock against 1, 4 or 8 activation rows, the format
decoded on the vector unit, the scales too. The reference each format
reproduces is ggml-quants.c's `dequantize_row_<fmt>`; `phi-vpu
matmul-check` compares the card with it for every type and row count.

The quantized loop (`rows_slice_q`): the activation rows go in groups
of 8, 4 or 1 (the kernels' variants), the group outermost; within a
group the thread's rows go in chunks of 16 and the superblock index is
the outer loop of the chunk, so the 1 KiB activation block per
superblock and the chunk's accumulators stay in L1 while the weights
stream (the kernels prefetch two and four rows ahead; the seventh
argument carries the stride). With 114 or 228 pool threads the threads
of one core work on the same chunk at the same superblock, alternating
rows (`slice_rows`), sharing the activation block in the one L1; in
practice 57 threads, one per core, is fastest for every shape measured
(`docs/results/2026-09-23-quantized-kernels.md`). A Q8_0 row whose
length is not a multiple of 256 finishes with a scalar tail.

`VPU_MM_PROBE` (99) is a diagnostic: `phi_probe` writes what the
kernels' instructions produce into `d`, and `phi_bench` times the raw
issue and streaming rates and the Q4_K kernels in L1 and streaming;
`phi-vpu matmul-check --probe` prints them.

The window offsets must be whole 4 KiB blocks (the card reads and
writes the window with O_DIRECT); the host rounds lengths up and the
card's buffers are sized to the rounded lengths, plus 64 bytes of slack
past every buffer, which the unaligned load pairs may read.

Mappings: `big_alloc` tries a huge-page mapping first and falls back to
4 KiB pages; `struct mapping` records the length actually mapped, and
`big_free` unmaps exactly that. The first version unmapped the huge
rounded length whatever the mapping was, and once the huge pages ran
out (2026-09-23, card 1, 185 resident slices in) a free of a small-page
buffer unmapped the pool threads' stacks with it.

Limits (the host checks them before offering an op to ggml): a tensor
slice up to 512 MiB, activations and results up to 64 MiB per multiply,
4096 resident tensors, k up to 65536 for the quantized formats. The
card's memory is the limit for residency, about 3.5 GB of weights per
card beside the worker.

Not done: batched and broadcast multiplies (attention's, which ggml
keeps on the CPU), the other quantized types (Q3_K, IQ4_NL, IQ3_S stay
on the host), a reduction tree matching the host's summation order
(results differ from the CPU's in the last bits, as any two
implementations do).
