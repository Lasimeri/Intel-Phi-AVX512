# vpu_matmul.c and vpu_matmul.h: the card as a matrix-multiply engine

The service behind the ggml backend (`host/crates/phi-ggml`): four
request kinds on the worker's doorbell, with a descriptor
(`struct vpu_matmul`, 128 bytes) in the control area at
`VPU_OFF_MATMUL`.

- `VPU_K_UPLOAD`: the host has put a tensor's rows in the window; the
  card copies them into its own memory (huge pages while it has them,
  else 4 KiB pages; each mapping remembers its own length and kind, see
  below) and keeps them under an id. A tensor uploaded again under the
  same id replaces the old one.
- `VPU_K_MATMUL_ID`: the same with one matrix per column, ggml's
  `MUL_MAT_ID`, which is where a mixture-of-experts model keeps almost
  all of its weights (the section below).
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

## The rows per chunk, and the diagnostics

`ROW_CHUNK` is 32 rows, measured rather than argued: the host can
override it per request (`reserved[0]`, `phi-vpu matmul-check --chunk`),
and at 4096 x 5120 Q4_K the sweep was 0.780, 0.604, 0.570, **0.552**,
0.605 ms at n 1 for 4, 8, 16, 32 and 64 rows, and 33.6, 31.2, 22.4,
**20.9**, 20.9 ms at n 64. `ROW_CHUNK_MAX` (64) sizes the accumulator
array on the thread's stack.

The probe request (`VPU_MM_PROBE`) also times, across the whole pool at
the request's thread count: `phi_bench` streaming (each thread its own 4
MiB) for the card's aggregate read bandwidth, `phi_bench` register
multiply-adds for its aggregate issue rate, and 1000 empty
`vpu_pool_map` rounds for what a dispatch costs with nothing to do.
`docs/results/2026-09-23-ceilings-and-residency.md` has the numbers and
what they settle (one thread per core; the kernels are issue bound).

The activation row stride (`nb_b`) is the caller's to choose, and it
matters: the kernels take the rows as memory operands, so a stride that
is a multiple of 4 KiB puts every row of a group in the same L1 sets.
The host pads it (`phi-ggml`, `B_PAD` 256 bytes), which is twice the
arithmetic at n 8 and above.

## A mixture of experts (`VPU_K_MATMUL_ID`)

The same multiply with one matrix per column: `a` holds `experts`
matrices of `m` rows one after another, the `n = n_used * n_tokens`
columns each name an expert in the int32 array at `b_off` (`ids_bytes`
of them, the activation rows following), and column `p = j + t *
n_used` multiplies expert `ids[p]` by b's row `(j % b_rows) + t *
b_rows`. An id that would read past the slice is refused before any
thread runs. `rows_slice_id` walks the columns and calls the ordinary
row slicing for each, so a column is one activation row: right for
generation, where every column of a token is a different expert; a
batch's columns are grouped by expert first (below,
`docs/results/2026-09-23-mixture-of-experts.md`).

## Grouping a mixture's columns by expert

A mixture's `n = n_used * n_tokens` columns each name an expert, and
several of them name the same one. `groups_mixture` sorts the columns by
expert (a counting sort, two passes over the ids) and cuts each expert's
run into groups of eight, four or one, so the card reads that expert's
rows once per group instead of once per column: at 512 tokens with eight
experts used that is about 512 groups against 4096 columns. An ordinary
multiply's groups are just its consecutive columns (`groups_plain`), so
one loop serves both.

The kernels take the group's activation rows as an **array of pointers**
(`rdx`), not a stride, because the columns that chose an expert belong
to whichever tokens chose it. That costs nothing: the prologue loads
T - 1 pointers where it used to compute T - 1 addresses
(`kernelgen/quant.md`).

Measured on card 0, the mixture multiply of the 35B-A3B at 512 tokens:
the card's compute fell from 39.2 ms to 20.4 ms. What it did not do is
make the card faster than the host on those multiplies: the host groups
the same way and its kernels work in int8, so per unit of work it stays
about twice as quick, and the backend's judgement still decides case by
case (`host/crates/phi-ggml/src/lib.md`).

## Two threads per core, measured again

With the columns grouped, a thread's working set is about 24 KiB (eight
activation blocks and `chunk` x 8 accumulator vectors) against a 32 KiB
L1, so two threads on a core cannot both hold one. Splitting a group's
columns between them (four each, both walking every row, reading the
same weight bytes) halves that to 12 KiB and needs no recombination,
since the halves write different columns. It is implemented and it is
still not better: q6_K at n 64 gives 188 GFLOP/s against 209 for one
thread per core, at every chunk from 8 to 64.

The reason is that the loop is not short of issue. The same kernel on a
superblock held in L1 goes from 413 GFLOP/s across 57 threads to 668
across 114, so the second thread has issue to spare; the real loop runs
at about 65 percent of that rate because its activation blocks arrive
from L2, and a second thread doubles that traffic. Shrinking the
footprint does not help either: at one thread per core, chunks of 64,
32, 16 and 8 give 206, 208, 197 and 161 GFLOP/s, because what matters is
amortizing each 8 KiB block load over more rows, not fitting in L1.

## Float16 activations

`b_type` in the descriptor (0 float32, 1 float16) says which of two
kernel tables the service uses: `struct qfmt` now holds `k[2][3]`, the
float32 kernels and their `h` twins, indexed by the request's `b_type`
and by the group's row count. A row's stride is then `k * 2` instead of
`k * 4`, and `q8_0_tail`, which finishes the weights past the last whole
superblock on the scalar unit, reads halves or floats by the same flag.

The card refuses `b_type` for a float weight type (`shape_ok`), because
only the generated quantized kernels have float16 twins; the float
formats go through `phi_dot4_*`, which reads float32 rows. The host
sends float32 for them (`host/crates/phi-ggml/src/lib.md`).

Nothing about this costs the card anything: the up-conversion is a field
of the memory operand, so the same 128 fused multiply-adds per eight-row
call read half the bytes (`kernelgen/quant.md` has the rates).

## A feed-forward block in one request (VPU_K_FFN)

`struct vpu_ffn` (192 bytes at `VPU_OFF_FFN`, 13440, just past the
matmul descriptor in the control area) describes one card's share of a
SwiGLU feed-forward block: a run `lo..hi` of the intermediate, resident
as three slices the host uploaded once: `rows` rows of the gate matrix,
the same rows of the up matrix, and the same **columns** of the down
matrix (every one of its `m_out` rows, cut at the run's superblocks, so
it is an ordinary quantized matrix of `m_out` rows of `rows` weights).
One request then computes, for every column c of the activations,

```text
h[c][i] = silu(gate[i] . x[c]) * (up[i] . x[c])      i in the run
d[c][o] = sum over the run of down[o][i] * h[c][i]    every output row o
```

and returns `d`, a partial sum over the run, which the host adds to its
own (`host/crates/phi-ggml/src/ffn.md`). The intermediate never leaves
the card: the request's input is the block's input and its output is
the block's output, and one round trip replaces three.

`ffn_run` checks each of the three as the ordinary multiply it is
(`shape_ok`), then runs two dispatches:

1. `ffn_slice` on every thread: the thread's rows of the run, by single
   rows (the balance any multiply gets), gate then up through
   `rows_range_q` (the quantized loop with an explicit row range; the
   old `rows_slice_q` is now that plus `slice_rows`), then the SwiGLU of
   exactly those rows (`swiglu_range`: whole vectors, and the vector
   holding either end under a lane mask, `kernelgen/glu.md`). No other
   thread writes those rows, so the three steps need no barrier.
2. The down projection over `h`, an ordinary quantized multiply through
   `rows_slice` whose activations are `h`'s columns.

`h` is float32 unless the request says float16 (`h_type` 1): float16
lets the down projection use its faster `h` kernels, but overflows past
65504, and the host does not know the intermediate's range. Its columns
are a quarter of a page longer than the run, for the same L1 set reason
as the host's activation padding.

Measured on card 0 (`phi-vpu matmul-check`, one card's share of the 27B:
gate and up Q5_K 4352 x 5120, down Q6_K 5120 x 4352, compute only, best
of 4), against the three multiplies it replaces with the same work:

| n | fused, float32 h | fused, float16 h | the three |
| --- | --- | --- | --- |
| 1 | 2.620 ms | 2.513 | 2.404 |
| 8 | 4.798 | 4.807 | 4.409 |
| 64 | 35.433 | **33.767** | 36.860 |

The compute is within 9 percent either way; what the fusion removes is
the transport, which the same log puts at about 0.7 ms of pull and 0.9
of push per request at these sizes, paid once instead of three times.
The down projection's shape costs a little (5120 rows of 4352 weights
reads its activations more often per weight than 1280 rows of 17408),
which is the remaining gap at n 1 and 8.

`VPU_MM_SWIGLU` (98) is the SwiGLU kernel on its own for the checker:
`m` floats of g at `b_off`, `m` of u at `b_off + nb_b`; `b_type` 0 runs
`phi_swiglu` over whole vectors, 1 and 2 run `swiglu_range` in float16
and float32 over abutting ranges that start and end inside vectors,
leaving the lanes outside them as 0xdead, so a mask off by one shows.

## How a request's data crosses (2026-09-23 night)

A multiply at one token moves 5 to 35 KB each way, and a request used to
move it with one `pread` and one `pwrite` on the block device. Those are
round trips through the card's block layer and the host's daemon, and at
generation, where requests come every millisecond or so with gaps
between them, each cost 0.2 to 0.35 ms (the stack's block-pipeline note:
an 85 us floor back to back, about 85 more after an idle gap). On the
27B, per token, the slower card spent about 70 ms pulling and 97 ms
pushing against 195 ms computing, and the host waited 65 ms for it.

`pull_data` and `push_data` now move the data through the worker's
mapping of the window instead, in whole 64-byte vectors
(`kernelgen/copy.md`):

| transfer | how |
| --- | --- |
| a pull of 4 KiB or less | one thread's 64-byte loads |
| a push of 16 KiB or less | one thread's 64-byte stores (29 us for 16 KiB) |
| either, up to 2 MiB | split across the pool (`copy_pool`): one load in flight per core, 2.6 GB/s at 1 MiB on card 0 |
| larger | the block device, whose DMA catches up by 4 MiB |

The thresholds are the probe's crossovers, on both cards
(`matmul-check --probe`: the pool against the block device at 16 KiB,
64 KiB, 1 MiB and 4 MiB). The worker's `-m 0` sends everything through
the block device again. Every format, mixture and feed-forward case of
`matmul-check` passes through it on both cards.

On the 27B at one token: the host's wait for the cards fell from 64 to
12 ms per token, the slower card's pulls from about 70 ms to 16 and its
pushes from 97 to 7, and generation went from 1.66 to 1.83 tokens per
second (`docs/results/2026-09-23-redundancy-and-transport.md`).

A mixture's expert ids are checked once before any thread runs, negative
ones included (2026-09-24): the quantized path counts columns by id, and
an id of -1 would have written before its buffer instead of failing the
request.
