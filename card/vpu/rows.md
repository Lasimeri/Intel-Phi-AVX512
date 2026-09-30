# rows.S: the row loops of the matrix service

What a thread of the pool does with its slice of a request: which rows
are its own, how the request's columns are grouped so that a weight row
is read once per group, what it asks the cache for before it starts, its
share of a fused pull of the activations, and the loop that multiplies
its rows with the kernels of `kernels.S`. The design is the C worker's
(`vpu_matmul.md`: "Grouping a mixture's columns by expert", "One token's
multiply, and whose lines the pool reads", "Two threads per core");
this document says what the assembly keeps and what it must reproduce
exactly.

- `slice_rows`: rows `i0..i1` of slice `s` of `n`: with 57 slices every
  core has one; with 114 (two threads a core) the slices `s` and `s + 57`
  share a core and take alternate rows of the same chunk, so the
  activation block they read is one and the same in their shared L1.
- `groups_plain`, `groups_mixture`, `groups_own`: the column groups
  (`struct group` in `matmul.inc`: a matrix, up to eight activation rows
  as pointers, their destination columns, the count 1, 4 or 8). An
  ordinary multiply's are its consecutive columns; a mixture's are the
  columns that chose each expert (a counting sort), so an expert's rows
  are read once per group rather than once per column. A request of a
  few columns (`OWN_GROUPS_MAX`) has each thread build the groups it
  reads in its own region (`groups_own`, an insertion sort up to 32
  columns, the counting sort above), rather than the dispatcher one
  table for all, whose lines had been read by every core the request
  before. An id of -1 is a column without an expert on this card (the
  host's whole-expert placement): no group.
- `warm_slice`: a small slice's whole working set asked for before it
  runs (`vprefetch1`: the card's cores are in order and every cold line
  costs its whole miss); the table and the activation rows are the same
  lines for every core, so each starts on a different line of them.
- `copy_share`, `fused_setup`, `copy64_nt`: a small request's
  activations copied inside its compute dispatch by the first cores'
  threads (at least `PULL_LINES` lines each: a row copied a line or two
  per core and then read by all 57 was seven times slower than one
  written by one core), counted so the others can wait for the last
  copy. The copy's stores are the no-read, non-globally-ordered kind
  (`vmovnrngoaps`, ordered by the locked add that counts the copy): a
  plain store leaves the line dirty in the copying core's L2, from where
  fifty-six readers then take it one at a time, and the request is slow
  or fast by whether the lines happened to be written back before the
  readers came (below, "The slow mode").
- `vpu_phase`: each slice's clock after the copy wait, after the
  activation warm-up and after its rows, for the `phases` line the
  matrix service prints at `-v -v` (`matmul.md`).
- `rows_range_q`: rows `i0..i1` of one quantized matrix by every group:
  chunks of `ROW_CHUNK` rows; the groups outermost so their activation
  rows are read from L2 for every weight row of the chunk; the
  superblock outermost within a group so its activation block and the
  accumulators stay in L1 while the weights stream; a kernel call per
  row and superblock with the seventh argument (the distance to the row
  this thread does next, which the kernels' prefetch follows) pushed
  around the call; then the sums. Two threads on a core divide an
  eight-column group by columns (each walks every row, taking four of the
  eight) rather than by rows, so they read the same weight bytes.
- `run_pieces`, `rows_slice_q`, `rows_slice_multi`: a slice's rows of one
  matrix, or of each matrix of a request of several, cut as one.

## The sums: the x87 unit, as the C compiled

A result is the sixteen partial sums of its accumulator vector added
together, plus, for a Q8_0 row whose length is not a multiple of 256, the
row's last blocks done here. The C worker wrote both as plain C float
arithmetic, which the card toolchain compiles to the x87 unit (the
`knc64-x87` ABI: no SSE), and in x87 a chain of additions stays in the
80-bit register until it is stored, rounding once. The card's results
therefore depend on exactly that: `fldz`, sixteen `fadds` in lane order,
then for a tail `s = sum over 32 of xv * q` per block (`fimuls` by the
signed byte widened to 16 bits, as clang emitted) and `tail += d * s`,
then `sum += tail`, then one `fstps` into the result. `rows_range_q`
and `matmul.S`'s float path do exactly this, read off the compiled C
(`knc-cc -O2 -S vpu_matmul.c`, 2026-09-30), so that a result is the same
float to the bit, and a model's text under the assembly worker is the
same byte for byte as under the C one. A vector reduction would be
faster and would round differently; it is not used for that reason.

## Registers and frames

The pool calls a slice function as `fn(arg, slice, nslices)` in `rdi`,
`esi`, `edx` (System V), and the kernels are called as C would call
them (`kernels.md`). `rows_range_q` keeps its frame (the accumulators,
the scratch table, the row pointers, its variables) 64-byte aligned
below `rbp`, which is not the standard frame pointer here but the
aligned base, with the pre-alignment stack pointer saved in the frame.
Its row loop carries the superblock pointer and the accumulator pointer
in registers the kernels keep (`r14`, `r15`, the count in `r13`) and
adds the row stride to them; the first version recomputed both with
three multiplies and a dozen frame accesses per kernel call, which on
this in-order core was about 100 ns on top of a 280 ns kernel (the
2048-row mixture shape: 1.03 ms against the C's 0.90, now 0.86).
`run_pieces` takes its last two arguments on the stack as the C did.
Everything is written for the card's scalar core: no SSE, no `cmov`.

## The slow mode (2026-09-30)

With the C worker's plain stores in the fused copy, the one-row mixture
shape (128 x 2048, eight experts) computed in 95 us at best but 190 us at
the median, where the C worker's median was 105 to 165 us; the per-slice
phases showed every thread's row loop three to four times slower in the
slow requests, with no dependence on which experts were drawn, and the
mode absent with `-m 0` (the activations arriving by DMA into memory).
The activation row is written by the eight copying threads and read by
every other core right after: a line dirty in a copier's L2 is served
from there, one reader at a time, and whether the request is slow is
whether those lines were still dirty when the readers came. The no-read
stores leave nothing in the copier's L2; the median fell to 93 us over
two restarts (270 of 300 requests under 100 us), the best unchanged.
The C worker has the same exposure with a different luck of timing.

## Gates

Step 1c (2026-09-30, card 1, the C worker on card 0): `phi-vpu
matmul-check` passes every case for f32, f16, q4_K, q5_K, q6_K, q8_0 and
iq4_xs: the plain multiplies at 1, 4, 8 and 13 columns, the mixtures
(including two with every third column at -1 through both groupings),
up to four matrices in one request, the fused feed-forward with both
intermediates; the full run's text is compared with the C worker's in
`matmul.md`. Defects met: `groups_own`'s many-column path returned a
stale register as the group count (the walk ran off the table on the
first 64-column mixture); the down projection of a feed-forward request
took its matrix from a register two calls had clobbered (a general
protection fault in the one-row Q6_K kernel on a garbage superblock
address).
