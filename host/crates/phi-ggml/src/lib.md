# lib.rs: the cards beside the host, one multiply at a time

`libggml_phi.so`, loaded by an unmodified llama.cpp through
`GGML_BACKEND_PATH` (`scripts/phi-ggml.sh` sets it). ggml's scheduler
gives the backend every `MUL_MAT` and `MUL_MAT_ID` whose weight type
and shape `csrc/ggml-phi.c` accepts; this file shares each one by rows of the
weight matrix between the host and the cards, **when the cards are worth
using for it**, which it decides per weight tensor from what it measures
("Mixtures of experts, and when a card is worth using" below: a card
costs about 0.45 ms to involve, so a multiply the host finishes sooner
stays with the host and this table does not apply to it):

| who | rows | how |
| --- | --- | --- |
| the host | the first `r0`, and at prompt sizes the gaps below | ggml's own CPU kernels on a private CPU backend (the C glue), started after the cards |
| each card | a share of the top rows, resident in its memory | uploaded once (`plan`), then `K_MATMUL` on the card's threads with the kernels of `card/vpu/vpu_matmul_kernel.S` |

The three calls of one multiply are `phi_ggml_begin` (plan and upload
the weight's shares on first sight; copy the activations into each
card's window; ring the doorbells; report the host's row ranges through
`phi_ggml_host_range`), the host's own rows (in C, while the cards
work), and `phi_ggml_end` (wait for each card's reply, copy its rows
into the result with the result's row stride). The card's per-call floor
(two DMA round trips, about 0.5 ms) hides under the host's part as long
as the host's part is longer, which for a 27B model it is
(`docs/results/2026-09-23-quantized-kernels.md`); for a 0.5B model it is
not, and the backend now keeps out of the way there rather than being
slower than the CPU alone ("Two things decided before any measurement"
below: 798 tokens per second of prompt processing against the CPU's
768, where the first version of this managed 11.3 against 19.4).

Shares: a fraction of every weight matrix's rows per card, sized here to
fill the cards' budget (below, "The share is sized from what is offered";
`PHI_GGML_FRACTION` fixes it instead), the card's rows a multiple of 64
so the host's remainder keeps ggml's
fast paths (measured 0.43 against 7.7 ms for an odd count), until a card
has `PHI_GGML_CARD_BYTES` (4.4 GB) resident or refuses an upload, after
which the card keeps nothing more of later tensors. At eight activation
rows or more (a prompt) each card computes only a share of its slice and
the host the rest of it as another range: the cards are slower per flop
than the host is, and faster per weight byte, so the two cases want
different shares. Only tensors ggml names `*.weight` are shared;
anything else the host does whole.

**That share is measured, not set.** A multiply is over when the slower
of its two sides is over, so the share to aim at is the one that
finishes them together, and both sides are already timed: the host's
part here and each card's own total in its reply. Every batch multiply
adds its relative gap to a running sum; every eighth, the share moves
by half the average gap and the sum resets; after 24 moves it is left
alone (`PP_WINDOW`, `PP_STEPS`, `PP_MIN`). `PHI_GGML_PP_SHARE` (0.75) is
where it starts and `PHI_GGML_PP_ADAPT=0` freezes it there, which is
what comparing two fixed shares needs.

It is one share for the model rather than one per tensor, which is not
an approximation but the only version that can learn: a prompt pass
visits each weight tensor about three times and makes some hundreds of
batch multiplies in total, and what is being learned is a property of
the two sides rather than of a tensor. Both wrong versions were built
and measured first (`docs/results/2026-09-23-share-and-fusion.md`): per
tensor per call oscillates on the 3.7x pool-wake variance, per tensor
over a window never fills it.

Measured only where one batch size runs throughout, which is what
`llama-bench` does. One number serves every batch of eight rows or more,
so a server mixing long prompts with short ones feeds them all to the
same estimator, and after `PP_STEPS` moves it stops wherever the last
windows left it. That is untested here; no server measurement exists to
say whether it wants classes, and none has been invented. `PP_ADAPT=0`
pins the old behaviour if it turns out to.

`PHI_GGML_CARDS` names the cards (a comma list of indices; default every
card whose window exists), `PHI_GGML_THREADS` the card threads per
multiply (57; the pool must be that size, more spins), `PHI_GGML_VERBOSE`
prints every multiply with the host part, the wait, and each card's
pull, compute and push. The host side's thread count is the glue's
(`PHI_GGML_HOST_THREADS`, 12): the card daemons serve the DMA on the
host, and a host thread sharing their CPU stalls ggml's barrier for a
timeslice (7 ms per multiply at 15 threads, measured 2026-09-23).

**The program's own thread count has to come down too**, and by more
than that number suggests. Everything a model does that is not a matrix
multiply stays with the calling program's threads, and those contend
with the card daemons the same way. On the 27B at one token
(`llama-bench -n 8`, both cards, 2026-09-23):

| threads given to llama.cpp | tokens per second |
| --- | --- |
| 12 | **1.53** |
| 14 | 1.01 |
| 16 | 0.29 |

against 1.00 for the host alone at 16. Sixteen is a 5x loss, not a few
percent, because every one of the 419 multiplies a token costs waits on
a barrier that a descheduled thread holds. `scripts/phi-ggml.sh` does
not set `-t` for the program it runs, so a benchmark or a server has to:
give the cards' host 12 of this machine's 16 hardware threads. Any
measurement that leaves it at the program's default is measuring
oversubscription.

Window layout per card: the tensor slice being uploaded at 128 MiB (up
to 512 MiB), the activations at 640 MiB (up to 64 MiB), the result at
704 MiB (up to 64 MiB), all above the seamless path's areas, so both can
share a worker (`phi_vpu::matmul`).

`ggml_backend_init` is exported here (a `cdylib` exports only what Rust
declares) and returns the registration built in C.

`PHI_GGML_CARD_BYTES` is 4.4 GB per card now, which is what a 5.5 GiB
card holds beside its worker when the seamless path's page pool is left
empty (`phi-vpu-worker -e 0`, which `scripts/phi-ggml.sh` passes). For
the 27B this is the binding constraint on generation: the host is the
long pole in every multiply and the cards wait, so what they hold is
what they contribute
(`docs/results/2026-09-23-ceilings-and-residency.md`).

The activation rows go into the window a quarter of a page further apart
than the tensor's own rows (`B_PAD` 256 bytes, and the card is told that
stride). The card's kernels take those rows as memory operands, and at
k 5120 the tensor's stride is 5 x 4096: eight rows then take the same
sets of a 64-set L1 and the eight-row kernels run at a quarter of their
instruction count. This is worth about twice the card's prompt-size
arithmetic (127 to 240 GFLOP/s at n 64) and nothing at n 1, where one
row cannot conflict with itself. The batch share followed it from 0.5 to
0.75, and then stopped being a constant at all (above).

## Mixtures of experts, and when a card is worth using

`phi_ggml_begin_id` is the same three steps for ggml's `MUL_MAT_ID`,
which is what an MoE model's expert weights go through: `a` holds one
matrix per expert, `ids` names the expert each column wants, and each
card keeps the same rows of every expert (one upload, the experts one
after another, so the expert an id names is `rows * nb_a` into the
slice). The host computes its own rows with ggml's own MUL_MAT_ID over a
leaf alias, so the two sides need no agreement about which expert a
column chose: each does its rows of whatever the id says.

A card costs about 0.45 ms to involve at all, and an MoE layer's
multiply at one token is a few megabytes, which the host finishes
sooner. So the backend judges every weight tensor by what it measures,
at one token and at a batch separately: the host did `1 - share` of the
rows in `t_host`, so alone it would have taken `t_host / (1 - share)`;
a multiply that took longer than that with the cards, plainly longer
once or barely longer twice, stops going to them at that batch size
(`Split::avoid`, counted in `too_small`). This is what keeps a mixture's
small multiplies on the host at generation and its batched ones there
too, for now: the card computes a mixture one column at a time where
ggml groups the tokens that chose the same expert
(`docs/results/2026-09-23-mixture-of-experts.md`).

Note when measuring: the host's own best thread count is not the same
for every model, and the comparison must use it. An MoE at one token is
faster on 12 host threads than on 16 (7.91 against 6.39 tokens per
second on the 35B-A3B), so a split measured at 12 against a host at 16
flatters itself by 24 percent. `--mmap 0` matters too, on a host whose
memory the model and the card windows together fill.

## Two things decided before any measurement

The judgement above costs one or two bad multiplies per tensor to
learn, which is nothing on a model whose tensors are visited thousands
of times and the whole cost on a model visited three times. Two cases
are settled without it, both from measurements already in hand rather
than from a rule about shapes:

- **`PHI_GGML_MIN_BYTES` (4 MB).** The weights a multiply would take off
  the host, its rows once or once per column for a mixture. Below this
  there is nothing for a card to win: a round trip is 0.45 ms and the
  host reads weights far faster than 10 GB/s. Measured on the 27B, this
  is worth 2 to 4 percent by itself, because the marginal small
  multiplies it keeps at home were ones the judgement would have taken
  two calls to refuse.
- **Float weights at a batch.** At one token the card's float path is
  its best case (52.5 GB/s of weights against 21 for Q4_K: nothing is
  decoded), but at n 64 it reaches 34 GFLOP/s against 240 for the
  quantized kernels, because `phi_dot4_*` was never restructured the way
  they were. So float weights are shared at generation and left whole at
  prompt sizes. On a 0.5B float16 model this is the difference between
  768 and 798 tokens per second of prompt processing, against 460 with a
  variance of 330 while the judgement learned it one run at a time.

Both are conservative: neither refuses anything a card could have won,
and everything above them is still judged by measurement.

## The activations cross as float16

For a quantized weight type the rows of `b` are converted to float16
before they are copied into a card's window (`to_f16`, F16C on this
host, about one instruction per two elements; `have_f16c` checks the
CPUID bit and the conversion falls back to float32 without it). The card
reads them with `{float16}` on the memory operand, which costs it
nothing (`card/vpu/vpu_matmul.md`), so the link carries half the bytes
and the card's L2 holds half the activation block. The float weight
types keep float32, because only the generated quantized kernels have
float16 twins.

`PHI_GGML_ACT=0` sends float32 instead, which is what the comparison in
`docs/results/2026-09-23-float16-activations.md` needs. Interleaved on
the 27B, two rounds each:

| Qwen3.8-27B UD-Q4_K_XL | pp512 | tg32 |
| --- | --- | --- |
| host alone, 16 threads | 8.97 | 0.98 |
| both cards, float32 activations | 12.05, 12.02 | 1.45, 1.47 |
| both cards, float16 activations | **12.40, 12.35** | **1.61, 1.54** |

The card's row stride is `k * 2 + B_PAD` rather than `nb_b + B_PAD`; the
padding is unchanged and for the same reason (`B_PAD`, the L1 set
conflict).

## Feed-forward blocks, fused (ffn.rs)

The backend also takes ggml's SwiGLU (the split form llama.cpp builds,
`ggml_swiglu_split(gate, up)`), so that a feed-forward block's gate, up,
SwiGLU and down arrive in one sub-graph. The C glue finds the blocks by
structure and runs each as one request per card through `ffn.rs`: each
card holds a run of the intermediate as gate and up rows and down's
columns, and returns a partial sum the host adds to its own; the
intermediate never crosses the link. `ffn.md` has the shape,
`docs/results/2026-09-23-ffn-per-request.md` the numbers.

The two paths share what should be shared: the batch share and its
estimator (`pp_feed`, which both feed), the per-tensor judgement's rule
(per block for the fused one), the float16 activations, and the cards'
budget. They do not share tensors: a block's three are the fused path's
alone once it has planned them (`Ctx::ffn_members`), because it holds
`ffn_down` by columns and the plain path would want rows; if the plain
path had given them to the cards first, those slices are freed.

It is **off by default**; `PHI_GGML_FFN=1` turns it on. It measured
neutral on both models here, and a fused block never writes its
intermediates, which a program's eval callback can read from outside
the sub-graph this backend sees (llama-imatrix does, for `ffn_down`'s
input; `csrc/ggml-phi.md`). Unset, SwiGLU is not taken and the graph
splits as it did before. `PHI_GGML_FFN_H16=1` keeps the intermediate as
float16 on the card (faster down kernels, and it overflows past 65504,
so only where the activations are known to be bounded).

## Two small savings (2026-09-23 night)

The activations of a multiply are the same for every card, so they are
prepared once, in the first card's window (converted to float16 there),
and copied to the others (`copy_window`), rather than converted once per
card. And a plain multiply whose largest possible card part, every
card's `fraction` of its rows, cannot reach `min_bytes` is never given
to the cards, so its rows are no longer uploaded on first sight either:
the budget goes to tensors that will use it. Both measured neutral on
the 27B as speed; the second matters for what the cards can hold
(`docs/results/2026-09-23-redundancy-and-transport.md`).

## The share is sized from what is offered (2026-09-23 night)

Each card keeps the same fraction of every weight matrix, so the
fraction that fills a card is its budget over the bytes of every weight
it could be given. `scripts/phi-ggml.sh` used to take that from the model
file's size, and on Qwen3.8-27B UD-Q4_K_XL the cards filled to 3.48 GB
of 4.4: a fifth of the file never reaches this backend. Some of it is
types the cards have no kernel for, and 2.3 GB is Q4_K that llama.cpp's
CPU backend repacks into its own interleaved layout on this AVX2 host
(`ggml/src/ggml-cpu/repack.cpp`), in a buffer type that is not a host
buffer. Nothing outside the process can know which, since it depends on
the host's CPU and the program's own placement.

So the backend counts. The glue notes every weight whose multiply it
accepts (`phi_ggml_note_weight`, from `supports_op`, which the scheduler
calls for every multiply of a graph before computing any of it) and
declines outright a weight in a buffer it cannot read as ggml lays it
out (`weight_readable`), so repacked weights are neither taken nor
counted. At the first multiply `settle_fraction` sets the share to 97
percent of the smallest budget over that total (3 percent kept back for
rows rounding up to 64 and for the output matrix, which comes last),
never more than an equal split between the host and the cards
(`share_cap`, a third each with two cards). On the 27B that is 13.7 GB
offered, a share of 31.2 percent and 4.23 GB per card. On the 35B-A3B,
whose Q4_K experts are repacked, it is 5.1 GB offered and the cap. What
a full card is worth is in
`docs/results/2026-09-23-redundancy-and-transport.md`.

The cap was first an equal split between the cards, half each on two,
which leaves the host no rows at all, and the one-token judgement
(`Split::avoid`) cannot work without them: it takes the host's time
alone to be its part's time over its part's share, and with no part
that is the overhead of an empty multiply, so every tensor looks slower
with the cards and is taken off them. The 35B-A3B reached the cap and
generated 8.30 tokens a second against 9.05 at 0.203. `PHI_GGML_FRACTION`
is held to the same cap, with a message when it is set above it.

## Each process starts with empty cards

A card's uploads outlived the process that made them: nothing freed them
at exit (`phi_ggml_free_all` existed and nothing called it), and a new
process replaced them only id by id. Running the 27B and then the
35B-A3B left 4.2 GB of the first on each card under the second's
uploads, the workers ran out of huge pages, took ordinary memory for the
rest, and both cards' kernels killed them for want of memory. `open` now
frees everything on every card before anything is uploaded, which also
covers a process that crashed. It follows that one process at a time
uses the cards: a second one started while the first runs frees the
first's slices under it.

## Offload: the cards' rows leave the host (2026-09-24)

By default a card's rows are a copy. The host keeps the whole model, and
it reads the cards' rows back whenever it computes a multiply alone:
when the multiply is under `min_bytes`, when the judgement found the
cards not paying on that tensor, for float weights at a batch, past the
window's limits, and for the host's share of every card slice at a
batch. On the 35B-A3B Q4_K_M that was 1,059 MB over sixteen tokens in
198 such multiplies (`--verbose`, which now counts it: "the cards' rows
read by the host so far").

`PHI_GGML_OFFLOAD=1` makes the cards' rows theirs alone, for a model
larger than this host's memory. Every multiply of a tensor with resident
rows goes to the cards, whatever it measures and however small; at a
batch each card computes all of its slice (the batch share is 1 and
does not move); and after each upload the pages of the rows now on the
cards are dropped with `madvise(MADV_PAGEOUT)` (`drop_pages`), whole
pages only, since the pages at either end may hold the host's rows. The
same run then read 0 MB of the cards' rows and dropped 3.25 GB of pages.

It needs the model mapped from its file (`--load-mode mmap`, llama.cpp's
default): a dropped page of a file is read back from the file if it is
ever touched, so a miss costs time, never a wrong result. A model read
into ordinary memory (`--load-mode none`) is left alone
(`file_backed`, from `/proc/self/maps`), because paging that out would
push the weights into swap. Two paths are outside it: a multiply the
glue declines at the scheduler (`supports_op`) runs on llama.cpp's CPU
backend over the whole weight, and the fused feed-forward
(`PHI_GGML_FFN=1`) keeps its own rules.

What it is worth, on models near and past the host's memory, is in
`docs/results/2026-09-24-offload-past-memory.md`.
