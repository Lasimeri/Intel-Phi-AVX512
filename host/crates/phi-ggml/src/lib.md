# lib.rs: the cards beside the host, one multiply at a time

`libggml_phi.so`, loaded by an unmodified llama.cpp through
`GGML_BACKEND_PATH` (`scripts/phi-ggml.sh` sets it). ggml's scheduler
gives the backend every `MUL_MAT` whose weight type and shape
`csrc/ggml-phi.c` accepts; this file shares each one by rows of the
weight matrix between the host and the cards:

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
(`docs/results/2026-09-23-quantized-kernels.md`); for a 0.5B model it
is not, and the split is slower than the CPU alone there.

Shares: `PHI_GGML_FRACTION` (0.2) of every weight matrix's rows per
card, its rows a multiple of 64 so the host's remainder keeps ggml's
fast paths (measured 0.43 against 7.7 ms for an odd count), until a card
has `PHI_GGML_CARD_BYTES` (3.4 GB) resident or refuses an upload, after
which the card keeps nothing more of later tensors. At eight activation
rows or more (a prompt) each card computes only `PHI_GGML_PP_SHARE`
(0.5) of its slice and the host the rest of it as another range: the
cards are slower per flop than the host is, and faster per weight byte,
so the two cases want different shares. Only tensors ggml names
`*.weight` are shared; anything else the host does whole.

`PHI_GGML_CARDS` names the cards (a comma list of indices; default every
card whose window exists), `PHI_GGML_THREADS` the card threads per
multiply (57; the pool must be that size, more spins), `PHI_GGML_VERBOSE`
prints every multiply with the host part, the wait, and each card's
pull, compute and push. The host side's thread count is the glue's
(`PHI_GGML_HOST_THREADS`, 12): the card daemons serve the DMA on the
host, and a host thread sharing their CPU stalls ggml's barrier for a
timeslice (7 ms per multiply at 15 threads, measured 2026-09-23).

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
row cannot conflict with itself. `PHI_GGML_PP_SHARE` followed it from
0.5 to 0.75.
