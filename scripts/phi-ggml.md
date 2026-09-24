# phi-ggml.sh: llama.cpp with the cards beside the host

```
scripts/phi-ggml.sh [--card N] [--verbose] <command> [args...]
```

Runs an ordinary build of a ggml program (llama.cpp, unmodified, built
for this host) with the cards as a ggml device: `GGML_BACKEND_PATH` names
this repository's `libggml_phi.so` (`host/crates/phi-ggml`), which ggml
loads at start and lists as the accelerator "Phi". llama.cpp's scheduler
then gives it every `MUL_MAT` and `MUL_MAT_ID` it accepts (float32,
float16, Q4_K, Q5_K, Q6_K, Q8_0 or IQ4_XS weights, float32 activations,
no batch dimensions, under the window's limits) and keeps the rest on
the CPU. `MUL_MAT_ID` is the mixture-of-experts multiply, which is where
almost all of an MoE model's weights are. Each such multiply is shared
by rows: every card keeps a share of the weight rows
resident (uploaded once) and multiplies them on its 57 threads with the
kernels of `card/vpu/vpu_matmul_kernel.S` while the host computes its
own rows with ggml's CPU kernels; the results are gathered per multiply.

This is the counterpart of `scripts/phi512.sh`, which intercepts a
program's own AVX-512 instructions: here the program is a normal one for
this CPU, and the AVX-512 work is what the backend ships to the cards,
whole operators at a time, the way a GPU is used, beside the CPU rather
than instead of it.

Settings, all environment variables (`host/crates/phi-ggml/src/lib.md`):
`PHI_GGML_CARDS` (which cards; default every card whose window exists;
`--card N` sets it to one), `PHI_GGML_FRACTION` (rows per card; unset,
the backend sizes it to fill the budget, below; either way at most an
equal split with the host, a third on two cards),
`PHI_GGML_CARD_BYTES` (resident bytes per card, 4.4 GB),
`PHI_GGML_PP_SHARE` (where the cards' share at prompt sizes starts,
0.75; it then follows what the two sides measure, and
`PHI_GGML_PP_ADAPT=0` holds it still),
`PHI_GGML_FFN` (unset or 0: three multiplies per feed-forward block, as
before; 1: one request per card, its intermediate never crossing the
link, and never written on the host either, so not with a program that
reads intermediates through an eval callback, such as llama-imatrix),
`PHI_GGML_FFN_H16` (0: the card keeps the intermediate as float32; 1:
float16, faster and overflowing past 65504),
`PHI_GGML_HOST_POOL` (1: the host's rows run on a threadpool of their
own; 0: ggml's disposable one per graph) and `PHI_GGML_HOST_POLL` (0: that
pool's threads do not spin between graphs),
`PHI_GGML_LIB` (another build of `libggml_phi.so` to load instead of this
repository's, so two builds can be compared interleaved),
`PHI_GGML_OFFLOAD` (1: the cards' rows leave the host, for a model
larger than its memory; needs `--load-mode mmap`, below),
`PHI_GGML_HOST_THREADS` (the host's threads for its rows, 12: leave the
card daemons a CPU each, and give the program the same `-t`),
`PHI_GGML_THREADS` (card threads, 57). `--verbose`
(`PHI_GGML_VERBOSE=1`) prints every multiply with the host part, the wait
and each card's timings, and every slice kept resident. The workers
must know the matmul service and its formats (deploy from this tree:
`scripts/phi-vpu.sh -c N deploy`, then `start`).

## The share fills the budget

Each card keeps the same fraction of every weight matrix's rows, and what
that should be is the card's budget over the weights it could be given:
too small and the card's memory sits empty, too large and the budget
runs out partway through the model, leaving the last layers entirely to
the host. This script used to compute it from the model file's size,
which on the 27B left the cards at 3.48 GB of 4.4, because a fifth of the
file never reaches the backend (types the cards take no kernel for, and
Q4_K that llama.cpp repacks for its own CPU kernels). The backend now
counts the weights it is offered and sizes the share itself at the first
multiply (`host/crates/phi-ggml/src/lib.md`): 31.2 percent and 4.23 GB
per card on the 27B. The host always keeps a part of every matrix at
least the size of a card's, or the backend could not time the cards
against it.
Because every process starts by freeing the cards, run one at a time.

Workers this script starts are given `-e 0` (the seamless path's 512 MiB
page pool left to the multiplies) and `PHI_VPU_HUGEPAGES=2400`, both for
the same reason: on this model the cards' residency is what bounds
generation, not their speed
(`docs/results/2026-09-23-ceilings-and-residency.md`).

The cards' host windows (`~/.config/phi/cards`, the sibling stack's
`HOSTMEM` column) need only 768 MiB for this backend. They are 6G each
by default, and two of those on a 31 GiB host leave less page cache than
a 17.6 GB model needs: the host then reads weights from the NVMe while
it works (pp64 6.22 against 9.33). 2G each is the setting here.

`PHI_GGML_MIN_BYTES` (4 MB) is the least a multiply must take off the
host before a card is asked at all; below it a card cannot beat its own
round trip, and asking anyway costs a measurement
(`host/crates/phi-ggml/src/lib.md`).

## A model larger than the host's memory

By default each card's rows are a copy and the host keeps the whole
model. With `PHI_GGML_OFFLOAD=1` and the model mapped (`--load-mode
mmap`, or no load-mode option at all), the rows on the cards leave the
host's memory after the upload and the host never reads them again, so
only the host's part of the model needs to stay resident: the cards'
8.5 GB or so come off what the page cache has to hold
(`host/crates/phi-ggml/src/lib.md`, "Offload").
