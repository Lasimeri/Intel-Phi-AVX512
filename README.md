# Intel Xeon Phi as an AVX-512 co-processor

A Xeon Phi 3120 (Knights Corner) card executes AVX-512 on behalf of a
host CPU that has none. An ordinary program, compiled with `-mavx512f`
and neither modified nor asked to cooperate, runs on the host; every
AVX-512 instruction it reaches is carried, with the code around it, to
the card's vector units and executed there as MVEX, the card's own
encoding of the same operations. Nothing is interpreted. Results are
bit-identical to what real AVX-512 hardware produces, on everything the
tests here cover; a review of 2026-09-24 found cases that were not (masked
stores and masked unaligned moves, fixed 2026-09-25) and others still open
(listed under "Still open at the end of the day" in
[`docs/results/2026-09-25-review-transparent-path.md`](docs/results/2026-09-25-review-transparent-path.md)).

```
scripts/phi512.sh ./my-avx512-program        # that is all of it
scripts/phi512.sh --verbose ./my-avx512-program   # each region the card ran, its phases and times
```

The host here is a Ryzen 7 5800X (AVX2 and FMA3, no AVX-512) with two
cards. Measured on card 0 with `tools/avx512-seamless-test.c` (a
polynomial, a dot product and an integer kernel, each checked lane for
lane against a scalar reference), against the only other way to run that
code on this host, a software emulator:

| elements | polynomial | dot product | integers | emulated polynomial |
| --- | --- | --- | --- | --- |
| 65536 | 10.1 to 12.7 ms | 5.3 to 6.6 ms | 7.8 to 7.9 ms | 22.5 ms |
| 1048576 | 23 to 37 ms | 23.7 to 25.2 ms | 18.1 to 18.3 ms | 376 ms |
| 16777216 | 207 to 375 ms | 411 ms | 341 ms | about 6 s |

(Two runs each, 2026-09-25, with the fetch and the write-back of 4 KiB pages
staged through a huge page, `card/vpu/vpu_exec.md`; the emulated column is 2026-09-22.)
These are the numbers of the sound path. On 2026-09-22 the same test ran in 7.7,
3.3 and 5.5 ms at 65536 and 98, 139 and 91 ms at 16 M, before a phase's
chunks became 4 KiB-page mappings in which only the declared pages are
accessible: an access the planner did not declare now faults instead of
reading whatever the card held, and every fetch pays for scattered pages
and a change of protection on 57 threads
([`docs/results/2026-09-22-ggml-backend.md`](docs/results/2026-09-22-ggml-backend.md),
"the price of soundness"; the transport itself is unchanged, as the
stack's block-path timer shows,
[`card/examples/blkbench.md`](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/card/examples/blkbench.md)).

## What you need

- The cards' software stack, the sibling repository
  [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A): it
  boots the card, serves its memory, pins the shared window, and gives
  you the `phi` command. Install it and bring a card up (`phi up`); its
  card kernel must carry patch 0030 (the vector unit across a signal
  handler), which every kernel built there since 2026-09-22 does. The
  scripts here find the stack through `PHI_STACK_ROOT`, the `phi`
  command on PATH, or a checkout next to this one or in `$HOME`, named
  `Intel-Phi-3120A` (a `git clone`) or `Intel Phi 3120A`
  ([`scripts/stack.md`](scripts/stack.md)).
- `make build` here: `libphi512.so` (the library the wrapper preloads),
  `libggml_phi.so` (the ggml backend), `phi-vpu` (the explicit driver),
  the translator and the encoder.
- The card's worker is deployed and started by the wrapper itself the
  first time (`scripts/phi-vpu.sh -c N deploy|start` by hand); it lives
  on the card's persistent disk after that.

## How it works

The wrapper preloads `libphi512.so`. The host CPU raises SIGILL at the
program's first AVX-512 instruction; the library's handler walks the
program's code graph from there to a region bounded by what the card
cannot run (a call, a return, an SSE instruction the host runs itself),
rewrites the EVEX instructions to MVEX in place at the byte level (the
two encodings share their operand bytes; `host/crates/avx512-xlate`),
plans the region (`host/crates/phi512/src/plan.md`: what the registers
hold, which memory the region touches, which loop to split), and sends
it to the card's worker through the shared window. The card maps the
program's code and memory at the program's own addresses, runs the
loop across its threads, and returns the register file and what the
region wrote; the program resumes after the region with its registers
and memory as if the host had executed it. `host/crates/phi512/src/
offload.md` and `card/vpu/vpu_exec.md` have the whole of it;
`docs/results/2026-09-22-seamless-card.md` has how it got there and what
each piece cost.

The explicit path is still there for a program written for the card:
`phi-vpu poly` hands a kernel to the worker directly (`scripts/phi-vpu.sh`).
The software emulator that preceded the card path stays as the fallback
behind `--emulate`.

## The cards as GPUs for AVX-512, beside the CPU: the ggml backend

The instruction-level path above is exact but pays a fixed cost per
region, and a program like llama.cpp has millions of tiny regions per
token (`docs/results/2026-09-22-full-avx512.md`). For it the cards are
used the way GPUs are: whole operators at a time. `host/crates/phi-ggml`
builds `libggml_phi.so`, a ggml backend that an unmodified llama.cpp
loads through `GGML_BACKEND_PATH`; its scheduler hands the backend
every matrix multiply it accepts (`MUL_MAT` and `MUL_MAT_ID`, the
mixture-of-experts one, with float16, float32 and llama.cpp's Q4_K,
Q5_K, Q6_K, Q8_0 and IQ4_XS weights), and the backend shares each
one by rows: every card keeps a share of the weight matrix resident and
multiplies it on its 57 threads with the kernels of
`card/vpu/vpu_matmul_kernel.S` (the quantized formats decoded on the
vector unit, `host/crates/phi-vpu/src/bin/kernelgen/quant.md`), while
the host computes the rest with ggml's own CPU kernels; the results are
gathered per multiply. A card is asked only when it would finish the
multiply sooner: three rules decide that, one of them from what the
backend times on each weight tensor as it runs
(`host/crates/phi-ggml/src/lib.md`). The program itself is an ordinary
build for this host.

```
scripts/phi-ggml.sh ./llama-cli -m model.gguf -p "..." -t 12   # the host and every card that is up
scripts/phi-ggml.sh --verbose ...                              # each multiply, the host part and each card's times
```

Qwen3.8-27B (UD-Q4_K_XL, 17.6 GB) on the 5800X alone and with both
cards, each keeping 31 percent of every weight matrix it is offered (as
much as its 4.4 GB holds), llama-bench, 2026-09-23 night
(`docs/results/2026-09-23-redundancy-and-transport.md`):

| | pp512 tok/s | tg32 tok/s |
| --- | --- | --- |
| host alone, 16 threads | 9.20, 9.19 | 1.10, 1.10 |
| host (12 threads) and both cards | **14.53, 14.44** | **1.94, 1.94** |

The model is resident (`--load-mode none`) and the host and the split
are **interleaved**, two rounds each, because this host's own
throughput drifts by as much as a quarter over tens of minutes: a split
measured now against a baseline measured an hour ago says nothing. The
calling program gets 12 threads with the cards and 16 alone: with 16 the
split loses five times over at generation, its threads fighting the
card daemons (`docs/results/2026-09-23-float16-activations.md`). That is
+58 percent at prompt processing and +76 at generation. What got it
there since the first split's 12.29 / 1.63: the activations cross as
float16, which the card up-converts for nothing, and the share of each
multiply the cards take at a batch is measured rather than set
(`docs/results/2026-09-23-share-and-fusion.md`); then, at generation,
each request's data crosses through the host window itself in 64-byte
vector loads and stores split across the card's threads rather than
through the block device, whose fixed cost per request was most of the
cards' time (1.64 to 1.85), and the backend sizes the cards' share from
the weights it is actually offered, which fills them (1.85 to 1.91,
`docs/results/2026-09-23-redundancy-and-transport.md`). A feed-forward block
can also go to each card as one request whose intermediate never leaves
it (`PHI_GGML_FFN=1`, `host/crates/phi-ggml/src/ffn.md`): on this model
that is neutral (14.68 and 14.59 / 1.63 and 1.61), because here the link
is not what bounds either phase, and it is off by default because it
never writes the intermediates a program's eval callback may read
(`docs/results/2026-09-23-ffn-per-request.md`).

With llama-server and the MTP draft (the earlier split, same day):
host alone 6.93 / 2.31, host and both cards **9.56 / 2.72**.

A mixture-of-experts model works the same way: ggml runs its expert
weights through MUL_MAT_ID, which the cards take as well, each keeping
the same rows of every expert, and the answers match the host's token
for token. (Until 2026-09-24 that held at generation only: at a batch the
plain split read every expert after the first at the wrong stride, so a
prompt's outputs were wrong while its speed was measured; fixed, and
checked on a 1,800-token prompt, in
`docs/results/2026-09-24-q8-remeasured-and-moe-stride.md`. The offloaded
split was never affected.) It is worth less so far: Qwen3.8-35B-A3B-Distill
(Q4_K_M, 20.2 GiB, 256 experts with 8 used per token) generates at 9.06
and 9.06 tokens per second with both cards against 7.89 and 7.99 for the
host alone at its best thread count (12), 14 percent faster, and 6.90 and
6.85 on 16; its pp512 is within 3 percent of the host's (87.34 and 87.10
against 84.99 and 85.88 on 12, 88.12 and 92.16 on 16). Offloaded it
generates at 8.69: its use there is the host's memory. Interleaved on
2026-09-25 (`docs/results/2026-09-25-35b-q4km-remeasured.md`), replacing
the 2026-09-23 figures (+26 and about -8 percent), whose prompt side was
measured with the stride defect live and whose host ran on 16 threads. A
card costs about 0.45 ms to involve and an MoE layer's multiply at one
token is a few megabytes, so the backend times
both sides on each weight tensor and leaves with the host what the
cards would not finish sooner, which on that model is most of the
expert work (`docs/results/2026-09-23-mixture-of-experts.md`).

A model larger than this host's 31 GiB runs from its file, the page
cache holding what it can. By default the cards' rows are a copy, so
they save the host no memory; with `PHI_GGML_OFFLOAD=1` they leave the
host after the upload and it never reads them again. The same model at
Q8_0 (37.8 GB) generates at 5.03 and 5.15 tokens per second that way,
against 4.01 and 3.75 on the host alone, and processes a prompt at 56.9
and 56.6 against 28.0 and 21.8 (`docs/results/2026-09-24-offload-past-memory.md`).
Re-measured that evening with less page cache free (about 21 GB): 4.98
and 4.87 against 3.39 and 3.32 at generation, 46.5 and 58.8 against 24.0
and 35.9 at pp512, each run re-reading 60 to 104 GB from the NVMe
(`docs/results/2026-09-24-q8-remeasured-and-moe-stride.md`). On the
corrected backend the next morning (measured at bd166f4, which budgets
uploads in whole 2 MiB pages): 4.91 and 4.77 at generation, within the day before's
spread, and 60.8 and 46.2 at pp512, against a host alone that drifted
from 3.52 to 2.68 inside the run (same record, "On the corrected backend").

Token generation is bound by weight bandwidth, and the cards add theirs
to the host's; what bounds it now is how much of the model they hold,
since two 6 GB cards take half of 17.6 GB and the host is the long pole
in every multiply with the other half. Prompt processing is bound by
arithmetic, where the cards reach 240 GFLOP/s each on Q4_K once the
activation rows are written into the window a quarter of a page apart
(their L1 has 64 sets, and a 5 x 4096 row stride puts eight rows in the
same ones).

## Layout

| path | what |
| --- | --- |
| `host/crates/phi512` | the preloaded library: the SIGILL handler, the planner, the dispatcher (and the emulator, as the fallback) |
| `host/crates/avx512-xlate` | EVEX to MVEX: the byte-level rewriter for the seamless path and the builder-based translator for kernels |
| `host/crates/knc-mvex` | the MVEX encoder the translator builds on |
| `host/crates/phi-vpu` | the protocol with the card worker, the shared window, the explicit driver |
| `host/crates/phi-ggml` | `libggml_phi.so`, the ggml backend: matrix multiplies shared by rows between the host and the cards |
| `card/vpu` | the card-side worker: the exec engine and the explicit path's thread pool; built on the card by `scripts/phi-vpu.sh deploy` |
| `card/examples` | the AVX-512 kernel and its translation the explicit path and the ground-truth check use |
| `tools` | the seamless, narrow and review tests, the conformance programs, the protocol layout check |
| `scripts` | `phi512.sh` (the wrapper), `phi-vpu.sh` (the worker), `phi512-check.sh` and `phi512-ground.sh` (conformance against hardware and against the card), `phi512-install.sh` (system-wide preload) |
| `docs/research` | the precision audit that gates the translation, and the transparency design |
| `docs/results` | dated records with every number |

`make check` runs the documentation rules, formatting, lints, tests and
the protocol layout check; `CONTRIBUTING.md` has the rules.

## The repositories

| repository | what | how it is found |
| --- | --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack: boots them, serves their memory, the `phi` command | `PHI_STACK_ROOT`, else `phi` on PATH, else a checkout next to this one, else in `$HOME` |
| Intel-Phi-AVX512 (this one) | the cards as an AVX-512 co-processor | `PHI_AVX512_ROOT`, else a checkout next to Intel-Phi-Jev, else in `$HOME` |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev) | `xks`, a local Jev (TypeSafe System One) whose model runs through `libggml_phi.so` (its `cards` site) and `phi512.sh` (its `avx512` site) | by Mechanical-Jev |
| [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | `mjev`, the asking side of that Jev | |

Cloned side by side, the repositories find each other without
configuration, under each one's clone name or the spaced one; the model
and llama.cpp paths are set in Intel-Phi-Jev's `xks.conf`, and the cards
need the stack's `phi` command with a card up. What Intel-Phi-Jev
consumes from here (`scripts/phi512.sh`, `scripts/phi-vpu.sh`,
`host/target/release/libggml_phi.so`, the `PHI_GGML_*` and `PHI_VPU_*`
variables) stays as it is across changes; [`CONTRIBUTING.md`](CONTRIBUTING.md)
has the rules the repositories share. MIT ([`LICENSE-MIT`](LICENSE-MIT)).

## History

This began inside the stack's repository and was split out on
2026-09-22 (stack commit 0f49eac) so that the co-processor and the stack
can move independently. The commits before the split are in that
repository's history.
