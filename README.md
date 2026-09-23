# Intel Xeon Phi as an AVX-512 co-processor

A Xeon Phi 3120 (Knights Corner) card executes AVX-512 on behalf of a
host CPU that has none. An ordinary program, compiled with `-mavx512f`
and neither modified nor asked to cooperate, runs on the host; every
AVX-512 instruction it reaches is carried, with the code around it, to
the card's vector units and executed there as MVEX, the card's own
encoding of the same operations. Nothing is interpreted. Results are
bit-identical to what real AVX-512 hardware produces.

```
scripts/phi512.sh ./my-avx512-program        # that is all of it
scripts/phi512.sh --verbose ./my-avx512-program   # each region the card ran, its phases and times
```

The host here is a Ryzen 7 5800X (AVX2 and FMA3, no AVX-512) with two
cards. Measured 2026-09-22 on card 0 with `tools/avx512-seamless-test.c`
(a polynomial, a dot product and an integer kernel, each checked lane
for lane against a scalar reference), against the only other way to run
that code on this host, a software emulator:

| elements | polynomial | dot product | integers | emulated polynomial |
| --- | --- | --- | --- | --- |
| 65536 | 7.7 ms | 3.3 ms | 5.5 ms | 22.5 ms |
| 1048576 | 16 ms | 11 ms | 11 ms | 376 ms |
| 16777216 | 98 ms | 139 ms | 91 ms | about 6 s |

At 16 M elements the loop itself runs in 4.4 ms on the card's 57 cores;
the rest is moving 64 MiB each way over the card's Gen2 x8 link.

## What you need

- The cards' software stack, the sibling repository
  [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A): it
  boots the card, serves its memory, pins the shared window, and gives
  you the `phi` command. Install it and bring a card up (`phi up`); its
  card kernel must carry patch 0030 (the vector unit across a signal
  handler), which every kernel built there since 2026-09-22 does. The
  scripts here find the stack through `PHI_STACK_ROOT`, the `phi`
  command on PATH, or a directory named `Intel Phi 3120A` next to this
  one (`scripts/stack.md`).
- `make build` here: `libphi512.so` (the library the wrapper preloads),
  `phi-vpu` (the explicit driver), the translator and the encoder.
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
gathered per multiply. The program itself is an ordinary build for this
host.

```
scripts/phi-ggml.sh ./llama-cli -m model.gguf -p "..." -t 12   # the host and every card that is up
scripts/phi-ggml.sh --verbose ...                              # each multiply, the host part and each card's times
```

Qwen3.8-27B (UD-Q4_K_XL, 17.6 GB) on the 5800X alone and with both
cards, each keeping a quarter of every weight matrix, llama-bench,
2026-09-23 (`docs/results/2026-09-23-ceilings-and-residency.md`):

| | pp64 tok/s | pp512 tok/s | tg16 tok/s |
| --- | --- | --- | --- |
| host alone, 16 threads | 9.33 | 9.24 | 1.07 |
| host (12 threads) and both cards | 11.56 | 11.79 | 1.51 |
| llama-server with the MTP draft: host alone | 7.79 | | 2.26 |
| llama-server with the MTP draft: host and both cards | 6.00 | | 2.77 |

A mixture-of-experts model works the same way: ggml runs its expert
weights through MUL_MAT_ID, which the cards take as well, each keeping
the same rows of every expert, and the answers match the host's token
for token. It is worth much less so far: Qwen3.8-35B-A3B-Distill
(Q4_K_M, 20.2 GiB, 256 experts with 8 used per token) gains 3 to 5
percent at generation and loses 11 percent at pp512 against the best
the host does alone. A card costs about 0.45 ms to involve and an MoE
layer's multiply at one token is a few megabytes, so the backend times
both sides on each weight tensor and leaves with the host what the
cards would not finish sooner, which on that model is most of the
expert work (`docs/results/2026-09-23-mixture-of-experts.md`).

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
| `card/vpu` | the card-side worker: the exec engine and the explicit path's thread pool; built on the card by `scripts/phi-vpu.sh deploy` |
| `card/examples` | the AVX-512 kernel and its translation the explicit path and the ground-truth check use |
| `tools` | the seamless test, the conformance programs, the protocol layout check |
| `scripts` | `phi512.sh` (the wrapper), `phi-vpu.sh` (the worker), `phi512-check.sh` and `phi512-ground.sh` (conformance against hardware and against the card), `phi512-install.sh` (system-wide preload) |
| `docs/research` | the precision audit that gates the translation, and the transparency design |
| `docs/results` | dated records with every number |

`make check` runs the documentation rules, formatting, lints, tests and
the protocol layout check; `CONTRIBUTING.md` has the rules.

## History

This began inside the stack's repository and was split out on
2026-09-22 (stack commit 0f49eac) so that the co-processor and the stack
can move independently. The commits before the split are in that
repository's history.
