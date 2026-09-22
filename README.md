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
