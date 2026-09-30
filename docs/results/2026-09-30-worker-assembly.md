# 2026-09-30: the card worker in assembly

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up under
the stack's assembly `phictl` (Intel-Phi-3120A, 2026-09-29), every card
access over the stack's control socket (`phi -c N run`, `put`, `get`;
SSH off). Card 0 is the 3120A on a Gen2 x8 link, card 1 the
subsystem-3608 card on the chipset's Gen2 x4; both 57 cores, 1.1 GHz
(the time-stamp counter measured at 1099.8 ticks per microsecond on
every CPU), workers at 114 threads. This repository at 6484cf0 plus this
record's change (step 1e of the port); the C worker being compared is
the one at 98f8cc1 (`vpu_worker.c`, `vpu_exec.c`, `vpu_matmul.c`,
compiled by the stack's `knc-cc -O2`), kept on card 1 beside the
assembly one as `phi-vpu-worker.c` and swapped in by hand. llama.cpp
`build-native` at f5b9bd3 (unchanged).

## What was done

The card worker (`card/vpu`), 3390 lines of C plus 7232 of generated
kernel bytes, is now 17276 lines of x86-64 assembly and include files
(`worker.S` 1609, `exec.S` 2731, `matmul.S` 3126, `rows.S` 1334,
`text.S` 324, `kernels.S` 7236, and `defs.inc`, `proto.inc`,
`matmul.inc`, `mvex.inc`), assembled on the host by GNU `as`, linked
static by `ld` with no libc (173000 bytes), audited by the stack's
`phi-isa-audit` (14822 instructions, 6715 of them the card's vector
instructions, nothing Knights Corner does not run). The 41 kernels are
hand-maintained macro lines over `mvex.inc`, an encoder of the card's
MVEX prefix as assembler macros, which the first step gated byte for
byte against the generated file it replaced. The plan of 2026-09-30 (the whole repository to assembly, the card
worker first) set the steps and their gates; the steps and their commits:

| step | what | gate | commit |
| --- | --- | --- | --- |
| 1a | `mvex.inc`, `kernels.S`; kernelgen deleted; the scripts over the control socket | the 62730 bytes of the kernels identical, through GNU `as` and through the card toolchain | 8f5c6ed |
| 1b | `worker.S`, `text.S`, `defs.inc`, `proto.inc`: the pool by `clone` with a block per thread as `fs`, the doorbell, POLY30 | `phi-vpu status` and `poly` bit-exact at 65536, 1M, 16M | 8425a17 |
| 1c | `matmul.S`, `rows.S`, `matmul.inc`: the matrix service | `matmul-check` every case, the full output identical to the C worker's after timings are stripped | 98f8cc1 |
| 1d | `exec.S`: the seamless path's engine | the Rust `libphi512.so` over the new worker: seamless test bit-exact (65536, 1M, 16M), narrow test 66 forms, review test, `phi512-ground.sh`; the two workers' logs identical after addresses and times are stripped (2357 lines) | 6484cf0 |
| 1e | the probe and the `-t` trace, the measurement below, the C sources deleted, `build.sh` and `deploy` on the binary | this record | this commit |

## The fixed cost per request, same card, interleaved

`phi-vpu -c 1 matmul-check --moe --repeat 300` (one token's mixture
multiplies, q4_K, 8 of 256 experts at random over four tensors, the
35B-A3B's gate or up share 128 x 2048 and its down share 448 x 512 x 8
rows, and the gate and up as two requests and as one) and
`--moe-shape 2048,2048,1`, workers started with `-t 100` (the trace
only; no per-request log line), the assembly worker and the C worker
alternating on card 1: A then B. Card total is the card's own clock from
the request seen to the reply written, best of 300 and the median; the
host round trip is the driver's.

| shape | assembly A | C A | assembly B | C B |
| --- | --- | --- | --- | --- |
| 128 x 2048, 1 row: card total ms | 0.195 (0.206) | 0.206 (0.235) | 0.195 (0.209) | 0.212 (0.323) |
| same, host round trip median | 0.210 | 0.246 | 0.213 | 0.327 |
| gate and up as two requests | 0.391 (0.417) | 0.423 (0.519) | 0.398 (0.420) | 0.432 (0.654) |
| gate and up as one request | 0.274 (0.292) | 0.286 (0.309) | 0.270 (0.290) | 0.287 (0.360) |
| 448 x 512, 8 rows | 0.207 (0.220) | 0.212 (0.230) | 0.206 (0.220) | 0.220 (0.234) |
| 2048 x 2048, 1 row | 1.047 (1.099) | 1.114 (1.216) | 1.071 (1.111) | 1.137 (1.248) |

The assembly worker is ahead on every line, best and median, in both
repeats: 5 to 8 percent at the best, 4 to 35 percent at the median (the
C worker's medians vary between its two runs; the assembly's do not, for
the reason under "The slow mode").

The stages, from the worker's own trace (`-t 100`: microseconds per
request, the mean of the last hundred of each kind):

| kind | worker | read request | descriptors | pull | groups | compute | push | reply |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matmul_id 128 x 2048 | assembly A | 2.0 | 4.1 | 4.1 | 3.0 | 192.0 | 10.6 | 1.3 |
| | C A | 1.6 | 3.8 | 7.8 | 2.8 | 241.9 | 11.3 | 1.2 |
| | assembly B | 1.5 | 3.3 | 3.3 | 3.3 | 212.8 | 10.3 | 1.0 |
| | C B | 1.5 | 3.9 | 7.6 | 2.8 | 309.9 | 10.7 | 1.8 |
| matmul_more (gate and up) | assembly A | 2.1 | 3.7 | 8.5 | 3.9 | 288.6 | 20.4 | 1.3 |
| | C A | 1.6 | 3.3 | 11.5 | 4.3 | 291.3 | 20.3 | 1.7 |
| | assembly B | 1.5 | 4.2 | 7.5 | 3.6 | 263.6 | 21.3 | 1.1 |
| | C B | 1.5 | 3.8 | 11.9 | 3.9 | 317.3 | 19.6 | 1.4 |
| matmul_id 448 x 512 x 8 | assembly A | 1.5 | 3.3 | 2.7 | 2.8 | 184.5 | 34.0 | 1.0 |
| | C A | 1.5 | 4.1 | 6.9 | 2.7 | 191.8 | 33.2 | 1.1 |
| | assembly B | 1.5 | 3.2 | 2.8 | 2.6 | 184.7 | 32.7 | 1.0 |
| | C B | 1.5 | 3.9 | 7.2 | 2.7 | 219.9 | 33.3 | 1.1 |

The pull stage (the ids through the mapping, the buffers, the window)
is 3 to 4 microseconds against 7 to 8; the compute stage (the fused
pull inside it, the dispatch, the rows) 185 to 213 against 192 to 310;
the rest within a microsecond of each other.

The probe's ceilings (`matmul-check --probe`, card 1, each worker in
turn, no trace), which do not cross the link:

| | assembly | C |
| --- | --- | --- |
| one dispatch across 57 threads, no work | 13.0 us | 14.0 us |
| the eight-row Q4_K kernel on an L1-resident superblock, every thread | 579 ns per call (403 GFLOP/s) | 790 ns (296) |
| aggregate read bandwidth, 4 MiB a thread, prefetched | 73.1 GB/s | 66.8 GB/s |
| aggregate vector issue, register FMAs | 777 GFLOP/s | 755 GFLOP/s |

The one-thread rates (the streaming walks, the kernels on one
superblock, 275 to 279 ns a call for the one-row Q4_K kernel) are the
same to a few nanoseconds: the kernels are the same bytes. The pool
lines differ by the pool: the C's second thread on each core spins on
the generation word between the C's 2000-look rounds with a library
clock call, the assembly's with the counter, and the assembly's
completion count is on eight lines (below).

## What it took to get there

The first assembly of each step was slower than the C on some line,
and the C worker's own measurements (its `-v -v` lines, its `-t`
trace, `matmul-check --moe`) found each cause. In order:

- **The log line in the measured stages.** The assembly worker prints
  a request-start line at `-v -v` that the C never had; measured at
  `-v -v`, its write sat inside the pull stage (7 microseconds) and the
  comparison was unfair. Every number above is without per-request
  logging.
- **The row loop's scalar overhead** (`rows.md`, "Registers and
  frames"): three multiplies and a dozen frame accesses per kernel call
  where the C compiler had hoisted them: about 100 nanoseconds on top of
  a 280-nanosecond kernel on this in-order core, 13 percent on the
  2048-row shape (1.026 ms against the C's 0.904). The loop carries the
  block and accumulator pointers in registers the kernels keep and adds
  the stride: 0.858, ahead of the C.
- **The clock** (`worker.md`, "The clock"): `clock_gettime` is a system
  call in a binary with no libc to find the vDSO through, where the C
  worker's was a library call; the request path takes the time six or
  seven times and every idle pool thread every 2000 spin rounds. The
  worker now scales the time-stamp counter by a rate measured at start.
  A first version divided by the wrong register half (`rdx` holds bits
  64 to 127 of a `divq` dividend, not a shift by 32) and printed times
  a hundred thousand times too large; found with a standalone test on
  the card.
- **The slow mode** (`rows.md`, "The slow mode"): with the copy's plain
  stores, the one-row mixture shape computed in 95 microseconds at best
  and 190 at the median, the C worker at 105 to 165; per-slice phase
  stamps showed every thread's row loop three to four times slower in
  the slow requests, independent of the experts drawn, and the mode
  absent with `-m 0`. The activation row written by eight copying
  threads and then read by every core was being served, when still
  dirty, from the copiers' L2 one reader at a time; no-read stores
  (`vmovnrngoaps`, ordered by the locked add that counts the copy) took
  the median to 93 microseconds over two restarts, 270 of 300 requests
  under 100. The C worker has the same exposure at a different rate.
- **The completion count** (`worker.md`, "The pool"): with the wake
  faster, the 57 cores' locked adds to the one `pool_done` line arrived
  together and queued: by the trace, 11 microseconds from the last
  thread's end to the dispatcher seeing the count, where the C worker's
  slower wake spread them over 5. Eight counters on eight lines (core
  modulo eight), summed by the dispatcher: 4 microseconds, a dispatch of
  nothing 18 to 11 by the trace and 13.0 against the C's 14.0 by the
  probe.
- **A copy past the end of a range** (`exec.md`): the sum made for the
  write-back slot's capacity check was left in the register that carried
  the page count, so a second range staged into a slot copied extra
  pages; the seamless test, which stages one range a slot, had passed,
  the narrow and review tests found it.

Two things the port did not change: the x87 sums (`rows.md`, "The
sums") reproduce the compiled C's rounding, which is why every result
above is the same float to the bit; and the kernels, whose bytes the
first step froze.

## The texts

llama-server (`scripts/phi-ggml.sh`, Qwen3.8-35B-A3B Q4_K_M, `-t 12 -c
8192 -b 512 -ub 512 -np 1 --no-repack`, `PHI_GGML_CARDS=1
PHI_GGML_OFFLOAD=1`), a warm-up request, then the code prompt and the
prose prompt of the placement record, 64 greedy tokens each, once with
the C worker on card 1 and once with the assembly worker on card 1: the
three texts the same byte for byte (225, 225 and 286 bytes). Two
things the first attempt got wrong, kept here because they will bite
again: the texts of card 0 and card 1 are not comparable (the backend's
judge measures each card and can place a tensor differently), and
without `PHI_GGML_OFFLOAD=1` the judge keeps measuring during the run,
so on card 1 the same prompt gave two texts from one server, the second
request's differing from the first's at a near-tie token; offloaded,
the cards' rows are theirs alone and nothing is judged, which is the
configuration every record with a byte-for-byte claim ran.

## What is deleted, what stays

`vpu_worker.c`, `vpu_exec.c`, `vpu_matmul.c`, `vpu_exec_regs.h` and the
C `build.sh` are gone (in the history before 6484cf0); `build-asm.sh`
is `build.sh`; `scripts/phi-vpu.sh deploy` builds on the host and puts
the binary (nothing compiles on a card any more; the `build-here` verb
is gone). The C design records (`vpu_worker.md`, `vpu_exec.md`,
`vpu_matmul.md`) stay, headed by what replaced them: their measurements
and defects are the design's history, and the assembly keeps every
mechanism they name. The protocol headers (`vpu_proto.h`, `vpu_exec.h`,
`vpu_matmul.h`) stay as the readable contract `tools/vpu-layout-check.c`
reads; `proto.inc` carries the same numbers. The generated-kernel
crate (kernelgen) and this repository's knc-mvex copy go with step 4b,
when the seamless rewriter's encoder no longer needs the crate.

The next steps of the port are the host side: `libggml_phi.so`, the
driver, `libphi512.so` with its translator and emulator, and phi-pld,
each gated against its Rust reference before the reference goes.
