# 2026-09-23: what bounds the cards, and what bounds the split

The morning's pass measured the two ceilings a matrix multiply on the
card runs into, settled the threads-per-core question with the pool
sized to the work, and found that the thing bounding the 27B end to end
is neither: it is how much of the model each card can hold. The numbers
below are from `phi-vpu -c N matmul-check` (rates and probe) and
`llama-bench` on Qwen3.8-27B UD-Q4_K_XL, the same model and build as
`2026-09-23-quantized-kernels.md`.

## The instrument first

The rate a single timed request reports varies by 3.7x on an unchanged
card: 0.590, 0.621 and 2.190 ms for the same 4096 x 5120 Q4_K multiply
at n 1. The spread is the pool's first request after an idle gap: the
threads have parked (`-s`, 200 ms) and the host's own work between
requests is longer than that, so one request pays 56 futex wakes and the
first touch of its stacks. `matmul-check` now uploads once and times the
multiply `--repeat` times (7 by default), reporting the best and the
median; every number here is a best of 7, and the medians are within a
few percent of them. Without this, none of the comparisons below are
distinguishable from noise.

## The two ceilings

`--probe` now runs two benchmarks across the whole pool, each thread on
its own 4 MiB (the card's aggregate rates, which is what a multiply is
measured against; one thread's rate times 57 is not):

| threads (pool sized to match) | aggregate read, prefetched | aggregate vector issue | one dispatch, no work |
| --- | --- | --- | --- |
| 57 (one per core) | 76.9 GB/s | 809.6 GFLOP/s (2.25 ns per FMA per thread) | 40 to 48 us |
| 114 (two per core) | 75.8 GB/s | 1116.2 GFLOP/s (3.27 ns) | 61 us |
| 228 (four per core) | 73.5 GB/s | 846.5 GFLOP/s (8.62 ns) | 220 us |

So the memory system is saturated by one thread per core: a second
thread adds nothing to bandwidth. The vector unit is not: a second
thread on the core adds 38 percent of issue, which is the in-order
core's rule that one thread cannot issue on consecutive cycles. A fourth
thread loses it again.

The multiply itself, measured with the pool sized to the slices (the
earlier session's "everything twice as slow" was 57 slices in a 228
pool, 171 threads spinning on the generation word beside the workers):

| Q4_K 4096 x 5120 | n 1 | n 8 | n 64 |
| --- | --- | --- | --- |
| 57 threads | 0.578 ms | 2.868 ms | 22.490 ms |
| 114 threads | 0.718 | 2.914 | 24.514 |
| 228 threads | 0.997 | 3.748 | 30.369 |

One thread per core stands, for the reason the ceilings give: at n 1 the
kernel is against memory, where the second thread adds nothing and costs
interleaved streams; at n 64 it is against issue, where the second
thread's 38 percent does not cover splitting the core's L1 between two
sets of activation blocks and accumulators.

## The kernels are issue bound, exactly

Counting the instructions the generator emits per superblock against the
kernel's measured time on a superblock held in L1:

| kernel | instructions | 2.08 ns each | measured |
| --- | --- | --- | --- |
| phi_q4k_1 | 132 | 274 ns | 277 ns |
| phi_q4k_8 | 267 | 555 ns | 518 ns |
| phi_q5k_1 | 192 | 399 ns | 408 ns (streaming) |

There is no stall to remove and no prefetch to tune: every instruction
issues at the one-thread rate and the kernel is the instruction count.
Making a quantized multiply faster on one thread per core means emitting
fewer instructions per weight, nothing else. At n 1, 57 threads at 132
instructions per 144 bytes is 0.53 GB/s per thread, 30 GB/s for the
card, against the 76.9 GB/s the memory system would give: the weights
are decoded, not fetched.

## Two things that were free

**The row chunk.** Rows go in chunks so a chunk's weights and one group
of activation rows sit in the core's L2 together. 16 was tuned when the
scale decode was still scalar; the card now takes it from the request
(`matmul-check --chunk`), so it was measured:

| rows per chunk | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| Q4_K n 1 | 0.780 ms | 0.604 | 0.570 | **0.552** | 0.605 |
| Q4_K n 64 | 33.631 | 31.200 | 22.363 | **20.944** | 20.940 |

32 is the default now.

**The card's page pool.** The seamless path pre-faults 256 huge pages
(512 MiB) at start, which a worker serving only matrix multiplies never
touches; `-e 0` leaves them, and every huge page left is card memory a
slice of the model can sit in. `scripts/phi-ggml.sh` starts workers with
`-e 0` and reserves 2400 huge pages (4.7 GiB of the card's 5.5).

## An invalid opcode: the card has no CMOV

A `cmovne` added to the bench kernel's prologue killed the worker with
`trap invalid opcode` (the card's `dmesg`, three times, once per probe).
The scalar side of a Knights Corner core is a P54C, which predates CMOV;
the cross compiler never emits one, so nothing had found this before.
Hand-written scalar code in `kernelgen` uses a branch.

## The host window was starving the page cache

Each card is given host memory through `/dev/shm/phi-hostmem*`, 6 GiB by
default (`~/.config/phi/cards`, the sibling stack's column). Two cards
take 12 GiB of this host's 31, which leaves less page cache than the
17.6 GB model needs, and the host reads weights from the NVMe while it
works. That, not any change here, is what moved `llama-bench` pp64 on
the host alone from 9.34 to 6.22 and back:

| host alone, 16 threads | pp64 | tg16 |
| --- | --- | --- |
| 12 GiB of windows | 6.22 | 1.06 |
| 4 GiB of windows | 9.33 | 1.07 |

The backend needs 768 MiB of window (`matmul.rs`: OFF_D + D_MAX), so the
cards are configured with 2G each now. Prompt processing is compute
bound on the host and shows the cache; generation is bound by what the
host reads either way and barely moved.

## What bounds the split: residency

Per multiply at generation, with 23 percent of every weight matrix on
each card (`PHI_GGML_VERBOSE=1`, one 4096-row layer matrix):

```
host part 1.993 ms, waited 0.005 ms more;
card 0 rows 4032: 1.476 ms (pull 0.136, compute 0.832, push 0.365)
card 1 rows 4032: 1.517 ms (pull 0.170, compute 0.852, push 0.347)
```

The host is the long pole in every multiply and the cards wait. They
cannot take more rows than they hold, and they hold what their 5.5 GiB
of memory allows: about 4.4 GB each, 50 percent of this model between
them. Faster kernels or cheaper transport would not move generation on
this model at all; only more resident rows would, and there are none to
be had. `scripts/phi-ggml.sh` now sets the share from the model's own
size (`budget / bytes`, the `-m` argument) so the cards fill evenly over
the whole model instead of running out partway through it.

The host's threads were re-measured on the clean host, since the 12 that
`ggml-phi.c` documents had been found under the old memory pressure (and
the code still said 15):

| llama.cpp -t | the backend's private CPU backend | pp64 | tg16 |
| --- | --- | --- | --- |
| 12 | 12 | **9.96** | **1.52** |
| 14 | 14 | 9.98 | 1.42 |
| 16 | 16 | 5.33 | 0.18 |
| 4 | 16 | 5.28 | 0.23 |
| 4 | 12 | 8.89 | 1.36 |

12 is right for both pools: the private backend with one thread per CPU
leaves nothing for llama.cpp's own pool or the two card daemons, and
ggml's barrier spins, so a descheduled thread costs a timeslice.

The cards' share at prompt sizes was measured too (`PHI_GGML_PP_SHARE`,
the part of its resident rows a card computes at n 8 or more): 0.5 gives
pp64 9.96, 0.75 gives 9.23, 1.0 gives 7.54. The host's Q4_K kernels work
in int8 against the cards' float32, so per row of arithmetic the host is
about twice the two cards together; half is the balance.

## The eight activation rows were all in one L1 set

At n 1 the real loop costs 384 ns per superblock against the kernel's
277 in L1: close. At n 8 it costs 2.0 us against 518 ns: four times.
Nothing about the weights explains it (a shape small enough to hold
every thread's weights in cache is no faster), and the card's aggregate
issue is within 8 percent of one thread's, so the cores are not
interfering.

The activation rows are. The kernels take them as memory operands, `k`
floats apart: at k 5120 that is 20480 bytes, 5 x 4096, so the eight rows
of a group land in the same 64 sets of a 64-set, 8-way, 32 KiB L1, and
every vector position evicts the one before it. At k 512 the stride is
2048 and the rows fall into two set groups, which is why the same total
work was faster at k 512 than at k 5120 (171 against 128 GFLOP/s)
despite eight times the rows.

The fix is the stride, not the kernel: the host writes the activation
rows into the card's window a quarter of a page further apart
(`B_PAD` 256, which keeps the 64-byte alignment the operands need), and
the card is told that stride. `matmul-check --pad` measured it:

| Q4_K 4096 x 5120, bytes added to the stride | 0 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| n 8 | 2.747 ms | 2.663 | 1.805 | **1.389** | 1.400 | 1.401 |
| n 64 | 21.105 | 17.173 | 12.984 | **11.159** | 11.177 | 11.084 |

n 1 does not move (one row cannot conflict with itself), and 256 bytes
is where it settles: four sets between rows, so eight rows spread over
32 of the 64 sets. Every quantized format about doubles at n 8 and n 64:

| 4096 x 5120, 57 threads | n 1 | n 8 | n 64 |
| --- | --- | --- | --- |
| q4_K | 0.62 ms | 239 GFLOP/s | 240 GFLOP/s |
| q5_K | 0.74 | 214 | 214 |
| q6_K | 0.72 | 172 | 210 |
| q8_0 | 0.53 | 233 | 239 |
| iq4_xs | 0.64 | 228 | 185 |

240 GFLOP/s is 30 percent of the card's 810 GFLOP/s issue ceiling, which
is what a kernel spending 8 of every 15 instructions on fused
multiply-adds can reach. The float kernels (`phi_dot4_*`, a different
loop) did not move: f16 is still 33 GFLOP/s at n 64.

With the cards twice as fast at prompt sizes, their share there was
re-measured: `PHI_GGML_PP_SHARE` 0.5 gives pp64 10.40, **0.75 gives
11.72**, 1.0 gives 11.58. 0.75 is the default now.

## Where this leaves the 27B

| Qwen3.8-27B UD-Q4_K_XL | pp64 | pp512 | tg16 |
| --- | --- | --- | --- |
| host alone, 16 threads | 9.33 | 9.24 | 1.07 |
| host and both cards, as of yesterday | 9.05 | - | 1.45 |
| host and both cards, now | 11.56 | 11.79 | 1.51 |

Prompt processing is 24 to 28 percent above the host alone, where
yesterday it was 3 percent below it: that is the activation stride and
the share that followed from it. Generation is 41 percent above the host
alone and will not move further on this model: the host is the long pole
in every multiply, it is the long pole because it holds half the
weights, and it holds half of them because two 6 GB cards cannot hold
more of 17.6 GB. Faster kernels do not help a card that is already
waiting. A model that fits on the cards is the shape where the kernels'
instruction count, not residency, is what bounds it.
