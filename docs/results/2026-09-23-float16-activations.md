# 2026-09-23: half the link, none of the arithmetic

The card's bandwidth is what it has least of and its vector units are
what it has most of, so the way to spend one on the other is to make
every byte that crosses the link carry more work. The first of those is
free: the activation rows go over as float16 and the card's memory
operands up-convert them for nothing, because `{float16}` is a field of
the MVEX operand and not an instruction
(`knc-mvex/src/conv.md`). Same kernel, same instruction count, half the
bytes.

This note also corrects a measurement rule that was costing 5x, found
while benchmarking the above.

## What changed

`kernelgen` emits every quantized kernel twice, `phi_q4k_8` and
`phi_q4k_8h`, differing in the product's memory operand and in the 32
byte rather than 64 byte distance between an activation row's vectors
(`host/crates/phi-vpu/src/bin/kernelgen/quant.md`). The descriptor gained
`b_type` (0 float32, 1 float16) and the card indexes a `k[2][3]` table
with it (`card/vpu/vpu_matmul.md`). The backend converts with F16C
before the copy into the window and sets a row stride of `k * 2 + B_PAD`
(`host/crates/phi-ggml/src/lib.md`). Float weight types are refused by
the card and sent float32 by the host, because their kernels are the
`phi_dot4_*` family and are not generated.

Every format still matches the host's own dot products:
`matmul-check --act 1` passes f32, f16, Q4_K, Q5_K, Q6_K, Q8_0 and
IQ4_XS at n 1, 4, 8 and 13 and as mixtures of eight experts.

## On the card

`phi-vpu -c 0 matmul-check --threads 57 --pad 256 --repeat 4`, 4096 x
5120, the card's compute time only, GFLOP/s, best of 4:

| format | n 1 f32 | n 1 f16 | n 8 f32 | n 8 f16 | n 64 f32 | n 64 f16 |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | 75.1 | 76.7 | 240.1 | **277.9** | 239.8 | **286.2** |
| Q5_K | 53.1 | 53.2 | 211.5 | **238.9** | 213.8 | **248.1** |
| Q6_K | 54.4 | 55.8 | 197.2 | **231.6** | 207.8 | **242.2** |
| Q8_0 | 72.2 | 70.7 | 234.5 | **262.7** | 239.2 | **279.9** |
| IQ4_XS | 60.6 | 68.0 | 191.5 | **252.9** | 230.1 | **267.7** |

At one activation row there is nothing to win, which is the point worth
keeping: a single row is one vector per superblock and the kernel is
against the weight bytes, so halving the row halves something that was
not the constraint. From eight rows up it is 12 to 19 percent, and that
is the L2 traffic the smaller rows do not cause, not the link: the
kernel is timed with the rows already in card memory.

## End to end

Interleaved, `-lm none`, two rounds of each configuration, the host
alone on 16 hardware threads and the split on 12 (below):

| Qwen3.8-27B UD-Q4_K_XL | pp512 | tg32 |
| --- | --- | --- |
| host alone | 8.97 | 0.98 |
| both cards, float32 activations | 12.05, 12.02 | 1.45, 1.47 |
| both cards, float16 activations | **12.40, 12.35** | **1.61, 1.54** |

float16 against float32 is +2.8 percent at the prompt and +7.9 at
generation, and the split against the host alone is +38 and +61.

| Qwen3.8-35B-A3B Q4_K_M | pp512 | tg32 |
| --- | --- | --- |
| host alone | 90.26 | 6.62 |
| both cards, float32 activations | 83.51, 84.70 | 8.53, 8.38 |
| both cards, float16 activations | 85.01, 86.04 | 7.91, 8.47 |

On the mixture model the two are the same within the spread. That is
consistent with the card measurement rather than against it: this
model's multiplies that reach a card at generation are small, and the
backend's judgement leaves most of the mixture with the host anyway
(`docs/results/2026-09-23-mixture-of-experts.md`), so there are few
eight-row-or-more calls for the halving to help.

## The 5x that was in the method, not the code

The first run of the table above reported generation at 0.29 tokens per
second against the host's 0.98, where this repository had recorded 1.63
against 1.13 the day before. It was not a regression. `llama-bench` had
been given `-t 16` in both configurations, and with the cards involved
that is oversubscription:

| threads given to llama.cpp, both cards | 27B tg8 |
| --- | --- |
| 12 | **1.53** |
| 14 | 1.01 |
| 16 | 0.29 |

The host alone at 16 gives 1.00. Only the matrix multiplies run on the
backend's own 12-thread pool; everything else a model does stays with
the program's threads, and those contend with the two card daemons
serving DMA. A token is 419 multiplies, each a barrier, and a barrier
whose last thread has been descheduled costs a timeslice. `lib.md`
already carried the rule for the backend's own pool (7 ms per multiply
at 15 threads); what was missing is that the calling program needs the
same restraint, and that the penalty is a factor, not a percentage.

Verbose accounting of the bad run, for the record: the multiplies of one
token summed to 720 ms of a 3400 ms token, so 2.7 s of it was the
program's own threads fighting the daemons outside the backend
entirely.

`ggml_backend_init` now prints the reminder next to the thread count it
sets for itself.

## What this does not do

It does not reduce the return path. The card still writes float32
results, and at one token those are the larger direction: the verbose
per-multiply figures on the 27B are pull 0.25 ms against push 0.4. The
store is one float per (row, column), scattered by column for a mixture,
so a float16 result is not the free operand change the load side was and
would need the result loop rebuilt around whole vectors first. Measured
per tensor before it is worth building: the return beats the send only
when `share * m > k`, which the feed-forward's first two matrices
satisfy and its third cannot.
