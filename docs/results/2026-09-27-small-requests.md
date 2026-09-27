# 2026-09-27: one token's multiplies on the cards, 0.95 ms to 0.33

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up.
llama.cpp `build-native` at f5b9bd3 (unchanged). This repository at
9fef58e plus this record's changes, all in the card worker
(`card/vpu/vpu_worker.c`, `card/vpu/vpu_matmul.c`) and one diagnostic
in `phi-vpu` (`matmul-check --moe`); the backend library is unchanged.

## Why

The offloaded mode (`PHI_GGML_OFFLOAD=1`) is how Intel-Phi-Jev runs its
subject, Qwen3.8-35B-A3B Q4_K_M, with `--no-repack`: the cards hold 41
percent of its 20.9 GB of weights and the host never reads them. It
generated at 4.84 tokens per second against 7.74 for the host alone
(`2026-09-26-repacking-and-determinism.md`). One offloaded llama-server
request with `PHI_GGML_VERBOSE=1`, its 64 generated tokens totalled per
token (the ledger of the backend's own lines):

| per token, 209 ms | ms |
| --- | --- |
| 118 multiplies with the cards: the host's part, then its wait | 21.4 + 92.7 |
| of which the slower card's compute | 92.6 |
| 385 multiplies on the host alone (the vocabulary matrix 22.7 of it) | 73.0 |
| everything else llama.cpp does | about 22 |

A card multiply took 0.95 ms on average, most of it compute, for a few
rows of eight experts: tens of microseconds of arithmetic.

## Taking it apart

Two instruments, both kept:

- `phi-vpu-worker -t N` (`card/vpu/vpu_worker.md`, "What a dispatch
  costs") stamps each pool dispatch with the time stamp counter. Across
  the generation phase of the same request: 610 to 680 us per multiply,
  of which 6 us for the last thread to see the work, 13 us for the
  dispatcher to see it done, and the rest the slices, every one of them
  slow. An empty dispatch costs 29 us.
- `phi-vpu matmul-check --moe` (`host/crates/phi-vpu/src/matmul.md`)
  reproduces the request with the weights as cold as between two visits
  of a layer. As it was, on card 0: gate/up 1.13 ms of compute
  (median, 300 requests), 1.31 ms round trip; down 0.47 and 0.66. It
  reproduces the ledger's figure.

Then, one at a time (`card/vpu/vpu_matmul.md`, "One token's multiply,
and whose lines the pool reads", has the table):

1. **Cold lines.** Each thread now asks for its whole working set before
   computing, when it is small (`warm_slice`, `vprefetch1`, 192 KiB at
   most). Down: 0.47 to 0.13 ms. Gate/up: 1.13 to 1.00.
2. **Who wrote the activations.** On one thread gate/up's shared
   activation row costs what eight separate rows cost (2.9 ms both); on
   57 threads it took 1.0 ms against 0.18. A row pulled by one thread
   (k 1024) was fast; this one (4.4 KB) was pulled by the whole pool, a
   line or two per core, and every core then read all of it. Pulled by
   one thread it took 0.14 ms. Pooled pulls now give each thread at
   least 32 lines (`PULL_LINES`, swept at 16, 32, 64 and 128).
3. **The log.** `phi-vpu.sh` starts workers with `-v`, which wrote two
   lines per request into a file on the card's host-backed disk, one of
   them before the reply: 35 to 60 us a request. The matrix service's
   lines now need `-v -v`; the command line is unchanged.
4. **Not false sharing**: results written to lines of each thread's own
   changed 1.0 ms to 0.94.

After all three, on card 0: gate/up 0.14 ms compute and 0.28 ms round
trip, down 0.12 and 0.27. Card 1 (Gen2 x4 through the chipset): 0.17 and
0.33, 0.12 and 0.30.

## End to end

`llama-server`, the 35B-A3B Q4_K_M, `-t 12 -c 2048 -b 512 -ub 512 -np 1`,
one warm-up request and three timed ones: the same 688-token prompt,
`cache_prompt: false`, 64 tokens greedy (`temperature 0`, `seed 1`).
The rates are llama.cpp's own. The worker binary was swapped on both
cards between servers; the backend library, the model and the command
were the same. In the order run:

| worker | configuration | pp (688) tok/s | tg64 tok/s |
| --- | --- | --- | --- |
| before | offloaded, `--no-repack` | 63.83, 63.77, 63.28 | 4.78, 4.74, 4.72 |
| after | offloaded, `--no-repack` | 63.60, 63.45, 63.49 | **7.49, 7.06, 7.48** |
| after | offloaded, `--no-repack` | 55.34, 54.94, 54.97 | **7.42, 7.31, 7.43** |
| before | offloaded, `--no-repack` | 54.61, 54.71, 54.43 | 4.76, 4.80, 4.70 |
| after | default split, repacked | 72.55, 73.72, 73.77 | 8.99, 8.84, 8.96 |
| before | default split, repacked | 83.04, 85.05, 85.98 | 8.81, 8.76, 8.77 |
| (none) | host alone, repacked | 82.68, 82.28, 82.69 | 7.56, 7.77, 7.77 |
| after | default split, repacked | 79.46, 81.55, 81.86 | 8.82, 8.62, 8.59 |
| before | default split, repacked | 83.89, 85.28, 84.58 | 8.87, 8.44, 8.72 |
| after | default split, repacked | 83.20, 85.01, 85.56 | 8.98, 8.90, 8.97 |
| after (as committed) | offloaded, `--no-repack` | 63.27, 63.12, 62.80 | **7.42, 7.31, 7.39** |

- **Offloaded: 4.75 to 7.4 tokens per second at generation (+55
  percent)**, now within 5 percent of the host alone (7.74 without
  repacking, 7.70 with, above) while the cards hold 41 percent of the
  weights. The prompt is unchanged: the prompt's multiplies are large,
  and neither the warm-up (over 192 KiB) nor the pull rule (a batch's
  rows are hundreds of lines a thread) changes them. Its two levels,
  63 and 55, are the host's drift between the two rounds, and both
  workers read the same within a round.
- **The default split is unchanged within its spread** (generation 8.93,
  8.68 and 8.95 against 8.78 and 8.68): its judgement already keeps the
  small multiplies with the host. The first prompt figure after the
  change (73) was not repeated (81 and 85 after it).

The last row is the binary as committed, which differs from the one
measured above it only in the pull rule being a constant rather than a
variable of the same value.

## Verified

- `phi-vpu matmul-check` (every format, mixtures, the feed-forward) on
  both cards with the committed worker: all pass.
- The generated text is the same byte for byte with the old worker and
  the new (the same request, compared with `cmp`), and within each
  server every request gave the same text.
- The cards did the work: in the profiled request, 118 multiplies a
  token went to both cards and the host read none of their rows ("the
  cards' rows read by the host so far 0.0 MB").

## What a token costs now

The same profile with the new worker (7.24 tokens per second with the
verbose lines on):

| per token, 138 ms | ms |
| --- | --- |
| 118 multiplies with the cards: the host's part, then its wait | 21.7 + 19.1 |
| 385 multiplies on the host alone | 73.6 |
| everything else llama.cpp does | about 24 |

The cards' requests are no longer the long pole. What is: the host's
multiplies of tensors that have no rows on the cards. The vocabulary
matrix (22.8 ms a token) because the budget is spent before the model's
end, and mid-sized dense matrices (8192 x 2048, 2048 x 4096, 4096 x
2048: 34 ms together) because their card part falls under
`PHI_GGML_MIN_BYTES`. One share for every matrix spends most of the
budget on experts, of which a token reads 8 in 256, and a gate or up
expert's 64-row rounding makes its share 25 percent where 20.5 was
sized. Next: a share per class, the dense matrices first, sized with the
rounding counted.
