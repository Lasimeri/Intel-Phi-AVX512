# 2026-09-27: two threads per core

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up.
llama.cpp `build-native` at f5b9bd3 (unchanged). This repository at
c54d412 plus this record's changes, all in the card worker
(`card/vpu/vpu_worker.md` and `card/vpu/vpu_matmul.md`, "Two threads per
core"), the two defaults (`scripts/phi-vpu.sh` starts 114 threads, the
backend asks for 114), and `phi-vpu matmul-check --moe-shape M,K,0`.

## Why

The request was two threads on each of the cards' 57 cores: the
hardware threads of a core share its in-order issue slots, and one
thread cannot issue every cycle. The matrix service had been capped at
58 slices since 2026-09-23, when a second thread measured no better at a
prompt-sized batch.

## Where two threads lost

Lifting the cap and starting the workers at 114, the Q8_0 35B-A3B
offloaded (below) processed prompts faster and generated slower. The
pool trace (`phi-vpu-worker -t N`) put the loss in costs that grow with
the thread count: an empty dispatch 62.8 us at 114 against 42.5 at 57
(113 threads acknowledging on one counter, 114 job records rewritten a
dispatch), the second thread of core 55 on the hardware thread of a
block-device poller, and threads that had finished spinning on the issue
slots of the thread of their core still computing. The request-stage
trace then found a mixture's group table costing 12 to 27 us to write,
and the pull a dispatch of its own. The fixes are listed in the two
documents above.

## On the card

`phi-vpu matmul-check --moe --only q8_0 --act 1 --repeat 300 --threads T
--moe-shape 64,2048,1 --moe-shape 64,512,8 --moe-shape 2752,2048,0`, card
0, median host round trip in ms:

| request | before, 57 | before, 114 | after, 57 | after, 114 |
| --- | --- | --- | --- | --- |
| gate or up, 64 x 2048 | 0.262 | 0.284 | 0.217 | 0.220 |
| gate and up as one request | 0.307 | 0.368 | 0.279 | 0.261 |
| down, 64 x 512, eight rows | 0.186 | 0.250 | 0.110 | 0.113 |
| dense 2752 x 2048, one column | 0.335 | 0.360 | 0.277 | 0.232 |

An empty dispatch: 14.5 us at 57, 15.1 at 114.

## End to end

`llama-server`, Qwen3.8-35B-A3B Q8_0 (37.8 GB, more than this host's
memory, so offloaded from its mapped file), `--no-repack -t 12 -c 2048
-b 512 -ub 512 -np 1`, `PHI_GGML_OFFLOAD=1`, both cards' workers started
at T threads and `PHI_GGML_THREADS=T`; one warm-up request and three
timed (the 688-token prompt of the earlier records, 64 tokens greedy).
The first timed request of each server still pays for pages the host
reads from the NVMe (this model does not fit), so the steady two are
the comparison. The share: 2.1 GB of dense weights at a third a card,
34.2 GB of experts at 3.1 percent. In the order run:

| worker | T | pp (688) tok/s, steady | tg64 tok/s |
| --- | --- | --- | --- |
| before | 57 | 57.56, 57.93 | 7.87, 7.72, 7.93 |
| before | 114 | 61.60, 66.31 | 6.82, 7.38, 7.52 |
| after | 114 | 69.17, 70.74 | 7.92, 8.36, 8.34 |
| after | 57 | 66.17, 66.77 | 8.33, 8.06, 8.01 |
| after | 114 | 74.09, 74.81 | 8.21, 8.10, 8.41 |

- **The prompt: 57.8 to 72 tokens per second (+25 percent)**, and two
  threads per core ahead of one by 9 percent on the same worker.
- **Generation: 7.8 to 8.3 (+6 percent)**, one and two threads per core
  level: for this model it is bound by the host reading the experts it
  keeps (the cards hold 3.1 percent of them) from a page cache smaller
  than the model, not by the cards.
- Two further runs of this configuration with the pool trace on read
  5.7 to 7.4 at generation: the host's paging, not the trace, since the
  prompt rates of those runs were low too. Only interleaved runs are
  compared here.

## Verified

- `phi-vpu matmul-check` passes every case at 57 and 114 threads on
  card 0 (and at 114 on card 1), with 128-row cases added that reach the
  split of a core's rows and of a group's columns between its two
  threads.
- The generated text is the same byte for byte in all five runs above,
  before and after, at 57 and at 114 threads: every row is still
  computed whole by one thread in the same order.
- The seamless path shares the pool. `tools/avx512-seamless-test.c` under
  `scripts/phi512.sh` passes lane for lane at 65536, 1M and 16M elements on
  the 114-thread worker, and `scripts/phi512-ground.sh` agrees to the bit.
  It was slower on that pool (the 1M test 18 to 25 percent): its copies
  now take one slice per core, which leaves it 4 to 11 percent behind 57,
  and `scripts/phi512.sh` starts its worker at 57 (`card/vpu/vpu_worker.md`).
