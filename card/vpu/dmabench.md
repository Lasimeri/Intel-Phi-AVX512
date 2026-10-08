# dmabench.S: phi-vpu-dmabench, the card-owned DMA channel against the window copy

```
scripts/phi-vpu.sh -c N dmabench [R]     # builds, pushes, runs on card N (no worker running)
phi-vpu-dmabench [R]                     # on the card, as root: R repetitions (1000; 1 to 4096)
```

A standalone measurement of `cdma.S` before any worker uses it. It maps
4 MiB of the host window at offset 1 GiB (above the worker's 768 MiB,
`WIN_BYTES`, so no byte the worker or the host's driver owns is touched),
opens channel 7 with its nonce line there, and for each size (64 bytes,
4, 5, 8, 16 and 64 KiB) runs R repetitions of three copies:

| copy | what it is |
| --- | --- |
| dma | `cdma_submit` then `cdma_wait`, into the landing area; the submit (descriptor line, fence, doorbell) timed apart from the whole |
| copy 1 thread | this thread's 64-byte vector loads from the uncached window, four loads then four stores (`phi_copy64`'s shape) |
| copy N threads | the worker's fused pull (`fused_setup` in `rows.S`): one thread a core on cores 0 to N-1, N = lines / 32 (at least 1, at most 56), each a contiguous share with no-read stores (`vmovnrngoaps`) and a locked count |

Each repetition first writes a new pattern into the window (different for
every repetition and size), reads the area's last line back (the posted
writes are in host memory before the copy starts), then **poisons the
destination with this core's stores**, so its lines are dirty in this
core's cache when the copy lands; after the copy every 8-byte word is
compared. Times are the time-stamp counter, converted with its rate
measured against the kernel's clock over 20 ms; each line prints the
median, the 99th percentile, the minimum and the maximum.

Then two checks that can fail:

- **cross-core**: R/4 repetitions at 4 and 64 KiB poisoned on core 56,
  copied by the engine, verified on core 10 and again on core 56: a line
  left dirty in core 56's cache would show on core 10 as poison.
- **control**: the descriptors written and the doorbell not rung
  (`cdma_nodoorbell`), a 50 ms timeout: the wait must return 0 with error
  9 (`CE_TIMEOUT`) and all 512 words of the 4 KiB destination differ from
  the source. Last, since a timeout closes the channel.

The main thread runs on CPU 0 (core 56's last hardware thread, the
worker's dispatcher's place); the pool on CPU 1 + 4i. Only one process may
own channel 7 (`cdma.md`): with a worker running that uses it, the bench
stops at `cdma_open` (error 5).

## Card 3, 2026-10-08, R = 1000

`scripts/phi-vpu.sh -c 3 dmabench 1000` equivalent (the record's
`bench.sh`), card 3's worker stopped, cards 0 to 2 serving. Microseconds,
median / 99th percentile:

| bytes | dma | of it the submit | copy 1 thread | copy N threads (N) |
| --- | --- | --- | --- | --- |
| 64 | 1.92 / 2.91 | 0.30 | 0.79 / 1.54 | 6.41 / 11.68 (1) |
| 4096 | 3.17 / 4.07 | 0.31 | 49.30 / 75.90 | 39.06 / 58.70 (2) |
| 5120 | 3.49 / 4.32 | 0.31 | 61.89 / 87.22 | 46.50 / 67.77 (2) |
| 8192 | 4.30 / 5.01 | 0.32 | 99.60 / 128.61 | 39.12 / 59.87 (4) |
| 16384 | 6.64 / 18.20 | 0.36 | 199.32 / 232.60 | 39.16 / 60.94 (8) |
| 65536 | 20.31 / 33.44 | 0.49 | 827.06 / 964.42 | 41.82 / 70.70 (32) |

Words wrong: 0 for every copy and size (6000 repetitions, 6501 copies by
the engine, the 256-entry ring wrapped about 100 times); cross-core 0 and
0 at both sizes; the control timed out with error 9 and all 512 words
untouched. The one-line copy by one thread (0.79 us) is the price of one
uncached load's round trip, and the split copy's 64-byte row is mostly
the bench pool's wake. The engine costs about 2 us fixed plus 3.2 GB/s;
the split copy is flat at about 39 to 46 us because every copier reads
its 32 lines one round trip after another.
