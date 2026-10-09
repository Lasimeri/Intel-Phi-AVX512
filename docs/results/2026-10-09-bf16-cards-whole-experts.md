# 2026-10-09: the cards on BF16 Flash-Next, whole experts, and what caps decode

Qwen3.8-Flash-Next BF16 (48 blocks, 512 experts of 640 by 2560, 10 a
token) on the rack: one llama-server (`PHI_ENGINE=one`), blocks 0 to 8
whole and every block's attention on the four RTX 3080s, the experts of
blocks 9 to 47 (196.3 GB) in host memory, the four cards on generation
steps (`PHI_GGML_TG_ONLY=2`). Every rate is llama-server's
`predicted_per_second` of a 48-token greedy completion on three prompts
of this project's own questions (`bench/bench-engine.sh`); every card
figure is the worker's stats line (`vpu_proto.md`, offset 320) sampled
before and after.

## What was wrong, and the fix

The backend pairs a layer's gate and up as one request
(`VPU_K_MATMUL_MORE`) when both weights have "a card type of 2 or more"
(`glue.S id_pair`), meant as "quantized"; `MM_BF16` is 7. The card's
multi-matrix path wanted a quantized base (`matmul.S`, `quantized()`
before the pieces) and answered `VPU_E_REQUEST`: "the card rejected the
request (rows 0..640, columns 10)", a compute error, so whole-expert
placement never ran on BF16.

Now a float base takes pieces of its own type, run by
`rows_slice_multi_f` (each piece as the single-matrix float path takes
it; no groups), and `id_pair` pairs two quantized types or two equal
float types. `matmul-check --only bf16` covers the request on all four
cards (61 and 128 rows, mixtures, up to four matrices in one request).

## Measured on the cards

Per request during generation, whole experts, 114 threads (the stats
line over 4776 requests on four cards, no errors):

| stage | ms |
| --- | --- |
| pull (the card's DMA channel) | 0.012 |
| compute | 0.089 |
| push | 0.046 to 0.051 |
| total, doorbell to reply | 0.157 to 0.163 |

The host's wait for a card after its own rows: 0.003 ms a multiply
(the verbose ledger, 4345 multiplies). The cards never delay the host.

## Routing skew (the ids log, 3 prompts, 144 generated tokens)

| most used experts of a layer | share of generation selections | uniform routing would give |
| --- | --- | --- |
| 12.5% (64 of 512) | 83.9% | 26.0% |
| 25.0% | 98.2% | 44.7% |

A small sample from few prompts; the harness's own traffic over hours
is the sample that matters. The ranked placement from it
(`expert-placement.c rank`, 48 a layer; the backend takes 10 a card, 40
a layer) covers 75.5% of the same selections in-sample.

## The link rates that bound a cache's miss path

`bench/gpu-h2d-bw.c` (CUDA driver API, pinned host memory, 128 MiB
copies, 20 repeats; the GPUs are PCIe Gen4 x8 per nvidia-smi):

| | GB/s |
| --- | --- |
| one GPU, host to device | 13.4 |
| four GPUs at once, host to device | 53.7 |
| four GPUs at once, device to host | 52.8 |

The host's own read rate through llama.cpp's BF16 kernels is about 60
GB/s (15.3 tok/s with 3.83 GB of experts a token). So streaming misses
into VRAM over DMA does not beat the CPU computing them; what a cache
in VRAM or on the cards buys is the hit share, and only that.

## Generation rates

Three things spoiled the first measurements, each found by measuring:

- **A misplanned engine.** Restarts planned the GPUs while the stopped
  engine's memory still read as used, so the plan put every block's
  experts in host memory (`-ts 0/0/0/48`, 241.6 GB offered to the cards,
  8 whole experts a card). The "241.6 GB" once blamed on the backend was
  this. llama.phi's `phi-serve.sh` now waits on the GPUs' used memory
  (97549738c, f6c29ab3e).
- **The page cache.** The BF16 file (330 GB) is larger than RAM (251
  GB). After a restart, a decode can stall on experts read back from the
  NVMe: one 128-token decode took 48,000 major faults and 2.2 GB of reads
  (0.24 tok/s), and decodes run 7 to 8 tok/s for several more minutes
  before settling. `bench/bench-setups3.sh` warms each start until two
  decodes in a row take under 2000 major faults and agree within 10
  percent, and records every test's faults (`/proc/vmstat`).
- **The harness.** After every restart it re-read its whole context
  (40k then 84k tokens at 103 to 117 tok/s, about 13 minutes) and then
  generated in the other slot, its TUI pause notwithstanding (the engine
  log's slot timeline, `bench/phi-engine-setups3-last.log`). Every
  card run below shared its steps with the harness's own decode.

Prompts are this repository's README (7392 tokens) and
`card/vpu/vpu_matmul.md` (7683); decodes are three short questions,
128 greedy tokens each with `ignore_eos`; "beside" sends the second
prompt and a decode 3 s into it.

| configuration | prefill tok/s | decode tok/s | beside a prefill | faults in the decodes |
| --- | --- | --- | --- | --- |
| one engine, no cards, settled (repeat at 03:16) | 106.2 | 5.2, 14.2, 14.6 | decode 8.8, prefill 6.4 | 16,811 (most in the first) |
| one engine, cards, 12 host threads | 120.8 | 6.0, 3.1, 5.8 | decode 5.9, prefill 126.1 | 6,556 |
| one engine, cards, 32 host threads | 118.5 | 5.79 mean | decode 6.7, prefill 126.6 | 1,132 |
| two processes: GPUs prefill, CPU and cards decode | 15.4 | 2.4, 4.0, 2.1 | decode 3.6, prefill 13.9 | 1,503,040 (8.8 GB read) |

What holds:

- **The two-process form thrashes on BF16.** Its decode server sees no
  GPU, so every block's experts (241.6 GB) and the dense weights (8.7
  GB) are host memory beside the prefill server's: the prefill took 1.1
  million major faults (12.2 GB read) and the decodes 1.5 million. With
  this file on this host, the decode side cannot hold its weights in RAM;
  the Q8_0 file (176 GB) would fit.
- **The one engine without cards is the fastest measured form**,
  settled at 14.2 to 14.6 tok/s, prompts at 106 to 120 tok/s.
- **The card runs are confounded**: 5 to 6 tok/s at 12 and at 32 host
  threads, with the harness decoding beside them (the sum of the two
  sequences near 12 tok/s). They do not show what the cards cost alone.
  Thread count is not the limit: 32 threads gave what 12 did.

The clean comparison is the harness's own decode rate on the engine
log, with no benchmark traffic, cards against no cards over the same
context range.
